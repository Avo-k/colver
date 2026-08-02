//! Where IS-DD's determinized worlds come from.
//!
//! A [`WorldSource`] answers one question: *"given everything I have publicly
//! observed this deal, hand me `n` plausible assignments of the hidden cards."*
//! It has the same lifecycle as a player — [`init_deal`](WorldSource::init_deal)
//! then [`observe`](WorldSource::observe) for every action — because the good
//! samplers are conditional on the whole history, not just the current position.
//!
//! Three implementations:
//!
//! | source | how | quality (auction argmax / play argmax) |
//! |--------|-----|----------------------------------------|
//! | [`SidecarWorldSource`] | playgen transformer on a remote GPU, batched | 70% / 88% |
//! | [`LocalPlaygenSource`] | the same transformer, in-process on CPU | same distribution, ~50× slower |
//! | [`UniformWorldSource`] | constraint-uniform determinization | 15% / 71% |
//!
//! (Percentages are `bench_world_cred` on the production playgen model: how
//! often the reference policy would replay the observed hidden action given
//! the sampled world. Uniform worlds are cheap and legal but implausible.)
//!
//! # Why this is a trait and not a flag
//!
//! World generation used to live *outside* IS-DD — the web server sampled
//! worlds from the GPU sidecar and pushed them in, while the arena did not.
//! The two therefore measured different agents. Making the source a component
//! that the agent **owns** means every caller that builds an IS-DD agent gets
//! the same worlds by construction.
//!
//! # Failure
//!
//! A configured source that fails returns [`AgentError::WorldSource`] rather
//! than silently falling back to uniform worlds: quietly swapping the sampler
//! changes playing strength by several points per deal and would corrupt every
//! measurement taken while the GPU was down. Callers that prefer a weaker
//! answer to no answer wrap the source in [`FallbackPolicy::Uniform`].

use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::Arc;
use std::time::Duration;

use rand::RngCore;

use crate::agent::AgentError;
use crate::card::card_count;
use crate::determinize::determinize_greedy;
use crate::playgen::infer::{PlaygenModel, PlaygenSampler};
use crate::state::GameState;

/// Hidden-card assignment for all four seats, as `CardSet` bitmasks of the
/// cards **still in hand** at the position it was sampled for.
pub type World = [u32; 4];

/// A stateful sampler of determinized worlds.
pub trait WorldSource: Send {
    /// Short tag for logs and stats, e.g. `"sidecar"`, `"uniform"`.
    fn name(&self) -> &'static str;

    /// Start a new deal. `state` is the pre-auction position; only
    /// `observer`'s own hand may be read from it.
    fn init_deal(&mut self, state: &GameState, observer: u8);

    /// Observe an action by any seat. `state_before` is the position before it.
    fn observe(&mut self, state_before: &GameState, player: u8, action: u8);

    /// Sample up to `n` worlds for the current position. Returning fewer than
    /// `n` (including zero) is allowed and means "that is all I can produce
    /// here" — over-constrained endgames do this legitimately.
    fn worlds(
        &mut self,
        state: &GameState,
        observer: u8,
        n: usize,
        rng: &mut dyn RngCore,
    ) -> Result<Vec<World>, AgentError>;
}

/// Keep only worlds that are consistent with what the observer knows: the
/// right number of cards per seat, and the observer's own hand untouched.
/// A sampler that returns garbage should lose those worlds, not poison the
/// aggregation with impossible positions.
pub fn retain_valid(worlds: Vec<World>, state: &GameState, observer: u8) -> Vec<World> {
    worlds
        .into_iter()
        .filter(|hands| {
            hands[observer as usize] == state.hands[observer as usize]
                && (0..4).all(|p| card_count(hands[p]) == card_count(state.hands[p]))
        })
        .collect()
}

// ══════════════════════════════════════════════════════════════════════
//  Uniform
// ══════════════════════════════════════════════════════════════════════

/// Constraint-uniform determinization: every assignment consistent with the
/// hard facts (voids, trump ceiling, cards already played) is equally likely.
///
/// This is the coverage floor. It needs no model and never fails, but it
/// ignores everything the auction and the play revealed about *how likely*
/// each assignment is, which is most of the available information.
#[derive(Default)]
pub struct UniformWorldSource;

impl WorldSource for UniformWorldSource {
    fn name(&self) -> &'static str {
        "uniform"
    }

    fn init_deal(&mut self, _state: &GameState, _observer: u8) {}

    fn observe(&mut self, _state_before: &GameState, _player: u8, _action: u8) {}

    fn worlds(
        &mut self,
        state: &GameState,
        observer: u8,
        n: usize,
        rng: &mut dyn RngCore,
    ) -> Result<Vec<World>, AgentError> {
        let mut rng = RngAdapter(rng);
        Ok((0..n)
            .filter_map(|_| determinize_greedy(state, observer, &mut rng).map(|s| s.hands))
            .collect())
    }
}

// ══════════════════════════════════════════════════════════════════════
//  Local playgen (CPU, in-process)
// ══════════════════════════════════════════════════════════════════════

/// The playgen transformer running in-process on CPU.
///
/// Same model and same distribution as [`SidecarWorldSource`] (the forward
/// pass is validated bit-for-bit against the GPU one), but ~50× slower: it is
/// the right choice for offline tools and tests, not for a per-move budget.
pub struct LocalPlaygenSource {
    sampler: PlaygenSampler,
    temperature: f32,
}

impl LocalPlaygenSource {
    pub fn new(model: Arc<PlaygenModel>, temperature: f32) -> Self {
        LocalPlaygenSource { sampler: PlaygenSampler::new(model), temperature }
    }

    /// The underlying sampler, for analysis surfaces that need more than
    /// worlds (card marginals, bid policy, mid-auction deals).
    pub fn sampler_mut(&mut self) -> &mut PlaygenSampler {
        &mut self.sampler
    }
}

impl WorldSource for LocalPlaygenSource {
    fn name(&self) -> &'static str {
        "playgen_local"
    }

    fn init_deal(&mut self, state: &GameState, observer: u8) {
        self.sampler.init_deal(state, observer);
    }

    fn observe(&mut self, state_before: &GameState, player: u8, action: u8) {
        self.sampler.record_action(state_before, player, action);
    }

    fn worlds(
        &mut self,
        state: &GameState,
        observer: u8,
        n: usize,
        rng: &mut dyn RngCore,
    ) -> Result<Vec<World>, AgentError> {
        // The transformer streams its weights once per token step for the
        // whole batch, so sampling is done in lockstep chunks.
        const BATCH: usize = 16;
        let mut out = Vec::with_capacity(n);
        let mut rng = RngAdapter(rng);
        while out.len() < n {
            let want = BATCH.min(n - out.len());
            let batch = self.sampler.generate_worlds_batch(state, want, self.temperature, &mut rng);
            if batch.is_empty() {
                break; // dead end or sequence too long: stop asking
            }
            out.extend(batch);
        }
        Ok(retain_valid(out, state, observer))
    }
}

/// Bridges the object-safe `&mut dyn RngCore` of [`WorldSource`] to the
/// `impl Rng` generics used throughout the sampler code.
struct RngAdapter<'a>(&'a mut dyn RngCore);

impl rand::RngCore for RngAdapter<'_> {
    fn next_u32(&mut self) -> u32 {
        self.0.next_u32()
    }
    fn next_u64(&mut self) -> u64 {
        self.0.next_u64()
    }
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.0.fill_bytes(dest)
    }
    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        self.0.try_fill_bytes(dest)
    }
}

// ══════════════════════════════════════════════════════════════════════
//  GPU sidecar
// ══════════════════════════════════════════════════════════════════════

/// The playgen transformer running on a remote GPU (`playgen_gpu_server`).
///
/// The request carries the *replayable deal* — dealer, initial hands, the
/// action prefix, the observer — and the server rebuilds the sampler state by
/// replay before sampling a batch. That keeps this client stateless apart from
/// the history it accumulates, and means a restarted sidecar needs no session.
///
/// Sampling ~256 worlds costs one round trip (~200 ms on a 4090), against
/// several seconds on CPU, which is what makes 100%-playgen worlds affordable
/// inside a per-move budget.
pub struct SidecarWorldSource {
    /// Base URL, e.g. `http://gpu-host:8003`, without a trailing slash.
    url: String,
    timeout: Duration,
    temperature: f32,
    dealer: u8,
    initial_hands: World,
    history: Vec<(u8, u8)>,
}

impl SidecarWorldSource {
    pub fn new(url: impl Into<String>, temperature: f32, timeout: Duration) -> Self {
        let url = url.into().trim_end_matches('/').to_string();
        SidecarWorldSource {
            url,
            timeout,
            temperature,
            dealer: 0,
            initial_hands: [0; 4],
            history: Vec::new(),
        }
    }

    /// Check that the sidecar is up. Worth calling once at agent construction
    /// so a misconfigured URL fails at startup rather than mid-deal.
    pub fn health_check(&self) -> Result<String, AgentError> {
        http_request(&self.url, "GET", "/health", None, self.timeout)
    }

    fn request_body(&self, observer: u8, n: usize) -> String {
        let mut s = String::with_capacity(64 + self.history.len() * 8);
        s.push_str(&format!(
            "{{\"dealer\":{},\"hands\":[{},{},{},{}],\"actions\":[",
            self.dealer,
            self.initial_hands[0],
            self.initial_hands[1],
            self.initial_hands[2],
            self.initial_hands[3],
        ));
        for (i, (p, a)) in self.history.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&format!("[{p},{a}]"));
        }
        s.push_str(&format!(
            "],\"observer\":{},\"n_worlds\":{},\"temperature\":{}}}",
            observer, n, self.temperature
        ));
        s
    }
}

impl WorldSource for SidecarWorldSource {
    fn name(&self) -> &'static str {
        "playgen_sidecar"
    }

    fn init_deal(&mut self, state: &GameState, _observer: u8) {
        self.dealer = state.dealer;
        self.initial_hands = state.hands;
        self.history.clear();
    }

    fn observe(&mut self, _state_before: &GameState, player: u8, action: u8) {
        self.history.push((player, action));
    }

    fn worlds(
        &mut self,
        state: &GameState,
        observer: u8,
        n: usize,
        _rng: &mut dyn RngCore,
    ) -> Result<Vec<World>, AgentError> {
        if n == 0 {
            return Ok(Vec::new());
        }
        let body = self.request_body(observer, n);
        let resp = http_request(&self.url, "POST", "/play_worlds", Some(&body), self.timeout)?;
        let worlds = parse_hands(&resp).ok_or_else(|| {
            AgentError::WorldSource(format!(
                "{}: malformed /play_worlds response ({} bytes)",
                self.url,
                resp.len()
            ))
        })?;
        Ok(retain_valid(worlds, state, observer))
    }
}

// ══════════════════════════════════════════════════════════════════════
//  Fallback policy
// ══════════════════════════════════════════════════════════════════════

/// What to do when the primary source fails or comes up short.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FallbackPolicy {
    /// Propagate the error. The default, and the only honest setting for
    /// anything whose numbers will be compared: an agent that silently
    /// downgrades its world sampler is a different agent.
    #[default]
    Strict,
    /// Top up with constraint-uniform worlds and keep playing. For interactive
    /// use where finishing the deal matters more than exact strength; the
    /// substitution is still visible in `Stats::worlds`.
    Uniform,
}

/// Wraps a source with a fallback policy, so the choice lives in one place
/// instead of at every call site.
pub struct PolicyWorldSource {
    inner: Box<dyn WorldSource>,
    policy: FallbackPolicy,
    uniform: UniformWorldSource,
}

impl PolicyWorldSource {
    pub fn new(inner: Box<dyn WorldSource>, policy: FallbackPolicy) -> Self {
        PolicyWorldSource { inner, policy, uniform: UniformWorldSource }
    }
}

impl WorldSource for PolicyWorldSource {
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn init_deal(&mut self, state: &GameState, observer: u8) {
        self.inner.init_deal(state, observer);
    }

    fn observe(&mut self, state_before: &GameState, player: u8, action: u8) {
        self.inner.observe(state_before, player, action);
    }

    fn worlds(
        &mut self,
        state: &GameState,
        observer: u8,
        n: usize,
        rng: &mut dyn RngCore,
    ) -> Result<Vec<World>, AgentError> {
        match self.inner.worlds(state, observer, n, rng) {
            Ok(w) => Ok(w),
            Err(e) => match self.policy {
                FallbackPolicy::Strict => Err(e),
                    FallbackPolicy::Uniform => self.uniform.worlds(state, observer, n, rng),
            },
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
//  Minimal HTTP + JSON
// ══════════════════════════════════════════════════════════════════════
//
// `colver-core` deliberately has no HTTP or JSON dependency: the sidecar
// protocol is a handful of integers, and pulling in an async stack for it
// would leak into every binary that links the crate. These two helpers are
// the whole client.

fn split_host(url: &str) -> Result<String, AgentError> {
    let rest = url
        .strip_prefix("http://")
        .ok_or_else(|| AgentError::WorldSource(format!("unsupported URL scheme: {url}")))?;
    let host = rest.split('/').next().unwrap_or(rest);
    if host.is_empty() {
        return Err(AgentError::WorldSource(format!("no host in URL: {url}")));
    }
    Ok(if host.contains(':') { host.to_string() } else { format!("{host}:80") })
}

/// One blocking request/response. Returns the response body on 2xx.
fn http_request(
    url: &str,
    method: &str,
    path: &str,
    body: Option<&str>,
    timeout: Duration,
) -> Result<String, AgentError> {
    let host = split_host(url)?;
    let fail = |ctx: &str, e: std::io::Error| {
        AgentError::WorldSource(format!("{host}{path}: {ctx}: {e}"))
    };

    let mut stream = TcpStream::connect(&host).map_err(|e| fail("connect", e))?;
    stream.set_read_timeout(Some(timeout)).map_err(|e| fail("set timeout", e))?;
    stream.set_write_timeout(Some(timeout)).map_err(|e| fail("set timeout", e))?;

    let mut req = format!("{method} {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n");
    if let Some(b) = body {
        req.push_str("Content-Type: application/json\r\n");
        req.push_str(&format!("Content-Length: {}\r\n", b.len()));
    }
    req.push_str("\r\n");
    if let Some(b) = body {
        req.push_str(b);
    }
    stream.write_all(req.as_bytes()).map_err(|e| fail("write", e))?;

    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).map_err(|e| fail("read", e))?;
    let text = String::from_utf8_lossy(&raw);

    let (head, payload) = text
        .split_once("\r\n\r\n")
        .ok_or_else(|| AgentError::WorldSource(format!("{host}{path}: truncated response")))?;
    let status = head
        .split_whitespace()
        .nth(1)
        .and_then(|c| c.parse::<u16>().ok())
        .ok_or_else(|| AgentError::WorldSource(format!("{host}{path}: no status line")))?;
    if !(200..300).contains(&status) {
        return Err(AgentError::WorldSource(format!("{host}{path}: HTTP {status}")));
    }
    Ok(payload.to_string())
}

/// Pull `{"hands": [[a,b,c,d], ...]}` out of a sidecar response.
///
/// A hand-rolled scan rather than a JSON parser: the payload is a single array
/// of unsigned integers under a known key, and both ends of this protocol live
/// in this repository. Returns `None` if the key is absent or the integer count
/// is not a multiple of four.
fn parse_hands(json: &str) -> Option<Vec<World>> {
    let start = json.find("\"hands\"")? + "\"hands\"".len();
    let rest = &json[start..];
    let open = rest.find('[')?;
    let rest = &rest[open..];

    // Find the end of the outer array by depth, so trailing fields are ignored.
    let mut depth = 0i32;
    let mut end = None;
    for (i, c) in rest.char_indices() {
        match c {
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    end = Some(i);
                    break;
                }
            }
            _ => {}
        }
    }
    let body = &rest[..=end?];

    let mut nums: Vec<u32> = Vec::new();
    let mut cur: Option<u64> = None;
    for c in body.chars() {
        if let Some(d) = c.to_digit(10) {
            cur = Some(cur.unwrap_or(0) * 10 + d as u64);
        } else if let Some(v) = cur.take() {
            nums.push(u32::try_from(v).ok()?);
        }
    }
    if let Some(v) = cur {
        nums.push(u32::try_from(v).ok()?);
    }

    // An **empty** `hands` array is a valid answer, not a parse failure: the
    // `WorldSource` contract says a source that runs dry — as it legitimately
    // does in an over-constrained endgame — returns an empty batch rather than
    // erroring. Folding that into `None` turned `{"hands":[]}` into
    // "malformed response", which under the default `Strict` fallback is a hard
    // error mid-deal. Measured at 8% of decisions when labelling with IS-DD.
    // Only a non-empty list that does not group into 4s is actually malformed.
    if nums.len() % 4 != 0 {
        return None;
    }
    Some(nums.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A dry source answers `{"hands":[]}`. That is "no worlds here", which the
    /// search handles, and not a malformed response, which it treats as a hard
    /// error. Conflating the two failed ~8% of decisions in an endgame-heavy
    /// workload.
    #[test]
    fn empty_hands_is_dry_not_malformed() {
        assert_eq!(parse_hands(r#"{"hands":[]}"#), Some(Vec::new()));
        // Still malformed when the numbers do not group into seats.
        assert_eq!(parse_hands(r#"{"hands":[[1,2,3]]}"#), None);
        assert_eq!(parse_hands(r#"{"worlds":3}"#), None);
    }

    #[test]
    fn parses_sidecar_hands() {
        let json = r#"{"hands":[[1,2,3,4],[4294967295,0,7,8]]}"#;
        let got = parse_hands(json).unwrap();
        assert_eq!(got, vec![[1, 2, 3, 4], [4294967295, 0, 7, 8]]);
    }

    #[test]
    fn rejects_ragged_hands() {
        assert!(parse_hands(r#"{"hands":[[1,2,3]]}"#).is_none());
        assert!(parse_hands(r#"{"worlds":[]}"#).is_none());
    }

    #[test]
    fn ignores_fields_after_hands() {
        let json = r#"{"hands":[[1,2,3,4]],"worlds":9}"#;
        assert_eq!(parse_hands(json).unwrap(), vec![[1, 2, 3, 4]]);
    }

    #[test]
    fn host_gets_default_port() {
        assert_eq!(split_host("http://example.com").unwrap(), "example.com:80");
        assert_eq!(split_host("http://example.com:8003").unwrap(), "example.com:8003");
        assert!(split_host("https://example.com").is_err());
    }
}
