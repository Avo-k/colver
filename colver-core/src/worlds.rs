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
/// right number of cards per seat, the observer's own hand untouched, and the
/// belote announcement honoured. A sampler that returns garbage should lose
/// those worlds, not poison the aggregation with impossible positions.
///
/// The belote clause is here rather than only in the determinizers because
/// playgen cannot deduce it: the announcement is not part of the token stream
/// (see `docs/belief/playgen.md`), so the model only ever sees the King or Queen
/// of trump being played, never the fact that the seat announced while doing so.
/// Measured rejection rate: **15.4%** of its worlds at the positions concerned,
/// against 40.1% for a blind uniform draw.
///
/// Known bounded edge: `run_search` reads an empty batch as "the source has run
/// dry" and stops asking, so a *small* request whose worlds are all rejected
/// costs the rest of that decision's worlds (they come from the local uniform
/// fallback instead). Only reachable on the last one or two worlds of a count-mode
/// budget — under a deadline the batch is `world_batch` (128) and cannot be
/// emptied by a 15% filter.
pub fn retain_valid(worlds: Vec<World>, state: &GameState, observer: u8) -> Vec<World> {
    let facts = crate::play::belote_facts(state);
    worlds
        .into_iter()
        .filter(|hands| {
            hands[observer as usize] == state.hands[observer as usize]
                && (0..4).all(|p| card_count(hands[p]) == card_count(state.hands[p]))
                && facts.allows(hands)
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
///
/// # Plusieurs sidecars
///
/// L'URL peut être une **liste séparée par des virgules**. Les requêtes sont
/// alors réparties en tourniquet sur un compteur *global* au processus, et non
/// par instance : une génération de masse construit un `IsDdPlayer` par siège
/// et par thread, donc des compteurs indépendants partant tous de zéro
/// enverraient la première requête de chacun au même GPU.
///
/// Le tourniquet est volontairement aveugle à la charge. Un vrai équilibrage
/// (au moins chargé) suppose de savoir ce que chaque sidecar a en file, ce que
/// ce client ne voit pas ; et pour des GPU de débits différents c'est la
/// *proportion* qu'il faudrait régler, pas l'ordre. Répéter une URL dans la
/// liste fait exactement ça : `a,a,b` envoie deux tiers du trafic à `a`.
pub struct SidecarWorldSource {
    /// Base URLs, e.g. `http://gpu-host:8003`, without a trailing slash.
    urls: Vec<String>,
    timeout: Duration,
    temperature: f32,
    dealer: u8,
    initial_hands: World,
    history: Vec<(u8, u8)>,
}

/// Tourniquet partagé par tous les `SidecarWorldSource` du processus.
static RR: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

impl SidecarWorldSource {
    pub fn new(url: impl Into<String>, temperature: f32, timeout: Duration) -> Self {
        let raw = url.into();
        let urls: Vec<String> = raw
            .split(',')
            .map(|u| u.trim().trim_end_matches('/').to_string())
            .filter(|u| !u.is_empty())
            .collect();
        SidecarWorldSource {
            urls: if urls.is_empty() { vec![raw] } else { urls },
            timeout,
            temperature,
            dealer: 0,
            initial_hands: [0; 4],
            history: Vec::new(),
        }
    }

    /// The sidecar this request goes to.
    fn pick(&self) -> &str {
        if self.urls.len() == 1 {
            return &self.urls[0];
        }
        let i = RR.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        &self.urls[i % self.urls.len()]
    }

    /// Check that **every** configured sidecar is up. Worth calling once at
    /// agent construction so a misconfigured URL fails at startup rather than
    /// mid-deal — and with a list, so that a run does not silently send a third
    /// of its worlds into a hole.
    pub fn health_check(&self) -> Result<String, AgentError> {
        let mut last = String::new();
        for u in &self.urls {
            last = http_request(u, "GET", "/health", None, self.timeout)?;
        }
        Ok(last)
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
        let worlds = self.worlds_unfiltered(observer, n)?;
        Ok(retain_valid(worlds, state, observer))
    }
}

impl SidecarWorldSource {
    /// Full deals sampled from a **mid-auction** position (`POST /auction_deals`).
    ///
    /// Distinct from [`WorldSource::worlds`], which serves the *play* phase and
    /// returns the cards each seat has **left**. Here nothing has been played,
    /// so a world is four complete hands — and, crucially, playgen v2 completes
    /// the auction with its own bid head, so the hands it invents are ones that
    /// **explain the bids already heard**. A uniform draw cannot do that: it
    /// hands a random hand to the seat that just bid 100♥, and the world then
    /// contradicts the auction it was drawn under.
    ///
    /// Same body, same round-robin, same timeout as the play-phase route — only
    /// the path differs, which is why it lives here rather than in a second type.
    ///
    /// **Découpé en plusieurs requêtes**, parce que le sidecar plafonne à son
    /// `max_worlds` (512) et le fait **en silence** : on lui demande 1024, il en
    /// rend 512, sans erreur ni champ qui le dise. Un appelant qui complète la
    /// différence autrement — mondes uniformes, par exemple — se retrouve avec
    /// la moitié d'un échantillon qui ne sait rien de l'enchère, et rien ne
    /// l'en avertit. On redemande donc jusqu'à avoir le compte.
    pub fn auction_deals(&mut self, observer: u8, n: usize) -> Result<Vec<World>, AgentError> {
        let mut out: Vec<World> = Vec::with_capacity(n);
        // Garde-fou : un sidecar qui rendrait un seul monde par appel ferait
        // boucler indéfiniment. 64 tours couvrent 32 000 mondes à 512.
        for _ in 0..64 {
            if out.len() >= n {
                break;
            }
            let body = self.request_body(observer, n - out.len());
            let url = self.pick();
            let resp = http_request(url, "POST", "/auction_deals", Some(&body), self.timeout)?;
            let batch = parse_hands(&resp).ok_or_else(|| {
                AgentError::WorldSource(format!(
                    "{}: malformed /auction_deals response ({} bytes)",
                    url,
                    resp.len()
                ))
            })?;
            // Un lot vide veut dire « je n'ai plus rien » : insister bouclerait.
            if batch.is_empty() {
                break;
            }
            out.extend(batch);
        }
        out.truncate(n);
        Ok(out)
    }

    /// The sidecar's raw answer, before [`retain_valid`]. Exposed so a benchmark
    /// can count what the filter throws away — a rejection rate that only means
    /// something if it is measured on the unfiltered stream.
    pub fn worlds_unfiltered(&mut self, observer: u8, n: usize) -> Result<Vec<World>, AgentError> {
        if n == 0 {
            return Ok(Vec::new());
        }
        let body = self.request_body(observer, n);
        let url = self.pick();
        let resp = http_request(url, "POST", "/play_worlds", Some(&body), self.timeout)?;
        parse_hands(&resp).ok_or_else(|| {
            AgentError::WorldSource(format!(
                "{}: malformed /play_worlds response ({} bytes)",
                url,
                resp.len()
            ))
        })
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
    // tiny_http passe en `Transfer-Encoding: chunked` au-delà de ~32 Ko et
    // envoie un `Content-Length` en dessous — donc **la même route change de
    // cadrage avec la taille de sa réponse**. Mesuré : 512 mondes = 21 949
    // octets avec longueur annoncée, 1024 mondes = 44 Ko en blocs.
    //
    // Le bug que ça donne ne ressemble pas à un problème de transport : les
    // tailles de bloc sont écrites **en hexadécimal dans le corps**, donc
    // `parse_hands` les compte comme des entiers de plus et rend « réponse
    // malformée ». Latent jusqu'ici parce que rien ne demandait assez de
    // mondes pour franchir le seuil.
    if head.to_ascii_lowercase().contains("transfer-encoding: chunked") {
        return dechunk(payload).ok_or_else(|| {
            AgentError::WorldSource(format!("{host}{path}: chunked response is malformed"))
        });
    }
    Ok(payload.to_string())
}

/// Recoller un corps `Transfer-Encoding: chunked` : `<taille hexa>\r\n<données>\r\n`,
/// jusqu'à une taille nulle.
fn dechunk(body: &str) -> Option<String> {
    let mut out = String::with_capacity(body.len());
    let mut rest = body;
    loop {
        let (line, tail) = rest.split_once("\r\n")?;
        // Une extension de bloc (`1a2b;nom=valeur`) est légale : on ne garde
        // que ce qui précède le `;`.
        let size_hex = line.split(';').next().unwrap_or(line).trim();
        let size = usize::from_str_radix(size_hex, 16).ok()?;
        if size == 0 {
            return Some(out);
        }
        if tail.len() < size {
            return None; // corps tronqué
        }
        out.push_str(&tail[..size]);
        // Le `\r\n` qui suit les données du bloc.
        rest = tail.get(size + 2..)?;
    }
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

    /// Le corps en blocs doit se recoller **à l'identique**. Sans ça, les
    /// tailles de bloc — écrites en hexadécimal dans le corps — se lisent comme
    /// des entiers de plus, et `parse_hands` rend « réponse malformée » sur une
    /// réponse parfaitement valide.
    #[test]
    fn a_chunked_body_is_reassembled() {
        let body = "10\r\n{\"hands\":[[1,2,3\r\n5\r\n,4]]}\r\n0\r\n\r\n";
        assert_eq!(dechunk(body).as_deref(), Some("{\"hands\":[[1,2,3,4]]}"));
    }

    /// Et le tout doit traverser `parse_hands` : c'est l'enchaînement qui a
    /// cassé en production, pas le déchunkage seul.
    #[test]
    fn a_chunked_hands_payload_parses() {
        let body = "10\r\n{\"hands\":[[1,2,3\r\n5\r\n,4]]}\r\n0\r\n\r\n";
        let joined = dechunk(body).unwrap();
        assert_eq!(parse_hands(&joined), Some(vec![[1, 2, 3, 4]]));
        // Le contrôle qui donne son sens au test : sans déchunkage, les « 10 »
        // et « 8 » entrent dans le compte et le rendent non multiple de 4.
        assert_eq!(parse_hands(body), None, "le corps brut doit bien être illisible");
    }

    #[test]
    fn a_truncated_chunked_body_is_rejected() {
        assert_eq!(dechunk("20\r\ntrop court\r\n0\r\n\r\n"), None);
        assert_eq!(dechunk("pas-de-taille\r\n"), None);
    }

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

    /// Une URL unique reste une URL unique ; une liste se répartit en
    /// tourniquet, et répéter une entrée pondère la répartition — c'est le seul
    /// réglage de proportion offert entre GPU de débits différents.
    #[test]
    fn url_list_round_robins() {
        let one = SidecarWorldSource::new("http://a:1/", 0.8, Duration::from_secs(1));
        assert_eq!(one.urls, vec!["http://a:1"]);
        assert_eq!(one.pick(), "http://a:1");

        let many =
            SidecarWorldSource::new(" http://a:1 , http://b:2/ ,", 0.8, Duration::from_secs(1));
        assert_eq!(many.urls, vec!["http://a:1", "http://b:2"]);
        let picks: Vec<&str> = (0..4).map(|_| many.pick()).collect();
        assert_eq!(picks[0], picks[2], "le tourniquet a une période de 2");
        assert_ne!(picks[0], picks[1]);

        let weighted =
            SidecarWorldSource::new("http://a:1,http://a:1,http://b:2", 0.8, Duration::from_secs(1));
        assert_eq!(weighted.urls.len(), 3);
    }

    #[test]
    fn host_gets_default_port() {
        assert_eq!(split_host("http://example.com").unwrap(), "example.com:80");
        assert_eq!(split_host("http://example.com:8003").unwrap(), "example.com:8003");
        assert!(split_host("https://example.com").is_err());
    }
}
