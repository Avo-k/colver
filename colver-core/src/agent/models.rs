//! Process-wide model cache.
//!
//! Weight files are read once per path and shared behind an `Arc`; each player
//! then *instantiates* its own net from those weights, because inference takes
//! `&mut self` (scratch buffers) and players run concurrently.
//!
//! This is also the **single** place that knows how to interpret a weight file:
//! hidden size, layer count, dueling head, and the 411-vs-415 observation
//! layout are all auto-detected here. Every previous copy of that detection
//! (arena, PyO3, half a dozen benches) was an opportunity to get it wrong — and
//! getting it wrong makes a model play legal-but-random cards rather than
//! failing loudly.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::belief_net::BeliefNet;
use crate::bid_net::BidNet;
use crate::dmc_net::DmcNet;
use crate::playgen::infer::PlaygenModel;

use super::AgentError;

fn read_floats(path: &str) -> std::io::Result<Vec<f32>> {
    let data = std::fs::read(path)?;
    Ok(data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

// ── DMC play nets ────────────────────────────────────────────────────

/// Raw weights of a DMC play net plus the shape auto-detected from the file.
pub struct DmcWeights {
    floats: Vec<f32>,
    pub hidden: usize,
    /// 411 = canonical (`OBS_DIM_TR`), 415 = legacy. Determines whether the
    /// caller must convert masks and actions between canonical and physical
    /// suit space — see [`super::dmc::DmcPlayer`].
    pub obs_dim: usize,
    pub dueling: bool,
}

impl DmcWeights {
    fn load(path: &str) -> std::io::Result<Self> {
        let net = DmcNet::load(path)?;
        let (hidden, obs_dim, dueling) = (net.hidden(), net.obs_dim(), net.is_dueling());
        drop(net);
        Ok(DmcWeights { floats: read_floats(path)?, hidden, obs_dim, dueling })
    }

    /// Les flottants bruts, pour un backend qui construit ses propres tenseurs
    /// (le déroulement groupé sur GPU). Rendus tels quels : la disposition est
    /// documentée par `DmcNet::from_floats`, qui reste la référence.
    pub fn floats(&self) -> &[f32] {
        &self.floats
    }

    /// Build a fresh net for one player. `residual` enables the skip
    /// connections of the triforge/DouDou50 architecture — same weights,
    /// different forward pass, so it cannot be detected from the file.
    pub fn instantiate(&self, residual: bool) -> DmcNet {
        let mut net = DmcNet::from_floats(&self.floats, self.hidden, self.obs_dim, self.dueling)
            .expect("DMC weights validated at load time");
        net.set_residual(residual);
        net
    }
}

// ── Bid nets ─────────────────────────────────────────────────────────

/// Raw weights of a bid net plus its auto-detected shape.
pub struct BidWeights {
    floats: Vec<f32>,
    pub hidden: usize,
    /// 108 = plain, 110/113/117 = score-aware v1/v2/v3.
    pub obs_dim: usize,
    pub dueling: bool,
    pub layers: usize,
}

impl BidWeights {
    fn load(path: &str, hidden_hint: usize) -> std::io::Result<Self> {
        let net = BidNet::load_with_hidden(path, hidden_hint)?;
        let (hidden, obs_dim, dueling, layers) =
            (net.hidden(), net.obs_dim(), net.is_dueling(), net.layers());
        drop(net);
        Ok(BidWeights { floats: read_floats(path)?, hidden, obs_dim, dueling, layers })
    }

    pub fn instantiate(&self) -> BidNet {
        BidNet::from_floats_with_layers(
            &self.floats,
            self.hidden,
            self.obs_dim,
            self.dueling,
            self.layers,
        )
        .expect("bid weights validated at load time")
    }
}

// ── Cache ────────────────────────────────────────────────────────────

type Cache<T> = OnceLock<Mutex<HashMap<String, Arc<T>>>>;

static DMC_CACHE: Cache<DmcWeights> = OnceLock::new();
static BID_CACHE: Cache<BidWeights> = OnceLock::new();
static PLAYGEN_CACHE: Cache<PlaygenModel> = OnceLock::new();

fn cached<T, F>(cache: &Cache<T>, key: String, load: F) -> Result<Arc<T>, AgentError>
where
    F: FnOnce() -> std::io::Result<T>,
{
    let map = cache.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = map.lock().expect("model cache poisoned");
    if let Some(hit) = map.get(&key) {
        return Ok(hit.clone());
    }
    let loaded = Arc::new(load().map_err(|e| AgentError::Model(format!("{key}: {e}")))?);
    map.insert(key, loaded.clone());
    Ok(loaded)
}

/// Load (or reuse) DMC play-net weights.
pub fn dmc_weights(path: &str) -> Result<Arc<DmcWeights>, AgentError> {
    cached(&DMC_CACHE, path.to_string(), || DmcWeights::load(path))
}

/// Load (or reuse) bid-net weights. `hidden_hint` is only consulted for files
/// whose hidden size is ambiguous; the detected value wins.
pub fn bid_weights(path: &str, hidden_hint: usize) -> Result<Arc<BidWeights>, AgentError> {
    cached(&BID_CACHE, format!("{path}#{hidden_hint}"), || {
        BidWeights::load(path, hidden_hint)
    })
}

/// Load (or reuse) a playgen world-sampler model (~13 MB, read-only at
/// inference, so a single copy is shared by every player in the process).
pub fn playgen_model(path: &str) -> Result<Arc<PlaygenModel>, AgentError> {
    cached(&PLAYGEN_CACHE, path.to_string(), || PlaygenModel::load(path))
}

/// Load a belief net. Not cached: `BeliefNet` inference needs `&mut self` and
/// the type exposes no weights/instantiate split, so each owner loads its own.
/// Falls back to `hidden = 256` for the bid-belief (COLVBB) files, whose header
/// does not pin the size down.
pub fn belief_net(path: &str) -> Result<BeliefNet, AgentError> {
    BeliefNet::load(path)
        .or_else(|_| BeliefNet::load_with_hidden(path, 256))
        .map_err(|e| AgentError::Model(format!("{path}: {e}")))
}
