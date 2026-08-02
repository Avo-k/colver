//! Sidecar d'inférence GPU playgen pour la prod colver.
//!
//! Tourne sur une machine avec CUDA (la 3090 de l'hôte) ; le web (VM
//! Docker sans GPU) l'appelle en HTTP et retombe sur l'inférence CPU locale
//! (PyO3) en cas d'erreur ou de timeout — le chemin CPU produit des mondes de
//! la même distribution (forward validé bit-à-bit), le sidecar n'est qu'un
//! accélérateur opportuniste.
//!
//! Protocole : le client envoie la *partie rejouable* (donne initiale +
//! actions + observateur) ; le serveur reconstruit le PlaygenSampler par
//! replay (même code que la prod CPU) puis échantillonne sur GPU.
//!
//! POST /beliefs        → marginales [4][32] p(carte chez joueur)
//! POST /auction_deals  → mains complètes 8 cartes/siège (phase enchères)
//! POST /play_worlds    → mains restantes (phase jeu, pour injection IS-DD)
//! GET  /health         → état + device
//!
//! Requête JSON commune :
//!   { "dealer": 0, "hands": [u32;4], "actions": [[joueur, action], ...],
//!     "observer": 2, "n_worlds": 30, "temperature": 0.8, "seed": null }
//!
//! Usage :
//!   CUDARC_CUDA_VERSION=13010 cargo build --bin playgen_gpu_server \
//!     --features gpu_server --release
//!   ./playgen_gpu_server --playgen models/playgen_v2_half.bin --port 8003

use std::sync::Arc;

use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

use colver_core::playgen::gpu::{GpuPlaygen, WorldBatchItem};
use colver_core::playgen::infer::{PlaygenModel, PlaygenSampler};
use colver_core::state::{GameState, Phase};

/// Borne dure sur le batch (VRAM prévisible : ~3 Mo/lane de KV cache).
const MAX_WORLDS: usize = 512;

#[derive(Deserialize)]
struct WorldsRequest {
    dealer: u8,
    hands: [u32; 4],
    actions: Vec<(u8, u8)>,
    observer: u8,
    n_worlds: usize,
    temperature: f32,
    #[serde(default)]
    seed: Option<u64>,
}

#[derive(Serialize)]
struct WorldsResponse {
    hands: Vec<[u32; 4]>,
}

#[derive(Serialize)]
struct BeliefsResponse {
    marginals: [[f32; 32]; 4],
    worlds: usize,
}

/// Rejoue la partie et rend (sampler prêt, état courant).
fn replay(model: &Arc<PlaygenModel>, req: &WorldsRequest) -> Result<(PlaygenSampler, GameState), String> {
    let state0 = GameState::new(req.dealer, req.hands);
    let mut sampler = PlaygenSampler::new(model.clone());
    sampler.init_deal(&state0, req.observer);
    let mut state = state0;
    for &(p, a) in &req.actions {
        if state.phase == Phase::Done {
            return Err("actions after end of deal".into());
        }
        if state.current_player() != p {
            return Err(format!("action by {} but current player is {}", p, state.current_player()));
        }
        if state.legal_actions() & (1u64 << a) == 0 {
            return Err(format!("illegal action {} for player {}", a, p));
        }
        sampler.record_action(&state, p, a);
        state.step(a);
    }
    Ok((sampler, state))
}


/// What a queued job wants from the GPU.
enum JobKind {
    /// Mid-play worlds. Batchable across unrelated positions.
    Play,
    /// Same worlds, reduced to card marginals on the host. Batchable.
    Beliefs,
    /// Mid-auction deals. **Not** batchable — the auction path has its own
    /// lockstep state machine and no multi-position variant.
    Auction,
}

enum JobOut {
    Worlds(Vec<[u32; 4]>),
    Beliefs(BeliefsResponse),
}

/// A parsed, replayed request waiting for GPU time.
///
/// Replay happens on the handler thread, so the CPU cost of rebuilding the
/// sampler is spread across handlers instead of serializing in front of the GPU.
struct Job {
    kind: JobKind,
    sampler: PlaygenSampler,
    state: GameState,
    n: usize,
    temperature: f32,
    /// An explicit seed means the caller wants a reproducible draw, which a
    /// shared batch RNG cannot give: lanes interleave their draws. Such a job
    /// is therefore run **alone**. No current client sends one.
    seed: Option<u64>,
    t0: std::time::Instant,
    reply: std::sync::mpsc::Sender<Result<JobOut, String>>,
}

struct Server {
    model: Arc<PlaygenModel>,
    gpu: GpuPlaygen,
}

impl Server {

    /// Run a group of play/beliefs jobs as **one** GPU batch.
    ///
    /// This is the whole point of the rework: a request costs ~220 ms whether it
    /// asks for 1 world or 256, because the cost is the sequential token loop,
    /// not the arithmetic. Serving requests one at a time therefore left most of
    /// the GPU idle no matter how many clients were waiting.
    fn run_batch(&self, jobs: Vec<Job>, rng: &mut StdRng) {
        let items: Vec<WorldBatchItem> = jobs
            .iter()
            .map(|j| WorldBatchItem {
                sampler: &j.sampler,
                state: &j.state,
                n_worlds: j.n,
                temperature: j.temperature,
            })
            .collect();

        match self.gpu.generate_worlds_multi(&items, rng) {
            Err(e) => {
                let msg = format!("gpu error: {e}");
                for j in jobs {
                    let _ = j.reply.send(Err(msg.clone()));
                }
            }
            Ok(per_item) => {
                for (j, scored) in jobs.into_iter().zip(per_item.into_iter()) {
                    let worlds: Vec<[u32; 4]> = scored.into_iter().map(|(w, _)| w).collect();
                    let out = match j.kind {
                        JobKind::Beliefs => {
                            if worlds.is_empty() {
                                Err("no worlds generated".into())
                            } else {
                                Ok(JobOut::Beliefs(marginals_of(&worlds)))
                            }
                        }
                        _ => Ok(JobOut::Worlds(worlds)),
                    };
                    let _ = j.reply.send(out);
                }
            }
        }
    }


    /// Run one job on its own — the auction path, and any seeded request.
    fn run_alone(&self, job: Job, rng: &mut StdRng) {
        let mut own;
        let rng: &mut StdRng = match job.seed {
            Some(s) => {
                own = StdRng::seed_from_u64(s);
                &mut own
            }
            None => rng,
        };
        let out = match job.kind {
            JobKind::Auction => self
                .gpu
                .generate_deals_from_auction_scored(
                    &job.sampler.prefix_tokens(),
                    &job.state,
                    job.sampler.observer(),
                    job.sampler.observer_hand(),
                    job.sampler.bid_entries_count(),
                    job.n,
                    job.temperature,
                    rng,
                )
                .map_err(|e| format!("gpu error: {e}"))
                .map(|w| JobOut::Worlds(w.into_iter().map(|(h, _)| h).collect())),
            _ => self
                .gpu
                .generate_worlds_scored(&job.sampler, &job.state, job.n, job.temperature, rng)
                .map_err(|e| format!("gpu error: {e}"))
                .map(|w| {
                    let hands: Vec<[u32; 4]> = w.into_iter().map(|(h, _)| h).collect();
                    match job.kind {
                        JobKind::Beliefs => JobOut::Beliefs(marginals_of(&hands)),
                        _ => JobOut::Worlds(hands),
                    }
                }),
        };
        let _ = job.reply.send(out);
    }

    /// Parse, replay, hand to the GPU worker, wait, serialise.
    ///
    /// Replay runs here, on the handler thread, so rebuilding the sampler for N
    /// clients costs one CPU core each instead of serialising in front of the GPU.
    fn enqueue(
        &self,
        path: &str,
        req: &WorldsRequest,
        tx: &std::sync::mpsc::Sender<Job>,
        t0: std::time::Instant,
    ) -> Result<String, String> {
        let (sampler, state) = replay(&self.model, req)?;
        let kind = match path {
            "/beliefs" => JobKind::Beliefs,
            "/auction_deals" => JobKind::Auction,
            _ => JobKind::Play,
        };
        match kind {
            JobKind::Auction if state.phase != Phase::Bidding => {
                return Err("position is not in bidding phase".into())
            }
            JobKind::Auction if sampler.is_dead() => {
                return Err("over-long auction: playgen disabled for this deal".into())
            }
            JobKind::Play | JobKind::Beliefs if state.phase != Phase::Playing => {
                return Err("position is not in play phase".into())
            }
            _ => {}
        }

        let (rtx, rrx) = std::sync::mpsc::channel();
        tx.send(Job {
            kind,
            sampler,
            state,
            n: req.n_worlds.min(MAX_WORLDS),
            temperature: req.temperature,
            seed: req.seed,
            t0,
            reply: rtx,
        })
        .map_err(|_| "gpu worker gone".to_string())?;

        match rrx.recv().map_err(|_| "gpu worker dropped the job".to_string())?? {
            JobOut::Worlds(h) => Ok(serde_json::to_string(&WorldsResponse { hands: h }).unwrap()),
            JobOut::Beliefs(b) => Ok(serde_json::to_string(&b).unwrap()),
        }
    }

}

/// p(carte → joueur) sur un ensemble de mondes. Extrait de `beliefs` pour que
/// le chemin groupé puisse réduire sans repasser par le GPU.
fn marginals_of(worlds: &[[u32; 4]]) -> BeliefsResponse {
    let mut counts = [[0u32; 32]; 4];
    for hands in worlds {
        for p in 0..4 {
            let mut h = hands[p];
            while h != 0 {
                counts[p][h.trailing_zeros() as usize] += 1;
                h &= h - 1;
            }
        }
    }
    let total = worlds.len() as f32;
    let mut marginals = [[0.0f32; 32]; 4];
    for p in 0..4 {
        for c in 0..32 {
            marginals[p][c] = counts[p][c] as f32 / total;
        }
    }
    BeliefsResponse { marginals, worlds: worlds.len() }
}

fn json_response(status: u32, body: String) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    let header = tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..])
        .expect("static header");
    tiny_http::Response::from_string(body)
        .with_status_code(tiny_http::StatusCode(status as u16))
        .with_header(header)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut playgen_path = String::from("models/playgen_v2/playgen_v2_half.bin");
    let mut port = 8003u16;
    let mut bind = String::from("0.0.0.0");
    // Au moins autant que de clients simultanés attendus (IS-DD en arène en
    // lance un par thread rayon), sinon le groupeur ne voit jamais la charge.
    let mut handlers = 64usize;
    // Plafond de lanes par lot GPU. ~3 Mo de KV cache par lane, donc 1024
    // lanes ≈ 2 Go — large sous les 24 Go, et bien au-delà du genou de débit.
    let mut lane_budget = 1024usize;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--playgen" => { playgen_path = args[i + 1].clone(); i += 2; }
            "--port" => { port = args[i + 1].parse().unwrap(); i += 2; }
            "--bind" => { bind = args[i + 1].clone(); i += 2; }
            "--handlers" => { handlers = args[i + 1].parse().unwrap(); i += 2; }
            "--lane-budget" => { lane_budget = args[i + 1].parse().unwrap(); i += 2; }
            other => { eprintln!("unknown arg {}", other); std::process::exit(1); }
        }
    }

    let model = Arc::new(PlaygenModel::load(&playgen_path).expect("load playgen model"));
    assert!(model.v2, "playgen_gpu_server requires a v2 (COLVPG02) model");
    let device = candle_core::Device::new_cuda(0).expect("CUDA device 0");
    let gpu = GpuPlaygen::new(&model, device).expect("upload model to GPU");
    let (d, n_layers) = (model.d, model.n_layers);
    let server = Arc::new(Server { model: model.clone(), gpu });

    let addr = format!("{bind}:{port}");
    let http = Arc::new(tiny_http::Server::http(&addr).expect("bind HTTP server"));
    println!(
        "playgen_gpu_server: modèle {} (d={} L={}), CUDA OK, écoute sur {}",
        playgen_path, d, n_layers, addr
    );
    println!(
        "  {handlers} threads d'accueil, lot GPU jusqu'à {lane_budget} lanes"
    );

    let (tx, rx) = std::sync::mpsc::channel::<Job>();

    // ── Le worker GPU : unique propriétaire du device ──────────────────
    //
    // Il bloque sur la première requête, puis ramasse *tout ce qui est déjà
    // arrivé* jusqu'à `lane_budget` lanes et le passe en un seul lot. Aucune
    // fenêtre d'attente artificielle : sous charge la file est toujours garnie,
    // et à vide une requête isolée part immédiatement — la latence à un seul
    // client ne se dégrade pas.
    let worker_server = server.clone();
    let worker = std::thread::spawn(move || {
        let mut rng = StdRng::from_entropy();
        let mut batches = 0u64;
        let mut jobs_total = 0u64;
        while let Ok(first) = rx.recv() {
            // L'enchère n'a pas de variante multi-positions, et une requête
            // graînée doit rester reproductible : les deux partent seules.
            if matches!(first.kind, JobKind::Auction) || first.seed.is_some() {
                worker_server.run_alone(first, &mut rng);
                continue;
            }
            let mut batch = vec![first];
            let mut lanes = batch[0].n;
            while lanes < lane_budget {
                match rx.try_recv() {
                    Ok(j) => {
                        if matches!(j.kind, JobKind::Auction) || j.seed.is_some() {
                            worker_server.run_alone(j, &mut rng);
                        } else {
                            lanes += j.n;
                            batch.push(j);
                        }
                    }
                    Err(_) => break,
                }
            }
            batches += 1;
            jobs_total += batch.len() as u64;
            let t0 = std::time::Instant::now();
            let (n, l) = (batch.len(), lanes);
            // Queue wait included: what a caller feels is time-since-arrival,
            // and batching trades a little of that for throughput.
            let waited = batch.iter().map(|j| j.t0.elapsed()).max().unwrap_or_default();
            worker_server.run_batch(batch, &mut rng);
            println!(
                "lot: {n} requêtes, {l} lanes → {:.0} ms GPU ({:.1} ms/requête), \
                 attente max {:.0} ms, moyenne {:.1} req/lot",
                t0.elapsed().as_secs_f64() * 1e3,
                t0.elapsed().as_secs_f64() * 1e3 / n as f64,
                waited.as_secs_f64() * 1e3,
                jobs_total as f64 / batches as f64,
            );
        }
    });

    // ── Les threads d'accueil : parse + rejeu, puis attente du GPU ──────
    //
    // Il en faut au moins autant que de clients simultanés, sinon les requêtes
    // s'empilent dans la file d'acceptation TCP et n'atteignent jamais le
    // groupeur — le débit retomberait à celui du serveur séquentiel.
    let mut threads = Vec::new();
    for _ in 0..handlers {
        let http = http.clone();
        let server = server.clone();
        let tx = tx.clone();
        threads.push(std::thread::spawn(move || loop {
            let mut request = match http.recv() {
                Ok(r) => r,
                Err(_) => break,
            };
            let url = request.url().to_string();
            let method = request.method().clone();
            let t0 = std::time::Instant::now();

            let response = match (method, url.as_str()) {
                (tiny_http::Method::Get, "/health") => json_response(
                    200,
                    format!(
                        "{{\"status\":\"ok\",\"model\":\"{}\",\"max_worlds\":{}}}",
                        d, MAX_WORLDS
                    ),
                ),
                (tiny_http::Method::Post, path @ ("/beliefs" | "/auction_deals" | "/play_worlds")) => {
                    let mut body = String::new();
                    if request.as_reader().read_to_string(&mut body).is_err() {
                        json_response(400, "{\"error\":\"unreadable body\"}".into())
                    } else {
                        match serde_json::from_str::<WorldsRequest>(&body) {
                            Err(e) => {
                                json_response(400, format!("{{\"error\":\"bad request: {e}\"}}"))
                            }
                            Ok(req) => match server.enqueue(path, &req, &tx, t0) {
                                Ok(body) => json_response(200, body),
                                Err(e) => {
                                    eprintln!("{} error: {}", path, e);
                                    json_response(422, format!("{{\"error\":\"{e}\"}}"))
                                }
                            },
                        }
                    }
                }
                _ => json_response(404, "{\"error\":\"not found\"}".into()),
            };
            let _ = request.respond(response);
        }));
    }
    drop(tx);
    for t in threads {
        let _ = t.join();
    }
    let _ = worker.join();
}
