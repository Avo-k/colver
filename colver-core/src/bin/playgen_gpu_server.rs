//! Sidecar d'inférence GPU playgen pour la prod colver.
//!
//! Tourne sur une machine avec CUDA (la 3090 de l'hôte moxxi) ; le web (VM
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

use colver_core::playgen::gpu::GpuPlaygen;
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

fn rng_for(req: &WorldsRequest) -> StdRng {
    match req.seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    }
}

struct Server {
    model: Arc<PlaygenModel>,
    gpu: GpuPlaygen,
}

impl Server {
    fn play_worlds(&self, req: &WorldsRequest) -> Result<Vec<[u32; 4]>, String> {
        let (sampler, state) = replay(&self.model, req)?;
        if state.phase != Phase::Playing {
            return Err("position is not in play phase".into());
        }
        let n = req.n_worlds.min(MAX_WORLDS);
        let mut rng = rng_for(req);
        let worlds = self
            .gpu
            .generate_worlds_scored(&sampler, &state, n, req.temperature, &mut rng)
            .map_err(|e| format!("gpu error: {e}"))?;
        Ok(worlds.into_iter().map(|(w, _)| w).collect())
    }

    fn auction_deals(&self, req: &WorldsRequest) -> Result<Vec<[u32; 4]>, String> {
        let (sampler, state) = replay(&self.model, req)?;
        if state.phase != Phase::Bidding {
            return Err("position is not in bidding phase".into());
        }
        if sampler.is_dead() {
            return Err("over-long auction: playgen disabled for this deal".into());
        }
        let n = req.n_worlds.min(MAX_WORLDS);
        let mut rng = rng_for(req);
        let worlds = self
            .gpu
            .generate_deals_from_auction_scored(
                &sampler.prefix_tokens(),
                &state,
                sampler.observer(),
                sampler.observer_hand(),
                sampler.bid_entries_count(),
                n,
                req.temperature,
                &mut rng,
            )
            .map_err(|e| format!("gpu error: {e}"))?;
        Ok(worlds.into_iter().map(|(w, _)| w).collect())
    }

    /// Marginales p(carte → joueur) sur les mondes de jeu — même sémantique
    /// que `IsDdSearch::playgen_marginals` (cartes non vues seulement).
    fn beliefs(&self, req: &WorldsRequest) -> Result<BeliefsResponse, String> {
        let worlds = self.play_worlds(req)?;
        if worlds.is_empty() {
            return Err("no worlds generated".into());
        }
        let mut counts = [[0u32; 32]; 4];
        for hands in &worlds {
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
        Ok(BeliefsResponse { marginals, worlds: worlds.len() })
    }
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
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--playgen" => { playgen_path = args[i + 1].clone(); i += 2; }
            "--port" => { port = args[i + 1].parse().unwrap(); i += 2; }
            "--bind" => { bind = args[i + 1].clone(); i += 2; }
            other => { eprintln!("unknown arg {}", other); std::process::exit(1); }
        }
    }

    let model = Arc::new(PlaygenModel::load(&playgen_path).expect("load playgen model"));
    assert!(model.v2, "playgen_gpu_server requires a v2 (COLVPG02) model");
    let device = candle_core::Device::new_cuda(0).expect("CUDA device 0");
    let gpu = GpuPlaygen::new(&model, device).expect("upload model to GPU");
    let server = Server { model, gpu };

    let addr = format!("{bind}:{port}");
    let http = tiny_http::Server::http(&addr).expect("bind HTTP server");
    println!(
        "playgen_gpu_server: modèle {} (d={} L={}), CUDA OK, écoute sur {}",
        playgen_path, server.model.d, server.model.n_layers, addr
    );

    // Boucle séquentielle : une requête à la fois = VRAM bornée, pas de
    // contention GPU interne. Les requêtes durent ~0.1-2 s.
    for mut request in http.incoming_requests() {
        let url = request.url().to_string();
        let method = request.method().clone();
        let t0 = std::time::Instant::now();

        let response = match (method, url.as_str()) {
            (tiny_http::Method::Get, "/health") => json_response(
                200,
                format!(
                    "{{\"status\":\"ok\",\"model\":\"{}\",\"max_worlds\":{}}}",
                    server.model.d, MAX_WORLDS
                ),
            ),
            (tiny_http::Method::Post, path @ ("/beliefs" | "/auction_deals" | "/play_worlds")) => {
                let mut body = String::new();
                use std::io::Read;
                if request.as_reader().read_to_string(&mut body).is_err() {
                    json_response(400, "{\"error\":\"unreadable body\"}".into())
                } else {
                    match serde_json::from_str::<WorldsRequest>(&body) {
                        Err(e) => json_response(400, format!("{{\"error\":\"bad request: {e}\"}}")),
                        Ok(req) => {
                            let result = match path {
                                "/beliefs" => server
                                    .beliefs(&req)
                                    .map(|r| serde_json::to_string(&r).unwrap()),
                                "/auction_deals" => server
                                    .auction_deals(&req)
                                    .map(|h| serde_json::to_string(&WorldsResponse { hands: h }).unwrap()),
                                _ => server
                                    .play_worlds(&req)
                                    .map(|h| serde_json::to_string(&WorldsResponse { hands: h }).unwrap()),
                            };
                            match result {
                                Ok(body) => {
                                    println!(
                                        "{} n={} obs={} → {:.0} ms",
                                        path,
                                        req.n_worlds,
                                        req.observer,
                                        t0.elapsed().as_secs_f64() * 1e3
                                    );
                                    json_response(200, body)
                                }
                                Err(e) => {
                                    eprintln!("{} error: {}", path, e);
                                    json_response(422, format!("{{\"error\":\"{e}\"}}"))
                                }
                            }
                        }
                    }
                }
            }
            _ => json_response(404, "{\"error\":\"not found\"}".into()),
        };
        let _ = request.respond(response);
    }
}
