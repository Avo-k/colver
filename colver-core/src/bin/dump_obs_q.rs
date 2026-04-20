/// Dump (obs, q_values) pairs for verifying PyTorch reimplementation.
///
/// Writes a binary file:
///   header: u32 obs_dim, u32 num_actions, u32 n_samples
///   for each sample: obs_dim f32 (obs) + num_actions f32 (q-values)
///
/// Usage:
///   cargo run -p colver-core --bin dump_obs_q --release -- <model> <n> <out.bin>

use colver_core::bid_net::BidNet;
use colver_core::bid_obs;
use colver_core::state::GameState;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::io::Write;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).map(|s| s.as_str()).unwrap_or("models/bid_v5_isdd/bid_nn_final.bin");
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
    let out_path = args.get(3).map(|s| s.as_str()).unwrap_or("/tmp/dump_obs_q.bin");

    let mut net = BidNet::load_with_hidden(model_path, 512).unwrap();
    let obs_dim = net.obs_dim();
    let num_actions = 43u32;
    let mut rng = StdRng::seed_from_u64(12345);

    let mut f = std::io::BufWriter::new(std::fs::File::create(out_path).unwrap());
    f.write_all(&(obs_dim as u32).to_le_bytes()).unwrap();
    f.write_all(&num_actions.to_le_bytes()).unwrap();
    f.write_all(&(n as u32).to_le_bytes()).unwrap();

    for i in 0..n {
        // Vary the scenario: dealer 0 (pos1) for first half, dealer 1 (pos2 opener) for second
        let dealer = if i < n / 2 { 3 } else { 0 };
        let state = GameState::deal_random(dealer, &mut rng);
        let mut buf = vec![0.0f32; obs_dim];
        match obs_dim {
            108 => bid_obs::write_bid_observation(&mut buf, 0, &state, &[]),
            110 => bid_obs::write_bid_observation_score_aware(&mut buf, 0, &state, &[], 0, 0),
            113 => bid_obs::write_bid_observation_score_aware_v2(&mut buf, 0, &state, &[], 0, 0),
            _ => panic!("unsupported obs_dim {}", obs_dim),
        }
        let q = net.evaluate(&buf);

        for v in &buf {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        for v in &q {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
    }
    eprintln!("Wrote {} samples to {}", n, out_path);
}
