/// Retrospective evaluation of DMC checkpoints with duplicate matching.
///
/// Loads each checkpoint .bin file and runs dmc_eval::run_eval() with
/// seeded duplicate matching to get reliable historical eval data.
///
/// Usage:
///   cargo run -p colver-core --bin retro_eval --release --features rand -- \
///     --model-dir models/ \
///     --bid-model models/bid_nn_final.bin \
///     --baseline models/dmc_35.bin \
///     --min-step 1000000 --max-step 16000000 --step-interval 1000000 \
///     --output retro_evals.txt

use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

use colver_core::bid_net::BidNet;
use colver_core::dmc_eval::{EvalConfig, run_eval};
use colver_core::dmc_net::DmcNet;

fn get_arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn get_arg_or(args: &[String], name: &str, default: &str) -> String {
    get_arg(args, name).unwrap_or_else(|| default.to_string())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let model_dir = get_arg_or(&args, "--model-dir", "models");
    let bid_model_path = get_arg(&args, "--bid-model");
    let baseline_path = get_arg(&args, "--baseline");
    let min_step: usize = get_arg_or(&args, "--min-step", "1000000").parse().unwrap();
    let max_step: usize = get_arg_or(&args, "--max-step", "16000000").parse().unwrap();
    let step_interval: usize = get_arg_or(&args, "--step-interval", "1000000").parse().unwrap();
    let random_matches: usize = get_arg_or(&args, "--random-matches", "100").parse().unwrap();
    let checkpoint_matches: usize = get_arg_or(&args, "--checkpoint-matches", "50").parse().unwrap();
    let isdd_matches: usize = get_arg_or(&args, "--isdd-matches", "30").parse().unwrap();
    let isdd_time_ms: u32 = get_arg_or(&args, "--isdd-time-ms", "20").parse().unwrap();
    let output_path = get_arg_or(&args, "--output", "retro_evals.txt");

    println!("=== Retrospective Eval (Duplicate Matching) ===");
    println!("Model dir: {}", model_dir);
    println!("Bid model: {:?}", bid_model_path);
    println!("Baseline: {:?}", baseline_path);
    println!("Steps: {}..{} (interval {})", min_step, max_step, step_interval);
    println!("Matches: rand={}, ckpt={}, isdd={} ({}ms/move)", random_matches, checkpoint_matches, isdd_matches, isdd_time_ms);
    println!("Output: {}", output_path);
    println!();

    // Load bid model
    let mut bid_net = bid_model_path.as_ref().map(|path| {
        BidNet::load(path).unwrap_or_else(|e| panic!("Failed to load bid model {}: {}", path, e))
    });
    if bid_net.is_some() {
        println!("Loaded bid model");
    }

    // Load baseline checkpoint
    let mut baseline_net = baseline_path.as_ref().map(|path| {
        DmcNet::load(path).unwrap_or_else(|e| panic!("Failed to load baseline {}: {}", path, e))
    });
    if baseline_net.is_some() {
        println!("Loaded baseline checkpoint");
    }

    let config = EvalConfig {
        random_matches,
        checkpoint_matches,
        isdd_matches,
        isdd_time_ms,
    };

    let out_file = File::create(&output_path).expect("Failed to create output file");
    let mut out = BufWriter::new(out_file);

    let total_start = Instant::now();
    let mut step = min_step;
    while step <= max_step {
        let path = format!("{}/dmc_{}.bin", model_dir, step);

        // Check if file exists
        if !std::path::Path::new(&path).exists() {
            eprintln!("Skipping step {} — {} not found", step, path);
            step += step_interval;
            continue;
        }

        print!("Step {:>10} ... ", step);
        std::io::stdout().flush().ok();

        let mut q_net = match DmcNet::load(&path) {
            Ok(n) => n,
            Err(e) => {
                eprintln!("Failed to load {}: {}", path, e);
                step += step_interval;
                continue;
            }
        };

        let result = run_eval(&mut q_net, &mut baseline_net, &mut bid_net, &config);

        let mut parts = Vec::new();
        if random_matches > 0 {
            parts.push(format!("rand {:.0}%", result.rand_wr * 100.0));
        }
        if baseline_net.is_some() && checkpoint_matches > 0 {
            parts.push(format!("ckpt {:.0}%", result.ckpt_wr * 100.0));
        }
        if isdd_matches > 0 {
            parts.push(format!("isdd {:.0}%", result.isdd_wr * 100.0));
        }

        let line = format!("{} [EVAL] {} ({:.0}s)", step, parts.join(" | "), result.elapsed);
        println!("{}", line.trim_start_matches(&format!("{} ", step)));
        writeln!(out, "{}", line).expect("Failed to write output");
        out.flush().expect("Failed to flush output");

        step += step_interval;
    }

    let total_elapsed = total_start.elapsed().as_secs_f64();
    println!("\nDone! Total time: {:.0}s ({:.1} min)", total_elapsed, total_elapsed / 60.0);
    println!("Results written to {}", output_path);
}
