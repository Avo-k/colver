/// Standalone DD pool generator — no CUDA/candle dependency.
/// Generates pre-solved deals for bid NN training.
///
/// Usage:
///   cargo run -p colver-core --bin gen_pool --release -- \
///     --output data/pools/dd_pool.bin --count 1000000 --seed 42

use std::time::Instant;

fn main() {
    let mut output = String::from("data/pools/dd_pool.bin");
    let mut count: usize = 1_000_000;
    let mut seed: u64 = 42;
    let mut chunk_size: usize = 100_000;

    // Minimal arg parsing (no clap dep)
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--output" | "-o" => {
                i += 1;
                output = args[i].clone();
            }
            "--count" | "-n" => {
                i += 1;
                count = args[i].parse().expect("invalid count");
            }
            "--seed" | "-s" => {
                i += 1;
                seed = args[i].parse().expect("invalid seed");
            }
            "--chunk" => {
                i += 1;
                chunk_size = args[i].parse().expect("invalid chunk size");
            }
            "--help" | "-h" => {
                eprintln!("gen_pool: generate DD-solved deal pool for bid training");
                eprintln!("  --output/-o  Output file (default: data/pools/dd_pool.bin)");
                eprintln!("  --count/-n   Number of deals (default: 1000000)");
                eprintln!("  --seed/-s    RNG seed (default: 42)");
                eprintln!("  --chunk      Checkpoint every N deals (default: 100000)");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4);

    eprintln!("gen_pool: {} deals, seed={}, threads={}", count, seed, threads);
    eprintln!("  output: {}", output);
    eprintln!("  checkpoint every {} deals", chunk_size);

    let start = Instant::now();
    let pool = colver_core::bid_train_env::DealPool::generate_with_checkpoints(
        count, seed, &output, chunk_size,
    );

    let elapsed = start.elapsed().as_secs_f64();
    eprintln!(
        "\nDone: {} deals in {:.1}s ({:.0} deals/s)",
        pool.len(),
        elapsed,
        pool.len() as f64 / elapsed
    );
}
