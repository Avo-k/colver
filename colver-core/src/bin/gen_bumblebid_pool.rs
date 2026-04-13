/// Pre-tokenize DD pool into Bumblebid format for fast GPU training.
///
/// Reads a COLVDD01 pool, extracts card tokens for all 4 seats per deal,
/// writes a compact binary that Python can mmap/load directly into GPU tensors.
///
/// Output format "COLVBB01":
///   Magic: "COLVBB01" (8B)
///   Count: u64 LE (number of deals)
///   Per deal (85 bytes):
///     dealer: u8
///     dd_pts: [u8; 4]
///     For seat 0..3 (80 bytes):
///       primary_ids: [u8; 10]  — [CLS, POS_x, rank0..rank7]
///       suit_ids:    [u8; 10]  — [S_NULL, S_NULL, suit0..suit7]
///
/// Usage:
///   cargo run -p colver-core --bin gen_bumblebid_pool --release -- \
///     --input data/pools/dd_2.5M.bin --output data/pools/bumblebid_2.5M.bin

use std::io::{BufReader, BufWriter, Read, Write};
use std::time::Instant;

// Bumblebid token IDs (must match scripts/bumblebid/model.py)
const P_CLS: u8 = 1;
const P_POS0: u8 = 2;
const P_RANK0: u8 = 6;
const S_NULL: u8 = 4;

fn main() {
    let mut input = String::from("data/pools/dd_2.5M.bin");
    let mut output = String::from("data/pools/bumblebid_2.5M.bin");

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--input" | "-i" => {
                i += 1;
                input = args[i].clone();
            }
            "--output" | "-o" => {
                i += 1;
                output = args[i].clone();
            }
            "--help" | "-h" => {
                eprintln!("gen_bumblebid_pool: pre-tokenize DD pool for Bumblebid training");
                eprintln!("  --input/-i   Input COLVDD01 pool (default: data/pools/dd_2.5M.bin)");
                eprintln!("  --output/-o  Output file (default: data/pools/bumblebid_2.5M.bin)");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let t0 = Instant::now();

    // Load COLVDD01 pool
    let mut f = BufReader::new(std::fs::File::open(&input).expect("cannot open input"));
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic).unwrap();
    assert!(
        &magic == b"COLVDD01" || &magic == b"COLVDR01",
        "Bad magic: expected COLVDD01 or COLVDR01, got {:?}",
        std::str::from_utf8(&magic)
    );
    let is_enriched = &magic == b"COLVDR01";

    let mut count_buf = [0u8; 8];
    f.read_exact(&mut count_buf).unwrap();
    let count = u64::from_le_bytes(count_buf) as usize;
    eprintln!("Loading {} deals from {} ...", count, input);

    // Write output — COLVBB02 if enriched (includes real_pts), COLVBB01 otherwise
    let mut out = BufWriter::new(std::fs::File::create(&output).expect("cannot create output"));
    if is_enriched {
        out.write_all(b"COLVBB02").unwrap();
    } else {
        out.write_all(b"COLVBB01").unwrap();
    }
    out.write_all(&(count as u64).to_le_bytes()).unwrap();

    for _ in 0..count {
        // Read one deal from input
        let mut dealer = [0u8; 1];
        f.read_exact(&mut dealer).unwrap();

        let mut hands = [0u32; 4];
        for h in &mut hands {
            let mut buf = [0u8; 4];
            f.read_exact(&mut buf).unwrap();
            *h = u32::from_le_bytes(buf);
        }

        let mut dd_pts = [0u8; 4];
        f.read_exact(&mut dd_pts).unwrap();

        let mut real_pts = [0u8; 4];
        if is_enriched {
            f.read_exact(&mut real_pts).unwrap();
        }

        // Write dealer + dd_pts (+ real_pts if enriched)
        out.write_all(&dealer).unwrap();
        out.write_all(&dd_pts).unwrap();
        if is_enriched {
            out.write_all(&real_pts).unwrap();
        }

        // Tokenize each seat
        for seat in 0u8..4 {
            let hand = hands[seat as usize];

            // Extract cards as (rank, suit) sorted by suit*8 + rank
            let mut cards = [(0u8, 0u8); 8];
            let mut n = 0;
            for bit in 0..32u8 {
                if hand & (1u32 << bit) != 0 {
                    let rank = bit % 8;
                    let suit = bit / 8;
                    cards[n] = (rank, suit);
                    n += 1;
                }
            }
            assert_eq!(n, 8, "Hand must have exactly 8 cards, got {}", n);
            cards.sort_by_key(|&(r, s)| s * 8 + r);

            // Position relative to dealer
            let pos = (seat + 4 - dealer[0]) % 4;

            // Build primary_ids[10]
            let mut primary = [0u8; 10];
            primary[0] = P_CLS;
            primary[1] = P_POS0 + pos;
            for j in 0..8 {
                primary[2 + j] = P_RANK0 + cards[j].0;
            }

            // Build suit_ids[10]
            let mut suits = [0u8; 10];
            suits[0] = S_NULL;
            suits[1] = S_NULL;
            for j in 0..8 {
                suits[2 + j] = cards[j].1;
            }

            out.write_all(&primary).unwrap();
            out.write_all(&suits).unwrap();
        }
    }

    out.flush().unwrap();
    let elapsed = t0.elapsed();
    let file_size = std::fs::metadata(&output).unwrap().len();
    eprintln!(
        "Wrote {} deals to {} ({:.1} MB) in {:.1}s",
        count,
        output,
        file_size as f64 / 1_048_576.0,
        elapsed.as_secs_f64()
    );
}
