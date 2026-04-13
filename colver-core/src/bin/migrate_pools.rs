/// Migrate old pool files to new data/deals/ layout.
///
/// 1. Copy dd_5M.bin → data/deals/base_5M.bin (as-is, COLVDD01)
/// 2. Extract DMC scores from dd_5M_enriched.bin → data/deals/scores_dmc_5M.sc (COLVSC01)
/// 3. Merge 5 IS-DD seq files → data/deals/scores_isdd_500k.sc (COLVSC01)
/// 4. Move old files to data/deals/archive/
///
/// Usage:
///   cargo run --bin migrate_pools --release

use std::io::{Read, Write};

use colver_core::bid_train_env::DealPool;

fn main() {
    std::fs::create_dir_all("data/deals/archive").expect("Failed to create data/deals/archive");

    // 1. Copy base pool
    println!("=== Step 1: Copy base pool ===");
    if std::path::Path::new("data/deals/base_5M.bin").exists() {
        println!("  data/deals/base_5M.bin already exists, skipping");
    } else {
        std::fs::copy("data/pools/dd_5M.bin", "data/deals/base_5M.bin")
            .expect("Failed to copy dd_5M.bin");
        println!("  Copied dd_5M.bin → data/deals/base_5M.bin");
    }

    // 2. Extract DMC scores from enriched pool
    println!("\n=== Step 2: Extract DMC scores ===");
    if std::path::Path::new("data/deals/scores_dmc_5M.sc").exists() {
        println!("  data/deals/scores_dmc_5M.sc already exists, skipping");
    } else {
        extract_scores_from_enriched(
            "data/pools/dd_5M_enriched.bin",
            "dmc",
            0,
            "data/deals/scores_dmc_5M.sc",
        );
    }

    // 3. Merge IS-DD score files
    println!("\n=== Step 3: Merge IS-DD scores ===");
    if std::path::Path::new("data/deals/scores_isdd_500k.sc").exists() {
        println!("  data/deals/scores_isdd_500k.sc already exists, skipping");
    } else {
        let isdd_files = [
            ("data/pools/dd_100k_seq_enriched_isdd.bin", 0usize),
            ("data/pools/dd_100k_seq2_enriched_isdd.bin", 100_000),
            ("data/pools/dd_100k_seq3_enriched_isdd.bin", 200_000),
            ("data/pools/dd_100k_seq4_enriched_isdd.bin", 300_000),
            ("data/pools/dd_100k_seq5_enriched_isdd.bin", 400_000),
        ];

        let mut all_scores: Vec<[u8; 4]> = Vec::with_capacity(500_000);
        for (path, expected_offset) in &isdd_files {
            if !std::path::Path::new(path).exists() {
                eprintln!("  WARNING: {} not found, skipping", path);
                continue;
            }
            let scores = read_real_pts_from_enriched(path);
            println!(
                "  {} → {} scores (expected offset {})",
                path, scores.len(), expected_offset
            );
            assert_eq!(
                all_scores.len(),
                *expected_offset,
                "Score offset mismatch: expected {}, got {}",
                expected_offset,
                all_scores.len()
            );
            all_scores.extend_from_slice(&scores);
        }

        println!("  Total: {} IS-DD scores", all_scores.len());
        DealPool::save_scores("isdd", 0, &all_scores, "data/deals/scores_isdd_500k.sc")
            .expect("Failed to save IS-DD scores");
        println!("  Saved data/deals/scores_isdd_500k.sc");
    }

    // 4. Archive old files
    println!("\n=== Step 4: Archive old files ===");
    let archive_files = [
        "dd_2.5M.bin",
        "dd_2.5M_b.bin",
        "dd_pool_enriched.bin",
        "dd_pool_enriched_1M.bin",
        "dd_1k_enriched_isdd.bin",
        "dd_100k_enriched_isdd.bin",
        "dd_100k_seq_enriched_isdd.bin",
        "dd_100k_seq2_enriched_isdd.bin",
        "dd_100k_seq3_enriched_isdd.bin",
        "dd_100k_seq4_enriched_isdd.bin",
        "dd_100k_seq5_enriched_isdd.bin",
        "dd_oracle_best_62k.bin",
        "dd_100k_ns_dmc_ew_isdd.bin",
        "dd_100k_ns_isdd_ew_dmc.bin",
        "bumblebid_2.5M.bin",
        "bumblebid_5M_enriched.bin",
    ];

    for name in &archive_files {
        let src = format!("data/pools/{}", name);
        let dst = format!("data/deals/archive/{}", name);
        if std::path::Path::new(&src).exists() {
            if std::path::Path::new(&dst).exists() {
                println!("  {} already archived, skipping", name);
            } else {
                std::fs::rename(&src, &dst).unwrap_or_else(|e| {
                    eprintln!("  WARNING: failed to move {}: {}", name, e);
                });
                println!("  {} → archive/", name);
            }
        }
    }

    // Move logs too
    for entry in std::fs::read_dir("data/pools").unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name().to_string_lossy().to_string();
        if name.ends_with(".log") {
            let dst = format!("data/deals/archive/{}", name);
            if !std::path::Path::new(&dst).exists() {
                std::fs::rename(entry.path(), &dst).ok();
                println!("  {} → archive/", name);
            }
        }
    }

    println!("\n=== Done ===");
    println!("New layout:");
    println!("  data/deals/base_5M.bin          (5M deals, DD only)");
    println!("  data/deals/scores_dmc_5M.sc     (5M DMC scores)");
    println!("  data/deals/scores_isdd_500k.sc  (500K IS-DD scores, offset 0)");
    println!("  data/deals/archive/             (old files)");
    println!();
    println!("Usage in training:");
    println!("  --pool data/deals/base_5M.bin --scores data/deals/scores_dmc_5M.sc");
    println!("  --pool data/deals/base_5M.bin --scores data/deals/scores_isdd_500k.sc");
}

/// Read real_pts from a COLVDR01 enriched file, returning just the score array.
fn read_real_pts_from_enriched(path: &str) -> Vec<[u8; 4]> {
    let mut f = std::io::BufReader::new(std::fs::File::open(path).expect("Failed to open"));

    let mut magic = [0u8; 8];
    f.read_exact(&mut magic).unwrap();
    assert_eq!(&magic, b"COLVDR01", "Bad magic in {}", path);

    let mut count_buf = [0u8; 8];
    f.read_exact(&mut count_buf).unwrap();
    let count = u64::from_le_bytes(count_buf) as usize;

    let mut scores = Vec::with_capacity(count);
    for _ in 0..count {
        // Skip dealer(1) + hands(16) + dd_pts(4) = 21 bytes
        let mut skip = [0u8; 21];
        f.read_exact(&mut skip).unwrap();

        let mut real_pts = [0u8; 4];
        f.read_exact(&mut real_pts).unwrap();
        scores.push(real_pts);
    }

    scores
}

/// Extract scores from a COLVDR01 enriched pool into a COLVSC01 file.
fn extract_scores_from_enriched(enriched_path: &str, name: &str, offset: usize, output_path: &str) {
    let scores = read_real_pts_from_enriched(enriched_path);
    println!(
        "  Extracted {} scores from {} (layer: '{}')",
        scores.len(),
        enriched_path,
        name
    );
    DealPool::save_scores(name, offset, &scores, output_path)
        .expect("Failed to save scores");
    println!("  Saved {}", output_path);
}
