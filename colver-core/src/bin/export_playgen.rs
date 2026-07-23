//! Export a playgen safetensors checkpoint to flat-f32 weights for pure-Rust
//! inference (`playgen::infer::PlaygenModel`). V1 → COLVPG01; `--v2` (bid
//! head, physical suits, 122-token positions) → COLVPG02 with the bid head
//! appended after the card head.
//!
//! Usage:
//!   cargo run -p colver-core --bin export_playgen --features dmc_train --release -- \
//!     models/playgen/playgen_final.safetensors models/playgen/playgen_final.bin \
//!     --d-model 256 --layers 4 --heads 8 [--v2]

use candle_core::Device;
use colver_core::playgen::model::{PlaygenConfig, PlaygenTrainer};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: export_playgen <in.safetensors> <out.bin> [--d-model N --layers N --heads N --v2]");
        std::process::exit(1);
    }
    let input = &args[1];
    let output = &args[2];
    let mut d_model = 256usize;
    let mut n_layers = 4usize;
    let mut n_heads = 8usize;
    let mut v2 = false;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--d-model" => { d_model = args[i + 1].parse().unwrap(); i += 2; }
            "--layers" => { n_layers = args[i + 1].parse().unwrap(); i += 2; }
            "--heads" => { n_heads = args[i + 1].parse().unwrap(); i += 2; }
            "--v2" => { v2 = true; i += 1; }
            other => { eprintln!("unknown arg {}", other); std::process::exit(1); }
        }
    }

    let cfg = if v2 {
        PlaygenConfig::v2(d_model, n_layers, n_heads)
    } else {
        PlaygenConfig::v1(d_model, n_layers, n_heads)
    };
    let mut trainer = PlaygenTrainer::with_config(cfg, 1e-4, 0.0, Device::Cpu)
        .expect("trainer init");
    trainer.load_checkpoint(input).expect("load checkpoint");

    let data = trainer.varmap.data().lock().unwrap();
    let get = |name: &str| -> Vec<f32> {
        let t = data
            .get(name)
            .unwrap_or_else(|| panic!("missing tensor {}", name))
            .as_tensor();
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    };

    let mut floats: Vec<f32> = Vec::new();
    floats.extend(get("primary_emb.weight"));
    floats.extend(get("suit_emb.weight"));
    floats.extend(get("actor_emb.weight"));
    floats.extend(get("seg_emb.weight"));
    floats.extend(get("pos_emb.weight"));
    for l in 0..n_layers {
        floats.extend(get(&format!("layers.{}.attn_norm.weight", l)));
        floats.extend(get(&format!("layers.{}.qkv_proj.weight", l)));
        floats.extend(get(&format!("layers.{}.qkv_proj.bias", l)));
        floats.extend(get(&format!("layers.{}.out_proj.weight", l)));
        floats.extend(get(&format!("layers.{}.out_proj.bias", l)));
        floats.extend(get(&format!("layers.{}.ffn_norm.weight", l)));
        floats.extend(get(&format!("layers.{}.ffn.w_gate.weight", l)));
        floats.extend(get(&format!("layers.{}.ffn.w_gate.bias", l)));
        floats.extend(get(&format!("layers.{}.ffn.w_up.weight", l)));
        floats.extend(get(&format!("layers.{}.ffn.w_up.bias", l)));
        floats.extend(get(&format!("layers.{}.ffn.w_down.weight", l)));
        floats.extend(get(&format!("layers.{}.ffn.w_down.bias", l)));
    }
    floats.extend(get("out_norm.weight"));
    floats.extend(get("head.weight"));
    floats.extend(get("head.bias"));
    if v2 {
        floats.extend(get("bid_head.weight"));
        floats.extend(get("bid_head.bias"));
    }

    let mut bytes: Vec<u8> = Vec::with_capacity(20 + floats.len() * 4);
    bytes.extend_from_slice(if v2 { b"COLVPG02" } else { b"COLVPG01" });
    bytes.extend_from_slice(&(d_model as u32).to_le_bytes());
    bytes.extend_from_slice(&(n_layers as u32).to_le_bytes());
    bytes.extend_from_slice(&(n_heads as u32).to_le_bytes());
    for f in &floats {
        bytes.extend_from_slice(&f.to_le_bytes());
    }
    std::fs::write(output, &bytes).expect("write output");
    println!(
        "Exported {} ({} params, {:.1} MB) -> {}",
        input,
        floats.len(),
        bytes.len() as f64 / 1e6,
        output
    );
}
