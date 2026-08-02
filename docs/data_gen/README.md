# Data Generation Documentation

Pool generation, deal enrichment, replay storage.

## Pools

A "pool" is a pre-computed set of deals + DD points (and optionally "real" play points from a play model). Stored in `data/pools/`.

- pools.md — list of all pools, formats, sources
- [pool_staleness.md](pool_staleness.md) — **quand un pool est-il assez périmé pour justifier une regénération ?** Mesuré : la dérive d'IS-DD ne suffit pas (87 % de l'écart est du bruit d'échantillonnage). Contient la règle « un label DD symétrique ne peut pas voir la force de jeu ».

## Enrichment Methods

Adding "real play" points to a DD pool by simulating the game with a play model.

- enrichment_methods.md — DMC (GPU-batched), IS-DD (CPU rayon), mixed-team variants, speeds, calibration

## Binaries

| Binary | What it does |
|--------|-------------|
| [gen_pool.rs](../../colver-core/src/bin/gen_pool.rs) | Generate fresh DD pool (no CUDA dep) |
| [enrich_pool.rs](../../colver-core/src/bin/enrich_pool.rs) | DMC GPU enrichment (`--sequential --offset N` for matchable deals) |
| [enrich_pool_isdd.rs](../../colver-core/src/bin/enrich_pool_isdd.rs) | IS-DD CPU enrichment |
| [enrich_pool_mixed.rs](../../colver-core/src/bin/enrich_pool_mixed.rs) | Mixed teams (NS=DMC, EW=IS-DD or vice versa) |
| [replay_dmc_vs_isdd.rs](../../colver-core/src/bin/replay_dmc_vs_isdd.rs) | Save full action replays for play comparison |

## Binary formats

- **COLVDD01** (DD only): magic[8] + count[8] + per-deal(dealer[1] + hands[16] + dd_pts[4]) = 21B/deal
- **COLVDR01** (enriched): COLVDD01 + real_pts[4] per deal = 25B/deal
- **COLVGM01** (replay): magic[8] + count[8] + per-game(dealer[1] + hands[16] + num_actions[1] + actions[N])
