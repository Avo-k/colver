# Data Generation Documentation

Pool generation, deal enrichment, replay storage.

## Pools

A "pool" is a pre-computed set of deals + DD points (and optionally "real" play points from a play model). Stored in `data/pools/`.

- pools.md — list of all pools, formats, sources
- [pool_staleness.md](pool_staleness.md) — **quand un pool est-il assez périmé pour justifier une regénération ?** Mesuré : la dérive d'IS-DD ne suffit pas (87 % de l'écart est du bruit d'échantillonnage). Contient la règle « un label DD symétrique ne peut pas voir la force de jeu ».

## Couches de scores

Une *couche* (`COLVSC01`) est un `[u8; 4]` par donne — les points cartes N-S sous
chaque atout, sous jeu fort. C'est l'entrée de la reward du bidder.

- [isdd_score_layer_v2.md](isdd_score_layer_v2.md) — **plan de regénération** avec des
  mondes playgen au lieu d'uniformes. Contient les deux arguments qui portent la forme
  du fichier (`CardPoints` rend la factorisation par l'atout exacte ; `dd_pts` détermine
  le camp preneur, donc `[u8;8]` au prix de `[u8;4]`), le budget gradué par rang
  d'atout (−31 %, dérivé de ce que la boucle consulte réellement), les quatre
  invariants sur le donneur, et les trois mesures à faire avant d'engager les heures.
  La **mesure A est faite** (§4) : l'enchère synthétique est hors distribution sur la
  **forme** du préfixe — une seule annonce là où le réel en compte 2 à 4 dans 69 % des
  cas, jamais contestée là où le réel l'est à 81 %, jamais coinchée là où le réel l'est
  à 26 % — et non sur l'identité du preneur, dont le camp est bon à 89,4 %.

## Donnes complètes

Une *donne complète* garde la trajectoire — l'enchère telle qu'elle s'est jouée
et les 32 cartes dans l'ordre — là où un pool ne garde qu'une étiquette par
donne. C'est ce qu'il faut pour entraîner un playgen sur du jeu fort.

- [isdd_games.md](isdd_games.md) — `gen_games_isdd` → `COLVGM01`. Contient le
  profil (**93 % du temps est de l'attente du sidecar playgen, 7 % du solve
  DD**), les optimisations qui valent 2,5×, et trois impasses mesurées : TF32
  (3-5× plus lent), la fenêtre d'attention rétrécie (candle refuse les matmuls
  non contigus), et le calendrier de mondes **montant** (plus lent à total égal).

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
