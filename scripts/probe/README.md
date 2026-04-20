# Hidden-layer probe infrastructure (bid NN v5)

Sonde la couche cachée du bid NN pour découvrir des features que les 17 features
agrégées du `distill_bid` ne captent pas.

## Quickstart

```bash
# 1. Générer le dataset de probe (720k obs + hand features + NN decision)
#    Temps : ~15 min CPU single-thread.
cargo build --release --bin dump_probe_data
./target/release/dump_probe_data models/bid_v5_isdd/bid_nn_final.bin 80000 /tmp/probe_data.bin

# 2. Extraire les activations des 3 couches cachées sur GPU
#    Temps : ~2s sur 4090.
PYTHONPATH=scripts/probe uv run python scripts/probe/extract_activations.py

# 3. Fit linear probes par couche, identifier top neurones
#    Temps : ~3 min.
PYTHONPATH=scripts/probe uv run python scripts/probe/fit_linear_probes.py

# 4. Caractériser les top neurones (decision tree sur features agrégées)
PYTHONPATH=scripts/probe uv run python scripts/probe/characterize_neurons.py

# 5. Découvrir quelles features engineered expliquent les "mystery" neurones
PYTHONPATH=scripts/probe uv run python scripts/probe/discover_features.py

# 6. Mesurer le gain XGBoost avec les nouvelles features
PYTHONPATH=scripts/probe uv run python scripts/probe/measure_feature_gain.py

# 7. Ablation incrémentale (trouver le set minimal)
PYTHONPATH=scripts/probe uv run python scripts/probe/minimal_feature_set.py

# 8. Investigation opp80 spécifique → trouve `opp_best_other_ts`
PYTHONPATH=scripts/probe uv run python scripts/probe/opp80_investigate.py
```

## Artefacts produits

| Fichier | Contenu |
|---|---|
| `/tmp/probe_data.bin` | Dataset brut : obs + hand_features + nn_action (377 MB) |
| `/tmp/probe_activations.npz` | Activations des 3 couches (599 MB) |
| `/tmp/probe_results.json` | Accuracies par couche par scénario + top neurones |
| `/tmp/probe_neuron_concepts.md` | Carte des neurones top-h2 avec arbres de décision |
| `/tmp/probe_discovered_features.json` | Corrélations neurones ↔ features engineered |
| `/tmp/probe_final_results.json` | XGBoost gain par scénario avec Set B (tous extras) |
| `/tmp/minimal_sets_results.json` | Ablation incrémentale |
| `/tmp/opp80_inv.log` | Investigation opp80 |

Copies persistantes dans [data/probe/](../../data/probe/).

## Reproduire sur un autre modèle

Le pipeline accepte n'importe quel modèle de bid NN (108, 110, ou 113 dim) :

```bash
./target/release/dump_probe_data models/bid_v2/bid_nn_final.bin 80000 /tmp/probe_data_v2.bin
# puis changer le path dans extract_activations.py (OUT_PATH + model path en ligne ~35)
```

## Découvertes principales (2026-04-20)

Voir [docs/bid/interpretability/probe_morning_report.md](../../docs/bid/interpretability/probe_morning_report.md).

Résumé : deux features manquent aux 17 agrégées, qui comptent ensemble pour ~+20pp d'accuracy XGBoost :
1. **Par-couleur J / 9 / count** (12 binaires) → +3-4pp partout hors opp80
2. **`opp_best_other_ts`** (max trump_score excluant la couleur de l'adv) → +18-20pp sur opp80

## Fichiers

- [bid_net_torch.py](bid_net_torch.py) — PyTorch reimplementation du BidNet Rust (matching 5e-6)
- [verify_torch_matches_rust.py](verify_torch_matches_rust.py) — test de conformité
- [extract_activations.py](extract_activations.py) — forward GPU batché
- [fit_linear_probes.py](fit_linear_probes.py) — probes logistiques par couche
- [characterize_neurons.py](characterize_neurons.py) — interprétation neurone par neurone
- [discover_features.py](discover_features.py) — mise en correspondance mystère → candidate
- [measure_feature_gain.py](measure_feature_gain.py) — gain XGBoost final
- [minimal_feature_set.py](minimal_feature_set.py) — ablation
- [opp80_investigate.py](opp80_investigate.py) — deep dive sur le scénario défensif
- [human_rules_v2.py](human_rules_v2.py) — tentative de règles v2 (pas concluant, règles v1 restent)
