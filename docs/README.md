# Colver Documentation

Belote Contrée engine + RL training stack.

## Top-level

- [ARCHITECTURE.md](ARCHITECTURE.md) — workspace layout, key subsystems, observation formats
- [agents.md](agents.md) — **the `Player` / `WorldSource` layer**: how a bot is built and driven, and the bot-spec format
- [RULES.md](RULES.md) — game rules summary (**as implemented in Colver**)
- [rules-survey/](rules-survey/) — **what the rest of the world actually does**: ~594 rulebooks (federations, tournaments, clubs, apps, open source) compared axis by axis. Start at [rules-survey/SYNTHESE.md](rules-survey/SYNTHESE.md)
- [BENCH.md](BENCH.md) — performance benchmarks
- [arena_results.md](arena_results.md) — **global arena leaderboard** (the main eval metric)
- [classement_et_scoring.md](classement_et_scoring.md) — **l'Elo du web et la fonction d'utilité des bots** : pourquoi c'est le même sujet, et pourquoi Dédé optimise des points cartes là où le barème a une marche
- [engine_todo.md](engine_todo.md) — backlog moteur & modèles (règles, données, entraînement, zoo)
- [idees/](idees/) — **pistes cadrées mais pas lancées** : ce qu'on ferait, ce que ça coûterait, la mesure qui trancherait. Une fiche d'ici a le droit de ne jamais être faite — c'est ce qui la distingue d'un backlog
- [deal_bias.md](deal_bias.md) — traditional gather-cut dealing vs competition shuffling (bias study)
- [Official FFB rules](https://www.ffbelote.org/wp-content/uploads/2015/11/REGLES-DE-LA-BELOTE-CONTREE.pdf) (ffbelote.org) — the reference text. Third-party rulebooks are **not redistributed here**: the raw corpus lives unversioned in `data/rules-corpus/`, retrievable with `docs/rules-survey/_refetch.py`

## Subsystems

- [bid/](bid/) — bidding strategies, bid NN training, reward studies, interpretability
- [play/](play/) — play methods (DD solver, IS-DD, DMC, IS-MCTS), play NN training
- [belief/](belief/) — belief models for hidden card inference
- [data_gen/](data_gen/) — pool generation, enrichment methods, replay formats
- [training/](training/) — training pipelines and how to invoke them
- [superpowers/](superpowers/) — agent-driven plans and design specs

## Web frontend

- [web_todo.md](web_todo.md) — backlog web (non implémenté)
- [web_annonces_next_steps.md](web_annonces_next_steps.md) — Analyse annonce: prochaines étapes
- [web_analyse_jeu.md](web_analyse_jeu.md) — Analyse du jeu de la carte: design + pièges d'implémentation
- [web_compter.md](web_compter.md) — Compter les points: page d'entraînement au comptage en cours de donne

## Quick links by topic

| Topic | Main doc |
|-------|----------|
| Latest bid champion | [bid/reward_studies/v3_max_signal_results.md](bid/reward_studies/v3_max_signal_results.md) |
| Reward signal study | [bid/reward_studies/v3_reward_study.md](bid/reward_studies/v3_reward_study.md) |
| Distilled bid rules (XGBoost) | [bid/interpretability/bid_rules_xgb.md](bid/interpretability/bid_rules_xgb.md) |
| Classer les mains (index canonique + code) | [bid/interpretability/hand_classification.md](bid/interpretability/hand_classification.md) |
| DD solver — timings (**the one table**) | [play/dd_solver.md](play/dd_solver.md#performance) |
| DD solver — measured dead ends, read before optimising | [play/dd_solver_optimization.md](play/dd_solver_optimization.md) |
| Triforge joint training | [play/experiments/triforge.md](play/experiments/triforge.md) |
| Belief net + IS-DD | [belief/README.md](belief/README.md) |
| Playgen world sampler (transformer) | [belief/playgen.md](belief/playgen.md) |
| Building / running a bot | [agents.md](agents.md) |
| Training commands | [training/overview.md](training/overview.md) |
| Auction-conditioned bid labels (**negative** — read before retrying) | [bid/experiments/auction_conditioned_labels.md](bid/experiments/auction_conditioned_labels.md) |
| Un pool DD périmé : ce que ça coûte (**ne pas regénérer pour ça**) | [data_gen/pool_staleness.md](data_gen/pool_staleness.md) |
| Générer des donnes **complètes** jouées par IS-DD (enchère réelle + 32 cartes) | [data_gen/isdd_games.md](data_gen/isdd_games.md) |
| Regénérer la **couche de scores** du bidder avec des mondes playgen | [data_gen/isdd_score_layer_v2.md](data_gen/isdd_score_layer_v2.md) |
| Registre des mesures (provenance + agrégats, versionné) | [measurements/README.md](measurements/README.md) |
| Un corpus de jeu fort pour playgen v3 (**idée cadrée, pas lancée**) | [idees/corpus_isdd_playgen_v3.md](idees/corpus_isdd_playgen_v3.md) |
| Le coût d'un pas de décodage playgen, et le décodage spéculatif (**cadré ; le levier a été pris, le spéculatif reste en attente d'une mesure**) | [idees/decodage_speculatif_playgen.md](idees/decodage_speculatif_playgen.md) |
