# Registre des mesures

`index.jsonl` — une ligne par mesure lancée, **versionnée**. Métadonnées de provenance
et agrégats seulement ; le brut vit à côté, en local.

## Pourquoi

Ouvert le 2026-08-02, après avoir payé ~50 min de GPU pour les trois régimes de
[§1.7 du plan v7](../bid/bid_v7_plan.md) et **n'avoir rien gardé** : les scripts
d'analyse n'écrivaient que sur stdout, et l'affichage tronquait à 25 mains sur 120.
Reposer la moindre question sur ces données — une médiane au lieu d'une moyenne, un
intervalle bootstrap, un sous-ensemble par famille de mains — imposait de tout relancer.

## Où va quoi

| | chemin | versionné | contenu |
|---|---|---|---|
| registre | `docs/measurements/index.jsonl` | **oui** | provenance + agrégats, quelques ko |
| brut | `data/analysis/<script>/<horodatage>__<tag>.json` | non (`data/` est gitignoré) | tout, y compris les écarts monde par monde |

Le registre est sous `docs/` et non sous `data/` pour être versionné sans percer
d'exception dans les règles d'ignore de `data/`. Conséquence assumée : **le brut ne
survit pas à un changement de machine**, seul le registre le fait. C'est le bon
compromis tant que le brut se recalcule ; si une mesure devient irremplaçable, la
sortir de `data/`.

## Provenance

Chaque entrée porte le SHA du dépôt **et s'il était sale**, l'`argv`, la graine, et un
sha256 de chaque fichier de poids consulté. Ce dernier point est le plus important : les
`.bin` ne sont pas dans git et changent sans prévenir, donc un écart de points ne veut
rien dire sans l'empreinte du modèle qui l'a produit. Une mesure faite sur un arbre sale
n'est pas reproductible, et l'entrée le dit.

## Usage

Les scripts instrumentés écrivent tout seuls. `--tag` nomme le run, `--no-log` désactive
pour un essai jetable.

```bash
uv run python scripts/analysis/bid_equivariance.py --deals 400 --prior "100C"
uv run python scripts/analysis/bid_q_flatness.py --hands 120 --worlds 300 --prior "100C P"
```

Relire sans recalculer :

```bash
python3 -c "import json;[print(json.loads(l)['tag'], json.loads(l)['summary']) for l in open('docs/measurements/index.jsonl')]"
```

## Entrées `INCOMPLET`

Les trois `bid_q_flatness` du 2026-08-02 sont **reconstruites depuis stdout**, le run
ayant précédé l'instrumentation : agrégats justes, mais ni brut monde par monde ni les
95 lignes de détail tronquées à l'affichage. Elles sont là pour que les chiffres cités
en §1.7 aient une trace ; les relancer (~50 min, sidecar requis) les remplacerait par
des entrées complètes.

## Scripts instrumentés

- `bid_equivariance.py` — 0,8 s pour 400 donnes × 23 permutations, donc utilisable comme
  test de non-régression à chaque checkpoint
- `bid_q_flatness.py` — ~13-25 min par régime, c'est celui qui motivait tout ceci
- `dd_solver_bench.py` — enveloppe du binaire Rust `bench_dd`. Un binaire ne peut pas appeler
  `runlog` lui-même, donc ce script l'exécute, lit son tableau et journalise le run. Cas
  particulier de provenance : il n'y a aucun modèle à hacher, mais **le résultat dépend des
  drapeaux de compilation autant que du code** — `RUSTFLAGS`, les features et l'horodatage du
  binaire sont donc enregistrés, ainsi que la charge machine (`loadavg`), sans laquelle un
  temps ne veut rien dire ici : un même binaire varie de 20 % selon ce que fait l'autre agent.

Les autres (`bid_candidates.py`, `bid_capot_probe.py`, `hand_classes.py`,
`card_importance.py`) ne le sont **pas encore**.
