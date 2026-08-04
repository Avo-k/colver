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

## Un chiffre dans un markdown : récité ou argumenté ?

Le registre dit d'où vient un chiffre. Reste la question d'à côté : **combien de fois a-t-on le
droit de l'écrire ?** Ouverte le 2026-08-02 après avoir propagé un même temps de solve à la main
dans cinq documents, dont un le donnait encore dans une version antérieure de deux ordres de
grandeur.

La règle tient en une distinction, et c'est elle qui décide, pas le type du chiffre :

- **Récité** — le document *reproduit* la mesure sans en tirer de conclusion (un tableau de perf
  recopié dans une vue d'ensemble). **À ne pas dupliquer** : un seul document possède le chiffre,
  les autres en donnent au plus une valeur arrondie d'orientation et **un lien**. Le mode d'échec
  d'un chiffre récité périmé est bénin — il contredit visiblement sa source — mais il est
  fréquent et il use la confiance dans tout le reste.
- **Argumenté** — la phrase *conclut* quelque chose du chiffre (« calibré contre 13,5 ms, donc il
  reste un ordre de grandeur inutilisé » ; « le coût s'effondre de 23 000× quand le budget ne
  baisse que de 8× »). **Le chiffre reste dans la phrase**, et surtout il ne doit **jamais** être
  substitué automatiquement : une mise à jour rendrait la phrase **cohérente et fausse**, ce qui
  est bien pire que périmé. Même famille que `quick_tricks` — quelque chose qui a l'air juste et
  ne l'est plus. Écrire dans la phrase *ce dont elle dépend* (« ce raisonnement porte sur le
  rapport entre les formes, pas sur leur valeur absolue »).

Corollaire sur l'outillage : **on vérifie, on ne génère pas.** Générer les markdown depuis un
magasin impose une étape de build, met des fichiers générés dans git et ne protège pas les
chiffres argumentés, qui sont les dangereux. Le problème n'a jamais été qu'un chiffre soit faux,
c'est que **personne ne s'en aperçoive** — un contrôle qui échoue suffit. Et il devra tolérer
l'incertitude de la mesure (~9 % sur les temps du solveur), ce qui oblige à l'enregistrer :
bénéfice au moins égal à la synchronisation.

État actuel : **la déduplication est faite, le contrôle automatique n'existe pas.** Trois
documents qui bougent deux fois par an ne le justifient pas encore.

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
- `isdd_worlds_per_budget.py` — combien de mondes une recherche IS-DD traverse *vraiment*
  par pli, sous une échéance. `determinizations` n'est pas ce nombre dès que
  `time_limit_ms` est posé, et l'écart va de ×0,1 à ×280 selon le pli. Deux régimes à
  mesurer séparément (`--parallel` ou non) : ils ne donnent pas le même agent.
  ~2 min pour 40 donnes × 2 échéances
- `dd_solver_bench.py` — enveloppe du binaire Rust `bench_dd`. Un binaire ne peut pas appeler
  `runlog` lui-même, donc ce script l'exécute, lit son tableau et journalise le run. Cas
  particulier de provenance : il n'y a aucun modèle à hacher, mais **le résultat dépend des
  drapeaux de compilation autant que du code** — `RUSTFLAGS`, les features et l'horodatage du
  binaire sont donc enregistrés, ainsi que la charge machine (`loadavg`), sans laquelle un
  temps ne veut rien dire ici : un même binaire varie de 20 % selon ce que fait l'autre agent.

- `seat_influence.py` — plan factoriel 2⁴ : la même donne rejouée dans les 16 façons de
  répartir DouDou50 et l'Oracle DD sur les quatre sièges, **enchère figée**. Les deux
  joueurs étant déterministes, les 16 résultats sont exacts et l'effet d'un siège se lit
  en différences appariées, sans bruit d'échantillonnage intra-donne. ~7 min pour 4 000
  donnes sur 8 cœurs, **sans GPU**. Deux points de méthode qui valent d'être réutilisés :
  (1) le run porte son propre **contrôle d'exactitude** — quatre Oracles doivent réaliser
  la valeur DD de la position d'entame, et le taux est affiché à chaque fois ; (2) c'est
  l'un des rares cas où **tirer les donnes au hasard est correct**, une main étant uniforme
  par les règles — c'est le *contrat* qui doit être réaliste, et il sort du bidder v6.
  `--from <json>` re-dépouille un run sans recalculer, ce pour quoi `runlog` existe.
  Deux couples d'occupants, qui ne répondent pas à la même question : `doudou50 → oracle`
  donne l'**enveloppe** (le plus grand écart possible, pour dimensionner un classement),
  `doudou35 → doudou50` donne le **régime réaliste** (deux joueurs imparfaits, comme deux
  humains). Mesurer l'enveloppe seule, c'est risquer de conclure à une échelle où personne
  ne joue.
- `belote_facts.py` — enveloppe de `bench_belote_facts`. Fréquence de la déduction de
  belote sur donnes jouées (~30 s pour 50 000 donnes) et fraction de mondes impossibles
  rendus par playgen (~2 min, sidecar requis, donc le sha256 du modèle est enregistré).
- `belote_regret.py` — dépouille l'A/B `COLVER_NO_BELOTE_FACTS=1` du même binaire. **C'est
  le patron à réutiliser quand l'arène est trop grossière** : au lieu de comparer deux bots
  sur des donnes entières, on compare la *carte choisie* à un juge qui résout la même
  position avec 6,7× plus de mondes. Effet borné à ±0,03 pt DD/décision en 6 minutes, là où
  un h2h de deux heures aurait rendu « non résolu ». Graines distinctes entre un bras et son
  juge, sinon le bras hérite des mondes qui le notent.
- `belote_ab_diff.py` — l'A/B apparié donne par donne (`bench_belote_ab`), gardé comme
  instrument de contrôle en configuration de production. Peu sensible : à ~4 décisions
  contraintes par donne, il faudrait des dizaines de milliers de donnes pour voir ce que le
  précédent borne en 2 000 décisions.

- `bid_contract_ranks.py` — ce que la **boucle d'entraînement du bidder** consulte
  réellement dans la couche de scores : le rang de l'atout contracté, du point de vue du
  camp qui l'a pris. Deux régimes (poids de v6 à ε = 0,02, init aléatoire à ε = 0,30),
  ~5 min chacun. Point de méthode qui coûte cher à re-découvrir : le rang « de la donne »,
  les quatre atouts classés en mélangeant les deux camps, est un **piège** — il rend
  presque uniforme (37,8/23,3/20,6/18,4) là où la bonne lecture donne 58,4/22,7/11,7/7,2.
  Vingt points d'écart au rang 0, et la mauvaise version se lit comme « le bidder
  n'ordonne pas ses couleurs ».
- `taker_position.py` — enveloppe de `bench_taker_position`. L'enchère **synthétique** du
  générateur de couche de scores est-elle dans la distribution des vraies ? Rejoue l'enchère
  de 43 076 donnes, résout les 4 atouts en DD et applique la construction sur la **même**
  donne, donc l'accord se lit apparié. ~2 min sur `isdd_games_v1.bin`, ~6 min sur
  `playgen_games_9M.bin` (dont 4 de lecture : `GameReplay::load_all` charge les 9 M de
  donnes avant d'en examiner 43 000). Deux corpus **par nécessité, pas par prudence** :
  « dans la distribution » se dit par rapport à celui sur lequel playgen a appris. Leur
  accord à 0,5 pp près est le résultat qui rend la référence utilisable — l'enchère est une
  propriété de bid v6, pas du joueur de cartes derrière lui.
  `--bid-model` ajoute les variantes d'enchère candidates et, avec elles, **le témoin le
  plus fort qu'on puisse construire ici** : le même pilote lancé *sans* masque doit
  reproduire les enchères du corpus **à l'identique**, le réseau étant déterministe et
  ayant produit ces donnes. Mesuré à 99,99 %. Sans lui, un historique mal suivi ou un
  score passé du mauvais côté se lirait comme une propriété de la variante — et c'est
  ainsi qu'une variante a été proposée puis réfutée en 7 min de CPU, avant de coûter des
  heures de GPU.
  ⚠️ Ne pas rediriger sa sortie dans `head` : le SIGPIPE tue le processus **avant**
  l'écriture du JSON, et la mesure a l'air d'avoir réussi.

- `prefix_label.py` — enveloppe de `bench_prefix_label` (mesure B) : la même case
  `(donne, atout)` étiquetée par IS-DD sous quatre **préfixes d'enchère** différents, pour
  savoir si le préfixe déplace l'étiquette d'une couche de scores. ~1 h de GPU pour
  2 000 donnes × 5 bras. Trois points de méthode réutilisables :
  (1) **le cinquième bras est le même préfixe avec une autre graine** — deux étiquetages
  IS-DD de la même case ne rendent pas le même nombre, donc sans ce plancher un écart ne
  se distingue pas de deux tirages du même bras ; le digest publie chaque contraste
  **rapporté à ce plancher** (`in_control_sd`), parce qu'un z énorme sur 0,3 point ne
  change rien à une couche ;
  (2) les préfixes ne nomment pas tous le même atout, donc **on ne compare que ce qui
  partage une case** ;
  (3) **96 threads, pas 256** — deux jeux de joueurs par thread font 2 048 clients IS-DD
  contre 64 threads d'accueil du sidecar, et les timeouts en cascade abandonnent des
  donnes qui ne sont pas un tirage au hasard (ce sont celles jouées pendant la saturation).
- `check_score_layer.py` — vérifie une couche de scores **pendant** sa génération, sur le
  fichier partiel. Cinq contrôles, dont deux valent d'être réutilisés ailleurs :
  (1) des **valeurs arithmétiquement impossibles** (163-251 pour des points cartes) sont le
  genre de résidu qui a trahi le bug `quick_tricks` ; les chercher coûte une ligne ;
  (2) le taux d'accord sur le **meilleur atout** est publié **avec son plancher simulé** —
  deux étiquetages du même procédé sont déjà en désaccord ~30 % du temps à σ = 17 par
  étiquette, donc le taux brut ne veut rien dire seul. Mesuré 69,5 % contre un plancher de
  70,7 % : **null sans puissance**, et le script le dit en toutes lettres plutôt que de
  laisser lire « les deux couches sont équivalentes ».
- `bench_capot_prior` (binaire, mesure C) — `P(capot | mes 8 cartes)` par simulation :
  une main, K complétions des 24 cartes restantes, 4 solves chacune. **Sans GPU** : c'est
  le solveur DD qu'on interroge, pas playgen, donc il tourne sur les cœurs libres pendant
  qu'une mesure GPU occupe la carte. Le donneur est **retiré à chaque complétion** — il
  décide qui entame donc il change la valeur DD, et le fixer répondrait à une autre
  question.

- `dd_ab_revs.sh` / `dd_ab_flags.sh` / `dd_ablation.sh` — les harnais A/B du solveur : le
  premier alterne deux **révisions git**, le deuxième trois **cibles de compilation** bâties de
  la même source, le troisième cinq **configurations d'heuristiques** dans un seul binaire.
  Ils ne passent pas par `runlog` (ce sont des scripts shell) mais appliquent la même règle,
  qui est la raison d'être des deux premiers : **alterner et garder le minimum**, jamais A puis
  B. Un binaire inchangé mesuré deux fois varie de 20 % ici. `dd_ablation.sh` s'en dispense
  parce que sa métrique est le **compte de nœuds**, exact et insensible à l'ordonnanceur —
  c'est la sortie qu'offre ce plancher de bruit quand on peut l'obtenir. Résultats à
  journaliser à la main — cf. les entrées `dd_target_cpu`, `dd_ablation` et `dd_tt_size`.

- `bench_tt_size` (binaire) — même remarque, avec une facilité qui vaut d'être connue : il
  mesure les tailles **dans l'ordre demandé**, donc répéter la liste (`--sizes 16,18,16,18`)
  suffit à les entrelacer. Sans ça il produisait ici une conclusion fausse *et* actionnable.

Les autres (`bid_candidates.py`, `bid_capot_probe.py`, `hand_classes.py`,
`card_importance.py`) ne le sont **pas encore**.
