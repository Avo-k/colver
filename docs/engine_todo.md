# Backlog moteur & modèles (non implémenté)

Pendant de [web_todo.md](web_todo.md), côté moteur : règles, données,
entraînement, zoo de modèles, arène. (Dernière mise à jour : 2026-08-01)

Rangé par **dépendance**, pas par thème : le §1 conditionne tout le reste, et
il n'y a rien d'utile à mesurer tant qu'il n'est pas fait. L'effort est en
journées de travail humain, très grossièrement ; le temps machine est indiqué à
part, parce qu'il se lance la nuit et ne coûte pas la même chose.

Un fil relie presque toutes les entrées : **il n'existe nulle part de version de
règles**. Les en-têtes de fichiers portent une version de *format*
(`COLVDD01`, `COLVSC01`, `COLVGM01`, `COLVPG02`), les checkpoints ne portent
rien du tout, et l'invalidation d'un résultat se dit **en prose** — trois
avertissements en tête de [arena_results.md](arena_results.md), quatre encadrés
« BREAKING » dans [CLAUDE.md](../CLAUDE.md). Le web a déjà résolu ce
problème pour lui-même (`ANALYSIS_VERSION`, `PRAGMA user_version`) : une donnée
calculée sous une version périmée se recalcule toute seule. Le moteur n'a pas
d'équivalent, donc la péremption y est une affaire de mémoire humaine.

---

## 1. Geler les règles, puis tout réétalonner dessus

### 1.1 Une version de règles, portée par les fichiers (1 j, préalable à tout)

Le barème a changé trois fois :

| Date | Commit | Changement |
|---|---|---|
| 2026-04-20 | `0a4c5ca` | surcoinche ×3 (était ×4), contré = base + contrat×mult (était 320/640 + contrat×mult), capot = contrat à 250 (était forfait 500/1000/2000) |
| 2026-07-28 | `274723a` | la base est le total cartes de la donne — 162, ou 252 sur capot réalisé (était 160) |
| 2026-07-31 | `e295ee7` | plus d'arrondi : marque au point près (`round10` supprimé) |

[RULES.md](RULES.md) décrit l'état actuel et [scoring.rs](../colver-core/src/engine/scoring.rs)
l'implémente, avec 18 tests. Ce qui manque est le lien entre les deux et un
artefact : rien, dans `bid_nn_final.bin` ou `base_5M.bin`, ne dit sous quel
barème il a été produit.

Le travail :

- Une constante `RULES_VERSION` dans `scoring.rs`, incrémentée à chaque
  changement, avec le changelog ci-dessus juste à côté (il vit aujourd'hui dans
  CLAUDE.md, c'est-à-dire hors du code qu'il décrit).
- L'écrire dans les en-têtes qu'on régénère de toute façon au §1.2
  (`COLVDD01` → `COLVDD02` avec un champ), et dans les checkpoints candle
  (`.safetensors` accepte des métadonnées, `.bin` non — un `.json` à côté
  suffit).
- Le charger et **refuser** un modèle dont la version ne correspond pas, plutôt
  que de jouer avec. C'est ce qui transforme les trois avertissements de
  `arena_results.md` en erreur au démarrage.

**Une décision à écrire noir sur blanc pendant qu'on y est** : la FFB arrondit
la marque à la dizaine (sa §9.2), nous non — c'est le seul écart *connu* avec
les règles officielles, il est assumé (il rend la base 162 visible et aligne le
moteur sur le compteur `/score`), mais il est aujourd'hui une note de bas de
page. S'il doit tenir, il doit être une décision datée avec sa raison.

### 1.2 Régénérer le socle de données (0,5 j humain, ~4 j machine)

Tout `data/deals/` est périmé, pour deux raisons cumulées :

| Fichier | Date | Périmé par |
|---|---|---|
| `base_5M.bin` (105 Mo) | 2026-04-11 | `quick_tricks` (retiré le 2026-07-23) rendait une **valeur DD fausse sur ~25 % des appels** à `solve_for_trump` |
| `scores_isdd_5M.sc` | 2026-04-23 | même solveur, **plus** les deux changements de barème de juillet |
| `scores_dmc_5M.sc` | 2026-04-11 | idem, et produit par DouDou50 qui est lui-même à réentraîner (§1.3) |

C'est la couche la plus profonde de la pile : le bidder s'entraîne dessus, les
labels distillés en dérivent, et l'expérience négative sur les labels
conditionnés à l'enchère notait déjà que sa source était périmée
([auction_conditioned_labels.md:138](bid/experiments/auction_conditioned_labels.md)).

Coûts machine, sur les chiffres du dépôt :

- `gen_pool` : ~244 donnes/s avec `RUSTFLAGS="-C target-cpu=native"` → **~5 h 45
  pour 5 M**, checkpointé toutes les 100 k, sans dépendance CUDA.
- Enrichissement IS-DD : [bid_v5.md](bid/strategies/bid_v5.md) chiffre +0,5 M à
  ~7 h, soit ~14 h/M → **~70 h pour 5 M** (extrapolation, à re-mesurer). C'est
  le poste dominant, et c'est là qu'il faut décider si 5 M sont nécessaires ou
  si 2 M suffisent.
- Enrichissement DMC : GPU-batché, beaucoup moins cher, mais il faut le nouveau
  modèle de jeu — donc **après** le §1.3, pas avant.

Ordre imposé : base DD → réentraînement jeu → couche DMC → réentraînement
annonce. La couche IS-DD peut se lancer en parallèle du réentraînement jeu,
puisqu'elle ne dépend que du solveur.

### 1.3 Réentraîner ce qui dépend du barème (0,5 j humain, ~3-4 j GPU)

Ce qui dépend du barème, et ce qui n'en dépend pas :

- **DouDou50** — `models/dmc_50.bin`, **2026-04-02**, publié dans la release
  `v0.4.0`. Antérieur aux **trois** changements. Sa reward est exactement la
  différence de score de donne ([joint_env.rs:56](../colver-core/src/joint_env.rs#L56)),
  donc il a appris la valeur relative d'une chute, d'un contré et d'un capot
  sous un barème qui n'existe plus. C'est le modèle le plus utilisé du projet :
  jeu par défaut du site, mode rapide, revue de partie, page d'analyse du jeu,
  génération des plis de `/problemes/compter`.
- **Bid v6** — `models/bid_v6_isdd_resume/`, **2026-04-25/26**, release `v0.7.0`.
  Postérieur au premier changement, antérieur aux deux de juillet. Sa reward est
  une Δ-probabilité de victoire calculée sur des scores de match cumulés
  (`match_sim`), donc doublement dépendante du barème : par la valeur de chaque
  donne, et par la vitesse à laquelle on atteint 2000.
- **`win_probability`** — le sigmoïde calibré de
  [bid_obs.rs:25-33](../colver-core/src/bid/bid_obs.rs#L25-L33), « ajusté sur
  10 k matchs », est **une entrée de l'observation** de v5 et v6 (case 110), pas
  seulement une reward d'entraînement. Le réajuster déplace la distribution
  d'entrée : ce n'est pas un correctif, c'est un réentraînement. Le binaire
  existe (`calibrate_winprob`), il faut le relancer sur le nouveau barème
  **avant** de relancer l'entraînement d'annonce.
- **Playgen v2** — le seul gros modèle qui **survit** au changement : le barème
  n'entre pas dans ses tokens, il prédit des cartes et des annonces. Mais son
  corpus a été généré par v6 + DouDou50, donc son postérieur est conditionné sur
  deux politiques qui vont bouger. À régénérer ensuite, pas en urgence.
- **Belief nets** — même statut : ils prédisent des localisations de cartes, pas
  des points. Ils ont en revanche leur propre dette (`belief_v2`/`v3` à
  réentraîner depuis la correction `TrumpCeilingTracker` du 2026-07-21).

Le chemin le plus court est le triforge (`scripts/training/triforge.sh`), qui
alterne annonce et jeu et produit les deux dans une même passe cohérente — à
condition de partir de zéro et non de `--resume-*`, puisque les poids actuels
encodent l'ancien barème (et que `--resume-*` ne recharge de toute façon ni le
compteur de pas, ni l'epsilon, ni le buffer).

### 1.4 Un seul jeu de modèles, publié ensemble (1-2 j)

Aujourd'hui les quatre modèles de production viennent de **quatre releases
différentes** ([_model.py:12-15](../python/colver/_model.py#L12-L15)) :

| Modèle | Release | Date |
|---|---|---|
| `dmc_50.bin` | v0.4.0 | 2026-04-02 |
| `bid_v6_isdd.bin` | v0.7.0 | 2026-04-26 |
| `belief_v4_fix_v2.bin` | v0.7.0 | 2026-04-26 |
| `playgen_v2_final.bin` | v0.8.0 | 2026-07-23 |

Aucun manifeste ne dit que ces quatre-là vont ensemble, et
`download_*_model()` fait un `urlretrieve` **sans vérifier quoi que ce soit** —
ni taille, ni somme de contrôle. La classe de bug que ça laisse ouverte s'est
déjà produite : la prod a servi `playgen_v2_half.bin`, un checkpoint
intermédiaire, pendant une journée entière, pendant que tous les bancs
mesuraient `playgen_v2_final.bin` — détecté à l'œil, pas par le code
([playgen.md](belief/playgen.md)).

Le correctif est un `models/manifest.toml` versionné dans le dépôt : par modèle,
un nom, une URL, un `sha256`, une `RULES_VERSION`, une dimension d'observation.
Consommé par `_model.py`, par l'image Docker, par le sidecar, et par les TOML
d'arène de référence — un seul endroit à changer pour promouvoir un champion, et
une vérification qui échoue bruyamment au lieu de jouer avec le mauvais fichier.

C'est aussi ce qui rend le §1.1 exécutable côté site : le serveur refuse de
démarrer si le manifeste et le moteur ne sont pas sur la même version de règles.

### 1.5 La distribution des donnes est-elle une règle ? (décision, puis 0 ou beaucoup)

[deal_bias.md](deal_bias.md) mesure que la distribution traditionnelle (plis
empilés par camp, une seule coupe, donne 3-3-2) est **fortement** biaisée vers
les mains distributionnelles : coupes +42 %, couleurs de 5+ cartes +68 %,
+8,7 pp de coinches, +4,4 pp de chutes — et le régime s'installe **dès la
première donne** jouée.

Le moteur, lui, distribue uniformément (`GameState::deal_random`), en
entraînement comme sur le site : les deux distributions coïncident, donc rien
n'est cassé aujourd'hui. Mais l'étude note elle-même sa conséquence : « Bid v6
was trained on uniformly dealt pools. Facing a clumpier deal distribution, its
calibration is slightly out of distribution. » Le jour où le site distribue à la
main comme une vraie table — ce qui est un choix produit défendable — tous les
modèles deviennent hors distribution.

C'est donc à trancher **maintenant**, pendant qu'on regénère les pools : si la
réponse est « traditionnel », c'est le pool du §1.2 qu'il faut générer ainsi, et
ça ne coûte rien de plus. Décidé après, ça coûte une deuxième régénération.

---

## 2. Élaguer l'écosystème

Rien ici ne bloque le §1, mais tout y devient plus facile : c'est du bruit qu'on
ne veut pas re-mesurer.

### 2.1 153 bots d'arène, dont 64 checkpoints de balayage (0,5 j)

`arena/bots/` contient **153 TOML**, dont **64** sont des checkpoints d'une même
expérience (`bid_v3_exp_*_1000000.toml` … `_5000000.toml`, `play_v2_*`,
`v4_sa_*`). Ils ont servi une fois, à choisir un point d'arrêt, et ils polluent
`arena list` et tout round-robin lancé sans `--bots`.

Découpage proposé, sans rien supprimer : `arena/bots/` garde les ~10 bots de
référence cités dans [arena_results.md](arena_results.md), le reste va dans
`arena/bots/sweeps/` et `arena/bots/archive/`. `list` ne montre que le premier
niveau.

### 2.2 ~9 Go de checkpoints dans `models/` (0,5 j)

Dominé par `triforge` (2,8 Go), `play_v3_max` (1 Go), `belief_checkpoints`
(392 Mo) et `bumblebid` (386 Mo — une architecture explicitement abandonnée).
Le motif est le même partout : chaque run garde tous ses checkpoints
intermédiaires, en `.bin` **et** en `.safetensors`.

Règle simple à appliquer : un run terminé garde `final` (les deux formats, le
`.safetensors` sert au resume) plus les checkpoints explicitement cités par un
document ; les autres partent. Le §1.3 va en ajouter plusieurs gigaoctets, donc
autant le faire avant.

### 2.3 `matches.csv` : 296 lignes, dont la majorité invalides (1 j)

Trois invalidations successives se superposent — rotation du donneur
(2026-04-20), correction du solveur (2026-07-23), refonte des agents
(2026-07-24, les bots IS-DD ont changé de source de mondes) — et les changements
de barème en ajoutent une quatrième, qui touche directement la colonne
`avg_margin`. CLAUDE.md estime déjà 52 % du fichier périmé avant même juillet.

La péremption est aujourd'hui **de la prose**. Elle devrait être une colonne :
`rules_version` + `engine_rev` (le hash court), écrites par `arena.rs` à
l'enregistrement, et `arena results` qui filtre par défaut sur la version
courante. Les vieilles lignes restent lisibles avec un drapeau, plutôt que d'être
effacées ou de mentir.

À faire **après** le §1.1, dont il consomme la constante — et juste avant la
grande re-mesure, pour qu'elle produise un fichier propre dès la première ligne.

### 2.4 60 binaires, dont on ne sait plus lesquels portent (0,5 j)

`colver-core/src/bin/` compte **60 fichiers**. Certains sont l'outillage courant
(`arena`, `gen_pool`, `train_*`, `playgen_gpu_server`), beaucoup sont des
expériences uniques (`bench_*`, `*_experiment`, `distill_*`, `maxi_diagnose`).
Chacun coûte une compilation à chaque `cargo build --release` du workspace.

Un tableau en tête de [training/overview.md](training/overview.md) — porteur /
banc / archive — suffirait ; le déplacement physique vers un crate `xtask` ou
un dossier `bin/archive` est un second temps facultatif.

Deux d'entre eux ont un défaut connu et non corrigé : `bench_logp_cred.rs` et
`bench_world_compress.rs` **tirent encore leurs positions du flux que les
échantillonneurs consomment** — exactement le biais corrigé dans
`bench_world_cred` le 2026-07-23. Leurs comparaisons entre checkpoints sont donc
invalides ; à corriger ou à marquer archive.

---

## 3. Regroupé depuis les autres documents

Ces listes vivaient éparpillées ; elles restent dans leur document d'origine
(elles y ont leur contexte), mais leur existence se lit ici.

**Playgen** ([belief/playgen.md](belief/playgen.md), « Next steps ») — corpus de
10 M parties + modèle plus gros (d=384 L=6) et outil de fusion COLVGM01 ;
génération de mondes en lockstep batché (K mondes qui avancent ensemble → des
matvec en forme de gemm), le point qui pourrait faire basculer le verdict
temps-de-calcul contre IS-DD ; inférence SIMD ou distillation ; marginales
implicites vs `bid_belief_v4` via `eval_beliefs` ; audit coinche/surcoinche puis
passage complet au classement ; mode *scorer* (mains révélées dans le prompt)
pour une agrégation DD pondérée par importance.

**Annonce v5/v6** ([bid/strategies/bid_v5.md](bid/strategies/bid_v5.md),
« Known limitations ») — l'EMA τ=0,005 est probablement 10 à 100× trop agressive
(essayer 1e-5) ; les features de score v2 n'ont jamais été isolées sans le clip
ni l'EMA ; asymétrie de taille de pool entre v5_isdd (1 M) et v5_max (1,5 M),
qu'un run apparié lèverait — le §1.2 est l'occasion de le faire au bon format.

**Interprétabilité** ([bid/interpretability/probe_morning_report.md](bid/interpretability/probe_morning_report.md),
« Ce qui reste à creuser ») — neurones sans corrélation (probablement des motifs
d'historique d'enchères) ; probe à 3 classes pass/bid/coinche au lieu du binaire
actuel ; refaire la distillation à score non nul (1500-800, 500-1500) pour voir
les neurones qui encodent la pression du score ; SHAP sur `opp_best_other_ts`.

**Croyances** ([belief/README.md](belief/README.md) + mémoire) — `belief_v2` et
`belief_v3` sont à réentraîner depuis la correction des deux fausses exclusions
de `TrumpCeilingTracker` (2026-07-21) ; les h2h historiques du réseau de
croyances sont à refaire depuis que `use_nn_beliefs` le consulte réellement
(même date) — un des rares chiffres du dépôt qui mesurait un modèle **chargé
mais jamais lu**.

**Dérive documentaire** — [data_gen/README.md](data_gen/README.md) renvoie à
`data/pools/`, `pools.md` et `enrichment_methods.md` : aucun des trois n'existe
(le dossier réel est `data/deals/`). À corriger en même temps que le §1.2, qui
réécrit ces formats de toute façon.

---

## 4. Idées, pas encore des tâches

### 4.1 Un étalon unique de force

Le projet mesure la force de trois manières qui ne se parlent pas : l'arène
(% de matchs à 2000 points), les évaluations en cours d'entraînement (points par
donne contre un adversaire fixe), et l'Elo du site (à la donne, humains et bots
mêlés). Le paradoxe v6 est né exactement là : v6 perd 16 à 26 points par donne
contre v5 à toutes les sondes de score, et lui gagne 55-65 % des matchs.

Une seule échelle publiée, avec son intervalle de confiance, rendrait
comparables des choses qui ne le sont pas aujourd'hui — y compris entre un
modèle et un humain, ce que l'Elo du site sait déjà faire pour les bots.

### 4.2 Aucun test Python, et rien qui teste en CI

436 tests Rust, **zéro** test Python. Côté web c'est le §2.5 de
[web_todo.md](web_todo.md) ; côté moteur le trou est ailleurs : la couche PyO3
(`colver-py`) est le point de passage obligé du site vers le moteur, et les
conversions canoniques (`cardset_to_canonical` / `card_to_physical`) y ont un
mode d'échec silencieux — sans elles, le modèle joue des coups légaux au hasard,
sans jamais lever d'erreur. C'est précisément le genre de défaut qu'un test de
non-régression attrape et qu'une partie jouée à la main ne montre pas.

### 4.3 Le sidecar est un point de défaillance unique

`SidecarWorldSource` est la source de mondes **par défaut** de tout agent IS-DD
depuis le 2026-07-24. En `fallback = "uniform"` (ce que fait le site), un
sidecar saturé ou tombé fait retomber tout le monde sur des mondes uniformes —
silencieusement, et donc Dédé devient plus faible sans que rien ne le dise.
C'est le pendant moteur du §2.2 de [web_todo.md](web_todo.md) (« qualité fixe,
latence bornée ») : la dégradation doit sortir par un canal visible.

### 4.4 Deux donnes en base dont les actions ne collent pas aux mains

Détecté le 2026-08-01 par le contrôle du dix de der de `/problemes/compter` :
2 des 17 donnes terminées de la base de dev ont des `actions` incohérentes avec
leurs `hands` (cartes jouées deux fois, d'autres jamais). Cause inconnue.

Ça figure au §4.5 de [web_todo.md](web_todo.md) comme un problème de données,
mais la cause est peut-être ici : **`env.step()` ne valide pas la légalité** —
c'est le contrat assumé d'un moteur RL, et c'est ce qui rend possible d'écrire
une donne impossible. Un `debug_assert` sur la légalité dans `step`, actif dans
les builds de développement du site, coûterait zéro en release et aurait attrapé
ces deux donnes à l'écriture.
