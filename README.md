<p align="center">
  <img src="images/colver.png" alt="Logo Colver" width="200">
</p>

<p align="center">
  <a href="https://colver.net/"><img src="https://img.shields.io/badge/jouer-colver.net-2ea44f?logo=firefoxbrowser&logoColor=white" alt="Jouer sur colver.net"></a>
  <a href="https://pypi.org/project/colver/"><img src="https://img.shields.io/pypi/v/colver?logo=pypi&logoColor=white&label=PyPI&color=blue" alt="Version PyPI"></a>
  <a href="https://huggingface.co/collections/Avo-k/colver-belote-contree-6a71df4a723e6734fe623a65"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20mod%C3%A8les-Hugging%20Face-FFD21E" alt="Modèles sur Hugging Face"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/licence-MIT-lightgrey" alt="Licence MIT"></a>
</p>

# Colver

**[Read in English](README.en.md)**

Environnement de Belote Contrée rapide pour l'apprentissage par renforcement. Moteur Rust avec bindings Python.

**Jouer en ligne : [colver.net](https://colver.net)** — un site public auto-hébergé : parties solo ou multijoueur, comptes, classement et analyse.

## Démarrage rapide

```bash
pip install colver
```

```python
import colver

env = colver.Env.deal(dealer=1, seed=35)   # donne reproductible ; Sud ouvre les enchères
# Sud tient   ♠ —   ♥ 10 J   ♦ K Q J 9   ♣ K 8

# Qu'est-ce que Sud annonce ?
env.load_bid_model(colver.download_bid_model())   # récupéré depuis le Hub au premier appel
print(colver.Env.action_name(env.action_bid_nn()["best_action"], 0))
# 120♦

# Et qu'est-ce que Sud entame ? Le modèle déroule l'enchère aux quatre sièges ;
# personne ne surenchérit, Sud joue donc 120♦ pour Nord-Sud.
while env.phase() == 0:
    env.step(env.action_bid_nn()["best_action"])

env.load_dmc_model(colver.download_model())
print(colver.Env.action_name(env.action_dmc_with_stats()["best_action"], 1))
# J♦ — le maître atout, en 0,6 ms sur CPU
```

Les deux appels rendent aussi `q_values`, le classement complet du modèle, déjà restreint
aux actions légales.

Les poids vivent sur le [Hub Hugging Face](https://huggingface.co/collections/Avo-k/colver-belote-contree-6a71df4a723e6734fe623a65)
— un dépôt par modèle, chacun avec une fiche qui dit ce qu'il fait, ce qu'il vaut et ce
qu'il ne sait pas faire. `download_*()` les met en cache dans `~/.cache/colver/models/`
(GitHub Releases reste en repli) ; `COLVER_MODEL_PATH` et consorts permettent de pointer
vers ses propres fichiers.

## Caractéristiques

- **~1.4M rollouts/sec** en mono-thread (phase de jeu), ~895K rollouts/sec sur une donne complète
- **État de jeu `Copy` de <=96 octets** pour un clonage MCTS performant
- **Six agents IA** — réseau Q DMC, IS-DD, oracle DD, Smart/Naive IS-MCTS, et heuristique
- **Enchères par réseau de neurones** — "Bid V6 IS-DD", un Dueling DQN score-aware et belote-aware (117→512³→43) entraîné 75M étapes sur points réels IS-DD avec simulation de match complète
- **Interprétabilité ML** — distillation XGBoost + sondage de couche cachée révèlent le système de scoring implicite du NN, traduit en règles utilisables par un humain (88-94% d'accord)
- **Modèle de mondes Playgen** — un transformer causal qui prolonge une donne à partir de ce qu'un seul siège voit ; le dérouler révèle les mains cachées, et c'est de là que viennent désormais les mondes échantillonnés par IS-DD
- **Une seule couche d'agents** — un bot est une spec TOML (`[bid]` / `[play]` / `[worlds]`), construite côté Rust et pilotée à l'identique par l'arène, le web et `colver.Agent`
- **Interface web** — jouez en solo ou en salon multijoueur, observez, rejouez avec analyse carte par carte, entraînez-vous sur des problèmes (FastAPI + WebSocket)
- **Bindings Python** via PyO3 — `Env`, `Agent`, `Analyst` et `Beliefs` avec stubs de types complets, installable depuis PyPI
- Zéro dépendances dans le cœur (seulement `rand` derrière un feature flag)

## Interface Web

Jouez contre les agents IA directement dans le navigateur sur **[colver.net](https://colver.net)**.

Ce qui est publié — ce dépôt et le paquet `colver` — est le **moteur** : les règles, la recherche, les agents et les liaisons Python. C'est ce qu'il faut pour vérifier qu'une donne n'est pas truquée et que les bots jouent ce qu'ils prétendent, et c'est à ça que sert de le publier. Le site, lui, est le produit : il n'est pas distribué.

Quatre destinations : **Jouer**, **Analyser**, **Apprendre**, **Classement**. Le compte est facultatif — c'est lui qui rattache les parties à un joueur, rend les parties reprenables et les fait classer.

### Jouer

**Humain vs IA** — Jouez en Sud contre trois bots. Exactement deux réglages, tous deux choisis avant la donne :

- **Tempo** — `Standard` (Dédé, ~40 s la donne) ou `Rapide` (DouDou50, ~15 s). Le couplage est voulu : une recherche IS-DD coûte du temps réel par coup, donc un tempo rapide n'est honnête que derrière un bot qui répond instantanément. Les quatre sièges IA jouent le même bot — une table où le partenaire est plus faible que les adversaires ne dit rien de la façon dont vous avez joué.
- **Format** — une donne sèche (défaut), ou une partie en 1000 / 2000 points. Le score cumulé part aux bots, et Bid V6 le lit : il n'annonce pas la même chose à 900-200 qu'à 0-0.

Vos cartes sont jouées instantanément au clic ; la pause appartient à la position *précédant* un coup, et le bot réfléchit dedans, pas par-dessus. Le passe forcé est joué pour vous, le dernier pli — où plus personne n'a de choix — se déroule tout seul, et la dernière levée reste 2 s à l'écran avant le panneau de fin. Connecté, on peut quitter une partie et la reprendre plus tard.

![Onglet Jouer](images/screenshots/tab-play.png)

**Salon multijoueur** — Des salons à code de 4 caractères ; les bots occupent les sièges que les humains ne prennent pas. L'hôte choisit le tempo et le format. Chaque état diffusé est filtré sur la main du destinataire et pivoté pour qu'il soit toujours assis en Sud. Les sièges sont liés aux comptes : un joueur déconnecté retrouve le sien.

![Salon](images/screenshots/tab-salon.png)

**IA vs IA** — Observez des parties IA contre IA avec toutes les mains visibles. Assignez un agent différent à chacune des 4 places. Avancez action par action, jouez des plis entiers, ou utilisez la lecture automatique. Le panneau de stats affiche les Q-values, scores DD ou évaluations de main. Collez une chaîne CFN pour charger une position spécifique.

![Onglet Regarder](images/screenshots/tab-watch.png)

### Analyser

**Rejouer** — Parcourez et rejouez les donnes passées (jouées, observées ou partagées), coup par coup. Deux passes d'analyse indépendantes s'y ajoutent, chacune mise en cache et chargée séparément pour que la lente ne bloque jamais la rapide :

- la **passe DD** — le coût exact de chaque carte, plus une revue d'enchère portant deux avis par annonce : les Q-values de Bid V6 et la tête d'enchère 43-voies de playgen ;
- la **revue d'agents** — ce que DouDou50, l'Oracle et Dédé auraient joué à chaque carte non forcée, envoyée au fil du calcul.

Chaque carte et chaque annonce porte un lien vers sa page d'analyse, et le chemin du retour.

![Rejouer](images/screenshots/tab-replay.png)

**Annonces** — Composez une main de 8 cartes, posez les enchères qui ont précédé votre tour, et voyez ce que *Bid V6 IS-DD* annoncerait — les Q-values de chaque action légale, plus un panneau "Facteurs clés" distillé par XGBoost. Deux tableaux jouent ensuite la main sur des centaines de distributions des 24 autres cartes : **Jeu parfait** (chaque donne résolue en double-dummy par l'Oracle — un plafond théorique) et **Jeu réel** (l'enchère complète menée par le NN aux quatre places, puis 8 plis joués par DouDou50 — ce qui arrive vraiment). Un onglet par annonce analysée, donc deux annonces sur la même main se comparent côte à côte ; les mains analysées restent dans une barre latérale. Tourne côté serveur ou entièrement dans le navigateur via WASM ("Calcul local").

![Onglet Annonces](images/screenshots/tab-annonces.png)

**Jeu de la carte** — Une ligne par carte jouable à une position donnée, pour répondre à deux questions qu'il ne faut pas confondre : *les mondes de l'information set* (des donnes compatibles avec ce que le siège pouvait savoir, chacune résolue en double-dummy — c'est là qu'on juge la décision) et *le vrai monde* (un seul solve exact, sur la donne telle qu'elle était). Un troisième bloc force la carte et laisse DouDou50 finir la donne. Une carte deuxième dans la vraie donne mais meilleure dans 70 % des mondes était un bon coup contre de la malchance.

![Jeu de la carte](images/screenshots/tab-analyse-jeu.png)

**Croyances** — Regardez *playgen* localiser les cartes cachées au fil d'une donne, pendant l'enchère comme pendant le jeu. Avancez pas à pas et lisez les barres de probabilité par carte face à la vérité terrain, depuis chacun des quatre sièges.

![Onglet Croyances](images/screenshots/tab-croyances.png)

**Problèmes d'annonce** — Problèmes d'enchères. Voyez une main et l'historique des enchères, puis trouvez la bonne annonce. L'IA évalue votre réponse.

**Problèmes de jeu** — Problèmes de jeu de la carte. Voyez une position en cours de partie et trouvez la meilleure carte. Comparez votre choix au jeu optimal du solveur DD.

### Apprendre

**Aide-mémoire** — Aide-mémoire visuel : ordre de force et valeur des cartes (atout / non-atout), points de la donne, règles d'enchères.

**Guide des annonces** — Guide de stratégie visuel dérivé du bot par ML : pondération par carte, règles de décision par position, règle du miroir en défense. 88-94% d'accord avec le NN, mémorisable en quelques minutes.

![Onglet Annoncer](images/screenshots/tab-annoncer.png)

**Marquer les points** — Compteur de points pour vos vraies parties. Choisissez un score cible (1000-3000), ajoutez des manches avec un formulaire qui calcule automatiquement les scores exacts aux règles FFB (contrat, multiplicateur coinche/surcoinche, points faits, belote), et suivez la probabilité de victoire mise à jour après chaque manche. Tout vit dans le navigateur — pas de serveur, pas de compte.

![Onglet Marquer](images/screenshots/tab-score.png)

### Classement

Un Elo par entité classée — un compte humain ou un type de bot — mis à jour donne par donne dès que les quatre sièges sont identifiables. Les bots ont un K plus faible : ce sont des points de repère, et ils occupent jusqu'à trois sièges par partie.

## Compilation et exécution

Nécessite Rust stable (édition 2021) et Python 3.10+.

```bash
# Tests du moteur (382 passés, 52 ignorés)
cargo test -p colver-core

# Benchmark de performance
cargo run -p colver-core --bin bench --release

# Démo MCTS vs aléatoire
cargo run -p colver-core --bin mcts_demo --release -- 100

# Démo Smart IS-MCTS vs aléatoire + vs naïf
cargo run -p colver-core --bin smart_ismcts_demo --release -- 100

# Bindings Python (via uv)
uv sync
uv run python3 -c "import colver; env = colver.Env(); print(env.reset())"

# Bot contre bot, 200 matchs dans chaque sens (un bot est un TOML dans arena/bots/)
cargo run --bin arena --release -- h2h web_dede web_doudou --matches 200

# Entraînement conjoint enchère + jeu (GPU, candle)
cargo run -p colver-core --bin train_joint --features dmc_train --release -- --num-envs 256 --steps 35000000

# Sidecar GPU playgen — la source de mondes d'IS-DD
cargo run -p colver-core --bin playgen_gpu_server --features gpu_server --release -- --playgen models/playgen/playgen_v2_final.bin --port 8003
export COLVER_PLAYGEN_GPU_URL=http://localhost:8003
```

Sans `$COLVER_PLAYGEN_GPU_URL`, les bots IS-DD retombent sur des mondes uniformes sous contraintes (le web l'annonce dans les stats de la décision ; l'arène, elle, refuse de construire le bot si la spec ne le demande pas explicitement).

## Agents IA

### Oracle — Solveur DD (`solver.rs`)

Solveur double-dummy en information parfaite qui voit les 4 mains — il *triche*. Alpha-beta avec tables de transposition, PVS, killer moves et élagage par équivalence de cartes. Calcule la carte optimale exacte en ~24 ms sur donne complète, ~150 us en mi-partie et ~1,4 us en finale (mesure du 2026-08-03, 1 thread, sur un corpus figé ; la dispersion de mesure est de ~9 %, ce qui dit combien de chiffres ont un sens). Utile comme borne supérieure. Le banc qui les produit est dans le dépôt : `cargo build --release --features "parallel solver_stats" --bin bench_dd`.

### Dédé — IS-DD (`is_dd.rs`)

Recherche Information Set Double-Dummy : échantillonner des donnes compatibles avec ce que ce siège peut savoir, résoudre chacune exactement avec le solveur alpha-beta DD, agréger. Les contraintes dures (coupes révélées par le jeu, plafond d'atout, cartes déjà tombées) sont des faits et s'appliquent toujours. Les mondes viennent d'une `WorldSource` que l'agent possède — **playgen via le sidecar GPU par défaut**, avec un échantillonnage uniforme sous contraintes en repli. Un **réseau de croyances** peut en plus pondérer le tirage. IS-DD se prononce « is Dédé » — d'où le surnom.

### DouDou50 — Réseau Q DMC (`dmc_net.rs`)

Agent par apprentissage par renforcement de style [DouZero](https://arxiv.org/abs/2106.06135). Un réseau Q choisit les cartes à jouer en une seule passe forward — **sans arbre de recherche**. Modèle de jeu par défaut, entraîné sur 50M étapes avec l'enchérisseur NN gelé (phase play-only du triforge).

**Architecture** : ResNet Dueling DQN 411→1024→1024→1024→32 avec LayerNorm et skip connections (~2.6M paramètres). Utilise un encodage canonique des couleurs (pas d'augmentation nécessaire). Inférence en Rust pur (~1ms/décision, pas de PyTorch nécessaire). Agent le plus fort dans l'ensemble.

L'ancien modèle **DouDou35** (415→1024³→32, obs legacy, 35M étapes) reste supporté. *DouDou* = en référence à DouZero.

### Playgen — modèle de mondes (`playgen/`)

Un transformer causal (10,7M paramètres) entraîné à prolonger une donne de manière autorégressive à partir du préfixe visible par un seul observateur. Le dérouler révèle les mains cachées : un déroulement *est* donc un monde déterminisé tiré d'une postérieure apprise, et non d'un mélange uniforme — c'est ce qu'IS-DD échantillonne, et ce que la page Croyances affiche. La v2 porte en plus une tête d'enchère 43-voies, qui rend possible l'échantillonnage en cours d'enchère (et, par curiosité, l'usage de playgen comme enchérisseur). Il tourne sur CPU ou CUDA ; en production un sidecar (`playgen_gpu_server`) le sert en HTTP, ~50x plus vite que sur CPU.

### Anciens agents de recherche

**Smart IS-MCTS** (`smart_ismcts.rs`) — [IS-MCTS](https://doi.org/10.1109/TCIAIG.2012.2200894) pondéré par croyances heuristiques. **Naive IS-MCTS** (`naive_ismcts.rs`) — Déterminisation par ensemble sans croyances. Les deux sont configurables par le format de bot décrit dans [docs/agents.md](docs/agents.md).

### Bid V6 IS-DD — Enchérisseur NN (`bid_net.rs`)

Dueling DQN **117→512→512→512→43** avec observation score-aware v3 (features de score de match + 4 bits de belote). Entraîné **75M étapes sur points réels IS-DD** (pas DD oracle) avec reward belote-aware et **simulation de match** complète (scores cumulés, rotation du donneur, reset à 2000). Bat le champion précédent Bid V5 dans les deux jeux d'évaluation (55.8% en jeu DMC, 57.3% / +181 pts en jeu IS-DD). `BidNet::load` détecte automatiquement la taille cachée et obs_dim (108 / 110 / 113 / 117).

Versions précédentes toujours supportées via auto-détection :
- **Bid V5 IS-DD** — obs score-aware 113-dim, 25M étapes sur points réels IS-DD
- **Bid V3 Max** — 108-dim, entraîné sur `max(DMC, IS-DD)` points réels (20M étapes)
- **Bid à Dédé** (v2) — 108-dim, reward DD oracle
- **Bid à Doudou** (v1) — 114→256² dueling, self-play DouZero

**Interprétabilité** : distillation XGBoost et sondage linéaire sur la couche cachée révèlent que le scoring implicite du NN diffère nettement de l'évaluation classique (ex. V atout = +11 effectif, 9 = +4, A atout = +1, A latéral = 0 net ; plus une anti-synergie V×9 = −2). Traduit en un arbre de décision à 5 features atteignant 88-94% d'accord avec le NN. L'espace des mains est **énumérable** — 472 579 mains distinctes à 8 cartes, indexées bijectivement — ce qui fait de la politique d'ouverture d'un enchérisseur une table finie plutôt qu'une boîte noire : [docs/bid/interpretability/hand_classification.md](docs/bid/interpretability/hand_classification.md).

## Comparaison des agents

| Agent | Type | Vitesse/coup | Notes |
|---|---|---|---|
| Oracle (DD) | Solveur DD (triche) | 24 ms -> 1,4 us (pli 1 -> pli 8) | Borne supérieure |
| Dédé (IS-DD) | Solveur DD sur mondes échantillonnés | budget (1 s sur le web) | Plus fort avec recherche |
| **DouDou50** | **Réseau Q (ResNet)** | **<1ms** | Plus fort, sans recherche |
| Smart IS-MCTS | Recherche + croyances | ~9ms | Budget configurable |
| Naive IS-MCTS | Recherche | ~8ms | Budget configurable |

**Note** : Les agents à base de recherche voient leur force augmenter avec le budget de temps. L'agent DMC n'utilise aucune recherche — la décision est prise en une seule inférence.

## Un bot est une spec, pas un chemin de code

Un bot est un fichier TOML décrivant un enchérisseur, un joueur de cartes et (pour IS-DD) une source de mondes. L'arène, le frontend web et `colver.Agent` lisent la même spec et obtiennent le même agent — il n'existe aucune seconde implémentation.

```toml
[bid]
strategy = "nn"                    # heuristic|improved|smart|maxi|nn|playgen|…
model = "models/bid_v6_isdd_resume/bid_nn_final.bin"

[play]
method = "isdd"                    # isdd|dmc|dmc_then_isdd|ismcts|smart_ismcts|oracle_dd|heuristic
time_ms = 1000

[worlds]                           # IS-DD uniquement ; sidecar par défaut
source = "sidecar"
url = "http://localhost:8003"
```

Tester une nouvelle combinaison, c'est écrire un fichier, pas recompiler. `arena/bots/`
en contient trois : `web_dede` et `web_doudou` sont **exactement** les deux bots que
colver.net fait jouer (modes standard et rapide), `heuristic_baseline` n'a besoin d'aucun
poids. Les trois ne dépendent que des modèles publiés sur le Hub. Le zoo complet et les
résultats accumulés restent privés — ce sont nos mesures, pas une API.

## Architecture

**Workspace :** `colver-core` (Rust pur) + `colver-py` (PyO3/NumPy FFI) + `colver-wasm` (solveur dans le navigateur) + `python/colver/web` (FastAPI/WebSocket)

### Représentation des cartes

Système de bitmask : `Card = u8` (0-31), `CardSet = u32` (bitmask). Disposition : Pique\[0-7\], Cœur\[8-15\], Carreau\[16-23\], Trèfle\[24-31\]. Dans chaque couleur : 7, 8, 9, V, D, R, 10, A (ordre de force hors atout). Force à l'atout : V > 9 > A > 10 > R > D > 8 > 7.

### État de jeu

`GameState` est `Copy` et fait <=96 octets (vérifié à la compilation) pour un clonage MCTS rapide. Contient les mains, le pli courant, le contrat, les points/plis par équipe, l'état des enchères, le bitmask des cartes jouées, le suivi des coupes et de la belote.

### Encodage des actions

| Phase | Actions | Encodage |
|---|---|---|
| Enchères | 43 total | 0=PASSE, 1-36=enchères (9 valeurs x 4 couleurs), 37-40=capot x 4, 41=COINCHE, 42=SURCOINCHE |
| Jeu | 32 total | Index de carte 0-31 directement |

### Déroulement

Enchères → Jeu → Fin. Les enchères se terminent après 3 passes consécutives, une surcoinche, ou 4 passes (donne nulle). Le jeu comporte 8 plis de 4 cartes. Total des points de carte = 152 ; avec dix de der = 162 (normal) ou 252 (capot).

## API Python

```python
import colver

print(colver.__version__)  # "0.10.0"

# Environnement unique
env = colver.Env()
obs, legal_actions = env.reset()
obs, reward, done, legal_actions = env.step(action)

env.current_player()       # 0-3
env.phase()                # 0=Enchères, 1=Jeu, 2=Fin
env.legal_action_mask()    # tableau numpy (43,)
env.rewards()              # [score_NS, score_EO]
env.bid_improved()         # action d'enchère improved_bid
env.deal_outcome()         # [résultat_NS, résultat_EO] binaire
env.get_observation()      # vecteur d'observation de 415 flottants
env.action_naive_ismcts(20)  # action IS-MCTS naïf (20ms)
env.action_smart_ismcts(20)  # action IS-MCTS intelligent (20ms)

# Réseau Q DMC (si les poids du modèle sont téléchargés)
model = colver.model_path()  # ~/.cache/colver/models/dmc_final.bin
if model:
    env.load_dmc_model(str(model))
    result = env.action_dmc_with_stats()  # {"best_action": 5, "q_values": [...]}

# Un bot complet à partir d'une spec — le même TOML que lit l'arène
agent = colver.Agent(spec_toml, seat=2)
agent.init_deal(env)
agent.observe(env, action)       # toutes les actions, y compris les vôtres
decision = agent.decide(env)     # {"action": …, plus les stats de la méthode}

# Playgen comme analyste : marginales des cartes cachées, mondes, politique d'enchère
analyst = colver.Analyst("models/playgen/playgen_v2_final.bin")
analyst.init_deal(env, observer=2)
probs = analyst.marginals(env, n_worlds=50)
```

## Performance

| Charge | Débit | Latence |
|---|---|---|
| Rollout phase de jeu | 1.4M/sec | ~720 ns |
| Rollout donne complète | 895K/sec | ~1118 ns |
| Partie MCTS (1000 iter) vs aléatoire | — | 8 ms |
| Partie Smart IS-MCTS (20x50) vs aléatoire | — | 9 ms |
| Inférence DMC Q-Network | — | <1 ms |

## Règles

Implémente la Belote Contrée avec 4 couleurs (Pique, Cœur, Carreau, Trèfle). Comptage en mode "points faits + points demandés", les multiplicateurs coinche (x2) et surcoinche (x3) ne portant que sur la valeur du contrat. Sur une chute, la défense prend le contrat *et* tous les points cartes, quel que soit le partage réel des plis.

Les scores sont **exacts — rien n'est arrondi**. La FFB arrondit la marque à la dizaine ; ici une chute marque 162 + le contrat, pas 160, de sorte que le moteur et la feuille de score du web affichent toujours le même chiffre. Voir le [règlement FFB](https://www.ffbelote.org/wp-content/uploads/2015/11/REGLES-DE-LA-BELOTE-CONTREE.pdf) pour le texte officiel complet, et [docs/rules-survey/](docs/rules-survey/) pour ce que ~594 règlements publiés en font réellement.

## Références

- Kocsis, L. et Szepesvári, C. (2006). [Bandit Based Monte-Carlo Planning](https://link.springer.com/chapter/10.1007/11871842_29). *ECML*.
- Cowling, P.I., Powley, E.J. et Whitehouse, D. (2012). [Information Set Monte Carlo Tree Search](https://doi.org/10.1109/TCIAIG.2012.2200894). *IEEE Transactions on Computational Intelligence and AI in Games*.
- Zha, D. et al. (2021). [DouZero: Mastering DouDiZhu with Self-Play Deep Reinforcement Learning](https://arxiv.org/abs/2106.06135). *ICML*.
- Auer, P., Cesa-Bianchi, N. et Fischer, P. (2002). [Finite-time Analysis of the Multiarmed Bandit Problem](https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf). *Machine Learning*.

## Remerciements

Merci à **Ronan Guillou**, joueur de coinche aguerri, pour ses conseils avisés sur le jeu et pour avoir été le premier testeur — son bon sens a guidé de nombreux choix d'interface.

## Licence

**MIT** — voir [LICENSE](LICENSE). Ça couvre tout ce dépôt : le moteur, les liaisons Python et WASM, l'arène, les documents publiés.

Les **poids des modèles** ne sont pas dans le dépôt. Ils vivent sur [Hugging Face](https://huggingface.co/collections/Avo-k/colver-belote-contree-6a71df4a723e6734fe623a65) sous leur propre licence, et `colver.download_*()` va les y chercher. Un modèle est un artefact d'entraînement, pas du code : le versionner alourdirait le clone de dizaines de mégaoctets pour un fichier qui change à chaque campagne.

Le **site colver.net** n'est pas dans ce dépôt et n'est pas sous cette licence.
