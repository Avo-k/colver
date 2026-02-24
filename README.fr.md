<p align="center">
  <img src="images/colver.png" alt="Logo Colver" width="200">
</p>

# Colver

**[Read in English](README.md)**

Environnement de Belote Contree rapide pour l'apprentissage par renforcement. Moteur Rust avec bindings Python.

**Demo en ligne : [avok.me/colver/](https://avok.me/colver/)** — tourne sur un Raspberry Pi.

## Caracteristiques

- **~1.4M rollouts/sec** en mono-thread (phase de jeu), ~895K rollouts/sec sur une donne complete
- **Etat de jeu `Copy` de <=96 octets** pour un clonage MCTS performant
- **Six agents IA** — reseau Q DMC, IS-DD avec reseau de croyances, oracle DD, Smart/Naive IS-MCTS, et heuristique
- **Encheres par reseau de neurones** — "Le Bide a Dede", un Dueling DQN entraine sur des donnes resolues en double-dummy
- **Reseau de croyances** — prediction NN de la localisation des cartes pour la recherche IS-DD
- **Interface web** — jouez contre l'IA, observez, analysez et resolvez des problemes (FastAPI + WebSocket)
- **Bindings Python** via PyO3 — classe `Env` avec stubs de types complets, installable depuis PyPI
- Zero dependances dans le coeur (seulement `rand` derriere un feature flag)

## Interface Web

Jouez contre les agents IA directement dans le navigateur sur **[avok.me/colver/](https://avok.me/colver/)**, ou lancez-le en local :

```bash
uv run python -m colver.web
# Ou : uv run colver-web
# Ouvrir http://localhost:8000
```

**Humain vs IA** — Jouez en tant que Sud contre des adversaires IA. Choisissez l'agent pour vos adversaires (Est/Ouest) et votre partenaire (Nord) independamment. La partie suit les regles officielles FFB : encheres avec coinche/surcoinche, puis 8 plis. Les cartes sont jouees instantanement au clic ; le curseur de pause controle le delai de l'IA.

![Onglet Jouer](images/screenshots/tab-play.png)

**IA vs IA** — Observez des parties IA contre IA avec toutes les mains visibles. Assignez un agent different a chacune des 4 places. Avancez action par action, jouez des plis entiers, ou utilisez la lecture automatique. Le panneau de stats affiche les Q-values, scores DD ou evaluations de main. Collez une chaine CFN pour charger une position specifique.

![Onglet Regarder](images/screenshots/tab-watch.png)

**Rejouer** — Parcourez et rejouez les parties passees (jouees ou observees). Cliquez sur une entree pour la rejouer pas a pas.

**Annonces** — Composez une main de 8 cartes, choisissez votre position dans le tour d'encheres, et voyez ce que *Le Bide a Dede* (l'encherisseur NN) annoncerait — avec les Q-values pour chaque action legale.

![Onglet Annonces](images/screenshots/tab-annonces.png)

**Croyances** — Visualisez comment le reseau de croyances et le modele heuristique predisent la localisation des cartes au fil d'une partie. Generez une partie aleatoire, avancez pas a pas, et observez les barres de probabilite par carte avec marquage de la verite terrain et statistiques de precision. Changez de perspective (N/E/S/O) et comparez les predictions NN vs heuristiques cote a cote.

![Onglet Croyances](images/screenshots/tab-croyances.png)

**Problemes d'annonce** — Problemes d'encheres. Voyez une main et l'historique des encheres, puis trouvez la bonne annonce. L'IA evalue votre reponse.

**Problemes de jeu** — Problemes de jeu de la carte. Voyez une position en cours de partie et trouvez la meilleure carte. Comparez votre choix au jeu optimal du solveur DD.

## Compilation et execution

Necessite Rust 1.70+ et Python 3.10+.

```bash
# Tests (357 tests)
cargo test -p colver-core

# Benchmark de performance
cargo run -p colver-core --bin bench --release

# Demo MCTS vs aleatoire
cargo run -p colver-core --bin mcts_demo --release -- 100

# Demo Smart IS-MCTS vs aleatoire + vs naif
cargo run -p colver-core --bin smart_ismcts_demo --release -- 100

# Bindings Python (via uv)
uv sync
uv run python3 -c "import colver; env = colver.Env(); print(env.reset())"

# Interface web (jouer contre l'IA)
uv run python -m colver.web

# Entrainement DMC (reseau Q)
PYTHONPATH=scripts uv run python scripts/train_dmc.py --num-envs 256 --steps 20000000

# Evaluation DMC vs IS-MCTS
PYTHONPATH=scripts uv run python scripts/eval_dmc.py models/dmc_final.pt --baseline smart --time-ms 20 --both-sides
```

## Agents IA

### Oracle — Solveur DD (`solver.rs`)

Solveur double-dummy en information parfaite qui voit les 4 mains — il *triche*. Alpha-beta avec tables de transposition, PVS, killer moves et elagage par equivalence de cartes. Calcule la carte optimale exacte en ~7ms (mediane). Utile comme borne superieure.

### Dede — IS-DD (`is_dd.rs`)

Recherche Information Set Double-Dummy. Maintient un modele probabiliste de croyances sur les cartes cachees — mis a jour apres chaque action via des contraintes dures (coupes, plafond d'atout) et des signaux faibles (encheres, conventions de jeu). Peut etre augmente par un **reseau de croyances** (prediction NN de la localisation des cartes, 330→512→512→128, ~2Mo). Echantillonne des mains adverses ponderees par ces croyances, puis resout chaque monde exactement avec le solveur alpha-beta DD. IS-DD se prononce « is Dede » — d'ou le surnom.

### DouDou — Reseau Q DMC (`dmc_net.rs`)

Agent par apprentissage par renforcement de style [DouZero](https://arxiv.org/abs/2106.06135). Un reseau Q choisit les cartes a jouer en une seule passe forward — **sans arbre de recherche**. Entraine par self-play avec Prioritized Experience Replay sur 35M etapes.

**Architecture** : Dueling DQN 415→1024→1024→1024→32 avec LayerNorm (~2.6M parametres). Inference en Rust pur (~1ms/decision, pas de PyTorch necessaire). Agent le plus fort dans l'ensemble. *DouDou* = en reference a DouZero.

### Anciens agents de recherche

**Smart IS-MCTS** (`smart_ismcts.rs`) — [IS-MCTS](https://doi.org/10.1109/TCIAIG.2012.2200894) pondere par croyances heuristiques. **Naive IS-MCTS** (`naive_ismcts.rs`) — Determinisation par ensemble sans croyances. Les deux sont configurables et documentes dans [docs/SMART_ISMCTS.md](docs/SMART_ISMCTS.md).

### Le Bide a Dede — Encherisseur NN (`bid_net.rs`)

Dueling DQN (114→256→256→43) entraine sur 1M de donnes resolues en DD. Encherisseur par defaut pour tous les agents. Bat le meilleur encherisseur heuristique 70-76% sur tous les moteurs de jeu.

## Comparaison des agents

| Agent | Type | Vitesse/coup | Notes |
|---|---|---|---|
| Oracle (DD) | Solveur DD (triche) | ~7ms | Borne superieure |
| Dede (IS-DD) | Solveur DD + croyances | ~20ms | Plus fort avec recherche |
| **DouDou** | **Reseau Q** | **<1ms** | Plus fort, sans recherche |
| Smart IS-MCTS | Recherche + croyances | ~9ms | Budget configurable |
| Naive IS-MCTS | Recherche | ~8ms | Budget configurable |

**Note** : Les agents a base de recherche voient leur force augmenter avec le budget de temps. L'agent DMC n'utilise aucune recherche — la decision est prise en une seule inference.

## Architecture

**Workspace :** `colver-core` (Rust pur) + `colver-py` (PyO3/NumPy FFI) + `colver-web` (FastAPI/WebSocket)

### Representation des cartes

Systeme de bitmask : `Card = u8` (0-31), `CardSet = u32` (bitmask). Disposition : Pique\[0-7\], Coeur\[8-15\], Carreau\[16-23\], Trefle\[24-31\]. Dans chaque couleur : 7, 8, 9, V, D, R, 10, A (ordre de force hors atout). Force a l'atout : V > 9 > A > 10 > R > D > 8 > 7.

### Etat de jeu

`GameState` est `Copy` et fait <=96 octets (verifie a la compilation) pour un clonage MCTS rapide. Contient les mains, le pli courant, le contrat, les points/plis par equipe, l'etat des encheres, le bitmask des cartes jouees, le suivi des coupes et de la belote.

### Encodage des actions

| Phase | Actions | Encodage |
|---|---|---|
| Encheres | 43 total | 0=PASSE, 1-36=encheres (9 valeurs x 4 couleurs), 37-40=capot x 4, 41=COINCHE, 42=SURCOINCHE |
| Jeu | 32 total | Index de carte 0-31 directement |

### Deroulement

Encheres → Jeu → Fin. Les encheres se terminent apres 3 passes consecutives, une surcoinche, ou 4 passes (donne nulle). Le jeu comporte 8 plis de 4 cartes. Total des points de carte = 152 ; avec dix de der = 162 (normal) ou 252 (capot).

## API Python

```python
import colver

print(colver.__version__)  # "0.3.2"

# Environnement unique
env = colver.Env()
obs, legal_actions = env.reset()
obs, reward, done, legal_actions = env.step(action)

env.current_player()       # 0-3
env.phase()                # 0=Encheres, 1=Jeu, 2=Fin
env.legal_action_mask()    # tableau numpy (43,)
env.rewards()              # [score_NS, score_EO]
env.bid_improved()         # action d'enchere improved_bid
env.deal_outcome()         # [resultat_NS, resultat_EO] binaire
env.get_observation()      # vecteur d'observation de 415 flottants
env.action_naive_ismcts(20)  # action IS-MCTS naif (20ms)
env.action_smart_ismcts(20)  # action IS-MCTS intelligent (20ms)

# Reseau Q DMC (si les poids du modele sont telecharges)
model = colver.model_path()  # ~/.cache/colver/models/dmc_final.bin
if model:
    env.load_dmc_model(str(model))
    result = env.action_dmc_with_stats()  # {"best_action": 5, "q_values": [...]}
```

## Performance

| Charge | Debit | Latence |
|---|---|---|
| Rollout phase de jeu | 1.4M/sec | ~720 ns |
| Rollout donne complete | 895K/sec | ~1118 ns |
| Partie MCTS (1000 iter) vs aleatoire | — | 8 ms |
| Partie Smart IS-MCTS (20x50) vs aleatoire | — | 9 ms |
| Inference DMC Q-Network | — | <1 ms |

## Docker

L'image Docker permet de deployer l'interface web sur n'importe quelle machine, y compris un Raspberry Pi (ARM64).

```bash
# Build et lancement
docker build -t colver .
docker run -p 8000:8000 colver

# Ou avec Docker Compose
docker compose up -d

# Cross-build pour Raspberry Pi (ARM64)
docker buildx build --platform linux/arm64 -t colver .
```

L'image fait ~257 Mo (pas de dependance PyTorch). Tous les agents tournent en Rust pur et fonctionnent sur toutes les architectures.

## Regles

Implemente la Belote Contree avec 4 couleurs (Pique, Coeur, Carreau, Trefle). Comptage en mode "points faits + points demandes". Voir `REGLES-DE-LA-BELOTE-CONTREE.pdf` pour le reglement complet FFB.

## References

- Kocsis, L. et Szepesvari, C. (2006). [Bandit Based Monte-Carlo Planning](https://link.springer.com/chapter/10.1007/11871842_29). *ECML*.
- Cowling, P.I., Powley, E.J. et Whitehouse, D. (2012). [Information Set Monte Carlo Tree Search](https://doi.org/10.1109/TCIAIG.2012.2200894). *IEEE Transactions on Computational Intelligence and AI in Games*.
- Zha, D. et al. (2021). [DouZero: Mastering DouDiZhu with Self-Play Deep Reinforcement Learning](https://arxiv.org/abs/2106.06135). *ICML*.
- Auer, P., Cesa-Bianchi, N. et Fischer, P. (2002). [Finite-time Analysis of the Multiarmed Bandit Problem](https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf). *Machine Learning*.

## Remerciements

Merci a **Ronan Guillou**, joueur de coinche aguerri, pour ses conseils avises sur le jeu et pour avoir ete le premier testeur — son bon sens a guide de nombreux choix d'interface.
