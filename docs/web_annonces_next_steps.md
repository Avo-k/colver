# Page Annonces — pistes d'amélioration (non implémentées)

Notes d'idées pour la page `/analyse/annonces`. Rien ici n'est engagé ; c'est un
backlog de réflexion. (Dernière mise à jour : 2026-07-23)

## 1. ~~Oracle : choisir entre « Bandeau » et « Tableau complet »~~ — FAIT (2026-07-23)

Résolu en option A : bandeau seul (renommé « Réussite par contrat »), Moyennes
affichées en premier, cellules ≤5 % en quasi-noir (« n'arrive jamais »), % au
survol. Reste possible un jour : afficher le % dans la cellule sur écran large.

## 2. Simulations : augmenter le nombre par défaut

Défaut actuel : 50. L'Oracle serveur fait 50 solves en ~0,2-2 s et DouDou50
50 parties en ~1,7 s — il y a de la marge. Points à considérer :

- Des défauts **différents par box** ? L'Oracle local (WASM single-thread) est
  le chemin le plus lent ; DouDou est toujours serveur.
- Le streaming de progression existe déjà, donc un défaut à 200-500 reste
  agréable : les tables se remplissent en direct.
- Coût CPU serveur : DouDou est l'endpoint cher (~35 forwards du réseau 1024³
  par sim). À 500 sims par évaluation, surveiller la charge si le trafic monte.

## 3. Distributions crédibles via les beliefs (au lieu du pur hasard)

Aujourd'hui, les mains adverses sont tirées **uniformément** parmi les 24
cartes restantes — l'historique d'enchères saisi n'influence pas la donne.
Avec `history=90♠ par Est`, les mondes générés ne donnent pas plus d'atouts
pique à Est qu'au hasard, alors que son annonce est une information forte.

Idée : échantillonner les mains conditionnées à l'historique d'enchères avec
les modèles de beliefs existants :

- **Bid Belief NN v4** (`models/bid_belief_v4.bin`, obs 108 → 96) : probabilité
  de localisation de chaque carte chez chaque joueur, conditionnée à l'enchère.
  C'est le bon modèle pour la phase d'annonces (le play belief net regarde le
  jeu de la carte). Côté Rust, `BeliefState` + `apply_nn_bid_beliefs()` font
  déjà ce calcul pour BisDd.
- **Playgen** (world sampler transformer) : alternative qui génère des mondes
  complets ; déjà branché sur la page croyances.
- Procédure d'échantillonnage : tirage séquentiel des cartes selon les probas
  par joueur avec contraintes dures (8 cartes/joueur), ou rejet/importance
  sampling sur des tirages uniformes pondérés par la vraisemblance du monde.

Effets attendus :

- **Oracle** : le « plafond théorique » devient conditionné aux annonces —
  beaucoup plus pertinent dès qu'un historique est saisi.
- **DouDou50** : les auctions simulées repartent de mains compatibles avec ce
  qui a déjà été dit, donc des taux de réussite plus réalistes.
- UI : proposer un interrupteur « mondes crédibles / mondes uniformes » (ou
  n'activer les beliefs que si un historique est présent), et afficher la
  vraisemblance moyenne des mondes tirés pour garder l'honnêteté statistique.

Attention : l'échantillonnage belief introduit un biais si le modèle est mal
calibré — garder le mode uniforme comme référence comparable.

## 4. ~~Déplacer « Voir 10 exemples de distribution »~~ — FAIT (2026-07-23)

Déplacé en box repliée pleine largeur sous les colonnes DouDou50/Oracle.
À garder en tête : si le sampling belief-conditionné (§3) arrive, ces exemples
deviennent le moyen naturel de *vérifier à l'œil* la crédibilité des mondes
tirés — ça pourrait mériter une place plus visible à ce moment-là.
