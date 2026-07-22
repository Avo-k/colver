# Page Annonces — pistes d'amélioration (non implémentées)

Notes d'idées pour la page `/analyse/annonces`. Rien ici n'est engagé ; c'est un
backlog de réflexion. (Dernière mise à jour : 2026-07-23)

## 1. Oracle : choisir entre « Bandeau » et « Tableau complet »

Les deux vues disent la même chose (le % de réussite par couleur × palier) ;
seule la forme change. Il faut en garder une seule, ou les fusionner :

- **Option A — garder le bandeau seul**, avec les % au survol (déjà en title)
  et éventuellement le % affiché dans la cellule au-delà d'une certaine largeur
  d'écran. Le plus compact.
- **Option B — garder le tableau complet seul**, en lui appliquant le dégradé
  de couleur du bandeau (aujourd'hui il n'a que 4 classes de couleur).
- **Option C — bandeau par défaut + tableau complet replié** dans un
  `<details>` (comme « Voir 10 exemples de distribution »).

À trancher aussi : la place des marqueurs ▴80/▴50/▴20 si le bandeau disparaît
(ils vivent sous le bandeau ; le tableau Moyennes porte déjà Sûr/Tendu).

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

## 4. Déplacer « Voir 10 exemples de distribution »

Le repli vit en bas de la box Oracle (il y a suivi les donnes qu'il montre).
Emplacement à repenser — pistes : une box/onglet à part, près de la synthèse
DouDou, ou un lien discret dans l'en-tête Oracle. À noter que si le
sampling belief-conditionné (§3) arrive, ces exemples deviennent le moyen
naturel de *vérifier à l'œil* la crédibilité des mondes tirés — ça pourrait
mériter une place plus visible à ce moment-là.
