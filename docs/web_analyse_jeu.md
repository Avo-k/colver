# Analyse du jeu de la carte — design

Page d'analyse jumelle de « Analyse annonce » (`views/annonces.js`), mais pour
une décision de **carte** au lieu d'une décision d'annonce. Route proposée :
`/analyse/jeu`, à côté de `/analyse/annonces` et `/analyse/croyances`.

**Implémentée** (2026-07-26). Ce document garde le raisonnement de conception ;
les écarts entre le design et le code livré sont signalés en § 3 et § 10.

## Décisions actées

| | Choix |
|---|---|
| Source de la position | **CFN importé** (on arrive surtout depuis Rejouer) |
| Forme du résultat | **Une seule table indexée par carte candidate** |
| Périmètre des points | **Points de la donne** (pas de score de match, pas de Δ winprob) |

## 1. Ce qui porte la position

État de la page = **CFN complet 4 sections + index d'action** :

    /analyse/jeu?cfn=<full-game CFN>&i=<action idx>&obs=<siège>

- Le CFN 4 sections est celui de [`game_notation.py`](../python/colver/web/game_notation.py)
  (`<dealer>:<hands> <auction> <tricks> <contract>`), produit par
  `game_manager.compute_game_cfn()` et relu par `parse_full_cfn()`.
- **Le CFN cœur 3 sections ne suffit pas.** En phase `Playing`,
  `cfn.rs::format_contract` n'émet que le contrat résolu (`160hNS`) : la
  séquence d'enchères disparaît. Or l'obs de jeu canonique porte un historique
  d'enchères sur `[275:347]` (12×6) et playgen tokenise tout le préfixe visible,
  enchère comprise. Importer un CFN cœur poserait donc aux modèles une question
  qu'ils n'ont jamais vue à l'entraînement, et priverait playgen de son
  meilleur signal de conditionnement — c'est précisément l'enchère qui dit où
  est la force. Le CFN 4 sections est obligatoire, pas un confort.
- `ReplaySession.import_cfn()` exige une **partie complète** (`"CFN ne décrit
  pas une partie complète"` sinon). C'est sans friction pour le passage depuis
  Rejouer, et ça implique que **les quatre mains sont connues**. À exploiter
  (colonne « vrai monde », § 3), à condition d'être explicite sur quelles
  colonnes sont honnêtes vis-à-vis de l'information set et lesquelles non.

**Passage depuis Rejouer.** `replay.js` détient déjà `game_cfn` et un curseur de
coup : un lien « Analyser cette carte → » par action de jeu suffit, aucun
nouveau calcul serveur. C'est le chemin d'entrée principal.

**Observateur.** Par défaut le siège qui doit jouer ; commutable, comme les
boutons `Obs:` de la page Croyances. Il détermine *l'information set* depuis
lequel on échantillonne les mondes — donc tout le contenu de la colonne Oracle.

## 2. Les deux Oracles — le cœur de la page

Il y a deux questions distinctes, et les confondre viderait la page de son
intérêt :

1. **Oracle sur la vraie donne** — un solve, exact, toutes mains visibles :
   « quel était le meilleur coup en information parfaite ? ». C'est déjà ce que
   calcule [`analysis.py`](../python/colver/web/analysis.py) et ce que Rejouer
   affiche. **Aucune information nouvelle.**
2. **Oracle sur les mondes de l'information set** — on échantillonne des mondes
   compatibles avec ce que l'observateur pouvait savoir, on résout chacun, on
   agrège : « était-ce un bon choix *compte tenu de ce que ce siège savait* ? ».
   **C'est le contenu neuf de la page.**

Les deux sont affichés, dans des colonnes distinctement étiquetées. Elles ne
doivent jamais être fusionnées ni moyennées ensemble.

## 3. La table

**Écart au design : lignes = `legal_actions()`, pas `legal_actions_reduced()`.**
Le design prévoyait l'ensemble réduit pour ne pas payer deux fois la même
réponse. À l'implémentation ça casse : la réduction rend **un représentant par
classe sans dire à quelle classe appartient une carte donnée**, et les bots
choisissent dans l'ensemble complet. Un bot répondant 8♠ quand la réduction a
gardé 9♠ n'avait aucune ligne où atterrir, et son badge disparaissait
silencieusement — c'est exactement ce qu'on a observé, colonne Avis vide.

Reconstruire la relation d'équivalence en Python dupliquerait
`solver::reduce_equivalent`. Les rôles sont donc redistribués :

- `legal_actions_reduced()` → **y a-t-il une décision ?** (`len(reduced) < 2` =
  carte forcée, le test que `PlayProblemSession` utilise déjà). `legal` peut
  contenir plusieurs cartes quand `reduced` n'en a qu'une.
- `legal_actions()` → **les lignes.** Un solve les couvre toutes, donc la
  complétude est gratuite côté Oracle, et le budget de déroulements divise déjà
  par le nombre de lignes, donc elle est auto-limitante côté Jeu réel.

Une carte absente de `reduced` est équivalente à une autre : la ligne porte un
marqueur `≡`. On ne prétend pas dire *laquelle* — la réduction ne le dit pas.

Validation croisée utile : deux cartes équivalentes doivent afficher des
chiffres Oracle **identiques**. Sur la position de test, 8♠ et 9♠ donnent tous
deux 39,1 / 74 % / 14 %. Leurs colonnes *Jeu réel* diffèrent en revanche
(−106 vs −117) — et c'est correct, pas du bruit : équivalent en double-dummy
n'est pas équivalent pour DouDou50, dont l'observation change selon la carte
qui reste en main.

Groupes de colonnes :

| Groupe | Colonnes |
|---|---|
| Carte | pastille `cardChipHtml` |
| Oracle · information set | pts DD moyens (équipe du siège qui joue) · médiane · % de mondes où c'est la meilleure carte · **p(perte ≥ 10 pts)** |
| Jeu réel | pts de donne espérés · % contrat réussi (Wilson) · n |
| Vrai monde | coût DD exact sur la vraie donne (le `cost` d'`analysis.py`) |
| Avis | badges DouDou50 / Dédé / Oracle sur la ligne qu'ils choisiraient |

- La ligne réellement jouée est surlignée ; le regret est le Δ à la meilleure
  ligne du groupe considéré.
- `p(perte ≥ 10 pts)` est la colonne qui justifie la page. Une moyenne cache
  « gratuite dans 80 % des mondes, catastrophique dans 20 % », et c'est
  exactement l'information qui manque au joueur.
- Réutilisation directe : `visit-bars` / `visit-row`, `cardChipHtml`,
  `wilsonLower` + `confidenceClass` + les classes `doudou-high/mid/low`
  d'`annonces.js` (les échantillons par carte seront petits, la teinte doit
  porter la fiabilité).
- Le **désaccord entre les trois avis** est du contenu en soi et mérite d'être
  signalé explicitement. C'est la seule page qui peut montrer « Dédé et
  l'Oracle divergent ici, et voilà ce que ça coûte ».
- Rappel : **Dédé est lié à un siège** (croyances, suivi des coupes,
  échantillonneur de mondes tournent depuis un point de vue). Quatre instances,
  une par siège, comme dans `agent_review.py`. Interroger une instance unique
  lui donnerait une information que ce siège n'a jamais eue.

## 4. Les mondes

Il faut échantillonner des mondes **conditionnés à un préfixe de jeu**, pas
d'enchère. Ce qui existe et ce qui manque :

- `PlaygenAnalyst::play_worlds` — **existe** (`playgen/analysis.rs:131`), pas de
  binding PyO3.
- Route sidecar `POST /play_worlds` — **existe** (`playgen_gpu_server.rs`), pas
  exposée par `playgen_gpu.py` (qui n'expose que `beliefs` et `auction_deals`).

Deux ajouts d'une dizaine de lignes chacun. Repli constraint-uniforme si le
sidecar est absent, **annoncé** comme le badge `worlds_source` /
`worlds_counts` de la page annonces.

## 5. Budget

- **Oracle** : un `solve_scores()` par monde couvre **toutes les lignes** d'un
  coup (il rend les points NS pour chaque carte légale, plus `best_card`). Le
  coût est donc en `n_mondes`, indépendant du nombre de candidates. Il **s'effondre
  à mesure que la donne se vide** — de la donne complète à la microseconde en
  finale, coûts par forme dans
  [play/dd_solver.md](play/dd_solver.md#performance) — ce qui est exactement ce
  que `WORLDS_BY_CARDS_LEFT` exploite. Fenêtre glissante parallèle sur
  `_DD_EXECUTOR`, exactement comme `_run_annonces_sim`.
  - *(Cette ligne annonçait « ~13,5 ms en milieu de donne » : un des quatre
    chiffres périmés que la campagne du 2026-08-02 a invalidés, trop pessimiste
    de deux ordres de grandeur. Les budgets actuels ont été calibrés dessus et
    laissent donc de la précision inutilisée en fin de donne.)*
- `solve_windowed_reuse_tt` est fait pour ce profil (« lots de positions
  quasi identiques : les mondes échantillonnés d'une même main »). **Sa prémisse
  est fausse** : les mondes d'une même main ne sont pas groupés (36 % tombent à
  plus de 40 points de la moyenne), donc amorcer une fenêtre étroite ne vaut
  1,04× au mieux et les deux points d'entrée n'ont aucun appelant en production
  ([play/dd_solver_optimization.md](play/dd_solver_optimization.md) §2.3).
  Réserve fail-soft, si on y revient quand même : le résultat n'est exact que si
  `alpha < v < beta`, sinon c'est une borne et il faut réélargir. Traiter une
  borne comme une valeur est exactement le défaut de `quick_tricks`.
- **Jeu réel** : il faut forcer chaque candidate séparément → `n_cartes ×
  n_mondes` déroulements. **C'est l'axe qui domine le budget.** Le pool est
  dimensionné pour l'Oracle et les déroulements s'en partagent
  `n_mondes / n_cartes`.
- Les compteurs ne doivent **pas** être des constantes comme
  `ORACLE_SIMS` / `REAL_SIMS` : ici le coût baisse avec le numéro de pli (moins
  de cartes à résoudre, moins à dérouler). Les faire varier avec le pli.
- **Pool de mondes partagé**, comme aux annonces (`world_total = max`) : les
  deux groupes de colonnes doivent décrire le même échantillon, sinon on
  compare deux populations différentes.

## 6. Streaming

Même forme que `_run_annonces_sim` et `agent_review.stream()` : `("start",
total)`, une mise à jour par monde avec barre de progression
(`updateProgressBar` se réutilise tel quel), puis `done`. Tâche annulable à la
navigation, chaque pas dans `asyncio.to_thread` (la recherche Rust relâche le
GIL). Tiroir « Voir 10 exemples de distribution » conservé — il vaut encore
plus cher ici qu'aux annonces.

## 7. Hors périmètre, volontairement

- **Panneau XGB** : l'interprétabilité n'existe que pour l'annonce. Ne rien
  promettre. Substitut honnête si besoin plus tard : les marginales playgen
  (données de la page Croyances), qui expliquent *pourquoi* une carte semble
  bonne — « Dédé croit l'As d'atout chez Est à 72 % ».
- **Score de match / Δ probabilité de gagner** : C1 s'arrête aux points de la
  donne. À rouvrir plus tard, ça rejoint le paradoxe donne/match de v6 et
  l'entrée score manquante de playgen v3.
- **Constructeur de position carte par carte** : redondant avec l'import CFN.
- **Le bandeau couleur × palier** des annonces : mauvais axes ici.
- **La synthèse d'enchère** (« Nord après votre annonce », « surenchère
  adverse ») : pas d'analogue. Remplacée le cas échéant par une synthèse de
  pli (qui remporte ce pli, qui prend le dernier).

## 7 bis. L'exploration libre (2026-08-06)

*Livré. §10.6 de [rejouer_analyse_erreurs.md](idees/rejouer_analyse_erreurs.md),
où vit le raisonnement complet ; ici seulement ce qui touche cette page.*

La page prend une **branche** : `?v=<cartes>`, les cartes posées à la place de la
suite réelle à partir de `i`. Le serveur analyse `actions[:i] + branche` et tout
l'aval — vrai monde, avis, mondes échantillonnés, clé de cache — reçoit cette
liste-là sans rien savoir de l'exploration.

- **Le CFN ne bouge pas.** Il porte la vraie donne, donc les 32 cartes, dont la
  colonne « vrai monde » a besoin (§2). Réécrire le CFN à chaque coup exploré
  aurait perdu exactement ce que le §1 s'est donné du mal à obtenir.
- **Le joueur joue les quatre sièges**, un `▸` par ligne du tableau. Toute la
  donne est visible ici : il n'y a personne à qui déléguer, et un commutateur de
  bot aurait ajouté une troisième évaluation à expliquer.
- **La branche est validée coup par coup, côté serveur** (`variation.line(...,
  limit=0)`). `env.step()` ne valide rien : un `v=` bricolé produirait une
  position que le moteur aurait acceptée en silence — même famille que la fuite
  d'`integrity.py`.
- **« La carte jouée » n'existe que sur la vraie ligne.** Hors d'elle, le liseré
  « jouée » désignerait une carte que personne n'a jouée dans cette variante.
- **La clé de cache porte la position effective**, donc une branche qui rejoint
  la ligne réelle retombe sur l'entrée de la position réelle, et reculer dans une
  exploration est instantané.

**La suite en jeu parfait** (`card_analysis_line`) est ce qui rend l'exploration
lisible : sans elle on pousse des cartes sans jamais voir où ça mène. Elle vient
du même moteur que les variantes de Rejouer — pile de coups vide — donc les deux
pages ne peuvent pas se contredire. Un déroulé coûte ≤150 ms à l'entame et des
microsecondes en fin de donne, et il part **après** la position pour ne pas
retarder le premier rendu.

## 8. Cas limites

- **Une seule carte légale** → aucune décision, le dire, comme le passe forcé
  côté annonces. Idem si `legal_actions_reduced()` s'effondre à une classe.
- **Index en phase d'enchères** → renvoyer vers `/analyse/annonces`.
- Index après la dernière carte.
- CFN invalide / partie incomplète → message, pas de page vide.

## 9. Pièges rencontrés à l'implémentation

- **Deux échelles de points dans la même table.** La colonne Oracle est en
  points *cartes* double-dummy (0-252 avec capot et dix de der), le Jeu réel en
  points *de donne marqués* (contrat compris, un 160 réussi passe 320). Côte à
  côte et de magnitude voisine, ça invite à les soustraire. Le Jeu réel est donc
  rendu en **écart signé Nord-Sud moins Est-Ouest**, qui ne peut pas se lire
  comme un total de points cartes.
- **« Contrat réussi » est l'issue du preneur.** Sur un siège en défense, un
  taux élevé désigne la *pire* carte. `real_win` relit l'événement du côté du
  siège qui joue, et l'en-tête bascule en « Contrat chuté » — sans quoi la
  colonne se classe à l'envers une fois sur deux.
- **`send()` jette silencieusement quand le socket n'est pas `OPEN`.** La page
  s'atteint normalement par URL directe (lien de Rejouer, signet), donc la vue
  se monte avant l'ouverture : sans relais sur `onOpen`, elle restait vide sans
  message d'erreur. Le symptôme n'apparaît qu'à froid — en navigation SPA la
  socket est déjà ouverte et tout marche.
- **`.section-title` n'est pas flex** : la barre de progression et le badge de
  mondes tombaient chacun sur leur ligne, et le badge héritait des majuscules du
  titre, se lisant comme un second titre. Corollaire trouvé plus tard (mobile,
  2026-08-06) : le flex seul les fait alors **se superposer** sous ~500px, et
  `flex-wrap` ne suffit pas — `.sim-progress` est en `flex: 1; min-width: 0`,
  donc elle s'écrase à 12px au lieu de pousser le badge à la ligne, et son
  compteur, en `nowrap`, déborde. C'est le plancher (`min-width: 160px`) qui
  déclenche le retour, pas le `flex-wrap`.
- **Deux `.so` peuvent cohabiter dans `python/colver/`.** Un
  `_colver.cpython-312-*.so` périmé masque le `_colver.abi3.so` fraîchement
  construit (CPython préfère le tag spécifique) : `maturin develop` réussit et
  le nouveau binding reste introuvable. Supprimer l'ancien.
- **Le serveur web ne recharge pas à chaud.** Une modification Python demande un
  redémarrage ; sans ça on débogue l'ancien code, ce qui a fait chercher un bug
  d'équivalence là où il n'y en avait plus.

## 10. À trancher plus tard

- Colonne « Jeu réel » avec **Dédé** plutôt que DouDou50 : plus représentatif de
  la production, mais lié au siège (4 instances) et bien plus lent.
- Panneau « pourquoi » à base de marginales playgen.
- Position de la page dans `FELT_ROUTES` du routeur : elle montre un pli, mais
  son contenu principal est une table — les pages d'analyse sont sur fond
  neutre aujourd'hui.
