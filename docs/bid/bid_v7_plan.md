# Plan d'entraînement — Bid v7

*Ouvert le 2026-08-02. Document vivant : chaque question de la §2 se ferme par une
mesure, qui remonte en §1 avec son chiffre, ou tombe.*

Référence courante : **Bid v6 ISDD** (`models/bid_v6_isdd_resume/bid_nn_final.bin`,
obs 117 score-aware v3, 75M pas). Tout ce qui suit est mesuré contre lui.

Le contexte matériel : **les pools DD sont déjà périmés** (retrait de `quick_tricks`
le 2026-07-23, puis le correctif de règle d'atout du 2026-08-01, qui élargit
l'ensemble des coups légaux). Une regénération est due de toute façon — v7 est donc
le bon moment pour changer ce qui se décide au moment de fabriquer les données, pas
seulement au moment d'entraîner.

---

## 1. Ce dont on est sûr

### 1.1 v6 n'est pas équivariant aux couleurs, et l'erreur dépasse largement sa marge de décision

L'obs d'annonce porte la main en bits bruts (`[0:32]`, [bid_obs.rs](../../colver-core/src/bid/bid_obs.rs)).
Rien dans un MLP n'impose que deux mains identiques à un renommage de couleurs près
donnent la même annonce. Mesure sur 400 donnes × 23 permutations non triviales :

| bidder | annonces qui basculent sous renommage |
|---|---|
| `bid_improved_v2` (contrôle) | 3,1 % |
| `bid_roro` (contrôle) | 0,9 % |
| **Bid v6** | **24,6 %** |

Les contrôles ne sont pas nuls parce qu'un heuristique départage les ex æquo par
indice de couleur — comportement déterministe et légitime. Ils servent surtout à
valider l'arithmétique de permutation : une erreur de mapping donnerait ~75 % partout.

Le chiffre qui compte est un rapport de deux échelles :

- erreur d'équivariance du **vecteur Q** (max sur les actions, après permutation
  inverse) : médiane **0,037**, p90 0,054, p99 0,070 — soit **4,7 % de l'étendue des
  Q** (0,79 en médiane). En absolu, c'est petit.
- écart **top1 − top2**, la marge qui décide réellement de l'annonce : médiane
  **0,0042**. 97,8 % des positions ont un top-2 séparé de moins de 0,03.

**Le bruit de symétrie vaut 8,8× la marge de décision.** C'est la cause mécanique des
24,6 %. Ce n'est pas une curiosité esthétique : un quart des annonces de v6 est
décidé par du bruit d'apprentissage plutôt que par le contenu de la main.

### 1.2 La fonction Q est plate au sommet

Corollaire du point précédent, mais il tient tout seul : v6 sépare ses deux
meilleures options par 0,0042 sur une étendue de 0,79, soit **0,5 % de sa propre
échelle**. Que ce soit un défaut de calibration ou le reflet d'une réalité du jeu
(beaucoup d'annonces sont vraiment presque équivalentes) reste ouvert — cf. §2.2 —
mais le fait est établi et il conditionne toute lecture d'un argmax de v6.

### 1.3 Le capot est une action morte

Sur **3000 enchères** jouées par v6 aux quatre sièges : **0 capot**. Contrats à 160
(le plafond non-capot) : 0,47 %. Donnes passées : 0,1 %.

Sondes directes, v6 premier à parler, sur des mains **capot forcé** — huit levées
garanties quelle que soit la répartition des 24 autres cartes :

| main | annonce v6 | Q de l'annonce | meilleur Q capot |
|---|---|---|---|
| J9AKQ10 ♠ + A-10 ♥ | 160♠ | 0,599 | 0,453 |
| J9AKQ10 ♥ + A-10 ♣ | 160♥ | 0,584 | 0,420 |
| **les 8 carreaux** | **140♦** | **0,655** | 0,455 |

La troisième ligne est le cas le plus trivial du jeu — huit cartes d'une couleur,
huit levées par construction — et l'écart y est de 0,20 sur une étendue de 0,79 :
**ce n'est pas un ex æquo, c'est une réponse fausse et confiante**.

Le coût, mesuré par continuation d'enchère (§4, 200 mondes playgen, écart de points
marqués N-S − E-O) :

| annonce forcée | Q v6 | écart moyen | vs 140♦ (apparié) | contrat final |
|---|---|---|---|---|
| 140♦ *(choix de v6)* | 0,655 | +436,5 ± 17,8 | (réf) | 140♦ 147/200 |
| 150♦ | 0,546 | +849,3 ± 13,8 | **+412,8 ± 17,1** | 150♦xx 106, CAPOT♦xx 90 |
| **CAPOT♦** | 0,455 | **+1022,0 ± 0,0** | **+585,5 ± 17,8** | CAPOT♦xx 200/200 |

**v6 laisse 585 points par donne**, pas les 110-200 estimés d'abord à la main : les
adversaires coinchent le capot, on surcoinche, et 252 + 250×3 + 20 de belote = 1022,
identique dans les 200 mondes puisque le capot est forcé. L'écart-type nul est aussi
un contrôle du pipeline.

**Nuance importante, et rassurante** : sur une main ordinaire, v6 évalue le capot
correctement (Q = −0,699, valeur réelle mesurée −714 ± 17, contrat réussi 2 % du
temps). Le défaut n'est donc pas « le capot est réprimé partout » mais « le capot
n'est jamais annoncé *là où il faut* » — l'action n'a jamais reçu de gradient positif.

Sur les deux premières lignes il sature le plafond (160) : la signature d'une action
jamais explorée plutôt que d'une évaluation prudente.

### 1.4 L'espace des mains est petit, et déjà couvert

À permutation de couleurs près (les couleurs sont interchangeables avant que l'atout
soit nommé) :

| | brut | classes distinctes |
|---|---|---|
| une main de 8 cartes | 10 518 300 | **472 579** |
| une main, atout fixé (mod S₃) | 10 518 300 | 1 820 803 |
| une donne complète 4×8 | 9,96 × 10¹⁶ | 4,15 × 10¹⁵ |

(Burnside, pas une division par 24 : 7,5 % des mains ont une symétrie de couleur.
Vérifié par énumération exhaustive des 10 518 300 mains, et le même code de Burnside
validé par force brute sur des jeux réduits.)

Ces deux constantes ne vivent plus dans un script : elles sont calculées en `const fn`
et asserties dans le moteur ([hand_class.rs](../../colver-core/src/hand_class.rs)), qui
fournit aussi l'index bijectif (cf. §3.6) et un code de main lisible pour nommer les
strates — [interpretability/hand_classification.md](interpretability/hand_classification.md).

Conséquence directe : un pool de 5M donnes fournit 20M mains, et le coupon collector
n'en demande que **6,2M** pour voir les 472 579 classes. **La couverture de l'espace
des mains n'est pas un problème et ne peut pas être une piste d'amélioration.**

Ce qui est rare, c'est la queue :

| famille | fréquence | occurrences attendues dans un pool 5M (20M mains) |
|---|---|---|
| main contenant les 6 gros atouts d'une couleur | 1,24 × 10⁻⁴ | ~2 470 |
| main monocolore (8 cartes d'une couleur) | 3,80 × 10⁻⁷ | ~8 |

Donc le signal capot **existe** dans les données — il est simplement à 10⁻⁴, et
surtout la trajectoire d'enchère ne l'atteint jamais : le modèle joue ses propres
enchères, il n'annonce jamais capot, donc il n'apprend jamais ce que ça vaut. Boucle
fermée, que ni plus de données ni plus de pas ne rouvriront.

### 1.5 Le solve fenêtré ne réduit pas le coût du scoring IS-DD — et pourquoi

**Résultat négatif, mesuré le 2026-08-02.** Le poste dominant d'une regénération de
pool est le scoring IS-DD : ~16 donnes/s sur 32 cœurs, soit **~87 h pour 5M donnes**,
et il n'a aucun chemin GPU (il n'existe pas de solveur DD CUDA ; `gen_pool` est
explicitement « no CUDA dep »). Par comparaison, DouDou score les mêmes 5M en ~3 min
sur 4090 — un facteur ~1500 en faveur de la couche qu'on a justement abandonnée.

`solve_windowed_reuse_tt` / `solve_for_trump_windowed` ont été écrits pour exactement
cette boucle : les mondes échantillonnés d'une même main partagent les 8 cartes de
l'observateur, donc leurs valeurs DD devraient se regrouper, et une fenêtre étroite
amorcée par la moyenne courante devrait élaguer davantage. **Ils n'ont jamais été
branchés** — IS-DD appelle `solve_with_scores` en fenêtre pleine
([is_dd.rs:1042](../../colver-core/src/search/is_dd.rs#L1042)). Vérification par
`bench_solve_window`, 40 positions × 40 mondes × 4 couleurs = 6400 solves par point,
~28 ms/solve en fenêtre pleine :

| `delta` | accélération | taux de re-recherche | écarts de valeur |
|---|---|---|---|
| 20 | **0,96×** | 63,9 % | 0 |
| 40 | 1,03× | 36,3 % | 0 |
| 80 | 1,03× | 12,1 % | 0 |
| 120 | 1,04× | 9,6 % | 0 |

**Au mieux 1,04×** — 87 h deviendraient 84 h. Ce n'est pas un levier, et l'optimisation
ne mérite pas d'être branchée. Aucun écart de valeur nulle part : la correction du
solve fenêtré tient, c'est sa prémisse qui est fausse.

**Et c'est la prémisse qui est intéressante.** Les taux de re-recherche mesurent
directement la dispersion des valeurs DD entre mondes d'une même main : **36 % des
mondes s'écartent de plus de 40 points de la moyenne courante, 12 % de plus de 80**,
sur une échelle 0-252. Les mondes d'une même main ne sont pas du tout groupés. C'est
le pendant expérimental du « une main seule n'explique que ~17 % de la variance DD » :
la même chose vue depuis l'autre bout.

*Conséquence pour v7* : le label d'une position d'enchère est une moyenne sur des
mondes très dispersés, donc **le nombre de déterminisations pèse lourd sur la qualité
du label**. À budget CPU constant, scorer 1M donnes avec beaucoup de mondes est
peut-être meilleur que 5M avec 20 — hypothèse à trancher avec `bench_label_variance`
avant d'engager les 87 h. Cf. §2.8.

### 1.7 Le Q plat reflète un jeu réellement plat — et v6 ordonne juste

*Mesuré le 2026-08-02, [bid_q_flatness.py](../../scripts/analysis/bid_q_flatness.py),
40 mains d'ouverture × 300 mondes playgen × 3 candidates = 36 000 déroulements.*

Pour chaque main : la valeur réelle du top-1 de v6, celle de son top-2, et celle de sa
**8ᵉ** annonce comme contrôle positif. Écarts de points marqués, appariés sur le même
pool de mondes, puis moyennés sur les mains.

| | Δ vs top-1 | lecture |
|---|---|---|
| top-2 | **−8,20 ± 4,31** | le top-1 est vraiment le meilleur, mais de peu (1,9 σ) |
| 8ᵉ annonce | **−114,50 ± 6,48** | **contrôle positif**, 17,7 σ — la mesure a toute la puissance voulue |

- top-1 bat top-2 sur **60 %** des mains ;
- corrélation **ΔQ ↔ Δréel = +0,504** alors que le ΔQ médian de ces paires vaut 0,0041.

**Conclusion : l'hypothèse (a) l'emporte.** Le sommet du Q est plat parce que le jeu
l'est — deux annonces voisines valent à ~8 points près — et non parce que v6
sous-discriminerait. Le contrôle positif l'établit : le même protocole détecte sans
peine les 114 points qui séparent le top-1 d'une annonce médiocre. Mieux, v6 ordonne
correctement ces quasi-égalités : un écart de Q de 0,004 prédit encore le bon
gagnant à r = 0,50.

**Ce que ça change pour §3.1.** Le bruit de symétrie fait basculer 24,6 % des annonces,
mais il les fait basculer *entre options qui valent ~8 points d'écart* : le coût
attendu est de l'ordre de **2 points par donne**, pas des dizaines. La canonicalisation
reste justifiée — hygiène, capacité libérée, et c'est elle qui rend §3.6 bien définie —
mais **il ne faut pas en attendre un gain de force spectaculaire**. Mise en regard
utile : le bug capot vaut 585 points sur ~10⁻⁴ des donnes, soit ~0,06 pt/donne. En
espérance, la symétrie pèse donc plus lourd que le capot, d'un ordre de grandeur.

*Limite* : mesuré sur des annonces d'**ouverture** seulement. Une sonde avec préfixe
(partenaire à 100♣) donnait −86 ± 25 pour la 3ᵉ candidate — les décisions de milieu
d'enchère sont probablement moins plates. À refaire avec `--prior`.

### 1.6 Acquis antérieurs qui contraignent v7

Établis avant ce document, à ne pas re-litiger :

- **Le label doit être une continuation d'enchère, pas une valeur de contrat.**
  Distiller « valeur de jouer ce contrat » fait sur-annoncer : 47,3 % puis 43-44 %
  vs v6. Le pipeline est réutilisable, c'est la cible qui est fausse.
  [experiments/auction_conditioned_labels.md](experiments/auction_conditioned_labels.md)
- **Jamais de E[Y|X] substitué par échantillon avant un seuil** (erreur de Jensen) :
  on calibre en échantillonnant.
- **Deux features manquantes identifiées par sonde de couche cachée** — J/9 par
  couleur, `opp_best_other_ts` — ferment l'écart 77 % → 97 %.
  [interpretability/probe_morning_report.md](interpretability/probe_morning_report.md)
- **Le modèle joue ses propres enchères** (ε-greedy sur sa politique) ; l'oracle
  supervise la loss, jamais la trajectoire. Sinon on obtient des enchères dégénérées
  (annonce optimale → 3 passes) qui n'enseignent rien de la communication.
- **Une main seule n'explique que ~17 % de la variance de l'issue DD.** Le reste est
  dans l'historique d'enchères. Plafond structurel de toute approche « main → annonce ».
- **v6 bat v5** 55,8 % (jeu DMC) / 57,3 % +181 (jeu IS-DD), mais perd -16 à -26
  pts/donne à chaque sonde de score : le paradoxe donne/match n'est pas expliqué.

---

## 2. Ce qui reste à prouver

Chaque question est formulée pour pouvoir tomber.

**2.1 — Canonicaliser l'obs d'annonce améliore-t-il la force, ou seulement la cohérence ?**
L'erreur d'équivariance est grande *relativement à la marge*, mais petite en absolu
(4,7 % de l'étendue). Il se peut que supprimer ce bruit ne change rien à l'arène
parce que les options qu'il permute sont réellement d'égale valeur.
*Test* : entraîner v7 à budget identique avec et sans canonicalisation, h2h 1000
matchs. *Critère* : ≥ 52 % pour la version canonique, et flip rate mesuré à 0 %.

**2.2 — Le Q plat est-il un défaut de calibration ou la réalité du jeu ?**
**Fermée le 2026-08-02 : c'est le jeu, et v6 ordonne juste.** Voir §1.7.

**2.3 — Réveiller le capot rapporte-t-il quelque chose en arène ?**
Les capots forcés sont à ~10⁻⁴. Même parfaitement joués, le gain en % de matchs sera
minuscule ; le gain en points sur ces donnes est de 110 à 200. *Test* : v7 avec capot
réveillé vs v7 sans, h2h. *Attente honnête* : indétectable en % de matchs, visible en
pts/donne conditionnellement à la famille de mains. **À traiter comme une correction
de justesse, pas comme un gain de force** — ne pas la vendre autrement.

**2.4 — La stratification sans poids d'importance fait-elle sur-annoncer ?**
Hypothèse forte au vu de §1.5. Suréchantillonner les mains fortes sans corriger, c'est
enseigner un mauvais a priori sur ce que tiennent les adversaires. *Test* : trois
variantes (uniforme / stratifié pondéré / stratifié non pondéré) à budget égal.
*Critère* : distribution des valeurs de contrat + h2h.

**2.5 — Les mondes playgen sont-ils assez fidèles pour servir de cible d'annonce ?**
`bench_world_cred` dit que playgen domine les autres sources **en phase de jeu**.
Rien ne le dit pour l'annonce. *Test* : la même méthodologie, positions d'enchère
uniquement, playgen vs uniforme vs belief. Attention à la règle maison : **ne jamais
tirer les questions du flux que consomme le testé**.

**2.6 — Le paradoxe donne/match de v6 est-il un artefact de la reward Δ-winprob ?**
Ouvert depuis v6. Il conditionne le choix de reward pour v7, donc il faut au moins
une hypothèse testée avant de figer le signal.

**2.7 — Une architecture équivariante bat-elle une canonicalisation ?**
Alternative à 2.1 : un encodeur par couleur partagé + agrégation invariante (deep
sets) au lieu d'un ordre canonique choisi à la main. Plus propre, plus cher, et
l'ordre canonique a déjà fait ses preuves côté jeu (obs 411). *À ne tenter que si
2.1 est positif.*

**2.8 — Où mettre les 87 h de CPU : plus de donnes, ou plus de mondes par donne ?**
**Tranchée le 2026-08-02 : les donnes gagnent. Garder 5M × 20, ne pas payer plus de
déterminisations.** Ce qui suit remonterait en §1 s'il était mesuré plutôt que déduit.

`bench_label_variance` (200 donnes × 60 redonnes × 4 couleurs, siège 0 figé) décompose
la variance du label DD :

| | pts |
|---|---|
| bruit des 24 cartes invisibles (intra-main) | **44,7** |
| part expliquée par la main visible (inter-main) | **28,7** |
| → part de la variance que l'annonceur **ne peut pas voir** | **71 %** |

(R² de la main seule = 29 %. À rapprocher — sans les confondre — du « ~17 % » cité
ailleurs dans le dépôt : ici c'est à couleur d'atout et siège fixés, pas sur le
meilleur contrat. Les deux disent la même chose, pas au même endroit.)

Le bruit de label à k mondes vaut `44,7/√k`, et **le budget de solves est `N × k`** :

| k | erreur type | rapport signal/bruit | donnes pour 100M solves |
|---|---|---|---|
| **20** | 10,0 pts | 2,9 | **5,0M** |
| 60 | 5,8 pts | 5,0 | 1,7M |
| 100 | 4,5 pts | 6,4 | 1,0M |
| 240 | 2,9 pts | 9,9 | 0,4M |

L'argument est statistique et non empirique : l'information portée par un échantillon
est ∝ `1/Var = k/σ²`, donc l'information totale est `N·k/σ²` — **constante à budget
`N·k` fixé**. Les deux allocations sont équivalentes au premier ordre, et trois effets
de second ordre départagent, tous dans le même sens :

1. **La couverture.** 5M donnes = 20M mains, assez pour voir les 472 579 classes
   (§1.4 : 6,2M tirages) ; 1M donnes = 4M mains, pas assez. Et le label dépend du
   couple (main, enchère), donc plus de donnes = plus de contextes distincts.
2. **Un bruit non biaisé ne biaise pas un ajustement aux moindres carrés** — il ralentit
   la convergence, il ne déplace pas l'optimum. Or à k=20 le rapport signal/bruit est
   déjà de 2,9 : le bruit est nettement sous le signal.
3. **Moyenner avant un seuil est précisément l'erreur de Jensen** consignée en §1.6.
   Si la reward applique un seuil au label, des labels par échantillon *bruités* sont
   la bonne entrée d'une calibration par échantillonnage — les pré-moyenner serait faux.

Et la marge réelle est meilleure que ce tableau : IS-DD n'échantillonne pas des
redonnes uniformes mais des mondes playgen conditionnés par l'enchère, dont la
dispersion est plus faible que les 44,7 pts mesurés ici en uniforme.

*Reste ouvert* : la valeur marginale de 5M contre 1M à k identique (87 h contre 17 h).
C'est une question de rendement décroissant, pas d'allocation.

---

## 3. Pistes d'amélioration

Par rapport valeur/coût décroissant.

### 3.1 Canonicaliser l'obs d'annonce *(coût faible, effet structurel)*
Comme l'obs de jeu 411 (`canonical_play_order`). Avant l'enchère il n'y a pas d'atout
pour ancrer l'ordre : trier les couleurs par (longueur, motif de rangs) décroissant.
La sortie devant nommer une couleur, il faut remapper l'action comme le fait
`card_to_physical`. Espace d'entrée effectif ÷ ~22, équivariance gratuite, et une
source de bruit qui vaut 9× la marge de décision qui disparaît. Le tri existe dans
[suit_perm.rs](../../colver-core/src/suit_perm.rs).
*Effet secondaire utile* : rend la §3.6 bien définie.

### 3.2 Suite de sondes stratifiée, en **évaluation** *(coût nul, risque nul)*
Quelques centaines de mains construites par famille — capot forcé, 8 atouts, 7
atouts, coupe franche + longue, mains limites 150/160, mains de relance — et on lit
ce que le bidder annonce. Aucun risque de polluer l'entraînement. **Ça aurait attrapé
§1.3 il y a des mois.** À faire tourner à chaque checkpoint, comme un test.

### 3.3 Réveiller les actions mortes *(coût moyen)*
ε-greedy ne suffit pas : à 10⁻⁴ de fréquence et 43 actions, l'action capot ne reçoit
essentiellement aucun gradient. Options, à tester dans cet ordre :
- forcer une proportion plancher de transitions capot dans le replay (avec poids) ;
- bonus d'exploration par compte d'action (UCB-like) sur la politique de collecte ;
- amorçage supervisé : sur les mains capot forcé du pool, un lot de labels durs.

### 3.4 Les deux features manquantes du probe *(coût faible, effet mesuré)*
J/9 par couleur et `opp_best_other_ts` : 77 % → 97 % d'accord sur la sonde. C'est le
seul lead de ce document dont l'effet est *déjà* chiffré. À intégrer à l'obs v7.

### 3.5 Génération stratifiée, avec poids d'importance conservés *(coût moyen, risque réel)*
Stratifier **sur l'issue, pas sur la main** : l'espace des mains est saturé (§1.4),
ce qui manque c'est la queue des contrats optimaux. Un plancher par bucket de
« meilleure valeur DD » (≥ 150, capot) est plus direct qu'un plancher par forme.
Garder `p_uniforme / p_stratifié` dans la loss, ou tenir deux flux séparés. À
construire dans [gen_pool.rs](../../colver-core/src/bin/gen_pool.rs) au moment de la
regénération due.

### 3.6 Table exhaustive de la politique d'ouverture *(coût faible, interprétabilité)*
Premier à parler, historique vide, score 0-0 → l'annonce est une fonction pure de la
main, donc une table de 472 579 entrées (1 820 803 à atout fixé). Ce n'est pas une
approximation type SHAP, c'est *le modèle*, exhaustivement : on peut trier par classe,
croiser avec le DD, et lire directement où il se trompe.

**L'outillage existe depuis le 2026-08-02** : `hand_class_id` / `hand_from_class_id`
([hand_class.rs](../../colver-core/src/hand_class.rs)) est une bijection sur les 472 579
classes, donc l'énumération est acquise et testée. **Le blocage restant est entièrement
§3.1** : tant que l'obs n'est pas canonicalisée, la même classe donne jusqu'à 24 réponses
différentes (§1.1) et la table n'est pas bien définie. Le même module fournit `HandCode`,
qui nomme les strates de §3.5 et les familles de §3.2 —
[interpretability/hand_classification.md](interpretability/hand_classification.md).

### 3.7 Évaluateur d'annonces candidates sur mondes playgen
Voir §4 : c'est autant un outil de diagnostic qu'une source potentielle de labels.

### 3.8 Croyance playgen **en entrée** du bidder *(la piste la plus prometteuse, coût à cadrer)*

**L'idée.** Les 71 % de §2.8 sont une mesure *a priori* : le bench redistribue les 24
cartes uniformément, sans conditionner sur une enchère. Or l'enchère est de la
communication — passer, soutenir, surenchérir, tout cela renseigne. C'est là qu'est le
gisement, et il est déjà chiffré : playgen conditionné sur le préfixe d'enchère
resserre la dispersion a posteriori de **44,6 → 34,9 pts (−21,6 %)**, sans biais
([experiments/auction_conditioned_labels.md](experiments/auction_conditioned_labels.md)).
Au passage, ces 44,6 confirment indépendamment les 44,7 mesurés en §2.8.

**Ce qui distingue cette piste de l'échec consigné.** Ce document est un échec — les
modèles distillés perdent 3 à 7 pts — mais il portait sur l'usage du posterior comme
**cible**. L'utiliser comme **entrée** n'a jamais été tenté, et c'est le bon geste pour
un estimateur qu'on sait bon (RMSE −21,5 %, non biaisé) mais qu'on ne sait pas seuiller :
on donne l'information au réseau et on laisse le RL décider quoi en faire.

**Forme de la feature.** Surtout pas une moyenne de distributions : moyenner des
marginales par carte détruit les corrélations, et « soit Est a les atouts, soit Ouest »
est précisément ce qu'on cherche à capturer. La forme correcte est **une espérance de
scalaires évalués monde par monde** — chaque déroulement étant un tirage joint cohérent,
toute fonction évaluée dessus garde la structure. Candidats : `P(on finit preneur)`,
`P(le partenaire soutient)`, `P(les adversaires passent au-dessus)`,
`E[valeur du contrat final]`, `P(coinche)`. C'est le pendant correct de l'erreur de
Jensen consignée en §1.6.

**Déroulement d'enchère seule, pas de monde complet.** Le coût playgen est
proportionnel au nombre de tokens générés, et
`generate_deals_from_auction_scored` complète l'enchère **puis déroule le jeu des
acteurs cachés** pour révéler les cartes. S'arrêter à la triple passe supprime la
partie dominante : ~16 pas de décodage au lieu de 64. On perd les mains cachées, mais
on garde ce qu'on voulait — la distribution des issues d'enchère, qui *est* la
communication.

**Le coût, mesuré le 2026-08-02** — tableau complet dans
[BENCH.md](../BENCH.md#playgen-decode--prefill-v2-2026-08-02). Le fait qui décide :
**le prefill est par batch et quasi plat en taille de batch** (106,8 ms à B=1 →
221,7 ms à B=512), donc sur un déroulement court il domine le décodage. Précalculer la
feature sur 5M donnes × ~6 positions × 32 déroulements :

| stratégie | pool 5M | pool 1M |
|---|---|---|
| API actuelle (un préfixe par batch) | 1540 GPU-h — hors de portée | 308 GPU-h |
| batch inter-positions (changement de code) | 182 GPU-h | **36 GPU-h** |

**Repli si le précalcul reste trop cher : distiller la feature.** Un petit réseau
entraîné à prédire ces espérances depuis (main, préfixe d'enchère), puis interrogé à
l'entraînement. Même geste que `bid_belief_v4` — mais avec une cible qu'on a des
raisons de croire, ce qui répond à la faiblesse de ce dernier.

**Deux réserves.**
1. *Aucune information nouvelle au sens strict* : la croyance est une fonction du
   préfixe, que le réseau a déjà. C'est une aide de **représentation**. Le précédent
   qui la justifie est interne au dépôt : la sonde de couche cachée a montré que deux
   features pourtant dérivables de la main faisaient passer l'accord de 77 % à 97 %.
2. *Circularité.* Playgen est un clone de comportement de v6 : sa continuation prédit
   ce que **v6** annoncerait. En self-play, v7 s'en éloigne. Remède : itérer,
   comme triforge.

**Et une conséquence inconfortable pour §5.** Dans le corpus playgen, les tokens
d'enchère viennent de bid v6 et les tokens de jeu de DouDou. Un DouDou plus fort
améliore donc surtout la partie *jeu* du modèle — celle que le déroulement d'enchère
seule n'utilise pas. **Plus on rend cette feature bon marché, moins un meilleur DouDou
l'améliore** : les deux chantiers ne se renforcent pas autant qu'espéré, et un
réentraînement de DouDou redevient un sujet de force de jeu, pas de qualité d'annonce.

---

## 4. Outil — évaluer les annonces candidates par continuation d'enchère

**But.** Pour une main + un préfixe d'enchère donnés, estimer la valeur réelle de
chacune des annonces candidates (typiquement le top-5 des Q de v6, plus le capot),
en laissant l'enchère **se poursuivre** et la donne se jouer, sur N mondes playgen.

**Pourquoi ça n'est pas déjà ce que fait la page Annonces.** La page fixe la question
(« ce contrat, à ce seuil, passe-t-il ? ») et échantillonne les mondes. Ici on veut
comparer des *actions*, chacune suivie de sa propre continuation d'enchère — c'est-à-dire
exactement la cible de label validée en §1.5.

**Le biais d'Oracle ne se moyenne pas.** Point important, parce que l'intuition
contraire est naturelle : un solve DD est un *majorant* de ce qui est réalisable dans
ce monde (il voit les quatre mains). Moyenner 1000 majorants donne un majorant, pas
la vérité — l'espérance d'un max n'est pas le max d'une espérance. Augmenter le
nombre de mondes réduit la variance et **ne touche pas** ce biais. C'est déjà la
doctrine du dépôt (« DD oracle: training signal, not a player »). Donc :
- l'Oracle sert à **borner et à trier** (si le capot n'est pas atteignable en DD, la
  question est close) ;
- la **valeur** se lit sur la continuation jouée par un vrai joueur, jamais sur le DD.

**Ce qui existe déjà.** L'essentiel :
- `_playgen_gpu.auction_deals(dealer, hands, prior_pairs, seat, n, temp)` —
  échantillonnage de mondes **conditionné par le préfixe d'enchère** (sidecar GPU,
  lot unique) ; repli CPU `Analyst.auction_deals` (~0,4 s/monde).
- `_run_doudou_sim_with_hands(hands, bid_model, dmc_model, dealer, prior_actions)`
  ([server.py](../../python/colver/web/server.py)) — **rejoue le préfixe puis
  continue l'enchère au NN (`bid_a_dd`) et joue la donne**. C'est déjà une
  continuation complète.

**Fait** — [scripts/analysis/bid_candidates.py](../../scripts/analysis/bid_candidates.py),
sans serveur :

```bash
export COLVER_PLAYGEN_GPU_URL=http://localhost:8003   # avant l'import (cf. plus bas)
uv run python scripts/analysis/bid_candidates.py --hand "AD TD KD QD JD 9D 8D 7D" \
    --worlds 400 --top 4
```

Il force chaque candidate, laisse l'enchère se poursuivre au NN et la donne se jouer
par DouDou50 (le fichier que sert le web, via `colver.model_path()`), et rend par
candidate : écart de points marqués moyen/médian, écart **apparié** vs le top-1 de v6,
taux de prise, taux de réussite et contrat final le plus fréquent, avec le Q de v6 en
regard. Les candidates sont le top-Q de v6 ∪ {meilleur capot} ∪ {Passe}, plus ce que
`--add` demande.

Points de méthode qui portent le résultat :
- **pool de mondes partagé** par toutes les candidates, donc comparaison appariée
  (même principe que `world_total` dans `annonces_sim`) ;
- **le gain de l'appariement est modéré**, pas spectaculaire : forcer une annonce
  différente fait diverger l'enchère, donc une partie de la corrélation entre mondes
  est perdue. Mesuré : ±21 non apparié → ±16 apparié entre annonces voisines ;
- **garde-fou** : on vérifie que chaque monde contient bien la main du siège, sinon on
  s'arrête (un échantillonneur incohérent produirait des chiffres plausibles et faux) ;
- **découpage en lots de 512** — le sidecar plafonne à `max_worlds` et tronque
  silencieusement au-delà.

**Coût.** ~0,4 s/monde en **repli CPU** : 1000 mondes × 5 candidates ≈ 33 min par
main — inutilisable en boucle. Le sidecar GPU génère les mondes en un lot et ramène
ça à quelques minutes. **Le pool de mondes se génère une seule fois par (main,
préfixe)** et se réutilise pour toutes les candidates : c'est la continuation jouée,
pas la génération, qui domine ensuite.

**Le sidecar, en local.** La machine de dev a une RTX 4090 ; rien n'oblige à subir le
repli CPU. Il n'est simplement pas lancé (rien sur 8003, `COLVER_PLAYGEN_GPU_URL`
absente des profils shell et du `.env` du dépôt) :

```bash
CUDARC_CUDA_VERSION=13010 cargo build --release --bin playgen_gpu_server --features gpu_server
./target/release/playgen_gpu_server --playgen models/playgen/playgen_v2_final.bin --port 8003 &
export COLVER_PLAYGEN_GPU_URL=http://localhost:8003
```

⚠️ **La variable est lue à l'import du module**, pas à l'appel :
`playgen_gpu.GPU_URL` et `agents.SIDECAR_URL` sont des constantes de module. Elle
doit être dans l'environnement **avant** le démarrage du process — l'exporter ensuite
ne fait rien, et le repli uniforme est silencieux pour qui ne lit pas `probe()`.

---

## 5. Ordre de travail proposé

1. ~~§4 — l'évaluateur de candidates~~ **fait** (2026-08-02).
2. ~~§2.2 — le Q plat~~ **fermée** (2026-08-02, §1.7) : c'est le jeu qui est plat, et
   v6 ordonne juste (r = +0,50). §3.1 reste à faire mais pour ~2 pts/donne, pas plus.
   *Suite naturelle* : refaire la mesure **avec préfixe d'enchère**, où les premiers
   indices montrent des écarts bien plus larges (−86 ± 25 sur une 3ᵉ candidate).
3. §3.2 — la suite de sondes. Immédiat, sans risque, et c'est le garde-fou qui manquait.
4. §3.1 — canonicalisation de l'obs, avec le flip rate comme test de non-régression.
5. §3.4 — les deux features du probe, à intégrer avant de figer l'obs v7.
6. ~~§2.8 — arbitrer donnes contre mondes~~ **tranché** (2026-08-02) : 5M × 20, les
   donnes gagnent. Reste à décider 5M contre 1M à k identique — 87 h contre 17 h.
7. Regénération du pool (due de toute façon) avec §3.5 dedans.
8. §3.3 — actions mortes, une fois qu'on a une sonde qui mesure le progrès.

---

## Reproduction

**Le sidecar playgen doit tourner** pour tout ce qui échantillonne des mondes :
`playgen-up` avant, **`playgen-down` après** (5,5 Go de VRAM résidents tant qu'il vit,
il n'y a pas de libération à l'inactivité). Cf. CLAUDE.md, « Playgen sidecar discipline ».

Versés dans `scripts/analysis/` :
- [bid_candidates.py](../../scripts/analysis/bid_candidates.py) — §1.3, §4 ;
- [bid_q_flatness.py](../../scripts/analysis/bid_q_flatness.py) — §1.7 ;
- [bid_equivariance.py](../../scripts/analysis/bid_equivariance.py) — §1.1-1.2 : taux de
  bascule avec contrôles heuristiques, échelle des Q, erreur d'équivariance du vecteur Q ;
- [bid_capot_probe.py](../../scripts/analysis/bid_capot_probe.py) — §1.3 : fréquence du
  capot sur N enchères auto-jouées, sondes capot forcé, rareté des familles ;
- [hand_classes.py](../../scripts/analysis/hand_classes.py) — §1.4 : Burnside vérifié par
  force brute (`--verify`), et concentration des codes de main ;
- [card_importance.py](../../scripts/analysis/card_importance.py) — ce que vaut chaque
  carte, mesuré au solve apparié ; justifie le contenu de `HandCode`.

*Note de reproductibilité* : le « 97,8 % des positions sous 0,03 » de §1.1 est une
statistique d'échantillon. `bid_equivariance.py` avec sa graine par défaut rend 96,5 % —
même quantité, tirage différent. Les autres chiffres de §1.1 se reproduisent au centième.
