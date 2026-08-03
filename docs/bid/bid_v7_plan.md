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

⚠️ **Tous les chiffres de cette section portent sur l'annonce d'ouverture** — `flip_rate`,
`q_scale` et `q_equivariance` mesurent après `redeal_with_hands`, sans aucun `step`. Le
taux de bascule et la marge changent beaucoup avec le préfixe d'enchère, et **le coût
attendu ne se lit sur aucun des deux pris seul** : §1.7.

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

**Et « 20 » n'a jamais été le nombre de mondes** (mesuré le 2026-08-03,
[isdd_worlds_per_budget.py](../../scripts/analysis/isdd_worlds_per_budget.py)).
`enrich_pool_isdd` tourne en **mode temps** à 20 ms/coup, et en mode temps la boucle
d'IS-DD ne sort que sur l'échéance : `determinizations` est inatteignable. Le nombre réel,
médiane par pli en régime séquentiel — celui du pool :

| pli | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| mondes | **2** | 4 | 14 | 54 | 314 | 1 052 | 5 615 |

Au pli 1 un monde **est** une donne complète, donc 20 ms achètent **deux échantillons** —
d'un postérieur dont un tiers des mondes s'écarte de plus de 40 points, comme le dit le
tableau juste au-dessus. C'est la profondeur derrière `scores_isdd_5M.sc`, et c'est un
levier bien plus gros que tout ce qui reste dans le solveur. Détail et le second effet —
`parallel` change l'agent et pas seulement sa vitesse — dans
[../play/is_dd.md](../play/is_dd.md#worlds-per-budget-measured-2026-08-03).

*Conséquence sur la mesure de péremption du pool* : le bras B0 tournait en mode **compte**
à 20 mondes, donc ~10× plus profond qu'avril au pli 1. L'écart de plancher de bruit entre
les deux est plus grand qu'estimé, ce qui **renforce** le « au moins 87 % » de
[../data_gen/pool_staleness.md](../data_gen/pool_staleness.md) sans rien changer au verdict.

### 1.6 Acquis antérieurs qui contraignent v7

Établis avant ce document, à ne pas re-litiger :

- **Le label doit être une continuation d'enchère, pas une valeur de contrat.**
  Distiller « valeur de jouer ce contrat » fait sur-annoncer : 47,3 % puis 43-44 %
  vs v6. Le pipeline est réutilisable, c'est la cible qui est fausse.
  [experiments/auction_conditioned_labels.md](experiments/auction_conditioned_labels.md)
  **Ligne close le 2026-08-02** : la dernière hypothèse ouverte — rescorer le contrat
  final par un vrai déroulement au lieu de la table `P(isdd | dd)` — ne vaut pas le
  coup. La table est ajustée sur des labels qui, remesurés aujourd'hui, ne bougent que
  de 9,3 pts sur 24 de bruit. Ne pas rouvrir sans une idée neuve sur **la cible**, pas
  sur son scoring.
- **Le pool n'est pas à regénérer pour cause de dérive d'IS-DD** (mesuré 2026-08-02,
  1000 donnes × 4 couleurs) : décalage moyen nul, excès sur le plancher de bruit = 12,8 %
  de la variance d'un label, contre ~117 GPU-jours à l'échelle 5M. Le motif de
  regénération est un **changement de format**, pas la fraîcheur des chiffres — ce qui en
  fait une décision de §5 (périmètre de v7) et non un préalable.
  Corollaire méthodologique à ne pas réapprendre : **un label DD symétrique ne peut pas
  voir la force de jeu**, les points cartes N-S étant à somme constante.
  [../data_gen/pool_staleness.md](../data_gen/pool_staleness.md)
- **Jamais de E[Y|X] substitué par échantillon avant un seuil** (erreur de Jensen) :
  on calibre en échantillonnant.
- **Le 77 % → 97 % de la sonde mesure le *distillat*, pas le réseau** (relu le
  2026-08-02, §3.4). J/9 par couleur ne manquait qu'à XGBoost : `obs[0:32]` est la main
  brute. Seul `opp_best_other_ts` — un max sur les couleurs privé de celle qu'un
  adversaire annonce — est une information que le réseau ne lisait pas directement.
  [interpretability/probe_morning_report.md](interpretability/probe_morning_report.md)
- **Le modèle joue ses propres enchères** (ε-greedy sur sa politique) ; l'oracle
  supervise la loss, jamais la trajectoire. Sinon on obtient des enchères dégénérées
  (annonce optimale → 3 passes) qui n'enseignent rien de la communication.
- **Une main seule n'explique que ~17 % de la variance de l'issue DD.** Le reste est
  dans l'historique d'enchères. Plafond structurel de toute approche « main → annonce ».
- **v6 bat v5** 55,8 % (jeu DMC) / 57,3 % +181 (jeu IS-DD), mais perd -16 à -26
  pts/donne à chaque sonde de score : le paradoxe donne/match n'est pas expliqué.

### 1.7 La platitude dépend du **type de décision**, pas de la profondeur d'enchère

*Mesuré le 2026-08-02. Platitude : [bid_q_flatness.py](../../scripts/analysis/bid_q_flatness.py),
120 mains × 300 mondes playgen × 3 candidates = 108 000 déroulements **par régime**,
graine commune donc **mains appariées d'un régime à l'autre**. Bascules :
[bid_equivariance.py](../../scripts/analysis/bid_equivariance.py) `--prior`, 400 donnes
× 23 permutations.*

Pour chaque main : la valeur réelle du top-1 de v6, celle de son top-2, et celle de sa
**8ᵉ** annonce comme contrôle positif. Trois régimes, définis par le préfixe d'enchère —
le siège qui décide est toujours `len(prior)` crans après le premier parleur, donc la
longueur du préfixe suffit à les nommer :

| régime | Δréel(top2 − top1) | contrôle positif (8ᵉ) | top1 > top2 | r(ΔQ, Δréel) | ΔQ médian |
|---|---|---|---|---|---|
| **ouverture** | **−8,94 ± 2,12** | −111,7 ± 3,8 | 65 % | +0,509 | 0,0042 |
| **adversaire a ouvert 100♣** | **−74,76 ± 8,75** | −210,0 ± 10,7 | 81 % | +0,708 | 0,0423 |
| **partenaire a ouvert 100♣** | **−1,03 ± 2,94** | −338,0 ± 8,6 | 53 % | +0,111 | 0,0150 |

La première ligne **reproduit la mesure à 40 mains** (−8,20 ± 4,31, contrôle −114,5,
r = +0,504) à trois fois l'échantillon : le résultat d'ouverture tient.

**Ce n'est pas la profondeur de l'enchère qui compte, c'est le type de décision.**
*Contester* l'ouverture adverse met 75 points en jeu, 8,4× l'ouverture, et c'est là que
v6 ordonne le mieux. *Soutenir* son partenaire est au contraire parfaitement plat :
top-1 et top-2 sont interchangeables à 1 point près, 53 % — pile ou face. Le r = +0,111
n'y est pas un défaut de v6 : **on n'ordonne pas ce qui est identique**. Et le contrôle
positif y est le plus large des trois (−338), donc la décision compte énormément dans
l'absolu — c'est seulement son *sommet* qui est plat.

#### Ce que ça change pour §3.1, et pourquoi le taux de bascule seul induit en erreur

Les 24,6 % de §1.1 sont mesurés **à l'ouverture** (`flip_rate` ne joue aucun `step`).
Mesurés par régime, taux de bascule et coût d'une bascule varient **en sens inverse** :

| régime | bascules v6 | bruit / marge | coût d'une bascule | **coût attendu** |
|---|---|---|---|---|
| ouverture | **24,58 %** | 8,8× | 8,94 pts | **2,2 pts** |
| adversaire a ouvert | **5,24 %** | 0,9× | 74,76 pts | **3,9 pts** |
| partenaire a ouvert | **19,88 %** | 2,9× | 1,03 pt | **0,2 pt** |

**Le régime le plus coûteux est celui qui bascule le moins.** Contester ne bascule que
5 % du temps — la marge y est 10× plus large, le bruit de symétrie ne la franchit
presque plus — mais chaque bascule coûte 75 points. À l'inverse, les 19,9 % de bascules
du régime « soutien » ne coûtent rien, puisqu'elles permutent des options équivalentes.
**Un taux de bascule seul ne dit donc rien du coût** : c'est le produit qui compte, et
c'est le régime jamais mesuré jusqu'ici qui domine.

Conséquences :
- la canonicalisation vaut **plus** que les ~2 pts/décision estimés d'après l'ouverture,
  et son gisement est le régime de **contestation**, pas l'ouverture ;
- l'ordre de grandeur reste néanmoins de quelques points par décision, donc **l'arène
  ne pourra pas trancher §2.1** — cf. la question de puissance ouverte en §2.9 ;
- l'erreur d'équivariance du vecteur Q, elle, est **stable** (0,037 / 0,040 / 0,043,
  soit ~5 % de l'étendue partout) : c'est bien la marge qui bouge, pas le bruit.

*Deux limites à ne pas oublier.* (1) `bascules × Δ(top1,top2)` est un **minorant** : une
bascule ne va pas toujours vers le top-2. (2) Le *coût* d'une bascule n'est mesuré qu'à
**un seul niveau d'enchère** (100) ; le taux de bascule, lui, est cartographié sur 20
régimes en §1.8.

#### Le taux de bascule, cartographié — l'ouverture est un plateau

*[bid_margin_sweep.py](../../scripts/analysis/bid_margin_sweep.py), 20 régimes × 400
donnes × 23 permutations, **17 s au total**.* L'asymétrie de coût est le point de
méthode : la marge coûte 0,8 s par régime, le coût d'une bascule 13 à 25 min. On balaie
donc large pour presque rien, et on ne dépense le GPU que là où la carte le justifie.

| régime | bascules | ctrl | marge | bruit/marge |
|---|---|---|---|---|
| ouverture (1er / 2e / 3e / 4e de parole) | **24,6 / 25,4 / 26,2 / 24,6 %** | ≤3,1 % | 0,0042-0,0069 | 7,1-8,8× |
| contestation 100 / 110 / 130 / 150 | 5,2 / 6,5 / **1,0 / 0,00 %** | ≤0,3 % | 0,042 → 0,136 | 0,9 → 0,3× |
| contestation 100, couleurs ♠/♥/♦ | 4,8 / 3,8 / 5,2 % | ≤0,3 % | ~0,044 | ~0,9× |
| soutien 100 / 130 | 19,9 / 3,9 % | ≤0,3 % | 0,015 / 0,089 | 2,9 / 0,9× |
| 2e tour (part. nous relance à 130) | 2,1 % | 0,00 % | 0,111 | 0,5× |

- **L'ouverture est un plateau, pas un point** : les quatre positions de parole donnent
  toutes ~25 % et une marge de 0,004-0,007. Les 24,6 % de §1.1 caractérisent le régime
  entier.
- **Une annonce sur la table change tout** : 0-10 % au lieu de ~25 %.
- **Plus l'enchère est haute, plus v6 est décidé** — jusqu'à **0 bascule sur 9200** à 150.
- **La couleur ne change rien** à niveau égal : le taux est une propriété du *niveau*.
- Les contrôles heuristiques restent ≤3,1 % partout, ce qui valide la permutation du
  préfixe (`apply_prior`) dans les 20 régimes.

### 1.8 Ce que l'arène peut voir — et la conversion en points par donne

*[arena_power.py](../../scripts/analysis/arena_power.py), 2000 donnes jouées (enchère v6,
jeu DouDou50) puis 20 000 matchs simulés par point de la courbe.*

**Fréquence des régimes** — 16 055 décisions réelles, **8,0 par donne** (passes forcées
exclues, les quatre sièges) :

| régime | part | par donne | niveaux dominants |
|---|---|---|---|
| **contestation** | **57,7 %** | 4,63 | 110→23 %, 120→19 %, 90→17 %, 100→17 % |
| soutien | 23,6 % | 1,89 | |
| ouverture | 16,0 % | 1,28 | |
| notre enchère | 2,8 % | 0,22 | |

**Le format** : σ = **314,9 pts** d'écart par donne, et une partie ne dure que
**10,4 donnes**. C'est court, et c'est ce qui rend la variance dominante — cf. §2.6.

**Seuil de détectabilité** (borne optimiste = duplicate matching parfait, pessimiste =
donnes indépendantes ; l'arène duplique, donc la vérité penche vers l'optimiste) :

| matchs | taux requis | δ détectable |
|---|---|---|
| 1 000 | ≥ 53,10 % | **4 à 10** pts/donne |
| 2 000 | ≥ 52,19 % | 2 à 10 |
| 5 000 | ≥ 51,39 % | 1 à 4 |
| 10 000 | ≥ 50,98 % | 1 à 2 |

*Contrôle* : à δ = 0 la simulation rend **50,00 %**. Une première version reversait tout
l'écart d'une donne au camp gagnant et rendait 58 % — en contrée les deux camps marquent,
la course à 2000 ne se déduit pas de la seule différence. Le pool est en outre
**symétrisé** (chaque donne entre aussi retournée), sans quoi l'asymétrie d'un
échantillon fini se propage dans la courbe.

#### La conversion, et une estimation antérieure à corriger

En pondérant §1.7 par ces fréquences, et en ne comptant que les décisions d'**une seule
équipe** (2 sièges sur 4, soit ~4 décisions par donne) :

| régime | décisions/donne/équipe | × bascules | × coût | = pts/donne |
|---|---|---|---|---|
| contestation | 2,32 | 5,24 % | 74,76 | **9,1** |
| ouverture | 0,64 | 24,58 % | 8,94 | 1,4 |
| soutien | 0,95 | 19,88 % | 1,03 | 0,2 |
| | | | **total** | **~10,7** |

**Ceci corrige l'estimation « ~2 pts/donne » de §1.7**, qui ne portait que sur
l'ouverture et confondait points par *décision* et par *donne*. À ~10,7 on est
**au-dessus** du seuil de 1000 matchs, pas en dessous : l'arène peut voir §2.1.

*Deux réserves qui empêchent d'en faire une promesse.* (1) ~10,7 est la valeur **en
jeu**, pas le gain : la canonicalisation achète de la *cohérence*, pas de la *justesse*.
Le gain réel vient de ce que les 24 étiquetages d'une classe fusionnent, donc que
l'entraînement voit 24× plus de signal — un effet d'apprentissage qu'aucune mesure sur v6
ne peut prédire. (2) La contestation pèse 85 % du total et son coût ne vient que du
niveau 100, alors que 90-120 dominent.

### 1.9 Il n'y a pas de paradoxe donne/match — l'audit de 2026-04-26 mesurait autre chose

*[bid_ev_by_score.py](../../scripts/analysis/bid_ev_by_score.py), 2000 donnes par sonde,
chacune jouée **deux fois, v6 dans un camp puis dans l'autre à score miroir** — un
appariement qui annule l'asymétrie de siège et de donneur. Les deux specs ne diffèrent
que par le modèle d'annonce (même DouDou50 au jeu), donc l'écart est bien celui du bidder.*

| score (v6 − adv) | Δ deal-EV de v6 | |
|---|---|---|
| **0 - 0** | **+12,00 ± 3,71** | +3,2 σ — la sonde qui décide |
| 900 - 200 | +22,88 ± 3,99 | +5,7 σ |
| 200 - 900 | +5,65 ± 4,10 | +1,4 σ |
| 1500 - 1000 | +16,54 ± 3,76 | +4,4 σ |
| 1000 - 1500 | +17,10 ± 3,80 | +4,5 σ |
| 1700 - 1700 | +10,60 ± 4,55 | +2,3 σ |

**v6 est meilleur en points par donne, à tous les scores** — l'inverse de ce que le
dossier affirmait (« −16 à −26 pts/donne à chaque sonde », 2026-04-26).

**Le contrôle qui tranche entre les deux mesures.** L'écart mesuré doit prédire le taux
de matchs connu. Via la courbe de §1.8 : +14,1 pts/donne en moyenne → **~55-56 %**, à
comparer aux **55,8 %** (jeu DMC) et **57,3 %** (jeu IS-DD) enregistrés en arène. Sous
l'ancien chiffre de −22, la même courbe prédirait **~41 %**, c'est-à-dire que v6
*perdrait* ses matchs. C'était exactement ça, le « paradoxe » : deux mesures dont une
seule est compatible avec l'arène.

**Pourquoi l'ancien chiffre.** L'audit différenciait **deux « nets » mesurés séparément**
(v5 +57,6, v6 +35,2) au lieu d'un face-à-face apparié. Une telle métrique pénalise
l'agressivité par construction : un bidder qui prend plus de contrats marginaux mais
rentables baisse sa moyenne par donne déclarée tout en gagnant plus. C'est précisément ce
que la note d'origine observait sans en tirer la conséquence — v6 « annonce 140-160 trois
fois plus souvent » et chute plus (45 % contre 38 %). Le chiffre est en outre antérieur à
quatre ruptures (solveur 07-23, base 162 le 07-28, fin de l'arrondi le 07-31, règle
d'atout le 08-01), même si leur effet documenté sur un score de donne est petit (~2,4 pts).

*Réserve* : le code de l'audit de 2026-04-26 n'a pas été relu, donc ce qu'il appelait
« net » est reconstitué à partir de sa note. Ce qui est établi, c'est que les deux
protocoles diffèrent et qu'**un seul des deux est compatible avec l'arène**.

**Ce que ça ferme.** §2.6 était le seul bloqueur explicite sur le choix de reward.
Δ-winprob ne sacrifie rien en deal-EV : v6 domine v5 sur les deux tableaux. Donc (1) v7
garde la reward sans arbitrage à faire, et (2) **le pts/donne redevient un diagnostic
valide** — la doctrine « le match-arena est la vérité, point » de la note d'origine était
une conséquence de l'artefact, pas du modèle.

*Corollaire méthodologique* : ne jamais comparer deux bidders par des moyennes mesurées
séparément. Un face-à-face apparié sur les mêmes donnes, camps échangés, coûte le même
temps de calcul et ne peut pas produire ce genre d'illusion.

---

### 1.10 L'équivariance de v6 **oscille** d'un checkpoint à l'autre — et ça condamne l'ablation à budget réduit

*Mesuré le 2026-08-03, [bid_checkpoint_ladder.py](../../scripts/analysis/bid_checkpoint_ladder.py),
les 13 checkpoints du resume de v6 × 3 régimes × 300 donnes × 23 permutations (6 900
comparaisons par case, donnes **appariées** d'un checkpoint à l'autre).*

C'était une calibration de puissance, pas une mesure de v6 : avant de dépenser des runs
d'ablation à budget réduit, savoir si un checkpoint à 10 M range les bras comme un
checkpoint à 30 M. Réponse : **non, et pas non plus à 30 M**.

| pas | 2,5M | 5M | 7,5M | 10M | 12,5M | 15M | 17,5M | 20M | 22,5M | 25M | 27,5M | 30M |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ouverture | 40,6 | 26,0 | 35,5 | 24,5 | 38,3 | 35,7 | **22,4** | **34,9** | 25,5 | 28,2 | 27,0 | 25,3 |
| contestation | 10,3 | 11,4 | 8,1 | 12,3 | 8,3 | 6,6 | 6,2 | 5,1 | 6,0 | 8,8 | 5,6 | 4,7 |
| soutien | 15,4 | 15,0 | 18,0 | 16,1 | 17,9 | 22,7 | 15,3 | **29,1** | 12,5 | 12,4 | 16,2 | 20,4 |

L'erreur type binomiale d'une case est de **0,5 pt**, et les donnes sont les mêmes partout.
Les écarts de 12,5 pts entre 17,5 M et 20 M — **en fin d'entraînement** — ne sont donc pas du
bruit de mesure : c'est le réseau qui bouge.

**Le mécanisme est déjà en §1.2, il suffit de le lire dans l'autre sens.** Le taux de bascule
compte de quel côté d'une arête tombent des ex æquo dont la marge médiane est 0,0044, quand
l'erreur d'équivariance vaut 9,5× cette marge. Il mesure donc la position d'un couteau, pas
une propriété stable du modèle. §1.1 et §1.2 ne sont pas deux constats, c'est le même.

**Trois conséquences, dans l'ordre de leur coût.**

1. **Aucune comparaison entre bras ne se lit sur un checkpoint.** Un bras s'évalue sur la
   **moyenne de ses derniers checkpoints** — ils sont déjà écrits tous les 2,5 M, donc c'est
   gratuit. C'est la version mesurée de la leçon « un pic d'éval isolé n'est pas un signal ».
2. **Une ablation à budget réduit biaise *dans le sens de l'hypothèse*.** En moyennant par
   fenêtres pour sortir de l'oscillation, la fenêtre 7,5-15 M donne 33,5 % à l'ouverture contre
   26,5 % pour 22,5-30 M : le bras physique **achète encore de la symétrie avec ses pas** à
   10 M et a fini à 30 M. Comparer canonique et physique à 10 M crédite donc la
   canonicalisation d'un écart qui se serait refermé tout seul. C'est le pire cas possible —
   un biais qui a le signe de ce qu'on veut montrer.
3. **Le taux de bascule n'est pas un juge pour v7**, il est nul par construction sur le bras
   canonique. Il reste un **contrôle de câblage** (un bras canonique qui ne rend pas 0 est mal
   branché), pas une mesure de force.

*Et ça renforce §3.1 par un chemin qu'on n'avait pas* : la symétrie n'est pas seulement mal
apprise, elle n'est **jamais fixée** — 75 M de pas plus tard elle vagabonde encore de ±8 pts.
Ce n'est pas un entraînement trop court.

## 2. Ce qui reste à prouver

Chaque question est formulée pour pouvoir tomber.

**2.1 — Canonicaliser l'obs d'annonce améliore-t-il la force, ou seulement la cohérence ?**
L'erreur d'équivariance est grande *relativement à la marge*, mais petite en absolu
(4,7 % de l'étendue). §1.7 a depuis chiffré le coût attendu par régime — **3,9 pts par
décision en contestation, 2,2 à l'ouverture, 0,2 en soutien** — donc l'effet existe et
se concentre là où on ne le cherchait pas.
*Test* : entraîner v7 à budget identique avec et sans canonicalisation.
*Critère* : flip rate mesuré à 0 % **dans les trois régimes** (non-régression, gratuit),
plus une sonde appariée en pts/décision par régime via l'outil de §4.
⚠️ **Le critère « ≥ 52 % en h2h 1000 matchs » a d'abord été retiré ici, puis rétabli.**
Le retrait supposait « quelques points par décision, donc sous l'erreur type ». §1.8 a
depuis fait la conversion : l'effet se chiffre à **~10,7 pts/donne**, contre un seuil de
détectabilité de **4 à 10 pts/donne à 1000 matchs**. C'est donc à la limite basse du
visible — *critère retenu* : **h2h 2000 matchs** (seuil 2-10), l'arène en confirmation et
la sonde appariée par régime en mesure principale. Le premier raisonnement confondait
points par *décision* et par *donne*.
*Implémentation* : **faite le 2026-08-02** (`write_bid_observation_canonical`,
`--canonical`, `canonical = true` dans le TOML). Reste l'entraînement.

**2.2 — Le Q plat est-il un défaut de calibration ou la réalité du jeu ?**
**Fermée le 2026-08-02 : c'est le jeu, et v6 ordonne juste** — mais seulement à
l'ouverture et en soutien ; en contestation le jeu n'est pas plat du tout (75 pts) et
v6 y ordonne encore mieux. Voir §1.7.

**2.3 — Réveiller le capot rapporte-t-il quelque chose en arène ?**
Les capots forcés sont à ~10⁻⁴, et §1.3 a depuis chiffré ce qui s'y perd : **585 points
par donne** — pas les 110-200 estimés à la main avant mesure — soit **~0,06 pt/donne**
en espérance. *Test* : v7 avec capot réveillé vs v7 sans, h2h. *Attente honnête* : indétectable en % de matchs, visible en
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
**Fermée le 2026-08-02 : il n'y a pas de paradoxe.** v6 est *meilleur* en deal-EV que
v5, à tous les scores. Voir §1.9. Conséquence directe : **la reward n'est pas un
bloqueur pour v7** et le pts/donne reste un diagnostic valide.

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

⚠️ **Le `k` de ce tableau est un paramètre d'allocation, pas le `k` du pool actuel.**
Mesuré le 2026-08-03, `scores_isdd_5M.sc` tourne à k = 2 au pli 1 et k > 5 000 au pli 7
(§1.5) : le budget est une échéance, jamais un compte. L'arbitrage ci-dessous — donnes
contre mondes à `N × k` fixé — n'en est pas invalidé, il porte sur ce qu'on **choisirait**.
Mais toute lecture de « le pool a k = 20 » l'est.

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

**2.9 — Que peut réellement résoudre l'arène, et à partir de quel écart ?**
*Ouverte le 2026-08-02.* Plusieurs questions de cette section posent un critère en % de
matchs sans avoir vérifié que l'instrument peut l'atteindre. Ordre de grandeur : quelques
points par donne cumulés sur une partie à 2000 pèsent ~1 % de score, soit ~51 % de
matchs, quand 1000 matchs ont déjà ~1,6 pp d'erreur type. **Un critère à 52 % est alors
un test de la puissance de l'arène, pas de l'hypothèse.**
*Test* : une table de conversion pts/donne → % de matchs (par simulation sur des scores
de donne déjà enregistrés, sans réentraîner quoi que ce soit), et le seuil de
détectabilité à 1000 / 2000 / 5000 matchs.
*Conséquence attendue* : réaffecter chaque question de §2 à l'instrument qui a la
puissance voulue — sonde appariée en pts/décision pour §2.1 et §2.3, arène réservée aux
effets de plus de ~4-5 pp. Le réflexe existe déjà dans ce document sous forme de
**contrôle positif** (§1.7) ; il s'agit de l'étendre aux critères d'arène.

---

## 3. Pistes d'amélioration

Par rapport valeur/coût décroissant.

### 3.1 Canonicaliser l'obs d'annonce *(coût faible, effet structurel)*
**Implémentée le 2026-08-02.** Comme l'obs de jeu 411 (`canonical_play_order`), à ceci
près qu'avant l'enchère il n'y a pas d'atout pour ancrer l'ordre : les couleurs sont
triées par (longueur, motif de rangs) décroissant. Espace d'entrée effectif ÷ ~22,
équivariance gratuite, et disparition d'une source de bruit qui vaut 8,8× la marge de
décision à l'ouverture.

Ce que l'écriture a appris, et qui ne se voyait pas depuis le plan :

- **Le départage des ex æquo doit lire l'enchère.** 7,5 % des mains ont deux couleurs de
  lanes identiques. Départager par indice physique semble inoffensif — mêmes lanes, même
  bloc main — et ne l'est pas : si l'enchère nomme une des deux, un renommage déplace
  cette mention vers l'autre et les deux positions, qui sont le même problème, cessent de
  se canonicaliser pareil. Le départage se fait donc par la valeur la plus haute annoncée
  dans la couleur, puis le slot le plus précoce ; l'indice physique n'est plus qu'un
  dernier recours, atteint seulement quand les deux couleurs sont indiscernables dans
  *toute* l'observation. Trouvé par le test d'invariance sur 40 donnes × 24 permutations,
  pas à la lecture.
- **Un réseau canonique pèse exactement le même nombre d'octets qu'un réseau physique**,
  donc rien ne le distingue à l'ouverture du fichier. D'où un drapeau explicite des deux
  côtés (`--canonical`, `canonical = true`) — se tromper est silencieux et rend une
  annonce légale dans la mauvaise couleur, exactement le piège
  `cardset_to_canonical` / `card_to_physical` du côté jeu.
- **La canonicalisation *remplace* l'augmentation de couleurs**, elle ne s'y ajoute pas :
  la forme canonique est déjà invariante, donc permuter un échantillon ne ferait que
  décorréler son obs de son action. `--canonical` coupe `augment_bid_batch`.
*Effet secondaire utile* : rend la §3.6 bien définie.
*Ce qu'on en attend, depuis §1.7* : le gisement n'est pas l'ouverture (2,2 pts/décision)
mais la **contestation** (3,9 pts/décision), où le bruit ne franchit la marge que 5 % du
temps mais coûte 75 points quand il la franchit. À vérifier régime par régime, pas en
moyenne — et pas à l'arène (§2.9).

### 3.2 Suite de sondes stratifiée, en **évaluation** *(coût nul, risque nul)*
**Faite le 2026-08-02** — [bid_probes.py](../../scripts/analysis/bid_probes.py), 9 familles
construites × 3 régimes × 200 mains, **0,4 s**, aucune donne jouée, aucun monde échantillonné.
Référence v6 figée dans [../measurements/bid_probes_v6.json](../measurements/bid_probes_v6.json) ;
`--baseline` diffe un checkpoint contre elle. À faire tourner à chaque checkpoint, comme un test.

Les familles sont **construites** et non tirées, pour deux raisons : celles qui décident sont
à ~10⁻⁴ (on ne les verrait jamais), et une famille construite est *identique* d'un checkpoint
à l'autre, donc deux runs se comparent directement.

**Une seule assertion est dure**, parce qu'une seule réponse est prouvable : huit cartes d'une
même couleur prennent les huit levées quoi qu'il arrive, et un capot réussi marque 502 contre
412 pour un 160 tous plis. **v6 la rate à 100 % dans les trois régimes** (exit 1) — c'est §1.3,
désormais sous test. Il n'y a volontairement pas de drapeau pour la taire.

Ce que la sonde a immédiatement ajouté :

| observation | régime | lecture |
|---|---|---|
| `main_pauvre` annonce **80♠ dans 14 %** | ouverture | 8 cartes prises parmi les douze 7-8-9 du paquet — il n'y a rien à annoncer |
| `main_pauvre` **relance à 120♣ dans 18 %** | soutien | pire : c'est une relance au-dessus du partenaire, sans une seule carte à points |
| `belote_seche` passe à 95-100 % | tous | correct — la belote ne rachète pas une main faible |
| `quatre_as` passe à 35 % / 77 % | ouv. / cont. | pas d'anomalie visible |
| marge médiane **0,004 à ouverture, 0,03-0,10 en contestation** | | reproduit §1.7 sur une population construite et non tirée |

**Équivariance, cas exact et sans échantillonnage.** Les quatre mains de huit atouts sont des
permutations de couleur exactes les unes des autres : les quatre réponses *doivent* être la
même annonce. Ramenées dans un repère commun :

| régime | les 4 réponses | étendue |
|---|---|---|
| ouverture | 140 · 140 · 140 · **150** | 10 pts |
| contestation | 150 · 150 · **160** · **160** | 10 pts |
| soutien | **160** · 140 · 140 · **160** | 20 pts |

⚠️ **Le préfixe doit être permuté avec la main.** Une première version ne permutait que la main
et sortait une étendue de **160 pts** en soutien, « Passe sur huit atouts » compris — artefact :
la main ♣ était la seule dont la longue était la couleur que le partenaire venait d'annoncer,
donc les quatre positions n'étaient pas les mêmes. Même piège qu'`apply_prior` dans
[bid_equivariance.py](../../scripts/analysis/bid_equivariance.py). Le chiffre honnête est
10-20 pts, et il reste une violation exacte sur le cas le plus trivial du jeu.

### 3.3 Réveiller les actions mortes *(coût moyen)*
ε-greedy ne suffit pas : à 10⁻⁴ de fréquence et 43 actions, l'action capot ne reçoit
essentiellement aucun gradient. Options, à tester dans cet ordre :
- forcer une proportion plancher de transitions capot dans le replay (avec poids) ;
- bonus d'exploration par compte d'action (UCB-like) sur la politique de collecte ;
- amorçage supervisé : sur les mains capot forcé du pool, un lot de labels durs.

### 3.4 Les features du probe — **une seule survit à la relecture** *(fait le 2026-08-02)*

Ce paragraphe disait : « J/9 par couleur et `opp_best_other_ts` : 77 % → 97 % d'accord sur
la sonde. C'est le seul lead de ce document dont l'effet est *déjà* chiffré. » **Ce n'est
pas ce que la mesure dit.**

Le 77 % → 97 % est le gain d'un **XGBoost qui distille le réseau** à partir de 17 features
de main agrégées. Ce n'est pas une mesure sur le réseau, c'est une mesure sur le
*substitut*. Le rapport le dit lui-même : « le probe linéaire h0 disait que l'info existait
dans l'obs », et une régression linéaire sur les 512 activations de la couche 0 atteint
93 % là où XGBoost plafonne à 77 %.

Conséquences, séparément pour les deux familles :

- **J/9 par couleur : rien à ajouter.** `obs[0:32]` est la **main brute** — J♠ est
  l'indice 3, 9♠ l'indice 2. Les huit bits sont déjà là, exactement. Les rajouter en
  features, c'est recopier huit entrées du réseau à côté d'elles-mêmes. Ils manquaient au
  jeu de features du distillat, jamais au réseau.
- **`opp_best_other_ts` : à garder.** C'est un max sur les couleurs **privé de celle qu'un
  adversaire annonce** — une interaction main × historique d'enchère, pas une reformulation
  de la main. C'est aussi ce que l'ablation du rapport isole : ajouter les 12 features
  per-suit laissait opp80 à 82 %, l'exclusion seule l'a porté à 97 %. Le réseau sait la
  calculer (h0 le prouve) ; la lui donner explicitement est le même geste que les 5
  features de score dérivées de v5, qui sont elles aussi dérivables et ont été ajoutées.

**Implémenté** : `BID_OBS_DIM_V7 = 123`, drapeau `--sa-features-v7`, clé TOML inchangée
(la largeur, elle, se détecte au fichier de poids). Layout :

| indices | contenu | sous renommage |
|---|---|---|
| [117:121] | `evaluate_for_trump(main, couleur)/35` | **permute** |
| [121] | `opp_best_other_ts` | invariant |
| [122] | `opp_second_other_ts` | invariant |

Les 4 scores par couleur sont gardés malgré le point précédent, pour une raison qui n'est
pas celle du rapport : la sonde a trouvé dans la dernière couche de v5 **quatre détecteurs
de qualité-couleur en parallèle** plutôt qu'un agrégat au meilleur. Les exposer nomme ce
que le réseau construit déjà, et rend l'exclusion de [121] lisible comme un max sur trois
d'entre eux au lieu d'une fonction du bitmap. C'est un pari sur le biais inductif, pas sur
l'information — et l'ablation de §5 doit le traiter comme tel.

Deux tests le tiennent : `v7_tail_excludes_only_the_opponents_suit` (l'exclusion ne
s'applique qu'à un adversaire — si elle dégénérait en « meilleur des quatre », v7 serait v6
plus six flottants redondants) et `canonical_bid_obs_is_invariant_under_suit_renaming`,
étendu à 123, qui échoue bien si le bloc [117:121] n'est pas permuté (vérifié par mutation).

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

**En attendant, la version approchée est faite** (2026-08-03) : la politique de v6 lue
par familles `HandCode`, chaque ligne avec son accord *et son plafond* —
[interpretability/bid_rules_v6.md](interpretability/bid_rules_v6.md), métrologie dans
[interpretability/rule_ceiling.md](interpretability/rule_ceiling.md). Trois retombées qui
concernent v7 :
- **Le plafond chiffre le coût de §1.1 côté lisibilité** : à l'ouverture, aucune règle
  insensible aux couleurs ne peut dépasser **97,4 % sur annoncer/passer ni 83,5 % sur
  l'action exacte**. Un bidder canonique porte ce plafond à 100 % par construction, ce
  qui est un argument pour §3.1 indépendant du gain en force.
- **v2 plafonne au même endroit que v6** (96,8 % / 83,1 %, et 7 points de mains stables
  en *plus*) : l'incohérence de symétrie ne vient pas de l'entraînement.
- **Négatif à ne pas re-tenter** : entraîner la règle distillée sur le réseau *symétrisé*
  plutôt que sur sa réponse brute ne vaut rien (+0,5 pt). La réponse à l'identité est un
  tirage non biaisé dans l'orbite, et l'apprenant moyenne ce bruit tout seul.

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
exactement la cible de label validée en §1.6.

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

## 4bis. Plan d'entraînement v7 — les bras, le budget, et ce qui juge

*Écrit le 2026-08-03, après §1.10 qui en fixe la forme.*

### Ce qu'on entraîne

Trois bras, **même budget, même graine, même pool**, tous depuis zéro :

| bras | obs | drapeaux | ce qu'il isole |
|---|---|---|---|
| **P** | 117 physique | `--sa-features-v3` | reproduction de v6 au budget des autres — sans lui, « v7 > v6 » confond l'obs et les 75 M de pas |
| **C** | 117 canonique | `--sa-features-v3 --canonical` | §3.1 seule |
| **V** | 123 canonique | `--sa-features-v7 --canonical` | §3.1 + §3.4 |

**P est le bras qu'on est tenté de couper, et c'est celui qui porte l'attribution.** v6 existe
à 75 M ; comparer un bras à 30 M contre lui mesurerait le budget autant que l'obs.

**Ce qu'on n'a *pas* : un bras qui sépare les 6 flottants de v7 en deux.** V mélange les 4
scores par couleur (un biais inductif) et les 2 réductions (une information). Un quatrième bras
à 119 les séparerait, et il n'est pas prévu — parce que §1.10 dit qu'un bras coûte ~19 h et que
la question qu'il trancherait est mineure devant les deux autres. **À dire, pas à cacher** : si
V bat C, on ne saura pas lequel des deux blocs l'a fait.

### Ce que les deux blocs de v7 valent avant d'entraîner

`[121]`/`[122]` sont, par construction, le meilleur et le deuxième meilleur `trump_score`
**après exclusion**. Donc à l'ouverture — rien à exclure — `[122]` **est** `second_trump_score`,
et en défense `[121]` **est** `opp_best_other_ts`. Ce n'était pas le dessein ; c'est tombé
comme ça, et ça se trouve être exactement les deux quantités que deux méthodes indépendantes
ont désignées :

- la sonde de couche cachée trouve `opp_best_other_ts` en défense
  ([interpretability/probe_morning_report.md](interpretability/probe_morning_report.md)) ;
- l'extraction de règles trouve la **deuxième couleur** comme le seul ajout de features qui
  paie, et note que `opp_best_other_ts` en est reconstructible et dominé — « le concept trouvé
  par le probe est le bon, la forme particulière ne l'est pas »
  ([interpretability/rule_ceiling.md](interpretability/rule_ceiling.md) §3).

⚠️ **Ces deux chiffres restent des mesures sur des *substituts*** — un XGBoost et un arbre qui
distillent le réseau depuis des features agrégées. C'est exactement l'erreur corrigée en §3.4 :
ils disent que le concept porte de l'information *que le distillat n'avait pas*, pas qu'il en
porte pour le réseau, qui lit la main en bits bruts. Ils justifient de **garder** le bloc, pas
d'en attendre un gain chiffré.

*Un faux ami à écarter tout de suite.* `rule_ceiling` §2 mesure que **viser le réseau symétrisé
n'apporte rien** à l'arbre, et on pourrait y lire un argument contre §3.1. Ce n'en est pas un :
là-bas c'est la **cible** qui est symétrisée pour un apprenant dont l'entrée l'était déjà, et
moyenner un bruit non biaisé est gratuit avec 84 000 mains. Ici c'est l'**entrée** qui est
canonicalisée, ce qui divise l'espace à couvrir par ~22 et lie 24 copies de paramètres. Les deux
phrases se ressemblent et ne parlent pas de la même chose.

### Le budget, dérivé et non estimé

L'échelle de checkpoints du resume de v6 donne le débit : 12 intervalles de 2,5 M en 5 797 s en
moyenne, soit **431 pas/s**. Donc **30 M ≈ 19,3 h par bras**, et les 75 M de v6 valaient ~48 h.

Trois bras à 30 M = **~58 h en séquentiel**. À vérifier avant de s'engager : le réseau
d'annonce est minuscule (117→512³→43), donc le GPU n'est probablement pas le goulot et deux ou
trois bras peuvent tenir de front — **une co-exécution courte le dira**, et c'est la seule
mesure à faire avant de lancer. Ne pas le supposer : si les bras se ralentissent mutuellement,
un run de 19 h en devient un de 40.

Et **cette campagne ne dépend pas d'une regénération de pool** : le pool actuel n'est pas à
refaire pour cause de dérive (§1.6), et l'obs qui change ne touche pas le format `(donne,
dd_pts)`.

### Ce qui juge, et à quelle puissance

Dans cet ordre, du moins cher au plus cher :

1. **Contrôle de câblage** *(secondes)* — `bid_equivariance` doit rendre **0,0 %** sur C et V
   dans les trois régimes. Non nul = mauvais branchement du masque ou de l'action, pas une
   régression de force. C'est le seul usage licite du taux de bascule ici (§1.10).
2. **Sonde stratifiée** *(secondes)* — `bid_probes --baseline docs/measurements/bid_probes_v6.json`
   sur chaque bras. Ne classe pas, mais attrape les régressions grossières et l'assertion dure
   du capot. C'est ce qui dira si §3.3 s'est réveillée toute seule.
3. **Sonde appariée en points par décision** *(minutes par main, sidecar requis)* —
   `bid_candidates.py`, §4. C'est **la** mesure principale, celle que §2.9 a désignée : elle
   lit un écart en points sur une continuation d'enchère réellement jouée, par régime, sur des
   mondes partagés.
4. **Arène** *(heures)* — h2h contre v6 à **2 000 matchs par direction**, réservé au bras
   gagnant. §1.8 donne le seuil : en dessous de ~4-10 pts/donne, 1 000 matchs ne voient rien.

**Et pour tous les trois derniers : chaque bras est évalué sur la moyenne de ses 4 derniers
checkpoints, jamais sur `bid_nn_final`.** C'est la conséquence directe de §1.10 — un checkpoint
isolé bouge de 12 pts d'un intervalle de 2,5 M au suivant, en fin d'entraînement.

### Ce qui ferait abandonner

- **C ≈ P sur la sonde appariée** : la canonicalisation ne coûte rien et reste (l'équivariance
  exacte a une valeur propre, cf. §3.6 qui en dépend), mais v7 n'est alors qu'une hygiène et il
  faut aller chercher la force ailleurs — §3.8.
- **V < C** : les 6 flottants nuisent. Retomber sur 117 canonique, et n'y revenir qu'avec le
  bras à 119 qui aurait dit lequel des deux blocs.
- **P ≪ v6 à 75 M** : 30 M ne suffit pas à reproduire la référence, et toute la campagne se
  relit à budget plus grand avant de conclure quoi que ce soit.

## 5. Ordre de travail proposé

1. ~~§4 — l'évaluateur de candidates~~ **fait** (2026-08-02).
2. ~~§2.2 — le Q plat~~ **fermée** (2026-08-02, §1.7), puis **étendue aux préfixes**
   (2026-08-02) : la platitude dépend du *type de décision*, et le coût du bruit de
   symétrie se concentre sur la **contestation**, régime jamais mesuré auparavant.
3. **§2.9 — la puissance de l'arène.** Remontée ici parce qu'elle conditionne le
   *critère d'acceptation* de tout ce qui suit : §3.1 comme §3.3 produisent des effets
   de quelques points par décision, que l'arène ne sait pas voir. Tant qu'on n'a pas le
   seuil de détectabilité, chaque expérience risque de se conclure par un « pas de
   différence » qui ne veut rien dire.
4. §3.2 — la suite de sondes. Immédiat, sans risque, et c'est le garde-fou qui manquait.
   À construire **par régime** (ouverture / contestation / soutien), la leçon de §1.7
   étant qu'une moyenne sur les régimes cache l'essentiel.
5. §3.1 — canonicalisation de l'obs, avec le flip rate **des trois régimes** comme test
   de non-régression (`bid_equivariance.py --prior`).
6. ~~§3.4 — les deux features du probe~~ **fait** (2026-08-02) : obs v7 à 123, et une
   seule des deux familles survit à la relecture de la mesure qui la justifiait.
7. ~~§2.8 — arbitrer donnes contre mondes~~ **tranché** (2026-08-02) : 5M × 20, les
   donnes gagnent. Reste à décider 5M contre 1M à k identique — 87 h contre 17 h.
8. Regénération du pool (due de toute façon) avec §3.5 dedans.
9. §3.3 — actions mortes, une fois qu'on a une sonde qui mesure le progrès.

**Périmètre acté le 2026-08-02**, en trois phases séparées par les *données* qu'elles
demandent — c'est le seul découpage qui rende chaque brique attribuable :

- **Phase 0** — §3.2, la suite de sondes. Aucune donnée neuve. C'est l'instrument. ✅
- **Phase 1** — §3.1 canonicalisation + §3.4 les features. **Aucune donnée neuve** non plus :
  le pool est `(donne, dd_pts)` et l'obs qui change ne le touche pas. Le plan d'entraînement
  et les bras sont en §4bis. ✅ pour le code, la campagne reste à lancer.
- **Phase 2** — §3.8, la croyance playgen **en entrée**. Demande un groupage de l'enchère et
  ~36 GPU-h de précalcul, donc une décision à part.

La regénération du pool (§3.5) et le « 5M contre 1M à k identique » sont **découplés** :
§1.6 dit que le pool n'est pas à refaire pour cause de dérive, et §4bis ne l'attend pas.

*Ce qui a débloqué §3.8 sans qu'on le remarque* : son chiffrage à 36 GPU-h supposait « un
groupage inter-positions (changement de code) ». `generate_worlds_multi` / `WorldBatchItem`
existent désormais — mais **pour les mondes de jeu seulement** ; le chemin d'enchère
(`JobKind::Auction`) part encore en `run_alone`, commenté « Not batchable ». C'est donc une
extension bornée d'une machinerie déjà écrite, plus un coût à cadrer.

---

## Reproduction

**Le sidecar playgen doit tourner** pour tout ce qui échantillonne des mondes :
`playgen-up` avant, **`playgen-down` après** (5,5 Go de VRAM résidents tant qu'il vit,
il n'y a pas de libération à l'inactivité). Cf. CLAUDE.md, « Playgen sidecar discipline ».

**Les runs se journalisent** depuis le 2026-08-02 : provenance + agrégats dans le
registre versionné [docs/measurements/index.jsonl](../measurements/index.jsonl), brut
monde par monde dans `data/analysis/`. Avant de relancer quoi que ce soit d'ici,
regarder si la mesure y est déjà — les trois régimes de §1.7 coûtent ~50 min de GPU.
Les entrées marquées `INCOMPLET` sont antérieures à l'instrumentation : agrégats justes,
brut absent. Cf. [docs/measurements/README.md](../measurements/README.md).

Versés dans `scripts/analysis/` :
- [bid_candidates.py](../../scripts/analysis/bid_candidates.py) — §1.3, §4 ;
- [bid_q_flatness.py](../../scripts/analysis/bid_q_flatness.py) — §1.7. `--prior ""` /
  `"100C"` / `"100C P"` pour les trois régimes ; **même `--seed` = mains appariées**,
  c'est ce qui autorise à comparer les régimes entre eux ;
- [bid_equivariance.py](../../scripts/analysis/bid_equivariance.py) — §1.1-1.2 : taux de
  bascule avec contrôles heuristiques, échelle des Q, erreur d'équivariance du vecteur Q.
  `--prior` mesure les trois mêmes régimes ; il **permute aussi le préfixe** (`apply_prior`),
  sans quoi on comparerait deux positions différentes. Les contrôles heuristiques restent
  la validation de l'arithmétique : ≤ 2,6 % dans les trois régimes ;
- [bid_capot_probe.py](../../scripts/analysis/bid_capot_probe.py) — §1.3 : fréquence du
  capot sur N enchères auto-jouées, sondes capot forcé, rareté des familles ;
- [hand_classes.py](../../scripts/analysis/hand_classes.py) — §1.4 : Burnside vérifié par
  force brute (`--verify`), et concentration des codes de main ;
- [card_importance.py](../../scripts/analysis/card_importance.py) — ce que vaut chaque
  carte, mesuré au solve apparié ; justifie le contenu de `HandCode`.

*Note de reproductibilité* : le « 97,8 % des positions sous 0,03 » de §1.1 est une
statistique d'échantillon. `bid_equivariance.py` avec sa graine par défaut rend 96,5 % —
même quantité, tirage différent. Les autres chiffres de §1.1 se reproduisent au centième.
