# Regénérer la couche de scores IS-DD

*Ouvert le 2026-08-04. Plan d'exécution : ce qui est décidé, ce qui reste à mesurer,
et les pièges qui produiraient des chiffres plausibles et faux.*

> **État au 2026-08-06.** Les **trois mesures sont faites** (A §4, B §4, C §6) et la
> génération **tourne** — `gen_score_layer` sur `base_5M[0..500k]`, deux GPU,
> 1,3 donnes/s, sortie `data/deals/scores_isdd_v2.sc` + `.ranks`, reprise à chaque
> checkpoint. Compter **~111 h** pour les 500 k. `tail_100k` n'est **pas** commencé et
> a perdu deux de ses trois justifications (§6, §11). Avant tout entraînement sur cette
> couche, lire **§9 — `--pool-size` est obligatoire**, sinon le trainer regénère un
> million de donnes en silence.
>
> **La reprise a été exercée pour de vrai** (arrêt volontaire le 2026-08-05 à 21 h 02,
> reprise le 2026-08-06 à 01 h 49) : 95 055 étiquettes, 95 055 rangs et 40 414 rejeux
> relus, 43 donnes perdues — celles d'après le dernier checkpoint. Le débit repart au
> même chiffre, donc **une pause ne coûte que le lot en cours**. Deux règles apprises
> à cette occasion : arrêter dans l'ordre *superviseur → générateur → chien de garde →
> sidecar* (ils se relancent mutuellement), et **ne pas recompiler entre deux tronçons**
> — un binaire différent produirait une couche à deux régimes que rien dans le fichier
> ne signalerait.

Une **couche de scores** (`COLVSC01`) est un tableau `[u8; 4]` par donne : les points
cartes N-S sous chaque atout, sous jeu fort. C'est l'entrée de la reward du bidder —
`train_bid_nn --reward real --scores <fichier>`. La couche courante,
`data/deals/scores_isdd_5M.sc`, date d'avril 2026.

## 1. Pourquoi la refaire

Quatre défauts cumulés, dont trois indiscutables :

| | |
|---|---|
| **mondes uniformes, pas playgen** | `enrich_pool_isdd` appelle `IsDdSearch::search()` **sans `WorldSource`** ([is_dd.rs:1059](../../colver-core/src/search/is_dd.rs#L1059), commentaire : *« sampling worlds from beliefs / constraint-uniform only »*). C'est un IS-DD nettement plus faible que le Dédé de production |
| **profondeur dérisoire là où ça compte** | mode temps à 20 ms/coup ⇒ **2 mondes au pli 1**, contre >5 000 au pli 7 ([bid_v7_plan §1.5](../bid/bid_v7_plan.md)) — or c'est l'entame qui décide de la valeur d'une annonce |
| **antérieure à deux ruptures** | retrait de `quick_tricks` (2026-07-23) et correctif de règle d'atout (2026-08-01) |
| **contrat-aveugle, et depuis peu incohérente** | cf. §3 |

**Résidu visible du bug `quick_tricks`** : 58 cases sur 20 M portent une valeur
arithmétiquement impossible — entre 241 et 249, alors que le maximum hors capot est 162
(152 + dix de der) et le capot 252. Toutes **juste sous le capot**. Corrélation, pas
preuve, mais le volume invisible (valeurs fausses *plausibles*) est par construction
plus grand, et il se concentre là où on veut justement miner.

⚠️ **Le motif de regénération n'est pas la dérive.** [pool_staleness.md](pool_staleness.md)
a mesuré que le vieillissement d'IS-DD ne justifie pas les heures : 87 % de l'écart est
du bruit d'échantillonnage. Ce qui justifie ce chantier est le **changement de source de
mondes** — playgen au lieu d'uniforme — pas la fraîcheur des chiffres.

## 2. Ce qu'on génère

| | contenu | format | état |
|---|---|---|---|
| *(rien à générer)* | les **500 k premières donnes de `base_5M.bin`** | `COLVDD01` | — |
| `scores_isdd_v2.sc` | points cartes N-S par atout, IS-DD fort | `COLVSC01`, `[u8;4]` | **en cours** |
| `scores_isdd_v2.sc.ranks` | rang de préfixe de chaque case (§4) | `COLVRK01`, `[u8;4]` | **en cours** |
| `tail_100k.bin` | strates **construites** via `hand_from_class_id` | `COLVDD01` | pas commencé, et à re-justifier (§6, §11) |

**Réutiliser les donnes existantes plutôt que d'en tirer de neuves**, pour trois
raisons : une donne (32 cartes + donneur) est indépendante des règles, donc seules ses
`dd_pts` sont périmées et `--reward real` ne les lit pas ; le donneur y est déjà
équilibré (25,03 / 24,99 / 24,98 / 24,99 %) ; et surtout **l'A/B ancienne couche contre
nouvelle devient direct, à donnes identiques**.

**Le format ne bouge pas.** `COLVSC01` est un drop-in derrière `--scores`, les couches
composent par `offset`/`count`, et rien dans le trainer ne change. C'est ce qui rend ce
plan exécutable sans toucher à la campagne d'entraînement.

## 3. L'objectif : `CardPoints`, et pourquoi ce n'est pas un repli

`enrich_pool_isdd.rs:79` pose aujourd'hui `objective: PlayObjective::DealScore`. Or
sous cet objectif `world_value` lit `world.contract`
([is_dd.rs:650-661](../../colver-core/src/search/is_dd.rs#L650)) — et le binaire joue
via `GameState::setup_dd`, qui code en dur **contrat 80, preneur N-S, sans contré**
([state.rs:188-190](../../colver-core/src/engine/state.rs#L188), commentaire
`// 80 points — irrelevant for solver`, vrai jusqu'au 2026-08-03 et faux depuis).

Toute couche produite aujourd'hui serait donc étiquetée par un joueur qui croit que
**N-S prend à 80 dans les quatre couleurs de toutes les donnes**. Ce n'est pas du bruit,
c'est un **biais** : un preneur à 80 sécurise 82 points et lâche le reste.

**`[u8;4]` et `DealScore` sont incompatibles par construction.** Un tableau indexé par
le seul atout ne peut porter qu'une quantité indépendante du contrat. Les points cartes
en sont une ; l'écart de score de donne, non.

**Rien de ce qui t'importe n'est perdu.** Réussite du contrat, belote et rebelote,
contré, surcontré, capot, dix de der — tout est calculé par `compute_scores()`
([bid_train_env.rs:869-957](../../colver-core/src/bid/bid_train_env.rs#L869)) au moment
de l'entraînement, avec le **vrai** contrat que l'enchère du modèle a produit :

```
points cartes N-S (la couche)  →  ns_pts
points E-O = 162 − ns_pts (ou 252 sur capot)
contrat réel : valeur, atout, camp preneur, coinche
belote : D+R d'atout dans la même main, détectée depuis les 4 mains
→ compute_deal_score()
```

**Ce qu'on perd honnêtement** : sous `CardPoints`, le *jeu lui-même* est aveugle au
contrat. Un vrai preneur à 80 sécurise, une vraie défense à 160 va chercher la chute ;
IS-DD, lui, maximise ses points des deux côtés. C'est une approximation du **style de
jeu**, pas du barème. Le seul chiffre dessus : le h2h `DealScore` vs `CardPoints` donne
49,7 % au match et 50,8 % à la donne — indistinguable de zéro.

## 4. L'enchère synthétique

**Elle est obligatoire, et ce n'est pas un choix de conception.** Le préfixe que playgen
consomme à une position de jeu est ([tokens.rs:6](../../colver-core/src/playgen/tokens.rs#L6)) :

```
[BOS] [OBSPOS_d] [h1..h8] [bid tokens ×B≤24] ([ACT_a] [CARD])×P≤32
```

Les jetons d'enchère sont dedans, avec l'acteur codé **relativement à l'observateur**
(`tokens.rs:13`). Sans enchère, pas de monde playgen.

### La construction

```
camp    = N-S si dd_pts[t] > 81, sinon E-O          ← gratuit, déjà dans le pool
siège   = argmax evaluate_for_trump dans ce camp    ← qui des deux partenaires annonce
valeur  = palier plausible dérivé de ce score
enchère = passes depuis (dealer+1)%4 jusqu'au siège, puis <valeur>t, puis P P P
```

**Le preneur est *choisi*, pas *imposé*, et la nuance porte tout.** On connaît les
quatre mains à la génération. Désigner un preneur au hasard reviendrait à annoncer
110♥ depuis un siège sans cœur — playgen, qui ne voit que l'annonce, placerait alors
des cœurs chez lui dans les mondes échantillonnés, ce qui est **factuellement faux**.
En prenant le siège qui *tient* réellement la couleur, l'inférence de playgen est
correcte. La pathologie du « pass-pass-pass-80 » sur une main énorme n'apparaît jamais
dans cette construction.

**Il n'y a aucun lien entre cette enchère et celle du modèle.** Ce sont deux enchères
à deux moments : celle-ci, hors ligne, ne juge aucune politique — elle produit un
nombre. Celle du modèle, en ligne, produit le contrat que `compute_scores` applique.

### Ce que cette construction rate — mesure A, 2026-08-04

*[bench_taker_position.rs](../../colver-core/src/bin/bench_taker_position.rs) +
[taker_position.py](../../scripts/analysis/taker_position.py). 43 076 donnes, enchère
rejouée puis les 4 atouts résolus en DD, donc **comparaison appariée** sur la même donne
et non deux marginales.*

**Le témoin d'abord, parce que c'est lui qui rend le reste lisible.** « Dans la
distribution » se dit par rapport au corpus sur lequel playgen a *appris*
(`playgen_games_9M.bin`, joué par DouDou50), pas par rapport à n'importe quelle vraie
enchère. Les deux corpus — celui-là et `isdd_games_v1.bin`, joué par IS-DD — s'accordent
**à 0,5 pp près sur chacune des statistiques ci-dessous**. L'enchère est donc une
propriété de **bid v6** et pas du joueur de cartes derrière lui, et la référence tient.

**Le décalage est sur la *forme* du préfixe, pas sur l'identité du preneur :**

| | enchères réelles | la construction |
|---|--:|--:|
| première annonce par le siège qui parle en premier | **80,1 %** | ≈ 25 % (= la position du preneur) |
| une seule annonce dans toute l'enchère | **11,9 %** | 100 % |
| enchère contestée (les deux camps annoncent) | **81,3 %** | 0 % |
| coinchée | **25,7 %** | 0 % |
| longueur du préfixe | 8,18 jetons en moyenne | 4 à 7 (47,8 % du réel) |

La position, elle, décale bien moins — mais dans un sens qu'il faut nommer : **parler
tard fait prendre**. Le donneur, qui parle en dernier, emporte 31,3 % des contrats
contre 18,2 % au premier parleur, alors que c'est ce dernier qui ouvre 80 % du temps.
La construction, qui choisit le siège sur la seule force de main, sort presque plate.

| | pos 0 (premier) | pos 1 | pos 2 | pos 3 (donneur) |
|---|--:|--:|--:|--:|
| enchères réelles | 18,2 % | 21,8 % | 28,8 % | 31,3 % |
| construction, atout réel | 25,2 % | 25,6 % | 25,6 % | 23,6 % |
| construction, les 4 atouts | 28,3 % | 22,9 % | 27,2 % | 21,7 % |

soit **10,9 pp** de distance en variation totale.

**L'accord apparié dit d'où vient l'erreur.** Le camp est bon **89,4 %** du temps — la
construction se trompe rarement de côté. Le siège ne l'est que **61,8 %**, et même en
lui donnant le bon camp, `argmax evaluate_for_trump` désigne le vrai preneur **71,4 %**
du temps. Le choix du siège est donc le maillon faible, mais il pèse peu à côté de la
forme.

⚠️ **Et « valeur = palier plausible dérivé de ce score » n'est pas soutenu par les
données.** Sur les vraies enchères, la valeur annoncée reste entre **112 et 124** pour
tous les scores `evaluate_for_trump` de 1 à 31 (n ≥ 30 par ligne), et ne monte qu'au-delà
de 32. La relation est plate et même en U : dans une enchère contestée à 81 %, la valeur
est décidée par la **pression de l'enchère**, pas par la main du preneur. Une échelle
tirée de ce score serait une règle inventée présentée comme mesurée.

### La suite de A : « v6 masqué sur la couleur » — proposé, puis réfuté

*Même run, `--bid-model`. La variante a été proposée ici même sur un raisonnement, puis
mesurée avant d'engager la moindre heure de GPU. Elle ne survit pas.*

L'idée était de laisser bid v6 mener l'enchère avec le masque légal réduit à `PASS`,
`COINCHE`, `SURCOINCHE` et aux paliers de l'atout cible : contestation, coinche, longueur
et valeur sortiraient de la politique au lieu d'être fabriquées, et l'atout resterait
celui qu'on veut étiqueter.

**Le témoin d'abord — et il est aussi fort qu'on peut l'espérer.** Le même pilote, lancé
**sans masque** sur les donnes du corpus, doit reproduire l'enchère à l'identique : le
réseau est déterministe et c'est lui qui a produit ces donnes. Mesuré : **99,99 %**
(43 028 / 43 031), et les cinq statistiques se reproduisent à la deuxième décimale. Le
pilote est donc juste, et les chiffres de la variante masquée sont interprétables — sans
ce contrôle, un historique mal suivi ou un score passé du mauvais côté se lirait comme
une propriété de la variante.

| | corpus | **v6 libre** (témoin) | **v6 masqué** |
|---|--:|--:|--:|
| 1re annonce par le premier parleur | 80,06 % | 80,06 % | 29,56 % |
| une seule annonce | 11,87 % | 11,87 % | 78,58 % |
| **contestée** | **81,26 %** | 81,25 % | **0,06 %** |
| coinchée | 25,74 % | 25,74 % | 9,01 % |
| longueur du préfixe | 8,18 | 8,18 | 5,89 |
| cases sans aucune enchère | — | — | **12,64 %** |

Elle échoue sur les cinq cibles, et la valeur s'effondre avec : **66,8 %** des contrats
masqués sont à 80 ou 90, contre 11,1 % en réel.

**La raison est structurelle, et c'est le vrai résultat de la journée :**

> **la contestation *est* le mécanisme qui sélectionne l'atout.** On conteste parce qu'on
> préfère *sa* couleur. Forcer les quatre sièges sur une seule la supprime — les
> adversaires n'ont plus rien à dire, donc ils passent (0,06 %), et le preneur sans
> opposition annonce le minimum. Demander « une enchère par case `(donne, atout)` » et
> « une enchère réaliste » revient à demander deux valeurs à la même variable.

Aucune construction à atout imposé ne peut donc être dans la distribution — ni celle du
plan, ni une version plus soignée. Ce n'est pas un défaut de réglage.

### L'épluchage : ne rien fabriquer, retirer

*Idée de l'utilisateur, mesurée dans la foulée. Elle change la réponse.*

Toutes les constructions précédentes **fabriquaient** une enchère. Celle-ci en **retire**
une : on prend l'enchère réelle, on remplace sa dernière annonce par une passe, on
referme. Ce qui reste est un vrai préfixe — chaque action y a été choisie par v6 dans la
situation où elle a eu lieu. On recommence tant qu'il reste des annonces, ce qui descend
la chaîne des couleurs annoncées.

**Le plafond est structurel, et il n'était pas évident** : on ne peut retirer que des
annonces qui ont eu lieu, et deux annonces *dans la même couleur* ne donnent qu'une case.
Or une vraie enchère ne nomme que **2,17 couleurs distinctes** en moyenne — 57,7 % en
nomment exactement deux, 13,4 % une seule, 2,0 % les quatre.

| niveau | cases | sans enchère | atout neuf | contestée | = v6 redemandé |
|---|--:|--:|--:|--:|--:|
| **or** (enchère libre) | 43 031 | 0,0 % | 100 % | 81,3 % | — |
| **−1** | 43 031 | 11,9 % | **84,2 %** | 67,0 % | 78,8 % |
| **−2** | 37 922 | 26,2 % | 18,0 % | 56,2 % | 56,3 % |
| **−3** | 27 996 | 35,3 % | 14,2 % | 38,7 % | 38,1 % |

Deux lectures :

1. **Le premier épluchage paie, les suivants beaucoup moins.** −1 tombe sur une couleur
   neuve **84,2 %** du temps ; −2 et −3 seulement 18 % et 14 %. Ce n'est pas un défaut de
   la méthode, c'est le plafond des 2,17 couleurs qui mord : les annonces qui restent sont
   dans des couleurs déjà couvertes. La chaîne entière couvre **2,09** cases sur 4
   (**2,37** si l'on redemande à v6, cf. ci-dessous). Les valeurs suivent la descente :
   116 → 107 → 101 → 95.
2. **La qualité se dégrade doucement, pas d'un coup.** Même à −3, l'enchère est contestée
   38,7 % du temps — contre **0 %** pour la construction du plan et 0,06 % pour v6 masqué.
   Un préfixe épluché reste incomparablement plus réaliste qu'un préfixe fabriqué.

**Refermer par des passes affirmées, ou redemander à v6 ?** La troncature affirme que
personne ne relance ; c'est vrai **78,8 %** du temps au premier épluchage, 56,3 % au
deuxième, 38,1 % au troisième. **Redemander à v6 est le bon choix** : même coût (~0,4 ms),
continuation réelle au lieu d'affirmée, et **meilleure couverture** (2,37 contre 2,09,
avec seulement 1,1 % de donnes réduites à une seule couleur contre 13,4 %). La troncature
n'achète qu'un atout déterministe — dont on n'a pas besoin, puisqu'on étiquette la case
sur laquelle l'enchère tombe.

### Relance ou ouverture : ce qui sépare l'argent du bronze

*Deuxième idée de l'utilisateur, et elle ne remplace pas l'épluchage — elle le **gradue**.*

Toutes les annonces retirées ne coûtent pas la même chose. Si l'annonce retirée était une
**relance** de son auteur, celui-ci reste visible dans le préfixe : il a annoncé plus tôt,
playgen sait qu'il tient quelque chose, on lui a seulement retiré une enchère de plus. Si
c'était son **ouverture**, la passe forcée le rend **muet** — et playgen en déduit qu'il
n'a rien, alors qu'il tient précisément la couleur qui décidait de la donne. C'est le
mensonge le plus cher qu'on puisse mettre dans un préfixe.

| épluchage | relance | **ouverture** |
|---|--:|--:|
| −1 | 32,6 % | **67,4 %** |
| −2 | 12,9 % | 87,1 % |
| −3 | 3,6 % | 96,4 % |

**Le cas coûteux est le cas courant, pas l'exception** — la contre-intuition ici, c'est
que les enchères sont courtes : l'annonce gagnante est le plus souvent la première de son
auteur. Et refuser catégoriquement de rendre un siège muet coûte cher : la chaîne tombe à
**1,32** case couverte au lieu de 2,37, avec **68,3 %** des donnes réduites à leur seule
case « or ».

D'où le compromis : **on n'arrête pas la chaîne, on étiquette la qualité de chaque case**.
Le critère « relance ou ouverture » est exactement ce qui sépare l'argent du bronze, et il
est gratuit — le siège d'une action se déduit sans rejeu (`seat(i) = (dealer+1+i) % 4`,
l'enchère avançant d'un siège par action).

### Pourquoi les deux cases restantes ne peuvent pas être sauvées

L'épluchage porte le compte de **1 case sur 4 à ~2,4**. Les ~1,6 restantes sont les
couleurs que **personne n'a annoncées** — et pour celles-là il n'existe aucune enchère
réaliste *sur cette donne*, parce que la raison pour laquelle personne ne les a annoncées
est que personne n'avait la main pour. Demander « et si l'atout avait été carreau ? »
contredit la donne elle-même.

**Ce n'est donc pas un défaut d'imagination, c'est la question posée.** Un `[u8;4]`
demande une contrefactuelle pour ~1,6 de ses 4 entrées, et aucune construction ne la
rendra plausible. C'est aussi, précisément, ce que le bidder doit apprendre : *pourquoi
ne pas annoncer cette couleur*. Le label reste utile ; c'est son préfixe qui est
irréductiblement hors distribution.

### La forme du générateur

| rang | source de l'enchère | ce qu'on a menti | cases/donne | contestée |
|---|---|---|--:|--:|
| **or** | enchère libre de v6 | rien | 1,00 | 81,3 % |
| **argent** | épluchage, **relance** retirée | une enchère de plus, l'auteur reste visible | ~0,3 | 67 % |
| **bronze** | épluchage, **ouverture** retirée | un siège devient muet | ~1,0 | 39-67 % |
| **fer** | construction §4 | tout le préfixe | ~1,6 | 0 % |

### Mesure B — ce que chaque rang coûte, 2026-08-05

*[bench_prefix_label.rs](../../colver-core/src/bin/bench_prefix_label.rs) +
[prefix_label.py](../../scripts/analysis/prefix_label.py). 1 985 donnes × 5 bras
= 9 925 étiquetages IS-DD, 33 min sur deux GPU. La même case `(donne, atout)` étiquetée
sous chaque préfixe, donc **comparaison appariée**.*

**Le témoin d'abord** : le même préfixe étiqueté deux fois avec deux graines donne un
écart-type apparié de **24,37 points cartes**. C'est le bruit propre de l'étiqueteur à
40 mondes/décision, et il est **non biaisé** — il se moyenne sur des centaines de
milliers de donnes.

**L'ordre du mensonge *est* l'ordre du coût. L'hypothèse tient.**

| rang | points cartes **du preneur**, contre le fer | z |
|---|--:|--:|
| **or** (enchère réelle) | **+4,36 ± 0,65** | +6,7 |
| **argent** (relance retirée) | **+4,34 ± 1,18** | +3,7 |
| **bronze** (ouverture retirée) | **+1,92 ± 0,73** | +2,6 |
| fer (construit) | référence | — |

Lecture : **un préfixe réaliste fait jouer le preneur ~4 points mieux**, parce que
playgen y place correctement la force et qu'IS-DD cherche donc avec de bonnes croyances.
Et **l'argent vaut l'or** — retirer une relance ne coûte rien, l'auteur reste visible ;
c'est rendre un siège muet qui coûte la moitié du bénéfice. La distinction relance /
ouverture était la bonne.

⚠️ **Il a fallu orienter par le preneur pour voir quoi que ce soit.** En points N-S
bruts, or − fer ne donne que **+2,22** : l'effet change de signe selon le camp qui prend
(+6,69 quand N-S prend, −2,10 quand c'est E-O) et s'annule à moitié à la moyenne. C'est
**exactement** le piège de [bid_v7_plan §1.11](../bid/bid_v7_plan.md) — rang « de la
donne » contre rang « vu du preneur » — et je suis retombé dedans, y compris dans la
première version du dépouillement journalisé.

⚠️ **Et « petit devant le bruit » ne veut pas dire « négligeable ».** 4,36 points ne font
que 0,18× l'écart-type du témoin, ce qui invite à conclure que le préfixe ne se voit pas.
C'est faux : le bruit du témoin est **non biaisé** et se moyenne, le décalage est
**systématique** et ne se moyenne pas. Il reste dans chaque étiquette, dans le même sens.

### Ce que B impose au générateur

Une couche **mixte** — or sur la case que v6 annonce, fer sur les autres — porte donc un
écart **entre ses propres cases** : celle que v6 a choisie est mieux étiquetée que les
trois autres, ce qui incline le futur bidder vers la politique qu'on cherche justement à
dépasser. Un préfixe uniforme n'aurait pas ce défaut, mais il l'échangerait contre un
biais commun (−4 points au preneur partout, donc sous-annonce systématique).

**On ne tranche pas ça sur un run de plusieurs jours : on enregistre de quoi le trancher
après.** `gen_score_layer` écrit un fichier `<couche>.ranks` — un octet par case, quatre
par donne, magic `COLVRK01` — qui dit quel rang de préfixe a produit chaque étiquette.
Avec lui, la correction se calcule, s'annule ou se mesure. Sans lui, la couche est un
mélange irrécupérable.

⚠️ **Deux réserves à ne pas perdre de vue.**

1. **Le biais de sélection de la case « or »** : c'est v6 qui la choisit, donc ce sont les
   atouts que *v6* aime qui reçoivent le bon préfixe. Si v7 s'en écarte — ce qui est
   l'objectif — la couverture se déplace sans que rien ne le signale.
2. **L'ordre de l'épluchage n'est pas l'ordre des rangs DD de §5.** L'épluchage descend
   les couleurs *annoncées* ; le budget dégressif descend les rangs de `dd_pts`. Les deux
   coïncident probablement souvent, mais **ce n'est pas mesuré** et il ne faut pas les
   confondre en câblant l'un sur l'autre.

### Le camp preneur est déterminé, donc `[u8;8]` n'est pas nécessaire

Les points cartes sont à somme constante, donc pour un atout donné **un seul camp peut
tenir un contrat** : `dd_pts[t] > 81` ⟺ N-S. Mesuré sur 1 M de donnes : 49,63 % N-S /
50,37 % E-O, la symétrie attendue.

**On obtient donc le bénéfice de `[u8;8]` — des croyances accordées au preneur — au
coût de `[u8;4]`**, avec un solve déjà payé.

⚠️ **Ce n'est pas gratuit sur toute la population.** Mesuré ([bid_v7_plan §1.11](../bid/bid_v7_plan.md)) :
le preneur est le camp désigné par `dd_pts` dans **78,9 %** des contrats atteints par
une politique entraînée, et il est sous 80 points DD — incapable de tenir le contrat
minimum — dans **20,2 %**. Ce cinquième d'épisodes est joué, à l'étiquetage, sous des
croyances qui ne correspondent pas. C'est la borne de l'arbitrage `[u8;4]` / `[u8;8]`,
et c'est ce que la **mesure B** doit trancher.

**Deux chiffres pour la même chose, et ils ne portent pas sur la même population.** La
mesure A donne **89,4 %** là où §1.11 donne 78,9 %. Ce n'est pas une contradiction :
89,4 % porte sur les contrats que v6 atteint **en jeu**, donc sur des atouts qu'il a
choisis parce que le camp y était net ; 78,9 % porte sur ceux que la **boucle
d'entraînement** atteint, ε-greedy comprise, donc sur des cases bien plus douteuses.
C'est la seconde qui dimensionne le risque, puisque c'est elle qui consulte la couche.
Les citer ensemble sans dire ça ferait croire que le problème est deux fois plus petit
qu'il n'est.

## 5. Le budget par atout : gradué, jamais élagué

*Mesuré le 2026-08-04, [bid_contract_ranks.py](../../scripts/analysis/bid_contract_ranks.py).*

La reward ne lit qu'**une** case par épisode. Rang de l'atout contracté, du point de vue
du camp qui l'a pris :

| régime | rang 0 | rang 1 | rang 2 | rang 3 | top-2 |
|---|--:|--:|--:|--:|--:|
| tardif (v6, ε = 0,02) | 58,4 | 22,7 | 11,7 | 7,2 | 81,1 % |
| début (init aléatoire, ε = 0,30) | 29,9 | 25,2 | 23,8 | 21,1 | 55,0 % |

**Un élagage sec est exclu** : une case sans label serait consultée à vide dans 44,9 %
des épisodes au début de l'entraînement — précisément quand le modèle apprend à ne pas
annoncer n'importe quoi.

**Et un repli sur `dd_pts` serait pire que rien** : un solve DD voit les quatre mains,
c'est un **majorant** pour le preneur. Les cases élaguées paraîtraient donc *meilleures*
que les étiquetées, ce qui pousse à explorer exactement les contrats qu'il faut fuir.
Deux échelles de label dans une même reward est le défaut à ne pas introduire.

D'où un budget gradué — une seule échelle, tous les labels en IS-DD, pas au même prix :

| rang de l'atout (depuis `dd_pts`) | budget | poids |
|---|---|--:|
| 0 et 1 | `dets_schedule` plein | 2,00 |
| 2 | moitié | 0,50 |
| 3 | quart | 0,25 |
| | **total** | **2,75 / 4 = −31 %** |

## 6. La diversité : deux fichiers, pas un fichier pondéré

La canonicalisation de l'obs rend `hand_class_id` / `hand_from_class_id` exploitable —
bijection testée sur les 472 579 classes — donc on peut enfin **construire** une
distribution de mains au lieu de la subir.

- **`base_500k`** — tirage uniforme, majoritaire. C'est l'a priori honnête :
  sur-échantillonner les mains fortes sans corriger enseigne au modèle un mauvais a
  priori sur ce que tiennent les adversaires ([bid_v7_plan §2.4](../bid/bid_v7_plan.md)).
- **`tail_100k`** — strates choisies pour être **décisives**, pas pour être rares.

**Deux fichiers plutôt qu'un fichier pondéré** : le ratio de mélange *est* le poids
d'importance, et le garder comme drapeau d'entraînement le rend testable (§2.4 demande
trois variantes à budget égal). Baké dans le fichier, il cesse d'être mesurable.

Ce que la queue contient dépend de la **mesure C**. Candidats identifiables gratuitement
depuis `dd_pts` :

| strate | fréquence dans `base_5M` |
|---|---|
| capot N-S atteignable en DD (≥ 1 couleur) | **16,08 %** (803 803 donnes) |
| capot pour l'un ou l'autre camp | 26,31 % |
| cases (donne, atout) à 252 | 4,57 % (913 476) |
| meilleur atout ≥ 150 | 17,89 % |

⚠️ **Un capot atteignable *en DD* n'est pas un capot annonçable** — le solveur voit les
quatre mains. La quantité qui dimensionne la strate est `P(capot | mes 8 cartes)`.

### Mesure C — `P(capot | mes 8 cartes)`, 2026-08-04

*[bench_capot_prior.rs](../../colver-core/src/bin/bench_capot_prior.rs) +
[capot_prior.py](../../scripts/analysis/capot_prior.py). 1 200 mains × 80 complétions des
24 cartes restantes × 4 solves = 384 000 solves, **sans GPU**, 9 min sur 16 cœurs. Le
donneur est retiré à chaque complétion : il décide qui entame, donc il change la valeur
DD, et la moyenne sur les positions est ce qu'un bidder voit avant de connaître la sienne.*

| | |
|---|--:|
| capot N-S atteignable en DD (marginale, **vue des 4 mains**) | 16,08 % |
| `P(capot \| ma main)`, un atout quelconque | 15,75 % |
| **`P(capot \| ma main)`, à l'atout que j'annoncerais** | **8,68 %** |

Les deux premières se ressemblent **par coïncidence** ; c'est la troisième qui décide,
parce qu'on annonce *une* couleur et pas « l'une des quatre ». Confondre les deux
premières avec la troisième était le piège.

**Et la queue est courte, ce qui change ce que la strate peut enseigner :**

| seuil (au meilleur atout) | part des mains |
|---|--:|
| P ≥ 10 % | 34,5 % |
| P ≥ 25 % | 6,1 % |
| P ≥ 50 % | **0,17 %** (2 mains sur 1 200) |
| P ≥ 75 % | **0** |

**Aucune main ne rend le capot majoritairement vrai.** Une strate « mains à capot » ne
peut donc pas apprendre au modèle *à annoncer capot ici* — au mieux *ici c'est à 25 %*.
`tail_100k` doit donc contenir des mains où le capot est **envisageable**, et le modèle
y apprend surtout à **ne pas le prendre**. C'est un objectif différent de celui que §6
supposait, et il est plus modeste.

**Bonne nouvelle : la strate se construit par filtre, sans simuler.**

| `eval_max` | n | P(capot) | points moyens |
|---|--:|--:|--:|
| [15 ; 20) | 467 | 6,9 % | 113 |
| [20 ; 25) | 248 | 14,6 % | 136 |
| [25 ; 30) | 87 | **25,0 %** | 158 |
| [30 ; 35) | 10 | 27,9 % | 163 |

`evaluate_for_trump` — le repère le moins cher qui existe, déjà dans le moteur — sépare
d'un facteur **18** entre les tranches basses et hautes, et le centile supérieur en
P(capot) a un `eval_max` moyen de 29,1 contre 17,0 sur l'ensemble. Pas besoin de payer
384 000 solves pour bâtir `tail_100k` : un seuil sur `eval_max` suffit.

## 7. Les quatre invariants sur le donneur

Le siège qui parle en premier **est** celui qui entame — c'est vrai dans le moteur, pas
seulement en théorie :

| endroit | code |
|---|---|
| enchère | [state.rs:131](../../colver-core/src/engine/state.rs#L131) — premier parleur = `(dealer+1) % 4` |
| fin d'enchère → jeu | [bidding.rs:172-174](../../colver-core/src/engine/bidding.rs#L172) — `trick_lead = current_player = (dealer+1) % 4` |
| étiquetage | [state.rs:180](../../colver-core/src/engine/state.rs#L180) — `setup_dd` : `first = (dealer+1) % 4` |

Le donneur voyage avec la donne et n'est **jamais** réassigné : `reset_from_deal` fait
`GameState::new(deal.dealer, deal.hands)`, et en `--match-sim` la rotation 0→1→2→3 tire
une *autre* donne au bon donneur au lieu de réétiqueter celle-ci — le commentaire de
`dealer_index` le dit : *« each deal keeps its original dealer (so ISDD scores stay
valid) »*.

D'où quatre contraintes sur le générateur, dont deux ne se voient pas :

1. **L'enchère synthétique commence ses passes à `(dealer+1) % 4`.** Émettre depuis le
   siège 0 décalerait tous les acteurs, dont le codage est relatif à l'observateur —
   playgen échantillonnerait des mondes pour une autre table, sans erreur.
2. **La position du preneur par rapport au donneur varie, et c'est correct.** Mais la
   distribution produite par `argmax evaluate_for_trump` n'est pas celle des vraies
   enchères : parler tôt donne l'initiative sur une couleur. C'est la **mesure A**.
3. **Donneur à ~25 % par fichier *et par strate*.** `sample_with_dealer` dégrade en
   silence sur un bucket mince — 64 rejets puis « la première donne qui correspond »,
   donc la *même* donne resservie.
4. **Le siège de la main construite est tiré indépendamment du donneur.**
   `hand_from_class_id` rend une main, pas une donne ; fixer les deux apprendrait au
   modèle que les mains fortes parlent toujours à la même position.

## 8. Le coût

### Ce qui tourne réellement, et son débit mesuré (2026-08-05)

```bash
./target/release/gen_score_layer --pool data/deals/base_5M.bin --offset 0 --count 500000 \
  --threads 160 --checkpoint 500 --out data/deals/scores_isdd_v2.sc \
  --url "http://localhost:8003,http://localhost:8003,http://localhost:8003,http://192.168.1.23:8003"
```

| | mesuré |
|---|--:|
| débit, **deux GPU** (4090 ×3/4 + 3090 ×1/4) | **1,25 donnes/s** = 5,0 étiquetages/s |
| un GPU seul | 0,33 → 0,72 donnes/s selon la charge |
| 500 000 donnes à ce rythme | **~111 h ≈ 4,6 jours** |
| par nuit de 9 h | ~40 000 donnes |

**Trois écarts assumés entre le plan et ce qui tourne**, pour que le document ne décrive
pas un binaire qui n'existe pas :

1. **Budget plat à 40 mondes, pas gradué.** Le −31 % de §5 demande trois jeux de joueurs
   par thread (un par budget), donc 3× les clients IS-DD sur un sidecar qui plafonne déjà
   — le réglage à 96-160 threads a coûté deux faux départs. Gain refusé au profit de la
   robustesse d'un run de plusieurs jours.
2. **Pas de `dets_schedule` décroissant — et c'est désormais un choix mesuré, pas un
   report.** Voir « Le calendrier de mondes ne rapporte pas ici » ci-dessous.
3. **Pas encore de `tail_100k`.** La mesure C (§6) a montré que la strate se filtre sur
   `evaluate_for_trump` sans simuler ; elle se construira à part, quand le corps de la
   couche existera.

**160 threads, et pas plus** : deux jeux de joueurs par thread en font 2 048 clients
IS-DD contre 64 threads d'accueil du sidecar, ce qui produit 1 050 lanes par lot et des
timeouts en cascade. Et les donnes perdues ainsi ne sont pas un tirage au hasard — ce
sont celles jouées pendant la saturation.

**Reprise, et la variante qui laisse la prod tranquille.** Tuer et relancer repart du
dernier checkpoint : la même commande, ou celle-ci qui n'utilise que le GPU local et rend
sa 3090 à colver.net pour la journée.

```bash
./target/release/gen_score_layer --pool data/deals/base_5M.bin --count 500000 \
  --threads 96 --out data/deals/scores_isdd_v2.sc --url "http://localhost:8003"
```

⚠️ **L'URL reste explicite, volontairement.** Un script d'enveloppe avec le sidecar de
moxxi câblé dedans rendrait trop facile d'envoyer par mégarde de la charge sur la prod ;
mesuré une nuit à 1/4 de pondération, elle tient (`no_playgen: 0` sous charge, cf. §9),
mais c'est une décision à reprendre à chaque lancement, pas un défaut.

### Le calendrier de mondes ne rapporte pas ici (2026-08-05)

`dets_schedule = "40,40,40,30,20,15,15"` vaut **1,24×** sur `gen_games_isdd`
([isdd_games.md](isdd_games.md)). Transporté ici, il ne vaut presque rien. Deux mesures,
400 donnes chacune :

| | résultat |
|---|---|
| **le prix** — l'étiquette bouge-t-elle ? | **−0,22 ± 0,59 pt** (z = −0,4), soit **0,009×** le bruit d'une étiquette. Contrôle : **100 %** de rangs de préfixe identiques entre les deux bras, donc la comparaison porte bien sur le seul budget de mondes |
| **le gain** — A/B **alterné**, 4 tours, minimum par bras | plat `329 322 335 321` → 321 s ; calendrier `301 305 327 309` → 301 s ⇒ **1,069×** (médiane 1,060×) |

**Pourquoi le 1,24× ne se transporte pas, et c'est la partie réutilisable.** Le calendrier
retire **28,6 %** des mondes (280 → 200) et ne gagne que ~6 % de temps, parce qu'il les
retire là où **un monde coûte le moins cher** : en fin de donne il reste peu de cartes
cachées, donc peu de pas de décodage. C'est l'asymétrie de
[« un total égal n'est pas un coût égal »](isdd_games.md) prise dans l'autre sens — un
**compte** égal de mondes n'est pas un **coût** égal, et couper dans les moins chers ne
rapporte pas.

**Décision : on ne bascule pas.** 7 h gagnées sur 111 h, contre soit une couche à deux
régimes, soit jeter les donnes déjà produites. Le calcul ne se retourne à aucun moment du
run.

⚠️ **Note de méthode.** Une première tentative avait lancé les deux bras **l'un après
l'autre** et donnait 1,08× — proche du bon chiffre, ce qui est de la chance et ne valide
pas la méthode : la charge de la machine varie de 20 % ici, plus que l'effet cherché. Le
chiffre retenu vient de l'alternance. Et les plages se **chevauchent** (un lot du
calendrier à 327 s dépasse le meilleur lot plat) : l'effet est réel mais petit devant le
bruit de charge, ce qui est une raison de plus de ne pas réorganiser un run pour lui.

### L'estimation d'origine, gardée pour mémoire

Base : `gen_games_isdd` fait **2,62 donnes/s** sur une 3090 à 40 mondes/décision
([isdd_games.md](isdd_games.md)), soit ~32 recherches IS-DD par donne. 600 k donnes ×
4 atouts = 2,4 M déroulements.

| | 1 GPU | 2 GPU |
|---|--:|--:|
| à plat | 254 h | 127 h |
| + `dets_schedule` décroissant (**1,24×** mesuré) | 205 h | 102 h |
| + budget gradué (§5, −31 %) | **141 h** | **~70 h ≈ 3 jours** |

L'écart avec le mesuré (111 h contre 127 h à plat sur 2 GPU) vient de ce que la 4090
locale est plus rapide que la 3090 de référence, et que les quatre étiquetages d'une
donne partagent le coût fixe de sa lecture.

Leviers, avec leur prix :
- **`dets_schedule = "40,40,40,30,20,15,15"`** — 1,24×, et le gain vient surtout du
  total réduit. Cohérent avec `isdd_dets_by_stage` : tout le regret au-dessus de
  0,10 pt DD est à 8-6 cartes restantes, **zéro sous 3 cartes**.
- **`worlds.url` en liste séparée par des virgules** — tourniquet global au processus ;
  répéter une URL pondère entre GPU de débits différents.
- ⚠️ **v2-belote-small (1,65×) est un arbitrage, pas une optimisation** : ses mondes
  sont à **2,09× le bruit d'échantillonnage** de ceux de v2. Sur un corpus
  d'étiquetage, non.

## 9. Garde-fous

Tous déjà écrits ailleurs ; il s'agit de les brancher.

- **Fraîcheur du sidecar** : `curl <url>/health` → `sidecar.fresh: true`. Le précédent
  qui a motivé ce contrôle est exactement ce cas — 21 h de prod sur un sidecar périmé,
  mondes rejetés en silence.
- **Contrainte belote** appliquée à la source dans les `determinize*` et dans
  `worlds::retain_valid`, plus **sur-commande** (`source_fill`) : ~15 % des mondes sont
  écartés aux positions à belote, donc demander `n` en rend ~0,85 n.
- **Éclats + reprise** (`--shard` / `--merge`) : un run de 140 h sans ça ne laisse rien
  à 95 %.
- **Vérification avant écriture** : rejouer chaque donne, ne rien écrire si une seule
  échoue. `GameState::step` ne valide pas la légalité.
- **Équilibre des donneurs** asserté par fichier et par strate (§7.3).
- **Journalisation** : `runlog.save` pour chaque mesure.

### ⚠️ Une couche partielle ne s'entraîne pas sur le pool entier

Trouvé le 2026-08-04, avant le premier run. `RewardMode::RealOnly` fait
`self.real_pts.map(...).unwrap_or(ns_dd_pts)`
([bid_train_env.rs:1005](../../colver-core/src/bid/bid_train_env.rs#L1005)) : une donne
que la couche ne couvre pas **retombe sur `dd_pts`, sans un mot**. Et
`DealPool::load_or_generate` ne tronque pas — il rend le pool entier dès qu'il est assez
grand, donc `--pool-size` ne restreint pas l'échantillonnage.

Conséquence : entraîner avec `--pool-file base_5M.bin --score-file scores_isdd_v2.sc`
alors que la couche couvre 30 k donnes ferait tirer 99,4 % d'épisodes étiquetés en
**valeur DD périmée** (antérieure au retrait de `quick_tricks`) et 0,6 % en IS-DD frais.
C'est exactement le défaut que §5 refuse pour l'élagage — **deux échelles de label dans
une même reward** — et il arriverait ici par omission plutôt que par choix.

**La parade est un fichier, pas un drapeau** : tronquer le pool au préfixe couvert
(`base_5M.bin` → `base_<N>.bin`, en-tête `COLVDD01` + 21 o/donne) et entraîner sur
celui-là. Aucune modification du trainer, et le décompte devient vérifiable — le pool et
la couche doivent annoncer le même nombre de donnes.

#### ⚠️⚠️ Tronquer ne suffit pas : `--pool-size` est obligatoire

Vérifié à blanc le 2026-08-05, et c'est pire que ce qui précède.
`DealPool::load_or_generate(path, n, …)` **regénère** dès que le fichier contient moins
de `n` donnes, et `--pool-size` vaut **1 000 000 par défaut**. Donner un pool tronqué à
3 013 donnes sans toucher ce drapeau produit :

```
  Pool has 3013 deals but 1000000 requested, generating more...
  --- Chunk 1 : generating 500000 deals (3013/1000000 total) ---
```

…c'est-à-dire **997 000 donnes fraîches ajoutées au fichier tronqué**, puis un
entraînement dont 99,7 % des épisodes retombent sur `dd_pts`. La troncature est annulée
en silence, et le fichier d'entrée est modifié au passage.

#### La recette validée de bout en bout

```bash
N=$(python3 -c "import struct;d=open('data/deals/scores_isdd_v2.sc','rb').read();\
nl=struct.unpack('<H',d[8:10])[0];print(struct.unpack('<I',d[10+nl:14+nl])[0])")
uv run python scripts/analysis/truncate_pool.py data/deals/base_5M.bin "$N" "data/deals/base_$N.bin"

./target/release/train_bid_nn --num-envs 256 --hidden 512 --layers 3 \
  --pool-size "$N" --pool-file "data/deals/base_$N.bin" \
  --scores data/deals/scores_isdd_v2.sc \
  --reward real --score-aware --sa-features-v3 --canonical --match-sim \
  --reward-clip 1.0 ...   # le reste comme scripts/training/v6_isdd.sh
```

Le contrôle qui dit que c'est bon, à lire dans les vingt premières lignes du log :

```
  Loaded 3013 deals in 0.0s                      ← PAS de « generating more »
  Activated score layer 'isdd_v2' (3523 deals)
  Canonical suit ordering: obs, mask and stored actions in canonical space
```

`--canonical` est le seul changement d'architecture de v7 (§ bid_v7_plan) : les six
flottants faits main ont été retirés, donc **v6 lui-même est le témoin à budget égal**.

## 10. Ce qui n'est pas décidé — trois mesures d'abord

| | quoi | coût | ce qu'elle décide |
|---|---|---|---|
| ~~**A**~~ | ✅ **faite le 2026-08-04** (§4) : oui, le préfixe est hors distribution, sur la **forme** (une seule annonce contre 11,9 % de réel, 0 % de contestation contre 81,3 %, 0 % de coinche contre 25,7 %) et non sur l'identité du preneur (camp bon à 89,4 %). A aussi **réfuté « v6 masqué sur la couleur »**, la variante qu'elle-même avait suggérée | 7 min, 0 GPU | a **fermé** la famille des enchères à atout imposé, retiré l'échelle valeur ↔ force, et fixé la forme du générateur : une case « or » par donne, trois « argent » |
| ~~**B**~~ | ✅ **faite le 2026-08-05** (§4) : le préfixe déplace l'étiquette de **+4,36 pt pour le preneur** (or contre fer, z = +6,7), et l'ordre des quatre rangs est celui que A prédisait | 33 min, 2 GPU | la hiérarchie de préfixes est **gardée**, et le fichier `.ranks` enregistre de quoi corriger l'écart entre cases plus tard |
| ~~**C**~~ | ✅ **faite le 2026-08-04** (§6) : `P(capot \| main)` à l'atout annoncé vaut **8,68 %**, pas 16,08 % ; **aucune** main ne dépasse 75 % et deux sur 1 200 dépassent 50 % | 9 min, **0 GPU** | `tail_100k` se filtre sur `eval_max` (facteur 18 entre tranches), et il enseigne à **ne pas** annoncer capot plutôt qu'à l'annoncer |

**A ne dispensait pas de B, et B a contredit ce que A laissait espérer.** A pariait que
si l'écart tombait sous le plancher de bruit, « la construction bon marché suffit et tout
le §4 devient une note de bas de page ». L'écart *est* petit devant le bruit par étiquette
(0,18×) — et la conclusion est pourtant l'inverse, parce que **le bruit se moyenne et le
décalage non**. Garder la prédiction ratée ici plutôt que la réécrire : c'est elle qui
montre où le raisonnement était faux.

**Ce que A aura coûté et rapporté, parce que c'est le patron à réutiliser.** Sept minutes
de CPU ont produit un résultat, puis tué la solution que ce résultat suggérait, avant
qu'elle entre dans un générateur ou consomme une heure de GPU. Deux choses l'ont permise
et aucune n'est optionnelle : la comparaison **appariée** (même donne, pas deux
marginales) et un **témoin qui doit rendre l'identité** — ici, v6 se rejouant lui-même à
99,99 %. Un chiffre de variante sans ce témoin ne distingue pas une propriété de
l'enchère d'un défaut du pilote.

⚠️ **La référence de bruit annoncée ici était la mauvaise.** Le plan visait les
**44,7 pts** de dispersion intra-main ([bid_v7_plan §2.8](../bid/bid_v7_plan.md)) — une
dispersion *entre donnes*, qui ne dit rien de la reproductibilité d'une étiquette. Le bon
plancher est celui que B a **mesuré avec son bras témoin** : **24,37 pts** entre deux
étiquetages de la même case. Une référence empruntée à une autre population n'est pas un
plancher ; il faut le mesurer dans l'expérience qui s'en sert.

## 11. Le risque principal, nommé

On échange **5 M × 4 étiquettes faibles** contre **600 k × 4 fortes** : 8× moins de
donnes.

[bid_v7_plan §2.8](../bid/bid_v7_plan.md) avait tranché « les donnes gagnent » — mais
cet arbitrage repose explicitement sur *« un bruit non biaisé ne biaise pas un
ajustement aux moindres carrés »*, mesuré en mondes uniformes. **Changer la source de
mondes change ce que vaut un monde, et l'argument ne se transporte pas tel quel.**

**Deuxième effet — mesuré, et beaucoup plus petit que craint** ([bench_class_coverage.rs](../../colver-core/src/bin/bench_class_coverage.rs)) :

| donnes étiquetées | mains | classes couvertes | vues 1 seule fois |
|---|--:|--:|--:|
| 100 k | 400 k | 56,46 % | 62,9 % |
| **500 k** | **2 M** | **97,59 %** | **7,7 %** |
| 1 M | 4 M | 99,78 % | 0,8 % |
| 5 M | 20 M | 100,00 % | 0,0 % |

À 500 k donnes, **90,1 %** des classes sont vues au moins deux fois et 42,3 % au moins
cinq. Le risque de couverture était donc surestimé, et l'erreur est identifiable : les
**6,2 M** de tirages du coupon collector ([bid_v7_plan §1.4](../bid/bid_v7_plan.md))
mesurent le coût du **dernier** coupon, pas celui des 97 premiers pourcents. La
distribution aide en plus — mesurée quasi **uniforme** (le centile le plus fréquent porte
2,5 % des mains, contre 1,0 % à l'uniforme exacte), donc la couverture monte vite et ne
traîne qu'à la toute fin.

Ce que ça retire à `tail_100k` : sa justification par la **couverture**. Ce qu'il lui
reste, et qui est mesuré : la strate décisive de §6 — et la mesure C a montré qu'elle se
filtre sur `evaluate_for_trump` sans simuler.

**Mitigation intégrée** : les couches composent par `offset`/`count`, et les 500 k
donnes sont **les mêmes** que celles de l'ancienne couche. L'A/B ancienne étiquette
contre nouvelle est donc direct, à donnes identiques — c'est la mesure qui dira si
l'échange valait le coup.

### Ce que l'A/B donne aux premières 1 000 donnes (2026-08-05)

*[check_score_layer.py](../../scripts/analysis/check_score_layer.py), lancé sur le
fichier **partiel** pendant la génération — un défaut systématique découvert à
100 000 donnes coûte tout ce qui a été calculé.*

| contrôle | résultat |
|---|---|
| valeurs arithmétiquement impossibles (163-251) | **0** — le résidu `quick_tricks` a disparu |
| cases avec un **vrai** préfixe | **2,36 / 4** — la mesure A prédisait 2,37 |
| contre la valeur DD, orienté preneur | **−8,76 pt** — le bon signe : à information incomplète le preneur rend moins que le double dummy |
| contre l'ancienne couche, orienté preneur | **+1,66 pt** (σ 30,6 ⇒ z ≈ 3,4) ; **7,6 %** d'étiquettes identiques |

**Et le contrôle qui compte, avec son plancher.** Une couche ne donne pas un niveau, elle
fait **choisir un atout** : le bidder compare les cases d'une même donne. Les deux couches
désignent le même meilleur atout dans **69,5 %** des (donne, camp) à ≥ 2 options.

Ce taux ne se lit pas seul. Deux étiquetages **du même procédé** sont déjà en désaccord :
la mesure B donne σ = 24,4 apparié, soit ~17,2 par étiquette, et re-bruiter la couche
neuve contre elle-même donne un plancher de **70,7 %**.

> **Mesuré 69,5 %, plancher 70,7 % : le test est au plancher.**

⚠️ **Ce n'est pas « les deux couches sont équivalentes ».** C'est un **null sans
puissance** : l'argmax d'une case est dominé par le bruit d'étiquetage, pour l'ancienne
comme pour la nouvelle. Conclure l'équivalence ici serait la même erreur que lire « pas
d'effet » dans un h2h d'arène trop court.

**Ce qui reste détectable, et qui décide** : le décalage **systématique** de +1,66 pt au
preneur. Le bruit se moyenne sur 500 000 donnes ; ce décalage-là, non. C'est aussi
pourquoi l'arbitrage de [bid_v7_plan §2.8](../bid/bid_v7_plan.md) — *« les donnes
gagnent »* — survit à ce constat : un bruit **non biaisé** ne biaise pas un ajustement,
donc il vaut mieux 500 000 étiquettes bruitées que 125 000 propres. Le lever du bruit
(4× les mondes pour le diviser par deux) coûterait 4× les donnes et n'achèterait rien.

**Corollaire méthodologique** : ne pas chercher à valider une couche par l'accord d'argmax.
Le seul juge qui reste est **en aval** — un bidder entraîné sur chacune, comparé en arène.

## 12. Ce qu'on ne fait pas, et pourquoi

- **Pas d'IS-DD en direct dans la boucle RL.** La boucle consomme ~11 000 donnes/s
  (431 pas/s × 256 envs ÷ ~10 actions par enchère) ; `gen_games_isdd` en produit 2,62.
  Facteur ~4 000, et un run complet demanderait ~700 M à 1 Md de donnes jouées.
- **Pas de corpus de trajectoires de match.** Un match de 2000 points est un
  *chaînage* : la donne *k+1* ne dépend de la donne *k* que par le score cumulé et le
  donneur. `--match-sim` le fait déjà, et **seul le bidder lit le score de match** —
  ni IS-DD ni playgen v2 ne le voient. Ce qui manque au label n'est pas la trajectoire,
  c'est le conditionnement au contrat (§3-4).
- **Pas de poids figés dans le fichier** (§6).
- **Pas de labels conditionnels au contrat** tant que la mesure B n'a pas dit que ça
  bouge quelque chose (§4).
