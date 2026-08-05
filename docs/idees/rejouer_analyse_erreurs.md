# Rejouer : compter les erreurs, montrer l'alternative, dire qui avait raison

*Écrite le 2026-08-05, **révisée le même jour** : les étapes 10.1 à 10.4 ont été
écrites dans la foulée et la croyance n°1 est mesurée. Les chiffres viennent de
`scripts/analysis/replay_error_scale.py` et `replay_error_grid.py` ; les
commandes sont données à chaque section et les runs sont dans
`docs/measurements/index.jsonl`.*

**État** : la bascule d'échelle, la classification à cinq catégories, le panneau
« Moments de la donne », le coût IS-DD par coup, l'« Analyse rapide » d'une
annonce (§7 bis) et la **courbe de la donne** (§7) sont **faits**. Restent les
variantes déroulées (§10.5) et l'exploration libre (§10.6).

**L'idée de départ** (utilisateur, 2026-08-05), en trois demandes successives :

1. Dire qu'il y a *tant* d'erreurs dans la donne, une erreur étant un coup que
   l'Oracle désapprouve, et pour chacune **tracer la ligne alternative** — la
   position résolue à l'avantage de celui qui s'est trompé, pour montrer comment
   ça aurait continué. Jeu de la carte seulement.
2. Plus tard : laisser l'utilisateur **tester une alternative** et explorer des
   lignes lui-même.
3. Croiser **deux évaluations** de la position — le solveur DD (« en jeu parfait,
   telle équipe gagne ») et IS-DD (« avec l'information disponible, c'est plutôt
   ça qui va se passer ») — parce que leur désaccord est en soi une information.
   Puis un compteur d'erreurs par siège, cliquable, pour naviguer de l'une à
   l'autre.

Les trois tiennent. Ce qui suit dit dans quel ordre, ce que ça coûte, et
**pourquoi la définition d'erreur actuelle est le préalable à tout le reste**.

---

## 1. Ce qui est déjà là — presque tout

Avant d'ajouter quoi que ce soit, l'inventaire, parce qu'il change ce qu'il
reste à faire :

| brique | où | ce qu'elle rend |
|---|---|---|
| coût DD par carte, carte optimale, catégorie | `analysis.py` → table `analysis` | `cost`, `best`, `category` par action |
| résumé par siège | `analysis._summarize` | coût total, moyen, comptes par catégorie |
| avis de DouDou50 / Oracle / Dédé par carte | `agent_review.py` → table `agent_review` | la carte que chacun aurait jouée |
| liste des coups colorée par catégorie | `replay.js::buildMovesList` | navigation au clic |
| creuser une position | `/analyse/jeu` + `card_analysis.py` | les deux Oracles, une ligne par carte légale |

**Le calcul n'est pas le problème.** Ce qui manque est une *hiérarchie* : la page
dit tout, à plat, et ne dit pas où regarder.

---

## 2. Le préalable : l'échelle actuelle désigne les mauvaises erreurs

`analysis.CATEGORIES` note les cartes en **points cartes**. Or le score de donne
est une fonction **en escalier** de ces points, pas une fonction linéaire. C'est
exactement l'argument qui a fait basculer IS-DD sur `PlayObjective::DealScore` le
2026-08-03 ; il n'a jamais été appliqué aux pages d'analyse.

### 2.1 Les quatre régimes

Pour un contrat de valeur `V` et `p` = points cartes du preneur :

| régime | preneur marque | défense marque | dépend de `p` ? |
|---|---|---|---|
| `p + belote < V` — **chute** | 0 | `162 + V×mult` | **non — plat** |
| `p + belote ≥ V`, non contré | `p + V` | `162 − p` | oui, **pente 2** sur l'écart |
| contré / surcontré réussi | `162 + V×mult` | 0 | **non — plat** |
| capot annoncé | plat des deux côtés | | non |

Trois conséquences :

1. **Sous le seuil, un point carte vaut rigoureusement zéro.** La défense qui
   fait chuter encaisse `162 + V` quel que soit le partage réel des plis.
2. **Au seuil, un point carte vaut `4V`.** Passer de `V−1` à `V` fait basculer
   l'écart de `−(162+V)` à `3V−162`, soit exactement `4V`.
3. **Dans un contrat normal tenu, `coût_score = 2 × coût_points_cartes`.** C'est
   le seul régime où les deux échelles disent la même chose. C'est aussi le plus
   fréquent, et c'est pourquoi l'échelle actuelle *a l'air* de marcher.

### 2.2 Ce que ça change, mesuré

```
uv run python scripts/analysis/replay_error_scale.py scale --deals 60 --seed 42
```

60 donnes (enchère bid v6, jeu DouDou50 aux quatre sièges), 1205 décisions non
forcées. Le barème réimplémenté dans le script est validé contre `env.rewards()`
sur **60/60** donnes — sans quoi le script refuse de conclure.

| | points cartes | score de donne |
|---|---|---|
| décisions à coût nul | 81,6 % | **90,9 %** |
| « fautes » (seuil haut) | 9 | 44 |

Et le chiffre qui tranche : **32 décisions sur 1057 que la page affiche
aujourd'hui « ✓ Meilleur coup » ou « ✓ Bon coup » coûtent au score de donne**,
les pires à `2 points cartes → 1264 points de score`, `3 → 684`, `2 → 480`. Le
sens inverse est rare : **1 « faute » sur 9 seulement** est gratuite.

La conversion étant **monotone non décroissante**, la meilleure carte ne change
pas — seule l'amplitude bouge. Mais c'est l'amplitude qui définit « erreur ».

### 2.3 Deux donnes à lire

```
uv run python scripts/analysis/replay_error_scale.py example --want bon   --seed 1
uv run python scripts/analysis/replay_error_scale.py example --want faute --seed 5
```

**Donne A — le coup noté « bon » qui perd la donne.** Contrat 100♥ par N-S.

```
  Nord   ♠ —        ♥ 10 R V 9  ♦ A V     ♣ R V
  Est    ♠ D 7      ♥ A D 8     ♦ R       ♣ 9 8
  Sud    ♠ 10 R V 8 ♥ —         ♦ 9 7     ♣ D 7
  Ouest  ♠ A 9      ♥ 7         ♦ 10 D 8  ♣ A 10
```

Pli 3, Sud défausse (il a une coupe à l'atout) :

```
   carte    pts cartes preneur           score N-S − E-O
    D♣            100        tenu             +138
    V♠             99        CHUTE            −262
    R♠             99        CHUTE            −262
    9♦             97        CHUTE            −262   ← joué
```

Trois lectures :

- La D♣ vaut exactement les 3 points qui manquent, et c'est la **seule** carte
  qui tient le contrat. L'échelle actuelle note ce coup **−3 → « bon coup »**.
  L'écart réel est de 400, soit `4V` avec `V = 100`.
- **Elle ordonne des coups identiques et confond des coups opposés.** V♠ et R♠
  laissent 99 au preneur, *plus* que le 9♦ à 97 : notées −1 contre −3, donc
  « meilleures ». En score de donne elles sont **strictement équivalentes** —
  99 < 100, le contrat est mort pareil.
- **Elle désigne la mauvaise erreur.** Le plus gros chiffre de la donne est le
  7♠ d'Est au pli 6, à −42 points cartes ; le contrat est alors acquis, on est en
  régime « pente 2 », ça coûte 84. Cinq fois moins que les coups des plis 3 à 5,
  que la même échelle note 3, 10, 10 et 7. **Un écran qui trie par points cartes
  met le pli 6 en tête et cache les quatre coups qui ont décidé la donne.**

**Donne B — la « faute » qui ne coûte rien.** Contrat 130♥ par E-O, Nord en
défense, pli 4 :

```
   carte    pts cartes preneur           score N-S − E-O
    A♠             90        CHUTE            +292
    V♣            100        CHUTE            +292
    A♦            115        CHUTE            +292   ← joué
    R♦            138        tenu             −244
    V♦            138        tenu             −244
```

La vraie décision est **binaire** : ne pas jouer ♦ dans le Roi ou le Valet. Parmi
les trois cartes qui font chuter, le choix est rigoureusement indifférent.
L'échelle actuelle note pourtant l'A♦ **−25 → « erreur »**, un cran au-dessus des
coups qui ont décidé la donne A. Confirmé par le résultat réel : 47-115 en points
cartes, **marqué 292-0**.

### 2.4 Ce que ça implique pour le code

- **Ne pas réécrire le barème côté Python.** Un binding qui rend les scores de
  donne par carte, réutilisant `scoring::deal_score_from_card_points` comme le
  fait `is_dd::world_value`, garde une seule implémentation. La copie du script
  de mesure n'est tolérable que parce qu'elle est validée à chaque run.
- **La belote se recalcule depuis les mains initiales** (`is_dd::world_belote`),
  pas depuis `state.belote` qui ne compte que ce qui a déjà été joué. Piège déjà
  documenté, à ne pas retomber dedans.
- **Le départage change.** En score de donne la classe des cartes optimales
  s'élargit (donne A : cinq cartes à −262). Désigner *une* carte comme « celle de
  l'Oracle » y est encore plus arbitraire qu'aujourd'hui — il faut afficher la
  classe (« une de ces trois »), pas le représentant que le tie-break a choisi.
- **Les points cartes restent la bonne échelle ailleurs.** Pour « quelle carte
  prend le plus de plis », `/analyse/jeu` a raison de les afficher. C'est la
  phrase « voici vos erreurs » qui doit basculer.

---

## 3. Les deux évaluations sont déjà calculées, et jetées

C'est le meilleur rapport valeur/coût de toute la fiche.

**DD.** `analysis._analyze_sync` appelle `env.solve_scores()` à chaque décision
et ne garde que `cost` et `best`. Le `max(scores)` du même appel **est** la
valeur DD de la position. Zéro calcul supplémentaire.

**IS-DD.** `Agent.decide` rend déjà `candidates` = (carte, score), et depuis le
passage à `PlayObjective::DealScore` ces scores sont en **écart de score de donne
N-S − E-O** (`agents.py:416`, `score_scale = "deal_score"`). Or
`agent_review._ask` fait `int(agent.decide(env)["action"])` — **et jette tout le
reste**.

Donc les deux évaluations sont dans la **même unité** une fois le §2 appliqué,
elles se superposent directement, et il n'y a que de la plomberie à écrire.

### 3.1 La grille de lecture

À chaque coup, deux écarts : ce qu'il coûte en DD (vérité omnisciente) et en
IS-DD (ce que le siège au trait pouvait anticiper).

| coût DD | coût IS-DD | ce que ça dit | quoi en faire |
|---|---|---|---|
| 0 | 0 | rien à voir | ne rien afficher |
| **> 0** | **> 0** | **vraie erreur** — visible depuis ce qu'il savait | **la compter** |
| > 0 | 0 | malchance — la carte était juste, la donne ne l'était pas | montrer, ne pas compter |
| 0 | > 0 | coup heureux — ça paraissait mauvais et ça ne l'était pas | montrer, ne pas compter |

C'est la réponse à « parfois alignées, parfois pas ». Et c'est plus fin que
comparer les *cartes choisies* : on a une amplitude, donc on peut dire « tu as
perdu 400 points et tu ne pouvais pas le savoir » — une information, pas un
reproche.

**Sans ce filtre, le compteur accuse un joueur d'erreurs qu'il ne pouvait pas
voir.** L'Oracle voit les quatre mains ; c'est tout le propos des deux Oracles de
[web_analyse_jeu.md](../web_analyse_jeu.md) §2.

### 3.1 bis La grille remplie — le filtre sert, et il réserve une surprise

```
uv run python scripts/analysis/replay_error_grid.py --deals 25 --seed 17
```

⚠️ Demande le sidecar playgen. 485 décisions sur 25 donnes, budget IS-DD de
500 ms/carte (celui d'`agent_review` en production) :

| | compte | part des décisions |
|---|---|---|
| **erreur** — les deux désapprouvent | 40 | 8,2 % |
| **malchance** — l'Oracle seul | 12 | 2,5 % |
| **coup heureux** — Dédé seul | **76** | 15,7 % |
| rien à signaler | 357 | 73,6 % |

**23,1 % des écarts DD sont de la malchance** (12 sur 52). Presque un écart sur
quatre n'était pas visible depuis le siège qui jouait : le filtre change
réellement le compteur, et la croyance n°1 est levée. Le coût DD ne permettrait
pas de les séparer — médiane 20 pour les erreurs, 16 pour les malchances.

Parmi les 40 erreurs : **33 imprécisions et 7 fautes décisives**. La scission
demandée est donc peuplée des deux côtés, à peu près une sur six.

**La surprise est ailleurs** : les « coups heureux » sont **deux fois plus
nombreux que les erreurs**. Deux lectures possibles, et la mesure ne les sépare
pas — de la vraie chance, ou une faiblesse de Dédé à 500 ms et ~40 mondes. Un
juge imparfait désapprouve parfois à tort, et cette case-là est précisément
celle où son erreur à lui se déverse.

Conséquence d'interface, tirée de ce chiffre : **les coups heureux ne vont pas
dans la liste des moments**. Les y mettre noierait les 52 coups qui ont coûté
quelque chose sous 76 coups qui n'ont rien coûté. Ils restent annotés sur le
coup lui-même et dans la couleur de la liste.

### 3.2 Le piège : IS-DD est seat-bound

`agent_review` construit quatre instances, une par siège, et n'interroge que
celle du siège au trait. Deux points consécutifs ne sont donc **pas dans le même
repère informationnel** : la *pente* entre eux ne veut rien dire, et une courbe
continue tracée là-dessus serait un artefact.

Deux sorties, dans cet ordre :

- **Marqueurs par coup**, chacun lu depuis son auteur. Gratuit — c'est déjà
  calculé — et suffisant pour la grille du §3.1.
- **Une courbe continue pour un siège donné** (celui du joueur qui regarde) :
  il faut interroger IS-DD à *toutes* les positions, cartes forcées et tours des
  autres compris. ~32 recherches au lieu de ~20, donc la revue passe de ~9 s à
  **~16 s** au budget `COLVER_REVIEW_ISDD_MS` actuel. Extension, pas point de
  départ.

---

## 4. La ligne alternative : le coût n'est pas un critère

```
uv run python scripts/analysis/replay_error_scale.py variation --deals 12 --seed 7
```

Dérouler une ligne DD jusqu'au 8e pli, en bouclant `action_oracle_dd()` :

| cartes en main | médiane | p90 | max |
|---|---|---|---|
| 8 | 30,2 ms | 71,2 ms | 143,9 ms |
| 7 | 4,6 ms | 14,5 ms | 28,0 ms |
| 6 | 1,8 ms | 3,4 ms | 5,0 ms |
| ≤ 5 | < 1 ms | | |

**Toutes les variantes d'une donne : 0,12 s en médiane, 0,35 s au pire.** Moins
que les quatre solves d'`_oracle_bids` que `analysis.py` fait déjà.

⚠️ **Dispersion.** Deux exécutions du même code à quelques minutes d'écart ont
rendu 23,9 et 30,2 ms sur la ligne « 8 cartes », soit **26 % d'écart** — cohérent
avec les ~20 % que le projet documente pour toute mesure au chronomètre sous
charge. Seul l'ordre de grandeur compte ici, et il est le même dans les deux
runs : deux chiffres significatifs seraient de trop.

Conséquences : pas de « à la demande », pas de binding Rust dédié, ça rentre dans
le blob `analysis` au prochain bump de version. Une variante pèse ~30 octets en
cartes ; l'envoyer au client est gratuit.

### 4.1 Il faut deux lignes, pas une

La suite réelle contient d'autres erreurs des deux camps. La comparer à une
variante parfaite fait dire n'importe quoi à l'écart. Le comparatif honnête est
**ligne DD après le coup joué** contre **ligne DD après le coup Oracle** : leur
écart *est* exactement le chiffre affiché. Il faut alors dire clairement que la
première n'est pas ce qui s'est passé.

La donne A l'illustre : le DD dit « chute après 9♦ », et **le contrat a été tenu
en réalité** (146-16 → marqué 246-16) parce qu'Ouest a rendu l'erreur au coup
suivant (9♠, −414). Une variante annoncée comme « voilà ce qui serait arrivé »
mentirait ; c'est un **plafond**, et dans la variante les adversaires jouent en
DD omniscient.

---

## 5. DouDou50 comme raccourci : impasse mesurée

**L'intuition de départ** : dérouler les lignes hypothétiques avec DouDou50
« pour que ce soit rapide ». Elle est inversée.

```
uv run python scripts/analysis/replay_error_scale.py rollout --deals 8 --seed 11
```

Même position, déroulé complet DouDou50 contre un solve DD :

| cartes en main | DouDou50 | solve DD | rapport |
|---|---|---|---|
| 8 | 8,03 ms | 5,14 ms | 1,6× |
| 6 | 6,03 ms | 0,79 ms | 7,6× |
| 4 | 4,45 ms | 0,14 ms | 32,0× |
| 2 | 2,31 ms | 0,12 ms | 20,1× |

⚠️ **La ligne « 8 cartes » est la moins stable** : un second run a donné 2,2× au
lieu de 1,6× (le solve DD y varie de 3,5 à 5,1 ms selon la charge, DouDou50
bouge à peine). Les lignes de fin de donne, elles, sont reproductibles à ~10 %.
La conclusion ne dépend pas de ce chiffre-là.

**DouDou50 est plus lent que le solveur partout**, d'un facteur ~2 au pli 1 à
~30 en fin de donne, et l'écart se creuse à mesure que la donne se vide. Un
déroulé, c'est 32
passes d'un réseau 411→1024³→32 dont le coût ne baisse pas avec la donne, alors
que le solve s'effondre à des microsecondes dès qu'elle se vide. Même famille que
« le mur est le sidecar, pas le solveur » de
[isdd_games.md](../data_gen/isdd_games.md).

**Ça ne tue pas l'idée, ça la reclasse.** Une ligne DouDou50 n'est pas un
raccourci vers IS-DD, c'est une **troisième** évaluation — « ce qui arrive avec
des joueurs corrects mais non omniscients », un échantillon déterministe unique
sur la vraie donne. Elle a un sens propre, et pour « telle équipe est censée
gagner » elle est sans doute plus honnête que le DD. À garder pour plus tard :
deux évaluations qui se contredisent, c'est déjà beaucoup à expliquer à l'écran ;
trois, c'est illisible.

---

## 6. Combien d'erreurs par donne

```
uv run python scripts/analysis/replay_error_scale.py errors --deals 40 --seed 23
```

Décisions au coût non nul **en score de donne**, 40 donnes par configuration :

| | moyenne/donne | médiane | max | donnes sans erreur |
|---|---|---|---|---|
| 4× DouDou50 | 2,05 | 2 | 7 | 14/40 (35 %) |
| un siège à l'heuristique | 2,30 | 2 | 7 | 11/40 (28 %) |

Réparties assez également entre les quatre sièges (25 / 23 / 15 / 19 sur ~200
décisions chacun) : ce n'est pas un artefact d'un siège particulier.

L'estimation initiale de l'utilisateur (3 à 10) décrit la **queue**, pas le cas
courant. Deux conséquences pour l'écran :

- **Un tiers des donnes n'ont aucune erreur.** Le compteur doit être beau à
  zéro : « donne jouée sans faute » est un résultat, pas un vide.
- Le pire cas reste à 7-8, donc une liste dépliée tient toujours à l'écran. Pas
  de pagination, pas de « top 3 ».

⚠️ **C'est par donne.** Sur une partie en 2000 points (~10-12 donnes) ça ferait
20 à 30 erreurs — un cumul qui aurait sa place sur la feuille de marque
`/analyse/partie`, pas dans Rejouer.

⚠️ **Ces donnes sont jouées par des bots.** Un humain moyen fera davantage
d'erreurs, et le passage de DouDou50 à l'heuristique n'a coûté que +0,3 par donne
— c'est une borne basse, pas une prédiction pour un joueur du site.

---

## 7. L'interface

Ce que l'utilisateur décrit — compteur en haut, par camp ou par siège, cliquable
pour aller d'une erreur à l'autre — est la bonne forme. Quatre précisions.

**Le vocabulaire** (livré ; les libellés sont isolés dans `CATEGORY_UI`) :

| clé | libellé | badge | quand |
|---|---|---|---|
| `parfait` | Meilleur coup | ✓ | rien à gagner ailleurs |
| `imprecision` | Imprécision | `?!` | des points perdus, le contrat tient |
| `decisive` | Faute décisive | `??` | le contrat bascule |
| `malchance` | Malchance | `≈` | l'Oracle seul désapprouve |
| `aubaine` | Coup heureux | `!` | Dédé seul désapprouve |

Deux familles qui ne se mélangent pas, et **c'est la palette qui doit le dire** :
les trois premières sont des jugements (jaune → rouge), les deux dernières des
explications (bleu, violet — délibérément hors de la gamme du blâme). Sans ça,
« Malchance » se lit comme un cran de faute alors que le coup était bon.

**Par siège, regroupé par camp.** Le résumé actuel est déjà par siège, et en solo
trois sièges sur quatre sont des bots : un total par camp cache le seul chiffre
qui intéresse le joueur.

**Le compteur arrive en deux temps, et il faut l'assumer.** L'écart DD est
disponible avec l'analyse (~1 s) ; le filtre IS-DD attend `agent_review` (~9 s).
Donc afficher « 3 écarts », puis raffiner en « 3 écarts, dont 2 évitables ».
Annoncer un compteur définitif puis le voir baisser tout seul serait pire que
d'attendre.

**Pour la position, une courbe en points cartes du preneur** (livré 2026-08-05,
après une discussion sur trois formes possibles). Un seul axe, trois tracés :

- **l'aire** — ce que le preneur a déjà ramassé, huit marches ;
- **la ligne** — ce qu'il fera en jeu parfait depuis chaque position, **verte
  au-dessus du seuil, rouge en dessous** ;
- **l'horizontale** — le seuil, qui monte par paliers pendant l'enchère puis se
  fige. L'enchère est ainsi la première moitié du même graphe, dans la même
  unité, sans second tracé.

Les deux premiers convergent forcément au 8ᵉ pli — vérifié sur 6 donnes, et à
l'écran : ramassé 111, projection finale 111. Le passage de la ligne sous le
seuil **est** une faute décisive, donc la courbe et le panneau des moments
racontent la même histoire par construction. Sur la donne regardée, la
projection fait 124 → 107 → 121 → 103 → 111 : le contrat bascule quatre fois, et
ça se voit d'un coup d'œil.

**Un seul camp.** Les points cartes sont à somme constante (162) : la courbe de
l'autre camp en est le miroir exact, elle doublerait l'encre sans ajouter un
bit. Même raison que le « orienter par le preneur, jamais par N-S ».

Deux formes ont été écartées. Le **regret cumulé** (zéro = jeu parfait) : la
somme des regrets n'est pas un total — chacun est mesuré indépendamment à sa
position, donc deux coups peuvent perdre la même donne et leurs coûts
s'additionner au-delà de ce que la donne vaut (vu : 904 sur une donne à ~460).
Un tableau peut le dire en note ; une courbe cumulative le raconterait comme une
accumulation réelle, à chaque pixel. Et **la valeur DD en écart de score signé** :
correcte, mais elle demande d'expliquer le double-mort, là où « voilà où on en
est, voilà où on va, voilà où il faut arriver » se lit sans glossaire.

**Pas d'IS-DD sur ce graphe** : sa valeur est seat-bound, donc deux points
consécutifs ne sont pas dans le même repère informationnel et la pente entre eux
ne veut rien dire. Il reste en marqueur sur le coup.

**Le clic amène sur la position *avant* le coup** — c'est la décision qu'on veut
revoir, pas son résultat. Là, tout est déjà en place : `best` donne la carte de
l'Oracle, les trois bots donnent leur avis, la variante montre la suite.

---

## 7 bis. ✅ Les annonces : le même écran, mais pas le même juge — **fait**

*(demande utilisateur, 2026-08-05, implémentée le même jour)*

**Livré** : un bouton « Analyse rapide » dans le panneau du coup, WS
`replay_bid_quick`, 160 donnes jouées par annonce (~4 s), résultat mis en cache
par `sim_cache` sur `(main, enchère précédente, annonce forcée)` — donc
réutilisable d'une donne à l'autre. Il rend deux chiffres et **simule aussi
l'annonce de v6 quand elle diffère**, ce qui met les deux au même barème.

Trois cas vus à l'écran, qui montrent que la lecture tient :

| situation | ce que dit l'analyse rapide |
|---|---|
| 110♠ annoncé, v6 d'accord | passe 25 %, +34 pts (une seule ligne) |
| 110♣ annoncé, v6 disait 100♣ | joué 43 % / +72 · **v6 57 % / +138** — v6 avait raison |
| **passe**, v6 disait 130♣ | **v6 : passe 5 %, −346 pts** — le passe était juste, v6 se trompait |

Le dernier cas est celui qui justifie tout : un écran qui note les annonces au Q
de v6 aurait marqué ce passe « erreur ».

**Deux dénominateurs différents, et c'est voulu** (épinglé par
`tests/test_quick_bid.py`) : le taux de réussite est *conditionnel* — forcer une
annonce n'empêche pas les adversaires de surenchérir, donc il porte sur les
donnes où le camp garde le contrat, et l'infobulle dit à quelle fréquence c'est
arrivé. L'espérance, elle, porte sur **toutes** les simulations, surenchères et
donnes passées comprises. Le premier juge le contrat, le second juge la
décision.

**Ce qui reste ouvert** : le seuil. Aucune mesure ne dit combien de points
d'espérance perdus font une faute d'annonce — à la carte, « le contrat bascule »
est un fait binaire ; ici il faut choisir. Tant que ce seuil n'existe pas, les
annonces **n'entrent pas** dans le compteur d'erreurs ni dans les moments de la
donne : le bouton informe, il ne juge pas.

---

### Le raisonnement d'origine

Tout ce qui précède ne parle que du jeu de la carte. Les annonces ont déjà leur
ligne dans Rejouer, mais **jugée par le Q de bid v6** (`model_best`, `q_best`) —
c'est-à-dire en supposant que v6 a raison. Or v6 se trompe aussi, et un écran
qui appelle « erreur » tout désaccord avec lui n'enseigne rien : il apprend à
imiter un bidder, pas à annoncer.

La machinerie du bon juge existe déjà, sur `/analyse/annonces` : **taux de
réussite en pourcentage + espérance de points**, sur des mondes échantillonnés.
C'est objectif au sens où ça ne dépend d'aucun modèle de référence, et ça note
donc **les humains et v6 sur le même barème**. Les deux chiffres se lisent
ensemble : un contrat qui passe 55 % du temps peut être meilleur qu'un qui passe
80 %, si l'espérance suit.

Ce que ça demande, dans l'ordre :

1. Décider le **seuil d'erreur d'annonce**. Il n'est pas donné par la mesure : à
   la carte, « le contrat bascule » est un fait binaire ; à l'annonce, il faut
   choisir combien de points d'espérance perdus font une faute. La marche de
   `4V` n'existe pas ici.
2. Réutiliser `annonces_sim` hors de la page annonces, en le **cachant** — c'est
   plusieurs centaines de déroulements par annonce analysée, sans commune mesure
   avec le solve d'une carte.
3. Ne juger **que les annonces qui étaient des décisions**. Un passe forcé n'en
   est pas une (`only_pass_is_legal`), et la plupart des passes ne se discutent
   pas.

Le point 2 est ce qui bloque : à ~200-1000 sims par annonce et une dizaine
d'annonces par donne, ce n'est pas du même ordre que la seconde que coûte
l'analyse des cartes.

**Et la sortie est un bouton, pas un balayage** (proposition utilisateur,
2026-08-05) : un « Analyse rapide » sur l'annonce courante, qui rend les deux
chiffres *en place* dans le panneau du coup, sans quitter Rejouer. Ça retourne
le problème — au lieu de payer une dizaine d'annonces par donne pour toutes les
donnes ouvertes, on paie **une** annonce, celle que le joueur regarde, quand il
le demande. Le coût cesse d'être un obstacle, et le résultat se met en cache par
`(main, enchère précédente)` — la même clé que les mains enregistrées de la page
annonces, donc réutilisable d'une donne à l'autre.

Le lien « Analyser cette annonce → » vers la page complète reste à côté : le
bouton donne le chiffre, le lien donne le tableau.

## 8. L'exploration libre (demande 2) change l'architecture de la 1

L'exploration, c'est une pile de coups au-dessus d'une position plus quelqu'un
qui complète. **La variante du §4 en est le cas dégénéré** : pile = un coup,
complété par DD. Construire 1 comme un cas particulier de 2 évite d'écrire deux
fois le moteur de variantes.

Où : le repérage et la variante dans **Rejouer** (c'est là qu'on arrive, le tapis
et la liste des coups y sont), l'exploration libre dans **`/analyse/jeu`**, qui a
déjà le CFN complet, l'index d'action, les deux Oracles et le tableau par carte.
Rejouer = repérer, `/analyse/jeu` = creuser.

Question non tranchée : **qui joue les trois autres sièges** pendant
l'exploration ? DD parfait (cohérent avec les chiffres affichés, instantané),
DouDou50 (réaliste, ~5 ms/coup), ou Dédé (fort mais ~1 s/coup et il faut le
sidecar). Probablement DD par défaut avec un commutateur.

---

## 9. Ce qu'on croit sans l'avoir mesuré

1. ✅ **Que le filtre IS-DD retire une part utile des écarts DD.** **Mesuré**
   (§3.1 bis) : 23,1 % des écarts DD sont de la malchance, et le coût DD ne
   permet pas de les distinguer. Le filtre sert. En revanche la mesure a ouvert
   une question que la fiche n'avait pas prévue — 76 « coups heureux » contre 40
   erreurs, sans qu'on sache départager la chance de la faiblesse de Dédé.
   La mesure qui trancherait : refaire la grille à un budget IS-DD nettement
   plus élevé (2000 ms) et regarder si cette case se vide. Si elle se vide,
   c'était Dédé ; sinon, c'est le jeu.
2. **Que les joueurs humains font plus d'erreurs que 2,3 par donne.** Plausible,
   mais la seule dégradation testée (un siège à l'heuristique) n'a rendu que
   +0,3. La base de prod permettrait de le mesurer pour de vrai.
3. **Que la bande tenu/chuté se lit mieux qu'une courbe.** Opinion de conception,
   appuyée sur la forme en escalier mais pas sur un test d'usage.
4. **Que les seuils de catégorie se transposent.** `CATEGORIES` a des bornes
   calibrées en points cartes (4 / 14 / 29). Les équivalents en score de donne ne
   sont pas de simples multiples : la marche vaut `4V`, donc tout coup qui fait
   basculer le contrat est au-delà de n'importe quel seuil. Il faudra peut-être
   deux familles — « a fait basculer la donne » et « a coûté des points » —
   plutôt qu'une échelle graduée.
5. **Que le coût de la variante reste négligeable sur la prod.** Mesuré sur la
   machine de dev, à un solve à la fois. Le serveur web sert d'autres donnes en
   même temps.

---

## 10. Prochaines étapes

Chacune est utile seule, dans cet ordre.

### 10.1 ✅ Basculer le coût en score de donne — **fait**

`scoring.rs` gagne `total_card_points`, `final_belote`, `deal_score_delta` et
`contract_made`, toutes publiques : les deux premières **descendaient d'`is_dd.rs`
où elles étaient privées**, le barème n'ayant le droit d'exister qu'une fois.
`solve_scores` rend désormais `deal_scores` et `contract_made` **depuis le même
solve** — aucune recherche supplémentaire, juste de l'arithmétique par carte.
`analysis.py` passe en v7 et les deux exceptions de `_is_fresh` sont retirées,
comme leur commentaire l'annonçait.

Épinglé par quatre tests Rust, dont `deal_score_is_flat_below_the_contract` et
`deal_score_step_at_the_threshold_is_four_times_the_contract` — les deux
propriétés qui motivent tout le changement. Le binding est validé contre la
conversion Python du script de mesure (elle-même validée contre `rewards()`) :
**1064 cartes, 0 écart**.

### 10.2 ✅ Compteur d'erreurs + navigation — **fait**

Panneau « Moments de la donne » : compteur par siège groupé par camp, liste
triée par coût, clic vers la position **d'avant** le coup. Le compteur ne compte
que les fautes — malchance et coup heureux sont des explications, les additionner
serait un contresens.

### 10.3 ✅ Mesurer le filtre IS-DD — **fait** (§3.1 bis)

23,1 % des écarts DD sont de la malchance. `scripts/analysis/replay_error_grid.py`.

### 10.4 ✅ Exposer les deux évaluations — **fait pour les marqueurs**

`agent_review` garde `isdd_cost` par coup (v3) au lieu de jeter `candidates`, et
`replay.js` croise les deux avis. **La bande tenu/chuté n'est pas faite** — c'est
ce qui reste de cette étape.

### 10.5 Les variantes déroulées (~1 j, sans GPU)

Les deux lignes DD par erreur, dans le blob. Puis, séparément, les rejouer sur le
tapis — là il faut des états complets, donc une requête dédiée.

### 10.6 L'exploration libre (gros, à cadrer à part)

Dans `/analyse/jeu`. À ne pas commencer avant que 10.5 ait figé la forme du
moteur de variantes.

---

## Fichiers concernés

- `python/colver/web/analysis.py` — `CATEGORIES`, `_analyze_sync`, `_summarize`,
  `ANALYSIS_VERSION`, `_is_fresh`, `_DD_COMPATIBLE_VERSIONS`
- `python/colver/web/agent_review.py` — `_ask` (jette `candidates`), `_Runner.step`
- `python/colver/web/agents.py` — `decision_stats`, `score_scale`
- `python/colver/web/card_analysis.py` — le moteur de `/analyse/jeu`
- `python/colver/web/static/js/views/replay.js` — `CATEGORY_UI`,
  `renderAnalysisSummary`, `buildMovesList`, `replayRenderMoveStats`
- `colver-core/src/engine/scoring.rs` — `deal_score_from_card_points`
- `colver-core/src/search/is_dd.rs` — `world_value`, `world_belote`,
  `total_card_points` : le précédent exact de la conversion à faire
- `colver-py/src/lib.rs` — `solve_scores`, `action_oracle_dd`, `Agent::decide`
- `scripts/analysis/replay_error_scale.py` — les cinq mesures de l'échelle
- `scripts/analysis/replay_error_grid.py` — la grille DD × IS-DD (**sidecar
  requis**), qui passe par `analysis._analyze_sync` et `agent_review._Runner`,
  donc par le code de production
- [web_analyse_jeu.md](../web_analyse_jeu.md) — les deux Oracles, à ne jamais
  fusionner
- [classement_et_scoring.md](../classement_et_scoring.md) — §3.3, le niveau 3
  (score de partie) que `DealScore` ne voit toujours pas
