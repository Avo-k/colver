# Classement et scoring — état, défauts, et ce qu'il faudrait

Ce document rassemble tout ce qu'on sait du classement Elo du web et de la fonction
d'utilité que les bots optimisent. Les deux sujets sont le même : **on ne peut pas classer
des joueurs sur un objectif qu'on n'a pas défini**, et il se trouve que l'objectif
qu'optimise Dédé n'est pas celui qui décide une partie.

Tout ce qui est chiffré ici est mesuré ; ce qui ne l'est pas est marqué comme tel.

---

## 1. Ce que fait le système aujourd'hui

`python/colver/web/elo.py`, ~190 lignes :

- **une donne = un match noté** (pas la partie en 1000/2000) ;
- **équipe = moyenne des deux partenaires**, espérance Elo classique sur cet écart ;
- **score 1 / 0** (0,5 seulement si `rewards()` est exactement égal — jamais vu) ;
- **K = 32 pour un humain, 8 pour un bot** ; un bot occupant plusieurs sièges voit ses
  deltas sommés ;
- départ à 1000, `rate_game` idempotent, backfill au démarrage.

### État de la base de prod (3 août 2026, 1 073 donnes valides, 4 comptes, 2 bots)

| Entité | Elo affiché | Donnes notées | Parcours (min → max) |
|---|---|---|---|
| Darju | 1175,7 | 166 | 905 → 1180 |
| Dédé (bot) | 1044,4 | 881 | 1000 → 1119 |
| DouDou (bot) | 988,6 | **11** | — |
| Julesder | 954,2 | 649 | 817 → 1097 |
| Leuy | 871,2 | 141 | 773 → 1065 |
| Avo-k | 866,9 | 49 | 829 → 984 |

Bilans en solo (un humain, trois sièges Dédé) et Elo d'équilibre correspondant :

| | donnes | gagnées | Elo d'équilibre¹ | IC 95 % | affiché |
|---|---|---|---|---|---|
| Julesder | 577 | 44,9 % | 973 | [915, 1030] | 954 |
| Darju | 151 | 49,7 % | 1040 | [928, 1151] | **1176** |
| Leuy | 55 | 41,8 % | 930 | [730, 1112] | 871 |
| Avo-k | 21 | 28,6 % | 726 | [251, 1015] | 867 |

¹ le code calcule l'espérance sur la *moyenne* des partenaires, donc un humain assis à côté
de Dédé ne pèse que la moitié de l'écart d'équipe : l'Elo individuel d'équilibre vaut
`Elo(Dédé) + 2 × elo(taux observé)`. La formule est cohérente, c'est bien son point fixe.

### Le mouvement du classement est du bruit, et c'est mesuré

Simulation d'un joueur de niveau *exactement* égal au bot (50 % vrai), adversaire ancré,
K = 32, 650 donnes, 4 000 tirages :

```
K=32  n=650   écart-type du classement final = 55 Elo
              amplitude médiane du parcours  = 271 Elo
```

Amplitudes réellement observées : Julesder **280**, Darju **275**, Leuy **292**. Darju à
1176 n'a pas progressé — il a 75 V / 76 D en solo et une série chaude sur ses 40 dernières
donnes.

Précision atteignable à formule inchangée :

```
  100 donnes -> ± 136 Elo        1000 donnes -> ± 43 Elo
  200 donnes -> ±  96 Elo        2000 donnes -> ± 30 Elo
  500 donnes -> ±  61 Elo        5000 donnes -> ± 19 Elo
```

L'écart affiché entre Darju (1176) et Leuy (871) — 305 points — tient entièrement dans les
barres d'erreur combinées.

---

## 2. Les six défauts du système actuel

1. **K = 32 sur une donne.** Calibré pour une partie d'échecs, où le résultat est ~du
   signal. Une donne de contrée est ~de la distribution.
2. **Score binaire.** Gagner de 10 points ou de 400 vaut pareil ; la marge est jetée.
3. **Un siège sur quatre.** L'écart individuel est dilué de moitié par le partenaire, donc
   le bruit double quand on le reconvertit en Elo individuel.
4. **Pas d'ancre.** Dédé flotte (1000 → 1044, pic 1119). Tout le monde est mesuré contre un
   mètre-étalon qui bouge : si des joueurs plus faibles arrivent, Dédé monte et tous les
   humains existants se dévaluent en silence. En prime `K_USER = 32` contre `K_BOT = 8`
   rend le système **non conservatif** (±24 points créés ou détruits par donne).
5. **Dédé et DouDou ne se rencontrent jamais.** `pacing.py` assoit le *même* bot sur les
   quatre sièges IA, donc leur écart n'est estimé qu'à travers les humains — et DouDou
   repose sur **11 donnes**. Son 988,6 ne veut rien dire (±400 Elo au bas mot).
6. **`elo_ratings.games` compte des sièges, pas des donnes.** Dédé affiche 2 540 pour
   881 donnes jouées. Cosmétique, mais la page ment.

---

## 3. Le vrai problème : trois objectifs emboîtés, et on optimise le mauvais

C'est le cœur du sujet, et il explique les coups « contre-intuitifs » qu'on observe.

### 3.1 Ce que le barème dit réellement

`engine/scoring.rs::compute_deal_score`, contrat non contré, preneur = N-S, valeur `V`,
points cartes du preneur `x` (sur 162, ou 252 sur capot) :

| cas | N-S | E-O |
|---|---|---|
| `x + belote ≥ V` (réussi) | `x + V + belote` | `162 − x + belote` |
| `x + belote < V` (chute) | **0** | **`162 + V + belote`** |

D'où l'écart N-S − E-O en fonction de `x` :

```
  x < V   :  −(162 + V)          ← CONSTANT. pente 0.
  x ≥ V   :  2x + V − 162        ← pente 2.
  saut en x = V : 4V             ← pour V = 100, un saut de 400 points.
```

**Une fois la chute acquise, un point carte de plus vaut exactement zéro.** La défense
marque `162 + V` quel que soit le partage réel des plis. Et près du seuil, un point carte
vaut jusqu'à quatre fois la valeur du contrat.

### 3.2 Ce qu'IS-DD optimise

**Corrigé le 2026-08-03 : le défaut est `PlayObjective::DealScore`.** Chaque monde est
converti en écart de score de donne N-S − E-O — contrat réussi ou chuté, valeur du
contrat, contré/surcontré, capot, dix de der, belote — **avant** d'entrer dans la moyenne
(`IsDdSearch::world_value`). Ce qui suit décrit l'ancien défaut, `card_points`, encore
atteignable par `[play] objective = "card_points"` (bot `web_dede_cardpts`).

Jusque-là `search/is_dd.rs` agrégeait `ns_pts` — les **points cartes** rendus par
`solve_with_scores` — et prenait l'argmax :

```rust
score_sum[card] += ns_pts as f64 * cw;   // points cartes, 0-252
...
avg = score_sum[card] / weight_sum[card];
```

Dédé maximisait donc l'espérance de points cartes, jamais le score de donne : il ne savait
ni si le contrat était déjà tombé, ni s'il était déjà assuré, ni où était le seuil. C'est
l'équivalent exact du *money game* au backgammon — jouer chaque donne comme si les points
comptaient linéairement et indéfiniment.

Les trois cas signalés à l'usage tombaient pile là-dessus :

| situation | ancien objectif (`card_points`) | objectif actuel (`deal_score`) |
|---|---|---|
| **l'adversaire a déjà chuté** | continue à se battre pour des points | la pente est nulle, tous les coups se valent — plus d'air « au hasard » à défendre |
| **en défense, près du seuil** | maximise E[x] | maximise **P(x < V)** : une ligne qui sécurise 82 points bat une ligne à 90 de moyenne qui descend parfois à 75 |
| **en attaque, contrat déjà assuré** | maximise E[x] | identique — la pente est bien 2 au-dessus du seuil, l'ancien objectif était déjà correct ici |

### 3.2 bis Pourquoi c'est l'agrégation, et pas le solveur

L'écart de score est une fonction **monotone non décroissante** des points cartes du
preneur. Dans un monde déterminisé, où tout est décidé, les deux objectifs classent donc
les coups à l'identique, et le camp qui minimise l'un minimise l'autre : `solve_with_scores`
n'a rien à savoir du contrat et n'a pas bougé.

L'écart n'existe qu'à la moyenne, parce que `E[f(x)] ≠ f(E[x])` dès qu'il y a une marche.
Trois mondes à 90/70/70 sous un contrat à 80 donnent une espérance de 76,7 — « chute » —
alors que la bonne lecture est un tiers de contrat réussi contre deux tiers de chute, et
que la marche de `4V` paie largement la tentative. **C'est la raison pour laquelle on ne
rattrape pas ça après coup sur une moyenne de points cartes.**

Corollaire sur la belote : `scoring.rs` la compte dans `taker_total` pour décider de la
réussite, donc elle ne vaut pas « 20 points de plus au bout » — elle **déplace le seuil de
20 points**. `world_belote_for` la recalcule dans chaque monde depuis `hands | played_by`,
et non depuis `state.belote` qui ne compte que ce qui a déjà été joué. Épinglé par
`belote_moves_the_contract_threshold_not_just_the_total`.

### 3.3 Trois objectifs, du plus faux au plus juste

1. **Points cartes** — ce qu'optimise le solveur DD, et donc Dédé. Linéaire, sans seuil.
2. **Score de donne** — marche + rampe, saut de `4V` au seuil. C'est l'objectif correct
   d'une **donne isolée**.
3. **Équité de match** — `P(gagner la partie | score courant)`. Au-dessus de 950-200 la
   dérivée est quasi nulle : marquer 300 de plus ne change rien. C'est l'objectif correct
   **dans une partie**.

Chaque niveau est strictement meilleur que le précédent. **Depuis le 2026-08-03 le jeu de
la carte est au niveau 2** ; l'enchère est partiellement au niveau 3 (bid v6 lit une obs
score-aware à 117 dimensions, et `AgentTable.set_scores` lui passe le score cumulé).

### 3.4 Ce que ça coûte de corriger

Passer du niveau 1 au niveau 2 est **du post-traitement pur, sans toucher au solveur** :
`solve_with_scores` rend déjà les points cartes par carte et par monde ; il suffit de
convertir chaque monde en écart de score de donne via le contrat, puis de moyenner.
Le monde connaît les quatre mains, donc la belote et le capot sont calculables. **Fait.**

Passer au niveau 3 demande en plus une **table d'équité de match** — exactement l'objet
que le backgammon appelle *match equity table*. Là c'est un chantier, et c'est ce qui
manque encore : à 1900-200, +260 et +180 valent pareil, et `DealScore` ne le voit pas.

⚠️ Toujours non mesuré : **combien le niveau 2 vaut en force de jeu**. L'analyse dit que
l'objectif du niveau 1 est faux, pas de combien il coûte. Le seul chiffre disponible reste
les 40 donnes de R2 ci-dessous, qui ne tranchent rien. La bascule a été faite sur
l'argument de barème, pas sur une mesure de force — c'est un choix assumé.

---

## 4. Ce que font les autres jeux

### Backgammon — l'analogie la plus proche, et de loin

Même structure : deux camps, énormément de chance par partie, et des matchs en N points.
Le backgammon a formalisé exactement notre problème sous le nom **money game vs match
play** :

- en *money game* on maximise l'espérance de points — c'est notre niveau 1/2 ;
- en *match play* on maximise `P(gagner le match)`, lue dans une **match equity table**.
  Les décisions correctes **changent** selon le score : on prend des risques qu'on ne
  prendrait jamais en money, et inversement. Gnu Backgammon embarque les deux évaluations
  et ne les mélange jamais.
- La **règle de Crawford** existe précisément parce que l'équité devient dégénérée près du
  but.
- Côté classement, **FIBS** fait deux choses qu'on devrait copier : on note le **match**,
  pas la partie, et **K est mis à l'échelle par √(longueur du match)** — plus le match est
  long, moins il est bruité, plus il pèse. Plus un K plus élevé pour les joueurs peu
  expérimentés (< 400 parties).

**À prendre :** la distinction money/match comme cadre, le K ∝ √longueur, l'idée que la
table d'équité est un objet de première classe.

### Bridge — la leçon sur le bruit

- **Duplicate** : tout le monde joue les mêmes donnes, on classe sur la performance
  relative. C'est ce qui permet de classer en une soirée ce qui prendrait des milliers de
  donnes autrement. **Non transposable en ligne** contre des bots, mais l'idée l'est :
  comparer à une référence sur *la même* donne (voir §5).
- **Butler / datum** : on ne note pas le résultat, on note l'**écart au par** de la donne.
- **IMP vs matchpoints** : la même donne demande des coups différents selon le barème — les
  *safety plays* sont corrects en IMPs et faux en matchpoints. Le bridge organise des
  compétitions séparées pour les deux, et personne ne trouve ça étrange. C'est exactement
  l'argument pour séparer « donne isolée » de « partie ».

**À prendre :** l'écart au par comme métrique, et le fait que deux barèmes = deux jeux.

### Échecs — Glicko-2 et les cadences

- **Glicko-2** (Lichess, Chess.com) ajoute une **incertitude explicite** (RD) et une
  volatilité. C'est la bonne réponse à « bouger beaucoup au début puis de moins en moins »,
  et le RD *remonte* après une inactivité — un K décroissant seul ne sait pas faire ça.
- **Classements séparés par cadence** : c'est le précédent qu'on invoque pour deux Elo.
  Attention cependant, **ce n'est pas le même argument** : aux échecs le *jeu* est
  identique, seule l'horloge change. Ici c'est la **fonction d'objectif** qui change, donc
  la politique optimale change. L'analogie juste est bridge/backgammon, pas blitz/classique
  — et elle est *plus forte*, pas moins.

### TrueSkill / TrueSkill 2 (Xbox) — le seul conçu pour les équipes

μ/σ bayésien, **nativement multijoueur et par équipes**. C'est le meilleur ajustement
structurel à du 2v2 à partenaires tournants, et TS2 sait en plus attribuer un crédit
individuel à partir de statistiques par joueur. C'est ce qu'il faudrait si le salon
décollait.

### LoL / Riot

MMR caché + confiance, matchs de placement à fort mouvement, et **rang affiché découplé du
MMR**. À prendre : ne pas afficher le nombre brut au dixième près.

### Ce qui ne s'applique pas

- **Le duplicate au sens strict** : impossible en solo contre des bots (le joueur verrait
  la donne deux fois).
- **Les points de maître du bridge** : c'est de l'accumulation, pas du niveau.
- **Un Elo à somme nulle strict** : avec un bot sur trois sièges, la conservation n'a pas
  de sens ; l'ancrage la remplace avantageusement.

---

## 5. Recommandations, par ordre de rapport valeur / effort

### R1 — Ancrer les bots (petit, débloque tout le reste)

`K_BOT = 0`, Dédé figé à 1000 par définition, DouDou à `1000 − écart mesuré`. Les bots se
notent entre eux **à l'arène**, pas contre les joueurs. L'échelle devient absolue et le
défaut n° 4 disparaît.

**L'objection « ça casse la somme nulle » ne tient pas : le système n'est déjà pas à somme
nulle.** Avec `K_USER = 32` et `K_BOT = 8`, la somme des deltas d'une donne solo vaut
`32(s−e) + 8[(s−e) − 2(s−e)] = 24(s−e)`, soit ±24 points créés ou détruits à chaque donne.
La conservation a été abandonnée le jour où les deux K ont divergé. Et elle n'était de
toute façon pas l'objectif : elle sert à stabiliser la moyenne d'un pool où tout le monde
joue contre tout le monde, alors qu'ici une seule entité tient trois sièges sur quatre et
881 donnes sur 1073. Dédé n'est pas un joueur du pool, il **est** l'échelle — sauf qu'elle
bouge (1000 → 1044, pic 1119), si bien qu'un afflux de joueurs faibles dévalue en silence
tous les inscrits.

C'est de la pratique standard : les listes de moteurs d'échecs (CCRL, SSDF) sont ancrées
sur un moteur de référence à valeur fixe, exactement pour que l'échelle ne dérive pas
quand des moteurs plus forts entrent dans la liste.

**Le vrai coût, à traiter dès la mise en place** : l'ancrage déplace la dérive de « qui
joue » vers « quelle version du bot ». Il faut donc figer une référence **nommée et
versionnée** (`dede@2026-08 = 1000`), distincte du bot que les joueurs rencontrent, et
appliquer un décalage explicite et daté quand le bot évolue — une migration décidée, pas
un ajustement silencieux. Mesuré le 2026-08-03 : les réglages *internes* de Dédé ne
menacent pas l'ancre (64 mondes contre 256 = 50,0 % ± 4,3 sur 531 donnes) ; seuls de vrais
changements de modèle la déplaceront.

Détail qui compte : `K_BOT = 0`, pas « bot remis à 1000 » — il faut que le delta soit nul,
sinon il redérive entre deux remises.

Prérequis : un h2h Dédé ↔ DouDou **au niveau de la donne**. Le compteur existe depuis le
2026-08-03 (`MatchResult.deal_wins`, ligne `Par donne:` de `arena h2h`) ; il manque le
chiffre.

### R2 — Corriger l'objectif du jeu de la carte (niveau 1 → 2)

Convertir chaque monde en écart de **score de donne** avant d'agréger, au lieu de moyenner
des points cartes. Post-traitement pur, aucun changement du solveur. C'est ce qui fait
disparaître les coups « au hasard » après une chute acquise et qui rend les lignes de
sécurité en défense.

**Implémenté le 2026-08-03**, puis **passé en défaut le même jour** (`PlayObjective::DealScore`,
`[play] objective`). Le bras de contrôle est `web_dede_cardpts` ; `web_dede` reflète la prod.

**Aucune mesure de force ne l'appuie**, et c'est délibéré : 40 donnes en duplicate ont donné
6-6 sur les 12 donnes divergentes, marge +8,7 — rien de tranchable, alors que **95 % des
donnes se jouent différemment**. La bascule repose sur l'argument de barème (§3.1-3.2 bis),
qui ne dépend d'aucun échantillon : sous le seuil un point carte vaut *exactement* zéro.
Un h2h au volume reste utile pour chiffrer le gain, pas pour décider du défaut.

⚠️ `DealScore` change **l'échelle de `card_scores`** (±500 signé au lieu de 0-252). Traité :
le blob de stats porte `score_scale`, et `watch.js` / `prob-jeu.js` affichent l'unité. Un
`card_scores` d'IS-DD et un d'Oracle **ne se soustraient pas**.

### R3 — Noter la marge plutôt que le binaire

`s = σ(écart de points marqués / échelle)` au lieu de 1/0. Gratuit, divise la variance par
~2. Défaut n° 2.

### R4 — Le par : la seule voie vers un classement fiable en quelques dizaines de donnes

Noter l'**écart au par** plutôt que le résultat : `analysis.py` calcule déjà le coût DD de
chaque carte, `agent_review.py` dit ce que Dédé / DouDou / l'Oracle auraient joué à chaque
décision. Une note « coût moyen par décision » a une variance d'un ou deux ordres de
grandeur sous un 1/0, et elle est *juste* — bien jouer des cartes pourries devient un
succès.

Dépend de R2 : le par doit être calculé dans le bon objectif, sinon on classe les joueurs
sur leur capacité à maximiser des points cartes qui ne comptent pas.

### R5 — Glicko-2 (ou une rampe de K), **après** R3/R4

Un K décroissant seul ne corrige rien : il fige plus proprement un bruit déjà accumulé.
L'ordre compte.

### R6 — Afficher l'incertitude

« 973 ± 58 » ou un palier, pas un nombre au dixième. Une série chaude cesse alors de
ressembler à une progression.

---

## 6. Un Elo ou deux ?

**Conceptuellement : deux.** Une donne isolée et une partie en 1000/2000 ne récompensent
pas la même chose — l'une demande de maximiser le score de la donne, l'autre de gérer une
équité de match (savoir quand prendre un risque à 900-950, quand se coucher à 200-950).
Bridge et backgammon considèrent ça comme deux disciplines, et ils ont raison : la
politique optimale diffère, donc le classement mesure autre chose.

**Pratiquement : un seul, pour l'instant.** On a 4 comptes, 1 073 donnes et 155 parties.
Le plancher de bruit est déjà de ±53 Elo ; couper l'échantillon en deux le dégrade encore.
Deux classements à ±150 Elo chacun n'informent personne.

**Donc :**

1. **Enregistrer le format sur chaque événement noté dès maintenant** (`matches.target` est
   déjà en base — il suffit de le porter dans `elo_history`). C'est gratuit et c'est ce qui
   rend la séparation possible plus tard sans rejouer l'histoire.
2. **Garder un seul classement**, mais **pondérer par la longueur** à la façon de FIBS :
   `K ∝ √(nombre de donnes de l'événement)`. Une partie en 2000 points pèse ~3× une donne
   isolée, ce qui est à peu près le rapport de leur contenu en information. Ça reconnaît la
   différence de fiabilité sans couper le pool.
3. **Rouvrir la question de la séparation vers ~5 000 donnes par format.** Le seuil n'est
   pas arbitraire : c'est là que l'IC descend sous ±20 Elo, donc là où deux classements
   distincts commenceraient à dire deux choses différentes plutôt que deux bruits.

Le format « une donne » restant le défaut du produit (`target = 0`), c'est lui qui
accumulera le volume le plus vite — la séparation, quand elle viendra, se fera sans doute
en détachant les parties longues d'un socle « donne » déjà bien peuplé.

---

## 7. Ce qui est mesuré et ce qui ne l'est pas

**Mesuré :** l'état de la base, les taux de victoire en solo et leurs IC, l'amplitude de
bruit par simulation, le fait qu'IS-DD agrège des points cartes, la forme exacte du barème.

**Non mesuré :**

- de combien R2 améliore la force de jeu (l'analyse dit que l'objectif est faux, pas de
  combien) ;
- l'écart réel Dédé ↔ DouDou au niveau de la donne — un smoke de 20 matchs donne 85 % / +922
  au niveau *match*, avec ±16 pp d'IC, et l'Elo du web les place à 55 points d'écart ;
- ce que vaut une pondération `√longueur` sur nos volumes ;
- si le plateau de séparation est bien vers 5 000 donnes (c'est une extrapolation de la
  courbe d'IC, pas une mesure) ;
- **le plateau de mondes d'IS-DD.** Le commit `bdc6611` annonce « vers 60, pas 240 » ;
  avec les barres d'erreur, les intervalles se recouvrent tous à l'entame (0,066 ± 0,072
  à 60 contre 0,096 ± 0,177 à 240, n = 33), et un det-sweep antérieur mesuré en *force
  de jeu* pointe dans l'autre sens (+29 à 60 contre +227 à 240). Les deux mesures se
  contredisent et aucune n'est solide. Détail et réserve :
  `scripts/analysis/isdd_dets_by_stage.py`.

---

---

## 8. Journal des mesures

### 2026-08-03 — `web_dede_w64` contre `web_dede` (64 mondes contre 256)

```
  RESULT: 50.0% vs 50.0%   ·   Wins 25 — 25   ·   marge +14 pour w64
  Dir 1 (w64=NS): 14-11    ·   Dir 2 (web_dede=NS): 14-11
  Par donne: 265 — 265 (0 nulles) → 50,0 % ± 4,3 (IC95)   ·   531 donnes
```

Les deux directions donnent 14-11 pour le camp Nord-Sud : l'avantage de siège s'annule
exactement, le duplicate matching a fait son travail.

**Ça réfute le det-sweep du 2026-07-24**, qui donnait +29 de marge à 60 mondes contre +227
à 240 : un écart pareil serait trivialement visible sur 531 donnes. Réconciliation
plausible plutôt que « il s'est trompé » : ce sweep **précède le passage aux mondes
playgen**. Avec un échantillonneur médiocre il faut beaucoup de tirages pour approcher la
bonne postérieure ; avec celle de playgen il en faut moins. Les deux peuvent être justes
dans leur contexte.

**Réserve importante — la mesure est plus faible qu'elle n'en a l'air.** Mondes réellement
résolus par recherche, moyennés sur les deux bots :

| cartes | 8 | 7 | 6 | 5 | 4 | 3 | 2 |
|---|---|---|---|---|---|---|---|
| résolus/recherche | 114 | 125 | 126 | 114 | 73 | 37 | 34 |

À 4 threads la latence d'un aller-retour monte à 613-667 ms (contre 164-227 ms en solo),
donc dès 4 cartes restantes le budget par coup est **plus court qu'un seul aller-retour** :
les deux bots tombent sous leurs plafonds respectifs et **deviennent le même bot**. La
comparaison ne distingue réellement qu'entre 8 et 5 cartes, où `web_dede` tournait à
~165-190 mondes contre 64. **La conclusion porte sur les quatre premiers plis, pas sur la
donne entière.**

Contexte : arbre `HEAD b40601c` + diff non commité `79867c3139ae285a` (la contrainte dure
belote, en cours par ailleurs). Lancé avec `--no-save` pour cette raison.

### 2026-08-03 — `web_dede` contre `web_doudou` : l'ancre

```
  RESULT: web_dede 72.0% vs web_doudou 28.0%   ·   Wins 36 — 14   ·   marge +615
  Dir 1 (dede=NS): 18-7    ·    Dir 2 (doudou=NS): 7-18
  Par donne: 284 — 229 (0 nulles) → 55,4 % ± 4,3 (IC95)   ·   514 donnes
```

→ **+37 Elo, IC95 [+7, +68]**, d'où `BOT_ELO = {dede: 1000, doudou: 963}`.

**Le même écart vaut +164 Elo au niveau match** (72 %), parce qu'un match agrège ~10
donnes et amplifie le même avantage par coup. C'est 37 qu'il faut : `elo.py` note à la
donne. Le chiffre qu'on retient dépend entièrement de ce qu'on appelle « une partie ».

Précision modeste (±30), mais sans commune mesure avec ce qu'on avait : DouDou était à
988,6 sur **11 donnes**. À resserrer si besoin — ±15 Elo demanderait ~2 000 donnes.

Arbre propre, `HEAD 158e4b4`, contrainte belote active. Ligne sauvegardée dans
`matches.csv`.

### 2026-08-03 — le repli belief-net n'est plus à zéro

```
  100 % playgen 97,4 %  ·  partielles 2,5 %  ·  sans playgen 0,1 %
  repli 0,42 %  (contre 0,00 % le matin)   ·   remplissage 92-96 % (contre 97-99 %)
```

Effet prédit de la contrainte belote : `retain_valid` rejette des mondes playgen devenus
invalides, le remplissage baisse, la file sèche parfois. 0,42 % reste marginal mais **ce
n'est plus zéro**, donc la décision « supprimer le belief net » n'est plus évidente. À
re-mesurer une fois la contrainte belote commitée. (Réserve : la contention à 4 threads
peut expliquer une part de la baisse ; les deux causes sont confondues dans ce run.)

---

## Voir aussi

- [agents.md](agents.md) — comment un bot est assemblé, format des specs
- [arena_results.md](arena_results.md) — le classement des bots
- [play/is_dd.md](play/is_dd.md) — l'agrégation IS-DD dont §3.2 parle
- `python/colver/web/elo.py` — l'implémentation actuelle
- `colver-core/src/engine/scoring.rs` — le barème dont §3.1 tire la marche
