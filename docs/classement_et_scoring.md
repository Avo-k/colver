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

**Et il y a plus direct que les barres d'erreur.** L'échelle mesurée en §8 dit que *toute*
l'étendue du jeu de la carte — du joueur à règles écrit à la main jusqu'à l'omniscience
double-mort — vaut **171 Elo** :

| barreau | Elo vs DouDou50 |
|---|---|
| Règles (`method = "rule"`) | −59 |
| Heuristique | −41 |
| DouDou35 | −29 |
| **DouDou50** | **0** |
| Oracle DD (jeu parfait) | +112 |

Les 305 points affichés entre deux humains valent donc **1,8 fois l'étendue entière du
jeu**. Il n'y a pas besoin de simuler quoi que ce soit pour conclure : l'écart est plus
large que ce que le jeu permet.

⚠️ **Mesuré à enchère figée** (v6 aux quatre sièges), donc ces 171 Elo décrivent la carte
seule. Un humain qui annonce mal perd davantage, et l'étendue humaine réelle est plus
grande — mais elle l'est *par l'enchère*, pas par la carte.

---

## 2. Les six défauts du système actuel

1. **K = 32 sur une donne.** Calibré pour une partie d'échecs, où le résultat est ~du
   signal. Une donne de contrée est ~de la distribution.
2. **Score binaire.** Gagner de 10 points ou de 400 vaut pareil ; la marge est jetée.
3. **Un siège sur quatre.** L'écart individuel est dilué de moitié par le partenaire, donc
   le bruit double quand on le reconvertit en Elo individuel. **Et la dilution n'est pas
   exactement un facteur 2** : la contribution d'un siège dépend de la force de son
   partenaire, ce qu'un modèle « équipe = moyenne des deux » ne peut pas représenter.
   L'ampleur dépend de l'écart de niveau — ×1,70 entre DouDou et le jeu parfait, mais
   seulement ×1,21 entre deux joueurs proches (§8). À l'échelle où vivent deux humains,
   l'additivité est donc une approximation à ~20 %, pas une erreur de principe.
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

**La bascule repose sur l'argument de barème (§3.1-3.2 bis), pas sur une mesure de force**,
et deux mesures successives n'ont rien trouvé. L'argument, lui, ne dépend d'aucun
échantillon : sous le seuil un point carte vaut *exactement* zéro.

**Mesure 1 — 40 donnes en duplicate** : 6-6 sur les 12 donnes divergentes, marge +8,7,
alors que **95 % des donnes se jouent différemment**. Rien de tranchable.

**Mesure 2 — h2h 300 matchs** (`web_dede` vs `web_dede_cardpts`, 150/direction, 2026-08-03,
sidecar sur le GPU de moxxi, 3 h 10) :

| niveau | deal_score | card_points | IC95 |
|---|---|---|---|
| matchs (n = 300) | 49,7 % (149) | 50,3 % (151) | ± 5,7 pp |
| **donnes (n = 3 123)** | **50,8 %** (1 585) | 49,2 % (1 538) | **± 1,8 pp** → [49,0 ; 52,6] |
| marge moyenne | +20 pts/match | — | +3 Elo/donne (échelle 316) |

Les trois indicateurs sont cohérents avec zéro. Le niveau donne est le plus puissant et
penche de +0,8 pp pour le nouvel objectif ; son intervalle contient 50. **Pas de
différence mesurable, et pas de régression non plus.**

⚠️ **Le run a tourné dégradé, et le biais n'est pas neutre.** Compteurs de la course :
**~1,3 s par aller-retour au sidecar** — plus que le budget de 1 000 ms par coup — et IS-DD
**90 à 100 % du temps en attente**, d'où **~36 mondes résolus par décision** (15,7 M reçus,
13,4 M jetés sans être résolus, soit 85,7 %). La prod en autorise 256 et le plateau mesuré
est à ~60. Causes non départagées depuis ce run : latence du tunnel SSH vers moxxi, et file
d'attente d'un seul sidecar servant 8 chercheurs concurrents — la seconde est probablement
dominante, un RTT de LAN valant ~1 ms et non 1 300.

L'A/B reste **valide** : les deux bras ont subi la même dégradation, la comparaison est
symétrique. Mais elle porte sur l'objectif *à 36 mondes par décision*. Et la direction du
biais est défavorable au nouvel objectif : il existe pour lire un seuil dans la
distribution des mondes, or moins de mondes = estimation plus bruitée de `P(au-dessus du
seuil)` = marche plus floue. **Ce run sous-estime donc plausiblement l'effet**, sans que ce
soit démontré.

**Ligne close pour l'arène.** Deux mesures, deux nuls, et le plancher de bruit d'un h2h
(±1,8 pp au mieux, au niveau donne) est du même ordre que l'effet plausible. C'est le même
mur que pour la belote, où la réponse est venue d'une mesure **appariée à la décision**
(`bench_belote_ab`) et non d'un h2h. Un équivalent pour l'objectif comparerait les deux
cartes choisies position par position sur les mêmes mondes, avec un juge à fort nombre de
mondes. **Non fait, et à ne pas retenter par l'arène.**

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

#### Ce que le par ne peut **pas** être : l'Oracle

Une version de R4 consistait à encadrer le joueur entre deux repères — ce qu'aurait fait
DouDou, ce qu'aurait fait l'Oracle — et à noter sa position entre les deux, éventuellement
en divisant par l'écart. **Le plan factoriel de la §8 la réfute par la mesure**, et pas
au titre d'une réserve théorique :

- à partenaire et adversaires figés, remplacer un siège par l'Oracle **fait moins bien
  13,1 % du temps** et ne change rien 26,1 % du temps. Le jeu parfait n'améliore
  réellement que **61 % des échanges** ;
- une donne sur six (17,3 %) donne le **même score dans les 16 configurations** : le
  dénominateur « écart entre les deux repères » y est nul.

Et ce n'est **pas** une particularité du double-mort. Le même plan appliqué à deux joueurs
proches (DouDou35 → DouDou50) donne **31,4 %** d'échanges où le meilleur des deux fait
moins bien. Plus l'écart de niveau est petit, plus le meilleur joueur perd d'échanges
individuels — un repère, quel qu'il soit, n'est donc pas un plafond. Dans le cas de l'Oracle
s'y ajoute une cause propre : le DD suppose que **la défense joue parfaitement aussi**, donc
le vrai plafond (meilleure réponse à l'adversaire réel) est *au-dessus* de lui.

Ce qui reste utilisable de l'idée : `|d_oracle − d_bot|` comme **poids** (une donne sans
marge de manœuvre ne doit pas peser autant qu'une donne à 800 points d'amplitude), jamais
comme diviseur ni comme borne.

#### Et ce que le par devra affronter : la contribution n'est pas additive

Toujours §8 : l'effet d'un siège vaut **+50,7** quand son partenaire est faible et
**+86,3** quand il est fort (±0,9). Un bon partenaire *amplifie* au lieu de masquer, et ce
n'est **pas** un artefact de la marche du barème — en points cartes purs, sans aucun seuil,
l'amplification monte à ×1,97. Les adversaires, eux, ne changent rien.

C'est le défaut n° 3 sous sa forme forte : « équipe = moyenne des deux partenaires » est
additif par construction, alors que la contribution d'un joueur dépend de qui est à côté de
lui. Les deux estimations Elo du même écart divergent en conséquence : **+112** en changeant
les deux partenaires, **+87** en n'en changeant qu'un (×2 pour la dilution).

**Mais l'effet est proportionnel à l'écart de niveau** : entre deux joueurs proches il
tombe à ×1,21, et les deux estimations Elo à +29 contre +26. Ce n'est donc pas une raison
d'abandonner Elo — c'est une raison de ne pas lui demander mieux que ~20 % de justesse sur
la décomposition d'une équipe. Un par calculé **par décision** échappe entièrement au
problème, puisqu'il n'attribue jamais à un joueur un résultat produit à quatre.

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
bruit par simulation, le fait qu'IS-DD agrège des points cartes, la forme exacte du barème,
et — depuis le plan factoriel de §8 — **l'influence d'un siège sur une donne** : son effet
par rôle, sa dépendance au partenaire, la fraction de donnes sans aucune prise, l'écart Elo
total entre DouDou50 et le jeu parfait (87-112), et le nombre de donnes qu'il faudrait pour
séparer deux joueurs proches (876).

**Non mesuré :**

- de combien R2 améliore la force de jeu (l'analyse dit que l'objectif est faux, pas de
  combien) ;
- l'écart réel Dédé ↔ DouDou au niveau de la donne — un smoke de 20 matchs donne 85 % / +922
  au niveau *match*, avec ±16 pp d'IC, et l'Elo du web les place à 55 points d'écart.
  **C'est devenu la mesure la plus urgente** : §8 borne à ~110 Elo tout l'espace entre
  DouDou50 et l'omniscience, donc l'ancre à 50 points donne à Dédé près de la moitié du
  chemin, ce qui demande vérification ;
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

### 2026-08-03 — Plan factoriel 2⁴ : combien un seul siège peut-il déplacer une donne ?

`scripts/analysis/seat_influence.py`. Sur une donne dont **l'enchère est figée** (bidder v6
aux quatre sièges, actions rejouées à l'identique), le jeu de la carte est rejoué avec
chaque siège tenu soit par un occupant **A**, soit par un occupant **B** — **16
configurations**, 3 995 donnes, 63 920 rejeux par run, ~7 min sur 8 cœurs, sans GPU. Les
deux occupants étant déterministes, les 16 résultats sont exacts : aucun bruit
d'échantillonnage à l'intérieur d'une donne, et l'effet d'un siège se lit en différences
appariées.

**Contrôle porté par le run** : quand B est l'Oracle, les quatre Oracles doivent réaliser
exactement la valeur DD de la position d'entame — **160/160** à chaque fois.

⚠️ **Les trois premiers runs ont été jetés.** Ils tournaient avec un
`python/colver/_colver.abi3.so` compilé à 09:26, une heure avant `1edd349` qui déplaçait le
départage des ex æquo **dans `solver.rs`**. L'Oracle mesuré n'était donc pas celui de la
production, et le `--tiebreak` de la ligne de commande n'atteignait même pas le parseur.
Ce qui l'a révélé : deux bras d'A/B rendant des chiffres **identiques au bit près** —
ce n'est pas un résultat nul, c'est une panne. Le script refuse maintenant de tourner si un
`.rs` du cœur est plus récent que le `.so`, et hache le `.so` dans la provenance.

#### Deux régimes, et ils ne disent pas la même chose

| grandeur | **DouDou50 → Oracle**<br>(l'enveloppe) | **DouDou35 → DouDou50**<br>(régime réaliste) |
|---|---|---|
| effet d'un siège — preneur | **+82,2 ± 2,4** | **+21,4 ± 2,4** |
| — partenaire du preneur | +78,7 ± 2,4 | +25,9 ± 2,4 |
| — défenseur | +56,5 ± 1,4 | +10,8 ± 1,4 |
| donnes où les 16 configurations donnent le même score | 17,3 % | 16,7 % |
| marge de manœuvre (max − min) | méd. 396, p90 804 | méd. 416, p90 804 |
| échanges où le **fort** fait *moins bien* que le faible | **13,1 %** (+26,1 % sans effet) | **31,4 %** (+24,2 %) |
| effet selon le partenaire (faible → fort) | +50,7 → +86,3 (**×1,70**) | +15,6 → +18,8 (**×1,21**) |
| … la même chose en points cartes purs | ×1,97 | ×1,15 |
| effet selon les adversaires | plat | plat |
| écart Elo (2 sièges / 1 siège ×2) | **+112 / +87** | **+29 / +26** |
| donnes pour les séparer, non apparié | 59 | **876** |

**Cinq lectures.**

1. **Une donne sur six n'offre aucune prise** (17,3 % / 16,7 %), quel que soit le régime.
   Tout classement qui pèse les donnes également gaspille ce sixième.
2. **Aucun joueur de référence n'est un plafond**, et ce n'est pas une particularité du
   double-mort. L'Oracle fait moins bien que DouDou50 dans 13,1 % des échanges appariés —
   mais **DouDou50 fait moins bien que DouDou35 dans 31,4 %**. Plus l'écart de niveau est
   petit, plus le meilleur joueur perd d'échanges individuels. C'est ce qui ferme
   définitivement la version « bornes » de R4 : le dénominateur est nul une fois sur six et
   du mauvais signe une fois sur trois.
3. **La non-additivité est réelle mais elle est proportionnelle à l'écart de niveau.**
   ×1,70 sur l'enveloppe, seulement ×1,21 sur le régime réaliste. À l'échelle où vivent
   deux humains, le modèle additif d'Elo est donc *approximativement* correct — la
   correction à lui apporter est de l'ordre de 20 %, pas de 70 %. Même signature dans les
   deux traductions Elo : +112 contre +87 sur l'enveloppe (écart de 29 %), +29 contre +26
   sur le régime réaliste (12 %).
4. **La marche du barème n'explique pas la complémentarité — c'était l'hypothèse, elle est
   fausse.** Relus en points cartes purs, sans aucun seuil de contrat, les mêmes échanges
   donnent ×1,97 et ×1,15, donc *davantage* dans les deux cas. C'est une propriété du jeu
   de la carte : un partenaire qui joue bien donne davantage sur quoi jouer.
5. **Le plafond de vitesse d'un classement, et c'est le chiffre à retenir.** Séparer deux
   joueurs distants de ~28 Elo demande **876 donnes** sans appariement. La base de prod en
   compte 1 073 **au total, tous comptes confondus**. Le tout premier chiffre de ce
   document — « le mouvement observé est indiscernable du bruit » — n'est donc pas un défaut
   de réglage : c'est la physique du jeu.

**L'écart total DouDou50 → jeu parfait vaut +87 à +112 Elo.** C'est tout l'espace
disponible au-dessus de DouDou, et il met l'ancre `doudou = 950` sous tension : 50 points
donnent à Dédé la moitié du chemin vers l'omniscience. Le vrai plafond est un peu plus haut
que +112 (l'Oracle n'est pas la meilleure réponse à un adversaire imparfait), mais le h2h
direct `web_dede` / `web_doudou` relu à la marge reste la mesure à faire avant de retoucher
ces valeurs.

**Le départage des ex æquo compte, et `1edd349` était un vrai gain.** Le même plan avec
`--tiebreak dearest` : effet preneur +73,3 au lieu de +82,2, « fait moins bien » 17,0 % au
lieu de 13,1 %, Elo +94/+74 au lieu de +112/+87. Choisir la carte la moins chère parmi les
DD-équivalentes vaut donc ~18 Elo à l'Oracle contre un adversaire imparfait — invisible en
double-mort par construction, puisque les deux réalisent la même valeur DD.

#### L'échelle complète : cinq barreaux, un ordre connu

Comparer deux joueurs *proches* est le pire banc d'essai pour valider une métrique de
classement : on ne dispose d'aucun ordre de référence, puisque l'ordre est précisément ce
qu'on cherche à établir. D'où une échelle à cinq barreaux, tous mesurés contre la **même
référence** (DouDou50) pour éviter les erreurs de chaînage, tous en CPU pur.

| barreau | Elo vs DouDou50 | effet d'un siège / donne | donnes pour le séparer de DD50 |
|---|---|---|---|
| Règles | **−59** | +31,0 ± 1,2 | 146 |
| Heuristique | **−41** | +21,9 ± 1,2 | 217 |
| DouDou35 | **−29** | +17,2 ± 1,1 | **876** |
| DouDou50 | 0 | — | — |
| Oracle DD | **+112** | +68,5 ± 1,3 | 59 |

L'ordre est monotone et sans ambiguïté, ce qui en fait le banc d'essai qu'il faut pour
valider R4 : quatre écarts de tailles très différentes (17, 12, 29, 112 Elo), dont on
connaît le classement d'avance.

**Deux surprises.** L'heuristique écrite à la main n'est pas « nulle » — 41 Elo sous
DouDou50, moins que ce que beaucoup supposeraient. Et **l'étendue totale ne fait que
171 Elo**, ce qui est le chiffre le plus utile de tout ce document : il borne par le haut
ce qu'un classement de jeu de la carte peut légitimement afficher (cf. §1).

L'explication est cohérente avec le reste des mesures : 40 % des décisions sont **forcées**,
15 à 19 % des donnes sont **figées** quel que soit l'occupant, et l'enchère est la même
pour tout le monde. Il reste peu de place pour se distinguer.

#### « DouDou50 est meilleur que DouDou35 » — d'où on le sait

Ce n'est **pas** une hypothèse posée en nommant A et B : le plan mesure le signe, et il
sort positif. Statistique au niveau de la donne (une valeur par donne, donc pas de sièges
comptés comme indépendants) : **+17,2 ± 1,10 points, IC 95 % [+15,1 ; +19,4], z = 15,6**.

Mais **le taux de victoire, lui, ne dit presque rien** : DouDou50 ne l'emporte que sur
**52,4 %** des donnes (z = 3,0), contre 78,0 % pour l'Oracle face à DouDou50 (z = 35,5).
C'est exactement l'argument de R3 sous une autre forme — à ce niveau d'écart, la marge
porte le signal et le signe est quasi aveugle.

Deux vérifications, parce que la configuration pouvait tout expliquer :

- **La dimension d'observation est la bonne.** 10 260 612 − 10 244 228 = 16 384 octets
  = 4 096 flottants = (415 − 411) × 1024. L'auto-détection donne donc 415 (héritée) pour
  `dmc_35.bin` et 411 (canonique) pour `dmc_50.bin`, comme attendu.
- **Le passage résiduel est correctement désactivé.** C'est le seul réglage libre du
  couple et il est *indétectable depuis le fichier de poids*. Contrôle mesuré sur
  1 999 donnes : forcer `residual = true` sur `dmc_35.bin` coûte **−63,5 points par siège
  et −114 Elo**. Le réglage utilisé n'est donc pas celui qui bride le modèle.

**Il n'existe aucune corroboration externe** : `matches.csv` ne contient aucun h2h
`dmc_35` contre `dmc_50`, et ses 14 lignes DouDou35 datent toutes de mars, donc d'avant
les deux ruptures de barème et la correction du solveur. Ce plan factoriel *est* la mesure.

**Confusion résiduelle, non mesurée** : l'enchère est celle de v6 aux quatre sièges, alors
que DouDou35 a été entraîné à côté de bid v2 et DouDou50 en triforge avec des bidders plus
récents. Une part des +29 Elo peut donc être « mieux accordé à la distribution de contrats
de v6 » plutôt que « joue mieux la carte ». Ça ne menace pas le signe, ça borne
l'interprétation.

**Limite du plan** : il mesure l'influence d'un siège *entre deux occupants donnés*. Un
humain n'est aucun des trois. Les deux régimes bornent la question par en haut et par le
milieu ; ce n'est pas une note.

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
