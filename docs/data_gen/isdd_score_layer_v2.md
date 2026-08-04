# Regénérer la couche de scores IS-DD

*Ouvert le 2026-08-04. Plan d'exécution : ce qui est décidé, ce qui reste à mesurer,
et les pièges qui produiraient des chiffres plausibles et faux.*

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

| | contenu | format |
|---|---|---|
| *(rien à générer)* | les **500 k premières donnes de `base_5M.bin`** | `COLVDD01` |
| `tail_100k.bin` | strates **construites** via `hand_from_class_id` | `COLVDD01` |
| `scores_isdd_v2.sc` | points cartes N-S par atout, IS-DD fort | `COLVSC01`, `[u8;4]` |

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

### Ce que la mesure A change au plan

Elle **ne dit pas** que l'étiquette est fausse — seulement que le préfixe est hors
distribution. Est-ce que ça déplace les points cartes ? C'est la **mesure B**, et elle
reste à faire : au milieu d'une donne le préfixe porte aussi les cartes jouées, bien plus
informatives que 8 jetons d'enchère.

Ce qu'elle change, c'est la **liste des variantes que B doit départager**. « Minimale »
et « plausible » étaient deux devinettes ; il y a maintenant cinq chiffres à viser, et
un candidat qui les atteint tous sans être réglé à la main :

> **l'enchère de v6 elle-même, masquée sur la couleur cible** — à chaque tour, le masque
> légal est réduit à `PASS`, `COINCHE`, `SURCOINCHE` et aux paliers de l'atout `t`, et
> v6 choisit dedans. La contestation, la coinche, la longueur et la valeur sortent de la
> politique au lieu d'être fabriquées, et l'atout reste celui qu'on veut étiqueter.

Le coût est négligeable : ~8 passes avant de 117→512³→43 par enchère, quatre enchères par
donne, soit **~1,6 ms** contre les ~1,5 s que coûtent les quatre labellisations IS-DD de
la même donne (§8) — **de l'ordre de 0,1 %**. Ce qui resterait décalé, et qu'il faudra
mesurer et non supposer : forcer les quatre sièges sur une seule couleur retire aux
adversaires la possibilité de contester *dans la leur*, donc le taux de contestation
retombera quelque part entre 0 et 81 %.

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
quatre mains. La quantité qui dimensionne la strate est `P(capot | mes 8 cartes)`, et
elle n'est pas mesurée.

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

Base : `gen_games_isdd` fait **2,62 donnes/s** sur une 3090 à 40 mondes/décision
([isdd_games.md](isdd_games.md)), soit ~32 recherches IS-DD par donne. 600 k donnes ×
4 atouts = 2,4 M déroulements.

| | 1 GPU | 2 GPU |
|---|--:|--:|
| à plat | 254 h | 127 h |
| + `dets_schedule` décroissant (**1,24×** mesuré) | 205 h | 102 h |
| + budget gradué (§5, −31 %) | **141 h** | **~70 h ≈ 3 jours** |

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

## 10. Ce qui n'est pas décidé — trois mesures d'abord

| | quoi | coût | ce qu'elle décide |
|---|---|---|---|
| ~~**A**~~ | ✅ **faite le 2026-08-04** (§4) : oui, le préfixe est hors distribution, sur la **forme** (une seule annonce contre 11,9 % de réel, 0 % de contestation contre 81,3 %, 0 % de coinche contre 25,7 %) et non sur l'identité du preneur (camp bon à 89,4 %) | 4 min, 0 GPU | a remplacé les variantes devinées de B par **v6 masqué sur la couleur**, et retiré l'échelle valeur ↔ force, non soutenue |
| **B** | variantes d'enchère (minimale / v6 masqué sur la couleur) **et camp du preneur**, écart apparié en points cartes | ~2 h GPU | la variante à retenir, et `[u8;4]` contre `[u8;8]` (§4) |
| **C** | `P(capot \| mes 8 cartes)` sur les 913 476 cases à `dd_pts == 252` | quelques h GPU | le contenu de `tail_100k` (§6) et le plancher d'exploration conditionnée |

**A ne dispense pas de B, elle la recadre.** Un préfixe hors distribution n'est un défaut
que s'il déplace l'étiquette ; au milieu d'une donne, playgen lit aussi les cartes déjà
jouées, bien plus informatives que 8 jetons d'enchère. B mesure ce déplacement contre le
plancher de bruit ci-dessous — et si l'écart est en dessous, la construction bon marché
suffit et tout le §4 devient une note de bas de page.

Référence de bruit pour A et B : les **44,7 pts** de dispersion intra-main
([bid_v7_plan §2.8](../bid/bid_v7_plan.md)). Un écart nettement en dessous ⇒ prendre la
variante la moins chère et passer à la suite.

## 11. Le risque principal, nommé

On échange **5 M × 4 étiquettes faibles** contre **600 k × 4 fortes** : 8× moins de
donnes.

[bid_v7_plan §2.8](../bid/bid_v7_plan.md) avait tranché « les donnes gagnent » — mais
cet arbitrage repose explicitement sur *« un bruit non biaisé ne biaise pas un
ajustement aux moindres carrés »*, mesuré en mondes uniformes. **Changer la source de
mondes change ce que vaut un monde, et l'argument ne se transporte pas tel quel.**

Deuxième effet : 500 k donnes = 2 M mains, contre les 6,2 M de tirages que le coupon
collector demande pour voir les 472 579 classes ([bid_v7_plan §1.4](../bid/bid_v7_plan.md)).
La couverture de classes n'est plus acquise — c'est précisément la raison d'être de
`tail_100k`, et non un effet secondaire à absorber.

**Mitigation intégrée** : les couches composent par `offset`/`count`, et les 500 k
donnes sont **les mêmes** que celles de l'ancienne couche. L'A/B ancienne étiquette
contre nouvelle est donc direct, à donnes identiques — c'est la mesure qui dira si
l'échange valait le coup.

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
