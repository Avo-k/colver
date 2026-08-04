# Un corpus de jeu fort pour playgen v3

*État au 2026-08-04. Rien de ce qui suit n'est lancé.*

**L'idée** : `gen_games_isdd` sait produire des donnes entières jouées par le
joueur de référence du projet (bid v6 + IS-DD sur mondes playgen). Le corpus
playgen actuel, lui, a été joué par DouDou50 et **entièrement à 0-0**. D'où deux
envies qui vont ensemble : un corpus de *jeu fort*, et un corpus de *parties* en
2000 points plutôt que de donnes isolées.

Tout est chiffré depuis la nuit du 2026-08-04. Ce qui manque n'est pas le
raisonnement, c'est trois mesures et une journée de format.

---

## 1. Le chiffre qui cadre tout : IS-DD coûte 40× DouDou50

| générateur | débit | source |
|---|---|---|
| `generate_game_data` (DouDou50) | ~117 donnes/s | horodatages des huit `playgen_moxxi_1M_s10*` — 2 h 23 par million |
| `gen_games_isdd` (bid v6 + IS-DD) | **2,89 donnes/s** | run réel de 38 000 donnes, 13 138 s ([isdd_games.md](../data_gen/isdd_games.md)) |

Refaire les 9M donnes du corpus actuel en IS-DD, c'est **36 jours** sur une
carte. Le corpus IS-DD ne sera donc pas de la même taille, et **c'est ça la vraie
question** — pas le débit.

### Ce que ça donne en temps

Une 3090, `--dets-schedule 40,40,40,30,20,15,15` (le calendrier décroissant,
1,24× — extrapolé du banc flat-40 vers le débit soutenu, donc ±10 %) :

| donnes | ≈ parties (8,8 donnes/partie) | 1× 3090 | 3090 + 4090 |
|---|---|---|---|
| 100 k | 11 k | **7,7 h** | ~3,2 h |
| 300 k | 34 k | 23 h | ~10 h |
| 500 k | 57 k | 39 h | ~16 h |
| 1 M | 114 k | 3,2 j | ~1,3 j |

- **8,8 donnes par partie** vient de `calibrate_winprob`
  ([bid_v4_score_aware.md](../bid/strategies/bid_v4_score_aware.md)), pas d'une
  estimation.
- La colonne à deux GPU **suppose** la 4090 à ~1,4× la 3090 pour cette charge.
  Jamais mesuré : la carte était prise toutes les nuits.
- Le client tient largement — 1,96 s de solve DD par donne sur 32 cœurs, soit
  ~16 donnes/s de capacité, contre 8-9 demandées. **Le mur reste le GPU.**
- Le mode partie ne coûte **rien** en débit : mêmes donnes, seulement enchaînées.

### Où le lancer, et ce que ça coûte aux joueurs

Générer sur moxxi sature le GPU **par construction**, et c'est le GPU du sidecar
de prod. Mesuré le 2026-08-04 : un simple bras d'*entraînement* fait déjà tomber
Dédé de **108 à 91 mondes par coup**, en silence — la prod est en mode temps,
donc l'échéance ne bouge pas et rien ne signale la dégradation
([is_dd.md](../play/is_dd.md)). Une génération de 1 à 3 jours coûterait bien
plus. Le partage propre est : **génération sur la 4090, moxxi laissée à la
prod**.

---

## 2. Le mode partie existe déjà — c'est le format qui bloque

`gen_games_isdd --match-mode` enchaîne les donnes en parties de 2000 points,
donneur qui tourne, score cumulé passé à `MatchContext` donc lu par bid v6. Ce
n'est pourtant **pas le défaut**, et pas par prudence :

> `COLVGM01` ne transporte pas le score de partie.

Un playgen entraîné là-dessus verrait v6 annoncer 110 sur une main à 90 sans que
*rien dans son entrée* ne l'explique. **Ce n'est pas de l'information en plus,
c'est de l'entropie irréductible en plus** — le corpus serait strictement pire
que celui à 0-0.

### Ce que le score vaut, mesuré

Perplexité de la tête d'enchère de `playgen_v2_final` sur des corpus où v6
annonce à un score imposé, 20 000 donnes chacun, **mêmes donnes** (seules les
annonces changent), tour 1 seulement — le seul apparié :

| écart entre les camps | vs 0-0 |
|---|---|
| ≤ 600 (600-400, 1000-500, 1800-1200) | −0,011 à +0,006 — **bruit** (SE ≈ 0,008) |
| 1200 (1200-0, 300-1500, 1500-300, 1800-600) | **+0,074 à +0,121** |

À comparer aux **+0,0028** que coûte de diviser les paramètres par 3,3 sur cette
même tête : **29×**. C'est le seul levier restant, la tête d'enchère étant
saturée en capacité *et* en données (un modèle 3,3× plus petit, à 8 % du budget,
est à 1,6 % du meilleur).

**Mais le régime est étroit** : ce qui pénalise est l'**écart entre les camps**,
pas la fin de partie. 1800-1200 — un camp à 200 points de gagner — ne coûte
rien. Une partie serrée jusqu'au bout ne gagne rien à ce que playgen connaisse
le score.

### Le travail

1. **COLVGM02** : 2 champs de score par donne, 58 → 62 octets.
2. **Tokeniseur** : deux jetons d'en-tête bucketés après `P_OBSPOS`, **relatifs
   à l'observateur**.
3. **Le contrôle qui va avec, non négociable** : (1500,300) et (300,1500)
   doivent coïncider une fois moyennés sur les quatre sièges. C'est exactement
   le bug qui a mordu la sonde de perplexité — `write_bid_observation_score_aware_v3`
   prend `my_score, opp_score` relatifs à l'annonceur, et passer `(ns, ew)` brut
   fait croire aux quatre joueurs qu'ils sont dans le même camp.

Compter ~1 journée. Ça peut s'écrire pendant qu'autre chose tourne.

---

## 3. Non, ce corpus n'entraînera pas un meilleur bidder

C'est le point où le plan se scinde en deux, et il vaut mieux le savoir avant de
dépenser trois jours.

`train_bid_nn` mange un **pool** (COLVDD01 + couches COLVSC01) : des
*étiquettes* — ce que chaque atout vaut sur cette main. `gen_games_isdd` produit
des *trajectoires*. La seconde forme ne se convertit pas en la première : une
donne jouée réalise **un** contrat, celui que v6 a choisi, pas les quatre. Et
c'est un échantillon **biaisé par le choix de v6**, ce qui est le pire cas pour
une étiquette.

Ce qu'on tirerait des trajectoires est du **clonage de comportement de v6** — et
c'est déjà fait, déjà mesuré : la tête d'enchère de playgen v2 *est* ce clone, et
elle fait **48,2 %** en h2h contre v6 sur 3000 matchs. Un clone ne bat pas son
maître.

Le chemin vers un meilleur bidder passe par une meilleure **couche de scores**,
donc par `enrich_pool_isdd` sur `base_5M` — un autre run, un autre budget.

L'intuition « meilleur corpus → meilleur bidder » est juste, mais **en boucle
longue** : meilleur playgen → meilleurs mondes IS-DD → meilleur jeu → meilleures
couches de scores → meilleur bidder. Deux générations distinctes, pas une.

---

## 4. L'ordre, pour ne pas payer deux fois

Si **v7 gagne l'arène, le corpus doit être généré avec v7**. Sinon playgen v3
apprend la distribution d'enchères de v6 et se retrouve hors distribution le jour
où v7 passe en prod — exactement le problème que le jeton de score est censé
résoudre. Générer avant, c'est risquer de jeter 1 à 3 jours de GPU.

> campagne v7 (~29 h) → COLVGM02 → génération

---

## 5. playgen v3-small n'est pas le raccourci qu'il paraît

Le modèle réduit (d=256 L=4, 3,22M contre 10,74M) rend **4,32 donnes/s contre
2,62**, soit **1,65×** — le plus gros levier de débit restant. Deux choses le
disqualifient pour *ce* corpus.

**Ses mondes ne sont pas ceux de v2** : 2,09× le plancher de bruit
d'échantillonnage, mesuré par `bench_prefill_eq` contre le témoin honnête (deux
tirages du *même* modèle, qui s'écartent toujours de ~0,013). C'est un autre
échantillonneur, pas une optimisation.

**Et son déficit est concentré exactement là où IS-DD décide.** Continuations
cumulées, à budget d'échantillons égal :

| depuis le pli | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| v3-small / v2 | **3,6×** | **3,4×** | **3,2×** | 2,8× | 2,4× | 1,9× | 1,5× | 1,06× |

Or `isdd_dets_by_stage` place tout le regret au-dessus de 0,10 point DD à **8-6
cartes restantes** — les plis 1 à 3 — et **zéro sous 3 cartes**. Les deux courbes
se superposent : v3-small dégrade les mondes précisément aux plis qui décident,
et rejoint v2 là où plus rien ne se joue. Même leçon que le calendrier de mondes :
le début de donne est cher *et* c'est là que ça compte.

### La comparaison qui rendrait la question décidable

| | mondes/décision | donnes/s |
|---|---|---|
| v2 | 20 | 3,93 |
| v3-small | 40 | ~3,9 |

**Même prix.** Donc ce n'est pas « moins cher mais moins bon », c'est « deux fois
plus de mondes de moindre qualité contre moitié moins de mondes fidèles ». Ça,
ça se tranche.

Et pendant une génération **le juge est exact et quasi gratuit** : on connaît les
quatre mains, donc `solve_with_scores` sur la vraie donne donne le coût DD de la
carte que chaque bras a choisie. Pas de biais — un juge à base de playgen noterait
chaque bras sur sa distribution d'origine. C'est le patron de
[`belote_regret.py`](../measurements/README.md) : mesurer **à la décision**, pas
en arène, parce que l'effet plausible est sous le plancher de bruit d'un h2h.

---

## 6. La boucle corpus ↔ modèle, qu'on n'a pas encore regardée en face

On échantillonne les mondes avec playgen *vN* pour produire le corpus qui
entraîne *vN+1*. **Un vN+1 plus gros est un échantillonneur plus lent pour le
corpus de vN+2.**

Ce n'est pas théorique : `playgen_v3_large2` s'entraîne en d=512 L=8, **25,3M
paramètres, 2,4× v2**. Son coût par monde se situe entre **1,3× et 2,4×** celui
de v2 selon que le lot est borné par les lancements de noyaux (~2,1 ms par pas,
que ce soit 1 lane ou 40) ou par l'arithmétique. L'écart entre ces deux bornes
est trop grand pour dimensionner un run de trois jours dessus.

Rien dans les mesures actuelles ne dit où s'arrête ce compromis — et c'est
peut-être la question la plus intéressante de la fiche.

---

## 7. Prochaines étapes

Rangées par rapport valeur / coût. Les trois premières ne demandent **aucune
génération**.

### 7.1 Combien de donnes distinctes faut-il vraiment ? (~7 h GPU)

C'est le paramètre qui décide entre 7 h et 3 jours de génération, et **personne
ne l'a mesuré**.

v2 a vu 30,7M échantillons (160K pas × lot 192) tirés de 9M donnes, soit **3,4
vues par donne** sur 96 possibles (4 observateurs × 24 permutations de couleurs).
À 300 k donnes ce serait 100 vues par donne. L'augmentation n'est pas de
l'information nouvelle, et on ne sait pas où ça décroche.

**Ça se teste sur le corpus qu'on a déjà** : sous-échantillonner les 9M à
100 k / 300 k / 1 M et entraîner à budget d'échantillons constant, en lisant la
perplexité *par pli* (pas seulement au pli 1 — c'est l'erreur du commit `321547a`,
qui avait conclu « v2 est saturé » sur la seule colonne la moins informative de
la table). Config v3-small à 60K pas, ~2,5 h par point.

⚠️ Risque connu de la manip : un modèle plus petit peut saturer sur moins de
données que v2, donc la courbe mesurée est une **borne basse** sur le corpus
nécessaire.

### 7.2 Quelle fraction des donnes se joue à plus de 1000 points d'écart ? (minutes, sans GPU)

C'est le trou explicite de la note playgen v3, et il **gate le jeton de score**
en entier : la pénalité vaut +0,09 à écart 1200 et **zéro** à écart ≤600. Si 80 %
des donnes se jouent au coude à coude, COLVGM02 n'achète presque rien.

Des matchs joués par un bot rapide, histogramme de l'écart à chaque donne. Des
milliers de donnes/s, aucun GPU. **À faire avant de s'engager sur trois jours de
génération.**

### 7.3 Ce que coûte un monde v3-large (minutes, après export)

Voir §6. Une fois `playgen_v3_large2` exporté en `.bin`, un banc de sidecar donne
le coût par monde et referme l'incertitude 1,3×–2,4×.

### 7.4 Le second GPU en tourniquet (~2×, sans changer le joueur)

`worlds.url` accepte déjà une liste séparée par des virgules, répartie sur un
compteur global au processus. **~2× sans changer d'un iota l'échantillonneur qui
produit le corpus** — ça domine v3-small sur tous les axes. Jamais mesuré,
uniquement parce que la 4090 était prise. Le contrôle de santé vérifie les deux
URL au démarrage, donc une carte éteinte échoue au lieu de dégrader en silence.

### 7.5 Si on veut trancher v3-small : l'A/B à coût égal

v3-small @ 40 mondes contre v2 @ 20 mondes, mêmes positions, juge = DD exact sur
la vraie donne (§5). Pas d'arène.

---

## 8. Idées de la fiche qui méritent leur propre mesure

**Le corpus pourrait porter l'étiquette DD, presque gratuitement.** Pendant la
génération on connaît les quatre mains : un `solve_with_scores` de plus par
décision ajoute ~2,5 % au coût *de solve* à 40 mondes — et **~0,1 % au mur**,
puisque le générateur attend le GPU 96 % du temps. On obtiendrait un corpus
annoté du coup DD-optimal et du regret de chaque coup joué, ce qui ouvre : des
cibles de distillation, un juge non biaisé pour tout A/B d'échantillonneur, et
une mesure directe de ce que vaut le joueur qui produit le corpus. Demande un
champ de plus dans COLVGM02 — donc **à décider en même temps que le format**, pas
après.

**Mélanger les corpus.** 9M donnes DouDou50 + 300 k donnes IS-DD sur-échantillonnées.
Standard, pas cher, et ça évite de choisir entre « beaucoup » et « fort ». Brouille
en revanche la prémisse « jeu fort », et personne n'a mesuré ce que le mélange
donne — à traiter comme une troisième condition de la manip §7.1, pas comme un
repli.

---

## Fichiers concernés

- `colver-core/src/bin/gen_games_isdd.rs` — `--match-mode` (`MATCH_TARGET`,
  rotation du donneur), `--dets-schedule`, le profil
- `colver-core/src/playgen/tokens.rs` — là où les deux jetons de score
  atterriraient
- `colver-core/src/game_replay.rs` — `COLVGM01`
- [data_gen/isdd_games.md](../data_gen/isdd_games.md) — le générateur, ses
  mesures, ses impasses
- [belief/playgen.md](../belief/playgen.md) — les tables de perplexité citées ici
- [play/is_dd.md](../play/is_dd.md) — mondes par budget, regret par stade
