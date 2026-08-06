# Décodage spéculatif pour playgen — et le budget d'un pas

*Écrite le 2026-08-04. **Révisée le même jour** : le §5.1 de la première version
a été implémenté et mesuré entre-temps (`a49d46d`), ce qui transforme deux
croyances en résultats. Ce qui reste sans mesure est toujours signalé comme tel.*

**L'idée de départ** (utilisateur, 2026-08-04) : `playgen_v2belote_large2` s'entraîne
en d=512 L=8, 2,4× les paramètres de v2. Plutôt que de payer ce modèle plein pot
sur le sidecar, entraîner un **v3-mini** qui sert de brouillon et faire du
**décodage spéculatif** — le mini propose K cartes, le large les valide en un
passage, on garde la distribution exacte du large pour une fraction du coût.

L'idée est juste dans son principe et, telle quelle, elle vise **le mauvais poste
de coût**. Ce qui suit dit pourquoi, ce que ça rendrait quand même, et ce qui a
rendu plus pour moins cher.

---

## 1. Ce qu'on a sous la main

| modèle | config | params | couches | statut |
|---|---|---|---|---|
| **v2** | d=384 L=6 H=8 | 10,74 M | 6 | `playgen_v2_final.bin`, servi par le sidecar de prod |
| **v2-belote-small** | d=256 L=4 H=8 | 3,22 M | 4 | fini (120 k pas) — **brouillon déjà disponible, même tokeniseur** |
| **v2-belote-large** | d=512 L=8 H=8 | 25,34 M | 8 | `large` 30 k pas, puis `large2` relancé dessus (batch 96, lr 2e-4, 80 k visés) |

Point qui compte pour toute la suite : v2-belote-large est **2,4× v2 en paramètres mais
seulement 1,33× en couches**.

Et v2-belote-small existe déjà, entraîné sur le même corpus au même format : si on
voulait tenter le spéculatif, **le brouillon ne coûterait aucun entraînement**.
C'est l'argument le plus fort en faveur de l'idée.

---

## 2. Le budget d'un pas : le modèle en est une portion mineure

Le décodage spéculatif achète du **coût de modèle**. Dans notre boucle, le coût
de modèle est presque nul.

Un `forward_step` sur v2-belote-large à 40 lanes, au crayon :

| poste | coût |
|---|---|
| arithmétique (2 × 25,3 M × 40 ≈ 2 GFLOP) | ~0,06 ms à pleine vitesse, ~0,3 ms à 20 % d'efficacité |
| lecture des poids (101 Mo à ~900 Go/s) | ~0,11 ms |
| **mesuré** (v2, plus petit encore) | **~2,1 ms** |

Sur v2 c'est encore plus net : ~0,02 ms d'arithmétique et ~0,05 ms de poids pour
un pas mesuré à 2,1 ms — **le modèle pèse moins de 5 % du pas**.

Le reste est du surcoût fixe, et depuis le profilage de
[isdd_games.md](../data_gen/isdd_games.md) on sait où :

- **l'attention pèse ~40 % du forward pour ~2,5 % des FLOP** — ce sont `lanes ×
  têtes` gemms minuscules à M = 1, le pire cas pour cuBLAS, et leur coût est
  proportionnel à `cap` (dimensionné sur la position la plus longue du lot), pas
  à la position courante ;
- **~35-40 noyaux CUDA par couche** dans `forward_step` (rmsnorm en 6
  opérations, softmax non fusionné, biais séparés des matmuls, narrow / reshape
  / transpose) ; × 8 couches ≈ 300 lancements par forward ;
- **une synchronisation GPU→CPU par carte** : `card_logits` finit par `to_vec2`.
  Elle est **structurelle**, pas accidentelle — le masque de légalité se calcule
  côté CPU par la machine à états (`gens[m].hidden_mask`, `legal_for_hand`), donc
  il faut redescendre les logits.
- **le masque `mh_dec` reconstruit et ré-uploadé à chaque pas** alors qu'une
  seule colonne change.

C'est exactement le diagnostic qui a déjà payé trois fois : préfixe groupé
(1,47×), retrait de lanes (1,22×), fusion ACT+CARD (1,62×) — tous les trois
attaquent le **nombre de pas** ou la **largeur de l'attention**, aucun ne touche
à la taille du modèle.

⚠️ **Incohérence toujours en place** : l'en-tête de `gpu.rs` annonce « ~6 ms/pas
quasi indépendant du batch sur une 4090 pour le modèle v2 » et le commentaire de
`forward_prefill` « ~2,1 ms qu'il y ait 1 lane ou 40 ». L'en-tête est
vraisemblablement antérieur aux optimisations ; il n'a pas été corrigé.

---

## 3. Le coût suit les couches, pas les paramètres

Ce n'est pas une hypothèse, on a le point de mesure :

> v2-belote-small rend **4,32 donnes/s contre 2,62** pour v2, soit **1,65×**
> ([isdd_games.md](../data_gen/isdd_games.md)).

Or v2-belote-small est **3,3× plus petit en paramètres** mais seulement **1,5× moins
profond** (4 couches contre 6). Le 1,65 mesuré est du côté du rapport de couches,
pas du rapport de paramètres — ce qu'on attend si le pas est borné par le nombre
de lancements de noyaux, lequel est proportionnel à la profondeur.

**Deux conséquences, et ce sont les deux qui décident.**

### 3.1 Le brouillon ne serait pas bon marché

Un brouillon v2-belote-small (L=4) face à v2-belote-large (L=8) ne coûte pas 1/8 d'un pas
cible — il en coûte **~1/2**. Dans les formules de décodage spéculatif, c'est ce
ratio `c` qui plafonne tout, et il est huit fois plus mauvais que ce que le
compte de paramètres laisse croire.

### 3.2 v2-belote-large est probablement déjà abordable en prod

Si le coût suit les couches, v2-belote-large ≈ **1,33× v2 par pas**. La latence de prod
est passée à **75 ms par coup** (2026-08-04), puis la fusion ACT+CARD a encore
retiré ~38 % sur le chemin d'IS-DD, dans un budget de 1200 ms. Même en supposant
le pire (2,4×, régime arithmétique pur) on resterait loin sous l'échéance.

**On peut vraisemblablement servir v2-belote-large tel quel.** Si c'est vrai, la
prémisse entière de l'idée tombe : il n'y a pas de facture à réduire.

*Toujours non mesuré.* C'est la mesure §7.1, et elle coûte dix minutes.

---

## 4. Ce que le spéculatif rendrait quand même

Speedup = `(1 − α^(K+1)) / (1 − α) / (K·c + 1)`, avec `c` = coût d'un pas
brouillon rapporté à un pas cible, `α` = taux d'acceptation, `K` = profondeur de
spéculation.

À `c = 0,5` (v2-belote-small contre v2-belote-large) :

| α | K=2 | K=3 | K=4 |
|---|---|---|---|
| 0,80 | 1,22× | 1,18× | 1,12× |
| 0,85 | 1,29× | 1,27× | 1,24× |
| 0,90 | 1,36× | 1,38× | 1,37× |
| 0,95 | 1,42× | 1,48× | 1,51× |

**Le plafond est ~1,5×, et seulement avec une acceptation de 95 %.** À comparer
au 1,62× qu'a rendu la fusion ACT+CARD, sans second modèle et sans approximation.

### L'acceptation, elle, serait probablement bonne

C'est le point qui surprend : ce n'est **pas** α qui pose problème.
`bench_playgen_ppl` donne un écart v2-belote-small / v2 de **+1,7 % à +11,7 % de
perplexité par carte** — les deux modèles sont proches token par token. C'est le
*produit* sur les cartes restantes qui fait les 3,6× de continuations cumulées au
pli 1, pas un désaccord local. Un α de 0,85-0,90 est plausible.

**Le problème est `c`, pas `α`.** Ce qui coûte cher dans un pas ne rétrécit pas
avec le modèle.

### Et deux érosions supplémentaires

1. **Le lot désynchronise.** 40 à 256 lanes indépendantes, chacune accepte un
   nombre différent de tokens par cycle. Le nombre de cycles est fixé par la lane
   la plus malchanceuse, pas par la moyenne. C'est le même mur que `steps_max`,
   que le retrait de lanes a contourné en triant par pas décroissants — ici il
   revient, et il n'y a rien à trier.

2. **L'acceptation sera la pire là où sont les tokens.** Un monde d'entame
   demande 24 cartes cachées, un monde de fin 3 ; et c'est précisément au pli 1
   que small et large divergent le plus (3,6× cumulé, contre 1,06× au pli 8). La
   moyenne pondérée par le volume de tokens penche du mauvais côté.

Ce deuxième point **tue aussi la variante « cascade »** (petit modèle en fin de
donne, gros au début, aiguillé par requête) : le coût et le besoin de capacité
sont **co-localisés dans les premiers plis**. On ne peut pas économiser là où
c'est cher sans dégrader là où ça compte. Même leçon que le calendrier de mondes
par stade (`dets_schedule`) et que le §5 de
[corpus_isdd_playgen_v3.md](corpus_isdd_playgen_v3.md).

### Le pincement — désormais mesuré, plus déduit

- **À petit lot** (prod, latence) : borné par les lancements → le brouillon n'est
  pas bon marché → gain mince.
- **À gros lot** (génération de masse, débit) : l'arithmétique redevient réelle →
  la passe de vérification à K tokens coûte vraiment K× → on rend ce qu'on gagne.

La fusion ACT+CARD donne les deux régimes sur le même changement : **1,62× à 40
mondes contre 1,33× à 256**. Retirer un lancement sur deux rend nettement moins
quand le forward fait du vrai calcul. C'est la même bascule qui s'appliquerait à
un brouillon.

Le décodage spéculatif brille au milieu : lot 1, borné par la bande passante
mémoire, gros modèle. Notre modèle fait 25 M de paramètres et tourne à 40-256
lanes. **On n'est dans aucun des deux régimes qui le justifient.**

---

## 5. Ce qui coûte moins et rend plus

### 5.1 ✅ Fusionner les deux forwards par carte — **fait, 1,62×** (`a49d46d`)

C'était le §5.1 de la première version de cette fiche, en « à faire ». Implémenté
et mesuré dans la foulée. Détail complet, mesures et validation :
[isdd_games.md §4](../data_gen/isdd_games.md).

Le raisonnement était : chaque carte coûtait **deux** `forward_step`, un pour le
jeton ACT (dont on lit les logits) et un pour le jeton CARD (**dont la sortie
était jetée**) ; or l'ACT du coup suivant est déterministe dès que la carte est
tirée, donc les deux partent en un bloc de 2 jetons via `forward_prefill`.

**Ce que l'implémentation a trouvé en plus** : la **dernière carte d'une lane n'a
jamais besoin d'être poussée** — un k/v n'est lu que par les positions
ultérieures de la *même* lane, et il n'y en a plus. Le décodage passe donc de
`2·steps` à `steps` lancements exactement, pas à `steps + 1`, et ça se compose
proprement avec le retrait de lanes.

**Prédiction contre mesure** : la fiche annonçait « ~1,8× », on a obtenu 1,62× à
40 mondes et 1,33× à 256. La direction était bonne, l'amplitude un peu
optimiste — un bloc de 2 jetons n'est pas tout à fait gratuit par rapport à un
bloc de 1. C'est ce qui recalibre la croyance 1 du §6.

### 5.2 Les cartes forcées, brouillon gratuit et exact — **toujours ouvert**

Quand le masque de légalité n'a **qu'un seul bit**, aucun forward n'est
nécessaire pour *choisir* — la règle décide. Le bloc du §5.1 étant maintenant en
place, on peut étendre : `[carte_i, act_{i+1}, carte_{i+1}, act_{i+2}]` en un
passage.

C'est du décodage spéculatif avec un brouillon **parfait et gratuit : les règles
du jeu**. Acceptation 100 %, aucun modèle à entraîner, et la machinerie de bloc
existe déjà.

Ampleur inconnue mais bornable : le branchement **uniforme sur le masque** tombe
à 4,10 au pli 7 et **1,56 au pli 8**. Les cartes forcées existent, elles sont
concentrées en fin de donne — donc là où il y a peu de cartes. Compter quelques
pour cent, pas un facteur.

### 5.3 Le reste du surcoût fixe

Le profilage a redistribué les priorités depuis la première version.

**La cible est l'attention** : ~40 % du forward pour ~2,5 % des FLOP, avec un
coût proportionnel à `cap` alors que seules les colonnes `[0, t]` portent quelque
chose. Diviser la largeur par deux diviserait un poste à 40 %. **Bloqué par
candle 0.9.2**, qui refuse de multiplier des tenseurs non contigus (`narrow` sur
l'axe `cap` en produit un) ; recopier soi-même coûterait plus que ce qu'on
économise. Rouvrable avec un chemin gemm à `lda`, ou un noyau d'attention dédié.

Ensuite, par ordre décroissant de rapport gain/effort supposé :

- **masque persistant sur le device** : une seule colonne change par pas,
  aujourd'hui on ré-uploade `n_act × cap` flottants ;
- **moins de noyaux par couche** : softmax fusionné si candle en a un, biais
  fusionnés dans les matmuls, rmsnorm en un noyau ;
- **graphes CUDA** : la réponse directe au surcoût de lancement. Support candle
  0.9 **non vérifié** ; probablement absent, donc piste ouverte et pas plan.

❌ **fp16 / tensor cores : à ne pas rouvrir sans raison neuve.** La première
version de cette fiche le proposait pour le régime à 256 lanes. TF32 a été
essayé depuis : **3 à 5× plus lent** (0,44 et 0,83 donnes/s contre 2,18 et 2,28),
une régression franche, probablement une bascule de cuBLAS vers un noyau sans
tensor cores pour ces formes. fp16 n'est pas TF32, mais c'est la même famille de
chemins sur les mêmes formes minuscules — la charge de la preuve a changé de
camp.

---

## 6. Ce qu'on croit sans l'avoir mesuré

La section qui justifie cette page. Statut mis à jour après `a49d46d`.

1. ✅ **Que le pas est borné par les lancements de noyaux à petit lot, et par
   l'arithmétique à gros lot.** C'était la croyance la plus load-bearing ;
   **confirmée** par la fusion (1,62× à 40 lanes, 1,33× à 256 — le même
   changement rend moins quand le forward calcule vraiment).
2. **Que le coût par pas est proportionnel aux couches.** Toujours appuyé par un
   unique couple (v2-belote-small 1,65× v2, rapport de couches 1,5×). Deux points
   feraient une droite ; on en a un. **C'est la croyance dont tout le reste
   dépend, et c'est celle qui n'a pas bougé.**
3. **Que v2-belote-large coûtera ~1,33× v2.** Extrapolation de la croyance 2. L'écart
   plausible va de 1,33× à 2,4×, et 2,4× resterait tenable en prod.
4. **Que α vaudrait 0,85-0,90.** Inféré d'un écart de perplexité, alors que
   l'acceptation dépend de la distance en variation totale entre les deux
   modèles. Deux quantités différentes ; la première ne borne pas la seconde.
5. ✅ **Que la fusion ACT+carte rendrait ~1,8×.** Mesurée à **1,62× / 1,33×**
   selon la largeur du lot. Optimiste d'environ 10 % à petit lot, et la fiche
   n'avait pas anticipé que le régime à 256 lanes couperait le gain de moitié.
6. **Que la prod n'a pas besoin de plus de mondes.** Le balayage de
   déterminisations plafonne vers ~240 et la prod tourne à `--lane-budget 256` ;
   donc accélérer le sidecar pour la *prod* pourrait ne rien acheter en force de
   jeu, contrairement à la génération hors ligne où le gain est linéaire.
   **Toujours pas revérifié**, et c'est maintenant la question la plus utile de
   la liste : trois optimisations se sont enchaînées sans que personne ne dise
   ce que la dernière achète réellement à un joueur.

---

## 7. Prochaines étapes

### 7.1 Le coût par pas de v2-belote-large et de v2-belote-small (~10 min, GPU)

**La mesure qui tranche toute la fiche**, et elle est plus facile qu'avant :
`bench_sidecar_ab` (ajouté par `a49d46d`) fait l'A/B de latence alterné entre
deux sidecars, ce qui se faisait à la main jusqu'ici. Exporter `playgen_v2belote_large2`
en `.bin`, le mettre en face de `v2belote_small_120000.bin`, à 40 **et** à 256 lanes
(les deux régimes du §4 rendent des réponses différentes).

- Rapport ≈ 2 (couches) → `c ≈ 0,5`, le spéculatif plafonne à 1,5×, **et
  v2-belote-large est abordable en prod tel quel**. Fiche close côté spéculatif.
- Rapport ≈ 8 (paramètres) → tout le raisonnement ci-dessus tombe, l'idée
  redevient sérieuse et il faut mesurer α.

Au passage : ça permet de corriger l'en-tête de `gpu.rs` (§2).

### 7.2 Ce qu'une génération plus rapide achète à un joueur (arène ou décision)

Croyance 6. Trois optimisations du sidecar se sont enchaînées en deux jours
(1,47× · 1,22× · 1,62×) et **personne n'a mesuré ce que la prod y gagne en force
de jeu**, seulement en latence. Si le plateau de déterminisations est vraiment
atteint, la réponse est « rien », et le prochain effort d'optimisation doit aller
à la génération hors ligne — pas au sidecar de prod.

À mesurer **à la décision** (regret DD sur des positions fixées à budget de temps
égal), pas en arène : l'effet plausible est sous le plancher de bruit d'un h2h.
Même patron que `bench_belote_ab`.

### 7.3 Si et seulement si 7.1 dit « rapport 8 » : mesurer α (~1 h, GPU)

Sur des positions du corpus retenu, dérouler v2-belote-large en teacher-forcing et
calculer `E[min(1, q_large/p_small)]` carte par carte, **ventilé par pli**. C'est
le seul chiffre qui manquerait alors, et il se mesure sans écrire une ligne de
décodage spéculatif.

Rappel de méthode si on va jusqu'à l'implémentation : le spéculatif change
l'ordre de consommation du RNG, donc **les mondes ne seront pas les mêmes** —
même piège que le retrait de lanes. L'A/B se fait sur les marginales contre un
témoin même-modèle (`bench_prefill_eq`), jamais monde par monde.

*(Le « écrire la fusion ACT+carte » de la première version était le §7.3 — il est
fait, voir §5.1.)*

---

## 8. Quand le spéculatif redeviendrait intéressant

L'ordre est inversé par rapport à l'intuition de départ :

> Le décodage spéculatif ne devient intéressant qu'**après** avoir supprimé le
> surcoût fixe — parce que c'est lui qui rend le brouillon cher. Et une fois le
> surcoût supprimé, le pas est assez petit pour qu'on n'en ait plus besoin.

La fusion ACT+CARD illustre les deux moitiés à la fois : elle a retiré un
lancement sur deux (donc rapproché du régime où `c` refléterait enfin les
paramètres), **et** elle a rendu 1,62× — c'est-à-dire plus que le plafond du
spéculatif dans le régime actuel.

Les conditions qui le feraient rentrer :

- un modèle nettement plus gros (100 M+), où l'arithmétique et la bande passante
  dominent enfin le lancement ;
- un régime à petit lot où la latence d'**une** séquence compte (ce n'est pas
  notre cas : on tire 40 à 256 mondes indépendants, la parallélisation naturelle
  est le lot, pas la séquence) ;
- ou des graphes CUDA / des noyaux fusionnés qui font tomber le surcoût fixe,
  après quoi `c` refléterait enfin le compte de paramètres et le tableau du §4
  serait à refaire.

---

## Fichiers concernés

- `colver-core/src/playgen/gpu.rs` — `forward_step`, `forward_prefill`, la boucle
  de décodage de `generate_worlds_multi` (désormais fusionnée, ablation
  `COLVER_PLAYGEN_NO_FUSE=1`), le profileur `COLVER_PLAYGEN_PROFILE`, le retrait
  de lanes
- `colver-core/src/playgen/infer.rs` — `sample_masked`, `WorldLogp`, `PlayGenSpec`
- `colver-core/src/bin/playgen_gpu_server.rs` — le sidecar
- `colver-core/src/bin/bench_sidecar_ab.rs` — A/B de latence alterné entre deux
  sidecars (c'est l'outil du §7.1)
- `colver-core/src/bin/bench_prefill_eq.rs` — le témoin honnête pour tout A/B
  d'échantillonneur
- [data_gen/isdd_games.md](../data_gen/isdd_games.md) — **où vivent les chiffres**
  du sidecar : les quatre optimisations, les impasses (TF32, fenêtre
  d'attention), la ventilation du forward
- [belief/playgen.md](../belief/playgen.md) — tables de perplexité et de
  branchement citées ici
- [corpus_isdd_playgen_v3.md](corpus_isdd_playgen_v3.md) — §5 et §6, la même
  tension coût / capacité vue depuis le corpus
