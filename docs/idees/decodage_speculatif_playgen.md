# Décodage spéculatif pour playgen — et le budget d'un pas

*État au 2026-08-04. **Rien de ce qui suit n'est mesuré spécifiquement pour cette
fiche** : tous les chiffres sont soit repris d'une mesure existante (source
citée), soit dérivés au crayon et signalés comme tels.*

**L'idée de départ** (utilisateur, 2026-08-04) : `playgen_v3_large2` s'entraîne en
d=512 L=8, 2,4× les paramètres de v2. Plutôt que de payer ce modèle plein pot sur
le sidecar, entraîner un **v3-mini** qui sert de brouillon et faire du **décodage
spéculatif** — le mini propose K cartes, le large les valide en un passage, on
garde la distribution exacte du large pour une fraction du coût.

L'idée est juste dans son principe et, telle quelle, elle vise **le mauvais poste
de coût**. Ce qui suit dit pourquoi, ce que ça rendrait quand même, et ce qui
rendrait plus pour moins cher.

---

## 1. Ce qu'on a sous la main

| modèle | config | params | couches | statut |
|---|---|---|---|---|
| **v2** | d=384 L=6 H=8 | 10,74 M | 6 | `playgen_v2_final.bin`, servi par le sidecar de prod |
| **v3-small** | d=256 L=4 H=8 | 3,22 M | 4 | fini (120 k pas) — **brouillon déjà disponible, même tokeniseur** |
| **v3-large** | d=512 L=8 H=8 | 25,34 M | 8 | `large` 30 k pas, puis `large2` relancé dessus (batch 96, lr 2e-4, 80 k visés) |

Point qui compte pour toute la suite : v3-large est **2,4× v2 en paramètres mais
seulement 1,33× en couches**.

Et v3-small existe déjà, entraîné sur le même corpus au même format : si on
voulait tenter le spéculatif, **le brouillon ne coûterait aucun entraînement**.
C'est l'argument le plus fort en faveur de l'idée.

---

## 2. Le budget d'un pas : ~85-95 % de surcoût fixe

Le décodage spéculatif achète du **coût de modèle**. Dans notre boucle, le coût
de modèle est presque nul.

Un `forward_step` sur v3-large à 40 lanes, au crayon :

| poste | coût |
|---|---|
| arithmétique (2 × 25,3 M × 40 ≈ 2 GFLOP) | ~0,06 ms à pleine vitesse, ~0,3 ms à 20 % d'efficacité |
| lecture des poids (101 Mo à ~900 Go/s) | ~0,11 ms |
| **mesuré** (v2, plus petit encore) | **~2,1 ms**, `forward_prefill` |

Sur v2 c'est encore plus net : ~0,02 ms d'arithmétique et ~0,05 ms de poids pour
un pas mesuré à 2,1 ms — **le modèle pèse moins de 5 % du pas**.

Le reste est du surcoût fixe, et on sait où il est :

- **~35-40 noyaux CUDA par couche** dans `forward_step` (rmsnorm en 6 opérations,
  softmax non fusionné, biais séparés des matmuls, narrow/reshape/transpose) ;
  × 8 couches ≈ 300 lancements par forward.
- **une synchronisation GPU→CPU par carte** : `card_logits` finit par `to_vec2`.
  Elle est **structurelle**, pas accidentelle — le masque de légalité se calcule
  côté CPU par la machine à états (`gens[m].hidden_mask`, `legal_for_hand`), donc
  il faut redescendre les logits.
- **le masque `mh_dec` reconstruit et ré-uploadé à chaque pas** alors qu'une
  seule colonne change (`Tensor::from_slice` de `n_act × cap` flottants).

C'est exactement le diagnostic qui a déjà payé une fois : le préfixe groupé
(1,47×) et le retrait de lanes (1,22×) attaquaient tous les deux le **nombre de
pas**, pas la taille du modèle.

⚠️ **Incohérence à lever** : l'en-tête de `gpu.rs` annonce « ~6 ms/pas quasi
indépendant du batch sur une 4090 pour le modèle v2 » et le commentaire de
`forward_prefill` « ~2,1 ms qu'il y ait 1 lane ou 40 ». L'un des deux est
périmé — probablement l'en-tête, antérieur aux deux optimisations. Toute cette
fiche raisonne sur 2,1 ms ; si c'est 6, les conclusions se renforcent, elles ne
s'inversent pas.

---

## 3. Le coût suit les couches, pas les paramètres

Ce n'est pas une hypothèse, on a le point de mesure :

> v3-small rend **4,32 donnes/s contre 2,62** pour v2, soit **1,65×**
> ([isdd_games.md](../data_gen/isdd_games.md)).

Or v3-small est **3,3× plus petit en paramètres** mais seulement **1,5× moins
profond** (4 couches contre 6). Le 1,65 mesuré est du côté du rapport de couches,
pas du rapport de paramètres — ce qu'on attend si le pas est borné par le nombre
de lancements de noyaux, lequel est proportionnel à la profondeur.

**Deux conséquences, et ce sont les deux qui décident.**

### 3.1 Le brouillon ne serait pas bon marché

Un brouillon v3-small (L=4) face à v3-large (L=8) ne coûte pas 1/8 d'un pas
cible — il en coûte **~1/2**. Dans les formules de décodage spéculatif, c'est ce
ratio `c` qui plafonne tout, et il est huit fois plus mauvais que ce que le
compte de paramètres laisse croire.

### 3.2 v3-large est probablement déjà abordable en prod

Si le coût suit les couches, v3-large ≈ **1,33× v2 par pas**. La latence de prod
est passée à **75 ms par coup** (A/B alterné du 2026-08-04) dans un budget de
1200 ms. À 1,33× ça ferait ~100 ms ; même en supposant le double (2,4×, régime
arithmétique pur) on serait à ~180 ms.

**On peut vraisemblablement servir v3-large tel quel.** Si c'est vrai, la
prémisse entière de l'idée tombe : il n'y a pas de facture à réduire.

*Non mesuré.* C'est la mesure §7.1, et elle coûte dix minutes.

---

## 4. Ce que le spéculatif rendrait quand même

Speedup = `(1 − α^(K+1)) / (1 − α) / (K·c + 1)`, avec `c` = coût d'un pas
brouillon rapporté à un pas cible, `α` = taux d'acceptation, `K` = profondeur de
spéculation.

À `c = 0,5` (v3-small contre v3-large) :

| α | K=2 | K=3 | K=4 |
|---|---|---|---|
| 0,80 | 1,22× | 1,18× | 1,12× |
| 0,85 | 1,29× | 1,27× | 1,24× |
| 0,90 | 1,36× | 1,38× | 1,37× |
| 0,95 | 1,42× | 1,48× | 1,51× |

**Le plafond est ~1,5×, et seulement avec une acceptation de 95 %.**

### L'acceptation, elle, serait probablement bonne

C'est le point qui surprend : ce n'est **pas** α qui pose problème.
`bench_playgen_ppl` donne un écart v3-small / v2 de **+1,7 % à +11,7 % de
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

### Le pincement, en une phrase

- **À petit lot** (prod, latence) : borné par les lancements → le brouillon n'est
  pas bon marché → gain mince.
- **À gros lot** (génération de masse, débit) : l'arithmétique redevient réelle
  (~13 GFLOP par forward à 256 lanes) → la passe de vérification à K tokens coûte
  vraiment K× → on rend ce qu'on gagne.

Le décodage spéculatif brille au milieu : lot 1, borné par la bande passante
mémoire, gros modèle. Notre modèle fait 25 M de paramètres et tourne à 40-256
lanes. **On n'est dans aucun des deux régimes qui le justifient.**

---

## 5. Ce qui coûterait moins et rendrait plus

### 5.1 Fusionner les deux forwards par carte (~1,8×, exact)

**Le plus gros levier identifié, et il ne demande pas de second modèle.**

Aujourd'hui chaque carte coûte **deux** `forward_step` :

```rust
let t_slot = lmax + 2 * step_i;        // jeton ACT   → rend les logits
...
let t_slot = lmax + 2 * step_i + 1;    // jeton carte → n'écrit que le KV
```

Or le jeton ACT du pas *i+1* est **entièrement déterminé** dès que la carte du
pas *i* est tirée : c'est `P_ACT0 + r` avec `r = (gens[m].current + 4 −
observer) % 4`, et `gens[m].step(actor, card)` vient justement de mettre
`current` à jour. Rien n'est échantillonné là.

Donc `[carte_i, act_{i+1}]` peut passer en **un seul bloc de 2 positions**,
logits lus à la seconde. La machinerie existe déjà : c'est `forward_prefill`,
qui prend un bloc de `t` jetons, écrit le KV à `t0` par `slice_set`, et porte la
causalité *et* le remplissage dans son masque additif. Le `act_0` initial part
avec le préfixe, gratuitement.

**48 forwards → 25 par monde d'entame.** Le `cap` ne change pas (il vaut déjà
`lmax + 2·steps`). Une seule synchronisation par carte, comme aujourd'hui.

Prudence : mathématiquement identique mais **pas bit-à-bit** (formes de matmul
différentes ⇒ ordre de réduction flottant différent), exactement comme le
préfixe groupé. Il faut donc l'épingler par le même contrôle que
`prefill_batched_matches_sequential`, et l'A/B par la méthode de
`bench_prefill_eq` — marginales p(carte → siège) contre un **témoin qui est le
même modèle contre lui-même** (~0,013 d'écart de base). Comparer les mondes un à
un ne dit rien.

### 5.2 Les cartes forcées, brouillon gratuit et exact

Quand le masque de légalité n'a **qu'un seul bit**, aucun forward n'est
nécessaire pour *choisir* — la règle décide. Avec le bloc du §5.1 on peut alors
étendre : `[carte_i, act_{i+1}, carte_{i+1}, act_{i+2}]` en un passage.

C'est du décodage spéculatif avec un brouillon **parfait et gratuit : les règles
du jeu**. Acceptation 100 %, aucun modèle à entraîner.

Ampleur inconnue mais bornable : le branchement **uniforme sur le masque** tombe
à 4,10 au pli 7 et **1,56 au pli 8**. Les cartes forcées existent, elles sont
concentrées en fin de donne — donc là où il y a peu de cartes. Compter quelques
pour cent, pas un facteur. À ne faire qu'en supplément du §5.1.

### 5.3 Le reste du surcoût fixe

Par ordre décroissant de rapport gain/effort supposé :

- **masque persistant sur le device** : une seule colonne change par pas,
  aujourd'hui on ré-uploade `n_act × cap` flottants deux fois par carte ;
- **moins de noyaux par couche** : softmax fusionné si candle en a un, biais
  fusionnés dans les matmuls, rmsnorm en un noyau ;
- **fp16/bf16** sur les matmuls : inutile à 40 lanes (borné par les lancements),
  probablement payant à 256 où l'arithmétique redevient réelle ;
- **graphes CUDA** : la réponse directe au surcoût de lancement. Support candle
  0.9 **non vérifié** ; probablement absent, donc à traiter comme une piste
  ouverte et pas comme un plan.

---

## 6. Ce qu'on croit sans l'avoir mesuré

La section qui justifie cette page. Tout ce qui suit est une **croyance**, pas un
résultat.

1. **Que le pas est borné par les lancements de noyaux à 40 lanes.** Un seul
   point de mesure l'appuie (2,1 ms indépendant de 1 vs 40 lanes) et il porte sur
   v2, pas sur v3-large. À 256 lanes, on croit que ça bascule vers
   l'arithmétique — jamais vérifié.
2. **Que le coût par pas est proportionnel aux couches.** Appuyé par un unique
   couple (v3-small 1,65× v2, rapport de couches 1,5×). Deux points feraient une
   droite ; on en a un.
3. **Que v3-large coûtera ~1,33× v2.** Extrapolation de la croyance 2. L'écart
   plausible va de 1,33× à 2,4×, et 2,4× resterait tenable en prod.
4. **Que α vaudrait 0,85-0,90.** Inféré d'un écart de perplexité, alors que
   l'acceptation dépend de la distance en variation totale entre les deux
   modèles. Deux quantités différentes ; la première ne borne pas la seconde.
5. **Que la fusion ACT+carte rend ~1,8× et pas 2×.** On suppose qu'un bloc de 2
   jetons coûte à peu près un bloc de 1. Cohérent avec « borné par les
   lancements », donc la croyance 5 dépend de la croyance 1.
6. **Que la prod n'a pas besoin de plus de mondes.** Le balayage de
   déterminisations plafonne vers ~240 et la prod tourne à `--lane-budget 256` ;
   donc accélérer le sidecar pour la *prod* pourrait ne rien acheter en force de
   jeu, contrairement à la génération hors ligne où le gain est linéaire. Jamais
   revérifié depuis les optimisations du 2026-08-04.

---

## 7. Prochaines étapes

Rangées par rapport valeur / coût. Les deux premières coûtent des minutes.

### 7.1 Le coût par pas de v3-large et de v3-small (~10 min, GPU)

**La mesure qui tranche toute la fiche.** Exporter `playgen_v3_large2` en `.bin`,
lancer le sidecar dessus puis sur `v3_small_120000.bin`, `COLVER_PLAYGEN_PROFILE=1`,
lire les ms/pas à 40 et à 256 lanes.

- Rapport ≈ 2 (couches) → `c ≈ 0,5`, le spéculatif plafonne à 1,5×, **et
  v3-large est abordable en prod tel quel**. Fiche close côté spéculatif.
- Rapport ≈ 8 (paramètres) → tout le raisonnement ci-dessus tombe, l'idée
  redevient sérieuse et il faut mesurer α.

Au passage : ça lève l'incohérence 6 ms / 2,1 ms du §2.

### 7.2 La ventilation d'un pas (~10 min, même run)

Le profileur découpe déjà `forward_step` en `qkv / cat cache KV / attention /
proj sortie / FFN`, plus `embed / forward / logits+sync / échantillonnage / cat
masque`. Ça dit directement combien pèsent la synchronisation et le ré-upload du
masque, donc quoi attaquer en premier au §5.3.

### 7.3 Écrire la fusion ACT+carte (~une demi-journée, sans GPU pour l'écrire)

Indépendant de tout le reste, exact, un seul modèle. Avec son contrôle
(`prefill_batched_matches_sequential`) et son A/B (`bench_prefill_eq` contre le
témoin même-modèle). C'est **la chose à faire** si on veut du débit playgen.

### 7.4 Si et seulement si 7.1 dit « rapport 8 » : mesurer α (~1 h, GPU)

Sur des positions du corpus retenu, dérouler v3-large en teacher-forcing et
calculer `E[min(1, q_large/p_small)]` carte par carte, **ventilé par pli**. C'est
le seul chiffre qui manquerait alors, et il se mesure sans écrire une ligne de
décodage spéculatif.

Rappel de méthode si on va jusqu'à l'implémentation : le spéculatif change
l'ordre de consommation du RNG, donc **les mondes ne seront pas les mêmes** —
même piège que le retrait de lanes. L'A/B se fait sur les marginales contre un
témoin, jamais monde par monde.

---

## 8. Quand le spéculatif redeviendrait intéressant

L'ordre est inversé par rapport à l'intuition de départ :

> Le décodage spéculatif ne devient intéressant qu'**après** avoir supprimé le
> surcoût fixe — parce que c'est lui qui rend le brouillon cher. Et une fois le
> surcoût supprimé, le pas est assez petit pour qu'on n'en ait plus besoin.

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
  de décodage de `generate_worlds_multi` (deux forwards par carte), le profileur
  `COLVER_PLAYGEN_PROFILE`, le retrait de lanes
- `colver-core/src/playgen/infer.rs` — `sample_masked`, `WorldLogp`, `PlayGenSpec`
- `colver-core/src/bin/playgen_gpu_server.rs` — le sidecar
- `colver-core/src/bin/bench_prefill_eq.rs` — le témoin honnête pour tout A/B
  d'échantillonneur
- [belief/playgen.md](../belief/playgen.md) — tables de perplexité et de
  branchement citées ici
- [data_gen/isdd_games.md](../data_gen/isdd_games.md) — débits v2 / v3-small
- [corpus_isdd_playgen_v3.md](corpus_isdd_playgen_v3.md) — §5 et §6, la même
  tension coût / capacité vue depuis le corpus
