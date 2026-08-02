# Optimisation du solveur DD — journal de campagne

**Campagne du 2026-08-02.** Ce document existe pour une seule raison : **ne pas réexplorer**.
Il enregistre autant ce qui a marché que ce qui n'a pas marché, et pour chaque échec il donne
la mesure, la cause et la commande qui le rejoue. Une piste fermée sans trace est une piste
qu'on rouvrira dans six mois.

Référence courante du solveur : [dd_solver.md](dd_solver.md). Ici, c'est l'historique du
raisonnement.

## Bilan

| | avant | livré | |
|---|---:|---:|---:|
| `solve_with_scores`, donne complète | 1 448 045 nœuds | 1 163 727 | **0,804×** |
| `solve_with_scores`, mondes IS-DD | 55 862 | 48 994 | **0,877×** |
| `solve_with_scores`, mi-donne / finale | 9 061 / 89 | 8 707 / 88 | 0,961× / 0,989× |
| **au chronomètre** (10 tours entrelacés, 10/10) | 4,89 s | 3,94 s | **0,806×** |
| **`gen_pool`, vrai binaire** | 33,2 s | 22,8 s | **0,686×** |

Deux changements le portent : une **recherche courte au sommet** pour choisir le premier coup
(§6) et une **fenêtre entre cartes racine** dans `solve_with_scores` (§8). Aucune valeur DD ne
bouge — 11 527 valeurs par carte, 0 écart, et 392 tests.

**Et le reste de ce document est surtout une liste de choses qui ne marchent pas**, chacune avec
son chiffre : agrandir la TT (§2.1, §2.1bis), MTD(f) (§2.2), l'amorçage entre mondes (§2.3,
§2.3bis), un `apply_play` allégé (§2.4), `-C target-cpu=native` (§2.5), quatre raffinements
écrits à la main de l'ordre statique (§6), un calendrier de profondeur (§6), l'ordre racine par
regard (§8), et toute fonction d'évaluation (§7). C'est le contenu principal, pas l'annexe.

---

## 0. Le point de départ, et pourquoi il fallait d'abord mesurer

Avant cette campagne, **rien ne mesurait ce qui coûte cher**.

`dd_bench` ne connaît qu'une forme : la donne complète résolue par `solve_for_trump`. Or le
travail qui domine le CPU DD du projet est la déterminisation IS-DD — de l'ordre de **2 800
core-h** pour une couche de scores sur 5 M de donnes, contre **~180 core-h** pour `gen_pool` —
et son unité est `solve_with_scores`, qui n'était mesurée nulle part.

Et les chiffres publiés ne s'accordaient pas : **13,5 / 14,9 / 28 / 77 ms** pour des choses
toutes appelées « un solve », aucune avec un corpus ni une forme déclarés. Le tableau de
`dd_solver.md` annonçait 77 ms par solve là où la mesure en donne 34,6.

D'où l'ordre imposé : **le harnais d'abord, les optimisations ensuite.**

### Le profil, une fois mesuré

Les chiffres eux-mêmes — temps par forme, percentiles, dispersion de mesure — vivent dans
[dd_solver.md § Performance](dd_solver.md#performance) et **nulle part ailleurs**. Ce document
n'en garde que les deux qui portent un raisonnement :

- **~22 ns par nœud**, dominé par la **sonde TT**, un accès aléatoire dans 2 Mo : le solveur est
  limité par la **latence mémoire**, pas par le débit d'instructions.
- **Distribution très asymétrique** : les 10 % de solves les plus durs portent **40 % des nœuds**.

Deux conséquences qui expliquent presque tous les résultats plus bas :

1. **22 ns par nœud, c'est déjà serré.** Il n'y a pas de gras à retirer par nœud, et c'est
   pourquoi les micro-optimisations reviennent négatives ou nulles.
2. **Ce qui n'aide que la donne médiane ne vaut presque rien.** Le temps est dans la queue.

---

## 1. Ce qui a été implémenté

### 1.1 La TT s'estampille au lieu de se vider — **18,9× sur les finales**

**Le problème.** La table de transposition n'est valable que pour un couple (donne, atout) :
`position_hash` porte les cartes jouées et le pli en cours, et *déduit* les mains — ce qui ne
tient que si la donne initiale est fixe. Elle ne porte pas l'atout non plus. Elle était donc
`memset` à zéro à chaque point d'entrée.

Ce memset coûte **28,8 µs à plat**. Négligeable devant une donne complète à 35 ms. Écrasant
devant une finale : mesuré sur le corpus, une position à **8 cartes restantes explore 89 nœuds
et prenait 32,9 µs — dont 88 % de memset**. Ce sont exactement les positions où IS-DD en fin de
donne et `/analyse/jeu` passent leur temps.

**L'implémentation** (`solver.rs`). Les entrées de la TT sont des `u64` dont les bits 15-1
étaient inutilisés. On y met une **époque de 15 bits**. `TtBuf::begin_solve` incrémente
l'époque au lieu d'effacer ; une sonde rejette toute entrée qui ne porte pas l'estampille
courante. L'effacement n'a lieu qu'au débordement, **une fois sur 32 767 solves**.

Conséquence d'API : `new_tt_buffer()` rend désormais un `solver::TtBuf` et non un `Vec<u64>`,
et tous les points d'entrée prennent `&mut TtBuf`.

**Le piège qui a failli manger le gain.** Garder `&mut TtBuf` dans la récursion fait recharger
le pointeur du `Vec` à chaque sonde *et* à chaque écriture : **-16 % de nœuds/s**, plus que le
memset qu'on retire. `begin_solve` rend donc `(&mut [u64], u64)` — la tranche et l'estampille —
et `alphabeta` reprend un `&mut [u64]` avec l'estampille en registre. Si quelqu'un « nettoie »
ça un jour en repassant `&mut TtBuf`, la régression revient en silence.

**Le résultat**, A/B entrelacé, minimum sur 3 tours :

| forme | nœuds/pos | époque | memset | gain |
|---|---|---|---|---|
| donne complète | 1 448 045 | 38 327 µs | 38 904 µs | 1,02× |
| mi-partie | 9 061 | 185,7 µs | 212,6 µs | **1,14×** |
| finale | 89 | 1,5 µs | 28,9 µs | **18,9×** |
| mondes IS-DD | 55 862 | 1 327 µs | 1 350 µs | 1,02× |

Compte de nœuds **identique au nœud près**, checksum de valeurs identique. Le gain est ciblé,
pas global : c'est une suppression de coût fixe, donc il vaut ce que le coût fixe pesait.

**Tests qui le tiennent** : `test_epoch_invalidates_across_deals` (une table réutilisée ne doit
jamais servir une entrée d'une autre donne — comparaison contre une table neuve),
`test_epoch_wrap_is_sound` (le débordement efface, il ne revalide pas), et
`test_tt_pack_negative_score_leaves_epoch_intact` (un score négatif stocké en complément à deux
ne doit pas déborder dans le champ d'époque, sinon des entrées périmées paraissent fraîches).

### 1.2 L'équivalence de cartes devient une table — gain non mesurable, mais prouvé

`reduce_equivalent` tourne à **chaque nœud interne** et faisait, côté atout, deux `sort_unstable`
plus un balayage en O(n·m).

Les tables de points disent que tout ça se réduit :

- **Couleur.** `PLAIN_POINTS = [0,0,0,2,3,4,10,11]` : seuls le 7, le 8 et le 9 peuvent être à
  égalité (0 point), tous les autres rangs ont une valeur unique. Et la seule paire séparée par
  une carte est **{7,9}**, séparée par le 8. La réduction est donc une fonction de trois bits de
  jouabilité et d'un bit de « le 8 est-il chez les autres » — **16 cas, une table**.
- **Atout.** `TRUMP_POINTS = [0,0,14,20,3,4,10,11]` : la seule égalité est **{7,8}**, dont les
  forces `TRUMP_STRENGTH` sont **0 et 1 — adjacentes**. Rien ne peut se glisser entre elles, le
  test « y a-t-il une carte outstanding entre les deux » est donc **vide par construction**, et
  toute la fonction se ramène à : *si le 7 et le 8 d'atout sont jouables, le 7 est redondant*.

**Vérifié par exhaustion**, pas par échantillon : `test_plain_lut_matches_reference` et
`test_trump_rule_matches_reference` comparent les remplacements aux boucles d'origine (gardées
sous `#[cfg(test)]`) sur **65 536 entrées × 4 couleurs** chacun. Un troisième test,
`test_equivalence_derivation_assumptions_still_hold`, épingle les hypothèses : si une règle
revalorise une carte, il casse au lieu de laisser la table mentir en silence.

**Gain de vitesse : non mesurable** au bruit près. Gardé pour la simplicité et parce que la
dérivation est maintenant assertée.

### 1.3 Deux gaspillages francs chez les appelants

- **`OraclePlayer` faisait deux recherches complètes par carte** (`agent/dmc.rs`) :
  `solve_with_scores`, puis `solve_best_card` sur le même arbre, chacune avec sa propre table de
  2 Mo fraîche. `scores.best_card` était déjà là. Il garde en plus sa TT d'un coup à l'autre.
- **Le binding PyO3 `solve_scores` ne relâchait pas le GIL** (`colver-py/src/lib.rs`), alors que
  `card_analysis.py` l'éclate sur un pool de threads et que le commentaire voisin affirme « le
  solveur relâche le GIL ». C'est vrai de `solve_all_suits`, pas de celui-là : le fan-out de
  `/analyse/jeu` était **sérialisé**, 200 à 500 solves à la file.

### 1.5 IID d'ordonnancement au sommet → **§6**

Une recherche courte choisit le premier coup à essayer près de la racine. Vient en droite ligne
des mesures du §5. Détail, garde, constantes et réplication au **§6**.

### 1.6 Fenêtre entre cartes racine → **§8**

`solve_with_scores` donnait à chaque carte une fenêtre pleine alors que la précédente venait de
dire, à quatre points près, où était la réponse. **C'est la seule optimisation qui morde
franchement sur les mondes IS-DD**, donc sur les ~2 800 core-h d'une couche de scores. Détail au
**§8**.

**Les deux ensemble : 0,805× au chronomètre**, 0,804× en nœuds sur donne complète et 0,877× sur
les mondes.

### 1.4 Le harnais — `bench_dd`, et la discipline de mesure

`colver-core/src/bin/bench_dd.rs`, quatre sous-commandes :

```bash
cargo build --release --features "parallel solver_stats" --bin bench_dd

# corpus figé — à construire une fois, puis à garder
./target/release/bench_dd build --out data/analysis/dd_corpus_v1.bin \
    --pool data/deals/base_5M.bin --games data/training/heldout_20k_s90210.bin

# mesure + toutes les valeurs par carte
./target/release/bench_dd run --corpus data/analysis/dd_corpus_v1.bin \
    --values cand.vals --repeats 5

# la porte : doit dire EXACT MATCH
./target/release/bench_dd diff --a baseline.vals --b cand.vals

# le plafond d'un amorçage de fenêtre parfait, par écart toléré (§2.3bis)
./target/release/bench_dd oracle --corpus data/analysis/dd_corpus_v1.bin --deltas 1,5,20,40

# ce que valent PVS / coups tueurs / historique (§3) — 5 configs + porte d'exactitude
scripts/analysis/dd_ablation.sh

# le plafond d'un ordonnancement parfait, avec découpe par difficulté (§5)
./target/release/bench_dd ordering --corpus data/analysis/dd_corpus_v1.bin --threads 8

# balayage de taille de TT, 1 thread et N (§2.1bis) — répéter la liste entrelace
./target/release/bench_tt_size --deals 400 --threads 32 --sizes 16,18,16,18,16,18

# A/B alternant deux révisions git, minimum sur N tours
scripts/analysis/dd_ab_revs.sh <rev-de-référence> 3

# A/B de trois cibles de compilation construites depuis la même source (§2.5)
scripts/analysis/dd_ab_flags.sh

# et la version journalisée, qui écrit dans docs/measurements/index.jsonl
python3 scripts/analysis/dd_solver_bench.py --tag <nom> --repeats 5 --note "..."
```

**Le motif à réutiliser, c'est `oracle`.** Devant une famille d'idées qui ne diffèrent que par
la qualité d'une estimation — amorcer une fenêtre, ordonner des coups, choisir une borne — il
est presque toujours moins cher de mesurer ce que ferait l'estimation **parfaite** que d'en
construire une bonne. Il a servi deux fois et a répondu deux choses opposées en une commande
chacune : plafond bas, famille close (§2.3bis) ; plafond haut, et on sait à quoi comparer une
règle avant de l'écrire (§5). Les bornes du §4.2 sont le troisième candidat évident.

Le corpus fait **2 120 positions** en quatre formes : donnes complètes (depuis `base_5M.bin` —
ses `dd_pts` sont périmés mais ses `hands` ne sont qu'une distribution de donnes et restent
valables), mi-partie et finales **tirées de vraies parties jouées** (COLVGM01), et les lots de
mondes déterminisés qui sont l'unité réelle d'IS-DD.

**Quatre règles apprises à la dure, chacune après s'être fait avoir :**

1. **Le compte de nœuds d'abord, le temps ensuite.** Sur un CPU hybride P/E sous WSL2, le temps
   au mur ne distingue pas « meilleur élagage » de « tombé sur un cœur P ». D'où la feature
   `solver_stats`.
2. **Ne jamais comparer deux exécutions séquentielles.** Un même binaire mesuré deux fois ici a
   varié de **20 %** parce qu'un autre travail avait démarré entre les deux — plus grand que la
   plupart des gains cherchés. Mesurée ainsi, la TT à époque semblait partir de **-20 %** ; en
   A/B entrelacé elle est à +2 % à +1 800 %. `--ab` entrelace dans un seul processus,
   `dd_ab_revs.sh` alterne deux binaires construits depuis deux révisions. Les deux gardent le
   **minimum**, parce qu'une charge concurrente ne fait qu'ajouter du temps.
3. **Le corpus est un fichier, écrit une fois et gardé — jamais une graine rejouée.** Aucun
   générateur du dépôt n'est reproductible : `generate_chunk_into` distribue les indices de
   slot par `AtomicUsize::fetch_add` à N workers, donc le flux RNG qui atterrit à un indice
   donné dépend de l'ordonnancement des threads.
4. **L'exactitude est une porte, pas un contrôle a posteriori.** Chaque run écrit **toutes** les
   valeurs par carte ; `diff` exige l'égalité stricte. C'est la seule chose qui sépare une
   optimisation d'un second `quick_tricks` — dont le verdict « 25 % de valeurs fausses » a été
   obtenu avec un harnais **jamais commité**, et donc irrejouable.

---

## 2. Ce qui a été tenté et ne sera pas implémenté

Toutes les pistes ci-dessous ont été **mesurées**, pas estimées. Chacune donne sa cause et sa
commande de réplication.

### 2.1 Agrandir la table de transposition — **non, 2,4× plus lent**

**L'hypothèse, qui semblait solide.** Sur les solves les plus durs, la TT est remplie à
**99,4 %** avec **21,5 écritures par slot** : presque chaque écriture évince une entrée vivante.
Le folklore disait « une TT plus grande a régressé », et j'ai cru que la cause en était le
memset qui grandit avec la table — donc qu'en supprimant le memset (§1.1), une grande table
deviendrait gagnante.

**La mesure dit le contraire**, et le memset n'y est pour rien (colonne `search_ms` = temps hors
memset, 60 donnes × 4 couleurs, mono-thread) :

| log2 entrées | taille | nœuds (M) | recherche seule |
|---|---|---|---|
| 14 | 0,1 Mo | 175,0 | 3 721 ms |
| 16 | 0,5 Mo | 126,1 | **2 908 ms** |
| **18 (actuel)** | **2,1 Mo** | **106,3** | **2 968 ms** |
| 20 | 8,4 Mo | 103,6 | 4 496 ms |
| 22 | 33,6 Mo | 102,7 | 5 654 ms |
| 24 | 134,2 Mo | 103,1 | 7 221 ms |

**Pourquoi.** Passer de 2 Mo à 134 Mo ne gagne que **3 % de nœuds** — le taux d'écrasement de
21,5× ne coûte presque rien parce que le parcours en profondeur d'abord réutilise surtout les
entrées récentes — et coûte **2,4× en temps** en défauts de cache. `1<<18` est à l'optimum ;
`1<<16` (qui tient en L2) est même marginalement meilleur en temps au prix de 19 % de nœuds en
plus. Le folklore avait raison, mais pas pour la raison qu'on lui prêtait.

**Réplication** : passer une tranche de taille arbitraire (puissance de deux) à
`solve_for_trump_reuse_tt` — le solveur masque avec `len()-1`, donc toute taille est légale —
et chronométrer le memset séparément à chaque taille. Aujourd'hui il faut construire un
`TtBuf::with_log2_size(n)`.

**Le volet 32 threads a été exécuté depuis, et ne change rien** — voir §2.1bis.

### 2.1bis Le même balayage en 32 threads — **non, la constante est confirmée**

C'était « le levier ouvert le moins cher du dépôt » : en 32 threads, 32 × 2 Mo = 64 Mo de
working set pour 36 Mo de L3 sur ce 13900K, et `1<<16` ramènerait ça à 16 Mo. `bench_tt_size`
existait pour trancher et n'avait jamais tourné. Il a tourné.

Le bench a d'abord été réparé sur deux points qui l'auraient rendu ininterprétable : il allouait
**une TT par donne** (la production en alloue une par worker et la réestampille), et il ne
comptait **pas les nœuds** — or deux effets opposés vivent dans le temps mesuré. Une table plus
grande **entre en collision moins souvent, donc élague mieux et visite moins de nœuds** ; elle
**sort du cache, donc chaque sonde coûte plus cher**. Sans la colonne de nœuds les deux sont
indiscernables. 400 donnes × 4 couleurs, minimum sur des tailles **alternées** :

| bits | par thread | nœuds/solve | ms/solve 1T | ns/nœud 32T | ms/solve 32T |
|---|---|---:|---:|---:|---:|
| 14 | 128 Ko | 1 119 298 | 21,66 | 48,7 | 54,47 |
| 16 | 512 Ko | 802 587 | 16,03 | 48,2 | **35,41** |
| **18 (actuel)** | **2 Mo** | **664 321** | **14,59** | 55,4 | 36,79 |
| 20 | 8 Mo | 622 525 | 22,45 | 100,3 | 62,45 |
| 22 | 32 Mo | 613 764 | — | 175,9 | 107,98 |

**512 Ko et 2 Mo sont à égalité** : 2 Mo gagne de 7 % à 1 thread, 512 Ko de 4 % en 32, les deux
sous le plancher de bruit (~9 %, et la machine était chargée). Ce n'est pas une indécision, c'est
le mécanisme : **512 Ko visite 1,21× les nœuds à 0,83× le coût par nœud — produit 1,00**. Le
compromis est plat autour de la constante actuelle, ce qui est précisément la raison pour
laquelle il n'y a rien à y gagner. Les voisins d'un facteur 8 sont eux nettement moins bons
(+46 % à 128 Ko, +52 % à 8 Mo, 2,6× à 32 Mo).

L'hypothèse du thrash L3 n'est donc pas fausse — elle mord à partir de 8 Mo par thread — mais
elle ne mord pas là où on est. **Ne pas toucher `TT_SIZE`.**

**Piège rencontré, et c'est le plus instructif de la mesure.** La première passe, non alternée,
disait que l'optimum à 1 thread était **128 Ko, 14 % devant 2 Mo**. C'était du bruit :
l'alternance le retourne complètement (128 Ko finit 46 % *derrière*). Un balayage de tailles
mesurées l'une après l'autre est exactement le motif que
[la règle du dépôt interdit](../measurements/README.md) — et il produisait ici une conclusion non
seulement fausse mais actionnable, donc du code inutile écrit avec confiance. `bench_tt_size`
mesure les tailles dans l'ordre demandé : **répéter la liste suffit à les entrelacer**
(`--sizes 16,18,16,18,16,18`).

**Réplication** :
```bash
cargo build --release --features "parallel solver_stats" --bin bench_tt_size
./target/release/bench_tt_size --deals 400 --threads 32 --sizes 14,16,18,20,14,16,18,20,14,16,18,20
```

### 2.2 MTD(f) / recherche binaire sur la valeur — **non, 1,94× plus lent**

**L'hypothèse.** C'est *la* grande technique des solveurs DD de bridge : au lieu de chercher la
valeur, on pose des questions booléennes (« ≥ N plis ? ») à fenêtre nulle, qui élaguent
beaucoup plus, et on trouve la valeur par dichotomie. Le folklore du dépôt disait « MTD(f) a
régressé », sans qu'aucune mesure n'existe nulle part — la chaîne `MTD` n'apparaît dans aucun
fichier.

**La mesure** (320 solves, donnes complètes) :

| | nœuds/solve | ratio |
|---|---|---|
| fenêtre pleine `[0,252]` | 466 966 | 1,00× |
| **une** sonde à fenêtre nulle à la vraie valeur | 197 954 | **0,42×** |
| recherche binaire complète (7,3 sondes) | 905 584 | **1,94×** |

**Pourquoi ça ne peut pas marcher ici.** Une sonde à fenêtre nulle n'est que **2,4× moins chère**
qu'une recherche pleine, alors qu'il en faut **7,3**. Et la raison est structurelle : au bridge
la valeur est un **nombre de plis (0-13)**, donc trois ou quatre sondes suffisent. Ici c'est un
**total de points (0-252)**. L'intervalle est vingt fois plus large, la dichotomie vingt fois
plus longue, et le gain par sonde n'est pas vingt fois plus grand. **Ce n'est pas un défaut
d'implémentation : c'est le domaine.** Ne pas réessayer sans changer d'objectif de recherche.

**Réplication** : `solve_for_trump_windowed(hands, dealer, trump, &mut tt, v-1, v)` pour la
sonde, et une dichotomie exploitant le fail-soft (le résultat hors fenêtre est une borne, donc
il resserre l'intervalle plus vite qu'un pas de 1). Compter les nœuds avec `solver_stats`.

### 2.3 Fenêtre étroite amorcée par la moyenne des mondes — **non, 1,04× au mieux**

Déjà mesuré et documenté dans [bid_v7_plan.md](../bid/bid_v7_plan.md) §1.5, le même jour :
δ=20 → 0,96× ; δ=40/80/120 → 1,03-1,04×. Zéro écart de valeur partout — **la correction du
solve fenêtré tient, c'est sa prémisse qui est fausse.**

La prémisse était que les mondes échantillonnés d'une même main ont des valeurs DD groupées.
Les taux de re-recherche la réfutent directement : **36 % des mondes s'écartent de plus de 40
points de la moyenne courante, 12 % de plus de 80**, sur une échelle 0-252.

**Attention à ne pas confondre avec l'écart *entre cartes racine*, qui lui est petit** (médiane
4 points, 63,5 % des décisions dans une bande de 10). Ce sont deux quantités différentes, et
c'est la première qui gouverne l'amorçage entre mondes. `solve_windowed_reuse_tt` et
`solve_for_trump_windowed` restent dans le code sans aucun appelant en production.

Et §2.3bis ci-dessous ferme la question au-delà de cet amorçage-là : **aucun** amorçage, si
parfait soit-il, ne vaut la peine.

### 2.3bis L'oracle de fenêtre — **la famille entière est bornée, et la borne est basse**

§2.3 a mesuré *un* amorçage et l'a réfuté. Restait l'objection évidente : un meilleur amorçage
aurait peut-être marché. Plutôt que d'en construire un deuxième, on mesure le **plafond** de la
famille — on résout chaque position deux fois, la seconde avec une fenêtre centrée sur la
réponse qu'on vient d'obtenir. Aucune heuristique ne peut battre la réponse elle-même.

`bench_dd oracle --deltas 1,5,10,20,40,80`, corpus figé, 2 120 positions, nœuds exacts.
Fraction de la recherche pleine fenêtre qui **survit** :

| forme | nœuds/pos | ±1 | ±5 | ±10 | ±20 | ±40 | ±80 |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 722 051 | **0,503** | 0,653 | 0,746 | 0,905 | 0,984 | 0,999 |
| mid | 5 623 | 0,665 | 0,767 | 0,850 | 0,919 | 0,982 | 0,999 |
| end | 62 | 0,849 | 0,904 | 0,941 | 0,990 | 1,001 | 1,001 |
| worlds | 30 676 | 0,569 | 0,731 | 0,804 | 0,947 | 0,997 | 0,996 |

**Un amorçage parfait ne fait que 2× sur une donne complète**, et le bénéfice s'évapore vite :
9,5 % à ±20, 1,6 % à ±40, rien à ±80. Sur les finales il n'y a jamais rien à prendre.

Le chiffre de décision n'est pas le gain mais le **seuil de justesse**. Un amorceur précis à ±δ
paie la recherche fenêtrée toujours, et une recherche complète à chaque fois qu'il rate : il ne
s'amortit que s'il encadre la vraie valeur *plus souvent* que la fraction que sa fenêtre laisse
debout. Soit, sur une donne complète : **90,5 % des amorçages dans ±20, ou 98,4 % dans ±40**.

C'est exactement la colonne à lire contre la dispersion mesurée en §2.3 — **36 % des mondes
s'écartent de plus de 40 points**. Les deux courbes ne se croisent nulle part. La ligne est
close : il n'y a pas de « meilleur amorçage » à chercher, et les deux entrées fenêtrées
peuvent rester sans appelant sans que ce soit un regret.

### 2.4 Un `apply_play` allégé pour le solveur — **non, 0,977×, donc plus lent**

**L'hypothèse.** `GameState` fait 84 octets et le solveur en copie l'intégralité à chaque nœud,
alors qu'il n'en lit jamais ~50 : `trick_history` (32 o), `voids`, `belote`, les champs
d'enchère. Pire, `apply_play` **entretient** ces champs à chaque coup : suivi des coupes,
`check_belote`, écriture de `trick_history` à chaque pli résolu. Tout ça est mort dans le
solveur, qui voit les quatre mains et rend des points cartes.

**L'implémentation testée.** Un `apply_play_dd` identique moins ces trois postes, avec un test
différentiel sur 3 000 donnes jouées jusqu'au bout vérifiant l'égalité de tout ce que la
recherche lit (mains, pli, points, plis gagnés, état terminal).

**La mesure**, A/B alternant deux révisions, minimum sur 3 tours :

| forme | référence | candidat | ratio |
|---|---|---|---|
| donne complète | 34 563,6 µs | 35 395,5 µs | **0,976×** |
| mi-partie | 189,9 µs | 190,0 µs | 0,999× |
| finale | 1,5 µs | 1,5 µs | 1,000× |
| mondes | 1 253,4 µs | 1 266,0 µs | 0,990× |
| **total** | | | **0,977×** |

Compte de nœuds identique, `EXACT MATCH` sur les valeurs. **Retirer de vraies instructions fait
perdre 2,3 %**, et de façon cohérente sur les deux formes à grand arbre.

**Pourquoi, vraisemblablement.** À ~22 ns par nœud le solveur est **limité par la latence
mémoire** de la sonde TT, pas par le débit d'instructions : les écritures supprimées se
faisaient « gratuitement » dans l'ombre du défaut de cache. En les retirant on a surtout changé
la disposition du code et l'inlining, pour un coût net.

**Ce que ça enterre au passage.** C'était la **moitié facile** de l'idée plus large d'un état
solveur compact de ~32 octets. Si supprimer le travail mort coûte 2,3 %, parier que réduire la
copie de 84 à 32 octets rapporte quelque chose, c'est parier contre la mesure. **Ne pas
entreprendre le refactor de l'état compact sans avoir d'abord une raison neuve.**

**Réplication** : `git show` du commit qui l'a introduit puis retiré, ou réécrire un
`apply_play_dd` sautant `voids`, `check_belote` et l'écriture de `trick_history`, brancher les
cinq `apply_play` de `solver.rs` dessus, et lancer `dd_ab_revs.sh HEAD 3`.

### 2.5 `-C target-cpu=native` / `x86-64-v3` — **non, 0 %, et l'instruction qui comptait était déjà là**

**L'hypothèse.** rustc compile par défaut pour la cible `x86-64` de base, dont les seules
extensions sont `fxsr`, `sse`, `sse2` — le jeu d'instructions de 2003. Or le chemin chaud est de
l'itération de bits sur `CardSet = u32` : 12 `trailing_zeros` dans `card.rs`, 8 dans
`solver.rs`. Sans BMI1 ni SSE4.2, ces primitives devaient coûter plusieurs µops là où le CPU
sait faire en une.

**La mesure**, trois binaires bâtis de la **même source** avec trois `RUSTFLAGS`, alternés sur
5 tours, minimum par configuration (`scripts/analysis/dd_ab_flags.sh`) :

| forme | base | `x86-64-v3` | `native` |
|---|---|---|---|
| donne complète | 32 283 µs | 32 272 µs — **1,000×** | 32 688 µs — **0,988×** |
| mi-partie | 169,1 µs | 169,0 µs — 1,001× | 172,0 µs — 0,983× |
| finale | 1,4 µs | 1,4 µs — 1,000× | 1,4 µs — 1,000× |
| mondes | 1 127,5 µs | 1 156,7 µs — 0,975× | 1 138,6 µs — 0,990× |
| **total** | | **0,998×** | **0,986×** |

Nœuds identiques, `EXACT MATCH` sur les valeurs. `v3` est à **0,03 %** de la base sur le
minimum (les deux convergent vers le même plancher) ; `native` est nominalement **plus lent**.

**Vérifier que les drapeaux sont bien arrivés est ici la moitié du travail.** Un « aucune
différence » entre trois binaires identiques ne vaut rien, et `rustc --print cfg` **ignore
`RUSTFLAGS`** (c'est un mécanisme cargo) — un diagnostic bâti dessus dit « 3 features » pour les
trois et ne prouve rien. La preuve est dans le désassemblage :

| | `tzcnt` | `popcnt` | `blsr` | `andn` | `vpxor` |
|---|---|---|---|---|---|
| base | **32** | 0 | 0 | 0 | 48 |
| v3 | **32** | 0 | 20 | 1 | 69 |
| native | **32** | 0 | 21 | 4 | 123 |

**Et c'est cette ligne qui explique tout : `tzcnt` est déjà présent 32 fois en baseline.** LLVM
émet l'encodage préfixé `F3 0F BC`, que les CPU sans BMI1 décodent comme un simple `bsf` — donc
la primitive dominante recevait déjà le meilleur encodage possible **sans le drapeau**.
`popcnt` n'apparaît nulle part, même en `native` : ces trois sites d'appel sont froids ou
repliés à la compilation. Ce que `v3`/`native` ajoutent réellement — `blsr`, `andn`, plus
d'AVX — est réel mais tombe **à côté du chemin critique**, lequel est limité par la latence
mémoire de la sonde TT (~22 ns/nœud). Même cause que §2.4.

**Ce que ça ferme.** L'arbitrage sur les wheels manylinux n'a pas à être ouvert : il n'y a rien
à gagner, donc pas de `.cargo/config.toml`, pas de risque de `SIGILL` chez un utilisateur PyPI
dont le CPU serait plus vieux que le runner CI. **Ne pas rouvrir sans mécanisme neuf.**

**Portée.** Mono-thread uniquement. Le « 2,4× » de `CLAUDE.md` porte sur `gen_pool`, un **autre
binaire en 32 threads**, et **groupe `native` avec le LTO fat** sans répartition — il n'est ni
confirmé ni infirmé ici. Mais le mécanisme trouvé (`tzcnt` déjà émis) ne dépend pas du nombre de
threads, donc l'attente pour ce cas-là est également basse.

**Réplication** : `scripts/analysis/dd_ab_flags.sh 5`. Le script vérifie aussi l'égalité des
comptes de nœuds — même source, donc toute divergence dénoncerait le harnais, pas le drapeau.

### 2.6 Dédupliquer les coups racine équivalents — **non, seulement 5,2 % de gain possible**

`solve_with_scores` cherche **tous** les coups racine à fenêtre pleine, sans appliquer
`reduce_equivalent` (à dessein : ses consommateurs ont besoin d'une valeur exacte par carte).
Deux cartes DD-équivalentes ont par définition la même valeur, donc on pourrait n'en résoudre
qu'une.

**Mesuré sur 5 991 points de décision** : 21 640 coups légaux pour 20 508 après réduction, soit
**5,2 % de recherches racine redondantes**. Le plafond est trop bas pour justifier le risque.

### 2.7 Le défaut suspecté dans `reduce_equivalent` — **ce n'en est pas un**

Signalé pendant la reconnaissance, et le mécanisme est réel : `apply_play` fixe `played_cards`
immédiatement alors que `current_trick` n'est vidé que dans `resolve_trick`, donc
`outstanding = ALL & !played & !hand` traite **une carte posée sur le pli en cours comme
absente**. Un 8 sur la table ne bloque donc pas la fusion 7/9, alors qu'il est un concurrent
vivant pour ce pli précis — et la fusion supprime le 7, c'est-à-dire l'option de *ne pas*
prendre le pli.

**Mesuré.** La forme exacte — je tiens le 7 et le 9 d'une couleur, pas le 8, le 8 est sur le pli,
et le 9 prendrait le pli là où le 7 le laisserait — se présente **443 fois sur 398 184
décisions (0,11 %)**. Sur **165** de ces positions résolues à ≤ 20 cartes restantes et comparées
à une référence **sans aucune réduction** : **0 écart de valeur**.

**Verdict** : majorant à 95 % de ~1,8 % d'erreur *sachant la forme*, donc ~0,002 % des décisions,
et non testé au-delà de 20 cartes restantes. **Ce n'est pas un défaut de données.** Un correctif
conservateur (faire bloquer la fusion par les cartes du pli en cours) ne change lui non plus
aucune valeur ; il n'a pas été retenu pour ne pas payer une réduction plus faible contre un
risque non observé.

**Réplication** : ajouter un basculement à `reduce_equivalent` (mode 0 = actuel, 1 = les cartes
du pli en cours bloquent aussi, 2 = aucune réduction), et comparer les sorties de
`solve_with_scores` sur les positions où la forme se présente. Le mode 2 est la référence : il
ne peut pas se tromper.

### 2.8 Le décodage `ns == 0` de `solve()` — hasard théorique, jamais observé

`solve` décode le score E-O par `if ns == 252 || ns == 0 { 252 - ns } else { 162 - ns }`,
c'est-à-dire qu'il suppose qu'un score N-S nul implique un capot adverse. C'est faux en théorie :
`resolve_trick` crédite le capot sur `tricks_won == 8`, pas sur les points, et il existe **11
cartes à 0 point** par donne — un camp peut donc gagner un pli sans marquer.

Plus profond : quand `ns == 0`, minimiser N-S est **indifférent** entre « capot » et « laisser
N-S voler un pli sans valeur », alors que le score E-O diffère de 90 points. **La valeur E-O
n'est donc pas déterminée** par une recherche qui ne maximise que N-S.

**Mesuré** : sur 1 600 solves, **108 cas `ns == 0`, tous des capots E-O réels** (0 pli gagné par
N-S en rejouant la ligne optimale). 0 occurrence du cas pathologique.

**Verdict** : garde-fou à poser un jour, pas un bug à corriger en urgence. À noter que le
contrôle « ns + ew ∈ {162, 252} » de `dd_bench` **ne peut pas l'attraper** : il dérive `ew` de
`ns` avec la formule qu'il teste ensuite, donc il est vrai par construction.

---

## 3. Le folklore d'ordonnancement — mesuré, et il avait raison

Trois chiffres traînaient dans les notes historiques sans corpus ni machine : PVS ~+37 %, coups
tueurs ~+38 %, historique ~+16 %. Ils ont été vérifiés, et l'intérêt n'est pas seulement de
savoir s'ils tiennent : **c'est la seule mesure qui dise si la queue est un échec
d'ordonnancement ou une difficulté intrinsèque** (§4).

Feature `solver_ablation` (compilée hors du binaire par défaut) : trois interrupteurs
d'environnement qui éteignent une heuristique **à l'exécution**, pour que les configurations
partagent un binaire, un corpus et un codegen. Nœuds par position sur donne complète :

| config | nœuds/pos | part de la recherche que l'heuristique retire | folklore |
|---|---:|---:|---:|
| référence | 1 448 045 | — | — |
| sans PVS | 2 155 637 | **32,8 %** | 37 % |
| sans coups tueurs | 2 347 754 | **38,3 %** | 38 % |
| sans historique | 1 704 864 | **15,1 %** | 16 % |
| sans les trois | 5 075 486 | **71,5 %** | — |

Les coups tueurs et l'historique tombent **au point près** ; PVS est quelques points en dessous
de sa réputation. Un folklore qui se vérifie à ce niveau ne vient pas de nulle part : ces
chiffres ont bien été mesurés un jour, ils n'ont simplement jamais été écrits avec leur harnais.

**Les trois sont sur-additifs** : ensemble ils valent 3,50×, alors que le produit de leurs
contributions individuelles ne prédit que 2,84×. Ils ne se recouvrent pas, ils se complètent —
un coup tueur ne sert que si la fenêtre est déjà serrée, et PVS ne serre la fenêtre que si le
premier coup est bon.

**Ce que ça dit de la queue** : l'ordonnancement en place retire déjà 71,5 % de l'arbre, et ce
n'est pas un mécanisme fatigué. La lecture honnête est que ça **ne tranche pas** §4.2 — ça
établit que le levier est vivant, pas qu'il reste dedans un facteur 20. La mesure qui
trancherait est différente : comparer, sur les seules positions de la queue, l'arbre réel à
l'arbre d'un ordonnancement oracle (meilleur coup en premier à chaque nœud, lu d'un premier
solve). C'est le pendant exact de l'oracle de fenêtre du §2.3bis, et c'est la prochaine à faire.

Restent non mesurés, et à ne pas citer comme acquis :
- TT à deux niveaux : « surcoût de sonde »
- ordonnancement enrichi par le maître du pli partiel : « surcoût > bénéfice »

**Réplication** : `scripts/analysis/dd_ablation.sh`. Le script finit par une porte
d'exactitude — une ablation change l'*ordre* de la recherche, jamais la réponse, donc les
quatre configurations doivent rendre `EXACT MATCH` contre la référence. Elles le font.

---

## 4. Où il reste du temps à prendre

**La queue, et à peu près rien d'autre.** Les 10 % de solves les plus durs portent 40 % des
nœuds ; la médiane fait 317 k nœuds quand le pire fait 6,0 M. Un solveur qui n'explose pas sur
ces donnes-là vaudrait plus que toutes les micro-optimisations réunies — et le profil dit que
les micro-optimisations, elles, sont épuisées.

**Et l'oracle a été exécuté : la queue est bien un échec d'ordonnancement.** §5.

**Les trois familles sont désormais bornées, et une seule a rendu quelque chose.** Le tableau
qui résume la campagne :

| famille | plafond mesuré | issue |
|---|---:|---|
| amorçage **entre mondes** (§2.3bis) | 2,0× avec un seed **exact** | fermée — il faudrait 98,4 % de justesse à ±40 |
| amorçage **entre cartes racine** (§8) | — | **pris : 0,904× de plus, et 0,914× sur les mondes** |
| ordonnancement des coups (§5) | **6,0×**, ~8× sur la queue | **partiellement pris : l'IID (§6)** |
| bornes / évaluation (§7) | 1,09× avec une évaluation **parfaite** | fermée |

**Total livré : 0,805× au chronomètre sur `solve_with_scores`, 0,728× sur le vrai `gen_pool`.**

Ce qui reste vient donc entièrement de la deuxième ligne, et la suite est au §6 « ce qui reste
sur la table » : un horizon moins naïf, une fenêtre plus large payée moins cher. Pas de règle de
contrée à écrire — §5 a montré que les échecs d'ordre sont entre cartes de même nature.

Et **trois pistes qui figuraient ici et n'y sont plus** :

- `-C target-cpu=native`. Il n'existe toujours aucun `.cargo/config.toml`, donc tout le dépôt
  compile pour la cible x86-64 de base — mais c'est une constatation sans conséquence, pas une
  piste. Mesuré à 0 %, cause identifiée, §2.5.
- **`bench_tt_size` en 32 threads**, qui était en tête de cette liste. Exécuté : la constante
  est confirmée et le compromis est plat autour d'elle, §2.1bis.
- **L'amorçage de fenêtre**, sous toutes ses formes. Le plafond de la famille entière est
  mesuré et il est bas, §2.3bis.

---

## 5. L'oracle d'ordonnancement — **la queue est bien un échec d'ordre, et le plafond est haut**

C'est le premier résultat de la campagne qui ouvre quelque chose au lieu de le fermer.

L'hypothèse tenait depuis le début sans jamais être testée : un arbre 20× plus gros que la
médiane serait la signature d'un mauvais ordre plutôt que d'une donne difficile. §3 ne la
testait pas — savoir que les heuristiques génériques retirent 71,5 % dit que le levier
fonctionne, pas qu'il reste quelque chose dedans.

Même construction qu'au §2.3bis, appliquée à l'ordre : une première recherche **enregistre le
meilleur coup à chaque nœud** dans une table **sans éviction** — c'est ce qui en fait un oracle
et non une TT préchauffée, la vraie table tournant à 99,4 % de remplissage et perdant presque
tout. Une seconde recherche rejoue la position avec ces coups placés en tête. Une troisième,
qui enregistre encore, montre que le chiffre a convergé. Aucune règle, si bonne soit-elle en
contrée, ne peut battre le fait de rejouer la réponse.

`bench_dd ordering`, 2 120 positions, unité = **un solve racine** (pas `solve_with_scores` : ne
pas comparer ces nœuds à ceux du §3). Fraction de la recherche qui survit :

| forme | nœuds/pos | positions enregistrées | oracle | itéré |
|---|---:|---:|---:|---:|
| full | 722 051 | 228 769 | **0,214** | 0,212 |
| mid | 5 623 | 1 801 | 0,418 | 0,417 |
| end | 62 | 15 | 0,748 | 0,748 |
| worlds | 30 676 | 9 708 | **0,251** | 0,251 |

Et la découpe par difficulté, qui est le vrai livrable — **la queue s'améliore nettement plus
que la médiane** :

| donne complète | n | nœuds/pos | part des nœuds | oracle |
|---|---:|---:|---:|---:|
| 10 % les plus durs | 80 | 3 111 143 | **43,1 %** | **0,173** |
| 15 % suivants | 120 | 1 150 711 | 23,9 % | 0,218 |
| 50 % du milieu | 400 | 430 698 | 29,8 % | 0,259 |
| 25 % les plus faciles | 200 | 91 923 | 3,2 % | 0,316 |

| mondes (l'unité d'IS-DD) | n | nœuds/pos | part des nœuds | oracle |
|---|---:|---:|---:|---:|
| 10 % les plus durs | 72 | 225 715 | **73,6 %** | **0,218** |
| 15 % suivants | 108 | 40 632 | 19,9 % | 0,311 |
| 50 % du milieu | 360 | 3 997 | 6,5 % | 0,441 |
| 25 % les plus faciles | 180 | 46 | 0,0 % | 0,816 |

La concentration de la queue est re-dérivée au passage et tient : **43,1 %** des nœuds sur donne
complète, et **73,6 %** sur les mondes — où le déséquilibre est bien plus violent que le « 40 % »
qu'on citait. Les finales, elles, n'ont rien à donner (0,748) : elles sont trop petites pour que
l'ordre pèse.

### Le plafond est mesuré *par en dessous*

Le 0,214 sous-estime. La passe 2 visite des nœuds que la passe 1 n'a jamais atteints, et
l'oracle y est muet : **plus la passe 1 explore, meilleur est l'oracle**. Une exécution avec
coups tueurs et historique éteints (PVS gardé — c'est une technique de recherche, pas d'ordre)
explore 1 342 408 nœuds au lieu de 722 051, enregistre **322 869** positions au lieu de 228 769,
et tombe à **0,089** — soit 119 474 nœuds en absolu, contre 154 519 par l'autre chemin. Sur le
décile le plus dur elle descend à **0,050**, soit 399 000 nœuds là où l'ordre actuel en dépense
3,11 M.

En unité commune, donc :

| | donne complète | décile le plus dur |
|---|---:|---:|
| ordre statique seul (ni tueurs ni historique) | 1 342 408 | 7 979 782 |
| **ordre actuel** | **722 051** | **3 111 143** |
| ordre parfait | ≤ 119 474 | ≤ 398 989 |
| **plafond restant** | **≥ 6,0×** | **≥ 7,8×** |

Les heuristiques génériques ont donc pris à peu près la moitié du chemin, et il en reste six
fois plus que ce qu'elles ont pris. C'est la première fois qu'un chiffre de cette campagne
justifie d'écrire du code.

### Ce que ça n'autorise pas à dire

**C'est un plafond, pas un gain.** L'oracle lit une recherche déjà terminée ; une règle n'aura
que la position. Le §2.3bis est là pour rappeler ce que valent les plafonds : celui de
l'amorçage était à 2× et la famille est morte quand même, parce qu'aucune estimation réelle
n'approchait la justesse requise. La question suivante n'est donc pas « quelle règle écrire »
mais **« ce que l'oracle choisit est-il prédictible depuis la position ? »** — dumper les coups
enregistrés avec les traits de leur position et regarder ce qu'une règle simple en récupère.
Le harnais pour ça est déjà là : c'est la table de `solver_oracle`.

Deuxième réserve, structurelle : le gain porte là où les nœuds sont, donc sur `full` et sur les
mondes durs — c'est-à-dire sur `gen_pool` et sur IS-DD à l'entame. Le web (mi-donne et finales)
n'en verra presque rien.

**Réplication** :
```bash
cargo build --release --features "parallel solver_stats solver_oracle" --bin bench_dd
./target/release/bench_dd ordering --corpus data/analysis/dd_corpus_v1.bin --threads 8
```
La table est **par thread et sans borne** (~650 Mo de pic résident à 8 threads) et n'est unique
qu'à l'intérieur d'un couple (donne, atout) — `position_hash` dérive les mains des cartes jouées
et ne clé pas sur l'atout. `cmd_ordering` la vide entre deux positions ; ne pas le faire
fabriquerait un oracle qui connaît une autre partie. Porte d'exactitude intégrée : les trois
passes doivent rendre la même valeur, l'ordre ne change jamais la réponse.

---

## 6. L'IID d'ordonnancement — **implémenté, 0,90× en nœuds et 0,92× au chrono**

Le premier gain livré de la campagne, et il tombe directement du §5.

**Le raisonnement, en trois mesures.** L'oracle dit qu'un ordre parfait vaut 6× (§5). La table
de confusion dit qu'aucune règle statique ne l'atteindra : ~70 % des échecs ont le bon coup et
le coup essayé **dans la même catégorie** — la plus grosse cellule est defausse→defausse, 27 %
tôt et 50 % tard. Et les quatre raffinements écrits à la main du score statique le confirment :
0,991× à 1,030×, autrement dit rien. La raison est structurelle — le score statique n'est
consulté **qu'après** le coup de TT et les coups tueurs, qui portent déjà 38,3 % et 15,1 % : il
ne lui reste presque pas de levier.

La troisième mesure dit où aller. En restreignant l'indice parfait à une fenêtre de profondeur :

| donne complète | racine seule | + ses enfants | 1ᵉʳ pli | 2 plis | 4 plis |
|---|---:|---:|---:|---:|---:|
| fraction restante | **0,705** | 0,635 | 0,541 | 0,388 | 0,269 |

**Ordonner la racine seule vaut 1,42×.** Un nœud, une décision. Ce qui sépare deux cartes de
même nature, ce n'est pas une règle, c'est **regarder** — donc au sommet, quand la table n'a
rien à proposer, on lance une recherche courte et on prend sa réponse comme premier coup.

### Pourquoi ce n'est pas un second `quick_tricks`

**La valeur de la recherche courte ne sort jamais de l'ordonnancement.** À l'horizon elle rend
les points déjà ramassés et s'arrête — aucune estimation du reste, aucune affirmation sur
l'avenir. `quick_tricks` était un défaut pour la raison inverse : son approximation atteignait
une **valeur rendue**. Un ordre peut être arbitrairement mauvais et ne coûter que du temps.
C'est la porte `diff` qui prouve que la distinction a tenu, et elle a tenu à chaque variante.

### Ce que ça donne

| forme | avant | après | |
|---|---:|---:|---:|
| full | 1 448 045 | 1 287 334 | **0,889×** |
| worlds | 55 862 | 53 609 | **0,960×** |
| mid | 9 061 | 9 061 | 1,000× |
| end | 89 | 89 | 1,000× |
| **ALL** | **566 953** | **505 542** | **0,892×** |

Au chronomètre, A/B **entrelacé** a 8 threads, minimum sur 8 tours : **0,913×** (reference
6,77-7,07 s, IID 6,18-6,53 s).

Deux details valent la moitie du gain. Le regard rend une valeur pour **chaque** coup de la
position, pas seulement pour le meilleur ; la premiere version n'en gardait qu'un et jetait le
reste, alors que classer les huit ne coute qu'un tri (0,895 -> 0,892). Et l'horizon **credite le
pli en cours** a qui le prend : sans ca un regard de 6 plis s'arrete au milieu d'un pli et note
pareil la ligne qui vient d'emporter 20 points et celle qui les a donnes (0,919 -> 0,904 a 4
plis de regard).

### La garde n'est pas un réglage, c'est la différence entre un gain et un désastre

Sans la garde sur les cartes restantes, mesuré : les **finales deviennent 3,8× plus lentes** et
la mi-donne 1,56×. Sur une position qui explore 89 nœuds, un regard à 6 plis est plus gros que
la recherche entière qu'il prétend aider. Seules les donnes complètes ont un arbre assez
profond pour rembourser — et ce sont elles qui portent les nœuds, donc la garde ne coûte rien
et supprime toute régression. **C'est le corpus à quatre formes qui a attrapé ça** : une
mesure agrégée aurait montré 0,902× et caché un facteur 3,8 sur les positions exactes où
`/analyse/jeu` et `agent_review` passent leur temps.

### Deux leçons de méthode, tirées du réglage lui-même

**1. Le compte de nœuds cesse d'être un bon substitut du temps dès qu'on change la *nature* des
nœuds.** Une variante plus profonde et plus large (8/8 avec calendrier décroissant) fait **1,3
point de moins en nœuds totaux** — surcharge comprise, facturée au prix fort — et pourtant elle
**égalise au chronomètre** (0,917× contre 0,913×). Un nœud de regard n'est pas interchangeable
avec un nœud de recherche. La règle du dépôt « les nœuds d'abord » vaut pour comparer deux
recherches de même nature ; ici il a fallu trancher au chrono.

**2. L'agrégat du corpus aurait choisi les mauvaises constantes.** Il est dominé par les donnes
complètes (1 158 M nœuds sur 1 202 M), alors que les heures DD réelles du projet sont surtout
dans les **mondes** (~2 800 core-h pour une couche de scores contre ~180 pour `gen_pool`).
Optimiser le total désigne 8/8/calendrier et une garde à 28 — deux choix qui **abandonnent tout
le gain sur les mondes** :

| config | full | worlds | mid | end |
|---|---:|---:|---:|---:|
| 8/8/calendrier, garde 28 | 0,869 | **1,000** | 1,00 | 1,00 |
| 8/8/calendrier, garde 24 | 0,880 | 0,969 | 1,00 | 1,00 |
| **6/4/plat, garde 24** | 0,889 | **0,960** | 1,00 | 1,00 |
| 8/8/calendrier, garde 22 | 0,880 | 0,968 | **1,07** | 1,00 |

**Toujours choisir par forme, jamais sur le total** — et la garde à 22, qui semble gagner sur
les mondes, fait régresser la mi-donne de 7 %, c'est-à-dire le chemin d'analyse du web.

### Les constantes, et leur plateau

`IID_DEPTH = 6`, `IID_TOP = 4`, `IID_MIN_CARDS = 24`, `IID_EVAL = 1`, `IID_SCHED = 0`. Chacune
est au milieu d'un plateau, pas sur une pointe — c'est ce qui les rend sûres :

| profondeur | 4 | 5 | **6** | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|
| nœuds | 0,919 | 0,911 | **0,898** | 0,897 | 0,910 | 0,938 |

| fenêtre | 3 | **4** | 5 | 6 | 8 | 12 |
|---|---:|---:|---:|---:|---:|---:|
| nœuds | 0,928 | **0,902** | 0,909 | 0,935 | 1,085 | 2,312 |

| cartes min | 30 | 28 | 26 | **24** | 20 | 0 |
|---|---:|---:|---:|---:|---:|---:|
| nœuds | 0,927 | 0,899 | 0,899 | **0,898** | 0,899 | 0,902 |

La fenêtre est le paramètre dangereux : à 12 plis l'IID coûte **2,3× plus qu'il ne rapporte**.

**Réplication** :
```bash
cargo build --release --features "parallel solver_stats solver_ablation" --bin bench_dd
COLVER_DD_IID_DEPTH=0 ./target/release/bench_dd run --corpus data/analysis/dd_corpus_v1.bin \
    --threads 32 --values off.vals              # l'éteindre
./target/release/bench_dd run --corpus data/analysis/dd_corpus_v1.bin \
    --threads 32 --values on.vals               # le défaut
./target/release/bench_dd diff --a off.vals --b on.vals    # doit dire EXACT MATCH
```

### Validé sur le vrai binaire — et le gain dépend du point d'entrée

Le corpus est un substitut. `gen_pool`, lui, est la charge réelle : 10 000 donnes, 5 tours
**entrelacés**, minimum — **46,7 s → 34,0 s, soit 0,728×**, et les cinq tours vont tous dans le
même sens. C'est nettement plus que ce que le corpus annonçait, et la raison est structurelle :

| unité | full | worlds | mid | end |
|---|---:|---:|---:|---:|
| `solve_for_trump` — un solve racine (`gen_pool`) | **0,743** | 0,811 | 0,930 | 1,000 |
| `solve_with_scores` — le tableau par carte (IS-DD) | 0,889 | 0,960 | 1,000 | 1,000 |

**`solve_with_scores` donne à chaque coup racine une fenêtre pleine `[0, 252]`.** Les frères ne
partagent donc rien, et le levier d'un bon coup racine se dilue sur huit sous-recherches
indépendantes. Un solve unique, lui, profite entièrement de la fenêtre serrée qu'établit le
premier coup s'il est le bon. C'est la même asymétrie que mesure la fenêtre de profondeur du
§5 — et elle veut dire qu'**annoncer un seul chiffre pour « le gain de l'IID » serait faux** :
`gen_pool` gagne 1,35×, une couche de scores IS-DD 1,12×.

### Ce qui reste sur la table

Le plafond de l'ordonnancement peut se recompter maintenant que l'IID est en place, et le calcul
porte son propre contrôle : **l'arbre parfaitement ordonné ne dépend pas du chemin par lequel on
y arrive**. Avant l'IID, 722 051 × 0,214 = **154 519** nœuds ; après, 536 250 × 0,285 =
**152 831**. Les deux plancher s'accordent, ce qui valide la construction de l'oracle.

| | avant IID | avec IID | plancher | pris | reste |
|---|---:|---:|---:|---:|---:|
| full | 722 051 | 536 250 | ~153 000 | **33 %** | **3,5×** |
| worlds | 30 676 | 24 883 | ~7 300 | **25 %** | **3,4×** |

**L'IID a pris le tiers facile ; il reste un facteur 3,5.** Et §5 dit exactement pourquoi c'est
dur : le reste est de la discrimination *à l'intérieur d'une catégorie*, profond dans l'arbre,
là où un regard ne peut plus être payé. Le plafond du §5 était 6× ; l'IID en prend **1,35×**. Il ne touche que la racine et ses trois
plis, avec un regard qui ne voit qu'un pli et demi. Les pistes suivantes, dans l'ordre :

1. **Un regard qui voit plus loin sans coûter plus cher.** L'horizon actuel rend les points
   ramassés, ce qui est le signal le plus pauvre possible. Une évaluation d'horizon un peu
   moins naïve permettrait d'aller moins profond pour la même qualité d'ordre — et la courbe
   dit que la profondeur est chère (0,938× à 9 plis).
2. **Étendre la fenêtre en la payant moins.** À 8 plis de fenêtre le coût explose (1,085×)
   parce que l'IID tourne à chaque nœud. Ne le lancer qu'aux nœuds dont le sous-arbre s'annonce
   gros — qu'il faudrait savoir prédire, ce qui est la même question un cran plus loin.
3. **Un réseau de politique au sommet reste hors de portée**, et le calcul est simple : un
   solve de donne complète coûte ~15,9 ms, la racine parfaite en rend 4,7 ms, et DouDou50 coûte
   ~1 ms par évaluation — mais il faudrait qu'il corrige une bonne part des **15,3 %** de
   racines que l'ordre actuel rate déjà, alors qu'il prédit du bon jeu à information
   incomplète, pas le coup DD. L'IID fait le même travail pour ~0,3 % du budget de nœuds.

---

## 7. Le troisième oracle — les bornes, **et la famille est fermée aussi**

La borne haute du solveur suppose que N-S ramasse **tout ce qui reste dans la donne**, la basse
qu'il ne ramasse plus rien. C'est aussi lâche qu'une borne peut l'être, et §4 en faisait la
dernière grande piste ouverte. C'est aussi exactement ce que `quick_tricks` tentait de resserrer
avant de rendre des valeurs fausses — raison de plus pour borner la question avant que
quelqu'un ne recommence.

Même construction qu'au §2.3bis et au §5. Une passe enregistre la vraie valeur de chaque nœud
que la recherche a résolu **exactement**, une seconde s'en sert comme borne à ± slack près.
Slack 0 = l'évaluation parfaite.

| forme | nœuds/pos | ±0 | ±2 | ±5 | ±10 | ±20 | ±40 |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 554 108 | **0,917** | 0,915 | 0,937 | 0,966 | 0,996 | 0,995 |
| mid | 5 204 | 0,941 | 0,941 | 0,950 | 0,993 | 0,985 | 0,988 |
| end | 62 | 0,962 | 0,971 | 0,971 | 0,982 | 0,981 | 1,000 |
| worlds | 26 576 | 0,915 | 0,907 | 0,923 | 0,952 | 0,977 | 0,995 |

**Une évaluation parfaite ne rend que 8,3 %.** À ±10 il reste 3,4 %, à ±20 plus rien. Aucune
fonction d'évaluation, si bonne et si gratuite soit-elle, ne vaut la peine d'être écrite ici.

**Pourquoi.** L'alpha-bêta a déjà resserré la fenêtre quand on arrive au nœud. La borne grossière
ne se déclenche que dans les cas extrêmes — là où le camp mène déjà tellement que même « il
prend tout le reste » ne suffit pas — et resserrer une borne qui ne coupait presque jamais ne
coupe presque jamais plus.

### La porte a d'abord attrapé une vraie erreur, et elle est instructive

Le premier montage stockait la valeur **absolue** du nœud, et a échoué `EXACT MATCH` dès la
première position. Cause : `position_hash` porte `played_cards`, qui est un **ensemble** — deux
positions ayant les mêmes cartes jouées mais un partage de plis différent se retrouvent sur le
même hachage, et **seul l'avenir leur est commun**. C'est précisément pour ça que la TT stocke
un `future_score` relatif à `ns_base` et non un score absolu.

Ce détail passait pour du style. C'est un invariant, et une carte auxiliaire indexée sur le même
hachage doit le respecter. La leçon générale : **toute structure clée sur `position_hash` ne
peut porter que des quantités relatives aux points déjà faits.**

**Réplication** :
```bash
cargo build --release --features "parallel solver_stats solver_oracle" --bin bench_dd
./target/release/bench_dd bounds --corpus data/analysis/dd_corpus_v1.bin \
    --threads 8 --deltas 0,2,5,10,20,40
```

---

## 8. La fenêtre entre cartes racine — **le seul amorçage qui paie**, et il paie où sont les heures

§2.3bis avait fermé la famille des amorçages de fenêtre. Elle reste fermée, et pourtant celui-ci
marche : parce que ce n'est pas la même quantité qu'on amorce.

**§2.3 le disait déjà, en note d'avertissement.** L'écart qui avait tué l'amorçage était celui
**entre mondes échantillonnés** d'une même main — 36 % s'écartent de plus de 40 points. L'écart
**entre cartes racine d'une même position** est une autre quantité, et il est petit : **médiane
4 points, 63,5 % des décisions dans une bande de 10**. Cette note était là depuis le début ; ce
qui manquait, c'était de remarquer qu'un endroit du code jetait exactement cette information.

Car `solve_with_scores` donnait à **chaque** carte racine une fenêtre pleine `[0, 252]` — c'est
ce que le §6 a fini par identifier comme la raison pour laquelle l'IID y rend moins que sur un
solve unique. Chaque carte repart de zéro alors que la précédente vient de dire, à quatre points
près, où se situe la réponse.

La carte *i* est donc cherchée dans `[v(i-1) − 8, v(i-1) + 8]`, et **re-cherchée en fenêtre
pleine si le résultat sort de la fenêtre** — fail-soft : dedans c'est une valeur, dehors c'est
une borne. Prendre la borne pour la valeur est précisément le défaut de `quick_tricks` ; la
re-recherche est le prix de ne pas le commettre, et la porte `diff` prouve qu'il a été payé.

### Ce que ça donne

| forme | avant campagne | + IID (§6) | + fenêtre | total |
|---|---:|---:|---:|---:|
| full | 1 448 045 | 1 287 334 | **1 163 727** | **0,804×** |
| worlds | 55 862 | 53 609 | **48 994** | **0,877×** |
| mid | 9 061 | 9 061 | 8 707 | 0,961× |
| end | 89 | 89 | 88 | 0,989× |

Au chronomètre, trois configurations **entrelacées** à 8 threads, minimum sur 8 tours :
**rien 4,97 s → IID 4,65 s (0,936×) → IID + fenêtre 4,00 s (0,805×)**. La dispersion par tour
est de 3 %, la plus propre de la campagne.

**C'est la seule optimisation qui aide `worlds` franchement** (0,877×), donc la seule qui morde
vraiment sur les ~2 800 core-h d'une couche de scores IS-DD. La largeur 8 est au fond d'un bassin
plat (0,905× contre 0,910-0,912 de 5 à 10, 0,954× à ±20) — élargir et la fenêtre cesse d'élaguer,
resserrer et chaque carte paie une re-recherche.

### Deux essais adjacents, tous deux rejetés — et par des portes différentes

**Ordonner aussi les cartes racine par le regard.** La fenêtre s'amorce sur la carte
précédente, donc l'ordre décide de son taux de réussite : un bon ordre groupe les cartes de
valeur voisine. Mesuré : **1,1 % de nœuds en moins et 3,1 % de temps en plus**, les 8 tours
entrelacés d'accord. `ROOT_IID = false`, gardé derrière l'interrupteur parce que la mesure vaut
mieux que le code effacé. **Troisième fois de la campagne que le compte de nœuds pointe à
l'envers** — il est exact, mais il cesse d'être un substitut fidèle du temps dès qu'un
changement modifie la *nature* des nœuds.

**Et sa première version était fausse, pas seulement lente.** Elle rangeait l'ensemble
**réduit** — un représentant par classe d'équivalence — alors que `solve_with_scores` doit une
valeur à **chaque carte légale**. Elle en laissait donc tomber, et paraissait 3 % plus rapide
pour cette raison exacte. C'est le même piège que `legal_actions_reduced` dans `/analyse/jeu`,
et seule la porte `diff` l'a vu. `shallow_rank_moves` prend désormais l'ensemble à ranger en
argument, avec le contrat écrit à côté.

**Effet de bord assumé : le départage des ex æquo est devenu explicite.** En réordonnant la
racine, `solve_with_scores` et `solve_best_card` ont cessé de désigner la même carte — deux
cartes **également optimales**, mais la réponse dépendait d'un détail interne de la recherche.
Les deux départagent maintenant par **indice de carte le plus bas**, ce qui rend la « meilleure
carte » fonction de la position et non du parcours. Changement de comportement possible sur les
positions à égalité ; **inerte sur les 2 120 positions du corpus** (`best card moved: 0`).

### La leçon, qui vaut au-delà d'ici

Une famille fermée l'est **pour la quantité mesurée**, pas pour son nom. « L'amorçage de fenêtre
ne marche pas » était vrai de l'écart entre mondes et faux de l'écart entre cartes. Les deux
chiffres étaient dans le même document depuis le début, l'un sous l'autre, avec un avertissement
explicite de ne pas les confondre — et la conclusion a quand même été généralisée d'un cran de
trop. **Avant de classer une piste comme fermée, vérifier de quelle quantité parle la mesure qui
l'a fermée.**
