# Optimisation du solveur DD — journal de campagne

**Campagne du 2026-08-02.** Ce document existe pour une seule raison : **ne pas réexplorer**.
Il enregistre autant ce qui a marché que ce qui n'a pas marché, et pour chaque échec il donne
la mesure, la cause et la commande qui le rejoue. Une piste fermée sans trace est une piste
qu'on rouvrira dans six mois.

Référence courante du solveur : [dd_solver.md](dd_solver.md). Ici, c'est l'historique du
raisonnement.

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

### 1.4 Le harnais — `bench_dd`, et la discipline de mesure

`colver-core/src/bin/bench_dd.rs`, trois sous-commandes :

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

# A/B alternant deux révisions git, minimum sur N tours
scripts/analysis/dd_ab_revs.sh <rev-de-référence> 3

# et la version journalisée, qui écrit dans docs/measurements/index.jsonl
python3 scripts/analysis/dd_solver_bench.py --tag <nom> --repeats 5 --note "..."
```

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

**Ce qui reste ouvert** : ce balayage est **mono-thread**. En 32 threads, 32 × 2 Mo = 64 Mo pour
36 Mo de L3 sur ce 13900K ; `1<<16` donnerait 16 Mo. `bench_tt_size` existe précisément pour ça
et **n'a jamais été exécuté**. C'est le levier ouvert le moins cher du dépôt.

**Réplication** : passer une tranche de taille arbitraire (puissance de deux) à
`solve_for_trump_reuse_tt` — le solveur masque avec `len()-1`, donc toute taille est légale —
et chronométrer le memset séparément à chaque taille. Aujourd'hui il faut construire un
`TtBuf::with_log2_size(n)`.

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

## 3. Folklore restant — jamais mesuré, à ne pas citer comme acquis

Ces chiffres vivent dans les notes historiques sans corpus, sans machine et sans artefact. Ils
sont plausibles ; ils ne sont **pas** des mesures, et il faut les traiter comme des expériences
candidates plutôt que comme des faits :

- PVS ~+37 %
- coups tueurs + ordonnancement simple ~+38 %
- heuristique d'historique ~+16 %
- TT à deux niveaux : « surcoût de sonde »
- ordonnancement enrichi par le maître du pli partiel : « surcoût > bénéfice »

Le harnais existe maintenant pour trancher chacun d'eux en une demi-heure.

---

## 4. Où il reste du temps à prendre

**La queue, et à peu près rien d'autre.** Les 10 % de solves les plus durs portent 40 % des
nœuds ; la médiane fait 317 k nœuds quand le pire fait 6,0 M. Un solveur qui n'explose pas sur
ces donnes-là vaudrait plus que toutes les micro-optimisations réunies — et le profil dit que
les micro-optimisations, elles, sont épuisées.

Pistes non encore mesurées, par rapport gain/risque décroissant :

1. **`bench_tt_size` en 32 threads** (§2.1). Le bench existe, il n'a jamais tourné, et
   l'hypothèse est concrète : 64 Mo de working set pour 36 Mo de L3.
2. **Ordre des coups sur les positions de la queue.** Un arbre 20× plus gros que la médiane est
   une signature d'échec d'ordonnancement, pas de difficulté intrinsèque.
3. **Bornes plus fines.** La borne haute actuelle (`points + tout le reste + dix de der`) est
   très lâche. Toute borne plus serrée doit être **saine** — c'est exactement là que
   `quick_tricks` s'est planté, et la porte `diff` est là pour ça.
Et **une piste qui figurait ici et n'y est plus** : `-C target-cpu=native`. Il n'existe toujours
aucun `.cargo/config.toml`, donc tout le dépôt compile pour la cible x86-64 de base — mais c'est
désormais une constatation sans conséquence, pas une piste. Mesuré à 0 %, cause identifiée, §2.5.
