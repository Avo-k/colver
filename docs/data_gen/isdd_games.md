# Générer des donnes complètes jouées par IS-DD

`gen_games_isdd` produit des **donnes entières** — enchère réelle, tous les
tours, et les 32 cartes dans l'ordre — jouées par le joueur de référence du
projet (bid v6 + IS-DD sur mondes playgen). Sortie au format `COLVGM01`, celui
que `train_playgen --games` consomme déjà.

C'est ce qui manquait. Les deux générateurs existants jettent la trajectoire :

| binaire | enchère | ce qui est gardé |
|---------|---------|------------------|
| `gen_pool` | aucune (atout imposé, les 4 couleurs) | la valeur DD |
| `enrich_pool_isdd` | aucune (`setup_dd`) | les points N-S finaux |
| **`gen_games_isdd`** | **réelle** | **toutes les actions** |

Les deux premiers produisent une *étiquette* par donne, ce qui suffit à
entraîner un bidder. Entraîner un playgen sur du jeu fort demande la
trajectoire.

```bash
# Un GPU
COLVER_PLAYGEN_GPU_URL=http://gpu-host:8003 \
cargo run --release --features parallel --bin gen_games_isdd -- \
  --deals 100000 --dets 40 --threads 256 --out data/training/isdd_games.bin

# Deux GPU : liste séparée par des virgules, tourniquet global au processus
--url http://localhost:8003,http://moxxi:8003

# Relire un corpus et le rejouer intégralement
cargo run --release --bin gen_games_isdd -- --check data/training/isdd_games.bin
```

`--match-mode` enchaîne les donnes en parties de 2000 points au lieu de les
tirer indépendantes. Ce n'est pas cosmétique : bid v6 lit une observation
*score-aware*, donc il annonce autrement à 900-200 qu'à 0-0, et le corpus
playgen actuel est **entièrement à 0-0**. C'est le manque relevé dans
[playgen v3](../belief/playgen.md).

## Le corpus se vérifie à l'écriture

`GameState::step` **ne valide pas la légalité** — contrat d'un moteur RL. Côté
web ça a laissé six donnes fausses entrer en base ([integrity.py](../../python/colver/web/integrity.py)).
Ici l'enjeu est plus grand : une donne incohérente dans un corpus
d'entraînement n'est pas un incident visible, c'est du gradient sur une
position qui n'existe pas. Chaque donne est donc rejouée avant écriture — toute
action légale à son tour, dernière action terminale — et **rien n'est écrit si
une seule échoue**. `--check` refait la vérification depuis le fichier, ce qui
ferme l'aller-retour écriture → lecture.

Taille : **58 octets par donne**, ~40 actions. 1 M de donnes = 58 Mo, soit
exactement le gabarit de `playgen_games_1M.bin`.

## Où passe le temps

Profil mesuré le 2026-08-04 (sidecar playgen v2_final sur une 3090, client sur
32 cœurs, 40 mondes par décision) :

- **une donne = 28 décisions IS-DD** sur 32 cartes plus l'enchère. **34 %
  d'entre elles ne demandent aucun monde** : coup forcé, ou position résolue
  par les contraintes dures.
- **93 % du temps de thread est de l'attente du sidecar, 7 % du solve DD.**

C'est l'inverse de l'intuition. Le solveur DD est le composant cher *par
appel*, mais il s'effondre avec la donne — 6,8 s cumulées à huit cartes
restantes, 0 s à deux — alors que le coût d'un monde playgen reste du même
ordre à tous les stades. **Le générateur est limité par le GPU, et le CPU est
oisif à plus de 80 %.** Toute l'optimisation est donc partie du sidecar.

Corollaire sur la concurrence : `--threads` doit être **très au-dessus de
`nproc`**. Un thread alterne « dormir en attendant des mondes » et « brûler un
cœur à résoudre » ; un thread par cœur laisse les deux moitiés inoccupées à
tour de rôle. Le débit plafonne vers 256 threads sur 32 cœurs (2,61 / 2,55 /
2,54 donnes/s à 256 / 512 / 768) — au-delà le GPU est saturé et la
sur-souscription n'achète plus rien. `[play] parallel` doit rester **faux** :
il est fait pour la latence d'un coup, et à ce niveau de concurrence le pool
rayon est déjà plein.

## Ce qui a été changé, et ce que ça vaut

Toutes les mesures sont des **A/B alternés** (jamais deux exécutions
séquentielles : la charge dérive de 20 % sur cette machine), 150 donnes par
tour, 3 tours, 40 mondes par décision, 256 threads.

### 1. Préfixe groupé — **1,47×**

`generate_worlds_multi` déroulait le préfixe **un jeton à la fois**, par le
même `forward_step` que le décodage. Or un pas est borné par le *lancement de
noyaux*, pas par l'arithmétique : ~2,1 ms qu'il y ait 1 lane ou 40. Un préfixe
de 40 jetons coûtait donc ~100 ms contre 25-50 ms pour tout le décodage — le
profileur interne (`COLVER_PLAYGEN_PROFILE=1`) donne **65 à 81 % du coût d'une
requête**, et d'autant plus que la donne avance (préfixe long, peu de cartes à
tirer).

C'est pourtant la seule phase du modèle sans dépendance séquentielle : les
jetons sont tous connus d'avance. `forward_prefill` les passe en un appel, avec
un masque qui porte à la fois la causalité et l'alignement à droite des
préfixes de longueurs différentes.

Le découpage éventuel se fait sur l'axe **temps**, pas sur les lanes : le cache
KV est un tenseur `[lanes, …]` par couche, donc un bloc de jetons s'écrit
simplement à la colonne `t0`, alors que découper par lane demanderait d'écrire
à travers une vue.

### 2. Retrait des lanes finies au décodage — **~1,22×**

Le décodage tourne `steps_max` fois, et `steps_max` est un **maximum sur le
lot**. Une requête à deux cartes restantes (6 cartes cachées, 12 pas) qui
partage un lot avec une entame (24 cartes, 48 pas) faisait 36 pas pour rien :
masquée, mais calculée.

Les positions sont donc rangées par nombre de pas décroissant, ce qui fait des
lanes encore actives un **préfixe** de la table — « retirer » les lanes finies
devient un `narrow` sur l'axe des lanes, une vue et non une copie. (`narrow`
sur l'axe 0 d'un tenseur contigu garde la contiguïté et un décalage nul, donc
les `slice_set` de `forward_step` écrivent bien dans le cache partagé.)

Mesuré par ablation **dans le même binaire** (`COLVER_PLAYGEN_NO_RETIRE=1`),
pas par comparaison de deux compilations : médianes 2,29 contre 1,87 donnes/s.
Mesure bruitée (un tour à 1,27 sur trois), l'ordre de grandeur seul est fiable.

Ensemble, 1 + 2 valent **1,49×** au débit de bout en bout (2,62 contre 1,76
donnes/s, client identique).

### 3. Sur-commande de mondes — **~1,05×**

`retain_valid` écarte les mondes que la belote rend impossibles : le sampler ne
voit pas l'annonce. Demander `n` en rendait ~0,85 n, et il fallait un **second
aller-retour** pour finir le compte — 1,15 aller-retour par décision mesuré.

Or un aller-retour coûte une séquence de jetons entière sur le GPU, alors que
les lanes supplémentaires d'une même requête sont quasi gratuites : le coût
d'un lot est dominé par le nombre de pas, pas par la largeur. La recherche
sur-commande donc, sur une moyenne mobile du taux de rendu observé.

Ça ne change **pas** le nombre de mondes résolus — le mode compte s'arrête à
`determinizations` — seulement lesquels. Effet de bord bénéfique : moins de
décisions terminent en repli local (mondes uniformes), donc un peu moins de
dilution de l'agrégat.

### 4. Plusieurs sidecars

`worlds.url` accepte une **liste séparée par des virgules**, répartie en
tourniquet sur un compteur *global au processus* — par instance, tous les
`IsDdPlayer` (un par siège et par thread) partant de zéro enverraient leur
première requête au même GPU. Répéter une URL pondère : `a,a,b` envoie deux
tiers du trafic à `a`, ce qui est le seul réglage offert entre GPU de débits
différents.

## Ce que ça donne

| configuration | donnes/s | ×  |
|---|---|---|
| départ (32 threads, sidecar d'origine) | 1,05 | 1,00 |
| + threads réglés (256) + sur-commande | 1,76 | 1,68 |
| + préfixe groupé + retrait de lanes | **2,62** | **2,50** |

Une 3090, 40 mondes par décision. À 2,62 donnes/s : 100 k donnes en 10,6 h,
1 M en 4,4 jours — avant le second GPU.

## Ce qui a été mesuré et **écarté**

### playgen v3-small : 1,65× plus vite, mais ce n'est pas le même joueur

Le modèle réduit (d=256 L=4 contre d=384 L=6, 3,3× moins de paramètres) rend
**4,32 donnes/s contre 2,62**, soit 1,65×. C'est le plus gros levier restant.

Mais ses mondes ne sont **pas ceux de v2**. `bench_prefill_eq` compare les
marginales p(carte → siège) sur les mêmes positions, avec pour témoin deux
tirages du *même* modèle — la seule référence honnête, deux échantillons de
512 mondes s'écartant toujours de ~0,013 en moyenne. Résultat : **2,09× le
bruit d'échantillonnage** (écart moyen 0,028, max 0,68 sur une case).

Donc : v3-small ne se substitue pas à v2 sans changer le joueur qui produit le
corpus. C'est un arbitrage — 1,65× de débit contre un sampler différent — et
pas une optimisation. Il reste à mesurer ce que ce sampler vaut *en force de
jeu* (`bench_world_cred`, ou un h2h), ce qui n'a pas été fait.

C'est aussi pourquoi `bench_prefill_eq` existe : les changements 1 et 2 sont
mathématiquement identiques mais **pas bit-à-bit** (les matmuls changent de
forme, donc l'ordre de réduction flottant), et le retrait de lanes change en
plus l'ordre de consommation du RNG. Comparer les mondes un à un ne dirait donc
rien. Les deux passent à **1,001×** et **1,06×** du témoin.

### Ce qui n'a pas été fait

- **Cache du préfixe KV côté sidecar.** Le serveur est sans état : il rejoue la
  donne à chaque requête. Dans une génération de masse, les ~8 décisions d'un
  même siège partagent un préfixe qui ne fait que croître. Mais depuis le
  préfixe groupé, le prefill ne coûte plus qu'un ou deux pas — la marge est
  bien plus petite qu'elle ne l'était.
- **Connexions persistantes.** Le client ouvre un TCP par requête
  (`Connection: close`). Sur 1,8 ms de RTT c'est 0,5 % d'une requête de 350 ms,
  donc invisible ; mais ça laisse ~3 100 sockets en `TIME_WAIT` à 2 donnes/s,
  et cette borne se rapprochera si le débit monte encore.
- **Groupement par stade de donne.** Le retrait de lanes en couvre l'essentiel
  (le décodage) sans découper les lots. Reste que `cap`, et donc la largeur de
  l'attention, est dimensionné sur la position la plus longue du lot.

## Fichiers

- `colver-core/src/bin/gen_games_isdd.rs` — le générateur, le profil, `--check`
- `arena/bots/gen_isdd.toml` — le joueur qui produit le corpus, figé
- `colver-core/src/bin/bench_prefill_eq.rs` — deux sidecars, même distribution ?
- `colver-core/src/playgen/gpu.rs` — `forward_prefill`, retrait de lanes
- `colver-core/src/search/is_dd.rs` — `source_fill`, la sur-commande
- `colver-core/src/worlds.rs` — la liste d'URL et son tourniquet
