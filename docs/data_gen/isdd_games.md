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

# Rassembler les éclats d'un run INTERROMPU (et seulement interrompu :
# un run terminé a déjà fusionné puis effacé les siens)
cargo run --release --bin gen_games_isdd -- --merge data/training/isdd_games.bin \
  --out data/training/isdd_games.bin
```

⚠️ Cette commande a `--out` égal au préfixe, donc elle **écrit par-dessus le
corpus**. Lancée après un run *terminé* — qui a supprimé ses éclats — elle
rassemblait zéro donne et écrasait le corpus par un fichier vide de 16 octets,
en sortant 0. Elle refuse désormais d'écrire quoi que ce soit sans éclat, mais
le piège valait d'être nommé : « aucun éclat » est exactement l'état que laisse
un run **réussi**, pas seulement une faute de frappe.

Un run de plusieurs heures écrit des **éclats** (`--shard`, 5000 donnes par
défaut) au fil de l'eau, les fusionne à la fin et les efface alors seulement.
`GameReplay::write_all` n'écrivant qu'en une fois, sans ça une interruption à
95 % ne laisserait rien.

`--merge` diffère de [`scripts/training/merge_colvgm.py`](../../scripts/training/merge_colvgm.py)
sur un point : le script vérifie la structure (magie, parcours des
enregistrements de longueur variable), `--merge` **rejoue** en plus chaque
donne. Pour recoller deux corpus déjà validés, le script suffit et va plus
vite ; pour rattraper des éclats d'un run qui s'est mal terminé, on veut le
rejeu.

`--match-mode` enchaîne les donnes en parties de 2000 points au lieu de les
tirer indépendantes. Ce n'est pas cosmétique : bid v6 lit une observation
*score-aware*, donc il annonce autrement à 900-200 qu'à 0-0, et le corpus
playgen actuel est **entièrement à 0-0**. C'est le manque relevé dans
[playgen v3](../belief/playgen.md).

⚠️ **Mais ce n'est pas encore le bon défaut, et pour une raison qui tient au
format.** `COLVGM01` ne porte pas le score de partie. Un playgen entraîné sur
des donnes de mode partie verrait donc des enchères qu'il **ne peut pas
expliquer** — la variable qui les décide n'est pas dans son entrée. Ce n'est
pas de l'information en plus, c'est de l'entropie irréductible en plus. Le
défaut reste donc la donne indépendante à 0-0, cohérente avec elle-même ;
`--match-mode` devient le bon choix le jour où le format transporte le score,
et c'est exactement le changement que réclame la note playgen v3.

Ce qui change *déjà* sans mode partie, et qui est l'objet du binaire :
l'enchère est **réellement jouée** — tous les tours, contres et surcontres
compris — au lieu d'un atout imposé.

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

`--check` décrit aussi le **contenu**, pas seulement la structure : un corpus
peut être parfaitement rejouable et inutilisable, rien n'interdisant à un
joueur mal configuré de passer neuf donnes sur dix. Et il donne au passage une
validation gratuite — le corpus produit ici et `heldout_20k_s90210.bin`, un
corpus playgen existant, ont **le même profil** :

| | ce générateur | `heldout_20k_s90210.bin` |
|---|---|---|
| donnes passées | 0,1 % | 0,1 % |
| annonces par donne | 8,2 | 8,2 |
| preneur N-S | 50,5 % | 49,9 % |
| contrat réussi | 64,6 % | 64,0 % |
| contré | 25,2 % | 25,1 % |

Ce n'est donc pas un artefact de ce binaire : **un quart des contrats sont
contrés**, chez les bidders de ce projet, depuis toujours. Si c'est loin d'une
table humaine, un playgen entraîné là-dessus apprendra une distribution
d'enchères biaisée — mais c'est une question sur le *bidder*, pas sur le
générateur, et elle n'est pas tranchée ici.

### La recette d'acceptation : le tokeniseur de playgen lui-même

`--check` dit que le corpus est un corpus. Ce qui dit qu'il est **utilisable**,
c'est le tokeniseur qui le consommera — il vérifie que la carte réellement
jouée est toujours dans le masque visible par l'observateur, ce qu'aucune
vérification de format ne peut voir :

```bash
COLVER_GAMES=$PWD/data/training/isdd_games.bin \
  cargo test -p colver-core --release --lib validate_games_file -- --ignored --nocapture
```

Sur 5 076 donnes générées, les deux variantes passent :

```
validated 5070 games, 648960 preds, 13.0% forced, avg hidden-mask size 12.0
validated 5076 games (0 skipped): 648960 play preds, 165132 bid preds
```

Deux choses à y lire. **`0 skipped`** : toutes les donnes sont exploitables.
Et **165 132 prédictions d'enchère**, soit 32,5 par donne — c'est exactement ce
que les générateurs à atout imposé produisent en quantité *nulle*, et la raison
d'être de ce binaire. Les 13,0 % de coups forcés et la taille de masque moyenne
de 12,0 collent aux valeurs publiées pour le corpus de référence
(« ~13.5% forced, avg hidden-actor mask ≈ 12 cards »), donc la distribution est
indiscernable de celle sur laquelle playgen a déjà été entraîné.

## Survivre à une nuit

Un run de plusieurs heures rencontre des pannes passagères, et le comportement
par défaut était de mourir dessus. **Mesuré à mes dépens** : un run de 28 000
donnes s'est arrêté à 5 076 parce qu'un second sidecar s'est mis à partager le
GPU. Les lectures ont expiré (6 s), et sous `fallback = "strict"` — le seul
réglage honnête pour un corpus — chaque expiration rendait une erreur qui
faisait sortir *tous* les threads.

Trois règles en découlent, et elles se tiennent ensemble :

1. **Une erreur de source jette la donne, pas le run.** La donne en cours n'est
   jamais enregistrée à moitié, son jeton est rendu (`--deals N` reste un compte
   de donnes *complètes*), et le run continue.
2. **Le budget d'erreurs se lit en secondes de panne, pas en nombre
   d'erreurs.** Sous une coupure totale, chaque thread produit une erreur par
   expiration de lecture : elles s'accumulent à `threads / 6` par seconde, soit
   43/s à 256 threads. Un budget fixe à 50 abandonnerait après **une seconde**
   de coupure. D'où `threads × 4`, qui vaut ~25 s quel que soit le parallélisme.
3. **Un run incomplet sort 1**, avec le compte demandé et le compte obtenu, et
   ses éclats sont conservés. `--deals N` est une commande, pas un souhait.

Et la leçon d'exploitation qui va avec : **ne pas partager le GPU d'un run en
cours**. Le sidecar n'a pas planté, il a ralenti — assez pour que les lectures
expirent. C'est la même famille que les sidecars oisifs qui mangent la VRAM,
plus bas.

### Plusieurs sidecars : le contrôle de santé ferme la porte

`--url a,b` vérifie **les deux** au démarrage. Vérifié de bout en bout : avec
`b` éteint, les 256 threads échouent à la construction en nommant l'URL morte,
et rien n'est écrit — au lieu d'envoyer silencieusement la moitié des mondes
dans un trou et de produire un corpus à moitié dégradé. Avec les deux vivants,
les deux servent (175 lots contre 57 sur un test court).

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

Une 3090, 40 mondes par décision.

Et par nombre de mondes, la question qui décide du budget :

| mondes/décision | donnes/s | 100 k donnes | attente sidecar |
|---|---|---|---|
| 20 | 3,93 | 7,1 h | 71 % |
| 40 | 2,34 | 11,9 h | 63 % |
| 60 | 1,70 | 16,3 h | 61 % |

Le débit décroît **moins vite que le nombre de mondes** (×3 de mondes ne coûte
que ×2,3 de temps) : une part du travail par décision ne dépend pas du compte —
le préfixe, la partie non échantillonnée des décisions, le rejeu côté sidecar.
Doubler de 20 à 40 mondes coûte donc 1,7× et non 2×.

La part d'attente du sidecar **descend** quand les mondes montent, parce que
c'est le solve DD qui grossit : à 20 mondes la génération est encore
franchement limitée par le GPU, à 60 les deux ressources se rapprochent. C'est
le signe que l'optimisation a fait son travail — au départ ce rapport était de
93/7.

### Un calendrier de mondes par stade — 1,24×, et une erreur à ne pas refaire

Le besoin de mondes n'est pas uniforme : `isdd_dets_by_stage` place tout le
regret au-dessus de 0,10 point DD à **8-6 cartes restantes**, et **zéro en
dessous de 3 cartes à n'importe quel budget**. Un monde tiré en finale n'achète
rien. D'où `--dets-schedule 40,40,40,30,20,15,15` (de 8 cartes à 2) :

| réglage | mondes/donne | donnes/s |
|---|---|---|
| plat 40 | 280 | 2,02 |
| `60,60,60,30,30,20,20` | **280** | 1,63 |
| `40,40,40,30,20,15,15` | 200 | **2,50** |

⚠️ **La ligne du milieu est le piège, et j'y suis tombé.** Le raisonnement
« même total de mondes, donc même coût, donc redistribuer vers l'entame est
gratuit » est **faux** : un monde à 8 cartes restantes demande 24 cartes
cachées au sampler, soit 48 pas de décodage, contre 6 cartes et 12 pas à deux
cartes restantes. Un monde d'entame coûte donc ~4× un monde de finale sur le
GPU — et bien davantage encore au solveur. À total égal, le calendrier montant
est **plus lent de 20 %**, pas neutre.

Le sens qui paie est donc le sens **décroissant**, et il paie deux fois : il
coupe les mondes là où ils sont à la fois les moins utiles et… les moins chers.
Le gain vient surtout du nombre, pas de la redistribution.

⚠️ **La VRAM libre est un paramètre caché de ces mesures.** Une série entière a
été mesurée 30 % trop lente parce que trois sidecars de test oisifs occupaient
21 Go des 24 de la carte : le sidecar actif n'a jamais planté, il a juste
ralenti. Vérifier `nvidia-smi` **avant** de croire un chiffre, et tuer les
sidecars d'expérience — ils ne rendent jamais leur VRAM tant qu'ils vivent.

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

### TF32 : **3 à 5× plus lent**, à ne pas réessayer

L'attention et les FFN sont en f32 sur une carte Ampere, qui sait faire du TF32
à ~8× le débit. `NVIDIA_TF32_OVERRIDE=1` sur le sidecar rend **0,44 et 0,83
donnes/s contre 2,18 et 2,28** en f32 — pas un gain amoindri, une régression
d'un facteur 3 à 5. Cause non instrumentée (bascule de cuBLAS vers un noyau
sans tensor cores pour ces formes, très probablement). L'idée est close pour ce
modèle et ces tailles de lot.

### Rétrécir la fenêtre d'attention : bloqué par candle

L'attention pèse **40 % du forward** pour ~2,5 % des FLOP — ce sont `lanes ×
têtes` gemms minuscules à M = 1, le pire cas pour cuBLAS. Son coût est
proportionnel à `cap = lmax + 2 × steps_max`, alors qu'au pas `t` seules les
colonnes `[0, t]` portent quelque chose : diviser la largeur par deux
diviserait un poste à 40 %.

Refusé par la bibliothèque : `narrow` sur l'axe `cap` donne un tenseur non
contigu, et **candle 0.9.2 ne sait pas multiplier des tenseurs non contigus —
il rend une erreur au lieu de recopier** :

```
matmul is only supported for contiguous tensors
rstride: Layout { shape: [256, 8, 48, 19], stride: [31488, 3936, 82, 1] }
```

Recopier soi-même coûterait plus que l'attention économisée. Rouvrable si
candle gagne un chemin gemm à `lda`, ou avec un noyau d'attention dédié.

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
