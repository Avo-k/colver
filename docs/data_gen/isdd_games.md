# Générer des donnes complètes jouées par IS-DD

`gen_games_isdd` produit des **donnes entières** — enchère réelle, tous les
tours, et les 32 cartes dans l'ordre — jouées par le joueur de référence du
projet (bid v6 + IS-DD sur mondes playgen). Sortie au format `COLVGM02`, celui
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

Le chemin le plus court, qui enchaîne sidecar → génération → vérification →
restitution de la VRAM, et qui porte les trois disciplines de ce document sous
forme de garde-fous plutôt que de prose :

```bash
COLVER_GEN_GPU_HOST=moxxi scripts/training/gen_isdd_corpus.sh --deals 100000
```

Il refuse de démarrer si une génération tourne déjà, si des éclats d'un run
précédent traînent, ou s'il reste moins de 10 Go de VRAM ; il ne tue que le
sidecar qu'il a lui-même démarré, y compris sur Ctrl-C. À la main :

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

## Le score de partie est dans le format (COLVGM02, 2026-08-04)

Jusqu'ici `--match-mode` produisait un corpus qu'on ne pouvait pas exploiter :
`COLVGM01` ne portait pas le score, donc un playgen entraîné dessus voyait des
enchères qu'il **ne pouvait pas expliquer** — la variable qui les décide n'était
pas dans son entrée. Ce n'est pas de l'information manquante, c'est de
l'entropie irréductible ajoutée, et elle est chiffrée : +0,074 à +0,121 de
perplexité sur la tête d'enchère dès 1200 points d'écart, contre +0,0028 pour
diviser les paramètres du modèle par 3,3.

Le format porte donc deux `u16` de plus par donne — **le cumul de partie
*avant* la donne**, celui que l'annonceur a lu. 62 octets au lieu de 58.

```text
COLVGM01  dealer(1) hands(16)                  n(1) actions(n)
COLVGM02  dealer(1) hands(16) score_ns/ew(4)   n(1) actions(n)
```

Trois décisions, chacune contre une erreur précise :

1. **Le score d'avant, jamais celui d'après.** Le score d'après est une
   conséquence de la donne à prédire : le donner reviendrait à montrer la
   réponse. `gen_games_isdd` le capture donc avant `play_and_record`, et l'ordre
   des deux lignes est l'invariant, pas un détail de style.
2. **N-S et E-O bruts dans le fichier, « moi / l'adversaire » dans le
   tokeniseur.** Le fichier porte le fait objectif ; seule la tokenisation sait
   qui observe. Passer l'ordre brut aux quatre sièges ferait croire à E et O
   qu'ils mènent quand ce sont N et S — l'erreur exacte déjà commise une fois
   sur `write_bid_observation_score_aware_v3`, et que
   `score_tokens_are_observer_relative` épingle ici.
3. **Les deux versions se relisent, la sortie est toujours du v2.** Un
   `COLVGM01` rend 0-0, ce qui n'est pas un repli arbitraire : *tous* les corpus
   existants viennent de donnes indépendantes, donc 0-0 y est la vérité. En
   revanche les enregistrements n'ont pas la même longueur, donc concaténer des
   corps bruts de l'un sous le magic de l'autre produirait un fichier qui se
   relit sans erreur et décale tout — le pire mode de défaillance possible pour
   un corpus. `merge_colvgm.py` convertit au lieu de concaténer.

**Côté modèle, le vocabulaire change, donc le magic aussi.** Deux jetons
d'en-tête après `OBSPOS`, bucketés par 200 points (10 seaux) et relatifs à
l'observateur, portent le vocabulaire primaire de 31 à 41 et la séquence de 122
à 124. Un modèle entraîné ainsi est un **`COLVPG03`** : sans magic distinct,
charger un v2 avec le vocabulaire v3 décalerait silencieusement toutes les
matrices suivantes. `train_playgen --v3`, `export_playgen --v3`.

L'invariant qui rend la version bon marché à raisonner est testé :
**v3 = v2 plus deux jetons d'en-tête, et rien d'autre** — mêmes cibles, mêmes
masques, positions décalées de 2 (`v3_is_v2_plus_two_header_tokens`).

⚠️ **Un v3 s'entraîne et s'évalue, il ne s'échantillonne pas encore.** Le
préfixe d'inférence se construit depuis un `GameState`, qui ne porte pas le
cumul de partie ; il faudrait le transporter depuis `MatchContext` à travers
`WorldSource` puis le protocole HTTP du sidecar. Rien de tout cela n'est fait,
et rien ne serait validable de bout en bout tant qu'aucun v3 n'existe. En
attendant, `PlaygenSampler::new` et `playgen_gpu_server` **refusent** un
COLVPG03 en le nommant, plutôt que de produire des mondes tirés d'un préfixe
que le modèle n'a jamais vu.

### Combien de donnes sont réellement dans le régime qui compte

`--check` rend l'histogramme des écarts de score en début de donne. C'est le
dénominateur qui manquait : la pénalité mesurée est **nulle** à 600 points
d'écart et franche à 1200, donc ce qu'une entrée « score » peut récupérer est
proportionnel à la part du corpus au-delà du seuil — part qui n'avait jamais été
mesurée sur du jeu réel.

Premier relevé (90 donnes seulement, **indicatif**) :

| écart | <300 | 300-599 | 600-899 | 900-1199 | ≥1200 |
|---|---|---|---|---|---|
| part des donnes | 43 % | 29 % | 13 % | 13 % | **1 %** |

Si ça se confirme sur un vrai corpus, le régime franchement pénalisé est
**rare**, et le gain attendu bien plus petit que le +0,074-0,121 ne le laisse
croire. À refaire sur ≥10 000 donnes avant d'en conclure quoi que ce soit —
et c'est désormais gratuit, l'histogramme sort de `--check`.

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

Taille : **62 octets par donne** en `COLVGM02`, ~40 actions. 1 M de donnes =
62 Mo (58 Mo en `COLVGM01`, sans les quatre octets de score).

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

### Emprunter le GPU de la **prod** : mesuré une fois, ça passe (2026-08-06)

La section précédente dit de ne pas partager le GPU d'un run en cours. La
question inverse — un run peut-il partager le GPU des **joueurs** ? — a été
mesurée, et la réponse est oui, avec une marge confortable et un levier connu
si elle se referme.

Configuration mesurée : un run de 500 000 donnes avec le sidecar de prod en
quatrième URL, donc **un quart du trafic sur le GPU de moxxi** pendant qu'un
joueur faisait une partie de trois donnes sur colver.net.

```bash
./target/release/gen_score_layer --pool data/deals/base_5M.bin \
  --offset 0 --count 500000 --threads 160 --checkpoint 500 \
  --out data/deals/scores_isdd_v2.sc --games data/training/isdd_games_v2.bin \
  --url http://localhost:8003,http://localhost:8003,http://localhost:8003,http://192.168.1.23:8003
```

**La charge sur le sidecar de prod est massive** — deux ordres de grandeur :

| | témoin (jeu seul) | pendant le run |
|---|---|---|
| requêtes/min | 20 | **2 100** (×107) |
| lanes/min | 3 700 | **84 400** (×23) |
| attente en file par lot | 0 ms | **45 ms** |

**Ce que Dédé paie est bien plus modeste**, parce que le groupage du sidecar
amortit : latence côté sidecar (GPU + attente) de la requête de Dédé,
**93 ms → 193 ms de médiane** (p90 150 → 303, max 417). Soit **×2,1, +100 ms
par décision**.

**Et le joueur ne voit rien**, parce que `pacing.hold` fait compter la
recherche *dans* la pause d'affichage au lieu de s'y ajouter. Le tempo standard
impose 1,4 s → 0,9 s par carte, et les trois lignes `IS-DD donne terminée` de
la partie disent :

```
16 décisions, 226.4 mondes/déc (min 64,  méd 256), 693 ms/coup
15 décisions, 244.8 mondes/déc (min 88,  méd 256), 302 ms/coup
14 décisions, 251.8 mondes/déc (min 197, méd 256), 305 ms/coup
```

Tout est sous le **plancher** de 0,9 s, donc il reste au moins 200 ms de marge
sur la pire donne. La force de jeu ne bouge pas non plus : médiane 256 mondes,
soit le plafond `ISDD_MAX_WORLDS`, sur les trois donnes. Le plancher de 64 n'a
mordu qu'**une fois** (`min 64` exactement sur la première donne = budget de
1200 ms dépassé) : c'est le seul coup qui a pu être visible, et c'est le
comportement voulu — sous pression, la dégradation se paie en latence, qui se
voit, et non en force de jeu, qui ne se voit pas.

**Le levier si ça se referme** est `ISDD_MAX_WORLDS` (256), pas
`ISDD_MIN_WORLDS` (64) : c'est le plafond qui fabrique la contention, et le
genou mesuré est à 60 mondes. Baisser le plancher ne rendrait que du jeu plus
faible, sans rendre de GPU.

**Ce qui ne se déduit pas des durées de donne** : les donnes du joueur ont duré
143 / 113 / 144 s contre ~68 s pour le témoin de la veille, et **ce n'est pas
le GPU**. Le tempo à lui seul impose ~42 s par donne, et le total de recherche
de Dédé fait 11 s puis 4,5 s — *à l'intérieur* des pauses. Le reste est du
temps de réflexion humain, qui varie d'un joueur à l'autre bien plus que tout
ce qui est mesuré ici. Une durée de donne ne peut pas servir de jauge.

Réplication — le témoin est la partie d'un joueur **avant** le démarrage du
run, dans le même journal, et il faut vérifier qu'il est en `pacing = standard`
(sinon c'est DouDou50, qui ne touche pas le sidecar) :

```bash
# 1. les lots du sidecar de prod (le journal tourne : extraire avant qu'il parte)
ssh moxxi "journalctl -u playgen-gpu --since '13 hours ago' --no-pager -o short-iso" \
  | grep 'lot:' > lots.txt
# 2. la vue de Dédé, une ligne par donne, côté web
ssh moxxi-docker "docker logs colver-colver-1 2>&1 | grep 'IS-DD donne'"
# 3. le mode de la partie, sinon la comparaison ne vaut rien
ssh moxxi-docker "docker exec colver-colver-1 python -c \"import sqlite3,os; \
  print(sqlite3.connect(os.environ['COLVER_DB_PATH']).execute( \
  'select id,pacing from matches order by rowid desc limit 5').fetchall())\""
```

Isoler la requête de Dédé dans `lots.txt` demande de retirer les requêtes du
run : elles font **42 lanes** de médiane contre ~256 pour Dédé, donc un lot le
contient si `lanes − 42 × (requêtes − 1) ≥ 200`. Sans ce filtre on mesure des
lots de génération à six requêtes, qui atteignent 240 lanes tout seuls — la
première lecture donnait 163 ms au lieu de 193 par cette erreur-là.

Deux réserves sur la portée : la latence est mesurée **côté sidecar**, pas en
aller-retour client (le LAN est le même dans les deux fenêtres, donc l'écart
est bon, pas le niveau) ; et c'est **un joueur, trois donnes**. Ça suffit à
dire « la marge est d'un facteur 3 », pas à cadrer une file d'attente.

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

**Combien de marge reste-t-il avant que le CPU devienne le mur ?** À lire sur
un run **non sur-souscrit** (32 threads, un par cœur), le seul où la mesure
sépare le calcul de l'attente d'ordonnancement : **1,96 s de solve DD par
donne**, soit ~16 donnes/s de capacité sur 32 cœurs. On en est à 2,8. **Le GPU
peut donc gagner encore ~6× avant que le CPU ne borne quoi que ce soit** —
autrement dit un second GPU, puis un troisième, se paient entièrement.

⚠️ Ne pas lire ce chiffre sur un run à 256 threads : `total_us − source_us` y
compte aussi le temps où un thread est *prêt mais pas ordonnancé*, ce qui
gonfle « solve DD » à 15 s par donne. Le même run qui semblerait alors saturer
le CPU en utilise en réalité le huitième.

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

### 4. Fusion ACT+CARD au décodage — **1,62× de latence** (2026-08-04)

Un coup coûtait **deux** forwards : le jeton `ACT`, dont on lit les logits de
carte, puis le jeton `CARD` — **dont la sortie était jetée**. Il ne servait qu'à
écrire son k/v dans le cache pour que les positions suivantes l'attendent.

Or l'`ACT` du coup suivant est **déterministe** depuis la machine à états
publique : à l'instant où la carte `i` est tirée, `CARD_i` *et* `ACT_{i+1}` sont
tous les deux connus. Rien n'oblige à les séparer. Ils partent donc dans **un**
appel à 2 jetons, avec la causalité intra-bloc — exactement ce que
`forward_prefill` fait déjà pour le préfixe. Le décodage passe de `2·steps`
lancements à `steps`.

C'est le même argument que le préfixe groupé, appliqué au dernier endroit où on
payait un lancement séquentiel pour un jeton qui ne porte aucune décision : un
pas est borné par le **lancement de noyaux** (~2,1 ms, 1 lane ou 40), donc ce
qui compte est le *nombre* de forwards, pas leur largeur.

**La dernière carte d'une lane n'est jamais poussée**, et c'est ce qui rend la
fusion compatible avec le retrait de lanes (§2) : un k/v n'est lu que par les
positions *ultérieures de la même lane* — le cache est `[lanes, …]`, chaque lane
n'attend qu'elle-même — et il n'y en a plus. Une lane retirée entre `i-1` et `i`
voit donc son dernier `CARD` disparaître, ce qui est correct **et** économise un
lancement de plus. Les lanes actives restant un préfixe de la table,
`narrow(0, 0, n_act)` continue d'aligner les indices sans rien changer.

A/B alterné à la requête (`bench_sidecar_ab`, un thread client, latence vue du
client, réseau et HTTP compris), ablation `COLVER_PLAYGEN_NO_FUSE=1` **dans le
même binaire** :

| cartes restantes | 8 | 7 | 6 | 5 | 4 | 3 | 2 |
|---|---|---|---|---|---|---|---|
| ratio fusion / ablation | 0,613 | 0,613 | 0,615 | 0,617 | 0,623 | 0,631 | 0,646 |

**1,62× à 40 mondes par requête** (médiane appariée, 84 requêtes, 45,4 contre
73,3 ms), monotone dans le bon sens — le gain suit le nombre de pas, donc il est
le plus grand à l'entame — et **aucune régression à aucun stade**.

**Le gain dépend de la largeur du lot, un seul chiffre serait faux.** À 256
mondes par requête il tombe à **1,33×** (125 contre 169 ms) : à cette largeur le
forward fait de l'arithmétique réelle et n'est plus purement borné par le
lancement, donc diviser le compte de lancements par deux achète moins. Les deux
régimes existent : un joueur seul sur le web envoie une requête étroite (~40
mondes, `dets_schedule`), la génération de masse sature les 256 lanes. Confirmé
hors réseau par `bench_playgen_batch --positions 12 --worlds 20` : 15,8 → 11,6
ms/position, soit 1,36× à 240 lanes.

**Ce que ça ne touche pas.** Seul `generate_worlds_multi` (route
`/play_worlds`, le chemin d'IS-DD) est fusionné. `generate_worlds_scored` et
`auction_round` sont inchangés — visible dans le même bench : le chemin « une
par une » mesure 116,2 contre 116,5 ms, c'est-à-dire rien.

**Validation.** Comme le préfixe groupé, la fusion est mathématiquement
identique mais les matmuls passent de M=1 à M=2, donc l'ordre de réduction
flottant change et l'égalité bit-à-bit n'est pas la bonne attente. Deux
mesures :

- `bench_prefill_eq --positions 40 --worlds 256` entre un sidecar fusionné et un
  sidecar sous ablation : **A↔B / témoin = 1,007**, et A↔A' = 1,02 du témoin
  (le sampler n'a pas non plus *réduit* sa variabilité) ;
- `bench_playgen_batch` à graine fixe : les trois empreintes sont **identiques
  au bit près** entre fusion et ablation, y compris celle du chemin groupé
  (`0x01e73b5169574643`), et les mondes de la vérification K=1 aussi. Sur cette
  charge, le changement d'ordre de réduction n'a fait basculer **aucun** tirage.

⚠️ Ce changement modifie `playgen::SURFACE` (`build.rs` hache `src/playgen/`),
donc `/health` passera à `fresh: false` dès que le web sera déployé, jusqu'à ce
que le sidecar soit **reconstruit et redémarré à la main**. C'est l'alarme qui
fonctionne, pas un incident.

### 5. Plusieurs sidecars

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

**Au débit soutenu, c'est mieux que ça : 2,89 donnes/s.** Mesuré sur un vrai run
de 38 000 donnes (13 138 s), contre 2,62 sur des bancs de 150. Un run long garde
le groupeur du sidecar plein, ce qu'une rafale de banc ne fait pas — c'est donc
2,89 qu'il faut utiliser pour dimensionner un corpus : **100 k donnes en 9,6 h,
1 M en 4 jours** sur une seule carte.

Et à cette échelle le profil se déplace encore : **96,1 % d'attente sidecar
contre 3,9 % de solve DD** (256 threads). Le GPU est si saturé que presque tout
le temps de thread est de la file d'attente. Plus de GPU est le seul levier
qui reste.

Corpus produit et validé le 2026-08-04 : `data/training/isdd_games_v1.bin`,
**43 076 donnes**, 1 729 157 actions, 0 irrejouable, **1 408 660 prédictions
d'enchère**.

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

### playgen v2-belote-small : 1,65× plus vite, mais ce n'est pas le même joueur

Le modèle réduit (d=256 L=4 contre d=384 L=6, 3,3× moins de paramètres) rend
**4,32 donnes/s contre 2,62**, soit 1,65×. C'est le plus gros levier restant.

Mais ses mondes ne sont **pas ceux de v2**. `bench_prefill_eq` compare les
marginales p(carte → siège) sur les mêmes positions, avec pour témoin deux
tirages du *même* modèle — la seule référence honnête, deux échantillons de
512 mondes s'écartant toujours de ~0,013 en moyenne. Résultat : **2,09× le
bruit d'échantillonnage** (écart moyen 0,028, max 0,68 sur une case).

Donc : v2-belote-small ne se substitue pas à v2 sans changer le joueur qui produit le
corpus. C'est un arbitrage — 1,65× de débit contre un sampler différent — et
pas une optimisation. Il reste à mesurer ce que ce sampler vaut *en force de
jeu* (`bench_world_cred`, ou un h2h), ce qui n'a pas été fait.

C'est aussi pourquoi `bench_prefill_eq` existe : les changements 1 et 2 sont
mathématiquement identiques mais **pas bit-à-bit** (les matmuls changent de
forme, donc l'ordre de réduction flottant), et le retrait de lanes change en
plus l'ordre de consommation du RNG. Comparer les mondes un à un ne dirait donc
rien. Les deux passent à **1,001×** et **1,06×** du témoin.

### Ce que la prod y gagne : ~1,9× de latence par coup

Le préfixe groupé n'a pas été fait pour la prod, mais c'est elle qui en profite
le plus, et pour une raison structurelle : **un joueur seul sur le web envoie
une requête à la fois**, donc le préfixe n'est amorti sur rien. C'est exactement
le régime où il pesait 65 à 81 % du coût.

Mesuré en A/B alterné sur la configuration de prod (`--lane-budget 256`), un
seul thread client, latence vue du client (réseau et HTTP compris) :

| | ms par requête |
|---|---|
| ancien binaire | 142 · 123 · 143 |
| nouveau | 85 · 66 · 75 |

**142 → 75 ms, soit 1,9×.** Dans le budget de 1 200 ms d'un coup de Dédé, ça
double à peu près le nombre d'allers-retours possibles, donc le nombre de mondes
réellement cherchés. C'est un gain de *force de jeu* pour les joueurs, pas
seulement de débit hors-ligne — et il ne coûte rien puisque la distribution des
mondes est inchangée (`bench_prefill_eq`, 1,001× du témoin).

⚠️ **Le sidecar se déploie à la main, séparément du webhook.** Fait le
2026-08-04 : sources rsync vers `~/playgen/colver` sur moxxi, `cargo build
--release --bin playgen_gpu_server --features gpu_server` (avec `nvcc` dans le
`PATH` et `CUDARC_CUDA_VERSION`), `systemctl restart playgen-gpu`. Contrôle :
`curl colver.net/health` doit rendre `sidecar.fresh: true` — c'est-à-dire la
même empreinte de sources des deux côtés. Un `false` signifie que l'un des deux
est resté en arrière.

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
