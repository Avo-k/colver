# Annoncer en simulant la donne

`strategy = "rollout"` — la simulation de la page Annonces, tournée en politique
d'enchère. À son tour de parole, le bot met quelques annonces en concurrence,
joue la donne jusqu'au bout pour chacune (mondes playgen, reste de l'enchère par
bid v6, jeu par DouDou50), et annonce celle dont l'**espérance d'écart de score**
est la meilleure.

Code : [`colver-core/src/agent/bid_rollout.rs`](../../colver-core/src/agent/bid_rollout.rs).
Sonde : `bench_bid_rollout`. Criblage : `scripts/analysis/rollout_bid_sweep.sh`.

Il n'a pas de poids à lui. Ce qu'il vaut est ce que valent son réseau de
référence, son joueur de cartes, ses mondes et son budget de simulation.

## Le résultat : **négatif** (2026-08-06)

`rollout_probe_1024` (1024 mondes playgen × 5 candidates, sondage, GPU) contre
son témoin exact `v6_isdd_75M`, **deux échantillons indépendants** (graines 42 et
20260806) :

| run | matchs | donnes | par donne | marge/match |
|---|---|---|---|---|
| pilote | 20×2 | 355 | 51,5 % ± 5,2 | +116 |
| validation | 100×2 | 1 884 | 50,6 % ± 2,3 | +26 |
| **combiné** | 120×2 | **2 235** | **50,8 % ± 2,1** | — |

**Indiscernable de 50 %** — l'intervalle combiné est [48,7 ; 52,9]. Simuler la
donne à chaque parole, pour ~1 000× le coût d'un appel au réseau, ne joue **ni
mieux ni moins bien** que v6 seul. L'effet vrai, quel qu'il soit, est borné à
~±2 points de pourcentage par donne.

⚠️ Deux lectures à ne pas faire :

- **Le taux de matchs (46 %) n'est pas un résultat.** À 200 matchs son erreur
  type vaut 3,5 pp, donc son intervalle contient 50 % aussi. Le taux par donne
  porte 9 fois plus d'observations ; c'est lui qu'on lit.
- **La marge par match ne dit rien non plus.** Le pilote donnait +116, ce qui
  suggérait un bidder qui « gagne plus gros quand il gagne » ; la validation
  donne +26. La différence est du bruit, et l'hypothèse ne survit pas.

Ce que le négatif dit et ne dit pas : il dit que **cette** simulation, avec
**ces** mondes et **ce** joueur de cartes, ne corrige pas v6 sur l'agrégat d'un
h2h. Il ne dit pas qu'elle ne corrige jamais rien — un h2h ne voit pas un effet
de quelques points par donne, et la question « où la simulation change-t-elle
d'avis, et à quel coût réel » est une mesure **appariée à la décision**, du même
genre que celles faites pour la belote et pour `DealScore`. Elle n'a pas été
faite.

## Le témoin est obligatoire, et il en existe un exact

`v6_isdd_75M` est **ce bot moins la simulation** : même `bid_v6_isdd_resume`,
même `play_v2/play_final`. Tout h2h contre lui mesure la simulation et rien
d'autre.

## Ce qui coûte, et ce qui ne coûte pas

**Un monde simulé, c'est 32 passes avant de DouDou50** — 178 µs pièce
([BENCH.md](../BENCH.md)), soit **~5,7 ms de plancher théorique**. Mesuré, une
donne simulée coûte :

| mondes de | donne simulée | dont sidecar |
|---|---|---|
| tirage uniforme | **5,4 / 7,0 ms** | — |
| playgen (`/auction_deals`) | **8,2 / 8,1 ms** | 1,1 ms à 128 mondes, 0,27 ms à 512 |

(deux tours alternés, 128 mondes × 5 candidates ; l'écart entre les deux tours
d'une même colonne dit la dispersion de mesure.)

**Le sidecar n'est pas le coût.** La route `/auction_deals` rend **512 mondes en
716 ms**, et son temps est *plat en n* (32 mondes coûtent 773 ms, 128 en coûtent
615) : c'est un lot GPU, on paie la séquence de jetons, pas les lanes. Amorti sur
512 × 5 déroulements, ça fait **0,27 ms sur ~8 ms, soit 3 %**.

Ce qui reste — ~7 contre ~6 ms — n'est pas de la latence réseau mais du travail
en plus, et c'est plutôt bon signe : **un monde uniforme fait plus souvent une
donne passée** (mains incohérentes, personne ne prend), donc pas de phase de jeu
et pas de coût. Le tirage uniforme était en partie bon marché parce qu'il était
dégénéré.

C'est la réponse à « pourquoi la page d'analyse va vite et pas ça ». Elle ne va
pas vite : `annonces_doudou` déroule ses 1 000 donnes en ~11-20 s de CPU. Ce qui
la fait *paraître* rapide, c'est qu'elle en fait **une** (une main, une annonce),
qu'elle diffuse l'avancement au fil de l'eau, et qu'un second passage sur la même
main tombe dans `sim_cache`. Un bidder, lui, refait ce calcul à chaque parole, de
chaque donne, de chaque match.

### Le déroulement groupé sur GPU : **8×**, et le goulot change de place

`[bid] gpu = true` ([`dmc/gpu_rollout.rs`](../../colver-core/src/dmc/gpu_rollout.rs),
feature `dmc_train`) déroule toutes les lanes **en lockstep** : une fois l'enchère
finie, chaque lane a exactement 32 cartes à jouer, donc à l'étape *k* toutes
posent leur *k*-ième carte et **une seule passe avant les couvre toutes**.

| budget | CPU séquentiel | CPU parallèle (30× le CPU) | **GPU en lot** | gain |
|---|---|---|---|---|
| 512 × 5 | 16,0 s | 11,3 s | **1,44 s** | **7,8×** |
| 1024 × 5 | ~32 s | 23,3 s | **2,83 s** | **8,2×** |

La donne simulée passe de 5,4-8,2 ms à **0,6 ms**. C'est bien la mécanique
attendue : le lot amortit la lecture des poids, et le produit matrice-vecteur
devient un produit matrice-matrice.

**Contrôle obligatoire, et il passe : écart max 3,3e-6** entre les Q du GPU et
ceux de `DmcNet::evaluate` sur 64 positions de profondeurs variées. Deux pièges
silencieux vivent sur ce chemin — l'orientation des matrices et l'espace
canonique 411 — et **aucun ne lève d'erreur** : le réseau rendrait une carte
légale et sans rapport, ce qui se lit comme un joueur un peu faible. `3,3e-6`
est de l'ordre de réduction flottant ; une matrice transposée coûterait des
unités de Q. `bench_bid_rollout --gpu` refuse de mesurer au-delà de 0,05.

**Et le goulot a bougé — ce qu'on vient d'optimiser est devenu le plus petit
poste.** Décomposition à 512×5, en retirant le sidecar (mondes uniformes) :

| poste | temps | part |
|---|---|---|
| appel sidecar `/auction_deals` | **737 ms** | 51 % |
| enchère des simulations, restée sur CPU | ~500 ms | 35 % |
| jeu sur GPU + écriture des observations | ~200 ms | 14 % |

(1 440 ms avec playgen, 703 ms avec des mondes uniformes ⇒ 737 ms de sidecar ;
l'enchère s'estime à 2 560 lanes × ~6 annonces × 33 µs.)

⚠️ **Ça périme ce que cette page disait plus haut** (« le sidecar ne coûte que
3 % ») : c'était vrai quand les déroulements prenaient 20 s. Le même 0,7 s est
maintenant la moitié du budget. Un pourcentage n'est jamais une propriété du
composant, seulement du rapport du moment.

Les deux leviers qui restent, dans l'ordre :
1. **Le sidecar.** Sa route est *plate en n* (716 ms pour 32 comme pour 512),
   donc c'est un plancher de latence par décision, pas un coût par monde. Et
   `MAX_WORLDS = 512` est en dur dans `playgen_gpu_server.rs` alors que le
   `lane_budget` vaut déjà 1024 : demander 1024 mondes coûte **deux requêtes
   séquentielles**, d'où les 2,83 s. Relever la constante rendrait le budget
   1024 presque aussi cher que 512.
2. **L'enchère des simulations.** Même traitement que le jeu — `BidNet` est un
   MLP lui aussi — mais elle ne se met pas en lockstep aussi proprement : les
   lanes en sortent à des instants différents, et certaines par une donne
   passée. Non fait.

### Le parallélisme CPU ne sauve pas : **1,4× pour 30× le CPU**

`parallel` éclate les déroulements sur rayon. A/B **alterné**, deux budgets,
machine à 32 cœurs :

| budget | séquentiel | parallèle (2 971 % CPU, 33 fils) | gain |
|---|---|---|---|
| 128 × 5 | 3 623 / 3 795 ms | **2 684 / 2 826 ms** | 1,35× |
| 512 × 5 | 16 002 ms | **11 259 ms** | 1,42× |

Trente fois le CPU pour 40 % de mur. **DouDou50 est limité par la bande passante
mémoire, pas par le calcul** : ses 2,55 M paramètres font **10,2 Mo relus à chaque
carte jouée**, donc à 178 µs la passe un seul fil tire déjà ~57 Go/s — l'essentiel
de ce qu'une machine de bureau sait fournir. Les fils suivants se disputent le
reste.

Et il y a une cause aggravante, mesurable et corrigeable : `DmcNet::from_floats`
fait `to_vec()` sur chaque matrice, donc **chaque instance possède sa propre copie
de 10 Mo**. À 32 fils cela fait 320 Mo qui se disputent la DRAM, là où une copie
unique partagée tiendrait en L3 (32 Mo) et serait lue par tous depuis le cache.
Passer `DmcWeights` en tranches `Arc<[f32]>` partagées est un changement contenu,
mais il touche un type porteur utilisé par l'arène, `gen_games_isdd` et le web :
c'est un chantier à part, **non fait ici**, et c'est le seul levier connu qui
puisse rendre ce bidder (et le reste) vraiment rapide.

⚠️ Une autre session occupait ~6 des 32 cœurs pendant ces mesures. C'est pour ça
qu'elles sont **alternées** — et les deux tours d'une même ligne s'écartent de
5 %, ce qui borne la dérive. Le raisonnement sur la bande passante ne dépend de
toute façon pas de la charge : il sort de 10,2 Mo × 32 cartes.

**En arène, `parallel` ne sert à rien de toute façon** : les matchs y sont déjà
parallèles — et ils se heurtent au même mur.

Coût d'arène, mesuré (`arena h2h`, 114 s/match à 40×4=160 unités) et proportionnel
à `sims × candidates` :

**Mesuré avec le chemin GPU** (`rollout_probe_1024` contre `v6_isdd_75M`, 20
matchs/direction, 6 fils) : **1 381 s pour 40 matchs**, soit **1,7 match/minute**
et ~35 s le match à 1024×5. Un run de 100 matchs/direction coûte donc **~2 h**.

⚠️ **L'arène est limitée par le sidecar, pas par les cœurs — et ça change deux
réglages.** Le sidecar sérialise `/auction_deals` (`run_alone` : ces requêtes ne
sont jamais groupées entre elles, contrairement à `/play_worlds`), et il est
occupé à ~87 % pendant un run. Conséquences, toutes deux trouvées en lançant :

1. **`--threads 6`, pas 32.** Il faut juste assez de fils pour tenir le sidecar
   occupé pendant que le CPU/GPU travaille (~1,3 s de calcul contre ~1,1 s
   d'attente ⇒ 2 à 3 fils). Au-delà, un fil n'ajoute que de la file. Le témoin
   qui le dit d'un coup d'œil : à 32 fils, `user 2m37s` pour `real 3m15s` —
   moins d'un cœur occupé sur 32.
2. **`timeout_ms = 120000` dans `[worlds]`.** Le délai doit couvrir **la file**,
   pas la requête : avec N matchs concurrents le N-ième attend N × ~1,1 s, donc
   le défaut de 6 s expire dès ~5 matchs en parallèle et l'arène s'arrête sur
   `read: Resource temporarily unavailable`.

Ces durées ne sont pas un mauvais réglage : ce sont les 11-20 s de la page
Annonces multipliées par le nombre de paroles d'une arène. Le budget qui les
ramène à quelque chose de raisonnable est `sims`, et la table de séparation
ci-dessous dit à partir d'où il cesse d'être du bruit.

⚠️ Le nombre de donnes par match dépend de la force du bot : le premier essai a
rendu **7,5 donnes par match** au lieu des ~13 habituelles, un bras qui chute
franchissant les 2000 points plus vite. Le coût par match bouge donc avec le
réglage, dans les deux sens.

## Ce que ça sépare, et le contrôle qui le dit

`quick_bid_spread.py` (2026-08-06) avait mesuré la dispersion de cette même
simulation : **σ ≈ 310 à 370 points par monde**, et l'écart *vrai* entre deux
annonces voisines (X contre X+10, même couleur) de **quelques points**,
compatible avec zéro à 600 simulations.

L'écart affiché entre la première et la deuxième candidate **ne peut pas servir
de preuve** : c'est un maximum sur des estimations bruitées, donc positif même
sous bruit pur. L'instrument est un contrôle — **deux tirages indépendants de la
même décision**, et on regarde s'ils désignent la même annonce plus souvent que
le hasard.

| budget (mondes × candidates) | A et B d'accord | hasard | erreur type mesurée |
|---|---|---|---|
| 20 × 4 | **2/20 (10 %)** | 25 % | ±85 pts |
| 80 × 4 | 9/20 (45 %) | 25 % | ±37 pts |
| 240 × 2 | 12/20 (60 %) | 50 % | ±23 pts |

**À 20 mondes le bidder tire au sort** — il ne corrige pas v6, il le randomise,
en mille fois plus cher qu'un appel au réseau. L'erreur type suit 1/√n (85 → 37
pour un facteur 4), donc 512 mondes rendent ±15. Le budget n'est pas un réglage
de confort, c'est la condition d'existence du bot.

## Les trois leviers

**Mondes partagés (`common random numbers`).** Les `sims` mondes sont tirés une
fois et rejoués par *chaque* candidate. L'appariement rend 1,25 à 1,38× sur
l'écart type de la différence (ρ mesuré à 0,36-0,48 : forcer une autre annonce
change toute l'enchère). Utile, pas décisif. Corollaire d'implémentation : la
boucle est « pour chaque monde, pour chaque candidate », jamais l'inverse — une
échéance qui coupe entre deux candidates laisserait les premières avec un monde
de plus, et le classement lirait ce déséquilibre comme un écart de valeur. C'est
aussi pourquoi la version parallèle **n'a pas d'échéance du tout**.

**Les mondes viennent de playgen** (`[worlds] source = "sidecar"`, `/auction_deals`).
Ce n'est pas un raffinement. Un tirage uniforme donne une main au hasard au siège
qui vient d'annoncer 100♥ : **le monde contredit l'enchère sous laquelle il est
tiré**, et v6 fera passer ce siège dans la suite de la simulation. Le contrat
simulé est alors systématiquement moins disputé qu'il ne le sera — biais de
sur-annonce, croissant avec la longueur de l'enchère déjà entendue. playgen v2
complète l'enchère avec sa propre tête d'annonces, donc les mains qu'il invente
**expliquent les annonces déjà entendues**. `rollout_probe_512_unif` est l'A/B
qui chiffre ce que ça vaut.

⚠️ C'est aussi une correction d'un décalage réel : sur la page Annonces,
`annonces_sim` échantillonne bien par playgen, mais `annonces_doudou` — le chemin
chaud, celui que la première version de ce bidder a copié — fait des mélanges
uniformes.

**Le choix des candidates (`candidate_mode`), qui décide de la question posée.**
C'est le levier qui compte : la simulation ne peut trancher que ce dont l'écart
dépasse son bruit.

- `top` — les meilleures au Q du réseau. **Le pire choix quand le réseau est
  confiant** : ses cinq meilleures sont alors cinq paliers de la même couleur
  (`120♦ 110♦ 100♦ 90♦` observé tel quel), c'est-à-dire exactement la question à
  laquelle la simulation ne sait pas répondre — et « passe » n'est pas en lice.
  Gardé pour l'A/B.
- `probe` (défaut) — **le sondage** : le réseau dit où chercher, la simulation
  regarde autour.
  1. l'annonce du réseau ;
  2. **passe** — l'alternative dont l'écart au reste est le plus grand, donc la
     seule que le budget puisse trancher à coup sûr ;
  3. la meilleure annonce de la **deuxième couleur** du réseau, même loin dans
     son classement — changer de couleur est une décision à grande amplitude, et
     un top-K ne la propose presque jamais ;
  4. deux voisines dans la couleur : −10 et +10 si les deux sont légales, sinon
     +10 et +20. On explore toujours deux paliers ; l'enchère en cours décide
     seulement lesquels.

## Réglages

```toml
[bid]
strategy = "rollout"
model = "models/bid_v6_isdd_resume/bid_nn_final.bin"  # a priori ET parole des 4 sièges en simulation
hidden = 512
score_aware = true
sims = 512                # mondes par décision
candidates = 5            # plafond (0 = pas de plafond)
candidate_mode = "probe"  # probe (défaut) | top
objective = "margin"      # margin (défaut) | winrate
parallel = false          # true pour le web, inutile en arène
time_ms = 0               # échéance par décision, 0 = pas d'horloge (ignorée si parallel)
# play_model / play_residual : par défaut ceux de [play] — on simule avec le joueur qu'on est

[play]
method = "dmc"
model = "models/play_v2/play_final.bin"
residual = true

[worlds]                  # même section que pour le jeu, même défaut sidecar
source = "sidecar"        # sidecar | playgen (CPU) | uniform
fallback = "strict"
```

`objective = "winrate"` maximise la fraction de mondes où mon camp marque plus
que l'autre — le chiffre-phare de la page Annonces. **Ce n'est pas le même
objectif** que `margin` : le barème est très asymétrique (une chute donne
162 + contrat à l'adversaire), donc un contrat qui passe souvent mais coûte cher
quand il tombe est meilleur sous `winrate` et pire sous `margin`. Offert pour
l'A/B, pas parce qu'il est le bon.

Budget **et** origine des mondes sont dans le label (`rollout:…@512x5+pg`) : deux
`rollout` à 128 et 512 mondes, ou l'un playgen et l'autre uniforme, ne sont pas
le même joueur, et `matches.csv` doit pouvoir les distinguer.

## Protocole

⚠️ **`playgen-up` d'abord**, et `playgen-down` à la fin — les bots par défaut
échantillonnent sur le sidecar et refusent de se construire sans lui
(`fallback = "strict"`). Cf. la discipline du sidecar dans `CLAUDE.md`.

```bash
# 1. Sonder une architecture — coût et reproductibilité (~2 min)
cargo run -p colver-core --release --features parallel --bin bench_bid_rollout -- \
    --deals 20 --sims 512 --candidates 5 --mode probe --worlds sidecar

# 2. Cribler, rien n'est écrit dans matches.csv
scripts/analysis/rollout_bid_sweep.sh 30

# 3. Valider la survivante, et elle seule
./target/release/arena h2h rollout_probe_512 v6_isdd_75M --matches 300 --threads 32
```

⚠️ **Un criblage ne couronne pas un gagnant.** À 30 matchs par direction la
statistique lisible est « Par donne » (~780 donnes, ±3,5 pp), pas le taux de
matchs (±6,5 pp) : ça élimine ce qui est franchement moins bon, ça ne distingue
pas deux architectures à 2 pp l'une de l'autre. Et cribler puis valider *le
meilleur du criblage* est un biais de sélection — il faut choisir la candidate
sur un argument, pas sur son classement au criblage.
