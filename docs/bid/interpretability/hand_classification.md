# Classification des mains — index canonique et code lisible

*Écrit le 2026-08-02. Module : [`colver-core/src/hand_class.rs`](../../../colver-core/src/hand_class.rs),
exposé en Python (`colver.hand_class_id`, `colver.hand_code`, `colver.matadors`).*

Les quatre couleurs sont interchangeables tant qu'aucun atout n'est nommé : une main et
la même main couleurs échangées sont la même position. Tout ce document en découle.

Deux couches, qui ne servent pas à la même chose et qu'il ne faut pas confondre.

| | quoi | pour quoi |
|---|---|---|
| **Index canonique** | exact, sans perte, bijectif | clé de table, déduplication, **énumération** de l'espace |
| **Code** | avec perte, lisible | stratifier un générateur, lire les décisions d'un bidder |

---

## 1. L'index canonique

`hand_class_id(main) → [0, 472 579)` et `hand_class_id_trump(main, atout) → [0, 1 820 803)`
(à atout désigné le groupe tombe de S₄ à S₃). Deux mains équivalentes ont le même index,
deux mains inéquivalentes en ont deux différents, et `hand_from_class_id{,_trump}`
parcourt la bijection en sens inverse.

**Les deux constantes ne sont pas codées en dur** : elles tombent d'une table de
dénombrement calculée en `const fn` à la compilation, et un test vérifie qu'elles valent
les nombres de Burnside. Ce n'est pas `10 518 300 / 24` — **7,5 % des mains ont une
symétrie de couleur**, et ce quotient n'est même pas entier. Distribution des tailles
d'orbite pour une main :

| taille d'orbite | orbites | ce que c'est |
|---|---:|---|
| 1 | 28 | deux cartes par couleur, **mêmes rangs partout** (= C(8,2) ✓) |
| 4 | 1 205 | même motif dans trois couleurs |
| 6 | 896 | deux paires de couleurs jumelles |
| 12 | 65 227 | deux couleurs jumelles |
| 24 | 405 223 | aucune symétrie |

Pour une donne complète 4×8 la symétrie exige que *les quatre joueurs* aient le même
motif dans deux couleurs : moins de 0,005 % des donnes, et le rapport monte à 23,99885 —
presque 24, mais pas 24. D'où **4 148 577 738 928 080 donnes distinctes**.

### Ce que l'énumérabilité débloque

Premier à parler, historique vide, score 0-0 → l'annonce est une **fonction pure de la
main**. La politique d'ouverture d'un bidder est donc une table de 472 579 entrées, qu'on
peut écrire en entier. Ce n'est pas une approximation type SHAP, c'est *le modèle*. On
peut trier par classe, croiser avec le DD, et lire directement où il se trompe.

⚠️ **Impossible tant que le bidder n'est pas équivariant.** v6 donne jusqu'à 24 réponses
différentes pour une même classe — mesuré par `scripts/analysis/bid_equivariance.py` —
donc la table n'est pas bien définie. La canonicalisation de l'observation d'enchère
(`write_bid_observation_canonical`, dans `colver-core/src/bid/bid_obs.rs`) est la
dépendance : un réseau entraîné dessus est équivariant par construction.

### Détail d'implémentation qui vaut d'être connu

Le rang se calcule en O(4), pas en O(4×256). Un uplet canonique étant décroissant,
« son premier élément est `< m` » équivaut à « tous ses éléments sont `≤ m−1` » : les
uplets qui précèdent sont comptés d'un seul coup par `COUNTS[k][m−1][p]`. Sans cette
identité, les tests exhaustifs coûtaient 551 s en debug ; avec, 236 s — et **0,37 s en
release**, d'où le `#[ignore]` sur les trois tests qui parcourent tout l'espace.

---

## 2. Combien vaut une carte ? (la mesure qui décide du code)

Le contenu du code n'est pas une opinion de joueur. Il vient d'une mesure appariée sur le
solveur DD : donne aléatoire, on échange une carte de la main contre **la plus faible
carte de la même couleur détenue par un adversaire**, on re-résout. Le reste de la donne
est identique entre les deux solves, donc la variance de la donne s'annule.

Perte moyenne en points DD (échelle 0-252), 600 donnes pour l'atout, 400 pour le côté :

| à l'atout | | à côté | |
|---|---|---|---|
| **J** | **+49,2** | **A** | **+26,0** |
| **9** | **+18,9** | **10** | **+6,3** |
| A | +9,5 | K | +1,5 |
| 10 | +5,6 | 9 | +0,0 |
| K | +1,8 | 8 | +0,0 |
| Q | +0,9 | D | −0,1 |
| 8 | +0,4 | **J** | **−0,5** |

*IC 95 % ≈ ±1 à ±2 sur les grosses lignes, ±0,1 sur les petites.*

**Il y a une falaise, pas une pente.** Deux cartes d'atout portent 68 des ~86 points
d'importance de la couleur. À côté, tout ce qui est sous le 10 est du bruit statistique :
la Dame vaut −0,1 ± 0,2, c'est-à-dire rien.

Curiosité qui tient la mesure : **le Valet de côté est négativement valorisé**
(−0,51 ± 0,07). Le donner à l'adversaire rapporte un demi-point — ce sont 2 points de
mangeaille qu'on récupère au pli plutôt que de les défausser soi-même.

### ⚠️ Le piège du protocole, et il n'est pas anodin

« La plus faible carte de la couleur » n'est pas la même selon le rôle de la couleur. À
l'atout l'ordre est **J 9 A 10 K Q 8 7**, et le Valet a un indice de rang *bas* (3) tout
en étant la carte la plus forte. Sélectionner le remplaçant par l'ordre naturel produit
donc parfois « on donne son Roi et on reçoit le Valet » — un échange qui *améliore* la
main.

Symptôme : les lignes K et Q de l'atout ressortaient **négatives** (−4,2 et −4,6), et
l'As et le 10 étaient écrasés vers zéro. D'où deux passes séparées, chacune avec son
ordre, chacune ne lisant que le solve où la couleur a le bon rôle. **Contrôle qui
l'attrape : chaque colonne doit être monotone dans l'ordre de force de son rôle.**

### Portée

Valeur **marginale, sur fond aléatoire, en jeu parfait**, donneur fixé. Ce n'est pas une
valeur d'enchère : le J d'atout ne vaut pas 49 dans toutes les mains, il vaut ça en
moyenne contre un remplaçant faible.

---

## 3. Le code

Atout désigné, insensible aux couleurs, emboîté du grossier au fin via `coarsen(level)`.
`HandCode` est `Copy + Eq + Hash + Ord` : regrouper, c'est `coarsen` puis `HashMap`.

```
T5.J9AT.A1/A1/x1        cinq atouts, J-9-A-10 détenus, deux As secs et une basse
T8.J9AT.-/-/-.B         les huit carreaux (belote comprise)
```

| composante | contenu | justification |
|---|---|---|
| `T<n>` | longueur d'atout | porte les longues et les coupes franches |
| `J9AT` | lesquels des quatre gros atouts | les seuls > 5 pts DD |
| `A1` `AT3` `T2` `x4` | par couleur de côté : As / Dix / longueur | l'As (+26) et le Dix (+6,3) ; le reste est sous le point |
| `.B` | belote (K+Q d'atout) | **invisible au DD** — 0 point carte, mais 20 de marque |

Ce qui est **délibérément absent** : Dame, Valet, 9, 8, 7 de côté, tous mesurés sous le
point. Et le rang exact des cartes de côté au-delà de l'As et du Dix.

⚠️ **Absence mesurée coûteuse, découverte le 2026-08-03** : le code décrit une main
**avec un atout désigné**, or une décision d'enchère *compare des atouts candidats*. Le J
et le 9 d'une couleur de côté — ce qui déciderait d'y jouer l'atout — sont perdus, et ce
sont les cartes qui pèsent +49 et +19 dès qu'on les regarde comme atout. Ajouter au code
le descripteur de la deuxième meilleure couleur (longueur + lesquels de J/9/A/10) vaut
**+1,5 pt à l'ouverture et +22,7 pt en défense** dans une table de correspondance sur la
politique de v6. En défense, `trump+2e` (2 647
codes) **bat `full` (5 028 codes) de 9,7 points** : moitié moins de familles, dix points
de mieux.

La suite `length → trump → shape → tops → full` raffine le **côté** ; il manque un **axe
orthogonal** qui porte les autres atouts possibles. Ce n'est pas un cran de plus sur le
même axe — il en faut moins et il explique davantage.

C'est bien un angle mort **du code**, et de lui seul : les 17 features agrégées de la
distillation se calculent sur *la meilleure couleur de la main*, donc elles n'ont jamais
perdu cette information (+0,8 point en défense quand on la leur ajoute explicitement).
`HandCode` la perd parce qu'il faut lui *désigner* un atout, et qu'en défense l'atout
désigné est celui de l'adversaire.

### Granularité, comptée exactement sur les 10 518 300 mains

| niveau | codes | 50 % des mains dans | 90 % dans | plus gros code |
|---|---:|---:|---:|---:|
| `length` | 9 | 2 | 4 | 35,83 % |
| `trump` | 80 | 8 | 28 | 13,16 % |
| `shape` | 339 | 28 | 122 | 5,01 % |
| `tops` | 5 277 | 388 | 1 927 | 0,62 % |
| `full` | 6 654 | 420 | 2 281 | 0,51 % |

**~80 codes suffisent à décrire ce qui décide de l'annonce, et 28 couvrent 90 % des
mains.** On passe de 472 579 classes exactes à quelques dizaines de familles lisibles en
n'ayant jeté que des cartes mesurées sous 2 points.

### Matadors

`matadors(main, atout)` rend la valeur **signée** à la manière du « mit N / ohne N
Spitzen » du Skat : longueur de la série ininterrompue des plus gros atouts, positive si
on la détient, négative si l'adversaire la détient. `0` est impossible — soit on a le
Valet, soit on ne l'a pas. `HandCode::top_run()` en est la version plafonnée à 4, puisque
le code ne garde que J-9-A-10.

---

## 4. D'où vient ce schéma (précédents)

**Le Skat** est le précédent le plus direct : ses *Spitzen* comptent exactement la série
des plus gros atouts, et ce nombre **multiplie la valeur du contrat** — c'est le barème
officiel, pas une heuristique de joueur. La mesure du §2 le valide indépendamment : la
masse est bien concentrée en tête de série.

**Le bridge** apporte deux briques transposables et une qui ne l'est pas :
- la **notation de forme** (5-4-3-1, longueurs triées) — insensible aux couleurs par
  construction ; sur 8 cartes il n'y a que 15 formes ;
- le **Losing Trick Count** (perdantes parmi les cartes du haut de chaque couleur), dont
  l'analogue contrée se compte dans l'ordre d'atout ;
- les **points Milton (4-3-2-1)** en revanche **ne transposent pas** : ils supposent que
  la force suit l'ordre naturel des rangs, ce que l'atout casse entièrement. Le bridge
  l'a d'ailleurs appris à ses dépens — Zar Points, Bergen, contrôles sont autant de
  rustines empilées sur un scalaire trop pauvre. Argument pour mesurer plutôt que
  sculpter.

**Le poker** donne la méthode, pas le contenu : les solveurs font d'abord une
**isomorphie de couleurs** (canonicalisation exacte, sans perte), *puis* un **bucketing
par similarité d'issue**. La leçon : la couche exacte se fait par construction, la couche
grossière se fait en clusterisant sur des issues mesurées, pas sur des traits inventés.
C'est la structure adoptée ici — et c'est aussi ce qui reste à faire, cf. §6.

---

## 5. À quoi ça sert

1. **Stratifier un générateur.** Le code nomme les strates. Attention : l'espace des
   mains est déjà saturé (5M donnes = 20M mains, le coupon collector n'en demande que
   6,2M), donc stratifier **sur l'issue**, pas sur la main — cf. §3.5 du plan v7.
2. **Nommer les familles d'une suite de sondes** (§3.2 du plan) : capot forcé, 8 atouts,
   coupe franche + longue, mains limites.
3. **Lire une politique d'annonce**, une fois l'obs canonicalisée (§1 ci-dessus, §3.6 du
   plan). Sans attendre la canonicalisation, le code sert déjà à **écrire** la politique
   et à **localiser** son résidu : « 93 % d'accord » devient une liste de familles où la
   règle tombe, donc un domaine de validité. Ordre de grandeur : **73 familles retrouvent
   87 % des décisions annoncer/passer de v6** à l'ouverture, à 7 points d'un XGBoost qui
   n'est pas lisible du tout.
4. **Dédupliquer** : deux donnes tirées au hasard ne sont jamais dans la même classe
   (une sur 830 millions), donc l'augmentation par les 24 permutations de
   [`suit_perm.rs`](../../../colver-core/src/suit_perm.rs) multiplie effectivement le
   corpus par ~24 sans doublon — les collisions sont de l'ordre de 0,005 %.

---

## 6. Quel niveau de code choisir — tranché par la mesure

**Fait le 2026-08-03**, dans les deux sens, et le verdict est net.

### 6.1 Contre la valeur DD — `tops` sature, et la main ne dit qu'un quart

60 000 mains × **3** répartitions des 24 autres cartes × 4 atouts = 720 000 solves
(`scripts/analysis/hand_code_dd_variance.py`, non publié). Les
répétitions sont ce qui rend la mesure lisible : sans elles on ne peut pas séparer « le
code est grossier » de « la donne est bruitée ». Variance expliquée, estimée par ANOVA
(l'ICC, pas un R² brut — un R² monte mécaniquement avec le nombre de groupes) :

| niveau | codes | atout ancré | meilleur atout |
|---|---:|---:|---:|
| **la main elle-même** | 60 000 | **23,5 %** | **17,1 %** ← plafond |
| `length` | 7 | 8,6 % | 4,9 % |
| `trump` | 73 | 15,7 % *(67 % du plafond)* | 7,8 % *(46 %)* |
| `shape` | 252 | 16,6 % *(71 %)* | 9,6 % *(56 %)* |
| `tops` | 2 983 | 23,9 % *(saturé)* | 17,1 % *(saturé)* |
| `full` | 3 703 | 24,0 % *(saturé)* | 17,1 % *(saturé)* |

Trois lectures.

1. **La main n'explique que 23,5 % de la variance de sa propre valeur DD** (17,1 % pour
   le meilleur atout). Les trois quarts restants sont la répartition des 24 autres
   cartes. Aucune évaluation de main, aussi parfaite soit-elle, ne peut faire mieux —
   c'est le chiffre qui dit pourquoi l'enchère est un jeu de communication et non
   d'évaluation.
2. **`tops` sature le plafond** ; `full` n'ajoute rien. C'est exactement ce que la
   conception du code prédit : `full` ne rajoute que la belote, qui vaut **0 point
   carte** et est donc invisible au DD par construction. Une confirmation interne, pas
   une coïncidence.
3. **`trump` (73 codes) capte les deux tiers de ce que la main peut dire** de la valeur à
   l'atout ancré, mais moins de la moitié pour le meilleur atout — encore l'angle mort
   des autres couleurs.

Les 102 % de `tops` sont du bruit d'estimation : un grossissement de la main ne peut pas
expliquer plus que la main. Deux estimateurs non biaisés qui se croisent à 2 % près.

### 6.2 Contre la politique de v6 — et ce n'est pas la même réponse

Précision hors échantillon d'une table `code → une réponse`, niveau par niveau.
Le classement diffère de celui du §6.1, ce qui est
le point : imiter un bidder et prédire une valeur ne sont pas la même cible. En défense,
`trump+2e` (2 647 codes) **bat `full` (5 028 codes) de 9,7 points** — impossible si les
deux cibles étaient interchangeables.

### 6.3 Ce qui reste ouvert

Le sens inverse : clusteriser les classes sur leur **vecteur** de valeurs DD (une par
atout) et regarder si les clusters retombent sur le code. Là où ils divergent, on aurait
une composante manquante trouvée par les données plutôt que par le raisonnement. Les
deux mesures ci-dessus en désignent déjà une (la qualité d'atout des autres couleurs) ;
rien ne dit qu'elle soit la seule.

---

## Reproduction

```bash
# concentration des codes (lit le binding Rust, ~1 min)
uv run python scripts/analysis/hand_classes.py
# dénombrements re-vérifiés par force brute (quelques minutes)
uv run python scripts/analysis/hand_classes.py --verify
# importance des cartes (~5 min sur 32 cœurs)
uv run python scripts/analysis/card_importance.py --deals 600

# §6.1 — variance DD expliquée par niveau (~31 min, 8 threads, 720 000 solves)
uv run python scripts/analysis/hand_code_dd_variance.py --hands 60000 --reps 3 --tag dd-variance

# nombre de codes par niveau, vérifié côté Rust
cargo test -p colver-core --release --lib hand_class -- --ignored
```

Le nombre de codes (9 / 80 / 339 / 5 277 / 6 654) est épinglé par le test Rust ; les
colonnes de concentration viennent du script Python, qui interroge le binding plutôt que
de réimplémenter la logique.

**Les comptes de codes du §6 sont plus bas que ceux du §3.3** (73 au lieu de 80, 2 983 au
lieu de 5 277) : le §3.3 énumère *tous* les codes possibles, le §6 ne voit que ceux qui
sortent d'un tirage de 60 000 mains, et l'ancre y est toujours la couleur la mieux notée
— les codes qui décrivent une main sous un atout qu'on ne choisirait jamais n'y
apparaissent pas.
