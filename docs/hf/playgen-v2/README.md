---
license: mit
language:
  - fr
tags:
  - belote
  - contree
  - coinche
  - card-games
  - imperfect-information
  - world-model
  - transformer
  - autoregressive
---

# Colver — playgen v2 : le réseau qui devine les mains cachées

Il répond à la question centrale de tout jeu de cartes : **« qui a quoi ? »**

C'est un transformer causal qui continue une donne carte par carte à partir de ce qu'un
joueur peut voir. Dérouler la continuation jusqu'au bout fait tomber les 32 cartes, donc
révèle une distribution complète et plausible des mains cachées. Il alimente l'agent à
recherche de [colver.net](https://colver.net) : échantillonner des mondes possibles, les
résoudre exactement, agréger.

[Colver](https://github.com/Avo-k/colver) est un moteur de Belote Contrée écrit en Rust,
utilisable depuis Python.

*Règles appliquées : [colver.net/regles](https://colver.net/regles) — et [pourquoi ces choix](https://colver.net/regles/choix).*

## Essayer en 30 secondes

```bash
pip install colver
```

```python
import random
import colver

RANGS = ["7", "8", "9", "V", "D", "R", "10", "A"]
COULEURS = "♠♥♦♣"                # une carte = couleur × 8 + rang
CARREAU = 2

# Une donne au hasard, reproductible
paquet = list(range(32))
random.Random(21).shuffle(paquet)
mains = [sorted(paquet[i * 8:(i + 1) * 8]) for i in range(4)]

# siège 0 : ♠ D      ♥ D 7      ♦ A R V 9   ♣ A
# siège 1 : ♠ V 9 8  ♥ 10 V     ♦ —         ♣ D 9 7
# siège 2 : ♠ A R 7  ♥ —        ♦ 8 7       ♣ R V 8
# siège 3 : ♠ 10     ♥ A R 9 8  ♦ 10 D      ♣ 10

env = colver.Env.deal_with_hands(dealer=3, hands=mains)
env.set_contract(trump=CARREAU, value=110, team=0, coinche=0)
env.set_phase_playing()

poids = colver.download_playgen_model()  # Hub → ~/.cache/colver/models/
analyste = colver.Analyst(poids)
analyste.init_deal(env, observer=0)      # on regarde depuis le siège 0

# p(carte -> siège) : 4 sièges × 32 cartes, chaque carte somme à 1
p = analyste.marginals(env, n_worlds=200)

for nom, c in [("A♦", 23), ("R♥", 13)]:
    print(nom, " ".join(f"siège{s}={p[s][c]:.2f}" for s in range(4)))
```

```
A♦  siège0=1.00  siège1=0.00  siège2=0.00  siège3=0.00
R♥  siège0=0.00  siège1=0.50  siège2=0.00  siège3=0.50
```

L'As de carreau est chez nous : le modèle en est **certain**, parce que nous l'avons en
main. Le Roi de cœur est ailleurs, et le modèle hésite entre deux sièges à parts égales.
C'est ce qu'on attend d'un modèle de croyances — la certitude là où elle est justifiée,
l'incertitude ailleurs.

Plus la donne avance, plus ces probabilités se resserrent.

⚠️ **Regardez toujours combien de mondes ont réellement survécu.** `marginals` ne le dit
pas : il agrège les mondes obtenus, quel qu'en soit le nombre. Si un seul survit, chaque
carte ressort à 1,00 — une postérieure d'apparence catégorique, calculée sur un unique
échantillon. Passez par `play_worlds` (ci-dessous) quand cette taille d'échantillon
compte.

## Tirer des donnes complètes plutôt que des probabilités

```python
mondes = analyste.play_worlds(env, n_worlds=100)

print(len(mondes))                     # -> ~10, jamais 100 : voir ci-dessous
print([len(m) for m in mondes[0]])     # -> [8, 8, 8, 8]  cartes restantes par siège

# Le tirage est aléatoire : deux appels ne rendent ni le même nombre ni les mêmes mondes.
```

Chaque monde est une donne entièrement déterminée, qu'un solveur double-dummy peut
résoudre exactement. C'est ce que consomme l'agent à recherche.

**On en reçoit toujours moins qu'on en demande.** Les continuations qui n'aboutissent pas
sont écartées, et la fonction rend `None` si aucune ne survit. Au premier pli — le cas le
plus dur, 24 cartes inconnues à placer — le rendement mesuré tourne autour de **10 %**,
et varie beaucoup d'une donne à l'autre : de 6 à 18 mondes pour 100 demandés selon la
main. Il remonte à mesure que la donne se vide et que les contraintes se resserrent. **Sur-commander est donc la norme**, pas une
précaution.

La main de l'observateur, elle, est toujours exacte : le modèle ne réinvente jamais ce
que le joueur voit.

## L'idée, pour qui vient d'ailleurs

Les réseaux de croyances classiques prédisent les emplacements de cartes en une passe,
mais ne rendent que des **marginales indépendantes** : ils ne peuvent pas représenter
« si l'Ouest a le Valet d'atout, il a probablement le 9 aussi ». La factorisation
autorégressive dilue la tâche sur beaucoup de jetons et **capture la structure jointe
gratuitement**. C'est la partie transférable à d'autres jeux à information imparfaite.

Deux choix de conception portent le reste :

- **Le masque fait l'arithmétique d'ensembles**, pas le modèle. Coupes révélées, plafonds
  d'atout, cartes déjà tombées : tout ça est calculé par le moteur et imposé au softmax,
  à l'entraînement comme à l'inférence. Le réseau n'apprend que l'inférence et la
  stratégie — jamais à compter des cartes.
- **L'acteur du coup suivant est donné, pas prédit.** Il est déterministe depuis les
  règles, donc le modèle ne dépense aucune capacité à calculer qui remporte un pli.

## Le modèle en bref

| | |
|---|---|
| Type | transformer causal, `d=384`, 6 couches, 8 têtes |
| Sorties | 32 voies (carte) **+ 43 voies (enchère)** |
| Taille | 43 Mo — 10,74 M paramètres |
| Entraînement | 9 M de donnes, 160 K pas, ~20 h sur RTX 3090 |
| Vitesse | ~93 ms par monde sur CPU ; en pratique servi par GPU |
| sha256 | `3cb43a8cae84…` |

Contrairement aux autres modèles de Colver, celui-ci **porte un en-tête** (`COLVPG02`) :
son format est identifiable depuis le fichier.

## Ce qu'il vaut

Facteur de branchement effectif sur les sièges cachés — combien de cartes différentes le
modèle juge plausibles à chaque instant. Plus bas est meilleur ; « uniforme » est le
tirage au hasard parmi les coups légaux, c'est-à-dire l'arithmétique d'ensembles seule.

| pli | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| playgen v2 | 4,76 | 4,47 | 4,32 | 3,94 | 3,35 | 2,63 | 1,88 | 1,13 |
| uniforme | 22,96 | 19,57 | 16,27 | 13,04 | 9,89 | 6,90 | 4,10 | 1,56 |
| **gain** | **4,82×** | 4,38× | 3,77× | 3,31× | 2,95× | 2,63× | 2,18× | **1,38×** |

Le gain **décroît de bout en bout** : 4,8× au premier pli, 1,4× au dernier. En fin de
donne les contraintes dures font presque tout le travail, et tirer au hasard suffit
à peu près.

Sa tête d'enchère est assez bonne pour servir **directement de bidder** : utilisée telle
quelle, elle atteint 48,2 % en h2h sur 3000 matchs contre
[Bid v6](https://huggingface.co/Avo-k/colver-bid-v6), alors qu'elle n'a jamais été
entraînée pour annoncer et qu'elle ne voit pas le score de la partie.

## Ce qu'il ne sait pas faire

1. **Il ne voit pas l'annonce de belote.** L'annonce est publique, mais elle n'existe
   nulle part dans ce qu'on lui donne à lire : il voit tomber un Roi d'atout, jamais le
   fait que le joueur a annoncé en le posant. Résultat mesuré : **15 à 16 % des mondes
   qu'il produit aux positions concernées sont impossibles** et doivent être rejetés en
   aval. (Un tirage uniforme en produit 40,1 % — il en a donc appris une bonne partie
   tout seul, mais pas tout.)
2. **Il est saturé en capacité sur la tête d'enchère.** Un modèle 3,3× plus petit, avec
   8 % du budget de données, arrive à 1,6 % de celui-ci. Grossir n'est pas la piste.
3. **Il ne voit pas le score de la partie**, donc ne peut pas modéliser une fin de partie
   serrée.
4. À 43 Mo il ne tient plus en cache L3 : l'inférence CPU est limitée par la mémoire.

## Servir le modèle sur GPU

Pour un usage intensif, Colver fournit un sidecar HTTP :

```bash
playgen_gpu_server --playgen playgen_v2_final.bin --port 8003
```

```toml
[play]
method = "isdd"
[worlds]
source = "sidecar"
url = "http://localhost:8003"
```

Détails de tokenisation et d'architecture :
[docs/belief/playgen.md](https://github.com/Avo-k/colver/blob/master/docs/belief/playgen.md).

## Règles

Corpus généré avant le correctif de règle du 2026-08-02, qui élargit légèrement
l'ensemble des coups légaux : 0,076 % des décisions concernées, 0,014 % effectivement
changées.

## Licence

MIT.

```bibtex
@software{colver,
  author = {Avo-k},
  title  = {Colver: a Belote Contrée engine with RL agents},
  url    = {https://github.com/Avo-k/colver}
}
```
