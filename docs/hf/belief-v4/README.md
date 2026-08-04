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
  - belief-state
---

# Colver — belief v4 : où sont les cartes ?

À partir de ce qu'un joueur voit, il estime **la probabilité que chaque carte soit dans
chaque main adverse**. C'est la couche de croyances de l'agent à recherche de
[colver.net](https://colver.net) : elle pondère l'échantillonnage des mains possibles.

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

poids = colver.download_belief_model()   # Hub → ~/.cache/colver/models/

croyances = colver.Beliefs.replay(
    dealer=3, hands=mains, actions=[], observer=0, belief_model=poids
)
p = croyances.weights(env)      # {"nn": 4×32, "heuristic": 4×32}

as_pique = 7                    # A♠ — une carte que nous ne voyons pas
print("réseau     ", " ".join(f"{p['nn'][s][as_pique]:.2f}" for s in range(4)))
print("heuristique", " ".join(f"{p['heuristic'][s][as_pique]:.2f}" for s in range(4)))
```

```
réseau      0.00 0.32 0.44 0.24
heuristique 0.00 0.33 0.33 0.33
```

Pour l'As de pique (A♠) : le siège 0 est exclu, nous savons que nous ne l'avons pas.
L'heuristique répartit le reste **à plat**, faute de mieux. Le réseau penche vers le
siège 2 — qui le détient effectivement dans cette donne.

Ne lisez pas ce coup d'œil comme une mesure de justesse : sur une carte, avoir raison est
un coup de chance à une chance sur trois. Ce que l'exemple montre, c'est que le réseau a
une *opinion* là où l'heuristique n'en a aucune. Sa qualité réelle, c'est sa perte de
validation.

## Le modèle en bref

| | |
|---|---|
| Type | MLP, 304 entrées → 96 sorties (32 cartes × 3 mains cachées) |
| Taille | 1,9 Mo — 470 112 paramètres |
| Perte de validation | 0,8797 |
| sha256 | `6d141252ea8b…` |

## Ce qu'il ne sait pas faire

**Il rend des probabilités indépendantes carte par carte.** Il ne peut donc pas
représenter les corrélations qui font l'essentiel de la lecture d'une main — « si l'Ouest
a le Valet d'atout, il a probablement le 9 aussi ». C'est une limite d'architecture, pas
d'entraînement : la sortie est une liste de marginales, il n'y a pas de place pour dire
que deux cartes voyagent ensemble.

C'est exactement ce qui a motivé
[playgen](https://huggingface.co/Avo-k/colver-playgen-v2), un transformer autorégressif
qui échantillonne des mains entières et capture la structure jointe gratuitement.
Playgen est aujourd'hui la source de mondes par défaut ; ce réseau-ci reste utile comme
couche de pondération, bien moins chère (1,9 Mo contre 43 Mo).

**Il ne porte pas les certitudes.** Coupes révélées, plafonds d'atout, cartes déjà
tombées, belote annoncée : ce sont des *faits*, appliqués directement par le moteur sans
passer par aucun modèle. Ce réseau ne fournit que le doux, jamais le dur.

## Ce qui n'a pas été mesuré

Point d'honnêteté, parce que la confusion est facile.

Le gain d'arène souvent cité pour « le belief net » de Colver — **54,8 %, +111 pts sur
1000 matchs** — appartient à un **autre fichier**, `belief_v3.bin`, qui est celui que
charge la configuration du bot en question. Il **n'est pas mesuré pour ce modèle-ci** :
aucun h2h dédié n'a été fait pour `belief_v4_fix_v2`. Ce qu'on en sait est sa perte de
validation, et le fait qu'il ait été réentraîné après une correction de contraintes
(`TrumpCeilingTracker`, 2026-07-21) qui rendait fausses celles vues par son prédécesseur.

Ce qui est établi plus largement : un réseau de croyances *réellement consulté* vaut
+3 à +6 points de pourcentage. Un premier résultat à 0 pp s'expliquait par un défaut de
câblage — le réseau était chargé mais jamais interrogé.

## L'utiliser dans un bot

```toml
[play]
method = "isdd"

[belief]
model = "belief_v4_fix_v2.bin"
```

Les croyances douces sont **désactivées par défaut** : il faut les demander.

## Détail à connaître si vous écrivez votre propre chargeur

Le fichier s'appelle « v4 » mais attend l'observation **V2, de 304 flottants** — pas la
V3 de 380 vers laquelle le nom oriente. Le « v4 » désigne la génération d'entraînement,
pas la disposition d'entrée. La bibliothèque s'en sort en essayant les combinaisons
connues ; un chargeur maison peut se tromper silencieusement. Référence :
[`belief_obs.rs`](https://github.com/Avo-k/colver/blob/master/colver-core/src/belief/belief_obs.rs).

Un modèle frère, `bid_belief_v4.bin`, fait le même travail pendant l'**enchère**
(log(p) de −0,9565, contre −1,0209 pour l'heuristique). Il n'est pas publié ici.

## Règles

Entraîné avant le correctif de règle du 2026-08-02, qui élargit légèrement l'ensemble des
coups légaux : 0,076 % des décisions concernées, 0,014 % effectivement changées.

## Licence

MIT.

```bibtex
@software{colver,
  author = {Avo-k},
  title  = {Colver: a Belote Contrée engine with RL agents},
  url    = {https://github.com/Avo-k/colver}
}
```
