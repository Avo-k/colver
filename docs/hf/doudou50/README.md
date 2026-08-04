---
license: mit
language:
  - fr
tags:
  - belote
  - contree
  - coinche
  - card-games
  - reinforcement-learning
  - imperfect-information
  - dueling-dqn
  - deep-monte-carlo
---

# Colver — DouDou50 : le réseau qui joue les cartes

Donnez-lui une position, il vous dit **quelle carte jouer**, en une demi-milliseconde et
sans aucune recherche. C'est le joueur du mode « rapide » de
[colver.net](https://colver.net).

Colver est un moteur de Belote Contrée écrit en Rust, utilisable depuis Python.

[Code source](https://github.com/Avo-k/colver) · [PyPI](https://pypi.org/project/colver/) · [Jouer en ligne](https://colver.net)

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

def nom_carte(c):
    return RANGS[c % 8] + COULEURS[c // 8]

env = colver.Env.deal_with_hands(dealer=3, hands=mains)

# On saute l'enchère : contrat de 110 à carreau, pris par Nord-Sud
env.set_contract(trump=CARREAU, value=110, team=0, coinche=0)
env.set_phase_playing()

poids = colver.download_model()          # Hub → ~/.cache/colver/models/
env.load_dmc_model(poids)

reponse = env.action_dmc_with_stats()
print(nom_carte(reponse["best_action"]))   # -> V♦
print(reponse["elapsed_ms"])               # -> ~0.5
```

Il entame **Valet de carreau** — la plus forte carte à l'atout. En une demi-milliseconde.

La carte est reproductible : le réseau est déterministe, seul le chrono bouge.

> ⚠️ La réponse contient aussi `reponse["q_values"]`, les 32 valeurs brutes. **Ne les
> décodez pas comme des cartes** : `best_action` est reconverti en indice physique, mais
> `q_values` reste dans l'espace **canonique** du réseau (atout en couleur 0, les trois
> autres réordonnées). Les deux ne sont donc pas dans le même référentiel, et Python
> n'expose pas la conversion. Utilisez `best_action` ; le classement complet n'est
> exploitable qu'en Rust pour l'instant.

## Le faire jouer une donne entière

```python
spec = f'''
[bid]
strategy = "improved_v2"

[play]
method = "dmc"
model = "{poids}"
residual = true
'''
bot = colver.Agent(spec, seat=0)
bot.init_deal(env)
print(bot.action(env))
```

**`residual = true` n'est pas optionnel** — voir plus bas.

## Le modèle en bref

| | |
|---|---|
| Type | Dueling DQN résiduel (MLP), 411 entrées → 32 valeurs Q |
| Taille | 10,2 Mo — 2 561 057 paramètres |
| Entraînement | 50 M pas de self-play |
| Vitesse | ~0,5 ms par coup, sur CPU |
| sha256 | `f9fb4c4bc9ea…` |

En entrée : la main, le pli en cours, les cartes déjà tombées, les coupes révélées, le
contrat, les scores et l'historique d'enchères. Aucune recherche à l'inférence — c'est
un réseau qui répond, pas un agent qui réfléchit.

C'est cette vitesse qui en fait aussi le générateur du corpus de 9 M de donnes ayant
entraîné le modèle de mondes
[playgen v2](https://huggingface.co/Avo-k/colver-playgen-v2).

## Ce qu'il vaut

Son erreur absolue moyenne face au solveur double-dummy est de **~19 points**.

Le fait intéressant est ailleurs : l'agent à recherche du même projet (IS-DD) a une
erreur moyenne comparable, mais **les deux se trompent différemment**. Sur une même
donne, ils divergent sur **29 cartes sur 32**. DouDou50 encaisse ses As tout de suite,
IS-DD tire les atouts. Deux joueurs d'égale force moyenne ne jouent pas le même jeu.

## Ce qu'il ne sait pas faire

- **Il ne raisonne pas sur les mains cachées.** Face à l'agent IS-DD — qui échantillonne
  des mondes possibles et les résout exactement — il perd nettement. C'est un compromis
  de latence assumé : 0,5 ms contre 1,2 s par coup.
- **Il a appris contre un bidder plus ancien** (v2, pas le v6 actuel), donc la
  distribution de contrats qu'il a vue n'est plus tout à fait celle de la production.

## Le piège à connaître

Les mêmes poids se lisent comme un MLP simple **ou** comme un réseau résiduel, et rien
dans le fichier ne dit lequel. Sans les connexions résiduelles, l'erreur face au solveur
passe de ~19 à **~25 points** — sans la moindre exception levée, juste un jeu plus
faible. D'où `residual = true` dans la spec, ou `net.set_residual(true)` en Rust.

De même, ce modèle attend une observation **canonique** : les couleurs sont réordonnées
pour que l'atout occupe toujours l'emplacement 0. La bibliothèque s'en charge ; si vous
écrivez votre propre chargeur, il faut convertir le masque légal vers cet espace et
reconvertir la carte choisie — sans quoi le réseau joue au hasard parmi les coups légaux.

Détails dans
[`dmc_net.rs`](https://github.com/Avo-k/colver/blob/master/colver-core/src/dmc/dmc_net.rs)
et [`dmc_obs.rs`](https://github.com/Avo-k/colver/blob/master/colver-core/src/dmc/dmc_obs.rs).
Un modèle à 415 entrées est l'ancien DouDou35, non résiduel et à observation physique.

## Règles

Barème « points faits + demandés », base 162 (252 sur capot réalisé), contre ×2 et
surcontre ×3 sur la valeur du contrat seule, aucun arrondi.

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
