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
---

# Colver — Bid v6 : le réseau qui annonce

Donnez-lui une main de 8 cartes, il vous dit **quoi annoncer**. C'est le modèle
d'enchère qui joue contre vous sur [colver.net](https://colver.net), et la référence
contre laquelle tout nouveau bidder du projet est mesuré.

Colver est un moteur de Belote Contrée écrit en Rust, utilisable depuis Python.

[Code source](https://github.com/Avo-k/colver) · [PyPI](https://pypi.org/project/colver/) · [Jouer en ligne](https://colver.net)

*Règles appliquées : [colver.net/regles](https://colver.net/regles) — et [pourquoi ces choix](https://colver.net/regles/choix).*

## Essayer en 30 secondes

```bash
pip install colver
```

```python
import colver

env = colver.Env.deal(dealer=1, seed=35)   # donne reproductible ; Sud ouvre les enchères
# Sud tient : ♠ —   ♥ 10 V   ♦ R D V 9   ♣ R 8

poids = colver.download_bid_model()        # Hub → ~/.cache/colver/models/
env.load_bid_model(poids)

reponse = env.action_bid_nn()
print(colver.Env.action_name(reponse["best_action"], 0))     # -> 120♦
```

Le modèle annonce **120 à carreau**. C'est là qu'est la main : quatre cartes dont le
Valet et le 9, les deux plus fortes à l'atout, avec le Roi et la Dame derrière.

## Voir *toutes* ses préférences, pas seulement son choix

```python
top = sorted(reponse["q_values"], key=lambda x: -x[1])[:5]
for action, q in top:
    print(f"{colver.Env.action_name(action, 0):>6}  {q:.3f}")
```

```
  120♦  0.136
  110♦  0.130
  100♦  0.108
  130♦  0.104
   90♦  0.089
```

Les trois premières valeurs tiennent dans **0,028** — c'est un défaut connu et mesuré du
modèle, décrit plus bas. Le classement dit clairement « carreau, autour de 120 » ; il ne
dit pas grand-chose sur le choix exact entre 100 et 130.

Le modèle est entièrement déterministe : mêmes cartes, mêmes valeurs, à la décimale près.

## Lire la réponse

`Env.action_name(action, 0)` s'en charge, mais le codage tient en quatre lignes :

| Action | Signification |
|---|---|
| `0` | Passe |
| `1` à `36` | annonce — `valeur = 80 + (a-1)//4 × 10`, `couleur = (a-1) % 4` |
| `37` à `40` | capot, une par couleur |
| `41` / `42` | contre / surcontre |

Les couleurs sont numérotées **♠ 0, ♥ 1, ♦ 2, ♣ 3**, partout et pour tout — cartes comme
annonces.

`action_bid_nn` rend un `best_action` **et** des `q_values` déjà restreints aux actions
légales de la position — 41 entrées sur 43 à l'ouverture ci-dessus, jamais les 43. Si
vous chargez les poids vous-même à partir de `env.get_bid_observation()`, en revanche, le
réseau note bien les 43 actions, y compris celles qui sont interdites : masquez par
`env.legal_actions()` avant de prendre l'argmax.

## Le faire jouer une donne entière

Plutôt que de piloter le réseau à la main, on peut asseoir un bot complet :

```python
spec = f'''
[bid]
strategy = "nn"
model = "{poids}"
hidden = 512
score_aware = true
'''
bot = colver.Agent(spec, seat=2)
bot.init_deal(env)
print(bot.action(env))
```

`score_aware = true` compte : ce modèle voit le score de la partie et n'annonce pas
pareil à 900-200 qu'à 0-0.

## Le modèle en bref

| | |
|---|---|
| Type | Dueling DQN (MLP), 117 entrées → 43 valeurs Q |
| Taille | 2,4 Mo — 611 372 paramètres |
| Entraînement | 75 M pas, contre un pool de 5 M de donnes étiquetées par IS-DD |
| Vitesse | quelques microsecondes par décision, sur CPU |
| sha256 | `9443671cab1e35bb…` |

En entrée : la main, l'historique des annonces, la position à table, le score de la
partie et 4 bits « j'ai la belote ». Le modèle **joue ses propres enchères** pendant
l'entraînement — l'oracle ne fait que superviser la perte. Le faire annoncer à
l'oracle produirait des enchères dégénérées (annonce optimale → 3 passes) qui
n'apprennent rien de la dimension de dialogue.

## Ce qu'il vaut

Face à son prédécesseur v5, sur des matchs en 2000 points appariés en duplicate :

| Adversaire | Jeu de la carte | Résultat |
|---|---|---|
| Bid v5 ISDD | DouDou50 | **55,8 %** |
| Bid v5 ISDD | IS-DD | **57,3 %**, +181 pts/match |

## Ce qu'il ne sait pas faire

Trois défauts **mesurés**, qui motivent une v7 :

1. **Il n'est pas indifférent au nom des couleurs.** 24,6 % de ses annonces changent si
   l'on renomme les couleurs, alors que rien ne distingue Pique de Trèfle avant qu'un
   atout soit nommé.
2. **Ses meilleures valeurs sont trop proches** — visible dans l'exemple ci-dessus. Le
   choix entre 100 et 120 est peu informé.
3. **Il n'annonce jamais capot** : 0 fois sur 3000 enchères.

À noter aussi : une main seule n'explique que ~17 % de la variance de l'issue. L'essentiel
du signal vient de l'historique des annonces — ce que partenaire et adversaires ont dit.

## Détails d'implémentation

Le fichier est un tableau de flottants 32 bits **sans en-tête** : la bibliothèque déduit
l'architecture de sa taille. Si vous écrivez votre propre chargeur, la disposition des
poids et l'ordre exact des 117 entrées sont dans
[`bid_net.rs`](https://github.com/Avo-k/colver/blob/master/colver-core/src/bid/bid_net.rs)
et [`bid_obs.rs`](https://github.com/Avo-k/colver/blob/master/colver-core/src/bid/bid_obs.rs).

Un point qui n'est **pas** détectable depuis le fichier : ce modèle attend l'observation
physique, pas canonique (la canonicalisation par couleur arrive en v7). Se tromper est
silencieux — le réseau rend une annonce légale dans la mauvaise couleur.

## Règles

Barème « points faits + demandés », base 162 (252 sur capot réalisé), contre ×2 et
surcontre ×3 sur la valeur du contrat seule, aucun arrondi.

Entraîné avant le correctif de règle du 2026-08-02, qui élargit légèrement l'ensemble des
coups légaux. Effet mesuré : 0,076 % des décisions concernées, 0,014 % effectivement
changées. Le modèle est très légèrement hors-distribution, pas invalide.

## Licence

MIT.

```bibtex
@software{colver,
  author = {Avo-k},
  title  = {Colver: a Belote Contrée engine with RL agents},
  url    = {https://github.com/Avo-k/colver}
}
```
