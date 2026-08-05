"""Une variante : des coups posés sur une position, puis le jeu parfait jusqu'au bout.

C'est le §8 de [rejouer_analyse_erreurs.md](../../../docs/idees/rejouer_analyse_erreurs.md) :
la « ligne alternative » de Rejouer et l'exploration libre de `/analyse/jeu` sont
**la même chose à un paramètre près** — la pile de coups vaut un seul coup pour
la première, et ce que le joueur a poussé pour la seconde. Écrire deux moteurs
aurait garanti qu'ils divergent.

Deux choses valent d'être dites avant de s'en servir.

**Le déroulé est un plafond, pas une prédiction.** Après le coup, les quatre
sièges jouent en double-dummy : ils voient les 32 cartes. La suite réelle
contient d'autres erreurs des deux camps, donc comparer une variante à ce qui
s'est passé fait dire n'importe quoi à l'écart. Le seul comparatif honnête est
**variante après le coup joué** contre **variante après le coup de l'Oracle** :
leur écart *est* le coût affiché, par construction.

**On n'emprunte jamais l'`Env` de l'appelant.** `set_state_bytes` restaure le
`GameState` et lui seul : `play_order`, `played_by` et `bid_history` sont des
champs du wrapper PyO3, **hors** des 84 octets. Un aller-retour
instantané-restauration laisserait donc les cartes de la variante dans
`play_order`, d'où sortent l'observation DMC et `to_cfn` — une corruption
invisible pour un consommateur DD, fatale pour tous les autres. Le moteur
reconstruit donc sa propre partie par rejeu : ~0,3 ms, contre des dizaines de
millisecondes de solve.
"""

import colver

# 32 cartes dans une donne : une variante ne peut pas être plus longue. La garde
# n'est pas défensive contre le moteur, elle empêche une boucle infinie si
# `action_oracle_dd` rendait un jour un coup illégal.
MAX_CARDS = 32


def _replay(dealer, hands, prefix):
    """L'`Env` à la fin de `prefix`, construit à neuf. Voir l'en-tête du module."""
    env = colver.Env.deal_with_hands(int(dealer), [list(h) for h in hands])
    for a in prefix:
        if env.is_terminal():
            break
        env.step(int(a))
    return env


def complete(env):
    """Déroule la donne en jeu parfait depuis ici. **Modifie `env`.**

    Rend les cartes jouées, dans l'ordre. Chaque coup passe par
    `action_oracle_dd`, donc par le même départage des ex æquo que l'Oracle du
    reste du site (moins de points de carte, puis indice le plus bas) : la ligne
    est une fonction de la position, elle ne dépend pas de l'ordre de la boucle
    racine.
    """
    cards = []
    while not env.is_terminal() and len(cards) < MAX_CARDS:
        card = int(env.action_oracle_dd())
        env.step(card)
        cards.append(card)
    return cards


def outcome(env):
    """Ce que la donne a rapporté, une fois terminée.

    `taker_pts` est en points **cartes** (0-252, dix de der compris) du côté du
    preneur — la même unité que la courbe de Rejouer. `score` est l'écart de
    score **marqué** N-S − E-O, celui de `cost_score`. Les deux ne se
    soustraient pas.
    """
    contract = env.get_contract()
    taker = int(contract["team"]) if contract else 0
    points = env.get_points()
    rewards = env.rewards()
    return {
        "taker": taker,
        "taker_pts": int(points[taker]),
        "made": bool(rewards[taker] > 0),
        "score": int(rewards[0] - rewards[1]),
    }


def line(dealer, hands, prefix, moves=(), *, limit=MAX_CARDS):
    """La variante : `moves`, puis le jeu parfait jusqu'à la fin de la donne.

    Rend `None` si un coup de `moves` est illégal — c'est le cas normal quand un
    client propose une carte qui n'est pas dans la main du siège au trait, pas
    un incident.

    `limit` borne le **déroulé**, pas les coups poussés : à 0 on obtient la
    position sans la suite, ce qui sert à valider une pile avant de l'afficher.
    """
    env = _replay(dealer, hands, prefix)
    if env.is_terminal() or int(env.phase()) != 1:
        return None

    # Combien de cartes sont déjà sur le tapis quand la variante commence. Le
    # client en a besoin pour découper la ligne en plis : sans ça, une suite de
    # 31 cartes se lit comme une liste et plus comme une donne.
    trick_pos = sum(1 for c in env.get_current_trick() if int(c) >= 0)

    played = []
    for card in moves:
        card = int(card)
        if env.is_terminal() or card not in [int(a) for a in env.legal_actions()]:
            return None
        env.step(card)
        played.append(card)

    cards = []
    while not env.is_terminal() and len(cards) < limit:
        card = int(env.action_oracle_dd())
        env.step(card)
        cards.append(card)

    out = {"moves": played, "cards": cards, "trick_pos": trick_pos,
           "complete": bool(env.is_terminal())}
    if env.is_terminal():
        out.update(outcome(env))
    return out


def error_lines(dealer, hands, prefix, played_card, best_card):
    """Les deux lignes d'une erreur : ce qui suit le coup joué, et le coup de l'Oracle.

    Une seule ligne ne dit rien. La suite *réelle* n'est pas comparable — elle
    contient d'autres erreurs — et une variante parfaite comparée à elle
    raconterait n'importe quoi. Ces deux-ci partagent tout sauf le premier coup,
    donc leur écart de score **est** `cost_score`, ce qu'un test vérifie.

    Rend `None` dès qu'une des deux ne se déroule pas : une moitié de comparatif
    se lirait comme un comparatif.
    """
    a = line(dealer, hands, prefix, [int(played_card)])
    if a is None or not a["complete"]:
        return None
    b = line(dealer, hands, prefix, [int(best_card)])
    if b is None or not b["complete"]:
        return None
    return {"played": a, "best": b}
