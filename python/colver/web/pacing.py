"""Bot pacing modes — how fast the table plays, and which bot sits at it.

Two bundles, `standard` and `rapide`. Each pairs a bot with a display tempo,
and the pairing is not cosmetic: an IS-DD search costs real wall-clock per
move, so a fast tempo is only honest behind a bot that answers instantly.
Dédé gets the slow tempo (its thinking hides inside the pause), DouDou50 gets
the fast one (~1 ms of inference, so the pause is pure display).

All four seats always run the same bot — a table where the partner is weaker
than the opponents tells you nothing about how well you played.

Card and trick pauses decay linearly across the 8 tricks: late tricks carry
fewer real decisions, so they need less time to read. `standard` keeps a floor
so that even at trick 8 a human can still see who cut and who took it.
"""

import asyncio

MODES = {
    "standard": {
        "bot": "dede",
        "think_ms": 1200,
        "card": (1.4, 0.9),    # (trick 1, trick 8)
        "trick": (1.6, 1.2),
        "bid": 0.9,
    },
    "rapide": {
        "bot": "doudou",
        # Only consulted on the degraded path below: DouDou50 ignores it.
        "think_ms": 400,
        "card": (0.6, 0.25),
        "trick": (0.5, 0.3),
        "bid": 0.35,
    },
}

DEFAULT_MODE = "standard"

_LAST_TRICK = 7  # 0-based index of the 8th trick

# La dernière levée d'une donne est la seule qu'on ne verrait jamais : le
# panneau de fin recouvre la table, et il arrive dans la même image que le pli.
# On la tient donc plus longtemps qu'un pli ordinaire, et sans regarder le mode
# — ce n'est pas un tempo, c'est le dernier regard sur le pli qui a décidé la
# donne, une fois par donne.
DEAL_END_HOLD = 2.0

# Le dernier pli ne contient aucune décision : chacun n'a plus qu'une carte.
# Il se déroule donc tout seul, à son propre tempo et sans regarder le mode —
# 1 s avant la première carte, 0,3 s avant chacune des suivantes. Pour un siège
# humain ces délais sont une échéance et non une attente : le joueur garde la
# main sur sa carte, et s'il ne la pose pas le serveur la joue pour lui, comme
# il joue déjà un passe forcé. La dernière image, elle, reste tenue
# `DEAL_END_HOLD` : c'est le seul moment où on regarde le pli, pas où on joue.
LAST_TRICK_LEAD = 1.0
LAST_TRICK_CARD = 0.3


def normalize(mode):
    """Coerce anything a client sent into a known mode name."""
    return mode if mode in MODES else DEFAULT_MODE


def resolve(mode, doudou_available=True):
    """Mode name -> (bot type, IS-DD budget in ms, degraded flag).

    `degraded` is True when the mode's bot is unavailable and we fell back to
    Dédé on a short budget. The caller is expected to say so rather than
    silently seat a different bot than the one advertised.
    """
    mode = normalize(mode)
    spec = MODES[mode]
    bot = spec["bot"]
    if bot == "doudou" and not doudou_available:
        return "dede", spec["think_ms"], True
    return bot, spec["think_ms"], False


def mode_for_bot(bot):
    """Bot déjà assis -> mode qui va avec. L'inverse de `resolve`.

    Une donne interrompue enregistre le bot qui tenait les sièges
    (`games.agents`), jamais le mode qui l'avait choisi — hors partie il n'y a
    pas de `matches.pacing` où le lire. Or la reprise a besoin du tempo autant
    que du bot : sans lui une donne « rapide » repartirait derrière celui de
    Dédé. Le repli dégradé se relit donc en `standard`, qui est bien le tempo
    qu'il avait.
    """
    for name, spec in MODES.items():
        if spec["bot"] == bot:
            return name
    return DEFAULT_MODE


def _taper(bounds, trick_idx):
    start, floor = bounds
    t = max(0, min(_LAST_TRICK, int(trick_idx)))
    return floor + (start - floor) * (1 - t / _LAST_TRICK)


def bid_delay(mode):
    """Pause after an auction action."""
    return MODES[normalize(mode)]["bid"]


def last_trick_delay(cards_in_trick):
    """Délai avant la carte n° `cards_in_trick` (0-based) du dernier pli."""
    return LAST_TRICK_LEAD if int(cards_in_trick) <= 0 else LAST_TRICK_CARD


def card_delay(mode, trick_idx, cards_in_trick=None):
    """Pause after a card that does not complete the trick.

    `cards_in_trick` : cartes déjà posées sur le pli en cours. Il n'est lu que
    sur la dernière levée, qui a son propre tempo (cf. `last_trick_delay`) ;
    `None` = l'appelant ne sait pas, on retombe sur le tempo du mode.
    """
    if cards_in_trick is not None and int(trick_idx) == _LAST_TRICK:
        return last_trick_delay(cards_in_trick)
    return _taper(MODES[normalize(mode)]["card"], trick_idx)


def trick_delay(mode, trick_idx, deal_over=False):
    """Hold of a completed trick, before the four cards are swept away.

    `deal_over` : ce pli termine la donne, cf. `DEAL_END_HOLD`.
    """
    if deal_over:
        return DEAL_END_HOLD
    return _taper(MODES[normalize(mode)]["trick"], trick_idx)


def move_delay(mode, phase, tricks_completed, cards_in_trick=None):
    """Pause for a non-trick-completing action, from the surrounding state.

    `phase` / `tricks_completed` / `cards_in_trick` come straight off the env
    (0 = bidding).
    """
    if int(phase) == 0:
        return bid_delay(mode)
    return card_delay(mode, tricks_completed, cards_in_trick)


async def hold(target, elapsed=0.0):
    """Sleep whatever is left of `target` after `elapsed` seconds already spent.

    Bot thinking counts *toward* the pause instead of adding to it, so the
    tempo a player sees is the same whichever bot is seated — otherwise Dédé's
    1.2 s of search would stack on top of every pause and the standard mode
    would run at nearly twice its advertised pace.
    """
    rest = target - elapsed
    if rest > 0:
        await asyncio.sleep(rest)
