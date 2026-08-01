"""Vérifier qu'une donne enregistrée décrit une partie qui a pu être jouée.

Le pendant en lecture du garde-fou en écriture (`game_manager.check_legal`) :
même prédicat, deux moments. Le garde-fou empêche d'écrire une donne fausse ;
ce module trouve celles qui sont déjà en base, écrites avant lui.

**Ce qu'on teste, et pourquoi ça suffit.** Une donne est rejouable si, partant
de `hands` et `dealer`, chacune de ses `actions` est légale à son tour et que la
dernière rend la donne terminale. Ces deux conditions se suffisent : un journal
entièrement légal ne peut ni jouer deux fois la même carte, ni en oublier une,
donc les 152 points cartes et le dix de der tombent juste par construction.
C'est d'ailleurs par eux que la corruption a été trouvée (l'assertion de
`CountingSession._payload`, cf. `docs/web_todo.md` §4.5) — mais c'était le
symptôme ; la légalité est la cause, et elle nomme le coup fautif.

**Marquer, pas effacer** : une donne fausse est un incident, et l'effacer
effacerait la trace de l'incident avec elle. `games.invalid = 1` la retire de
tout ce qui la sert (Rejouer, analyses, listings, comptage, Elo) sans la perdre.

**Chaque ligne n'est examinée qu'une fois** (`games.checked_at`), donc le scan
au démarrage est borné par les donnes terminées depuis le dernier lancement, pas
par la taille de la base.
"""

import asyncio
import logging

import colver
import colver.web.database as db

logger = logging.getLogger(__name__)

# Le scan tourne en tâche de fond au démarrage, en même temps que le backfill
# Elo et les premières parties. Rejouer une donne coûte des microsecondes, mais
# une base rattrapée d'un bloc monopoliserait la boucle d'événements : on rend
# la main tous les `_YIELD_EVERY` contrôles.
_YIELD_EVERY = 50


def check_deal(game):
    """Rend None si la donne se rejoue, sinon la raison en clair.

    Pure et synchrone : prend la ligne telle que `db.get_game` la rend (mains,
    actions et agents déjà décodés) et ne touche à rien.
    """
    try:
        hands = [list(h) for h in game["hands"]]
        if len(hands) != 4 or sum(len(h) for h in hands) != 32:
            return f"distribution invalide ({sum(len(h) for h in hands)} cartes)"
        env = colver.Env.deal_with_hands(int(game["dealer"]) % 4, hands)
    except Exception as e:  # noqa: BLE001 — une ligne illisible est une ligne fausse
        return f"donne illisible : {type(e).__name__}: {e}"

    for i, entry in enumerate(game["actions"], start=1):
        try:
            action = int(entry["action"] if isinstance(entry, dict) else entry)
        except (TypeError, ValueError, KeyError):
            return f"coup {i} : action illisible ({entry!r})"
        if env.is_terminal():
            return f"coup {i} : la donne était déjà terminée"
        if action not in list(env.legal_actions()):
            name = colver.Env.action_name(action, int(env.phase()))
            return (f"coup {i} : {name} illégal pour le siège "
                    f"{int(env.current_player())}")
        env.step(action)

    if not env.is_terminal():
        return f"journal incomplet ({len(game['actions'])} coups, donne non finie)"
    return None


async def scan(limit=None):
    """Examiner les donnes terminées jamais examinées. Rend (vues, fausses).

    Idempotent et reprenable : `checked_at` marque ce qui est fait, donc une
    coupure en plein scan ne coûte que le tour en cours, et un redémarrage ne
    recompte pas ce qui l'a déjà été.
    """
    conn = await db.get_db()
    sql = ("SELECT id FROM games WHERE is_complete = 1 AND checked_at IS NULL "
           "ORDER BY created_at")
    if limit:
        sql += f" LIMIT {int(limit)}"
    rows = await conn.execute_fetchall(sql)
    seen = 0
    bad = []
    for (game_id,) in rows:
        game = await db.get_game(game_id, include_incomplete=True, include_invalid=True)
        if game is None:
            continue
        reason = check_deal(game)
        await db.mark_game_checked(game_id, reason)
        seen += 1
        if reason is not None:
            bad.append((game_id, reason))
            logger.warning("donne %s incohérente — %s", game_id, reason)
        if seen % _YIELD_EVERY == 0:
            await asyncio.sleep(0)
    if seen:
        logger.info("intégrité : %d donne(s) vérifiée(s), %d écartée(s)",
                    seen, len(bad))
    return seen, bad
