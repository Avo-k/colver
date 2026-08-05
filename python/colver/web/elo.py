"""Classement des joueurs et des bots.

**L'unité classée est la partie en 2000 points** (2026-08-03), pas la donne. Une
donne isolée ou une partie en 1000 restent jouables, analysables et partageables :
elles ne comptent simplement pas au classement. C'est le format des tournois
réels, c'est l'unité de l'arène — donc l'ancre d'un bot est une mesure directe au
lieu d'une conversion — et c'est le seul levier honnête qui élargisse l'échelle.

**Abandonner vaut défaite.** Sans ça, quitter quand on perd serait gratuit. Ce qui
rend la règle juste est qu'une partie interrompue **se reprend**
(`server._resume_match`) : l'abandon est un acte délibéré, pas un accident de
connexion.

**Les bots sont l'étalon, pas des joueurs.** Leur note est figée. Avant le
2026-08-03 ils dérivaient avec la population — Dédé était monté de 1000 à 1044,
pic 1119, uniquement parce que les humains perdent contre lui — et comme tout le
monde est mesuré contre eux, l'arrivée de joueurs plus faibles dévaluait en
silence les inscrits.

## La note n'est plus une mise à jour incrémentale (2026-08-05)

Elle est le **posterior exact** recalculé depuis le bilan complet, et publiée sous
sa forme **conservatrice** `mu - 2*sigma` (convention TrueSkill/Xbox). Trois
défauts mesurés motivaient le changement, tous invisibles à l'usage :

1. **Le classement ordonnait par inexpérience.** Corrélation de Spearman entre
   « parties jouées » et « note affichée » : **−0,89** sur la base de prod du
   2026-08-05. Tout le monde partait de 1000, les humains gagnent ~24 % de leurs
   parties contre Dédé, donc tout le monde descendait — et le tableau classait
   surtout qui avait fait le moins de chemin. Le 1er avait joué **une** partie et
   l'avait perdue ; le meilleur bilan réel (4/12) était **6e sur 7**.
2. **Le nombre affiché n'était pas l'estimation du niveau.** La récurrence K part
   de 1000 et converge lentement : simulé sur 20 000 tirages, un joueur réellement
   à 550 affiche encore **832 après 12 parties** (biais +282) et il lui faut
   ~300 parties pour arriver. Coller un « ± » autour de ce nombre-là aurait
   attaché un intervalle de MLE à un centre qui n'en est pas un.
3. **Le seuil d'affichage n'achetait rien.** À 5 parties l'IC95 vaut ±609 Elo ;
   masquer en dessous ne rendait pas le reste fiable. Il n'existe plus : la note
   conservatrice place un joueur non confirmé **en bas** au lieu de le cacher,
   donc l'incertitude se lit dans la position.

Avec `mu - 2*sigma`, un nouveau venu **entre par le bas et monte en jouant** — à
niveau constant, jouer réduit sigma donc remonte la note. C'est exactement
l'inverse du défaut n° 1, et ça vient de la structure, pas d'un correctif.

## Pourquoi mu0 et tau sont FIGÉS, et pas ajustés sur la population

Les estimer par Bayes empirique était la première idée, et c'est une régression.
Mesuré sur la base de prod : ré-estimer `mu0` à chaque partie fait bouger la note
**des autres joueurs** de +28 à +32 quand quelqu'un gagne une partie, et de −101 à
−150 à l'arrivée d'un joueur qui perd ses 20 premières.

C'est **exactement le défaut que `K_BOT = 0` a fermé**, remonté d'un étage : au
lieu que ce soit l'ancre du bot qui dérive avec la population, c'est le prior. Une
fois `mu0` et `tau` figés, le couplage tombe à **exactement zéro** — la note
devient une fonction pure du bilan personnel, donc publiable et vérifiable à la
main, et personne ne bouge sans jouer.

Le coût est déplacé, pas supprimé : `mu0` vieillira si la population se déplace.
C'est le même problème que l'ancre des bots et il a le même remède —
`ANCHOR_VERSION`, un décalage explicite et daté plutôt qu'un ajustement silencieux.
Signal supplémentaire qu'il s'agit bien d'une convention et non d'une mesure :
`mu0` ajusté vaut 560 à `tau = 2` mais 508 à `tau = 200`, et `tau` lui-même n'est
**pas identifiable** — le maximum de vraisemblance le place à 0 (les 7 bilans de
prod sont compatibles avec 7 joueurs identiques) et tout `tau` jusqu'à ~300 est
dans le bruit.

`tau = 300` plutôt que 150 pour une raison précise : l'ordre du classement est
inchangé de 100 à 300, mais un prior serré coûte ~2,4× en parties à un joueur
d'exception qui doit prouver son niveau. Figer `tau` bas reviendrait à inscrire
dans la constante « personne ici n'est fort », ce qui est vrai aujourd'hui et sera
faux au premier bon joueur qui s'inscrit.

## Deux échelles, et une seule conversion

L'échelle **interne** est celle des mesures : les écarts entre bots viennent de
`arena h2h` et la constante Elo y vaut 400. L'échelle **d'affichage** en est une
transformée affine, appliquée en un seul endroit (`to_display`), choisie pour se
lire comme aux échecs : un nouveau joueur lit 800, un joueur confirmé de niveau
moyen lit 1600, Dédé 2200.

⚠️ **Ça ne crée aucune précision.** Multiplier les écarts multiplie le bruit à
l'identique : l'intervalle du mieux classé passe de ±253 à ±484 dans la même
opération. C'est un choix de lisibilité, à ne jamais présenter comme un gain de
fiabilité.

`rate_match` est idempotent (une ligne d'`elo_history` par partie × entité), ce qui
rend le backfill du démarrage sûr à chaque boot.
"""

import asyncio
import logging
import math

import numpy as np

import colver.web.database as db

logger = logging.getLogger(__name__)

# ===== Étalonnage figé =====================================================
#
# Version de l'étalonnage. À incrémenter — et à redocumenter — dès qu'un bot
# change de modèle, ou que `PRIOR_MEAN` / `PRIOR_SD` / l'affichage bougent. Le
# suffixe dit l'unité : une note « donne » et une note « partie » ne se comparent
# pas, et le passage de l'une à l'autre a multiplié tous les écarts par ~3,4.
ANCHOR_VERSION = "2026-08-post-match"

# Seules les parties à cette cible comptent. Une donne isolée (`target = 0`, le
# défaut du site) et une partie en 1000 restent jouables et analysables.
#
# Ouvrir plus tard aux parties en 1000 est la décision la plus réversible du lot
# — il suffirait d'un poids `sqrt(1000/2000) = 0,71`, à la manière de FIBS au
# backgammon. À garder en réserve si le classement se révèle trop vide.
RATED_TARGET = 2000

# Prior sur le niveau d'un humain, échelle interne. CONVENTIONS, pas ajustements
# — voir l'en-tête : les ré-estimer couple les joueurs entre eux.
PRIOR_MEAN = 550.0
PRIOR_SD = 300.0

# Note publiée = mu - CONSERVATISM * sigma. 2 plutôt que le 3 de TrueSkill : à
# 3 sigma un joueur à 12 parties afficherait une note négative sur l'échelle
# interne, et le tableau se lirait comme une punition de l'inexpérience.
CONSERVATISM = 2.0

# Transformation d'affichage, définie par deux points humains lisibles. Les bots
# atterrissent où les mesures les mettent (Dédé 2200, DouDou50 1973, Oracle 2480).
# Ancrer sur un BOT serait le piège : l'écart humain → DouDou50 vaut 280 en
# interne, soit plus que l'écart DouDou → Dédé (170), donc mettre DouDou à 1000
# enverrait les joueurs réels **sous zéro**.
DISPLAY_NEW = 800.0      # ce que lit un joueur sans aucune partie classée
DISPLAY_TYPICAL = 1600.0  # ce que lit un joueur confirmé de niveau moyen

_INTERNAL_NEW = PRIOR_MEAN - CONSERVATISM * PRIOR_SD
DISPLAY_SCALE = (DISPLAY_TYPICAL - DISPLAY_NEW) / (PRIOR_MEAN - _INTERNAL_NEW)
DISPLAY_OFFSET = DISPLAY_NEW - _INTERNAL_NEW * DISPLAY_SCALE

# Note figée de chaque bot, **échelle interne, unité de partie**.
#
# Dédé vaut 1000 par définition : c'est lui l'origine de l'échelle.
#
# DouDou est 170 points en dessous, mesuré par `arena h2h web_dede web_doudou`
# (36-14 sur 50 matchs, soit +164 Elo, IC95 [+58 ; +270] — précision modeste,
# mais sans commune mesure avec le « 988,6 sur 11 donnes » d'avant).
BOT_ELO = {
    "dede": 1000.0,
    "doudou": 830.0,
}

# Repli d'un bot non étalonné. Il le traite comme l'égal de Dédé — c'est le bon
# défaut (mieux vaut une hypothèse visible et fausse qu'un bot qui dérive) mais
# **tout bot ajouté doit passer par un h2h avant d'être assis en production**.
START_ELO = 1000.0

# Grille d'intégration du posterior. Large des deux côtés : un joueur qui perd
# ses 20 premières parties a un posterior qui déborde franchement sous le prior.
_GRID = np.arange(-2000.0, 4000.0, 1.0)
_PRIOR_LOGP = -0.5 * ((_GRID - PRIOR_MEAN) / PRIOR_SD) ** 2


def to_display(x):
    """Échelle interne (celle des mesures) → échelle affichée."""
    return DISPLAY_OFFSET + DISPLAY_SCALE * x


def from_display(x):
    return (x - DISPLAY_OFFSET) / DISPLAY_SCALE


def bot_elo(name):
    """Note figée d'un bot, échelle interne."""
    return BOT_ELO.get(name, START_ELO)


def posterior(record):
    """(moyenne, sigma) a posteriori sur l'échelle interne.

    `record` : itérable de `(score, partner_elo, opp_elo)`, une entrée par partie
    classée, toutes sur l'échelle interne. `score` vaut 1 (victoire) ou 0.

    La vraisemblance est celle qu'`elo.py` a toujours utilisée : l'équipe vaut la
    moyenne de ses deux joueurs, donc l'écart d'équipe ne porte que **la moitié**
    de l'écart individuel — c'est la dilution par le partenaire, et c'est la
    raison de fond pour laquelle il faut des centaines de parties en solo.
    """
    lp = _PRIOR_LOGP.copy()
    for score, partner, opp in record:
        p = 1.0 / (1.0 + 10 ** ((opp - (_GRID + partner) / 2) / 400.0))
        np.clip(p, 1e-15, 1 - 1e-15, out=p)
        lp += score * np.log(p) + (1.0 - score) * np.log1p(-p)
    w = np.exp(lp - lp.max())
    w /= w.sum()
    mean = float((_GRID * w).sum())
    return mean, float(math.sqrt(float(((_GRID - mean) ** 2 * w).sum())))


def note_of(mean, sd):
    """Note publiée : la borne basse, sur l'échelle d'affichage."""
    return to_display(mean - CONSERVATISM * sd)


_lock = asyncio.Lock()  # serialize read-modify-write across concurrent matches


def _seat_entities(game, player_rows):
    """Chaque siège → une entité notable `(kind, ref)`, ou None si la partie
    n'est pas notable (un siège anonyme, un bot inconnu).

    Les sièges d'une partie ne changent pas d'une donne à l'autre (seul le
    donneur tourne), donc n'importe quelle donne rend le même tableau.
    """
    agents = game["agents"]
    humans = {row["seat"]: row["user_id"] for row in player_rows}
    if game["mode"] == "play" and game["human_seat"] is not None:
        if game.get("user_id") is None:
            return None  # anonymous solo game
        humans[game["human_seat"]] = game["user_id"]

    seats = []
    for s in range(4):
        if s in humans:
            seats.append(("user", str(humans[s])))
        else:
            bot = agents.get(str(s))
            if not bot or bot == "human":
                return None
            seats.append(("bot", bot))
    return seats


async def _match_seats(conn, match_id):
    """Entités des quatre sièges d'une partie, lues sur sa première donne saine."""
    rows = await conn.execute_fetchall(
        "SELECT id FROM games WHERE match_id = ? AND is_complete = 1 AND invalid = 0 "
        "ORDER BY deal_no LIMIT 1",
        (match_id,),
    )
    if not rows:
        return None
    game = await db.get_game(rows[0][0])
    if game is None:
        return None
    players = await conn.execute_fetchall(
        "SELECT seat, user_id FROM game_players WHERE game_id = ?", (rows[0][0],))
    return _seat_entities(game, [dict(r) for r in players])


async def _losing_team(conn, match_id, owner_id):
    """Camp qui perd une partie abandonnée : celui de qui a abandonné.

    `db.abandon_match` n'accepte que le propriétaire d'une partie solo, donc
    `matches.user_id` désigne bien l'abandonnant. Si on ne sait pas le placer, on
    ne note pas — mieux vaut une partie non notée qu'une défaite attribuée au
    hasard.
    """
    if owner_id is None:
        return None
    rows = await conn.execute_fetchall(
        "SELECT human_seat FROM games WHERE match_id = ? AND user_id = ? "
        "AND human_seat IS NOT NULL ORDER BY deal_no LIMIT 1",
        (match_id, owner_id),
    )
    if not rows or rows[0][0] is None:
        return None
    return rows[0][0] % 2


async def _level_internal(conn, ent):
    """Niveau courant d'une entité, échelle interne — pour servir de partenaire
    ou d'adversaire dans la vraisemblance d'un autre joueur.

    Un bot rend son ancre. Un humain rend sa moyenne a posteriori courante, ou le
    prior s'il n'a rien joué.

    ⚠️ **Approximation, et elle ne mord pas aujourd'hui.** Le posterior exact
    d'une partie humain-contre-humain serait *joint* sur les quatre sièges. Les
    33 parties classées de la prod sont toutes en solo (un humain, trois bots),
    donc ce chemin ne s'emprunte pas encore. Le jour où le salon sera classé, il
    faudra soit itérer cette passe jusqu'au point fixe, soit passer à un vrai
    modèle joint.
    """
    if ent[0] == "bot":
        return bot_elo(ent[1])
    rows = await conn.execute_fetchall(
        "SELECT level FROM elo_ratings WHERE kind = ? AND ref = ?", ent)
    if not rows or rows[0][0] is None:
        return PRIOR_MEAN
    return from_display(rows[0][0])


async def _record(conn, ent):
    """Le bilan complet d'une entité, dans l'ordre où il s'est constitué."""
    rows = await conn.execute_fetchall(
        "SELECT e.score, e.partner_elo, e.opp_elo FROM elo_history e "
        "JOIN matches m ON m.id = e.match_id "
        "WHERE e.kind = ? AND e.ref = ? ORDER BY m.created_at",
        ent,
    )
    return [(r[0], r[1], r[2]) for r in rows]


async def _store(conn, ent, record):
    """Recalcule le posterior d'une entité depuis son bilan et l'écrit."""
    if ent[0] == "bot":
        # Un étalon ne se calcule pas : il EST l'échelle. On garde quand même sa
        # ligne pour que le tableau se lise d'un seul SELECT, et son compteur de
        # parties pour l'afficher.
        note = level = to_display(bot_elo(ent[1]))
        sigma = 0.0
    else:
        mean, sd = posterior(record)
        note, level, sigma = note_of(mean, sd), to_display(mean), DISPLAY_SCALE * sd
    await conn.execute(
        "INSERT INTO elo_ratings (kind, ref, elo, level, sigma, games, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(kind, ref) DO UPDATE SET "
        "  elo = excluded.elo, level = excluded.level, sigma = excluded.sigma, "
        "  games = excluded.games, updated_at = excluded.updated_at",
        (*ent, round(note, 2), round(level, 2), round(sigma, 2),
         len(record), db._now()),
    )
    return note


async def rate_match(match_id):
    """Note une partie terminée en 2000 points. Idempotent ; ne lève jamais."""
    try:
        async with _lock:
            return await _rate_match_locked(match_id)
    except Exception:
        logger.exception("rating of match %s failed", match_id)
        return False


async def _rate_match_locked(match_id):
    conn = await db.get_db()
    rows = await conn.execute_fetchall(
        "SELECT 1 FROM elo_history WHERE match_id = ? LIMIT 1", (match_id,))
    if rows:
        return False  # already rated

    rows = await conn.execute_fetchall(
        "SELECT target, is_complete, winner, abandoned, user_id FROM matches WHERE id = ?",
        (match_id,),
    )
    if not rows:
        return False
    target, is_complete, winner, abandoned, owner_id = rows[0]
    if target != RATED_TARGET or not is_complete:
        return False

    # Une partie dont une donne est en quarantaine a un score cumulé faux : elle
    # ne doit pas entrer au classement (même règle que `integrity.scan`).
    bad = await conn.execute_fetchall(
        "SELECT 1 FROM games WHERE match_id = ? AND invalid = 1 LIMIT 1", (match_id,))
    if bad:
        return False

    if abandoned:
        loser = await _losing_team(conn, match_id, owner_id)
        if loser is None:
            return False
        score_ns = 0.0 if loser == 0 else 1.0
    else:
        if winner is None:
            return False
        score_ns = 1.0 if winner == 0 else 0.0

    seats = await _match_seats(conn, match_id)
    if seats is None:
        return False

    levels = {}
    for ent in set(seats):
        levels[ent] = await _level_internal(conn, ent)

    # Une entité peut tenir plusieurs sièges (un bot en tient trois en solo). On
    # ne l'inscrit qu'une fois, sur son premier siège : `games` compte des
    # **parties**, plus des sièges — Dédé affichait 2 540 pour 881 donnes jouées.
    seen = set()
    for seat, ent in enumerate(seats):
        if ent in seen:
            continue
        seen.add(ent)
        team = seat % 2
        score = score_ns if team == 0 else 1.0 - score_ns
        partner = levels[seats[seat ^ 2]]
        opp = (levels[seats[(seat + 1) % 4]] + levels[seats[(seat + 3) % 4]]) / 2

        before = await _record(conn, ent)
        note_before = (note_of(*posterior(before)) if ent[0] == "user"
                       else to_display(bot_elo(ent[1])))
        await conn.execute(
            "INSERT INTO elo_history "
            "(match_id, kind, ref, score, partner_elo, opp_elo, delta, elo_after) "
            "VALUES (?, ?, ?, ?, ?, ?, 0, 0)",
            (match_id, *ent, score, partner, opp),
        )
        note_after = await _store(conn, ent, before + [(score, partner, opp)])
        # `delta` / `elo_after` gardent leurs noms et changent de sens : ce n'est
        # plus le pas d'une récurrence K mais **le déplacement de la note publiée
        # causé par cette partie**, ce qui est ce que « combien m'a rapporté ce
        # match » veut dire. Les deux consommateurs (`list_matches`, `get_match`)
        # n'ont rien à changer.
        await conn.execute(
            "UPDATE elo_history SET delta = ?, elo_after = ? "
            "WHERE match_id = ? AND kind = ? AND ref = ?",
            (round(note_after - note_before, 2), round(note_after, 2), match_id, *ent),
        )
    await conn.commit()
    return True


async def backfill():
    """Note toutes les parties terminées au format classé, la plus vieille d'abord."""
    conn = await db.get_db()
    rows = await conn.execute_fetchall(
        "SELECT id FROM matches WHERE is_complete = 1 AND target = ? "
        "AND id NOT IN (SELECT DISTINCT match_id FROM elo_history) "
        "ORDER BY created_at",
        (RATED_TARGET,),
    )
    rated = 0
    for (match_id,) in rows:
        if await rate_match(match_id):
            rated += 1
    if rated:
        logger.info("backfill: rated %d match(es)", rated)


async def get_rating(kind, ref):
    """Note, niveau estimé et incertitude d'une entité, échelle d'affichage."""
    conn = await db.get_db()
    rows = await conn.execute_fetchall(
        "SELECT elo, level, sigma, games FROM elo_ratings WHERE kind = ? AND ref = ?",
        (kind, str(ref)))
    if kind == "bot":
        anchor = to_display(bot_elo(str(ref)))
        return {"elo": anchor, "level": anchor, "uncertainty": 0.0,
                "games": rows[0][3] if rows else 0}
    if not rows:
        mean, sd = posterior([])
        return {"elo": round(note_of(mean, sd), 1), "level": round(to_display(mean), 1),
                "uncertainty": round(DISPLAY_SCALE * sd * 2, 1), "games": 0}
    elo_, level, sigma, games = rows[0]
    return {"elo": round(elo_, 1), "level": round(level, 1),
            "uncertainty": round((sigma or 0.0) * 2, 1), "games": games}


async def leaderboard():
    """Entités classées, meilleure d'abord.

    Deux colonnes, parce qu'elles disent deux choses : la **note** est ce qu'on
    peut prouver (elle sert au tri), le **niveau** est la meilleure estimation
    avec son incertitude. L'écart entre les deux est le prix de l'inexpérience,
    et il se referme en jouant.

    **Plus de seuil d'affichage.** Il masquait un joueur sous 5 parties, ce qui
    demandait ensuite d'expliquer une disparition ; la note conservatrice le place
    en bas, ce qui dit la même chose sans rien cacher.
    """
    conn = await db.get_db()
    rows = await conn.execute_fetchall(
        "SELECT r.kind, r.ref, r.elo, r.level, r.sigma, r.games, u.username "
        "FROM elo_ratings r "
        "LEFT JOIN users u ON r.kind = 'user' AND u.id = CAST(r.ref AS INTEGER) "
        "ORDER BY r.elo DESC",
    )
    return [
        {
            "kind": kind, "ref": ref,
            "elo": round(elo_, 1),
            "level": round(level, 1) if level is not None else round(elo_, 1),
            "uncertainty": round((sigma or 0.0) * 2, 1),
            "games": games,
            "name": username if kind == "user" else ref,
        }
        for kind, ref, elo_, level, sigma, games, username in rows
    ]


async def standing(kind, ref):
    """État du classement d'une entité.

    `ranked` / `needed` / `remaining` survivent au retrait du seuil : ils sont lus
    par `/api/me` et par la page Compte. Tout le monde est désormais classé — un
    joueur sans partie apparaît simplement au plancher.
    """
    r = await get_rating(kind, ref)
    return {**r, "ranked": True, "needed": 0, "remaining": 0}
