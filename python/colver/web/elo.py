"""Elo ratings for users and bot types.

**L'unité notée est la partie en 2000 points** (2026-08-03), plus la donne. Une
donne isolée ou une partie en 1000 restent jouables, analysables et partageables :
elles ne comptent simplement pas au classement.

Trois raisons, dans l'ordre :

1. **C'est le format des tournois réels.** On note ce que les gens jouent en
   compétition, comme les échecs notent des parties et non des coups.
2. **C'est l'unité de l'arène.** `arena h2h` joue déjà des parties en 2000, donc
   le chiffre qu'elle rend est *directement* celui à écrire dans `BOT_ELO`. Avant,
   il fallait convertir entre deux échelles — et c'est exactement de là que venait
   le bricolage de l'ancre DouDou (fourchette modélisée [+28, +52], arrondie à 50).
3. **C'est le seul levier honnête qui élargisse l'échelle.** Mesuré **×3,4** sur
   deux couples indépendants (DouDou35→DouDou50 : 62,5 % des parties contre 46,0 %
   des donnes ; Heuristique→DouDou50 : 69,3 % contre 42,7 % — 1 200 parties
   chacun). L'étendue du jeu de la carte passe de 171 à ~580 Elo. Multiplier les
   écarts par une constante, à l'inverse, n'aurait rien créé : signal et bruit
   auraient été multipliés à l'identique.

Une partie est notée quand ses quatre sièges sont identifiables (bots, ou humains
avec un compte) : rating d'équipe = moyenne des deux partenaires, espérance Elo
classique, score 1/0 selon le vainqueur.

**Abandonner vaut défaite.** Sans ça, quitter quand on perd serait gratuit — et
sur ~10 donnes, ça arriverait. L'argument qui rend la règle juste est qu'une
partie interrompue **se reprend** (`server._resume_match`) : l'abandon est donc un
acte délibéré, pas un accident de connexion. Limite connue : en salon, un joueur
qui quitte tue la partie sans la clore (`is_complete = 0`), donc elle n'est jamais
notée — la porte reste ouverte de ce côté.

**Les bots sont l'étalon, pas des joueurs** (2026-08-03). Leur Elo est figé et
`K_BOT = 0` : ils ne bougent jamais. Avant ça ils dérivaient avec la population —
Dédé était monté de 1000 à 1044, pic 1119, uniquement parce que les humains
perdent contre lui — et comme tout le monde est mesuré contre eux, l'arrivée de
joueurs plus faibles dévaluait en silence les inscrits.

Trois choses à savoir sur ce choix :

- **Ça ne casse pas la somme nulle, elle était déjà cassée.** Avec `K_USER = 32`
  et `K_BOT = 8`, la somme des deltas d'une donne solo valait `24(s−e)`, soit
  ±24 points créés ou détruits à chaque donne. Et la conservation n'a de sens
  que dans un pool où tout le monde joue contre tout le monde, pas quand une
  entité tient trois sièges sur quatre.
- **C'est la pratique standard** des listes de moteurs (CCRL, SSDF) : ancrer sur
  une référence fixe pour que l'échelle ne dérive pas quand la population change.
- **Le coût est déplacé, pas supprimé** : la dérive passe de « qui joue » à
  « quelle version du bot ». D'où `ANCHOR_VERSION` — quand un bot change, il faut
  mesurer le nouveau contre l'ancien à l'arène et décaler **explicitement**, pas
  laisser l'Elo s'ajuster tout seul.

`rate_match` is idempotent (elo_history has one row per match × entity), which
makes the startup backfill safe to run on every boot. Les lignes des bots y sont
écrites malgré un delta toujours nul : c'est ce qui garantit l'idempotence même
sur une partie sans humain.
"""

import asyncio
import logging

import colver.web.database as db

logger = logging.getLogger(__name__)

START_ELO = 1000.0
K_USER = 32.0

# Seules les parties à cette cible comptent. Une donne isolée (`target = 0`, le
# défaut du site) et une partie en 1000 restent jouables et analysables : elles
# ne sont simplement pas notées.
#
# Ouvrir plus tard aux parties en 1000 est la décision la plus réversible du lot
# — il suffirait d'un poids `√(1000/2000) = 0,71`, à la manière de FIBS au
# backgammon. À garder en réserve si le classement se révèle trop vide.
RATED_TARGET = 2000

# Parties nécessaires pour apparaître au classement. En dessous, l'entité est
# notée (son Elo se construit) mais reste **masquée** du tableau.
#
# ⚠️ **Ce seuil n'est pas un seuil de précision, et il ne faut pas le lire comme
# tel.** En solo l'humain n'est qu'un des deux de son équipe, donc l'écart
# individuel est dilué de moitié et l'erreur-type vaut ~695/√n :
#
#     5 parties → ±609 Elo (IC95)      20 → ±305      50 → ±193      100 → ±136
#
# À 5 comme à 10 parties, l'intervalle reste plus large que l'étendue entière du
# jeu de la carte (~580 Elo). Monter la barre à 10 coûterait 2 à 3 heures de jeu
# avant le moindre retour sans franchir aucun seuil utile : on n'achète rien avec
# la sévérité. Ce qui achètera vraiment de la précision est ailleurs — noter la
# **marge** de la partie plutôt que le seul vainqueur (R3 transposé, dont
# l'échelle reste à mesurer), puis le par par décision (R4).
MIN_RATED_MATCHES = 5

# Les bots ne bougent pas. Zéro, et non « petit » : un K non nul les fait
# redériver entre deux recalages, ce qui est exactement le défaut qu'on ferme.
K_BOT = 0.0

# Version de l'étalonnage. À incrémenter — et à redocumenter — dès qu'un bot
# change de modèle ou de configuration de fond, sinon l'échelle bouge en silence.
# Le suffixe dit l'unité : un Elo « donne » et un Elo « match » ne se comparent
# pas, et le passage de l'un à l'autre a multiplié tous les écarts par ~3,4.
ANCHOR_VERSION = "2026-08-match"

# Elo figé de chaque bot, **en unité de partie**.
#
# Dédé vaut 1000 par définition : c'est lui l'origine de l'échelle.
#
# DouDou est 170 points en dessous. Ce nombre est la conversion en unité de
# partie (×3,4, mesuré) de l'écart de 50 points qui valait à la donne — lequel
# était déjà un arrondi dans une fourchette modélisée [+28, +52], et non une
# mesure. **Il hérite donc de toute l'incertitude de son prédécesseur, amplifiée
# par la conversion.**
#
# La bonne nouvelle est que ce bricolage a maintenant une date de péremption :
# depuis que le site note en parties de 2000, `arena h2h web_dede web_doudou`
# rend **directement** le chiffre à écrire ici, sans conversion ni modèle. Il
# faut un GPU tranquille (Dédé passe par le sidecar playgen, et la contention le
# pénaliserait seul, biaisant l'ancre vers le bas). C'est la mesure la plus
# urgente du dossier.
#
# Repères mesurés le 2026-08-03 pour lire l'échelle (`arena h2h`, 1 200 parties
# de 2000 points chacun, enchère v6 partout, seul le jeu de la carte varie) :
#
#     Heuristique -141   ·   DouDou35 -89   ·   DouDou50 0   ·   Oracle DD ~+380
#
# soit ~580 Elo entre le joueur à règles et l'omniscience double-mort. L'enchère
# ajoute ~270 de plus (`seat_influence.py --a b_petit_bide --b b_v6` et voisins).
# Détail : `docs/classement_et_scoring.md` §8.
BOT_ELO = {
    "dede": 1000.0,
    "doudou": 830.0,  # 50 à la donne × 3,4 — conversion, pas mesure
}


def k_for(matches_played):
    """K décroissant : la note converge vite, puis se stabilise.

    Trois paliers plutôt qu'une formule : c'est une approximation grossière de
    ce que Glicko-2 fait proprement avec son RD, et le jour où Glicko-2 arrivera
    il remplacera ces paliers sans rien casser d'autre.

    ⚠️ **Cet outil n'était pas légitime avant aujourd'hui.** Tant que la métrique
    était fausse — score binaire à la donne, bots non ancrés — un K décroissant
    ne corrigeait rien : il figeait plus proprement un bruit déjà accumulé.
    Maintenant que l'unité est la partie, que les bots sont l'étalon et que le
    classement ne bouge plus qu'avec de l'information, la décroissance est
    exactement ce qu'il faut. L'ordre comptait.
    """
    if matches_played < 10:
        return 64.0
    if matches_played < 30:
        return 32.0
    return 24.0


_lock = asyncio.Lock()  # serialize read-modify-write across concurrent matches


def _seat_entities(game, player_rows):
    """Map each seat to a rated entity (kind, ref), or None if unratable.

    Les sièges d'une partie ne changent pas d'une donne à l'autre (seul le
    donneur tourne), donc n'importe quelle donne de la partie rend le même
    tableau — on prend la première.
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


async def rate_match(match_id):
    """Rate one finished 2000-point match. Idempotent; never raises."""
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

    # Ratings courants. Celui d'un bot ne se lit pas en base : c'est une
    # constante d'étalonnage, et la base n'en garde une copie que pour que le
    # classement affiché puisse la lire d'un seul SELECT.
    ratings = {}
    for ent in set(seats):
        r = await conn.execute_fetchall(
            "SELECT elo, games FROM elo_ratings WHERE kind = ? AND ref = ?", ent)
        played = r[0][1] if r else 0
        if ent[0] == "bot":
            ratings[ent] = (bot_elo(ent[1]), played)
        else:
            ratings[ent] = (r[0][0] if r else START_ELO, played)

    team_elo = [
        (ratings[seats[0]][0] + ratings[seats[2]][0]) / 2,
        (ratings[seats[1]][0] + ratings[seats[3]][0]) / 2,
    ]
    expected_ns = 1.0 / (1.0 + 10 ** ((team_elo[1] - team_elo[0]) / 400))

    # Deltas agrégés par entité (un type de bot peut tenir plusieurs sièges).
    deltas = {}
    for seat, ent in enumerate(seats):
        team = seat % 2
        s = score_ns if team == 0 else 1.0 - score_ns
        e = expected_ns if team == 0 else 1.0 - expected_ns
        k = k_for(ratings[ent][1]) if ent[0] == "user" else K_BOT
        deltas[ent] = deltas.get(ent, 0.0) + k * (s - e)

    now = db._now()
    for ent, delta in deltas.items():
        new_elo = ratings[ent][0] + delta
        # `games` compte des **parties**, une par entité — plus des sièges. Dédé
        # affichait 2 540 pour 881 donnes jouées, et la page mentait.
        await conn.execute(
            "INSERT INTO elo_ratings (kind, ref, elo, games, updated_at) "
            "VALUES (?, ?, ?, 1, ?) "
            "ON CONFLICT(kind, ref) DO UPDATE SET elo = ?, games = games + 1, updated_at = ?",
            (*ent, new_elo, now, new_elo, now),
        )
        await conn.execute(
            "INSERT INTO elo_history (match_id, kind, ref, delta, elo_after) "
            "VALUES (?, ?, ?, ?, ?)",
            (match_id, *ent, round(delta, 2), round(new_elo, 2)),
        )
    await conn.commit()
    return True


async def backfill():
    """Rate every finished rated-format match not yet in elo_history, oldest first."""
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


# `score_from_margin` a vécu ici entre R3 et le passage à la partie. La note à la
# marge reste une bonne idée — une partie gagnée 2000-400 en dit plus qu'un
# 2000-1950 — mais elle demande l'**écart-type des marges de partie**, qui n'est
# pas mesuré. L'inventer serait refaire exactement l'erreur que ce module passe
# son temps à documenter. Une partie agrège déjà ~10 donnes, donc le binaire y
# est bien moins pauvre qu'à la donne : le manque est réel mais borné.
#
# `arena.rs` garde sa ligne « Note à la marge », qui reste juste — elle décrit
# l'échelle **par donne**, utile pour comparer deux bots, et qui n'est plus celle
# du site. Les deux ne sont plus couplées.


def bot_elo(name):
    """Elo figé d'un bot. Un bot inconnu vaut l'origine de l'échelle.

    Le repli sur `START_ELO` n'est pas anodin : il fait qu'un nouveau type de bot
    non étalonné est traité comme l'égal de Dédé. C'est le bon défaut — il vaut
    mieux une hypothèse visible et fausse qu'un bot qui dérive — mais tout bot
    ajouté doit passer par un h2h avant d'être assis en production.
    """
    return BOT_ELO.get(name, START_ELO)


async def get_rating(kind, ref):
    if kind == "bot":
        conn = await db.get_db()
        rows = await conn.execute_fetchall(
            "SELECT games FROM elo_ratings WHERE kind = 'bot' AND ref = ?", (str(ref),))
        return {"elo": bot_elo(str(ref)), "games": rows[0][0] if rows else 0}
    conn = await db.get_db()
    rows = await conn.execute_fetchall(
        "SELECT elo, games FROM elo_ratings WHERE kind = ? AND ref = ?",
        (kind, str(ref)))
    if not rows:
        return {"elo": START_ELO, "games": 0}
    return {"elo": round(rows[0][0], 1), "games": rows[0][1]}


async def leaderboard():
    """Entités classées, meilleure d'abord, avec le pseudo pour les humains.

    Un humain n'apparaît qu'à partir de `MIN_RATED_MATCHES` parties : en dessous,
    son Elo existe et se construit, mais l'afficher serait publier du bruit. Les
    bots sont toujours là — ce sont les étalons, leur Elo ne dépend d'aucune
    partie jouée.

    `provisional` et `remaining` accompagnent chaque ligne pour que la page
    puisse dire « encore 3 parties » plutôt que de faire disparaître quelqu'un
    sans explication.
    """
    conn = await db.get_db()
    rows = await conn.execute_fetchall(
        "SELECT r.kind, r.ref, r.elo, r.games, u.username "
        "FROM elo_ratings r "
        "LEFT JOIN users u ON r.kind = 'user' AND u.id = CAST(r.ref AS INTEGER) "
        "ORDER BY r.elo DESC",
    )
    out = []
    for kind, ref, elo, games, username in rows:
        provisional = kind == "user" and games < MIN_RATED_MATCHES
        if provisional:
            continue
        out.append({
            "kind": kind,
            "ref": ref,
            "elo": round(elo, 1),
            "games": games,
            "name": username if kind == "user" else ref,
        })
    return out


async def standing(kind, ref):
    """État du classement d'une entité, y compris quand elle n'est pas classée."""
    r = await get_rating(kind, ref)
    games = r["games"]
    ranked = kind == "bot" or games >= MIN_RATED_MATCHES
    return {
        **r,
        "ranked": ranked,
        "needed": MIN_RATED_MATCHES,
        "remaining": max(0, MIN_RATED_MATCHES - games) if kind == "user" else 0,
    }
