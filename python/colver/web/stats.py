"""Les chiffres « vie du site » : classements légers et statistiques personnelles.

Ce module est le pendant non-noble d'`elo.py`. L'Elo note la **partie en 2000
points** et rien d'autre ; ici on regarde tout le reste — les donnes isolées,
les parties en 1000, l'assiduité, les faits d'armes.

**La distinction qui structure tout le fichier : compter n'est pas estimer.**

Un *compte* (capots réalisés, jours de jeu, coinches) est exact. Il n'a pas
d'intervalle de confiance parce qu'il n'estime rien, et il peut donc être trié
sans mentir, même sur trois lignes.

Un *taux* (donnes gagnées, contrats tenus) est une estimation, et à l'échelle
de ce site il n'ordonne personne : mesuré sur le corpus réel, une seule paire de
joueurs sur trois se sépare sur les donnes gagnées, aucune sur les chutes en
défense, aucune sur la hauteur d'annonce, aucune sur les capots en taux. C'est
arithmétique et non conjoncturel — un taux à la donne récolte une observation
par donne, quand le coût DD par décision en récolte cinq.

**Donc : les comptes vont au classement, les taux vont dans « Mes stats ».**
Aucun taux n'est trié entre joueurs par ce module, et chacun voyage avec son
`n` et son intervalle (Wilson pour une proportion, normal pour une moyenne) —
faute de quoi trois lignes de bruit se lisent comme un ordre.

**Deux échelles de points, à ne jamais confondre** : `games.points_ns/ew` sont
les points *cartes* (0-252), `games.score_ns/ew` les points *marqués* (v16).
« Gagner une donne » se décide au second. C'est de cette confusion que venait
le bug de `user_game_stats`, qui comptait comme victoire toute chute où le
preneur gardait la majorité des cartes.
"""

import logging
import math

import colver.web.database as db

logger = logging.getLogger(__name__)

# Un siège humain identifié, quel que soit le mode. C'est la brique de toutes
# les requêtes du module : le solo porte le joueur sur `games.user_id` +
# `human_seat`, le salon sur `game_players`, et rien ne doit dépendre de cette
# différence en aval.
#
# Les sièges pairs sont N-S, les impairs E-O — d'où les quatre `CASE` qui
# réorientent les points du côté du joueur, une fois pour toutes.
_SEATS = """
SELECT g.id AS game_id, g.user_id AS uid, g.human_seat AS seat,
       g.created_at, g.match_id, g.actions, g.contract,
       CASE WHEN g.human_seat % 2 = 0 THEN g.points_ns ELSE g.points_ew END AS my_pts,
       CASE WHEN g.human_seat % 2 = 0 THEN g.points_ew ELSE g.points_ns END AS opp_pts,
       CASE WHEN g.human_seat % 2 = 0 THEN g.score_ns ELSE g.score_ew END AS my_score,
       CASE WHEN g.human_seat % 2 = 0 THEN g.score_ew ELSE g.score_ns END AS opp_score
FROM games g
WHERE g.mode = 'play' AND g.is_complete = 1 AND g.invalid = 0
  AND g.user_id IS NOT NULL AND g.human_seat IS NOT NULL
UNION ALL
SELECT g.id, gp.user_id, gp.seat,
       g.created_at, g.match_id, g.actions, g.contract,
       CASE WHEN gp.seat % 2 = 0 THEN g.points_ns ELSE g.points_ew END,
       CASE WHEN gp.seat % 2 = 0 THEN g.points_ew ELSE g.points_ns END,
       CASE WHEN gp.seat % 2 = 0 THEN g.score_ns ELSE g.score_ew END,
       CASE WHEN gp.seat % 2 = 0 THEN g.score_ew ELSE g.score_ns END
FROM games g JOIN game_players gp ON gp.game_id = g.id
WHERE g.mode = 'multi' AND g.is_complete = 1 AND g.invalid = 0
"""

# Le siège qui a pris le contrat : la **dernière** annonce chiffrée de
# l'enchère (actions 1-40 ; 41/42 sont contre et surcontre, 0 la passe).
# `games.contract` ne porte que l'*équipe* (`$.team`), or en solo trois sièges
# sur quatre sont des bots : « mon camp a pris » ne dit pas « j'ai pris ».
#
# `json_each` numérote les éléments dans `key`, donc « la dernière » est un
# simple ORDER BY DESC — aucun rejeu de la donne.
_TAKER = """
    (SELECT json_extract(j.value, '$.player') FROM json_each(s.actions) j
      WHERE json_extract(j.value, '$.phase') = 0
        AND json_extract(j.value, '$.action') BETWEEN 1 AND 40
      ORDER BY j.key DESC LIMIT 1)
"""

# Combien de fois ce siège a annoncé, coinché, passé — et la hauteur moyenne de
# ses annonces chiffrées. `action` encode `(valeur-80)/10*4 + couleur + 1` pour
# 1-40, d'où la valeur : 80 + 10*((action-1)/4). Les capots (37-40) tombent à
# 160 par cette formule, ce qui est faux (un capot vaut 250) mais volontaire :
# la hauteur mesure l'agressivité de l'enchère, et 8 capots annoncés sur 1 177
# donnes ne doivent pas tirer une moyenne à eux seuls. Ils sont comptés à part.
_BID_AGG = """
    (SELECT COUNT(*) FROM json_each(s.actions) j
      WHERE json_extract(j.value, '$.phase') = 0
        AND json_extract(j.value, '$.player') = s.seat
        AND json_extract(j.value, '$.action') = 0) AS passes,
    (SELECT COUNT(*) FROM json_each(s.actions) j
      WHERE json_extract(j.value, '$.phase') = 0
        AND json_extract(j.value, '$.player') = s.seat
        AND json_extract(j.value, '$.action') BETWEEN 1 AND 40) AS bids,
    (SELECT SUM(80 + 10 * ((json_extract(j.value, '$.action') - 1) / 4))
       FROM json_each(s.actions) j
      WHERE json_extract(j.value, '$.phase') = 0
        AND json_extract(j.value, '$.player') = s.seat
        AND json_extract(j.value, '$.action') BETWEEN 1 AND 36) AS bid_sum,
    (SELECT COUNT(*) FROM json_each(s.actions) j
      WHERE json_extract(j.value, '$.phase') = 0
        AND json_extract(j.value, '$.player') = s.seat
        AND json_extract(j.value, '$.action') BETWEEN 1 AND 36) AS bid_n,
    (SELECT COUNT(*) FROM json_each(s.actions) j
      WHERE json_extract(j.value, '$.phase') = 0
        AND json_extract(j.value, '$.player') = s.seat
        AND json_extract(j.value, '$.action') BETWEEN 37 AND 40) AS capots_bid,
    (SELECT COUNT(*) FROM json_each(s.actions) j
      WHERE json_extract(j.value, '$.phase') = 0
        AND json_extract(j.value, '$.player') = s.seat
        AND json_extract(j.value, '$.action') = 41) AS coinches,
    (SELECT COUNT(*) FROM json_each(s.actions) j
      WHERE json_extract(j.value, '$.phase') = 0
        AND json_extract(j.value, '$.player') = s.seat
        AND json_extract(j.value, '$.action') = 42) AS surcoinches
"""


# ===== Intervalles =====

def wilson(successes, total, z=1.96):
    """Intervalle de Wilson pour une proportion. Rend (p, bas, haut) en %.

    Wilson et non Wald : à n = 20 et p proche de 0 ou 1, l'intervalle normal
    sort de [0, 1] et prétend à une précision qu'il n'a pas. C'est déjà le
    choix fait ailleurs dans le site (`annonces.js`).
    """
    if not total:
        return None, None, None
    p = successes / total
    d = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / d
    demi = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return (round(100 * p, 1),
            round(100 * max(0.0, centre - demi), 1),
            round(100 * min(1.0, centre + demi), 1))


def mean_ci(values, z=1.96):
    """Moyenne, demi-intervalle à 95 % et médiane d'un échantillon.

    La médiane accompagne toujours la moyenne ici : les scores de donne ont des
    queues lourdes (une chute contrée marque plusieurs centaines de points d'un
    coup), et une moyenne seule s'y lit mal.
    """
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "ci": None, "median": None}
    mean = sum(values) / n
    if n < 2:
        return {"n": n, "mean": round(mean, 1), "ci": None,
                "median": round(float(values[0]), 1)}
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    ordered = sorted(values)
    mid = n // 2
    median = ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    return {
        "n": n,
        "mean": round(mean, 1),
        "ci": round(z * math.sqrt(var / n), 1),
        "median": round(float(median), 1),
    }


def _pct(values, q):
    """Percentile par interpolation linéaire, sur une liste déjà triée."""
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    pos = q * (len(values) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(values) - 1)
    return float(values[lo] + (values[hi] - values[lo]) * (pos - lo))


# ===== Classement « vie du site » =====

async def leaderboard():
    """Les deux tableaux de l'onglet « Vie du site ». **Que des comptes.**

    Aucun taux n'est trié ici, et aucun bot n'y figure : ces tableaux disent qui
    fait vivre le site, pas qui joue le mieux. Un bot tient trois sièges sur
    quatre et jouerait tous les jours — il gagnerait chaque colonne sans que ça
    veuille dire quoi que ce soit.

    Les jours sont comptés en **UTC**, pas en heure locale : c'est la même règle
    pour tout le monde et elle ne dépend pas du fuseau du serveur. Conséquence
    assumée — une donne jouée à 01 h du matin en France compte pour la veille.
    """
    conn = await db.get_db()
    rows = await conn.execute_fetchall(f"""
        WITH s AS ({_SEATS}),
        base AS (
            SELECT uid,
                   COUNT(*) AS deals,
                   COUNT(DISTINCT date(created_at)) AS days,
                   SUM(CASE WHEN my_pts = 252 THEN 1 ELSE 0 END) AS capots_for,
                   SUM(CASE WHEN opp_pts = 252 THEN 1 ELSE 0 END) AS capots_against,
                   MAX(date(created_at)) AS last_day
              FROM s GROUP BY uid
        ),
        acts AS (
            SELECT uid,
                   SUM((SELECT COUNT(*) FROM json_each(s.actions) j
                         WHERE json_extract(j.value, '$.phase') = 0
                           AND json_extract(j.value, '$.player') = s.seat
                           AND json_extract(j.value, '$.action') = 41)) AS coinches,
                   SUM(CASE WHEN {_TAKER} = s.seat THEN 1 ELSE 0 END) AS takes
              FROM s GROUP BY uid
        ),
        formats AS (
            SELECT m.user_id AS uid,
                   SUM(CASE WHEN m.target = 1000 THEN 1 ELSE 0 END) AS m1000,
                   SUM(CASE WHEN m.target = 2000 THEN 1 ELSE 0 END) AS m2000
              FROM matches m
             WHERE m.is_complete = 1 AND m.abandoned = 0 AND m.user_id IS NOT NULL
             GROUP BY m.user_id
        )
        SELECT u.id, u.username, base.deals, base.days, base.last_day,
               base.capots_for, base.capots_against,
               COALESCE(acts.coinches, 0), COALESCE(acts.takes, 0),
               COALESCE(formats.m1000, 0), COALESCE(formats.m2000, 0)
          FROM base
          JOIN users u ON u.id = base.uid
          LEFT JOIN acts ON acts.uid = base.uid
          LEFT JOIN formats ON formats.uid = base.uid
         ORDER BY base.deals DESC
    """)

    streaks = await _streaks(conn)
    out = []
    for (uid, name, deals, days, last_day, cap_for, cap_against,
         coinches, takes, m1000, m2000) in rows:
        out.append({
            "user_id": uid,
            "name": name,
            "deals": deals,
            "days": days,
            "density": round(deals / days, 1) if days else None,
            "streak": streaks.get(uid, 0),
            "last_day": last_day,
            "capots_for": cap_for or 0,
            "capots_against": cap_against or 0,
            "coinches": coinches or 0,
            "takes": takes or 0,
            "matches_1000": m1000 or 0,
            "matches_2000": m2000 or 0,
        })
    return out


async def _streaks(conn):
    """Jours consécutifs joués, en cours, par joueur.

    Technique des « îlots » : on numérote les jours distincts d'un joueur, puis
    on retranche le rang au jour. Deux jours consécutifs donnent la même clé,
    un trou en crée une nouvelle — un GROUP BY suffit ensuite à mesurer chaque
    plage sans boucle.

    Une série n'est « en cours » que si elle touche aujourd'hui ou hier.
    Exiger aujourd'hui la ferait tomber à zéro tous les matins pour tout le
    monde ; l'étendre plus loin appellerait « en cours » une série finie.
    """
    rows = await conn.execute_fetchall(f"""
        WITH s AS ({_SEATS}),
        days AS (SELECT DISTINCT uid, date(created_at) AS d FROM s),
        numbered AS (
            SELECT uid, d, ROW_NUMBER() OVER (PARTITION BY uid ORDER BY d) AS rn
              FROM days
        ),
        islands AS (
            SELECT uid, d, date(d, '-' || rn || ' days') AS grp FROM numbered
        )
        SELECT uid, COUNT(*) AS len, MAX(d) AS last_day
          FROM islands GROUP BY uid, grp
    """)
    best = {}
    for uid, length, last_day in rows:
        if last_day in (_today(), _yesterday()):
            best[uid] = max(best.get(uid, 0), length)
    return best


def _today():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).date().isoformat()


def _yesterday():
    from datetime import datetime, timedelta, timezone
    return (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()


# ===== « Mes stats » =====

async def my_stats(user_id):
    """Le portrait chiffré d'un joueur. Personnel, jamais classé.

    Tout ce qui est ici est un taux ou une moyenne, donc une estimation : chaque
    entrée porte son `n` et son intervalle, et l'appelant doit les afficher.
    Un chiffre sans son `n` sur un corpus de cinquante donnes est une affirmation
    que les données ne soutiennent pas.

    Une donne dont le score marqué n'est pas encore rattrapé (`score_ns IS NULL`)
    est exclue des taux qui en dépendent, jamais comptée comme une défaite.
    """
    conn = await db.get_db()
    rows = await conn.execute_fetchall(f"""
        WITH s AS ({_SEATS})
        SELECT s.game_id, s.created_at, s.match_id, s.seat,
               s.my_pts, s.opp_pts, s.my_score, s.opp_score,
               json_extract(s.contract, '$.value') AS contract_value,
               json_extract(s.contract, '$.team') AS contract_team,
               json_extract(s.contract, '$.coinche') AS coinche_level,
               {_TAKER} AS taker,
               {_BID_AGG}
          FROM s WHERE s.uid = ?
         ORDER BY s.created_at
    """, (user_id,))

    deals = len(rows)
    if deals == 0:
        return {"deals": 0}

    margins, wins, scored = [], 0, 0
    takes, takes_held, take_values = 0, 0, []
    defenses, defenses_won = 0, 0
    passes = bids = bid_sum = bid_n = capots_bid = coinches = surcoinches = 0
    capots_for = capots_against = 0
    days = set()

    for r in rows:
        (_gid, created, _match, seat, my_pts, opp_pts, my_score, opp_score,
         c_value, c_team, c_coinche, taker,
         n_pass, n_bid, b_sum, b_n, n_capot, n_coinche, n_surcoinche) = r

        days.add(created[:10])
        passes += n_pass or 0
        bids += n_bid or 0
        bid_sum += b_sum or 0
        bid_n += b_n or 0
        capots_bid += n_capot or 0
        coinches += n_coinche or 0
        if my_pts == 252:
            capots_for += 1
        if opp_pts == 252:
            capots_against += 1

        surcoinches += n_surcoinche or 0

        if my_score is None or opp_score is None:
            continue  # rattrapage des scores pas encore passé
        scored += 1
        margins.append(my_score - opp_score)
        if my_score > opp_score:
            wins += 1

        if taker is None or c_team is None:
            continue  # donne passée (4 passes) : ni preneur ni défense
        # Le contrat est tenu si le camp preneur marque : sous ce barème une
        # chute lui donne exactement 0, et un contrat réussi au moins 3V − 162.
        taker_scored = (my_score if taker % 2 == seat % 2 else opp_score) > 0
        if taker == seat:
            takes += 1
            take_values.append(c_value)
            if taker_scored:
                takes_held += 1
        elif taker % 2 != seat % 2:
            defenses += 1
            if not taker_scored:
                defenses_won += 1

    won_pct, won_lo, won_hi = wilson(wins, scored)
    held_pct, held_lo, held_hi = wilson(takes_held, takes)
    def_pct, def_lo, def_hi = wilson(defenses_won, defenses)

    return {
        "deals": deals,
        "scored": scored,
        "days": len(days),
        "density": round(deals / len(days), 1) if days else None,
        "won": {"n": scored, "k": wins, "pct": won_pct, "lo": won_lo, "hi": won_hi},
        "margin": mean_ci(margins),
        "takes": {
            "n": takes,
            "per_100": round(100 * takes / deals, 1) if deals else None,
            "avg_value": round(sum(take_values) / len(take_values), 1)
                         if take_values else None,
            "held_pct": held_pct, "held_lo": held_lo, "held_hi": held_hi,
        },
        "defense": {"n": defenses, "k": defenses_won,
                    "pct": def_pct, "lo": def_lo, "hi": def_hi},
        "bidding": {
            "pass_pct": round(100 * passes / (passes + bids), 1)
                        if (passes + bids) else None,
            "decisions": passes + bids,
            "avg_height": round(bid_sum / bid_n, 1) if bid_n else None,
            "height_n": bid_n,
            "capots": capots_bid,
        },
        "coinches": coinches,
        "surcoinches": surcoinches,
        "capots_for": capots_for,
        "capots_against": capots_against,
        "tempo": await _tempo(user_id),
    }


# Au-delà, l'écart entre deux donnes d'une même partie ne mesure plus une
# donne : le joueur s'est absenté et il est revenu. Une donne coûte ~42 s au
# tempo standard et ~16 s au tempo rapide, donc ce plafond est ~20× la valeur
# attendue — assez large pour garder toute donne réellement jouée, assez serré
# pour écarter un déjeuner. Les écarts retirés sont **comptés et rendus**, pour
# qu'une troncature ne passe pas pour une mesure complète.
_TEMPO_MAX_S = 15 * 60


async def _tempo(user_id):
    """Durée d'une donne, en secondes, mesurée **à l'intérieur d'une partie**.

    Rien n'enregistre la fin d'une donne : seul `created_at` existe. L'écart
    entre deux `created_at` consécutifs est donc la seule mesure disponible — et
    elle n'a de sens qu'entre deux donnes d'une **même partie**, qui
    s'enchaînent. Entre deux donnes isolées, l'écart mesure surtout le temps
    passé ailleurs : mesuré sur le corpus réel, la moyenne monte à 2 353 s pour
    une médiane de 80 s.

    Deux protections, parce que la médiane seule ne suffit pas : le plafond
    ci-dessus écarte les interruptions, et on rend des percentiles plutôt
    qu'une moyenne. Le compte accompagne le chiffre — trois donnes ne doivent
    pas se lire comme une mesure.
    """
    conn = await db.get_db()
    rows = await conn.execute_fetchall(f"""
        WITH s AS ({_SEATS}),
        mine AS (
            SELECT match_id, created_at,
                   LAG(created_at) OVER (PARTITION BY match_id ORDER BY created_at)
                       AS prev
              FROM s WHERE uid = ? AND match_id IS NOT NULL
        )
        SELECT (julianday(created_at) - julianday(prev)) * 86400.0
          FROM mine WHERE prev IS NOT NULL
    """, (user_id,))
    raw = [float(r[0]) for r in rows if r[0] is not None and r[0] > 0]
    gaps = sorted(g for g in raw if g <= _TEMPO_MAX_S)
    if not gaps:
        return {"n": 0, "dropped": len(raw)}
    return {
        "n": len(gaps),
        "dropped": len(raw) - len(gaps),
        "p25": round(_pct(gaps, 0.25)),
        "median": round(_pct(gaps, 0.5)),
        "p75": round(_pct(gaps, 0.75)),
    }
