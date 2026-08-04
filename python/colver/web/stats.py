"""Le portrait chiffré d'un joueur : tout ce qu'on peut dire de lui sans rien calculer.

Ce module est le pendant personnel d'`elo.py`. L'Elo note la **partie en 2000
points** et ordonne des joueurs ; ici on ne classe rien — on décrit quelqu'un.

**Pourquoi il n'y a pas de classement ici, et pourquoi il ne faut pas en
remettre un.** Un onglet « Vie du site » a existé une journée : des comptes de
capots, de jours joués et de séries, triés entre joueurs. Il est parti parce
qu'il n'apprenait rien. Sur une communauté de quelques comptes, ordonner des
gens par leur assiduité produit un tableau que tout le monde lit une fois. Les
mêmes chiffres, rendus à leur propriétaire, sont bien plus intéressants : « vous
prenez une fois sur quatre » dit quelque chose, « vous êtes troisième en nombre
de prises » ne dit rien.

**Compter n'est pas estimer**, et la distinction survit au changement de forme.
Un compte (donnes jouées, capots, série) est exact et se donne tel quel. Un taux
(donnes gagnées, contrats tenus) est une estimation : sur quelques dizaines
d'observations il porte un intervalle de dix points, donc il voyage toujours
avec son `n` et son intervalle — Wilson pour une proportion, écart-type pour une
moyenne — et l'affichage refuse de le montrer sous cinq observations.

**Deux échelles de points, à ne jamais confondre** : `games.points_ns/ew` sont
les points *cartes* (0-252), `games.score_ns/ew` les points *marqués* (v16).
« Gagner une donne » se décide au second. C'est de cette confusion que venait le
bug de `user_game_stats`, qui comptait comme victoire toute chute où le preneur
gardait la majorité des cartes.

**Tout ce fichier est gratuit** : du SQL sur des colonnes déjà écrites, plus
JSON1 sur `games.actions` et `games.hands`. Rien n'y rejoue une donne, rien n'y
appelle un modèle. Ce qui coûte — le jeu comparé à l'oracle — vit dans
`analysis.py` et ne se déclenche qu'à la demande explicite du joueur.
"""

import json
import logging
import math

import colver.web.database as db
from colver.web.analysis import ANALYSIS_VERSION

logger = logging.getLogger(__name__)

# Un siège humain identifié, quel que soit le mode. C'est la brique de toutes
# les requêtes du module : le solo porte le joueur sur `games.user_id` +
# `human_seat`, le salon sur `game_players`, et rien ne doit dépendre de cette
# différence en aval.
#
# Les sièges pairs sont N-S, les impairs E-O — d'où les quatre `CASE` qui
# réorientent les points du côté du joueur, une fois pour toutes.
_SEATS = """
SELECT g.id AS game_id, g.user_id AS uid, g.human_seat AS seat, g.mode,
       g.created_at, g.match_id, g.actions, g.contract, g.hands,
       CASE WHEN g.human_seat % 2 = 0 THEN g.points_ns ELSE g.points_ew END AS my_pts,
       CASE WHEN g.human_seat % 2 = 0 THEN g.points_ew ELSE g.points_ns END AS opp_pts,
       CASE WHEN g.human_seat % 2 = 0 THEN g.score_ns ELSE g.score_ew END AS my_score,
       CASE WHEN g.human_seat % 2 = 0 THEN g.score_ew ELSE g.score_ns END AS opp_score
FROM games g
WHERE g.mode = 'play' AND g.is_complete = 1 AND g.invalid = 0
  AND g.user_id IS NOT NULL AND g.human_seat IS NOT NULL
UNION ALL
SELECT g.id, gp.user_id, gp.seat, g.mode,
       g.created_at, g.match_id, g.actions, g.contract, g.hands,
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
# sur quatre sont des bots : « mon camp a pris » ne dit pas « j'ai pris », et
# c'est précisément la question à laquelle cette page répond.
#
# `json_each` numérote les éléments dans `key`, donc « la dernière » est un
# simple ORDER BY DESC — aucun rejeu de la donne.
_TAKER = """
    (SELECT json_extract(j.value, '$.player') FROM json_each(s.actions) j
      WHERE json_extract(j.value, '$.phase') = 0
        AND json_extract(j.value, '$.action') BETWEEN 1 AND 40
      ORDER BY j.key DESC LIMIT 1)
"""

# La belote : Dame **et** Roi d'atout dans la main initiale du siège. Elle est
# déductible sans rejouer quoi que ce soit — `games.hands` porte la donne, le
# contrat porte l'atout — alors que `state.belote` ne compte que ce qui a déjà
# été *joué*. Les rangs sont ceux de `card.rs` : Dame = 4, Roi = 5, décalés de
# 8 par couleur. Rend 2 quand les deux y sont.
_BELOTE = """
    (SELECT COUNT(*) FROM json_each(json_extract(s.hands, '$[' || s.seat || ']')) k
      WHERE k.value IN (json_extract(s.contract, '$.trump') * 8 + 4,
                        json_extract(s.contract, '$.trump') * 8 + 5))
"""

# Combien de fois ce siège a annoncé, coinché, passé — et la hauteur moyenne de
# ses annonces chiffrées. `action` encode `(valeur-80)/10*4 + couleur + 1` pour
# 1-40, d'où la valeur : 80 + 10*((action-1)/4). Les capots (37-40) tombent à
# 160 par cette formule, ce qui est faux (un capot vaut 250) mais volontaire :
# la hauteur mesure l'agressivité de l'enchère ordinaire, et un capot annoncé
# tirerait une moyenne à lui seul. Ils sont comptés à part.
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

SUITS = ["♠", "♥", "♦", "♣"]

# Au-delà, l'écart entre deux donnes d'une même partie ne mesure plus une
# donne : le joueur s'est absenté et il est revenu. Une donne coûte ~42 s au
# tempo standard et ~16 s au tempo rapide, donc ce plafond est ~20× la valeur
# attendue — assez large pour garder toute donne réellement jouée, assez serré
# pour écarter un déjeuner. Les écarts retirés sont **comptés et rendus**, pour
# qu'une troncature ne passe pas pour une mesure complète.
_TEMPO_MAX_S = 15 * 60


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


def rate(successes, total, extra=None):
    """Un taux prêt à afficher : valeur, bornes, effectifs."""
    p, lo, hi = wilson(successes, total)
    out = {"n": total, "k": successes, "pct": p, "lo": lo, "hi": hi}
    if extra:
        out.update(extra)
    return out


def mean_ci(values, z=1.96):
    """Moyenne, demi-intervalle à 95 % et médiane d'un échantillon.

    La médiane accompagne toujours la moyenne ici : les scores de donne ont des
    queues lourdes (une chute contrée marque plusieurs centaines de points d'un
    coup), et une moyenne seule s'y lit mal. Mesuré sur le corpus réel : moyenne
    +38 pour une médiane de +206.
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


# ===== Le portrait =====

async def my_stats(user_id):
    """Tout ce qu'on sait d'un joueur sans rien calculer.

    Une seule requête ramène une ligne par donne jouée, avec ses agrégats
    d'enchère déjà réduits en SQL ; le reste se compte en Python, où la logique
    se lit. Le volume est borné par ce que la personne a joué — quelques
    centaines de lignes — et non par la taille de la base.

    Une donne dont le score marqué n'est pas encore rattrapé (`score_ns IS NULL`)
    est exclue des taux qui en dépendent, jamais comptée comme une défaite.
    """
    conn = await db.get_db()
    rows = await conn.execute_fetchall(f"""
        WITH s AS ({_SEATS})
        SELECT s.game_id, s.created_at, s.match_id, s.seat, s.mode,
               s.my_pts, s.opp_pts, s.my_score, s.opp_score,
               json_extract(s.contract, '$.value') AS contract_value,
               json_extract(s.contract, '$.team') AS contract_team,
               json_extract(s.contract, '$.trump') AS trump,
               json_extract(s.contract, '$.coinche') AS coinche_level,
               {_TAKER} AS taker,
               {_BELOTE} AS belote_cards,
               {_BID_AGG}
          FROM s WHERE s.uid = ?
         ORDER BY s.created_at
    """, (user_id,))

    if not rows:
        return {"deals": 0}

    margins, wins, scored = [], 0, 0
    takes, takes_held, take_values, take_trumps = 0, 0, [], [0, 0, 0, 0]
    partner_takes, opponent_takes, passed_deals = 0, 0, 0
    defenses, defenses_won = 0, 0
    passes = bids = bid_sum = bid_n = capots_bid = coinches = surcoinches = 0
    capots_for = capots_against = belotes = contres = 0
    days, modes = set(), {"play": 0, "multi": 0}

    for r in rows:
        (_gid, created, _match, seat, mode, my_pts, opp_pts, my_score, opp_score,
         c_value, c_team, trump, c_coinche, taker, belote_cards,
         n_pass, n_bid, b_sum, b_n, n_capot, n_coinche, n_surcoinche) = r

        days.add(created[:10])
        modes[mode] = modes.get(mode, 0) + 1
        passes += n_pass or 0
        bids += n_bid or 0
        bid_sum += b_sum or 0
        bid_n += b_n or 0
        capots_bid += n_capot or 0
        coinches += n_coinche or 0
        surcoinches += n_surcoinche or 0
        if my_pts == 252:
            capots_for += 1
        if opp_pts == 252:
            capots_against += 1
        if belote_cards == 2:
            belotes += 1
        if c_coinche:
            contres += 1

        if taker is None or c_team is None:
            passed_deals += 1
        elif taker == seat:
            takes += 1
            take_values.append(c_value)
            if trump is not None and 0 <= trump < 4:
                take_trumps[trump] += 1
        elif taker % 2 == seat % 2:
            partner_takes += 1
        else:
            opponent_takes += 1

        if my_score is None or opp_score is None:
            continue  # rattrapage des scores pas encore passé
        scored += 1
        margins.append(my_score - opp_score)
        if my_score > opp_score:
            wins += 1

        if taker is None or c_team is None:
            continue
        # Le contrat est tenu si le camp preneur marque : sous ce barème une
        # chute lui donne exactement 0, et un contrat réussi au moins 3V − 162.
        taker_scored = (my_score if taker % 2 == seat % 2 else opp_score) > 0
        if taker == seat and taker_scored:
            takes_held += 1
        elif taker % 2 != seat % 2:
            defenses += 1
            if not taker_scored:
                defenses_won += 1

    deals = len(rows)
    contested = takes + partner_takes + opponent_takes
    return {
        "deals": deals,
        "scored": scored,
        "days": len(days),
        "streak": await _streak(user_id),
        "density": round(deals / len(days), 1) if days else None,
        "modes": modes,
        "won": rate(wins, scored),
        "margin": mean_ci(margins),
        # Qui prend, sur les donnes qui ont trouvé preneur. Les quatre passes
        # sont exclues du dénominateur : personne n'y a « pris ».
        "who_takes": {
            "n": contested,
            "me": takes,
            "partner": partner_takes,
            "opponents": opponent_takes,
            "passed": passed_deals,
            "me_pct": round(100 * takes / contested, 1) if contested else None,
            "partner_pct": round(100 * partner_takes / contested, 1) if contested else None,
        },
        "takes": {
            "n": takes,
            "per_100": round(100 * takes / deals, 1) if deals else None,
            "avg_value": round(sum(take_values) / len(take_values), 1)
                         if take_values else None,
            "held": rate(takes_held, takes),
            "trumps": [
                {"suit": SUITS[i], "n": n,
                 "pct": round(100 * n / takes, 1) if takes else None}
                for i, n in enumerate(take_trumps)
            ],
        },
        "defense": rate(defenses_won, defenses),
        "bidding": {
            "decisions": passes + bids,
            "pass": rate(passes, passes + bids),
            "avg_height": round(bid_sum / bid_n, 1) if bid_n else None,
            "height_n": bid_n,
            "capots": capots_bid,
        },
        "coinches": coinches,
        "surcoinches": surcoinches,
        "contres_played": contres,
        "capots_for": capots_for,
        "capots_against": capots_against,
        "belotes": belotes,
        "partners": await _partners(user_id),
        "tempo": await _tempo(user_id),
        "activity": await activity(user_id),
    }


async def _streak(user_id):
    """Jours consécutifs joués, en cours.

    Technique des « îlots » : on numérote les jours distincts, puis on retranche
    le rang au jour. Deux jours consécutifs donnent la même clé, un trou en crée
    une nouvelle — un GROUP BY suffit ensuite à mesurer chaque plage sans
    boucle.

    Une série n'est « en cours » que si elle touche aujourd'hui ou hier. Exiger
    aujourd'hui la ferait tomber à zéro tous les matins pour tout le monde ;
    l'étendre plus loin appellerait « en cours » une série finie.

    Les jours sont comptés en **UTC** : la même règle pour tout le monde, qui ne
    dépend pas du fuseau du serveur. Conséquence assumée — une donne jouée à
    01 h du matin en France compte pour la veille.
    """
    conn = await db.get_db()
    rows = await conn.execute_fetchall(f"""
        WITH s AS ({_SEATS}),
        days AS (SELECT DISTINCT date(created_at) AS d FROM s WHERE uid = ?),
        numbered AS (SELECT d, ROW_NUMBER() OVER (ORDER BY d) AS rn FROM days),
        islands AS (SELECT d, date(d, '-' || rn || ' days') AS grp FROM numbered)
        SELECT COUNT(*) AS len, MAX(d) AS last_day
          FROM islands GROUP BY grp ORDER BY last_day DESC LIMIT 1
    """, (user_id,))
    if not rows:
        return 0
    length, last_day = rows[0]
    return length if last_day in (_today(), _yesterday()) else 0


def _today():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).date().isoformat()


def _yesterday():
    from datetime import datetime, timedelta, timezone
    return (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()


async def activity(user_id, weeks=12):
    """Donnes par jour, en colonnes de semaines — la matière du calendrier.

    Rend `weeks × 7` entiers, du plus ancien au plus récent, alignés sur les
    semaines : l'indice `i` a pour jour de semaine `i % 7` (0 = lundi). C'est ce
    qui fait que les lignes du calendrier sont des jours de semaine et non un
    décalage arbitraire — sans l'alignement, la grille ne dit plus rien.

    La fenêtre s'arrête à la fin de la semaine **en cours**, pas à aujourd'hui :
    la dernière colonne est donc partielle, et les cases à venir valent `None`
    plutôt que 0. Un zéro dit « vous n'avez pas joué », un `None` dit « ce jour
    n'existe pas encore » — les afficher pareil ferait passer la fin de semaine
    pour une panne d'assiduité.

    Jours en **UTC**, comme `_streak` : une seule règle pour tout le monde.
    """
    from datetime import datetime, timedelta, timezone
    today = datetime.now(timezone.utc).date()
    end = today + timedelta(days=6 - today.weekday())      # dimanche de la semaine
    start = end - timedelta(days=weeks * 7 - 1)            # lundi, weeks plus tôt

    conn = await db.get_db()
    rows = await conn.execute_fetchall(f"""
        WITH s AS ({_SEATS})
        SELECT date(created_at) AS d, COUNT(*) AS n
          FROM s WHERE uid = ? AND date(created_at) >= ?
         GROUP BY d
    """, (user_id, start.isoformat()))
    counts = {r[0]: r[1] for r in rows}

    days = []
    for i in range(weeks * 7):
        d = start + timedelta(days=i)
        days.append(None if d > today else counts.get(d.isoformat(), 0))
    return {
        "days": days,
        "weeks": weeks,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "max": max((c for c in days if c), default=0),
    }


async def _partners(user_id, limit=5):
    """Les humains avec qui ce joueur a le plus souvent fait équipe.

    Le partenaire d'un siège est en face : `(seat + 2) % 4`. SQLite n'a pas
    d'opérateur XOR, et c'est la même chose sur 0-3.

    **Salon seulement** : en solo le partenaire est toujours un bot, donc la
    ligne dirait « Dédé » pour tout le monde et n'apprendrait rien. Une liste
    vide veut dire « vous n'avez encore joué qu'avec des bots », et l'affichage
    le dit ainsi plutôt que de montrer une section vide.
    """
    conn = await db.get_db()
    rows = await conn.execute_fetchall("""
        SELECT u.username, COUNT(*) AS n
          FROM game_players me
          JOIN games g ON g.id = me.game_id
          JOIN game_players mate ON mate.game_id = me.game_id
                                AND mate.seat = (me.seat + 2) % 4
          JOIN users u ON u.id = mate.user_id
         WHERE me.user_id = ? AND g.is_complete = 1 AND g.invalid = 0
         GROUP BY u.id ORDER BY n DESC, u.username LIMIT ?
    """, (user_id, limit))
    return [{"name": r[0], "deals": r[1]} for r in rows]


async def _tempo(user_id):
    """Durée d'une donne, en secondes, mesurée **à l'intérieur d'une partie**.

    Rien n'enregistre la fin d'une donne : seul `created_at` existe. L'écart
    entre deux `created_at` consécutifs est donc la seule mesure disponible — et
    elle n'a de sens qu'entre deux donnes d'une **même partie**, qui
    s'enchaînent. Entre deux donnes isolées, l'écart mesure surtout le temps
    passé ailleurs : mesuré sur le corpus réel, la moyenne monte à 2 353 s pour
    une médiane de 80 s.

    Deux protections, parce que la médiane seule ne suffit pas : le plafond
    `_TEMPO_MAX_S` écarte les interruptions, et on rend des percentiles plutôt
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


# ===== Le jeu comparé à l'oracle =====
#
# Contrairement à tout ce qui précède, ces chiffres ne sont pas gratuits : ils
# viennent du solveur double-mort, un solve par décision. On ne les calcule
# donc jamais tout seul — le joueur appuie sur un bouton (`analysis.bulk`), et
# ce qui suit ne fait que relire ce qui est déjà en cache.

async def oracle_stats(user_id):
    """Ce que le solveur dit du jeu de la carte de ce joueur.

    Ne calcule rien : agrège les analyses **déjà en cache** (`analysis.data`)
    pour les donnes du joueur, à la version courante. La couverture est rendue
    avec, et elle est la première chose à afficher — sans elle, un joueur qui
    n'a analysé que ses belles donnes lirait une moyenne flatteuse sans savoir
    qu'elle ne porte que sur un dixième de son jeu.

    **On mesure le coût, pas l'égalité à la carte préférée du solveur.** Une
    décision est « sans perte » quand `cost == 0`, pas quand elle égale
    `best_card` : 57,8 % des positions ont plusieurs cartes DD-optimales, et
    laquelle `solve_with_scores` renvoie dépend de l'ordre de sa boucle racine.
    Compter les égalités à `best_card` déclarerait fautives des décisions que le
    solveur valorise exactement pareil — mesuré 87,4 % contre 59,0 %.

    **Les coups forcés sont exclus du dénominateur.** Quand une seule carte est
    légale il n'y a pas de décision, et les compter gonflerait la part de coups
    parfaits d'un tiers sans rien dire du joueur.
    """
    conn = await db.get_db()
    rows = await conn.execute_fetchall(f"""
        WITH s AS ({_SEATS})
        SELECT s.seat, a.data
          FROM s JOIN analysis a ON a.game_id = s.game_id
         WHERE s.uid = ? AND a.version = ?
    """, (user_id, ANALYSIS_VERSION))

    total = await _analysable_count(user_id)
    decisions = forced = 0
    cost_sum = 0
    costs = []
    counts = {}
    for seat, blob in rows:
        try:
            moves = json.loads(blob).get("moves") or []
        except Exception:  # noqa: BLE001 — un blob illisible ne perd pas le reste
            logger.warning("analyse illisible pour l'utilisateur %s", user_id)
            continue
        for m in moves:
            if m.get("player") != seat:
                continue
            if m.get("forced"):
                forced += 1
                continue
            cost = int(m.get("cost") or 0)
            decisions += 1
            cost_sum += cost
            costs.append(cost)
            label = m.get("category") or "parfait"
            counts[label] = counts.get(label, 0) + 1

    analysed = len(rows)
    out = {
        "analysed": analysed,
        "total": total,
        "pending": max(0, total - analysed),
        "decisions": decisions,
        "forced": forced,
        "counts": counts,
    }
    if decisions:
        out["avg_cost"] = round(cost_sum / decisions, 2)
        out["clean"] = rate(counts.get("parfait", 0), decisions)
        # L'écart-type des coûts est large et la moyenne seule se lit mal :
        # l'intervalle dit à partir de quand deux chiffres diffèrent vraiment.
        out["cost_ci"] = mean_ci(costs)["ci"]
    return out


async def _analysable_count(user_id):
    """Combien de donnes de ce joueur peuvent être analysées, en tout."""
    conn = await db.get_db()
    rows = await conn.execute_fetchall(f"""
        WITH s AS ({_SEATS}) SELECT COUNT(*) FROM s WHERE uid = ?
    """, (user_id,))
    return rows[0][0] if rows else 0


async def unanalysed_games(user_id, limit=2000):
    """Les donnes du joueur qui n'ont pas d'analyse à la version courante.

    Rendues de la plus récente à la plus ancienne : si le travail est
    interrompu, ce qui a été fait est ce que le joueur regarde le plus.
    """
    conn = await db.get_db()
    rows = await conn.execute_fetchall(f"""
        WITH s AS ({_SEATS})
        SELECT s.game_id FROM s
          LEFT JOIN analysis a ON a.game_id = s.game_id AND a.version = ?
         WHERE s.uid = ? AND a.game_id IS NULL
         ORDER BY s.created_at DESC LIMIT ?
    """, (ANALYSIS_VERSION, user_id, limit))
    return [r[0] for r in rows]
