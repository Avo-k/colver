"""Post-game oracle analysis: DD cost of every card played + bid review.

For each play-phase action of a stored game, the position is re-solved with
the double-dummy solver (all hands known) and the played card is compared to
the DD-optimal one. The cost is expressed in real card points (0-162 scale,
252 with capot) from the acting team's perspective.

Bid moves get three extra signals: the bid NN's preferred action from the same
position (model annonce), what the playgen world model would have announced
from that seat's own view (its 43-way bid head, v2 models only), and the
DD-best contract each team could have declared on the deal (oracle annonce,
one solve per trump suit).

Results are cached in the `analysis` table; computation runs in a thread — genuinely so
only since 2026-08-02, when `Env.solve_scores` started releasing the GIL; before that this
whole pass blocked the event loop. Cost is dominated by `_oracle_bids` (four full-deal
solves, ~70 ms); the per-card solves run 34.6 ms at trick 1 down to 1.5 µs at trick 8.
"""

import asyncio
import json
import logging

import colver
import colver.web.database as db
import colver.web.variation as variation

logger = logging.getLogger(__name__)

# v5 : invalide tout cache antérieur au filtre `get_game` (2026-08-01) — une
# analyse calculée en pleine donne sur une donne terminée depuis est partielle,
# indiscernable d'une complète, et la migration v9 ne peut pas l'identifier.
# v6 : l'avis de bid v6 est calculé au score de partie de la donne, plus à 0-0.
# v7 : le coût d'une carte se lit en **score de donne** et non plus en points
#      cartes, et la catégorie avec lui (2026-08-05).
# v8 : le blob porte `curve` — points ramassés, projection DD et seuil, de quoi
#      tracer la donne du premier coup au dernier (2026-08-05).
# v9 : chaque erreur porte ses **deux** variantes déroulées en jeu parfait — la
#      suite du coup joué et celle du coup de l'Oracle (2026-08-06).
# v10 : chaque coup porte `n_legal`, et le blob la **marche du barème**. Sans
#      les deux, `cost_score == 0` se lisait « l'Oracle approuve cette carte »
#      alors qu'il veut dire, 59,7 % du temps, « aucune carte légale ne change
#      le score de donne » — 88,3 % sous contré (2026-08-06).
ANALYSIS_VERSION = 10

# Versions dont les valeurs DD restent bonnes. Ni v7, ni v8, ni v9 ne touchent au
# barème ou aux coups légaux — elles relisent le même solve, ou en gardent une
# valeur qui était jetée — donc `oracle_bids` d'une analyse antérieure reste
# exact. À vider au prochain bump qui touche vraiment aux règles : là, tout ce
# qui précède est périmé.
_DD_COMPATIBLE_VERSIONS = (5, 6, 7, 8, 9, 10)

# ── Ce qu'est une erreur ──
#
# Jusqu'au 2026-08-05 le coût d'une carte se lisait en **points cartes**, avec
# des seuils (4 / 14 / 29). C'était faux pour ce que la page affiche, parce que
# le score de donne est une fonction **en escalier** de ces points : plat sous le
# seuil du contrat, marche de `4V` au seuil, pente 2 seulement dans un contrat
# normal tenu. Mesuré sur 1205 décisions
# (`scripts/analysis/replay_error_scale.py scale`) :
#
#   - **32 coups sur 1057 notés « ✓ »** coûtaient au score, jusqu'à 1264 points ;
#   - **8 « fautes » sur 9** coûtaient bien quelque chose, mais l'échelle notait
#     −42 un coup qui ne décidait plus rien et −3 celui qui perdait la donne ;
#   - trier par points cartes met donc en tête de liste des coups qui n'ont rien
#     coûté, et cache ceux qui ont tout coûté.
#
# La catégorie se lit désormais en score de donne, et se scinde selon **le seul
# prédicat qui compte** : le contrat a-t-il basculé ? Perdre 30 points dans un
# contrat acquis et faire chuter le contrat ne sont pas le même événement.
#
# Ces trois catégories sont ce que le DD seul permet de dire. Le client en
# dérive deux de plus — `malchance` et `aubaine` — en croisant avec l'avis
# d'IS-DD, qui arrive plus tard (`agent_review`) : voir `replay.js`.
CAT_PARFAIT = "parfait"          # rien à gagner à jouer autre chose
CAT_IMPRECISION = "imprecision"  # coûte des points, le contrat tient quand même
CAT_DECISIVE = "decisive"        # le contrat bascule

# L'échelle points cartes reste calculée et publiée (`cost`) : c'est la bonne
# réponse à « quelle carte prend le plus de plis », et `/analyse/jeu` l'affiche.
# Elle n'est simplement plus ce qui décide du mot « erreur ».
CATEGORIES = [CAT_PARFAIT, CAT_IMPRECISION, CAT_DECISIVE]

_locks = {}  # game_id -> asyncio.Lock, to avoid duplicate computations


def _categorize(cost_score, swing):
    """La catégorie d'un coup, depuis son coût **en score de donne**."""
    if cost_score <= 0:
        return CAT_PARFAIT
    return CAT_DECISIVE if swing else CAT_IMPRECISION


def _best_contract(suits, team):
    """DD-best declarable contract for a team from per-suit DD points.

    Returns {suit, pts, value} — value 0 when no contract is makeable (< 80),
    250 (capot) when the team takes every point (162).
    """
    pts = [int(s[team]) for s in suits]
    best_suit = max(range(4), key=lambda s: pts[s])
    p = pts[best_suit]
    if p >= 162:
        value = 250
    else:
        value = min(160, p // 10 * 10) if p >= 80 else 0
    return {"suit": best_suit, "pts": p, "value": value}


def _oracle_bids(env):
    """DD points per trump suit on the deal as it really was, both teams."""
    dd = env.solve_all_suits()
    suits = [[int(ns), int(ew)] for ns, ew in dd["suits"]]
    return {
        "suits": suits,
        "best": [_best_contract(suits, 0), _best_contract(suits, 1)],
    }


def _playgen_analysts(env, model_path):
    """One playgen analyst per seat, or None if the model can't be loaded.

    Seat-bound like the IS-DD instances of the agent review: the bid head is
    read from the acting seat's own view, so asking a single instance would
    condition the policy on a hand that seat never saw.
    """
    if not model_path:
        return None
    try:
        analysts = []
        for seat in range(4):
            a = colver.Analyst(model_path)
            a.init_deal(env, seat)
            analysts.append(a)
        return analysts
    except Exception:
        return None


def _analyze_sync(game, bid_model_path=None, playgen_model_path=None,
                  match_scores=None):
    """Replay the stored actions, solving each play decision. CPU-bound.

    `match_scores` est le cumul `[NS, EW]` **avant** cette donne. Il ne change
    que l'avis de bid v6, dont l'observation est score-aware : la même main
    s'annonce autrement à 900-200 qu'à 0-0. Ni le solveur DD ni la tête
    d'enchère de playgen ne le lisent — le premier ne connaît que les cartes,
    la seconde n'a été entraînée que sur des donnes isolées.
    """
    env = colver.Env.deal_with_hands(game["dealer"], game["hands"])
    scores = [int(match_scores[0]), int(match_scores[1])] if match_scores else [0, 0]
    env.set_match_scores(scores[0], scores[1])

    # Oracle annonces: DD solve of the full deal, one solve per trump suit
    oracle_bids = None
    try:
        oracle_bids = _oracle_bids(env)
    except Exception:
        pass

    if bid_model_path:
        try:
            env.load_bid_model(bid_model_path)
        except Exception:
            bid_model_path = None

    analysts = _playgen_analysts(env, playgen_model_path)
    had_playgen = analysts is not None

    # ── La courbe de la donne ──
    #
    # Un seul axe, en points cartes du **preneur**, et trois tracés qui s'y
    # lisent ensemble : ce qu'il a déjà ramassé, ce qu'il fera en jeu parfait
    # depuis chaque position, et le seuil à atteindre.
    #
    # **Orientée preneur, jamais N-S.** Les points cartes sont à somme
    # constante (162), donc tracer les deux camps dessinerait deux courbes
    # miroir : la seconde n'ajoute pas un bit d'information.
    #
    # La projection est plate tant que personne ne se trompe — jouer le meilleur
    # coup ne change pas la valeur du nœud — et décroche exactement à une erreur,
    # de sa hauteur. Son passage sous le seuil **est** une « faute décisive » :
    # la courbe et le panneau des moments racontent donc la même chose par
    # construction.
    curve_bids = []   # montée du seuil pendant l'enchère
    curve_pts = []    # points ramassés par le preneur, après chaque pli
    curve_dd = []     # projection en jeu parfait, à chaque décision

    moves = []
    bids = []
    # La marche de l'escalier, renseignée à la première décision de jeu. Reste
    # `None` sur une donne passée : sans contrat il n'y a pas d'échelle.
    score_step = None
    # Le journal en entiers nus : c'est le préfixe que le moteur de variantes
    # rejoue pour se construire sa propre partie (il n'emprunte jamais `env` —
    # cf. l'en-tête de `variation`).
    journal = [int(e["action"]) for e in game["actions"]]
    for idx, entry in enumerate(game["actions"]):
        if env.is_terminal():
            break
        phase = int(env.phase())
        player = int(env.current_player())
        action = int(entry["action"])
        if phase == 0 and (bid_model_path or analysts):
            bid = {"idx": idx, "player": player, "action": action}
            if bid_model_path:
                try:
                    result = env.action_bid_nn()
                    q = {int(a): float(v) for a, v in result["q_values"]}
                    best = int(result["best_action"])
                    bid.update({
                        "model_best": best,
                        "q_best": round(q.get(best, 0.0), 3),
                        "q_played": round(q[action], 3) if action in q else None,
                    })
                except Exception:
                    pass
            if analysts:
                try:
                    # None on v1 weights (no bid head) — then playgen stays silent.
                    pol = analysts[player].bid_policy(env, 1.0)
                except Exception:
                    pol = None
                if pol:
                    best = max(range(len(pol)), key=lambda a: pol[a])
                    bid.update({
                        "playgen_best": best,
                        "playgen_p": round(float(pol[best]), 4),
                        "playgen_p_played": (round(float(pol[action]), 4)
                                             if action < len(pol) else None),
                    })
            if len(bid) > 3:
                bids.append(bid)
        if phase == 0 and action != 0 and action < 41:
            # Le seuil monte par paliers pendant l'enchère, puis se fige au
            # contrat. C'est la même horizontale que pendant le jeu, donc
            # l'enchère est la première moitié du même graphe — pas un second
            # tracé. Coinche et surcoinche (41, 42) multiplient l'enjeu sans
            # déplacer le seuil : elles sortent d'ici.
            value = 250 if 37 <= action <= 40 else 80 + (action - 1) // 4 * 10
            curve_bids.append([idx, value, player % 2])
        if analysts:
            if phase == 0:
                for a in analysts:
                    a.observe(env, action)
            else:
                analysts = None  # the auction is over; drop the four samplers
        if phase == 1:
            legals = list(env.legal_actions())
            if action not in legals:
                break  # corrupt record — stop rather than emit nonsense
            if score_step is None:
                # La marche du barème, prise à la **première** décision de jeu :
                # c'est une constante du contrat, sauf à la frontière du capot,
                # que `total_card_points` ne sait lever qu'avant le premier pli
                # ramassé (cf. `scoring::deal_score_step`).
                score_step = int(env.deal_score_step())
            if len(legals) == 1:
                moves.append({
                    "idx": idx, "player": player, "action": action,
                    "best": action, "cost": 0, "cost_score": 0, "forced": True,
                })
            else:
                result = env.solve_scores()
                # `card_pts`, pas `scores` : ce nom-là est déjà pris par le
                # score de **partie**, posé en tête de fonction et republié dans
                # le blob. Le masquer faisait sortir `"match_scores"` avec la
                # table DD de la dernière décision résolue — un champ qui ment,
                # invisible tant que personne ne le lit.
                card_pts = dict(result["scores"])
                # Le même solve, passé au barème. Deux échelles qui ne se
                # soustraient pas : points cartes 0-252 d'un côté, écart de
                # score marqué de l'autre.
                deal = {int(c): int(v) for c, v in result["deal_scores"]}
                made = {int(c): bool(v) for c, v in result["contract_made"]}
                team = player % 2

                best_ns = max(card_pts.values()) if team == 0 else min(card_pts.values())
                cost = ((best_ns - card_pts[action]) if team == 0
                        else (card_pts[action] - best_ns))

                # `best_ns` **est** la valeur DD du nœud : ce que N-S fera si les
                # quatre joueurs jouent parfaitement à partir d'ici. Elle était
                # jetée après le calcul du coût ; c'est la ligne de projection.
                taker = env.get_contract()["team"]
                total = 252 if best_ns in (0, 252) else 162
                curve_dd.append([idx, best_ns if taker == 0 else total - best_ns])

                best_deal = max(deal.values()) if team == 0 else min(deal.values())
                cost_score = ((best_deal - deal[action]) if team == 0
                              else (deal[action] - best_deal))

                # La classe des cartes optimales, pas son représentant. En score
                # de donne elle s'élargit — cinq cartes peuvent valoir
                # exactement pareil — et désigner l'une d'elles comme « la »
                # carte de l'Oracle promet une précision qui n'existe pas.
                best_class = sorted(c for c, v in deal.items() if v == best_deal)
                swing = made[action] != made[best_class[0]]

                move = {
                    "idx": idx, "player": player, "action": action,
                    "best": int(result["best_card"]),
                    "best_class": best_class,
                    # Combien de cartes il y avait à départager. Sans ce
                    # nombre, `len(best_class)` ne dit pas si l'Oracle a
                    # *choisi* ou s'il est **indifférent** — et `cost_score`
                    # vaut 0 dans les deux cas. Cf. `_categorize`.
                    "n_legal": len(legals),
                    "cost": int(cost),
                    "cost_score": int(cost_score),
                    "swing": swing,
                    "category": _categorize(cost_score, swing),
                }
                if cost_score > 0:
                    # Les deux variantes, seulement sur une erreur : ailleurs
                    # elles seraient identiques, le coup joué *étant* celui de
                    # l'Oracle. `best_card` est optimal en points cartes et la
                    # conversion au barème est monotone, donc il appartient
                    # aussi à `best_class` — la classe optimale en score de
                    # donne. Coût mesuré : ~43 ms par donne pour toutes les
                    # erreurs réunies, contre ~1 s pour l'analyse entière.
                    move["var"] = variation.error_lines(
                        game["dealer"], game["hands"], journal[:idx],
                        action, int(result["best_card"]))
                moves.append(move)
        env.step(action)
        if phase == 1:
            # Après le `step` : `get_points` ne compte un pli qu'une fois
            # résolu. Le dix de der (10, ou 100 sur capot) y est déjà, donc la
            # dernière marche est plus haute que les autres — c'est le jeu, pas
            # un artefact.
            taker = env.get_contract()["team"]
            curve_pts.append([idx, int(env.get_points()[taker])])

    curve = _curve(env, curve_bids, curve_pts, curve_dd)
    summary = _summarize(moves)
    return {
        "version": ANALYSIS_VERSION,
        "match_scores": scores,
        "playgen": had_playgen,
        "moves": moves,
        "bids": bids,
        "oracle_bids": oracle_bids,
        "curve": curve,
        # L'unité de `cost_score` et d'`isdd_cost` : `4V` sur un contrat normal,
        # `2(162 + V·mult)` sous coinche. Les deux régimes sont dans un rapport
        # de plus de deux, donc **aucun seuil absolu ne peut servir les deux** —
        # tout seuil sur ces échelles s'exprime en fraction de ce nombre.
        "score_step": score_step,
        "summary": summary,
    }


def _curve(env, bids, points, dd):
    """Les trois séries de la courbe, plus le seuil à atteindre.

    `threshold` est en points **cartes** : `scoring` ajoute la belote au total du
    preneur pour décider de la réussite, donc une belote de 20 **abaisse la barre
    de 20** au lieu d'ajouter 20 points au bout. Tracer l'horizontale à la valeur
    nue du contrat ferait mentir la courbe sur les donnes à belote.

    Rend `None` sur une donne passée : sans contrat il n'y a ni preneur, ni
    seuil, ni rien à projeter.
    """
    contract = env.get_contract()
    if not contract or not contract.get("value"):
        return None
    taker = int(contract["team"])
    # `get_contract()["value"]` est **déjà** en points (80-160, 250) — c'est
    # `Contract::point_value()` côté Rust, pas le `value` brut du contrat, qui
    # lui vaut 8-16 et 25. Le multiplier par 10 donnait un seuil à 1200.
    value = int(contract["value"])
    try:
        belote = [int(b) for b in env.belote_final()]
    except Exception:  # noqa: BLE001 — un binding manquant ne perd pas la courbe
        belote = [0, 0]
    return {
        "taker": taker,
        "trump": int(contract["trump"]),
        "value": value,
        "coinche": int(contract.get("coinche", 0)),
        "belote": belote,
        "threshold": value - belote[taker],
        "capot": value == 250,
        "bids": bids,
        "points": points,
        "dd": dd,
    }


def _summarize(moves):
    """Par siège : ce que ses décisions ont coûté, dans les deux échelles.

    `total_cost` reste en points cartes — la page Stats agrège des historiques
    dessus. `total_cost_score` est ce qui compte : ce que le siège a réellement
    laissé sur la table, contrat compris.
    """
    players = []
    for p in range(4):
        pm = [m for m in moves if m["player"] == p]
        decisions = [m for m in pm if not m.get("forced")]
        total_cost = sum(m["cost"] for m in decisions)
        total_cost_score = sum(m.get("cost_score", 0) for m in decisions)
        counts = dict.fromkeys(CATEGORIES, 0)
        for m in decisions:
            counts[m["category"]] = counts.get(m["category"], 0) + 1
        players.append({
            "player": p,
            "moves": len(pm),
            "forced": len(pm) - len(decisions),
            "decisions": len(decisions),
            "total_cost": total_cost,
            "avg_cost": round(total_cost / len(decisions), 1) if decisions else 0.0,
            "total_cost_score": total_cost_score,
            "counts": counts,
        })
    return {"players": players}


def _is_fresh(cached, playgen_model_path, match_scores=None):
    """A cached row is stale on a version bump, and also when it was computed
    without the playgen model while that model is now available — otherwise a
    single failed load would leave the game without a playgen annonce forever.

    v6 portait une exception : une ligne v5 restait valable pour une donne jouée
    à 0-0, les deux versions y calculant le même blob. Elle **a été retirée au
    bump v7**, comme son commentaire l'annonçait : v7 lit le coût des cartes en
    score de donne, que ni v5 ni v6 n'ont jamais écrit. Aucune ligne antérieure
    n'est récupérable, et tout le cache se recalcule — ~1 s par donne.
    """
    if cached is None:
        return False
    if cached.get("version") != ANALYSIS_VERSION:
        return False
    return bool(cached.get("playgen")) or not playgen_model_path


def _seat_at(game, action_idx):
    """(phase, seat) at `action_idx` of a stored game, replaying the journal."""
    env = colver.Env.deal_with_hands(game["dealer"], game["hands"])
    for entry in game["actions"][:action_idx]:
        if env.is_terminal():
            return None, None
        env.step(int(entry["action"]))
    if env.is_terminal():
        return None, None
    return int(env.phase()), int(env.current_player())


async def true_world(game_id, action_idx):
    """Ce que la donne réelle permettait, du point de vue du siège qui parle.

    Le pendant, pour une annonce, de `card_analysis.true_world` : une valeur
    **exacte** sur la distribution telle qu'elle était, à ne jamais fusionner
    avec les mondes échantillonnés de la page annonces. Ces mondes répondent à
    « cette annonce était-elle bonne ? » ; cette ligne-ci répond à « qu'est-ce
    que cette donne-là autorisait ? ». Les confondre transformerait la première
    question en « a-t-elle marché ? ».

    Les points sont rendus du côté de l'équipe du siège analysé — la page
    l'assied toujours en Sud, donc dans son repère c'est Nord-Sud.

    Le solve est déjà en cache dès que Rejouer a analysé la donne ; sinon il
    coûte quatre solves (~300 ms) et n'est pas mis en cache : une ligne
    `analysis` partielle serait relue comme une analyse complète.
    """
    game = await db.get_game(game_id)
    if game is None:
        return None, "Partie introuvable"
    if not 0 <= action_idx < len(game["actions"]):
        return None, "Index hors de la donne"

    phase, seat = _seat_at(game, action_idx)
    if phase != 0:
        return None, "Ce coup n'est pas une annonce"

    cached = await db.get_analysis(game_id)
    bids = cached.get("oracle_bids") if cached else None
    # Une version antérieure a pu être calculée sous d'autres règles de jeu :
    # un barème ou un coup légal qui change périme les valeurs DD. v5 fait
    # exception — le passage à v6 n'a touché que le score lu par le bidder, les
    # valeurs DD sont les mêmes.
    if not bids or cached.get("version") not in _DD_COMPATIBLE_VERSIONS:
        try:
            bids = await asyncio.to_thread(
                lambda: _oracle_bids(
                    colver.Env.deal_with_hands(game["dealer"], game["hands"])))
        except Exception as e:  # noqa: BLE001 — la page vit très bien sans
            return None, f"Solve impossible : {e}"

    team = seat % 2
    return {
        "seat": seat,
        "team": team,
        "pts": [int(s[team]) for s in bids["suits"]],
        "best": bids["best"][team],
        # La main du siège, pour que le client n'affiche cette ligne que tant
        # que la main à l'écran est bien celle de la donne (cf. les mains
        # enregistrées, dont la clé ne porte pas la donne d'origine).
        "hand": sorted(int(c) for c in game["hands"][seat]),
    }, None


async def match_scores_before(game):
    """Le cumul `[NS, EW]` d'avant cette donne, ou 0-0 hors partie.

    Ce que bid v6 voyait au moment d'annoncer. Hors partie il n'y a rien à lire
    et 0-0 est la vérité, pas un défaut de repli.
    """
    ctx = await db.deal_match_context(game)
    return list(ctx["before"]) if ctx else [0, 0]


async def get_or_compute(game_id, bid_model_path=None, playgen_model_path=None):
    """Return the cached analysis for a game, computing it on first request.

    La donne est relue **avant** de statuer sur le cache : la fraîcheur dépend
    désormais du score de partie d'avant la donne, qui n'est pas dans le blob
    en cache. Ça coûte un SELECT indexé de plus sur un cache chaud, et ça ferme
    au passage le cas d'une donne re-scorée après coup par
    `integrity.backfill_scores` — son analyse avait alors été calculée sur un
    cumul incomplet.
    """
    game = await db.get_game(game_id)
    if game is None:
        return None, "Partie introuvable"
    if not game["actions"]:
        return None, "Aucune action à analyser"
    scores = await match_scores_before(game)

    cached = await db.get_analysis(game_id)
    if _is_fresh(cached, playgen_model_path, scores):
        return cached, None

    lock = _locks.setdefault(game_id, asyncio.Lock())
    async with lock:
        # Another request may have computed it while we waited on the lock
        cached = await db.get_analysis(game_id)
        if _is_fresh(cached, playgen_model_path, scores):
            return cached, None
        analysis = await asyncio.to_thread(
            _analyze_sync, game, bid_model_path, playgen_model_path, scores)
        await db.save_analysis(game_id, json.dumps(analysis))
    _locks.pop(game_id, None)
    return analysis, None


# ===== Analyse en masse, à la demande du joueur =====
#
# Le pendant du « demander une analyse » de lichess : le joueur appuie sur un
# bouton, le serveur passe ses donnes au solveur, et la page suit l'avancement.
#
# **Pourquoi à la demande et pas en tâche de fond.** Un balayage automatique
# ferait le travail pour des donnes que personne ne regardera jamais, et il
# faudrait l'ordonnancer contre les parties en cours. À la demande, le coût est
# payé par qui en tire la valeur, et le joueur sait ce qu'il attend.
#
# **Pourquoi l'analyse complète et non un mode « DD seul ».** Les têtes
# d'enchère playgen sont l'essentiel du coût (~185 ms sur ~230) et les
# statistiques de jeu n'en ont pas besoin. Mais les sauter écrirait un blob que
# `_is_fresh` déclare périmé dès que le modèle est là — donc recalculé à la
# première ouverture de Rejouer, et le travail serait fait deux fois. On calcule
# donc exactement ce que Rejouer affiche : la page du joueur devient instantanée
# par la même occasion.

# Un seul balayage à la fois pour tout le serveur. Ce sont des recherches DD
# pleine donne, les plus chères du site : deux en parallèle prendraient les
# cœurs des gens qui jouent. La file est implicite — un second appel se voit
# refuser et réessaiera.
_bulk_gate = asyncio.Semaphore(1)

# État par joueur, en mémoire. Perdu au redémarrage, et c'est sans conséquence :
# le travail déjà fait est en base, donc relancer reprend où ça s'est arrêté.
_bulk_jobs = {}


def bulk_status(user_id):
    """Où en est le balayage de ce joueur, ou None s'il n'y en a jamais eu."""
    return _bulk_jobs.get(user_id)


async def bulk(user_id, game_ids, *, bid_model_path=None, playgen_model_path=None):
    """Analyser les donnes d'un joueur, une par une, en publiant l'avancement.

    Rend immédiatement si un balayage tourne déjà pour ce joueur. Chaque donne
    passe par `get_or_compute`, donc le verrou par donne et le cache sont ceux
    du reste du site : une donne analysée entre-temps par Rejouer n'est pas
    refaite.

    Une donne qui échoue est comptée et sautée. Le balayage ne doit pas mourir
    sur une donne : `errors` le dit à l'arrivée plutôt que de laisser le joueur
    devant un compteur bloqué.
    """
    running = _bulk_jobs.get(user_id)
    if running and running.get("running"):
        return running

    job = {"running": True, "done": 0, "total": len(game_ids), "errors": 0}
    _bulk_jobs[user_id] = job
    if not game_ids:
        job["running"] = False
        return job

    async def run():
        try:
            async with _bulk_gate:
                for game_id in game_ids:
                    try:
                        _, err = await get_or_compute(
                            game_id,
                            bid_model_path=bid_model_path,
                            playgen_model_path=playgen_model_path)
                        if err:
                            job["errors"] += 1
                    except Exception:  # noqa: BLE001 — une donne ne tue pas le lot
                        logger.exception("analyse en masse : donne %s", game_id)
                        job["errors"] += 1
                    job["done"] += 1
        finally:
            job["running"] = False

    job["task"] = asyncio.create_task(run())
    return job
