#!/usr/bin/env python3
"""Étude de style et de niveau, joueur par joueur, sur les parties stockées.

Lit une base `colver.db`, rejoue chaque donne complète et agrège, pour chaque
identité assise à la table (humain connecté, invité, ou bot), deux familles de
mesures :

**Style** — ce que le joueur *fait*, sans jugement : taux de passe, seuil
d'annonce (force de main au moment où il annonce), optimisme de l'annonce face
au contrat réalisable en double-mort, attaque atout, coupe, surcoupe, points
donnés au partenaire ou à l'adversaire…

**Niveau** — à quelle distance du meilleur coup il joue :

- **coût DD** (table `analysis`) : le solveur double-mort revoit chaque carte
  avec les quatre mains visibles et chiffre en points la perte du coup joué.
  C'est un plafond, pas un juge : un coup correct sous incertitude peut coûter
  cher en double-mort. D'où la mesure complémentaire —
- **accord avec Dédé / DouDou50 / l'Oracle** (table `agent_review`) : à chaque
  carte non forcée, ce que les bots de référence auraient joué de ce siège-là.
  Dédé est instancié par siège : il ne voit que ce que le siège voyait.
- **accord avec Bid v6** sur les enchères, et perte de Q moyenne.

Les deux tables sont celles que la page « Rejouer » remplit à la demande ; ce
script les calcule pour les donnes qui ne les ont pas encore (`--compute`,
`--compute-review`) plutôt que d'ignorer ces donnes en silence.

Usage :
    uv run python scripts/analysis/player_study.py                    # base par défaut
    uv run python scripts/analysis/player_study.py --db /chemin/colver.db
    uv run python scripts/analysis/player_study.py --compute          # remplit `analysis`
    uv run python scripts/analysis/player_study.py --compute-review   # + `agent_review`
    uv run python scripts/analysis/player_study.py --json etude.json --md etude.md
"""

import argparse
import json
import math
import os
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

DEFAULT_DB = Path.home() / ".local" / "share" / "colver" / "colver.db"

# ---------------------------------------------------------------------------
# Cartes — cf. CLAUDE.md § Card Representation
# ---------------------------------------------------------------------------

SUITS = ["♠", "♥", "♦", "♣"]
RANKS = ["7", "8", "9", "V", "D", "R", "10", "A"]

# rang -> force à l'atout : V(3) > 9(2) > A(7) > 10(6) > R(5) > D(4) > 8(1) > 7(0)
TRUMP_ORDER = [0, 1, 6, 7, 2, 3, 4, 5]
PLAIN_PTS = [0, 0, 0, 2, 3, 4, 10, 11]
TRUMP_PTS = [0, 0, 14, 20, 3, 4, 10, 11]

BID_VALUES = list(range(80, 170, 10))  # 80..160


def suit_of(c):
    return c >> 3


def rank_of(c):
    return c & 7


def card_pts(c, trump):
    return TRUMP_PTS[rank_of(c)] if suit_of(c) == trump else PLAIN_PTS[rank_of(c)]


def card_name(c):
    return f"{RANKS[rank_of(c)]}{SUITS[suit_of(c)]}"


def trick_winner(entries, trump):
    """Siège qui tient le pli, sur les cartes déjà posées `[(siège, carte)]`."""
    lead_suit = suit_of(entries[0][1])
    best_seat, best_key = None, None
    for seat, c in entries:
        s = suit_of(c)
        if s == trump:
            key = (2, TRUMP_ORDER[rank_of(c)])
        elif s == lead_suit:
            key = (1, rank_of(c))
        else:
            key = (0, -1)
        if best_key is None or key > best_key:
            best_key, best_seat = key, seat
    return best_seat


def decode_bid(a):
    """(genre, valeur, couleur) — genre ∈ pass|bid|capot|coinche|surcoinche."""
    if a == 0:
        return "pass", None, None
    if a <= 36:
        i = a - 1
        return "bid", BID_VALUES[i // 4], i % 4
    if a <= 40:
        return "capot", 250, a - 37
    if a == 41:
        return "coinche", None, None
    return "surcoinche", None, None


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


def open_db(path):
    if not os.path.exists(path):
        sys.exit(f"Base introuvable : {path}")
    # `timeout` : plusieurs shards écrivent dans la même base pendant un
    # `--compute-review` parallèle. Sans ça le premier verrou perdu tue un
    # shard et son travail en cours.
    conn = sqlite3.connect(path, timeout=60.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 60000")
    return conn


def load_games(conn, mode_filter=None):
    """Donnes complètes, actions et mains déjà décodées."""
    rows = conn.execute(
        "SELECT * FROM games WHERE is_complete = 1 AND mode IN ('play','multi')"
        " ORDER BY created_at"
    ).fetchall()
    games = []
    for row in rows:
        g = dict(row)
        try:
            g["hands"] = json.loads(g["hands"])
            g["agents"] = json.loads(g["agents"] or "{}")
            g["actions"] = json.loads(g["actions"] or "[]")
            g["contract"] = json.loads(g["contract"]) if g["contract"] else None
        except (TypeError, ValueError):
            continue
        if not g["actions"] or len(g["hands"]) != 4:
            continue
        if mode_filter and g["mode"] != mode_filter:
            continue
        games.append(g)
    return games


def seat_identities(conn, game):
    """Qui occupait chaque siège : `[(clé, libellé, humain?)]` dans l'ordre N,E,S,O.

    Même logique que `database.game_seat_names` : `games.agents` seul ne suffit
    pas (le siège humain y vaut « human » en solo, et en salon un pseudo ne se
    distingue pas d'une clé de bot).
    """
    humans = {}
    for seat, username in conn.execute(
        "SELECT gp.seat, u.username FROM game_players gp"
        " LEFT JOIN users u ON u.id = gp.user_id WHERE gp.game_id = ?",
        (game["id"],),
    ):
        humans[seat] = username
    if game["mode"] == "play" and game["human_seat"] is not None:
        name = None
        if game["user_id"] is not None:
            r = conn.execute(
                "SELECT username FROM users WHERE id = ?", (game["user_id"],)
            ).fetchone()
            name = r[0] if r else None
        humans[game["human_seat"]] = name

    out = []
    for s in range(4):
        if s in humans:
            if humans[s]:
                out.append((f"@{humans[s]}", humans[s], True))
            else:
                # Partie jouée sans compte : un humain, mais anonyme. Toutes les
                # parties invité tombent dans le même seau — on ne peut pas les
                # séparer, et le dire vaut mieux que de les faire disparaître.
                out.append(("@?invité", "Invité (anonyme)", True))
        else:
            kind = game["agents"].get(str(s)) or "?"
            out.append((f"bot:{kind}", kind, False))
    return out


# ---------------------------------------------------------------------------
# Accumulateur
# ---------------------------------------------------------------------------


class Stats:
    """Compteurs plats + séries, agrégés par identité."""

    def __init__(self, key, label, human):
        self.key = key
        self.label = label
        self.human = human
        self.c = defaultdict(float)
        self.series = defaultdict(list)
        self.games = set()
        self.suits_bid = defaultdict(int)
        self.partners = defaultdict(int)

    def add(self, name, v=1):
        self.c[name] += v

    def push(self, name, v):
        self.series[name].append(v)

    def rate(self, num, den):
        d = self.c[den]
        return self.c[num] / d if d else None

    def mean(self, name):
        s = self.series[name]
        return sum(s) / len(s) if s else None

    def sem(self, name):
        """Erreur-type de la moyenne — ce qui dit si un écart tient debout."""
        s = self.series[name]
        n = len(s)
        if n < 2:
            return None
        m = sum(s) / n
        var = sum((x - m) ** 2 for x in s) / (n - 1)
        return math.sqrt(var / n)


# ---------------------------------------------------------------------------
# Rejeu d'une donne : extraction des traits de style
# ---------------------------------------------------------------------------


def replay_features(colver, game):
    """Rejoue la donne et rend une liste d'événements par siège.

    Rend `(events, meta)` : `events[seat]` est un dict de compteurs, `meta`
    porte le contrat, le preneur et l'issue.
    """
    env = colver.Env.deal_with_hands(game["dealer"], game["hands"])
    ev = [defaultdict(float) for _ in range(4)]
    hand_scores = [None] * 4  # force de main (heuristique) par siège
    for s in range(4):
        try:
            hand_scores[s] = env.evaluate_hand(s)
        except Exception:  # noqa: BLE001
            hand_scores[s] = None

    declarer = None
    contract_value = 0
    contract_suit = None
    coincher = None
    first_bid_done = [False] * 4
    truncated = False

    for entry in game["actions"]:
        if env.is_terminal():
            break
        action = int(entry["action"])
        phase = int(env.phase())
        seat = int(env.current_player())
        legals = list(env.legal_actions())
        if action not in legals:
            # Enregistrement corrompu (carte rejouée, pli à cinq cartes…). On
            # s'arrête là, comme le fait `analysis._analyze_sync`, mais on le
            # *dit* : les traits de style relevés jusqu'ici restent valables,
            # l'issue de la donne non.
            truncated = True
            break

        if phase == 0:
            kind, value, suit = decode_bid(action)
            if kind in ("bid", "capot"):
                declarer = seat
                contract_value = value
                contract_suit = suit
            elif kind == "coinche":
                coincher = seat
            # Une prise de parole ne compte que s'il y avait un choix : passer
            # quand passer est le seul coup légal n'est pas une décision.
            if len(legals) > 1:
                e = ev[seat]
                e["bid_turns"] += 1
                hs = hand_scores[seat]
                if kind == "pass":
                    e["bid_pass"] += 1
                    if hs:
                        e["_pass_strength_sum"] += hs["best_score"]
                        e["_pass_strength_n"] += 1
                else:
                    e["bid_spoke"] += 1
                    if hs:
                        e["_bid_strength_sum"] += hs["best_score"]
                        e["_bid_strength_n"] += 1
                if kind in ("bid", "capot"):
                    e["bid_made"] += 1
                    e["_bid_value_sum"] += value
                    e[f"bid_suit_{suit}"] += 1
                    if not first_bid_done[seat]:
                        first_bid_done[seat] = True
                        e["_open_value_sum"] += value
                        e["_open_n"] += 1
                        # Annonce-t-il dans sa meilleure couleur ?
                        if hs and suit == hs["best_suit"]:
                            e["bid_open_best_suit"] += 1
                if kind == "capot":
                    e["bid_capot"] += 1
                if kind == "coinche":
                    e["bid_coinche"] += 1
                if kind == "surcoinche":
                    e["bid_surcoinche"] += 1

        else:
            _play_features(env, ev, seat, action, legals, game)

        env.step(action)

    contract = env.get_contract() or {}
    trump = contract.get("trump")
    team = contract.get("team")
    terminal = bool(env.is_terminal())
    rewards = list(env.rewards()) if terminal else None
    made = None
    if team is not None and terminal:
        made = rewards[team] > 0

    meta = {
        "terminal": terminal,
        "truncated": truncated,
        "declarer": declarer,
        "contract_value": contract_value,
        "contract_suit": contract_suit,
        "trump": trump,
        "team": team,
        "coinche": contract.get("coinche", 0),
        "coincher": coincher,
        "made": made,
        "rewards": rewards,
        "hand_scores": hand_scores,
    }
    return ev, meta


def _play_features(env, ev, seat, action, legals, game):
    """Traits de style d'une carte jouée."""
    contract = env.get_contract() or {}
    trump = contract.get("trump")
    if trump is None:
        return
    declarer_team = contract.get("team")
    attacking = declarer_team is not None and seat % 2 == declarer_team

    lead = int(env.get_trick_lead())
    trick = env.get_current_trick()
    order = [(lead + i) % 4 for i in range(4)]
    entries = [(s, trick[s]) for s in order if trick[s] >= 0]
    pos = len(entries)
    hand = env.get_hands()[seat]

    e = ev[seat]
    e["cards"] += 1
    if len(legals) > 1:
        e["free_cards"] += 1

    role = "att" if attacking else "def"
    played_suit = suit_of(action)

    if pos == 0:
        # --- entame / ouverture de pli ---
        e["leads"] += 1
        e[f"leads_{role}"] += 1
        if played_suit == trump:
            e["lead_trump"] += 1
            e[f"lead_trump_{role}"] += 1
        if rank_of(action) == 7 and played_suit != trump:
            e["lead_ace"] += 1
        if len(env.get_played_cards()) == 0:
            e["opening_leads"] += 1
            if played_suit == trump:
                e["opening_lead_trump"] += 1
        return

    lead_suit = suit_of(entries[0][1])
    holder = trick_winner(entries, trump)
    partner_holds = holder % 2 == seat % 2
    could_follow = any(suit_of(c) == lead_suit for c in legals)
    pts = card_pts(action, trump)

    # Points versés sur le pli : au partenaire (« mettre du point ») ou à
    # l'adversaire (« cadeau »). Ne comptent que si un coup moins cher existait.
    cheapest = min(card_pts(c, trump) for c in legals)
    last_to_play = pos == 3
    if partner_holds:
        e["partner_holds_turns"] += 1
        e["_partner_pts_sum"] += pts
        if pts >= 10 and cheapest < pts:
            e["partner_fed"] += 1
    else:
        e["opp_holds_turns"] += 1
        e["_opp_pts_sum"] += pts
        if pts >= 10 and cheapest < pts:
            e["opp_gifted"] += 1
    if last_to_play and partner_holds:
        e["last_on_partner"] += 1
        e["_last_on_partner_pts"] += pts

    if could_follow:
        return

    # --- coupe : plus de carte dans la couleur demandée ---
    if lead_suit == trump:
        return
    has_trump = any(suit_of(c) == trump for c in hand)
    if not has_trump:
        return

    opp_trumps = [
        TRUMP_ORDER[rank_of(c)]
        for s, c in entries
        if suit_of(c) == trump and s % 2 != seat % 2
    ]
    best_opp = max(opp_trumps) if opp_trumps else None
    ruffed = played_suit == trump

    e["ruff_chances"] += 1
    if ruffed:
        e["ruffs"] += 1
    if partner_holds:
        # Couper le pli que tient déjà son partenaire : dépense d'atout.
        e["ruff_chances_partner"] += 1
        if ruffed:
            e["ruffs_on_partner"] += 1
    else:
        e["ruff_chances_opp"] += 1
        if ruffed:
            e["ruffs_on_opp"] += 1

    if best_opp is None:
        return

    # L'adversaire a coupé. Peut-on surcouper ?
    can_over = any(
        suit_of(c) == trump and TRUMP_ORDER[rank_of(c)] > best_opp for c in legals
    )
    if can_over:
        e["overruff_chances"] += 1
        if ruffed and TRUMP_ORDER[rank_of(action)] > best_opp:
            e["overruffs"] += 1
    else:
        # « Ne pisse pas » : sous-couper est légal mais presque toujours perdant.
        has_discard = any(suit_of(c) != trump for c in legals)
        if has_discard:
            e["undertrump_chances"] += 1
            if ruffed:
                e["undertrumped"] += 1


# ---------------------------------------------------------------------------
# Agrégation
# ---------------------------------------------------------------------------


def merge(stats, ev, meta, seat, game, analysis, review):
    st = stats
    st.games.add(game["id"])
    st.add("deals")
    st.add(f"deals_{'multi' if game['mode'] == 'multi' else 'solo'}")

    # ---- enchères ----
    for k, v in ev.items():
        if k.startswith("_") or k.startswith("bid_suit_"):
            st.add(k, v)
    for s in range(4):
        st.suits_bid[s] += ev.get(f"bid_suit_{s}", 0)
    for k in (
        "bid_turns", "bid_pass", "bid_spoke", "bid_made", "bid_capot",
        "bid_coinche", "bid_surcoinche", "bid_open_best_suit",
    ):
        st.add(k, ev.get(k, 0))

    if meta["declarer"] == seat:
        st.add("declared")
        st.push("declared_value", meta["contract_value"])
        if meta["made"] is not None:
            st.add("declared_made" if meta["made"] else "declared_down")
        # Optimisme : ce qu'il a annoncé face au meilleur contrat que son camp
        # pouvait réellement tenir en double-mort.
        ob = (analysis or {}).get("oracle_bids")
        if ob and meta["team"] is not None:
            dd_value = ob["best"][meta["team"]]["value"]
            st.push("overbid", meta["contract_value"] - dd_value)
            st.push("dd_makeable", dd_value)
    if meta["coincher"] == seat and meta["made"] is not None:
        st.add("coinches")
        # Coincher, c'est parier la chute : le coup est gagné si le contrat tombe.
        if not meta["made"]:
            st.add("coinches_right")

    if meta["team"] is not None and meta["made"] is not None:
        if seat % 2 == meta["team"]:
            st.add("as_attacker")
            if meta["made"]:
                st.add("as_attacker_won")
        else:
            st.add("as_defender")
            if not meta["made"]:
                st.add("as_defender_won")
    # Une donne tronquée n'a pas d'issue : la compter 0 point ferait mentir la
    # moyenne marquée.
    if meta["rewards"] is not None:
        st.push("marked", meta["rewards"][seat % 2])
    else:
        st.add("deals_no_outcome")

    # ---- jeu : style ----
    for k in (
        "cards", "free_cards", "leads", "leads_att", "leads_def", "lead_trump",
        "lead_trump_att", "lead_trump_def", "lead_ace", "opening_leads",
        "opening_lead_trump", "partner_holds_turns", "partner_fed",
        "opp_holds_turns", "opp_gifted", "last_on_partner",
        "ruff_chances", "ruffs", "ruff_chances_partner", "ruffs_on_partner",
        "ruff_chances_opp", "ruffs_on_opp", "overruff_chances", "overruffs",
        "undertrump_chances", "undertrumped",
    ):
        st.add(k, ev.get(k, 0))

    # ---- jeu : niveau (coût DD) ----
    if analysis:
        summ = (analysis.get("summary") or {}).get("players")
        if summ and seat < len(summ):
            p = summ[seat]
            st.add("dd_decisions", p["decisions"])
            st.add("dd_forced", p["forced"])
            st.add("dd_cost", p["total_cost"])
            for label, n in p["counts"].items():
                st.add(f"dd_{label}", n)
        # Le coût DD n'est pas comparable d'un rôle à l'autre : le preneur et
        # la défense n'ont ni les mêmes décisions ni les mêmes marges. On garde
        # donc les deux séries à part, en plus du total.
        role = None
        if meta["team"] is not None:
            role = "att" if seat % 2 == meta["team"] else "def"
        for m in analysis.get("moves") or []:
            if m["player"] == seat and not m.get("forced"):
                st.push("dd_cost_per_move", m["cost"])
                if role:
                    st.push(f"dd_cost_{role}", m["cost"])
        for b in analysis.get("bids") or []:
            if b["player"] != seat:
                continue
            if "model_best" in b:
                st.add("bid_judged")
                if b["model_best"] == b["action"]:
                    st.add("bid_agree_v6")
                if b.get("q_played") is not None:
                    st.push("bid_qloss", b["q_best"] - b["q_played"])
            if b.get("playgen_best") is not None:
                st.add("bid_judged_pg")
                if b["playgen_best"] == b["action"]:
                    st.add("bid_agree_playgen")

    # ---- jeu : niveau (accord avec les bots de référence) ----
    if review:
        for m in review.get("moves") or []:
            if m["player"] != seat or m.get("forced"):
                continue
            role = None
            if meta["team"] is not None:
                role = "att" if seat % 2 == meta["team"] else "def"
            for bot in ("isdd", "doudou", "oracle"):
                if m.get(bot) is None:
                    continue
                st.add(f"agree_{bot}_n")
                if m[bot] == m["action"]:
                    st.add(f"agree_{bot}")
                if bot == "isdd" and role:
                    st.add(f"agree_isdd_{role}_n")
                    if m[bot] == m["action"]:
                        st.add(f"agree_isdd_{role}")


# ---------------------------------------------------------------------------
# Calcul des tables manquantes
# ---------------------------------------------------------------------------


def load_cached(conn, table, version_key, version):
    out = {}
    try:
        rows = conn.execute(f"SELECT game_id, data FROM {table}").fetchall()
    except sqlite3.OperationalError:
        return out
    for gid, data in rows:
        try:
            blob = json.loads(data)
        except ValueError:
            continue
        if blob.get(version_key) == version:
            out[gid] = blob
    return out


def compute_missing_analysis(conn, games, cached, verbose=True):
    """Remplit `analysis` pour les donnes qui ne l'ont pas. Quelques s/donne."""
    import colver
    from colver.web import analysis as wa

    todo = [g for g in games if g["id"] not in cached]
    if not todo:
        return 0
    bid_model = _path(colver.bid_model_path)
    playgen_model = _path(colver.playgen_model_path)
    if verbose:
        print(f"[analysis] {len(todo)} donne(s) à résoudre (solveur DD)…",
              file=sys.stderr)
    done = 0
    for i, g in enumerate(todo, 1):
        t0 = time.time()
        try:
            blob = wa._analyze_sync(g, bid_model, playgen_model)
        except Exception as exc:  # noqa: BLE001
            print(f"[analysis] {g['id']} : {exc}", file=sys.stderr)
            continue
        conn.execute(
            "INSERT OR REPLACE INTO analysis (game_id, created_at, data)"
            " VALUES (?, datetime('now'), ?)",
            (g["id"], json.dumps(blob)),
        )
        conn.commit()
        cached[g["id"]] = blob
        done += 1
        if verbose:
            print(f"  [{i}/{len(todo)}] {g['id']} {time.time() - t0:.1f}s",
                  file=sys.stderr)
    return done


def compute_missing_review(conn, games, cached, verbose=True, shard=None, of=None):
    """Remplit `agent_review`. Lent, et il faut le sidecar playgen.

    Le coût par donne est surtout de la *latence* — quatre IS-DD par siège, un
    aller-retour au sidecar à chaque observation — pas du calcul. Plusieurs
    shards en parallèle tiennent donc l'échelle sans s'affamer mutuellement :
    `--shard i --of n` prend une donne sur n. Chaque donne est écrite dès
    qu'elle est finie, donc un shard relancé reprend là où il en était.
    """
    import colver
    from colver.web import agent_review as ar

    todo = [g for g in games if g["id"] not in cached]
    if of:
        # Le partage porte sur l'identifiant, pas sur la position dans la liste.
        # Chaque shard construit son `todo` à son propre démarrage, donc les
        # listes n'ont ni la même longueur ni le même décalage : un `i % of`
        # ferait tomber la même donne dans plusieurs shards *et* en laisserait
        # d'autres sans preneur. La clé doit être stable d'un processus à
        # l'autre — d'où md5 plutôt que `hash()`, qui est salé par processus.
        import hashlib
        todo = [g for g in todo
                if int(hashlib.md5(g["id"].encode()).hexdigest(), 16) % of
                == (shard or 0)]
    if not todo:
        return 0
    play_model = _path(colver.model_path)
    belief_model = _path(colver.belief_model_path)
    if not os.environ.get("COLVER_PLAYGEN_GPU_URL"):
        print("[review] COLVER_PLAYGEN_GPU_URL absent : Dédé échantillonnera "
              "des mondes uniformes, plus faibles que la production.",
              file=sys.stderr)
    if verbose:
        print(f"[review] {len(todo)} donne(s) à passer aux bots de référence…",
              file=sys.stderr)
    done = 0
    for i, g in enumerate(todo, 1):
        t0 = time.time()
        try:
            runner = ar._Runner(g, play_model=play_model,
                                belief_model=belief_model)
            runner.start()
            while runner.step() is not None:
                pass
            blob = runner.finish()
        except Exception as exc:  # noqa: BLE001
            print(f"[review] {g['id']} : {exc}", file=sys.stderr)
            continue
        conn.execute(
            "INSERT OR REPLACE INTO agent_review (game_id, created_at, data)"
            " VALUES (?, datetime('now'), ?)",
            (g["id"], json.dumps(blob)),
        )
        conn.commit()
        cached[g["id"]] = blob
        done += 1
        if verbose:
            print(f"  [{i}/{len(todo)}] {g['id']} {time.time() - t0:.1f}s",
                  file=sys.stderr)
    return done


def _path(fn):
    try:
        p = fn()
    except Exception:  # noqa: BLE001
        return None
    return str(p) if p else None


# ---------------------------------------------------------------------------
# Rapport
# ---------------------------------------------------------------------------


def pct(x, digits=1):
    return "—" if x is None else f"{100 * x:.{digits}f} %"


def num(x, digits=1):
    return "—" if x is None else f"{x:.{digits}f}"


def signed(x, digits=1):
    return "—" if x is None else f"{x:+.{digits}f}"


def profile(st):
    """Toutes les mesures d'une identité, sous forme sérialisable."""
    c = st.c
    strength_bid = (c["_bid_strength_sum"] / c["_bid_strength_n"]
                    if c["_bid_strength_n"] else None)
    strength_pass = (c["_pass_strength_sum"] / c["_pass_strength_n"]
                     if c["_pass_strength_n"] else None)
    return {
        "key": st.key,
        "label": st.label,
        "human": st.human,
        "deals": int(c["deals"]),
        "deals_solo": int(c["deals_solo"]),
        "deals_multi": int(c["deals_multi"]),
        # --- enchères ---
        "bid_turns": int(c["bid_turns"]),
        "pass_rate": st.rate("bid_pass", "bid_turns"),
        "bid_rate": st.rate("bid_made", "bid_turns"),
        "coinche_rate": st.rate("bid_coinche", "bid_turns"),
        "coinches": int(c["coinches"]),
        "coinche_hit": st.rate("coinches_right", "coinches"),
        "capots": int(c["bid_capot"]),
        "mean_bid_value": (c["_bid_value_sum"] / c["bid_made"]
                           if c["bid_made"] else None),
        "mean_open_value": (c["_open_value_sum"] / c["_open_n"]
                            if c["_open_n"] else None),
        "open_best_suit_rate": (c["bid_open_best_suit"] / c["_open_n"]
                                if c["_open_n"] else None),
        "strength_when_bidding": strength_bid,
        "strength_when_passing": strength_pass,
        "strength_gap": (strength_bid - strength_pass
                         if strength_bid is not None
                         and strength_pass is not None else None),
        "suits_bid": {SUITS[s]: int(n) for s, n in st.suits_bid.items() if n},
        "declared": int(c["declared"]),
        "declared_rate": (c["declared"] / c["deals"] if c["deals"] else None),
        "mean_declared_value": st.mean("declared_value"),
        "contract_success": st.rate("declared_made", "declared"),
        "overbid": st.mean("overbid"),
        "overbid_sem": st.sem("overbid"),
        "dd_makeable": st.mean("dd_makeable"),
        "bid_agree_v6": st.rate("bid_agree_v6", "bid_judged"),
        "bid_judged": int(c["bid_judged"]),
        "bid_qloss": st.mean("bid_qloss"),
        "bid_agree_playgen": st.rate("bid_agree_playgen", "bid_judged_pg"),
        # --- résultats ---
        "as_attacker": int(c["as_attacker"]),
        "attack_win": st.rate("as_attacker_won", "as_attacker"),
        "as_defender": int(c["as_defender"]),
        "defense_win": st.rate("as_defender_won", "as_defender"),
        "mean_marked": st.mean("marked"),
        # --- style de jeu ---
        "cards": int(c["cards"]),
        "free_cards": int(c["free_cards"]),
        "lead_trump_rate": st.rate("lead_trump", "leads"),
        "lead_trump_att": st.rate("lead_trump_att", "leads_att"),
        "lead_trump_def": st.rate("lead_trump_def", "leads_def"),
        "lead_ace_rate": st.rate("lead_ace", "leads"),
        "opening_lead_trump": st.rate("opening_lead_trump", "opening_leads"),
        "ruff_rate": st.rate("ruffs", "ruff_chances"),
        "ruff_on_partner": st.rate("ruffs_on_partner", "ruff_chances_partner"),
        "ruff_on_opp": st.rate("ruffs_on_opp", "ruff_chances_opp"),
        "ruff_chances": int(c["ruff_chances"]),
        "overruff_rate": st.rate("overruffs", "overruff_chances"),
        "overruff_chances": int(c["overruff_chances"]),
        "undertrump_rate": st.rate("undertrumped", "undertrump_chances"),
        "undertrump_chances": int(c["undertrump_chances"]),
        "partner_fed_rate": st.rate("partner_fed", "partner_holds_turns"),
        "partner_pts": (c["_partner_pts_sum"] / c["partner_holds_turns"]
                        if c["partner_holds_turns"] else None),
        "opp_gifted_rate": st.rate("opp_gifted", "opp_holds_turns"),
        "opp_pts": (c["_opp_pts_sum"] / c["opp_holds_turns"]
                    if c["opp_holds_turns"] else None),
        # --- niveau ---
        "dd_decisions": int(c["dd_decisions"]),
        "dd_avg_cost": (c["dd_cost"] / c["dd_decisions"]
                        if c["dd_decisions"] else None),
        "dd_cost_sem": st.sem("dd_cost_per_move"),
        "dd_perfect": st.rate("dd_parfait", "dd_decisions"),
        "dd_blunder": ((c["dd_erreur"] + c["dd_faute"]) / c["dd_decisions"]
                       if c["dd_decisions"] else None),
        "dd_counts": {k: int(c[f"dd_{k}"]) for k in
                      ("parfait", "bon", "imprecision", "erreur", "faute")},
        "dd_cost_att": st.mean("dd_cost_att"),
        "dd_cost_att_n": len(st.series["dd_cost_att"]),
        "dd_cost_att_sem": st.sem("dd_cost_att"),
        "dd_cost_def": st.mean("dd_cost_def"),
        "dd_cost_def_n": len(st.series["dd_cost_def"]),
        "dd_cost_def_sem": st.sem("dd_cost_def"),
        "agree_isdd": st.rate("agree_isdd", "agree_isdd_n"),
        "agree_isdd_n": int(c["agree_isdd_n"]),
        "agree_isdd_att": st.rate("agree_isdd_att", "agree_isdd_att_n"),
        "agree_isdd_att_n": int(c["agree_isdd_att_n"]),
        "agree_isdd_def": st.rate("agree_isdd_def", "agree_isdd_def_n"),
        "agree_isdd_def_n": int(c["agree_isdd_def_n"]),
        "agree_doudou": st.rate("agree_doudou", "agree_doudou_n"),
        "agree_doudou_n": int(c["agree_doudou_n"]),
        "agree_oracle": st.rate("agree_oracle", "agree_oracle_n"),
        "agree_oracle_n": int(c["agree_oracle_n"]),
    }


def table(rows, cols):
    """Tableau Markdown, colonnes `[(titre, clé|callable, align)]`."""
    head = "| " + " | ".join(t for t, _, _ in cols) + " |"
    sep = "| " + " | ".join("---:" if a == "r" else ":---"
                            for _, _, a in cols) + " |"
    body = []
    for r in rows:
        cells = []
        for _, f, _ in cols:
            cells.append(str(f(r) if callable(f) else r.get(f, "—")))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *body])


def render(profiles, corpus):
    p = profiles
    humans = [x for x in p if x["human"]]
    bots = [x for x in p if not x["human"]]
    out = []
    w = out.append

    w("# Étude des joueurs\n")
    w(f"- **Donnes complètes analysées** : {corpus['deals']}")
    w(f"- **Période** : {corpus['from']} → {corpus['to']}")
    w(f"- **Donnes avec coût DD** (`analysis`) : {corpus['with_analysis']}")
    w(f"- **Donnes avec revue des bots** (`agent_review`) : "
      f"{corpus['with_review']}")
    w(f"- **Identités humaines** : {len(humans)} — **bots** : {len(bots)}\n")

    if corpus["truncated"]:
        w(f"> ⚠️ **{corpus['truncated']} donne(s) à l'enregistrement "
          "incohérent** (carte rejouée, pli à cinq cartes) : le rejeu s'arrête "
          "à la première action illégale. Les traits de style relevés avant ce "
          "point comptent, l'issue de la donne non. Identifiants : "
          + ", ".join("`" + i + "`" for i in corpus["truncated_ids"]) + "\n")

    if not humans:
        w("> Aucun siège humain identifié dans ce corpus.\n")

    # ------------------------------------------------------------------
    w("## 1. Qui joue le mieux ?\n")
    w("Deux mesures indépendantes, et il faut les lire ensemble.\n")
    w("- **Coût DD** = points perdus par décision, le solveur double-mort "
      "voyant les quatre mains. C'est un *plafond*, pas un arbitre : un coup "
      "juste sous incertitude peut coûter cher en double-mort, et un défenseur "
      "en a plus souvent l'occasion qu'un preneur. Plus bas = mieux.")
    w("- **Accord Dédé** = part des cartes non forcées où le joueur a choisi "
      "ce que Dédé (IS-DD, l'agent de production) aurait joué **de ce "
      "siège-là**, avec la seule information dont ce siège disposait. "
      "C'est la mesure « à information égale ».")
    w("- `±` est l'erreur-type de la moyenne : deux joueurs dont les "
      "intervalles se recouvrent ne sont pas départagés.\n")

    rank = sorted(
        [x for x in p if x["dd_decisions"] >= 1],
        key=lambda x: (x["dd_avg_cost"] if x["dd_avg_cost"] is not None else 1e9),
    )
    w(table(rank, [
        ("Joueur", lambda r: ("**" + r["label"] + "**") if r["human"]
         else r["label"] + " *(bot)*", "l"),
        ("Donnes", lambda r: r["deals"], "r"),
        ("Décisions", lambda r: r["dd_decisions"], "r"),
        ("Coût DD / coup", lambda r: num(r["dd_avg_cost"], 2), "r"),
        ("±", lambda r: num(r["dd_cost_sem"], 2), "r"),
        ("Coups parfaits", lambda r: pct(r["dd_perfect"], 0), "r"),
        ("Erreurs+fautes", lambda r: pct(r["dd_blunder"], 0), "r"),
        ("Accord Dédé", lambda r: pct(r["agree_isdd"], 0), "r"),
        ("(n)", lambda r: r["agree_isdd_n"], "r"),
        ("Accord DouDou", lambda r: pct(r["agree_doudou"], 0), "r"),
        ("Accord Oracle", lambda r: pct(r["agree_oracle"], 0), "r"),
    ]))
    w("")
    w("### Le même classement, séparé par rôle\n")
    w("Preneur et défenseur ne jouent pas les mêmes coups : la défense a plus "
      "souvent l'occasion de perdre des points, et un total agrégé départage "
      "donc en partie des rôles plutôt que des joueurs. À ne comparer qu'en "
      "colonne.\n")
    w(table(rank, [
        ("Joueur", lambda r: ("**" + r["label"] + "**") if r["human"]
         else r["label"] + " *(bot)*", "l"),
        ("Coût DD preneur", lambda r: num(r["dd_cost_att"], 2), "r"),
        ("±", lambda r: num(r["dd_cost_att_sem"], 2), "r"),
        ("(n)", lambda r: r["dd_cost_att_n"], "r"),
        ("Coût DD défense", lambda r: num(r["dd_cost_def"], 2), "r"),
        ("±", lambda r: num(r["dd_cost_def_sem"], 2), "r"),
        ("(n)", lambda r: r["dd_cost_def_n"], "r"),
        ("Accord Dédé preneur", lambda r: pct(r["agree_isdd_att"], 0), "r"),
        ("(n)", lambda r: r["agree_isdd_att_n"], "r"),
        ("Accord Dédé défense", lambda r: pct(r["agree_isdd_def"], 0), "r"),
        ("(n)", lambda r: r["agree_isdd_def_n"], "r"),
    ]))
    w("")

    # ------------------------------------------------------------------
    w("## 2. Les enchères\n")
    w("**Seuil d'annonce** : force de main (heuristique `evaluate_hand`, "
      "meilleure couleur) selon qu'il annonce ou qu'il passe. L'écart dit à "
      "quel point son annonce est *sélective* — un écart faible signale "
      "quelqu'un qui annonce presque indépendamment de sa main.\n")
    w("**Écart au plafond DD** : valeur annoncée moins le meilleur contrat que "
      "son camp aurait tenu en double-mort, moyenné sur les donnes où il est "
      "preneur. **Ce nombre est négatif pour tout le monde, Dédé compris, et "
      "c'est normal** : le plafond suppose que les quatre mains sont visibles "
      "et que la défense joue parfaitement elle aussi, donc aucun enchérisseur "
      "réel ne devrait l'atteindre. Seul le classement compte — près de zéro = "
      "annonce ambitieuse, très négatif = prudent. Biais à garder en tête : "
      "la mesure ne porte que sur les donnes où il a pris, donc quelqu'un qui "
      "ne prend qu'avec des mains énormes y paraîtra artificiellement prudent.\n")
    w(table(sorted(p, key=lambda x: -x["deals"]), [
        ("Joueur", lambda r: ("**" + r["label"] + "**") if r["human"]
         else r["label"] + " *(bot)*", "l"),
        ("Prises de parole", lambda r: r["bid_turns"], "r"),
        ("Passe", lambda r: pct(r["pass_rate"], 0), "r"),
        ("Annonce", lambda r: pct(r["bid_rate"], 0), "r"),
        ("Preneur", lambda r: pct(r["declared_rate"], 0), "r"),
        ("Contrat moyen", lambda r: num(r["mean_declared_value"], 0), "r"),
        ("Ouverture moy.", lambda r: num(r["mean_open_value"], 0), "r"),
        ("Force si annonce", lambda r: num(r["strength_when_bidding"], 0), "r"),
        ("Force si passe", lambda r: num(r["strength_when_passing"], 0), "r"),
        ("Écart", lambda r: signed(r["strength_gap"], 0), "r"),
        ("Écart plafond DD", lambda r: signed(r["overbid"], 1), "r"),
        ("Contrat tenu", lambda r: pct(r["contract_success"], 0), "r"),
        ("Accord Bid v6", lambda r: pct(r["bid_agree_v6"], 0), "r"),
        ("Perte Q", lambda r: num(r["bid_qloss"], 3), "r"),
        ("Coinches", lambda r: r["coinches"], "r"),
        ("dont justes", lambda r: pct(r["coinche_hit"], 0), "r"),
    ]))
    w("")

    # ------------------------------------------------------------------
    w("## 3. Le style de jeu\n")
    w("Chaque taux est rapporté à ses *occasions*, pas au nombre de cartes : "
      "« coupe » compte les fois où le joueur n'avait plus la couleur "
      "demandée **et** tenait de l'atout.\n")
    w(table(sorted(p, key=lambda x: -x["deals"]), [
        ("Joueur", lambda r: ("**" + r["label"] + "**") if r["human"]
         else r["label"] + " *(bot)*", "l"),
        ("Cartes", lambda r: r["cards"], "r"),
        ("Entame atout", lambda r: pct(r["opening_lead_trump"], 0), "r"),
        ("Attaque atout (preneur)", lambda r: pct(r["lead_trump_att"], 0), "r"),
        ("Attaque atout (défense)", lambda r: pct(r["lead_trump_def"], 0), "r"),
        ("Ouvre à l'As", lambda r: pct(r["lead_ace_rate"], 0), "r"),
        ("Coupe", lambda r: pct(r["ruff_rate"], 0), "r"),
        ("(occasions)", lambda r: r["ruff_chances"], "r"),
        ("Coupe sur le partenaire", lambda r: pct(r["ruff_on_partner"], 0), "r"),
        ("Surcoupe", lambda r: pct(r["overruff_rate"], 0), "r"),
        ("Sous-coupe", lambda r: pct(r["undertrump_rate"], 0), "r"),
        ("(occasions)", lambda r: r["undertrump_chances"], "r"),
        ("Points au partenaire", lambda r: num(r["partner_pts"], 1), "r"),
        ("Points à l'adversaire", lambda r: num(r["opp_pts"], 1), "r"),
    ]))
    w("")

    # ------------------------------------------------------------------
    w("## 4. Résultats\n")
    w(table(sorted(p, key=lambda x: -(x["mean_marked"] or 0)), [
        ("Joueur", lambda r: ("**" + r["label"] + "**") if r["human"]
         else r["label"] + " *(bot)*", "l"),
        ("Donnes", lambda r: r["deals"], "r"),
        ("Points marqués / donne", lambda r: num(r["mean_marked"], 0), "r"),
        ("Preneur (n)", lambda r: r["as_attacker"], "r"),
        ("Contrats tenus", lambda r: pct(r["attack_win"], 0), "r"),
        ("Défenseur (n)", lambda r: r["as_defender"], "r"),
        ("Chutes obtenues", lambda r: pct(r["defense_win"], 0), "r"),
    ]))
    w("")

    # ------------------------------------------------------------------
    if humans:
        w("## 5. Portraits\n")
        w("Chaque jugement est relatif au corpus — médiane des joueurs humains, "
          "et Dédé comme repère. Aucun seuil absolu : l'échelle de "
          "`evaluate_hand` et le taux normal de coupe ne se devinent pas, et "
          "une constante mal calibrée rendrait le même verdict pour tout le "
          "monde.\n")
        ref = reference(p)
        for h in sorted(humans, key=lambda x: -x["deals"]):
            w(f"### {h['label']}\n")
            for line in portrait(h, p, ref):
                w(f"- {line}")
            w("")

    return "\n".join(out)


def reference(profiles):
    """Les deux étalons d'un portrait : la médiane humaine, et Dédé.

    Un seuil absolu choisi à l'avance ne vaut rien ici — l'échelle de
    `evaluate_hand` comme le taux normal de coupe ne se devinent pas, et une
    constante mal calibrée produit un verdict confiant et faux pour *tout le
    monde*. Chaque jugement se lit donc par rapport au corpus.
    """
    humans = [p for p in profiles if p["human"]]
    keys = [k for k in (humans[0] if humans else {})
            if isinstance((humans[0] if humans else {}).get(k), (int, float))]
    med = {}
    for k in keys:
        vals = sorted(p[k] for p in humans if isinstance(p.get(k), (int, float)))
        if vals:
            med[k] = vals[len(vals) // 2]
    dede = next((p for p in profiles if p["key"] == "bot:dede"), None)
    return {"median": med, "dede": dede}


def _vs(value, ref_value, unit="", digits=1, higher="plus", lower="moins",
        same="au niveau de la médiane", tol=0.0):
    """Sens de l'écart à l'étalon, avec une bande morte : sans elle le joueur
    médian s'entend dire qu'il est « moins X » que lui-même."""
    if value is None or ref_value is None:
        return None
    delta = value - ref_value
    if abs(delta) <= tol:
        word = same
    else:
        word = higher if delta > 0 else lower
    return f"{word} ({num(value, digits)}{unit} contre {num(ref_value, digits)}{unit})"


def portrait(h, allp, ref):
    """Quelques phrases sur un joueur, chacune adossée au corpus."""
    lines = []
    med = ref["median"]
    dede = ref["dede"] or {}
    n = h["deals"]
    lines.append(f"**{n} donne(s)** ({h['deals_solo']} en solo, "
                 f"{h['deals_multi']} en salon), {h['cards']} cartes jouées, "
                 f"{h['dd_decisions']} décisions chiffrées.")

    if n < 20:
        lines.append("⚠️ **Échantillon trop mince pour un portrait fiable** — "
                     "tout ce qui suit est indicatif.")

    pr = h["pass_rate"]
    if pr is not None and med.get("pass_rate") is not None:
        m = med["pass_rate"]
        tag = ("prudent" if pr > m + 0.04
               else "offensif" if pr < m - 0.04 else "dans la moyenne")
        lines.append(f"Enchères : passe {pct(pr, 0)} de ses prises de parole "
                     f"(médiane humaine {pct(m, 0)}, Dédé "
                     f"{pct(dede.get('pass_rate'), 0)}) — **{tag}**.")

    if h["overbid"] is not None and h["declared"]:
        ov, m = h["overbid"], med.get("overbid")
        sem = h["overbid_sem"]
        tag = ""
        if m is not None:
            tag = (" — **plus ambitieux que la médiane**" if ov > m + 3
                   else " — **plus prudent que la médiane**" if ov < m - 3
                   else " — dans la médiane")
        lines.append(
            f"Preneur sur {h['declared']} donne(s), contrat moyen "
            f"{num(h['mean_declared_value'], 0)} pour "
            f"{num(h['dd_makeable'], 0)} tenable en double-mort : écart "
            f"{signed(ov, 1)}{'' if sem is None else ' ± ' + num(sem, 1)} pts "
            f"(médiane humaine {signed(m, 1)}, Dédé "
            f"{signed(dede.get('overbid'), 1)}){tag}. "
            f"Contrat tenu {pct(h['contract_success'], 0)}.")

    if h["strength_gap"] is not None:
        m = med.get("strength_gap")
        cmp_ = _vs(h["strength_gap"], m, digits=1, tol=0.3,
                   higher="plus sélectif", lower="moins sélectif",
                   same="aussi sélectif que la médiane")
        lines.append(
            f"Sélectivité : main à {num(h['strength_when_bidding'], 0)} quand "
            f"il annonce contre {num(h['strength_when_passing'], 0)} quand il "
            f"passe, soit un écart de {signed(h['strength_gap'], 1)}"
            + (f" — **{cmp_}**." if cmp_ else "."))

    if h["ruff_rate"] is not None and h["ruff_chances"] >= 20:
        rp = h["ruff_on_partner"]
        dp = dede.get("ruff_on_partner")
        tail = ""
        if rp is not None and dp is not None:
            tail = (" — **plus dépensier en atout que Dédé**"
                    if rp > dp + 0.05 else
                    " — **plus économe en atout que Dédé**"
                    if rp < dp - 0.05 else " — comme Dédé")
        lines.append(
            f"Atout : coupe {pct(h['ruff_rate'], 0)} de ses occasions "
            f"({h['ruff_chances']}), dont {pct(rp, 0)} quand son partenaire "
            f"tenait déjà le pli (Dédé {pct(dp, 0)}){tail}.")

    if h["undertrump_chances"] >= 5 and h["undertrump_rate"] is not None:
        lines.append(
            f"« Ne pisse pas » : sous-coupe {pct(h['undertrump_rate'], 0)} des "
            f"{h['undertrump_chances']} fois où il pouvait se défausser à la "
            f"place (Dédé {pct(dede.get('undertrump_rate'), 0)}).")

    if h["opp_pts"] is not None:
        lines.append(
            f"Points : donne {num(h['partner_pts'], 1)} pt/coup au partenaire "
            f"qui tient le pli (médiane {num(med.get('partner_pts'), 1)}), et "
            f"{num(h['opp_pts'], 1)} pt/coup à l'adversaire qui le tient "
            f"(médiane {num(med.get('opp_pts'), 1)}).")

    if h["dd_avg_cost"] is not None:
        lines.append(
            f"Niveau (DD) : {num(h['dd_avg_cost'], 2)} pt perdu par décision "
            f"sur {h['dd_decisions']} (médiane humaine "
            f"{num(med.get('dd_avg_cost'), 2)}, Dédé "
            f"{num(dede.get('dd_avg_cost'), 2)}), {pct(h['dd_perfect'], 0)} de "
            f"coups parfaits, {pct(h['dd_blunder'], 0)} d'erreurs ou fautes.")
    if h["agree_isdd"] is not None:
        lines.append(
            f"Niveau (à information égale) : joue le coup de Dédé "
            f"{pct(h['agree_isdd'], 0)} du temps sur {h['agree_isdd_n']} "
            f"décisions (médiane humaine {pct(med.get('agree_isdd'), 0)}), "
            f"celui de DouDou50 {pct(h['agree_doudou'], 0)}, "
            f"celui de l'Oracle {pct(h['agree_oracle'], 0)}.")
    if h["bid_agree_v6"] is not None:
        lines.append(
            f"Enchères vs Bid v6 : même annonce {pct(h['bid_agree_v6'], 0)} du "
            f"temps sur {h['bid_judged']} prises de parole (médiane "
            f"{pct(med.get('bid_agree_v6'), 0)}), perte de Q moyenne "
            f"{num(h['bid_qloss'], 3)}.")
    return lines


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=os.environ.get("COLVER_DB_PATH", str(DEFAULT_DB)))
    ap.add_argument("--mode", choices=["play", "multi"], default=None,
                    help="ne garder que le solo ou que le salon")
    ap.add_argument("--min-deals", type=int, default=1,
                    help="masquer les identités sous ce nombre de donnes")
    ap.add_argument("--compute", action="store_true",
                    help="calculer les coûts DD manquants (table `analysis`)")
    ap.add_argument("--compute-review", action="store_true",
                    help="calculer la revue des bots manquante (`agent_review`) "
                         "— lent, et il faut $COLVER_PLAYGEN_GPU_URL")
    ap.add_argument("--limit", type=int, default=None,
                    help="ne traiter que les N donnes les plus récentes")
    ap.add_argument("--shard", type=int, default=None,
                    help="avec --of : ne calculer que la part i des donnes")
    ap.add_argument("--of", type=int, default=None,
                    help="nombre de shards parallèles pour --compute-review")
    ap.add_argument("--quiet-report", action="store_true",
                    help="ne rien écrire sur stdout (shards de calcul)")
    ap.add_argument("--json", dest="json_out", default=None)
    ap.add_argument("--md", dest="md_out", default=None)
    args = ap.parse_args()

    import colver  # noqa: F401 — coûteux, on ne l'importe qu'ici

    conn = open_db(args.db)
    games = load_games(conn, args.mode)
    if args.limit:
        games = games[-args.limit:]
    if not games:
        sys.exit("Aucune donne complète dans cette base.")

    from colver.web.analysis import ANALYSIS_VERSION
    from colver.web.agent_review import REVIEW_VERSION

    analyses = load_cached(conn, "analysis", "version", ANALYSIS_VERSION)
    reviews = load_cached(conn, "agent_review", "version", REVIEW_VERSION)

    if args.compute:
        compute_missing_analysis(conn, games, analyses)
    if args.compute_review:
        compute_missing_review(conn, games, reviews,
                               shard=args.shard, of=args.of)
    if args.quiet_report:
        return

    stats = {}
    truncated = []
    for g in games:
        ids = seat_identities(conn, g)
        ev, meta = replay_features(colver, g)
        if meta["truncated"]:
            truncated.append(g["id"])
        an = analyses.get(g["id"])
        rv = reviews.get(g["id"])
        for seat, (key, label, human) in enumerate(ids):
            st = stats.get(key)
            if st is None:
                st = stats[key] = Stats(key, label, human)
            merge(st, ev[seat], meta, seat, g, an, rv)

    profiles = [profile(s) for s in stats.values()
                if s.c["deals"] >= args.min_deals]
    profiles.sort(key=lambda x: (not x["human"], -x["deals"]))

    corpus = {
        "deals": len(games),
        "from": games[0]["created_at"][:10],
        "to": games[-1]["created_at"][:10],
        "with_analysis": sum(1 for g in games if g["id"] in analyses),
        "with_review": sum(1 for g in games if g["id"] in reviews),
        "truncated": len(truncated),
        "truncated_ids": truncated[:20],
        "db": args.db,
    }

    report = render(profiles, corpus)
    if args.md_out:
        Path(args.md_out).write_text(report, encoding="utf-8")
        print(f"→ {args.md_out}", file=sys.stderr)
    else:
        print(report)
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"corpus": corpus, "players": profiles},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"→ {args.json_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
