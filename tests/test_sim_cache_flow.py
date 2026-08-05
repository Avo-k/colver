"""Le cache vu du protocole : deux passages sur la même position.

Les tests de `test_sim_cache.py` couvrent les clés et le stockage. Ceux-ci
pilotent le vrai gestionnaire `_run_card_analysis` et vérifient ce que le client
reçoit — c'est là que se joue le risque réel, un cache servi n'étant utile que
si les messages qu'il rejoue suffisent à peindre la page.

Deux choses sont bouchonnées, et seulement deux : `sample_worlds` (le sidecar
n'existe pas en test, donc les mondes retomberaient en « uniform » et rien ne
serait mis en cache — comportement correct, mais qui ne prouve rien ici) et
`opinions` (Dédé exige le sidecar à la construction). Tout le reste — le rejeu
de la position, les solves DD, l'agrégation, l'écriture — tourne pour de vrai.
"""

import asyncio

import pytest

import colver
import colver.web.card_analysis as card_analysis
import colver.web.game_manager as game_manager
import colver.web.server as server
import colver.web.sim_cache as sim_cache


class FakeWS:
    """Un WebSocket réduit à ce que le gestionnaire en attend."""

    def __init__(self):
        self.sent = []

    async def send_json(self, payload):
        self.sent.append(payload)

    def types(self):
        return [m["type"] for m in self.sent]

    def last(self, kind):
        return next(m for m in reversed(self.sent) if m["type"] == kind)


def _position(played_deal, seed=3):
    """Une position de jeu réelle : CFN complet + index d'une décision.

    On vise la deuxième carte du premier pli : suffisamment de cartes cachées
    pour que l'analyse ait du sens, et une position que le rejeu atteint vite.
    """
    hands, actions = played_deal(seed=seed)
    ids = [a["action"] for a in actions]
    cfn = game_manager.compute_game_cfn(0, hands, ids)
    play_idxs = [i for i, a in enumerate(actions) if a["phase"] == 1]
    for idx in play_idxs:
        pos = card_analysis.describe(0, hands, ids, idx)
        if "error" not in pos and not pos["forced"]:
            return hands, ids, cfn, idx
    pytest.skip("aucune décision non forcée dans cette donne")


@pytest.fixture
def small_budget(monkeypatch):
    """Réduire l'échantillon : ce qui se teste ici est le câblage, pas la
    précision. Le budget de prod (200 à 500 mondes, chacun un solve sur donne
    quasi complète) coûterait une demi-minute par test."""
    monkeypatch.setattr(card_analysis, "plan",
                        lambda pos: {"oracle_worlds": 4, "real_worlds": 2})


@pytest.fixture
def stub_opinions(monkeypatch):
    """Avis figés : Dédé exige le sidecar à la construction, absent en test."""
    monkeypatch.setattr(
        card_analysis, "opinions",
        lambda *a, **kw: {"oracle": 0, "isdd": 0, "doudou": 0})
    # `doudou_expected` se lit sur ce chemin de modèle ; on le fixe pour que le
    # test ne dépende pas des poids présents sur la machine.
    monkeypatch.setattr(server, "DMC_MODEL_PATH", None)


@pytest.fixture
def stub_expensive(monkeypatch, small_budget, stub_opinions):
    """Mondes déclarés playgen, pour que le cache ait le droit de s'écrire.

    Sur une machine de dev le sampler local répond déjà « playgen » ; sur une
    CI sans poids il répondrait « uniform » et rien ne serait gardé. On force
    donc la source pour que ces tests disent la même chose des deux côtés.
    """
    real = card_analysis.sample_worlds

    def fake_sample(*args, **kwargs):
        worlds, _src = real(*args, **kwargs)
        return worlds, "playgen"

    monkeypatch.setattr(card_analysis, "sample_worlds", fake_sample)


@pytest.mark.asyncio
async def test_second_visit_is_served_from_the_database(
        clean_db, played_deal, stub_expensive):
    _hands, _ids, cfn, idx = _position(played_deal)
    data = {"cfn": cfn, "idx": idx, "req_id": 1}

    cold = FakeWS()
    await server._run_card_analysis(cold, data)
    assert "card_analysis_done" in cold.types()
    # Le calcul a bien traversé la phase de mondes.
    assert "card_analysis_update" in cold.types()

    rows = await clean_db.execute_fetchall(
        "SELECT COUNT(*) FROM analysis_cache WHERE kind = ?", (sim_cache.KIND_CARD,))
    assert rows[0][0] == 1, "le résultat complet aurait dû être gardé"

    warm = FakeWS()
    await server._run_card_analysis(warm, data)

    # Un service depuis la base ne repasse par aucun monde.
    assert "card_analysis_update" not in warm.types()
    assert warm.types() == ["card_analysis_position", "card_analysis_truth",
                            "card_analysis_opinions", "card_analysis_done"]

    hot, cached = cold.last("card_analysis_done"), warm.last("card_analysis_done")
    assert cached["cached"] is True
    assert cached["rows"] == hot["rows"]
    assert cached["completed"] == hot["completed"]
    assert cached["worlds_source"] == hot["worlds_source"]
    # Le chrono du premier calcul ne doit pas être resservi comme celui-ci.
    assert cached["elapsed_ms"] is None
    assert warm.last("card_analysis_truth")["truth"] == cold.last("card_analysis_truth")["truth"]


@pytest.mark.asyncio
async def test_uniform_worlds_are_never_stored(
        clean_db, played_deal, monkeypatch, small_budget, stub_opinions):
    """Sans sidecar, le résultat est juste mais dégradé : il ne se fige pas.

    C'est la panne qu'`agent_review` a eue en cache avec le sidecar éteint —
    ici la règle est prise à l'écriture, donc elle ne peut pas être oubliée à
    la lecture.
    """
    real = card_analysis.sample_worlds
    monkeypatch.setattr(
        card_analysis, "sample_worlds",
        lambda *a, **kw: (real(*a, **kw)[0], "uniform"))
    _hands, _ids, cfn, idx = _position(played_deal, seed=4)

    ws = FakeWS()
    await server._run_card_analysis(ws, {"cfn": cfn, "idx": idx, "req_id": 1})
    assert ws.last("card_analysis_done")["worlds_source"] == "uniform"

    rows = await clean_db.execute_fetchall("SELECT COUNT(*) FROM analysis_cache")
    assert rows[0][0] == 0


@pytest.mark.asyncio
async def test_a_cancelled_analysis_leaves_nothing_behind(
        clean_db, played_deal, stub_expensive):
    """Une analyse interrompue (le joueur clique ailleurs) ne fige rien.

    L'écriture est après l'envoi du message final, donc une annulation ne peut
    pas la traverser — mais c'est le genre d'invariant qu'un déplacement de
    ligne casse en silence.
    """
    _hands, _ids, cfn, idx = _position(played_deal, seed=5)
    ws = FakeWS()
    task = asyncio.create_task(
        server._run_card_analysis(ws, {"cfn": cfn, "idx": idx, "req_id": 1}))
    await asyncio.sleep(0)  # laisser partir la position, pas plus
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    rows = await clean_db.execute_fetchall("SELECT COUNT(*) FROM analysis_cache")
    assert rows[0][0] == 0


@pytest.mark.asyncio
async def test_doudou_only_path_is_cached(clean_db, monkeypatch):
    """`annonces_doudou` — le chemin réellement emprunté par la page.

    L'Oracle tourne en WASM dans le navigateur, donc le serveur ne voit que
    cette moitié-là : c'est elle qui coûte ~20 s de CPU par évaluation, et elle
    n'était pas dans le premier jet de ce cache.
    """
    monkeypatch.setattr(server, "BID_MODEL_PATH", "fake-bid.bin")
    monkeypatch.setattr(server, "DMC_MODEL_PATH", "fake-dmc.bin")
    calls = []

    def fake_sim(hand, remaining, *a, **kw):
        calls.append(1)
        return {"doudou": {"void": False, "trump": 0, "value": 90, "team": 0,
                           "coinche": 0, "achieved": True,
                           "auction": [], "scores": [180.0, 0.0]}}

    monkeypatch.setattr(server, "_run_single_doudou_sim", fake_sim)
    hand = [0, 1, 2, 3, 4, 5, 6, 7]
    data = {"hand": hand, "num_sims": 3, "req_id": 1}

    cold = FakeWS()
    await server._run_annonces_doudou(cold, data)
    assert len(calls) == 3

    warm = FakeWS()
    await server._run_annonces_doudou(warm, dict(data))
    assert len(calls) == 3, "la seconde évaluation ne doit rien rejouer"
    assert warm.types() == ["annonces_doudou_done"]
    done = warm.last("annonces_doudou_done")
    assert done["cached"] is True
    assert done["doudou_cells"] == cold.last("annonces_doudou_done")["doudou_cells"]

    # Forcer une annonce est une autre question : autre clé, donc recalcul.
    await server._run_annonces_doudou(FakeWS(), {**data, "forced_action": 1})
    assert len(calls) == 6


@pytest.mark.asyncio
async def test_an_illegal_forced_bid_is_refused_even_with_an_entry(
        clean_db, monkeypatch):
    """Le contrôle de légalité passe **avant** le cache.

    Sinon une entrée écrite sous une enchère donnée servirait de laissez-passer
    à une demande impossible — un cache ne doit jamais élargir ce qui est
    acceptable.
    """
    monkeypatch.setattr(server, "BID_MODEL_PATH", "fake-bid.bin")
    monkeypatch.setattr(server, "DMC_MODEL_PATH", "fake-dmc.bin")
    monkeypatch.setattr(
        server, "_run_single_doudou_sim",
        lambda *a, **kw: {"doudou": {"void": True}})
    hand = [0, 1, 2, 3, 4, 5, 6, 7]
    # 80♠ après 80♠ : sous-enchère, donc illégale quelle que soit la donne.
    data = {"hand": hand, "num_sims": 2, "req_id": 1,
            "prior_actions": [1], "forced_action": 1}

    ws = FakeWS()
    await server._run_annonces_doudou(ws, data)
    assert "illégale" in ws.sent[0]["error"]

    rows = await clean_db.execute_fetchall(
        "SELECT COUNT(*) FROM analysis_cache WHERE kind = ?",
        (sim_cache.KIND_DOUDOU,))
    assert rows[0][0] == 0


@pytest.mark.asyncio
async def test_annonces_sim_serves_both_tables_from_one_entry(
        clean_db, monkeypatch):
    """Le repli serveur : une entrée porte l'Oracle **et** le Jeu réel.

    Le client reconstruit chaque tableau depuis son seul message final, donc
    deux messages suffisent à repeindre la page — c'est ce qui rend un service
    depuis la base indistinguable d'un calcul, l'attente en moins.
    """
    monkeypatch.setattr(server, "BID_MODEL_PATH", "fake-bid.bin")
    monkeypatch.setattr(server, "DMC_MODEL_PATH", "fake-dmc.bin")
    hand = [0, 1, 2, 3, 4, 5, 6, 7]
    key = sim_cache.bid_key(hand, [], 200, 1000)
    await sim_cache.put(sim_cache.KIND_BID, key, sim_cache.BID_SIM_VERSION, {
        "oracle": {"completed": 200, "total": 200, "elapsed_ms": None,
                   "success_counts": [[1] * 10] * 4, "oracle_synth": {},
                   "sampled_deals": [], "sampled_sources": [],
                   "worlds_source": "playgen", "worlds_counts": {"playgen": 200}},
        "doudou": {"completed": 1000, "total": 1000, "elapsed_ms": None,
                   "doudou_cells": [], "doudou_stats": {}},
    })

    ws = FakeWS()
    await server._run_annonces_sim(ws, {"hand": hand, "req_id": 7,
                                        "oracle_sims": 200, "doudou_sims": 1000})
    assert ws.types() == ["annonces_sim_done", "annonces_doudou_done"]
    assert all(m["cached"] is True for m in ws.sent)
    assert all(m["req_id"] == 7 for m in ws.sent)
    assert ws.sent[0]["worlds_source"] == "playgen"
    assert ws.sent[1]["total"] == 1000


@pytest.mark.asyncio
async def test_a_later_divergence_shares_the_entry(
        clean_db, played_deal, stub_expensive):
    """Deux CFN qui ne diffèrent qu'**après** la position analysée : une entrée.

    C'est ce que la clé promet, et c'est ce qui fait qu'un cache sert : sans ça
    chaque donne aurait ses propres entrées même sur des positions identiques.
    """
    hands, ids, cfn, idx = _position(played_deal, seed=6)
    ws = FakeWS()
    await server._run_card_analysis(ws, {"cfn": cfn, "idx": idx, "req_id": 1})
    rows = await clean_db.execute_fetchall("SELECT COUNT(*) FROM analysis_cache")
    assert rows[0][0] == 1

    # Même préfixe, suite différente : on rejoue la donne en changeant un coup
    # postérieur à `idx`, ce qui doit retomber sur la même entrée.
    env = colver.Env.deal_with_hands(0, [list(h) for h in hands])
    for a in ids[:idx + 1]:
        env.step(int(a))
    variant = list(ids[:idx + 1])
    while not env.is_terminal():
        legal = list(env.legal_actions())
        a = legal[-1] if int(env.phase()) == 1 else int(env.bid_improved())
        variant.append(int(a))
        env.step(int(a))
    assert variant != ids

    other_cfn = game_manager.compute_game_cfn(0, hands, variant)
    assert other_cfn != cfn

    ws2 = FakeWS()
    await server._run_card_analysis(ws2, {"cfn": other_cfn, "idx": idx, "req_id": 2})
    assert ws2.last("card_analysis_done").get("cached") is True
    rows = await clean_db.execute_fetchall("SELECT COUNT(*) FROM analysis_cache")
    assert rows[0][0] == 1
