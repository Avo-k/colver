"""Cache en base des deux simulations d'analyse (`analysis_cache`, migration v18).

Ce que ces tests tiennent, dans l'ordre d'importance :

1. **une clé décrit une position, pas une requête** — un CFN qui diffère par des
   coups postérieurs à l'index analysé doit retomber sur la même entrée, sinon
   le cache ne sert jamais entre deux donnes qui partagent un début ;
2. **un résultat dégradé ne s'écrit pas**, et une entrée écrite dégradée ne
   survit pas au retour du composant manquant — c'est la panne qu'`agent_review`
   a eue avec le sidecar éteint ;
3. **l'éviction est bornée**, la clé étant non bornée par construction.
"""

import json

import pytest

import colver.web.database as db
import colver.web.sim_cache as sim_cache


# ── les clés ──

def test_bid_key_ignores_hand_order():
    """La main est un ensemble : deux ordres de saisie sont la même question."""
    a = sim_cache.bid_key([5, 1, 30, 12, 3, 20, 8, 25], [], 200, 1000)
    b = sim_cache.bid_key([30, 25, 20, 12, 8, 5, 3, 1], [], 200, 1000)
    assert a == b


def test_bid_key_separates_auction_and_budget():
    base = sim_cache.bid_key([0, 1, 2, 3, 4, 5, 6, 7], [], 200, 1000)
    assert base != sim_cache.bid_key([0, 1, 2, 3, 4, 5, 6, 7], [5], 200, 1000)
    assert base != sim_cache.bid_key([0, 1, 2, 3, 4, 5, 6, 7], [], 400, 1000)
    assert base != sim_cache.bid_key([0, 1, 2, 3, 4, 5, 6, 7], [], 200, 500)


def test_card_key_ignores_actions_after_the_index():
    """Le cœur du gain : la clé ne porte que le préfixe.

    Deux donnes qui partagent leur début et divergent **après** la position
    analysée décrivent la même question. Inclure la suite diviserait le cache
    sans qu'aucun chiffre du tableau ne change.
    """
    hands = [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15],
             [16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31]]
    a = sim_cache.card_key(0, hands, [1, 0, 0, 0, 7, 15], 5)
    b = sim_cache.card_key(0, hands, [1, 0, 0, 0, 7, 23, 31], 5)
    assert a == b
    # …mais le préfixe lui-même sépare.
    assert a != sim_cache.card_key(0, hands, [1, 0, 0, 0, 6, 15], 5)
    assert a != sim_cache.card_key(1, hands, [1, 0, 0, 0, 7, 15], 5)


# ── le stockage ──

@pytest.mark.asyncio
async def test_roundtrip_and_version_gate(clean_db):
    await sim_cache.put(sim_cache.KIND_BID, "k1", 1, {"n": 42})
    assert await sim_cache.get(sim_cache.KIND_BID, "k1", 1) == {"n": 42}
    # Un bump de version périme sans effacer : l'entrée reste, elle ne sort plus.
    assert await sim_cache.get(sim_cache.KIND_BID, "k1", 2) is None
    # Les genres ne se mélangent pas malgré la clé commune.
    assert await sim_cache.get(sim_cache.KIND_CARD, "k1", 1) is None


@pytest.mark.asyncio
async def test_put_replaces_in_place(clean_db):
    await sim_cache.put(sim_cache.KIND_CARD, "k", 1, {"v": 1})
    await sim_cache.put(sim_cache.KIND_CARD, "k", 1, {"v": 2})
    assert await sim_cache.get(sim_cache.KIND_CARD, "k", 1) == {"v": 2}
    rows = await clean_db.execute_fetchall("SELECT COUNT(*) FROM analysis_cache")
    assert rows[0][0] == 1


@pytest.mark.asyncio
async def test_unreadable_blob_is_a_miss_not_a_crash(clean_db):
    """Un cache est un confort : illisible, il se comporte comme absent."""
    await clean_db.execute(
        "INSERT INTO analysis_cache (kind, cache_key, version, created_at, "
        "used_at, hits, data) VALUES (?, ?, ?, ?, ?, 0, ?)",
        (sim_cache.KIND_BID, "bad", 1, "2026-08-05T00:00:00", "2026-08-05T00:00:00",
         "{ pas du json"))
    await clean_db.commit()
    assert await sim_cache.get(sim_cache.KIND_BID, "bad", 1) is None


@pytest.mark.asyncio
async def test_lru_eviction_keeps_the_most_recently_used(clean_db, monkeypatch):
    """La clé est non bornée (un CFN se tape à la main) : la table doit l'être."""
    monkeypatch.setattr(db, "SIM_CACHE_MAX_ROWS", 3)
    for i in range(5):
        await sim_cache.put(sim_cache.KIND_CARD, f"k{i}", 1, {"i": i})
        # `used_at` est à la seconde : sans écart explicite, l'ordre d'éviction
        # entre deux écritures de la même seconde serait indéterminé.
        await clean_db.execute(
            "UPDATE analysis_cache SET used_at = ? WHERE cache_key = ?",
            (f"2026-08-0{i + 1}T00:00:00", f"k{i}"))
        await clean_db.commit()

    rows = await clean_db.execute_fetchall(
        "SELECT cache_key FROM analysis_cache ORDER BY cache_key")
    assert [r[0] for r in rows] == ["k2", "k3", "k4"]


@pytest.mark.asyncio
async def test_stats_count_services_not_days(clean_db):
    """`hits` doit compter les services : c'est le seul chiffre qui dise si les
    clés se rejoignent, donc si le cache sert à quelque chose."""
    await sim_cache.put(sim_cache.KIND_BID, "a", 1, {})
    await sim_cache.put(sim_cache.KIND_CARD, "b", 1, {})
    stats = await db.sim_cache_stats()
    assert stats[sim_cache.KIND_BID]["rows"] == 1
    assert stats[sim_cache.KIND_CARD]["rows"] == 1
    assert stats[sim_cache.KIND_BID]["hits"] == 0

    for _ in range(3):
        assert await sim_cache.get(sim_cache.KIND_BID, "a", 1) == {}
    # Une version périmée n'est pas un service.
    assert await sim_cache.get(sim_cache.KIND_BID, "a", 2) is None
    stats = await db.sim_cache_stats()
    assert stats[sim_cache.KIND_BID]["hits"] == 3
    assert stats[sim_cache.KIND_CARD]["hits"] == 0


# ── ce qui a le droit d'être gardé ──

def _bid_blob(**over):
    blob = {"worlds_source": "playgen", "completed": 200,
            "doudou": {"completed": 1000}}
    blob.update(over)
    return blob


def test_bid_degraded_worlds_are_not_cached():
    """Sans sidecar les mondes retombent en uniforme : justes, mais moins bons.

    Les figer les rendrait permanents bien après le retour du GPU.
    """
    assert sim_cache.bid_cacheable(_bid_blob(), 200, 1000, doudou_expected=True)
    for src in ("uniform", "mixte", None):
        assert not sim_cache.bid_cacheable(
            _bid_blob(worlds_source=src), 200, 1000, doudou_expected=True)


def test_bid_partial_sample_is_not_cached():
    assert not sim_cache.bid_cacheable(
        _bid_blob(completed=137), 200, 1000, doudou_expected=True)
    assert not sim_cache.bid_cacheable(
        _bid_blob(doudou={"completed": 600}), 200, 1000, doudou_expected=True)


def test_bid_without_doudou_only_cached_when_none_expected():
    blob = _bid_blob(doudou=None)
    assert not sim_cache.bid_cacheable(blob, 200, 1000, doudou_expected=True)
    assert sim_cache.bid_cacheable(blob, 200, 1000, doudou_expected=False)


def _card_blob(**over):
    blob = {"worlds_source": "playgen", "completed": 300,
            "opinions": {"oracle": 3, "isdd": 4, "doudou": 5}}
    blob.update(over)
    return blob


def test_card_needs_full_worlds_and_opinions():
    assert sim_cache.card_cacheable(_card_blob(), 300, doudou_expected=True)
    assert not sim_cache.card_cacheable(
        _card_blob(worlds_source="uniform"), 300, doudou_expected=True)
    assert not sim_cache.card_cacheable(
        _card_blob(completed=299), 300, doudou_expected=True)
    # Un avis manquant = un agent en panne pendant le calcul. On recalculera.
    assert not sim_cache.card_cacheable(
        _card_blob(opinions={"oracle": 3, "isdd": 4}), 300, doudou_expected=True)
    assert not sim_cache.card_cacheable(
        _card_blob(opinions={"oracle": 3, "doudou": 5}), 300, doudou_expected=True)
    assert sim_cache.card_cacheable(
        _card_blob(opinions={"oracle": 3, "isdd": 4}), 300, doudou_expected=False)


@pytest.mark.asyncio
async def test_json_shape_survives_the_roundtrip(clean_db):
    """Le blob passe par `json.dumps` : les clés entières deviendraient du texte.

    `truth.ns` est indexé par carte et le serveur l'écrit déjà en chaînes
    (`str(c)`) ; ce test épingle qu'on ne réintroduit pas d'entier en clé
    ailleurs, ce qui rendrait un cache hit subtilement différent d'un calcul.
    """
    payload = {"truth": {"best_card": 7, "ns": {"7": 90, "12": 84}},
               "opinions": {"oracle": 7, "isdd": 7, "doudou": 12},
               "result": {"rows": [{"card": 7, "dd": 90.0}], "elapsed_ms": None}}
    await sim_cache.put(sim_cache.KIND_CARD, "k", 1, payload)
    got = await sim_cache.get(sim_cache.KIND_CARD, "k", 1)
    assert got == json.loads(json.dumps(payload))
    assert got == payload
