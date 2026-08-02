"""La « vraie donne » sur la page annonces (web_todo §4.2).

Une valeur exacte à côté de mondes échantillonnés : ce qui se teste ici, c'est
qu'elle décrive bien *cette* donne-là et *ce* siège-là, et qu'elle n'apparaisse
pas quand on ne connaît pas les quatre mains.
"""

import json

import pytest

import colver
import colver.web.analysis as analysis
import colver.web.database as db

from conftest import await_sync


async def _store(hands, actions, dealer=0, complete=True):
    game_id = await db.create_game("play", dealer, hands, {"0": "doudou"}, human_seat=2)
    for entry in actions:
        await db.append_action(game_id, entry)
    if complete:
        await db.complete_game(game_id, 80, 82, None)
    return game_id


def _first_bid_idx(actions):
    return next(i for i, a in enumerate(actions) if a["phase"] == 0)


def _first_play_idx(actions):
    return next(i for i, a in enumerate(actions) if a["phase"] == 1)


class TestTrueWorld:
    async def test_rend_les_points_du_camp_qui_parle(self, clean_db, played_deal):
        hands, actions = played_deal(seed=1)
        game_id = await _store(hands, actions)
        idx = _first_bid_idx(actions)

        tw, err = await analysis.true_world(game_id, idx)
        assert err is None

        seat = actions[idx]["player"]
        assert tw["seat"] == seat
        assert tw["team"] == seat % 2
        assert tw["hand"] == sorted(hands[seat])

        # Les mêmes valeurs que le solveur, lues du côté de ce camp.
        env = colver.Env.deal_with_hands(0, [list(h) for h in hands])
        expected = [int(s[seat % 2]) for s in env.solve_all_suits()["suits"]]
        assert tw["pts"] == expected

    async def test_sert_le_cache_de_rejouer(self, clean_db, played_deal):
        """Arriver depuis Rejouer ne doit rien recalculer : l'analyse de la
        donne a déjà résolu les quatre couleurs."""
        hands, actions = played_deal(seed=2)
        game_id = await _store(hands, actions)
        idx = _first_bid_idx(actions)
        seat = actions[idx]["player"]

        faux = [[10, 20], [30, 40], [50, 60], [70, 80]]
        await db.save_analysis(game_id, json.dumps({
            "version": analysis.ANALYSIS_VERSION,
            "oracle_bids": {"suits": faux,
                            "best": [{"suit": 0, "pts": 0, "value": 0}] * 2},
        }))

        tw, err = await analysis.true_world(game_id, idx)
        assert err is None
        assert tw["pts"] == [s[seat % 2] for s in faux]

    async def test_un_cache_d_une_autre_version_est_recalcule(self, clean_db, played_deal):
        """Un barème ou un coup légal qui change périme les valeurs DD."""
        hands, actions = played_deal(seed=2)
        game_id = await _store(hands, actions)
        idx = _first_bid_idx(actions)

        faux = [[10, 20], [30, 40], [50, 60], [70, 80]]
        await db.save_analysis(game_id, json.dumps({
            "version": analysis.ANALYSIS_VERSION - 1,
            "oracle_bids": {"suits": faux,
                            "best": [{"suit": 0, "pts": 0, "value": 0}] * 2},
        }))

        tw, err = await analysis.true_world(game_id, idx)
        assert err is None
        assert tw["pts"] != [s[actions[idx]["player"] % 2] for s in faux]

    async def test_refuse_un_coup_qui_n_est_pas_une_annonce(self, clean_db, played_deal):
        hands, actions = played_deal(seed=3)
        game_id = await _store(hands, actions)
        tw, err = await analysis.true_world(game_id, _first_play_idx(actions))
        assert tw is None and err

    async def test_refuse_un_index_hors_de_la_donne(self, clean_db, played_deal):
        hands, actions = played_deal(seed=3)
        game_id = await _store(hands, actions)
        for idx in (-1, len(actions)):
            tw, err = await analysis.true_world(game_id, idx)
            assert tw is None and err

    async def test_refuse_une_donne_inconnue(self, clean_db):
        tw, err = await analysis.true_world("zzzz", 0)
        assert tw is None and err

    async def test_refuse_une_donne_en_cours(self, clean_db, played_deal):
        """Les quatre mains sont dans la ligne : une donne non terminée ne
        divulgue rien, `get_game` la filtre déjà."""
        hands, actions = played_deal(seed=4)
        game_id = await _store(hands, actions[:6], complete=False)
        tw, err = await analysis.true_world(game_id, 0)
        assert tw is None and err


class TestWebSocket:
    """Le message que la page envoie en arrivant depuis Rejouer."""

    @pytest.fixture
    def client(self, clean_db):
        pytest.importorskip("httpx", reason="fastapi.testclient a besoin de httpx")
        from fastapi.testclient import TestClient

        import colver.web.server as server

        with TestClient(server.app) as c:
            yield c

    def test_repond_a_la_page(self, client, played_deal):
        hands, actions = played_deal(seed=7)
        game_id = await_sync(_store(hands, actions))
        idx = _first_bid_idx(actions)

        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "annonces_true_world",
                          "game_id": game_id, "action_idx": idx})
            msg = ws.receive_json()

        assert msg["type"] == "annonces_true_world"
        assert "error" not in msg
        assert msg["seat"] == actions[idx]["player"]
        assert len(msg["pts"]) == 4

    def test_une_requete_bancale_ne_casse_pas_la_socket(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "annonces_true_world",
                          "game_id": "zzzz", "action_idx": "pas un entier"})
            msg = ws.receive_json()
            assert msg["type"] == "annonces_true_world" and msg["error"]
            # La socket sert toujours après coup.
            ws.send_json({"type": "annonces_true_world",
                          "game_id": "zzzz", "action_idx": 0})
            assert ws.receive_json()["error"]
