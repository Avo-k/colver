"""Le score de partie transporté jusqu'aux pages d'analyse (web_todo §4.1).

Ce n'est pas un affichage de plus. Bid v6 lit une observation *score-aware* :
la même main s'annonce autrement à 900-200 qu'à 0-0. Toutes les pages
d'analyse raisonnaient à 0-0, donc l'avis « Bid V6 » affiché sous une annonce
répondait à une autre question que celle que le joueur s'était posée à la
table.

Trois choses se testent ici, et la troisième est celle qui coûterait le plus
cher à casser :

- **Le cumul est celui d'*avant* la donne** — c'est le seul état qui existait
  au moment d'annoncer. Il se recalcule depuis `games.score_ns/ew` (v16), pas
  depuis `matches.points_ns/ew` qui ne porte que le total de la partie.
- **Une partie en cours ne dit son score qu'à son propriétaire.** Une donne
  close est publique et son identifiant fait quatre caractères ; sans ce
  filtre, Rejouer donnerait le score en direct d'une table où l'on joue encore.
- **Le bump v5 → v6 ne jette pas le cache des donnes jouées à 0-0.** Les deux
  versions calculent alors exactement le même blob, et re-solver toutes les
  donnes isolées du site — son cas par défaut — coûterait des milliers de
  recherches DD pleine donne pour un résultat identique.
"""

import pytest

import colver.web.analysis as analysis
import colver.web.database as db

from conftest import await_sync

pytest.importorskip("httpx", reason="fastapi.testclient a besoin de httpx")

from fastapi.testclient import TestClient  # noqa: E402

import colver.web.server as server  # noqa: E402


async def _user(name="alice"):
    return await db.create_user(name, "$2b$12$fake")


async def _deal(match_id, deal_no, hands, actions, *, scores=(160, 0),
                user_id=None, complete=True, invalid=False):
    """Une donne d'une partie, enregistrée comme le ferait la production."""
    game_id = await db.create_game(
        "play", 0, hands, {str(s): "doudou" for s in range(4)},
        human_seat=2, user_id=user_id, match_id=match_id, deal_no=deal_no)
    for entry in actions:
        await db.append_action(game_id, entry)
    if complete:
        await db.complete_game(
            game_id, 92, 60, None,
            score_ns=None if scores is None else scores[0],
            score_ew=None if scores is None else scores[1])
    if invalid:
        await db.mark_game_checked(game_id, "carte jouée deux fois")
    return game_id


class TestContexteDeDonne:
    async def test_rien_hors_partie(self, clean_db, played_deal):
        """`target = 0` ne crée pas de ligne `matches` : c'est le cas par
        défaut du site, et 0-0 y est la vérité, pas un repli."""
        hands, actions = played_deal(seed=1)
        game_id = await _deal(None, None, hands, actions)
        game = await db.get_game(game_id)
        assert await db.deal_match_context(game) is None
        assert await analysis.match_scores_before(game) == [0, 0]

    async def test_le_cumul_est_celui_d_avant_la_donne(self, clean_db, played_deal):
        uid = await _user()
        match_id = await db.create_match("play", 2000, user_id=uid, human_seat=2)
        hands, actions = played_deal(seed=2)
        await _deal(match_id, 1, hands, actions, scores=(160, 0), user_id=uid)
        await _deal(match_id, 2, hands, actions, scores=(0, 262), user_id=uid)
        third = await _deal(match_id, 3, hands, actions, scores=(180, 0), user_id=uid)
        await db.update_match(match_id, 340, 262, 3, True, 0)

        ctx = await db.deal_match_context(await db.get_game(third))
        # 160 + 0 pour N-S, 0 + 262 pour E-O : la troisième donne ne se compte
        # pas elle-même.
        assert ctx["before"] == [160, 262]
        assert ctx["score"] == [180, 0]
        assert ctx["after"] == [340, 262]
        assert ctx["deal_no"] == 3
        assert ctx["target"] == 2000
        assert ctx["unscored_before"] == 0

    async def test_la_premiere_donne_est_a_zero(self, clean_db, played_deal):
        """Le seul cas où 0-0 est à la fois un début de partie et la vérité."""
        uid = await _user()
        match_id = await db.create_match("play", 2000, user_id=uid, human_seat=2)
        hands, actions = played_deal(seed=3)
        first = await _deal(match_id, 1, hands, actions, user_id=uid)
        ctx = await db.deal_match_context(await db.get_game(first))
        assert ctx["before"] == [0, 0]

    async def test_une_donne_sans_score_marque_se_dit(self, clean_db, played_deal):
        """`integrity.backfill_scores` n'a pas encore rattrapé cette donne : le
        cumul est trop bas, et le taire le ferait passer pour exact."""
        uid = await _user()
        match_id = await db.create_match("play", 2000, user_id=uid, human_seat=2)
        hands, actions = played_deal(seed=4)
        await _deal(match_id, 1, hands, actions, scores=None, user_id=uid)
        await _deal(match_id, 2, hands, actions, scores=(160, 0), user_id=uid)
        third = await _deal(match_id, 3, hands, actions, user_id=uid)

        ctx = await db.deal_match_context(await db.get_game(third))
        assert ctx["before"] == [160, 0]
        assert ctx["unscored_before"] == 1

    async def test_une_donne_ecartee_ne_compte_pas(self, clean_db, played_deal):
        """Même règle que la feuille de marque : une donne en quarantaine
        décrit une partie impossible, elle ne compte nulle part."""
        uid = await _user()
        match_id = await db.create_match("play", 2000, user_id=uid, human_seat=2)
        hands, actions = played_deal(seed=5)
        await _deal(match_id, 1, hands, actions, scores=(500, 0),
                    user_id=uid, invalid=True)
        await _deal(match_id, 2, hands, actions, scores=(160, 0), user_id=uid)
        third = await _deal(match_id, 3, hands, actions, user_id=uid)

        ctx = await db.deal_match_context(await db.get_game(third))
        assert ctx["before"] == [160, 0]


class TestFraicheurDuCache:
    """v5 → v6 n'a changé que le score lu par le bidder."""

    def test_v5_reste_bonne_a_zero_zero(self):
        cached = {"version": 5, "playgen": True}
        assert analysis._is_fresh(cached, None, [0, 0])
        assert analysis._is_fresh(cached, None, None)

    def test_v5_est_perimee_des_qu_il_y_a_un_score(self):
        cached = {"version": 5, "playgen": True}
        assert not analysis._is_fresh(cached, None, [940, 620])

    def test_v6_est_toujours_bonne(self):
        cached = {"version": 6, "playgen": True}
        assert analysis._is_fresh(cached, None, [940, 620])

    def test_une_version_anterieure_reste_perimee(self):
        assert not analysis._is_fresh({"version": 4, "playgen": True}, None, [0, 0])


class TestWebSocket:
    """Ce que `replay_loaded` transporte — et ce qu'il refuse de transporter."""

    @pytest.fixture
    def client(self, clean_db):
        with TestClient(server.app) as c:
            yield c

    @staticmethod
    def _load(client, game_id, cookies=None):
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "replay_load", "game_id": game_id})
            while True:
                msg = ws.receive_json()
                if msg["type"] in ("replay_loaded", "error"):
                    return msg

    def test_une_partie_close_porte_son_score(self, client, played_deal):
        hands, actions = played_deal(seed=6)

        async def _setup():
            uid = await _user()
            match_id = await db.create_match("play", 2000, user_id=uid, human_seat=2)
            await _deal(match_id, 1, hands, actions, scores=(160, 0), user_id=uid)
            gid = await _deal(match_id, 2, hands, actions, scores=(0, 262), user_id=uid)
            await db.update_match(match_id, 160, 262, 2, True, 1)
            return gid

        game_id = await_sync(_setup())
        msg = self._load(client, game_id)
        assert msg["type"] == "replay_loaded"
        # Publique comme la donne l'est : pas de session, et le score est là.
        assert msg["match"]["before"] == [160, 0]
        assert msg["match"]["target"] == 2000
        # `owner_id` ne sort jamais du serveur — il ne sert qu'au filtre.
        assert "owner_id" not in msg["match"]

    def test_une_partie_en_cours_se_tait_devant_un_inconnu(self, client, played_deal):
        """L'identifiant d'une donne fait quatre caractères et la donne est
        publique : sans ce filtre, n'importe qui lirait le score en direct
        d'une table où l'on joue encore."""
        hands, actions = played_deal(seed=7)

        async def _setup():
            uid = await _user()
            match_id = await db.create_match("play", 2000, user_id=uid, human_seat=2)
            return await _deal(match_id, 1, hands, actions, user_id=uid)

        game_id = await_sync(_setup())
        msg = self._load(client, game_id)
        assert msg["type"] == "replay_loaded"
        assert msg["match"] is None

    def test_une_donne_isolee_n_a_pas_de_partie(self, client, played_deal):
        hands, actions = played_deal(seed=8)
        game_id = await_sync(_deal(None, None, hands, actions))
        assert self._load(client, game_id)["match"] is None


class TestBidEval:
    """Le score envoyé par la page annonces atteint bien le réseau."""

    @pytest.fixture
    def client(self, clean_db):
        with TestClient(server.app) as c:
            yield c

    @staticmethod
    def _eval(client, hand, scores):
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "bid_eval", "hand": hand,
                          "prior_actions": [], "scores": scores})
            return ws.receive_json()

    def test_le_score_est_renvoye_tel_qu_utilise(self, client):
        """Sans cet écho, deux réponses différentes sur la même main
        n'auraient aucune explication visible à l'écran."""
        msg = self._eval(client, [0, 1, 2, 3, 8, 9, 16, 24], [940, 620])
        if msg.get("error"):
            pytest.skip(f"pas de modèle d'enchères ici : {msg['error']}")
        assert msg["scores"] == [940, 620]

    @pytest.mark.parametrize("bad", ["940-620", [940], None, ["a", "b"], [-5, 10**9]])
    def test_un_score_bancal_retombe_sur_zero(self, bad):
        """La valeur vient de l'URL : elle est bornée avant d'atteindre une
        observation qui normalise sur la cible."""
        assert server._client_match_scores({"scores": bad}) in ([0, 0], [0, 5000])

    def test_les_bornes(self):
        assert server._client_match_scores({"scores": [940, 620]}) == [940, 620]
        assert server._client_match_scores({}) == [0, 0]
