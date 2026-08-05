"""L'historique des parties : la liste, et la feuille de marque.

Deux choses se testent ici, et la seconde est celle qui compte.

- **La liste** (`db.list_matches`) est le complément exact de
  `list_open_matches` : parties closes, abandons compris. Le siège du joueur ne
  se lit pas au même endroit selon le mode — `matches.human_seat` en solo,
  `game_players` en salon — et c'est ce qui décide de quel côté le score est lu.
- **La feuille** (`db.get_match`) recalcule un cumul donne par donne à partir
  des points *marqués* (migration v16). Il n'était pas calculable avant elle, et
  il peut ne pas retomber sur `matches.points_ns/ew` : la ligne de la partie a
  été écrite au fil du jeu, les scores par donne d'une vieille partie ont été
  rejoués sous le barème courant. **L'écart doit être rendu, pas masqué** — les
  tests l'épinglent explicitement, sinon la première divergence en production
  serait lue comme un bug de la feuille.
"""

import pytest

import colver.web.database as db

pytest.importorskip("httpx", reason="fastapi.testclient a besoin de httpx")

from fastapi.testclient import TestClient  # noqa: E402

import colver.web.server as server  # noqa: E402

PW = "motdepasse12"

# Une enchère minimale : Sud (siège 2) annonce 100♠, les trois autres passent.
# Il n'en faut pas plus pour que le preneur soit lisible — c'est la **dernière
# annonce chiffrée** qui le désigne, en phase 0 uniquement.
_AUCTION = [
    {"player": 2, "action": 9, "phase": 0},   # 100♠
    {"player": 3, "action": 0, "phase": 0},
    {"player": 0, "action": 0, "phase": 0},
    {"player": 1, "action": 0, "phase": 0},
]
_CONTRACT = {"value": 100, "trump": 0, "team": 0, "coinche": 0}


async def _user(name="alice"):
    return await db.create_user(name, "$2b$12$fake")


async def _deal(match_id, deal_no, *, user_id=None, scores=(160, 0),
                dealer=0, mode="play", seat=2, complete=True, invalid=False,
                auction=True):
    """Une donne d'une partie, enregistrée comme le ferait la production."""
    agents = {str(s): "doudou" for s in range(4)}
    if mode == "play":
        agents[str(seat)] = "human"
    game_id = await db.create_game(
        mode, dealer, [[0], [1], [2], [3]], agents,
        human_seat=seat if mode == "play" else None,
        user_id=user_id, match_id=match_id, deal_no=deal_no)
    if auction:
        for entry in _AUCTION:
            await db.append_action(game_id, entry)
    if complete:
        await db.complete_game(game_id, 92, 60, _CONTRACT,
                               score_ns=None if scores is None else scores[0],
                               score_ew=None if scores is None else scores[1])
    if invalid:
        await db.mark_game_checked(game_id, "carte jouée deux fois")
    return game_id


async def _match(user_id, *, target=2000, deals=((160, 0), (0, 262)),
                 winner=0, mode="play", seat=2, abandoned=False):
    """Une partie close et ses donnes, dans l'ordre."""
    match_id = await db.create_match(mode, target, user_id=user_id,
                                     pacing="standard",
                                     human_seat=seat if mode == "play" else None)
    totals = [0, 0]
    for i, scores in enumerate(deals, start=1):
        await _deal(match_id, i, user_id=user_id, scores=scores,
                    dealer=(i - 1) % 4, mode=mode, seat=seat)
        if scores:
            totals = [totals[0] + scores[0], totals[1] + scores[1]]
    if abandoned:
        await db.abandon_match(match_id, user_id)
    else:
        await db.update_match(match_id, totals[0], totals[1], len(deals),
                              True, winner)
    return match_id


class TestListe:
    async def test_ne_rend_que_les_parties_closes(self, clean_db):
        uid = await _user()
        done = await _match(uid)
        open_id = await db.create_match("play", 2000, user_id=uid)
        listed = await db.list_matches(uid)
        assert [m["id"] for m in listed] == [done]
        # Et l'autre liste dit exactement l'inverse : les deux se complètent
        # sans se recouvrir.
        assert [m["id"] for m in await db.list_open_matches(uid)] == [open_id]

    async def test_une_partie_abandonnee_est_un_resultat(self, clean_db):
        """L'Elo la compte comme une défaite (`elo._losing_team`) : la cacher
        ferait deux comptes différents de la même chose."""
        uid = await _user()
        match_id = await _match(uid, abandoned=True)
        listed = await db.list_matches(uid)
        assert len(listed) == 1
        assert listed[0]["id"] == match_id
        assert listed[0]["abandoned"] is True
        assert listed[0]["winner"] is None

    async def test_le_siege_vient_de_human_seat_en_solo(self, clean_db):
        uid = await _user()
        await _match(uid, seat=1)
        assert (await db.list_matches(uid))[0]["user_seat"] == 1

    async def test_un_invite_de_salon_voit_la_partie_de_l_hote(self, clean_db):
        """`matches.user_id` ne désigne que l'hôte : sans la seconde branche
        d'appartenance, les trois autres joueurs ne verraient rien."""
        host, guest = await _user("alice"), await _user("bob")
        match_id = await db.create_match("multi", 2000, user_id=host)
        game_id = await _deal(match_id, 1, mode="multi")
        await db.add_game_player(game_id, 0, host)
        await db.add_game_player(game_id, 3, guest)
        await db.update_match(match_id, 2010, 900, 1, True, 0)

        assert [m["id"] for m in await db.list_matches(guest)] == [match_id]
        # Et son siège se lit sur `game_players`, pas sur `matches.human_seat`
        # (NULL en salon) : Ouest est en Est-Ouest, donc il a perdu.
        assert (await db.list_matches(guest))[0]["user_seat"] == 3

    async def test_non_notee_rend_None_et_pas_zero(self, clean_db):
        """« Non classée » et « notée 0 » sont deux choses différentes."""
        uid = await _user()
        await _match(uid, target=1000)
        assert (await db.list_matches(uid))[0]["elo_delta"] is None

    async def test_ordre_et_isolation(self, clean_db):
        alice, bob = await _user("alice"), await _user("bob")
        first = await _match(alice)
        second = await _match(alice)
        await _match(bob)
        conn = await db.get_db()
        await conn.execute("UPDATE matches SET created_at = '2020-01-01' WHERE id = ?",
                           (first,))
        await conn.commit()
        assert [m["id"] for m in await db.list_matches(alice)] == [second, first]


class TestFeuilleDeMarque:
    async def test_le_cumul_suit_les_points_marques(self, clean_db):
        uid = await _user()
        match_id = await _match(uid, deals=((160, 0), (0, 262), (250, 0)))
        sheet = await db.get_match(match_id)
        assert [g["deal_no"] for g in sheet["games"]] == [1, 2, 3]
        assert [(g["total_ns"], g["total_ew"]) for g in sheet["games"]] == [
            (160, 0), (160, 262), (410, 262)]
        # Et il retombe sur le total de la partie : c'est le cas normal.
        assert (sheet["sheet_ns"], sheet["sheet_ew"]) == (410, 262)
        assert (sheet["points_ns"], sheet["points_ew"]) == (410, 262)

    async def test_le_cumul_n_est_pas_les_points_cartes(self, clean_db):
        """Le piège que la migration v16 a fermé : sommer `points_ns/ew` donne
        des chiffres plausibles et faux (ici 92 par donne au lieu du score)."""
        uid = await _user()
        sheet = await db.get_match(await _match(uid, deals=((160, 0),)))
        g = sheet["games"][0]
        assert (g["points_ns"], g["points_ew"]) == (92, 60)
        assert (g["score_ns"], g["score_ew"]) == (160, 0)
        assert sheet["sheet_ns"] == 160

    async def test_le_preneur_est_un_siege(self, clean_db):
        """`games.contract` ne porte que le camp : en solo trois sièges sur
        quatre sont des bots, donc « Nord-Sud a pris » ne dit pas qui."""
        uid = await _user()
        sheet = await db.get_match(await _match(uid, deals=((160, 0),)))
        assert sheet["games"][0]["taker"] == 2
        assert sheet["games"][0]["contract"]["team"] == 0

    async def test_une_donne_en_plan_ne_figure_pas_sur_la_feuille(self, clean_db):
        uid = await _user()
        match_id = await _match(uid, deals=((160, 0),))
        await _deal(match_id, 2, user_id=uid, complete=False)
        sheet = await db.get_match(match_id)
        assert [g["deal_no"] for g in sheet["games"]] == [1]

    async def test_une_donne_en_quarantaine_sort_et_se_compte(self, clean_db):
        """Elle décrit une partie impossible : elle quitte la feuille, mais
        l'écart qu'elle creuse dans le cumul doit être explicable."""
        uid = await _user()
        match_id = await _match(uid, deals=((160, 0), (0, 262)))
        second = (await db.get_match(match_id))["games"][1]["id"]
        await db.mark_game_checked(second, "carte jouée deux fois")
        sheet = await db.get_match(match_id)
        assert [g["deal_no"] for g in sheet["games"]] == [1]
        assert sheet["invalid_deals"] == 1
        assert (sheet["sheet_ns"], sheet["sheet_ew"]) == (160, 0)
        assert (sheet["points_ns"], sheet["points_ew"]) == (160, 262)

    async def test_une_donne_sans_score_marque_est_comptee_a_part(self, clean_db):
        """Le rattrapage (`integrity.backfill_scores`) n'est pas passé : on ne
        l'additionne pas comme un zéro, on dit qu'elle manque."""
        uid = await _user()
        match_id = await db.create_match("play", 2000, user_id=uid, human_seat=2)
        await _deal(match_id, 1, user_id=uid, scores=(160, 0))
        await _deal(match_id, 2, user_id=uid, scores=None)
        await db.update_match(match_id, 160, 262, 2, True, 0)
        sheet = await db.get_match(match_id)
        assert sheet["unscored_deals"] == 1
        assert sheet["sheet_ns"] == 160
        assert sheet["games"][1]["score_ns"] is None
        assert sheet["games"][1]["total_ns"] == 160

    async def test_les_sieges_sont_nommes(self, clean_db):
        uid = await _user("alice")
        sheet = await db.get_match(await _match(uid, deals=((160, 0),)))
        assert sheet["seats"][2] == {"name": "alice", "bot": False}
        assert sheet["seats"][0] == {"name": "doudou", "bot": True}


class TestRoutes:
    @pytest.fixture
    def client(self, clean_db):
        with TestClient(server.app) as c:
            yield c

    def _login(self, client, name="alice"):
        r = client.post("/api/auth/register", json={"username": name, "password": PW})
        assert r.status_code == 200, r.text
        return r.json()["user"]["id"]

    def test_status_done_liste_les_parties_closes(self, client):
        from conftest import await_sync
        uid = self._login(client)
        match_id = await_sync(_match(uid))
        # Le défaut est resté « en cours » : deux appelants l'utilisaient avant
        # que le paramètre existe.
        assert client.get("/api/me/matches").json() == []
        rows = client.get("/api/me/matches?status=done").json()
        assert [m["id"] for m in rows] == [match_id]

    def test_l_historique_demande_un_compte(self, client):
        assert client.get("/api/me/matches?status=done").status_code == 401

    def test_la_feuille_d_une_partie_en_cours_est_introuvable(self, client):
        from conftest import await_sync
        uid = self._login(client)
        open_id = await_sync(db.create_match("play", 2000, user_id=uid))
        await_sync(_deal(open_id, 1, user_id=uid))
        assert client.get(f"/api/matches/{open_id}").status_code == 404

    def test_la_feuille_est_publique_mais_ne_fuit_pas_le_compte(self, client):
        """Comme `/api/games/{id}` : une partie close est partageable. Son
        propriétaire, lui, n'a pas à sortir d'ici."""
        from conftest import await_sync
        uid = self._login(client)
        match_id = await_sync(_match(uid))
        client.post("/api/auth/logout")
        blob = client.get(f"/api/matches/{match_id}").json()
        assert blob["target"] == 2000 and len(blob["games"]) == 2
        assert "user_id" not in blob
        assert "hands" not in blob["games"][0]

    def test_partie_inconnue(self, client):
        assert client.get("/api/matches/zzzz").status_code == 404
