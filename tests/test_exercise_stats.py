"""Le progrès sur « Compter les points » suit le compte (migration v19).

C'est la seule donnée du site qu'un vidage de cache détruisait sans recours :
une analyse se recalcule, une donne est en base, un record d'exercice n'existait
nulle part ailleurs. Ce qui se teste ici est ce qui rend un compteur partagé
entre appareils correct — l'incrément, et le fait que le client ne décide pas
de son propre record.
"""

import pytest
from fastapi.testclient import TestClient

import colver.web.database as db
from colver.web.server import app

from conftest import await_sync


@pytest.fixture
def client(clean_db):
    with TestClient(app) as c:
        yield c


def _register(client, username="alice"):
    resp = client.post("/api/auth/register",
                       json={"username": username, "password": "hunter2hunter2"})
    assert resp.status_code == 200, resp.text
    return resp


def _attempt(client, **over):
    body = {"exercise": "compter", "variant": "debutant|chrono",
            "delta": 0, "exact": True, "streak": 1}
    body.update(over)
    return client.post("/api/me/exercises", json=body)


class TestApi:
    def test_anonymous_gets_an_empty_answer_not_an_error(self, client):
        """La page marche sans compte : elle retombe sur son localStorage.

        Un 401 obligerait l'appelant à écrire une gestion d'erreur pour un cas
        qui n'en est pas un.
        """
        resp = client.get("/api/me/exercises")
        assert resp.status_code == 200
        assert resp.json() == {"stats": {}, "synced": False}

    def test_anonymous_cannot_record(self, client):
        assert _attempt(client).status_code == 401

    def test_attempts_accumulate(self, client):
        _register(client)
        _attempt(client, delta=0, exact=True, streak=1)
        _attempt(client, delta=6, exact=False, streak=0)
        body = _attempt(client, delta=4, exact=False, streak=0).json()
        s = body["stats"]["debutant|chrono"]
        assert s["plays"] == 3
        assert s["exact"] == 1
        assert s["sumAbsDelta"] == 10
        assert s["streak"] == 0
        assert s["best"] == 1

    def test_best_is_a_server_side_max(self, client):
        """Un record ne se déclare pas : il se constate.

        C'est la seule valeur de la table qu'un client aurait intérêt à pousser
        lui-même, et celle qu'un second onglet écraserait en envoyant la sienne.
        """
        _register(client)
        for n in (1, 2, 3):
            _attempt(client, delta=0, exact=True, streak=n)
        # Une série cassée ne doit pas emporter le record avec elle.
        body = _attempt(client, delta=8, exact=False, streak=0).json()
        s = body["stats"]["debutant|chrono"]
        assert s["streak"] == 0
        assert s["best"] == 3

    def test_variants_are_independent(self, client):
        _register(client)
        _attempt(client, variant="debutant|chrono", delta=0, exact=True, streak=1)
        _attempt(client, variant="expert|carte", delta=12, exact=False, streak=0)
        stats = client.get("/api/me/exercises").json()["stats"]
        assert stats["debutant|chrono"]["plays"] == 1
        assert stats["expert|carte"]["plays"] == 1
        assert stats["expert|carte"]["sumAbsDelta"] == 12

    def test_two_accounts_do_not_share(self, client):
        _register(client, "alice")
        _attempt(client, delta=0, exact=True, streak=4)
        client.post("/api/auth/logout")
        _register(client, "bob")
        assert client.get("/api/me/exercises").json()["stats"] == {}

    def test_unknown_exercise_is_refused(self, client):
        """`exercise` est une liste fermée : sinon la table est remplissable
        à volonté avec des clés inventées."""
        _register(client)
        assert _attempt(client, exercise="../../etc").status_code == 400
        assert client.get("/api/me/exercises?exercise=nimporte").status_code == 400

    def test_absurd_values_are_clamped_not_stored(self, client):
        """Les compteurs sont cumulés : une valeur du client y reste pour
        toujours."""
        _register(client)
        body = _attempt(client, delta=10 ** 9, exact=False, streak=10 ** 9).json()
        s = body["stats"]["debutant|chrono"]
        assert s["sumAbsDelta"] == 504
        assert s["best"] == 100_000

    def test_missing_variant_is_refused(self, client):
        _register(client)
        assert _attempt(client, variant="").status_code == 400

    def test_deleting_an_account_takes_its_progress(self, client):
        """Rien ne se rattache à ce progrès, contrairement à une donne de salon :
        il part entièrement."""
        _register(client)
        _attempt(client)
        me = client.get("/api/me").json()["user"]
        resp = client.post("/api/account/delete",
                           json={"password": "hunter2hunter2", "confirm": "alice"})
        assert resp.status_code == 200, resp.text
        rows = await_sync(db.get_db())
        left = await_sync(rows.execute_fetchall(
            "SELECT COUNT(*) FROM exercise_stats WHERE user_id = ?", (me["id"],)))
        assert left[0][0] == 0
