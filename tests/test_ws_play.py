"""Un aller-retour complet sur le vrai protocole WebSocket.

`fastapi.testclient` suffit à piloter le protocole de bout en bout, sans serveur
à lancer : c'est l'outil qui manquait pour tester la couche la plus retorse du
site. Ces tests jouent une donne comme un client le ferait — couper, revenir,
reprendre, finir.

**Piège qui vaut pour tout test de ce protocole** : le premier `game_state`
d'une donne peut déjà être le tour du joueur (le donneur est tiré au sort), donc
un harnais qui lit « un message de plus pour attendre son tour » se bloque une
fois sur quatre, sur une donne sur laquelle le serveur, lui, attend. On lit donc
toujours jusqu'à une **condition**, jamais un nombre de messages.

Les bots sont ceux du serveur, sans poids chargés (cf. `conftest`) : ils
retombent sur les règles du moteur, ce qui est exactement ce qu'il faut ici — on
teste le protocole, pas la force de jeu.
"""

import pytest

pytest.importorskip("httpx", reason="fastapi.testclient a besoin de httpx")

from fastapi.testclient import TestClient  # noqa: E402

import colver.web.database as db  # noqa: E402
import colver.web.server as server  # noqa: E402

from conftest import await_sync  # noqa: E402


@pytest.fixture
def client(clean_db, no_tempo):
    """Un client HTTP+WS sur l'app réelle, sur une base neuve.

    Dépend de `clean_db` : `server` importe le module `database`, donc c'est la
    même connexion qui est remplacée pour lui.
    """
    with TestClient(server.app) as c:
        yield c


def _register(client, username="alice", password="motdepasse12"):
    r = client.post("/api/auth/register",
                    json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return r


class Table:
    """Un client de la page Jouer : lit les messages et suit l'état utile.

    `game_id` n'accompagne que le **premier** `game_state` d'une donne — les
    suivants, émis pendant le tour des bots, ne le répètent pas. Un harnais qui
    le relit sur le dernier message reçu tombe donc sur un `KeyError` une fois
    la main passée. On le mémorise ici, une fois pour toutes.
    """

    def __init__(self, ws, human_seat=2):
        self.ws = ws
        self.human_seat = human_seat
        self.game_id = None
        self.state = None
        self.seen = []   # tout ce qui est arrivé, pour les tests de fuite

    def until(self, predicate, limit=400):
        """Lire jusqu'à ce qu'un message satisfasse `predicate`. Rend le message.

        `limit` est un garde-fou de test : sans lui, une condition qui n'arrive
        jamais bloquerait la suite au lieu d'échouer.
        """
        for _ in range(limit):
            msg = self.ws.receive_json()
            self.seen.append(msg)
            if msg.get("game_id"):
                self.game_id = msg["game_id"]
            if isinstance(msg.get("state"), dict):
                self.state = msg["state"]
            if predicate(msg):
                return msg
        raise AssertionError("message attendu jamais reçu")

    def start(self, mode="rapide", target=0):
        self.ws.send_json({"type": "start_game", "mode": mode,
                           "human_seat": self.human_seat, "target": target})
        return self.wait_turn()

    def wait_turn(self):
        """Avancer jusqu'au tour du joueur — ou jusqu'à la fin de la donne.

        Jamais « un message de plus » : le donneur est tiré au sort, donc le
        premier `game_state` *peut* déjà être notre tour, et compter les
        messages bloquerait une fois sur quatre.
        """
        self.until(lambda m: (
            m.get("type") == "game_state"
            and isinstance(m.get("state"), dict)
            and not m["state"].get("deal_end_hold")
            and (m["state"]["is_terminal"]
                 or m["state"]["current_player"] == self.human_seat)))
        return self.state

    def play(self, action):
        self.ws.send_json({"type": "play", "human_seat": self.human_seat,
                           "action": action})


class TestUneDonneDeBoutEnBout:
    def test_jouer_une_donne_entiere(self, client):
        _register(client)
        with client.websocket_connect("/ws") as ws:
            t = Table(ws)
            state = t.start()
            assert t.game_id
            moves = 0
            while not state["is_terminal"] and moves < 45:
                t.play(state["legal_actions"][0])
                state = t.wait_turn()
                moves += 1
            assert state["is_terminal"]

        row = await_sync(db.get_game(t.game_id))
        assert row is not None and row["is_complete"] is True
        assert len(row["actions"]) > 0

    def test_la_donne_jouee_par_le_protocole_est_coherente(self, client):
        """Le meilleur contrôle de bout en bout qu'on ait : la donne écrite par
        une vraie session doit se rejouer. C'est le prédicat du §4.5, appliqué
        à ce que le serveur vient d'écrire lui-même."""
        from colver.web import integrity
        _register(client)
        with client.websocket_connect("/ws") as ws:
            t = Table(ws)
            state = t.start()
            moves = 0
            while not state["is_terminal"] and moves < 45:
                t.play(state["legal_actions"][0])
                state = t.wait_turn()
                moves += 1
        row = await_sync(db.get_game(t.game_id))
        assert integrity.check_deal(row) is None

    def test_les_mains_des_autres_ne_partent_pas_avant_la_fin(self, client):
        """Une donne en cours ne divulgue rien : le client ne reçoit que sa
        propre main, et `initial_hands` attend l'état terminal.

        Le solo envoyait `initial_hands` dès la distribution : le jeu des trois
        autres était dans la console pendant toute la donne."""
        _register(client)
        with client.websocket_connect("/ws") as ws:
            t = Table(ws)
            state = t.start()
            if state["is_terminal"]:
                pytest.skip("donne morte sur quatre passes")
            # Aucun des messages reçus jusqu'ici ne porte les quatre mains…
            assert all("initial_hands" not in m for m in t.seen)
            # …et l'état ne montre que la nôtre.
            hands = state["hands"]
            assert len(hands[2]) > 0
            assert all(hands[s] == [] for s in (0, 1, 3))

    def test_les_quatre_mains_arrivent_a_l_etat_terminal(self, client):
        """C'est là qu'elles se montrent : `showEndOfGameReview` en a besoin."""
        _register(client)
        with client.websocket_connect("/ws") as ws:
            t = Table(ws)
            state = t.start()
            moves = 0
            while not state["is_terminal"] and moves < 45:
                t.play(state["legal_actions"][0])
                state = t.wait_turn()
                moves += 1
            final = [m for m in t.seen if "initial_hands" in m]
            assert final, "les mains ne sont jamais rendues, même à la fin"
            assert sorted(c for h in final[-1]["initial_hands"] for c in h) \
                == list(range(32))


class TestCoupIllegal:
    """§4.5 : le gestionnaire `play` prenait `data["action"]` tel quel."""

    def test_une_carte_qu_on_n_a_pas_est_refusee(self, client):
        _register(client)
        with client.websocket_connect("/ws") as ws:
            t = Table(ws)
            state = t.start()
            if state["is_terminal"]:
                pytest.skip("donne morte sur quatre passes")
            legal = list(state["legal_actions"])
            illegal = next(a for a in range(43) if a not in legal)

            def journal():
                return len(await_sync(
                    db.get_game(t.game_id, include_incomplete=True))["actions"])

            before = journal()
            t.play(illegal)
            # Le serveur renvoie la position, inchangée — il ne tombe pas.
            echo = t.until(lambda m: m.get("type") == "game_state")
            assert echo["state"]["legal_actions"] == legal
            assert journal() == before, "un coup illégal a été écrit en base"

            # …et la socket est toujours vivante : le coup légal passe.
            t.play(legal[0])
            assert t.wait_turn() is not None
            assert journal() > before

    def test_une_action_absurde_ne_tue_pas_la_socket(self, client):
        _register(client)
        with client.websocket_connect("/ws") as ws:
            t = Table(ws)
            state = t.start()
            if state["is_terminal"]:
                pytest.skip("donne morte sur quatre passes")
            legal = list(state["legal_actions"])
            t.play("pas un entier")
            t.until(lambda m: m.get("type") == "game_state")
            t.play(legal[0])
            assert t.wait_turn() is not None


class TestCouperEtReprendre:
    def test_une_donne_coupee_se_retrouve_et_se_reprend(self, client):
        """Le cas signalé par un joueur : quitter la page en pleine donne puis
        revenir. Avant §2.7 la donne disparaissait — redonne gratuite."""
        _register(client)
        with client.websocket_connect("/ws") as ws:
            t = Table(ws)
            state = t.start()
            if state["is_terminal"]:
                pytest.skip("donne morte sur quatre passes")
            t.play(state["legal_actions"][0])
            t.wait_turn()
            game_id = t.game_id
            # Coupure brutale : on ferme la socket sans rien dire au serveur.

        with client.websocket_connect("/ws") as ws:
            t2 = Table(ws)
            t2.ws.send_json({"type": "play_status"})
            open_msg = t2.until(lambda m: m.get("type") == "play_open")
            assert open_msg["deal"] is not None
            assert open_msg["deal"]["game_id"] == game_id
            assert open_msg["deal"]["moves"] > 0

            t2.ws.send_json({"type": "resume_deal"})
            t2.until(lambda m: m.get("type") == "game_state")
            assert t2.game_id == game_id

    def test_renoncer_efface_la_donne(self, client):
        _register(client)
        with client.websocket_connect("/ws") as ws:
            t = Table(ws)
            if t.start()["is_terminal"]:
                pytest.skip("donne morte sur quatre passes")

        with client.websocket_connect("/ws") as ws:
            t2 = Table(ws)
            t2.ws.send_json({"type": "drop_deal"})
            assert t2.until(lambda m: m.get("type") == "play_open")["deal"] is None

    def test_un_anonyme_n_a_rien_a_reprendre(self, client):
        """Sans compte, aucune identité à laquelle rattacher la donne : c'est
        le trou que le §2.1 ferme, pas la reprise."""
        with client.websocket_connect("/ws") as ws:
            t = Table(ws)
            if t.start()["is_terminal"]:
                pytest.skip("donne morte sur quatre passes")

        with client.websocket_connect("/ws") as ws:
            t2 = Table(ws)
            t2.ws.send_json({"type": "play_status"})
            assert t2.until(lambda m: m.get("type") == "play_open")["deal"] is None


class TestSante:
    def test_health_repond(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["db"] is True
        assert body["invalid_deals"] == 0

