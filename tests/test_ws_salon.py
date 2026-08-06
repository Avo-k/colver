"""Une donne de salon jouée jusqu'au bout, sur le vrai protocole.

Le salon n'avait **aucun** test qui aille au terme d'une donne — et c'est
exactement là qu'il cassait : `Room._close_deal` appelait `self.bots`, qui
n'existe que sur la `PlaySession`. L'`AttributeError` remontait jusqu'à
`_drive`, dont le `except Exception` interrompait la partie « suite à une
erreur » à la fin de **chaque** donne, sans que rien ne soit écrit en base.

D'où les deux assertions qui discriminent, et pourquoi elles sont deux :
`room_error` dit que le pilote est tombé, `is_complete` dit que la donne a été
close. Une seule des deux laisserait passer la moitié du défaut.

Un seul humain à table, trois bots : le bogue est dans la clôture de donne, qui
ne dépend pas du nombre de joueurs. Deux sockets demanderaient d'entrelacer
deux lectures bloquantes — `TestClient` n'offre pas de réception non bloquante,
donc ce serait un harnais à threads pour aucune couverture supplémentaire ici.
"""

import pytest

pytest.importorskip("httpx", reason="fastapi.testclient a besoin de httpx")

from fastapi.testclient import TestClient  # noqa: E402

import colver.web.database as db  # noqa: E402
import colver.web.rooms as rooms  # noqa: E402
import colver.web.server as server  # noqa: E402

from conftest import await_sync  # noqa: E402
from test_ws_play import _register  # noqa: E402


@pytest.fixture
def client(clean_db, no_tempo):
    """Client HTTP+WS sur l'app réelle, salons vidés de part et d'autre.

    `rooms.ROOMS` et `rooms.USER_ROOM` sont des dictionnaires de module : un
    salon laissé par un test précédent survivrait à la base neuve, et
    `USER_ROOM` renverrait le test suivant vers un salon fantôme.
    """
    rooms.ROOMS.clear()
    rooms.USER_ROOM.clear()
    with TestClient(server.app) as c:
        try:
            yield c
        finally:
            # Couper les pilotes **depuis la boucle de l'app** (`portal.call`),
            # pas depuis le fil de test : `Task.cancel()` n'est pas sûr entre
            # fils, et une boucle endormie dans `select()` ne le verrait pas.
            #
            # Et ici, pas dans le corps des tests : un test qui échoue laisse
            # son salon vivant, la fermeture de la connexion aiosqlite l'attend,
            # et le teardown pend — donc **tout échec se présenterait comme un
            # blocage**, ce qui est la pire façon de rapporter un échec.
            c.portal.call(_stop_rooms)


async def _stop_rooms():
    for room in list(rooms.ROOMS.values()):
        room.stop()
    rooms.ROOMS.clear()
    rooms.USER_ROOM.clear()


class Salon:
    """Un client de la page Salon : hôte assis en Sud, trois bots.

    L'état diffusé est tourné pour que le spectateur soit toujours le siège
    d'affichage 2 (cf. `rooms.rotate_state`), donc « c'est à nous » se lit
    `current_player == 2` — comme le fait `GameTable` côté client.
    """

    MY_SEAT = 2

    def __init__(self, ws):
        self.ws = ws
        self.state = None
        self.game_id = None
        self.seen = []
        # Position à laquelle on a joué pour la dernière fois. Un pli complet
        # est diffusé **deux fois** (l'instantané, puis la table balayée), et
        # les deux images peuvent porter notre tour : sans ce repère on
        # renverrait la même carte, que le pilote rejetterait en `room_error`.
        self._played_at = None

    def until(self, predicate, limit=600):
        """Lire jusqu'à ce qu'un message satisfasse `predicate`.

        ⚠️ `room_error` interrompt immédiatement. Ce n'est pas du zèle : quand
        le pilote tombe, il diffuse cette erreur **puis se tait à jamais**, donc
        un harnais qui se contente d'attendre sa condition bloque au lieu
        d'échouer. C'est exactement ce qu'a fait la première version de ce
        fichier sur le code bogué — un test qui pend est un test qu'on
        désactive, pas un test qui alerte.
        """
        for _ in range(limit):
            msg = self.ws.receive_json()
            self.seen.append(msg)
            if msg.get("type") == "room_game_state":
                self.game_id = msg.get("game_id") or self.game_id
                self.state = msg["state"]
            if predicate(msg):
                return msg
            if msg.get("type") == "room_error":
                raise AssertionError(f"le pilote du salon a lâché : {msg['msg']}")
        raise AssertionError("message attendu jamais reçu")

    @property
    def errors(self):
        return [m for m in self.seen if m.get("type") == "room_error"]

    def create(self, mode="rapide", target=0):
        self.ws.send_json({"type": "room_create"})
        state = self.until(lambda m: m.get("type") == "room_state")
        self.ws.send_json({"type": "room_config", "mode": mode, "target": target})
        self.until(lambda m: m.get("type") == "room_state"
                   and m.get("target") == target)
        return state["code"]

    def start(self):
        self.ws.send_json({"type": "room_start"})

    def leave(self):
        """Quitter le salon — indispensable en fin de test, pas décoratif.

        Le pilote d'un salon est une tâche asyncio qui **survit à la fermeture
        de la socket** : c'est voulu, un joueur doit pouvoir se reconnecter à sa
        partie (`handle_disconnect` garde les salons en jeu). En test, elle
        attend alors une action humaine qui n'arrivera jamais, et l'arrêt de
        l'application l'attend à son tour — le teardown pend, sans qu'aucun
        test n'ait échoué. `room_leave` la coupe **depuis la boucle**, ce qu'un
        `task.cancel()` lancé du fil de test ne ferait pas proprement.
        """
        self.ws.send_json({"type": "room_leave"})
        self.until(lambda m: m.get("type") == "room_left")

    def _server_plays_for_us(self, state):
        """Positions où le pilote joue à notre place — il ne faut rien envoyer.

        Deux cas, et les rater se paie cher : l'action part dans
        `Room.action_queue`, **personne ne la consomme sur le moment**, et elle
        ressort au tour humain suivant, où elle est presque toujours illégale.
        C'est ce que faisait ce harnais une fois sur quatre.

        - **Passe forcé** (`only_pass_is_legal`) : le serveur passe lui-même
          plutôt que d'offrir un bouton unique. Côté client c'est « phase
          d'enchère, une seule action légale, et c'est PASSE » — l'interface
          réelle cache le panneau d'enchère dans cet état, pour la même raison.
        - **Dernier pli** (`in_last_trick`) : plus aucune décision, les deux
          pilotes le déroulent seuls sur une échéance. Se lit ici sur la main —
          une carte restante — et non sur `tricks_won`, que l'image d'un pli
          complet décrémente exprès.
        """
        if state["phase"] == 0:
            return list(state["legal_actions"]) == [0]
        return len(state["hands"][self.MY_SEAT]) == 1

    def _position(self, msg):
        """Repère monotone de la position, pour ne jouer qu'une fois par tour.

        La main rétrécit à chaque carte, l'historique s'allonge à chaque
        annonce : ensemble ils bougent à tous les coups, y compris les nôtres.
        """
        return (len(msg["state"]["hands"][self.MY_SEAT]),
                len(msg.get("bid_history") or []))

    def play_deal(self, limit=80):
        """Jouer jusqu'à l'état terminal réel de la donne. Rend ce message.

        ⚠️ Une donne jouée émet **deux** images terminales — l'instantané du
        dernier pli (`deal_end_hold`, que le client montre avant de recouvrir la
        table) puis l'état terminal réel — mais une donne morte sur quatre
        passes n'en émet qu'**une**. Un appelant qui rend la première et va
        lire la seconde attend donc pour toujours une fois sur N. On consomme
        le maintien ici, pour que la donne se termine toujours au même endroit.
        """
        for _ in range(limit):
            msg = self.until(lambda m: (
                m.get("type") == "room_game_state"
                and (m["state"]["is_terminal"]
                     or (m["state"]["current_player"] == self.MY_SEAT
                         and m["state"]["legal_actions"]
                         and self._position(m) != self._played_at))))
            if msg["state"]["is_terminal"]:
                if msg["state"].get("deal_end_hold"):
                    continue
                return msg
            if self._server_plays_for_us(msg["state"]):
                continue
            self._played_at = self._position(msg)
            self.ws.send_json({"type": "room_play",
                               "action": msg["state"]["legal_actions"][0]})
        raise AssertionError("la donne ne se termine pas")


class TestUneDonneDeSalon:
    def test_une_donne_va_jusqu_au_bout_et_est_close(self, client):
        """La régression : le pilote tombait dans `_close_deal`."""
        _register(client)
        with client.websocket_connect("/ws") as ws:
            s = Salon(ws)
            s.create(target=0)
            s.start()
            assert s.play_deal()["state"]["is_terminal"]
            assert not s.errors, s.errors
            s.leave()

        row = await_sync(db.get_game(s.game_id))
        assert row is not None, "la donne n'a jamais été close en base"
        assert row["is_complete"] is True

    def test_la_donne_de_salon_se_rejoue(self, client):
        """Même contrôle qu'en solo : ce que le pilote écrit doit se rejouer."""
        from colver.web import integrity
        _register(client)
        with client.websocket_connect("/ws") as ws:
            s = Salon(ws)
            s.create(target=0)
            s.start()
            s.play_deal()
            s.leave()
        row = await_sync(db.get_game(s.game_id))
        assert integrity.check_deal(row) is None

    def test_une_partie_enchaine_ses_donnes(self, client):
        """Le pilote reste garé sur `next_deal_requested` entre deux donnes.

        Ce chemin-là n'était pas testé non plus, et il passe par le même
        `_close_deal` — à ceci près qu'ici la partie doit *survivre* à la
        clôture au lieu de s'arrêter.
        """
        _register(client)
        with client.websocket_connect("/ws") as ws:
            s = Salon(ws)
            s.create(target=1000)
            s.start()
            end = s.play_deal()
            first = s.game_id
            assert not s.errors, s.errors
            # Entre deux donnes la partie reste en jeu, et seul l'hôte enchaîne.
            assert end["awaiting_next_deal"] is True
            assert end["is_host"] is True

            ws.send_json({"type": "room_next_deal"})
            s.until(lambda m: m.get("type") == "room_game_state"
                    and m.get("game_id") not in (None, first))
            assert s.game_id != first
            assert not s.errors, s.errors
            s.leave()

        assert await_sync(db.get_game(first))["is_complete"] is True
