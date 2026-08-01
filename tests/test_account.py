"""§2.1 — cycle de vie du compte : mot de passe, adresse, suppression.

Trois règles traversent `auth.py`, et ce sont elles qu'on teste ici plutôt que
les codes de retour un par un :

1. le formulaire d'oubli **ne dit jamais si un compte existe** ;
2. tout changement d'identifiant **révoque les autres sessions** ;
3. une opération sensible **redemande le mot de passe**, même connecté.
"""

import pytest

pytest.importorskip("httpx", reason="fastapi.testclient a besoin de httpx")

from fastapi.testclient import TestClient  # noqa: E402

import colver.web.auth as auth  # noqa: E402
import colver.web.database as db  # noqa: E402
import colver.web.mail as mail  # noqa: E402
import colver.web.server as server  # noqa: E402

from conftest import await_sync  # noqa: E402

PW = "motdepasse12"
PW2 = "unautremotdepasse"


@pytest.fixture
def sent(monkeypatch):
    """Intercepter les courriels — aucun SMTP n'est joignable en test."""
    box = []

    def _send(to, subject, body):
        box.append({"to": to, "subject": subject, "body": body})
        return True

    monkeypatch.setattr(mail, "send", _send)
    return box


@pytest.fixture
def client(clean_db):
    with TestClient(server.app) as c:
        yield c


def _register(client, username="alice", password=PW, email=None):
    payload = {"username": username, "password": password}
    if email is not None:
        payload["email"] = email
    r = client.post("/api/auth/register", json=payload)
    assert r.status_code == 200, r.text
    return r


def _reset_link(box):
    """Le lien contenu dans le dernier courriel."""
    import re
    m = re.search(r"https?://\S+token=(\S+)", box[-1]["body"])
    assert m, box[-1]["body"]
    return m.group(1)


class TestInscription:
    def test_adresse_facultative(self, client):
        _register(client)
        assert client.get("/api/me").json()["user"]["email"] is None

    def test_adresse_enregistree_et_normalisee(self, client):
        _register(client, email="  Alice@Example.COM ")
        assert client.get("/api/me").json()["user"]["email"] == "alice@example.com"

    def test_adresse_invalide_refusee(self, client):
        r = client.post("/api/auth/register",
                        json={"username": "bob", "password": PW, "email": "pasuneadresse"})
        assert r.status_code == 400
        assert client.get("/api/me").json()["user"] is None

    def test_adresse_deja_prise_ne_bloque_pas_l_inscription(self, client):
        """Le compte est créé, il lui manque son recours — on ne perd pas une
        inscription pour ça, l'intéressé corrigera depuis son compte."""
        _register(client, "alice", email="a@example.com")
        client.post("/api/auth/logout")
        r = client.post("/api/auth/register",
                        json={"username": "bob", "password": PW,
                              "email": "a@example.com"})
        assert r.status_code == 200
        assert r.json()["user"]["email"] is None


class TestChangerMotDePasse:
    def test_chemin_nominal(self, client):
        _register(client)
        r = client.post("/api/auth/password",
                        json={"current_password": PW, "new_password": PW2})
        assert r.status_code == 200
        client.post("/api/auth/logout")
        assert client.post("/api/auth/login",
                           json={"username": "alice", "password": PW}).status_code == 401
        assert client.post("/api/auth/login",
                           json={"username": "alice", "password": PW2}).status_code == 200

    def test_l_actuel_est_exige(self, client):
        _register(client)
        r = client.post("/api/auth/password",
                        json={"current_password": "faux", "new_password": PW2})
        assert r.status_code == 403
        # …et rien n'a changé
        client.post("/api/auth/logout")
        assert client.post("/api/auth/login",
                           json={"username": "alice", "password": PW}).status_code == 200

    def test_un_nouveau_trop_court_est_refuse(self, client):
        _register(client)
        r = client.post("/api/auth/password",
                        json={"current_password": PW, "new_password": "court"})
        assert r.status_code == 400

    def test_anonyme_refuse(self, client):
        assert client.post("/api/auth/password",
                           json={"current_password": PW,
                                 "new_password": PW2}).status_code == 401

    def test_les_autres_sessions_tombent_la_sienne_survit(self, client, clean_db):
        """Une session ouverte ailleurs est exactement ce dont on veut se
        débarrasser en changeant de mot de passe."""
        _register(client)
        with TestClient(server.app) as other:
            assert other.post("/api/auth/login",
                              json={"username": "alice", "password": PW}).status_code == 200
            assert other.get("/api/me").json()["user"] is not None

            assert client.post("/api/auth/password",
                               json={"current_password": PW,
                                     "new_password": PW2}).status_code == 200

            assert other.get("/api/me").json()["user"] is None   # révoquée
            assert client.get("/api/me").json()["user"] is not None  # la nôtre tient


class TestChangerAdresse:
    def test_poser_et_retirer(self, client):
        _register(client)
        r = client.post("/api/auth/email",
                        json={"password": PW, "email": "alice@example.com"})
        assert r.status_code == 200
        assert client.get("/api/me").json()["user"]["email"] == "alice@example.com"

        assert client.post("/api/auth/email",
                           json={"password": PW, "email": ""}).status_code == 200
        assert client.get("/api/me").json()["user"]["email"] is None

    def test_le_mot_de_passe_est_exige(self, client):
        _register(client)
        r = client.post("/api/auth/email",
                        json={"password": "faux", "email": "alice@example.com"})
        assert r.status_code == 403

    def test_adresse_deja_prise(self, client):
        _register(client, "alice", email="a@example.com")
        client.post("/api/auth/logout")
        _register(client, "bob")
        r = client.post("/api/auth/email",
                        json={"password": PW, "email": "a@example.com"})
        assert r.status_code == 409


class TestOubli:
    def test_lien_recu_puis_nouveau_mot_de_passe(self, client, sent):
        _register(client, "alice", email="alice@example.com")
        client.post("/api/auth/logout")

        assert client.post("/api/auth/forgot",
                           json={"identifier": "alice"}).status_code == 200
        assert len(sent) == 1 and sent[0]["to"] == "alice@example.com"

        token = _reset_link(sent)
        r = client.post("/api/auth/reset",
                        json={"token": token, "new_password": PW2})
        assert r.status_code == 200
        # On est reconnecté par la même réponse
        assert client.get("/api/me").json()["user"]["username"] == "alice"
        client.post("/api/auth/logout")
        assert client.post("/api/auth/login",
                           json={"username": "alice", "password": PW2}).status_code == 200

    def test_l_adresse_marche_aussi_comme_identifiant(self, client, sent):
        _register(client, "alice", email="alice@example.com")
        client.post("/api/auth/logout")
        client.post("/api/auth/forgot", json={"identifier": "alice@example.com"})
        assert len(sent) == 1

    def test_un_lien_ne_sert_qu_une_fois(self, client, sent):
        _register(client, "alice", email="alice@example.com")
        client.post("/api/auth/logout")
        client.post("/api/auth/forgot", json={"identifier": "alice"})
        token = _reset_link(sent)
        assert client.post("/api/auth/reset",
                           json={"token": token, "new_password": PW2}).status_code == 200
        assert client.post("/api/auth/reset",
                           json={"token": token, "new_password": "encoreautrechose"}
                           ).status_code == 400

    def test_une_nouvelle_demande_annule_la_precedente(self, client, sent):
        """Sinon chaque demande laisse une clé de plus derrière elle."""
        _register(client, "alice", email="alice@example.com")
        client.post("/api/auth/logout")
        client.post("/api/auth/forgot", json={"identifier": "alice"})
        first = _reset_link(sent)
        client.post("/api/auth/forgot", json={"identifier": "alice"})
        second = _reset_link(sent)
        assert first != second
        assert client.post("/api/auth/reset",
                           json={"token": first, "new_password": PW2}).status_code == 400
        assert client.post("/api/auth/reset",
                           json={"token": second, "new_password": PW2}).status_code == 200

    def test_un_jeton_invente_est_refuse(self, client):
        r = client.post("/api/auth/reset",
                        json={"token": "n-importe-quoi", "new_password": PW2})
        assert r.status_code == 400

    def test_toutes_les_sessions_tombent(self, client, clean_db, sent):
        """Quelqu'un a peut-être pris le compte : c'est le moment de le mettre
        dehors partout."""
        _register(client, "alice", email="alice@example.com")
        with TestClient(server.app) as other:
            other.post("/api/auth/login", json={"username": "alice", "password": PW})
            assert other.get("/api/me").json()["user"] is not None
            client.post("/api/auth/logout")
            client.post("/api/auth/forgot", json={"identifier": "alice"})
            client.post("/api/auth/reset",
                        json={"token": _reset_link(sent), "new_password": PW2})
            assert other.get("/api/me").json()["user"] is None

    @pytest.mark.parametrize("identifier", ["inconnu", "rien@example.com", ""])
    def test_ne_dit_jamais_si_un_compte_existe(self, client, sent, identifier):
        """La réponse doit être identique pour un pseudo connu et un inconnu,
        sinon ce formulaire public devient un annuaire."""
        _register(client, "alice", email="alice@example.com")
        client.post("/api/auth/logout")
        known = client.post("/api/auth/forgot", json={"identifier": "alice"})
        unknown = client.post("/api/auth/forgot", json={"identifier": identifier})
        assert unknown.status_code == known.status_code
        assert unknown.json() == known.json()

    def test_un_compte_sans_adresse_repond_pareil(self, client, sent):
        _register(client, "alice")   # sans adresse
        client.post("/api/auth/logout")
        r = client.post("/api/auth/forgot", json={"identifier": "alice"})
        assert r.status_code == 200 and r.json()["ok"] is True
        assert sent == []            # …mais rien n'est parti

    def test_sans_smtp_la_demande_aboutit_quand_meme(self, client, monkeypatch):
        """`mail.send` journalise le lien au lieu de l'envoyer : la
        réinitialisation reste utilisable de bout en bout en développement."""
        monkeypatch.setattr(mail, "SMTP_HOST", "")
        _register(client, "alice", email="alice@example.com")
        client.post("/api/auth/logout")
        assert client.post("/api/auth/forgot",
                           json={"identifier": "alice"}).status_code == 200


class TestSuppression:
    async def _uid(self, name="alice"):
        return (await db.get_user_by_username(name))["id"]

    def test_le_compte_part_et_les_donnes_restent(self, client, played_deal):
        """Une donne de salon appartient à quatre joueurs : l'effacer prendrait
        la partie des trois autres avec elle."""
        _register(client)
        uid = await_sync(db.get_user_by_username("alice"))["id"]
        hands, actions = played_deal(seed=1)
        game_id = await_sync(db.create_game(
            "play", 0, hands, {"0": "doudou"}, human_seat=2, user_id=uid))
        await_sync(db.complete_game(game_id, 80, 82, None))

        r = client.post("/api/account/delete",
                        json={"password": PW, "confirm": "alice"})
        assert r.status_code == 200
        assert client.get("/api/me").json()["user"] is None
        assert await_sync(db.get_user_by_username("alice")) is None

        row = await_sync(db.get_game(game_id))
        assert row is not None, "la donne a disparu avec le compte"
        assert row["user_id"] is None, "la donne est restée rattachée au compte"

    def test_le_mot_de_passe_est_exige(self, client):
        r = client.post("/api/account/delete",
                        json={"password": "faux", "confirm": "alice"})
        assert r.status_code == 401     # pas connecté
        _register(client)
        r = client.post("/api/account/delete",
                        json={"password": "faux", "confirm": "alice"})
        assert r.status_code == 403
        assert client.get("/api/me").json()["user"] is not None

    def test_le_pseudo_doit_etre_confirme(self, client):
        """Garde-fou contre le clic malheureux : c'est irréversible."""
        _register(client)
        r = client.post("/api/account/delete",
                        json={"password": PW, "confirm": "bob"})
        assert r.status_code == 400
        assert client.get("/api/me").json()["user"] is not None

    def test_le_pseudo_redevient_disponible(self, client):
        _register(client)
        client.post("/api/account/delete", json={"password": PW, "confirm": "alice"})
        assert client.post("/api/auth/register",
                           json={"username": "alice", "password": PW2}).status_code == 200


class TestCorpsDuCourriel:
    def test_ne_prejuge_pas_de_qui_a_demande(self):
        """Le formulaire est public : une adresse peut recevoir ce message sans
        que son propriétaire ait rien fait."""
        subject, body = mail.reset_email("alice", "https://colver.net/x?token=t", 2)
        assert "alice" in body
        assert "token=t" in body
        assert "2 h" in body
        assert "n'avez rien demandé" in body
        assert "colver" in subject.lower()

    def test_forgot_ne_met_pas_le_mot_de_passe_dans_le_message(self):
        _, body = mail.reset_email("alice", "https://colver.net/x?token=t", 2)
        assert PW not in body


class TestRateLimit:
    def test_les_tentatives_sont_plafonnees(self, client, monkeypatch):
        """Le budget protège les bcrypt ; un échec le consomme, un succès le
        rend (cf. `_refund`)."""
        _register(client)
        codes = [client.post("/api/auth/password",
                             json={"current_password": "faux", "new_password": PW2}
                             ).status_code for _ in range(10)]
        assert 429 in codes, codes
        # Le plafond est partagé avec login/register : on remet le compteur à
        # zéro pour ne pas contaminer les tests suivants du même processus.
        auth._AUTH_LIMITER.refund("testclient")
