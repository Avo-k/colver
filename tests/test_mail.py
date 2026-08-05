"""Un courriel « envoyé » ne l'est pas forcément.

Panne du 2026-08-05, restée invisible trois jours : la configuration SMTP était
bonne, l'authentification passait, le relais répondait `250 OK queued` — et
Mailjet jetait chaque message, l'expéditeur `no-reply@colver.net` n'étant pas
validé sur le compte. `mail.send` journalisait « courriel envoyé », `/health` ne
disait rien du courriel, donc **aucune trace côté colver** ne distinguait ça
d'une remise réussie.

Les deux garde-fous testés ici sont ce qui rend la même panne lisible la
prochaine fois : la réponse du relais dans le journal (elle porte l'identifiant
de file avec lequel on interroge le relais), et le bloc `mail` de `/health`.
"""

import logging

import pytest

pytest.importorskip("httpx", reason="fastapi.testclient a besoin de httpx")

from fastapi.testclient import TestClient  # noqa: E402

import colver.web.mail as mail  # noqa: E402
import colver.web.server as server  # noqa: E402


@pytest.fixture
def client(clean_db):
    with TestClient(server.app) as c:
        yield c


class TestReponseDuRelais:
    """`sendmail` jette la réponse au `DATA` ; `_Traced` la retient."""

    def _traced(self, reply):
        class _Parent:
            def data(self, msg):
                return 250, reply

        class _Traced(mail._Traced, _Parent):
            pass

        t = _Traced()
        code, _ = t.data(b"")
        return t, code

    def test_identifiant_de_file_retenu(self):
        """C'est la seule prise pour retrouver le message chez le relais."""
        t, code = self._traced(b"OK queued as d319c1a0-a681-4cc5-9f21-8b347c73caf8")
        assert code == 250
        assert t.data_reply == "OK queued as d319c1a0-a681-4cc5-9f21-8b347c73caf8"

    def test_reponse_non_ascii_ne_leve_pas(self):
        """Un relais peut répondre ce qu'il veut ; le journal ne doit pas
        devenir le point de panne d'un envoi qui, lui, a réussi."""
        t, _ = self._traced(b"OK queu\xffed")
        assert t.data_reply.startswith("OK queu")

    def test_reponse_deja_decodee(self):
        t, _ = self._traced("250 OK")
        assert t.data_reply == "250 OK"


class TestJournal:
    def test_sans_smtp_le_lien_part_au_journal(self, caplog):
        """Le repli de développement : la réinitialisation reste testable de
        bout en bout, il faut juste aller lire le lien dans les logs."""
        with caplog.at_level(logging.WARNING, logger="colver.web.mail"):
            assert mail.send("a@example.com", "sujet", "corps https://x/?token=zz") is False
        assert "SMTP non configuré" in caplog.text
        assert "token=zz" in caplog.text

    def test_envoi_journalise_la_reponse_du_relais(self, monkeypatch, caplog):
        """Le message ne dit pas « envoyé » — le protocole s'arrête au relais,
        et c'est précisément la confusion qui a masqué la panne."""
        sent = {}

        class _FakeSMTP:
            data_reply = "OK queued as abc-123"

            def __init__(self, host, port, timeout=None):
                sent["host"] = host

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def starttls(self, context=None):
                pass

            def login(self, user, password):
                sent["login"] = user

            def send_message(self, msg):
                sent["from"] = msg["From"]
                return {}

        monkeypatch.setattr(mail, "SMTP_HOST", "relais.example.com")
        monkeypatch.setattr(mail, "SMTP_USER", "cle")
        monkeypatch.setattr(mail, "_SMTP", _FakeSMTP)
        with caplog.at_level(logging.INFO, logger="colver.web.mail"):
            assert mail.send("a@example.com", "sujet", "corps") is True
        assert "abc-123" in caplog.text
        assert "accepté par le relais" in caplog.text
        assert "envoyé" not in caplog.text

    def test_un_relais_qui_refuse_ne_leve_jamais(self, monkeypatch, caplog):
        """`auth.forgot` répond la même chose dans tous les cas : une panne
        d'envoi qui deviendrait une 500 dirait au visiteur que l'adresse
        existe."""
        class _Boom:
            def __init__(self, *a, **kw):
                raise OSError("relais injoignable")

        monkeypatch.setattr(mail, "SMTP_HOST", "relais.example.com")
        monkeypatch.setattr(mail, "_SMTP", _Boom)
        with caplog.at_level(logging.ERROR, logger="colver.web.mail"):
            assert mail.send("a@example.com", "sujet", "corps") is False
        assert "en échec" in caplog.text


class TestStatus:
    def test_sans_smtp(self, monkeypatch):
        monkeypatch.setattr(mail, "SMTP_HOST", "")
        s = mail.status()
        assert s == {"configured": False, "host": None, "sender": None}

    def test_avec_smtp(self, monkeypatch):
        monkeypatch.setattr(mail, "SMTP_HOST", "relais.example.com")
        monkeypatch.setattr(mail, "MAIL_FROM", "colver <no-reply@colver.net>")
        s = mail.status()
        assert s["configured"] is True
        assert s["host"] == "relais.example.com"
        assert s["sender"] == "colver <no-reply@colver.net>"

    def test_aucun_secret_publie(self, monkeypatch):
        """`/health` est public."""
        monkeypatch.setattr(mail, "SMTP_HOST", "relais.example.com")
        monkeypatch.setattr(mail, "SMTP_USER", "cle-publique")
        monkeypatch.setattr(mail, "SMTP_PASSWORD", "secret-a-ne-pas-fuiter")
        blob = repr(mail.status())
        assert "secret-a-ne-pas-fuiter" not in blob
        assert "cle-publique" not in blob


class TestHealth:
    def test_bloc_mail_publie(self, client, monkeypatch):
        monkeypatch.setattr(mail, "SMTP_HOST", "relais.example.com")
        body = client.get("/health").json()
        assert body["mail"]["configured"] is True
        assert body["mail"]["host"] == "relais.example.com"

    def test_sans_smtp_le_service_reste_ok(self, client, monkeypatch):
        """Pas de SMTP est un choix légitime (dev, déploiement sans compte).
        Dégrader ici apprendrait à ignorer le champ — même raisonnement que
        pour `sidecar.fresh is None`."""
        monkeypatch.setattr(mail, "SMTP_HOST", "")
        body = client.get("/health").json()
        assert body["mail"]["configured"] is False
        assert body["status"] == "ok"

    def test_comptes_avec_adresse_comptes(self, client, monkeypatch):
        """Ce qui rend `configured: false` lisible pour une supervision : c'est
        le nombre de joueurs à qui l'interface promet un recours."""
        monkeypatch.setattr(mail, "SMTP_HOST", "")
        assert client.get("/health").json()["mail"]["accounts_with_email"] == 0
        client.post("/api/auth/register",
                    json={"username": "alice", "password": "motdepasse12",
                          "email": "alice@example.com"})
        client.post("/api/auth/logout")
        client.post("/api/auth/register",
                    json={"username": "bob", "password": "motdepasse12"})
        body = client.get("/health").json()
        assert body["mail"]["accounts_with_email"] == 1
