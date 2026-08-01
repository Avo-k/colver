"""Le repli du sidecar playgen doit s'entendre.

Trois silences distincts, et c'est le premier qui a laissé la prod jouer plus
d'un jour sur des mondes contraints-uniformes :

1. **aucun sidecar configuré** — le spec disait `source = "uniform"` sans que
   rien ne le journalise ;
2. **configuré mais injoignable** — `/health` rapportait une *variable
   d'environnement*, pas l'état du serveur au bout ;
3. **une décision qui se replie** — `worlds_source` partait au client, jamais
   au journal, et personne ne regarde une interface à trois heures du matin.
"""

import logging

import pytest

pytest.importorskip("httpx", reason="fastapi.testclient a besoin de httpx")

from fastapi.testclient import TestClient  # noqa: E402

import colver.web.agents as agents  # noqa: E402
import colver.web.playgen_gpu as playgen_gpu  # noqa: E402
import colver.web.server as server  # noqa: E402


@pytest.fixture(autouse=True)
def no_probe_cache(monkeypatch):
    """La sonde est mise en cache 30 s : sans remise à zéro, le premier test
    imposerait son verdict aux suivants."""
    monkeypatch.setattr(playgen_gpu, "_probe_cache", None)


@pytest.fixture
def client(clean_db):
    with TestClient(server.app) as c:
        yield c


class TestProbe:
    def test_non_configure(self, monkeypatch):
        monkeypatch.setattr(playgen_gpu, "GPU_URL", "")
        p = playgen_gpu.probe()
        assert p["configured"] is False and p["reachable"] is False

    def test_injoignable_est_rapporte_avec_sa_raison(self, monkeypatch):
        """Port fermé sur le loopback : la sonde doit rendre `reachable: False`
        et *dire pourquoi*, pas se contenter d'un booléen."""
        monkeypatch.setattr(playgen_gpu, "GPU_URL", "http://127.0.0.1:1")
        monkeypatch.setattr(playgen_gpu, "PROBE_TIMEOUT", 0.5)
        p = playgen_gpu.probe()
        assert p["configured"] is True
        assert p["reachable"] is False
        assert p["detail"]

    def test_joignable(self, monkeypatch):
        class _Resp:
            def read(self):
                return b'{"status":"ok","model":"384","max_worlds":512}'

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(playgen_gpu, "GPU_URL", "http://sidecar:8003")
        monkeypatch.setattr(playgen_gpu.urllib.request, "urlopen",
                            lambda *a, **k: _Resp())
        p = playgen_gpu.probe()
        assert p["reachable"] is True
        assert "384" in p["detail"]

    def test_le_resultat_est_mis_en_cache(self, monkeypatch):
        calls = []

        def _boom(*a, **k):
            calls.append(1)
            raise OSError("injoignable")

        monkeypatch.setattr(playgen_gpu, "GPU_URL", "http://sidecar:8003")
        monkeypatch.setattr(playgen_gpu.urllib.request, "urlopen", _boom)
        playgen_gpu.probe()
        playgen_gpu.probe()
        playgen_gpu.probe()
        assert len(calls) == 1, "la sonde doit être mise en cache"
        assert playgen_gpu.probe(force=True) and len(calls) == 2


class TestHealth:
    def test_sans_sidecar_attendu_le_service_reste_ok(self, client, monkeypatch):
        """Une machine de dev sans GPU est un cas normal, pas une panne."""
        monkeypatch.setattr(playgen_gpu, "GPU_URL", "")
        monkeypatch.setattr(agents, "REQUIRE_SIDECAR", False)
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["sidecar"]["configured"] is False
        assert body["sidecar"]["required"] is False

    def test_sidecar_exige_et_absent_degrade(self, client, monkeypatch):
        """C'est exactement le cas qui est passé inaperçu en production."""
        monkeypatch.setattr(playgen_gpu, "GPU_URL", "")
        monkeypatch.setattr(agents, "REQUIRE_SIDECAR", True)
        r = client.get("/health")
        body = r.json()
        assert body["status"] == "degraded"
        assert body["sidecar"]["required"] is True
        # …mais le service répond : un sidecar absent affaiblit le jeu, il
        # n'empêche pas de jouer.
        assert r.status_code == 200

    def test_sidecar_exige_et_injoignable_degrade(self, client, monkeypatch):
        monkeypatch.setattr(playgen_gpu, "GPU_URL", "http://127.0.0.1:1")
        monkeypatch.setattr(playgen_gpu, "PROBE_TIMEOUT", 0.5)
        monkeypatch.setattr(agents, "REQUIRE_SIDECAR", True)
        body = client.get("/health").json()
        assert body["status"] == "degraded"
        assert body["sidecar"]["configured"] is True
        assert body["sidecar"]["reachable"] is False

    def test_une_url_configuree_ne_suffit_plus_a_dire_que_tout_va_bien(
            self, client, monkeypatch):
        """La régression à empêcher : `sidecar_configured` était calculé sur la
        seule variable d'environnement, donc il valait `true` alors que rien ne
        répondait au bout."""
        monkeypatch.setattr(playgen_gpu, "GPU_URL", "http://127.0.0.1:1")
        monkeypatch.setattr(playgen_gpu, "PROBE_TIMEOUT", 0.5)
        body = client.get("/health").json()
        assert body["models"]["sidecar_configured"] is True
        assert body["sidecar"]["reachable"] is False


class TestJournal:
    def test_absence_de_sidecar_dite_au_demarrage(self, monkeypatch, caplog):
        monkeypatch.setattr(agents, "SIDECAR_URL", "")
        monkeypatch.setattr(agents, "REQUIRE_SIDECAR", False)
        with caplog.at_level(logging.WARNING, logger="colver.web.agents"):
            agents.log_startup_state()
        assert any(r.levelno == logging.WARNING for r in caplog.records)
        assert "uniforme" in caplog.text.lower()

    def test_absence_alors_qu_il_est_exige_est_une_erreur(self, monkeypatch, caplog):
        monkeypatch.setattr(agents, "SIDECAR_URL", "")
        monkeypatch.setattr(agents, "REQUIRE_SIDECAR", True)
        with caplog.at_level(logging.WARNING, logger="colver.web.agents"):
            agents.log_startup_state()
        assert any(r.levelno == logging.ERROR for r in caplog.records)

    def test_sidecar_present_est_dit_sans_alarme(self, monkeypatch, caplog):
        monkeypatch.setattr(agents, "SIDECAR_URL", "http://sidecar:8003")
        with caplog.at_level(logging.INFO, logger="colver.web.agents"):
            agents.log_startup_state()
        assert caplog.records and all(r.levelno == logging.INFO
                                      for r in caplog.records)


class TestDecisionDegradee:
    @pytest.fixture(autouse=True)
    def reset_counter(self, monkeypatch):
        monkeypatch.setattr(agents, "_degraded", {"since": 0.0, "count": 0})

    def _stats(self, source):
        return {"worlds_source": source, "determinizations": 12}

    def test_une_decision_playgen_ne_dit_rien(self, monkeypatch, caplog):
        monkeypatch.setattr(agents, "SIDECAR_URL", "http://sidecar:8003")
        with caplog.at_level(logging.WARNING, logger="colver.web.agents"):
            agents._note_degraded(0, self._stats("playgen-gpu"))
        assert caplog.records == []

    def test_une_decision_repliee_se_journalise(self, monkeypatch, caplog):
        monkeypatch.setattr(agents, "SIDECAR_URL", "http://sidecar:8003")
        with caplog.at_level(logging.WARNING, logger="colver.web.agents"):
            agents._note_degraded(0, self._stats("cpu"))
        assert "dégradé" in caplog.text

    def test_sans_sidecar_configure_on_ne_se_plaint_pas(self, monkeypatch, caplog):
        """Uniforme est alors le mode nominal ; c'est le démarrage qui l'a dit,
        une ligne par coup n'apprendrait rien."""
        monkeypatch.setattr(agents, "SIDECAR_URL", "")
        with caplog.at_level(logging.WARNING, logger="colver.web.agents"):
            agents._note_degraded(0, self._stats("cpu"))
        assert caplog.records == []

    def test_le_journal_est_plafonne(self, monkeypatch, caplog):
        """~24 coups de bot par donne : sans plafond, une panne de sidecar
        noierait le journal au lieu de le renseigner."""
        monkeypatch.setattr(agents, "SIDECAR_URL", "http://sidecar:8003")
        with caplog.at_level(logging.WARNING, logger="colver.web.agents"):
            for _ in range(50):
                agents._note_degraded(0, self._stats("cpu"))
        assert len(caplog.records) == 1
        # …mais rien n'est perdu : le compte est reporté dans la ligne suivante.
        assert agents._degraded["count"] == 49
