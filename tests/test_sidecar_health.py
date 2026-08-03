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

    def _stats(self, source, worlds=None):
        """Stats d'une décision ayant réellement échantillonné des mondes.

        Le détail compte : la garde de `_note_degraded` lit `worlds`, pas
        `worlds_source`. Une décision sans mondes n'est pas dégradée, elle n'a
        simplement pas cherché.
        """
        if worlds is None:
            worlds = {"injected": 0, "playgen": 0, "belief": 0, "uniform": 12}
        return {"worlds_source": source, "determinizations": 12, "worlds": worlds}

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

    def test_un_coup_force_ne_declenche_pas_lalarme(self, monkeypatch, caplog):
        """Sur un coup forcé, `run_search` sort avant d'échantillonner et rend
        des compteurs à zéro. `decision_stats` étiquette alors `"cpu"` — ce qui
        ne dit rien du sidecar. Les 14 alertes présentes en prod le 2026-08-03
        étaient toutes de cette forme : l'alarme criait au loup."""
        monkeypatch.setattr(agents, "SIDECAR_URL", "http://sidecar:8003")
        forced = {"worlds_source": "cpu", "determinizations": 0,
                  "worlds": {"injected": 0, "playgen": 0, "belief": 0, "uniform": 0}}
        with caplog.at_level(logging.WARNING, logger="colver.web.agents"):
            agents._note_degraded(0, forced)
        assert caplog.records == []
        # …et elle n'est pas non plus comptée pour la ligne suivante.
        assert agents._degraded["count"] == 0

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


class TestCompteursMondes:
    """« La file playgen s'assèche-t-elle ? » doit avoir une réponse chiffrée.

    Les compteurs par décision existaient déjà côté Rust (`WorldCounts`) mais
    n'étaient agrégés nulle part : ils partaient au client et disparaissaient.
    Or c'est ce chiffre, et lui seul, qui dit si le belief net sert encore —
    puisqu'il n'est consulté que quand la file est vide.
    """

    @pytest.fixture(autouse=True)
    def reset(self, monkeypatch):
        # Objet de module : sans remise à zéro il fuit d'un test à l'autre.
        monkeypatch.setattr(
            agents, "_WORLD_STATS", dict.fromkeys(agents._WORLD_STATS, 0))

    def _decision(self, injected=0, belief=0, uniform=0, playgen=0):
        return {"worlds": {"injected": injected, "playgen": playgen,
                           "belief": belief, "uniform": uniform}}

    def test_une_decision_entierement_playgen(self):
        agents._note_worlds(self._decision(injected=64))
        s = agents.world_stats()
        assert (s["decisions"], s["sampled"], s["all_playgen"]) == (1, 1, 1)
        assert s["worlds_injected"] == 64
        assert s["partial"] == s["no_playgen"] == s["no_sampling"] == 0

    def test_une_file_qui_seche_en_cours_de_recherche(self):
        """Le cas qui décide du sort du belief net : playgen a fourni, puis
        s'est tu, et la recherche a fini sur des mondes de repli."""
        agents._note_worlds(self._decision(injected=40, belief=20, uniform=4))
        s = agents.world_stats()
        assert (s["partial"], s["all_playgen"], s["no_playgen"]) == (1, 0, 0)
        assert s["worlds_belief"] == 20
        assert s["worlds_uniform"] == 4

    def test_un_coup_force_ne_compte_pas_comme_echantillonne(self):
        agents._note_worlds(self._decision())
        s = agents.world_stats()
        assert (s["decisions"], s["no_sampling"], s["sampled"]) == (1, 1, 0)

    def test_world_stats_rend_une_copie(self):
        """Sinon un appelant de /health pourrait muter les compteurs vivants."""
        agents.world_stats()["decisions"] = 999
        assert agents.world_stats()["decisions"] == 0
