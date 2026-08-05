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

import collections
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


class TestFraicheur:
    """Joignable ne veut pas dire à jour.

    Le sidecar se déploie **à la main**, séparément du webhook. Le 2026-08-03 un
    commit titré « feat(elo) » a livré la contrainte belote dans le sampler
    playgen sans le dire, et la prod a tourné 21 h sur un sidecar périmé : il
    fabriquait des mondes que `retain_valid` rejetait ensuite (~15,4 % aux
    positions à belote), donc Dédé cherchait sur moins de mondes qu'il n'en
    demandait. Rien ne le disait — `/health` le voyait joignable, et il l'était.
    """

    def _probe_with_surface(self, monkeypatch, remote, ours):
        class _Resp:
            def read(self):
                body = '{"status":"ok","model":"384","max_worlds":512'
                if remote is not None:
                    body += f',"surface":"{remote}"'
                return (body + "}").encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(playgen_gpu, "GPU_URL", "http://sidecar:8003")
        monkeypatch.setattr(playgen_gpu, "_OUR_SURFACE", ours)
        monkeypatch.setattr(playgen_gpu.urllib.request, "urlopen",
                            lambda *a, **k: _Resp())
        return playgen_gpu.probe()

    def test_memes_sources_est_frais(self, monkeypatch):
        p = self._probe_with_surface(monkeypatch, "abc123", "abc123")
        assert p["fresh"] is True

    def test_sources_differentes_est_perime(self, monkeypatch):
        """Le cas du 2026-08-03, cette fois visible."""
        p = self._probe_with_surface(monkeypatch, "vieux1", "neuf99")
        assert p["fresh"] is False
        # …et le message doit dire quoi faire, pas seulement que c'est faux.
        assert "vieux1" in p["surface"] and "neuf99" in p["surface"]

    def test_sidecar_muet_est_inconnu_pas_perime(self, monkeypatch):
        """Un sidecar d'avant cette fonctionnalité ne publie pas de `surface`.
        Le déclarer périmé apprendrait à ignorer le champ — une alerte qui crie
        au loup ne se lit plus, exactement comme les 14 fausses alertes de
        `_note_degraded` sur les coups forcés."""
        p = self._probe_with_surface(monkeypatch, None, "neuf99")
        assert p["fresh"] is None

    def test_binding_muet_est_inconnu_aussi(self, monkeypatch):
        p = self._probe_with_surface(monkeypatch, "abc123", None)
        assert p["fresh"] is None

    def test_injoignable_ne_conclut_rien(self, monkeypatch):
        """Injoignable est déjà signalé par `reachable`. Le compter aussi comme
        périmé ferait crier deux fois la même panne."""
        monkeypatch.setattr(playgen_gpu, "GPU_URL", "http://127.0.0.1:1")
        monkeypatch.setattr(playgen_gpu, "PROBE_TIMEOUT", 0.5)
        p = playgen_gpu.probe()
        assert p["reachable"] is False
        assert p["fresh"] is None


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

    def _sidecar_replying(self, monkeypatch, surface):
        class _Resp:
            def read(self):
                return (
                    '{"status":"ok","model":"384","max_worlds":512,'
                    f'"surface":"{surface}"}}'
                ).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(playgen_gpu, "GPU_URL", "http://sidecar:8003")
        monkeypatch.setattr(playgen_gpu.urllib.request, "urlopen",
                            lambda *a, **k: _Resp())

    def test_sidecar_exige_et_perime_degrade(self, client, monkeypatch):
        """Joignable, donc l'ancien /health disait « ok » — c'est précisément le
        silence de 21 h du 2026-08-03."""
        self._sidecar_replying(monkeypatch, "vieux1")
        monkeypatch.setattr(playgen_gpu, "_OUR_SURFACE", "neuf99")
        monkeypatch.setattr(agents, "REQUIRE_SIDECAR", True)
        r = client.get("/health")
        body = r.json()
        assert body["sidecar"]["reachable"] is True
        assert body["sidecar"]["fresh"] is False
        assert body["status"] == "degraded"
        # …mais on sert : un sidecar périmé affaiblit le jeu, il ne l'empêche pas.
        assert r.status_code == 200

    def test_perime_mais_non_exige_ne_degrade_pas(self, client, monkeypatch):
        """Sur une machine de dev, un sidecar plus vieux que le checkout est la
        règle et non une panne. Même arbitrage que la joignabilité."""
        self._sidecar_replying(monkeypatch, "vieux1")
        monkeypatch.setattr(playgen_gpu, "_OUR_SURFACE", "neuf99")
        monkeypatch.setattr(agents, "REQUIRE_SIDECAR", False)
        body = client.get("/health").json()
        assert body["sidecar"]["fresh"] is False
        assert body["status"] == "ok"

    def test_a_jour_et_exige_reste_ok(self, client, monkeypatch):
        self._sidecar_replying(monkeypatch, "pareil")
        monkeypatch.setattr(playgen_gpu, "_OUR_SURFACE", "pareil")
        monkeypatch.setattr(agents, "REQUIRE_SIDECAR", True)
        body = client.get("/health").json()
        assert body["sidecar"]["fresh"] is True
        assert body["status"] == "ok"

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


class TestToutesLesDecisionsComptent:
    """Une décision IS-DD compte, **d'où qu'elle vienne**.

    Les compteurs vivaient dans `AgentTable.decide`, donc seul le jeu réel y
    entrait. Or `agent_review` et `card_analysis` construisent leurs
    `colver.Agent` à la main : une revue, c'est ~20 recherches, et `/health`
    publiait un bloc `worlds` entièrement à zéro juste après. Constaté sur la
    prod le 2026-08-06. Une jauge qui sous-rapporte est pire qu'aucune : elle
    rassure.

    Mais les deux compteurs ne répondent pas à la même question, et c'est ce que
    `window` sépare — voir `note_decision`.
    """

    @pytest.fixture(autouse=True)
    def reset(self, monkeypatch):
        monkeypatch.setattr(
            agents, "_WORLD_STATS", dict.fromkeys(agents._WORLD_STATS, 0))
        monkeypatch.setattr(agents, "_RECENT_WORLDS", collections.deque(maxlen=200))

    @staticmethod
    def _isdd(injected=64, dets=64):
        return {"source": "isdd", "action": 0, "candidates": [],
                "determinizations": dets,
                "worlds": {"injected": injected, "playgen": 0,
                           "belief": 0, "uniform": 0}}

    def test_une_decision_de_jeu_alimente_les_deux(self):
        agents.note_decision("dede", self._isdd())
        assert agents.world_stats()["worlds_injected"] == 64
        assert agents.recent_worlds_per_decision()["n"] == 1

    def test_une_decision_d_analyse_compte_sans_polluer_la_jauge(self):
        """C'est le cœur : l'origine des mondes est budget-indépendante, la
        fenêtre ne l'est pas. Une revue à 500 ms tirerait la moyenne vers le bas
        dès qu'un joueur ouvre Rejouer, et ça se lirait comme une pression GPU."""
        agents.note_decision("dede", self._isdd(injected=80, dets=80), window=False)
        s = agents.world_stats()
        assert (s["decisions"], s["sampled"], s["all_playgen"]) == (1, 1, 1)
        assert s["worlds_injected"] == 80
        assert agents.recent_worlds_per_decision() is None

    def test_une_decision_qui_n_est_pas_is_dd_est_ignoree(self):
        """DouDou50 traverse le même point d'appel dans `card_analysis`."""
        assert agents.note_decision("doudou", {"source": "dmc", "action": 3}) is None
        assert agents.note_decision("dede", None) is None
        assert agents.world_stats()["decisions"] == 0

    def test_la_revue_d_agents_passe_bien_par_le_compteur(self, monkeypatch):
        """Le lien qui manquait. Testé au point d'appel réel, pas sur un double
        de `note_decision` : c'est l'oubli de l'appel qui était le défaut."""
        import colver.web.agent_review as review

        class _Agent:
            def decide(self, env):
                return TestToutesLesDecisionsComptent._isdd(injected=48, dets=48) | {
                    "action": 7, "candidates": [(7, 1.0), (9, 0.0)]}

        card, cost = review._ask_isdd(_Agent(), None, 0, 7)
        assert card == 7 and cost == 0.0
        assert agents.world_stats()["worlds_injected"] == 48
        assert agents.recent_worlds_per_decision() is None  # hors jauge
