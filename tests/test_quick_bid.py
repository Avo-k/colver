"""« Analyse rapide » d'une annonce : les deux chiffres, et ce qu'ils comptent.

Le bouton de Rejouer rend un taux de réussite et une espérance, simulés sur des
donnes jouées. L'intérêt par rapport à la note existante (le Q de bid v6) est
qu'il **ne dépend d'aucun modèle de référence** : il note l'humain et v6 au même
barème. Encore faut-il que les deux chiffres comptent ce qu'ils prétendent, et
ils n'ont pas le même dénominateur — c'est ce que ces tests épinglent.
"""

from colver.web import server


class TestLecture:
    """`_quick_bid_readout` : deux chiffres, deux dénominateurs."""

    @staticmethod
    def _stats(**kw):
        base = {"ns_contracts": 0, "ns_achieved": 0,
                "pts_ns_sum": 0.0, "pts_ew_sum": 0.0, "pts_n": 0}
        base.update(kw)
        return base

    def test_le_taux_porte_sur_les_donnes_ou_le_camp_tient_le_contrat(self):
        """Forcer une annonce ne garantit pas de rester preneur : les
        adversaires peuvent surenchérir. Le taux de réussite est donc
        conditionnel, et `taker_pct` dit sur quelle part il porte."""
        out = server._quick_bid_readout(self._stats(
            ns_contracts=40, ns_achieved=10, pts_n=100,
            pts_ns_sum=1000.0, pts_ew_sum=200.0))
        assert out["made_pct"] == 25       # 10/40, pas 10/100
        assert out["taker_pct"] == 40      # N-S a gardé l'enchère 40 % du temps
        assert out["contracts"] == 40

    def test_l_esperance_porte_sur_toutes_les_simulations(self):
        """Y compris les donnes où le camp n'a pas eu le contrat, et les donnes
        passées : c'est ce qui juge la *décision*, pas seulement le contrat."""
        out = server._quick_bid_readout(self._stats(
            ns_contracts=10, ns_achieved=5, pts_n=100,
            pts_ns_sum=5000.0, pts_ew_sum=1600.0))
        assert out["expected"] == 30       # (5000 - 1600) / 100 = 34, à la dizaine

    def test_l_esperance_est_arrondie_a_la_dizaine(self):
        """Pas de la coquetterie : à `QUICK_BID_SIMS` simulations l'intervalle à
        95 % sur ce chiffre vaut ±54 points (mesuré 2026-08-06 : σ ≈ 340 sur
        l'écart de score, quelle que soit l'annonce). Une unité
        affichée doit rester au-dessus du bruit qu'elle cache, sans quoi le
        panneau invite à départager deux annonces sur 14 points de rien."""
        for raw, shown in [(34, 30), (35, 40), (-346, -350), (4, 0)]:
            out = server._quick_bid_readout(self._stats(
                pts_n=100, pts_ns_sum=float(raw * 100), pts_ew_sum=0.0))
            assert out["expected"] == shown

    def test_une_esperance_negative_reste_negative(self):
        """Le cas qui juge v6 : une annonce qu'il recommande peut être perdante.
        Observé à −346 sur une vraie donne."""
        out = server._quick_bid_readout(self._stats(
            ns_contracts=80, ns_achieved=4, pts_n=100,
            pts_ns_sum=100.0, pts_ew_sum=34_700.0))
        assert out["made_pct"] == 5
        assert out["expected"] == -350

    def test_aucun_contrat_ne_divise_pas_par_zero(self):
        """Une annonce systématiquement surenchérie : pas de taux à donner, et
        un `None` plutôt qu'un 0 qui se lirait « ça ne passe jamais »."""
        out = server._quick_bid_readout(self._stats(pts_n=50, pts_ns_sum=10.0))
        assert out["made_pct"] is None
        assert out["expected"] == 0

    def test_aucune_simulation_ne_divise_pas_par_zero(self):
        out = server._quick_bid_readout(self._stats())
        assert out["made_pct"] is None
        assert out["expected"] is None
        assert out["taker_pct"] is None


class TestDispersion:
    """La dispersion de l'écart de score, et ce qu'on a le droit d'en dire.

    Mesuré le 2026-08-06 (`scripts/analysis/quick_bid_spread.py`, 3 mains, 600
    sims par annonce, plus 400 paires appariées) : l'écart N-S − E-O a un écart
    type de 310 à 370 points pour une moyenne comprise entre −84 et −26. Cet
    écart type ne décrit pas l'annonce — il vaut ~340 pour toutes — parce que la
    distribution est à deux bosses et que la moyenne tombe dans le creux.
    **91,2 % de sa variance est expliquée par l'issue** (σ dans une case : 65 à
    125 points). D'où deux agrégats et
    aucun « ± σ » à l'écran : `pts_gap_sq_sum` pour l'incertitude *sur la
    moyenne*, et `outcomes` pour la dispersion réelle.
    """

    @staticmethod
    def _dd(team, achieved, ns, ew):
        return {"void": False, "trump": 0, "value": 80, "team": team,
                "coinche": 0, "achieved": achieved, "auction": [],
                "scores": [float(ns), float(ew)]}

    def _fold(self, *dds):
        cells, stats = server._doudou_new_cells(), server._doudou_new_stats()
        for dd in dds:
            server._doudou_accumulate(cells, stats, dd)
        return stats

    def test_chaque_issue_a_sa_case_et_son_signe(self):
        """Quatre cases, et toutes comptées dans le repère Nord-Sud − Est-Ouest :
        un contrat adverse réussi est donc un nombre négatif."""
        stats = self._fold(
            self._dd(0, True, 180, 52),     # N-S prend et passe → +128
            self._dd(0, False, 0, 242),     # N-S prend et chute → −242
            self._dd(1, True, 52, 180),     # E-O prend et passe → −128
            self._dd(1, False, 242, 0),     # E-O prend et chute → +242
        )
        assert stats["outcomes"]["ns_made"] == {"n": 1, "sum": 128.0}
        assert stats["outcomes"]["ns_set"] == {"n": 1, "sum": -242.0}
        assert stats["outcomes"]["ew_made"] == {"n": 1, "sum": -128.0}
        assert stats["outcomes"]["ew_set"] == {"n": 1, "sum": 242.0}

    def test_la_somme_des_cases_redonne_l_esperance(self):
        """Invariant de lecture : le chiffre-phare doit se déduire de la
        décomposition affichée sous lui, sinon les deux se contrediraient."""
        stats = self._fold(self._dd(0, True, 180, 52), self._dd(0, False, 0, 242),
                           self._dd(1, True, 52, 180))
        total = sum(b["sum"] for b in stats["outcomes"].values())
        assert total == stats["pts_ns_sum"] - stats["pts_ew_sum"]
        assert sum(b["n"] for b in stats["outcomes"].values()) == stats["pts_n"]

    def test_une_donne_passee_n_entre_dans_aucune_case(self):
        """Pas de contrat, donc pas d'issue — mais la donne compte comme nulle
        dans le chiffre-phare, dont le dénominateur n'est pas le même."""
        cells, stats = server._doudou_new_cells(), server._doudou_new_stats()
        server._doudou_accumulate(cells, stats, {"void": True})
        assert all(b["n"] == 0 for b in stats["outcomes"].values())
        assert stats["deal_draws"] == 1 and stats["pts_n"] == 0

    def test_l_intervalle_porte_sur_la_moyenne_pas_sur_les_donnes(self):
        """σ/√n, pas σ. Deux donnes à ±100 : la dispersion vaut 100 points mais
        l'incertitude sur leur moyenne vaut 1,96 × 100/√2 ≈ 139."""
        stats = self._fold(self._dd(0, True, 100, 0), self._dd(0, False, 0, 100))
        assert stats["pts_gap_sq_sum"] == 20_000.0
        assert abs(server.gap_ci95(stats) - 1.96 * (10_000 / 2) ** 0.5) < 1e-6

    def test_un_echantillon_trop_court_ne_rend_pas_d_intervalle(self):
        """Une seule donne, ou des donnes toutes identiques : `None` plutôt
        qu'un ±0 qui se lirait « mesure exacte »."""
        assert server.gap_ci95(self._fold(self._dd(0, True, 100, 0))) is None
        assert server.gap_ci95(self._fold(self._dd(0, True, 100, 0),
                                          self._dd(0, True, 100, 0))) is None

    def test_un_blob_d_avant_la_v2_du_cache_ne_ment_pas(self):
        """`pts_gap_sq_sum` absent (entrée écrite avant le bump) : pas
        d'intervalle inventé à partir de sommes qui n'existent pas."""
        assert server.gap_ci95({"pts_n": 100, "pts_ns_sum": 3400.0,
                                "pts_ew_sum": 0.0}) is None


class TestBudget:
    def test_le_budget_est_reglable_et_raisonnable(self):
        """160 donnes ≈ 4 s, assez peu pour qu'un bouton réponde.

        Assez pour le **taux** de réussite, dont l'intervalle à 95 % vaut ±8
        points — pas pour l'espérance, dont l'intervalle vaut ±54 points à ce
        budget. C'est pour ça qu'elle s'affiche arrondie à la dizaine et en
        retrait : sur ce panneau, le chiffre qui sépare deux annonces est le
        taux."""
        assert 50 <= server.QUICK_BID_SIMS <= 500
