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
        assert out["expected"] == 34       # (5000 - 1600) / 100

    def test_une_esperance_negative_reste_negative(self):
        """Le cas qui juge v6 : une annonce qu'il recommande peut être perdante.
        Observé à −346 sur une vraie donne."""
        out = server._quick_bid_readout(self._stats(
            ns_contracts=80, ns_achieved=4, pts_n=100,
            pts_ns_sum=100.0, pts_ew_sum=34_700.0))
        assert out["made_pct"] == 5
        assert out["expected"] == -346

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


class TestBudget:
    def test_le_budget_est_reglable_et_raisonnable(self):
        """160 donnes ≈ 4 s. Assez pour séparer deux annonces, assez peu pour
        qu'un bouton réponde — et l'intervalle à 95 % vaut ±8 points, donc
        l'écran n'affiche pas de décimale."""
        assert 50 <= server.QUICK_BID_SIMS <= 500
