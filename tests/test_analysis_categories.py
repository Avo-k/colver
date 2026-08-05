"""Ce que « une erreur » veut dire depuis le bump v7 (2026-08-05).

Le coût d'une carte se lit en **score de donne** et non plus en points cartes.
Ce n'est pas un changement d'unité cosmétique : le score de donne est une
fonction **en escalier** des points cartes — plat sous le seuil du contrat,
marche de `4V` au seuil, pente 2 seulement dans un contrat normal tenu. Mesuré
sur 1205 décisions (`scripts/analysis/replay_error_scale.py scale`), l'ancienne
échelle notait « ✓ Bon coup » 32 coups qui coûtaient jusqu'à 1264 points, et
mettait en tête de liste des coups qui ne décidaient plus rien.

Ces tests épinglent les trois propriétés dont tout le reste dépend.
"""

import colver
import pytest

from colver.web import analysis


class TestCategorisation:
    """`_categorize` ne lit que le score de donne et la bascule."""

    def test_cout_nul_est_parfait(self):
        assert analysis._categorize(0, False) == "parfait"
        assert analysis._categorize(0, True) == "parfait"

    def test_sans_bascule_c_est_une_imprecision(self):
        assert analysis._categorize(2, False) == "imprecision"
        assert analysis._categorize(600, False) == "imprecision"

    def test_avec_bascule_c_est_une_faute_decisive(self):
        assert analysis._categorize(400, True) == "decisive"

    def test_le_cout_en_points_cartes_n_entre_plus_dans_la_decision(self):
        """L'ancienne échelle graduait à 4 / 14 / 29 points cartes. Un coup peut
        coûter 2 points cartes et renverser la donne, ou 42 et ne rien décider :
        la signature ne prend donc plus le coût en points cartes du tout."""
        import inspect
        params = list(inspect.signature(analysis._categorize).parameters)
        assert params == ["cost_score", "swing"]


class TestSolveScoresRendLesDeuxEchelles:
    """Le binding rend les deux lectures du **même** solve, plus la bascule."""

    @staticmethod
    def _position():
        """Une donne au premier pli, contrat forcé pour que le seuil existe."""
        hands = [list(range(i * 8, (i + 1) * 8)) for i in range(4)]
        env = colver.Env.deal_with_hands(0, hands)
        env.set_contract(0, 10, 0, 0)  # 100♠ par N-S
        env.set_phase_playing()
        return env

    def test_les_trois_cles_sont_la_et_couvrent_les_memes_cartes(self):
        env = self._position()
        res = env.solve_scores()
        cards = {int(c) for c, _ in res["scores"]}
        assert cards == {int(c) for c, _ in res["deal_scores"]}
        assert cards == {int(c) for c, _ in res["contract_made"]}
        assert int(res["best_card"]) in cards

    def test_les_deux_echelles_ne_sont_pas_la_meme(self):
        """Points cartes 0-252 d'un côté, écart de score signé de l'autre. Si
        elles coïncidaient partout, le changement n'aurait servi à rien."""
        env = self._position()
        res = env.solve_scores()
        pts = {int(c): int(v) for c, v in res["scores"]}
        deal = {int(c): int(v) for c, v in res["deal_scores"]}
        assert any(pts[c] != deal[c] for c in pts)

    def test_la_conversion_est_monotone(self):
        """Plus de points cartes pour N-S ne peut jamais faire baisser l'écart
        de score N-S — c'est ce qui garantit que `best_card`, calculé en points
        cartes, reste optimal dans l'autre échelle."""
        env = self._position()
        res = env.solve_scores()
        pts = {int(c): int(v) for c, v in res["scores"]}
        deal = {int(c): int(v) for c, v in res["deal_scores"]}
        for a in pts:
            for b in pts:
                if pts[a] < pts[b]:
                    assert deal[a] <= deal[b]

    def test_contract_made_suit_le_seuil(self):
        """`contract_made` ne se déduit pas du signe de l'écart : il faut que le
        prédicat colle aux points cartes du preneur, seuil du contrat compris."""
        env = self._position()
        res = env.solve_scores()
        pts = {int(c): int(v) for c, v in res["scores"]}
        made = {int(c): bool(v) for c, v in res["contract_made"]}
        # Preneur N-S, contrat 100 : tenu ⟺ N-S ≥ 100 (aucune belote ici).
        for card, ns in pts.items():
            assert made[card] == (ns >= 100), (card, ns, made[card])

    def test_une_bascule_coute_toujours_au_moins_4v(self):
        """La marche au seuil vaut `4V` — épinglé côté Rust dans
        `scoring.rs::deal_score_step_at_the_threshold_is_four_times_the_contract`,
        vérifié ici de bout en bout : si deux cartes tombent de part et d'autre
        du seuil, l'écart entre elles ne peut pas être petit."""
        env = self._position()
        res = env.solve_scores()
        deal = {int(c): int(v) for c, v in res["deal_scores"]}
        made = {int(c): bool(v) for c, v in res["contract_made"]}
        tenu = [deal[c] for c in deal if made[c]]
        chute = [deal[c] for c in deal if not made[c]]
        if tenu and chute:
            assert min(tenu) - max(chute) >= 4 * 100


class TestResume:
    """Le résumé par siège porte les deux totaux."""

    def test_les_deux_totaux_sont_publies(self):
        moves = [
            {"player": 0, "cost": 3, "cost_score": 400, "category": "decisive"},
            {"player": 0, "cost": 42, "cost_score": 84, "category": "imprecision"},
            {"player": 0, "cost": 0, "cost_score": 0, "category": "parfait"},
            {"player": 2, "cost": 9, "cost_score": 0, "forced": True},
        ]
        summary = analysis._summarize(moves)
        nord = summary["players"][0]
        assert nord["total_cost_score"] == 484
        assert nord["total_cost"] == 45
        assert nord["counts"] == {"parfait": 1, "imprecision": 1, "decisive": 1}
        # Une carte forcée n'est pas une décision.
        sud = summary["players"][2]
        assert sud["decisions"] == 0 and sud["forced"] == 1

    def test_les_anciennes_categories_ne_font_pas_planter_le_resume(self):
        """Une ligne d'analyse antérieure se recalcule, mais elle peut être lue
        le temps que `get_or_compute` rende la main."""
        moves = [{"player": 1, "cost": 20, "cost_score": 0, "category": "faute"}]
        summary = analysis._summarize(moves)
        assert summary["players"][1]["counts"]["faute"] == 1


@pytest.mark.parametrize("version", [4, 5, 6])
def test_toute_analyse_anterieure_est_perimee(version):
    """v7 écrit `cost_score`, que personne avant lui ne produisait."""
    assert not analysis._is_fresh({"version": version, "playgen": True}, None, [0, 0])
