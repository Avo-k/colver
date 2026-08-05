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
        """Une position de jeu née d'une **vraie** enchère.

        Un contrat posé à la main par `set_contract` marche, mais la première
        version distribuait les 8 atouts au même siège : N-S ramassait 252
        points quelle que soit la carte, donc `contract_made` était vrai partout
        et les deux tests de seuil ne mordaient sur rien. Il faut une donne où
        les cartes se disputent.

        (`set_contract` prend d'ailleurs des **points** et les divise par 10 ;
        `set_contract(0, 10, …)` est un contrat à 10, pas à 100 — c'est ce que
        le commentaire d'origine annonçait à tort.)

        On cherche en outre une position où le contrat **bascule** selon la
        carte : c'est la seule sur laquelle le test de la marche à `4V` dise
        quelque chose, et c'est aussi celle qui distingue les deux échelles.
        """
        import random
        for seed in range(50):
            rng = random.Random(seed)
            deck = list(range(32))
            rng.shuffle(deck)
            hands = [sorted(deck[i * 8:(i + 1) * 8]) for i in range(4)]
            env = colver.Env.deal_with_hands(0, hands)
            while int(env.phase()) == 0 and not env.is_terminal():
                a = env.bid_improved_v2()
                env.step(a if a in list(env.legal_actions()) else 0)
            if env.is_terminal() or int(env.phase()) != 1:
                continue  # donne passée : pas de contrat, donc pas de seuil
            for _ in range(12):
                if env.is_terminal():
                    break
                if len(env.legal_actions()) > 1:
                    made = [bool(v) for _c, v in env.solve_scores()["contract_made"]]
                    if len(set(made)) > 1:
                        return env
                env.step(int(env.action_oracle_dd()))
        raise AssertionError("aucune position à contrat basculant trouvée")

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
        prédicat colle aux points cartes du preneur, seuil du contrat compris —
        belote comprise, puisqu'elle **abaisse la barre** au lieu d'ajouter
        20 points au bout."""
        env = self._position()
        res = env.solve_scores()
        contract = env.get_contract()
        taker = int(contract["team"])
        seuil = int(contract["value"]) - int(env.belote_final()[taker])
        pts = {int(c): int(v) for c, v in res["scores"]}
        made = {int(c): bool(v) for c, v in res["contract_made"]}
        for card, ns in pts.items():
            total = 252 if ns in (0, 252) else 162
            taker_pts = ns if taker == 0 else total - ns
            assert made[card] == (taker_pts >= seuil), (card, taker_pts, seuil)

    def test_une_bascule_coute_toujours_au_moins_4v(self):
        """La marche au seuil vaut `4V` — épinglé côté Rust dans
        `scoring.rs::deal_score_step_at_the_threshold_is_four_times_the_contract`,
        vérifié ici de bout en bout : si deux cartes tombent de part et d'autre
        du seuil, l'écart entre elles ne peut pas être petit.

        ⚠️ **Orienté preneur, jamais Nord-Sud.** `deal_scores` est un écart
        signé N-S − E-O : un contrat tenu par Est-Ouest y est *négatif*. La
        première rédaction comparait les deux ensembles dans le repère N-S et
        ne l'a jamais su, parce que la position qu'elle se donnait ne faisait
        jamais basculer le contrat — la garde `if tenu and chute` était
        toujours fausse.
        """
        env = self._position()
        res = env.solve_scores()
        deal = {int(c): int(v) for c, v in res["deal_scores"]}
        made = {int(c): bool(v) for c, v in res["contract_made"]}
        contract = env.get_contract()
        value = int(contract["value"])
        sign = 1 if int(contract["team"]) == 0 else -1
        tenu = [sign * deal[c] for c in deal if made[c]]
        chute = [sign * deal[c] for c in deal if not made[c]]
        assert tenu and chute, "la position choisie doit faire basculer le contrat"
        assert min(tenu) - max(chute) >= 4 * value


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


class _FakeEnv:
    """Le minimum que `_curve` interroge."""

    def __init__(self, value, team, belote=(0, 0), coinche=0, trump=1):
        self._c = {"value": value, "team": team, "coinche": coinche, "trump": trump}
        self._b = list(belote)

    def get_contract(self):
        return self._c

    def belote_final(self):
        return self._b


class TestCourbe:
    """Le seuil de la courbe : l'unité, et la belote."""

    def test_la_valeur_du_contrat_est_deja_en_points(self):
        """`get_contract()['value']` rend `Contract::point_value()`, donc 110 et
        non 11. Une première version le multipliait par 10 et traçait un seuil à
        1100 — la courbe entière était fausse, sans rien casser par ailleurs."""
        c = analysis._curve(_FakeEnv(110, 1), [], [[10, 40]], [[9, 100]])
        assert c["value"] == 110
        assert c["threshold"] == 110

    def test_la_belote_abaisse_la_barre(self):
        """`scoring` ajoute la belote au total du preneur pour décider de la
        réussite : elle **déplace le seuil** au lieu d'ajouter 20 points au
        bout. Tracer l'horizontale à la valeur nue ferait mentir la courbe sur
        toutes les donnes à belote."""
        c = analysis._curve(_FakeEnv(150, 0, belote=(20, 0)), [], [], [[1, 90]])
        assert c["threshold"] == 130
        # La belote de l'adversaire ne déplace pas la barre du preneur.
        c = analysis._curve(_FakeEnv(150, 0, belote=(0, 20)), [], [], [[1, 90]])
        assert c["threshold"] == 150

    def test_le_capot_est_signale(self):
        c = analysis._curve(_FakeEnv(250, 1), [], [], [[1, 252]])
        assert c["capot"] is True
        assert c["threshold"] == 250

    def test_une_donne_passee_n_a_pas_de_courbe(self):
        """Sans contrat il n'y a ni preneur, ni seuil, ni rien à projeter."""
        assert analysis._curve(_FakeEnv(0, 0), [], [], []) is None


class TestVariantes:
    """Les deux lignes d'une erreur, dans le blob (§10.5).

    Elles ne sont calculées **que** sur une erreur : ailleurs le coup joué est
    celui de l'Oracle, donc les deux lignes seraient la même, et le déroulé
    serait payé pour rien sur les ~90 % de coups qui ne coûtent rien.
    """

    @staticmethod
    def _analyse(seed):
        hands = None
        import random
        rng = random.Random(seed)
        deck = list(range(32))
        rng.shuffle(deck)
        hands = [sorted(deck[i * 8:(i + 1) * 8]) for i in range(4)]
        env = colver.Env.deal_with_hands(0, hands)
        actions = []
        while not env.is_terminal():
            a = (env.bid_improved_v2() if int(env.phase()) == 0
                 else int(env.action_heuristic_play()))
            if a not in list(env.legal_actions()):
                a = 0
            env.step(a)
            actions.append({"action": a})
        return analysis._analyze_sync(
            {"dealer": 0, "hands": hands, "actions": actions})

    def test_une_erreur_porte_ses_deux_lignes(self):
        for seed in range(7, 20):
            blob = self._analyse(seed)
            errs = [m for m in blob["moves"] if m.get("cost_score", 0) > 0]
            if not errs:
                continue
            for m in errs:
                v = m["var"]
                assert v["played"]["moves"] == [m["action"]]
                assert v["best"]["moves"] == [m["best"]]
                assert v["played"]["complete"] and v["best"]["complete"]
                # L'invariant qui rend le comparatif honnête : les deux lignes
                # ne diffèrent que par le premier coup, donc leur écart de score
                # **est** le chiffre affiché juste au-dessus.
                sign = 1 if m["player"] % 2 == 0 else -1
                got = sign * (v["best"]["score"] - v["played"]["score"])
                assert got == m["cost_score"], m["idx"]
            return
        pytest.fail("aucune erreur trouvée en 13 donnes — test sans objet")

    def test_un_bon_coup_n_a_pas_de_variante(self):
        blob = self._analyse(7)
        for m in blob["moves"]:
            if m.get("cost_score", 0) == 0:
                assert "var" not in m

    def test_le_coup_de_l_oracle_appartient_a_la_classe_optimale(self):
        """La ligne « Oracle » part de `best_card`, optimal en points cartes,
        alors que la classe affichée est optimale en **score de donne**. Les
        deux coïncident parce que la conversion est monotone — si elle cessait
        de l'être, la variante montrerait une carte absente de la classe."""
        blob = self._analyse(7)
        for m in blob["moves"]:
            if m.get("best_class"):
                assert m["best"] in m["best_class"], m["idx"]


@pytest.mark.parametrize("version", [4, 5, 6, 7, 8])
def test_toute_analyse_anterieure_est_perimee(version):
    """v7 écrit `cost_score`, v8 la courbe, v9 les variantes : rien
    d'antérieur n'est complet."""
    assert not analysis._is_fresh({"version": version, "playgen": True}, None, [0, 0])
