"""`match_state.Match` — le compteur d'une partie en 1000 / 2000 points.

Pas de base, pas de moteur : de l'arithmétique et deux invariants qui, s'ils
lâchent, faussent une partie entière sans rien afficher d'anormal.
"""

from colver.web.match_state import DEFAULT_TARGET, Match, normalize_target


class TestNormalizeTarget:
    def test_cibles_connues(self):
        assert [normalize_target(v) for v in (0, 1000, 2000)] == [0, 1000, 2000]

    def test_tout_le_reste_retombe_sur_le_defaut(self):
        for junk in (None, "", "abc", -1, 1500, 3000, {}, [1000]):
            assert normalize_target(junk) == DEFAULT_TARGET

    def test_une_cible_numerique_est_tronquee_avant_d_etre_validee(self):
        """`int()` d'abord, appartenance ensuite : « 1000.5 » et « 1000 » (en
        texte) donnent une partie en 1000 points. Sans danger, mais autant que
        ce soit écrit — c'est ce qui distingue une valeur tolérée d'un oubli."""
        assert normalize_target(1000.5) == 1000
        assert normalize_target("2000") == 2000
        assert normalize_target(1999.9) == DEFAULT_TARGET


class TestRecord:
    def test_cumule_les_scores(self):
        m = Match(2000)
        m.record("aaaa", [162, 0])
        m.next_deal()
        m.record("bbbb", [0, 250])
        assert m.totals == [162, 250]
        assert len(m.deals) == 2

    def test_record_est_idempotent(self):
        """Deux chemins mènent à la fin d'une donne en solo (le coup humain
        terminal, puis `_run_ai_turns`). Sans ce garde-fou la donne serait
        comptée deux fois et le score de la partie serait faux."""
        m = Match(2000)
        assert m.record("aaaa", [162, 0]) is True
        assert m.record("aaaa", [162, 0]) is False
        assert m.totals == [162, 0]
        assert len(m.deals) == 1

    def test_le_donneur_est_memorise_par_donne(self):
        m = Match(2000, dealer=0)
        m.record("aaaa", [10, 0])
        m.next_deal()
        m.record("bbbb", [10, 0])
        assert [d["dealer"] for d in m.deals] == [0, 1]


class TestFinished:
    def test_donne_unique_finit_apres_une_donne(self):
        m = Match(0)
        assert m.finished is False
        m.record("aaaa", [0, 162])
        assert m.finished is True
        assert m.winner == 1

    def test_partie_continue_sous_la_cible(self):
        m = Match(1000)
        m.record("aaaa", [900, 100])
        assert m.finished is False
        assert m.winner is None

    def test_egalite_a_la_cible_ne_termine_pas(self):
        """Les deux camps marquent à chaque donne et peuvent franchir la cible
        ensemble ; à égalité parfaite on rejoue."""
        m = Match(1000)
        m.record("aaaa", [1010, 1010])
        assert max(m.totals) >= 1000
        assert m.finished is False
        assert m.winner is None

    def test_un_ecart_apres_egalite_tranche(self):
        m = Match(1000)
        m.record("aaaa", [1010, 1010])
        m.next_deal()
        m.record("bbbb", [0, 90])
        assert m.finished is True
        assert m.winner == 1


class TestRestore:
    def test_reprend_le_score_stocke_et_non_la_somme_des_donnes(self):
        """`games.points_ns/ew` sont les points *cartes* ; le score *marqué*
        n'est cumulé que dans `matches.points_ns/ew`. Une partie reprise doit
        repartir de ce cumul-là, et les donnes déjà jouées revenir sans score
        plutôt qu'avec un chiffre de la mauvaise unité."""
        deals = [{"game_id": "aaaa", "dealer": 0}, {"game_id": "bbbb", "dealer": 1}]
        m = Match.restore(2000, [380, 120], deals, dealer=2, match_id="mmmm")
        assert m.totals == [380, 120]
        assert m.id == "mmmm"
        assert m.dealer == 2
        assert m.deal_no == 3
        assert [d["scores"] for d in m.deals] == [None, None]

    def test_partie_reprise_deja_gagnee_est_finie(self):
        m = Match.restore(1000, [1020, 300], [{"game_id": "a", "dealer": 0}], dealer=1)
        assert m.finished is True
        assert m.winner == 0

    def test_une_donne_rejouee_ne_se_recompte_pas(self):
        """La donne reprise porte l'identifiant qu'elle avait déjà ; si elle est
        dans `deals`, c'est qu'elle était terminée et déjà comptée."""
        m = Match.restore(2000, [380, 120], [{"game_id": "aaaa", "dealer": 0}], dealer=1)
        assert m.record("aaaa", [162, 0]) is False
        assert m.totals == [380, 120]
