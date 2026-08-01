"""§2.7 — une donne interrompue se reprend à son coup près.

Le test central est la **fidélité du rejeu** : une donne coupée à un coup
quelconque puis rejouée depuis son journal doit rendre exactement la position
qu'on avait, et pas seulement une position légale. On compare donc terme à
terme, à toutes les coupures — la faute qu'on cherche (un registre d'affichage
reconstruit de travers, un pli qui manque) est invisible sur la seule position
du moteur.
"""

import pytest

import colver
from colver.web.game_manager import PlaySession

# Coupures : avant tout coup, en pleine enchère, sur les premiers plis, en
# plein pli, et tard dans la donne.
CUTS = [0, 1, 3, 6, 13, 20, 31]


def _live_session(hands, actions, cut, dealer=0):
    """La session telle qu'elle serait après `cut` coups joués normalement."""
    s = PlaySession(ai_types={}, human_seat=2, dealer=dealer, hands=hands)
    for entry in actions[:cut]:
        s.play_action(entry["action"])
    return s


def _resumed_session(hands, actions, cut, dealer=0):
    """La même, reconstruite depuis le journal — le chemin de la reprise."""
    s = PlaySession(ai_types={}, human_seat=2, dealer=dealer, hands=hands)
    s.replay(actions[:cut])
    return s


def _fingerprint(s):
    """Tout ce qui doit se retrouver à l'identique après une reprise."""
    env = s.env
    return {
        "cfn": env.to_cfn(),
        "phase": int(env.phase()),
        "current_player": int(env.current_player()),
        "hands": [list(h) for h in env.get_hands()],
        "trick": list(env.get_current_trick()),
        "contract": env.get_contract(),
        "points": list(env.get_points()),
        "tricks_won": list(env.get_tricks_won()),
        "belote": list(env.get_belote()),
        "legal": list(env.legal_actions()) if not env.is_terminal() else [],
        # Registres d'affichage : ils ne vivent que côté Python, donc rien dans
        # le moteur ne les rattraperait s'ils étaient reconstruits de travers.
        "history": [dict(h) for h in s.history],
        "bid_history": [dict(b) for b in s.bid_history],
        "completed_tricks": [dict(t) for t in s.completed_tricks],
        "initial_hands": [list(h) for h in s.initial_hands],
    }


class TestFideliteDuRejeu:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    @pytest.mark.parametrize("cut", CUTS)
    def test_position_et_registres_identiques(self, played_deal, seed, cut):
        hands, actions = played_deal(seed=seed)
        cut = min(cut, len(actions))
        assert _fingerprint(_live_session(hands, actions, cut)) \
            == _fingerprint(_resumed_session(hands, actions, cut))

    def test_le_rejeu_n_annonce_ni_pli_ni_belote(self, played_deal):
        """Le client reçoit une position, pas un coup : rien à faire clignoter,
        et surtout pas une belote annoncée une seconde fois."""
        hands, actions = played_deal(seed=1)
        s = _resumed_session(hands, actions, 20)
        assert s.trick_just_completed is False
        assert s._belote_event is None
        assert s._belote_player is None

    def test_une_donne_reprise_se_finit_normalement(self, played_deal):
        """Ce que la reprise doit permettre : continuer, pas seulement afficher."""
        hands, actions = played_deal(seed=2)
        s = _resumed_session(hands, actions, 17)
        for entry in actions[17:]:
            s.play_action(entry["action"])
        assert s.env.is_terminal()
        assert len(s.completed_tricks) == 8

    def test_rejeu_a_zero_coup_est_une_donne_neuve(self, played_deal):
        hands, actions = played_deal(seed=3)
        s = _resumed_session(hands, actions, 0)
        assert s.history == []
        # Le moteur range chaque main ; c'est l'ensemble qui doit coïncider.
        assert [sorted(h) for h in s.env.get_hands()] == [sorted(h) for h in hands]


class TestDistribution:
    def test_les_mains_imposees_sont_respectees(self, deal):
        hands = deal(11)
        s = PlaySession(ai_types={}, human_seat=2, dealer=1, hands=hands)
        assert [sorted(h) for h in s.env.get_hands()] == [sorted(h) for h in hands]
        assert int(s.env.get_dealer()) == 1

    def test_le_donneur_impose_est_respecte(self):
        for dealer in range(4):
            s = PlaySession(ai_types={}, human_seat=2, dealer=dealer)
            assert int(s.env.get_dealer()) == dealer

    def test_initial_hands_est_un_instantane(self, played_deal):
        """Il est envoyé en fin de donne pour le récapitulatif : il doit décrire
        la distribution, pas ce qu'il reste en main."""
        hands, actions = played_deal(seed=7)
        s = _live_session(hands, actions, 20)
        assert sorted(c for h in s.initial_hands for c in h) == list(range(32))


class TestPasseForce:
    def test_predicat_faux_sur_une_enchere_ordinaire(self):
        from colver.web.game_manager import only_pass_is_legal
        env = colver.Env()
        env.reset()
        assert only_pass_is_legal(env) is False
        assert len(list(env.legal_actions())) > 1

    def test_predicat_faux_en_phase_de_jeu(self, played_deal):
        """Une seule carte jouable n'est *pas* un passe forcé : le prédicat ne
        vaut qu'en enchère, sinon le dernier pli serait passé en boucle."""
        from colver.web.game_manager import in_last_trick, only_pass_is_legal
        hands, actions = played_deal(seed=4)
        s = _resumed_session(hands, actions, len(actions) - 3)
        if s.env.phase() == 1 and not s.env.is_terminal():
            assert only_pass_is_legal(s.env) is False
            assert in_last_trick(s.env) is True
