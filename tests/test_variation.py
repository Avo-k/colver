"""Le moteur de variantes : deux invariants, et les deux façons de le casser.

Une variante déroule la donne en jeu parfait après un ou plusieurs coups posés.
Tout ce que Rejouer et `/analyse/jeu` affichent en découle, donc deux propriétés
doivent tenir sans exception :

1. **Le déroulé rend exactement la valeur DD du coup.** Les quatre sièges y
   jouent en minimax sur les 32 cartes, donc le total du preneur *est* la valeur
   du nœud. Si ça diverge, c'est le solveur ou le déroulé qui ment, et les deux
   pages afficheraient un chiffre et une ligne qui ne se correspondent pas.
2. **L'écart entre les deux lignes d'une erreur est le coût affiché.** C'est la
   seule chose qui rende le comparatif honnête (§4.1 de la fiche) : les deux
   lignes partagent tout sauf le premier coup.
"""

import colver
import pytest

from colver.web import variation


def _first_decisions(hands, ids, dealer=0, limit=6):
    """`(idx, env)` pour les premières décisions de jeu non forcées."""
    env = colver.Env.deal_with_hands(dealer, [list(h) for h in hands])
    out = []
    for idx, a in enumerate(ids):
        if env.is_terminal():
            break
        if int(env.phase()) == 1 and len(env.legal_actions()) > 1:
            out.append(idx)
            if len(out) >= limit:
                break
        env.step(int(a))
    return out


class TestInvariants:
    def test_le_deroule_rend_la_valeur_dd_du_coup(self, played_deal):
        """Invariant 1, sur toutes les décisions d'une donne complète."""
        hands, actions = played_deal(seed=11)
        ids = [a["action"] for a in actions]
        env = colver.Env.deal_with_hands(0, [list(h) for h in hands])
        checked = 0
        for idx, a in enumerate(ids):
            if env.is_terminal():
                break
            if int(env.phase()) == 1 and len(env.legal_actions()) > 1:
                res = env.solve_scores()
                pts = {int(c): int(v) for c, v in res["scores"]}
                taker = int(env.get_contract()["team"])
                for card in list(pts)[:3]:  # trois cartes suffisent par position
                    v = variation.line(0, hands, ids[:idx], [card])
                    assert v is not None and v["complete"]
                    # `pts` est côté Nord-Sud ; la variante rend le preneur.
                    total = 252 if pts[card] in (0, 252) else 162
                    want = pts[card] if taker == 0 else total - pts[card]
                    assert v["taker_pts"] == want, (idx, card)
                    checked += 1
            env.step(int(a))
        assert checked > 20, "la donne n'a pas offert assez de décisions"

    def test_l_ecart_des_deux_lignes_est_le_cout(self, played_deal):
        """Invariant 2 : ce que le panneau de Rejouer promet au lecteur.

        Plusieurs donnes, parce qu'un tiers d'entre elles se jouent sans la
        moindre erreur (mesuré : 14 sur 40) — un seed figé rendrait ce test
        vide une fois sur trois, sans échouer.
        """
        errors = 0
        for seed in range(12, 24):
            if errors >= 5:
                break
            hands, actions = played_deal(seed=seed)
            ids = [a["action"] for a in actions]
            env = colver.Env.deal_with_hands(0, [list(h) for h in hands])
            for idx, a in enumerate(ids):
                if env.is_terminal():
                    break
                if int(env.phase()) == 1 and len(env.legal_actions()) > 1:
                    res = env.solve_scores()
                    deal = {int(c): int(v) for c, v in res["deal_scores"]}
                    team = int(env.current_player()) % 2
                    best_deal = (max(deal.values()) if team == 0
                                 else min(deal.values()))
                    cost = ((best_deal - deal[a]) if team == 0
                            else (deal[a] - best_deal))
                    if cost > 0:
                        lines = variation.error_lines(
                            0, hands, ids[:idx], a, int(res["best_card"]))
                        assert lines is not None
                        sign = 1 if team == 0 else -1
                        got = sign * (lines["best"]["score"]
                                      - lines["played"]["score"])
                        assert got == cost, (seed, idx, a)
                        errors += 1
                env.step(int(a))
        assert errors >= 5, "pas assez d'erreurs trouvées — test sans objet"


class TestLegalite:
    """`env.step()` ne valide rien : c'est ici que la légalité se contrôle."""

    def test_un_coup_illegal_rend_none(self, played_deal):
        hands, actions = played_deal(seed=13)
        ids = [a["action"] for a in actions]
        idx = _first_decisions(hands, ids, limit=1)[0]
        env = colver.Env.deal_with_hands(0, [list(h) for h in hands])
        for a in ids[:idx]:
            env.step(int(a))
        illegal = next(c for c in range(32) if c not in list(env.legal_actions()))
        assert variation.line(0, hands, ids[:idx], [illegal]) is None

    def test_limit_zero_valide_sans_derouler(self, played_deal):
        """Le contrôle que le serveur fait sur une branche reçue du client.

        Sans `limit=0` il faudrait dérouler la donne entière pour savoir si une
        pile de coups tient — c'est-à-dire payer un déroulé complet à chaque
        touche pressée dans l'exploration.
        """
        hands, actions = played_deal(seed=13)
        ids = [a["action"] for a in actions]
        idx = _first_decisions(hands, ids, limit=1)[0]
        legal = variation.line(0, hands, ids[:idx], [ids[idx]], limit=0)
        assert legal is not None
        assert legal["cards"] == [] and legal["complete"] is False
        assert "taker_pts" not in legal

    def test_une_annonce_n_est_pas_une_position_de_variante(self, played_deal):
        hands, actions = played_deal(seed=13)
        ids = [a["action"] for a in actions]
        assert int(actions[0]["phase"]) == 0
        assert variation.line(0, hands, ids[:0], []) is None


class TestDecoupageEnPlis:
    def test_trick_pos_situe_la_variante_dans_le_pli(self, played_deal):
        """Le client découpe la ligne en plis avec ce seul entier. Faux, il
        dessine des plis à cheval et la donne devient illisible."""
        hands, actions = played_deal(seed=14)
        ids = [a["action"] for a in actions]
        play_idxs = [i for i, a in enumerate(actions) if a["phase"] == 1]
        first = play_idxs[0]
        for k in range(4):
            v = variation.line(0, hands, ids[:first + k], [], limit=0)
            assert v is not None and v["trick_pos"] == k

    def test_la_ligne_va_jusqu_a_la_derniere_carte(self, played_deal):
        hands, actions = played_deal(seed=14)
        ids = [a["action"] for a in actions]
        play_idxs = [i for i, a in enumerate(actions) if a["phase"] == 1]
        first = play_idxs[0]
        v = variation.line(0, hands, ids[:first], [])
        assert v["complete"] is True
        assert len(v["moves"]) + len(v["cards"]) == 32


@pytest.mark.parametrize("seed", [21, 22])
def test_une_variante_ne_touche_pas_la_partie_de_l_appelant(played_deal, seed):
    """Le moteur reconstruit sa propre partie, il n'emprunte pas celle-ci.

    `set_state_bytes` ne restaure que le `GameState` : `play_order`,
    `played_by` et `bid_history` sont des champs du wrapper PyO3, hors des
    84 octets. Un instantané-restauration laisserait donc les cartes de la
    variante dans `play_order`, d'où sortent l'observation DMC et `to_cfn`.
    """
    hands, actions = played_deal(seed=seed)
    ids = [a["action"] for a in actions]
    idx = _first_decisions(hands, ids, limit=1)[0]
    env = colver.Env.deal_with_hands(0, [list(h) for h in hands])
    for a in ids[:idx]:
        env.step(int(a))
    before = (list(env.get_play_order()), list(env.get_bid_history()),
              [list(h) for h in env.get_hands()])

    variation.line(0, hands, ids[:idx], [ids[idx]])

    assert list(env.get_play_order()) == before[0]
    assert list(env.get_bid_history()) == before[1]
    assert [list(h) for h in env.get_hands()] == before[2]
