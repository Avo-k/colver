"""L'exploration libre de `/analyse/jeu`, vue du protocole.

Le CFN reste celui de la **vraie** donne ; la branche (`line`) dit seulement par
où on est passé. Quatre choses en découlent qu'on ne peut vérifier qu'ici :

- une branche légale déplace la position d'un coup, et le siège au trait change ;
- une branche illégale est **refusée**, parce que `env.step()` ne valide rien et
  qu'une carte absente d'une main produirait une position sans erreur visible ;
- une branche qui rejoint la ligne réelle retombe sur la **même entrée de
  cache** que la position réelle — c'est ce que la clé promet ;
- la suite en jeu parfait part avec la position, sinon on pousse des cartes sans
  jamais voir où ça mène.

Mêmes bouchons que `test_sim_cache_flow` : les mondes et les avis, qui exigent
le sidecar. Le rejeu, les solves et le déroulé tournent pour de vrai.
"""

import pytest

import colver
import colver.web.card_analysis as card_analysis
import colver.web.game_manager as game_manager
import colver.web.server as server
import colver.web.sim_cache as sim_cache


class FakeWS:
    def __init__(self):
        self.sent = []

    async def send_json(self, payload):
        self.sent.append(payload)

    def types(self):
        return [m["type"] for m in self.sent]

    def last(self, kind):
        return next(m for m in reversed(self.sent) if m["type"] == kind)

    def has(self, kind):
        return any(m["type"] == kind for m in self.sent)


@pytest.fixture
def stubs(monkeypatch):
    monkeypatch.setattr(card_analysis, "plan",
                        lambda pos: {"oracle_worlds": 4, "real_worlds": 2})
    monkeypatch.setattr(card_analysis, "opinions",
                        lambda *a, **kw: {"oracle": 0, "isdd": 0, "doudou": 0})
    monkeypatch.setattr(server, "DMC_MODEL_PATH", None)
    real = card_analysis.sample_worlds
    monkeypatch.setattr(card_analysis, "sample_worlds",
                        lambda *a, **kw: (real(*a, **kw)[0], "playgen"))


def _two_decisions(played_deal, seed=3):
    """`(hands, ids, cfn, i, j)` — deux décisions non forcées consécutives.

    `j` suit immédiatement `i` dans le journal : rejouer la carte réelle de `i`
    comme branche doit donc rendre exactement la position `j`.
    """
    hands, actions = played_deal(seed=seed)
    ids = [a["action"] for a in actions]
    cfn = game_manager.compute_game_cfn(0, hands, ids)
    play_idxs = [i for i, a in enumerate(actions) if a["phase"] == 1]
    for i, j in zip(play_idxs, play_idxs[1:], strict=False):
        if j != i + 1:
            continue
        a = card_analysis.describe(0, hands, ids, i)
        b = card_analysis.describe(0, hands, ids, j)
        if "error" in a or "error" in b or a["forced"] or b["forced"]:
            continue
        return hands, ids, cfn, i, j
    pytest.skip("pas deux décisions non forcées consécutives dans cette donne")


@pytest.mark.asyncio
async def test_pousser_la_carte_reelle_donne_la_position_suivante(
        clean_db, played_deal, stubs):
    """La branche est un chemin, pas une réécriture : jouer la carte réelle
    redonne la position réelle d'après, siège compris."""
    hands, ids, cfn, i, j = _two_decisions(played_deal)

    direct = FakeWS()
    await server._run_card_analysis(direct, {"cfn": cfn, "idx": j, "req_id": 1})

    branched = FakeWS()
    await server._run_card_analysis(
        branched, {"cfn": cfn, "idx": i, "line": [ids[i]], "req_id": 2})

    a = direct.last("card_analysis_position")
    b = branched.last("card_analysis_position")
    assert b["line"] == [ids[i]]
    assert b["on_real"] is True
    assert b["position"]["seat"] == a["position"]["seat"]
    assert b["position"]["hands"] == a["position"]["hands"]
    assert b["position"]["current_trick"] == a["position"]["current_trick"]
    # La carte réellement jouée survit à la branche tant qu'on est sur la vraie
    # ligne — c'est elle qui porte le liseré « jouée » du tableau.
    assert b["position"]["played_action"] == a["position"]["played_action"]


@pytest.mark.asyncio
async def test_une_branche_qui_rejoint_le_reel_partage_l_entree(
        clean_db, played_deal, stubs):
    """La clé porte la position effective, donc deux chemins vers la même
    position ne se paient qu'une fois."""
    hands, ids, cfn, i, j = _two_decisions(played_deal, seed=6)

    await server._run_card_analysis(FakeWS(), {"cfn": cfn, "idx": j, "req_id": 1})
    rows = await clean_db.execute_fetchall("SELECT COUNT(*) FROM analysis_cache")
    assert rows[0][0] == 1

    ws = FakeWS()
    await server._run_card_analysis(
        ws, {"cfn": cfn, "idx": i, "line": [ids[i]], "req_id": 2})
    assert ws.last("card_analysis_done").get("cached") is True
    rows = await clean_db.execute_fetchall("SELECT COUNT(*) FROM analysis_cache")
    assert rows[0][0] == 1, "la même position ne doit pas se calculer deux fois"


@pytest.mark.asyncio
async def test_une_branche_illegale_est_refusee(clean_db, played_deal, stubs):
    """`env.step()` avale une carte absente de la main sans rien dire : sans ce
    contrôle, toute la page décrirait une position qui n'a jamais pu exister."""
    hands, ids, cfn, i, _j = _two_decisions(played_deal, seed=4)
    env = colver.Env.deal_with_hands(0, [list(h) for h in hands])
    for a in ids[:i]:
        env.step(int(a))
    illegal = next(c for c in range(32) if c not in list(env.legal_actions()))

    ws = FakeWS()
    await server._run_card_analysis(
        ws, {"cfn": cfn, "idx": i, "line": [illegal], "req_id": 1})

    assert ws.types() == ["card_analysis_error"]
    err = ws.last("card_analysis_error")
    assert "variante" in err["error"].lower()
    # Le client a besoin de savoir qu'il y a une branche d'où revenir : sans ce
    # champ la page est un cul-de-sac.
    assert err["branch"] == 1
    rows = await clean_db.execute_fetchall("SELECT COUNT(*) FROM analysis_cache")
    assert rows[0][0] == 0


@pytest.mark.asyncio
async def test_la_suite_en_jeu_parfait_accompagne_la_position(
        clean_db, played_deal, stubs):
    hands, ids, cfn, i, _j = _two_decisions(played_deal, seed=5)
    ws = FakeWS()
    await server._run_card_analysis(ws, {"cfn": cfn, "idx": i, "req_id": 1})

    line = ws.last("card_analysis_line")["line"]
    assert line["complete"] is True
    assert line["taker"] in (0, 1)
    assert isinstance(line["made"], bool)
    # La ligne part de la position analysée et va jusqu'à la 32e carte.
    played = sum(len(h) for h in ws.last("card_analysis_position")["position"]["played"])
    assert len(line["cards"]) == 32 - played

    # Et elle est resservie depuis la base au second passage, sinon la page se
    # repeindrait sans son panneau le plus lisible.
    warm = FakeWS()
    await server._run_card_analysis(warm, {"cfn": cfn, "idx": i, "req_id": 2})
    assert warm.last("card_analysis_done")["cached"] is True
    assert warm.last("card_analysis_line")["line"] == line


@pytest.mark.asyncio
async def test_une_branche_divergente_n_est_plus_sur_la_ligne_reelle(
        clean_db, played_deal, stubs):
    """Hors de la vraie ligne, « la carte jouée » n'existe plus. L'afficher
    quand même désignerait une carte que personne n'a jouée dans cette
    variante."""
    hands, ids, cfn, i, _j = _two_decisions(played_deal, seed=7)
    env = colver.Env.deal_with_hands(0, [list(h) for h in hands])
    for a in ids[:i]:
        env.step(int(a))
    other = next(c for c in env.legal_actions() if int(c) != ids[i])

    ws = FakeWS()
    await server._run_card_analysis(
        ws, {"cfn": cfn, "idx": i, "line": [int(other)], "req_id": 1})
    pos = ws.last("card_analysis_position")
    assert pos["on_real"] is False
    assert pos["position"]["played_action"] is None


@pytest.mark.asyncio
async def test_le_cache_d_une_variante_ne_pollue_pas_la_donne_reelle(
        clean_db, played_deal, stubs):
    """Deux positions différentes, deux entrées. Une variante partage la clé de
    la position réelle **seulement** si elle y mène vraiment."""
    hands, ids, cfn, i, _j = _two_decisions(played_deal, seed=8)
    env = colver.Env.deal_with_hands(0, [list(h) for h in hands])
    for a in ids[:i]:
        env.step(int(a))
    # La branche doit mener à une **décision** : sur une position forcée le
    # gestionnaire s'arrête avant le cache, et le test compterait une entrée de
    # moins sans que rien ne soit cassé.
    other = next(
        (int(c) for c in env.legal_actions()
         if int(c) != ids[i]
         and not card_analysis.describe(
             0, hands, ids[:i] + [int(c)], i + 1).get("forced", True)),
        None)
    if other is None:
        pytest.skip("aucune variante ne mène à une décision ici")

    await server._run_card_analysis(FakeWS(), {"cfn": cfn, "idx": i, "req_id": 1})
    await server._run_card_analysis(
        FakeWS(), {"cfn": cfn, "idx": i, "line": [other], "req_id": 2})

    rows = await clean_db.execute_fetchall(
        "SELECT COUNT(*) FROM analysis_cache WHERE kind = ?",
        (sim_cache.KIND_CARD,))
    assert rows[0][0] == 2
