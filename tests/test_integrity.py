"""§4.5 — une donne enregistrée doit décrire une partie qui a pu être jouée.

Deux moitiés du même prédicat : `game_manager.check_legal` empêche d'écrire une
donne fausse, `integrity` trouve celles qui sont déjà en base. Les tests vont
par paires (une donne saine, une donne corrompue) parce que le risque n'est pas
seulement de rater une corruption — c'est aussi d'écarter une donne valable.
"""

import colver
import colver.web.database as db
from colver.web import integrity
from colver.web.game_manager import IllegalAction, PlaySession, check_legal


async def _store(hands, actions, dealer=0, mode="play", complete=True):
    game_id = await db.create_game(mode, dealer, hands, {"0": "doudou"}, human_seat=2)
    for entry in actions:
        await db.append_action(game_id, entry)
    if complete:
        await db.complete_game(game_id, 80, 82, None)
    return game_id


class TestCheckLegal:
    def test_laisse_passer_un_coup_legal(self):
        env = colver.Env()
        env.reset()
        legal = list(env.legal_actions())
        assert check_legal(env, legal[0]) == legal[0]

    def test_refuse_un_coup_hors_des_legaux(self):
        env = colver.Env()
        env.reset()
        legal = list(env.legal_actions())
        bad = next(a for a in range(43) if a not in legal)
        try:
            check_legal(env, bad)
            raise AssertionError("coup illégal accepté")
        except IllegalAction as e:
            assert str(bad) in str(e)

    def test_refuse_tout_coup_sur_une_donne_terminee(self, played_deal):
        hands, actions = played_deal(seed=3)
        env = colver.Env.deal_with_hands(0, [list(h) for h in hands])
        for entry in actions:
            env.step(entry["action"])
        assert env.is_terminal()
        try:
            check_legal(env, 0)
            raise AssertionError("coup accepté après la fin de la donne")
        except IllegalAction:
            pass

    def test_illegal_action_est_une_value_error(self):
        """Les appelants qui attrapaient `ValueError` (la reprise) n'ont rien à
        changer."""
        assert issubclass(IllegalAction, ValueError)


class TestSessionRefuse:
    """Le garde-fou est dans `_record_action`, donc sur le chemin unique par
    lequel une action entre — clic humain, coup de bot ou journal rejoué."""

    def test_un_coup_refuse_ne_laisse_rien_derriere(self):
        s = PlaySession(ai_types={}, human_seat=2)
        legal = list(s.env.legal_actions())
        bad = next(a for a in range(43) if a not in legal)
        try:
            s.play_action(bad)
            raise AssertionError("coup illégal accepté par la session")
        except IllegalAction:
            pass
        assert s.history == []
        assert s.bid_history == []
        assert list(s.env.legal_actions()) == legal  # position intacte

    def test_un_coup_legal_passe_toujours(self):
        s = PlaySession(ai_types={}, human_seat=2)
        legal = list(s.env.legal_actions())
        s.play_action(legal[0])
        assert len(s.history) == 1

    def test_le_rejeu_refuse_un_journal_corrompu(self, played_deal):
        hands, actions = played_deal(seed=5, corrupt_at=25)
        s = PlaySession(ai_types={}, human_seat=2, dealer=0, hands=hands)
        try:
            s.replay(actions)
            raise AssertionError("journal corrompu rejoué sans erreur")
        except IllegalAction as e:
            assert "coup 26" in str(e)  # le rang du coup fautif, 1-based


class TestCheckDeal:
    def test_une_donne_jouee_se_rejoue(self, played_deal):
        hands, actions = played_deal(seed=1)
        assert integrity.check_deal(
            {"dealer": 0, "hands": hands, "actions": actions}) is None

    def test_une_carte_impossible_est_vue(self, played_deal):
        hands, actions = played_deal(seed=1, corrupt_at=20)
        reason = integrity.check_deal(
            {"dealer": 0, "hands": hands, "actions": actions})
        assert reason is not None and "coup 21" in reason

    def test_un_journal_tronque_est_vu(self, played_deal):
        """Une ligne `is_complete = 1` dont les actions ne finissent pas la
        donne ment sur son propre état."""
        hands, actions = played_deal(seed=1)
        reason = integrity.check_deal(
            {"dealer": 0, "hands": hands, "actions": actions[:-4]})
        assert reason is not None and "incomplet" in reason

    def test_des_actions_apres_la_fin_sont_vues(self, played_deal):
        hands, actions = played_deal(seed=1)
        reason = integrity.check_deal(
            {"dealer": 0, "hands": hands, "actions": actions + [actions[-1]]})
        assert reason is not None and "terminée" in reason

    def test_une_distribution_incomplete_est_vue(self, played_deal):
        hands, actions = played_deal(seed=1)
        amputee = [list(h) for h in hands]
        amputee[0] = amputee[0][:-1]
        reason = integrity.check_deal(
            {"dealer": 0, "hands": amputee, "actions": actions})
        assert reason is not None

    def test_une_ligne_illisible_est_une_ligne_fausse(self):
        assert integrity.check_deal(
            {"dealer": 0, "hands": "n'importe quoi", "actions": []}) is not None


class TestScan:
    async def test_ecarte_la_fausse_et_garde_la_saine(self, clean_db, played_deal):
        hands, good = played_deal(seed=2)
        _, bad = played_deal(seed=2, corrupt_at=22)
        ok_id = await _store(hands, good)
        bad_id = await _store(hands, bad)

        seen, flagged = await integrity.scan()
        assert seen == 2
        assert [g for g, _ in flagged] == [bad_id]

        assert await db.get_game(ok_id) is not None
        assert await db.get_game(bad_id) is None
        served = [g["id"] for g in await db.list_games()]
        assert served == [ok_id]

    async def test_chaque_ligne_n_est_examinee_qu_une_fois(self, clean_db, played_deal):
        hands, good = played_deal(seed=4)
        await _store(hands, good)
        assert (await integrity.scan())[0] == 1
        assert (await integrity.scan())[0] == 0

    async def test_une_donne_en_cours_n_est_pas_examinee(self, clean_db, played_deal):
        """Un préfixe n'est pas une corruption : la donne n'est simplement pas
        finie, et la reprise s'en charge."""
        hands, good = played_deal(seed=6)
        await _store(hands, good[:10], complete=False)
        assert (await integrity.scan()) == (0, [])

    async def test_la_donne_ecartee_garde_sa_trace(self, clean_db, played_deal):
        """On marque, on n'efface pas : une donne fausse est un incident."""
        hands, bad = played_deal(seed=8, corrupt_at=18)
        bad_id = await _store(hands, bad)
        await integrity.scan()
        listed = await db.list_invalid_games()
        assert [g["id"] for g in listed] == [bad_id]
        assert listed[0]["reason"]
        row = await db.get_game(bad_id, include_invalid=True)
        assert row is not None

    async def test_les_analyses_en_cache_partent_avec_elle(self, clean_db, played_deal):
        """Elles ont été calculées depuis un état impossible, et `get_or_compute`
        sert le cache avant de relire la donne."""
        hands, bad = played_deal(seed=9, corrupt_at=15)
        bad_id = await _store(hands, bad)
        await db.save_analysis(bad_id, "{}")
        await db.save_agent_review(bad_id, "{}")
        await integrity.scan()
        assert await db.get_analysis(bad_id) is None
        assert await db.get_agent_review(bad_id) is None
