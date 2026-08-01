"""`pacing` — le tempo d'affichage et le bot qui va avec.

Ce qui compte ici n'est pas la valeur exacte des pauses mais les propriétés qui,
si elles cèdent, changent le jeu sans rien casser visiblement : le mode dégradé
doit **s'avouer**, et la pause doit absorber le temps de réflexion du bot au lieu
de s'y ajouter.
"""

import asyncio
import time

import pytest

from colver.web import pacing


class TestResolve:
    def test_standard_assied_dede(self):
        bot, think_ms, degraded = pacing.resolve("standard", doudou_available=True)
        assert (bot, degraded) == ("dede", False)
        assert think_ms > 0

    def test_rapide_assied_doudou(self):
        assert pacing.resolve("rapide", doudou_available=True)[:1] == ("doudou",)

    def test_rapide_sans_doudou_degrade_et_le_dit(self):
        """Le repli est Dédé sur un budget court — mais il doit être *annoncé*,
        sinon le joueur croit affronter le bot rapide."""
        bot, think_ms, degraded = pacing.resolve("rapide", doudou_available=False)
        assert bot == "dede"
        assert degraded is True
        assert think_ms == pacing.MODES["rapide"]["think_ms"]

    def test_standard_ne_degrade_jamais(self):
        assert pacing.resolve("standard", doudou_available=False)[2] is False

    @pytest.mark.parametrize("junk", [None, "", "turbo", 0, 1.5])
    def test_mode_inconnu_retombe_sur_le_defaut(self, junk):
        assert pacing.normalize(junk) == pacing.DEFAULT_MODE
        assert pacing.resolve(junk)[0] == pacing.MODES[pacing.DEFAULT_MODE]["bot"]


class TestModeForBot:
    def test_aller_retour_pour_chaque_mode(self):
        for name, spec in pacing.MODES.items():
            assert pacing.mode_for_bot(spec["bot"]) == name

    def test_bot_inconnu_retombe_sur_le_defaut(self):
        """Une donne reprise dont le bot ne se relit pas (repli dégradé, agent
        retiré) doit repartir sur un tempo, pas sur une exception."""
        assert pacing.mode_for_bot(None) == pacing.DEFAULT_MODE
        assert pacing.mode_for_bot("inconnu") == pacing.DEFAULT_MODE


class TestDelais:
    def test_les_pauses_decroissent_sur_les_huit_plis(self):
        d = [pacing.card_delay("standard", t) for t in range(8)]
        assert d == sorted(d, reverse=True)
        assert d[0] > d[-1]

    def test_standard_garde_un_plancher_lisible(self):
        """Au 8e pli il faut encore pouvoir lire qui a coupé."""
        assert pacing.card_delay("standard", 7) >= 0.9
        assert pacing.trick_delay("standard", 7) >= 1.2

    def test_rapide_est_partout_plus_court_que_standard(self):
        for t in range(8):
            assert pacing.card_delay("rapide", t) < pacing.card_delay("standard", t)

    def test_index_de_pli_hors_bornes_reste_borne(self):
        assert pacing.card_delay("standard", -3) == pacing.card_delay("standard", 0)
        assert pacing.card_delay("standard", 99) == pacing.card_delay("standard", 7)

    def test_dernier_pli_hors_tempo_du_mode(self):
        """Le dernier pli ne contient aucune décision : il a son propre tempo,
        le même dans les deux modes."""
        for mode in pacing.MODES:
            assert pacing.card_delay(mode, 7, cards_in_trick=0) == pacing.LAST_TRICK_LEAD
            for n in (1, 2, 3):
                assert pacing.card_delay(mode, 7, cards_in_trick=n) == pacing.LAST_TRICK_CARD

    def test_fin_de_donne_tenue_plus_longtemps_que_le_mode(self):
        for mode in pacing.MODES:
            assert pacing.trick_delay(mode, 7, deal_over=True) == pacing.DEAL_END_HOLD
            assert (pacing.trick_delay(mode, 7, deal_over=True)
                    > pacing.trick_delay(mode, 7))

    def test_move_delay_lit_la_phase(self):
        assert pacing.move_delay("standard", 0, 0) == pacing.bid_delay("standard")
        assert pacing.move_delay("standard", 1, 3) == pacing.card_delay("standard", 3)


class TestHold:
    async def test_la_reflexion_se_deduit_de_la_pause(self):
        """`hold` ne dort que le reste : sans ça les 1,2 s de Dédé s'ajouteraient
        à chaque pause et le mode standard tournerait au double de son tempo."""
        t0 = time.monotonic()
        await pacing.hold(0.20, elapsed=0.15)
        spent = time.monotonic() - t0
        assert 0.03 <= spent < 0.15

    async def test_une_reflexion_plus_longue_que_la_pause_ne_dort_pas(self):
        t0 = time.monotonic()
        await pacing.hold(0.05, elapsed=1.0)
        assert time.monotonic() - t0 < 0.02

    async def test_hold_rend_la_main_a_la_boucle(self):
        """Une pause ne doit jamais bloquer l'event loop : les autres tables
        jouent pendant."""
        marker = []

        async def other():
            await asyncio.sleep(0)
            marker.append("vivant")

        await asyncio.gather(pacing.hold(0.05), other())
        assert marker == ["vivant"]
