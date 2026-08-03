"""Les bots sont l'étalon du classement, pas des joueurs.

Avant le 2026-08-03 ils étaient notés comme tout le monde (K = 8) et dérivaient
avec la population : Dédé était monté de 1000 à 1044, pic 1119, uniquement parce
que les humains perdent contre lui. Comme *tout* le monde est mesuré contre eux,
l'arrivée de joueurs plus faibles dévaluait en silence les inscrits.

Ce qui est épinglé ici, c'est donc surtout une **non-variation** : quoi qu'il
arrive, l'Elo d'un bot ne bouge pas. Une non-variation ne se voit pas à l'usage
— elle se voit six mois plus tard, quand l'échelle a glissé.
"""

import re
from pathlib import Path

import colver.web.database as db
from colver.web import elo


async def _store_solo(hands, actions, user_id, human_seat=2, bot="dede"):
    """Une donne solo terminée : un humain identifié, trois sièges bot."""
    agents = {str(s): bot for s in range(4) if s != human_seat}
    agents[str(human_seat)] = "human"
    game_id = await db.create_game("play", 0, hands, agents,
                                   human_seat=human_seat, user_id=user_id)
    for entry in actions:
        await db.append_action(game_id, entry)
    await db.complete_game(game_id, 80, 82, None)
    return game_id


class TestAncrage:
    def test_un_bot_ne_bouge_pas(self):
        assert elo.K_BOT == 0.0, "un K non nul laisse le bot redériver"

    def test_les_ancres_sont_celles_qui_ont_ete_posees(self):
        """Dédé vaut 1000 par définition ; DouDou est à 50 en dessous.

        ⚠️ **Ces 50 points sont un arrondi, pas une mesure.** L'écart direct a
        été mesuré à +37 dans la métrique *binaire*, périmée depuis R3 ; ré-estimé
        par modèle dans la métrique à la marge il tombe dans [+28, +52], et 50 est
        un chiffre rond dans le haut de cette fourchette. Ce test épingle donc la
        valeur *courante*, pas une valeur juste — le jour où un h2h
        Dédé-contre-DouDou rendra la ligne « Note à la marge », il faudra la
        changer *ici et dans `elo.py`*, ensemble.
        """
        assert elo.bot_elo("dede") == 1000.0
        assert elo.bot_elo("dede") - elo.bot_elo("doudou") == 50.0

    def test_un_bot_inconnu_vaut_l_origine(self):
        """Repli délibéré : mieux vaut une hypothèse visible qu'une dérive."""
        assert elo.bot_elo("un_bot_qui_n_existe_pas") == elo.START_ELO


class TestNoteALaMarge:
    """R3 — une donne se note à la marge, plus au signe.

    Le barème interdit les marges proches de zéro : un contrat réussi rapporte au
    moins `3V − 162` (+78 à V=80), une chute exactement `−(162 + V)` (−242).
    Zéro donne sur 2 999 mesurées sous 78 points d'écart. Le signe ne bouge donc
    presque jamais alors que la marge suit — c'est ce qui rendait « Dédé gagne
    55,4 % des donnes » incompatible avec « Dédé gagne 72 % des matchs ».
    """

    def test_symetrique_autour_de_zero(self):
        for m in (0, 100, 300, 900):
            assert elo.score_from_margin(m) + elo.score_from_margin(-m) == 1.0

    def test_une_donne_nulle_vaut_un_demi(self):
        assert elo.score_from_margin(0) == 0.5

    def test_la_marge_change_la_note(self):
        """La régression à empêcher : revenir à un score binaire déguisé."""
        petite = elo.score_from_margin(100)
        grosse = elo.score_from_margin(600)
        assert 0.5 < petite < grosse < 1.0
        assert grosse - petite > 0.1, "l'échelle écrase trop, on est retombé au binaire"

    def test_l_echelle_est_celle_mesuree(self):
        # Écart-type des marges de donne, 2 999 donnes bot contre bot.
        assert elo.MARGIN_SCALE == 316.0
        # Une marge « typique » (médiane |marge| = 272) doit être informative
        # sans saturer : ni ~0,5 (échelle trop grande) ni ~1 (trop petite).
        typique = elo.score_from_margin(272)
        assert 0.75 < typique < 0.95, f"marge médiane -> {typique:.3f}"

    def test_meme_echelle_que_l_arene(self):
        """L'ancre des bots est calculée par `arena h2h` dans cette métrique.

        Deux valeurs différentes donneraient une ancre incohérente avec
        l'échelle qu'elle est censée ancrer — et le décalage serait invisible,
        les deux nombres vivant dans deux langages.
        """
        rust = (Path(__file__).resolve().parents[1]
                / "colver-core" / "src" / "bin" / "arena.rs").read_text()
        m = re.search(r"const MARGIN_SCALE: f64 = ([0-9.]+);", rust)
        assert m, "MARGIN_SCALE introuvable dans arena.rs"
        assert float(m.group(1)) == elo.MARGIN_SCALE


class TestNotationDUneDonne:
    async def _rate(self, clean_db, played_deal, user_id=1):
        hands, actions = played_deal(seed=11)
        gid = await _store_solo([list(h) for h in hands], actions, user_id)
        assert await elo.rate_game(gid) is True
        return gid

    async def test_le_bot_reste_a_son_ancre_apres_une_donne(self, clean_db, played_deal):
        await self._rate(clean_db, played_deal)
        r = await elo.get_rating("bot", "dede")
        assert r["elo"] == 1000.0
        # …et il a bien joué : le compteur avance même sans que l'Elo bouge.
        assert r["games"] > 0

    async def test_l_humain_bouge(self, clean_db, played_deal):
        await self._rate(clean_db, played_deal)
        r = await elo.get_rating("user", 1)
        assert r["elo"] != elo.START_ELO, "l'humain doit être noté"
        assert r["games"] == 1

    async def test_la_notation_est_idempotente(self, clean_db, played_deal):
        """Le backfill tourne à chaque démarrage : re-noter ne doit rien faire.

        Les lignes des bots sont écrites malgré un delta toujours nul, et c'est
        précisément ce qui garantit l'idempotence — sans elles, une donne sans
        humain n'aurait aucune trace dans `elo_history`.
        """
        gid = await self._rate(clean_db, played_deal)
        before = await elo.get_rating("user", 1)
        assert await elo.rate_game(gid) is False, "seconde notation acceptée"
        assert await elo.get_rating("user", 1) == before

    async def test_le_classement_publie_l_ancre_du_bot(self, clean_db, played_deal):
        """Le tableau lit `elo_ratings.elo` d'un seul SELECT : la copie en base
        doit donc valoir l'ancre, pas une valeur dérivée."""
        await self._rate(clean_db, played_deal)
        board = await elo.leaderboard()
        dede = next(r for r in board if r["kind"] == "bot" and r["ref"] == "dede")
        assert dede["elo"] == 1000.0
