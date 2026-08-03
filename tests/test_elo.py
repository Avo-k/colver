"""L'unité classée est la partie en 2000 points, et les bots en sont l'étalon.

Deux invariants, tous deux invisibles à l'usage :

- **une non-variation** — quoi qu'il arrive, l'Elo d'un bot ne bouge pas. Avant le
  2026-08-03 ils dérivaient avec la population (Dédé était monté de 1000 à 1044,
  pic 1119) et, comme *tout* le monde est mesuré contre eux, l'arrivée de joueurs
  plus faibles dévaluait en silence les inscrits ;
- **une abstention** — une donne isolée et une partie en 1000 ne comptent pas. Le
  jour où quelqu'un les rebranchera « pour avoir plus de données », l'échelle
  changera d'un facteur ~3,4 sans que rien ne le signale.
"""

import colver.web.database as db
from colver.web import elo


async def _match(target=2000, winner=0, deals=1, user_id=1, human_seat=2,
                 bot="dede", complete=True, abandoned=False):
    """Une partie terminée : un humain identifié, trois sièges bot."""
    match_id = await db.create_match("play", target, user_id=user_id,
                                     human_seat=human_seat)
    agents = {str(s): bot for s in range(4) if s != human_seat}
    agents[str(human_seat)] = "human"
    hands = [list(range(8 * i, 8 * i + 8)) for i in range(4)]
    for n in range(deals):
        gid = await db.create_game("play", 0, hands, agents, human_seat=human_seat,
                                   user_id=user_id, match_id=match_id, deal_no=n + 1)
        await db.complete_game(gid, 80, 82, {"value": 80})
    if abandoned:
        # `db.abandon_match` exige `is_complete = 0` : en production on concède
        # une partie **en cours**. La clôturer d'abord en ferait un no-op — et
        # c'est exactement ce qui a fait échouer la première version de ce test.
        await db.update_match(match_id, 2000, 900, deals, False, None)
        await db.abandon_match(match_id, user_id)
    else:
        await db.update_match(match_id, 2000, 900, deals, complete, winner)
    return match_id


class TestAncrage:
    def test_un_bot_ne_bouge_pas(self):
        assert elo.K_BOT == 0.0, "un K non nul laisse le bot redériver"

    def test_les_ancres_sont_celles_qui_ont_ete_posees(self):
        """Dédé vaut 1000 par définition ; DouDou est 170 en dessous.

        ⚠️ **Ces 170 points sont une conversion, pas une mesure.** L'écart valait
        50 à la donne — lui-même un arrondi dans une fourchette modélisée
        [+28, +52] — et 50 × 3,4 (facteur mesuré donne → partie) donne 170. Il
        hérite donc de toute l'incertitude de son prédécesseur, amplifiée.

        Ce test épingle la valeur *courante*, pas une valeur juste. Le jour où un
        `arena h2h web_dede web_doudou` tournera sur un GPU tranquille, il rendra
        directement le bon chiffre — c'est tout l'intérêt d'être passé à l'unité
        de l'arène — et il faudra le changer *ici et dans `elo.py`*, ensemble.
        """
        assert elo.bot_elo("dede") == 1000.0
        assert elo.bot_elo("dede") - elo.bot_elo("doudou") == 170.0

    def test_un_bot_inconnu_vaut_l_origine(self):
        """Repli délibéré : mieux vaut une hypothèse visible qu'une dérive."""
        assert elo.bot_elo("un_bot_qui_n_existe_pas") == elo.START_ELO

    def test_l_unite_est_dite_dans_la_version_d_etalonnage(self):
        """Un Elo « donne » et un Elo « partie » ne se comparent pas (×3,4)."""
        assert elo.ANCHOR_VERSION.endswith("match")


class TestKParPaliers:
    def test_le_k_decroit(self):
        assert elo.k_for(0) > elo.k_for(10) > elo.k_for(30)

    def test_les_paliers_sont_ceux_annonces(self):
        assert elo.k_for(0) == elo.k_for(9) == 64.0
        assert elo.k_for(10) == elo.k_for(29) == 32.0
        assert elo.k_for(30) == elo.k_for(500) == 24.0


class TestSeulesLesPartiesEn2000Comptent:
    async def test_une_partie_en_2000_est_notee(self, clean_db):
        assert await elo.rate_match(await _match(target=2000)) is True

    async def test_une_donne_isolee_n_est_pas_notee(self, clean_db):
        """`target = 0` est le **défaut du site** : c'est le cas fréquent."""
        assert await elo.rate_match(await _match(target=0)) is False

    async def test_une_partie_en_1000_n_est_pas_notee(self, clean_db):
        assert await elo.rate_match(await _match(target=1000)) is False

    async def test_une_partie_inachevee_n_est_pas_notee(self, clean_db):
        assert await elo.rate_match(await _match(complete=False)) is False

    async def test_une_partie_contenant_une_donne_invalide_est_ecartee(self, clean_db):
        """Son score cumulé est faux : même règle que `integrity.scan`."""
        mid = await _match()
        conn = await db.get_db()
        await conn.execute("UPDATE games SET invalid = 1 WHERE match_id = ?", (mid,))
        await conn.commit()
        assert await elo.rate_match(mid) is False


class TestNotation:
    async def test_l_humain_monte_en_gagnant(self, clean_db):
        await elo.rate_match(await _match(winner=0, human_seat=2))  # siège 2 = N-S
        assert (await elo.get_rating("user", 1))["elo"] > elo.START_ELO

    async def test_l_humain_descend_en_perdant(self, clean_db):
        await elo.rate_match(await _match(winner=1, human_seat=2))
        assert (await elo.get_rating("user", 1))["elo"] < elo.START_ELO

    async def test_le_bot_reste_a_son_ancre(self, clean_db):
        await elo.rate_match(await _match())
        r = await elo.get_rating("bot", "dede")
        assert r["elo"] == 1000.0
        assert r["games"] == 1, "le compteur avance même sans que l'Elo bouge"

    async def test_games_compte_des_parties_pas_des_sieges(self, clean_db):
        """Dédé tient trois sièges sur quatre ; ça reste **une** partie.

        Le compteur affichait 2 540 pour 881 donnes jouées, et la page mentait.
        """
        await elo.rate_match(await _match())
        assert (await elo.get_rating("bot", "dede"))["games"] == 1

    async def test_la_notation_est_idempotente(self, clean_db):
        """Le backfill tourne à chaque démarrage : re-noter ne doit rien faire."""
        mid = await _match()
        assert await elo.rate_match(mid) is True
        before = await elo.get_rating("user", 1)
        assert await elo.rate_match(mid) is False, "seconde notation acceptée"
        assert await elo.get_rating("user", 1) == before


class TestAbandon:
    async def test_abandonner_vaut_defaite(self, clean_db):
        """Sans ça, quitter en étant mené serait gratuit.

        La partie est marquée gagnée par N-S (`winner=0`) *et* abandonnée par
        l'humain assis en N-S : c'est l'abandon qui doit l'emporter, sinon on
        noterait une victoire à qui a quitté la table.
        """
        await elo.rate_match(await _match(winner=0, human_seat=2, abandoned=True))
        assert (await elo.get_rating("user", 1))["elo"] < elo.START_ELO


class TestSeuilDAffichage:
    async def test_un_humain_sous_le_seuil_n_apparait_pas(self, clean_db):
        await elo.rate_match(await _match())
        board = await elo.leaderboard()
        assert not [r for r in board if r["kind"] == "user"], \
            "un classement sur 1 partie vaut ±609 Elo : c'est du bruit publié"
        assert [r for r in board if r["kind"] == "bot"], "les étalons restent visibles"

    async def test_il_apparait_au_seuil(self, clean_db):
        for _ in range(elo.MIN_RATED_MATCHES):
            await elo.rate_match(await _match())
        assert [r for r in await elo.leaderboard() if r["kind"] == "user"]

    async def test_standing_dit_ce_qui_reste_a_jouer(self, clean_db):
        await elo.rate_match(await _match())
        st = await elo.standing("user", 1)
        assert st["ranked"] is False
        assert st["remaining"] == elo.MIN_RATED_MATCHES - 1

    async def test_le_classement_publie_l_ancre_du_bot(self, clean_db):
        """Le tableau lit `elo_ratings.elo` d'un seul SELECT : la copie en base
        doit donc valoir l'ancre, pas une valeur dérivée."""
        await elo.rate_match(await _match())
        board = await elo.leaderboard()
        dede = next(r for r in board if r["kind"] == "bot" and r["ref"] == "dede")
        assert dede["elo"] == 1000.0
