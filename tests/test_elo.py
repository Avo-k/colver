"""L'unité classée est la partie en 2000 points, et les bots en sont l'étalon.

Quatre invariants, tous invisibles à l'usage :

- **une non-variation** — quoi qu'il arrive, la note d'un bot ne bouge pas. Avant
  le 2026-08-03 ils dérivaient avec la population (Dédé était monté de 1000 à
  1044, pic 1119) et, comme *tout* le monde est mesuré contre eux, l'arrivée de
  joueurs plus faibles dévaluait en silence les inscrits ;
- **une abstention** — une donne isolée et une partie en 1000 ne comptent pas. Le
  jour où quelqu'un les rebranchera « pour avoir plus de données », l'échelle
  changera d'un facteur ~3,4 sans que rien ne le signale ;
- **un découplage** — la note d'un joueur ne dépend que de son propre bilan.
  Ré-estimer le prior sur la population (Bayes empirique) était la première idée,
  et elle déplaçait la note des autres de ±30 à chaque partie ;
- **une monotonie** — à niveau constant, jouer fait *monter* la note. C'est ce qui
  ferme le défaut qui a motivé toute la refonte : le tableau ordonnait par
  inexpérience (Spearman −0,89 entre parties jouées et note affichée).
"""

import pytest

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
        """Une note « donne » et une note « partie » ne se comparent pas (×3,4)."""
        assert elo.ANCHOR_VERSION.endswith("match")


class TestEchelleDAffichage:
    """L'échelle interne est celle des mesures ; l'affichage en est une transformée
    affine, appliquée en un seul endroit. Les deux points humains sont le contrat.
    """

    def test_les_deux_points_humains_sont_ceux_annonces(self):
        assert elo.to_display(elo._INTERNAL_NEW) == elo.DISPLAY_NEW
        assert elo.to_display(elo.PRIOR_MEAN) == elo.DISPLAY_TYPICAL

    def test_la_conversion_est_reversible(self):
        for x in (-500.0, 0.0, 550.0, 1000.0, 1210.0):
            assert abs(elo.from_display(elo.to_display(x)) - x) < 1e-9

    def test_ancrer_sur_un_bot_enverrait_les_humains_sous_zero(self):
        """Le piège de la lecture « échiquéenne » : mettre DouDou à 1000 demande
        k ≈ 3,7, et le joueur typique tombe alors sous 0. L'écart humain →
        DouDou50 (280 en interne) est plus grand que DouDou → Dédé (170)."""
        assert elo.bot_elo("dede") - elo.bot_elo("doudou") == 170.0
        assert elo.PRIOR_MEAN < elo.bot_elo("doudou") - 170.0

    def test_les_bots_se_lisent_dans_la_plage_annoncee(self):
        assert round(elo.to_display(elo.bot_elo("dede"))) == 2200
        assert round(elo.to_display(elo.bot_elo("doudou"))) == 1973


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


def _note(record):
    return elo.note_of(*elo.posterior(record))


class TestPosterior:
    """Le posterior est une fonction pure du bilan : il se teste sans base."""

    def test_gagner_vaut_mieux_que_perdre(self):
        gagne, _ = elo.posterior([(1.0, 1000.0, 1000.0)])
        perd, _ = elo.posterior([(0.0, 1000.0, 1000.0)])
        assert gagne > elo.PRIOR_MEAN > perd

    def test_jouer_reduit_l_incertitude(self):
        sigmas = [elo.posterior([(0.5, 1000.0, 1000.0)] * n)[1]
                  for n in (0, 5, 20, 100)]
        assert sigmas == sorted(sigmas, reverse=True)
        # Sans partie, le posterior EST le prior — au bruit de discrétisation de
        # la grille près (pas de 1 Elo, soit 1/12 de variance ajoutée).
        assert sigmas[0] == pytest.approx(elo.PRIOR_SD, abs=0.01)

    def test_la_note_monte_a_niveau_constant(self):
        """**La propriété qui ferme le défaut d'origine.** Un joueur exactement au
        niveau du prior voit son niveau estimé rester sur place ; seul sigma
        tombe, donc sa note monte. Un nouveau venu entre par le bas et grimpe, au
        lieu d'entrer au milieu et de couler — ce qui faisait que le tableau
        ordonnait par inexpérience (Spearman −0,89).
        """
        p = 1.0 / (1.0 + 10 ** ((1000.0 - elo.PRIOR_MEAN) / 800.0))
        record = [(p, 1000.0, 1000.0)]
        notes = [_note(record * n) for n in (0, 5, 20, 100)]
        assert notes == sorted(notes)
        assert notes[-1] - notes[0] > 500, "la montée doit être franche, pas un epsilon"
        # Le niveau, lui, tient sa place : il ne dérive que de quelques points
        # (la vraisemblance logistique n'est pas symétrique autour de son mode,
        # donc la moyenne a posteriori bouge un peu), sans commune mesure avec la
        # montée de la note.
        niveaux = [elo.posterior(record * n)[0] for n in (0, 5, 20, 100)]
        assert all(abs(x - elo.PRIOR_MEAN) < 20.0 for x in niveaux), \
            "le niveau doit rester sur place : c'est sigma qui porte la montée"

    def test_la_note_est_toujours_sous_le_niveau(self):
        """`mu - 2 sigma` : la note est ce qu'on peut prouver, jamais l'estimation
        elle-même. Les afficher comme un seul nombre était le défaut n° 2."""
        for n in (0, 1, 10, 100):
            mean, sd = elo.posterior([(0.5, 1000.0, 1000.0)] * n)
            assert _note([(0.5, 1000.0, 1000.0)] * n) < elo.to_display(mean)

    def test_un_joueur_sans_partie_lit_le_plancher(self):
        assert _note([]) == pytest.approx(elo.DISPLAY_NEW, abs=0.05)


class TestNotation:
    async def test_l_humain_monte_en_gagnant(self, clean_db):
        await elo.rate_match(await _match(winner=0, human_seat=2))  # siège 2 = N-S
        gagne = await elo.get_rating("user", 1)
        await db.get_db()
        assert gagne["level"] > elo.DISPLAY_TYPICAL

    async def test_l_humain_descend_en_perdant(self, clean_db):
        await elo.rate_match(await _match(winner=1, human_seat=2))
        assert (await elo.get_rating("user", 1))["level"] < elo.DISPLAY_TYPICAL

    async def test_le_bot_reste_a_son_ancre(self, clean_db):
        await elo.rate_match(await _match())
        r = await elo.get_rating("bot", "dede")
        assert r["elo"] == elo.to_display(1000.0)
        assert r["uncertainty"] == 0.0, "un étalon n'a pas d'intervalle"
        assert r["games"] == 1, "le compteur avance même sans que la note bouge"

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
        assert (await elo.get_rating("user", 1))["level"] < elo.DISPLAY_TYPICAL


class TestDecouplage:
    """La note d'un joueur ne dépend que de son propre bilan.

    Ré-estimer le prior sur la population (Bayes empirique) semblait plus
    rigoureux et c'était une régression : mesuré sur la base de prod, une seule
    partie gagnée par X déplaçait la note de tous les autres de +28 à +32, et
    l'arrivée d'un joueur qui perd ses 20 premières les faisait tous chuter de 100
    à 150. C'est le défaut que `K_BOT = 0` avait fermé pour les bots, remonté d'un
    étage. D'où `PRIOR_MEAN` / `PRIOR_SD` **figés**.
    """

    async def test_la_note_ne_bouge_pas_quand_quelqu_un_d_autre_joue(self, clean_db):
        await elo.rate_match(await _match(user_id=1))
        avant = await elo.get_rating("user", 1)
        for _ in range(5):
            await elo.rate_match(await _match(user_id=2, winner=1))
        assert await elo.get_rating("user", 1) == avant

    async def test_deux_bilans_identiques_donnent_deux_notes_identiques(self, clean_db):
        for uid in (1, 2):
            await elo.rate_match(await _match(user_id=uid, winner=1))
        a, b = await elo.get_rating("user", 1), await elo.get_rating("user", 2)
        assert a == b


class TestSeuilDAffichage:
    """Le seuil masque l'affichage, **il ne suspend pas la notation**.

    Et ce n'est pas un seuil de précision : à 5 parties l'IC95 vaut encore
    ±609 Elo. C'est un choix éditorial — ne pas publier le nom de quelqu'un qui a
    joué une partie. Depuis que la note est conservatrice il n'est plus
    structurellement nécessaire (un joueur non confirmé se range tout seul en
    bas), donc le retirer resterait correct.
    """

    async def test_un_joueur_sous_le_seuil_n_apparait_pas(self, clean_db):
        await elo.rate_match(await _match())
        board = await elo.leaderboard()
        assert not [r for r in board if r["kind"] == "user"]
        assert [r for r in board if r["kind"] == "bot"], "les étalons restent visibles"

    async def test_il_apparait_au_seuil(self, clean_db):
        for _ in range(elo.MIN_RATED_MATCHES):
            await elo.rate_match(await _match())
        assert [r for r in await elo.leaderboard() if r["kind"] == "user"]

    async def test_la_note_se_construit_quand_meme(self, clean_db):
        """Masqué ≠ non noté : le bilan s'accumule, et le joueur arrive au seuil
        avec la note qu'il a méritée, pas avec une note neuve."""
        for _ in range(elo.MIN_RATED_MATCHES - 1):
            await elo.rate_match(await _match(winner=0))
        r = await elo.get_rating("user", 1)
        assert r["games"] == elo.MIN_RATED_MATCHES - 1
        assert r["level"] > elo.DISPLAY_TYPICAL, "quatre victoires doivent compter"

    async def test_standing_dit_ce_qui_reste_a_jouer(self, clean_db):
        """Sans ça, la page ferait disparaître quelqu'un sans explication."""
        await elo.rate_match(await _match())
        st = await elo.standing("user", 1)
        assert st["ranked"] is False
        assert st["remaining"] == elo.MIN_RATED_MATCHES - 1
        assert st["level"] and st["uncertainty"], \
            "la note provisoire doit être rendue, c'est ce que la page affiche"

    async def test_un_bot_n_est_jamais_masque(self, clean_db):
        st = await elo.standing("bot", "dede")
        assert st["ranked"] is True and st["remaining"] == 0

    async def test_le_classement_publie_l_ancre_du_bot(self, clean_db):
        """Le tableau lit `elo_ratings.elo` d'un seul SELECT : la copie en base
        doit donc valoir l'ancre, pas une valeur dérivée."""
        await elo.rate_match(await _match())
        board = await elo.leaderboard()
        dede = next(r for r in board if r["kind"] == "bot" and r["ref"] == "dede")
        assert dede["elo"] == elo.to_display(1000.0)


class TestListeDesDonnes:
    """« Mes donnes » ne doit rien devoir à `elo_history`.

    Le passage à la partie (v14) a reconstruit `elo_history` sur `match_id`,
    mais `list_games` a gardé un sous-select sur `elo_history.game_id` pour
    afficher une variation d'Elo par donne. Résultat : `no such column` dès
    qu'un joueur connecté ouvrait `/compte`, et une liste vide.

    Ce qui a laissé passer le bug est ici l'essentiel : le sous-select n'était
    assemblé **que** dans la branche `user_id is not None`, et aucun test
    n'appelait `list_games` avec un joueur. D'où ces deux-là, qui parcourent les
    deux chemins — le solo (`games.user_id`) et le salon (`game_players`).
    """

    async def test_les_donnes_d_un_joueur_se_listent(self, clean_db):
        await _match(deals=3)
        rows = await db.list_games(user_id=1)
        assert len(rows) == 3
        assert all(r["user_seat"] == 2 for r in rows)

    async def test_le_chemin_salon_aussi(self, clean_db):
        """Une donne de salon ne porte pas `human_seat` : le siège se lit sur
        `game_players`, l'autre moitié du `COALESCE`."""
        hands = [list(range(8 * i, 8 * i + 8)) for i in range(4)]
        gid = await db.create_game("multi", 0, hands, {str(s): "dede" for s in range(4)})
        await db.add_game_player(gid, 1, 7)
        await db.complete_game(gid, 80, 82, {"value": 80})
        rows = await db.list_games(user_id=7)
        assert [r["id"] for r in rows] == [gid]
        assert rows[0]["user_seat"] == 1

    async def test_une_donne_ne_porte_plus_de_variation_d_elo(self, clean_db):
        """L'unité notée est la partie : un `elo_delta` par donne serait dix fois
        le même chiffre, et laisserait croire que chaque donne l'a gagné."""
        await _match(deals=2)
        assert all("elo_delta" not in r for r in await db.list_games(user_id=1))
