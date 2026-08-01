"""§2.7 — retrouver en base la donne (ou la partie) laissée en plan.

`pending_deal` est la porte d'entrée de la reprise, et c'est aussi une lecture
qui rend **les quatre mains en clair** : ce qu'elle refuse de rendre compte
autant que ce qu'elle rend. D'où les tests d'isolation entre comptes.
"""

import colver.web.database as db


async def _user(name="alice"):
    return await db.create_user(name, "$2b$12$fake")


async def _deal(user_id, actions, hands, *, complete=False, match_id=None,
                deal_no=None, mode="play"):
    game_id = await db.create_game(
        mode, 0, hands, {"0": "doudou", "1": "doudou", "2": "human", "3": "doudou"},
        human_seat=2, user_id=user_id, match_id=match_id, deal_no=deal_no)
    for entry in actions:
        await db.append_action(game_id, entry)
    if complete:
        await db.complete_game(game_id, 80, 82, None)
    return game_id


class TestPendingDeal:
    async def test_rend_la_donne_en_cours_avec_de_quoi_la_rejouer(
            self, clean_db, played_deal):
        hands, actions = played_deal(seed=1)
        uid = await _user()
        game_id = await _deal(uid, actions[:12], hands)

        row = await db.pending_deal(uid)
        assert row is not None
        assert row["id"] == game_id
        assert len(row["actions"]) == 12
        assert [sorted(h) for h in row["hands"]] == [sorted(h) for h in hands]
        assert row["dealer"] == 0

    async def test_ignore_une_donne_terminee(self, clean_db, played_deal):
        hands, actions = played_deal(seed=1)
        uid = await _user()
        await _deal(uid, actions, hands, complete=True)
        assert await db.pending_deal(uid) is None

    async def test_ne_rend_pas_la_donne_d_un_autre_compte(self, clean_db, played_deal):
        """La ligne porte les quatre mains : la servir au mauvais compte serait
        donner le jeu des trois autres."""
        hands, actions = played_deal(seed=1)
        alice, bob = await _user("alice"), await _user("bob")
        await _deal(alice, actions[:8], hands)
        assert await db.pending_deal(bob) is None
        assert (await db.pending_deal(alice)) is not None

    async def test_anonyme_n_a_rien_a_reprendre(self, clean_db, played_deal):
        """Sans compte, aucune identité à laquelle rattacher une donne — c'est
        ce qui laisse la redonne gratuite au joueur anonyme (§2.1)."""
        hands, actions = played_deal(seed=1)
        await _deal(None, actions[:8], hands)
        assert await db.pending_deal(None) is None
        assert await db.pending_deal(0) is None

    async def test_donne_hors_partie_et_donne_de_partie_ne_se_confondent_pas(
            self, clean_db, played_deal):
        """Sans `match_id`, on ne veut que la donne isolée — reprendre celle
        d'une partie hors de sa partie perdrait le score cumulé."""
        hands, actions = played_deal(seed=1)
        uid = await _user()
        match_id = await db.create_match("play", 2000, user_id=uid, pacing="standard",
                                         human_seat=2)
        in_match = await _deal(uid, actions[:5], hands, match_id=match_id, deal_no=1)

        assert await db.pending_deal(uid) is None            # aucune donne isolée
        assert (await db.pending_deal(uid, match_id=match_id))["id"] == in_match

        lone = await _deal(uid, actions[:9], hands)
        assert (await db.pending_deal(uid))["id"] == lone

    async def test_le_salon_ne_se_reprend_pas_par_cette_porte(
            self, clean_db, played_deal):
        """En salon c'est le pilote qui tient la donne, et il survit à la
        déconnexion d'un joueur."""
        hands, actions = played_deal(seed=1)
        uid = await _user()
        await _deal(uid, actions[:6], hands, mode="multi")
        assert await db.pending_deal(uid) is None


class TestDropDeal:
    async def test_efface_une_donne_en_cours(self, clean_db, played_deal):
        hands, actions = played_deal(seed=1)
        uid = await _user()
        game_id = await _deal(uid, actions[:7], hands)
        await db.drop_deal(game_id)
        assert await db.pending_deal(uid) is None

    async def test_n_efface_jamais_une_donne_terminee(self, clean_db, played_deal):
        """Une donne jouée est un résultat : Elo, score de partie, historique."""
        hands, actions = played_deal(seed=1)
        uid = await _user()
        game_id = await _deal(uid, actions, hands, complete=True)
        await db.drop_deal(game_id)
        assert await db.get_game(game_id) is not None


class TestOpenMatches:
    async def test_une_partie_ouverte_se_liste_avec_ses_reglages(
            self, clean_db, played_deal):
        uid = await _user()
        match_id = await db.create_match("play", 1000, user_id=uid,
                                         pacing="rapide", human_seat=1)
        await db.update_match(match_id, 380, 120, 2, False)
        listed = await db.list_open_matches(uid)
        assert len(listed) == 1
        assert listed[0]["id"] == match_id
        # Le rythme est un réglage de partie : sans lui, une partie « rapide »
        # reprise repartirait derrière Dédé.
        assert listed[0]["pacing"] == "rapide"
        assert listed[0]["human_seat"] == 1
        assert (listed[0]["points_ns"], listed[0]["points_ew"]) == (380, 120)
        assert listed[0]["pending"] is False

    async def test_pending_signale_une_donne_en_plan(self, clean_db, played_deal):
        hands, actions = played_deal(seed=1)
        uid = await _user()
        match_id = await db.create_match("play", 2000, user_id=uid)
        await _deal(uid, actions[:4], hands, match_id=match_id, deal_no=1)
        assert (await db.list_open_matches(uid))[0]["pending"] is True

    async def test_une_partie_concedee_disparait_des_listes(self, clean_db):
        uid = await _user()
        match_id = await db.create_match("play", 2000, user_id=uid)
        assert await db.abandon_match(match_id, uid) is True
        assert await db.list_open_matches(uid) == []
        row = await db.get_match(match_id)
        # Une partie jouée jusqu'au bout a toujours un vainqueur ; l'abandon
        # est le seul cas `is_complete = 1` sans `winner`.
        assert row["is_complete"] is True and row["winner"] is None

    async def test_on_ne_concede_pas_la_partie_d_un_autre(self, clean_db):
        alice, bob = await _user("alice"), await _user("bob")
        match_id = await db.create_match("play", 2000, user_id=alice)
        assert await db.abandon_match(match_id, bob) is False
        assert len(await db.list_open_matches(alice)) == 1

    async def test_load_open_match_ne_rend_que_les_donnes_terminees(
            self, clean_db, played_deal):
        """Les donnes déjà jouées servent à compter et à numéroter ; celle qui
        est en plan n'y figure que par son donneur."""
        hands, actions = played_deal(seed=1)
        uid = await _user()
        match_id = await db.create_match("play", 2000, user_id=uid)
        await _deal(uid, actions, hands, complete=True, match_id=match_id, deal_no=1)
        await _deal(uid, actions[:5], hands, match_id=match_id, deal_no=2)

        row = await db.load_open_match(match_id, uid)
        assert len(row["deals"]) == 1
        assert row["pending_dealer"] == 0

    async def test_load_open_match_verifie_le_proprietaire(self, clean_db):
        alice, bob = await _user("alice"), await _user("bob")
        match_id = await db.create_match("play", 2000, user_id=alice)
        assert await db.load_open_match(match_id, bob) is None
