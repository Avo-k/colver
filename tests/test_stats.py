"""Les chiffres « vie du site », et l'invariant de barème sur lequel ils reposent.

Trois familles, et la deuxième est la plus importante :

- **v16** — une donne enregistre ses points *marqués*, pas seulement ses points
  *cartes*. C'est ce qui répare `user_game_stats`, qui déclarait victorieuse
  toute chute où le preneur gardait la majorité des cartes.
- **« contrat tenu ⟺ le camp preneur marque »** — tout le calcul des contrats
  se fait en SQL grâce à cette équivalence, qui est une propriété du barème et
  non une approximation. Elle est vérifiée ici contre le moteur, sur des donnes
  réellement jouées, points cartes et belote à l'appui.
- **les agrégats** — que le preneur, les coinches et les capots se lisent bien
  depuis `games.actions` sans rejouer la donne.
"""

import colver
import colver.web.database as db
import colver.web.integrity as integrity
import colver.web.stats as stats


async def _store_played(hands, actions, *, user_id, seat, dealer=0,
                        mode="play", scores=True, created_at=None):
    """Une donne jouée, enregistrée comme le ferait la production."""
    agents = {str(s): "doudou" for s in range(4)}
    if mode == "play":
        agents[str(seat)] = "human"
        gid = await db.create_game(mode, dealer, hands, agents,
                                   human_seat=seat, user_id=user_id)
    else:
        gid = await db.create_game(mode, dealer, hands, agents)
        await db.add_game_player(gid, seat, user_id)
    for entry in actions:
        await db.append_action(gid, entry)

    env = colver.Env.deal_with_hands(dealer, [list(h) for h in hands])
    for entry in actions:
        env.step(int(entry["action"]))
    pts = list(env.get_points())
    rw = list(env.rewards())
    await db.complete_game(gid, pts[0], pts[1], env.get_contract(),
                           score_ns=rw[0] if scores else None,
                           score_ew=rw[1] if scores else None)
    if created_at is not None:
        conn = await db.get_db()
        await conn.execute("UPDATE games SET created_at = ? WHERE id = ?",
                           (created_at, gid))
        await conn.commit()
    return gid, env


class TestScoresEnregistres:
    async def test_la_cloture_ecrit_les_deux_echelles(self, clean_db, played_deal):
        """Points cartes ET points marqués : ni l'un ni l'autre ne se déduit."""
        uid = await db.create_user("alice", "x")
        hands, actions = played_deal(seed=1)
        gid, env = await _store_played(hands, actions, user_id=uid, seat=2)
        row = await db.get_game(gid)
        assert [row["points_ns"], row["points_ew"]] == list(env.get_points())
        assert [row["score_ns"], row["score_ew"]] == list(env.rewards())

    async def test_le_rattrapage_remplit_les_donnes_anciennes(self, clean_db,
                                                              played_deal):
        """Les donnes closes avant v16 n'ont pas de score : on le rejoue."""
        uid = await db.create_user("alice", "x")
        hands, actions = played_deal(seed=2)
        gid, env = await _store_played(hands, actions, user_id=uid, seat=2,
                                       scores=False)
        assert (await db.get_game(gid))["score_ns"] is None
        assert await db.games_missing_scores() == [gid]

        assert await integrity.backfill_scores() == 1
        row = await db.get_game(gid)
        assert [row["score_ns"], row["score_ew"]] == list(env.rewards())
        # Idempotent : plus rien à rattraper au démarrage suivant.
        assert await db.games_missing_scores() == []
        assert await integrity.backfill_scores() == 0


class TestVictoireAuScoreMarque:
    """Le bug que v16 ferme, isolé sur la seule ligne qui le produisait."""

    async def _chute_gardant_les_cartes(self, uid):
        """Une donne où le preneur fait 90 points cartes sur 162 — et marque 0.

        C'est le cas exact que l'ancien code inversait : 110♠ annoncé, 90 points
        faits, contrat chuté. Écrite à la main plutôt que cherchée dans des
        donnes tirées au hasard — l'événement est rare, et le test doit porter
        sur lui, pas sur la chance du tirage.
        """
        hands = [list(range(8 * i, 8 * i + 8)) for i in range(4)]
        gid = await db.create_game("play", 0, hands, {"2": "human"},
                                   human_seat=2, user_id=uid)
        await db.complete_game(gid, 90, 72, {"value": 110, "team": 0},
                               score_ns=0, score_ew=272)
        return gid

    async def test_une_chute_n_est_pas_une_victoire(self, clean_db):
        uid = await db.create_user("alice", "x")
        await self._chute_gardant_les_cartes(uid)
        st = await db.user_game_stats(uid)
        assert st["games"] == 1
        assert st["wins"] == 0, (
            "le preneur garde 90 points cartes sur 162 mais marque 0 : "
            "compter les cartes inverse le résultat")

    async def test_une_donne_sans_score_sort_du_denominateur(self, clean_db,
                                                            played_deal):
        """Plutôt que d'être comptée perdue en attendant le rattrapage."""
        uid = await db.create_user("alice", "x")
        hands, actions = played_deal(seed=3)
        await _store_played(hands, actions, user_id=uid, seat=2, scores=False)
        assert (await db.user_game_stats(uid))["games"] == 0
        await integrity.backfill_scores()
        assert (await db.user_game_stats(uid))["games"] == 1


class TestInvariantDeBareme:
    """« Contrat tenu ⟺ le camp preneur marque » — la clé de tout le SQL.

    Sous ce barème une chute donne exactement 0 au preneur, et un contrat réussi
    au moins `contrat + points cartes`. Donc `score[camp preneur] > 0` est un
    test de réussite exact, qui n'a besoin ni de rejouer la donne ni de savoir
    qui tenait la belote — alors que la comparaison naïve
    `points cartes >= valeur` se trompe sur toute donne où le preneur annonce
    une belote (elle compte dans le total du preneur avant le seuil).
    """

    def test_sur_des_donnes_reellement_jouees(self, played_deal):
        vus = 0
        for seed in range(120):
            hands, actions = played_deal(seed=seed)
            env = colver.Env.deal_with_hands(0, [list(h) for h in hands])
            for entry in actions:
                env.step(int(entry["action"]))
            contract = env.get_contract()
            if not contract:
                continue  # donne passée : ni preneur ni contrat
            team = int(contract["team"])
            value = int(contract["value"])
            pts = list(env.get_points())
            belote = list(env.get_belote())
            marque = int(env.rewards()[team])

            # Référence indépendante : le preneur tient son contrat si ses
            # points cartes, belote comprise, atteignent la valeur annoncée.
            tenu = pts[team] + (20 if belote[team] else 0) >= value
            assert (marque > 0) is tenu, (
                f"seed {seed}: contrat {value} team {team}, "
                f"cartes {pts[team]}, belote {belote[team]}, marqué {marque}")
            vus += 1
        assert vus > 100, f"trop peu de donnes contractées testées ({vus})"

    def test_un_preneur_qui_chute_marque_exactement_zero(self, played_deal):
        chutes = 0
        for seed in range(120):
            hands, actions = played_deal(seed=seed)
            env = colver.Env.deal_with_hands(0, [list(h) for h in hands])
            for entry in actions:
                env.step(int(entry["action"]))
            contract = env.get_contract()
            if not contract:
                continue
            team = int(contract["team"])
            pts = list(env.get_points())
            belote = list(env.get_belote())
            if pts[team] + (20 if belote[team] else 0) < int(contract["value"]):
                assert int(env.rewards()[team]) == 0
                chutes += 1
        assert chutes > 0, "aucune chute dans l'échantillon : test sans portée"


class TestLectureDuJournal:
    """Preneur, coinches et hauteur d'annonce se lisent en SQL, sans rejeu."""

    async def test_le_preneur_lu_en_sql_est_le_bon_siege(self, clean_db,
                                                         played_deal):
        uid = await db.create_user("alice", "x")
        expected = {}
        for seed in range(12):
            hands, actions = played_deal(seed=seed)
            gid, env = await _store_played(hands, actions, user_id=uid, seat=2)
            contract = env.get_contract()
            if not contract:
                continue
            # Le dernier siège à avoir annoncé un chiffre : la référence.
            last = [a for a in actions if a["phase"] == 0 and 1 <= a["action"] <= 40]
            expected[gid] = last[-1]["player"] if last else None

        conn = await db.get_db()
        rows = await conn.execute_fetchall(f"""
            WITH s AS ({stats._SEATS})
            SELECT s.game_id, {stats._TAKER}, json_extract(s.contract, '$.team')
              FROM s WHERE s.uid = ?
        """, (uid,))
        seen = 0
        for gid, taker, team in rows:
            if gid not in expected:
                continue
            assert taker == expected[gid], f"donne {gid}"
            assert taker % 2 == team, "le preneur doit être du camp du contrat"
            seen += 1
        assert seen >= 8, f"échantillon trop maigre ({seen})"

    async def test_les_annonces_du_siege_sont_comptees(self, clean_db,
                                                       played_deal):
        uid = await db.create_user("alice", "x")
        hands, actions = played_deal(seed=5)
        await _store_played(hands, actions, user_id=uid, seat=2)
        mine = [a for a in actions if a["phase"] == 0 and a["player"] == 2]
        st = await stats.my_stats(uid)
        assert st["bidding"]["decisions"] == len(mine)
        assert st["bidding"]["pass_pct"] == round(
            100 * sum(1 for a in mine if a["action"] == 0) / len(mine), 1)


class TestClassementVieDuSite:
    async def test_les_comptes_par_joueur(self, clean_db, played_deal):
        alice = await db.create_user("alice", "x")
        bob = await db.create_user("bob", "x")
        for seed in range(5):
            hands, actions = played_deal(seed=seed)
            await _store_played(hands, actions, user_id=alice, seat=2,
                                created_at=f"2026-08-0{seed + 1}T12:00:00+00:00")
        hands, actions = played_deal(seed=9)
        await _store_played(hands, actions, user_id=bob, seat=1, mode="multi",
                            created_at="2026-08-01T12:00:00+00:00")

        board = await stats.leaderboard()
        by_name = {r["name"]: r for r in board}
        assert by_name["alice"]["deals"] == 5
        assert by_name["alice"]["days"] == 5
        assert by_name["bob"]["deals"] == 1
        # Trié sur les donnes : le plus assidu d'abord.
        assert [r["name"] for r in board] == ["alice", "bob"]

    async def test_un_bot_n_apparait_jamais(self, clean_db, played_deal):
        """Ces tableaux disent qui fait vivre le site. Un bot tient trois sièges
        sur quatre et jouerait tous les jours."""
        alice = await db.create_user("alice", "x")
        hands, actions = played_deal(seed=1)
        await _store_played(hands, actions, user_id=alice, seat=2)
        assert all(r["name"] == "alice" for r in await stats.leaderboard())

    async def test_la_serie_compte_les_jours_consecutifs(self, clean_db,
                                                         played_deal):
        """Et elle n'est « en cours » que si elle touche aujourd'hui ou hier."""
        from datetime import datetime, timedelta, timezone
        alice = await db.create_user("alice", "x")
        today = datetime.now(timezone.utc).date()
        for back in (2, 1, 0):
            hands, actions = played_deal(seed=back)
            day = (today - timedelta(days=back)).isoformat()
            await _store_played(hands, actions, user_id=alice, seat=2,
                                created_at=f"{day}T12:00:00+00:00")
        board = await stats.leaderboard()
        assert board[0]["streak"] == 3

        bob = await db.create_user("bob", "x")
        hands, actions = played_deal(seed=7)
        await _store_played(hands, actions, user_id=bob, seat=2,
                            created_at="2020-01-01T12:00:00+00:00")
        board = {r["name"]: r for r in await stats.leaderboard()}
        assert board["bob"]["streak"] == 0, "une série ancienne n'est pas en cours"
        assert board["bob"]["days"] == 1

    async def test_les_capots_sont_comptes_des_deux_cotes(self, clean_db):
        alice = await db.create_user("alice", "x")
        hands = [list(range(8 * i, 8 * i + 8)) for i in range(4)]
        for pts, seat in (((252, 0), 2), ((0, 252), 2)):
            gid = await db.create_game("play", 0, hands, {"2": "human"},
                                       human_seat=seat, user_id=alice)
            await db.complete_game(gid, pts[0], pts[1], {"value": 250, "team": 0},
                                   score_ns=1, score_ew=0)
        row = (await stats.leaderboard())[0]
        assert row["capots_for"] == 1
        assert row["capots_against"] == 1


class TestMesStats:
    async def test_les_taux_portent_leur_n_et_leur_intervalle(self, clean_db,
                                                              played_deal):
        alice = await db.create_user("alice", "x")
        for seed in range(10):
            hands, actions = played_deal(seed=seed)
            await _store_played(hands, actions, user_id=alice, seat=2)
        st = await stats.my_stats(alice)
        assert st["deals"] == 10
        assert st["won"]["n"] == 10
        assert st["won"]["lo"] <= st["won"]["pct"] <= st["won"]["hi"]
        assert st["margin"]["n"] == 10
        assert st["margin"]["median"] is not None
        # Prises et défenses partitionnent les donnes contractées.
        assert st["takes"]["n"] + st["defense"]["n"] <= 10

    async def test_un_joueur_sans_donne(self, clean_db):
        alice = await db.create_user("alice", "x")
        assert (await stats.my_stats(alice)) == {"deals": 0}

    async def test_wilson_reste_dans_les_bornes(self):
        """À n petit, un intervalle normal sortirait de [0, 100]."""
        p, lo, hi = stats.wilson(0, 5)
        assert p == 0.0 and lo == 0.0 and 0 < hi < 100
        p, lo, hi = stats.wilson(5, 5)
        assert p == 100.0 and hi == 100.0 and 0 < lo < 100
        assert stats.wilson(0, 0) == (None, None, None)


class TestTempo:
    """Une interruption n'est pas une donne longue."""

    async def _deals_at(self, uid, stamps):
        """Des donnes d'une même partie, aux instants donnés."""
        conn = await db.get_db()
        mid = await db.create_match("play", 1000, user_id=uid, human_seat=2)
        hands = [list(range(8 * i, 8 * i + 8)) for i in range(4)]
        for n, stamp in enumerate(stamps):
            gid = await db.create_game("play", 0, hands, {"2": "human"},
                                       human_seat=2, user_id=uid,
                                       match_id=mid, deal_no=n + 1)
            await db.complete_game(gid, 90, 72, {"value": 90, "team": 0},
                                   score_ns=172, score_ew=0)
            await conn.execute("UPDATE games SET created_at = ? WHERE id = ?",
                               (stamp, gid))
        await conn.commit()

    async def test_les_pauses_longues_sont_exclues_et_dites(self, clean_db):
        alice = await db.create_user("alice", "x")
        # Six donnes à 60 s d'intervalle, puis une reprise deux heures plus tard.
        stamps = [f"2026-08-01T12:{m:02d}:00+00:00" for m in range(0, 6)]
        stamps.append("2026-08-01T14:30:00+00:00")
        await self._deals_at(alice, stamps)
        tempo = (await stats.my_stats(alice))["tempo"]
        assert tempo["n"] == 5, "les cinq écarts d'une minute sont gardés"
        assert tempo["dropped"] == 1, "et la reprise à deux heures est écartée"
        assert tempo["median"] == 60

    async def test_sans_partie_il_n_y_a_pas_de_tempo(self, clean_db, played_deal):
        """Une donne isolée n'a pas de suivante : l'écart ne mesurerait que du
        temps passé ailleurs."""
        alice = await db.create_user("alice", "x")
        hands, actions = played_deal(seed=1)
        await _store_played(hands, actions, user_id=alice, seat=2)
        assert (await stats.my_stats(alice))["tempo"]["n"] == 0
