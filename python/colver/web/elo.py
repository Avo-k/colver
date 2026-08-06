"""Classement des joueurs et des bots.

**L'unité classée est la partie en 2000 points** (2026-08-03), pas la donne. Une
donne isolée ou une partie en 1000 restent jouables, analysables et partageables :
elles ne comptent simplement pas au classement. C'est le format des tournois
réels, c'est l'unité de l'arène — donc l'ancre d'un bot est une mesure directe au
lieu d'une conversion — et c'est le seul levier honnête qui élargisse l'échelle.

**Abandonner vaut défaite.** Sans ça, quitter quand on perd serait gratuit. Ce qui
rend la règle juste est qu'une partie interrompue **se reprend**
(`server._resume_match`) : l'abandon est un acte délibéré, pas un accident de
connexion.

**Les bots sont l'étalon, pas des joueurs.** Leur note est figée. Avant le
2026-08-03 ils dérivaient avec la population — Dédé était monté de 1000 à 1044,
pic 1119, uniquement parce que les humains perdent contre lui — et comme tout le
monde est mesuré contre eux, l'arrivée de joueurs plus faibles dévaluait en
silence les inscrits.

## La note n'est plus une mise à jour incrémentale (2026-08-05)

Elle est le **posterior exact** recalculé depuis le bilan complet, et publiée sous
sa forme **conservatrice** `mu - 2*sigma` (convention TrueSkill/Xbox). Trois
défauts mesurés motivaient le changement, tous invisibles à l'usage :

1. **Le classement ordonnait par inexpérience.** Corrélation de Spearman entre
   « parties jouées » et « note affichée » : **−0,89** sur la base de prod du
   2026-08-05. Tout le monde partait de 1000, les humains gagnent ~24 % de leurs
   parties contre Dédé, donc tout le monde descendait — et le tableau classait
   surtout qui avait fait le moins de chemin. Le 1er avait joué **une** partie et
   l'avait perdue ; le meilleur bilan réel (4/12) était **6e sur 7**.
2. **Le nombre affiché n'était pas l'estimation du niveau.** La récurrence K part
   de 1000 et converge lentement : simulé sur 20 000 tirages, un joueur réellement
   à 550 affiche encore **832 après 12 parties** (biais +282) et il lui faut
   ~300 parties pour arriver. Coller un « ± » autour de ce nombre-là aurait
   attaché un intervalle de MLE à un centre qui n'en est pas un.
3. **Le seuil d'affichage ne servait pas à ce qu'on croyait.** Il était présenté
   comme un seuil de précision, et il ne l'est pas : à 5 parties l'IC95 vaut
   encore ±609 Elo, masquer en dessous ne rend pas le reste fiable. Il **reste
   en place** (`MIN_RATED_MATCHES`) pour la seule raison qui tient — ne pas
   lister publiquement quelqu'un qui a essayé une partie — et non plus au titre
   d'une précision qu'il n'achète pas. Ce que la note conservatrice change, c'est
   qu'il n'est plus *structurellement* nécessaire : un joueur non confirmé se
   placerait tout seul en bas.

Avec `mu - 2*sigma`, un nouveau venu **entre par le bas et monte en jouant** — à
niveau constant, jouer réduit sigma donc remonte la note. C'est exactement
l'inverse du défaut n° 1, et ça vient de la structure, pas d'un correctif.

## Pourquoi mu0 et tau sont FIGÉS, et pas ajustés sur la population

Les estimer par Bayes empirique était la première idée, et c'est une régression.
Mesuré sur la base de prod : ré-estimer `mu0` à chaque partie fait bouger la note
**des autres joueurs** de +28 à +32 quand quelqu'un gagne une partie, et de −101 à
−150 à l'arrivée d'un joueur qui perd ses 20 premières.

C'est **exactement le défaut que `K_BOT = 0` a fermé**, remonté d'un étage : au
lieu que ce soit l'ancre du bot qui dérive avec la population, c'est le prior. Une
fois `mu0` et `tau` figés, le couplage tombe à **exactement zéro** — la note
devient une fonction pure du bilan personnel, donc publiable et vérifiable à la
main, et personne ne bouge sans jouer.

Le coût est déplacé, pas supprimé : `mu0` vieillira si la population se déplace.
C'est le même problème que l'ancre des bots et il a le même remède —
`ANCHOR_VERSION`, un décalage explicite et daté plutôt qu'un ajustement silencieux.
Signal supplémentaire qu'il s'agit bien d'une convention et non d'une mesure :
`mu0` ajusté vaut 560 à `tau = 2` mais 508 à `tau = 200`, et `tau` lui-même n'est
**pas identifiable** — le maximum de vraisemblance le place à 0 (les 7 bilans de
prod sont compatibles avec 7 joueurs identiques) et tout `tau` jusqu'à ~300 est
dans le bruit.

`tau = 300` plutôt que 150 pour une raison précise : l'ordre du classement est
inchangé de 100 à 300, mais un prior serré coûte ~2,4× en parties à un joueur
d'exception qui doit prouver son niveau. Figer `tau` bas reviendrait à inscrire
dans la constante « personne ici n'est fort », ce qui est vrai aujourd'hui et sera
faux au premier bon joueur qui s'inscrit.

## La marge compte, le binaire ne suffisait pas (2026-08-06)

Gagner 2000-1900 et gagner 2000-200 comptaient pareil. C'est jeter la seule
information qu'une partie produise en plus de son vainqueur, et elle est
abondante : l'écart-type des marges signées vaut **1047 points** sur la prod.
Le score d'une partie est donc `sigma(marge / MARGIN_SCALE)` au lieu de 1/0.

**Mesuré avant d'être écrit**, par leave-one-out apparié sur les 56 parties des
joueurs à au moins 4 parties — on estime le niveau depuis toutes les *autres*
parties, puis on prédit l'issue de celle qu'on a tenue à l'écart :

| | log-perte |
|---|---|
| pile ou face | 0,6931 |
| binaire 1/0 | 0,6770 |
| **marge** | **0,6567** |

Tout le pouvoir prédictif du classement binaire valait 0,016 nat au-dessus du
hasard ; la marge en ajoute 0,020, donc elle **double** l'information utile.
Positif aux trois échelles essayées (600, 1047, 1600), donc le résultat ne tient
pas au réglage exact de la constante. ⚠️ L'IC de ce test est **optimiste** : les
plis partagent presque toutes leurs données, donc les 56 observations ne sont pas
indépendantes. Le signe est solide, la taille de l'effet l'est moins.

**Ce que ça n'achète pas, et il ne faut pas le promettre** : l'incertitude
affichée ne bouge pas (sigma 143 → 141 sur le joueur le plus mesuré). C'est
structurel — la courbure d'une vraisemblance de Bernoulli ne dépend pas du score
observé, seulement de la probabilité prédite. La marge déplace le **centre** vers
le bon endroit ; resserrer l'**intervalle** demanderait une vraie vraisemblance
sur la marge (gaussienne / Thurstone), qui est un autre objet.

**Un décalage vers le haut, une fois, et il ne faut pas le corriger en réglant
la constante.** Adoucir un score le tire vers 1/2 ; la population gagne 28 % de
ses parties, donc son score moyen monte — mesuré, **+2,8 points de pourcentage**,
soit **+4 à +92 points d'affichage** selon le joueur (moyenne ~+35) au moment de
la bascule. **L'ordre du classement ne change pas.** La part commune de ce
décalage est un biais réel au sens « le niveau est défini par une probabilité de
victoire » ; la part qui *varie* d'un joueur à l'autre est précisément
l'information qu'on est venu chercher. Choisir `MARGIN_SCALE = 600` annulerait le
décalage sur la base d'aujourd'hui — et ce serait ajuster une constante sur la
population, donc rouvrir le couplage que `PRIOR_MEAN` / `PRIOR_SD` figés ont
fermé. L'échelle reste une propriété du **format** (l'écart-type de ses marges),
pas de qui joue.

**Le vainqueur reste l'autorité sur l'issue, la marge ne module que l'ampleur.**
`soft_score` prend le signe de `winner` et la magnitude de `|marge|` : une ligne
`matches` dont les points contrediraient le vainqueur donnerait une note de
mauvais signe, et ce serait un renversement silencieux. Corollaire, un **abandon**
garde 0/1 — le score au moment où l'on quitte la table n'est pas la marge d'une
partie jouée jusqu'au bout.

## Deux échelles, et une seule conversion

L'échelle **interne** est celle des mesures : les écarts entre bots viennent de
`arena h2h` et la constante Elo y vaut 400. L'échelle **d'affichage** en est une
transformée affine, appliquée en un seul endroit (`to_display`), choisie pour se
lire comme aux échecs : un nouveau joueur lit 800, un joueur confirmé de niveau
moyen lit 1600, Dédé 2200.

⚠️ **Ça ne crée aucune précision.** Multiplier les écarts multiplie le bruit à
l'identique : l'intervalle du mieux classé passe de ±253 à ±484 dans la même
opération. C'est un choix de lisibilité, à ne jamais présenter comme un gain de
fiabilité.

`rate_match` est idempotent (une ligne d'`elo_history` par partie × entité), ce qui
rend le backfill du démarrage sûr à chaque boot.
"""

import asyncio
import logging
import math

import numpy as np

import colver.web.database as db
import colver.web.match_state as match_state

logger = logging.getLogger(__name__)

# ===== Étalonnage figé =====================================================
#
# Version de l'étalonnage. À incrémenter — et à redocumenter — dès qu'un bot
# change de modèle, ou que `PRIOR_MEAN` / `PRIOR_SD` / l'affichage bougent. Le
# suffixe dit l'unité : une note « donne » et une note « partie » ne se comparent
# pas, et le passage de l'une à l'autre a multiplié tous les écarts par ~3,4.
ANCHOR_VERSION = "2026-08-post-match"

# Parties nécessaires pour **apparaître** au tableau. En dessous, l'entité est
# notée — son bilan s'accumule, sa note se construit — mais elle reste masquée.
#
# ⚠️ **Ce n'est pas un seuil de précision, et il ne faut pas le lire comme tel.**
# À 5 parties l'IC95 vaut encore ±609 Elo, soit plus que l'étendue entière du jeu
# de la carte : le franchir ne rend rien fiable. La raison est éditoriale — on ne
# publie pas le nom de quelqu'un qui a joué une partie et s'est arrêté.
#
# Depuis que la note est conservatrice (`mu - 2*sigma`), le seuil n'est plus
# *structurellement* nécessaire : un joueur non confirmé se range tout seul en
# bas du tableau. Le retirer resterait donc correct, et c'est un arbitrage
# produit, pas un arbitrage de mesure.
MIN_RATED_MATCHES = 5

# Seules les parties à cette cible comptent. Une donne isolée (`target = 0`, le
# défaut du site) et une partie en 1000 restent jouables et analysables.
#
# Ouvrir plus tard aux parties en 1000 est la décision la plus réversible du lot
# — il suffirait d'un poids `sqrt(1000/2000) = 0,71`, à la manière de FIBS au
# backgammon. À garder en réserve si le classement se révèle trop vide.
RATED_TARGET = 2000

# Prior sur le niveau d'un humain, échelle interne. CONVENTIONS, pas ajustements
# — voir l'en-tête : les ré-estimer couple les joueurs entre eux.
PRIOR_MEAN = 550.0
PRIOR_SD = 300.0

# Échelle de la marge d'une partie en 2000 points : l'écart-type des marges
# signées, **mesuré** sur les 58 parties classées non abandonnées de la prod au
# 2026-08-06 (médiane |marge| 954, p10 410, p90 1627, max 2138). Il valait 962
# sur 32 parties la veille : c'est une estimation qui bougera, à re-mesurer quand
# le volume aura triplé.
#
# ⚠️ **Cette constante est par format.** Une partie en 1000 points produit
# mécaniquement des marges plus petites (4,7 donnes en moyenne contre 10,2) :
# le jour où `RATED_TARGET` s'ouvrira, il faudra une échelle par cible, sans quoi
# les parties courtes seraient toutes lues comme serrées.
MARGIN_SCALE = 1047.0

# Note publiée = mu - CONSERVATISM * sigma. 2 plutôt que le 3 de TrueSkill : à
# 3 sigma un joueur à 12 parties afficherait une note négative sur l'échelle
# interne, et le tableau se lirait comme une punition de l'inexpérience.
CONSERVATISM = 2.0

# Transformation d'affichage, définie par deux points humains lisibles. Les bots
# atterrissent où les mesures les mettent (Dédé 2200, DouDou50 1973, Oracle 2480).
# Ancrer sur un BOT serait le piège : l'écart humain → DouDou50 vaut 280 en
# interne, soit plus que l'écart DouDou → Dédé (170), donc mettre DouDou à 1000
# enverrait les joueurs réels **sous zéro**.
DISPLAY_NEW = 800.0      # ce que lit un joueur sans aucune partie classée
DISPLAY_TYPICAL = 1600.0  # ce que lit un joueur confirmé de niveau moyen

_INTERNAL_NEW = PRIOR_MEAN - CONSERVATISM * PRIOR_SD
DISPLAY_SCALE = (DISPLAY_TYPICAL - DISPLAY_NEW) / (PRIOR_MEAN - _INTERNAL_NEW)
DISPLAY_OFFSET = DISPLAY_NEW - _INTERNAL_NEW * DISPLAY_SCALE

# Note figée de chaque bot, **échelle interne, unité de partie**.
#
# Dédé vaut 1000 par définition : c'est lui l'origine de l'échelle.
#
# DouDou est 170 points en dessous, mesuré par `arena h2h web_dede web_doudou`
# (36-14 sur 50 matchs, soit +164 Elo, IC95 [+58 ; +270] — précision modeste,
# mais sans commune mesure avec le « 988,6 sur 11 donnes » d'avant).
BOT_ELO = {
    "dede": 1000.0,
    "doudou": 830.0,
}

# Repli d'un bot non étalonné. Il le traite comme l'égal de Dédé — c'est le bon
# défaut (mieux vaut une hypothèse visible et fausse qu'un bot qui dérive) mais
# **tout bot ajouté doit passer par un h2h avant d'être assis en production**.
START_ELO = 1000.0

# Grille d'intégration du posterior. Large des deux côtés : un joueur qui perd
# ses 20 premières parties a un posterior qui déborde franchement sous le prior.
_GRID = np.arange(-2000.0, 4000.0, 1.0)
_PRIOR_LOGP = -0.5 * ((_GRID - PRIOR_MEAN) / PRIOR_SD) ** 2


def to_display(x):
    """Échelle interne (celle des mesures) → échelle affichée."""
    return DISPLAY_OFFSET + DISPLAY_SCALE * x


def from_display(x):
    return (x - DISPLAY_OFFSET) / DISPLAY_SCALE


def bot_elo(name):
    """Note figée d'un bot, échelle interne."""
    return BOT_ELO.get(name, START_ELO)


def posterior(record):
    """(moyenne, sigma) a posteriori sur l'échelle interne.

    `record` : itérable de `(score, partner_elo, opp_elo)`, une entrée par partie
    classée, toutes sur l'échelle interne. `score` vaut 1 (victoire) ou 0.

    La vraisemblance est celle qu'`elo.py` a toujours utilisée : l'équipe vaut la
    moyenne de ses deux joueurs, donc l'écart d'équipe ne porte que **la moitié**
    de l'écart individuel — c'est la dilution par le partenaire, et c'est la
    raison de fond pour laquelle il faut des centaines de parties en solo.
    """
    lp = _PRIOR_LOGP.copy()
    for score, partner, opp in record:
        p = 1.0 / (1.0 + 10 ** ((opp - (_GRID + partner) / 2) / 400.0))
        np.clip(p, 1e-15, 1 - 1e-15, out=p)
        lp += score * np.log(p) + (1.0 - score) * np.log1p(-p)
    w = np.exp(lp - lp.max())
    w /= w.sum()
    mean = float((_GRID * w).sum())
    return mean, float(math.sqrt(float(((_GRID - mean) ** 2 * w).sum())))


def soft_score(winner, points_ns, points_ew):
    """Score de la partie **pour N-S**, dans ]0, 1[, modulé par la marge.

    Le signe vient de `winner`, jamais des points : ceux-ci ne servent qu'à
    l'ampleur. Une ligne `matches` incohérente (points d'un camp, victoire de
    l'autre) donnerait sinon une note de mauvais signe, en silence.
    """
    margin = abs(int(points_ns or 0) - int(points_ew or 0))
    s = 1.0 / (1.0 + 10 ** (-margin / MARGIN_SCALE))
    return s if winner == 0 else 1.0 - s


def note_of(mean, sd):
    """Note publiée : la borne basse, sur l'échelle d'affichage."""
    return to_display(mean - CONSERVATISM * sd)


_lock = asyncio.Lock()  # serialize read-modify-write across concurrent matches


def _seat_entities(game, player_rows):
    """Chaque siège → une entité notable `(kind, ref)`, ou None si la partie
    n'est pas notable (un siège anonyme, un bot inconnu).

    Les sièges d'une partie ne changent pas d'une donne à l'autre (seul le
    donneur tourne), donc n'importe quelle donne rend le même tableau.
    """
    agents = game["agents"]
    humans = {row["seat"]: row["user_id"] for row in player_rows}
    if game["mode"] == "play" and game["human_seat"] is not None:
        if game.get("user_id") is None:
            return None  # anonymous solo game
        humans[game["human_seat"]] = game["user_id"]

    seats = []
    for s in range(4):
        if s in humans:
            seats.append(("user", str(humans[s])))
        else:
            bot = agents.get(str(s))
            if not bot or bot == "human":
                return None
            seats.append(("bot", bot))
    return seats


async def _match_deals(conn, match_id):
    """Les donnes saines d'une partie, dans l'ordre, avec les sièges de chacune.

    Rend `[(entités, donne)]`. Une donne dont les sièges ne sont pas notables
    est écartée : elle ne peut renseigner ni le prorata ni les compteurs.
    """
    rows = await conn.execute_fetchall(
        "SELECT id FROM games WHERE match_id = ? AND is_complete = 1 AND invalid = 0 "
        "ORDER BY deal_no",
        (match_id,),
    )
    out = []
    for (game_id,) in rows:
        game = await db.get_game(game_id)
        if game is None:
            continue
        players = await conn.execute_fetchall(
            "SELECT seat, user_id FROM game_players WHERE game_id = ?", (game_id,))
        seats = _seat_entities(game, [dict(r) for r in players])
        if seats is not None:
            out.append((seats, game))
    return out


async def _match_seats(conn, match_id):
    """Entités des quatre sièges d'une partie, lues sur sa première donne saine."""
    deals = await _match_deals(conn, match_id)
    return deals[0][0] if deals else None


def _miss_runs(deals):
    """Reconstruire la pendule de chaque siège depuis le journal des donnes.

    Rend `(plus longue série, cumul)` par siège. **Aucune colonne à ajouter** :
    le drapeau `auto` d'une action *est* la trace d'un temps écoulé, écrit au
    moment où il l'a été, et c'est déjà lui que la revue d'analyse lit pour ne
    pas attribuer le coût d'un coup à quelqu'un qui ne l'a pas choisi. Le solo
    ne le pose que sur son message d'écho, jamais dans le journal : ici il ne
    peut donc désigner qu'une table partagée.

    Les coups joués par le bot **après** un forfait ne portent pas ce drapeau —
    le siège est devenu un siège de bot ordinaire. La plus longue série est donc
    bien atteinte avant la reprise, et elle y reste.
    """
    runs = [0] * 4
    best = [0] * 4
    total = [0] * 4
    for _seats, game in deals:
        for action in game["actions"]:
            seat = int(action["player"])
            if action.get("auto"):
                runs[seat] += 1
                total[seat] += 1
                best[seat] = max(best[seat], runs[seat])
            else:
                runs[seat] = 0
    return best, total


async def _seat_levels(conn, deals, levels):
    """Le niveau de chaque siège, **au prorata des donnes**.

    Un siège peut changer de main en cours de partie : après un forfait, l'IA le
    reprend jusqu'au bout. Le partenaire de l'absent n'a alors pas joué toute la
    partie avec la même personne, et les adversaires n'ont pas eu la même tâche
    — donc ni `partner_elo` ni `opp_elo` ne peuvent être un chiffre unique lu
    sur la première donne. On pondère par le nombre de donnes.

    `_seat_entities` énonce justement l'hypothèse inverse (« les sièges d'une
    partie ne changent pas d'une donne à l'autre ») : c'est elle que le forfait
    casse, et elle ne tenait que faute de mécanisme pour la casser.
    """
    seat_levels = [0.0] * 4
    for seat in range(4):
        acc = 0.0
        for seats, _game in deals:
            ent = seats[seat]
            if ent not in levels:
                levels[ent] = await _level_internal(conn, ent)
            acc += levels[ent]
        seat_levels[seat] = acc / len(deals)
    return seat_levels


async def _losing_team(conn, match_id, owner_id):
    """Camp qui perd une partie abandonnée : celui de qui a abandonné.

    `db.abandon_match` n'accepte que le propriétaire d'une partie solo, donc
    `matches.user_id` désigne bien l'abandonnant. Si on ne sait pas le placer, on
    ne note pas — mieux vaut une partie non notée qu'une défaite attribuée au
    hasard.
    """
    if owner_id is None:
        return None
    rows = await conn.execute_fetchall(
        "SELECT human_seat FROM games WHERE match_id = ? AND user_id = ? "
        "AND human_seat IS NOT NULL ORDER BY deal_no LIMIT 1",
        (match_id, owner_id),
    )
    if not rows or rows[0][0] is None:
        return None
    return rows[0][0] % 2


async def _level_internal(conn, ent):
    """Niveau courant d'une entité, échelle interne — pour servir de partenaire
    ou d'adversaire dans la vraisemblance d'un autre joueur.

    Un bot rend son ancre. Un humain rend sa moyenne a posteriori courante, ou le
    prior s'il n'a rien joué.

    ⚠️ **Approximation, et elle ne mord pas aujourd'hui.** Le posterior exact
    d'une partie humain-contre-humain serait *joint* sur les quatre sièges. Les
    33 parties classées de la prod sont toutes en solo (un humain, trois bots),
    donc ce chemin ne s'emprunte pas encore. Le jour où le salon sera classé, il
    faudra soit itérer cette passe jusqu'au point fixe, soit passer à un vrai
    modèle joint.
    """
    if ent[0] == "bot":
        return bot_elo(ent[1])
    rows = await conn.execute_fetchall(
        "SELECT level FROM elo_ratings WHERE kind = ? AND ref = ?", ent)
    if not rows or rows[0][0] is None:
        return PRIOR_MEAN
    return from_display(rows[0][0])


async def _record(conn, ent):
    """Le bilan complet d'une entité, dans l'ordre où il s'est constitué."""
    rows = await conn.execute_fetchall(
        "SELECT e.score, e.partner_elo, e.opp_elo FROM elo_history e "
        "JOIN matches m ON m.id = e.match_id "
        "WHERE e.kind = ? AND e.ref = ? ORDER BY m.created_at",
        ent,
    )
    return [(r[0], r[1], r[2]) for r in rows]


async def _store(conn, ent, record):
    """Recalcule le posterior d'une entité depuis son bilan et l'écrit."""
    if ent[0] == "bot":
        # Un étalon ne se calcule pas : il EST l'échelle. On garde quand même sa
        # ligne pour que le tableau se lise d'un seul SELECT, et son compteur de
        # parties pour l'afficher.
        note = level = to_display(bot_elo(ent[1]))
        sigma = 0.0
    else:
        mean, sd = posterior(record)
        note, level, sigma = note_of(mean, sd), to_display(mean), DISPLAY_SCALE * sd
    await conn.execute(
        "INSERT INTO elo_ratings (kind, ref, elo, level, sigma, games, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(kind, ref) DO UPDATE SET "
        "  elo = excluded.elo, level = excluded.level, sigma = excluded.sigma, "
        "  games = excluded.games, updated_at = excluded.updated_at",
        (*ent, round(note, 2), round(level, 2), round(sigma, 2),
         len(record), db._now()),
    )
    return note


async def rate_match(match_id):
    """Note une partie terminée en 2000 points. Idempotent ; ne lève jamais."""
    try:
        async with _lock:
            return await _rate_match_locked(match_id)
    except Exception:
        logger.exception("rating of match %s failed", match_id)
        return False


async def _rate_match_locked(match_id):
    conn = await db.get_db()
    rows = await conn.execute_fetchall(
        "SELECT 1 FROM elo_history WHERE match_id = ? LIMIT 1", (match_id,))
    if rows:
        return False  # already rated

    rows = await conn.execute_fetchall(
        "SELECT target, is_complete, winner, abandoned, user_id, points_ns, points_ew "
        "FROM matches WHERE id = ?",
        (match_id,),
    )
    if not rows:
        return False
    target, is_complete, winner, abandoned, owner_id, points_ns, points_ew = rows[0]
    if target != RATED_TARGET or not is_complete:
        return False

    # Une partie dont une donne est en quarantaine a un score cumulé faux : elle
    # ne doit pas entrer au classement (même règle que `integrity.scan`).
    bad = await conn.execute_fetchall(
        "SELECT 1 FROM games WHERE match_id = ? AND invalid = 1 LIMIT 1", (match_id,))
    if bad:
        return False

    if abandoned:
        loser = await _losing_team(conn, match_id, owner_id)
        if loser is None:
            return False
        # Pas de marge exploitable : le score d'une partie quittée n'est pas
        # celui d'une partie jouée jusqu'au bout.
        score_ns = 0.0 if loser == 0 else 1.0
    else:
        if winner is None:
            return False
        score_ns = soft_score(winner, points_ns, points_ew)

    deals = await _match_deals(conn, match_id)
    if not deals:
        return False
    seats = deals[0][0]

    levels = {}
    # Les niveaux **par siège**, pondérés par les donnes : après un forfait, le
    # siège change de main en cours de partie.
    seat_levels = await _seat_levels(conn, deals, levels)
    # Ce que le temps de jeu a coûté, relu depuis le journal.
    best_run, misses = _miss_runs(deals)

    # Une entité peut tenir plusieurs sièges (un bot en tient trois en solo). On
    # ne l'inscrit qu'une fois, sur son premier siège : `games` compte des
    # **parties**, plus des sièges — Dédé affichait 2 540 pour 881 donnes jouées.
    seen = set()
    for seat, ent in enumerate(seats):
        if ent in seen:
            continue
        seen.add(ent)
        team = seat % 2
        score = score_ns if team == 0 else 1.0 - score_ns

        # Les deux sanctions du temps de jeu, et leur ordre. Un siège qui a
        # forfait dépasse forcément les deux seuils : c'est la **défaite** qui
        # l'emporte, sinon partir serait gratuit — et ne pas compter la partie
        # serait précisément le moyen de sortir sans rien payer d'une partie mal
        # engagée. La sanction faible ne doit pas absorber la forte.
        if ent[0] == "user" and best_run[seat] >= match_state.MISSES_TO_FORFEIT:
            score = 0.0
        elif ent[0] == "user" and misses[seat] >= match_state.MISSES_TOTAL_UNRATED:
            # Laisser filer la pendule, c'est déléguer ses décisions au bot : la
            # partie cesse de compter **pour ce joueur-là**. Les trois autres
            # gardent la leur — leur résultat est légitime, et le posterior de
            # chaque entité se recalcule depuis son propre bilan, donc en
            # omettre une ligne ne déséquilibre rien.
            logger.info("partie %s : siège %s hors classement (%d coups au temps)",
                        match_id, seat, misses[seat])
            continue

        partner = seat_levels[seat ^ 2]
        opp = (seat_levels[(seat + 1) % 4] + seat_levels[(seat + 3) % 4]) / 2

        before = await _record(conn, ent)
        note_before = (note_of(*posterior(before)) if ent[0] == "user"
                       else to_display(bot_elo(ent[1])))
        await conn.execute(
            "INSERT INTO elo_history "
            "(match_id, kind, ref, score, partner_elo, opp_elo, delta, elo_after) "
            "VALUES (?, ?, ?, ?, ?, ?, 0, 0)",
            (match_id, *ent, score, partner, opp),
        )
        note_after = await _store(conn, ent, before + [(score, partner, opp)])
        # `delta` / `elo_after` gardent leurs noms et changent de sens : ce n'est
        # plus le pas d'une récurrence K mais **le déplacement de la note publiée
        # causé par cette partie**, ce qui est ce que « combien m'a rapporté ce
        # match » veut dire. Les deux consommateurs (`list_matches`, `get_match`)
        # n'ont rien à changer.
        await conn.execute(
            "UPDATE elo_history SET delta = ?, elo_after = ? "
            "WHERE match_id = ? AND kind = ? AND ref = ?",
            (round(note_after - note_before, 2), round(note_after, 2), match_id, *ent),
        )
    await conn.commit()
    return True


async def backfill():
    """Note toutes les parties terminées au format classé, la plus vieille d'abord."""
    conn = await db.get_db()
    rows = await conn.execute_fetchall(
        "SELECT id FROM matches WHERE is_complete = 1 AND target = ? "
        "AND id NOT IN (SELECT DISTINCT match_id FROM elo_history) "
        "ORDER BY created_at",
        (RATED_TARGET,),
    )
    rated = 0
    for (match_id,) in rows:
        if await rate_match(match_id):
            rated += 1
    if rated:
        logger.info("backfill: rated %d match(es)", rated)


async def get_rating(kind, ref):
    """Note, niveau estimé et incertitude d'une entité, échelle d'affichage."""
    conn = await db.get_db()
    rows = await conn.execute_fetchall(
        "SELECT elo, level, sigma, games FROM elo_ratings WHERE kind = ? AND ref = ?",
        (kind, str(ref)))
    if kind == "bot":
        anchor = to_display(bot_elo(str(ref)))
        return {"elo": anchor, "level": anchor, "uncertainty": 0.0,
                "games": rows[0][3] if rows else 0}
    if not rows:
        mean, sd = posterior([])
        return {"elo": round(note_of(mean, sd), 1), "level": round(to_display(mean), 1),
                "uncertainty": round(DISPLAY_SCALE * sd * 2, 1), "games": 0}
    elo_, level, sigma, games = rows[0]
    return {"elo": round(elo_, 1), "level": round(level, 1),
            "uncertainty": round((sigma or 0.0) * 2, 1), "games": games}


async def leaderboard():
    """Entités classées, meilleure d'abord.

    Deux colonnes, parce qu'elles disent deux choses : la **note** est ce qu'on
    peut prouver (elle sert au tri), le **niveau** est la meilleure estimation
    avec son incertitude. L'écart entre les deux est le prix de l'inexpérience,
    et il se referme en jouant.

    Un humain n'apparaît qu'à partir de `MIN_RATED_MATCHES` parties. En dessous
    sa note existe et se construit ; c'est `standing()` qui la lui rend, pour que
    la page puisse dire « encore 3 parties » plutôt que de le faire disparaître
    sans explication. Les bots sont toujours là — ce sont les étalons, leur note
    ne dépend d'aucune partie jouée.
    """
    conn = await db.get_db()
    rows = await conn.execute_fetchall(
        "SELECT r.kind, r.ref, r.elo, r.level, r.sigma, r.games, u.username "
        "FROM elo_ratings r "
        "LEFT JOIN users u ON r.kind = 'user' AND u.id = CAST(r.ref AS INTEGER) "
        "ORDER BY r.elo DESC",
    )
    return [
        {
            "kind": kind, "ref": ref,
            "elo": round(elo_, 1),
            "level": round(level, 1) if level is not None else round(elo_, 1),
            "uncertainty": round((sigma or 0.0) * 2, 1),
            "games": games,
            "name": username if kind == "user" else ref,
        }
        for kind, ref, elo_, level, sigma, games, username in rows
        if kind == "bot" or games >= MIN_RATED_MATCHES
    ]


async def standing(kind, ref):
    """État du classement d'une entité, **y compris quand elle n'est pas classée**.

    C'est ce qui permet à la page de dire « encore 3 parties, note provisoire
    1120 » au lieu de laisser un joueur se demander pourquoi il ne se voit pas.
    Lu par `/api/me`, donc par Classement, Mes stats et Compte.
    """
    r = await get_rating(kind, ref)
    games = r["games"]
    return {
        **r,
        "ranked": kind == "bot" or games >= MIN_RATED_MATCHES,
        "needed": MIN_RATED_MATCHES,
        "remaining": max(0, MIN_RATED_MATCHES - games) if kind == "user" else 0,
    }
