"""Cache en base des simulations d'analyse.

`annonces_doudou` et `annonces_sim` (page Annonces) et `card_analysis`
(`/analyse/jeu`) sont les calculs les plus chers du site — mesuré à l'écran :
11,5 s pour les 1 000 déroulements d'une évaluation d'annonce — et étaient
jusqu'ici entièrement jetés à la fin de la requête. Ce module leur donne le même
traitement que `analysis` / `agent_review` : une clé, une version, un blob.

**Le chemin chaud de la page Annonces est `annonces_doudou`, pas
`annonces_sim`.** L'Oracle tourne en WASM **dans le navigateur** (`evalLocal` →
`wasmBridge.runOracleSim`), donc le serveur ne voit d'habitude que les 1 000
donnes jouées par Dédé ; `annonces_sim` n'est que le repli quand le WASM échoue.
C'est aussi `annonces_doudou` que déclenche « Analyser une autre annonce ». Ne
cacher que `annonces_sim` n'aurait servi presque personne — vérifier quel
message le client envoie avant de mesurer un coût serveur sur cette page.

Ce qui ne se devine pas non plus :

**La clé porte la position, pas la requête.** Pour une carte on hache
`(dealer, mains initiales, actions jusqu'à l'index)` et non le CFN reçu : deux
CFN qui décrivent la même position doivent partager leur entrée, et un CFN
diffère par des blancs ou par des coups *postérieurs* à la position analysée,
qui n'entrent dans aucun calcul. Pour une annonce, la main triée et les
enchères précédentes.

**Le score de partie n'est pas dans la clé d'une annonce**, alors qu'il est
dans celle de la barre latérale côté client (`handSig`). Ce n'est pas un oubli
et les deux ont raison : le score gouverne le panneau de Q du réseau v6, qui
est *score-aware* et se calcule ailleurs (`bid_eval`), mais la simulation elle
ne le voit pas — l'Oracle est du double-mort, DouDou50 est aveugle au score, et
`_run_doudou_sim_with_hands` ne pose aucun `set_match_scores`. L'ajouter à la
clé fragmenterait le cache pour un résultat identique au bit près.

**Le cache est partagé entre tous les joueurs.** Une position n'appartient à
personne : elle est soit tapée dans un formulaire, soit tirée d'une donne
terminée, qui est publique. Rien d'identifiant n'entre dans la clé, donc deux
joueurs qui étudient la même main paient une fois.

**Un résultat dégradé ne s'écrit pas.** Sans sidecar les mondes retombent sur
un mélange uniforme, sans poids DouDou50 la seconde phase ne tourne pas : ces
résultats sont justes mais moins bons, et les figer les rendrait permanents
bien après le retour du composant manquant. C'est la panne qu'`analysis.
_is_fresh` a dû rattraper après coup pour playgen, et celle qui a laissé une
revue de bot calculée sidecar éteint en cache pour toujours (`1669e7c`). Ici la
règle est prise à l'écriture, ce qui est plus simple à tenir : on n'écrit que
ce qui est complet.
"""

import hashlib
import json
import logging

from colver.web import database as db

logger = logging.getLogger(__name__)

# ⚠️ À bumper dès que le résultat change **sans que la clé change** : nouveau
# barème, changement du solveur (le départage des ex æquo du 2026-08-03 en est
# un — il change la carte désignée par l'Oracle), nouveaux poids servis sous le
# même chemin, ou budget de simulation retouché. `REVIEW_VERSION` a manqué ce
# rendez-vous une fois ; le seul garde-fou est de le lire ici avant de toucher
# à l'un de ces quatre.
BID_SIM_VERSION = 1
# v2 : le blob porte la suite en jeu parfait depuis la position (`line`).
CARD_SIM_VERSION = 2
DOUDOU_SIM_VERSION = 1

KIND_BID = "bid_sim"
KIND_CARD = "card_sim"
# `annonces_doudou` est un genre à part et **pas** la phase 2 de `annonces_sim`,
# malgré un tableau identique à l'écran : celle-ci déroule des mondes playgen,
# celui-là des mélanges uniformes (`_run_single_doudou_sim`). Deux échantillons
# différents, donc deux entrées — les confondre servirait un tableau pour
# l'autre selon le chemin qu'a pris le client.
KIND_DOUDOU = "doudou_sim"

# Le client n'affiche que 10 donnes d'exemple, tirées au hasard parmi celles
# qui ont servi. En garder 200 en base multiplierait le blob par cinq pour un
# aperçu ; on en garde donc juste de quoi remplir la vignette.
SAMPLE_KEEP = 10


def _digest(payload):
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def bid_key(hand, prior_actions, oracle_sims, doudou_sims):
    """La clé d'une analyse d'annonce.

    Les deux budgets en font partie : ce sont des constantes du client, et
    servir un échantillon de 200 mondes à qui en demande 1 000 rendrait des
    intervalles faux sans que rien ne le signale. Le score de partie, lui, n'y
    est pas — voir l'en-tête du module.
    """
    return _digest({
        "hand": sorted(int(c) for c in hand),
        "prior": [int(a) for a in prior_actions],
        "oracle": int(oracle_sims),
        "doudou": int(doudou_sims),
    })


def doudou_key(hand, prior_actions, forced_action, num_sims):
    """La clé du « Jeu réel » seul — le chemin le plus emprunté de la page.

    L'Oracle tourne en WASM dans le navigateur (`evalLocal`), donc le serveur ne
    voit que cette moitié-là ; c'est aussi ce que déclenche « Analyser une autre
    annonce », d'où `forced_action` dans la clé : forcer 110♦ et forcer 120♠ sur
    la même main sont deux questions.
    """
    return _digest({
        "hand": sorted(int(c) for c in hand),
        "prior": [int(a) for a in prior_actions],
        "forced": None if forced_action is None else int(forced_action),
        "sims": int(num_sims),
    })


def card_key(dealer, initial_hands, actions, idx):
    """La clé d'une analyse de carte : la position, et rien d'autre.

    Les actions **postérieures** à `idx` sont exclues volontairement. Elles ne
    servent qu'à l'affichage (« la carte réellement jouée »), qui est recalculé
    à chaque service, et les inclure diviserait le cache entre des positions
    identiques venues de donnes différentes.
    """
    return _digest({
        "dealer": int(dealer),
        "hands": [sorted(int(c) for c in h) for h in initial_hands],
        "prefix": [int(a) for a in actions[:idx]],
    })


async def get(kind, key, version):
    """L'entrée en cache, ou None. Ne lève jamais : un cache est un confort."""
    try:
        return await db.get_sim_cache(kind, key, version)
    except Exception:  # noqa: BLE001 — une panne de cache ne perd pas la page
        logger.exception("lecture du cache d'analyse : %s", kind)
        return None


async def put(kind, key, version, payload):
    """Écrire une entrée. Ne lève jamais, pour la même raison."""
    try:
        await db.put_sim_cache(kind, key, version, json.dumps(payload))
    except Exception:  # noqa: BLE001
        logger.exception("écriture du cache d'analyse : %s", kind)


def bid_cacheable(blob, oracle_sims, doudou_sims, *, doudou_expected):
    """Un résultat d'annonce mérite-t-il d'être gardé ?

    Non si les mondes ne viennent pas de playgen (« uniform » ou « mixte »
    veulent dire sidecar absent ou génération en échec), non si l'Oracle n'est
    pas allé au bout, non si la phase Dédé était attendue et manque.
    """
    if blob.get("worlds_source") != "playgen":
        return False
    if blob.get("completed") != oracle_sims:
        return False
    if doudou_expected:
        dd = blob.get("doudou")
        if not dd or dd.get("completed") != doudou_sims:
            return False
    return True


def card_cacheable(blob, n_worlds, *, doudou_expected):
    """Idem pour une carte : mondes playgen, échantillon complet, avis au complet."""
    if blob.get("worlds_source") != "playgen":
        return False
    if blob.get("completed") != n_worlds:
        return False
    op = blob.get("opinions") or {}
    if "oracle" not in op or "isdd" not in op:
        return False
    if doudou_expected and "doudou" not in op:
        return False
    return True
