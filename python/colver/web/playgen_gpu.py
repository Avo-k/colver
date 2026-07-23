"""Client du sidecar d'inférence GPU playgen (``playgen_gpu_server`` Rust).

Le web tourne dans une VM sans GPU ; le sidecar tourne sur l'hôte (3090) et
échantillonne les mondes playgen en batch (~500 mondes/s contre ~10/s en CPU).
Toutes les fonctions retournent ``None`` en cas d'indisponibilité (sidecar
éteint, timeout, GPU occupé) — l'appelant retombe alors sur le chemin CPU
PyO3 existant, qui produit des mondes de la même distribution.

Activation : variable d'environnement ``COLVER_PLAYGEN_GPU_URL``
(ex. ``http://192.168.1.23:8003``). Non définie = désactivé, zéro overhead.
"""

import json
import os
import urllib.request

GPU_URL = os.environ.get("COLVER_PLAYGEN_GPU_URL", "").rstrip("/")
TIMEOUT = float(os.environ.get("COLVER_PLAYGEN_GPU_TIMEOUT", "6"))


def enabled() -> bool:
    return bool(GPU_URL)


def _post(path: str, payload: dict):
    if not GPU_URL:
        return None
    try:
        req = urllib.request.Request(
            GPU_URL + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def _payload(dealer, initial_hands, actions, observer, n_worlds, temperature):
    """initial_hands: 4 listes de cartes (0-31). actions: [(player, action)]."""
    return {
        "dealer": int(dealer),
        "hands": [sum(1 << int(c) for c in h) for h in initial_hands],
        "actions": [[int(p), int(a)] for p, a in actions],
        "observer": int(observer),
        "n_worlds": int(n_worlds),
        "temperature": float(temperature),
    }


def _mask_to_cards(mask: int) -> list:
    return [c for c in range(32) if mask & (1 << c)]


def beliefs(dealer, initial_hands, actions, observer, n_worlds=200, temperature=0.8):
    """Marginales playgen [4][32] (même format que Env.get_playgen_beliefs)."""
    resp = _post("/beliefs", _payload(dealer, initial_hands, actions, observer, n_worlds, temperature))
    if not resp or "marginals" not in resp:
        return None
    return resp["marginals"]


def auction_deals(dealer, initial_hands, actions, observer, n_worlds, temperature=1.0):
    """Donnes complètes conditionnées à l'enchère en cours — deals[monde][siège]
    = liste de cartes (même format que Env.playgen_sample_auction_deals)."""
    resp = _post("/auction_deals", _payload(dealer, initial_hands, actions, observer, n_worlds, temperature))
    if not resp or "hands" not in resp:
        return None
    return [[_mask_to_cards(seat_mask) for seat_mask in world] for world in resp["hands"]]


def play_worlds(dealer, initial_hands, actions, observer, n_worlds=24, temperature=0.8):
    """Mondes de jeu (mains restantes, bitmasks u32 par siège) pour injection
    dans IS-DD via Env.dede_inject_worlds."""
    resp = _post("/play_worlds", _payload(dealer, initial_hands, actions, observer, n_worlds, temperature))
    if not resp or "hands" not in resp:
        return None
    return resp["hands"]
