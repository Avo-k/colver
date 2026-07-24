"""Client du sidecar d'inférence GPU playgen (``playgen_gpu_server`` Rust).

Ne sert plus qu'aux **pages d'analyse** (marginales, donnes conditionnées à
l'enchère). Les mondes IS-DD ne passent plus par ici : l'agent Rust parle au
sidecar lui-même (``colver_core::worlds::SidecarWorldSource``), ce qui est
précisément ce qui empêche le web et l'arène de jouer des agents différents.

Le web tourne dans une VM sans GPU ; le sidecar tourne sur l'hôte (3090) et
échantillonne en batch (~500 mondes/s contre ~10/s en CPU). Toutes les
fonctions retournent ``None`` en cas d'indisponibilité (sidecar éteint,
timeout, GPU occupé) — l'appelant retombe alors sur le chemin CPU PyO3.

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
    """Marginales playgen [4][32] (même format que Analyst.marginals)."""
    resp = _post("/beliefs", _payload(dealer, initial_hands, actions, observer, n_worlds, temperature))
    if not resp or "marginals" not in resp:
        return None
    return resp["marginals"]


def auction_deals(dealer, initial_hands, actions, observer, n_worlds, temperature=1.0):
    """Donnes complètes conditionnées à l'enchère en cours — deals[monde][siège]
    = liste de cartes (même format que Analyst.auction_deals)."""
    resp = _post("/auction_deals", _payload(dealer, initial_hands, actions, observer, n_worlds, temperature))
    if not resp or "hands" not in resp:
        return None
    return [[_mask_to_cards(seat_mask) for seat_mask in world] for world in resp["hands"]]
