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
(ex. ``http://gpu-host:8003``). Non définie = désactivé, zéro overhead.
"""

import json
import logging
import os
import time
import urllib.request

logger = logging.getLogger(__name__)

GPU_URL = os.environ.get("COLVER_PLAYGEN_GPU_URL", "").rstrip("/")
TIMEOUT = float(os.environ.get("COLVER_PLAYGEN_GPU_TIMEOUT", "6"))

# Sonde de disponibilité : court, et mis en cache. `/health` doit rester
# instantané, mais « une URL est configurée » n'est pas « le sidecar répond » —
# c'est justement l'écart qui a laissé la prod tourner sans playgen sans que
# rien ne le dise. On sonde donc pour de vrai, avec un délai serré et un cache
# assez court pour être utile et assez long pour qu'un tableau de bord qui
# rafraîchit ne martèle pas le GPU.
PROBE_TIMEOUT = float(os.environ.get("COLVER_PLAYGEN_GPU_PROBE_TIMEOUT", "1.5"))
PROBE_TTL = float(os.environ.get("COLVER_PLAYGEN_GPU_PROBE_TTL", "30"))

_probe_cache = None  # (instant monotone, résultat)

# Empreinte des sources playgen/engine de *ce* binaire. Le sidecar publie la
# sienne ; deux valeurs identiques disent que les deux ont été construits sur le
# même code. C'est le seul contrôle automatique de la fraîcheur du sidecar, qui
# se déploie à la main et que le webhook ne touche pas.
try:
    from colver._colver import PLAYGEN_SURFACE as _OUR_SURFACE
except ImportError:  # binding trop ancien pour porter la constante
    _OUR_SURFACE = None


def _freshness(remote_surface):
    """`(fresh, detail)` — l'écart de code entre ce conteneur et le sidecar.

    Trois états, et **« inconnu » n'est pas « périmé »** : un sidecar d'avant
    cette fonctionnalité ne publie pas de `surface`, et le crier périmé
    apprendrait aux lecteurs à ignorer le champ. On ne conclut que quand les
    deux côtés savent répondre.
    """
    if _OUR_SURFACE is None or not remote_surface:
        return None, "inconnue (un des deux côtés ne la publie pas)"
    if remote_surface == _OUR_SURFACE:
        return True, _OUR_SURFACE
    return False, (
        f"sidecar {remote_surface} ≠ web {_OUR_SURFACE} — "
        "sidecar construit sur d'autres sources playgen/engine, "
        "le rebâtir (docs/belief/playgen.md)"
    )


def enabled() -> bool:
    return bool(GPU_URL)


def probe(force=False):
    """État réel du sidecar : `{configured, reachable, detail, age_s}`.

    Synchrone et bornée par `PROBE_TIMEOUT` — appelable depuis la boucle
    asyncio sans la bloquer plus longtemps que ça, mais préférer
    `asyncio.to_thread` sur un chemin sensible à la latence.
    """
    global _probe_cache
    if not GPU_URL:
        return {"configured": False, "reachable": False,
                "detail": "COLVER_PLAYGEN_GPU_URL non définie", "age_s": 0.0,
                "fresh": None, "surface": "sidecar non configuré"}
    now = time.monotonic()
    if not force and _probe_cache is not None and now - _probe_cache[0] < PROBE_TTL:
        cached = dict(_probe_cache[1])
        cached["age_s"] = round(now - _probe_cache[0], 1)
        return cached
    try:
        with urllib.request.urlopen(GPU_URL + "/health", timeout=PROBE_TIMEOUT) as resp:
            body = json.loads(resp.read())
        fresh, surface = _freshness(body.get("surface"))
        result = {"configured": True, "reachable": True,
                  "detail": f"model {body.get('model')}, "
                            f"max_worlds {body.get('max_worlds')}",
                  "fresh": fresh, "surface": surface}
    except Exception as e:  # noqa: BLE001 — toute panne se rapporte pareil
        # Injoignable : la fraîcheur n'est pas « fausse », elle est indécidable.
        result = {"configured": True, "reachable": False,
                  "detail": f"{type(e).__name__}: {e}",
                  "fresh": None, "surface": "sidecar injoignable"}
    _probe_cache = (now, result)
    out = dict(result)
    out["age_s"] = 0.0
    return out


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


def play_worlds(dealer, initial_hands, actions, observer, n_worlds, temperature=1.0):
    """Mondes d'une position de **jeu** — worlds[monde][siège] = cartes qu'il
    lui **reste** (même format que Analyst.play_worlds).

    Ce ne sont pas des donnes complètes : les cartes déjà jouées n'y sont pas.
    L'appelant qui veut une position résoluble doit y réinjecter, pour chaque
    siège, les cartes qu'il a déjà posées.
    """
    resp = _post("/play_worlds", _payload(dealer, initial_hands, actions, observer, n_worlds, temperature))
    if not resp or "hands" not in resp:
        return None
    return [[_mask_to_cards(seat_mask) for seat_mask in world] for world in resp["hands"]]
