"""Garde-fous en mémoire contre les clients trop gourmands.

Deux outils, tous deux par clé (en pratique l'IP client) :

- `RateLimiter` : fenêtre glissante, au plus N passages par fenêtre — pour les
  routes qui brûlent du CPU à chaque appel (un bcrypt par tentative de login).
- `ConnectionCap` : plafond de connexions simultanées, par clé et global —
  pour le WebSocket, dont chaque socket peut lancer des simulations coûteuses
  (une `annonces_sim` ≈ 200 solves DD).

Tout est en mémoire et par processus : le site tourne sur un seul uvicorn,
un redémarrage remet les compteurs à zéro et c'est très bien comme ça. Aucun
verrou : les deux classes ne sont touchées que depuis la boucle asyncio, et
aucune méthode ne rend la main entre lecture et écriture.
"""

import time
from collections import deque

# Purge des clés silencieuses au-delà de ce nombre d'entrées, pour que le dict
# d'un limiteur ne grossisse pas sans borne au fil des IP vues une seule fois.
_PRUNE_THRESHOLD = 1024


class RateLimiter:
    """Au plus `limit` passages par `window` secondes, en fenêtre glissante."""

    def __init__(self, limit, window):
        self.limit = limit
        self.window = window
        self._hits = {}  # clé -> deque des instants récents (time.monotonic)

    def allow(self, key):
        """True si ce passage est accepté — et alors il compte dans la fenêtre."""
        now = time.monotonic()
        if len(self._hits) > _PRUNE_THRESHOLD:
            self._prune(now)
        hits = self._hits.setdefault(key, deque())
        while hits and now - hits[0] > self.window:
            hits.popleft()
        if len(hits) >= self.limit:
            return False
        hits.append(now)
        return True

    def retry_after(self, key):
        """Secondes avant qu'un passage redevienne possible (header Retry-After)."""
        hits = self._hits.get(key)
        if not hits:
            return 0
        return max(1, int(self.window - (time.monotonic() - hits[0])) + 1)

    def refund(self, key):
        """Rend le passage le plus récent — pour un appel finalement légitime
        (un login réussi n'est pas du brute-force)."""
        hits = self._hits.get(key)
        if hits:
            hits.pop()

    def reset(self, key=None):
        """Oublier les passages d'une clé, ou de toutes.

        Le limiteur est un objet de module partagé par tout le processus : sans
        remise à zéro, un test qui épuise le budget le laisse épuisé pour les
        suivants (ils viennent tous de la même « IP »). Utile aussi à un
        exploitant qui débloque un joueur à la main.
        """
        if key is None:
            self._hits.clear()
        else:
            self._hits.pop(key, None)

    def _prune(self, now):
        dead = [k for k, h in self._hits.items() if not h or now - h[-1] > self.window]
        for k in dead:
            del self._hits[k]


class ConnectionCap:
    """Plafond de connexions simultanées : `per_key` par clé, `total` en tout.

    `acquire` avant d'accepter la connexion, `release` dans un `finally` —
    sans quoi la première exception fait fuir un slot pour toujours.
    """

    def __init__(self, per_key, total):
        self.per_key = per_key
        self.total = total
        self._counts = {}  # clé -> connexions ouvertes
        self._total = 0

    def acquire(self, key):
        if self._total >= self.total or self._counts.get(key, 0) >= self.per_key:
            return False
        self._counts[key] = self._counts.get(key, 0) + 1
        self._total += 1
        return True

    def release(self, key):
        n = self._counts.get(key, 0)
        if n <= 1:
            self._counts.pop(key, None)
        else:
            self._counts[key] = n - 1
        if self._total > 0:
            self._total -= 1
