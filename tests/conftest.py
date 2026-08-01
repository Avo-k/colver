"""Fixtures communes aux tests de la couche web.

Deux contraintes dictent presque tout ce fichier.

**Les modèles ne doivent pas se télécharger.** `colver.web.server` cherche ses
poids à l'import et, s'il ne les trouve pas, les télécharge — ce qui rendrait la
suite dépendante du réseau et lente en CI. On neutralise donc les quatre
`download_*` **avant** que le moindre module de test n'importe le serveur (code
au niveau du fichier, pas fixture : `conftest.py` est importé le premier). Une
machine de dev qui a ses modèles dans `./models/` les garde ; une CI qui ne les
a pas tourne sans, et aucun test ne doit en dépendre.

**La base est un module global.** `database._db` est une connexion unique
publiée une fois migrée : sans remise à zéro entre les tests, le premier fichier
imposerait sa base à tous les autres. `clean_db` la remplace par une base
temporaire, migrations comprises, et la referme — une connexion aiosqlite
oubliée laisse un thread vivant et le processus de test ne rend jamais la main.
"""

import os
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))

import colver  # noqa: E402

for _name in ("download_model", "download_bid_model",
              "download_belief_model", "download_playgen_model"):
    if hasattr(colver, _name):
        setattr(colver, _name, lambda *a, **k: None)

# Aucun sidecar en test : un agent IS-DD irait sinon frapper une URL de prod
# héritée de l'environnement.
os.environ.pop("COLVER_PLAYGEN_GPU_URL", None)

import colver.web.database as db  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_rate_limits():
    """Remettre les compteurs de débit à zéro entre les tests.

    `auth._AUTH_LIMITER` est un objet de module, et tous les tests sortent de la
    même « IP » (`testclient`) : sans ça, le premier fichier qui teste un échec
    d'authentification laisse le budget vide pour tous les suivants, et les
    échecs qui s'ensuivent n'ont rien à voir avec ce qu'ils prétendent tester.
    """
    import colver.web.auth as _auth
    _auth._AUTH_LIMITER.reset()
    yield
    _auth._AUTH_LIMITER.reset()


@pytest.fixture
async def clean_db(tmp_path, monkeypatch):
    """Une base neuve, migrée, rendue à l'état d'origine après le test."""
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "colver.db"))
    monkeypatch.setattr(db, "_db", None)
    conn = await db.get_db()
    try:
        yield conn
    finally:
        await conn.close()
        monkeypatch.setattr(db, "_db", None)


def await_sync(coro):
    """Exécuter une coroutine depuis un test synchrone.

    Les tests qui pilotent `TestClient` sont synchrones (il porte sa propre
    boucle dans un thread), mais ils ont besoin de regarder la base. On ouvre
    donc une boucle à part plutôt que d'essayer de se greffer sur la sienne : la
    connexion `database._db` a été créée par `clean_db` sur une troisième
    boucle, et aiosqlite sérialise tout dans son propre thread — elle se laisse
    utiliser depuis n'importe laquelle.
    """
    import asyncio
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.fixture
def deal():
    """Une distribution reproductible : quatre mains de huit cartes.

    Tirée avec un `random.Random` local plutôt qu'en semant le module : les
    sessions tirent elles aussi dans `random`, et une graine globale ferait
    dépendre le résultat de l'ordre des tests.
    """
    import random

    def _make(seed=0):
        deck = list(range(32))
        random.Random(seed).shuffle(deck)
        return [deck[i * 8:(i + 1) * 8] for i in range(4)]

    return _make


@pytest.fixture
def played_deal(deal):
    """Une donne jouée jusqu'au bout, rendue comme (mains, journal d'actions).

    Jouée à l'heuristique du moteur, sans réseau ni poids : ce que ces tests
    vérifient est la mécanique de la session, pas la force du bot.
    """
    def _play(seed=0, dealer=0, corrupt_at=None):
        hands = deal(seed)
        env = colver.Env.deal_with_hands(dealer, [list(h) for h in hands])
        actions = []
        while not env.is_terminal():
            phase = int(env.phase())
            action = int(env.bid_improved()) if phase == 0 \
                else int(env.action_heuristic_play())
            if corrupt_at is not None and len(actions) == corrupt_at and phase == 1:
                # Une carte que ce siège n'a pas. `env.step()` l'avale sans
                # rien dire — c'est tout le sujet de `check_legal`.
                legal = list(env.legal_actions())
                action = next(c for c in range(32) if c not in legal)
            actions.append({"player": int(env.current_player()),
                            "action": action, "phase": phase})
            env.step(action)
        return hands, actions

    return _play
