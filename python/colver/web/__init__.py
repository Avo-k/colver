"""Colver web UI — ``pip install colver[web]`` to enable."""

import os


def main() -> None:
    """Start the Colver web server."""
    try:
        import uvicorn  # noqa: F401
    except ImportError:
        raise SystemExit(
            "Web dependencies not installed. Run:\n"
            "  pip install colver[web]"
        )
    # L'import du serveur configure la journalisation applicative
    # (basicConfig) avant le téléchargement des modèles — leurs messages
    # passent donc déjà par elle.
    from colver.web.server import app

    # Horodater les lignes uvicorn comme les nôtres. On mute le dict de config
    # par défaut plutôt que d'en fournir un neuf : uvicorn garde ses handlers
    # à lui (propagate=False sur ses loggers), donc rien n'est écrit deux
    # fois. L'access log uvicorn tient lieu de journal de requêtes ; il reste
    # à INFO quel que soit COLVER_LOG_LEVEL pour ne jamais le perdre.
    from uvicorn.config import LOGGING_CONFIG

    LOGGING_CONFIG["formatters"]["default"]["fmt"] = (
        "%(asctime)s %(levelprefix)s %(message)s")
    LOGGING_CONFIG["formatters"]["access"]["fmt"] = (
        '%(asctime)s %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s')

    # Derrière Caddy : ne croire les en-têtes X-Forwarded-* (IP client, schéma
    # https) que s'ils viennent du proxy lui-même. En Docker le proxy n'arrive
    # pas en 127.0.0.1 — passer son IP ou un CIDR (ex. 172.16.0.0/12) via
    # COLVER_FORWARDED_ALLOW_IPS. Jamais "*" : un client pourrait alors forger
    # X-Forwarded-For et contourner les limites par IP.
    forwarded = os.environ.get("COLVER_FORWARDED_ALLOW_IPS", "127.0.0.1")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        proxy_headers=True,
        forwarded_allow_ips=forwarded,
    )
