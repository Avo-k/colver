"""Colver web UI — ``pip install colver[web]`` to enable."""


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

    uvicorn.run(app, host="0.0.0.0", port=8000)
