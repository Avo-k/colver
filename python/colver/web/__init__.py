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
    from colver.web.server import app

    uvicorn.run(app, host="0.0.0.0", port=8000)
