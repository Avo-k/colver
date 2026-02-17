# Stage 1: Build the PyO3 wheel and create venv with dependencies
FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS builder

# Install Rust toolchain (needed for maturin/PyO3 build)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    --default-toolchain 1.83.0 --profile minimal
ENV PATH="/root/.cargo/bin:$PATH"

# uv best practices for Docker
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Install dependencies first (cached layer — only invalidated when deps change)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-editable --no-dev --extra web

# Copy source and build/install the project
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable --no-dev --extra web

# Stage 2: Lightweight runtime (no Rust, no uv, no torch)
FROM python:3.12-slim-bookworm

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Entrypoint auto-downloads model if missing
COPY entrypoint.sh /app/entrypoint.sh

ENV PATH="/app/.venv/bin:$PATH"
ENV COLVER_MODEL_PATH="/app/models/dmc_final.bin"
WORKDIR /app
EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "-m", "colver.web"]
