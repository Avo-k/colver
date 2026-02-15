# Stage 1: Build the Rust/PyO3 wheel
FROM rust:1.83-bookworm AS builder

# Install Python + maturin build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip python3-venv && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --break-system-packages maturin

WORKDIR /build
COPY colver-core/ colver-core/
COPY colver-py/ colver-py/
COPY Cargo.toml Cargo.lock ./

RUN maturin build --release -m colver-py/Cargo.toml -o /wheels

# Stage 2: Lightweight runtime (no torch needed — DouDou uses Rust inference)
FROM python:3.11-slim-bookworm

COPY --from=builder /wheels/*.whl /tmp/
RUN pip install --no-cache-dir "/tmp/colver-*.whl[web]" && rm /tmp/*.whl

COPY colver-web/ /app/colver-web/
COPY images/cards/ /app/images/cards/
# Copy DMC model weights if available (enables DouDou agent)
COPY models/dmc_final.bi[n] /app/models/

WORKDIR /app
EXPOSE 8000

CMD ["python", "colver-web/backend/server.py"]
