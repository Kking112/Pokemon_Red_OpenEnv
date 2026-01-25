# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Multi-stage build for Pokemon Red OpenEnv
# Builds a containerized PyBoy emulator environment for RL training
#
# Build: openenv build
#
# Run (you must provide your own ROM file):
#   docker run -p 8000:8000 -v /path/to/PokemonRed.gb:/rom/PokemonRed.gb pokemonred-env:latest
#
# The init.state file is included in the repository. Additional .state files
# may be provided for starting at different points in the game.
#
# Environment variables:
#   POKEMON_HEADLESS: Run without display (default: true)
#   POKEMON_ACTION_FREQ: Emulator ticks per action (default: 24)
#   POKEMON_ROM_PATH: Path to ROM file (default: /rom/PokemonRed.gb)
#   POKEMON_INIT_STATE: Path to save state (default: /app/env/server/init.state)

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# Install build dependencies for PyBoy and SDL2
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Build arguments
ARG BUILD_MODE=in-repo
ARG ENV_NAME=pokemonred_env

# Copy environment code
COPY . /app/env

WORKDIR /app/env

# Ensure uv is available
RUN if ! command -v uv >/dev/null 2>&1; then \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

# Install dependencies using uv sync
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
    uv sync --frozen --no-install-project --no-editable; \
    else \
    uv sync --no-install-project --no-editable; \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
    uv sync --frozen --no-editable; \
    else \
    uv sync --no-editable; \
    fi

# Final runtime stage
FROM ${BASE_IMAGE}

WORKDIR /app

# Install runtime dependencies for PyBoy headless operation
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsdl2-2.0-0 \
    libsdl2-image-2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from builder
COPY --from=builder /app/env/.venv /app/.venv

# Copy the environment code
COPY --from=builder /app/env /app/env

# Set PATH to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Set PYTHONPATH for imports
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# Configure SDL for headless operation (no display, no audio)
ENV SDL_VIDEODRIVER=dummy
ENV SDL_AUDIODRIVER=dummy

# Default environment configuration
ENV POKEMON_HEADLESS=true
ENV POKEMON_ACTION_FREQ=24
ENV POKEMON_ROM_PATH=/rom/PokemonRed.gb
ENV POKEMON_INIT_STATE=/app/env/server/init.state
ENV POKEMON_MAX_STEPS=163840
ENV POKEMON_SESSION_PATH=/tmp/pokemon_sessions

# Create directories
RUN mkdir -p /tmp/pokemon_sessions /rom

# VOLUME for user-provided ROM file
# Users mount their ROM here: -v /path/to/PokemonRed.gb:/rom/PokemonRed.gb
VOLUME /rom

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run the FastAPI server
CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000"]
