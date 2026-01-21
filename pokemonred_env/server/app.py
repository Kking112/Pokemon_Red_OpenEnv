# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Pokemon Red Environment.

This module creates an HTTP server that exposes the PokemonRedEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

Environment variables:
    POKEMON_HEADLESS: Run emulator without display (default: "true")
    POKEMON_ACTION_FREQ: Emulator ticks per action (default: "24")
    POKEMON_ROM_PATH: Path to Pokemon Red ROM (default: "/app/env/server/PokemonRed.gb")
    POKEMON_INIT_STATE: Path to initial save state (default: "/app/env/server/init.state")
    POKEMON_MAX_STEPS: Maximum steps per episode (default: "163840")
"""

import os

# try:
#     from openenv.core.env_server import create_app
# except ImportError as e:
#     raise ImportError(
#         "openenv is required for the server. Install with:\n    pip install openenv"
#     ) from e
from openenv.core.env_server import create_app

from ..models import PokemonRedAction, PokemonRedObservation
from ..config import PokemonRedConfig
from .pokemonred_env_environment import PokemonRedEnvironment


def create_pokemon_environment():
    """
    Factory function that creates PokemonRedEnvironment with config from env vars.
    
    This enables WebSocket session support where each client gets their own
    environment instance with isolated state.
    """
    config = PokemonRedConfig(
        headless=os.getenv("POKEMON_HEADLESS", "true").lower() == "true",
        action_freq=int(os.getenv("POKEMON_ACTION_FREQ", "24")),
        gb_path=os.getenv("POKEMON_ROM_PATH", "/app/env/server/PokemonRed.gb"),
        init_state=os.getenv("POKEMON_INIT_STATE", "/app/env/server/init.state"),
        max_steps=int(os.getenv("POKEMON_MAX_STEPS", "163840")),
        session_path=os.getenv("POKEMON_SESSION_PATH", "/tmp/pokemon_sessions"),
    )
    return PokemonRedEnvironment(config)


# Create the FastAPI app with web interface and README integration
# Pass the factory function for WebSocket session support
app = create_app(
    create_pokemon_environment,
    PokemonRedAction,
    PokemonRedObservation,
    env_name="pokemonred",
)


# def main(host: str = "0.0.0.0", port: int = 8000):
#     """
#     Entry point for direct execution.

#     Args:
#         host: Host address to bind to (default: "0.0.0.0")
#         port: Port number to listen on (default: 8000)
#     """
#     import uvicorn
#     uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Pokemon Red OpenEnv Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
