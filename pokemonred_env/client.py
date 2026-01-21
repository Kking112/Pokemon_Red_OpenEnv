# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Pokemon Red Environment Client.

This module provides the client for connecting to a Pokemon Red Environment server
via WebSocket for persistent sessions.
"""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import PokemonRedAction, PokemonRedObservation, PokemonRedState

if TYPE_CHECKING:
    from openenv.core.containers.runtime import ContainerProvider


class PokemonRedEnv(EnvClient[PokemonRedAction, PokemonRedObservation, PokemonRedState]):
    """
    Client for Pokemon Red Environment.

    This client maintains a persistent WebSocket connection to the environment
    server, enabling efficient multi-step interactions with lower latency.

    Example:
        >>> # Connect to a running server
        >>> with PokemonRedEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.screen_shape)
        ...
        ...     result = client.step(PokemonRedAction(action=4))  # Press A
        ...     print(result.reward, result.done)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = PokemonRedEnv.from_docker_image("pokemonred-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(PokemonRedAction(action=0))  # Press Down
        ... finally:
        ...     client.close()

    Example from HuggingFace Hub:
        >>> # Connect to hosted environment
        >>> client = PokemonRedEnv.from_hub("openenv/pokemonred")
        >>> with client:
        ...     result = client.reset()
        ...     for _ in range(100):
        ...         action = PokemonRedAction(action=random.randint(0, 6))
        ...         result = client.step(action)
    """

    def _step_payload(self, action: PokemonRedAction) -> Dict[str, Any]:
        """
        Convert PokemonRedAction to JSON payload for step request.

        Args:
            action: PokemonRedAction instance.

        Returns:
            Dictionary representation suitable for JSON encoding.
        """
        return {"action": action.action}

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[PokemonRedObservation]:
        """
        Parse server response into StepResult[PokemonRedObservation].

        Args:
            payload: JSON response from server.

        Returns:
            StepResult with PokemonRedObservation.
        """
        obs_data = payload.get("observation", {})

        observation = PokemonRedObservation(
            screen_b64=obs_data.get("screen_b64", ""),
            screen_shape=obs_data.get("screen_shape", [144, 160, 3]),
            health=obs_data.get("health", 0.0),
            level_sum=obs_data.get("level_sum", 0),
            badges=obs_data.get("badges", [0] * 8),
            position=obs_data.get("position", [0, 0, 0]),
            in_battle=obs_data.get("in_battle", False),
            seen_coords_count=obs_data.get("seen_coords_count", 0),
            legal_actions=obs_data.get("legal_actions", list(range(7))),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> PokemonRedState:
        """
        Parse server response into PokemonRedState object.

        Args:
            payload: JSON response from /state endpoint.

        Returns:
            PokemonRedState object with environment state information.
        """
        return PokemonRedState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            total_reward=payload.get("total_reward", 0.0),
            reset_count=payload.get("reset_count", 0),
            badges_obtained=payload.get("badges_obtained", 0),
            max_level_sum=payload.get("max_level_sum", 0),
            events_triggered=payload.get("events_triggered", 0),
        )
