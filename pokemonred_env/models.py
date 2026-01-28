# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Pokemon Red OpenEnv Environment.

This module defines the Action, Observation, and State types for the Pokemon Red
Game Boy emulator environment using PyBoy.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import ConfigDict, Field

from openenv.core.env_server.types import Action, Observation, State


class PokemonRedAction(Action):
    """
    Action for Pokemon Red environment.

    Represents a single Game Boy button press.

    Attributes:
        action: Discrete action index (0-6) mapping to:
                0=Down, 1=Left, 2=Right, 3=Up, 4=A, 5=B, 6=Start
    """

    action: int = Field(
        default=0,
        ge=0,
        le=6,
        description="Button index: 0=Down, 1=Left, 2=Right, 3=Up, 4=A, 5=B, 6=Start",
    )


class PokemonRedObservation(Observation):
    """
    Multi-modal observation from Pokemon Red environment.

    Contains the game screen, player stats, and game state information
    suitable for both vision-based and structured RL agents.

    Attributes:
        screen_b64: Base64-encoded PNG of the current game frame.
        screen_shape: Shape of the screen array [height, width, channels].
        health: Party HP fraction in range [0, 1].
        level_sum: Sum of all party Pokemon levels.
        badges: 8-element list of badge flags (0 or 1).
        position: Current position as [x, y, map_id].
        in_battle: Whether player is currently in a battle.
        seen_coords_count: Number of unique coordinates visited (exploration metric).
        legal_actions: List of valid action indices.

    Note:
        The `done` and `reward` fields (inherited from base Observation) are included
        for convenience when observations are used standalone. When using StepResult,
        prefer StepResult.done and StepResult.reward as the authoritative values.
        
        Additional attributes (like `stacked_image` from FrameStackWrapper) can be
        added dynamically via the 'extra' configuration.
    """

    model_config = ConfigDict(extra='allow')

    screen_b64: str = Field(default="", description="Base64-encoded PNG frame")
    screen_shape: List[int] = Field(
        default_factory=lambda: [144, 160, 3], description="Frame dimensions [H,W,C]"
    )
    health: float = Field(default=0.0, ge=0.0, le=1.0, description="Party HP fraction")
    level_sum: int = Field(default=0, ge=0, description="Sum of party Pokemon levels")
    badges: List[int] = Field(
        default_factory=lambda: [0] * 8, description="8-element badge flags"
    )
    position: List[int] = Field(
        default_factory=lambda: [0, 0, 0], description="Player position [x, y, map_id]"
    )
    in_battle: bool = Field(default=False, description="Whether in battle")
    seen_coords_count: int = Field(
        default=0, description="Unique coordinates visited (exploration)"
    )
    legal_actions: List[int] = Field(
        default_factory=lambda: list(range(7)), description="Valid action indices"
    )


class PokemonRedState(State):
    """
    State for Pokemon Red environment.

    Tracks episode progress and game state for the server.

    Attributes:
        total_reward: Cumulative reward for current episode.
        reset_count: Number of times environment has been reset.
        badges_obtained: Number of gym badges obtained (0-8).
        max_level_sum: Maximum level sum achieved this episode.
        events_triggered: Count of game event flags triggered.
    """

    total_reward: float = Field(default=0.0, description="Cumulative episode reward")
    reset_count: int = Field(default=0, description="Number of resets")
    badges_obtained: int = Field(default=0, ge=0, le=8, description="Badges count")
    max_level_sum: int = Field(default=0, description="Max level sum achieved")
    events_triggered: int = Field(default=0, description="Event flags triggered")