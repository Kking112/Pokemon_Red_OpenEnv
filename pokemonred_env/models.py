# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Pokemonred Env Environment.

The pokemonred_env environment is a simple test environment that echoes back messages.
"""

from pydantic import Field

from openenv.core.env_server.types import Action, Observation


class PokemonRedAction(Action):
    """
    Represents a single button press or action sequence.
    """
    action: int = 0  # Discrete action index (0-6)
    action_name: Optional[str] = None  # Human readable action name


class PokemonRedObservation(Observation):
    """
    Multi-modal observation for Pokemon Red.
    """
    # Screen as base64 encoded image
    screen: Dict[str, str] = field(default_factory=dict)
    
    # Game metrics
    health: List[float] = field(default_factory=list)
    level: List[float] = field(default_factory=list)
    badges: List[int] = field(default_factory=list)
    events: List[int] = field(default_factory=list)
    map: List[List[List[int]]] = field(default_factory=list)
    recent_actions: List[int] = field(default_factory=list)
    
    # Game State
    in_battle: int = 0
    position: List[int] = field(default_factory=list)
    has_text: int = 0
    game_text_raw: List[int] = field(default_factory=list)
    
    # Green Agent Metrics
    green_metrics: Dict[str, Any] = field(default_factory=dict)

class PokemonRedState(State):
    """
    Persistent state for the Pokemon Red environment server.
    """
    step_count: int = 0
    total_reward: float = 0.0
    reset_count: int = 0