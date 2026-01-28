# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Pokemon Red Environment Server Implementation.

This module wraps the PyBoy Game Boy emulator and exposes Pokemon Red
as an OpenEnv Environment for RL training.
"""

from __future__ import annotations

import base64
import io
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

try:
    from pyboy import PyBoy
    from pyboy.utils import WindowEvent
except ImportError as e:
    raise ImportError(
        "PyBoy is required for the Pokemon Red environment. "
        "Install with: pip install pyboy"
    ) from e

from openenv.core.env_server import Environment
from openenv.core.env_server.types import Action, Observation

# Import resolution for both package and standalone modes
import sys
import os
_is_package_mode = __name__.startswith("pokemonred_env.")

if _is_package_mode:
    try:
        from ..models import PokemonRedAction, PokemonRedObservation, PokemonRedState
        from ..config import PokemonRedConfig
    except ImportError:
        # Fallback if relative imports fail
        from models import PokemonRedAction, PokemonRedObservation, PokemonRedState
        from config import PokemonRedConfig
else:
    from models import PokemonRedAction, PokemonRedObservation, PokemonRedState
    from config import PokemonRedConfig

from .global_map import local_to_global, GLOBAL_MAP_SHAPE

# Memory addresses for Pokemon Red
EVENT_FLAGS_START = 0xD747
EVENT_FLAGS_END = 0xD87E
IS_IN_BATTLE_ADDR = 0xD057
PARTY_SIZE_ADDR = 0xD163

# Party Pokemon HP addresses
HP_ADDRESSES = [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]
MAX_HP_ADDRESSES = [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]
LEVEL_ADDRESSES = [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]


class PokemonRedEnvironment(Environment):
    """
    Pokemon Red Environment wrapper for OpenEnv.

    This environment wraps Pokemon Red via PyBoy emulator and provides
    a clean interface for RL training with rich observations and
    configurable reward shaping.

    Supported actions: 0=Down, 1=Left, 2=Right, 3=Up, 4=A, 5=B, 6=Start

    Args:
        config: PokemonRedConfig with environment settings.

    Example:
        >>> config = PokemonRedConfig(headless=True)
        >>> env = PokemonRedEnvironment(config)
        >>> obs = env.reset()
        >>> obs = env.step(PokemonRedAction(action=4))  # Press A
    """

    # Enable concurrent WebSocket sessions
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # Action mappings
    VALID_ACTIONS = [
        WindowEvent.PRESS_ARROW_DOWN,
        WindowEvent.PRESS_ARROW_LEFT,
        WindowEvent.PRESS_ARROW_RIGHT,
        WindowEvent.PRESS_ARROW_UP,
        WindowEvent.PRESS_BUTTON_A,
        WindowEvent.PRESS_BUTTON_B,
        WindowEvent.PRESS_BUTTON_START,
    ]
    RELEASE_ACTIONS = [
        WindowEvent.RELEASE_ARROW_DOWN,
        WindowEvent.RELEASE_ARROW_LEFT,
        WindowEvent.RELEASE_ARROW_RIGHT,
        WindowEvent.RELEASE_ARROW_UP,
        WindowEvent.RELEASE_BUTTON_A,
        WindowEvent.RELEASE_BUTTON_B,
        WindowEvent.RELEASE_BUTTON_START,
    ]
    ACTION_NAMES = ["Down", "Left", "Right", "Up", "A", "B", "Start"]

    def __init__(self, config: PokemonRedConfig):
        """Initialize Pokemon Red environment."""
        super().__init__()
        self.config = config
        
        # Initialize state
        self._state = PokemonRedState(episode_id=str(uuid.uuid4()))
        
        # Session path for saving states
        self.session_path = Path(config.session_path)
        self.session_path.mkdir(exist_ok=True, parents=True)

        # Initialize PyBoy emulator
        window_type = "null" if config.headless else "SDL2"
        self.pyboy = PyBoy(
            config.gb_path,
            window=window_type,
            sound_emulated=False,  # Disable sound for headless operation
        )
        if not config.headless:
            self.pyboy.set_emulation_speed(6)

        # Tracking state
        self.seen_coords: Dict[str, int] = {}
        self.explore_map = np.zeros(GLOBAL_MAP_SHAPE, dtype=np.uint8)
        self._prev_state_dict: Dict[str, Any] = {}

        # Initialize reward manager if configured
        self.reward_manager: Optional[Any] = None
        if config.use_modular_rewards:
            from rewards import RewardManager
            self.reward_manager = RewardManager(global_scale=config.reward_scale)
            self.reward_manager.register_defaults({
                "exploration_weight": config.exploration_weight,
                "badge_weight": config.badge_weight,
                "level_weight": config.level_weight,
                "event_weight": config.event_weight,
            })

        # Load event mappings for detailed event tracking
        events_path = Path(__file__).parent / "events.json"
        with open(events_path) as f:
            self._event_mappings: Dict[str, str] = json.load(f)

    def reset(self) -> Observation:
        """
        Reset the environment and return initial observation.

        Returns:
            Initial PokemonRedObservation for the agent.
        """
        # Reset state tracking
        self._state = PokemonRedState(episode_id=str(uuid.uuid4()))
        self.seen_coords = {}
        self.explore_map.fill(0)
        self._prev_state_dict = {}

        # Reset reward manager if enabled
        if self.reward_manager:
            self.reward_manager.reset()

        # Load initial save state
        with open(self.config.init_state, "rb") as f:
            self.pyboy.load_state(f)

        # Tick once to render initial frame
        self.pyboy.tick(1, True)

        return self._make_observation(reward=0.0, done=False)

    def step(self, action: Action) -> Observation:
        """
        Execute agent's action and return resulting observation.

        Args:
            action: PokemonRedAction containing the button to press.

        Returns:
            Observation after action execution.

        Raises:
            ValueError: If action is not a PokemonRedAction.
        """
        if not isinstance(action, PokemonRedAction):
            raise ValueError(f"Expected PokemonRedAction, got {type(action)}")

        # Validate action
        action_idx = action.action
        if action_idx < 0 or action_idx >= len(self.VALID_ACTIONS):
            raise ValueError(f"Invalid action: {action_idx}. Valid range: [0, 6]")

        # Execute action with press/release timing
        press_duration = 8
        self.pyboy.send_input(self.VALID_ACTIONS[action_idx])
        self.pyboy.tick(press_duration, False)
        self.pyboy.send_input(self.RELEASE_ACTIONS[action_idx])
        self.pyboy.tick(self.config.action_freq - press_duration - 1, False)
        self.pyboy.tick(1, True)  # Render on final tick

        # Update exploration tracking
        self._update_exploration()

        # Calculate reward
        current_state = self._get_state_dict()
        reward = self._calculate_reward(current_state, self._prev_state_dict)
        self._prev_state_dict = current_state

        # Update state
        self._state.step_count += 1
        self._state.total_reward += reward
        self._state.badges_obtained = current_state["badge_count"]
        self._state.max_level_sum = max(
            self._state.max_level_sum, current_state["level_sum"]
        )
        self._state.events_triggered = current_state["event_count"]

        # Check termination
        done = self._state.step_count >= self.config.max_steps
        if self.config.terminate_on_blackout and self._check_blackout():
            done = True

        return self._make_observation(reward=reward, done=done)

    @property
    def state(self) -> PokemonRedState:
        """Get current environment state."""
        return self._state

    def _make_observation(self, reward: float, done: bool) -> PokemonRedObservation:
        """Create observation from current game state."""
        # Capture screen as base64 PNG
        screen = self.pyboy.screen.ndarray[:, :, :3].astype(np.uint8)
        img = Image.fromarray(screen)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        screen_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Read game state
        x, y, map_id = self._get_position()
        health = self._get_hp_fraction()
        level_sum = self._get_level_sum()
        badges = self._get_badges_list()
        in_battle = self._is_in_battle()

        return PokemonRedObservation(
            screen_b64=screen_b64,
            screen_shape=list(screen.shape),
            health=health,
            level_sum=level_sum,
            badges=badges,
            position=[x, y, map_id],
            in_battle=in_battle,
            seen_coords_count=len(self.seen_coords),
            legal_actions=list(range(7)),
            done=done,
            reward=reward,
            metadata={
                "action_names": self.ACTION_NAMES,
                "step_count": self._state.step_count,
                "total_reward": self._state.total_reward,
            },
        )

    def _get_position(self) -> Tuple[int, int, int]:
        """Get player position (x, y, map_id)."""
        return (
            self.pyboy.memory[0xD362],
            self.pyboy.memory[0xD361],
            self.pyboy.memory[0xD35E],
        )

    def _update_exploration(self) -> None:
        """Update exploration tracking with current position."""
        x, y, map_id = self._get_position()
        coord_key = f"{x}:{y}:{map_id}"
        self.seen_coords[coord_key] = self.seen_coords.get(coord_key, 0) + 1

        # Update global exploration map
        gx, gy = local_to_global(y, x, map_id)
        if 0 <= gx < GLOBAL_MAP_SHAPE[0] and 0 <= gy < GLOBAL_MAP_SHAPE[1]:
            self.explore_map[gx, gy] = 255

    def _get_hp_fraction(self) -> float:
        """Get party HP as fraction [0, 1]."""
        party_size = min(self.pyboy.memory[PARTY_SIZE_ADDR], 6)
        if party_size == 0:
            return 0.0
        hp_sum = sum(self._read_hp(HP_ADDRESSES[i]) for i in range(party_size))
        max_hp_sum = sum(self._read_hp(MAX_HP_ADDRESSES[i]) for i in range(party_size))
        return hp_sum / max(max_hp_sum, 1)

    def _read_hp(self, addr: int) -> int:
        """Read 16-bit HP value from memory."""
        return 256 * self.pyboy.memory[addr] + self.pyboy.memory[addr + 1]

    def _get_level_sum(self) -> int:
        """Get sum of party Pokemon levels."""
        party_size = min(self.pyboy.memory[PARTY_SIZE_ADDR], 6)
        return sum(self.pyboy.memory[LEVEL_ADDRESSES[i]] for i in range(party_size))

    def _get_badges_list(self) -> List[int]:
        """Get 8-element list of badge flags."""
        badge_byte = self.pyboy.memory[0xD356]
        return [int(b) for b in f"{badge_byte:08b}"][::-1]

    def _is_in_battle(self) -> bool:
        """Check if player is in battle."""
        return self.pyboy.memory[IS_IN_BATTLE_ADDR] != 0

    def _check_blackout(self) -> bool:
        """Check if player has blacked out (all Pokemon fainted outside battle)."""
        return self._get_hp_fraction() == 0.0 and not self._is_in_battle()

    def _get_event_count(self) -> int:
        """Count triggered event flags."""
        count = 0
        for addr in range(EVENT_FLAGS_START, EVENT_FLAGS_END):
            count += self.pyboy.memory[addr].bit_count()
        return count

    def _get_event_details(self) -> Dict[str, bool]:
        """
        Get detailed event flags with human-readable names.
        
        Returns:
            Dictionary mapping event names to their triggered status.
            Only includes named events (skips generic hex codes).
        """
        events = {}
        for key, name in self._event_mappings.items():
            # Skip unnamed events (generic hex codes like "0x123" or short codes)
            if name.startswith("0x") or len(name) <= 3:
                continue
            addr_str, bit = key.split("-")
            addr = int(addr_str, 16)
            bit_pos = int(bit)
            events[name] = bool(self.pyboy.memory[addr] & (1 << bit_pos))
        return events

    def _get_state_dict(self) -> Dict[str, Union[int, float]]:
        """Get current game state as dictionary for reward calculation."""
        return {
            "seen_coords_count": len(self.seen_coords),
            "badge_count": self.pyboy.memory[0xD356].bit_count(),
            "level_sum": self._get_level_sum(),
            "event_count": self._get_event_count(),
            "hp_fraction": self._get_hp_fraction(),
        }

    def _calculate_reward(
        self, current: Dict[str, Any], previous: Dict[str, Any]
    ) -> float:
        """
        Calculate reward based on state changes.
        
        Simple built-in reward shaping. For more complex reward functions,
        see the rewards/ module for modular components.
        """
        # Use modular reward manager if configured
        if self.reward_manager:
            return self.reward_manager.calculate(current, previous)
        
        # Fallback to built-in reward calculation
        if not previous:
            return 0.0

        reward = 0.0
        
        # Exploration reward
        new_coords = current["seen_coords_count"] - previous.get("seen_coords_count", 0)
        reward += new_coords * 0.02

        # Badge reward
        new_badges = current["badge_count"] - previous.get("badge_count", 0)
        reward += new_badges * 5.0

        # Level up reward
        level_diff = current["level_sum"] - previous.get("level_sum", 0)
        reward += max(0, level_diff) * 1.0

        # Event progress reward
        event_diff = current["event_count"] - previous.get("event_count", 0)
        reward += max(0, event_diff) * 0.1

        return reward * self.config.reward_scale

    def export_state(self, path: Optional[str] = None) -> bytes:
        """
        Export current emulator state for checkpointing.
        
        Args:
            path: Optional file path to save state to disk.
        
        Returns:
            Raw state bytes.
        """
        buffer = io.BytesIO()
        self.pyboy.save_state(buffer)
        state_bytes = buffer.getvalue()
        if path:
            with open(path, 'wb') as f:
                f.write(state_bytes)
        return state_bytes

    def import_state(self, state_bytes: bytes) -> None:
        """
        Import emulator state from bytes.
        
        Args:
            state_bytes: Raw state bytes from export_state().
        """
        buffer = io.BytesIO(state_bytes)
        self.pyboy.load_state(buffer)

    def close(self) -> None:
        """Clean up environment resources."""
        if hasattr(self, 'pyboy') and self.pyboy is not None:
            self.pyboy.stop()
            self.pyboy = None

    def __del__(self) -> None:
        """Destructor to ensure cleanup on garbage collection."""
        self.close()

    def __enter__(self) -> "PokemonRedEnvironment":
        """Context manager entry."""
        return self

    def __exit__(self, *args) -> None:
        """Context manager exit with cleanup."""
        self.close()
