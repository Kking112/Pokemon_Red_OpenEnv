"""
src/envs/pokemon_red/server/env.py
--------------------------------
Core Pokemon Red environment logic ported to OpenEnv.
"""
import json
import uuid
import base64
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from PIL import Image
from pyboy import PyBoy
from pyboy.utils import WindowEvent
from einops import repeat

from core.env_server import Environment, Transform
from pokemonred_env.models import PokemonRedAction, PokemonRedObservation, PokemonRedState
from pokemonred_env.config import PokemonRedConfig
from pokemonred_env.rewards.manager import RewardManager
from .global_map import local_to_global, GLOBAL_MAP_SHAPE

# Memory addresses (Constants)
EVENT_FLAGS_START = 0xD747
EVENT_FLAGS_END = 0xD87E
MUSEUM_TICKET = (0xD754, 0)
TILEMAP_START = 0xC3A0
TILEMAP_SIZE = 360
TILEMAP_WIDTH = 20
TILEMAP_HEIGHT = 18
TEXT_BOX_ID_ADDR = 0xD125
IS_IN_BATTLE_ADDR = 0xD057

# Character Map
POKEMON_CHARMAP = {
    0x50: '', 0x4F: '\n', 0x7F: ' ',
    **{0x80 + i: chr(ord('A') + i) for i in range(26)},
    **{0xA0 + i: chr(ord('a') + i) for i in range(26)},
    **{0xF6 + i: str(i) for i in range(10)},
    0xE3: '-', 0xE6: '?', 0xE7: '!', 0xE8: '.',
    0xEF: '♂', 0xF5: '♀', 0xBA: ':', 0xE0: "'",
    0x9C: '(', 0x9D: ')', 0xE1: 'PK', 0xE2: 'MN',
    0x75: '...', 0xF4: ',', 0xF3: '/', 0xF1: '*', 0xF2: '*',
}
TEXT_BOX_BORDER_TILES = {0x79, 0x7A, 0x7B, 0x7C, 0x7D, 0x7E, 0x6F}

class PokemonRedEnv(Environment):

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, config: PokemonRedConfig, transform: Transform | None = None):
        super().__init__(transform)
        self.config = config
        self.reward_manager = RewardManager(config.dict())
        self._state = PokemonRedState()
        # Paths
        self.session_path = Path(config.session_path)
        self.session_path.mkdir(exist_ok=True, parents=True)
        
        # Emulator - disable sound to prevent buffer overrun in headless mode
        head = "null" if config.headless else "SDL2"
        self.pyboy = PyBoy(
            config.gb_path,
            window=head,
            sound_emulated=False  # Disable sound to prevent buffer overrun errors.. furthermore sound is not needed for anything
        )
        if not config.headless:
            self.pyboy.set_emulation_speed(6)
            
        # State tracking
        self.recent_actions = np.zeros((config.frame_stacks,), dtype=np.uint8)
        self.seen_coords = {}
        self.explore_map = np.zeros(GLOBAL_MAP_SHAPE, dtype=np.uint8)
        self.coords_pad = 12
        self.step_count = 0
        self.total_reward = 0.0
        self.reset_count = 0
        self.action_names = ["Down", "Left", "Right", "Up", "A", "B", "Start"]
        
        # Actions
        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN, WindowEvent.PRESS_ARROW_LEFT, WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP, WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START,
        ]
        self.release_actions = [
            WindowEvent.RELEASE_ARROW_DOWN, WindowEvent.RELEASE_ARROW_LEFT, WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP, WindowEvent.RELEASE_BUTTON_A, WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_BUTTON_START,
        ]

    @property
    def state(self) -> PokemonRedState:
        return self._state

    def reset(self) -> PokemonObservation:
        self._state = PokemonRedState()
        # Load state
        with open(self.config.init_state, "rb") as f:
            self.pyboy.load_state(f)
            
        # Reset trackers
        self.seen_coords = {}
        self.explore_map.fill(0)
        self.recent_actions.fill(0)
        self.step_count = 0
        self.total_reward = 0.0
        self.reset_count += 1
        self.reward_manager.reset()
        
        return self._get_obs()

    def step(self, action: PokemonAction) -> PokemonObservation:
        # Resolve action
        act_idx = action.action
        
        self.pyboy.send_input(self.valid_actions[act_idx])
        press_step = 8
        self.pyboy.tick(press_step, False) # Do not render every tick for performance
        self.pyboy.send_input(self.release_actions[act_idx])
        self.pyboy.tick(self.config.action_freq - press_step - 1, False)
        self.pyboy.tick(1, True) # Render on last tick
        
        # Update trackers
        self._update_recent_actions(act_idx)
        self._update_exploration()
        
        # Calc Reward
        state_dict = self._get_state_dict()
        reward = self.reward_manager.update(state_dict)
        self.total_reward += reward
        self.step_count += 1
        self._state.step_count = self.step_count
        
        return self._get_obs()

    def _get_obs(self) -> PokemonObservation:
        screen_arr = self.pyboy.screen.ndarray[:, :, :3].astype(np.uint8)
        # Convert to base64 for transport if needed, or keep raw. 
        # OpenEnv typically expects serializable JSON. 
        # We will use a simplified dict representation for now.
        import io
        img = Image.fromarray(screen_arr)
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        b64_img = base64.b64encode(bio.getvalue()).decode("utf-8")
        
        return PokemonObservation(
            screen={"b64": b64_img, "shape": list(screen_arr.shape)},
            health=[self._read_hp_fraction()],
            level=[float(x) for x in self._fourier_encode(self._get_levels_sum())],
            badges=self._get_badges_binary(),
            events=self._read_event_bits(),
            map=self._get_explore_map().tolist(),
            recent_actions=self.recent_actions.tolist(),
            in_battle=1 if self._is_in_battle() else 0,
            position=list(self._get_game_coords()),
            has_text=1 if self._has_active_text() else 0,
            game_text_raw=self._read_tilemap_raw().tolist()
        )

    # --- Helpers ---
    def _update_recent_actions(self, action: int):
        self.recent_actions = np.roll(self.recent_actions, 1)
        self.recent_actions[0] = action

    def _get_game_coords(self) -> Tuple[int, int, int]:
        return (self.pyboy.memory[0xD362], self.pyboy.memory[0xD361], self.pyboy.memory[0xD35E])

    def _update_exploration(self):
        x, y, map_n = self._get_game_coords()
        # Local
        coord_string = f"x:{x} y:{y} m:{map_n}"
        self.seen_coords[coord_string] = self.seen_coords.get(coord_string, 0) + 1
        # Global map
        gx, gy = local_to_global(y, x, map_n)
        if gx < GLOBAL_MAP_SHAPE[0] and gy < GLOBAL_MAP_SHAPE[1]:
            self.explore_map[gx, gy] = 255

    def _read_hp_fraction(self) -> float:
        # Simplified for brevity - assume party leader or sum
        # Using the same logic as reference
        hp_sum = sum(self._read_hp(a) for a in [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248])
        max_hp_sum = sum(self._read_hp(a) for a in [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269])
        return hp_sum / max(max_hp_sum, 1)

    def _read_hp(self, start: int) -> int:
        return 256 * self.pyboy.memory[start] + self.pyboy.memory[start + 1]

    def _fourier_encode(self, val: float) -> np.ndarray:
        return np.sin(val * 2 ** np.arange(8))

    def _get_levels_sum(self) -> int:
        return sum(self.pyboy.memory[a] for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268])

    def _get_badges_binary(self) -> List[int]:
        badge_byte = self.pyboy.memory[0xD356]
        return [int(b) for b in f"{badge_byte:08b}"]

    def _read_event_bits(self) -> List[int]:
        # Truncate for performance? Reference reads ALL events.
        # We will read 100 bytes from start
        return [] # Placeholder to avoid large payload in observation for now, logic exists in reference

    def _get_explore_map(self) -> np.ndarray:
        # Return a window around current position
        # Simplified for now
        return np.zeros((10, 10, 1))

    def _is_in_battle(self) -> bool:
        return self.pyboy.memory[IS_IN_BATTLE_ADDR] != 0

    def _read_tilemap_raw(self) -> np.ndarray:
        return np.array([self.pyboy.memory[TILEMAP_START + i] for i in range(TILEMAP_SIZE)], dtype=np.uint8)

    def _has_active_text(self) -> bool:
        # Simplified heuristic
        return self.pyboy.memory[TEXT_BOX_ID_ADDR] != 0

    def _get_state_dict(self) -> Dict[str, Any]:
        """Extract state dictionary for RewardManager."""
        return {
            "event_count": self._get_event_count(),
            "seen_coords_count": len(self.seen_coords),
            "badge_count": self.pyboy.memory[0xD356].bit_count(),
            "level_sum": self._get_levels_sum(),
        }

    def _get_event_count(self) -> int:
        # Sum bits in event flag range
        count = 0
        for i in range(EVENT_FLAGS_START, EVENT_FLAGS_END):
            count += self.pyboy.memory[i].bit_count()
        return count
