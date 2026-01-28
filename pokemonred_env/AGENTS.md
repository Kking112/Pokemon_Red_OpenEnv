# Pokemon Red OpenEnv Environment - Agent Documentation

This document provides comprehensive documentation for AI agents working with the Pokemon Red OpenEnv environment codebase. It covers the architecture, implementation details, and known issues that require attention.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Memory Addresses and Game State](#memory-addresses-and-game-state)
5. [Reward System](#reward-system)
6. [Action Space](#action-space)
7. [Observation Space](#observation-space)
8. [Wrappers](#wrappers)
9. [Known Issues and Improvements](#known-issues-and-improvements)

---

## Overview

The Pokemon Red OpenEnv is a reinforcement learning environment that wraps the Game Boy game "Pokemon Red" using the PyBoy emulator. It follows the OpenEnv specification created by Meta/PyTorch and Hugging Face, providing a standardized HTTP/WebSocket API for RL agents.

### Key Features

- **OpenEnv Compliant**: Implements the standard `reset()`, `step()`, and `state()` API
- **VLA-Ready**: Base64-encoded screen observations for Vision-Language-Action models
- **Modular Rewards**: Pluggable reward components (exploration, badges, levels, events)
- **Green Agent**: Built-in compute efficiency tracking
- **Docker Isolated**: Containerized emulator environment

### Technology Stack

- **Emulator**: PyBoy (Game Boy emulator in Python)
- **API Framework**: FastAPI with WebSocket support
- **Serialization**: Pydantic models for type-safe data structures
- **Image Processing**: PIL/Pillow and NumPy

---

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Application                        │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  PokemonRedEnv (EnvClient)                                 │ │
│  │  - _step_payload(): Convert action to JSON                 │ │
│  │  - _parse_result(): Parse server response                  │ │
│  │  - _parse_state(): Parse state response                    │ │
│  └────────────────────────┬───────────────────────────────────┘ │
└───────────────────────────┼─────────────────────────────────────┘
                            │ WebSocket / HTTP
                            │ (reset, step, state)
┌───────────────────────────▼─────────────────────────────────────┐
│                     Docker Container                             │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  FastAPI Server (app.py)                                   │ │
│  │  - POST /reset                                             │ │
│  │  - POST /step                                              │ │
│  │  - GET /state                                              │ │
│  │  - WS /ws                                                  │ │
│  └────────────────────────┬───────────────────────────────────┘ │
│                           │                                      │
│  ┌────────────────────────▼───────────────────────────────────┐ │
│  │  PokemonRedEnvironment (Environment base)                  │ │
│  │  - PyBoy emulator instance                                 │ │
│  │  - Exploration tracking (seen_coords, explore_map)         │ │
│  │  - Memory reading for game state                           │ │
│  │  - Reward calculation                                      │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### File Structure

```
pokemonred_env/
├── __init__.py              # Package exports
├── client.py                # PokemonRedEnv HTTP/WebSocket client
├── config.py                # PokemonRedConfig Pydantic model
├── models.py                # Action, Observation, State models
├── openenv.yaml             # OpenEnv manifest
├── pyproject.toml           # Dependencies and metadata
├── rewards/                 # Modular reward system
│   ├── __init__.py
│   ├── base.py              # BaseRewardComponent ABC
│   ├── manager.py           # RewardManager
│   ├── exploration.py       # Exploration rewards
│   ├── badge.py             # Badge rewards
│   ├── level.py             # Level-up rewards
│   └── event.py             # Event flag rewards
├── wrappers/                # Environment wrappers
│   ├── __init__.py
│   ├── green_agent.py       # Compute efficiency tracking
│   └── frame_stack.py       # Temporal frame stacking
├── server/
│   ├── __init__.py
│   ├── app.py               # FastAPI server
│   ├── pokemonred_env_environment.py  # Core environment
│   ├── global_map.py        # Map coordinate utilities
│   ├── events.json          # Event flag mappings
│   ├── map_data.json        # Map coordinate data
│   ├── pokered.sym          # Symbol table for memory addresses
│   ├── has_pokedex.state    # Default save state
│   ├── Dockerfile           # Container definition
│   └── requirements.txt     # Server dependencies
└── states/                  # Game save states
    ├── init.state           # Game start state
    ├── has_pokedex.state    # After getting Pokedex
    ├── has_pokedex_nballs.state
    └── fast_text_start.state
```

---

## Core Components

### 1. PokemonRedEnvironment (Server-Side)

Located in `server/pokemonred_env_environment.py`, this is the core environment implementation.

**Key Attributes:**
- `pyboy`: PyBoy emulator instance
- `config`: PokemonRedConfig settings
- `seen_coords`: Dict tracking visited coordinates
- `explore_map`: NumPy array for global exploration visualization
- `_prev_state_dict`: Previous state for delta reward calculation
- `_state`: Current PokemonRedState tracking episode progress

**Action Execution Flow:**
1. Receive action index (0-6)
2. Map to PyBoy WindowEvent (press event)
3. Execute press for 8 ticks
4. Send release event
5. Execute remaining ticks (action_freq - 8 - 1)
6. Render final tick

### 2. PokemonRedEnv (Client-Side)

Located in `client.py`, extends the OpenEnv `EnvClient` base class.

**Key Methods:**
- `_step_payload()`: Converts PokemonRedAction to JSON
- `_parse_result()`: Parses server response to StepResult
- `_parse_state()`: Parses state endpoint response

### 3. Configuration (PokemonRedConfig)

Located in `config.py`, defines all configurable parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `headless` | `True` | Run emulator without display |
| `action_freq` | `24` | Emulator ticks per action |
| `max_steps` | `163840` | Max steps per episode |
| `reward_scale` | `1.0` | Global reward multiplier |
| `init_state` | (path) | Initial save state file |
| `gb_path` | (path) | ROM file location |

---

## Memory Addresses and Game State

The environment reads game state directly from Game Boy memory:

### Player State

| Address | Data | Description |
|---------|------|-------------|
| `0xD362` | X position | Player X coordinate |
| `0xD361` | Y position | Player Y coordinate |
| `0xD35E` | Map ID | Current map identifier |
| `0xD356` | Badges | 8-bit badge flags |
| `0xD057` | Battle flag | Non-zero when in battle |

### Party Pokemon (6 slots)

| Slot | HP Address | Max HP Address | Level Address |
|------|------------|----------------|---------------|
| 1 | `0xD16C` | `0xD18D` | `0xD18C` |
| 2 | `0xD198` | `0xD1B9` | `0xD1B8` |
| 3 | `0xD1C4` | `0xD1E5` | `0xD1E4` |
| 4 | `0xD1F0` | `0xD211` | `0xD210` |
| 5 | `0xD21C` | `0xD23D` | `0xD23C` |
| 6 | `0xD248` | `0xD269` | `0xD268` |

### Event Flags

Event flags are stored in the range `0xD747` to `0xD87E`. Each byte contains 8 flags. The `events.json` file maps these to human-readable event names (e.g., "Got Pokedex", "Beat Brock", "Got HM01").

---

## Reward System

### Built-in Reward Calculation

The environment has a simple built-in reward function in `_calculate_reward()`:

```python
reward = 0.0
reward += new_coords * 0.02        # Exploration
reward += new_badges * 5.0         # Badge acquisition
reward += max(0, level_diff) * 1.0 # Level ups
reward += max(0, event_diff) * 0.1 # Event progression
return reward * config.reward_scale
```

### Modular Reward System

For more sophisticated reward shaping, use the `rewards/` module:

**BaseRewardComponent** (`rewards/base.py`):
- Abstract base class for all reward components
- Tracks cumulative reward per episode
- Supports enable/disable and weight configuration

**RewardManager** (`rewards/manager.py`):
- Aggregates multiple reward components
- Provides reward breakdowns for debugging
- Supports global scaling factor

**Available Components:**
- `ExplorationReward`: Rewards new coordinate visits
- `BadgeReward`: Rewards gym badge acquisition
- `LevelUpReward`: Rewards party level increases
- `EventReward`: Rewards triggering game event flags

---

## Action Space

7 discrete actions mapping to Game Boy inputs:

| Index | Button | Description |
|-------|--------|-------------|
| 0 | Down | Move down |
| 1 | Left | Move left |
| 2 | Right | Move right |
| 3 | Up | Move up |
| 4 | A | Confirm/Interact |
| 5 | B | Cancel/Run |
| 6 | Start | Menu |

**Note**: The SELECT button is NOT exposed as an action.

---

## Observation Space

### PokemonRedObservation Fields

| Field | Type | Description |
|-------|------|-------------|
| `screen_b64` | `str` | Base64-encoded PNG (160×144 RGB) |
| `screen_shape` | `List[int]` | `[144, 160, 3]` |
| `health` | `float` | Party HP fraction `[0.0, 1.0]` |
| `level_sum` | `int` | Sum of all party Pokemon levels |
| `badges` | `List[int]` | 8-element badge flags |
| `position` | `List[int]` | `[x, y, map_id]` |
| `in_battle` | `bool` | Whether player is in battle |
| `seen_coords_count` | `int` | Unique tiles visited |
| `legal_actions` | `List[int]` | Always `[0,1,2,3,4,5,6]` |
| `done` | `bool` | Episode termination flag |
| `reward` | `float` | Step reward |
| `metadata` | `Dict` | Additional context |

### Screen Decoding

```python
import base64
from PIL import Image
import io

screen_bytes = base64.b64decode(obs.screen_b64)
image = Image.open(io.BytesIO(screen_bytes))
screen_array = np.array(image)  # Shape: (144, 160, 3)
```

---

## Wrappers

### GreenAgentTracker (`wrappers/green_agent.py`)

Tracks compute efficiency metrics for sustainable AI development:

- `step_time_ms`: Wall-clock time per step
- `avg_step_time_ms`: Running average step time
- `total_time_s`: Total episode time
- `peak_memory_mb`: Peak memory usage
- `cpu_percent`: CPU utilization

### FrameStackWrapper (`wrappers/frame_stack.py`)

Stacks multiple frames for temporal context:

- Configurable stack size (default: 4)
- Grid or horizontal layout options
- Supports multi-action stepping

---

## Known Issues and Improvements

### Critical Issues

#### 1. **Modular Reward System Not Integrated with Core Environment**

**Location**: `server/pokemonred_env_environment.py` lines 295-325

**Problem**: The environment has a built-in `_calculate_reward()` method that duplicates the logic in the `rewards/` module. The modular `RewardManager` is never actually used by the server-side environment.

**Impact**: Users cannot configure custom reward functions without modifying the core environment code.

**Suggested Fix**: 
- Add optional `RewardManager` parameter to `PokemonRedEnvironment.__init__()`
- If provided, use it instead of the built-in reward calculation
- Add reward config options to `PokemonRedConfig`

```python
# In __init__:
self.reward_manager = reward_manager or self._create_default_manager()

# In _calculate_reward:
if self.reward_manager:
    return self.reward_manager.calculate(current, previous)
```

---

#### 2. **FrameStackWrapper Incompatible with Current Observation Model**

**Location**: `wrappers/frame_stack.py` lines 121-128

**Problem**: The `_get_image()` method expects observations to have an `image` or `screen` attribute, but `PokemonRedObservation` uses `screen_b64` (base64 string). The wrapper will raise `AttributeError`.

**Impact**: FrameStackWrapper cannot be used with the current environment without modification.

**Suggested Fix**:
```python
def _get_image(self, obs) -> Image.Image:
    """Extract PIL Image from observation."""
    if hasattr(obs, 'screen_b64'):
        import base64
        from io import BytesIO
        screen_bytes = base64.b64decode(obs.screen_b64)
        return Image.open(BytesIO(screen_bytes))
    elif hasattr(obs, 'image'):
        return obs.image
    elif hasattr(obs, 'screen'):
        return obs.screen
    else:
        raise AttributeError("Observation has no 'screen_b64', 'image', or 'screen' attribute")
```

---

#### 3. **PyBoy Emulator Not Properly Closed on Environment Destruction**

**Location**: `server/pokemonred_env_environment.py`

**Problem**: The `PokemonRedEnvironment` class initializes a PyBoy instance but never defines a `close()` method to properly shut it down. This can lead to resource leaks when running multiple episodes or when the server restarts.

**Impact**: Memory leaks and potential file handle exhaustion in long-running deployments.

**Suggested Fix**:
```python
def close(self) -> None:
    """Clean up environment resources."""
    if hasattr(self, 'pyboy') and self.pyboy is not None:
        self.pyboy.stop()
        self.pyboy = None

def __del__(self):
    self.close()
```

---

### Moderate Issues

#### 4. **Global Map Coordinate Conversion Has Silent Failures**

**Location**: `server/global_map.py` lines 17-31

**Problem**: When a map ID is not found or coordinates are out of bounds, the function prints a message and returns the center of the map instead of raising an error or returning a sentinel value.

**Impact**: Exploration tracking may be incorrect for unknown maps, with no programmatic way to detect this.

**Suggested Fix**:
```python
def local_to_global(r: int, c: int, map_n: int, strict: bool = False) -> Tuple[int, int]:
    try:
        map_x, map_y = MAP_DATA[map_n]["coordinates"]
        gy = r + map_y + MAP_ROW_OFFSET
        gx = c + map_x + MAP_COL_OFFSET
        if 0 <= gy < GLOBAL_MAP_SHAPE[0] and 0 <= gx < GLOBAL_MAP_SHAPE[1]:
            return gy, gx
        if strict:
            raise ValueError(f"Coordinates out of bounds: ({gx}, {gy})")
        # Log warning instead of print
        logging.warning(f"coord out of bounds! global: ({gx}, {gy}) game: ({r}, {c}, {map_n})")
        return GLOBAL_MAP_SHAPE[0] // 2, GLOBAL_MAP_SHAPE[1] // 2
    except KeyError:
        if strict:
            raise KeyError(f"Map id {map_n} not found in map_data.json")
        logging.warning(f"Map id {map_n} not found in map_data.json")
        return GLOBAL_MAP_SHAPE[0] // 2, GLOBAL_MAP_SHAPE[1] // 2
```

---

#### 5. **Import Fallback Pattern Could Mask Real Import Errors**

**Location**: `server/app.py` lines 41-49, `server/pokemonred_env_environment.py` lines 37-38

**Problem**: The try/except import fallback catches all `ImportError` exceptions, which could hide genuine import failures (e.g., missing dependencies) by silently falling back to the wrong import path.

**Impact**: Debugging import issues becomes difficult as errors are silently swallowed.

**Suggested Fix**:
```python
import sys
import os

# Determine if running as package or standalone
_running_as_package = __name__ != "__main__" and "." in __name__

if _running_as_package:
    from ..models import PokemonRedAction, PokemonRedObservation
    from ..config import PokemonRedConfig
    from .pokemonred_env_environment import PokemonRedEnvironment
else:
    # Standalone mode - add parent to path explicitly
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from models import PokemonRedAction, PokemonRedObservation
    from config import PokemonRedConfig
    from server.pokemonred_env_environment import PokemonRedEnvironment
```

---

#### 6. **Observation `done` Flag Set Redundantly**

**Location**: `models.py` line 96-97, `client.py` lines 95-104

**Problem**: The `done` flag is set both inside `PokemonRedObservation` (inherited from base) AND as a field in `StepResult`. The client parses it from both `observation` and `payload.done`, which is redundant and could lead to inconsistency.

**Impact**: Potential confusion about authoritative source of `done` flag.

**Suggested Fix**: Remove `done` and `reward` from `PokemonRedObservation` since they belong in `StepResult`. Or clearly document that `StepResult.done` is authoritative.

---

#### 7. **Events.json Not Used for Detailed Event Tracking**

**Location**: `server/events.json`, `server/pokemonred_env_environment.py`

**Problem**: The `events.json` file contains detailed mappings of event flag addresses to human-readable names, but the environment only counts total event flags without using this mapping.

**Impact**: Lost opportunity for detailed event-based rewards and agent progress tracking.

**Suggested Fix**:
```python
def _get_event_details(self) -> Dict[str, bool]:
    """Get detailed event flags with human-readable names."""
    events = {}
    for key, name in self.event_mappings.items():
        addr_str, bit = key.split("-")
        addr = int(addr_str, 16)
        bit_pos = int(bit)
        events[name] = bool(self.pyboy.memory[addr] & (1 << bit_pos))
    return events
```

---

### Minor Issues / Enhancements

#### 8. **No Episode Termination on Pokemon Faint/Blackout**

**Location**: `server/pokemonred_env_environment.py` line 192

**Problem**: Episode only terminates when `max_steps` is reached. There's no termination when all Pokemon faint (blackout), which is a natural episode boundary in the game.

**Suggested Fix**: Add blackout detection:
```python
def _check_blackout(self) -> bool:
    """Check if player has blacked out (all Pokemon fainted)."""
    return self._get_hp_fraction() == 0.0 and not self._is_in_battle()

# In step():
done = self._state.step_count >= self.config.max_steps or self._check_blackout()
```

---

#### 9. **HP Reading May Return Invalid Values for Empty Party Slots**

**Location**: `server/pokemonred_env_environment.py` lines 255-259

**Problem**: The code reads HP from all 6 party slots regardless of actual party size. Empty slots may contain garbage data.

**Suggested Fix**: Check party size first:
```python
PARTY_SIZE_ADDR = 0xD163

def _get_hp_fraction(self) -> float:
    party_size = min(self.pyboy.memory[PARTY_SIZE_ADDR], 6)
    hp_sum = sum(self._read_hp(HP_ADDRESSES[i]) for i in range(party_size))
    max_hp_sum = sum(self._read_hp(MAX_HP_ADDRESSES[i]) for i in range(party_size))
    return hp_sum / max(max_hp_sum, 1)
```

---

#### 10. **Missing Type Hints in Some Functions**

**Location**: Multiple files

**Problem**: Some internal methods lack type hints (e.g., `_get_state_dict()` return type).

**Suggested Fix**: Add complete type annotations for better IDE support and documentation:
```python
def _get_state_dict(self) -> Dict[str, Union[int, float]]:
    ...
```

---

#### 11. **Config Path Default May Not Work in All Deployment Modes**

**Location**: `config.py` line 19

**Problem**: `init_state` default path `/app/src/envs/pokemon_red/server/init.state` assumes a specific deployment structure that may not match the actual Docker layout.

**Suggested Fix**: Use a more flexible default or remove the default entirely:
```python
init_state: str = Field(
    default="",  # Empty means "use environment variable or auto-detect"
    description="Path to initial save state"
)
```

---

#### 12. **Legal Actions Always Returns Full Range**

**Location**: `server/pokemonred_env_environment.py` line 226

**Problem**: `legal_actions` always returns `[0,1,2,3,4,5,6]` regardless of game state. During battles or menus, some actions may be more relevant than others.

**Impact**: Agent cannot leverage action masking for more efficient learning.

**Suggested Enhancement** (Low Priority):
```python
def _get_legal_actions(self) -> List[int]:
    """Return contextually relevant actions."""
    if self._is_in_battle():
        # In battle, directional inputs navigate menu
        return [0, 1, 2, 3, 4, 5]  # Exclude Start
    return list(range(7))
```

---

#### 13. **No Support for Save State Export**

**Location**: `server/pokemonred_env_environment.py`

**Problem**: The environment can load save states but cannot export the current state for later resumption or agent checkpointing.

**Suggested Fix**:
```python
def export_state(self, path: Optional[str] = None) -> bytes:
    """Export current emulator state."""
    buffer = io.BytesIO()
    self.pyboy.save_state(buffer)
    state_bytes = buffer.getvalue()
    if path:
        with open(path, 'wb') as f:
            f.write(state_bytes)
    return state_bytes
```

---

#### 14. **Green Agent Tracker psutil Dependency Is Optional But Not Documented**

**Location**: `wrappers/green_agent.py` lines 19-23

**Problem**: The code handles missing `psutil` gracefully, but `psutil` is listed as a required dependency in `pyproject.toml`. This inconsistency is confusing.

**Suggested Fix**: Either make psutil truly optional with proper documentation, or remove the fallback code since it's a required dependency.

---

#### 15. **Duplicate Dependency Specifications**

**Location**: `pyproject.toml` lines 20-22

**Problem**: Both `openenv>=0.1.0` (PyPI) and `openenv-core @ git+...` (GitHub) are listed as dependencies. This could cause version conflicts.

**Suggested Fix**: Use only one source. The GitHub version is newer:
```toml
dependencies = [
    # ... other deps ...
    "openenv-core @ git+https://github.com/meta-pytorch/OpenEnv.git",
    # Remove: "openenv>=0.1.0",
]
```

---

## Implementation Priority

For agents implementing fixes, prioritize in this order:

### High Priority (Should be fixed first)
1. Issue #2: FrameStackWrapper incompatibility
2. Issue #3: PyBoy not properly closed
3. Issue #1: Modular reward system integration

### Medium Priority
4. Issue #4: Silent failures in global map conversion
5. Issue #8: Episode termination on blackout
6. Issue #9: HP reading for empty party slots
7. Issue #6: Redundant done flag

### Low Priority (Nice to have)
8. Issue #5: Import fallback pattern
9. Issue #7: Events.json not used
10. Issue #10-15: Minor enhancements

---

## Additional Notes for Agent Developers

### Running the Environment Locally

```bash
# Navigate to the pokemonred_env directory
cd pokemonred_env

# Install dependencies
uv sync

# Run the server (requires ROM file in server/)
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Testing Changes

After making changes, verify:
1. Server starts without errors
2. `reset()` returns valid observation
3. `step()` executes actions correctly
4. Reward calculation matches expected behavior
5. No resource leaks (check memory over multiple episodes)

### PyBoy Documentation Reference

For advanced emulator interactions, refer to:
- https://docs.pyboy.dk/
- https://docs.pyboy.dk/plugins/game_wrapper_pokemon_gen1.html

The `GameWrapperPokemonGen1` class provides additional methods like `game_area_collision()` for walkability detection that could enhance the observation space.
