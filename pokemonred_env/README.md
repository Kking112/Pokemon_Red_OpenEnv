---
title: Pokemon Red OpenEnv Environment
emoji: 🎮
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - pokemon
  - gameboy
  - vla
---

# 🎮 Pokemon Red OpenEnv Environment

A production-ready Reinforcement Learning environment for training AI agents to play **Pokemon Red** using the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) standard. Built with PyBoy emulator for accurate Game Boy emulation.

![Pokemon Red](https://upload.wikimedia.org/wikipedia/en/f/f1/Pok%C3%A9mon_Red_cover.png)

## ✨ Features

- **OpenEnv Compliant**: Full HTTP/WebSocket API with type-safe Pydantic models
- **VLA-Ready**: Base64-encoded screen observations for Vision-Language-Action models
- **Modular Rewards**: Pluggable reward components (exploration, badges, levels, events)
- **Green Agent**: Built-in compute efficiency tracking for sustainable AI
- **Docker Isolated**: Reproducible containerized emulator environment
- **Multi-Mode Deployment**: Docker, `uv run`, `openenv serve`, or Python module

## 🚀 Quick Start

### Install from HuggingFace Space

```python
from pokemonred_env import PokemonRedEnv, PokemonRedAction

# Connect to deployed environment
env = PokemonRedEnv(base_url="https://your-space.hf.space")

# Reset and get initial observation
result = env.reset()
print(f"Screen shape: {result.observation.screen_shape}")
print(f"Badges: {result.observation.badges}")

# Take actions: 0=Down, 1=Left, 2=Right, 3=Up, 4=A, 5=B, 6=Start
for _ in range(100):
    action = PokemonRedAction(action=4)  # Press A button
    result = env.step(action)
    print(f"Reward: {result.reward:.2f}, Health: {result.observation.health:.1%}")

env.close()
```

### Run Locally with Docker

```bash
# Build the Docker image
cd pokemonred_env
openenv build

# Run the server (mount your ROM file)
docker run -p 8000:8000 \
  -v /path/to/your/PokemonRed.gb:/rom/PokemonRed.gb \
  pokemonred-env:latest
```

> ⚠️ **ROM Required**: You must provide your own `PokemonRed.gb` ROM file. The `init.state` save file and additional state files for different game starting points are included in the repository.

NOTE: The init.state file begins the game after the opening credits and choosing of name, etc., where the player starts in the top floor of his parent's house.

## 🎯 Environment Details

### Action Space

| Index | Button | Description |
|-------|--------|-------------|
| 0 | Down | Move down |
| 1 | Left | Move left |
| 2 | Right | Move right |
| 3 | Up | Move up |
| 4 | A | Confirm/Interact |
| 5 | B | Cancel/Run |
| 6 | Start | Menu |

```python
from pokemonred_env import PokemonRedAction

# Press the A button
action = PokemonRedAction(action=4)
```

### Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `screen_b64` | `str` | Base64-encoded PNG screenshot (160×144 RGB) |
| `screen_shape` | `List[int]` | Shape of screen array `[144, 160, 3]` |
| `health` | `float` | Party HP fraction `[0.0, 1.0]` |
| `level_sum` | `int` | Sum of all party Pokemon levels |
| `badges` | `List[int]` | 8-element badge flags `[0,0,0,0,0,0,0,0]` |
| `position` | `List[int]` | Player position `[x, y, map_id]` |
| `in_battle` | `bool` | Whether player is in battle |
| `seen_coords_count` | `int` | Number of unique tiles visited |
| `legal_actions` | `List[int]` | Always `[0,1,2,3,4,5,6]` |
| `done` | `bool` | Episode termination flag |
| `reward` | `float` | Step reward |
| `metadata` | `Dict` | Additional context |

### Decoding the Screen

```python
import base64
from PIL import Image
import io

# Decode base64 screen to PIL Image
screen_bytes = base64.b64decode(result.observation.screen_b64)
image = Image.open(io.BytesIO(screen_bytes))

# Convert to numpy array for ML
import numpy as np
screen_array = np.array(image)  # Shape: (144, 160, 3)
```

## 🏆 Reward System

The environment uses a modular reward system with four components:

| Component | Default Weight | Trigger |
|-----------|----------------|---------|
| **Exploration** | 0.02 | New map coordinate visited |
| **Badge** | 5.0 | Gym badge obtained |
| **Level** | 1.0 | Party Pokemon level increase |
| **Event** | 0.1 | Story event flag triggered |

### Custom Reward Configuration

```python
from pokemonred_env.rewards import RewardManager, ExplorationReward, BadgeReward

# Create custom reward manager
manager = RewardManager(global_scale=1.0)
manager.register(ExplorationReward(weight=0.05))  # Increase exploration reward
manager.register(BadgeReward(weight=10.0))  # Double badge reward
```

## 🌱 Green Agent Tracking

Built-in compute efficiency metrics for sustainable AI development:

```python
from pokemonred_env.wrappers import GreenAgentTracker

tracker = GreenAgentTracker()

# Track a step
result, metrics = tracker.track_step(lambda: env.step(action))

print(f"Step time: {metrics['green_step_time_ms']:.2f}ms")
print(f"Memory: {metrics['green_current_memory_mb']:.1f}MB")
print(f"CPU: {metrics['green_cpu_percent']:.1f}%")
```

## 🤖 Training a VLA Agent

### Basic PPO Training Loop

```python
import torch
from pokemonred_env import PokemonRedEnv, PokemonRedAction
import base64
from PIL import Image
import io
import numpy as np

def decode_screen(screen_b64: str) -> np.ndarray:
    """Decode base64 screen to numpy array."""
    img = Image.open(io.BytesIO(base64.b64decode(screen_b64)))
    return np.array(img)

# Connect to environment
env = PokemonRedEnv.from_docker_image("pokemonred-env:latest")

try:
    for episode in range(1000):
        result = env.reset()
        episode_reward = 0
        
        for step in range(2048):
            # Get screen observation
            screen = decode_screen(result.observation.screen_b64)
            
            # Your VLA model inference here
            # action_id = model.predict(screen, result.observation)
            action_id = np.random.randint(0, 7)  # Random for demo
            
            # Take action
            result = env.step(PokemonRedAction(action=action_id))
            episode_reward += result.reward
            
            if result.observation.done:
                break
        
        print(f"Episode {episode}: Reward = {episode_reward:.2f}")
finally:
    env.close()
```

### Integration with Stable-Baselines3

```python
import gymnasium as gym
from stable_baselines3 import PPO
from pokemonred_env import PokemonRedEnv

# Wrap as Gymnasium environment
class PokemonGymEnv(gym.Env):
    def __init__(self, base_url: str):
        self.env = PokemonRedEnv(base_url=base_url)
        self.action_space = gym.spaces.Discrete(7)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(144, 160, 3), dtype=np.uint8
        )
    
    def reset(self, **kwargs):
        result = self.env.reset()
        return self._decode_obs(result.observation), {}
    
    def step(self, action):
        result = self.env.step(PokemonRedAction(action=int(action)))
        obs = self._decode_obs(result.observation)
        return obs, result.reward, result.observation.done, False, {}
    
    def _decode_obs(self, obs):
        img = Image.open(io.BytesIO(base64.b64decode(obs.screen_b64)))
        return np.array(img)

# Train with PPO
env = PokemonGymEnv("http://localhost:8000")
model = PPO("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)
```

## ⚙️ Environment Variables

Configure the environment via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `POKEMON_HEADLESS` | `true` | Run emulator without display |
| `POKEMON_ACTION_FREQ` | `24` | Emulator ticks per action |
| `POKEMON_ROM_PATH` | `/rom/PokemonRed.gb` | Path to ROM file (mount point) |
| `POKEMON_INIT_STATE` | `/app/env/server/has_pokedex.state` | Path to initial save state |
| `POKEMON_MAX_STEPS` | `163840` | Max steps per episode |
| `POKEMON_SESSION_PATH` | `/tmp/pokemon_sessions` | Session data directory |

## 📁 Project Structure

```
pokemonred_env/
├── __init__.py              # Package exports
├── client.py                # PokemonRedEnv HTTP/WebSocket client
├── config.py                # PokemonRedConfig Pydantic model
├── models.py                # Action, Observation, State models
├── openenv.yaml             # OpenEnv manifest
├── pyproject.toml           # Dependencies and metadata
├── rewards/                 # Modular reward system
│   ├── base.py              # BaseRewardComponent ABC
│   ├── manager.py           # RewardManager
│   ├── exploration.py       # Exploration rewards
│   ├── badge.py             # Badge rewards
│   ├── level.py             # Level-up rewards
│   └── event.py             # Event flag rewards
├── wrappers/                # Environment wrappers
│   └── green_agent.py       # Compute efficiency tracking
└── server/
    ├── app.py               # FastAPI server
    ├── pokemonred_env_environment.py  # Core environment
    ├── global_map.py        # Map coordinate utilities
    └── Dockerfile           # Container definition
```

## 🚢 Deployment

### Push to HuggingFace Spaces

```bash
# Validate environment
openenv validate --verbose

# Push to HuggingFace
openenv push --repo-id your-username/pokemon-red-openenv
```

### Endpoints

After deployment, your space provides:

| Endpoint | Description |
|----------|-------------|
| `/web` | Interactive web interface |
| `/docs` | OpenAPI/Swagger documentation |
| `/health` | Health check endpoint |
| `/ws` | WebSocket endpoint for low-latency sessions |
| `POST /reset` | Reset environment |
| `POST /step` | Execute action |
| `GET /state` | Get current state |

## 📜 License

This environment wrapper is open source. Pokemon Red ROM and assets are © Nintendo/Game Freak and must be obtained separately.

## 🙏 Acknowledgments

- [PyBoy](https://github.com/Baekalfen/PyBoy) - Game Boy emulator
- [OpenEnv](https://github.com/meta-pytorch/OpenEnv) - Environment standard
- [PokemonRedExperiments](https://github.com/PWhiddy/PokemonRedExperiments) - Original RL research
