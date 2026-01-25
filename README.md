# Pokemon Red OpenEnv

A port of an existing Pokemon Red RL Environment from the Gymnasium API to the **OpenEnv API**, enabling VLMs/VLAs as policies for playing Pokemon Red. This project is built for [The OpenEnv Challenge](https://github.com/meta-pytorch/OpenEnv) hackathon.

![Pokemon Red](https://upload.wikimedia.org/wikipedia/en/f/f1/Pok%C3%A9mon_Red_cover.png)

## 🎯 Overview

This repository provides a **production-ready RL environment** for training AI agents to play Pokemon Red using modern reinforcement learning frameworks. The environment exposes the game through HTTP/WebSocket APIs following the OpenEnv standard.

**Key Features:**
- **VLA-Ready**: Base64-encoded screen observations for Vision-Language-Action models
- **OpenEnv Compliant**: Type-safe Pydantic models, HTTP/WebSocket API
- **Modular Rewards**: Exploration, badges, levels, and event-based rewards
- **Green Agent**: Built-in compute efficiency tracking

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/Pokemon_Red_OpenEnv.git
cd Pokemon_Red_OpenEnv/pokemonred_env

# Install dependencies
pip install -e .

# Or with uv
uv sync
```

## 🚀 Quick Start

```python
from pokemonred_env import PokemonRedEnv, PokemonRedAction

# Connect to running server
env = PokemonRedEnv(base_url="http://localhost:8000")

# Reset environment
result = env.reset()

# Take actions (0=Down, 1=Left, 2=Right, 3=Up, 4=A, 5=B, 6=Start)
for _ in range(100):
    result = env.step(PokemonRedAction(action=4))  # Press A
    print(f"Reward: {result.reward:.2f}")

env.close()
```

---

## 🤖 Training with RL Frameworks

### TRL (Transformer Reinforcement Learning)

[TRL](https://github.com/huggingface/trl) by HuggingFace provides trainers for RLHF, PPO, and GRPO. Perfect for training VLMs on Pokemon Red.

#### GRPO Training with Custom Reward

```python
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForVision2Seq, AutoProcessor
from pokemonred_env import PokemonRedEnv, PokemonRedAction
import base64
from PIL import Image
import io

# Connect to environment
env = PokemonRedEnv(base_url="http://localhost:8000")

# Create dataset of game prompts
prompts = [
    {"prompt": "You are playing Pokemon Red. What action should you take?"}
    for _ in range(1000)
]
dataset = Dataset.from_list(prompts)

# Custom reward function using game state
def pokemon_reward_function(completions, **kwargs):
    """Evaluate completions by executing them in the environment."""
    rewards = []
    for completion in completions:
        # Parse action from completion (e.g., "Press A" -> 4)
        action_id = parse_action_from_text(completion)
        result = env.step(PokemonRedAction(action=action_id))
        rewards.append(result.reward)
    return rewards

# Configure GRPO trainer
trainer = GRPOTrainer(
    model="Qwen/Qwen2-VL-7B-Instruct",
    reward_funcs=pokemon_reward_function,
    train_dataset=dataset,
    args=GRPOConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=1,
        learning_rate=5e-6,
        output_dir="pokemon_grpo_outputs",
        num_generations=4,
    ),
)

trainer.train()
```

#### PPO Training for Classic RL

```python
from trl import PPOTrainer, PPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
tokenizer.pad_token = tokenizer.eos_token

# Configure PPO
config = PPOConfig(
    model_name="gpt2-medium",
    learning_rate=1.41e-5,
    batch_size=128,
    mini_batch_size=16,
    ppo_epochs=4,
    target_kl=0.1,
)

# Initialize trainer with environment feedback
ppo_trainer = PPOTrainer(
    args=config,
    processing_class=tokenizer,
    model=model,
    train_dataset=dataset,
)
```

---

### Unsloth (Fast Fine-tuning)

[Unsloth](https://github.com/unslothai/unsloth) enables **2x faster training** with **70% less VRAM**. Excellent for fine-tuning VLMs on game data.

#### GRPO Training with Vision Models

```python
from unsloth import FastVisionModel, PatchFastRL
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset
from pokemonred_env import PokemonRedEnv, PokemonRedAction
import base64, io
from PIL import Image

# Load vision-language model with Unsloth optimizations
model, tokenizer = FastVisionModel.from_pretrained(
    model_name="unsloth/Qwen2-VL-7B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,  # 70% less VRAM
)

# Add LoRA adapters
model = FastVisionModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    finetune_vision_layers=True,
    finetune_language_layers=True,
)

# Patch for RL training (GSPO for vision)
model = PatchFastRL(
    model,
    FastLanguageModel=FastVisionModel,
    algorithm="grpo",
)

# Connect to Pokemon environment
env = PokemonRedEnv(base_url="http://localhost:8000")

def decode_screen(screen_b64: str) -> Image.Image:
    """Decode base64 screen to PIL Image."""
    return Image.open(io.BytesIO(base64.b64decode(screen_b64)))

# Custom reward function for vision tasks
def pokemon_vision_reward(samples):
    rewards = []
    for sample in samples:
        # Parse action from model output
        action_text = sample.get("output", "")
        action_id = parse_action_from_text(action_text)
        
        # Execute in environment
        result = env.step(PokemonRedAction(action=action_id))
        rewards.append(result.reward)
    return rewards

# Create trainer
trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=vision_dataset,
    reward_funcs=[pokemon_vision_reward],
    args=GRPOConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=1,
        learning_rate=5e-6,
        output_dir="pokemon_unsloth_outputs",
        num_generations=4,
    ),
)

trainer.train()
model.save_pretrained("pokemon_vla_model")
```

#### FP8 Acceleration

```python
from unsloth import FastLanguageModel, PatchFastRL

# Load with FP8 for maximum speed
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Enable FP8 acceleration
model = PatchFastRL(
    model,
    FastLanguageModel=FastLanguageModel,
    algorithm="grpo",
    use_fp8=True,  # 2x faster training
)
```

---

### TorchRL (PyTorch RL)

[TorchRL](https://github.com/pytorch/rl) is PyTorch's official RL library with TensorDict support for efficient data handling.

#### PPO Training with Custom Environment

```python
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import TensorDictReplayBuffer, LazyTensorStorage
from torchrl.envs import EnvBase
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from pokemonred_env import PokemonRedEnv, PokemonRedAction
import base64
import numpy as np
from PIL import Image

class PokemonTorchRLEnv(EnvBase):
    """TorchRL wrapper for Pokemon Red OpenEnv."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        super().__init__()
        self.env = PokemonRedEnv(base_url=base_url)
        
    def _reset(self, tensordict=None):
        result = self.env.reset()
        screen = self._decode_screen(result.observation.screen_b64)
        return TensorDict({
            "observation": torch.tensor(screen, dtype=torch.float32) / 255.0,
            "done": torch.tensor([result.observation.done]),
        })
    
    def _step(self, tensordict):
        action = tensordict["action"].item()
        result = self.env.step(PokemonRedAction(action=int(action)))
        screen = self._decode_screen(result.observation.screen_b64)
        return TensorDict({
            "observation": torch.tensor(screen, dtype=torch.float32) / 255.0,
            "reward": torch.tensor([result.reward]),
            "done": torch.tensor([result.observation.done]),
        })
    
    def _decode_screen(self, screen_b64: str) -> np.ndarray:
        img = Image.open(io.BytesIO(base64.b64decode(screen_b64)))
        return np.array(img)

# Create environment
env = PokemonTorchRLEnv()

# Vision encoder (CNN for game screen)
vision_encoder = nn.Sequential(
    nn.Conv2d(3, 32, 8, stride=4),
    nn.ReLU(),
    nn.Conv2d(32, 64, 4, stride=2),
    nn.ReLU(),
    nn.Conv2d(64, 64, 3, stride=1),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(3136, 512),
    nn.ReLU(),
)

# Actor network
actor_net = nn.Sequential(
    vision_encoder,
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Linear(128, 7),  # 7 actions
)

# Critic network
critic_net = nn.Sequential(
    vision_encoder,
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
)

actor = TensorDictModule(actor_net, in_keys=["observation"], out_keys=["logits"])
critic = ValueOperator(critic_net, in_keys=["observation"])

# Setup PPO training
collector = SyncDataCollector(
    env,
    actor,
    frames_per_batch=1000,
    total_frames=1_000_000,
)

loss_fn = ClipPPOLoss(actor, critic)
adv_fn = GAE(value_network=critic, average_gae=True, gamma=0.99, lmbda=0.95)
optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-4)

# Training loop
for batch in collector:
    adv_fn(batch)
    loss = loss_fn(batch)
    optimizer.zero_grad()
    loss["loss_objective"].backward()
    optimizer.step()
```

---

## 🏗️ Architecture

```
Pokemon_Red_OpenEnv/
├── README.md                    # This file (RL integration guide)
├── pokemonred_env/              # OpenEnv environment package
│   ├── README.md                # Environment documentation
│   ├── client.py                # HTTP/WebSocket client
│   ├── models.py                # Pydantic action/observation models
│   ├── config.py                # Configuration
│   ├── rewards/                 # Modular reward system
│   │   ├── base.py              # BaseRewardComponent ABC
│   │   ├── manager.py           # RewardManager
│   │   ├── exploration.py       # Exploration rewards
│   │   ├── badge.py             # Badge rewards
│   │   └── level.py             # Level-up rewards
│   ├── wrappers/                # Environment wrappers
│   │   └── green_agent.py       # Compute efficiency tracking
│   └── server/                  # FastAPI server
│       ├── app.py               # HTTP/WebSocket endpoints
│       ├── pokemonred_env_environment.py  # Core PyBoy environment
│       └── Dockerfile           # Container definition
```

---

## 📊 Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `screen_b64` | `str` | Base64 PNG screenshot (160×144) |
| `health` | `float` | Party HP fraction [0.0, 1.0] |
| `level_sum` | `int` | Sum of party Pokemon levels |
| `badges` | `List[int]` | 8-element badge flags |
| `position` | `List[int]` | Player [x, y, map_id] |
| `in_battle` | `bool` | Battle state |

## 🎮 Action Space

| Index | Button | Index | Button |
|-------|--------|-------|--------|
| 0 | Down | 4 | A |
| 1 | Left | 5 | B |
| 2 | Right | 6 | Start |
| 3 | Up | | |

---

## 🚢 Deployment

```bash
# Build Docker image
cd pokemonred_env
openenv build

# Run (provide your own ROM)
docker run -p 8000:8000 \
  -v /path/to/PokemonRed.gb:/rom/PokemonRed.gb \
  pokemonred-env:latest

# Push to HuggingFace
openenv push --repo-id your-username/pokemon-red-openenv
```

---

## 📜 Citations

This is a heavily modified version of:
- [PokemonRedExperiments](https://github.com/PWhiddy/PokemonRedExperiments) by Peter Whidden

Built with:
- [OpenEnv](https://github.com/meta-pytorch/OpenEnv) - Environment standard
- [PyBoy](https://github.com/Baekalfen/PyBoy) - Game Boy emulator
- [TRL](https://github.com/huggingface/trl) - Transformer RL
- [Unsloth](https://github.com/unslothai/unsloth) - Fast fine-tuning
- [TorchRL](https://github.com/pytorch/rl) - PyTorch RL library

## 📄 License

This environment wrapper is open source. Pokemon Red ROM and assets are © Nintendo/Game Freak and must be obtained separately.