---
name: Pokemon HRL VLM Agent
overview: Implement a Hierarchical Reinforcement Learning system with Gemma3-12B as Manager and LFM2.5-VL-1.2B as Worker to play Pokemon Red, using Unsloth for efficient training on RTX 5090.
todos:
  - id: phase1-setup
    content: "Phase 1: Create project structure, pyproject.toml, and abstract interfaces"
    status: completed
  - id: phase1-unsloth
    content: "Phase 1: Implement Unsloth utilities for model loading"
    status: completed
  - id: phase1-manager
    content: "Phase 1: Implement GemmaManager with instruction generation"
    status: completed
  - id: phase1-worker
    content: "Phase 1: Implement LiquidWorker with action selection"
    status: completed
  - id: phase1-verify
    content: "Phase 1: Create verification script, confirm models fit in VRAM"
    status: completed
  - id: phase2-wrapper
    content: "Phase 2: Implement HRL environment wrapper"
    status: completed
  - id: phase2-obs
    content: "Phase 2: Implement observation builder with configurable formats"
    status: completed
  - id: phase2-events
    content: "Phase 2: Implement event detector for Manager triggering"
    status: completed
  - id: phase2-test
    content: "Phase 2: Test random agent through full pipeline"
    status: cancelled
  - id: phase3-rewards
    content: "Phase 3: Implement Manager and Worker reward calculators"
    status: completed
  - id: phase3-verifier
    content: "Phase 3: Implement instruction verifier"
    status: completed
  - id: phase4-ppo
    content: "Phase 4: Implement PPO training with TRL (or CleanRL fallback)"
    status: completed
  - id: phase4-buffer
    content: "Phase 4: Implement hierarchical rollout buffer"
    status: completed
  - id: phase4-trainer
    content: "Phase 4: Implement main HRL trainer"
    status: completed
  - id: phase5-curriculum
    content: "Phase 5: Implement optional curriculum wrapper"
    status: completed
  - id: phase6-config
    content: "Phase 6: Create configuration schema and loader"
    status: completed
  - id: phase6-main
    content: "Phase 6: Create main.py entry point with wandb integration"
    status: completed
isProject: false
---

# Pokemon Hierarchical VLM Agent Implementation Plan

## Project Structure

Create a new repository `pokemon-vla-agent` in the same parent directory as `Pokemon_Red_OpenEnv`:

```
/home/neo/Desktop/Projects/OpenSource/Pokemon_Red_OpenEnv/
├── OpenEnv_Challenge/Pokemon_Red_OpenEnv/  # Existing environment
└── PokemonRed_VLM_Agent/                    # New agent repo (create here)
```

---

## Phase 1: Project Setup and Model Loading

### 1.1 Initialize Project Structure

Create the directory structure from [OVERVIEW.md](../PokemonRed_VLM_Agent/OVERVIEW.md) Section 4:

- `configs/` - YAML configuration files
- `core/` - Abstract interfaces and utilities
- `models/` - Manager and Worker implementations
- `rewards/` - Reward components
- `training/` - PPO and trainer classes
- `wrappers/` - Environment wrappers

### 1.2 Set Up Dependencies

Create `pyproject.toml` with:

```toml
[project]
requires-python = ">=3.13, <3.14"

dependencies = [
    "torch>=2.5.0",
    "transformers>=5.0.0",
    "unsloth",
    "trl>=0.12.0",
    "peft>=0.14.0",
    "wandb",
    "pydantic>=2.0",
    "pyyaml",
    "pillow",
    "numpy",
]
```

Also add the local `pokemonred_env`  (path: /home/neo/Desktop/Projects/OpenSource/Pokemon_Red_OpenEnv/OpenEnv_Challenge/Pokemon_Red_OpenEnv/pokemonred_env) as an editable dependency.

NOTE: Make sure that the python version is 3.13.X, NOT 3.14!

### 1.3 Implement Abstract Agent Interface

File: `core/interfaces.py`

```python
class AbstractAgent(nn.Module, ABC):
    @abstractmethod
    def load_backbone(self, config: AgentConfig) -> None: ...
    @abstractmethod
    def forward(self, images: List[Image], text: str) -> Union[torch.Tensor, str]: ...
    @abstractmethod
    def save_adapter(self, path: str) -> None: ...
    def freeze(self) -> None: ...
    def unfreeze(self) -> None: ...
```

### 1.4 Implement Unsloth Utilities

File: `core/unsloth_utils.py`

- `load_model_with_unsloth()` - Load model with Unsloth optimizations
- `setup_lora_adapters()` - Configure LoRA for training
- `get_quantization_config()` - 4-bit or BF16 config

### 1.5 Implement Manager (GemmaManager)

File: `models/manager.py`

- Load Gemma3-12B via Unsloth in 4-bit
- Add instruction generation head
- Add value head for PPO
- Implement `generate_instruction(frame, metadata) -> str`

### 1.6 Implement Worker (LiquidWorker)

File: `models/worker.py`

- Load LFM2.5-VL-1.2B via Unsloth
- Add action head (7 outputs for Pokemon Red actions)
- Add value head for PPO
- Implement `select_action(frames, instruction) -> int or List[int]`

### 1.7 Verification Script

File: `scripts/verify_models.py`

- Load both models
- Perform dummy forward pass
- Report VRAM usage
- Confirm no OOM on RTX 5090

---

## Phase 2: Environment Integration

### 2.1 HRL Wrapper

File: `wrappers/hrl_wrapper.py`

Wrap `PokemonRedEnv` client:

```python
class HRLEnvWrapper:
    def __init__(self, env: PokemonRedEnv, config: EnvConfig):
        self.env = env
        self.frame_stack = FrameStackWrapper(env, stack_size=4)
        self.current_instruction: str = ""
        self.steps_since_manager: int = 0
        
    def should_trigger_manager(self) -> bool:
        # Check step count or game events
        
    def step(self, action: int) -> Tuple[Observation, float, bool, dict]:
        # Execute action, track state changes
```

### 2.2 Observation Builder

File: `wrappers/observation_builder.py`

Build observations for Manager and Worker:

- `build_manager_observation(obs, metadata) -> ManagerInput`
- `build_worker_observation(obs, instruction, metadata) -> WorkerInput`

Configurable formats: single frame, stacked frames, structured metadata.

### 2.3 Event Detector

File: `core/event_detector.py`

Detect game events for Manager triggering:

```python
class EventDetector:
    def detect_events(self, prev_obs, curr_obs) -> List[GameEvent]:
        events = []
        if prev_obs.in_battle != curr_obs.in_battle:
            events.append(GameEvent.BATTLE_STATE_CHANGE)
        if prev_obs.position[2] != curr_obs.position[2]:
            events.append(GameEvent.MAP_CHANGE)
        # ... more event types
        return events
```

### 2.4 Random Agent Test

File: `scripts/test_random_agent.py`

- Connect to environment server
- Run random actions through Worker (random weights)
- Log video to wandb
- Verify full pipeline works

---

## Phase 3: Reward System

### 3.1 Manager Rewards

File: `rewards/manager_rewards.py`

Long-horizon game progress rewards:

```python
class ManagerRewardCalculator:
    def calculate(self, trajectory: List[Observation]) -> float:
        # Exploration: new coordinates
        # Badges: gym badge acquisition
        # Levels: party level increases
        # Events: game event flags
```

### 3.2 Worker Rewards

File: `rewards/worker_rewards.py`

Instruction-following rewards:

```python
class WorkerRewardCalculator:
    def calculate(self, instruction: str, obs_before, obs_after) -> float:
        # Verify instruction completion
        # Partial progress credit
        # Efficiency bonus
```

### 3.3 Instruction Verifier

File: `rewards/instruction_verifier.py`

Verify instruction completion:

```python
class InstructionVerifier:
    def verify_movement(self, instruction: str, pos_before, pos_after) -> float:
        # Parse direction from instruction
        # Check coordinate delta matches
        
    def verify_navigation(self, instruction: str, pos, target) -> float:
        # Check if closer to target
        
    def verify_interaction(self, instruction: str, events_before, events_after) -> float:
        # Check if relevant event triggered
```

---

## Phase 4: PPO Training

### 4.1 PPO Implementation

File: `training/ppo.py`

Using TRL (fallback to CleanRL):

```python
class HierarchicalPPOTrainer:
    def __init__(self, manager: AbstractAgent, worker: AbstractAgent, config: TrainingConfig):
        self.manager_trainer = PPOTrainer(manager, ...)  # TRL
        self.worker_trainer = PPOTrainer(worker, ...)
        
    def train_step(self, rollout_buffer: RolloutBuffer):
        # Update worker on instruction-following
        # Update manager on game progress (less frequently)
```

### 4.2 Rollout Buffer

File: `core/memory.py`

Store trajectories for PPO:

```python
class HierarchicalRolloutBuffer:
    def __init__(self, config: BufferConfig):
        self.manager_buffer = RolloutBuffer(...)  # Long-horizon
        self.worker_buffer = RolloutBuffer(...)   # Per-step
        
    def add_worker_step(self, obs, action, reward, done, log_prob, value): ...
    def add_manager_episode(self, obs, instruction, cumulative_reward): ...
```

### 4.3 Main Trainer

File: `training/trainer.py`

Orchestrate the full training loop:

```python
class HRLTrainer:
    def train(self):
        for episode in range(num_episodes):
            obs = self.env.reset()
            instruction = self.manager.generate_instruction(obs)
            
            while not done:
                if self.env.should_trigger_manager():
                    instruction = self.manager.generate_instruction(obs)
                    
                action = self.worker.select_action(obs, instruction)
                obs, reward, done, info = self.env.step(action)
                
                # Compute rewards, store in buffers
                
            # PPO updates
            self.ppo_trainer.train_step(self.rollout_buffer)
```

---

## Phase 5: Curriculum Learning (Optional Wrapper)

### 5.1 Curriculum Wrapper

File: `training/curriculum.py`

Easily enable/disable curriculum:

```python
class CurriculumWrapper:
    def __init__(self, base_trainer: HRLTrainer, config: CurriculumConfig):
        self.levels = [
            CurriculumLevel(1, "single_button", ...),
            CurriculumLevel(2, "movement", ...),
            CurriculumLevel(3, "navigation", ...),
            CurriculumLevel(4, "complex_goals", ...),
        ]
        self.current_level = 1
        self.enabled = config.curriculum_enabled
        
    def get_instruction(self, obs) -> str:
        if not self.enabled:
            return self.manager.generate_instruction(obs)
        return self.levels[self.current_level].generate_synthetic_instruction(obs)
```

This wrapper can be disabled via config to use full Manager generation.

---

## Phase 6: Configuration and Entry Point

### 6.1 Configuration Schema

File: `configs/defaults.yaml`

See [OVERVIEW.md](../PokemonRed_VLM_Agent/OVERVIEW.md) Section 9 for full config schema.

### 6.2 Config Loader

File: `core/config.py`

Pydantic models for type-safe configuration:

```python
class ManagerConfig(BaseModel):
    model: str = "google/gemma-3-12b-it"
    trigger_steps: int = 64
    trigger_on_events: bool = True
    instruction_granularity: Literal["low", "mid", "high", "adaptive"] = "mid"
    frozen: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
```

### 6.3 Main Entry Point

File: `main.py`

```python
@click.command()
@click.option("--config", default="configs/defaults.yaml")
def main(config: str):
    cfg = load_config(config)
    wandb.init(project=cfg.logging.project_name)
    
    manager = GemmaManager(cfg.manager)
    worker = LiquidWorker(cfg.worker)
    env = HRLEnvWrapper(PokemonRedEnv(...), cfg.environment)
    
    trainer = HRLTrainer(manager, worker, env, cfg.training)
    trainer.train()
```

---

## Environment Changes Needed

Create `ENV_CHANGES_NEEDED.md` documenting required environment modifications:

1. **Menu State Detection** - Memory read for active menu type
2. **Move Selection Detection** - Battle menu cursor position
3. **Additional Event Flags** - NPC interaction, item pickup, etc.

These will be delegated to a separate agent/session for implementation.

---

## Testing Strategy

### Unit Tests

- Model loading and forward pass
- Reward calculation
- Instruction verification

### Integration Tests

- Full episode with random agent
- PPO update cycle
- Checkpoint save/load

### Manual Verification

- wandb video logs show sensible behavior
- VRAM usage stays within bounds
- Training loss decreases

---

## Key Files Summary

| File | Purpose |

|------|---------|

| `core/interfaces.py` | Abstract agent base class |

| `models/manager.py` | Gemma3-12B Manager |

| `models/worker.py` | LFM2.5-VL-1.2B Worker |

| `wrappers/hrl_wrapper.py` | Environment wrapper |

| `rewards/instruction_verifier.py` | Verify instruction completion |

| `training/ppo.py` | PPO training logic |

| `training/trainer.py` | Main training orchestration |

| `training/curriculum.py` | Optional curriculum wrapper |

| `configs/defaults.yaml` | Default configuration |

| `main.py` | Entry point |