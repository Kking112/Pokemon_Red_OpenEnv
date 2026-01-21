"""
pokemonred_env/config.py
--------------------------------
Configuration schema for Pokemon Red environment.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any

class PokemonRedConfig(BaseModel):
    """
    Configuration for Pokemon Red Environment.
    """
    # Core settings
    session_path: str = Field("session_pokemon_vla", description="Path to save session data")
    save_final_state: bool = Field(False, description="Save state at end of episode")
    print_rewards: bool = Field(False, description="Print reward details to stdout")
    headless: bool = Field(True, description="Run emulator without window")
    init_state: str = Field("/app/src/envs/pokemon_red/server/init.state", description="Path to initial save state")
    action_freq: int = Field(24, description="Emulator ticks per action")
    max_steps: int = Field(163840, description="Max steps per episode")
    save_video: bool = Field(False, description="Record video of episodes")
    fast_video: bool = Field(True, description="Optimize video recording speed")
    frame_stacks: int = Field(5, description="Number of recent actions to track")

    # Paths (relative to possible mount points)
    gb_path: str = Field("pokemonred_env/server/PokemonRed.gb", description="Path to ROM file")

    # Reward Configuration
    reward_scale: float = Field(1.0, description="Global reward scaling factor")
    explore_weight: float = Field(1.0, description="Weight for exploration reward")
    
    # Feature Flags
    include_game_text: bool = Field(True, description="Enable text extraction")
    text_detection_threshold: int = Field(10, description="Min chars to consider text active")
    
    # Instance ID
    instance_id: str = Field("default", description="Unique ID for this env instance")

    # Extra options mapping
    extra: Dict[str, Any] = Field(default_factory=dict, description="Additional backend options")
