# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Exploration reward component for Pokemon Red."""

from typing import Any, Dict

from .base import BaseRewardComponent


class ExplorationReward(BaseRewardComponent):
    """
    Rewards visiting new map coordinates.
    
    Encourages the agent to explore the game world by providing
    reward for each new unique (x, y, map_id) coordinate visited.
    
    Attributes:
        weight: Reward per new coordinate (default 0.02).
    """
    
    def __init__(self, weight: float = 0.02, enabled: bool = True):
        super().__init__(name="exploration", weight=weight, enabled=enabled)
    
    def calculate(
        self, state: Dict[str, Any], prev_state: Dict[str, Any]
    ) -> float:
        """Calculate reward for new coordinates visited."""
        current_count = state.get("seen_coords_count", 0)
        previous_count = prev_state.get("seen_coords_count", 0)
        new_coords = current_count - previous_count
        return max(0.0, float(new_coords))
