# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Level up reward component for Pokemon Red."""

from typing import Any, Dict

from .base import BaseRewardComponent


class LevelUpReward(BaseRewardComponent):
    """
    Rewards party Pokemon level increases.
    
    Tracks the sum of all party Pokemon levels and provides
    reward when the total increases (from battles/training).
    
    Attributes:
        weight: Reward per level gained (default 1.0).
    """
    
    def __init__(self, weight: float = 1.0, enabled: bool = True):
        super().__init__(name="level", weight=weight, enabled=enabled)
    
    def calculate(
        self, state: Dict[str, Any], prev_state: Dict[str, Any]
    ) -> float:
        """Calculate reward for level increases."""
        current_levels = state.get("level_sum", 0)
        previous_levels = prev_state.get("level_sum", 0)
        level_gain = current_levels - previous_levels
        return max(0.0, float(level_gain))
