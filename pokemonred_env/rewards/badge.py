# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Badge reward component for Pokemon Red."""

from typing import Any, Dict

from .base import BaseRewardComponent


class BadgeReward(BaseRewardComponent):
    """
    Rewards obtaining gym badges.
    
    Provides significant reward for each gym badge obtained,
    as badges represent major game progression milestones.
    
    Attributes:
        weight: Reward per badge (default 5.0).
    """
    
    def __init__(self, weight: float = 5.0, enabled: bool = True):
        super().__init__(name="badge", weight=weight, enabled=enabled)
    
    def calculate(
        self, state: Dict[str, Any], prev_state: Dict[str, Any]
    ) -> float:
        """Calculate reward for badges obtained."""
        current_badges = state.get("badge_count", 0)
        previous_badges = prev_state.get("badge_count", 0)
        new_badges = current_badges - previous_badges
        return max(0.0, float(new_badges))
