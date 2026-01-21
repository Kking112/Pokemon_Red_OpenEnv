# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Event reward component for Pokemon Red."""

from typing import Any, Dict

from .base import BaseRewardComponent


class EventReward(BaseRewardComponent):
    """
    Rewards triggering in-game event flags.
    
    Pokemon Red uses event flags to track story progression,
    item collection, NPC interactions, etc. This component
    rewards activating new event flags.
    
    Attributes:
        weight: Reward per event triggered (default 0.1).
    """
    
    def __init__(self, weight: float = 0.1, enabled: bool = True):
        super().__init__(name="event", weight=weight, enabled=enabled)
    
    def calculate(
        self, state: Dict[str, Any], prev_state: Dict[str, Any]
    ) -> float:
        """Calculate reward for new events triggered."""
        current_events = state.get("event_count", 0)
        previous_events = prev_state.get("event_count", 0)
        new_events = current_events - previous_events
        return max(0.0, float(new_events))
