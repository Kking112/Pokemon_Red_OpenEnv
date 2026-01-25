# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Reward manager for aggregating modular reward components."""

from typing import Any, Dict, List, Optional, Type

from .base import BaseRewardComponent


class RewardManager:
    """
    Manages multiple reward components and aggregates their outputs.
    
    Supports dynamic registration, per-component weighting, and
    detailed reward breakdowns for debugging/analysis.
    
    Attributes:
        components: List of registered reward components.
        global_scale: Global scaling factor applied to total reward.
    
    Example:
        >>> manager = RewardManager(global_scale=1.0)
        >>> manager.register(ExplorationReward(weight=2.0))
        >>> manager.register(BadgeReward(weight=5.0))
        >>> 
        >>> reward = manager.calculate(current_state, prev_state)
        >>> breakdown = manager.get_breakdown()
    """
    
    def __init__(self, global_scale: float = 1.0):
        """
        Initialize reward manager.
        
        Args:
            global_scale: Global scaling factor for total reward.
        """
        self.components: List[BaseRewardComponent] = []
        self.global_scale = global_scale
        self._last_breakdown: Dict[str, float] = {}
    
    def register(self, component: BaseRewardComponent) -> "RewardManager":
        """
        Register a reward component.
        
        Args:
            component: Reward component to register.
        
        Returns:
            Self for method chaining.
        """
        self.components.append(component)
        return self
    
    def register_defaults(self, config: Optional[Dict[str, Any]] = None) -> "RewardManager":
        """
        Register default reward components with optional config.
        
        Args:
            config: Optional config dict with component weights.
        
        Returns:
            Self for method chaining.
        """
        from .exploration import ExplorationReward
        from .badge import BadgeReward
        from .level import LevelUpReward
        from .event import EventReward
        
        config = config or {}
        
        self.register(ExplorationReward(
            weight=config.get("exploration_weight", 0.02)
        ))
        self.register(BadgeReward(
            weight=config.get("badge_weight", 5.0)
        ))
        self.register(LevelUpReward(
            weight=config.get("level_weight", 1.0)
        ))
        self.register(EventReward(
            weight=config.get("event_weight", 0.1)
        ))
        
        return self
    
    def calculate(
        self, state: Dict[str, Any], prev_state: Dict[str, Any]
    ) -> float:
        """
        Calculate total reward from all components.
        
        Args:
            state: Current game state.
            prev_state: Previous game state.
        
        Returns:
            Total weighted and scaled reward.
        """
        self._last_breakdown = {}
        total = 0.0
        
        for component in self.components:
            reward = component.get_reward(state, prev_state)
            self._last_breakdown[component.name] = reward
            total += reward
        
        return total * self.global_scale
    
    def reset(self) -> None:
        """Reset all components for new episode."""
        self._last_breakdown = {}
        for component in self.components:
            component.reset()
    
    def get_breakdown(self) -> Dict[str, float]:
        """Get reward breakdown by component from last calculation."""
        return self._last_breakdown.copy()
    
    def get_cumulative_breakdown(self) -> Dict[str, float]:
        """Get cumulative rewards by component for current episode."""
        return {
            comp.name: comp.cumulative_reward
            for comp in self.components
        }
