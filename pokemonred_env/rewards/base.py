# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Base class for modular reward components."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseRewardComponent(ABC):
    """
    Abstract base class for reward components.
    
    Each reward component calculates a specific aspect of the reward signal
    (exploration, badges, levels, etc.) and can be enabled/weighted via config.
    
    Attributes:
        name: Unique identifier for this reward component.
        weight: Scaling factor applied to calculated reward.
        enabled: Whether this component is active.
    
    Example:
        >>> class MyReward(BaseRewardComponent):
        ...     def calculate(self, state, prev_state):
        ...         return state["score"] - prev_state.get("score", 0)
        >>> 
        >>> reward = MyReward(weight=2.0)
        >>> reward.calculate({"score": 100}, {"score": 50})
        100.0  # (100 - 50) * 2.0
    """
    
    def __init__(self, name: str, weight: float = 1.0, enabled: bool = True):
        """
        Initialize reward component.
        
        Args:
            name: Unique identifier for this component.
            weight: Scaling factor for reward (default 1.0).
            enabled: Whether component is active (default True).
        """
        self.name = name
        self.weight = weight
        self.enabled = enabled
        self._cumulative = 0.0
    
    @abstractmethod
    def calculate(
        self, state: Dict[str, Any], prev_state: Dict[str, Any]
    ) -> float:
        """
        Calculate reward for this component.
        
        Args:
            state: Current game state dictionary.
            prev_state: Previous game state dictionary.
        
        Returns:
            Unweighted reward value for this component.
        """
        pass
    
    def get_reward(
        self, state: Dict[str, Any], prev_state: Dict[str, Any]
    ) -> float:
        """
        Get weighted reward if enabled.
        
        Args:
            state: Current game state.
            prev_state: Previous game state.
        
        Returns:
            Weighted reward if enabled, 0.0 otherwise.
        """
        if not self.enabled:
            return 0.0
        reward = self.calculate(state, prev_state) * self.weight
        self._cumulative += reward
        return reward
    
    def reset(self) -> None:
        """Reset component state for new episode."""
        self._cumulative = 0.0
    
    @property
    def cumulative_reward(self) -> float:
        """Total reward from this component in current episode."""
        return self._cumulative
