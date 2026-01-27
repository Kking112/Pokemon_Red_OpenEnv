"""
Frame stacking wrapper for temporal observation fusion.

Provides frame stacking functionality to give the VLM agent temporal context
by combining multiple sequential frames into a single fused image.
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


class FrameStackWrapper:
    """
    Stack multiple frames into a single observation for temporal context.
    
    Can use grid layout (2x2 for 4 frames) or horizontal concatenation.
    This wrapper is designed to work with any environment that returns
    observations with an `image` attribute.
    
    Args:
        env: Base environment to wrap.
        stack_size: Number of frames to stack (default: 4).
        layout: How to arrange frames - "grid" or "horizontal".
    
    Example:
        >>> env = PokemonRedEnv()
        >>> stacked_env = FrameStackWrapper(env, stack_size=4)
        >>> obs = stacked_env.reset()
        >>> # obs.stacked_image contains a 2x2 grid of the last 4 frames
    """
    
    def __init__(
        self,
        env,
        stack_size: int = 4,
        layout: str = "grid",
    ):
        self.env = env
        self.stack_size = stack_size
        self.layout = layout
        self.frame_buffer: deque = deque(maxlen=stack_size)
    
    def reset(self):
        """
        Reset environment and initialize frame buffer.
        
        Fills the frame buffer with copies of the initial frame.
        
        Returns:
            Initial observation with stacked_image attribute added.
        """
        obs = self.env.reset()
        
        # Get the image from observation
        image = self._get_image(obs)
        
        # Fill buffer with initial frame
        for _ in range(self.stack_size):
            self.frame_buffer.append(image.copy())
        
        # Add stacked image to observation
        self._add_stacked_image(obs)
        return obs
    
    def step(self, action: int):
        """
        Execute single action and update frame buffer.
        
        Args:
            action: Action index to execute.
            
        Returns:
            Observation with stacked_image attribute added.
        """
        obs = self.env.step(action)
        
        # Add new frame to buffer
        image = self._get_image(obs)
        self.frame_buffer.append(image)
        
        # Add stacked image to observation
        self._add_stacked_image(obs)
        return obs
    
    def step_multi(self, actions: List[int]):
        """
        Execute multiple actions and return fused observation.
        
        Useful for multi-action mode where the agent outputs N actions
        and receives the resulting stacked frames.
        
        Args:
            actions: List of action indices to execute.
            
        Returns:
            Final observation with stacked_image containing frames
            from each action execution.
        """
        obs = None
        cumulative_reward = 0.0
        
        for action in actions:
            obs = self.env.step(action)
            image = self._get_image(obs)
            self.frame_buffer.append(image)
            cumulative_reward += getattr(obs, 'reward', 0.0) or 0.0
        
        if obs is not None:
            self._add_stacked_image(obs)
            # Update reward to be cumulative
            if hasattr(obs, 'reward'):
                obs.reward = cumulative_reward
        
        return obs
    
    def _get_image(self, obs) -> Image.Image:
        """Extract PIL Image from observation."""
        if hasattr(obs, 'image'):
            return obs.image
        elif hasattr(obs, 'screen'):
            return obs.screen
        else:
            raise AttributeError("Observation has no 'image' or 'screen' attribute")
    
    def _add_stacked_image(self, obs) -> None:
        """Add stacked_image attribute to observation."""
        stacked = self._create_fused_frame()
        obs.stacked_image = stacked
    
    def _create_fused_frame(self) -> Image.Image:
        """Combine frame buffer into single image."""
        frames = list(self.frame_buffer)
        
        if len(frames) == 0:
            raise ValueError("Frame buffer is empty")
        
        if self.layout == "grid":
            return self._grid_layout(frames)
        elif self.layout == "horizontal":
            return self._horizontal_layout(frames)
        else:
            raise ValueError(f"Unknown layout: {self.layout}")
    
    def _grid_layout(self, frames: List[Image.Image]) -> Image.Image:
        """
        Arrange frames in a grid (e.g., 2x2 for 4 frames).
        
        For 4 frames, creates:
        +---+---+
        | 1 | 2 |
        +---+---+
        | 3 | 4 |
        +---+---+
        
        Args:
            frames: List of PIL Images to arrange.
            
        Returns:
            Combined grid image.
        """
        n = len(frames)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        
        w, h = frames[0].size
        grid = Image.new("RGB", (cols * w, rows * h))
        
        for i, frame in enumerate(frames):
            x = (i % cols) * w
            y = (i // cols) * h
            grid.paste(frame, (x, y))
        
        return grid
    
    def _horizontal_layout(self, frames: List[Image.Image]) -> Image.Image:
        """
        Arrange frames in a horizontal strip.
        
        Oldest frame on left, newest on right for clear temporal ordering.
        
        Args:
            frames: List of PIL Images to arrange.
            
        Returns:
            Horizontally concatenated image.
        """
        n = len(frames)
        w, h = frames[0].size
        strip = Image.new("RGB", (n * w, h))
        
        for i, frame in enumerate(frames):
            strip.paste(frame, (i * w, 0))
        
        return strip
    
    @property
    def current_state(self):
        """Delegate to wrapped environment."""
        return self.env.current_state
    
    def close(self) -> None:
        """Close wrapped environment."""
        if hasattr(self.env, 'close'):
            self.env.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
