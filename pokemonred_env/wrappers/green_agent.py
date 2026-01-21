# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Green Agent compute efficiency tracker for Pokemon Red OpenEnv.

This module implements compute efficiency tracking as required by
The OpenEnv Challenge evaluation criteria. It measures wall-clock time,
memory usage, and provides transparency metrics for sustainability analysis.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, TypeVar

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

T = TypeVar("T")


@dataclass
class GreenMetrics:
    """
    Compute efficiency metrics for a single step or episode.
    
    Attributes:
        step_time_ms: Wall-clock time for last step in milliseconds.
        avg_step_time_ms: Running average step time in milliseconds.
        total_steps: Total steps executed.
        total_time_s: Total wall-clock time in seconds.
        peak_memory_mb: Peak memory usage in megabytes.
        current_memory_mb: Current memory usage in megabytes.
        cpu_percent: CPU utilization percentage (if available).
    """
    step_time_ms: float = 0.0
    avg_step_time_ms: float = 0.0
    total_steps: int = 0
    total_time_s: float = 0.0
    peak_memory_mb: float = 0.0
    current_memory_mb: float = 0.0
    cpu_percent: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for JSON serialization."""
        return {
            "green_step_time_ms": round(self.step_time_ms, 3),
            "green_avg_step_time_ms": round(self.avg_step_time_ms, 3),
            "green_total_steps": self.total_steps,
            "green_total_time_s": round(self.total_time_s, 3),
            "green_peak_memory_mb": round(self.peak_memory_mb, 2),
            "green_current_memory_mb": round(self.current_memory_mb, 2),
            "green_cpu_percent": round(self.cpu_percent, 1),
        }


class GreenAgentTracker:
    """
    Tracks compute efficiency metrics for RL environment steps.
    
    Designed for the OpenEnv Challenge "Green Agent" evaluation criteria,
    measuring the computational cost of environment interactions to
    promote sustainable AI development.
    
    Example:
        >>> tracker = GreenAgentTracker()
        >>> 
        >>> # Track a step
        >>> result, metrics = tracker.track_step(lambda: env.step(action))
        >>> print(f"Step took {metrics['green_step_time_ms']:.2f}ms")
        >>> 
        >>> # Get summary
        >>> summary = tracker.get_summary()
        >>> print(f"Avg step: {summary['green_avg_step_time_ms']:.2f}ms")
    
    Attributes:
        enabled: Whether tracking is active.
        metrics: Current aggregated metrics.
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize Green Agent tracker.
        
        Args:
            enabled: Whether to enable tracking (default True).
                    Disable for benchmarking without overhead.
        """
        self.enabled = enabled
        self._process: Optional[Any] = None
        if HAS_PSUTIL:
            self._process = psutil.Process(os.getpid())
        
        self.reset()
    
    def reset(self) -> None:
        """Reset tracking metrics for new episode."""
        self._total_steps = 0
        self._total_time = 0.0
        self._peak_memory = 0.0
        self._last_step_time = 0.0
    
    def track_step(self, step_fn: Callable[[], T]) -> tuple[T, Dict[str, Any]]:
        """
        Execute and track a step function.
        
        Args:
            step_fn: Callable that executes the environment step.
        
        Returns:
            Tuple of (step result, metrics dictionary).
        """
        if not self.enabled:
            return step_fn(), {}
        
        # Measure memory before
        start_memory = self._get_memory_mb()
        
        # Time the step
        start_time = time.perf_counter()
        result = step_fn()
        elapsed = time.perf_counter() - start_time
        
        # Update tracking
        self._last_step_time = elapsed
        self._total_time += elapsed
        self._total_steps += 1
        
        # Update peak memory
        current_memory = self._get_memory_mb()
        self._peak_memory = max(self._peak_memory, current_memory)
        
        # Build metrics
        metrics = self.get_metrics()
        return result, metrics.to_dict()
    
    def track_reset(self, reset_fn: Callable[[], T]) -> tuple[T, Dict[str, Any]]:
        """
        Execute and track a reset function.
        
        Args:
            reset_fn: Callable that executes the environment reset.
        
        Returns:
            Tuple of (reset result, metrics dictionary).
        """
        self.reset()
        return self.track_step(reset_fn)
    
    def get_metrics(self) -> GreenMetrics:
        """Get current aggregated metrics."""
        avg_step = self._total_time / max(self._total_steps, 1)
        
        return GreenMetrics(
            step_time_ms=self._last_step_time * 1000,
            avg_step_time_ms=avg_step * 1000,
            total_steps=self._total_steps,
            total_time_s=self._total_time,
            peak_memory_mb=self._peak_memory,
            current_memory_mb=self._get_memory_mb(),
            cpu_percent=self._get_cpu_percent(),
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary as dictionary."""
        return self.get_metrics().to_dict()
    
    def _get_memory_mb(self) -> float:
        """Get current process memory in MB."""
        if self._process is not None:
            try:
                return self._process.memory_info().rss / (1024 * 1024)
            except Exception:
                pass
        return 0.0
    
    def _get_cpu_percent(self) -> float:
        """Get current CPU utilization percentage."""
        if self._process is not None:
            try:
                return self._process.cpu_percent()
            except Exception:
                pass
        return 0.0
