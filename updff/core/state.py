"""
State space management for the forecasting framework.

This module defines the core data structures for representing system state,
parameters, and observations in a hazard-agnostic manner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator


@dataclass
class StateSpaceSpec:
    """
    Specification of a hazard's state space.
    
    Defines the dimensionality, semantics, and constraints of the state,
    parameter, observation, and intervention spaces for a hazard module.
    
    Attributes:
        state_dim: Dimensionality of the state vector
        state_names: Human-readable names for each state variable
        state_bounds: (min, max) bounds for each state variable
        param_dim: Dimensionality of the parameter vector
        param_names: Human-readable names for each parameter
        param_bounds: (min, max) bounds for each parameter (for constraints)
        observation_dim: Dimensionality of the observation vector
        observation_names: Human-readable names for each observable
        intervention_dim: Dimensionality of the intervention/control vector
        intervention_names: Human-readable names for each intervention type
        state_dtype: Data type for state arrays
    """
    
    state_dim: int
    state_names: List[str]
    state_bounds: List[Tuple[float, float]]
    param_dim: int
    param_names: List[str]
    param_bounds: List[Tuple[float, float]]
    observation_dim: int
    observation_names: List[str]
    intervention_dim: int = 0
    intervention_names: List[str] = field(default_factory=list)
    state_dtype: np.dtype = field(default=np.float64)
    
    def __post_init__(self):
        """Validate specification consistency."""
        assert len(self.state_names) == self.state_dim, \
            f"state_names length ({len(self.state_names)}) != state_dim ({self.state_dim})"
        assert len(self.state_bounds) == self.state_dim, \
            f"state_bounds length ({len(self.state_bounds)}) != state_dim ({self.state_dim})"
        assert len(self.param_names) == self.param_dim, \
            f"param_names length ({len(self.param_names)}) != param_dim ({self.param_dim})"
        assert len(self.param_bounds) == self.param_dim, \
            f"param_bounds length ({len(self.param_bounds)}) != param_dim ({self.param_dim})"
        assert len(self.observation_names) == self.observation_dim, \
            f"observation_names length ({len(self.observation_names)}) != observation_dim ({self.observation_dim})"
    
    def get_state_index(self, name: str) -> int:
        """Get index of a state variable by name."""
        return self.state_names.index(name)
    
    def get_param_index(self, name: str) -> int:
        """Get index of a parameter by name."""
        return self.param_names.index(name)
    
    def validate_state(self, values: np.ndarray) -> bool:
        """Check if state values satisfy bounds."""
        if values.shape[-1] != self.state_dim:
            return False
        for i, (lo, hi) in enumerate(self.state_bounds):
            if np.any(values[..., i] < lo) or np.any(values[..., i] > hi):
                return False
        return True
    
    def validate_params(self, values: np.ndarray) -> bool:
        """Check if parameter values satisfy bounds."""
        if values.shape[-1] != self.param_dim:
            return False
        for i, (lo, hi) in enumerate(self.param_bounds):
            if np.any(values[..., i] < lo) or np.any(values[..., i] > hi):
                return False
        return True


@dataclass
class State:
    """
    Current state of the dynamical system.
    
    Represents both the state estimate and its associated uncertainty.
    
    Attributes:
        values: State vector of shape (state_dim,) or (n_samples, state_dim)
        timestamp: Time coordinate (can be datetime or float)
        covariance: Uncertainty covariance matrix (state_dim, state_dim)
        metadata: Additional state information
    """
    
    values: np.ndarray
    timestamp: Union[datetime, float]
    covariance: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure values is a numpy array."""
        self.values = np.atleast_1d(np.asarray(self.values))
        if self.covariance is not None:
            self.covariance = np.asarray(self.covariance)
    
    @property
    def dim(self) -> int:
        """Dimensionality of the state."""
        return self.values.shape[-1]
    
    @property
    def is_ensemble(self) -> bool:
        """Whether this represents an ensemble of states."""
        return self.values.ndim == 2
    
    @property
    def n_samples(self) -> int:
        """Number of samples if ensemble, else 1."""
        return self.values.shape[0] if self.is_ensemble else 1
    
    @property
    def mean(self) -> np.ndarray:
        """Mean state value."""
        if self.is_ensemble:
            return np.mean(self.values, axis=0)
        return self.values
    
    @property
    def std(self) -> np.ndarray:
        """Standard deviation of state."""
        if self.is_ensemble:
            return np.std(self.values, axis=0)
        elif self.covariance is not None:
            return np.sqrt(np.diag(self.covariance))
        return np.zeros(self.dim)
    
    def percentile(self, q: float) -> np.ndarray:
        """Compute q-th percentile of state ensemble."""
        if self.is_ensemble:
            return np.percentile(self.values, q, axis=0)
        return self.values
    
    def credible_interval(self, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute (1-alpha) credible interval.
        
        Returns (lower, upper) bounds.
        """
        lower = self.percentile(100 * alpha / 2)
        upper = self.percentile(100 * (1 - alpha / 2))
        return lower, upper
    
    def copy(self) -> State:
        """Create a deep copy of this state."""
        return State(
            values=self.values.copy(),
            timestamp=self.timestamp,
            covariance=self.covariance.copy() if self.covariance is not None else None,
            metadata=self.metadata.copy()
        )
    
    @classmethod
    def from_ensemble(
        cls, 
        samples: np.ndarray, 
        timestamp: Union[datetime, float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> State:
        """Create state from ensemble samples."""
        return cls(
            values=samples,
            timestamp=timestamp,
            covariance=np.cov(samples, rowvar=False) if samples.shape[0] > 1 else None,
            metadata=metadata or {}
        )


@dataclass
class Parameters:
    """
    Model parameters with uncertainty quantification.
    
    Attributes:
        values: Parameter vector of shape (param_dim,) or (n_samples, param_dim)
        names: Parameter names
        covariance: Uncertainty covariance matrix
        bounds: (min, max) bounds for each parameter
        metadata: Additional parameter information
    """
    
    values: np.ndarray
    names: Optional[List[str]] = None
    covariance: Optional[np.ndarray] = None
    bounds: Optional[List[Tuple[float, float]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure values is numpy array."""
        self.values = np.atleast_1d(np.asarray(self.values))
        if self.covariance is not None:
            self.covariance = np.asarray(self.covariance)
    
    @property
    def dim(self) -> int:
        """Dimensionality of parameters."""
        return self.values.shape[-1]
    
    @property
    def is_ensemble(self) -> bool:
        """Whether this represents ensemble of parameter samples."""
        return self.values.ndim == 2
    
    @property
    def n_samples(self) -> int:
        """Number of samples if ensemble, else 1."""
        return self.values.shape[0] if self.is_ensemble else 1
    
    @property
    def mean(self) -> np.ndarray:
        """Mean parameter values."""
        if self.is_ensemble:
            return np.mean(self.values, axis=0)
        return self.values
    
    @property
    def std(self) -> np.ndarray:
        """Standard deviation of parameters."""
        if self.is_ensemble:
            return np.std(self.values, axis=0)
        elif self.covariance is not None:
            return np.sqrt(np.diag(self.covariance))
        return np.zeros(self.dim)
    
    def get(self, name: str) -> Union[float, np.ndarray]:
        """Get parameter value by name."""
        if self.names is None:
            raise ValueError("Parameter names not defined")
        idx = self.names.index(name)
        if self.is_ensemble:
            return self.values[:, idx]
        return self.values[idx]
    
    def set(self, name: str, value: Union[float, np.ndarray]) -> None:
        """Set parameter value by name."""
        if self.names is None:
            raise ValueError("Parameter names not defined")
        idx = self.names.index(name)
        if self.is_ensemble:
            self.values[:, idx] = value
        else:
            self.values[idx] = value
    
    def sample(self, n_samples: int, rng: Optional[np.random.Generator] = None) -> Parameters:
        """
        Sample parameters from posterior distribution.
        
        If already ensemble, resamples. If point estimate with covariance,
        samples from multivariate normal.
        """
        rng = rng or np.random.default_rng()
        
        if self.is_ensemble:
            # Resample from existing ensemble
            indices = rng.choice(self.n_samples, size=n_samples, replace=True)
            samples = self.values[indices]
        elif self.covariance is not None:
            # Sample from multivariate normal
            samples = rng.multivariate_normal(self.values, self.covariance, size=n_samples)
        else:
            # No uncertainty - just replicate
            samples = np.tile(self.values, (n_samples, 1))
        
        # Apply bounds if specified
        if self.bounds is not None:
            for i, (lo, hi) in enumerate(self.bounds):
                samples[:, i] = np.clip(samples[:, i], lo, hi)
        
        return Parameters(
            values=samples,
            names=self.names,
            covariance=np.cov(samples, rowvar=False) if n_samples > 1 else None,
            bounds=self.bounds,
            metadata=self.metadata
        )
    
    def copy(self) -> Parameters:
        """Create a deep copy."""
        return Parameters(
            values=self.values.copy(),
            names=self.names.copy() if self.names else None,
            covariance=self.covariance.copy() if self.covariance is not None else None,
            bounds=self.bounds.copy() if self.bounds else None,
            metadata=self.metadata.copy()
        )
    
    def to_dict(self) -> Dict[str, Union[float, np.ndarray]]:
        """Convert to dictionary."""
        if self.names is None:
            return {f"param_{i}": v for i, v in enumerate(self.mean)}
        return {name: self.get(name) for name in self.names}


@dataclass
class Observation:
    """
    Observed data point with measurement uncertainty.
    
    Attributes:
        values: Observation vector
        timestamp: Time of observation
        noise_covariance: Measurement noise covariance
        observation_type: Type identifier for the observation
        metadata: Additional observation information
    """
    
    values: np.ndarray
    timestamp: Union[datetime, float]
    noise_covariance: Optional[np.ndarray] = None
    observation_type: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure values is numpy array."""
        self.values = np.atleast_1d(np.asarray(self.values))
        if self.noise_covariance is not None:
            self.noise_covariance = np.asarray(self.noise_covariance)
    
    @property
    def dim(self) -> int:
        """Dimensionality of observation."""
        return len(self.values)
    
    @property
    def noise_std(self) -> np.ndarray:
        """Standard deviation of observation noise."""
        if self.noise_covariance is not None:
            return np.sqrt(np.diag(self.noise_covariance))
        return np.zeros(self.dim)


@dataclass
class Intervention:
    """
    Control action or policy intervention.
    
    Attributes:
        intervention_type: Type/name of intervention (e.g., "vaccination", "lockdown")
        magnitude: Strength/intensity of intervention [0, 1] typically
        start_time: When intervention begins
        end_time: When intervention ends (None for indefinite)
        spatial_scope: List of spatial units affected (None for all)
        target_groups: Population groups affected (None for all)
        parameters: Additional intervention-specific parameters
        metadata: Additional information
    """
    
    intervention_type: str
    magnitude: float
    start_time: Union[datetime, float]
    end_time: Optional[Union[datetime, float]] = None
    spatial_scope: Optional[List[int]] = None
    target_groups: Optional[List[str]] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_active(self, time: Union[datetime, float]) -> bool:
        """Check if intervention is active at given time."""
        if self.end_time is None:
            return time >= self.start_time
        return self.start_time <= time <= self.end_time
    
    def get_magnitude_at(self, time: Union[datetime, float]) -> float:
        """Get intervention magnitude at given time (allows for ramp-up)."""
        if not self.is_active(time):
            return 0.0
        
        # Check for ramp-up period
        ramp_days = self.parameters.get("ramp_up_days", 0)
        if ramp_days > 0 and isinstance(time, (int, float)):
            elapsed = time - self.start_time
            if elapsed < ramp_days:
                return self.magnitude * (elapsed / ramp_days)
        
        return self.magnitude


class ObservationSeries:
    """
    Time series of observations.
    
    Manages a collection of observations over time for data assimilation
    and model calibration.
    """
    
    def __init__(self, observations: Optional[List[Observation]] = None):
        """Initialize with optional list of observations."""
        self._observations: List[Observation] = []
        if observations:
            for obs in observations:
                self.add(obs)
    
    def add(self, observation: Observation) -> None:
        """Add observation, maintaining time ordering."""
        self._observations.append(observation)
        self._observations.sort(key=lambda x: x.timestamp)
    
    def get_range(
        self,
        start: Union[datetime, float],
        end: Union[datetime, float]
    ) -> List[Observation]:
        """Get observations in time range [start, end]."""
        return [
            obs for obs in self._observations
            if start <= obs.timestamp <= end
        ]
    
    def get_at(self, timestamp: Union[datetime, float], tolerance: float = 0.0) -> Optional[Observation]:
        """Get observation at or near timestamp."""
        for obs in self._observations:
            if isinstance(timestamp, datetime) and isinstance(obs.timestamp, datetime):
                diff = abs((obs.timestamp - timestamp).total_seconds())
                if diff <= tolerance:
                    return obs
            elif abs(obs.timestamp - timestamp) <= tolerance:
                return obs
        return None
    
    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert to arrays of (timestamps, values)."""
        timestamps = np.array([obs.timestamp for obs in self._observations])
        values = np.stack([obs.values for obs in self._observations])
        return timestamps, values
    
    @property
    def timestamps(self) -> List[Union[datetime, float]]:
        """List of observation timestamps."""
        return [obs.timestamp for obs in self._observations]
    
    @property
    def values(self) -> np.ndarray:
        """Array of observation values."""
        if not self._observations:
            return np.array([])
        return np.stack([obs.values for obs in self._observations])
    
    def __len__(self) -> int:
        return len(self._observations)
    
    def __iter__(self):
        return iter(self._observations)
    
    def __getitem__(self, idx: int) -> Observation:
        return self._observations[idx]
