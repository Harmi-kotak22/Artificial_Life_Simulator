"""
Unified interface for hazard modules.

This module defines the abstract interface that all hazard-specific
modules must implement to integrate with the core forecasting engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from updff.core.state import State, StateSpaceSpec, Parameters, Observation, Intervention
from updff.core.distribution import Distribution

if TYPE_CHECKING:
    pass


class HazardModule(ABC):
    """
    Abstract base class for all hazard modules.
    
    A hazard module encapsulates the domain-specific dynamics of a
    particular disaster type (disease, flood, earthquake, etc.) while
    conforming to a unified interface for the core forecasting engine.
    
    The interface specifies:
    1. State space definition
    2. Stochastic transition dynamics
    3. Observation model
    4. Intervention effects
    5. Prior parameter distributions
    """
    
    @abstractmethod
    def get_state_spec(self) -> StateSpaceSpec:
        """
        Return the state space specification.
        
        Defines dimensionality and semantics of state variables,
        parameters, observations, and interventions.
        
        Returns:
            StateSpaceSpec with full specification
        """
        pass
    
    @abstractmethod
    def initialize_state(
        self,
        initial_conditions: Dict[str, Any],
        uncertainty: Dict[str, float]
    ) -> State:
        """
        Create initial state from conditions.
        
        Args:
            initial_conditions: Dictionary of initial values
            uncertainty: Optional uncertainty (std) for each state
            
        Returns:
            Initialized State object
        """
        pass
    
    @abstractmethod
    def transition(
        self,
        state: State,
        params: Parameters,
        interventions: List[Intervention],
        dt: float,
        n_samples: int = 1
    ) -> List[State]:
        """
        Stochastic state transition.
        
        This is the core dynamics function that implements:
        P(X_{t+dt} | X_t, Θ, U)
        
        Args:
            state: Current state
            params: Model parameters
            interventions: Active interventions
            dt: Time step size
            n_samples: Number of next-state samples to generate
            
        Returns:
            List of n_samples next states sampled from transition distribution
        """
        pass
    
    @abstractmethod
    def observe(
        self,
        state: State,
        observation_noise: np.ndarray
    ) -> Observation:
        """
        Generate observation from state.
        
        Implements the observation model: Y = h(X) + η
        
        Args:
            state: Current (latent) state
            observation_noise: Observation noise sample
            
        Returns:
            Noisy observation
        """
        pass
    
    @abstractmethod
    def log_likelihood(
        self,
        observation: Observation,
        state: State,
        params: Parameters
    ) -> float:
        """
        Compute log P(Y | X, Θ).
        
        Used for inference, model comparison, and filtering.
        
        Args:
            observation: Observed data
            state: Hypothesized state
            params: Model parameters
            
        Returns:
            Log probability of observation given state
        """
        pass
    
    @abstractmethod
    def get_prior(self) -> Dict[str, Distribution]:
        """
        Return prior distributions over parameters.
        
        Encodes domain knowledge and physical constraints.
        
        Returns:
            Dictionary mapping parameter names to prior distributions
        """
        pass
    
    @abstractmethod
    def apply_intervention(
        self,
        intervention: Intervention,
        params: Parameters,
        state: State
    ) -> Parameters:
        """
        Modify parameters to reflect intervention effect.
        
        Args:
            intervention: The intervention to apply
            params: Current parameters
            state: Current state (may affect intervention efficacy)
            
        Returns:
            Modified parameters
        """
        pass
    
    def validate_state(self, state: State) -> bool:
        """
        Check if state satisfies physical constraints.
        
        Default implementation checks bounds from state spec.
        Override for additional domain-specific constraints.
        
        Args:
            state: State to validate
            
        Returns:
            True if valid, False otherwise
        """
        spec = self.get_state_spec()
        return spec.validate_state(state.values)
    
    def validate_params(self, params: Parameters) -> bool:
        """
        Check if parameters satisfy constraints.
        
        Args:
            params: Parameters to validate
            
        Returns:
            True if valid, False otherwise
        """
        spec = self.get_state_spec()
        return spec.validate_params(params.values)
    
    def get_observable_names(self) -> List[str]:
        """Get names of observable quantities."""
        return self.get_state_spec().observation_names
    
    def get_intervention_names(self) -> List[str]:
        """Get names of available intervention types."""
        return self.get_state_spec().intervention_names
    
    def compute_reproduction_number(
        self,
        state: State,
        params: Parameters
    ) -> float:
        """
        Compute effective reproduction number (disease-specific).
        
        Override in disease modules.
        
        Returns:
            R_t value
        """
        raise NotImplementedError("R_t computation not implemented for this hazard")
    
    def compute_risk_score(
        self,
        state: State,
        params: Parameters
    ) -> float:
        """
        Compute overall risk score for current state.
        
        Provides a single summary metric for decision support.
        
        Returns:
            Risk score in [0, 1]
        """
        raise NotImplementedError("Risk score not implemented for this hazard")
    
    def describe(self) -> str:
        """Return human-readable description of the hazard module."""
        spec = self.get_state_spec()
        return (
            f"{self.__class__.__name__}\n"
            f"  State variables ({spec.state_dim}): {spec.state_names}\n"
            f"  Parameters ({spec.param_dim}): {spec.param_names}\n"
            f"  Observations ({spec.observation_dim}): {spec.observation_names}\n"
            f"  Interventions ({spec.intervention_dim}): {spec.intervention_names}"
        )


class CompositeHazardModule(HazardModule):
    """
    Composite hazard module for cascading/coupled disasters.
    
    Combines multiple hazard modules where one hazard can trigger
    or amplify another (e.g., flood -> disease outbreak).
    """
    
    def __init__(
        self,
        primary_hazard: HazardModule,
        secondary_hazard: HazardModule,
        coupling_function: callable
    ):
        """
        Initialize composite hazard.
        
        Args:
            primary_hazard: Primary hazard module
            secondary_hazard: Secondary (triggered) hazard
            coupling_function: Function mapping primary state to secondary driver
        """
        self.primary = primary_hazard
        self.secondary = secondary_hazard
        self.coupling = coupling_function
    
    def get_state_spec(self) -> StateSpaceSpec:
        """Combined state space."""
        primary_spec = self.primary.get_state_spec()
        secondary_spec = self.secondary.get_state_spec()
        
        return StateSpaceSpec(
            state_dim=primary_spec.state_dim + secondary_spec.state_dim,
            state_names=primary_spec.state_names + secondary_spec.state_names,
            state_bounds=primary_spec.state_bounds + secondary_spec.state_bounds,
            param_dim=primary_spec.param_dim + secondary_spec.param_dim,
            param_names=primary_spec.param_names + secondary_spec.param_names,
            param_bounds=primary_spec.param_bounds + secondary_spec.param_bounds,
            observation_dim=primary_spec.observation_dim + secondary_spec.observation_dim,
            observation_names=primary_spec.observation_names + secondary_spec.observation_names,
            intervention_dim=primary_spec.intervention_dim + secondary_spec.intervention_dim,
            intervention_names=primary_spec.intervention_names + secondary_spec.intervention_names,
        )
    
    def initialize_state(
        self,
        initial_conditions: Dict[str, Any],
        uncertainty: Dict[str, float]
    ) -> State:
        """Initialize both hazard states."""
        # Split conditions by prefix or just pass through
        primary_state = self.primary.initialize_state(initial_conditions, uncertainty)
        secondary_state = self.secondary.initialize_state(initial_conditions, uncertainty)
        
        combined_values = np.concatenate([primary_state.values, secondary_state.values])
        combined_cov = None
        if primary_state.covariance is not None and secondary_state.covariance is not None:
            combined_cov = np.block([
                [primary_state.covariance, np.zeros((primary_state.dim, secondary_state.dim))],
                [np.zeros((secondary_state.dim, primary_state.dim)), secondary_state.covariance]
            ])
        
        return State(
            values=combined_values,
            timestamp=primary_state.timestamp,
            covariance=combined_cov
        )
    
    def transition(
        self,
        state: State,
        params: Parameters,
        interventions: List[Intervention],
        dt: float,
        n_samples: int = 1
    ) -> List[State]:
        """Coupled transition dynamics."""
        # Split state
        primary_dim = self.primary.get_state_spec().state_dim
        primary_state = State(
            values=state.values[:primary_dim],
            timestamp=state.timestamp
        )
        secondary_state = State(
            values=state.values[primary_dim:],
            timestamp=state.timestamp
        )
        
        # Split params
        primary_param_dim = self.primary.get_state_spec().param_dim
        primary_params = Parameters(values=params.values[:primary_param_dim])
        secondary_params = Parameters(values=params.values[primary_param_dim:])
        
        results = []
        for _ in range(n_samples):
            # Evolve primary
            next_primary = self.primary.transition(
                primary_state, primary_params, interventions, dt, 1
            )[0]
            
            # Compute coupling effect
            coupling_effect = self.coupling(next_primary, primary_params)
            
            # Evolve secondary with coupling
            # (Coupling modifies secondary parameters or state)
            modified_secondary_params = self._apply_coupling(
                secondary_params, coupling_effect
            )
            
            next_secondary = self.secondary.transition(
                secondary_state, modified_secondary_params, interventions, dt, 1
            )[0]
            
            # Combine
            combined_values = np.concatenate([next_primary.values, next_secondary.values])
            results.append(State(
                values=combined_values,
                timestamp=next_primary.timestamp
            ))
        
        return results
    
    def _apply_coupling(
        self,
        params: Parameters,
        coupling_effect: Dict[str, float]
    ) -> Parameters:
        """Apply coupling effect to secondary parameters."""
        modified = params.copy()
        for key, value in coupling_effect.items():
            if params.names and key in params.names:
                idx = params.names.index(key)
                modified.values[idx] *= value
        return modified
    
    def observe(
        self,
        state: State,
        observation_noise: np.ndarray
    ) -> Observation:
        """Combined observation."""
        primary_dim = self.primary.get_state_spec().state_dim
        primary_obs_dim = self.primary.get_state_spec().observation_dim
        
        primary_state = State(values=state.values[:primary_dim], timestamp=state.timestamp)
        secondary_state = State(values=state.values[primary_dim:], timestamp=state.timestamp)
        
        primary_obs = self.primary.observe(primary_state, observation_noise[:primary_obs_dim])
        secondary_obs = self.secondary.observe(secondary_state, observation_noise[primary_obs_dim:])
        
        return Observation(
            values=np.concatenate([primary_obs.values, secondary_obs.values]),
            timestamp=state.timestamp
        )
    
    def log_likelihood(
        self,
        observation: Observation,
        state: State,
        params: Parameters
    ) -> float:
        """Combined log-likelihood."""
        primary_dim = self.primary.get_state_spec().state_dim
        primary_obs_dim = self.primary.get_state_spec().observation_dim
        primary_param_dim = self.primary.get_state_spec().param_dim
        
        primary_state = State(values=state.values[:primary_dim], timestamp=state.timestamp)
        secondary_state = State(values=state.values[primary_dim:], timestamp=state.timestamp)
        
        primary_obs = Observation(values=observation.values[:primary_obs_dim], timestamp=observation.timestamp)
        secondary_obs = Observation(values=observation.values[primary_obs_dim:], timestamp=observation.timestamp)
        
        primary_params = Parameters(values=params.values[:primary_param_dim])
        secondary_params = Parameters(values=params.values[primary_param_dim:])
        
        return (
            self.primary.log_likelihood(primary_obs, primary_state, primary_params) +
            self.secondary.log_likelihood(secondary_obs, secondary_state, secondary_params)
        )
    
    def get_prior(self) -> Dict[str, Distribution]:
        """Combined priors."""
        priors = {}
        priors.update(self.primary.get_prior())
        priors.update(self.secondary.get_prior())
        return priors
    
    def apply_intervention(
        self,
        intervention: Intervention,
        params: Parameters,
        state: State
    ) -> Parameters:
        """Apply intervention to appropriate hazard."""
        # Determine which hazard the intervention applies to
        primary_interventions = set(self.primary.get_intervention_names())
        
        primary_param_dim = self.primary.get_state_spec().param_dim
        primary_dim = self.primary.get_state_spec().state_dim
        
        primary_params = Parameters(values=params.values[:primary_param_dim])
        secondary_params = Parameters(values=params.values[primary_param_dim:])
        
        primary_state = State(values=state.values[:primary_dim], timestamp=state.timestamp)
        secondary_state = State(values=state.values[primary_dim:], timestamp=state.timestamp)
        
        if intervention.intervention_type in primary_interventions:
            primary_params = self.primary.apply_intervention(
                intervention, primary_params, primary_state
            )
        else:
            secondary_params = self.secondary.apply_intervention(
                intervention, secondary_params, secondary_state
            )
        
        return Parameters(
            values=np.concatenate([primary_params.values, secondary_params.values])
        )
