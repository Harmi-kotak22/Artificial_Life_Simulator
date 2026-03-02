"""
Uncertainty propagation methods for the forecasting framework.

This module implements various techniques for propagating uncertainty
through nonlinear dynamical systems, including Monte Carlo methods,
unscented transforms, and ensemble-based approaches.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import linalg

from updff.core.state import State, Parameters


class PropagationMethod(Enum):
    """Available uncertainty propagation methods."""
    MONTE_CARLO = "monte_carlo"
    UNSCENTED = "unscented"
    ENSEMBLE_KALMAN = "ensemble_kalman"
    LINEARIZED = "linearized"


@dataclass
class PropagationResult:
    """
    Result of uncertainty propagation.
    
    Attributes:
        mean: Propagated mean state
        covariance: Propagated state covariance
        samples: Optional ensemble samples
        weights: Optional sample weights (for importance sampling)
    """
    
    mean: np.ndarray
    covariance: np.ndarray
    samples: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None
    
    @property
    def std(self) -> np.ndarray:
        """Standard deviation from covariance diagonal."""
        return np.sqrt(np.diag(self.covariance))
    
    def credible_interval(self, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """Compute credible interval from samples or Gaussian approximation."""
        if self.samples is not None:
            if self.weights is not None:
                # Weighted quantiles
                sorted_idx = np.argsort(self.samples, axis=0)
                sorted_samples = np.take_along_axis(self.samples, sorted_idx, axis=0)
                cum_weights = np.cumsum(self.weights[sorted_idx], axis=0)
                lower_idx = np.searchsorted(cum_weights[:, 0], alpha / 2)
                upper_idx = np.searchsorted(cum_weights[:, 0], 1 - alpha / 2)
                return sorted_samples[lower_idx], sorted_samples[upper_idx]
            else:
                lower = np.percentile(self.samples, 100 * alpha / 2, axis=0)
                upper = np.percentile(self.samples, 100 * (1 - alpha / 2), axis=0)
                return lower, upper
        else:
            # Gaussian approximation
            from scipy import stats
            z = stats.norm.ppf(1 - alpha / 2)
            return self.mean - z * self.std, self.mean + z * self.std


class UncertaintyPropagator(ABC):
    """
    Abstract base class for uncertainty propagation.
    
    Propagates probability distributions through nonlinear
    state transition functions.
    """
    
    @abstractmethod
    def propagate(
        self,
        state: State,
        params: Parameters,
        transition_fn: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
        dt: float,
        process_noise: Optional[np.ndarray] = None
    ) -> PropagationResult:
        """
        Propagate state uncertainty through transition function.
        
        Args:
            state: Current state with uncertainty
            params: Model parameters with uncertainty
            transition_fn: State transition function f(state, params, dt) -> next_state
            dt: Time step
            process_noise: Process noise covariance matrix
            
        Returns:
            PropagationResult with propagated statistics
        """
        pass


class MonteCarloUncertainty(UncertaintyPropagator):
    """
    Monte Carlo uncertainty propagation.
    
    Propagates uncertainty by sampling from the prior distribution,
    applying the transition function to each sample, and computing
    statistics from the transformed samples.
    
    This is the most general method, applicable to arbitrary nonlinear
    functions and non-Gaussian distributions.
    """
    
    def __init__(
        self,
        n_samples: int = 1000,
        seed: Optional[int] = None
    ):
        """
        Initialize Monte Carlo propagator.
        
        Args:
            n_samples: Number of Monte Carlo samples
            seed: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)
    
    def propagate(
        self,
        state: State,
        params: Parameters,
        transition_fn: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
        dt: float,
        process_noise: Optional[np.ndarray] = None
    ) -> PropagationResult:
        """
        Monte Carlo uncertainty propagation.
        
        Samples from joint state-parameter distribution and propagates
        each sample through the transition function.
        """
        # Sample states
        if state.is_ensemble:
            # Resample from existing ensemble
            state_samples = state.values[
                self.rng.choice(state.n_samples, self.n_samples, replace=True)
            ]
        elif state.covariance is not None:
            # Sample from Gaussian
            state_samples = self.rng.multivariate_normal(
                state.values, state.covariance, size=self.n_samples
            )
        else:
            # No uncertainty - replicate
            state_samples = np.tile(state.values, (self.n_samples, 1))
        
        # Sample parameters
        param_samples = params.sample(self.n_samples, self.rng).values
        
        # Propagate each sample
        next_samples = np.zeros_like(state_samples)
        for i in range(self.n_samples):
            next_samples[i] = transition_fn(state_samples[i], param_samples[i], dt)
            
            # Add process noise if specified
            if process_noise is not None:
                next_samples[i] += self.rng.multivariate_normal(
                    np.zeros(state.dim), process_noise
                )
        
        # Compute statistics
        mean = np.mean(next_samples, axis=0)
        covariance = np.cov(next_samples, rowvar=False)
        
        # Ensure covariance is 2D
        if covariance.ndim == 0:
            covariance = np.array([[covariance]])
        
        return PropagationResult(
            mean=mean,
            covariance=covariance,
            samples=next_samples,
            weights=None
        )


class UnscentedTransform(UncertaintyPropagator):
    """
    Unscented Transform for uncertainty propagation.
    
    Uses a deterministic set of sigma points to capture the mean
    and covariance of the prior, propagates these points through
    the nonlinear function, and recovers posterior statistics.
    
    More efficient than Monte Carlo for Gaussian-like distributions
    and smooth nonlinearities.
    """
    
    def __init__(
        self,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0
    ):
        """
        Initialize Unscented Transform.
        
        Args:
            alpha: Spread of sigma points (small positive value)
            beta: Prior knowledge about distribution (2 for Gaussian)
            kappa: Secondary scaling parameter (usually 0)
        """
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
    
    def _compute_sigma_points(
        self,
        mean: np.ndarray,
        covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute sigma points and weights.
        
        Returns:
            sigma_points: (2n+1, n) array of sigma points
            weights_mean: Weights for mean computation
            weights_cov: Weights for covariance computation
        """
        n = len(mean)
        lambda_ = self.alpha**2 * (n + self.kappa) - n
        
        # Sigma points
        sigma_points = np.zeros((2 * n + 1, n))
        sigma_points[0] = mean
        
        # Square root of covariance
        try:
            sqrt_cov = linalg.cholesky((n + lambda_) * covariance, lower=True)
        except linalg.LinAlgError:
            # Add small regularization if not positive definite
            sqrt_cov = linalg.cholesky(
                (n + lambda_) * (covariance + 1e-6 * np.eye(n)), lower=True
            )
        
        for i in range(n):
            sigma_points[i + 1] = mean + sqrt_cov[:, i]
            sigma_points[n + i + 1] = mean - sqrt_cov[:, i]
        
        # Weights
        weights_mean = np.zeros(2 * n + 1)
        weights_cov = np.zeros(2 * n + 1)
        
        weights_mean[0] = lambda_ / (n + lambda_)
        weights_cov[0] = lambda_ / (n + lambda_) + (1 - self.alpha**2 + self.beta)
        
        for i in range(1, 2 * n + 1):
            weights_mean[i] = 1 / (2 * (n + lambda_))
            weights_cov[i] = 1 / (2 * (n + lambda_))
        
        return sigma_points, weights_mean, weights_cov
    
    def propagate(
        self,
        state: State,
        params: Parameters,
        transition_fn: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
        dt: float,
        process_noise: Optional[np.ndarray] = None
    ) -> PropagationResult:
        """
        Unscented Transform propagation.
        """
        # Use mean state and covariance
        mean = state.mean
        cov = state.covariance if state.covariance is not None else np.eye(state.dim) * 1e-6
        
        # Compute sigma points
        sigma_points, w_m, w_c = self._compute_sigma_points(mean, cov)
        
        # Propagate sigma points
        n_sigma = len(sigma_points)
        propagated = np.zeros_like(sigma_points)
        
        for i in range(n_sigma):
            propagated[i] = transition_fn(sigma_points[i], params.mean, dt)
        
        # Recover statistics
        prop_mean = np.sum(w_m[:, None] * propagated, axis=0)
        
        diff = propagated - prop_mean
        prop_cov = np.sum(
            w_c[:, None, None] * (diff[:, :, None] @ diff[:, None, :]),
            axis=0
        )
        
        # Add process noise
        if process_noise is not None:
            prop_cov += process_noise
        
        return PropagationResult(
            mean=prop_mean,
            covariance=prop_cov,
            samples=propagated,
            weights=w_m
        )


class EnsembleKalmanPropagator(UncertaintyPropagator):
    """
    Ensemble Kalman Filter style propagation.
    
    Maintains an ensemble of states and propagates each member
    through the dynamics. Used in data assimilation and high-dimensional
    state estimation.
    """
    
    def __init__(
        self,
        n_ensemble: int = 100,
        inflation_factor: float = 1.0,
        localization_radius: Optional[float] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize Ensemble Kalman propagator.
        
        Args:
            n_ensemble: Number of ensemble members
            inflation_factor: Multiplicative inflation (> 1 increases spread)
            localization_radius: Distance for covariance localization (optional)
            seed: Random seed
        """
        self.n_ensemble = n_ensemble
        self.inflation = inflation_factor
        self.localization_radius = localization_radius
        self.rng = np.random.default_rng(seed)
    
    def propagate(
        self,
        state: State,
        params: Parameters,
        transition_fn: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
        dt: float,
        process_noise: Optional[np.ndarray] = None
    ) -> PropagationResult:
        """
        Ensemble propagation with optional inflation.
        """
        # Initialize ensemble if not provided
        if state.is_ensemble:
            ensemble = state.values
            if ensemble.shape[0] != self.n_ensemble:
                # Resample to target ensemble size
                indices = self.rng.choice(ensemble.shape[0], self.n_ensemble, replace=True)
                ensemble = ensemble[indices]
        elif state.covariance is not None:
            ensemble = self.rng.multivariate_normal(
                state.values, state.covariance, size=self.n_ensemble
            )
        else:
            # Add small perturbations
            ensemble = state.values + self.rng.normal(
                0, 0.01, size=(self.n_ensemble, state.dim)
            )
        
        # Sample parameters for each ensemble member
        param_ensemble = params.sample(self.n_ensemble, self.rng).values
        
        # Propagate ensemble
        propagated = np.zeros_like(ensemble)
        for i in range(self.n_ensemble):
            propagated[i] = transition_fn(ensemble[i], param_ensemble[i], dt)
            
            # Add process noise
            if process_noise is not None:
                propagated[i] += self.rng.multivariate_normal(
                    np.zeros(state.dim), process_noise
                )
        
        # Compute statistics
        mean = np.mean(propagated, axis=0)
        
        # Apply inflation
        if self.inflation != 1.0:
            propagated = mean + self.inflation * (propagated - mean)
        
        covariance = np.cov(propagated, rowvar=False)
        if covariance.ndim == 0:
            covariance = np.array([[covariance]])
        
        return PropagationResult(
            mean=mean,
            covariance=covariance,
            samples=propagated,
            weights=None
        )
    
    def update_with_observation(
        self,
        ensemble: np.ndarray,
        observation: np.ndarray,
        observation_operator: Callable[[np.ndarray], np.ndarray],
        obs_noise_cov: np.ndarray
    ) -> np.ndarray:
        """
        Ensemble Kalman update step.
        
        Args:
            ensemble: Prior ensemble (n_ensemble, state_dim)
            observation: Observation vector
            observation_operator: Function mapping state to observation space
            obs_noise_cov: Observation noise covariance
            
        Returns:
            Updated ensemble
        """
        n_ens = ensemble.shape[0]
        
        # Predicted observations for each ensemble member
        pred_obs = np.array([observation_operator(e) for e in ensemble])
        
        # Ensemble means
        ens_mean = np.mean(ensemble, axis=0)
        pred_obs_mean = np.mean(pred_obs, axis=0)
        
        # Ensemble anomalies
        ens_anom = ensemble - ens_mean
        pred_obs_anom = pred_obs - pred_obs_mean
        
        # Covariances
        Pxy = (ens_anom.T @ pred_obs_anom) / (n_ens - 1)
        Pyy = (pred_obs_anom.T @ pred_obs_anom) / (n_ens - 1) + obs_noise_cov
        
        # Kalman gain
        K = Pxy @ np.linalg.inv(Pyy)
        
        # Update ensemble members
        updated = np.zeros_like(ensemble)
        for i in range(n_ens):
            # Perturb observation
            perturbed_obs = observation + self.rng.multivariate_normal(
                np.zeros(len(observation)), obs_noise_cov
            )
            innovation = perturbed_obs - pred_obs[i]
            updated[i] = ensemble[i] + K @ innovation
        
        return updated


class LinearizedPropagator(UncertaintyPropagator):
    """
    Linearized (Extended Kalman) uncertainty propagation.
    
    Uses first-order Taylor expansion of the transition function
    to propagate covariance. Fast but may be inaccurate for highly
    nonlinear systems.
    """
    
    def __init__(self, finite_diff_eps: float = 1e-6):
        """
        Initialize linearized propagator.
        
        Args:
            finite_diff_eps: Step size for numerical Jacobian
        """
        self.eps = finite_diff_eps
    
    def _numerical_jacobian(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        x: np.ndarray
    ) -> np.ndarray:
        """Compute Jacobian via finite differences."""
        n = len(x)
        fx = f(x)
        m = len(fx)
        J = np.zeros((m, n))
        
        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += self.eps
            J[:, i] = (f(x_plus) - fx) / self.eps
        
        return J
    
    def propagate(
        self,
        state: State,
        params: Parameters,
        transition_fn: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
        dt: float,
        process_noise: Optional[np.ndarray] = None
    ) -> PropagationResult:
        """
        Linearized propagation using first-order approximation.
        """
        mean = state.mean
        cov = state.covariance if state.covariance is not None else np.eye(state.dim) * 1e-6
        
        # Propagate mean
        prop_mean = transition_fn(mean, params.mean, dt)
        
        # Compute Jacobian at mean
        f = lambda x: transition_fn(x, params.mean, dt)
        F = self._numerical_jacobian(f, mean)
        
        # Propagate covariance: P' = F P F^T + Q
        prop_cov = F @ cov @ F.T
        if process_noise is not None:
            prop_cov += process_noise
        
        return PropagationResult(
            mean=prop_mean,
            covariance=prop_cov,
            samples=None,
            weights=None
        )


def create_propagator(
    method: Union[PropagationMethod, str],
    **kwargs
) -> UncertaintyPropagator:
    """
    Factory function to create uncertainty propagator.
    
    Args:
        method: Propagation method
        **kwargs: Method-specific parameters
        
    Returns:
        UncertaintyPropagator instance
    """
    if isinstance(method, str):
        method = PropagationMethod(method.lower())
    
    propagators = {
        PropagationMethod.MONTE_CARLO: MonteCarloUncertainty,
        PropagationMethod.UNSCENTED: UnscentedTransform,
        PropagationMethod.ENSEMBLE_KALMAN: EnsembleKalmanPropagator,
        PropagationMethod.LINEARIZED: LinearizedPropagator,
    }
    
    return propagators[method](**kwargs)
