"""
Log-likelihood functions for Bayesian inference.

Provides various likelihood functions for comparing model predictions
with observed data, supporting count data, continuous data, and composites.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats
from scipy.special import gammaln

from updff.core.state import State, Observation, Parameters
from updff.core.distribution import Distribution


class LogLikelihood(ABC):
    """
    Abstract base class for log-likelihood computation.
    
    Log-likelihood measures how probable the observed data is given
    model predictions and parameters.
    """
    
    @abstractmethod
    def __call__(
        self,
        observed: np.ndarray,
        predicted: np.ndarray,
        params: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Compute log-likelihood.
        
        Args:
            observed: Observed values [T, D]
            predicted: Model predictions [T, D]
            params: Optional additional parameters
            
        Returns:
            Log-likelihood value (higher is better fit)
        """
        pass
    
    def gradient(
        self,
        observed: np.ndarray,
        predicted: np.ndarray,
        params: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Compute gradient of log-likelihood w.r.t. predictions.
        
        Default implementation uses numerical differentiation.
        """
        eps = 1e-7
        grad = np.zeros_like(predicted, dtype=float)
        
        for idx in np.ndindex(predicted.shape):
            pred_plus = predicted.copy()
            pred_minus = predicted.copy()
            pred_plus[idx] += eps
            pred_minus[idx] -= eps
            grad[idx] = (self(observed, pred_plus, params) - 
                        self(observed, pred_minus, params)) / (2 * eps)
        
        return grad


class GaussianLikelihood(LogLikelihood):
    """
    Gaussian (Normal) likelihood for continuous observations.
    
    log P(Y|μ, σ²) = -0.5 * Σ[(Y - μ)² / σ² + log(2πσ²)]
    """
    
    def __init__(
        self,
        noise_std: Union[float, np.ndarray] = 1.0,
        heteroscedastic: bool = False
    ):
        """
        Args:
            noise_std: Observation noise standard deviation
            heteroscedastic: If True, noise_std scales with predicted mean
        """
        self.noise_std = np.atleast_1d(noise_std)
        self.heteroscedastic = heteroscedastic
    
    def __call__(
        self,
        observed: np.ndarray,
        predicted: np.ndarray,
        params: Optional[Dict[str, float]] = None
    ) -> float:
        observed = np.atleast_1d(observed)
        predicted = np.atleast_1d(predicted)
        
        if self.heteroscedastic:
            # Noise proportional to signal
            sigma = self.noise_std * np.maximum(np.abs(predicted), 1e-6)
        else:
            sigma = np.broadcast_to(self.noise_std, observed.shape)
        
        # Gaussian log-likelihood
        residuals = observed - predicted
        ll = -0.5 * np.sum((residuals / sigma) ** 2 + np.log(2 * np.pi * sigma ** 2))
        
        return float(ll)
    
    def gradient(
        self,
        observed: np.ndarray,
        predicted: np.ndarray,
        params: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        observed = np.atleast_1d(observed)
        predicted = np.atleast_1d(predicted)
        
        if self.heteroscedastic:
            sigma = self.noise_std * np.maximum(np.abs(predicted), 1e-6)
        else:
            sigma = np.broadcast_to(self.noise_std, observed.shape)
        
        # d/dμ of log-likelihood
        grad = (observed - predicted) / (sigma ** 2)
        return grad


class PoissonLikelihood(LogLikelihood):
    """
    Poisson likelihood for count data.
    
    log P(k|λ) = k * log(λ) - λ - log(k!)
    
    Appropriate for rare events and count data where variance ≈ mean.
    """
    
    def __init__(self, min_rate: float = 1e-6):
        """
        Args:
            min_rate: Minimum rate to avoid log(0)
        """
        self.min_rate = min_rate
    
    def __call__(
        self,
        observed: np.ndarray,
        predicted: np.ndarray,
        params: Optional[Dict[str, float]] = None
    ) -> float:
        observed = np.atleast_1d(observed).astype(float)
        predicted = np.atleast_1d(predicted).astype(float)
        
        # Ensure positive rate
        rate = np.maximum(predicted, self.min_rate)
        
        # Poisson log-likelihood
        # log P(k|λ) = k*log(λ) - λ - log(k!)
        ll = np.sum(observed * np.log(rate) - rate - gammaln(observed + 1))
        
        return float(ll)
    
    def gradient(
        self,
        observed: np.ndarray,
        predicted: np.ndarray,
        params: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        observed = np.atleast_1d(observed).astype(float)
        predicted = np.atleast_1d(predicted).astype(float)
        
        rate = np.maximum(predicted, self.min_rate)
        
        # d/dλ of log P(k|λ) = k/λ - 1
        grad = observed / rate - 1
        return grad


class NegativeBinomialLikelihood(LogLikelihood):
    """
    Negative Binomial likelihood for overdispersed count data.
    
    Parameterized by mean μ and dispersion k (larger k = less overdispersion).
    Variance = μ + μ²/k
    
    As k → ∞, approaches Poisson distribution.
    """
    
    def __init__(
        self,
        dispersion: float = 10.0,
        estimate_dispersion: bool = False
    ):
        """
        Args:
            dispersion: Overdispersion parameter k
            estimate_dispersion: If True, estimate k from data
        """
        self.dispersion = dispersion
        self.estimate_dispersion = estimate_dispersion
    
    def __call__(
        self,
        observed: np.ndarray,
        predicted: np.ndarray,
        params: Optional[Dict[str, float]] = None
    ) -> float:
        observed = np.atleast_1d(observed).astype(float)
        predicted = np.atleast_1d(predicted).astype(float)
        
        # Get dispersion
        k = params.get("k", self.dispersion) if params else self.dispersion
        
        if self.estimate_dispersion and params is None:
            # Moment-based estimator for k
            var_obs = np.var(observed - predicted) if len(observed) > 1 else predicted.mean()
            mean_pred = np.mean(predicted) + 1e-6
            k = max(0.1, mean_pred ** 2 / max(var_obs - mean_pred, 1e-6))
        
        # Ensure positive mean
        mu = np.maximum(predicted, 1e-6)
        
        # Negative binomial log-likelihood
        # Uses alternative parameterization: P(y|μ,k) with variance = μ + μ²/k
        # p = k / (k + μ), r = k
        p = k / (k + mu)
        
        # log P(y|k, p) = log(Γ(y+k)) - log(Γ(y+1)) - log(Γ(k)) + k*log(p) + y*log(1-p)
        ll = np.sum(
            gammaln(observed + k) - gammaln(observed + 1) - gammaln(k) +
            k * np.log(p) + observed * np.log(1 - p + 1e-10)
        )
        
        return float(ll)
    
    def estimate_dispersion_mle(
        self,
        observed: np.ndarray,
        predicted: np.ndarray
    ) -> float:
        """Estimate dispersion parameter using maximum likelihood."""
        from scipy.optimize import minimize_scalar
        
        def neg_ll(k):
            if k <= 0:
                return np.inf
            return -self(observed, predicted, {"k": k})
        
        result = minimize_scalar(neg_ll, bounds=(0.01, 100), method='bounded')
        return result.x


class BinomialLikelihood(LogLikelihood):
    """
    Binomial likelihood for proportion/rate data.
    
    log P(k|n, p) = log(n choose k) + k*log(p) + (n-k)*log(1-p)
    """
    
    def __init__(self, trials: Union[int, np.ndarray] = 1):
        """
        Args:
            trials: Number of trials (n)
        """
        self.trials = trials
    
    def __call__(
        self,
        observed: np.ndarray,
        predicted: np.ndarray,
        params: Optional[Dict[str, float]] = None
    ) -> float:
        observed = np.atleast_1d(observed).astype(float)
        predicted = np.atleast_1d(predicted).astype(float)
        
        n = params.get("trials", self.trials) if params else self.trials
        n = np.broadcast_to(n, observed.shape)
        
        # Clamp probability
        p = np.clip(predicted, 1e-10, 1 - 1e-10)
        k = observed
        
        # Binomial log-likelihood
        ll = np.sum(
            gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1) +
            k * np.log(p) + (n - k) * np.log(1 - p)
        )
        
        return float(ll)


class CompositeLogLikelihood(LogLikelihood):
    """
    Composite likelihood from multiple observation types.
    
    Combines likelihoods for different data streams (e.g., cases, deaths, hospitalizations).
    """
    
    def __init__(
        self,
        likelihoods: Dict[str, LogLikelihood],
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            likelihoods: Dict mapping observation names to likelihood functions
            weights: Optional weights for each component
        """
        self.likelihoods = likelihoods
        self.weights = weights or {k: 1.0 for k in likelihoods}
    
    def __call__(
        self,
        observed: np.ndarray,
        predicted: np.ndarray,
        params: Optional[Dict[str, float]] = None
    ) -> float:
        # Assumes observed and predicted are dicts keyed by observation type
        if isinstance(observed, dict) and isinstance(predicted, dict):
            total_ll = 0.0
            for key, likelihood in self.likelihoods.items():
                if key in observed and key in predicted:
                    ll = likelihood(observed[key], predicted[key], params)
                    total_ll += self.weights.get(key, 1.0) * ll
            return total_ll
        else:
            # Single array - use first likelihood
            first_ll = next(iter(self.likelihoods.values()))
            return first_ll(observed, predicted, params)


@dataclass
class LikelihoodResult:
    """Result from likelihood computation."""
    log_likelihood: float
    n_observations: int
    aic: float  # Akaike Information Criterion
    bic: float  # Bayesian Information Criterion
    deviance: float
    residuals: Optional[np.ndarray] = None
    
    @classmethod
    def compute(
        cls,
        observed: np.ndarray,
        predicted: np.ndarray,
        likelihood: LogLikelihood,
        n_params: int,
        params: Optional[Dict[str, float]] = None
    ) -> "LikelihoodResult":
        """Compute likelihood and information criteria."""
        ll = likelihood(observed, predicted, params)
        n = len(np.atleast_1d(observed).ravel())
        
        # Information criteria
        aic = 2 * n_params - 2 * ll
        bic = n_params * np.log(n) - 2 * ll
        
        # Deviance (relative to saturated model)
        deviance = -2 * ll
        
        # Residuals
        residuals = np.atleast_1d(observed) - np.atleast_1d(predicted)
        
        return cls(
            log_likelihood=ll,
            n_observations=n,
            aic=aic,
            bic=bic,
            deviance=deviance,
            residuals=residuals.ravel()
        )


def select_likelihood(
    observation_type: str,
    **kwargs
) -> LogLikelihood:
    """
    Factory function to select appropriate likelihood.
    
    Args:
        observation_type: Type of observation data
        **kwargs: Additional arguments for likelihood constructor
        
    Returns:
        Appropriate LogLikelihood instance
    """
    type_map = {
        "cases": NegativeBinomialLikelihood,
        "deaths": NegativeBinomialLikelihood,
        "hospitalizations": NegativeBinomialLikelihood,
        "counts": PoissonLikelihood,
        "rates": GaussianLikelihood,
        "continuous": GaussianLikelihood,
        "proportions": BinomialLikelihood,
        "binary": BinomialLikelihood,
    }
    
    likelihood_class = type_map.get(observation_type.lower(), GaussianLikelihood)
    return likelihood_class(**kwargs)
