"""
Proper scoring rules for probabilistic forecast evaluation.

Proper scoring rules incentivize honest probability reporting.
A forecaster maximizes their expected score by reporting their true beliefs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats


class ProperScoringRule(ABC):
    """
    Abstract base class for proper scoring rules.
    
    A scoring rule S(P, y) measures the quality of a probabilistic
    forecast P when outcome y is observed.
    
    A scoring rule is "proper" if the expected score is maximized
    when the forecaster reports their true beliefs.
    """
    
    @abstractmethod
    def score(
        self,
        observed: np.ndarray,
        **forecast_params
    ) -> np.ndarray:
        """
        Compute score for each forecast-observation pair.
        
        Convention: Lower scores are better (like a loss function).
        """
        pass
    
    def mean_score(
        self,
        observed: np.ndarray,
        **forecast_params
    ) -> float:
        """Compute mean score across all forecasts."""
        scores = self.score(observed, **forecast_params)
        return float(np.mean(scores))


class CRPSScore(ProperScoringRule):
    """
    Continuous Ranked Probability Score.
    
    CRPS generalizes MAE to probabilistic forecasts.
    For a deterministic forecast, CRPS = MAE.
    
    CRPS = E|X - y| - 0.5 * E|X - X'|
    
    where X, X' are independent draws from the forecast distribution.
    """
    
    def score(
        self,
        observed: np.ndarray,
        ensemble: Optional[np.ndarray] = None,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute CRPS.
        
        Either provide ensemble forecasts or mean/std for Gaussian.
        """
        observed = np.atleast_1d(observed).ravel()
        
        if ensemble is not None:
            return self._crps_ensemble(observed, ensemble)
        elif mean is not None and std is not None:
            return self._crps_gaussian(observed, mean, std)
        else:
            raise ValueError("Provide either ensemble or (mean, std)")
    
    def _crps_ensemble(
        self,
        observed: np.ndarray,
        ensemble: np.ndarray
    ) -> np.ndarray:
        """CRPS from ensemble."""
        ensemble = np.atleast_2d(ensemble)
        
        if ensemble.shape[1] == len(observed) and ensemble.shape[0] != len(observed):
            ensemble = ensemble.T
        
        n = len(observed)
        scores = np.zeros(n)
        
        for i in range(n):
            y = observed[i]
            ens = ensemble[i] if ensemble.shape[0] > 1 else ensemble[0]
            m = len(ens)
            
            # E|X - y|
            term1 = np.mean(np.abs(ens - y))
            
            # E|X - X'|
            term2 = 0.0
            for j in range(m):
                for k in range(j + 1, m):
                    term2 += np.abs(ens[j] - ens[k])
            term2 = 2 * term2 / (m * (m - 1)) if m > 1 else 0.0
            
            scores[i] = term1 - 0.5 * term2
        
        return scores
    
    def _crps_gaussian(
        self,
        observed: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray
    ) -> np.ndarray:
        """Closed-form CRPS for Gaussian."""
        mean = np.atleast_1d(mean).ravel()
        std = np.atleast_1d(std).ravel()
        
        z = (observed - mean) / std
        phi = stats.norm.cdf(z)
        pdf = stats.norm.pdf(z)
        
        scores = std * (z * (2 * phi - 1) + 2 * pdf - 1 / np.sqrt(np.pi))
        
        return scores


class LogScore(ProperScoringRule):
    """
    Logarithmic Score (negative log likelihood).
    
    LogScore = -log P(y | forecast)
    
    Heavily penalizes confident wrong forecasts.
    Sensitive to the tails of the distribution.
    """
    
    def __init__(self, distribution: str = "normal"):
        """
        Args:
            distribution: "normal", "poisson", "negbinom", "ensemble"
        """
        self.distribution = distribution
    
    def score(
        self,
        observed: np.ndarray,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        ensemble: Optional[np.ndarray] = None,
        dispersion: float = 1.0
    ) -> np.ndarray:
        """
        Compute log score.
        """
        observed = np.atleast_1d(observed).ravel()
        
        if self.distribution == "normal":
            mean = np.atleast_1d(mean).ravel()
            std = np.atleast_1d(std).ravel()
            log_probs = stats.norm.logpdf(observed, loc=mean, scale=std)
            
        elif self.distribution == "poisson":
            mean = np.atleast_1d(mean).ravel()
            log_probs = stats.poisson.logpmf(observed.astype(int), mu=mean)
            
        elif self.distribution == "negbinom":
            mean = np.atleast_1d(mean).ravel()
            std = np.atleast_1d(std).ravel()
            var = std ** 2
            n = mean ** 2 / np.maximum(var - mean, 1e-10)
            p = mean / np.maximum(var, 1e-10)
            log_probs = stats.nbinom.logpmf(observed.astype(int), n=n, p=1 - p)
            
        elif self.distribution == "ensemble":
            # Kernel density estimation
            ensemble = np.atleast_2d(ensemble)
            if ensemble.shape[1] == len(observed) and ensemble.shape[0] != len(observed):
                ensemble = ensemble.T
            
            log_probs = np.zeros(len(observed))
            for i in range(len(observed)):
                ens = ensemble[i] if ensemble.shape[0] > 1 else ensemble[0]
                kde = stats.gaussian_kde(ens)
                log_probs[i] = np.log(kde(observed[i])[0] + 1e-10)
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")
        
        return -log_probs  # Negative so lower is better


class IntervalScore(ProperScoringRule):
    """
    Interval Score for prediction intervals.
    
    IS_α(l, u, y) = (u - l) + (2/α)(l - y)⁺ + (2/α)(y - u)⁺
    
    Rewards narrow intervals, penalizes observations outside.
    """
    
    def __init__(self, alpha: float = 0.1):
        """
        Args:
            alpha: Significance level (0.1 for 90% interval)
        """
        self.alpha = alpha
    
    def score(
        self,
        observed: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Compute interval score."""
        observed = np.atleast_1d(observed).ravel()
        lower = np.atleast_1d(lower).ravel()
        upper = np.atleast_1d(upper).ravel()
        
        width = upper - lower
        below_penalty = (2 / self.alpha) * np.maximum(lower - observed, 0)
        above_penalty = (2 / self.alpha) * np.maximum(observed - upper, 0)
        
        return width + below_penalty + above_penalty


class WeightedIntervalScore(ProperScoringRule):
    """
    Weighted Interval Score - combines multiple interval scores.
    
    Approximates CRPS by averaging interval scores at multiple levels.
    
    WIS = (1/K) * Σ IS_{α_k}
    """
    
    def __init__(
        self,
        quantile_levels: List[float] = None
    ):
        """
        Args:
            quantile_levels: Quantile levels to use (symmetric around 0.5)
        """
        if quantile_levels is None:
            # Default: 23 quantile pairs (0.01 to 0.99)
            quantile_levels = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 
                             0.3, 0.35, 0.4, 0.45, 0.5]
        self.quantile_levels = quantile_levels
    
    def score(
        self,
        observed: np.ndarray,
        ensemble: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Compute WIS from ensemble."""
        observed = np.atleast_1d(observed).ravel()
        ensemble = np.atleast_2d(ensemble)
        
        if ensemble.shape[1] == len(observed) and ensemble.shape[0] != len(observed):
            ensemble = ensemble.T
        
        n = len(observed)
        total_score = np.zeros(n)
        n_intervals = 0
        
        for alpha_half in self.quantile_levels:
            if alpha_half >= 0.5:
                continue
            
            alpha = 2 * alpha_half
            lower_q = alpha_half * 100
            upper_q = (1 - alpha_half) * 100
            
            lower = np.percentile(ensemble, lower_q, axis=1)
            upper = np.percentile(ensemble, upper_q, axis=1)
            
            is_scorer = IntervalScore(alpha=alpha)
            total_score += is_scorer.score(observed, lower, upper)
            n_intervals += 1
        
        # Add median absolute deviation
        median = np.percentile(ensemble, 50, axis=1)
        total_score += 0.5 * np.abs(observed - median)
        
        return total_score / (n_intervals + 0.5)


class BrierScore(ProperScoringRule):
    """
    Brier Score for binary probabilistic forecasts.
    
    BS = (p - o)²
    
    where p is forecast probability and o ∈ {0, 1}.
    """
    
    def score(
        self,
        observed: np.ndarray,
        predicted_prob: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Compute Brier score."""
        observed = np.atleast_1d(observed).ravel()
        predicted_prob = np.atleast_1d(predicted_prob).ravel()
        
        return (predicted_prob - observed) ** 2


def compute_skill_score(
    forecast_score: float,
    reference_score: float,
    perfect_score: float = 0.0
) -> float:
    """
    Compute skill score relative to reference forecast.
    
    SS = (S_ref - S_forecast) / (S_ref - S_perfect)
    
    SS = 1 means perfect forecast
    SS = 0 means same as reference
    SS < 0 means worse than reference
    
    Args:
        forecast_score: Score of the forecast being evaluated
        reference_score: Score of reference forecast (e.g., climatology)
        perfect_score: Score of perfect forecast (usually 0)
        
    Returns:
        Skill score in (-∞, 1]
    """
    if reference_score == perfect_score:
        return 1.0 if forecast_score == perfect_score else 0.0
    
    return (reference_score - forecast_score) / (reference_score - perfect_score)


def decompose_brier_score(
    observed: np.ndarray,
    forecast_probs: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """
    Decompose Brier Score into reliability, resolution, and uncertainty.
    
    BS = Reliability - Resolution + Uncertainty
    
    - Reliability: measures calibration (lower is better)
    - Resolution: measures discrimination (higher is better)
    - Uncertainty: depends only on base rate
    """
    observed = np.atleast_1d(observed).ravel()
    forecast_probs = np.atleast_1d(forecast_probs).ravel()
    
    n = len(observed)
    base_rate = np.mean(observed)
    
    # Bin forecasts
    bin_edges = np.linspace(0, 1, n_bins + 1)
    
    reliability = 0.0
    resolution = 0.0
    
    for i in range(n_bins):
        mask = (forecast_probs >= bin_edges[i]) & (forecast_probs < bin_edges[i + 1])
        n_k = np.sum(mask)
        
        if n_k > 0:
            o_k = np.mean(observed[mask])  # Observed frequency in bin
            f_k = np.mean(forecast_probs[mask])  # Mean forecast in bin
            
            reliability += n_k * (f_k - o_k) ** 2
            resolution += n_k * (o_k - base_rate) ** 2
    
    reliability /= n
    resolution /= n
    uncertainty = base_rate * (1 - base_rate)
    
    brier_score = float(np.mean((forecast_probs - observed) ** 2))
    
    return {
        "brier_score": brier_score,
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "reliability_check": reliability - resolution + uncertainty  # Should equal BS
    }
