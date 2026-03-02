"""
Forecast evaluation metrics.

Provides point forecast metrics and probabilistic forecast metrics
for comprehensive forecast evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats
from scipy.integrate import quad


@dataclass
class ForecastMetrics:
    """
    Comprehensive forecast metrics container.
    
    Stores and computes various metrics for forecast evaluation.
    """
    
    # Point forecast metrics
    mae: float = 0.0           # Mean Absolute Error
    rmse: float = 0.0          # Root Mean Squared Error
    mape: float = 0.0          # Mean Absolute Percentage Error
    bias: float = 0.0          # Mean Error (bias)
    
    # Probabilistic metrics
    crps: float = 0.0          # Continuous Ranked Probability Score
    log_score: float = 0.0     # Log Score (negative log likelihood)
    brier_score: float = 0.0   # Brier Score (for binary outcomes)
    
    # Interval metrics
    coverage_50: float = 0.0   # 50% interval coverage
    coverage_90: float = 0.0   # 90% interval coverage
    coverage_95: float = 0.0   # 95% interval coverage
    interval_score_50: float = 0.0
    interval_score_90: float = 0.0
    
    # Sharpness (spread of forecasts)
    mean_interval_width_50: float = 0.0
    mean_interval_width_90: float = 0.0
    
    # Sample size
    n_forecasts: int = 0
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "Forecast Evaluation Metrics",
            "=" * 50,
            "",
            "Point Forecast Metrics:",
            f"  MAE:  {self.mae:.4f}",
            f"  RMSE: {self.rmse:.4f}",
            f"  MAPE: {self.mape:.2%}" if self.mape < 100 else f"  MAPE: {self.mape:.2f}",
            f"  Bias: {self.bias:.4f}",
            "",
            "Probabilistic Metrics:",
            f"  CRPS:      {self.crps:.4f}",
            f"  Log Score: {self.log_score:.4f}",
            "",
            "Calibration (Coverage):",
            f"  50% CI: {self.coverage_50:.1%}",
            f"  90% CI: {self.coverage_90:.1%}",
            f"  95% CI: {self.coverage_95:.1%}",
            "",
            "Sharpness (Interval Width):",
            f"  50% CI width: {self.mean_interval_width_50:.2f}",
            f"  90% CI width: {self.mean_interval_width_90:.2f}",
            "",
            f"Based on {self.n_forecasts} forecasts"
        ]
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "mae": self.mae,
            "rmse": self.rmse,
            "mape": self.mape,
            "bias": self.bias,
            "crps": self.crps,
            "log_score": self.log_score,
            "coverage_50": self.coverage_50,
            "coverage_90": self.coverage_90,
            "coverage_95": self.coverage_95,
            "interval_score_50": self.interval_score_50,
            "interval_score_90": self.interval_score_90,
            "sharpness_50": self.mean_interval_width_50,
            "sharpness_90": self.mean_interval_width_90,
            "n_forecasts": self.n_forecasts
        }


def mean_absolute_error(
    observed: np.ndarray,
    predicted: np.ndarray
) -> float:
    """
    Compute Mean Absolute Error.
    
    MAE = (1/n) * Σ|y_i - ŷ_i|
    """
    observed = np.atleast_1d(observed).ravel()
    predicted = np.atleast_1d(predicted).ravel()
    return float(np.mean(np.abs(observed - predicted)))


def root_mean_squared_error(
    observed: np.ndarray,
    predicted: np.ndarray
) -> float:
    """
    Compute Root Mean Squared Error.
    
    RMSE = sqrt((1/n) * Σ(y_i - ŷ_i)²)
    """
    observed = np.atleast_1d(observed).ravel()
    predicted = np.atleast_1d(predicted).ravel()
    return float(np.sqrt(np.mean((observed - predicted) ** 2)))


def mean_absolute_percentage_error(
    observed: np.ndarray,
    predicted: np.ndarray,
    epsilon: float = 1e-10
) -> float:
    """
    Compute Mean Absolute Percentage Error.
    
    MAPE = (1/n) * Σ|y_i - ŷ_i| / |y_i|
    """
    observed = np.atleast_1d(observed).ravel()
    predicted = np.atleast_1d(predicted).ravel()
    
    # Avoid division by zero
    mask = np.abs(observed) > epsilon
    if not np.any(mask):
        return float('inf')
    
    return float(np.mean(np.abs(observed[mask] - predicted[mask]) / np.abs(observed[mask])))


def bias(
    observed: np.ndarray,
    predicted: np.ndarray
) -> float:
    """
    Compute bias (mean error).
    
    Bias = (1/n) * Σ(ŷ_i - y_i)
    
    Positive bias = over-prediction, Negative = under-prediction.
    """
    observed = np.atleast_1d(observed).ravel()
    predicted = np.atleast_1d(predicted).ravel()
    return float(np.mean(predicted - observed))


def crps(
    observed: np.ndarray,
    ensemble: np.ndarray
) -> float:
    """
    Compute Continuous Ranked Probability Score from ensemble.
    
    CRPS measures the integrated squared difference between the
    forecast CDF and the step function at the observation.
    
    Lower is better. CRPS = 0 for perfect forecast.
    
    Args:
        observed: Observed values [n]
        ensemble: Ensemble forecasts [n, n_members] or [n_members] for single obs
        
    Returns:
        Mean CRPS across all observations
    """
    observed = np.atleast_1d(observed).ravel()
    ensemble = np.atleast_2d(ensemble)
    
    if ensemble.shape[0] == 1 and len(observed) > 1:
        # Single ensemble for multiple observations - broadcast
        ensemble = np.tile(ensemble, (len(observed), 1))
    elif ensemble.shape[1] == len(observed) and ensemble.shape[0] != len(observed):
        # Transpose if needed
        ensemble = ensemble.T
    
    n_obs = len(observed)
    crps_values = np.zeros(n_obs)
    
    for i in range(n_obs):
        y = observed[i]
        ens = ensemble[i] if ensemble.shape[0] > 1 else ensemble[0]
        
        # Sort ensemble
        ens_sorted = np.sort(ens)
        m = len(ens_sorted)
        
        # CRPS formula using ensemble
        # CRPS = E|X - y| - 0.5 * E|X - X'|
        term1 = np.mean(np.abs(ens_sorted - y))
        
        # Compute E|X - X'| efficiently
        term2 = 0.0
        for j in range(m):
            for k in range(j + 1, m):
                term2 += np.abs(ens_sorted[j] - ens_sorted[k])
        term2 = 2 * term2 / (m * (m - 1)) if m > 1 else 0.0
        
        crps_values[i] = term1 - 0.5 * term2
    
    return float(np.mean(crps_values))


def crps_gaussian(
    observed: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray
) -> float:
    """
    Compute CRPS for Gaussian forecast distribution.
    
    Closed-form solution for Gaussian:
    CRPS = σ * [z * (2Φ(z) - 1) + 2φ(z) - 1/√π]
    
    where z = (y - μ) / σ
    """
    observed = np.atleast_1d(observed).ravel()
    mean = np.atleast_1d(mean).ravel()
    std = np.atleast_1d(std).ravel()
    
    # Standardized error
    z = (observed - mean) / std
    
    # CDF and PDF of standard normal
    phi = stats.norm.cdf(z)
    pdf = stats.norm.pdf(z)
    
    # CRPS formula
    crps_vals = std * (z * (2 * phi - 1) + 2 * pdf - 1 / np.sqrt(np.pi))
    
    return float(np.mean(crps_vals))


def log_score(
    observed: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    distribution: str = "normal"
) -> float:
    """
    Compute Log Score (negative log predictive density).
    
    LogScore = -log P(y | forecast)
    
    Lower is better.
    
    Args:
        observed: Observed values
        mean: Forecast means
        std: Forecast standard deviations
        distribution: "normal", "poisson", "negbinom"
        
    Returns:
        Mean log score
    """
    observed = np.atleast_1d(observed).ravel()
    mean = np.atleast_1d(mean).ravel()
    std = np.atleast_1d(std).ravel()
    
    if distribution == "normal":
        log_probs = stats.norm.logpdf(observed, loc=mean, scale=std)
    elif distribution == "poisson":
        log_probs = stats.poisson.logpmf(observed.astype(int), mu=mean)
    elif distribution == "negbinom":
        # Parameterize by mean and variance
        var = std ** 2
        n = mean ** 2 / (var - mean + 1e-10)
        p = mean / var
        log_probs = stats.nbinom.logpmf(observed.astype(int), n=n, p=1 - p)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    
    # Return negative mean (so lower is better)
    return float(-np.mean(log_probs))


def interval_score(
    observed: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.1
) -> float:
    """
    Compute Interval Score for prediction intervals.
    
    IS_α = (u - l) + (2/α)(l - y)𝟙(y < l) + (2/α)(y - u)𝟙(y > u)
    
    Rewards narrow intervals but penalizes when observation is outside.
    
    Args:
        observed: Observed values
        lower: Lower bound of interval
        upper: Upper bound of interval
        alpha: Significance level (0.1 for 90% CI)
        
    Returns:
        Mean interval score
    """
    observed = np.atleast_1d(observed).ravel()
    lower = np.atleast_1d(lower).ravel()
    upper = np.atleast_1d(upper).ravel()
    
    # Interval width
    width = upper - lower
    
    # Penalty for observations below lower bound
    below_penalty = (2 / alpha) * (lower - observed) * (observed < lower)
    
    # Penalty for observations above upper bound
    above_penalty = (2 / alpha) * (observed - upper) * (observed > upper)
    
    scores = width + below_penalty + above_penalty
    
    return float(np.mean(scores))


def brier_score(
    observed: np.ndarray,
    predicted_prob: np.ndarray
) -> float:
    """
    Compute Brier Score for binary probabilistic forecasts.
    
    BS = (1/n) * Σ(p_i - o_i)²
    
    where o_i ∈ {0, 1} and p_i ∈ [0, 1]
    
    Lower is better. BS = 0 for perfect forecast.
    """
    observed = np.atleast_1d(observed).ravel()
    predicted_prob = np.atleast_1d(predicted_prob).ravel()
    
    return float(np.mean((predicted_prob - observed) ** 2))


def coverage_probability(
    observed: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray
) -> float:
    """
    Compute empirical coverage probability.
    
    What fraction of observations fall within the prediction interval?
    
    For a well-calibrated X% interval, coverage should be ≈ X%.
    """
    observed = np.atleast_1d(observed).ravel()
    lower = np.atleast_1d(lower).ravel()
    upper = np.atleast_1d(upper).ravel()
    
    in_interval = (observed >= lower) & (observed <= upper)
    
    return float(np.mean(in_interval))


def sharpness(
    lower: np.ndarray,
    upper: np.ndarray
) -> float:
    """
    Compute sharpness (mean interval width).
    
    Narrower intervals = sharper forecasts (assuming good calibration).
    """
    lower = np.atleast_1d(lower).ravel()
    upper = np.atleast_1d(upper).ravel()
    
    return float(np.mean(upper - lower))


def compute_all_metrics(
    observed: np.ndarray,
    ensemble: np.ndarray,
    forecast_mean: Optional[np.ndarray] = None,
    forecast_std: Optional[np.ndarray] = None
) -> ForecastMetrics:
    """
    Compute all forecast metrics from ensemble forecasts.
    
    Args:
        observed: Observed values [n]
        ensemble: Ensemble forecasts [n, n_members]
        forecast_mean: Optional pre-computed means
        forecast_std: Optional pre-computed stds
        
    Returns:
        ForecastMetrics with all metrics computed
    """
    observed = np.atleast_1d(observed).ravel()
    ensemble = np.atleast_2d(ensemble)
    
    if ensemble.shape[1] == len(observed) and ensemble.shape[0] != len(observed):
        ensemble = ensemble.T
    
    n = len(observed)
    
    # Compute statistics from ensemble
    if forecast_mean is None:
        forecast_mean = np.mean(ensemble, axis=1)
    if forecast_std is None:
        forecast_std = np.std(ensemble, axis=1)
    
    # Quantiles for intervals
    q025 = np.percentile(ensemble, 2.5, axis=1)
    q25 = np.percentile(ensemble, 25, axis=1)
    q75 = np.percentile(ensemble, 75, axis=1)
    q975 = np.percentile(ensemble, 97.5, axis=1)
    q05 = np.percentile(ensemble, 5, axis=1)
    q95 = np.percentile(ensemble, 95, axis=1)
    
    metrics = ForecastMetrics(
        mae=mean_absolute_error(observed, forecast_mean),
        rmse=root_mean_squared_error(observed, forecast_mean),
        mape=mean_absolute_percentage_error(observed, forecast_mean),
        bias=bias(observed, forecast_mean),
        crps=crps(observed, ensemble),
        log_score=log_score(observed, forecast_mean, forecast_std),
        coverage_50=coverage_probability(observed, q25, q75),
        coverage_90=coverage_probability(observed, q05, q95),
        coverage_95=coverage_probability(observed, q025, q975),
        interval_score_50=interval_score(observed, q25, q75, alpha=0.5),
        interval_score_90=interval_score(observed, q05, q95, alpha=0.1),
        mean_interval_width_50=sharpness(q25, q75),
        mean_interval_width_90=sharpness(q05, q95),
        n_forecasts=n
    )
    
    return metrics
