"""
Calibration assessment for probabilistic forecasts.

Provides tools to assess whether forecast probabilities match
observed frequencies (calibration/reliability).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats


@dataclass
class CalibrationAssessment:
    """
    Comprehensive calibration assessment results.
    """
    
    # PIT histogram data
    pit_values: np.ndarray
    pit_histogram: np.ndarray
    pit_bins: np.ndarray
    
    # Reliability diagram data
    forecast_probs: np.ndarray
    observed_freqs: np.ndarray
    bin_counts: np.ndarray
    
    # Statistical tests
    ks_statistic: float = 0.0
    ks_pvalue: float = 0.0
    chi2_statistic: float = 0.0
    chi2_pvalue: float = 0.0
    
    # Calibration error
    mean_calibration_error: float = 0.0
    max_calibration_error: float = 0.0
    
    # Coverage at different levels
    coverage_by_level: Dict[float, float] = None
    
    def is_calibrated(self, significance: float = 0.05) -> bool:
        """Check if forecasts are calibrated at given significance level."""
        return self.ks_pvalue > significance
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "Calibration Assessment",
            "=" * 50,
            "",
            "Statistical Tests:",
            f"  KS test: statistic={self.ks_statistic:.4f}, p-value={self.ks_pvalue:.4f}",
            f"  Chi² test: statistic={self.chi2_statistic:.4f}, p-value={self.chi2_pvalue:.4f}",
            "",
            "Calibration Error:",
            f"  Mean: {self.mean_calibration_error:.4f}",
            f"  Max:  {self.max_calibration_error:.4f}",
            "",
            f"Calibrated (α=0.05): {'Yes' if self.is_calibrated() else 'No'}",
        ]
        
        if self.coverage_by_level:
            lines.append("")
            lines.append("Coverage by Nominal Level:")
            for level, coverage in sorted(self.coverage_by_level.items()):
                diff = coverage - level
                lines.append(f"  {level:.0%}: {coverage:.1%} (diff: {diff:+.1%})")
        
        return "\n".join(lines)


def pit_histogram(
    observed: np.ndarray,
    ensemble: np.ndarray,
    n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Probability Integral Transform (PIT) histogram.
    
    For calibrated forecasts, PIT values should be uniformly distributed.
    
    PIT = F(y) where F is the forecast CDF and y is the observation.
    
    Args:
        observed: Observed values [n]
        ensemble: Ensemble forecasts [n, n_members]
        n_bins: Number of histogram bins
        
    Returns:
        pit_values: PIT values [n]
        histogram: Bin counts [n_bins]
        bin_edges: Bin edges [n_bins + 1]
    """
    observed = np.atleast_1d(observed).ravel()
    ensemble = np.atleast_2d(ensemble)
    
    if ensemble.shape[1] == len(observed) and ensemble.shape[0] != len(observed):
        ensemble = ensemble.T
    
    n = len(observed)
    pit_values = np.zeros(n)
    
    for i in range(n):
        y = observed[i]
        ens = ensemble[i] if ensemble.shape[0] > 1 else ensemble[0]
        
        # Compute CDF at observation: proportion of ensemble < observation
        pit_values[i] = np.mean(ens < y) + 0.5 * np.mean(ens == y)
    
    # Histogram
    histogram, bin_edges = np.histogram(pit_values, bins=n_bins, range=(0, 1))
    
    return pit_values, histogram, bin_edges


def pit_gaussian(
    observed: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray
) -> np.ndarray:
    """
    Compute PIT values for Gaussian forecasts.
    
    PIT = Φ((y - μ) / σ)
    """
    observed = np.atleast_1d(observed).ravel()
    mean = np.atleast_1d(mean).ravel()
    std = np.atleast_1d(std).ravel()
    
    z = (observed - mean) / std
    pit_values = stats.norm.cdf(z)
    
    return pit_values


def reliability_diagram(
    observed: np.ndarray,
    forecast_probs: np.ndarray,
    n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute reliability diagram data for binary outcomes.
    
    For calibrated forecasts, forecast probability should equal
    observed frequency in each bin.
    
    Args:
        observed: Binary observations (0 or 1)
        forecast_probs: Forecast probabilities [0, 1]
        n_bins: Number of bins
        
    Returns:
        bin_centers: Center of each probability bin
        observed_freqs: Observed frequency in each bin
        bin_counts: Number of forecasts in each bin
    """
    observed = np.atleast_1d(observed).ravel()
    forecast_probs = np.atleast_1d(forecast_probs).ravel()
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    observed_freqs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)
    
    for i in range(n_bins):
        mask = (forecast_probs >= bin_edges[i]) & (forecast_probs < bin_edges[i + 1])
        bin_counts[i] = np.sum(mask)
        
        if bin_counts[i] > 0:
            observed_freqs[i] = np.mean(observed[mask])
    
    return bin_centers, observed_freqs, bin_counts


def coverage_test(
    observed: np.ndarray,
    ensemble: np.ndarray,
    nominal_levels: List[float] = None
) -> Dict[float, float]:
    """
    Test coverage at multiple nominal levels.
    
    For well-calibrated forecasts, X% prediction intervals should
    contain X% of observations.
    
    Args:
        observed: Observed values
        ensemble: Ensemble forecasts
        nominal_levels: Coverage levels to test
        
    Returns:
        Dict mapping nominal level to empirical coverage
    """
    if nominal_levels is None:
        nominal_levels = [0.5, 0.8, 0.9, 0.95, 0.99]
    
    observed = np.atleast_1d(observed).ravel()
    ensemble = np.atleast_2d(ensemble)
    
    if ensemble.shape[1] == len(observed) and ensemble.shape[0] != len(observed):
        ensemble = ensemble.T
    
    coverage = {}
    
    for level in nominal_levels:
        alpha = 1 - level
        lower_q = alpha / 2 * 100
        upper_q = (1 - alpha / 2) * 100
        
        lower = np.percentile(ensemble, lower_q, axis=1)
        upper = np.percentile(ensemble, upper_q, axis=1)
        
        in_interval = (observed >= lower) & (observed <= upper)
        coverage[level] = float(np.mean(in_interval))
    
    return coverage


def calibration_error(
    observed: np.ndarray,
    forecast_probs: np.ndarray,
    n_bins: int = 10,
    weighted: bool = True
) -> Tuple[float, float]:
    """
    Compute calibration error for probability forecasts.
    
    Args:
        observed: Binary observations
        forecast_probs: Forecast probabilities
        n_bins: Number of bins
        weighted: Weight by bin count (ECE vs MCE)
        
    Returns:
        mean_error: Mean/Expected Calibration Error
        max_error: Maximum Calibration Error
    """
    bin_centers, observed_freqs, bin_counts = reliability_diagram(
        observed, forecast_probs, n_bins
    )
    
    # Calibration gap in each bin
    gaps = np.abs(bin_centers - observed_freqs)
    
    # Mask empty bins
    valid = bin_counts > 0
    
    if not np.any(valid):
        return 0.0, 0.0
    
    if weighted:
        # Expected Calibration Error
        weights = bin_counts[valid] / np.sum(bin_counts[valid])
        mean_error = float(np.sum(weights * gaps[valid]))
    else:
        mean_error = float(np.mean(gaps[valid]))
    
    max_error = float(np.max(gaps[valid]))
    
    return mean_error, max_error


def ks_test_uniformity(pit_values: np.ndarray) -> Tuple[float, float]:
    """
    Kolmogorov-Smirnov test for PIT uniformity.
    
    Tests H0: PIT values are uniformly distributed.
    
    Returns:
        statistic: KS test statistic
        pvalue: p-value (reject H0 if pvalue < α)
    """
    pit_values = np.atleast_1d(pit_values).ravel()
    
    # Compare to uniform distribution
    result = stats.kstest(pit_values, 'uniform')
    
    return float(result.statistic), float(result.pvalue)


def chi2_test_uniformity(
    pit_values: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, float]:
    """
    Chi-squared test for PIT uniformity.
    
    Tests H0: PIT histogram is uniform.
    """
    pit_values = np.atleast_1d(pit_values).ravel()
    
    # Compute histogram
    observed_counts, _ = np.histogram(pit_values, bins=n_bins, range=(0, 1))
    
    # Expected counts under uniformity
    expected_counts = np.ones(n_bins) * len(pit_values) / n_bins
    
    # Chi-squared test
    result = stats.chisquare(observed_counts, expected_counts)
    
    return float(result.statistic), float(result.pvalue)


def assess_calibration(
    observed: np.ndarray,
    ensemble: np.ndarray,
    n_bins: int = 10
) -> CalibrationAssessment:
    """
    Comprehensive calibration assessment.
    
    Args:
        observed: Observed values
        ensemble: Ensemble forecasts
        n_bins: Number of bins for histograms
        
    Returns:
        CalibrationAssessment with all diagnostics
    """
    observed = np.atleast_1d(observed).ravel()
    
    # PIT analysis
    pit_values, pit_hist, pit_bin_edges = pit_histogram(observed, ensemble, n_bins)
    
    # Statistical tests
    ks_stat, ks_pval = ks_test_uniformity(pit_values)
    chi2_stat, chi2_pval = chi2_test_uniformity(pit_values, n_bins)
    
    # Coverage at multiple levels
    coverage = coverage_test(observed, ensemble)
    
    # Calibration error (using PIT as proxy for probabilities)
    bin_centers = (pit_bin_edges[:-1] + pit_bin_edges[1:]) / 2
    expected_uniform = np.ones(n_bins) / n_bins
    actual_freq = pit_hist / np.sum(pit_hist)
    
    gaps = np.abs(expected_uniform - actual_freq)
    mean_cal_error = float(np.mean(gaps))
    max_cal_error = float(np.max(gaps))
    
    return CalibrationAssessment(
        pit_values=pit_values,
        pit_histogram=pit_hist,
        pit_bins=pit_bin_edges,
        forecast_probs=bin_centers,
        observed_freqs=actual_freq,
        bin_counts=pit_hist,
        ks_statistic=ks_stat,
        ks_pvalue=ks_pval,
        chi2_statistic=chi2_stat,
        chi2_pvalue=chi2_pval,
        mean_calibration_error=mean_cal_error,
        max_calibration_error=max_cal_error,
        coverage_by_level=coverage
    )
