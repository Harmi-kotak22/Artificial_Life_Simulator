"""
Forecast diagnostics and analysis tools.

Provides tools for analyzing forecast performance across
different conditions, time horizons, and ensemble properties.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats


@dataclass
class ForecastDiagnostics:
    """
    Comprehensive forecast diagnostics.
    """
    
    # Residual analysis
    residuals: np.ndarray = None
    standardized_residuals: np.ndarray = None
    residual_mean: float = 0.0
    residual_std: float = 0.0
    residual_skewness: float = 0.0
    residual_kurtosis: float = 0.0
    residual_autocorr: np.ndarray = None
    
    # Horizon analysis
    horizon_mae: Dict[int, float] = None
    horizon_coverage: Dict[int, float] = None
    
    # Ensemble diagnostics
    ensemble_spread: float = 0.0
    spread_skill_ratio: float = 0.0
    ensemble_rank_histogram: np.ndarray = None
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "Forecast Diagnostics",
            "=" * 50,
            "",
            "Residual Analysis:",
            f"  Mean:     {self.residual_mean:.4f}",
            f"  Std:      {self.residual_std:.4f}",
            f"  Skewness: {self.residual_skewness:.4f}",
            f"  Kurtosis: {self.residual_kurtosis:.4f}",
        ]
        
        if self.horizon_mae:
            lines.append("")
            lines.append("Performance by Horizon:")
            for h, mae in sorted(self.horizon_mae.items()):
                cov = self.horizon_coverage.get(h, 0) if self.horizon_coverage else 0
                lines.append(f"  Day {h}: MAE={mae:.2f}, Coverage={cov:.1%}")
        
        lines.append("")
        lines.append("Ensemble Diagnostics:")
        lines.append(f"  Mean Spread: {self.ensemble_spread:.2f}")
        lines.append(f"  Spread-Skill Ratio: {self.spread_skill_ratio:.2f}")
        
        return "\n".join(lines)


def residual_analysis(
    observed: np.ndarray,
    predicted: np.ndarray,
    predicted_std: Optional[np.ndarray] = None,
    max_lag: int = 10
) -> Dict[str, Any]:
    """
    Comprehensive residual analysis.
    
    Args:
        observed: Observed values
        predicted: Point predictions
        predicted_std: Prediction standard deviations
        max_lag: Maximum lag for autocorrelation
        
    Returns:
        Dict with residual statistics
    """
    observed = np.atleast_1d(observed).ravel()
    predicted = np.atleast_1d(predicted).ravel()
    
    residuals = observed - predicted
    
    # Basic statistics
    results = {
        "residuals": residuals,
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals)),
        "median": float(np.median(residuals)),
        "skewness": float(stats.skew(residuals)),
        "kurtosis": float(stats.kurtosis(residuals)),
    }
    
    # Standardized residuals
    if predicted_std is not None:
        predicted_std = np.atleast_1d(predicted_std).ravel()
        std_residuals = residuals / predicted_std
        results["standardized_residuals"] = std_residuals
        results["std_residual_mean"] = float(np.mean(std_residuals))
        results["std_residual_std"] = float(np.std(std_residuals))
    
    # Autocorrelation
    n = len(residuals)
    autocorr = np.zeros(max_lag)
    for lag in range(1, max_lag + 1):
        if n > lag:
            autocorr[lag - 1] = np.corrcoef(residuals[:-lag], residuals[lag:])[0, 1]
    results["autocorrelation"] = autocorr
    
    # Ljung-Box test for independence
    if n > max_lag:
        # Q = n(n+2) Σ (r_k^2 / (n-k))
        q_stat = n * (n + 2) * np.sum(autocorr ** 2 / (n - np.arange(1, max_lag + 1)))
        lb_pvalue = 1 - stats.chi2.cdf(q_stat, df=max_lag)
        results["ljung_box_statistic"] = float(q_stat)
        results["ljung_box_pvalue"] = float(lb_pvalue)
    
    # Normality tests
    if n > 8:
        shapiro_stat, shapiro_pval = stats.shapiro(residuals[:min(n, 5000)])
        results["shapiro_statistic"] = float(shapiro_stat)
        results["shapiro_pvalue"] = float(shapiro_pval)
    
    return results


def forecast_horizon_analysis(
    observed: np.ndarray,
    ensemble: np.ndarray,
    horizons: List[int],
    forecast_origin: int = 0
) -> Dict[str, Dict[int, float]]:
    """
    Analyze forecast performance by prediction horizon.
    
    Forecasts typically degrade with increasing horizon.
    
    Args:
        observed: Observed values at each time point
        ensemble: Ensemble forecasts [n_times, n_members]
        horizons: List of forecast horizons to analyze
        forecast_origin: Starting point of forecast
        
    Returns:
        Dict with metrics at each horizon
    """
    observed = np.atleast_1d(observed).ravel()
    ensemble = np.atleast_2d(ensemble)
    
    if ensemble.shape[1] == len(observed) and ensemble.shape[0] != len(observed):
        ensemble = ensemble.T
    
    results = {
        "mae": {},
        "rmse": {},
        "coverage_90": {},
        "crps": {},
        "spread": {}
    }
    
    for h in horizons:
        if forecast_origin + h >= len(observed):
            continue
        
        idx = forecast_origin + h
        y = observed[idx]
        ens = ensemble[idx] if idx < ensemble.shape[0] else ensemble[-1]
        
        # Point forecast = ensemble mean
        pred = np.mean(ens)
        
        # Metrics
        results["mae"][h] = float(np.abs(y - pred))
        results["rmse"][h] = float((y - pred) ** 2)  # Will sqrt later for aggregates
        
        # Coverage
        q05, q95 = np.percentile(ens, [5, 95])
        results["coverage_90"][h] = 1.0 if q05 <= y <= q95 else 0.0
        
        # Spread
        results["spread"][h] = float(np.std(ens))
        
        # CRPS (simplified)
        term1 = np.mean(np.abs(ens - y))
        m = len(ens)
        term2 = 0.0
        for i in range(min(m, 100)):  # Limit for speed
            for j in range(i + 1, min(m, 100)):
                term2 += np.abs(ens[i] - ens[j])
        term2 = 2 * term2 / (m * (m - 1)) if m > 1 else 0.0
        results["crps"][h] = float(term1 - 0.5 * term2)
    
    return results


def ensemble_diagnostics(
    observed: np.ndarray,
    ensemble: np.ndarray,
    n_bins: Optional[int] = None
) -> Dict[str, Any]:
    """
    Diagnose ensemble forecast properties.
    
    Args:
        observed: Observed values
        ensemble: Ensemble forecasts [n_times, n_members]
        n_bins: Number of bins for rank histogram
        
    Returns:
        Dict with ensemble diagnostics
    """
    observed = np.atleast_1d(observed).ravel()
    ensemble = np.atleast_2d(ensemble)
    
    if ensemble.shape[1] == len(observed) and ensemble.shape[0] != len(observed):
        ensemble = ensemble.T
    
    n_times = min(len(observed), ensemble.shape[0])
    n_members = ensemble.shape[1]
    
    if n_bins is None:
        n_bins = n_members + 1
    
    # Ensemble statistics
    ensemble_mean = np.mean(ensemble[:n_times], axis=1)
    ensemble_std = np.std(ensemble[:n_times], axis=1)
    
    # Spread-skill
    mean_spread = float(np.mean(ensemble_std))
    rmse = float(np.sqrt(np.mean((observed[:n_times] - ensemble_mean) ** 2)))
    spread_skill_ratio = mean_spread / rmse if rmse > 0 else float('inf')
    
    # Rank histogram
    # For each observation, count how many ensemble members are below
    ranks = np.zeros(n_times, dtype=int)
    for t in range(n_times):
        y = observed[t]
        ens = ensemble[t]
        ranks[t] = np.sum(ens < y)
    
    rank_histogram, _ = np.histogram(ranks, bins=n_bins, range=(0, n_members))
    
    # Interpret rank histogram
    # Uniform = well calibrated
    # U-shaped = underdispersed
    # Dome-shaped = overdispersed
    # Asymmetric = biased
    
    expected = n_times / n_bins
    chi2_stat = np.sum((rank_histogram - expected) ** 2 / expected)
    chi2_pval = 1 - stats.chi2.cdf(chi2_stat, df=n_bins - 1)
    
    # Reliability index
    delta = np.zeros(n_bins)
    for i in range(n_bins):
        delta[i] = (rank_histogram[i] - expected) ** 2
    reliability_index = float(np.sum(delta) / (n_bins * expected))
    
    return {
        "n_members": n_members,
        "mean_spread": mean_spread,
        "rmse": rmse,
        "spread_skill_ratio": spread_skill_ratio,
        "rank_histogram": rank_histogram,
        "rank_chi2_statistic": float(chi2_stat),
        "rank_chi2_pvalue": float(chi2_pval),
        "reliability_index": reliability_index,
        "is_calibrated": chi2_pval > 0.05,
        "interpretation": _interpret_rank_histogram(rank_histogram, expected)
    }


def _interpret_rank_histogram(histogram: np.ndarray, expected: float) -> str:
    """Interpret rank histogram shape."""
    n_bins = len(histogram)
    
    # Check for U-shape (underdispersion)
    edge_avg = (histogram[0] + histogram[-1]) / 2
    center_avg = np.mean(histogram[n_bins // 4 : 3 * n_bins // 4])
    
    if edge_avg > 1.5 * expected and center_avg < expected:
        return "U-shaped: Ensemble is underdispersed (too narrow)"
    
    # Check for dome-shape (overdispersion)
    if center_avg > 1.5 * expected and edge_avg < expected:
        return "Dome-shaped: Ensemble is overdispersed (too wide)"
    
    # Check for asymmetry (bias)
    left_sum = np.sum(histogram[:n_bins // 2])
    right_sum = np.sum(histogram[n_bins // 2:])
    
    if left_sum > 1.5 * right_sum:
        return "Left-skewed: Ensemble has positive bias (overpredicts)"
    if right_sum > 1.5 * left_sum:
        return "Right-skewed: Ensemble has negative bias (underpredicts)"
    
    # Check for uniformity
    cv = np.std(histogram) / np.mean(histogram)
    if cv < 0.3:
        return "Approximately uniform: Ensemble is well calibrated"
    
    return "Mixed pattern: May need further investigation"


def compare_forecasts(
    observed: np.ndarray,
    forecasts: Dict[str, np.ndarray],
    significance: float = 0.05
) -> Dict[str, Any]:
    """
    Compare multiple forecasting methods.
    
    Args:
        observed: Observed values
        forecasts: Dict mapping method name to ensemble forecasts
        significance: Significance level for statistical tests
        
    Returns:
        Comparison results including rankings and statistical tests
    """
    from updff.validation.metrics import compute_all_metrics
    
    observed = np.atleast_1d(observed).ravel()
    
    results = {
        "metrics": {},
        "rankings": {},
        "pairwise_tests": {}
    }
    
    # Compute metrics for each method
    for name, ensemble in forecasts.items():
        metrics = compute_all_metrics(observed, ensemble)
        results["metrics"][name] = metrics.to_dict()
    
    # Rank by different criteria
    methods = list(forecasts.keys())
    
    # Rank by CRPS (lower is better)
    crps_values = [results["metrics"][m]["crps"] for m in methods]
    crps_ranks = np.argsort(crps_values)
    results["rankings"]["crps"] = {methods[i]: rank + 1 for rank, i in enumerate(crps_ranks)}
    
    # Rank by MAE
    mae_values = [results["metrics"][m]["mae"] for m in methods]
    mae_ranks = np.argsort(mae_values)
    results["rankings"]["mae"] = {methods[i]: rank + 1 for rank, i in enumerate(mae_ranks)}
    
    # Rank by calibration (coverage closest to 90%)
    cov_values = [abs(results["metrics"][m]["coverage_90"] - 0.9) for m in methods]
    cov_ranks = np.argsort(cov_values)
    results["rankings"]["calibration"] = {methods[i]: rank + 1 for rank, i in enumerate(cov_ranks)}
    
    # Pairwise Diebold-Mariano tests
    for i, method1 in enumerate(methods):
        for method2 in methods[i + 1:]:
            ens1 = np.atleast_2d(forecasts[method1])
            ens2 = np.atleast_2d(forecasts[method2])
            
            if ens1.shape[1] == len(observed):
                ens1 = ens1.T
            if ens2.shape[1] == len(observed):
                ens2 = ens2.T
            
            # Compare squared errors
            pred1 = np.mean(ens1, axis=1)[:len(observed)]
            pred2 = np.mean(ens2, axis=1)[:len(observed)]
            
            d = (observed - pred1) ** 2 - (observed - pred2) ** 2
            
            # DM statistic
            d_mean = np.mean(d)
            d_var = np.var(d) / len(d)
            dm_stat = d_mean / np.sqrt(d_var) if d_var > 0 else 0
            dm_pval = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
            
            key = f"{method1}_vs_{method2}"
            results["pairwise_tests"][key] = {
                "dm_statistic": float(dm_stat),
                "p_value": float(dm_pval),
                "significant": dm_pval < significance,
                "better": method1 if d_mean < 0 else method2
            }
    
    # Overall best
    avg_ranks = {}
    for method in methods:
        ranks = [
            results["rankings"]["crps"][method],
            results["rankings"]["mae"][method],
            results["rankings"]["calibration"][method]
        ]
        avg_ranks[method] = np.mean(ranks)
    
    results["overall_best"] = min(avg_ranks, key=avg_ranks.get)
    results["average_ranks"] = avg_ranks
    
    return results
