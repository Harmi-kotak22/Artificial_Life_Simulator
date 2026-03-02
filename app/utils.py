"""
Helper utilities for the UPDFF Web Application.

Provides functions for data conversion, visualization helpers,
and integration between the frontend and backend framework.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta


def ensemble_to_quantiles(
    ensemble: np.ndarray,
    quantiles: List[float] = [0.05, 0.25, 0.5, 0.75, 0.95]
) -> Dict[str, np.ndarray]:
    """
    Convert ensemble forecasts to quantile statistics.
    
    Args:
        ensemble: Array of shape (time, n_ensemble) or (time, n_ensemble, n_variables)
        quantiles: List of quantiles to compute
        
    Returns:
        Dictionary with quantile arrays
    """
    result = {
        'mean': np.mean(ensemble, axis=1),
        'std': np.std(ensemble, axis=1)
    }
    
    for q in quantiles:
        result[f'q{int(q*100):02d}'] = np.percentile(ensemble, q * 100, axis=1)
    
    return result


def forecast_to_dataframe(
    forecast: np.ndarray,
    start_date: Optional[datetime] = None,
    compartment_names: List[str] = ['S', 'E', 'I', 'R'],
    quantiles: List[float] = [0.05, 0.25, 0.5, 0.75, 0.95]
) -> pd.DataFrame:
    """
    Convert forecast array to a tidy DataFrame.
    
    Args:
        forecast: Array of shape (time, n_ensemble, n_compartments)
        start_date: Starting date for the forecast
        compartment_names: Names of compartments
        quantiles: Quantiles to include
        
    Returns:
        DataFrame with columns for date, compartment, mean, std, and quantiles
    """
    n_time, n_ens, n_comp = forecast.shape
    
    if start_date is None:
        start_date = datetime.now()
    
    dates = [start_date + timedelta(days=i) for i in range(n_time)]
    
    rows = []
    for t, date in enumerate(dates):
        for c, comp_name in enumerate(compartment_names):
            data = forecast[t, :, c]
            row = {
                'date': date,
                'day': t + 1,
                'compartment': comp_name,
                'mean': np.mean(data),
                'std': np.std(data),
                'min': np.min(data),
                'max': np.max(data)
            }
            for q in quantiles:
                row[f'q{int(q*100):02d}'] = np.percentile(data, q * 100)
            rows.append(row)
    
    return pd.DataFrame(rows)


def compute_epidemic_metrics(
    forecast: np.ndarray,
    population: int,
    infectious_idx: int = 2,
    recovered_idx: int = 3
) -> Dict[str, Any]:
    """
    Compute key epidemic metrics from forecast.
    
    Args:
        forecast: Array of shape (time, n_ensemble, n_compartments)
        population: Total population
        infectious_idx: Index of infectious compartment
        recovered_idx: Index of recovered compartment
        
    Returns:
        Dictionary of metrics
    """
    I_forecast = forecast[:, :, infectious_idx]
    R_forecast = forecast[:, :, recovered_idx]
    
    # Mean trajectory
    mean_I = np.mean(I_forecast, axis=1)
    mean_R = np.mean(R_forecast, axis=1)
    
    # Peak metrics
    peak_day = int(np.argmax(mean_I) + 1)
    peak_mean = float(np.max(mean_I))
    peak_q05 = float(np.percentile(I_forecast[peak_day-1], 5))
    peak_q95 = float(np.percentile(I_forecast[peak_day-1], 95))
    
    # Attack rate (total infected)
    final_R = mean_R[-1]
    attack_rate = final_R / population * 100
    
    # Epidemic duration (time above 1% of peak)
    threshold = peak_mean * 0.01
    above_threshold = mean_I > threshold
    if np.any(above_threshold):
        start_idx = np.argmax(above_threshold)
        end_idx = len(above_threshold) - np.argmax(above_threshold[::-1])
        duration = int(end_idx - start_idx)
    else:
        duration = 0
    
    # Doubling time (early phase)
    if len(mean_I) > 5 and mean_I[0] > 0:
        early_growth = mean_I[:10]
        log_growth = np.log(early_growth + 1)
        if len(log_growth) > 1:
            growth_rate = np.polyfit(np.arange(len(log_growth)), log_growth, 1)[0]
            doubling_time = np.log(2) / growth_rate if growth_rate > 0 else float('inf')
        else:
            doubling_time = float('inf')
    else:
        doubling_time = float('inf')
    
    return {
        'peak_day': peak_day,
        'peak_mean': peak_mean,
        'peak_ci': (peak_q05, peak_q95),
        'attack_rate': attack_rate,
        'duration': duration,
        'doubling_time': doubling_time
    }


def compute_Rt(
    S_forecast: np.ndarray,
    R0: float,
    population: int
) -> np.ndarray:
    """
    Compute effective reproduction number from susceptible fraction.
    
    Args:
        S_forecast: Susceptible population over time (time, n_ensemble)
        R0: Basic reproduction number
        population: Total population
        
    Returns:
        Rt values (time, n_ensemble)
    """
    return R0 * (S_forecast / population)


def compare_scenarios(
    baseline: np.ndarray,
    intervention: np.ndarray,
    population: int,
    infectious_idx: int = 2
) -> Dict[str, Any]:
    """
    Compare baseline and intervention scenarios.
    
    Args:
        baseline: Baseline forecast array
        intervention: Intervention forecast array
        population: Total population
        infectious_idx: Index of infectious compartment
        
    Returns:
        Dictionary of comparison metrics
    """
    base_I = baseline[:, :, infectious_idx]
    int_I = intervention[:, :, infectious_idx]
    
    base_mean = np.mean(base_I, axis=1)
    int_mean = np.mean(int_I, axis=1)
    
    # Peak comparison
    base_peak = np.max(base_mean)
    int_peak = np.max(int_mean)
    peak_reduction = (base_peak - int_peak) / base_peak * 100 if base_peak > 0 else 0
    
    # Peak timing
    base_peak_day = int(np.argmax(base_mean) + 1)
    int_peak_day = int(np.argmax(int_mean) + 1)
    peak_delay = int_peak_day - base_peak_day
    
    # Total cases
    base_total = np.sum(base_mean)
    int_total = np.sum(int_mean)
    total_reduction = (base_total - int_total) / base_total * 100 if base_total > 0 else 0
    
    # Person-days of infection avoided
    person_days_avoided = base_total - int_total
    
    return {
        'peak_reduction_pct': peak_reduction,
        'peak_delay_days': peak_delay,
        'total_reduction_pct': total_reduction,
        'person_days_avoided': person_days_avoided,
        'baseline_peak': base_peak,
        'intervention_peak': int_peak,
        'baseline_peak_day': base_peak_day,
        'intervention_peak_day': int_peak_day
    }


def generate_synthetic_observations(
    true_forecast: np.ndarray,
    reporting_rate: float = 0.3,
    dispersion: float = 5.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate synthetic observations from true forecast for testing.
    
    Args:
        true_forecast: True state trajectory (time, n_compartments)
        reporting_rate: Fraction of true cases observed
        dispersion: Negative binomial dispersion parameter
        seed: Random seed
        
    Returns:
        Synthetic observed cases
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Use infectious compartment (index 2)
    true_I = true_forecast[:, 2] if true_forecast.ndim > 1 else true_forecast
    
    # Apply reporting rate
    expected_obs = true_I * reporting_rate
    
    # Add observation noise (negative binomial)
    observations = np.zeros_like(expected_obs)
    for t, exp in enumerate(expected_obs):
        if exp > 0:
            # Convert to negative binomial parameters
            r = dispersion
            p = r / (r + exp)
            observations[t] = np.random.negative_binomial(r, p)
        else:
            observations[t] = 0
    
    return observations


def format_number(value: float, precision: int = 2) -> str:
    """Format large numbers with K, M suffixes."""
    if abs(value) >= 1_000_000:
        return f"{value/1_000_000:.{precision}f}M"
    elif abs(value) >= 1_000:
        return f"{value/1_000:.{precision}f}K"
    else:
        return f"{value:.{precision}f}"


def calculate_herd_immunity_threshold(R0: float) -> float:
    """
    Calculate herd immunity threshold.
    
    Args:
        R0: Basic reproduction number
        
    Returns:
        Fraction of population needed for herd immunity
    """
    return 1 - 1/R0 if R0 > 1 else 0


def estimate_final_size(R0: float) -> float:
    """
    Estimate final epidemic size using final size equation.
    
    Solves: R_inf = 1 - exp(-R0 * R_inf)
    
    Args:
        R0: Basic reproduction number
        
    Returns:
        Final fraction of population infected (attack rate)
    """
    if R0 <= 1:
        return 0.0
    
    # Newton-Raphson iteration
    R_inf = 0.5
    for _ in range(100):
        f = R_inf - (1 - np.exp(-R0 * R_inf))
        df = 1 - R0 * np.exp(-R0 * R_inf)
        if abs(df) < 1e-10:
            break
        R_inf_new = R_inf - f / df
        if abs(R_inf_new - R_inf) < 1e-10:
            break
        R_inf = np.clip(R_inf_new, 0.001, 0.999)
    
    return R_inf


# Color schemes for consistent visualization
COLORS = {
    'susceptible': '#2ecc71',
    'exposed': '#f39c12',
    'infectious': '#e74c3c',
    'recovered': '#3498db',
    'baseline': '#e74c3c',
    'intervention': '#2ecc71',
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'warning': '#d62728',
    'info': '#17becf'
}


# Pathogen presets for the UI
PATHOGEN_PRESETS = {
    'COVID-like': {
        'R0': 2.5,
        'generation_time': 5.0,
        'infectious_period': 8.0,
        'latent_period': 3.0,
        'ifr': 0.01,
        'hospitalization_rate': 0.05
    },
    'Influenza-like': {
        'R0': 1.5,
        'generation_time': 3.0,
        'infectious_period': 5.0,
        'latent_period': 2.0,
        'ifr': 0.001,
        'hospitalization_rate': 0.01
    },
    'Measles-like': {
        'R0': 15.0,
        'generation_time': 12.0,
        'infectious_period': 8.0,
        'latent_period': 10.0,
        'ifr': 0.002,
        'hospitalization_rate': 0.02
    },
    'Cholera-like': {
        'R0': 2.0,
        'generation_time': 5.0,
        'infectious_period': 7.0,
        'latent_period': 1.0,
        'ifr': 0.01,
        'hospitalization_rate': 0.1
    }
}
