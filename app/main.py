"""
UPDFF - Universal Probabilistic Disaster Forecasting Framework
Interactive Web Dashboard - User-Friendly Version

This dashboard uses real-world parameters similar to those used by:
- CDC (Centers for Disease Control and Prevention)
- WHO (World Health Organization)
- ECDC (European Centre for Disease Prevention and Control)

Run with: streamlit run app/main.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json

# Page configuration - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Disease Outbreak Forecaster",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load GHSI Health Data
@st.cache_data
def load_ghsi_data():
    """Load Global Health Security Index data for regional parameters"""
    try:
        ghsi_path = os.path.join(os.path.dirname(__file__), 'data', 'ghsi_combined_data.json')
        with open(ghsi_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # Fallback embedded data for key countries
        return {
            "countries": {
                "United States": {"overall": 75.9, "prevention": 83.1, "detection": 98.2, "health_system": 73.8, "vaccination_rate": 69.5, "health_expenditure_per_capita": 11648.5},
                "United Kingdom": {"overall": 77.9, "prevention": 68.3, "detection": 87.3, "health_system": 80.3, "vaccination_rate": 75.2, "health_expenditure_per_capita": 4866.0},
                "India": {"overall": 42.8, "prevention": 34.8, "detection": 43.5, "health_system": 42.7, "vaccination_rate": 67.2, "health_expenditure_per_capita": 63.8},
                "Brazil": {"overall": 59.7, "prevention": 54.6, "detection": 66.7, "health_system": 56.8, "vaccination_rate": 81.8, "health_expenditure_per_capita": 680.8},
                "Germany": {"overall": 70.0, "prevention": 58.7, "detection": 84.6, "health_system": 76.2, "vaccination_rate": 76.2, "health_expenditure_per_capita": 5906.8},
            }
        }

# Load GHSI data at startup
GHSI_DATA = load_ghsi_data()

def get_country_health_params(country_name):
    """Get health parameters for a country with statistical distributions"""
    countries = GHSI_DATA.get("countries", {})
    
    # Try exact match first, then partial match
    country_data = countries.get(country_name)
    if not country_data:
        # Try partial match
        for name, data in countries.items():
            if country_name.lower() in name.lower() or name.lower() in country_name.lower():
                country_data = data
                break
    
    if not country_data:
        # Default values (global average)
        country_data = {
            "overall": 50.0,
            "prevention": 50.0,
            "detection": 50.0,
            "health_system": 50.0,
            "vaccination_rate": 60.0,
            "health_expenditure_per_capita": 500.0
        }
    
    return country_data

def sample_health_params_from_distribution(country_data, n_samples=1):
    """
    Sample health parameters from statistical distributions.
    Returns dict of sampled values for use in Monte Carlo simulation.
    
    Distributions used:
    - Hygiene (prevention): Beta distribution (bounded 0-100)
    - Healthcare: Log-Normal (always positive, right-skewed)
    - Detection: Normal (symmetric around measured value)
    - Immunity (vaccination): Beta distribution
    """
    samples = {}
    
    # Prevention/Hygiene: Beta distribution
    # Convert 0-100 score to Beta parameters
    prevention = country_data.get("prevention", 50) / 100
    alpha_prev = max(1, prevention * 10)
    beta_prev = max(1, (1 - prevention) * 10)
    samples["hygiene"] = np.random.beta(alpha_prev, beta_prev, n_samples) * 100
    
    # Health System: Log-Normal distribution
    health_sys = country_data.get("health_system", 50)
    mu = np.log(max(1, health_sys))
    sigma = 0.15  # ~15% variance
    samples["healthcare"] = np.random.lognormal(mu, sigma, n_samples)
    samples["healthcare"] = np.clip(samples["healthcare"], 10, 100)
    
    # Detection Rate: Normal distribution
    detection = country_data.get("detection", 50)
    samples["detection"] = np.random.normal(detection, detection * 0.10, n_samples)
    samples["detection"] = np.clip(samples["detection"], 20, 100)
    
    # Vaccination/Immunity: Beta distribution
    vax_rate = country_data.get("vaccination_rate", 60) / 100
    alpha_vax = max(1, vax_rate * 8)
    beta_vax = max(1, (1 - vax_rate) * 8)
    samples["immunity"] = np.random.beta(alpha_vax, beta_vax, n_samples) * 100
    
    # Medication affordability: inverse of health expenditure (higher exp = more expensive)
    health_exp = country_data.get("health_expenditure_per_capita", 500)
    # Normalize: India ($64) = high affordability (85), US ($11k) = low affordability (45)
    affordability = 90 - (np.log(health_exp + 1) / np.log(12000)) * 50
    samples["medication"] = np.random.normal(affordability, 8, n_samples)
    samples["medication"] = np.clip(samples["medication"], 30, 95)
    
    return samples

def apply_health_modifiers_to_seir(health_params, disease_type: str = "Custom Disease"):
    """
    Apply regional health parameters to calculate SEIR modifiers.
    Uses disease-specific vaccine efficacy from peer-reviewed sources.
    
    Returns dict with:
    - r0_modifier: Multiplier for R0 (based on hygiene)
    - gamma_modifier: Multiplier for recovery rate (based on healthcare+medication)
    - immunity_factor: Fraction of population effectively immune (vaccination × disease-specific efficacy)
    - detection_modifier: Multiplier for detection rate
    - efficacy_source: Citation for vaccine efficacy data
    """
    # Extract sampled values (handle both array and scalar)
    hygiene = np.atleast_1d(health_params["hygiene"])[0]
    healthcare = np.atleast_1d(health_params["healthcare"])[0]
    medication = np.atleast_1d(health_params["medication"])[0]
    detection = np.atleast_1d(health_params["detection"])[0]
    immunity = np.atleast_1d(health_params["immunity"])[0]  # Vaccination rate %
    
    # R0 modifier: Hygiene affects transmission
    # Higher hygiene = lower transmission (range: 0.85-1.15, softened from 0.75-1.25)
    r0_modifier = 1.15 - (hygiene / 100) * 0.30
    
    # Gamma modifier: Healthcare + Medication affect recovery rate
    # Higher values = faster recovery (range: 0.95-1.15)
    healthcare_factor = 0.95 + (healthcare / 100) * 0.15
    medication_factor = 0.97 + (medication / 100) * 0.06
    gamma_modifier = healthcare_factor * medication_factor
    
    # ================================================================
    # DISEASE-SPECIFIC IMMUNITY FACTOR
    # ================================================================
    # Get vaccine efficacy against infection for this specific disease
    vaccine_data = get_vaccine_efficacy(disease_type, months_since_vaccination=4.0)
    vaccine_efficacy = vaccine_data['efficacy']
    
    # immunity = vaccination coverage (%), vaccine_efficacy = % of vaccinated actually protected
    # Also account for natural immunity (assume 10-20% of unvaccinated have prior infection)
    vaccination_immunity = (immunity / 100) * vaccine_efficacy
    natural_immunity = ((100 - immunity) / 100) * 0.15  # 15% of unvaccinated have natural immunity
    
    # Total effective immunity (cannot exceed 95%)
    immunity_factor = min(0.95, vaccination_immunity + natural_immunity)
    
    # Detection modifier: How well the country detects cases
    # Higher = more cases detected (range: 0.85-1.10, softened)
    detection_modifier = 0.85 + (detection / 100) * 0.25
    
    return {
        'r0_modifier': r0_modifier,
        'gamma_modifier': gamma_modifier,
        'immunity_factor': immunity_factor,
        'detection_modifier': detection_modifier,
        'vaccine_efficacy': vaccine_efficacy,
        'efficacy_source': vaccine_data['source']
    }

# ============================================================
# HYBRID ENSEMBLE FORECAST
# Combines: Renewal Equation (40%) + Trend/ARIMA (35%) + SEIR (25%)
# Based on CDC FluSight ensemble methodology
# ============================================================
def hybrid_ensemble_forecast(
    actual_data,
    n_days,
    n_sims=500,
    population=1_000_000,
    detection_rate=50,
    calib_fraction=0.30,
    spread_rate=2.5,
    r0_decay=15,
    time_varying=True,
    use_health_params=False,
    country_health=None,
    selected_disease="Custom Disease",
    disease_profile=None,
):
    """
    Hybrid Ensemble Forecast combining three complementary models:
    
    1. **Renewal Equation** (weight=0.40): Data-driven, captures R_t trajectory.
       Used by CDC's Epiforecast, LSHTM, etc.
    2. **Trend Extrapolation** (weight=0.35): Polynomial trend + weekly seasonality + 
       changepoint detection. Captures multi-wave patterns better than SEIR.
    3. **SEIR Model** (weight=0.25): Mechanistic compartmental model with stochastic 
       transitions.Provides epidemiological constraints.
    
    Uses train/test split: first `calib_fraction` of data for calibration, rest for 
    honest validation.
    
    Returns:
        all_new_cases: np.array(n_days, n_sims) - ensemble predictions
        all_infected: np.array(n_days, n_sims) - active infections (SEIR component)
        diagnostics: dict with calibration info
    """
    
    all_new_cases = np.zeros((n_days, n_sims))
    all_infected = np.zeros((n_days, n_sims))
    
    # ---- Input sanitization (handles any country/duration/disease) ----
    actual_data = np.array(actual_data, dtype=float)
    actual_data = np.nan_to_num(actual_data, nan=0.0, posinf=0.0, neginf=0.0)
    actual_data = np.maximum(actual_data, 0.0)  # No negative cases
    
    # Guard: if data is too short, return simple noise-around-mean
    if n_days < 7:
        mean_val = max(np.mean(actual_data), 1.0)
        for sim in range(n_sims):
            all_new_cases[:, sim] = np.maximum(0, mean_val * np.random.lognormal(0, 0.3, n_days))
        diagnostics = {
            'calib_days': 0, 'r_t_full': np.ones(n_days), 'weekly_pattern': np.ones(7),
            'has_changepoint': False, 'init_R0': 1.0, 'val_mae': 0.0, 'val_corr': 0.0,
            'n_renewal': 0, 'n_trend': 0, 'n_seir': n_sims, 'scale_factor': 1.0,
        }
        return all_new_cases, all_infected, diagnostics
    
    # ---- Health modifiers (precompute defaults) ----
    r0_modifier = 1.0
    gamma_modifier = 1.0
    immunity_factor = 0.0
    detection_modifier = 1.0
    
    if use_health_params and country_health:
        _hs = sample_health_params_from_distribution(country_health)
        _mod = apply_health_modifiers_to_seir(_hs, disease_type=selected_disease)
        r0_modifier = _mod['r0_modifier']
        gamma_modifier = _mod['gamma_modifier']
        immunity_factor = _mod['immunity_factor']
        detection_modifier = _mod['detection_modifier']
    
    # ---- Calibration split (ensure we always leave validation days) ----
    calib_days = max(7, int(n_days * calib_fraction))
    calib_days = min(calib_days, n_days - 7)  # Always leave at least 7 days for validation
    calib_days = max(7, calib_days)  # But need at least 7 for calibration
    calib_data = actual_data[:calib_days]
    
    # Smooth data for parameter estimation
    smooth_all = pd.Series(actual_data).rolling(7, min_periods=1, center=True).mean().values
    smooth_all = np.nan_to_num(smooth_all, nan=0.0)
    smooth_all = np.maximum(smooth_all, 0.1)
    calib_smooth = smooth_all[:calib_days]
    
    # ---- Weekly reporting pattern (from calibration only) ----
    weekly_pattern = np.ones(7)
    if calib_days >= 14:
        for dow in range(7):
            dow_vals = calib_data[dow::7]
            if len(dow_vals) >= 2 and np.mean(calib_data) > 0:
                weekly_pattern[dow] = np.mean(dow_vals) / (np.mean(calib_data) + 1e-8)
        weekly_pattern = weekly_pattern / (np.mean(weekly_pattern) + 1e-8)
        weekly_pattern = np.clip(weekly_pattern, 0.3, 2.0)
    
    # ---- Serial interval distribution (for renewal equation) ----
    # Use disease profile if available, otherwise COVID-like
    if disease_profile:
        gen_time = disease_profile.get('latent_days', 3) + disease_profile.get('infectious_days', 7) * 0.5
    else:
        gen_time = 5.0  # COVID default
    
    # Discretized gamma-like serial interval (mean = gen_time)
    max_si = min(21, max(calib_days - 2, 3))  # At least 3, at most 21
    si_days = np.arange(1, max_si + 1)
    # Gamma PDF approximation: shape=gen_time, rate=1
    shape_param = max(1.5, gen_time)
    si_weights = (si_days ** (shape_param - 1)) * np.exp(-si_days)
    si_weights = si_weights / (np.sum(si_weights) + 1e-12)
    
    # ---- Estimate R_t from calibration data (Cori method) ----
    r_t_calib = np.ones(calib_days)
    if max_si < calib_days:
        for t in range(max_si, calib_days):
            denom = 0.0
            for s in range(len(si_weights)):
                if t - s - 1 >= 0:
                    denom += si_weights[s] * calib_smooth[t - s - 1]
            if denom > 1.0:
                r_t_calib[t] = calib_smooth[t] / denom
            else:
                r_t_calib[t] = r_t_calib[t - 1] if t > 0 else 1.0
    r_t_calib = np.clip(r_t_calib, 0.1, 4.0)
    
    # Extrapolate R_t beyond calibration with MEAN-REVERSION toward 1.0
    # (Prevents runaway exponential growth in the renewal equation)
    valid_rt = r_t_calib[max_si:]
    if len(valid_rt) >= 5:
        # Use the LAST few R_t values as the starting point for extrapolation
        recent_rt = np.mean(valid_rt[-min(7, len(valid_rt)):])
        recent_rt = np.clip(recent_rt, 0.3, 3.0)
        
        r_t_full = np.ones(n_days)
        # Fill calibration period with actual estimates
        for i in range(min(max_si, n_days)):
            r_t_full[i] = r_t_calib[i] if i < len(r_t_calib) else 1.0
        for i in range(max_si, min(calib_days, n_days)):
            idx = i - max_si
            if idx < len(valid_rt):
                r_t_full[i] = valid_rt[idx]
            else:
                r_t_full[i] = recent_rt
        
        # Beyond calibration: mean-revert toward 1.0 with half-life of 14 days
        reversion_rate = np.log(2) / 14.0  # Half-life = 14 days
        for i in range(calib_days, n_days):
            days_beyond = i - calib_days
            reversion = np.exp(-reversion_rate * days_beyond)
            r_t_full[i] = 1.0 + (recent_rt - 1.0) * reversion
        r_t_full = np.clip(r_t_full, 0.2, 3.0)
    else:
        rt_mean = np.mean(r_t_calib[max_si:]) if len(r_t_calib) > max_si else 1.2
        rt_mean = np.clip(rt_mean, 0.5, 3.0)
        r_t_full = np.ones(n_days) * rt_mean
    
    # Maximum plausible daily cases (for capping all models)
    # Scale-aware: based on actual data peaks AND population
    data_max = np.max(actual_data) if np.max(actual_data) > 0 else 100
    max_daily_cases = max(
        data_max * 3.0,                    # 3x observed peak
        np.mean(actual_data) * 10.0,       # 10x average
        population * 0.01,                  # 1% of population per day max
        1000                                # Absolute minimum
    )
    max_daily_cases = min(max_daily_cases, population * 0.05)  # Never exceed 5% of pop/day
    
    # ---- Trend/Changepoint detection (from calibration only) ----
    # Detect if there's a peak/trough in calibration data
    if calib_days >= 21:
        # Use piecewise linear: find best breakpoint
        best_bp = calib_days // 2
        best_err = np.inf
        for bp in range(7, calib_days - 7):
            seg1 = calib_smooth[:bp]
            seg2 = calib_smooth[bp:]
            c1 = np.polyfit(np.arange(len(seg1)), seg1, 1)
            c2 = np.polyfit(np.arange(len(seg2)), seg2, 1)
            err = np.sum((seg1 - np.polyval(c1, np.arange(len(seg1))))**2) + \
                  np.sum((seg2 - np.polyval(c2, np.arange(len(seg2))))**2)
            if err < best_err:
                best_err = err
                best_bp = bp
        
        # Get slopes before and after breakpoint
        slope_before = np.polyfit(np.arange(best_bp), calib_smooth[:best_bp], 1)[0]
        slope_after = np.polyfit(np.arange(calib_days - best_bp), calib_smooth[best_bp:], 1)[0]
        has_changepoint = (slope_before > 0 and slope_after < 0) or (slope_before < 0 and slope_after > 0)
    else:
        has_changepoint = False
        slope_after = 0
    
    # ================================================================
    # MODEL 1: RENEWAL EQUATION (40% weight, sims 0 to n_sims*0.4)
    # I(t) = R(t) * sum_s[ I(t-s) * w(s) ]
    # ================================================================
    n_renewal = int(n_sims * 0.4)
    
    for sim in range(n_renewal):
        # Per-sim R_t noise
        rt_scale = np.random.lognormal(0, 0.20)
        rt_drift_std = 0.03  # Daily random walk on R_t
        
        # Initialize with actual calibration data (+ noise)
        pred = np.zeros(n_days)
        noise_scale = np.random.uniform(0.8, 1.2)
        pred[:calib_days] = calib_smooth[:calib_days] * noise_scale
        
        # Apply per-sim R_t trajectory
        sim_rt = r_t_full.copy() * rt_scale
        for d in range(max(calib_days, max_si), n_days):
            # Random walk on R_t (small drift)
            sim_rt[d] *= np.random.lognormal(0, rt_drift_std)
            sim_rt[d] = np.clip(sim_rt[d], 0.1, 3.5)
            
            # Renewal equation: I(t) = R(t) * sum_s[I(t-s)*w(s)]
            infectiousness = 0.0
            for s in range(len(si_weights)):
                if d - s - 1 >= 0:
                    infectiousness += si_weights[s] * pred[d - s - 1]
            
            pred[d] = min(max(0, sim_rt[d] * infectiousness), max_daily_cases)
        
        # Apply weekly pattern + observation noise, cap at max
        for d in range(n_days):
            dow = d % 7
            obs_noise = np.random.lognormal(0, 0.15)
            pred[d] = min(max(0, pred[d] * weekly_pattern[dow] * obs_noise), max_daily_cases)
        
        all_new_cases[:, sim] = pred
    
    # ================================================================
    # MODEL 2: TREND EXTRAPOLATION (35% weight)
    # Polynomial + weekly seasonal + changepoint-aware
    # ================================================================
    n_trend = int(n_sims * 0.35)
    
    for sim in range(n_trend):
        pred = np.zeros(n_days)
        
        # Fit polynomial to calibration smooth data
        t_calib = np.arange(calib_days)
        
        # Choose polynomial degree based on data complexity
        if has_changepoint and calib_days >= 21:
            poly_deg = min(4, max(2, calib_days // 10))
        else:
            poly_deg = min(3, max(1, calib_days // 10))
        poly_deg = min(poly_deg, calib_days - 1)  # Can't fit degree >= n_points
        
        # Add per-sim data perturbation for ensemble diversity
        perturbed = calib_smooth.copy()
        perturbed *= np.random.lognormal(0, 0.1, calib_days)
        
        try:
            coeffs = np.polyfit(t_calib, perturbed, poly_deg)
            
            # Extrapolate to all days
            t_all = np.arange(n_days)
            trend = np.polyval(coeffs, t_all)
            
            # Dampen extrapolation beyond calibration to prevent wild swings
            for d in range(calib_days, n_days):
                days_beyond = d - calib_days
                # Damping: gradually pull toward last calibration value  
                damping = np.exp(-days_beyond / (n_days * 0.5))
                last_val = trend[calib_days - 1]
                trend[d] = trend[d] * damping + last_val * (1 - damping)
            
            trend = np.maximum(trend, 0)
            
            # Residual noise from calibration
            calib_residuals = calib_data - np.polyval(coeffs, t_calib)
            residual_std = np.std(calib_residuals)
            
            # Add noise + weekly pattern, cap at max
            noise = np.random.normal(0, max(residual_std, 1) * 1.2, n_days)
            for d in range(n_days):
                dow = d % 7
                pred[d] = min(max(0, (trend[d] + noise[d]) * weekly_pattern[dow]), max_daily_cases)
        except Exception:
            # Fallback: constant extrapolation from last calibration value
            pred[:calib_days] = calib_smooth[:calib_days]
            last_val = calib_smooth[-1] if calib_days > 0 else 100
            for d in range(calib_days, n_days):
                pred[d] = max(0, last_val * np.random.lognormal(0, 0.2))
        
        all_new_cases[:, n_renewal + sim] = pred
    
    # ================================================================
    # MODEL 3: STOCHASTIC SEIR (25% weight)
    # Mechanistic model with time-varying R_t
    # ================================================================
    n_seir = n_sims - n_renewal - n_trend
    sigma = 1.0 / 3.0  # Latent period
    gamma_param = 1.0 / 7.0  # Infectious period
    
    if disease_profile:
        sigma = 1.0 / max(0.5, disease_profile.get('latent_days', 3))
        gamma_param = 1.0 / max(0.5, disease_profile.get('infectious_days', 7))
    
    # Estimate initial R0 from calibration
    if len(r_t_calib[max_si:]) > 0:
        init_R0 = np.mean(r_t_calib[max_si:max_si + min(7, len(r_t_calib) - max_si)])
    else:
        init_R0 = spread_rate
    
    for sim in range(n_seir):
        # Per-simulation uncertainty
        sim_R0 = init_R0 * np.random.lognormal(0, 0.35)
        sim_R0 *= r0_modifier
        sim_sigma = sigma * np.random.lognormal(0, 0.20)
        sim_gamma = gamma_param * np.random.lognormal(0, 0.20) * gamma_modifier
        
        # Decay rate
        decay_rate = (r0_decay / 100) / 7 * np.random.uniform(0.5, 1.5)
        
        # Initial conditions from data
        init_I = max(1, int(actual_data[0] * np.random.lognormal(0, 0.5))) if actual_data[0] > 0 else max(1, int(np.mean(actual_data[:7]) + 1))
        eff_pop = population * (1 - immunity_factor) if use_health_params else population
        eff_pop = max(eff_pop, init_I * 10)  # Ensure population > initial cases
        
        S = max(0, eff_pop - init_I * 2)
        E = init_I
        I_val = init_I
        R_val = population - eff_pop if use_health_params else 0
        
        eff_detect = detection_rate * (detection_modifier if use_health_params else 1.0) / 100
        
        for day in range(n_days):
            # Time-varying R0 using estimated trajectory + decay
            if time_varying:
                current_R0 = max(0.3, sim_R0 * np.exp(-decay_rate * day))
            else:
                current_R0 = sim_R0
            
            sim_beta = current_R0 * sim_gamma
            
            new_E = np.random.poisson(max(0, sim_beta * S * I_val / population))
            new_E = min(new_E, int(max(0, S)))
            new_I = np.random.poisson(max(0, sim_sigma * E))
            new_I = min(new_I, int(max(0, E)))
            new_R = np.random.poisson(max(0, sim_gamma * I_val))
            new_R = min(new_R, int(max(0, I_val)))
            
            S -= new_E
            E += new_E - new_I
            I_val += new_I - new_R
            R_val += new_R
            
            dow = day % 7
            all_new_cases[day, n_renewal + n_trend + sim] = max(0, new_I * eff_detect * weekly_pattern[dow])
            all_infected[day, n_renewal + n_trend + sim] = I_val
    
    # ================================================================
    # ENSEMBLE CALIBRATION: Scale SEIR predictions to match data
    # (Only uses calibration data, not validation data!)
    # ================================================================
    seir_start = n_renewal + n_trend
    seir_calib_mean = np.mean(all_new_cases[:calib_days, seir_start:])
    calib_actual_mean = np.mean(calib_data)
    scale_factor = 1.0  # default
    
    if seir_calib_mean > 0 and calib_actual_mean > 0:
        scale_factor = calib_actual_mean / seir_calib_mean
        scale_factor = np.clip(scale_factor, 0.01, 50.0)
        all_new_cases[:, seir_start:] *= scale_factor
    
    # Hard cap ALL predictions at max plausible value
    all_new_cases = np.clip(all_new_cases, 0, max_daily_cases)
    all_new_cases = np.nan_to_num(all_new_cases, nan=0.0, posinf=0.0, neginf=0.0)
    
    # ---- Diagnostics ----
    val_pred_mean = np.mean(all_new_cases[calib_days:], axis=1)
    val_actual = actual_data[calib_days:]
    
    if len(val_actual) > 5 and len(val_pred_mean) > 5:
        val_mae = np.mean(np.abs(val_actual - val_pred_mean))
        val_corr = np.corrcoef(val_actual, val_pred_mean)[0, 1]
        val_corr = 0.0 if np.isnan(val_corr) else val_corr
    else:
        val_mae = 0.0
        val_corr = 0.0
    
    diagnostics = {
        'calib_days': calib_days,
        'r_t_full': r_t_full,
        'weekly_pattern': weekly_pattern,
        'has_changepoint': has_changepoint,
        'init_R0': init_R0,
        'val_mae': val_mae,
        'val_corr': val_corr,
        'n_renewal': n_renewal,
        'n_trend': n_trend,
        'n_seir': n_seir,
        'scale_factor': scale_factor if seir_calib_mean > 0 and calib_actual_mean > 0 else 1.0,
    }
    
    return all_new_cases, all_infected, diagnostics


# Custom CSS for better styling and readability
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .explain-box {
        background-color: #f0f7ff;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.95rem;
        color: #1a1a1a;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
        color: #1a1a1a;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
        color: #1a1a1a;
    }
    .danger-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
        color: #1a1a1a;
    }
    .metric-explain {
        font-size: 0.85rem;
        color: #666;
        margin-top: 0.3rem;
    }
    .real-world-example {
        background-color: #e8f5e9;
        border: 1px solid #4caf50;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #1a1a1a;
    }
    .step-indicator {
        background-color: #1f77b4;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin-right: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# REAL-WORLD DISEASE PROFILES
# Based on CDC, WHO, and peer-reviewed epidemiological studies
# ============================================================

DISEASE_PROFILES = {
    "COVID-19 (Original Strain)": {
        "description": "The original SARS-CoV-2 strain that emerged in late 2019",
        "spread_rate": 2.5,  # R0
        "spread_rate_range": (2.0, 3.0),
        "days_until_contagious": 3,  # Latent period
        "days_contagious": 8,  # Infectious period
        "days_to_symptoms": 5,  # Incubation period
        "percent_showing_symptoms": 70,
        "percent_hospitalized": 5,
        "percent_fatal": 1.0,
        "source": "CDC, WHO (2020-2021)"
    },
    "COVID-19 (Delta Variant)": {
        "description": "More transmissible variant that emerged in late 2020",
        "spread_rate": 5.0,
        "spread_rate_range": (4.0, 6.0),
        "days_until_contagious": 2,
        "days_contagious": 7,
        "days_to_symptoms": 4,
        "percent_showing_symptoms": 75,
        "percent_hospitalized": 7,
        "percent_fatal": 1.5,
        "source": "CDC, Public Health England (2021)"
    },
    "COVID-19 (Omicron Variant)": {
        "description": "Highly transmissible but generally milder variant",
        "spread_rate": 8.0,
        "spread_rate_range": (6.0, 10.0),
        "days_until_contagious": 2,
        "days_contagious": 5,
        "days_to_symptoms": 3,
        "percent_showing_symptoms": 60,
        "percent_hospitalized": 2,
        "percent_fatal": 0.3,
        "source": "WHO, CDC (2022)"
    },
    "Seasonal Influenza": {
        "description": "Regular seasonal flu that circulates annually",
        "spread_rate": 1.3,
        "spread_rate_range": (1.1, 1.5),
        "days_until_contagious": 1,
        "days_contagious": 5,
        "days_to_symptoms": 2,
        "percent_showing_symptoms": 67,
        "percent_hospitalized": 1,
        "percent_fatal": 0.1,
        "source": "CDC Seasonal Flu Reports"
    },
    "Pandemic Influenza (1918-like)": {
        "description": "Severe pandemic flu similar to the 1918 Spanish Flu",
        "spread_rate": 2.0,
        "spread_rate_range": (1.5, 2.5),
        "days_until_contagious": 1,
        "days_contagious": 7,
        "days_to_symptoms": 2,
        "percent_showing_symptoms": 80,
        "percent_hospitalized": 10,
        "percent_fatal": 2.5,
        "source": "Historical records, CDC"
    },
    "Measles": {
        "description": "Highly contagious viral disease (vaccine-preventable)",
        "spread_rate": 15.0,
        "spread_rate_range": (12.0, 18.0),
        "days_until_contagious": 4,
        "days_contagious": 8,
        "days_to_symptoms": 10,
        "percent_showing_symptoms": 90,
        "percent_hospitalized": 25,
        "percent_fatal": 0.2,
        "source": "WHO, CDC"
    },
    "Ebola": {
        "description": "Severe hemorrhagic fever with high fatality",
        "spread_rate": 1.8,
        "spread_rate_range": (1.4, 2.2),
        "days_until_contagious": 2,
        "days_contagious": 10,
        "days_to_symptoms": 8,
        "percent_showing_symptoms": 95,
        "percent_hospitalized": 80,
        "percent_fatal": 50,
        "source": "WHO Ebola Response (2014-2016)"
    },
    "Mpox (Monkeypox)": {
        "description": "Viral zoonotic disease, 2022 global outbreak (Clade IIb)",
        "spread_rate": 1.8,
        "spread_rate_range": (1.1, 2.4),
        "days_until_contagious": 1,
        "days_contagious": 21,
        "days_to_symptoms": 7,
        "percent_showing_symptoms": 85,
        "percent_hospitalized": 5,
        "percent_fatal": 0.1,
        "source": "WHO, CDC — 2022 Multi-country Outbreak"
    },
    "Custom Disease": {
        "description": "Set your own parameters for a hypothetical or new disease",
        "spread_rate": 2.0,
        "spread_rate_range": (1.0, 20.0),
        "days_until_contagious": 3,
        "days_contagious": 7,
        "days_to_symptoms": 5,
        "percent_showing_symptoms": 70,
        "percent_hospitalized": 5,
        "percent_fatal": 1.0,
        "source": "User-defined"
    }
}

# ============================================================
# VACCINE EFFICACY AGAINST INFECTION - Disease-Specific Data
# ============================================================
# This measures efficacy against INFECTION (S→E transition), NOT against
# severe disease/hospitalization which is typically higher.
# Values represent population-level effectiveness accounting for waning.

VACCINE_EFFICACY_DATA = {
    "COVID-19 (Original Strain)": {
        "efficacy_vs_infection": 0.85,  # 85% initially, high efficacy
        "efficacy_range": (0.80, 0.92),
        "waning_per_month": 0.05,  # ~5% reduction per month
        "source": "Polack FP et al. NEJM 2020 (BNT162b2 Phase 3 Trial)",
        "notes": "mRNA vaccines showed 90-95% efficacy vs symptomatic infection initially"
    },
    "COVID-19 (Delta Variant)": {
        "efficacy_vs_infection": 0.55,  # Reduced vs Delta
        "efficacy_range": (0.45, 0.65),
        "waning_per_month": 0.06,
        "source": "Lopez Bernal J et al. NEJM 2021 (UK Effectiveness Study)",
        "notes": "Two doses Pfizer: 88% vs Alpha, 67% vs Delta; AstraZeneca lower"
    },
    "COVID-19 (Omicron Variant)": {
        "efficacy_vs_infection": 0.35,  # Significant immune escape
        "efficacy_range": (0.25, 0.50),
        "waning_per_month": 0.08,
        "source": "Andrews N et al. NEJM 2022 (UK Health Security Agency)",
        "notes": "Booster restores to ~65-75%, but wanes to 30-40% after 3 months"
    },
    "Seasonal Influenza": {
        "efficacy_vs_infection": 0.45,  # Highly variable by season/match
        "efficacy_range": (0.30, 0.60),
        "waning_per_month": 0.02,
        "source": "CDC Flu Vaccine Effectiveness Reports 2010-2023",
        "notes": "10-60% depending on strain match; average ~40-50%"
    },
    "Pandemic Influenza (1918-like)": {
        "efficacy_vs_infection": 0.55,  # Modern pandemic flu vaccines
        "efficacy_range": (0.40, 0.70),
        "waning_per_month": 0.03,
        "source": "Osterholm MT et al. Lancet Infect Dis 2012 (Meta-analysis)",
        "notes": "No 1918 vaccine existed; estimate based on modern pandemic preparedness"
    },
    "Measles": {
        "efficacy_vs_infection": 0.97,  # MMR highly effective
        "efficacy_range": (0.95, 0.99),
        "waning_per_month": 0.001,  # Very durable immunity
        "source": "WHO Position Paper on Measles Vaccines 2017",
        "notes": "Two doses MMR: 97% efficacy; one dose: 93%"
    },
    "Ebola": {
        "efficacy_vs_infection": 0.975,  # rVSV-ZEBOV highly effective
        "efficacy_range": (0.90, 0.99),
        "waning_per_month": 0.005,
        "source": "Henao-Restrepo AM et al. Lancet 2017 (Guinea Ring Vaccination Trial)",
        "notes": "100% efficacy in contacts vaccinated within 10 days; 97.5% overall"
    },
    "Mpox (Monkeypox)": {
        "efficacy_vs_infection": 0.85,  # JYNNEOS (MVA-BN) vaccine
        "efficacy_range": (0.75, 0.90),
        "waning_per_month": 0.01,
        "source": "Dalton AF et al. MMWR 2023 (CDC JYNNEOS Effectiveness Study)",
        "notes": "JYNNEOS 2-dose: ~85% effectiveness; ACAM2000 higher but more adverse events"
    },
    "Custom Disease": {
        "efficacy_vs_infection": 0.50,  # Default moderate efficacy
        "efficacy_range": (0.30, 0.70),
        "waning_per_month": 0.04,
        "source": "User-defined / Generic estimate",
        "notes": "Default value; adjust based on specific disease characteristics"
    }
}

def get_vaccine_efficacy(disease_type: str, months_since_vaccination: float = 3.0) -> dict:
    """
    Get vaccine efficacy against infection for a specific disease.
    
    Args:
        disease_type: Name of disease (must match DISEASE_PROFILES keys)
        months_since_vaccination: Average time since vaccination in population
    
    Returns:
        dict with 'efficacy', 'range', 'source'
    """
    # Find best match for disease type
    efficacy_data = VACCINE_EFFICACY_DATA.get(disease_type)
    
    if not efficacy_data:
        # Try partial match (e.g., "COVID" matches "COVID-19 (Omicron Variant)")
        for key in VACCINE_EFFICACY_DATA:
            if disease_type.lower() in key.lower() or key.lower() in disease_type.lower():
                efficacy_data = VACCINE_EFFICACY_DATA[key]
                break
    
    if not efficacy_data:
        # Default fallback
        efficacy_data = VACCINE_EFFICACY_DATA["Custom Disease"]
    
    # Calculate waned efficacy
    base_efficacy = efficacy_data["efficacy_vs_infection"]
    waning = efficacy_data["waning_per_month"] * months_since_vaccination
    waned_efficacy = max(0.10, base_efficacy - waning)  # Floor at 10%
    
    return {
        'efficacy': waned_efficacy,
        'base_efficacy': base_efficacy,
        'range': efficacy_data["efficacy_range"],
        'source': efficacy_data["source"],
        'notes': efficacy_data["notes"]
    }

# Intervention effectiveness based on real-world studies
INTERVENTION_EFFECTS = {
    "School Closures": {
        "description": "Closing schools and universities to reduce contact among children and young adults",
        "transmission_reduction": 25,
        "delay_days": 3,
        "evidence": "CDC studies show 20-30% reduction in flu transmission",
        "real_example": "During H1N1 (2009), school closures reduced spread by 25-30% in affected areas"
    },
    "Work From Home (50% workforce)": {
        "description": "Half of the workforce works remotely, reducing workplace contacts",
        "transmission_reduction": 20,
        "delay_days": 1,
        "evidence": "Modeling studies estimate 15-25% reduction",
        "real_example": "COVID-19 telework policies reduced workplace transmission significantly"
    },
    "Mask Mandate (Surgical Masks)": {
        "description": "Required mask-wearing in public indoor spaces",
        "transmission_reduction": 30,
        "delay_days": 7,
        "evidence": "Meta-analysis shows 30-40% reduction with good compliance",
        "real_example": "Kansas counties with mask mandates had 6% decrease vs 100% increase without"
    },
    "Social Distancing (6 feet)": {
        "description": "Maintaining physical distance in public spaces",
        "transmission_reduction": 40,
        "delay_days": 3,
        "evidence": "Each meter of distance reduces risk by ~80%",
        "real_example": "Studies show transmission risk drops significantly beyond 1 meter"
    },
    "Lockdown (Stay-at-Home Order)": {
        "description": "Strict movement restrictions with only essential activities allowed",
        "transmission_reduction": 70,
        "delay_days": 14,
        "evidence": "Wuhan lockdown reduced R from 3.86 to 0.32",
        "real_example": "Italy's lockdown reduced Rt from 3.0 to 0.5 within 3 weeks"
    },
    "Vaccination (50% coverage)": {
        "description": "Half of the population vaccinated with effective vaccine",
        "transmission_reduction": 45,
        "delay_days": 30,
        "evidence": "Depends on vaccine efficacy, typically 60-95% protection",
        "real_example": "COVID vaccines reduced transmission by 40-60% even against variants"
    },
    "Testing & Isolation": {
        "description": "Widespread testing with isolation of positive cases",
        "transmission_reduction": 35,
        "delay_days": 7,
        "evidence": "Early case identification can reduce spread by 30-50%",
        "real_example": "South Korea's aggressive testing limited COVID-19 spread effectively"
    },
    "Travel Restrictions": {
        "description": "Limiting travel between regions or countries",
        "transmission_reduction": 15,
        "delay_days": 14,
        "evidence": "More effective early in outbreak; delays rather than prevents",
        "real_example": "Travel bans delayed COVID-19 spread to new regions by 2-3 weeks"
    }
}

# Sidebar navigation
st.sidebar.markdown("## 🏥 Outbreak Forecaster")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "What would you like to do?",
    ["📖 Learn How This Works", "🔮 Forecast an Outbreak", "⚖️ Compare Interventions", 
     "✅ Validate Forecast", "🎮 Agent Simulation", "📊 Understanding Results", "❓ FAQ & Help"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 About This Tool")
st.sidebar.info(
    "This forecasting tool uses the same methods and parameters that health organizations "
    "like the **CDC** and **WHO** use to predict disease outbreaks.\n\n"
    "All default values come from peer-reviewed research and official health agency reports."
)


# ============================================================
# LEARN HOW THIS WORKS PAGE
# ============================================================
if page == "📖 Learn How This Works":
    st.markdown('<h1 class="main-header">🏥 Understanding Disease Outbreak Forecasting</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Learn how health organizations predict the spread of infectious diseases</p>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="explain-box">
    <strong>🎯 What is this tool?</strong><br><br>
    This is a <strong>disease outbreak forecaster</strong> - similar to weather forecasting, but for epidemics. 
    Just like meteorologists predict tomorrow's weather using scientific models, epidemiologists (disease scientists) 
    use mathematical models to predict how diseases will spread.<br><br>
    <strong>Why is this useful?</strong> Forecasts help hospitals prepare beds, help governments decide on policies, 
    and help you understand what might happen in your community.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("## 🧪 The Key Numbers That Predict Disease Spread")
    st.markdown("""
    Health organizations like the CDC and WHO track specific numbers to forecast outbreaks. 
    Here are the most important ones:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 1️⃣ Spread Rate (Reproduction Number)")
        st.markdown("""
        <div class="explain-box">
        <strong>What it means:</strong> How many people, on average, will catch the disease from one infected person.<br><br>
        <strong>Real examples:</strong>
        <ul>
        <li>Seasonal Flu: 1.3 people</li>
        <li>COVID-19 (Original): 2.5 people</li>
        <li>Measles: 15 people</li>
        </ul>
        <strong>Why it matters:</strong> If this number is above 1, the outbreak grows. Below 1, it shrinks.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 2️⃣ Time Until Contagious")
        st.markdown("""
        <div class="explain-box">
        <strong>What it means:</strong> After getting infected, how many days until you can spread it to others.<br><br>
        <strong>Real examples:</strong>
        <ul>
        <li>Flu: 1 day (very quick)</li>
        <li>COVID-19: 2-3 days</li>
        <li>Measles: 4 days</li>
        </ul>
        <strong>Why it matters:</strong> Shorter times mean faster spread and less time to identify and isolate cases.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 3️⃣ Days Contagious")
        st.markdown("""
        <div class="explain-box">
        <strong>What it means:</strong> How many days an infected person can spread the disease to others.<br><br>
        <strong>Real examples:</strong>
        <ul>
        <li>Flu: 5 days</li>
        <li>COVID-19: 5-8 days</li>
        <li>Ebola: 10 days</li>
        </ul>
        <strong>Why it matters:</strong> Longer contagious periods mean more opportunity to spread the disease.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 4️⃣ Detection Rate")
        st.markdown("""
        <div class="explain-box">
        <strong>What it means:</strong> What percentage of actual cases get officially reported/counted.<br><br>
        <strong>Real examples:</strong>
        <ul>
        <li>Flu: ~10-20% (most go unreported)</li>
        <li>COVID-19: ~20-50% (varies by testing)</li>
        <li>Hospitalized diseases: ~80-90%</li>
        </ul>
        <strong>Why it matters:</strong> The true number of cases is often much higher than reported numbers.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("## 📈 How Outbreaks Typically Unfold")
    
    # Create illustrative epidemic curve
    days = np.arange(0, 120)
    
    # Generate typical epidemic curve
    peak_day = 45
    cases = 10000 * np.exp(-0.5 * ((days - peak_day) / 20) ** 2)
    
    fig = go.Figure()
    
    # Add phases
    fig.add_vrect(x0=0, x1=20, fillcolor="green", opacity=0.1, line_width=0,
                  annotation_text="Early Growth", annotation_position="top")
    fig.add_vrect(x0=20, x1=35, fillcolor="yellow", opacity=0.1, line_width=0,
                  annotation_text="Rapid Spread", annotation_position="top")
    fig.add_vrect(x0=35, x1=55, fillcolor="red", opacity=0.1, line_width=0,
                  annotation_text="Peak", annotation_position="top")
    fig.add_vrect(x0=55, x1=90, fillcolor="blue", opacity=0.1, line_width=0,
                  annotation_text="Decline", annotation_position="top")
    fig.add_vrect(x0=90, x1=120, fillcolor="gray", opacity=0.1, line_width=0,
                  annotation_text="End", annotation_position="top")
    
    fig.add_trace(go.Scatter(x=days, y=cases, mode='lines', 
                            line=dict(color='#e74c3c', width=4),
                            name='Daily New Cases'))
    
    fig.update_layout(
        title="Typical Outbreak Pattern (Epidemic Curve)",
        xaxis_title="Days Since First Case",
        yaxis_title="Number of New Cases Per Day",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="explain-box">
    <strong>📊 Understanding the Epidemic Curve:</strong><br><br>
    1. <strong style="color: green;">Early Growth (Green):</strong> Few cases, disease spreading undetected. 
       This is when interventions are most effective.<br>
    2. <strong style="color: #DAA520;">Rapid Spread (Yellow):</strong> Cases doubling quickly. 
       Community spread established.<br>
    3. <strong style="color: red;">Peak (Red):</strong> Maximum daily cases. Hospitals most stressed.<br>
    4. <strong style="color: blue;">Decline (Blue):</strong> Fewer new cases as more people become immune or 
       interventions take effect.<br>
    5. <strong style="color: gray;">End (Gray):</strong> Outbreak winding down, but vigilance still needed.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("## 🎲 Why Forecasts Show Ranges, Not Exact Numbers")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="explain-box">
        <strong>Uncertainty is Normal and Important!</strong><br><br>
        Weather forecasters don't say "It will rain exactly 2.3 inches tomorrow." 
        They say "There's a 70% chance of 1-3 inches of rain."<br><br>
        <strong>Disease forecasts work the same way.</strong> We show you:<br>
        • A <strong>most likely</strong> outcome (the middle line)<br>
        • A <strong>range of possibilities</strong> (the shaded area)<br><br>
        The shaded area shows where we're 90% confident the true number will fall.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Demo uncertainty plot
        np.random.seed(42)
        days_demo = np.arange(1, 31)
        mean_cases = 100 * np.exp(0.1 * days_demo)
        
        fig_unc = go.Figure()
        
        # 90% CI
        fig_unc.add_trace(go.Scatter(
            x=np.concatenate([days_demo, days_demo[::-1]]),
            y=np.concatenate([mean_cases * 1.5, (mean_cases * 0.6)[::-1]]),
            fill='toself', fillcolor='rgba(31, 119, 180, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Range of possibilities (90% confident)'
        ))
        
        fig_unc.add_trace(go.Scatter(
            x=days_demo, y=mean_cases,
            mode='lines', line=dict(color='#1f77b4', width=3),
            name='Most likely outcome'
        ))
        
        fig_unc.update_layout(
            title="Forecast with Uncertainty Range",
            xaxis_title="Days from Now",
            yaxis_title="Projected Cases",
            height=300,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        st.plotly_chart(fig_unc, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("## ✅ Ready to Try It?")
    st.markdown("""
    Now that you understand the basics, head to **"🔮 Forecast an Outbreak"** in the sidebar 
    to create your own disease outbreak forecast!
    """)


# ============================================================
# FORECAST AN OUTBREAK PAGE
# ============================================================
elif page == "🔮 Forecast an Outbreak":
    st.markdown("# 🔮 Create Your Disease Outbreak Forecast")
    
    st.markdown("""
    <div class="explain-box">
    <strong>How to use this tool:</strong><br>
    1. Choose a disease or set custom parameters<br>
    2. Enter information about your population<br>
    3. Click "Generate Forecast" to see predictions<br><br>
    All default values are based on real CDC and WHO data.
    </div>
    """, unsafe_allow_html=True)
    
    # Step 1: Disease Selection
    st.markdown('<span class="step-indicator">Step 1</span> **Select the Disease**', unsafe_allow_html=True)
    
    disease_choice = st.selectbox(
        "Choose a disease to model",
        list(DISEASE_PROFILES.keys()),
        help="Select a known disease with real-world parameters, or choose 'Custom Disease' to set your own"
    )
    
    profile = DISEASE_PROFILES[disease_choice]
    
    # Show disease info
    st.markdown(f"""
    <div class="real-world-example">
    <strong>📋 About {disease_choice}:</strong><br>
    {profile['description']}<br><br>
    <strong>Data source:</strong> {profile['source']}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Step 2: Disease Parameters
    st.markdown('<span class="step-indicator">Step 2</span> **Disease Characteristics**', unsafe_allow_html=True)
    st.markdown("*These values are pre-filled based on real-world data. Adjust if needed.*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        spread_rate = st.slider(
            "🦠 How many people does one infected person spread it to?",
            min_value=0.5, max_value=20.0, value=float(profile['spread_rate']), step=0.1,
            help=f"Scientific term: Basic Reproduction Number (R₀). "
                 f"Real-world range for {disease_choice}: {profile['spread_rate_range'][0]}-{profile['spread_rate_range'][1]}"
        )
        
        if spread_rate > 1:
            if spread_rate < 2:
                st.caption("📊 Low spread rate - similar to seasonal flu")
            elif spread_rate < 5:
                st.caption("📊 Moderate spread - similar to COVID-19")
            elif spread_rate < 10:
                st.caption("📊 High spread - similar to chickenpox")
            else:
                st.caption("📊 Very high spread - similar to measles")
        else:
            st.caption("📊 Below 1 means the outbreak will naturally die out")
        
        days_until_contagious = st.slider(
            "⏱️ Days after infection until person becomes contagious",
            min_value=0, max_value=14, value=int(profile['days_until_contagious']),
            help="How long from getting infected to being able to spread it. "
                 "Also called the 'latent period' or 'pre-infectious period'."
        )
        st.caption(f"ℹ️ For {disease_choice}, this is typically {profile['days_until_contagious']} days")
    
    with col2:
        days_contagious = st.slider(
            "📅 How many days is a person contagious?",
            min_value=1, max_value=21, value=int(profile['days_contagious']),
            help="Duration someone can spread the disease to others. "
                 "Also called the 'infectious period'."
        )
        st.caption(f"ℹ️ For {disease_choice}, this is typically {profile['days_contagious']} days")
        
        detection_rate = st.slider(
            "🔍 What percentage of cases get officially reported?",
            min_value=5, max_value=100, value=30, step=5,
            help="Not everyone who gets sick gets tested or reported. "
                 "For COVID-19, CDC estimates only 20-50% of cases were reported."
        )
        if detection_rate < 30:
            st.caption("📊 Low detection - many cases go unnoticed (common for mild diseases)")
        elif detection_rate < 60:
            st.caption("📊 Moderate detection - typical for diseases with testing programs")
        else:
            st.caption("📊 High detection - typical for severe diseases requiring hospitalization")
        
        fatality_rate = st.slider(
            "💀 Infection Fatality Rate (IFR %)",
            min_value=0.0, max_value=60.0,
            value=float(profile.get('percent_fatal', 1.0)),
            step=0.1,
            help="Percentage of infected people who die. "
                 f"Default for {disease_choice}: {profile.get('percent_fatal', 1.0)}% (from CDC/WHO data)"
        )
        if fatality_rate < 0.5:
            st.caption("📊 Very low fatality - similar to seasonal flu")
        elif fatality_rate < 2:
            st.caption("📊 Low fatality - similar to COVID-19")
        elif fatality_rate < 10:
            st.caption("📊 Moderate fatality - significant public health concern")
        else:
            st.caption("📊 High fatality - severe disease (e.g. Ebola, MERS)")
    
    st.markdown("---")
    
    # Step 3: Population
    st.markdown('<span class="step-indicator">Step 3</span> **Your Population**', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        population_preset = st.selectbox(
            "Select a population size",
            ["Small town (10,000)", "Small city (100,000)", "Medium city (500,000)",
             "Large city (1,000,000)", "Metropolitan area (5,000,000)", 
             "🌍 USA (330 million)", "🌍 UK/France/Germany (60-80 million)", 
             "🌍 India/China (1.4 billion)", "Custom"],
            index=1
        )
        
        pop_map = {
            "Small town (10,000)": 10000,
            "Small city (100,000)": 100000,
            "Medium city (500,000)": 500000,
            "Large city (1,000,000)": 1000000,
            "Metropolitan area (5,000,000)": 5000000,
            "🌍 USA (330 million)": 330000000,
            "🌍 UK/France/Germany (60-80 million)": 67000000,
            "🌍 India/China (1.4 billion)": 1400000000
        }
        
        if population_preset == "Custom":
            population = st.number_input("Enter population size", 
                                        min_value=1000, max_value=2000000000, value=100000, step=1000000,
                                        help="Max: 2 billion (supports all countries including India & China)")
        else:
            population = pop_map[population_preset]
            st.info(f"Population: **{population:,}** people")
    
    with col2:
        initial_cases = st.number_input(
            "How many people are currently infected?",
            min_value=1, max_value=100000, value=10,
            help="The number of known active cases right now. "
                 "This is where the outbreak starts in the forecast."
        )
        
        existing_immunity = st.slider(
            "What % of people already have immunity?",
            min_value=0, max_value=90, value=0, step=5,
            help="From previous infection or vaccination. "
                 "Higher immunity means slower spread."
        )
        if existing_immunity > 0:
            herd_threshold = (1 - 1/spread_rate) * 100 if spread_rate > 1 else 0
            if existing_immunity > herd_threshold:
                st.success(f"✅ Above herd immunity threshold ({herd_threshold:.0f}%) - outbreak unlikely to spread widely")
            else:
                st.info(f"ℹ️ Herd immunity threshold for this disease: {herd_threshold:.0f}%")
    
    # Population consistency tip
    st.markdown(f"""
    <div class="explain-box">
    <strong>💡 Tip for Validation:</strong><br>
    If you plan to validate this forecast against real country data, consider matching the population:<br>
    • USA: ~330 million<br>
    • UK/France/Germany: ~60-83 million<br>
    • Or the system will automatically normalize data to "per 100,000 population" for fair comparison
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Step 4: Forecast Settings
    st.markdown('<span class="step-indicator">Step 4</span> **Forecast Settings**', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        forecast_days = st.slider(
            "How many days ahead to forecast?",
            min_value=14, max_value=180, value=60, step=7,
            help="Longer forecasts have more uncertainty. "
                 "CDC typically forecasts 4-8 weeks ahead."
        )
        st.caption("ℹ️ CDC's COVID forecasts typically look 4 weeks (28 days) ahead")
    
    with col2:
        confidence_level = st.select_slider(
            "How much uncertainty to show?",
            options=["Show narrow range (50%)", "Show medium range (80%)", "Show wide range (90%)"],
            value="Show wide range (90%)",
            help="Wider ranges are more likely to contain the true outcome but less precise"
        )
        ci_map = {"Show narrow range (50%)": (25, 75), 
                  "Show medium range (80%)": (10, 90), 
                  "Show wide range (90%)": (5, 95)}
        ci_lower, ci_upper = ci_map[confidence_level]
    
    st.markdown("---")
    
    # Generate Forecast Button
    if st.button("🚀 Generate Forecast", type="primary", use_container_width=True):
        with st.spinner("Running forecast simulations... This may take a moment."):
            
            # Run actual forecast
            np.random.seed(42)
            n_simulations = 500
            
            # Calculate epidemic parameters
            beta = spread_rate / days_contagious  # Transmission rate
            sigma = 1.0 / max(days_until_contagious, 0.5)  # Rate of becoming infectious
            gamma = 1.0 / days_contagious  # Recovery rate
            
            # Initialize
            initial_immune = int(existing_immunity / 100 * population)
            initial_I = initial_cases
            initial_E = initial_cases // 2  # Some exposed
            initial_S = population - initial_I - initial_E - initial_immune
            
            # Infection Fatality Rate from user slider
            ifr = fatality_rate / 100.0
            
            # Storage for all simulations (SEIRD)
            all_S = np.zeros((forecast_days, n_simulations))
            all_E = np.zeros((forecast_days, n_simulations))
            all_I = np.zeros((forecast_days, n_simulations))
            all_R = np.zeros((forecast_days, n_simulations))
            all_D = np.zeros((forecast_days, n_simulations))
            all_new_cases = np.zeros((forecast_days, n_simulations))
            all_new_deaths = np.zeros((forecast_days, n_simulations))
            
            progress_bar = st.progress(0, text="Simulating possible outbreak scenarios...")
            
            for sim in range(n_simulations):
                # Add parameter uncertainty (WIDER for better coverage in validation)
                # Using log-normal to ensure positive values and capture real-world variance
                sim_beta = beta * np.random.lognormal(0, 0.35)  # ~35% CV for transmission
                sim_sigma = sigma * np.random.lognormal(0, 0.20)  # ~20% CV for latent period
                sim_gamma = gamma * np.random.lognormal(0, 0.20)  # ~20% CV for recovery
                
                # Also add initial condition uncertainty (critical for coverage)
                init_uncertainty = np.random.uniform(0.6, 1.4)  # ±40% initial condition variance
                sim_initial_I = max(1, int(initial_I * init_uncertainty))
                sim_initial_E = max(1, int(initial_E * init_uncertainty))
                sim_initial_S = population - sim_initial_I - sim_initial_E - initial_immune
                
                S, E, I, R, D = float(sim_initial_S), float(sim_initial_E), float(sim_initial_I), float(initial_immune), 0.0
                
                for day in range(forecast_days):
                    # Stochastic SEIRD model
                    new_exposed = np.random.poisson(max(0, sim_beta * S * I / population))
                    new_exposed = min(new_exposed, S)
                    
                    new_infectious = np.random.poisson(max(0, sim_sigma * E))
                    new_infectious = min(new_infectious, E)
                    
                    new_leaving_I = np.random.poisson(max(0, sim_gamma * I))
                    new_leaving_I = min(new_leaving_I, I)
                    
                    # Split I→R and I→D using IFR
                    new_dead = int(np.random.binomial(int(new_leaving_I), ifr))
                    new_recovered = int(new_leaving_I) - new_dead
                    
                    S = S - new_exposed
                    E = E + new_exposed - new_infectious
                    I = I + new_infectious - new_leaving_I
                    R = R + new_recovered
                    D = D + new_dead
                    
                    all_S[day, sim] = S
                    all_E[day, sim] = E
                    all_I[day, sim] = I
                    all_R[day, sim] = R
                    all_D[day, sim] = D
                    all_new_cases[day, sim] = new_infectious
                    all_new_deaths[day, sim] = new_dead
                
                if (sim + 1) % 50 == 0:
                    progress_bar.progress((sim + 1) / n_simulations, 
                                         text=f"Simulating scenario {sim + 1}/{n_simulations}...")
            
            progress_bar.empty()
            
            # Store results - include all parameters needed by other pages
            st.session_state.forecast_results = {
                'S': all_S, 'E': all_E, 'I': all_I, 'R': all_R, 'D': all_D,
                'new_cases': all_new_cases,
                'new_deaths': all_new_deaths,
                'days': forecast_days,
                'population': population,
                'disease': disease_choice,
                'spread_rate': spread_rate,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'detection_rate': detection_rate,
                'ifr': ifr,
                # Additional parameters for Compare Interventions & Validate Forecast
                'initial_infected': initial_cases,
                'days_until_contagious': days_until_contagious,
                'days_contagious': days_contagious,
                'existing_immunity': existing_immunity
            }
            
            st.success("✅ Forecast complete!")
    
    # Display Results
    if "forecast_results" in st.session_state:
        results = st.session_state.forecast_results
        
        st.markdown("---")
        st.markdown("## 📊 Your Forecast Results")
        
        # Calculate statistics
        I_data = results['I']
        new_cases = results['new_cases']
        days = np.arange(1, results['days'] + 1)
        
        mean_I = np.mean(I_data, axis=1)
        lower_I = np.percentile(I_data, results['ci_lower'], axis=1)
        upper_I = np.percentile(I_data, results['ci_upper'], axis=1)
        
        mean_new = np.mean(new_cases, axis=1)
        lower_new = np.percentile(new_cases, results['ci_lower'], axis=1)
        upper_new = np.percentile(new_cases, results['ci_upper'], axis=1)
        
        # Death statistics
        D_data = results.get('D', np.zeros_like(I_data))
        new_deaths_data = results.get('new_deaths', np.zeros_like(new_cases))
        mean_D = np.mean(D_data, axis=1)
        mean_new_deaths = np.mean(new_deaths_data, axis=1)
        lower_new_deaths = np.percentile(new_deaths_data, results['ci_lower'], axis=1)
        upper_new_deaths = np.percentile(new_deaths_data, results['ci_upper'], axis=1)
        total_deaths_forecast = mean_D[-1] if len(mean_D) > 0 else 0
        ifr_display = results.get('ifr', 0) * 100
        
        # Key findings
        peak_day = np.argmax(mean_I) + 1
        peak_cases = np.max(mean_I)
        total_infected = np.max(np.mean(results['R'], axis=1))
        
        st.markdown("### 🎯 Key Findings")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "📅 Peak Expected On",
                f"Day {peak_day}",
                help="When the outbreak is likely to reach maximum cases"
            )
        
        with col2:
            st.metric(
                "📈 Peak Active Cases",
                f"{peak_cases:,.0f}",
                help="Maximum number of people infected at one time"
            )
        
        with col3:
            attack_rate = total_infected / results['population'] * 100
            st.metric(
                "👥 Total Eventually Infected",
                f"{attack_rate:.1f}%",
                help="Percentage of population that will get infected"
            )
        
        with col4:
            if peak_cases > results['population'] * 0.05:
                severity = "🔴 Severe"
            elif peak_cases > results['population'] * 0.01:
                severity = "🟡 Moderate"
            else:
                severity = "🟢 Mild"
            st.metric("Outbreak Severity", severity)
        
        with col5:
            st.metric(
                "💀 Estimated Deaths",
                f"{total_deaths_forecast:,.0f}",
                help=f"Based on IFR of {ifr_display:.1f}% for {results['disease']}"
            )
        
        # Main forecast chart
        st.markdown("### 📈 Forecast: People Currently Infected Over Time")
        
        fig = go.Figure()
        
        # Uncertainty band
        fig.add_trace(go.Scatter(
            x=np.concatenate([days, days[::-1]]),
            y=np.concatenate([upper_I, lower_I[::-1]]),
            fill='toself',
            fillcolor='rgba(231, 76, 60, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name=f'Possible range ({100 - results["ci_lower"]*2}% confidence)',
            hoverinfo='skip'
        ))
        
        # Mean line
        fig.add_trace(go.Scatter(
            x=days, y=mean_I,
            mode='lines',
            line=dict(color='#e74c3c', width=3),
            name='Most likely outcome'
        ))
        
        # Peak marker
        fig.add_vline(x=peak_day, line_dash="dash", line_color="gray",
                     annotation_text=f"Peak: Day {peak_day}")
        
        fig.update_layout(
            xaxis_title="Days from Now",
            yaxis_title="Number of People Infected",
            height=450,
            hovermode='x unified',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Explanation of results
        st.markdown(f"""
        <div class="explain-box">
        <strong>📖 What does this mean?</strong><br><br>
        Based on the characteristics of <strong>{results['disease']}</strong> with a spread rate of 
        <strong>{results['spread_rate']}</strong>:<br><br>
        
        • The outbreak will likely <strong>peak around day {peak_day}</strong> with approximately 
        <strong>{peak_cases:,.0f} people infected</strong> at the same time.<br><br>
        
        • The shaded area shows the range of possible outcomes. We're <strong>{100 - results['ci_lower']*2}% confident</strong> 
        the actual number will fall within this range.<br><br>
        
        • By the end of the outbreak, approximately <strong>{attack_rate:.1f}% of the population</strong> 
        ({total_infected:,.0f} people) will have been infected.<br><br>
        
        <strong>💡 Note:</strong> This forecast assumes no new interventions are implemented. 
        See "Compare Interventions" to explore how different measures could change these projections.
        </div>
        """, unsafe_allow_html=True)
        
        # New daily cases chart
        st.markdown("### 📊 Forecast: New Cases Per Day")
        
        fig2 = go.Figure()
        
        # Apply detection rate for "reported" cases
        reported_mean = mean_new * results['detection_rate'] / 100
        reported_lower = lower_new * results['detection_rate'] / 100
        reported_upper = upper_new * results['detection_rate'] / 100
        
        # True cases
        fig2.add_trace(go.Scatter(
            x=np.concatenate([days, days[::-1]]),
            y=np.concatenate([upper_new, lower_new[::-1]]),
            fill='toself', fillcolor='rgba(231, 76, 60, 0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            name='True cases (range)', hoverinfo='skip'
        ))
        fig2.add_trace(go.Scatter(
            x=days, y=mean_new,
            mode='lines', line=dict(color='#e74c3c', width=3),
            name='True new cases'
        ))
        
        # Reported cases
        fig2.add_trace(go.Scatter(
            x=days, y=reported_mean,
            mode='lines', line=dict(color='#3498db', width=2, dash='dash'),
            name=f'Reported cases ({results["detection_rate"]}% detected)'
        ))
        
        fig2.update_layout(
            xaxis_title="Days from Now",
            yaxis_title="New Cases Per Day",
            height=400,
            hovermode='x unified',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown(f"""
        <div class="warning-box">
        <strong>⚠️ Important: True Cases vs. Reported Cases</strong><br><br>
        The <strong style="color: #e74c3c;">red line</strong> shows the estimated <strong>true</strong> number of new infections each day.<br>
        The <strong style="color: #3498db;">blue dashed line</strong> shows what gets <strong>officially reported</strong> 
        (only {results['detection_rate']}% of true cases).<br><br>
        This is why official case counts always underestimate the true spread of disease.
        </div>
        """, unsafe_allow_html=True)
        
        # Deaths chart
        if total_deaths_forecast > 0:
            st.markdown(f"### 💀 Forecast: Deaths Over Time (IFR = {ifr_display:.1f}%)")
            
            fig3 = go.Figure()
            
            fig3.add_trace(go.Scatter(
                x=np.concatenate([days, days[::-1]]),
                y=np.concatenate([upper_new_deaths, lower_new_deaths[::-1]]),
                fill='toself', fillcolor='rgba(100, 100, 100, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Deaths (range)', hoverinfo='skip'
            ))
            fig3.add_trace(go.Scatter(
                x=days, y=mean_new_deaths,
                mode='lines', line=dict(color='#7f8c8d', width=3),
                name='Daily deaths'
            ))
            fig3.add_trace(go.Scatter(
                x=days, y=mean_D,
                mode='lines', line=dict(color='#2c3e50', width=2, dash='dash'),
                name='Cumulative deaths'
            ))
            
            fig3.update_layout(
                xaxis_title="Days from Now",
                yaxis_title="Deaths",
                height=400,
                hovermode='x unified',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            
            st.plotly_chart(fig3, use_container_width=True)
            
            st.markdown(f"""
            <div class="warning-box">
            <strong>⚠️ Mortality Projection:</strong><br><br>
            Based on the <strong>Infection Fatality Rate (IFR) of {ifr_display:.1f}%</strong> for {results['disease']}:<br>
            • Estimated total deaths: <strong>{total_deaths_forecast:,.0f}</strong><br>
            • Peak daily deaths: <strong>{np.max(mean_new_deaths):,.0f}</strong> (around day {np.argmax(mean_new_deaths)+1})<br><br>
            <em>IFR varies by disease, age, and healthcare quality. These are population-average estimates.</em>
            </div>
            """, unsafe_allow_html=True)
        
        # Download data
        st.markdown("### 💾 Download Forecast Data")
        
        df = pd.DataFrame({
            'Day': days,
            'Infected_Mean': mean_I,
            'Infected_Lower': lower_I,
            'Infected_Upper': upper_I,
            'NewCases_Mean': mean_new,
            'NewCases_Reported': reported_mean,
            'NewDeaths_Mean': mean_new_deaths,
            'CumulativeDeaths_Mean': mean_D
        })
        
        st.download_button(
            label="📥 Download as CSV",
            data=df.to_csv(index=False),
            file_name=f"outbreak_forecast_{results['disease'].replace(' ', '_')}.csv",
            mime="text/csv"
        )


# ============================================================
# COMPARE INTERVENTIONS PAGE
# ============================================================
elif page == "⚖️ Compare Interventions":
    st.markdown("# ⚖️ How Different Interventions Affect Outbreaks")
    
    st.markdown("""
    <div class="explain-box">
    <strong>What is this?</strong><br>
    This tool lets you compare what happens with and without public health interventions. 
    All intervention effectiveness values are based on real-world studies and CDC/WHO data.
    </div>
    """, unsafe_allow_html=True)
    
    # Check if forecast exists and use those parameters
    has_forecast = "forecast_results" in st.session_state
    
    if has_forecast:
        forecast = st.session_state.forecast_results
        st.markdown(f"""
        <div class="success-box">
        <strong>✅ Using Your Forecast Settings</strong><br>
        Parameters from your forecast are pre-filled: <strong>{forecast.get('disease', 'Unknown')}</strong> 
        with population of <strong>{forecast.get('population', 0):,}</strong> for <strong>{forecast.get('days', 90)} days</strong>.
        </div>
        """, unsafe_allow_html=True)
    
    # Disease selection
    st.markdown("### 1️⃣ Select the Disease")
    
    # Pre-select disease from forecast if available
    disease_list = list(DISEASE_PROFILES.keys())[:-1]  # Exclude "Custom"
    default_disease_idx = 0
    if has_forecast:
        forecast_disease = forecast.get('disease', '')
        if forecast_disease in disease_list:
            default_disease_idx = disease_list.index(forecast_disease)
    
    disease_for_intervention = st.selectbox(
        "Which disease are you modeling?",
        disease_list,
        index=default_disease_idx,
        key="intervention_disease"
    )
    
    profile = DISEASE_PROFILES[disease_for_intervention]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        # Pre-fill population from forecast
        default_pop = forecast.get('population', 500000) if has_forecast else 500000
        intervention_population = st.number_input(
            "Population size",
            min_value=10000, max_value=10000000000, 
            value=default_pop, 
            step=50000,
            key="intervention_population"
        )
    with col2:
        # Pre-fill initial cases from forecast
        default_cases = forecast.get('initial_infected', 50) if has_forecast else 50
        intervention_start_cases = st.number_input(
            "Current number of cases",
            min_value=1, max_value=100000, 
            value=min(default_cases, 100000),
            key="intervention_start_cases"
        )
    with col3:
        # Pre-fill days from forecast
        default_days = forecast.get('days', 90) if has_forecast else 90
        intervention_days = st.number_input(
            "Forecast duration (days)",
            min_value=30, max_value=365,
            value=default_days,
            key="intervention_days"
        )
    
    st.markdown("---")
    
    # Intervention selection
    st.markdown("### 2️⃣ Select Intervention(s) to Compare")
    
    intervention_choices = st.multiselect(
        "Choose one or more interventions (you can combine multiple)",
        list(INTERVENTION_EFFECTS.keys()),
        default=[list(INTERVENTION_EFFECTS.keys())[0]],
        help="Select multiple interventions to model their combined effect"
    )
    
    if not intervention_choices:
        st.warning("⚠️ Please select at least one intervention to compare")
        st.stop()
    
    # Show details for all selected interventions
    if len(intervention_choices) == 1:
        intervention = INTERVENTION_EFFECTS[intervention_choices[0]]
        st.markdown(f"""
        <div class="real-world-example">
        <strong>📋 About "{intervention_choices[0]}":</strong><br><br>
        <strong>Description:</strong> {intervention['description']}<br><br>
        <strong>Expected reduction in spread:</strong> ~{intervention['transmission_reduction']}%<br><br>
        <strong>Time to see effects:</strong> {intervention['delay_days']} days after implementation<br><br>
        <strong>Evidence:</strong> {intervention['evidence']}<br><br>
        <strong>Real-world example:</strong> {intervention['real_example']}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="real-world-example">
        <strong>📋 Selected {len(intervention_choices)} Combined Interventions:</strong><br><br>
        """, unsafe_allow_html=True)
        
        for choice in intervention_choices:
            intervention = INTERVENTION_EFFECTS[choice]
            st.markdown(f"""
            <div style="margin-left: 1rem; margin-bottom: 0.5rem;">
            <strong>• {choice}:</strong> {intervention['transmission_reduction']}% reduction (starts day {intervention['delay_days']})<br>
            <em style="font-size: 0.85rem;">{intervention['description']}</em>
            </div>
            """, unsafe_allow_html=True)
        
        # Calculate combined effect
        total_reduction = 0
        for choice in intervention_choices:
            total_reduction += INTERVENTION_EFFECTS[choice]['transmission_reduction']
        
        # Cap at 95% (can't reduce transmission by more than 95%)
        total_reduction = min(total_reduction, 95)
        
        st.markdown(f"""
        <div style="margin-top: 1rem; padding: 0.5rem; background-color: #e8f5e9; border-radius: 5px;">
        <strong>Combined Effect:</strong> ~{total_reduction}% total reduction in transmission (capped at 95%)
        </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Allow adjustment
    with st.expander("⚙️ Adjust intervention parameters"):
        if len(intervention_choices) == 1:
            adjusted_reduction = st.slider(
                "Transmission reduction (%)",
                min_value=5, max_value=90,
                value=INTERVENTION_EFFECTS[intervention_choices[0]]['transmission_reduction'],
                help="How much the intervention reduces disease spread"
            )
        else:
            # For multiple interventions, calculate combined default
            combined_default = min(sum(INTERVENTION_EFFECTS[c]['transmission_reduction'] for c in intervention_choices), 95)
            adjusted_reduction = st.slider(
                "Combined transmission reduction (%)",
                min_value=5, max_value=95,
                value=combined_default,
                help="Combined effect of all selected interventions"
            )
        
        adjusted_start_day = st.slider(
            "Start intervention(s) on day",
            min_value=1, max_value=60,
            value=14,
            help="When the intervention(s) begin"
        )
    
    if st.button("🔄 Run Comparison", type="primary", use_container_width=True):
        with st.spinner("Simulating scenarios..."):
            np.random.seed(42)
            
            # Use the intervention_days from the form (which is pre-filled from forecast)
            horizon = intervention_days
            n_sims = 300
            
            beta = profile['spread_rate'] / profile['days_contagious']
            sigma = 1.0 / max(profile['days_until_contagious'], 0.5)
            gamma = 1.0 / profile['days_contagious']
            comp_ifr = profile.get('percent_fatal', 1.0) / 100.0
            
            # Storage
            baseline_I = np.zeros((horizon, n_sims))
            intervention_I = np.zeros((horizon, n_sims))
            baseline_new_cases = np.zeros((horizon, n_sims))
            intervention_new_cases = np.zeros((horizon, n_sims))
            baseline_D = np.zeros((horizon, n_sims))
            intervention_D = np.zeros((horizon, n_sims))
            
            for sim in range(n_sims):
                # Parameter variation
                sim_beta = np.random.normal(beta, beta * 0.1)
                sim_sigma = np.random.normal(sigma, sigma * 0.1)
                sim_gamma = np.random.normal(gamma, gamma * 0.1)
                
                # Baseline scenario
                S, E, I, R, D = intervention_population - intervention_start_cases, intervention_start_cases // 2, intervention_start_cases, 0, 0
                
                for day in range(horizon):
                    new_E = np.random.poisson(max(0, sim_beta * S * I / intervention_population))
                    new_E = min(new_E, S)
                    new_I = np.random.poisson(max(0, sim_sigma * E))
                    new_I = min(new_I, E)
                    leaving_I = np.random.poisson(max(0, sim_gamma * I))
                    leaving_I = min(leaving_I, I)
                    new_dead = int(np.random.binomial(int(leaving_I), comp_ifr))
                    new_R = int(leaving_I) - new_dead
                    
                    S, E, I, R, D = S - new_E, E + new_E - new_I, I + new_I - leaving_I, R + new_R, D + new_dead
                    baseline_I[day, sim] = I
                    baseline_new_cases[day, sim] = new_I
                    baseline_D[day, sim] = D
                
                # Intervention scenario
                S, E, I, R, D = intervention_population - intervention_start_cases, intervention_start_cases // 2, intervention_start_cases, 0, 0
                
                for day in range(horizon):
                    if day >= adjusted_start_day:
                        eff_beta = sim_beta * (1 - adjusted_reduction / 100)
                    else:
                        eff_beta = sim_beta
                    
                    new_E = np.random.poisson(max(0, eff_beta * S * I / intervention_population))
                    new_E = min(new_E, S)
                    new_I = np.random.poisson(max(0, sim_sigma * E))
                    new_I = min(new_I, E)
                    leaving_I = np.random.poisson(max(0, sim_gamma * I))
                    leaving_I = min(leaving_I, I)
                    new_dead = int(np.random.binomial(int(leaving_I), comp_ifr))
                    new_R = int(leaving_I) - new_dead
                    
                    S, E, I, R, D = S - new_E, E + new_E - new_I, I + new_I - leaving_I, R + new_R, D + new_dead
                    intervention_I[day, sim] = I
                    intervention_new_cases[day, sim] = new_I
                    intervention_D[day, sim] = D
            
            st.session_state.intervention_results = {
                'baseline': baseline_I,
                'intervention': intervention_I,
                'baseline_new_cases': baseline_new_cases,
                'intervention_new_cases': intervention_new_cases,
                'baseline_D': baseline_D,
                'intervention_D': intervention_D,
                'intervention_name': ' + '.join(intervention_choices) if len(intervention_choices) > 1 else intervention_choices[0],
                'intervention_count': len(intervention_choices),
                'start_day': adjusted_start_day,
                'reduction': adjusted_reduction,
                'disease': disease_for_intervention,
                'population': intervention_population,
                'days': horizon,
                'initial_infected': intervention_start_cases,
                'ifr': comp_ifr
            }
            
            st.success("✅ Comparison complete!")
    
    # Display results
    if "intervention_results" in st.session_state:
        r = st.session_state.intervention_results
        
        st.markdown("---")
        st.markdown("## 📊 Comparison Results")
        
        days = np.arange(1, r['baseline'].shape[0] + 1)
        
        base_mean = np.mean(r['baseline'], axis=1)
        base_lower = np.percentile(r['baseline'], 10, axis=1)
        base_upper = np.percentile(r['baseline'], 90, axis=1)
        
        int_mean = np.mean(r['intervention'], axis=1)
        int_lower = np.percentile(r['intervention'], 10, axis=1)
        int_upper = np.percentile(r['intervention'], 90, axis=1)
        
        # Impact metrics
        base_peak = np.max(base_mean)
        int_peak = np.max(int_mean)
        peak_reduction = (base_peak - int_peak) / base_peak * 100
        
        base_peak_day = np.argmax(base_mean) + 1
        int_peak_day = np.argmax(int_mean) + 1
        peak_delay = int_peak_day - base_peak_day
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "📉 Peak Cases Reduced By",
                f"{peak_reduction:.0f}%",
                f"{int_peak:,.0f} vs {base_peak:,.0f}",
                delta_color="inverse"
            )
        
        with col2:
            st.metric(
                "📅 Peak Delayed By",
                f"{max(0, peak_delay)} days",
                f"Day {int_peak_day} vs Day {base_peak_day}"
            )
        
        with col3:
            baseline_deaths = np.mean(r['baseline_D'][-1, :]) if 'baseline_D' in r else 0
            intervention_deaths = np.mean(r['intervention_D'][-1, :]) if 'intervention_D' in r else 0
            lives_saved = baseline_deaths - intervention_deaths
            st.metric(
                "💚 Lives Saved",
                f"~{lives_saved:,.0f}",
                help=f"SEIRD model: {baseline_deaths:,.0f} deaths without vs {intervention_deaths:,.0f} with intervention (IFR={r.get('ifr', 0.01)*100:.1f}%)"
            )
        
        # Comparison chart
        fig = go.Figure()
        
        # Baseline
        fig.add_trace(go.Scatter(
            x=np.concatenate([days, days[::-1]]),
            y=np.concatenate([base_upper, base_lower[::-1]]),
            fill='toself', fillcolor='rgba(231, 76, 60, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='No intervention (range)', hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=days, y=base_mean,
            mode='lines', line=dict(color='#e74c3c', width=3),
            name='Without intervention'
        ))
        
        # Intervention
        fig.add_trace(go.Scatter(
            x=np.concatenate([days, days[::-1]]),
            y=np.concatenate([int_upper, int_lower[::-1]]),
            fill='toself', fillcolor='rgba(46, 204, 113, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='With intervention (range)', hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=days, y=int_mean,
            mode='lines', line=dict(color='#27ae60', width=3),
            name=f'With {r["intervention_name"]}'
        ))
        
        # Intervention start line
        fig.add_vline(x=r['start_day'], line_dash="dash", line_color="blue",
                     annotation_text="Intervention starts")
        
        fig.update_layout(
            title=f"Impact of {r['intervention_name']} on {r['disease']} Outbreak",
            xaxis_title="Days",
            yaxis_title="Number of Active Cases",
            height=500,
            hovermode='x unified',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Dynamic messaging based on intervention count
        intervention_count = r.get('intervention_count', 1)
        intervention_text = "these interventions are" if intervention_count > 1 else f"{r['intervention_name']} is"
        
        st.markdown(f"""
        <div class="success-box">
        <strong>📖 What This Means:</strong><br><br>
        If <strong>{r['intervention_name']}</strong> {'are' if intervention_count > 1 else 'is'} implemented on <strong>day {r['start_day']}</strong> 
        with <strong>{r['reduction']}% {'combined ' if intervention_count > 1 else ''}effectiveness</strong>:<br><br>
        
        • The peak of the outbreak would be reduced by approximately <strong>{peak_reduction:.0f}%</strong><br>
        • Instead of <strong>{base_peak:,.0f}</strong> people infected at peak, there would be <strong>{int_peak:,.0f}</strong><br>
        • The peak would occur <strong>{max(0, peak_delay)} days later</strong>, giving hospitals more time to prepare<br><br>
        
        <strong>💡 Key insight:</strong> Earlier intervention = better results. 
        The same intervention{'s' if intervention_count > 1 else ''} implemented earlier would have an even bigger impact.
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# VALIDATE FORECAST PAGE
# ============================================================
elif page == "✅ Validate Forecast":
    st.markdown("# ✅ Validate Your Forecast Against Real Data")
    
    st.markdown("""
    <div class="explain-box">
    <strong>What is this?</strong><br>
    This page allows you to compare your forecast predictions with actual disease outbreak data 
    from trusted sources like the CDC, WHO, and Our World in Data. This helps you understand 
    how accurate your forecast model is and learn from real-world patterns.
    </div>
    """, unsafe_allow_html=True)
    
    # Check if user has created a forecast
    has_forecast = "forecast_results" in st.session_state
    
    if has_forecast:
        forecast = st.session_state.forecast_results
        st.markdown(f"""
        <div class="success-box">
        <strong>✅ Forecast Found</strong><br>
        You created a forecast for <strong>{forecast.get('disease', 'Unknown')}</strong> 
        with <strong>{forecast.get('days', 'N/A')} days</strong> duration.<br>
        The validation will automatically match this timeframe.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="warning-box">
        <strong>⚠️ No Forecast Created Yet</strong><br>
        Go to <strong>"🔮 Forecast an Outbreak"</strong> to create a forecast first, 
        then return here to validate it against real data.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 1️⃣ Select Data Source & Disease")
    
    col1, col2 = st.columns(2)
    
    with col1:
        data_source = st.selectbox(
            "Choose a trusted data source",
            ["Johns Hopkins CSSE (COVID-19)",
             "Ebola (2014-2016 West Africa)",
             "Mpox/Monkeypox (2022 Global)",
             "Upload Your Own Data"],
            help="These are official sources used by researchers and health organizations",
            key="validation_data_source"
        )
    
    with col2:
        if "COVID" in data_source:
            validation_country = st.selectbox(
                "Select country/region",
                ["United States", "United Kingdom", "Germany", "France", "Italy", 
                 "Spain", "India", "Brazil", "Japan", "South Korea", "Australia",
                 "Canada", "Mexico", "South Africa"],
                index=0,
                key="validation_country"
            )
        elif "Ebola" in data_source:
            validation_country = st.selectbox(
                "Select country/region",
                ["Guinea", "Liberia", "Sierra Leone", "West Africa Combined"],
                index=3,
                help="WHO data from 2014-2016 West Africa Ebola epidemic",
                key="validation_country"
            )
        elif "Mpox" in data_source:
            validation_country = st.selectbox(
                "Select country/region",
                ["United States", "United Kingdom", "Spain", "Germany", "Brazil", "France", "Global Combined"],
                index=6,
                help="OWID/WHO data from 2022 global Mpox outbreak",
                key="validation_country"
            )
        else:
            validation_country = st.text_input("Enter location", "United States", key="validation_location_text")
    
    # Date range selection - auto-filled from forecast if available
    st.markdown("### 2️⃣ Select Time Period")
    
    # Disease-specific recommended dates
    if "Ebola" in data_source:
        st.markdown("""
        <div class="explain-box">
        <strong>💡 Recommended Start Dates for Ebola (2014-2016):</strong><br>
        • <strong>Mar 22, 2014</strong> - First WHO reports from Guinea<br>
        • <strong>Jun 1, 2014</strong> - Spread to Liberia & Sierra Leone<br>
        • <strong>Sep 1, 2014</strong> - Peak transmission phase<br>
        • <strong>Jan 1, 2015</strong> - Decline phase begins<br>
        Choose a date when cases were starting to rise for the most meaningful comparison!
        </div>
        """, unsafe_allow_html=True)
    elif "Mpox" in data_source:
        st.markdown("""
        <div class="explain-box">
        <strong>💡 Recommended Start Dates for Mpox/Monkeypox (2022):</strong><br>
        • <strong>May 7, 2022</strong> - First international cases detected<br>
        • <strong>Jun 1, 2022</strong> - Early exponential growth<br>
        • <strong>Jul 1, 2022</strong> - Rapid global spread<br>
        • <strong>Aug 1, 2022</strong> - Peak and declining phase<br>
        Choose a date when cases were starting to rise for the most meaningful comparison!
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="explain-box">
        <strong>💡 Recommended Start Dates for Interesting Comparisons:</strong><br>
        • <strong>Jun 1, 2020</strong> - First US wave peak<br>
        • <strong>Nov 1, 2020</strong> - Winter surge start<br>
        • <strong>Dec 1, 2021</strong> - Omicron wave start<br>
        • <strong>Jun 1, 2022</strong> - BA.5 wave<br>
        Choose a date when cases were starting to rise for the most meaningful comparison!
        </div>
        """, unsafe_allow_html=True)
    
    if has_forecast:
        st.info(f"📅 Time period will match your forecast: **{forecast.get('days', 90)} days** from selected start date")
    
    col1, col2 = st.columns(2)
    
    today = datetime.now().date()
    
    with col1:
        # Set default dates based on data source
        if "Ebola" in data_source:
            default_start = datetime(2014, 6, 1).date()
            min_date = datetime(2014, 3, 1).date()
        elif "Mpox" in data_source:
            default_start = datetime(2022, 5, 7).date()
            min_date = datetime(2022, 5, 1).date()
        else:
            default_start = datetime(2020, 6, 1).date()
            min_date = datetime(2020, 1, 1).date()
        
        start_date = st.date_input(
            "Start date (historical outbreak to compare against)",
            value=default_start,
            min_value=min_date,
            max_value=today,
            help="Select when the historical outbreak you want to compare against started",
            key="validation_start_date"
        )
    
    with col2:
        if has_forecast:
            # Auto-calculate end date based on forecast duration
            forecast_days = forecast.get('days', 90)
            auto_end_date = start_date + timedelta(days=forecast_days)
            
            # Check if calculated end date exceeds today
            if auto_end_date > today:
                # Cap at today and show warning
                capped_end_date = today
                st.warning(f"⚠️ Forecast duration ({forecast_days} days) extends beyond today. End date capped at {today.strftime('%Y-%m-%d')}. For full validation, select an earlier start date.")
            else:
                capped_end_date = auto_end_date
            
            # Display the calculated end date
            st.text_input(
                "End date (auto-calculated from forecast)",
                value=capped_end_date.strftime('%Y-%m-%d'),
                disabled=True,
                help=f"Automatically set to {forecast_days} days after start date to match your forecast",
                key="validation_end_date_display"
            )
            # Store the actual date object for later use
            end_date = capped_end_date
        else:
            # Set default end date based on data source
            if "Ebola" in data_source:
                default_end = datetime(2015, 6, 30).date()
            elif "Mpox" in data_source:
                default_end = datetime(2022, 12, 31).date()
            else:
                default_end = datetime(2023, 3, 31).date()
            
            end_date = st.date_input(
                "End date",
                value=default_end,
                min_value=min_date,
                max_value=today,
                key="validation_end_date"
            )
    
    # Fetch data button
    if data_source == "Upload Your Own Data":
        st.markdown("### 📤 Upload Your Data")
        uploaded_file = st.file_uploader(
            "Upload a CSV file with columns: 'date' and 'cases' (or 'new_cases')",
            type=['csv']
        )
        
        if uploaded_file is not None:
            try:
                user_data = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(user_data)} rows of data")
                st.session_state.validation_data = user_data
                st.session_state.validation_source = "User Upload"
            except Exception as e:
                st.error(f"Error loading file: {e}")
    
    else:
        if st.button("📥 Fetch Real-World Data", type="primary"):
            with st.spinner("Fetching data from trusted sources..."):
                try:
                    # Fetch data based on source
                    if "Johns Hopkins" in data_source:
                        # Fetch REAL Johns Hopkins COVID-19 data
                        st.info(f"📊 Fetching real COVID-19 data from Johns Hopkins for {validation_country}...")
                        
                        @st.cache_data(ttl=3600, show_spinner=False)
                        def load_jhu_covid_data(country, start, end):
                            """
                            Fetch actual COVID-19 data from Johns Hopkins GitHub repository
                            """
                            import requests
                            from io import StringIO
                            
                            # Johns Hopkins time series data URL
                            url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
                            
                            try:
                                response = requests.get(url, timeout=30)
                                response.raise_for_status()
                                
                                df = pd.read_csv(StringIO(response.text))
                                
                                # Map country names
                                country_map = {
                                    "United States": "US",
                                    "United Kingdom": "United Kingdom",
                                    "South Korea": "Korea, South",
                                    "Taiwan": "Taiwan*"
                                }
                                jhu_country = country_map.get(country, country)
                                
                                # Filter for country (aggregate all provinces/states)
                                country_df = df[df['Country/Region'] == jhu_country]
                                
                                if len(country_df) == 0:
                                    raise ValueError(f"Country '{country}' not found in Johns Hopkins data")
                                
                                # Sum across all provinces/states for the country
                                # Date columns start at index 4
                                date_cols = df.columns[4:]
                                cumulative_cases = country_df[date_cols].sum()
                                
                                # Convert to DataFrame
                                dates = pd.to_datetime(date_cols, format='%m/%d/%y')
                                result_df = pd.DataFrame({
                                    'date': dates,
                                    'total_cases': cumulative_cases.values
                                })
                                
                                # Calculate daily new cases
                                result_df['new_cases'] = result_df['total_cases'].diff().fillna(0)
                                result_df['new_cases'] = result_df['new_cases'].clip(lower=0)  # No negative cases
                                
                                # Filter by date range
                                result_df = result_df[(result_df['date'] >= pd.to_datetime(start)) & 
                                                     (result_df['date'] <= pd.to_datetime(end))]
                                
                                if len(result_df) == 0:
                                    raise ValueError(f"No data available for date range {start} to {end}")
                                
                                # Add 7-day smoothed average
                                result_df['new_cases_smoothed'] = result_df['new_cases'].rolling(7, center=True, min_periods=1).mean()
                                
                                result_df = result_df.reset_index(drop=True)
                                return result_df
                                
                            except requests.exceptions.RequestException as e:
                                raise Exception(f"Network error: {e}")
                        
                        try:
                            filtered_df = load_jhu_covid_data(validation_country, start_date, end_date)
                            
                            st.session_state.validation_data = filtered_df
                            st.session_state.validation_source = f"Johns Hopkins CSSE - {validation_country}"
                            st.session_state.validation_location = validation_country
                            
                            first_day_cases = int(filtered_df['new_cases'].iloc[0])
                            st.success(f"✅ Loaded {len(filtered_df)} days of REAL Johns Hopkins COVID-19 data")
                            st.info(f"📈 First day ({filtered_df['date'].iloc[0].strftime('%Y-%m-%d')}): **{first_day_cases:,}** new cases")
                            
                        except Exception as fetch_error:
                            st.warning(f"⚠️ Could not fetch real data: {fetch_error}")
                            st.info("🔄 Falling back to modeled data pattern...")
                            
                            # Fallback to modeled data if fetch fails
                            @st.cache_data(ttl=3600, show_spinner=False)
                            def load_modeled_covid_data(country, start, end):
                                """Generate modeled COVID-19 data as fallback"""
                                dates = pd.date_range(start=start, end=end, freq='D')
                                n_days = len(dates)
                                
                                # Use start date to vary the pattern
                                start_dt = pd.to_datetime(start)
                                seed_val = start_dt.year * 1000 + start_dt.month * 100 + start_dt.day
                                np.random.seed(seed_val)
                                
                                if country == "United States":
                                    peak_multiplier = 80000
                                elif country == "United Kingdom":
                                    peak_multiplier = 50000
                                elif country == "India":
                                    peak_multiplier = 100000
                                else:
                                    peak_multiplier = 30000
                                
                                # Generate varied initial cases based on date
                                initial_cases = int(100 + np.random.uniform(0, 500) * ((start_dt.month + start_dt.day) % 10))
                                
                                peak_day = n_days // 3
                                new_cases = np.zeros(n_days)
                                
                                for i in range(n_days):
                                    if i < peak_day:
                                        progress = i / peak_day
                                        new_cases[i] = initial_cases + (peak_multiplier - initial_cases) * (1 / (1 + np.exp(-10 * (progress - 0.5))))
                                    else:
                                        days_after_peak = i - peak_day
                                        decline = np.exp(-0.05 * days_after_peak)
                                        secondary_bump = 0.3 * np.exp(-0.01 * (days_after_peak - n_days/3)**2)
                                        new_cases[i] = peak_multiplier * (decline + secondary_bump)
                                
                                day_of_week = pd.to_datetime(dates).dayofweek
                                weekend_effect = np.where((day_of_week == 5) | (day_of_week == 6), 0.7, 1.0)
                                noise = np.random.lognormal(0, 0.15, n_days)
                                new_cases = (new_cases * noise * weekend_effect).astype(int)
                                new_cases = np.maximum(new_cases, 10)
                                
                                new_cases_smoothed = pd.Series(new_cases).rolling(7, center=True, min_periods=1).mean()
                                
                                return pd.DataFrame({
                                    'date': dates,
                                    'new_cases': new_cases,
                                    'total_cases': np.cumsum(new_cases),
                                    'new_cases_smoothed': new_cases_smoothed
                                })
                            
                            filtered_df = load_modeled_covid_data(validation_country, start_date, end_date)
                            st.session_state.validation_data = filtered_df
                            st.session_state.validation_source = f"Historical COVID-19 Pattern (Modeled) - {validation_country}"
                            st.session_state.validation_location = validation_country
                            st.success(f"✅ Generated {len(filtered_df)} days of modeled COVID-19 data")
                    
                    elif "Ebola" in data_source:
                        # Load bundled Ebola 2014-2016 data
                        st.info(f"📊 Loading Ebola outbreak data for {validation_country}...")
                        
                        @st.cache_data(show_spinner=False)
                        def load_ebola_data(location, start, end):
                            """Load pre-bundled Ebola 2014-2016 West Africa data"""
                            import os
                            csv_path = os.path.join(os.path.dirname(__file__), "data", "ebola_west_africa_2014_2016.csv")
                            df = pd.read_csv(csv_path)
                            df['date'] = pd.to_datetime(df['date'])
                            
                            # Filter by location
                            df = df[df['location'] == location].copy()
                            
                            # Filter by date range
                            df = df[(df['date'] >= pd.to_datetime(start)) & 
                                    (df['date'] <= pd.to_datetime(end))]
                            
                            # Add smoothed column
                            df['new_cases_smoothed'] = df['new_cases'].rolling(4, center=True, min_periods=1).mean()
                            
                            df = df.reset_index(drop=True)
                            return df
                        
                        filtered_df = load_ebola_data(validation_country, start_date, end_date)
                        
                        if len(filtered_df) == 0:
                            st.error(f"❌ No Ebola data found for {validation_country} in the selected date range")
                        else:
                            st.session_state.validation_data = filtered_df
                            st.session_state.validation_source = f"Ebola (WHO 2014-2016) - {validation_country}"
                            st.session_state.validation_location = validation_country
                            total_cases = int(filtered_df['total_cases'].iloc[-1])
                            total_deaths = int(filtered_df['total_deaths'].iloc[-1])
                            st.success(f"✅ Loaded {len(filtered_df)} weeks of Ebola data for {validation_country} ({total_cases:,} cases, {total_deaths:,} deaths)")
                    
                    elif "Mpox" in data_source:
                        # Load bundled Mpox 2022 data
                        st.info(f"📊 Loading Mpox outbreak data for {validation_country}...")
                        
                        @st.cache_data(show_spinner=False)
                        def load_mpox_data(location, start, end):
                            """Load pre-bundled Mpox 2022 global outbreak data"""
                            import os
                            csv_path = os.path.join(os.path.dirname(__file__), "data", "mpox_2022_global.csv")
                            df = pd.read_csv(csv_path)
                            df['date'] = pd.to_datetime(df['date'])
                            
                            # Filter by location
                            df = df[df['location'] == location].copy()
                            
                            # Filter by date range
                            df = df[(df['date'] >= pd.to_datetime(start)) & 
                                    (df['date'] <= pd.to_datetime(end))]
                            
                            # Add smoothed column
                            df['new_cases_smoothed'] = df['new_cases'].rolling(4, center=True, min_periods=1).mean()
                            
                            df = df.reset_index(drop=True)
                            return df
                        
                        filtered_df = load_mpox_data(validation_country, start_date, end_date)
                        
                        if len(filtered_df) == 0:
                            st.error(f"❌ No Mpox data found for {validation_country} in the selected date range")
                        else:
                            st.session_state.validation_data = filtered_df
                            st.session_state.validation_source = f"Mpox (OWID/WHO 2022) - {validation_country}"
                            st.session_state.validation_location = validation_country
                            total_cases = int(filtered_df['total_cases'].iloc[-1])
                            st.success(f"✅ Loaded {len(filtered_df)} weeks of Mpox data for {validation_country} ({total_cases:,} confirmed cases)")
                    
                    else:
                        st.warning("⚠️ This data source is not yet implemented. Please try Johns Hopkins, Ebola, or Mpox.")
                
                except Exception as e:
                    error_msg = str(e)
                    st.error(f"❌ Error fetching data: {error_msg}")
                    
                    # Provide specific troubleshooting based on error type
                    if "SSL" in error_msg or "ssl" in error_msg.lower():
                        st.markdown("""
                        <div class="warning-box">
                        <strong>🔒 SSL Connection Issue Detected</strong><br><br>
                        
                        This error occurs when there's a problem with secure connections. Try these solutions:<br><br>
                        
                        <strong>Option 1: Use Bundled Data</strong><br>
                        Select "Ebola" or "Mpox" data sources - these don't require external downloads.<br><br>
                        
                        <strong>Option 2: Upload Your Own Data</strong><br>
                        Select "Upload Your Own Data" and provide a CSV file with 'date' and 'cases' columns.<br><br>
                        
                        <strong>Option 3: Try a Different Network</strong><br>
                        • Switch from WiFi to mobile hotspot (or vice versa)<br>
                        • If on corporate/school network, it may have a firewall blocking access<br>
                        • Try again in a few minutes - the issue might be temporary<br><br>
                        
                        <strong>Technical Note:</strong> Your firewall or network security settings are preventing 
                        secure connections to GitHub's servers where the data is hosted.
                        </div>
                        """, unsafe_allow_html=True)
                    elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                        st.info("⏱️ **Network Timeout**: The connection is too slow. Try using Ebola/Mpox bundled data or uploading your own data instead.")
                    else:
                        st.info("💡 Try using **Ebola** or **Mpox** (bundled data) or **Upload Your Own Data** for offline validation.")
    
    # Display fetched data and validation
    if "validation_data" in st.session_state and st.session_state.validation_data is not None:
        real_data = st.session_state.validation_data
        
        # ============================================================
        # CACHE INVALIDATION: Clear forecast when data changes
        # ============================================================
        # Create a unique signature for the current data
        current_data_signature = f"{len(real_data)}_{str(real_data['date'].iloc[0])[:10]}_{str(real_data['date'].iloc[-1])[:10]}_{int(real_data['new_cases'].sum())}"
        
        # Check if we have a cached forecast and if it matches current data
        if "comparison_forecast" in st.session_state:
            cached_signature = st.session_state.comparison_forecast.get('data_signature', '')
            if cached_signature != current_data_signature:
                # Data has changed - clear the cached forecast
                del st.session_state.comparison_forecast
                st.info("📊 Date range changed - please regenerate the forecast")
        
        # Store current signature for later use
        st.session_state.current_data_signature = current_data_signature
        
        # Determine how much data to show based on forecast
        if "forecast_results" in st.session_state:
            forecast = st.session_state.forecast_results
            forecast_days = forecast.get('days', len(real_data))
            n_display = min(len(real_data), forecast_days)
            display_data = real_data.iloc[:n_display].copy()
            data_limited = True
        elif "comparison_forecast" in st.session_state:
            n_display = len(real_data)
            display_data = real_data.copy()
            data_limited = False
        else:
            n_display = len(real_data)
            display_data = real_data.copy()
            data_limited = False
        
        st.markdown("---")
        st.markdown(f"### 📊 Real-World Data: {st.session_state.validation_source}")
        
        if data_limited:
            st.info(f"📅 Showing **{n_display} days** of data to match your forecast duration. Full dataset has {len(real_data)} days.")
        
        # Show data preview
        with st.expander("📋 View Raw Data"):
            st.dataframe(display_data.head(20))
            st.caption(f"Showing first 20 of {len(display_data)} records")
        
        # Plot real data - ONLY the portion that will be compared
        fig_real = go.Figure()
        
        if 'new_cases' in display_data.columns:
            fig_real.add_trace(go.Scatter(
                x=display_data['date'],
                y=display_data['new_cases'],
                mode='lines+markers',
                name='Actual New Cases',
                line=dict(color='#e74c3c', width=2),
                marker=dict(size=4)
            ))
            
            if 'new_cases_smoothed' in display_data.columns:
                fig_real.add_trace(go.Scatter(
                    x=display_data['date'],
                    y=display_data['new_cases_smoothed'],
                    mode='lines',
                    name='7-day Average',
                    line=dict(color='#3498db', width=3)
                ))
        
        fig_real.update_layout(
            title=f"Actual Disease Cases ({n_display} days) - {st.session_state.validation_source}",
            xaxis_title="Date",
            yaxis_title="New Cases",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_real, use_container_width=True)
        
        # Key statistics from displayed data (same portion as comparison)
        st.markdown("### 📈 Real Data Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_cases = display_data['new_cases'].sum()
            st.metric("Total Cases", f"{total_cases:,.0f}")
        
        with col2:
            peak_cases = display_data['new_cases'].max()
            st.metric("Peak Daily Cases", f"{peak_cases:,.0f}")
        
        with col3:
            peak_idx = display_data['new_cases'].idxmax()
            peak_date = display_data.loc[peak_idx, 'date']
            if isinstance(peak_date, pd.Timestamp):
                peak_date = peak_date.strftime('%Y-%m-%d')
            st.metric("Peak Date", str(peak_date)[:10])
        
        with col4:
            avg_cases = display_data['new_cases'].mean()
            st.metric("Average Daily Cases", f"{avg_cases:,.0f}")
        
        st.markdown("---")
        
        # Compare with forecast
        st.markdown("### 3️⃣ Compare With Your Forecast")
        
        if "forecast_results" not in st.session_state:
            st.markdown("""
            <div class="warning-box">
            <strong>⚠️ No forecast to compare</strong><br><br>
            You haven't created a forecast yet. Go to <strong>"🔮 Forecast an Outbreak"</strong> 
            to generate predictions, then come back here to validate them against real data.
            </div>
            """, unsafe_allow_html=True)
            
            st.info("💡 **Quick Test:** You can still generate a comparison forecast below to see how validation works!")
            
            # Option to create a quick forecast for comparison
            with st.expander("🔄 Generate Quick Comparison Forecast (Advanced Fitting)"):
                st.markdown("""
                <div class="explain-box">
                <strong>🎯 Advanced Parameter Fitting</strong><br>
                This uses Maximum Likelihood Estimation (MLE) to find the best parameters that explain your real data.
                Choose between manual parameters or auto-fit from the data.
                </div>
                """, unsafe_allow_html=True)
                
                fitting_mode = st.radio(
                    "Parameter Selection Mode",
                    ["🔧 Manual (set your own)", "🎯 Auto-Fit from Data (MLE)", "📈 Time-Varying R₀ (captures interventions)"],
                    index=1,
                    help="Auto-fit estimates optimal parameters from the observed data"
                )
                
                col1, col2, col3 = st.columns(3)
                
                # Create a unique key based on data to force widget updates
                data_key = f"{len(real_data)}_{int(real_data['new_cases'].iloc[0])}_{str(real_data['date'].iloc[0])[:10]}"
                actual_initial = max(1, int(real_data['new_cases'].iloc[0]))
                
                with col1:
                    compare_population = st.number_input(
                        "Population",
                        min_value=100000, max_value=2000000000, 
                        value=330000000 if "United States" in st.session_state.get('validation_location', '') else 10000000,
                        step=1000000,
                        key=f"pop_{data_key}"
                    )
                
                if "Manual" in fitting_mode:
                    with col2:
                        compare_spread_rate = st.slider(
                            "Spread rate (R₀)",
                            min_value=0.5, max_value=10.0, value=2.5, step=0.1,
                            key=f"r0_{data_key}"
                        )
                    with col3:
                        compare_initial_cases = st.number_input(
                            "Initial infected",
                            min_value=1, max_value=1000000,
                            value=actual_initial,
                            step=100,
                            key=f"init_{data_key}"
                        )
                    st.info(f"📊 From data: First day has **{actual_initial:,}** reported cases")
                    time_varying = False
                    use_alternative_model = False
                    r0_decay = 0  # Not used in manual mode
                    
                elif "Auto-Fit" in fitting_mode:
                    # MLE-based parameter estimation
                    st.info("📊 Estimating optimal R₀ from observed case growth...")
                    
                    # Use first 14-21 days for calibration (ensure at least 3 data points)
                    calib_days = max(3, min(21, len(real_data) - 7))
                    calib_days = min(calib_days, len(real_data))  # Can't exceed data length
                    calib_data = real_data['new_cases'].iloc[:calib_days].values
                    
                    # Smooth the data
                    if 'new_cases_smoothed' in real_data.columns:
                        calib_smooth = real_data['new_cases_smoothed'].iloc[:calib_days].values
                    else:
                        calib_smooth = pd.Series(calib_data).rolling(7, min_periods=1, center=True).mean().values
                    
                    # Filter valid data points (lower threshold for diseases with small numbers)
                    max_case = np.max(calib_smooth) if len(calib_smooth) > 0 else 0
                    threshold = min(10, max(1, max_case * 0.05))  # Adaptive: 5% of peak or 10, whichever is smaller
                    valid_mask = calib_smooth > threshold
                    valid_data = calib_smooth[valid_mask]
                    valid_days = np.arange(len(calib_smooth))[valid_mask]
                    
                    if len(valid_data) >= 5:
                        # Fit exponential growth: log(cases) = log(a) + r*t
                        log_cases = np.log(valid_data + 1)
                        coeffs = np.polyfit(valid_days, log_cases, 1)
                        growth_rate = coeffs[0]  # Daily growth rate
                        initial_log = coeffs[1]
                        
                        # R₀ estimation: r ≈ (R₀ - 1) * gamma
                        gamma_est = 1/7  # Assume 7-day infectious period
                        estimated_R0 = max(0.5, min(12.0, 1 + growth_rate / gamma_est))
                        
                        # Estimate initial cases from intercept
                        estimated_initial = max(1, int(np.exp(initial_log)))
                        
                        # Calculate fit quality
                        predicted_log = coeffs[0] * valid_days + coeffs[1]
                        r_squared = 1 - np.sum((log_cases - predicted_log)**2) / np.sum((log_cases - np.mean(log_cases))**2)
                        
                        compare_spread_rate = estimated_R0
                        compare_initial_cases = estimated_initial
                        
                        with col2:
                            st.metric("Estimated R₀", f"{estimated_R0:.2f}", help="From exponential fit")
                        with col3:
                            st.metric("Fit Quality (R²)", f"{r_squared:.3f}", help="1.0 = perfect fit")
                        
                        st.success(f"✅ MLE Fit: R₀={estimated_R0:.2f}, Initial={estimated_initial:,}, Growth={growth_rate*100:.1f}%/day")
                        
                        # WARN if data doesn't match exponential growth pattern
                        if r_squared < 0.5:
                            st.warning(f"""
                            ⚠️ **Poor Exponential Fit (R² = {r_squared:.2f})**
                            
                            This data doesn't follow a simple exponential growth pattern. Possible reasons:
                            - **Multiple overlapping outbreaks** (common in national-level data)
                            - **Already past peak** (declining or flat phase)
                            - **Heavy reporting noise** (weekend effects, data dumps)
                            
                            **Recommendations:**
                            - Try a **single state** instead of national data
                            - Select a **clear growth phase** (Omicron Dec 2021 works well)
                            - Use **Time-Varying R₀ mode** if data shows decline
                            """)
                        
                        if estimated_R0 < 1.2:
                            st.warning(f"""
                            ⚠️ **Near-Zero Growth Detected (R₀ = {estimated_R0:.2f})**
                            
                            An R₀ close to 1.0 means the outbreak is **stable/flat**, not growing.
                            This often happens when:
                            - National data averages multiple regional outbreaks
                            - Selected period is **between waves** or **at plateau**
                            
                            **The SEIR model will struggle** because it expects clear growth dynamics.
                            
                            **Try these dates for cleaner data:**
                            - 🇺🇸 US Omicron: **Dec 15, 2021** to **Jan 15, 2022**
                            - 🇺🇸 US Alpha: **Nov 1, 2020** to **Dec 1, 2020**
                            - 🇬🇧 UK Delta: **Jun 1, 2021** to **Jul 1, 2021**
                            """)
                    else:
                        compare_spread_rate = 2.5
                        compare_initial_cases = actual_initial
                        st.warning(f"⚠️ Insufficient data for MLE. Using defaults (Initial: {actual_initial:,}).")
                    time_varying = False
                    use_alternative_model = False
                    r0_decay = 0
                    
                else:  # Time-varying R₀
                    st.info("📈 Using time-varying R₀ to capture intervention effects (declining transmission over time)")
                    
                    # Estimate initial R0 from early growth
                    calib_days = min(14, len(real_data) - 7)
                    calib_data = real_data['new_cases'].iloc[:calib_days].values
                    if 'new_cases_smoothed' in real_data.columns:
                        calib_smooth = real_data['new_cases_smoothed'].iloc[:calib_days].values
                    else:
                        calib_smooth = pd.Series(calib_data).rolling(7, min_periods=1, center=True).mean().values
                    
                    valid_mask = calib_smooth > 10
                    valid_data = calib_smooth[valid_mask]
                    valid_days = np.arange(calib_days)[valid_mask]
                    
                    if len(valid_data) >= 5:
                        log_cases = np.log(valid_data + 1)
                        coeffs = np.polyfit(valid_days, log_cases, 1)
                        growth_rate = coeffs[0]
                        gamma_est = 1/7
                        initial_R0 = max(1.0, min(12.0, 1 + growth_rate / gamma_est))
                    else:
                        initial_R0 = 3.0
                    
                    with col2:
                        compare_spread_rate = st.slider(
                            "Initial R₀",
                            min_value=1.0, max_value=12.0, 
                            value=float(min(12.0, max(1.0, initial_R0))),
                            step=0.1,
                            help="R₀ at the start of the outbreak",
                            key=f"tv_r0_{data_key}"
                        )
                    with col3:
                        r0_decay = st.slider(
                            "R₀ decay rate (%/week)",
                            min_value=0, max_value=50, value=15,
                            help="How fast R₀ decreases due to interventions/behavior change",
                            key=f"tv_decay_{data_key}"
                        )
                    
                    compare_initial_cases = actual_initial
                    time_varying = True
                    use_alternative_model = False
                    st.info(f"📉 R₀ will decay from {compare_spread_rate:.1f} by {r0_decay}% per week | Initial cases: **{actual_initial:,}**")
                
                # Detection rate slider for all modes
                detection_rate = st.slider(
                    "Detection/Reporting Rate (%)",
                    min_value=10, max_value=100, value=50,
                    help="What fraction of true cases get reported? Higher = model predicts fewer total cases",
                    key=f"detect_{data_key}"
                )
                
                # Regional Health Parameters Section
                st.markdown("---")
                st.markdown("### 🌍 Regional Health Parameters (GHSI)")
                
                use_health_params = st.checkbox(
                    "Apply regional health modifiers",
                    value=True,
                    help="Use Global Health Security Index data to adjust disease dynamics based on country characteristics"
                )
                
                if use_health_params:
                    # Get country from validation location
                    val_location = st.session_state.get('validation_location', 'United States')
                    
                    # Country selection with auto-detection
                    available_countries = list(GHSI_DATA.get("countries", {}).keys())
                    ghsi_countries = [c for c in available_countries if GHSI_DATA["countries"].get(c, {}).get("overall")]
                    
                    # Find best match for current validation location
                    default_idx = 0
                    for i, c in enumerate(ghsi_countries):
                        if val_location.lower() in c.lower() or c.lower() in val_location.lower():
                            default_idx = i
                            break
                    
                    selected_country = st.selectbox(
                        "Select Country for Health Parameters",
                        ghsi_countries if ghsi_countries else ["United States", "India", "United Kingdom", "Brazil", "Germany"],
                        index=default_idx,
                        key=f"country_{data_key}"
                    )
                    
                    # Load and display health parameters
                    country_health = get_country_health_params(selected_country)
                    
                    col_h1, col_h2, col_h3, col_h4 = st.columns(4)
                    with col_h1:
                        st.metric("🧹 Hygiene", f"{country_health.get('prevention', 50):.0f}/100", 
                                 help="Prevention score - affects transmission rate")
                    with col_h2:
                        st.metric("🏥 Healthcare", f"{country_health.get('health_system', 50):.0f}/100",
                                 help="Health system score - affects recovery rate")
                    with col_h3:
                        st.metric("💉 Immunity", f"{country_health.get('vaccination_rate', 60):.0f}%",
                                 help="Vaccination rate - reduces susceptible population")
                    with col_h4:
                        health_exp = country_health.get('health_expenditure_per_capita', 500)
                        st.metric("💊 Health Exp.", f"${health_exp:,.0f}",
                                 help="Per capita - higher = less affordable medication")
                    
                    st.caption(f"📊 Data source: Global Health Security Index 2021 + OWID + World Bank")
                    
                    # Disease type selection for vaccine efficacy
                    st.markdown("#### 🦠 Disease-Specific Vaccine Efficacy")
                    
                    # Auto-detect disease type from data source
                    val_source = st.session_state.get('validation_source', '')
                    disease_options = list(VACCINE_EFFICACY_DATA.keys())
                    if 'COVID' in val_source.upper() or 'Johns Hopkins' in val_source:
                        default_disease_idx = disease_options.index("COVID-19 (Omicron Variant)") if "COVID-19 (Omicron Variant)" in disease_options else 2
                    elif 'Ebola' in val_source:
                        default_disease_idx = disease_options.index("Ebola") if "Ebola" in disease_options else 6
                    elif 'Mpox' in val_source:
                        default_disease_idx = disease_options.index("Mpox (Monkeypox)") if "Mpox (Monkeypox)" in disease_options else 7
                    else:
                        default_disease_idx = disease_options.index("Custom Disease") if "Custom Disease" in disease_options else len(disease_options) - 1
                    
                    selected_disease = st.selectbox(
                        "Select Disease Type (for vaccine efficacy calculation)",
                        disease_options,
                        index=default_disease_idx,
                        help="Disease-specific vaccine efficacy affects how much vaccination reduces susceptible population",
                        key=f"disease_{data_key}"
                    )
                    
                    # Display vaccine efficacy info
                    vax_info = get_vaccine_efficacy(selected_disease)
                    vax_rate = country_health.get('vaccination_rate', 60)
                    effective_immunity = (vax_rate / 100) * vax_info['efficacy'] * 100
                    
                    col_v1, col_v2 = st.columns(2)
                    with col_v1:
                        st.metric(
                            "💉 Vaccine Efficacy (vs infection)",
                            f"{vax_info['efficacy']*100:.0f}%",
                            help=f"Base: {vax_info['base_efficacy']*100:.0f}%, Range: {vax_info['range'][0]*100:.0f}-{vax_info['range'][1]*100:.0f}%"
                        )
                    with col_v2:
                        st.metric(
                            "🛡️ Effective Population Immunity",
                            f"{effective_immunity:.1f}%",
                            help=f"Vaccination rate ({vax_rate:.0f}%) × Vaccine efficacy ({vax_info['efficacy']*100:.0f}%)"
                        )
                    
                    st.caption(f"📚 Source: {vax_info['source']}")
                    
                    # Use a simple info box instead of expander (can't nest expanders)
                    with st.popover("ℹ️ About this efficacy data"):
                        st.markdown(f"""
                        **{selected_disease}**
                        - Base efficacy against infection: **{vax_info['base_efficacy']*100:.0f}%**
                        - After 4 months waning: **{vax_info['efficacy']*100:.0f}%**
                        - Notes: {vax_info['notes']}
                        
                        *This is efficacy against INFECTION (preventing S→E transition), not against severe disease.*
                        """)
                else:
                    selected_country = None
                    country_health = None
                    selected_disease = "Custom Disease"
                
                if st.button("🚀 Generate Fitted Forecast", type="primary", key=f"gen_{data_key}"):
                    with st.spinner("Running optimized forecast simulation..."):
                        np.random.seed(42)
                        n_days = len(real_data)
                        n_sims = 500
                        
                        actual_data = real_data['new_cases'].values
                        
                        # Get disease profile if available
                        _disease_profile = DISEASE_PROFILES.get(selected_disease) if selected_disease != "Custom Disease" else None
                        
                        # ============================================================
                        # HYBRID ENSEMBLE FORECAST
                        # Renewal Equation (40%) + Trend (35%) + SEIR (25%)
                        # ============================================================
                        all_new_cases, all_infected, diag = hybrid_ensemble_forecast(
                            actual_data=actual_data,
                            n_days=n_days,
                            n_sims=n_sims,
                            population=compare_population,
                            detection_rate=detection_rate,
                            calib_fraction=0.30,
                            spread_rate=compare_spread_rate,
                            r0_decay=r0_decay if time_varying else 0,
                            time_varying=time_varying,
                            use_health_params=use_health_params,
                            country_health=country_health,
                            selected_disease=selected_disease,
                            disease_profile=_disease_profile,
                        )
                        
                        calib_days = diag['calib_days']
                        
                        st.info(f"📊 **Train/Test Split**: {calib_days} days ({30}%) calibration, {n_days - calib_days} days validation")
                        st.success(
                            f"📈 **Hybrid Ensemble**: Renewal({diag['n_renewal']}) + "
                            f"Trend({diag['n_trend']}) + SEIR({diag['n_seir']}) | "
                            f"R₀ trajectory [{diag['r_t_full'][0]:.2f} → {diag['r_t_full'][-1]:.2f}] | "
                            f"Changepoint: {'Yes' if diag['has_changepoint'] else 'No'}"
                        )
                        
                        if diag['val_corr'] != 0:
                            st.success(f"✅ **Honest Validation** (unseen {n_days - calib_days} days): "
                                      f"MAE={diag['val_mae']:,.0f}, Correlation={diag['val_corr']:.2f}")
                        
                        model_name = "Hybrid Ensemble"
                        
                        st.session_state.comparison_forecast = {
                            'new_cases': all_new_cases,
                            'infected': all_infected,
                            'dates': real_data['date'].values,
                            'days': n_days,
                            'disease': f'Fitted Forecast ({model_name})',
                            'population': compare_population,
                            'R0': compare_spread_rate,
                            'time_varying': time_varying,
                            'detection_rate': detection_rate,
                            'model_type': model_name,
                            'data_signature': st.session_state.get('current_data_signature', '')
                        }
                        st.success(f"✅ Generated {n_days}-day {model_name} forecast with {n_sims} ensemble members!")
                        st.rerun()
        
        else:
            # Use existing forecast
            forecast = st.session_state.forecast_results
            
            # Check if forecast days match real data days
            forecast_days = forecast.get('days', 90)
            real_days = len(real_data)
            
            if abs(forecast_days - real_days) > 7:  # More than 7 days difference
                st.warning(f"⚠️ **Timeframe Mismatch:** Your forecast was for {forecast_days} days, but real data has {real_days} days. Comparison will use the shorter timeframe ({min(forecast_days, real_days)} days).")
            else:
                st.success(f"✅ **Perfect Match:** Your forecast ({forecast_days} days) aligns with real data ({real_days} days)")
            
            # Check population consistency
            forecast_population = forecast.get('population', 0)
            
            # Map validation location to real population
            country_populations = {
                "United States": 330_000_000,
                "United Kingdom": 67_000_000,
                "Germany": 83_000_000,
                "France": 67_000_000,
                "Italy": 60_000_000,
                "Spain": 47_000_000,
                "India": 1_400_000_000,
                "Brazil": 215_000_000,
                "Japan": 125_000_000,
                "South Korea": 52_000_000,
                "Australia": 26_000_000,
                "Canada": 39_000_000,
                "Mexico": 130_000_000,
                "South Africa": 60_000_000,
                "World": 8_000_000_000
            }
            
            validation_location = st.session_state.get('validation_location', '')
            real_population = country_populations.get(validation_location, forecast_population)
            
            # Check for significant population mismatch
            if forecast_population > 0 and real_population > 0:
                pop_ratio = max(forecast_population, real_population) / min(forecast_population, real_population)
                
                if pop_ratio > 2.0:  # More than 2x difference
                    st.warning(f"""
                    ⚠️ **Population Mismatch Detected!**
                    
                    - **Your Forecast:** {forecast_population:,} population
                    - **Real Data ({validation_location}):** {real_population:,} population
                    - **Difference:** {pop_ratio:.1f}x
                    
                    Comparing absolute case numbers is **not meaningful** with different populations!
                    """)
                    
                    # Offer normalization option
                    normalize_data = st.checkbox(
                        "✅ **Normalize to per 100,000 population** (Recommended for fair comparison)",
                        value=True,
                        help="Convert both forecast and real data to cases per 100k population for fair comparison"
                    )
                    
                    if normalize_data:
                        st.session_state.use_normalization = True
                        st.session_state.forecast_pop = forecast_population
                        st.session_state.real_pop = real_population
                        st.info("📊 Data will be shown as **cases per 100,000 population** for fair comparison")
                    else:
                        st.session_state.use_normalization = False
                        st.warning("⚠️ Comparison will use absolute numbers. Metrics may not be meaningful.")
                else:
                    st.session_state.use_normalization = False
                    st.success(f"✅ Population sizes are similar enough for direct comparison")
            
            st.info(f"📊 Using your forecast: **{forecast.get('disease', 'Unknown Disease')}** | Population: **{forecast_population:,}** | Duration: **{forecast_days} days**")
        
        # If we have a comparison forecast, show validation results
        if "comparison_forecast" in st.session_state or "forecast_results" in st.session_state:
            
            if "comparison_forecast" in st.session_state:
                comp = st.session_state.comparison_forecast
                forecast_new_cases = comp['new_cases']
            else:
                forecast = st.session_state.forecast_results
                forecast_new_cases = forecast['new_cases']
            
            # Clean forecast data - remove any NaN/Inf values
            forecast_new_cases = np.nan_to_num(forecast_new_cases, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Align data lengths
            n_compare = min(len(real_data), forecast_new_cases.shape[0])
            
            actual_cases = real_data['new_cases'].values[:n_compare]
            actual_cases = np.nan_to_num(actual_cases, nan=0.0, posinf=0.0, neginf=0.0)
            
            forecast_mean = np.mean(forecast_new_cases[:n_compare], axis=1)
            forecast_lower = np.percentile(forecast_new_cases[:n_compare], 10, axis=1)
            forecast_upper = np.percentile(forecast_new_cases[:n_compare], 90, axis=1)
            
            # Apply normalization if enabled
            use_normalization = st.session_state.get('use_normalization', False)
            y_axis_label = "New Cases"
            
            if use_normalization:
                forecast_pop = st.session_state.get('forecast_pop', 100_000)
                real_pop = st.session_state.get('real_pop', 100_000)
                
                # Normalize to per 100k population
                actual_cases = (actual_cases / real_pop) * 100_000
                forecast_mean = (forecast_mean / forecast_pop) * 100_000
                forecast_lower = (forecast_lower / forecast_pop) * 100_000
                forecast_upper = (forecast_upper / forecast_pop) * 100_000
                
                y_axis_label = "New Cases per 100,000 Population"
            
            st.markdown("### 📉 Forecast vs Reality Comparison")
            
            # Comparison plot
            fig_compare = go.Figure()
            
            # Use actual dates from real_data instead of day numbers
            comparison_dates = real_data['date'].values[:n_compare]
            
            # Forecast uncertainty band
            fig_compare.add_trace(go.Scatter(
                x=np.concatenate([comparison_dates, comparison_dates[::-1]]),
                y=np.concatenate([forecast_upper, forecast_lower[::-1]]),
                fill='toself',
                fillcolor='rgba(52, 152, 219, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Forecast Range (80%)',
                hoverinfo='skip'
            ))
            
            # Forecast mean
            fig_compare.add_trace(go.Scatter(
                x=comparison_dates, y=forecast_mean,
                mode='lines',
                line=dict(color='#3498db', width=3),
                name='Forecast (Mean)'
            ))
            
            # Actual data - same as shown in the first graph
            fig_compare.add_trace(go.Scatter(
                x=comparison_dates, y=actual_cases,
                mode='lines+markers',
                line=dict(color='#e74c3c', width=2),
                marker=dict(size=4),
                name='Actual Cases'
            ))
            
            comparison_title = "Forecast vs Actual Cases"
            if use_normalization:
                comparison_title += " (Normalized per 100k Population)"
            
            fig_compare.update_layout(
                title=comparison_title,
                xaxis_title="Date",
                yaxis_title=y_axis_label,
                height=500,
                hovermode='x unified',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            
            st.plotly_chart(fig_compare, use_container_width=True)
            
            # Calculate validation metrics
            st.markdown("### 📊 Forecast Accuracy Metrics")
            
            if use_normalization:
                st.info("📊 **All metrics shown below are calculated on normalized data (per 100k population)**")
            
            # Clean data - remove NaN/Inf values for metrics calculation
            actual_clean = np.nan_to_num(actual_cases, nan=0.0, posinf=0.0, neginf=0.0)
            forecast_clean = np.nan_to_num(forecast_mean, nan=0.0, posinf=0.0, neginf=0.0)
            lower_clean = np.nan_to_num(forecast_lower, nan=0.0, posinf=0.0, neginf=0.0)
            upper_clean = np.nan_to_num(forecast_upper, nan=1e10, posinf=1e10, neginf=0.0)
            
            # Find valid indices (both actual and forecast > 0)
            valid_mask = (actual_clean > 0) & (forecast_clean > 0)
            
            if np.sum(valid_mask) > 5:  # Need at least 5 valid points
                actual_valid = actual_clean[valid_mask]
                forecast_valid = forecast_clean[valid_mask]
                
                # Calculate metrics on valid data only
                mae = np.mean(np.abs(actual_valid - forecast_valid))
                rmse = np.sqrt(np.mean((actual_valid - forecast_valid) ** 2))
                mape = np.mean(np.abs((actual_valid - forecast_valid) / (actual_valid + 1))) * 100
                bias = np.mean(forecast_valid - actual_valid)
                
                # Correlation with safety check
                if np.std(actual_valid) > 0 and np.std(forecast_valid) > 0:
                    correlation = np.corrcoef(actual_valid, forecast_valid)[0, 1]
                    if np.isnan(correlation):
                        correlation = 0.0
                else:
                    correlation = 0.0
            else:
                # Fallback if not enough valid data
                mae = np.mean(np.abs(actual_clean - forecast_clean))
                rmse = np.sqrt(np.mean((actual_clean - forecast_clean) ** 2))
                mape = 0.0
                bias = np.mean(forecast_clean - actual_clean)
                correlation = 0.0
            
            # Coverage - how often actual falls within forecast range (use all data)
            coverage = np.mean((actual_clean >= lower_clean) & (actual_clean <= upper_clean)) * 100
            
            # Determine units for display
            units = "cases per 100k" if use_normalization else "cases"
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Mean Absolute Error (MAE)",
                    f"{mae:,.0f} {units}",
                    help="Average difference between forecast and actual (lower is better)"
                )
                st.metric(
                    "Root Mean Square Error (RMSE)",
                    f"{rmse:,.0f} {units}",
                    help="Penalizes large errors more heavily (lower is better)"
                )
            
            with col2:
                st.metric(
                    "Coverage (80% interval)",
                    f"{coverage:.1f}%",
                    delta=f"{coverage - 80:.1f}% from ideal",
                    delta_color="off" if abs(coverage - 80) < 10 else "inverse",
                    help="Percentage of actual values within forecast range (should be ~80%)"
                )
                st.metric(
                    "Correlation",
                    f"{correlation:.2f}",
                    help="How well forecast tracks actual pattern (1.0 = perfect)"
                )
            
            with col3:
                st.metric(
                    "Forecast Bias",
                    f"{bias:+,.0f} {units}",
                    help="Positive = over-predicting, Negative = under-predicting"
                )
                st.metric(
                    "Mean Absolute % Error",
                    f"{mape:.1f}%",
                    help="Average percentage error (lower is better)"
                )
            
            # Interpretation
            if coverage >= 70 and coverage <= 90:
                coverage_status = "✅ Well-calibrated"
                coverage_color = "success"
            elif coverage < 70:
                coverage_status = "⚠️ Under-confident (too narrow)"
                coverage_color = "warning"
            else:
                coverage_status = "⚠️ Over-confident (too wide)"
                coverage_color = "warning"
            
            if correlation > 0.7:
                pattern_status = "✅ Captured pattern well"
            elif correlation > 0.4:
                pattern_status = "⚠️ Partially captured pattern"
            else:
                pattern_status = "❌ Missed the pattern"
            
            st.markdown(f"""
            <div class="explain-box">
            <strong>📖 What These Results Mean:</strong><br><br>
            
            <strong>Coverage: {coverage_status}</strong><br>
            Your 80% prediction interval contained {coverage:.1f}% of actual values. 
            Ideally this should be around 80% - if it's much lower, your forecast was too confident; 
            if much higher, it was too uncertain.<br><br>
            
            <strong>Pattern: {pattern_status}</strong><br>
            Correlation of {correlation:.2f} means your forecast 
            {"accurately tracked the ups and downs of the outbreak" if correlation > 0.7 else "partially captured the outbreak dynamics" if correlation > 0.4 else "didn't capture the outbreak pattern well"}.<br><br>
            
            <strong>Bias: {'Over-predicting' if bias > 0 else 'Under-predicting'} by {abs(bias):,.0f} {units}/day on average</strong><br>
            {"Your forecast tends to predict more cases than actually occurred." if bias > 0 else "Your forecast tends to predict fewer cases than actually occurred."}
            {f"<br><br><strong>Note:</strong> Metrics are calculated on normalized data (per 100k population) for fair comparison between different population sizes." if use_normalization else ""}
            </div>
            """, unsafe_allow_html=True)
            
            # ============================================================
            # FORECAST QUALITY SCORE & ACTIONABLE IMPROVEMENTS
            # ============================================================
            st.markdown("### 🎯 Forecast Quality Score")
            
            # Ensure all metrics are valid numbers (not NaN)
            mae = 0.0 if np.isnan(mae) else mae
            rmse = 0.0 if np.isnan(rmse) else rmse
            mape = 0.0 if np.isnan(mape) else mape
            bias = 0.0 if np.isnan(bias) else bias
            correlation = 0.0 if np.isnan(correlation) else correlation
            
            # Calculate composite quality score (0-100)
            coverage_score = max(0, 100 - abs(coverage - 80) * 2)  # Penalize deviation from 80%
            correlation_score = max(0, min(100, correlation * 100)) if correlation > 0 else 0
            actual_mean = np.mean(actual_clean[actual_clean > 0]) if np.any(actual_clean > 0) else 1
            bias_score = max(0, 100 - min(100, abs(bias) / (actual_mean + 1) * 100))
            mape_score = max(0, 100 - min(100, mape))
            
            overall_score = (coverage_score * 0.30 + correlation_score * 0.35 + 
                           bias_score * 0.20 + mape_score * 0.15)
            
            # Determine quality level
            if overall_score >= 80:
                quality_emoji = "🌟"
                quality_text = "Excellent"
                quality_color = "#27ae60"
            elif overall_score >= 60:
                quality_emoji = "✅"
                quality_text = "Good"
                quality_color = "#2ecc71"
            elif overall_score >= 40:
                quality_emoji = "⚠️"
                quality_text = "Fair"
                quality_color = "#f39c12"
            else:
                quality_emoji = "❌"
                quality_text = "Needs Improvement"
                quality_color = "#e74c3c"
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Overall Score", f"{overall_score:.0f}/100", quality_text)
            with col2:
                st.metric("Coverage Score", f"{coverage_score:.0f}", help="How well-calibrated uncertainty is")
            with col3:
                st.metric("Pattern Score", f"{correlation_score:.0f}", help="How well trend is captured")
            with col4:
                st.metric("Accuracy Score", f"{bias_score:.0f}", help="How close to actual values")
            
            st.markdown(f"""
            <div style="background-color: {quality_color}20; border-left: 4px solid {quality_color}; padding: 1rem; margin: 1rem 0; border-radius: 0 8px 8px 0;">
            <strong>{quality_emoji} {quality_text} Forecast</strong><br>
            Your forecast scored <strong>{overall_score:.0f}/100</strong>. 
            {"Great job! Your model is well-calibrated and captures the outbreak dynamics." if overall_score >= 70 else
             "Good foundation, but there's room for improvement. See suggestions below." if overall_score >= 50 else
             "The forecast needs significant improvement. Follow the suggestions below to enhance accuracy."}
            </div>
            """, unsafe_allow_html=True)
            
            # ============================================================
            # SMART IMPROVEMENT SUGGESTIONS
            # ============================================================
            st.markdown("### 💡 How to Improve Your Forecast")
            
            suggestions = []
            
            # Coverage-based suggestions
            if coverage < 50:
                suggestions.append({
                    "priority": 1,
                    "title": "🎯 Fix Narrow Uncertainty Bands",
                    "problem": f"Coverage is only {coverage:.1f}% (should be ~80%)",
                    "solution": """Your prediction intervals are too tight. Try:
• **Use the 'Generate Fitted Forecast' above** with Auto-Fit mode
• **Enable Time-Varying R₀** to capture interventions
• The updated model uses wider parameter uncertainty automatically"""
                })
            elif coverage > 95:
                suggestions.append({
                    "priority": 2,
                    "title": "📉 Tighten Uncertainty Bands",
                    "problem": f"Coverage is {coverage:.1f}% (too wide)",
                    "solution": "Your intervals are too conservative. Reduce parameter uncertainty or use MLE fitting."
                })
            
            # Correlation-based suggestions
            if correlation < 0.3:
                suggestions.append({
                    "priority": 1,
                    "title": "📊 Fix Pattern Mismatch",
                    "problem": f"Correlation is only {correlation:.2f}",
                    "solution": """The forecast shape doesn't match reality. This usually means:
• **Wrong outbreak phase** - Real data might be declining while model predicts growth
• **Use Time-Varying R₀ mode** - This captures intervention effects and behavioral changes
• **Check dates** - Ensure you're comparing the same time period
• **Try shorter horizons** - 7-14 days instead of 60+ days for better accuracy"""
                })
            elif correlation < 0.6:
                suggestions.append({
                    "priority": 2,
                    "title": "📈 Improve Pattern Capture",
                    "problem": f"Correlation is moderate ({correlation:.2f})",
                    "solution": "Try Time-Varying R₀ mode or adjust initial conditions to better match the data."
                })
            
            # Bias-based suggestions
            avg_cases = np.mean(actual_cases)
            bias_pct = abs(bias) / (avg_cases + 1) * 100
            
            if bias_pct > 50:
                if bias < 0:
                    suggestions.append({
                        "priority": 1,
                        "title": "📈 Fix Under-Prediction",
                        "problem": f"Forecast is {abs(bias):,.0f} {units}/day too low ({bias_pct:.0f}% bias)",
                        "solution": f"""Your model predicts fewer cases than observed. Try:
• **Increase R₀** - Use Auto-Fit mode to estimate from data (currently suggests R₀ based on growth)
• **Increase Detection Rate** - If set too low, reported cases will be under-predicted
• **Match Initial Cases** - Set to exactly {int(actual_cases[0]):,} (first day of real data)"""
                    })
                else:
                    suggestions.append({
                        "priority": 1,
                        "title": "📉 Fix Over-Prediction",
                        "problem": f"Forecast is {abs(bias):,.0f} {units}/day too high ({bias_pct:.0f}% bias)",
                        "solution": """Your model predicts more cases than observed. Try:
• **Use Time-Varying R₀** - Reality had interventions (lockdowns, masks) reducing spread
• **Lower R₀** - The effective R₀ was probably lower due to public health measures
• **Increase Detection Rate** - If real data is "all detected cases", use 80-100%"""
                    })
            
            # Show suggestions
            if suggestions:
                # Sort by priority
                suggestions.sort(key=lambda x: x['priority'])
                
                for i, sugg in enumerate(suggestions):
                    with st.expander(f"{'🔴' if sugg['priority'] == 1 else '🟡'} {sugg['title']}", expanded=(i == 0)):
                        st.markdown(f"**Problem:** {sugg['problem']}")
                        st.markdown(f"**Solution:**\n{sugg['solution']}")
            else:
                st.success("✅ **Your forecast looks great!** All metrics are within acceptable ranges.")
            
            # Quick action buttons
            st.markdown("### 🚀 Quick Actions")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Re-generate with Auto-Fit", use_container_width=True):
                    st.info("👆 Scroll up to the 'Generate Quick Comparison Forecast' section and select 'Auto-Fit from Data (MLE)' mode")
            with col2:
                if st.button("📈 Try Time-Varying R₀", use_container_width=True):
                    st.info("👆 Scroll up and select 'Time-Varying R₀ (captures interventions)' mode")
            
            # Residual analysis
            st.markdown("### 📈 Detailed Analysis")
            
            with st.expander("📊 Residual Analysis (Forecast Errors Over Time)"):
                residuals = actual_cases - forecast_mean
                
                fig_resid = make_subplots(rows=2, cols=1, 
                                          subplot_titles=("Forecast Errors Over Time", "Error Distribution"))
                
                # Residuals over time - use dates instead of day numbers
                fig_resid.add_trace(go.Scatter(
                    x=comparison_dates, y=residuals,
                    mode='lines+markers',
                    line=dict(color='#9b59b6'),
                    name='Residuals'
                ), row=1, col=1)
                fig_resid.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
                
                # Histogram
                fig_resid.add_trace(go.Histogram(
                    x=residuals,
                    nbinsx=20,
                    marker_color='#9b59b6',
                    name='Error Distribution'
                ), row=2, col=1)
                
                fig_resid.update_xaxes(title_text="Date", row=1, col=1)
                fig_resid.update_xaxes(title_text="Residual Value", row=2, col=1)
                fig_resid.update_yaxes(title_text="Error (Cases)", row=1, col=1)
                fig_resid.update_yaxes(title_text="Frequency", row=2, col=1)
                fig_resid.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig_resid, use_container_width=True)
                
                st.markdown("""
                <div class="explain-box">
                <strong>Reading the residuals:</strong><br>
                • Residuals should be randomly scattered around zero<br>
                • Patterns (trends, cycles) suggest systematic forecast errors<br>
                • The histogram should be roughly bell-shaped and centered at zero
                </div>
                """, unsafe_allow_html=True)
            
            with st.expander("📅 Weekly/Monthly Breakdown"):
                weekly_actual = []
                weekly_forecast = []
                
                for i in range(0, n_compare, 7):
                    end_i = min(i + 7, n_compare)
                    weekly_actual.append(np.sum(actual_cases[i:end_i]))
                    weekly_forecast.append(np.sum(forecast_mean[i:end_i]))
                
                weekly_df = pd.DataFrame({
                    'Week': range(1, len(weekly_actual) + 1),
                    'Actual': weekly_actual,
                    'Forecast': weekly_forecast,
                    'Error': np.array(weekly_forecast) - np.array(weekly_actual),
                    'Error %': (np.array(weekly_forecast) - np.array(weekly_actual)) / (np.array(weekly_actual) + 1) * 100
                })
                
                st.dataframe(weekly_df.style.format({
                    'Actual': '{:,.0f}',
                    'Forecast': '{:,.0f}',
                    'Error': '{:+,.0f}',
                    'Error %': '{:+.1f}%'
                }))
            
            # Download validation report
            st.markdown("### 💾 Download Validation Report")
            
            # Compute projected deaths from forecast
            if "comparison_forecast" in st.session_state:
                val_disease = comp.get('disease', '')
            else:
                val_disease = forecast.get('disease', '') if 'forecast' in dir() else st.session_state.get('forecast_results', {}).get('disease', '')
            # Try to find the disease in profiles for IFR
            val_ifr = 0.01  # default 1%
            for dname, dprof in DISEASE_PROFILES.items():
                if dname.lower() in val_disease.lower() or val_disease.lower() in dname.lower():
                    val_ifr = dprof.get('percent_fatal', 1.0) / 100.0
                    break
            
            forecast_deaths_mean = forecast_mean * val_ifr
            total_projected_deaths = np.sum(forecast_deaths_mean)
            actual_total_cases = np.sum(actual_cases)
            
            if val_ifr > 0:
                st.markdown("### 💀 Projected Mortality Estimates")
                col_d1, col_d2, col_d3 = st.columns(3)
                with col_d1:
                    st.metric(
                        "Projected Deaths (from forecast)",
                        f"{total_projected_deaths:,.0f}",
                        help=f"Forecast cases × IFR ({val_ifr*100:.1f}%)"
                    )
                with col_d2:
                    actual_deaths_est = actual_total_cases * val_ifr
                    st.metric(
                        "Estimated Deaths (from real data)",
                        f"{actual_deaths_est:,.0f}",
                        help=f"Actual cases × IFR ({val_ifr*100:.1f}%)"
                    )
                with col_d3:
                    st.metric(
                        "IFR Used",
                        f"{val_ifr*100:.1f}%",
                        help="Infection Fatality Rate from disease profile"
                    )
                st.caption(f"⚠️ Death estimates are approximate, derived by applying the disease IFR ({val_ifr*100:.1f}%) to case counts.")
            
            
            report_df = pd.DataFrame({
                'Date': comparison_dates,
                'Day': np.arange(1, n_compare + 1),
                'Actual_Cases': actual_cases,
                'Forecast_Mean': forecast_mean,
                'Forecast_Lower': forecast_lower,
                'Forecast_Upper': forecast_upper,
                'Error': actual_cases - forecast_mean,
                'Within_Range': (actual_cases >= forecast_lower) & (actual_cases <= forecast_upper)
            })
            
            st.download_button(
                label="📥 Download Validation Data (CSV)",
                data=report_df.to_csv(index=False),
                file_name="forecast_validation_report.csv",
                mime="text/csv"
            )


# ============================================================
# UNDERSTANDING RESULTS PAGE
# ============================================================
elif page == "📊 Understanding Results":
    st.markdown("# 📊 How to Understand Forecast Results")
    
    st.markdown("""
    <div class="explain-box">
    This page explains how to read and interpret the forecasts, 
    using the same metrics that public health officials use.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("## 📈 Reading the Forecast Charts")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Demo chart
        np.random.seed(42)
        days = np.arange(1, 61)
        mean_curve = 5000 * np.exp(-0.5 * ((days - 30) / 12) ** 2)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=np.concatenate([days, days[::-1]]),
            y=np.concatenate([mean_curve * 1.6, (mean_curve * 0.5)[::-1]]),
            fill='toself', fillcolor='rgba(231, 76, 60, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Uncertainty range (90% confidence)'
        ))
        
        fig.add_trace(go.Scatter(
            x=days, y=mean_curve,
            mode='lines', line=dict(color='#e74c3c', width=3),
            name='Most likely outcome'
        ))
        
        # Annotations
        fig.add_annotation(x=30, y=5000, text="Peak", showarrow=True, arrowhead=2)
        fig.add_annotation(x=10, y=1000, text="Growth phase", showarrow=False)
        fig.add_annotation(x=50, y=1000, text="Decline phase", showarrow=False)
        
        fig.update_layout(height=350, title="Example Forecast Chart")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="explain-box">
        <strong>How to read this:</strong><br><br>
        
        🔴 <strong>Red line</strong> = Most likely outcome<br><br>
        
        🟡 <strong>Shaded area</strong> = Range of possibilities<br><br>
        
        The wider the shaded area, the more uncertain the forecast is.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("## 🔢 Key Numbers Explained")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### Peak Day
        
        <div class="explain-box">
        <strong>What it is:</strong><br>
        The day when the most people will be sick at once.<br><br>
        
        <strong>Why it matters:</strong><br>
        Hospitals need to prepare for this day. If peak cases exceed hospital capacity, people may not get proper care.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        ### Peak Cases
        
        <div class="explain-box">
        <strong>What it is:</strong><br>
        The maximum number of people infected at the same time.<br><br>
        
        <strong>Why it matters:</strong><br>
        This determines how overwhelmed healthcare systems will be. "Flattening the curve" means reducing this number.
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        ### Attack Rate
        
        <div class="explain-box">
        <strong>What it is:</strong><br>
        The total percentage of people who will eventually get infected.<br><br>
        
        <strong>Why it matters:</strong><br>
        Higher attack rates mean more total cases, more hospitalizations, and more deaths over the entire outbreak.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("## 📊 What Health Officials Look For")
    
    st.markdown("""
    When public health officials (like at the CDC) review forecasts, they focus on:
    """)
    
    criteria = [
        ("Hospital Capacity", "Will peak cases exceed available hospital beds?", 
         "If projected cases > beds, need interventions or surge capacity"),
        ("Timing", "When should we act?", 
         "Earlier intervention = better outcomes, but every day of restriction has costs"),
        ("Uncertainty", "How confident are we?",
         "Wide uncertainty bands mean we should prepare for worse-case scenarios"),
        ("Trend Direction", "Is it getting better or worse?",
         "Rising trends need action; declining trends may allow loosening restrictions")
    ]
    
    for title, question, answer in criteria:
        with st.expander(f"**{title}**: {question}"):
            st.markdown(f"""
            <div class="explain-box">
            <strong>What they look for:</strong><br>
            {answer}
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("## ⚠️ Limitations to Keep in Mind")
    
    st.markdown("""
    <div class="warning-box">
    <strong>Important caveats about any disease forecast:</strong><br><br>
    
    1. <strong>Forecasts are not predictions</strong> - They show what COULD happen based on current trends, 
       not what WILL happen.<br><br>
    
    2. <strong>Human behavior changes</strong> - If people hear about an outbreak, they may change behavior, 
       which changes the outbreak trajectory.<br><br>
    
    3. <strong>Data is imperfect</strong> - We never know the true number of cases. 
       Many infections go undetected.<br><br>
    
    4. <strong>Models simplify reality</strong> - Real epidemics involve complex human networks, 
       but models use simplified assumptions.<br><br>
    
    5. <strong>Longer forecasts are less reliable</strong> - Like weather, disease forecasts become 
       less accurate the further ahead you look.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# AGENT SIMULATION PAGE
# ============================================================
elif page == "🎮 Agent Simulation":
    from simulation import (
        run_agent_simulation,
        build_target_curve_from_forecast,
        build_target_curve_from_validation,
        build_animation_figure,
        build_seir_curves_figure,
        compute_statistics,
    )

    st.markdown('<h1 class="main-header">🎮 Agent-Based Disease Simulation</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Watch disease spread through a virtual population - Artificial Life in action!</p>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="explain-box">
    <strong>🧬 What is Agent-Based Simulation?</strong><br><br>
    This visualization shows <strong>individual agents (people)</strong> moving in a virtual space.
    Each dot represents a person, and colors show their disease state:
    <ul>
    <li>🟢 <strong>Green</strong> = Susceptible (can catch the disease)</li>
    <li>🟡 <strong>Yellow</strong> = Exposed (infected but not yet contagious)</li>
    <li>🔴 <strong>Red</strong> = Infectious (can spread the disease)</li>
    <li>🔵 <strong>Blue</strong> = Recovered (immune)</li>
    </ul>
    <strong>Two modes available:</strong>
    <ul>
    <li><strong>Forecast / Validation guided</strong> – agents follow the epidemic curve produced by the ensemble forecast (curve-guided ABM with proportional controller)</li>
    <li><strong>Custom</strong> – pure emergent simulation driven only by R₀ and disease timers</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ── data-source picker ──────────────────────────────────────────
    has_forecast   = "forecast_results" in st.session_state
    has_validation = ("validation_data" in st.session_state
                      and st.session_state.validation_data is not None)
    has_comparison = "comparison_forecast" in st.session_state
    
    data_source_option = "custom"
    
    if has_forecast or has_validation or has_comparison:
        st.success("✅ Data found! Choose a data source for the simulation:")
        
        options = ["🔧 Custom Parameters"]
        if has_forecast:
            options.append("🔮 Forecast Data")
        if has_comparison:
            options.append("📈 Validation Forecast (Hybrid Ensemble)")
        if has_validation:
            val_src = st.session_state.get('validation_source', 'Real Data')
            options.append(f"📊 Validation Real Data ({val_src})")
        
        if has_forecast:
            default_idx = 1
        elif has_comparison:
            default_idx = 1
        elif has_validation:
            default_idx = 1
        else:
            default_idx = 0
        
        data_source_choice = st.radio(
            "Select data source:",
            options,
            index=default_idx,
            horizontal=True
        )
        
        if "Forecast Data" in data_source_choice:
            data_source_option = "forecast"
        elif "Validation Forecast" in data_source_choice:
            data_source_option = "comparison"
        elif "Validation Real" in data_source_choice:
            data_source_option = "validation"
        else:
            data_source_option = "custom"
    else:
        st.warning("⚠️ No data found! Please go to '🔮 Forecast an Outbreak' or '✅ Validate Forecast' first, or use custom parameters below.")
    
    # ── simulation settings ─────────────────────────────────────────
    st.markdown("### ⚙️ Simulation Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_agents = st.slider(
            "Number of Agents",
            min_value=50, max_value=500, value=200,
            help="More agents = more realistic but slower animation"
        )
    
    with col2:
        if data_source_option == "forecast":
            sim_days = st.session_state.forecast_results.get('days', 30)
            st.metric("Simulation Days", sim_days)
        elif data_source_option == "comparison":
            sim_days = st.session_state.comparison_forecast.get('days', 30)
            st.metric("Simulation Days", sim_days)
        elif data_source_option == "validation":
            sim_days = len(st.session_state.validation_data)
            st.metric("Simulation Days", sim_days)
        else:
            sim_days = st.slider(
                "Simulation Days",
                min_value=10, max_value=365, value=30
            )
    
    with col3:
        animation_speed = st.slider(
            "Animation Speed (ms/frame)",
            min_value=50, max_value=500, value=150,
            help="Lower = faster animation"
        )
    
    # ── disease parameters & target curve ───────────────────────────
    target_curve = None    # None → pure emergent mode
    
    if data_source_option == "forecast":
        forecast = st.session_state.forecast_results
        sim_R0         = forecast.get('R0', forecast.get('spread_rate', 2.5))
        sim_incubation = forecast.get('days_until_contagious', 3)
        sim_infectious = forecast.get('days_contagious', 7)
        sim_ifr        = forecast.get('ifr', 0.01)
        target_curve   = build_target_curve_from_forecast(forecast, n_agents)
        st.info(f"📊 **Curve-Guided Mode** — agents track the forecast epidemic curve\n\n"
                f"R₀ = {sim_R0:.1f} · Duration = {sim_days} days · "
                f"Disease = {forecast.get('disease', 'N/A')}")
    
    elif data_source_option == "comparison":
        comp = st.session_state.comparison_forecast
        sim_R0         = comp.get('R0', 2.5)
        sim_incubation = 3
        sim_infectious = 7
        sim_ifr        = comp.get('ifr', 0.01)
        target_curve   = build_target_curve_from_forecast(comp, n_agents)
        st.info(f"📊 **Curve-Guided Mode** — agents track the validation ensemble curve\n\n"
                f"R₀ = {sim_R0:.1f} · Duration = {sim_days} days · "
                f"Model = {comp.get('model_type', 'Hybrid Ensemble')}")
    
    elif data_source_option == "validation":
        val_data = st.session_state.validation_data
        # estimate R₀ from early growth
        try:
            cd = max(3, min(21, len(val_data) - 7))
            cd = min(cd, len(val_data))
            cs = pd.Series(val_data['new_cases'].iloc[:cd].values).rolling(
                7, min_periods=1, center=True).mean().values
            mx = np.max(cs) if len(cs) > 0 else 0
            thr = min(10, max(1, mx * 0.05))
            vm = cs > thr
            vd = cs[vm]
            vi = np.arange(len(cs))[vm]
            if len(vd) >= 5:
                lc = np.log(vd + 1)
                cf = np.polyfit(vi, lc, 1)
                sim_R0 = float(np.clip(1 + cf[0] / (1/7), 0.8, 8.0))
            else:
                sim_R0 = 2.5
        except Exception:
            sim_R0 = 2.5
        sim_incubation = 3
        sim_infectious = 7
        sim_ifr        = 0.01
        target_curve   = build_target_curve_from_validation(val_data, n_agents)
        val_src = st.session_state.get('validation_source', 'Real Data')
        st.info(f"📊 **Curve-Guided Mode** — agents track the real-world case curve\n\n"
                f"Estimated R₀ = {sim_R0:.2f} · Duration = {sim_days} days · "
                f"Source = {val_src}")
    
    else:
        st.markdown("### 🦠 Disease Parameters")
        col1, col2, col3 = st.columns(3)
        with col1:
            sim_R0 = st.slider("R₀ (Spread Rate)", 1.0, 6.0, 2.5, 0.1)
        with col2:
            sim_incubation = st.slider("Incubation Period (days)", 1, 10, 3)
        with col3:
            sim_infectious = st.slider("Infectious Period (days)", 3, 21, 7)
        sim_ifr = st.slider("Infection Fatality Rate (%)", 0.0, 60.0, 1.0, 0.1) / 100.0
        st.info("🔧 **Custom Mode** — pure emergent ABM, no target curve")
    
    # ── run simulation ──────────────────────────────────────────────
    if st.button("🚀 Generate Agent Simulation", type="primary"):
        with st.spinner("Creating agent-based simulation... This may take a moment."):
            frames_data = run_agent_simulation(
                n_agents=n_agents,
                sim_days=sim_days,
                sim_R0=sim_R0,
                sim_incubation=sim_incubation,
                sim_infectious=sim_infectious,
                target_curve=target_curve,
                ifr=sim_ifr,
                seed=42,
            )
            
            # ── animated scatter ────────────────────────────────────
            fig_anim = build_animation_figure(frames_data, animation_speed)
            st.plotly_chart(fig_anim, use_container_width=True)
            
            # ── SEIR curves ────────────────────────────────────────
            st.markdown("### 📈 Disease Spread Over Time (from simulation)")
            fig_curves = build_seir_curves_figure(frames_data)
            st.plotly_chart(fig_curves, use_container_width=True)
            
            # ── statistics ──────────────────────────────────────────
            st.markdown("### 📊 Simulation Statistics")
            stats = compute_statistics(frames_data, n_agents)
            
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col1:
                st.metric("Peak Infections", stats['peak_infected'])
            with col2:
                st.metric("Peak Day", f"Day {stats['peak_day']}")
            with col3:
                st.metric("Total Infected", stats['total_infected'])
            with col4:
                st.metric("Attack Rate", f"{stats['attack_rate']:.1f}%")
            with col5:
                st.metric("💀 Total Deaths", stats['total_deaths'])
            with col6:
                st.metric("Mortality Rate", f"{stats['mortality_rate']:.1f}%")
            
            st.success("✅ Simulation complete! Use the Play button to watch the disease spread through the population.")
            
            mode_label = "Curve-Guided" if target_curve is not None else "Custom (Emergent)"
            st.markdown(f"""
            <div class="explain-box">
            <strong>🔬 What You're Seeing ({mode_label} Mode):</strong><br><br>
            This is an <strong>agent-based model</strong> — a core technique in Artificial Life simulation.
            Each agent follows simple rules:
            <ul>
            <li>Move randomly in space (Brownian-style motion with wall-bounce)</li>
            <li>If infectious and near a susceptible agent → may transmit disease</li>
            <li>Progress through S → E → I → R states based on timers</li>
            </ul>
            {"<strong>Curve guidance:</strong> A proportional controller adjusts the transmission probability each day so the aggregate new-infection count tracks the forecast/validation case curve. This means the micro-level agent interactions produce aggregate behavior that follows the macro-level forecast." if target_curve is not None else "<strong>Emergent behavior:</strong> The epidemic curve emerges naturally from these simple rules — no one tells the simulation to create that characteristic peak-and-decline shape!"}
            </div>
            """, unsafe_allow_html=True)


# ============================================================
# FAQ & HELP PAGE - CHATBOT
# ============================================================
elif page == "❓ FAQ & Help":
    st.markdown("# 🤖 FAQ Chatbot")
    st.markdown('<p class="sub-header">Ask me anything about disease forecasting!</p>', unsafe_allow_html=True)
    
    # ── Knowledge Base ──────────────────────────────────────────────
    FAQ_KNOWLEDGE = {
        "parameters": {
            "keywords": ["parameter", "disease", "data", "source", "where", "come from", "cdc", "who", "r0", "spread rate", "values", "default"],
            "question": "Where do the disease parameters come from?",
            "answer": """All default disease parameters are based on **peer-reviewed scientific literature** and official reports from health organizations:

- **CDC** (Centers for Disease Control and Prevention)
- **WHO** (World Health Organization)  
- **ECDC** (European Centre for Disease Prevention and Control)

**Examples:**
- COVID-19 parameters → CDC COVID Data Tracker & WHO situation reports
- Influenza parameters → CDC seasonal flu surveillance data
- Measles parameters → WHO measles fact sheets
- Ebola parameters → WHO Ebola Response Team publications"""
        },
        "uncertainty": {
            "keywords": ["range", "uncertainty", "confidence", "band", "interval", "why range", "not exact", "shaded", "wide"],
            "question": "Why do forecasts show a range instead of exact numbers?",
            "answer": """Disease spread involves **many random factors**:

- Who meets whom on any given day
- Whether a particular encounter leads to transmission
- Individual variation in how contagious people are
- Testing and reporting variability

By running **500 Monte Carlo simulations**, we capture this randomness and show you the **range of possible outcomes**, not just one guess.

The shaded region represents the **95% confidence interval** — we expect the true value to fall within this range 95% of the time."""
        },
        "accuracy": {
            "keywords": ["accurate", "accuracy", "reliable", "trust", "how good", "correct", "wrong", "error", "validation"],
            "question": "How accurate are these forecasts?",
            "answer": """No forecast is perfectly accurate — just like weather forecasts! However, these models use the **same mathematical frameworks** that professional epidemiologists use.

**Accuracy depends on:**
- Quality of input parameters
- How far ahead you're forecasting (shorter = better)
- Whether conditions change (new variants, interventions, behavior)

**Our validation page** lets you test the model against real historical data and see metrics like MAE, RMSE, and MAPE.

**Rule of thumb:** Short-term forecasts (1-2 weeks) are typically more reliable than longer-term ones."""
        },
        "r0": {
            "keywords": ["r0", "r-naught", "r naught", "spread rate", "reproduction", "basic reproduction", "what is r", "meaning of r"],
            "question": "What does R₀ (spread rate) mean?",
            "answer": """The spread rate (**R₀** or "R-naught") answers one simple question:

> *"If one person gets infected in a population where no one is immune, how many people will they spread it to?"*

**Interpretation:**
- **R₀ = 2** → each person infects 2 others → outbreak grows exponentially
- **R₀ = 1** → each person infects 1 other → outbreak stays stable  
- **R₀ < 1** → each person infects less than 1 → outbreak dies out

**Examples:**
- Measles: R₀ ≈ 12-18 (extremely contagious)
- COVID-19 (original): R₀ ≈ 2.5-3.5
- Seasonal flu: R₀ ≈ 1.2-1.4

This is why interventions aim to **reduce R below 1**."""
        },
        "underreporting": {
            "keywords": ["reported", "cases", "true", "actual", "real", "undercount", "underreport", "detection", "testing", "why lower"],
            "question": "Why are reported cases always lower than true cases?",
            "answer": """Several reasons for **underreporting**:

1. **Mild/no symptoms** — Many people don't feel sick enough to get tested
2. **Limited testing** — Not everyone who wants a test can get one
3. **Reporting delays** — Lag between testing and official recording
4. **Test sensitivity** — Tests don't catch 100% of true cases

**Real-world example:**
For COVID-19, CDC estimates the true number of infections was **2-4x higher** than reported cases.

Our model includes a **detection rate slider** to account for this!"""
        },
        "interventions": {
            "keywords": ["intervention", "mask", "lockdown", "vaccine", "social distance", "reduce", "prevention", "control", "how work"],
            "question": "How do interventions work in the model?",
            "answer": """Interventions reduce the **effective spread rate**:

- **Masks** → Block respiratory droplets → Lower transmission
- **Social distancing** → Fewer contact opportunities → Lower transmission  
- **Lockdowns** → Drastically reduce contacts → ~70% reduction (based on Wuhan/Italy studies)
- **Vaccination** → Makes people immune → Fewer susceptible people

The **Compare Interventions** page lets you stack multiple interventions and see the combined effect on the epidemic curve.

Intervention effectiveness percentages come from **systematic reviews and meta-analyses** of real-world data."""
        },
        "decision": {
            "keywords": ["real", "decision", "use", "actual", "policy", "government", "official", "educational", "purpose"],
            "question": "Can I use this for real decision-making?",
            "answer": """This tool is for **educational purposes** and general understanding.

For actual public health decisions, officials use:
- More detailed data (age structure, geographic spread, hospital capacity)
- **Multiple models** compared in ensemble forecasts
- Expert judgment and local context
- Real-time updating as new data arrives

Think of this as a **simplified teaching version** that demonstrates the same concepts used by professionals.

For real forecasts, see the **CDC COVID-19 Forecast Hub** which combines 50+ models."""
        },
        "seir": {
            "keywords": ["seir", "exposed", "infectious", "infected", "contagious", "recovered", "susceptible", "compartment", "difference", "state"],
            "question": "What's the difference between 'exposed' and 'infectious'?",
            "answer": """The **SEIR model** tracks disease stages:

1. **S (Susceptible)** — Can catch the disease
2. **E (Exposed)** — Infected but virus is still multiplying. **Cannot spread it yet.**
3. **I (Infectious)** — Virus has multiplied enough. **Can now spread to others.**
4. **R (Recovered)** — Immune system has cleared infection. Cannot catch it again (for now).

**Important:** For many diseases (including COVID-19), you can be **contagious BEFORE you have symptoms**! This is why asymptomatic spread is so dangerous.

The **Agent Simulation** page visualizes these states as colored dots:
🟢 Susceptible → 🟡 Exposed → 🔴 Infectious → 🔵 Recovered"""
        },
        "ensemble": {
            "keywords": ["ensemble", "hybrid", "model", "algorithm", "renewal", "arima", "trend", "how forecast", "method", "technique"],
            "question": "How does the hybrid ensemble forecasting work?",
            "answer": """Our **Hybrid Ensemble** combines 3 different models:

| Model | Weight | What it does |
|-------|--------|--------------|
| 🔬 **Renewal Equation** | 40% | Estimates time-varying R(t) from case data |
| 📈 **Trend Extrapolation** | 35% | Statistical pattern matching (ARIMA-style) |
| 🧬 **SEIR Model** | 25% | Classic epidemiological compartments |

**Why combine them?**
- No single model is best for all situations
- Renewal captures R changes; Trend captures patterns; SEIR adds biological constraints
- Weighted blending produces more robust forecasts than any single model

We run **500 Monte Carlo simulations** with parameter uncertainty to generate confidence intervals."""
        },
        "agent": {
            "keywords": ["agent", "simulation", "abm", "artificial life", "dots", "animation", "visualize", "watch", "spread"],
            "question": "What is the Agent-Based Simulation?",
            "answer": """The **Agent-Based Model (ABM)** is an **Artificial Life** visualization:

- Each colored dot is a virtual "person" (agent)
- Agents move randomly in a 2D space
- When an infectious agent gets near a susceptible one → possible transmission
- Agents progress through **S → E → I → R** states

**Two modes:**
1. **Curve-Guided** — Agents follow the forecast epidemic curve (proportional controller adjusts transmission daily)
2. **Custom/Emergent** — Pure bottom-up simulation; epidemic curve emerges naturally from agent interactions

**Colors:**
🟢 Susceptible · 🟡 Exposed · 🔴 Infectious · 🔵 Recovered

This makes abstract forecasts **tangible and intuitive**!"""
        },
        "data": {
            "keywords": ["data", "source", "johns hopkins", "jhu", "ebola", "mpox", "covid", "upload", "csv", "where get"],
            "question": "What data sources are available?",
            "answer": """The app supports multiple data sources:

| Source | Disease | Coverage |
|--------|---------|----------|
| **Johns Hopkins CSSE** | COVID-19 | Global, 2020-2023 |
| **Bundled CSV** | Ebola | West Africa 2014-2016 |
| **Bundled CSV** | Mpox/Monkeypox | Global 2022 |
| **Upload** | Any disease | Your own CSV file |

**For uploads**, your CSV needs at minimum:
- A `date` column
- A `new_cases` or `cases` column

The validation page automatically handles different date formats and normalizes data for comparison."""
        },
        "deaths": {
            "keywords": ["death", "deaths", "dead", "die", "dying", "fatal", "fatality", "ifr", "mortality", "seird", "killed", "percent_fatal"],
            "question": "How does the app model deaths?",
            "answer": """💀 **The app uses a SEIRD model** — adding a **D (Dead)** compartment to the classic SEIR framework.

**How it works:**
- When an infectious agent's recovery timer expires, the model rolls a random check against the **Infection Fatality Rate (IFR)**
- With probability **IFR** → the individual dies (state D)
- With probability **1 − IFR** → the individual recovers (state R)

**IFR values by disease:**
- COVID-19 Original: **1.0%**
- COVID-19 Delta: **1.5%**
- COVID-19 Omicron: **0.3%**
- Seasonal Flu: **0.1%**
- Pandemic Flu: **2.5%**
- Measles: **0.2%**
- Ebola: **50%**
- Mpox: **0.1%**

**Where you'll see deaths:**
- 📈 **Forecast page** — death chart with daily & cumulative deaths
- ✅ **Validation page** — projected mortality estimates
- ⚖️ **Compare Interventions** — lives saved calculation
- 🎮 **Agent Simulation** — dead agents shown as ⚫ black dots that stop moving"""
        },
        "hello": {
            "keywords": ["hello", "hi", "hey", "greetings", "help", "start", "what can you"],
            "question": "Hello!",
            "answer": """👋 **Hello! I'm the FAQ Chatbot for the Disease Forecasting app.**

I can answer questions about:
- 🦠 **Disease parameters** (R₀, incubation, infectious period)
- 📊 **Forecasting methods** (SEIR, ensemble, uncertainty)
- 🎯 **Validation & accuracy** (metrics, confidence intervals)
- 💉 **Interventions** (vaccines, masks, lockdowns)
- 🎮 **Agent simulation** (Artificial Life visualization)
- 📁 **Data sources** (Johns Hopkins, Ebola, Mpox)
- 💀 **Deaths & mortality** (SEIRD model, IFR)

**Try asking:**
- "What is R₀?"
- "How accurate are the forecasts?"
- "How does the ensemble work?"
- "What do the colors mean in the simulation?"
- "How does the app model deaths?" """
        }
    }
    
    # ── Chatbot Matching Function ───────────────────────────────────
    def find_best_answer(user_query):
        """Find the best matching FAQ answer using keyword matching."""
        query_lower = user_query.lower()
        
        best_match = None
        best_score = 0
        
        for topic, data in FAQ_KNOWLEDGE.items():
            score = 0
            for keyword in data["keywords"]:
                if keyword in query_lower:
                    # Longer keywords get higher scores
                    score += len(keyword.split())
            
            if score > best_score:
                best_score = score
                best_match = topic
        
        if best_match and best_score >= 1:
            return FAQ_KNOWLEDGE[best_match]["answer"]
        else:
            return None
    
    # ── Initialize Chat History ─────────────────────────────────────
    if "faq_messages" not in st.session_state:
        st.session_state.faq_messages = [
            {"role": "assistant", "content": """👋 **Welcome to the FAQ Chatbot!**

I can answer questions about disease forecasting, the SEIR model, interventions, data sources, and more.

**Try asking me:**
- "What is R₀?"
- "How does the ensemble forecasting work?"
- "Why do forecasts show a range?"
- "What data sources are available?"

Or just type your question below!"""}
        ]
    
    # ── Display Chat History ────────────────────────────────────────
    for message in st.session_state.faq_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # ── Chat Input ──────────────────────────────────────────────────
    if user_input := st.chat_input("Ask me anything about disease forecasting..."):
        # Add user message to history
        st.session_state.faq_messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Find answer
        answer = find_best_answer(user_input)
        
        if answer:
            response = answer
        else:
            response = f"""🤔 I'm not sure about that specific question. 

Here are topics I can help with:
- **R₀ / spread rate** — "What is R₀?"
- **SEIR model** — "What are the disease states?"
- **Forecasting accuracy** — "How accurate are forecasts?"
- **Interventions** — "How do vaccines work in the model?"
- **Ensemble method** — "How does hybrid forecasting work?"
- **Agent simulation** — "What is the agent-based model?"
- **Data sources** — "What data is available?"
- **Uncertainty** — "Why do forecasts show a range?"

Try rephrasing your question, or ask one of these!"""
        
        # Add assistant response
        st.session_state.faq_messages.append({"role": "assistant", "content": response})
        
        with st.chat_message("assistant"):
            st.markdown(response)
    
    # ── Sidebar Quick Topics ────────────────────────────────────────
    st.sidebar.markdown("### 💡 Quick Topics")
    quick_topics = [
        "What is R₀?",
        "How accurate are forecasts?",
        "How does ensemble work?",
        "What is SEIR?",
        "What data sources exist?",
    ]
    for topic in quick_topics:
        if st.sidebar.button(topic, key=f"quick_{topic}"):
            st.session_state.faq_messages.append({"role": "user", "content": topic})
            answer = find_best_answer(topic)
            st.session_state.faq_messages.append({"role": "assistant", "content": answer})
            st.rerun()
    
    if st.sidebar.button("🗑️ Clear Chat"):
        st.session_state.faq_messages = [st.session_state.faq_messages[0]]
        st.rerun()
    
    # ── Resources Section (collapsible) ─────────────────────────────
    st.markdown("---")
    
    with st.expander("📚 External Resources"):
        st.markdown("""
**Reliable sources for disease forecasting:**

- **[CDC COVID-19 Forecasting](https://www.cdc.gov/coronavirus/2019-ncov/science/forecasting/forecasts-cases.html)** — Weekly forecasts from the CDC
- **[WHO Disease Outbreak News](https://www.who.int/emergencies/disease-outbreak-news)** — Global outbreak reports
- **[COVID-19 Forecast Hub](https://covid19forecasthub.org/)** — Ensemble of 50+ forecasting models
- **[Our World in Data](https://ourworldindata.org/explorers/coronavirus-data-explorer)** — Clear visualizations
        """)
    
    with st.expander("🛠️ Technical Details"):
        st.markdown("""
**Model Types:**
- Custom Forecast: Stochastic SEIR with Monte Carlo sampling
- Validation Forecast: Hybrid Ensemble (Renewal 40% + Trend 35% + SEIR 25%)

**Simulation:** 500 Monte Carlo runs with Poisson-distributed transitions

**Parameters:**
- β (beta) = R₀ / infectious_period
- σ (sigma) = 1 / latent_period  
- γ (gamma) = 1 / infectious_period

**Agent Simulation:** SEIR ABM on 100×100 grid with proximity-based transmission
        """)


# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Data Sources:** CDC, WHO, ECDC")
st.sidebar.markdown("**Version:** 2.0 (User-Friendly)")
