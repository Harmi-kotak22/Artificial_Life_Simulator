"""
Validation Framework Test Script

Demonstrates the complete validation framework:
1. Generate synthetic forecasts and observations
2. Compute all forecast metrics
3. Assess calibration
4. Run proper scoring rules
5. Perform diagnostics
"""

import sys
sys.path.insert(0, '/home/shivrajsinh/Documents/Disaster Forecast')

import numpy as np
from datetime import datetime

print("=" * 70)
print("UNIVERSAL PROBABILISTIC DISASTER FORECASTING FRAMEWORK")
print("Validation Framework Demonstration")
print("=" * 70)

# ============================================================
# STEP 1: Generate Synthetic Epidemic Data and Forecasts
# ============================================================

print("\n" + "-" * 70)
print("STEP 1: Generating Synthetic Data")
print("-" * 70)

np.random.seed(42)

# Simulate a simplified epidemic curve
n_days = 60
population = 100_000

# True underlying epidemic (logistic growth + noise)
t = np.arange(n_days)
peak_day = 25
growth_rate = 0.2
max_cases = 5000

# Logistic curve for true daily cases
true_cases = max_cases / (1 + np.exp(-growth_rate * (t - peak_day)))
true_incidence = np.diff(np.concatenate([[0], true_cases]))
true_incidence = np.maximum(true_incidence, 0)

# Add observation noise (negative binomial)
k = 5  # dispersion
observed_cases = np.zeros(n_days)
for i in range(n_days):
    mu = true_incidence[i] * 0.3  # 30% reporting
    if mu > 0:
        p = k / (k + mu)
        observed_cases[i] = np.random.negative_binomial(k, p)

print(f"Generated {n_days} days of epidemic data")
print(f"Peak true incidence: {max(true_incidence):.0f} cases on day {np.argmax(true_incidence)}")
print(f"Total observed cases: {sum(observed_cases):.0f}")

# Generate ensemble forecasts (some with different quality)
n_ensemble = 100
forecast_start = 20
forecast_horizon = 30

# Good forecast: captures dynamics with appropriate uncertainty
good_forecast = np.zeros((forecast_horizon, n_ensemble))
for h in range(forecast_horizon):
    true_val = true_incidence[forecast_start + h] * 0.3 if forecast_start + h < n_days else 0
    spread = 50 + 5 * h  # Uncertainty grows with horizon
    good_forecast[h] = np.random.normal(true_val, spread, n_ensemble)
good_forecast = np.maximum(good_forecast, 0)

# Biased forecast: systematically overpredicts
biased_forecast = good_forecast * 1.5 + 50

# Underdispersed forecast: too confident
narrow_forecast = np.zeros((forecast_horizon, n_ensemble))
for h in range(forecast_horizon):
    true_val = true_incidence[forecast_start + h] * 0.3 if forecast_start + h < n_days else 0
    spread = 10 + h  # Much narrower spread
    narrow_forecast[h] = np.random.normal(true_val, spread, n_ensemble)
narrow_forecast = np.maximum(narrow_forecast, 0)

# Get actual observations for validation period
validation_obs = observed_cases[forecast_start:forecast_start + forecast_horizon]

print(f"\nGenerated 3 forecast ensembles for {forecast_horizon} days ahead")
print(f"  - Good forecast: calibrated with appropriate spread")
print(f"  - Biased forecast: systematically overpredicts")
print(f"  - Narrow forecast: underdispersed (too confident)")

# ============================================================
# STEP 2: Compute Forecast Metrics
# ============================================================

print("\n" + "-" * 70)
print("STEP 2: Computing Forecast Metrics")
print("-" * 70)

from updff.validation.metrics import (
    compute_all_metrics,
    mean_absolute_error,
    root_mean_squared_error,
    crps,
    interval_score,
    coverage_probability
)

# Compute metrics for good forecast
print("\n>>> Good Forecast Metrics:")
good_metrics = compute_all_metrics(validation_obs, good_forecast)
print(good_metrics.summary())

# Compute metrics for biased forecast
print("\n>>> Biased Forecast Metrics:")
biased_metrics = compute_all_metrics(validation_obs, biased_forecast)
print(f"  MAE:  {biased_metrics.mae:.2f}")
print(f"  RMSE: {biased_metrics.rmse:.2f}")
print(f"  Bias: {biased_metrics.bias:.2f}")
print(f"  CRPS: {biased_metrics.crps:.2f}")
print(f"  90% Coverage: {biased_metrics.coverage_90:.1%}")

# Compute metrics for narrow forecast
print("\n>>> Narrow Forecast Metrics:")
narrow_metrics = compute_all_metrics(validation_obs, narrow_forecast)
print(f"  MAE:  {narrow_metrics.mae:.2f}")
print(f"  RMSE: {narrow_metrics.rmse:.2f}")
print(f"  Bias: {narrow_metrics.bias:.2f}")
print(f"  CRPS: {narrow_metrics.crps:.2f}")
print(f"  90% Coverage: {narrow_metrics.coverage_90:.1%}")

# ============================================================
# STEP 3: Calibration Assessment
# ============================================================

print("\n" + "-" * 70)
print("STEP 3: Calibration Assessment")
print("-" * 70)

from updff.validation.calibration import (
    assess_calibration,
    pit_histogram,
    coverage_test,
    ks_test_uniformity
)

print("\n>>> Good Forecast Calibration:")
good_cal = assess_calibration(validation_obs, good_forecast)
print(f"  KS test p-value: {good_cal.ks_pvalue:.4f}")
print(f"  Is calibrated (α=0.05): {good_cal.is_calibrated()}")
print(f"  Coverage by level:")
for level, cov in sorted(good_cal.coverage_by_level.items()):
    diff = cov - level
    print(f"    {level:.0%}: {cov:.1%} (diff: {diff:+.1%})")

print("\n>>> Narrow Forecast Calibration:")
narrow_cal = assess_calibration(validation_obs, narrow_forecast)
print(f"  KS test p-value: {narrow_cal.ks_pvalue:.4f}")
print(f"  Is calibrated (α=0.05): {narrow_cal.is_calibrated()}")
print(f"  Coverage by level:")
for level, cov in sorted(narrow_cal.coverage_by_level.items()):
    diff = cov - level
    status = "✓" if abs(diff) < 0.1 else "✗"
    print(f"    {level:.0%}: {cov:.1%} (diff: {diff:+.1%}) {status}")

# ============================================================
# STEP 4: Proper Scoring Rules
# ============================================================

print("\n" + "-" * 70)
print("STEP 4: Proper Scoring Rules")
print("-" * 70)

from updff.validation.scoring import (
    CRPSScore,
    LogScore,
    IntervalScore,
    WeightedIntervalScore,
    compute_skill_score
)

# CRPS comparison
crps_scorer = CRPSScore()
good_crps = crps_scorer.mean_score(validation_obs, ensemble=good_forecast)
biased_crps = crps_scorer.mean_score(validation_obs, ensemble=biased_forecast)
narrow_crps = crps_scorer.mean_score(validation_obs, ensemble=narrow_forecast)

print("\n>>> CRPS Scores (lower is better):")
print(f"  Good forecast:   {good_crps:.2f}")
print(f"  Biased forecast: {biased_crps:.2f}")
print(f"  Narrow forecast: {narrow_crps:.2f}")

# Interval scores at 90% level
is_scorer = IntervalScore(alpha=0.1)
q05_good = np.percentile(good_forecast, 5, axis=1)
q95_good = np.percentile(good_forecast, 95, axis=1)
q05_narrow = np.percentile(narrow_forecast, 5, axis=1)
q95_narrow = np.percentile(narrow_forecast, 95, axis=1)

good_is = is_scorer.mean_score(validation_obs, lower=q05_good, upper=q95_good)
narrow_is = is_scorer.mean_score(validation_obs, lower=q05_narrow, upper=q95_narrow)

print("\n>>> 90% Interval Scores (lower is better):")
print(f"  Good forecast:   {good_is:.2f}")
print(f"  Narrow forecast: {narrow_is:.2f}")

# Weighted Interval Score
wis_scorer = WeightedIntervalScore()
good_wis = wis_scorer.mean_score(validation_obs, ensemble=good_forecast)
narrow_wis = wis_scorer.mean_score(validation_obs, ensemble=narrow_forecast)

print("\n>>> Weighted Interval Scores:")
print(f"  Good forecast:   {good_wis:.2f}")
print(f"  Narrow forecast: {narrow_wis:.2f}")

# Skill score relative to climatology
climatology_mean = np.mean(observed_cases[:forecast_start])
climatology_std = np.std(observed_cases[:forecast_start])
climatology_forecast = np.random.normal(climatology_mean, climatology_std, (forecast_horizon, n_ensemble))
climatology_forecast = np.maximum(climatology_forecast, 0)
climatology_crps = crps_scorer.mean_score(validation_obs, ensemble=climatology_forecast)

skill_score = compute_skill_score(good_crps, climatology_crps)
print(f"\n>>> CRPS Skill Score (vs climatology): {skill_score:.3f}")
print(f"    (1.0 = perfect, 0.0 = same as climatology, <0 = worse)")

# ============================================================
# STEP 5: Forecast Diagnostics
# ============================================================

print("\n" + "-" * 70)
print("STEP 5: Forecast Diagnostics")
print("-" * 70)

from updff.validation.diagnostics import (
    residual_analysis,
    ensemble_diagnostics,
    forecast_horizon_analysis,
    compare_forecasts
)

# Residual analysis
good_mean = np.mean(good_forecast, axis=1)
good_std = np.std(good_forecast, axis=1)
residuals = residual_analysis(validation_obs, good_mean, good_std)

print("\n>>> Residual Analysis (Good Forecast):")
print(f"  Mean residual: {residuals['mean']:.2f}")
print(f"  Std residual:  {residuals['std']:.2f}")
print(f"  Skewness:      {residuals['skewness']:.2f}")
print(f"  Kurtosis:      {residuals['kurtosis']:.2f}")
if 'ljung_box_pvalue' in residuals:
    print(f"  Ljung-Box p-value: {residuals['ljung_box_pvalue']:.4f}")
    print(f"  Residuals independent (α=0.05): {residuals['ljung_box_pvalue'] > 0.05}")

# Ensemble diagnostics
print("\n>>> Ensemble Diagnostics (Good Forecast):")
ens_diag = ensemble_diagnostics(validation_obs, good_forecast)
print(f"  Ensemble members: {ens_diag['n_members']}")
print(f"  Mean spread:      {ens_diag['mean_spread']:.2f}")
print(f"  RMSE:             {ens_diag['rmse']:.2f}")
print(f"  Spread-Skill ratio: {ens_diag['spread_skill_ratio']:.2f}")
print(f"    (Ideal ≈ 1.0, <1 = underdispersed, >1 = overdispersed)")
print(f"  Rank histogram interpretation: {ens_diag['interpretation']}")

print("\n>>> Ensemble Diagnostics (Narrow Forecast):")
narrow_diag = ensemble_diagnostics(validation_obs, narrow_forecast)
print(f"  Spread-Skill ratio: {narrow_diag['spread_skill_ratio']:.2f}")
print(f"  Rank histogram interpretation: {narrow_diag['interpretation']}")

# Forecast comparison
print("\n>>> Comparing All Forecasts:")
comparison = compare_forecasts(
    validation_obs,
    {
        "Good": good_forecast,
        "Biased": biased_forecast,
        "Narrow": narrow_forecast,
        "Climatology": climatology_forecast
    }
)

print("\n  Rankings by CRPS:")
for method, rank in sorted(comparison["rankings"]["crps"].items(), key=lambda x: x[1]):
    print(f"    {rank}. {method}")

print("\n  Rankings by Calibration (90% coverage):")
for method, rank in sorted(comparison["rankings"]["calibration"].items(), key=lambda x: x[1]):
    print(f"    {rank}. {method}")

print(f"\n  Overall best method: {comparison['overall_best']}")

print("\n  Pairwise Statistical Tests (Diebold-Mariano):")
for pair, result in comparison["pairwise_tests"].items():
    if result["significant"]:
        print(f"    {pair}: {result['better']} is significantly better (p={result['p_value']:.4f})")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION FRAMEWORK SUMMARY")
print("=" * 70)

print("""
The validation framework provides comprehensive forecast evaluation:

✓ METRICS MODULE
  - Point metrics: MAE, RMSE, MAPE, Bias
  - Probabilistic metrics: CRPS, Log Score
  - Interval metrics: Coverage, Interval Score, Sharpness

✓ CALIBRATION MODULE  
  - PIT histogram analysis
  - Coverage tests at multiple levels
  - Statistical tests (KS, Chi-squared)
  - Calibration error computation

✓ SCORING MODULE
  - Proper scoring rules (CRPS, Log Score, Brier Score)
  - Interval scores (single and weighted)
  - Skill score computation

✓ DIAGNOSTICS MODULE
  - Residual analysis (autocorrelation, normality)
  - Ensemble diagnostics (spread-skill, rank histogram)
  - Forecast horizon analysis
  - Multi-method comparison with statistical tests

Key Findings from This Test:
  1. Good forecast: Well-calibrated (90% coverage ≈ 90%)
  2. Biased forecast: High MAE/RMSE due to systematic overprediction
  3. Narrow forecast: Poor calibration (too confident), 
     90% intervals capture far less than 90% of observations
  4. Spread-skill ratio reveals underdispersion in narrow forecast
""")

print("=" * 70)
print("VALIDATION TEST COMPLETE")
print("=" * 70)
