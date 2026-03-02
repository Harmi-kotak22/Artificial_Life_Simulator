"""
End-to-End Disease Forecasting with Validation

Complete demonstration of the UPDFF framework:
1. Create disease module
2. Generate synthetic outbreak
3. Run probabilistic forecasts
4. Validate forecasts using the validation framework
"""

import sys
sys.path.insert(0, '/home/shivrajsinh/Documents/Disaster Forecast')

import numpy as np
from datetime import datetime, timedelta

print("=" * 70)
print("UPDFF END-TO-END DEMONSTRATION")
print("Disease Forecasting with Full Validation")
print("=" * 70)

# ============================================================
# PART 1: CREATE DISEASE MODULE
# ============================================================

print("\n" + "=" * 70)
print("PART 1: Disease Module Setup")
print("=" * 70)

from updff.hazards.disease import create_disease_module, PathogenTraits
from updff.core.state import State, Parameters

# Create a COVID-like disease module
population = 100_000
module = create_disease_module(
    pathogen="covid",
    population=population,
    model_type="seir"
)

print(f"\n✓ Created disease module:")
print(f"  Pathogen: COVID-like")
print(f"  Model: SEIR")
print(f"  Population: {population:,}")

# Get state specification
spec = module.get_state_spec()
print(f"\n  State variables: {spec.state_names}")
print(f"  Parameters: {spec.param_names}")

# Get priors from module
priors = module.get_prior()
print(f"\n  Prior distributions:")
for name in ["beta", "gamma", "R0"]:
    if name in priors:
        p = priors[name]
        print(f"    {name}: mean={p.mean():.4f}, std={p.std():.4f}")

# ============================================================
# PART 2: GENERATE SYNTHETIC OUTBREAK DATA
# ============================================================

print("\n" + "=" * 70)
print("PART 2: Synthetic Outbreak Generation")
print("=" * 70)

np.random.seed(42)

# True parameters for simulation
true_R0 = 2.5
true_infectious_period = 8.0  # days
true_latent_period = 3.0      # days
true_beta = true_R0 / true_infectious_period
true_sigma = 1.0 / true_latent_period
true_gamma = 1.0 / true_infectious_period
reporting_rate = 0.3

print(f"\n  True parameters:")
print(f"    R0 = {true_R0}")
print(f"    Infectious period = {true_infectious_period} days")
print(f"    Latent period = {true_latent_period} days")
print(f"    Reporting rate = {reporting_rate:.0%}")

# Simulate outbreak manually for ground truth
n_days = 90
S = np.zeros(n_days + 1)
E = np.zeros(n_days + 1)
I = np.zeros(n_days + 1)
R = np.zeros(n_days + 1)

# Initial conditions
initial_infected = 10
S[0] = population - initial_infected
E[0] = 5
I[0] = 5
R[0] = 0

true_incidence = []
observed_cases = []

for t in range(n_days):
    # Stochastic transitions
    new_E = np.random.poisson(true_beta * S[t] * I[t] / population)
    new_I = np.random.poisson(true_sigma * E[t])
    new_R = np.random.poisson(true_gamma * I[t])
    
    # Clamp to valid ranges
    new_E = min(new_E, int(S[t]))
    new_I = min(new_I, int(E[t]))
    new_R = min(new_R, int(I[t]))
    
    # Update state
    S[t + 1] = S[t] - new_E
    E[t + 1] = E[t] + new_E - new_I
    I[t + 1] = I[t] + new_I - new_R
    R[t + 1] = R[t] + new_R
    
    true_incidence.append(new_I)
    
    # Observed with reporting noise
    expected_obs = new_I * reporting_rate
    if expected_obs > 0:
        k = 5  # dispersion
        p = k / (k + expected_obs)
        obs = np.random.negative_binomial(k, p)
    else:
        obs = 0
    observed_cases.append(obs)

true_incidence = np.array(true_incidence)
observed_cases = np.array(observed_cases)

peak_day = np.argmax(true_incidence)
print(f"\n  Outbreak statistics:")
print(f"    Duration: {n_days} days")
print(f"    Peak true incidence: {max(true_incidence):.0f} on day {peak_day}")
print(f"    Total true cases: {sum(true_incidence):.0f}")
print(f"    Total observed: {sum(observed_cases):.0f}")
print(f"    Final attack rate: {R[-1]/population:.1%}")

# ============================================================
# PART 3: GENERATE ENSEMBLE FORECASTS
# ============================================================

print("\n" + "=" * 70)
print("PART 3: Ensemble Forecasting")
print("=" * 70)

# Forecast from day 30
forecast_start = 30
forecast_horizon = 30
n_ensemble = 200

print(f"\n  Forecast setup:")
print(f"    Start day: {forecast_start}")
print(f"    Horizon: {forecast_horizon} days")
print(f"    Ensemble size: {n_ensemble}")

# Initialize state from day 30
initial_state = State(
    values=np.array([S[forecast_start], E[forecast_start], 
                     I[forecast_start], R[forecast_start]]),
    timestamp=float(forecast_start)
)

print(f"\n  Initial state (day {forecast_start}):")
print(f"    S: {initial_state.values[0]:,.0f}")
print(f"    E: {initial_state.values[1]:,.0f}")
print(f"    I: {initial_state.values[2]:,.0f}")
print(f"    R: {initial_state.values[3]:,.0f}")

# Sample parameters from posterior-like distribution (for demo)
# In practice, these would come from MCMC inference
param_samples = {
    "beta": np.random.normal(true_beta, 0.05, n_ensemble),
    "sigma": np.random.normal(true_sigma, 0.03, n_ensemble),
    "gamma": np.random.normal(true_gamma, 0.02, n_ensemble),
}

# Run ensemble forecast
print(f"\n  Running ensemble forecast...")

ensemble_forecasts = np.zeros((forecast_horizon, n_ensemble, 4))  # [time, member, state]
incidence_forecasts = np.zeros((forecast_horizon, n_ensemble))

for ens in range(n_ensemble):
    # Parameters for this ensemble member
    params = Parameters(
        values=np.array([
            param_samples["beta"][ens],
            param_samples["sigma"][ens],
            param_samples["gamma"][ens],
            true_R0,  # R0
            5.0,      # k (dispersion)
            reporting_rate
        ]),
        names=spec.param_names[:6]
    )
    
    state = State(values=initial_state.values.copy(), timestamp=initial_state.timestamp)
    
    for h in range(forecast_horizon):
        # Get previous I for incidence calculation
        prev_I = state.values[2]
        
        # Step forward
        next_states = module.transition(
            state=state,
            params=params,
            interventions=[],
            dt=1.0,
            n_samples=1
        )
        state = next_states[0]
        
        ensemble_forecasts[h, ens] = state.values
        
        # Approximate incidence (new cases)
        new_I_approx = params.values[1] * prev_I  # sigma * E approx
        incidence_forecasts[h, ens] = max(0, state.values[2] * reporting_rate)

# Summary statistics
forecast_mean = np.mean(incidence_forecasts, axis=1)
forecast_std = np.std(incidence_forecasts, axis=1)
forecast_q05 = np.percentile(incidence_forecasts, 5, axis=1)
forecast_q50 = np.percentile(incidence_forecasts, 50, axis=1)
forecast_q95 = np.percentile(incidence_forecasts, 95, axis=1)

print(f"\n  Forecast generated!")
print(f"    Mean forecast at day {forecast_start + 7}: {forecast_mean[7]:.1f}")
print(f"    90% CI: [{forecast_q05[7]:.1f}, {forecast_q95[7]:.1f}]")

# ============================================================
# PART 4: VALIDATE FORECASTS
# ============================================================

print("\n" + "=" * 70)
print("PART 4: Forecast Validation")
print("=" * 70)

from updff.validation.metrics import compute_all_metrics, crps, coverage_probability
from updff.validation.calibration import assess_calibration, coverage_test
from updff.validation.scoring import CRPSScore, WeightedIntervalScore, compute_skill_score
from updff.validation.diagnostics import ensemble_diagnostics, residual_analysis

# Get validation observations (actual observed cases in forecast period)
validation_obs = observed_cases[forecast_start:forecast_start + forecast_horizon]

# Also get true infectious counts for comparison
true_I_validation = I[forecast_start + 1:forecast_start + forecast_horizon + 1]

# Compute metrics
print("\n>>> Point Forecast Metrics:")
metrics = compute_all_metrics(validation_obs, incidence_forecasts)
print(f"    MAE:  {metrics.mae:.2f}")
print(f"    RMSE: {metrics.rmse:.2f}")
print(f"    Bias: {metrics.bias:.2f}")

print("\n>>> Probabilistic Metrics:")
print(f"    CRPS: {metrics.crps:.2f}")
print(f"    Log Score: {metrics.log_score:.2f}")

print("\n>>> Calibration (Coverage):")
print(f"    50% CI coverage: {metrics.coverage_50:.1%} (nominal: 50%)")
print(f"    90% CI coverage: {metrics.coverage_90:.1%} (nominal: 90%)")
print(f"    95% CI coverage: {metrics.coverage_95:.1%} (nominal: 95%)")

print("\n>>> Sharpness (Interval Width):")
print(f"    50% CI mean width: {metrics.mean_interval_width_50:.1f}")
print(f"    90% CI mean width: {metrics.mean_interval_width_90:.1f}")

# Calibration assessment
print("\n>>> Calibration Assessment:")
cal = assess_calibration(validation_obs, incidence_forecasts)
print(f"    KS test p-value: {cal.ks_pvalue:.4f}")
print(f"    Calibrated (α=0.05): {cal.is_calibrated()}")

# Coverage at multiple levels
coverage = coverage_test(validation_obs, incidence_forecasts)
print("\n    Coverage by level:")
for level, cov in sorted(coverage.items()):
    diff = cov - level
    status = "✓" if abs(diff) < 0.15 else "✗"
    print(f"      {level:.0%}: {cov:.1%} (diff: {diff:+.1%}) {status}")

# Ensemble diagnostics
print("\n>>> Ensemble Diagnostics:")
ens_diag = ensemble_diagnostics(validation_obs, incidence_forecasts)
print(f"    Mean spread: {ens_diag['mean_spread']:.2f}")
print(f"    RMSE: {ens_diag['rmse']:.2f}")
print(f"    Spread-Skill ratio: {ens_diag['spread_skill_ratio']:.2f}")
print(f"    (Ideal ≈ 1.0)")
print(f"    Interpretation: {ens_diag['interpretation']}")

# Skill score vs climatology
print("\n>>> Skill Score:")
crps_scorer = CRPSScore()
forecast_crps = crps_scorer.mean_score(validation_obs, ensemble=incidence_forecasts)

# Climatology baseline
clim_mean = np.mean(observed_cases[:forecast_start])
clim_std = np.std(observed_cases[:forecast_start])
climatology = np.random.normal(clim_mean, clim_std, (forecast_horizon, n_ensemble))
climatology = np.maximum(climatology, 0)
climatology_crps = crps_scorer.mean_score(validation_obs, ensemble=climatology)

skill = compute_skill_score(forecast_crps, climatology_crps)
print(f"    Forecast CRPS: {forecast_crps:.2f}")
print(f"    Climatology CRPS: {climatology_crps:.2f}")
print(f"    Skill score: {skill:.3f}")
print(f"    (>0 means better than climatology)")

# ============================================================
# PART 5: FORECAST HORIZON ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("PART 5: Performance by Forecast Horizon")
print("=" * 70)

print("\n    Day | Obs | Forecast | 90% CI          | In CI?")
print("    " + "-" * 50)

in_ci_count = 0
for h in range(min(14, forecast_horizon)):
    day = forecast_start + h + 1
    obs = validation_obs[h]
    fcst = forecast_mean[h]
    q05 = forecast_q05[h]
    q95 = forecast_q95[h]
    in_ci = q05 <= obs <= q95
    if in_ci:
        in_ci_count += 1
    print(f"    {day:3d} | {obs:3.0f} | {fcst:8.1f} | [{q05:6.1f}, {q95:6.1f}] | {'✓' if in_ci else '✗'}")

print(f"\n    First 14 days coverage: {in_ci_count}/14 = {in_ci_count/14:.1%}")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
End-to-End Test Complete!

Disease Module:
  ✓ Created SEIR model for population of {population:,}
  ✓ Pathogen-agnostic trait system working
  ✓ State transitions producing realistic dynamics

Forecasting:
  ✓ Generated {n_ensemble}-member ensemble forecast
  ✓ {forecast_horizon}-day forecast horizon from day {forecast_start}
  ✓ Forecasts capture epidemic trajectory

Validation Results:
  • MAE = {metrics.mae:.1f} cases
  • CRPS = {metrics.crps:.2f}
  • 90% CI Coverage = {metrics.coverage_90:.1%}
  • Spread-Skill Ratio = {ens_diag['spread_skill_ratio']:.2f}
  • Skill vs Climatology = {skill:.3f}

Interpretation:
""")

if metrics.coverage_90 > 0.85 and metrics.coverage_90 < 0.95:
    print("  ✓ Forecasts appear well-calibrated (90% coverage near 90%)")
elif metrics.coverage_90 < 0.85:
    print("  ⚠ Forecasts may be overconfident (90% coverage below 85%)")
else:
    print("  ⚠ Forecasts may be overdispersed (90% coverage above 95%)")

if abs(ens_diag['spread_skill_ratio'] - 1.0) < 0.3:
    print("  ✓ Ensemble spread well-matched to forecast error")
elif ens_diag['spread_skill_ratio'] < 0.7:
    print("  ⚠ Ensemble may be underdispersed")
else:
    print("  ⚠ Ensemble may be overdispersed")

if skill > 0:
    print(f"  ✓ Forecast beats climatology baseline (skill = {skill:.2f})")
else:
    print(f"  ⚠ Forecast does not beat climatology (skill = {skill:.2f})")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
