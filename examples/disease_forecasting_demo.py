"""
Example: Disease Outbreak Forecasting

Demonstrates how to use the UPDFF framework to:
1. Create a disease forecasting module
2. Calibrate parameters from historical data
3. Generate probabilistic forecasts
4. Evaluate forecast uncertainty
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import framework components
from updff.core.state import State, Parameters, Observation, ObservationSeries
from updff.core.distribution import Normal, LogNormal, Gamma
from updff.core.forecast import ForecastingEngine, ForecastConfig
from updff.hazards.disease import (
    DiseaseModule,
    PathogenTraits,
    create_disease_module
)
from updff.inference.likelihood import NegativeBinomialLikelihood
from updff.inference.mcmc import MetropolisHastings, MCMCResult
from updff.inference.optimizer import MaximumAPosteriori


def generate_synthetic_outbreak(
    population: int = 100_000,
    r0: float = 2.5,
    infectious_period: float = 7.0,
    n_days: int = 60,
    initial_infected: int = 10,
    reporting_rate: float = 0.3,
    dispersion: float = 5.0,
    seed: int = 42
) -> dict:
    """
    Generate synthetic outbreak data for demonstration.
    
    Uses a simplified SEIR model to create realistic epidemic curves.
    """
    np.random.seed(seed)
    
    # Model parameters
    beta = r0 / infectious_period
    sigma = 1.0 / 3.0  # 3-day latent period
    gamma = 1.0 / infectious_period
    
    # Initial state
    S = population - initial_infected
    E = initial_infected // 2
    I = initial_infected - E
    R = 0
    
    # Storage
    true_incidence = []
    reported_cases = []
    states = {"S": [S], "E": [E], "I": [I], "R": [R]}
    
    for day in range(n_days):
        # Stochastic transitions
        new_exposed = np.random.poisson(beta * S * I / population)
        new_infectious = np.random.poisson(sigma * E)
        new_recovered = np.random.poisson(gamma * I)
        
        # Ensure valid transitions
        new_exposed = min(new_exposed, S)
        new_infectious = min(new_infectious, E)
        new_recovered = min(new_recovered, I)
        
        # Update state
        S = S - new_exposed
        E = E + new_exposed - new_infectious
        I = I + new_infectious - new_recovered
        R = R + new_recovered
        
        # Record
        true_incidence.append(new_infectious)
        
        # Observed cases (with reporting delay and noise)
        expected_reported = new_infectious * reporting_rate
        if expected_reported > 0:
            k = dispersion
            p = k / (k + expected_reported)
            observed = np.random.negative_binomial(k, p)
        else:
            observed = 0
        reported_cases.append(observed)
        
        states["S"].append(S)
        states["E"].append(E)
        states["I"].append(I)
        states["R"].append(R)
    
    return {
        "true_incidence": np.array(true_incidence),
        "reported_cases": np.array(reported_cases),
        "states": {k: np.array(v) for k, v in states.items()},
        "true_params": {
            "R0": r0,
            "infectious_period": infectious_period,
            "beta": beta,
            "sigma": sigma,
            "gamma": gamma,
            "reporting_rate": reporting_rate
        },
        "population": population
    }


def run_parameter_estimation_demo(data: dict, n_training_days: int = 30):
    """
    Demonstrate parameter estimation from data.
    """
    print("\n" + "=" * 60)
    print("PARAMETER ESTIMATION DEMO")
    print("=" * 60)
    
    # Use first n_training_days for fitting
    training_data = data["reported_cases"][:n_training_days]
    population = data["population"]
    
    # Create disease module
    print(f"\nCreating disease module (SEIR) for population of {population:,}")
    module = create_disease_module(
        pathogen="covid",  # Uses COVID-like priors
        population=population,
        model_type="seir"
    )
    
    # Get priors from module
    priors = module.get_prior()
    param_names = ["beta", "sigma", "gamma", "reporting_rate"]
    
    print("\nPrior distributions:")
    for name in param_names:
        if name in priors:
            prior = priors[name]
            print(f"  {name}: mean={prior.mean():.4f}, std={prior.std():.4f}")
    
    # Define likelihood function
    def log_likelihood(params: np.ndarray) -> float:
        """Compute log-likelihood given parameters."""
        beta, sigma, gamma, reporting = params
        
        if any(p <= 0 for p in params):
            return -np.inf
        if reporting > 1:
            return -np.inf
        
        # Simple forward simulation
        S = population - 10
        E = 5
        I = 5
        R = 0
        
        log_lik = 0.0
        likelihood = NegativeBinomialLikelihood(dispersion=5.0)
        
        for day in range(len(training_data)):
            # Deterministic transitions for speed
            new_E = beta * S * I / population
            new_I = sigma * E
            new_R = gamma * I
            
            S -= new_E
            E += new_E - new_I
            I += new_I - new_R
            R += new_R
            
            # Expected observed
            expected = I * reporting
            observed = training_data[day]
            
            # Add to likelihood
            log_lik += likelihood(
                np.array([observed]),
                np.array([expected])
            )
        
        return log_lik
    
    # Define log-prior function
    def log_prior(params: np.ndarray) -> float:
        """Compute log-prior probability."""
        beta, sigma, gamma, reporting = params
        
        if any(p <= 0 for p in params):
            return -np.inf
        if reporting > 1:
            return -np.inf
        
        log_p = 0.0
        log_p += LogNormal.from_mean_std(0.3, 0.15).log_prob(np.array([beta]))[0]
        log_p += Gamma.from_mean_std(0.33, 0.1).log_prob(np.array([sigma]))[0]
        log_p += Gamma.from_mean_std(0.14, 0.05).log_prob(np.array([gamma]))[0]
        log_p += Normal(0.3, 0.1).log_prob(np.array([reporting]))[0]
        
        return log_p
    
    # MAP estimation
    print("\n" + "-" * 40)
    print("Running MAP estimation...")
    
    optimizer = MaximumAPosteriori(
        param_names=param_names,
        param_bounds=[(0.01, 2.0), (0.01, 1.0), (0.01, 1.0), (0.01, 0.99)]
    )
    
    # Initial guess from priors
    initial_params = np.array([0.3, 0.33, 0.14, 0.3])
    
    result = optimizer.fit(
        log_likelihood_fn=log_likelihood,
        log_prior_fn=log_prior,
        initial_params=initial_params
    )
    
    print(f"\nMAP estimates:")
    estimated = result.to_dict()
    true = data["true_params"]
    
    print(f"  {'Parameter':<20} {'Estimated':>12} {'True':>12} {'Error':>10}")
    print(f"  {'-'*54}")
    
    for name in param_names:
        est_val = estimated[name]
        if name in true:
            true_val = true[name]
            error = abs(est_val - true_val) / true_val * 100
            print(f"  {name:<20} {est_val:>12.4f} {true_val:>12.4f} {error:>9.1f}%")
        else:
            print(f"  {name:<20} {est_val:>12.4f}")
    
    # MCMC for uncertainty
    print("\n" + "-" * 40)
    print("Running MCMC for uncertainty quantification...")
    
    sampler = MetropolisHastings(
        param_names=param_names,
        proposal_scale=0.01
    )
    
    def log_posterior(params):
        ll = log_likelihood(params)
        lp = log_prior(params)
        if not np.isfinite(ll) or not np.isfinite(lp):
            return -np.inf
        return ll + lp
    
    mcmc_result = sampler.sample(
        log_prob_fn=log_posterior,
        initial_state=result.optimal_params,
        n_samples=2000,
        n_warmup=500
    )
    
    print(f"\nMCMC diagnostics:")
    print(f"  Acceptance rate: {mcmc_result.acceptance_rate:.2%}")
    print(f"\nPosterior summary (95% CI):")
    
    cis = mcmc_result.credible_interval(0.05)
    means = mcmc_result.mean()
    
    for name in param_names:
        ci = cis[name]
        mean = means[name]
        print(f"  {name}: {mean:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")
    
    return result, mcmc_result


def run_forecasting_demo(data: dict, fit_params: np.ndarray, n_training: int = 30, forecast_days: int = 14):
    """
    Demonstrate probabilistic forecasting.
    """
    print("\n" + "=" * 60)
    print("PROBABILISTIC FORECASTING DEMO")
    print("=" * 60)
    
    population = data["population"]
    
    # Create disease module with estimated parameters
    module = create_disease_module(
        pathogen="covid",
        population=population,
        model_type="seir"
    )
    
    # Extract estimated parameters
    beta, sigma, gamma, reporting = fit_params
    
    # Initialize state from last training day
    true_states = data["states"]
    
    initial_state = State(
        values=np.array([
            true_states["S"][n_training],
            true_states["E"][n_training],
            true_states["I"][n_training],
            true_states["R"][n_training]
        ]),
        timestamp=float(n_training)
    )
    
    params = Parameters(
        values=np.array([beta, sigma, gamma, data["true_params"]["R0"], 5.0, reporting]),
        names=["beta", "sigma", "gamma", "R0", "k", "reporting_rate"]
    )
    
    print(f"\nInitial state at day {n_training}:")
    print(f"  S: {initial_state.values[0]:,.0f}")
    print(f"  E: {initial_state.values[1]:,.0f}")
    print(f"  I: {initial_state.values[2]:,.0f}")
    print(f"  R: {initial_state.values[3]:,.0f}")
    
    # Generate ensemble forecast
    print(f"\nGenerating {forecast_days}-day ensemble forecast...")
    
    n_ensemble = 500
    forecasts = np.zeros((n_ensemble, forecast_days, 4))  # [ensemble, time, compartment]
    
    for ens in range(n_ensemble):
        state = State(values=initial_state.values.copy(), timestamp=initial_state.timestamp)
        
        for day in range(forecast_days):
            next_states = module.transition(
                state=state,
                params=params,
                interventions=[],
                dt=1.0,
                n_samples=1
            )
            state = next_states[0]
            forecasts[ens, day] = state.values
    
    # Compute statistics
    forecast_mean = np.mean(forecasts, axis=0)
    forecast_std = np.std(forecasts, axis=0)
    forecast_q05 = np.percentile(forecasts, 5, axis=0)
    forecast_q95 = np.percentile(forecasts, 95, axis=0)
    
    # Compare to truth
    true_I = np.array(true_states["I"][n_training + 1:n_training + 1 + forecast_days])
    forecast_I = forecast_mean[:, 2]
    
    print(f"\nForecast validation (Infectious compartment):")
    print(f"  {'Day':>4} {'Forecast':>10} {'90% CI':>20} {'True':>10} {'In CI?':>8}")
    print(f"  {'-'*56}")
    
    in_ci = 0
    for day in range(min(forecast_days, len(true_I))):
        f_val = forecast_I[day]
        q05 = forecast_q05[day, 2]
        q95 = forecast_q95[day, 2]
        true_val = true_I[day]
        
        covered = q05 <= true_val <= q95
        if covered:
            in_ci += 1
        
        print(f"  {day + 1:>4} {f_val:>10,.0f} [{q05:>8,.0f}, {q95:>8,.0f}] {true_val:>10,.0f} {'✓' if covered else '✗':>8}")
    
    coverage = in_ci / min(forecast_days, len(true_I)) * 100
    print(f"\n  90% CI coverage: {coverage:.1f}%")
    
    # Compute forecast metrics
    mae = np.mean(np.abs(forecast_I[:len(true_I)] - true_I))
    rmse = np.sqrt(np.mean((forecast_I[:len(true_I)] - true_I) ** 2))
    
    print(f"\n  Mean Absolute Error: {mae:,.1f}")
    print(f"  Root Mean Square Error: {rmse:,.1f}")
    
    return {
        "forecast_mean": forecast_mean,
        "forecast_std": forecast_std,
        "forecast_q05": forecast_q05,
        "forecast_q95": forecast_q95,
        "truth": true_I
    }


def run_scenario_analysis_demo(data: dict, fit_params: np.ndarray, n_training: int = 30):
    """
    Demonstrate scenario analysis with interventions.
    """
    print("\n" + "=" * 60)
    print("SCENARIO ANALYSIS DEMO")
    print("=" * 60)
    
    from updff.core.state import Intervention
    
    population = data["population"]
    
    # Create module
    module = create_disease_module(
        pathogen="covid",
        population=population,
        model_type="seir"
    )
    
    # Extract parameters
    beta, sigma, gamma, reporting = fit_params
    
    # Initial state
    true_states = data["states"]
    initial_state = State(
        values=np.array([
            true_states["S"][n_training],
            true_states["E"][n_training],
            true_states["I"][n_training],
            true_states["R"][n_training]
        ]),
        timestamp=float(n_training)
    )
    
    params = Parameters(
        values=np.array([beta, sigma, gamma, data["true_params"]["R0"], 5.0, reporting]),
        names=["beta", "sigma", "gamma", "R0", "k", "reporting_rate"]
    )
    
    forecast_days = 60
    
    # Define scenarios
    scenarios = {
        "baseline": [],
        "mild_intervention": [
            Intervention(
                intervention_type="social_distancing",
                start_time=float(n_training),
                end_time=float(n_training + forecast_days),
                magnitude=0.3,  # 30% contact reduction
                target_variable="beta"
            )
        ],
        "strong_intervention": [
            Intervention(
                intervention_type="social_distancing",
                start_time=float(n_training),
                end_time=float(n_training + forecast_days),
                magnitude=0.6,  # 60% contact reduction
                target_variable="beta"
            )
        ],
    }
    
    print("\nScenarios:")
    print("  1. Baseline: No intervention")
    print("  2. Mild intervention: 30% contact reduction")
    print("  3. Strong intervention: 60% contact reduction")
    
    # Run forecasts for each scenario
    results = {}
    n_ensemble = 200
    
    print(f"\nRunning {forecast_days}-day forecasts for each scenario...")
    
    for scenario_name, interventions in scenarios.items():
        forecasts = np.zeros((n_ensemble, forecast_days))
        
        for ens in range(n_ensemble):
            state = State(values=initial_state.values.copy(), timestamp=initial_state.timestamp)
            
            for day in range(forecast_days):
                next_states = module.transition(
                    state=state,
                    params=params,
                    interventions=interventions,
                    dt=1.0,
                    n_samples=1
                )
                state = next_states[0]
                forecasts[ens, day] = state.values[2]  # Infectious
        
        results[scenario_name] = {
            "mean": np.mean(forecasts, axis=0),
            "q05": np.percentile(forecasts, 5, axis=0),
            "q95": np.percentile(forecasts, 95, axis=0),
            "peak": np.max(np.mean(forecasts, axis=0)),
            "peak_day": np.argmax(np.mean(forecasts, axis=0)),
            "cumulative": np.sum(np.mean(forecasts, axis=0))
        }
    
    print(f"\nScenario comparison:")
    print(f"  {'Scenario':<20} {'Peak Infections':>15} {'Peak Day':>10} {'Cumulative':>15}")
    print(f"  {'-'*60}")
    
    for name, result in results.items():
        print(f"  {name:<20} {result['peak']:>15,.0f} {result['peak_day']:>10} {result['cumulative']:>15,.0f}")
    
    # Reduction analysis
    baseline_peak = results["baseline"]["peak"]
    print(f"\nIntervention impact:")
    for name in ["mild_intervention", "strong_intervention"]:
        reduction = (baseline_peak - results[name]["peak"]) / baseline_peak * 100
        print(f"  {name}: {reduction:.1f}% peak reduction")
    
    return results


def main():
    """Main entry point for demonstration."""
    print("=" * 60)
    print("UNIVERSAL PROBABILISTIC DISASTER FORECASTING FRAMEWORK")
    print("Disease Module Demonstration")
    print("=" * 60)
    
    # Generate synthetic data
    print("\nGenerating synthetic outbreak data...")
    data = generate_synthetic_outbreak(
        population=100_000,
        r0=2.5,
        infectious_period=7.0,
        n_days=90,
        initial_infected=10,
        reporting_rate=0.3,
        seed=42
    )
    
    print(f"Generated {len(data['reported_cases'])} days of outbreak data")
    print(f"Total reported cases: {sum(data['reported_cases']):,}")
    print(f"Peak day: {np.argmax(data['true_incidence'])}")
    print(f"Peak incidence: {max(data['true_incidence']):,}")
    
    # Run demonstrations
    n_training = 30
    
    # 1. Parameter estimation
    fit_result, mcmc_result = run_parameter_estimation_demo(data, n_training)
    
    # 2. Probabilistic forecasting
    forecast_result = run_forecasting_demo(
        data, 
        fit_result.optimal_params,
        n_training=n_training,
        forecast_days=14
    )
    
    # 3. Scenario analysis
    scenario_results = run_scenario_analysis_demo(
        data,
        fit_result.optimal_params,
        n_training=n_training
    )
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("""
Key capabilities demonstrated:
1. ✓ Parameter estimation from noisy surveillance data
2. ✓ Uncertainty quantification via MCMC
3. ✓ Probabilistic forecasting with credible intervals
4. ✓ Scenario analysis comparing intervention strategies

The framework provides:
- Pathogen-agnostic disease modeling
- Bayesian parameter inference
- Ensemble-based uncertainty propagation
- Decision support for public health planning
""")


if __name__ == "__main__":
    main()
