"""
Main forecasting engine that orchestrates all components.

This module provides the high-level API for the forecasting framework,
integrating hazard modules, inference, ensemble execution, and validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from updff.core.state import (
    State, StateSpaceSpec, Parameters, Observation, 
    Intervention, ObservationSeries
)
from updff.core.distribution import Distribution, Empirical, Normal
from updff.core.uncertainty import UncertaintyPropagator, MonteCarloUncertainty
from updff.core.ensemble import EnsembleExecutor, EnsembleConfig, EnsembleResult
from updff.core.scenario import Scenario, ScenarioManager

logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """
    Complete probabilistic forecast output.
    
    Contains all information needed for decision support including
    point estimates, uncertainty quantification, and scenario comparisons.
    """
    
    # Temporal metadata
    forecast_date: Union[datetime, float]
    horizon_days: int
    timestamps: np.ndarray
    
    # Point estimates
    mean_forecast: np.ndarray           # Shape: (horizon, state_dim)
    median_forecast: np.ndarray
    
    # Uncertainty quantification
    std_forecast: np.ndarray
    credible_interval_50: Tuple[np.ndarray, np.ndarray]  # 25th, 75th percentile
    credible_interval_90: Tuple[np.ndarray, np.ndarray]  # 5th, 95th percentile
    credible_interval_95: Tuple[np.ndarray, np.ndarray]  # 2.5th, 97.5th percentile
    
    # Full posterior
    ensemble_trajectories: np.ndarray   # Shape: (n_samples, horizon, state_dim)
    
    # Risk metrics
    peak_timing_distribution: Optional[Distribution] = None
    peak_magnitude_distribution: Optional[Distribution] = None
    cumulative_distribution: Optional[Distribution] = None
    
    # Calibration metadata
    parameter_posteriors: Dict[str, Distribution] = field(default_factory=dict)
    convergence_diagnostics: Dict[str, float] = field(default_factory=dict)
    
    # Scenario comparison (if multiple scenarios)
    scenario_name: str = "default"
    
    def get_forecast_at(self, day: int, state_idx: int = 0) -> Dict[str, float]:
        """Get forecast summary for a specific day."""
        return {
            "mean": float(self.mean_forecast[day, state_idx]),
            "median": float(self.median_forecast[day, state_idx]),
            "std": float(self.std_forecast[day, state_idx]),
            "ci_50_lower": float(self.credible_interval_50[0][day, state_idx]),
            "ci_50_upper": float(self.credible_interval_50[1][day, state_idx]),
            "ci_90_lower": float(self.credible_interval_90[0][day, state_idx]),
            "ci_90_upper": float(self.credible_interval_90[1][day, state_idx]),
            "ci_95_lower": float(self.credible_interval_95[0][day, state_idx]),
            "ci_95_upper": float(self.credible_interval_95[1][day, state_idx]),
        }
    
    def exceedance_probability(
        self, 
        threshold: float, 
        state_idx: int = 0
    ) -> np.ndarray:
        """Compute P(X > threshold) at each time step."""
        exceeded = self.ensemble_trajectories[:, :, state_idx] > threshold
        return np.mean(exceeded, axis=0)
    
    def time_to_threshold(
        self, 
        threshold: float, 
        state_idx: int = 0
    ) -> Distribution:
        """Distribution of time to reach threshold."""
        crossing_times = []
        for traj in self.ensemble_trajectories:
            crossings = np.where(traj[:, state_idx] > threshold)[0]
            if len(crossings) > 0:
                crossing_times.append(crossings[0])
            else:
                crossing_times.append(np.nan)
        
        valid_times = [t for t in crossing_times if not np.isnan(t)]
        if len(valid_times) == 0:
            return Normal(loc=np.inf, scale=0)
        return Empirical(np.array(valid_times))
    
    def summarize(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Forecast Summary: {self.scenario_name}",
            "=" * 50,
            f"Forecast date: {self.forecast_date}",
            f"Horizon: {self.horizon_days} days",
            f"Ensemble size: {self.ensemble_trajectories.shape[0]}",
            "",
            "End-of-horizon forecasts (state 0):",
        ]
        
        final = self.get_forecast_at(-1, 0)
        lines.append(f"  Mean: {final['mean']:.2f}")
        lines.append(f"  Median: {final['median']:.2f}")
        lines.append(f"  95% CI: [{final['ci_95_lower']:.2f}, {final['ci_95_upper']:.2f}]")
        
        if self.peak_magnitude_distribution is not None:
            peak_mean = float(self.peak_magnitude_distribution.mean())
            peak_std = float(self.peak_magnitude_distribution.std())
            lines.append(f"\nPeak magnitude: {peak_mean:.2f} ± {peak_std:.2f}")
        
        if self.peak_timing_distribution is not None:
            time_mean = float(self.peak_timing_distribution.mean())
            time_std = float(self.peak_timing_distribution.std())
            lines.append(f"Peak timing: day {time_mean:.1f} ± {time_std:.1f}")
        
        return "\n".join(lines)


@dataclass
class CalibrationResult:
    """
    Result of model calibration to historical data.
    
    Attributes:
        parameter_posterior: Posterior distribution over parameters
        convergence_diagnostics: MCMC convergence statistics
        calibration_period: Time range used for calibration
        held_out_performance: Performance on held-out data (if any)
    """
    
    parameter_posterior: Parameters
    convergence_diagnostics: Dict[str, float]
    calibration_period: Tuple[Union[datetime, float], Union[datetime, float]]
    held_out_performance: Optional[Dict[str, float]] = None
    trace: Optional[Any] = None  # MCMC trace for diagnostics


class ForecastingEngine:
    """
    Main forecasting engine that orchestrates all components.
    
    Integrates:
    - Hazard module (disease, flood, etc.)
    - Inference engine (Bayesian parameter estimation)
    - Ensemble execution
    - Scenario comparison
    - Validation
    """
    
    def __init__(
        self,
        hazard_module: 'HazardModule',
        ensemble_size: int = 1000,
        inference_method: Optional['InferenceEngine'] = None,
        propagation_method: Optional[UncertaintyPropagator] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize forecasting engine.
        
        Args:
            hazard_module: The hazard-specific module implementing dynamics
            ensemble_size: Number of ensemble members for forecasting
            inference_method: Method for parameter inference (default: MCMC)
            propagation_method: Uncertainty propagation method
            seed: Random seed for reproducibility
        """
        self.hazard = hazard_module
        self.ensemble_size = ensemble_size
        self.inference = inference_method
        self.propagator = propagation_method or MonteCarloUncertainty(n_samples=ensemble_size)
        self.rng = np.random.default_rng(seed)
        
        # State
        self._calibrated = False
        self._calibration_result: Optional[CalibrationResult] = None
        self._current_state: Optional[State] = None
        self._parameter_posterior: Optional[Parameters] = None
        
        # Ensemble executor
        self._executor = EnsembleExecutor(
            EnsembleConfig(n_ensemble=ensemble_size, seed=seed)
        )
        
        # Scenario manager
        self.scenarios = ScenarioManager()
        
        logger.info(f"Initialized ForecastingEngine with {type(hazard_module).__name__}")
    
    @property
    def is_calibrated(self) -> bool:
        """Whether the model has been calibrated to data."""
        return self._calibrated
    
    @property
    def state_spec(self) -> StateSpaceSpec:
        """Get state space specification from hazard module."""
        return self.hazard.get_state_spec()
    
    def calibrate(
        self,
        observations: Union[ObservationSeries, List[Observation]],
        calibration_window: Optional[int] = None,
        n_samples: int = 2000,
        n_warmup: int = 1000,
        n_chains: int = 4,
        prior_override: Optional[Dict[str, Distribution]] = None
    ) -> CalibrationResult:
        """
        Calibrate model parameters to historical observations.
        
        Performs Bayesian inference to obtain posterior distributions
        over model parameters.
        
        Args:
            observations: Historical observation data
            calibration_window: Number of most recent observations to use
            n_samples: Number of posterior samples
            n_warmup: Number of warmup/burn-in samples
            n_chains: Number of MCMC chains
            prior_override: Override default parameter priors
            
        Returns:
            CalibrationResult with parameter posteriors
        """
        logger.info("Starting model calibration")
        
        # Convert to ObservationSeries if needed
        if isinstance(observations, list):
            obs_series = ObservationSeries(observations)
        else:
            obs_series = observations
        
        # Apply window if specified
        if calibration_window and len(obs_series) > calibration_window:
            obs_list = list(obs_series)[-calibration_window:]
            obs_series = ObservationSeries(obs_list)
        
        # Get priors
        priors = self.hazard.get_prior()
        if prior_override:
            priors.update(prior_override)
        
        # Run inference
        if self.inference is not None:
            result = self.inference.fit(
                model=self.hazard,
                observations=list(obs_series),
                prior=priors,
                n_samples=n_samples,
                n_warmup=n_warmup,
                n_chains=n_chains
            )
            self._parameter_posterior = result.parameter_posterior
            self._calibration_result = result
        else:
            # Simplified calibration without full inference
            # Use prior samples with likelihood weighting
            self._parameter_posterior = self._simple_calibration(obs_series, priors)
            self._calibration_result = CalibrationResult(
                parameter_posterior=self._parameter_posterior,
                convergence_diagnostics={},
                calibration_period=(obs_series.timestamps[0], obs_series.timestamps[-1])
            )
        
        self._calibrated = True
        logger.info("Calibration complete")
        
        return self._calibration_result
    
    def _simple_calibration(
        self,
        observations: ObservationSeries,
        priors: Dict[str, Distribution]
    ) -> Parameters:
        """
        Simplified calibration using importance sampling.
        
        Used when no explicit inference engine is provided.
        """
        n_samples = 10000
        
        # Sample from priors
        param_names = list(priors.keys())
        param_samples = np.zeros((n_samples, len(param_names)))
        
        for i, (name, prior) in enumerate(priors.items()):
            param_samples[:, i] = prior.sample(n_samples)
        
        # Compute log-likelihoods
        log_weights = np.zeros(n_samples)
        
        _, obs_values = observations.to_arrays()
        
        for i in range(n_samples):
            params = Parameters(
                values=param_samples[i],
                names=param_names
            )
            
            # Simple likelihood: sum of log-likelihoods for each observation
            log_lik = 0.0
            for t, obs in enumerate(observations):
                # Estimate state at observation time (simplified)
                if t == 0:
                    state = self.hazard.initialize_state(
                        {"prevalence": obs.values[0] / 1000},  # Rough estimate
                        {}
                    )
                else:
                    state = self.hazard.transition(
                        state, params, [], dt=1.0, n_samples=1
                    )[0]
                
                log_lik += self.hazard.log_likelihood(obs, state, params)
            
            log_weights[i] = log_lik
        
        # Normalize weights
        log_weights -= np.max(log_weights)  # For numerical stability
        weights = np.exp(log_weights)
        weights /= weights.sum()
        
        # Resample according to weights
        indices = self.rng.choice(n_samples, size=self.ensemble_size, p=weights)
        resampled = param_samples[indices]
        
        return Parameters(
            values=resampled,
            names=param_names,
            covariance=np.cov(resampled, rowvar=False)
        )
    
    def set_initial_state(
        self,
        initial_conditions: Dict[str, Any],
        uncertainty: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Set initial state for forecasting.
        
        Args:
            initial_conditions: Dictionary of initial state values
            uncertainty: Optional uncertainty (std) for each state variable
        """
        self._current_state = self.hazard.initialize_state(
            initial_conditions, 
            uncertainty or {}
        )
        logger.info(f"Set initial state: {initial_conditions}")
    
    def forecast(
        self,
        horizon_days: int,
        interventions: Optional[List[Intervention]] = None,
        scenarios: Optional[List[Scenario]] = None,
        dt: float = 1.0,
        start_date: Optional[Union[datetime, float]] = None
    ) -> Union[ForecastResult, Dict[str, ForecastResult]]:
        """
        Generate probabilistic forecasts.
        
        Args:
            horizon_days: Number of days to forecast
            interventions: Interventions for single scenario
            scenarios: Multiple scenarios to compare
            dt: Time step size
            start_date: Forecast start date
            
        Returns:
            ForecastResult or dict of results for multiple scenarios
        """
        if not self._calibrated and self._parameter_posterior is None:
            logger.warning("Model not calibrated. Using prior parameters.")
            # Sample from priors
            priors = self.hazard.get_prior()
            param_names = list(priors.keys())
            param_samples = np.zeros((self.ensemble_size, len(param_names)))
            for i, (name, prior) in enumerate(priors.items()):
                param_samples[:, i] = prior.sample(self.ensemble_size)
            self._parameter_posterior = Parameters(
                values=param_samples,
                names=param_names
            )
        
        if self._current_state is None:
            raise ValueError("Initial state not set. Call set_initial_state() first.")
        
        start_date = start_date or datetime.now()
        
        # Handle multiple scenarios
        if scenarios:
            return self._forecast_scenarios(
                horizon_days, scenarios, dt, start_date
            )
        
        # Single scenario forecast
        return self._forecast_single(
            horizon_days, interventions or [], dt, start_date, "default"
        )
    
    def _forecast_single(
        self,
        horizon_days: int,
        interventions: List[Intervention],
        dt: float,
        start_date: Union[datetime, float],
        scenario_name: str
    ) -> ForecastResult:
        """Generate forecast for a single scenario."""
        logger.info(f"Generating {horizon_days}-day forecast for scenario: {scenario_name}")
        
        # Execute ensemble
        result = self._executor.execute(
            initial_state=self._current_state,
            parameters=self._parameter_posterior,
            transition_fn=self._make_transition_fn(),
            n_steps=horizon_days,
            dt=dt,
            interventions=interventions,
            process_noise=None,  # Handled in hazard module
            start_time=start_date
        )
        
        # Compute additional statistics
        peak_stats = result.peak_statistics(state_idx=0)
        cumulative_stats = result.cumulative_statistics(state_idx=0)
        
        # Build forecast result
        return ForecastResult(
            forecast_date=start_date,
            horizon_days=horizon_days,
            timestamps=result.timestamps,
            mean_forecast=result.mean,
            median_forecast=result.median,
            std_forecast=result.std,
            credible_interval_50=(result.quantile_25, result.quantile_75),
            credible_interval_90=(result.quantile_05, result.quantile_95),
            credible_interval_95=result.credible_interval(0.05),
            ensemble_trajectories=result.trajectories,
            peak_timing_distribution=peak_stats["peak_time_distribution"],
            peak_magnitude_distribution=peak_stats["peak_magnitude_distribution"],
            cumulative_distribution=cumulative_stats["cumulative_distribution"],
            parameter_posteriors={
                name: Empirical(self._parameter_posterior.values[:, i])
                for i, name in enumerate(self._parameter_posterior.names or [])
            },
            convergence_diagnostics=self._calibration_result.convergence_diagnostics if self._calibration_result else {},
            scenario_name=scenario_name
        )
    
    def _forecast_scenarios(
        self,
        horizon_days: int,
        scenarios: List[Scenario],
        dt: float,
        start_date: Union[datetime, float]
    ) -> Dict[str, ForecastResult]:
        """Generate forecasts for multiple scenarios."""
        results = {}
        
        for scenario in scenarios:
            results[scenario.name] = self._forecast_single(
                horizon_days,
                scenario.interventions,
                dt,
                start_date,
                scenario.name
            )
        
        return results
    
    def _make_transition_fn(self) -> Callable:
        """Create transition function for ensemble execution."""
        def transition(state: np.ndarray, params: np.ndarray, dt: float) -> np.ndarray:
            # Wrap arrays in State/Parameters
            state_obj = State(values=state, timestamp=0)
            params_obj = Parameters(
                values=params,
                names=self._parameter_posterior.names if self._parameter_posterior else None
            )
            
            # Call hazard module transition
            next_states = self.hazard.transition(
                state_obj, params_obj, [], dt=dt, n_samples=1
            )
            
            return next_states[0].values
        
        return transition
    
    def replay_historical(
        self,
        observations: Union[ObservationSeries, List[Observation]],
        forecast_horizon: int = 14,
        step_size: int = 7
    ) -> List[Tuple[ForecastResult, np.ndarray]]:
        """
        Replay historical data with rolling forecasts.
        
        Used for validation and calibration assessment.
        
        Args:
            observations: Historical observations
            forecast_horizon: Days to forecast at each step
            step_size: Days between forecast origins
            
        Returns:
            List of (forecast, actual) tuples
        """
        if isinstance(observations, list):
            obs_series = ObservationSeries(observations)
        else:
            obs_series = observations
        
        results = []
        n_obs = len(obs_series)
        
        for origin_idx in range(0, n_obs - forecast_horizon, step_size):
            # Calibrate on data up to origin
            calibration_data = ObservationSeries(list(obs_series)[:origin_idx + 1])
            if len(calibration_data) < 7:
                continue  # Need minimum data
            
            self.calibrate(calibration_data)
            
            # Set initial state from last observation
            last_obs = obs_series[origin_idx]
            self.set_initial_state(
                {"prevalence": float(last_obs.values[0])},
                {"prevalence": 0.1}
            )
            
            # Generate forecast
            forecast = self.forecast(
                horizon_days=forecast_horizon,
                start_date=last_obs.timestamp
            )
            
            # Get actual values
            actual = np.array([
                obs_series[origin_idx + i + 1].values[0]
                for i in range(forecast_horizon)
                if origin_idx + i + 1 < n_obs
            ])
            
            results.append((forecast, actual))
        
        return results
    
    def compare_scenarios(
        self,
        baseline_name: str,
        scenario_results: Dict[str, ForecastResult],
        state_idx: int = 0
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare scenario forecasts against baseline.
        
        Args:
            baseline_name: Name of baseline scenario
            scenario_results: Results from forecast()
            state_idx: State variable to compare
            
        Returns:
            Comparison statistics for each scenario
        """
        if baseline_name not in scenario_results:
            raise ValueError(f"Baseline '{baseline_name}' not found in results")
        
        baseline = scenario_results[baseline_name]
        comparisons = {}
        
        for name, result in scenario_results.items():
            if name == baseline_name:
                continue
            
            # Paired comparison using ensemble members
            diff = (
                result.ensemble_trajectories[:, :, state_idx] -
                baseline.ensemble_trajectories[:, :, state_idx]
            )
            
            # Cumulative difference
            cumulative_diff = np.sum(diff, axis=1)
            
            # Peak difference
            baseline_peaks = np.max(baseline.ensemble_trajectories[:, :, state_idx], axis=1)
            scenario_peaks = np.max(result.ensemble_trajectories[:, :, state_idx], axis=1)
            peak_diff = scenario_peaks - baseline_peaks
            
            comparisons[name] = {
                "cumulative_reduction_mean": -float(np.mean(cumulative_diff)),
                "cumulative_reduction_std": float(np.std(cumulative_diff)),
                "cumulative_reduction_ci_95": (
                    float(np.percentile(-cumulative_diff, 2.5)),
                    float(np.percentile(-cumulative_diff, 97.5))
                ),
                "peak_reduction_mean": -float(np.mean(peak_diff)),
                "peak_reduction_std": float(np.std(peak_diff)),
                "peak_reduction_ci_95": (
                    float(np.percentile(-peak_diff, 2.5)),
                    float(np.percentile(-peak_diff, 97.5))
                ),
                "prob_better_cumulative": float(np.mean(cumulative_diff < 0)),
                "prob_better_peak": float(np.mean(peak_diff < 0)),
            }
        
        return comparisons
