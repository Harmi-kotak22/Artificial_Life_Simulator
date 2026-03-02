"""
Ensemble execution engine for parallel simulation.

This module manages the execution of ensemble forecasts, handling
parallelization, memory management, and ensemble statistics.
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from updff.core.state import State, Parameters, Observation, Intervention
from updff.core.distribution import Distribution, Empirical

logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """
    Configuration for ensemble execution.
    
    Attributes:
        n_ensemble: Number of ensemble members
        n_workers: Number of parallel workers (None = auto)
        use_multiprocessing: Use process-based parallelism
        seed: Random seed for reproducibility
        chunk_size: Number of trajectories per chunk
        store_full_trajectories: Whether to store all time steps
        trajectory_thin: Thinning factor for stored trajectories
    """
    
    n_ensemble: int = 1000
    n_workers: Optional[int] = None
    use_multiprocessing: bool = False
    seed: Optional[int] = None
    chunk_size: int = 100
    store_full_trajectories: bool = True
    trajectory_thin: int = 1


@dataclass
class EnsembleTrajectory:
    """
    Single ensemble member trajectory.
    
    Attributes:
        states: List of states over time
        timestamps: Time points
        parameters: Parameters used for this trajectory
        metadata: Additional trajectory information
    """
    
    states: List[np.ndarray]
    timestamps: List[Union[datetime, float]]
    parameters: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_array(self) -> np.ndarray:
        """Convert states to (n_times, state_dim) array."""
        return np.stack(self.states)
    
    @property
    def final_state(self) -> np.ndarray:
        """Get final state."""
        return self.states[-1]


@dataclass
class EnsembleResult:
    """
    Result of ensemble execution.
    
    Contains full ensemble of trajectories and computed statistics.
    """
    
    # Core data
    trajectories: np.ndarray  # (n_ensemble, n_times, state_dim)
    timestamps: np.ndarray
    parameters: np.ndarray    # (n_ensemble, param_dim)
    
    # Precomputed statistics
    mean: np.ndarray          # (n_times, state_dim)
    std: np.ndarray           # (n_times, state_dim)
    median: np.ndarray        # (n_times, state_dim)
    
    # Quantiles
    quantile_05: np.ndarray   # 5th percentile
    quantile_25: np.ndarray   # 25th percentile
    quantile_75: np.ndarray   # 75th percentile
    quantile_95: np.ndarray   # 95th percentile
    
    # Metadata
    n_ensemble: int
    config: EnsembleConfig
    execution_time: float
    
    def get_state_at(self, time_idx: int) -> State:
        """Get state distribution at time index."""
        return State.from_ensemble(
            self.trajectories[:, time_idx, :],
            self.timestamps[time_idx]
        )
    
    def get_distribution_at(self, time_idx: int, state_idx: int = 0) -> Distribution:
        """Get empirical distribution for a state variable at time."""
        samples = self.trajectories[:, time_idx, state_idx]
        return Empirical(samples)
    
    def credible_interval(
        self, 
        alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute (1-alpha) credible interval.
        
        Returns:
            (lower, upper) arrays of shape (n_times, state_dim)
        """
        lower = np.percentile(self.trajectories, 100 * alpha / 2, axis=0)
        upper = np.percentile(self.trajectories, 100 * (1 - alpha / 2), axis=0)
        return lower, upper
    
    def exceedance_probability(
        self,
        threshold: float,
        state_idx: int = 0
    ) -> np.ndarray:
        """
        Compute P(X > threshold) at each time.
        
        Args:
            threshold: Exceedance threshold
            state_idx: Index of state variable
            
        Returns:
            Array of exceedance probabilities (n_times,)
        """
        exceeded = self.trajectories[:, :, state_idx] > threshold
        return np.mean(exceeded, axis=0)
    
    def peak_statistics(self, state_idx: int = 0) -> Dict[str, Any]:
        """
        Compute statistics about epidemic/event peak.
        
        Returns:
            Dictionary with peak timing and magnitude distributions
        """
        # Find peak for each trajectory
        peaks = np.max(self.trajectories[:, :, state_idx], axis=1)
        peak_times = np.argmax(self.trajectories[:, :, state_idx], axis=1)
        
        return {
            "peak_magnitude_mean": np.mean(peaks),
            "peak_magnitude_std": np.std(peaks),
            "peak_magnitude_median": np.median(peaks),
            "peak_magnitude_ci_95": (np.percentile(peaks, 2.5), np.percentile(peaks, 97.5)),
            "peak_time_mean": np.mean(peak_times),
            "peak_time_std": np.std(peak_times),
            "peak_time_median": np.median(peak_times),
            "peak_time_ci_95": (np.percentile(peak_times, 2.5), np.percentile(peak_times, 97.5)),
            "peak_magnitude_distribution": Empirical(peaks),
            "peak_time_distribution": Empirical(peak_times),
        }
    
    def cumulative_statistics(self, state_idx: int = 0) -> Dict[str, Any]:
        """
        Compute cumulative (integral) statistics.
        
        For disease: total infections
        For flood: total water volume
        """
        # Simple trapezoidal integration
        dt = 1.0  # Assume unit time steps
        if len(self.timestamps) > 1:
            if isinstance(self.timestamps[0], datetime):
                dt = (self.timestamps[1] - self.timestamps[0]).total_seconds() / 86400
            else:
                dt = self.timestamps[1] - self.timestamps[0]
        
        cumulative = np.trapz(self.trajectories[:, :, state_idx], dx=dt, axis=1)
        
        return {
            "cumulative_mean": np.mean(cumulative),
            "cumulative_std": np.std(cumulative),
            "cumulative_median": np.median(cumulative),
            "cumulative_ci_95": (np.percentile(cumulative, 2.5), np.percentile(cumulative, 97.5)),
            "cumulative_distribution": Empirical(cumulative),
        }


class EnsembleExecutor:
    """
    Executes ensemble of forward simulations.
    
    Manages parallel execution, memory, and ensemble statistics.
    """
    
    def __init__(self, config: Optional[EnsembleConfig] = None):
        """
        Initialize ensemble executor.
        
        Args:
            config: Ensemble configuration
        """
        self.config = config or EnsembleConfig()
        self.rng = np.random.default_rng(self.config.seed)
    
    def execute(
        self,
        initial_state: State,
        parameters: Parameters,
        transition_fn: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
        n_steps: int,
        dt: float = 1.0,
        interventions: Optional[List[Intervention]] = None,
        process_noise: Optional[np.ndarray] = None,
        start_time: Union[datetime, float] = 0.0
    ) -> EnsembleResult:
        """
        Execute ensemble of forward simulations.
        
        Args:
            initial_state: Initial state (will be sampled if has uncertainty)
            parameters: Parameters (will be sampled for each ensemble member)
            transition_fn: State transition function f(state, params, dt) -> next_state
            n_steps: Number of time steps
            dt: Time step size
            interventions: List of interventions to apply
            process_noise: Process noise covariance
            start_time: Starting time
            
        Returns:
            EnsembleResult with trajectories and statistics
        """
        import time
        start = time.time()
        
        n_ensemble = self.config.n_ensemble
        
        # Sample initial states
        if initial_state.is_ensemble:
            # Resample to target ensemble size
            indices = self.rng.choice(
                initial_state.n_samples, n_ensemble, replace=True
            )
            initial_states = initial_state.values[indices]
        elif initial_state.covariance is not None:
            initial_states = self.rng.multivariate_normal(
                initial_state.values, initial_state.covariance, size=n_ensemble
            )
        else:
            # No uncertainty - replicate with small perturbation
            initial_states = np.tile(initial_state.values, (n_ensemble, 1))
            initial_states += self.rng.normal(0, 1e-6, initial_states.shape)
        
        # Sample parameters
        param_samples = parameters.sample(n_ensemble, self.rng).values
        
        # Initialize trajectories array
        state_dim = initial_state.dim
        trajectories = np.zeros((n_ensemble, n_steps + 1, state_dim))
        trajectories[:, 0, :] = initial_states
        
        # Generate timestamps
        timestamps = self._generate_timestamps(start_time, n_steps, dt)
        
        # Execute simulations
        logger.info(f"Executing {n_ensemble} ensemble members for {n_steps} steps")
        
        if self.config.n_workers and self.config.n_workers > 1:
            # Parallel execution
            trajectories = self._execute_parallel(
                trajectories, param_samples, transition_fn, dt,
                interventions, process_noise, timestamps
            )
        else:
            # Sequential execution
            trajectories = self._execute_sequential(
                trajectories, param_samples, transition_fn, dt,
                interventions, process_noise, timestamps
            )
        
        # Compute statistics
        mean = np.mean(trajectories, axis=0)
        std = np.std(trajectories, axis=0)
        median = np.median(trajectories, axis=0)
        
        execution_time = time.time() - start
        logger.info(f"Ensemble execution completed in {execution_time:.2f}s")
        
        return EnsembleResult(
            trajectories=trajectories,
            timestamps=np.array(timestamps),
            parameters=param_samples,
            mean=mean,
            std=std,
            median=median,
            quantile_05=np.percentile(trajectories, 5, axis=0),
            quantile_25=np.percentile(trajectories, 25, axis=0),
            quantile_75=np.percentile(trajectories, 75, axis=0),
            quantile_95=np.percentile(trajectories, 95, axis=0),
            n_ensemble=n_ensemble,
            config=self.config,
            execution_time=execution_time
        )
    
    def _execute_sequential(
        self,
        trajectories: np.ndarray,
        param_samples: np.ndarray,
        transition_fn: Callable,
        dt: float,
        interventions: Optional[List[Intervention]],
        process_noise: Optional[np.ndarray],
        timestamps: List
    ) -> np.ndarray:
        """Execute ensemble members sequentially."""
        n_ensemble = trajectories.shape[0]
        n_steps = trajectories.shape[1] - 1
        
        for i in range(n_ensemble):
            state = trajectories[i, 0, :]
            params = param_samples[i]
            
            for t in range(n_steps):
                # Apply interventions if any
                effective_params = self._apply_interventions(
                    params, interventions, timestamps[t]
                )
                
                # Transition
                next_state = transition_fn(state, effective_params, dt)
                
                # Add process noise
                if process_noise is not None:
                    next_state += self.rng.multivariate_normal(
                        np.zeros(len(state)), process_noise
                    )
                
                trajectories[i, t + 1, :] = next_state
                state = next_state
        
        return trajectories
    
    def _execute_parallel(
        self,
        trajectories: np.ndarray,
        param_samples: np.ndarray,
        transition_fn: Callable,
        dt: float,
        interventions: Optional[List[Intervention]],
        process_noise: Optional[np.ndarray],
        timestamps: List
    ) -> np.ndarray:
        """Execute ensemble members in parallel using threading."""
        n_ensemble = trajectories.shape[0]
        
        def simulate_member(args):
            idx, initial_state, params = args
            rng = np.random.default_rng(self.config.seed + idx if self.config.seed else None)
            
            state = initial_state.copy()
            member_trajectory = [state.copy()]
            
            for t in range(len(timestamps) - 1):
                effective_params = self._apply_interventions(
                    params, interventions, timestamps[t]
                )
                
                next_state = transition_fn(state, effective_params, dt)
                
                if process_noise is not None:
                    next_state += rng.multivariate_normal(
                        np.zeros(len(state)), process_noise
                    )
                
                member_trajectory.append(next_state.copy())
                state = next_state
            
            return idx, np.array(member_trajectory)
        
        # Prepare arguments
        args_list = [
            (i, trajectories[i, 0, :], param_samples[i])
            for i in range(n_ensemble)
        ]
        
        # Use thread pool (safer for numpy operations)
        n_workers = self.config.n_workers or 4
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(simulate_member, args_list))
        
        # Collect results
        for idx, trajectory in results:
            trajectories[idx] = trajectory
        
        return trajectories
    
    def _apply_interventions(
        self,
        params: np.ndarray,
        interventions: Optional[List[Intervention]],
        time: Union[datetime, float]
    ) -> np.ndarray:
        """Apply active interventions to modify parameters."""
        if interventions is None:
            return params
        
        modified_params = params.copy()
        
        for intervention in interventions:
            if intervention.is_active(time):
                # Apply intervention effect
                # This is a simplified version - actual implementation
                # would depend on hazard module
                magnitude = intervention.get_magnitude_at(time)
                # Default: reduce first parameter (e.g., transmission rate)
                modified_params[0] *= (1 - magnitude)
        
        return modified_params
    
    def _generate_timestamps(
        self,
        start_time: Union[datetime, float],
        n_steps: int,
        dt: float
    ) -> List[Union[datetime, float]]:
        """Generate list of timestamps."""
        if isinstance(start_time, datetime):
            return [start_time + timedelta(days=i * dt) for i in range(n_steps + 1)]
        else:
            return [start_time + i * dt for i in range(n_steps + 1)]


class ScenarioEnsembleExecutor(EnsembleExecutor):
    """
    Extended executor for scenario comparison.
    
    Executes multiple scenarios with shared initial conditions
    for proper comparison.
    """
    
    def execute_scenarios(
        self,
        initial_state: State,
        parameters: Parameters,
        transition_fn: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
        n_steps: int,
        scenarios: Dict[str, List[Intervention]],
        dt: float = 1.0,
        process_noise: Optional[np.ndarray] = None,
        start_time: Union[datetime, float] = 0.0,
        share_initial_conditions: bool = True
    ) -> Dict[str, EnsembleResult]:
        """
        Execute multiple scenarios for comparison.
        
        Args:
            initial_state: Initial state
            parameters: Parameters
            transition_fn: Transition function
            n_steps: Number of steps
            scenarios: Dictionary of scenario name -> interventions
            dt: Time step
            process_noise: Process noise
            start_time: Start time
            share_initial_conditions: Use same initial samples for all scenarios
            
        Returns:
            Dictionary of scenario name -> EnsembleResult
        """
        results = {}
        
        # Pre-sample initial conditions if sharing
        if share_initial_conditions:
            n_ensemble = self.config.n_ensemble
            if initial_state.is_ensemble:
                indices = self.rng.choice(
                    initial_state.n_samples, n_ensemble, replace=True
                )
                shared_initial = initial_state.values[indices]
            elif initial_state.covariance is not None:
                shared_initial = self.rng.multivariate_normal(
                    initial_state.values, initial_state.covariance, size=n_ensemble
                )
            else:
                shared_initial = np.tile(initial_state.values, (n_ensemble, 1))
            
            # Create shared initial state
            shared_state = State.from_ensemble(shared_initial, initial_state.timestamp)
            
            # Pre-sample parameters
            shared_params = parameters.sample(n_ensemble, self.rng)
        
        for scenario_name, interventions in scenarios.items():
            logger.info(f"Executing scenario: {scenario_name}")
            
            if share_initial_conditions:
                result = self.execute(
                    shared_state, shared_params, transition_fn,
                    n_steps, dt, interventions, process_noise, start_time
                )
            else:
                result = self.execute(
                    initial_state, parameters, transition_fn,
                    n_steps, dt, interventions, process_noise, start_time
                )
            
            results[scenario_name] = result
        
        return results
    
    def compare_scenarios(
        self,
        results: Dict[str, EnsembleResult],
        baseline: str,
        state_idx: int = 0
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare scenarios against baseline.
        
        Args:
            results: Scenario results
            baseline: Name of baseline scenario
            state_idx: State variable to compare
            
        Returns:
            Comparison statistics for each scenario
        """
        baseline_result = results[baseline]
        comparisons = {}
        
        for name, result in results.items():
            if name == baseline:
                continue
            
            # Compute differences (paired by ensemble member)
            diff = (
                result.trajectories[:, :, state_idx] -
                baseline_result.trajectories[:, :, state_idx]
            )
            
            # Cumulative difference
            cumulative_diff = np.sum(diff, axis=1)
            
            # Peak difference
            baseline_peaks = np.max(baseline_result.trajectories[:, :, state_idx], axis=1)
            scenario_peaks = np.max(result.trajectories[:, :, state_idx], axis=1)
            peak_diff = scenario_peaks - baseline_peaks
            
            comparisons[name] = {
                "cumulative_reduction_mean": -np.mean(cumulative_diff),
                "cumulative_reduction_std": np.std(cumulative_diff),
                "cumulative_reduction_ci_95": (
                    np.percentile(-cumulative_diff, 2.5),
                    np.percentile(-cumulative_diff, 97.5)
                ),
                "peak_reduction_mean": -np.mean(peak_diff),
                "peak_reduction_std": np.std(peak_diff),
                "peak_reduction_ci_95": (
                    np.percentile(-peak_diff, 2.5),
                    np.percentile(-peak_diff, 97.5)
                ),
                "prob_better_cumulative": np.mean(cumulative_diff < 0),
                "prob_better_peak": np.mean(peak_diff < 0),
            }
        
        return comparisons
