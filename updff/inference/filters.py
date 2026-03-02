"""
Sequential Monte Carlo (Particle) Filters.

Provides particle filtering algorithms for online state estimation
and parameter inference with streaming data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats
from scipy.special import logsumexp

import logging

logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    """Result from particle filtering."""
    
    # State estimates over time
    state_mean: np.ndarray  # [T, state_dim]
    state_std: np.ndarray   # [T, state_dim]
    state_quantiles: Dict[float, np.ndarray]  # quantile -> [T, state_dim]
    
    # Log-likelihood (marginal)
    log_likelihood: float
    
    # Effective sample size over time
    ess_history: np.ndarray  # [T]
    
    # Final particle set
    final_particles: np.ndarray  # [n_particles, state_dim]
    final_weights: np.ndarray    # [n_particles]
    
    # Parameter posteriors (if doing parameter estimation)
    param_posterior: Optional[Dict[str, np.ndarray]] = None


class ParticleFilter(ABC):
    """
    Abstract base class for particle filters.
    
    Particle filters approximate the filtering distribution P(X_t | Y_{1:t})
    using a weighted set of particles.
    """
    
    def __init__(
        self,
        n_particles: int = 1000,
        resample_threshold: float = 0.5
    ):
        """
        Args:
            n_particles: Number of particles
            resample_threshold: Resample when ESS < threshold * n_particles
        """
        self.n_particles = n_particles
        self.resample_threshold = resample_threshold
    
    @abstractmethod
    def filter(
        self,
        observations: np.ndarray,
        transition_fn: Callable[[np.ndarray, int], np.ndarray],
        observation_likelihood_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        initial_distribution: Callable[[int], np.ndarray],
        **kwargs
    ) -> FilterResult:
        """
        Run particle filter on sequence of observations.
        
        Args:
            observations: Observation sequence [T, obs_dim]
            transition_fn: State transition P(X_t | X_{t-1}), returns [n_particles, state_dim]
            observation_likelihood_fn: log P(Y_t | X_t), returns [n_particles]
            initial_distribution: Samples from P(X_0), returns [n_particles, state_dim]
            
        Returns:
            FilterResult with state estimates and diagnostics
        """
        pass
    
    def _effective_sample_size(self, weights: np.ndarray) -> float:
        """Compute effective sample size."""
        return 1.0 / np.sum(weights ** 2)
    
    def _systematic_resample(
        self,
        particles: np.ndarray,
        weights: np.ndarray,
        rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Systematic resampling."""
        n = len(weights)
        
        # Cumulative sum
        cumsum = np.cumsum(weights)
        
        # Systematic sample points
        u = (rng.random() + np.arange(n)) / n
        
        # Resample indices
        indices = np.searchsorted(cumsum, u)
        indices = np.clip(indices, 0, n - 1)
        
        # Resample particles
        new_particles = particles[indices].copy()
        new_weights = np.ones(n) / n
        
        return new_particles, new_weights


class BootstrapFilter(ParticleFilter):
    """
    Bootstrap particle filter (Sequential Importance Resampling).
    
    The simplest and most robust particle filter. Uses the transition
    distribution as the importance distribution.
    """
    
    def filter(
        self,
        observations: np.ndarray,
        transition_fn: Callable[[np.ndarray, int], np.ndarray],
        observation_likelihood_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        initial_distribution: Callable[[int], np.ndarray],
        **kwargs
    ) -> FilterResult:
        rng = np.random.default_rng(kwargs.get("seed"))
        
        T = len(observations)
        
        # Initialize particles
        particles = initial_distribution(self.n_particles)  # [N, D]
        state_dim = particles.shape[1]
        weights = np.ones(self.n_particles) / self.n_particles
        
        # Storage
        state_means = np.zeros((T, state_dim))
        state_stds = np.zeros((T, state_dim))
        quantile_storage = {q: np.zeros((T, state_dim)) for q in [0.025, 0.25, 0.5, 0.75, 0.975]}
        ess_history = np.zeros(T)
        total_log_likelihood = 0.0
        
        for t in range(T):
            # Propagate particles through transition
            if t > 0:
                particles = transition_fn(particles, t)
            
            # Compute weights from observation likelihood
            obs = observations[t]
            log_likes = observation_likelihood_fn(particles, obs)
            
            # Normalize weights
            max_ll = np.max(log_likes)
            log_weights = np.log(weights + 1e-300) + log_likes
            log_norm = logsumexp(log_weights)
            weights = np.exp(log_weights - log_norm)
            
            # Update marginal likelihood
            total_log_likelihood += log_norm
            
            # Compute state estimates
            state_means[t] = np.average(particles, weights=weights, axis=0)
            state_stds[t] = np.sqrt(np.average((particles - state_means[t]) ** 2, 
                                               weights=weights, axis=0))
            
            for q in quantile_storage:
                # Weighted quantile approximation
                sorted_idx = np.argsort(particles, axis=0)
                for d in range(state_dim):
                    cum_weights = np.cumsum(weights[sorted_idx[:, d]])
                    idx = np.searchsorted(cum_weights, q)
                    idx = min(idx, len(particles) - 1)
                    quantile_storage[q][t, d] = particles[sorted_idx[idx, d], d]
            
            # ESS
            ess = self._effective_sample_size(weights)
            ess_history[t] = ess
            
            # Resample if needed
            if ess < self.resample_threshold * self.n_particles:
                particles, weights = self._systematic_resample(particles, weights, rng)
        
        return FilterResult(
            state_mean=state_means,
            state_std=state_stds,
            state_quantiles=quantile_storage,
            log_likelihood=total_log_likelihood,
            ess_history=ess_history,
            final_particles=particles,
            final_weights=weights
        )


class AuxiliaryParticleFilter(ParticleFilter):
    """
    Auxiliary particle filter.
    
    Uses a look-ahead step to improve importance sampling when
    observations are highly informative.
    """
    
    def __init__(
        self,
        n_particles: int = 1000,
        resample_threshold: float = 0.5,
        auxiliary_weight_fn: Optional[Callable] = None
    ):
        super().__init__(n_particles, resample_threshold)
        self.auxiliary_weight_fn = auxiliary_weight_fn
    
    def filter(
        self,
        observations: np.ndarray,
        transition_fn: Callable[[np.ndarray, int], np.ndarray],
        observation_likelihood_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        initial_distribution: Callable[[int], np.ndarray],
        auxiliary_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
        **kwargs
    ) -> FilterResult:
        """
        Args:
            auxiliary_fn: Computes auxiliary weights g(y_t | x_{t-1})
                         If None, uses E[p(y_t | x_t) | x_{t-1}] approximation
        """
        rng = np.random.default_rng(kwargs.get("seed"))
        
        T = len(observations)
        
        # Initialize
        particles = initial_distribution(self.n_particles)
        state_dim = particles.shape[1]
        weights = np.ones(self.n_particles) / self.n_particles
        
        # Storage
        state_means = np.zeros((T, state_dim))
        state_stds = np.zeros((T, state_dim))
        quantile_storage = {q: np.zeros((T, state_dim)) for q in [0.025, 0.5, 0.975]}
        ess_history = np.zeros(T)
        total_log_likelihood = 0.0
        
        for t in range(T):
            obs = observations[t]
            
            if t > 0:
                # Compute auxiliary weights
                if auxiliary_fn is not None:
                    aux_log_weights = auxiliary_fn(particles, obs)
                else:
                    # Use current particle likelihood as approximation
                    aux_log_weights = observation_likelihood_fn(particles, obs)
                
                # First stage weights
                log_first = np.log(weights + 1e-300) + aux_log_weights
                log_norm = logsumexp(log_first)
                first_weights = np.exp(log_first - log_norm)
                
                # Resample with first-stage weights
                indices = rng.choice(
                    self.n_particles,
                    size=self.n_particles,
                    p=first_weights
                )
                particles = particles[indices]
                aux_used = aux_log_weights[indices]
                
                # Propagate
                particles = transition_fn(particles, t)
                
                # Second stage: correct for auxiliary weights
                log_likes = observation_likelihood_fn(particles, obs)
                log_weights = log_likes - aux_used
                log_norm = logsumexp(log_weights)
                weights = np.exp(log_weights - log_norm)
            else:
                # First time step
                log_likes = observation_likelihood_fn(particles, obs)
                log_weights = np.log(weights + 1e-300) + log_likes
                log_norm = logsumexp(log_weights)
                weights = np.exp(log_weights - log_norm)
            
            total_log_likelihood += log_norm
            
            # State estimates
            state_means[t] = np.average(particles, weights=weights, axis=0)
            state_stds[t] = np.sqrt(np.average((particles - state_means[t]) ** 2,
                                               weights=weights, axis=0))
            
            for q in quantile_storage:
                sorted_idx = np.argsort(particles, axis=0)
                for d in range(state_dim):
                    cum_weights = np.cumsum(weights[sorted_idx[:, d]])
                    idx = np.searchsorted(cum_weights, q)
                    idx = min(idx, len(particles) - 1)
                    quantile_storage[q][t, d] = particles[sorted_idx[idx, d], d]
            
            ess = self._effective_sample_size(weights)
            ess_history[t] = ess
            
            # Standard resampling if ESS low
            if ess < self.resample_threshold * self.n_particles:
                particles, weights = self._systematic_resample(particles, weights, rng)
        
        return FilterResult(
            state_mean=state_means,
            state_std=state_stds,
            state_quantiles=quantile_storage,
            log_likelihood=total_log_likelihood,
            ess_history=ess_history,
            final_particles=particles,
            final_weights=weights
        )


class LiuWestFilter(ParticleFilter):
    """
    Liu-West filter for combined state and parameter estimation.
    
    Uses kernel smoothing to maintain diversity in parameter particles
    while estimating both states and static parameters.
    """
    
    def __init__(
        self,
        n_particles: int = 1000,
        resample_threshold: float = 0.5,
        shrinkage: float = 0.99
    ):
        """
        Args:
            shrinkage: Shrinkage parameter h² = 1 - a², where a ∈ (0, 1)
                      Higher shrinkage = more regularization
        """
        super().__init__(n_particles, resample_threshold)
        self.shrinkage = shrinkage
        self.a = np.sqrt(1 - shrinkage ** 2)
    
    def filter(
        self,
        observations: np.ndarray,
        transition_fn: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
        observation_likelihood_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        initial_state_distribution: Callable[[int], np.ndarray],
        initial_param_distribution: Callable[[int], np.ndarray],
        param_names: Optional[List[str]] = None,
        **kwargs
    ) -> FilterResult:
        """
        Args:
            transition_fn: P(X_t | X_{t-1}, θ), takes (states, params, t)
            observation_likelihood_fn: log P(Y_t | X_t, θ), takes (states, params, obs)
            initial_param_distribution: Samples from P(θ)
        """
        rng = np.random.default_rng(kwargs.get("seed"))
        
        T = len(observations)
        
        # Initialize
        states = initial_state_distribution(self.n_particles)
        params = initial_param_distribution(self.n_particles)
        state_dim = states.shape[1]
        param_dim = params.shape[1]
        
        weights = np.ones(self.n_particles) / self.n_particles
        
        # Storage
        state_means = np.zeros((T, state_dim))
        state_stds = np.zeros((T, state_dim))
        ess_history = np.zeros(T)
        total_log_likelihood = 0.0
        
        param_history = []
        
        for t in range(T):
            obs = observations[t]
            
            # Propagate states given params
            if t > 0:
                states = transition_fn(states, params, t)
            
            # Compute weights
            log_likes = observation_likelihood_fn(states, params, obs)
            log_weights = np.log(weights + 1e-300) + log_likes
            log_norm = logsumexp(log_weights)
            weights = np.exp(log_weights - log_norm)
            total_log_likelihood += log_norm
            
            # State estimates
            state_means[t] = np.average(states, weights=weights, axis=0)
            state_stds[t] = np.sqrt(np.average((states - state_means[t]) ** 2,
                                               weights=weights, axis=0))
            
            ess = self._effective_sample_size(weights)
            ess_history[t] = ess
            
            # Resample if needed with Liu-West parameter perturbation
            if ess < self.resample_threshold * self.n_particles:
                # Weighted mean and variance of params
                param_mean = np.average(params, weights=weights, axis=0)
                param_var = np.average((params - param_mean) ** 2, weights=weights, axis=0)
                
                # Resample
                indices = rng.choice(self.n_particles, size=self.n_particles, p=weights)
                states = states[indices]
                
                # Perturb parameters (kernel density smoothing)
                old_params = params[indices]
                
                # Shrink toward mean
                shrunk_params = self.a * old_params + (1 - self.a) * param_mean
                
                # Add noise
                noise_std = self.shrinkage * np.sqrt(param_var)
                params = shrunk_params + rng.normal(0, noise_std, size=params.shape)
                
                weights = np.ones(self.n_particles) / self.n_particles
            
            param_history.append(params.copy())
        
        # Final parameter posterior
        final_params = params
        if param_names:
            param_posterior = {
                name: final_params[:, i] for i, name in enumerate(param_names)
            }
        else:
            param_posterior = {f"param_{i}": final_params[:, i] 
                             for i in range(param_dim)}
        
        return FilterResult(
            state_mean=state_means,
            state_std=state_stds,
            state_quantiles={0.5: state_means},  # Simplified
            log_likelihood=total_log_likelihood,
            ess_history=ess_history,
            final_particles=states,
            final_weights=weights,
            param_posterior=param_posterior
        )


class IteratedBatchImportanceSampling:
    """
    Iterated Batch Importance Sampling (IBIS) for offline parameter estimation.
    
    Sequential Monte Carlo sampler that processes observations in batches
    and uses MCMC moves to rejuvenate particles.
    """
    
    def __init__(
        self,
        n_particles: int = 1000,
        n_mcmc_moves: int = 5
    ):
        self.n_particles = n_particles
        self.n_mcmc_moves = n_mcmc_moves
    
    def run(
        self,
        observations: np.ndarray,
        log_likelihood_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        log_prior_fn: Callable[[np.ndarray], np.ndarray],
        prior_sampler: Callable[[int], np.ndarray],
        param_names: List[str],
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Run IBIS for parameter estimation.
        
        Args:
            observations: All observations [T, obs_dim]
            log_likelihood_fn: log P(Y_{1:t} | θ) for batch
            log_prior_fn: log P(θ)
            prior_sampler: Samples from P(θ)
            
        Returns:
            Dict with parameter posteriors
        """
        rng = np.random.default_rng(kwargs.get("seed"))
        
        # Initialize from prior
        particles = prior_sampler(self.n_particles)
        weights = np.ones(self.n_particles) / self.n_particles
        
        T = len(observations)
        
        for t in range(T):
            # Incremental weight update
            obs_batch = observations[:t + 1]
            
            log_likes = np.array([
                log_likelihood_fn(p, obs_batch) for p in particles
            ])
            
            log_weights = np.log(weights + 1e-300) + log_likes
            log_norm = logsumexp(log_weights)
            weights = np.exp(log_weights - log_norm)
            
            # Check ESS
            ess = 1.0 / np.sum(weights ** 2)
            
            if ess < self.n_particles / 2:
                # Resample
                indices = rng.choice(self.n_particles, size=self.n_particles, p=weights)
                particles = particles[indices]
                weights = np.ones(self.n_particles) / self.n_particles
                
                # MCMC rejuvenation
                cov = np.cov(particles, rowvar=False)
                for _ in range(self.n_mcmc_moves):
                    for i in range(self.n_particles):
                        proposal = particles[i] + rng.multivariate_normal(
                            np.zeros(len(particles[i])), 0.1 * cov
                        )
                        
                        curr_ll = log_likelihood_fn(particles[i], obs_batch) + log_prior_fn(particles[i])
                        prop_ll = log_likelihood_fn(proposal, obs_batch) + log_prior_fn(proposal)
                        
                        if np.log(rng.random()) < prop_ll - curr_ll:
                            particles[i] = proposal
        
        # Return posterior
        return {
            name: particles[:, i] for i, name in enumerate(param_names)
        }
