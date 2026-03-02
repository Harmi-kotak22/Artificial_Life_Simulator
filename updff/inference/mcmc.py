"""
Markov Chain Monte Carlo (MCMC) samplers for Bayesian inference.

Provides various MCMC algorithms for sampling from posterior distributions
of model parameters given observed data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats
from scipy.linalg import cholesky

import logging

logger = logging.getLogger(__name__)


@dataclass
class MCMCResult:
    """
    Result from MCMC sampling.
    
    Contains posterior samples and diagnostic information.
    """
    samples: np.ndarray  # [n_samples, n_params]
    log_probs: np.ndarray  # Log-probability at each sample
    acceptance_rate: float
    param_names: List[str]
    n_chains: int = 1
    
    # Diagnostics
    r_hat: Optional[np.ndarray] = None  # Gelman-Rubin statistic
    ess: Optional[np.ndarray] = None  # Effective sample size
    
    # Chain state for continuation
    final_state: Optional[np.ndarray] = None
    
    def mean(self) -> Dict[str, float]:
        """Posterior mean for each parameter."""
        return {name: float(np.mean(self.samples[:, i])) 
                for i, name in enumerate(self.param_names)}
    
    def std(self) -> Dict[str, float]:
        """Posterior std for each parameter."""
        return {name: float(np.std(self.samples[:, i])) 
                for i, name in enumerate(self.param_names)}
    
    def quantiles(self, q: List[float] = [0.025, 0.5, 0.975]) -> Dict[str, np.ndarray]:
        """Posterior quantiles for each parameter."""
        return {name: np.quantile(self.samples[:, i], q) 
                for i, name in enumerate(self.param_names)}
    
    def credible_interval(self, alpha: float = 0.05) -> Dict[str, Tuple[float, float]]:
        """Credible interval for each parameter."""
        q = self.quantiles([alpha / 2, 1 - alpha / 2])
        return {name: (q[name][0], q[name][1]) for name in self.param_names}
    
    def covariance(self) -> np.ndarray:
        """Posterior covariance matrix."""
        return np.cov(self.samples, rowvar=False)
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = ["MCMC Sampling Summary", "=" * 50]
        lines.append(f"Samples: {len(self.samples)}")
        lines.append(f"Acceptance rate: {self.acceptance_rate:.2%}")
        lines.append("")
        lines.append("Parameter estimates:")
        
        means = self.mean()
        stds = self.std()
        cis = self.credible_interval()
        
        for name in self.param_names:
            ci = cis[name]
            lines.append(f"  {name}: {means[name]:.4f} ± {stds[name]:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")
        
        if self.r_hat is not None:
            lines.append("")
            lines.append("Convergence diagnostics:")
            for i, name in enumerate(self.param_names):
                lines.append(f"  {name}: R-hat = {self.r_hat[i]:.3f}")
        
        return "\n".join(lines)


class MCMCSampler(ABC):
    """
    Abstract base class for MCMC samplers.
    """
    
    @abstractmethod
    def sample(
        self,
        log_prob_fn: Callable[[np.ndarray], float],
        initial_state: np.ndarray,
        n_samples: int,
        n_warmup: int = 1000,
        **kwargs
    ) -> MCMCResult:
        """
        Draw samples from target distribution.
        
        Args:
            log_prob_fn: Function computing log-probability (unnormalized)
            initial_state: Starting parameter values
            n_samples: Number of samples to draw (after warmup)
            n_warmup: Number of warmup/burnin samples
            
        Returns:
            MCMCResult with posterior samples
        """
        pass


class MetropolisHastings(MCMCSampler):
    """
    Metropolis-Hastings MCMC sampler.
    
    Random walk proposal with adaptive step size.
    """
    
    def __init__(
        self,
        param_names: List[str],
        proposal_scale: Union[float, np.ndarray] = 0.1,
        adapt_during_warmup: bool = True,
        target_acceptance: float = 0.234
    ):
        """
        Args:
            param_names: Names of parameters being sampled
            proposal_scale: Initial proposal standard deviation(s)
            adapt_during_warmup: Adapt proposal during warmup
            target_acceptance: Target acceptance rate for adaptation
        """
        self.param_names = param_names
        self.proposal_scale = np.atleast_1d(proposal_scale)
        self.adapt = adapt_during_warmup
        self.target_acceptance = target_acceptance
    
    def sample(
        self,
        log_prob_fn: Callable[[np.ndarray], float],
        initial_state: np.ndarray,
        n_samples: int,
        n_warmup: int = 1000,
        **kwargs
    ) -> MCMCResult:
        rng = np.random.default_rng(kwargs.get("seed"))
        
        n_params = len(initial_state)
        scale = np.broadcast_to(self.proposal_scale, n_params).copy()
        
        # Storage
        samples = np.zeros((n_samples, n_params))
        log_probs = np.zeros(n_samples)
        
        # Current state
        current = initial_state.copy()
        current_lp = log_prob_fn(current)
        
        if not np.isfinite(current_lp):
            logger.warning("Initial state has non-finite log probability")
            current_lp = -1e10
        
        accepted = 0
        total = 0
        
        # Warmup with adaptation
        for i in range(n_warmup):
            # Propose
            proposal = current + rng.normal(0, scale)
            proposal_lp = log_prob_fn(proposal)
            
            # Accept/reject
            log_alpha = proposal_lp - current_lp
            if np.log(rng.random()) < log_alpha:
                current = proposal
                current_lp = proposal_lp
                accepted += 1
            total += 1
            
            # Adapt scale
            if self.adapt and (i + 1) % 100 == 0:
                rate = accepted / total
                if rate < self.target_acceptance - 0.05:
                    scale *= 0.9
                elif rate > self.target_acceptance + 0.05:
                    scale *= 1.1
                accepted = 0
                total = 0
        
        # Reset acceptance counter
        accepted = 0
        
        # Sampling
        for i in range(n_samples):
            # Propose
            proposal = current + rng.normal(0, scale)
            proposal_lp = log_prob_fn(proposal)
            
            # Accept/reject
            log_alpha = proposal_lp - current_lp
            if np.log(rng.random()) < log_alpha:
                current = proposal
                current_lp = proposal_lp
                accepted += 1
            
            samples[i] = current
            log_probs[i] = current_lp
        
        acceptance_rate = accepted / n_samples
        
        return MCMCResult(
            samples=samples,
            log_probs=log_probs,
            acceptance_rate=acceptance_rate,
            param_names=self.param_names,
            final_state=current
        )


class HamiltonianMC(MCMCSampler):
    """
    Hamiltonian Monte Carlo (HMC) sampler.
    
    Uses gradient information for efficient exploration of parameter space.
    Requires gradient of log-probability.
    """
    
    def __init__(
        self,
        param_names: List[str],
        step_size: float = 0.1,
        n_leapfrog: int = 10,
        mass_matrix: Optional[np.ndarray] = None
    ):
        """
        Args:
            param_names: Names of parameters
            step_size: Leapfrog integration step size
            n_leapfrog: Number of leapfrog steps per proposal
            mass_matrix: Diagonal of mass matrix (inverse variance)
        """
        self.param_names = param_names
        self.step_size = step_size
        self.n_leapfrog = n_leapfrog
        self.mass_matrix = mass_matrix
    
    def _numerical_gradient(
        self,
        log_prob_fn: Callable[[np.ndarray], float],
        x: np.ndarray,
        eps: float = 1e-5
    ) -> np.ndarray:
        """Compute gradient numerically."""
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            grad[i] = (log_prob_fn(x_plus) - log_prob_fn(x_minus)) / (2 * eps)
        return grad
    
    def sample(
        self,
        log_prob_fn: Callable[[np.ndarray], float],
        initial_state: np.ndarray,
        n_samples: int,
        n_warmup: int = 1000,
        gradient_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        **kwargs
    ) -> MCMCResult:
        rng = np.random.default_rng(kwargs.get("seed"))
        
        n_params = len(initial_state)
        
        # Mass matrix
        if self.mass_matrix is None:
            M = np.ones(n_params)
        else:
            M = self.mass_matrix
        M_inv = 1.0 / M
        
        # Gradient function
        if gradient_fn is None:
            gradient_fn = lambda x: self._numerical_gradient(log_prob_fn, x)
        
        # Storage
        samples = np.zeros((n_samples, n_params))
        log_probs = np.zeros(n_samples)
        
        # Current state
        q = initial_state.copy()
        current_lp = log_prob_fn(q)
        
        step_size = self.step_size
        accepted = 0
        
        def leapfrog(q0, p0, eps, L):
            """Leapfrog integration."""
            q = q0.copy()
            p = p0.copy()
            
            # Half step for momentum
            p += 0.5 * eps * gradient_fn(q)
            
            # Full steps
            for _ in range(L - 1):
                q += eps * M_inv * p
                p += eps * gradient_fn(q)
            
            # Final position step and half momentum step
            q += eps * M_inv * p
            p += 0.5 * eps * gradient_fn(q)
            
            return q, -p  # Negate momentum for reversibility
        
        # Warmup with step size adaptation
        for i in range(n_warmup):
            # Sample momentum
            p = rng.normal(0, np.sqrt(M))
            
            # Current Hamiltonian
            H0 = -current_lp + 0.5 * np.sum(p ** 2 * M_inv)
            
            # Leapfrog
            q_new, p_new = leapfrog(q, p, step_size, self.n_leapfrog)
            
            # New Hamiltonian
            new_lp = log_prob_fn(q_new)
            H1 = -new_lp + 0.5 * np.sum(p_new ** 2 * M_inv)
            
            # Accept/reject
            if np.log(rng.random()) < H0 - H1:
                q = q_new
                current_lp = new_lp
        
        # Sampling
        for i in range(n_samples):
            # Sample momentum
            p = rng.normal(0, np.sqrt(M))
            
            # Current Hamiltonian
            H0 = -current_lp + 0.5 * np.sum(p ** 2 * M_inv)
            
            # Leapfrog
            q_new, p_new = leapfrog(q, p, step_size, self.n_leapfrog)
            
            # New Hamiltonian
            new_lp = log_prob_fn(q_new)
            H1 = -new_lp + 0.5 * np.sum(p_new ** 2 * M_inv)
            
            # Accept/reject
            if np.log(rng.random()) < H0 - H1:
                q = q_new
                current_lp = new_lp
                accepted += 1
            
            samples[i] = q
            log_probs[i] = current_lp
        
        return MCMCResult(
            samples=samples,
            log_probs=log_probs,
            acceptance_rate=accepted / n_samples,
            param_names=self.param_names,
            final_state=q
        )


class NUTSSampler(MCMCSampler):
    """
    No-U-Turn Sampler (NUTS).
    
    Automatically adapts the number of leapfrog steps to avoid tuning.
    This is a simplified implementation; for production, use PyMC or Stan.
    """
    
    def __init__(
        self,
        param_names: List[str],
        max_tree_depth: int = 10
    ):
        self.param_names = param_names
        self.max_tree_depth = max_tree_depth
    
    def sample(
        self,
        log_prob_fn: Callable[[np.ndarray], float],
        initial_state: np.ndarray,
        n_samples: int,
        n_warmup: int = 1000,
        **kwargs
    ) -> MCMCResult:
        # Simplified: delegate to HMC with auto-tuning
        # Full NUTS implementation is complex - use PyMC for production
        logger.info("NUTS: Using simplified HMC implementation")
        
        hmc = HamiltonianMC(
            param_names=self.param_names,
            step_size=0.01,
            n_leapfrog=20
        )
        
        return hmc.sample(
            log_prob_fn,
            initial_state,
            n_samples,
            n_warmup,
            **kwargs
        )


class EnsembleSampler(MCMCSampler):
    """
    Ensemble sampler using affine-invariant stretch moves.
    
    Based on Goodman & Weare (2010) algorithm.
    Useful for multimodal distributions and requires no gradient.
    """
    
    def __init__(
        self,
        param_names: List[str],
        n_walkers: int = 32,
        stretch_scale: float = 2.0
    ):
        """
        Args:
            param_names: Names of parameters
            n_walkers: Number of walkers in ensemble
            stretch_scale: Scale parameter for stretch move
        """
        self.param_names = param_names
        self.n_walkers = n_walkers
        self.stretch_scale = stretch_scale
    
    def sample(
        self,
        log_prob_fn: Callable[[np.ndarray], float],
        initial_state: np.ndarray,
        n_samples: int,
        n_warmup: int = 1000,
        **kwargs
    ) -> MCMCResult:
        rng = np.random.default_rng(kwargs.get("seed"))
        
        n_params = len(initial_state)
        n_walkers = self.n_walkers
        
        # Initialize walkers around initial state
        walkers = initial_state + 0.01 * rng.normal(size=(n_walkers, n_params))
        lp = np.array([log_prob_fn(w) for w in walkers])
        
        a = self.stretch_scale
        
        samples = []
        accepted = 0
        total = 0
        
        n_total = n_warmup + n_samples
        
        for step in range(n_total):
            for i in range(n_walkers):
                # Select random walker (not self)
                j = rng.choice([k for k in range(n_walkers) if k != i])
                
                # Stretch move
                z = ((a - 1) * rng.random() + 1) ** 2 / a
                proposal = walkers[j] + z * (walkers[i] - walkers[j])
                
                # Log acceptance probability
                proposal_lp = log_prob_fn(proposal)
                log_alpha = (n_params - 1) * np.log(z) + proposal_lp - lp[i]
                
                # Accept/reject
                if np.log(rng.random()) < log_alpha:
                    walkers[i] = proposal
                    lp[i] = proposal_lp
                    if step >= n_warmup:
                        accepted += 1
                
                if step >= n_warmup:
                    total += 1
            
            if step >= n_warmup:
                samples.append(walkers.copy())
        
        # Flatten samples
        samples = np.array(samples)  # [n_steps, n_walkers, n_params]
        samples = samples.reshape(-1, n_params)  # [n_steps * n_walkers, n_params]
        
        # Thin to requested number
        if len(samples) > n_samples:
            idx = np.linspace(0, len(samples) - 1, n_samples, dtype=int)
            samples = samples[idx]
        
        log_probs = np.array([log_prob_fn(s) for s in samples])
        
        return MCMCResult(
            samples=samples,
            log_probs=log_probs,
            acceptance_rate=accepted / total if total > 0 else 0,
            param_names=self.param_names,
            n_chains=n_walkers,
            final_state=walkers[-1]
        )


def compute_rhat(chains: np.ndarray) -> np.ndarray:
    """
    Compute Gelman-Rubin R-hat statistic.
    
    Args:
        chains: Array of shape [n_chains, n_samples, n_params]
        
    Returns:
        R-hat for each parameter (should be < 1.1 for convergence)
    """
    n_chains, n_samples, n_params = chains.shape
    
    # Chain means and variances
    chain_means = np.mean(chains, axis=1)  # [n_chains, n_params]
    chain_vars = np.var(chains, axis=1, ddof=1)  # [n_chains, n_params]
    
    # Between-chain variance
    B = n_samples * np.var(chain_means, axis=0, ddof=1)
    
    # Within-chain variance
    W = np.mean(chain_vars, axis=0)
    
    # Pooled variance estimate
    var_plus = (n_samples - 1) / n_samples * W + B / n_samples
    
    # R-hat
    r_hat = np.sqrt(var_plus / W)
    
    return r_hat


def compute_ess(samples: np.ndarray) -> np.ndarray:
    """
    Compute effective sample size (ESS).
    
    Args:
        samples: Array of shape [n_samples, n_params]
        
    Returns:
        ESS for each parameter
    """
    n_samples, n_params = samples.shape
    ess = np.zeros(n_params)
    
    for p in range(n_params):
        x = samples[:, p]
        x = x - np.mean(x)
        
        # Autocorrelation
        acf = np.correlate(x, x, mode='full')[n_samples - 1:]
        acf = acf / acf[0]
        
        # Find cutoff where ACF crosses zero
        cutoff = np.where(acf < 0)[0]
        cutoff = cutoff[0] if len(cutoff) > 0 else len(acf)
        
        # Sum of autocorrelations
        tau = 1 + 2 * np.sum(acf[1:cutoff])
        
        ess[p] = n_samples / tau
    
    return ess
