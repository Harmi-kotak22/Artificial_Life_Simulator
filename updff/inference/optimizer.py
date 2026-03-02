"""
Optimization-based inference methods.

Provides Maximum Likelihood Estimation (MLE) and Maximum A Posteriori (MAP)
estimation for point estimates of model parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import minimize, differential_evolution, dual_annealing

import logging

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result from parameter optimization."""
    
    optimal_params: np.ndarray
    param_names: List[str]
    log_likelihood: float
    log_posterior: Optional[float] = None
    
    # Optimization diagnostics
    converged: bool = True
    n_iterations: int = 0
    n_function_evals: int = 0
    
    # Uncertainty from Hessian
    hessian: Optional[np.ndarray] = None
    param_std: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to parameter dictionary."""
        return {name: float(self.optimal_params[i]) 
                for i, name in enumerate(self.param_names)}
    
    def confidence_interval(self, alpha: float = 0.05) -> Dict[str, Tuple[float, float]]:
        """
        Compute confidence intervals from Hessian (Wald intervals).
        
        These are approximate and assume local quadratic approximation is valid.
        """
        if self.param_std is None:
            raise ValueError("No uncertainty estimate available")
        
        from scipy import stats
        z = stats.norm.ppf(1 - alpha / 2)
        
        intervals = {}
        for i, name in enumerate(self.param_names):
            val = self.optimal_params[i]
            std = self.param_std[i]
            intervals[name] = (val - z * std, val + z * std)
        
        return intervals
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = ["Optimization Result", "=" * 50]
        lines.append(f"Log-likelihood: {self.log_likelihood:.4f}")
        if self.log_posterior is not None:
            lines.append(f"Log-posterior: {self.log_posterior:.4f}")
        lines.append(f"Converged: {self.converged}")
        lines.append(f"Iterations: {self.n_iterations}")
        lines.append("")
        lines.append("Optimal parameters:")
        
        for i, name in enumerate(self.param_names):
            val = self.optimal_params[i]
            if self.param_std is not None:
                std = self.param_std[i]
                lines.append(f"  {name}: {val:.6f} ± {std:.6f}")
            else:
                lines.append(f"  {name}: {val:.6f}")
        
        return "\n".join(lines)


class MaximumLikelihood:
    """
    Maximum Likelihood Estimation (MLE).
    
    Finds parameters that maximize P(Data | θ).
    """
    
    def __init__(
        self,
        param_names: List[str],
        param_bounds: Optional[List[Tuple[float, float]]] = None,
        method: str = "L-BFGS-B"
    ):
        """
        Args:
            param_names: Names of parameters
            param_bounds: (min, max) bounds for each parameter
            method: Optimization method
        """
        self.param_names = param_names
        self.param_bounds = param_bounds
        self.method = method
    
    def fit(
        self,
        log_likelihood_fn: Callable[[np.ndarray], float],
        initial_params: np.ndarray,
        compute_hessian: bool = True,
        **kwargs
    ) -> OptimizationResult:
        """
        Find MLE parameters.
        
        Args:
            log_likelihood_fn: Function computing log P(Data | θ)
            initial_params: Starting parameter values
            compute_hessian: Compute Hessian for uncertainty estimates
            
        Returns:
            OptimizationResult with optimal parameters
        """
        # Minimize negative log-likelihood
        def objective(params):
            ll = log_likelihood_fn(params)
            if not np.isfinite(ll):
                return 1e10
            return -ll
        
        result = minimize(
            objective,
            initial_params,
            method=self.method,
            bounds=self.param_bounds,
            options={"maxiter": kwargs.get("max_iter", 1000)}
        )
        
        # Compute Hessian for uncertainty
        hessian = None
        param_std = None
        
        if compute_hessian and result.success:
            try:
                hessian = self._numerical_hessian(objective, result.x)
                # Standard errors from inverse Hessian
                cov = np.linalg.inv(hessian)
                param_std = np.sqrt(np.diag(cov))
            except Exception as e:
                logger.warning(f"Hessian computation failed: {e}")
        
        return OptimizationResult(
            optimal_params=result.x,
            param_names=self.param_names,
            log_likelihood=-result.fun,
            converged=result.success,
            n_iterations=result.nit if hasattr(result, 'nit') else 0,
            n_function_evals=result.nfev,
            hessian=hessian,
            param_std=param_std
        )
    
    def fit_global(
        self,
        log_likelihood_fn: Callable[[np.ndarray], float],
        bounds: List[Tuple[float, float]],
        method: str = "differential_evolution",
        **kwargs
    ) -> OptimizationResult:
        """
        Global optimization for MLE (useful for multimodal likelihoods).
        
        Args:
            bounds: Parameter bounds (required for global optimization)
            method: "differential_evolution" or "dual_annealing"
        """
        def objective(params):
            ll = log_likelihood_fn(params)
            if not np.isfinite(ll):
                return 1e10
            return -ll
        
        if method == "differential_evolution":
            result = differential_evolution(
                objective,
                bounds,
                maxiter=kwargs.get("max_iter", 1000),
                seed=kwargs.get("seed")
            )
        elif method == "dual_annealing":
            result = dual_annealing(
                objective,
                bounds,
                maxiter=kwargs.get("max_iter", 1000),
                seed=kwargs.get("seed")
            )
        else:
            raise ValueError(f"Unknown global optimization method: {method}")
        
        return OptimizationResult(
            optimal_params=result.x,
            param_names=self.param_names,
            log_likelihood=-result.fun,
            converged=result.success if hasattr(result, 'success') else True,
            n_iterations=result.nit if hasattr(result, 'nit') else 0,
            n_function_evals=result.nfev
        )
    
    def _numerical_hessian(
        self,
        func: Callable[[np.ndarray], float],
        x: np.ndarray,
        eps: float = 1e-5
    ) -> np.ndarray:
        """Compute Hessian numerically."""
        n = len(x)
        hessian = np.zeros((n, n))
        
        f0 = func(x)
        
        for i in range(n):
            for j in range(i, n):
                x_pp = x.copy()
                x_pm = x.copy()
                x_mp = x.copy()
                x_mm = x.copy()
                
                x_pp[i] += eps
                x_pp[j] += eps
                x_pm[i] += eps
                x_pm[j] -= eps
                x_mp[i] -= eps
                x_mp[j] += eps
                x_mm[i] -= eps
                x_mm[j] -= eps
                
                hessian[i, j] = (func(x_pp) - func(x_pm) - func(x_mp) + func(x_mm)) / (4 * eps ** 2)
                hessian[j, i] = hessian[i, j]
        
        return hessian


class MaximumAPosteriori:
    """
    Maximum A Posteriori (MAP) estimation.
    
    Finds parameters that maximize P(θ | Data) ∝ P(Data | θ) * P(θ).
    """
    
    def __init__(
        self,
        param_names: List[str],
        param_bounds: Optional[List[Tuple[float, float]]] = None,
        method: str = "L-BFGS-B"
    ):
        self.param_names = param_names
        self.param_bounds = param_bounds
        self.method = method
    
    def fit(
        self,
        log_likelihood_fn: Callable[[np.ndarray], float],
        log_prior_fn: Callable[[np.ndarray], float],
        initial_params: np.ndarray,
        compute_hessian: bool = True,
        **kwargs
    ) -> OptimizationResult:
        """
        Find MAP parameters.
        
        Args:
            log_likelihood_fn: log P(Data | θ)
            log_prior_fn: log P(θ)
            initial_params: Starting values
            
        Returns:
            OptimizationResult with optimal parameters
        """
        def log_posterior(params):
            ll = log_likelihood_fn(params)
            lp = log_prior_fn(params)
            return ll + lp
        
        def objective(params):
            post = log_posterior(params)
            if not np.isfinite(post):
                return 1e10
            return -post
        
        result = minimize(
            objective,
            initial_params,
            method=self.method,
            bounds=self.param_bounds,
            options={"maxiter": kwargs.get("max_iter", 1000)}
        )
        
        # Uncertainty from Hessian
        hessian = None
        param_std = None
        
        if compute_hessian and result.success:
            try:
                hessian = self._numerical_hessian(objective, result.x)
                cov = np.linalg.inv(hessian)
                param_std = np.sqrt(np.diag(cov))
            except Exception as e:
                logger.warning(f"Hessian computation failed: {e}")
        
        ll = log_likelihood_fn(result.x)
        
        return OptimizationResult(
            optimal_params=result.x,
            param_names=self.param_names,
            log_likelihood=ll,
            log_posterior=-result.fun,
            converged=result.success,
            n_iterations=result.nit if hasattr(result, 'nit') else 0,
            n_function_evals=result.nfev,
            hessian=hessian,
            param_std=param_std
        )
    
    def _numerical_hessian(
        self,
        func: Callable[[np.ndarray], float],
        x: np.ndarray,
        eps: float = 1e-5
    ) -> np.ndarray:
        """Compute Hessian numerically."""
        n = len(x)
        hessian = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                x_pp = x.copy()
                x_pm = x.copy()
                x_mp = x.copy()
                x_mm = x.copy()
                
                x_pp[i] += eps
                x_pp[j] += eps
                x_pm[i] += eps
                x_pm[j] -= eps
                x_mp[i] -= eps
                x_mp[j] += eps
                x_mm[i] -= eps
                x_mm[j] -= eps
                
                hessian[i, j] = (func(x_pp) - func(x_pm) - func(x_mp) + func(x_mm)) / (4 * eps ** 2)
                hessian[j, i] = hessian[i, j]
        
        return hessian


class ProfileLikelihood:
    """
    Profile likelihood for confidence intervals and identifiability analysis.
    
    Computes profile likelihood by optimizing out nuisance parameters
    for each value of the parameter of interest.
    """
    
    def __init__(self, param_names: List[str]):
        self.param_names = param_names
    
    def compute_profile(
        self,
        log_likelihood_fn: Callable[[np.ndarray], float],
        mle_params: np.ndarray,
        profile_param_idx: int,
        profile_values: np.ndarray,
        param_bounds: Optional[List[Tuple[float, float]]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute profile likelihood.
        
        Args:
            log_likelihood_fn: Log-likelihood function
            mle_params: MLE parameter values
            profile_param_idx: Index of parameter to profile
            profile_values: Values of parameter to evaluate
            param_bounds: Bounds for optimization
            
        Returns:
            Dict with profile values and log-likelihoods
        """
        n_params = len(mle_params)
        profile_lls = np.zeros(len(profile_values))
        
        # Indices of nuisance parameters
        nuisance_idx = [i for i in range(n_params) if i != profile_param_idx]
        nuisance_bounds = None
        if param_bounds:
            nuisance_bounds = [param_bounds[i] for i in nuisance_idx]
        
        for k, val in enumerate(profile_values):
            # Fix profiled parameter, optimize others
            def constrained_objective(nuisance_params):
                params = np.zeros(n_params)
                params[profile_param_idx] = val
                for i, idx in enumerate(nuisance_idx):
                    params[idx] = nuisance_params[i]
                
                ll = log_likelihood_fn(params)
                if not np.isfinite(ll):
                    return 1e10
                return -ll
            
            # Initial guess from MLE
            initial = mle_params[nuisance_idx]
            
            result = minimize(
                constrained_objective,
                initial,
                method="L-BFGS-B",
                bounds=nuisance_bounds
            )
            
            profile_lls[k] = -result.fun
        
        return {
            "values": profile_values,
            "log_likelihood": profile_lls,
            "deviance": 2 * (np.max(profile_lls) - profile_lls)
        }
    
    def likelihood_ratio_ci(
        self,
        log_likelihood_fn: Callable[[np.ndarray], float],
        mle_params: np.ndarray,
        profile_param_idx: int,
        alpha: float = 0.05,
        param_bounds: Optional[List[Tuple[float, float]]] = None,
        n_points: int = 50
    ) -> Tuple[float, float]:
        """
        Compute likelihood-ratio based confidence interval.
        
        Uses the chi-squared approximation: deviance ≤ χ²_{1,1-α}
        """
        from scipy import stats as sp_stats
        
        # Chi-squared threshold
        threshold = sp_stats.chi2.ppf(1 - alpha, df=1)
        
        # Get bounds for profiling
        mle_val = mle_params[profile_param_idx]
        
        if param_bounds and param_bounds[profile_param_idx]:
            lb, ub = param_bounds[profile_param_idx]
        else:
            # Default search range
            lb = mle_val * 0.1
            ub = mle_val * 3.0
        
        # Search for lower bound
        profile_values = np.linspace(lb, mle_val, n_points)
        profile = self.compute_profile(
            log_likelihood_fn,
            mle_params,
            profile_param_idx,
            profile_values,
            param_bounds
        )
        
        lower_bound = profile_values[0]
        for i, dev in enumerate(profile["deviance"]):
            if dev < threshold:
                lower_bound = profile_values[i]
                break
        
        # Search for upper bound
        profile_values = np.linspace(mle_val, ub, n_points)
        profile = self.compute_profile(
            log_likelihood_fn,
            mle_params,
            profile_param_idx,
            profile_values,
            param_bounds
        )
        
        upper_bound = profile_values[-1]
        for i, dev in enumerate(profile["deviance"]):
            if dev > threshold:
                upper_bound = profile_values[i - 1] if i > 0 else profile_values[0]
                break
        
        return lower_bound, upper_bound


class GradientDescent:
    """
    Simple gradient descent optimizer with momentum.
    
    Useful when custom gradients are available.
    """
    
    def __init__(
        self,
        param_names: List[str],
        learning_rate: float = 0.01,
        momentum: float = 0.9
    ):
        self.param_names = param_names
        self.learning_rate = learning_rate
        self.momentum = momentum
    
    def fit(
        self,
        objective_fn: Callable[[np.ndarray], float],
        gradient_fn: Callable[[np.ndarray], np.ndarray],
        initial_params: np.ndarray,
        max_iter: int = 1000,
        tol: float = 1e-6,
        **kwargs
    ) -> OptimizationResult:
        """Run gradient descent."""
        params = initial_params.copy()
        velocity = np.zeros_like(params)
        
        prev_obj = objective_fn(params)
        
        for i in range(max_iter):
            grad = gradient_fn(params)
            
            # Momentum update
            velocity = self.momentum * velocity - self.learning_rate * grad
            params = params + velocity
            
            obj = objective_fn(params)
            
            # Check convergence
            if abs(prev_obj - obj) < tol:
                break
            
            prev_obj = obj
        
        return OptimizationResult(
            optimal_params=params,
            param_names=self.param_names,
            log_likelihood=-prev_obj,
            converged=True,
            n_iterations=i + 1,
            n_function_evals=i + 1
        )
