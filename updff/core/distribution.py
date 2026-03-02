"""
Probability distribution abstractions for the forecasting framework.

This module provides a unified interface for working with probability
distributions, supporting both parametric distributions (Normal, Gamma, etc.)
and empirical distributions from samples.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar, Union

import numpy as np
from scipy import stats
from scipy.special import gammaln

T = TypeVar('T', bound=np.ndarray)


class Distribution(ABC, Generic[T]):
    """
    Abstract base class for probability distributions.
    
    All distributions must support sampling, density evaluation,
    and basic statistical summaries.
    """
    
    @abstractmethod
    def sample(self, n: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Draw n samples from the distribution.
        
        Args:
            n: Number of samples to draw
            rng: Random number generator (optional)
            
        Returns:
            Array of samples with shape (n,) or (n, dim) for multivariate
        """
        pass
    
    @abstractmethod
    def log_prob(self, value: np.ndarray) -> np.ndarray:
        """
        Compute log probability density/mass at value.
        
        Args:
            value: Point(s) at which to evaluate log density
            
        Returns:
            Log probability values
        """
        pass
    
    def prob(self, value: np.ndarray) -> np.ndarray:
        """Probability density/mass at value."""
        return np.exp(self.log_prob(value))
    
    @abstractmethod
    def mean(self) -> np.ndarray:
        """Expected value of the distribution."""
        pass
    
    @abstractmethod
    def variance(self) -> np.ndarray:
        """Variance of the distribution."""
        pass
    
    def std(self) -> np.ndarray:
        """Standard deviation."""
        return np.sqrt(self.variance())
    
    @abstractmethod
    def quantile(self, q: float) -> np.ndarray:
        """
        Compute q-th quantile.
        
        Args:
            q: Probability level in [0, 1]
            
        Returns:
            Quantile value
        """
        pass
    
    def credible_interval(self, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute (1-alpha) credible interval.
        
        Args:
            alpha: Significance level (e.g., 0.05 for 95% CI)
            
        Returns:
            Tuple of (lower, upper) bounds
        """
        lower = self.quantile(alpha / 2)
        upper = self.quantile(1 - alpha / 2)
        return lower, upper
    
    def cdf(self, value: np.ndarray) -> np.ndarray:
        """Cumulative distribution function."""
        raise NotImplementedError("CDF not implemented for this distribution")
    
    def entropy(self) -> float:
        """Differential entropy of the distribution."""
        raise NotImplementedError("Entropy not implemented for this distribution")


@dataclass
class Normal(Distribution):
    """
    Univariate or multivariate normal distribution.
    
    Attributes:
        loc: Mean (scalar or vector)
        scale: Standard deviation (scalar) or covariance matrix
    """
    
    loc: Union[float, np.ndarray]
    scale: Union[float, np.ndarray]
    
    def __post_init__(self):
        self.loc = np.atleast_1d(self.loc)
        self.scale = np.atleast_1d(self.scale)
        self._is_multivariate = self.scale.ndim == 2
    
    @property
    def dim(self) -> int:
        """Dimensionality."""
        return len(self.loc)
    
    def sample(self, n: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        if self._is_multivariate:
            return rng.multivariate_normal(self.loc, self.scale, size=n)
        else:
            return rng.normal(self.loc, self.scale, size=(n, self.dim)).squeeze()
    
    def log_prob(self, value: np.ndarray) -> np.ndarray:
        value = np.atleast_1d(value)
        if self._is_multivariate:
            return stats.multivariate_normal.logpdf(value, self.loc, self.scale)
        else:
            return stats.norm.logpdf(value, self.loc, self.scale).sum(axis=-1)
    
    def mean(self) -> np.ndarray:
        return self.loc
    
    def variance(self) -> np.ndarray:
        if self._is_multivariate:
            return np.diag(self.scale)
        return self.scale ** 2
    
    def quantile(self, q: float) -> np.ndarray:
        if self._is_multivariate:
            # For multivariate, return marginal quantiles
            return self.loc + np.sqrt(np.diag(self.scale)) * stats.norm.ppf(q)
        return stats.norm.ppf(q, self.loc, self.scale)
    
    def cdf(self, value: np.ndarray) -> np.ndarray:
        if self._is_multivariate:
            raise NotImplementedError("CDF not implemented for multivariate normal")
        return stats.norm.cdf(value, self.loc, self.scale)
    
    def entropy(self) -> float:
        if self._is_multivariate:
            return 0.5 * self.dim * (1 + np.log(2 * np.pi)) + 0.5 * np.linalg.slogdet(self.scale)[1]
        return stats.norm.entropy(self.loc, self.scale).sum()


@dataclass
class LogNormal(Distribution):
    """
    Log-normal distribution.
    
    If X ~ LogNormal(mu, sigma), then log(X) ~ Normal(mu, sigma).
    
    Attributes:
        mu: Mean of the underlying normal
        sigma: Standard deviation of the underlying normal
    """
    
    mu: float
    sigma: float
    
    def sample(self, n: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return rng.lognormal(self.mu, self.sigma, size=n)
    
    def log_prob(self, value: np.ndarray) -> np.ndarray:
        return stats.lognorm.logpdf(value, s=self.sigma, scale=np.exp(self.mu))
    
    def mean(self) -> np.ndarray:
        return np.exp(self.mu + self.sigma**2 / 2)
    
    def variance(self) -> np.ndarray:
        return (np.exp(self.sigma**2) - 1) * np.exp(2 * self.mu + self.sigma**2)
    
    def quantile(self, q: float) -> np.ndarray:
        return stats.lognorm.ppf(q, s=self.sigma, scale=np.exp(self.mu))
    
    def cdf(self, value: np.ndarray) -> np.ndarray:
        return stats.lognorm.cdf(value, s=self.sigma, scale=np.exp(self.mu))
    
    @classmethod
    def from_mean_std(cls, mean: float, std: float) -> LogNormal:
        """Create LogNormal from desired mean and standard deviation."""
        var = std ** 2
        mu = np.log(mean**2 / np.sqrt(var + mean**2))
        sigma = np.sqrt(np.log(var / mean**2 + 1))
        return cls(mu=mu, sigma=sigma)


@dataclass
class Gamma(Distribution):
    """
    Gamma distribution.
    
    Parameterized by shape (alpha) and rate (beta), or alternatively
    by mean and standard deviation.
    
    Attributes:
        alpha: Shape parameter (> 0)
        beta: Rate parameter (> 0)
    """
    
    alpha: float  # shape
    beta: float   # rate
    
    def sample(self, n: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return rng.gamma(self.alpha, 1.0 / self.beta, size=n)
    
    def log_prob(self, value: np.ndarray) -> np.ndarray:
        return stats.gamma.logpdf(value, a=self.alpha, scale=1.0 / self.beta)
    
    def mean(self) -> np.ndarray:
        return np.array(self.alpha / self.beta)
    
    def variance(self) -> np.ndarray:
        return np.array(self.alpha / self.beta**2)
    
    def quantile(self, q: float) -> np.ndarray:
        return stats.gamma.ppf(q, a=self.alpha, scale=1.0 / self.beta)
    
    def cdf(self, value: np.ndarray) -> np.ndarray:
        return stats.gamma.cdf(value, a=self.alpha, scale=1.0 / self.beta)
    
    @classmethod
    def from_mean_std(cls, mean: float, std: float) -> Gamma:
        """Create Gamma from desired mean and standard deviation."""
        var = std ** 2
        beta = mean / var
        alpha = mean * beta
        return cls(alpha=alpha, beta=beta)


@dataclass
class Beta(Distribution):
    """
    Beta distribution on [0, 1].
    
    Attributes:
        alpha: First shape parameter (> 0)
        beta: Second shape parameter (> 0)
    """
    
    alpha: float
    beta: float
    
    def sample(self, n: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return rng.beta(self.alpha, self.beta, size=n)
    
    def log_prob(self, value: np.ndarray) -> np.ndarray:
        return stats.beta.logpdf(value, self.alpha, self.beta)
    
    def mean(self) -> np.ndarray:
        return np.array(self.alpha / (self.alpha + self.beta))
    
    def variance(self) -> np.ndarray:
        ab = self.alpha + self.beta
        return np.array(self.alpha * self.beta / (ab**2 * (ab + 1)))
    
    def quantile(self, q: float) -> np.ndarray:
        return stats.beta.ppf(q, self.alpha, self.beta)
    
    def cdf(self, value: np.ndarray) -> np.ndarray:
        return stats.beta.cdf(value, self.alpha, self.beta)
    
    @classmethod
    def from_mean_std(cls, mean: float, std: float) -> Beta:
        """Create Beta from desired mean and standard deviation."""
        var = std ** 2
        common = mean * (1 - mean) / var - 1
        alpha = mean * common
        beta = (1 - mean) * common
        return cls(alpha=max(0.01, alpha), beta=max(0.01, beta))


@dataclass
class Uniform(Distribution):
    """
    Uniform distribution on [low, high].
    
    Attributes:
        low: Lower bound
        high: Upper bound
    """
    
    low: float
    high: float
    
    def sample(self, n: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return rng.uniform(self.low, self.high, size=n)
    
    def log_prob(self, value: np.ndarray) -> np.ndarray:
        return stats.uniform.logpdf(value, self.low, self.high - self.low)
    
    def mean(self) -> np.ndarray:
        return np.array((self.low + self.high) / 2)
    
    def variance(self) -> np.ndarray:
        return np.array((self.high - self.low)**2 / 12)
    
    def quantile(self, q: float) -> np.ndarray:
        return self.low + q * (self.high - self.low)
    
    def cdf(self, value: np.ndarray) -> np.ndarray:
        return stats.uniform.cdf(value, self.low, self.high - self.low)


@dataclass
class Exponential(Distribution):
    """
    Exponential distribution.
    
    Attributes:
        rate: Rate parameter (lambda > 0)
    """
    
    rate: float
    
    def sample(self, n: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return rng.exponential(1.0 / self.rate, size=n)
    
    def log_prob(self, value: np.ndarray) -> np.ndarray:
        return stats.expon.logpdf(value, scale=1.0 / self.rate)
    
    def mean(self) -> np.ndarray:
        return np.array(1.0 / self.rate)
    
    def variance(self) -> np.ndarray:
        return np.array(1.0 / self.rate**2)
    
    def quantile(self, q: float) -> np.ndarray:
        return stats.expon.ppf(q, scale=1.0 / self.rate)
    
    def cdf(self, value: np.ndarray) -> np.ndarray:
        return stats.expon.cdf(value, scale=1.0 / self.rate)


@dataclass
class NegativeBinomial(Distribution):
    """
    Negative Binomial distribution for overdispersed count data.
    
    Parameterized by mean (mu) and dispersion (k).
    Variance = mu + mu^2/k
    
    Attributes:
        mu: Mean (> 0)
        k: Dispersion parameter (> 0, smaller = more overdispersed)
    """
    
    mu: float
    k: float
    
    def sample(self, n: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        # Gamma-Poisson mixture
        rate = rng.gamma(self.k, self.mu / self.k, size=n)
        return rng.poisson(rate)
    
    def log_prob(self, value: np.ndarray) -> np.ndarray:
        value = np.atleast_1d(value)
        p = self.k / (self.k + self.mu)
        return stats.nbinom.logpmf(value, n=self.k, p=p)
    
    def mean(self) -> np.ndarray:
        return np.array(self.mu)
    
    def variance(self) -> np.ndarray:
        return np.array(self.mu + self.mu**2 / self.k)
    
    def quantile(self, q: float) -> np.ndarray:
        p = self.k / (self.k + self.mu)
        return stats.nbinom.ppf(q, n=self.k, p=p)


@dataclass
class Empirical(Distribution):
    """
    Empirical distribution from samples.
    
    Represents a distribution using Monte Carlo samples, supporting
    resampling and kernel density estimation for continuous approximation.
    
    Attributes:
        samples: Array of samples
        weights: Optional importance weights
    """
    
    samples: np.ndarray
    weights: Optional[np.ndarray] = None
    
    def __post_init__(self):
        self.samples = np.atleast_1d(np.asarray(self.samples))
        if self.weights is not None:
            self.weights = np.asarray(self.weights)
            self.weights = self.weights / self.weights.sum()  # Normalize
        self._n_samples = len(self.samples)
    
    @property
    def dim(self) -> int:
        """Dimensionality of samples."""
        if self.samples.ndim == 1:
            return 1
        return self.samples.shape[1]
    
    def sample(self, n: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        indices = rng.choice(self._n_samples, size=n, replace=True, p=self.weights)
        return self.samples[indices]
    
    def log_prob(self, value: np.ndarray) -> np.ndarray:
        """
        Approximate log probability using kernel density estimation.
        
        For discrete approximation, returns log of empirical frequency.
        """
        # Use scipy's gaussian_kde for continuous approximation
        if self.dim == 1:
            kde = stats.gaussian_kde(self.samples, weights=self.weights)
            return np.log(kde(value) + 1e-10)
        else:
            kde = stats.gaussian_kde(self.samples.T, weights=self.weights)
            return np.log(kde(value.T) + 1e-10)
    
    def mean(self) -> np.ndarray:
        if self.weights is not None:
            return np.average(self.samples, weights=self.weights, axis=0)
        return np.mean(self.samples, axis=0)
    
    def variance(self) -> np.ndarray:
        if self.weights is not None:
            mean = self.mean()
            return np.average((self.samples - mean)**2, weights=self.weights, axis=0)
        return np.var(self.samples, axis=0)
    
    def quantile(self, q: float) -> np.ndarray:
        if self.weights is not None:
            # Weighted quantile
            sorted_indices = np.argsort(self.samples, axis=0)
            sorted_samples = np.take_along_axis(self.samples, sorted_indices, axis=0)
            cumweights = np.cumsum(np.take_along_axis(
                np.broadcast_to(self.weights[:, None], self.samples.shape) if self.samples.ndim > 1 
                else self.weights, 
                sorted_indices, axis=0
            ), axis=0)
            idx = np.searchsorted(cumweights.flatten() if self.dim == 1 else cumweights[:, 0], q)
            return sorted_samples[min(idx, len(sorted_samples) - 1)]
        return np.percentile(self.samples, q * 100, axis=0)
    
    def ess(self) -> float:
        """Effective sample size accounting for weights."""
        if self.weights is None:
            return float(self._n_samples)
        return 1.0 / np.sum(self.weights**2)


@dataclass
class MixtureDistribution(Distribution):
    """
    Mixture of distributions.
    
    Attributes:
        components: List of component distributions
        weights: Mixing weights (must sum to 1)
    """
    
    components: List[Distribution]
    weights: np.ndarray
    
    def __post_init__(self):
        self.weights = np.asarray(self.weights)
        self.weights = self.weights / self.weights.sum()  # Normalize
    
    def sample(self, n: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        # Sample component indices
        indices = rng.choice(len(self.components), size=n, p=self.weights)
        # Sample from each component
        samples = []
        for i, component in enumerate(self.components):
            n_i = np.sum(indices == i)
            if n_i > 0:
                samples.append(component.sample(n_i, rng))
        return np.concatenate(samples)
    
    def log_prob(self, value: np.ndarray) -> np.ndarray:
        value = np.atleast_1d(value)
        log_probs = np.array([
            np.log(w) + comp.log_prob(value)
            for w, comp in zip(self.weights, self.components)
        ])
        return np.logaddexp.reduce(log_probs, axis=0)
    
    def mean(self) -> np.ndarray:
        return sum(w * comp.mean() for w, comp in zip(self.weights, self.components))
    
    def variance(self) -> np.ndarray:
        # Law of total variance
        means = np.array([comp.mean() for comp in self.components])
        variances = np.array([comp.variance() for comp in self.components])
        overall_mean = self.mean()
        return (
            sum(w * v for w, v in zip(self.weights, variances)) +
            sum(w * (m - overall_mean)**2 for w, m in zip(self.weights, means))
        )
    
    def quantile(self, q: float) -> np.ndarray:
        # Approximate by sampling
        samples = self.sample(10000)
        return np.percentile(samples, q * 100, axis=0)


class TruncatedDistribution(Distribution):
    """
    Truncated distribution with bounds.
    
    Attributes:
        base_distribution: Underlying distribution to truncate
        lower: Lower truncation bound
        upper: Upper truncation bound
    """
    
    def __init__(
        self,
        base_distribution: Distribution,
        lower: float = -np.inf,
        upper: float = np.inf
    ):
        self.base = base_distribution
        self.lower = lower
        self.upper = upper
        # Compute normalization constant
        if hasattr(self.base, 'cdf'):
            self._log_Z = np.log(self.base.cdf(upper) - self.base.cdf(lower) + 1e-10)
        else:
            self._log_Z = 0.0  # Approximate
    
    def sample(self, n: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        # Rejection sampling
        samples = []
        while len(samples) < n:
            candidates = self.base.sample(n * 2, rng)
            valid = (candidates >= self.lower) & (candidates <= self.upper)
            samples.extend(candidates[valid])
        return np.array(samples[:n])
    
    def log_prob(self, value: np.ndarray) -> np.ndarray:
        value = np.atleast_1d(value)
        in_bounds = (value >= self.lower) & (value <= self.upper)
        log_p = np.where(in_bounds, self.base.log_prob(value) - self._log_Z, -np.inf)
        return log_p
    
    def mean(self) -> np.ndarray:
        # Approximate by sampling
        samples = self.sample(10000)
        return np.mean(samples, axis=0)
    
    def variance(self) -> np.ndarray:
        samples = self.sample(10000)
        return np.var(samples, axis=0)
    
    def quantile(self, q: float) -> np.ndarray:
        samples = self.sample(10000)
        return np.percentile(samples, q * 100, axis=0)


def create_distribution(
    dist_type: str,
    params: Dict[str, Any]
) -> Distribution:
    """
    Factory function to create distributions from specification.
    
    Args:
        dist_type: Distribution type name
        params: Distribution parameters
        
    Returns:
        Distribution instance
    """
    distributions = {
        "normal": Normal,
        "lognormal": LogNormal,
        "gamma": Gamma,
        "beta": Beta,
        "uniform": Uniform,
        "exponential": Exponential,
        "negative_binomial": NegativeBinomial,
    }
    
    if dist_type.lower() not in distributions:
        raise ValueError(f"Unknown distribution type: {dist_type}")
    
    return distributions[dist_type.lower()](**params)
