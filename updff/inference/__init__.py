"""
Inference module for Bayesian parameter learning.

Provides MCMC, particle filtering, and maximum likelihood estimation
for learning model parameters from observational data.
"""

from updff.inference.likelihood import (
    LogLikelihood,
    GaussianLikelihood,
    PoissonLikelihood,
    NegativeBinomialLikelihood,
    CompositeLogLikelihood
)
from updff.inference.mcmc import (
    MCMCSampler,
    MetropolisHastings,
    HamiltonianMC,
    NUTSSampler,
    MCMCResult
)
from updff.inference.filters import (
    ParticleFilter,
    BootstrapFilter,
    AuxiliaryParticleFilter,
    FilterResult
)
from updff.inference.optimizer import (
    MaximumLikelihood,
    MaximumAPosteriori,
    OptimizationResult
)

__all__ = [
    # Likelihood
    "LogLikelihood",
    "GaussianLikelihood",
    "PoissonLikelihood",
    "NegativeBinomialLikelihood",
    "CompositeLogLikelihood",
    # MCMC
    "MCMCSampler",
    "MetropolisHastings",
    "HamiltonianMC",
    "NUTSSampler",
    "MCMCResult",
    # Particle filters
    "ParticleFilter",
    "BootstrapFilter",
    "AuxiliaryParticleFilter",
    "FilterResult",
    # Optimization
    "MaximumLikelihood",
    "MaximumAPosteriori",
    "OptimizationResult",
]
