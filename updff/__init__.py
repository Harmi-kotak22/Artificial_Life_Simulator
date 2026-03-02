"""
Universal Probabilistic Disaster Forecasting Framework (UPDFF)

A modular, disaster-agnostic, probabilistic forecasting framework designed
as a decision-support system that learns latent dynamics from historical
data and produces calibrated uncertainty-quantified forecasts.
"""

__version__ = "0.1.0"

from updff.core.forecast import ForecastingEngine
from updff.core.state import State, StateSpaceSpec
from updff.core.distribution import Distribution
from updff.core.ensemble import EnsembleExecutor
from updff.hazards.interface import HazardModule

__all__ = [
    "ForecastingEngine",
    "State",
    "StateSpaceSpec", 
    "Distribution",
    "EnsembleExecutor",
    "HazardModule",
]
