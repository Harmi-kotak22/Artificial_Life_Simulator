"""
Validation framework for probabilistic forecast evaluation.

Provides metrics, calibration assessment, and proper scoring rules
for evaluating forecast quality and reliability.
"""

from updff.validation.metrics import (
    ForecastMetrics,
    crps,
    log_score,
    interval_score,
    brier_score,
    mean_absolute_error,
    root_mean_squared_error,
    coverage_probability,
    sharpness
)
from updff.validation.calibration import (
    CalibrationAssessment,
    pit_histogram,
    reliability_diagram,
    coverage_test,
    calibration_error
)
from updff.validation.scoring import (
    ProperScoringRule,
    CRPSScore,
    LogScore,
    IntervalScore,
    WeightedIntervalScore,
    compute_skill_score
)
from updff.validation.diagnostics import (
    ForecastDiagnostics,
    residual_analysis,
    forecast_horizon_analysis,
    ensemble_diagnostics
)

__all__ = [
    # Metrics
    "ForecastMetrics",
    "crps",
    "log_score",
    "interval_score",
    "brier_score",
    "mean_absolute_error",
    "root_mean_squared_error",
    "coverage_probability",
    "sharpness",
    # Calibration
    "CalibrationAssessment",
    "pit_histogram",
    "reliability_diagram",
    "coverage_test",
    "calibration_error",
    # Scoring
    "ProperScoringRule",
    "CRPSScore",
    "LogScore",
    "IntervalScore",
    "WeightedIntervalScore",
    "compute_skill_score",
    # Diagnostics
    "ForecastDiagnostics",
    "residual_analysis",
    "forecast_horizon_analysis",
    "ensemble_diagnostics",
]
