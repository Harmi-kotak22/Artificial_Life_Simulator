"""
Scenario management for the forecasting framework.

This module handles scenario definition, comparison, and analysis
for decision support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np

from updff.core.state import Intervention


@dataclass
class Scenario:
    """
    A scenario represents a specific set of assumptions and interventions.
    
    Attributes:
        name: Human-readable scenario name
        description: Detailed description
        interventions: List of interventions in this scenario
        parameter_overrides: Override specific parameters
        initial_condition_modifier: Modify initial conditions
        metadata: Additional scenario information
    """
    
    name: str
    description: str = ""
    interventions: List[Intervention] = field(default_factory=list)
    parameter_overrides: Dict[str, float] = field(default_factory=dict)
    initial_condition_modifier: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_intervention(self, intervention: Intervention) -> Scenario:
        """Add an intervention to the scenario (fluent interface)."""
        self.interventions.append(intervention)
        return self
    
    def with_parameter(self, name: str, value: float) -> Scenario:
        """Override a parameter value (fluent interface)."""
        self.parameter_overrides[name] = value
        return self
    
    def __repr__(self) -> str:
        return f"Scenario(name='{self.name}', interventions={len(self.interventions)})"


class ScenarioManager:
    """
    Manages scenario definitions and comparisons.
    
    Provides utilities for creating standard scenarios, comparing
    outcomes, and generating decision-relevant summaries.
    """
    
    def __init__(self):
        """Initialize scenario manager."""
        self.scenarios: Dict[str, Scenario] = {}
        self._baseline_name: Optional[str] = None
    
    def add_scenario(self, scenario: Scenario) -> None:
        """Add a scenario to the manager."""
        self.scenarios[scenario.name] = scenario
    
    def set_baseline(self, name: str) -> None:
        """Set the baseline scenario for comparisons."""
        if name not in self.scenarios:
            raise ValueError(f"Scenario '{name}' not found")
        self._baseline_name = name
    
    @property
    def baseline(self) -> Optional[Scenario]:
        """Get baseline scenario."""
        if self._baseline_name:
            return self.scenarios.get(self._baseline_name)
        return None
    
    def create_baseline(self, name: str = "baseline") -> Scenario:
        """Create a baseline (no intervention) scenario."""
        scenario = Scenario(
            name=name,
            description="Baseline scenario with no interventions"
        )
        self.add_scenario(scenario)
        self.set_baseline(name)
        return scenario
    
    def create_intervention_scenario(
        self,
        name: str,
        intervention_type: str,
        magnitude: float,
        start_time: Union[datetime, float],
        end_time: Optional[Union[datetime, float]] = None,
        description: str = ""
    ) -> Scenario:
        """
        Create a scenario with a single intervention.
        
        Args:
            name: Scenario name
            intervention_type: Type of intervention
            magnitude: Intervention strength
            start_time: When intervention begins
            end_time: When intervention ends (None for indefinite)
            description: Scenario description
            
        Returns:
            Created scenario
        """
        intervention = Intervention(
            intervention_type=intervention_type,
            magnitude=magnitude,
            start_time=start_time,
            end_time=end_time
        )
        
        scenario = Scenario(
            name=name,
            description=description or f"{intervention_type} at {magnitude*100:.0f}% starting at {start_time}",
            interventions=[intervention]
        )
        
        self.add_scenario(scenario)
        return scenario
    
    def create_vaccination_scenario(
        self,
        name: str,
        daily_rate: float,
        start_time: Union[datetime, float],
        coverage_target: float = 0.8,
        ramp_up_days: int = 14
    ) -> Scenario:
        """
        Create a vaccination scenario.
        
        Args:
            name: Scenario name
            daily_rate: Daily vaccination rate (fraction of population)
            start_time: When vaccination begins
            coverage_target: Target coverage level
            ramp_up_days: Days to reach full vaccination rate
            
        Returns:
            Created scenario
        """
        intervention = Intervention(
            intervention_type="vaccination",
            magnitude=daily_rate,
            start_time=start_time,
            parameters={
                "coverage_target": coverage_target,
                "ramp_up_days": ramp_up_days,
                "vaccine_efficacy": 0.9,  # Default
            }
        )
        
        scenario = Scenario(
            name=name,
            description=f"Vaccination at {daily_rate*100:.1f}%/day, target {coverage_target*100:.0f}%",
            interventions=[intervention]
        )
        
        self.add_scenario(scenario)
        return scenario
    
    def create_lockdown_scenario(
        self,
        name: str,
        reduction: float,
        start_time: Union[datetime, float],
        duration_days: float,
        description: str = ""
    ) -> Scenario:
        """
        Create a lockdown/social distancing scenario.
        
        Args:
            name: Scenario name
            reduction: Contact reduction fraction (0.5 = 50% reduction)
            start_time: Lockdown start
            duration_days: Duration in days
            description: Optional description
            
        Returns:
            Created scenario
        """
        if isinstance(start_time, datetime):
            from datetime import timedelta
            end_time = start_time + timedelta(days=duration_days)
        else:
            end_time = start_time + duration_days
        
        intervention = Intervention(
            intervention_type="social_distancing",
            magnitude=reduction,
            start_time=start_time,
            end_time=end_time,
            parameters={"duration_days": duration_days}
        )
        
        scenario = Scenario(
            name=name,
            description=description or f"{reduction*100:.0f}% contact reduction for {duration_days} days",
            interventions=[intervention]
        )
        
        self.add_scenario(scenario)
        return scenario
    
    def get_all_interventions(self) -> Dict[str, List[Intervention]]:
        """Get interventions for all scenarios."""
        return {
            name: scenario.interventions
            for name, scenario in self.scenarios.items()
        }
    
    def list_scenarios(self) -> List[str]:
        """List all scenario names."""
        return list(self.scenarios.keys())
    
    def summarize(self) -> str:
        """Generate a summary of all scenarios."""
        lines = ["Scenario Summary", "=" * 40]
        
        for name, scenario in self.scenarios.items():
            is_baseline = " (BASELINE)" if name == self._baseline_name else ""
            lines.append(f"\n{name}{is_baseline}")
            lines.append("-" * len(name))
            lines.append(scenario.description or "No description")
            lines.append(f"Interventions: {len(scenario.interventions)}")
            
            for i, intervention in enumerate(scenario.interventions, 1):
                lines.append(
                    f"  {i}. {intervention.intervention_type}: "
                    f"{intervention.magnitude*100:.1f}% from {intervention.start_time}"
                )
        
        return "\n".join(lines)


@dataclass
class ScenarioComparison:
    """
    Result of comparing two scenarios.
    
    Attributes:
        scenario_name: Name of the compared scenario
        baseline_name: Name of the baseline scenario
        metrics: Computed comparison metrics
    """
    
    scenario_name: str
    baseline_name: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def get_reduction(self, metric: str) -> float:
        """Get reduction for a metric (positive = scenario is better)."""
        return self.metrics.get(f"{metric}_reduction_mean", 0.0)
    
    def get_reduction_ci(self, metric: str, level: float = 0.95) -> tuple:
        """Get confidence interval for reduction."""
        ci_key = f"{metric}_reduction_ci_{int(level*100)}"
        return self.metrics.get(ci_key, (0.0, 0.0))
    
    def probability_better(self, metric: str) -> float:
        """Probability that scenario is better than baseline for metric."""
        return self.metrics.get(f"prob_better_{metric}", 0.5)
    
    def summarize(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Comparison: {self.scenario_name} vs {self.baseline_name}",
            "=" * 50
        ]
        
        for key, value in self.metrics.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.3f}")
            elif isinstance(value, tuple) and len(value) == 2:
                lines.append(f"  {key}: ({value[0]:.3f}, {value[1]:.3f})")
        
        return "\n".join(lines)
