"""
Main disease forecasting module.

Pathogen-agnostic epidemiological engine implementing the HazardModule
interface for integration with the core forecasting framework.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats

from updff.core.state import State, StateSpaceSpec, Parameters, Observation, Intervention
from updff.core.distribution import Distribution, Normal, LogNormal, Gamma, Beta, NegativeBinomial
from updff.hazards.interface import HazardModule
from updff.hazards.disease.traits import PathogenTraits, TransmissionModality
from updff.hazards.disease.compartments import (
    CompartmentalModel, 
    CompartmentDefinition,
    CompartmentType,
    TransitionDefinition
)
from updff.hazards.disease.transmission import ContactNetwork, TransmissionModel

logger = logging.getLogger(__name__)


class DiseaseModule(HazardModule):
    """
    Pathogen-agnostic epidemiological module.
    
    Implements disease dynamics for any infectious disease by combining:
    - Abstract pathogen traits (no hardcoded disease specifics)
    - Flexible compartmental structure
    - Network-based transmission
    - Intervention modeling
    
    This module does NOT hardcode any specific pathogen. Disease behavior
    is entirely defined through the PathogenTraits descriptor.
    """
    
    def __init__(
        self,
        pathogen_traits: PathogenTraits,
        population_size: int,
        model_type: str = "seir",
        contact_network: Optional[ContactNetwork] = None,
        stochastic: bool = True,
        observation_model: str = "negative_binomial",
        reporting_rate: float = 0.5,
        reporting_delay: int = 0
    ):
        """
        Initialize disease module.
        
        Args:
            pathogen_traits: Abstract pathogen characteristics
            population_size: Total population
            model_type: Compartmental model type ("sir", "seir", "seirs", etc.)
            contact_network: Optional network structure for heterogeneous mixing
            stochastic: Use stochastic dynamics
            observation_model: Observation noise model ("poisson", "negative_binomial")
            reporting_rate: Fraction of cases reported
            reporting_delay: Days between infection and reporting
        """
        self.traits = pathogen_traits
        self.population = population_size
        self.model_type = model_type
        self.stochastic = stochastic
        self.observation_model = observation_model
        self.reporting_rate = reporting_rate
        self.reporting_delay = reporting_delay
        
        # Create compartmental model
        self.compartmental = self._create_compartmental_model(model_type)
        
        # Set up contact network
        if contact_network is not None:
            self.network = contact_network
        else:
            # Single well-mixed population
            from updff.hazards.disease.transmission import PopulationNode
            nodes = [PopulationNode(id=0, name="population", population=population_size)]
            self.network = ContactNetwork(nodes)
        
        # Transmission model
        self.transmission = TransmissionModel(
            self.network,
            base_transmission_rate=0.5  # Will be updated from parameters
        )
        
        # Cache state spec
        self._state_spec = self._build_state_spec()
        
        logger.info(
            f"Initialized DiseaseModule for {pathogen_traits.name} "
            f"with {model_type.upper()} model, N={population_size}"
        )
    
    def _create_compartmental_model(self, model_type: str) -> CompartmentalModel:
        """Create compartmental model based on type."""
        model_factories = {
            "sir": CompartmentalModel.create_sir,
            "seir": CompartmentalModel.create_seir,
            "seirs": CompartmentalModel.create_seirs,
            "seir_hosp": CompartmentalModel.create_seir_with_hospitalization,
            "seir_vax": CompartmentalModel.create_seir_with_vaccination,
        }
        
        if model_type.lower() not in model_factories:
            logger.warning(f"Unknown model type {model_type}, defaulting to SEIR")
            model_type = "seir"
        
        return model_factories[model_type.lower()](self.population)
    
    def _build_state_spec(self) -> StateSpaceSpec:
        """Build state space specification."""
        state_names = self.compartmental.state_names
        state_dim = len(state_names)
        
        # Parameters derived from pathogen traits
        param_names = [
            "beta",           # Transmission rate
            "sigma",          # 1 / latent period
            "gamma",          # 1 / infectious period
            "R0",             # Basic reproduction number
            "k",              # Overdispersion
            "reporting_rate", # Observation probability
        ]
        
        if self.model_type in ["seirs"]:
            param_names.append("omega")  # Waning immunity rate
        
        if self.model_type in ["seir_vax"]:
            param_names.extend(["vaccination_rate", "vaccine_efficacy"])
        
        return StateSpaceSpec(
            state_dim=state_dim,
            state_names=state_names,
            state_bounds=[(0, self.population)] * state_dim,
            param_dim=len(param_names),
            param_names=param_names,
            param_bounds=[
                (0.0, 10.0),     # beta
                (0.01, 1.0),     # sigma
                (0.01, 1.0),     # gamma
                (0.1, 30.0),     # R0
                (0.01, 10.0),    # k
                (0.0, 1.0),      # reporting_rate
            ] + [(0.0, 1.0)] * (len(param_names) - 6),
            observation_dim=1,
            observation_names=["reported_cases"],
            intervention_dim=4,
            intervention_names=["vaccination", "social_distancing", "testing_isolation", "treatment"]
        )
    
    def get_state_spec(self) -> StateSpaceSpec:
        """Return state space specification."""
        return self._state_spec
    
    def initialize_state(
        self,
        initial_conditions: Dict[str, Any],
        uncertainty: Dict[str, float]
    ) -> State:
        """
        Initialize disease state.
        
        Args:
            initial_conditions: Dict with keys like:
                - "prevalence": initial infection prevalence (fraction)
                - "infected": initial number infected
                - "recovered_fraction": initial immunity
            uncertainty: Uncertainty (std) for initial values
            
        Returns:
            Initialized State
        """
        state = np.zeros(self.compartmental.n_compartments)
        
        # Get initial infected
        if "infected" in initial_conditions:
            initial_infected = initial_conditions["infected"]
        elif "prevalence" in initial_conditions:
            initial_infected = int(initial_conditions["prevalence"] * self.population)
        else:
            initial_infected = 10  # Default seed
        
        # Get initial recovered
        if "recovered_fraction" in initial_conditions:
            initial_recovered = int(initial_conditions["recovered_fraction"] * self.population)
        elif "recovered" in initial_conditions:
            initial_recovered = initial_conditions["recovered"]
        else:
            initial_recovered = 0
        
        # Assign to compartments based on model type
        compartment_names = self.compartmental.state_names
        
        # Susceptible
        s_idx = compartment_names.index("S") if "S" in compartment_names else 0
        
        # Infectious (find first infectious compartment)
        i_idx = None
        for name in ["I", "I_mild", "I_a"]:
            if name in compartment_names:
                i_idx = compartment_names.index(name)
                break
        if i_idx is None:
            i_idx = 1  # Default
        
        # Recovered
        r_idx = compartment_names.index("R") if "R" in compartment_names else -1
        
        # Exposed (if SEIR)
        e_idx = compartment_names.index("E") if "E" in compartment_names else None
        
        # Distribute initial infected
        if e_idx is not None:
            # Split between E and I
            state[e_idx] = initial_infected // 2
            state[i_idx] = initial_infected - state[e_idx]
        else:
            state[i_idx] = initial_infected
        
        # Set recovered
        if r_idx >= 0:
            state[r_idx] = initial_recovered
        
        # Susceptible = remainder
        state[s_idx] = self.population - np.sum(state)
        
        # Build covariance from uncertainty
        covariance = None
        if uncertainty:
            var = np.zeros(len(state))
            for name, std in uncertainty.items():
                if name in compartment_names:
                    idx = compartment_names.index(name)
                    var[idx] = (std * self.population) ** 2
                elif name == "prevalence":
                    if i_idx is not None:
                        var[i_idx] = (std * self.population) ** 2
            covariance = np.diag(var)
        
        return State(
            values=state,
            timestamp=initial_conditions.get("timestamp", 0.0),
            covariance=covariance
        )
    
    def transition(
        self,
        state: State,
        params: Parameters,
        interventions: List[Intervention],
        dt: float,
        n_samples: int = 1
    ) -> List[State]:
        """
        Stochastic state transition.
        
        Implements P(X_{t+dt} | X_t, Θ, U).
        """
        rng = np.random.default_rng()
        
        # Convert params to dict
        param_dict = self._params_to_dict(params)
        
        # Apply interventions
        param_dict = self._apply_intervention_effects(param_dict, interventions, state)
        
        next_states = []
        for _ in range(n_samples):
            # Step the compartmental model
            next_values = self.compartmental.step(
                state.values,
                param_dict,
                dt,
                stochastic=self.stochastic,
                rng=rng
            )
            
            # Ensure non-negative and sum to population
            next_values = np.maximum(next_values, 0)
            
            # Update timestamp
            if isinstance(state.timestamp, datetime):
                from datetime import timedelta
                next_time = state.timestamp + timedelta(days=dt)
            else:
                next_time = state.timestamp + dt
            
            next_states.append(State(
                values=next_values,
                timestamp=next_time,
                covariance=None  # Will be computed from ensemble
            ))
        
        return next_states
    
    def _params_to_dict(self, params: Parameters) -> Dict[str, float]:
        """Convert Parameters object to dictionary."""
        param_dict = {}
        
        if params.names:
            for i, name in enumerate(params.names):
                param_dict[name] = float(params.values[i] if params.values.ndim == 1 else params.values[0, i])
        else:
            # Default mapping
            spec = self.get_state_spec()
            for i, name in enumerate(spec.param_names):
                if i < len(params.values):
                    param_dict[name] = float(params.values[i] if params.values.ndim == 1 else params.values[0, i])
        
        return param_dict
    
    def _apply_intervention_effects(
        self,
        params: Dict[str, float],
        interventions: List[Intervention],
        state: State
    ) -> Dict[str, float]:
        """Apply intervention effects to parameters."""
        modified = params.copy()
        
        for intervention in interventions:
            if not intervention.is_active(state.timestamp):
                continue
            
            magnitude = intervention.get_magnitude_at(state.timestamp)
            int_type = intervention.intervention_type
            
            if int_type == "social_distancing":
                # Reduce transmission rate
                modified["beta"] = modified.get("beta", 0.5) * (1 - magnitude)
            
            elif int_type == "vaccination":
                # Handled in compartmental model transitions
                modified["vaccination_rate"] = magnitude
            
            elif int_type == "testing_isolation":
                # Effectively reduces infectious period
                modified["gamma"] = modified.get("gamma", 0.1) * (1 + magnitude)
            
            elif int_type == "treatment":
                # Reduces mortality and speeds recovery
                # This would need model-specific handling
                pass
            
            elif int_type == "lockdown":
                # Strong transmission reduction
                modified["beta"] = modified.get("beta", 0.5) * (1 - 0.8 * magnitude)
            
            elif int_type == "mask_mandate":
                # Moderate transmission reduction
                modified["beta"] = modified.get("beta", 0.5) * (1 - 0.3 * magnitude)
        
        return modified
    
    def observe(
        self,
        state: State,
        observation_noise: np.ndarray
    ) -> Observation:
        """
        Generate noisy observation from state.
        
        Models under-reporting and observation noise.
        """
        # Get infectious compartment(s)
        compartment_names = self.compartmental.state_names
        true_cases = 0
        
        for i, name in enumerate(compartment_names):
            comp = self.compartmental.compartments.get(name)
            if comp and comp.is_infectious:
                true_cases += state.values[i]
        
        # Apply reporting rate
        expected_reported = true_cases * self.reporting_rate
        
        # Add observation noise
        if self.observation_model == "negative_binomial" and expected_reported > 0:
            # Overdispersed count data
            k = 10.0  # Dispersion parameter
            p = k / (k + expected_reported)
            observed = np.random.negative_binomial(k, p)
        elif expected_reported > 0:
            # Poisson
            observed = np.random.poisson(expected_reported)
        else:
            observed = 0
        
        # Observation covariance (for filtering)
        # Var(Y) ≈ expected_reported / reporting_rate^2 for Poisson
        obs_var = max(expected_reported / self.reporting_rate, 1.0)
        
        return Observation(
            values=np.array([float(observed)]),
            timestamp=state.timestamp,
            noise_covariance=np.array([[obs_var]]),
            observation_type="reported_cases"
        )
    
    def log_likelihood(
        self,
        observation: Observation,
        state: State,
        params: Parameters
    ) -> float:
        """
        Compute log P(Y | X, Θ).
        
        Likelihood of observing Y given latent state X.
        """
        # Get expected observed cases
        param_dict = self._params_to_dict(params)
        reporting = param_dict.get("reporting_rate", self.reporting_rate)
        
        # Sum infectious compartments
        compartment_names = self.compartmental.state_names
        true_cases = 0
        for i, name in enumerate(compartment_names):
            comp = self.compartmental.compartments.get(name)
            if comp and comp.is_infectious:
                true_cases += state.values[i]
        
        expected = true_cases * reporting
        observed = observation.values[0]
        
        if expected <= 0:
            expected = 0.1  # Avoid log(0)
        
        if self.observation_model == "negative_binomial":
            k = param_dict.get("k", 10.0)
            # NegBinom log-likelihood
            dist = NegativeBinomial(mu=expected, k=k)
            return float(dist.log_prob(np.array([observed])))
        else:
            # Poisson
            return float(stats.poisson.logpmf(int(observed), expected))
    
    def get_prior(self) -> Dict[str, Distribution]:
        """
        Return prior distributions derived from pathogen traits.
        """
        # Sample mean values from traits for prior construction
        r0_mean = float(self.traits.base_reproduction_number.mean())
        gen_time_mean = float(self.traits.generation_time.mean())
        latent_mean = float(self.traits.latent_period.mean())
        infectious_mean = float(self.traits.infectious_period.mean())
        k_mean = float(self.traits.overdispersion.mean())
        
        # Derive beta from R0 and infectious period
        # R0 = beta * D, so beta = R0 / D
        beta_mean = r0_mean / infectious_mean
        
        priors = {
            "beta": LogNormal.from_mean_std(beta_mean, beta_mean * 0.3),
            "sigma": Gamma.from_mean_std(1.0 / latent_mean, 0.5 / latent_mean),  # 1/latent
            "gamma": Gamma.from_mean_std(1.0 / infectious_mean, 0.5 / infectious_mean),  # 1/infectious
            "R0": self.traits.base_reproduction_number,
            "k": self.traits.overdispersion,
            "reporting_rate": Beta.from_mean_std(self.reporting_rate, 0.1),
        }
        
        if self.model_type == "seirs":
            immunity_days = float(self.traits.immunity_duration.mean())
            priors["omega"] = Gamma.from_mean_std(1.0 / immunity_days, 0.5 / immunity_days)
        
        return priors
    
    def apply_intervention(
        self,
        intervention: Intervention,
        params: Parameters,
        state: State
    ) -> Parameters:
        """Apply intervention effect to parameters."""
        param_dict = self._params_to_dict(params)
        modified_dict = self._apply_intervention_effects(param_dict, [intervention], state)
        
        # Convert back to Parameters
        new_values = np.array([modified_dict.get(n, 0.0) for n in params.names or self._state_spec.param_names])
        
        return Parameters(
            values=new_values,
            names=params.names,
            covariance=params.covariance,
            bounds=params.bounds
        )
    
    def compute_reproduction_number(
        self,
        state: State,
        params: Parameters
    ) -> float:
        """Compute effective reproduction number Rt."""
        param_dict = self._params_to_dict(params)
        return self.compartmental.compute_reproduction_number(state.values, param_dict)
    
    def compute_risk_score(
        self,
        state: State,
        params: Parameters
    ) -> float:
        """
        Compute risk score in [0, 1].
        
        Based on Rt, case growth, and healthcare capacity.
        """
        rt = self.compute_reproduction_number(state, params)
        
        # Get current prevalence
        compartment_names = self.compartmental.state_names
        total_infectious = 0
        for i, name in enumerate(compartment_names):
            comp = self.compartmental.compartments.get(name)
            if comp and comp.is_infectious:
                total_infectious += state.values[i]
        
        prevalence = total_infectious / self.population
        
        # Risk score components
        rt_score = min(1.0, max(0.0, (rt - 0.5) / 2.0))  # 0 at Rt=0.5, 1 at Rt=2.5
        prevalence_score = min(1.0, prevalence * 100)  # 0 at 0%, 1 at 1%
        
        # Combined score
        risk = 0.6 * rt_score + 0.4 * prevalence_score
        
        return float(np.clip(risk, 0.0, 1.0))
    
    def summarize_state(self, state: State) -> str:
        """Generate human-readable state summary."""
        compartment_names = self.compartmental.state_names
        lines = ["Disease State Summary", "=" * 40]
        
        for i, name in enumerate(compartment_names):
            count = int(state.values[i])
            pct = 100 * count / self.population
            lines.append(f"  {name}: {count:,} ({pct:.2f}%)")
        
        return "\n".join(lines)


def create_disease_module(
    pathogen: str = "covid",
    population: int = 1_000_000,
    model_type: str = "seir"
) -> DiseaseModule:
    """
    Factory function to create disease modules with pre-configured traits.
    
    Args:
        pathogen: Pathogen type ("covid", "influenza", "measles", "cholera", "custom")
        population: Population size
        model_type: Compartmental model type
        
    Returns:
        Configured DiseaseModule
    """
    from updff.hazards.disease.traits import (
        create_covid_like_traits,
        create_influenza_traits,
        create_measles_like_traits,
        create_cholera_like_traits,
        PathogenTraits
    )
    
    trait_factories = {
        "covid": create_covid_like_traits,
        "influenza": create_influenza_traits,
        "measles": create_measles_like_traits,
        "cholera": create_cholera_like_traits,
        "custom": PathogenTraits,
    }
    
    if pathogen.lower() not in trait_factories:
        logger.warning(f"Unknown pathogen {pathogen}, using generic traits")
        traits = PathogenTraits()
    else:
        traits = trait_factories[pathogen.lower()]()
    
    return DiseaseModule(
        pathogen_traits=traits,
        population_size=population,
        model_type=model_type
    )
