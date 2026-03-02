"""
Compartmental models for disease dynamics.

Provides flexible, configurable compartmental structures that can
represent arbitrary disease progression pathways.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.integrate import odeint, solve_ivp

from updff.core.distribution import Distribution


class CompartmentType(Enum):
    """Types of epidemiological compartments."""
    SUSCEPTIBLE = "S"
    EXPOSED = "E"  # Infected but not yet infectious
    INFECTIOUS = "I"
    RECOVERED = "R"
    DECEASED = "D"
    VACCINATED = "V"
    HOSPITALIZED = "H"
    ICU = "C"
    QUARANTINED = "Q"


@dataclass
class CompartmentDefinition:
    """
    Definition of a single epidemiological compartment.
    
    Attributes:
        name: Compartment identifier
        compartment_type: Type classification
        is_infectious: Whether individuals can transmit
        is_symptomatic: Whether individuals show symptoms
        is_isolated: Whether individuals are isolated
        relative_infectiousness: Compared to reference (usually symptomatic I)
        can_die: Whether mortality can occur from this compartment
        mortality_rate: Rate of mortality (if applicable)
    """
    
    name: str
    compartment_type: CompartmentType
    is_infectious: bool = False
    is_symptomatic: bool = False
    is_isolated: bool = False
    relative_infectiousness: float = 1.0
    can_die: bool = False
    mortality_rate: float = 0.0


@dataclass
class TransitionDefinition:
    """
    Definition of a transition between compartments.
    
    Attributes:
        source: Source compartment name
        target: Target compartment name
        rate_parameter: Name of the rate parameter
        is_infection: Whether this is an infection transition (depends on prevalence)
        branching_probability: For branching transitions, probability of this branch
        condition: Optional condition function for transition
    """
    
    source: str
    target: str
    rate_parameter: str
    is_infection: bool = False
    branching_probability: Optional[float] = None
    condition: Optional[Callable[[Dict[str, float]], bool]] = None


class CompartmentalModel:
    """
    Generalized compartmental model supporting arbitrary structures.
    
    This class allows construction of any compartmental disease model
    by specifying compartments and transitions. Supports:
    - Standard models: SIR, SEIR, SEIRS, SEIS
    - Complex models: SEIR with hospitalization, age structure
    - Custom models: Any directed graph of compartments
    
    The model can operate in:
    - Deterministic mode (ODE integration)
    - Stochastic mode (tau-leaping, Gillespie)
    """
    
    def __init__(
        self,
        compartments: List[CompartmentDefinition],
        transitions: List[TransitionDefinition],
        population_size: int
    ):
        """
        Initialize compartmental model.
        
        Args:
            compartments: List of compartment definitions
            transitions: List of transition definitions
            population_size: Total population
        """
        self.compartments = {c.name: c for c in compartments}
        self.compartment_list = compartments
        self.transitions = transitions
        self.population_size = population_size
        
        # Build index mappings
        self.name_to_idx = {c.name: i for i, c in enumerate(compartments)}
        self.idx_to_name = {i: c.name for i, c in enumerate(compartments)}
        
        # Identify infectious compartments
        self.infectious_compartments = [
            c.name for c in compartments if c.is_infectious
        ]
        
        # Build transition matrix structure
        self._build_transition_structure()
    
    def _build_transition_structure(self):
        """Build internal data structures for transitions."""
        self.outflows: Dict[str, List[TransitionDefinition]] = {
            c.name: [] for c in self.compartment_list
        }
        self.inflows: Dict[str, List[TransitionDefinition]] = {
            c.name: [] for c in self.compartment_list
        }
        
        for trans in self.transitions:
            self.outflows[trans.source].append(trans)
            self.inflows[trans.target].append(trans)
    
    @property
    def n_compartments(self) -> int:
        """Number of compartments."""
        return len(self.compartments)
    
    @property
    def state_names(self) -> List[str]:
        """Names of state variables."""
        return [c.name for c in self.compartment_list]
    
    def compute_force_of_infection(
        self,
        state: np.ndarray,
        params: Dict[str, float],
        contact_rate: float = 1.0
    ) -> float:
        """
        Compute force of infection (λ).
        
        λ = β × Σ (ρ_i × I_i / N)
        
        Where ρ_i is relative infectiousness of compartment i.
        
        Args:
            state: Current state vector
            params: Model parameters (must include 'beta')
            contact_rate: Contact rate modifier
            
        Returns:
            Force of infection
        """
        beta = params.get("beta", params.get("transmission_rate", 0.5))
        
        total_infectious = 0.0
        for comp_name in self.infectious_compartments:
            comp = self.compartments[comp_name]
            idx = self.name_to_idx[comp_name]
            total_infectious += comp.relative_infectiousness * state[idx]
        
        return beta * contact_rate * total_infectious / self.population_size
    
    def derivatives(
        self,
        state: np.ndarray,
        t: float,
        params: Dict[str, float]
    ) -> np.ndarray:
        """
        Compute state derivatives for ODE integration.
        
        Args:
            state: Current state vector
            t: Current time
            params: Model parameters
            
        Returns:
            State derivatives
        """
        dydt = np.zeros_like(state)
        
        # Compute force of infection
        lambda_t = self.compute_force_of_infection(state, params)
        
        for trans in self.transitions:
            src_idx = self.name_to_idx[trans.source]
            tgt_idx = self.name_to_idx[trans.target]
            
            # Get transition rate
            if trans.is_infection:
                # Infection transition depends on force of infection
                rate = lambda_t
            else:
                # Standard transition with parameter
                rate = params.get(trans.rate_parameter, 0.1)
            
            # Apply branching probability if specified
            if trans.branching_probability is not None:
                rate *= trans.branching_probability
            
            # Compute flow
            flow = rate * state[src_idx]
            
            dydt[src_idx] -= flow
            dydt[tgt_idx] += flow
        
        return dydt
    
    def simulate_deterministic(
        self,
        initial_state: np.ndarray,
        params: Dict[str, float],
        t_span: Tuple[float, float],
        dt: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Deterministic simulation using ODE integration.
        
        Args:
            initial_state: Initial compartment populations
            params: Model parameters
            t_span: (start_time, end_time)
            dt: Output time step
            
        Returns:
            (times, states) arrays
        """
        t_eval = np.arange(t_span[0], t_span[1] + dt, dt)
        
        solution = solve_ivp(
            fun=lambda t, y: self.derivatives(y, t, params),
            t_span=t_span,
            y0=initial_state,
            t_eval=t_eval,
            method='RK45'
        )
        
        return solution.t, solution.y.T
    
    def simulate_stochastic(
        self,
        initial_state: np.ndarray,
        params: Dict[str, float],
        t_span: Tuple[float, float],
        dt: float = 1.0,
        rng: Optional[np.random.Generator] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stochastic simulation using tau-leaping.
        
        Args:
            initial_state: Initial compartment populations
            params: Model parameters
            t_span: (start_time, end_time)
            dt: Time step for tau-leaping
            rng: Random number generator
            
        Returns:
            (times, states) arrays
        """
        rng = rng or np.random.default_rng()
        
        times = [t_span[0]]
        states = [initial_state.copy()]
        
        state = initial_state.copy().astype(float)
        t = t_span[0]
        
        while t < t_span[1]:
            # Compute force of infection
            lambda_t = self.compute_force_of_infection(state, params)
            
            # Compute transitions
            for trans in self.transitions:
                src_idx = self.name_to_idx[trans.source]
                tgt_idx = self.name_to_idx[trans.target]
                
                # Get rate
                if trans.is_infection:
                    rate = lambda_t
                else:
                    rate = params.get(trans.rate_parameter, 0.1)
                
                if trans.branching_probability is not None:
                    rate *= trans.branching_probability
                
                # Expected transitions
                expected = rate * state[src_idx] * dt
                
                # Sample from Poisson
                if expected > 0 and state[src_idx] > 0:
                    # Use Poisson for small numbers, Normal approximation for large
                    if expected < 100:
                        n_transitions = rng.poisson(expected)
                    else:
                        n_transitions = max(0, int(rng.normal(expected, np.sqrt(expected))))
                    
                    # Can't transition more than available
                    n_transitions = min(n_transitions, int(state[src_idx]))
                    
                    state[src_idx] -= n_transitions
                    state[tgt_idx] += n_transitions
            
            t += dt
            times.append(t)
            states.append(state.copy())
        
        return np.array(times), np.array(states)
    
    def step(
        self,
        state: np.ndarray,
        params: Dict[str, float],
        dt: float,
        stochastic: bool = True,
        rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """
        Single time step.
        
        Args:
            state: Current state
            params: Parameters
            dt: Time step
            stochastic: Use stochastic dynamics
            rng: Random generator
            
        Returns:
            Next state
        """
        if stochastic:
            _, states = self.simulate_stochastic(
                state, params, (0, dt), dt, rng
            )
            return states[-1]
        else:
            _, states = self.simulate_deterministic(
                state, params, (0, dt), dt
            )
            return states[-1]
    
    def compute_reproduction_number(
        self,
        state: np.ndarray,
        params: Dict[str, float]
    ) -> float:
        """
        Compute effective reproduction number Rt.
        
        Rt = R0 × S/N
        """
        s_idx = self.name_to_idx.get("S", 0)
        susceptible_fraction = state[s_idx] / self.population_size
        
        r0 = params.get("R0", params.get("beta", 0.5) * params.get("infectious_period", 7.0))
        
        return r0 * susceptible_fraction
    
    @classmethod
    def create_sir(cls, population_size: int) -> CompartmentalModel:
        """Create a simple SIR model."""
        compartments = [
            CompartmentDefinition("S", CompartmentType.SUSCEPTIBLE),
            CompartmentDefinition("I", CompartmentType.INFECTIOUS, is_infectious=True, is_symptomatic=True),
            CompartmentDefinition("R", CompartmentType.RECOVERED),
        ]
        
        transitions = [
            TransitionDefinition("S", "I", "beta", is_infection=True),
            TransitionDefinition("I", "R", "gamma"),
        ]
        
        return cls(compartments, transitions, population_size)
    
    @classmethod
    def create_seir(cls, population_size: int) -> CompartmentalModel:
        """Create a SEIR model with exposed compartment."""
        compartments = [
            CompartmentDefinition("S", CompartmentType.SUSCEPTIBLE),
            CompartmentDefinition("E", CompartmentType.EXPOSED),
            CompartmentDefinition("I", CompartmentType.INFECTIOUS, is_infectious=True, is_symptomatic=True),
            CompartmentDefinition("R", CompartmentType.RECOVERED),
        ]
        
        transitions = [
            TransitionDefinition("S", "E", "beta", is_infection=True),
            TransitionDefinition("E", "I", "sigma"),  # 1/latent period
            TransitionDefinition("I", "R", "gamma"),  # 1/infectious period
        ]
        
        return cls(compartments, transitions, population_size)
    
    @classmethod
    def create_seirs(cls, population_size: int) -> CompartmentalModel:
        """Create a SEIRS model with waning immunity."""
        compartments = [
            CompartmentDefinition("S", CompartmentType.SUSCEPTIBLE),
            CompartmentDefinition("E", CompartmentType.EXPOSED),
            CompartmentDefinition("I", CompartmentType.INFECTIOUS, is_infectious=True, is_symptomatic=True),
            CompartmentDefinition("R", CompartmentType.RECOVERED),
        ]
        
        transitions = [
            TransitionDefinition("S", "E", "beta", is_infection=True),
            TransitionDefinition("E", "I", "sigma"),
            TransitionDefinition("I", "R", "gamma"),
            TransitionDefinition("R", "S", "omega"),  # Waning immunity
        ]
        
        return cls(compartments, transitions, population_size)
    
    @classmethod
    def create_seir_with_hospitalization(cls, population_size: int) -> CompartmentalModel:
        """Create SEIR model with hospitalization and death."""
        compartments = [
            CompartmentDefinition("S", CompartmentType.SUSCEPTIBLE),
            CompartmentDefinition("E", CompartmentType.EXPOSED),
            CompartmentDefinition("I_mild", CompartmentType.INFECTIOUS, is_infectious=True, is_symptomatic=True),
            CompartmentDefinition("I_severe", CompartmentType.INFECTIOUS, is_infectious=True, is_symptomatic=True, relative_infectiousness=0.5),
            CompartmentDefinition("H", CompartmentType.HOSPITALIZED, is_infectious=False, is_isolated=True),
            CompartmentDefinition("R", CompartmentType.RECOVERED),
            CompartmentDefinition("D", CompartmentType.DECEASED, can_die=True),
        ]
        
        transitions = [
            TransitionDefinition("S", "E", "beta", is_infection=True),
            TransitionDefinition("E", "I_mild", "sigma", branching_probability=0.8),
            TransitionDefinition("E", "I_severe", "sigma", branching_probability=0.2),
            TransitionDefinition("I_mild", "R", "gamma_mild"),
            TransitionDefinition("I_severe", "H", "hospitalization_rate"),
            TransitionDefinition("H", "R", "recovery_rate_hospital", branching_probability=0.9),
            TransitionDefinition("H", "D", "mortality_rate_hospital", branching_probability=0.1),
        ]
        
        return cls(compartments, transitions, population_size)
    
    @classmethod
    def create_seir_with_vaccination(cls, population_size: int) -> CompartmentalModel:
        """Create SEIR model with vaccination compartment."""
        compartments = [
            CompartmentDefinition("S", CompartmentType.SUSCEPTIBLE),
            CompartmentDefinition("V", CompartmentType.VACCINATED),  # Vaccinated but susceptible (reduced)
            CompartmentDefinition("E", CompartmentType.EXPOSED),
            CompartmentDefinition("I", CompartmentType.INFECTIOUS, is_infectious=True, is_symptomatic=True),
            CompartmentDefinition("R", CompartmentType.RECOVERED),
        ]
        
        transitions = [
            TransitionDefinition("S", "E", "beta", is_infection=True),
            TransitionDefinition("S", "V", "vaccination_rate"),
            TransitionDefinition("V", "E", "beta_v", is_infection=True),  # Reduced transmission for vaccinated
            TransitionDefinition("E", "I", "sigma"),
            TransitionDefinition("I", "R", "gamma"),
        ]
        
        return cls(compartments, transitions, population_size)


class AgeStructuredCompartmentalModel(CompartmentalModel):
    """
    Compartmental model with age structure.
    
    Extends the base model to support age groups with different
    contact patterns and disease progression rates.
    """
    
    def __init__(
        self,
        base_compartments: List[CompartmentDefinition],
        base_transitions: List[TransitionDefinition],
        age_groups: List[str],
        contact_matrix: np.ndarray,
        population_by_age: Dict[str, int]
    ):
        """
        Initialize age-structured model.
        
        Args:
            base_compartments: Compartment definitions (will be replicated per age)
            base_transitions: Transition definitions
            age_groups: List of age group names (e.g., ["0-17", "18-64", "65+"])
            contact_matrix: Age × Age contact rate matrix
            population_by_age: Population in each age group
        """
        self.age_groups = age_groups
        self.n_ages = len(age_groups)
        self.contact_matrix = contact_matrix
        self.population_by_age = population_by_age
        
        # Create expanded compartments and transitions
        compartments, transitions = self._expand_by_age(
            base_compartments, base_transitions
        )
        
        total_population = sum(population_by_age.values())
        super().__init__(compartments, transitions, total_population)
    
    def _expand_by_age(
        self,
        base_compartments: List[CompartmentDefinition],
        base_transitions: List[TransitionDefinition]
    ) -> Tuple[List[CompartmentDefinition], List[TransitionDefinition]]:
        """Expand compartments and transitions by age group."""
        compartments = []
        transitions = []
        
        for age in self.age_groups:
            for comp in base_compartments:
                new_comp = CompartmentDefinition(
                    name=f"{comp.name}_{age}",
                    compartment_type=comp.compartment_type,
                    is_infectious=comp.is_infectious,
                    is_symptomatic=comp.is_symptomatic,
                    is_isolated=comp.is_isolated,
                    relative_infectiousness=comp.relative_infectiousness,
                    can_die=comp.can_die,
                    mortality_rate=comp.mortality_rate
                )
                compartments.append(new_comp)
            
            for trans in base_transitions:
                new_trans = TransitionDefinition(
                    source=f"{trans.source}_{age}",
                    target=f"{trans.target}_{age}",
                    rate_parameter=f"{trans.rate_parameter}_{age}" if trans.is_infection else trans.rate_parameter,
                    is_infection=trans.is_infection,
                    branching_probability=trans.branching_probability,
                    condition=trans.condition
                )
                transitions.append(new_trans)
        
        return compartments, transitions
    
    def compute_force_of_infection(
        self,
        state: np.ndarray,
        params: Dict[str, float],
        contact_rate: float = 1.0
    ) -> np.ndarray:
        """
        Compute age-specific force of infection.
        
        λ_a = β × Σ_b (C_ab × I_b / N_b)
        
        Returns force of infection for each age group.
        """
        beta = params.get("beta", 0.5)
        n_base_compartments = len(self.compartment_list) // self.n_ages
        
        # Compute infectious proportion by age
        infectious_by_age = np.zeros(self.n_ages)
        pop_by_age = np.zeros(self.n_ages)
        
        for a, age in enumerate(self.age_groups):
            pop_by_age[a] = self.population_by_age[age]
            
            for i, comp in enumerate(self.compartment_list):
                if age in comp.name and comp.is_infectious:
                    idx = self.name_to_idx[comp.name]
                    infectious_by_age[a] += comp.relative_infectiousness * state[idx]
        
        # Age-specific force of infection
        lambda_t = np.zeros(self.n_ages)
        for a in range(self.n_ages):
            for b in range(self.n_ages):
                if pop_by_age[b] > 0:
                    lambda_t[a] += (
                        beta * contact_rate *
                        self.contact_matrix[a, b] *
                        infectious_by_age[b] / pop_by_age[b]
                    )
        
        return lambda_t
