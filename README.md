# Universal Probabilistic Disaster Forecasting Framework (UPDFF)

## Project Vision

A modular, disaster-agnostic, probabilistic forecasting framework designed as a **decision-support system** that learns latent dynamics from historical data and produces calibrated uncertainty-quantified forecasts across multiple hazard domains.

**This is NOT a deterministic simulator** — it is a probabilistic forecaster that outputs risk distributions, confidence intervals, and scenario-dependent impact estimates.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Core Abstractions](#core-abstractions)
3. [Mathematical Foundations](#mathematical-foundations)
4. [Development Phases](#development-phases)
5. [MVP Specification](#mvp-specification)
6. [Hazard Module Interface](#hazard-module-interface)
7. [Disease Module Deep Dive](#disease-module-deep-dive)
8. [Inference & Learning](#inference--learning)
9. [Validation & Trustworthiness](#validation--trustworthiness)
10. [Data Requirements](#data-requirements)
11. [Technical Stack](#technical-stack)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DECISION SUPPORT INTERFACE                          │
│    (Risk Maps │ Confidence Intervals │ Scenario Analysis │ Policy Eval)     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CORE FORECASTING ENGINE                           │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────────────────┐  │
│  │  Spatiotemporal │  │   Uncertainty    │  │    Ensemble Execution      │  │
│  │ State Manager   │  │   Propagator     │  │    & Scenario Engine       │  │
│  └─────────────────┘  └──────────────────┘  └────────────────────────────┘  │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────────────────┐  │
│  │   Calibration   │  │    Validation    │  │   Historical Replay        │  │
│  │   Engine        │  │    Framework     │  │   & Backtesting            │  │
│  └─────────────────┘  └──────────────────┘  └────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
┌───────────────────────┐ ┌─────────────────────┐ ┌─────────────────────────┐
│   INFERENCE ENGINE    │ │  DATA ASSIMILATION  │ │  INTERVENTION MODELER   │
│ ├─ Bayesian Inference │ │ ├─ Ensemble Kalman  │ │ ├─ Policy Scenarios     │
│ ├─ MCMC Samplers      │ │ ├─ Particle Filter  │ │ ├─ Control Parameters   │
│ ├─ Variational Inf.   │ │ ├─ State Estimation │ │ ├─ Response Surfaces    │
│ └─ Neural Density Est.│ │ └─ Obs. Operators   │ │ └─ Impact Estimation    │
└───────────────────────┘ └─────────────────────┘ └─────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PLUGGABLE HAZARD MODULES                               │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐   │
│  │   DISEASE    │ │    FLOOD     │ │  EARTHQUAKE  │ │     CYCLONE      │   │
│  │   MODULE     │ │    MODULE    │ │    MODULE    │ │     MODULE       │   │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────────┘   │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                        │
│  │   WILDFIRE   │ │   DROUGHT    │ │   [CUSTOM]   │ ... Extensible         │
│  │   MODULE     │ │    MODULE    │ │    MODULE    │                        │
│  └──────────────┘ └──────────────┘ └──────────────┘                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA LAYER                                        │
│  ┌────────────────────┐  ┌─────────────────────┐  ┌───────────────────────┐ │
│  │ Historical Events  │  │ Environmental Data  │  │ Population/Geographic │ │
│  │ (Calibration)      │  │ (Weather, Climate)  │  │ (Spatial Networks)    │ │
│  └────────────────────┘  └─────────────────────┘  └───────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Abstractions

### 1. State Space Definition

Every hazard operates in a unified state space `S` defined as:

```
S = (X, Θ, U, t)

Where:
  X ∈ ℝⁿ     : Observable state vector (e.g., infected count, water level)
  Θ ∈ ℝᵐ     : Latent parameter vector (e.g., transmission rate, porosity)
  U ∈ ℝᵏ     : Control/intervention vector (e.g., vaccination rate, evacuation)
  t ∈ ℝ⁺     : Time coordinate
```

### 2. Unified Hazard Interface

```python
class HazardModule(Protocol):
    """Abstract interface all hazard modules must implement."""
    
    def define_state_space(self) -> StateSpaceSpec:
        """Define dimensionality and semantics of state variables."""
        ...
    
    def transition_dynamics(
        self, 
        state: State, 
        params: Parameters,
        dt: float,
        noise_samples: np.ndarray
    ) -> Distribution[State]:
        """
        Stochastic state transition: P(X_{t+dt} | X_t, Θ, U)
        Returns a probability distribution over next states.
        """
        ...
    
    def observation_model(
        self, 
        state: State, 
        obs_noise: ObservationNoise
    ) -> Distribution[Observation]:
        """
        Observation likelihood: P(Y_t | X_t)
        Maps latent state to observable quantities.
        """
        ...
    
    def intervention_effect(
        self, 
        intervention: Intervention,
        state: State,
        params: Parameters
    ) -> Parameters:
        """
        Modify parameters based on intervention.
        Returns updated parameter distribution.
        """
        ...
    
    def prior_distribution(self) -> Distribution[Parameters]:
        """
        Prior beliefs over latent parameters.
        Encodes domain knowledge and constraints.
        """
        ...
```

### 3. Probability Distribution Abstraction

```python
class Distribution(Generic[T]):
    """Represents uncertainty over a quantity."""
    
    def sample(self, n: int) -> List[T]:
        """Draw n samples from the distribution."""
        ...
    
    def log_prob(self, value: T) -> float:
        """Compute log probability density/mass."""
        ...
    
    def mean(self) -> T:
        """Expected value."""
        ...
    
    def variance(self) -> T:
        """Variance/covariance."""
        ...
    
    def quantile(self, q: float) -> T:
        """q-th quantile."""
        ...
    
    def credible_interval(self, alpha: float) -> Tuple[T, T]:
        """(1-alpha) credible interval."""
        ...
```

---

## Mathematical Foundations

### State Evolution Model

The system models disaster dynamics as a **Hidden Markov Model (HMM)** with continuous state spaces:

#### Transition Equation (Process Model)
```
X_{t+1} = f(X_t, Θ, U_t) + ε_t
where ε_t ~ N(0, Q_t)
```

#### Observation Equation
```
Y_t = h(X_t) + η_t  
where η_t ~ N(0, R_t)
```

#### Parameter Evolution (for time-varying parameters)
```
Θ_t = g(Θ_{t-1}) + ω_t
where ω_t ~ N(0, W_t)
```

### Uncertainty Propagation

Given initial uncertainty `P(X_0, Θ)`, propagate forward via:

**Chapman-Kolmogorov Equation:**
```
P(X_{t+1}) = ∫ P(X_{t+1} | X_t, Θ) P(X_t, Θ) dX_t dΘ
```

**Implemented via:**
1. **Monte Carlo sampling** for general nonlinear dynamics
2. **Ensemble Kalman Filter (EnKF)** for high-dimensional state spaces
3. **Particle Filters (Sequential Monte Carlo)** for multimodal distributions
4. **Unscented Transform** for Gaussian approximations

### Forecast Output

For any forecast horizon `τ`, the system produces:

```
P(X_{t+τ} | Y_{1:t}, U_{t:t+τ})

Output includes:
- E[X_{t+τ}]                    : Point forecast (mean)
- Var[X_{t+τ}]                  : Uncertainty estimate
- [X_{t+τ}^{α/2}, X_{t+τ}^{1-α/2}]  : Credible interval
- P(X_{t+τ} > threshold)        : Exceedance probability
- Full posterior samples        : For downstream analysis
```

---

## Development Phases

### Phase 0: Project Setup [Week 1]
- Repository structure and build system
- Core dependencies and type definitions
- Logging, configuration, and testing infrastructure

### Phase 1: Core Framework MVP [Weeks 2-4]
**Goal: Minimal working forecasting engine with one hazard module**

- [ ] State space abstraction and management
- [ ] Basic uncertainty propagation (Monte Carlo)
- [ ] Simple ensemble execution
- [ ] Hazard module interface definition
- [ ] Basic disease module (SIR-based, single population)
- [ ] Simple forward forecasting
- [ ] Output: probability distributions and confidence intervals

### Phase 2: Inference Engine [Weeks 5-7]
**Goal: Learn parameters from historical data**

- [ ] Bayesian inference framework
- [ ] MCMC samplers (NUTS, HMC)
- [ ] Ensemble Kalman Filter for data assimilation
- [ ] Parameter posterior estimation
- [ ] Historical replay and calibration
- [ ] Convergence diagnostics

### Phase 3: Advanced Disease Module [Weeks 8-10]
**Goal: Full pathogen-agnostic epidemiological engine**

- [ ] Network-based transmission over heterogeneous populations
- [ ] Abstract pathogen trait descriptors
- [ ] Spatial metapopulation structure
- [ ] Age-structured compartments
- [ ] Intervention modeling (vaccination, NPIs)
- [ ] Hybrid mechanistic + data-driven dynamics

### Phase 4: Additional Hazard Modules [Weeks 11-14]
**Goal: Demonstrate framework generalization**

- [ ] Flood module (hydrological + hydraulic)
- [ ] Earthquake module (seismic hazard)
- [ ] Cyclone module (track + intensity)
- [ ] Wildfire module (spread dynamics)

### Phase 5: Validation & Decision Support [Weeks 15-17]
**Goal: Trustworthy, usable system**

- [ ] Cross-validation framework
- [ ] Probabilistic calibration metrics
- [ ] Falsifiability criteria
- [ ] Risk map generation
- [ ] Scenario comparison interface
- [ ] Policy impact estimation

### Phase 6: Production Hardening [Weeks 18-20]
- [ ] Performance optimization
- [ ] Scalability for large ensembles
- [ ] API design and documentation
- [ ] Deployment infrastructure

---

## MVP Specification

### Scope
**Single hazard domain**: Infectious disease outbreaks
**Single geography**: Configurable region with population data
**Core capabilities**:
1. Load historical outbreak data
2. Infer pathogen parameters from data
3. Generate probabilistic forecasts
4. Output uncertainty-quantified predictions

### MVP Components

```
updff/
├── core/
│   ├── state.py              # State space management
│   ├── distribution.py       # Probability distribution classes
│   ├── uncertainty.py        # Uncertainty propagation
│   ├── ensemble.py           # Ensemble execution engine
│   └── forecast.py           # Forecast generation and output
├── inference/
│   ├── base.py               # Inference interface
│   ├── mcmc.py               # MCMC samplers
│   └── likelihood.py         # Likelihood computation
├── hazards/
│   ├── interface.py          # HazardModule protocol
│   └── disease/
│       ├── engine.py         # Epidemiological engine
│       ├── compartments.py   # Compartmental models
│       ├── traits.py         # Pathogen trait descriptors
│       └── transmission.py   # Transmission dynamics
├── data/
│   ├── loaders.py            # Data loading utilities
│   └── preprocessing.py      # Data cleaning and formatting
├── validation/
│   ├── metrics.py            # Forecast evaluation metrics
│   └── calibration.py        # Probabilistic calibration
└── outputs/
    ├── distributions.py      # Output formatting
    └── visualization.py      # Basic plotting
```

### MVP Data Requirements

1. **Historical case counts** (time series): Daily/weekly confirmed cases
2. **Population data**: Total population, optional age distribution
3. **Time stamps**: Dates for all observations

### MVP Output Format

```python
@dataclass
class ForecastResult:
    """Probabilistic forecast output."""
    
    # Temporal metadata
    forecast_date: datetime
    horizon_days: int
    timestamps: List[datetime]
    
    # Point estimates
    mean_forecast: np.ndarray           # Shape: (horizon,)
    median_forecast: np.ndarray
    
    # Uncertainty quantification
    std_forecast: np.ndarray
    credible_interval_50: Tuple[np.ndarray, np.ndarray]  # 25th, 75th percentile
    credible_interval_90: Tuple[np.ndarray, np.ndarray]  # 5th, 95th percentile
    credible_interval_95: Tuple[np.ndarray, np.ndarray]  # 2.5th, 97.5th percentile
    
    # Full posterior
    ensemble_trajectories: np.ndarray   # Shape: (n_samples, horizon)
    
    # Risk metrics
    peak_timing_distribution: Distribution
    peak_magnitude_distribution: Distribution
    cumulative_distribution: Distribution
    
    # Calibration metadata
    parameter_posteriors: Dict[str, Distribution]
    convergence_diagnostics: Dict[str, float]
```

---

## Hazard Module Interface

### Complete Interface Specification

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, TypeVar
from dataclasses import dataclass
import numpy as np

T = TypeVar('T')

@dataclass
class StateSpaceSpec:
    """Specification of a hazard's state space."""
    state_dim: int
    state_names: List[str]
    state_bounds: List[Tuple[float, float]]
    param_dim: int
    param_names: List[str]
    param_bounds: List[Tuple[float, float]]
    observation_dim: int
    observation_names: List[str]
    intervention_dim: int
    intervention_names: List[str]

@dataclass 
class State:
    """Current state of the system."""
    values: np.ndarray          # State vector
    covariance: np.ndarray      # Uncertainty covariance
    timestamp: float            # Current time
    
@dataclass
class Parameters:
    """Model parameters with uncertainty."""
    values: np.ndarray          # Parameter vector
    covariance: np.ndarray      # Parameter uncertainty

@dataclass
class Observation:
    """Observed data."""
    values: np.ndarray          # Observation vector
    noise_covariance: np.ndarray  # Observation uncertainty
    timestamp: float

@dataclass
class Intervention:
    """Control action / policy intervention."""
    type: str
    magnitude: float
    start_time: float
    duration: float
    spatial_scope: Optional[List[int]] = None


class HazardModule(ABC):
    """Abstract base class for all hazard modules."""
    
    @abstractmethod
    def get_state_spec(self) -> StateSpaceSpec:
        """Return the state space specification."""
        pass
    
    @abstractmethod
    def initialize_state(
        self, 
        initial_conditions: Dict,
        uncertainty: Dict
    ) -> State:
        """Create initial state from conditions."""
        pass
    
    @abstractmethod
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
        
        Returns n_samples of next states drawn from P(X_{t+dt} | X_t, Θ, U).
        This is the core dynamics function.
        """
        pass
    
    @abstractmethod
    def observe(
        self,
        state: State,
        observation_noise: np.ndarray
    ) -> Observation:
        """
        Generate observation from state.
        
        Implements the observation model Y = h(X) + η.
        """
        pass
    
    @abstractmethod
    def log_likelihood(
        self,
        observation: Observation,
        state: State,
        params: Parameters
    ) -> float:
        """
        Compute log P(Y | X, Θ).
        
        Used for inference and model comparison.
        """
        pass
    
    @abstractmethod
    def get_prior(self) -> Dict[str, 'Distribution']:
        """
        Return prior distributions over parameters.
        
        Encodes domain knowledge and physical constraints.
        """
        pass
    
    @abstractmethod
    def apply_intervention(
        self,
        intervention: Intervention,
        params: Parameters
    ) -> Parameters:
        """
        Modify parameters to reflect intervention effect.
        """
        pass
    
    def validate_state(self, state: State) -> bool:
        """Check state satisfies physical constraints."""
        spec = self.get_state_spec()
        for i, (lo, hi) in enumerate(spec.state_bounds):
            if not (lo <= state.values[i] <= hi):
                return False
        return True
```

---

## Disease Module Deep Dive

### Pathogen-Agnostic Design

The disease module does **NOT** hardcode any specific pathogen. Instead, diseases are parameterized via **trait descriptors**:

```python
@dataclass
class PathogenTraits:
    """
    Abstract characterization of pathogen behavior.
    All distributions allow uncertainty quantification.
    """
    
    # Transmission characteristics
    transmission_modality: List[str]  # ["respiratory", "contact", "vector", "fecal-oral"]
    base_reproduction_number: Distribution  # R0 prior
    generation_time: Distribution           # Serial interval distribution
    overdispersion: Distribution            # k parameter for negative binomial
    
    # Clinical progression
    incubation_period: Distribution         # Time from infection to symptoms
    infectious_period: Distribution         # Duration of infectiousness
    latent_period: Distribution             # Time from infection to infectious
    
    # Clinical outcomes
    asymptomatic_fraction: Distribution     # Proportion never symptomatic
    symptomatic_fraction: Distribution      # Proportion with symptoms
    hospitalization_rate: Distribution      # P(hospitalization | symptomatic)
    fatality_rate: Distribution             # Infection fatality ratio
    
    # Immunity
    immunity_duration: Distribution         # Duration of post-infection immunity
    immunity_waning_rate: Distribution      # Rate of immunity loss
    cross_immunity: Dict[str, float]        # Cross-protection with variants
    
    # Environmental sensitivity
    temperature_sensitivity: Optional[Callable]
    humidity_sensitivity: Optional[Callable]
    seasonality_pattern: Optional[Callable]
```

### Generalized Compartmental Structure

Instead of fixed compartments, the model supports **configurable compartmental graphs**:

```python
@dataclass
class CompartmentDefinition:
    """Definition of a single epidemiological compartment."""
    name: str
    is_infectious: bool
    is_symptomatic: bool
    is_isolated: bool
    relative_infectiousness: float  # Relative to reference compartment
    
@dataclass
class TransitionDefinition:
    """Definition of a transition between compartments."""
    source: str
    target: str
    rate_parameter: str       # Name of rate parameter
    rate_distribution: Distribution
    is_stochastic: bool
    depends_on_prevalence: bool  # For force of infection terms

class CompartmentalModel:
    """
    Generalized compartmental model supporting arbitrary structures.
    
    Examples:
    - SIR: S → I → R
    - SEIR: S → E → I → R  
    - SEIRS: S → E → I → R → S
    - SEIAHR: S → E → I_a/I_s → H → R/D
    """
    
    def __init__(
        self,
        compartments: List[CompartmentDefinition],
        transitions: List[TransitionDefinition],
        population_structure: PopulationStructure
    ):
        self.compartments = {c.name: c for c in compartments}
        self.transitions = transitions
        self.population = population_structure
        self._build_transition_matrix()
```

### Network-Based Transmission

Transmission occurs over a **contact network** representing population heterogeneity:

```python
class ContactNetwork:
    """
    Heterogeneous contact structure for transmission.
    
    Supports:
    - Age-structured contact matrices
    - Spatial metapopulation graphs
    - Household/workplace/community layers
    - Time-varying contact patterns
    """
    
    def __init__(
        self,
        nodes: List[PopulationNode],
        edges: List[ContactEdge],
        contact_matrices: Dict[str, np.ndarray]
    ):
        self.nodes = nodes
        self.adjacency = self._build_adjacency(edges)
        self.contact_matrices = contact_matrices
    
    def compute_force_of_infection(
        self,
        state: State,
        params: Parameters,
        time: float
    ) -> np.ndarray:
        """
        Compute λ_i(t) = Σ_j β_{ij}(t) * I_j(t) / N_j
        
        Where β_{ij} encodes:
        - Base transmission rate
        - Contact rate between groups i,j
        - Environmental modifiers
        - Intervention effects
        """
        pass

@dataclass
class PopulationNode:
    """A node in the contact network (e.g., age group, location)."""
    id: int
    population: int
    age_distribution: np.ndarray
    location: Optional[Tuple[float, float]]
    attributes: Dict[str, Any]

@dataclass  
class ContactEdge:
    """Contact intensity between population nodes."""
    source: int
    target: int
    contact_rate: float
    contact_type: str  # "household", "school", "work", "community"
```

### Hybrid Mechanistic + Data-Driven Modeling

The disease module combines:

1. **Mechanistic core**: ODE/stochastic compartmental dynamics based on epidemiological theory
2. **Learned components**: Data-driven time-varying parameters

```python
class HybridEpidemiologicalModel:
    """
    Combines mechanistic dynamics with learned components.
    """
    
    def __init__(
        self,
        mechanistic_model: CompartmentalModel,
        parameter_model: ParameterModel,
        residual_model: Optional[ResidualModel] = None
    ):
        self.mechanistic = mechanistic_model
        self.parameters = parameter_model
        self.residual = residual_model
    
    def forward(
        self,
        state: State,
        time: float,
        dt: float,
        n_samples: int
    ) -> List[State]:
        """
        Forward simulation with uncertainty.
        
        1. Sample parameters from learned posterior
        2. Run mechanistic dynamics
        3. Add residual/correction from data-driven model
        4. Propagate uncertainty
        """
        # Get time-varying parameters from learned model
        param_samples = self.parameters.sample(time, n_samples)
        
        next_states = []
        for params in param_samples:
            # Mechanistic forward step
            mech_state = self.mechanistic.step(state, params, dt)
            
            # Data-driven residual correction (optional)
            if self.residual is not None:
                correction = self.residual.predict(state, time)
                mech_state = self._apply_correction(mech_state, correction)
            
            next_states.append(mech_state)
        
        return next_states


class ParameterModel(ABC):
    """
    Learns time-varying parameters from data.
    
    Implementations:
    - Gaussian Process regression
    - Bayesian neural network
    - Spline-based smooth functions
    - Change-point models
    """
    
    @abstractmethod
    def fit(
        self,
        observations: List[Observation],
        prior: Distribution
    ) -> 'ParameterModel':
        """Fit to historical data."""
        pass
    
    @abstractmethod
    def sample(self, time: float, n_samples: int) -> List[Parameters]:
        """Sample parameters at given time."""
        pass
```

---

## Inference & Learning

### Parameter Inference Strategy

```
┌──────────────────────────────────────────────────────────────────┐
│                    INFERENCE PIPELINE                            │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Historical Data          Prior Knowledge                        │
│       │                         │                                │
│       ▼                         ▼                                │
│  ┌─────────────────────────────────────────────┐                │
│  │         LIKELIHOOD CONSTRUCTION             │                │
│  │  L(Θ) = Π_t P(Y_t | X_t, Θ) P(X_t | X_{t-1}, Θ)             │
│  └─────────────────────────────────────────────┘                │
│                         │                                        │
│                         ▼                                        │
│  ┌─────────────────────────────────────────────┐                │
│  │         POSTERIOR COMPUTATION               │                │
│  │  P(Θ | Y_{1:T}) ∝ L(Θ) × P(Θ)              │                │
│  │                                             │                │
│  │  Methods:                                   │                │
│  │  • MCMC (NUTS/HMC) for full posterior       │                │
│  │  • Variational Inference for scalability    │                │
│  │  • Ensemble Kalman for high dimensions      │                │
│  │  • ABC for intractable likelihoods          │                │
│  └─────────────────────────────────────────────┘                │
│                         │                                        │
│                         ▼                                        │
│  ┌─────────────────────────────────────────────┐                │
│  │         PARAMETER POSTERIORS                │                │
│  │  Returns: Samples from P(Θ | Data)          │                │
│  │  • Point estimates (mean, MAP)              │                │
│  │  • Credible intervals                       │                │
│  │  • Correlation structure                    │                │
│  └─────────────────────────────────────────────┘                │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Implemented Inference Methods

```python
class InferenceEngine(ABC):
    """Base class for parameter inference."""
    
    @abstractmethod
    def fit(
        self,
        model: HazardModule,
        observations: List[Observation],
        prior: Dict[str, Distribution],
        **kwargs
    ) -> InferenceResult:
        """Run inference and return posterior."""
        pass


class MCMCInference(InferenceEngine):
    """
    Markov Chain Monte Carlo inference.
    
    Supports:
    - Metropolis-Hastings
    - Hamiltonian Monte Carlo (HMC)
    - No-U-Turn Sampler (NUTS)
    """
    
    def __init__(
        self,
        sampler: str = "nuts",
        n_chains: int = 4,
        n_warmup: int = 1000,
        n_samples: int = 2000,
        target_accept: float = 0.8
    ):
        self.sampler = sampler
        self.n_chains = n_chains
        self.n_warmup = n_warmup
        self.n_samples = n_samples
        self.target_accept = target_accept


class EnsembleKalmanInference(InferenceEngine):
    """
    Ensemble Kalman Filter/Smoother for state and parameter estimation.
    
    Suitable for:
    - High-dimensional state spaces
    - Online/sequential estimation
    - Approximately Gaussian posteriors
    """
    
    def __init__(
        self,
        n_ensemble: int = 100,
        inflation_factor: float = 1.05,
        localization_radius: Optional[float] = None
    ):
        self.n_ensemble = n_ensemble
        self.inflation = inflation_factor
        self.localization = localization_radius


class ParticleFilterInference(InferenceEngine):
    """
    Sequential Monte Carlo / Particle Filter.
    
    Suitable for:
    - Non-Gaussian posteriors
    - Multimodal distributions
    - Online estimation
    """
    
    def __init__(
        self,
        n_particles: int = 1000,
        resampling_threshold: float = 0.5,
        proposal: str = "bootstrap"
    ):
        self.n_particles = n_particles
        self.threshold = resampling_threshold
        self.proposal = proposal
```

---

## Validation & Trustworthiness

### Probabilistic Forecast Evaluation

```python
class ForecastEvaluator:
    """
    Comprehensive evaluation of probabilistic forecasts.
    """
    
    def evaluate(
        self,
        forecasts: List[ForecastResult],
        observations: List[Observation]
    ) -> EvaluationReport:
        """
        Compute all evaluation metrics.
        """
        return EvaluationReport(
            # Point forecast accuracy
            rmse=self.compute_rmse(forecasts, observations),
            mae=self.compute_mae(forecasts, observations),
            
            # Probabilistic calibration
            pit_histogram=self.compute_pit(forecasts, observations),
            calibration_error=self.compute_calibration_error(forecasts, observations),
            
            # Sharpness (uncertainty appropriate?)
            interval_width=self.compute_interval_widths(forecasts),
            
            # Proper scoring rules
            crps=self.compute_crps(forecasts, observations),
            log_score=self.compute_log_score(forecasts, observations),
            
            # Coverage
            coverage_50=self.compute_coverage(forecasts, observations, 0.50),
            coverage_90=self.compute_coverage(forecasts, observations, 0.90),
            coverage_95=self.compute_coverage(forecasts, observations, 0.95),
        )
    
    def compute_crps(
        self,
        forecasts: List[ForecastResult],
        observations: List[Observation]
    ) -> float:
        """
        Continuous Ranked Probability Score.
        
        CRPS(F, y) = E_F[|X - y|] - 0.5 * E_F[|X - X'|]
        
        Measures both calibration and sharpness.
        """
        pass
    
    def compute_pit(
        self,
        forecasts: List[ForecastResult],
        observations: List[Observation]
    ) -> np.ndarray:
        """
        Probability Integral Transform histogram.
        
        For calibrated forecasts, PIT values should be uniform.
        """
        pass
```

### Falsifiability Criteria

The framework embeds explicit falsifiability checks:

```python
class FalsifiabilityChecker:
    """
    Verify forecasts make falsifiable predictions.
    """
    
    def check_falsifiability(
        self,
        forecast: ForecastResult,
        confidence_level: float = 0.95
    ) -> FalsifiabilityReport:
        """
        A forecast is falsifiable if there exist observable outcomes
        that would be inconsistent with it at the given confidence level.
        """
        
        # Compute prediction intervals
        lower, upper = forecast.credible_interval_95
        
        # Check intervals are finite and non-trivial
        is_bounded = np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))
        is_informative = np.mean(upper - lower) < self.trivial_width_threshold
        
        # Identify falsifying observations
        falsifying_region = self._compute_falsifying_region(forecast, confidence_level)
        
        return FalsifiabilityReport(
            is_falsifiable=is_bounded and is_informative,
            prediction_interval=(lower, upper),
            falsifying_region=falsifying_region,
            interval_width=np.mean(upper - lower)
        )
```

---

## Data Requirements

### Minimal Data (MVP)
| Data Type | Format | Required |
|-----------|--------|----------|
| Case counts | Time series (date, count) | Yes |
| Population | Integer | Yes |
| Dates | ISO 8601 | Yes |

### Standard Data
| Data Type | Format | Required |
|-----------|--------|----------|
| Case counts by region | Time series with location | Recommended |
| Age-stratified cases | Time series with age groups | Recommended |
| Deaths/hospitalizations | Time series | Recommended |
| Testing data | Tests performed, positivity | Recommended |
| Mobility data | Relative change indices | Optional |
| Weather data | Temperature, humidity | Optional |

### Extended Data (Full Capability)
| Data Type | Format | Required |
|-----------|--------|----------|
| Contact matrices | Age × Age matrices | For age-structured models |
| Spatial network | Graph (nodes, edges) | For metapopulation models |
| Genomic surveillance | Variant frequencies | For variant tracking |
| Intervention timelines | Start/end dates, type | For policy analysis |

---

## Technical Stack

### Core Dependencies
```
numpy>=1.24.0          # Array operations
scipy>=1.10.0          # Scientific computing
pandas>=2.0.0          # Data handling
xarray>=2023.1.0       # Multi-dimensional arrays

# Probabilistic programming
pymc>=5.0.0            # Bayesian inference
arviz>=0.15.0          # Inference diagnostics
numpyro>=0.12.0        # JAX-based inference (optional)

# Machine learning
scikit-learn>=1.2.0    # Utilities
torch>=2.0.0           # Neural components (optional)

# Visualization
matplotlib>=3.7.0      # Plotting
plotly>=5.14.0         # Interactive plots

# Utilities  
pydantic>=2.0.0        # Data validation
networkx>=3.0          # Graph structures
numba>=0.57.0          # JIT compilation
```

### Project Structure
```
disaster-forecast/
├── README.md
├── pyproject.toml
├── setup.py
├── requirements.txt
├── .gitignore
│
├── updff/                      # Main package
│   ├── __init__.py
│   ├── core/                   # Core framework
│   │   ├── __init__.py
│   │   ├── state.py            # State space management
│   │   ├── distribution.py     # Probability distributions
│   │   ├── uncertainty.py      # Uncertainty propagation
│   │   ├── ensemble.py         # Ensemble execution
│   │   ├── forecast.py         # Forecast generation
│   │   └── scenario.py         # Scenario management
│   │
│   ├── inference/              # Parameter inference
│   │   ├── __init__.py
│   │   ├── base.py             # Inference interface
│   │   ├── mcmc.py             # MCMC methods
│   │   ├── kalman.py           # Ensemble Kalman
│   │   ├── particle.py         # Particle filters
│   │   └── likelihood.py       # Likelihood computation
│   │
│   ├── hazards/                # Hazard modules
│   │   ├── __init__.py
│   │   ├── interface.py        # HazardModule protocol
│   │   ├── disease/            # Disease module
│   │   │   ├── __init__.py
│   │   │   ├── engine.py       # Main disease engine
│   │   │   ├── compartments.py # Compartmental models
│   │   │   ├── traits.py       # Pathogen traits
│   │   │   ├── network.py      # Contact networks
│   │   │   └── transmission.py # Transmission dynamics
│   │   ├── flood/              # Flood module
│   │   ├── earthquake/         # Earthquake module
│   │   ├── cyclone/            # Cyclone module
│   │   └── wildfire/           # Wildfire module
│   │
│   ├── data/                   # Data handling
│   │   ├── __init__.py
│   │   ├── loaders.py          # Data loaders
│   │   ├── preprocessing.py    # Data preprocessing
│   │   └── schemas.py          # Data schemas
│   │
│   ├── validation/             # Validation framework
│   │   ├── __init__.py
│   │   ├── metrics.py          # Evaluation metrics
│   │   ├── calibration.py      # Calibration assessment
│   │   ├── backtesting.py      # Historical backtesting
│   │   └── falsifiability.py   # Falsifiability checks
│   │
│   └── outputs/                # Output generation
│       ├── __init__.py
│       ├── results.py          # Result classes
│       ├── visualization.py    # Plotting
│       └── export.py           # Export utilities
│
├── tests/                      # Test suite
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
├── examples/                   # Example notebooks/scripts
│   ├── disease_forecast.py
│   └── notebooks/
│
├── data/                       # Sample data
│   └── examples/
│
└── docs/                       # Documentation
    ├── api/
    ├── tutorials/
    └── theory/
```

---

## Usage Example (Target API)

```python
from updff import ForecastingEngine
from updff.hazards.disease import DiseaseModule, PathogenTraits
from updff.inference import MCMCInference
from updff.data import load_outbreak_data

# 1. Define pathogen characteristics (abstract, not hardcoded)
pathogen = PathogenTraits(
    transmission_modality=["respiratory"],
    base_reproduction_number=LogNormal(mean=2.5, std=0.5),
    generation_time=Gamma(mean=5.0, std=1.5),
    incubation_period=LogNormal(mean=5.2, std=0.4),
    infectious_period=Gamma(mean=7.0, std=2.0),
    asymptomatic_fraction=Beta(alpha=2, beta=3),
)

# 2. Create disease module
disease = DiseaseModule(
    pathogen_traits=pathogen,
    population_structure="age_structured",
    spatial_structure="metapopulation"
)

# 3. Load historical data
data = load_outbreak_data("historical_cases.csv")

# 4. Initialize forecasting engine
engine = ForecastingEngine(
    hazard_module=disease,
    inference_method=MCMCInference(n_samples=2000),
    ensemble_size=1000
)

# 5. Calibrate to historical data
engine.calibrate(data, calibration_window=60)  # 60 days

# 6. Generate probabilistic forecast
forecast = engine.forecast(
    horizon_days=28,
    interventions=[
        Intervention(type="vaccination", magnitude=0.02, start_time=7)
    ],
    scenarios=[
        {"name": "baseline", "interventions": []},
        {"name": "with_vaccination", "interventions": [vaccination]},
    ]
)

# 7. Access results
print(f"Mean forecast (day 14): {forecast.mean_forecast[14]:.0f}")
print(f"95% CI: [{forecast.credible_interval_95[0][14]:.0f}, "
      f"{forecast.credible_interval_95[1][14]:.0f}]")
print(f"P(peak > 10000): {forecast.peak_exceedance_prob(10000):.2%}")

# 8. Visualize
forecast.plot_trajectories()
forecast.plot_uncertainty_bands()
forecast.plot_scenario_comparison()
```

---

## Key Design Principles

1. **Probabilistic by default**: Every output includes uncertainty quantification
2. **Modular architecture**: Hazards are pluggable; core engine is hazard-agnostic
3. **Scientific rigor**: Based on established statistical and epidemiological theory
4. **Interpretability**: Mechanistic components remain interpretable
5. **Falsifiability**: Forecasts make testable predictions
6. **Calibration**: Historical data validates, not trains
7. **Robustness**: Designed for distribution shift and incomplete data
8. **Extensibility**: New hazards can be added without core changes

---

## License

MIT License (or appropriate open-source license)

---

## Contributing

See CONTRIBUTING.md for guidelines.

---

## References

- Held, L., & Meyer, S. (2019). Probabilistic forecasting in infectious disease epidemiology
- Reich, N. G., et al. (2019). Accuracy of real-time multi-model ensemble forecasts
- Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation
- Evensen, G. (2009). Data Assimilation: The Ensemble Kalman Filter
