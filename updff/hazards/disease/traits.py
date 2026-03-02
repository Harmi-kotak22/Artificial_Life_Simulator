"""
Pathogen trait descriptors for disease modeling.

Defines abstract characteristics of pathogens without hardcoding
any specific disease, enabling transferability across known and
hypothetical pathogens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from updff.core.distribution import Distribution, LogNormal, Gamma, Beta, Normal


class TransmissionModality(Enum):
    """Modes of pathogen transmission."""
    RESPIRATORY_DROPLET = "respiratory_droplet"
    RESPIRATORY_AEROSOL = "respiratory_aerosol"
    DIRECT_CONTACT = "direct_contact"
    INDIRECT_CONTACT = "indirect_contact"  # Fomites
    FECAL_ORAL = "fecal_oral"
    VECTOR_BORNE = "vector_borne"
    VERTICAL = "vertical"  # Parent to offspring
    SEXUAL = "sexual"
    BLOOD_BORNE = "blood_borne"


class ClinicalSeverity(Enum):
    """Severity classification."""
    ASYMPTOMATIC = "asymptomatic"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


@dataclass
class PathogenTraits:
    """
    Abstract characterization of pathogen behavior.
    
    All characteristics are specified as probability distributions
    to capture uncertainty and enable Bayesian inference.
    
    This abstraction allows the same epidemiological engine to model
    any infectious disease by simply changing the trait descriptors.
    
    Attributes:
        name: Pathogen/disease name (for labeling)
        transmission_modalities: How the pathogen spreads
        base_reproduction_number: R0 - average secondary infections
        generation_time: Time between successive infections
        serial_interval: Time between symptom onsets
        overdispersion: Heterogeneity in transmission (k parameter)
        incubation_period: Time from infection to symptom onset
        latent_period: Time from infection to infectiousness
        infectious_period: Duration of infectiousness
        presymptomatic_infectious_period: Infectious before symptoms
        asymptomatic_fraction: Proportion never developing symptoms
        relative_infectiousness_asymptomatic: Compared to symptomatic
        hospitalization_rate: P(hospitalization | symptomatic)
        icu_rate: P(ICU | hospitalized)
        fatality_rate: Infection fatality ratio (IFR)
        immunity_duration: Duration of post-infection immunity
        immunity_waning_rate: Rate of immunity loss
        vaccine_efficacy_infection: Vaccine efficacy against infection
        vaccine_efficacy_severe: Vaccine efficacy against severe disease
        environmental_survival: Survival time on surfaces
        temperature_sensitivity: Effect of temperature on transmission
        humidity_sensitivity: Effect of humidity on transmission
        seasonality_amplitude: Amplitude of seasonal forcing
        seasonality_phase: Phase of seasonal peak (day of year)
    """
    
    # Identification
    name: str = "Generic Pathogen"
    
    # Transmission characteristics
    transmission_modalities: List[TransmissionModality] = field(
        default_factory=lambda: [TransmissionModality.RESPIRATORY_DROPLET]
    )
    
    # Core epidemiological parameters
    base_reproduction_number: Distribution = field(
        default_factory=lambda: LogNormal.from_mean_std(2.5, 0.5)
    )
    generation_time: Distribution = field(
        default_factory=lambda: Gamma.from_mean_std(5.0, 1.5)
    )
    serial_interval: Distribution = field(
        default_factory=lambda: Gamma.from_mean_std(5.5, 2.0)
    )
    overdispersion: Distribution = field(
        default_factory=lambda: LogNormal.from_mean_std(0.5, 0.3)
    )  # k parameter; smaller = more overdispersed
    
    # Temporal progression
    incubation_period: Distribution = field(
        default_factory=lambda: LogNormal.from_mean_std(5.2, 1.5)
    )
    latent_period: Distribution = field(
        default_factory=lambda: Gamma.from_mean_std(3.0, 1.0)
    )
    infectious_period: Distribution = field(
        default_factory=lambda: Gamma.from_mean_std(7.0, 2.0)
    )
    presymptomatic_infectious_period: Distribution = field(
        default_factory=lambda: Gamma.from_mean_std(2.0, 0.5)
    )
    
    # Clinical presentation
    asymptomatic_fraction: Distribution = field(
        default_factory=lambda: Beta.from_mean_std(0.3, 0.1)
    )
    relative_infectiousness_asymptomatic: Distribution = field(
        default_factory=lambda: Beta.from_mean_std(0.5, 0.15)
    )
    
    # Clinical outcomes
    hospitalization_rate: Distribution = field(
        default_factory=lambda: Beta.from_mean_std(0.05, 0.02)
    )
    icu_rate: Distribution = field(
        default_factory=lambda: Beta.from_mean_std(0.25, 0.1)
    )  # Given hospitalization
    fatality_rate: Distribution = field(
        default_factory=lambda: Beta.from_mean_std(0.01, 0.005)
    )  # IFR
    
    # Immunity
    immunity_duration: Distribution = field(
        default_factory=lambda: Gamma.from_mean_std(365, 90)
    )  # Days
    immunity_waning_rate: Distribution = field(
        default_factory=lambda: LogNormal.from_mean_std(0.003, 0.001)
    )  # Per day
    
    # Vaccine
    vaccine_efficacy_infection: Distribution = field(
        default_factory=lambda: Beta.from_mean_std(0.7, 0.1)
    )
    vaccine_efficacy_severe: Distribution = field(
        default_factory=lambda: Beta.from_mean_std(0.9, 0.05)
    )
    vaccine_efficacy_waning_rate: Distribution = field(
        default_factory=lambda: LogNormal.from_mean_std(0.002, 0.001)
    )
    
    # Environmental factors
    environmental_survival: Distribution = field(
        default_factory=lambda: Gamma.from_mean_std(24, 12)
    )  # Hours on surfaces
    
    temperature_sensitivity: Optional[Callable[[float], float]] = None
    humidity_sensitivity: Optional[Callable[[float], float]] = None
    
    seasonality_amplitude: Distribution = field(
        default_factory=lambda: Beta.from_mean_std(0.2, 0.1)
    )
    seasonality_phase: Distribution = field(
        default_factory=lambda: Normal(loc=0, scale=30)
    )  # Days from Jan 1
    
    # Age-specific modifiers (optional)
    age_susceptibility: Optional[Dict[str, float]] = None
    age_severity: Optional[Dict[str, float]] = None
    
    def sample_parameters(
        self,
        n_samples: int = 1,
        rng: Optional[np.random.Generator] = None
    ) -> Dict[str, np.ndarray]:
        """
        Sample a complete set of parameters from trait distributions.
        
        Args:
            n_samples: Number of parameter sets to sample
            rng: Random number generator
            
        Returns:
            Dictionary mapping parameter names to sampled values
        """
        params = {
            "R0": self.base_reproduction_number.sample(n_samples, rng),
            "generation_time": self.generation_time.sample(n_samples, rng),
            "serial_interval": self.serial_interval.sample(n_samples, rng),
            "overdispersion_k": self.overdispersion.sample(n_samples, rng),
            "incubation_period": self.incubation_period.sample(n_samples, rng),
            "latent_period": self.latent_period.sample(n_samples, rng),
            "infectious_period": self.infectious_period.sample(n_samples, rng),
            "presymptomatic_period": self.presymptomatic_infectious_period.sample(n_samples, rng),
            "asymptomatic_fraction": self.asymptomatic_fraction.sample(n_samples, rng),
            "relative_infectiousness_asymptomatic": self.relative_infectiousness_asymptomatic.sample(n_samples, rng),
            "hospitalization_rate": self.hospitalization_rate.sample(n_samples, rng),
            "icu_rate": self.icu_rate.sample(n_samples, rng),
            "ifr": self.fatality_rate.sample(n_samples, rng),
            "immunity_duration": self.immunity_duration.sample(n_samples, rng),
            "immunity_waning_rate": self.immunity_waning_rate.sample(n_samples, rng),
            "vaccine_efficacy_infection": self.vaccine_efficacy_infection.sample(n_samples, rng),
            "vaccine_efficacy_severe": self.vaccine_efficacy_severe.sample(n_samples, rng),
            "seasonality_amplitude": self.seasonality_amplitude.sample(n_samples, rng),
            "seasonality_phase": self.seasonality_phase.sample(n_samples, rng),
        }
        return params
    
    def compute_transmission_rate(
        self,
        r0: float,
        infectious_period: float
    ) -> float:
        """
        Compute transmission rate (beta) from R0 and infectious period.
        
        β = R0 / D where D is the infectious period.
        """
        return r0 / infectious_period
    
    def compute_basic_reproduction_number(
        self,
        beta: float,
        infectious_period: float,
        susceptible_fraction: float = 1.0
    ) -> float:
        """
        Compute R0 from transmission rate.
        
        R0 = β × D × S
        """
        return beta * infectious_period * susceptible_fraction
    
    def compute_effective_reproduction_number(
        self,
        r0: float,
        susceptible_fraction: float,
        intervention_effect: float = 0.0
    ) -> float:
        """
        Compute effective reproduction number Rt.
        
        Rt = R0 × S × (1 - intervention_effect)
        """
        return r0 * susceptible_fraction * (1 - intervention_effect)
    
    def get_seasonality_modifier(
        self,
        day_of_year: int,
        amplitude: float,
        phase: float
    ) -> float:
        """
        Compute seasonal forcing modifier.
        
        Uses sinusoidal model: 1 + A × cos(2π(t - φ)/365)
        """
        return 1.0 + amplitude * np.cos(2 * np.pi * (day_of_year - phase) / 365)
    
    def summarize(self) -> str:
        """Generate human-readable summary of pathogen traits."""
        lines = [
            f"Pathogen: {self.name}",
            "=" * 50,
            f"Transmission: {', '.join(m.value for m in self.transmission_modalities)}",
            f"R0: {float(self.base_reproduction_number.mean()):.2f} "
            f"(std: {float(self.base_reproduction_number.std()):.2f})",
            f"Generation time: {float(self.generation_time.mean()):.1f} days",
            f"Incubation period: {float(self.incubation_period.mean()):.1f} days",
            f"Infectious period: {float(self.infectious_period.mean()):.1f} days",
            f"Asymptomatic fraction: {float(self.asymptomatic_fraction.mean())*100:.1f}%",
            f"Hospitalization rate: {float(self.hospitalization_rate.mean())*100:.2f}%",
            f"IFR: {float(self.fatality_rate.mean())*100:.3f}%",
            f"Immunity duration: {float(self.immunity_duration.mean()):.0f} days",
        ]
        return "\n".join(lines)


# Pre-defined trait profiles for common pathogens
# These serve as templates and can be customized

def create_influenza_traits() -> PathogenTraits:
    """Create trait descriptor for influenza-like illness."""
    return PathogenTraits(
        name="Influenza-like",
        transmission_modalities=[
            TransmissionModality.RESPIRATORY_DROPLET,
            TransmissionModality.RESPIRATORY_AEROSOL,
        ],
        base_reproduction_number=LogNormal.from_mean_std(1.3, 0.2),
        generation_time=Gamma.from_mean_std(3.0, 0.8),
        serial_interval=Gamma.from_mean_std(3.0, 1.0),
        incubation_period=LogNormal.from_mean_std(2.0, 0.5),
        latent_period=Gamma.from_mean_std(1.5, 0.5),
        infectious_period=Gamma.from_mean_std(5.0, 1.0),
        asymptomatic_fraction=Beta.from_mean_std(0.25, 0.1),
        hospitalization_rate=Beta.from_mean_std(0.01, 0.005),
        fatality_rate=Beta.from_mean_std(0.001, 0.0005),
        seasonality_amplitude=Beta.from_mean_std(0.3, 0.1),
        seasonality_phase=Normal(loc=15, scale=10),  # Mid-January peak
    )


def create_covid_like_traits() -> PathogenTraits:
    """Create trait descriptor for COVID-19-like illness."""
    return PathogenTraits(
        name="COVID-like",
        transmission_modalities=[
            TransmissionModality.RESPIRATORY_DROPLET,
            TransmissionModality.RESPIRATORY_AEROSOL,
            TransmissionModality.INDIRECT_CONTACT,
        ],
        base_reproduction_number=LogNormal.from_mean_std(2.5, 0.5),
        generation_time=Gamma.from_mean_std(5.0, 1.5),
        serial_interval=Gamma.from_mean_std(5.5, 2.0),
        overdispersion=LogNormal.from_mean_std(0.3, 0.15),
        incubation_period=LogNormal.from_mean_std(5.2, 1.5),
        latent_period=Gamma.from_mean_std(3.0, 1.0),
        infectious_period=Gamma.from_mean_std(8.0, 2.0),
        presymptomatic_infectious_period=Gamma.from_mean_std(2.5, 0.8),
        asymptomatic_fraction=Beta.from_mean_std(0.35, 0.1),
        hospitalization_rate=Beta.from_mean_std(0.05, 0.02),
        icu_rate=Beta.from_mean_std(0.25, 0.1),
        fatality_rate=Beta.from_mean_std(0.01, 0.005),
        immunity_duration=Gamma.from_mean_std(180, 60),
    )


def create_measles_like_traits() -> PathogenTraits:
    """Create trait descriptor for measles-like highly transmissible illness."""
    return PathogenTraits(
        name="Measles-like",
        transmission_modalities=[
            TransmissionModality.RESPIRATORY_AEROSOL,
        ],
        base_reproduction_number=LogNormal.from_mean_std(15.0, 3.0),
        generation_time=Gamma.from_mean_std(12.0, 2.0),
        serial_interval=Gamma.from_mean_std(12.0, 3.0),
        incubation_period=LogNormal.from_mean_std(10.0, 2.0),
        latent_period=Gamma.from_mean_std(8.0, 1.5),
        infectious_period=Gamma.from_mean_std(8.0, 1.0),
        asymptomatic_fraction=Beta.from_mean_std(0.05, 0.02),
        hospitalization_rate=Beta.from_mean_std(0.2, 0.05),
        fatality_rate=Beta.from_mean_std(0.001, 0.0005),
        immunity_duration=Gamma.from_mean_std(36500, 3650),  # ~100 years (lifelong)
    )


def create_cholera_like_traits() -> PathogenTraits:
    """Create trait descriptor for waterborne illness like cholera."""
    return PathogenTraits(
        name="Cholera-like",
        transmission_modalities=[
            TransmissionModality.FECAL_ORAL,
            TransmissionModality.DIRECT_CONTACT,
        ],
        base_reproduction_number=LogNormal.from_mean_std(2.0, 0.5),
        generation_time=Gamma.from_mean_std(5.0, 2.0),
        serial_interval=Gamma.from_mean_std(5.0, 2.0),
        incubation_period=LogNormal.from_mean_std(2.0, 1.0),
        latent_period=Gamma.from_mean_std(1.5, 0.5),
        infectious_period=Gamma.from_mean_std(7.0, 2.0),
        asymptomatic_fraction=Beta.from_mean_std(0.75, 0.1),  # Most infections asymptomatic
        hospitalization_rate=Beta.from_mean_std(0.1, 0.03),
        fatality_rate=Beta.from_mean_std(0.01, 0.005),  # With treatment
        immunity_duration=Gamma.from_mean_std(1095, 365),  # ~3 years
    )
