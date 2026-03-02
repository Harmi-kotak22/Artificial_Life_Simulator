"""
Disease outbreak forecasting module.

Pathogen-agnostic epidemiological engine for infectious disease modeling.
"""

from updff.hazards.disease.engine import DiseaseModule, create_disease_module
from updff.hazards.disease.traits import (
    PathogenTraits,
    create_covid_like_traits,
    create_influenza_traits,
    create_measles_like_traits,
    create_cholera_like_traits
)
from updff.hazards.disease.compartments import CompartmentalModel, CompartmentDefinition
from updff.hazards.disease.transmission import ContactNetwork, TransmissionModel

__all__ = [
    "DiseaseModule",
    "create_disease_module",
    "PathogenTraits",
    "create_covid_like_traits",
    "create_influenza_traits", 
    "create_measles_like_traits",
    "create_cholera_like_traits",
    "CompartmentalModel",
    "CompartmentDefinition",
    "ContactNetwork",
    "TransmissionModel",
]
