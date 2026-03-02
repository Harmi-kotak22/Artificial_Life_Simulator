"""
Contact network and transmission dynamics for disease modeling.

Implements network-based transmission over heterogeneous populations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import networkx as nx

from updff.core.distribution import Distribution


@dataclass
class PopulationNode:
    """
    A node in the contact network representing a population subgroup.
    
    Could represent:
    - Geographic location (city, district)
    - Age group
    - Risk group
    - Household
    
    Attributes:
        id: Unique identifier
        name: Human-readable name
        population: Total population in this node
        location: Geographic coordinates (optional)
        attributes: Additional node attributes
    """
    
    id: int
    name: str
    population: int
    location: Optional[Tuple[float, float]] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    # State variables (populated during simulation)
    compartment_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class ContactEdge:
    """
    An edge representing contact between population nodes.
    
    Attributes:
        source: Source node ID
        target: Target node ID
        contact_rate: Average contacts per time unit
        contact_type: Type of contact (e.g., "household", "work", "random")
        bidirectional: Whether contact is symmetric
        time_varying: Optional function for time-varying contact
    """
    
    source: int
    target: int
    contact_rate: float
    contact_type: str = "general"
    bidirectional: bool = True
    time_varying: Optional[Callable[[float], float]] = None
    
    def get_rate(self, time: float) -> float:
        """Get contact rate at given time."""
        if self.time_varying is not None:
            return self.contact_rate * self.time_varying(time)
        return self.contact_rate


class ContactNetwork:
    """
    Network structure for disease transmission.
    
    Supports heterogeneous contact patterns including:
    - Spatial metapopulation networks
    - Age-structured contact matrices
    - Multi-layer networks (household, work, community)
    - Time-varying contact patterns
    """
    
    def __init__(
        self,
        nodes: List[PopulationNode],
        edges: Optional[List[ContactEdge]] = None,
        contact_matrix: Optional[np.ndarray] = None
    ):
        """
        Initialize contact network.
        
        Args:
            nodes: List of population nodes
            edges: List of contact edges (for graph-based network)
            contact_matrix: Contact matrix (for matrix-based network)
        """
        self.nodes = {n.id: n for n in nodes}
        self.node_list = nodes
        self.n_nodes = len(nodes)
        
        # Build adjacency structure
        if edges is not None:
            self.edges = edges
            self.adjacency = self._build_adjacency(edges)
        elif contact_matrix is not None:
            self.contact_matrix = contact_matrix
            self.edges = self._matrix_to_edges(contact_matrix)
            self.adjacency = self._build_adjacency(self.edges)
        else:
            self.edges = []
            self.adjacency = {n.id: [] for n in nodes}
            self.contact_matrix = np.eye(self.n_nodes)
    
    def _build_adjacency(self, edges: List[ContactEdge]) -> Dict[int, List[ContactEdge]]:
        """Build adjacency list from edges."""
        adjacency = {n.id: [] for n in self.node_list}
        
        for edge in edges:
            adjacency[edge.source].append(edge)
            if edge.bidirectional:
                # Add reverse edge
                reverse = ContactEdge(
                    source=edge.target,
                    target=edge.source,
                    contact_rate=edge.contact_rate,
                    contact_type=edge.contact_type,
                    bidirectional=False,  # Don't duplicate
                    time_varying=edge.time_varying
                )
                adjacency[edge.target].append(reverse)
        
        return adjacency
    
    def _matrix_to_edges(self, matrix: np.ndarray) -> List[ContactEdge]:
        """Convert contact matrix to edge list."""
        edges = []
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if matrix[i, j] > 0:
                    edges.append(ContactEdge(
                        source=i,
                        target=j,
                        contact_rate=matrix[i, j],
                        bidirectional=False
                    ))
        return edges
    
    def get_contacts(self, node_id: int, time: float = 0.0) -> List[Tuple[int, float]]:
        """
        Get contacts and rates for a node.
        
        Args:
            node_id: Node to query
            time: Current time (for time-varying contacts)
            
        Returns:
            List of (neighbor_id, contact_rate) tuples
        """
        return [
            (edge.target, edge.get_rate(time))
            for edge in self.adjacency[node_id]
        ]
    
    def compute_force_of_infection(
        self,
        node_id: int,
        infectious_by_node: Dict[int, float],
        beta: float,
        time: float = 0.0
    ) -> float:
        """
        Compute force of infection at a node.
        
        λ_i(t) = β × Σ_j C_ij(t) × I_j(t) / N_j
        
        Args:
            node_id: Node to compute FOI for
            infectious_by_node: Number infectious in each node
            beta: Transmission rate
            time: Current time
            
        Returns:
            Force of infection
        """
        foi = 0.0
        
        for neighbor_id, contact_rate in self.get_contacts(node_id, time):
            neighbor_pop = self.nodes[neighbor_id].population
            if neighbor_pop > 0:
                infectious = infectious_by_node.get(neighbor_id, 0)
                foi += contact_rate * infectious / neighbor_pop
        
        return beta * foi
    
    def get_total_population(self) -> int:
        """Total population across all nodes."""
        return sum(n.population for n in self.node_list)
    
    def to_networkx(self) -> nx.Graph:
        """Convert to NetworkX graph for analysis."""
        G = nx.Graph()
        
        for node in self.node_list:
            G.add_node(node.id, **{
                "name": node.name,
                "population": node.population,
                **node.attributes
            })
        
        for edge in self.edges:
            if not G.has_edge(edge.source, edge.target):
                G.add_edge(
                    edge.source, edge.target,
                    weight=edge.contact_rate,
                    contact_type=edge.contact_type
                )
        
        return G
    
    @classmethod
    def create_fully_connected(
        cls,
        populations: List[Tuple[str, int]],
        base_contact_rate: float = 1.0
    ) -> ContactNetwork:
        """Create fully connected network."""
        nodes = [
            PopulationNode(id=i, name=name, population=pop)
            for i, (name, pop) in enumerate(populations)
        ]
        
        edges = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                edges.append(ContactEdge(
                    source=i, target=j,
                    contact_rate=base_contact_rate
                ))
        
        return cls(nodes, edges)
    
    @classmethod
    def create_metapopulation(
        cls,
        populations: List[Tuple[str, int, Tuple[float, float]]],
        distance_decay: float = 0.1,
        within_rate: float = 10.0,
        between_rate: float = 0.1
    ) -> ContactNetwork:
        """
        Create spatial metapopulation network.
        
        Contact rate between locations decays with distance.
        
        Args:
            populations: List of (name, population, (lat, lon))
            distance_decay: Decay rate with distance
            within_rate: Contact rate within location
            between_rate: Base contact rate between locations
        """
        nodes = [
            PopulationNode(id=i, name=name, population=pop, location=loc)
            for i, (name, pop, loc) in enumerate(populations)
        ]
        
        edges = []
        for i, node_i in enumerate(nodes):
            # Within-location contacts
            edges.append(ContactEdge(
                source=i, target=i,
                contact_rate=within_rate,
                contact_type="local"
            ))
            
            # Between-location contacts
            for j, node_j in enumerate(nodes):
                if i >= j:
                    continue
                
                # Compute distance
                lat1, lon1 = node_i.location
                lat2, lon2 = node_j.location
                dist = np.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)
                
                # Distance-decay contact rate
                rate = between_rate * np.exp(-distance_decay * dist)
                
                if rate > 1e-6:  # Only add significant edges
                    edges.append(ContactEdge(
                        source=i, target=j,
                        contact_rate=rate,
                        contact_type="travel"
                    ))
        
        return cls(nodes, edges)
    
    @classmethod
    def create_from_contact_matrix(
        cls,
        group_names: List[str],
        group_populations: List[int],
        contact_matrix: np.ndarray
    ) -> ContactNetwork:
        """
        Create network from contact matrix (e.g., age-structured).
        
        Args:
            group_names: Names of groups (e.g., age groups)
            group_populations: Population in each group
            contact_matrix: Contact rates between groups
        """
        nodes = [
            PopulationNode(id=i, name=name, population=pop)
            for i, (name, pop) in enumerate(zip(group_names, group_populations))
        ]
        
        return cls(nodes, contact_matrix=contact_matrix)


class TransmissionModel:
    """
    Transmission dynamics over a contact network.
    
    Computes transmission rates accounting for:
    - Network structure
    - Heterogeneous susceptibility/infectiousness
    - Environmental factors
    - Interventions
    """
    
    def __init__(
        self,
        network: ContactNetwork,
        base_transmission_rate: float = 0.5,
        overdispersion_k: Optional[float] = None
    ):
        """
        Initialize transmission model.
        
        Args:
            network: Contact network
            base_transmission_rate: Base transmission probability per contact
            overdispersion_k: Dispersion parameter for superspreading (None = Poisson)
        """
        self.network = network
        self.beta = base_transmission_rate
        self.k = overdispersion_k
    
    def compute_new_infections(
        self,
        susceptible_by_node: Dict[int, int],
        infectious_by_node: Dict[int, float],
        time: float = 0.0,
        intervention_effect: float = 0.0,
        rng: Optional[np.random.Generator] = None
    ) -> Dict[int, int]:
        """
        Compute new infections at each node.
        
        Args:
            susceptible_by_node: Susceptible count by node
            infectious_by_node: Infectious count by node (can be fractional for compartmental)
            time: Current time
            intervention_effect: Reduction in transmission (0-1)
            rng: Random generator for stochastic simulation
            
        Returns:
            New infections by node
        """
        rng = rng or np.random.default_rng()
        effective_beta = self.beta * (1 - intervention_effect)
        
        new_infections = {}
        
        for node_id, susceptible in susceptible_by_node.items():
            if susceptible <= 0:
                new_infections[node_id] = 0
                continue
            
            # Compute force of infection
            foi = self.network.compute_force_of_infection(
                node_id, infectious_by_node, effective_beta, time
            )
            
            # Probability of infection per susceptible
            p_infection = 1 - np.exp(-foi)
            
            # Sample new infections
            if self.k is not None:
                # Negative binomial for overdispersion
                # Mean = susceptible * p_infection
                # Var = mean + mean^2/k
                mean = susceptible * p_infection
                if mean > 0:
                    # Gamma-Poisson mixture
                    rate = rng.gamma(self.k, mean / self.k)
                    n_infected = min(rng.poisson(rate), susceptible)
                else:
                    n_infected = 0
            else:
                # Binomial
                n_infected = rng.binomial(susceptible, p_infection)
            
            new_infections[node_id] = n_infected
        
        return new_infections
    
    def compute_reproduction_number(
        self,
        susceptible_fraction_by_node: Dict[int, float],
        infectious_period: float,
        intervention_effect: float = 0.0
    ) -> float:
        """
        Compute effective reproduction number Rt.
        
        Accounts for network structure and heterogeneity.
        
        Args:
            susceptible_fraction_by_node: S/N by node
            infectious_period: Duration of infectiousness
            intervention_effect: Transmission reduction
            
        Returns:
            Effective reproduction number
        """
        effective_beta = self.beta * (1 - intervention_effect)
        
        # Weighted average over network
        total_pop = self.network.get_total_population()
        weighted_rt = 0.0
        
        for node in self.network.node_list:
            node_pop = node.population
            if node_pop == 0:
                continue
            
            # Local R0 contribution
            total_contacts = sum(
                rate for _, rate in self.network.get_contacts(node.id)
            )
            
            susceptible = susceptible_fraction_by_node.get(node.id, 1.0)
            local_rt = effective_beta * total_contacts * infectious_period * susceptible
            
            weighted_rt += local_rt * node_pop / total_pop
        
        return weighted_rt
    
    def apply_intervention(
        self,
        intervention_type: str,
        magnitude: float,
        target_nodes: Optional[List[int]] = None
    ) -> float:
        """
        Compute transmission reduction from intervention.
        
        Args:
            intervention_type: Type of intervention
            magnitude: Intervention strength (0-1)
            target_nodes: Nodes affected (None = all)
            
        Returns:
            Overall transmission reduction factor
        """
        intervention_effects = {
            "social_distancing": magnitude,  # Direct contact reduction
            "mask_mandate": magnitude * 0.5,  # Partial transmission reduction
            "lockdown": magnitude * 0.8,  # Strong contact reduction
            "school_closure": magnitude * 0.3,  # Moderate effect
            "work_from_home": magnitude * 0.2,  # Moderate effect
            "vaccination": 0.0,  # Handled separately in immunity
            "testing_isolation": magnitude * 0.5,  # Reduces infectious period effectively
        }
        
        return intervention_effects.get(intervention_type, magnitude * 0.3)


# Standard contact matrices
def get_polymod_contact_matrix(country: str = "default") -> Tuple[List[str], np.ndarray]:
    """
    Get POLYMOD-style contact matrix.
    
    Returns age groups and contact rates between them.
    """
    age_groups = ["0-4", "5-14", "15-24", "25-44", "45-64", "65+"]
    
    # Simplified symmetric contact matrix (contacts per day)
    # Based on POLYMOD studies
    contact_matrix = np.array([
        [3.0, 1.5, 0.5, 2.0, 1.0, 0.5],  # 0-4
        [1.5, 8.0, 2.0, 1.5, 1.0, 0.5],  # 5-14
        [0.5, 2.0, 6.0, 3.0, 1.5, 0.5],  # 15-24
        [2.0, 1.5, 3.0, 5.0, 2.5, 1.0],  # 25-44
        [1.0, 1.0, 1.5, 2.5, 4.0, 1.5],  # 45-64
        [0.5, 0.5, 0.5, 1.0, 1.5, 2.5],  # 65+
    ])
    
    return age_groups, contact_matrix
