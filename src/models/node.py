"""
Node class representing a sensor node as an autonomous agent.

Each node has:
- Private state: energy level, position, fairness debt
- Observable state: alive/dead status
- Actions: bid, join cluster, transmit, receive

Updated for ABC (Auction-Based Clustering) protocol.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


class NodeRole(Enum):
    """Possible roles for a sensor node."""
    UNDECIDED = 0
    CLUSTER_HEAD = 1
    CLUSTER_MEMBER = 2
    SLEEPING = 3
    BACKUP = 4  # Backup CH


@dataclass
class Node:
    """
    Represents a wireless sensor node as an agent in the MAS.

    Attributes:
        id: Unique identifier
        x, y: Position coordinates
        initial_energy: Starting energy (J)
        current_energy: Remaining energy (J)
        role: Current role in the network
        cluster_head_id: ID of the CH this node belongs to (if member)
        debt: Fairness debt (positive = served too much, negative = credit)
        aggressiveness: Bidding aggressiveness multiplier m_i(t)
        rounds_as_ch: Total rounds served as cluster head
        bid: Current bid value (calculated each epoch)

        # Bandit learning state for aggressiveness
        bandit_q_values: Q-value estimates for each action (m value)
        bandit_action_counts: Number of times each action was selected
        bandit_current_action: Index of currently selected action
        bandit_was_ch: Whether node was CH this round (for reward)
    """
    id: int
    x: float
    y: float
    initial_energy: float = 2.0  # Default from spec: N(mu=2.0, sigma=0.2)
    current_energy: float = field(init=False)
    role: NodeRole = NodeRole.UNDECIDED
    cluster_head_id: Optional[int] = None
    debt: float = 0.0  # Fairness debt D_i(t), replaces fairness_credit
    aggressiveness: float = 1.0  # m_i(t), bandit-tuned
    rounds_as_ch: int = 0
    bid: float = 0.0  # Current bid value

    # Bandit learning state
    bandit_q_values: dict = field(default_factory=dict)  # action_idx -> Q-value
    bandit_action_counts: dict = field(default_factory=dict)  # action_idx -> count
    bandit_current_action: int = 2  # Default action index (m=1.0 in default action set)
    bandit_was_ch: bool = False  # Track if was CH this round

    def __post_init__(self):
        self.current_energy = self.initial_energy
        # Initialize bandit state if not already done
        if not self.bandit_q_values:
            self.bandit_q_values = {}
        if not self.bandit_action_counts:
            self.bandit_action_counts = {}
    
    @property
    def is_alive(self) -> bool:
        """Node is alive if it has remaining energy."""
        return self.current_energy > 0
    
    @property
    def residual_energy_ratio(self) -> float:
        """Fraction of initial energy remaining."""
        return self.current_energy / self.initial_energy if self.initial_energy > 0 else 0
    
    def distance_to(self, other: "Node") -> float:
        """Euclidean distance to another node."""
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
    
    def distance_to_point(self, x: float, y: float) -> float:
        """Euclidean distance to a coordinate."""
        return np.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)
    
    def consume_energy(self, amount: float) -> bool:
        """
        Consume energy. Returns True if successful, False if node dies.
        """
        if amount > self.current_energy:
            self.current_energy = 0
            return False
        self.current_energy -= amount
        return True
    
    def reset_for_round(self):
        """Reset per-round state (role, cluster assignment, bid, bandit flag)."""
        self.role = NodeRole.UNDECIDED
        self.cluster_head_id = None
        self.bid = 0.0  # Clear stale bid to avoid backup selection pollution
        self.bandit_was_ch = False  # Reset bandit reward flag for this round
    
    def become_cluster_head(self):
        """Transition to cluster head role."""
        self.role = NodeRole.CLUSTER_HEAD
        self.cluster_head_id = self.id
        self.rounds_as_ch += 1
        self.bandit_was_ch = True  # Mark for bandit reward

    def become_backup(self, ch_id: int):
        """Become backup CH for a cluster."""
        self.role = NodeRole.BACKUP
        self.cluster_head_id = ch_id

    def join_cluster(self, ch_id: int):
        """Join a cluster as a member."""
        self.role = NodeRole.CLUSTER_MEMBER
        self.cluster_head_id = ch_id

    def go_to_sleep(self):
        """Enter sleep mode for this round."""
        self.role = NodeRole.SLEEPING


@dataclass 
class NodeStats:
    """Statistics tracked for a node across simulation."""
    total_tx_energy: float = 0.0
    total_rx_energy: float = 0.0
    packets_sent: int = 0
    packets_received: int = 0
    rounds_survived: int = 0


def create_random_nodes(
    n: int,
    width: float,
    height: float,
    initial_energy: float = 2.0,
    seed: Optional[int] = None
) -> list[Node]:
    """
    Create n nodes randomly distributed in a rectangular area.
    All nodes have the same initial energy (homogeneous).

    Args:
        n: Number of nodes
        width: Area width (meters)
        height: Area height (meters)
        initial_energy: Initial energy per node (J)
        seed: Random seed for reproducibility

    Returns:
        List of Node objects
    """
    rng = np.random.default_rng(seed)
    nodes = []
    for i in range(n):
        x = rng.uniform(0, width)
        y = rng.uniform(0, height)
        nodes.append(Node(id=i, x=x, y=y, initial_energy=initial_energy))
    return nodes


def create_heterogeneous_nodes(
    n: int,
    width: float,
    height: float,
    energy_mean: float = 2.0,
    energy_std: float = 0.2,
    energy_min: float = 0.1,
    seed: Optional[int] = None
) -> list[Node]:
    """
    Create n nodes with heterogeneous initial energy (Gaussian distribution).
    Per spec: E_i^(0) ~ N(mu=2.0, sigma=0.2), clipped at > 0.1

    Args:
        n: Number of nodes
        width: Area width (meters)
        height: Area height (meters)
        energy_mean: Mean initial energy (J)
        energy_std: Std dev of initial energy (J)
        energy_min: Minimum energy (clipping threshold)
        seed: Random seed for reproducibility

    Returns:
        List of Node objects with heterogeneous energy
    """
    rng = np.random.default_rng(seed)
    nodes = []
    for i in range(n):
        x = rng.uniform(0, width)
        y = rng.uniform(0, height)
        # Sample energy from Gaussian, clip to minimum
        energy = max(energy_min, rng.normal(energy_mean, energy_std))
        nodes.append(Node(id=i, x=x, y=y, initial_energy=energy))
    return nodes


if __name__ == "__main__":
    # Quick test - homogeneous
    print("=== Homogeneous Nodes ===")
    nodes = create_random_nodes(10, 100, 100, seed=42)
    for node in nodes[:3]:
        print(f"  Node {node.id}: ({node.x:.2f}, {node.y:.2f}), E={node.current_energy:.3f}J")

    # Test heterogeneous
    print("\n=== Heterogeneous Nodes ===")
    nodes_het = create_heterogeneous_nodes(10, 100, 100, seed=42)
    for node in nodes_het[:5]:
        print(f"  Node {node.id}: ({node.x:.2f}, {node.y:.2f}), E={node.current_energy:.3f}J")

    # Test distance
    d = nodes[0].distance_to(nodes[1])
    print(f"\nDistance between node 0 and 1: {d:.2f} m")

    # Test role assignment
    nodes[0].become_cluster_head()
    nodes[1].join_cluster(nodes[0].id)
    nodes[2].become_backup(nodes[0].id)
    print(f"\nNode 0 role: {nodes[0].role.name}")
    print(f"Node 1 role: {nodes[1].role.name}, CH: {nodes[1].cluster_head_id}")
    print(f"Node 2 role: {nodes[2].role.name}, CH: {nodes[2].cluster_head_id}")
