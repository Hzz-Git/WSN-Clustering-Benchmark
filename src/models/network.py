"""
Network class for managing the WSN topology.

Handles:
- Distance matrix computation
- Neighbor discovery
- Base station reference
- Communication range calculations
"""

from typing import Optional
import numpy as np
from .node import Node


class Network:
    """
    Represents the wireless sensor network topology.

    Attributes:
        nodes: List of all nodes in the network
        width: Network area width (meters)
        height: Network area height (meters)
        bs_x: Base station x coordinate
        bs_y: Base station y coordinate
        comm_range: Communication range (meters)
        distance_matrix: Pairwise distances between nodes
        distances_to_bs: Distance from each node to BS
    """

    def __init__(
        self,
        nodes: list[Node],
        width: float = 100.0,
        height: float = 100.0,
        bs_x: float = 50.0,
        bs_y: float = 100.0,
        comm_range: float = 30.0,
    ):
        self.nodes = nodes
        self.width = width
        self.height = height
        self.bs_x = bs_x
        self.bs_y = bs_y
        self.comm_range = comm_range

        # Compute distance matrix once (N x N)
        self._compute_distance_matrix()
        self._compute_distances_to_bs()

    def _compute_distance_matrix(self):
        """Compute pairwise Euclidean distances between all nodes."""
        n = len(self.nodes)
        self.distance_matrix = np.zeros((n, n))

        for i, node_i in enumerate(self.nodes):
            for j, node_j in enumerate(self.nodes):
                if i < j:
                    d = np.sqrt((node_i.x - node_j.x)**2 + (node_i.y - node_j.y)**2)
                    self.distance_matrix[i, j] = d
                    self.distance_matrix[j, i] = d

    def _compute_distances_to_bs(self):
        """Compute distance from each node to the base station."""
        self.distances_to_bs = np.array([
            np.sqrt((node.x - self.bs_x)**2 + (node.y - self.bs_y)**2)
            for node in self.nodes
        ])

    def get_distance(self, node_i: Node, node_j: Node) -> float:
        """Get distance between two nodes."""
        return self.distance_matrix[node_i.id, node_j.id]

    def get_distance_to_bs(self, node: Node) -> float:
        """Get distance from node to base station."""
        return self.distances_to_bs[node.id]

    def get_neighbors(self, node: Node, radius: Optional[float] = None) -> list[Node]:
        """
        Get all neighbors of a node within communication range.

        Args:
            node: The node to find neighbors for
            radius: Custom radius (defaults to comm_range)

        Returns:
            List of neighbor nodes (excluding the node itself)
        """
        r = radius if radius is not None else self.comm_range
        neighbors = []
        for other in self.nodes:
            if other.id != node.id and other.is_alive:
                if self.distance_matrix[node.id, other.id] <= r:
                    neighbors.append(other)
        return neighbors

    def get_alive_nodes(self) -> list[Node]:
        """Get all nodes that are still alive."""
        return [n for n in self.nodes if n.is_alive]

    def count_alive(self) -> int:
        """Count number of alive nodes."""
        return sum(1 for n in self.nodes if n.is_alive)

    def get_hop_count_estimate(self, node: Node) -> float:
        """
        Estimate hop count from node to BS.
        Approximation: h_i = distance_to_bs / comm_range

        Args:
            node: The node to estimate hops for

        Returns:
            Estimated hop count (fractional)
        """
        dist = self.distances_to_bs[node.id]
        return dist / self.comm_range

    def reset_all_nodes(self):
        """Reset all nodes for a new round."""
        for node in self.nodes:
            if node.is_alive:
                node.reset_for_round()

    def get_total_energy(self) -> float:
        """Get total residual energy across all nodes."""
        return sum(n.current_energy for n in self.nodes)

    def get_energy_variance(self) -> float:
        """Get variance of residual energy among alive nodes."""
        alive = self.get_alive_nodes()
        if len(alive) < 2:
            return 0.0
        energies = [n.current_energy for n in alive]
        return np.var(energies)

    def get_node_by_id(self, node_id: int) -> Optional[Node]:
        """Get node by its ID."""
        if 0 <= node_id < len(self.nodes):
            return self.nodes[node_id]
        return None


if __name__ == "__main__":
    from .node import create_heterogeneous_nodes

    # Create test network
    nodes = create_heterogeneous_nodes(20, 100, 100, seed=42)
    network = Network(nodes, bs_x=50, bs_y=100, comm_range=30)

    print(f"Network: {len(nodes)} nodes, {network.count_alive()} alive")
    print(f"Total energy: {network.get_total_energy():.2f} J")
    print(f"Energy variance: {network.get_energy_variance():.4f}")

    # Test neighbor discovery
    test_node = nodes[0]
    neighbors = network.get_neighbors(test_node)
    print(f"\nNode 0 at ({test_node.x:.1f}, {test_node.y:.1f})")
    print(f"  Distance to BS: {network.get_distance_to_bs(test_node):.2f} m")
    print(f"  Hop count estimate: {network.get_hop_count_estimate(test_node):.2f}")
    print(f"  Neighbors within {network.comm_range}m: {len(neighbors)}")
    for n in neighbors[:3]:
        print(f"    Node {n.id}: {network.get_distance(test_node, n):.2f}m away")
