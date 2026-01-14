"""
LEACH (Low-Energy Adaptive Clustering Hierarchy) Algorithm

Reference: Heinzelman et al., "Energy-Efficient Communication Protocol
for Wireless Microsensor Networks," HICSS 2000.

Key features:
- Probabilistic CH election based on threshold function T(n)
- Rotation mechanism to distribute CH role
- Simple, low overhead
"""

from typing import Optional, Set
import numpy as np
from .base import ClusteringAlgorithm
from ..models.node import Node, NodeRole
from ..models.network import Network
from ..models.cluster import Cluster, form_clusters_from_heads
from ..models.energy import EnergyModel


class LEACHClustering(ClusteringAlgorithm):
    """
    LEACH Clustering Algorithm.

    Protocol:
    1. Each node decides to become CH based on threshold T(n)
    2. CHs broadcast advertisement
    3. Non-CH nodes join nearest CH
    4. CH rotation every round
    """

    def __init__(
        self,
        network: Network,
        energy_model: EnergyModel,
        config: dict
    ):
        super().__init__(network, energy_model, config)

        # Load LEACH parameters
        leach_cfg = config.get('leach', {})
        self.p = leach_cfg.get('p', 0.05)  # Desired CH percentage

        # Track nodes that have been CH in current cycle
        self.was_ch_this_cycle: Set[int] = set()
        self.rounds_per_cycle = int(1 / self.p)  # 1/p rounds per cycle
        self.current_round_in_cycle = 0

    @property
    def name(self) -> str:
        return "LEACH"

    def setup(self):
        """Initialize algorithm state."""
        self.was_ch_this_cycle = set()
        self.current_round_in_cycle = 0

    def _calculate_threshold(self, node: Node) -> float:
        """
        Calculate threshold T(n) for CH election.

        T(n) = p / (1 - p * (r mod 1/p))  if n in G (eligible)
        T(n) = 0                           otherwise

        where:
        - p = desired percentage of CHs
        - r = current round number
        - G = set of nodes that haven't been CH in current cycle
        """
        # Check if node was CH in current cycle
        if node.id in self.was_ch_this_cycle:
            return 0.0

        # Calculate threshold
        r_mod = self.current_round_in_cycle
        denominator = 1 - self.p * r_mod

        if denominator <= 0:
            return 1.0  # Guarantee selection

        return self.p / denominator

    def elect_cluster_heads(self) -> list[Node]:
        """
        LEACH probabilistic CH election.
        Each eligible node becomes CH if random < T(n).
        """
        # Update cycle tracking
        self.current_round_in_cycle += 1
        if self.current_round_in_cycle >= self.rounds_per_cycle:
            # New cycle
            self.current_round_in_cycle = 0
            self.was_ch_this_cycle.clear()

        alive_nodes = self.network.get_alive_nodes()
        heads = []

        for node in alive_nodes:
            threshold = self._calculate_threshold(node)

            if np.random.random() < threshold:
                heads.append(node)
                self.was_ch_this_cycle.add(node.id)
                # Control message: CH advertisement
                self.control_messages += 1

        # If no CH elected (rare), force selection
        if not heads and alive_nodes:
            # Select node with highest energy
            best = max(alive_nodes, key=lambda n: n.current_energy)
            heads.append(best)
            self.was_ch_this_cycle.add(best.id)
            self.control_messages += 1

        return heads

    def form_clusters(self, heads: list[Node]) -> list[Cluster]:
        """
        Form clusters by assigning each non-CH node to nearest CH.
        """
        if not heads:
            return []

        clusters = form_clusters_from_heads(heads, self.network.nodes, self.network)

        # Control messages: join requests from members
        for cluster in clusters:
            self.control_messages += cluster.member_count

        return clusters


class LEACHCentralized(ClusteringAlgorithm):
    """
    LEACH-C (Centralized) Clustering Algorithm.

    BS collects all node info and optimally selects CHs.
    This serves as an upper bound on LEACH performance.
    """

    def __init__(
        self,
        network: Network,
        energy_model: EnergyModel,
        config: dict
    ):
        super().__init__(network, energy_model, config)

        leach_cfg = config.get('leach', {})
        self.p = leach_cfg.get('p', 0.05)

    @property
    def name(self) -> str:
        return "LEACH-C"

    def setup(self):
        """Initialize algorithm state."""
        pass

    def elect_cluster_heads(self) -> list[Node]:
        """
        Centralized CH selection based on energy and position.
        BS selects nodes with above-average energy as CH candidates.
        """
        alive_nodes = self.network.get_alive_nodes()
        if not alive_nodes:
            return []

        # Calculate average energy
        avg_energy = np.mean([n.current_energy for n in alive_nodes])

        # Filter nodes with above-average energy
        candidates = [n for n in alive_nodes if n.current_energy >= avg_energy]

        if not candidates:
            candidates = alive_nodes

        # Select k CHs (k = p * N)
        k = max(1, int(self.p * len(alive_nodes)))

        # Score based on energy and centrality (distance to centroid)
        centroid_x = np.mean([n.x for n in alive_nodes])
        centroid_y = np.mean([n.y for n in alive_nodes])

        def score(node):
            energy_score = node.current_energy / node.initial_energy
            dist_to_centroid = np.sqrt((node.x - centroid_x)**2 + (node.y - centroid_y)**2)
            # Higher energy and more central = higher score
            return energy_score / (dist_to_centroid + 1)

        candidates.sort(key=score, reverse=True)

        # Select top k with spacing
        heads = []
        min_spacing = 20.0  # Minimum distance between CHs

        for node in candidates:
            if len(heads) >= k:
                break

            # Check spacing
            too_close = False
            for head in heads:
                if self.network.get_distance(node, head) < min_spacing:
                    too_close = True
                    break

            if not too_close:
                heads.append(node)

        # If not enough heads, add more without spacing constraint
        for node in candidates:
            if len(heads) >= k:
                break
            if node not in heads:
                heads.append(node)

        # Control messages: BS broadcasts cluster assignments
        self.control_messages += len(alive_nodes)

        return heads

    def form_clusters(self, heads: list[Node]) -> list[Cluster]:
        """
        BS assigns nodes to CHs optimally (nearest CH).
        """
        return form_clusters_from_heads(heads, self.network.nodes, self.network)


if __name__ == "__main__":
    from ..models.node import create_heterogeneous_nodes
    from ..models.network import Network
    from ..models.energy import EnergyModel
    import yaml

    # Load config
    with open("config/default.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Create network
    nodes = create_heterogeneous_nodes(
        n=50,
        width=100,
        height=100,
        energy_mean=2.0,
        energy_std=0.2,
        seed=42
    )
    network = Network(nodes, bs_x=50, bs_y=100, comm_range=30)
    energy_model = EnergyModel()

    # Test LEACH
    print("=== LEACH Test ===\n")
    leach = LEACHClustering(network, energy_model, config)
    leach.setup()

    for _ in range(3):
        stats = leach.run_epoch()
        print(f"Epoch {stats['epoch']}: {stats['alive_nodes']} alive, "
              f"{stats['num_clusters']} clusters, "
              f"{stats['control_messages']} ctrl msgs, "
              f"E={stats['total_energy']:.3f}J")

    # Test LEACH-C
    print("\n=== LEACH-C Test ===\n")
    nodes2 = create_heterogeneous_nodes(n=50, width=100, height=100, seed=42)
    network2 = Network(nodes2, bs_x=50, bs_y=100, comm_range=30)

    leach_c = LEACHCentralized(network2, energy_model, config)
    leach_c.setup()

    for _ in range(3):
        stats = leach_c.run_epoch()
        print(f"Epoch {stats['epoch']}: {stats['alive_nodes']} alive, "
              f"{stats['num_clusters']} clusters, "
              f"{stats['control_messages']} ctrl msgs, "
              f"E={stats['total_energy']:.3f}J")
