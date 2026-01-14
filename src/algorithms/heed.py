"""
HEED (Hybrid Energy-Efficient Distributed) Clustering Algorithm

Reference: O. Younis and S. Fahmy, "HEED: A hybrid, energy-efficient,
distributed clustering approach for ad hoc sensor networks,"
IEEE Trans. Mobile Comput., vol. 3, no. 4, pp. 366-379, Oct. 2004.

Key features:
- Two-parameter CH selection: residual energy (primary), intra-cluster cost (secondary)
- Iterative multi-round clustering until all nodes are covered
- Tentative and Final CH states
"""

from typing import Optional
import numpy as np
from .base import ClusteringAlgorithm
from ..models.node import Node, NodeRole
from ..models.network import Network
from ..models.cluster import Cluster, form_clusters_from_heads
from ..models.energy import EnergyModel


class HEEDClustering(ClusteringAlgorithm):
    """
    HEED Clustering Algorithm.

    Protocol:
    1. Initialize CH probability based on residual energy
    2. Iterate until all nodes are covered:
       - Nodes probabilistically become tentative/final CHs
       - Double probability each iteration
    3. Uncovered nodes join nearest CH or become CH
    """

    def __init__(
        self,
        network: Network,
        energy_model: EnergyModel,
        config: dict
    ):
        super().__init__(network, energy_model, config)

        # Load HEED parameters
        heed_cfg = config.get('heed', {})
        self.c_prob = heed_cfg.get('c_prob', 0.05)  # Initial CH probability
        self.p_min = heed_cfg.get('p_min', 1e-4)    # Minimum probability threshold

        # Clustering parameters
        cluster_cfg = config.get('clustering', {})
        self.comm_range = config.get('network', {}).get('comm_range', 30.0)

        # Control message sizes (bits) - from config or defaults
        packets_cfg = config.get('packets', {})
        self.ctrl_bits_status = packets_cfg.get('control_size_heed', 800)  # Tentative status
        self.ctrl_bits_final = packets_cfg.get('control_size_heed', 800)  # Final announcement
        self.ctrl_bits_join = packets_cfg.get('control_size_heed', 800)  # Join request

    @property
    def name(self) -> str:
        return "HEED"

    def setup(self):
        """Initialize algorithm state."""
        pass

    def _calculate_amrp(self, node: Node, neighbors: list[Node]) -> float:
        """
        Calculate AMRP (Average Minimum Reachability Power) for a node.
        This is the intra-cluster communication cost metric.

        AMRP = average of min power needed to reach each neighbor
        Approximated as average distance^2 to neighbors
        """
        if not neighbors:
            return float('inf')

        costs = []
        for neighbor in neighbors:
            dist = self.network.get_distance(node, neighbor)
            # Cost proportional to d^2 (free-space model)
            costs.append(dist ** 2)

        return np.mean(costs)

    def elect_cluster_heads(self) -> list[Node]:
        """
        HEED CH election using iterative probabilistic selection.
        """
        alive_nodes = self.network.get_alive_nodes()
        if not alive_nodes:
            return []

        # Find max energy among alive nodes
        e_max = max(n.current_energy for n in alive_nodes)

        # Initialize CH probabilities
        ch_prob = {}
        for node in alive_nodes:
            # CH_prob = C_prob * (E_residual / E_max)
            prob = self.c_prob * (node.current_energy / e_max)
            ch_prob[node.id] = max(prob, self.p_min)

        # Track CH status: None, 'tentative', 'final'
        ch_status = {n.id: None for n in alive_nodes}

        # Track which CH each node would join (for cost calculation)
        my_cluster_head = {n.id: None for n in alive_nodes}

        # Iterative phase
        iteration = 0
        max_iterations = 20  # Safety limit

        while iteration < max_iterations:
            iteration += 1
            all_final = True

            for node in alive_nodes:
                if ch_prob[node.id] >= 1.0:
                    continue  # Already decided

                all_final = False

                # Get neighbors
                neighbors = self.network.get_neighbors(node, self.comm_range)
                alive_neighbors = [n for n in neighbors if n.is_alive]

                # Calculate AMRP for this node
                my_amrp = self._calculate_amrp(node, alive_neighbors)

                # Find least-cost CH among neighbors (tentative or final)
                min_cost = float('inf')
                best_ch = None
                for neighbor in alive_neighbors:
                    if ch_status[neighbor.id] in ('tentative', 'final'):
                        n_amrp = self._calculate_amrp(neighbor,
                            [n for n in self.network.get_neighbors(neighbor, self.comm_range) if n.is_alive])
                        if n_amrp < min_cost:
                            min_cost = n_amrp
                            best_ch = neighbor

                if best_ch is None:
                    # No CH neighbor, consider becoming CH
                    if np.random.random() < ch_prob[node.id]:
                        ch_status[node.id] = 'tentative'
                        my_cluster_head[node.id] = node
                else:
                    # Found a CH neighbor
                    if my_amrp < min_cost:
                        # I'm better, become tentative CH
                        if np.random.random() < ch_prob[node.id]:
                            ch_status[node.id] = 'tentative'
                            my_cluster_head[node.id] = node
                    else:
                        # Join the best CH
                        my_cluster_head[node.id] = best_ch

                # Control message: broadcast CH status (within comm_range)
                if ch_status[node.id] == 'tentative':
                    self.ctrl_broadcast_fixed(node, self.comm_range, self.ctrl_bits_status)

            # Double probabilities
            for node in alive_nodes:
                ch_prob[node.id] = min(1.0, ch_prob[node.id] * 2)

            if all_final:
                break

        # Finalization phase
        final_heads = []
        for node in alive_nodes:
            if ch_status[node.id] == 'tentative':
                # Become final CH
                ch_status[node.id] = 'final'
                final_heads.append(node)
                # Final CH announcement (within comm_range)
                self.ctrl_broadcast_fixed(node, self.comm_range, self.ctrl_bits_final)
            elif my_cluster_head[node.id] is None:
                # Uncovered node becomes CH
                ch_status[node.id] = 'final'
                final_heads.append(node)
                # Announcement for newly-formed CH
                self.ctrl_broadcast_fixed(node, self.comm_range, self.ctrl_bits_final)

        return final_heads

    def form_clusters(self, heads: list[Node]) -> list[Cluster]:
        """
        Form clusters by assigning each non-CH node to the CH with minimum AMRP.
        """
        if not heads:
            return []

        clusters = {h.id: Cluster(h) for h in heads}
        head_ids = {h.id for h in heads}

        # Assign non-CH nodes
        for node in self.network.nodes:
            if not node.is_alive or node.id in head_ids:
                continue

            # Find best CH based on AMRP (approximated by distance)
            min_dist = float('inf')
            best_ch = None

            for head in heads:
                if not head.is_alive:
                    continue
                dist = self.network.get_distance(node, head)
                if dist <= self.comm_range and dist < min_dist:
                    min_dist = dist
                    best_ch = head

            # If no CH in range, find nearest CH
            if best_ch is None:
                for head in heads:
                    if not head.is_alive:
                        continue
                    dist = self.network.get_distance(node, head)
                    if dist < min_dist:
                        min_dist = dist
                        best_ch = head

            if best_ch is not None:
                clusters[best_ch.id].add_member(node)
                # Join request (unicast to chosen CH)
                self.ctrl_unicast(node, best_ch, self.ctrl_bits_join)

        return list(clusters.values())


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

    # Create algorithm
    heed = HEEDClustering(network, energy_model, config)
    heed.setup()

    # Run a few epochs
    print(f"=== {heed.name} Test ===\n")
    for _ in range(3):
        stats = heed.run_epoch()
        print(f"Epoch {stats['epoch']}: {stats['alive_nodes']} alive, "
              f"{stats['num_clusters']} clusters, "
              f"{stats['control_messages']} ctrl msgs, "
              f"E={stats['total_energy']:.3f}J")
