"""
Auction-Based Clustering (ABC) Algorithm

Single-round sealed-bid auction with fairness credit for WSN clustering.
Based on PROPOSED_METHOD_SPEC.md

Key features:
- Bid formula: b_i(t) = m_i(t) * (alpha*E_ratio + beta/(h+eps)) - lambda*D_i(t)
- Fairness debt system with capacity-weighted shares
- Bootstrap phase for first epoch (timer-based seed selection)
- Normal epochs: bid collection and winner selection
"""

from typing import Optional
import numpy as np
from .base import ClusteringAlgorithm
from ..models.node import Node, NodeRole
from ..models.network import Network
from ..models.cluster import Cluster, form_clusters_from_heads
from ..models.energy import EnergyModel


class AuctionClustering(ClusteringAlgorithm):
    """
    Auction-Based Clustering (ABC) with Fairness Credit.

    Protocol phases:
    - Epoch 0 (Bootstrap): Timer-based seed CH selection
    - Epoch >= 1 (Normal): Sealed-bid auction within clusters
    """

    def __init__(
        self,
        network: Network,
        energy_model: EnergyModel,
        config: dict
    ):
        super().__init__(network, energy_model, config)

        # Load auction parameters from config
        auction_cfg = config.get('auction', {})
        self.alpha = auction_cfg.get('alpha', 5.0)
        self.beta = auction_cfg.get('beta', 1.0)
        self.gamma = auction_cfg.get('gamma', 0.2)  # Link quality (unused in simplified)
        self.lambda_ = auction_cfg.get('lambda_', 2.0)
        self.epsilon = auction_cfg.get('epsilon', 0.1)
        self.eta = auction_cfg.get('eta', 0.1)
        self.d_max = auction_cfg.get('d_max', 3.0)
        self.m_default = auction_cfg.get('m_default', 1.0)

        # Bandit learning parameters
        self.m_actions = auction_cfg.get('m_actions', [0.5, 0.8, 1.0, 1.2, 1.5, 2.0])
        self.bandit_epsilon = auction_cfg.get('bandit_epsilon', 0.1)  # Exploration rate
        self.bandit_alpha = auction_cfg.get('bandit_alpha', 0.1)  # Learning rate
        self.use_bandit = auction_cfg.get('use_bandit', True)  # Enable/disable bandit

        # Energy-efficiency based reward (new)
        self.ref_ema = None  # EMA of median cost_per_bit (baseline)
        self.ref_ema_rho = auction_cfg.get('ref_ema_rho', 0.3)  # EMA smoothing factor

        # Clustering parameters
        cluster_cfg = config.get('clustering', {})
        self.join_radius = cluster_cfg.get('join_radius', 30.0)
        self.spacing_radius = cluster_cfg.get('spacing_radius', 30.0)

        # Re-cluster interval
        sim_cfg = config.get('simulation', {})
        self.recluster_interval = sim_cfg.get('recluster_interval', 50)

        # Implicit bidding parameters (only willing nodes bid)
        self.implicit_bidding = auction_cfg.get('implicit_bidding', True)
        self.willingness_threshold = auction_cfg.get('willingness_threshold', 0.3)
        self.min_energy_ratio = auction_cfg.get('min_energy_ratio', 0.2)

        # Control message sizes (bits) - from config or defaults
        packets_cfg = config.get('packets', {})
        self.ctrl_bits_announce = packets_cfg.get('control_size_auction', 960)  # Seed/CH announce
        self.ctrl_bits_bid = packets_cfg.get('control_size_auction', 960)  # Bid message
        self.ctrl_bits_result = packets_cfg.get('control_size_auction', 960)  # Auction result
        self.ctrl_bits_join = packets_cfg.get('control_size_auction', 960)  # Join request

        # Fairness parameters
        fairness_cfg = config.get('fairness', {})
        share_mode = fairness_cfg.get('share_mode', 'alive_only')
        self.share_alive_only = (share_mode == 'alive_only')

    @property
    def name(self) -> str:
        return "ABC-Auction"

    def setup(self):
        """Initialize algorithm state."""
        # Find default action index (m=1.0)
        default_action_idx = 2  # Default to index 2 (m=1.0)
        for i, m in enumerate(self.m_actions):
            if abs(m - self.m_default) < 0.01:
                default_action_idx = i
                break

        # Initialize all nodes with default aggressiveness and bandit state
        for node in self.network.nodes:
            node.aggressiveness = self.m_default
            node.debt = 0.0
            node.bid = 0.0
            node.bandit_was_ch = False

            # Initialize bandit Q-values (optimistic initialization)
            node.bandit_q_values = {i: 0.5 for i in range(len(self.m_actions))}
            node.bandit_action_counts = {i: 0 for i in range(len(self.m_actions))}
            node.bandit_current_action = default_action_idx

    def calculate_bid(self, node: Node) -> float:
        """
        Calculate bid for a node using the exact formula from spec.

        b_i(t) = m_i(t) * (alpha * E_i(t)/E_i^(0) + beta/(h_i(t)+eps)) - lambda * D_i(t)

        Simplified: Link quality term (gamma * Q) is dropped.
        """
        if not node.is_alive:
            return 0.0

        # Energy term: alpha * E_residual / E_initial
        energy_ratio = node.current_energy / node.initial_energy
        energy_term = self.alpha * energy_ratio

        # Closeness term: beta / (hop_count + epsilon)
        # h_i = distance_to_bs / comm_range
        h_i = self.network.get_hop_count_estimate(node)
        closeness_term = self.beta / (h_i + self.epsilon)

        # Aggressiveness multiplier
        m_i = node.aggressiveness

        # Fairness penalty
        fairness_penalty = self.lambda_ * node.debt

        # Final bid
        bid = m_i * (energy_term + closeness_term) - fairness_penalty

        return bid

    def wants_to_be_ch(self, node: Node) -> bool:
        """
        Check if node is willing to be CH (implicit bidding).

        A node only sends a bid if:
        1. Energy ratio > min_energy_ratio (has enough energy)
        2. Debt < threshold (hasn't served too recently)

        This reduces control overhead by filtering non-competitive nodes.
        """
        if not node.is_alive:
            return False

        # Energy filter: skip if too low energy
        energy_ratio = node.current_energy / node.initial_energy
        if energy_ratio < self.min_energy_ratio:
            return False

        # Debt filter: skip if recently served as CH (high debt)
        # Nodes with high debt are penalized anyway, so they won't win
        if node.debt > self.d_max * 0.5:  # debt > 1.5 means recently served
            return False

        # Calculate local willingness score (same as bid but simpler)
        h_i = self.network.get_hop_count_estimate(node)
        local_score = (self.alpha * energy_ratio
                       + self.beta / (h_i + self.epsilon)
                       - self.lambda_ * node.debt)

        return local_score > self.willingness_threshold

    def _bandit_select_action(self, node: Node) -> None:
        """
        Select aggressiveness action using epsilon-greedy strategy.

        Updates node.aggressiveness based on selected action.
        """
        if not self.use_bandit or not node.is_alive:
            return

        import random

        # Epsilon-greedy selection
        if random.random() < self.bandit_epsilon:
            # Explore: random action
            action_idx = random.randint(0, len(self.m_actions) - 1)
        else:
            # Exploit: choose best Q-value
            best_q = -float('inf')
            action_idx = node.bandit_current_action
            for idx, q in node.bandit_q_values.items():
                if q > best_q:
                    best_q = q
                    action_idx = idx

        # Apply selected action
        node.bandit_current_action = action_idx
        node.aggressiveness = self.m_actions[action_idx]
        node.bandit_was_ch = False  # Reset CH flag for this round

    def _bandit_update_q_values(self) -> None:
        """
        Update Q-values for all nodes based on energy-efficiency rewards.

        NEW reward function (learns "be CH when it's efficient for me"):
        - For CH who survived: reward = clip((ref - cost_per_bit) / (ref + eps), -1, +1)
          where cost_per_bit = spent_energy / bits_out
          and ref = EMA of median(cost_per_bit of all CHs this epoch)
        - For CH who died: reward = -1.0 (penalty for over-aggression)
        - For non-CH: reward = 0.0 (neutral)

        Q-value update: Q(a) = Q(a) + alpha * (reward - Q(a))
        """
        if not self.use_bandit:
            return

        # Get config values for bits_out calculation
        data_bits = int(self.config.get('packets', {}).get('data_size', 32000))
        agg_ratio = float(self.config.get('clustering', {}).get('aggregation_ratio', 0.5))
        eps = 1e-9

        # Step 1: Calculate cost_per_bit for all CHs
        ch_costs = []  # List of (node, cost_per_bit)
        for cluster in self.clusters:
            ch = cluster.head
            if not ch.bandit_was_ch:
                continue

            # Get spent energy (set by base class after data phase)
            spent = getattr(ch, '_spent_energy', 0.0)

            # Calculate bits_out (same formula as energy model)
            num_members = len(cluster.get_alive_members())
            bits_out = int(data_bits * (num_members + 1) * agg_ratio)

            if bits_out > 0:
                cost_per_bit = spent / (bits_out + eps)
                ch_costs.append((ch, cost_per_bit))

        # Step 2: Calculate reference (median of CH costs) and update EMA
        if ch_costs:
            costs_only = [c for _, c in ch_costs]
            median_cost = float(np.median(costs_only))

            # Update EMA of reference
            if self.ref_ema is None:
                self.ref_ema = median_cost
            else:
                self.ref_ema = (1 - self.ref_ema_rho) * self.ref_ema + self.ref_ema_rho * median_cost
        else:
            # No CHs this round, keep old ref
            if self.ref_ema is None:
                self.ref_ema = 1e-6  # Small default

        ref = self.ref_ema + eps

        # Step 3: Calculate rewards and update Q-values
        # Build a map of node -> cost_per_bit for CHs
        ch_cost_map = {node.id: cost for node, cost in ch_costs}

        for node in self.network.nodes:
            action_idx = node.bandit_current_action

            # Determine reward
            if not node.bandit_was_ch:
                # Non-CH: neutral reward
                reward = 0.0
            elif not node.is_alive:
                # CH who died: strong penalty
                reward = -1.0
            else:
                # CH who survived: reward based on efficiency
                cost = ch_cost_map.get(node.id, ref)  # Use ref if not found
                # Positive if better than ref, negative if worse
                reward = (ref - cost) / (ref + eps)
                reward = max(-1.0, min(1.0, reward))  # Clip to [-1, 1]

            # Update Q-value using exponential moving average
            old_q = node.bandit_q_values.get(action_idx, 0.5)
            new_q = old_q + self.bandit_alpha * (reward - old_q)
            node.bandit_q_values[action_idx] = new_q

            # Update action count
            node.bandit_action_counts[action_idx] = node.bandit_action_counts.get(action_idx, 0) + 1

    def elect_cluster_heads(self) -> list[Node]:
        """
        Elect cluster heads based on epoch.
        - Epoch 0: Bootstrap with timer-based seed selection
        - Epoch >= 1: Re-use existing structure or global reselection
        """
        # Bandit: Select aggressiveness for all nodes at start of epoch
        for node in self.network.get_alive_nodes():
            self._bandit_select_action(node)

        if self.epoch == 1:
            # Bootstrap phase
            return self._bootstrap_seed_selection()
        elif self.epoch % self.recluster_interval == 0:
            # Periodic global re-clustering
            return self._global_ch_reselection()
        else:
            # Intra-cluster auction for next CH
            return self._intra_cluster_auction()

    def _bootstrap_seed_selection(self) -> list[Node]:
        """
        Bootstrap phase (Epoch 0): Timer-based seed CH selection.

        Nodes with higher bids have lower timers and announce first.
        Once a node becomes a seed, nearby nodes (within R_join) become members.
        """
        alive_nodes = self.network.get_alive_nodes()

        # Calculate bids for all nodes
        for node in alive_nodes:
            node.bid = self.calculate_bid(node)

        # Sort by bid descending (higher bid = announces first)
        sorted_nodes = sorted(alive_nodes, key=lambda n: n.bid, reverse=True)

        seeds = []
        assigned = set()

        for node in sorted_nodes:
            if node.id in assigned:
                continue

            # Check spacing: no existing seed within spacing_radius
            too_close = False
            for seed in seeds:
                if self.network.get_distance(node, seed) < self.spacing_radius:
                    too_close = True
                    break

            if not too_close:
                # Become seed (CH)
                seeds.append(node)
                assigned.add(node.id)

                # Mark nearby nodes as assigned (they'll join this seed)
                for other in alive_nodes:
                    if other.id not in assigned:
                        if self.network.get_distance(node, other) <= self.join_radius:
                            assigned.add(other.id)

                # Control message: SEED_ANNOUNCE broadcast (fixed radius)
                self.ctrl_broadcast_fixed(node, self.join_radius, self.ctrl_bits_announce)

        return seeds

    def _global_ch_reselection(self) -> list[Node]:
        """
        Global CH reselection with spacing constraint.
        Similar to bootstrap but considers current debt.
        """
        alive_nodes = self.network.get_alive_nodes()

        # Calculate bids for all nodes
        for node in alive_nodes:
            node.bid = self.calculate_bid(node)

        # Sort by bid descending
        sorted_nodes = sorted(alive_nodes, key=lambda n: n.bid, reverse=True)

        heads = []
        assigned = set()

        for node in sorted_nodes:
            if node.id in assigned:
                continue

            # Check spacing
            too_close = False
            for head in heads:
                if self.network.get_distance(node, head) < self.spacing_radius:
                    too_close = True
                    break

            if not too_close:
                heads.append(node)
                assigned.add(node.id)

                # Mark nearby nodes
                for other in alive_nodes:
                    if other.id not in assigned:
                        if self.network.get_distance(node, other) <= self.join_radius:
                            assigned.add(other.id)

                # Control message: CH announce broadcast (fixed radius)
                self.ctrl_broadcast_fixed(node, self.join_radius, self.ctrl_bits_announce)

        return heads

    def _intra_cluster_auction(self) -> list[Node]:
        """
        Intra-cluster auction: Current CH collects bids and selects next CH.

        With implicit bidding: Only willing nodes send bids (reduced overhead).
        Each cluster independently selects its next CH based on bids.
        """
        new_heads = []

        for cluster in self.clusters:
            if not cluster.is_head_alive():
                # CH died, try to promote backup or skip
                if cluster.backup and cluster.backup.is_alive:
                    new_heads.append(cluster.backup)
                    # Backup promotion announcement to cluster
                    self.ctrl_broadcast_to_set(cluster.backup, cluster.all_nodes,
                                               self.ctrl_bits_announce)
                continue

            # Get all alive nodes in cluster
            all_candidates = [n for n in cluster.all_nodes if n.is_alive]

            if not all_candidates:
                continue

            if self.implicit_bidding:
                # IMPLICIT BIDDING: Only willing nodes send bids
                willing_candidates = [n for n in all_candidates if self.wants_to_be_ch(n)]

                # Fallback: if no willing nodes, current CH stays or pick best
                if not willing_candidates:
                    # Use current CH if still alive
                    if cluster.head.is_alive:
                        new_heads.append(cluster.head)
                        # Just announcement (no bids collected)
                        self.ctrl_broadcast_to_set(cluster.head, cluster.all_nodes,
                                                   self.ctrl_bits_result)
                        continue
                    else:
                        # Force selection from all candidates
                        willing_candidates = all_candidates

                # Calculate bids only for willing nodes
                for node in willing_candidates:
                    node.bid = self.calculate_bid(node)

                # Sort by bid, select top
                willing_candidates.sort(key=lambda n: n.bid, reverse=True)
                primary = willing_candidates[0]
                new_heads.append(primary)

                # Control messages: willing nodes send bids to current CH (unicast)
                for candidate in willing_candidates:
                    self.ctrl_unicast(candidate, cluster.head, self.ctrl_bits_bid)

                # Result broadcast from current CH to cluster
                self.ctrl_broadcast_to_set(cluster.head, cluster.all_nodes,
                                           self.ctrl_bits_result)
            else:
                # ORIGINAL: All nodes send bids
                for node in all_candidates:
                    node.bid = self.calculate_bid(node)

                all_candidates.sort(key=lambda n: n.bid, reverse=True)
                primary = all_candidates[0]
                new_heads.append(primary)

                # Control messages: all candidates send bids (unicast)
                for candidate in all_candidates:
                    self.ctrl_unicast(candidate, cluster.head, self.ctrl_bits_bid)

                # Result broadcast
                self.ctrl_broadcast_to_set(cluster.head, cluster.all_nodes,
                                           self.ctrl_bits_result)

        return new_heads

    def form_clusters(self, heads: list[Node]) -> list[Cluster]:
        """
        Form clusters by assigning each non-CH node to nearest CH.
        Also selects backup CH for each cluster.
        """
        if not heads:
            return []

        # Use utility function to form basic clusters
        clusters = form_clusters_from_heads(heads, self.network.nodes, self.network)

        # Select backup for each cluster (second-highest bid)
        for cluster in clusters:
            candidates = [n for n in cluster.members if n.is_alive]
            if candidates:
                # Calculate bids if not already done
                for node in candidates:
                    if node.bid == 0:
                        node.bid = self.calculate_bid(node)

                candidates.sort(key=lambda n: n.bid, reverse=True)
                # Remove top candidate from members and make backup
                backup = candidates[0]
                cluster.members.remove(backup)
                cluster.set_backup(backup)

        # Control messages: join requests from members (unicast to CH)
        for cluster in clusters:
            for member in cluster.members:
                if member.is_alive:
                    self.ctrl_unicast(member, cluster.head, self.ctrl_bits_join)
            # Backup also sends join (it's not in members list after removal)
            if cluster.backup and cluster.backup.is_alive:
                self.ctrl_unicast(cluster.backup, cluster.head, self.ctrl_bits_join)

        return clusters

    def _update_state(self):
        """
        Update fairness debt and bandit Q-values after each epoch.

        Debt update:
        D_i(t+1) = clip(D_i(t) + eta * (1{i=CH} - share_i), -D_max, D_max)

        Bandit update:
        Q(a) = Q(a) + alpha * (r - Q(a))
        """
        # Update fairness debt
        for cluster in self.clusters:
            if not cluster.all_nodes:
                continue

            # Calculate shares: share_i = E_i^(0) / sum(E_j^(0))
            shares = cluster.calculate_shares(alive_only=self.share_alive_only)

            for node in cluster.all_nodes:
                if not node.is_alive:
                    continue

                # Indicator: was this node the CH?
                was_ch = 1.0 if node.role == NodeRole.CLUSTER_HEAD else 0.0
                share_i = shares.get(node.id, 0.0)

                # Update debt
                new_debt = node.debt + self.eta * (was_ch - share_i)

                # Clip to [-D_max, D_max]
                node.debt = max(-self.d_max, min(self.d_max, new_debt))

        # Update bandit Q-values based on rewards
        self._bandit_update_q_values()


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
    abc = AuctionClustering(network, energy_model, config)
    abc.setup()

    # Run a few epochs
    print(f"=== {abc.name} Test ===\n")
    for _ in range(3):
        stats = abc.run_epoch()
        print(f"Epoch {stats['epoch']}: {stats['alive_nodes']} alive, "
              f"{stats['num_clusters']} clusters, "
              f"{stats['control_messages']} ctrl msgs, "
              f"E={stats['total_energy']:.3f}J")

    # Show cluster details
    print("\nCluster details:")
    for c in abc.clusters[:3]:
        print(f"  {c}")
