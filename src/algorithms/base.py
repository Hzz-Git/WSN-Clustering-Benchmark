"""
Base class for clustering algorithms.

All clustering algorithms (ABC, HEED, LEACH) inherit from this base class
and implement the same interface for fair comparison.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from ..models.node import Node
from ..models.network import Network
from ..models.cluster import Cluster
from ..models.energy import EnergyModel


class ClusteringAlgorithm(ABC):
    """
    Abstract base class for WSN clustering algorithms.

    All algorithms must implement:
    - setup(): Initialize algorithm state
    - elect_cluster_heads(): CH election phase
    - form_clusters(): Cluster formation phase
    - run_epoch(): Execute one complete epoch
    """

    def __init__(
        self,
        network: Network,
        energy_model: EnergyModel,
        config: dict
    ):
        """
        Initialize the clustering algorithm.

        Args:
            network: Network object with nodes and topology
            energy_model: Energy consumption model
            config: Algorithm configuration parameters
        """
        self.network = network
        self.energy_model = energy_model
        self.config = config
        self.clusters: list[Cluster] = []
        self.epoch: int = 0
        self.control_messages: int = 0  # Track overhead count
        self.control_energy_j: float = 0.0  # Track control energy consumption (Joules)

    @property
    def name(self) -> str:
        """Algorithm name for logging."""
        return self.__class__.__name__

    # =========================================================================
    # Control Message Helpers (unified energy accounting for all algorithms)
    # =========================================================================

    def ctrl_unicast(self, sender: Node, receiver: Node, bits: int) -> float:
        """
        Send a unicast control message from sender to receiver.

        Deducts TX energy from sender, RX energy from receiver.

        Args:
            sender: Node sending the message
            receiver: Node receiving the message
            bits: Size of control message in bits

        Returns:
            Total energy consumed (TX + RX) in Joules
        """
        if not sender.is_alive:
            return 0.0

        dist = self.network.get_distance(sender, receiver)
        tx_energy = self.energy_model.control_tx_energy(bits, dist)
        rx_energy = self.energy_model.control_rx_energy(bits)

        sender.consume_energy(tx_energy)
        if receiver.is_alive:
            receiver.consume_energy(rx_energy)

        total = tx_energy + (rx_energy if receiver.is_alive else 0.0)
        self.control_messages += 1
        self.control_energy_j += total
        return total

    def ctrl_broadcast_fixed(self, sender: Node, radius: float, bits: int) -> float:
        """
        Broadcast a control message with fixed transmission radius.

        Use for discovery/advertisement where TX power should be consistent
        regardless of who actually receives (avoids d² bias from sparse areas).

        Args:
            sender: Node broadcasting
            radius: Fixed transmission radius (e.g., join_radius, comm_range)
            bits: Size of control message in bits

        Returns:
            Total energy consumed in Joules
        """
        if not sender.is_alive:
            return 0.0

        # TX energy based on fixed radius (not d_max to receivers)
        tx_energy = self.energy_model.control_tx_energy(bits, radius)
        sender.consume_energy(tx_energy)

        # All alive neighbors within radius receive
        receivers = [n for n in self.network.get_neighbors(sender, radius)
                     if n.is_alive and n.id != sender.id]

        rx_energy_per = self.energy_model.control_rx_energy(bits)
        total_rx = 0.0
        for r in receivers:
            r.consume_energy(rx_energy_per)
            total_rx += rx_energy_per

        total = tx_energy + total_rx
        self.control_messages += 1
        self.control_energy_j += total
        return total

    def ctrl_broadcast_to_set(self, sender: Node, receivers: list[Node], bits: int) -> float:
        """
        Broadcast a control message to a known set of receivers.

        Use for cluster-wide notifications where receiver set is already known
        (e.g., auction result to cluster members). TX power based on d_max.

        Args:
            sender: Node broadcasting
            receivers: List of nodes that will receive (filtered for alive)
            bits: Size of control message in bits

        Returns:
            Total energy consumed in Joules
        """
        if not sender.is_alive:
            return 0.0

        # Filter alive receivers, exclude sender
        alive_receivers = [r for r in receivers if r.is_alive and r.id != sender.id]

        if not alive_receivers:
            # No receivers, still count message but minimal energy
            self.control_messages += 1
            return 0.0

        # TX energy based on max distance to any receiver
        d_max = max(self.network.get_distance(sender, r) for r in alive_receivers)
        tx_energy = self.energy_model.control_tx_energy(bits, d_max)
        sender.consume_energy(tx_energy)

        # All receivers pay RX
        rx_energy_per = self.energy_model.control_rx_energy(bits)
        total_rx = 0.0
        for r in alive_receivers:
            r.consume_energy(rx_energy_per)
            total_rx += rx_energy_per

        total = tx_energy + total_rx
        self.control_messages += 1
        self.control_energy_j += total
        return total

    def ctrl_broadcast_from_bs(self, receivers: list[Node], bits: int) -> float:
        """
        Broadcast from base station to nodes.

        BS has unlimited energy, so only receivers pay RX energy.
        Use for LEACH-C centralized assignments.

        Args:
            receivers: List of nodes receiving the broadcast
            bits: Size of control message in bits

        Returns:
            Total RX energy consumed in Joules
        """
        alive_receivers = [r for r in receivers if r.is_alive]

        rx_energy_per = self.energy_model.control_rx_energy(bits)
        total_rx = 0.0
        for r in alive_receivers:
            r.consume_energy(rx_energy_per)
            total_rx += rx_energy_per

        self.control_messages += 1
        self.control_energy_j += total_rx
        return total_rx

    # =========================================================================

    @abstractmethod
    def setup(self):
        """Initialize algorithm state before first epoch."""
        pass

    @abstractmethod
    def elect_cluster_heads(self) -> list[Node]:
        """
        Elect cluster heads for the current epoch.

        Returns:
            List of nodes elected as cluster heads
        """
        pass

    @abstractmethod
    def form_clusters(self, heads: list[Node]) -> list[Cluster]:
        """
        Form clusters around elected cluster heads.

        Args:
            heads: List of cluster head nodes

        Returns:
            List of formed clusters
        """
        pass

    def run_epoch(self) -> dict:
        """
        Execute one complete epoch of the algorithm.

        Returns:
            Dict with epoch statistics
        """
        self.epoch += 1
        self.control_messages = 0
        self.control_energy_j = 0.0

        # Reset nodes for new epoch
        self.network.reset_all_nodes()

        # Phase 1: CH Election
        heads = self.elect_cluster_heads()

        # Phase 2: Cluster Formation
        self.clusters = self.form_clusters(heads)

        # Snapshot energy BEFORE data phase (for bandit reward calculation)
        self._snapshot_energy_before()

        # Phase 3: Data Transmission
        self._data_transmission_phase()

        # Calculate spent energy AFTER data phase
        self._calculate_spent_energy()

        # Phase 4: Update algorithm state (fairness, etc.)
        self._update_state()

        return self._collect_epoch_stats()

    def _snapshot_energy_before(self):
        """Snapshot energy of all nodes before data transmission."""
        for node in self.network.nodes:
            node._energy_before_data = node.current_energy

    def _calculate_spent_energy(self):
        """Calculate energy spent during data phase for each node."""
        for node in self.network.nodes:
            before = getattr(node, '_energy_before_data', node.current_energy)
            node._spent_energy = max(0.0, before - node.current_energy)

    def _data_transmission_phase(self):
        """
        Execute data transmission phase.
        Members transmit to CH, CH aggregates and transmits to BS.
        """
        data_bits = int(self.config.get('packets', {}).get('data_size', 32000))
        agg_ratio = float(self.config.get('clustering', {}).get('aggregation_ratio', 0.5))

        for cluster in self.clusters:
            if not cluster.is_head_alive():
                continue

            ch = cluster.head
            alive_members = cluster.get_alive_members()

            # Members transmit to CH
            for member in alive_members:
                dist_to_ch = self.network.get_distance(member, ch)
                energy = self.energy_model.member_energy_per_round(data_bits, dist_to_ch)
                member.consume_energy(energy)

            # CH receives, aggregates, and transmits to BS
            dist_to_bs = self.network.get_distance_to_bs(ch)
            energy = self.energy_model.ch_energy_per_round(
                len(alive_members), data_bits, dist_to_bs, agg_ratio
            )
            ch.consume_energy(energy)

    def _update_state(self):
        """Update algorithm-specific state after epoch. Override in subclasses."""
        pass

    def _collect_epoch_stats(self) -> dict:
        """Collect statistics for the current epoch."""
        alive_nodes = self.network.get_alive_nodes()
        alive = len(alive_nodes)
        total_energy = self.network.get_total_energy()

        # Energy statistics for fairness analysis
        if alive > 0:
            energies = [n.current_energy for n in alive_nodes]
            energy_mean = np.mean(energies)
            energy_std = np.std(energies)
            energy_min = np.min(energies)
            energy_max = np.max(energies)
        else:
            energy_mean = energy_std = energy_min = energy_max = 0.0

        # Cluster statistics
        cluster_sizes = [c.size for c in self.clusters]
        num_chs = len(self.clusters)
        avg_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0
        cluster_size_std = np.std(cluster_sizes) if cluster_sizes else 0

        # Throughput: count data packets delivered (1 per alive member + 1 per CH)
        data_packets = sum(len(c.get_alive_members()) + 1 for c in self.clusters if c.is_head_alive())

        return {
            'epoch': self.epoch,
            'alive_nodes': alive,
            'total_energy': total_energy,
            'energy_mean': energy_mean,
            'energy_std': energy_std,
            'energy_min': energy_min,
            'energy_max': energy_max,
            'num_clusters': num_chs,
            'control_messages': self.control_messages,
            'control_energy_j': self.control_energy_j,
            'data_packets': data_packets,
            'cluster_sizes': cluster_sizes,
            'avg_cluster_size': avg_cluster_size,
            'cluster_size_std': cluster_size_std,
        }

    def is_network_alive(self) -> bool:
        """Check if any nodes are still alive."""
        return self.network.count_alive() > 0

    def get_clusters(self) -> list[Cluster]:
        """Get current clusters."""
        return self.clusters
