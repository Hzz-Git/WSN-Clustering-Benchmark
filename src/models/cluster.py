"""
Cluster class for managing cluster state.

A cluster consists of:
- One Cluster Head (CH) - Primary
- Optional Backup CH
- Multiple Cluster Members (CM)
"""

from typing import Optional
from .node import Node, NodeRole


class Cluster:
    """
    Represents a cluster in the WSN.

    Attributes:
        id: Cluster identifier (usually same as CH node id)
        head: The cluster head node
        backup: The backup CH node (optional)
        members: List of cluster member nodes
    """

    def __init__(self, head: Node, cluster_id: Optional[int] = None):
        self.id = cluster_id if cluster_id is not None else head.id
        self.head = head
        self.backup: Optional[Node] = None
        self.members: list[Node] = []

        # Set the head's role
        head.become_cluster_head()

    @property
    def all_nodes(self) -> list[Node]:
        """Get all nodes in the cluster (head + backup + members)."""
        nodes = [self.head] + self.members
        if self.backup is not None:
            nodes.append(self.backup)
        return nodes

    @property
    def size(self) -> int:
        """Total number of nodes in the cluster."""
        return len(self.all_nodes)

    @property
    def member_count(self) -> int:
        """Number of members (excluding head and backup)."""
        return len(self.members)

    def add_member(self, node: Node):
        """Add a node as a cluster member."""
        node.join_cluster(self.head.id)
        self.members.append(node)

    def set_backup(self, node: Node):
        """Set a node as the backup CH."""
        node.become_backup(self.head.id)
        self.backup = node

    def promote_backup(self) -> bool:
        """
        Promote backup to primary CH if head dies.

        Returns:
            True if backup was promoted, False if no backup available
        """
        if self.backup is not None and self.backup.is_alive:
            self.head = self.backup
            self.head.become_cluster_head()
            self.backup = None
            return True
        return False

    def is_head_alive(self) -> bool:
        """Check if the cluster head is still alive."""
        return self.head.is_alive

    def get_alive_members(self) -> list[Node]:
        """Get all alive members (including backup if alive and not promoted to head)."""
        alive = [m for m in self.members if m.is_alive]
        # Include backup as a transmitting member (it should also send data)
        if self.backup is not None and self.backup.is_alive and self.backup != self.head:
            alive.append(self.backup)
        return alive

    def get_total_initial_energy(self) -> float:
        """Get sum of initial energy of all nodes in cluster."""
        return sum(n.initial_energy for n in self.all_nodes)

    def calculate_shares(self, alive_only: bool = False) -> dict[int, float]:
        """
        Calculate capacity-weighted duty shares for fairness.
        share_i = E_i^(0) / sum(E_j^(0)) for j in cluster

        Args:
            alive_only: If True, only consider alive nodes in denominator.
                        This prevents dead nodes from distorting shares.

        Returns:
            Dict mapping node_id to share value
        """
        if alive_only:
            # Only alive nodes contribute to total
            nodes = [n for n in self.all_nodes if n.is_alive]
        else:
            # All nodes (original behavior)
            nodes = self.all_nodes

        if not nodes:
            return {}

        total = sum(n.initial_energy for n in nodes)
        if total == 0:
            return {n.id: 0.0 for n in nodes}

        return {n.id: n.initial_energy / total for n in nodes}

    def __repr__(self) -> str:
        backup_str = f", backup={self.backup.id}" if self.backup else ""
        return f"Cluster(id={self.id}, head={self.head.id}{backup_str}, members={len(self.members)})"


def form_clusters_from_heads(heads: list[Node], all_nodes: list[Node], network) -> list[Cluster]:
    """
    Form clusters by assigning each non-CH node to nearest CH.

    Args:
        heads: List of nodes elected as cluster heads
        all_nodes: All nodes in the network
        network: Network object for distance calculations

    Returns:
        List of Cluster objects
    """
    clusters = {h.id: Cluster(h) for h in heads}
    head_ids = {h.id for h in heads}

    # Assign each non-CH alive node to nearest CH
    for node in all_nodes:
        if not node.is_alive or node.id in head_ids:
            continue

        # Find nearest CH
        min_dist = float('inf')
        nearest_ch = None
        for head in heads:
            if not head.is_alive:
                continue
            dist = network.get_distance(node, head)
            if dist < min_dist:
                min_dist = dist
                nearest_ch = head

        if nearest_ch is not None:
            clusters[nearest_ch.id].add_member(node)

    return list(clusters.values())


if __name__ == "__main__":
    from .node import create_random_nodes

    # Create test nodes
    nodes = create_random_nodes(10, 100, 100, seed=42)

    # Create a cluster with node 0 as head
    cluster = Cluster(nodes[0])
    cluster.add_member(nodes[1])
    cluster.add_member(nodes[2])
    cluster.set_backup(nodes[3])

    print(cluster)
    print(f"All nodes: {[n.id for n in cluster.all_nodes]}")
    print(f"Shares: {cluster.calculate_shares()}")

    # Test roles
    print(f"\nNode 0 (head): {nodes[0].role.name}")
    print(f"Node 1 (member): {nodes[1].role.name}")
    print(f"Node 3 (backup): {nodes[3].role.name}")
