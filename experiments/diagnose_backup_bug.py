#!/usr/bin/env python3
"""
Diagnose the backup node bug.

Checks if backup nodes are participating in data transmission.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation import load_config
from src.algorithms.auction import AuctionClustering
from src.models.node import create_heterogeneous_nodes
from src.models.network import Network
from src.models.energy import EnergyModel
import numpy as np


def diagnose():
    config = load_config("config/default.yaml")

    net_cfg = config['network']
    energy_cfg = config['energy']

    np.random.seed(42)
    nodes = create_heterogeneous_nodes(
        n=net_cfg['num_nodes'],
        width=net_cfg['area_width'],
        height=net_cfg['area_height'],
        energy_mean=energy_cfg['initial_mean'],
        energy_std=energy_cfg['initial_std'],
        seed=42
    )

    network = Network(
        nodes,
        bs_x=net_cfg['bs_x'],
        bs_y=net_cfg['bs_y'],
        comm_range=net_cfg['comm_range']
    )

    energy_model = EnergyModel(
        e_elec=energy_cfg.get('e_elec', 50e-9),
        e_amp=energy_cfg.get('e_amp', 100e-12),
        e_da=energy_cfg.get('e_da', 5e-9),
    )

    algo = AuctionClustering(network, energy_model, config)
    algo.setup()

    print("=" * 70)
    print("BACKUP NODE BUG DIAGNOSIS")
    print("=" * 70)

    # Run a few epochs
    for epoch in range(1, 6):
        stats = algo.run_epoch()

        alive_nodes = len(network.get_alive_nodes())
        num_clusters = len(algo.clusters)
        num_chs = num_clusters

        # Count members that actually transmit
        transmitting_members = sum(len(c.get_alive_members()) for c in algo.clusters)

        # Count backup nodes
        backup_count = sum(1 for c in algo.clusters if c.backup is not None and c.backup.is_alive)

        # Expected: alive_nodes = CHs + transmitting_members + backups
        # If backups are silent: transmitting_members = alive_nodes - num_chs - backup_count

        expected_transmitting = alive_nodes - num_chs  # Should be this if backups transmit
        actual_transmitting = transmitting_members
        silent_nodes = expected_transmitting - actual_transmitting

        print(f"\nEpoch {epoch}:")
        print(f"  Alive nodes:          {alive_nodes}")
        print(f"  Cluster heads:        {num_chs}")
        print(f"  Backup nodes:         {backup_count}")
        print(f"  Transmitting members: {actual_transmitting}")
        print(f"  Expected (no bug):    {expected_transmitting}")
        print(f"  SILENT NODES:         {silent_nodes} {'<-- BUG!' if silent_nodes > 0 else '(OK)'}")
        print(f"  Data packets:         {stats['data_packets']}")

    print("\n" + "=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)

    if silent_nodes > 0:
        print(f"\nBUG CONFIRMED: {silent_nodes} nodes per epoch are NOT transmitting data!")
        print("These backup nodes save energy unfairly, inflating FND/LND metrics.")
        print("\nIMPACT ESTIMATE:")
        print(f"  - {backup_count} backups * ~{silent_nodes} silent = significant energy savings")
        print(f"  - This could explain 10-30% better FND compared to baselines")
    else:
        print("\nNo bug detected - all non-CH nodes are transmitting.")


if __name__ == "__main__":
    diagnose()
