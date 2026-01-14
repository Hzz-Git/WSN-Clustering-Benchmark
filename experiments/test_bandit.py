#!/usr/bin/env python3
"""
Test script to verify bandit learning is working correctly.
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
from collections import Counter


def test_bandit():
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
    print("BANDIT LEARNING TEST")
    print("=" * 70)
    print(f"Action space (m values): {algo.m_actions}")
    print(f"Exploration rate (epsilon): {algo.bandit_epsilon}")
    print(f"Learning rate (alpha): {algo.bandit_alpha}")
    print(f"Bandit enabled: {algo.use_bandit}")
    print("=" * 70)

    # Track aggressiveness distribution over time
    m_history = []

    # Run simulation
    for epoch in range(1, 101):
        stats = algo.run_epoch()

        # Collect current aggressiveness values
        m_values = [n.aggressiveness for n in network.nodes if n.is_alive]
        m_history.append(m_values)

        if epoch in [1, 10, 25, 50, 100]:
            print(f"\n--- Epoch {epoch} ---")
            print(f"Alive nodes: {stats['alive_nodes']}")

            # Show aggressiveness distribution
            m_counts = Counter([round(m, 1) for m in m_values])
            print(f"Aggressiveness distribution: {dict(sorted(m_counts.items()))}")

            # Show Q-value stats for a sample node
            sample_node = [n for n in network.nodes if n.is_alive][0]
            print(f"\nSample node {sample_node.id} Q-values:")
            for action_idx, m_val in enumerate(algo.m_actions):
                q_val = sample_node.bandit_q_values.get(action_idx, 0)
                count = sample_node.bandit_action_counts.get(action_idx, 0)
                print(f"  m={m_val}: Q={q_val:.3f}, count={count}")

    # Final analysis
    print("\n" + "=" * 70)
    print("FINAL ANALYSIS")
    print("=" * 70)

    # Compare initial vs final aggressiveness distribution
    initial_m = m_history[0]
    final_m = m_history[-1]

    print(f"\nInitial m distribution: mean={np.mean(initial_m):.3f}, std={np.std(initial_m):.3f}")
    print(f"Final m distribution:   mean={np.mean(final_m):.3f}, std={np.std(final_m):.3f}")

    # Show which actions were most popular
    final_counts = Counter([round(m, 1) for m in final_m])
    print(f"\nFinal aggressiveness distribution:")
    for m_val in sorted(final_counts.keys()):
        pct = final_counts[m_val] / len(final_m) * 100
        bar = "#" * int(pct / 2)
        print(f"  m={m_val}: {final_counts[m_val]:3d} ({pct:5.1f}%) {bar}")

    # Check if learning happened (Q-values diverged from initial)
    q_value_changes = []
    for node in network.nodes:
        if node.is_alive:
            for action_idx in range(len(algo.m_actions)):
                q = node.bandit_q_values.get(action_idx, 0.5)
                q_value_changes.append(abs(q - 0.5))  # Initial was 0.5

    print(f"\nQ-value deviation from initial (0.5):")
    print(f"  Mean: {np.mean(q_value_changes):.4f}")
    print(f"  Max:  {np.max(q_value_changes):.4f}")

    if np.mean(q_value_changes) > 0.05:
        print("\nBANDIT LEARNING IS WORKING!")
    else:
        print("\nWARNING: Q-values haven't changed much - check implementation")

    print("=" * 70)


if __name__ == "__main__":
    test_bandit()
