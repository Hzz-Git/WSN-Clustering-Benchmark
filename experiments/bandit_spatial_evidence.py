#!/usr/bin/env python3
"""
Task D2: Bandit Spatial Strategy Evidence

Output:
1. mean_m vs distance-to-BS by quantiles (tracked every 50 epochs)
2. CH cost_per_bit median trend over time
3. Q-value evolution by distance group
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import copy
import csv
import numpy as np
from datetime import datetime
from src.simulation import load_config, Simulation
from src.algorithms.auction import AuctionClustering


def pearsonr(x, y):
    """Simple Pearson correlation."""
    x, y = np.array(x), np.array(y)
    if len(x) < 2:
        return 0.0
    mx, my = np.mean(x), np.mean(y)
    sx, sy = np.std(x), np.std(y)
    if sx == 0 or sy == 0:
        return 0.0
    return np.mean((x - mx) * (y - my)) / (sx * sy)


class BanditTracker:
    """Track bandit learning metrics over epochs."""

    def __init__(self, network, algorithm):
        self.network = network
        self.algorithm = algorithm
        self.history = []  # List of per-epoch snapshots

        # Compute distance quantiles
        distances = [network.get_distance_to_bs(n) for n in network.nodes]
        self.q25, self.q50, self.q75 = np.percentile(distances, [25, 50, 75])

    def categorize_node(self, node):
        """Categorize node by distance to BS."""
        d = self.network.get_distance_to_bs(node)
        if d <= self.q25:
            return 'near'
        elif d <= self.q75:
            return 'mid'
        else:
            return 'far'

    def snapshot(self, epoch):
        """Take a snapshot of bandit state."""
        alive = [n for n in self.network.nodes if n.is_alive]
        if not alive:
            return

        # m by distance group
        m_by_group = {'near': [], 'mid': [], 'far': []}
        for node in alive:
            group = self.categorize_node(node)
            m_by_group[group].append(node.aggressiveness)

        # Overall statistics
        all_m = [n.aggressiveness for n in alive]
        all_distances = [self.network.get_distance_to_bs(n) for n in alive]
        corr = pearsonr(all_m, all_distances)

        # Q-values by group (average Q for current action)
        q_by_group = {'near': [], 'mid': [], 'far': []}
        for node in alive:
            group = self.categorize_node(node)
            action = node.bandit_current_action
            q = node.bandit_q_values.get(action, 0.5)
            q_by_group[group].append(q)

        snapshot = {
            'epoch': epoch,
            'alive': len(alive),
            'm_mean': np.mean(all_m),
            'm_std': np.std(all_m),
            'm_near': np.mean(m_by_group['near']) if m_by_group['near'] else np.nan,
            'm_mid': np.mean(m_by_group['mid']) if m_by_group['mid'] else np.nan,
            'm_far': np.mean(m_by_group['far']) if m_by_group['far'] else np.nan,
            'n_near': len(m_by_group['near']),
            'n_mid': len(m_by_group['mid']),
            'n_far': len(m_by_group['far']),
            'm_dist_corr': corr,
            'q_near': np.mean(q_by_group['near']) if q_by_group['near'] else np.nan,
            'q_mid': np.mean(q_by_group['mid']) if q_by_group['mid'] else np.nan,
            'q_far': np.mean(q_by_group['far']) if q_by_group['far'] else np.nan,
            'ref_ema': self.algorithm.ref_ema if self.algorithm.ref_ema else 0,
        }
        self.history.append(snapshot)


def run_bandit_tracking():
    config = load_config("config/default.yaml")
    config['simulation']['max_epochs'] = 300
    config['auction']['use_bandit'] = True

    print("="*80)
    print("BANDIT SPATIAL STRATEGY EVIDENCE")
    print("="*80)

    # Create simulation manually to access algorithm
    from src.models.node import create_heterogeneous_nodes
    from src.models.network import Network
    from src.models.energy import EnergyModel

    seed = 42
    np.random.seed(seed)

    # Create network
    net_cfg = config.get('network', {})
    energy_cfg = config.get('energy', {})

    nodes = create_heterogeneous_nodes(
        n=net_cfg.get('num_nodes', 50),
        width=net_cfg.get('area_width', 100.0),
        height=net_cfg.get('area_height', 100.0),
        energy_mean=energy_cfg.get('initial_mean', 2.0),
        energy_std=energy_cfg.get('initial_std', 0.2),
        energy_min=energy_cfg.get('initial_min', 0.1),
        seed=seed
    )
    network = Network(
        nodes,
        bs_x=net_cfg.get('bs_x', 50.0),
        bs_y=net_cfg.get('bs_y', 100.0),
        comm_range=net_cfg.get('comm_range', 30.0)
    )
    energy_model = EnergyModel(
        e_elec=energy_cfg.get('e_elec', 50e-9),
        e_amp=energy_cfg.get('e_amp', 100e-12),
        e_da=energy_cfg.get('e_da', 5e-9),
    )

    # Create algorithm
    abc = AuctionClustering(network, energy_model, config)
    abc.setup()

    # Create tracker
    tracker = BanditTracker(network, abc)

    print(f"Distance quantiles: Q25={tracker.q25:.1f}m, Q50={tracker.q50:.1f}m, Q75={tracker.q75:.1f}m")

    # Run simulation with tracking
    max_epochs = config['simulation']['max_epochs']
    snapshot_interval = 10

    for epoch in range(1, max_epochs + 1):
        stats = abc.run_epoch()

        if epoch % snapshot_interval == 0 or epoch == 1:
            tracker.snapshot(epoch)

        if not abc.is_network_alive():
            print(f"Network dead at epoch {epoch}")
            break

        if epoch % 50 == 0:
            alive = len([n for n in network.nodes if n.is_alive])
            print(f"Epoch {epoch}: alive={alive}, ref_ema={abc.ref_ema:.2e}")

    # Final snapshot
    tracker.snapshot(epoch)

    # Output CSV
    output_dir = Path("results/data/bandit_evidence")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = output_dir / f"bandit_spatial_{timestamp}.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=tracker.history[0].keys())
        writer.writeheader()
        writer.writerows(tracker.history)

    print(f"\nResults saved to: {csv_path}")

    # Print summary
    print("\n" + "="*80)
    print("SPATIAL STRATEGY SUMMARY")
    print("="*80)

    print("\n--- m (aggressiveness) by Distance Group Over Time ---")
    print(f"{'Epoch':>6} | {'Alive':>5} | {'m_near':>7} | {'m_mid':>7} | {'m_far':>7} | {'corr':>6}")
    print("-" * 55)

    for snap in tracker.history[::3]:  # Every 3rd snapshot
        print(f"{snap['epoch']:>6} | {snap['alive']:>5} | "
              f"{snap['m_near']:>7.3f} | {snap['m_mid']:>7.3f} | {snap['m_far']:>7.3f} | "
              f"{snap['m_dist_corr']:>6.3f}")

    # Final analysis
    print("\n--- Final State Analysis ---")
    final = tracker.history[-1]
    early = tracker.history[min(5, len(tracker.history)-1)]

    print(f"\nEarly (epoch {early['epoch']}):")
    print(f"  m_near={early['m_near']:.3f}, m_mid={early['m_mid']:.3f}, m_far={early['m_far']:.3f}")
    print(f"  m-distance correlation: {early['m_dist_corr']:.3f}")

    print(f"\nFinal (epoch {final['epoch']}):")
    print(f"  m_near={final['m_near']:.3f}, m_mid={final['m_mid']:.3f}, m_far={final['m_far']:.3f}")
    print(f"  m-distance correlation: {final['m_dist_corr']:.3f}")

    # Learning signal
    m_spread_early = early['m_near'] - early['m_far']
    m_spread_final = final['m_near'] - final['m_far']

    print(f"\n--- Learning Signal ---")
    print(f"  m spread (near - far) early: {m_spread_early:.3f}")
    print(f"  m spread (near - far) final: {m_spread_final:.3f}")
    print(f"  Spread increase: {m_spread_final - m_spread_early:.3f}")

    if m_spread_final > m_spread_early + 0.05:
        print("\n  CONCLUSION: Bandit learned spatial strategy (near=aggressive, far=conservative)")
    elif abs(final['m_dist_corr']) > 0.2:
        print(f"\n  CONCLUSION: Spatial correlation maintained (r={final['m_dist_corr']:.3f})")
    else:
        print("\n  CONCLUSION: Weak spatial signal (may need more epochs or tuning)")

    # ref_ema trend
    print("\n--- ref_ema (Cost Reference) Trend ---")
    ref_values = [s['ref_ema'] for s in tracker.history if s['ref_ema'] > 0]
    if ref_values:
        print(f"  Initial: {ref_values[0]:.2e} J/bit")
        print(f"  Final: {ref_values[-1]:.2e} J/bit")
        print(f"  Trend: {'increasing' if ref_values[-1] > ref_values[0] else 'decreasing'}")


if __name__ == "__main__":
    run_bandit_tracking()
