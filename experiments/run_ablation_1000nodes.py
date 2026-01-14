#!/usr/bin/env python3
"""
Full Ablation Study for ABC Algorithm - 1000 NODES

Tests all key components with larger network:
1. ABC-full: Complete algorithm (fairness ON, bandit ON)
2. ABC-no-debt: Fairness OFF (lambda=0)
3. ABC-no-bandit: Bandit OFF (fixed m=1.0)
4. ABC-minimal: Both OFF (baseline auction only)

1000 nodes in 300x300m area with 2.0J energy
"""

import sys
import copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation import load_config
from src.algorithms.auction import AuctionClustering
from src.models.node import create_heterogeneous_nodes
from src.models.network import Network
from src.models.energy import EnergyModel
import numpy as np


def gini_coefficient(values):
    """Calculate Gini coefficient."""
    values = np.array(values)
    values = values[values > 0]
    if len(values) == 0:
        return 0.0
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    cumsum = np.cumsum(sorted_vals)
    return (2 * np.sum((np.arange(1, n+1) * sorted_vals)) - (n + 1) * cumsum[-1]) / (n * cumsum[-1])


def run_single_trial(config: dict, seed: int, variant: str) -> dict:
    """Run a single trial with specific ablation variant."""
    cfg = copy.deepcopy(config)

    # 1000 NODE CONFIGURATION
    cfg['network']['num_nodes'] = 1000
    cfg['network']['area_width'] = 300.0  # Larger area for 1000 nodes
    cfg['network']['area_height'] = 300.0
    cfg['network']['bs_x'] = 150.0  # BS at center-top
    cfg['network']['bs_y'] = 300.0
    cfg['network']['comm_range'] = 50.0  # Larger comm range for sparser network

    cfg['energy']['initial_mean'] = 2.0  # Normal energy
    cfg['energy']['initial_std'] = 0.2
    cfg['simulation']['max_epochs'] = 500
    cfg['clustering']['join_radius'] = 50.0
    cfg['clustering']['spacing_radius'] = 50.0

    # Apply ablation based on variant
    if variant == 'full':
        cfg['auction']['lambda_'] = 2.0
        cfg['auction']['use_bandit'] = True
    elif variant == 'no-debt':
        cfg['auction']['lambda_'] = 0.0
        cfg['auction']['use_bandit'] = True
    elif variant == 'no-bandit':
        cfg['auction']['lambda_'] = 2.0
        cfg['auction']['use_bandit'] = False
    elif variant == 'minimal':
        cfg['auction']['lambda_'] = 0.0
        cfg['auction']['use_bandit'] = False

    net_cfg = cfg['network']
    energy_cfg = cfg['energy']

    np.random.seed(seed)
    nodes = create_heterogeneous_nodes(
        n=net_cfg['num_nodes'],
        width=net_cfg['area_width'],
        height=net_cfg['area_height'],
        energy_mean=energy_cfg['initial_mean'],
        energy_std=energy_cfg['initial_std'],
        seed=seed
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

    algo = AuctionClustering(network, energy_model, cfg)
    algo.setup()

    max_epochs = cfg['simulation']['max_epochs']

    fnd = hnd = lnd = None
    initial_nodes = len([n for n in network.nodes if n.is_alive])
    half_threshold = initial_nodes // 2

    energy_gini_history = []
    m_history = []  # Track aggressiveness over time

    for epoch in range(1, max_epochs + 1):
        stats = algo.run_epoch()

        alive = stats['alive_nodes']
        dead = initial_nodes - alive

        # Track metrics
        alive_nodes_list = [n for n in network.nodes if n.is_alive]
        energies = [n.current_energy for n in alive_nodes_list]
        m_values = [n.aggressiveness for n in alive_nodes_list]

        if energies:
            energy_gini_history.append(gini_coefficient(energies))
        if m_values:
            m_history.append(np.mean(m_values))

        if fnd is None and dead >= 1:
            fnd = epoch
        if hnd is None and dead >= half_threshold:
            hnd = epoch
        if alive == 0:
            lnd = epoch
            break

        # Progress update every 100 epochs
        if epoch % 100 == 0:
            print(f"    Epoch {epoch}: {alive} alive, avg_m={np.mean(m_values):.2f}")

    if lnd is None:
        lnd = max_epochs
    if hnd is None:
        hnd = max_epochs
    if fnd is None:
        fnd = max_epochs

    # Calculate mid-phase averages
    mid_start = len(energy_gini_history) // 4
    mid_end = 3 * len(energy_gini_history) // 4
    avg_gini = np.mean(energy_gini_history[mid_start:mid_end]) if energy_gini_history else 0

    # Average m during mid-phase (when bandit should have learned)
    avg_m_mid = np.mean(m_history[mid_start:mid_end]) if m_history else 1.0
    final_m = m_history[-1] if m_history else 1.0

    return {
        'variant': variant,
        'trial': seed,
        'fnd': fnd,
        'hnd': hnd,
        'lnd': lnd,
        'avg_gini': avg_gini,
        'avg_m_mid': avg_m_mid,
        'final_m': final_m,
    }


def main():
    config = load_config("config/default.yaml")

    variants = ['full', 'no-debt', 'no-bandit', 'minimal']
    variant_names = {
        'full': 'ABC-full (debt+bandit)',
        'no-debt': 'ABC-no-debt (bandit only)',
        'no-bandit': 'ABC-no-bandit (debt only)',
        'minimal': 'ABC-minimal (auction only)',
    }

    num_trials = 10  # Fewer trials due to larger network
    base_seed = 42

    print("=" * 80)
    print("FULL ABC ABLATION STUDY - 1000 NODES")
    print("=" * 80)
    print("Configuration:")
    print("  - Nodes: 1000")
    print("  - Area: 300m x 300m")
    print("  - Energy: 2.0J (normal)")
    print("  - Max Epochs: 500")
    print("  - Comm Range: 50m")
    print(f"  - Trials: {num_trials}")
    print("=" * 80)

    all_results = {v: [] for v in variants}

    for variant in variants:
        print(f"\n{'='*70}")
        print(f"Running {variant_names[variant]}")
        print('='*70)

        for trial in range(num_trials):
            seed = base_seed + trial
            print(f"\n  Trial {trial+1}/{num_trials} (seed={seed})...")

            result = run_single_trial(config, seed, variant)
            all_results[variant].append(result)

            print(f"  -> FND={result['fnd']}, HND={result['hnd']}, LND={result['lnd']}, "
                  f"avg_m={result['avg_m_mid']:.2f}, final_m={result['final_m']:.2f}")

    # Results table
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY (1000 NODES)")
    print('='*80)

    print(f"\n{'Variant':<30} {'FND':<15} {'HND':<15} {'LND':<15} {'Gini':<10} {'Avg m':<8}")
    print("-" * 93)

    summary = {}
    for variant in variants:
        results = all_results[variant]
        fnds = [r['fnd'] for r in results]
        hnds = [r['hnd'] for r in results]
        lnds = [r['lnd'] for r in results]
        ginis = [r['avg_gini'] for r in results]
        ms = [r['avg_m_mid'] for r in results]

        print(f"{variant_names[variant]:<30} "
              f"{np.mean(fnds):.1f} ({np.std(fnds):.1f})   "
              f"{np.mean(hnds):.1f} ({np.std(hnds):.1f})   "
              f"{np.mean(lnds):.1f} ({np.std(lnds):.1f})   "
              f"{np.mean(ginis):.3f}     "
              f"{np.mean(ms):.2f}")

        summary[variant] = {
            'fnd': np.mean(fnds), 'fnd_std': np.std(fnds),
            'hnd': np.mean(hnds), 'hnd_std': np.std(hnds),
            'lnd': np.mean(lnds), 'lnd_std': np.std(lnds),
            'gini': np.mean(ginis), 'm': np.mean(ms)
        }

    # Component analysis
    print(f"\n{'='*80}")
    print("COMPONENT CONTRIBUTION ANALYSIS")
    print('='*80)

    minimal = summary['minimal']
    full = summary['full']

    print(f"\n1. FAIRNESS DEBT contribution (no-bandit vs minimal):")
    debt_only = summary['no-bandit']
    fnd_pct = (debt_only['fnd']/minimal['fnd']-1)*100 if minimal['fnd'] > 0 else 0
    lnd_pct = (debt_only['lnd']/minimal['lnd']-1)*100 if minimal['lnd'] > 0 else 0
    print(f"   FND: {debt_only['fnd'] - minimal['fnd']:+.1f} epochs ({fnd_pct:+.1f}%)")
    print(f"   LND: {debt_only['lnd'] - minimal['lnd']:+.1f} epochs ({lnd_pct:+.1f}%)")
    print(f"   Gini: {debt_only['gini'] - minimal['gini']:+.4f}")

    print(f"\n2. BANDIT LEARNING contribution (no-debt vs minimal):")
    bandit_only = summary['no-debt']
    fnd_pct = (bandit_only['fnd']/minimal['fnd']-1)*100 if minimal['fnd'] > 0 else 0
    lnd_pct = (bandit_only['lnd']/minimal['lnd']-1)*100 if minimal['lnd'] > 0 else 0
    print(f"   FND: {bandit_only['fnd'] - minimal['fnd']:+.1f} epochs ({fnd_pct:+.1f}%)")
    print(f"   LND: {bandit_only['lnd'] - minimal['lnd']:+.1f} epochs ({lnd_pct:+.1f}%)")
    print(f"   Avg m: {bandit_only['m']:.2f} (learned aggressiveness)")

    print(f"\n3. COMBINED effect (full vs minimal):")
    fnd_pct = (full['fnd']/minimal['fnd']-1)*100 if minimal['fnd'] > 0 else 0
    lnd_pct = (full['lnd']/minimal['lnd']-1)*100 if minimal['lnd'] > 0 else 0
    print(f"   FND: {full['fnd'] - minimal['fnd']:+.1f} epochs ({fnd_pct:+.1f}%)")
    print(f"   LND: {full['lnd'] - minimal['lnd']:+.1f} epochs ({lnd_pct:+.1f}%)")
    print(f"   Gini: {full['gini'] - minimal['gini']:+.4f}")
    print(f"   Avg m: {full['m']:.2f}")

    # Save results
    output_path = Path("results/data/ablation_1000nodes")
    output_path.mkdir(parents=True, exist_ok=True)

    import csv
    with open(output_path / "ablation_1000nodes_summary.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['variant', 'trial', 'fnd', 'hnd', 'lnd', 'avg_gini', 'avg_m_mid', 'final_m'])
        for variant, results in all_results.items():
            for r in results:
                writer.writerow([variant, r['trial'], r['fnd'], r['hnd'], r['lnd'],
                               r['avg_gini'], r['avg_m_mid'], r['final_m']])

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print('='*80)


if __name__ == "__main__":
    main()
