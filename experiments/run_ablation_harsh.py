#!/usr/bin/env python3
"""
Ablation Study for ABC Algorithm - HARSH CONDITIONS

Tests with lower energy to force network death and show clear differentiation:
1. ABC-full: Complete algorithm (lambda=2.0, m=1.0)
2. ABC-no-debt: Fairness OFF (lambda=0)

Harsh conditions:
- Initial energy: 0.5J (instead of 2.0J)
- Max epochs: 1000 (run until death)
- Track energy variance and Gini coefficient

Usage:
    python experiments/run_ablation_harsh.py
"""

import sys
import copy
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation import load_config
from src.algorithms.auction import AuctionClustering
from src.models.node import create_heterogeneous_nodes
from src.models.network import Network
from src.models.energy import EnergyModel
import numpy as np


def gini_coefficient(values):
    """Calculate Gini coefficient (0 = perfect equality, 1 = perfect inequality)."""
    values = np.array(values)
    values = values[values > 0]  # Only alive nodes
    if len(values) == 0:
        return 0.0
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    cumsum = np.cumsum(sorted_vals)
    return (2 * np.sum((np.arange(1, n+1) * sorted_vals)) - (n + 1) * cumsum[-1]) / (n * cumsum[-1])


def run_single_trial_harsh(config: dict, seed: int, variant: str) -> dict:
    """
    Run a single trial with harsh conditions.
    """
    cfg = copy.deepcopy(config)

    # HARSH CONDITIONS
    cfg['energy']['initial_mean'] = 0.5  # Much lower energy
    cfg['energy']['initial_std'] = 0.1
    cfg['simulation']['max_epochs'] = 1000  # Run longer

    # Apply ablation
    if variant == 'no-debt':
        cfg['auction']['lambda_'] = 0.0

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

    # Track energy balance over time
    energy_gini_history = []
    energy_std_history = []

    for epoch in range(1, max_epochs + 1):
        stats = algo.run_epoch()

        alive = stats['alive_nodes']
        dead = initial_nodes - alive

        # Track energy balance
        energies = [n.current_energy for n in network.nodes if n.is_alive]
        if energies:
            energy_std_history.append(np.std(energies))
            energy_gini_history.append(gini_coefficient(energies))

        if fnd is None and dead >= 1:
            fnd = epoch
        if hnd is None and dead >= half_threshold:
            hnd = epoch
        if alive == 0:
            lnd = epoch
            break

    if lnd is None:
        lnd = max_epochs
    if hnd is None:
        hnd = max_epochs
    if fnd is None:
        fnd = max_epochs

    # Calculate average Gini in middle phase (when fairness matters most)
    mid_start = len(energy_gini_history) // 4
    mid_end = 3 * len(energy_gini_history) // 4
    avg_gini_mid = np.mean(energy_gini_history[mid_start:mid_end]) if energy_gini_history else 0
    avg_std_mid = np.mean(energy_std_history[mid_start:mid_end]) if energy_std_history else 0

    return {
        'variant': variant,
        'trial': seed,
        'fnd': fnd,
        'hnd': hnd,
        'lnd': lnd,
        'avg_gini': avg_gini_mid,
        'avg_energy_std': avg_std_mid,
        'final_gini': energy_gini_history[-1] if energy_gini_history else 0,
    }


def main():
    config = load_config("config/default.yaml")

    variants = ['full', 'no-debt']
    num_trials = 15
    base_seed = 42

    print("=" * 70)
    print("ABC ABLATION STUDY - HARSH CONDITIONS")
    print("=" * 70)
    print("Conditions: Initial Energy = 0.5J, Max Epochs = 1000")
    print("Variants: ABC-full (fairness ON) vs ABC-no-debt (fairness OFF)")
    print(f"Trials: {num_trials}")
    print("=" * 70)

    all_results = {v: [] for v in variants}

    for variant in variants:
        label = "ABC-full (fairness ON)" if variant == 'full' else "ABC-no-debt (fairness OFF)"
        print(f"\n{'='*50}")
        print(f"Running {label}")
        print('='*50)

        for trial in range(num_trials):
            seed = base_seed + trial
            print(f"  Trial {trial+1}/{num_trials} (seed={seed})...", end=" ", flush=True)

            result = run_single_trial_harsh(config, seed, variant)
            all_results[variant].append(result)

            print(f"FND={result['fnd']}, HND={result['hnd']}, LND={result['lnd']}, Gini={result['avg_gini']:.3f}")

    # Summary
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print('='*70)

    print(f"\n{'Variant':<30} {'FND':<15} {'HND':<15} {'LND':<15} {'Avg Gini':<12}")
    print("-" * 87)

    summary = {}
    for variant in variants:
        results = all_results[variant]
        fnds = [r['fnd'] for r in results]
        hnds = [r['hnd'] for r in results]
        lnds = [r['lnd'] for r in results]
        ginis = [r['avg_gini'] for r in results]

        label = "ABC-full (fairness ON)" if variant == 'full' else "ABC-no-debt (fairness OFF)"

        fnd_str = f"{np.mean(fnds):.1f} ({np.std(fnds):.1f})"
        hnd_str = f"{np.mean(hnds):.1f} ({np.std(hnds):.1f})"
        lnd_str = f"{np.mean(lnds):.1f} ({np.std(lnds):.1f})"
        gini_str = f"{np.mean(ginis):.4f}"

        print(f"{label:<30} {fnd_str:<15} {hnd_str:<15} {lnd_str:<15} {gini_str:<12}")

        summary[variant] = {
            'fnd_mean': np.mean(fnds), 'fnd_std': np.std(fnds),
            'hnd_mean': np.mean(hnds), 'hnd_std': np.std(hnds),
            'lnd_mean': np.mean(lnds), 'lnd_std': np.std(lnds),
            'gini_mean': np.mean(ginis),
        }

    # Statistical comparison
    print(f"\n{'='*70}")
    print("FAIRNESS IMPACT ANALYSIS")
    print('='*70)

    full = summary['full']
    no_debt = summary['no-debt']

    fnd_diff = full['fnd_mean'] - no_debt['fnd_mean']
    hnd_diff = full['hnd_mean'] - no_debt['hnd_mean']
    lnd_diff = full['lnd_mean'] - no_debt['lnd_mean']
    gini_diff = full['gini_mean'] - no_debt['gini_mean']

    print(f"\nFairness ON vs OFF:")
    print(f"  FND: {fnd_diff:+.1f} epochs ({fnd_diff/no_debt['fnd_mean']*100:+.1f}%)")
    print(f"  HND: {hnd_diff:+.1f} epochs ({hnd_diff/no_debt['hnd_mean']*100:+.1f}%)")
    print(f"  LND: {lnd_diff:+.1f} epochs ({lnd_diff/no_debt['lnd_mean']*100:+.1f}%)")
    print(f"  Gini: {gini_diff:+.4f} ({'better' if gini_diff < 0 else 'worse'} balance)")

    # Interpretation
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print('='*70)

    if fnd_diff > 0:
        print(f"+ Fairness DELAYS first node death by {fnd_diff:.1f} epochs")
    else:
        print(f"- Fairness ACCELERATES first node death by {-fnd_diff:.1f} epochs")

    if lnd_diff > 0:
        print(f"+ Fairness EXTENDS network lifetime by {lnd_diff:.1f} epochs")
    else:
        print(f"- Fairness REDUCES network lifetime by {-lnd_diff:.1f} epochs")

    if gini_diff < 0:
        print(f"+ Fairness IMPROVES energy balance (lower Gini)")
    else:
        print(f"- Fairness WORSENS energy balance (higher Gini)")

    # Save results
    output_path = Path("results/data/ablation_harsh")
    output_path.mkdir(parents=True, exist_ok=True)

    import csv
    with open(output_path / "ablation_harsh_summary.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['variant', 'trial', 'fnd', 'hnd', 'lnd', 'avg_gini', 'avg_energy_std'])
        for variant, results in all_results.items():
            for r in results:
                writer.writerow([variant, r['trial'], r['fnd'], r['hnd'], r['lnd'],
                               r['avg_gini'], r['avg_energy_std']])

    print(f"\nResults saved to: {output_path}")
    print('='*70)


if __name__ == "__main__":
    main()
