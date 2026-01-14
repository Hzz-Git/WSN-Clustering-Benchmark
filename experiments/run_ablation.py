#!/usr/bin/env python3
"""
Ablation Study for ABC Algorithm.

Tests the contribution of individual components:
1. ABC-full: Complete algorithm (lambda=2.0, m=1.0)
2. ABC-no-debt: Fairness OFF (lambda=0)
3. ABC-high-m: Higher aggressiveness (m=1.5)

Usage:
    python experiments/run_ablation.py
    python experiments/run_ablation.py --trials 15 --verbose
"""

import sys
import argparse
import copy
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation import Simulation, load_config
from src.algorithms.auction import AuctionClustering
from src.models.node import create_heterogeneous_nodes
from src.models.network import Network
from src.models.energy import EnergyModel
from src.metrics.collectors import MetricsCollector, save_results
import numpy as np


def run_single_trial(config: dict, seed: int, variant: str) -> dict:
    """
    Run a single trial with specific ablation variant.

    Args:
        config: Base config dict
        seed: Random seed
        variant: One of 'full', 'no-debt', 'high-m'

    Returns:
        Trial results dict
    """
    # Deep copy config to avoid mutation
    cfg = copy.deepcopy(config)

    # Apply ablation modifications
    if variant == 'no-debt':
        cfg['auction']['lambda_'] = 0.0  # Disable fairness debt
    elif variant == 'high-m':
        cfg['auction']['m_default'] = 1.5  # Higher aggressiveness
    # 'full' uses default config

    # Create network
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

    # Create algorithm
    algo = AuctionClustering(network, energy_model, cfg)
    algo.setup()

    # Run simulation
    max_epochs = cfg['simulation']['max_epochs']

    epoch_data = []
    fnd = hnd = lnd = None
    initial_nodes = len([n for n in network.nodes if n.is_alive])
    half_threshold = initial_nodes // 2

    for epoch in range(1, max_epochs + 1):
        stats = algo.run_epoch()
        epoch_data.append(stats)

        alive = stats['alive_nodes']
        dead = initial_nodes - alive

        # Track lifetime metrics
        if fnd is None and dead >= 1:
            fnd = epoch
        if hnd is None and dead >= half_threshold:
            hnd = epoch
        if alive == 0:
            lnd = epoch
            break

    # If network survived entire simulation
    if lnd is None:
        lnd = max_epochs
    if hnd is None:
        hnd = max_epochs
    if fnd is None:
        fnd = max_epochs

    return {
        'algorithm': f'ABC-{variant}',
        'variant': variant,
        'trial': seed,
        'fnd': fnd,
        'hnd': hnd,
        'lnd': lnd,
        'final_alive': epoch_data[-1]['alive_nodes'] if epoch_data else 0,
        'total_epochs': len(epoch_data),
        'epoch_data': epoch_data
    }


def run_ablation_experiment(
    config_path: str = "config/default.yaml",
    num_trials: int = 15,
    base_seed: int = 42,
    output_dir: str = "results/data/ablation",
    verbose: bool = True
):
    """
    Run ablation experiment comparing ABC variants.
    """
    # Load config
    config = load_config(config_path)

    variants = ['full', 'no-debt', 'high-m']
    variant_names = {
        'full': 'ABC-full (lambda=2.0, m=1.0)',
        'no-debt': 'ABC-no-debt (lambda=0)',
        'high-m': 'ABC-high-m (m=1.5)'
    }

    print("=" * 60)
    print("ABC Ablation Study")
    print("=" * 60)
    print(f"Variants: {list(variant_names.values())}")
    print(f"Trials per variant: {num_trials}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Results storage
    all_results = {v: [] for v in variants}

    for variant in variants:
        print(f"\n{'='*50}")
        print(f"Running {variant_names[variant]}")
        print('='*50)

        for trial in range(num_trials):
            seed = base_seed + trial

            if verbose:
                print(f"  Trial {trial+1}/{num_trials} (seed={seed})...", end=" ")

            result = run_single_trial(config, seed, variant)
            all_results[variant].append(result)

            if verbose:
                print(f"FND={result['fnd']}, HND={result['hnd']}, LND={result['lnd']}")

        # Print variant summary
        fnds = [r['fnd'] for r in all_results[variant]]
        hnds = [r['hnd'] for r in all_results[variant]]
        lnds = [r['lnd'] for r in all_results[variant]]

        print(f"\n{variant_names[variant]} Summary:")
        print(f"  FND: {np.mean(fnds):.1f} +/- {np.std(fnds):.1f}")
        print(f"  HND: {np.mean(hnds):.1f} +/- {np.std(hnds):.1f}")
        print(f"  LND: {np.mean(lnds):.1f} +/- {np.std(lnds):.1f}")

    # Save results
    print(f"\n{'='*50}")
    print("Saving results...")
    print('='*50)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save CSV summary
    import csv
    summary_file = output_path / "ablation_summary.csv"
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['variant', 'trial', 'fnd', 'hnd', 'lnd', 'final_alive'])
        for variant, results in all_results.items():
            for r in results:
                writer.writerow([variant, r['trial'], r['fnd'], r['hnd'], r['lnd'], r['final_alive']])

    print(f"Saved: {summary_file}")

    # Print final comparison table
    print(f"\n{'='*60}")
    print("ABLATION RESULTS")
    print('='*60)
    print(f"\n{'Variant':<25} {'FND':<18} {'HND':<18} {'LND':<18}")
    print("-" * 79)

    baseline_fnd = None
    for variant in variants:
        fnds = [r['fnd'] for r in all_results[variant]]
        hnds = [r['hnd'] for r in all_results[variant]]
        lnds = [r['lnd'] for r in all_results[variant]]

        fnd_str = f"{np.mean(fnds):.1f} ({np.std(fnds):.1f})"
        hnd_str = f"{np.mean(hnds):.1f} ({np.std(hnds):.1f})"
        lnd_str = f"{np.mean(lnds):.1f} ({np.std(lnds):.1f})"

        print(f"{variant_names[variant]:<25} {fnd_str:<18} {hnd_str:<18} {lnd_str:<18}")

        if variant == 'full':
            baseline_fnd = np.mean(fnds)
            baseline_hnd = np.mean(hnds)
            baseline_lnd = np.mean(lnds)

    # Impact analysis
    print(f"\n{'='*60}")
    print("COMPONENT IMPACT (vs ABC-full)")
    print('='*60)

    for variant in ['no-debt', 'high-m']:
        fnds = [r['fnd'] for r in all_results[variant]]
        hnds = [r['hnd'] for r in all_results[variant]]
        lnds = [r['lnd'] for r in all_results[variant]]

        fnd_diff = np.mean(fnds) - baseline_fnd
        hnd_diff = np.mean(hnds) - baseline_hnd
        lnd_diff = np.mean(lnds) - baseline_lnd

        print(f"\n{variant_names[variant]}:")
        print(f"  FND: {fnd_diff:+.1f} epochs ({fnd_diff/baseline_fnd*100:+.1f}%)")
        print(f"  HND: {hnd_diff:+.1f} epochs ({hnd_diff/baseline_hnd*100:+.1f}%)")
        print(f"  LND: {lnd_diff:+.1f} epochs ({lnd_diff/baseline_lnd*100:+.1f}%)")

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_dir}")
    print('='*60)

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="ABC Ablation Study"
    )
    parser.add_argument(
        '--config', '-c',
        default='config/default.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--trials', '-t',
        type=int,
        default=15,
        help='Number of trials per variant'
    )
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='Base random seed'
    )
    parser.add_argument(
        '--output', '-o',
        default='results/data/ablation',
        help='Output directory'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        default=True,
        help='Verbose output'
    )

    args = parser.parse_args()

    run_ablation_experiment(
        config_path=args.config,
        num_trials=args.trials,
        base_seed=args.seed,
        output_dir=args.output,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
