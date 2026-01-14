#!/usr/bin/env python3
"""
Main comparison experiment - 1000 NODES

Compares ABC (Auction), HEED, and LEACH algorithms with 1000 nodes.
"""

import sys
import copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation import Simulation, load_config, ALGORITHMS
from src.metrics.collectors import MetricsCollector, save_results
import numpy as np


def run_comparison_1000nodes(
    config_path: str = "config/default.yaml",
    algorithms: list = None,
    num_trials: int = 10,
    base_seed: int = 42,
    output_dir: str = "results/data/comparison_1000nodes",
):
    """Run comparison with 1000 nodes."""

    # Load and modify config for 1000 nodes
    config = load_config(config_path)

    # 1000 NODE CONFIGURATION
    config['network']['num_nodes'] = 1000
    config['network']['area_width'] = 300.0
    config['network']['area_height'] = 300.0
    config['network']['bs_x'] = 150.0
    config['network']['bs_y'] = 300.0
    config['network']['comm_range'] = 50.0

    config['energy']['initial_mean'] = 2.0
    config['energy']['initial_std'] = 0.2
    config['simulation']['max_epochs'] = 500

    config['clustering']['join_radius'] = 50.0
    config['clustering']['spacing_radius'] = 50.0

    # Algorithms to compare
    if algorithms is None:
        algorithms = ['auction', 'heed', 'leach']

    print("=" * 70)
    print("WSN CLUSTERING ALGORITHM COMPARISON - 1000 NODES")
    print("=" * 70)
    print("Configuration:")
    print(f"  - Nodes: 1000")
    print(f"  - Area: 300m x 300m")
    print(f"  - Energy: 2.0J")
    print(f"  - Max Epochs: 500")
    print(f"  - Comm Range: 50m")
    print(f"  - Algorithms: {algorithms}")
    print(f"  - Trials: {num_trials}")
    print("=" * 70)

    all_results = {}
    collector = MetricsCollector()

    for algo_name in algorithms:
        print(f"\n{'='*60}")
        print(f"Running {algo_name.upper()}")
        print('='*60)

        sim = Simulation(config)
        results = []

        for trial in range(num_trials):
            seed = base_seed + trial
            print(f"\n  Trial {trial+1}/{num_trials} (seed={seed})...", end=" ", flush=True)

            result = sim.run(algo_name, seed=seed, verbose=False)
            result['trial'] = trial
            results.append(result)

            print(f"FND={result['fnd']}, HND={result['hnd']}, LND={result['lnd']}")

        all_results[algo_name] = results
        collector.collect_batch(results)

        # Print summary
        stats = collector.get_summary_stats(algo_name)
        print(f"\n{algo_name.upper()} Summary:")
        print(f"  FND: {stats['fnd_mean']:.1f} +/- {stats['fnd_std']:.1f}")
        print(f"  HND: {stats['hnd_mean']:.1f} +/- {stats['hnd_std']:.1f}")
        print(f"  LND: {stats['lnd_mean']:.1f} +/- {stats['lnd_std']:.1f}")

    # Save results
    print(f"\n{'='*60}")
    print("Saving results...")
    print('='*60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for algo_name, results in all_results.items():
        save_results(results, output_dir, prefix=algo_name)

    # Final comparison table
    print(f"\n{'='*70}")
    print("FINAL COMPARISON (1000 NODES)")
    print('='*70)

    print(f"\n{'Algorithm':<15} {'FND':<20} {'HND':<20} {'LND':<20}")
    print("-" * 75)

    comparison_data = []
    for algo_name in algorithms:
        stats = collector.get_summary_stats(algo_name)
        fnd_str = f"{stats['fnd_mean']:.1f} ({stats['fnd_std']:.1f})"
        hnd_str = f"{stats['hnd_mean']:.1f} ({stats['hnd_std']:.1f})"
        lnd_str = f"{stats['lnd_mean']:.1f} ({stats['lnd_std']:.1f})"
        print(f"{algo_name.upper():<15} {fnd_str:<20} {hnd_str:<20} {lnd_str:<20}")

        comparison_data.append({
            'algorithm': algo_name,
            'fnd_mean': stats['fnd_mean'],
            'hnd_mean': stats['hnd_mean'],
            'lnd_mean': stats['lnd_mean'],
        })

    # Calculate improvements
    if 'auction' in algorithms:
        print(f"\n{'='*70}")
        print("ABC IMPROVEMENT OVER BASELINES")
        print('='*70)

        auction_stats = collector.get_summary_stats('auction')

        for baseline in algorithms:
            if baseline == 'auction':
                continue
            baseline_stats = collector.get_summary_stats(baseline)

            if baseline_stats['fnd_mean'] and auction_stats['fnd_mean']:
                fnd_imp = ((auction_stats['fnd_mean'] - baseline_stats['fnd_mean'])
                          / baseline_stats['fnd_mean'] * 100)
                hnd_imp = ((auction_stats['hnd_mean'] - baseline_stats['hnd_mean'])
                          / baseline_stats['hnd_mean'] * 100)
                lnd_imp = ((auction_stats['lnd_mean'] - baseline_stats['lnd_mean'])
                          / baseline_stats['lnd_mean'] * 100)

                print(f"\nvs {baseline.upper()}:")
                print(f"  FND: {fnd_imp:+.1f}%")
                print(f"  HND: {hnd_imp:+.1f}%")
                print(f"  LND: {lnd_imp:+.1f}%")

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_dir}")
    print('='*70)

    return all_results


if __name__ == "__main__":
    run_comparison_1000nodes()
