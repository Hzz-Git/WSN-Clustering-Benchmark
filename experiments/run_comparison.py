#!/usr/bin/env python3
"""
Main experiment script for comparing clustering algorithms.

Runs ABC (Auction), HEED, and LEACH with multiple trials
and generates comparison plots and statistics.

Usage:
    python experiments/run_comparison.py
    python experiments/run_comparison.py --trials 15 --verbose
    python experiments/run_comparison.py --algorithms auction heed
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation import Simulation, load_config, ALGORITHMS
from src.metrics.collectors import MetricsCollector, save_results
from src.utils.visualization import generate_all_plots, plot_alive_nodes, plot_lifetime_comparison
import numpy as np


def run_experiment(
    config_path: str = "config/default.yaml",
    algorithms: list = None,
    num_trials: int = 15,
    base_seed: int = 42,
    output_dir: str = "results/data/comparison",
    verbose: bool = True
):
    """
    Run comparison experiment.

    Args:
        config_path: Path to config file
        algorithms: List of algorithm names (default: all)
        num_trials: Number of trials per algorithm
        base_seed: Base random seed
        output_dir: Output directory for results
        verbose: Print progress
    """
    # Load config
    config = load_config(config_path)

    # Default algorithms
    if algorithms is None:
        algorithms = ['auction', 'heed', 'leach']

    # Validate algorithms
    for algo in algorithms:
        if algo.lower() not in ALGORITHMS:
            print(f"Error: Unknown algorithm '{algo}'")
            print(f"Available: {list(ALGORITHMS.keys())}")
            return

    print("=" * 60)
    print("WSN Clustering Algorithm Comparison")
    print("=" * 60)
    print(f"Algorithms: {algorithms}")
    print(f"Trials per algorithm: {num_trials}")
    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Run experiments
    all_results = {}
    collector = MetricsCollector()

    for algo_name in algorithms:
        print(f"\n{'='*50}")
        print(f"Running {algo_name.upper()}")
        print('='*50)

        sim = Simulation(config)
        results = sim.run_trials(
            algo_name,
            num_trials=num_trials,
            base_seed=base_seed,
            verbose=verbose
        )

        all_results[algo_name] = results
        collector.collect_batch(results)

        # Print summary
        stats = collector.get_summary_stats(algo_name)
        print(f"\n{algo_name.upper()} Summary:")
        print(f"  FND: {stats['fnd_mean']:.1f} +/- {stats['fnd_std']:.1f}")
        print(f"  HND: {stats['hnd_mean']:.1f} +/- {stats['hnd_std']:.1f}")
        print(f"  LND: {stats['lnd_mean']:.1f} +/- {stats['lnd_std']:.1f}")

    # Save results
    print(f"\n{'='*50}")
    print("Saving results...")
    print('='*50)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save per-algorithm results
    for algo_name, results in all_results.items():
        save_results(results, output_dir, prefix=algo_name)

    # Save combined results
    all_flat = []
    for results in all_results.values():
        all_flat.extend(results)
    save_results(all_flat, output_dir, prefix="combined")

    # Generate plots
    print(f"\n{'='*50}")
    print("Generating plots...")
    print('='*50)

    figures_dir = output_path.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    generate_all_plots(all_results, str(figures_dir), prefix="comparison")

    # Print final comparison
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print('='*60)

    comparison_data = []
    for algo_name in algorithms:
        stats = collector.get_summary_stats(algo_name)
        comparison_data.append({
            'Algorithm': algo_name.upper(),
            'FND': f"{stats['fnd_mean']:.1f} ({stats['fnd_std']:.1f})",
            'HND': f"{stats['hnd_mean']:.1f} ({stats['hnd_std']:.1f})",
            'LND': f"{stats['lnd_mean']:.1f} ({stats['lnd_std']:.1f})",
        })

    # Print as table
    print(f"\n{'Algorithm':<12} {'FND':<18} {'HND':<18} {'LND':<18}")
    print("-" * 66)
    for row in comparison_data:
        print(f"{row['Algorithm']:<12} {row['FND']:<18} {row['HND']:<18} {row['LND']:<18}")

    # Calculate improvements
    if 'auction' in algorithms and len(algorithms) > 1:
        auction_stats = collector.get_summary_stats('auction')

        print(f"\n{'='*50}")
        print("ABC Improvement over Baselines")
        print('='*50)

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

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"Figures saved to: {figures_dir}")
    print('='*60)


def main():
    parser = argparse.ArgumentParser(
        description="WSN Clustering Algorithm Comparison Experiment"
    )
    parser.add_argument(
        '--config', '-c',
        default='config/default.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--algorithms', '-a',
        nargs='+',
        default=['auction', 'heed', 'leach'],
        help='Algorithms to compare'
    )
    parser.add_argument(
        '--trials', '-t',
        type=int,
        default=15,
        help='Number of trials per algorithm'
    )
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='Base random seed'
    )
    parser.add_argument(
        '--output', '-o',
        default='results/data/comparison',
        help='Output directory'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    run_experiment(
        config_path=args.config,
        algorithms=args.algorithms,
        num_trials=args.trials,
        base_seed=args.seed,
        output_dir=args.output,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
