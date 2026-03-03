#!/usr/bin/env python3
"""
Fairness lambda ablation: test if fairness debt mechanism helps.

Compares lambda=0 (no fairness) vs lambda=2.0 (with fairness).

Usage:
    python fairness_lambda_ablation.py --seeds 20
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import copy
import csv
import numpy as np
from datetime import datetime
from src.simulation import load_config, Simulation


def gini_coefficient(values):
    """Calculate Gini coefficient for energy distribution."""
    values = np.array(values)
    if len(values) == 0 or np.sum(values) == 0:
        return 0.0
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    return (2 * np.sum((np.arange(1, n + 1) * sorted_vals)) - (n + 1) * np.sum(sorted_vals)) / (n * np.sum(sorted_vals))


def run_ablation(config, lambda_val, num_seeds=20, seed_base=42):
    """Run ablation for a specific lambda value."""
    config = copy.deepcopy(config)
    config['auction']['lambda_'] = lambda_val

    auc_list = []
    gini_list = []
    fnd_list = []

    for trial in range(num_seeds):
        seed = seed_base + trial
        sim = Simulation(config, seed=seed)
        result = sim.run('auction', seed=seed, verbose=False)

        auc_list.append(result['auc'])
        fnd = result.get('fnd')
        if fnd is not None:
            fnd_list.append(fnd)

        # Gini of final energy distribution
        final_energies = [n.current_energy for n in sim.network.nodes if n.is_alive]
        gini = gini_coefficient(final_energies) if final_energies else 0.0
        gini_list.append(gini)

    return {
        'lambda': lambda_val,
        'AUC_mean': np.mean(auc_list),
        'AUC_std': np.std(auc_list),
        'Gini_mean': np.mean(gini_list),
        'Gini_std': np.std(gini_list),
        'FND_mean': np.mean(fnd_list) if fnd_list else float('nan'),
        'FND_std': np.std(fnd_list) if fnd_list else float('nan'),
    }


def main():
    parser = argparse.ArgumentParser(description='Fairness lambda ablation')
    parser.add_argument('--seeds', type=int, default=20,
                       help='Number of seeds per configuration')
    parser.add_argument('--data_bits', type=int, default=32000,
                       help='Data packet size')
    parser.add_argument('--ctrl_scale', type=float, default=1.0,
                       help='Control bits multiplier')
    parser.add_argument('--max_epochs', type=int, default=300,
                       help='Maximum epochs')
    parser.add_argument('--outdir', type=str, default='results/data/fairness/',
                       help='Output directory')

    args = parser.parse_args()

    # Load base config
    config = load_config("config/default.yaml")
    config['simulation']['max_epochs'] = args.max_epochs
    config['packets']['data_size'] = args.data_bits
    config['control']['bits_multiplier'] = args.ctrl_scale

    print("="*70)
    print("FAIRNESS LAMBDA ABLATION: Does fairness debt help?")
    print("="*70)
    print(f"Config: data_bits={args.data_bits}, ctrl_scale={args.ctrl_scale}")
    print(f"Seeds: {args.seeds}")
    print("="*70)

    results = []

    # Test lambda=0 (no fairness) vs lambda=2.0 (with fairness)
    for lambda_val in [0.0, 2.0]:
        label = "no fairness" if lambda_val == 0 else "with fairness"
        print(f"\nRunning lambda={lambda_val} ({label})...")
        metrics = run_ablation(config, lambda_val, num_seeds=args.seeds)
        results.append(metrics)
        print(f"  AUC={metrics['AUC_mean']:.4f}±{metrics['AUC_std']:.4f}, "
              f"Gini={metrics['Gini_mean']:.4f}±{metrics['Gini_std']:.4f}, "
              f"FND={metrics['FND_mean']:.1f}±{metrics['FND_std']:.1f}")

    # Save CSV
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = outdir / f"fairness_lambda_ablation_{timestamp}.csv"

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {csv_path}")

    # Summary and statistical comparison
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Lambda':<10} | {'AUC':>18} | {'Gini':>18} | {'FND':>12}")
    print("-"*65)
    for r in results:
        print(f"{r['lambda']:<10} | {r['AUC_mean']:.4f}±{r['AUC_std']:.4f} | "
              f"{r['Gini_mean']:.4f}±{r['Gini_std']:.4f} | "
              f"{r['FND_mean']:.1f}±{r['FND_std']:.1f}")

    # Effect size
    no_fair = results[0]
    with_fair = results[1]
    auc_diff = with_fair['AUC_mean'] - no_fair['AUC_mean']
    gini_diff = with_fair['Gini_mean'] - no_fair['Gini_mean']

    print("\n" + "="*70)
    print("EFFECT OF FAIRNESS (lambda=2.0 minus lambda=0)")
    print("="*70)
    print(f"  AUC change:  {auc_diff:+.4f} ({'better' if auc_diff > 0 else 'worse' if auc_diff < 0 else 'same'})")
    print(f"  Gini change: {gini_diff:+.4f} ({'more equal' if gini_diff < 0 else 'less equal' if gini_diff > 0 else 'same'})")

    # Rough significance check
    pooled_std_auc = np.sqrt((no_fair['AUC_std']**2 + with_fair['AUC_std']**2) / 2)
    effect_size_auc = abs(auc_diff) / (pooled_std_auc + 1e-9)

    print(f"\n  Effect size (Cohen's d for AUC): {effect_size_auc:.2f}")
    if effect_size_auc < 0.2:
        print("  Interpretation: Negligible effect")
    elif effect_size_auc < 0.5:
        print("  Interpretation: Small effect")
    elif effect_size_auc < 0.8:
        print("  Interpretation: Medium effect")
    else:
        print("  Interpretation: Large effect")


if __name__ == "__main__":
    main()
