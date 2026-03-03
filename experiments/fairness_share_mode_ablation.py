#!/usr/bin/env python3
"""
Task 4: Fairness share_mode ablation experiment.

Compares alive_only vs all_nodes share calculation modes.
Outputs CSV for Fig 4.

Usage:
    python fairness_share_mode_ablation.py --seeds 20 --outdir results/data/fairness/
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
    cumsum = np.cumsum(sorted_vals)
    return (2 * np.sum((np.arange(1, n + 1) * sorted_vals)) - (n + 1) * np.sum(sorted_vals)) / (n * np.sum(sorted_vals))


def run_ablation(config, share_mode, num_seeds=20, seed_base=42):
    """Run ablation for a specific share_mode."""
    config = copy.deepcopy(config)
    config['fairness']['share_mode'] = share_mode

    auc_list = []
    gini_list = []
    ctrl_frac_list = []

    for trial in range(num_seeds):
        seed = seed_base + trial
        sim = Simulation(config, seed=seed)
        result = sim.run('auction', seed=seed, verbose=False)

        auc_list.append(result['auc'])

        # Gini of final energy distribution
        final_energies = [n.current_energy for n in sim.network.nodes if n.is_alive]
        gini = gini_coefficient(final_energies) if final_energies else 0.0
        gini_list.append(gini)

        # Control energy fraction
        total_ctrl_energy = sum(h.get('control_energy_j', 0) for h in result['history'])
        total_energy_spent = result['initial_energy'] - result['final_energy']
        ctrl_frac = total_ctrl_energy / (total_energy_spent + 1e-9)
        ctrl_frac_list.append(ctrl_frac)

    return {
        'share_mode': share_mode,
        'AUC_mean': np.mean(auc_list),
        'AUC_std': np.std(auc_list),
        'Gini_mean': np.mean(gini_list),
        'Gini_std': np.std(gini_list),
        'ctrl_frac_mean': np.mean(ctrl_frac_list),
    }


def main():
    parser = argparse.ArgumentParser(description='Fairness share_mode ablation')
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
    print("FAIRNESS SHARE_MODE ABLATION")
    print("="*70)
    print(f"Config: data_bits={args.data_bits}, ctrl_scale={args.ctrl_scale}")
    print(f"Seeds: {args.seeds}")
    print("="*70)

    results = []

    # Test both modes
    for share_mode in ['all_nodes', 'alive_only']:
        print(f"\nRunning share_mode={share_mode}...")
        metrics = run_ablation(config, share_mode, num_seeds=args.seeds)
        results.append(metrics)
        print(f"  AUC={metrics['AUC_mean']:.4f}±{metrics['AUC_std']:.4f}, "
              f"Gini={metrics['Gini_mean']:.4f}±{metrics['Gini_std']:.4f}")

    # Save CSV
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = outdir / f"fairness_ablation_{timestamp}.csv"

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {csv_path}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Mode':<12} | {'AUC':>15} | {'Gini':>15}")
    print("-"*50)
    for r in results:
        print(f"{r['share_mode']:<12} | {r['AUC_mean']:.4f}±{r['AUC_std']:.4f} | "
              f"{r['Gini_mean']:.4f}±{r['Gini_std']:.4f}")


if __name__ == "__main__":
    main()
