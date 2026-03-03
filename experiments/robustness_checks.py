#!/usr/bin/env python3
"""
Task 5: Minimal robustness checks for VTC submission.

Tests:
1. BS positions: Edge (default) vs Center
2. Density scaling: N in {50, 100, 200} with constant density

Evaluates at 2-3 representative regime points to verify phase transition persists.

Usage:
    python robustness_checks.py --seeds 10 --outdir results/data/robustness/
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
    """Calculate Gini coefficient."""
    values = np.array(values)
    if len(values) == 0 or np.sum(values) == 0:
        return 0.0
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    return (2 * np.sum((np.arange(1, n + 1) * sorted_vals)) - (n + 1) * np.sum(sorted_vals)) / (n * np.sum(sorted_vals))


def run_config(config, algo_name, num_seeds=10, seed_base=42):
    """Run simulation for a single configuration."""
    auc_list = []
    ctrl_frac_list = []

    for trial in range(num_seeds):
        seed = seed_base + trial
        sim = Simulation(config, seed=seed)
        result = sim.run(algo_name, seed=seed, verbose=False)

        auc_list.append(result['auc'])

        total_ctrl_energy = sum(h.get('control_energy_j', 0) for h in result['history'])
        total_energy_spent = result['initial_energy'] - result['final_energy']
        ctrl_frac = total_ctrl_energy / (total_energy_spent + 1e-9)
        ctrl_frac_list.append(ctrl_frac)

    return {
        'AUC_mean': np.mean(auc_list),
        'AUC_std': np.std(auc_list),
        'ctrl_frac_mean': np.mean(ctrl_frac_list),
    }


def main():
    parser = argparse.ArgumentParser(description='Robustness checks')
    parser.add_argument('--seeds', type=int, default=10,
                       help='Number of seeds per configuration')
    parser.add_argument('--max_epochs', type=int, default=300,
                       help='Maximum epochs')
    parser.add_argument('--outdir', type=str, default='results/data/robustness/',
                       help='Output directory')

    args = parser.parse_args()

    # Base configuration
    base_config = load_config("config/default.yaml")
    base_config['simulation']['max_epochs'] = args.max_epochs

    # Algorithms
    algorithms = [
        ('auction', 'ABC', 'local'),
        ('heed', 'HEED', 'local'),
        ('leach', 'LEACH-L', 'local'),
        ('leach', 'LEACH', 'global'),
    ]

    # Representative regime points (demonstrating phase transition)
    regime_points = [
        {'data_bits': 8000, 'ctrl_scale': 10.0, 'name': 'ctrl-dominated'},
        {'data_bits': 32000, 'ctrl_scale': 0.1, 'name': 'data-dominated'},
        {'data_bits': 32000, 'ctrl_scale': 10.0, 'name': 'mixed'},
    ]

    # BS positions
    bs_positions = [
        {'name': 'edge', 'bs_x': 50.0, 'bs_y': 100.0},
        {'name': 'center', 'bs_x': 50.0, 'bs_y': 50.0},
    ]

    # Density scaling (constant density)
    # Base: N0=50, A0=100x100 = 10000
    base_N = 50
    base_A = 100.0 * 100.0
    node_counts = [50, 100, 200]

    results = []

    print("="*80)
    print("ROBUSTNESS CHECKS")
    print("="*80)
    print(f"Seeds: {args.seeds}, Max epochs: {args.max_epochs}")
    print(f"Regimes: {[r['name'] for r in regime_points]}")
    print(f"BS positions: {[b['name'] for b in bs_positions]}")
    print(f"Node counts: {node_counts}")
    print("="*80)

    total_configs = len(regime_points) * len(bs_positions) * len(node_counts) * len(algorithms)
    run_count = 0

    for regime in regime_points:
        for bs_pos in bs_positions:
            for N in node_counts:
                # Calculate scaled area (constant density)
                A = base_A * (N / base_N)
                W = H = np.sqrt(A)

                # Scale BS position proportionally
                if bs_pos['name'] == 'edge':
                    bs_x, bs_y = W / 2, H
                else:
                    bs_x, bs_y = W / 2, H / 2

                for algo_key, algo_label, discovery_mode in algorithms:
                    run_count += 1
                    print(f"\n[{run_count}/{total_configs}] {regime['name']} | "
                          f"BS={bs_pos['name']} | N={N} | {algo_label}")

                    # Configure
                    config = copy.deepcopy(base_config)
                    config['packets']['data_size'] = regime['data_bits']
                    config['control']['bits_multiplier'] = regime['ctrl_scale']
                    config['control']['discovery_radius_mode'] = discovery_mode
                    config['network']['num_nodes'] = N
                    config['network']['area_width'] = W
                    config['network']['area_height'] = H
                    config['network']['bs_x'] = bs_x
                    config['network']['bs_y'] = bs_y

                    # Scale comm_range and join_radius proportionally
                    scale_factor = np.sqrt(N / base_N)
                    config['network']['comm_range'] = 30.0 * scale_factor
                    config['clustering']['join_radius'] = 30.0 * scale_factor

                    # Run
                    metrics = run_config(config, algo_key, num_seeds=args.seeds)

                    row = {
                        'regime': regime['name'],
                        'data_bits': regime['data_bits'],
                        'ctrl_scale': regime['ctrl_scale'],
                        'bs_position': bs_pos['name'],
                        'N': N,
                        'area': f'{W:.0f}x{H:.0f}',
                        'algorithm': algo_label,
                        'AUC_mean': metrics['AUC_mean'],
                        'AUC_std': metrics['AUC_std'],
                        'ctrl_frac': metrics['ctrl_frac_mean'],
                    }
                    results.append(row)

                    print(f"  AUC={metrics['AUC_mean']:.4f}±{metrics['AUC_std']:.4f}, "
                          f"ctrl_frac={metrics['ctrl_frac_mean']*100:.1f}%")

    # Save CSV
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = outdir / f"robustness_auc_{timestamp}.csv"

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n\nResults saved to: {csv_path}")

    # Summary analysis
    print("\n" + "="*80)
    print("ROBUSTNESS SUMMARY")
    print("="*80)

    # Check if phase transition persists across BS positions and N
    print("\n--- Phase Transition Persistence ---")
    for regime in regime_points:
        print(f"\n{regime['name'].upper()} regime:")
        for algo_label in ['ABC', 'HEED', 'LEACH-L', 'LEACH']:
            matching = [r for r in results
                       if r['regime'] == regime['name']
                       and r['algorithm'] == algo_label]
            if matching:
                aucs = [r['AUC_mean'] for r in matching]
                print(f"  {algo_label}: AUC range [{min(aucs):.3f}, {max(aucs):.3f}] "
                      f"across {len(matching)} configs")

    # Check relative rankings by regime
    print("\n--- Winner by Regime (averaged) ---")
    for regime in regime_points:
        matching = [r for r in results if r['regime'] == regime['name']]
        algo_aucs = {}
        for algo in ['ABC', 'HEED', 'LEACH-L', 'LEACH']:
            algo_matches = [r['AUC_mean'] for r in matching if r['algorithm'] == algo]
            if algo_matches:
                algo_aucs[algo] = np.mean(algo_matches)

        if algo_aucs:
            winner = max(algo_aucs, key=algo_aucs.get)
            print(f"  {regime['name']}: winner={winner} (AUC={algo_aucs[winner]:.4f})")


if __name__ == "__main__":
    main()
