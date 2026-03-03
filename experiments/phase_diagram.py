#!/usr/bin/env python3
"""
Task D1: Phase Diagram - Control vs Data Plane Dominance

2D sweep across:
- data_bits: [500, 2000, 8000, 32000]
- control_bits_scale: [0.1, 1, 10]

Algorithms: ABC, HEED, LEACH-local, LEACH-global

Output: CSV with HND, LND, Gini, control_energy_fraction, data_packets
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

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


def run_single_config(config, algo_name, num_trials=5, seed_base=42):
    """Run simulation for a single configuration."""
    max_epochs = config['simulation']['max_epochs']

    auc_list = []
    hnd_list = []
    lnd_list = []
    gini_list = []
    ctrl_frac_list = []
    packets_list = []

    for trial in range(num_trials):
        seed = seed_base + trial
        sim = Simulation(config, seed=seed)
        result = sim.run(algo_name, seed=seed, verbose=False)

        # AUC (primary metric)
        auc_list.append(result['auc'])

        # Legacy metrics
        hnd = result['hnd'] if result['hnd'] is not None else max_epochs
        lnd = result['lnd'] if result['lnd'] is not None else max_epochs

        # Control energy fraction
        total_ctrl_energy = sum(h.get('control_energy_j', 0) for h in result['history'])
        total_energy_spent = result['initial_energy'] - result['final_energy']
        ctrl_frac = total_ctrl_energy / (total_energy_spent + 1e-9)

        # Gini of final energy distribution
        final_energies = [n.current_energy for n in sim.network.nodes if n.is_alive]
        gini = gini_coefficient(final_energies) if final_energies else 0.0

        # Total data packets
        total_packets = sum(h['data_packets'] for h in result['history'])

        hnd_list.append(hnd)
        lnd_list.append(lnd)
        gini_list.append(gini)
        ctrl_frac_list.append(ctrl_frac)
        packets_list.append(total_packets)

    return {
        'AUC_mean': np.mean(auc_list),
        'AUC_std': np.std(auc_list),
        'HND_mean': np.mean(hnd_list),
        'HND_std': np.std(hnd_list),
        'LND_mean': np.mean(lnd_list),
        'LND_std': np.std(lnd_list),
        'Gini_mean': np.mean(gini_list),
        'ctrl_frac_mean': np.mean(ctrl_frac_list),
        'packets_mean': np.mean(packets_list),
    }


def main():
    base_config = load_config("config/default.yaml")
    base_config['simulation']['max_epochs'] = 300

    # Sweep parameters
    data_bits_values = [500, 2000, 8000, 32000]
    control_scale_values = [0.1, 1.0, 10.0]

    # Algorithms to test
    # LEACH with both local and global discovery modes
    algorithms = [
        ('auction', 'ABC', 'local'),
        ('heed', 'HEED', 'local'),
        ('leach', 'LEACH-L', 'local'),
        ('leach', 'LEACH', 'global'),
    ]

    num_trials = 5

    # Output file
    output_dir = Path("results/data/phase_diagram")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"phase_diagram_{timestamp}.csv"

    print("="*80)
    print("PHASE DIAGRAM EXPERIMENT")
    print("="*80)
    print(f"Data bits: {data_bits_values}")
    print(f"Control scale: {control_scale_values}")
    print(f"Algorithms: {[a[1] for a in algorithms]}")
    print(f"Trials per config: {num_trials}")
    print(f"Output: {csv_path}")
    print("="*80)

    results = []
    total_runs = len(data_bits_values) * len(control_scale_values) * len(algorithms)
    run_count = 0

    for data_bits in data_bits_values:
        for ctrl_scale in control_scale_values:
            for algo_key, algo_label, discovery_mode in algorithms:
                run_count += 1
                print(f"\n[{run_count}/{total_runs}] {algo_label} | data={data_bits} | ctrl_scale={ctrl_scale}")

                # Configure
                config = copy.deepcopy(base_config)
                config['packets']['data_size'] = data_bits
                config['control']['bits_multiplier'] = ctrl_scale
                config['control']['discovery_radius_mode'] = discovery_mode

                # Run
                metrics = run_single_config(config, algo_key, num_trials=num_trials)

                # Record
                row = {
                    'algorithm': algo_label,
                    'data_bits': data_bits,
                    'ctrl_scale': ctrl_scale,
                    'discovery_mode': discovery_mode,
                    'AUC_mean': metrics['AUC_mean'],
                    'AUC_std': metrics['AUC_std'],
                    'HND_mean': metrics['HND_mean'],
                    'HND_std': metrics['HND_std'],
                    'LND_mean': metrics['LND_mean'],
                    'LND_std': metrics['LND_std'],
                    'Gini': metrics['Gini_mean'],
                    'ctrl_frac': metrics['ctrl_frac_mean'],
                    'packets': metrics['packets_mean'],
                }
                results.append(row)

                print(f"  AUC={metrics['AUC_mean']:.4f}±{metrics['AUC_std']:.4f}, "
                      f"HND={metrics['HND_mean']:.1f}, "
                      f"ctrl_frac={metrics['ctrl_frac_mean']*100:.1f}%")

    # Write CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n\nResults saved to: {csv_path}")

    # Print summary table
    print("\n" + "="*80)
    print("PHASE DIAGRAM SUMMARY")
    print("="*80)

    # Group by control scale for comparison (AUC as primary metric)
    for ctrl_scale in control_scale_values:
        print(f"\n--- Control Scale: {ctrl_scale}x ---")
        print(f"{'Algorithm':<10} | {'data=500':>12} | {'data=2000':>12} | {'data=8000':>12} | {'data=32000':>12}")
        print("-" * 70)

        for algo_label in ['ABC', 'HEED', 'LEACH-L', 'LEACH']:
            row_str = f"{algo_label:<10} |"
            for data_bits in data_bits_values:
                matching = [r for r in results
                           if r['algorithm'] == algo_label
                           and r['data_bits'] == data_bits
                           and r['ctrl_scale'] == ctrl_scale]
                if matching:
                    r = matching[0]
                    row_str += f" AUC={r['AUC_mean']:.3f} |"
                else:
                    row_str += f" {'N/A':>10} |"
            print(row_str)

    # Key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    # Find where HEED drops most (control-plane dominated region)
    heed_results = [r for r in results if r['algorithm'] == 'HEED']
    if heed_results:
        heed_by_ctrl = {}
        for r in heed_results:
            key = r['ctrl_scale']
            if key not in heed_by_ctrl:
                heed_by_ctrl[key] = []
            heed_by_ctrl[key].append(r['HND_mean'])

        print("\nHEED HND by control scale (averaged over data_bits):")
        for scale in sorted(heed_by_ctrl.keys()):
            avg_hnd = np.mean(heed_by_ctrl[scale])
            print(f"  ctrl_scale={scale}: avg HND = {avg_hnd:.1f}")

    # Find control_frac by region
    print("\nControl energy fraction by region:")
    for ctrl_scale in control_scale_values:
        for data_bits in [500, 32000]:  # Low vs high data
            for algo in ['ABC', 'HEED']:
                matching = [r for r in results
                           if r['algorithm'] == algo
                           and r['data_bits'] == data_bits
                           and r['ctrl_scale'] == ctrl_scale]
                if matching:
                    r = matching[0]
                    regime = "low-data" if data_bits == 500 else "high-data"
                    print(f"  {algo} @ {regime}, ctrl_scale={ctrl_scale}: ctrl_frac={r['ctrl_frac']*100:.1f}%")


if __name__ == "__main__":
    main()
