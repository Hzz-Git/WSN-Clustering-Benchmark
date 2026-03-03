#!/usr/bin/env python3
"""
Check A: control.enabled=False should regress to baseline (control energy disabled).
Check B: control.bits_multiplier=10 should show significant impact on HEED/ABC.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import copy
import numpy as np
from src.simulation import load_config, Simulation


def run_check(config, check_name: str, num_trials: int = 5, max_epochs: int = 300):
    """Run a single check configuration."""
    config = copy.deepcopy(config)
    config['simulation']['max_epochs'] = max_epochs

    algorithms = ['auction', 'heed', 'leach']
    results = {}

    for algo_name in algorithms:
        fnd_list = []
        hnd_list = []
        lnd_list = []
        ctrl_energy_list = []

        for trial in range(num_trials):
            seed = 42 + trial
            sim = Simulation(config, seed=seed)
            result = sim.run(algo_name, seed=seed, verbose=False)

            total_ctrl_energy = sum(h.get('control_energy_j', 0) for h in result['history'])

            fnd = result['fnd'] if result['fnd'] is not None else max_epochs
            hnd = result['hnd'] if result['hnd'] is not None else max_epochs
            lnd = result['lnd'] if result['lnd'] is not None else max_epochs

            fnd_list.append(fnd)
            hnd_list.append(hnd)
            lnd_list.append(lnd)
            ctrl_energy_list.append(total_ctrl_energy)

        results[algo_name] = {
            'FND_mean': np.mean(fnd_list), 'FND_std': np.std(fnd_list),
            'HND_mean': np.mean(hnd_list), 'HND_std': np.std(hnd_list),
            'LND_mean': np.mean(lnd_list), 'LND_std': np.std(lnd_list),
            'ctrl_energy_mJ': np.mean(ctrl_energy_list) * 1000,
        }

    print(f"\n{'='*70}")
    print(f"{check_name}")
    print('='*70)
    print("\n| Algorithm | FND | HND | LND | Ctrl Energy (mJ) |")
    print("|-----------|-----|-----|-----|------------------|")
    for algo, r in results.items():
        print(f"| {algo.upper():9} | {r['FND_mean']:.1f}±{r['FND_std']:.1f} | "
              f"{r['HND_mean']:.1f}±{r['HND_std']:.1f} | "
              f"{r['LND_mean']:.1f}±{r['LND_std']:.1f} | "
              f"{r['ctrl_energy_mJ']:.1f} |")

    return results


def main():
    base_config = load_config("config/default.yaml")

    # Baseline snapshot (from previous run)
    print("\n" + "="*70)
    print("BASELINE SNAPSHOT (control disabled, from before fixes)")
    print("="*70)
    print("| AUCTION | 132.8±9.2 | 181.6±12.8 | 300.0±0.0 | N/A |")
    print("| HEED    | 36.8±6.9  | 218.0±22.2 | 300.0±0.0 | N/A |")
    print("| LEACH   | 54.0±8.6  | 135.6±10.5 | 217.8±13.6| N/A |")

    # Check A: control.enabled = False (should regress to baseline)
    config_a = copy.deepcopy(base_config)
    config_a['control'] = config_a.get('control', {})
    config_a['control']['enabled'] = False
    results_a = run_check(config_a, "CHECK A: control.enabled=False (expect baseline regression)")

    # Check B: control.bits_multiplier = 10 (should show significant impact)
    config_b = copy.deepcopy(base_config)
    config_b['control'] = config_b.get('control', {})
    config_b['control']['enabled'] = True
    config_b['control']['bits_multiplier'] = 10.0
    results_b = run_check(config_b, "CHECK B: control.bits_multiplier=10 (expect significant impact)")

    # Current default (bits_multiplier = 1.0)
    config_default = copy.deepcopy(base_config)
    config_default['control'] = config_default.get('control', {})
    config_default['control']['enabled'] = True
    config_default['control']['bits_multiplier'] = 1.0
    results_default = run_check(config_default, "DEFAULT: control.enabled=True, bits_multiplier=1.0")

    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    print("\nCheck A Validation (control disabled should match baseline):")
    baseline = {'auction': {'FND': 132.8, 'HND': 181.6, 'LND': 300.0},
                'heed': {'FND': 36.8, 'HND': 218.0, 'LND': 300.0},
                'leach': {'FND': 54.0, 'HND': 135.6, 'LND': 217.8}}
    for algo in ['auction', 'heed', 'leach']:
        fnd_diff = abs(results_a[algo]['FND_mean'] - baseline[algo]['FND'])
        hnd_diff = abs(results_a[algo]['HND_mean'] - baseline[algo]['HND'])
        status = "PASS" if fnd_diff < 10 and hnd_diff < 20 else "FAIL"
        print(f"  {algo.upper()}: FND diff={fnd_diff:.1f}, HND diff={hnd_diff:.1f} -> {status}")

    print("\nCheck B Validation (10x control should reduce lifetime, esp. HEED):")
    for algo in ['auction', 'heed', 'leach']:
        fnd_drop = results_default[algo]['FND_mean'] - results_b[algo]['FND_mean']
        hnd_drop = results_default[algo]['HND_mean'] - results_b[algo]['HND_mean']
        ctrl_increase = results_b[algo]['ctrl_energy_mJ'] / (results_default[algo]['ctrl_energy_mJ'] + 0.001)
        print(f"  {algo.upper()}: FND drop={fnd_drop:.1f}, HND drop={hnd_drop:.1f}, ctrl_E ratio={ctrl_increase:.1f}x")

    print("\n" + "="*70)
    print("DONE")
    print("="*70)


if __name__ == "__main__":
    main()
