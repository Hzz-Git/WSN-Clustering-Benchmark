#!/usr/bin/env python3
"""
Post-fix comparison against baseline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.simulation import load_config, Simulation

def run_comparison():
    config = load_config("config/default.yaml")

    num_trials = 5
    seed_base = 42
    max_epochs = 300
    config['simulation']['max_epochs'] = max_epochs

    algorithms = ['auction', 'heed', 'leach']
    results = {}

    for algo_name in algorithms:
        print(f"\n{'='*60}")
        print(f"Running {algo_name.upper()} ({num_trials} trials)...")
        print('='*60)

        fnd_list = []
        hnd_list = []
        lnd_list = []
        ctrl_energy_list = []
        ctrl_msgs_list = []
        throughput_list = []

        for trial in range(num_trials):
            seed = seed_base + trial
            sim = Simulation(config, seed=seed)
            result = sim.run(algo_name, seed=seed, verbose=False)

            total_ctrl_energy = sum(h.get('control_energy_j', 0) for h in result['history'])
            total_ctrl_msgs = sum(h['control_messages'] for h in result['history'])
            total_packets = sum(h['data_packets'] for h in result['history'])

            fnd = result['fnd'] if result['fnd'] is not None else max_epochs
            hnd = result['hnd'] if result['hnd'] is not None else max_epochs
            lnd = result['lnd'] if result['lnd'] is not None else max_epochs

            fnd_list.append(fnd)
            hnd_list.append(hnd)
            lnd_list.append(lnd)
            ctrl_energy_list.append(total_ctrl_energy)
            ctrl_msgs_list.append(total_ctrl_msgs)
            throughput_list.append(total_packets)

            print(f"  Trial {trial+1}: FND={fnd}, HND={hnd}, LND={lnd}, ctrl_E={total_ctrl_energy*1000:.1f}mJ")

        results[algo_name] = {
            'FND_mean': np.mean(fnd_list), 'FND_std': np.std(fnd_list),
            'HND_mean': np.mean(hnd_list), 'HND_std': np.std(hnd_list),
            'LND_mean': np.mean(lnd_list), 'LND_std': np.std(lnd_list),
            'ctrl_energy_mJ': np.mean(ctrl_energy_list) * 1000,
            'ctrl_msgs': np.mean(ctrl_msgs_list),
            'throughput': np.mean(throughput_list),
        }

    # Print comparison
    print("\n\n" + "="*80)
    print("POST-FIX RESULTS SUMMARY")
    print("="*80)
    print("\n| Algorithm | FND | HND | LND | Ctrl Energy (mJ) | Ctrl Msgs | Throughput |")
    print("|-----------|-----|-----|-----|------------------|-----------|------------|")
    for algo, r in results.items():
        print(f"| {algo.upper():9} | {r['FND_mean']:.1f}±{r['FND_std']:.1f} | "
              f"{r['HND_mean']:.1f}±{r['HND_std']:.1f} | "
              f"{r['LND_mean']:.1f}±{r['LND_std']:.1f} | "
              f"{r['ctrl_energy_mJ']:.1f} | "
              f"{r['ctrl_msgs']:.0f} | "
              f"{r['throughput']:.0f} |")

    # Compare with baseline
    print("\n\nBASELINE (from baseline_snapshot.md):")
    print("| AUCTION | 132.8±9.2 | 181.6±12.8 | 300.0±0.0 | N/A | 15965 | 9987 |")
    print("| HEED    | 36.8±6.9  | 218.0±22.2 | 300.0±0.0 | N/A | 20696 | 10298 |")
    print("| LEACH   | 54.0±8.6  | 135.6±10.5 | 217.8±13.6| N/A | 7190  | 6742 |")

    print("\n\nKEY CHANGES:")
    print("1. Control energy now affects lifetime (non-zero)")
    print("2. HEED should have lower lifetime due to high control overhead")
    print("3. Bandit should show diversity in m values (not all max)")

if __name__ == "__main__":
    run_comparison()
