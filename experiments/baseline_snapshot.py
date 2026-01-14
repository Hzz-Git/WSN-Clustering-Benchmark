#!/usr/bin/env python3
"""
Baseline snapshot before control-energy + bandit-reward fix.
Captures FND/HND/LND, control_messages, throughput for regression comparison.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.simulation import load_config, Simulation

def run_baseline():
    config = load_config("config/default.yaml")

    # Use fewer trials for quick baseline (5 trials)
    num_trials = 5
    seed_base = 42
    max_epochs = 300
    config['simulation']['max_epochs'] = max_epochs

    algorithms = ['auction', 'heed', 'leach']
    results = {}

    for algo_name in algorithms:
        print(f"\n{'='*60}")
        print(f"Running {algo_name.upper()} baseline ({num_trials} trials)...")
        print('='*60)

        fnd_list = []
        hnd_list = []
        lnd_list = []
        ctrl_msgs_list = []
        throughput_list = []

        for trial in range(num_trials):
            seed = seed_base + trial
            sim = Simulation(config, seed=seed)
            result = sim.run(algo_name, seed=seed, verbose=False)

            # Extract metrics from history
            total_ctrl = sum(h['control_messages'] for h in result['history'])
            total_packets = sum(h['data_packets'] for h in result['history'])

            fnd = result['fnd'] if result['fnd'] is not None else max_epochs
            hnd = result['hnd'] if result['hnd'] is not None else max_epochs
            lnd = result['lnd'] if result['lnd'] is not None else max_epochs

            fnd_list.append(fnd)
            hnd_list.append(hnd)
            lnd_list.append(lnd)
            ctrl_msgs_list.append(total_ctrl)
            throughput_list.append(total_packets)

            print(f"  Trial {trial+1}: FND={fnd}, HND={hnd}, LND={lnd}, ctrl={total_ctrl}, packets={total_packets}")

        results[algo_name] = {
            'FND_mean': np.mean(fnd_list),
            'FND_std': np.std(fnd_list),
            'HND_mean': np.mean(hnd_list),
            'HND_std': np.std(hnd_list),
            'LND_mean': np.mean(lnd_list),
            'LND_std': np.std(lnd_list),
            'ctrl_msgs_mean': np.mean(ctrl_msgs_list),
            'ctrl_msgs_std': np.std(ctrl_msgs_list),
            'throughput_mean': np.mean(throughput_list),
            'throughput_std': np.std(throughput_list),
        }

    # Write results to markdown
    output_path = Path("results/baseline_snapshot.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("# Baseline Snapshot (Before Control-Energy + Bandit-Reward Fix)\n\n")
        f.write(f"**Config**: default.yaml, {num_trials} trials, max {max_epochs} epochs\n\n")
        f.write("## Results Summary\n\n")
        f.write("| Algorithm | FND | HND | LND | Control Msgs | Throughput |\n")
        f.write("|-----------|-----|-----|-----|--------------|------------|\n")

        for algo, r in results.items():
            f.write(f"| {algo.upper()} | {r['FND_mean']:.1f}±{r['FND_std']:.1f} | "
                    f"{r['HND_mean']:.1f}±{r['HND_std']:.1f} | "
                    f"{r['LND_mean']:.1f}±{r['LND_std']:.1f} | "
                    f"{r['ctrl_msgs_mean']:.0f}±{r['ctrl_msgs_std']:.0f} | "
                    f"{r['throughput_mean']:.0f}±{r['throughput_std']:.0f} |\n")

        f.write("\n## Notes\n\n")
        f.write("- Control messages are **counted only**, not charged energy\n")
        f.write("- Bandit reward: 1.0 if CH, 0.1 otherwise (broken incentive)\n")
        f.write("- This is the baseline to compare against after fixes\n")

    print(f"\n\nBaseline saved to: {output_path}")
    print("\n" + "="*60)
    print("BASELINE SUMMARY")
    print("="*60)
    for algo, r in results.items():
        print(f"\n{algo.upper()}:")
        print(f"  FND: {r['FND_mean']:.1f} ± {r['FND_std']:.1f}")
        print(f"  HND: {r['HND_mean']:.1f} ± {r['HND_std']:.1f}")
        print(f"  LND: {r['LND_mean']:.1f} ± {r['LND_std']:.1f}")
        print(f"  Control msgs: {r['ctrl_msgs_mean']:.0f} ± {r['ctrl_msgs_std']:.0f}")
        print(f"  Throughput: {r['throughput_mean']:.0f} ± {r['throughput_std']:.0f}")

if __name__ == "__main__":
    run_baseline()
