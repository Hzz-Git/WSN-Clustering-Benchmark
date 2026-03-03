#!/usr/bin/env python3
"""
AUC* Experiment: Normalized lifetime metric using LND as horizon.

Key changes from original phase diagram:
1. AUC* = (1 / (N × LND)) × Σa(t) for t=1 to LND
   - Eliminates saturation (AUC ≈ 1) problem
   - Each trial uses its own LND as the horizon

2. N = 100 nodes (optionally 200)

3. Only 3 representative workload points:
   - Data-heavy / low-overhead: L_data = 32000, s = 0.1
   - Control-heavy / light-payload: L_data = 500, s = 10
   - Middle point: L_data = 8000, s = 1

Output: AUC*, φ_c, FND, HND, LND
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import copy
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from src.simulation import load_config, Simulation


def run_until_lnd(config, algo_name, seed):
    """
    Run simulation until all nodes die (LND).

    Returns:
        dict with AUC*, FND, HND, LND, phi_c, and history
    """
    sim = Simulation(config, seed=seed)

    # Setup
    np.random.seed(seed)
    sim.network = sim.setup_network(seed)
    energy_model = sim.setup_energy_model()
    sim.algorithm = sim.setup_algorithm(algo_name, sim.network, energy_model)
    sim.algorithm.setup()

    history = []
    initial_nodes = sim.network.count_alive()
    half_nodes = initial_nodes // 2
    initial_energy = sim.network.get_total_energy()

    fnd_epoch = None
    hnd_epoch = None
    lnd_epoch = None

    # Run until all nodes dead (with safety limit)
    max_safety_epochs = 10000  # Safety limit to prevent infinite loops
    epoch = 0

    while epoch < max_safety_epochs:
        epoch += 1
        stats = sim.algorithm.run_epoch()
        history.append(stats)

        alive = stats['alive_nodes']

        # Track FND
        if fnd_epoch is None and alive < initial_nodes:
            fnd_epoch = epoch

        # Track HND
        if hnd_epoch is None and alive <= half_nodes:
            hnd_epoch = epoch

        # Track LND and stop
        if alive == 0:
            lnd_epoch = epoch
            break

    # If network didn't die (safety limit reached), use last epoch as LND
    if lnd_epoch is None:
        lnd_epoch = epoch
        print(f"  WARNING: Network didn't die within {max_safety_epochs} epochs")

    # Calculate AUC* = (1 / (N × LND)) × Σa(t)
    # This is the normalized area under the alive-node curve
    alive_sum = sum(h['alive_nodes'] for h in history)
    auc_star = alive_sum / (initial_nodes * lnd_epoch)

    # Calculate control energy fraction φ_c
    total_ctrl_energy = sum(h.get('control_energy_j', 0) for h in history)
    total_energy_spent = initial_energy - sim.network.get_total_energy()
    phi_c = total_ctrl_energy / (total_energy_spent + 1e-9)

    return {
        'auc_star': auc_star,
        'fnd': fnd_epoch,
        'hnd': hnd_epoch,
        'lnd': lnd_epoch,
        'phi_c': phi_c,
        'initial_nodes': initial_nodes,
        'initial_energy': initial_energy,
        'history': history,
    }


def run_workload_point(config, algo_name, num_trials=10, seed_base=42):
    """
    Run multiple trials for a single workload configuration.

    Returns:
        dict with mean and std for all metrics
    """
    auc_star_list = []
    fnd_list = []
    hnd_list = []
    lnd_list = []
    phi_c_list = []

    for trial in range(num_trials):
        seed = seed_base + trial
        result = run_until_lnd(config, algo_name, seed)

        auc_star_list.append(result['auc_star'])
        fnd_list.append(result['fnd'] if result['fnd'] is not None else result['lnd'])
        hnd_list.append(result['hnd'] if result['hnd'] is not None else result['lnd'])
        lnd_list.append(result['lnd'])
        phi_c_list.append(result['phi_c'])

    return {
        'auc_star_mean': np.mean(auc_star_list),
        'auc_star_std': np.std(auc_star_list),
        'fnd_mean': np.mean(fnd_list),
        'fnd_std': np.std(fnd_list),
        'hnd_mean': np.mean(hnd_list),
        'hnd_std': np.std(hnd_list),
        'lnd_mean': np.mean(lnd_list),
        'lnd_std': np.std(lnd_list),
        'phi_c_mean': np.mean(phi_c_list),
        'phi_c_std': np.std(phi_c_list),
    }


def main():
    base_config = load_config("config/default.yaml")

    # Configuration: N = 100 nodes
    num_nodes = 100
    base_config['network']['num_nodes'] = num_nodes

    # 3 representative workload points
    workload_points = [
        {'name': 'Data-heavy (L=32k, s=0.1)', 'data_bits': 32000, 'ctrl_scale': 0.1},
        {'name': 'Middle (L=8k, s=1)', 'data_bits': 8000, 'ctrl_scale': 1.0},
        {'name': 'Control-heavy (L=500, s=10)', 'data_bits': 500, 'ctrl_scale': 10.0},
    ]

    # Algorithms
    algorithms = [
        ('auction', 'ABC', 'local'),
        ('heed', 'HEED', 'local'),
        ('leach', 'LEACH-L', 'local'),
        ('leach', 'LEACH', 'global'),
    ]

    num_trials = 10

    # Output setup
    output_dir = Path("results/data/auc_star")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"auc_star_N{num_nodes}_{timestamp}.csv"
    fig_path = output_dir / f"auc_star_N{num_nodes}_{timestamp}.pdf"

    print("=" * 80)
    print(f"AUC* EXPERIMENT (N = {num_nodes} nodes)")
    print("=" * 80)
    print(f"Workload points: {len(workload_points)}")
    print(f"Algorithms: {[a[1] for a in algorithms]}")
    print(f"Trials per config: {num_trials}")
    print("=" * 80)

    results = []
    total_runs = len(workload_points) * len(algorithms)
    run_count = 0

    for wp in workload_points:
        for algo_key, algo_label, discovery_mode in algorithms:
            run_count += 1
            print(f"\n[{run_count}/{total_runs}] {algo_label} | {wp['name']}")

            # Configure
            config = copy.deepcopy(base_config)
            config['packets']['data_size'] = wp['data_bits']
            config['control']['bits_multiplier'] = wp['ctrl_scale']
            config['control']['discovery_radius_mode'] = discovery_mode

            # Run trials
            metrics = run_workload_point(config, algo_key, num_trials=num_trials)

            # Record
            row = {
                'workload': wp['name'],
                'data_bits': wp['data_bits'],
                'ctrl_scale': wp['ctrl_scale'],
                'algorithm': algo_label,
                'AUC_star_mean': metrics['auc_star_mean'],
                'AUC_star_std': metrics['auc_star_std'],
                'FND_mean': metrics['fnd_mean'],
                'FND_std': metrics['fnd_std'],
                'HND_mean': metrics['hnd_mean'],
                'HND_std': metrics['hnd_std'],
                'LND_mean': metrics['lnd_mean'],
                'LND_std': metrics['lnd_std'],
                'phi_c_mean': metrics['phi_c_mean'],
                'phi_c_std': metrics['phi_c_std'],
            }
            results.append(row)

            print(f"  AUC*={metrics['auc_star_mean']:.4f}+/-{metrics['auc_star_std']:.4f}, "
                  f"LND={metrics['lnd_mean']:.0f}+/-{metrics['lnd_std']:.0f}, "
                  f"phi_c={metrics['phi_c_mean']*100:.1f}%")

    # Write CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n\nResults saved to: {csv_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    # Table header
    print(f"\n{'Workload':<30} | {'Algorithm':<10} | {'AUC*':>12} | {'phi_c':>8} | {'FND':>8} | {'HND':>8} | {'LND':>8}")
    print("-" * 100)

    for wp in workload_points:
        for algo_label in ['ABC', 'HEED', 'LEACH-L', 'LEACH']:
            matching = [r for r in results
                       if r['algorithm'] == algo_label
                       and r['data_bits'] == wp['data_bits']
                       and r['ctrl_scale'] == wp['ctrl_scale']]
            if matching:
                r = matching[0]
                print(f"{wp['name']:<30} | {algo_label:<10} | "
                      f"{r['AUC_star_mean']:.4f}+/-{r['AUC_star_std']:.3f} | "
                      f"{r['phi_c_mean']*100:5.1f}% | "
                      f"{r['FND_mean']:6.0f} | "
                      f"{r['HND_mean']:6.0f} | "
                      f"{r['LND_mean']:6.0f}")
        print("-" * 100)

    # Create bar chart
    create_bar_chart(results, workload_points, fig_path, num_nodes)
    print(f"\nFigure saved to: {fig_path}")


def create_bar_chart(results, workload_points, fig_path, num_nodes):
    """Create a grouped bar chart for AUC* comparison."""

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    algos = ['ABC', 'HEED', 'LEACH-L', 'LEACH']
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    x = np.arange(len(algos))
    width = 0.6

    for idx, wp in enumerate(workload_points):
        ax = axes[idx]

        auc_means = []
        auc_stds = []
        phi_c_vals = []

        for algo in algos:
            matching = [r for r in results
                       if r['algorithm'] == algo
                       and r['data_bits'] == wp['data_bits']
                       and r['ctrl_scale'] == wp['ctrl_scale']]
            if matching:
                r = matching[0]
                auc_means.append(r['AUC_star_mean'])
                auc_stds.append(r['AUC_star_std'])
                phi_c_vals.append(r['phi_c_mean'])
            else:
                auc_means.append(0)
                auc_stds.append(0)
                phi_c_vals.append(0)

        bars = ax.bar(x, auc_means, width, yerr=auc_stds, capsize=3,
                      color=colors, edgecolor='black', linewidth=0.5)

        # Annotate with phi_c
        for i, (bar, phi_c) in enumerate(zip(bars, phi_c_vals)):
            height = bar.get_height()
            ax.annotate(f'$\\phi_c$={phi_c*100:.0f}%',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3),
                       textcoords='offset points',
                       ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Algorithm')
        ax.set_ylabel('AUC*')
        ax.set_title(f"{wp['name']}\n(L={wp['data_bits']}, s={wp['ctrl_scale']})")
        ax.set_xticks(x)
        ax.set_xticklabels(algos, rotation=15, ha='right')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle(f'AUC* Comparison (N={num_nodes} nodes)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
