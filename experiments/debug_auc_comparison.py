#!/usr/bin/env python3
"""
Debug script: Compare AUC vs AUC* and analyze alive-node curves.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import copy
import numpy as np
import matplotlib.pyplot as plt
from src.simulation import load_config, Simulation


def run_and_get_curve(config, algo_name, seed, max_epochs=500):
    """Run simulation and return alive-node curve."""
    sim = Simulation(config, seed=seed)
    np.random.seed(seed)
    sim.network = sim.setup_network(seed)
    energy_model = sim.setup_energy_model()
    sim.algorithm = sim.setup_algorithm(algo_name, sim.network, energy_model)
    sim.algorithm.setup()

    history = []
    initial_nodes = sim.network.count_alive()

    epoch = 0
    while epoch < max_epochs:
        epoch += 1
        stats = sim.algorithm.run_epoch()
        history.append(stats)
        if stats['alive_nodes'] == 0:
            break

    alive_curve = [h['alive_nodes'] for h in history]
    lnd = len(alive_curve)

    # Original AUC (fixed T)
    alive_sum = sum(alive_curve)
    # Pad with zeros if died early
    auc_original = alive_sum / (initial_nodes * max_epochs)

    # AUC* (normalized by LND)
    auc_star = alive_sum / (initial_nodes * lnd)

    return {
        'alive_curve': alive_curve,
        'lnd': lnd,
        'auc_original': auc_original,
        'auc_star': auc_star,
        'initial_nodes': initial_nodes,
    }


def main():
    base_config = load_config("config/default.yaml")

    # Test with BOTH N=50 (original) and N=100
    for num_nodes in [50, 100]:
        print(f"\n{'='*60}")
        print(f"N = {num_nodes} nodes")
        print('='*60)

        base_config['network']['num_nodes'] = num_nodes

        # Data-heavy workload (L=32000, s=0.1)
        config = copy.deepcopy(base_config)
        config['packets']['data_size'] = 32000
        config['control']['bits_multiplier'] = 0.1

        algorithms = [
            ('auction', 'ABC', 'local'),
            ('heed', 'HEED', 'local'),
            ('leach', 'LEACH-L', 'local'),
            ('leach', 'LEACH', 'global'),
        ]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        print(f"\nData-heavy workload (L=32000, s=0.1):")
        print(f"{'Algorithm':<12} | {'AUC(T=500)':>12} | {'AUC*':>12} | {'LND':>8} | {'FND':>8}")
        print("-" * 60)

        for algo_key, algo_label, discovery_mode in algorithms:
            cfg = copy.deepcopy(config)
            cfg['control']['discovery_radius_mode'] = discovery_mode

            result = run_and_get_curve(cfg, algo_key, seed=42, max_epochs=500)

            print(f"{algo_label:<12} | {result['auc_original']:>12.4f} | {result['auc_star']:>12.4f} | "
                  f"{result['lnd']:>8} | {np.argmax(np.array(result['alive_curve']) < num_nodes)+1 if min(result['alive_curve']) < num_nodes else 'N/A':>8}")

            # Plot alive curve
            axes[0].plot(result['alive_curve'], label=f"{algo_label} (LND={result['lnd']})")

            # Plot normalized curve (fraction of initial)
            normalized = np.array(result['alive_curve']) / num_nodes
            axes[1].plot(normalized, label=algo_label)

        axes[0].set_xlabel('Round')
        axes[0].set_ylabel('Alive Nodes')
        axes[0].set_title(f'Alive Curves (N={num_nodes}, L=32000, s=0.1)')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].set_xlabel('Round')
        axes[1].set_ylabel('Fraction Alive')
        axes[1].set_title(f'Normalized Alive Curves')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"results/data/auc_star/debug_curves_N{num_nodes}.pdf", dpi=150)
        plt.close()
        print(f"\nCurve plot saved to: results/data/auc_star/debug_curves_N{num_nodes}.pdf")

        # Now explain the difference
        print(f"\n--- Analysis ---")
        print("AUC(T=500) = sum(alive) / (N * 500)  <- fixed horizon, rewards longer life")
        print("AUC*       = sum(alive) / (N * LND)  <- variable horizon, rewards 'fullness' of curve")
        print("\nIf a protocol dies fast but keeps nodes alive until death -> high AUC*")
        print("If a protocol has early deaths but long tail -> low AUC*")


if __name__ == "__main__":
    main()
