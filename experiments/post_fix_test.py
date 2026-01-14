#!/usr/bin/env python3
"""
Post-fix regression test.
Verifies control energy and bandit reward are working.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.simulation import load_config, Simulation

def run_test():
    config = load_config("config/default.yaml")
    config['simulation']['max_epochs'] = 100  # Quick test

    print("=" * 70)
    print("POST-FIX REGRESSION TEST")
    print("=" * 70)

    algorithms = ['auction', 'heed', 'leach']

    for algo_name in algorithms:
        print(f"\n--- Testing {algo_name.upper()} ---")

        sim = Simulation(config, seed=42)
        result = sim.run(algo_name, seed=42, verbose=False)

        # Get final stats
        total_ctrl_energy = sum(h.get('control_energy_j', 0) for h in result['history'])
        total_ctrl_msgs = sum(h['control_messages'] for h in result['history'])
        total_data_energy = result['initial_energy'] - result['final_energy'] - total_ctrl_energy

        print(f"  FND: {result['fnd']}, HND: {result['hnd']}, LND: {result['lnd']}")
        print(f"  Total epochs: {result['total_epochs']}")
        print(f"  Control messages: {total_ctrl_msgs}")
        print(f"  Control energy: {total_ctrl_energy * 1000:.2f} mJ")
        print(f"  Data energy: {total_data_energy * 1000:.2f} mJ")
        print(f"  Control/Total ratio: {total_ctrl_energy / (result['initial_energy'] - result['final_energy'] + 1e-9) * 100:.2f}%")

        # For ABC, check bandit learning
        if algo_name == 'auction':
            alive_nodes = [n for n in sim.network.nodes if n.is_alive]
            if alive_nodes:
                m_values = [n.aggressiveness for n in alive_nodes]
                print(f"  Alive nodes: {len(alive_nodes)}")
                print(f"  m (aggressiveness): mean={np.mean(m_values):.2f}, std={np.std(m_values):.2f}")
                print(f"  m distribution: {dict(zip(*np.unique(m_values, return_counts=True)))}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE - Check that control_energy_j > 0 for all algorithms")
    print("=" * 70)

if __name__ == "__main__":
    run_test()
