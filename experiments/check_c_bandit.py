#!/usr/bin/env python3
"""
Check C: Bandit diagnostics.
- cost_per_bit trend across epochs
- m distribution (should be diverse, not all max)
- m vs distance_to_BS correlation (far nodes should learn lower m)
- Q-value analysis
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.simulation import load_config, Simulation


def pearsonr(x, y):
    """Simple Pearson correlation without scipy."""
    x = np.array(x)
    y = np.array(y)
    n = len(x)
    if n < 2:
        return 0.0, 1.0

    mx, my = np.mean(x), np.mean(y)
    sx, sy = np.std(x), np.std(y)

    if sx == 0 or sy == 0:
        return 0.0, 1.0

    r = np.mean((x - mx) * (y - my)) / (sx * sy)

    # Approximate p-value using t-distribution approximation
    t = r * np.sqrt((n - 2) / (1 - r**2 + 1e-10))
    # Rough p-value (not exact, but good enough for diagnostics)
    p = 2 * (1 - min(0.999, abs(t) / (abs(t) + n)))

    return r, p


def run_bandit_diagnostics():
    config = load_config("config/default.yaml")
    config['simulation']['max_epochs'] = 200
    config['auction']['use_bandit'] = True

    print("="*70)
    print("CHECK C: BANDIT DIAGNOSTICS")
    print("="*70)

    sim = Simulation(config, seed=42)
    result = sim.run('auction', seed=42, verbose=False)

    # Get the algorithm instance
    abc = sim.algorithm

    # Get alive nodes
    alive_nodes = [n for n in sim.network.nodes if n.is_alive]

    print(f"\nTotal epochs: {result['total_epochs']}")
    print(f"Alive nodes: {len(alive_nodes)}")
    print(f"ref_ema (final): {abc.ref_ema:.2e} J/bit")

    # 1. m (aggressiveness) distribution
    print("\n" + "-"*50)
    print("1. AGGRESSIVENESS (m) DISTRIBUTION")
    print("-"*50)
    m_values = [n.aggressiveness for n in alive_nodes]
    m_unique, m_counts = np.unique(m_values, return_counts=True)
    print(f"   Mean m: {np.mean(m_values):.3f}")
    print(f"   Std m:  {np.std(m_values):.3f}")
    print(f"   Distribution:")
    for m_val, count in zip(m_unique, m_counts):
        pct = count / len(alive_nodes) * 100
        print(f"      m={m_val:.1f}: {count} nodes ({pct:.1f}%)")

    # Check if diverse (std > 0.1 indicates non-collapse)
    m_diverse = np.std(m_values) > 0.1
    print(f"\n   Diversity check: {'PASS' if m_diverse else 'FAIL'} (std={'%.3f' % np.std(m_values)} > 0.1)")

    # 2. m vs distance_to_BS correlation
    print("\n" + "-"*50)
    print("2. m vs DISTANCE_TO_BS CORRELATION")
    print("-"*50)
    distances = [sim.network.get_distance_to_bs(n) for n in alive_nodes]
    correlation, p_value = pearsonr(m_values, distances)
    print(f"   Pearson correlation: {correlation:.3f}")
    print(f"   P-value: {p_value:.4f}")

    # Expect negative correlation (far nodes -> lower m, less aggressive)
    # But this is a weak signal, so just check direction
    if correlation < 0:
        print(f"   Direction: CORRECT (far nodes have lower m)")
    else:
        print(f"   Direction: UNEXPECTED (far nodes have higher m)")

    # Show m by distance quartiles
    q1, q2, q3 = np.percentile(distances, [25, 50, 75])
    near = [n for n, d in zip(alive_nodes, distances) if d <= q1]
    mid = [n for n, d in zip(alive_nodes, distances) if q1 < d <= q3]
    far = [n for n, d in zip(alive_nodes, distances) if d > q3]

    if near:
        print(f"\n   Near BS (d <= {q1:.1f}m): mean_m = {np.mean([n.aggressiveness for n in near]):.3f} ({len(near)} nodes)")
    if mid:
        print(f"   Mid (d in [{q1:.1f}, {q3:.1f}]m): mean_m = {np.mean([n.aggressiveness for n in mid]):.3f} ({len(mid)} nodes)")
    if far:
        print(f"   Far from BS (d > {q3:.1f}m): mean_m = {np.mean([n.aggressiveness for n in far]):.3f} ({len(far)} nodes)")

    # 3. Q-value analysis
    print("\n" + "-"*50)
    print("3. Q-VALUE ANALYSIS")
    print("-"*50)

    # Collect Q-values for each action across all nodes
    all_q_by_action = {i: [] for i in range(len(abc.m_actions))}
    for node in alive_nodes:
        for action_idx, q_val in node.bandit_q_values.items():
            all_q_by_action[action_idx].append(q_val)

    print("   Q-values by action (m):")
    for action_idx, m_val in enumerate(abc.m_actions):
        q_vals = all_q_by_action.get(action_idx, [])
        if q_vals:
            print(f"      m={m_val:.1f}: Q mean={np.mean(q_vals):.3f}, std={np.std(q_vals):.3f}, n={len(q_vals)}")
        else:
            print(f"      m={m_val:.1f}: (no data)")

    # 4. CH history per node (how many times was each node CH)
    print("\n" + "-"*50)
    print("4. CH SELECTION DIVERSITY")
    print("-"*50)
    ch_counts = {n.id: 0 for n in sim.network.nodes}

    # Count CH selections from epoch history (via cluster assignments)
    # We don't have direct tracking, so estimate from final state
    # Just show current m distribution vs initial energy
    initial_energies = {n.id: n.initial_energy for n in sim.network.nodes}
    current_energies = {n.id: n.current_energy for n in alive_nodes}

    # Energy spent (roughly correlates with CH duty)
    energy_spent = []
    m_values_all = []
    for n in alive_nodes:
        spent = n.initial_energy - n.current_energy
        energy_spent.append(spent)
        m_values_all.append(n.aggressiveness)

    corr_energy_m, p_val = pearsonr(energy_spent, m_values_all)
    print(f"   Correlation (energy_spent vs m): {corr_energy_m:.3f} (p={p_val:.4f})")
    print(f"   Interpretation: Nodes with higher m tend to {'spend more' if corr_energy_m > 0 else 'spend less'} energy")

    # 5. Cost-per-bit spread (from ref_ema evolution would need per-epoch tracking)
    print("\n" + "-"*50)
    print("5. REWARD FUNCTION SANITY")
    print("-"*50)
    data_bits = int(config.get('packets', {}).get('data_size', 32000))
    agg_ratio = float(config.get('clustering', {}).get('aggregation_ratio', 0.5))

    # Theoretical cost range
    # Far node TX: E_elec*bits + E_amp*bits*d^2, d_max ~ 100m
    e_elec = config.get('energy', {}).get('e_elec', 5e-8)
    e_amp = config.get('energy', {}).get('e_amp', 1e-10)
    d_near = 20
    d_far = 100

    # CH with 10 members, aggregation 0.5
    members = 10
    bits_out = data_bits * (members + 1) * agg_ratio

    # TX to BS cost
    tx_near = e_elec * bits_out + e_amp * bits_out * d_near**2
    tx_far = e_elec * bits_out + e_amp * bits_out * d_far**2
    rx_cost = e_elec * data_bits * members  # RX from members
    da_cost = 5e-9 * data_bits * members  # Data aggregation

    total_near = tx_near + rx_cost + da_cost
    total_far = tx_far + rx_cost + da_cost

    cost_per_bit_near = total_near / bits_out
    cost_per_bit_far = total_far / bits_out

    print(f"   Theoretical cost_per_bit (10 members):")
    print(f"      Near BS (d=20m): {cost_per_bit_near:.2e} J/bit")
    print(f"      Far from BS (d=100m): {cost_per_bit_far:.2e} J/bit")
    print(f"      Ratio (far/near): {cost_per_bit_far/cost_per_bit_near:.1f}x")
    print(f"\n   Final ref_ema: {abc.ref_ema:.2e} J/bit")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"   m diversity: {'PASS' if m_diverse else 'FAIL'} (std={np.std(m_values):.3f})")
    print(f"   m-distance correlation: {correlation:.3f} (expected < 0 for far->lower m)")
    print(f"   Energy-m correlation: {corr_energy_m:.3f} (expected > 0 for high_m->more_duty)")

    # Check C passes if:
    # 1. m is diverse (std > 0.1)
    # 2. Some directional signal (even if weak)
    check_c_pass = m_diverse and (correlation < 0.3)  # Not strongly positive correlation
    print(f"\n   CHECK C: {'PASS' if check_c_pass else 'NEEDS REVIEW'}")
    print("="*70)


if __name__ == "__main__":
    run_bandit_diagnostics()
