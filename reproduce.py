#!/usr/bin/env python3
"""
One-click reproduction script for all paper results and figures.

Usage:
    python reproduce.py          # Run everything (simulations + figures)
    python reproduce.py --quick  # Figures only (requires existing CSV data in artifacts/)

Outputs saved to artifacts/ directory:
    - sweep_metrics.csv           Fixed-horizon sweep data (10 trials per cell)
    - sweep_rtd_metrics.csv       Run-to-death sweep data
    - fig_aucT_winner_N100.pdf    Figure 1a: AUC_T winner map
    - fig_aucstar_winner_N100.pdf Figure 1b: AUC* winner map
    - fig_mnl_winner_N100.pdf     Figure 1c: MNL winner map
    - fig_phi_c_all_N100.pdf      Figure 2:  phi_c line graph
"""

import sys
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def run_simulations():
    """Run the full 2D sweep experiments (takes ~30-60 min)."""
    print("=" * 60)
    print("STEP 1/3: Running simulations")
    print("  4 protocols x 4 payloads x 3 control scales x 10 trials")
    print("=" * 60)

    from experiments.final_n100_experiment import main as run_full
    run_full()


def generate_figures():
    """Generate all paper figures from CSV data."""
    print("\n" + "=" * 60)
    print("STEP 2/3: Generating figures")
    print("=" * 60)

    artifacts = ROOT / "artifacts"

    # Check data exists
    for csv in ["sweep_metrics.csv", "sweep_rtd_metrics.csv"]:
        if not (artifacts / csv).exists():
            print(f"ERROR: {csv} not found in artifacts/.")
            print("Run without --quick to generate data first.")
            sys.exit(1)

    # Figure 1a: AUC_T winner map
    print("\n  [1/4] AUC_T winner map...")
    from experiments.generate_single_winner_map import load_sweep_data, generate_single_auc_winner_map
    means = load_sweep_data()
    if means:
        generate_single_auc_winner_map(means)

    # Figure 1b-c: AUC* and MNL winner maps
    print("  [2/4] AUC* and MNL winner maps...")
    from experiments.generate_rtd_winner_maps import load_rtd_data, generate_single_winner_map
    rtd_means = load_rtd_data()
    if rtd_means:
        generate_single_winner_map(rtd_means, 'auc_star', '$\\mathrm{AUC}^*$',
                                   'fig_aucstar_winner_N100', fmt='.2f')
        generate_single_winner_map(rtd_means, 'mean_lifetime', 'MNL',
                                   'fig_mnl_winner_N100', fmt='.0f')

    # Figure 2: phi_c line graph
    print("  [3/4] phi_c line graph...")
    from experiments.generate_phi_c_line_graph import load_sweep_data as load_phi, generate_phi_c_line_graph
    phi_data = load_phi()
    if phi_data:
        generate_phi_c_line_graph(phi_data)

    print("  [4/4] Done.")


def print_summary():
    """Print output file listing."""
    print("\n" + "=" * 60)
    print("STEP 3/3: Summary")
    print("=" * 60)

    artifacts = ROOT / "artifacts"
    expected = [
        "sweep_metrics.csv",
        "sweep_rtd_metrics.csv",
        "fig_aucT_winner_N100.pdf",
        "fig_aucstar_winner_N100.pdf",
        "fig_mnl_winner_N100.pdf",
        "fig_phi_c_all_N100.pdf",
    ]

    for f in expected:
        path = artifacts / f
        status = "OK" if path.exists() else "MISSING"
        print(f"  [{status}] artifacts/{f}")

    print("\nAll outputs in:", artifacts.absolute())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce all paper results and figures.")
    parser.add_argument("--quick", action="store_true",
                        help="Skip simulations, generate figures from existing CSV data only.")
    args = parser.parse_args()

    start = time.time()

    if not args.quick:
        run_simulations()

    generate_figures()
    print_summary()

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.0f}s")
