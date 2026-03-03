#!/usr/bin/env python3
"""
Generate single AUC_T winner map (no HEED phi_c panel).
Style matches fig_sweep_rtd_N100.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

# IEEE-friendly typography (single-column 3.5" target)
plt.rcParams.update({
    "font.family": "Arial",
    "font.weight": "normal",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

ARTIFACTS_DIR = Path("artifacts")

SWEEP_LDATA = [500, 2000, 8000, 32000]
SWEEP_S = [0.1, 1.0, 10.0]

def load_sweep_data():
    """Load sweep data from CSV and aggregate by taking mean over trials."""
    csv_path = ARTIFACTS_DIR / "sweep_metrics.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        # Aggregate by (protocol, Ldata, s), taking mean over trials
        grouped = df.groupby(['protocol', 'Ldata', 's']).agg({'auc_T': 'mean', 'phi_c_T': 'mean'}).reset_index()
        means = {}
        for _, row in grouped.iterrows():
            key = (row['protocol'], int(row['Ldata']), row['s'])
            means[key] = {'auc_T': row['auc_T'], 'phi_c_T': row['phi_c_T']}
        return means
    else:
        print("ERROR: sweep_metrics.csv not found. Run final_n100_experiment.py first.")
        return None


def generate_single_auc_winner_map(means):
    """Generate single AUC_T winner map matching RTD style."""
    protocols = ['ABC', 'HEED', 'LEACH-L', 'LEACH']
    proto_colors = {'ABC': '#2ecc71', 'HEED': '#3498db', 'LEACH-L': '#e74c3c', 'LEACH': '#9b59b6'}

    n_ldata = len(SWEEP_LDATA)
    n_s = len(SWEEP_S)

    # Compact size for 3-up figure row (fits 0.32*textwidth at 2-column width)
    fig, ax = plt.subplots(1, 1, figsize=(2.6, 1.9))

    # Build winner grid
    winner_grid = np.zeros((n_s, n_ldata))
    auc_grid = np.zeros((n_s, n_ldata))
    winner_names = [['' for _ in range(n_ldata)] for _ in range(n_s)]

    for i, s in enumerate(SWEEP_S):
        for j, Ldata in enumerate(SWEEP_LDATA):
            best_auc, best_proto = -1, None
            for proto in protocols:
                key = (proto, Ldata, s)
                if key in means:
                    auc = means[key]['auc_T']
                    if auc > best_auc:
                        best_auc = auc
                        best_proto = proto
            winner_names[i][j] = best_proto
            auc_grid[i][j] = best_auc
            winner_grid[i][j] = protocols.index(best_proto) if best_proto else -1

    cmap = mcolors.ListedColormap([proto_colors[p] for p in protocols])
    ax.imshow(winner_grid, cmap=cmap, aspect='auto', vmin=0, vmax=len(protocols)-1)

    for i in range(n_s):
        for j in range(n_ldata):
            text = f"{auc_grid[i][j]:.2f}"
            color = 'white' if winner_names[i][j] in ['LEACH-L', 'LEACH'] else 'black'
            ax.text(j, i, text, ha='center', va='center', fontsize=8, fontweight='normal', color=color)

    ax.set_xticks(range(n_ldata))
    ax.set_xticklabels([str(L) for L in SWEEP_LDATA])
    ax.set_yticks(range(n_s))
    ax.set_yticklabels([str(s) for s in SWEEP_S])
    ax.set_xlabel('$L_{\\mathrm{data}}$ (bits)', labelpad=8)
    ax.set_ylabel('Control scale $s$', labelpad=1)
    ax.tick_params(axis='x', pad=1)

    fig.subplots_adjust(left=0.20, right=0.98, top=0.96, bottom=0.30)

    for ext in ['pdf', 'png']:
        plt.savefig(ARTIFACTS_DIR / f"fig_aucT_winner_N100.{ext}", dpi=300)
    plt.close()
    print(f"Saved: fig_aucT_winner_N100.pdf/.png")


if __name__ == "__main__":
    means = load_sweep_data()
    if means:
        generate_single_auc_winner_map(means)
