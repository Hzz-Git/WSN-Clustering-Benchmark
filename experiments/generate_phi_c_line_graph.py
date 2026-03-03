#!/usr/bin/env python3
"""
Generate IEEE-style line graph for control-energy fraction phi_c.
Replaces the heatmap figure (fig_phi_c_all_N100) with a 3-panel line plot,
one panel per control scale s in {0.1, 1.0, 10.0}.

Reads: artifacts/sweep_metrics.csv
Saves: artifacts/fig_phi_c_all_N100.pdf/.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# IEEE-friendly typography (matching existing winner-map scripts)
plt.rcParams.update({
    "font.family": "Arial",
    "font.weight": "normal",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
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

PROTOCOLS = ['ABC', 'HEED', 'LEACH-L', 'LEACH']

# Distinct color + marker + linestyle for grayscale safety
PROTO_STYLE = {
    'ABC':     {'color': '#2ecc71', 'marker': 'o', 'linestyle': '-'},
    'HEED':    {'color': '#3498db', 'marker': 's', 'linestyle': '--'},
    'LEACH-L': {'color': '#e74c3c', 'marker': '^', 'linestyle': ':'},
    'LEACH': {'color': '#9b59b6', 'marker': 'D', 'linestyle': '-.'},
}


def load_sweep_data():
    """Load sweep_metrics.csv and compute per-cell mean and stderr of phi_c_T."""
    csv_path = ARTIFACTS_DIR / "sweep_metrics.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run final_n100_experiment.py first.")
        return None

    df = pd.read_csv(csv_path)
    grouped = df.groupby(['protocol', 'Ldata', 's'])['phi_c_T'].agg(['mean', 'std', 'count']).reset_index()
    grouped['stderr'] = grouped['std'] / np.sqrt(grouped['count'])

    data = {}
    for _, row in grouped.iterrows():
        key = (row['protocol'], int(row['Ldata']), row['s'])
        data[key] = {'mean': row['mean'], 'stderr': row['stderr']}
    return data


def generate_phi_c_line_graph(data):
    """Generate 3-panel line graph: one panel per control scale s."""
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.4), sharey=True)

    # Horizontal jitter multipliers (in log space) to separate overlapping markers
    jitter = {'ABC': 0.92, 'HEED': 0.97, 'LEACH-L': 1.03, 'LEACH': 1.08}

    for panel_idx, s_val in enumerate(SWEEP_S):
        ax = axes[panel_idx]

        for proto in PROTOCOLS:
            style = PROTO_STYLE[proto]
            means = []
            stderrs = []
            x_positions = []
            for Ldata in SWEEP_LDATA:
                key = (proto, Ldata, s_val)
                x_positions.append(Ldata * jitter[proto])
                if key in data:
                    means.append(data[key]['mean'])
                    stderrs.append(data[key]['stderr'])
                else:
                    means.append(np.nan)
                    stderrs.append(0)

            ax.errorbar(
                x_positions, means, yerr=stderrs,
                label=proto,
                color=style['color'],
                marker=style['marker'],
                linestyle=style['linestyle'],
                linewidth=1.5,
                markersize=5,
                capsize=3,
                capthick=0.8,
            )

        ax.set_xscale('log')
        ax.set_xticks(SWEEP_LDATA)
        ax.set_xticklabels(['500', '2k', '8k', '32k'])
        ax.set_xlim(350, 45000)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f'({chr(97 + panel_idx)}) $s = {s_val}$')
        ax.set_xlabel('$L_{\\mathrm{data}}$ (bits)')
        ax.grid(True, alpha=0.3, linewidth=0.5, linestyle=':')

        ax.set_ylabel('$\\phi_c$')
        ax.tick_params(labelleft=True)

    # Shared legend below panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4,
               frameon=True, framealpha=0.95, edgecolor='black',
               fancybox=False, bbox_to_anchor=(0.5, -0.02))

    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.28, wspace=0.30)

    for ext in ['pdf', 'png']:
        out = ARTIFACTS_DIR / f"fig_phi_c_all_N100.{ext}"
        fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: fig_phi_c_all_N100.pdf/.png")


if __name__ == "__main__":
    data = load_sweep_data()
    if data:
        generate_phi_c_line_graph(data)
