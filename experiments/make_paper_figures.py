#!/usr/bin/env python3
"""
Paper-ready figure generator for VTC submission.

Generates vector PDF figures from phase diagram and bandit evidence CSVs.

Usage:
    python make_paper_figures.py --phase_csv results/data/phase_diagram/phase_diagram_XXXX.csv \
                                  --bandit_csv results/data/bandit_evidence/bandit_spatial_XXXX.csv \
                                  --outdir figs/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


# Paper-friendly styling
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# Algorithm colors (consistent across figures)
ALGO_COLORS = {
    'ABC': '#1f77b4',        # Blue
    'HEED': '#ff7f0e',       # Orange
    'LEACH-L': '#2ca02c',    # Green
    'LEACH': '#d62728',    # Red
    'LEACH-local': '#2ca02c',
    'LEACH-global': '#d62728',
}

ALGO_LABELS = {
    'ABC': 'ABC',
    'HEED': 'HEED',
    'LEACH-L': 'LEACH-local',
    'LEACH': 'LEACH-global',
}


def load_phase_csv(path):
    """Load and validate phase diagram CSV."""
    df = pd.read_csv(path)
    required_cols = ['algorithm', 'data_bits', 'ctrl_scale', 'AUC_mean']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    return df


def load_bandit_csv(path):
    """Load and validate bandit evidence CSV."""
    df = pd.read_csv(path)
    return df


def fig1_phase_auc_winner_and_ctrlfrac(df, outdir, ctrlfrac_algo='HEED'):
    """
    Figure 1: Two-panel phase diagram
    (a) AUC winner map
    (b) Control energy fraction heatmap
    """
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))

    data_bits_vals = sorted(df['data_bits'].unique())
    ctrl_scale_vals = sorted(df['ctrl_scale'].unique())

    n_data = len(data_bits_vals)
    n_ctrl = len(ctrl_scale_vals)

    # Map algorithm to color index
    algo_list = ['ABC', 'HEED', 'LEACH-L', 'LEACH']
    algo_to_idx = {a: i for i, a in enumerate(algo_list)}
    colors = [ALGO_COLORS[a] for a in algo_list]
    cmap = ListedColormap(colors)

    # (a) AUC Winner Map
    ax = axes[0]
    winner_grid = np.zeros((n_ctrl, n_data))
    auc_grid = np.zeros((n_ctrl, n_data))

    for i, ctrl_scale in enumerate(ctrl_scale_vals):
        for j, data_bits in enumerate(data_bits_vals):
            subset = df[(df['data_bits'] == data_bits) & (df['ctrl_scale'] == ctrl_scale)]
            if len(subset) == 0:
                continue
            # Find winner (highest AUC_mean, tie-break by std then alphabetical)
            best = subset.sort_values(
                ['AUC_mean', 'AUC_std', 'algorithm'],
                ascending=[False, True, True]
            ).iloc[0]
            winner_grid[i, j] = algo_to_idx.get(best['algorithm'], 0)
            auc_grid[i, j] = best['AUC_mean']

    im = ax.imshow(winner_grid, cmap=cmap, aspect='auto', vmin=0, vmax=len(algo_list)-1)

    # Add text annotations
    for i in range(n_ctrl):
        for j in range(n_data):
            ctrl_scale = ctrl_scale_vals[i]
            data_bits = data_bits_vals[j]
            subset = df[(df['data_bits'] == data_bits) & (df['ctrl_scale'] == ctrl_scale)]
            if len(subset) == 0:
                continue
            best = subset.sort_values(
                ['AUC_mean', 'AUC_std', 'algorithm'],
                ascending=[False, True, True]
            ).iloc[0]
            label = best['algorithm'].replace('LEACH-', 'L-')
            auc = best['AUC_mean']
            ax.text(j, i, f"{label}\n{auc:.2f}", ha='center', va='center', fontsize=7,
                   color='white' if best['algorithm'] in ['ABC', 'LEACH'] else 'black')

    ax.set_xticks(range(n_data))
    ax.set_xticklabels([str(d) for d in data_bits_vals])
    ax.set_yticks(range(n_ctrl))
    ax.set_yticklabels([f'{c}x' for c in ctrl_scale_vals])
    ax.set_xlabel('Data Packet Size (bits)')
    ax.set_ylabel('Control Scale')
    ax.set_title('(a) AUC Winner')

    # Legend (outside plot to avoid covering data)
    legend_elements = [Patch(facecolor=ALGO_COLORS[a], label=ALGO_LABELS.get(a, a))
                      for a in algo_list]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=2, fontsize=7, frameon=False)

    # (b) Control energy fraction heatmap for specified algorithm
    ax = axes[1]
    ctrl_frac_grid = np.zeros((n_ctrl, n_data))

    for i, ctrl_scale in enumerate(ctrl_scale_vals):
        for j, data_bits in enumerate(data_bits_vals):
            subset = df[(df['data_bits'] == data_bits) &
                       (df['ctrl_scale'] == ctrl_scale) &
                       (df['algorithm'] == ctrlfrac_algo)]
            if len(subset) > 0:
                ctrl_frac_grid[i, j] = subset.iloc[0]['ctrl_frac']

    im2 = ax.imshow(ctrl_frac_grid, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

    # Add text annotations
    for i in range(n_ctrl):
        for j in range(n_data):
            val = ctrl_frac_grid[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8,
                   color='white' if val > 0.5 else 'black')

    ax.set_xticks(range(n_data))
    ax.set_xticklabels([str(d) for d in data_bits_vals])
    ax.set_yticks(range(n_ctrl))
    ax.set_yticklabels([f'{c}x' for c in ctrl_scale_vals])
    ax.set_xlabel('Data Packet Size (bits)')
    ax.set_ylabel('Control Scale')
    ax.set_title(f'(b) $\\phi_c$ ({ctrlfrac_algo})')

    cbar = fig.colorbar(im2, ax=ax, shrink=0.8)
    cbar.set_label('$\\phi_c$')

    plt.tight_layout()
    outpath = outdir / 'phase_auc_winner_and_ctrlfrac.pdf'
    plt.savefig(outpath, format='pdf')
    plt.close()
    print(f"Generated: {outpath}")


def fig2_auc_vs_ctrlfrac(df, outdir):
    """
    Figure 2: AUC vs control energy fraction, faceted by control scale.
    Three panels: (a) s=0.1, (b) s=1.0, (c) s=10.0
    Each panel shows 4 protocols with trajectories across payload sizes.
    """
    ctrl_scales = [0.1, 1.0, 10.0]

    fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharey=True)

    for ax_idx, ctrl_scale in enumerate(ctrl_scales):
        ax = axes[ax_idx]
        subset_scale = df[df['ctrl_scale'] == ctrl_scale]

        for algo in ['ABC', 'HEED', 'LEACH-L', 'LEACH']:
            subset_algo = subset_scale[subset_scale['algorithm'] == algo]
            if len(subset_algo) == 0:
                continue

            color = ALGO_COLORS.get(algo, 'gray')
            label = ALGO_LABELS.get(algo, algo)

            # Sort by data_bits for trajectory
            subset_algo = subset_algo.sort_values('data_bits')

            # Plot trajectory line with markers
            ax.plot(subset_algo['ctrl_frac'], subset_algo['AUC_mean'],
                   c=color, alpha=0.7, linewidth=1.5, marker='o', markersize=6,
                   markeredgecolor='none', zorder=2)

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('$\\phi_c$')
        ax.set_title(f'({"abc"[ax_idx]}) $s={ctrl_scale}$')
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('AUC')

    # Protocol legend (colors only)
    from matplotlib.lines import Line2D
    protocol_handles = [Line2D([0], [0], color=ALGO_COLORS[a], linewidth=2, marker='o',
                               markersize=5, label=ALGO_LABELS.get(a, a))
                       for a in ['ABC', 'HEED', 'LEACH-L', 'LEACH']]

    fig.legend(handles=protocol_handles,
              loc='upper center', bbox_to_anchor=(0.5, -0.02),
              ncol=4, fontsize=8, frameon=False)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    outpath = outdir / 'auc_vs_ctrlfrac.pdf'
    plt.savefig(outpath, format='pdf')
    plt.close()
    print(f"Generated: {outpath}")


def fig3_bandit_spatial_policy(df, outdir):
    """
    Figure 3: Bandit spatial policy learning.
    (a) m by distance group over time
    (b) m-distance correlation over time
    """
    if df is None or len(df) == 0:
        print("Skipping fig3: no bandit data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    # (a) m by distance group
    ax = axes[0]
    if 'm_near' in df.columns:
        # Plot with NaN handling (matplotlib will break lines at NaN)
        ax.plot(df['epoch'], df['m_near'], 'b-', label='Near BS', linewidth=1.5)
        ax.plot(df['epoch'], df['m_mid'], 'g--', label='Mid', linewidth=1.5)
        ax.plot(df['epoch'], df['m_far'], 'r:', label='Far from BS', linewidth=1.5)
        ax.set_xlabel('Round')
        ax.set_ylabel('Mean Aggressiveness ($m_i$)')
        ax.set_title('(a) Learned $m_i$ by Distance Group')
        ax.legend(loc='lower left', fontsize=8, frameon=True)
        ax.grid(True, alpha=0.3)
        # Set y-axis to start from 0.4 (min action is 0.5) to make curves clearer
        ax.set_ylim(0.4, None)

    # (b) m-distance correlation
    ax = axes[1]
    if 'm_dist_corr' in df.columns:
        ax.plot(df['epoch'], df['m_dist_corr'], 'k-', linewidth=1.5)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Round')
        ax.set_ylabel('Correlation ($m_i$ vs $d_{i,\\mathrm{BS}}$)')
        ax.set_title('(b) $m_i$--Distance Correlation')
        ax.set_ylim(-1, 1)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = outdir / 'bandit_spatial_policy.pdf'
    plt.savefig(outpath, format='pdf')
    plt.close()
    print(f"Generated: {outpath}")


def fig4_fairness_share_mode(fairness_csv_path, outdir):
    """
    Figure 4: Fairness share mode tradeoff (Gini vs AUC).
    Reads from fairness ablation CSV.
    """
    if not fairness_csv_path or not Path(fairness_csv_path).exists():
        print("Skipping fig4: no fairness ablation data")
        return

    df = pd.read_csv(fairness_csv_path)

    fig, ax = plt.subplots(figsize=(4, 3))

    mode_labels = {'alive_only': 'Alive-only shares', 'all_nodes': 'All-node shares'}
    for mode in df['share_mode'].unique():
        subset = df[df['share_mode'] == mode]
        color = '#1f77b4' if mode == 'alive_only' else '#ff7f0e'
        label = mode_labels.get(mode, mode)
        ax.errorbar(subset['Gini_mean'].values[0], subset['AUC_mean'].values[0],
                   xerr=subset['Gini_std'].values[0], yerr=subset['AUC_std'].values[0],
                   fmt='o', color=color, label=label, capsize=4, markersize=8)

    ax.set_xlabel('Gini Coefficient')
    ax.set_ylabel('AUC')
    ax.legend(loc='lower left', fontsize=9, frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_title('Share Mode Tradeoff')

    plt.tight_layout()
    outpath = outdir / 'fairness_share_mode_tradeoff.pdf'
    plt.savefig(outpath, format='pdf')
    plt.close()
    print(f"Generated: {outpath}")


def fig_supplementary_heatmaps(df, outdir):
    """
    Supplementary: AUC heatmaps for each algorithm.
    """
    data_bits_vals = sorted(df['data_bits'].unique())
    ctrl_scale_vals = sorted(df['ctrl_scale'].unique())
    n_data = len(data_bits_vals)
    n_ctrl = len(ctrl_scale_vals)

    algos = df['algorithm'].unique()
    n_algos = len(algos)

    fig, axes = plt.subplots(1, n_algos, figsize=(3*n_algos, 3))
    if n_algos == 1:
        axes = [axes]

    for ax, algo in zip(axes, algos):
        grid = np.zeros((n_ctrl, n_data))
        for i, ctrl_scale in enumerate(ctrl_scale_vals):
            for j, data_bits in enumerate(data_bits_vals):
                subset = df[(df['data_bits'] == data_bits) &
                           (df['ctrl_scale'] == ctrl_scale) &
                           (df['algorithm'] == algo)]
                if len(subset) > 0:
                    grid[i, j] = subset.iloc[0]['AUC_mean']

        im = ax.imshow(grid, cmap='viridis', aspect='auto', vmin=0, vmax=1)

        for i in range(n_ctrl):
            for j in range(n_data):
                val = grid[i, j]
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7,
                       color='white' if val < 0.5 else 'black')

        ax.set_xticks(range(n_data))
        ax.set_xticklabels([str(d) for d in data_bits_vals], fontsize=7)
        ax.set_yticks(range(n_ctrl))
        ax.set_yticklabels([f'{c}x' for c in ctrl_scale_vals], fontsize=7)
        ax.set_xlabel('Data bits')
        ax.set_title(algo)

    plt.tight_layout()
    outpath = outdir / 'phase_auc_heatmaps_by_algo.pdf'
    plt.savefig(outpath, format='pdf')
    plt.close()
    print(f"Generated: {outpath}")


def main():
    parser = argparse.ArgumentParser(description='Generate paper-ready figures')
    parser.add_argument('--phase_csv', type=str, required=True,
                       help='Path to phase diagram CSV')
    parser.add_argument('--bandit_csv', type=str, default=None,
                       help='Path to bandit spatial evidence CSV')
    parser.add_argument('--fairness_csv', type=str, default=None,
                       help='Path to fairness ablation CSV')
    parser.add_argument('--outdir', type=str, default='figs/',
                       help='Output directory for figures')
    parser.add_argument('--ctrlfrac_algo', type=str, default='HEED',
                       help='Algorithm for ctrl_frac heatmap in Fig 1b')

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading phase data from: {args.phase_csv}")
    phase_df = load_phase_csv(args.phase_csv)

    bandit_df = None
    if args.bandit_csv and Path(args.bandit_csv).exists():
        print(f"Loading bandit data from: {args.bandit_csv}")
        bandit_df = load_bandit_csv(args.bandit_csv)

    # Generate figures
    print("\nGenerating figures...")
    fig1_phase_auc_winner_and_ctrlfrac(phase_df, outdir, args.ctrlfrac_algo)
    fig2_auc_vs_ctrlfrac(phase_df, outdir)
    fig3_bandit_spatial_policy(bandit_df, outdir)
    fig4_fairness_share_mode(args.fairness_csv, outdir)
    fig_supplementary_heatmaps(phase_df, outdir)

    print(f"\nAll figures saved to: {outdir}")


if __name__ == "__main__":
    main()
