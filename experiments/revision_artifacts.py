#!/usr/bin/env python3
"""
VTC Revision Artifacts Generator

Generates all deliverables for VTC paper revision:
A) Checkpoint run-to-death table (3 workloads, run until LND)
B) Full 2D sweep with T_fixed=300 (12 cells)
C) Figures: winner map, phi_c heatmap, per-protocol heatmaps

Key: Same node deployment used across all protocols for each trial (fairness).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import copy
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime

from src.models.node import create_heterogeneous_nodes
from src.models.network import Network
from src.models.energy import EnergyModel
from src.algorithms.auction import AuctionClustering
from src.algorithms.heed import HEEDClustering
from src.algorithms.leach import LEACHClustering
from src.simulation import load_config

# ============================================================================
# Configuration
# ============================================================================

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

# Checkpoint workloads (run-to-death)
CHECKPOINT_WORKLOADS = [
    {'name': 'Data-heavy', 'Ldata': 32000, 's': 0.1},
    {'name': 'Middle', 'Ldata': 8000, 's': 1.0},
    {'name': 'Control-heavy', 'Ldata': 500, 's': 10.0},
]

# Full sweep grid
SWEEP_LDATA = [500, 2000, 8000, 32000]
SWEEP_S = [0.1, 1.0, 10.0]

# Protocols
PROTOCOLS = [
    ('auction', 'ABC', 'local'),
    ('heed', 'HEED', 'local'),
    ('leach', 'LEACH-L', 'local'),
    ('leach', 'LEACH', 'global'),
]

# ============================================================================
# Core simulation functions
# ============================================================================

def create_shared_network(config, seed):
    """Create network with deterministic node placement for fair comparison."""
    net_cfg = config.get('network', {})
    energy_cfg = config.get('energy', {})

    nodes = create_heterogeneous_nodes(
        n=net_cfg.get('num_nodes', 50),
        width=net_cfg.get('area_width', 100.0),
        height=net_cfg.get('area_height', 100.0),
        energy_mean=energy_cfg.get('initial_mean', 2.0),
        energy_std=energy_cfg.get('initial_std', 0.2),
        energy_min=energy_cfg.get('initial_min', 0.1),
        seed=seed
    )

    network = Network(
        nodes=nodes,
        width=net_cfg.get('area_width', 100.0),
        height=net_cfg.get('area_height', 100.0),
        bs_x=net_cfg.get('bs_x', 50.0),
        bs_y=net_cfg.get('bs_y', 100.0),
        comm_range=net_cfg.get('comm_range', 30.0)
    )

    return network


def reset_network_energy(network, seed):
    """Reset network to initial energy state (same deployment, fresh energy)."""
    energy_cfg = {'initial_mean': 2.0, 'initial_std': 0.2, 'initial_min': 0.1}
    np.random.seed(seed)

    for node in network.nodes:
        energy = np.random.normal(energy_cfg['initial_mean'], energy_cfg['initial_std'])
        energy = max(energy_cfg['initial_min'], energy)
        node.initial_energy = energy
        node.current_energy = energy
        node._is_alive = True


def create_algorithm(algo_key, network, energy_model, config, discovery_mode):
    """Create algorithm instance."""
    cfg = copy.deepcopy(config)
    cfg['control']['discovery_radius_mode'] = discovery_mode

    if algo_key == 'auction':
        return AuctionClustering(network, energy_model, cfg)
    elif algo_key == 'heed':
        return HEEDClustering(network, energy_model, cfg)
    elif algo_key == 'leach':
        return LEACHClustering(network, energy_model, cfg)
    else:
        raise ValueError(f"Unknown algorithm: {algo_key}")


def run_to_death(network, algorithm, max_epochs=10000):
    """
    Run simulation until all nodes dead (LND).

    Returns dict with:
    - alive_curve: list of alive counts per round
    - ctrl_energy_curve: list of control energy per round
    - data_energy_curve: list of data energy per round
    - fnd, hnd, lnd: milestone rounds
    - auc_star: normalized AUC over LND
    - phi_c: control energy fraction over LND
    - mean_node_lifetime: average node lifetime
    """
    algorithm.setup()

    N = len(network.nodes)
    initial_energy = network.get_total_energy()

    alive_curve = []
    ctrl_energy_curve = []

    fnd = None
    hnd = None
    lnd = None
    half_n = N // 2

    for epoch in range(1, max_epochs + 1):
        # Snapshot energy before round
        energy_before = network.get_total_energy()

        stats = algorithm.run_epoch()
        alive = stats['alive_nodes']
        ctrl_energy = stats.get('control_energy_j', 0)

        # Data energy = total spent - control energy
        energy_after = network.get_total_energy()
        total_spent = energy_before - energy_after
        data_energy = max(0, total_spent - ctrl_energy)

        alive_curve.append(alive)
        ctrl_energy_curve.append(ctrl_energy)

        # Track milestones
        if fnd is None and alive < N:
            fnd = epoch
        if hnd is None and alive <= half_n:
            hnd = epoch
        if alive == 0:
            lnd = epoch
            break

    if lnd is None:
        lnd = len(alive_curve)
        print(f"  WARNING: Network didn't die within {max_epochs} epochs")

    # Compute metrics over horizon 1..LND
    alive_sum = sum(alive_curve)
    auc_star = alive_sum / (N * lnd)
    mean_node_lifetime = alive_sum / N  # = AUC_star * LND

    total_ctrl = sum(ctrl_energy_curve)
    total_energy_spent = initial_energy - network.get_total_energy()
    phi_c = total_ctrl / (total_energy_spent + 1e-12)

    return {
        'alive_curve': alive_curve,
        'ctrl_energy_curve': ctrl_energy_curve,
        'fnd': fnd if fnd else lnd,
        'hnd': hnd if hnd else lnd,
        'lnd': lnd,
        'auc_star': auc_star,
        'phi_c': phi_c,
        'mean_node_lifetime': mean_node_lifetime,
        'N': N,
    }


def run_fixed_horizon(network, algorithm, T_fixed=300):
    """
    Run simulation for EXACTLY T_fixed rounds (for 2D sweep).

    IMPORTANT: If network dies before T_fixed (LND < T_fixed),
    we pad remaining rounds with a(t)=0 and energy=0.
    This ensures AUC_T is always computed over exactly T_fixed rounds.

    Returns dict with:
    - auc_T: AUC over fixed horizon = (1/(N*T_fixed)) * sum_{t=1..T_fixed} a(t)
    - phi_c_T: control fraction over fixed horizon
    - fnd, hnd, lnd: milestones (None if not reached within T_fixed)
    """
    algorithm.setup()

    N = len(network.nodes)
    initial_energy = network.get_total_energy()

    alive_curve = []
    ctrl_energy_curve = []
    data_energy_curve = []

    fnd = None
    hnd = None
    lnd = None
    half_n = N // 2
    network_dead = False

    for epoch in range(1, T_fixed + 1):
        if network_dead:
            # Network already dead - pad with zeros
            alive_curve.append(0)
            ctrl_energy_curve.append(0.0)
            data_energy_curve.append(0.0)
            continue

        energy_before = network.get_total_energy()

        stats = algorithm.run_epoch()
        alive = stats['alive_nodes']
        ctrl_energy = stats.get('control_energy_j', 0)

        # Data energy = total spent - control energy
        energy_after = network.get_total_energy()
        total_spent = energy_before - energy_after
        data_energy = max(0, total_spent - ctrl_energy)

        alive_curve.append(alive)
        ctrl_energy_curve.append(ctrl_energy)
        data_energy_curve.append(data_energy)

        if fnd is None and alive < N:
            fnd = epoch
        if hnd is None and alive <= half_n:
            hnd = epoch
        if lnd is None and alive == 0:
            lnd = epoch
            network_dead = True

    # Compute metrics over EXACTLY T_fixed rounds
    # AUC_T = (1/(N*T_fixed)) * sum_{t=1..T_fixed} a(t)
    alive_sum = sum(alive_curve)
    auc_T = alive_sum / (N * T_fixed)

    # phi_c_T = sum Ectrl / sum (Ectrl + Edata) over T_fixed rounds
    total_ctrl = sum(ctrl_energy_curve)
    total_data = sum(data_energy_curve)
    total_energy = total_ctrl + total_data
    phi_c_T = total_ctrl / (total_energy + 1e-12)

    return {
        'auc_T': auc_T,
        'phi_c_T': phi_c_T,
        'fnd': fnd,  # None if not reached within T_fixed
        'hnd': hnd,
        'lnd': lnd,
    }


# ============================================================================
# A) Checkpoint run-to-death experiments
# ============================================================================

def run_checkpoint_experiments(base_config, num_trials=15):
    """Run checkpoint workloads to death, with shared deployment per trial."""

    print("\n" + "="*70)
    print("A) CHECKPOINT RUN-TO-DEATH EXPERIMENTS")
    print("="*70)

    all_results = []

    for wp in CHECKPOINT_WORKLOADS:
        print(f"\n--- Workload: {wp['name']} (Ldata={wp['Ldata']}, s={wp['s']}) ---")

        # Configure workload
        config = copy.deepcopy(base_config)
        config['packets']['data_size'] = wp['Ldata']
        config['control']['bits_multiplier'] = wp['s']

        for trial in range(num_trials):
            seed = 42 + trial
            print(f"  Trial {trial+1}/{num_trials} (seed={seed})")

            # Create shared network for this trial
            network_template = create_shared_network(config, seed)
            energy_model = EnergyModel(
                e_elec=config['energy']['e_elec'],
                e_amp=config['energy']['e_amp'],
                e_da=config['energy']['e_da']
            )

            for algo_key, algo_label, discovery_mode in PROTOCOLS:
                # Reset network to initial state
                reset_network_energy(network_template, seed)

                # Create fresh algorithm instance
                algorithm = create_algorithm(
                    algo_key, network_template, energy_model, config, discovery_mode
                )

                # Run to death
                result = run_to_death(network_template, algorithm)

                # Record
                all_results.append({
                    'workload': wp['name'],
                    'Ldata': wp['Ldata'],
                    's': wp['s'],
                    'protocol': algo_label,
                    'trial': trial,
                    'seed': seed,
                    'auc_star': result['auc_star'],
                    'phi_c': result['phi_c'],
                    'fnd': result['fnd'],
                    'hnd': result['hnd'],
                    'lnd': result['lnd'],
                    'mean_node_lifetime': result['mean_node_lifetime'],
                })

    return all_results


def export_checkpoint_csv(results):
    """Export checkpoint results to CSV."""
    csv_path = ARTIFACTS_DIR / "checkpoints_metrics.csv"

    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['workload', 'Ldata', 's', 'protocol', 'trial', 'seed',
                     'auc_star', 'phi_c', 'fnd', 'hnd', 'lnd', 'mean_node_lifetime']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved: {csv_path}")
    return csv_path


def generate_checkpoint_latex_table(results):
    """Generate LaTeX table for checkpoint results."""

    # Aggregate by workload + protocol
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(list))

    for r in results:
        key = (r['workload'], r['protocol'])
        agg[key]['auc_star'].append(r['auc_star'])
        agg[key]['phi_c'].append(r['phi_c'])
        agg[key]['fnd'].append(r['fnd'])
        agg[key]['hnd'].append(r['hnd'])
        agg[key]['lnd'].append(r['lnd'])
        agg[key]['mean_node_lifetime'].append(r['mean_node_lifetime'])

    # Generate LaTeX
    latex = r"""\begin{table*}[t]
\centering
\caption{Run-to-death metrics for three representative workloads (N=50, 15 trials).
AUC$^*$ (run-to-death) $= \frac{1}{N \cdot \mathrm{LND}} \sum_{t=1}^{\mathrm{LND}} a(t)$;
$\phi_c$ = control energy fraction over $[1,\mathrm{LND}]$;
Mean node lifetime $= \frac{1}{N} \sum_{t=1}^{\mathrm{LND}} a(t) = \mathrm{AUC}^* \times \mathrm{LND}$.}
\label{tab:checkpoints}
\footnotesize
\begin{tabular}{llcccccc}
\toprule
\textbf{Workload} & \textbf{Protocol} & \textbf{AUC$^*$} & \textbf{$\phi_c$ (\%)} & \textbf{FND} & \textbf{HND} & \textbf{LND} & \textbf{Mean Lifetime} \\
\midrule
"""

    workload_order = ['Data-heavy', 'Middle', 'Control-heavy']
    protocol_order = ['ABC', 'HEED', 'LEACH-L', 'LEACH']

    for wp_name in workload_order:
        first_in_workload = True
        for proto in protocol_order:
            key = (wp_name, proto)
            if key not in agg:
                continue

            data = agg[key]
            auc_mean = np.mean(data['auc_star'])
            auc_std = np.std(data['auc_star'])
            phi_mean = np.mean(data['phi_c']) * 100
            phi_std = np.std(data['phi_c']) * 100
            fnd_mean = np.mean(data['fnd'])
            fnd_std = np.std(data['fnd'])
            hnd_mean = np.mean(data['hnd'])
            hnd_std = np.std(data['hnd'])
            lnd_mean = np.mean(data['lnd'])
            lnd_std = np.std(data['lnd'])
            mlt_mean = np.mean(data['mean_node_lifetime'])
            mlt_std = np.std(data['mean_node_lifetime'])

            wp_col = wp_name if first_in_workload else ""
            first_in_workload = False

            latex += f"{wp_col} & {proto} & "
            latex += f"{auc_mean:.3f}$\\pm${auc_std:.3f} & "
            latex += f"{phi_mean:.1f}$\\pm${phi_std:.1f} & "
            latex += f"{fnd_mean:.0f}$\\pm${fnd_std:.0f} & "
            latex += f"{hnd_mean:.0f}$\\pm${hnd_std:.0f} & "
            latex += f"{lnd_mean:.0f}$\\pm${lnd_std:.0f} & "
            latex += f"{mlt_mean:.1f}$\\pm${mlt_std:.1f} \\\\\n"

        latex += r"\midrule" + "\n"

    latex = latex.rstrip(r"\midrule" + "\n")
    latex += r"""
\bottomrule
\end{tabular}
\end{table*}
"""

    tex_path = ARTIFACTS_DIR / "table_checkpoints.tex"
    with open(tex_path, 'w') as f:
        f.write(latex)

    print(f"Saved: {tex_path}")
    return tex_path


# ============================================================================
# B) Full 2D sweep experiments
# ============================================================================

def run_sweep_experiments(base_config, num_trials=5, T_fixed=300):
    """Run full 2D sweep with fixed horizon."""

    print("\n" + "="*70)
    print("B) FULL 2D SWEEP EXPERIMENTS (T_fixed=300)")
    print("="*70)

    all_results = []
    total_cells = len(SWEEP_LDATA) * len(SWEEP_S) * len(PROTOCOLS)
    cell_count = 0

    for Ldata in SWEEP_LDATA:
        for s in SWEEP_S:
            print(f"\n--- Cell: Ldata={Ldata}, s={s} ---")

            config = copy.deepcopy(base_config)
            config['packets']['data_size'] = Ldata
            config['control']['bits_multiplier'] = s

            for trial in range(num_trials):
                seed = 42 + trial

                # Create shared network
                network_template = create_shared_network(config, seed)
                energy_model = EnergyModel(
                    e_elec=config['energy']['e_elec'],
                    e_amp=config['energy']['e_amp'],
                    e_da=config['energy']['e_da']
                )

                for algo_key, algo_label, discovery_mode in PROTOCOLS:
                    cell_count += 1

                    # Reset network
                    reset_network_energy(network_template, seed)

                    # Create algorithm
                    algorithm = create_algorithm(
                        algo_key, network_template, energy_model, config, discovery_mode
                    )

                    # Run fixed horizon
                    result = run_fixed_horizon(network_template, algorithm, T_fixed)

                    all_results.append({
                        'protocol': algo_label,
                        'Ldata': Ldata,
                        's': s,
                        'trial': trial,
                        'seed': seed,
                        'auc_T': result['auc_T'],
                        'phi_c_T': result['phi_c_T'],
                        'fnd': result['fnd'] if result['fnd'] else 'NA',
                        'hnd': result['hnd'] if result['hnd'] else 'NA',
                        'lnd': result['lnd'] if result['lnd'] else 'NA',
                    })

            print(f"  Completed {cell_count} / {total_cells * num_trials} runs")

    return all_results


def export_sweep_csv(results):
    """Export sweep results to CSV."""
    csv_path = ARTIFACTS_DIR / "sweep_metrics.csv"

    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['protocol', 'Ldata', 's', 'trial', 'seed',
                     'auc_T', 'phi_c_T', 'fnd', 'hnd', 'lnd']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved: {csv_path}")
    return csv_path


# ============================================================================
# C) Figure generation
# ============================================================================

def aggregate_sweep_results(results):
    """Aggregate sweep results by (protocol, Ldata, s) -> mean values."""
    from collections import defaultdict

    agg = defaultdict(lambda: {'auc_T': [], 'phi_c_T': []})

    for r in results:
        key = (r['protocol'], r['Ldata'], r['s'])
        agg[key]['auc_T'].append(r['auc_T'])
        agg[key]['phi_c_T'].append(r['phi_c_T'])

    # Compute means
    means = {}
    for key, data in agg.items():
        means[key] = {
            'auc_T_mean': np.mean(data['auc_T']),
            'auc_T_std': np.std(data['auc_T']),
            'phi_c_T_mean': np.mean(data['phi_c_T']),
            'phi_c_T_std': np.std(data['phi_c_T']),
        }

    return means


def generate_winner_map(sweep_means):
    """Generate Fig1: AUC_T winner map."""

    fig, ax = plt.subplots(figsize=(8, 6))

    protocols = ['ABC', 'HEED', 'LEACH-L', 'LEACH']
    proto_colors = {'ABC': '#2ecc71', 'HEED': '#3498db',
                   'LEACH-L': '#e74c3c', 'LEACH': '#9b59b6'}

    # Create grid
    n_ldata = len(SWEEP_LDATA)
    n_s = len(SWEEP_S)

    winner_grid = np.zeros((n_s, n_ldata))
    auc_grid = np.zeros((n_s, n_ldata))
    winner_names = [['' for _ in range(n_ldata)] for _ in range(n_s)]

    for i, s in enumerate(SWEEP_S):
        for j, Ldata in enumerate(SWEEP_LDATA):
            best_auc = -1
            best_proto = None

            for proto in protocols:
                key = (proto, Ldata, s)
                if key in sweep_means:
                    auc = sweep_means[key]['auc_T_mean']
                    if auc > best_auc:
                        best_auc = auc
                        best_proto = proto

            winner_names[i][j] = best_proto
            auc_grid[i][j] = best_auc
            winner_grid[i][j] = protocols.index(best_proto) if best_proto else -1

    # Create color map
    cmap = mcolors.ListedColormap([proto_colors[p] for p in protocols])

    im = ax.imshow(winner_grid, cmap=cmap, aspect='auto', vmin=0, vmax=len(protocols)-1)

    # Annotate
    for i in range(n_s):
        for j in range(n_ldata):
            text = f"{winner_names[i][j]}\n{auc_grid[i][j]:.3f}"
            ax.text(j, i, text, ha='center', va='center', fontsize=10, fontweight='bold',
                   color='white' if winner_names[i][j] in ['LEACH-L', 'LEACH'] else 'black')

    ax.set_xticks(range(n_ldata))
    ax.set_xticklabels([str(L) for L in SWEEP_LDATA])
    ax.set_yticks(range(n_s))
    ax.set_yticklabels([str(s) for s in SWEEP_S])
    ax.set_xlabel('Data packet size $L_{data}$ (bits)', fontsize=12)
    ax.set_ylabel('Control scale $s$', fontsize=12)
    ax.set_title('(a) AUC (T=300) Winner Map', fontsize=12, fontweight='bold')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=proto_colors[p], label=p) for p in protocols]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))

    plt.tight_layout()

    for ext in ['pdf', 'png']:
        path = ARTIFACTS_DIR / f"fig1_winner_map_aucT.{ext}"
        plt.savefig(path, dpi=300, bbox_inches='tight')

    plt.close()
    print(f"Saved: fig1_winner_map_aucT.pdf/.png")


def generate_phi_c_heatmap_heed(sweep_means):
    """Generate Fig1b: phi_c heatmap for HEED."""

    fig, ax = plt.subplots(figsize=(7, 5))

    n_ldata = len(SWEEP_LDATA)
    n_s = len(SWEEP_S)

    phi_c_grid = np.zeros((n_s, n_ldata))

    for i, s in enumerate(SWEEP_S):
        for j, Ldata in enumerate(SWEEP_LDATA):
            key = ('HEED', Ldata, s)
            if key in sweep_means:
                phi_c_grid[i][j] = sweep_means[key]['phi_c_T_mean']

    im = ax.imshow(phi_c_grid, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

    # Annotate
    for i in range(n_s):
        for j in range(n_ldata):
            val = phi_c_grid[i][j]
            color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                   fontsize=11, fontweight='bold', color=color)

    ax.set_xticks(range(n_ldata))
    ax.set_xticklabels([str(L) for L in SWEEP_LDATA])
    ax.set_yticks(range(n_s))
    ax.set_yticklabels([str(s) for s in SWEEP_S])
    ax.set_xlabel('Data packet size $L_{data}$ (bits)', fontsize=12)
    ax.set_ylabel('Control scale $s$', fontsize=12)
    ax.set_title('(b) HEED $\\phi_c$ (T=300)', fontsize=12, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('$\\phi_c$', fontsize=11)

    plt.tight_layout()

    for ext in ['pdf', 'png']:
        path = ARTIFACTS_DIR / f"fig1b_phi_c_heed.{ext}"
        plt.savefig(path, dpi=300, bbox_inches='tight')

    plt.close()
    print(f"Saved: fig1b_phi_c_heed.pdf/.png")


def generate_per_protocol_heatmaps(sweep_means):
    """Generate Fig2: Per-protocol AUC_T heatmaps."""

    protocols = ['ABC', 'HEED', 'LEACH-L', 'LEACH']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    n_ldata = len(SWEEP_LDATA)
    n_s = len(SWEEP_S)

    for idx, proto in enumerate(protocols):
        ax = axes[idx]

        auc_grid = np.zeros((n_s, n_ldata))

        for i, s in enumerate(SWEEP_S):
            for j, Ldata in enumerate(SWEEP_LDATA):
                key = (proto, Ldata, s)
                if key in sweep_means:
                    auc_grid[i][j] = sweep_means[key]['auc_T_mean']

        im = ax.imshow(auc_grid, cmap='viridis', aspect='auto', vmin=0, vmax=1)

        # Annotate
        for i in range(n_s):
            for j in range(n_ldata):
                val = auc_grid[i][j]
                color = 'white' if val < 0.6 else 'black'
                ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                       fontsize=10, fontweight='bold', color=color)

        ax.set_xticks(range(n_ldata))
        ax.set_xticklabels([str(L) for L in SWEEP_LDATA])
        ax.set_yticks(range(n_s))
        ax.set_yticklabels([str(s) for s in SWEEP_S])
        ax.set_xlabel('$L_{data}$ (bits)')
        ax.set_ylabel('Control scale $s$')
        ax.set_title(f'{proto}', fontsize=12, fontweight='bold')

    # Shared colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('AUC (T=300)', fontsize=11)

    fig.suptitle('Per-Protocol AUC (T=300) Heatmaps', fontsize=14, fontweight='bold', y=0.98)

    for ext in ['pdf', 'png']:
        path = ARTIFACTS_DIR / f"fig2_aucT_heatmaps.{ext}"
        plt.savefig(path, dpi=300, bbox_inches='tight')

    plt.close()
    print(f"Saved: fig2_aucT_heatmaps.pdf/.png")


def generate_combined_figure(sweep_means):
    """Generate combined Fig1: winner map + phi_c heatmap side by side."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    protocols = ['ABC', 'HEED', 'LEACH-L', 'LEACH']
    proto_colors = {'ABC': '#2ecc71', 'HEED': '#3498db',
                   'LEACH-L': '#e74c3c', 'LEACH': '#9b59b6'}

    n_ldata = len(SWEEP_LDATA)
    n_s = len(SWEEP_S)

    # Panel (a): Winner map
    winner_grid = np.zeros((n_s, n_ldata))
    auc_grid = np.zeros((n_s, n_ldata))
    winner_names = [['' for _ in range(n_ldata)] for _ in range(n_s)]

    for i, s in enumerate(SWEEP_S):
        for j, Ldata in enumerate(SWEEP_LDATA):
            best_auc = -1
            best_proto = None

            for proto in protocols:
                key = (proto, Ldata, s)
                if key in sweep_means:
                    auc = sweep_means[key]['auc_T_mean']
                    if auc > best_auc:
                        best_auc = auc
                        best_proto = proto

            winner_names[i][j] = best_proto
            auc_grid[i][j] = best_auc
            winner_grid[i][j] = protocols.index(best_proto) if best_proto else -1

    cmap = mcolors.ListedColormap([proto_colors[p] for p in protocols])
    im1 = ax1.imshow(winner_grid, cmap=cmap, aspect='auto', vmin=0, vmax=len(protocols)-1)

    for i in range(n_s):
        for j in range(n_ldata):
            text = f"{winner_names[i][j]}\n{auc_grid[i][j]:.2f}"
            ax1.text(j, i, text, ha='center', va='center', fontsize=9, fontweight='bold',
                    color='white' if winner_names[i][j] in ['LEACH-L', 'LEACH'] else 'black')

    ax1.set_xticks(range(n_ldata))
    ax1.set_xticklabels([str(L) for L in SWEEP_LDATA])
    ax1.set_yticks(range(n_s))
    ax1.set_yticklabels([str(s) for s in SWEEP_S])
    ax1.set_xlabel('$L_{data}$ (bits)', fontsize=11)
    ax1.set_ylabel('Control scale $s$', fontsize=11)
    ax1.set_title('(a) AUC (T=300) Winner Map', fontsize=12, fontweight='bold')

    # Panel (b): phi_c for HEED
    phi_c_grid = np.zeros((n_s, n_ldata))

    for i, s in enumerate(SWEEP_S):
        for j, Ldata in enumerate(SWEEP_LDATA):
            key = ('HEED', Ldata, s)
            if key in sweep_means:
                phi_c_grid[i][j] = sweep_means[key]['phi_c_T_mean']

    im2 = ax2.imshow(phi_c_grid, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

    for i in range(n_s):
        for j in range(n_ldata):
            val = phi_c_grid[i][j]
            color = 'white' if val > 0.5 else 'black'
            ax2.text(j, i, f"{val:.2f}", ha='center', va='center',
                    fontsize=10, fontweight='bold', color=color)

    ax2.set_xticks(range(n_ldata))
    ax2.set_xticklabels([str(L) for L in SWEEP_LDATA])
    ax2.set_yticks(range(n_s))
    ax2.set_yticklabels([str(s) for s in SWEEP_S])
    ax2.set_xlabel('$L_{data}$ (bits)', fontsize=11)
    ax2.set_ylabel('Control scale $s$', fontsize=11)
    ax2.set_title('(b) HEED $\\phi_c$ (T=300)', fontsize=12, fontweight='bold')

    cbar = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar.set_label('$\\phi_c$', fontsize=10)

    plt.tight_layout()

    for ext in ['pdf', 'png']:
        path = ARTIFACTS_DIR / f"fig1_combined.{ext}"
        plt.savefig(path, dpi=300, bbox_inches='tight')

    plt.close()
    print(f"Saved: fig1_combined.pdf/.png")


# ============================================================================
# Optional: N=100 robustness
# ============================================================================

def run_robustness_N100(base_config, num_trials=10):
    """Run checkpoint workloads with N=100 for robustness check."""

    print("\n" + "="*70)
    print("OPTIONAL: N=100 ROBUSTNESS CHECK")
    print("="*70)

    config = copy.deepcopy(base_config)
    config['network']['num_nodes'] = 100

    all_results = []

    for wp in CHECKPOINT_WORKLOADS:
        print(f"\n--- Workload: {wp['name']} (N=100) ---")

        config['packets']['data_size'] = wp['Ldata']
        config['control']['bits_multiplier'] = wp['s']

        for trial in range(num_trials):
            seed = 42 + trial

            network_template = create_shared_network(config, seed)
            energy_model = EnergyModel(
                e_elec=config['energy']['e_elec'],
                e_amp=config['energy']['e_amp'],
                e_da=config['energy']['e_da']
            )

            for algo_key, algo_label, discovery_mode in PROTOCOLS:
                reset_network_energy(network_template, seed)
                algorithm = create_algorithm(
                    algo_key, network_template, energy_model, config, discovery_mode
                )
                result = run_to_death(network_template, algorithm)

                all_results.append({
                    'workload': wp['name'],
                    'protocol': algo_label,
                    'trial': trial,
                    'auc_star': result['auc_star'],
                    'phi_c': result['phi_c'],
                    'fnd': result['fnd'],
                    'hnd': result['hnd'],
                    'lnd': result['lnd'],
                })

    # Generate LaTeX table
    generate_robustness_latex_table(all_results)

    return all_results


def generate_checkpoint_bar_chart(results):
    """Generate bar chart for AUC* checkpoint results (3 workloads)."""
    from collections import defaultdict

    # Aggregate by workload + protocol
    agg = defaultdict(lambda: defaultdict(list))
    for r in results:
        key = (r['workload'], r['protocol'])
        agg[key]['auc_star'].append(r['auc_star'])
        agg[key]['phi_c'].append(r['phi_c'])
        agg[key]['lnd'].append(r['lnd'])

    workloads = ['Data-heavy', 'Middle', 'Control-heavy']
    protocols = ['ABC', 'HEED', 'LEACH-L', 'LEACH']
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for idx, wp in enumerate(workloads):
        ax = axes[idx]
        x = np.arange(len(protocols))
        width = 0.6

        auc_means = []
        auc_stds = []
        phi_c_vals = []
        lnd_vals = []

        for proto in protocols:
            key = (wp, proto)
            if key in agg:
                auc_means.append(np.mean(agg[key]['auc_star']))
                auc_stds.append(np.std(agg[key]['auc_star']))
                phi_c_vals.append(np.mean(agg[key]['phi_c']))
                lnd_vals.append(np.mean(agg[key]['lnd']))
            else:
                auc_means.append(0)
                auc_stds.append(0)
                phi_c_vals.append(0)
                lnd_vals.append(0)

        bars = ax.bar(x, auc_means, width, yerr=auc_stds, capsize=3,
                      color=colors, edgecolor='black', linewidth=0.5)

        # Annotate with phi_c and LND
        for i, (bar, phi_c, lnd) in enumerate(zip(bars, phi_c_vals, lnd_vals)):
            height = bar.get_height()
            ax.annotate(f'$\\phi_c$={phi_c*100:.0f}%\nLND={lnd:.0f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3),
                       textcoords='offset points',
                       ha='center', va='bottom', fontsize=7)

        ax.set_xlabel('Protocol')
        ax.set_ylabel('AUC*')
        ax.set_title(f'{wp}', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(protocols, rotation=15, ha='right')
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('AUC* (Run-to-Death) for Checkpoint Workloads (N=50)', fontsize=12, fontweight='bold')
    plt.tight_layout()

    for ext in ['pdf', 'png']:
        path = ARTIFACTS_DIR / f"fig_checkpoint_aucstar.{ext}"
        plt.savefig(path, dpi=300, bbox_inches='tight')

    plt.close()
    print(f"Saved: fig_checkpoint_aucstar.pdf/.png")


def generate_robustness_latex_table(results):
    """Generate LaTeX table for N=100 robustness results."""
    from collections import defaultdict

    agg = defaultdict(lambda: defaultdict(list))
    for r in results:
        key = (r['workload'], r['protocol'])
        for metric in ['auc_star', 'phi_c', 'fnd', 'hnd', 'lnd']:
            agg[key][metric].append(r[metric])

    latex = r"""\begin{table}[t]
\centering
\caption{Robustness check with N=100 nodes (10 trials).}
\label{tab:robustness-n100}
\footnotesize
\begin{tabular}{llccccc}
\toprule
\textbf{Workload} & \textbf{Protocol} & \textbf{AUC$^*$} & \textbf{$\phi_c$} & \textbf{FND} & \textbf{HND} & \textbf{LND} \\
\midrule
"""

    for wp_name in ['Data-heavy', 'Middle', 'Control-heavy']:
        first = True
        for proto in ['ABC', 'HEED', 'LEACH-L', 'LEACH']:
            key = (wp_name, proto)
            if key not in agg:
                continue

            data = agg[key]
            wp_col = wp_name if first else ""
            first = False

            latex += f"{wp_col} & {proto} & "
            latex += f"{np.mean(data['auc_star']):.3f} & "
            latex += f"{np.mean(data['phi_c'])*100:.1f}\\% & "
            latex += f"{np.mean(data['fnd']):.0f} & "
            latex += f"{np.mean(data['hnd']):.0f} & "
            latex += f"{np.mean(data['lnd']):.0f} \\\\\n"
        latex += r"\midrule" + "\n"

    latex = latex.rstrip(r"\midrule" + "\n")
    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""

    tex_path = ARTIFACTS_DIR / "table_robustness_N100.tex"
    with open(tex_path, 'w') as f:
        f.write(latex)

    print(f"Saved: {tex_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("="*70)
    print("VTC REVISION ARTIFACTS GENERATOR")
    print("="*70)
    print(f"Output directory: {ARTIFACTS_DIR.absolute()}")

    # Load base config
    base_config = load_config("config/default.yaml")

    # A) Checkpoint experiments (run-to-death)
    checkpoint_results = run_checkpoint_experiments(base_config, num_trials=15)
    export_checkpoint_csv(checkpoint_results)
    generate_checkpoint_latex_table(checkpoint_results)

    # B) Full 2D sweep (T_fixed=300)
    sweep_results = run_sweep_experiments(base_config, num_trials=5, T_fixed=300)
    export_sweep_csv(sweep_results)

    # C) Generate figures
    print("\n" + "="*70)
    print("C) GENERATING FIGURES")
    print("="*70)

    sweep_means = aggregate_sweep_results(sweep_results)
    generate_winner_map(sweep_means)
    generate_phi_c_heatmap_heed(sweep_means)
    generate_per_protocol_heatmaps(sweep_means)
    generate_combined_figure(sweep_means)

    # Optional: N=100 robustness
    run_robustness_N100(base_config, num_trials=10)

    print("\n" + "="*70)
    print("ALL ARTIFACTS GENERATED SUCCESSFULLY")
    print("="*70)
    print(f"\nFiles in {ARTIFACTS_DIR}:")
    for f in sorted(ARTIFACTS_DIR.iterdir()):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
