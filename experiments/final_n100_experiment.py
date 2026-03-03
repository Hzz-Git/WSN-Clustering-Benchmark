#!/usr/bin/env python3
"""
Final N=100 Experiment for VTC Revision
Deliverable 1: 2D Sweep (T=300) - Winner Map + phi_c heatmaps
Deliverable 2: Run-to-death table (3 workloads × 4 protocols)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict

# IEEE-friendly typography (Arial, >=8pt, single/double-column ready)
plt.rcParams.update({
    "font.family": "Arial",
    "font.weight": "normal",
    "font.size": 8,
    "axes.titlesize": 8,
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

N_NODES = 100
NUM_TRIALS = 10

SWEEP_LDATA = [500, 2000, 8000, 32000]
SWEEP_S = [0.1, 1.0, 10.0]

CHECKPOINT_WORKLOADS = [
    {'name': 'Data-heavy', 'Ldata': 32000, 's': 0.1},
    {'name': 'Middle', 'Ldata': 8000, 's': 1.0},
    {'name': 'Control-heavy', 'Ldata': 500, 's': 10.0},
]

PROTOCOLS = [
    ('auction', 'ABC', 'local'),
    ('heed', 'HEED', 'local'),
    ('leach', 'LEACH-L', 'local'),
    ('leach', 'LEACH', 'global'),
]

# ============================================================================
# Core functions
# ============================================================================

def create_shared_network(config, seed):
    """Create network with deterministic placement."""
    net_cfg = config.get('network', {})
    energy_cfg = config.get('energy', {})

    nodes = create_heterogeneous_nodes(
        n=net_cfg.get('num_nodes', N_NODES),
        width=net_cfg.get('area_width', 100.0),
        height=net_cfg.get('area_height', 100.0),
        energy_mean=energy_cfg.get('initial_mean', 2.0),
        energy_std=energy_cfg.get('initial_std', 0.2),
        energy_min=energy_cfg.get('initial_min', 0.1),
        seed=seed
    )

    return Network(
        nodes=nodes,
        width=net_cfg.get('area_width', 100.0),
        height=net_cfg.get('area_height', 100.0),
        bs_x=net_cfg.get('bs_x', 50.0),
        bs_y=net_cfg.get('bs_y', 100.0),
        comm_range=net_cfg.get('comm_range', 30.0)
    )


def reset_network_energy(network, seed):
    """Reset network to initial energy state."""
    np.random.seed(seed)
    for node in network.nodes:
        energy = np.random.normal(2.0, 0.2)
        energy = max(0.1, energy)
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


def run_fixed_horizon(network, algorithm, T_fixed=300):
    """Run for exactly T_fixed rounds (pad zeros if network dies early)."""
    algorithm.setup()
    N = len(network.nodes)
    initial_energy = network.get_total_energy()

    alive_curve = []
    ctrl_energy_curve = []
    data_energy_curve = []
    network_dead = False

    for epoch in range(1, T_fixed + 1):
        if network_dead:
            alive_curve.append(0)
            ctrl_energy_curve.append(0.0)
            data_energy_curve.append(0.0)
            continue

        energy_before = network.get_total_energy()
        stats = algorithm.run_epoch()
        alive = stats['alive_nodes']
        ctrl_energy = stats.get('control_energy_j', 0)
        energy_after = network.get_total_energy()
        data_energy = max(0, (energy_before - energy_after) - ctrl_energy)

        alive_curve.append(alive)
        ctrl_energy_curve.append(ctrl_energy)
        data_energy_curve.append(data_energy)

        if alive == 0:
            network_dead = True

    auc_T = sum(alive_curve) / (N * T_fixed)
    total_ctrl = sum(ctrl_energy_curve)
    total_data = sum(data_energy_curve)
    phi_c_T = total_ctrl / (total_ctrl + total_data + 1e-12)

    return {'auc_T': auc_T, 'phi_c_T': phi_c_T}


def run_to_death(network, algorithm, max_epochs=15000):
    """Run until all nodes dead."""
    algorithm.setup()
    N = len(network.nodes)
    initial_energy = network.get_total_energy()

    alive_curve = []
    ctrl_energy_curve = []
    fnd, hnd, lnd = None, None, None
    half_n = N // 2

    for epoch in range(1, max_epochs + 1):
        energy_before = network.get_total_energy()
        stats = algorithm.run_epoch()
        alive = stats['alive_nodes']
        ctrl_energy = stats.get('control_energy_j', 0)

        alive_curve.append(alive)
        ctrl_energy_curve.append(ctrl_energy)

        if fnd is None and alive < N:
            fnd = epoch
        if hnd is None and alive <= half_n:
            hnd = epoch
        if alive == 0:
            lnd = epoch
            break

    if lnd is None:
        lnd = len(alive_curve)
        print(f"    WARNING: Network didn't die within {max_epochs} epochs")

    alive_sum = sum(alive_curve)
    auc_star = alive_sum / (N * lnd)
    mean_lifetime = alive_sum / N

    total_ctrl = sum(ctrl_energy_curve)
    total_spent = initial_energy - network.get_total_energy()
    phi_c = total_ctrl / (total_spent + 1e-12)

    return {
        'fnd': fnd if fnd else lnd,
        'hnd': hnd if hnd else lnd,
        'lnd': lnd,
        'phi_c': phi_c,
        'auc_star': auc_star,
        'mean_lifetime': mean_lifetime,
    }


# ============================================================================
# Deliverable 1: 2D Sweep
# ============================================================================

def run_2d_sweep(base_config):
    """Run 2D sweep with T=300."""
    print("\n" + "="*60)
    print("DELIVERABLE 1: 2D SWEEP (T=300, N=100)")
    print("="*60)

    results = defaultdict(lambda: {'auc_T': [], 'phi_c_T': []})

    total = len(SWEEP_LDATA) * len(SWEEP_S) * len(PROTOCOLS) * NUM_TRIALS
    count = 0

    for Ldata in SWEEP_LDATA:
        for s in SWEEP_S:
            config = copy.deepcopy(base_config)
            config['network']['num_nodes'] = N_NODES
            config['packets']['data_size'] = Ldata
            config['control']['bits_multiplier'] = s

            for trial in range(NUM_TRIALS):
                seed = 42 + trial
                network = create_shared_network(config, seed)
                energy_model = EnergyModel(
                    e_elec=config['energy']['e_elec'],
                    e_amp=config['energy']['e_amp'],
                    e_da=config['energy']['e_da']
                )

                for algo_key, algo_label, discovery_mode in PROTOCOLS:
                    count += 1
                    reset_network_energy(network, seed)
                    algorithm = create_algorithm(algo_key, network, energy_model, config, discovery_mode)
                    r = run_fixed_horizon(network, algorithm, T_fixed=300)

                    key = (algo_label, Ldata, s)
                    results[key]['auc_T'].append(r['auc_T'])
                    results[key]['phi_c_T'].append(r['phi_c_T'])

            print(f"  Cell (Ldata={Ldata}, s={s}) done [{count}/{total}]")

    # Aggregate means
    means = {}
    for key, data in results.items():
        means[key] = {
            'auc_T': np.mean(data['auc_T']),
            'phi_c_T': np.mean(data['phi_c_T']),
        }

    return means


def generate_sweep_figures(means):
    """Generate winner map and phi_c heatmaps."""
    protocols = ['ABC', 'HEED', 'LEACH-L', 'LEACH']
    proto_colors = {'ABC': '#2ecc71', 'HEED': '#3498db', 'LEACH-L': '#e74c3c', 'LEACH': '#9b59b6'}

    n_ldata = len(SWEEP_LDATA)
    n_s = len(SWEEP_S)

    # Figure 1: Winner Map + HEED phi_c
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel (a): Winner Map
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
    ax1.imshow(winner_grid, cmap=cmap, aspect='auto', vmin=0, vmax=len(protocols)-1)

    for i in range(n_s):
        for j in range(n_ldata):
            text = f"{winner_names[i][j]}\n{auc_grid[i][j]:.2f}"
            color = 'white' if winner_names[i][j] in ['LEACH-L', 'LEACH'] else 'black'
            ax1.text(j, i, text, ha='center', va='center', fontsize=10, fontweight='bold', color=color)

    ax1.set_xticks(range(n_ldata))
    ax1.set_xticklabels([str(L) for L in SWEEP_LDATA])
    ax1.set_yticks(range(n_s))
    ax1.set_yticklabels([str(s) for s in SWEEP_S])
    ax1.set_xlabel('$L_{data}$ (bits)', fontsize=11)
    ax1.set_ylabel('Control scale $s$', fontsize=11)
    ax1.set_title('(a) AUC (T=300) Winner Map', fontsize=12, fontweight='bold')

    # Panel (b): HEED phi_c
    phi_c_grid = np.zeros((n_s, n_ldata))
    for i, s in enumerate(SWEEP_S):
        for j, Ldata in enumerate(SWEEP_LDATA):
            key = ('HEED', Ldata, s)
            if key in means:
                phi_c_grid[i][j] = means[key]['phi_c_T']

    im2 = ax2.imshow(phi_c_grid, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    for i in range(n_s):
        for j in range(n_ldata):
            val = phi_c_grid[i][j]
            color = 'white' if val > 0.5 else 'black'
            ax2.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=10, fontweight='bold', color=color)

    ax2.set_xticks(range(n_ldata))
    ax2.set_xticklabels([str(L) for L in SWEEP_LDATA])
    ax2.set_yticks(range(n_s))
    ax2.set_yticklabels([str(s) for s in SWEEP_S])
    ax2.set_xlabel('$L_{data}$ (bits)', fontsize=11)
    ax2.set_ylabel('Control scale $s$', fontsize=11)
    ax2.set_title('(b) HEED $\\phi_c$ (T=300)', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=ax2, shrink=0.8, label='$\\phi_c$')

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(ARTIFACTS_DIR / f"fig_sweep_N100.{ext}", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: fig_sweep_N100.pdf/.png")

    # Figure 2: Per-protocol phi_c heatmaps (compact, double-column width)
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.4))
    axes = np.atleast_1d(axes)

    for idx, proto in enumerate(protocols):
        ax = axes[idx]
        phi_grid = np.zeros((n_s, n_ldata))
        for i, s in enumerate(SWEEP_S):
            for j, Ldata in enumerate(SWEEP_LDATA):
                key = (proto, Ldata, s)
                if key in means:
                    phi_grid[i][j] = means[key]['phi_c_T']

        im = ax.imshow(phi_grid, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        for i in range(n_s):
            for j in range(n_ldata):
                val = phi_grid[i][j]
                color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=8, fontweight='normal', color=color)

        ax.set_xticks(range(n_ldata))
        ax.set_xticklabels([str(L) for L in SWEEP_LDATA])
        ax.tick_params(axis='x', pad=1)
        ax.set_yticks(range(n_s))
        ax.set_yticklabels([str(s) for s in SWEEP_S])
        ax.set_xlabel('')
        if idx == 0:
            ax.set_ylabel('Control scale s')
        else:
            ax.set_ylabel('')
        ax.set_title(f'{proto}')

    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.34, wspace=0.25)
    fig.text(0.5, 0.18, 'L_data (bits)', ha='center', va='center')
    cbar_ax = fig.add_axes([0.25, 0.05, 0.50, 0.05])
    fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label='phi_c (T=300)')

    for ext in ['pdf', 'png']:
        plt.savefig(ARTIFACTS_DIR / f"fig_phi_c_all_N100.{ext}", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: fig_phi_c_all_N100.pdf/.png")


# ============================================================================
# Deliverable 3: 2D Sweep Run-to-Death (AUC* and MNL winner maps)
# ============================================================================

def run_2d_sweep_rtd(base_config, max_epochs=6000):
    """Run 2D sweep to death for AUC* and MNL."""
    print("\n" + "="*60)
    print("DELIVERABLE 3: 2D SWEEP RUN-TO-DEATH (N=100)")
    print("="*60)

    results = defaultdict(lambda: {'auc_star': [], 'mean_lifetime': [], 'phi_c': [], 'lnd': []})

    total = len(SWEEP_LDATA) * len(SWEEP_S) * len(PROTOCOLS) * NUM_TRIALS
    count = 0

    for Ldata in SWEEP_LDATA:
        for s in SWEEP_S:
            config = copy.deepcopy(base_config)
            config['network']['num_nodes'] = N_NODES
            config['packets']['data_size'] = Ldata
            config['control']['bits_multiplier'] = s

            for trial in range(NUM_TRIALS):
                seed = 42 + trial
                network = create_shared_network(config, seed)
                energy_model = EnergyModel(
                    e_elec=config['energy']['e_elec'],
                    e_amp=config['energy']['e_amp'],
                    e_da=config['energy']['e_da']
                )

                for algo_key, algo_label, discovery_mode in PROTOCOLS:
                    count += 1
                    reset_network_energy(network, seed)
                    algorithm = create_algorithm(algo_key, network, energy_model, config, discovery_mode)
                    r = run_to_death(network, algorithm, max_epochs=max_epochs)

                    key = (algo_label, Ldata, s)
                    results[key]['auc_star'].append(r['auc_star'])
                    results[key]['mean_lifetime'].append(r['mean_lifetime'])
                    results[key]['phi_c'].append(r['phi_c'])
                    results[key]['lnd'].append(r['lnd'])

            print(f"  Cell (Ldata={Ldata}, s={s}) done [{count}/{total}]")

    # Aggregate means
    means = {}
    for key, data in results.items():
        means[key] = {
            'auc_star': np.mean(data['auc_star']),
            'mean_lifetime': np.mean(data['mean_lifetime']),
            'phi_c': np.mean(data['phi_c']),
            'lnd': np.mean(data['lnd']),
        }

    return means


def generate_sweep_figures_rtd(means):
    """Generate AUC* and MNL winner maps (same style as AUC_T)."""
    protocols = ['ABC', 'HEED', 'LEACH-L', 'LEACH']
    proto_colors = {'ABC': '#2ecc71', 'HEED': '#3498db', 'LEACH-L': '#e74c3c', 'LEACH': '#9b59b6'}

    n_ldata = len(SWEEP_LDATA)
    n_s = len(SWEEP_S)

    # Figure: AUC* Winner Map + MNL Winner Map
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel (a): AUC* Winner Map
    winner_grid = np.zeros((n_s, n_ldata))
    auc_grid = np.zeros((n_s, n_ldata))
    winner_names = [['' for _ in range(n_ldata)] for _ in range(n_s)]

    for i, s in enumerate(SWEEP_S):
        for j, Ldata in enumerate(SWEEP_LDATA):
            best_auc, best_proto = -1, None
            for proto in protocols:
                key = (proto, Ldata, s)
                if key in means:
                    auc = means[key]['auc_star']
                    if auc > best_auc:
                        best_auc = auc
                        best_proto = proto
            winner_names[i][j] = best_proto
            auc_grid[i][j] = best_auc
            winner_grid[i][j] = protocols.index(best_proto) if best_proto else -1

    cmap = mcolors.ListedColormap([proto_colors[p] for p in protocols])
    ax1.imshow(winner_grid, cmap=cmap, aspect='auto', vmin=0, vmax=len(protocols)-1)

    for i in range(n_s):
        for j in range(n_ldata):
            text = f"{winner_names[i][j]}\n{auc_grid[i][j]:.2f}"
            color = 'white' if winner_names[i][j] in ['LEACH-L', 'LEACH'] else 'black'
            ax1.text(j, i, text, ha='center', va='center', fontsize=10, fontweight='bold', color=color)

    ax1.set_xticks(range(n_ldata))
    ax1.set_xticklabels([str(L) for L in SWEEP_LDATA])
    ax1.set_yticks(range(n_s))
    ax1.set_yticklabels([str(s) for s in SWEEP_S])
    ax1.set_xlabel('$L_{data}$ (bits)', fontsize=11)
    ax1.set_ylabel('Control scale $s$', fontsize=11)
    ax1.set_title('(a) AUC* (Run-to-Death) Winner Map', fontsize=12, fontweight='bold')

    # Panel (b): MNL Winner Map
    winner_grid2 = np.zeros((n_s, n_ldata))
    mnl_grid = np.zeros((n_s, n_ldata))
    winner_names2 = [['' for _ in range(n_ldata)] for _ in range(n_s)]

    for i, s in enumerate(SWEEP_S):
        for j, Ldata in enumerate(SWEEP_LDATA):
            best_mnl, best_proto = -1, None
            for proto in protocols:
                key = (proto, Ldata, s)
                if key in means:
                    mnl = means[key]['mean_lifetime']
                    if mnl > best_mnl:
                        best_mnl = mnl
                        best_proto = proto
            winner_names2[i][j] = best_proto
            mnl_grid[i][j] = best_mnl
            winner_grid2[i][j] = protocols.index(best_proto) if best_proto else -1

    ax2.imshow(winner_grid2, cmap=cmap, aspect='auto', vmin=0, vmax=len(protocols)-1)

    for i in range(n_s):
        for j in range(n_ldata):
            text = f"{winner_names2[i][j]}\n{mnl_grid[i][j]:.0f}"
            color = 'white' if winner_names2[i][j] in ['LEACH-L', 'LEACH'] else 'black'
            ax2.text(j, i, text, ha='center', va='center', fontsize=10, fontweight='bold', color=color)

    ax2.set_xticks(range(n_ldata))
    ax2.set_xticklabels([str(L) for L in SWEEP_LDATA])
    ax2.set_yticks(range(n_s))
    ax2.set_yticklabels([str(s) for s in SWEEP_S])
    ax2.set_xlabel('$L_{data}$ (bits)', fontsize=11)
    ax2.set_ylabel('Control scale $s$', fontsize=11)
    ax2.set_title('(b) MNL (Run-to-Death) Winner Map', fontsize=12, fontweight='bold')

    # Add legend for protocol colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=proto_colors[p], label=p) for p in protocols]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10,
               frameon=True, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for legend
    for ext in ['pdf', 'png']:
        plt.savefig(ARTIFACTS_DIR / f"fig_sweep_rtd_N100.{ext}", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: fig_sweep_rtd_N100.pdf/.png")


# ============================================================================
# Deliverable 2: Run-to-death Table
# ============================================================================

def run_checkpoint_experiments(base_config):
    """Run 3 workloads × 4 protocols to death."""
    print("\n" + "="*60)
    print("DELIVERABLE 2: RUN-TO-DEATH TABLE (N=100)")
    print("="*60)

    results = []

    for wp in CHECKPOINT_WORKLOADS:
        print(f"\n--- {wp['name']} (Ldata={wp['Ldata']}, s={wp['s']}) ---")

        config = copy.deepcopy(base_config)
        config['network']['num_nodes'] = N_NODES
        config['packets']['data_size'] = wp['Ldata']
        config['control']['bits_multiplier'] = wp['s']

        for algo_key, algo_label, discovery_mode in PROTOCOLS:
            metrics = {'fnd': [], 'hnd': [], 'lnd': [], 'phi_c': [], 'auc_star': [], 'mean_lifetime': []}

            for trial in range(NUM_TRIALS):
                seed = 42 + trial
                network = create_shared_network(config, seed)
                energy_model = EnergyModel(
                    e_elec=config['energy']['e_elec'],
                    e_amp=config['energy']['e_amp'],
                    e_da=config['energy']['e_da']
                )
                reset_network_energy(network, seed)
                algorithm = create_algorithm(algo_key, network, energy_model, config, discovery_mode)
                r = run_to_death(network, algorithm)

                for k in metrics:
                    metrics[k].append(r[k])

            results.append({
                'workload': wp['name'],
                'protocol': algo_label,
                'fnd': np.mean(metrics['fnd']),
                'hnd': np.mean(metrics['hnd']),
                'lnd': np.mean(metrics['lnd']),
                'phi_c': np.mean(metrics['phi_c']),
                'auc_star': np.mean(metrics['auc_star']),
                'mean_lifetime': np.mean(metrics['mean_lifetime']),
            })
            print(f"  {algo_label}: LND={np.mean(metrics['lnd']):.0f}, AUC*={np.mean(metrics['auc_star']):.3f}")

    return results


def print_checkpoint_table(results):
    """Print formatted table."""
    print("\n" + "="*100)
    print("RUN-TO-DEATH TABLE (N=100, {} trials)".format(NUM_TRIALS))
    print("="*100)
    print(f"{'Workload':<15} | {'Protocol':<10} | {'FND':>6} | {'HND':>6} | {'LND':>6} | {'phi_c':>7} | {'AUC*':>6} | {'MeanLife':>10}")
    print("-"*100)

    for r in results:
        print(f"{r['workload']:<15} | {r['protocol']:<10} | {r['fnd']:>6.0f} | {r['hnd']:>6.0f} | {r['lnd']:>6.0f} | {r['phi_c']*100:>6.1f}% | {r['auc_star']:>6.3f} | {r['mean_lifetime']:>10.1f}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("="*60)
    print(f"FINAL EXPERIMENT: N={N_NODES}, Trials={NUM_TRIALS}")
    print("="*60)

    base_config = load_config("config/default.yaml")

    # Deliverable 1: 2D Sweep (T=300)
    sweep_means = run_2d_sweep(base_config)
    generate_sweep_figures(sweep_means)

    # Deliverable 2: Run-to-death table (3 workloads)
    checkpoint_results = run_checkpoint_experiments(base_config)
    print_checkpoint_table(checkpoint_results)

    # Deliverable 3: 2D Sweep Run-to-Death (AUC* and MNL winner maps)
    sweep_rtd_means = run_2d_sweep_rtd(base_config, max_epochs=6000)
    generate_sweep_figures_rtd(sweep_rtd_means)

    print("\n" + "="*60)
    print("DONE! Artifacts in:", ARTIFACTS_DIR.absolute())
    print("="*60)


if __name__ == "__main__":
    main()
