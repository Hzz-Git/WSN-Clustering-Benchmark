#!/usr/bin/env python3
"""
Generate separate AUC* and MNL winner maps (split from fig_sweep_rtd).
Style matches fig_aucT_winner_N100.
Saves RTD data to CSV to avoid re-running.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from collections import defaultdict

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

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

SWEEP_LDATA = [500, 2000, 8000, 32000]
SWEEP_S = [0.1, 1.0, 10.0]
N_NODES = 100
NUM_TRIALS = 10

PROTOCOLS = [
    ('auction', 'ABC', 'local'),
    ('heed', 'HEED', 'local'),
    ('leach', 'LEACH-L', 'local'),
    ('leach', 'LEACH', 'global'),
]


def load_rtd_data():
    """Load run-to-death sweep data from CSV."""
    csv_path = ARTIFACTS_DIR / "sweep_rtd_metrics.csv"
    if csv_path.exists():
        print(f"Loading existing RTD data from {csv_path}", flush=True)
        df = pd.read_csv(csv_path)
        grouped = df.groupby(['protocol', 'Ldata', 's']).agg({
            'auc_star': 'mean',
            'mean_lifetime': 'mean',
            'phi_c': 'mean'
        }).reset_index()
        means = {}
        for _, row in grouped.iterrows():
            key = (row['protocol'], int(row['Ldata']), row['s'])
            means[key] = {
                'auc_star': row['auc_star'],
                'mean_lifetime': row['mean_lifetime'],
                'phi_c': row['phi_c']
            }
        return means
    else:
        print("sweep_rtd_metrics.csv not found. Will run RTD sweep.", flush=True)
        return None


def generate_single_winner_map(means, metric, metric_label, filename, fmt='.2f'):
    """Generate a single winner map for the given metric."""
    protocols = ['ABC', 'HEED', 'LEACH-L', 'LEACH']
    proto_colors = {'ABC': '#2ecc71', 'HEED': '#3498db', 'LEACH-L': '#e74c3c', 'LEACH': '#9b59b6'}

    n_ldata = len(SWEEP_LDATA)
    n_s = len(SWEEP_S)

    # Compact size for 3-up figure row (fits 0.32*textwidth at 2-column width)
    fig, ax = plt.subplots(1, 1, figsize=(2.6, 1.9))

    winner_grid = np.zeros((n_s, n_ldata))
    value_grid = np.zeros((n_s, n_ldata))
    winner_names = [['' for _ in range(n_ldata)] for _ in range(n_s)]

    for i, s in enumerate(SWEEP_S):
        for j, Ldata in enumerate(SWEEP_LDATA):
            best_val, best_proto = -1, None
            for proto in protocols:
                key = (proto, Ldata, s)
                if key in means:
                    val = means[key][metric]
                    if val > best_val:
                        best_val = val
                        best_proto = proto
            winner_names[i][j] = best_proto
            value_grid[i][j] = best_val
            winner_grid[i][j] = protocols.index(best_proto) if best_proto else -1

    cmap = mcolors.ListedColormap([proto_colors[p] for p in protocols])
    ax.imshow(winner_grid, cmap=cmap, aspect='auto', vmin=0, vmax=len(protocols)-1)

    for i in range(n_s):
        for j in range(n_ldata):
            text = f"{value_grid[i][j]:{fmt}}"
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
        plt.savefig(ARTIFACTS_DIR / f"{filename}.{ext}", dpi=300)
    plt.close()
    print(f"Saved: {filename}.pdf/.png")


def run_rtd_sweep():
    """Run the RTD sweep and save data to CSV."""
    from src.models.node import create_heterogeneous_nodes
    from src.models.network import Network
    from src.models.energy import EnergyModel
    from src.algorithms.auction import AuctionClustering
    from src.algorithms.heed import HEEDClustering
    from src.algorithms.leach import LEACHClustering
    from src.simulation import load_config

    print("="*60, flush=True)
    print("RUNNING RTD SWEEP (N=100, 10 trials, max 6000 rounds)", flush=True)
    print("="*60, flush=True)

    base_config = load_config("config/default.yaml")
    results = defaultdict(lambda: {'auc_star': [], 'mean_lifetime': [], 'phi_c': [], 'lnd': []})
    all_rows = []  # For CSV saving

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
                np.random.seed(seed)

                # Create network
                net_cfg = config.get('network', {})
                energy_cfg = config.get('energy', {})
                nodes = create_heterogeneous_nodes(
                    n=N_NODES,
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
                energy_model = EnergyModel(
                    e_elec=config['energy']['e_elec'],
                    e_amp=config['energy']['e_amp'],
                    e_da=config['energy']['e_da']
                )

                for algo_key, algo_label, discovery_mode in PROTOCOLS:
                    count += 1

                    # Reset network energy
                    np.random.seed(seed)
                    for node in network.nodes:
                        energy = np.random.normal(2.0, 0.2)
                        energy = max(0.1, energy)
                        node.initial_energy = energy
                        node.current_energy = energy
                        node._is_alive = True

                    # Create algorithm
                    cfg = copy.deepcopy(config)
                    cfg['control']['discovery_radius_mode'] = discovery_mode
                    if algo_key == 'auction':
                        algorithm = AuctionClustering(network, energy_model, cfg)
                    elif algo_key == 'heed':
                        algorithm = HEEDClustering(network, energy_model, cfg)
                    elif algo_key == 'leach':
                        algorithm = LEACHClustering(network, energy_model, cfg)

                    # Run to death
                    algorithm.setup()
                    N = len(network.nodes)
                    initial_energy = network.get_total_energy()
                    alive_curve = []
                    ctrl_energy_curve = []
                    fnd, hnd, lnd = None, None, None
                    half_n = N // 2

                    for epoch in range(1, 6001):
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

                    alive_sum = sum(alive_curve)
                    auc_star = alive_sum / (N * lnd)
                    mean_lifetime = alive_sum / N
                    total_ctrl = sum(ctrl_energy_curve)
                    total_spent = initial_energy - network.get_total_energy()
                    phi_c = total_ctrl / (total_spent + 1e-12)

                    key = (algo_label, Ldata, s)
                    results[key]['auc_star'].append(auc_star)
                    results[key]['mean_lifetime'].append(mean_lifetime)
                    results[key]['phi_c'].append(phi_c)
                    results[key]['lnd'].append(lnd)

                    # Save per-trial data
                    all_rows.append({
                        'protocol': algo_label, 'Ldata': Ldata, 's': s, 'trial': trial,
                        'auc_star': auc_star, 'mean_lifetime': mean_lifetime,
                        'phi_c': phi_c, 'lnd': lnd, 'fnd': fnd, 'hnd': hnd
                    })

            print(f"  Cell (Ldata={Ldata}, s={s}) done [{count}/{total}]", flush=True)

    # Save all trial data to CSV
    pd.DataFrame(all_rows).to_csv(ARTIFACTS_DIR / "sweep_rtd_metrics.csv", index=False)
    print(f"Saved: sweep_rtd_metrics.csv", flush=True)

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


if __name__ == "__main__":
    means = load_rtd_data()

    if means is None:
        means = run_rtd_sweep()

    if means:
        generate_single_winner_map(means, 'auc_star', '$\\mathrm{AUC}^*$', 'fig_aucstar_winner_N100', fmt='.2f')
        generate_single_winner_map(means, 'mean_lifetime', 'MNL', 'fig_mnl_winner_N100', fmt='.0f')
        print("Done!", flush=True)
