"""
Visualization Utilities for WSN Simulation Results.

Generates:
- Network lifetime plots (alive nodes vs epoch)
- Energy consumption plots
- Bar charts for FND/HND/LND comparison
- Network topology visualization
- Cluster boundary plots
"""

from typing import Optional, Dict, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11


def plot_alive_nodes(
    results_by_algo: Dict[str, List[dict]],
    output_path: Optional[str] = None,
    title: str = "Network Lifetime Comparison",
    show_ci: bool = True
):
    """
    Plot alive nodes vs epoch for multiple algorithms.

    Args:
        results_by_algo: Dict mapping algorithm name to list of results
        output_path: Path to save figure (None to show)
        title: Plot title
        show_ci: Show confidence interval bands
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {'auction': '#2ecc71', 'abc': '#2ecc71', 'heed': '#3498db', 'leach': '#e74c3c', 'leach-c': '#9b59b6'}

    for algo_name, results in results_by_algo.items():
        # Extract alive nodes time series
        all_alive = []
        for r in results:
            history = r.get('history', [])
            alive = [h.get('alive_nodes', 0) for h in history]
            all_alive.append(alive)

        if not all_alive:
            continue

        # Pad to same length
        max_len = max(len(a) for a in all_alive)
        padded = []
        for a in all_alive:
            padded.append(a + [0] * (max_len - len(a)))

        arr = np.array(padded)
        epochs = np.arange(1, max_len + 1)

        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)

        color = colors.get(algo_name.lower(), '#7f8c8d')
        label = algo_name.upper()

        ax.plot(epochs, mean, label=label, color=color, linewidth=2)

        if show_ci and len(results) > 1:
            ax.fill_between(epochs, mean - std, mean + std, alpha=0.2, color=color)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Number of Alive Nodes', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_energy(
    results_by_algo: Dict[str, List[dict]],
    output_path: Optional[str] = None,
    title: str = "Total Network Energy vs Epoch"
):
    """
    Plot total network energy vs epoch.

    Args:
        results_by_algo: Dict mapping algorithm name to list of results
        output_path: Path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {'auction': '#2ecc71', 'abc': '#2ecc71', 'heed': '#3498db', 'leach': '#e74c3c'}

    for algo_name, results in results_by_algo.items():
        all_energy = []
        for r in results:
            history = r.get('history', [])
            energy = [h.get('total_energy', 0) for h in history]
            all_energy.append(energy)

        if not all_energy:
            continue

        max_len = max(len(e) for e in all_energy)
        padded = []
        for e in all_energy:
            padded.append(e + [0] * (max_len - len(e)))

        arr = np.array(padded)
        epochs = np.arange(1, max_len + 1)

        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)

        color = colors.get(algo_name.lower(), '#7f8c8d')

        ax.plot(epochs, mean, label=algo_name.upper(), color=color, linewidth=2)
        if len(results) > 1:
            ax.fill_between(epochs, mean - std, mean + std, alpha=0.2, color=color)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Total Residual Energy (J)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_lifetime_comparison(
    results_by_algo: Dict[str, List[dict]],
    output_path: Optional[str] = None,
    title: str = "Network Lifetime Metrics Comparison"
):
    """
    Bar chart comparing FND, HND, LND across algorithms.

    Args:
        results_by_algo: Dict mapping algorithm name to list of results
        output_path: Path to save figure
        title: Plot title
    """
    metrics_data = {'Algorithm': [], 'FND': [], 'HND': [], 'LND': [],
                   'FND_std': [], 'HND_std': [], 'LND_std': []}

    for algo_name, results in results_by_algo.items():
        fnds = [r['fnd'] for r in results if r.get('fnd') is not None]
        hnds = [r['hnd'] for r in results if r.get('hnd') is not None]
        lnds = [r['lnd'] for r in results if r.get('lnd') is not None]

        metrics_data['Algorithm'].append(algo_name.upper())
        metrics_data['FND'].append(np.mean(fnds) if fnds else 0)
        metrics_data['HND'].append(np.mean(hnds) if hnds else 0)
        metrics_data['LND'].append(np.mean(lnds) if lnds else 0)
        metrics_data['FND_std'].append(np.std(fnds) if len(fnds) > 1 else 0)
        metrics_data['HND_std'].append(np.std(hnds) if len(hnds) > 1 else 0)
        metrics_data['LND_std'].append(np.std(lnds) if len(lnds) > 1 else 0)

    df = pd.DataFrame(metrics_data)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(df))
    width = 0.25

    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    metrics = ['FND', 'HND', 'LND']

    for i, metric in enumerate(metrics):
        bars = ax.bar(x + i * width, df[metric], width, label=metric,
                     yerr=df[f'{metric}_std'], capsize=3, color=colors[i])

    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_ylabel('Epoch', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(df['Algorithm'])
    ax.legend(fontsize=11)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_network_topology(
    nodes: list,
    clusters: list,
    bs_position: tuple = (50, 100),
    output_path: Optional[str] = None,
    title: str = "Network Topology",
    epoch: Optional[int] = None
):
    """
    Plot network topology with clusters.

    Args:
        nodes: List of Node objects
        clusters: List of Cluster objects
        bs_position: (x, y) of base station
        output_path: Path to save figure
        title: Plot title
        epoch: Epoch number (for title)
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Color palette for clusters
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(clusters))))

    # Plot cluster members and heads
    for i, cluster in enumerate(clusters):
        color = colors[i % len(colors)]

        # Plot members
        member_x = [m.x for m in cluster.members if m.is_alive]
        member_y = [m.y for m in cluster.members if m.is_alive]
        ax.scatter(member_x, member_y, c=[color], s=50, alpha=0.7, marker='o')

        # Plot CH
        if cluster.head.is_alive:
            ax.scatter(cluster.head.x, cluster.head.y, c=[color], s=200,
                      marker='*', edgecolors='black', linewidths=1.5,
                      label=f'CH {cluster.head.id}' if i < 5 else None)

            # Draw lines from members to CH
            for m in cluster.members:
                if m.is_alive:
                    ax.plot([m.x, cluster.head.x], [m.y, cluster.head.y],
                           c=color, alpha=0.3, linewidth=0.5)

    # Plot dead nodes
    dead_x = [n.x for n in nodes if not n.is_alive]
    dead_y = [n.y for n in nodes if not n.is_alive]
    if dead_x:
        ax.scatter(dead_x, dead_y, c='gray', s=30, marker='x', alpha=0.5, label='Dead')

    # Plot base station
    ax.scatter(bs_position[0], bs_position[1], c='red', s=300, marker='^',
              edgecolors='black', linewidths=2, label='Base Station', zorder=10)

    # Draw lines from CHs to BS
    for cluster in clusters:
        if cluster.head.is_alive:
            ax.plot([cluster.head.x, bs_position[0]], [cluster.head.y, bs_position[1]],
                   c='red', alpha=0.3, linewidth=1, linestyle='--')

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)

    title_str = title
    if epoch is not None:
        title_str += f' (Epoch {epoch})'
    ax.set_title(title_str, fontsize=14)

    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 110)
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=9)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_control_overhead(
    results_by_algo: Dict[str, List[dict]],
    output_path: Optional[str] = None,
    title: str = "Cumulative Control Message Overhead"
):
    """
    Plot cumulative control messages over epochs.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {'auction': '#2ecc71', 'abc': '#2ecc71', 'heed': '#3498db', 'leach': '#e74c3c'}

    for algo_name, results in results_by_algo.items():
        all_cumsum = []
        for r in results:
            history = r.get('history', [])
            ctrl = [h.get('control_messages', 0) for h in history]
            cumsum = np.cumsum(ctrl)
            all_cumsum.append(cumsum)

        if not all_cumsum:
            continue

        max_len = max(len(c) for c in all_cumsum)
        padded = []
        for c in all_cumsum:
            last_val = c[-1] if len(c) > 0 else 0
            padded.append(list(c) + [last_val] * (max_len - len(c)))

        arr = np.array(padded)
        epochs = np.arange(1, max_len + 1)

        mean = np.mean(arr, axis=0)
        color = colors.get(algo_name.lower(), '#7f8c8d')

        ax.plot(epochs, mean, label=algo_name.upper(), color=color, linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Cumulative Control Messages', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper left', fontsize=11)
    ax.set_xlim(left=0)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_energy_std(
    results_by_algo: Dict[str, List[dict]],
    output_path: Optional[str] = None,
    title: str = "Energy Balance (Standard Deviation)"
):
    """
    Plot energy standard deviation over epochs - KEY FAIRNESS METRIC.
    Lower std = better load balancing.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {'auction': '#2ecc71', 'abc': '#2ecc71', 'heed': '#3498db', 'leach': '#e74c3c'}

    for algo_name, results in results_by_algo.items():
        all_std = []
        for r in results:
            history = r.get('history', [])
            std = [h.get('energy_std', 0) for h in history]
            all_std.append(std)

        if not all_std:
            continue

        max_len = max(len(s) for s in all_std)
        padded = []
        for s in all_std:
            padded.append(s + [0] * (max_len - len(s)))

        arr = np.array(padded)
        epochs = np.arange(1, max_len + 1)

        mean = np.mean(arr, axis=0)
        color = colors.get(algo_name.lower(), '#7f8c8d')

        ax.plot(epochs, mean, label=algo_name.upper(), color=color, linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Energy Std Dev (J)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_xlim(left=0)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_throughput(
    results_by_algo: Dict[str, List[dict]],
    output_path: Optional[str] = None,
    title: str = "Cumulative Throughput (Data Packets Delivered)"
):
    """
    Plot cumulative data packets delivered to BS.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {'auction': '#2ecc71', 'abc': '#2ecc71', 'heed': '#3498db', 'leach': '#e74c3c'}

    for algo_name, results in results_by_algo.items():
        all_cumsum = []
        for r in results:
            history = r.get('history', [])
            packets = [h.get('data_packets', 0) for h in history]
            cumsum = np.cumsum(packets)
            all_cumsum.append(cumsum)

        if not all_cumsum:
            continue

        max_len = max(len(c) for c in all_cumsum)
        padded = []
        for c in all_cumsum:
            last_val = c[-1] if len(c) > 0 else 0
            padded.append(list(c) + [last_val] * (max_len - len(c)))

        arr = np.array(padded)
        epochs = np.arange(1, max_len + 1)

        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
        color = colors.get(algo_name.lower(), '#7f8c8d')

        ax.plot(epochs, mean, label=algo_name.upper(), color=color, linewidth=2)
        if len(results) > 1:
            ax.fill_between(epochs, mean - std, mean + std, alpha=0.2, color=color)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Cumulative Data Packets', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper left', fontsize=11)
    ax.set_xlim(left=0)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_num_clusters(
    results_by_algo: Dict[str, List[dict]],
    output_path: Optional[str] = None,
    title: str = "Number of Cluster Heads per Epoch"
):
    """
    Plot number of clusters over time.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {'auction': '#2ecc71', 'abc': '#2ecc71', 'heed': '#3498db', 'leach': '#e74c3c'}

    for algo_name, results in results_by_algo.items():
        all_chs = []
        for r in results:
            history = r.get('history', [])
            chs = [h.get('num_clusters', 0) for h in history]
            all_chs.append(chs)

        if not all_chs:
            continue

        max_len = max(len(c) for c in all_chs)
        padded = []
        for c in all_chs:
            padded.append(c + [0] * (max_len - len(c)))

        arr = np.array(padded)
        epochs = np.arange(1, max_len + 1)

        mean = np.mean(arr, axis=0)
        color = colors.get(algo_name.lower(), '#7f8c8d')

        ax.plot(epochs, mean, label=algo_name.upper(), color=color, linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Number of Cluster Heads', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_xlim(left=0)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def generate_all_plots(
    results_by_algo: Dict[str, List[dict]],
    output_dir: str,
    prefix: str = ""
):
    """
    Generate all standard plots.

    Args:
        results_by_algo: Dict mapping algorithm name to list of results
        output_dir: Output directory
        prefix: Filename prefix
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    p = f"{prefix}_" if prefix else ""

    # Alive nodes plot
    plot_alive_nodes(
        results_by_algo,
        output_path / f"{p}alive_nodes.png"
    )

    # Energy plot
    plot_energy(
        results_by_algo,
        output_path / f"{p}energy.png"
    )

    # Lifetime comparison
    plot_lifetime_comparison(
        results_by_algo,
        output_path / f"{p}lifetime_comparison.png"
    )

    # Control overhead
    plot_control_overhead(
        results_by_algo,
        output_path / f"{p}control_overhead.png"
    )

    # Energy balance (fairness)
    plot_energy_std(
        results_by_algo,
        output_path / f"{p}energy_std.png"
    )

    # Throughput
    plot_throughput(
        results_by_algo,
        output_path / f"{p}throughput.png"
    )

    # Number of clusters
    plot_num_clusters(
        results_by_algo,
        output_path / f"{p}num_clusters.png"
    )

    print(f"All plots saved to {output_path}")


if __name__ == "__main__":
    # Test with mock data
    print("=== Visualization Test ===\n")

    # Generate mock results
    def mock_history(initial_nodes, decay_rate, max_epochs):
        history = []
        alive = initial_nodes
        energy = initial_nodes * 2.0  # 2J per node
        for epoch in range(1, max_epochs + 1):
            if alive > 0 and np.random.random() < decay_rate:
                alive -= 1
                energy -= 0.05
            history.append({
                'epoch': epoch,
                'alive_nodes': alive,
                'total_energy': max(0, energy),
                'num_clusters': max(1, alive // 10),
                'control_messages': np.random.randint(10, 30)
            })
            if alive == 0:
                break
        return history

    mock_results = {
        'auction': [
            {'algorithm': 'auction', 'fnd': 180, 'hnd': 300, 'lnd': 450,
             'history': mock_history(50, 0.01, 500)}
            for _ in range(3)
        ],
        'heed': [
            {'algorithm': 'heed', 'fnd': 150, 'hnd': 270, 'lnd': 400,
             'history': mock_history(50, 0.012, 500)}
            for _ in range(3)
        ],
        'leach': [
            {'algorithm': 'leach', 'fnd': 120, 'hnd': 240, 'lnd': 380,
             'history': mock_history(50, 0.014, 500)}
            for _ in range(3)
        ]
    }

    # Test plots
    plot_alive_nodes(mock_results, title="Test: Alive Nodes")
    plot_lifetime_comparison(mock_results, title="Test: Lifetime Comparison")
