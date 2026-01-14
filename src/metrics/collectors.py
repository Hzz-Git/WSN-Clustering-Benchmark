"""
Metrics Collection for WSN Simulations.

Collects and processes:
- Network lifetime metrics (FND, HND, LND)
- Energy metrics (total, variance, distribution)
- Cluster metrics (sizes, count, orphans)
- Control overhead
"""

from typing import Optional
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from pathlib import Path
import json


@dataclass
class SimulationMetrics:
    """Container for all metrics from a simulation run."""
    algorithm: str
    seed: int
    trial: int = 0

    # Lifetime metrics
    fnd: Optional[int] = None  # First Node Dead
    hnd: Optional[int] = None  # Half Nodes Dead
    lnd: Optional[int] = None  # Last Node Dead

    # Initial state
    initial_nodes: int = 0
    initial_energy: float = 0.0

    # Final state
    final_alive: int = 0
    final_energy: float = 0.0
    total_epochs: int = 0

    # Time series data
    alive_per_epoch: list = field(default_factory=list)
    energy_per_epoch: list = field(default_factory=list)
    clusters_per_epoch: list = field(default_factory=list)
    control_msgs_per_epoch: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'algorithm': self.algorithm,
            'seed': self.seed,
            'trial': self.trial,
            'fnd': self.fnd,
            'hnd': self.hnd,
            'lnd': self.lnd,
            'initial_nodes': self.initial_nodes,
            'initial_energy': self.initial_energy,
            'final_alive': self.final_alive,
            'final_energy': self.final_energy,
            'total_epochs': self.total_epochs,
        }


class MetricsCollector:
    """
    Collects metrics from simulation history.
    """

    def __init__(self):
        self.results: list[SimulationMetrics] = []

    def collect_from_result(self, result: dict) -> SimulationMetrics:
        """
        Extract metrics from a simulation result dict.

        Args:
            result: Result dict from Simulation.run()

        Returns:
            SimulationMetrics object
        """
        history = result.get('history', [])

        metrics = SimulationMetrics(
            algorithm=result.get('algorithm', 'unknown'),
            seed=result.get('seed', 0),
            trial=result.get('trial', 0),
            fnd=result.get('fnd'),
            hnd=result.get('hnd'),
            lnd=result.get('lnd'),
            initial_nodes=result.get('initial_nodes', 0),
            initial_energy=result.get('initial_energy', 0.0),
            final_alive=result.get('final_alive', 0),
            final_energy=result.get('final_energy', 0.0),
            total_epochs=result.get('total_epochs', len(history)),
            alive_per_epoch=[h.get('alive_nodes', 0) for h in history],
            energy_per_epoch=[h.get('total_energy', 0.0) for h in history],
            clusters_per_epoch=[h.get('num_clusters', 0) for h in history],
            control_msgs_per_epoch=[h.get('control_messages', 0) for h in history],
        )

        self.results.append(metrics)
        return metrics

    def collect_batch(self, results: list[dict]) -> list[SimulationMetrics]:
        """Collect metrics from multiple results."""
        return [self.collect_from_result(r) for r in results]

    def get_summary_stats(self, algorithm: Optional[str] = None) -> dict:
        """
        Calculate summary statistics.

        Args:
            algorithm: Filter by algorithm name (None for all)

        Returns:
            Dict with summary statistics
        """
        filtered = self.results
        if algorithm:
            # Match case-insensitively and handle partial matches
            algo_lower = algorithm.lower()
            filtered = [r for r in self.results
                       if algo_lower in r.algorithm.lower() or r.algorithm.lower() in algo_lower]

        if not filtered:
            return {'fnd_mean': 0, 'fnd_std': 0, 'hnd_mean': 0, 'hnd_std': 0,
                    'lnd_mean': 0, 'lnd_std': 0}

        fnds = [r.fnd for r in filtered if r.fnd is not None]
        hnds = [r.hnd for r in filtered if r.hnd is not None]
        lnds = [r.lnd for r in filtered if r.lnd is not None]

        total_ctrl = [sum(r.control_msgs_per_epoch) for r in filtered]

        return {
            'algorithm': algorithm or 'all',
            'num_trials': len(filtered),
            'fnd_mean': np.mean(fnds) if fnds else None,
            'fnd_std': np.std(fnds) if fnds else None,
            'hnd_mean': np.mean(hnds) if hnds else None,
            'hnd_std': np.std(hnds) if hnds else None,
            'lnd_mean': np.mean(lnds) if lnds else None,
            'lnd_std': np.std(lnds) if lnds else None,
            'ctrl_msgs_mean': np.mean(total_ctrl) if total_ctrl else None,
            'ctrl_msgs_std': np.std(total_ctrl) if total_ctrl else None,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        return pd.DataFrame([r.to_dict() for r in self.results])

    def get_timeseries_dataframe(self, metric: str = 'alive') -> pd.DataFrame:
        """
        Get time series data as DataFrame.

        Args:
            metric: 'alive', 'energy', 'clusters', or 'control_msgs'

        Returns:
            DataFrame with epochs as rows, trials as columns
        """
        metric_map = {
            'alive': 'alive_per_epoch',
            'energy': 'energy_per_epoch',
            'clusters': 'clusters_per_epoch',
            'control_msgs': 'control_msgs_per_epoch',
        }

        attr = metric_map.get(metric, 'alive_per_epoch')

        data = {}
        for r in self.results:
            col_name = f"{r.algorithm}_trial{r.trial}"
            data[col_name] = getattr(r, attr)

        # Pad to same length
        max_len = max(len(v) for v in data.values()) if data else 0
        for k, v in data.items():
            data[k] = v + [np.nan] * (max_len - len(v))

        df = pd.DataFrame(data)
        df.index.name = 'epoch'
        return df


def save_results(
    results: list[dict],
    output_dir: str,
    prefix: str = "results"
):
    """
    Save simulation results to files.

    Args:
        results: List of result dicts
        output_dir: Output directory path
        prefix: Filename prefix
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save summary CSV
    collector = MetricsCollector()
    collector.collect_batch(results)
    df = collector.to_dataframe()
    df.to_csv(output_path / f"{prefix}_summary.csv", index=False)

    # Save time series
    for metric in ['alive', 'energy']:
        ts_df = collector.get_timeseries_dataframe(metric)
        ts_df.to_csv(output_path / f"{prefix}_{metric}_timeseries.csv")

    # Save full results as JSON (for debugging)
    # Strip history for smaller file
    summary_results = []
    for r in results:
        summary = {k: v for k, v in r.items() if k != 'history'}
        summary_results.append(summary)

    with open(output_path / f"{prefix}_results.json", 'w') as f:
        json.dump(summary_results, f, indent=2, default=str)

    print(f"Results saved to {output_path}")


def load_results(input_dir: str, prefix: str = "results") -> pd.DataFrame:
    """Load results from CSV."""
    input_path = Path(input_dir)
    return pd.read_csv(input_path / f"{prefix}_summary.csv")


def calculate_confidence_interval(data: list, confidence: float = 0.95) -> tuple:
    """
    Calculate confidence interval for data.

    Args:
        data: List of values
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    from scipy import stats

    n = len(data)
    if n < 2:
        mean = data[0] if data else 0
        return mean, mean, mean

    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)

    return mean, mean - h, mean + h


if __name__ == "__main__":
    # Test metrics collection
    print("=== Metrics Collector Test ===\n")

    # Mock results
    mock_results = [
        {
            'algorithm': 'ABC',
            'seed': 42,
            'trial': 0,
            'fnd': 150,
            'hnd': 280,
            'lnd': 450,
            'initial_nodes': 50,
            'initial_energy': 100.0,
            'final_alive': 0,
            'final_energy': 0.0,
            'total_epochs': 450,
            'history': [
                {'epoch': i, 'alive_nodes': max(0, 50 - i // 10), 'total_energy': max(0, 100 - i * 0.2), 'num_clusters': 5, 'control_messages': 20}
                for i in range(450)
            ]
        },
        {
            'algorithm': 'HEED',
            'seed': 42,
            'trial': 0,
            'fnd': 120,
            'hnd': 250,
            'lnd': 400,
            'initial_nodes': 50,
            'initial_energy': 100.0,
            'final_alive': 0,
            'final_energy': 0.0,
            'total_epochs': 400,
            'history': [
                {'epoch': i, 'alive_nodes': max(0, 50 - i // 8), 'total_energy': max(0, 100 - i * 0.25), 'num_clusters': 6, 'control_messages': 35}
                for i in range(400)
            ]
        }
    ]

    collector = MetricsCollector()
    collector.collect_batch(mock_results)

    # Test summary
    print("Summary for ABC:")
    print(collector.get_summary_stats('ABC'))

    print("\nSummary for HEED:")
    print(collector.get_summary_stats('HEED'))

    # Test DataFrame
    print("\nDataFrame:")
    print(collector.to_dataframe())
