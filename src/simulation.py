"""
Main Simulation Engine for WSN Clustering Experiments.

Handles:
- Network initialization
- Algorithm execution
- Metrics collection
- Multi-trial experiments
"""

from typing import Type, Optional
import numpy as np
import yaml
from pathlib import Path

from .models.node import create_heterogeneous_nodes, create_random_nodes
from .models.network import Network
from .models.energy import EnergyModel
from .algorithms.base import ClusteringAlgorithm
from .algorithms.auction import AuctionClustering
from .algorithms.heed import HEEDClustering
from .algorithms.leach import LEACHClustering, LEACHCentralized


# Algorithm registry
ALGORITHMS = {
    'auction': AuctionClustering,
    'abc': AuctionClustering,
    'heed': HEEDClustering,
    'leach': LEACHClustering,
    'leach-c': LEACHCentralized,
}


class Simulation:
    """
    Main simulation engine for WSN clustering experiments.
    """

    def __init__(self, config: dict, seed: Optional[int] = None):
        """
        Initialize simulation.

        Args:
            config: Configuration dictionary
            seed: Random seed for reproducibility
        """
        self.config = config
        self.seed = seed

        # Extract config sections
        self.net_cfg = config.get('network', {})
        self.energy_cfg = config.get('energy', {})
        self.sim_cfg = config.get('simulation', {})

        # Simulation parameters
        self.max_epochs = self.sim_cfg.get('max_epochs', 500)

        # State
        self.network: Optional[Network] = None
        self.algorithm: Optional[ClusteringAlgorithm] = None
        self.history: list[dict] = []

    def setup_network(self, seed: Optional[int] = None) -> Network:
        """
        Create and initialize the network.

        Args:
            seed: Random seed for node placement

        Returns:
            Initialized Network object
        """
        s = seed if seed is not None else self.seed

        # Create nodes with heterogeneous energy
        nodes = create_heterogeneous_nodes(
            n=self.net_cfg.get('num_nodes', 50),
            width=self.net_cfg.get('area_width', 100.0),
            height=self.net_cfg.get('area_height', 100.0),
            energy_mean=self.energy_cfg.get('initial_mean', 2.0),
            energy_std=self.energy_cfg.get('initial_std', 0.2),
            energy_min=self.energy_cfg.get('initial_min', 0.1),
            seed=s
        )

        # Create network
        network = Network(
            nodes=nodes,
            width=self.net_cfg.get('area_width', 100.0),
            height=self.net_cfg.get('area_height', 100.0),
            bs_x=self.net_cfg.get('bs_x', 50.0),
            bs_y=self.net_cfg.get('bs_y', 100.0),
            comm_range=self.net_cfg.get('comm_range', 30.0)
        )

        return network

    def setup_energy_model(self) -> EnergyModel:
        """Create energy model from config."""
        return EnergyModel(
            e_elec=self.energy_cfg.get('e_elec', 50e-9),
            e_fs=self.energy_cfg.get('e_fs', 10e-12),
            e_mp=self.energy_cfg.get('e_mp', 0.0013e-12),
            e_da=self.energy_cfg.get('e_da', 5e-9),
            d_crossover=self.energy_cfg.get('d_crossover', 87.0),
            use_two_mode=True,
        )

    def setup_algorithm(
        self,
        algorithm_name: str,
        network: Network,
        energy_model: EnergyModel
    ) -> ClusteringAlgorithm:
        """
        Create algorithm instance.

        Args:
            algorithm_name: Name of algorithm ('auction', 'heed', 'leach')
            network: Network object
            energy_model: Energy model

        Returns:
            Algorithm instance
        """
        if algorithm_name.lower() not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algorithm_name}. "
                           f"Available: {list(ALGORITHMS.keys())}")

        algo_class = ALGORITHMS[algorithm_name.lower()]
        return algo_class(network, energy_model, self.config)

    def run(
        self,
        algorithm_name: str,
        seed: Optional[int] = None,
        verbose: bool = False
    ) -> dict:
        """
        Run a single simulation trial.

        Args:
            algorithm_name: Name of algorithm to use
            seed: Random seed
            verbose: Print progress

        Returns:
            Dict with simulation results
        """
        # Setup
        s = seed if seed is not None else self.seed
        np.random.seed(s)

        self.network = self.setup_network(s)
        energy_model = self.setup_energy_model()
        self.algorithm = self.setup_algorithm(algorithm_name, self.network, energy_model)
        self.algorithm.setup()

        self.history = []

        # Track key metrics
        fnd_epoch = None  # First Node Dead
        hnd_epoch = None  # Half Nodes Dead
        lnd_epoch = None  # Last Node Dead
        initial_nodes = self.network.count_alive()
        half_nodes = initial_nodes // 2

        # Initial state
        initial_energy = self.network.get_total_energy()

        if verbose:
            print(f"Starting {self.algorithm.name} simulation")
            print(f"  Nodes: {initial_nodes}, Initial Energy: {initial_energy:.3f}J")

        # Main simulation loop
        for epoch in range(1, self.max_epochs + 1):
            # Run one epoch
            stats = self.algorithm.run_epoch()
            self.history.append(stats)

            alive = stats['alive_nodes']

            # Track FND
            if fnd_epoch is None and alive < initial_nodes:
                fnd_epoch = epoch

            # Track HND
            if hnd_epoch is None and alive <= half_nodes:
                hnd_epoch = epoch

            # Track LND
            if alive == 0:
                lnd_epoch = epoch
                break

            if verbose and epoch % 50 == 0:
                print(f"  Epoch {epoch}: {alive} alive, E={stats['total_energy']:.3f}J")

        # Final LND if network still alive at max_epochs
        if lnd_epoch is None:
            lnd_epoch = self.max_epochs

        # Calculate AUC (Alive-node area-under-curve)
        # AUC = (1/(N*T)) * sum(alive(t) for t in 1..T)
        # If simulation ends early, treat remaining epochs as alive=0
        alive_sum = sum(h['alive_nodes'] for h in self.history)
        # Add zeros for epochs not simulated (if network died early)
        # Division is always by max_epochs (T) to avoid censoring bias
        auc = alive_sum / (initial_nodes * self.max_epochs)

        # Compile results
        results = {
            'algorithm': self.algorithm.name,
            'seed': s,
            'initial_nodes': initial_nodes,
            'initial_energy': initial_energy,
            'fnd': fnd_epoch,
            'hnd': hnd_epoch,
            'lnd': lnd_epoch,
            'auc': auc,  # Alive-node area-under-curve [0,1]
            'final_alive': self.network.count_alive(),
            'final_energy': self.network.get_total_energy(),
            'total_epochs': len(self.history),
            'history': self.history,
        }

        if verbose:
            print(f"  Finished: FND={fnd_epoch}, HND={hnd_epoch}, LND={lnd_epoch}")

        return results

    def run_trials(
        self,
        algorithm_name: str,
        num_trials: int,
        base_seed: int = 42,
        verbose: bool = False
    ) -> list[dict]:
        """
        Run multiple simulation trials.

        Args:
            algorithm_name: Name of algorithm
            num_trials: Number of trials to run
            base_seed: Base random seed (incremented for each trial)
            verbose: Print progress

        Returns:
            List of results for each trial
        """
        results = []

        for trial in range(num_trials):
            seed = base_seed + trial
            if verbose:
                print(f"\nTrial {trial + 1}/{num_trials} (seed={seed})")

            result = self.run(algorithm_name, seed=seed, verbose=verbose)
            result['trial'] = trial
            results.append(result)

        return results


def load_config(config_path: str = "config/default.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_comparison(
    config: dict,
    algorithms: list[str],
    num_trials: int = 15,
    verbose: bool = True
) -> dict:
    """
    Run comparison of multiple algorithms.

    Args:
        config: Configuration dictionary
        algorithms: List of algorithm names
        num_trials: Number of trials per algorithm
        verbose: Print progress

    Returns:
        Dict mapping algorithm name to list of results
    """
    all_results = {}

    for algo_name in algorithms:
        print(f"\n{'='*50}")
        print(f"Running {algo_name.upper()}")
        print('='*50)

        sim = Simulation(config)
        results = sim.run_trials(algo_name, num_trials, verbose=verbose)
        all_results[algo_name] = results

        # Summary statistics
        fnds = [r['fnd'] for r in results if r['fnd'] is not None]
        hnds = [r['hnd'] for r in results if r['hnd'] is not None]
        lnds = [r['lnd'] for r in results]
        aucs = [r['auc'] for r in results]

        print(f"\n{algo_name} Summary ({num_trials} trials):")
        print(f"  FND: {np.mean(fnds):.1f} +/- {np.std(fnds):.1f}")
        print(f"  HND: {np.mean(hnds):.1f} +/- {np.std(hnds):.1f}")
        print(f"  LND: {np.mean(lnds):.1f} +/- {np.std(lnds):.1f}")
        print(f"  AUC: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WSN Clustering Simulation")
    parser.add_argument('--algorithm', '-a', default='auction',
                       choices=list(ALGORITHMS.keys()),
                       help='Algorithm to run')
    parser.add_argument('--trials', '-t', type=int, default=1,
                       help='Number of trials')
    parser.add_argument('--seed', '-s', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--config', '-c', default='config/default.yaml',
                       help='Config file path')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Run simulation
    sim = Simulation(config)

    if args.trials == 1:
        result = sim.run(args.algorithm, seed=args.seed, verbose=True)
        print(f"\nResults:")
        print(f"  FND: {result['fnd']}")
        print(f"  HND: {result['hnd']}")
        print(f"  LND: {result['lnd']}")
        print(f"  AUC: {result['auc']:.4f}")
    else:
        results = sim.run_trials(args.algorithm, args.trials, args.seed, args.verbose)

        fnds = [r['fnd'] for r in results if r['fnd'] is not None]
        hnds = [r['hnd'] for r in results if r['hnd'] is not None]
        lnds = [r['lnd'] for r in results]
        aucs = [r['auc'] for r in results]

        print(f"\nSummary ({args.trials} trials):")
        print(f"  FND: {np.mean(fnds):.1f} +/- {np.std(fnds):.1f}")
        print(f"  HND: {np.mean(hnds):.1f} +/- {np.std(hnds):.1f}")
        print(f"  LND: {np.mean(lnds):.1f} +/- {np.std(lnds):.1f}")
        print(f"  AUC: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
