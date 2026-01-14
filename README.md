# MAS-WSN: Multi-Agent Auction-Based Clustering for Wireless Sensor Networks

## Project Overview

This project implements and evaluates a **single-round sealed-bid auction mechanism** for cluster head election in wireless sensor networks, framed as a multi-agent system problem.

---

## TODO Plan

### Phase 1: Core Simulation Framework
- [ ] **1.1** Set up project structure
- [ ] **1.2** Implement `Node` class (agent representation)
  - Properties: id, position (x, y), initial_energy, current_energy, role (CH/CM/SLEEP), cluster_id
  - Methods: transmit(), receive(), aggregate(), update_energy()
- [ ] **1.3** Implement `Network` class
  - Node deployment (random uniform)
  - Distance calculations
  - Neighbor discovery
- [ ] **1.4** Implement `EnergyModel` class
  - First-order radio model
  - Transmit energy: E_tx = E_elec * k + E_amp * k * d^n
  - Receive energy: E_rx = E_elec * k
  - Aggregation energy: E_da * k
- [ ] **1.5** Implement `BaseStation` class
  - Location (outside sensing field)
  - Data reception from CHs

### Phase 2: Algorithm Implementations

#### 2.1 Proposed: Auction-Based Clustering (ABC)
- [ ] **2.1.1** Implement bid calculation function
  ```
  bid_i = f(residual_energy, avg_distance_to_neighbors, fairness_credit)
  ```
- [ ] **2.1.2** Implement sealed-bid auction mechanism
  - Broadcast phase
  - Winner determination (local)
  - CH announcement
- [ ] **2.1.3** Implement fairness credit system
  - Credit accumulation for non-CH rounds
  - Credit spending when becoming CH
- [ ] **2.1.4** Implement first-round seed CH selection
  - Grid-based or distributed leader election
- [ ] **2.1.5** Implement sleep scheduling
  - Duty cycle management
  - Wake-up coordination

#### 2.2 Baseline: LEACH
- [ ] **2.2.1** Implement probabilistic CH election
  ```
  T(n) = p / (1 - p * (r mod 1/p))  if n ∈ G, else 0
  ```
- [ ] **2.2.2** Implement cluster formation (join nearest CH)
- [ ] **2.2.3** Implement TDMA scheduling within clusters

#### 2.3 Baseline: HEED
- [ ] **2.3.1** Implement hybrid CH election
  - Primary: residual energy
  - Secondary: intra-cluster communication cost
- [ ] **2.3.2** Implement iterative clustering (multiple rounds)
- [ ] **2.3.3** Implement tentative/final CH states

#### 2.4 Baseline: LEACH-C (Centralized)
- [ ] **2.4.1** Implement centralized optimal CH selection
  - Simulated annealing or k-means based
- [ ] **2.4.2** BS broadcasts cluster assignments

#### 2.5 Baseline: Recent Method (Optional)
- [ ] **2.5.1** Implement simplified ANN-based or PSO-based method from recent literature

### Phase 3: Metrics & Data Collection
- [ ] **3.1** Implement metric collectors
  - `alive_nodes(round)` — number of alive nodes per round
  - `first_node_dead` — round when first node dies (FND)
  - `half_nodes_dead` — round when 50% nodes die (HND)
  - `last_node_dead` — round when last node dies (LND)
  - `residual_energy_std(round)` — energy balance metric
  - `total_data_delivered` — throughput
  - `control_overhead` — messages per round
  - `avg_cluster_size` and `cluster_size_variance`
  - `orphan_nodes` — nodes that couldn't join any CH
- [ ] **3.2** Implement data logging (CSV/JSON)
- [ ] **3.3** Implement checkpoint/resume for long simulations

### Phase 4: Experimental Runs

#### Experiment 1: Network Lifetime Comparison
- [ ] **4.1.1** Configure: N=100, Area=100x100m, E_init=0.5J
- [ ] **4.1.2** Run all 4-5 algorithms, 30 trials each
- [ ] **4.1.3** Collect: FND, HND, LND, alive_nodes curve

#### Experiment 2: Energy Balance Analysis
- [ ] **4.2.1** Same config as Exp 1
- [ ] **4.2.2** Collect: residual_energy_std over time
- [ ] **4.2.3** Visualize: box plots of final energy distribution

#### Experiment 3: Scalability
- [ ] **4.3.1** Vary N = {50, 100, 200, 500}
- [ ] **4.3.2** Fixed density: scale area proportionally
- [ ] **4.3.3** Collect: FND, overhead, runtime

#### Experiment 4: Network Density
- [ ] **4.4.1** Fixed N=100, vary area = {50x50, 100x100, 200x200}
- [ ] **4.4.2** Collect: cluster quality metrics, orphan nodes

#### Experiment 5: Parameter Sensitivity (Proposed Method)
- [ ] **4.5.1** Vary fairness credit weight
- [ ] **4.5.2** Vary energy weight in bid function
- [ ] **4.5.3** Analyze impact on lifetime and balance

### Phase 5: Analysis & Visualization
- [ ] **5.1** Statistical analysis script
  - Mean, std, 95% confidence intervals
  - Paired t-test or Wilcoxon signed-rank test
  - Effect size (Cohen's d)
- [ ] **5.2** Generate plots
  - Line plot: alive nodes vs. rounds (all methods)
  - Bar chart: FND/HND/LND comparison with error bars
  - Box plot: energy distribution at round 1000, 2000, etc.
  - Heatmap: node death locations
- [ ] **5.3** Generate summary tables (LaTeX-ready)

---

## Simulation Parameters

| Parameter | Symbol | Default Value |
|-----------|--------|---------------|
| Number of nodes | N | 100 |
| Network area | A | 100m × 100m |
| BS location | (x_bs, y_bs) | (50, 175) |
| Initial energy | E_init | 0.5 J |
| Electronics energy | E_elec | 50 nJ/bit |
| Amplifier (free space) | ε_fs | 10 pJ/bit/m² |
| Amplifier (multipath) | ε_mp | 0.0013 pJ/bit/m⁴ |
| Distance threshold | d_0 | 87 m |
| Data packet size | k_data | 4000 bits |
| Control packet size | k_ctrl | 200 bits |
| Aggregation energy | E_DA | 5 nJ/bit |
| Desired CH percentage | p | 5% |
| Max rounds | R_max | 5000 |
| Trials per experiment | T | 30 |

---

## Project Structure

```
MAS-WSN-Experiment/
├── README.md
├── requirements.txt
├── config/
│   └── default.yaml          # Simulation parameters
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── node.py           # Node/Agent class
│   │   ├── network.py        # Network class
│   │   ├── energy.py         # Energy model
│   │   └── base_station.py   # Base station
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── base.py           # Abstract base class
│   │   ├── auction.py        # Proposed auction-based
│   │   ├── leach.py          # LEACH baseline
│   │   ├── heed.py           # HEED baseline
│   │   └── leach_c.py        # LEACH-C baseline
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── collectors.py     # Metric collection
│   └── utils/
│       ├── __init__.py
│       ├── visualization.py  # Plotting functions
│       └── stats.py          # Statistical analysis
├── experiments/
│   ├── run_lifetime.py       # Experiment 1
│   ├── run_scalability.py    # Experiment 3
│   └── run_sensitivity.py    # Experiment 5
├── results/
│   ├── data/                 # Raw CSV results
│   └── figures/              # Generated plots
└── tests/
    └── test_energy_model.py  # Unit tests
```

---

## Getting Started

```bash
# Clone and setup
cd MAS-WSN-Experiment
pip install -r requirements.txt

# Run single simulation
python -m src.main --algorithm auction --trials 1

# Run full experiment
python experiments/run_lifetime.py --trials 30

# Generate figures
python -m src.utils.visualization --input results/data/ --output results/figures/
```

---

## Dependencies

```
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
seaborn>=0.12
pandas>=2.0
pyyaml>=6.0
tqdm>=4.65
```

---

## Timeline Estimate

| Phase | Estimated Time |
|-------|----------------|
| Phase 1: Core Framework | 2-3 days |
| Phase 2: Algorithms | 3-4 days |
| Phase 3: Metrics | 1 day |
| Phase 4: Experiments | 2-3 days (includes runtime) |
| Phase 5: Analysis | 1-2 days |
| **Total** | **~10-12 days** |

---

## Key Design Decisions to Make

Before implementation, clarify these in your paper:

1. **Bid Function**: What exact formula?
   - Option A: `bid = α * E_residual + β * (1/avg_dist) + γ * credit`
   - Option B: `bid = E_residual * credit / avg_dist`
   
2. **Fairness Credit Update**:
   - Additive: `credit += 1` each non-CH round
   - Multiplicative: `credit *= 1.1`
   - Cap or no cap?

3. **Winner Determination Scope**:
   - Global (highest bid wins)?
   - Local (highest in k-hop neighborhood)?
   
4. **Communication Model**:
   - Synchronous rounds?
   - How do nodes learn others' bids in "sealed-bid"?

---

## Notes

- Start with LEACH implementation to validate energy model
- Use fixed random seeds for reproducibility
- Log everything — you can always filter later
- Profile code if simulations are slow (NumPy vectorization helps)
