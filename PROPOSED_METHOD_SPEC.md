# PROPOSED METHOD SPECIFICATION
# Single-Round Sealed-Bid Auction with Fairness Credit for WSN Clustering

This document contains the **exact formulas and parameters** from the presentation.
Use this as the implementation reference.

---

## 1. OVERVIEW

**Protocol Name**: Auction-Based Clustering (ABC) with Fairness Credit and Sleep Scheduling

**Key Innovation**: 
- One tiny score per node compresses: battery, closeness, link quality, and turn-taking (fairness credit)
- One round: each node sends one sealed bid to the current CH; CH selects Primary + Backup, publishes schedule
- Predictable cost: exactly `s + 2` control messages per cluster of size `s`

---

## 2. BID/SCORE CALCULATION

### 2.1 Auction Score Formula (EXACT)

Each node i at epoch t calculates its bid:

```
b_i(t) = m_i(t) * ( α * E_i(t)/E_i^(0) + β * 1/(h_i(t) + ε) + γ * Q_{i→CH}(t) ) - λ * D_i(t)
```

### 2.2 Variable Definitions

| Symbol | Description | Notes |
|--------|-------------|-------|
| `E_i(t)` | Residual energy of node i at epoch t | Joules |
| `E_i^(0)` | Initial energy of node i | Joules |
| `h_i(t)` | Estimated hop count (or ETX) from node i *as CH* to the sink | ε > 0 stabilizer |
| `Q_{i→CH}(t)` | Measured link quality from node i to current CH | e.g., PRR or 1/ETX |
| `m_i(t)` | Aggressiveness knob (bandit-tuned, bounded) | ∈ {0.5, 0.8, 1.0, 1.2, 1.5, 2.0} |
| `α, β, γ, λ` | Weights for energy, closeness, link, and fairness | Tuned offline |
| `D_i(t)` | Fairness debt/credit | Higher debt lowers the bid |
| `ε` | Small constant to stabilize division | > 0 |

### 2.3 Weight Parameters (Default Values from Slide 8)

| Weight | Description | Default |
|--------|-------------|---------|
| `α` | Energy term weight | 5.0 |
| `β` | Closeness/hop term weight | 1.0 |
| `γ` | Link quality term weight | 0.2 |
| `λ` | Fairness debt penalty weight | 2.0 |
| `ε` | Stabilizer constant | 0.1 |

### 2.4 Term-by-Term Explanation

**Energy Term**: `α * E_i(t) / E_i^(0)`
- Normalized residual energy (ratio between 0 and 1)
- Nodes with more remaining energy get higher bids
- Encourages energy-rich nodes to become CH

**Closeness Term**: `β / (h_i(t) + ε)`
- Inverse of hop count to sink
- Nodes closer to BS (fewer hops) get higher bids
- Reduces long-range transmission cost

**Link Quality Term**: `γ * Q_{i→CH}(t)`
- PRR (Packet Reception Rate) or 1/ETX to current CH
- Better link = higher bid
- Ensures reliable intra-cluster communication

**Aggressiveness Multiplier**: `m_i(t) * (...)`
- Multiplies the entire capability score
- Bandit-tuned: node learns optimal aggressiveness over time
- Higher m = more aggressive bidding

**Fairness Penalty**: `- λ * D_i(t)`
- Subtracted from bid (not multiplied by m)
- Positive debt (was CH recently) → lower bid
- Negative debt (credit, hasn't been CH) → higher bid

### 2.5 Bandit for Aggressiveness Tuning

- Each node maintains a bandit over action space: `m ∈ {0.5, 0.8, 1.0, 1.2, 1.5, 2.0}`
- Uses ε-greedy with `ε = 0.1` (10% exploration)
- Reward signal: based on whether becoming CH was beneficial (energy-efficient)
- Bounded: m cannot go below 0.5 or above 2.0

### 2.6 Simplified Version (for simulation without link quality)

If link quality measurement is not simulated, use:

```
b_i(t) = m_i(t) * ( α * E_i(t)/E_i^(0) + β * 1/(h_i(t) + ε) ) - λ * D_i(t)
```

Where `h_i(t)` can be approximated as:
- `h_i(t) = ceil(d_i / R_comm)` — hops based on communication range
- Or simply `h_i(t) = d_i / R_comm` — fractional hops (distance proxy)

---

## 3. FAIRNESS CREDIT/DEBT SYSTEM

### 3.1 Debt Update Formula (EXACT)

After each epoch, update each node's debt with clipping:

```
D_i(t+1) = min{ D_max, max{ -D_max, D_i(t) + η * (1{i=CH} - share_i) } }
```

### 3.2 Share Calculation (EXACT)

```
share_i = E_i^(0) / Σ_{j∈C} E_j^(0)
```

### 3.3 Variable Definitions

| Symbol | Description |
|--------|-------------|
| `1{i=CH}` | Indicator: 1 if node i served as CH in the epoch, else 0 |
| `share_i` | Capacity-weighted duty share (by initial energy) within the cluster |
| `D_max` | Clipping bound on the credit/debt |
| `η` | Fairness step size |
| `C` | Node set of the current cluster |

### 3.4 Worked Example (from presentation)

**Setup**:
- Cluster C = {A, B, C}
- Initial energies: E^(0) = [2, 2, 1] Joules
- Shares: share = [2/5, 2/5, 1/5] = [0.4, 0.4, 0.2]
- Fairness step: η = 0.1

**If node A was CH this epoch**:

```
D_A(t+1) = D_A(t) + 0.1 × (1 - 0.4) = D_A(t) + 0.06    (debt increases)
D_B(t+1) = D_B(t) + 0.1 × (0 - 0.4) = D_B(t) - 0.04    (credit increases)
D_C(t+1) = D_C(t) + 0.1 × (0 - 0.2) = D_C(t) - 0.02    (credit increases)
```

**Effect on next epoch**:
- The term `-λ * D_i` in the bid formula:
  - Reduces A's bid (positive debt → penalty)
  - Lifts B and C's bids (negative debt = credit → bonus)
- Over time, CH duty converges toward shares: 40% / 40% / 20%

### 3.5 Intuition

- **Positive D_i (debt)**: Node has served as CH more than its fair share → bid penalty
- **Negative D_i (credit)**: Node has served as CH less than its fair share → bid bonus
- **Clipping [-D_max, D_max]**: Prevents unbounded accumulation
- **Share proportional to initial energy**: Nodes with more capacity should serve more

---

## 4. PROTOCOL PHASES

### Phase Flow (Normal Round, Epoch ≥ 1):

```
Broadcast CH → Bids → Allocation → Broadcast Result → Data/Sleep → Update
```

1. **Broadcast CH**: Current CH announces itself
2. **Bids**: All nodes calculate and send sealed bids to CH
3. **Allocation**: CH selects Primary (next CH) + Backup based on bids
4. **Broadcast Result**: CH publishes schedule (who transmits when)
5. **Data/Sleep**: Data transmission phase; some nodes may sleep
6. **Update**: Update fairness debt D_i for all nodes

### First Round (Epoch 0) - Bootstrap:

```
Listen (all nodes) → Calculate bid → Countdown timer → Seed/Member
```

1. **Listen**: All nodes listen for SEED_ANNOUNCE messages
2. **Calculate bid**: Each node computes its bid score
3. **Countdown timer**: Set timer inversely proportional to bid
   - `timer_i = T_max * (1 - normalized_bid_i)`
   - **Higher bid = Lower timer** (announces first)
4. **Decision**:
   - If timer expires before hearing SEED_ANNOUNCE → become SEED (CH)
   - If hear SEED_ANNOUNCE before timer expires → become Member

**Bootstrap parameters**:
- Join radius: `R_join = 30`
- Seed selection via spacing

---

## 5. CLUSTER ROLES

| Role | Symbol | Description |
|------|--------|-------------|
| Cluster Head (CH) | Primary (★) | Aggregates & transmits to BS |
| Backup | ☆ | Next in line if CH fails |
| Member | — | Senses & relays to CH |
| Sleeping | zZ | Low-duty cycle to save energy |
| Sentinel | — | 1 nearest member per CH, always awake |

---

## 6. ENERGY MODEL

### 6.1 Radio Energy Parameters

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Electronics energy | `E_elec` | 50 nJ/bit |
| Amplifier energy | `E_amp` | 100 pJ/bit/m² |
| Transmit power (relative) | `P_TX` | 0.05 |
| Receive power (relative) | `P_RX` | 0.04 |
| Idle power (relative) | `P_IDLE` | 0.002 |

### 6.2 Energy Consumption Formulas

**Transmit**:
```
E_tx(k, d) = E_elec * k + E_amp * k * d²
```

**Receive**:
```
E_rx(k) = E_elec * k
```

Where:
- `k` = packet size in bits
- `d` = distance in meters

---

## 7. EXPERIMENT PARAMETERS

### 7.1 Network Configuration

| Parameter | Value |
|-----------|-------|
| Number of nodes | N = 50 |
| Area | 100m × 100m (square field) |
| Base Station position | (50, 100) — at top center edge |
| Trials | 15 |

### 7.2 Initial Energy Distribution

```
E_i^(0) ~ N(μ = 2.0, σ = 0.2), clipped at > 0.1
```

- Heterogeneous initial energy (Gaussian distribution)
- Mean: 2.0 J
- Std dev: 0.2 J
- Minimum: 0.1 J (clipped)

### 7.3 Packet Sizes

| Packet Type | Size |
|-------------|------|
| Control (HEED) | 100 B |
| Control (Auction) | 120 B |
| Data | 4000 B |

### 7.4 Clustering Parameters

| Parameter | Value |
|-----------|-------|
| Join radius | R_join = 30 |
| Spacing radius (Epoch ≥ 1) | 30 |
| CH data load | 5 member packets per epoch |
| Aggregation ratio | 0.5× (CH sends half the received data to sink) |
| Sentinel policy | 1 nearest member per CH |

### 7.5 Algorithm-Specific Parameters

**Auction**:
| Parameter | Value |
|-----------|-------|
| α_base | 5.0 |
| β | 1.0 |
| γ | 0.2 |
| λ | 2.0 |
| η (fairness step) | 0.1 |
| ε (bandit exploration) | 0.1 |
| m (actions) | {0.5, 0.8, 1.0, 1.2, 1.5, 2.0} |

**HEED Baseline**:
| Parameter | Value |
|-----------|-------|
| C_prob | 0.05 |

---

## 8. METRICS TO COLLECT

Based on slide 9, track:

1. **Total Network Residual Energy** vs Epoch
   - Plot: Line graph with confidence bands
   - Compare: Auction (w/ bootstrap) vs HEED baseline

2. **Topology Visualization**
   - Cluster boundaries at epoch 10
   - Cluster boundaries at final epoch
   - Show CH positions (★), member positions, cluster circles

3. **Additional Metrics** (recommended):
   - First Node Dead (FND)
   - Half Nodes Dead (HND)
   - Last Node Dead (LND)
   - Energy variance across nodes
   - Number of orphan nodes
   - Control message overhead

---

## 9. PSEUDOCODE

### 9.1 Main Simulation Loop

```python
def simulate(nodes, config, num_epochs):
    # Initialize
    for node in nodes:
        node.energy = sample_initial_energy(config.mu, config.sigma)
        node.debt = 0.0
        node.m = 1.0  # default aggressiveness
    
    # Epoch 0: Bootstrap
    clusters = bootstrap_seed_selection(nodes, config.R_join)
    
    # Main loop
    for epoch in range(1, num_epochs):
        if count_alive(nodes) == 0:
            break
        
        # Phase 1: Bid calculation
        for node in alive_nodes(nodes):
            node.bid = calculate_bid(node, config)
        
        # Phase 2-3: CH collects bids, allocates roles
        for cluster in clusters:
            ch = cluster.head
            bids = collect_bids(cluster.members)
            primary, backup = select_top_two(bids)
            cluster.next_head = primary
        
        # Phase 4: Broadcast schedule
        # Phase 5: Data transmission
        for cluster in clusters:
            transmit_data(cluster, config)
        
        # Phase 6: Update debt
        for cluster in clusters:
            update_debt(cluster, config.eta)
        
        # Rotate CH
        for cluster in clusters:
            cluster.head = cluster.next_head
        
        # Re-cluster if needed (global reselection with spacing)
        if should_recluster(epoch):
            clusters = global_ch_reselection(nodes, config.spacing_radius)
    
    return collect_metrics(nodes)
```

### 9.2 Bid Calculation

```python
def calculate_bid(node, current_ch, bs_position, config):
    """
    Calculate bid using exact formula:
    b_i(t) = m_i(t) * (α * E_i(t)/E_i^(0) + β/(h_i(t)+ε) + γ * Q_{i→CH}(t)) - λ * D_i(t)
    """
    # Energy ratio term: α * E_i(t) / E_i^(0)
    energy_ratio = node.energy / node.initial_energy
    energy_term = config.alpha * energy_ratio
    
    # Closeness term: β / (h_i(t) + ε)
    # h_i = estimated hop count from node (as CH) to sink
    # Approximation: h_i = distance_to_bs / communication_range
    distance_to_bs = euclidean_distance(node.position, bs_position)
    h_i = distance_to_bs / config.comm_range  # or use ceil() for integer hops
    closeness_term = config.beta / (h_i + config.epsilon)
    
    # Link quality term: γ * Q_{i→CH}(t)
    # Q = PRR (packet reception rate) or 1/ETX
    # Approximation: Q = 1.0 for nodes within range, or model based on distance
    if current_ch is not None:
        distance_to_ch = euclidean_distance(node.position, current_ch.position)
        Q_i_ch = estimate_link_quality(distance_to_ch, config.comm_range)
    else:
        Q_i_ch = 1.0  # First round, no CH yet
    link_term = config.gamma * Q_i_ch
    
    # Aggressiveness multiplier: m_i(t) * (...)
    m_i = node.aggressiveness  # bandit-tuned value
    
    # Fairness penalty: - λ * D_i(t)
    fairness_penalty = config.lambda_ * node.debt
    
    # Final bid
    bid = m_i * (energy_term + closeness_term + link_term) - fairness_penalty
    
    return bid


def estimate_link_quality(distance, comm_range):
    """
    Estimate link quality Q based on distance.
    Simple model: Q decreases with distance.
    """
    if distance <= comm_range:
        # PRR model: quality degrades near edge of range
        return max(0.1, 1.0 - 0.5 * (distance / comm_range) ** 2)
    else:
        return 0.0  # Out of range
```

### 9.3 Debt Update

```python
def update_debt(cluster, config):
    """
    Update debt using exact formula:
    D_i(t+1) = min{D_max, max{-D_max, D_i(t) + η(1{i=CH} - share_i)}}
    
    share_i = E_i^(0) / Σ_{j∈C} E_j^(0)
    """
    # Calculate shares based on initial energy
    total_init_energy = sum(n.initial_energy for n in cluster.all_nodes)
    
    for node in cluster.all_nodes:
        # Capacity-weighted duty share
        share_i = node.initial_energy / total_init_energy
        
        # Indicator: was this node the CH?
        was_ch = 1.0 if node == cluster.head else 0.0
        
        # Dual-ascent update with clipping
        new_debt = node.debt + config.eta * (was_ch - share_i)
        
        # Clip to [-D_max, D_max]
        node.debt = min(config.d_max, max(-config.d_max, new_debt))
```

### 9.4 Bootstrap (Epoch 0)

```python
def bootstrap_seed_selection(nodes, R_join):
    seeds = []
    remaining = set(nodes)
    
    # Timer-based: nodes with higher bids announce first
    sorted_nodes = sorted(nodes, key=lambda n: n.bid, reverse=True)
    
    for node in sorted_nodes:
        if node not in remaining:
            continue
        
        # Check if any existing seed is within R_join
        can_be_seed = True
        for seed in seeds:
            if distance(node, seed) < R_join:
                can_be_seed = False
                break
        
        if can_be_seed:
            node.role = SEED
            seeds.append(node)
            # All nodes within R_join become members
            for other in list(remaining):
                if distance(node, other) <= R_join:
                    other.role = MEMBER
                    other.cluster_head = node
                    remaining.discard(other)
    
    return form_clusters(seeds, nodes)
```

---

## 10. KEY DIFFERENCES FROM LEACH/HEED

| Aspect | LEACH | HEED | Our Auction |
|--------|-------|------|-------------|
| CH Selection | Probabilistic (random) | Multi-round iteration | Single-round sealed-bid |
| Energy Awareness | No | Yes (residual energy) | Yes (in bid formula) |
| Fairness | Rotation by probability | None explicit | Explicit debt/credit |
| Message Overhead | Low | High (multiple rounds) | Low (s+2 per cluster) |
| Distance Awareness | No | Intra-cluster cost | Distance to BS |

---

## 11. IMPLEMENTATION CHECKLIST

### Core Components
- [ ] Node class with: position, energy, initial_energy, debt, aggressiveness (m), role, cluster_head, bid
- [ ] Energy model: E_tx = E_elec × k + E_amp × k × d²
- [ ] Cluster class with: head, members, all_nodes

### Bid Calculation
- [ ] Energy term: `α × E_i(t) / E_i^(0)`
- [ ] Closeness term: `β / (h_i(t) + ε)` where h_i = hop count or distance proxy
- [ ] Link quality term: `γ × Q_{i→CH}(t)` (PRR or 1/ETX)
- [ ] Aggressiveness multiplier: `m_i(t) × (...)`
- [ ] Fairness penalty: `- λ × D_i(t)`
- [ ] Full formula: `b_i(t) = m_i(t) × (α × E_ratio + β/(h+ε) + γ×Q) - λ×D_i(t)`

### Fairness System
- [ ] Share calculation: `share_i = E_i^(0) / Σ E_j^(0)` per cluster
- [ ] Debt update: `D_i(t+1) = clip(D_i(t) + η×(1{CH} - share_i), -D_max, D_max)`
- [ ] Parameters: η = 0.1, D_max = (to be set)

### Protocol Phases
- [ ] Bootstrap (Epoch 0): timer-based seed selection with R_join = 30
- [ ] Normal round: Broadcast CH → Bids → Allocation → Broadcast Result → Data/Sleep → Update
- [ ] Global CH reselection with spacing radius = 30

### Baselines
- [ ] HEED with C_prob = 0.05

### Experiments
- [ ] 15 trials with heterogeneous initial energy: N(μ=2.0, σ=0.2), clipped > 0.1
- [ ] Metrics: total residual energy vs epoch, topology visualization
- [ ] N = 50 nodes, 100×100 area, BS at (50, 100)

---

## 12. NOTES FOR IMPLEMENTATION

1. **BS Position**: (50, 100) means BS is OUTSIDE the 100×100 sensing area (at top edge)

2. **Heterogeneous Energy**: Unlike standard LEACH, nodes start with DIFFERENT energy levels (Gaussian)

3. **Aggregation**: CH aggregates 5 member packets into 0.5× → sends ~2.5 packets worth to BS

4. **Sentinel**: One member per CH stays awake always (doesn't sleep) for reliability

5. **Global Reselection**: In Epoch ≥ 1, CH selection considers spacing radius = 30 to avoid clustering CHs together
