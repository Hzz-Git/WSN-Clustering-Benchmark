# Control Energy & Bandit Reward Fix Report

Date: 2026-01-14

## Summary

Two critical bugs were identified and fixed:

1. **Control messages did not consume energy** - Only counted, never charged
2. **Bandit reward was broken** - Learned "be CH always" instead of "be CH when efficient"

## Changes Made

### 1. Control Message Energy Model (base.py)

Added four unified helpers with proper energy accounting:

```python
ctrl_unicast(sender, receiver, bits)      # TX + RX energy
ctrl_broadcast_fixed(sender, radius, bits) # Discovery broadcast (fixed TX radius)
ctrl_broadcast_to_set(sender, receivers, bits)  # Cluster notification (d_max TX)
ctrl_broadcast_from_bs(receivers, bits)   # BS broadcast (RX only)
```

Config switches added:
- `control.enabled` - Master switch (False = no energy, for regression testing)
- `control.bits_multiplier` - Scale factor for sensitivity testing

### 2. Bandit Reward Fix (auction.py)

**Old reward (broken):**
```python
reward = 1.0 if was_CH else 0.1  # All nodes learn high m
```

**New reward (energy-efficiency based):**
```python
cost_per_bit = spent_energy / bits_out
reward = clip((ref - cost_per_bit) / ref, -1, +1)
# ref = EMA of median CH cost_per_bit
```

This teaches: "Be CH when your cost is below median; avoid CH duty when expensive"

### 3. Node.reset_for_round() (node.py)

Added clearing of stale bid and bandit flag to prevent pollution.

### 4. LEACH Discovery Radius (leach.py)

Added config switch `control.discovery_radius_mode`:
- `"local"` - Use join_radius (30m)
- `"global"` - Use network diameter (~141m, original LEACH design)

## Verification Results

### Check A: Regression to Baseline (control.enabled=False)

| Algorithm | FND Diff | HND Diff | Status |
|-----------|----------|----------|--------|
| AUCTION | 2.6 | 3.0 | PASS |
| HEED | 0.0 | 0.0 | PASS |
| LEACH | 0.0 | 0.0 | PASS |

### Check B: Sensitivity (bits_multiplier=10)

| Algorithm | FND Drop | HND Drop | Ctrl Energy Ratio |
|-----------|----------|----------|-------------------|
| AUCTION | 30.0 | 27.8 | 8.1x |
| HEED | 3.0 | 73.8 | 6.3x |
| LEACH | 6.6 | 25.2 | 8.0x |

HEED most affected due to iterative status broadcasts (control-plane dominated).

### Check C: Bandit Learning Validation

| Metric | Value | Expected |
|--------|-------|----------|
| m diversity (std) | 0.487 | > 0.1 |
| m-distance correlation | -0.419 | < 0 |
| Near BS mean_m | 1.357 | Higher |
| Far BS mean_m | 0.714 | Lower |

Bandit learned spatial strategy: "aggressive near BS, conservative far from BS"

## New Config Keys

```yaml
control:
  enabled: true              # Master switch for control energy
  bits_announce: 960         # CH/seed announcement
  bits_bid: 960              # Auction bid message
  bits_result: 960           # Auction result broadcast
  bits_join: 800             # Join request
  bits_status: 800           # HEED tentative status
  discovery_radius_mode: local  # "local" or "global"
  bits_multiplier: 1.0       # Scale factor for sensitivity
```

## Files Modified

- `src/algorithms/base.py` - Control message helpers + config switches
- `src/algorithms/auction.py` - New bandit reward + control energy calls
- `src/algorithms/heed.py` - Control energy calls
- `src/algorithms/leach.py` - Control energy calls + discovery_radius_mode
- `src/models/node.py` - reset_for_round() fix
- `config/default.yaml` - New control section

## Conclusion

The fix enables fair comparison between protocols by properly accounting for control overhead.
Key finding: Protocol ranking depends on control-plane vs data-plane dominance.
