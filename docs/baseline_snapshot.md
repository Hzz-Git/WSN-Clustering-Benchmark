# Baseline Snapshot (Pre-Fix)

Captured: 2026-01-14
Config: N=50, max_epochs=300, 5 trials, seed_base=42

## Results (control energy disabled, original bandit reward)

| Algorithm | FND | HND | LND | Ctrl Msgs | Throughput |
|-----------|-----|-----|-----|-----------|------------|
| AUCTION | 132.8±9.2 | 181.6±12.8 | 300.0±0.0 | 15965 | 9987 |
| HEED | 36.8±6.9 | 218.0±22.2 | 300.0±0.0 | 20696 | 10298 |
| LEACH | 54.0±8.6 | 135.6±10.5 | 217.8±13.6 | 7190 | 6742 |

## Known Issues at Baseline

1. **Control messages counted but not charged energy** - `self.control_messages += 1` without energy deduction
2. **Bandit reward broken** - Rewarded "being CH" (1.0 if CH, 0.1 otherwise) instead of "being efficient as CH"
3. **Node.reset_for_round() didn't clear bid** - Stale bids could pollute backup selection

## Purpose

This snapshot serves as regression target for Check A validation.
When `control.enabled=False`, results should match these values within noise margin.
