# Baseline Snapshot (Before Control-Energy + Bandit-Reward Fix)

**Config**: default.yaml, 5 trials, max 300 epochs

## Results Summary

| Algorithm | FND | HND | LND | Control Msgs | Throughput |
|-----------|-----|-----|-----|--------------|------------|
| AUCTION | 132.8±9.2 | 181.6±12.8 | 300.0±0.0 | 15965±681 | 9987±436 |
| HEED | 36.8±6.9 | 218.0±22.2 | 300.0±0.0 | 20696±480 | 10298±422 |
| LEACH | 54.0±8.6 | 135.6±10.5 | 217.8±13.6 | 7190±524 | 6742±488 |

## Notes

- Control messages are **counted only**, not charged energy
- Bandit reward: 1.0 if CH, 0.1 otherwise (broken incentive)
- This is the baseline to compare against after fixes
