# Trend-Based Filling: Preventing Sudden Jumps

## The Problem with Direct Substitution

When filling missing values by directly using values from other feeders, we get **sudden jumps** because different feeders operate at different consumption levels.

### Example of the Problem:

```
Time:        T0    T1    T2    T3    T4    T5
Feeder A:    100   110   ???   ???   ???   120   (missing T2-T4)
Feeder B:    200   220   230   240   250   260
Feeder C:    150   165   170   175   180   185

Direct substitution (weighted average):
T2: (230×0.4 + 170×0.33 + ... ) = 195  ❌ HUGE JUMP from 110!
```

**Problem**: Feeder A normally operates around 100-120, but we're filling it with values around 200 because that's what other feeders show!

## The Solution: Trend-Based Filling

Instead of using absolute values from other feeders, we use their **percentage change (trend)** and apply it to Feeder A's baseline.

### How It Works:

1. **Find the baseline**: Last known value before the gap
2. **Calculate trends from other feeders**: How much did they change?
3. **Apply weighted average of trends**: To the baseline

### Same Example with Trend-Based Filling:

```
Time:        T0    T1    T2    T3    T4    T5
Feeder A:    100   110   ???   ???   ???   120   (missing T2-T4)
Feeder B:    200   220   230   240   250   260
Feeder C:    150   165   170   175   180   185

Baseline for Feeder A: 110 (last known value at T1)

For T2:
- Feeder B: 220→230 (ratio: 1.045, weight: 0.40)
- Feeder C: 165→170 (ratio: 1.030, weight: 0.33)
- Weighted ratio: (1.045×0.40 + 1.030×0.33 + ...) = 1.038

Filled value at T2: 110 × 1.038 = 114.2 ✅ Smooth continuation!

For T3:
- Feeder B: 220→240 (ratio: 1.091)
- Feeder C: 165→175 (ratio: 1.061)
- Weighted ratio: 1.078
- Filled value: 110 × 1.078 = 118.6 ✅ Still smooth!
```

## Key Benefits

### ✅ Preserves Baseline Level
Each feeder maintains its characteristic consumption level. No sudden jumps from 110 to 200!

### ✅ Follows Patterns
Still captures the temporal patterns and trends from correlated feeders.

### ✅ Weighted by Correlation
Feeders with stronger correlation have more influence on the trend calculation.

### ✅ Smooth Transitions
The filled values smoothly connect the last known value to the next known value.

## Visual Comparison

### Before (Direct Substitution):
```
Consumption
    │
250 │                    B─────B
    │                   /
200 │            B─────B
    │           /
150 │    C─────C─────C
    │
100 │ A──A  JUMP!
    │         ╲
 50 │          ???═══???  (filled with ~195)
    │                   ╲
  0 │                    A
    └─────────────────────────> Time
      T0  T1  T2  T3  T4  T5
```

### After (Trend-Based):
```
Consumption
    │
250 │                    B─────B
    │                   /
200 │            B─────B
    │           /
150 │    C─────C─────C
    │
100 │ A──A╌╌╌╌╌╌╌╌╌╌╌╌╌╌A  (smooth: 110→114→119→120)
    │      ╲          /
 50 │       ╲        /
    │        ╲      /
  0 │         ╲____/
    └─────────────────────────> Time
      T0  T1  T2  T3  T4  T5
```

## Mathematical Formula

For a missing value at time `t` in feeder `A`:

```
1. Find baseline: last_known_value_A (at time t₀)
2. For each correlated feeder i:
   ratio_i = value_i(t) / value_i(t₀)
3. Calculate weighted ratio:
   weighted_ratio = Σ(ratio_i × weight_i) / Σ(weight_i)
4. Fill value:
   filled_value_A(t) = last_known_value_A × weighted_ratio
```

Where:
- `weight_i` = correlation coefficient between feeder A and feeder i
- `t₀` = time of last known value before the gap
- `t` = current time being filled

## When This Approach Works Best

✅ **When feeders have similar patterns but different levels**
- Example: Residential vs. commercial areas (same daily pattern, different magnitudes)

✅ **When correlation exists but with level offset**
- Example: Two substations serving similar demographics but different sizes

✅ **For medium-length gaps (hours to days)**
- Long enough that simple interpolation isn't enough
- Short enough that trends from other feeders are still relevant

## Implementation Notes

- Minimum correlation threshold: 0.3 (too low correlation = unreliable trends)
- Only uses positive correlations (negative correlation means opposite patterns)
- Requires at least one value before the gap (baseline)
- Falls back to interpolation if no baseline or no correlated feeders available
