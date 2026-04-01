# ScaledSigmoid Recovery Mechanism: Gradient Analysis at Output Boundaries

## Summary

ScaledSigmoid (`scale * sigmoid(x) + shift`) provides a **self-correcting mechanism** against weight explosion. When weights grow large enough to push sigmoid output to its boundary (0 or 1), standard sigmoid has near-zero gradient and cannot recover. ScaledSigmoid reaches the same output values at smaller `x`, where the gradient is still meaningful — allowing the optimizer to pull weights back.

This is distinct from the "faster convergence" benefit observed in earlier experiments. It is specifically about **robustness to weight explosion** — the ability to self-correct once weights have grown too large.

## Mathematical Derivation

### Setup

- Standard sigmoid: `σ(x) = 1 / (1 + e^(-x))`
- ScaledSigmoid: `f(x) = scale * σ(x) + shift`
- Gradient: `f'(x) = scale * σ(x) * (1 - σ(x))`

### Finding x where output = 1

Solve `scale * σ(x) + shift = 1`:

```
σ(x) = (1 - shift) / scale
x = ln(σ(x) / (1 - σ(x)))
```

### Numerical Results

| Configuration | σ(x) required | x value | Gradient at output=1 |
|---------------|---------------|---------|----------------------|
| Original (scale=1.0, shift=0.0) | 1.0000 | ∞ | ≈ 0 |
| scale=1.01, shift=-0.005 | 0.9950 | ≈ 5.30 | ≈ 0.0050 |
| scale=1.05, shift=-0.025 | 0.9762 | ≈ 3.71 | ≈ 0.0244 |
| scale=1.1, shift=-0.05 | 0.9545 | ≈ 3.04 | ≈ 0.0457 |

### Detailed Computation

**scale=1.01, shift=-0.005:**
```
σ(x) = (1 + 0.005) / 1.01 = 1.005 / 1.01 ≈ 0.99505
x = ln(0.99505 / 0.00495) = ln(201.0) ≈ 5.30
gradient = 1.01 × 0.99505 × 0.00495 ≈ 0.00497
```

**scale=1.05, shift=-0.025:**
```
σ(x) = (1 + 0.025) / 1.05 = 1.025 / 1.05 ≈ 0.97619
x = ln(0.97619 / 0.02381) = ln(41.0) ≈ 3.71
gradient = 1.05 × 0.97619 × 0.02381 ≈ 0.02441
```

**scale=1.1, shift=-0.05:**
```
σ(x) = (1 + 0.05) / 1.1 = 1.05 / 1.1 ≈ 0.95455
x = ln(0.95455 / 0.04545) = ln(21.0) ≈ 3.04
gradient = 1.1 × 0.95455 × 0.04545 ≈ 0.04771
```

### Gradient Ratio Comparison

Relative to scale=1.01 at the output=1 boundary:

| Configuration | Gradient | Ratio vs scale=1.01 |
|---------------|----------|----------------------|
| scale=1.01 | 0.0050 | 1.0× |
| scale=1.05 | 0.0244 | **4.9×** |
| scale=1.1 | 0.0457 | **9.2×** |

## The Recovery Mechanism

### Standard Sigmoid — No Recovery

```
Weight explosion → x grows large → σ(x) → 1.0 → gradient → 0
    → optimizer receives no signal → weights stay large or keep growing
    → vicious cycle (positive feedback loop)
```

### ScaledSigmoid — Self-Correcting

```
Weight explosion → x grows large → but output reaches 1.0 at finite x
    → gradient is still meaningful (e.g., 0.024 for scale=1.05)
    → optimizer receives signal to reduce weights
    → weights shrink back → negative feedback loop (self-correcting)
```

### Key Insight

The recovery gradient scales roughly with `(scale - 1)`. Larger scale factors provide stronger recovery forces, but also introduce risks in gating mechanisms (see LSTM caveat below).

## Experimental Evidence

### FC/CNN Experiments (Supports Recovery Mechanism)

From the staircase test case (test case 6, 3 boundaries):

| Configuration | Final Weight Magnitude | Interpretation |
|---------------|----------------------|----------------|
| Original sigmoid | ~100 | Weights grow unbounded — no recovery force |
| scale=1.01 | ~95 | Minimal recovery — too weak |
| scale=1.05 | ~90 | Slight recovery — gradient still small at boundary |
| scale=1.1 | **~33** | Strong recovery — weights stabilized at 1/3 of original |

The scale=1.1 model achieved the **same loss** (≈ 0) with weights 3× smaller, demonstrating that the recovery mechanism prevents unnecessary weight growth.

### LSTM Experiments (Caveat — Recovery Breaks Gating)

In LSTM gates, the [0, 1] bound is a design feature, not a limitation. ScaledSigmoid in gates:
- scale=1.1 → forget gate can exceed 1.0 → cell state amplified each step → instability
- Results: accuracy **decreased** with larger scale factors (73.08% → 72.57%)

**Conclusion**: The recovery mechanism is beneficial when sigmoid is used as an **activation function**, but harmful when sigmoid is used as a **gate controller** where [0, 1] bounding is intentional.

### Direct Recovery Demo (`DemoRecovery.py`)

A standalone demo that directly tests the recovery mechanism:
- Task: simple threshold (X > 0.5)
- All weights intentionally initialized to 50.0 (simulating explosion)
- Trains with SGD (not Adam, to avoid adaptive LR masking the effect)
- Expected: sigmoid stays stuck (gradient ≈ 0), ScaledSigmoid recovers (gradient still meaningful)

Run: `python DemoRecovery.py`

## Applicable Scenarios

| Scenario | Recovery Helpful? | Reason |
|----------|-------------------|--------|
| FC networks with binary targets | ✅ Yes | Weights must push to extremes for 0/1 output |
| CNN activation layers | ✅ Yes | Prevents unnecessary weight growth |
| Tasks with many decision boundaries | ✅ Yes — larger scale needed | More boundaries = more saturation pressure |
| LSTM/GRU gates | ❌ No | Gate values > 1 or < 0 destabilize cell state |
| Output layer with softmax | ➖ Neutral | Softmax handles its own scaling |
