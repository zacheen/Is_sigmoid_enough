# Session Summary: 2026-03-31

## Main Objective

Fix `DemoRecovery.py` to demonstrate ScaledSigmoid's self-correction mechanism, then explore additional experiments (LR tolerance, multi-layer recovery) to further characterize ScaledSigmoid's advantages and limitations.

## Environment

- **Python**: `C:\Users\User\miniconda3\envs\py310_torch210_cuda118\python.exe` (Python 3.10.19)
- **Windows encoding**: `cp950` — Unicode symbols (`✓✗`) cause `UnicodeEncodeError`, use ASCII alternatives (`OK`/`FAIL`)
- **Matplotlib**: `plt.show()` blocks execution in CLI; always `plt.savefig()` before `plt.show()`

---

## Phase 1: Fixing DemoRecovery.py (Single Neuron)

### Problem
Three previous attempts to demo "recovery from weight explosion" all failed:
1. `init_weight=50` for all params → `sigmoid(100)=1.0` in float32 → gradient literally 0
2. `init_weight=10` fc2-only → still too saturated
3. Two-phase (high LR overshoot → normal LR) → weights don't change after switch

### Root Cause
When pre-activation `x > 15`, gradient is ~1e-7 for ALL sigmoid variants. The 1.1x scale factor doesn't help when the base gradient is essentially 0. Previous demos pushed weights too far into saturation.

### Solution: Single Neuron Demo with Moderate Init
Simplified to the purest possible test case:
- **Architecture**: Single neuron `y = activation(w * x + b)`, x=1, bias frozen
- **Init weight**: 0.8 (output ≈ 0.6, moderate — not saturated)
- **Explosion method**: Epoch 0 uses reverse backprop (`lr * -30`) to explode weights in one step
- **Recovery**: Epoch 1+ trains normally with SGD

### Key Insight: Gradient DIRECTION, Not Magnitude
The self-correction mechanism is about gradient **direction reversal**, not larger gradients:

| | Standard Sigmoid | ScaledSigmoid(1.1) |
|--|-----------------|---------------------|
| At w=5 (exploded) | output=0.993 **< target** | output=1.043 **> target** |
| Gradient direction | Pushes w **UP** (wrong) | Pushes w **DOWN** (correct) |
| Result | Stuck / keeps growing | Recovers toward equilibrium |

ScaledSigmoid's output can EXCEED [0,1], so when weights overshoot, the output overshoots the target, and the gradient reverses to pull weights back. Standard sigmoid never reaches 1.0, so gradient always pushes weights further up.

### Theoretical Equilibrium Points
For `scale * sigmoid(w_eq) + shift = target`:

| Config | w_eq (target=0.6) | w_eq (target=0.95) |
|--------|-------------------|---------------------|
| Sigmoid | 0.405 (finite) | 2.944 (large but finite) |
| SS(1.05) | 0.386 | 2.565 |
| SS(1.1) | 0.368 | 2.303 |
| SS(1.5) | 0.268 | 1.386 |

When target is inside sigmoid's natural range (e.g., 0.6), sigmoid CAN reach equilibrium at finite w. The advantage of ScaledSigmoid is most apparent when target = 0 or 1 (boundary values), where sigmoid's equilibrium is at w=±∞.

### Implementation Details
- Two optimizers: `rev_optimizer = SGD(lr=lr*-30)` for epoch 0, `optimizer = SGD(lr=lr)` for epoch 1+
- Or single optimizer: `optimizer.param_groups[0]['lr'] = lr * -10` at epoch 0, reset at epoch 1
- Both approaches are equivalent
- Bias frozen with `model.linear.bias.requires_grad = False`
- Scale=1.5 needs lower lr (5.0 vs 12.0) to prevent oscillation at equilibrium

### Oscillation Problem at High LR
When lr is too large relative to gradient sensitivity near the equilibrium point, weight overshoots equilibrium in both directions → persistent oscillation. This is especially bad for larger scale factors because the equilibrium point has larger gradients. Fix: reduce lr.

---

## Phase 2: LR Tolerance Experiment (DemoLrTolerance.py)

### Hypothesis (DISPROVEN)
"ScaledSigmoid tolerates a wider range of learning rates due to self-correction."

### Experiment Design
- Task: XOR (2→4→1 FC, SGD, 2000 epochs)
- LR sweep: 19 values from 0.001 to 50.0 (log-spaced)
- 3 seeds per (config, LR) pair, median loss
- Convergence threshold: MSE < 0.01

### Results
```
Sigmoid:          9/19 converged, range [1.0 ~ 30.0]
SS(1.05, -0.025): 9/19 converged, range [1.0 ~ 30.0]
SS(1.1, -0.05):   9/19 converged, range [1.0 ~ 30.0]
SS(1.5, -0.25):   9/19 converged, range [0.5 ~ 20.0]
```

### Conclusion: Window SHIFTS, Doesn't WIDEN
ScaledSigmoid's larger gradient means:
- **Low LR end**: converges earlier (gradient larger → faster convergence) ✓
- **High LR end**: diverges earlier (gradient larger → more oscillation) ✗

The tolerance window shifts LEFT (toward lower LR), not wider. SS(1.5) gained 0.5 at the low end but lost 30→20 at the high end.

### User's Key Insight
> "ScaledSigmoid 因為在範圍內的 gradient 比較大, 所以應該要對 LR 更敏感, 導致 LR 容忍度較小才對"

This was correct. The scale factor amplifies both useful gradients AND harmful gradients equally.

---

## Phase 3: Multi-Layer Recovery (DemoRecoveryMultiLayer.py)

### Hypothesis (DISPROVEN)
"ScaledSigmoid's self-correction scales to multi-layer networks."

### Experiment Design
- Network: 1→8→8→8→1 (3 hidden layers)
- Task: Threshold (X > 0.5), 20 data points
- Reverse backprop (×-500) at epoch 0, normal training for 3000 epochs

### Results
All configs stuck at 0% recovery. The explosion magnitude was proportional to scale factor:
```
Sigmoid   Layer 3: explode → 40.2
SS(1.1)   Layer 3: explode → 48.6   ← ejected further!
SS(1.5)   Layer 3: explode → 89.4   ← ejected furthest!
```

### Conclusion
ScaledSigmoid's larger gradient means it gets **ejected further** during gradient explosion. In multi-layer networks:
1. The reverse backprop hits ScaledSigmoid harder (larger gradient → larger weight change)
2. All variants end up in deep saturation where gradient ≈ 0
3. The self-correction mechanism (output overshoot at boundaries) is too weak to overcome multi-layer gradient vanishing

### User's Verified Conclusion
> "由於 ScaledSigmoid 梯度較大, 所以會被 eject 比較遠, 也導致其實如果發生 gradient explode sigmoid 的效果其實比較好"

---

## Consolidated Findings: ScaledSigmoid Double-Edged Sword

| Scenario | Sigmoid | ScaledSigmoid | Winner |
|----------|---------|---------------|--------|
| Normal training convergence speed | Slow | Fast | **ScaledSigmoid** |
| Weight magnitude (steady state) | Large | Small (has equilibrium) | **ScaledSigmoid** |
| Single neuron boundary recovery | Cannot recover | Self-corrects | **ScaledSigmoid** |
| Gradient explosion resilience | Less damage (small grad) | More damage (large grad) | **Sigmoid** |
| High LR stability | More stable | More oscillation | **Sigmoid** |
| Multi-layer explosion recovery | Both stuck, but less damaged | Both stuck, more damaged | **Sigmoid** |
| LSTM gates | Correct [0,1] range | Harmful (exceeds [0,1]) | **Sigmoid** |

### The Only Unique Advantage
ScaledSigmoid's **only** advantage that sigmoid cannot match is the **output boundary self-correction** — the fact that output can exceed [0,1], causing gradient reversal. But this advantage:
- Works clearly in **single neurons** (DemoRecovery proven)
- Is **diluted** in multi-layer networks
- Is **harmful** in gating mechanisms (LSTM)
- Is **overwhelmed** during actual gradient explosions

### The Real Practical Benefit
ScaledSigmoid's practical value is **faster convergence** in normal training (proven in Main.py test cases, LeNet5, VGG). This comes from larger gradients in the non-saturated region, not from the boundary self-correction mechanism.

---

## Files Modified This Session

| File | Status | Description |
|------|--------|-------------|
| `DemoRecovery.py` | **Modified** | Rewritten: single neuron, reverse backprop epoch 0, bias frozen |
| `DemoLrTolerance.py` | **Created then deleted by user** | XOR LR sweep — hypothesis disproven |
| `DemoRecoveryMultiLayer.py` | **Created then deleted by user** | Multi-layer recovery — hypothesis disproven |

## Files NOT Modified

| File | Note |
|------|------|
| `Main.py` | Test cases 1-7 intact |
| `ScaledSigmoid.py` | No changes needed |
| `docs/analysis/scaled-sigmoid-recovery-mechanism.md` | Needs update: add "limitations" section based on new findings |

---

## Decisions Made

1. **Bias frozen in DemoRecovery** — Isolates weight behavior, prevents bias from compensating
2. **Two separate optimizers for reverse backprop** — User preferred `rev_optimizer` + `optimizer` over single optimizer with lr switching (more readable)
3. **Removed scale=2.0 config** — User: "2 太大了", kept Original, 1.01, 1.05, 1.1, 1.5
4. **SGD over Adam** — Adam's adaptive LR masks the gradient differences being studied
5. **XOR for LR tolerance** — Fast (4 data points), requires non-linear separation, clear convergence criterion

---

## Key Lessons Learned

1. **float32 precision kills gradients**: `sigmoid(100) = 1.0` exactly in float32. Any init pushing pre-activation beyond ~15 makes gradient negligibly small for ALL variants.

2. **Gradient direction > magnitude**: ScaledSigmoid's recovery mechanism works because output EXCEEDS target (direction reversal), not because gradient is larger (it's only scale× larger).

3. **Larger gradient is double-edged**: Helps convergence speed but hurts stability (LR sensitivity, explosion resilience). Cannot claim "better" without specifying the scenario.

4. **Single-neuron effects don't always scale**: The clean self-correction seen in single neurons is diluted in multi-layer networks where gradient signal must propagate through multiple saturated layers.

5. **cp950 encoding on Windows**: Avoid Unicode symbols in print statements; use ASCII alternatives.

---

## Next Session Starting Points

### Documentation Updates Needed
- Update `docs/analysis/scaled-sigmoid-recovery-mechanism.md` with new "Limitations" section:
  - LR tolerance: window shifts, doesn't widen
  - Multi-layer: self-correction diluted, gradient explosion makes ScaledSigmoid worse
  - Add the "double-edged sword" table
- Update `docs/LESSONS.md` with new lessons from this session

### Potential Experiments
- **#4 Gradient flow visualization**: Show per-layer gradient magnitude in a deep network during normal training (not explosion). This would demonstrate the convergence speed advantage more directly.
- **Convergence speed benchmarking**: Rigorous comparison of "epochs to reach X% accuracy" across architectures — this is ScaledSigmoid's proven strength.

### Open Question
The user's research narrative needs reframing: ScaledSigmoid is NOT about "recovery from explosion" — it's about **faster convergence** and **bounded weight growth during normal training**. The boundary self-correction is a theoretical curiosity that works in isolation but doesn't scale to practical networks.
