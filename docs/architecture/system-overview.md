# System Architecture

## Overview

This is a PyTorch-based experimental framework for comparing the standard `nn.Sigmoid` activation function against a custom `ScaledSigmoid` activation. The goal is to determine whether scaling and shifting the sigmoid output improves training dynamics.

## Components

### ScaledSigmoid (`ScaledSigmoid.py`)
- Custom `nn.Module` implementing `scale * sigmoid(x) + shift`
- Inspired by YOLOv4's approach to "eliminating hard-to-reach boundaries"
- Configurable `scale` and `shift` parameters for experimentation
- Common configurations tested:
  - `scale=2.0, shift=-0.5` (output range: [-0.5, 1.5])
  - `scale=1.05, shift=-0.025`
  - `scale=1.1, shift=-0.05`

### SimpleNN (`Main.py`)
- A two-layer fully connected neural network (`fc1 -> activation -> fc2 -> activation`)
- Configurable input/hidden/output dimensions and activation function
- Uses a fixed seed (42) for reproducibility

### Training & Comparison (`Main.py`)
- `train_and_track()`: Trains a model using MSE loss, recording loss history, weight magnitude (FC2 mean absolute weight), and full weight snapshots per epoch
- `compare()`: Instantiates one original sigmoid model and three ScaledSigmoid variants, trains them all, and plots:
  1. Training loss convergence comparison
  2. Weight magnitude over time comparison
  3. Per-parameter weight history for each model

### Utility (`util.py`)
- `replace_sigmoid_with_modified()`: Recursively replaces all `nn.Sigmoid` modules in an existing model with `ScaledSigmoid` — useful for retrofitting pre-trained models

## Test Cases

| Case | Task | Input Distribution | Network Shape | Status |
|------|------|--------------------|---------------|--------|
| 0 | Odd/even classification | Uniform integers [0, 10000) | (1, ?, 1) | Failed — cannot achieve loss 0 |
| 1 | Threshold classification (X > 0.5) | Uniform [0, 1) | (1, 1, 1) | Success — notes on contradictory gradients with ScaledSigmoid |
| 2 | Threshold classification (X > 0.5) | Normal centered at 0.5 | (1, 1, 1) | Avoids stuck loss at 0.25 |
| 3 | Pulse wave (1 if -1 < X < 1) | Linspace [-5, 5] | (1, 3, 1) | Active test case — requires multi-node hidden layer |

## Output Artifacts

- `record/` — PNG screenshots of training results (loss curves, weight histories)
- `res/` — Research resources including ML proposal PDF and analytical graphs (intersection analysis)

## Running the Code

```bash
python Main.py
```

Requires: `torch`, `matplotlib`

## Design Patterns

- **Experiment-driven**: Each test case is a commented block in `__main__`; uncomment to activate
- **Side-by-side comparison**: All models trained with identical data, optimizer, and epochs
- **Reproducibility**: Fixed random seed (42) ensures consistent initialization
