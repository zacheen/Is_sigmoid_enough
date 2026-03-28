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

### LeNet-5 CNN (`LeNet5.py`)
- Classic CNN architecture originally designed with sigmoid activations
- Structure: `Conv2d(1,6,5) → Sigmoid → AvgPool → Conv2d(6,16,5) → Sigmoid → AvgPool → FC(400,120) → Sigmoid → FC(120,84) → Sigmoid → FC(84,10)`
- ~60K parameters — trains in minutes on CPU
- Dataset: MNIST (28×28 grayscale handwritten digits, 10 classes)
- Loss: CrossEntropyLoss (classification task, unlike the MSE used in SimpleNN)
- All hidden activations are `nn.Sigmoid`, making it a direct target for `replace_sigmoid_with_modified()`
- Extends the experiment from simple FC networks to convolutional architectures

### VGG-style CNN (`VggCifar10.py`)
- Small VGG-style architecture designed to stress-test sigmoid in a deeper network
- Structure: `Conv(3,32,3) → Sigmoid → Conv(32,32,3) → Sigmoid → MaxPool → Dropout → Conv(32,64,3) → Sigmoid → Conv(64,64,3) → Sigmoid → MaxPool → Dropout → FC(4096,256) → Sigmoid → Dropout → FC(256,10)`
- ~550K parameters — still trains in minutes on CPU
- Dataset: CIFAR-10 (32×32 RGB, 10 classes) with data augmentation (random flip, random crop)
- Loss: CrossEntropyLoss
- 4 conv + 2 FC layers — deep enough for vanishing gradient to matter with sigmoid
- CIFAR-10 is significantly harder than MNIST, so differences between sigmoid and ScaledSigmoid should be more pronounced

### Custom LSTM (`CustomLSTM.py`)
- Custom LSTM implementation that replaces hardcoded sigmoid gates with ScaledSigmoid
- PyTorch's `nn.LSTM` uses sigmoid internally in C++ — cannot be replaced via module swapping
- `ScaledSigmoidLSTMCell`: single cell with `scale * sigmoid(x) + shift` for input/forget/output gates, tanh for cell candidate
- `CustomLSTM`: multi-layer wrapper with same interface as `nn.LSTM(batch_first=True)`
- When `scale=1.0, shift=0.0`, behaves identically to standard LSTM
- **Key difference from CNN/FC experiments**: sigmoid here is a gate controller [0,1], not an activation function — extending beyond [0,1] may destabilize gating

### Sequential MNIST LSTM (`LstmSeqMnist.py`)
- Feeds MNIST as 28 time steps of 28-pixel rows into CustomLSTM
- Model: `CustomLSTM(28, 128, 1 layer) → Linear(128, 10)`
- ~50 epochs, Adam, CrossEntropyLoss, batch_size=128
- ~45-60 min on CPU
- Tests whether ScaledSigmoid gates help or hurt on a sequence classification task

### Shakespeare Character Generation LSTM (`LstmCharGen.py`)
- Character-level next-char prediction on tiny-shakespeare (~1MB)
- Model: `Embedding(vocab, 64) → CustomLSTM(64, 256, 2 layers) → Linear(256, vocab)`
- 30 epochs, Adam, CrossEntropyLoss, seq_length=100, batch_size=64
- ~50-70 min on CPU
- Tracks loss, char prediction accuracy, weight magnitude + generates text samples
- 2-layer LSTM tests whether ScaledSigmoid gate effects compound with depth

### Utility (`util.py`)
- `replace_sigmoid_with_modified()`: Recursively replaces all `nn.Sigmoid` modules in an existing model with `ScaledSigmoid` — useful for retrofitting pre-trained models

## Test Cases

| Case | Task | Input Distribution | Network Shape | Status |
|------|------|--------------------|---------------|--------|
| 0 | Odd/even classification | Uniform integers [0, 10000) | (1, ?, 1) | Failed — cannot achieve loss 0 |
| 1 | Threshold classification (X > 0.5) | Uniform [0, 1) | (1, 1, 1) | Success — notes on contradictory gradients with ScaledSigmoid |
| 2 | Threshold classification (X > 0.5) | Normal centered at 0.5 | (1, 1, 1) | Avoids stuck loss at 0.25 |
| 3 | Pulse wave (1 if -1 < X < 1) | Linspace [-5, 5] | (1, 3, 1) | Active test case — requires multi-node hidden layer |
| 4 | XOR (2D non-linear separability) | 4 discrete points {0,1}² | (2, 2, 1) | Forces saturation at all 4 outputs, non-linear boundary |
| 5 | Decimal→Binary encoding (multi-output) | Integers 0-7 | (1, 8, 3) | 3 output sigmoids must independently saturate |
| 6 | Staircase (multiple disjoint pulses) | Linspace [-5, 5] | (1, 6, 1) | 3 boundaries, 2 disjoint on-regions |
| 7 | 2D Checkerboard (continuous XOR) | 50×50 grid over [0, 2]² | (2, 4, 1) | Dense 2D XOR, 2500 points |
| — | MNIST digit classification (CNN) | MNIST dataset (28×28 grayscale) | LeNet-5 | Complete — too simple, ~97% for all variants |
| — | CIFAR-10 classification (CNN) | CIFAR-10 dataset (32×32 RGB) | VGG-style (4 conv + 2 FC) | Active — deeper network on harder task |
| — | Sequential MNIST (LSTM) | MNIST as 28×28 row sequences | LSTM(28→128, 1 layer) | Tests ScaledSigmoid in LSTM gates |
| — | Shakespeare char generation (LSTM) | tiny-shakespeare text | LSTM(64→256, 2 layers) | Generative task, 2-layer gate effects |

## Output Artifacts

- `record/` — PNG screenshots of training results (loss curves, weight histories)
- `res/` — Research resources including ML proposal PDF and analytical graphs (intersection analysis)

## Running the Code

```bash
python Main.py        # FC network experiments (threshold, pulse wave)
python LeNet5.py      # LeNet-5 on MNIST (simple, fast)
python VggCifar10.py  # VGG-style on CIFAR-10 (deeper, harder)
python LstmSeqMnist.py  # LSTM on Sequential MNIST (~50 min)
python LstmCharGen.py   # LSTM on Shakespeare text (~60 min)
```

Requires: `torch`, `matplotlib`, `torchvision` (for MNIST/LeNet-5 experiment)

## Design Patterns

- **Experiment-driven**: Each test case is a commented block in `__main__`; uncomment to activate
- **Side-by-side comparison**: All models trained with identical data, optimizer, and epochs
- **Reproducibility**: Fixed random seed (42) ensures consistent initialization
