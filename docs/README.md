# Sigmoid vs ScaledSigmoid — Project Reference

## Quick Start

```bash
# FC network experiments (threshold, pulse wave)
python Main.py

# CNN experiment (LeNet-5 on MNIST — simple baseline)
python LeNet5.py

# CNN experiment (VGG-style on CIFAR-10 — deeper, harder task)
python VggCifar10.py
```

**Requirements:** `torch`, `matplotlib`, `torchvision`

## Project Structure

```
├── Main.py              # FC network experiments — training loop, comparison & plotting
├── LeNet5.py            # CNN experiment — LeNet-5 on MNIST (simple baseline)
├── VggCifar10.py        # CNN experiment — VGG-style on CIFAR-10 (deeper, harder task)
├── ScaledSigmoid.py     # Custom activation: scale * sigmoid(x) + shift
├── util.py              # Utility to replace sigmoid in existing models
├── record/              # Saved experiment result screenshots (PNGs)
├── res/                 # Research resources (ML proposal PDF, analytical graphs)
└── docs/                # Documentation
    ├── README.md         # This file — project reference index
    ├── LESSONS.md        # Session log and lessons learned
    └── architecture/
        └── system-overview.md  # System architecture, components, test cases, design patterns
```

## Key Documentation

| Topic | Location |
|-------|----------|
| System architecture & components | [docs/architecture/system-overview.md](architecture/system-overview.md) |
| Test case descriptions & results | [docs/architecture/system-overview.md#test-cases](architecture/system-overview.md#test-cases) |
| Lessons learned | [docs/LESSONS.md](LESSONS.md) |
| ML research proposal | [res/ML_proposal.pdf](../res/ML_proposal.pdf) |

## Core Concepts

- **ScaledSigmoid**: `scale * sigmoid(x) + shift` — extends sigmoid output range to reduce boundary saturation
- **Comparison framework**: Trains original sigmoid and multiple ScaledSigmoid variants side-by-side, plotting loss convergence and weight magnitude
- **Test cases**:
  - FC networks: Cases 0-3 (threshold, pulse wave) in `Main.py` — simple saturation patterns
  - **New FC networks: Cases 4-7** (XOR, binary encoding, staircase, checkerboard) in `Main.py` — diverse saturation geometries
  - CNNs: LeNet-5 on MNIST, VGG-style on CIFAR-10
- **LeNet-5**: Classic CNN with all-sigmoid activations (~60K params), trained on MNIST — baseline CNN comparison (too simple to show differences)
- **VGG-style CNN**: Deeper 4-conv + 2-FC network (~550K params) on CIFAR-10 — harder task amplifies vanishing gradient differences

## Running Experiments

### FC Network Experiments (Main.py)

All FC test cases are defined as commented blocks in `Main.py`'s `__main__` section. To run a specific test:
1. Comment out the currently active test case
2. Uncomment your desired test case (lines define X, Y, node_size, optimizer)
3. Run `python Main.py`

Each test case will:
- Train 4 models side-by-side (original sigmoid + 3 ScaledSigmoid variants)
- Plot loss convergence, accuracy/weight magnitude, and per-parameter weight history
- Print final loss/weight summary

### CNN Experiments

Run LeNet-5 or VGG experiments independently:
```bash
python LeNet5.py    # ~2-3 min on CPU, 100 epochs default
python VggCifar10.py # ~15-20 min on CPU, 30 epochs default
```

## Configuration

Experiment parameters:
- **Test case data**: X, Y tensors defined in `Main.py`
- **Network shape**: `node_size` tuple (input_dim, hidden_dim, output_dim)
- **Optimizer**: `optim.SGD` or `optim.Adam`
- **Training epochs**: Set per test case (Main.py default ~20K, LeNet5=100, VGG=30)
- **ScaledSigmoid parameters**: Configured in `compare()` function — default is (scale, shift) pairs: (1.01, -0.005), (1.05, -0.025), (1.1, -0.05)
