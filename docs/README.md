# Sigmoid vs ScaledSigmoid — Project Reference

## Quick Start

```bash
# FC network experiments (threshold, pulse wave)
python Main.py

# CNN experiment (LeNet-5 on MNIST)
python LeNet5.py
```

**Requirements:** `torch`, `matplotlib`, `torchvision`

## Project Structure

```
├── Main.py              # FC network experiments — training loop, comparison & plotting
├── LeNet5.py            # CNN experiment — LeNet-5 on MNIST, sigmoid vs ScaledSigmoid
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
- **Test cases**: FC cases defined in `Main.py` `__main__` block (toggle by commenting/uncommenting); CNN case in `LeNet5.py`
- **LeNet-5**: Classic CNN with all-sigmoid activations (~60K params), trained on MNIST — extends comparison to convolutional architectures

## Configuration

All experiment parameters are set in `Main.py`'s `__main__` block:
- **Test case data** (X, Y tensors)
- **Network shape** (`node_size` tuple)
- **Optimizer** (`optim.SGD` or `optim.Adam`)
- **ScaledSigmoid parameters** are set in the `compare()` function
