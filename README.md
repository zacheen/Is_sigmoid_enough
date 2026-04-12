# Is Sigmoid Enough?

**Authors:** Yu-Chang Shih | Zhidong Huang | Julian Zhou

A research project investigating the properties and limitations of the Scaled Sigmoid activation function, originally introduced in YOLOv4.

---

## Research Question

> Since Scaled Sigmoid can avoid weight explosion and alleviate gradient vanishing, why hasn't it completely replaced the standard Sigmoid?

Starting from YOLO v4's bounding box offset prediction, we systematically study where Scaled Sigmoid helps, where it doesn't, and why.

---

## Background

In YOLO v4, the model predicts an object's center point offset within a grid cell using Sigmoid, which maps values to [0, 1]. When the center point is extremely close to the grid boundary, Sigmoid must output a value near 1 — requiring extremely large weights and biases, leading to **weight explosion**.

**Scaled Sigmoid** fixes this by stretching the output range:

```
out = scale * sigmoid(x) + shift
```

With appropriate scale/shift values (e.g. `scale=1.1, shift=-0.05`), the function can reach 1 at a finite input (~3.7), fundamentally eliminating the need for infinitely large weights.

---

## Key Findings

### Advantages of Scaled Sigmoid

| Property | Mechanism |
|----------|-----------|
| **Avoids weight explosion** | Outputs exceeding 1 produce negative gradients, creating a hedging effect that constrains parameter growth |
| **Avoids gradient vanishing** | Parameters stay in the high-gradient middle region, away from saturation zones |
| **Faster convergence** | Steeper curve at the same initialization point means larger update strides |

### Where It Works: CNN

Directly replacing standard Sigmoid with Scaled Sigmoid in a CNN yields measurably better accuracy and smaller, faster-decreasing weights. Reason: CNNs can require boundary-adjacent outputs where the gradient hedging mechanism is beneficial.

### Where It Doesn't: LSTM

In LSTMs, Sigmoid acts as a **gate** (forget gate, input gate, etc.) controlling how much information to retain. Gate values naturally hover between 0.5–0.8 — the middle of the curve where standard Sigmoid already has ample gradient. Scaled Sigmoid adds no benefit because the boundary value problem simply doesn't exist in this context.

### The Trade-off: Learning Rate Sensitivity

Scaled Sigmoid's larger gradients make it more sensitive to learning rate. Too high an LR causes parameters to "eject" far from the optimum. While recovery is possible if parameters remain in an active gradient region, most eject events push parameters into the saturation zone where gradient vanishes entirely — making recovery impossible. This is the **sole major disadvantage** of Scaled Sigmoid.

---

## Repository Structure

```
.
├── ScaledSigmoid.py      # Scaled Sigmoid activation module
├── Main.py               # Test cases: simple threshold → XOR → binary encoding
├── DemoRecovery.py       # Recovery from weight explosion demo (single neuron)
├── LeNet5.py             # LeNet-5 CNN experiments
├── VggCifar10.py         # VGG on CIFAR-10
├── LstmSeqMnist.py       # LSTM on Sequential MNIST
├── LstmCharGen.py        # LSTM character generation
├── CustomLSTM.py         # Custom LSTM implementation
├── util.py               # Shared utilities
├── record/               # Experiment logs and result plots
├── res/                  # Resource files
└── docs/                 # Documentation and lessons learned
```

---

## Experiments

### Test Case 4 (XOR)
Designed to force weight explosion: the network must drastically distort its output space to fit the XOR pattern. Under standard Sigmoid, weights diverge unboundedly. Under Scaled Sigmoid, weights stabilize within a finite range.

> Floating-point caveat: for inputs ≥ ~40, standard Sigmoid already truncates to 1.0 due to precision limits, masking explosion in simpler test cases — XOR avoids this pitfall.

### Recovery Demo
A single-neuron experiment with a deliberately reversed gradient at epoch 0 (simulating a bad hyperparameter step). Demonstrates that:
- Scaled Sigmoid can recover when ejected into an active gradient region, but recovers slower than standard Sigmoid
- In most practical cases (larger eject), both fail to recover once parameters hit the saturation zone

---

## Usage

```python
from ScaledSigmoid import ScaledSigmoid
import torch.nn as nn

# Drop-in replacement for nn.Sigmoid
act = ScaledSigmoid(scale=1.1, shift=-0.05)

# Run test case experiments
python Main.py

# Run recovery demo
python DemoRecovery.py
```

---

## Conclusion

Scaled Sigmoid is not a universal drop-in replacement for standard Sigmoid. Its benefits are conditional:

- **Use it** when your activation function must produce boundary-adjacent values (e.g., coordinate offset prediction in object detection)
- **Skip it** when gate values are naturally centered (e.g., LSTM gates)
- **Tune your LR carefully** — the larger gradients that make Scaled Sigmoid converge faster also make it easier to overshoot
