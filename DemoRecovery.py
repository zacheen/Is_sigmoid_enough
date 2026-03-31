import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from ScaledSigmoid import ScaledSigmoid

# ==========================================
# Demo: ScaledSigmoid RECOVERY from Exploded Weights
# ==========================================
# Simplest possible demo: SINGLE NEURON, no hidden layers.
#
#   y = activation(w * x + b),  x = 1.0,  target = 1.0
#
# Initialize w = 5 (simulating "exploded" weight).
# At w=5, pre-activation = 5:
#
#   Standard sigmoid:
#     output = sigmoid(5) = 0.993 < 1.0
#     → error = -0.007 → gradient pushes w UP → w keeps growing
#     → NEVER recovers, weight keeps exploding
#
#   ScaledSigmoid(1.1, -0.05):
#     output = 1.1 * sigmoid(5) - 0.05 = 1.043 > 1.0
#     → error = +0.043 → gradient pushes w DOWN → w decreases
#     → Equilibrium at w ≈ 3.04 where output = exactly 1.0
#     → RECOVERS to healthy weight!
#
# The key insight: it's not gradient magnitude, it's gradient DIRECTION.
# ScaledSigmoid OVERSHOOTS the target, creating a restoring force.
# ==========================================

INIT_WEIGHT = 8.0
LR = 32.0       # Large LR to amplify tiny gradients for visualization
EPOCHS = 1000

class SingleNeuron(nn.Module):
    def __init__(self, activation_fn):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.activation = activation_fn

    def forward(self, x):
        return self.activation(self.linear(x))


def train_single_neuron(activation_fn, x, target, init_w, init_b, lr, epochs):
    model = SingleNeuron(activation_fn)
    with torch.no_grad():
        model.linear.weight.fill_(init_w)
        model.linear.bias.fill_(init_b)
    model.linear.bias.requires_grad = False  # Freeze bias

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    w_history = []
    loss_history = []
    output_history = []
    grad_history = []

    for epoch in range(epochs):
        w_history.append(model.linear.weight.item())

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, target)
        loss.backward()

        grad_history.append(model.linear.weight.grad.item())
        optimizer.step()

        loss_history.append(loss.item())
        output_history.append(out.item())

    return w_history, loss_history, output_history, grad_history


if __name__ == "__main__":
    # Single data point: x=1, target=0.5
    x = torch.tensor([[1.0]])
    target = torch.tensor([[1.0]])

    # (name, activation, lr_override)
    # Scale=1.5 needs lower lr because its equilibrium (w≈1.6) has large gradient
    configs = [
        ("Original Sigmoid",            nn.Sigmoid(),                          LR),
        ("Scale=1.01, shift=-0.005",     ScaledSigmoid(scale=1.01, shift=-0.005), LR),
        ("Scale=1.05, shift=-0.025",     ScaledSigmoid(scale=1.05, shift=-0.025), LR),
        ("Scale=1.1, shift=-0.05",       ScaledSigmoid(scale=1.1, shift=-0.05),   LR),
        ("Scale=1.5, shift=-0.25",       ScaledSigmoid(scale=1.5, shift=-0.25),   5.0),  # Lower lr to prevent oscillation
    ]

    results = {}
    for name, act, config_lr in configs:
        w_hist, loss_hist, out_hist, grad_hist = train_single_neuron(
            act, x, target, init_w=INIT_WEIGHT, init_b=0.0, lr=config_lr, epochs=EPOCHS
        )
        results[name] = {
            'w': w_hist, 'loss': loss_hist, 'output': out_hist, 'grad': grad_hist
        }
        print(f"{name:35s} | w: {INIT_WEIGHT:.1f} → {w_hist[-1]:+.4f} | "
              f"output: {out_hist[-1]:.4f} | loss: {loss_hist[-1]:.6f} | "
              f"{'RECOVERED ↓' if w_hist[-1] < INIT_WEIGHT - 0.1 else 'STUCK/GROWING ↑'}")

    # Calculate theoretical equilibrium points
    print("\n--- Theoretical Equilibrium (where output = 1.0) ---")
    for name, act, _ in configs:
        if isinstance(act, ScaledSigmoid):
            # scale * sigmoid(w_eq) + shift = 1.0 → sigmoid(w_eq) = (1-shift)/scale
            sig_val = (1.0 - act.shift) / act.scale
            if 0 < sig_val < 1:
                w_eq = torch.log(torch.tensor(sig_val / (1 - sig_val))).item()
                print(f"  {name:35s} → w_eq = {w_eq:.4f}")
            else:
                print(f"  {name:35s} → no finite equilibrium")
        else:
            print(f"  {name:35s} → w_eq = +∞ (sigmoid never reaches 1.0)")

    # ==========================================
    # Plotting
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. WEIGHT TRAJECTORY — The key chart
    ax = axes[0, 0]
    for name, r in results.items():
        ax.plot(r['w'], label=name, linewidth=2)
    ax.axhline(y=INIT_WEIGHT, color='gray', linestyle='--', alpha=0.4, label=f'Init w={INIT_WEIGHT}')
    ax.set_title(f"Weight Recovery: init w={INIT_WEIGHT}", fontsize=12, fontweight='bold')
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Weight value")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. OUTPUT over time
    ax = axes[0, 1]
    for name, r in results.items():
        ax.plot(r['output'], label=name, linewidth=2)
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Target = 1.0')
    ax.set_title("Network Output (target = 1.0)")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Output")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. LOSS
    ax = axes[1, 0]
    for name, r in results.items():
        ax.plot(r['loss'], label=name, linewidth=2)
    ax.set_title("Loss (MSE)")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # 4. GRADIENT DIRECTION (first 500 epochs)
    ax = axes[1, 1]
    show_ep = min(500, EPOCHS)
    for name, r in results.items():
        ax.plot(r['grad'][:show_ep], label=name, linewidth=2)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_title(f"Weight Gradient (first {show_ep} epochs)")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("dL/dw\n(+: SGD pushes w down, −: pushes w up)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Recovery from Weight Explosion\n"
        f"Single neuron: y = act(w·x + b), x=1, target=1, init w={INIT_WEIGHT}, lr={LR}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig("record/DemoRecovery_result.png", dpi=150)
    plt.show()
