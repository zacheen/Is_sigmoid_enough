import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from ScaledSigmoid import ScaledSigmoid

# ==========================================
# Demo: ScaledSigmoid RECOVERY from Weight Explosion
# ==========================================
# Single neuron: y = activation(w * x + b), bias frozen.
#
# Method:
#   1. Start with normal weight (init_w → output ≈ 0.6)
#   2. Epoch 0: ONE reverse backprop (lr * -10) → weight explodes
#   3. Epoch 1+: Normal training → does it recover?
#
# After explosion, weight is large → output saturated:
#
#   Standard sigmoid:
#     output < target always (sigmoid asymptotically approaches but never reaches target)
#     → gradient keeps pushing w UP → weight stays stuck or grows further
#
#   ScaledSigmoid:
#     output > target (overshoots due to scale > 1)
#     → gradient REVERSES → pushes w DOWN → weight recovers!
#
# Key insight: it's not gradient magnitude, it's gradient DIRECTION.
# ScaledSigmoid overshoots the target, creating a restoring force.
# ==========================================

INIT_WEIGHT = 0.8
LR = 10.0       # Large LR to amplify tiny gradients for visualization
EPOCHS = 1300

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
    rev_optimizer = optim.SGD(model.parameters(), lr=lr*-30)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    w_history = []
    loss_history = []
    output_history = []
    grad_history = []

    for epoch in range(epochs):
        w_history.append(model.linear.weight.item())

        if epoch == 0 :
            rev_optimizer.zero_grad()
        else :
            optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, target)
        loss.backward()

        grad_history.append(model.linear.weight.grad.item())
        if epoch == 0 :
            rev_optimizer.step()
        else :
            optimizer.step()

        loss_history.append(loss.item())
        output_history.append(out.item())

    return w_history, loss_history, output_history, grad_history


if __name__ == "__main__":
    # Single data point: x=1, target=0.5
    x = torch.tensor([[1.0]])
    target = torch.tensor([[0.6]])

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
        exploded_w = w_hist[1] if len(w_hist) > 1 else w_hist[0]
        final_w = w_hist[-1]
        print(f"{name:35s} | w: {INIT_WEIGHT:.2f} → [explode:{exploded_w:+.2f}] → {final_w:+.4f} | "
              f"output: {out_hist[-1]:.4f} | loss: {loss_hist[-1]:.6f} | "
              f"{'RECOVERED ↓' if final_w < exploded_w - 0.1 else 'STUCK/GROWING ↑'}")

    # Calculate theoretical equilibrium points
    target_val = target.item()
    print(f"\n--- Theoretical Equilibrium (where output = {target_val:.2f}) ---")
    for name, act, _ in configs:
        if isinstance(act, ScaledSigmoid):
            # scale * sigmoid(w_eq) + shift = target → sigmoid(w_eq) = (target-shift)/scale
            sig_val = (target_val - act.shift) / act.scale
            if 0 < sig_val < 1:
                w_eq = torch.log(torch.tensor(sig_val / (1 - sig_val))).item()
                print(f"  {name:35s} → w_eq = {w_eq:.4f}")
            else:
                print(f"  {name:35s} → no finite equilibrium")
        else:
            # sigmoid(w_eq) = target → w_eq = ln(target/(1-target))
            if 0 < target_val < 1:
                w_eq = torch.log(torch.tensor(target_val / (1 - target_val))).item()
                print(f"  {name:35s} → w_eq = {w_eq:.4f}")
            else:
                print(f"  {name:35s} → w_eq = ±∞")

    # ==========================================
    # Plotting
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. WEIGHT TRAJECTORY — The key chart
    ax = axes[0, 0]
    for name, r in results.items():
        ax.plot(r['w'], label=name, linewidth=2)
    ax.axhline(y=INIT_WEIGHT, color='gray', linestyle='--', alpha=0.4, label=f'Init w={INIT_WEIGHT}')
    ax.axvline(x=1, color='red', linestyle=':', alpha=0.5, label='Reverse backprop')
    ax.set_title(f"Weight: init={INIT_WEIGHT} → explode → recover?", fontsize=12, fontweight='bold')
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Weight value")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. OUTPUT over time
    ax = axes[0, 1]
    for name, r in results.items():
        ax.plot(r['output'], label=name, linewidth=2)
    ax.axhline(y=target.item(), color='black', linestyle='--', alpha=0.5, label=f'Target = {target.item():.2f}')
    ax.axvline(x=1, color='red', linestyle=':', alpha=0.5)
    ax.set_title(f"Network Output (target = {target.item():.2f})")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Output")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. LOSS
    ax = axes[1, 0]
    for name, r in results.items():
        ax.plot(r['loss'], label=name, linewidth=2)
    ax.axvline(x=1, color='red', linestyle=':', alpha=0.5)
    ax.set_title("Loss (MSE)")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # 4. GRADIENT DIRECTION
    ax = axes[1, 1]
    for name, r in results.items():
        ax.plot(r['grad'], label=name, linewidth=2)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.axvline(x=1, color='red', linestyle=':', alpha=0.5)
    ax.set_title("Weight Gradient")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("dL/dw\n(+: SGD pushes w down, −: pushes w up)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Recovery from Weight Explosion (epoch 0: reverse backprop lr×-30)\n"
        f"Single neuron: y = act(w·x + b), x=1, target={target.item():.2f}, init w={INIT_WEIGHT}, lr={LR}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig("record/DemoRecovery_result.png", dpi=150)
    plt.show()
