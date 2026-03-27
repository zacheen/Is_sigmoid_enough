import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import copy
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from ScaledSigmoid import ScaledSigmoid

# ==========================================
# Define LeNet-5 Architecture (Sigmoid Version)
# ==========================================
class LeNet5(nn.Module):
    def __init__(self, activation_fn=None):
        super(LeNet5, self).__init__()
        if activation_fn is None:
            activation_fn = nn.Sigmoid()

        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),   # 28x28 -> 28x28
            copy.deepcopy(activation_fn),
            nn.AvgPool2d(kernel_size=2, stride=2),        # 28x28 -> 14x14
            nn.Conv2d(6, 16, kernel_size=5),               # 14x14 -> 10x10
            copy.deepcopy(activation_fn),
            nn.AvgPool2d(kernel_size=2, stride=2),         # 10x10 -> 5x5
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            copy.deepcopy(activation_fn),
            nn.Linear(120, 84),
            copy.deepcopy(activation_fn),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ==========================================
# Data Loading
# ==========================================
def get_mnist_loaders(batch_size=256):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # MNIST images are [0,1] after ToTensor, no extra normalization
        # so sigmoid output range aligns naturally with input range
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# ==========================================
# Training and Tracking
# ==========================================
def train_and_track(model, train_loader, test_loader, epochs=10, lr=0.001):
    """
    Train model and track loss + accuracy per epoch,
    plus mean absolute weight magnitude of the last FC layer.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loss_history = []
    test_acc_history = []
    weight_mag_history = []

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        running_loss = 0.0
        num_batches = 0
        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            num_batches += 1

        avg_loss = running_loss / num_batches
        train_loss_history.append(avg_loss)

        # --- Evaluation ---
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, Y_batch in test_loader:
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)
                total += Y_batch.size(0)
                correct += (predicted == Y_batch).sum().item()
        accuracy = correct / total
        test_acc_history.append(accuracy)

        # --- Weight magnitude (last FC layer) ---
        last_fc = model.classifier[-1]  # Linear(84, 10)
        weight_mag = last_fc.weight.abs().mean().item()
        weight_mag_history.append(weight_mag)

        print(f"  Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | Test Acc: {accuracy:.4f} | Weight Mag: {weight_mag:.4f}")

    return train_loss_history, test_acc_history, weight_mag_history

# ==========================================
# Comparison
# ==========================================
def compare(train_loader, test_loader, epochs=10):
    configs = [
        ("Original Sigmoid",                    nn.Sigmoid()),
        ("ScaledSigmoid(scale=1.01, shift=-0.005)", ScaledSigmoid(scale=1.01, shift=-0.005)),
        ("ScaledSigmoid(scale=1.05, shift=-0.025)", ScaledSigmoid(scale=1.05, shift=-0.025)),
        ("ScaledSigmoid(scale=1.1, shift=-0.05)",   ScaledSigmoid(scale=1.1, shift=-0.05)),
    ]

    results = {}
    for name, activation in configs:
        print(f"\n--- Training: {name} ---")
        torch.manual_seed(42)
        model = LeNet5(activation_fn=activation)
        loss_hist, acc_hist, wmag_hist = train_and_track(model, train_loader, test_loader, epochs=epochs)
        results[name] = {
            'loss': loss_hist,
            'accuracy': acc_hist,
            'weight_mag': wmag_hist,
        }

    # --- Print final summary ---
    print("\n" + "=" * 60)
    print("Final Results Summary")
    print("=" * 60)
    for name, r in results.items():
        print(f"  {name:45s} | Loss: {r['loss'][-1]:.4f} | Acc: {r['accuracy'][-1]:.4f}")

    # --- Plotting ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 14))

    for name, r in results.items():
        ax1.plot(r['loss'], label=name)
    ax1.set_title("Training Loss Convergence (LeNet-5 on MNIST)")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("CrossEntropy Loss")
    ax1.legend()
    ax1.grid(True)

    for name, r in results.items():
        ax2.plot(r['accuracy'], label=name)
    ax2.set_title("Test Accuracy Over Time")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)

    for name, r in results.items():
        ax3.plot(r['weight_mag'], label=name)
    ax3.set_title("Weight Magnitude Over Time (Last FC Layer)")
    ax3.set_xlabel("Epochs")
    ax3.set_ylabel("Mean Absolute Weight")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

    return results

# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    torch.manual_seed(42)

    print("Loading MNIST dataset...")
    train_loader, test_loader = get_mnist_loaders(batch_size=256)
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # Number of epochs — 10 is enough to see convergence differences
    # Increase to 20-30 for more detailed comparison
    EPOCHS = 10

    results = compare(train_loader, test_loader, epochs=EPOCHS)
