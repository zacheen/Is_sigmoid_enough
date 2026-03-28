import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from CustomLSTM import CustomLSTM

# ==========================================
# Device Selection
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# Sequential MNIST with Custom LSTM
# ==========================================
# Feed MNIST images as sequences: 28 time steps of 28-pixel rows.
# The LSTM must remember spatial patterns across rows to classify the digit.
# This tests ScaledSigmoid in LSTM gates — sigmoid here controls
# gating (how much to remember/forget), NOT activation output.
#
# Key difference from CNN experiments:
#   CNN: sigmoid is activation function → ScaledSigmoid extends output range
#   LSTM: sigmoid is gate controller → ScaledSigmoid may DESTABILIZE gating
# ==========================================

class LstmSeqMnist(nn.Module):
    def __init__(self, input_size=28, hidden_size=128, num_classes=10,
                 scale=1.0, shift=0.0):
        super(LstmSeqMnist, self).__init__()
        self.lstm = CustomLSTM(input_size=input_size, hidden_size=hidden_size,
                               num_layers=1, scale=scale, shift=shift)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        # x: (batch, 1, 28, 28) from MNIST loader
        x = x.squeeze(1)          # (batch, 28, 28) — 28 time steps, 28 features
        output, _ = self.lstm(x)   # output: (batch, 28, 128)
        out = output[:, -1, :]     # last time step: (batch, 128)
        return self.classifier(out)

# ==========================================
# Data Loading
# ==========================================
def get_mnist_loaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# ==========================================
# Training and Tracking
# ==========================================
def train_and_track(model, train_loader, test_loader, epochs=50, lr=0.001):
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
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
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
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)
                total += Y_batch.size(0)
                correct += (predicted == Y_batch).sum().item()
        accuracy = correct / total
        test_acc_history.append(accuracy)

        # --- Weight magnitude (last FC layer) ---
        last_fc = model.classifier[-1]
        weight_mag = last_fc.weight.abs().mean().item()
        weight_mag_history.append(weight_mag)

        print(f"  Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | Test Acc: {accuracy:.4f} | Weight Mag: {weight_mag:.4f}")

    return train_loss_history, test_acc_history, weight_mag_history

# ==========================================
# Comparison
# ==========================================
def compare(train_loader, test_loader, epochs=50):
    configs = [
        ("Original Sigmoid",                         1.0,   0.0),
        # ("ScaledSigmoid(scale=1.01, shift=-0.005)",  1.01, -0.005),
        ("ScaledSigmoid(scale=1.05, shift=-0.025)",  1.05, -0.025),
        ("ScaledSigmoid(scale=1.1, shift=-0.05)",    1.1,  -0.05),
    ]

    results = {}
    for name, scale, shift in configs:
        print(f"\n--- Training: {name} ---")
        torch.manual_seed(42)
        model = LstmSeqMnist(scale=scale, shift=shift).to(device)
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
    ax1.set_title("Training Loss Convergence (LSTM Sequential MNIST)")
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

    print(f"Using device: {device}")
    print("Loading MNIST dataset...")
    train_loader, test_loader = get_mnist_loaders(batch_size=128)
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # 50 epochs — enough to see convergence on sequential MNIST
    # Custom LSTM is slower than nn.LSTM (Python loop vs C++ kernel)
    # Expected: ~45-60 min on CPU
    EPOCHS = 80

    results = compare(train_loader, test_loader, epochs=EPOCHS)
