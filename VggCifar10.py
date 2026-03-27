import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import copy
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from ScaledSigmoid import ScaledSigmoid

# ==========================================
# Define Small VGG-style Architecture (Sigmoid Version)
# ==========================================
# Why this architecture?
#   - 4 conv layers + 2 FC layers = deep enough for vanishing gradient to matter
#   - CIFAR-10 is significantly harder than MNIST (~70-85% with sigmoid vs ~97% on MNIST)
#   - The depth amplifies differences between sigmoid and ScaledSigmoid
#   - ~550K params — still trains in minutes on CPU
# ==========================================
class VggStyle(nn.Module):
    def __init__(self, activation_fn=None):
        super(VggStyle, self).__init__()
        if activation_fn is None:
            activation_fn = nn.Sigmoid()

        # Block 1: 3x32x32 -> 32x32x32 -> 32x32x32 -> 32x16x16
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),    # 32x32 -> 32x32
            copy.deepcopy(activation_fn),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),   # 32x32 -> 32x32
            copy.deepcopy(activation_fn),
            nn.MaxPool2d(kernel_size=2, stride=2),          # 32x32 -> 16x16
            nn.Dropout(0.25),
        )

        # Block 2: 32x16x16 -> 64x16x16 -> 64x16x16 -> 64x8x8
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),   # 16x16 -> 16x16
            copy.deepcopy(activation_fn),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),   # 16x16 -> 16x16
            copy.deepcopy(activation_fn),
            nn.MaxPool2d(kernel_size=2, stride=2),          # 16x16 -> 8x8
            nn.Dropout(0.25),
        )

        # Classifier: 64*8*8=4096 -> 256 -> 10
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256),
            copy.deepcopy(activation_fn),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ==========================================
# Data Loading
# ==========================================
def get_cifar10_loaders(batch_size=128):
    # Standard CIFAR-10 normalization
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# ==========================================
# Training and Tracking
# ==========================================
def train_and_track(model, train_loader, test_loader, epochs=30, lr=0.001):
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
        last_fc = model.classifier[-1]  # Linear(256, 10)
        weight_mag = last_fc.weight.abs().mean().item()
        weight_mag_history.append(weight_mag)

        print(f"  Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | Test Acc: {accuracy:.4f} | Weight Mag: {weight_mag:.4f}")

    return train_loss_history, test_acc_history, weight_mag_history

# ==========================================
# Comparison
# ==========================================
def compare(train_loader, test_loader, epochs=30):
    configs = [
        ("Original Sigmoid",                         nn.Sigmoid()),
        ("ScaledSigmoid(scale=1.01, shift=-0.005)",  ScaledSigmoid(scale=1.01, shift=-0.005)),
        ("ScaledSigmoid(scale=1.05, shift=-0.025)",  ScaledSigmoid(scale=1.05, shift=-0.025)),
        ("ScaledSigmoid(scale=1.1, shift=-0.05)",    ScaledSigmoid(scale=1.1, shift=-0.05)),
    ]

    results = {}
    for name, activation in configs:
        print(f"\n--- Training: {name} ---")
        torch.manual_seed(42)
        model = VggStyle(activation_fn=activation)
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
    ax1.set_title("Training Loss Convergence (VGG-style on CIFAR-10)")
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

    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_loaders(batch_size=128)
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # 30 epochs — enough to see convergence differences on CIFAR-10
    # Sigmoid struggles more on deeper networks with harder tasks,
    # so differences should be more pronounced than MNIST/LeNet-5
    EPOCHS = 30

    results = compare(train_loader, test_loader, epochs=EPOCHS)
