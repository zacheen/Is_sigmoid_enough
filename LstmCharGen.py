import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import urllib.request
from torch.utils.data import Dataset, DataLoader
from CustomLSTM import CustomLSTM

# ==========================================
# Device Selection
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# Character-Level LSTM Text Generation
# ==========================================
# Train on Shakespeare text, predict next character.
# LSTM gates must learn complex temporal patterns in language —
# when to remember/forget context across sentences and dialogue.
#
# This is a generative task (not classification), so we track:
#   1. Training loss (CrossEntropy on next-char prediction)
#   2. Character prediction accuracy (top-1 next-char accuracy)
#   3. Weight magnitude (last FC layer)
# Plus qualitative text generation samples after training.
# ==========================================

# ==========================================
# Shakespeare Dataset
# ==========================================
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = "./data"
SHAKESPEARE_PATH = os.path.join(DATA_DIR, "shakespeare.txt")

class ShakespeareDataset(Dataset):
    def __init__(self, text, char_to_idx, seq_length=100):
        self.seq_length = seq_length
        self.char_to_idx = char_to_idx
        # Encode entire text as integer indices
        self.data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)

    def __len__(self):
        return (len(self.data) - 1) // self.seq_length

    def __getitem__(self, idx):
        start = idx * self.seq_length
        end = start + self.seq_length
        x = self.data[start:end]          # input: chars [0..seq_len-1]
        y = self.data[start+1:end+1]      # target: chars [1..seq_len]
        return x, y

def get_shakespeare_data(seq_length=100, batch_size=64):
    """Download Shakespeare text and create DataLoader."""
    os.makedirs(DATA_DIR, exist_ok=True)

    # Download if not cached
    if not os.path.exists(SHAKESPEARE_PATH):
        print("Downloading Shakespeare text...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, SHAKESPEARE_PATH)
        print("Download complete.")

    with open(SHAKESPEARE_PATH, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"Text length: {len(text)} characters")

    # Build vocabulary
    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    vocab_size = len(chars)
    print(f"Vocabulary size: {vocab_size} unique characters")

    dataset = ShakespeareDataset(text, char_to_idx, seq_length=seq_length)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader, vocab_size, char_to_idx, idx_to_char

# ==========================================
# Model
# ==========================================
class LstmCharGen(nn.Module):
    def __init__(self, vocab_size, embed_size=64, hidden_size=256, num_layers=2,
                 scale=1.0, shift=0.0):
        super(LstmCharGen, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = CustomLSTM(input_size=embed_size, hidden_size=hidden_size,
                               num_layers=num_layers, scale=scale, shift=shift)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, vocab_size),
        )

    def forward(self, x, state=None):
        # x: (batch, seq_length) — integer character indices
        embedded = self.embedding(x)           # (batch, seq_length, embed_size)
        output, state = self.lstm(embedded, state)  # (batch, seq_length, hidden_size)
        logits = self.classifier(output)       # (batch, seq_length, vocab_size)
        return logits, state

# ==========================================
# Training and Tracking
# ==========================================
def train_and_track(model, train_loader, epochs=30, lr=0.002):
    """
    Train model and track loss + char prediction accuracy per epoch,
    plus weight magnitude of the last FC layer.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loss_history = []
    train_acc_history = []
    weight_mag_history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()
            logits, _ = model(X_batch)

            # Reshape for CrossEntropyLoss: (batch*seq_len, vocab) vs (batch*seq_len)
            batch_size, seq_len, vocab_size = logits.size()
            loss = criterion(logits.view(-1, vocab_size), Y_batch.view(-1))
            loss.backward()
            # Gradient clipping to prevent explosion (common in RNN training)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

            # Top-1 character prediction accuracy
            _, predicted = torch.max(logits.view(-1, vocab_size), 1)
            total += Y_batch.numel()
            correct += (predicted == Y_batch.view(-1)).sum().item()

        avg_loss = running_loss / num_batches
        accuracy = correct / total
        train_loss_history.append(avg_loss)
        train_acc_history.append(accuracy)

        # Weight magnitude (last FC layer)
        last_fc = model.classifier[-1]
        weight_mag = last_fc.weight.abs().mean().item()
        weight_mag_history.append(weight_mag)

        print(f"  Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | Char Acc: {accuracy:.4f} | Weight Mag: {weight_mag:.4f}")

    return train_loss_history, train_acc_history, weight_mag_history

# ==========================================
# Text Generation
# ==========================================
def generate_sample(model, char_to_idx, idx_to_char, seed_text="ROMEO:", length=200, temperature=0.8):
    """Generate text from trained model for qualitative comparison."""
    model.eval()
    vocab_size = len(char_to_idx)

    # Encode seed text
    input_ids = torch.tensor([[char_to_idx.get(c, 0) for c in seed_text]], dtype=torch.long).to(device)
    state = None
    generated = list(seed_text)

    with torch.no_grad():
        # Feed seed characters to build up hidden state
        logits, state = model(input_ids, state)

        # Generate new characters one at a time
        next_input = input_ids[:, -1:]  # last character
        for _ in range(length):
            logits, state = model(next_input, state)
            # Apply temperature scaling
            probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_char_idx = torch.multinomial(probs, 1)
            generated.append(idx_to_char[next_char_idx.item()])
            next_input = next_char_idx

    return ''.join(generated)

# ==========================================
# Comparison
# ==========================================
def compare(train_loader, vocab_size, char_to_idx, idx_to_char, epochs=30):
    configs = [
        ("Original Sigmoid",                         1.0,   0.0),
        # ("ScaledSigmoid(scale=1.01, shift=-0.005)",  1.01, -0.005),
        ("ScaledSigmoid(scale=1.05, shift=-0.025)",  1.05, -0.025),
        ("ScaledSigmoid(scale=1.1, shift=-0.05)",    1.1,  -0.05),
    ]

    results = {}
    models = {}
    for name, scale, shift in configs:
        print(f"\n--- Training: {name} ---")
        torch.manual_seed(42)
        model = LstmCharGen(vocab_size=vocab_size, scale=scale, shift=shift).to(device)
        loss_hist, acc_hist, wmag_hist = train_and_track(model, train_loader, epochs=epochs)
        results[name] = {
            'loss': loss_hist,
            'accuracy': acc_hist,
            'weight_mag': wmag_hist,
        }
        models[name] = model

    # --- Print final summary ---
    print("\n" + "=" * 60)
    print("Final Results Summary")
    print("=" * 60)
    for name, r in results.items():
        print(f"  {name:45s} | Loss: {r['loss'][-1]:.4f} | Char Acc: {r['accuracy'][-1]:.4f}")

    # --- Generate text samples ---
    print("\n" + "=" * 60)
    print("Generated Text Samples (seed: 'ROMEO:')")
    print("=" * 60)
    for name, model in models.items():
        sample = generate_sample(model, char_to_idx, idx_to_char)
        print(f"\n--- {name} ---")
        print(sample[:300])
        print("...")

    # --- Plotting ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 14))

    for name, r in results.items():
        ax1.plot(r['loss'], label=name)
    ax1.set_title("Training Loss Convergence (LSTM Character Generation)")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("CrossEntropy Loss")
    ax1.legend()
    ax1.grid(True)

    for name, r in results.items():
        ax2.plot(r['accuracy'], label=name)
    ax2.set_title("Character Prediction Accuracy Over Time")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Top-1 Accuracy")
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
    print("Loading Shakespeare dataset...")
    train_loader, vocab_size, char_to_idx, idx_to_char = get_shakespeare_data(
        seq_length=100, batch_size=64
    )
    print(f"Train batches: {len(train_loader)}")

    # 30 epochs on Shakespeare with 2-layer LSTM
    # Custom LSTM is slower than nn.LSTM (Python loop vs C++ kernel)
    # Expected: ~50-70 min on CPU
    EPOCHS = 60

    results = compare(train_loader, vocab_size, char_to_idx, idx_to_char, epochs=EPOCHS)
