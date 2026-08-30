# src/train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os

from dataset import load_datasets
from model import PenaltyLSTM

# -- Config --
LABELS_FILE = "data/labels.csv"
KEYPOINTS_DIR = "data/keypoints"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 3e-4      # lower than before for fine-tuning
PATIENCE = 15
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")

# --- Data ---
train_ds, val_ds, test_ds = load_datasets(LABELS_FILE, KEYPOINTS_DIR)

# Count actual classes in training set and use loss re-weighting only
from collections import Counter
train_labels = [int(train_ds[i][1].item()) for i in range(len(train_ds))]
counts = Counter(train_labels)
print(f"Training class counts: L={counts[0]}, C={counts[1]}, R={counts[2]}")

# OPTION A: Use loss weights only (no sampler)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# Compute weights from actual training counts
total_train = len(train_labels)
n_classes = 3
weights = torch.tensor([
    total_train / (n_classes * counts[0]),
    total_train / (n_classes * counts[1]),
    total_train / (n_classes * counts[2]),
], dtype=torch.float32).to(device)
print(f"Class weights: L={weights[0]:.3f}, C={weights[1]:.3f}, R={weights[2]:.3f}")

# -- Model, Loss, Optimizer --
model = PenaltyLSTM(
    input_size=8, # I think that its still 8 not 30
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    num_classes=3,
    dropout=DROPOUT
).to(device)

# Use loss weighting only (no label smoothing for now)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# Monitor validation F1 (higher is better). Be more aggressive reducing LR when stuck.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.3, patience=5, min_lr=1e-6
)

# -- Training Loop --
best_val_acc = 0.0
best_f1 = 0.0
epochs_no_improve = 0

# If a best checkpoint exists, load it to continue training from that point
checkpoint_path = os.path.join(MODELS_DIR, "best_model.pth")
if os.path.exists(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", model.state_dict()))
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    best_f1 = ckpt.get("f1", best_f1)
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', 'N/A')} with F1 {best_f1:.3f}")

def macro_f1(preds, labels, n_classes=3):
    f1s = []
    preds = np.array(preds)
    labels = np.array(labels)
    for c in range(n_classes):
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1s.append(f1)
    return float(np.mean(f1s))

for epoch in range(EPOCHS):
    # Train
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0

    for kp, labels in train_loader:
        kp, labels = kp.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(kp)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping to prevent exploding gradients in LSTM
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        train_loss += loss.item()
        preds = logits.argmax(dim=1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    # Validate
    model.eval()
    val_correct, val_total = 0, 0
    val_preds_all, val_labels_all = [], []

    with torch.no_grad():
        for kp, labels in val_loader:
            kp, labels = kp.to(device), labels.to(device)
            logits = model(kp)
            preds = logits.argmax(dim=1)
            # Count correct predictions, not sum of label indices
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            val_preds_all.extend(preds.cpu().numpy())
            val_labels_all.extend(labels.cpu().numpy())

    train_acc = train_correct / train_total
    val_acc = val_correct / val_total
    avg_loss = train_loss / len(train_loader)

    # Per class accuracy
    val_preds_all = np.array(val_preds_all)
    val_labels_all = np.array(val_labels_all)
    for cls, name in enumerate(['L', 'C', 'R']):
        mask = val_labels_all == cls
        if mask.sum() > 0:
            cls_acc = (val_preds_all[mask] == cls).mean()
            print(f"  {name}: {cls_acc:.1%}", end="   ")
    print()

    # Macro F1 on validation set
    f1 = macro_f1(val_preds_all, val_labels_all)
    print(f"  Macro F1: {f1:.3f}")

    print(f"Epoch {epoch+1:02d}/{EPOCHS} |"
          f"Loss: {avg_loss:.4f}  |"
          f"Train Acc: {train_acc:.1%} |"
          f"Val Acc: {val_acc:.1%}")

    # Step scheduler on F1 (we're optimizing for F1)
    scheduler.step(f1)

    # Save best model based on Macro F1 and implement early stopping
    if f1 > best_f1:
        best_f1 = f1
        epochs_no_improve = 0
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
            "f1": f1,
            "config": {
                "input_size": 8,
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "num_classes": 3,
                "dropout": DROPOUT
            }
        }, os.path.join(MODELS_DIR, "best_model.pth"))
        print(f"  ✓ Saved best model — F1: {f1:.3f}, Val Acc: {val_acc:.1%}")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1} — no improvement for {PATIENCE} epochs")
            break

print(f"\nTraining complete. Best F1: {best_f1:.3f}")
