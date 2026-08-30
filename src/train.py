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
EPOCHS = 50
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")

# --- Data ---
train_ds, val_ds, test_ds = load_datasets(LABELS_FILE, KEYPOINTS_DIR)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# -- Class weights for CrossEntropyLoss based on distribution check
#weights = torch.tensor([0.823, 3.115, 0.683], dtype=torch.float32).to(device)  # L, C, R order

# More balanced weights — still upweights C but not as aggressively
weights = torch.tensor([1.0, 2.0, 0.8], dtype=torch.float32).to(device)

# -- Model, Loss, Optimizer --
model = PenaltyLSTM(
    input_size=8,   # <-- changed from 30
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    num_classes=3,
    dropout=DROPOUT
).to(device)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# Monitor validation accuracy (higher is better). Remove `verbose` for compatibility.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

# -- Training Loop --
best_val_acc = 0.0

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

    print(f"Epoch {epoch+1:02d}/{EPOCHS} |"
          f"Loss: {avg_loss:.4f}  |"
          f"Train Acc: {train_acc:.1%} |"
          f"Val Acc: {val_acc:.1%}")

    scheduler.step(val_acc)

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_acc": val_acc,
            "config": {
                "input_size": 30,
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "num_classes": 3,
                "dropout": DROPOUT
            }
        }, os.path.join(MODELS_DIR, "best_model.pth"))
        print(f"Saved new best model with val acc: {val_acc:.1%}")

print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.1%}")
