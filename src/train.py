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
weights = torch.tensor([0.823, 3.115, 0.683], dtype=torch.float32).to(device)  # L, C, R order

# -- Model, Loss, Optimizer --
model = PenaltyLSTM(
    input_size=30,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    num_classes=3,
    dropout=DROPOUT
).to(device)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# -- Training Loop --
