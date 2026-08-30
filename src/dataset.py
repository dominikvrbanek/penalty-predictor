#dataset.py
# The purpose of this file is to standardize the length of the video clips to 3 seconds, 
# and to ensure that the clips are centered around the penalty kick event. 
# This is done by trimming the clips to 3 seconds, and then padding them with black frames if they are shorter 
# than 3 seconds. The clips are then saved to a new directory, which is specified in the config file.

import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset

LABEL_MAP = {
    'L': 0,
    'C': 1,
    'R': 2
}
SEQUENCE_LENGTH = 60  # 2 seconds at 30 fps or try 90 for 3 seconds at 30 fps

class PenaltyDataset(Dataset):
    def __init__(self, rows, keypoints_dir):
        self.samples = []

        for row in rows:
            clip_id = row["clip_id"]
            direction = row["direction"].strip().upper()

            if direction not in LABEL_MAP:
                continue

            kp_path = os.path.join(keypoints_dir, f"keypoints_{clip_id}.json.npy")
            if not os.path.exists(kp_path):
                continue

            self.samples.append({
                "kp_path": kp_path,
                "label": LABEL_MAP[direction],
                "clip_id": clip_id
            })

        print(f"Dataset loaded: {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        kp = np.load(sample["kp_path"]) # shape: (T, 10, 3)

        # Normalize x/y coordinates to 0-1 range using max value per clip
        # Confidence scores (every 3rd value) are already in 0-1 range
        xy_max = kp[:, :, :2].max(axis=(0, 1), keepdims=True)
        if xy_max > 0:
            kp[:, :, :2] /= xy_max

        # Flatten from (T, 10, 3) to (T, 30)
        kp_flat = kp.reshape(kp.shape[0], -1)

        # Pad or truncate to SEQUENCE_LENGTH
        T = kp_flat.shape[0]
        if T < SEQUENCE_LENGTH:
            pad = np.zeros((SEQUENCE_LENGTH - T, kp_flat.shape[1]))
            kp_flat = np.vstack((kp_flat, pad)) # pad at start so last frames are a bit after contact
        else:
            kp_flat = kp_flat[:SEQUENCE_LENGTH] # take last N frames ending at contact

        kp_tensor = torch.tensor(kp_flat, dtype=torch.float32) # (60, 30)
        label_tensor = torch.tensor(sample["label"], dtype=torch.long)

        return kp_tensor, label_tensor


def load_dataset(labels_file, keypoints_dir, val_split=0.15, test_split=0.15, seed=42):
    with open(labels_file, "r") as f:
        rows = list(csv.DictReader(f))

    # Shuffle with fixed seed for reproducibility
    rng = np.random.default_rng(seed)
    rng.shuffle(rows)

    n = len(rows)
    n_test = int(n * test_split)
    n_val = int(n * val_split)

    test_rows = rows[:n_test]
    val_rows = rows[n_test:n_test + n_val]
    train_rows = rows[n_test + n_val:]

    print(f"Split - Train: {len(train_rows)}, Val: {len(val_rows)}, Test: {len(test_rows)}")

    train_ds = PenaltyDataset(train_rows, keypoints_dir)
    val_ds = PenaltyDataset(val_rows, keypoints_dir)
    test_ds = PenaltyDataset(test_rows, keypoints_dir)

    return train_ds, val_ds, test_ds

