# src/dataset.py
import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset

LABEL_MAP = {"L": 0, "C": 1, "R": 2}
SEQUENCE_LENGTH = 60

# Keypoint indices in our extracted data (already filtered to 10 keypoints)
# Order: left_shoulder(5), right_shoulder(6), left_elbow(7), right_elbow(8),
#        left_hip(11), right_hip(12), left_knee(13), right_knee(14),
#        left_ankle(15), right_ankle(16)
L_SHOULDER, R_SHOULDER = 0, 1
L_ELBOW,    R_ELBOW    = 2, 3
L_HIP,      R_HIP      = 4, 5
L_KNEE,     R_KNEE     = 6, 7
L_ANKLE,    R_ANKLE    = 8, 9

def compute_engineered_features(kp):
    """
    kp: (T, 10, 3) array of keypoints — x, y, confidence
    Returns: (T, F) array of engineered features
    """
    features = []

    for frame in kp:
        # frame shape: (10, 3) — each row is [x, y, conf]
        pts = frame[:, :2]   # (10, 2) — just x, y
        conf = frame[:, 2]   # (10,) — confidence

        f = []

        # --- 1. Hip angle (most predictive feature) ---
        # Angle of the line connecting left and right hip
        hip_vec = pts[R_HIP] - pts[L_HIP]
        hip_angle = np.arctan2(hip_vec[1], hip_vec[0])
        f.append(hip_angle)

        # --- 2. Shoulder angle ---
        shoulder_vec = pts[R_SHOULDER] - pts[L_SHOULDER]
        shoulder_angle = np.arctan2(shoulder_vec[1], shoulder_vec[0])
        f.append(shoulder_angle)

        # --- 3. Body lean — torso tilt left/right ---
        # Midpoint of hips vs midpoint of shoulders
        mid_hip = (pts[L_HIP] + pts[R_HIP]) / 2
        mid_shoulder = (pts[L_SHOULDER] + pts[R_SHOULDER]) / 2
        torso_vec = mid_shoulder - mid_hip
        torso_lean = np.arctan2(torso_vec[0], torso_vec[1])  # x/y = left/right lean
        f.append(torso_lean)

        # --- 4. Knee angles ---
        # Left knee: angle between hip->knee->ankle
        if conf[L_HIP] > 0.3 and conf[L_KNEE] > 0.3 and conf[L_ANKLE] > 0.3:
            v1 = pts[L_HIP] - pts[L_KNEE]
            v2 = pts[L_ANKLE] - pts[L_KNEE]
            cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            l_knee_angle = np.arccos(np.clip(cos_a, -1, 1))
        else:
            l_knee_angle = 0.0
        f.append(l_knee_angle)

        # Right knee
        if conf[R_HIP] > 0.3 and conf[R_KNEE] > 0.3 and conf[R_ANKLE] > 0.3:
            v1 = pts[R_HIP] - pts[R_KNEE]
            v2 = pts[R_ANKLE] - pts[R_KNEE]
            cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            r_knee_angle = np.arccos(np.clip(cos_a, -1, 1))
        else:
            r_knee_angle = 0.0
        f.append(r_knee_angle)

        # --- 5. Plant foot position relative to hips ---
        # Normalized x offset of each ankle from hip center
        mid_hip_x = mid_hip[0]
        body_width = np.linalg.norm(pts[R_HIP] - pts[L_HIP]) + 1e-6
        l_ankle_offset = (pts[L_ANKLE][0] - mid_hip_x) / body_width
        r_ankle_offset = (pts[R_ANKLE][0] - mid_hip_x) / body_width
        f.append(l_ankle_offset)
        f.append(r_ankle_offset)

        # --- 6. Average confidence for this frame ---
        f.append(conf.mean())

        features.append(f)

    return np.array(features, dtype=np.float32)  # (T, 8)


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
                "clip_id": clip_id,
            })

        print(f"Dataset loaded: {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        kp = np.load(sample["kp_path"])  # (T, 10, 3)

        # Compute engineered features instead of raw coordinates
        features = compute_engineered_features(kp)  # (T, 8)

        # Pad or truncate to fixed sequence length
        T = features.shape[0]
        if T < SEQUENCE_LENGTH:
            pad = np.zeros((SEQUENCE_LENGTH - T, features.shape[1]))
            features = np.vstack([pad, features])  # pad at start
        else:
            features = features[-SEQUENCE_LENGTH:]  # keep frames closest to contact

        # Replace any NaN/inf from bad keypoints
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        kp_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(sample["label"], dtype=torch.long)

        return kp_tensor, label_tensor


def load_datasets(labels_file, keypoints_dir, val_split=0.15,
                  test_split=0.15, seed=42):
    with open(labels_file, "r") as f:
        rows = list(csv.DictReader(f))

    rng = np.random.default_rng(seed)
    rng.shuffle(rows)

    n = len(rows)
    n_test = int(n * test_split)
    n_val = int(n * val_split)

    test_rows  = rows[:n_test]
    val_rows   = rows[n_test:n_test + n_val]
    train_rows = rows[n_test + n_val:]

    print(f"Split — Train: {len(train_rows)}, Val: {len(val_rows)}, Test: {len(test_rows)}")

    return (PenaltyDataset(train_rows, keypoints_dir),
            PenaltyDataset(val_rows,   keypoints_dir),
            PenaltyDataset(test_rows,  keypoints_dir))