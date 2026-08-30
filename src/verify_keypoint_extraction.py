#verify keypoint extraction
import numpy as np
import os 

KP_DIR = "data/keypoints"
files = [f for f in os.listdir(KP_DIR) if f.endswith(".npy")]

# Check a few random files
import random
for f in random.sample(files, 5):
    kp = np.load(os.path.join(KP_DIR, f))
    print(f"{f}: shape={kp.shape}, min={kp.min():.2f}, max={kp.max():.2f}")