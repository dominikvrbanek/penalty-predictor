#pose_extraction.py

'''
# first test if the GPU is available as pose extraction of 848 clips would be very slow on CPU
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")


GPU available: True
GPU name: NVIDIA GeForce RTX 4050 Laptop GPU
'''

import cv2
import os
import json
import mediapipe as mp
import numpy as np
import csv
from ultralytics import YOLO

CLIPS_DIR = "data/clips"
KEYPOINTS_DIR = "data/keypoints"
LABELS_FILE = "data/labels.csv"

os.makedirs(KEYPOINTS_DIR, exist_ok=True)

# Load YOLOv8 model for pose detection
pose_model = YOLO("yolov8n-pose.pt")


# COCO keypoints for pose estimation
RELEVANT_KEYPOINTS = {
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}

KP_INDICES = sorted(RELEVANT_KEYPOINTS.keys()) # [5, 6, 7, 8, 11, 12, 13, 14, 15, 16]\

EXCLUDED_CLIP_RANGES = [
    (488, 505),
    (514, 527),
]


def should_skip_clip(clip_id):
    try:
        clip_number = int(clip_id)
    except (TypeError, ValueError):
        return False

    return any(start <= clip_number <= end for start, end in EXCLUDED_CLIP_RANGES)


# function that extracts the pose keypoints from a video clip and saves them to a JSON file
def extract_pose_keypoints(clip_path, output_json):
    cap = cv2.VideoCapture(clip_path)
    sequence_keypoints = [] # list of pre-frame keypoint arrays

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = pose_model(frame, verbose=False)  # Run pose detection

        if results[0].keypoints is None or len(results[0].keypoints) == 0:
            # No keypoints detected, append a zero array
            sequence_keypoints.append(np.zeros((len(KP_INDICES), 3)).tolist())
            continue

        # Pick the largest detected person (assuming it's the player)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        kicker_idx = int(np.argmax(areas))

        kps = results[0].keypoints.data[kicker_idx].cpu().numpy()  # shape: (17, 3)  # Get keypoints for the largest person 
        # shape of kps is (17, 3) for COCO keypoints (x, y, confidence)

        # Extract only the relevant keypoints
        relevant_kps = kps[KP_INDICES, :].tolist()  # shape (10, 3)
        sequence_keypoints.append(relevant_kps)

    '''
    # Save the sequence of keypoints to a JSON file
    with open(output_json, 'w') as f:
        json.dump(sequence_keypoints, f)
    # shape of sequence_keypoints is (num_frames, 10, 3)
    '''

    cap.release()
    return np.array(sequence_keypoints)  # Return as a numpy array for further processing


def process_all_clips():
    # Read the labels CSV to get clip paths
    with open(LABELS_FILE, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    skipped = 0

    for row in rows:
        clip_id = row["clip_id"]
        if should_skip_clip(clip_id):
            print(f"Skipping excluded clip {clip_id}.")
            continue

        clip_path = row["clip_path"]
        output_path = os.path.join(KEYPOINTS_DIR, f"keypoints_{clip_id}.json")

        if os.path.exists(output_path):
            print(f"Skipping {clip_id}, keypoints already extracted.")
            continue

        if not os.path.exists(clip_path):
            print(f"Clip {clip_id} not found at {clip_path}. Skipping.")
            skipped += 1
            continue

        print(f"Extracting keypoints for clip {clip_id}...")
        kp_sequence = extract_pose_keypoints(clip_path, output_path)

        if len(kp_sequence) == 0:
            print(f"No keypoints detected for clip {clip_id}.")
            skipped += 1
            continue

        np.save(output_path, kp_sequence)
        print(f"Saved keypoints for clip {clip_id} to {output_path}")

    print(f"Processing complete. Skipped {skipped} clips due to missing files or no keypoints detected.")

if __name__ == "__main__":
    process_all_clips()

