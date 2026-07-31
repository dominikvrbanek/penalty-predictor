#extract_clips.py
import cv2
import os
import csv

RAW_DIR = "data/raw"
CLIPS_DIR = "data/clips"
LABELS_FILE = "data/labels.csv"

os.makedirs(CLIPS_DIR, exist_ok=True)

def extract_clip(video_path, start_sec, end_sec, clip_id):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    output_path = os.path.join(CLIPS_DIR, f"clip_{clip_id:03d}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps,
                          (w, h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret: 
            break
        out.write(frame)

    cap.release()
    out.release()
    print(f"Saved: {output_path}")
    return output_path

# --- Copy ---
# Format: (start_seconds, end_seconds, direction, foot)
# Direction: L / C / R From the keepers perspective which direction to dive
# Foot: left / right
PENALTIES = [
    # (10, 18, "L", "right"),   # example
]

video_file = os.path.join(RAW_DIR, "All Penalties In Regular Time | FIFA World Cup 2026™.mp4")

with open(LABELS_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["clip_id", "clip_path", "direction", "foot"])

    for i, (start, end, direction, foot) in enumerate(PENALTIES):
        path = extract_clip(video_file, start, end, i)
        writer.writerow([i, path, direction, foot])

print(f"\nExtracted {len(PENALTIES)} clips. Labels saved to {LABELS_FILE}")