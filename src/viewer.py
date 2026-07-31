#viewer.py - Timestamp collection while watching videos from the terminal
import cv2
import os
import time

VIDEO_FILE = "All Penalties In Regular Time FIFA World Cup 2026.mp4"


def resolve_video_path(root_dir=".."):
    raw_dir = os.path.join(root_dir, "data", "raw")
    desired_path = os.path.join(raw_dir, VIDEO_FILE)
    if os.path.exists(desired_path):
        return desired_path
    if os.path.isdir(raw_dir):
        for filename in os.listdir(raw_dir):
            if filename.lower().endswith(".mp4"):
                return os.path.join(raw_dir, filename)
    return desired_path


VIDEO_PATH = resolve_video_path()

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Unable to open video at {VIDEO_PATH!r}. Check that the file exists and the path is correct.")

fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
if fps <= 0:
    fps = 30.0
    print(f"Warning: unable to read FPS from video {VIDEO_PATH!r}. Falling back to {fps:g} fps.")

frame_interval = 1.0 / fps
last_time = time.perf_counter()

timestamps = []

print("Controls:")
print("  SPACE — mark timestamp")
print("  Q     — quit and print all timestamps")
print("  A/D   — skip back/forward 5 seconds")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- FIX 1: RESIZE THE FRAME ---
    # Scale the video down by 50% (or change to 30, 40, etc., depending on your screen)
    scale_percent = 50
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    current_sec = current_frame / fps

    # Overlay timestamp on frame
    display = frame.copy()
    cv2.putText(display, f"{current_sec:.2f}s", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    
    cv2.imshow("Penalty Viewer", display)

    # --- FIX 2: USE DYNAMIC WAIT TIME ---
    elapsed = time.perf_counter() - last_time
    remaining = frame_interval - elapsed
    if remaining > 0:
        time.sleep(remaining)
    last_time = time.perf_counter()

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord(' '):
        timestamps.append(round(current_sec, 2))
        print(f"Marked: {current_sec:.2f}s")
    elif key == ord('a'):  # rewind 5s
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_frame - fps * 5))
    elif key == ord('d'):  # forward 5s
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame + fps * 5)