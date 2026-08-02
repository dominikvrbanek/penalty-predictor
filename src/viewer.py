#viewer.py - Timestamp collection while watching videos from the terminal
import cv2
import os
import time
import json

# Video file name to look for in the raw data directory.
VIDEO_FILE = "EVERY PENALTY KICK TAKEN AT THE 2026 FIFA WORLD CUP.mp4"


def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def resolve_video_path(root_dir=None):
    if root_dir is None:
        root_dir = get_project_root()

    raw_dir = os.path.join(root_dir, "data", "raw")
    desired_path = os.path.join(raw_dir, VIDEO_FILE)
    if os.path.exists(desired_path):
        return desired_path
    if os.path.isdir(raw_dir):
        for filename in os.listdir(raw_dir):
            if filename.lower().endswith(".mp4"):
                return os.path.join(raw_dir, filename)
    return desired_path


def pair_timestamps(timestamps):
    return [[timestamps[i], timestamps[i + 1]] for i in range(0, len(timestamps) - 1, 2)]


def save_timestamps(timestamps, output_json=None):
    if output_json is None:
        output_json = os.path.join(get_project_root(), "data", "timestamps.json")
    elif not os.path.isabs(output_json):
        output_json = os.path.abspath(os.path.join(get_project_root(), output_json))

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    paired_timestamps = pair_timestamps(timestamps)
    with open(output_json, "w") as f:
        json.dump(paired_timestamps, f, indent=4)
    print(f"\nTimestamps saved to {output_json}")
    return paired_timestamps

import threading
import queue

def frame_producer(cap, fps, frame_queue):
    """Reads frames in background and puts them in a queue."""
    frame_interval = 1.0 / fps
    while True:
        t = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            frame_queue.put(None)  # signal end
            break
        frame_queue.put((cap.get(cv2.CAP_PROP_POS_FRAMES), frame))
        
        # Sleep remainder in producer thread
        elapsed = time.perf_counter() - t
        remaining = frame_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)

def run_viewer(video_path=None, output_json=None):
    if video_path is None:
        video_path = resolve_video_path()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video at {video_path!r}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = 1.0 / fps
    timestamps = []

    # Start background frame reader
    frame_queue = queue.Queue(maxsize=32)
    producer = threading.Thread(target=frame_producer, args=(cap, fps, frame_queue), daemon=True)
    producer.start()

    print("Controls:")
    print("  SPACE — mark timestamp")
    print("  Q     — quit")
    print("  A/D   — skip back/forward 5 seconds")
    print(cap.get(cv2.CAP_PROP_FPS))
    print(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    try:
        
        while True:
            try:
                item = frame_queue.get(timeout=2.0)
            except queue.Empty:
                break

            if item is None:
                print("\nEnd of video.")
                save_timestamps(timestamps, output_json)
                break

            current_frame, frame = item
            current_sec = current_frame / fps

            # Resize
            scale_percent = 40
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

            cv2.putText(frame, f"{current_sec:.2f}s", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.imshow("Penalty Viewer", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                save_timestamps(timestamps, output_json)
                break
            elif key == ord(' '):
                timestamps.append(round(current_sec, 2))
                print(f"Marked: {current_sec:.2f}s")
            elif key == ord('a'):
                # Flush queue and seek back
                with frame_queue.mutex:
                    frame_queue.queue.clear()
                new_frame = max(0, current_frame - fps * 5)
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            elif key == ord('d'):
                with frame_queue.mutex:
                    frame_queue.queue.clear()
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame + fps * 5)

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_viewer()