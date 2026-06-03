import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import PoseLandmarkerOptions, RunningMode
import numpy as np
import pickle
import subprocess
import urllib.request
from pathlib import Path
from tqdm import tqdm

MODEL_PATH = Path(__file__).parent / "pose_landmarker_full.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"


def ensure_model():
    if not MODEL_PATH.exists():
        print(f"Downloading MediaPipe pose model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")


def download_video(url: str, output_path: str) -> str:
    print(f"Downloading video: {url}")
    cmd = ["yt-dlp", "-f", "mp4", "-o", output_path, url]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed:\n{result.stderr}")
    return output_path


def extract_pose(video_path: str, output_pkl: str, visibility_threshold: float = 0.5):
    ensure_model()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")

    base_options = mp_tasks.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )

    frames_data = []
    skipped = 0
    frame_idx = 0

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        with tqdm(total=total_frames, desc="Extracting pose") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int(frame_idx * 1000 / fps)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                has_pose = (
                    result.pose_world_landmarks and len(result.pose_world_landmarks) > 0
                    and result.pose_landmarks and len(result.pose_landmarks) > 0
                )
                if has_pose:
                    lm_world = result.pose_world_landmarks[0]
                    lm_img   = result.pose_landmarks[0]
                    key_indices = [11, 12, 13, 14, 15, 16, 23, 24]
                    avg_vis = np.mean([lm_world[i].visibility for i in key_indices])

                    if avg_vis >= visibility_threshold:
                        world = np.array([[l.x, l.y, l.z] for l in lm_world])
                        image = np.array([[l.x, l.y, l.z] for l in lm_img])  # normalized [0,1]
                        frames_data.append((world, image))
                    else:
                        if frames_data:
                            frames_data.append(frames_data[-1])
                        else:
                            skipped += 1
                else:
                    if frames_data:
                        frames_data.append(frames_data[-1])
                    else:
                        skipped += 1

                frame_idx += 1
                pbar.update(1)

    cap.release()

    print(f"Extracted {len(frames_data)} frames ({skipped} skipped at start)")

    world_lm = np.array([f[0] for f in frames_data])  # (N, 33, 3) 3D world coords
    image_lm = np.array([f[1] for f in frames_data])  # (N, 33, 3) normalized image coords

    data = {
        "fps": fps,
        "width": width,
        "height": height,
        "landmarks": world_lm,       # retargeting에 사용
        "image_landmarks": image_lm, # 시각화에 사용
    }

    with open(output_pkl, "wb") as f:
        pickle.dump(data, f)

    print(f"Saved pose data to {output_pkl}")
    return data
