"""
SignBridge AI - Per-frame hand + body features for word-level (sequence) recognition.

Shared by scripts/extract_sequences.py (training data) and the live /api/word endpoint so
both produce exactly the same features. Body-relative coordinates (origin = shoulder centre,
scale = shoulder width) capture *where* the hand is (forehead, chin, chest); a wrist-centred
block captures the hand *shape*.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from app.config import MODELS_DIR

HAND_MODEL = MODELS_DIR / "hand_landmarker.task"
POSE_MODEL = MODELS_DIR / "pose_landmarker_lite.task"

# Pose landmarks used: nose, shoulders, elbows, wrists
POSE_IDX = [0, 11, 12, 13, 14, 15, 16]
HAND_BODY_DIM = 21 * 3          # hand landmarks relative to the body
HAND_SHAPE_DIM = 21 * 2         # hand landmarks relative to the wrist (shape only)
PER_HAND = HAND_BODY_DIM + HAND_SHAPE_DIM + 1   # + presence flag
POSE_DIM = len(POSE_IDX) * 3 + 1                # + presence flag
FEATURE_DIM = 2 * PER_HAND + POSE_DIM


def frame_features(hands: Sequence[Tuple[str, np.ndarray]], pose_lms: Optional[np.ndarray]) -> np.ndarray:
    """hands: list of (label, 21x3 array); pose_lms: 33x3 array or None -> FEATURE_DIM vector."""
    feat = np.zeros(FEATURE_DIM, dtype=np.float32)

    centre, scale, have_pose = np.zeros(3), 1.0, False
    if pose_lms is not None:
        ls, rs = pose_lms[11], pose_lms[12]
        width = float(np.linalg.norm(ls[:2] - rs[:2]))
        if width > 1e-4:
            centre, scale, have_pose = (ls + rs) / 2.0, width, True
    if have_pose:
        off = 2 * PER_HAND
        for k, i in enumerate(POSE_IDX):
            feat[off + k * 3: off + k * 3 + 3] = (pose_lms[i] - centre) / scale
        feat[off + len(POSE_IDX) * 3] = 1.0

    slots = {}
    for label, pts in hands:
        slots.setdefault(0 if label == "Left" else 1, pts)  # first detection per side wins
    for slot, pts in slots.items():
        off = slot * PER_HAND
        body = (pts - centre) / scale if have_pose else (pts - pts[0]) / max(np.linalg.norm(pts[9, :2] - pts[0, :2]), 1e-4)
        feat[off: off + HAND_BODY_DIM] = body.reshape(-1)
        shape = pts[:, :2] - pts[0, :2]
        shape = shape / max(float(np.linalg.norm(shape[9])), 1e-4)
        feat[off + HAND_BODY_DIM: off + HAND_BODY_DIM + HAND_SHAPE_DIM] = shape.reshape(-1)
        feat[off + HAND_BODY_DIM + HAND_SHAPE_DIM] = 1.0
    return feat


def make_landmarkers():
    """Creates (hand_landmarker, pose_landmarker) in VIDEO mode. Caller must close() them."""
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=vision.RunningMode.VIDEO, num_hands=2))
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=vision.RunningMode.VIDEO, num_poses=1))
    return hand, pose


def pad_to_square(frame_bgr: np.ndarray) -> np.ndarray:
    """Letterbox a frame to a square so normalised x/y coordinates have equal scale,
    matching the square training videos."""
    h, w = frame_bgr.shape[:2]
    if h == w:
        return frame_bgr
    side = max(h, w)
    out = np.zeros((side, side, 3), dtype=frame_bgr.dtype)
    y0, x0 = (side - h) // 2, (side - w) // 2
    out[y0:y0 + h, x0:x0 + w] = frame_bgr
    return out


class ClipExtractor:
    """Keeps the hand + pose landmarkers loaded and turns clips of BGR frames into features.
    Not thread-safe: callers serialise access."""

    def __init__(self):
        self.hand, self.pose = make_landmarkers()
        self._ts = 0  # MediaPipe VIDEO mode needs timestamps that only ever increase

    def extract(self, frames_bgr: List[np.ndarray], fps: float = 25.0) -> np.ndarray:
        """-> (n_frames, FEATURE_DIM) features."""
        import cv2
        import mediapipe as mp

        base = self._ts
        imgs = [mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(pad_to_square(f), cv2.COLOR_BGR2RGB)) for f in frames_bgr]
        stamps = [base + int(n * 1000.0 / fps) for n in range(len(imgs))]

        def run_hands():
            out = []
            for img, ts in zip(imgs, stamps):
                res = self.hand.detect_for_video(img, ts)
                out.append([(h[0].category_name, np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32))
                            for lms, h in zip(res.hand_landmarks, res.handedness)])
            return out

        def run_pose():
            out = []
            for img, ts in zip(imgs, stamps):
                res = self.pose.detect_for_video(img, ts)
                out.append(np.array([[p.x, p.y, p.z] for p in res.pose_landmarks[0]], dtype=np.float32)
                           if res.pose_landmarks else None)
            return out

        # Hand and pose trackers are independent, so run them side by side (MediaPipe releases the GIL).
        with ThreadPoolExecutor(max_workers=2) as pool:
            hand_job, pose_job = pool.submit(run_hands), pool.submit(run_pose)
            hands_per_frame, pose_per_frame = hand_job.result(), pose_job.result()

        rows = [frame_features(h, p) for h, p in zip(hands_per_frame, pose_per_frame)]
        self._ts = base + int(len(frames_bgr) * 1000.0 / fps) + 1000
        return np.stack(rows) if rows else np.zeros((0, FEATURE_DIM), dtype=np.float32)

    def close(self):
        self.hand.close()
        self.pose.close()


def extract_clip(frames_bgr: List[np.ndarray], fps: float = 25.0) -> np.ndarray:
    """One-off extraction (loads and closes the landmarkers)."""
    extractor = ClipExtractor()
    try:
        return extractor.extract(frames_bgr, fps)
    finally:
        extractor.close()
