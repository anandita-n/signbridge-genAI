"""
SignBridge AI - Extract landmark sequences from word-sign videos (WLASL100).

For every video it runs MediaPipe HandLandmarker (2 hands) and PoseLandmarker and stores a
(frames x FEATURE_DIM) float32 array. Body-relative coordinates (origin = shoulder centre,
scale = shoulder width) capture *where* the hand is (forehead, chin, chest), while a
wrist-centred hand-shape block captures *what* the hand looks like.

Usage (from the repo root):
    python scripts/extract_sequences.py --limit 6        # quick test on a few videos
    python scripts/extract_sequences.py --workers 4      # everything (resumable)

Output: data/wlasl100/seq/<class>_<videoid>.npz  with  features, label, word, split, signer
"""

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data" / "wlasl100"
VIDEOS = DATA / "videos"
OUT = DATA / "seq"
HAND_MODEL = ROOT / "models" / "hand_landmarker.task"
POSE_MODEL = ROOT / "models" / "pose_landmarker_lite.task"

from app.sequence_features import FEATURE_DIM, frame_features  # noqa: E402  (shared with the live API)

_worker = {}


def _init_worker():
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    hand_opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=vision.RunningMode.VIDEO, num_hands=2)
    pose_opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=vision.RunningMode.VIDEO, num_poses=1)
    _worker["mp"] = mp
    _worker["hand_cls"] = (vision.HandLandmarker, hand_opts)
    _worker["pose_cls"] = (vision.PoseLandmarker, pose_opts)


def process_video(job):
    path, meta = job
    out_file = OUT / f"{meta['cls']}_{meta['video_id']}.npz"
    if out_file.exists():
        return str(path), "skipped", 0, 0.0
    t0 = time.time()
    mp = _worker["mp"]
    hand_lm = _worker["hand_cls"][0].create_from_options(_worker["hand_cls"][1])
    pose_lm = _worker["pose_cls"][0].create_from_options(_worker["pose_cls"][1])
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames, hand_frames, n = [], 0, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ts = int(n * 1000.0 / fps)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            hres = hand_lm.detect_for_video(img, ts)
            pres = pose_lm.detect_for_video(img, ts)
            hands = []
            for lms, handed in zip(hres.hand_landmarks, hres.handedness):
                pts = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)
                hands.append((handed[0].category_name, pts))
            pose = None
            if pres.pose_landmarks:
                pose = np.array([[p.x, p.y, p.z] for p in pres.pose_landmarks[0]], dtype=np.float32)
            hand_frames += bool(hands)
            frames.append(frame_features(hands, pose))
            n += 1
    finally:
        cap.release()
        hand_lm.close()
        pose_lm.close()
    if not frames:
        return str(path), "empty", 0, time.time() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_file, features=np.stack(frames), label=meta["label"], word=meta["word"],
        split=meta["split"], signer=meta["signer"], hand_frames=hand_frames)
    return str(path), "ok", n, time.time() - t0


def build_jobs():
    labels = {}
    for line in (DATA / "folder2label_str.txt").read_text().splitlines():
        idx, word = line.split(maxsplit=1)
        labels[int(idx)] = word.strip()
    index = {}
    for entry in json.loads((DATA / "WLASL_v0.3.json").read_text()):
        for inst in entry["instances"]:
            index[int(inst["video_id"])] = (inst.get("split", "train"), inst.get("signer_id", -1))
    jobs, missing = [], 0
    for cls_dir in sorted(VIDEOS.iterdir(), key=lambda p: int(p.name)):
        cls = int(cls_dir.name)
        for mp4 in sorted(cls_dir.glob("*.mp4")):
            vid = int(mp4.stem)
            if vid not in index:
                missing += 1
                continue
            split, signer = index[vid]
            jobs.append((mp4, {"cls": cls, "video_id": mp4.stem, "label": cls,
                               "word": labels[cls], "split": split, "signer": signer}))
    return jobs, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="only process this many videos (test run)")
    args = ap.parse_args()

    jobs, missing = build_jobs()
    if args.limit:
        step = max(1, len(jobs) // args.limit)
        jobs = jobs[::step][: args.limit]
    print(f"{len(jobs)} videos to process ({missing} videos missing from metadata) | feature dim {FEATURE_DIM}", flush=True)

    t0, done, frames_total = time.time(), 0, 0
    with Pool(args.workers, initializer=_init_worker) as pool:
        for path, status, n, dt in pool.imap_unordered(process_video, jobs):
            done += 1
            frames_total += n
            if status not in ("ok", "skipped") or done % 25 == 0 or args.limit:
                print(f"[{done}/{len(jobs)}] {status} {Path(path).parent.name}/{Path(path).name} frames={n} {dt:.1f}s "
                      f"elapsed={time.time() - t0:.0f}s", flush=True)
    print(f"Finished {done} videos, {frames_total} new frames in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
