"""
SignBridge AI - Word recognition service: a recorded clip of frames -> top word guesses.

Loads models/word_model.pt (trained by scripts/train_word_model.py) and keeps the MediaPipe
hand + pose trackers warm between requests. Everything is optional: if PyTorch
or the model file is missing, `ready` is False and the rest of the app is unaffected.
"""

import threading
from typing import Any, Dict, List

import numpy as np

from app.config import MODELS_DIR

WORD_MODEL_PATH = MODELS_DIR / "word_model.pt"
MIN_HAND_FRAME_RATIO = 0.45  # below this the hands were barely visible, so a guess would be noise

MAX_FRAMES = 28        # frames actually tracked per recording
ACTIVE_WRIST_Y = 1.1   # wrist height (shoulder-widths below the shoulders) above which a hand counts as "signing"
TRIM_MARGIN = 3        # frames of context kept around the active stretch
MIN_ACTIVE_FRAMES = 6


def trim_to_activity(feats: np.ndarray) -> np.ndarray:
    """Cuts a recording down to the stretch where a hand is raised. Training clips are cut tightly
    around the sign, while a live recording also contains resting hands before and after it."""
    from app.sequence_features import PER_HAND
    if len(feats) == 0:
        return feats
    active = np.zeros(len(feats), dtype=bool)
    for slot in (0, 1):
        base = slot * PER_HAND
        present = feats[:, base + PER_HAND - 1] > 0
        active |= present & (feats[:, base + 1] < ACTIVE_WRIST_Y)
    idx = np.flatnonzero(active)
    if len(idx) < MIN_ACTIVE_FRAMES:
        return feats
    return feats[max(0, idx[0] - TRIM_MARGIN): idx[-1] + TRIM_MARGIN + 1]


class WordService:
    def __init__(self):
        self.recognizer = None
        self.extractor = None
        self.error = None
        self._lock = threading.Lock()
        self.load()

    def load(self):
        self.recognizer, self.error = None, None
        path = WORD_MODEL_PATH
        try:
            if path.exists():
                from app.word_model import WordRecognizer
                self.recognizer = WordRecognizer(path)
                print(f"[WordService] Loaded {path.name} with {len(self.recognizer.words)} words.")
            else:
                self.error = "models/word_model.pt not found (run scripts/train_word_model.py)"
        except Exception as e:  # torch missing, corrupt file, ...
            self.error = f"{type(e).__name__}: {e}"
            print(f"[WordService] Word recognition disabled: {self.error}")

    @property
    def ready(self) -> bool:
        return self.recognizer is not None

    @property
    def words(self) -> List[str]:
        return list(self.recognizer.words) if self.recognizer else []

    def features(self, frames_bgr: List[np.ndarray], fps: float = 20.0) -> np.ndarray:
        """Camera frames -> per-frame hand + body features (same path for recognising and teaching)."""
        if len(frames_bgr) > MAX_FRAMES:   # the model resamples to 32 steps anyway; fewer frames = faster
            keep = np.linspace(0, len(frames_bgr) - 1, MAX_FRAMES).round().astype(int)
            fps = fps * MAX_FRAMES / len(frames_bgr)
            frames_bgr = [frames_bgr[i] for i in keep]
        with self._lock:
            if self.extractor is None:
                from app.sequence_features import ClipExtractor
                self.extractor = ClipExtractor()
            return self.extractor.extract(frames_bgr, fps=fps)

    @staticmethod
    def hand_ratio(feats: np.ndarray) -> float:
        from app.sequence_features import PER_HAND
        if not len(feats):
            return 0.0
        return float(((feats[:, PER_HAND - 1] > 0) | (feats[:, 2 * PER_HAND - 1] > 0)).mean())

    def recognize(self, frames_bgr: List[np.ndarray], fps: float = 20.0, top_k: int = 3) -> Dict[str, Any]:
        """Returns {"top": [{word, confidence}, ...], "hand_frame_ratio": float}."""
        if not self.ready:
            raise RuntimeError(self.error or "word model not loaded")
        feats = self.features(frames_bgr, fps)
        ratio = self.hand_ratio(feats)
        if ratio < MIN_HAND_FRAME_RATIO:
            return {"top": [], "hand_frame_ratio": ratio}
        trimmed = trim_to_activity(feats)
        return {"top": self.recognizer.predict(trimmed, top_k=top_k), "hand_frame_ratio": ratio,
                "active_frames": len(trimmed), "total_frames": len(feats)}
