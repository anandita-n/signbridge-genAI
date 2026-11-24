"""
SignBridge AI - Gesture Classifier & Temporal Stabilizer
Loads trained classifier, performs confidence-gated inference,
and stabilizes predictions across consecutive video frames to assemble words.
Includes hold-duration progress tracking and single-commit lockouts to prevent accidental repeats.
"""

import time
from collections import deque, Counter
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import joblib

from app.config import (
    CLASSIFIER_MODEL_PATH,
    LABEL_ENCODER_PATH,
    SCALER_PATH,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_STABILITY_FRAMES
)
from app.landmark_extractor import LandmarkExtractor


class GestureClassifier:
    """Classifies normalized hand landmark vectors into ASL signs."""

    def __init__(self):
        self.clf = None
        self.le = None
        self.scaler = None
        self.classes_ = []
        self._load_artifacts()

    def _load_artifacts(self):
        try:
            self.clf = joblib.load(CLASSIFIER_MODEL_PATH)
            self.le = joblib.load(LABEL_ENCODER_PATH)
            self.scaler = joblib.load(SCALER_PATH)
            self.classes_ = [str(c) for c in self.le.classes_]
            print(f"[GestureClassifier] Loaded model successfully with {len(self.classes_)} classes.")
        except Exception as e:
            print(f"[GestureClassifier] Error loading model artifacts: {e}")
            self.clf = None

    def predict_features(
        self,
        features: np.ndarray,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    ) -> Dict[str, Any]:
        """
        Takes an 83-dimensional feature array and returns top prediction,
        confidence, and top-3 candidates.
        """
        if self.clf is None or self.scaler is None:
            return {
                "sign": "NONE",
                "raw_sign": "NONE",
                "confidence": 0.0,
                "is_confident": False,
                "candidates": []
            }

        feats = np.array(features, dtype=np.float32).reshape(1, -1)
        feats_scaled = self.scaler.transform(feats)

        proba = self.clf.predict_proba(feats_scaled)[0]
        top_indices = np.argsort(proba)[::-1][:3]

        top_idx = top_indices[0]
        top_sign = self.classes_[top_idx]
        top_conf = float(proba[top_idx])

        candidates = [
            {"sign": self.classes_[idx], "confidence": round(float(proba[idx]), 4)}
            for idx in top_indices
        ]

        is_confident = top_conf >= confidence_threshold

        return {
            "sign": top_sign,
            "raw_sign": top_sign,
            "confidence": round(top_conf, 4),
            "is_confident": is_confident,
            "candidates": candidates
        }

    def predict_raw_landmarks(
        self,
        landmarks_raw: List[Dict[str, float]],
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    ) -> Dict[str, Any]:
        """
        Convenience method to classify raw JS landmark array [{'x':..., 'y':..., 'z':...}].
        """
        features = LandmarkExtractor.from_client_landmarks(landmarks_raw)
        res = self.predict_features(features, confidence_threshold)
        res["features"] = features.tolist()
        return res


class TemporalStabilizer:
    """
    Stabilizes real-time predictions across rolling video frames.
    Implements a single-commit lockout system: when a sign is held steadily,
    it commits ONCE and then locks until the signer either changes gesture,
    resets their hand, or holds intentionally for an extended repeat interval.
    """

    def __init__(
        self,
        stability_threshold: int = 7,         # Frames to hold before committing (approx 0.45-0.5s)
        confidence_threshold: float = 0.35,   # Confidence threshold
        repeat_cooldown: float = 2.0          # Seconds to hold before allowing an intentional double letter
    ):
        self.stability_threshold = max(3, stability_threshold)
        self.confidence_threshold = confidence_threshold
        self.repeat_cooldown = repeat_cooldown

        self.current_holding_sign: Optional[str] = None
        self.consecutive_frames: int = 0
        self.is_locked: bool = False
        self.locked_sign: Optional[str] = None
        self.last_commit_time: float = 0.0

        self.latest_detected_sign: str = "---"

        # Accumulated text state
        self.char_stream: List[str] = []
        self.current_word: str = ""
        self.current_sentence: str = ""

    def process_prediction(self, prediction_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Feed each incoming frame prediction. Returns whether a character was committed,
        the hold progress (0.0 to 1.0), lock state, and updated text buffer.
        """
        raw_sign = prediction_result.get("raw_sign", "NONE")
        confidence = prediction_result.get("confidence", 0.0)
        is_confident = prediction_result.get("is_confident", False)

        now = time.time()
        committed_action = None

        valid_sign = is_confident and (raw_sign not in ["NONE", "UNCERTAIN"])

        if valid_sign:
            self.latest_detected_sign = raw_sign

            # Check if this is the same sign we are currently holding
            if raw_sign == self.current_holding_sign:
                self.consecutive_frames += 1
            else:
                # Sign has changed - reset hold counter and unlock
                self.current_holding_sign = raw_sign
                self.consecutive_frames = 1
                self.is_locked = False
                self.locked_sign = None

            # Check if locked (already committed this hold)
            if self.is_locked:
                # Only allow repeat if held continuously for repeat_cooldown
                if (now - self.last_commit_time) >= self.repeat_cooldown:
                    # Allow one intentional repeat (e.g. double letter)
                    committed_action = raw_sign
                    self.last_commit_time = now
                    self.consecutive_frames = 0
                    self._apply_sign_action(committed_action)
                progress = 1.0
            else:
                # Progress towards commit
                progress = min(1.0, round(self.consecutive_frames / self.stability_threshold, 2))

                if self.consecutive_frames >= self.stability_threshold:
                    # TRIGGER SINGLE COMMIT!
                    committed_action = raw_sign
                    self.is_locked = True
                    self.locked_sign = raw_sign
                    self.last_commit_time = now
                    self._apply_sign_action(committed_action)
                    progress = 1.0

        else:
            # No valid hand sign or hand dropped / moving
            # Gradually decay or reset hold if hand is lost for 2+ frames
            if self.consecutive_frames > 0:
                self.consecutive_frames = max(0, self.consecutive_frames - 2)
            if self.consecutive_frames == 0:
                self.current_holding_sign = None
                self.is_locked = False
                self.locked_sign = None
            progress = min(1.0, round(self.consecutive_frames / self.stability_threshold, 2))

        return {
            "active_sign": self.current_holding_sign or self.latest_detected_sign,
            "active_confidence": confidence,
            "stability_ratio": progress,
            "is_locked": self.is_locked,
            "committed_action": committed_action,
            "text_state": self.get_text_state()
        }

    def force_commit_sign(self, sign: Optional[str] = None) -> Dict[str, Any]:
        """Manually forces a sign to be committed right now."""
        target = sign or self.latest_detected_sign
        if target and target not in ["NONE", "UNCERTAIN", "---"]:
            self._apply_sign_action(target)
            self.last_commit_time = time.time()
            self.is_locked = True
            self.locked_sign = target
        return self.get_text_state()

    def force_commit_word(self, word: str) -> Dict[str, Any]:
        """Commits a whole word (e.g. from a motion sign like the HELLO wave)."""
        if self.current_word:
            self._apply_sign_action("SPACE")
        for ch in word.upper():
            self._apply_sign_action(ch)
        self._apply_sign_action("SPACE")
        self.last_commit_time = time.time()
        self.is_locked = True
        self.locked_sign = word.upper()
        return self.get_text_state()

    def _apply_sign_action(self, sign: str):
        if sign == "SPACE":
            if self.current_word:
                self.char_stream.append(" ")
                if self.current_sentence:
                    self.current_sentence += " " + self.current_word
                else:
                    self.current_sentence = self.current_word
                self.current_word = ""
        elif sign == "DEL":
            if self.current_word:
                self.current_word = self.current_word[:-1]
                if self.char_stream and self.char_stream[-1] != " ":
                    self.char_stream.pop()
            elif self.current_sentence:
                words = self.current_sentence.split()
                if words:
                    self.current_word = words[-1]
                    self.current_sentence = " ".join(words[:-1])
        elif sign == "CLEAR":
            self.clear()
        else:
            # Regular alphabet character A-Z
            self.char_stream.append(sign)
            self.current_word += sign

    def get_text_state(self) -> Dict[str, Any]:
        full_text = self.current_sentence
        if self.current_word:
            full_text = f"{full_text} {self.current_word}".strip()

        return {
            "char_stream": self.char_stream,
            "current_word": self.current_word,
            "current_sentence": self.current_sentence,
            "full_text": full_text
        }

    def clear(self):
        self.char_stream.clear()
        self.current_word = ""
        self.current_sentence = ""
        self.consecutive_frames = 0
        self.current_holding_sign = None
        self.is_locked = False
        self.locked_sign = None
        self.latest_detected_sign = "---"

    def set_text(self, text: str):
        self.char_stream = list(text)
        words = text.strip().split()
        if len(words) > 1:
            self.current_sentence = " ".join(words[:-1])
            self.current_word = words[-1]
        elif len(words) == 1:
            self.current_sentence = ""
            self.current_word = words[0]
        else:
            self.current_sentence = ""
            self.current_word = ""
