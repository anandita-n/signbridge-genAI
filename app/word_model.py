"""
SignBridge AI - Word-level sign recognition (sequence model)

A small 1D-CNN over a short clip of per-frame hand + body-pose features (see
scripts/extract_sequences.py for the feature layout). The clip is resampled to a fixed
number of frames, per-frame differences (motion) are appended, and the network outputs
a probability for each word in the vocabulary (WLASL100 by default).

PyTorch is only needed for this module, so the rest of the app keeps working without it.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

T_FRAMES = 32
PER_HAND = 21 * 3 + 21 * 2 + 1
POSE_DIM = 7 * 3 + 1
FEATURE_DIM = 2 * PER_HAND + POSE_DIM  # 234


def sample_frames(seq: np.ndarray, T: int = T_FRAMES, train: bool = False,
                  rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Pick T frames from a (n, FEATURE_DIM) clip. Training adds random crop + time jitter."""
    n = len(seq)
    if n == 0:
        return np.zeros((T, FEATURE_DIM), dtype=np.float32)
    if train and rng is not None and n > 8:
        lo = int(rng.uniform(0, 0.12) * n)
        hi = n - int(rng.uniform(0, 0.12) * n)
        if hi - lo >= 6:
            seq, n = seq[lo:hi], hi - lo
        idx = np.round(np.sort(rng.uniform(0, n - 1, T))).astype(int)
    else:
        idx = np.round(np.linspace(0, n - 1, T)).astype(int)
    return seq[idx].astype(np.float32)


def with_deltas(x: np.ndarray) -> np.ndarray:
    """(T, D) -> (T, 2D): features plus frame-to-frame change (motion)."""
    delta = np.vstack([np.zeros((1, x.shape[1]), dtype=np.float32), x[1:] - x[:-1]])
    return np.concatenate([x, delta], axis=1)


def build_model(num_classes: int, in_channels: int = FEATURE_DIM * 2):
    import torch.nn as nn

    class WordNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(in_channels, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
                nn.Dropout(0.2),
                nn.Conv1d(128, 128, 5, padding=2, stride=2), nn.BatchNorm1d(128), nn.ReLU(),
                nn.Conv1d(128, 256, 3, padding=1, stride=2), nn.BatchNorm1d(256), nn.ReLU(),
            )
            self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(512, num_classes))

        def forward(self, x):  # x: (B, C, T)
            h = self.features(x)
            pooled = __import__("torch").cat([h.mean(dim=2), h.amax(dim=2)], dim=1)
            return self.head(pooled)

    return WordNet()


class WordRecognizer:
    """Loads models/word_model.pt and classifies a clip of per-frame features."""

    def __init__(self, path: Path):
        import torch
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.words: List[str] = ckpt["words"]
        self.mean = np.asarray(ckpt["mean"], dtype=np.float32)
        self.std = np.asarray(ckpt["std"], dtype=np.float32)
        self.model = build_model(len(self.words))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self._torch = torch

    def predict(self, seq: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        x = with_deltas(sample_frames(seq))
        x = (x - self.mean) / self.std
        t = self._torch.from_numpy(x.T.copy()).unsqueeze(0)
        with self._torch.no_grad():
            probs = self._torch.softmax(self.model(t), dim=1)[0].numpy()
        order = np.argsort(probs)[::-1][:top_k]
        return [{"word": self.words[i], "confidence": round(float(probs[i]), 4)} for i in order]
