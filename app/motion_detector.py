"""
SignBridge AI - Motion and phrase sign detector (J, Z, HELLO, I LOVE YOU, YES, PLEASE)

J and Z are the only ASL letters signed with movement, so a single-frame classifier
cannot recognise them: J starts as the "I" handshape and Z starts as a pointing index
finger. This module tracks fingertip / palm paths over a couple of seconds and
recognises:
  * J     - "I" handshape (only the little finger extended) whose tip draws a downward
            stroke followed by a sideways hook.
  * Z     - pointing handshape (index extended, middle/ring curled) whose tip draws
            right, then diagonal down-left, then right again (or the mirror image).
  * HELLO - open hand waving side to side (at least three direction changes).
  * ILY   - "I love you" handshape (thumb, index and little finger out) held still.
  * YES   - fist nodding up and down (at least three direction changes).
  * PLEASE - flat open hand rubbing in a circle.

It works on any consistent 2D coordinates (normalised 0-1 or pixels) and is
mirror-invariant, so it does not matter whether the camera image is flipped.
The thresholds are heuristics measured on real hand landmarks; every frame is also
kept in an in-memory debug buffer (`MotionTracker.debug`) so they can be tuned
against a real camera.
"""

import time
from collections import deque
from typing import Dict, List, Optional, Sequence

import numpy as np

WINDOW_SECONDS = 2.2
COOLDOWN_SECONDS = 1.5
MIN_FRAMES = 8
GRACE_FRAMES = 8  # frames the handshape may be lost before the tracked path is discarded

# Finger extension ratio = |tip - wrist| / |knuckle - wrist| (measured on real hands:
# extended fingers are ~1.7-1.9, curled fingers ~0.5-1.2, more while moving/tilted).
EXTENDED_MIN = 1.45
CURLED_MAX = 1.45

# Landmark indices (MediaPipe): wrist=0, middle MCP=9
_FINGERS = {"index": (5, 8), "middle": (9, 12), "ring": (13, 16), "pinky": (17, 20)}

# "I love you": the thumb tip is this far from the middle knuckle (in hand sizes). Measured on real
# hands: thumb out (letters L, Y) 0.76-1.1, thumb tucked (A, S, E, I...) mostly under 0.5.
THUMB_OUT_MIN = 0.7
HOLD_SECONDS = 0.6          # a held handshape (ILY) must stay this long
NOD_MIN_SWING = 0.18        # hand sizes
MOTION_SIGNS = ("J", "Z", "HELLO", "ILY", "YES", "PLEASE")


def _ratios(points: np.ndarray) -> Dict[str, float]:
    wrist = points[0]
    out = {}
    for name, (mcp, tip) in _FINGERS.items():
        base = np.linalg.norm(points[mcp] - wrist)
        out[name] = float(np.linalg.norm(points[tip] - wrist) / base) if base > 1e-9 else 0.0
    return out


def handshape(points: np.ndarray) -> Optional[str]:
    """'J' (I-like), 'Z' (pointing), 'HELLO' (open hand, also used for PLEASE), 'ILY', 'YES' (fist) or None."""
    r = _ratios(points)
    hand_size = float(np.linalg.norm(points[0] - points[9]))
    thumb_out = hand_size > 1e-9 and float(np.linalg.norm(points[4] - points[9])) / hand_size >= THUMB_OUT_MIN
    if all(r[f] >= EXTENDED_MIN for f in ("index", "middle", "ring", "pinky")):
        return "HELLO"
    if (thumb_out and r["index"] >= EXTENDED_MIN and r["pinky"] >= EXTENDED_MIN
            and r["middle"] <= CURLED_MAX and r["ring"] <= CURLED_MAX):
        return "ILY"
    if (r["pinky"] >= EXTENDED_MIN and r["index"] <= CURLED_MAX
            and r["middle"] <= CURLED_MAX and r["ring"] <= CURLED_MAX):
        return "J"
    if (r["index"] >= EXTENDED_MIN and r["middle"] <= CURLED_MAX
            and r["ring"] <= CURLED_MAX and r["pinky"] <= 1.6):
        return "Z"
    if all(r[f] <= CURLED_MAX for f in ("index", "middle", "ring", "pinky")):
        return "YES"
    return None


def _rdp(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Ramer-Douglas-Peucker polyline simplification."""
    if len(points) < 3:
        return points
    start, end = points[0], points[-1]
    line = end - start
    norm = np.linalg.norm(line)
    if norm < 1e-9:
        dists = np.linalg.norm(points - start, axis=1)
    else:
        dists = np.abs(line[0] * (points[:, 1] - start[1]) - line[1] * (points[:, 0] - start[0])) / norm
    idx = int(np.argmax(dists))
    if dists[idx] > epsilon:
        left = _rdp(points[: idx + 1], epsilon)
        right = _rdp(points[idx:], epsilon)
        return np.vstack([left[:-1], right])
    return np.vstack([start, end])


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))))


def _segments(path: np.ndarray, epsilon: float = 0.25, min_len: float = 0.25,
              merge_deg: float = 35.0) -> List[np.ndarray]:
    """Simplify a path into straight strokes, merging near-collinear neighbours."""
    simple = _rdp(path, epsilon)
    segs = [simple[i + 1] - simple[i] for i in range(len(simple) - 1)]
    segs = [s for s in segs if np.linalg.norm(s) >= min_len]
    merged: List[np.ndarray] = []
    for s in segs:
        if merged and _angle(merged[-1], s) < merge_deg:
            merged[-1] = merged[-1] + s
        else:
            merged.append(s)
    return merged


def looks_like_z(path: np.ndarray) -> bool:
    """path: fingertip positions in hand-size units (y grows downward)."""
    segs = _segments(path)
    if len(segs) < 3:
        return False
    for i in range(len(segs) - 2):
        s1, s2, s3 = segs[i], segs[i + 1], segs[i + 2]
        sign = np.sign(s1[0])
        if sign == 0:
            continue
        top_ok = abs(s1[1]) <= 0.9 * abs(s1[0]) and np.linalg.norm(s1) >= 0.35
        bottom_ok = (abs(s3[1]) <= 0.9 * abs(s3[0]) and np.linalg.norm(s3) >= 0.35
                     and np.sign(s3[0]) == sign)
        diag_ok = (np.sign(s2[0]) == -sign and s2[1] > 0
                   and s2[1] >= 0.3 * abs(s2[0]) and np.linalg.norm(s2) >= 0.4)
        if top_ok and diag_ok and bottom_ok:
            return True
    return False


def looks_like_j(path: np.ndarray) -> bool:
    """A downward stroke followed by a sideways hook (curving back up or across)."""
    segs = _segments(path)
    for i in range(len(segs) - 1):
        down, hook = segs[i], segs[i + 1]
        down_len, hook_len = np.linalg.norm(down), np.linalg.norm(hook)
        if not (down[1] > 0 and abs(down[1]) >= 1.0 * abs(down[0]) and down_len >= 0.35):
            continue
        if hook_len < 0.2 or abs(hook[0]) < 0.3 * hook_len:
            continue  # the return stroke must have a real sideways component
        if 40 <= _angle(down, hook) <= 160:
            return True
    return False


def _count_swings(values: np.ndarray, min_swing: float) -> int:
    """Direction reversals of a 1-D signal whose swing is at least `min_swing`."""
    extreme = values[0]
    direction = 0  # +1 rising, -1 falling, 0 not moving yet
    swings = 0
    for v in values[1:]:
        if direction == 0:
            if v - extreme >= min_swing:
                direction, extreme = 1, v
            elif extreme - v >= min_swing:
                direction, extreme = -1, v
        elif direction == 1:
            if v > extreme:
                extreme = v
            elif extreme - v >= min_swing:
                direction, extreme, swings = -1, v, swings + 1
        else:
            if v < extreme:
                extreme = v
            elif v - extreme >= min_swing:
                direction, extreme, swings = 1, v, swings + 1
    return swings


def looks_like_wave(path: np.ndarray) -> bool:
    """Side-to-side oscillation: >= 3 horizontal direction changes of real amplitude."""
    if len(path) < 6:
        return False
    x, y = path[:, 0], path[:, 1]
    if (x.max() - x.min()) < 0.5:
        return False
    if (y.max() - y.min()) > 1.5 * (x.max() - x.min()):
        return False  # mostly vertical movement is not a wave
    return _count_swings(x, 0.35) >= 3


def looks_like_nod(path: np.ndarray) -> bool:
    """Up-and-down movement: >= 3 vertical direction changes, mostly vertical."""
    if len(path) < 6:
        return False
    x, y = path[:, 0], path[:, 1]
    if np.ptp(y) < 0.3 or np.ptp(x) > np.ptp(y):
        return False
    return _count_swings(y, NOD_MIN_SWING) >= 3


def looks_like_circle(path: np.ndarray) -> bool:
    """About one full turn in a consistent direction around the path's centre."""
    if len(path) < 10:
        return False
    x_range, y_range = np.ptp(path[:, 0]), np.ptp(path[:, 1])
    if min(x_range, y_range) < 0.5 or not 0.4 <= x_range / y_range <= 2.5:
        return False
    rel = path - path.mean(axis=0)
    radius = np.linalg.norm(rel, axis=1)
    if radius.mean() < 0.2 or radius.std() > 0.6 * radius.mean():
        return False
    steps = np.diff(np.unwrap(np.arctan2(rel[:, 1], rel[:, 0])))
    total = float(steps.sum())
    consistent = abs(total) / max(float(np.abs(steps).sum()), 1e-9)
    return abs(total) >= 2 * np.pi * 0.85 and consistent >= 0.7


class MotionTracker:
    """Feed it each frame's 21 hand points; it returns 'J', 'Z', 'HELLO', 'ILY', 'YES' or 'PLEASE' when signed."""

    def __init__(self, window_seconds: float = WINDOW_SECONDS, cooldown: float = COOLDOWN_SECONDS):
        self.window = window_seconds
        self.cooldown = cooldown
        self._shape: Optional[str] = None
        self._buf: deque = deque()  # (time, x, y, hand_size)
        self._misses = 0
        self.last_shape: Optional[str] = None   # handshape seen in the most recent frame
        self._ily_done = False   # ILY is a held sign: once fired, wait for the handshape to change
        self._last_trigger = float("-inf")
        # Rolling record of recent frames for offline tuning: not persisted anywhere.
        self.debug: deque = deque(maxlen=3000)

    def reset(self, clear_cooldown: bool = False):
        self._shape = None
        self._buf.clear()
        self._misses = 0
        if clear_cooldown:
            self._last_trigger = float("-inf")
            self._ily_done = False

    def update(self, landmarks: Sequence[Sequence[float]], now: Optional[float] = None) -> Optional[str]:
        now = time.time() if now is None else now
        pts = np.asarray([[p[0], p[1]] for p in landmarks], dtype=np.float64)
        if pts.shape != (21, 2):
            self.reset()
            return None

        hand_size = float(np.linalg.norm(pts[0] - pts[9]))
        shape = handshape(pts) if hand_size > 1e-9 else None
        self.debug.append({
            "t": round(now, 3), "shape": shape, "hand_size": round(hand_size, 4),
            "ratios": {k: round(v, 2) for k, v in _ratios(pts).items()} if hand_size > 1e-9 else {},
            "idx_tip": [round(float(pts[8][0]), 4), round(float(pts[8][1]), 4)],
            "pinky_tip": [round(float(pts[20][0]), 4), round(float(pts[20][1]), 4)],
            "palm": [round(float(pts[9][0]), 4), round(float(pts[9][1]), 4)],
        })

        self.last_shape = shape
        if shape != "ILY":
            self._ily_done = False
        if shape is None or (self._shape is not None and shape != self._shape):
            # Tolerate brief handshape flicker (a tilted hand), but not a different sign.
            if shape is None and self._shape is not None and self._misses < GRACE_FRAMES:
                self._misses += 1
                return None
            self.reset()
            if shape is None:
                return None
        if self._shape is None:
            self._shape = shape
        self._misses = 0

        if shape == "J":
            tip = pts[20]
        elif shape == "Z":
            tip = pts[8]
        else:
            tip = pts[9]  # palm centre for the wave
        self._buf.append((now, tip[0], tip[1], hand_size))
        while self._buf and now - self._buf[0][0] > self.window:
            self._buf.popleft()

        if len(self._buf) < MIN_FRAMES or now - self._last_trigger < self.cooldown:
            return None

        arr = np.array(self._buf)
        size = float(np.median(arr[:, 3]))
        path = self._smooth(arr[:, 1:3] / size)

        result = None
        if shape == "J":
            result = "J" if looks_like_j(path) else None
        elif shape == "Z":
            result = "Z" if looks_like_z(path) else None
        elif shape == "ILY":
            if not self._ily_done and now - self._buf[0][0] >= HOLD_SECONDS:
                result = "ILY"
                self._ily_done = True
        elif shape == "YES":
            result = "YES" if looks_like_nod(path) else None
        else:  # open hand: a circle is PLEASE, a side-to-side wave is HELLO
            if looks_like_circle(path):
                result = "PLEASE"
            elif looks_like_wave(path):
                result = "HELLO"
        if result:
            self._last_trigger = now
            self.reset()
        return result

    @staticmethod
    def _smooth(path: np.ndarray, k: int = 3) -> np.ndarray:
        if len(path) < k:
            return path
        kernel = np.ones(k) / k
        sm = np.column_stack([np.convolve(path[:, c], kernel, mode="valid") for c in range(2)])
        return np.vstack([path[:1], sm, path[-1:]])
