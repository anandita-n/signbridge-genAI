"""Unit tests for the motion / phrase sign detector (J, Z, HELLO, ILY, YES, PLEASE) using synthetic hand paths."""

import unittest
from unittest import mock

import numpy as np

from app.motion_detector import MotionTracker, handshape

FPS = 30.0
EXTENDED, CURLED = 1.9, 0.8


def make_hand(extended_fingers, offset=(0.0, 0.0), thumb_out=False):
    """21 landmarks: wrist at origin, finger MCPs at distance 1, tips at ratio 1.9 or 0.8.
    The thumb tip is tucked in by default; thumb_out=True sticks it out sideways."""
    pts = np.zeros((21, 2))
    pts[4] = (-0.9, -0.4) if thumb_out else (0.05, -0.7)
    angles = {"index": 100, "middle": 90, "ring": 80, "pinky": 70}  # degrees, pointing "up" (-y)
    idx = {"index": (5, 8), "middle": (9, 12), "ring": (13, 16), "pinky": (17, 20)}
    for name, (mcp, tip) in idx.items():
        a = np.radians(angles[name])
        d = np.array([np.cos(a), -np.sin(a)])
        pts[mcp] = d
        pts[tip] = d * (EXTENDED if name in extended_fingers else CURLED)
    return pts + np.asarray(offset)


def interpolate(waypoints, n_per_leg=12):
    out = []
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        for t in np.linspace(0, 1, n_per_leg, endpoint=False):
            out.append((1 - t) * np.asarray(a) + t * np.asarray(b))
    out.append(np.asarray(waypoints[-1]))
    return out


def run(tracker, fingers, path, noise=0.0, seed=0, thumb_out=False):
    """Feed a fingertip path; the whole hand translates so the drawing tip follows it."""
    rng = np.random.default_rng(seed)
    base = make_hand(fingers, thumb_out=thumb_out)
    tip_idx = 20 if "pinky" in fingers and "index" not in fingers else 8
    results = []
    for i, p in enumerate(path):
        hand = make_hand(fingers, offset=np.asarray(p) - base[tip_idx], thumb_out=thumb_out)
        hand = hand + rng.normal(0, noise, hand.shape)
        results.append(tracker.update(hand, now=i / FPS))
    return [r for r in results if r]


Z_PATH = [(0, 0), (1.3, 0), (0, 1.3), (1.3, 1.3)]            # right, down-left, right
J_PATH = [(0, 0), (0, 1.1), (-0.6, 0.9)]                       # down, then hook sideways/up


class TestHandshape(unittest.TestCase):
    def test_i_shape_is_j(self):
        self.assertEqual(handshape(make_hand({"pinky"})), "J")

    def test_pointing_shape_is_z(self):
        self.assertEqual(handshape(make_hand({"index"})), "Z")

    def test_open_hand_is_hello(self):
        self.assertEqual(handshape(make_hand({"index", "middle", "ring", "pinky"})), "HELLO")

    def test_fist_is_the_yes_shape(self):
        self.assertEqual(handshape(make_hand(set())), "YES")

    def test_thumb_index_pinky_is_ily(self):
        self.assertEqual(handshape(make_hand({"index", "pinky"}, thumb_out=True)), "ILY")

    def test_index_and_pinky_without_thumb_is_not_ily(self):
        self.assertIsNone(handshape(make_hand({"index", "pinky"})))

    def test_letter_y_and_l_shapes_are_not_ily(self):
        self.assertNotEqual(handshape(make_hand({"pinky"}, thumb_out=True)), "ILY")   # Y
        self.assertNotEqual(handshape(make_hand({"index"}, thumb_out=True)), "ILY")   # L


class TestMotionTracker(unittest.TestCase):

    def test_detects_z(self):
        self.assertEqual(run(MotionTracker(), {"index"}, interpolate(Z_PATH), noise=0.01), ["Z"])

    def test_detects_mirrored_z(self):
        mirrored = [(-x, y) for x, y in Z_PATH]
        self.assertEqual(run(MotionTracker(), {"index"}, interpolate(mirrored), noise=0.01), ["Z"])

    def test_detects_j(self):
        self.assertEqual(run(MotionTracker(), {"pinky"}, interpolate(J_PATH), noise=0.01), ["J"])

    def test_detects_mirrored_j(self):
        mirrored = [(-x, y) for x, y in J_PATH]
        self.assertEqual(run(MotionTracker(), {"pinky"}, interpolate(mirrored), noise=0.01), ["J"])

    def test_static_pose_does_nothing(self):
        still = [(0.0, 0.0)] * 60
        self.assertEqual(run(MotionTracker(), {"pinky"}, still, noise=0.005), [])
        self.assertEqual(run(MotionTracker(), {"index"}, still, noise=0.005), [])

    def test_horizontal_wave_is_not_z(self):
        wave = interpolate([(0, 0), (1.3, 0), (0, 0), (1.3, 0)])
        self.assertEqual(run(MotionTracker(), {"index"}, wave), [])

    def test_vertical_down_and_back_is_not_j(self):
        down_up = interpolate([(0, 0), (0, 1.1), (0, 0)])
        self.assertEqual(run(MotionTracker(), {"pinky"}, down_up), [])

    def test_wrong_handshape_never_triggers(self):
        open_hand = {"index", "middle", "ring", "pinky"}
        self.assertEqual(run(MotionTracker(), open_hand, interpolate(Z_PATH)), [])

    def test_wave_is_hello(self):
        open_hand = {"index", "middle", "ring", "pinky"}
        wave = interpolate([(0, 0), (0.9, 0.05), (0, 0), (0.9, 0.05), (0, 0), (0.9, 0.05)], 8)
        self.assertEqual(run(MotionTracker(), open_hand, wave, noise=0.01), ["HELLO"])

    def test_still_open_hand_is_not_hello(self):
        open_hand = {"index", "middle", "ring", "pinky"}
        self.assertEqual(run(MotionTracker(), open_hand, [(0.0, 0.0)] * 80, noise=0.01), [])

    def test_wave_with_fist_is_not_hello(self):
        wave = interpolate([(0, 0), (0.9, 0), (0, 0), (0.9, 0), (0, 0), (0.9, 0)], 8)
        self.assertEqual(run(MotionTracker(), set(), wave), [])

    def test_sloppy_curvy_z_is_detected(self):
        # Rounded corners, extra waypoints and noticeable jitter, like a hand-drawn Z.
        sloppy = [(0, 0), (0.6, -0.03), (1.2, 0.05), (1.3, 0.2), (0.9, 0.5), (0.4, 0.8),
                  (0.0, 1.1), (0.05, 1.3), (0.6, 1.28), (1.3, 1.32)]
        self.assertEqual(run(MotionTracker(), {"index"}, interpolate(sloppy, 5), noise=0.03), ["Z"])

    def test_handshape_flicker_does_not_lose_the_path(self):
        tracker = MotionTracker()
        base = make_hand({"index"})
        fired = []
        for i, p in enumerate(interpolate(Z_PATH)):
            fingers = {"index", "middle", "ring"} if i in (10, 11, 12, 24, 25) else {"index"}
            hand = make_hand(fingers, offset=np.asarray(p) - base[8])
            r = tracker.update(hand, now=i / FPS)
            if r:
                fired.append(r)
        self.assertEqual(fired, ["Z"])

    def test_cooldown_prevents_double_trigger(self):
        path = interpolate(Z_PATH) + interpolate(Z_PATH)[:5]
        self.assertEqual(len(run(MotionTracker(), {"index"}, path)), 1)


OPEN = {"index", "middle", "ring", "pinky"}
NOD = interpolate([(0, 0), (0, 0.5), (0, 0), (0, 0.5), (0, 0), (0, 0.5)], 6)


def circle(turns, radius=0.6, points_per_turn=24):
    t = np.linspace(0, 2 * np.pi * turns, int(points_per_turn * turns) + 1)
    return [(radius * np.cos(a), radius * np.sin(a)) for a in t]


class TestPhraseSigns(unittest.TestCase):

    def test_held_ily_is_detected(self):
        out = run(MotionTracker(), {"index", "pinky"}, [(0, 0)] * 40, noise=0.005, thumb_out=True)
        self.assertEqual(out, ["ILY"])

    def test_ily_flashed_briefly_is_ignored(self):
        self.assertEqual(run(MotionTracker(), {"index", "pinky"}, [(0, 0)] * 10, thumb_out=True), [])

    def test_ily_only_fires_once_while_held(self):
        self.assertEqual(len(run(MotionTracker(), {"index", "pinky"}, [(0, 0)] * 90, thumb_out=True)), 1)

    def test_fist_nod_is_yes(self):
        self.assertEqual(run(MotionTracker(), set(), NOD, noise=0.01), ["YES"])

    def test_still_fist_is_not_yes(self):
        self.assertEqual(run(MotionTracker(), set(), [(0, 0)] * 80, noise=0.01), [])

    def test_tiny_jitter_in_a_fist_is_not_yes(self):
        jitter = interpolate([(0, 0), (0, 0.08), (0, 0), (0, 0.08), (0, 0), (0, 0.08)], 6)
        self.assertEqual(run(MotionTracker(), set(), jitter), [])

    def test_open_hand_circle_is_please(self):
        self.assertEqual(run(MotionTracker(), OPEN, circle(1.3), noise=0.01), ["PLEASE"])

    def test_circle_in_the_other_direction_is_please(self):
        reverse = [(x, -y) for x, y in circle(1.3)]
        self.assertEqual(run(MotionTracker(), OPEN, reverse, noise=0.01), ["PLEASE"])

    def test_wave_is_still_hello_not_please(self):
        wave = interpolate([(0, 0), (0.9, 0.05), (0, 0), (0.9, 0.05), (0, 0), (0.9, 0.05)], 8)
        self.assertEqual(run(MotionTracker(), OPEN, wave, noise=0.01), ["HELLO"])

    def test_half_circle_is_not_please(self):
        self.assertEqual(run(MotionTracker(), OPEN, circle(0.5), noise=0.01), [])

    def test_nodding_an_open_hand_is_not_yes(self):
        self.assertNotIn("YES", run(MotionTracker(), OPEN, NOD))

    def test_circling_a_fist_is_not_please(self):
        self.assertNotIn("PLEASE", run(MotionTracker(), set(), circle(1.3)))


class TestMotionThroughApi(unittest.TestCase):
    """Drawing a Z / J through /api/recognize/landmarks commits the letter."""

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from app.main import app
        cls.client = TestClient(app)

    def _draw(self, fingers, waypoints, thumb_out=False, path=None):
        from app.main import motion_tracker, stabilizer
        motion_tracker.reset(clear_cooldown=True)
        stabilizer.clear()
        base = make_hand(fingers)
        tip_idx = 20 if "index" not in fingers else 8
        committed, last_text = [], ""
        # A fixed 30 fps clock keeps the test independent of how fast the machine answers requests.
        frame = iter(range(10_000))
        real_update = motion_tracker.update
        with mock.patch.object(motion_tracker, "update",
                               side_effect=lambda pts: real_update(pts, now=next(frame) / FPS)):
            for p in (path if path is not None else interpolate(waypoints)):
                hand = make_hand(fingers, offset=np.asarray(p) - base[tip_idx], thumb_out=thumb_out) * 0.1 + 0.5
                body = {"landmarks": [{"x": float(x), "y": float(y), "z": 0.0} for x, y in hand]}
                data = self.client.post("/api/recognize/landmarks", json=body).json()
                if data["stabilizer"].get("committed_action"):
                    committed.append(data["stabilizer"]["committed_action"])
                last_text = data["stabilizer"]["text_state"]["full_text"]
        return committed, last_text

    def test_z_is_committed_via_api(self):
        committed, text = self._draw({"index"}, Z_PATH)
        self.assertIn("Z", committed)
        self.assertTrue(text.endswith("Z"))

    def test_j_is_committed_via_api(self):
        committed, text = self._draw({"pinky"}, J_PATH)
        self.assertIn("J", committed)
        self.assertTrue(text.endswith("J"))

    def test_wave_commits_hello_word_via_api(self):
        open_hand = {"index", "middle", "ring", "pinky"}
        wave = [(0, 0), (0.9, 0.05), (0, 0), (0.9, 0.05), (0, 0), (0.9, 0.05)]
        committed, text = self._draw(open_hand, wave)
        self.assertIn("HELLO", committed)
        self.assertEqual(text, "HELLO")

    def test_ily_commits_the_phrase_via_api(self):
        committed, text = self._draw({"index", "pinky"}, None, thumb_out=True, path=[(0, 0)] * 45)
        self.assertIn("I LOVE YOU", committed)
        self.assertEqual(text.strip(), "I LOVE YOU")

    def test_yes_commits_via_api(self):
        committed, text = self._draw(set(), None, path=NOD)
        self.assertIn("YES", committed)
        self.assertEqual(text.strip(), "YES")

    def test_please_commits_via_api(self):
        committed, text = self._draw(OPEN, None, path=circle(1.3))
        self.assertIn("PLEASE", committed)
        self.assertEqual(text.strip(), "PLEASE")

    def test_static_prediction_never_commits_j_or_z(self):
        from app.main import motion_tracker, stabilizer
        motion_tracker.reset()
        stabilizer.clear()
        hand = make_hand({"pinky"}) * 0.1 + 0.5
        body = {"landmarks": [{"x": float(x), "y": float(y), "z": 0.0} for x, y in hand]}
        for _ in range(30):
            data = self.client.post("/api/recognize/landmarks", json=body).json()
            self.assertNotIn(data["stabilizer"].get("committed_action"), ("J", "Z"))


if __name__ == "__main__":
    unittest.main()
