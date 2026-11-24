"""Unit tests for the VLM image captioning engine (offline paths, no network)."""

import base64
import unittest

from app.vision_engine import VisionEngine, HANDSHAPE_DESCRIPTIONS


class _NoClientLLM:
    """Stand-in LLM engine with no Gemini client, forcing the offline path."""
    client = None
    model_name = "test-model"


class TestVisionEngine(unittest.TestCase):

    def setUp(self):
        self.engine = VisionEngine(_NoClientLLM())

    def test_offline_caption_uses_top_candidate(self):
        result = self.engine.caption_sign(
            b"fake-image-bytes",
            [{"sign": "M", "confidence": 0.6}, {"sign": "N", "confidence": 0.3}],
        )
        self.assertEqual(result["vlm_sign"], "M")
        self.assertIn("M", result["caption"])
        self.assertTrue(result["agrees_with_classifier"])
        self.assertIn("offline", result["provider"].lower())

    def test_offline_caption_without_candidates(self):
        result = self.engine.caption_sign(b"fake-image-bytes", None)
        self.assertEqual(result["vlm_sign"], "NONE")
        self.assertEqual(result["candidates"], [])

    def test_invalid_candidates_are_dropped(self):
        result = self.engine.caption_sign(
            b"x", [{"sign": "??", "confidence": 0.9}, {"sign": "B", "confidence": "bad"}, {"sign": "A", "confidence": 0.5}]
        )
        self.assertEqual([c["sign"] for c in result["candidates"]], ["A"])

    def test_parse_json_handles_code_fences(self):
        parsed = VisionEngine._parse_json('```json\n{"caption": "a fist", "sign": "A"}\n```')
        self.assertEqual(parsed["sign"], "A")

    def test_every_letter_has_a_description(self):
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            self.assertIn(letter, HANDSHAPE_DESCRIPTIONS)


class TestCaptionEndpoint(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from app.main import app, vision_engine
        cls.client = TestClient(app)
        # Force offline mode so the test never calls the network.
        cls._saved_llm = vision_engine.llm
        vision_engine.llm = _NoClientLLM()

    @classmethod
    def tearDownClass(cls):
        from app.main import vision_engine
        vision_engine.llm = cls._saved_llm

    def test_caption_endpoint_offline(self):
        image = "data:image/jpeg;base64," + base64.b64encode(b"fake").decode()
        resp = self.client.post(
            "/api/caption",
            json={"image_base64": image, "candidates": [{"sign": "L", "confidence": 0.8}]},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["vlm_sign"], "L")
        self.assertIn("caption", data)

    def test_caption_endpoint_blank_frame_has_no_hand(self):
        import numpy as np
        import cv2
        ok, buf = cv2.imencode(".jpg", np.full((240, 320, 3), 200, np.uint8))
        image = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
        resp = self.client.post(
            "/api/caption",
            json={"image_base64": image, "candidates": [{"sign": "M", "confidence": 0.5}]},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["vlm_sign"], "NONE")
        self.assertIn("hand check", data["provider"])

    def test_caption_endpoint_rejects_bad_image(self):
        resp = self.client.post("/api/caption", json={"image_base64": "!!!not base64!!!"})
        self.assertEqual(resp.status_code, 400)


if __name__ == "__main__":
    unittest.main()


class TestHandFacts(unittest.TestCase):
    """The measured finger facts and the hand crop that are given to the VLM."""

    @staticmethod
    def _hand(extended=(), thumb_out=False):
        pts = [[0.5, 0.8, 0.0] for _ in range(21)]
        pts[9] = [0.5, 0.6, 0.0]                       # middle knuckle 0.2 above the wrist
        for name, (mcp, tip) in {"index": (5, 8), "middle": (9, 12), "ring": (13, 16), "pinky": (17, 20)}.items():
            x = {"index": 0.44, "middle": 0.5, "ring": 0.56, "pinky": 0.62}[name]
            pts[mcp] = [x, 0.6, 0.0]
            pts[tip] = [x, 0.2 if name in extended else 0.68, 0.0]
        pts[4] = [0.3, 0.7, 0.0] if thumb_out else [0.5, 0.65, 0.0]
        return pts

    def test_facts_name_the_extended_fingers(self):
        from app.vision_engine import describe_hand
        text = describe_hand(self._hand(extended={"index", "middle"}))
        self.assertIn("index finger: fully extended", text)
        self.assertIn("ring finger: curled", text)
        self.assertIn("pointing up", text)

    def test_facts_report_thumb_out(self):
        from app.vision_engine import describe_hand
        self.assertIn("sticking out", describe_hand(self._hand(thumb_out=True)))
        self.assertIn("tucked", describe_hand(self._hand()))

    def test_crop_is_a_larger_square_around_the_hand(self):
        import cv2
        import numpy as np
        from app.vision_engine import crop_to_hand
        img = np.zeros((480, 640, 3), np.uint8)
        ok, buf = cv2.imencode(".jpg", img)
        out = cv2.imdecode(np.frombuffer(crop_to_hand(buf.tobytes(), self._hand()), np.uint8), cv2.IMREAD_COLOR)
        self.assertGreaterEqual(max(out.shape[:2]), 512)
        self.assertLess(out.shape[1], 640 * 2)

    def test_bad_landmarks_return_the_original_image(self):
        from app.vision_engine import crop_to_hand
        self.assertEqual(crop_to_hand(b"not-an-image", [[0, 0, 0]] * 21), b"not-an-image")

    def test_prompt_includes_measured_facts(self):
        engine = VisionEngine(_NoClientLLM())
        prompt = engine._build_prompt([{"sign": "U", "confidence": 0.6}], "- index finger: fully extended")
        self.assertIn("index finger: fully extended", prompt)
