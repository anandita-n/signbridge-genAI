"""Tests for the word-level recognition API (offline; uses blank frames and mocking)."""

import base64
import unittest
from unittest import mock

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app, stabilizer, word_service


def frame_data_url(value=128, size=96):
    ok, buf = cv2.imencode(".jpg", np.full((size, size, 3), value, np.uint8))
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


@unittest.skipUnless(word_service.ready, "word model not available")
class TestWordApi(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def setUp(self):
        stabilizer.clear()

    def test_vocab_lists_words(self):
        data = self.client.get("/api/word/vocab").json()
        self.assertTrue(data["ready"])
        self.assertEqual(len(data["words"]), 100)
        self.assertIn("what", data["words"])

    def test_status_reports_word_model(self):
        data = self.client.get("/api/status").json()
        self.assertTrue(data["word_model_ready"])
        self.assertEqual(data["word_vocab_size"], 100)

    def test_too_few_frames_rejected(self):
        resp = self.client.post("/api/word/recognize", json={"frames": [frame_data_url()] * 3})
        self.assertEqual(resp.status_code, 400)

    def test_blank_frames_report_no_hands(self):
        resp = self.client.post("/api/word/recognize", json={"frames": [frame_data_url()] * 12})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertFalse(data["accepted"])
        self.assertIsNone(data["word"])
        self.assertIn("No hands", data["message"])
        self.assertEqual(stabilizer.get_text_state()["full_text"], "")

    def test_confident_word_is_added_to_buffer(self):
        fake = {"top": [{"word": "help", "confidence": 0.9}, {"word": "want", "confidence": 0.05}],
                "hand_frame_ratio": 0.8}
        with mock.patch.object(word_service, "recognize", return_value=fake):
            data = self.client.post("/api/word/recognize", json={"frames": [frame_data_url()] * 10}).json()
        self.assertTrue(data["accepted"])
        self.assertEqual(data["word"], "help")
        self.assertEqual(data["text_state"]["full_text"], "HELP")

    def test_low_confidence_word_is_not_added(self):
        fake = {"top": [{"word": "help", "confidence": 0.2}], "hand_frame_ratio": 0.8}
        with mock.patch.object(word_service, "recognize", return_value=fake):
            data = self.client.post("/api/word/recognize", json={"frames": [frame_data_url()] * 10}).json()
        self.assertFalse(data["accepted"])
        self.assertIsNone(data["text_state"])
        self.assertEqual(stabilizer.get_text_state()["full_text"], "")

    def test_stricter_cutoff_rejects_medium_confidence(self):
        fake = {"top": [{"word": "help", "confidence": 0.45}], "hand_frame_ratio": 0.8}
        with mock.patch.object(word_service, "recognize", return_value=fake):
            data = self.client.post("/api/word/recognize",
                                    json={"frames": [frame_data_url()] * 10, "min_confidence": 0.5}).json()
        self.assertFalse(data["accepted"])
        self.assertEqual(stabilizer.get_text_state()["full_text"], "")

    def test_picking_a_guess_adds_the_word(self):
        data = self.client.post("/api/buffer/word", json={"word": "help"}).json()
        self.assertEqual(data["full_text"], "HELP")

    def test_picked_word_rejects_non_letters(self):
        self.assertEqual(self.client.post("/api/buffer/word", json={"word": "123"}).status_code, 400)

    def test_two_words_build_a_phrase(self):
        for word in ("want", "eat"):
            fake = {"top": [{"word": word, "confidence": 0.8}], "hand_frame_ratio": 0.9}
            with mock.patch.object(word_service, "recognize", return_value=fake):
                self.client.post("/api/word/recognize", json={"frames": [frame_data_url()] * 10})
        self.assertEqual(stabilizer.get_text_state()["full_text"], "WANT EAT")


if __name__ == "__main__":
    unittest.main()
