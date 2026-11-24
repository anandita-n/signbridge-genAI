# SignBridge AI

A webcam app that turns American Sign Language into English text and speech. It recognises fingerspelled letters, a few motion signs, and 100 whole words, then uses Gemini to turn the result into clean sentences.

## Features

- **Letters (A-Z):** MediaPipe hand landmarks feed a Random Forest classifier, with confidence gating and a temporal stabilizer to avoid jitter.
- **Motion signs:** J, Z, HELLO, PLEASE, YES and I LOVE YOU are detected by a rule-based fingertip-path tracker (`app/motion_detector.py`).
- **Whole words:** a 1D-CNN over hand and body-pose landmarks recognises 100 words from the WLASL100 vocabulary. Words are spotted automatically, or recorded manually. If the model is unsure, tap one of its top-5 guesses to add it.
- **Vision-language captioning:** "Caption This Sign" sends the camera frame to Gemini 2.5 Flash, which describes the handshape and picks the matching letter among the classifier's candidates. A hand check runs first, so empty frames are never sent.
- **Sentence refinement:** Gemini fixes spelling and grammar in four styles (casual, formal, concise, empathetic) and suggests replies for a two-way conversation.
- **Speech and history:** text-to-speech, a Deaf/hearing dialogue view, and an exportable history.

Without a Gemini key, or if the rate limit is hit, the app falls back to offline captioning and a rule-based grammar fixer.

## Tech stack

Python, FastAPI, scikit-learn, PyTorch (CPU), MediaPipe (browser Hands, plus Tasks HandLandmarker and PoseLandmarker on the server), OpenCV, NumPy, Google Gemini 2.5 Flash (`google-genai`), and a plain HTML/CSS/JavaScript front end.

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu   # needed for word recognition
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 and allow camera access.

For Gemini features, create a git-ignored `.env` file containing `GEMINI_API_KEY=your-key`.

## Accuracy and limitations

| Component | Result |
| :--- | :--- |
| Letters, held-out real data | 98.1%. This is optimistic: frames from the same recordings appear in both train and test. |
| Letters, synthetic-only model on real data | 35.6% (why real data is mixed into training). |
| Words, official WLASL100 test split | 81% top-1, 95% top-5 (chance is 1%). |
| Words, signers unseen in training | about 62% top-1, 84% top-5. Webcam results are often lower. |

- Only 100 words are supported, and common words such as "name" or "thank you" are missing.
- Motion-sign thresholds are heuristics, so they may need tuning for your camera.
- Word recognition runs on CPU and takes roughly 0.6 s on an idle machine, longer under load.
- One signer at a time, with the hand in good light. Facial expression and ASL grammar are not modelled.
- Gemini's free tier allows about 5 requests per minute.

## Training

```bash
python scripts/import_hf_landmarks.py     # real letter landmarks (sid220/asl-now-fingerspelling, MIT)
python scripts/train_model.py             # letter classifier

# Words: download aipieces/WLASL100 from Hugging Face into data/wlasl100/ first
python scripts/extract_sequences.py --workers 4
python scripts/train_word_model.py        # add --signer-split for the unseen-signer estimate
```

WLASL is for academic and non-commercial use only.

## Tests

```bash
python -m unittest discover tests
```

All 74 tests run offline, with no API key or network needed.

## Main API routes

| Route | Purpose |
| :--- | :--- |
| `POST /api/recognize/landmarks` | Classify hand landmarks and update the sign buffer |
| `POST /api/word/recognize` | Recognise a word from a short frame sequence |
| `POST /api/buffer/word` | Add a word the user picked from the guesses |
| `POST /api/caption` | Vision-language caption of a camera frame |
| `POST /api/refine` | Turn raw signs into a sentence |
| `POST /api/chat/reply` | Suggest replies for the hearing partner |
| `GET /api/status` | Model readiness and config |

## Privacy

Hand landmarks are computed in the browser. Frames are sent to Google only when you use captioning, and no video is stored on the server.
