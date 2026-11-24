/**
 * Signet — Every sign has something to say
 * Frontend controller: WebRTC, MediaPipe hand tracking, REST classification,
 * Generative AI sentence refinement, Web Speech TTS, ASL Guide with images.
 */

// ─── Application State ───────────────────────────────────────────────────────
const state = {
  cameraActive:     false,
  stream:           null,
  hands:            null,
  camera:           null,
  isMirrored:       true,
  detectorMode:     "client_mediapipe",
  selectedStyle:    "casual",
  confidenceThreshold: 0.35,
  stabilityFrames:  4,
  geminiKey:        "",
  selectedVoice:    "Samantha",
  lastPrediction:   null,
  fpsCounter:       { frames: 0, lastTime: performance.now(), fps: 0 },
  translationHistory: []
};

// ─── DOM Elements ─────────────────────────────────────────────────────────────
const elements = {
  // Tabs
  navTabs:  document.querySelectorAll(".nav-tab"),
  tabPanes: document.querySelectorAll(".tab-pane"),

  // Status
  backendStatus:    document.getElementById("backend-status-text"),
  geminiDot:        document.getElementById("gemini-dot"),
  geminiStatus:     document.getElementById("gemini-status-text"),

  // Video & Prediction
  video:            document.getElementById("webcam-video"),
  canvas:           document.getElementById("overlay-canvas"),
  toggleCameraBtn:  document.getElementById("toggle-camera-btn"),
  flipCameraBtn:    document.getElementById("flip-camera-btn"),
  cameraModeSelect: document.getElementById("camera-mode-select"),
  fpsBadge:         document.getElementById("fps-badge"),
  handStatusBadge:  document.getElementById("hand-status-badge"),
  predictedChar:    document.getElementById("predicted-char"),
  commitDetectedBtn:document.getElementById("commit-detected-btn"),
  holdRingCircle:   document.getElementById("hold-ring-circle"),
  holdStatusLabel:  document.getElementById("hold-status-label"),
  confidencePct:    document.getElementById("confidence-pct"),
  stabilityPct:     document.getElementById("stability-pct"),
  candidatesContainer: document.getElementById("candidates-container"),

  // Workspace & Buffer
  charStreamTokens: document.getElementById("char-stream-tokens"),
  fullTextInput:    document.getElementById("full-text-input"),
  actionSpaceBtn:   document.getElementById("action-space-btn"),
  actionBackspaceBtn: document.getElementById("action-backspace-btn"),
  actionClearBtn:   document.getElementById("action-clear-btn"),
  demoPhraseBtn:    document.getElementById("demo-phrase-btn"),

  // Generative AI
  styleChips:         document.querySelectorAll(".style-chip"),
  refineAiBtn:        document.getElementById("refine-ai-btn"),
  activeLlmTag:       document.getElementById("active-llm-tag"),
  refinedResultCard:  document.getElementById("refined-result-card"),
  refinedTextContent: document.getElementById("refined-text-content"),
  replyChipsList:     document.getElementById("reply-chips-list"),
  copyRefinedBtn:     document.getElementById("copy-refined-btn"),
  speakRefinedBtn:    document.getElementById("speak-refined-btn"),

  // Alphabet & History
  alphabetGridContainer: document.getElementById("alphabet-grid-container"),
  historyListContainer:  document.getElementById("history-list-container"),
  exportHistoryBtn:      document.getElementById("export-history-json-btn"),
  clearHistoryBtn:       document.getElementById("clear-history-btn"),

  // Settings Modal
  settingsModal:          document.getElementById("settings-modal"),
  openSettingsBtn:        document.getElementById("open-settings-btn"),
  closeSettingsBtn:       document.getElementById("close-settings-btn"),
  cancelSettingsBtn:      document.getElementById("cancel-settings-btn"),
  saveSettingsBtn:        document.getElementById("save-settings-btn"),
  settingGeminiKey:       document.getElementById("setting-gemini-key"),
  settingConfThreshold:   document.getElementById("setting-conf-threshold"),
  confValLabel:           document.getElementById("conf-val-label"),
  settingStabilityFrames: document.getElementById("setting-stability-frames"),
  stabValLabel:           document.getElementById("stab-val-label"),
  settingTtsVoice:        document.getElementById("setting-tts-voice")
};

// ─── Hand topology ────────────────────────────────────────────────────────────
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [9, 10], [10, 11], [11, 12],
  [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17]
];

// Warm botanical palette — matches the site colours
const FINGER_COLORS = {
  thumb:  "#a44a3f",   // terracotta
  index:  "#5a8a60",   // sage green
  middle: "#8a7060",   // warm brown
  ring:   "#f19c79",   // peach
  pinky:  "#cbdfbd",   // light sage
  palm:   "#c8bfaa"    // warm bone
};

// ─── ASL Guide data ──────────────────────────────────────────────────────────
const ASL_GUIDE_DATA = {
  "A": "Closed fist with thumb resting against index finger side.",
  "B": "Four fingers straight up together, thumb folded across palm.",
  "C": "All fingers curved outward in a smooth C-cup shape.",
  "D": "Index finger up; middle, ring, pinky curved to meet thumb tip.",
  "E": "All fingers curled down, fingertips resting on thumb edge.",
  "F": "Index and thumb form an 'O'; middle, ring, pinky extend up.",
  "G": "Index pointing forward horizontally, thumb parallel above.",
  "H": "Index and middle fingers extended forward horizontally.",
  "I": "Pinky extended up; other fingers in fist, thumb across.",
  "J": "Pinky up, sweeping downward in a 'J' arc.",
  "K": "Index up, middle at 45°, thumb between them.",
  "L": "Index pointing up, thumb out at 90° — L-shape.",
  "M": "Fist, thumb tucked under index, middle and ring fingers.",
  "N": "Fist, thumb tucked under index and middle fingers.",
  "O": "Fingers curved inward, tips touching thumb — an 'O'.",
  "P": "K-shape held inverted, pointing downward.",
  "Q": "G-shape held inverted, pointing downward.",
  "R": "Index and middle fingers crossed, held straight up.",
  "S": "Closed fist, thumb wrapped across the front of fingers.",
  "T": "Fist, thumb poking up between index and middle knuckles.",
  "U": "Index and middle fingers straight up together.",
  "V": "Index and middle fingers spread apart — peace sign.",
  "W": "Index, middle and ring fingers spread and straight up.",
  "X": "Index finger bent into a hook; other fingers closed.",
  "Y": "Thumb and pinky extended wide; middle three curled.",
  "Z": "Index finger traces a 'Z' path in the air.",
  "SPACE": "Flat open hand held forward — word separator.",
  "DEL":   "Closed fist with a flick — erases previous letter.",
  "CLEAR": "Open palm waving side to side — clears all text."
};

/* ════════════════════════════════════════════════════════════════
   1. Initialization & Event Listeners
════════════════════════════════════════════════════════════════ */

document.addEventListener("DOMContentLoaded", () => {
  setupNavigation();
  setupSettingsModal();
  setupBufferActions();
  setupGenerativeAiActions();
  setupSpeechVoices();
  loadSystemStatus();
  renderAlphabetGuide();
  loadTranslationHistory();

  // Camera mode change
  elements.cameraModeSelect.addEventListener("change", (e) => {
    state.detectorMode = e.target.value;
  });

  // Quick alphabet strip
  renderQuickAlphabetStrip();

  // Commit button & click on badge
  if (elements.commitDetectedBtn) {
    elements.commitDetectedBtn.addEventListener("click", commitCurrentSign);
  }
  elements.predictedChar.addEventListener("click", commitCurrentSign);

  // Camera
  elements.toggleCameraBtn.addEventListener("click", toggleCamera);
  elements.flipCameraBtn.addEventListener("click", () => {
    state.isMirrored = !state.isMirrored;
    elements.video.style.transform  = state.isMirrored ? "scaleX(-1)" : "scaleX(1)";
    elements.canvas.style.transform = state.isMirrored ? "scaleX(-1)" : "scaleX(1)";
  });

  // Keyboard shortcut: Spacebar commits sign when not typing
  document.addEventListener("keydown", (e) => {
    if (e.code === "Space" && document.activeElement !== elements.fullTextInput) {
      e.preventDefault();
      commitCurrentSign();
    }
  });

  // Sync manual text edits to backend buffer
  elements.fullTextInput.addEventListener("input", (e) => {
    fetch("/api/buffer/set", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: e.target.value })
    }).catch(console.error);
  });
});


/* ════════════════════════════════════════════════════════════════
   2. Tab Navigation
════════════════════════════════════════════════════════════════ */

function setupNavigation() {
  elements.navTabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const targetTab = tab.getAttribute("data-tab");
      elements.navTabs.forEach(t  => t.classList.remove("active"));
      elements.tabPanes.forEach(p => p.classList.remove("active"));
      tab.classList.add("active");
      const pane = document.getElementById(`tab-${targetTab}`);
      if (pane) pane.classList.add("active");
    });
  });
}


/* ════════════════════════════════════════════════════════════════
   3. Camera & Hand Landmark Tracking
════════════════════════════════════════════════════════════════ */

async function toggleCamera() {
  state.cameraActive ? stopCamera() : await startCamera();
}

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      audio: false
    });

    state.stream = stream;
    elements.video.srcObject = stream;
    await elements.video.play();

    elements.canvas.width  = elements.video.videoWidth  || 640;
    elements.canvas.height = elements.video.videoHeight || 480;

    state.cameraActive = true;
    elements.toggleCameraBtn.innerHTML = `Stop Camera`;
    elements.toggleCameraBtn.style.background = "linear-gradient(135deg, #c0392b, #96281b)";

    if (!state.hands && window.Hands) initClientMediaPipe();
    startFrameLoop();
  } catch (err) {
    console.error("Camera access error:", err);
    alert("Could not access camera: " + err.message + "\nPlease verify camera permissions.");
  }
}

function stopCamera() {
  if (state.stream) {
    state.stream.getTracks().forEach(t => t.stop());
    state.stream = null;
  }
  elements.video.srcObject = null;
  state.cameraActive = false;
  elements.toggleCameraBtn.innerHTML = `Start Camera`;
  elements.toggleCameraBtn.style.background = "";
  elements.fpsBadge.textContent = "0 FPS";
  elements.handStatusBadge.textContent = "Camera Off";
  const ctx = elements.canvas.getContext("2d");
  ctx.clearRect(0, 0, elements.canvas.width, elements.canvas.height);
}

function initClientMediaPipe() {
  try {
    if (!window.Hands) {
      console.warn("[MediaPipe] Hands not in window — using backend LiteRT.");
      state.detectorMode = "backend_mediapipe";
      if (elements.cameraModeSelect) elements.cameraModeSelect.value = "backend_mediapipe";
      return;
    }
    state.hands = new Hands({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
    });
    state.hands.setOptions({
      maxNumHands: 2,
      modelComplexity: 1,
      minDetectionConfidence: 0.45,
      minTrackingConfidence: 0.45
    });
    state.hands.onResults(onMediaPipeResults);
  } catch (e) {
    console.warn("[MediaPipe] Fallback to backend:", e);
    state.detectorMode = "backend_mediapipe";
    if (elements.cameraModeSelect) elements.cameraModeSelect.value = "backend_mediapipe";
  }
}

let isProcessingFrame = false;

function startFrameLoop() {
  async function frameStep() {
    if (!state.cameraActive) return;
    calculateFPS();

    if (state.detectorMode === "client_mediapipe" && state.hands) {
      if (!isProcessingFrame && elements.video.readyState >= 2) {
        isProcessingFrame = true;
        try {
          await state.hands.send({ image: elements.video });
        } catch (err) {
          console.warn("Client MediaPipe error — falling back to backend:", err);
          state.detectorMode = "backend_mediapipe";
          if (elements.cameraModeSelect) elements.cameraModeSelect.value = "backend_mediapipe";
        }
        isProcessingFrame = false;
      }
    } else {
      if (!isProcessingFrame && elements.video.readyState >= 2) {
        isProcessingFrame = true;
        await captureAndSendBackendFrame();
        isProcessingFrame = false;
      }
    }
    requestAnimationFrame(frameStep);
  }
  requestAnimationFrame(frameStep);
}

function calculateFPS() {
  const now = performance.now();
  state.fpsCounter.frames++;
  if (now - state.fpsCounter.lastTime >= 1000) {
    state.fpsCounter.fps = Math.round(
      (state.fpsCounter.frames * 1000) / (now - state.fpsCounter.lastTime)
    );
    elements.fpsBadge.textContent = `${state.fpsCounter.fps} FPS`;
    state.fpsCounter.frames = 0;
    state.fpsCounter.lastTime = now;
  }
}

// Letters and motion signs are one-handed, so with two hands in view we classify the one held higher
// (the resting hand is lower). A small margin stops the choice flipping when both hands are level.
let signingHandY = null;
let lastSigningLandmarks = null;   // the hand the caption button describes
let lastHandSeenAt = -Infinity;   // when the live tracker last saw a hand (used by the caption button)
function pickSigningHand(hands) {
  if (hands.length === 1) { signingHandY = hands[0][0].y; return hands[0]; }
  const sorted = [...hands].sort((a, b) => a[0].y - b[0].y);
  const higher = sorted[0], lower = sorted[1];
  if (signingHandY !== null && Math.abs(lower[0].y - signingHandY) < Math.abs(higher[0].y - signingHandY)
      && lower[0].y - higher[0].y < 0.08) {
    signingHandY = lower[0].y;
    return lower;
  }
  signingHandY = higher[0].y;
  return higher;
}

async function onMediaPipeResults(results) {
  const ctx = elements.canvas.getContext("2d");
  ctx.clearRect(0, 0, elements.canvas.width, elements.canvas.height);

  if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
    elements.handStatusBadge.textContent = "No Hand Detected";
    elements.handStatusBadge.style.color = "";
    autoWordTrack(null);
    updatePredictionDisplay({ sign: "---", confidence: 0.0, candidates: [] }, { stability_ratio: 0.0 });
    return;
  }

  elements.handStatusBadge.textContent = "Hand Tracked";
  elements.handStatusBadge.style.color = "#5a8a60";

  const hands = results.multiHandLandmarks;
  lastHandSeenAt = performance.now();
  const landmarks = pickSigningHand(hands);
  lastSigningLandmarks = landmarks;
  autoWordTrack(hands);
  hands.forEach(h => drawHandLandmarks(ctx, h, elements.canvas.width, elements.canvas.height));
  elements.handStatusBadge.textContent = hands.length > 1 ? "2 Hands Tracked" : "Hand Tracked";

  try {
    const resp = await fetch("/api/recognize/landmarks", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        landmarks: landmarks,
        confidence_threshold: state.confidenceThreshold
      })
    });
    const data = await resp.json();
    handlePredictionResult(data);
  } catch (err) {
    console.error("Classification error:", err);
  }
}

async function captureAndSendBackendFrame() {
  const tmp = document.createElement("canvas");
  tmp.width = 320; tmp.height = 240;
  tmp.getContext("2d").drawImage(elements.video, 0, 0, 320, 240);
  const dataUrl = tmp.toDataURL("image/jpeg", 0.7);

  try {
    const resp = await fetch("/api/recognize/frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_base64: dataUrl, confidence_threshold: state.confidenceThreshold })
    });
    const data = await resp.json();
    const ctx = elements.canvas.getContext("2d");
    ctx.clearRect(0, 0, elements.canvas.width, elements.canvas.height);

    if (data.hand_detected && data.landmarks_pixel) {
      elements.handStatusBadge.textContent = "Hand Tracked (Backend)";
      elements.handStatusBadge.style.color = "#5a8a60";
      drawPixelLandmarks(ctx, data.landmarks_pixel, data.bbox);
    } else {
      elements.handStatusBadge.textContent = "No Hand Detected";
    }
    handlePredictionResult(data);
  } catch (err) {
    console.error("Backend frame recognition error:", err);
  }
}


/* ════════════════════════════════════════════════════════════════
   4. Canvas Landmark Drawing
════════════════════════════════════════════════════════════════ */

function drawHandLandmarks(ctx, landmarks, width, height) {
  const pts = landmarks.map(lm => ({ x: lm.x * width, y: lm.y * height }));

  ctx.lineWidth = 2.6;
  ctx.lineCap = "round";
  HAND_CONNECTIONS.forEach(([i, j]) => {
    let c = FINGER_COLORS.palm;
    if (i <= 4 && j <= 4)       c = FINGER_COLORS.thumb;
    else if (i >= 5 && j <= 8)  c = FINGER_COLORS.index;
    else if (i >= 9 && j <= 12) c = FINGER_COLORS.middle;
    else if (i >= 13 && j <= 16)c = FINGER_COLORS.ring;
    else if (i >= 17 && j <= 20)c = FINGER_COLORS.pinky;
    ctx.strokeStyle = c;
    ctx.beginPath();
    ctx.moveTo(pts[i].x, pts[i].y);
    ctx.lineTo(pts[j].x, pts[j].y);
    ctx.stroke();
  });

  pts.forEach((pt, idx) => {
    const isTip = [4, 8, 12, 16, 20].includes(idx);
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, isTip ? 5.2 : 3.6, 0, 2 * Math.PI);
    ctx.fillStyle = isTip ? "#a44a3f" : "#f6f4d2";
    ctx.fill();
    ctx.lineWidth = 1.2;
    ctx.strokeStyle = "rgba(0,0,0,0.3)";
    ctx.stroke();
  });
}

function drawPixelLandmarks(ctx, pixelPoints, bbox) {
  const sx = elements.canvas.width / 320;
  const sy = elements.canvas.height / 240;

  ctx.lineWidth = 3;
  ctx.lineCap = "round";
  HAND_CONNECTIONS.forEach(([i, j]) => {
    ctx.strokeStyle = "#a44a3f";
    ctx.beginPath();
    ctx.moveTo(pixelPoints[i][0] * sx, pixelPoints[i][1] * sy);
    ctx.lineTo(pixelPoints[j][0] * sx, pixelPoints[j][1] * sy);
    ctx.stroke();
  });

  pixelPoints.forEach(pt => {
    ctx.beginPath();
    ctx.arc(pt[0] * sx, pt[1] * sy, 5, 0, 2 * Math.PI);
    ctx.fillStyle = "#cbdfbd";
    ctx.fill();
  });
}


/* ════════════════════════════════════════════════════════════════
   5. Prediction & Text Accumulator
════════════════════════════════════════════════════════════════ */

function handlePredictionResult(data) {
  if (!data || !data.prediction) return;
  const pred = data.prediction;
  const stab = data.stabilizer || {};
  updatePredictionDisplay(pred, stab);
  if (pred.motion_detected) {   // J, Z, HELLO, I LOVE YOU, YES or PLEASE was just recognised: not a word-model sign
    AUTO_WORD.signStart = null;
    AUTO_WORD.samples.length = 0;
    AUTO_WORD.busyUntil = performance.now() + 2500;
  }
  if (stab.committed_action) flashCommittedSign(stab.committed_action);
  if (stab.text_state) updateWorkspaceBuffer(stab.text_state);
}

function updatePredictionDisplay(pred, stab) {
  state.lastPrediction = pred;
  const sign    = (stab && stab.active_sign && stab.active_sign !== "NONE") ? stab.active_sign : (pred.sign || "---");
  const conf     = pred.confidence || 0.0;
  const progress = stab.stability_ratio || 0.0;
  const isLocked = stab.is_locked || false;

  elements.predictedChar.textContent   = sign;
  elements.confidencePct.textContent   = `${(conf * 100).toFixed(0)}%`;
  elements.stabilityPct.textContent    = isLocked ? "100%" : `${Math.round(progress * 100)}%`;

  if (elements.holdRingCircle) {
    const circumference = 226;
    elements.holdRingCircle.style.strokeDashoffset = circumference * (1 - Math.min(1.0, progress));
    isLocked
      ? elements.holdRingCircle.classList.add("locked")
      : elements.holdRingCircle.classList.remove("locked");
  }

  if (elements.holdStatusLabel) {
    if (isLocked) {
      elements.holdStatusLabel.textContent = "Recorded — release to sign next";
      elements.holdStatusLabel.className   = "hold-status-pill locked";
    } else if (progress > 0.15) {
      elements.holdStatusLabel.textContent = `Holding… ${Math.round(progress * 100)}%`;
      elements.holdStatusLabel.className   = "hold-status-pill holding";
    } else {
      elements.holdStatusLabel.textContent = "Ready";
      elements.holdStatusLabel.className   = "hold-status-pill";
    }
  }

  if (pred.candidates && pred.candidates.length > 0) {
    elements.candidatesContainer.innerHTML = pred.candidates.map(c =>
      `<span class="candidate-chip"><strong>${c.sign}</strong>: ${(c.confidence * 100).toFixed(0)}%</span>`
    ).join("");
  }
}

/* ════════════════════════════════════════════════════════════════
   Vision-Language Model: Image Captioning
════════════════════════════════════════════════════════════════ */

function captureVideoFrame() {
  const video = document.getElementById("webcam-video");
  if (!video || !video.videoWidth) return null;
  const canvas = document.createElement("canvas");
  const scale = Math.min(1, 640 / video.videoWidth);
  canvas.width  = Math.round(video.videoWidth * scale);
  canvas.height = Math.round(video.videoHeight * scale);
  canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.85);
}

async function captionCurrentSign() {
  const btn = document.getElementById("caption-sign-btn");
  const frame = captureVideoFrame();
  if (!frame) {
    alert("Start the camera first, then hold up a sign.");
    return;
  }

  const candidates = (state.lastPrediction && state.lastPrediction.candidates) || [];
  btn.disabled = true;
  btn.textContent = "Captioning…";
  try {
    const res = await fetch("/api/caption", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_base64: frame,
        candidates,
        landmarks: (performance.now() - lastHandSeenAt < 1500 && lastSigningLandmarks)
          ? lastSigningLandmarks.map(p => [p.x, p.y, p.z]) : null,
        hand_detected: performance.now() - lastHandSeenAt < 1500
      })
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    document.getElementById("caption-text").textContent      = data.caption || "";
    document.getElementById("caption-vlm-sign").textContent  = data.vlm_sign || "-";
    document.getElementById("caption-agreement").textContent = data.agrees_with_classifier
      ? "Agrees with classifier" : "Differs from classifier";
    document.getElementById("caption-reasoning").textContent = data.reasoning || "";
    document.getElementById("caption-provider").textContent  = data.provider || "";
    // When no hand was found the VLM was never called, so there is no letter to compare or use.
    const noLetter = !data.vlm_sign || data.vlm_sign === "NONE";
    document.getElementById("caption-vlm-sign").parentElement.style.display = noLetter ? "none" : "";
    document.getElementById("caption-agreement").style.display = noLetter ? "none" : "";
    document.getElementById("caption-use-btn").style.display   = noLetter ? "none" : "";
    document.getElementById("caption-result").style.display  = "block";
  } catch (e) {
    console.error("Caption error:", e);
    alert("Could not caption the image. Is the server running?");
  } finally {
    btn.disabled = false;
    btn.textContent = "Caption This Sign";
  }
}

async function useCaptionedLetter() {
  const sign = document.getElementById("caption-vlm-sign").textContent.trim();
  if (!sign || sign === "-" || sign === "NONE") return;
  try {
    const res = await fetch("/api/buffer/commit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: sign })
    });
    updateWorkspaceBuffer(await res.json());
    flashCommittedSign(sign);
  } catch (e) {
    console.error("Use captioned letter error:", e);
  }
}

/* ════════════════════════════════════════════════════════════════
   Word-level sign recognition (records ~2.5 s of frames, server classifies the clip)
════════════════════════════════════════════════════════════════ */

const WORD_RECORD_MS = 2500;
const WORD_FRAME_INTERVAL_MS = 50;

function captureSmallFrame() {
  const video = document.getElementById("webcam-video");
  if (!video || !video.videoWidth) return null;
  const canvas = document.createElement("canvas");
  const scale = Math.min(1, 480 / video.videoWidth);
  canvas.width  = Math.round(video.videoWidth * scale);
  canvas.height = Math.round(video.videoHeight * scale);
  canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.7);
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

/* Counts down, then records WORD_RECORD_MS of small frames. Pauses automatic word detection meanwhile. */
async function recordClip(statusEl) {
  AUTO_WORD.inFlight = true;
  try {
    for (const n of [3, 2, 1]) {
      statusEl.textContent = `Get ready... ${n}`;
      await sleep(700);
    }
    statusEl.textContent = "Recording - sign your word now";
    const frames = [];
    const start = performance.now();
    while (performance.now() - start < WORD_RECORD_MS) {
      const f = captureSmallFrame();
      if (f) frames.push(f);
      await sleep(WORD_FRAME_INTERVAL_MS);
    }
    const seconds = (performance.now() - start) / 1000;
    return { frames, fps: frames.length / seconds };
  } finally {
    AUTO_WORD.inFlight = false;
    AUTO_WORD.busyUntil = performance.now() + AUTO_WORD.cooldownMs;
    AUTO_WORD.signStart = null;
    AUTO_WORD.samples.length = 0;
  }
}

async function postJson(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {})
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

async function recordAndRecognizeWord() {
  const btn = document.getElementById("word-sign-btn");
  const status = document.getElementById("word-status");
  const result = document.getElementById("word-result");

  if (!state.cameraActive || !captureSmallFrame()) {
    status.textContent = "Start the camera first, then try again.";
    return;
  }

  btn.disabled = true;
  result.innerHTML = "";
  try {
    const clip = await recordClip(status);
    status.textContent = "Recognising...";
    showWordResult(await postJson("/api/word/recognize", { frames: clip.frames, fps: clip.fps, commit: true }));
  } catch (e) {
    console.error("Word recognition error:", e);
    status.textContent = `Could not recognise the word: ${e.message}`;
  } finally {
    btn.disabled = false;
  }
}

function showWordResult(data) {
  const status = document.getElementById("word-status");
  const result = document.getElementById("word-result");
  status.textContent = data.message;
  result.style.display = "flex";
  // Each guess is a button: if the model was unsure, tap the word you actually signed to add it.
  result.innerHTML = (data.top || []).map(t =>
    `<button type="button" class="candidate-chip word-pick" data-word="${escapeHtml(t.word)}" ` +
    `title="Add this word">${escapeHtml(t.word)}: ${(t.confidence * 100).toFixed(0)}%</button>`
  ).join("");
  result.querySelectorAll(".word-pick").forEach(btn => btn.addEventListener("click", async () => {
    try {
      const res = await fetch("/api/buffer/word", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ word: btn.dataset.word })
      });
      updateWorkspaceBuffer(await res.json());
      flashCommittedSign(btn.dataset.word.toUpperCase());
      status.textContent = `Added "${btn.dataset.word}".`;
    } catch (e) {
      status.textContent = "Could not add that word.";
    }
  }));
  if (data.accepted && data.text_state) {
    updateWorkspaceBuffer(data.text_state);
    flashCommittedSign(data.word.toUpperCase());
  }
}

/* Automatic word spotting: a moving sign followed by a rest is cut out of a rolling buffer of
   recent frames and sent to the word model. Static fingerspelled letters barely move the wrist,
   so they never trigger this. */
const AUTO_WORD = {
  frameEveryMs: 80,        // rolling frame buffer rate
  keepMs: 4500,            // how much history is kept
  windowMs: 900,           // motion is measured over this window
  moveThreshold: 1.8,      // wrist path length (in hand sizes) inside the window that counts as "moving"
  restMs: 400,             // stillness that ends a sign
  minSignMs: 700,
  maxSignMs: 3600,
  padMs: 250,
  cooldownMs: 1800,
  minConfidence: 0.5,
  frames: [],              // {t, url}
  samples: [],             // {t, x, y, size}
  signStart: null,
  lastMoveT: 0,
  busyUntil: 0,
  inFlight: false,
};

function autoWordEnabled() {
  const el = document.getElementById("auto-word-toggle");
  return !!(el && el.checked && state.cameraActive);
}

function autoWordCaptureTick() {
  if (!autoWordEnabled()) { AUTO_WORD.frames.length = 0; return; }
  const url = captureSmallFrame();
  if (!url) return;
  const t = performance.now();
  AUTO_WORD.frames.push({ t, url });
  while (AUTO_WORD.frames.length && t - AUTO_WORD.frames[0].t > AUTO_WORD.keepMs) AUTO_WORD.frames.shift();
}

/* landmarks: MediaPipe hand landmarks for the first hand, or null when no hand is visible */
function autoWordTrack(hands) {
  if (!autoWordEnabled()) return;
  const now = performance.now();
  if (hands && hands.length) {
    // Track the average of all visible wrists, so movement of either hand starts a word and
    // swapped hand order between frames does not look like movement.
    let x = 0, y = 0, size = 0;
    hands.forEach(h => {
      x += h[0].x; y += h[0].y;
      size += Math.max(Math.hypot(h[0].x - h[9].x, h[0].y - h[9].y), 0.02);
    });
    AUTO_WORD.samples.push({ t: now, x: x / hands.length, y: y / hands.length, size: size / hands.length });
  }
  while (AUTO_WORD.samples.length && now - AUTO_WORD.samples[0].t > AUTO_WORD.windowMs) AUTO_WORD.samples.shift();

  const s = AUTO_WORD.samples;
  let path = 0;
  for (let i = 1; i < s.length; i++) path += Math.hypot(s[i].x - s[i - 1].x, s[i].y - s[i - 1].y) / s[i].size;
  const moving = path > AUTO_WORD.moveThreshold;

  if (moving) {
    if (AUTO_WORD.signStart === null) AUTO_WORD.signStart = now - AUTO_WORD.windowMs;
    AUTO_WORD.lastMoveT = now;
    return;
  }
  if (AUTO_WORD.signStart === null) return;

  const rested = now - AUTO_WORD.lastMoveT;
  const length = AUTO_WORD.lastMoveT - AUTO_WORD.signStart;
  if (rested < AUTO_WORD.restMs && length < AUTO_WORD.maxSignMs) return;   // still waiting for the sign to end

  const start = AUTO_WORD.signStart;
  AUTO_WORD.signStart = null;
  AUTO_WORD.samples.length = 0;
  if (length < AUTO_WORD.minSignMs || now < AUTO_WORD.busyUntil || AUTO_WORD.inFlight) return;
  sendAutoWord(start - AUTO_WORD.padMs, AUTO_WORD.lastMoveT + AUTO_WORD.padMs);
}

async function sendAutoWord(fromT, toT) {
  const picked = AUTO_WORD.frames.filter(f => f.t >= fromT && f.t <= toT);
  if (picked.length < 10) return;
  AUTO_WORD.inFlight = true;
  const status = document.getElementById("word-status");
  status.textContent = "Checking for a word...";
  try {
    const seconds = (picked[picked.length - 1].t - picked[0].t) / 1000;
    const res = await fetch("/api/word/recognize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        frames: picked.map(f => f.url),
        fps: picked.length / Math.max(seconds, 0.1),
        commit: true,
        min_confidence: AUTO_WORD.minConfidence
      })
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    showWordResult(await res.json());
  } catch (e) {
    console.error("Automatic word detection error:", e);
    status.textContent = "Word detection failed. You can still use Record manually.";
  } finally {
    AUTO_WORD.inFlight = false;
    AUTO_WORD.busyUntil = performance.now() + AUTO_WORD.cooldownMs;
  }
}

setInterval(autoWordCaptureTick, AUTO_WORD.frameEveryMs);

async function loadWordVocab() {
  const el = document.getElementById("word-vocab");
  const btn = document.getElementById("word-sign-btn");
  try {
    const data = await (await fetch("/api/word/vocab")).json();
    if (!data.ready) {
      el.textContent = "The word model is not installed on this server.";
      btn.disabled = true;
      return;
    }
    el.textContent = data.words.join(", ");
  } catch (e) {
    el.textContent = "Could not load the word list.";
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const wordBtn = document.getElementById("word-sign-btn");
  if (wordBtn) {
    wordBtn.addEventListener("click", recordAndRecognizeWord);
    loadWordVocab();
  }

  const capBtn = document.getElementById("caption-sign-btn");
  const useBtn = document.getElementById("caption-use-btn");
  if (capBtn) capBtn.addEventListener("click", captionCurrentSign);
  if (useBtn) useBtn.addEventListener("click", useCaptionedLetter);
});

function renderQuickAlphabetStrip() {
  const strip = document.getElementById("quick-alphabet-strip");
  if (!strip) return;
  strip.innerHTML = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("").map(l =>
    `<button class="quick-letter-btn" onclick="window.testSignLetter('${l}')">${l}</button>`
  ).join("");
}

async function commitCurrentSign() {
  const currentSign = elements.predictedChar.textContent.trim();
  if (currentSign && currentSign !== "---" && currentSign !== "NONE" && currentSign !== "UNCERTAIN") {
    try {
      const res  = await fetch("/api/buffer/commit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: currentSign })
      });
      const data = await res.json();
      updateWorkspaceBuffer(data);
      flashCommittedSign(currentSign);
    } catch (e) {
      console.error("Manual commit error:", e);
    }
  }
}

function playCommitChime() {
  try {
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return;
    const ctx = new AC();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = "sine";
    osc.frequency.setValueAtTime(587.33, ctx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(880, ctx.currentTime + 0.08);
    gain.gain.setValueAtTime(0.12, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.08);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.09);
  } catch (e) {}
}

function flashCommittedSign(sign) {
  playCommitChime();
  elements.predictedChar.style.transform   = "scale(1.12)";
  elements.predictedChar.style.borderColor = "#cbdfbd";
  elements.predictedChar.style.color       = "#5a8a60";
  elements.charStreamTokens.style.borderColor = "#cbdfbd";
  setTimeout(() => {
    elements.predictedChar.style.transform   = "";
    elements.predictedChar.style.borderColor = "";
    elements.predictedChar.style.color       = "";
    elements.charStreamTokens.style.borderColor = "";
  }, 220);
}

function updateWorkspaceBuffer(textState) {
  if (textState.char_stream && textState.char_stream.length > 0) {
    elements.charStreamTokens.innerHTML = textState.char_stream.map(ch => {
      if (ch === " ") return `<span class="char-token space">␣</span>`;
      return `<span class="char-token">${ch}</span>`;
    }).join("");
  } else {
    elements.charStreamTokens.innerHTML =
      `<span style="color: var(--text-dim); font-size: 0.88rem;">Hold ASL letter gestures stably to register letters...</span>`;
  }
  if (document.activeElement !== elements.fullTextInput) {
    elements.fullTextInput.value = textState.full_text || "";
  }
}


/* ════════════════════════════════════════════════════════════════
   6. Buffer Controls & Demo
════════════════════════════════════════════════════════════════ */

function setupBufferActions() {
  elements.actionSpaceBtn.addEventListener("click", async () => {
    const data = await (await fetch("/api/buffer/space", { method: "POST" })).json();
    updateWorkspaceBuffer(data);
  });

  elements.actionBackspaceBtn.addEventListener("click", async () => {
    const data = await (await fetch("/api/buffer/backspace", { method: "POST" })).json();
    updateWorkspaceBuffer(data);
  });

  elements.actionClearBtn.addEventListener("click", async () => {
    const data = await (await fetch("/api/buffer/clear", { method: "POST" })).json();
    updateWorkspaceBuffer(data);
    elements.refinedResultCard.style.display = "none";
  });

  const DEMO_SEQUENCES = [
    "N E D   D O C T E R   W H E R E   H O S P I T A L",
    "H E L O   M Y   N A M E   A L E X   N I C E   M E T   Y O U",
    "W H A T   T I M E   M E T I N G   S T A R T",
    "T H A N K   Y O U   F O R   H E L P",
    "P L Z   W H E R E   R E S T R O M"
  ];
  let demoIdx = 0;

  elements.demoPhraseBtn.addEventListener("click", async () => {
    const seq  = DEMO_SEQUENCES[demoIdx % DEMO_SEQUENCES.length];
    demoIdx++;
    const data = await (await fetch("/api/buffer/set", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: seq })
    })).json();
    updateWorkspaceBuffer(data);
  });
}


/* ════════════════════════════════════════════════════════════════
   7. Generative AI Refinement
════════════════════════════════════════════════════════════════ */

function setupGenerativeAiActions() {
  // Style chips
  elements.styleChips.forEach(chip => {
    chip.addEventListener("click", () => {
      elements.styleChips.forEach(c => c.classList.remove("active"));
      chip.classList.add("active");
      state.selectedStyle = chip.getAttribute("data-style");
    });
  });

  // Refine with AI
  elements.refineAiBtn.addEventListener("click", async () => {
    const rawText = elements.fullTextInput.value.trim();
    if (!rawText) { alert("Please assemble or type some sign tokens first."); return; }

    elements.refineAiBtn.disabled = true;
    elements.refineAiBtn.innerHTML = `Refining…`;

    try {
      const resp = await fetch("/api/refine", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ raw_text: rawText, style: state.selectedStyle, context: [] })
      });
      const result = await resp.json();
      displayRefinedResult(result);

      await fetch("/api/history", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          raw_text: result.raw_text,
          refined_text: result.refined_text,
          style: result.style,
          provider: result.provider
        })
      });
      loadTranslationHistory();

    } catch (err) {
      console.error("Refine error:", err);
      alert("Error refining text: " + err.message);
    } finally {
      elements.refineAiBtn.disabled = false;
      elements.refineAiBtn.innerHTML = `Refine with Generative AI`;
    }
  });

  // Copy refined text
  elements.copyRefinedBtn.addEventListener("click", () => {
    const txt = elements.refinedTextContent.textContent.trim();
    navigator.clipboard.writeText(txt);
    elements.copyRefinedBtn.textContent = "Copied";
    setTimeout(() => { elements.copyRefinedBtn.textContent = "Copy"; }, 1500);
  });

  // Speak refined text
  elements.speakRefinedBtn.addEventListener("click", () => {
    speakText(elements.refinedTextContent.textContent.trim());
  });
}

function displayRefinedResult(result) {
  elements.refinedResultCard.style.display  = "flex";
  elements.refinedTextContent.textContent   = result.refined_text;
  elements.activeLlmTag.textContent         = result.provider;

  if (result.suggestions && result.suggestions.length > 0) {
    elements.replyChipsList.innerHTML = result.suggestions.map(s =>
      `<div class="suggestion-chip" onclick="window.speakSuggestion('${escapeHtml(s)}')">
        <span>${s}</span>
        <span style="font-size: 0.8rem; color: var(--text-muted);">Speak</span>
      </div>`
    ).join("");
  } else {
    elements.replyChipsList.innerHTML =
      `<span style="font-size: 0.8rem; color: var(--text-dim);">No quick suggestions generated.</span>`;
  }
}

window.speakSuggestion = function(text) { speakText(text); };

function escapeHtml(text) {
  return text.replace(/'/g, "\\'").replace(/"/g, '&quot;');
}


/* ════════════════════════════════════════════════════════════════
   8. Text-To-Speech
════════════════════════════════════════════════════════════════ */

function setupSpeechVoices() {
  function populateVoices() {
    if ("speechSynthesis" in window) {
      const voices = window.speechSynthesis.getVoices();
      if (voices.length > 0) {
        elements.settingTtsVoice.innerHTML = voices
          .filter(v => v.lang.startsWith("en"))
          .map(v => `<option value="${v.name}">${v.name} (${v.lang})</option>`)
          .join("");
      }
    }
  }
  populateVoices();
  if ("speechSynthesis" in window) window.speechSynthesis.onvoiceschanged = populateVoices;
}

function speakText(text) {
  if (!text) return;
  if ("speechSynthesis" in window) {
    window.speechSynthesis.cancel();
    const utt   = new SpeechSynthesisUtterance(text);
    utt.rate    = 1.0;
    utt.pitch   = 1.0;
    const voices = window.speechSynthesis.getVoices();
    const sel   = voices.find(v => v.name === state.selectedVoice);
    if (sel) utt.voice = sel;
    window.speechSynthesis.speak(utt);
  }
  fetch("/api/tts/speak", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, voice: state.selectedVoice, rate: 175 })
  }).catch(console.error);
}


/* ════════════════════════════════════════════════════════════════
   10. ASL Alphabet Guide (with real photos)
════════════════════════════════════════════════════════════════ */

// Reference drawings: public-domain "Sign language X.svg" files from Wikimedia Commons (see README).
const MOTION_LETTERS = new Set(["J", "Z"]);

function renderAlphabetGuide() {
  const letters = Object.keys(ASL_GUIDE_DATA);
  elements.alphabetGridContainer.innerHTML = letters.map(letter => {
    const desc = ASL_GUIDE_DATA[letter];
    const art = letter.length === 1
      ? `<img class="letter-pose" src="/static/img/asl/${letter}.svg" alt="ASL sign for ${letter}" loading="lazy" ` +
        `onerror="this.style.display='none'">`
      : "";
    const motion = MOTION_LETTERS.has(letter) ? `<div class="letter-motion">the hand moves along the arrow</div>` : "";
    return `
      <div class="letter-card" onclick="window.testSignLetter('${letter}')" title="Click to add ${letter} to the text">
        ${art}
        <div class="letter-badge-lg">${letter}</div>
        ${motion}
        <div class="letter-desc">${desc}</div>
      </div>
    `;
  }).join("");
}

window.testSignLetter = function(letter) {
  fetch("/api/buffer/set", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: (elements.fullTextInput.value + " " + letter).trim() })
  }).then(r => r.json()).then(updateWorkspaceBuffer);
};


/* ════════════════════════════════════════════════════════════════
   11. Translation History
════════════════════════════════════════════════════════════════ */

async function loadTranslationHistory() {
  try {
    const resp = await fetch("/api/history");
    const data = await resp.json();
    state.translationHistory = data.history || [];

    if (state.translationHistory.length === 0) {
      elements.historyListContainer.innerHTML =
        `<p style="color: var(--text-dim); font-size: 0.9rem;">No translation records yet. Refine a sentence in the studio to start logging history.</p>`;
      return;
    }

    elements.historyListContainer.innerHTML = state.translationHistory.map(item => `
      <div class="history-entry">
        <div style="display:flex; flex-direction:column; gap:5px; flex:1;">
          <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap;">
            <span style="font-size:0.72rem; color:var(--text-dim);">${item.timestamp}</span>
            <span class="provider-tag">${item.style}</span>
            <span style="font-size:0.72rem; color:var(--text-muted); font-weight:600;">${item.provider}</span>
          </div>
          <div style="font-size:0.82rem; color:var(--text-muted);">Signs: <code style="background:var(--bg-surface-elevated); padding:1px 5px; border-radius:4px;">${item.raw_text}</code></div>
          <div style="font-size:1rem; font-weight:600; color:var(--text-primary);">${item.refined_text}</div>
        </div>
        <button class="btn-icon" onclick="window.speakHistory('${escapeHtml(item.refined_text)}')" title="Speak">Speak</button>
      </div>
    `).join("");
  } catch (err) {
    console.error("History load error:", err);
  }
}

window.speakHistory = function(text) { speakText(text); };

elements.clearHistoryBtn.addEventListener("click", async () => {
  await fetch("/api/history", { method: "DELETE" });
  loadTranslationHistory();
});

elements.exportHistoryBtn.addEventListener("click", () => {
  const blob = new Blob([JSON.stringify(state.translationHistory, null, 2)], { type: "application/json" });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href     = url;
  a.download = `signet_history_${Date.now()}.json`;
  a.click();
});


/* ════════════════════════════════════════════════════════════════
   12. System Status & Settings Modal
════════════════════════════════════════════════════════════════ */

async function loadSystemStatus() {
  try {
    const resp = await fetch("/api/status");
    const data = await resp.json();
    if (data.gemini_active) {
      elements.geminiDot.classList.remove("inactive");
      elements.geminiStatus.textContent = `Gemini: Online (${data.llm_model})`;
    } else {
      elements.geminiDot.classList.add("inactive");
      elements.geminiStatus.textContent = "AI Engine: Offline Rules";
    }
    state.confidenceThreshold = data.confidence_threshold;
    state.stabilityFrames     = data.stability_frames;
    elements.settingConfThreshold.value  = data.confidence_threshold;
    elements.confValLabel.textContent    = data.confidence_threshold;
    elements.settingStabilityFrames.value = data.stability_frames;
    elements.stabValLabel.textContent    = data.stability_frames;
  } catch (e) {
    console.error("Status fetch error:", e);
  }
}

function setupSettingsModal() {
  elements.openSettingsBtn.addEventListener("click", () => elements.settingsModal.classList.add("open"));
  const closeModal = () => elements.settingsModal.classList.remove("open");
  elements.closeSettingsBtn.addEventListener("click", closeModal);
  elements.cancelSettingsBtn.addEventListener("click", closeModal);

  elements.settingConfThreshold.addEventListener("input", (e) => {
    elements.confValLabel.textContent = e.target.value;
  });
  elements.settingStabilityFrames.addEventListener("input", (e) => {
    elements.stabValLabel.textContent = e.target.value;
  });

  elements.saveSettingsBtn.addEventListener("click", async () => {
    const newKey   = elements.settingGeminiKey.value.trim();
    const newConf  = parseFloat(elements.settingConfThreshold.value);
    const newStab  = parseInt(elements.settingStabilityFrames.value, 10);
    const newVoice = elements.settingTtsVoice.value;
    state.selectedVoice = newVoice;

    const payload = { confidence_threshold: newConf, stability_frames: newStab };
    if (newKey) payload.gemini_api_key = newKey;

    try {
      await fetch("/api/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      loadSystemStatus();
      closeModal();
    } catch (e) {
      alert("Error saving settings: " + e.message);
    }
  });
}
