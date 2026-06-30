"""
gesture_detector.py — Hand Gesture Recognition for InfoTraff AI Vision Command Center.

Uses the MediaPipe Tasks GestureRecognizer (0.10+) rather than the deprecated
solutions API.  The built-in model recognises Thumb_Up, Open_Palm, and Victory
(peace sign) out of the box — no brittle landmark-angle math required.

Architecture decisions:
  ① GestureRecognizer in VIDEO mode: stateful between frames, temporal tracking
    built-in.  Better than IMAGE mode (stateless) or LIVE_STREAM (async complexity).
  ② Temporal smoothing via a 7-frame majority-vote deque: the gesture must appear
    in ≥4 of the last 7 frames before it's reported as stable.  Eliminates the
    single-frame flickers that would spam the event timeline at the demo.
  ③ Hand landmarks are drawn in pure OpenCV using HandLandmarksConnections,
    with a custom cyan palette matching the enterprise dark theme.
  ④ Auto-download: if gesture_recognizer.task is missing, __init__ fetches it
    from Google's MediaPipe CDN.  File is ~25 MB; only happens once.
  ⑤ MockGestureDetector: identical interface, no model required.  Used for
    UI testing and as a venue fallback if the download fails.
"""
from __future__ import annotations

import os
import time
import logging
import urllib.request
import random
from abc import ABC, abstractmethod
from collections import Counter, deque
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision

from utils.config import (
    GESTURE_MODEL_URL,
    GESTURE_MODEL_PATH,
    GESTURE_MIN_DETECTION_CONFIDENCE,
    GESTURE_MIN_TRACKING_CONFIDENCE,
    GESTURE_HISTORY_LEN,
    GESTURE_HISTORY_VOTES,
    GESTURE_TO_EVENT,
    ASSETS_DIR,
)
from utils.helpers import draw_overlay_text

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# MediaPipe's built-in canned-gesture category names → our internal gesture keys.
# Only the three gestures specified in the brief are mapped; all others are ignored.
_MP_CATEGORY_MAP: dict[str, str] = {
    "Thumb_Up":  "thumbs_up",
    "Open_Palm": "open_palm",
    "Victory":   "peace_sign",   # MediaPipe calls the ✌ gesture "Victory"
}

# OpenCV-safe display labels (no emoji — FONT_HERSHEY cannot render Unicode).
# Emoji appear in the Qt timeline via GESTURE_TO_EVENT from config.
GESTURE_DISPLAY: dict[str, str] = {
    "thumbs_up":  "Thumbs Up",
    "open_palm":  "Open Palm",
    "peace_sign": "Peace Sign",
}

# Hand landmark drawing palette (BGR)
_COLOR_LANDMARK   = (0, 229, 255)   # cyan filled dot
_COLOR_CONNECTION = (0, 170, 220)   # slightly darker cyan for connections
_COLOR_RING       = (255, 255, 255) # white ring around each dot

# Pre-fetch connections list once at module level (cheap attribute access)
_HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS


# ──────────────────────────────────────────────────────────────────────────────
# Data Model
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class GestureResult:
    """
    Structured output from one gesture-detection pass.

    Not frozen: hand_landmarks_list contains MediaPipe NormalizedLandmark objects
    which hold internal C++ references and are not safely hashable.
    """
    gesture:             Optional[str]   # "thumbs_up" | "open_palm" | "peace_sign" | None
    display_label:       str             # OpenCV-safe label, e.g. "Thumbs Up"
    event_text:          str             # Business event text for the dashboard timeline
    hand_count:          int             # Number of hands visible in this frame
    hand_landmarks_list: list            # Raw MediaPipe NormalizedLandmark lists (for draw)
    timestamp:           float           # Unix epoch

    @property
    def is_detected(self) -> bool:
        return self.gesture is not None

    @classmethod
    def none_detected(cls) -> "GestureResult":
        """Factory for a no-gesture frame (camera init, no hands visible, etc.)."""
        return cls(
            gesture=None,
            display_label="—",
            event_text="",
            hand_count=0,
            hand_landmarks_list=[],
            timestamp=time.time(),
        )

    def __repr__(self) -> str:
        return (
            f"GestureResult({self.gesture!r}, "
            f"hands={self.hand_count}, "
            f"label={self.display_label!r})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Abstract Base
# ──────────────────────────────────────────────────────────────────────────────
class BaseGestureDetector(ABC):
    """
    Interface contract shared by GestureDetector and MockGestureDetector.

    The rest of the application depends only on this interface so that the
    real vs mock swap is invisible to downstream consumers.
    """

    @abstractmethod
    def detect(self, frame: np.ndarray) -> GestureResult:
        """Run gesture recognition on a single BGR frame."""
        ...

    @abstractmethod
    def draw(self, frame: np.ndarray, result: GestureResult) -> None:
        """Render hand landmarks and gesture label onto frame in-place."""
        ...

    @abstractmethod
    def release(self) -> None:
        """Free model resources (call when the application closes)."""
        ...


# ──────────────────────────────────────────────────────────────────────────────
# Real Detector
# ──────────────────────────────────────────────────────────────────────────────
class GestureDetector(BaseGestureDetector):
    """
    MediaPipe GestureRecognizer (Tasks API) wrapper.

    Usage:
        detector = GestureDetector()           # loads model, auto-downloads if needed
        ...
        result = detector.detect(frame)        # call every frame
        detector.draw(frame, result)           # renders landmarks + HUD label
        detector.release()                     # on application exit
    """

    def __init__(self, model_path: str = GESTURE_MODEL_PATH) -> None:
        self._model_path = model_path

        # Auto-download model if missing
        if not os.path.exists(model_path):
            log.info("gesture_recognizer.task not found — downloading …")
            self._download_model(model_path)

        # Configure Tasks API in VIDEO mode.
        # VIDEO mode maintains internal state between frames for better tracking.
        # The timestamp must be strictly monotonically increasing (enforced below).
        options = vision.GestureRecognizerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=GESTURE_MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=GESTURE_MIN_TRACKING_CONFIDENCE,
            min_tracking_confidence=GESTURE_MIN_TRACKING_CONFIDENCE,
        )
        self._recognizer = vision.GestureRecognizer.create_from_options(options)

        # Monotonic timestamp guard (MediaPipe VIDEO mode requirement)
        self._last_ts_ms: int = 0

        # Temporal smoothing — deque of raw per-frame gesture strings (or None)
        self._history: deque[Optional[str]] = deque(maxlen=GESTURE_HISTORY_LEN)

        log.info("GestureDetector ready — model: %s", os.path.basename(model_path))

    # ── Core inference ────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> GestureResult:
        """
        Run gesture recognition on one BGR webcam frame.

        Args:
            frame: uint8 BGR numpy array (H × W × 3) from OpenCV.

        Returns:
            GestureResult with the temporally-smoothed gesture (or None).
        """
        # Enforce monotonically increasing timestamp required by VIDEO mode
        ts_ms = int(time.time() * 1000)
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms

        # MediaPipe expects RGB
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )

        mp_result = self._recognizer.recognize_for_video(mp_image, ts_ms)

        # Extract the highest-confidence gesture across all detected hands.
        # We pick the first mapped gesture found (multi-hand priority: hand 0).
        raw_gesture: Optional[str] = None
        for hand_gestures in (mp_result.gestures or []):
            if hand_gestures:
                category = hand_gestures[0].category_name
                mapped   = _MP_CATEGORY_MAP.get(category)
                if mapped:
                    raw_gesture = mapped
                    break

        # Temporal smoothing: require GESTURE_HISTORY_VOTES agreeing frames
        self._history.append(raw_gesture)
        stable = self._stable_gesture()

        return GestureResult(
            gesture=stable,
            display_label=GESTURE_DISPLAY.get(stable, "—") if stable else "—",
            event_text=GESTURE_TO_EVENT.get(stable, "")    if stable else "",
            hand_count=len(mp_result.hand_landmarks or []),
            hand_landmarks_list=list(mp_result.hand_landmarks or []),
            timestamp=time.time(),
        )

    # ── Temporal smoothing ────────────────────────────────────────────────
    def _stable_gesture(self) -> Optional[str]:
        """
        Return the most common gesture in the recent history, only if it
        meets the minimum-votes threshold.

        A gesture that flickers for 1–2 frames (e.g. boundary between
        "Thumb_Up" and "Open_Palm" as fingers splay) is swallowed here
        before it reaches the event timeline.
        """
        votes = [g for g in self._history if g is not None]
        if not votes:
            return None
        most_common, count = Counter(votes).most_common(1)[0]
        return most_common if count >= GESTURE_HISTORY_VOTES else None

    # ── Drawing ───────────────────────────────────────────────────────────
    def draw(self, frame: np.ndarray, result: GestureResult) -> None:
        """
        Render hand landmarks and gesture HUD label onto frame in-place.

        Draw order:
          1. Landmark connections (lines)
          2. Landmark points (dots with white ring)
          3. Gesture label HUD (bottom-left corner)
        """
        if not result.hand_landmarks_list:
            return

        h, w = frame.shape[:2]

        for hand_lms in result.hand_landmarks_list:
            _draw_hand_landmarks(frame, hand_lms, w, h)

        if result.gesture:
            _draw_gesture_hud(frame, result, h)

    # ── Resource cleanup ─────────────────────────────────────────────────
    def release(self) -> None:
        """Close the MediaPipe GestureRecognizer and free GPU/CPU resources."""
        try:
            self._recognizer.close()
            log.info("GestureDetector released.")
        except Exception:
            pass

    # ── Static helpers ────────────────────────────────────────────────────
    @staticmethod
    def _download_model(dest_path: str) -> None:
        """
        Download gesture_recognizer.task from Google's MediaPipe CDN.

        File size ~25 MB.  Progress is logged every 10%.
        """
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        def _progress(block_num: int, block_size: int, total_size: int) -> None:
            if total_size > 0:
                pct = block_num * block_size / total_size * 100
                if int(pct) % 10 == 0:
                    log.info("  Downloading gesture model … %d%%", min(int(pct), 100))

        log.info("Downloading gesture model → %s", dest_path)
        urllib.request.urlretrieve(GESTURE_MODEL_URL, dest_path, _progress)
        log.info("Gesture model download complete.")


# ──────────────────────────────────────────────────────────────────────────────
# Drawing helpers (module-level, reused by both real and mock draw())
# ──────────────────────────────────────────────────────────────────────────────
def _draw_hand_landmarks(
    frame: np.ndarray,
    landmarks: list,
    img_w: int,
    img_h: int,
) -> None:
    """
    Draw hand landmark skeleton using pure OpenCV (no mp.solutions dependency).

    Uses HandLandmarksConnections from the Tasks API directly.

    Args:
        frame:     BGR frame to draw on (in-place).
        landmarks: List of 21 NormalizedLandmark objects (.x, .y in [0,1]).
        img_w:     Frame width in pixels.
        img_h:     Frame height in pixels.
    """
    # Convert normalised coords → pixel coordinates
    pts = [(int(lm.x * img_w), int(lm.y * img_h)) for lm in landmarks]

    # ① Connection lines
    for conn in _HAND_CONNECTIONS:
        p1, p2 = pts[conn.start], pts[conn.end]
        cv2.line(frame, p1, p2, _COLOR_CONNECTION, 2, cv2.LINE_AA)

    # ② Landmark dots: filled colour + white ring (enterprise camera aesthetic)
    for pt in pts:
        cv2.circle(frame, pt, 5, _COLOR_LANDMARK, cv2.FILLED, cv2.LINE_AA)
        cv2.circle(frame, pt, 5, _COLOR_RING,     1,          cv2.LINE_AA)


def _draw_gesture_hud(frame: np.ndarray, result: GestureResult, img_h: int) -> None:
    """
    Overlay the detected gesture name in the bottom-left corner.
    Large enough to be read from 2 metres away at a conference booth.
    """
    label = f"  {result.display_label}  "
    draw_overlay_text(
        frame,
        label,
        origin=(20, img_h - 28),
        font_scale=0.72,
        thickness=2,
        text_color=(0, 230, 118),    # InfoTraff spring-green accent
        bg_color=(5, 10, 20),
        bg_alpha=0.72,
        padding=6,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Mock Detector
# ──────────────────────────────────────────────────────────────────────────────
class MockGestureDetector(BaseGestureDetector):
    """
    Simulates GestureDetector without requiring the .task model file.

    Randomly emits gestures with realistic inter-event gaps so the full
    UI pipeline (event timeline, sidebar, flashing alerts) can be tested
    without a physical hand in front of the camera.
    """

    _GESTURES = list(GESTURE_DISPLAY.keys())  # ["thumbs_up", "open_palm", "peace_sign"]

    def __init__(self) -> None:
        log.warning("MockGestureDetector active — synthetic gestures only.")
        self._rng     = random.Random(99)
        self._counter = 0
        self._current: Optional[str] = None
        self._duration = 0

    def detect(self, frame: np.ndarray) -> GestureResult:
        self._counter += 1

        # Change gesture every ~60 frames (2 s at 30 fps) with 30% chance
        if self._duration <= 0:
            self._current  = self._rng.choice(self._GESTURES + [None, None])
            self._duration = self._rng.randint(40, 90)
        else:
            self._duration -= 1

        gesture = self._current
        return GestureResult(
            gesture=gesture,
            display_label=GESTURE_DISPLAY.get(gesture, "—") if gesture else "—",
            event_text=GESTURE_TO_EVENT.get(gesture, "")    if gesture else "",
            hand_count=1 if gesture else 0,
            hand_landmarks_list=[],   # no real landmarks; draw() is a no-op
            timestamp=time.time(),
        )

    def draw(self, frame: np.ndarray, result: GestureResult) -> None:
        """Draw only the HUD label — no real landmarks to render."""
        if result.gesture:
            _draw_gesture_hud(frame, result, frame.shape[0])

    def release(self) -> None:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────
def get_gesture_detector() -> BaseGestureDetector:
    """
    Return the best available gesture detector.

    1. Tries to load the real GestureDetector (downloads model if needed).
    2. Falls back to MockGestureDetector on any failure:
       - mediapipe not installed
       - model download failed (no internet at venue)
       - model file corrupt / version mismatch
    """
    try:
        return GestureDetector()
    except FileNotFoundError:
        log.warning("gesture_recognizer.task not found and download failed — mock mode.")
        return MockGestureDetector()
    except Exception as exc:
        log.error("GestureDetector init failed (%s) — mock mode.", exc)
        return MockGestureDetector()
