"""
config.py — Centralised configuration for AI Vision Command Center.

Design principle: every magic number lives here.
No other module should contain hard-coded thresholds, colours, or paths.
Changing a parameter once here propagates across the whole system.
"""
import os
from typing import Tuple, List


# ──────────────────────────────────────────────────────────────────────────────
# Application Identity
# ──────────────────────────────────────────────────────────────────────────────
APP_NAME: str     = "AI Vision Command Center"
APP_VERSION: str  = "1.0.0"
ORGANIZATION: str = "InfoTraff"
APP_TAGLINE: str  = "See Everything. Miss Nothing."


# ──────────────────────────────────────────────────────────────────────────────
# Camera
# ──────────────────────────────────────────────────────────────────────────────
CAMERA_INDEX: int  = 0       # Default webcam (0 = system primary)
CAMERA_WIDTH: int  = 1200    # Requested capture width  (px)
CAMERA_HEIGHT: int = 720     # Requested capture height (px)
TARGET_FPS: int    = 30      # Frame-rate target for the capture loop


# ──────────────────────────────────────────────────────────────────────────────
# YOLO — Object Detection
# ──────────────────────────────────────────────────────────────────────────────
YOLO_MODEL: str          = "yolov8n.pt"  # nano = fastest; swap to yolov8s.pt for accuracy
YOLO_CONF_THRESHOLD: float = 0.40        # Min confidence to display a detection
YOLO_IOU_THRESHOLD: float  = 0.45        # NMS IoU threshold
YOLO_IMGSZ: int            = 640         # Inference image size (must be ÷32)

# Only these COCO class names will be tracked — everything else is ignored.
TRACKED_CLASSES: List[str] = [
    "person",
    "cell phone",
    "laptop",
    "bottle",
    "cup",
    "backpack",
    "chair",
]

# BGR colours for bounding boxes, matched to the dark-theme palette
CLASS_COLORS: dict[str, Tuple[int, int, int]] = {
    "person":     (0, 230, 118),    # spring green
    "cell phone": (0, 188, 255),    # electric blue
    "laptop":     (255, 167, 38),   # amber
    "bottle":     (179, 100, 255),  # violet
    "cup":        (0, 213, 255),    # gold-cyan
    "backpack":   (255, 82,  82),   # soft red
    "chair":      (100, 220, 100),  # muted green
}
DEFAULT_BOX_COLOR: Tuple[int, int, int] = (200, 200, 200)


# ──────────────────────────────────────────────────────────────────────────────
# Restricted Zone – FULL FRAME for testing (any person triggers breach)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_ZONE_POLYGON: List[Tuple[float, float]] = [
    (0.0, 0.0),
    (1.0, 0.0),
    (1.0, 1.0),
    (0.0, 1.0),
]

ZONE_COLOR_NORMAL: Tuple[int, int, int] = (0, 229, 255)   # cyan
ZONE_COLOR_ALERT:  Tuple[int, int, int] = (0,  50, 255)   # vivid red
ZONE_ALPHA: float       = 0.18    # Polygon fill opacity (0 = transparent)
ZONE_ALERT_BLINK_HZ: float = 2.5  # Flash frequency when alert is active (Hz)
ZONE_AUTO_SNAPSHOT: bool = True   # Auto-save a JPEG on every zone-entry edge


# ──────────────────────────────────────────────────────────────────────────────
# Gesture Recognition
# ──────────────────────────────────────────────────────────────────────────────
GESTURE_MIN_DETECTION_CONFIDENCE: float = 0.70
GESTURE_MIN_TRACKING_CONFIDENCE: float  = 0.60

# MediaPipe Tasks GestureRecognizer model — auto-downloaded on first run if missing.
GESTURE_MODEL_URL: str = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/1/gesture_recognizer.task"
)
GESTURE_MODEL_PATH: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets", "models", "gesture_recognizer.task",
)

# Temporal smoothing — majority vote over the last N frames.
GESTURE_HISTORY_LEN: int   = 7
GESTURE_HISTORY_VOTES: int = 4


# ──────────────────────────────────────────────────────────────────────────────
# Smart Event Engine
# ──────────────────────────────────────────────────────────────────────────────
# Business‑language event mapping
DETECTION_TO_EVENT: dict[str, str] = {
    "person":     "Customer footfall detected",
    "cell phone": "Mobile device usage detected",
    "laptop":     "Workstation activity detected",
    "bottle":     "Product handling detected",
    "cup":        "Beverage / product item identified",
    "backpack":   "Customer with bag — loss‑prevention flag",
    "chair":      "Seating zone occupied",
}

# Queue detection – lowered threshold to 2 for easier demo demonstration.
PEOPLE_QUEUE_THRESHOLD: int = 2
QUEUE_ALERT_EVENT: str = "⚠ Queue threshold exceeded — staff action required"

GESTURE_TO_EVENT: dict[str, str] = {
    "thumbs_up":  "Positive interaction — Thumbs Up",
    "open_palm":  "Attention signal — Open Palm",
    "peace_sign": "Peace / Victory sign detected",
}

# Zone breach
ZONE_BREACH_EVENT: str = "⚠ Unauthorised zone entry — Access Control Alert"

MAX_TIMELINE_EVENTS: int  = 150    # Ring-buffer cap
EVENT_DEDUP_SECONDS: float = 1.0   # Short dedup for demo responsiveness


# ──────────────────────────────────────────────────────────────────────────────
# UI / Dashboard (only used in PyQt version – kept for compatibility)
# ──────────────────────────────────────────────────────────────────────────────
WINDOW_WIDTH:  int = 1600
WINDOW_HEIGHT: int = 900
SIDEBAR_WIDTH: int = 340

COLOR_BG_DARK:        str = "#0D0F14"
COLOR_BG_CARD:        str = "#161A23"
COLOR_BG_CARD_HOVER:  str = "#1E2330"
COLOR_ACCENT:         str = "#00E5FF"
COLOR_ACCENT_WARN:    str = "#FF5252"
COLOR_ACCENT_SUCCESS: str = "#00E676"
COLOR_TEXT_PRIMARY:   str = "#E8EAED"
COLOR_TEXT_SECONDARY: str = "#8A9BB5"
COLOR_BORDER:         str = "#2A3045"
COLOR_TIMELINE_A:     str = "#12151F"
COLOR_TIMELINE_B:     str = "#161A23"

FONT_FAMILY:      str = "Inter, Segoe UI, Arial"
FONT_SIZE_HEADER: int = 18
FONT_SIZE_BODY:   int = 13
FONT_SIZE_SMALL:  int = 11

STARTUP_STEPS: List[str] = [
    "Loading Object Detection Model",
    "Initialising Gesture Recognition",
    "Configuring Access-Control Zone Monitor",
    "Starting Smart Event Engine",
    "Connecting to Camera Feed",
    "System Ready — See Everything. Miss Nothing.",
]


# ──────────────────────────────────────────────────────────────────────────────
# Filesystem Paths – CORRECTED (utils/config.py → project root)
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # two levels up
SNAPSHOTS_DIR: str = os.path.join(BASE_DIR, "snapshots")
LOGS_DIR: str      = os.path.join(BASE_DIR, "logs")
ASSETS_DIR: str    = os.path.join(BASE_DIR, "assets")