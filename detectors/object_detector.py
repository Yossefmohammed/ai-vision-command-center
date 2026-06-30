"""
object_detector.py — YOLOv8 Object Detection for InfoTraff AI Vision Command Center.

Architecture decisions:
  ① detect() and draw() are intentionally separate methods.
    Callers control rendering order (zone overlay → boxes → gestures → HUD).
  ② Output is a typed DetectionFrame dataclass — no raw tensors leak downstream.
    ZoneMonitor, EventEngine and the Dashboard only ever see DetectionFrame.
  ③ Class IDs are resolved from model.names at runtime, not hard-coded.
    Swapping yolov8n.pt → yolov8s.pt (or any custom model) works transparently.
  ④ warm_up() eliminates the "first-frame stutter" caused by CUDA JIT compilation.
    Must be called once during the startup loading screen.
  ⑤ MockObjectDetector provides a full stand-in when ultralytics is unavailable —
    used for UI-only testing and as a venue fallback if the model fails to load.
"""
from __future__ import annotations

import time
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np

from utils.config import (
    YOLO_MODEL,
    YOLO_CONF_THRESHOLD,
    YOLO_IOU_THRESHOLD,
    YOLO_IMGSZ,
    TRACKED_CLASSES,
    CLASS_COLORS,
    DEFAULT_BOX_COLOR,
)
from utils.helpers import draw_bounding_box

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data Model
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Detection:
    """
    A single detected object — immutable value object.

    frozen=True means every Detection is hash-able and safe to share across
    threads without copying, which matters since detect() runs in a background
    QThread and the results are read by the main (UI) thread.
    """
    class_name:  str
    confidence:  float
    x1: int
    y1: int
    x2: int
    y2: int
    color: Tuple[int, int, int]   # BGR, from config.CLASS_COLORS

    # ── Geometry helpers ──────────────────────────────────────────────────
    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2

    @property
    def bottom_center(self) -> Tuple[int, int]:
        """
        Foot-point of the bounding box.

        Preferred over centroid for zone-intrusion checks — a person's feet
        determine whether they are *standing inside* the zone, not their torso.
        Avoids false alerts when someone is outside but leans over the boundary.
        """
        return (self.x1 + self.x2) // 2, self.y2

    @property
    def area(self) -> int:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def __repr__(self) -> str:
        return (
            f"Detection({self.class_name!r} {self.confidence:.0%} "
            f"[{self.x1},{self.y1}→{self.x2},{self.y2}])"
        )


@dataclass
class DetectionFrame:
    """
    Structured output from one inference pass.

    Every consumer downstream (ZoneMonitor, EventEngine, Dashboard sidebar)
    uses only this dataclass — never raw YOLO result objects.
    This decouples the rest of the codebase from ultralytics entirely.
    """
    detections:   List[Detection]
    inference_ms: float     # Wall-clock time for model inference only
    timestamp:    float     # Unix epoch of this frame (time.time())

    # ── Computed views ────────────────────────────────────────────────────
    @property
    def people(self) -> List[Detection]:
        """All person detections (ZoneMonitor and EventEngine use this frequently)."""
        return [d for d in self.detections if d.class_name == "person"]

    @property
    def people_count(self) -> int:
        return sum(1 for d in self.detections if d.class_name == "person")

    def count(self, class_name: str) -> int:
        """Count detections for a specific class."""
        return sum(1 for d in self.detections if d.class_name == class_name)

    @property
    def object_counts(self) -> Dict[str, int]:
        """Dict of {class_name: count} for all detected objects."""
        counts: Dict[str, int] = {}
        for d in self.detections:
            counts[d.class_name] = counts.get(d.class_name, 0) + 1
        return counts

    @property
    def avg_confidence(self) -> float:
        """Mean confidence across all detections, or 0.0 if empty."""
        if not self.detections:
            return 0.0
        return sum(d.confidence for d in self.detections) / len(self.detections)

    @property
    def is_empty(self) -> bool:
        return len(self.detections) == 0

    @classmethod
    def empty(cls) -> "DetectionFrame":
        """Factory for a zero-detection frame (init state, camera lag, etc.)."""
        return cls(detections=[], inference_ms=0.0, timestamp=time.time())

    def __repr__(self) -> str:
        return (
            f"DetectionFrame(n={len(self.detections)}, "
            f"{self.inference_ms:.1f}ms, "
            f"people={self.people_count})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Abstract Base
# ──────────────────────────────────────────────────────────────────────────────
class BaseObjectDetector(ABC):
    """
    Interface contract for all object detectors.

    Both ObjectDetector (real YOLO) and MockObjectDetector implement this.
    The rest of the application depends only on this interface — never on a
    concrete class — which means swapping models requires zero changes elsewhere.
    """

    @abstractmethod
    def warm_up(self, h: int = 720, w: int = 1280) -> None:
        """Pre-load GPU kernels with a dummy inference."""
        ...

    @abstractmethod
    def detect(self, frame: np.ndarray) -> DetectionFrame:
        """Run inference on a single BGR frame. Returns DetectionFrame."""
        ...

    @abstractmethod
    def draw(self, frame: np.ndarray, result: DetectionFrame) -> None:
        """Render bounding boxes from result onto frame (in-place)."""
        ...

    @property
    @abstractmethod
    def device(self) -> str: ...

    @property
    @abstractmethod
    def device_label(self) -> str: ...

    @property
    @abstractmethod
    def model_name(self) -> str: ...


# ──────────────────────────────────────────────────────────────────────────────
# Real Detector (YOLOv8)
# ──────────────────────────────────────────────────────────────────────────────
class ObjectDetector(BaseObjectDetector):
    """
    YOLOv8 object detector wrapping the ultralytics API.

    Usage:
        detector = ObjectDetector()       # loads model, selects device
        detector.warm_up()               # call once during startup screen
        ...
        result = detector.detect(frame)  # call every frame
        detector.draw(frame, result)     # optional — renders bounding boxes
    """

    def __init__(self) -> None:
        try:
            from ultralytics import YOLO as _YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics not installed.\n"
                "Run:  pip install ultralytics"
            ) from exc

        self._device = self._select_device()
        log.info("ObjectDetector: loading %s on %s …", YOLO_MODEL, self._device)

        self._model = _YOLO(YOLO_MODEL)   # downloads yolov8n.pt on first run
        self._model.to(self._device)

        self._class_ids: Optional[List[int]] = self._resolve_class_ids()
        log.info(
            "ObjectDetector ready — %d classes, device=%s",
            len(self._class_ids or []), self._device,
        )

    # ── Device selection ──────────────────────────────────────────────────
    @staticmethod
    def _select_device() -> str:
        """
        Auto-select the fastest available compute device.
        Priority: CUDA GPU → Apple MPS → CPU
        """
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                log.info("CUDA GPU detected: %s", name)
                return "cuda"
            mps = getattr(torch.backends, "mps", None)
            if mps and mps.is_available():
                log.info("Apple Silicon MPS detected")
                return "mps"
        except ImportError:
            pass
        log.info("Falling back to CPU inference")
        return "cpu"

    # ── Class-ID resolution ───────────────────────────────────────────────
    def _resolve_class_ids(self) -> Optional[List[int]]:
        """
        Map TRACKED_CLASSES strings → integer COCO class IDs using model.names.

        By resolving from the model rather than a hard-coded mapping, we stay
        compatible with any YOLO model variant (YOLOv8n, v8s, custom fine-tunes).

        Returns None as a safe fallback (ultralytics interprets None as 'all classes').
        """
        name_to_id: Dict[str, int] = {
            v.lower(): k for k, v in self._model.names.items()
        }
        ids: List[int] = []
        for name in TRACKED_CLASSES:
            cid = name_to_id.get(name.lower())
            if cid is not None:
                ids.append(cid)
            else:
                log.warning("Class '%s' not found in model — skipping.", name)

        return ids if ids else None

    # ── Warm-up ───────────────────────────────────────────────────────────
    def warm_up(self, h: int = 720, w: int = 1280) -> None:
        """
        Eliminate first-frame stutter by pre-loading CUDA JIT-compiled kernels.

        On GPU, the first YOLO inference triggers kernel compilation which can
        take 3–5 seconds — completely unacceptable mid-demo at a conference.
        Running warm_up() during the startup screen hides this latency.
        """
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        self._model(
            dummy,
            conf=YOLO_CONF_THRESHOLD,
            iou=YOLO_IOU_THRESHOLD,
            imgsz=YOLO_IMGSZ,
            classes=self._class_ids,
            verbose=False,
        )
        log.info("ObjectDetector: warm-up inference complete.")

    # ── Inference ─────────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> DetectionFrame:
        """
        Run YOLOv8 on a single BGR frame captured from OpenCV.

        Args:
            frame: uint8 BGR numpy array, shape (H, W, 3).

        Returns:
            DetectionFrame containing typed Detection objects + timing metadata.
        """
        t0 = time.perf_counter()

        results = self._model(
            frame,
            conf=YOLO_CONF_THRESHOLD,
            iou=YOLO_IOU_THRESHOLD,
            imgsz=YOLO_IMGSZ,
            classes=self._class_ids,
            verbose=False,        # suppress ultralytics' per-frame console prints
        )

        inference_ms = (time.perf_counter() - t0) * 1000.0
        detections: List[Detection] = []

        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            for i in range(len(boxes)):
                cls_id     = int(boxes.cls[i].item())
                confidence = float(boxes.conf[i].item())
                xyxy       = boxes.xyxy[i].cpu().numpy().astype(int)
                class_name = self._model.names.get(cls_id, "unknown").lower()
                color      = CLASS_COLORS.get(class_name, DEFAULT_BOX_COLOR)

                detections.append(Detection(
                    class_name=class_name,
                    confidence=confidence,
                    x1=int(xyxy[0]), y1=int(xyxy[1]),
                    x2=int(xyxy[2]), y2=int(xyxy[3]),
                    color=color,
                ))

        return DetectionFrame(
            detections=detections,
            inference_ms=inference_ms,
            timestamp=time.time(),
        )

    # ── Drawing ───────────────────────────────────────────────────────────
    def draw(self, frame: np.ndarray, result: DetectionFrame) -> None:
        """
        Render bounding boxes for all detections onto frame in-place.

        Kept separate from detect() so the caller controls the draw order:
            1. draw_zone_polygon()         ← zone fill behind everything
            2. detector.draw()             ← boxes on top of zone
            3. gesture_detector.draw()     ← landmarks on top of boxes
            4. HUD overlays (FPS, alerts)  ← always topmost
        """
        for det in result.detections:
            draw_bounding_box(
                frame,
                det.x1, det.y1, det.x2, det.y2,
                det.class_name, det.confidence,
                color=det.color,
            )

    # ── Properties ────────────────────────────────────────────────────────
    @property
    def device(self) -> str:
        return self._device

    @property
    def device_label(self) -> str:
        """Human-readable device string for the dashboard status bar."""
        if self._device == "cuda":
            try:
                import torch
                return f"GPU · {torch.cuda.get_device_name(0)}"
            except Exception:
                return "GPU · CUDA"
        if self._device == "mps":
            return "GPU · Apple Silicon"
        return "CPU · Inference"

    @property
    def model_name(self) -> str:
        return YOLO_MODEL


# ──────────────────────────────────────────────────────────────────────────────
# Mock Detector (testing + venue fallback)
# ──────────────────────────────────────────────────────────────────────────────
class MockObjectDetector(BaseObjectDetector):
    """
    Simulates ObjectDetector without requiring ultralytics or a GPU.

    Two scenarios where this is used:
      ① Development / CI — test the full UI pipeline without the ML stack.
      ② Venue emergency — if yolov8n.pt fails to load at the conference,
           get_detector() falls back here so the demo still runs visually.

    Generates realistic but randomised detections to exercise the full pipeline.
    """

    _MOCK_CLASSES = TRACKED_CLASSES   # same class set as the real detector

    def __init__(self) -> None:
        log.warning(
            "MockObjectDetector active — generating synthetic detections. "
            "No real YOLO inference is running."
        )
        self._rng = random.Random(42)

    def warm_up(self, **_) -> None:  # type: ignore[override]
        pass   # nothing to pre-load

    def detect(self, frame: np.ndarray) -> DetectionFrame:
        h, w = frame.shape[:2]
        detections: List[Detection] = []

        # 0 to 3 random objects per frame
        for _ in range(self._rng.randint(0, 3)):
            cls   = self._rng.choice(self._MOCK_CLASSES)
            x1    = self._rng.randint(40, w // 2)
            y1    = self._rng.randint(40, h // 2)
            x2    = min(x1 + self._rng.randint(80, 220), w - 1)
            y2    = min(y1 + self._rng.randint(80, 220), h - 1)
            conf  = round(self._rng.uniform(0.45, 0.94), 2)
            color = CLASS_COLORS.get(cls, DEFAULT_BOX_COLOR)
            detections.append(Detection(
                class_name=cls, confidence=conf,
                x1=x1, y1=y1, x2=x2, y2=y2,
                color=color,
            ))

        return DetectionFrame(
            detections=detections,
            inference_ms=self._rng.uniform(4.5, 11.0),
            timestamp=time.time(),
        )

    def draw(self, frame: np.ndarray, result: DetectionFrame) -> None:
        for det in result.detections:
            draw_bounding_box(
                frame,
                det.x1, det.y1, det.x2, det.y2,
                det.class_name, det.confidence,
                color=det.color,
            )

    @property
    def device(self) -> str:
        return "cpu (mock)"

    @property
    def device_label(self) -> str:
        return "Mock · No GPU"

    @property
    def model_name(self) -> str:
        return "mock"


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────
def get_detector() -> BaseObjectDetector:
    """
    Factory function — returns the best available detector.

    Tries to create a real ObjectDetector first.
    Falls back to MockObjectDetector if ultralytics is missing or the model
    file cannot be loaded (e.g., no internet access at the venue).

    The rest of the application calls get_detector() and never instantiates
    ObjectDetector or MockObjectDetector directly — this isolates the fallback
    logic in one place.
    """
    try:
        return ObjectDetector()
    except ImportError:
        log.warning("ultralytics not available — using MockObjectDetector.")
        return MockObjectDetector()
    except Exception as exc:
        log.error("ObjectDetector init failed (%s) — using MockObjectDetector.", exc)
        return MockObjectDetector()
