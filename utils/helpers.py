"""
helpers.py — Shared utility functions for AI Vision Command Center.

Modules:
  FPSCounter          — Exponential Moving Average FPS tracker
  draw_overlay_text   — Semi-transparent label on an OpenCV frame
  draw_bounding_box   — Corner-accent bounding box with confidence badge
  save_snapshot       — Timestamped JPEG snapshot to disk
  norm_to_pixel       — Normalised → pixel coordinate conversion
  point_in_polygon    — Ray-casting point-in-polygon test
  get_bbox_*          — Bounding-box geometry helpers
"""
import time
import os
import cv2
import numpy as np
from datetime import datetime
from typing import Tuple, List, Optional

from utils.config import SNAPSHOTS_DIR


# ──────────────────────────────────────────────────────────────────────────────
# FPS Counter
# ──────────────────────────────────────────────────────────────────────────────
class FPSCounter:
    """
    Exponential Moving Average FPS tracker.

    Why EMA instead of a sliding window?
      - O(1) memory — no history list
      - Smooth: alpha=0.08 gives ~12-frame averaging, which looks stable on a
        dashboard without lagging too far behind transient slowdowns
      - Dead-simple to implement correctly
    """

    def __init__(self, alpha: float = 0.08) -> None:
        """
        Args:
            alpha: EMA smoothing factor (0 < alpha ≤ 1).
                   Lower = smoother but slower to react.
        """
        self._alpha = alpha
        self._fps: float = 0.0
        self._last_tick: float = time.perf_counter()

    def tick(self) -> float:
        """
        Register one processed frame.

        Call once per frame in the capture loop.
        Returns the smoothed FPS estimate.
        """
        now = time.perf_counter()
        dt = now - self._last_tick
        self._last_tick = now

        if dt > 0:
            instant = 1.0 / dt
            # Cold start: accept the first sample directly
            self._fps = (
                instant if self._fps == 0.0
                else self._alpha * instant + (1.0 - self._alpha) * self._fps
            )

        return self._fps

    @property
    def fps(self) -> float:
        """Current smoothed FPS (read-only)."""
        return self._fps

    @property
    def fps_str(self) -> str:
        """Formatted string, e.g. '28.4'."""
        return f"{self._fps:.1f}"


# ──────────────────────────────────────────────────────────────────────────────
# Frame Drawing Utilities (OpenCV / BGR)
# ──────────────────────────────────────────────────────────────────────────────
def draw_overlay_text(
    frame: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    *,
    font_scale: float = 0.52,
    thickness: int = 1,
    text_color: Tuple[int, int, int] = (230, 230, 230),
    bg_color: Tuple[int, int, int] = (10, 12, 20),
    bg_alpha: float = 0.60,
    padding: int = 5,
) -> None:
    """
    Render a text label with a semi-transparent dark pill on an OpenCV frame.

    Modifies `frame` in-place.

    Args:
        frame:      BGR frame to draw on.
        text:       Label string.
        origin:     (x, y) of the text baseline bottom-left.
        font_scale: OpenCV font scale.
        thickness:  Text stroke thickness.
        text_color: BGR text colour.
        bg_color:   BGR background rectangle colour.
        bg_alpha:   Background opacity (0 = fully transparent, 1 = opaque).
        padding:    Pixels of padding around the text.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    x, y = origin
    x1, y1 = x - padding, y - th - padding
    x2, y2 = x + tw + padding, y + baseline + padding

    # Clamp to frame bounds
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, cv2.FILLED)
    cv2.addWeighted(overlay, bg_alpha, frame, 1.0 - bg_alpha, 0, frame)
    cv2.putText(
        frame, text, (x, y),
        font, font_scale, text_color, thickness, cv2.LINE_AA,
    )


def draw_bounding_box(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    label: str,
    confidence: float,
    color: Tuple[int, int, int] = (0, 230, 118),
    thickness: int = 2,
) -> None:
    """
    Draw an enterprise-style bounding box with corner accents and a confidence badge.

    Corner-tick style (not a plain rectangle) is the visual language of
    professional CCTV software — it looks intentional rather than academic.

    Args:
        frame:      BGR frame to draw on (in-place).
        x1, y1:    Top-left corner.
        x2, y2:    Bottom-right corner.
        label:      Class name.
        confidence: Detection confidence [0, 1].
        color:      BGR corner / badge colour.
        thickness:  Corner line thickness.
    """
    # Corner tick length: proportional to box size, but bounded
    tick = max(14, min(28, (x2 - x1) // 6))

    corners = [
        (x1, y1, +1, +1),   # top-left
        (x2, y1, -1, +1),   # top-right
        (x1, y2, +1, -1),   # bottom-left
        (x2, y2, -1, -1),   # bottom-right
    ]
    for (cx, cy, dx, dy) in corners:
        cv2.line(frame, (cx, cy), (cx + dx * tick, cy),          color, thickness, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx,              cy + dy * tick), color, thickness, cv2.LINE_AA)

    # Thin full-border overlay (1 px) for readability at distance
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

    # Confidence badge (top-left of box, above the border)
    badge = f"{label}  {confidence:.0%}"
    draw_overlay_text(
        frame, badge,
        origin=(x1, max(y1 - 6, 20)),
        text_color=color,
        bg_alpha=0.65,
    )


def draw_zone_polygon(
    frame: np.ndarray,
    polygon: List[Tuple[int, int]],
    color: Tuple[int, int, int],
    alpha: float = 0.18,
    label: Optional[str] = "RESTRICTED ZONE",
) -> None:
    """
    Draw a semi-transparent filled polygon for the restricted zone overlay.

    Args:
        frame:   BGR frame (in-place).
        polygon: List of (x, y) pixel vertices.
        color:   BGR fill and border colour.
        alpha:   Fill opacity.
        label:   Optional label drawn at the polygon centroid.
    """
    pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))

    # Transparent fill
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

    # Border
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

    # Zone label at centroid
    if label and len(polygon) >= 3:
        cx = sum(p[0] for p in polygon) // len(polygon)
        cy = sum(p[1] for p in polygon) // len(polygon)
        draw_overlay_text(frame, label, (cx - 60, cy), text_color=color, bg_alpha=0.70)


# ──────────────────────────────────────────────────────────────────────────────
# Snapshot
# ──────────────────────────────────────────────────────────────────────────────
def save_snapshot(frame: np.ndarray, prefix: str = "event") -> Optional[str]:
    """
    Write a JPEG snapshot to the snapshots directory.

    Args:
        frame:  BGR OpenCV frame.
        prefix: Filename prefix (e.g. 'zone_breach').

    Returns:
        Absolute path to the saved file, or None on failure.
    """
    try:
        os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:21]
        path = os.path.join(SNAPSHOTS_DIR, f"{prefix}_{ts}.jpg")
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return path
    except OSError:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Geometry
# ──────────────────────────────────────────────────────────────────────────────
def norm_to_pixel(
    norm_pts: List[Tuple[float, float]],
    width: int,
    height: int,
) -> List[Tuple[int, int]]:
    """
    Convert normalised (0–1) polygon coordinates to pixel coordinates.

    Storing zone vertices as normalised values decouples the polygon definition
    from any specific camera resolution — the zone "scales" automatically when
    the resolution changes.

    Args:
        norm_pts: List of (x_norm, y_norm) tuples.
        width:    Frame width in pixels.
        height:   Frame height in pixels.

    Returns:
        List of (x_px, y_px) integer tuples.
    """
    return [(int(x * width), int(y * height)) for x, y in norm_pts]


def point_in_polygon(
    point: Tuple[int, int],
    polygon: List[Tuple[int, int]],
) -> bool:
    """
    Ray-casting point-in-polygon test (Pnpoly algorithm).

    O(n) where n = number of polygon vertices.  Fast enough for every frame
    when n < 20 (typical zone polygons have 4–8 vertices).

    Args:
        point:   (x, y) pixel coordinate to test.
        polygon: List of (x, y) vertex tuples (at least 3).

    Returns:
        True if point is inside the polygon.
    """
    px, py = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > py) != (yj > py):
            if px < (xj - xi) * (py - yi) / (yj - yi) + xi:
                inside = not inside
        j = i
    return inside


def get_bbox_center(x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int]:
    """Return the centroid pixel of a bounding box."""
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def get_bbox_bottom_center(x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int]:
    """
    Return the bottom-centre of a bounding box.

    Preferred over centroid for zone intrusion checks — a person's feet
    determine whether they are *inside* the zone, not their torso midpoint.
    This avoids false alerts when someone is standing just outside the zone
    but their upper body overlaps the polygon edges.
    """
    return ((x1 + x2) // 2, y2)


def clamp(value: float, lo: float, hi: float) -> float:
    """Return value clamped to [lo, hi]."""
    return max(lo, min(hi, value))
