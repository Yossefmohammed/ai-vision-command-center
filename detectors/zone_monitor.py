"""
zone_monitor.py — Restricted Zone Monitoring for InfoTraff AI Vision Command Center.

Watches every "person" detection each frame and tests whether their foot-point
sits inside the configured restricted-zone polygon.  Implements edge-detection
(entry / exit) rather than re-firing every frame a person is inside — this is
what makes the alert feel like a real access-control system instead of a
spammy console print.

Architecture decisions:
  ① Polygon stored normalised in config, resized to the current frame's pixel
     dimensions every call (helpers.norm_to_pixel) — survives resolution changes,
     and a future "drag to redraw the zone" UI editor only needs to write back
     normalised coordinates.
  ② Edge-triggered alerting: ZoneResult.breach_started fires exactly once on
     entry, breach_ended fires exactly once on exit.  EventEngine logs on the
     entry edge only — a person standing in the zone for 10s produces ONE
     timeline entry + ONE snapshot, not five.  This mirrors how real
     access-control / VMS systems alert (entry event, not a heartbeat).
  ③ Blink phase is computed from wall-clock time + ZONE_ALERT_BLINK_HZ, not a
     frame counter — keeps the flash rate visually stable even if FPS drops
     under load (e.g. when 3 detectors are running on one CPU core).
  ④ Snapshot-on-entry is debounced by the same edge — never spams disk, and
     is fully optional via config.ZONE_AUTO_SNAPSHOT.
  ⑤ Multi-person aware: occupant_count is exposed separately from the binary
     is_breach flag, since the dashboard sidebar wants the headcount even
     when whole-zone-clear is the visual state.
"""
from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

from utils.config import (
    DEFAULT_ZONE_POLYGON,
    ZONE_COLOR_NORMAL,
    ZONE_COLOR_ALERT,
    ZONE_ALPHA,
    ZONE_ALERT_BLINK_HZ,
    ZONE_AUTO_SNAPSHOT,
)
from utils.helpers import norm_to_pixel, point_in_polygon, draw_zone_polygon, save_snapshot
from detectors.object_detector import DetectionFrame

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data Model
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class ZoneResult:
    """Per-frame output of ZoneMonitor.check()."""
    occupant_count: int                  # People currently inside the polygon
    is_breach: bool                      # True for every frame >=1 person is inside
    breach_started: bool                 # True ONLY on the entry edge (0 → 1+ occupants)
    breach_ended: bool                   # True ONLY on the exit edge (1+ → 0 occupants)
    blink_on: bool                       # Current flash phase, for HUD / colour
    snapshot_path: Optional[str] = None  # Set only on the frame a snapshot was written

    @property
    def display_color(self) -> Tuple[int, int, int]:
        """BGR colour for the current frame — flashes red only while breached."""
        return ZONE_COLOR_ALERT if (self.is_breach and self.blink_on) else ZONE_COLOR_NORMAL


# ──────────────────────────────────────────────────────────────────────────────
# ZoneMonitor
# ──────────────────────────────────────────────────────────────────────────────
class ZoneMonitor:
    """
    Tracks restricted-zone occupancy across frames and emits edge-triggered
    breach signals consumed by EventEngine.

    Usage:
        monitor = ZoneMonitor()
        ...
        result = monitor.check(detection_frame, frame.shape[1], frame.shape[0], frame)
        monitor.draw(frame, frame.shape[1], frame.shape[0], result)
    """

    def __init__(self, polygon_norm: Optional[List[Tuple[float, float]]] = None) -> None:
        self._polygon_norm = polygon_norm or DEFAULT_ZONE_POLYGON
        self._was_breached: bool = False
        self._blink_clock = time.perf_counter()

    # ── Core check ────────────────────────────────────────────────────────
    def check(
        self,
        detections: DetectionFrame,
        frame_w: int,
        frame_h: int,
        frame: Optional[np.ndarray] = None,
    ) -> ZoneResult:
        """
        Test every detected person against the zone polygon for this frame.

        Args:
            detections: This frame's DetectionFrame from ObjectDetector.
            frame_w, frame_h: Current frame dimensions (zone scales to these).
            frame: Optional raw BGR frame — only needed if ZONE_AUTO_SNAPSHOT
                   is True, so the snapshot captures the actual entry moment.

        Returns:
            ZoneResult with occupancy, edge flags, blink phase, and snapshot path.
        """
        polygon_px = norm_to_pixel(self._polygon_norm, frame_w, frame_h)

        occupants = sum(
            1 for person in detections.people
            if point_in_polygon(person.bottom_center, polygon_px)
        )

        is_breach = occupants > 0
        breach_started = is_breach and not self._was_breached
        breach_ended = (not is_breach) and self._was_breached
        self._was_breached = is_breach

        # Blink phase derived from wall-clock time, independent of frame rate.
        elapsed = time.perf_counter() - self._blink_clock
        blink_on = int(elapsed * ZONE_ALERT_BLINK_HZ * 2) % 2 == 0

        snapshot_path = None
        if breach_started and ZONE_AUTO_SNAPSHOT and frame is not None:
            snapshot_path = save_snapshot(frame, prefix="zone_breach")
            if snapshot_path:
                log.info("Zone breach snapshot saved: %s", snapshot_path)

        return ZoneResult(
            occupant_count=occupants,
            is_breach=is_breach,
            breach_started=breach_started,
            breach_ended=breach_ended,
            blink_on=blink_on,
            snapshot_path=snapshot_path,
        )

    # ── Drawing ───────────────────────────────────────────────────────────
    def draw(self, frame: np.ndarray, frame_w: int, frame_h: int, result: ZoneResult) -> None:
        """
        Render the zone polygon + flashing label onto frame in-place.

        Called after ObjectDetector.draw() in the render pipeline so the zone
        fill sits visually *behind* bounding boxes but its border/label is
        still readable on top of the dimmed background.
        """
        polygon_px = norm_to_pixel(self._polygon_norm, frame_w, frame_h)
        label = (
            "RESTRICTED AREA ENTRY"
            if (result.is_breach and result.blink_on)
            else "RESTRICTED ZONE"
        )
        draw_zone_polygon(
            frame, polygon_px,
            color=result.display_color,
            alpha=ZONE_ALPHA,
            label=label,
        )

    # ── Runtime reconfiguration ───────────────────────────────────────────
    def reconfigure(self, polygon_norm: List[Tuple[float, float]]) -> None:
        """
        Replace the monitored polygon at runtime.

        Hook for a future "drag to redraw zone" UI control — takes normalised
        coordinates so it stays resolution-independent like the config default.
        """
        self._polygon_norm = polygon_norm
        self._was_breached = False  # reset edge-state to avoid a false breach_ended
