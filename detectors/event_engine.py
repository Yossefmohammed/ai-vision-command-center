"""
event_engine.py — Smart Event Engine for InfoTraff AI Vision Command Center.

Converts raw ML outputs (DetectionFrame, GestureResult, ZoneResult) into
business-language events for the live timeline.  This is the layer that makes
the demo "feel intelligent" rather than printing class names — every mapping
lives in config.DETECTION_TO_EVENT / GESTURE_TO_EVENT / etc, never hard-coded
here, so relabelling for a different vertical (retail vs. manufacturing vs.
highway patrol) is a config edit, not a code change.

Design decisions:
  ① Stateless per call w.r.t. detections — process() reads this frame's three
     results and fires zero or more EventLogger.log() calls.  Internal state
     is limited to two tiny edge-detectors (queue, gesture) that EventLogger's
     own 2-second dedup window cannot express on its own.
  ② Object events are logged once per *class* per frame, not once per instance
     — three "person" boxes produce one "Customer footfall detected" line, not
     three.  This keeps the timeline (and any downstream analytics counting
     subscriber callbacks) honest, on top of EventLogger's own message-level
     dedup.
  ③ Queue alert is edge-triggered against PEOPLE_QUEUE_THRESHOLD, logged with
     force=True — a queue breach is operationally more important than a
     routine footfall event and must never be silently swallowed by the
     generic dedup window.
  ④ Zone breach is logged ONLY on ZoneResult.breach_started (the entry edge
     computed in ZoneMonitor), with force=True.  Mirrors real access-control
     systems: one alert per entry, not one alert every 2s someone loiters.
  ⑤ Gesture events log only when the *value itself changes* between calls —
     a held Thumbs-Up should not re-log every couple of seconds just because
     the generic dedup window expired.  GestureDetector's own temporal
     smoothing already guarantees `gesture.gesture` is stable, not flickering,
     so this is a clean value-change comparison, not a debounce hack.
  ⑥ The constructor accepts an OPTIONAL `logger` argument. By default it
     falls back to EventLogger.get_instance() — the original single-process
     desktop-app behaviour, unchanged. But the Streamlit deployment runs many
     independent visitor sessions inside ONE shared Python process, and
     EventLogger.get_instance() is a process-wide singleton — without this
     injection point, every visitor's events would land in the same global
     log and bleed into each other's timelines. Passing a fresh `EventLogger()`
     (bypassing get_instance()) gives each Streamlit session its own isolated
     event stream with zero changes to EventLogger itself.
"""
from __future__ import annotations

import logging
from typing import Optional

from utils.config import (
    DETECTION_TO_EVENT,
    PEOPLE_QUEUE_THRESHOLD,
    QUEUE_ALERT_EVENT,
    ZONE_BREACH_EVENT,
)
from utils.logger import EventLogger
from detectors.object_detector import DetectionFrame
from detectors.gesture_detector import GestureResult
from detectors.zone_monitor import ZoneResult

log = logging.getLogger(__name__)


class EventEngine:
    """
    Translates this frame's detector outputs into timeline events.

    Usage (called once per processed frame, from the detection thread,
    AFTER ObjectDetector / GestureDetector / ZoneMonitor have all run):

        engine = EventEngine()                      # desktop app: shared singleton
        engine = EventEngine(logger=EventLogger())   # Streamlit: isolated per-session
        ...
        engine.process(detection_frame, gesture_result, zone_result)
    """

    def __init__(self, logger: Optional[EventLogger] = None) -> None:
        self._logger = logger or EventLogger.get_instance()
        self._last_gesture: Optional[str] = None
        self._queue_breached: bool = False

    # ── Entry point ───────────────────────────────────────────────────────
    def process(
        self,
        detections: DetectionFrame,
        gesture: GestureResult,
        zone: ZoneResult,
    ) -> None:
        """Run all event-translation rules for one processed frame."""
        self._process_objects(detections)
        self._process_queue(detections)
        self._process_gesture(gesture)
        self._process_zone(zone)

    # ── Rule 1: Object → business event ──────────────────────────────────
    def _process_objects(self, detections: DetectionFrame) -> None:
        """One business event per distinct detected class this frame."""
        seen_classes: set[str] = set()
        for det in detections.detections:
            if det.class_name in seen_classes:
                continue
            seen_classes.add(det.class_name)
            message = DETECTION_TO_EVENT.get(det.class_name)
            if message:
                self._logger.log(message, category="info", icon="●")

    # ── Rule 2: Queue threshold (edge-triggered) ─────────────────────────
    def _process_queue(self, detections: DetectionFrame) -> None:
        """
        Fire QUEUE_ALERT_EVENT exactly once when people_count crosses the
        threshold from below to at-or-above it — InfoTraff's headline
        retail use case (queue / footfall congestion detection).
        """
        is_queue = detections.people_count >= PEOPLE_QUEUE_THRESHOLD
        if is_queue and not self._queue_breached:
            self._logger.log(QUEUE_ALERT_EVENT, category="warning", icon="⚠", force=True)
        self._queue_breached = is_queue

    # ── Rule 3: Gesture (value-change triggered) ─────────────────────────
    def _process_gesture(self, gesture: GestureResult) -> None:
        """Fire a gesture event only when the stable gesture value changes."""
        if gesture.gesture and gesture.gesture != self._last_gesture and gesture.event_text:
            self._logger.log(gesture.event_text, category="gesture", icon="✋", force=True)
        self._last_gesture = gesture.gesture

    # ── Rule 4: Zone breach (edge-triggered) ─────────────────────────────
    def _process_zone(self, zone: ZoneResult) -> None:
        """Fire ZONE_BREACH_EVENT exactly once per entry, never per frame."""
        if zone.breach_started:
            self._logger.log(ZONE_BREACH_EVENT, category="zone", icon="⚠", force=True)
