"""
logger.py — Thread-safe event logging for AI Vision Command Center.

Two responsibilities:
  1. Maintain an in-memory ring-buffer of SmartEvents for the dashboard Timeline.
  2. Persist events to a rotating log file for post-demo analysis.

Threading model:
  - The camera/detection loop runs in a background QThread.
  - The dashboard Timeline widget lives in the Qt main thread.
  - EventLogger.log() is called from the background thread, then fires subscriber
    callbacks (which may dispatch Qt signals) — all protected by a mutex.
"""
import threading
import time
import logging
import os
from dataclasses import dataclass
from typing import Optional, List, Callable
from datetime import datetime
from collections import deque

from utils.config import (
    MAX_TIMELINE_EVENTS,
    EVENT_DEDUP_SECONDS,
    LOGS_DIR,
)


# ──────────────────────────────────────────────────────────────────────────────
# Data Model
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class SmartEvent:
    """A single business event produced by the EventEngine."""
    timestamp: float          # Unix epoch — time.time()
    message:   str            # Human-readable event text shown in the timeline
    category:  str = "info"   # "info" | "warning" | "gesture" | "zone"
    icon:      str = "●"      # Unicode glyph prefix for the timeline row

    @property
    def time_str(self) -> str:
        """HH:MM:SS formatted wall-clock time."""
        return datetime.fromtimestamp(self.timestamp).strftime("%H:%M:%S")

    # Category → colour mapping consumed by the Timeline widget
    CATEGORY_COLORS: dict = None  # filled after class definition (avoid mutable default)

    def __post_init__(self) -> None:
        if SmartEvent.CATEGORY_COLORS is None:
            SmartEvent.CATEGORY_COLORS = {
                "info":    "#00E5FF",   # cyan
                "warning": "#FF5252",   # red
                "gesture": "#00E676",   # green
                "zone":    "#FF5252",   # red  (same as warning — zone alerts are critical)
            }

    @property
    def color(self) -> str:
        """Return the hex colour string for this event's category."""
        return SmartEvent.CATEGORY_COLORS.get(self.category, "#8A9BB5")


# ──────────────────────────────────────────────────────────────────────────────
# EventLogger
# ──────────────────────────────────────────────────────────────────────────────
class EventLogger:
    """
    Thread-safe singleton event logger.

    Usage:
        logger = EventLogger.get_instance()
        logger.log("Employee entered monitored area", category="info", icon="👤")
        logger.subscribe(my_ui_callback)
    """

    _instance: Optional["EventLogger"] = None
    _class_lock = threading.Lock()

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # Bounded deque: O(1) appendleft + automatic eviction when maxlen reached.
        # Newest events at index 0 — matches the timeline's top-to-bottom display.
        self._events: deque[SmartEvent] = deque(maxlen=MAX_TIMELINE_EVENTS)

        # UI callbacks — called on every new event, from the logger thread.
        # Callbacks must be fast; they should only dispatch a Qt signal, not render.
        self._subscribers: List[Callable[[SmartEvent], None]] = []

        # De-duplication: maps message text → last time it was accepted
        self._last_seen: dict[str, float] = {}

        self._file_logger = self._setup_file_logger()

    # ── Singleton ─────────────────────────────────────────────────────────────
    @classmethod
    def get_instance(cls) -> "EventLogger":
        """Return the process-wide EventLogger (create on first call)."""
        with cls._class_lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    # ── Core API ──────────────────────────────────────────────────────────────
    def log(
        self,
        message: str,
        *,
        category: str = "info",
        icon: str = "●",
        force: bool = False,
    ) -> Optional[SmartEvent]:
        """
        Record a business event.

        Args:
            message:  Human-readable text (appears in the dashboard timeline).
            category: Visual category — controls timeline row colour.
            icon:     Unicode glyph shown before the message.
            force:    Bypass de-duplication (use for critical zone alerts).

        Returns:
            The new SmartEvent, or None if swallowed by de-duplication.
        """
        now = time.time()

        with self._lock:
            if not force:
                last_ts = self._last_seen.get(message, 0.0)
                if now - last_ts < EVENT_DEDUP_SECONDS:
                    return None            # duplicate — discard silently
            self._last_seen[message] = now

            event = SmartEvent(
                timestamp=now,
                message=message,
                category=category,
                icon=icon,
            )
            self._events.appendleft(event)
            # Snapshot subscribers while holding the lock to avoid TOCTOU
            subs = list(self._subscribers)

        # Notify outside the lock — callbacks may acquire their own locks
        # (e.g. Qt's internal signal machinery) and we must not risk deadlock.
        for cb in subs:
            try:
                cb(event)
            except Exception:
                pass  # never let a broken UI callback crash the detection loop

        if self._file_logger:
            self._file_logger.info("[%s] %s %s", category.upper(), icon, message)

        return event

    def get_recent(self, n: int = 50) -> List[SmartEvent]:
        """Return up to n most-recent events, newest first."""
        with self._lock:
            return list(self._events)[:n]

    def subscribe(self, callback: Callable[[SmartEvent], None]) -> None:
        """Register a callback to be invoked synchronously on every new event."""
        with self._lock:
            if callback not in self._subscribers:
                self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[SmartEvent], None]) -> None:
        with self._lock:
            self._subscribers = [s for s in self._subscribers if s is not callback]

    def clear(self) -> None:
        """Reset the event buffer (useful between demo sessions)."""
        with self._lock:
            self._events.clear()
            self._last_seen.clear()

    @property
    def event_count(self) -> int:
        with self._lock:
            return len(self._events)

    # ── Internal ──────────────────────────────────────────────────────────────
    @staticmethod
    def _setup_file_logger() -> Optional[logging.Logger]:
        """
        Create a session log file under logs/.
        Returns None if the directory cannot be written (graceful degradation).
        """
        try:
            os.makedirs(LOGS_DIR, exist_ok=True)
            session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = os.path.join(LOGS_DIR, f"session_{session_ts}.log")

            file_logger = logging.getLogger(f"aivc.{session_ts}")
            file_logger.setLevel(logging.INFO)
            file_logger.propagate = False   # don't pollute the root logger

            handler = logging.FileHandler(log_path, encoding="utf-8")
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S",
                )
            )
            file_logger.addHandler(handler)
            return file_logger

        except OSError:
            return None   # e.g. read-only filesystem — not fatal
