# streamlit_app.py — AI Vision Command Center (InfoTraff GITEX demo, Streamlit port)
#
# Why this file looks different from a "plain" Streamlit script:
#
#   ① Per-session event isolation. EventLogger.get_instance() is a process-wide
#      singleton (by design, for the original single-user PyQt6 desktop app).
#      Streamlit Cloud runs every visitor's session inside the SAME Python
#      process, so using the singleton here would bleed one visitor's timeline
#      into another's. Each session instead gets its OWN `EventLogger()`
#      (bypassing get_instance()), injected into EventEngine — see
#      detectors/event_engine.py's `logger=` parameter.
#
#   ② Two video sources, not one. Plain cv2.VideoCapture(0) only ever opens a
#      camera on the machine the Python process is running on. On Streamlit
#      Community Cloud that's the server — NOT the reviewer's laptop — so a
#      webcam-only build would show "no camera" forever once deployed. The
#      sidebar lets the visitor either try a local webcam (works when YOU run
#      this on your own machine) OR upload a short video clip, which is
#      processed by the real detection pipeline frame by frame and looped
#      continuously — so the hosted link is a genuine working demo for
#      anyone, anywhere, with no camera access required.
#
#   ③ The header now surfaces current time / FPS / camera status, matching
#      the original brief's dashboard spec (these were missing before).

import time
import tempfile
import os

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from utils.config import (
    CAMERA_WIDTH, CAMERA_HEIGHT, PEOPLE_QUEUE_THRESHOLD,
    APP_NAME, APP_TAGLINE, ORGANIZATION,
    COLOR_BG_DARK, COLOR_BG_CARD, COLOR_BG_CARD_HOVER, COLOR_ACCENT,
    COLOR_ACCENT_WARN, COLOR_ACCENT_SUCCESS, COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY, COLOR_BORDER,
)
from utils.helpers import FPSCounter
from utils.logger import EventLogger
from detectors.object_detector import get_detector
from detectors.gesture_detector import get_gesture_detector
from detectors.zone_monitor import ZoneMonitor
from detectors.event_engine import EventEngine

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_NAME,
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Theme — dark enterprise CCTV look, reusing the same colour tokens as the
# PyQt6 desktop build's ui/theme.py, so both products share one visual identity.
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
    .stApp {{
        background-color: {COLOR_BG_DARK};
    }}
    [data-testid="stSidebar"] {{
        background-color: {COLOR_BG_CARD};
        border-right: 1px solid {COLOR_BORDER};
    }}
    #MainMenu, footer, header {{ visibility: hidden; }}

    .avc-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        background-color: {COLOR_BG_CARD};
        border: 1px solid {COLOR_BORDER};
        border-radius: 14px;
        padding: 16px 24px;
        margin-bottom: 18px;
    }}
    .avc-title {{
        font-size: 22px;
        font-weight: 700;
        color: {COLOR_TEXT_PRIMARY};
        letter-spacing: 0.3px;
        margin: 0;
    }}
    .avc-subtitle {{
        font-size: 12.5px;
        color: {COLOR_ACCENT};
        margin: 0;
    }}
    .avc-stat {{
        font-size: 13px;
        color: {COLOR_TEXT_SECONDARY};
        padding: 0 14px;
        border-right: 1px solid {COLOR_BORDER};
    }}
    .avc-stat:last-child {{ border-right: none; }}
    .avc-stat b {{ color: {COLOR_TEXT_PRIMARY}; }}
    .avc-badge-live {{
        background-color: rgba(0, 230, 118, 0.15);
        color: {COLOR_ACCENT_SUCCESS};
        border: 1px solid {COLOR_ACCENT_SUCCESS};
        border-radius: 9px;
        padding: 4px 14px;
        font-weight: 700;
        font-size: 12px;
        margin-left: 14px;
        animation: avc-pulse-green 1.6s infinite;
    }}
    .avc-badge-offline {{
        background-color: rgba(255, 82, 82, 0.15);
        color: {COLOR_ACCENT_WARN};
        border: 1px solid {COLOR_ACCENT_WARN};
        border-radius: 9px;
        padding: 4px 14px;
        font-weight: 700;
        font-size: 12px;
        margin-left: 14px;
    }}
    @keyframes avc-pulse-green {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.55; }}
    }}

    .avc-card {{
        background-color: {COLOR_BG_CARD};
        border: 1px solid {COLOR_BORDER};
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 10px;
    }}
    .avc-card-label {{
        font-size: 10.5px;
        color: {COLOR_TEXT_SECONDARY};
        text-transform: uppercase;
        letter-spacing: 0.4px;
        font-weight: 600;
        margin: 0 0 2px 0;
    }}
    .avc-card-value {{
        font-size: 22px;
        font-weight: 700;
        color: {COLOR_TEXT_PRIMARY};
        margin: 0;
    }}
    .avc-card-value-accent {{ color: {COLOR_ACCENT}; font-size: 17px; }}
    .avc-card-value-alert {{ color: {COLOR_ACCENT_WARN}; }}
    .avc-card-alert-active {{
        border-color: {COLOR_ACCENT_WARN} !important;
        animation: avc-pulse-border 1.2s infinite;
    }}
    @keyframes avc-pulse-border {{
        0%, 100% {{ box-shadow: 0 0 0 0 rgba(255,82,82,0.4); }}
        50% {{ box-shadow: 0 0 0 6px rgba(255,82,82,0); }}
    }}

    .avc-timeline {{
        background-color: {COLOR_BG_CARD};
        border: 1px solid {COLOR_BORDER};
        border-radius: 12px;
        padding: 12px 16px;
        max-height: 230px;
        overflow-y: auto;
    }}
    .avc-timeline-row {{
        font-size: 13px;
        color: {COLOR_TEXT_PRIMARY};
        padding: 4px 0;
        border-bottom: 1px solid {COLOR_BG_CARD_HOVER};
    }}
    .avc-timeline-time {{ color: {COLOR_TEXT_SECONDARY}; margin-right: 8px; }}

    div[data-testid="stVideo"], .avc-video-frame img {{
        border-radius: 12px;
        border: 1px solid {COLOR_BORDER};
    }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session state — every detector + a PRIVATE EventLogger per visitor session
# ─────────────────────────────────────────────────────────────────────────────
if "initialised" not in st.session_state:
    st.session_state.event_logger     = EventLogger()          # NOT get_instance() — isolated per session
    st.session_state.object_detector  = get_detector()
    st.session_state.gesture_detector = get_gesture_detector()
    st.session_state.zone_monitor     = ZoneMonitor()
    st.session_state.event_engine     = EventEngine(logger=st.session_state.event_logger)
    st.session_state.fps_counter      = FPSCounter()
    st.session_state.session_start    = time.time()
    st.session_state.initialised      = True

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — branding + video source picker (read ONCE, before the live loop
# starts — Streamlit can't process new widget input while the loop below is
# running, so all source selection happens up-front)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"### 🎯 {APP_NAME}")
    st.caption(f"{ORGANIZATION} · {APP_TAGLINE}")
    st.divider()

    st.markdown("**Video Source**")
    source_mode = st.radio(
        "source_mode",
        ["💻 Try my webcam", "🎬 Upload a video clip"],
        label_visibility="collapsed",
        help="Webcam only works when you run this app on your own machine — "
             "a cloud server has no physical camera. Upload a clip to get a "
             "fully working live demo through this hosted link.",
    )
    uploaded_file = None
    if source_mode == "🎬 Upload a video clip":
        uploaded_file = st.file_uploader("Upload a short clip", type=["mp4", "mov", "avi", "mkv"])

    st.divider()
    st.markdown("#### 📊 Live Analytics")
    people_card      = st.empty()
    alerts_card      = st.empty()
    gesture_card     = st.empty()
    objects_card     = st.empty()
    phones_card      = st.empty()
    cups_card        = st.empty()
    bottles_card     = st.empty()
    confidence_card  = st.empty()

# ─────────────────────────────────────────────────────────────────────────────
# Header — title, tagline, current time, FPS, camera status (LIVE/OFFLINE)
# ─────────────────────────────────────────────────────────────────────────────
header_placeholder = st.empty()

def render_header(fps: float, device_label: str, is_live: bool) -> None:
    badge_cls = "avc-badge-live" if is_live else "avc-badge-offline"
    badge_txt = "● LIVE" if is_live else "● OFFLINE"
    header_placeholder.markdown(f"""
    <div class="avc-header">
        <div>
            <p class="avc-title">🎯 {APP_NAME}</p>
            <p class="avc-subtitle">{ORGANIZATION} — {APP_TAGLINE}</p>
        </div>
        <div style="display:flex; align-items:center;">
            <span class="avc-stat">⚡ <b>{device_label}</b></span>
            <span class="avc-stat">FPS <b>{fps:.1f}</b></span>
            <span class="avc-stat">{time.strftime('%H:%M:%S')}</span>
            <span class="{badge_cls}">{badge_txt}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

render_header(0.0, "Initialising…", False)

# ─────────────────────────────────────────────────────────────────────────────
# Main layout — video (left) + event timeline (full width, below)
# ─────────────────────────────────────────────────────────────────────────────
col_video, col_spacer = st.columns([3, 1])
video_placeholder = col_video.empty()

st.markdown("#### 📋 Live Event Timeline")
timeline_placeholder = st.empty()


# ─────────────────────────────────────────────────────────────────────────────
# Video source — webcam (local only) or uploaded clip (works everywhere,
# loops continuously so the "live feed" never just stops)
# ─────────────────────────────────────────────────────────────────────────────
class WebcamSource:
    """Local webcam, with the same animated 'no camera' fallback as the desktop app."""

    def __init__(self) -> None:
        self.cap = cv2.VideoCapture(0)
        self.is_live = self.cap.isOpened()
        if self.is_live:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.t0 = time.time()

    def read(self) -> np.ndarray:
        if self.is_live:
            ok, frame = self.cap.read()
            if ok:
                return frame
            self.is_live = False
        return self._synthetic_frame()

    def _synthetic_frame(self) -> np.ndarray:
        t = time.time() - self.t0
        shift = int((np.sin(t * 0.4) * 0.5 + 0.5) * 30)
        frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        frame[:, :, 0] = 14 + shift // 2
        frame[:, :, 1] = 10
        frame[:, :, 2] = 12
        cv2.putText(frame, "NO CAMERA DETECTED ON SERVER", (CAMERA_WIDTH // 2 - 260, CAMERA_HEIGHT // 2 - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (130, 150, 180), 2)
        cv2.putText(frame, "Use 'Upload a video clip' in the sidebar instead", (CAMERA_WIDTH // 2 - 280, CAMERA_HEIGHT // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 110, 140), 1)
        return frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()


class UploadedClipSource:
    """
    Plays an uploaded video file through the real detection pipeline,
    looping back to frame 0 at end-of-clip — keeps the "live" feel going
    indefinitely instead of freezing on the last frame.
    """

    def __init__(self, file_bytes: bytes, original_name: str) -> None:
        suffix = os.path.splitext(original_name)[1] or ".mp4"
        self._tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        self._tmp.write(file_bytes)
        self._tmp.flush()
        self.cap = cv2.VideoCapture(self._tmp.name)
        self.is_live = True  # "live" in the sense that the pipeline is actively running

    def read(self) -> np.ndarray:
        ok, frame = self.cap.read()
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        if not ok:
            # Corrupt/unreadable file — fall back to a dark placeholder rather than crash.
            frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        return frame

    def release(self) -> None:
        self.cap.release()
        try:
            os.unlink(self._tmp.name)
        except OSError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Frame processing — identical pipeline used by the PyQt6 desktop build
# ─────────────────────────────────────────────────────────────────────────────
def process_frame(frame: np.ndarray) -> dict:
    h, w = frame.shape[:2]
    detector          = st.session_state.object_detector
    gesture_detector  = st.session_state.gesture_detector
    zone_monitor      = st.session_state.zone_monitor
    event_engine      = st.session_state.event_engine

    detection_frame = detector.detect(frame)
    zone_result     = zone_monitor.check(detection_frame, w, h, frame)
    gesture_result  = gesture_detector.detect(frame)
    event_engine.process(detection_frame, gesture_result, zone_result)

    zone_monitor.draw(frame, w, h, zone_result)
    detector.draw(frame, detection_frame)
    gesture_detector.draw(frame, gesture_result)

    fps = st.session_state.fps_counter.tick()
    active_alerts = int(zone_result.is_breach) + int(detection_frame.people_count >= PEOPLE_QUEUE_THRESHOLD)

    return {
        "people": detection_frame.people_count,
        "alerts": active_alerts,
        "gesture": gesture_result.display_label if gesture_result.is_detected else None,
        "objects": detection_frame.object_counts,
        "confidence": detection_frame.avg_confidence,
        "device": detector.device_label,
        "fps": fps,
    }


def render_sidebar_cards(stats: dict) -> None:
    objects = stats["objects"]
    total_objects = sum(objects.values())

    people_card.markdown(
        f'<div class="avc-card"><p class="avc-card-label">People Count</p>'
        f'<p class="avc-card-value">{stats["people"]}</p></div>', unsafe_allow_html=True)

    alert_cls = "avc-card avc-card-alert-active" if stats["alerts"] > 0 else "avc-card"
    val_cls = "avc-card-value avc-card-value-alert" if stats["alerts"] > 0 else "avc-card-value"
    alerts_card.markdown(
        f'<div class="{alert_cls}"><p class="avc-card-label">⚠ Active Alerts</p>'
        f'<p class="{val_cls}">{stats["alerts"]}</p></div>', unsafe_allow_html=True)

    gesture_card.markdown(
        f'<div class="avc-card"><p class="avc-card-label">✋ Current Gesture</p>'
        f'<p class="avc-card-value avc-card-value-accent">{stats["gesture"] or "—"}</p></div>',
        unsafe_allow_html=True)

    objects_card.markdown(
        f'<div class="avc-card"><p class="avc-card-label">Objects Detected</p>'
        f'<p class="avc-card-value">{total_objects}</p></div>', unsafe_allow_html=True)

    phones_card.markdown(
        f'<div class="avc-card"><p class="avc-card-label">📱 Phones</p>'
        f'<p class="avc-card-value">{objects.get("cell phone", 0)}</p></div>', unsafe_allow_html=True)

    cups_card.markdown(
        f'<div class="avc-card"><p class="avc-card-label">☕ Cups</p>'
        f'<p class="avc-card-value">{objects.get("cup", 0)}</p></div>', unsafe_allow_html=True)

    bottles_card.markdown(
        f'<div class="avc-card"><p class="avc-card-label">🍶 Bottles</p>'
        f'<p class="avc-card-value">{objects.get("bottle", 0)}</p></div>', unsafe_allow_html=True)

    confidence_card.markdown(
        f'<div class="avc-card"><p class="avc-card-label">🎯 Avg Confidence</p>'
        f'<p class="avc-card-value">{stats["confidence"]*100:.0f}%</p></div>', unsafe_allow_html=True)


def render_timeline() -> None:
    events = st.session_state.event_logger.get_recent(20)
    if not events:
        timeline_placeholder.markdown(
            '<div class="avc-timeline" style="color:#5b6478;">No events yet — '
            'detections will appear here in real time.</div>', unsafe_allow_html=True)
        return
    rows = []
    for e in events:
        rows.append(
            f'<div class="avc-timeline-row">'
            f'<span class="avc-timeline-time">{e.time_str}</span>'
            f'<span style="color:{e.color};">{e.icon}</span> {e.message}</div>'
        )
    timeline_placeholder.markdown(
        f'<div class="avc-timeline">{"".join(rows)}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
video_source = None
try:
    if source_mode == "💻 Try my webcam":
        video_source = WebcamSource()
    elif uploaded_file is not None:
        video_source = UploadedClipSource(uploaded_file.getvalue(), uploaded_file.name)
    else:
        video_placeholder.info(
            "⬆️ Upload a short video clip in the sidebar to start the live AI demo — "
            "no camera needed. Or switch to **Try my webcam** if you're running this "
            "app on your own machine."
        )
        render_header(0.0, "Idle", False)
        st.stop()

    while True:
        frame = video_source.read()
        stats = process_frame(frame)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(Image.fromarray(frame_rgb), channels="RGB", width="stretch")

        render_header(stats["fps"], stats["device"], video_source.is_live)
        render_sidebar_cards(stats)
        render_timeline()

        time.sleep(0.05)  # ~20 fps target

except KeyboardInterrupt:
    pass
finally:
    if video_source is not None:
        video_source.release()
    st.session_state.gesture_detector.release()
