# AI Vision Command Center
**InfoTraff — GITEX AI Conference Live Demo (Task 1)**
*"See Everything. Miss Nothing."*

A real-time, webcam-driven AI analytics dashboard built to emulate InfoTraff's CCTV intelligence platform — queue detection, restricted-zone / access-control alerts, and engagement signals — for a live booth demo.

---

## What it does

Point a webcam at the scene and the dashboard reacts live:

| Module | What it detects | Business framing shown on screen |
|---|---|---|
| **Object Detection** (YOLOv8n) | person, cell phone, laptop, bottle, cup, backpack, chair | "Customer footfall detected," "Mobile device usage detected," etc. |
| **Restricted Zone Monitoring** | Any person entering a configured polygon | Flashing "RESTRICTED AREA ENTRY" + one access-control alert per entry (not spammed every frame) |
| **Queue Detection** | People count crossing a threshold | "⚠ Queue threshold exceeded — staff action required" |
| **Hand Gesture Recognition** (MediaPipe) | 👍 Thumbs Up · ✋ Open Palm · ✌ Peace Sign | "Positive interaction," "Attention signal," etc. |
| **Smart Event Engine** | Converts every raw detection above into the business-language timeline entries shown live | — |

Every detection is translated from a raw class name into retail/F&B/access-control language (`utils/config.py → DETECTION_TO_EVENT`), so a passerby immediately maps what they're seeing to their own store, factory floor, or office — this is what makes it read as a real platform rather than an object-detector tech demo.

PPE/safety-gear detection was deliberately scoped out: a GITEX booth visitor won't be wearing a hard hat or vest, so that module would sit idle for every passerby instead of triggering live. Every feature that *is* included reacts the instant someone walks up, with zero props required — which matters more for a live "wow" demo than feature count.

---

## Running it

```bash
git clone <your-repo-url>
cd ai-vision-command-center
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open the local URL Streamlit prints, select **💻 Try my webcam** in the sidebar, and the live AI pipeline starts immediately — both the YOLOv8n and MediaPipe gesture model files are bundled in this repo (`yolov8n.pt`, `assets/models/gesture_recognizer.task`), so there's no first-run download wait.

### On the hosted link (Streamlit Community Cloud)

Cloud servers have no physical camera attached, so `cv2.VideoCapture(0)` cannot return a webcam feed there — this is a hardware constraint of any browser/cloud deployment, not specific to this app. The sidebar therefore offers a second option:

**🎬 Upload a video clip** — runs a short clip you upload through the *exact same* real-time detection pipeline (same YOLO model, same MediaPipe gesture model, same zone/queue/event logic), looping it continuously so the dashboard stays "live." This lets anyone reviewing the hosted link see the real detection pipeline working end-to-end without needing camera access — without removing or replacing the webcam path, which remains the default, first option in the sidebar and is what was used to record the local demo video.

---

## Project structure

```
ai-vision-command-center/
├── streamlit_app.py          # Streamlit UI: header, sidebar cards, video, timeline
├── requirements.txt
├── packages.txt               # apt-level deps Streamlit Cloud needs for OpenCV/MediaPipe
├── yolov8n.pt                 # bundled YOLOv8 nano weights (no first-run download)
├── assets/models/
│   └── gesture_recognizer.task   # bundled MediaPipe gesture model
├── utils/
│   ├── config.py              # every tunable value lives here (thresholds, colours, event text)
│   ├── helpers.py             # FPS counter, drawing utilities, geometry (point-in-polygon, etc.)
│   └── logger.py              # thread-safe EventLogger + SmartEvent
├── detectors/
│   ├── object_detector.py     # YOLOv8 wrapper — typed Detection/DetectionFrame, mock fallback
│   ├── gesture_detector.py    # MediaPipe GestureRecognizer wrapper — temporal smoothing, mock fallback
│   ├── zone_monitor.py        # restricted-zone polygon check, edge-triggered alerts
│   └── event_engine.py        # raw detections → business-language timeline events
├── logs/                      # session log files (created at runtime)
└── snapshots/                 # auto-saved zone-breach snapshots (created at runtime)
```

---

## Design notes (the "why," not just the "what")

**Mock fallback everywhere.** `get_detector()` and `get_gesture_detector()` both try the real model first and transparently fall back to a synthetic mock implementation if the model can't load — no internet at a venue, a corrupted download, missing dependency, etc. The demo keeps running and looking alive either way. This was tested directly: in a sandboxed environment with no network access to the model CDNs, both detectors cleanly fell back to mock mode with zero crashes.

**Edge-triggered alerts, not per-frame spam.** A person standing in the restricted zone for ten seconds produces *one* timeline entry and *one* snapshot — not one every frame. The same applies to the queue alert (fires once when the threshold is crossed, not continuously) and gestures (fires once when the gesture value changes, not every two seconds it's held). This is what makes the timeline read as a real alerting system rather than a console log.

**Per-session isolation.** The underlying `EventLogger` is a process-wide singleton by default. Streamlit Community Cloud, however, runs every visitor's session inside the *same* Python process — so `streamlit_app.py` gives each session its own private `EventLogger()` instance (injected into `EventEngine` via its `logger=` parameter) instead of using the shared singleton, so two reviewers opening the link at the same time never see each other's event timelines.

**Business-language event mapping lives entirely in config.** `DETECTION_TO_EVENT`, `GESTURE_TO_EVENT`, `QUEUE_ALERT_EVENT`, and `ZONE_BREACH_EVENT` in `utils/config.py` are the only place class names get turned into the retail/access-control phrasing shown on screen — relabelling this for a different vertical (manufacturing, highway patrol, etc.) is a config edit, not a code change.

---

## Known constraints

- **No camera on the hosted cloud link** — use the upload-clip mode there, or run locally for the webcam path (see *Running it* above).
- **Cold-start time on Streamlit Cloud.** The first session after a cold start has to install `torch`/`ultralytics`/`mediapipe`, which can take a few minutes. `requirements.txt` pins a CPU-only `torch` wheel (`--extra-index-url https://download.pytorch.org/whl/cpu`) specifically to keep this install small and avoid Streamlit Cloud's free-tier build timeouts — full CUDA wheels are far larger than needed since the cloud tier has no GPU anyway.
- **Demo-tuned thresholds.** `PEOPLE_QUEUE_THRESHOLD` is set low (2) and the restricted zone covers the full frame, so a solo reviewer can trigger every alert without needing a second person in frame. Both are one-line changes in `utils/config.py` for a production deployment.
- **PPE detection not implemented** — see the note in *What it does* above for the reasoning.

---

## Submission details

- **AI model(s) used:** Claude (Anthropic) — used throughout for architecture decisions, code generation, debugging, and review, covered in the linked chat session.
- **Approach:** Built as a modular pipeline (`utils/` for shared config and helpers, `detectors/` for four independent AI modules: object detection, gesture recognition, zone monitoring, and a business-language event engine). Every raw detection is translated into InfoTraff's retail/access-control language via a config-driven event mapping, with edge-triggered (not per-frame) alerting so the live timeline reads as a real alerting system rather than a console log. PPE detection was intentionally left out, since it requires safety gear a booth visitor won't have on hand — every included feature instead triggers live with zero props required. Both the YOLOv8n and MediaPipe gesture models are bundled directly in the repo to avoid any first-run download dependency at the venue or on a reviewer's machine.
- **Full chat session:** see the linked conversation transcript submitted alongside this repo.
