# Local Object Monitor

Turn a university computer-vision project into a useful **local video analytics tool**.
Track objects across frames, define zones, measure occupancy, and export timestamped
entry, exit and dwell events. All inference runs on your own machine.

## Install

Python 3.10+ is required. A virtual environment is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

On Windows, activate with `.venv\Scripts\activate` instead. PyTorch accelerator
support depends on your hardware and installation. CPU inference works but may be
slower. Ultralytics downloads weights on the first run if they are not present;
subsequent runs can be local/offline with cached weights.

## Quick examples

### Measure traffic through a zone in an existing video

```bash
rto-monitor --source data_sets/video_data/people-detection.mp4 \
  --zone entrance:0.15,0.2,0.85,0.9 \
  --classes person --dwell 3 --save-video
```

### Monitor a webcam

```bash
rto-monitor --source 0 --classes person --preview \
  --zone left:0,0,0.5,1 --zone right:0.5,0,1,1
```

Press **q** or **Esc** to stop the preview. No raw or annotated video is recorded
unless `--save-video` is specified. Events and aggregate occupancy are always
saved to a new timestamped folder under `runs/monitor/`.

### Monitor an RTSP camera

```bash
rto-monitor --source 'rtsp://camera.example/stream' \
  --zone workbench:0.25,0.35,0.75,0.95 \
  --classes person --dwell 10 --max-frames 3000
```

Credentials in RTSP/HTTP URLs are removed from the saved summary. **Do not** place
passwords directly in shell commands if your shell history or process list is
accessible to other users. The camera feed must be reachable by OpenCV.

### Track vehicles in a parking area

```bash
rto-monitor --source parking.mp4 --classes car,truck,motorcycle \
  --zone bay:0.30,0.30,0.85,0.95 --tracker botsort.yaml
```

Supported class names are determined by the chosen model. Use `--classes all` to
monitor every class, or specify names separated by commas.

## Zone coordinates

Use `--zone NAME:X1,Y1,X2,Y2`. Values are **relative to the original video
dimensions** and must lie between 0 and 1. Top left is `0,0`, bottom right
is `1,1`; `X1 < X2` and `Y1 < Y2`.

A detection is inside a zone when its **bounding-box bottom centre** lies in that
rectangle. This provides a reasonable approximation of a person's foot position
without requiring segmentation geometry. Zone coordinates are not pixel values.
Zones can overlap and objects can generate events in multiple zones.

Omit `--zone` to use the full frame.

## Outputs

Each run creates a unique UTC timestamped directory:

```text
runs/monitor/20261008T120000_123456Z/
  events.jsonl
  occupancy.csv
  summary.json
  annotated.mp4    # only with --save-video; may be .avi on codec fallback
```

- **events.jsonl:** one JSON event per line: `enter`, `exit`, `dwell`.
  Contains zone, class, tracker ID, media-relative timestamp, frame, bounding
  box and confidence. Exit/dwell events also include elapsed visit duration.
- **occupancy.csv:** current object count per zone, per frame; zero counts are
  included. Occupancy can count objects that are not yet assigned a track ID.
- **summary.json:** frames, processing throughput, selected zones/classes,
  event totals and count of observations without track IDs.
- **annotated.mp4 / .avi:** optional local video with tracking overlays and
  zone outlines; requires a working OpenCV encoder.

Example entry event:

```json
{"type":"enter","zone":"entrance","class_name":"person","track_id":7,"time_s":2.1,"frame":64,"box":[100,50,180,240],"confidence":0.9123}
```

## Important distinctions and limitations

- **Occupancy** counts currently visible detections, not unique visitors.
- **Entry counts** count *visits to zones* tied to tracker IDs, not biometric
  identities or provably unique people. The same person may be counted again
  when tracking is lost for longer than `--lost-timeout` or on re-entry.
- **Dwell** fires once per uninterrupted visit after `--dwell` seconds;
  set `--dwell 0` to disable it. Missing tracks receive an exit after
  the configured grace period. Events on lost tracks are timestamped at their
  last observation rather than at the timeout.
- Video-file timestamps prefer OpenCV's position in milliseconds, falling
  back to frame index / FPS. Cameras use local monotonic elapsed time.
- Tracking reliability depends on lighting, occlusion, motion blur, model,
  tracker settings and camera angle. Validate accuracy for your use case.
- Sources are processed sequentially, one camera per process. Automatic RTSP
  reconnection, multiple synchronized cameras, a web dashboard, notification
  delivery and storage retention policies are **not yet implemented**.
- Runs are local by default, but they still contain object detections and may
  constitute personal data. Consider permission, retention and applicable law
  before monitoring identifiable people.
- The repository's MIT license does not remove third-party model, weight and
  Ultralytics licensing obligations; check their terms before distribution.

## CLI reference

```bash
rto-monitor --help
# Equivalent without installation:
python -m src.monitor --help
```

Useful settings: `--model` selects Ultralytics weights (default
`yolo11n-seg.pt`); `--confidence` defaults to 0.35, `--iou` to 0.5 and
`--imgsz` to 640. `--tracker` supports `bytetrack.yaml` (default) and
`botsort.yaml`. Use `--device cpu` to force CPU. `--max-frames` limits a
session. `--output-dir` selects the parent directory for new runs.

## Test without downloading weights

```bash
python -m unittest discover -s tests -p 'test_monitor_core.py' -v
```

The zone/event engine and a simulated camera-to-output pipeline have fast
offline tests. **Live camera performance and real model accuracy have not been
benchmarked as part of this change**. The existing project evaluation commands
remain available for comparative research.
