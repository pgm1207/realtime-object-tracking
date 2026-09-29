# Real-Time Object Tracking

Object **detection and instance segmentation** in video sequences using the YOLO-Seg
family, with COCO evaluation, a metrics dashboard, and a demo-video generator.

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-orange.svg)

This project fulfils **Task 2: Recognizing Objects in Video Sequences** — it evaluates
three state-of-the-art segmentation models on a representative COCO subset and produces
an annotated demo video with the best one. See
[docs/task2_solution.md](docs/task2_solution.md) for the report and methodology.

## Quick start

```bash
# 1. Install dependencies (use a virtualenv)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Evaluate three SoA models on 50 COCO images and build the dashboard
python src/evaluate_models.py --images 50 --models yolov8n-seg yolo11m-seg yolov9e-seg

# 3. Generate a demo video with the best model found
python src/create_demo_video.py --best-model --video data_sets/video_data/people-detection.mp4
```

Model weights (`models/pts/*.pt`) are downloaded automatically by Ultralytics on first
use; the first run therefore needs network access and can take a few minutes. The COCO
val2017 subset is downloaded by the evaluation/dataset tools on first run too.

## CLI tools

| Command | Purpose |
|---|---|
| `python src/evaluate_models.py --images 50 --models ... [--device cpu] [--no-vis]` | Evaluate models on COCO, write JSON + a dashboard. |
| `python src/create_demo_video.py --model <name> \| --best-model [--model-path W] [--video V] [--output O] [--device D]` | Produce an annotated demo video. |
| `python src/generate_dashboard.py [--results FILE] [--output FILE] [--show]` | Regenerate the metrics dashboard from a results JSON. |
| `python src/pipeline.py [--images N] [--models ...] [--demo-video V]` | Run evaluate → dashboard → demo in one step. |
| `python src/compare_models.py --models A B [--video V] [--output O]` | Side-by-side comparison video for several models. |
| `python src/app.py [--cli --model-type <name> --video-path V [--output O] [--no-display] [--device D]]` | Unified app: Tkinter GUI (default) or a headless CLI. |

### Device selection

Inference runs on the GPU when CUDA is available and falls back to CPU otherwise. Force a
device with `--device cpu|cuda|mps` (CLI) or the `RTO_DEVICE` environment variable.

### Headless / server use

The `--cli` path is headless-safe: it writes the annotated video to `--output` and only
opens a preview window when a display is detected (override with `--no-display`).

## Supported models

YOLOv8-Seg (`yolov8n/s/m/l/x-seg`), YOLOv9-Seg (`yolov9c/e-seg`) and YOLO11-Seg
(`yolo11n/s/m/l/x-seg`). All output a bounding box, a COCO class id/name, a confidence
score, and per-instance masks.

## Project structure

```
src/
  models.py            # YOLO wrapper + ModelManager (lazy heavy imports)
  video_utils.py       # Frame loops, codec-aware writers, standardised overlay
  evaluate_models.py   # COCO evaluation + CLI
  metrics_visualizer.py# Metrics dashboard / PR curves
  create_demo_video.py # Demo video generator + CLI
  compare_models.py    # Multi-model comparison video
  pipeline.py          # evaluate -> dashboard -> demo orchestrator
  app.py               # Tkinter GUI + headless --cli
  validation.py        # Input validators
  error_handling.py    # Logging + decorators
data_sets/
  dataset_manager.py   # COCO download / subset / compression
tests/                 # pytest unit + integration tests
```

## Tests

```bash
pip install -r dev-requirements.txt
pytest                       # unit tests; model-dependent tests skip without weights
pytest -m unit               # fast tests only
nox -s basic_tests           # via nox
```

## Documentation

- [Setup and installation](docs/setup_installation.md)
- [Usage guide](docs/usage_guide.md)
- [Task 2 solution](docs/task2_solution.md)
- [Testing & QA](docs/testing_qa.md)
- [Troubleshooting](docs/reproducibility_troubleshooting.md)

## License

MIT — see [LICENSE](LICENSE).
