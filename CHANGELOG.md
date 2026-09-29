# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed

- **Same-file conversion safety** — video/image conversions that target the input file
  are written to a temporary file and swapped, matching the audio path.
- **`process_video_with_model` / `process_webcam_with_model`** returned from inside a
  `finally` block (which swallowed exceptions); resources are now released first and the
  statistics returned afterwards.
- **Video capture leak** in `validate_video` — the capture is always released.
- **`VideoWriter` failures were silent** — output writers now try `avc1` → `mp4v` →
  MJPG/AVI and are checked with `isOpened()`; a container reporting no/NaN FPS now falls
  back to 30 FPS instead of creating a 0-FPS writer.
- **Overlay detections never drew** because `apply_standardized_layout` read `bbox` /
  `confidence` while the model produced `box` / `score`; detections now expose both keys.
- **`evaluate_models.py`** no longer raises `KeyError` when no inference succeeded, and a
  missing module-level `traceback` import was restored.
- **`app.py`**: the CLI `--model-type` choices no longer reference non-existent keys; the
  GUI `(Auto-Best)` option is honoured; the "Compress Datasets" action actually runs;
  the video preview uses PIL/Canvas correctly; and a scrollbar was bound to the wrong widget.
- **`dataset_manager.py`**: fixed an `UnboundLocalError` on download failure and an
  undefined `val2017_subset_dir` `NameError` in subset creation.
- **`validation.py`**: fixed an operator-precedence bug in the MPS availability check.
- Division-by-zero guards for FPS-derived dashboard metrics.
- Sample-video lookup no longer points at a non-existent `.../samples` directory.

### Changed

- **Device selection** is now first-class: `ModelManager`/`YOLOWrapper` accept an explicit
  `device`, honouring `RTO_DEVICE`/`CV_FORCE_DEVICE` and `FORCE_CPU_INFERENCE`. `--device`
  is wired through the CLIs.
- **Heavy dependencies are imported lazily** (`torch`, `torchvision`, `ultralytics`), so the
  package imports cleanly without them and the model paths resolve relative to the repo
  (works from any working directory).
- **Performance**: removed the artificial per-frame `time.sleep(0.01)`, dropped a duplicate
  progress callback, and added an optional `return_masks=False` for video output where masks
  are unnecessary.
- **`create_demo_video.py`** now supports custom weights (`--model-path`) and a `--device` flag.
- Added `pyproject.toml`, `LICENSE`, and fixed `pytest.ini` (`testpaths`, markers).
- Tests were rewritten against the real APIs and skip gracefully without model weights;
  model-dependent tests are marked `slow`.

### Removed

- Dead MMDetection leftovers (`src/default_runtime.py`, `models/configs/`).
