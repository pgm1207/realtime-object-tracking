"""Local-first video analytics CLI: tracking, zones, events and occupancy export.

Run with ``python -m src.monitor --source video.mp4 --zone gate:.2,.3,.8,.9``.
Imports torch/ultralytics only when actually running inference.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import threading
import time
from urllib.parse import urlsplit, urlunsplit

from .monitor_core import Observation, Zone, ZoneMonitor


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def nonnegative_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return number


def probability(value: str) -> float:
    number = nonnegative_float(value)
    if number > 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Monitor a local video, camera or RTSP feed; write tracking events and zone occupancy."
    )
    p.add_argument("--source", required=True, help="Video path, webcam index (0), or RTSP/HTTP URL")
    p.add_argument("--model", default="yolo11n-seg.pt", help="Ultralytics model name or local .pt weights")
    p.add_argument("--zone", action="append", default=[], metavar="NAME:X1,Y1,X2,Y2",
                   help="Normalized rectangular zone; repeat for multiple zones (default: full frame)")
    p.add_argument("--classes", default="person", help="Comma-separated model class names, or 'all'")
    p.add_argument("--confidence", type=probability, default=0.35)
    p.add_argument("--iou", type=probability, default=0.5)
    p.add_argument("--imgsz", type=positive_int, default=640)
    p.add_argument("--device", default=None, help="Ultralytics device, e.g. cpu, cuda:0, mps")
    p.add_argument("--tracker", choices=["bytetrack.yaml", "botsort.yaml"], default="bytetrack.yaml")
    p.add_argument("--dwell", type=nonnegative_float, default=5.0, metavar="SECONDS",
                   help="Emit one dwell event per continuous visit after N seconds; 0 disables")
    p.add_argument("--lost-timeout", type=nonnegative_float, default=1.0, metavar="SECONDS")
    p.add_argument("--max-frames", type=positive_int, help="Stop after N captured frames")
    p.add_argument("--output-dir", type=Path, default=Path("runs/monitor"),
                   help="Parent folder for timestamped run outputs")
    p.add_argument("--save-video", action="store_true", help="Opt in to saving annotated video")
    p.add_argument("--preview", action="store_true", help="Display live preview (q or Esc to quit)")
    p.add_argument("--preview-snapshot", type=Path,
                   help="Publish a periodically updated JPEG for a desktop frontend")
    return p


def _safe_source(source: str) -> str:
    """Never write RTSP/HTTP credentials to an analytics report."""
    parsed = urlsplit(source)
    if parsed.scheme not in ("rtsp", "rtsps", "http", "https"):
        return source
    host = parsed.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    port = f":{parsed.port}" if parsed.port is not None else ""
    return urlunsplit((parsed.scheme, host + port, parsed.path, parsed.query, ""))


def _source_arg(source: str):
    if source.isdecimal():
        return int(source), True
    if "://" in source:
        return source, True
    if not Path(source).is_file():
        raise FileNotFoundError(f"Input video does not exist: {source}")
    return source, False


def _resolve_classes(model, selected: str) -> tuple[list[int] | None, list[str]]:
    names = model.names
    lookup = {str(name).casefold(): int(idx) for idx, name in
              (names.items() if isinstance(names, dict) else enumerate(names))}
    if selected.strip().casefold() == "all":
        return None, list(lookup)
    wanted = [part.strip().casefold() for part in selected.split(",")]
    if not wanted or any(not name for name in wanted):
        raise ValueError("--classes must be comma-separated class names or 'all'")
    unknown = sorted(set(wanted) - lookup.keys())
    if unknown:
        raise ValueError(f"Unknown model classes: {', '.join(unknown)}. Available: {', '.join(lookup)}")
    selected_names = list(dict.fromkeys(wanted))
    return [lookup[name] for name in selected_names], selected_names


def _extract_observations(result) -> list[Observation]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    names = result.names
    ids = boxes.id.int().cpu().tolist() if boxes.id is not None else None
    coordinates = boxes.xyxy.cpu().tolist()
    classes = boxes.cls.int().cpu().tolist()
    confidences = boxes.conf.cpu().tolist()
    observations = []
    for n, (box, cls, confidence) in enumerate(zip(coordinates, classes, confidences)):
        name = names[cls]
        observations.append(Observation(
            class_name=str(name).casefold(), confidence=float(confidence),
            box=tuple(map(float, box)), track_id=int(ids[n]) if ids is not None else None,
        ))
    return observations


def _draw_zones(frame, zones: list[Zone], occupancy: dict, cv2):
    height, width = frame.shape[:2]
    for zone in zones:
        pt1 = (round(zone.x1 * width), round(zone.y1 * height))
        pt2 = (round(zone.x2 * width), round(zone.y2 * height))
        count = sum(number for (name, _cls), number in occupancy.items() if name == zone.name)
        cv2.rectangle(frame, pt1, pt2, (0, 215, 255), 2)
        cv2.putText(frame, f"{zone.name}: {count}", (pt1[0] + 5, max(20, pt1[1] + 22)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 215, 255), 2)
    return frame


def run(args: argparse.Namespace) -> dict:
    # Lazy imports keep --help and dependency-free tests fast.
    import cv2
    from ultralytics import YOLO
    from .video_utils import _safe_fps, open_video_writer

    zones = [Zone.parse(text) for text in args.zone] if args.zone else [Zone.parse("frame:0,0,1,1")]
    monitor = ZoneMonitor(zones, dwell_seconds=args.dwell, lost_timeout=args.lost_timeout)
    source, live = _source_arg(args.source)
    model = YOLO(args.model)
    class_ids, class_names = _resolve_classes(model, args.classes)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Cannot open video source. Check path, webcam permissions, or stream connectivity")
    fps = _safe_fps(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Video source returned invalid dimensions")

    run_dir = args.output_dir / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except OSError:
        cap.release()
        raise
    video_path = run_dir / "annotated.mp4"
    writer = None
    frames = 0
    last_seconds = 0.0
    started = time.perf_counter()
    stop_requested = threading.Event()
    # Graceful termination preserves valid JSON and releases capture/writer.
    signal_handler = None
    if threading.current_thread() is threading.main_thread():
        signal_handler = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, lambda _signum, _frame: stop_requested.set())
    last_snapshot_at = 0.0
    print(f"RTO_RUN_DIR={run_dir}", flush=True)
    print(f"Monitoring {args.source if not live else 'live source'}; reports: {run_dir}")
    try:
        if args.save_video:
            writer = open_video_writer(video_path, fps, (width, height))
            if writer is None:
                raise RuntimeError("Could not initialize an output video encoder")
        with (run_dir / "events.jsonl").open("w", encoding="utf-8") as event_file, \
                (run_dir / "occupancy.csv").open("w", newline="", encoding="utf-8") as metrics_file:
            csv_writer = csv.writer(metrics_file)
            csv_writer.writerow(["time_s", "frame", "zone", "count"])
            while not stop_requested.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                frames += 1
                if live:
                    seconds = time.perf_counter() - started
                else:
                    position_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    expected = (frames - 1) / fps
                    # Some codecs report zero/invalid or regressive timestamps.
                    candidate = position_ms / 1000 if math.isfinite(position_ms) and position_ms > 0 else expected
                    seconds = max(last_seconds, candidate)
                last_seconds = seconds

                kwargs = {"persist": True, "tracker": args.tracker, "conf": args.confidence,
                          "iou": args.iou, "imgsz": args.imgsz, "verbose": False}
                if class_ids is not None:
                    kwargs["classes"] = class_ids
                if args.device:
                    kwargs["device"] = args.device
                results = model.track(frame, **kwargs)
                if not results:
                    raise RuntimeError(f"Tracker returned no result for frame {frames}")
                result = results[0]
                observations = _extract_observations(result)
                events, occupancy = monitor.update(observations, seconds, frames, (width, height))
                for event in events:
                    event_file.write(json.dumps(event) + "\n")
                if events:
                    event_file.flush()
                for zone in zones:
                    count = sum(v for (name, _), v in occupancy.items() if name == zone.name)
                    csv_writer.writerow([f"{seconds:.3f}", frames, zone.name, count])

                if writer is not None or args.preview or args.preview_snapshot:
                    annotated = _draw_zones(result.plot(), zones, occupancy, cv2)
                    if args.preview_snapshot:
                        now = time.perf_counter()
                        if now - last_snapshot_at >= 0.2:
                            target = args.preview_snapshot
                            target.parent.mkdir(parents=True, exist_ok=True)
                            temporary = target.with_name(f".{target.stem}.tmp.jpg")
                            if cv2.imwrite(str(temporary), annotated):
                                os.replace(temporary, target)
                            last_snapshot_at = now
                    if writer is not None:
                        writer.write(annotated)
                    if args.preview:
                        cv2.imshow("Local Object Monitor (q to quit)", annotated)
                        if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                            break
                if args.max_frames and frames >= args.max_frames:
                    break
                if frames % 120 == 0:
                    print(f"Frames: {frames} | entries: {monitor.totals['enter']}", flush=True)
            for event in monitor.finish(last_seconds, frames):
                event_file.write(json.dumps(event) + "\n")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.preview:
            cv2.destroyAllWindows()
        if signal_handler is not None:
            signal.signal(signal.SIGTERM, signal_handler)

    elapsed = time.perf_counter() - started
    if frames == 0:
        raise RuntimeError("No readable frames were returned by the video source")
    saved_video = None
    if args.save_video:
        # open_video_writer may fall back from .mp4 to MJPG in .avi.
        for candidate in (video_path, video_path.with_suffix(".avi")):
            if candidate.exists() and candidate.stat().st_size > 0:
                saved_video = str(candidate)
                break
    summary = {
        "source": _safe_source(args.source), "model": args.model, "classes": class_names,
        "zones": [vars(zone) for zone in zones], "frames": frames,
        "media_duration_s": round(last_seconds, 3), "processing_seconds": round(elapsed, 3),
        "processing_fps": round(frames / elapsed, 2) if elapsed else 0,
        "events": monitor.totals, "untracked_observations": monitor.untracked_observations,
        "annotated_video": saved_video,
        "note": "Entry counts are zone visits by tracker ID, not unique people across all time.",
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "frames": frames, "events": monitor.totals}, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    p = parser()
    args = p.parse_args(argv)
    try:
        run(args)
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        p.exit(1, f"Monitor error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
