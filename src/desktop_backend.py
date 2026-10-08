"""Headless configuration adapter between the GNOME UI and monitor CLI.

This module intentionally avoids GTK, OpenCV and ML imports for fast testing.
"""
from dataclasses import dataclass
from pathlib import Path
import math
import sys

from .monitor_core import Zone


@dataclass(frozen=True)
class DesktopJob:
    source: str
    model: str = "yolo11n-seg.pt"
    classes: str = "person"
    zones: str = "frame:0,0,1,1"
    confidence: float = 0.35
    dwell: float = 5.0
    record: bool = False
    output_dir: Path = Path.home() / "Videos" / "Object Monitor"

    def command(self, snapshot: Path, *, executable: str | None = None) -> list[str]:
        source = self.source.strip()
        if not source:
            raise ValueError("Choose a video file, webcam number, or RTSP stream")
        model = self.model.strip()
        if not model:
            raise ValueError("Choose a YOLO model or provide weights")
        classes = self.classes.strip()
        if not classes:
            raise ValueError("Enter at least one class, or 'all'")
        zones = [part.strip() for part in self.zones.split(";")]
        parsed = [Zone.parse(zone) for zone in zones]
        if len({zone.name for zone in parsed}) != len(parsed):
            raise ValueError("Zone names must be unique")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        if not math.isfinite(self.dwell) or self.dwell < 0:
            raise ValueError("Dwell seconds must be zero or greater")
        if not snapshot.is_absolute() or not self.output_dir.is_absolute():
            raise ValueError("Preview and result paths must be absolute")
        args = [
            executable or sys.executable, "-m", "src.monitor",
            "--source", source, "--model", model, "--classes", classes,
            "--confidence", str(self.confidence), "--dwell", str(self.dwell),
            "--output-dir", str(self.output_dir),
            "--preview-snapshot", str(snapshot),
        ]
        for zone in parsed:
            args.extend(["--zone", f"{zone.name}:{zone.x1},{zone.y1},{zone.x2},{zone.y2}"])
        if self.record:
            args.append("--save-video")
        return args
