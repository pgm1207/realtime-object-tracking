"""Fast offline tests for the product-facing zone monitor (no model downloads)."""
import unittest

from src.monitor_core import Observation, Zone, ZoneMonitor
from src.monitor import _resolve_classes, _safe_source, parser


class ZoneTests(unittest.TestCase):
    def test_parse_and_foot_point(self):
        zone = Zone.parse("gate:0.25,0.25,0.75,0.75")
        self.assertEqual(zone.name, "gate")
        self.assertTrue(zone.contains((30, 20, 50, 60), (100, 100)))
        self.assertFalse(zone.contains((0, 0, 10, 10), (100, 100)))

    def test_reject_bad_zones(self):
        for text in ("invalid", "x:0,0,0,1", "x:0,0,1,2", "x:0,nan,1,1", "x:0,0,1", ":0,0,1,1"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                Zone.parse(text)


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.zone = Zone.parse("gate:0.2,0.2,0.8,0.8")
        self.person = Observation("person", 0.9, (30, 30, 60, 60), 7)
        self.outside = Observation("person", 0.9, (0, 0, 10, 10), 7)

    def test_entry_dwell_exit_once(self):
        monitor = ZoneMonitor([self.zone], dwell_seconds=2, lost_timeout=2)
        events, occupancy = monitor.update([self.person], 0, 1, (100, 100))
        self.assertEqual([e["type"] for e in events], ["enter"])
        self.assertEqual(occupancy[("gate", "person")], 1)
        events, _ = monitor.update([self.person], 1, 2, (100, 100))
        self.assertEqual(events, [])
        events, _ = monitor.update([self.person], 2.1, 3, (100, 100))
        self.assertEqual([e["type"] for e in events], ["dwell"])
        self.assertEqual(events[0]["duration_s"], 2.1)
        events, _ = monitor.update([self.person], 3, 4, (100, 100))
        self.assertEqual(events, [])
        events, occupancy = monitor.update([self.outside], 4, 5, (100, 100))
        self.assertEqual([e["type"] for e in events], ["exit"])
        self.assertEqual(events[0]["duration_s"], 4)
        self.assertEqual(occupancy, {})
        self.assertEqual(monitor.totals, {"enter": 1, "exit": 1, "dwell": 1})

    def test_missing_grace_prevents_false_exit(self):
        monitor = ZoneMonitor([self.zone], lost_timeout=1)
        monitor.update([self.person], 0, 1, (100, 100))
        events, _ = monitor.update([], 0.5, 2, (100, 100))
        self.assertEqual(events, [])
        events, _ = monitor.update([self.person], 1, 3, (100, 100))
        self.assertEqual(events, [])
        events, _ = monitor.update([], 3, 4, (100, 100))
        self.assertEqual([e["type"] for e in events], ["exit"])
        self.assertEqual(events[0]["time_s"], 1)
        self.assertEqual(events[0]["reason"], "lost")

    def test_recycled_id_creates_new_visit(self):
        monitor = ZoneMonitor([self.zone], lost_timeout=1)
        monitor.update([self.person], 0, 1, (100, 100))
        events, _ = monitor.update([self.person], 5, 2, (100, 100))
        self.assertEqual([e["type"] for e in events], ["exit", "enter"])

    def test_untracked_occupancy_never_creates_unique_entry(self):
        monitor = ZoneMonitor([self.zone])
        anonymous = Observation("person", 0.8, self.person.box, None)
        events, occupancy = monitor.update([anonymous], 0, 1, (100, 100))
        self.assertEqual(events, [])
        self.assertEqual(occupancy[("gate", "person")], 1)
        self.assertEqual(monitor.untracked_observations, 1)

    def test_duplicate_track_ignored_and_flush(self):
        monitor = ZoneMonitor([self.zone])
        events, occupancy = monitor.update([self.person, self.person], 0, 1, (100, 100))
        self.assertEqual(len(events), 1)
        self.assertEqual(occupancy[("gate", "person")], 1)
        self.assertEqual([e["reason"] for e in monitor.finish(1, 2)], ["end_of_stream"])
        self.assertEqual(monitor.finish(2, 3), [])

    def test_regressive_time_rejected(self):
        monitor = ZoneMonitor([self.zone])
        monitor.update([], 5, 1, (100, 100))
        with self.assertRaises(ValueError):
            monitor.update([], 4, 2, (100, 100))


class CliTests(unittest.TestCase):
    def test_parse_offline(self):
        args = parser().parse_args(["--source", "0", "--zone", "door:0,0,1,1", "--dwell", "0"])
        self.assertEqual(args.zone, ["door:0,0,1,1"])
        self.assertEqual(args.dwell, 0)

    def test_class_names_validation(self):
        class Model:
            names = {0: "person", 2: "car"}
        self.assertEqual(_resolve_classes(Model(), "person,car"), ([0, 2], ["person", "car"]))
        with self.assertRaises(ValueError):
            _resolve_classes(Model(), "elephant")

    def test_credentials_never_exported(self):
        safe = _safe_source("rtsp://user:secret@example.org:8554/live?quality=hd")
        self.assertEqual(safe, "rtsp://example.org:8554/live?quality=hd")
        self.assertNotIn("secret", safe)


class EndToEndOfflineTests(unittest.TestCase):
    def test_writes_events_csv_and_summary_without_gpu_or_network(self):
        """Use mocked camera+tracker to exercise real CLI orchestration and I/O."""
        import csv
        import json
        from pathlib import Path
        import sys
        import tempfile
        from types import ModuleType, SimpleNamespace
        from unittest.mock import patch
        from src.monitor import run

        class Tensor:
            def __init__(self, values):
                self.values = values
            def int(self):
                return self
            def cpu(self):
                return self
            def tolist(self):
                return self.values

        class Boxes:
            id = Tensor([12])
            xyxy = Tensor([[30, 30, 60, 60]])
            cls = Tensor([0])
            conf = Tensor([0.95])
            def __len__(self):
                return 1

        class FakeYOLO:
            names = {0: "person"}
            def __init__(self, name):
                self.name = name
            def track(self, frame, **kwargs):
                return [SimpleNamespace(boxes=Boxes(), names=self.names)]

        class FakeCapture:
            index = 0
            def __init__(self, source):
                pass
            def isOpened(self):
                return True
            def get(self, prop):
                return {1: 10, 2: 100, 3: 100, 4: self.index * 100}.get(prop, 0)
            def read(self):
                self.index += 1
                return (True, object()) if self.index <= 2 else (False, None)
            def release(self):
                pass

        cv2 = ModuleType("cv2")
        cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_WIDTH = 1, 2
        cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_POS_MSEC = 3, 4
        cv2.VideoCapture = FakeCapture
        ultralytics = ModuleType("ultralytics")
        ultralytics.YOLO = FakeYOLO
        video_utils = ModuleType("src.video_utils")
        video_utils._safe_fps = lambda fps: fps
        video_utils.open_video_writer = lambda *a: None

        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {
            "cv2": cv2, "ultralytics": ultralytics, "src.video_utils": video_utils,
        }):
            args = parser().parse_args(["--source", "0", "--output-dir", tmp])
            run(args)
            output = next(Path(tmp).iterdir())
            report = json.loads((output / "summary.json").read_text())
            self.assertEqual(report["frames"], 2)
            self.assertEqual(report["events"], {"enter": 1, "exit": 1, "dwell": 0})
            events = [json.loads(s) for s in (output / "events.jsonl").read_text().splitlines()]
            self.assertEqual([event["type"] for event in events], ["enter", "exit"])
            with (output / "occupancy.csv").open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([r["count"] for r in rows], ["1", "1"])


if __name__ == "__main__":
    unittest.main()
