"""Model-free tests of GNOME desktop command generation and Flatpak metadata."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from src.desktop_backend import DesktopJob


ROOT = Path(__file__).resolve().parent.parent
PREVIEW = Path("/tmp/object-monitor-preview.jpg")
OUTPUT = Path("/tmp/object-monitor-runs")


class DesktopJobTests(unittest.TestCase):
    def test_valid_command(self):
        job = DesktopJob(source="0", zones="door:0,0,0.5,1;room:0.5,0,1,1",
                         classes="person", record=True, output_dir=OUTPUT)
        cmd = job.command(PREVIEW, executable="/usr/bin/python3")
        self.assertEqual(cmd[:4], ["/usr/bin/python3", "-m", "src.monitor", "--source"])
        self.assertEqual(cmd.count("--zone"), 2)
        self.assertIn("--save-video", cmd)
        self.assertEqual(cmd[cmd.index("--preview-snapshot") + 1], str(PREVIEW))

    def test_recording_opt_in(self):
        job = DesktopJob(source="video.mp4", output_dir=OUTPUT)
        self.assertNotIn("--save-video", job.command(PREVIEW))

    def test_reject_empty_source(self):
        with self.assertRaisesRegex(ValueError, "source|video|camera"):
            DesktopJob(source=" ", output_dir=OUTPUT).command(PREVIEW)

    def test_reject_bad_zone_and_duplicate(self):
        for zones in ("foo:0,0,0,1", "a:0,0,1,1;a:0,0,1,1"):
            with self.subTest(zones=zones), self.assertRaises(ValueError):
                DesktopJob(source="0", zones=zones, output_dir=OUTPUT).command(PREVIEW)

    def test_reject_invalid_confidence(self):
        for number in (-1, 1.3, float("nan")):
            with self.subTest(value=number), self.assertRaises(ValueError):
                DesktopJob(source="0", confidence=number,
                           output_dir=OUTPUT).command(PREVIEW)

    def test_reject_negative_dwell(self):
        with self.assertRaises(ValueError):
            DesktopJob(source="0", dwell=-1, output_dir=OUTPUT).command(PREVIEW)

    def test_never_shell_execute_user_supplied_source(self):
        input_source = "weird filename ; touch /tmp/should-not-execute.mp4"
        cmd = DesktopJob(source=input_source, output_dir=OUTPUT).command(PREVIEW)
        self.assertEqual(cmd[cmd.index("--source") + 1], input_source)
        self.assertNotIn("sh", cmd[:3])

    def test_absolute_paths(self):
        with self.assertRaisesRegex(ValueError, "absolute"):
            DesktopJob(source="0", output_dir=Path("relative")).command(PREVIEW)


class FlatpakMetadataTests(unittest.TestCase):
    def test_manifest_valid_and_native_runtime(self):
        manifest = json.loads((ROOT / "flatpak" / "io.github.pgm1207.ObjectMonitor.json").read_text())
        self.assertEqual(manifest["app-id"], "io.github.pgm1207.ObjectMonitor")
        self.assertEqual(manifest["runtime"], "org.gnome.Platform")
        self.assertEqual(manifest["runtime-version"], "50")
        self.assertIn("--socket=wayland", manifest["finish-args"])
        self.assertNotIn("--filesystem=host", manifest["finish-args"])

    def test_no_host_python_or_host_spawn_in_launcher(self):
        launcher = (ROOT / "flatpak" / "object-monitor-launch").read_text()
        self.assertIn("python3 -m src.desktop", launcher)
        self.assertNotIn("flatpak-spawn", launcher)


if __name__ == "__main__":
    unittest.main()
