"""GNOME desktop frontend for the local object-monitoring CLI.

GTK and libadwaita are intentionally confined to this module. The CPU/GPU-heavy
inference process is isolated and its JPEG snapshots are polled asynchronously
without touching GTK from worker threads.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gdk, Gio, GLib, Gtk

from .desktop_backend import DesktopJob


APP_ID = "io.github.pgm1207.ObjectMonitor"


def _label(text: str, style: str | None = None, *, wrap: bool = False) -> Gtk.Label:
    label = Gtk.Label(label=text, xalign=0)
    label.set_wrap(wrap)
    if style:
        label.add_css_class(style)
    return label


class MonitorWindow(Adw.ApplicationWindow):
    def __init__(self, app: Adw.Application):
        super().__init__(application=app, title="Object Monitor", default_width=1120,
                         default_height=780)
        self.set_size_request(760, 540)
        self._process: subprocess.Popen | None = None
        self._stopping = False
        self._run_dir: Path | None = None
        self._last_snapshot_mtime = 0
        self._event_offset = 0
        self._event_totals = {"enter": 0, "exit": 0, "dwell": 0}
        self._chooser = None
        self._snapshot = Path(GLib.get_user_cache_dir()) / "object-monitor" / "preview.jpg"
        self._snapshot.parent.mkdir(parents=True, exist_ok=True)
        self._output_root = Path(GLib.get_user_data_dir()) / "object-monitor" / "runs"

        toolbar = Adw.ToolbarView()
        header = Adw.HeaderBar()
        header.set_title_widget(Adw.WindowTitle(title="Object Monitor", subtitle="Private video analytics"))
        toolbar.add_top_bar(header)
        self.set_content(toolbar)

        split = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        split.set_shrink_start_child(False)
        split.set_shrink_end_child(False)
        toolbar.set_content(split)

        sidebar_scroll = Gtk.ScrolledWindow()
        sidebar_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        sidebar_scroll.set_size_request(350, -1)
        split.set_start_child(sidebar_scroll)

        sidebar = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=18)
        sidebar.set_margin_top(18)
        sidebar.set_margin_bottom(18)
        sidebar.set_margin_start(18)
        sidebar.set_margin_end(18)
        sidebar_scroll.set_child(sidebar)

        title = _label("New analysis", "title-1")
        sidebar.append(title)
        sidebar.append(_label("Choose a source and define what you want to measure.",
                              wrap=True))

        input_group = Adw.PreferencesGroup(title="Video input")
        sidebar.append(input_group)
        self.source = Adw.EntryRow(title="File path, camera ID or RTSP URL")
        input_group.add(self.source)
        browse = Gtk.Button(icon_name="document-open-symbolic", valign=Gtk.Align.CENTER,
                            tooltip_text="Choose a video")
        browse.add_css_class("flat")
        browse.connect("clicked", self._choose_source)
        self.source.add_suffix(browse)

        self.model = Adw.EntryRow(title="YOLO model or weights")
        self.model.set_text("yolo11n-seg.pt")
        input_group.add(self.model)
        self.classes = Adw.EntryRow(title="Object classes")
        self.classes.set_text("person")
        input_group.add(self.classes)

        zone_group = Adw.PreferencesGroup(title="Tracking zones",
                                          description="Normalized x1,y1,x2,y2; separate multiple zones with semicolons")
        sidebar.append(zone_group)
        self.zones = Adw.EntryRow(title="Zones")
        self.zones.set_text("frame:0,0,1,1")
        zone_group.add(self.zones)
        self.dwell = Adw.EntryRow(title="Dwell time (seconds)")
        self.dwell.set_text("5")
        zone_group.add(self.dwell)
        self.confidence = Adw.EntryRow(title="Minimum confidence (0–1)")
        self.confidence.set_text("0.35")
        zone_group.add(self.confidence)

        storage_group = Adw.PreferencesGroup(title="Output")
        sidebar.append(storage_group)
        self.record = Adw.SwitchRow(title="Save annotated video",
                                     subtitle="Off by default; events and statistics are always saved")
        storage_group.add(self.record)

        actions = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        sidebar.append(actions)
        self.start_button = Gtk.Button(label="Start analysis", hexpand=True)
        self.start_button.add_css_class("suggested-action")
        self.start_button.add_css_class("pill")
        self.start_button.connect("clicked", self._start)
        actions.append(self.start_button)
        self.stop_button = Gtk.Button(label="Stop", sensitive=False)
        self.stop_button.connect("clicked", self._stop)
        actions.append(self.stop_button)

        self.status = _label("Ready. Select a video or enter camera 0.", wrap=True)
        self.status.add_css_class("dim-label")
        sidebar.append(self.status)

        workspace = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        workspace.set_margin_top(18)
        workspace.set_margin_bottom(18)
        workspace.set_margin_start(16)
        workspace.set_margin_end(18)
        split.set_end_child(workspace)

        workspace.append(_label("Preview", "title-2"))
        workspace.append(_label("Processed frames are displayed here while analysis runs.",
                                wrap=True))

        self.preview_stack = Gtk.Stack(vexpand=True, hexpand=True)
        self.preview_stack.set_transition_type(Gtk.StackTransitionType.CROSSFADE)
        workspace.append(self.preview_stack)

        empty = Adw.StatusPage(icon_name="video-display-symbolic",
                               title="No video loaded",
                               description="Start an analysis to see tracking and zone overlays.")
        self.preview_stack.add_named(empty, "empty")
        self.picture = Gtk.Picture(content_fit=Gtk.ContentFit.CONTAIN, can_shrink=True,
                                   hexpand=True, vexpand=True)
        self.preview_stack.add_named(self.picture, "image")
        self.preview_stack.set_visible_child_name("empty")

        stats = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=25)
        workspace.append(stats)
        self.entry_count = _label("Entries: 0", "heading")
        self.exit_count = _label("Exits: 0", "heading")
        self.dwell_count = _label("Dwell: 0", "heading")
        for item in (self.entry_count, self.exit_count, self.dwell_count):
            stats.append(item)

        bottom = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        workspace.append(bottom)
        bottom.append(_label("Activity", "title-3"))
        self.open_results = Gtk.Button(label="Open results", sensitive=False,
                                       halign=Gtk.Align.END, hexpand=True)
        self.open_results.connect("clicked", self._open_results)
        bottom.append(self.open_results)

        log_scroller = Gtk.ScrolledWindow(min_content_height=130, max_content_height=190)
        log_scroller.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        workspace.append(log_scroller)
        self.log_view = Gtk.TextView(editable=False, cursor_visible=False, monospace=True,
                                     wrap_mode=Gtk.WrapMode.WORD_CHAR)
        self.log_view.set_top_margin(8)
        self.log_view.set_bottom_margin(8)
        self.log_view.set_left_margin(10)
        self.log_view.set_right_margin(10)
        self.log_buffer = self.log_view.get_buffer()
        log_scroller.set_child(self.log_view)
        self._log("Ready. Files and model inference stay on this computer.")

        GLib.timeout_add(250, self._refresh)
        self.connect("close-request", self._on_close)

    def _choose_source(self, _button):
        dialog = Gtk.FileChooserNative.new("Choose a video", self, Gtk.FileChooserAction.OPEN,
                                            "Open", "Cancel")
        filters = Gtk.FileFilter()
        filters.set_name("Videos")
        for mime in ("video/mp4", "video/x-matroska", "video/x-msvideo", "video/quicktime",
                     "video/webm"):
            filters.add_mime_type(mime)
        dialog.add_filter(filters)
        dialog.connect("response", self._file_response)
        self._chooser = dialog
        dialog.show()

    def _file_response(self, dialog, response):
        if response == Gtk.ResponseType.ACCEPT:
            picked = dialog.get_file()
            if picked and picked.get_path():
                self.source.set_text(picked.get_path())
        dialog.destroy()
        self._chooser = None

    def _log(self, line: str):
        text = line.rstrip()[:500]
        if not text:
            return False
        buffer = self.log_buffer
        buffer.insert(buffer.get_end_iter(), text + "\n")
        if buffer.get_char_count() > 12000:
            start = buffer.get_start_iter()
            end = buffer.get_iter_at_offset(buffer.get_char_count() - 10000)
            buffer.delete(start, end)
        self.log_view.scroll_to_iter(buffer.get_end_iter(), 0.0, False, 0, 0)
        return False

    def _start(self, _button):
        if self._process is not None:
            return
        try:
            job = DesktopJob(source=self.source.get_text(),
                             model=self.model.get_text(),
                             classes=self.classes.get_text(),
                             zones=self.zones.get_text(),
                             confidence=float(self.confidence.get_text()),
                             dwell=float(self.dwell.get_text()),
                             record=self.record.get_active(),
                             output_dir=self._output_root)
            command = job.command(self._snapshot)
        except ValueError as exc:
            self.status.set_text(f"Check settings: {exc}")
            return
        self._snapshot.unlink(missing_ok=True)
        self._last_snapshot_mtime = 0
        self._run_dir = None
        self._event_offset = 0
        self._event_totals = {"enter": 0, "exit": 0, "dwell": 0}
        self.preview_stack.set_visible_child_name("empty")
        self._update_counts()
        self.open_results.set_sensitive(False)
        self._stopping = False
        try:
            self._process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
        except OSError as exc:
            self.status.set_text(f"Could not launch tracking engine: {exc}")
            return
        self.start_button.set_sensitive(False)
        self.stop_button.set_sensitive(True)
        self.status.set_text("Tracking in progress…")
        self._log("Starting local inference process…")
        threading.Thread(target=self._consume_output, args=(self._process,), daemon=True).start()

    def _consume_output(self, process):
        try:
            if process.stdout:
                for line in process.stdout:
                    GLib.idle_add(self._receive_line, line.rstrip())
            code = process.wait()
        except Exception as exc:
            GLib.idle_add(self._receive_line, f"Worker error: {exc}")
            code = process.poll()
        GLib.idle_add(self._complete, process, code)

    def _receive_line(self, line: str):
        if line.startswith("RTO_RUN_DIR="):
            self._run_dir = Path(line.split("=", 1)[1])
            self.open_results.set_sensitive(True)
        elif line:
            self._log(line)
        return False

    def _stop(self, _button):
        process = self._process
        if process is None or process.poll() is not None:
            return
        self._stopping = True
        self.status.set_text("Stopping gracefully…")
        self.stop_button.set_sensitive(False)
        process.terminate()
        GLib.timeout_add_seconds(8, self._force_stop, process)

    @staticmethod
    def _force_stop(process):
        if process.poll() is None:
            process.kill()
        return False

    def _complete(self, process, code):
        if self._process is not process:
            return False
        self._process = None
        self.start_button.set_sensitive(True)
        self.stop_button.set_sensitive(False)
        if self._stopping:
            self.status.set_text("Stopped. Available results have been saved.")
        elif code == 0:
            self.status.set_text("Analysis complete. Reports are ready.")
        else:
            self.status.set_text(f"Analysis failed (exit code {code}). Review activity log.")
        self._refresh()
        return False

    def _update_counts(self):
        self.entry_count.set_text(f"Entries: {self._event_totals['enter']}")
        self.exit_count.set_text(f"Exits: {self._event_totals['exit']}")
        self.dwell_count.set_text(f"Dwell: {self._event_totals['dwell']}")

    def _refresh(self):
        if self._snapshot.exists():
            try:
                stamp = self._snapshot.stat().st_mtime_ns
                if stamp != self._last_snapshot_mtime:
                    texture = Gdk.Texture.new_from_filename(str(self._snapshot))
                    self.picture.set_paintable(texture)
                    self.preview_stack.set_visible_child_name("image")
                    self._last_snapshot_mtime = stamp
            except (OSError, GLib.Error):
                pass
        if self._run_dir:
            event_path = self._run_dir / "events.jsonl"
            if event_path.exists():
                try:
                    with event_path.open(encoding="utf-8") as events:
                        events.seek(self._event_offset)
                        while True:
                            previous_offset = events.tell()
                            line = events.readline()
                            if not line:
                                break
                            if not line.endswith("\n"):
                                # A writer may still be appending this JSON event.
                                events.seek(previous_offset)
                                break
                            try:
                                event = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            category = event.get("type")
                            if category in self._event_totals:
                                self._event_totals[category] += 1
                        self._event_offset = events.tell()
                    self._update_counts()
                except OSError:
                    pass
        return True

    def _open_results(self, _button):
        if self._run_dir:
            try:
                Gio.AppInfo.launch_default_for_uri(self._run_dir.as_uri(), None)
            except GLib.Error as exc:
                self._log(f"Could not open results folder: {exc}")

    def _on_close(self, _window):
        if self._process and self._process.poll() is None:
            self._process.terminate()
        self._snapshot.unlink(missing_ok=True)
        return False


class ObjectMonitorApp(Adw.Application):
    def __init__(self):
        super().__init__(application_id=APP_ID, flags=Gio.ApplicationFlags.DEFAULT_FLAGS)

    def do_activate(self):
        window = self.props.active_window
        if window is None:
            window = MonitorWindow(self)
        window.present()


def main(argv: list[str] | None = None) -> int:
    return ObjectMonitorApp().run(argv if argv is not None else sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())
