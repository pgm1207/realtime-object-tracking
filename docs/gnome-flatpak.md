# GNOME Desktop and Flatpak

The GNOME client is written in **GTK4 + libadwaita (PyGObject)** and uses
the existing local monitor as a child process. It is not an Electron app and
does not require a web browser. The UI follows GNOME system light/dark themes.

## Current capabilities

- Select a video using the system file chooser (also accepts webcam index or
  RTSP/HTTP URL in the Source field).
- Configure a model, class filters, rectangular zones, confidence and dwell.
- Start, watch progress, and stop analysis without freezing the UI.
- See a live *annotated JPEG snapshot* refreshed at most five times per second.
  It is a low-rate preview, not a high-FPS video widget.
- See entry, exit and dwell event counters and worker logs.
- Open the output directory using the system's default file manager.
- All inference remains local; recordings are disabled unless selected.

The app does not yet include graphical zone drawing, per-track event browsing,
RTSP auto-reconnect or cross-camera identity association.

## Run natively on CachyOS / Arch

```bash
sudo pacman -S --needed python python-gobject gtk4 libadwaita
git clone https://github.com/pgm1207/realtime-object-tracking.git
cd realtime-object-tracking

# Important: system GTK bindings must be visible in the venv.
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e .

python -m src.desktop
# Or, after installing:
rto-desktop
```

For other Linux distributions install GTK4, libadwaita and their Python
GObject introspection bindings, then install Python project dependencies
inside a `--system-site-packages` virtual environment.

**AMD note:** PyTorch GPU acceleration depends on a matching ROCm build and
hardware support. The existing monitor auto-detects CUDA/CPU, not an optimized
universal AMD inference backend. Assume CPU inference until ROCm, ONNX Runtime
or another target backend has been verified.

## Build and install the Flatpak (developer preview)

The manifest targets **GNOME Platform 50**. It uses the GNOME SDK and
`flatpak-pip-generator` to resolve pinned PyPI wheel downloads with hashes;
the app does **not** execute the host's Python interpreter. GTK/libadwaita come
from the GNOME runtime. Machine-learning dependencies are bundled.

Install prerequisites (`flatpak`, `flatpak-builder`, the upstream
[`flatpak-pip-generator`](https://github.com/flatpak/flatpak-builder-tools/tree/master/pip)
with `requirements-parser`, plus a configured Flathub remote). Then:

```bash
# From the project root:
bash scripts/build-flatpak.sh
flatpak run io.github.pgm1207.ObjectMonitor
```

The script downloads the official runtime/SDK, generates a dependency module,
and invokes `flatpak-builder`. Its generated
`flatpak/python3-monitor-deps.json` is machine/toolchain-specific and is
intentionally gitignored. It must exist before calling flatpak-builder
directly.

**Build status:** This is a Flatpak *build recipe*, not a verified installable
release. Building the full dependency graph was not possible in the
development environment used for this PR. The PyTorch wheel in the initial
recipe can be very large and may need updating to match the SDK's Python ABI
and wheel availability. Full dependency resolution, install, GUI startup and
video inference still need testing on an x86_64 Flatpak build host. Do not
claim Steam Deck, NVIDIA or AMD GPU compatibility until tested.

### Sandbox permissions

- Wayland and fallback X11 are available for the GTK UI.
- Network access is used for model download and optionally RTSP.
- `--device=dri` permits graphics device access; it does *not* guarantee ROCm.
- Videos directory has read-only access; files selected from elsewhere through
  the GTK file chooser use the file portal.
- Analysis outputs live in persistent per-app user data under
  `~/.var/app/io.github.pgm1207.ObjectMonitor/data/object-monitor/runs/`
  when installed as a Flatpak. Cache snapshots live in the app-specific cache.
- Direct access to `/dev/video*` **is not granted by default**. To opt into
  webcam device access for a trusted local installation, run
  `flatpak override --user --device=all io.github.pgm1207.ObjectMonitor`.
  This significantly expands hardware permissions. Revert with
  `flatpak override --user --reset io.github.pgm1207.ObjectMonitor`.
  A camera-portal-based implementation would be preferable.

The Flatpak currently targets **x86_64** in its dependency generator. Do not
assume aarch64 or Steam Deck support until their artifacts and video APIs
are validated.

### Verify pure-Python components without a GPU or SDK

```bash
python -m compileall -q src
python -m unittest discover -s tests -p 'test_desktop_backend.py' -v
python -m unittest discover -s tests -p 'test_monitor_core.py' -v
python -m src.monitor --help
```

For a real Linux desktop smoke test, start the GUI with a short video,
confirm the preview image updates, use Stop, reopen its output folder, and
inspect `events.jsonl`, `occupancy.csv` and `summary.json`.
