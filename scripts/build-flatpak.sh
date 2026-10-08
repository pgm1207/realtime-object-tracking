#!/usr/bin/env bash
set -euo pipefail

# Build a self-contained GNOME Flatpak. flatpak-builder must not use host pip
# packages: the generator resolves pinned wheel URLs and hashes before build.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

for binary in flatpak flatpak-builder; do
  command -v "$binary" >/dev/null || {
    echo "Missing $binary. Install it using your distribution's package manager." >&2
    exit 2
  }
done

GENERATOR="${FLATPAK_PIP_GENERATOR:-flatpak-pip-generator}"
command -v "$GENERATOR" >/dev/null || {
  echo "Missing flatpak-pip-generator." >&2
  echo "Get it from https://github.com/flatpak/flatpak-builder-tools/tree/master/pip" >&2
  echo "Set FLATPAK_PIP_GENERATOR=/path/to/flatpak-pip-generator if needed." >&2
  exit 2
}

flatpak install --user -y flathub org.gnome.Platform//50 org.gnome.Sdk//50

(
  cd flatpak
  "$GENERATOR" \
    --runtime org.gnome.Sdk//50 \
    --requirements-file requirements-flatpak.txt \
    --output python3-monitor-deps \
    --wheel-arches=x86_64 \
    --prefer-wheels=torch,torchvision,opencv-python,numpy,scipy,pillow,pyyaml,matplotlib,contourpy,kiwisolver,polars,psutil,ultralytics-thop,triton,nvidia-cublas-cu12,nvidia-cuda-runtime-cu12,nvidia-cuda-nvrtc-cu12,nvidia-cuda-cupti-cu12,nvidia-cudnn-cu12,nvidia-cufft-cu12,nvidia-curand-cu12,nvidia-cusolver-cu12,nvidia-cusparse-cu12,nvidia-nccl-cu12,nvidia-nvjitlink-cu12,nvidia-nvtx-cu12
)

flatpak-builder --user --install --force-clean --install-deps-from=flathub \
  "$ROOT/build/flatpak" "$ROOT/flatpak/io.github.pgm1207.ObjectMonitor.json"

echo
echo "Installed. Run with:"
echo "  flatpak run io.github.pgm1207.ObjectMonitor"
echo
echo "Webcam access is not granted by default. Read docs/gnome-flatpak.md."
