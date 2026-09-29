"""
Minimal end-to-end pipeline test.

Uses the real ``ModelManager`` / ``process_video_with_model`` APIs and skips
when model weights are absent (to avoid downloads in CI).
"""
import sys
import tempfile
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
import numpy as np  # noqa: E402

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.models import ModelManager, DEFAULT_MODEL_PATHS  # noqa: E402
from src.validation import (  # noqa: E402
    InvalidInputError,
    validate_confidence_threshold,
    validate_iou_threshold,
    validate_device,
)

MODEL_NAME = "yolov8n-seg"


def _weights_or_skip():
    path = DEFAULT_MODEL_PATHS.get(MODEL_NAME)
    if path is None or not Path(path).exists():
        pytest.skip(f"Model weights not present: {path}")
    return path


@pytest.mark.integration
class TestPipeline:
    @pytest.mark.slow
    def test_inference_on_image(self, sample_image_path):
        model_path = _weights_or_skip()
        manager = ModelManager(MODEL_NAME, model_path=model_path, device="cpu")

        frame = cv2.imread(sample_image_path)
        assert frame is not None, "sample image should load"

        detections, segmentations, annotated = manager.predict(frame)
        assert annotated.shape == frame.shape
        assert isinstance(detections, list)

    @pytest.mark.slow
    def test_process_video(self, sample_video_path):
        from src.video_utils import process_video_with_model

        model_path = _weights_or_skip()
        manager = ModelManager(MODEL_NAME, model_path=model_path, device="cpu")

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.mp4"
            stats = process_video_with_model(
                sample_video_path, manager, output_path=str(out), add_fps=False
            )
            assert stats["frames_processed"] > 0
            assert out.exists() and out.stat().st_size > 0

    @pytest.mark.unit
    def test_invalid_thresholds_raise(self):
        with pytest.raises(InvalidInputError):
            validate_confidence_threshold(-0.1)
        with pytest.raises(InvalidInputError):
            validate_iou_threshold(1.1)
        with pytest.raises(InvalidInputError):
            validate_device("invalid_device")
