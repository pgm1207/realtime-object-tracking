"""
Integration tests: components working together.

Model-dependent tests skip when weights are absent so the suite runs without
network access or downloads.
"""
import json
import sys
import tempfile
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
import numpy as np  # noqa: E402,F401

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.models import ModelManager, DEFAULT_MODEL_PATHS  # noqa: E402
from src.video_utils import process_video_with_model  # noqa: E402

MODEL_NAME = "yolov8n-seg"


def _weights_or_skip():
    path = DEFAULT_MODEL_PATHS.get(MODEL_NAME)
    if path is None or not Path(path).exists():
        pytest.skip(f"Model weights not present: {path}")
    return path


@pytest.mark.integration
class TestDemoVideoIntegration:
    @pytest.mark.slow
    def test_demo_video_creation(self, sample_video_path):
        model_path = _weights_or_skip()
        manager = ModelManager(MODEL_NAME, model_path=model_path, device="cpu")

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "demo.mp4"
            stats = process_video_with_model(
                sample_video_path, manager, output_path=str(out), add_fps=True
            )
            assert stats["frames_processed"] > 0
            assert out.exists() and out.stat().st_size > 0

            cap = cv2.VideoCapture(str(out))
            try:
                assert cap.isOpened()
                assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) > 0
                assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) > 0
            finally:
                cap.release()


@pytest.mark.integration
class TestEvaluationResultsSchema:
    def test_latest_results_parse(self):
        results_dir = project_root / "inference" / "results"
        if not results_dir.exists():
            pytest.skip("No evaluation results directory found")
        files = sorted(results_dir.glob("*.json"))
        if not files:
            pytest.skip("No evaluation result files found")

        with open(files[-1]) as f:
            results = json.load(f)

        # ``save_results`` writes a mapping of model_type -> metrics; older runs
        # wrapped it as {"models": [...]}. Accept either shape.
        assert isinstance(results, dict)
        if "models" in results:
            assert isinstance(results["models"], list)
        else:
            assert results, "results should not be empty"
