"""
Unit tests for the models module.

These exercise the real, public surface of ``src.models`` and do not require
downloading model weights. Model-dependent tests skip when weights are absent.
"""
import os
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.models import (  # noqa: E402
    ModelManager,
    DEFAULT_MODEL_PATHS,
    YOLO_CLS_INDEX_TO_COCO_ID,
    COCO_CLASSES,
)


@pytest.mark.unit
class TestModelRegistry:
    def test_default_paths_are_defined(self):
        assert isinstance(DEFAULT_MODEL_PATHS, dict)
        assert len(DEFAULT_MODEL_PATHS) > 0
        # Every registered path is a Path object with a .pt file.
        for key, path in DEFAULT_MODEL_PATHS.items():
            assert isinstance(path, Path), f"{key} should map to a Path"
            assert path.suffix == ".pt", f"{key} should point to a .pt file"

    def test_contains_expected_segmentation_models(self):
        for key in ["yolov8n-seg", "yolov8x-seg", "yolo11m-seg"]:
            assert key in DEFAULT_MODEL_PATHS

    def test_yolo_to_coco_mapping(self):
        # 80 YOLO classes mapped to the official COCO category IDs.
        assert len(YOLO_CLS_INDEX_TO_COCO_ID) == 80
        assert YOLO_CLS_INDEX_TO_COCO_ID[0] == 1          # person
        assert YOLO_CLS_INDEX_TO_COCO_ID[79] == 90        # toothbrush
        assert set(YOLO_CLS_INDEX_TO_COCO_ID).issuperset(range(80))
        assert all(1 <= v <= 90 for v in YOLO_CLS_INDEX_TO_COCO_ID.values())

    def test_coco_classes_length(self):
        # 80 classes + background.
        assert len(COCO_CLASSES) == 91
        assert COCO_CLASSES[1] == "person"


@pytest.mark.unit
class TestModelManager:
    def test_missing_type_raises(self):
        # model_type is required.
        with pytest.raises(TypeError):
            ModelManager()  # type: ignore[call-arg]

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            ModelManager("definitely-not-a-model")

    def test_device_override_is_respected(self):
        # CPU is always a valid explicit request.
        manager = ModelManager("yolov8n-seg", device="cpu")
        assert str(manager.device).startswith("cpu")

    @pytest.mark.slow
    def test_predict_returns_expected_tuple(self):
        model_path = DEFAULT_MODEL_PATHS["yolov8n-seg"]
        if not model_path.exists():
            pytest.skip(f"Model weights not present: {model_path}")

        manager = ModelManager("yolov8n-seg", model_path=model_path, device="cpu")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections, segmentations, annotated = manager.predict(frame)

        assert isinstance(detections, list)
        assert isinstance(segmentations, list)
        assert annotated.shape == frame.shape
        # Detection dicts expose both naming conventions.
        for det in detections:
            assert "box" in det and "bbox" in det
            assert "score" in det and "confidence" in det
