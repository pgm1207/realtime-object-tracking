"""
Unit tests for video utilities module.
Tests video processing, loading, and saving functionality.
"""

import pytest
import os

cv2 = pytest.importorskip("cv2")
import numpy as np
from pathlib import Path

from src.video_utils import get_video_properties

@pytest.mark.unit
class TestVideoUtils:
    """Tests for video utility functions"""
    
    def test_get_video_properties(self, sample_video_path):
        """Test getting video properties"""
        cap = cv2.VideoCapture(sample_video_path)
        # get_video_properties returns (fps, total_frames, width, height)
        fps, frame_count, width, height = get_video_properties(cap)
        
        assert width > 0, "Video width should be positive"
        assert height > 0, "Video height should be positive"
        assert fps > 0, "Video FPS should be positive"
        assert frame_count > 0, "Video frame count should be positive"
        
        # Clean up
        cap.release()
