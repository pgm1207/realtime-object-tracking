"""
Test utilities for the Computer Vision Project.
Contains common functions, fixtures, and mock data for tests.
"""

import os
import sys
import cv2
import numpy as np
import torch
import pytest
from pathlib import Path

# Add the src directory to the path so tests can import modules
project_root = Path(__file__).parent.parent
src_dir = project_root / "src"
sys.path.insert(0, str(project_root))

# Create test data directory if it doesn't exist
TEST_DATA_DIR = project_root / "tests" / "test_data"
TEST_DATA_DIR.mkdir(exist_ok=True, parents=True)

# Create a sample image and sample video for testing
def create_test_image(height=640, width=640, color=(255, 0, 0)):
    """Create a simple test image (blue by default)"""
    return np.full((height, width, 3), color, dtype=np.uint8)

def create_test_image_with_shapes(height=640, width=640):
    """Create a test image with basic shapes for detection testing"""
    # Create blank image with white background
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Draw some shapes that resemble objects
    # Circle (could be detected as sports ball)
    cv2.circle(img, (width//4, height//4), 50, (0, 0, 255), -1)
    
    # Rectangle (could be detected as tv/monitor)
    cv2.rectangle(img, (width//2, height//4), (3*width//4, height//2), (0, 255, 0), -1)
    
    # Triangle (might not be detected as a specific class)
    pts = np.array([[width//4, 3*height//4], [width//2, 2*height//3], [3*width//4, 3*height//4]], np.int32)
    cv2.fillPoly(img, [pts], (255, 0, 0))
    
    return img

def create_test_video(output_path, num_frames=30, fps=15, height=640, width=640):
    """Create a test video with basic shapes moving"""
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    # Use a codec that matches the container (XVID is AVI-only and produces
    # unreadable .mp4 files on many systems).
    fourcc_str = "mp4v" if output_path.suffix.lower() in (".mp4", ".mov", ".m4v") else "XVID"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*fourcc_str),
        fps,
        (width, height)
    )
    
    for i in range(num_frames):
        # Create image with shapes that move slightly each frame
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Moving circle
        x_circle = width//4 + int(i * width/(4*num_frames))
        cv2.circle(img, (x_circle, height//4), 50, (0, 0, 255), -1)
        
        # Moving rectangle
        x_rect_start = width//2 - int(i * width/(8*num_frames))
        x_rect_end = 3*width//4 - int(i * width/(8*num_frames))
        cv2.rectangle(img, (x_rect_start, height//4), (x_rect_end, height//2), (0, 255, 0), -1)
        
        # Moving triangle
        x_offset = int(i * width/(4*num_frames))
        pts = np.array([
            [width//4 + x_offset, 3*height//4], 
            [width//2 + x_offset, 2*height//3], 
            [3*width//4 + x_offset, 3*height//4]
        ], np.int32)
        cv2.fillPoly(img, [pts], (255, 0, 0))
        
        writer.write(img)
    
    writer.release()
    return str(output_path)

# Pytest fixtures
@pytest.fixture
def sample_image_path():
    """Returns the path to a sample test image"""
    img_path = TEST_DATA_DIR / "test_image.jpg"
    if not img_path.exists():
        img = create_test_image_with_shapes()
        cv2.imwrite(str(img_path), img)
    return str(img_path)

@pytest.fixture
def sample_video_path():
    """Returns the path to a sample test video"""
    video_path = TEST_DATA_DIR / "test_video.mp4"
    if not video_path.exists():
        create_test_video(video_path)
    return str(video_path)

@pytest.fixture
def mock_detection_results():
    """Returns mock detection results in the format returned by the models"""
    # Create a mock detection result that mimics the format of model outputs
    # This will need to be customized based on your model's output format
    mock_results = {
        "boxes": torch.tensor([[100, 100, 200, 200], [300, 300, 400, 400]]),
        "scores": torch.tensor([0.9, 0.8]),
        "labels": torch.tensor([1, 2]),  # person, bicycle
        "masks": torch.ones((2, 640, 640)),  # Mock segmentation masks
        "time": 0.05  # 50ms inference time
    }
    return mock_results

# Check if CUDA is available for GPU tests
def is_cuda_available():
    """Check if CUDA is available"""
    return torch.cuda.is_available()

# Skip GPU tests if CUDA is not available
def skip_if_no_gpu():
    """Skip a test if CUDA is not available"""
    if not is_cuda_available():
        pytest.skip("GPU not available")
