"""
Input validation and error handling utilities for the Computer Vision Project.
Provides functions to validate inputs and handle errors gracefully.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Any, Tuple, Optional


class CVProjectError(Exception):
    """Base exception for Computer Vision Project errors"""
    pass


class ModelNotFoundError(CVProjectError):
    """Exception raised when a model file is not found"""
    pass


class InvalidInputError(CVProjectError):
    """Exception raised when input validation fails"""
    pass


class ProcessingError(CVProjectError):
    """Exception raised when processing fails"""
    pass


def validate_file_path(file_path: Union[str, Path], must_exist: bool = True) -> Path:
    """
    Validate a file path and convert it to a Path object.
    
    Args:
        file_path: Path to validate
        must_exist: Whether the file must exist
        
    Returns:
        Validated Path object
        
    Raises:
        InvalidInputError: If validation fails
    """
    if not file_path:
        raise InvalidInputError("File path cannot be empty")
    
    path = Path(file_path)
    
    if must_exist and not path.exists():
        raise InvalidInputError(f"File not found: {path}")
    
    return path


def validate_model_name(model_name: str, available_models: List[str]) -> str:
    """
    Validate a model name.
    
    Args:
        model_name: Name of the model to validate
        available_models: List of available model names
        
    Returns:
        Validated model name
        
    Raises:
        InvalidInputError: If the model name is invalid
    """
    if not model_name:
        raise InvalidInputError("Model name cannot be empty")
    
    if model_name not in available_models:
        # Give a helpful error message with available models
        raise InvalidInputError(
            f"Model '{model_name}' not found. Available models: {', '.join(available_models)}"
        )
    
    return model_name


def validate_image(image: Union[str, Path, np.ndarray]) -> Union[str, np.ndarray]:
    """
    Validate an image input, which can be a path or a numpy array.
    
    Args:
        image: Image to validate (path or numpy array)
        
    Returns:
        Validated image path or numpy array
        
    Raises:
        InvalidInputError: If validation fails
    """
    if isinstance(image, (str, Path)):
        # If it's a path, validate that it exists and is readable
        path = validate_file_path(image)
        
        # Check if it's a valid image file
        try:
            img = cv2.imread(str(path))
            if img is None:
                raise InvalidInputError(f"Could not read image file: {path}")
            return str(path)
        except Exception as e:
            raise InvalidInputError(f"Error reading image file: {path}. {str(e)}")
    
    elif isinstance(image, np.ndarray):
        # If it's a numpy array, validate shape and dtype
        if len(image.shape) != 3:
            raise InvalidInputError(f"Image must have 3 dimensions, got {len(image.shape)}")
        
        if image.shape[2] != 3:
            raise InvalidInputError(f"Image must have 3 channels, got {image.shape[2]}")
        
        return image
    
    else:
        raise InvalidInputError(f"Expected image path or numpy array, got {type(image).__name__}")


def validate_video(video_path: Union[str, Path]) -> str:
    """
    Validate a video file path.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Validated video file path
        
    Raises:
        InvalidInputError: If validation fails
    """
    path = validate_file_path(video_path)
    
    # Check if it's a valid video file
    cap = None
    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise InvalidInputError(f"Could not open video file: {path}")

        # Get some basic properties to verify it's a valid video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if width <= 0 or height <= 0 or frame_count <= 0:
            raise InvalidInputError(f"Invalid video dimensions or frame count: {path}")

        return str(path)
    except Exception as e:
        if isinstance(e, InvalidInputError):
            raise
        raise InvalidInputError(f"Error validating video file: {path}. {str(e)}")
    finally:
        # Always release the capture, including on the early-raise path above.
        if cap is not None:
            cap.release()


def validate_confidence_threshold(conf_threshold: float) -> float:
    """
    Validate a confidence threshold value.
    
    Args:
        conf_threshold: Confidence threshold to validate
        
    Returns:
        Validated confidence threshold
        
    Raises:
        InvalidInputError: If validation fails
    """
    if not isinstance(conf_threshold, (int, float)):
        raise InvalidInputError(f"Confidence threshold must be a number, got {type(conf_threshold).__name__}")
    
    if conf_threshold < 0 or conf_threshold > 1:
        raise InvalidInputError(f"Confidence threshold must be between 0 and 1, got {conf_threshold}")
    
    return float(conf_threshold)


def validate_iou_threshold(iou_threshold: float) -> float:
    """
    Validate an IoU (Intersection over Union) threshold value.
    
    Args:
        iou_threshold: IoU threshold to validate
        
    Returns:
        Validated IoU threshold
        
    Raises:
        InvalidInputError: If validation fails
    """
    if not isinstance(iou_threshold, (int, float)):
        raise InvalidInputError(f"IoU threshold must be a number, got {type(iou_threshold).__name__}")
    
    if iou_threshold < 0 or iou_threshold > 1:
        raise InvalidInputError(f"IoU threshold must be between 0 and 1, got {iou_threshold}")
    
    return float(iou_threshold)


def validate_device(device: str) -> str:
    """
    Validate a device string (e.g., 'cuda', 'cpu').
    
    Args:
        device: Device to validate
        
    Returns:
        Validated device string
        
    Raises:
        InvalidInputError: If validation fails
    """
    if device not in ('cuda', 'cpu', 'mps'):
        raise InvalidInputError(f"Device must be 'cuda', 'cpu', or 'mps', got '{device}'")

    try:
        import torch
    except ImportError:
        if device != 'cpu':
            print("Warning: PyTorch is not installed; using CPU.")
        return 'cpu'

    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        return 'cpu'

    if device == 'mps':
        mps = getattr(torch.backends, 'mps', None)
        if mps is None or not mps.is_available():
            print("Warning: MPS requested but not available. Falling back to CPU.")
            return 'cpu'

    return device
