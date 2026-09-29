"""
Consolidated Computer Vision Models
This file combines model wrappers (Mask R-CNN and YOLO-Seg) into a single file
for easier maintenance and use.
"""

import cv2
import numpy as np
import os
import time
from pathlib import Path

import requests

# Mapping from YOLO class index (0-79) to official COCO category ID (1-90)
# Based on standard COCO dataset category IDs
YOLO_CLS_INDEX_TO_COCO_ID = {
    0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16,
    15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32,
    28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46,
    41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56, 51: 57, 52: 58, 53: 59,
    54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76,
    67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90
}

# COCO Class Names (80 classes + background)
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# --- Model URLs and Default Paths ---
# Resolve paths relative to this file so the app works from any working
# directory, and allow overrides via environment variables (portability/tests).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("RTO_MODELS_DIR", PROJECT_ROOT / "models" / "pts"))
CONFIGS_DIR = Path(os.environ.get("RTO_CONFIGS_DIR", PROJECT_ROOT / "models" / "configs"))


def ensure_model_dirs() -> None:
    """Create the model/config directories on demand (avoids import-time side effects)."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

# Default paths to model weights/configs
DEFAULT_MODEL_PATHS = {
    # Mask-RCNN model commented out - only using YOLO models now
    # 'mask-rcnn': 'maskrcnn_resnet50_fpn_coco', # Will be re-enabled when needed
    
    # YOLOv8
    'yolov8n-seg': MODELS_DIR / 'yolov8n-seg.pt',
    'yolov8s-seg': MODELS_DIR / 'yolov8s-seg.pt',
    'yolov8m-seg': MODELS_DIR / 'yolov8m-seg.pt',
    'yolov8l-seg': MODELS_DIR / 'yolov8l-seg.pt',
    'yolov8x-seg': MODELS_DIR / 'yolov8x-seg.pt',
    
    # Aliases without the 'v' for backward compatibility
    'yolo8n-seg': MODELS_DIR / 'yolov8n-seg.pt',
    'yolo8s-seg': MODELS_DIR / 'yolov8s-seg.pt',
    'yolo8m-seg': MODELS_DIR / 'yolov8m-seg.pt',
    'yolo8l-seg': MODELS_DIR / 'yolov8l-seg.pt',
    'yolo8x-seg': MODELS_DIR / 'yolov8x-seg.pt',
    
    # YOLOv9
    'yolov9c-seg': MODELS_DIR / 'yolov9c-seg.pt',
    'yolov9e-seg': MODELS_DIR / 'yolov9e-seg.pt',
    'yolo9c-seg': MODELS_DIR / 'yolov9c-seg.pt',
    'yolo9e-seg': MODELS_DIR / 'yolov9e-seg.pt',
    
    # YOLOv11
    'yolo11n-seg': MODELS_DIR / 'yolo11n-seg.pt',
    'yolo11s-seg': MODELS_DIR / 'yolo11s-seg.pt',
    'yolo11m-seg': MODELS_DIR / 'yolo11m-seg.pt',
    'yolo11l-seg': MODELS_DIR / 'yolo11l-seg.pt',
    'yolo11x-seg': MODELS_DIR / 'yolo11x-seg.pt',
    
    # With 'v' for consistency
    'yolov11n-seg': MODELS_DIR / 'yolo11n-seg.pt',
    'yolov11s-seg': MODELS_DIR / 'yolo11s-seg.pt',
    'yolov11m-seg': MODELS_DIR / 'yolo11m-seg.pt',
    'yolov11l-seg': MODELS_DIR / 'yolo11l-seg.pt',
    'yolov11x-seg': MODELS_DIR / 'yolo11x-seg.pt',
}

# URLs for downloading models if needed
BASE_URL_V0 = 'https://github.com/ultralytics/assets/releases/download/v0.0.0/'
BASE_URL_V82 = 'https://github.com/ultralytics/assets/releases/download/v8.2.0/'
BASE_URL_V83 = 'https://github.com/ultralytics/assets/releases/download/v8.3.0/'

MODEL_URLS = {
    # YOLOv8
    'yolo8n-seg': BASE_URL_V0 + 'yolov8n-seg.pt',
    'yolo8s-seg': BASE_URL_V82 + 'yolov8s-seg.pt',
    'yolo8m-seg': BASE_URL_V82 + 'yolov8m-seg.pt',
    'yolo8l-seg': BASE_URL_V82 + 'yolov8l-seg.pt',
    'yolo8x-seg': BASE_URL_V82 + 'yolov8x-seg.pt',
    # YOLOv9
    'yolo9c-seg': BASE_URL_V83 + 'yolov9c-seg.pt',
    'yolo9e-seg': BASE_URL_V83 + 'yolov9e-seg.pt',
    # YOLOv11
    'yolo11n-seg': BASE_URL_V83 + 'yolo11n-seg.pt',
    'yolo11s-seg': BASE_URL_V83 + 'yolo11s-seg.pt',
    'yolo11m-seg': BASE_URL_V83 + 'yolo11m-seg.pt',
    'yolo11l-seg': BASE_URL_V83 + 'yolo11l-seg.pt',
    'yolo11x-seg': BASE_URL_V83 + 'yolo11x-seg.pt',
}

# --- Helper Function for Downloading ---
def _download_model_if_needed(model_path: Path, model_key: str):
    """Downloads the model file if it does not exist."""
    if model_path.exists():
        # Check if file size is greater than zero for placeholder files
        if model_path.stat().st_size > 0:
             print(f"Model found: {model_path}")
             return True
        else:
             print(f"Placeholder found, proceeding with download check: {model_path}")

    model_name = model_path.name
    # Skip download for Mask R-CNN as we'll use torchvision's pretrained model
    if model_name == 'maskrcnn_resnet50_fpn.pt':
        print(f"Note: {model_name} will be loaded from torchvision, not downloaded.")
        # Create an empty file as a placeholder if it doesn't exist
        if not model_path.exists():
            model_path.touch()
        return True # Indicate that it's handled

    # Use the provided model_key to look up the URL
    if model_key not in MODEL_URLS:
        print(f"Error: No download URL defined for model key: {model_key}")
        return False

    url = MODEL_URLS[model_key]
    if not url:
        print(f"Note: No download URL configured for {model_key}. Manual download required if file not present.")
        return model_path.exists() # Return true only if file already exists
        
    print(f"Downloading {model_path.name} ({model_key}) from {url} to {model_path}...")

    try:
        # Standard download method
        response = requests.get(url, stream=True, timeout=120) # Increased timeout
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192 # Increased block size

        with open(model_path, "wb") as f:
            downloaded_size = 0
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    # Basic progress indication
                    progress = int(50 * downloaded_size / total_size) if total_size else 0
                    print(f"\rDownloading: [{'=' * progress}{' ' * (50 - progress)}] {downloaded_size / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB", end='')
        print(f"\nModel {model_path.name} downloaded successfully.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"\nError downloading model {model_path.name}: {e}")
        # Clean up potentially incomplete file
        if model_path.exists():
            try:
                os.remove(model_path)
                print(f"Removed incomplete download: {model_path}")
            except OSError as rm_err:
                print(f"Error removing incomplete file {model_path}: {rm_err}")
        return False
    except Exception as e:
        print(f"\nAn unexpected error occurred during download: {e}")
        # Clean up potentially incomplete file if it exists
        if model_path.exists():
             try:
                 os.remove(model_path)
                 print(f"Removed potentially corrupt file: {model_path}")
             except OSError as rm_err:
                 print(f"Error removing file {model_path}: {rm_err}")
        return False


# --- YOLO Wrapper ---
class YOLOWrapper:
    """Wrapper for YOLO model inference using the Ultralytics library."""

    def __init__(self, model_path_or_name, conf_threshold=0.25, iou_threshold=0.45,
                 device=None, imgsz=None, half=False):
        """
        Initializes the YOLO model wrapper. Handles both detection and segmentation models.
        Relies on the ultralytics YOLO() constructor for loading and potential auto-download.

        Args:
            model_path_or_name (str or Path): Path or name of the YOLO weights file (.pt).
            conf_threshold (float): Confidence threshold for detections (default: 0.25).
            iou_threshold (float): IoU threshold for NMS (default: 0.45).
            device (str, optional): Force a device ("cpu", "cuda", "cuda:0", "mps").
                Auto-detected when None.
            imgsz (int, optional): Inference image size (e.g. 640) for a speed/accuracy trade-off.
            half (bool): Use FP16 inference where supported (faster on CUDA).
        """
        self.input_model_identifier = str(model_path_or_name)
        self.model = None
        self.model_path = None
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.half = half

        # Determine device (explicit override wins; otherwise CUDA if available).
        try:
            import torch
            if device:
                dev = str(device)
                if dev.startswith("cuda") and not torch.cuda.is_available():
                    print(f"Requested device '{dev}' but CUDA is unavailable; using CPU.")
                    dev = "cpu"
                self.device = torch.device(dev)
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError:
            self.device = device or "cpu"
        print(f"Using device: {self.device}")

        self.model_type_name = Path(self.input_model_identifier).stem
        print(f"Attempting to load YOLO model using identifier: {self.input_model_identifier}")
        print(f"Using conf_threshold={self.conf_threshold}, iou_threshold={self.iou_threshold}")

        try:
            from ultralytics import YOLO  # Imported lazily so the rest of the app works without it.

            self.model = YOLO(self.input_model_identifier)
            try:
                self.model.to(self.device)
            except Exception as move_err:
                print(f"Warning: could not move model to {self.device}: {move_err}")

            ckpt = getattr(self.model, "ckpt_path", None)
            if ckpt:
                self.model_path = Path(ckpt)
                print(f"YOLO model ({self.model_type_name}) loaded successfully from: {self.model_path}")
            else:
                existing = Path(self.input_model_identifier)
                self.model_path = existing if existing.exists() else None
                print(f"YOLO model ({self.model_type_name}) loaded successfully.")
        except ImportError:
            print("Error: 'ultralytics' library not found. Please install it (`pip install ultralytics`)")
            self.model = None
        except FileNotFoundError as fnf_error:
            print(f"Error loading YOLO model: {fnf_error}")
            print(f"The identifier '{self.input_model_identifier}' was not found locally, and automatic download failed.")
            self.model = None
        except Exception as e:
            print(f"An unexpected error occurred loading YOLO model {self.input_model_identifier}: {e}")
            self.model = None

    def predict(self, frame: np.ndarray, return_masks: bool = True):
        """
        Performs object detection and segmentation on a single frame.

        Args:
            frame (np.ndarray): Input video frame (BGR format).
            return_masks (bool): When False, skip per-instance mask extraction, which
                is significantly faster for video output where masks are not needed.

        Returns:
            tuple: (detections, segmentations, annotated_frame). Each detection is a
                dict with both ``box``/``bbox`` and ``score``/``confidence`` keys so
                downstream code can use either naming.
        """
        if self.model is None:
            print(f"Warning: YOLO model ({self.model_type_name}) not loaded. Returning empty results.")
            annotated_frame = frame.copy()
            cv2.putText(annotated_frame, "YOLO Model Not Loaded", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return [], [], annotated_frame

        try:
            start_time = time.time()

            predict_kwargs = dict(verbose=False, conf=self.conf_threshold, iou=self.iou_threshold)
            if self.imgsz:
                predict_kwargs["imgsz"] = self.imgsz
            if self.half:
                predict_kwargs["half"] = True
            results = self.model(frame, **predict_kwargs)

            detections = []
            segmentations = []
            annotated_frame = results[0].plot()

            for result in results:
                boxes = result.boxes
                names = result.names or {}

                for i in range(len(boxes)):
                    box = boxes[i]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    yolo_class_id = int(box.cls[0])
                    # Map YOLO's 0-indexed class ID to the official COCO category ID.
                    coco_category_id = YOLO_CLS_INDEX_TO_COCO_ID.get(yolo_class_id)
                    if coco_category_id is None:
                        continue

                    score = float(box.conf[0])
                    class_name = names.get(yolo_class_id, f"ID:{yolo_class_id}")

                    detections.append({
                        "box": [x1, y1, x2, y2],
                        "bbox": [x1, y1, x2, y2],      # alias used by some consumers
                        "confidence": score,           # alias used by some consumers
                        "class_id": coco_category_id,
                        "class_name": class_name,
                        "score": score,
                    })

                if return_masks and result.masks is not None:
                    try:
                        img_h, img_w = frame.shape[:2]
                        for mask_tensor in result.masks.data:
                            mask_np = mask_tensor.cpu().numpy()
                            binary_mask = (mask_np > 0.5).astype(np.uint8) * 255
                            if binary_mask.shape[:2] != (img_h, img_w):
                                binary_mask = cv2.resize(
                                    binary_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST
                                )
                            segmentations.append(binary_mask)
                    except Exception as mask_error:
                        print(f"Error processing masks: {mask_error}")

            total_time = time.time() - start_time
            fps = 1.0 / total_time if total_time > 0 else 0.0
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            return detections, segmentations, annotated_frame

        except Exception as e:
            print(f"Error during YOLO ({self.model_type_name}) prediction: {e}")
            import traceback
            traceback.print_exc()
            annotated_frame = frame.copy()
            cv2.putText(annotated_frame, "YOLO Prediction Error", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return [], [], annotated_frame


# --- Mask R-CNN Wrapper ---
class MaskRCNNWrapper:
    """Wrapper for Mask R-CNN model inference using torchvision."""
    
    def __init__(self, model_path=None):
        """
        Initializes the Mask R-CNN model wrapper.
        
        Args:
            model_path (str or Path, optional): Path is ignored as we use pretrained torchvision model.
        """
        print("Initializing MaskRCNNWrapper")

        try:
            import torch
            import torchvision
        except ImportError as e:
            print(f"Mask R-CNN unavailable (torch/torchvision not installed): {e}")
            self.model = None
            self.class_names = COCO_CLASSES
            return

        # Determine device (cuda or cpu)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        try:
            print("Loading Mask R-CNN model from torchvision...")
            # Load the pretrained model - use weights='DEFAULT' for newer torchvision, use pretrained=True for older versions
            try:
                self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')
            except TypeError:
                # Fall back to older torchvision API
                self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
            
            # Set model to evaluation mode
            self.model.eval()
            # Move model to the correct device
            self.model.to(self.device)
            # Store COCO class names
            self.class_names = COCO_CLASSES
            print("Mask R-CNN model loaded successfully.")
        except Exception as e:
            print(f"Error loading Mask R-CNN model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None

    def predict(self, frame: np.ndarray):
        """
        Performs instance segmentation on a single frame.
        
        Args:
            frame (np.ndarray): Input video frame (BGR format).
            
        Returns:
            tuple: A tuple containing:
                - detections (list): List of detected objects (dict with 'box', 'class_id', 'class_name', 'score').
                - segmentations (list): List of segmentation masks (binary masks for each detection).
                - annotated_frame (np.ndarray): Frame with visualizations drawn.
        """
        if self.model is None:
            print("Warning: Mask R-CNN model not loaded. Returning empty results.")
            annotated_frame = frame.copy()
            cv2.putText(annotated_frame, "Mask R-CNN Model Not Loaded", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return [], [], annotated_frame
            
        try:
            import torch
            from torchvision.transforms import functional as F

            # Start timing
            start_time = time.time()

            # Preprocess the input frame
            # Convert BGR (OpenCV) to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to a PyTorch tensor
            img_tensor = F.to_tensor(rgb_frame)
            
            # Move to the correct device and add batch dimension
            img_tensor = img_tensor.to(self.device).unsqueeze(0)
            
            preprocess_time = time.time() - start_time
            inference_start = time.time()
            
            # Perform inference
            with torch.no_grad():
                predictions = self.model(img_tensor)
                
            inference_time = time.time() - inference_start
            postprocess_start = time.time()
            
            # Process predictions
            boxes = predictions[0]['boxes'].cpu().numpy()
            scores = predictions[0]['scores'].cpu().numpy()
            labels = predictions[0]['labels'].cpu().numpy()
            masks = predictions[0]['masks'].cpu().numpy()
            
            # Filter predictions based on confidence threshold
            confidence_threshold = 0.5
            confident_indices = scores >= confidence_threshold
            
            # Format detections and segmentations
            detections = []
            segmentations = []
            annotated_frame = frame.copy()
            
            # Create a semi-transparent overlay for masks
            mask_overlay = np.zeros_like(frame)
            
            # Define colors for different classes (random but consistent)
            np.random.seed(42)  # For consistent colors across runs
            colors = [(int(np.random.randint(0, 256)), 
                       int(np.random.randint(0, 256)), 
                       int(np.random.randint(0, 256))) for _ in range(len(self.class_names))]
            
            for i, box in enumerate(boxes):
                if scores[i] >= confidence_threshold:
                    # Extract coordinates
                    x1, y1, x2, y2 = box.astype(int)
                    class_id = int(labels[i].item())
                    score = float(scores[i].item())
                    
                    # Get the mask for this detection (threshold at 0.5)
                    mask = masks[i, 0] > 0.5  # First channel of mask, threshold at 0.5
                    segmentations.append(mask.astype(np.uint8) * 255)  # Convert to uint8 for consistency
                    
                    # Get class name
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Unknown {class_id}"
                    
                    # Add to detections list
                    detections.append({
                        "box": [x1, y1, x2, y2],
                        "class_id": class_id,
                        "class_name": class_name,
                        "score": score
                    })
                    
                    # Use consistent color based on class ID
                    color = colors[class_id % len(colors)]
                    
                    # Visualize instance mask - add color where mask is True
                    colored_mask = np.zeros_like(frame)
                    mask_resized = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]))
                    colored_mask[mask_resized > 0] = color
                    
                    # Add mask overlay to frame
                    cv2.addWeighted(annotated_frame, 1.0, colored_mask, 0.5, 0, annotated_frame)
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label with class name and score
                    label_text = f"{class_name}: {score:.2f}"
                    cv2.rectangle(annotated_frame, (x1, y1-25), (x1 + len(label_text)*9, y1), color, -1)
                    cv2.putText(annotated_frame, label_text, (x1, y1-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            postprocess_time = time.time() - postprocess_start
            total_time = time.time() - start_time
            
            # Display timing information on frame
            cv2.putText(annotated_frame, f"FPS: {1/total_time:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Add detailed timing report to console
            print(f"Mask R-CNN timing: Total={total_time:.3f}s (FPS: {1/total_time:.1f}) | " 
                  f"Preprocess={preprocess_time:.3f}s | Inference={inference_time:.3f}s | "
                  f"Postprocess={postprocess_time:.3f}s | Detections={len(detections)} | Masks={len(segmentations)}")
            
            return detections, segmentations, annotated_frame
            
        except Exception as e:
            print(f"Error during Mask R-CNN prediction: {e}")
            import traceback
            traceback.print_exc()
            annotated_frame = frame.copy()
            cv2.putText(annotated_frame, "Mask R-CNN Prediction Error", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return [], [], annotated_frame


# --- Model Manager ---
class ModelManager:
    """Manages loading and accessing different computer vision models."""

    def __init__(self, model_type, model_path=None, config_path=None,
                 conf_threshold=0.25, iou_threshold=0.45,
                 device=None, imgsz=None, half=False):
        """
        Initializes the appropriate model wrapper based on the model type.

        Args:
            model_type (str): Model type key (e.g. 'yolov8n-seg').
            model_path (str or Path, optional): Model weights path. Defaults to the
                registered path for the type.
            config_path (str, optional): Reserved for future use.
            conf_threshold (float): Confidence threshold (default: 0.25).
            iou_threshold (float): NMS IoU threshold (default: 0.45).
            device (str, optional): Force an inference device ("cpu", "cuda", "mps").
            imgsz (int, optional): Inference image size.
            half (bool): Use FP16 where supported.
        """
        self.model_type = model_type.lower()
        self.config_path = config_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.half = half

        print(f"ModelManager: Using conf_threshold={self.conf_threshold}, iou_threshold={self.iou_threshold}")

        # Determine model path: Use provided path or default for the type
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATHS.get(self.model_type)
        if not self.model_path:
             raise ValueError(f"Model path not specified and no default path found for type: {self.model_type}")

        self.device = self._get_device(device)
        print(f"Using device: {self.device}")
        ensure_model_dirs()
        self.model_wrapper = self._initialize_model()

        if self.model_wrapper is None or not hasattr(self.model_wrapper, 'model') or self.model_wrapper.model is None:
             print(f"Warning: ModelManager failed to initialize a valid model wrapper for type '{self.model_type}' with path '{self.model_path}'.")
             # Optionally raise an error here if initialization must succeed
             # raise RuntimeError("Model wrapper initialization failed.")

    def _get_device(self, requested=None):
        """Determine the device to use: explicit argument, env override, else auto-detect."""
        # Explicit argument wins.
        if requested:
            dev = str(requested).lower()
            if dev.startswith("cuda"):
                try:
                    import torch
                    if not torch.cuda.is_available():
                        print("Requested CUDA but it is unavailable; using CPU.")
                        return "cpu"
                except ImportError:
                    return "cpu"
            print(f"Using requested device: {dev}")
            return dev

        # Environment overrides: RTO_DEVICE / CV_FORCE_DEVICE, or legacy FORCE_CPU_INFERENCE.
        env_device = os.environ.get("RTO_DEVICE") or os.environ.get("CV_FORCE_DEVICE")
        if env_device and env_device.lower() != "auto":
            print(f"Using device from environment: {env_device}")
            return env_device

        force_cpu = os.environ.get("FORCE_CPU_INFERENCE", "").lower() in ("1", "true", "yes", "cpu")
        try:
            import torch
            if torch.cuda.is_available() and not force_cpu:
                try:
                    torch.zeros(1).cuda()
                    print("CUDA is available and working. Using GPU for inference.")
                    return "cuda"
                except Exception as e:
                    print(f"CUDA reported available but failed ({e}); falling back to CPU.")
                    return "cpu"
            if force_cpu:
                print("Forcing CPU usage based on environment variable.")
            else:
                print("Using CPU for model inference (no CUDA GPU detected).")
            return "cpu"
        except ImportError:
            print("PyTorch not installed; using CPU device.")
            return "cpu"

    def _initialize_model(self):
        """Initialize the appropriate model based on type"""
        print(f"Initializing model manager for type: {self.model_type}")
        try:
            # Commenting out Mask R-CNN for now since we're only using YOLO models
            # Keep the code for future usability
            if self.model_type == 'mask-rcnn':
                # Comment out the Mask R-CNN initialization
                print("Mask R-CNN model is currently disabled. Using YOLO models only.")
                return None
                # Return MaskRCNNWrapper() # This line is commented out - re-enable when needed
                
            # More flexible handling of YOLO model types
            elif 'yolo' in self.model_type.lower():
                # Check if this is a segmentation model
                is_seg_model = 'seg' in self.model_type.lower()
                
                print(f"Loading YOLO model from: {self.model_path}")
                print(f"Model type: {self.model_type} (Segmentation: {is_seg_model})")
                
                # Create the wrapper with proper thresholds
                yolo_wrapper = YOLOWrapper(
                    self.model_path,
                    conf_threshold=self.conf_threshold,
                    iou_threshold=self.iou_threshold,
                    device=self.device,
                    imgsz=self.imgsz,
                    half=self.half,
                )
                
                # Store whether this is a segmentation model for reference
                yolo_wrapper.is_seg_model = is_seg_model
                return yolo_wrapper
                
            else:
                print(f"Error: Unsupported model type: {self.model_type}")
                return None
                
        except Exception as e:
             print(f"Error during model wrapper initialization for {self.model_type}: {e}")
             import traceback
             traceback.print_exc()
             return None

    def predict(self, frame, input_points=None, input_labels=None, return_masks=True):
        """
        Performs prediction using the selected model.

        Args:
            frame (np.ndarray): Input video frame.
            input_points (Optional[np.ndarray]): Not used by current models.
            input_labels (Optional[np.ndarray]): Not used by current models.
            return_masks (bool): Whether to compute segmentation masks (YOLO-Seg only).

        Returns:
            tuple: (detections, segmentations, annotated_frame). Returns
                   ([], [], frame) if the model wrapper is not valid or prediction fails.
        """
        if self.model_wrapper is None or not hasattr(self.model_wrapper, 'predict'):
            print(f"Error: Model wrapper for {self.model_type} is not initialized or lacks predict method.")
            annotated_frame = frame.copy()
            # Add error text to frame
            cv2.putText(annotated_frame, f"{self.model_type.upper()} Model Error", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return [], [], annotated_frame # Return empty results and annotated frame

        try:
            # All current models (YOLO-Seg) use the same predict signature.
            # Mask R-CNN doesn't accept return_masks, so fall back gracefully.
            try:
                results = self.model_wrapper.predict(frame, return_masks=return_masks)
            except TypeError:
                results = self.model_wrapper.predict(frame)
            
            # Ensure we have a proper 3-tuple result
            if not results or len(results) != 3:
                print(f"Warning: Invalid prediction results from {self.model_type} model")
                annotated_frame = frame.copy()
                cv2.putText(annotated_frame, f"{self.model_type.upper()} Invalid Results", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                return [], [], annotated_frame
                
            return results
            
        except Exception as e:
            print(f"Error during prediction with {self.model_type}: {e}")
            import traceback
            traceback.print_exc()
            annotated_frame = frame.copy()
            # Add error text to frame
            cv2.putText(annotated_frame, f"{self.model_type.upper()} Predict Error", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # Return empty lists and the annotated error frame
            return [], [], annotated_frame

# Define COLORS for visualization if not already defined globally
# Generate random colors for visualization
np.random.seed(42) # for reproducibility
COLORS = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(COCO_CLASSES))] # Use COCO classes length for consistency