"""
Model Evaluation Script for Computer Vision Project

This script evaluates two state-of-the-art object detection and segmentation models:
1. Mask R-CNN (TorchVision)
2. YOLO-Seg (YOLOv8 with segmentation by Ultralytics)

The evaluation measures:
- Validity: Precision, Recall, mAP
- Reliability: Consistency across different runs
- Objectivity: Performance on standard COCO validation dataset
- Efficiency: FPS, latency, and resource usage
"""

import os
import sys
import time
import json
import torch
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import tempfile
import argparse
import traceback
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

# Add the src directory to sys.path to allow for absolute imports from src
# This is useful when running scripts directly from the src directory or subdirectories
# And ensures that modules like print_model_summary can be found
current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent.parent # Assuming evaluate_models.py is in src, so two .parent calls to get to project root
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import validation and error handling
from validation import (
    validate_file_path, validate_model_name, validate_confidence_threshold, 
    validate_iou_threshold, validate_device, InvalidInputError
)
from error_handling import (
    log_info, log_warning, log_error, log_exception, 
    handle_exceptions, with_error_handling, print_friendly_error
)

# Import model manager
try:
    from models import ModelManager, DEFAULT_MODEL_PATHS
except ImportError:
    # If running directly from the src directory
    from models import ModelManager, DEFAULT_MODEL_PATHS
    try:
        from validation import (
            validate_file_path, validate_model_name, validate_confidence_threshold, 
            validate_iou_threshold, validate_device, InvalidInputError
        )
        from error_handling import (
            log_info, log_warning, log_error, log_exception, 
            handle_exceptions, with_error_handling, print_friendly_error
        )
    except ImportError:
        pass  # Will use functions without error handling if not available

# Configure paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
COCO_VAL_DIR = PROJECT_ROOT / "data_sets" / "image_data" / "coco" / "val2017"
COCO_ANNOT_FILE = PROJECT_ROOT / "data_sets" / "image_data" / "coco" / "annotations" / "instances_val2017.json"
RESULTS_DIR = PROJECT_ROOT / "inference" / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# Configure models for evaluation
# Dynamically build the list from DEFAULT_MODEL_PATHS
MODELS_TO_EVALUATE = []
# Add all models from DEFAULT_MODEL_PATHS
for model_key, model_path in DEFAULT_MODEL_PATHS.items():
    MODELS_TO_EVALUATE.append({"type": model_key, "path": model_path, "config": None})

# COCO val2017 has 5000 images total
COCO_VAL_TOTAL_IMAGES = 5000

class ModelEvaluator:
    """Class to evaluate model performance on COCO validation set"""
    
    def __init__(self, models_list=None):
        """
        Initialize the evaluator with models to test
        
        Args:
            models_list (list): List of model configurations to evaluate
        """
        self.models_list = models_list or MODELS_TO_EVALUATE
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}
        self.coco_gt = None
        self.coco_gt_api = None
        
        # Load COCO annotations if file exists
        if COCO_ANNOT_FILE.exists():
            with open(COCO_ANNOT_FILE, 'r') as f:
                self.coco_gt = json.load(f)
            log_info(f"Loaded COCO annotations from {COCO_ANNOT_FILE}")
            
            # Initialize COCO API for annotations
            try:
                self.coco_gt_api = COCO(COCO_ANNOT_FILE)
                log_info(f"Successfully initialized COCO API with {len(self.coco_gt_api.getImgIds())} images")
                log_info(f"Found {len(self.coco_gt_api.getCatIds())} categories in annotations")
            except Exception as e:
                log_warning(f"Warning: Failed to initialize COCO API: {e}")
        else:
            log_warning(f"Warning: COCO annotations file not found at {COCO_ANNOT_FILE}")
    
    def load_coco_val_images(self, max_images=100):
        """
        Load COCO validation images for evaluation
        
        Args:
            max_images (int): Maximum number of images to load
            
        Returns:
            list: List of (image, image_id) tuples
        """
        # Check if we have the COCO dataset
        if not COCO_VAL_DIR.exists() or not COCO_ANNOT_FILE.exists():
            log_info(f"COCO dataset not found. Downloading {max_images} images...")
            try:
                # Import and use DatasetManager to download the dataset
                from data_sets.dataset_manager import DatasetManager
                dataset_manager = DatasetManager()
                dataset_manager.setup_coco(subset_size=max_images)
                log_info("Dataset download complete. Now loading images...")
            except Exception as e:
                log_error(f"Error downloading dataset: {e}")
                import traceback
                traceback.print_exc()
                return []
                
        if not COCO_VAL_DIR.exists():
            log_error(f"Error: COCO validation directory not found at {COCO_VAL_DIR}")
            return []
        
        # If COCO API is initialized, use it to get image IDs
        if self.coco_gt_api:
            log_info("Using COCO API to get image IDs")
            image_ids = sorted(self.coco_gt_api.getImgIds())[:max_images]
            log_info(f"Selected {len(image_ids)} images from annotation file")
            
            images = []
            for img_id in tqdm(image_ids, desc="Loading images"):
                img_info = self.coco_gt_api.loadImgs(img_id)[0]
                filename = img_info['file_name']
                img_path = COCO_VAL_DIR / filename
                
                if not img_path.exists():
                    log_warning(f"Warning: Image file not found: {img_path}")
                    continue
                
                # Load image
                img = cv2.imread(str(img_path))
                if img is None:
                    log_warning(f"Warning: Could not load image {img_path}")
                    continue
                    
                images.append((img, img_id))
        else:
            # Fallback to directory search if COCO API not available
            log_info("Falling back to directory search for images")
            image_files = list(COCO_VAL_DIR.glob("*.jpg"))
            if not image_files:
                log_error(f"Error: No images found in {COCO_VAL_DIR}")
                return []
            
            # Limit number of images (useful for quick testing)
            image_files = image_files[:max_images]
            log_info(f"Loading {len(image_files)} images from directory")
            
            images = []
            for img_file in tqdm(image_files, desc="Loading images"):
                # Extract image ID from filename (e.g., 000000000139.jpg -> 139)
                img_id = int(img_file.stem)
                
                # Load image
                img = cv2.imread(str(img_file))
                if img is None:
                    log_warning(f"Warning: Could not load image {img_file}")
                    continue
                    
                images.append((img, img_id))
        
        log_info(f"Loaded {len(images)} images")
        return images
    
    def evaluate_model(self, model_config, images, save_visualizations=True, progress_callback=None):
        """
        Evaluate a single model on the validation set
        
        Args:
            model_config (dict): Model configuration with type, path, and config
            images (list): List of (image, image_id) tuples
            save_visualizations (bool): Whether to save visualization images
            progress_callback (callable, optional): Function to report progress. 
                                                   Called with (model_type, current_image_index, total_images).
            
        Returns:
            dict: Performance metrics
        """
        model_type = model_config["type"]
        model_path = model_config["path"]
        config_path = model_config["config"]
        
        # Get confidence and IoU thresholds from model_config or use defaults
        conf_threshold = model_config.get("conf_threshold", 0.25)
        iou_threshold = model_config.get("iou_threshold", 0.45)
        
        log_info(f"\nEvaluating {model_type.upper()} model with conf_threshold={conf_threshold}, iou_threshold={iou_threshold}...")
        
        # Initialize the model with proper thresholds
        try:
            model_manager = ModelManager(model_type, model_path, config_path, 
                                         conf_threshold=conf_threshold, 
                                         iou_threshold=iou_threshold)
            if model_manager.model_wrapper is None:
                log_error(f"Error: Failed to initialize {model_type} model")
                return None
        except Exception as e:
            log_exception(e, context=f"Error initializing {model_type} model")
            return None
        
        # Initialize metrics
        metrics = {
            "model_type": model_type,
            "total_images": len(images),
            "successful_inferences": 0,
            "total_detections": 0,
            "inference_times": [],
            "detection_counts": {}, # Counts per class
            "detection_counts_per_image": [], # Counts per image
            "classes_detected": set(),
            "failures": 0,
            "shape_metrics": {  # Added shape metrics storage
                "compactness": [],
                "convexity": [],
                "circularity": [],
                "by_size": {
                    "small": {"compactness": [], "convexity": [], "circularity": []},
                    "medium": {"compactness": [], "convexity": [], "circularity": []},
                    "large": {"compactness": [], "convexity": [], "circularity": []}
                }
            }
        }
        
        # Create directory for model visualizations
        vis_dir = RESULTS_DIR / f"{model_type}_visualizations"
        if save_visualizations:
            vis_dir.mkdir(exist_ok=True)
        
        # For COCO evaluation, we need to store predictions in COCO format
        coco_bbox_predictions = []  # For bounding box predictions
        coco_segm_predictions = []  # For segmentation mask predictions
        
        # Process each image
        for idx, (image, image_id) in enumerate(tqdm(images, desc=f"Evaluating {model_type}")):
            # Report progress via callback
            if progress_callback:
                try:
                    progress_callback(model_type, idx + 1, len(images))
                except Exception as cb_err:
                    log_warning(f"Warning: Progress callback failed: {cb_err}") # Avoid crashing evaluation due to callback error

            try:
                # Run inference with timing
                start_time = time.time()
                
                # Standard prediction for all our models
                detections, segmentations, annotated_image = model_manager.predict(image)
                    
                inference_time = time.time() - start_time
                
                # Record inference time
                metrics["inference_times"].append(inference_time)
                
                # Count successful inferences
                metrics["successful_inferences"] += 1
                
                # Record detection count for this image
                metrics["detection_counts_per_image"].append(len(detections))
                
                # Count total detections
                metrics["total_detections"] += len(detections)
                
                # Update class detection counts
                for det in detections:
                    class_name = det.get("class_name", "unknown")
                    metrics["detection_counts"][class_name] = metrics["detection_counts"].get(class_name, 0) + 1
                    metrics["classes_detected"].add(class_name)
                
                # Save visualization if requested
                if save_visualizations and idx < 20:  # Limit to first 20 images
                    cv2.imwrite(str(vis_dir / f"img_{image_id}_{model_type}.jpg"), annotated_image)
                
                # Format detections for COCO evaluation
                for det_idx, det in enumerate(detections):
                    # Get class ID for COCO format
                    class_id = det.get("class_id")

                    # Debug large sample of detections to understand class IDs
                    if idx < 3 and det_idx < 5:
                        log_info(f"DEBUG: {model_type} detection {det_idx} - class_id={class_id}, class_name={det.get('class_name', 'unknown')}, score={det.get('score', 0)}")
                    
                    # Skip if class_id is None or not recognized
                    if (class_id is None):
                        continue
                    
                    # Get bounding box coordinates
                    box = det.get("box", [0, 0, 0, 0])
                    if len(box) == 4:
                        x1, y1, x2, y2 = box
                        # COCO format is [x, y, width, height]
                        coco_box = [float(x1), float(y1), float(x2-x1), float(y2-y1)]
                        
                        # Add prediction to the list in COCO format
                        pred = {
                            'image_id': int(image_id),
                            'category_id': int(class_id), # Ensure category_id is integer
                            'bbox': coco_box,
                            'score': float(det.get("score", 0.0))
                        }
                        
                        coco_bbox_predictions.append(pred)
                
                # Check if this is a segmentation model based on model type or presence of segmentations
                # Modified to better detect segmentation models based on naming patterns and capabilities
                is_seg_model = 'seg' in model_type.lower() or any('seg' in k for k in DEFAULT_MODEL_PATHS.keys() if model_type in k)
                
                # Process segmentation masks if available
                if segmentations and (is_seg_model or len(segmentations) > 0):
                    for i, mask in enumerate(segmentations):
                        if i < len(detections) and mask is not None:  # Make sure we have a detection to pair with
                            det = detections[i]
                            class_id = det.get("class_id") # Get class_id again for segmentation
                            
                            # Skip if no class_id or if mask is invalid
                            if class_id is None or mask.shape[0] == 0:
                                continue
                            
                            # Convert binary mask to RLE format that COCO expects
                            try:
                                # Debug for sample masks
                                if idx < 2 and i < 2:
                                    log_info(f"DEBUG: {model_type} segm {i} - mask shape={mask.shape}, class_id={class_id}, class_name={det.get('class_name', 'unknown')}")
                                
                                # Ensure mask is binary (0/1)
                                binary_mask = (mask > 0).astype(np.uint8)
                                
                                # Get mask dimensions
                                mask_h, mask_w = binary_mask.shape[:2]
                                
                                # Debug mask size vs image size
                                if idx < 2 and i < 2:
                                    img_h, img_w = image.shape[:2]
                                    log_info(f"DEBUG: Image size: {img_w}x{img_h}, Mask size: {mask_w}x{mask_h}")
                                
                                # Resize mask if needed to match image dimensions
                                img_h, img_w = image.shape[:2]
                                if mask_h != img_h or mask_w != img_w:
                                    binary_mask = cv2.resize(binary_mask, (img_w, img_h), 
                                                           interpolation=cv2.INTER_NEAREST)
                                    if idx < 2 and i < 2:
                                        log_info(f"DEBUG: Resized mask to match image: {img_w}x{img_h}")
                                
                                # Calculate area 
                                area = float(np.sum(binary_mask))
                                if area < 1:  # Skip tiny masks
                                    continue
                                    
                                # Convert to RLE format
                                rle = maskUtils.encode(np.asfortranarray(binary_mask))
                                rle['counts'] = rle['counts'].decode('utf-8')  # Convert bytes to str for JSON
                                
                                # Create COCO format segmentation prediction
                                segm_pred = {
                                    'image_id': int(image_id),
                                    'category_id': int(class_id), # Ensure category_id is integer
                                    'segmentation': rle,
                                    'score': float(det.get("score", 0.0)),
                                    'area': area
                                }
                                
                                coco_segm_predictions.append(segm_pred)
                                
                                # --- SHAPE METRICS CALCULATION ---
                                shape_metrics = self.calculate_shape_metrics(binary_mask)
                                
                                # Store shape metrics
                                metrics["shape_metrics"]["compactness"].append(shape_metrics["compactness"])
                                metrics["shape_metrics"]["convexity"].append(shape_metrics["convexity"])
                                metrics["shape_metrics"]["circularity"].append(shape_metrics["circularity"])
                                
                                # Distribute by size
                                if shape_metrics["circularity"] > 0.5:  # Heuristic for small/medium/large based on circularity
                                    size_category = "small"
                                elif shape_metrics["circularity"] > 0.2:
                                    size_category = "medium"
                                else:
                                    size_category = "large"
                                
                                metrics["shape_metrics"]["by_size"][size_category]["compactness"].append(shape_metrics["compactness"])
                                metrics["shape_metrics"]["by_size"][size_category]["convexity"].append(shape_metrics["convexity"])
                                metrics["shape_metrics"]["by_size"][size_category]["circularity"].append(shape_metrics["circularity"])
                                
                            except Exception as mask_err:
                                log_exception(mask_err, context="Error converting segmentation mask")
                                continue

            except Exception as e:
                log_exception(e, context=f"Error processing image {image_id} with {model_type}")
                metrics["failures"] += 1
        
        # Calculate standard statistics
        if metrics["inference_times"]:
            metrics["mean_inference_time"] = float(np.mean(metrics["inference_times"]))
            metrics["median_inference_time"] = float(np.median(metrics["inference_times"]))
            metrics["min_inference_time"] = float(np.min(metrics["inference_times"]))
            metrics["max_inference_time"] = float(np.max(metrics["inference_times"]))
            metrics["std_inference_time"] = float(np.std(metrics["inference_times"]))
            mean_time = metrics["mean_inference_time"]
            metrics["fps"] = (1.0 / mean_time) if mean_time > 0 else 0.0
        else:
            # No successful inferences: avoid a KeyError later when reporting FPS.
            metrics["fps"] = 0.0
        
        metrics["classes_detected"] = list(metrics["classes_detected"])
        metrics["unique_classes_detected"] = len(metrics["classes_detected"])
        metrics["avg_detections_per_image"] = metrics["total_detections"] / max(metrics["successful_inferences"], 1)
        
        # Run COCO evaluation if we have ground truth annotations and predictions
        if self.coco_gt_api is not None and (coco_bbox_predictions or coco_segm_predictions):
            try:
                # Before evaluation, check validation format
                if coco_bbox_predictions:
                    # Verify a sample of predictions
                    log_info(f"\nDEBUG: First 3 bbox predictions from {len(coco_bbox_predictions)} total:")
                    for i, pred in enumerate(coco_bbox_predictions[:3]):
                        log_info(f"  {i}: image_id={pred['image_id']}, category_id={pred['category_id']}, score={pred['score']:.3f}")
                    
                    # Get ground truth categories for comparison
                    gt_cats = self.coco_gt_api.loadCats(self.coco_gt_api.getCatIds())
                    log_info(f"\nDEBUG: First 5 ground truth categories:")
                    for i, cat in enumerate(gt_cats[:5]):
                        log_info(f"  {i}: id={cat['id']}, name={cat['name']}")
                    
                    # Validate category IDs against ground truth
                    valid_cat_ids = set(cat['id'] for cat in gt_cats)
                    unknown_cat_ids = set(pred['category_id'] for pred in coco_bbox_predictions) - valid_cat_ids
                    
                    if unknown_cat_ids:
                        log_warning(f"\nWARNING: Found {len(unknown_cat_ids)} category IDs not in ground truth: {sorted(unknown_cat_ids)[:5]}...")
                        # Fix category IDs if needed - map to valid IDs or remove invalid ones
                        valid_predictions = []
                        for pred in coco_bbox_predictions:
                            if pred['category_id'] in valid_cat_ids:
                                valid_predictions.append(pred)
                        
                        if len(valid_predictions) < len(coco_bbox_predictions):
                            log_warning(f"WARNING: Removed {len(coco_bbox_predictions) - len(valid_predictions)} predictions with invalid category IDs")
                            coco_bbox_predictions = valid_predictions
                
                # Evaluate bounding box predictions
                if coco_bbox_predictions:
                    log_info(f"Running COCO bbox evaluation for {model_type} with {len(coco_bbox_predictions)} predictions")
                    
                    # Save predictions to a temporary file
                    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
                        json.dump(coco_bbox_predictions, f)
                        dt_bbox_file = f.name
                    
                    # Initialize the COCO detection object with predictions
                    try:
                        coco_dt = self.coco_gt_api.loadRes(dt_bbox_file)
                    except Exception as e:
                        log_error(f"\nERROR loading detection results: {e}")
                        log_info("Attempting to fix format issues...")
                        
                        # Try to fix potentially problematic predictions
                        fixed_predictions = []
                        for pred in coco_bbox_predictions:
                            try:
                                # Ensure all required fields are present and in correct format
                                fixed_pred = {
                                    'image_id': int(pred['image_id']),
                                    'category_id': int(pred['category_id']),
                                    'bbox': [float(x) for x in pred['bbox']],
                                    'score': float(pred['score'])
                                }
                                fixed_predictions.append(fixed_pred)
                            except Exception as fix_err:
                                log_warning(f"Skipping invalid prediction: {fix_err}")
                                continue
                        
                        # Save fixed predictions
                        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
                            json.dump(fixed_predictions, f)
                            dt_bbox_file = f.name
                        
                        coco_dt = self.coco_gt_api.loadRes(dt_bbox_file)
                    
                    # Create the COCO evaluator for bounding box
                    coco_bbox_eval = COCOeval(self.coco_gt_api, coco_dt, 'bbox')
                    
                    # Optional: Set specific image IDs to evaluate (only those we processed)
                    eval_img_ids = [img_id for _, img_id in images]
                    coco_bbox_eval.params.imgIds = eval_img_ids
                    
                    # Run evaluation
                    coco_bbox_eval.evaluate()
                    coco_bbox_eval.accumulate()
                    coco_bbox_eval.summarize()
                    
                    # Add COCO bbox metrics to our results
                    metrics["coco_metrics"] = {
                        "AP_IoU=0.50:0.95": float(coco_bbox_eval.stats[0]),  # AP at IoU=0.50:0.95
                        "AP_IoU=0.50": float(coco_bbox_eval.stats[1]),       # AP at IoU=0.50
                        "AP_IoU=0.75": float(coco_bbox_eval.stats[2]),       # AP at IoU=0.75
                        "AP_small": float(coco_bbox_eval.stats[3]),          # AP for small objects
                        "AP_medium": float(coco_bbox_eval.stats[4]),         # AP for medium objects
                        "AP_large": float(coco_bbox_eval.stats[5]),          # AP for large objects
                        "AR_max=1": float(coco_bbox_eval.stats[6]),          # AR given 1 detection per image
                        "AR_max=10": float(coco_bbox_eval.stats[7]),         # AR given 10 detections per image
                        "AR_max=100": float(coco_bbox_eval.stats[8]),        # AR given 100 detections per image
                        "AR_small": float(coco_bbox_eval.stats[9]),          # AR for small objects
                        "AR_medium": float(coco_bbox_eval.stats[10]),        # AR for medium objects
                        "AR_large": float(coco_bbox_eval.stats[11])          # AR for large objects
                    }

                    # Extract and store PR curve data for bounding boxes
                    # eval['precision'] is (T x R x K x A x M)
                    # T: IoU thresholds, R: Recall thresholds, K: Categories, A: Area ranges, M: Max detections
                    if coco_bbox_eval.eval['precision'].size > 0:
                        # Precision for the first IoU threshold (e.g., 0.5), all recall thresholds, 
                        # averaged over all categories, for 'all' area (index 0), and maxDets=100 (last index for M)
                        precision_values = coco_bbox_eval.eval['precision'][0, :, :, 0, -1] 
                        mean_precision = np.mean(precision_values, axis=1) # Average over categories (K dimension)
                        recall_values = coco_bbox_eval.params.recThrs
                        
                        metrics["coco_metrics"]["pr_curve_precision"] = mean_precision.tolist()
                        metrics["coco_metrics"]["pr_curve_recall"] = recall_values.tolist()
                    else:
                        metrics["coco_metrics"]["pr_curve_precision"] = []
                        metrics["coco_metrics"]["pr_curve_recall"] = []
                    
                    # Clean up the temporary file
                    os.unlink(dt_bbox_file)
                
                # Evaluate segmentation predictions for models with segmentation capability
                if coco_segm_predictions and is_seg_model:
                    log_info(f"Running COCO segmentation evaluation for {model_type} with {len(coco_segm_predictions)} predictions")
                    
                    # Save predictions to a temporary file
                    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
                        json.dump(coco_segm_predictions, f)
                        dt_segm_file = f.name
                    
                    # Initialize the COCO detection object with predictions
                    try:
                        coco_dt = self.coco_gt_api.loadRes(dt_segm_file)
                    except Exception as e:
                        log_error(f"\nERROR loading segmentation results: {e}")
                        log_info("Attempting to fix format issues...")
                        
                        # Try to fix potentially problematic predictions
                        fixed_predictions = []
                        for pred in coco_segm_predictions:
                            try:
                                # Ensure all required fields are present and in correct format
                                fixed_pred = {
                                    'image_id': int(pred['image_id']),
                                    'category_id': int(pred['category_id']),
                                    'segmentation': pred['segmentation'],
                                    'score': float(pred['score']),
                                    'area': float(pred['area']) if 'area' in pred else 100.0
                                }
                                fixed_predictions.append(fixed_pred)
                            except Exception as fix_err:
                                log_warning(f"Skipping invalid segmentation prediction: {fix_err}")
                                continue
                        
                        # Save fixed predictions
                        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
                            json.dump(fixed_predictions, f)
                            dt_segm_file = f.name
                        
                        coco_dt = self.coco_gt_api.loadRes(dt_segm_file)
                    
                    # Create the COCO evaluator for segmentation
                    coco_segm_eval = COCOeval(self.coco_gt_api, coco_dt, 'segm')
                    
                    # Optional: Set specific image IDs to evaluate (only those we processed)
                    eval_img_ids = [img_id for _, img_id in images]
                    coco_segm_eval.params.imgIds = eval_img_ids
                    
                    # Run evaluation
                    coco_segm_eval.evaluate()
                    coco_segm_eval.accumulate()
                    coco_segm_eval.summarize()
                    
                    # Add COCO segmentation metrics to our results
                    metrics["coco_segm_metrics"] = {
                        "Segm_AP_IoU=0.50:0.95": float(coco_segm_eval.stats[0]),  # AP at IoU=0.50:0.95
                        "Segm_AP_IoU=0.50": float(coco_segm_eval.stats[1]),       # AP at IoU=0.50
                        "Segm_AP_IoU=0.75": float(coco_segm_eval.stats[2]),       # AP at IoU=0.75
                        "Segm_AP_small": float(coco_segm_eval.stats[3]),          # AP for small objects
                        "Segm_AP_medium": float(coco_segm_eval.stats[4]),         # AP for medium objects
                        "Segm_AP_large": float(coco_segm_eval.stats[5]),          # AP for large objects
                        "Segm_AR_max=1": float(coco_segm_eval.stats[6]),          # AR given 1 detection per image
                        "Segm_AR_max=10": float(coco_segm_eval.stats[7]),         # Segm AR given 10 detections per image
                        "Segm_AR_max=100": float(coco_segm_eval.stats[8]),        # AR given 100 detections per image
                        "Segm_AR_small": float(coco_segm_eval.stats[9]),          # AR for small objects
                        "Segm_AR_medium": float(coco_segm_eval.stats[10]),        # AR for medium objects
                        "Segm_AR_large": float(coco_segm_eval.stats[11])          # AR for large objects
                    }
                    
                    # Clean up the temporary file
                    os.unlink(dt_segm_file)
                # Even if we don't have segmentation predictions, include basic metric structure for consistency
                elif is_seg_model and not coco_segm_predictions:
                    log_info(f"No valid segmentation predictions for {model_type}")
                    # Add empty segmentation metrics to maintain consistent structure
                    metrics["coco_segm_metrics"] = {
                        "Segm_AP_IoU=0.50:0.95": 0.0,
                        "Segm_AP_IoU=0.50": 0.0,
                        "Segm_AP_IoU=0.75": 0.0,
                        "Segm_AP_small": 0.0,
                        "Segm_AP_medium": 0.0,
                        "Segm_AP_large": 0.0
                    }
            except Exception as e:
                log_exception(e, context=f"Error during COCO evaluation for {model_type}")
                import traceback
                traceback.print_exc()
                
                metrics["coco_metrics"] = {"error": str(e)}
                if is_seg_model:
                    metrics["coco_segm_metrics"] = {"error": str(e)}
        
        # Print summary
        log_info(f"\n{model_type.upper()} Evaluation Summary:")
        log_info(f"- Successfully processed: {metrics['successful_inferences']}/{metrics['total_images']} images")
        log_info(f"- Average FPS: {metrics['fps']:.2f}")
        log_info(f"- Total detections: {metrics['total_detections']}")
        log_info(f"- Unique classes detected: {metrics['unique_classes_detected']}")
        log_info(f"- Average detections per image: {metrics['avg_detections_per_image']:.2f}")
        
        # Print COCO metrics if available
        if "coco_metrics" in metrics and metrics["coco_metrics"] and "error" not in metrics["coco_metrics"]:
            log_info("- COCO Detection Metrics:")
            log_info(f"  • mAP (IoU=0.50:0.95): {metrics['coco_metrics']['AP_IoU=0.50:0.95']:.4f}")
            log_info(f"  • AP (IoU=0.50): {metrics['coco_metrics']['AP_IoU=0.50']:.4f}")
            log_info(f"  • AP (IoU=0.75): {metrics['coco_metrics']['AP_IoU=0.75']:.4f}")
        
        # Print segmentation metrics if available
        if "coco_segm_metrics" in metrics and metrics["coco_segm_metrics"] and "error" not in metrics["coco_segm_metrics"]:
            log_info("- COCO Segmentation Metrics:")
            log_info(f"  • Segmentation mAP (IoU=0.50:0.95): {metrics['coco_segm_metrics']['Segm_AP_IoU=0.50:0.95']:.4f}")
            log_info(f"  • Segmentation AP (IoU=0.50): {metrics['coco_segm_metrics']['Segm_AP_IoU=0.50']:.4f}")
            log_info(f"  • Segmentation AP (IoU=0.75): {metrics['coco_segm_metrics']['Segm_AP_IoU=0.75']:.4f}")
        
        # If shape metrics were collected, include summary stats
        if metrics["shape_metrics"]["compactness"]:
            log_info("- Shape Metrics (Segmentation Quality):")
            log_info(f"  • Compactness: {np.mean(metrics['shape_metrics']['compactness']):.4f}")
            log_info(f"  • Convexity: {np.mean(metrics['shape_metrics']['convexity']):.4f}")
            log_info(f"  • Circularity: {np.mean(metrics['shape_metrics']['circularity']):.4f}")

        return metrics
    
    def run_reliability_test(self, max_images=20, num_runs=3):
        """
        Run multiple evaluations to test model reliability
        
        Args:
            max_images (int): Maximum number of images to evaluate
            num_runs (int): Number of evaluation runs
        
        Returns:
            dict: Reliability metrics for each model
        """
        log_info(f"\nRunning reliability test with {num_runs} evaluation runs")
        
        # Load validation images
        images = self.load_coco_val_images(max_images)
        if not images:
            log_error("Error: No images to evaluate")
            return {}
        
        # Run multiple evaluations for each model
        reliability_results = {}
        
        for model_config in self.models_list:
            model_type = model_config["type"]
            log_info(f"\nTesting reliability of {model_type} model...")
            
            run_metrics = []
            for run in range(num_runs):
                log_info(f"Run {run+1}/{num_runs}:")
                metrics = self.evaluate_model(model_config, images, save_visualizations=(run==0))
                if metrics:
                    run_metrics.append(metrics)
            
            if not run_metrics:
                log_warning(f"No successful evaluation runs for {model_type}")
                continue
            
            # Calculate reliability metrics
            fps_values = [m.get("fps", 0) for m in run_metrics]
            detection_counts = [m.get("total_detections", 0) for m in run_metrics]
            
            reliability_results[model_type] = {
                "fps_mean": np.mean(fps_values),
                "fps_std": np.std(fps_values),
                "fps_var": np.var(fps_values),
                "fps_coef_var": np.std(fps_values) / np.mean(fps_values) if np.mean(fps_values) > 0 else 0,
                "detection_mean": np.mean(detection_counts),
                "detection_std": np.std(detection_counts),
                "run_metrics": run_metrics
            }
            
            log_info(f"{model_type} Reliability Results:")
            log_info(f"- Mean FPS: {reliability_results[model_type]['fps_mean']:.2f} ± {reliability_results[model_type]['fps_std']:.2f}")
            log_info(f"- FPS Coefficient of Variation: {reliability_results[model_type]['fps_coef_var']:.3f}")
            log_info(f"- Mean Detections: {reliability_results[model_type]['detection_mean']:.1f} ± {reliability_results[model_type]['detection_std']:.1f}")
        
        # Save reliability results
        if reliability_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            reliability_file = RESULTS_DIR / f"reliability_results_{timestamp}.json"
            
            # Convert to serializable format 
            serializable_results = {}
            for model_type, metrics in reliability_results.items():
                serializable_metrics = {}
                for key, value in metrics.items():
                    if key != "run_metrics":  # Skip detailed run metrics - too large
                        serializable_metrics[key] = float(value) if isinstance(value, (np.float32, np.float64)) else value
                serializable_results[model_type] = serializable_metrics
            
            with open(reliability_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            log_info(f"Reliability results saved to {reliability_file}")
        
        return reliability_results
    
    def run_evaluation(self, max_images=50, save_visualizations=True, progress_callback=None):
        """
        Run evaluation on all configured models
        
        Args:
            max_images (int): Maximum number of images to evaluate
            save_visualizations (bool): Whether to save visualized detections
            progress_callback (callable, optional): Function to report progress during model evaluation.
        """
        log_info(f"\nEvaluating models using {max_images} images from COCO val2017 dataset")
        log_info(f"This represents {(max_images/COCO_VAL_TOTAL_IMAGES)*100:.1f}% of the full validation set")
        
        # Load validation images
        images = self.load_coco_val_images(max_images)
        if not images:
            log_error("Error: No images to evaluate")
            return
        
        # Evaluate each model
        results = {}
        for model_config in self.models_list:
            model_type = model_config["type"]
            # Pass the progress callback down to evaluate_model
            metrics = self.evaluate_model(model_config, images, save_visualizations, progress_callback)
            if metrics:
                results[model_type] = metrics
        
        # Save results
        self.results = results
        self.save_results()

    def save_results(self):
        """Save evaluation results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = RESULTS_DIR / f"evaluation_results_{timestamp}.json"
        
        # Convert sets to lists for JSON serialization
        serializable_results = {}
        for model_type, metrics in self.results.items():
            serializable_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, set):
                    serializable_metrics[key] = list(value)
                else:
                    serializable_metrics[key] = value
            serializable_results[model_type] = serializable_metrics
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        log_info(f"Results saved to {results_file}")

    def calculate_shape_metrics(self, binary_mask):
        """
        Calculate shape metrics for a segmentation mask.
        
        Args:
            binary_mask (np.ndarray): Binary segmentation mask (0-1 values)
            
        Returns:
            dict: Dictionary with shape metrics (compactness, convexity, circularity)
        """
        # Ensure mask is binary
        mask = (binary_mask > 0).astype(np.uint8)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Initialize metrics
        metrics = {
            'compactness': 0.0,
            'convexity': 0.0,
            'circularity': 0.0,
            'count': 0  # Number of contours analyzed
        }
        
        if not contours:
            return metrics
            
        # Calculate metrics for each contour and average them
        for contour in contours:
            # Skip tiny contours (likely noise)
            if cv2.contourArea(contour) < 10:  
                continue
                
            # Perimeter
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            # Area
            area = cv2.contourArea(contour)
            if area == 0:
                continue
                
            # Convex hull
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            hull_perimeter = cv2.arcLength(hull, True)
            
            # Calculate metrics
            # 1. Compactness: 4π*area/perimeter²
            compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
            
            # 2. Convexity: perimeter of convex hull / perimeter of contour
            convexity = hull_perimeter / perimeter if perimeter > 0 else 0
            
            # 3. Circularity: How closely shape resembles a circle
            # sqrt(area / π) / (maximum distance from center to any point)
            # Approximate using ratio of area to minimum enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius > 0:
                circle_area = np.pi * (radius * radius)
                circularity = area / circle_area if circle_area > 0 else 0
            else:
                circularity = 0
                
            # Update aggregate metrics
            metrics['compactness'] += compactness
            metrics['convexity'] += convexity
            metrics['circularity'] += circularity
            metrics['count'] += 1
            
        # Calculate averages
        if metrics['count'] > 0:
            metrics['compactness'] /= metrics['count']
            metrics['convexity'] /= metrics['count']
            metrics['circularity'] /= metrics['count']
            
        return metrics

@with_error_handling("Error preparing COCO dataset")
def prepare_coco_dataset(num_images=50, shuffle=True, seed=42):
    """
    Prepare the COCO dataset for evaluation.
    
    Args:
        num_images: Number of images to use for evaluation
        shuffle: Whether to shuffle the images
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (image_paths, image_ids)
    """
    # Validate inputs
    if not isinstance(num_images, int) or num_images <= 0:
        raise InvalidInputError(f"Number of images must be a positive integer, got {num_images}")
    
    # Ensure COCO directories exist
    coco_dir = validate_file_path(COCO_VAL_DIR, must_exist=False)
    coco_annot = validate_file_path(COCO_ANNOT_FILE, must_exist=False)
    
    if not coco_dir.exists() or not coco_annot.exists():
        log_info(f"COCO dataset not found at {coco_dir}. Attempting to download...")
        # This would normally call a download function
        # Since we don't have that implemented, we'll just raise an error
        raise InvalidInputError(
            f"COCO dataset not available. Please download manually:\n"
            f"1. Download val2017 images from https://cocodataset.org/\n"
            f"2. Extract to {coco_dir}\n"
            f"3. Download annotations from https://cocodataset.org/\n"
            f"4. Extract to {coco_annot.parent}"
        )
    
    # Load COCO annotations
    try:
        coco = COCO(str(coco_annot))
    except Exception as e:
        log_exception(e, "Failed to load COCO annotations")
        raise InvalidInputError(f"Failed to load COCO annotations: {str(e)}")
    
    # Get image IDs
    image_ids = list(sorted(coco.imgs.keys()))
    
    # Shuffle if needed
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(image_ids)
    
    # Select subset of images
    image_ids = image_ids[:num_images]
    
    # Get image paths
    image_paths = [str(COCO_VAL_DIR / coco.loadImgs(img_id)[0]['file_name']) for img_id in image_ids]
    
    log_info(f"Prepared {len(image_paths)} COCO images for evaluation")
    return image_paths, image_ids

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate computer vision models on COCO validation set")
    parser.add_argument("--images", type=int, default=50,
                        help=f"Number of COCO val2017 images to use (max {COCO_VAL_TOTAL_IMAGES}, default: 50). Increase to 200+ for more reliable mAP metrics.")
    parser.add_argument("--no-vis", action="store_true",
                        help="Skip saving individual detection visualizations (dashboard will still be generated)")
    
    # Dynamically generate model choices from DEFAULT_MODEL_PATHS
    model_choices = list(DEFAULT_MODEL_PATHS.keys())
    parser.add_argument("--models", type=str, nargs="+", choices=model_choices,
                        help="Specific models to evaluate (default: all)")
    
    # Add new argument for confidence threshold
    parser.add_argument("--conf-threshold", type=float, default=0.25,
                        help="Confidence threshold for detections (default: 0.25)")
    
    # Add new argument for IoU threshold
    parser.add_argument("--iou-threshold", type=float, default=0.45,
                        help="IoU threshold for NMS (default: 0.45)")

    # Force the inference device (otherwise auto-detected).
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"],
                        help="Force inference device (default: auto-detect)")

    args = parser.parse_args()

    if args.device:
        os.environ["RTO_DEVICE"] = args.device
    
    # Validate args
    max_images = min(max(1, args.images), COCO_VAL_TOTAL_IMAGES)
    if max_images != args.images:
        log_info(f"Adjusted number of images to {max_images} (valid range: 1-{COCO_VAL_TOTAL_IMAGES})")

    save_individual_visualizations = not args.no_vis # Renamed for clarity

    # Determine which models to evaluate
    all_available_model_configs = MODELS_TO_EVALUATE # This is the full list from global scope
      # Define default SoA models (ensure these keys exist in DEFAULT_MODEL_PATHS)
    DEFAULT_SOA_MODELS = ["yolov9e-seg", "yolo11m-seg", "yolov8n-seg"]

    selected_models_for_evaluation = []

    if args.models:
        # User specified models
        user_selected_model_types = args.models
        log_info(f"Evaluating user-specified models: {', '.join(user_selected_model_types)}")
        
        # Filter the full list of available model configs based on user selection
        selected_models_for_evaluation = [m_config for m_config in all_available_model_configs if m_config["type"] in user_selected_model_types]
        
        # Check if all user-specified models were found and warn if not
        found_model_types = [m_config["type"] for m_config in selected_models_for_evaluation]
        for sm_type in user_selected_model_types:
            if (sm_type not in found_model_types):
                log_warning(f"Warning: Specified model type '{sm_type}' not found in available model configurations. It will be skipped.")
    else:
        # No models specified by user, use default SoA models
        log_info(f"No models specified via --models argument. Evaluating default SoA models: {', '.join(DEFAULT_SOA_MODELS)}")
        selected_models_for_evaluation = [m_config for m_config in all_available_model_configs if m_config["type"] in DEFAULT_SOA_MODELS]
        
        # Verify that default models were found
        if len(selected_models_for_evaluation) < len(DEFAULT_SOA_MODELS):
            found_default_types = [m_config["type"] for m_config in selected_models_for_evaluation]
            missing_defaults = [m_type for m_type in DEFAULT_SOA_MODELS if m_type not in found_default_types]
            if missing_defaults:
                log_warning(f"Warning: The following default SoA models were not found in available model configurations and will be skipped: {', '.join(missing_defaults)}")
        
        if not selected_models_for_evaluation: # If even after trying defaults, none are available
             log_error(f"Error: None of the default SoA models ({', '.join(DEFAULT_SOA_MODELS)}) are available in the current model configurations.")
             available_model_names_from_configs = [m_config["type"] for m_config in all_available_model_configs]
             log_info(f"Available models are: {', '.join(available_model_names_from_configs) if available_model_names_from_configs else 'None'}")
             sys.exit(1)

    if not selected_models_for_evaluation:
        log_error("Error: No models selected or available for evaluation. Exiting.")
        available_model_names_from_configs = [m_config["type"] for m_config in all_available_model_configs]
        log_info(f"Available models are: {', '.join(available_model_names_from_configs) if available_model_names_from_configs else 'None'}")
        sys.exit(1)

    log_info(f"Final list of models to evaluate: {', '.join([m['type'] for m in selected_models_for_evaluation])}")

    # Add confidence and IoU thresholds to model configs
    for model_config in selected_models_for_evaluation:
        model_config["conf_threshold"] = args.conf_threshold
        model_config["iou_threshold"] = args.iou_threshold

    log_info("Starting model evaluation...")

    # Create evaluator with the determined list of models
    evaluator = ModelEvaluator(selected_models_for_evaluation)

    # Run evaluation
    evaluator.run_evaluation(max_images, save_individual_visualizations)

    log_info("Evaluation complete!")

    # Always generate the metrics dashboard after evaluation
    log_info("\nGenerating metrics dashboard...")
    try:
        # Try relative import first (when running as a module)
        try:
            from metrics_visualizer import MetricsVisualizer
        except ImportError:
            # Fall back to src-prefixed import (when running from project root)
            from src.metrics_visualizer import MetricsVisualizer
            
        visualizer = MetricsVisualizer()  # Will automatically use latest results file

        # Create the dashboard
        dashboard_path = visualizer.create_metrics_dashboard(show_plot=False) # Set show_plot=False to only save

        if dashboard_path:
            log_info(f"Metrics dashboard generated successfully at: {dashboard_path}")
        else:
            log_error("Failed to generate metrics dashboard.")

    except ImportError as e:
        log_warning(f"Warning: Could not import metrics visualizer: {e}")
        log_info("Skipping dashboard generation...")
    except Exception as e:
        log_error(f"Error generating dashboard: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()