"""
Demo Video Generator for Computer Vision Project

This script creates a demonstration video using the selected model (likely the best-performing model
from the evaluation). It processes a video file, detecting and segmenting objects in each frame,
and saves the output as a new video file with visualized detections.
"""

import os
import sys
import time
import json
from pathlib import Path
import argparse
import torch  # Add import for torch to fix NameError
from datetime import datetime  # Add datetime import

# Add the parent directory to sys.path for proper imports
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import project components - handle both relative and absolute imports
try:
    # Try relative imports first (when running from within src directory)
    from models import ModelManager, DEFAULT_MODEL_PATHS
    from video_utils import process_video_with_model
    # For pipeline functions, handle both import styles
    try:
        from pipeline import determine_best_model, find_latest_results
    except (ImportError, AttributeError):
        # If pipeline hasn't been fully imported yet, define the functions here
        def find_latest_results():
            """Find the latest evaluation results file"""
            result_files = list(RESULTS_DIR.glob("evaluation_results_*.json"))
            if not result_files:
                return None
            
            # Sort by modification time (newest first)
            latest_file = max(result_files, key=lambda f: f.stat().st_mtime)
            return latest_file
            
        def determine_best_model(results):
            """Simplified version for this file only"""
            if not results:
                return None, None
                
            # Use mAP and FPS as main criteria
            best_model = None
            best_score = -1
            best_details = {}
            
            for model_name, metrics in results.items():
                if "coco_metrics" not in metrics:
                    continue
                    
                coco_metrics = metrics["coco_metrics"]
                fps = metrics.get("fps", 0)
                mAP = coco_metrics.get("AP_IoU=0.50:0.95", 0)
                
                # Simple weighted score
                weighted_score = (mAP * 0.7) + ((min(fps, 30) / 30) * 0.3)
                
                if weighted_score > best_score:
                    best_score = weighted_score
                    best_model = model_name
                    best_details = {
                        "mAP": mAP,
                        "FPS": fps,
                        "weighted_score": weighted_score
                    }
            
            return best_model, best_details
except ImportError:
    # Fall back to absolute imports (when running from project root)
    from src.models import ModelManager, DEFAULT_MODEL_PATHS
    from src.video_utils import process_video_with_model
    try:
        from src.pipeline import determine_best_model, find_latest_results
    except (ImportError, AttributeError):
        # Define inline versions if needed
        def find_latest_results():
            """Find the latest evaluation results file"""
            result_files = list(RESULTS_DIR.glob("evaluation_results_*.json"))
            if not result_files:
                return None
            
            # Sort by modification time (newest first)
            latest_file = max(result_files, key=lambda f: f.stat().st_mtime)
            return latest_file
            
        def determine_best_model(results):
            """Simplified version for this file only"""
            if not results:
                return None, None
                
            # Use mAP and FPS as main criteria
            best_model = None
            best_score = -1
            best_details = {}
            
            for model_name, metrics in results.items():
                if "coco_metrics" not in metrics:
                    continue
                    
                coco_metrics = metrics["coco_metrics"]
                fps = metrics.get("fps", 0)
                mAP = coco_metrics.get("AP_IoU=0.50:0.95", 0)
                
                # Simple weighted score
                weighted_score = (mAP * 0.7) + ((min(fps, 30) / 30) * 0.3)
                
                if weighted_score > best_score:
                    best_score = weighted_score
                    best_model = model_name
                    best_details = {
                        "mAP": mAP,
                        "FPS": fps,
                        "weighted_score": weighted_score
                    }
            
            return best_model, best_details

# Import validation and error handling utilities
from src.validation import (
    validate_file_path, validate_model_name, validate_video,
    validate_confidence_threshold, validate_iou_threshold,
    validate_device, InvalidInputError
)
from src.error_handling import (
    log_info, log_warning, log_error, log_exception,
    handle_exceptions, with_error_handling, print_friendly_error
)

# Configure paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = PROJECT_ROOT / "data_sets" / "video_data" # Removed "samples"
OUTPUT_DIR = PROJECT_ROOT / "inference" / "output_videos"
RESULTS_DIR = PROJECT_ROOT / "inference" / "results"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

@with_error_handling("Error processing video with detections")
def create_demo_video(
    model_name="yolov8n-seg",
    video_path=None,
    output_path=None,
    model_path=None,
    conf_threshold=0.25,
    iou_threshold=0.45,
    device=None,
    show_progress=True, # This existing arg controls terminal progress, not GUI callback
    progress_callback=None # New argument for GUI progress updates
):
    """
    Create a demo video with object detection and segmentation.
    
    Args:
        model_name: Name of the model to use
        video_path: Path to the input video
        output_path: Path to save the output video
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
        device: Device to use for inference (cuda, cpu, mps)
        show_progress: Whether to show progress bar in terminal (legacy)
        progress_callback: Callback function for GUI progress updates.
                           Expected signature: callback(frame, current_frame, total_frames, model_type, progress)
                           'progress' is the percentage (0-100).
                           'frame' can be None for progress-only updates.
                           
    Returns:
        Path to the output video
    """
    try:
        # Resolve the device up front so the model is placed correctly.
        if device:
            device = validate_device(device)
        else:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        # Resolve the model: a known default name, or an explicit weights path.
        if model_path:
            model_path = validate_file_path(model_path)
            model_type = "yolo-custom"
            model_name = Path(model_path).stem
            log_info(f"Using custom model weights: {model_path}")
        else:
            from models import DEFAULT_MODEL_PATHS
            model_name = validate_model_name(model_name, list(DEFAULT_MODEL_PATHS.keys()))
            model_type = model_name
            model_path = None

        # Initialize manager with the resolved device (and optional custom weights)
        manager = ModelManager(
            model_type=model_type,
            model_path=model_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            device=device,
        )

        if video_path is None:
            # List available samples if no video provided
            samples = list(VIDEO_DIR.glob("*.mp4"))
            if not samples:
                raise InvalidInputError(
                    "No sample videos found and no video path provided. "
                    "Please provide a video path with --video."
                )
            video_path = samples[0]
            log_info(f"No video path provided. Using sample: {video_path}")

        video_path = validate_video(video_path)

        # Set default output path if not provided
        if output_path is None:
            video_name = Path(video_path).stem
            output_path = str(OUTPUT_DIR / f"{video_name}_{model_name}_demo.mp4")
        else:
            output_path = str(validate_file_path(output_path, must_exist=False))
            # Create parent directory if it doesn't exist
            Path(output_path).parent.mkdir(exist_ok=True, parents=True)
        
        # Use the model directly from the model manager rather than trying to load it
        log_info(f"Using model: {model_name}")
        # The model is already loaded in the manager during initialization
        # No need to call manager.load_model
        
        # Process video and generate output
        log_info(f"Processing video: {video_path}")
        stats = process_video_with_model(
            video_path,
            manager,
            output_path=output_path,
            add_fps=show_progress, # This controls FPS overlay on video frames
            callback=progress_callback # Pass the new callback here
        )
        
        # Save performance summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        performance_file = RESULTS_DIR / f"demo_performance_{timestamp}.json"
        
        performance_data = {
            "model_name": model_name,
            "video_path": str(video_path),
            "output_path": output_path,
            "conf_threshold": conf_threshold,
            "iou_threshold": iou_threshold,
            "device": device,
            "stats": stats,
            "timestamp": timestamp
        }
        
        with open(performance_file, "w") as f:
            json.dump(performance_data, f, indent=2)
        
        log_info(f"Demo video created: {output_path}")
        log_info(f"Performance summary saved: {performance_file}")
        
        return output_path
        
    except Exception as e:
        log_exception(e, "Error in create_demo_video")
        raise

def find_best_model_from_evaluation():
    """Find the best model from the latest evaluation results"""
    latest_results_file = find_latest_results()
    
    if not latest_results_file:
        print("No evaluation results found. Please run evaluate_models.py first.")
        return None
    
    print(f"Loading evaluation results from: {latest_results_file}")
    
    try:
        with open(latest_results_file, 'r') as f:
            results = json.load(f)
        
        best_model, details = determine_best_model(results)
        
        if best_model:
            print(f"\nBest model determined from evaluation: {best_model}")
            print(f"Selection criteria:")
            for criterion, value in details.items():
                print(f"- {criterion}: {value}")
            
            return best_model
        else:
            print("Could not determine best model from results.")
            return None
    
    except Exception as e:
        print(f"Error loading evaluation results: {e}")
        return None

def list_available_videos():
    """List videos available in the sample video directory"""
    if not VIDEO_DIR.is_dir():
        print(f"Error: Sample video directory not found: {VIDEO_DIR}")
        return []
    
    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    if not videos:
        print("No sample videos found.")
        return []
    
    print("\nAvailable sample videos:")
    for i, video in enumerate(videos, 1):
        print(f"{i}. {video.name}")
        
    return videos

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Create demo video with object detection/segmentation")
    
    # Model selection options
    model_group = parser.add_argument_group("Model Selection")
    model_selection = model_group.add_mutually_exclusive_group()
    model_selection.add_argument("--model", "-m", type=str, 
                       help="Model type to use (e.g., yolov8n-seg)")
    model_selection.add_argument("--best-model", action="store_true",
                       help="Use best model from latest evaluation results")
    model_group.add_argument("--model-path", type=str,
                       help="Path to model weights file (if not using default)")
    
    # Video options
    video_group = parser.add_argument_group("Video Options")
    video_group.add_argument("--video", "-v", type=str,
                       help="Video file path (if not specified, will list available samples)")
    video_group.add_argument("--output", "-o", type=str, 
                       help="Output video file path (if not specified, will use default naming)")
    
    # Detection parameters
    detection_group = parser.add_argument_group("Detection Parameters")
    detection_group.add_argument("--conf-threshold", type=float, default=0.25,
                       help="Confidence threshold for detections (default: 0.25)")
    detection_group.add_argument("--iou-threshold", type=float, default=0.45,
                       help="IoU threshold for NMS (default: 0.45)")
    detection_group.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"],
                       help="Force inference device (default: auto-detect)")
    
    # Other options
    other_group = parser.add_argument_group("Other Options")
    other_group.add_argument("--list-models", action="store_true",
                       help="List available models and exit")
    
    args = parser.parse_args()
    
    # List available models if requested
    if args.list_models:
        print("\nAvailable models:")
        for i, (model_key, path) in enumerate(DEFAULT_MODEL_PATHS.items(), 1):
            print(f"{i}. {model_key}")
        return
    
    # Determine model type from args
    model_type = None
    
    if args.best_model:
        model_type = find_best_model_from_evaluation()
        if not model_type:
            print("Could not determine best model. Please specify a model with --model.")
            return
    elif args.model:
        model_type = args.model
    elif args.model_path:
        # Custom weights provided; the model name is derived from the file.
        model_type = None
    else:
        # No model specified - show available models and ask user to choose
        print("\nAvailable models:")
        model_keys = list(DEFAULT_MODEL_PATHS.keys())
        for i, model_key in enumerate(model_keys, 1):
            print(f"{i}. {model_key}")
            
        try:
            selection = int(input("\nEnter model number to use (or 0 to exit): "))
            if selection == 0:
                print("Exiting...")
                return
            if selection < 1 or selection > len(model_keys):
                print(f"Invalid selection. Please choose a number between 1 and {len(model_keys)}")
                return
            
            model_type = model_keys[selection - 1]
            print(f"Selected model: {model_type}")
        except ValueError:
            print("Invalid input. Please enter a number.")
            return
    
    # If video not specified, show list of available videos
    if args.video is None:
        videos = list_available_videos()
        if not videos:
            return
        
        # Ask user to select a video
        try:
            selection = int(input("\nEnter video number to process (or 0 to exit): "))
            if selection == 0:
                print("Exiting...")
                return
            if selection < 1 or selection > len(videos):
                print(f"Invalid selection. Please choose a number between 1 and {len(videos)}")
                return
            
            video_path = videos[selection - 1]
        except ValueError:
            print("Invalid input. Please enter a number.")
            return
    else:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"Error: Video file not found: {video_path}")
            return
    
    # Create the demo video
    output_path = create_demo_video(
        model_name=model_type,
        video_path=video_path,
        output_path=args.output,
        model_path=args.model_path,
        device=args.device,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
    )
    
    if output_path:
        print(f"Demo video creation complete! Video saved to: {output_path}")
    else:
        print("Failed to create demo video.")

if __name__ == "__main__":
    main()