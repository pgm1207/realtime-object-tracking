"""
Side-by-Side Model Comparison Tool

This script creates side-by-side comparison videos showing multiple models
processing the same input video simultaneously. This provides a direct visual
comparison of detection quality, accuracy, and performance between models.
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add the parent directory to sys.path for proper imports
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import project components - handle both relative and absolute imports
try:
    # Try direct import first
    from video_utils import get_video_properties, open_video_writer
    from models import ModelManager, DEFAULT_MODEL_PATHS
except ImportError:
    try:
        # Try relative imports (when running from within src directory)
        from .video_utils import get_video_properties, open_video_writer
        from .models import ModelManager, DEFAULT_MODEL_PATHS
    except ImportError:
        # Fall back to absolute imports (when running from project root)
        from src.video_utils import get_video_properties, open_video_writer
        from src.models import ModelManager, DEFAULT_MODEL_PATHS

# Configure paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = PROJECT_ROOT / "data_sets" / "video_data"
OUTPUT_DIR = PROJECT_ROOT / "inference" / "output_videos"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def create_comparison_video(video_path, model_types, output_path=None, max_frames=None, 
                          conf_threshold=0.25, iou_threshold=0.45, fps_target=None):
    """
    Create a side-by-side comparison video showing multiple models processing the same input.
    
    Args:
        video_path (str or Path): Path to input video file
        model_types (list): List of model types to compare
        output_path (str or Path, optional): Path to save output video file
        max_frames (int, optional): Maximum number of frames to process
        conf_threshold (float): Confidence threshold for detections (0-1)
        iou_threshold (float): IoU threshold for NMS (0-1)
        fps_target (int, optional): Target FPS for output video
        
    Returns:
        str: Path to the output video file
    """
    # Input validation
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return None
        
    # Ensure we have at least two models to compare
    if len(model_types) < 2:
        print("Error: Need at least two models to compare.")
        return None
        
    # Determine output path if not specified
    if output_path is None:
        model_names = "_vs_".join([m.split('-')[0] for m in model_types])
        output_filename = f"{video_path.stem}_{model_names}_comparison.mp4"
        output_path = OUTPUT_DIR / output_filename
    else:
        output_path = Path(output_path)
        
    print(f"Processing video: {video_path}")
    print(f"Comparing models: {', '.join(model_types)}")
    print(f"Output will be saved to: {output_path}")
    print(f"Detection thresholds: confidence={conf_threshold}, IoU={iou_threshold}")
    
    # Initialize model managers
    model_managers = []
    for model_type in model_types:
        try:
            model_manager = ModelManager(
                model_type,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold
            )
            if model_manager.model_wrapper is None:
                print(f"Error: Failed to initialize {model_type} model")
                return None
            model_managers.append(model_manager)
        except Exception as e:
            print(f"Error initializing {model_type} model: {e}")
            return None
    
    # Open the input video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return None
        
    # Get video properties
    fps, total_frames, width, height = get_video_properties(cap)
    if fps_target is None:
        fps_target = fps
    
    # Limit frames if specified
    if max_frames is not None and max_frames > 0:
        total_frames = min(total_frames, max_frames)
    
    # Calculate layout for side-by-side display
    # For 2 models: 1x2 grid, For 3-4 models: 2x2 grid
    if len(model_types) <= 2:
        grid_cols, grid_rows = 2, 1
    else:
        grid_cols, grid_rows = 2, 2
        
    # Calculate the output video size based on grid layout
    # Scale factor to make the individual video tiles smaller
    scale_factor = 0.5 if len(model_types) > 2 else 0.65
    tile_width = int(width * scale_factor)
    tile_height = int(height * scale_factor)
    output_width = tile_width * grid_cols
    output_height = tile_height * grid_rows
    
    # Initialize video writer (with codec fallbacks; never fails silently)
    out = open_video_writer(output_path, fps_target, (output_width, output_height))
    if out is None:
        raise RuntimeError(f"Could not open a video writer for {output_path}")
    
    # Process the video frame by frame
    frame_idx = 0
    comparison_stats = {model_type: {"fps": [], "detections": 0} for model_type in model_types}
    
    try:
        # Use tqdm for progress bar
        with tqdm(total=total_frames, desc="Creating comparison video") as pbar:
            while True:
                # Read a frame
                ret, frame = cap.read()
                if not ret or (max_frames is not None and frame_idx >= max_frames):
                    break
                
                # Process the frame with each model and create a grid
                comparison_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
                
                for i, model_manager in enumerate(model_managers):
                    model_type = model_types[i]
                    
                    # Calculate grid position
                    grid_x = (i % grid_cols) * tile_width
                    grid_y = (i // grid_cols) * tile_height
                    
                    # Process with current model and time it
                    t_start = time.time()
                    detections, _, annotated_frame = model_manager.predict(frame)
                    processing_time = time.time() - t_start
                    
                    # Record stats
                    comparison_stats[model_type]["fps"].append(1.0 / max(processing_time, 0.001))
                    comparison_stats[model_type]["detections"] += len(detections)
                    
                    # Create a clean area for the text at the top of each frame
                    # Draw a dark background for better text visibility
                    cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], 40), (0, 0, 0), -1)
                    
                    # Add model name and FPS to annotated frame with improved visibility
                    fps_text = f"{model_type}: {1.0/processing_time:.1f} FPS | {len(detections)} objects"
                    cv2.putText(annotated_frame, fps_text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Resize the annotated frame to fit the grid
                    resized_frame = cv2.resize(annotated_frame, (tile_width, tile_height))
                    
                    # Place in comparison grid
                    comparison_frame[grid_y:grid_y+tile_height, grid_x:grid_x+tile_width] = resized_frame
                
                # Write the comparison frame
                out.write(comparison_frame)
                
                # Update progress
                frame_idx += 1
                pbar.update(1)
                
                # Show progress in console occasionally
                if frame_idx % 30 == 0:
                    progress = (frame_idx / total_frames) * 100
                    print(f"\rProgress: {progress:.1f}% (Frame {frame_idx}/{total_frames})", end="")
    
    except Exception as e:
        print(f"\nError during video processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Release resources
        cap.release()
        out.release()
        
    # Print comparison statistics
    print("\n\nComparison Statistics:")
    for model_type, stats in comparison_stats.items():
        avg_fps = sum(stats["fps"]) / len(stats["fps"]) if stats["fps"] else 0
        print(f"{model_type}:")
        print(f"  - Average FPS: {avg_fps:.2f}")
        print(f"  - Total detections: {stats['detections']}")
        print(f"  - Average detections per frame: {stats['detections'] / max(frame_idx, 1):.2f}")
    
    print(f"\nComparison video saved to: {output_path}")
    return str(output_path)

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
    parser = argparse.ArgumentParser(description="Create side-by-side model comparison videos")
    
    # Video options
    video_group = parser.add_argument_group("Video Options")
    video_group.add_argument("--video", "-v", type=str,
                       help="Video file path (if not specified, will list available samples)")
    video_group.add_argument("--output", "-o", type=str, 
                       help="Output video file path (if not specified, will use default naming)")
    video_group.add_argument("--max-frames", type=int, default=None,
                       help="Maximum number of frames to process (default: process entire video)")
    video_group.add_argument("--fps", type=int, default=None,
                       help="Target FPS for output video (default: same as input)")
    
    # Model options
    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument("--models", nargs="+", required=True,
                       help="List of models to compare (at least 2)")
    model_group.add_argument("--list-models", action="store_true",
                       help="List available models and exit")
    
    # Detection parameters
    detection_group = parser.add_argument_group("Detection Parameters")
    detection_group.add_argument("--conf-threshold", type=float, default=0.25,
                       help="Confidence threshold for detections (default: 0.25)")
    detection_group.add_argument("--iou-threshold", type=float, default=0.45,
                       help="IoU threshold for NMS (default: 0.45)")
    
    args = parser.parse_args()
    
    # List available models if requested
    if args.list_models:
        print("\nAvailable models:")
        for i, model_key in enumerate(sorted(DEFAULT_MODEL_PATHS.keys()), 1):
            print(f"{i}. {model_key}")
        return
    
    # Ensure we have at least 2 models
    if len(args.models) < 2:
        print("Error: Need at least two models to compare. Use --models to specify multiple models.")
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
            # Check if it's a sample video name
            sample_path = VIDEO_DIR / args.video
            if sample_path.exists():
                video_path = sample_path
            else:
                print(f"Error: Video file not found: {video_path}")
                return
    
    # Create the comparison video
    output_path = create_comparison_video(
        video_path, 
        args.models, 
        args.output, 
        args.max_frames,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        fps_target=args.fps
    )
    
    if output_path:
        print(f"Comparison video creation complete! Video saved to: {output_path}")
    else:
        print("Failed to create comparison video.")

if __name__ == "__main__":
    main()