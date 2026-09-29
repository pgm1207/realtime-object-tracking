"""
End-to-End Pipeline for Computer Vision Project

This script provides a complete pipeline that:
1. Evaluates multiple object detection and segmentation models
2. Generates performance metrics and visualizations
3. Creates a demo video using the best performing model

This allows running the entire Task #2 solution from a single command.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
import glob

# Add the parent directory to sys.path for proper imports
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import project components - use relative imports when running from src directory
try:
    # Try relative imports first (when running from within src directory)
    from evaluate_models import ModelEvaluator, COCO_VAL_TOTAL_IMAGES
    from create_demo_video import create_demo_video
    from models import DEFAULT_MODEL_PATHS
except ImportError:
    # Fall back to absolute imports (when running from project root)
    from src.evaluate_models import ModelEvaluator, COCO_VAL_TOTAL_IMAGES
    from src.create_demo_video import create_demo_video
    from src.models import DEFAULT_MODEL_PATHS

# Configure paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "inference" / "results"
VIDEO_DIR = PROJECT_ROOT / "data_sets" / "video_data"
OUTPUT_DIR = PROJECT_ROOT / "inference" / "output_videos"

# Create directories if they don't exist
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def find_latest_results():
    """Find the latest evaluation results file"""
    result_files = list(RESULTS_DIR.glob("evaluation_results_*.json"))
    if not result_files:
        return None
    
    # Sort by modification time (newest first)
    latest_file = max(result_files, key=lambda f: f.stat().st_mtime)
    return latest_file

def determine_best_model(results):
    """
    Determine the best model from evaluation results
    
    Args:
        results (dict): Evaluation results with metrics for each model
        
    Returns:
        tuple: (best_model_name, selection_details) or (None, None) if no valid results
    """
    if not results:
        return None, None
    
    # Initialize scores dictionary
    scores = {}
    
    # Weight factors for different metrics (higher = more important)
    WEIGHTS = {
        "mAP": 0.4,        # Mean Average Precision (main accuracy metric)
        "AP50": 0.2,       # AP at IoU=0.50 (standard detection metric)
        "AP75": 0.1,       # AP at IoU=0.75 (higher quality detections)
        "FPS": 0.3         # Frames per second (performance metric)
    }
    
    # Calculate weighted scores for each model
    for model_name, metrics in results.items():
        # Skip if no COCO metrics (evaluation failed)
        if "coco_metrics" not in metrics:
            continue
            
        coco_metrics = metrics["coco_metrics"]
        fps = metrics.get("fps", 0)
        
        # Get relevant metrics (with fallbacks if missing)
        mAP = coco_metrics.get("AP_IoU=0.50:0.95", 0)
        AP50 = coco_metrics.get("AP_IoU=0.50", 0)
        AP75 = coco_metrics.get("AP_IoU=0.75", 0)
        
        # Calculate weighted score
        weighted_score = (
            mAP * WEIGHTS["mAP"] +
            AP50 * WEIGHTS["AP50"] +
            AP75 * WEIGHTS["AP75"] +
            min(fps / 30.0, 1.0) * WEIGHTS["FPS"]  # Normalize FPS (cap at 30 FPS)
        )
        
        scores[model_name] = {
            "mAP": mAP,
            "AP50": AP50,
            "AP75": AP75,
            "FPS": fps,
            "weighted_score": weighted_score
        }
    
    # Find model with highest weighted score
    if not scores:
        return None, None
        
    best_model = max(scores.items(), key=lambda x: x[1]["weighted_score"])
    best_model_name = best_model[0]
    selection_details = best_model[1]
    
    return best_model_name, selection_details

def run_pipeline(args):
    """
    Run the complete pipeline: evaluation, visualization, and demo video creation
    
    Args:
        args: Command-line arguments
    """
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 80)
    print(f"COMPUTER VISION PIPELINE - Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Create output directory if specified
    output_dir = args.output_dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output will be saved to: {output_dir}")
    
    # STEP 1: Evaluate models
    print("\n--- STEP 1: MODEL EVALUATION ---")
    
    # Determine which models to evaluate
    if args.models:
        # Use specified models (up to 3)
        models_to_evaluate = args.models[:3]
        print(f"Evaluating specified models: {', '.join(models_to_evaluate)}")
    else:
        # Default to top 3 models by size for balanced evaluation
        default_models = ["yolov8n-seg", "yolov8m-seg", "yolov8x-seg"]
        models_to_evaluate = default_models
        print(f"Evaluating default models: {', '.join(models_to_evaluate)}")
    
    # Prepare model configs
    model_configs = []
    for model_name in models_to_evaluate:
        if model_name in DEFAULT_MODEL_PATHS:
            model_configs.append({
                "type": model_name,
                "path": DEFAULT_MODEL_PATHS[model_name],
                "config": None,
                "conf_threshold": args.conf_threshold,
                "iou_threshold": args.iou_threshold
            })
        else:
            print(f"Warning: Model {model_name} not found in default models. Skipping.")
    
    if not model_configs:
        print("Error: No valid models to evaluate.")
        return False
    
    # Create evaluator and run evaluation
    max_images = min(max(1, args.images), COCO_VAL_TOTAL_IMAGES)
    evaluator = ModelEvaluator(model_configs)
    
    # Define progress reporting callback
    def progress_callback(model_type, current, total):
        percent = (current / total) * 100
        print(f"Evaluating {model_type}: {current}/{total} images ({percent:.1f}%)")
    
    # Run the evaluation
    print(f"Evaluating models on {max_images} COCO validation images...")
    evaluator.run_evaluation(max_images, save_visualizations=True, progress_callback=progress_callback)
    
    # Save results path for later steps
    results_file = find_latest_results()
    if not results_file:
        print("Error: Evaluation results not found.")
        return False
    
    print(f"Evaluation complete. Results saved to: {results_file}")
    evaluation_time = time.time() - start_time
    print(f"Evaluation took {evaluation_time:.1f} seconds")
    
    # STEP 2: Generate metrics dashboard
    print("\n--- STEP 2: METRICS VISUALIZATION ---")
    
    try:
        # Try relative imports first (when running from within src directory)
        try:
            from metrics_visualizer import MetricsVisualizer
        except ImportError:
            # Fall back to absolute imports (when running from project root)
            from src.metrics_visualizer import MetricsVisualizer
        
        # Create visualizer with latest results
        visualizer = MetricsVisualizer(results_file=str(results_file))
        
        # Generate dashboard
        dashboard_path = visualizer.create_metrics_dashboard(show_plot=args.show_plots)
        if dashboard_path:
            print(f"Metrics dashboard generated at: {dashboard_path}")
            
            # Copy to output directory if specified
            if output_dir:
                import shutil
                output_dashboard = Path(output_dir) / f"metrics_dashboard_{timestamp}.png"
                shutil.copy(dashboard_path, output_dashboard)
                print(f"Dashboard copied to output directory: {output_dashboard}")
        else:
            print("Warning: Failed to generate metrics dashboard.")
        
        # Generate precision-recall curves
        pr_path = visualizer.plot_precision_recall_curves(show_plot=args.show_plots)
        if pr_path:
            print(f"Precision-recall curves generated at: {pr_path}")
            
            # Copy to output directory if specified
            if output_dir:
                import shutil
                output_pr = Path(output_dir) / f"pr_curves_{timestamp}.png"
                shutil.copy(pr_path, output_pr)
        
    except ImportError as e:
        print(f"Warning: Could not import metrics visualizer: {e}")
        print("Skipping dashboard generation...")
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()
    
    # STEP 3: Create demo video with best model
    if args.demo_video:
        print("\n--- STEP 3: DEMO VIDEO CREATION ---")
        
        # Load evaluation results to find best model
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            # Determine best model
            best_model, details = determine_best_model(results)
            
            if best_model:
                print(f"Best model from evaluation: {best_model}")
                if details:
                    print("Performance metrics:")
                    for key, value in details.items():
                        if key != "weighted_score":
                            print(f"- {key}: {value:.4f}")
                    print(f"- Combined score: {details['weighted_score']:.4f}")
                
                # Process video path
                video_path = args.demo_video
                if not os.path.exists(video_path):
                    # Check if it's a sample video name
                    sample_path = VIDEO_DIR / video_path
                    if sample_path.exists():
                        video_path = sample_path
                    else:
                        print(f"Error: Video file not found: {video_path}")
                        return False
                
                # Set output path
                output_video = None
                if output_dir:
                    output_video = Path(output_dir) / f"demo_video_{best_model}_{timestamp}.mp4"
                
                # Create demo video
                output_path = create_demo_video(
                    best_model,
                    video_path,
                    output_path=output_video,
                    conf_threshold=args.conf_threshold,
                    iou_threshold=args.iou_threshold
                )
                
                if output_path:
                    print(f"Demo video created successfully: {output_path}")
                else:
                    print("Failed to create demo video.")
            else:
                print("Could not determine best model from evaluation results.")
                return False
                
        except Exception as e:
            print(f"Error creating demo video: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Pipeline complete
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETED in {total_time:.1f} seconds")
    print("=" * 80)
    
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run complete Computer Vision pipeline")
    
    # Evaluation options
    eval_group = parser.add_argument_group("Evaluation Options")
    eval_group.add_argument("--images", type=int, default=50,
                        help=f"Number of COCO validation images to use (max {COCO_VAL_TOTAL_IMAGES}, default: 50)")
    eval_group.add_argument("--models", type=str, nargs="+",
                        help="Specific models to evaluate (default: yolov8n-seg, yolov8m-seg, yolov8x-seg)")
    
    # Demo video options
    video_group = parser.add_argument_group("Demo Video Options")
    video_group.add_argument("--demo-video", type=str,
                        help="Path to video for creating demo with best model")
    
    # Detection parameters
    detection_group = parser.add_argument_group("Detection Parameters")
    detection_group.add_argument("--conf-threshold", type=float, default=0.25,
                       help="Confidence threshold for detections (default: 0.25)")
    detection_group.add_argument("--iou-threshold", type=float, default=0.45,
                       help="IoU threshold for NMS (default: 0.45)")
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument("--output-dir", type=str,
                        help="Directory to save all outputs (creates if not exists)")
    output_group.add_argument("--show-plots", action="store_true",
                        help="Show visualization plots interactively")
    
    args = parser.parse_args()
    
    # Run the pipeline
    success = run_pipeline(args)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()