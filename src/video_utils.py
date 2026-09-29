"""
Video Processing Utilities for Computer Vision Project

This module provides shared utilities for processing video files with computer vision models.
It standardizes the visualization and processing logic across all parts of the application.
"""

import cv2
import numpy as np
import time
import math
from pathlib import Path
import threading

# Fallback when a container reports no/NaN FPS (common for phone/web MP4s).
DEFAULT_FPS = 30.0


def _safe_fps(fps, default=DEFAULT_FPS):
    """Return a usable FPS: the container value when sane, otherwise a default."""
    try:
        fps = float(fps)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(fps) or fps <= 0:
        return default
    return fps


def open_video_writer(output_path, fps, size):
    """Create a VideoWriter with codec fallbacks; returns None if none work.

    Tries H.264 (``avc1``) then ``mp4v`` for .mp4/.mov, and falls back to MJPG in
    an .avi container so output is still produced where H.264 is unavailable.
    The writer is verified with ``isOpened()`` so failures are not silent.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fps = _safe_fps(fps)

    ext = output_path.suffix.lower()
    if ext == ".avi":
        candidates = [("MJPG", output_path)]
    elif ext in (".mp4", ".mov", ".m4v"):
        candidates = [("avc1", output_path), ("mp4v", output_path)]
    else:
        candidates = [("mp4v", output_path)]
    # Final fallback so we always try to produce *some* output.
    candidates.append(("MJPG", output_path.with_suffix(".avi")))

    for fourcc_str, path in candidates:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(str(path), fourcc, fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    return None

def get_video_properties(video_capture):
    """
    Get the properties of a video from a VideoCapture object.
    
    Args:
        video_capture (cv2.VideoCapture): OpenCV VideoCapture object
        
    Returns:
        tuple: (fps, total_frames, width, height)
    """
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    return fps, total_frames, width, height

def process_video_with_model(
    video_path, 
    model_manager, 
    output_path=None, 
    callback=None, 
    stats_callback=None, 
    stop_event=None,
    pause_event=None,
    add_fps=True,
    return_masks=True,
):
    """
    Process a video with the specified model and generate outputs.
    
    Args:
        video_path (str or Path): Path to the input video file
        model_manager: Model manager instance with predict() method
        output_path (str or Path, optional): Path to save the processed video (if None, won't save)
        callback (callable, optional): Callback function for frame updates 
                                       callback(annotated_frame, frame_count, total_frames, model_type)
        stats_callback (callable, optional): Callback function for final statistics
        stop_event (threading.Event, optional): Event to signal process termination
        pause_event (threading.Event, optional): Event to signal process pausing
        add_fps (bool): Whether to add FPS counter to video
        return_masks (bool): Whether to compute segmentation masks (skip for speed
            when only the annotated video is needed)
        
    Returns:
        dict: Processing statistics
    """
    # Initialize statistics
    stats = {
        "frames_processed": 0,
        "total_frames": 0,
        "processing_time": 0.0,
        "actual_fps": 0.0,
        "detections": {},
        "model_type": model_manager.model_type,
        "model_device": model_manager.device
    }
    
    # Open the video file
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Cannot open video: {video_path}")
        return stats
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = _safe_fps(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stats["total_frames"] = total_frames

    # Initialize video writer if output path provided
    out = None
    if output_path:
        out = open_video_writer(output_path, fps, (width, height))
        if out is None:
            print(f"Warning: could not open a video writer for {output_path}; output will not be saved.")
    
    # Initialize counters
    frame_count = 0
    processing_errors = 0
    start_time = time.time()
    
    try:
        # Process each frame
        while cap.isOpened():
            # Check if we should stop
            if stop_event and stop_event.is_set():
                print("Stopping video processing due to stop event")
                break
                
            # Check if we should pause
            if pause_event and pause_event.is_set():
                time.sleep(0.1)  # Sleep briefly to avoid CPU spinning
                continue
            
            # Read the next frame
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            current_progress_percentage = (frame_count / total_frames * 100) if total_frames > 0 else 0

            try:
                # Process frame based on model type
                if model_manager.model_type == "sam":
                    # For SAM, create a grid of prompt points
                    h, w = frame.shape[:2]
                    grid_size = 3
                    y_points = np.linspace(h // 4, 3 * h // 4, grid_size, dtype=int)
                    x_points = np.linspace(w // 4, 3 * w // 4, grid_size, dtype=int)
                    input_points = np.array([[x, y] for y in y_points for x in x_points])
                    input_labels = np.ones(len(input_points), dtype=np.int32)
                    
                    # Run prediction with points and labels
                    results = model_manager.predict(frame, input_points=input_points, input_labels=input_labels)
                else:
                    # For other models (YOLO-Seg, Mask R-CNN)
                    results = model_manager.predict(frame, return_masks=return_masks)
                    
                # Unpack results
                if results and len(results) == 3:
                    detections, segmentations, annotated_frame = results
                    
                    # Calculate FPS for the standardized layout
                    elapsed = time.time() - start_time
                    current_fps = frame_count / elapsed if elapsed > 0 else 0
                    
                    # Apply standardized layout to the annotated frame with FPS info
                    annotated_frame = apply_standardized_layout(
                        frame=annotated_frame,
                        detections=detections,
                        model_type=model_manager.model_type,
                        frame_count=frame_count,
                        total_frames=total_frames,
                        width=width,
                        height=height,
                        fps=current_fps if add_fps else None  # Pass FPS to the layout function
                    )
                    
                    # Count detections by class for statistics
                    for det in detections:
                        class_name = det.get('class_name', 'unknown')
                        stats["detections"][class_name] = stats["detections"].get(class_name, 0) + 1
                    
                    # Write to output video if enabled
                    if out:
                        out.write(annotated_frame)
                    
                    # Update via callback
                    if callback:
                        # current_progress_percentage is already calculated above
                        callback(annotated_frame, frame_count, total_frames, model_manager.model_type, progress=current_progress_percentage)
                else:
                    # Handle invalid results
                    msg = "Model returned unexpected results format"
                    print(f"Error: {msg}")
                    annotated_frame = frame.copy()
                    cv2.putText(annotated_frame, msg, (30, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    if out:
                        out.write(annotated_frame)
                    
                    if callback:
                        # current_progress_percentage is already calculated above
                        callback(annotated_frame, frame_count, total_frames, model_manager.model_type, progress=current_progress_percentage)
                        
            except Exception as e:
                processing_errors += 1
                print(f"Error processing frame {frame_count}: {e}")
                import traceback
                traceback.print_exc()
                
                # Show the error frame
                error_frame = frame.copy()
                cv2.putText(error_frame, f"Error: {str(e)[:50]}", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                if out:
                    out.write(error_frame)
                    
                if callback:
                    # current_progress_percentage is already calculated above
                    callback(error_frame, frame_count, total_frames, model_manager.model_type, progress=current_progress_percentage)

    except Exception as e:
        print(f"Fatal error in video processing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Update statistics
        stats["frames_processed"] = frame_count
        stats["processing_time"] = time.time() - start_time
        stats["actual_fps"] = frame_count / stats["processing_time"] if stats["processing_time"] > 0 else 0

        # Clean up resources (always)
        cap.release()
        if out:
            out.release()

    # Provide final statistics
    if stats_callback:
        stats_callback(stats)
    return stats


def apply_standardized_layout(frame, detections, model_type, frame_count, total_frames, width, height, fps=None):
    """
    Apply a standardized layout to all frames regardless of model type
    
    Args:
        frame (np.ndarray): The frame to apply layout to
        detections (list): List of detection dictionaries
        model_type (str): Type of model being used
        frame_count (int): Current frame number
        total_frames (int): Total number of frames
        width (int): Frame width
        height (int): Frame height
        fps (float, optional): Frames per second to display, if None, FPS is not added
        
    Returns:
        np.ndarray: Frame with standardized layout applied
    """
    # Create a copy to work with
    result_frame = frame.copy()
    
    # 1. Add a consistent header bar
    header_height = 40
    cv2.rectangle(result_frame, (0, 0), (width, header_height), (0, 0, 0), -1)
    
    # Header text
    header_text = f"{model_type.upper()} | Frame: {frame_count}/{total_frames}"
    if fps is not None:
        header_text += f" | FPS: {fps:.1f}"
    
    cv2.putText(result_frame, header_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 2. Add consistent detection count display
    if detections:
        det_count = len(detections)
        class_counts = {}
        for det in detections:
            class_name = det.get('class_name', 'unknown')
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # List top classes with counts
        y_pos = header_height + 30
        cv2.putText(result_frame, f"Detections: {det_count}", 
                    (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        for i, (cls, count) in enumerate(sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:5]):
            y_pos += 25
            cv2.putText(result_frame, f"{cls}: {count}", 
                        (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(result_frame, "No detections", 
                    (10, header_height + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # 3. Ensure all bounding boxes are visible and properly labeled
    for det in detections:
        class_name = det.get('class_name', 'unknown')
        confidence = det.get('confidence', det.get('score', 0.0))
        bbox = det.get('bbox', det.get('box'))
        
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = [int(b) for b in bbox]
            
            # Draw consistent box styles
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw background for text
            text = f"{class_name} {confidence:.2f}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(result_frame, (x1, y1 - 20), (x1 + text_size[0] + 10, y1), (0, 255, 0), -1)
            
            # Draw text
            cv2.putText(result_frame, text, (x1 + 5, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return result_frame


def process_webcam_with_model(
    model_manager, 
    callback=None, 
    stats_callback=None, 
    stop_event=None,
    pause_event=None,
    camera_id=0,
    process_width=640,
    process_height=480,
    frame_skip=2
):
    """
    Process webcam input with the specified model in real-time.
    
    Args:
        model_manager: Model manager instance with predict() method
        callback (callable, optional): Callback function for frame updates
        stats_callback (callable, optional): Callback function for final statistics
        stop_event (threading.Event, optional): Event to signal process termination
        pause_event (threading.Event, optional): Event to signal process pausing
        camera_id (int): Camera device ID
        process_width (int): Width for processing (resize for performance)
        process_height (int): Height for processing (resize for performance)
        frame_skip (int): Process every Nth frame
        
    Returns:
        dict: Processing statistics
    """
    # Initialize statistics
    stats = {
        "frames_processed": 0,
        "total_frames": 0,
        "processing_time": 0.0,
        "actual_fps": 0.0,
        "detections": {},
        "model_type": model_manager.model_type,
        "model_device": model_manager.device
    }
    
    # Open webcam
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Cannot open webcam with ID {camera_id}")
        return stats
    
    # Get webcam properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Performance optimizations
    webcam_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    target_display_fps = min(15, webcam_fps / frame_skip)
    frame_interval_ms = int(1000 / target_display_fps) if target_display_fps > 0 else 100
    
    # Initialize counters
    frame_count = 0
    total_frames = 0
    processing_errors = 0
    start_time = time.time()
    last_update_time = 0
    
    try:
        # Process frames
        while cap.isOpened():
            # Check if we should stop
            if stop_event and stop_event.is_set():
                print("Stopping webcam processing due to stop event")
                break
                
            # Check if we should pause
            if pause_event and pause_event.is_set():
                time.sleep(0.1)  # Sleep briefly to avoid CPU spinning
                continue
            
            # Read the next frame
            ret, frame = cap.read()
            if not ret:
                print("Error reading from webcam")
                break
                
            total_frames += 1
            
            # Apply frame skipping
            if total_frames % frame_skip != 0:
                continue
            
            frame_count += 1
            current_time = int(time.time() * 1000)
            
            try:
                # Resize frame for processing
                frame_resized = cv2.resize(frame, (process_width, process_height))
                
                # Process frame based on model type
                if model_manager.model_type == "sam":
                    # For SAM, create a grid of prompt points
                    h, w = frame_resized.shape[:2]
                    grid_size = 3
                    y_points = np.linspace(h // 4, 3 * h // 4, grid_size, dtype=int)
                    x_points = np.linspace(w // 4, 3 * w // 4, grid_size, dtype=int)
                    input_points = np.array([[x, y] for y in y_points for x in x_points])
                    input_labels = np.ones(len(input_points), dtype=np.int32)
                    
                    # Run prediction with points and labels
                    results = model_manager.predict(frame_resized, input_points=input_points, input_labels=input_labels)
                else:
                    # For other models (YOLO-Seg, Mask R-CNN)
                    results = model_manager.predict(frame_resized)
                
                # Unpack results
                if results and len(results) == 3:
                    detections, segmentations, annotated_frame = results
                    
                    # Calculate FPS for display
                    elapsed = time.time() - start_time
                    current_fps = frame_count / elapsed if elapsed > 0 else 0
                    
                    # Apply standardized layout to the annotated frame
                    annotated_frame = apply_standardized_layout(
                        frame=annotated_frame,
                        detections=detections,
                        model_type=model_manager.model_type,
                        frame_count=frame_count,
                        total_frames=total_frames,
                        width=process_width,
                        height=process_height,
                        fps=current_fps  # Pass FPS to the layout function
                    )
                    
                    # Count detections by class for statistics
                    for det in detections:
                        class_name = det.get('class_name', 'Unknown')
                        if class_name != 'Unknown':
                            stats["detections"][class_name] = stats["detections"].get(class_name, 0) + 1
                    
                    # Limit display update rate
                    if (current_time - last_update_time) >= frame_interval_ms:
                        # Update via callback
                        if callback:
                            callback(annotated_frame, frame_count, total_frames, model_manager.model_type)
                        
                        last_update_time = current_time
                else:
                    # Handle invalid results if update interval passed
                    if (current_time - last_update_time) >= frame_interval_ms:
                        msg = "Model returned unexpected results format"
                        print(f"Error: {msg}")
                        annotated_frame = frame_resized.copy()
                        cv2.putText(annotated_frame, msg, (30, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        
                        if callback:
                            callback(annotated_frame, frame_count, total_frames, model_manager.model_type)
                        
                        last_update_time = current_time
            
            except Exception as e:
                processing_errors += 1
                print(f"Error processing webcam frame {frame_count}: {e}")
                
                # Only show errors occasionally to avoid spamming
                if processing_errors <= 3 or processing_errors % 50 == 0:
                    if (current_time - last_update_time) >= frame_interval_ms:
                        error_frame = frame.copy()
                        cv2.putText(error_frame, f"Error: {str(e)[:50]}", (30, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        if callback:
                            callback(error_frame, frame_count, total_frames, model_manager.model_type)
                        
                        last_update_time = current_time
    
    except Exception as e:
        print(f"Fatal error in webcam processing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Update statistics
        stats["frames_processed"] = frame_count
        stats["total_frames"] = total_frames
        stats["processing_time"] = time.time() - start_time
        stats["actual_fps"] = frame_count / stats["processing_time"] if stats["processing_time"] > 0 else 0

        # Clean up resources
        cap.release()

    # Provide final statistics
    if stats_callback:
        stats_callback(stats)
    return stats