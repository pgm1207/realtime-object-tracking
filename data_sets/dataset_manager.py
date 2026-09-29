import os
import sys
import json
import random
import zipfile
import requests
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict, Any, Callable

# Version identifier to verify we're running the correct file
VERSION = "1.4.1"  # Added better directory handling and corrupt file recovery

# Constants for URLs
COCO_BASE_URL = "http://images.cocodataset.org/"
COCO_VAL_IMG_URL = f"{COCO_BASE_URL}zips/val2017.zip"
COCO_ANN_URL = f"{COCO_BASE_URL}annotations/annotations_trainval2017.zip"

# Define progress stages for tracking overall progress
PROGRESS_STAGES = {
    "DOWNLOAD_ANNOTATIONS": {"weight": 0.2, "description": "Downloading annotations"},
    "EXTRACT_ANNOTATIONS": {"weight": 0.1, "description": "Extracting annotations"},
    "DOWNLOAD_IMAGES": {"weight": 0.3, "description": "Downloading images"},
    "EXTRACT_IMAGES": {"weight": 0.2, "description": "Extracting images"},
    "CREATE_SUBSET": {"weight": 0.2, "description": "Processing subset"}
}

class ProgressTracker:
    """Helper class to track multi-stage progress and report to callback functions"""
    
    def __init__(self, stages=None, callback: Optional[Callable]=None):
        """
        Initialize the progress tracker
        
        Args:
            stages (dict): Dictionary mapping stage names to their relative weights
            callback (callable): Function to call with progress updates
        """
        self.stages = stages or PROGRESS_STAGES
        self.callback = callback
        self.current_stage = None
        self.stage_progress = 0.0
        self.overall_progress = 0.0
        
    def update_stage(self, stage_name: str, stage_progress: float=0.0, stage_message: Optional[str]=None):
        """
        Update the current stage and progress
        
        Args:
            stage_name (str): Name of the current stage
            stage_progress (float): Progress within the current stage (0.0-1.0)
            stage_message (str, optional): Optional message to display
        """
        if stage_name not in self.stages:
            print(f"Warning: Unknown stage {stage_name}")
            return
            
        self.current_stage = stage_name
        self.stage_progress = min(max(0.0, stage_progress), 1.0)
        
        # Calculate overall progress based on stage weights
        total_progress = 0.0
        total_weight = sum(stage["weight"] for stage in self.stages.values())
        
        # Add progress from completed stages
        completed = False
        for name, info in self.stages.items():
            weight = info["weight"] / total_weight if total_weight > 0 else 0
            
            if completed:
                # Future stages have 0 progress
                continue
            elif name == stage_name:
                # Current stage has partial progress
                total_progress += weight * self.stage_progress
                completed = True
            else:
                # Past stages have 100% progress
                total_progress += weight
                
        self.overall_progress = total_progress
        
        # Prepare message
        if stage_message is None:
            stage_info = self.stages.get(stage_name, {})
            stage_message = stage_info.get("description", stage_name)
            if stage_progress > 0:
                stage_message = f"{stage_message} ({stage_progress*100:.1f}%)"
        
        # Call the callback with updated progress
        if self.callback:
            try:
                self.callback(
                    stage=stage_name,
                    stage_progress=self.stage_progress,
                    overall_progress=self.overall_progress,
                    message=stage_message
                )
            except Exception as e:
                print(f"Error in progress callback: {e}")
                
    def get_progress(self):
        """Get the current overall progress (0.0-1.0)"""
        return self.overall_progress

class DatasetManager:
    """Smart manager for COCO dataset"""

    def __init__(self, progress_callback: Optional[Callable]=None):
        """
        Initialize the dataset manager
        
        Args:
            progress_callback (callable, optional): Function to call with progress updates.
                Will be called with kwargs: stage, stage_progress, overall_progress, message
        """
        self.base_dir: Path = Path(__file__).parent
        self.image_dir: Path = self.base_dir / "image_data"
        self.coco_dir: Path = self.image_dir / "coco"
        self.coco_val_dir: Path = self.coco_dir / "val2017"
        self.coco_ann_dir: Path = self.coco_dir / "annotations"
        
        # Create base directories if they don't exist
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.coco_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up progress tracking
        self.progress = ProgressTracker(callback=progress_callback)

    def download_file(self, url: str, destination: Path, headers: Optional[Dict[str, str]] = None,
                     progress_stage: Optional[str] = None, retries: int = 3) -> bool:
        """
        Download file with progress bar and retry mechanism
        
        Args:
            url: URL to download from
            destination: Path to save file to
            headers: Request headers
            progress_stage: Name of the current progress stage for tracking
            retries: Number of download attempts before giving up
        """
        if headers is None:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

        # Ensure the parent directory exists
        destination.parent.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {os.path.basename(destination)} from {url}...")
        
        # Update initial progress for this stage
        if progress_stage:
            self.progress.update_stage(progress_stage, 0.0, f"Starting download of {os.path.basename(destination)}")
        
        # Define the temp path up front so the error handlers below can always
        # reference it, even if the request itself fails.
        temp_dest = destination.with_suffix('.tmp')

        for attempt in range(retries):
            try:
                response = requests.get(url, stream=True, headers=headers, timeout=60) # Added timeout
                response.raise_for_status()  # Raise an exception for 4XX/5XX responses

                total_size = int(response.headers.get('content-length', 0))
                downloaded_size = 0
                block_size = 1024 * 8 # Increased block size for potentially faster downloads

                with open(temp_dest, 'wb') as f, tqdm(
                        desc=os.path.basename(destination),
                        total=total_size,
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024,
                        miniters=1, # Update progress bar more frequently
                    ) as bar:
                    for data in response.iter_content(block_size):
                        downloaded_size += len(data)
                        bar.update(len(data))
                        f.write(data)
                        
                        # Update progress every ~1% or at least every 10 blocks
                        if progress_stage and total_size > 0:
                            progress = downloaded_size / total_size
                            # Only update every ~1% to avoid excessive callbacks
                            if int(progress * 100) % 1 == 0:
                                self.progress.update_stage(
                                    progress_stage, 
                                    progress, 
                                    f"Downloading {os.path.basename(destination)}: {progress*100:.1f}%"
                                )

                # Verify the file was downloaded correctly (basic size check)
                downloaded_size = temp_dest.stat().st_size
                if total_size != 0 and downloaded_size < total_size:
                    print(f"Warning: Downloaded size ({downloaded_size} B) is less than expected size ({total_size} B) for {destination}")
                    if attempt < retries - 1:
                        print(f"Attempting download again ({attempt + 2}/{retries})")
                        continue
                elif downloaded_size < 1000: # Less than 1KB is suspicious
                    print(f"Warning: {destination} seems small ({downloaded_size} B), download might be incomplete or invalid.")
                    if attempt < retries - 1:
                        print(f"Attempting download again ({attempt + 2}/{retries})")
                        continue

                # Move temp file to final destination
                if temp_dest.exists():
                    # On Windows, we need to remove the destination first if it exists
                    if destination.exists():
                        destination.unlink()
                    temp_dest.rename(destination)

                # Final progress update for this stage
                if progress_stage:
                    self.progress.update_stage(
                        progress_stage, 
                        1.0, 
                        f"Download complete: {os.path.basename(destination)}"
                    )
                    
                print(f"Successfully downloaded {os.path.basename(destination)}")
                return True
                
            except requests.exceptions.RequestException as e:
                print(f"Error downloading {url} (attempt {attempt + 1}/{retries}): {e}")
                # Remove the potentially corrupt file
                if temp_dest.exists():
                    temp_dest.unlink()
                
                if attempt >= retries - 1:
                    # Update progress with error
                    if progress_stage:
                        self.progress.update_stage(
                            progress_stage,
                            0.0, 
                            f"Error downloading {os.path.basename(destination)}: {str(e)}"
                        )
                        
                    return False
            except Exception as e:
                print(f"An unexpected error occurred during download of {url} (attempt {attempt + 1}/{retries}): {e}")
                if temp_dest.exists():
                    temp_dest.unlink()
                
                if attempt >= retries - 1:
                    # Update progress with error
                    if progress_stage:
                        self.progress.update_stage(
                            progress_stage,
                            0.0, 
                            f"Error: {str(e)}"
                        )
                        
                    return False
                
        return False # If we get here, all retries failed

    def _extract_zip(self, zip_path: Path, extract_to: Path, description: str = "Extracting",
                   progress_stage: Optional[str] = None, retries: int = 2):
        """
        Helper to extract zip files with progress.
        
        Args:
            zip_path: Path to zip file
            extract_to: Directory to extract to
            description: Description for console output
            progress_stage: Name of the current progress stage for tracking
            retries: Number of extraction attempts before giving up
        """
        print(f"{description} {zip_path.name} to {extract_to}...")
        
        # Make sure the extraction directory exists
        extract_to.mkdir(parents=True, exist_ok=True)
        
        # Update initial progress for this stage
        if progress_stage:
            self.progress.update_stage(progress_stage, 0.0, f"Starting extraction of {zip_path.name}")
        
        for attempt in range(retries):
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    # Count total files for progress tracking
                    total_files = len(zip_ref.namelist())
                    extracted_files = 0
                    
                    # Extract with progress tracking
                    for file in tqdm(zip_ref.namelist(), total=total_files, desc=f"Extracting {zip_path.name}"):
                        try:
                            zip_ref.extract(file, extract_to)
                            extracted_files += 1
                            
                            # Update progress approximately every 2%
                            if progress_stage and total_files > 0 and extracted_files % max(1, total_files // 50) == 0:
                                progress = extracted_files / total_files
                                self.progress.update_stage(
                                    progress_stage, 
                                    progress, 
                                    f"Extracting {zip_path.name}: {progress*100:.1f}%"
                                )
                        except Exception as e:
                            print(f"Error extracting file {file}: {e}")
                            # Continue with other files instead of failing completely
                
                # Final progress update for this stage
                if progress_stage:
                    self.progress.update_stage(
                        progress_stage, 
                        1.0, 
                        f"Extraction complete: {zip_path.name}"
                    )
                    
                print(f"Successfully extracted {zip_path.name}.")
                
                # If extraction completed, verify that expected directories exist
                if "annotations" in zip_path.name:
                    # Verify that annotations directory exists and contains files
                    annotations_dir = extract_to / "annotations"
                    if not annotations_dir.exists() or not any(annotations_dir.iterdir()):
                        if attempt < retries - 1:
                            print(f"Annotations directory missing or empty after extraction. Retry {attempt+2}/{retries}")
                            continue
                        else:
                            # Create empty directories to avoid errors
                            annotations_dir.mkdir(parents=True, exist_ok=True)
                            print("Warning: Creating empty annotations directory structure")
                
                if "val2017" in zip_path.name:
                    # Verify val2017 directory exists and contains files
                    val_dir = extract_to / "val2017"
                    if not val_dir.exists() or not any(val_dir.iterdir()):
                        if attempt < retries - 1:
                            print(f"val2017 directory missing or empty after extraction. Retry {attempt+2}/{retries}")
                            continue
                        else:
                            # Create empty directories to avoid errors
                            val_dir.mkdir(parents=True, exist_ok=True)
                            print("Warning: Creating empty val2017 directory structure")
                
                return True
                
            except zipfile.BadZipFile:
                print(f"Error: {zip_path} is not a valid zip file or is corrupted.")
                
                # Try to download again if this is not the last attempt
                if attempt < retries - 1:
                    print(f"Attempting to re-download the file (attempt {attempt+2}/{retries})...")
                    
                    # Determine which URL to use for re-downloading
                    if "annotations" in zip_path.name:
                        url = COCO_ANN_URL
                        stage = "DOWNLOAD_ANNOTATIONS"
                    else:
                        url = COCO_VAL_IMG_URL
                        stage = "DOWNLOAD_IMAGES"
                    
                    # Remove corrupted zip file
                    if zip_path.exists():
                        zip_path.unlink()
                    
                    # Try to re-download
                    if not self.download_file(url, zip_path, progress_stage=stage):
                        print(f"Re-download failed. Cannot proceed with extraction.")
                        # Update progress with error
                        if progress_stage:
                            self.progress.update_stage(
                                progress_stage,
                                0.0, 
                                f"Error: Failed to re-download {zip_path.name}"
                            )
                        continue
                
                else:
                    # Update progress with error
                    if progress_stage:
                        self.progress.update_stage(
                            progress_stage,
                            0.0, 
                            f"Error: {zip_path.name} is not a valid zip file"
                        )
                    
                    # Create necessary directories to avoid subsequent errors
                    if "annotations" in zip_path.name:
                        self.coco_ann_dir.mkdir(parents=True, exist_ok=True)
                    if "val2017" in zip_path.name:
                        self.coco_val_dir.mkdir(parents=True, exist_ok=True)
                    
                    return False
            except Exception as e:
                print(f"Error extracting {zip_path}: {e}")
                
                if attempt < retries - 1:
                    print(f"Retrying extraction ({attempt+2}/{retries})...")
                else:
                    # Update progress with error
                    if progress_stage:
                        self.progress.update_stage(
                            progress_stage,
                            0.0, 
                            f"Error extracting {zip_path.name}: {str(e)}"
                        )
                    
                    # Create necessary directories to avoid subsequent errors
                    if "annotations" in zip_path.name:
                        self.coco_ann_dir.mkdir(parents=True, exist_ok=True)
                    if "val2017" in zip_path.name:
                        self.coco_val_dir.mkdir(parents=True, exist_ok=True)
                    
                    return False
        
        # If we get here, all retries failed
        return False

    def _create_subset(self, subset_size: int):
        """
        Creates a subset of the COCO validation dataset.
        
        Args:
            subset_size: Number of images to include in subset
        """
        print(f"Creating COCO subset with {subset_size} images...")
        
        # Update initial progress for subset creation
        self.progress.update_stage("CREATE_SUBSET", 0.0, "Preparing to create COCO subset")
        
        # Ensure directories exist
        self.coco_val_dir.mkdir(parents=True, exist_ok=True)
        self.coco_ann_dir.mkdir(parents=True, exist_ok=True)

        # Define annotation paths
        full_ann_file = self.coco_ann_dir / "instances_val2017.json"
        subset_ann_file_temp = self.coco_ann_dir / "instances_val2017_subset.json"
        original_ann_backup = self.coco_ann_dir / "instances_val2017_original.json"

        if not full_ann_file.exists():
             # Check if original backup exists from a previous subset operation
            if original_ann_backup.exists():
                print("Using backed-up original annotations file.")
                shutil.copy2(original_ann_backup, full_ann_file)
                self.progress.update_stage("CREATE_SUBSET", 0.1, "Using backup annotations file")
            else:
                print(f"Error: Full annotation file not found at {full_ann_file} and no backup exists.")
                # Create a minimal empty annotation file to avoid errors
                with open(full_ann_file, 'w') as f:
                    json.dump({
                        'info': {},
                        'licenses': [],
                        'categories': [],
                        'images': [],
                        'annotations': []
                    }, f)
                print("Created empty annotation file as fallback")
                self.progress.update_stage("CREATE_SUBSET", 0.0, "Created empty annotation file as fallback")
                return False

        try:
            with open(full_ann_file, 'r') as f:
                data = json.load(f)
            self.progress.update_stage("CREATE_SUBSET", 0.2, "Loaded annotation file")
        except Exception as e:
            print(f"Error reading annotation file {full_ann_file}: {e}")
            self.progress.update_stage("CREATE_SUBSET", 0.0, f"Error reading annotation file: {str(e)}")
            return False

        # Select random images
        all_image_ids = [img['id'] for img in data['images']]
        if subset_size >= len(all_image_ids):
            print("Subset size requested is >= total images. Using all validation images.")
            selected_ids = set(all_image_ids)
            subset_size = len(all_image_ids) # Adjust size for messages
        else:
            selected_ids = set(random.sample(all_image_ids, subset_size))
        
        self.progress.update_stage("CREATE_SUBSET", 0.3, f"Selected {len(selected_ids)} images")

        # Create subset annotations data
        subset_data = {
            'info': data.get('info', {}),
            'licenses': data.get('licenses', []),
            'categories': data.get('categories', []),
            'images': [img for img in data['images'] if img['id'] in selected_ids],
            'annotations': [ann for ann in data.get('annotations', []) if ann['image_id'] in selected_ids]
        }
        
        self.progress.update_stage("CREATE_SUBSET", 0.4, "Created subset annotation data")

        # Download the full image dataset if not already extracted
        val_img_zip_path = self.coco_dir / "val2017.zip"
        if not val_img_zip_path.exists() and not any(self.coco_val_dir.iterdir()):
             print("Downloading COCO 2017 validation images for subset extraction...")
             # This function already has progress callbacks
             if not self.download_file(COCO_VAL_IMG_URL, val_img_zip_path, progress_stage="DOWNLOAD_IMAGES"):
                 print("Failed to download validation images. Cannot create subset.")
                 self.progress.update_stage("CREATE_SUBSET", 0.0, "Failed to download validation images")
                 return False

        # Extract needed images from the zip file directly if possible, or extract all then copy
        temp_extract_dir = None
        source_image_dir = None
        
        self.progress.update_stage("CREATE_SUBSET", 0.5, "Preparing to extract images for subset")

        if val_img_zip_path.exists():
            print("Extracting required validation images...")
            try:
                with zipfile.ZipFile(val_img_zip_path, 'r') as zip_ref:
                    # Extract only necessary files
                    required_files = [f"val2017/{img['file_name']}" for img in subset_data['images']]
                    
                    # Count for progress
                    total_extract = len(required_files)
                    extracted = 0
                    
                    for member in tqdm(required_files, desc="Extracting subset images"):
                        try:
                            zip_ref.extract(member, path=self.coco_dir)
                            extracted += 1
                            
                            # Update progress approximately every 2%
                            if extracted % max(1, total_extract // 50) == 0:
                                progress = extracted / total_extract
                                self.progress.update_stage(
                                    "CREATE_SUBSET", 
                                    0.5 + progress * 0.3, # 50% to 80% of subset creation
                                    f"Extracting subset images: {progress*100:.1f}%"
                                )
                                
                        except KeyError:
                            print(f"Warning: Image file {member} not found in zip archive.")
                        except Exception as e:
                            print(f"Error extracting file {member}: {e}")
                source_image_dir = self.coco_val_dir # Images are extracted directly into val2017
                print("Finished extracting subset images.")
                # Optionally remove the zip after extracting needed files
                # val_img_zip_path.unlink()
                # print(f"Removed {val_img_zip_path.name}")
            except Exception as e:
                 print(f"Error during selective extraction: {e}. Falling back to full extraction.")
                 self.progress.update_stage("CREATE_SUBSET", 0.5, "Error in selective extraction, falling back")
                 # Fallback: Extract all to temp then copy (original logic)
                 temp_extract_dir = self.coco_dir / "temp_extract"
                 temp_extract_dir.mkdir(parents=True, exist_ok=True)
                 if self._extract_zip(val_img_zip_path, temp_extract_dir, "Extracting full set for subset", "EXTRACT_IMAGES"):
                     source_image_dir = temp_extract_dir / "val2017"
                 else:
                     print("Failed to extract images. Cannot create subset.")
                     if temp_extract_dir.exists(): shutil.rmtree(temp_extract_dir)
                     self.progress.update_stage("CREATE_SUBSET", 0.0, "Failed to extract images")
                     return False
        elif self.coco_val_dir.exists() and any(self.coco_val_dir.iterdir()):
             print("Using existing extracted images in val2017 directory.")
             self.progress.update_stage("CREATE_SUBSET", 0.7, "Using existing extracted images")
             # Need to potentially prune existing images if they don't match the new subset
             print("Pruning existing val2017 directory for new subset...")
             required_filenames = {img['file_name'] for img in subset_data['images']}
             
             # Count for progress
             total_files = len(list(self.coco_val_dir.iterdir()))
             processed = 0
             
             for item in tqdm(list(self.coco_val_dir.iterdir()), desc="Checking existing images"):
                 if item.is_file() and item.name not in required_filenames:
                     item.unlink() # Remove files not in the selected subset
                 
                 processed += 1
                 # Update progress occasionally
                 if processed % max(1, total_files // 20) == 0:
                     progress = processed / total_files
                     self.progress.update_stage(
                         "CREATE_SUBSET", 
                         0.7 + progress * 0.1, # 70% to 80%
                         f"Pruning existing images: {progress*100:.1f}%"
                     )
                     
             source_image_dir = self.coco_val_dir # Source is the (now pruned) directory itself
        else:
             print("Warning: Cannot find validation images zip or extracted directory.")
             print("Creating empty validation directory structure...")
             self.coco_val_dir.mkdir(parents=True, exist_ok=True)
             self.progress.update_stage("CREATE_SUBSET", 0.0, "Warning: No image source found, created empty directory")
             return False


        # If we used a temporary directory for full extraction, copy selected images
        val2017_subset_dir = self.coco_val_dir  # selected images live in the validation dir
        if temp_extract_dir and source_image_dir:
            print(f"Copying {len(subset_data['images'])} selected images to {val2017_subset_dir}...")
            val2017_subset_dir.mkdir(parents=True, exist_ok=True) # Ensure target exists
            
            # Count for progress
            total_copy = len(subset_data['images'])
            copied = 0
            
            for img in tqdm(subset_data['images'], desc="Copying images"):
                src = source_image_dir / img['file_name']
                dst = val2017_subset_dir / img['file_name']
                if src.exists():
                    shutil.copy2(src, dst)
                    copied += 1
                    
                    # Update progress occasionally
                    if copied % max(1, total_copy // 20) == 0:
                        progress = copied / total_copy
                        self.progress.update_stage(
                            "CREATE_SUBSET", 
                            0.8 + progress * 0.1, # 80% to 90%
                            f"Copying subset images: {progress*100:.1f}%"
                        )
                else:
                    print(f"Warning: Source image {src} not found during copy.")
            # Clean up temporary extraction directory
            print(f"Removing temporary directory {temp_extract_dir}...")
            shutil.rmtree(temp_extract_dir)
            self.progress.update_stage("CREATE_SUBSET", 0.9, "Cleaning up temporary files")

        # Save modified annotations
        try:
            with open(subset_ann_file_temp, 'w') as f:
                json.dump(subset_data, f)
            self.progress.update_stage("CREATE_SUBSET", 0.95, "Saved subset annotation file")
        except Exception as e:
            print(f"Error writing subset annotation file {subset_ann_file_temp}: {e}")
            self.progress.update_stage("CREATE_SUBSET", 0.0, f"Error writing subset annotation file: {str(e)}")
            return False

        # Backup original annotations if not already done
        if not original_ann_backup.exists() and full_ann_file.exists():
            print(f"Backing up original annotations to {original_ann_backup.name}")
            shutil.copy2(full_ann_file, original_ann_backup)

        # Replace original annotations with subset annotations
        print(f"Replacing {full_ann_file.name} with subset annotations.")
        shutil.move(str(subset_ann_file_temp), str(full_ann_file)) # Use move for atomicity

        print(f"Successfully created COCO subset with {subset_size} images in {self.coco_val_dir}")
        self.progress.update_stage("CREATE_SUBSET", 1.0, "Subset creation complete")
        return True


    def setup_coco(self, subset_size: Optional[int] = None):
        """
        Download and set up COCO validation dataset (full or subset)
        
        Args:
            subset_size: Number of images for subset, or None for full dataset
        """
        # Create all necessary directories
        self.coco_dir.mkdir(parents=True, exist_ok=True)
        self.coco_ann_dir.mkdir(parents=True, exist_ok=True)
        self.coco_val_dir.mkdir(parents=True, exist_ok=True)

        # Define paths
        ann_zip_path = self.coco_dir / "annotations_trainval2017.zip"
        val_img_zip_path = self.coco_dir / "val2017.zip"
        full_ann_file = self.coco_ann_dir / "instances_val2017.json"
        original_ann_backup = self.coco_ann_dir / "instances_val2017_original.json"

        # 1. Download and Extract Annotations (needed for both full and subset)
        # Check if annotations are already extracted or if original backup exists (from previous subset op)
        if not full_ann_file.exists() and not original_ann_backup.exists():
            if not ann_zip_path.exists():
                if not self.download_file(COCO_ANN_URL, ann_zip_path, progress_stage="DOWNLOAD_ANNOTATIONS"):
                    print("Failed to download annotations. Aborting setup.")
                    # Create empty annotation directories to avoid further errors
                    self.coco_ann_dir.mkdir(parents=True, exist_ok=True)
                    return
            if not self._extract_zip(ann_zip_path, self.coco_dir, "Extracting annotations", "EXTRACT_ANNOTATIONS"):
                 print("Failed to extract annotations. Aborting setup.")
                 # Clean up potentially corrupted zip
                 if ann_zip_path.exists(): ann_zip_path.unlink()
                 # Create empty annotation directories to avoid further errors
                 self.coco_ann_dir.mkdir(parents=True, exist_ok=True)
                 return
            # Remove annotations zip file after successful extraction
            if ann_zip_path.exists(): ann_zip_path.unlink()
            print(f"Removed {ann_zip_path.name}")
        elif original_ann_backup.exists() and not full_ann_file.exists():
             print("Restoring original annotations from backup for setup...")
             shutil.copy2(original_ann_backup, full_ann_file)
             # Update progress for this stage
             self.progress.update_stage("EXTRACT_ANNOTATIONS", 1.0, "Using backup annotations")
        else:
            print("Annotations already seem to be present.")
            # Skip these stages in progress
            self.progress.update_stage("DOWNLOAD_ANNOTATIONS", 1.0, "Annotations already present")
            self.progress.update_stage("EXTRACT_ANNOTATIONS", 1.0, "Annotations already present")


        # 2. Handle Subset Creation if requested
        if subset_size is not None and subset_size > 0:
            # If val2017 directory exists, check if it needs removal/pruning for the new subset
            if self.coco_val_dir.exists():
                print(f"Existing val2017 directory found. Will be checked/pruned for subset of {subset_size} images.")
                # The _create_subset method now handles pruning/using existing files and progress updates

            if not self._create_subset(subset_size):
                 print("Failed to create subset. Aborting setup.")
                 # Ensure directories exist even if subset creation failed
                 self.coco_val_dir.mkdir(parents=True, exist_ok=True)
                 return # Stop if subset creation failed
            
            # Skip image download/extract stages in progress since we handled in create_subset 
            self.progress.update_stage("DOWNLOAD_IMAGES", 1.0, "Image subset handling complete")
            self.progress.update_stage("EXTRACT_IMAGES", 1.0, "Image subset handling complete")

        # 3. Handle Full Dataset Download/Extraction if not doing subset
        elif subset_size is None:
            # Restore original annotations if we were previously in a subset state
            if original_ann_backup.exists() and full_ann_file.exists():
                 print("Restoring original annotations from backup...")
                 shutil.move(str(original_ann_backup), str(full_ann_file)) # Move backup back
                 print("Original annotations restored.")
            elif original_ann_backup.exists() and not full_ann_file.exists():
                 print("Error: Backup annotation file exists but main one is missing. Cannot proceed.")
                 # Copy backup to main location to recover
                 shutil.copy2(original_ann_backup, full_ann_file)
                 print("Recovered from backup annotation file")

            # Check if images need downloading/extracting
            if not self.coco_val_dir.exists() or not any(self.coco_val_dir.iterdir()):
                 if not val_img_zip_path.exists():
                     print("Downloading COCO 2017 validation images (full set)...")
                     if not self.download_file(COCO_VAL_IMG_URL, val_img_zip_path, progress_stage="DOWNLOAD_IMAGES"):
                         print("Failed to download full validation images. Aborting setup.")
                         # Ensure val directory exists even if download failed
                         self.coco_val_dir.mkdir(parents=True, exist_ok=True)
                         return

                 if not self._extract_zip(val_img_zip_path, self.coco_dir, "Extracting full validation images", "EXTRACT_IMAGES"):
                     print("Failed to extract full validation images. Aborting setup.")
                     # Clean up potentially corrupted zip
                     if val_img_zip_path.exists(): val_img_zip_path.unlink()
                     # Ensure val directory exists even if extraction failed
                     self.coco_val_dir.mkdir(parents=True, exist_ok=True)
                     return
                 # Remove image zip file after successful extraction
                 if val_img_zip_path.exists(): val_img_zip_path.unlink()
                 print(f"Removed {val_img_zip_path.name}")
            else:
                 print("Full validation images seem to be already extracted.")
                 # Skip these stages in progress 
                 self.progress.update_stage("DOWNLOAD_IMAGES", 1.0, "Images already present")
                 self.progress.update_stage("EXTRACT_IMAGES", 1.0, "Images already present")
                 
            # Skip subset stage for full dataset
            self.progress.update_stage("CREATE_SUBSET", 1.0, "Using full dataset (no subset)")

        # Final check: make sure directories exist
        self.coco_val_dir.mkdir(parents=True, exist_ok=True)
        self.coco_ann_dir.mkdir(parents=True, exist_ok=True)

        print(f"COCO dataset setup complete at {self.coco_dir}")


    def compress_datasets(self):
        """Compress datasets to save space"""
        img_compressed_file = self.coco_dir / "val2017_compressed.zip"
        ann_compressed_file = self.coco_dir / "annotations_compressed.zip"

        # Ensure coco_dir exists
        self.coco_dir.mkdir(parents=True, exist_ok=True)

        # Compress val2017 images
        if self.coco_val_dir.exists() and any(self.coco_val_dir.iterdir()):
            print(f"Compressing COCO validation images to {img_compressed_file.name}...")
            try:
                with zipfile.ZipFile(img_compressed_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    # Walk relative to coco_dir to get correct archive names like 'val2017/...'
                    for root, _, files in os.walk(self.coco_val_dir):
                        for file in tqdm(files, desc=f"Compressing {self.coco_val_dir.name}"):
                            file_path = Path(root) / file
                            # Archive name relative to coco_dir (e.g., 'val2017/000...jpg')
                            arc_name = file_path.relative_to(self.coco_dir)
                            zipf.write(file_path, arc_name)

                print(f"Successfully compressed images to {img_compressed_file}")
                print(f"Removing original directory: {self.coco_val_dir}")
                shutil.rmtree(self.coco_val_dir)
            except Exception as e:
                print(f"Error compressing images: {e}")
                # Clean up potentially incomplete zip file
                if img_compressed_file.exists(): img_compressed_file.unlink()
        elif img_compressed_file.exists():
             print("Images seem to be already compressed.")
        else:
            print("COCO validation image directory not found or empty, nothing to compress.")

        # Compress annotations
        if self.coco_ann_dir.exists() and any(self.coco_ann_dir.iterdir()):
            print(f"Compressing COCO annotations to {ann_compressed_file.name}...")
            try:
                with zipfile.ZipFile(ann_compressed_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                     # Walk relative to coco_dir to get correct archive names like 'annotations/...'
                    for root, _, files in os.walk(self.coco_ann_dir):
                        for file in tqdm(files, desc=f"Compressing {self.coco_ann_dir.name}"):
                            file_path = Path(root) / file
                            # Archive name relative to coco_dir (e.g., 'annotations/instances_val2017.json')
                            arc_name = file_path.relative_to(self.coco_dir)
                            zipf.write(file_path, arc_name)

                print(f"Successfully compressed annotations to {ann_compressed_file}")
                print(f"Removing original directory: {self.coco_ann_dir}")
                shutil.rmtree(self.coco_ann_dir)
            except Exception as e:
                print(f"Error compressing annotations: {e}")
                # Clean up potentially incomplete zip file
                if ann_compressed_file.exists(): ann_compressed_file.unlink()
        elif ann_compressed_file.exists():
            print("Annotations seem to be already compressed.")
        else:
            print("COCO annotations directory not found or empty, nothing to compress.")

        if not img_compressed_file.exists() and not ann_compressed_file.exists():
             print("No data found to compress.")
        else:
             print("Compression complete. Use decompress command to restore.")


    def decompress_datasets(self):
        """Decompress datasets for use"""
        img_compressed_file = self.coco_dir / "val2017_compressed.zip"
        ann_compressed_file = self.coco_dir / "annotations_compressed.zip"
        decompressed = False

        # Ensure coco_dir exists
        self.coco_dir.mkdir(parents=True, exist_ok=True)

        # Decompress images
        if img_compressed_file.exists():
            if not self.coco_val_dir.exists() or not any(self.coco_val_dir.iterdir()):
                print(f"Decompressing {img_compressed_file.name}...")
                # Create val2017 directory if it doesn't exist
                self.coco_val_dir.mkdir(parents=True, exist_ok=True)
                # Extract directly into coco_dir, zip paths should be relative (e.g., 'val2017/...')
                if self._extract_zip(img_compressed_file, self.coco_dir, "Decompressing images"):
                    print("COCO images decompressed.")
                    img_compressed_file.unlink()
                    print(f"Removed {img_compressed_file.name}.")
                    decompressed = True
                else:
                    print(f"Failed to decompress {img_compressed_file.name}.")
            else:
                print("Image directory val2017 already exists. Skipping image decompression.")
                # Optionally offer to remove the compressed file anyway
                # img_compressed_file.unlink()
        else:
             if self.coco_val_dir.exists() and any(self.coco_val_dir.iterdir()):
                 print("Images seem to be already decompressed (directory exists).")
             else:
                 print("Compressed image file not found.")
                 self.coco_val_dir.mkdir(parents=True, exist_ok=True)
                 print("Created empty val2017 directory structure.")


        # Decompress annotations
        if ann_compressed_file.exists():
            if not self.coco_ann_dir.exists() or not any(self.coco_ann_dir.iterdir()):
                print(f"Decompressing {ann_compressed_file.name}...")
                # Create annotations directory if it doesn't exist
                self.coco_ann_dir.mkdir(parents=True, exist_ok=True) 
                # Extract directly into coco_dir, zip paths should be relative (e.g., 'annotations/...')
                if self._extract_zip(ann_compressed_file, self.coco_dir, "Decompressing annotations"):
                    print("COCO annotations decompressed.")
                    ann_compressed_file.unlink()
                    print(f"Removed {ann_compressed_file.name}.")
                    decompressed = True
                else:
                    print(f"Failed to decompress {ann_compressed_file.name}.")
            else:
                print("Annotations directory already exists. Skipping annotation decompression.")
                # Optionally offer to remove the compressed file anyway
                # ann_compressed_file.unlink()
        else:
            if self.coco_ann_dir.exists() and any(self.coco_ann_dir.iterdir()):
                 print("Annotations seem to be already decompressed (directory exists).")
            else:
                 print("Compressed annotation file not found.")
                 self.coco_ann_dir.mkdir(parents=True, exist_ok=True)
                 print("Created empty annotations directory structure.")

        if decompressed:
            print("Decompression finished. Datasets ready for use.")
        else:
            print("No datasets were decompressed.")

        # Final directory check
        self.coco_val_dir.mkdir(parents=True, exist_ok=True)
        self.coco_ann_dir.mkdir(parents=True, exist_ok=True)


    def delete_datasets(self):
        """Delete downloaded COCO dataset (images and annotations), leaving other data intact."""
        deleted = False
        # Specifically target the COCO directory for deletion
        if self.coco_dir.exists():
            print(f"Deleting COCO dataset directory: {self.coco_dir}...")
            try:
                shutil.rmtree(self.coco_dir)
                print("Successfully deleted COCO dataset directory.")
                deleted = True
            except Exception as e:
                print(f"Error deleting {self.coco_dir}: {e}")
        else:
            print("COCO dataset directory not found.")

        # Also check for stray COCO-related zip files at the base level and within coco_dir (if it still exists partially)
        stray_files = list(self.base_dir.glob("*.zip")) # Check in data_sets/
        if self.coco_dir.exists(): # Check inside coco_dir as well
             stray_files.extend(list(self.coco_dir.glob("*.zip")))
        else: # If coco_dir was deleted or never existed, check image_dir for zips
             stray_files.extend(list(self.image_dir.glob("*.zip")))


        for stray_zip in stray_files:
             # Only delete zips clearly related to COCO
             if "val2017" in stray_zip.name or "annotations" in stray_zip.name:
                 print(f"Deleting stray COCO zip file: {stray_zip}")
                 try:
                     stray_zip.unlink()
                     deleted = True
                 except Exception as e:
                     print(f"Error deleting stray file {stray_zip}: {e}")

        if deleted:
            print("COCO dataset deletion process finished.")
        else:
            print("No COCO data found to delete.")

        # Re-create empty directories to avoid subsequent errors
        self.coco_dir.mkdir(parents=True, exist_ok=True)
        self.coco_val_dir.mkdir(parents=True, exist_ok=True)
        self.coco_ann_dir.mkdir(parents=True, exist_ok=True)


def main():
    # Print version to confirm we're running the right script
    print(f"Dataset Manager version {VERSION}")

    parser = argparse.ArgumentParser(description=f"COCO Dataset Manager (v{VERSION})")
    parser.add_argument(
        'command',
        choices=['download', 'compress', 'decompress', 'delete', 'status'],
        help="Action to perform: 'download', 'compress', 'decompress', 'delete', 'status'."
    )
    parser.add_argument(
        '--subset',
        type=int,
        metavar='N',
        default=None,
        help="Download only a subset of N images (only valid with 'download' command)."
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help="Force action even if seemingly unnecessary (e.g., re-download, force delete confirmation)."
    )
    # Add status command later if needed

    args = parser.parse_args()
    manager = DatasetManager()

    if args.command == 'download':
        if args.subset is not None:
            if args.subset <= 0:
                print("Error: Subset size must be a positive integer.")
                sys.exit(1)
            print(f"Starting download and setup for a subset of {args.subset} images...")
            manager.setup_coco(subset_size=args.subset)
        else:
            print("Starting download and setup for the full COCO validation dataset...")
            manager.setup_coco(subset_size=None) # Explicitly None for full set
    elif args.command == 'compress':
        print("Starting dataset compression...")
        manager.compress_datasets()
    elif args.command == 'decompress':
        print("Starting dataset decompression...")
        manager.decompress_datasets()
    elif args.command == 'delete':
        confirm = False
        if args.force:
            confirm = True
        else:
            user_input = input("Are you sure you want to delete all downloaded/extracted datasets and compressed files? (yes/no): ").strip().lower()
            if user_input in ('yes', 'y'):
                confirm = True

        if confirm:
            print("Starting dataset deletion...")
            manager.delete_datasets()
        else:
            print("Deletion cancelled.")
    elif args.command == 'status':
        # Basic status check
        print("\nDataset Status:")
        print(f" Base Directory: {manager.base_dir}")
        print(f" Image Directory: {manager.image_dir} {'(Exists)' if manager.image_dir.exists() else '(Not Found)'}")
        if manager.image_dir.exists():
             print(f"  COCO Directory: {manager.coco_dir} {'(Exists)' if manager.coco_dir.exists() else '(Not Found)'}")
             if manager.coco_dir.exists():
                 # Check for extracted data
                 val_dir_exists = manager.coco_val_dir.exists() and any(manager.coco_val_dir.iterdir())
                 ann_dir_exists = manager.coco_ann_dir.exists() and any(manager.coco_ann_dir.iterdir())
                 print(f"   val2017 Images: {'Extracted' if val_dir_exists else 'Not Extracted'}")
                 print(f"   Annotations: {'Extracted' if ann_dir_exists else 'Not Extracted'}")

                 # Check for compressed files
                 img_zip = manager.coco_dir / "val2017_compressed.zip"
                 ann_zip = manager.coco_dir / "annotations_compressed.zip"
                 print(f"   Compressed Images: {'Exists' if img_zip.exists() else 'Not Found'}")
                 print(f"   Compressed Annotations: {'Exists' if ann_zip.exists() else 'Not Found'}")

                 # Check for subset indicator (backup annotation file)
                 backup_ann = manager.coco_ann_dir / "instances_val2017_original.json"
                 if backup_ann.exists():
                     print("   Mode: Likely configured as a SUBSET (original annotations backed up)")
                 elif val_dir_exists and ann_dir_exists:
                      print("   Mode: Likely configured as FULL dataset")

    print("\nDataset manager operation completed.")

if __name__ == "__main__":
    main()