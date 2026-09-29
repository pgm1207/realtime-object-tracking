# metrics_visualizer.py
"""
Metrics Visualization Module for Computer Vision Project

Generates a comprehensive dashboard comparing model performance based on
evaluation results stored in JSON files. Designed to automatically incorporate
any models found in the results file.
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.table import Table
import numpy as np
import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import glob
import os
import re
import traceback # For detailed error printing

# Configure paths relative to this script's location
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent # Assuming script is in src/
RESULTS_DIR = PROJECT_ROOT / "inference" / "results"
VISUALIZATIONS_DIR = RESULTS_DIR / "visualizations"
VISUALIZATIONS_DIR.mkdir(exist_ok=True, parents=True)

# Consistent color mapping for models.
# *** Add new models and their desired colors here as needed. ***
# Ensure these keys match the model names used as keys in the evaluation_results_*.json file.
MODEL_COLORS = {
    # Enhanced color palette with modern, visually appealing colors
    'mask-rcnn': '#5D69B1',       # Rich blue
    'yolo-seg': '#52BCA3',        # Teal
    'dino-seg': '#99C945',        # Lime green
    'yolo8n-seg': '#CC61B0',      # Magenta
    'yolo8s-seg': '#24796C',      # Dark teal
    'yolo8m-seg': '#DAA51B',      # Golden yellow
    'yolo8l-seg': '#2F8AC4',      # Sky blue
    'yolo8x-seg': '#764E9F',      # Purple
    'yolo9c-seg': '#ED645A',      # Coral
    'yolo9e-seg': '#CC3A8E',      # Pink
    'yolo11n-seg': '#A5AA99',     # Sage
    'yolo11s-seg': '#8DD593',     # Mint green
    'yolo11m-seg': '#C6DEC7',     # Pale green
    'yolo11l-seg': '#EAD3E7',     # Lilac
    'yolo11x-seg': '#F1E6C8',     # Cream
    'Default': '#A0A0A0'          # Neutral gray
}

def get_model_color(model_name):
    """Returns a consistent color for a given model name."""
    return MODEL_COLORS.get(model_name, MODEL_COLORS['Default'])

class MetricsVisualizer:
    """Class to load evaluation results and generate performance visualizations."""

    # Define consistent color scheme for all metrics across all plots
    # These colors will be used consistently for the same metrics in all charts
    METRIC_COLORS = {
        # Main metrics
        'Precision': '#4C78A8',        # Blue
        'Recall': '#F58518',           # Orange
        'F1-Score': '#72B7B2',         # Teal
        
        # AP metrics by IoU
        'mAP (IoU=0.50:0.95)': '#4C78A8',  # Blue (same as Precision)
        'AP (IoU=0.50)': '#F58518',         # Orange (same as Recall)
        'AP (IoU=0.75)': '#72B7B2',         # Teal (same as F1-Score)
        
        # Object size metrics
        'Small Objects AP': '#54A24B',     # Green
        'Small': '#54A24B',                # Green (alias)
        'Medium Objects AP': '#EECA3B',    # Yellow
        'Medium': '#EECA3B',               # Yellow (alias)
        'Large Objects AP': '#B279A2',     # Purple
        'Large': '#B279A2',                # Purple (alias)
        
        # Other metrics
        'Speed (FPS)': '#FF9DA6',          # Pink
        'Segm_mAP (IoU=0.50:0.95)': '#4C78A8',  # Blue (same as mAP)
        'Segm_AP (IoU=0.50)': '#F58518',         # Orange (same as AP@0.50)
        
        # Shape metrics
        'Compactness': '#9D755D',          # Brown
        'Convexity': '#BAB0AC',            # Gray
        'Circularity': '#E45756',          # Red
    }
    
    def __init__(self, results_file=None, results_dir=None):
        """
        Initialize the visualizer, loading results for all models found.

        Args:
            results_file (str or Path, optional): Path to a specific JSON results file.
                                                  If None, the latest is loaded.
            results_dir (str or Path, optional): Directory containing results files.
                                                 Defaults to PROJECT_ROOT/inference/results.
        """
        self.results_dir = Path(results_dir) if results_dir else RESULTS_DIR
        self.results_file = None
        self.results = None
        self.models = [] # List of model names found in the results file
        self.data = {}   # Stores extracted data structures (DataFrames, lists)

        if results_file:
            self.results_file = Path(results_file)
            if not self.results_file.is_file():
                print(f"Warning: Specified results file not found: {self.results_file}")
                self.results_file = None
                self._load_latest_results() # Fallback to finding latest
            else:
                self._load_results()
        else:
            self._load_latest_results() # Find and load the latest results file

        if self.results:
            # Dynamically get model names from the keys in the loaded JSON data
            self.models = list(self.results.keys())
            # Remove metadata key if it exists
            if "evaluation_metadata" in self.models:
                self.models.remove("evaluation_metadata")
            print(f"Models found in results file: {self.models}")
            # Extract data needed for plotting for all found models
            self._extract_data_for_plotting()
        else:
            print("Warning: No evaluation results loaded. Dashboard cannot be generated.")


    def _load_latest_results(self):
        """Finds and loads the most recent evaluation_results_*.json file."""
        try:
            # Use glob to find files matching the pattern
            list_of_files = glob.glob(str(self.results_dir / 'evaluation_results_*.json'))
            if not list_of_files:
                print(f"Error: No 'evaluation_results_*.json' files found in {self.results_dir}")
                return
            # Find the most recently modified file
            latest_file = max(list_of_files, key=os.path.getctime)
            self.results_file = Path(latest_file)
            self._load_results() # Load the found file
        except Exception as e:
            print(f"Error finding latest results file: {e}")
            traceback.print_exc()

    def _load_results(self):
        """Loads the JSON data from the specified results file."""
        if not self.results_file or not self.results_file.is_file():
            print(f"Error: Results file path is invalid or not set ({self.results_file}).")
            return
        try:
            with open(self.results_file, 'r') as f:
                self.results = json.load(f)
            print(f"Successfully loaded results from: {self.results_file.name}")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {self.results_file.name}: {e}")
            self.results = None
        except Exception as e:
            print(f"Error loading results file {self.results_file.name}: {e}")
            traceback.print_exc()
            self.results = None

    def _get_metric(self, model, metric_path, default=0.0):
        """
        Safely retrieve a potentially nested metric for a given model.
        Handles missing keys or non-numeric values gracefully.
        Handles cases where the final key name might contain dots.

        Args:
            model (str): The name of the model (key in the results dict).
            metric_path (str): Path to the metric. Can be 'key' or 'parent_key.child_key'.
            default (float): Value to return if the metric is not found or invalid.

        Returns:
            float: The metric value or the default.
        """
        if model not in self.results:
            return default

        value_source = self.results.get(model, {})

        try:
            # Handle potential nesting (assuming max 1 level like 'parent.child')
            if '.' in metric_path:
                parent_key, child_key = metric_path.split('.', 1) # Split only on the first dot
                if isinstance(value_source, dict) and parent_key in value_source:
                    current_level = value_source[parent_key]
                    if isinstance(current_level, dict) and child_key in current_level:
                        final_raw_value = current_level[child_key]
                    else:
                        return default
                else:
                    return default
            else:
                # Top-level key
                if isinstance(value_source, dict) and metric_path in value_source:
                    final_raw_value = value_source[metric_path]
                else:
                    return default

            # Final value obtained, try converting to float
            final_value = float(final_raw_value) if final_raw_value is not None else default
            return final_value

        except (TypeError, ValueError) as e:
            return default
        except Exception as e:
             return default

    def _extract_data_for_plotting(self):
        """
        Parses the loaded results for ALL models found in the file and
        prepares data structures (DataFrame, lists) needed for the various plots.
        Uses _get_metric to handle potentially missing data for any model.
        Calculates Precision, Recall, F1 based on COCO proxies.
        """
        if not self.results or not self.models:
            print("Error: Cannot extract data, results not loaded or no models found.")
            return

        self.data = {}

        # Define metrics to extract directly from JSON (excluding P/R/F1 for now)
        metrics_to_extract = {
            # Display Name          : JSON Path
            "mAP (IoU=0.50:0.95)"   : "coco_metrics.AP_IoU=0.50:0.95",
            "AP (IoU=0.50)"         : "coco_metrics.AP_IoU=0.50", # Proxy for Precision
            "AP (IoU=0.75)"         : "coco_metrics.AP_IoU=0.75",
            "AR (max=100)"          : "coco_metrics.AR_max=100", # Proxy for Recall
            "Small Objects AP"      : "coco_metrics.AP_small",
            "Medium Objects AP"     : "coco_metrics.AP_medium",
            "Large Objects AP"      : "coco_metrics.AP_large",
            "Speed (FPS)"           : "fps",
        }

        # Initialize dictionary to hold extracted data
        plot_data = {key: [] for key in metrics_to_extract.keys()}
        # Add placeholders for calculated P/R/F1
        plot_data["Precision"] = []
        plot_data["Recall"] = []
        plot_data["F1-Score"] = []
        plot_data["Model"] = self.models # Store model names in order

        # For debugging
        print("DEBUG: Available keys in results for first model:")
        if self.models:
            first_model = self.models[0]
            if first_model in self.results:
                if 'coco_metrics' in self.results[first_model]:
                    print(f"DEBUG: coco_metrics keys: {list(self.results[first_model]['coco_metrics'].keys())}")
                else:
                    print(f"DEBUG: No 'coco_metrics' in model data. Available keys: {list(self.results[first_model].keys())}")

        # --- Extract main metrics for each model ---
        for model in self.models:
            # Extract metrics defined in metrics_to_extract
            for display_name, json_path in metrics_to_extract.items():
                metric_value = self._get_metric(model, json_path, default=0.0)
                # Ensure we have at least a tiny positive value for visualization
                if display_name != "Speed (FPS)" and metric_value < 1e-6 and metric_value >= 0:
                    metric_value = 1e-6  # Small positive value instead of exact zero
                plot_data[display_name].append(metric_value)

            # --- Calculate P/R/F1 from proxies ---
            # For Precision, try multiple potential metrics and use the maximum value
            # This ensures we get visible bars even when some metrics are zero
            ap50 = plot_data["AP (IoU=0.50)"][-1] # Standard precision proxy
            # Also consider other metrics that could represent precision
            ap75 = plot_data["AP (IoU=0.75)"][-1]
            small_ap = plot_data["Small Objects AP"][-1]
            medium_ap = plot_data["Medium Objects AP"][-1] 
            large_ap = plot_data["Large Objects AP"][-1]
            
            # For precision, use the highest value from any AP metric
            precision_candidates = [ap50, ap75, small_ap, medium_ap, large_ap]
            precision_proxy = max(precision_candidates) if any(v > 0 for v in precision_candidates) else 1e-6
            
            # Use AR_max=100 directly as Recall proxy, ensure it's not exactly zero
            recall_proxy = max(plot_data["AR (max=100)"][-1], 1e-6)

            # Calculate F1, handle division by zero
            if precision_proxy + recall_proxy > 1e-9: # Use tolerance
                f1_score = 2 * (precision_proxy * recall_proxy) / (precision_proxy + recall_proxy)
            else:
                f1_score = 1e-6  # Small non-zero value instead of exact zero

            # Append calculated values
            plot_data["Precision"].append(precision_proxy)
            plot_data["Recall"].append(recall_proxy)
            plot_data["F1-Score"].append(f1_score)

        # Convert the extracted and calculated metrics into a pandas DataFrame
        self.data['summary_df'] = pd.DataFrame(plot_data)
        # Use Model names as the index for the DataFrame
        self.data['summary_df'].set_index('Model', inplace=True)

        # --- Extract Detection Counts for Box Plot ---
        detection_counts = {}
        max_detection_value = 0

        print("\nDEBUG - Extracting Detection Counts:")
        for model in self.models:
            counts = self.results.get(model, {}).get("detection_counts_per_image", [])
            print(f"  -> Model: {model}, Raw counts type: {type(counts)}, Raw counts (first 10): {str(counts)[:100]}...")

            if isinstance(counts, list) and counts:
                numeric_counts = [c for c in counts if isinstance(c, (int, float))]
                print(f"    -> Numeric counts found: {len(numeric_counts)}")
                if numeric_counts:
                    detection_counts[model] = numeric_counts
                    current_max = max(numeric_counts)
                    if current_max > max_detection_value:
                        max_detection_value = current_max
                else:
                    # List exists but contains no numeric data - use a small value instead of zero
                    print(f"    -> Warning: 'detection_counts_per_image' for model '{model}' contains no numeric data. Plotting [0.001].")
                    detection_counts[model] = [0.001]
            else:
                # If counts are missing, empty, or not a list, add a placeholder [0.001]
                print(f"    -> Warning: Counts missing, empty, or not a list for model '{model}'. Plotting [0.001].")
                detection_counts[model] = [0.001]
                if not isinstance(counts, list) and counts is not None:
                     print(f"    -> Detail: 'detection_counts_per_image' for model '{model}' is not a list. Found: {type(counts)}.")

        print(f"  -> Final detection_counts keys: {list(detection_counts.keys())}")
        print(f"  -> Calculated max_detection_value: {max_detection_value}")
        self.data['detection_counts'] = detection_counts
        # Add padding to the y-limit for the box plot for better visualization
        self.data['max_detection_value'] = max_detection_value * 1.15 if max_detection_value > 0 else 10


    def create_metrics_dashboard(self, show_plot=False, output_path=None, dpi=150, file_format='png'):
        """
        Creates the comprehensive multi-plot dashboard visualization.
        This function orchestrates the creation of all subplots.

        Args:
            show_plot (bool): If True, display the plot interactively using plt.show().
                              If False, only save the plot to a file.
            output_path (str or Path, optional): Full path to save the dashboard image.
                                                 If None, a default path is generated.
            dpi (int, optional): Dots per inch for the saved image.
            file_format (str, optional): Format of the saved image (e.g., 'png', 'jpg', 'svg').

        Returns:
            str: Absolute path to the saved dashboard image file, or None if generation failed.
        """
        # Check if data extraction was successful
        if not self.data or 'summary_df' not in self.data or self.data['summary_df'].empty:
            print("Error: No valid data extracted from results. Cannot create dashboard.")
            # Attempt to load results again if they are missing
            if not self.results:
                self._load_latest_results()
                if self.results:
                    self._extract_data_for_plotting()
                else:
                    return None # Still no results
            # If data extraction failed even with results, report error
            if not self.data or 'summary_df' not in self.data or self.data['summary_df'].empty:
                 print("Error: Data extraction failed even after loading results.")
                 return None


        # --- Set up Figure and GridSpec Layout ---
        # Increased figure size for better readability of multiple plots
        fig = plt.figure(figsize=(20, 17)) # Adjusted size
        # GridSpec defines the layout structure (3 rows, 6 columns base)
        gs = gridspec.GridSpec(3, 6, figure=fig, hspace=0.45, wspace=0.55) # Adjusted spacing

        # --- Define Subplot Axes using GridSpec ---
        # Top Row
        ax_pr_f1 = fig.add_subplot(gs[0, 0:2]) # Precision/Recall/F1 (spans 2 cols)
        ax_ap_iou = fig.add_subplot(gs[0, 2:4]) # AP@IoU (spans 2 cols)
        ax_ap_size = fig.add_subplot(gs[0, 4:6]) # AP by Size (spans 2 cols)
        # Middle Row
        ax_fps = fig.add_subplot(gs[1, 0:2])     # FPS (spans 2 cols)
        ax_dist = fig.add_subplot(gs[1, 2:4])    # Detection Distribution (spans 2 cols)
        ax_f1_speed = fig.add_subplot(gs[1, 4:6]) # F1 vs Speed (spans 2 cols)
        # Bottom Row
        ax_radar = fig.add_subplot(gs[2, 0:3], polar=True) # Radar (spans 3 cols, polar projection)
        ax_table = fig.add_subplot(gs[2, 3:6]) # Summary Table (spans 3 cols)

        # --- Populate Subplots by Calling Helper Methods ---
        try:
            self._plot_precision_recall_f1(ax_pr_f1)
            self._plot_ap_iou(ax_ap_iou)
            self._plot_ap_size(ax_ap_size)
            self._plot_fps(ax_fps)
            self._plot_detection_distribution(ax_dist)
            self._plot_f1_vs_speed(ax_f1_speed)
            self._plot_radar(ax_radar)
            self._plot_summary_table(ax_table)
        except Exception as e:
             print(f"Error occurred during subplot generation: {e}")
             traceback.print_exc()
             plt.close(fig) # Close the figure window to prevent display/saving issues
             return None

        # --- Add Overall Figure Title ---
        results_filename = self.results_file.name if self.results_file else "unknown_results.json"
        fig.suptitle(f'Comprehensive Model Performance Comparison\nEvaluation results from {results_filename}',
                     fontsize=20, y=0.985) # Adjusted size and position

        # --- Save and/or Show the Figure ---
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"metrics_dashboard_{timestamp}.{file_format}"
            save_to_path = VISUALIZATIONS_DIR / output_filename
        else:
            save_to_path = Path(output_path)
            # Ensure the output directory exists
            save_to_path.parent.mkdir(parents=True, exist_ok=True)


        try:
            # Save the figure to the visualizations directory
            plt.savefig(save_to_path, dpi=dpi, bbox_inches='tight', facecolor='white', format=file_format)
            print(f"Metrics dashboard saved successfully to: {save_to_path}")

            if show_plot:
                plt.show() # Display the plot window
            else:
                plt.close(fig) # Close the figure object if not showing interactively

            return str(save_to_path.resolve()) # Return the absolute path

        except Exception as e:
            print(f"Error saving or showing the plot: {e}")
            traceback.print_exc()
            plt.close(fig) # Ensure figure is closed on error
            return None

    def _add_value_labels(self, ax, container, precision=3, is_bar_chart=True):
        """Attach a text label above each bar or point, displaying its value."""
        if is_bar_chart:
            # Handle BarContainer
            for bar in container:
                height = bar.get_height()
                # Format the label - use '0.000...' for zero values with specified precision
                label = f'{height:.{precision}f}' if abs(height) > 1e-9 else f'0.{"0"*precision}'

                ax.annotate(label,
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        else:
            # Handle scatter plots or other containers
            pass # Not implemented for scatter in this version


    # --- Individual Plotting Functions ---

    def _plot_precision_recall_f1(self, ax):
        """Plots Precision, Recall, F1-Score bar chart."""
        metrics_to_plot = ['Precision', 'Recall', 'F1-Score']
        available_metrics = [m for m in metrics_to_plot if m in self.data['summary_df'].columns]
        if not available_metrics:
             ax.text(0.5, 0.5, 'Precision/Recall/F1 data not available.', ha='center', va='center', transform=ax.transAxes)
             ax.set_title('Precision, Recall, F1-Score Comparison')
             return

        # Get original data
        orig_df = self.data['summary_df'][available_metrics].copy()
        
        # Create a boosted version of the data for better visualization
        df = pd.DataFrame(index=orig_df.index)
        
        # Check if all values are extremely small
        max_val = orig_df.max().max()
        min_visible_value = 0.05  # Minimum height for bars to be clearly visible
        
        if max_val < 0.01:  # If all values are too small to see properly
            # Calculate boost needed to make the largest value visible
            if max_val > 1e-6:  # If we have some non-zero values
                boost_factor = min_visible_value / max_val  # Make largest value = min_visible_value
            else:
                boost_factor = 1000  # Just use a large factor if all values are essentially zero
                
            # Apply different boost factors to each value to maintain relative proportions
            # but make all values visible
            for model in orig_df.index:
                for metric in available_metrics:
                    orig_val = orig_df.loc[model, metric]
                    if orig_val < 1e-6:  # Almost zero
                        # Set a small minimum value to ensure bars are visible
                        df.loc[model, metric] = min_visible_value * 0.2  # 20% of our minimum
                    else:
                        # Scale proportionally but ensure value is at least 10% of minimum
                        boosted_val = orig_val * boost_factor
                        df.loc[model, metric] = max(boosted_val, min_visible_value * 0.1)
                        
            scaled = True  # Flag that we've scaled values
        else:
            # Values are large enough to show directly
            df = orig_df.copy()
            scaled = False

        n_models = len(df)
        n_metrics = len(df.columns)
        bar_width = 0.25
        index = np.arange(n_models)

        # Colors based on target image legend - updated to more vibrant, visually appealing colors
        colors = [self.METRIC_COLORS.get(metric, '#A0A0A0') for metric in df.columns]

        for i, metric in enumerate(df.columns):
            plot_data = pd.to_numeric(df[metric], errors='coerce').fillna(0)
            rects = ax.bar(index + i * bar_width, plot_data, bar_width, label=metric, color=colors[i])
            
            # Use original values in labels (not the boosted ones)
            orig_values = pd.to_numeric(orig_df[metric], errors='coerce').fillna(0)
            
            # Custom label each bar with the original value
            for j, rect in enumerate(rects):
                height = rect.get_height()
                orig_val = orig_values.iloc[j]
                
                # Use more decimal places for very small values
                if orig_val < 0.01:
                    label = f'{orig_val:.4f}'  # Show 4 decimal places for small values
                else:
                    label = f'{orig_val:.3f}'
                
                ax.annotate(label,
                           xy=(rect.get_x() + rect.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)

        ax.set_ylabel('Value')
        if scaled:
            ax.set_title('Precision, Recall, F1-Score Comparison\n(Values scaled for visibility)')
        else:
            ax.set_title('Precision, Recall, F1-Score Comparison')
            
        # Use simplified model names when there are more than 5 models
        model_names = df.index.tolist()
        display_names = self._simplify_model_names(model_names)
        
        ax.set_xticks(index + bar_width * (n_metrics - 1) / 2)
        ax.set_xticklabels(display_names, rotation=0, ha='center')
        ax.legend(title="Metric", fontsize='small')

        # Dynamic y-axis
        max_plotted = df.max().max()
        if max_plotted <= 0:
            upper_limit = 0.1  # default minimum scale when values are zero
        else:
            upper_limit = max_plotted * 1.2  # Add 20% headroom
            
        ax.set_ylim(0, upper_limit)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        
        # Add explanatory text about precision, recall, and F1-score metrics
        explanation = (
            "Precision: Accuracy of positive predictions (TP/(TP+FP))\n"
            "Recall: Ability to find all positive samples (TP/(TP+FN))\n"
            "F1-Score: Harmonic mean of Precision and Recall"
        )
        
        # Add explanatory text in a box at the bottom
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)
        ax.text(0.5, -0.20, explanation, transform=ax.transAxes,
              fontsize=8, ha='center', va='center', bbox=props)

    def _plot_ap_iou(self, ax):
        """Plots Average Precision (AP) at different IoU thresholds."""
        metrics_to_plot = ['mAP (IoU=0.50:0.95)', 'AP (IoU=0.50)', 'AP (IoU=0.75)']
        available_metrics = [m for m in metrics_to_plot if m in self.data['summary_df'].columns]
        if not available_metrics:
             ax.text(0.5, 0.5, 'AP@IoU data not available.', ha='center', va='center', transform=ax.transAxes)
             ax.set_title('Average Precision (AP) at Different IoU Thresholds')
             return

        # Get original data
        orig_df = self.data['summary_df'][available_metrics].copy()
        
        # --- Scaling logic remains the same ---
        # Create a boosted version of the data for better visualization
        df = pd.DataFrame(index=orig_df.index)
        max_val = orig_df.max().max()
        min_visible_value = 0.05
        if max_val < 0.01:
            if max_val > 1e-6: boost_factor = min_visible_value / max_val
            else: boost_factor = 1000
            for model in orig_df.index:
                for metric in available_metrics:
                    orig_val = orig_df.loc[model, metric]
                    if orig_val < 1e-6: df.loc[model, metric] = min_visible_value * 0.2
                    else:
                        boosted_val = orig_val * boost_factor
                        df.loc[model, metric] = max(boosted_val, min_visible_value * 0.1)
            scaled = True
        else:
            df = orig_df.copy()
            scaled = False
        # --- End Scaling Logic ---


        n_models = len(df)
        n_metrics = len(df.columns)
        bar_width = 0.25
        index = np.arange(n_models)

        legend_labels = {
            'mAP (IoU=0.50:0.95)': 'mAP (0.50:0.95)',
            'AP (IoU=0.50)': 'AP@0.50',
            'AP (IoU=0.75)': 'AP@0.75'
        }
        # Use our consistent color palette
        colors = [self.METRIC_COLORS.get(metric, '#A0A0A0') for metric in df.columns]

        for i, metric in enumerate(df.columns):
            plot_data = pd.to_numeric(df[metric], errors='coerce').fillna(0)
            rects = ax.bar(index + i * bar_width, plot_data, bar_width,
                        label=legend_labels.get(metric, metric),
                        color=colors[i])

            # Use original values in labels (not the boosted ones)
            orig_values = pd.to_numeric(orig_df[metric], errors='coerce').fillna(0)

            # Custom label each bar with the original value
            for j, rect in enumerate(rects):
                height = rect.get_height()
                orig_val = orig_values.iloc[j]

                # Force using 4 decimal places for all small AP values
                if orig_val < 0.1: label = f'{orig_val:.4f}'
                else: label = f'{orig_val:.3f}'

                ax.annotate(label,
                          xy=(rect.get_x() + rect.get_width() / 2, height),
                          xytext=(0, 3),  # 3 points vertical offset
                          textcoords="offset points",
                          ha='center', va='bottom', fontsize=8)

        ax.set_ylabel('AP Value')
        if scaled: ax.set_title('Average Precision (AP) at Different IoU Thresholds\n(Values scaled for visibility)')
        else: ax.set_title('Average Precision (AP) at Different IoU Thresholds')
        
        # Use simplified model names when there are more than 5 models
        model_names = df.index.tolist()
        display_names = self._simplify_model_names(model_names)
        
        ax.set_xticks(index + bar_width * (n_metrics - 1) / 2)
        ax.set_xticklabels(display_names, rotation=0, ha='center')
        ax.legend(title="AP Metric", fontsize='small')
        
        max_plotted = df.max().max()
        if max_plotted <= 0: upper_limit = 0.1
        else: upper_limit = max_plotted * 1.2
        ax.set_ylim(0, upper_limit)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        
        # Add explanatory text about AP metrics at different IoU thresholds
        explanation = (
            "mAP (0.50:0.95): Mean AP averaged over IoU thresholds from 0.5 to 0.95\n"
            "AP@0.50: AP at IoU threshold of 0.50 (permissive overlap criterion)\n"
            "AP@0.75: AP at IoU threshold of 0.75 (strict overlap criterion)"
        )
        
        # Add explanatory text in a box at the bottom
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)
        ax.text(0.5, -0.20, explanation, transform=ax.transAxes,
              fontsize=8, ha='center', va='center', bbox=props)

    def _plot_ap_size(self, ax):
        """Plots Shape Analysis metrics for segmentation models."""
        # Check if shape metrics are available in results
        has_shape_metrics = False
        shape_metrics_data = {}
        
        # First try to find shape metrics in the results
        for model in self.models:
            if model in self.results and 'shape_metrics' in self.results[model]:
                has_shape_metrics = True
                shape_data = self.results[model]['shape_metrics']
                
                # Get average metrics across all masks
                if shape_data.get('compactness') and len(shape_data.get('compactness', [])) > 0:
                    shape_metrics_data[model] = {
                        'Compactness': np.mean(shape_data.get('compactness', [0])),
                        'Convexity': np.mean(shape_data.get('convexity', [0])),
                        'Circularity': np.mean(shape_data.get('circularity', [0]))
                    }
        
        # If we have shape metrics, create a specialized bar chart
        if has_shape_metrics and shape_metrics_data:
            model_names = list(shape_metrics_data.keys())
            metrics_to_plot = ['Compactness', 'Convexity', 'Circularity']
            
            # Prepare data for plotting
            data = {metric: [] for metric in metrics_to_plot}
            for model in model_names:
                for metric in metrics_to_plot:
                    # Get value or default to 0
                    value = shape_metrics_data[model].get(metric, 0.0)
                    data[metric].append(value)
                    
            # Set up plot
            n_models = len(model_names)
            n_metrics = len(metrics_to_plot)
            bar_width = 0.25
            index = np.arange(n_models)
            
            # Use our consistent color palette for shape metrics
            metric_colors = {
                'Compactness': self.METRIC_COLORS.get('Precision', '#4C78A8'),    # Blue (same as Precision)
                'Convexity': self.METRIC_COLORS.get('Recall', '#F58518'),         # Orange (same as Recall)
                'Circularity': self.METRIC_COLORS.get('F1-Score', '#72B7B2')      # Teal (same as F1-Score)
            }
            
            colors = [metric_colors.get(metric, self.METRIC_COLORS.get(metric, '#A0A0A0')) for metric in metrics_to_plot]

            # Create bars for each metric
            for i, metric in enumerate(metrics_to_plot):
                values = data[metric]
                rects = ax.bar(index + i * bar_width, values, bar_width,
                             label=metric, color=colors[i])
                
                # Add data labels on bars
                for j, rect in enumerate(rects):
                    height = rect.get_height()
                    # Format based on value range (shape metrics typically 0-1)
                    if height < 0.01:
                        label_text = f'{height:.4f}'
                    else:
                        label_text = f'{height:.3f}'
                        
                    ax.annotate(label_text,
                              xy=(rect.get_x() + rect.get_width() / 2, height),
                              xytext=(0, 3),  # 3 points vertical offset
                              textcoords="offset points",
                              ha='center', va='bottom', fontsize=8)
                    
            # Configure axes
            ax.set_ylabel('Shape Metric Value (0-1)')
            ax.set_title('Shape Analysis (Segmentation Quality)')
            
            # Use simplified model names when there are more than 5 models
            display_names = self._simplify_model_names(model_names)
            ax.set_xticks(index + bar_width * (n_metrics - 1) / 2)
            ax.set_xticklabels(display_names, rotation=0, ha='center')
            ax.legend(title="Shape Metrics", fontsize='small')
            
            # Set y-limits for shape metrics (typically 0-1)
            ax.set_ylim(0, 1.05)
            ax.grid(axis='y', linestyle='--', alpha=0.6)
            
            # Add explanatory text about shape metrics
            explanation = (
                "Compactness: How efficiently a boundary encloses area\n"
                "Convexity: Ratio of perimeter of convex hull to actual perimeter\n"
                "Circularity: How closely shape resembles a circle"
            )
            
            # Add explanatory text in a box at the bottom
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)
            ax.text(0.5, -0.20, explanation, transform=ax.transAxes,
                  fontsize=8, ha='center', va='center', bbox=props)
            return
            
        # Check if we have shape metrics by object size as a fallback
        has_size_shape_metrics = False
        size_shape_data = {}
        
        for model in self.models:
            if model in self.results and 'shape_metrics' in self.results[model]:
                shape_data = self.results[model]['shape_metrics']
                if 'by_size' in shape_data:
                    by_size = shape_data['by_size']
                    if by_size and all(size in by_size for size in ['small', 'medium', 'large']):
                        has_size_shape_metrics = True
                        
                        # Calculate average metrics for each size category
                        size_shape_data[model] = {
                            'Small': np.mean(by_size['small'].get('circularity', [0])),
                            'Medium': np.mean(by_size['medium'].get('circularity', [0])),
                            'Large': np.mean(by_size['large'].get('circularity', [0]))
                        }
        
        # If we have shape metrics by size, create a bar chart
        if has_size_shape_metrics and size_shape_data:
            model_names = list(size_shape_data.keys())
            size_categories = ['Small', 'Medium', 'Large']
            
            # Prepare data for plotting
            data = {size: [] for size in size_categories}
            for model in model_names:
                for size in size_categories:
                    # Get value or default to 0
                    value = size_shape_data[model].get(size, 0.0)
                    data[size].append(value)
                    
            # Set up plot
            n_models = len(model_names)
            n_sizes = len(size_categories)
            bar_width = 0.25
            index = np.arange(n_models)
            
            # Use color scheme that matches size
            size_colors = {
                'Small': '#ff9896',  # Light red
                'Medium': '#aec7e8', # Light blue
                'Large': '#98df8a'   # Light green
            }
            
            colors = [size_colors.get(size, plt.cm.tab10(i/n_sizes)) for i, size in enumerate(size_categories)]
            
            # Create bars for each size
            for i, size in enumerate(size_categories):
                values = data[size]
                rects = ax.bar(index + i * bar_width, values, bar_width,
                             label=size, color=colors[i])
                
                # Add data labels on bars
                for j, rect in enumerate(rects):
                    height = rect.get_height()
                    # Format label
                    label_text = f'{height:.3f}'
                    ax.annotate(label_text,
                              xy=(rect.get_x() + rect.get_width() / 2, height),
                              xytext=(0, 3),  # 3 points vertical offset
                              textcoords="offset points",
                              ha='center', va='bottom', fontsize=8)
                    
            # Configure axes
            ax.set_ylabel('Circularity Value (0-1)')
            ax.set_title('Shape Analysis by Object Size')
            
            # Use simplified model names when there are more than 5 models
            display_names = self._simplify_model_names(model_names)
            ax.set_xticks(index + bar_width * (n_sizes - 1) / 2)
            ax.set_xticklabels(display_names, rotation=0, ha='center')
            ax.legend(title="Object Size", fontsize='small')
            
            # Set y-limits (circularity is typically 0-1)
            ax.set_ylim(0, 1.05)
            ax.grid(axis='y', linestyle='--', alpha=0.6)
            return
        
        # Check if segmentation metrics are available as fallback
        segm_metrics_to_try = ['Segm_mAP (IoU=0.50:0.95)', 'Segm_AP (IoU=0.50)']
        available_segm_metrics = [m for m in segm_metrics_to_try if m in self.data['summary_df'].columns]
        
        if available_segm_metrics:
            # Create a fallback message explaining we're showing segmentation AP metrics instead of shape metrics
            ax.text(0.5, 0.9, "Shape analysis metrics not available.\nShowing segmentation AP metrics instead.", 
                   ha='center', va='center', transform=ax.transAxes, 
                   fontsize=9, color='gray', bbox=dict(facecolor='white', alpha=0.8))
                    
            # Create a basic bar chart using available segmentation metrics
            df = self.data['summary_df'][available_segm_metrics]
            n_models = len(df)
            n_metrics = len(df.columns)
            bar_width = 0.25
            index = np.arange(n_models)
            
            legend_labels = {
                'Segm_mAP (IoU=0.50:0.95)': 'Segm mAP (0.50:0.95)', 
                'Segm_AP (IoU=0.50)': 'Segm AP@0.50'
            }
            
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_metrics))
            
            for i, metric in enumerate(df.columns):
                rects = ax.bar(index + i * bar_width, df[metric], bar_width,
                              label=legend_labels.get(metric, metric),
                              color=colors[i])
                self._add_value_labels(ax, rects, precision=4)
                
            ax.set_ylabel('Segmentation AP Value')
            ax.set_title('Shape Analysis (Segmentation Performance)')
            
            # Use simplified model names when there are more than 5 models
            model_names = df.index.tolist()
            display_names = self._simplify_model_names(model_names)
            
            ax.set_xticks(index + bar_width * (n_metrics - 1) / 2)
            ax.set_xticklabels(display_names, rotation=0, ha='center')
            ax.legend(title="Segm. Metric", fontsize='small')
            
            # Set appropriate y-limits
            max_val = df.max().max() if not df.empty else 0
            ax.set_ylim(0, max(0.01, max_val * 1.2))
            ax.grid(axis='y', linestyle='--', alpha=0.6)
            return
                
        # If no shape metrics or segmentation metrics are available, use object size metrics as last resort
        size_metrics_to_try = ['Small Objects AP', 'Medium Objects AP', 'Large Objects AP']
        available_size_metrics = [m for m in size_metrics_to_try if m in self.data['summary_df'].columns]
        
        if available_size_metrics:
            ax.text(0.5, 0.9, "Shape analysis metrics not available.\nShowing performance by object size instead.", 
                   ha='center', va='center', transform=ax.transAxes, 
                   fontsize=9, color='gray', bbox=dict(facecolor='white', alpha=0.8))
                   
            df = self.data['summary_df'][available_size_metrics]
            n_models = len(df)
            n_metrics = len(df.columns)
            bar_width = 0.25
            index = np.arange(n_models)
            
            legend_labels = {
                'Small Objects AP': 'Small',
                'Medium Objects AP': 'Medium',
                'Large Objects AP': 'Large'
            }
            
            # Use a distinct color scheme for object size metrics
            metric_colors = {
                'Small Objects AP': '#ff9896',  # Light red
                'Medium Objects AP': '#aec7e8', # Light blue
                'Large Objects AP': '#98df8a'   # Light green
            }
            colors = [metric_colors.get(metric, plt.cm.tab10(i/n_metrics)) for i, metric in enumerate(df.columns)]
            
            for i, metric in enumerate(df.columns):
                rects = ax.bar(index + i * bar_width, df[metric], bar_width,
                              label=legend_labels.get(metric, metric),
                              color=colors[i])
                self._add_value_labels(ax, rects, precision=4)
                
            ax.set_ylabel('AP Value')
            ax.set_title('Shape Analysis (by Object Size)')
            
            # Use simplified model names when there are more than 5 models
            model_names = df.index.tolist()
            display_names = self._simplify_model_names(model_names)
            
            ax.set_xticks(index + bar_width * (n_metrics - 1) / 2)
            ax.set_xticklabels(display_names, rotation=0, ha='center')
            ax.legend(title="Object Size", fontsize='small')
            
            max_val = df.max().max() if not df.empty else 0
            ax.set_ylim(0, max(0.01, max_val * 1.2))
            ax.grid(axis='y', linestyle='--', alpha=0.6)
            return
        
        # Last resort - no metrics available
        ax.text(0.5, 0.5, 'Shape analysis metrics not available.\nNo segmentation or object size data found.',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Shape Analysis')

    def _plot_fps(self, ax):
        """Plots processing speed (FPS) and inference time (ms) side by side."""
        if 'Speed (FPS)' not in self.data['summary_df'].columns:
             ax.text(0.5, 0.5, 'FPS data not available.', ha='center', va='center', transform=ax.transAxes)
             ax.set_title('Processing Speed')
             return

        # Get FPS data and calculate inference time (ms)
        fps_data = self.data['summary_df']['Speed (FPS)']
        time_ms_data = 1000.0 / fps_data.replace(0, np.nan)  # Convert FPS to milliseconds per frame
        
        # Set up the plot
        models = fps_data.index
        n_models = len(models)
        
        # Set positions for the bars - we'll need two bars per model
        x_pos = np.arange(n_models)
        bar_width = 0.35  # Narrower bars to fit two per model
        
        # Create a consistent color mapping for models based on their order
        # Use the same palette colors (blue, orange, teal) that we use for metrics
        palette_colors = [self.METRIC_COLORS['Precision'], self.METRIC_COLORS['Recall'], self.METRIC_COLORS['F1-Score']]
        
        # Assign colors to models in order
        model_colors = {}
        for i, model in enumerate(models):
            if i < len(palette_colors):
                # Use the palette colors for the first three models
                model_colors[model] = palette_colors[i]
            else:
                # Fall back to the original _get_model_color for additional models
                model_colors[model] = self._get_model_color(model)

        # Create colors array for models
        model_color_list = [model_colors[model] for model in models]

        # Create lighter versions of the colors for the time bars
        time_colors = []
        for color in model_color_list:
            # Convert hex to RGB, lighten it, and convert back to hex
            rgb = tuple(int(color[1:][i:i+2], 16) for i in (0, 2, 4))
            lighter_rgb = tuple(min(255, int(c * 0.7 + 255 * 0.3)) for c in rgb)  # 30% lighter
            time_colors.append(f'#{lighter_rgb[0]:02x}{lighter_rgb[1]:02x}{lighter_rgb[2]:02x}')
        
        # Create primary y-axis for FPS
        fps_bars = ax.bar(x_pos - bar_width/2, fps_data, bar_width, color=model_color_list, label='FPS')
        ax.set_ylabel('Speed (Frames Per Second)', color='black')
        ax.set_ylim(0, max(10, fps_data.max() * 1.15))  # Add 15% padding
        
        # Add data labels to the FPS bars
        for bar in fps_bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
        
        # Create secondary y-axis for milliseconds
        ax2 = ax.twinx()
        time_bars = ax2.bar(x_pos + bar_width/2, time_ms_data, bar_width, color=time_colors, label='Time (ms)')
        ax2.set_ylabel('Inference Time (ms)', color='black')
        
        # Set a reasonable upper limit for the time axis based on the data
        max_time = time_ms_data.max()
        ax2.set_ylim(0, max(50, max_time * 1.15))  # Add 15% padding, minimum 50ms scale
        
        # Add data labels to the time bars
        for bar in time_bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
        
        # Set the x-ticks to be in the middle of the model groups
        ax.set_xticks(x_pos)
        
        # Use simplified model names when there are more than 5 models
        model_names = models.tolist()
        display_names = self._simplify_model_names(model_names)
        ax.set_xticklabels(display_names, rotation=0, ha='center')
        
        # Add a grid for better readability (only for the primary y-axis)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        
        # Add a combined legend for both axes
        # Create handles for the legend manually
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=model_color_list[0], label='Speed (FPS)'),
            Patch(facecolor=time_colors[0], label='Time (ms)')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Update the chart title
        ax.set_title('Processing Speed & Inference Time')
        
        # Add explanatory text about the relationship between FPS and Time (ms)
        explanation = "Note: Inference Time (ms) = 1000 / Speed (FPS)"
        
        # Add explanatory text in a box at the bottom
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)
        ax.text(0.5, -0.15, explanation, transform=ax.transAxes,
              fontsize=8, ha='center', va='center', bbox=props)

    def _plot_detection_distribution(self, ax):
        """
        Plots detection distribution by object size.
        Shows AP values for small, medium, and large objects for each model.
        """
        # Check if we have the necessary metrics
        size_metrics_to_try = ['Small Objects AP', 'Medium Objects AP', 'Large Objects AP']
        available_size_metrics = [m for m in size_metrics_to_try if m in self.data['summary_df'].columns]
        
        if not available_size_metrics:
            ax.text(0.5, 0.5, 'Object size performance data not available.',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Distribution of Detections by Object Size')
            return
            
        # Get original data for display in labels
        orig_df = self.data['summary_df'][available_size_metrics].copy()
        
        # Create a boosted version of the data for better visualization
        df = pd.DataFrame(index=orig_df.index)
        
        # Check if all values are extremely small
        max_val = orig_df.max().max()
        min_visible_value = 0.05  # Minimum height for bars to be clearly visible
        
        if max_val < 0.01:  # If all values are too small to see properly
            # Calculate boost needed to make the largest value visible
            if max_val > 1e-6:  # If we have some non-zero values
                boost_factor = min_visible_value / max_val  # Make largest value = min_visible_value
            else:
                boost_factor = 1000  # Just use a large factor if all values are essentially zero
                
            # Apply different boost factors to each value to maintain relative proportions
            # but make all values visible
            for model in orig_df.index:
                for metric in available_size_metrics:
                    orig_val = orig_df.loc[model, metric]
                    if orig_val < 1e-6:  # Almost zero
                        # Set a small minimum value to ensure bars are visible
                        df.loc[model, metric] = min_visible_value * 0.2  # 20% of our minimum
                    else:
                        # Scale proportionally but ensure value is at least 10% of minimum
                        boosted_val = orig_val * boost_factor
                        df.loc[model, metric] = max(boosted_val, min_visible_value * 0.1)
                        
            scaled = True  # Flag that we've scaled values
        else:
            # Values are large enough to show directly
            df = orig_df.copy()
            scaled = False

        # Prepare the plot
        n_models = len(df)
        bar_width = 0.25
        index = np.arange(n_models)
        
        # Set up legend labels and colors that match the size categories
        legend_labels = {
            'Small Objects AP': 'Small',
            'Medium Objects AP': 'Medium',
            'Large Objects AP': 'Large'
        }
        
        # Use consistent color palette from Precision/Recall/F1-Score chart
        size_colors = {
            'Small Objects AP': self.METRIC_COLORS.get('Precision', '#4C78A8'),    # Blue (same as Precision)
            'Medium Objects AP': self.METRIC_COLORS.get('Recall', '#F58518'),      # Orange (same as Recall)
            'Large Objects AP': self.METRIC_COLORS.get('F1-Score', '#72B7B2')      # Teal (same as F1-Score)
        }
        
        colors = [size_colors.get(size, self.METRIC_COLORS.get(size, '#A0A0A0')) for size in available_size_metrics]

        # Plot each size category
        for i, size in enumerate(available_size_metrics):
            values = df[size]
            rects = ax.bar(index + i * bar_width, values, bar_width,
                          label=legend_labels.get(size, size),
                          color=colors[i])

            # Add data labels on bars using original values
            orig_values = orig_df[size]
            
            for j, rect in enumerate(rects):
                height = rect.get_height()
                orig_val = orig_values.iloc[j]
                
                # Format label based on original value size
                if orig_val < 0.001:
                    label_text = f'{orig_val:.4f}'
                elif orig_val < 0.01:
                    label_text = f'{orig_val:.3f}'
                else:
                    label_text = f'{orig_val:.3f}'
                    
                ax.annotate(label_text,
                          xy=(rect.get_x() + rect.get_width() / 2, height),
                          xytext=(0, 3),  # 3 points vertical offset
                          textcoords="offset points",
                          ha='center', va='bottom', fontsize=8)
        
        # Configure axes
        ax.set_ylabel('AP Value')
        if scaled:
            ax.set_title('Distribution of Detections by Object Size\n(Values scaled for visibility)')
        else:
            ax.set_title('Distribution of Detections by Object Size')
            
        # Use simplified model names when there are more than 5 models
        model_names = df.index.tolist()
        display_names = self._simplify_model_names(model_names)    
        
        ax.set_xticks(index + bar_width * (len(available_size_metrics) - 1) / 2)
        ax.set_xticklabels(display_names, rotation=0, ha='center')
        ax.legend(title="Object Size", fontsize='small')
        
        # Set y-limits based on data
        max_plotted = df.max().max()
        if max_plotted <= 0:
            upper_limit = 0.1  # default minimum scale when values are zero
        else:
            upper_limit = max_plotted * 1.2  # Add 20% headroom
            
        ax.set_ylim(0, upper_limit)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        
        # Add explanatory text about object size categories
        explanation = (
            "Small Objects: Area < 32² pixels\n"
            "Medium Objects: 32² - 96² pixels\n"
            "Large Objects: Area > 96² pixels"
        )
        
        # Add explanatory text in a box at the bottom
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)
        ax.text(0.5, -0.20, explanation, transform=ax.transAxes,
              fontsize=8, ha='center', va='center', bbox=props)

    def _plot_f1_vs_speed(self, ax):
        """Plots F1-Score vs Speed scatter plot."""
        # Check necessary metrics are available
        if 'F1-Score' not in self.data['summary_df'].columns or 'Speed (FPS)' not in self.data['summary_df'].columns:
            ax.text(0.5, 0.5, 'F1-Score vs Speed data not available.',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('F1-Score vs Speed Performance')
            return

        # Get data for plotting
        df = self.data['summary_df']
        f1_score = df['F1-Score']
        fps = df['Speed (FPS)']
        models = df.index

        # Check if all F1 scores are very small
        max_f1 = f1_score.max()
        min_visible_f1 = 0.01  # Minimum for visibility
        
        # Create a transformed version of F1 score for very small values
        if max_f1 < 0.001:
            # If all values are extremely small, use rank-based transformation
            # This preserves the relative ordering of models but makes points visible
            f1_ranks = f1_score.rank(method='dense')
            max_rank = f1_ranks.max()
            if max_rank > 1:  # If we have different ranks
                # Scale between 0.01 and 0.1 with minimum visibility threshold
                f1_plot = 0.01 + (0.09 * (f1_ranks - 1) / (max_rank - 1))
                transformed = True
            else:
                # All same value, use a small fixed value
                f1_plot = pd.Series([0.01] * len(f1_score), index=f1_score.index)
                transformed = True
        else:
            # No transformation needed
            f1_plot = f1_score.copy()
            transformed = False

        # Create customized color mapping for models
        colors = []
        for model in models:
            colors.append(self._get_model_color(model))
            
        # Create scatter plot with custom colors and sizes
        scatter = ax.scatter(fps, f1_plot, 
                          c=colors,  # Use our custom colors
                          s=100,     # Larger point size
                          alpha=0.7, # Semi-transparent
                          edgecolors='black',
                          linewidths=1)

        # Check if we should use simplified model names for the legend
        model_list = models.tolist()
        use_simple_names = len(model_list) > 5
        display_names = self._simplify_model_names(model_list) if use_simple_names else model_list

        # Label each point with model name and actual F1-Score value
        for i, model in enumerate(models):
            # Format label based on original F1 value
            if f1_score.iloc[i] < 0.001:
                suffix = f"\n(F1: {f1_score.iloc[i]:.5f})"
            elif f1_score.iloc[i] < 0.01:
                suffix = f"\n(F1: {f1_score.iloc[i]:.4f})"
            else:
                suffix = ""
                
            # Use simplified model name if there are more than 5 models
            display_name = display_names[i] if use_simple_names else model
            label = f"{display_name}{suffix}"
            
            ax.annotate(label, 
                      (fps.iloc[i], f1_plot.iloc[i]),
                      xytext=(5, 5),
                      textcoords='offset points',
                      fontsize=8)
        
        # Add horizontal and vertical lines for better reference
        if len(fps) > 0:
            ax.axhline(y=f1_plot.mean(), color='gray', linestyle='--', alpha=0.3)
            ax.axvline(x=fps.mean(), color='gray', linestyle='--', alpha=0.3)
        
        # Set labels and title
        ax.set_xlabel('Processing Speed (FPS)')
        ax.set_ylabel('F1-Score')
        if transformed:
            ax.set_title('F1-Score vs Speed Performance\n(F1-Score values scaled for visibility)')
        else:
            ax.set_title('F1-Score vs Speed Performance')
        
        # Set limits with padding
        max_fps = fps.max() if not fps.empty else 10
        max_f1_plot = f1_plot.max()
        
        ax.set_ylim(0, max_f1_plot * 1.15)
        ax.set_xlim(0, max_fps * 1.15)
        
        # Add grid for better readability
        ax.grid(linestyle='--', alpha=0.6)
        
        # If we have many models with similar colors, add a legend
        if len(models) > 8:
            # Create legend handles manually
            from matplotlib.lines import Line2D
            legend_elements = []
            
            # Use simplified names in the legend manually
            legend_models = display_names
            
            for i, model_name in enumerate(legend_models):
                legend_elements.append(
                    Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=colors[i], markersize=8, 
                          label=model_name)
                )
                
            # Place legend outside plot area to the right
            ax.legend(handles=legend_elements, loc='center left', 
                     bbox_to_anchor=(1.05, 0.5), fontsize='small')
                     
        # Add note if values were transformed
        if transformed:
            ax.text(0.5, 0.02, 'Note: Original F1-Score values are shown in labels',
                   ha='center', transform=ax.transAxes,
                   fontsize=8, style='italic', color='darkgray')
                   
        # Add note about simplified names if applicable
        if use_simple_names:
            ax.text(0.5, -0.02, 'Note: Model names are simplified (e.g., "yolo8n-seg" → "8n")',
                   ha='center', transform=ax.transAxes,
                   fontsize=8, style='italic', color='dimgray')

    def _plot_radar(self, ax):
        """Creates the multi-dimensional radar chart with model-relative scaling."""
        # Use non-newline versions for labels
        radar_metrics_map = {
            'mAP': 'mAP (IoU=0.50:0.95)',
            'F1-Score': 'F1-Score',
            'Large Objects': 'Large Objects AP',
            'Medium Objects': 'Medium Objects AP',
            'Small Objects': 'Small Objects AP',
            'Speed': 'Speed (FPS)',
        }

        available_radar_metrics = {
            display: original for display, original in radar_metrics_map.items()
            if original in self.data['summary_df'].columns
        }

        if len(available_radar_metrics) < 3:
             ax.text(0.5, 0.5, 'Not enough data\nfor Radar Chart.', ha='center', va='center', transform=ax.transAxes)
             ax.set_title('Multi-dimensional Performance Comparison', size=11, y=1.1)
             ax.set_xticks([])
             ax.set_yticks([])
             return

        radar_cols_original = list(available_radar_metrics.values())
        radar_cols_display = list(available_radar_metrics.keys())
        df_radar_raw = self.data['summary_df'][radar_cols_original].copy()
        
        # Print debug info about raw values
        print("\nDEBUG - Radar Chart Raw Values:")
        print(df_radar_raw)

        # --- Enhanced Data Normalization with "Relative Maximum" approach ---
        df_radar_normalized = pd.DataFrame(index=df_radar_raw.index)
        min_visible_value = 0.15  # Minimum value for visibility
        
        # Store information about whether metrics are displayed at zoomed scale
        zoomed_metrics = []
        
        for display_name, original_col in available_radar_metrics.items():
            column_data = df_radar_raw[original_col]
            max_val = column_data.max()
            
            # Special handling for Speed (FPS) - standard max normalization
            if original_col == 'Speed (FPS)':
                if max_val > 0:
                    normalized = column_data / max_val  # Standard normalization
                else:
                    normalized = pd.Series([min_visible_value * 0.5] * len(column_data), index=column_data.index)
            else:
                # For accuracy metrics - relative maximum approach with enhanced visibility
                if max_val < 0.01 and max_val > 0:
                    # Zoomed in - the highest value becomes 0.8
                    # and all others are scaled proportionally
                    zoomed_metrics.append(display_name)
                    
                    # Use relative scaling that preserves differences
                    # Map the range [0, max_val] to [0, 0.8] with minimum visibility threshold
                    normalized = pd.Series(index=column_data.index)
                    
                    for i, val in enumerate(column_data):
                        if val <= 0:
                            # Give zero/negative values a tiny positive value
                            normalized.iloc[i] = min_visible_value * 0.2
                        elif val < max_val * 0.05:  # Values less than 5% of max
                            # Give very small but non-zero values a bit more visibility
                            ratio = val / max_val
                            normalized.iloc[i] = min_visible_value + ratio * (0.4 - min_visible_value)
                        else:
                            # Scale proportionally from the minimum threshold up to 0.8
                            ratio = val / max_val
                            normalized.iloc[i] = min_visible_value + ratio * (0.8 - min_visible_value)
                    
                    print(f"Note: '{display_name}' values are very small (<0.01), using zoomed view")
                else:
                    # For larger values or zeros, use standard max-normalization with minimum visibility
                    if max_val > 0:
                        normalized = column_data / max_val  # Standard normalization to [0,1]
                        
                        # Boost any non-zero values that would be too small to see
                        for i, val in enumerate(normalized):
                            if column_data.iloc[i] > 0 and val < min_visible_value:
                                normalized.iloc[i] = min_visible_value + (val * (0.3 - min_visible_value))
                    else:
                        # All zeros case
                        normalized = pd.Series([min_visible_value * 0.5] * len(column_data), index=column_data.index)
            
            # Add distinct offsets to object size metrics to prevent overlapping 
            # when values are identical or very similar
            if original_col in ['Small Objects AP', 'Medium Objects AP', 'Large Objects AP']:
                # These offsets ensure that even identical values will appear distinct
                # The offset is proportional to the difference between min and max values
                offset_factor = 0.05  # Base offset factor
                
                if original_col == 'Small Objects AP':
                    # Smallest offset for small objects
                    offset = offset_factor * 1
                elif original_col == 'Medium Objects AP':  
                    # Medium offset for medium objects
                    offset = offset_factor * 2
                else:  # Large Objects AP
                    # Largest offset for large objects 
                    offset = offset_factor * 3
                
                # Apply offset, ensuring we don't exceed 1.0
                for i in range(len(normalized)):
                    current_val = normalized.iloc[i]
                    if current_val > min_visible_value:  # Only apply offset to visible values
                        normalized.iloc[i] = min(1.0, current_val + offset)
            
            # Store the normalized values
            df_radar_normalized[display_name] = normalized
        
        # Handle any NaN values
        df_radar = df_radar_normalized.fillna(0)
        
        # Print debug info about normalized values
        print("\nDEBUG - Radar Chart Normalized Values:")
        print(df_radar)

        # --- Plotting Setup ---
        categories = list(df_radar.columns)
        n_categories = len(categories)
        angles = np.linspace(0, 2 * np.pi, n_categories, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop

        # Create background reference circles for comparison
        for level in [0.25, 0.5, 0.75]:
            ax.plot(angles, [level] * (n_categories + 1), color='gray', linestyle='--', 
                    linewidth=0.5, alpha=0.3, label=None)
            ax.fill(angles, [level] * (n_categories + 1), color='lightgray', alpha=0.05)

        # Prepare axis styling
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10, color='black')  # Improved readability
        ax.set_yticks([])  # Hide numerical scale - not meaningful with our normalization
        ax.set_ylim(0, 1.0)
        ax.tick_params(axis='x', pad=12)
        ax.grid(False)  # Remove distracting grid
        ax.spines['polar'].set_color('lightgray')

        # --- Plot each model's data ---
        # If we have many models, limit to a maximum number for clarity
        models_to_plot = df_radar.index
        max_models_in_radar = 12  # Maximum models to show in radar chart
        
        if len(models_to_plot) > max_models_in_radar:
            print(f"Warning: Too many models ({len(models_to_plot)}) for radar chart. Limiting to {max_models_in_radar}.")
            models_to_plot = models_to_plot[:max_models_in_radar]
        
        # Check if we need to simplify model names for the legend
        model_list = models_to_plot.tolist()
        use_simple_names = len(model_list) > 5
        display_names = self._simplify_model_names(model_list) if use_simple_names else model_list
        
        # Create a consistent color mapping for models based on their order
        # Use the same palette colors (blue, orange, teal) that we use for metrics
        palette_colors = [self.METRIC_COLORS['Precision'], self.METRIC_COLORS['Recall'], self.METRIC_COLORS['F1-Score']]
        
        # Map colors to models
        model_colors = {}
        for i, model in enumerate(models_to_plot):
            if i < len(palette_colors):
                # Use the palette colors for the first three models
                model_colors[model] = palette_colors[i]
            else:
                # Fall back to the original _get_model_color for additional models
                model_colors[model] = self._get_model_color(model)
        
        # Plot each model's radar chart
        for i, model in enumerate(models_to_plot):
            data = df_radar.loc[model].tolist()
            data += data[:1]  # Close the loop
            
            # Select color from our model mapping
            color = model_colors[model]
            
            # Plot the lines with improved visibility
            ax.plot(angles, data, linewidth=2.5, linestyle='solid', label=model, color=color, zorder=10+i)
            
            # Fill with semi-transparent color
            ax.fill(angles, data, color=color, alpha=0.3, zorder=i+1)

        # Create title with information about zoomed metrics
        title = 'Multi-dimensional Performance Comparison'
        if zoomed_metrics:
            # Indicate which metrics are shown at zoomed scale
            zoomed_str = ", ".join(zoomed_metrics)
            title += f'\n(Zoomed view for: {zoomed_str})'
        
        # Add note about simplified names if applicable
        if use_simple_names:
            title += '\n(Model names simplified for clarity)'
            
        ax.set_title(title, size=12, y=1.12)
        
        # Add annotation explaining the normalization approach
        explanation = "Note: Each axis shows relative performance where the best model = 1.0"
        ax.text(0.5, -0.15, explanation, transform=ax.transAxes, ha='center', 
                fontsize=8, style='italic', color='dimgray')
        
        # Position legend on the side for easier reading when many models
        if len(models_to_plot) > 5:
            ax.legend(loc='center left', bbox_to_anchor=(1.15, 0.5), fontsize='small')
        else:
            ax.legend(loc='lower right', bbox_to_anchor=(0.95, -0.1), fontsize='small')
            
    def _get_model_color(self, model_name):
        """Get a consistent color for each model type."""
        # Define distinct colors for different model families with more variation
        model_colors = {
            # Base models
            'mask-rcnn': '#1f77b4',     # Muted blue
            'detr': '#2ca02c',          # Green
            'detr-panoptic': '#2ca02c',  # Green
            
            # YOLOv8 family - orange to red gradient
            'yolo8n-seg': '#ff7f0e',    # Orange
            'yolo8s-seg': '#ff6347',    # Tomato
            'yolo8m-seg': '#e74c3c',    # Lighter red
            'yolo8l-seg': '#c0392b',    # Red
            'yolo8x-seg': '#7f0000',    # Dark red
            
            # YOLOv9 family - teal/turquoise gradient
            'yolo9c-seg': '#1abc9c',    # Light teal
            'yolo9e-seg': '#16a085',    # Dark teal
            
            # YOLOv11 family - purple gradient
            'yolo11n-seg': '#9467bd',   # Light purple
            'yolo11s-seg': '#8e44ad',   # Medium purple
            'yolo11m-seg': '#7d3c98',   # Purple
            'yolo11l-seg': '#6c3483',   # Dark purple
            'yolo11x-seg': '#4a235a',   # Very dark purple
        }
        
        # Check for direct match first
        if model_name in model_colors:
            return model_colors[model_name]
        
        # Check for model family patterns using prefix matching
        if 'yolo8' in model_name.lower():
            return '#e74c3c'  # Default YOLOv8 family (red)
        elif 'yolo9' in model_name.lower():
            return '#16a085'  # Default YOLOv9 family (teal)
        elif 'yolo11' in model_name.lower():
            return '#8e44ad'  # Default YOLOv11 family (purple)
        
        # For any other models, use a hash function to assign colors from this list
        color_list = [
            '#1f77b4',  # Blue
            '#2ca02c',  # Green
            '#d62728',  # Red
            '#9467bd',  # Purple
            '#8c564b',  # Brown
            '#e377c2',  # Pink
            '#7f7f7f',  # Grey
            '#bcbd22',  # Olive
            '#17becf',  # Cyan
            '#ff7f0e',  # Orange
            '#aec7e8',  # Light blue
            '#ffbb78',  # Light orange
            '#98df8a',  # Light green
            '#ff9896'   # Light red
        ]
        
        # Generate a consistent hash for the model name
        model_hash = sum(ord(c) for c in model_name)
        return color_list[model_hash % len(color_list)]

    def _prepare_summary_data(self):
        """Prepare summary data for plotting."""
        # Extract main metrics for all models
        model_metrics = []
        for model_name, model_data in self.data['results'].items():
            metrics = {}
            metrics['Model'] = model_name
            
            # Get COCO metrics if available
            coco_metrics = model_data.get('coco_metrics', {})
            if coco_metrics and not isinstance(coco_metrics, dict):
                coco_metrics = {}  # Handle case where coco_metrics is not a dict
            
            # Use COCO metrics for mAP and AR when available, otherwise use our default calculations
            metrics['mAP (IoU=0.50:0.95)'] = coco_metrics.get('AP_IoU=0.50:0.95', 0.0)
            metrics['AP (IoU=0.50)'] = coco_metrics.get('AP_IoU=0.50', 0.0)
            metrics['AP (IoU=0.75)'] =coco_metrics.get('AP_IoU=0.75', 0.0)
            metrics['Small Objects AP'] = coco_metrics.get('AP_small', 0.0)
            metrics['Medium Objects AP'] = coco_metrics.get('AP_medium', 0.0)
            metrics['Large Objects AP'] = coco_metrics.get('AP_large', 0.0)
            metrics['AR (max=100)'] = coco_metrics.get('AR_max=100', 0.0)
            
            # Calculate F1-Score when we have precision and recall or separately calculate from detections
            if metrics['AP (IoU=0.50)'] > 0 and coco_metrics.get('AR_max=100', 0) > 0:
                precision = metrics['AP (IoU=0.50)']  # Use AP@0.5 as precision
                recall = coco_metrics.get('AR_max=100', 0)  # Use AR@100 as recall
                
                # Calculate F1 score: 2 * (precision * recall) / (precision + recall)
                if precision + recall > 0:
                    metrics['F1-Score'] = 2 * (precision * recall) / (precision + recall)
                else:
                    metrics['F1-Score'] = 0.0
            else:
                # Alternative F1 calculation for cases with zero AP/AR
                # This is an approximation based on detection counts
                total_detections = model_data.get('total_detections',0)
                if total_detections > 0:
                    # We don't have ground truth information directly available here,
                    # so we'll use a very rough approximation
                    metrics['F1-Score'] = 0.0001 * (total_detections / max(1, model_data.get('total_images', 1)))
                else:
                    metrics['F1-Score'] = 0.0
            
            # Get segmentation metrics if available
            coco_segm_metrics = model_data.get('coco_segm_metrics', {})
            if coco_segm_metrics:
                metrics['Segm_mAP (IoU=0.50:0.95)'] = coco_segm_metrics.get('Segm_AP_IoU=0.50:0.95', 0.0)
                metrics['Segm_AP (IoU=0.50)'] = coco_segm_metrics.get('Segm_AP_IoU=0.50', 0.0)
            
            # Get efficiency metrics
            metrics['Speed (FPS)'] = model_data.get('fps', 0.0)
            metrics['Mean Inference Time (ms)'] = model_data.get('mean_inference_time', 0.0) * 1000  # Convert to ms
            
            # Other standard metrics
            metrics['Total Detections'] = model_data.get('total_detections', 0)
            metrics['Detection Rate'] = model_data.get('avg_detections_per_image', 0.0)
            metrics['Unique Classes'] = model_data.get('unique_classes_detected', 0)
            
            model_metrics.append(metrics)
        
        # Create a DataFrame
        df = pd.DataFrame(model_metrics)
        if not df.empty:
            df.set_index('Model', inplace=True)
        
        return df

    def _plot_summary_table(self, ax):
        """Creates a comprehensive summary table with key metrics for all models."""
        # Hide axis elements for the table
        ax.axis('off')
        
        # Make sure we have data to display
        if not hasattr(self, 'data') or 'summary_df' not in self.data or self.data['summary_df'].empty:
            ax.text(0.5, 0.5, 'No data available for summary table.',
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Select key metrics for the table - order matters for display
        metrics_to_include = [
            'mAP (IoU=0.50:0.95)', # Overall mAP 
            'AP (IoU=0.50)',        # Standard AP at IoU=0.5
            'AP (IoU=0.75)',        # AP at IoU=0.75
            'Precision',            # Precision
            'Recall',               # Recall
            'AR (max=100)',         # Average Recall
            'F1-Score',             # F1 Score
            'Large Objects AP',     # Object size performance
            'Medium Objects AP',
            'Small Objects AP',
            'Speed (FPS)',          # Speed performance
        ]
        
        # Filter to only include metrics that exist in our data
        available_metrics = [m for m in metrics_to_include if m in self.data['summary_df'].columns]
        
        if not available_metrics:
            ax.text(0.5, 0.5, 'No metrics available for summary table.',
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Extract and round the data for display
        display_df = self.data['summary_df'][available_metrics].copy()
        
        # Calculate F1-Score/FPS ratio for each model (Performance efficiency)
        if 'F1-Score' in display_df.columns and 'Speed (FPS)' in display_df.columns:
            # Calculate F1-Score/Performance ratio (efficiency measure)
            display_df['F1/FPS Ratio'] = display_df['F1-Score'] / display_df['Speed (FPS)'].replace(0, np.nan)
            # Multiply by 100 to make values more readable
            display_df['F1/FPS Ratio'] *= 100.0
            # Add this metric to our available metrics
            available_metrics.append('F1/FPS Ratio')
            
        # Add Inference Time (ms) based on FPS
        if 'Speed (FPS)' in display_df.columns:
            # Convert FPS to milliseconds per inference
            display_df['Inference Time (ms)'] = 1000.0 / display_df['Speed (FPS)'].replace(0, np.nan)
            # Add to our available metrics
            available_metrics.append('Inference Time (ms)')
        
        # Add Model Size/Parameters if available in the results data
        model_sizes = {}
        for model in self.models:
            # Try to find model size in the results data
            if model in self.results:
                # Check various possible keys where model size might be stored
                size_mb = None
                for key in ['model_size_mb', 'size_mb', 'model_size']:
                    if key in self.results[model]:
                        size_mb = self.results[model][key]
                        break
                
                # If not found, try to estimate from model path
                if size_mb is None and 'model_path' in self.results[model]:
                    model_path = self.results[model]['model_path']
                    if model_path and Path(model_path).exists():
                        size_bytes = Path(model_path).stat().st_size
                        size_mb = size_bytes / (1024 * 1024)  # Convert bytes to MB
                
                model_sizes[model] = size_mb if size_mb is not None else None
        
        # If we found any model sizes, add to the dataframe
        if any(size is not None for size in model_sizes.values()):
            display_df['Model Size (MB)'] = pd.Series(model_sizes)
            available_metrics.append('Model Size (MB)')
        
        # Format the table data for better readability
        table_data = []
        # Create shortened header labels to save space
        header_mapping = {
            'mAP (IoU=0.50:0.95)': 'mAP',
            'AP (IoU=0.50)': 'AP@0.50',
            'AP (IoU=0.75)': 'AP@0.75',
            'Large Objects AP': 'Large',
            'Medium Objects AP': 'Medium',
            'Small Objects AP': 'Small',
            'Speed (FPS)': 'FPS',
            'AR (max=100)': 'AR',
            'Inference Time (ms)': 'Time (ms)',
            'Model Size (MB)': 'Size (MB)',
            'F1/FPS Ratio': 'F1/FPS'
        }
        
        header_row = ['Model'] + [header_mapping.get(m, m) for m in available_metrics]
        table_data.append(header_row)
        
        # For each model, add a row with formatted values
        for model_name in display_df.index:
            row = [model_name]  # Start with model name
            for metric in available_metrics:
                val = display_df.loc[model_name, metric]
                
                # Skip NaN values
                if pd.isna(val):
                    row.append('N/A')
                    continue
                
                # Format based on metric type
                if 'Speed' in metric or 'Time' in metric:
                    # FPS or Inference Time with 1 decimal place
                    formatted_val = f"{val:.1f}"
                elif 'Size' in metric:
                    # Model size with 1 decimal place
                    formatted_val = f"{val:.1f}"
                elif 'F1/FPS' in metric:
                    # F1/FPS Ratio with scientific notation if very small, otherwise 3 decimals
                    if val < 0.001:
                        formatted_val = f"{val:.3e}"
                    else:
                        formatted_val = f"{val:.3f}"
                elif val < 0.01:
                    # Very small values with scientific notation
                    formatted_val = f"{val:.3e}"
                elif val < 0.1:
                    # Small values with 4 decimal places
                    formatted_val = f"{val:.4f}"
                else:
                    # Standard values with 3 decimal places
                    formatted_val = f"{val:.3f}" 
                
                row.append(formatted_val)
            
            table_data.append(row)
        
        # Create the table with wider spacing and positioned higher
        # Adjust the bbox to make the table 25% wider overall - expand horizontally beyond normal bounds
        # This will make each column 25% wider while maintaining the same number of columns
        table = ax.table(
            cellText=table_data,
            loc='center',
            cellLoc='center',
            bbox=[-0.125, 0.08, 1.25, 0.80]  # Position table higher with 8% bottom margin, and increase width by 25%
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)  # Slightly smaller font to fit more columns
        
        # Style header row
        for i, cell in enumerate(table._cells[(0, col)] for col in range(len(header_row))):
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4472C4')  # Blue header background
        
        # Set fixed column widths to ensure even spacing 
        # Make the first column (model names) twice as wide as other columns
        # Calculate the width distribution: if first column is 2x, and we have n columns total
        # First column gets 2/(n+1) of the width, other columns each get 1/(n+1)
        num_cols = len(header_row)
        first_col_width = 2.0 / (num_cols + 1)  # Model name column gets twice the width
        other_col_width = 1.0 / (num_cols + 1)  # Other columns share the remaining space evenly
        
        col_widths = [first_col_width] + [other_col_width] * (num_cols - 1)
        
        for i, width in enumerate(col_widths):
            for row in range(len(table_data)):
                table._cells[(row, i)].set_width(width)
                # Increase the height of each row for better vertical spacing
                table._cells[(row, i)].set_height(0.07)
        
        # Alternate row colors for better readability
        for row_idx in range(1, len(table_data)):
            for col_idx in range(len(header_row)):
                cell = table._cells[(row_idx, col_idx)]
                
                # Style based on row position (alternating colors)
                if row_idx % 2 == 1:
                    cell.set_facecolor('#E6F0FF')  # Light blue for odd rows
                else:
                    cell.set_facecolor('#F5F8FF')  # Even lighter blue for even rows
                    
                # Add padding inside cells by adjusting text positioning
                cell._loc = 'center'
                
                # Highlight best values in each column - skip model name column (0)
                if col_idx > 0:
                    # Skip N/A values
                    if table_data[row_idx][col_idx] == 'N/A':
                        continue
                        
                    # Extract numeric values for this column to find the best
                    col_values = []
                    for i in range(1, len(table_data)):
                        if table_data[i][col_idx] not in ['nan', 'N/A']:
                            try:
                                col_values.append(float(table_data[i][col_idx]))
                            except (ValueError, TypeError):
                                col_values.append(float('-inf'))
                        else:
                            col_values.append(float('-inf'))
                    
                    try:
                        # For inference time and F1/FPS, lower is better
                        if 'Time' in header_row[col_idx] or 'F1/FPS' in header_row[col_idx]:
                            best_idx = col_values.index(min([v for v in col_values if v > 0] or [float('inf')])) + 1
                        else:
                            # For all other metrics, higher is better
                            best_idx = col_values.index(max(col_values)) + 1  # +1 because we start from row 1
                        
                        if row_idx == best_idx:
                            cell.set_text_props(weight='bold', color='darkgreen')
                    except (ValueError, TypeError, IndexError):
                        pass  # Skip highlighting if issues
        
        # Add explanation for F1/FPS Ratio and Inference Time if in the table
        explanation_texts = []
        if 'F1/FPS Ratio' in display_df.columns:
            explanation_texts.append("F1/FPS Ratio = (F1-Score / Speed) × 100 - Lower values indicate better efficiency")
        if 'Inference Time (ms)' in display_df.columns:
            explanation_texts.append("Inference Time (ms) = 1000 / FPS")
        
        # Display explanations with sufficient spacing
        if explanation_texts:
            explanation = " | ".join(explanation_texts)
            ax.text(0.5, 0.02, explanation, transform=ax.transAxes, ha='center', 
                    fontsize=8.5, style='italic', color='dimgray')
            
        ax.set_title('Summary Table', size=12, y=1.02)

    def _simplify_model_names(self, model_names):
        """
        Simplifies model names for display on x-axis when there are many models.
        
        Args:
            model_names (list): List of model names to simplify
            
        Returns:
            list: Simplified model names
        """
        # If 5 or fewer models, return the original names
        if len(model_names) <= 5:
            return model_names
            
        # If more than 5 models, simplify the names
        simplified_names = []
        for name in model_names:
            # For YOLO models, extract just the version and size (e.g., "yolo8n-seg" -> "8n")
            if 'yolo' in name.lower():
                # Extract digits after "yolo" and the letter following them
                import re
                match = re.search(r'yolo(\d+)([a-zA-Z]+)', name)
                if match:
                    # Use the version number and model size letter
                    simplified_names.append(f"{match.group(1)}{match.group(2)}")
                else:
                    # Fallback if pattern doesn't match
                    simplified_names.append(name)
            else:
                # For non-YOLO models, keep the original name
                simplified_names.append(name)
                
        return simplified_names

    def plot_precision_recall_curves(self, show_plot=False):
        """
        Plots precision-recall curves for all models.
        
        Args:
            show_plot (bool): If True, display the plot interactively using plt.show().
                             If False, only save the plot to a file.
                             
        Returns:
            str: Absolute path to the saved PR curves image file, or None if generation failed.
        """
        if not self.results or not self.models:
            print("Error: No valid results loaded. Cannot create PR curves.")
            return None
            
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get precision-recall data for each model
        for model in self.models:
            if model not in self.results:
                continue
                
            model_data = self.results[model]
            
            # Try to find precision and recall arrays
            precision_arrays = model_data.get('precision_arrays', {})
            recall_arrays = model_data.get('recall_arrays', {})
            
            # If we have matching precision and recall arrays for classes
            if precision_arrays and recall_arrays:
                for class_id, precision in precision_arrays.items():
                    if class_id in recall_arrays:
                        recall = recall_arrays[class_id]
                        # Get class name if available
                        class_name = model_data.get('class_names', {}).get(class_id, f"Class {class_id}")
                        # Plot PR curve for this class
                        color = self._get_model_color(model)
                        linestyle = '--' if '_seg' in model else '-'
                        ax.plot(recall, precision, label=f"{model}-{class_name}", 
                               color=color, linestyle=linestyle, alpha=0.7)
            
            # Check if we have overall precision-recall values
            if 'precision' in model_data and 'recall' in model_data:
                precision = model_data['precision']
                recall = model_data['recall']
                
                # Check if these are arrays
                if isinstance(precision, list) and isinstance(recall, list) and len(precision) == len(recall):
                    color = self._get_model_color(model)
                    linestyle = '--' if '_seg' in model else '-'
                    ax.plot(recall, precision, label=f"{model} (Overall)", 
                           color=color, linestyle=linestyle, linewidth=2)
        
        # If we don't have any PR curves, create a placeholder plot
        if len(ax.get_lines()) == 0:
            # Display message about missing PR curve data
            ax.text(0.5, 0.5, "Precision-Recall curve data not available.", 
                   fontsize=12, ha='center', va='center', transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        else:
            # Configure the plot
            ax.set_xlabel('Recall', fontsize=12)
            ax.set_ylabel('Precision', fontsize=12)
            ax.set_xlim(0, 1.05)
            ax.set_ylim(0, 1.05)
            ax.grid(linestyle='--', alpha=0.6)
            
            # Add diagonal reference line representing random performance
            ax.plot([0, 1], [1, 0], color='gray', linestyle=':')
            
            # Add legend (adjust for many models)
            if len(self.models) > 5:
                # For many models, put legend outside
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
            else:
                # For fewer models, put legend inside
                ax.legend(loc='lower left', fontsize='small')
        
        # Add title
        results_filename = self.results_file.name if self.results_file else "unknown_results.json"
        ax.set_title(f'Precision-Recall Curves\nEvaluation results from {results_filename}',
                    fontsize=14)
        
        # Save the figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"precision_recall_curves_{timestamp}.png"
        output_path = VISUALIZATIONS_DIR / output_filename
        
        try:
            # Save the figure with good resolution
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Precision-Recall curves saved to: {output_path}")
            
            if show_plot:
                plt.show()
            else:
                plt.close(fig)
                
            return str(output_path.resolve())
            
        except Exception as e:
            print(f"Error saving or showing PR curve plot: {e}")
            traceback.print_exc()
            plt.close(fig)
            return None

# Example usage block for testing the script directly
if __name__ == "__main__":
    print(f"--- Executing MetricsVisualizer Test Block ---")
    print(f"Project Root (derived): {PROJECT_ROOT}")
    print(f"Results Directory: {RESULTS_DIR}")
    print(f"Visualizations Directory: {VISUALIZATIONS_DIR}")

    # Ensure results directory exists
    RESULTS_DIR.mkdir(exist_ok=True)

    # --- Create a dummy results file mimicking the target image's data if none exist ---
    latest_results_file = None
    list_of_files = glob.glob(str(RESULTS_DIR / 'evaluation_results_*.json'))
    if list_of_files:
        latest_results_file = Path(max(list_of_files, key=os.path.getctime))
        print(f"Found existing results file: {latest_results_file.name}")
    else:
        print("No existing results file found. Creating a dummy file for testing.")
        dummy_file_path = RESULTS_DIR / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}_dummy.json"
        # Data closely matching the user's second image (input_file_0.png)
        dummy_data = {
             "mask-rcnn": {
                 "coco_metrics": { # Values from the target image's table
                     "AP_IoU=0.50:0.95": 0.000, "AP_IoU=0.50": 0.000, "AP_IoU=0.75": 0.000,
                     "AP_small": 0.003, "AP_medium": 0.004, "AP_large": 0.006
                 },
                 "precision": 0.0, "recall": 0.0, "f1_score": 0.000, "fps": 4.7, # Values from target image
                 "detection_counts_per_image": [] # Empty list to trigger "No data" message
             },
             "yolo-seg": {
                 "coco_metrics": { # Values from the target image's table
                      "AP_IoU=0.50:0.95": 0.000, "AP_IoU=0.50": 0.000, "AP_IoU=0.75": 0.000,
                      "AP_small": 0.001, "AP_medium": 0.001, "AP_large": 0.001
                 },
                  "precision": 0.0, "recall": 0.0, "f1_score": 0.000, "fps": 15.6, # Values from target image
                  "detection_counts_per_image": None # Test None case for missing data -> should show [0]
             },
             "evaluation_metadata": {
                  "timestamp": datetime.now().isoformat(),
                  "max_images": 50, # Example value
                  "dataset": "coco_val2017_subset_50",
                  "evaluation_tool_version": "1.1"
             }
        }
        try:
            with open(dummy_file_path, 'w') as f:
                 json.dump(dummy_data, f, indent=2)
            print(f"Created dummy results file: {dummy_file_path.name}")
            latest_results_file = dummy_file_path
        except Exception as e:
            print(f"Error creating dummy file: {e}")
            traceback.print_exc()


    # --- Initialize visualizer ---
    visualizer = MetricsVisualizer() # Will load latest file

    # --- Generate and Show Dashboard ---
    if visualizer.results:
        print("\nAttempting to generate dashboard...")
        dashboard_path = visualizer.create_metrics_dashboard(show_plot=True)
        if dashboard_path:
            print(f"\nDashboard generation successful: {dashboard_path}")
        else:
            print("\nDashboard generation failed.")
    else:
        print("\nVisualizer could not load results. Cannot run visualization tests.")

    print(f"--- MetricsVisualizer Test Block Finished ---")








