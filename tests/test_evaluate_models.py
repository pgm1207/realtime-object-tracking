"""
Unit tests for the evaluate_models module.
Tests evaluation metrics, data loading, and result formatting.
"""

import pytest
import os
import json
import numpy as np
import torch
from pathlib import Path

# Remove specific function import
from src.evaluate_models import ModelEvaluator

@pytest.mark.unit
class TestModelEvaluation:
    """Tests for model evaluation functions"""
    
    def test_coco_dataset_directory_exists(self):
        """Test that the COCO dataset directory exists"""
        coco_dir = Path("data_sets/image_data/coco/val2017")
        if not coco_dir.exists():
            pytest.skip("COCO validation dataset not available")
        
        assert coco_dir.exists(), "COCO validation directory should exist"
    
    def test_evaluation_output_structure(self):
        """Test the structure of evaluation output files"""
        # Find any existing evaluation results
        results_dir = Path("inference/results")
        if not results_dir.exists():
            pytest.skip("No evaluation results directory found")
        
        # Look for any .json result files
        result_files = list(results_dir.glob("evaluation_results_*.json"))
        if not result_files:
            pytest.skip("No evaluation result files found")
        
        # Load the most recent result file
        result_file = sorted(result_files)[-1]
        with open(result_file, "r") as f:
            results = json.load(f)
        
        # Check the structure of the results (accept both historical shapes).
        assert isinstance(results, dict), "Results should be a dictionary"
        if "models" in results:
            assert isinstance(results["models"], list), "Models should be a list"
            if results["models"]:
                model_result = results["models"][0]
                assert "model_name" in model_result, "Model result should contain 'model_name'"
                assert "metrics" in model_result, "Model result should contain 'metrics'"
                assert "avg_inference_time" in model_result, "Model result should contain 'avg_inference_time'"
        else:
            # Mapping of model_type -> metrics.
            assert results, "Results should not be empty"
            for model_type, metrics in results.items():
                assert isinstance(metrics, dict), f"{model_type} metrics should be a dict"
