"""
Unit tests for error handling utilities.
"""
import pytest
import logging
import sys
from pathlib import Path

from src.error_handling import (
    log_info, log_warning, log_error, log_exception,
    handle_exceptions, with_error_handling, robust_function
)

@pytest.mark.unit
class TestErrorHandling:
    """Tests for error handling functions"""
    
    def test_logging(self, caplog):
        """Test logging functions"""
        # Set log level to INFO to capture all logs
        caplog.set_level(logging.INFO)
        
        # Test info logging
        log_info("Test info message")
        assert "Test info message" in caplog.text
        
        # Test warning logging
        log_warning("Test warning message")
        assert "Test warning message" in caplog.text
        
        # Test error logging
        log_error("Test error message")
        assert "Test error message" in caplog.text
        
        # Test exception logging
        try:
            raise ValueError("Test exception")
        except ValueError as e:
            log_exception(e, "Test context")
        
        assert "Test context" in caplog.text
        assert "Test exception" in caplog.text
    
    def test_handle_exceptions_decorator(self):
        """Test the handle_exceptions decorator"""
        # Define a function that will raise an exception
        @handle_exceptions
        def failing_function():
            raise ValueError("Test error")
        
        # Test that the exception is caught and re-raised
        with pytest.raises(ValueError) as excinfo:
            failing_function()
        
        # The decorator logs context but re-raises the original exception unchanged.
        assert "Test error" in str(excinfo.value)
    
    def test_with_error_handling_decorator(self):
        """Test the with_error_handling decorator"""
        # Define a function with custom error message
        @with_error_handling("Custom message")
        def failing_function():
            raise ValueError("Test error")
        
        # Test that the exception is caught and re-raised
        with pytest.raises(ValueError) as excinfo:
            failing_function()
        
        # The decorator logs the custom message but re-raises the original error.
        assert "Test error" in str(excinfo.value)
    
    def test_robust_function_decorator(self):
        """Test the robust_function decorator"""
        # Define a function that returns a fallback value on error
        @robust_function(fallback_value="Fallback")
        def failing_function():
            raise ValueError("Test error")
        
        # Test that the function returns the fallback value
        assert failing_function() == "Fallback"
        
        # Define a function that returns a value on success
        @robust_function(fallback_value="Fallback")
        def successful_function():
            return "Success"
        
        # Test that the function returns the success value
        assert successful_function() == "Success"
