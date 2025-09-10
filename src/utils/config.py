"""
Configuration management for earthquake prediction system.
"""
import os
import platform
from pathlib import Path
from typing import Dict, Any

class Config:
    """Configuration class for earthquake prediction system."""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw" 
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    RESULTS_DIR = PROJECT_ROOT / "results"
    
    # USGS API Configuration
    USGS_API_ENDPOINT = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    DEFAULT_MIN_MAGNITUDE = 4.0
    DEFAULT_DAYS_BACK = 30
    
    # Model Configuration
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 5
    
    # Feature Engineering
    TEMPORAL_FEATURES = [
        'hour', 'day_of_week', 'month', 'quarter',
        'days_since_last_event', 'events_in_last_7_days',
        'events_in_last_30_days'
    ]
    
    SPATIAL_FEATURES = [
        'depth', 'latitude', 'longitude', 
        'distance_to_nearest_fault', 'regional_cluster'
    ]
    
    # Model hyperparameters
    MODEL_PARAMS = {
        'random_forest': {
            'n_estimators': 100,
            'max_depth': 20,
            'min_samples_split': 5,
            'random_state': RANDOM_STATE
        },
        'xgboost': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'random_state': RANDOM_STATE
        },
        'neural_network': {
            'hidden_layers': [128, 64, 32],
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001
        },
        'lstm': {
            'sequence_length': 10,
            'lstm_units': 50,
            'epochs': 100,
            'batch_size': 32
        }
    }
    
    # Ensemble configuration
    ENSEMBLE_WEIGHTS = {
        'random_forest': 0.3,
        'xgboost': 0.3,
        'neural_network': 0.2,
        'lstm': 0.2
    }
    
    # Evaluation metrics
    METRICS = ['mse', 'mae', 'r2', 'rmse']
    
    # Alert thresholds
    ALERT_THRESHOLDS = {
        'high_magnitude': 6.0,
        'high_confidence': 0.8,
        'cluster_density': 5  # events per day in region
    }
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist."""
        dirs = [
            cls.DATA_DIR, cls.RAW_DATA_DIR, 
            cls.PROCESSED_DATA_DIR, cls.RESULTS_DIR
        ]
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_model_config(cls, model_name: str) -> Dict[str, Any]:
        """Get configuration for specific model."""
        return cls.MODEL_PARAMS.get(model_name, {})
    
    @staticmethod
    def get_safe_n_jobs() -> int:
        """
        Get a safe n_jobs value that works across platforms.
        
        Returns:
            Safe n_jobs value for scikit-learn models
        """
        # On Windows, avoid using all cores to prevent _posixsubprocess error
        if platform.system() == 'Windows':
            # Use half the available cores, but at least 1
            import multiprocessing
            return max(1, multiprocessing.cpu_count() // 2)
        else:
            # On Unix-like systems, -1 is usually safe
            return -1
    
    @staticmethod
    def configure_joblib_backend():
        """
        Configure joblib backend for Windows compatibility and resource management.
        """
        try:
            import joblib
            # Set memory limit to prevent resource leaks
            joblib.parallel.JOBLIB_MULTIPROCESSING_MAX_MEMORY = "2G"
            
            if platform.system() == 'Windows':
                # Use threading backend on Windows to avoid multiprocessing issues
                joblib.parallel.DEFAULT_BACKEND = 'threading'
            else:
                # Configure proper resource cleanup for Unix systems
                import multiprocessing as mp
                mp.set_start_method('spawn', force=False)  # Safer for resource management
        except ImportError:
            # joblib not available, skip configuration
            pass
        except RuntimeError:
            # start_method already set, skip
            pass