"""
Configuration management for the WARND model.
"""

from typing import Any, Dict, Optional
from pathlib import Path
import json


class Config:
    """
    Configuration manager for DMD-based depression prediction.
    
    Stores model hyperparameters, data paths, and training settings.
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration.
        
        Args:
            config_dict: Dictionary of configuration parameters
        """
        # Default configuration
        self.config = {
            # Model parameters
            'dmd_rank': None,  # None for full rank
            'classifier_type': 'gradient_boosting',  # 'random_forest' or 'gradient_boosting'
            'random_state': 42,
            
            # Data parameters
            'stage1_features': 20,
            'stage2_features': 30,
            'stage2_timesteps': 354,
            'stage3_features': 15,
            'stage3_timepoints': 8,
            
            # Training parameters
            'test_size': 0.2,
            'validation_size': 0.2,
            'n_folds': 5,
            
            # Feature extraction
            'use_temporal_features': True,
            'use_frequency_features': True,
            'use_complexity_features': True,
            
            # Paths
            'data_dir': './data',
            'output_dir': './output',
            'model_dir': './models',
            
            # Evaluation
            'threshold': 0.5,
            'optimize_threshold': True,
        }
        
        # Update with provided configuration
        if config_dict:
            self.config.update(config_dict)
    
    def __getitem__(self, key: str) -> Any:
        """Get configuration value."""
        return self.config[key]
    
    def __setitem__(self, key: str, value: Any):
        """Set configuration value."""
        self.config[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default."""
        return self.config.get(key, default)
    
    def update(self, config_dict: Dict[str, Any]):
        """Update configuration."""
        self.config.update(config_dict)
    
    def save(self, path: str):
        """
        Save configuration to JSON file.
        
        Args:
            path: Path to save configuration
        """
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'Config':
        """
        Load configuration from JSON file.
        
        Args:
            path: Path to configuration file
            
        Returns:
            Config object
        """
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self.config.copy()
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Config({self.config})"
