"""
Configuration management for fraud detection system.
"""
import os
from typing import Dict, Any
import yaml

class Config:
    """Configuration manager for fraud detection system."""
    
    def __init__(self, config_path: str = None):
        """Initialize configuration."""
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), '../../config.yaml')
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            'data': {
                'input_file': 'synthetic_dirty_transaction_logs.csv',
                'validation_split': 0.2,
                'random_seed': 42
            },
            'parsing': {
                'date_formats': [
                    '%Y-%m-%d %H:%M:%S',
                    '%d/%m/%Y %H:%M:%S',
                    '%Y-%m-%d %H:%M:%S::%s::%s::%f::%s::%s'
                ],
                'amount_pattern': r'[£$€]?(\d+\.?\d*)',
                'user_pattern': r'user\d+',
                'location_pattern': r'(London|Glasgow|Birmingham|Liverpool|Cardiff|Leeds|Manchester|None)',
                'device_pattern': r'(iPhone 13|Samsung Galaxy S10|Pixel 6|Nokia 3310|Xiaomi Mi 11|Huawei P30|None)'
            },
            'features': {
                'numerical_features': ['amount', 'hour', 'day_of_week', 'is_weekend'],
                'categorical_features': ['transaction_type', 'location', 'device', 'currency'],
                'user_features': ['avg_amount', 'transaction_frequency', 'unique_locations', 'unique_devices'],
                'temporal_window_hours': 24,
                'location_risk_threshold': 0.1,
                'amount_zscore_threshold': 2.0
            },
            'models': {
                'isolation_forest': {
                    'contamination': 0.1,
                    'n_estimators': 100,
                    'random_state': 42
                },
                'dbscan': {
                    'eps': 0.5,
                    'min_samples': 5
                },
                'autoencoder': {
                    'hidden_dims': [64, 32, 16, 32, 64],
                    'learning_rate': 0.001,
                    'epochs': 100,
                    'batch_size': 64
                },
                'ensemble': {
                    'rule_weight': 0.3,
                    'isolation_weight': 0.25,
                    'autoencoder_weight': 0.25,
                    'clustering_weight': 0.2
                }
            },
            'evaluation': {
                'top_n_anomalies': 100,
                'anomaly_threshold': 0.7,
                'manual_validation_sample': 50
            },
            'visualization': {
                'figure_size': (12, 8),
                'color_palette': 'viridis',
                'anomaly_color': 'red',
                'normal_color': 'blue'
            }
        }
        
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
        
        return default_config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key path (e.g., 'models.isolation_forest.contamination')."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def update(self, key: str, value: Any) -> None:
        """Update configuration value."""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: str = None) -> None:
        """Save configuration to file."""
        save_path = path or self.config_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False)
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get full configuration."""
        return self._config

# Global configuration instance
config = Config()
