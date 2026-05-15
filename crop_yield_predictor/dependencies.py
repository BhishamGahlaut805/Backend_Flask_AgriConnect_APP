"""
Dependencies and configuration for crop yield predictor
All configuration now loaded from centralized config
"""

import logging
from datetime import datetime
from typing import Dict, Any
from config import Config


def load_config() -> Dict[str, Any]:
    """
    Load configuration from centralized Config
    Returns configuration dictionary with defaults
    """
    try:
        # Get training config from central Config
        config = {
            'training': Config.TRAINING_CONFIG,
            'model': {
                'hidden_size': 128,
                'num_layers': 2,
                'dropout': 0.2,
                'bidirectional': True
            },
            'data': {
                'required_columns': [
                    'farm_id', 'crop', 'season', 'year', 'window_num',
                    'start_date', 'end_date', 'is_season_end', 'yield'
                ],
                'feature_columns': Config.FEATURE_COLUMNS,
                'weather_columns': [
                    'avg_temperature_2m_mean',
                    'avg_precipitation_sum',
                    'relative_humidity_2m_mean',
                    'shortwave_radiation_sum'
                ],
                'soil_columns': [
                    'soil_pH',
                    'organic_matter_content'
                ],
                'management_columns': [
                    'plant_population_density',
                    'irrigation_type'
                ]
            }
        }

        return config

    except Exception as e:
        logging.warning(f"Using fallback config due to error: {str(e)}")
        return {
            'training': {
                'batch_size': 32,
                'epochs': 50,
                'lr': 0.001,
                'patience': 15,
                'min_delta': 0.0001,
                'aux_weight': 0.3
            },
            'model': {
                'hidden_size': 64,
                'num_layers': 2,
                'dropout': 0.2,
                'bidirectional': True
            },
            'data': {
                'required_columns': [
                    'farm_id', 'crop', 'season', 'year', 'window_num',
                    'start_date', 'end_date', 'is_season_end', 'yield'
                ],
                'feature_columns': [
                    'avg_temperature_2m_mean',
                    'avg_precipitation_sum',
                    'soil_pH',
                    'organic_matter_content',
                    'plant_population_density'
                ]
            }
        }


def setup_logging():
    """Setup logging for the module"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


logger = setup_logging()
