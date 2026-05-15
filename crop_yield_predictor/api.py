"""
Crop Yield Prediction API
Endpoints for yield prediction, model training, and scenario simulation
All data stored in MongoDB
"""

from flask import Blueprint, jsonify, request
from datetime import datetime
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any

from .load_crop_data import CropDataLoader
from .lstm_trainer import LSTMTrainer
from .lstm_predictor import LSTMPredictor
from .feature_engineer import FeatureEngineer
from mongo_storage import mongo_storage
from logging_config import logger

# Initialize API components
api_blueprint = Blueprint('agri_api', __name__, url_prefix='/api/v1')

# Enhanced Configuration
CONFIG = {
    'training': {
        'batch_size': 32,
        'epochs': 5,
        'lr': 0.001,
        'patience': 15,
        'min_delta': 0.0001,
        'aux_weight': 0.3
    },
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
        'feature_columns': [
            'avg_temperature_2m_mean',
            'avg_precipitation_sum',
            'soil_pH',
            'organic_matter_content',
            'plant_population_density'
        ]
    },
    'min_samples': 10  # Minimum number of complete seasons required for training
}


def validate_input_data(data: Dict, required_fields: List[str]) -> Optional[Dict]:
    """Validate input data and return error response if invalid"""
    missing = [field for field in required_fields if field not in data]
    if missing:
        return {
            'error': 'Missing required fields',
            'required': required_fields,
            'missing': missing,
            'received': list(data.keys()),
            'timestamp': datetime.now().isoformat()
        }
    return None


@api_blueprint.route('/predict', methods=['POST'])
def predict_yield():
    """Endpoint for yield prediction using existing model"""
    try:
        data = request.json
        validation = validate_input_data(data, ['farm_id', 'crop'])
        if validation:
            return jsonify(validation), 400

        farm_id = data['farm_id']
        crop = data['crop']

        logger.info(f"Predicting yield for farm {farm_id}, crop {crop}")

        # Load and prepare data
        data_loader = CropDataLoader(farm_id)
        df = data_loader.load_crop_data(crop)

        if df is None or df.empty:
            return jsonify({
                'error': 'No data found for this farm and crop',
                'farm_id': farm_id,
                'crop': crop,
                'timestamp': datetime.now().isoformat()
            }), 404

        # Initialize predictor
        predictor = LSTMPredictor(CONFIG)
        model_data = predictor.load_model(farm_id, crop)

        if not model_data:
            return jsonify({
                'error': 'Model not found',
                'solution': 'Train a model first using /train endpoint',
                'farm_id': farm_id,
                'crop': crop,
                'timestamp': datetime.now().isoformat()
            }), 404

        # Make prediction
        result = predictor.predict(model_data, df)
        if 'error' in result:
            return jsonify(result), 500

        # Prepare prediction document
        prediction_doc = {
            'farm_id': farm_id,
            'crop': crop,
            'predicted_yield': result['predicted_yield'],
            'attention_weights': result['attention_weights'],
            'feature_contributions': result.get('feature_contributions', {}),
            'input_stats': result.get('input_stats', {}),
            'model_version': model_data.get('timestamp', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'created_at': datetime.utcnow()
        }

        # Save to MongoDB using unified storage
        mongo_storage.save_yield_prediction(prediction_doc)

        # Prepare response
        response = {
            'status': 'success',
            'prediction': result['predicted_yield'],
            'attention_weights': result['attention_weights'],
            'feature_contributions': result.get('feature_contributions', {}),
            'timestamp': prediction_doc['timestamp']
        }

        logger.info(f"Prediction successful for farm {farm_id}, crop {crop}: {result['predicted_yield']}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Prediction failed',
            'details': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@api_blueprint.route('/train', methods=['POST'])
def train_model():
    """Endpoint for training a new model"""
    try:
        data = request.json
        validation = validate_input_data(data, ['farm_id', 'crop'])
        if validation:
            return jsonify(validation), 400

        farm_id = data['farm_id']
        crop = data['crop']

        logger.info(f"Training model for farm {farm_id}, crop {crop}")

        # Load and prepare data
        data_loader = CropDataLoader(farm_id)
        df = data_loader.load_crop_data(crop)

        if df is None or df.empty:
            return jsonify({
                'error': 'No data found for this farm and crop',
                'farm_id': farm_id,
                'crop': crop,
                'timestamp': datetime.now().isoformat()
            }), 404

        # Feature engineering
        fe = FeatureEngineer(CONFIG)
        df, preprocess_data = fe.preprocess(df)

        # Prepare sequences
        sequences, targets, metadata = data_loader.get_seasonal_data(df)

        # Validate training data
        if len(sequences) < CONFIG['min_samples']:
            return jsonify({
                'error': 'Insufficient training data',
                'samples': len(sequences),
                'minimum_required': CONFIG['min_samples'],
                'timestamp': datetime.now().isoformat()
            }), 400

        # Train model
        trainer = LSTMTrainer(CONFIG)
        model_path = trainer.train(
            sequences,
            targets,
            farm_id,
            crop
        )

        # Calculate yield statistics
        avg_yield = float(np.mean(targets))
        min_yield = float(np.min(targets))
        max_yield = float(np.max(targets))

        # Prepare metadata document
        metadata_doc = {
            'farm_id': farm_id,
            'crop': crop,
            'model_path': model_path,
            'features_used': list(preprocess_data.get('importance', {}).keys()),
            'feature_importance': preprocess_data.get('importance', {}),
            'training_samples': len(sequences),
            'created_at': datetime.utcnow(),
            'average_yield': avg_yield,
            'yield_range': {
                'min': min_yield,
                'max': max_yield
            },
            'config': CONFIG
        }

        # Save to MongoDB
        mongo_storage.save_yield_model_metadata(metadata_doc)

        # Also store training data for future reference
        mongo_storage.save_training_data(
            df,
            f"yield_training_{crop}",
            farm_id=farm_id,
            crop=crop,
            metadata={
                'samples': len(sequences),
                'features': metadata_doc['features_used']
            }
        )

        response = {
            'status': 'success',
            'farm_id': farm_id,
            'crop': crop,
            'samples': len(sequences),
            'features': metadata_doc['features_used'],
            'average_yield': avg_yield,
            'model_path': model_path,
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"Training successful for farm {farm_id}, crop {crop}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Training error: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Training failed',
            'details': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@api_blueprint.route('/simulate', methods=['POST'])
def simulate_scenarios():
    """Endpoint for what-if scenario analysis"""
    try:
        data = request.json
        validation = validate_input_data(data, ['farm_id', 'crop', 'scenarios'])
        if validation:
            return jsonify(validation), 400

        farm_id = data['farm_id']
        crop = data['crop']
        scenarios = data['scenarios']

        logger.info(f"Running simulations for farm {farm_id}, crop {crop}")

        # Load data and model
        data_loader = CropDataLoader(farm_id)
        df = data_loader.load_crop_data(crop)

        if df is None or df.empty:
            return jsonify({
                'error': 'No data found for this farm and crop',
                'timestamp': datetime.now().isoformat()
            }), 404

        predictor = LSTMPredictor(CONFIG)
        model_data = predictor.load_model(farm_id, crop)

        if not model_data:
            return jsonify({
                'error': 'Model not found',
                'solution': 'Train a model first using /train endpoint',
                'timestamp': datetime.now().isoformat()
            }), 404

        # Run scenarios
        results = predictor.what_if(model_data, df, scenarios)
        if 'error' in results:
            return jsonify(results), 500

        # Prepare simulation document
        simulation_doc = {
            'farm_id': farm_id,
            'crop': crop,
            'baseline': {
                'prediction': results['baseline']['predicted_yield'],
                'attention_weights': results['baseline']['attention_weights']
            },
            'scenarios': [],
            'model_version': model_data.get('timestamp', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'created_at': datetime.utcnow()
        }

        # Add scenario results
        for name, result in results.items():
            if name == 'baseline':
                continue
            simulation_doc['scenarios'].append({
                'name': name,
                'prediction': result.get('prediction', 0),
                'difference': result.get('difference', 0),
                'attention_changes': result.get('attention_changes', {})
            })

        # Save to MongoDB
        mongo_storage.save_simulation(simulation_doc)

        return jsonify({
            'status': 'success',
            'results': results,
            'timestamp': simulation_doc['timestamp']
        })

    except Exception as e:
        logger.error(f"Simulation error: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Simulation failed',
            'details': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@api_blueprint.route('/model-info/<farm_id>/<crop>', methods=['GET'])
def get_model_info(farm_id: str, crop: str):
    """Endpoint to get model information"""
    try:
        metadata = mongo_storage.get_yield_model_metadata(farm_id, crop)
        if not metadata:
            return jsonify({
                'error': 'Model metadata not found',
                'farm_id': farm_id,
                'crop': crop,
                'timestamp': datetime.now().isoformat()
            }), 404

        # Remove MongoDB _id field for clean response
        metadata.pop('_id', None)

        return jsonify({
            'status': 'success',
            'model_info': metadata,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Model info error: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Failed to retrieve model info',
            'details': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@api_blueprint.route('/predictions/<farm_id>/<crop>', methods=['GET'])
def get_prediction_history(farm_id: str, crop: str):
    """Get prediction history for a farm-crop combination"""
    try:
        predictions = mongo_storage.yield_predictions.find(
            {'farm_id': farm_id, 'crop': crop}
        ).sort('timestamp', -1).limit(10)

        result = []
        for pred in predictions:
            pred['_id'] = str(pred['_id'])
            result.append(pred)

        return jsonify({
            'status': 'success',
            'predictions': result,
            'count': len(result),
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error fetching prediction history: {e}")
        return jsonify({
            'error': 'Failed to fetch predictions',
            'details': str(e)
        }), 500


@api_blueprint.route('/models', methods=['GET'])
def list_models():
    """List all trained models"""
    try:
        models = list(mongo_storage.yield_models.find({}).sort('created_at', -1))

        result = []
        for model in models:
            result.append({
                'farm_id': model.get('farm_id'),
                'crop': model.get('crop'),
                'created_at': model.get('created_at').isoformat() if model.get('created_at') else None,
                'training_samples': model.get('training_samples', 0),
                'average_yield': model.get('average_yield', 0)
            })

        return jsonify({
            'status': 'success',
            'models': result,
            'count': len(result),
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return jsonify({
            'error': 'Failed to list models',
            'details': str(e)
        }), 500
        