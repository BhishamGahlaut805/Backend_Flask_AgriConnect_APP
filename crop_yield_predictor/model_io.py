"""
Model I/O - Save and load LSTM models from MongoDB GridFS
"""

import torch
import io
from datetime import datetime
from typing import Optional, Dict
from sklearn.preprocessing import MinMaxScaler

from mongo_storage import mongo_storage
from logging_config import logger


def save_lstm_model(
    model,
    scalers: Dict,
    farm_id: str,
    crop_name: str,
    config: dict
) -> str:
    """
    Save LSTM model directly into MongoDB GridFS
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_type = f"lstm_yield_{farm_id}_{crop_name}"

        # Create in-memory buffer
        buffer = io.BytesIO()

        # Get feature columns from model if available
        feature_columns = getattr(model, 'feature_columns', [])

        # Save checkpoint into memory
        checkpoint = {
            "model_state": model.state_dict(),
            "scaler_state": {k: v.__dict__ for k, v in scalers.items()},
            "config": config,
            "feature_columns": feature_columns,
            "timestamp": timestamp
        }

        torch.save(checkpoint, buffer)
        buffer.seek(0)

        # Store model in MongoDB GridFS
        file_id = mongo_storage.save_model(
            model_bytes=buffer.getvalue(),
            model_type=model_type,
            metadata={
                "farm_id": farm_id,
                "crop_name": crop_name,
                "model_type": "LSTM",
                "version": timestamp,
                "feature_count": len(feature_columns)
            }
        )

        # Save metadata separately for easy querying
        metadata = {
            "farm_id": farm_id,
            "crop": crop_name,
            "model_type": "LSTM",
            "gridfs_file_id": file_id,
            "input_features": feature_columns,
            "model_size": sum(p.numel() for p in model.parameters()),
            "created_at": datetime.utcnow(),
            "version": timestamp
        }

        mongo_storage.save_yield_model_metadata(metadata)

        logger.info(
            f"Successfully saved LSTM model to MongoDB GridFS | "
            f"Farm: {farm_id} | Crop: {crop_name} | ID: {file_id}"
        )

        return str(file_id)

    except Exception as e:
        logger.error(f"Failed to save model: {str(e)}")
        raise


def load_latest_lstm_model(
    farm_id: str,
    crop_name: str
) -> Optional[Dict]:
    """
    Load latest LSTM model from MongoDB GridFS
    """
    try:
        model_type = f"lstm_yield_{farm_id}_{crop_name}"

        # Load model bytes from GridFS
        model_bytes = mongo_storage.load_model(model_type)

        if not model_bytes:
            logger.warning(f"No model found for farm={farm_id}, crop={crop_name}")
            return None

        # Convert bytes to memory buffer
        buffer = io.BytesIO(model_bytes)

        # Set weights_only=False for pickle compatibility with scalers
        checkpoint = torch.load(buffer, map_location=torch.device("cpu"), weights_only=False)

        # Rebuild scalers
        scalers = {}
        for col, state in checkpoint.get("scaler_state", {}).items():
            scaler = MinMaxScaler()
            scaler.__dict__.update(state)
            scalers[col] = scaler

        logger.info(
            f"Successfully loaded model from MongoDB GridFS | "
            f"Farm: {farm_id} | Crop: {crop_name}"
        )

        return {
            "model_state": checkpoint.get("model_state"),
            "scalers": scalers,
            "config": checkpoint.get("config", {}),
            "feature_columns": checkpoint.get("feature_columns", []),
            "timestamp": checkpoint.get("timestamp", "unknown")
        }

    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return None


def delete_lstm_model(farm_id: str, crop_name: str) -> bool:
    """
    Delete LSTM model for a farm-crop combination
    """
    try:
        model_type = f"lstm_yield_{farm_id}_{crop_name}"

        # Delete from models collection
        model_doc = mongo_storage.models.find_one({"model_type": model_type})
        if model_doc and model_doc.get("gridfs_id"):
            from bson import ObjectId
            mongo_storage.fs.delete(ObjectId(model_doc["gridfs_id"]))

        # Delete metadata
        mongo_storage.models.delete_many({"model_type": model_type})
        mongo_storage.yield_models.delete_many({
            "farm_id": farm_id,
            "crop": crop_name
        })

        logger.info(f"Deleted model for farm={farm_id}, crop={crop_name}")
        return True

    except Exception as e:
        logger.error(f"Failed to delete model: {str(e)}")
        return False


def list_models_for_farm(farm_id: str) -> list:
    """
    List all models for a farm
    """
    try:
        models = list(mongo_storage.yield_models.find(
            {"farm_id": farm_id},
            {"_id": 0, "crop": 1, "created_at": 1, "version": 1}
        ).sort("created_at", -1))

        return models

    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}")
        return []
    