"""
Crop Recommendation Model Training
Trains RandomForest model for crop recommendation using state-specific data
Models stored in MongoDB, training data not stored
"""

import pandas as pd
import joblib
import io
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from datetime import datetime
from mongo_storage import mongo_storage
from logging_config import logger


class CropRecommendationTrainer:
    """Train and save state-specific crop recommendation models to MongoDB"""

    MODEL_TYPE_PREFIX = "crop_recommendation"

    def __init__(self, csv_path: str = None):
        """
        Initialize trainer with optional CSV path
        Default CSV path is relative to the module
        """
        if csv_path is None:
            # Get the directory where this file is located
            current_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(current_dir, "data_training", "training_data_crop_recommendation.csv")

        self.csv_path = csv_path
        self.master_df = None
        self._load_master_data()

    def _load_master_data(self):
        """Load the master training data from CSV file"""
        try:
            if os.path.exists(self.csv_path):
                self.master_df = pd.read_csv(self.csv_path)
                logger.info(f"Loaded master training data from {self.csv_path}: {len(self.master_df)} records")
            else:
                logger.error(f"Master training data not found at {self.csv_path}")
                self.master_df = pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to load master training data: {e}")
            self.master_df = pd.DataFrame()

    def get_model_id(self, state: str, season: str) -> str:
        """Generate unique model ID for state and season"""
        return f"{self.MODEL_TYPE_PREFIX}_{state.lower().replace(' ', '_')}_{season.lower()}"

    def filter_data_by_state(self, state: str) -> pd.DataFrame:
        """Filter master data by state"""
        if self.master_df.empty:
            self._load_master_data()

        if self.master_df.empty:
            return pd.DataFrame()

        # Filter by state (case-insensitive)
        filtered_df = self.master_df[self.master_df["State"].str.lower() == state.lower()]
        logger.info(f"Filtered data for state {state}: {len(filtered_df)} records")
        return filtered_df

    def train_model_for_state_season(self, state: str, season: str) -> dict:
        """
        Train model for specific state and season using filtered data
        Returns training results
        """
        # Filter data by state
        df = self.filter_data_by_state(state)

        if df.empty:
            raise ValueError(f"No training data found for state: {state}")

        # Filter by season if needed
        if season and season.lower() != "all":
            df = df[df["Season"].str.lower() == season.lower()]

        if df.empty:
            raise ValueError(f"No training data found for state {state} and season {season}")

        logger.info(f"Training model for state={state}, season={season} with {len(df)} records")

        # Encode categorical features
        le_crop = LabelEncoder()
        le_state = LabelEncoder()
        le_season = LabelEncoder()

        # Fit encoders on the filtered data
        df["Crop_encoded"] = le_crop.fit_transform(df["Crop"])
        df["State_encoded"] = le_state.fit_transform(df["State"])
        df["Season_encoded"] = le_season.fit_transform(df["Season"])

        # Select features
        features = [
            "State_encoded", "Season_encoded", "year", "Area", "Yield",
            "avg_temperature_2m_mean", "avg_temperature_2m_max", "avg_temperature_2m_min",
            "avg_relative_humidity_2m_mean", "avg_wind_speed_10m_max",
            "avg_precipitation_sum", "avg_shortwave_radiation_sum",
            "avg_surface_pressure_mean", "avg_cloud_cover_mean"
        ]

        # Ensure all features exist
        available_features = [f for f in features if f in df.columns]
        missing_features = [f for f in features if f not in df.columns]

        if missing_features:
            logger.warning(f"Missing features for {state}: {missing_features}")

        X = df[available_features]
        y = df["Crop_encoded"]

        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=300, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = model.score(X_test, y_test)

        logger.info(f"Model accuracy for {state}/{season}: {accuracy:.4f}")

        return {
            "model": model,
            "le_crop": le_crop,
            "le_state": le_state,
            "le_season": le_season,
            "accuracy": accuracy,
            "samples": len(df),
            "features": available_features,
            "crop_classes": list(le_crop.classes_),
            "state_classes": list(le_state.classes_),
            "season_classes": list(le_season.classes_)
        }

    def save_model_to_mongodb(self, state: str, season: str, training_result: dict) -> str:
        """
        Save trained model to MongoDB GridFS (only model, not training data)
        """
        try:
            model_id = self.get_model_id(state, season)

            # Save model
            model_buffer = io.BytesIO()
            joblib.dump(training_result["model"], model_buffer)
            model_buffer.seek(0)

            model_file_id = mongo_storage.save_model(
                model_buffer.getvalue(),
                f"{model_id}_model",
                metadata={
                    "type": "random_forest",
                    "state": state,
                    "season": season,
                    "accuracy": training_result["accuracy"]
                }
            )

            # Save encoders as a combined object
            encoders = {
                "le_crop": training_result["le_crop"],
                "le_state": training_result["le_state"],
                "le_season": training_result["le_season"]
            }

            encoders_buffer = io.BytesIO()
            joblib.dump(encoders, encoders_buffer)
            encoders_buffer.seek(0)

            encoders_file_id = mongo_storage.save_model(
                encoders_buffer.getvalue(),
                f"{model_id}_encoders",
                metadata={
                    "type": "label_encoders",
                    "state": state,
                    "season": season
                }
            )

            # Save metadata (no training data stored)
            metadata = {
                "model_type": self.MODEL_TYPE_PREFIX,
                "model_id": model_id,
                "state": state,
                "season": season,
                "model_gridfs_id": model_file_id,
                "encoders_gridfs_id": encoders_file_id,
                "accuracy": training_result["accuracy"],
                "samples": training_result["samples"],
                "features": training_result["features"],
                "crop_classes": training_result["crop_classes"],
                "state_classes": training_result["state_classes"],
                "season_classes": training_result["season_classes"],
                "saved_at": datetime.utcnow(),
                "active": True
            }

            # Deactivate previous version for this state/season
            mongo_storage.models.update_many(
                {
                    "model_type": self.MODEL_TYPE_PREFIX,
                    "state": state,
                    "season": season,
                    "active": True
                },
                {"$set": {"active": False}}
            )

            # Save metadata
            mongo_storage.models.insert_one(metadata)

            logger.info(f"Model saved to MongoDB for {state}/{season}. ID: {model_id}")
            return model_id

        except Exception as e:
            logger.error(f"Failed to save model to MongoDB: {e}")
            raise

    def load_model_from_mongodb(self, state: str, season: str) -> dict:
        """
        Load model and encoders from MongoDB for specific state and season
        Returns dict with model and encoders or None if not found
        """
        try:
            model_id = self.get_model_id(state, season)

            # Get metadata
            metadata = mongo_storage.models.find_one(
                {
                    "model_type": self.MODEL_TYPE_PREFIX,
                    "state": state,
                    "season": season,
                    "active": True
                }
            )

            if not metadata:
                logger.info(f"No active model found for {state}/{season}")
                return None

            # Load model
            model_bytes = mongo_storage.load_model(f"{model_id}_model")
            if not model_bytes:
                logger.warning(f"Model file not found for {state}/{season}")
                return None

            model_buffer = io.BytesIO(model_bytes)
            model = joblib.load(model_buffer)

            # Load encoders
            encoders_bytes = mongo_storage.load_model(f"{model_id}_encoders")
            if not encoders_bytes:
                logger.warning(f"Encoders file not found for {state}/{season}")
                return None

            encoders_buffer = io.BytesIO(encoders_bytes)
            encoders = joblib.load(encoders_buffer)

            logger.info(f"Model loaded from MongoDB for {state}/{season}")

            return {
                "model": model,
                "le_crop": encoders["le_crop"],
                "le_state": encoders["le_state"],
                "le_season": encoders["le_season"],
                "metadata": metadata
            }

        except Exception as e:
            logger.error(f"Failed to load model from MongoDB: {e}")
            return None

    def get_or_train_model(self, state: str, season: str) -> dict:
        """
        Get existing model from MongoDB or train a new one
        Returns model and encoders
        """
        # Try to load existing model
        model_data = self.load_model_from_mongodb(state, season)

        if model_data:
            logger.info(f"Using existing model for {state}/{season}")
            return model_data

        # Train new model
        logger.info(f"Training new model for {state}/{season}")
        training_result = self.train_model_for_state_season(state, season)
        self.save_model_to_mongodb(state, season, training_result)

        # Return the newly trained model
        return {
            "model": training_result["model"],
            "le_crop": training_result["le_crop"],
            "le_state": training_result["le_state"],
            "le_season": training_result["le_season"],
            "metadata": {
                "accuracy": training_result["accuracy"],
                "samples": training_result["samples"]
            }
        }

    def model_exists(self, state: str, season: str) -> bool:
        """Check if a model exists for given state and season"""
        count = mongo_storage.models.count_documents(
            {
                "model_type": self.MODEL_TYPE_PREFIX,
                "state": state,
                "season": season,
                "active": True
            }
        )
        return count > 0

    def list_available_states(self) -> list:
        """Get list of states available in master data"""
        if self.master_df.empty:
            self._load_master_data()

        if self.master_df.empty:
            return []

        return self.master_df["State"].unique().tolist()


# Global trainer instance
trainer = CropRecommendationTrainer()
