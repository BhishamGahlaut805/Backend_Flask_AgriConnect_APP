"""
LSTM Outbreak Predictor
Trains LSTM models for disease outbreak prediction using data from MongoDB
Stores models and scalers in MongoDB GridFS instead of local files
"""

import os
import io
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import joblib
import tempfile

from mongo_storage import mongo_storage
from logging_config import logger


class LSTMOutbreakPredictor:
    """LSTM predictor for disease outbreak with MongoDB storage"""

    def __init__(self):
        self.client = mongo_storage.client
        self.db = mongo_storage.db
        self.farm_col = mongo_storage.farms

        # Use MongoDB for run logs instead of local file
        self.run_log_collection = self.db["lstm_run_logs"]
        self._init_log_collection()

    def _init_log_collection(self):
        """Initialize run log collection with indexes"""
        self.run_log_collection.create_index([("farm_id", 1), ("date", 1)], unique=True)
        self.run_log_collection.create_index([("last_run", -1)])

    def _load_log(self) -> Dict[str, str]:
        """Load run logs from MongoDB"""
        try:
            logs = {}
            cursor = self.run_log_collection.find({})
            for log in cursor:
                logs[log["farm_id"]] = log["date"]
            return logs
        except Exception as e:
            logger.error(f"Failed to load run logs: {e}")
            return {}

    def _save_log(self, farm_id: str, date: str):
        """Save run log to MongoDB"""
        try:
            self.run_log_collection.update_one(
                {"farm_id": farm_id},
                {"$set": {"date": date, "last_run": datetime.utcnow()}},
                upsert=True
            )
        except Exception as e:
            logger.error(f"Failed to save run log for {farm_id}: {e}")

    def already_ran_today(self, farm_id: str) -> bool:
        """Check if already ran today for this farm"""
        try:
            today = datetime.utcnow().strftime('%Y-%m-%d')
            log = self.run_log_collection.find_one({"farm_id": str(farm_id), "date": today})
            return log is not None
        except Exception:
            return False

    def mark_today_done(self, farm_id: str):
        """Mark that prediction ran today"""
        today = datetime.utcnow().strftime('%Y-%m-%d')
        self._save_log(str(farm_id), today)

    def force_rerun_for_farm(self, farm_id: str, confirm: bool = False):
        """Force rerun for a farm (for debugging)"""
        if not confirm:
            logger.warning("Force rerun blocked. Set confirm=True to proceed.")
            return

        try:
            self.run_log_collection.delete_many({"farm_id": str(farm_id)})
            logger.info(f"Force rerun enabled for {farm_id}")
        except Exception as e:
            logger.error(f"Failed to force rerun for {farm_id}: {e}")

    def get_training_data_from_mongodb(self, farm_id: str) -> Optional[pd.DataFrame]:
        """Retrieve training data from MongoDB"""
        try:
            # Try to load from training_data collection first
            training_data = mongo_storage.load_training_data(
                "outbreak_training",
                farm_id=farm_id
            )

            if training_data is not None and isinstance(training_data, pd.DataFrame):
                return training_data.sort_values("date")

            # Fallback: Try to load from farm's training_csv_path if stored in GridFS
            farm_doc = self.farm_col.find_one({"farm_id": farm_id})
            if farm_doc and farm_doc.get("training_data_id"):
                training_data = mongo_storage.load_training_data(
                    "outbreak_training",
                    farm_id=farm_id
                )
                if training_data is not None:
                    return training_data.sort_values("date")

            # Alternative: Build from disease reports
            return self._build_training_data_from_reports(farm_id)

        except Exception as e:
            logger.error(f"Failed to load training data for {farm_id}: {e}")
            return None

    def _build_training_data_from_reports(self, farm_id: str) -> Optional[pd.DataFrame]:
        """Build training data from disease reports if no CSV available"""
        try:
            # Get farm details
            farm = self.farm_col.find_one({"farm_id": farm_id})
            if not farm:
                logger.warning(f"Farm {farm_id} not found")
                return None

            # Get disease reports for this farm
            reports = list(mongo_storage.disease_reports.find(
                {"farm_id": farm_id},
                sort=[("timestamp", 1)]
            ))

            if len(reports) < 10:
                logger.warning(f"Not enough reports for farm {farm_id}: {len(reports)}")
                return None

            # Convert reports to training format
            data = []
            for report in reports:
                row = {
                    "date": report.get("timestamp", datetime.utcnow()).strftime("%Y-%m-%d"),
                    "latitude": farm.get("latitude", 0),
                    "longitude": farm.get("longitude", 0),
                    "soil_temp_0cm": report.get("soil_temp", 0),
                    "soil_moisture_1_3cm": report.get("soil_moisture", 0),
                    "evapotranspiration": report.get("evapotranspiration", 0),
                    "cloud_cover_low": report.get("cloud_cover", 0),
                    "wind_gusts_10m": report.get("wind_gust", 0),
                    "temp": report.get("temperature", 0),
                    "humidity": report.get("humidity", 0),
                    "rain_1h": report.get("rainfall", 0),
                    "risk%": 1 if report.get("disease", "healthy") != "healthy" else 0,
                    "radius_km": report.get("distance_km", 0),
                    "weather_desc": report.get("weather_desc", "")
                }
                data.append(row)

            df = pd.DataFrame(data)

            # Save this built data back to MongoDB for future use
            mongo_storage.save_training_data(
                df,
                "outbreak_training",
                farm_id=farm_id,
                metadata={"source": "disease_reports", "records": len(df)}
            )

            return df

        except Exception as e:
            logger.error(f"Failed to build training data from reports: {e}")
            return None

    def _save_model_to_mongodb(self, model, scaler_X: StandardScaler, scaler_y: StandardScaler,
                               farm_id: str, feature_cols: List[str]) -> str:
        """Save LSTM model and scalers to MongoDB GridFS"""
        try:
            import tempfile

            # Create temporary directory for model files
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save Keras model
                model_path = os.path.join(temp_dir, "lstm_model.keras")
                model.save(model_path)

                # Save scalers
                scaler_X_path = os.path.join(temp_dir, "scaler_X.pkl")
                scaler_y_path = os.path.join(temp_dir, "scaler_y.pkl")
                joblib.dump(scaler_X, scaler_X_path)
                joblib.dump(scaler_y, scaler_y_path)

                # Save feature columns
                features_path = os.path.join(temp_dir, "feature_cols.pkl")
                joblib.dump(feature_cols, features_path)

                # Read and store each file in GridFS
                file_ids = {}

                with open(model_path, "rb") as f:
                    file_ids["model"] = mongo_storage.save_model(
                        f.read(),
                        f"lstm_outbreak_model_{farm_id}",
                        {"farm_id": farm_id, "type": "keras_model"}
                    )

                with open(scaler_X_path, "rb") as f:
                    file_ids["scaler_X"] = mongo_storage.save_model(
                        f.read(),
                        f"lstm_outbreak_scaler_X_{farm_id}",
                        {"farm_id": farm_id, "type": "scaler_X"}
                    )

                with open(scaler_y_path, "rb") as f:
                    file_ids["scaler_y"] = mongo_storage.save_model(
                        f.read(),
                        f"lstm_outbreak_scaler_y_{farm_id}",
                        {"farm_id": farm_id, "type": "scaler_y"}
                    )

                with open(features_path, "rb") as f:
                    file_ids["features"] = mongo_storage.save_model(
                        f.read(),
                        f"lstm_outbreak_features_{farm_id}",
                        {"farm_id": farm_id, "type": "feature_cols"}
                    )

                # Store metadata in farm document
                self.farm_col.update_one(
                    {"farm_id": farm_id},
                    {"$set": {
                        "lstm_model_ids": file_ids,
                        "lstm_model_updated": datetime.utcnow()
                    }}
                )

                logger.info(f"Model saved to MongoDB for farm {farm_id}")
                return file_ids.get("model", "")

        except Exception as e:
            logger.error(f"Failed to save model to MongoDB: {e}")
            raise

    def _load_model_from_mongodb(self, farm_id: str) -> Optional[Tuple[Sequential, StandardScaler, StandardScaler, List[str]]]:
        """Load LSTM model and scalers from MongoDB GridFS"""
        try:
            import tempfile

            farm_doc = self.farm_col.find_one({"farm_id": farm_id})
            if not farm_doc or "lstm_model_ids" not in farm_doc:
                logger.info(f"No stored model found for farm {farm_id}")
                return None

            file_ids = farm_doc["lstm_model_ids"]

            with tempfile.TemporaryDirectory() as temp_dir:
                model_path = os.path.join(temp_dir, "lstm_model.keras")
                scaler_X_path = os.path.join(temp_dir, "scaler_X.pkl")
                scaler_y_path = os.path.join(temp_dir, "scaler_y.pkl")
                features_path = os.path.join(temp_dir, "feature_cols.pkl")

                # Load model
                model_bytes = mongo_storage.load_model(f"lstm_outbreak_model_{farm_id}")
                if model_bytes:
                    with open(model_path, "wb") as f:
                        f.write(model_bytes)

                # Load scaler_X
                scaler_X_bytes = mongo_storage.load_model(f"lstm_outbreak_scaler_X_{farm_id}")
                if scaler_X_bytes:
                    with open(scaler_X_path, "wb") as f:
                        f.write(scaler_X_bytes)

                # Load scaler_y
                scaler_y_bytes = mongo_storage.load_model(f"lstm_outbreak_scaler_y_{farm_id}")
                if scaler_y_bytes:
                    with open(scaler_y_path, "wb") as f:
                        f.write(scaler_y_bytes)

                # Load features
                features_bytes = mongo_storage.load_model(f"lstm_outbreak_features_{farm_id}")
                if features_bytes:
                    with open(features_path, "wb") as f:
                        f.write(features_bytes)

                # Load the actual objects
                model = load_model(model_path)
                scaler_X = joblib.load(scaler_X_path)
                scaler_y = joblib.load(scaler_y_path)
                feature_cols = joblib.load(features_path)

                return model, scaler_X, scaler_y, feature_cols

        except Exception as e:
            logger.error(f"Failed to load model from MongoDB: {e}")
            return None

    def create_sequences(self, X: np.ndarray, y: np.ndarray, seq_len: int = 10):
        """Create sequences for LSTM training"""
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_len):
            X_seq.append(X[i:i+seq_len])
            y_seq.append(y[i+seq_len])
        return np.array(X_seq), np.array(y_seq)

    def train_and_predict(self, farm_doc: Dict) -> Optional[List[Dict]]:
        """Train LSTM model and generate predictions"""
        farm_id = farm_doc.get("farm_id")
        farm_name = farm_doc.get("farm_name", "UnknownFarm")

        if not farm_id:
            logger.warning("No farm_id in document")
            return None

        # Check if already ran today
        if self.already_ran_today(farm_id):
            logger.info(f"Already trained today: {farm_name}")
            return None

        # Get training data from MongoDB
        df = self.get_training_data_from_mongodb(farm_id)

        if df is None or len(df) < 10:
            logger.warning(f"Not enough data for {farm_name}: {len(df) if df is not None else 0} records")
            return None

        # Prepare data
        df = df.sort_values("date")

        # Drop non-numeric columns
        exclude_cols = ["date", "weather_desc", "_id", "farm_id"]
        numeric_cols = [col for col in df.columns if col not in exclude_cols]
        df = df[numeric_cols]

        # Drop rows with NaN values
        df = df.dropna()

        if len(df) < 10:
            logger.warning(f"Not enough valid data after cleaning for {farm_name}: {len(df)} records")
            return None

        target_cols = ["risk%", "radius_km"]

        # Ensure target columns exist
        for col in target_cols:
            if col not in df.columns:
                df[col] = 0

        feature_cols = [col for col in df.columns if col not in target_cols]

        if len(feature_cols) == 0:
            logger.warning(f"No feature columns for {farm_name}")
            return None

        X_data = df[feature_cols].values
        y_data = df[target_cols].values

        # Scale the data
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_scaled = scaler_X.fit_transform(X_data)
        y_scaled = scaler_y.fit_transform(y_data)

        # Create sequences
        seq_len = min(10, len(X_scaled) // 3)
        X_seq, y_seq = self.create_sequences(X_scaled, y_scaled, seq_len)

        if len(X_seq) < 5:
            logger.warning(f"Not enough sequences for {farm_name}: {len(X_seq)} sequences")
            return None

        # Split for training (use all data for training since we have limited data)
        X_train, y_train = X_seq, y_seq

        # Build model
        model = Sequential()
        model.add(LSTM(64, activation='relu', return_sequences=True,
                      input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(2))  # 2 outputs: risk% and radius_km

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # Early stopping
        early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

        # Train model
        model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=min(4, len(X_train)),
            verbose=0,
            callbacks=[early_stop]
        )

        # Generate future predictions
        recent_seq = X_scaled[-seq_len:].reshape(1, seq_len, len(feature_cols))
        future_preds_scaled = []

        for _ in range(10):  # Predict next 10 days
            pred = model.predict(recent_seq, verbose=0)[0]
            future_preds_scaled.append(pred)
            # Shift sequence
            new_row = X_scaled[-1:].reshape(1, 1, len(feature_cols))
            recent_seq = np.append(recent_seq[:, 1:, :], new_row, axis=1)

        future_preds = scaler_y.inverse_transform(future_preds_scaled)

        # Normalize risk values between 5% and 95%
        risk_vals = future_preds[:, 0]

        # Clip extreme values
        risk_vals = np.clip(risk_vals, 0, 100)

        # Apply sigmoid-like scaling for better distribution
        if np.std(risk_vals) > 0:
            # Normalize to 0-1 range
            norm_risk = (risk_vals - np.min(risk_vals)) / (np.max(risk_vals) - np.min(risk_vals))
            # Scale to 5-95% range
            risk_scaled = 5 + norm_risk * 90
        else:
            risk_scaled = np.full_like(risk_vals, 50.0)

        risk_scaled = np.clip(np.round(risk_scaled, 2), 5, 95)
        future_preds[:, 0] = risk_scaled

        # Generate results
        today = datetime.utcnow()
        result = [{
            "date": (today + timedelta(days=i+1)).strftime('%Y-%m-%d'),
            "predicted_risk%": round(float(p[0]), 2),
            "predicted_radius_Km": max(0, round(float(p[1]), 2))
        } for i, p in enumerate(future_preds)]

        # Save model to MongoDB
        self._save_model_to_mongodb(model, scaler_X, scaler_y, farm_id, feature_cols)

        # Update farm document with predictions
        self.farm_col.update_one(
            {"farm_id": farm_id},
            {"$set": {
                "lstm_prediction": result,
                "lstm_last_updated": datetime.utcnow(),
                "lstm_feature_cols": feature_cols,
                "lstm_seq_len": seq_len
            }}
        )

        # Mark as done for today
        self.mark_today_done(farm_id)

        logger.info(f"Successfully trained and stored predictions for {farm_name}")
        return result

    def update_predictions_only(self, farm_doc: Dict) -> Optional[List[Dict]]:
        """Generate predictions using existing model without retraining"""
        farm_id = farm_doc.get("farm_id")
        farm_name = farm_doc.get("farm_name", "UnknownFarm")

        # Load existing model
        model_data = self._load_model_from_mongodb(farm_id)
        if not model_data:
            logger.info(f"No existing model for {farm_name}, training new model")
            return self.train_and_predict(farm_doc)

        model, scaler_X, scaler_y, feature_cols = model_data

        # Get latest data for prediction
        df = self.get_training_data_from_mongodb(farm_id)
        if df is None or len(df) < 10:
            logger.warning(f"Not enough data for prediction for {farm_name}")
            return None

        # Prepare latest data
        df = df.sort_values("date")
        numeric_cols = [col for col in df.columns if col not in ["date", "weather_desc", "_id", "farm_id"]]
        df = df[numeric_cols].dropna()

        # Get only the feature columns used in training
        available_features = [col for col in feature_cols if col in df.columns]
        if len(available_features) != len(feature_cols):
            logger.warning(f"Missing features for {farm_name}. Expected {len(feature_cols)}, got {len(available_features)}")
            return self.train_and_predict(farm_doc)

        X_data = df[feature_cols].values
        X_scaled = scaler_X.transform(X_data)

        # Generate predictions
        seq_len = farm_doc.get("lstm_seq_len", 10)
        seq_len = min(seq_len, len(X_scaled))

        if len(X_scaled) < seq_len:
            logger.warning(f"Not enough data for sequence for {farm_name}")
            return None

        recent_seq = X_scaled[-seq_len:].reshape(1, seq_len, len(feature_cols))
        future_preds_scaled = []

        for _ in range(10):
            pred = model.predict(recent_seq, verbose=0)[0]
            future_preds_scaled.append(pred)
            new_row = X_scaled[-1:].reshape(1, 1, len(feature_cols))
            recent_seq = np.append(recent_seq[:, 1:, :], new_row, axis=1)

        future_preds = scaler_y.inverse_transform(future_preds_scaled)

        # Normalize risk values
        risk_vals = future_preds[:, 0]
        risk_vals = np.clip(risk_vals, 0, 100)

        if np.std(risk_vals) > 0:
            norm_risk = (risk_vals - np.min(risk_vals)) / (np.max(risk_vals) - np.min(risk_vals))
            risk_scaled = 5 + norm_risk * 90
        else:
            risk_scaled = np.full_like(risk_vals, 50.0)

        risk_scaled = np.clip(np.round(risk_scaled, 2), 5, 95)
        future_preds[:, 0] = risk_scaled

        # Generate results
        today = datetime.utcnow()
        result = [{
            "date": (today + timedelta(days=i+1)).strftime('%Y-%m-%d'),
            "predicted_risk%": round(float(p[0]), 2),
            "predicted_radius_Km": max(0, round(float(p[1]), 2))
        } for i, p in enumerate(future_preds)]

        # Update farm document
        self.farm_col.update_one(
            {"farm_id": farm_id},
            {"$set": {
                "lstm_prediction": result,
                "lstm_last_updated": datetime.utcnow()
            }}
        )

        logger.info(f"Predictions updated for {farm_name}")
        return result

    def run_for_all_farms(self, force_retrain: bool = False):
        """Run prediction for all farms in the database"""
        # Get farms that have data
        farms = list(self.farm_col.find({}))

        results = {
            "total": len(farms),
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "details": []
        }

        for farm in farms:
            farm_name = farm.get("farm_name", "Unknown")

            try:
                # Check if we have data for this farm
                df = self.get_training_data_from_mongodb(farm.get("farm_id"))

                if df is None or len(df) < 10:
                    results["skipped"] += 1
                    results["details"].append({
                        "farm_id": farm.get("farm_id"),
                        "farm_name": farm_name,
                        "status": "skipped",
                        "reason": "Insufficient data"
                    })
                    continue

                # Run prediction (train or update)
                if force_retrain:
                    prediction = self.train_and_predict(farm)
                else:
                    prediction = self.update_predictions_only(farm)

                if prediction:
                    results["successful"] += 1
                    results["details"].append({
                        "farm_id": farm.get("farm_id"),
                        "farm_name": farm_name,
                        "status": "success",
                        "predictions": len(prediction)
                    })
                else:
                    results["failed"] += 1
                    results["details"].append({
                        "farm_id": farm.get("farm_id"),
                        "farm_name": farm_name,
                        "status": "failed",
                        "reason": "Prediction returned None"
                    })

            except Exception as e:
                results["failed"] += 1
                results["details"].append({
                    "farm_id": farm.get("farm_id"),
                    "farm_name": farm_name,
                    "status": "failed",
                    "error": str(e)
                })
                logger.error(f"Failed for {farm_name}: {e}")

        # Store results in MongoDB
        self.db["lstm_batch_results"].insert_one({
            "timestamp": datetime.utcnow(),
            "force_retrain": force_retrain,
            "results": results
        })

        logger.info(f"LSTM batch completed: {results['successful']} successful, "
                   f"{results['failed']} failed, {results['skipped']} skipped")

        return results

    def get_predictions_for_farm(self, farm_id: str) -> Optional[List[Dict]]:
        """Get latest predictions for a specific farm"""
        farm = self.farm_col.find_one({"farm_id": farm_id})
        if farm:
            return farm.get("lstm_prediction")
        return None

    def get_prediction_summary(self, farm_id: str) -> Optional[Dict]:
        """Get summary of predictions for a farm"""
        predictions = self.get_predictions_for_farm(farm_id)
        if not predictions:
            return None

        risks = [p["predicted_risk%"] for p in predictions]

        return {
            "farm_id": farm_id,
            "max_risk": max(risks),
            "min_risk": min(risks),
            "avg_risk": sum(risks) / len(risks),
            "high_risk_days": sum(1 for r in risks if r > 50),
            "predictions": predictions
        }


# Singleton instance
lstm_outbreak_predictor = LSTMOutbreakPredictor()
