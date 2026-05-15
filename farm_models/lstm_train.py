'''
This file provides the code for training the LSTM model for predicting crop yields based on historical yield data and weather data. It reads the merged dataset, preprocesses it, and trains an LSTM model for each state-crop-season combination. The trained model is then used to predict the yield for the next year, and the results are stored in a MongoDB collection along with feature importance analysis for weather factors. The model and its performance metrics are also saved for future reference.
'''

import os
import json
import joblib
import logging
import numpy as np  #type: ignore
import pandas as pd
from datetime import datetime
from pymongo import MongoClient #type: ignore
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch #type: ignore
import torch.nn as nn #type: ignore
from torch.utils.data import DataLoader, TensorDataset #type: ignore

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB Service
class MongoService:
    def __init__(self):
        mongo_uri = os.getenv("MONGO_URI")
        self.client = MongoClient(mongo_uri)
        self.db = self.client["AgriSupportDB"]
        self.stateYield = self.db["StateYieldAll"]

    def save_prediction(self, data: dict):
        self.stateYield.update_one(
            {"state": data["state"], "crop": data["crop"], "season": data["season"], "version": data["version"]},
            {"$set": data},
            upsert=True
        )

# PyTorch LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=50, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Core Trainer
class YieldPredictor:
    def __init__(self, csv_path: str, model_dir="models"):
        self.df = pd.read_csv(csv_path)
        print("Available columns:", self.df.columns.tolist())
        self.mongo = MongoService()
        self.model_dir = r"C:\Users\bhish\OneDrive\Desktop\AgriSupport\ML\FarmModels\Models"
        os.makedirs(self.model_dir, exist_ok=True)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def preprocess(self):
        self.df = self.df[self.df["Season"] != "Total"]
        self.df["year"] = self.df["year"].astype(int)
        self.features = [
            'year', 'Area', 'avg_temperature_2m_mean', 'avg_temperature_2m_max',
            'avg_temperature_2m_min', 'avg_relative_humidity_2m_mean',
            'avg_wind_speed_10m_max', 'avg_precipitation_sum',
            'avg_shortwave_radiation_sum', 'avg_surface_pressure_mean',
            'avg_cloud_cover_mean'
        ]
        self.target = 'Yield'

    def create_sequences(self, data, n_steps=3):
        X, y = [], []
        for i in range(len(data) - n_steps):
            X.append(data[i:i + n_steps, :-1])
            y.append(data[i + n_steps, -1])
        return np.array(X), np.array(y)

    def train_for_group(self, group, crop, state, season):
        try:
            group = group.sort_values("year")
            data = group[self.features + [self.target]].values
            scaled_data = self.scaler.fit_transform(data)
            X, y = self.create_sequences(scaled_data)

            if len(X) < 1:
                logger.info(f"Skipping {state}-{crop}-{season}: not enough data for sequences")
                return None

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            # Convert to PyTorch tensors
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(self.device)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(self.device)

            # Loaders
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
            train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

            # Model, loss, optimizer
            input_size = X.shape[2]
            model = LSTMModel(input_size).to(self.device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            best_val_loss = float('inf')
            patience, counter = 10, 0

            for epoch in range(10):
                model.train()
                for xb, yb in train_loader:
                    optimizer.zero_grad()
                    output = model(xb)
                    loss = criterion(output, yb)
                    loss.backward()
                    optimizer.step()

                # Validation loss
                model.eval()
                with torch.no_grad():
                    val_losses = [criterion(model(xb), yb).item() for xb, yb in test_loader]
                val_loss = np.mean(val_losses)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict()
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        break

            # Load best model
            model.load_state_dict(best_model_state)
            model.eval()

            # Predict next year
            last_sequence = scaled_data[-3:, :-1]
            last_sequence_tensor = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                predicted_yield_scaled = model(last_sequence_tensor).cpu().numpy()

            dummy = np.zeros((1, scaled_data.shape[1]))
            dummy[0, -1] = predicted_yield_scaled[0, 0]
            predicted_yield = self.scaler.inverse_transform(dummy)[0, -1]

            # === Calculate Input Gradients for Feature Importance ===
            model.eval()
            last_sequence_tensor.requires_grad_(True)

            output = model(last_sequence_tensor)
            output.backward()

            # Get gradients of the output w.r.t. input features
            input_grads = last_sequence_tensor.grad.detach().cpu().numpy().squeeze(0)  # shape: (3, features)
            mean_grads = np.mean(np.abs(input_grads), axis=0)  # shape: (features,)

            # Only consider weather-related features
            weather_feature_names = {
                'avg_temperature_2m_mean': "Average Temperature (°C)",
                'avg_temperature_2m_max': "Max Temperature (°C)",
                'avg_temperature_2m_min': "Min Temperature (°C)",
                'avg_relative_humidity_2m_mean': "Relative Humidity (%)",
                'avg_wind_speed_10m_max': "Max Wind Speed (m/s)",
                'avg_precipitation_sum': "Precipitation (mm)",
                'avg_shortwave_radiation_sum': "Solar Radiation (W/m²)",
                'avg_surface_pressure_mean': "Surface Pressure (hPa)",
                'avg_cloud_cover_mean': "Cloud Cover (%)"
            }

            # Get indices of weather features
            feature_indices = [self.features.index(f) for f in weather_feature_names if f in self.features]

            # Map gradients to weather features
            weather_importance = {
                weather_feature_names[self.features[i]]: mean_grads[i] for i in feature_indices
            }

            # Normalize to percentages
            total = sum(weather_importance.values())
            weather_importance_percent = {
                k: round((v / total) * 100, 2)
                for k, v in sorted(weather_importance.items(), key=lambda item: item[1], reverse=True)
            }

            # Top 5 weather contributors
            top_weather_factors = dict(list(weather_importance_percent.items())[:5])


            next_year = group["year"].max() + 1
            record = {
                "state": state,
                "crop": crop,
                "season": season,
                "version": "1.0.0",
                "timestamp": datetime.utcnow(),
                "model_type": "LSTM",
                "features_used": self.features,
                "last_5_years": [],
                "prediction": {
                    "year": int(next_year),
                    "predicted_yield": round(float(predicted_yield), 2),
                    "confidence_interval": None
                },
                "model_performance": {
                    "train_samples": len(X_train),
                    "test_samples": len(X_test),
                    "last_loss": float(loss.item()),
                    "last_val_loss": float(val_loss)
                },
                "top_weather_factors": top_weather_factors
            }

            for _, row in group.tail(5).iterrows():
                record["last_5_years"].append({
                    "year": int(row["year"]),
                    "area_lakh_ha": round(row["Area"], 2),
                    "production_lakh_tonnes": round(row["Production"], 2),
                    "yield_kg_per_ha": round(row["Yield"], 2),
                    "weather_data": {
                        "avg_temperature": round(row["avg_temperature_2m_mean"], 2),
                        "avg_humidity": round(row["avg_relative_humidity_2m_mean"], 2),
                        "avg_precipitation": round(row["avg_precipitation_sum"], 2)
                    }
                })

            model_path = os.path.join(self.model_dir, f"{state}_{crop}_{season}_lstm.pt")
            torch.save(model.state_dict(), model_path)
            record["model_path"] = model_path

            return record

        except Exception as e:
            logger.error(f"Error training model for {state}-{crop}-{season}: {str(e)}")
            return None

    def train_and_store(self):
        grouped = self.df.groupby(["Crop", "State", "Season"])
        for (crop, state, season), group in grouped:
            if group.shape[0] < 5:
                logger.info(f"Skipping {state}-{crop}-{season}: not enough data")
                continue
            record = self.train_for_group(group, crop, state, season)
            if record:
                self.mongo.save_prediction(record)
                logger.info(f"Saved LSTM prediction for {state}-{crop}-{season}")

'''
# Run Script
if __name__ == "__main__":
    predictor = YieldPredictor(r"merged_lstm_dataset.csv")
    predictor.preprocess()
    predictor.train_and_store()
'''

