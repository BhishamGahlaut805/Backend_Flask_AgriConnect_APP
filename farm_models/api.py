"""
Crop Recommendation API
Recommends crops based on weather and farm data
Trains models on-demand using state-specific data from CSV
Models stored in MongoDB, training data not stored
"""

from flask import Blueprint, request, jsonify
from datetime import datetime
from mongo_storage import mongo_storage
from logging_config import logger
from .model_crop_recommendation import trainer
from .prepare_data import WeatherDataProcessor
#importing cors
from flask_cors import CORS

# Blueprint
CropRecommendationBp = Blueprint('Crop_api', __name__)
CORS(CropRecommendationBp)

@CropRecommendationBp.route("/recommend-crops", methods=["POST","OPTIONS"])
def recommend_crops():
    """
    Recommend crops based on farm data
    If model doesn't exist for the state, trains a new one using CSV data
    """
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    data = request.get_json(silent=True)

    try:
        # Input validation
        required_fields = ["state", "season", "lat", "lon"]
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({"error": f"Missing required fields: {missing}"}), 400

        state = data["state"].strip()
        season = data["season"].strip()
        year = data.get("year", datetime.now().year)
        area = float(data.get("area", 1.0))
        yield_val = float(data.get("yield", 2000.0))
        lat = float(data["lat"])
        lon = float(data["lon"])
        farm_id = data.get("farm_id", "farm_001")
        farm_name = data.get("farm_name", "DefaultFarm")

        logger.info(f"Recommendation request: state={state}, season={season}, farm={farm_name}")

        # Step 1: Get or train model for this state and season
        try:
            model_data = trainer.get_or_train_model(state, season)
        except ValueError as e:
            return jsonify({
                "error": str(e),
                "available_states": trainer.list_available_states()
            }), 400

        if not model_data:
            return jsonify({
                "error": f"Could not create model for state {state}. No training data available.",
                "available_states": trainer.list_available_states()
            }), 404

        # Step 2: Get weather data for the farm
        weather_processor = WeatherDataProcessor(lat, lon, farm_id, farm_name)
        weather_df = weather_processor.fetch_and_process_weather_data()

        if weather_df.empty:
            logger.warning(f"No weather data for {farm_name}, using defaults")
            weather_features = weather_processor._get_default_weather()
        else:
            weather_features = weather_processor.get_weather_summary(weather_df)

        # Step 3: Encode categorical variables using model's encoders
        try:
            state_enc = model_data["le_state"].transform([state])[0]
            season_enc = model_data["le_season"].transform([season])[0]
        except ValueError as e:
            return jsonify({
                "error": f"Invalid state or season: {str(e)}",
                "available_states": list(model_data["le_state"].classes_),
                "available_seasons": list(model_data["le_season"].classes_)
            }), 400

        # Step 4: Build feature vector for prediction
        features = [[
            state_enc,
            season_enc,
            year,
            area,
            yield_val,
            weather_features.get("avg_temperature_2m_mean", 25.0),
            weather_features.get("avg_temperature_2m_max", 30.0),
            weather_features.get("avg_temperature_2m_min", 20.0),
            weather_features.get("avg_relative_humidity_2m_mean", 60.0),
            weather_features.get("avg_wind_speed_10m_max", 10.0),
            weather_features.get("avg_precipitation_sum", 50.0),
            weather_features.get("avg_shortwave_radiation_sum", 200.0),
            weather_features.get("avg_surface_pressure_mean", 1013.0),
            weather_features.get("avg_cloud_cover_mean", 40.0)
        ]]

        # Step 5: Predict
        model = model_data["model"]
        le_crop = model_data["le_crop"]

        probs = model.predict_proba(features)[0]
        top_indices = probs.argsort()[-5:][::-1]

        top_crops = [
            {"crop": le_crop.inverse_transform([i])[0], "probability": float(probs[i])}
            for i in top_indices
        ]

        # Step 6: Log the prediction (optional)
        mongo_storage.save_prediction(
            farm_id=farm_id,
            prediction_type="crop_recommendation",
            result={
                "state": state,
                "season": season,
                "year": year,
                "recommendations": top_crops,
                "model_accuracy": model_data.get("metadata", {}).get("accuracy", 0)
            }
        )

        response = {
            "status": "success",
            "farm": {"id": farm_id, "name": farm_name, "lat": lat, "lon": lon},
            "state": state,
            "season": season,
            "year": year,
            "top_crops": top_crops,
            "model_info": {
                "accuracy": model_data.get("metadata", {}).get("accuracy", 0),
                "training_samples": model_data.get("metadata", {}).get("samples", 0)
            },
            "weather_used": {
                "temperature": weather_features.get("avg_temperature_2m_mean", 0),
                "humidity": weather_features.get("avg_relative_humidity_2m_mean", 0),
                "precipitation": weather_features.get("avg_precipitation_sum", 0),
                "date": weather_features.get("date", "latest")
            },
            "timestamp": datetime.utcnow().isoformat()
        }

        logger.info(f"Recommendation successful for {state}/{season}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Recommendation error: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@CropRecommendationBp.route("/train", methods=["POST"])
def train_model():
    """
    Explicitly train a model for a specific state and season
    """
    data = request.get_json()

    try:
        state = data.get("state")
        season = data.get("season", "all")

        if not state:
            return jsonify({"error": "state is required"}), 400

        logger.info(f"Explicit training request for state={state}, season={season}")

        # Train model
        training_result = trainer.train_model_for_state_season(state, season)

        # Save to MongoDB
        trainer.save_model_to_mongodb(state, season, training_result)

        return jsonify({
            "status": "success",
            "message": f"Model trained and saved for {state}/{season}",
            "accuracy": training_result["accuracy"],
            "samples": training_result["samples"],
            "features": training_result["features"],
            "crop_classes": training_result["crop_classes"][:10],  # Show first 10
            "timestamp": datetime.utcnow().isoformat()
        })

    except Exception as e:
        logger.error(f"Training error: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@CropRecommendationBp.route("/model-info", methods=["GET"])
def get_model_info():
    """Get information about models for a state"""
    state = request.args.get("state")
    season = request.args.get("season", "all")

    try:
        if state:
            query = {
                "model_type": trainer.MODEL_TYPE_PREFIX,
                "state": state,
                "active": True
            }
            if season and season != "all":
                query["season"] = season
        else:
            query = {"model_type": trainer.MODEL_TYPE_PREFIX, "active": True}

        models = list(mongo_storage.models.find(query))

        # Clean up for response
        result = []
        for model in models:
            result.append({
                "state": model.get("state"),
                "season": model.get("season"),
                "accuracy": model.get("accuracy", 0),
                "samples": model.get("samples", 0),
                "saved_at": model.get("saved_at").isoformat() if model.get("saved_at") else None
            })

        return jsonify({
            "status": "success",
            "models": result,
            "count": len(result),
            "available_states": trainer.list_available_states(),
            "timestamp": datetime.utcnow().isoformat()
        })

    except Exception as e:
        logger.error(f"Model info error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@CropRecommendationBp.route("/available-states", methods=["GET"])
def get_available_states():
    """Get list of states available for training"""
    try:
        states = trainer.list_available_states()
        return jsonify({
            "status": "success",
            "states": states,
            "count": len(states),
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        logger.error(f"Available states error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@CropRecommendationBp.route("/weather", methods=["POST"])
def get_weather():
    """Get weather data for a farm"""
    data = request.get_json()

    try:
        lat = float(data.get("lat"))
        lon = float(data.get("lon"))
        farm_id = data.get("farm_id", "unknown")
        farm_name = data.get("farm_name", "Unknown Farm")

        if not lat or not lon:
            return jsonify({"error": "lat and lon are required"}), 400

        processor = WeatherDataProcessor(lat, lon, farm_id, farm_name)
        df = processor.fetch_and_process_weather_data()

        if df.empty:
            return jsonify({"error": "Could not fetch weather data"}), 500

        summary = processor.get_weather_summary(df)

        return jsonify({
            "status": "success",
            "weather": summary,
            "farm": {"id": farm_id, "name": farm_name},
            "timestamp": datetime.utcnow().isoformat()
        })

    except Exception as e:
        logger.error(f"Weather error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@CropRecommendationBp.route("/delete-model", methods=["DELETE"])
def delete_model():
    """Delete a model for a specific state and season"""
    data = request.get_json()
    state = data.get("state")
    season = data.get("season")

    if not state or not season:
        return jsonify({"error": "state and season are required"}), 400

    try:
        model_id = trainer.get_model_id(state, season)

        # Deactivate model in metadata
        mongo_storage.models.update_many(
            {
                "model_type": trainer.MODEL_TYPE_PREFIX,
                "state": state,
                "season": season
            },
            {"$set": {"active": False}}
        )

        # Note: GridFS files are kept for audit, just deactivated

        return jsonify({
            "status": "success",
            "message": f"Model for {state}/{season} deactivated",
            "timestamp": datetime.utcnow().isoformat()
        })

    except Exception as e:
        logger.error(f"Delete model error: {str(e)}")
        return jsonify({"error": str(e)}), 500
