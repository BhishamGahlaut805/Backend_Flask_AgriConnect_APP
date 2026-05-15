import os
import time
import uuid
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO #type: ignore
from dotenv import load_dotenv
#importing bson
import bson

# Import modules
from config import Config
from mongo_storage import mongo_storage
from logging_config import logger

load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# CORS configuration
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")
CORS(app, supports_credentials=True, resources={
    r"/*": {"origins": ["http://localhost:5173", FRONTEND_URL]}
})

app.secret_key = Config.SECRET_KEY

# Socket.IO
socketio = SocketIO(app, cors_allowed_origins=["http://localhost:5173", FRONTEND_URL], async_mode="threading")

# ========== HEALTH CHECK ==========
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "mongodb": mongo_storage.client is not None
    })

# ========== IMPORT MODULES AFTER APP CREATION ==========
from api.agromonitoring import AgroMonitoringAPI
from api.openmeteo_api import WeatherDataProcessor
# from api.openhourly import OpenMeteoAPI
from api.soil_profile import SoilGridsFetcher
from weather_section.api import weather_bp

from crop_disease_outbreak.data_service.crop_reports import CropDiseasePredictor
from crop_disease_outbreak.lstm.lstm_outbreak import LSTMOutbreakPredictor

from crop_vs_weed.api import weed_bp, camera_manager, generate_frames

from crop_yield_predictor.api import api_blueprint

from data_service.prepare_data import DataService

from farm_models.api import CropRecommendationBp

from geospatial.haversine import HaversineDistance

from llm.api import Agribot_bp1

from market_price.market_prices import market_prices_bp

from mongodb_service.create_farm import CreateFarmService
from mongodb_service.disease_report_service import DiseaseReportService
from mongodb_service.update_service import SummaryUpdateService

# ========== REGISTER BLUEPRINTS ==========
app.register_blueprint(weed_bp)
app.register_blueprint(api_blueprint)
app.register_blueprint(market_prices_bp)
app.register_blueprint(weather_bp, url_prefix='/api/weather')
app.register_blueprint(CropRecommendationBp, url_prefix='/api/v1')
app.register_blueprint(Agribot_bp1, url_prefix="/api/agribot")

# ========== SOCKET EVENTS ==========
@socketio.on('connect', namespace='/weed')
def handle_connect():
    logger.info('Client connected to weed namespace')

@socketio.on('disconnect', namespace='/weed')
def handle_disconnect():
    logger.info('Client disconnected from weed namespace')

@socketio.on('start_stream', namespace='/weed')
def handle_start_stream(data):
    stream_type = data.get('type', 'webcam')
    logger.info(f'Starting {stream_type} stream')
    socketio.start_background_task(target=generate_frames, socketio=socketio)

@socketio.on('stop_stream', namespace='/weed')
def handle_stop_stream():
    camera_manager.set_streaming(False)
    logger.info('Stream stopped by client')

# ========== LAZY LOADING MODELS ==========
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        predictor = CropDiseasePredictor()
    return predictor

# ========== DISEASE PREDICTION API ==========
@app.route('/api/predictDisease', methods=['POST'])
def predict_disease():
    import tempfile
    import requests

    try:
        results = []
        diseased_count = 0
        total_count = 0

        model_type = (
            request.form.get('model_type', 'all').lower()
            if request.form
            else request.args.get('model_type', 'all').lower()
        )

        if model_type not in ['potato', 'cotton', 'all']:
            return jsonify({"error": "Invalid model_type"}), 400

        farm_id = request.form.get('farm_id', 'unknown_farm')
        farm_name = request.form.get('farm_name', 'Unknown Farm')

        predictor = get_predictor()

        # =========================================================
        # MULTIPART IMAGE UPLOAD
        # =========================================================
        if 'image' in request.files:

            images = request.files.getlist('image')

            lat_list = request.form.get('latitude', '')
            lon_list = request.form.get('longitude', '')

            latitudes = [
                float(x.strip())
                for x in lat_list.split(',')
                if x.strip()
            ]

            longitudes = [
                float(x.strip())
                for x in lon_list.split(',')
                if x.strip()
            ]

            for i, image in enumerate(images):

                temp_path = None

                try:
                    ext = os.path.splitext(image.filename)[-1]
                    unique_name = f"{uuid.uuid4().hex}{ext}"

                    image_bytes = image.read()

                    # ==========================================
                    # Upload to Cloudinary/GridFS
                    # ==========================================
                    upload_result = mongo_storage.save_upload(
                        image_bytes,
                        unique_name,
                        farm_id
                    )

                    # ==========================================
                    # Create temp file for ML prediction
                    # ==========================================
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=ext
                    ) as tmp_file:

                        tmp_file.write(image_bytes)
                        tmp_file.flush()

                        temp_path = tmp_file.name

                    # ==========================================
                    # Prediction
                    # ==========================================
                    total_count += 1

                    result = predictor.predict_crop_disease(
                        temp_path,
                        model_type
                    )

                    # ==========================================
                    # Image URL
                    # ==========================================
                    if upload_result.get("storage") == "cloudinary":
                        image_url = upload_result.get("cloudinary_url")
                    else:
                        image_url = f"/api/uploads/{upload_result['gridfs_id']}"

                    # ==========================================
                    # Metadata
                    # ==========================================
                    result["image"] = unique_name
                    result["image_url"] = image_url

                    result["latitude"] = (
                        latitudes[i]
                        if i < len(latitudes)
                        else None
                    )

                    result["longitude"] = (
                        longitudes[i]
                        if i < len(longitudes)
                        else None
                    )

                    # ==========================================
                    # Disease count
                    # ==========================================
                    if result["disease"].lower() != "healthy":
                        diseased_count += 1

                    # ==========================================
                    # Save report
                    # ==========================================
                    DiseaseReportService.save_report({
                        "farm_name": farm_name,
                        "farm_id": farm_id,
                        "latitude": result["latitude"],
                        "longitude": result["longitude"],
                        "crop": result["crop"],
                        "disease": result["disease"],
                        "confidence": result["confidence"],
                        "image_path": unique_name,
                        "image_storage": upload_result.get("storage", "unknown"),
                        "timestamp": datetime.utcnow()
                    })

                    results.append(result)

                except Exception as img_error:

                    logger.error(
                        f"Image processing failed: {img_error}",
                        exc_info=True
                    )

                    results.append({
                        "image": image.filename,
                        "error": str(img_error)
                    })

                finally:
                    # ==========================================
                    # Safe temp cleanup
                    # ==========================================
                    if temp_path and os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except Exception as cleanup_error:
                            logger.warning(
                                f"Temp cleanup failed: {cleanup_error}"
                            )

        # =========================================================
        # JSON IMAGE URL INPUT
        # =========================================================
        elif request.is_json:

            data = request.get_json(silent=True) or {}

            image_paths = data.get('images', [])
            coordinates = data.get('coordinates', [])

            farm_name = data.get('farm_name', 'Unknown Farm')
            farm_id = data.get('farm_id', 'unknown_farm')

            for i, image_path in enumerate(image_paths):

                temp_path = None

                try:
                    # ==========================================
                    # Remote image URL
                    # ==========================================
                    if image_path.startswith('http'):

                        response = requests.get(
                            image_path,
                            timeout=30
                        )

                        response.raise_for_status()

                        ext = os.path.splitext(image_path)[1]

                        if not ext:
                            ext = ".jpg"

                        with tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=ext
                        ) as tmp_file:

                            tmp_file.write(response.content)
                            tmp_file.flush()

                            temp_path = tmp_file.name

                    else:
                        temp_path = image_path

                    # ==========================================
                    # Validate file
                    # ==========================================
                    if not os.path.exists(temp_path):

                        results.append({
                            "image": image_path,
                            "error": "Image path not found"
                        })

                        continue

                    # ==========================================
                    # Prediction
                    # ==========================================
                    total_count += 1

                    result = predictor.predict_crop_disease(
                        temp_path,
                        model_type
                    )

                    filename = os.path.basename(image_path)

                    result["image"] = filename
                    result["image_url"] = image_path

                    coord = (
                        coordinates[i]
                        if i < len(coordinates)
                        else {}
                    )

                    result["latitude"] = coord.get("lat")
                    result["longitude"] = coord.get("lon")

                    if result["disease"].lower() != "healthy":
                        diseased_count += 1

                    DiseaseReportService.save_report({
                        "farm_name": farm_name,
                        "farm_id": farm_id,
                        "latitude": result["latitude"],
                        "longitude": result["longitude"],
                        "crop": result["crop"],
                        "disease": result["disease"],
                        "confidence": result["confidence"],
                        "image_path": filename,
                        "timestamp": datetime.utcnow()
                    })

                    results.append(result)

                except Exception as img_error:

                    logger.error(
                        f"JSON image processing failed: {img_error}",
                        exc_info=True
                    )

                    results.append({
                        "image": image_path,
                        "error": str(img_error)
                    })

                finally:
                    # ==========================================
                    # Cleanup downloaded temp files only
                    # ==========================================
                    if (
                        temp_path
                        and image_path.startswith('http')
                        and os.path.exists(temp_path)
                    ):
                        try:
                            os.remove(temp_path)
                        except Exception as cleanup_error:
                            logger.warning(
                                f"Temp cleanup failed: {cleanup_error}"
                            )

        else:
            return jsonify({
                "error": "Unsupported content type"
            }), 400

        # =========================================================
        # Update farm stats
        # =========================================================
        if farm_id:
            CreateFarmService.update_farm_analysis_stats(
                farm_id,
                total_count,
                diseased_count
            )

        return jsonify({
            "status": "success",
            "total_images": total_count,
            "diseased_images": diseased_count,
            "healthy_images": total_count - diseased_count,
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        })

    except Exception as e:

        logger.error(
            f"Prediction failed: {e}",
            exc_info=True
        )

        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

# ========== SERVE UPLOADS FROM MONGODB ==========
@app.route('/api/uploads/<file_id>')
def serve_upload(file_id):
    """Serve files stored in GridFS"""
    from bson.objectid import ObjectId
    try:
        grid_out = mongo_storage.fs.get(ObjectId(file_id))
        return send_from_directory(
            directory="",
            path=grid_out.filename,
            mimetype=grid_out.content_type
        )
    except Exception as e:
        return jsonify({"error": "File not found"}), 404

# ========== APP ENTRY POINT ==========
if __name__ == '__main__':
    PORT = int(os.getenv("PORT", 5500))
    socketio.run(app, debug=False, host='0.0.0.0', port=PORT)
