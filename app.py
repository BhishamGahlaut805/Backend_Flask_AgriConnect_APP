"""
AgriConnect Backend - Production Ready Flask Application
Deployment: Render.com
"""

import os
import sys
import time
import uuid
import logging
import tempfile
from datetime import datetime
from pathlib import Path

#import Eventlet for SocketIO async support
import eventlet
eventlet.monkey_patch()

# Configure logging FIRST - before any other imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log') if os.environ.get('LOG_FILE') else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress noisy TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Import Flask and extensions
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ========== CONFIGURATION CLASS ==========
class Config:
    """Application configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY', os.urandom(24).hex())
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    TESTING = False

    # MongoDB settings
    MONGO_URI = os.environ.get('MONGODB_URI', 'mongodb://localhost:27017/')
    MONGO_DB = os.environ.get('MONGO_DB_NAME', 'agriconnect')

    # Cloudinary settings
    CLOUDINARY_CLOUD_NAME = os.environ.get('CLOUDINARY_CLOUD_NAME')
    CLOUDINARY_API_KEY = os.environ.get('CLOUDINARY_API_KEY')
    CLOUDINARY_API_SECRET = os.environ.get('CLOUDINARY_API_SECRET')

    # Render specific
    PORT = int(os.environ.get('PORT', 10000))
    HOST = '0.0.0.0'

# ========== LAZY IMPORTS FOR HEAVY MODULES ==========
# These modules are imported only when needed to speed up startup

def lazy_import_mongo_storage():
    """Lazy import mongo_storage to avoid early failures"""
    try:
        from mongo_storage import mongo_storage
        return mongo_storage
    except Exception as e:
        logger.error(f"Failed to import mongo_storage: {e}")
        return None

def lazy_import_models():
    """Lazy import ML models to avoid loading them at startup"""
    global CropDiseasePredictor
    try:
        from crop_disease_outbreak.data_service.crop_reports import CropDiseasePredictor
        return True
    except Exception as e:
        logger.warning(f"Could not import CropDiseasePredictor: {e}")
        return False

# ========== CREATE FLASK APP ==========
app = Flask(__name__)
app.config.from_object(Config)

# ========== CORS CONFIGURATION ==========
FRONTEND_URLS = [
    "http://localhost:5173",
    "http://localhost:3000",
    os.environ.get("FRONTEND_URL", "https://agriconnect-frontend.onrender.com"),
    "https://*.onrender.com"
]

# Remove None values
FRONTEND_URLS = [url for url in FRONTEND_URLS if url]

CORS(app,
     supports_credentials=True,
     origins=FRONTEND_URLS,
     allow_headers=["Content-Type", "Authorization"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])

app.secret_key = Config.SECRET_KEY

# ========== SOCKET.IO WITH FALLBACK ==========
try:
    socketio = SocketIO(
        app,
        cors_allowed_origins=FRONTEND_URLS,
        async_mode='eventlet',
        logger=False,
        engineio_logger=False
    )
    logger.info("SocketIO initialized successfully")
except Exception as e:
    logger.warning(f"SocketIO initialization failed: {e}")
    socketio = None

# ========== HEALTH CHECK ENDPOINTS ==========
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Render"""
    mongo_status = False
    try:
        mongo_storage = lazy_import_mongo_storage()
        if mongo_storage and mongo_storage.client:
            mongo_storage.client.admin.command('ping')
            mongo_status = True
    except:
        pass

    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "AgriConnect Backend",
        "mongodb": mongo_status,
        "version": "2.0.0"
    })

@app.route('/', methods=['GET'])
def home():
    """Root endpoint"""
    return jsonify({
        "message": "AgriConnect API is running",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "api": "/api/*",
            "websocket": "/socket.io/*"
        }
    })

@app.route('/ready', methods=['GET'])
def readiness_check():
    """Readiness probe for Render"""
    return jsonify({"ready": True}), 200

# ========== ERROR HANDLERS ==========
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Resource not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {e}", exc_info=True)
    return jsonify({"error": str(e)}), 500

# ========== BLUEPRINT REGISTRATION WITH ERROR HANDLING ==========
def register_blueprint_safely(app, blueprint, url_prefix=None, name=None):
    """Safely register a blueprint with error handling"""
    try:
        if url_prefix:
            app.register_blueprint(blueprint, url_prefix=url_prefix)
        else:
            app.register_blueprint(blueprint)
        logger.info(f"Registered blueprint: {blueprint.name if hasattr(blueprint, 'name') else name}")
        return True
    except Exception as e:
        logger.warning(f"Failed to register blueprint {name}: {e}")
        return False

# Import and register blueprints with error handling
try:
    from weather_section.api import weather_bp
    register_blueprint_safely(app, weather_bp, url_prefix='/api/weather', name='weather')
except Exception as e:
    logger.warning(f"Weather module unavailable: {e}")

try:
    from crop_vs_weed.api import weed_bp
    register_blueprint_safely(app, weed_bp, name='weed')
except Exception as e:
    logger.warning(f"Weed module unavailable: {e}")

try:
    from crop_yield_predictor.api import api_blueprint
    register_blueprint_safely(app, api_blueprint, name='crop_yield')
except Exception as e:
    logger.warning(f"Crop yield module unavailable: {e}")

try:
    from market_price.market_prices import market_prices_bp
    register_blueprint_safely(app, market_prices_bp, name='market_prices')
except Exception as e:
    logger.warning(f"Market prices module unavailable: {e}")

try:
    from farm_models.api import CropRecommendationBp
    register_blueprint_safely(app, CropRecommendationBp, url_prefix='/api/v1', name='crop_recommendation')
except Exception as e:
    logger.warning(f"Crop recommendation module unavailable: {e}")

try:
    from llm.api import Agribot_bp1
    register_blueprint_safely(app, Agribot_bp1, url_prefix="/api/agribot", name='agribot')
except Exception as e:
    logger.warning(f"Agribot module unavailable: {e}")

# ========== MAIN API ENDPOINTS ==========

@app.route('/api/predictDisease', methods=['POST'])
def predict_disease():
    """Disease prediction endpoint with lazy loading"""
    import tempfile
    import requests

    try:
        # Lazy load the predictor only when needed
        if not lazy_import_models():
            return jsonify({"error": "ML models not available"}), 503

        predictor = CropDiseasePredictor()
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

        # Handle file uploads
        if 'image' in request.files:
            images = request.files.getlist('image')

            lat_list = request.form.get('latitude', '')
            lon_list = request.form.get('longitude', '')

            latitudes = [float(x.strip()) for x in lat_list.split(',') if x.strip()]
            longitudes = [float(x.strip()) for x in lon_list.split(',') if x.strip()]

            for i, image in enumerate(images):
                temp_path = None
                try:
                    ext = os.path.splitext(image.filename)[-1]
                    unique_name = f"{uuid.uuid4().hex}{ext}"
                    image_bytes = image.read()

                    # Save to storage if available
                    mongo_storage = lazy_import_mongo_storage()
                    if mongo_storage:
                        upload_result = mongo_storage.save_upload(image_bytes, unique_name, farm_id)
                    else:
                        upload_result = {"storage": "local", "gridfs_id": unique_name}

                    # Create temp file for prediction
                    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
                        tmp_file.write(image_bytes)
                        tmp_file.flush()
                        temp_path = tmp_file.name

                    total_count += 1
                    result = predictor.predict_crop_disease(temp_path, model_type)

                    if upload_result.get("storage") == "cloudinary":
                        image_url = upload_result.get("cloudinary_url")
                    else:
                        image_url = f"/api/uploads/{upload_result.get('gridfs_id', unique_name)}"

                    result["image"] = unique_name
                    result["image_url"] = image_url
                    result["latitude"] = latitudes[i] if i < len(latitudes) else None
                    result["longitude"] = longitudes[i] if i < len(longitudes) else None

                    if result["disease"].lower() != "healthy":
                        diseased_count += 1

                    results.append(result)

                except Exception as img_error:
                    logger.error(f"Image processing failed: {img_error}")
                    results.append({"image": image.filename, "error": str(img_error)})
                finally:
                    if temp_path and os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except:
                            pass

        return jsonify({
            "status": "success",
            "total_images": total_count,
            "diseased_images": diseased_count,
            "healthy_images": total_count - diseased_count,
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        })

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500

# ========== SOCKET.IO EVENT HANDLERS ==========
if socketio:
    try:
        from crop_vs_weed.api import generate_frames, camera_manager

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
            if camera_manager:
                camera_manager.set_streaming(False)
            logger.info('Stream stopped by client')
    except Exception as e:
        logger.warning(f"SocketIO event handlers setup failed: {e}")

# ========== FILE SERVING ==========
@app.route('/api/uploads/<file_id>')
def serve_upload(file_id):
    """Serve files from GridFS or local storage"""
    try:
        mongo_storage = lazy_import_mongo_storage()
        if mongo_storage:
            from bson.objectid import ObjectId
            grid_out = mongo_storage.fs.get(ObjectId(file_id))
            return send_from_directory(
                directory="",
                path=grid_out.filename,
                mimetype=grid_out.content_type
            )
        return jsonify({"error": "Storage not available"}), 503
    except Exception as e:
        logger.error(f"File serving failed: {e}")
        return jsonify({"error": "File not found"}), 404

# ========== APPLICATION FACTORY ==========
def create_app():
    """Application factory for better deployment"""
    return app

# ========== PRODUCTION ENTRY POINT ==========
# ========== PRODUCTION ENTRY POINT ==========
if __name__ == '__main__':
    PORT = int(os.environ.get("PORT", 10000))
    HOST = "0.0.0.0"

    print(f"=== STARTING SERVER ON {HOST}:{PORT} ===", flush=True)

    try:
        if socketio:
            socketio.run(
                app,
                host=HOST,
                port=PORT,
                debug=False
            )
        else:
            app.run(
                host=HOST,
                port=PORT,
                debug=False
            )
    except Exception as e:
        print(f"SERVER FAILED: {e}", flush=True)
        raise
