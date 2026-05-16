"""
AgriConnect Backend - Optimized for Render Free Tier
"""

import os

# MUST BE FIRST - TensorFlow memory optimization
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys
import logging
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from flask_socketio import SocketIO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ========== LOGGING ==========
# Use only stream handler - no file on ephemeral filesystem
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

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

# ========== CREATE FLASK APP ==========
app = Flask(__name__)
app.config.from_object(Config)
app.secret_key = Config.SECRET_KEY

# ========== CORS CONFIGURATION - NO WILDCARD SUBDOMAINS ==========
FRONTEND_URLS = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://localhost:5000",
]

frontend_env = os.environ.get("FRONTEND_URL")
if frontend_env:
    FRONTEND_URLS.append(frontend_env)

# Add Render production URL if present
render_url = os.environ.get("RENDER_EXTERNAL_URL")
if render_url:
    FRONTEND_URLS.append(render_url)

CORS(
    app,
    supports_credentials=True,
    resources={r"/*": {"origins": FRONTEND_URLS}}
)

# ========== SOCKET.IO WITH THREADING MODE - NO EVENTLET ==========
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    logger=False,
    engineio_logger=False,
    ping_timeout=60,
    ping_interval=25
)

logger.info("SocketIO initialized with threading mode")

# ========== LAZY IMPORTS FOR HEAVY MODULES ==========
def lazy_import_mongo_storage():
    """Lazy import mongo_storage to avoid early failures"""
    try:
        from mongo_storage import mongo_storage
        return mongo_storage
    except Exception as e:
        logger.error(f"Failed to import mongo_storage: {e}")
        return None

def lazy_import_crop_disease_predictor():
    """Lazy import CropDiseasePredictor only when needed"""
    try:
        from crop_disease_outbreak.data_service.crop_reports import CropDiseasePredictor
        return CropDiseasePredictor
    except Exception as e:
        logger.warning(f"Could not import CropDiseasePredictor: {e}")
        return None

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
        "version": "3.0.0"
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

# ========== ERROR HANDLERS - NO INTERNAL ERROR LEAKS ==========
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Resource not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    """Generic exception handler - doesn't leak internal details"""
    logger.error(f"Unhandled exception: {e}", exc_info=True)
    return jsonify({"error": "Internal server error"}), 500

# ========== OPTIMIZED BLUEPRINT LOADING ==========
# Only load lightweight blueprints at startup
# Heavy ML blueprints are loaded lazily via routes

def register_optional_blueprints():
    """Register only lightweight blueprints at startup"""
    modules = [
        ("weather_section.api", "weather_bp", "/api/weather"),
        ("crop_yield_predictor.api", "api_blueprint", None),
        ("market_price.market_prices", "market_prices_bp", None),
        ("farm_models.api", "CropRecommendationBp", "/api/v1"),
    ]

    for module_name, blueprint_name, prefix in modules:
        try:
            module = __import__(module_name, fromlist=[blueprint_name])
            blueprint = getattr(module, blueprint_name)

            if prefix:
                app.register_blueprint(blueprint, url_prefix=prefix)
            else:
                app.register_blueprint(blueprint)

            logger.info(f"Loaded {blueprint_name}")

        except Exception as e:
            logger.warning(f"Could not load {module_name}: {e}")

# Register lightweight blueprints
register_optional_blueprints()

# ========== LAZY-LOADED HEAVY BLUEPRINT ROUTES ==========
# Instead of importing heavy blueprints at startup, create routes
# that lazy-import internally

@app.route('/api/agribot/<path:subpath>', methods=['GET', 'POST'])
def lazy_agribot(subpath):
    """Lazy load Agribot blueprint only when accessed"""
    try:
        # Import only when route is called
        from llm.api import Agribot_bp1
        # Create a request context and forward to blueprint
        return Agribot_bp1.handle_request(subpath)
    except Exception as e:
        logger.error(f"Agribot route failed: {e}")
        return jsonify({"error": "Agribot service unavailable"}), 503

@app.route('/api/weed/<path:subpath>', methods=['GET', 'POST'])
def lazy_weed(subpath):
    """Lazy load Weed detection blueprint only when accessed"""
    try:
        # Import only when route is called - loads YOLO/OpenCV at that time
        from crop_vs_weed.api import weed_bp
        return weed_bp.handle_request(subpath)
    except Exception as e:
        logger.error(f"Weed detection route failed: {e}")
        return jsonify({"error": "Weed detection service unavailable"}), 503

# ========== MAIN API ENDPOINTS ==========

@app.route('/api/predictDisease', methods=['POST'])
def predict_disease():
    """Disease prediction endpoint with lazy loading"""
    import tempfile

    try:
        # Lazy load the predictor only when needed
        CropDiseasePredictor = lazy_import_crop_disease_predictor()
        if not CropDiseasePredictor:
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
        return jsonify({"status": "error", "error": "Prediction service error"}), 500

# ========== FILE SERVING - FIXED FOR GRIDFS ==========
@app.route('/api/uploads/<file_id>')
def serve_upload(file_id):
    """Serve files from GridFS - proper streaming from MongoDB"""
    try:
        mongo_storage = lazy_import_mongo_storage()

        if not mongo_storage:
            return jsonify({"error": "Storage unavailable"}), 503

        from bson import ObjectId

        # Get file from GridFS
        file_data = mongo_storage.fs.get(ObjectId(file_id))

        # Return as streaming response
        return Response(
            file_data.read(),
            mimetype=file_data.content_type or "application/octet-stream",
            headers={
                "Content-Disposition": f"inline; filename={file_data.filename}"
            }
        )

    except Exception as e:
        logger.error(f"Upload serve failed: {e}")
        return jsonify({"error": "File not found"}), 404

# ========== SOCKET.IO EVENT HANDLERS ==========
# Lazy load SocketIO handlers only when needed
_socketio_handlers_loaded = False

def load_socketio_handlers():
    """Lazy load SocketIO handlers to avoid loading YOLO at startup"""
    global _socketio_handlers_loaded
    if _socketio_handlers_loaded:
        return

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

        _socketio_handlers_loaded = True
        logger.info("SocketIO handlers loaded")

    except Exception as e:
        logger.warning(f"SocketIO event handlers setup failed: {e}")

# Load handlers on first connection
@socketio.on('connect')
def handle_global_connect():
    load_socketio_handlers()

# ========== APPLICATION FACTORY ==========
def create_app():
    """Application factory for better deployment"""
    return app

# ========== PRODUCTION ENTRY POINT ==========
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))

    logger.info(f"Starting AgriConnect server on port {port}")
    logger.info(f"CORS allowed origins: {FRONTEND_URLS}")
    logger.info(f"Async mode: threading (no eventlet)")

    socketio.run(
        app,
        host="0.0.0.0",
        port=port,
        debug=False,
        allow_unsafe_werkzeug=True
    )
    