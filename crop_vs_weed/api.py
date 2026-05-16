# top of file
from flask import Blueprint, current_app, render_template, request, jsonify, session
from flask_cors import CORS
import atexit
import cv2
import base64
import time
import numpy as np
import os
from config import Config
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from io import BytesIO
import threading
import logging
from mongo_storage import mongo_storage
from cloudinary_upload import CloudinaryUploader
from flask_socketio import emit

cloudinary_uploader = CloudinaryUploader()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

weed_bp = Blueprint("weed_bp", __name__, static_folder="static")
CORS(weed_bp)

# -----------------------------
# GLOBAL CACHE
# -----------------------------
_weed_model = None
_model_path = None


def get_socketio():
    return current_app.extensions['socketio']


def get_model_path():
    """
    Download model ONLY when needed
    """
    global _model_path

    if _model_path is None:
        from huggingface_hub import hf_hub_download

        HF_REPO_ID = os.getenv("HF_REPO_ID")

        logger.info("Downloading YOLO model from HuggingFace...")

        _model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="Crop_Weed_Detection_yolo_API.pt"
        )

        logger.info(f"YOLO model downloaded: {_model_path}")

    return _model_path


def get_weed_model():
    """
    Lazy-load YOLO model
    """
    global _weed_model

    if _weed_model is None:
        logger.info("Loading YOLO model into memory...")

        from ultralytics import YOLO

        model_path = get_model_path()

        _weed_model = YOLO(model_path)

        logger.info("YOLO model loaded successfully")

    return _weed_model

class CameraManager:
    def __init__(self):
        self.camera = None
        self.is_streaming = False
        self.lock = threading.Lock()
        self.camera_type = None  # 'webcam' or 'video'
        self.camera_source = None  # webcam index or video path

    def get_camera(self):
        with self.lock:
            return self.camera

    def set_camera(self, camera, camera_type, camera_source):
        with self.lock:
            if self.camera:
                try:
                    self.camera.release()
                except:
                    pass
            self.camera = camera
            self.camera_type = camera_type
            self.camera_source = camera_source

    def get_streaming(self):
        with self.lock:
            return self.is_streaming

    def set_streaming(self, streaming):
        with self.lock:
            self.is_streaming = streaming

    def release(self):
        with self.lock:
            if self.camera:
                try:
                    self.camera.release()
                except:
                    pass
                self.camera = None
            self.is_streaming = False
            self.camera_type = None
            self.camera_source = None

    def get_camera_info(self):
        with self.lock:
            return self.camera_type, self.camera_source

# Global camera manager
camera_manager = CameraManager()

# Flask routes

@weed_bp.route("/upload_image", methods=["POST"])
def upload_image():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        if file and allowed_file(file.filename):
            # Read file bytes
            file_bytes = file.read()
            filename = f"img_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            farm_id = request.form.get('farm_id', 'unknown')

            # Upload to Cloudinary/GridFS
            upload_result = mongo_storage.save_upload(file_bytes, filename, farm_id)

            # Process image from bytes
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                tmp.write(file_bytes)
                temp_path = tmp.name

            results = process_image(temp_path)
            os.unlink(temp_path)

            if not results:
                return jsonify({"error": "Failed to process image"}), 500

            graph_data = generate_detection_graphs(results)

            return jsonify({
                "success": True,
                "image_url": mongo_storage.get_upload_url(upload_result),
                "results": results,
                "graphs": graph_data
            })

        return jsonify({"error": "Invalid file type"}), 400
    except Exception as e:
        logger.error(f"Error in upload_image: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@weed_bp.route("/detect", methods=["POST"])
def detect_weed():
    try:
        model = get_weed_model()

        if not model:
            return jsonify({"error": "Model not loaded"}), 500

        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            tmp.write(file.read())
            temp_path = tmp.name

        results = process_image(temp_path)
        try:
            os.unlink(temp_path)
        except Exception:
            pass

        if not results:
            return jsonify({"error": "Failed to process image"}), 500

        return jsonify({"success": True, "results": results})
    except Exception as e:
        logger.exception(f"Error in detect_weed: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@weed_bp.route("/start_webcam", methods=['POST','GET'])
def start_webcam():
    try:
        if camera_manager.get_streaming():
            return jsonify({"success": False, "message": "Webcam already running"})

        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            return jsonify({"success": False, "message": "Failed to open webcam"})

        camera_manager.set_camera(camera, 'webcam', 0)
        camera_manager.set_streaming(True)

        # 🔑 start emitting frames
        threading.Thread(
            target=generate_frames,
            args=(get_socketio(),),
            daemon=True
        ).start()

        logger.info("Webcam started successfully")
        return jsonify({"success": True, "message": "Webcam started"})
    except Exception as e:
        logger.error(f"Error starting webcam: {e}")
        return jsonify({"success": False, "message": f"Error starting webcam: {str(e)}"})

@weed_bp.route("/stop_webcam",methods=['POST','GET'])
def stop_webcam():
    try:
        camera_type, _ = camera_manager.get_camera_info()
        if camera_type == 'webcam' and camera_manager.get_streaming():
            camera_manager.set_streaming(False)
            camera_manager.release()
            logger.info("Webcam stopped successfully")
            return jsonify({"success": True, "message": "Webcam stopped"})
        return jsonify({"success": False, "message": "Webcam not running"})
    except Exception as e:
        logger.error(f"Error stopping webcam: {e}")
        return jsonify({"success": False, "message": f"Error stopping webcam: {str(e)}"})

@weed_bp.route("/start_video", methods=['POST','GET'])
def start_video():
    try:
        # Prefer filename from client JSON body
        filename = None
        if request.method == 'POST' and request.is_json:
            payload = request.get_json(silent=True) or {}
            filename = payload.get('filename') or payload.get('video_path')

        # fallback to session (existing behavior)
        if not filename:
            video_path = session.get('video_path')
        else:
            # if client sent a relative path like "/static/uploads/xxx", convert to absolute
            if filename.startswith('/'):
                # make sure to translate to filesystem path if necessary
                # assuming uploaded files saved under Config.UPLOAD_FOLDER
                # if filename is "/static/uploads/video_xxx.mp4" convert to os.path.join(Config.UPLOAD_FOLDER, basename)
                filename_fs = os.path.basename(filename)
                video_path = os.path.join(Config.UPLOAD_FOLDER, filename_fs)
            else:
                video_path = os.path.join(Config.UPLOAD_FOLDER, filename)

        # final check
        if not video_path or not os.path.exists(video_path):
            logger.warning("start_video: video_path missing or not found: %s", video_path)
            return jsonify({"success": False, "message": "No video uploaded or file not found"}), 400

        if camera_manager.get_streaming():
            return jsonify({"success": False, "message": "Video already playing"})

        camera = cv2.VideoCapture(video_path)
        if not camera.isOpened():
            return jsonify({"success": False, "message": "Failed to open video file"})

        camera_manager.set_camera(camera, 'video', video_path)
        camera_manager.set_streaming(True)

        # start emitter thread using socketio stored in current_app
        socketio = get_socketio()
        threading.Thread(
            target=generate_frames,
            args=(socketio,),
            daemon=True
        ).start()

        logger.info("Video playback started successfully: %s", video_path)
        return jsonify({"success": True, "message": "Video playback started"})
    except Exception as e:
        logger.exception("Error starting video: %s", e)
        return jsonify({"success": False, "message": f"Error starting video: {str(e)}"}), 500

@weed_bp.route("/stop_video",methods=['POST','GET'])
def stop_video():
    try:
        camera_type, _ = camera_manager.get_camera_info()
        if camera_type == 'video' and camera_manager.get_streaming():
            camera_manager.set_streaming(False)
            camera_manager.release()
            logger.info("Video stopped successfully")
            return jsonify({"success": True, "message": "Video stopped"})
        return jsonify({"success": False, "message": "Video not playing"})
    except Exception as e:
        logger.error(f"Error stopping video: {e}")
        return jsonify({"success": False, "message": f"Error stopping video: {str(e)}"})

@weed_bp.route("/stop_streaming",methods=['POST','GET'])
def stop_streaming():
    try:
        if camera_manager.get_streaming():
            camera_manager.set_streaming(False)
            camera_manager.release()
            logger.info("Streaming stopped successfully")
            return jsonify({"success": True, "message": "Streaming stopped"})
        return jsonify({"success": False, "message": "No active streaming"})
    except Exception as e:
        logger.error(f"Error stopping streaming: {e}")
        return jsonify({"success": False, "message": f"Error stopping streaming: {str(e)}"})

# Helper functions
def allowed_file(filename, allowed_extensions=None):
    if allowed_extensions is None:
        allowed_extensions = Config.ALLOWED_EXTENSIONS

    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

def process_image(image_path):
    try:
        model = get_weed_model()

        if not model:
            logger.error("Model not loaded")
            return None

        frame = cv2.imread(image_path)
        if frame is None:
            logger.error(f"Failed to read image: {image_path}")
            return None

        results = model.predict(frame, verbose=False)
        if not results or len(results) == 0:
            logger.error("No results from model prediction")
            return None

        annotated = results[0].plot()

        # Save annotated image
        annotated_filename = f"annotated_{os.path.basename(image_path)}"
        annotated_path = os.path.join(Config.UPLOAD_FOLDER, annotated_filename)
        cv2.imwrite(annotated_path, annotated)

        # Collect detection stats
        detections = []
        class_counts = {"Soil": 0, "Weed": 0, "Cotton": 0}
        confidences = {"Soil": [], "Weed": [], "Cotton": []}

        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                label = results[0].names[cls_id]
                confidence = float(box.conf[0])
                class_counts[label] += 1
                confidences[label].append(confidence)
                detections.append({
                    "label": label,
                    "confidence": confidence,
                    "bbox": box.xywh[0].tolist() if hasattr(box.xywh[0], 'tolist') else box.xywh[0]
                })

        # Calculate average confidence
        avg_confidences = {}
        for label, confs in confidences.items():
            avg_confidences[label] = sum(confs) / len(confs) if confs else 0

        return {
            "detections": detections,
            "counts": class_counts,
            "confidences": avg_confidences,
            "annotated_image": f"/static/uploads/{annotated_filename}"
        }
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return None

def generate_detection_graphs(results):
    try:
        if not results:
            return None

        # Create pie chart of class distribution
        labels = []
        sizes = []
        colors = ['#00FF00', '#FF0000', '#FFA500']  # Green, Red, Orange

        for label, count in results['counts'].items():
            if count > 0:
                labels.append(label)
                sizes.append(count)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        if sizes:
            plt.pie(sizes, labels=labels, colors=colors[:len(sizes)], autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Class Distribution')

        # Create bar chart of average confidence
        plt.subplot(1, 2, 2)
        conf_labels = []
        conf_values = []

        for label, conf in results['confidences'].items():
            if conf > 0:
                conf_labels.append(label)
                conf_values.append(conf)

        if conf_values:
            conf_colors = [colors[i % len(colors)] for i in range(len(conf_labels))]
            bars = plt.bar(conf_labels, conf_values, color=conf_colors)
            plt.title('Average Confidence by Class')
            plt.ylabel('Confidence')
            plt.ylim(0, 1)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom')

        # Save the plot to a BytesIO object
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)

        # Encode the image
        graph_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

        return graph_data
    except Exception as e:
        logger.error(f"Error generating graphs: {e}")
        return None

def generate_frames(socketio):
    try:
        if not camera_manager.get_streaming():
            return

        camera = camera_manager.get_camera()
        if not camera or not camera.isOpened():
            socketio.emit("error", {"message": "Camera not available"}, namespace='/weed')
            return

        frame_count = 0
        start_time = time.time()

        # ensure model is loaded (lazy)
        model = get_weed_model()

        while camera_manager.get_streaming():
            try:
                ret, frame = camera.read()
                if not ret:
                    # Handle end of video or camera disconnect
                    camera_type, camera_source = camera_manager.get_camera_info()

                    if camera_type == 'video':
                        # For video files, restart from beginning
                        camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        # For webcam, try to reconnect
                        logger.warning("Camera disconnected, attempting to reconnect...")
                        camera.release()
                        time.sleep(1)
                        new_camera = cv2.VideoCapture(camera_source)
                        if new_camera.isOpened():
                            camera_manager.set_camera(new_camera, camera_type, camera_source)
                            camera = new_camera
                            continue
                        else:
                            socketio.emit("error", {"message": "Camera reconnection failed"}, namespace='/weed')
                            break

                # Process frame with model
                if model:
                    try:
                        results = model.predict(frame, verbose=False)
                        annotated = results[0].plot() if results and len(results) > 0 else frame
                    except Exception as e:
                        logger.error(f"Model prediction error: {e}")
                        results = None
                        annotated = frame

                    # Collect detection stats
                    detections = []
                    class_counts = {"Soil": 0, "Weed": 0, "Cotton": 0}

                    if results and results[0].boxes is not None:
                        for box in results[0].boxes:
                            cls_id = int(box.cls[0])
                            label = results[0].names[cls_id]
                            class_counts[label] += 1
                            detections.append(label)

                else:
                    annotated = frame
                    detections = []
                    class_counts = {"Soil": 0, "Weed": 0, "Cotton": 0}

                # Calculate FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time if elapsed_time > 0 else 0

                # Convert frame to base64 for browser
                _, buffer = cv2.imencode('.jpg', annotated)
                frame_b64 = base64.b64encode(buffer).decode("utf-8")

                # Send via Socket.IO with error handling
                try:
                    socketio.emit("frame", {
                        "image": frame_b64,
                        "detections": detections,
                        "counts": class_counts,
                        "fps": round(fps, 2)
                    }, namespace='/weed')
                except Exception as e:
                    logger.error(f"SocketIO emit error: {e}")
                    break

                # Control frame rate
                time.sleep(0.033)  # ~30 FPS

            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                socketio.emit("error", {"message": f"Frame processing error: {str(e)}"}, namespace='/weed')
                time.sleep(1)  # Prevent rapid error looping

    except Exception as e:
        logger.error(f"Error in generate_frames: {e}")
        socketio.emit("error", {"message": f"Stream error: {str(e)}"}, namespace='/weed')
    finally:
        # Cleanup
        camera_manager.release()

# Cleanup function
def cleanup():
    try:
        camera_manager.release()
        logger.info("[CLEANUP] Camera released and streaming stopped")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

atexit.register(cleanup)
