"""
Unified MongoDB Storage Service
Handles all database operations including:
- Models storage (GridFS)
- File uploads (Cloudinary + GridFS)
- Farms management
- Disease reports
- Statistics and summaries
- LLM chat history
- Training data
- Predictions
"""

import os
import io
import pickle
import gridfs
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, BinaryIO
from bson import ObjectId
from pymongo import MongoClient, DESCENDING, ASCENDING
from pymongo.errors import ConnectionFailure, DuplicateKeyError

logger = logging.getLogger(__name__)


class MongoStorage:
    """Unified MongoDB storage with GridFS support and all service methods"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._init_storage()

    def _init_storage(self):
        """Initialize MongoDB connections and collections"""
        from config import Config

        self.mongo_uri = Config.MONGO_URI
        self.client = MongoClient(self.mongo_uri)
        self.db = self.client[Config.MONGO_DB]
        self.fs = gridfs.GridFS(self.db)

        # ========== COLLECTIONS ==========
        # Core collections
        self.models = self.db["ml_models"]
        self.uploads = self.db["uploads"]
        self.logs = self.db["system_logs"]

        # Farm management
        self.farms = self.db["farms"]
        self.farm_stats = self.db["farm_stats"]
        self.user_summary = self.db["user_summary"]

        # Disease tracking
        self.disease_reports = self.db["disease_reports"]
        self.geo_stats = self.db["geo_stats"]

        # Predictions & Training
        self.predictions = self.db["predictions"]
        self.training_data = self.db["training_data"]
        self.crop_training_data = self.db["crop_training_data"]

        # Yield models
        self.yield_models = self.db["YieldModels"]
        self.yield_predictions = self.db["YieldPredictions"]
        self.what_if_simulations = self.db["WhatIfSimulations"]
        self.state_yield = self.db["StateYield"]
        self.state_yield_all = self.db["StateYieldAll"]

        # LLM & Chat
        self.chat_history = self.db["chat_history"]
        self.uploaded_files = self.db["uploaded_files"]
        self.stored_news = self.db["stored_news"]
        self.weather_data = self.db["weather_data"]
        self.bulletins = self.db["bulletins"]
        self.training_jobs = self.db["training_jobs"]
        self.training_status = self.db["training_status"]

        # Create indexes
        self._create_indexes()

    def _create_indexes(self):
        """Create necessary indexes for performance"""
        # Uploads
        self.uploads.create_index([("uploaded_at", DESCENDING)])
        self.uploads.create_index([("farm_id", 1)])

        # Farms
        self.farms.create_index([("farm_id", 1)], unique=True)
        self.farms.create_index([("user_id", 1)])

        # Disease reports
        self.disease_reports.create_index([("farm_id", 1), ("timestamp", DESCENDING)])
        self.disease_reports.create_index([("timestamp", DESCENDING)])
        self.disease_reports.create_index([("latitude", 1), ("longitude", 1)])

        # Predictions
        self.predictions.create_index([("farm_id", 1), ("type", 1), ("timestamp", DESCENDING)])

        # Farm stats
        self.farm_stats.create_index([("farm_id", 1), ("date", 1)], unique=True)

        # User summary
        self.user_summary.create_index([("user_id", 1)], unique=True)

        # Logs
        self.logs.create_index([("timestamp", DESCENDING)])
        self.logs.create_index([("level", 1)])

        # Chat history
        self.chat_history.create_index([("session_id", 1), ("timestamp", DESCENDING)])

        # News
        self.stored_news.create_index([("fetched_at", DESCENDING)])

        # Training jobs
        self.training_jobs.create_index([("created_at", DESCENDING)])
        self.training_status.create_index([("updated_at", DESCENDING)])

        # State yield
        self.state_yield_all.create_index([("state", 1), ("crop", 1), ("season", 1), ("version", 1)])

        # Models
        self.models.create_index([("model_type", 1), ("active", 1)])
        self.models.create_index([("farm_id", 1), ("crop", 1)])

    # ==================== HEALTH CHECK ====================

    def health_check(self) -> bool:
        """Check MongoDB connection health"""
        try:
            self.client.admin.command("ping")
            return True
        except ConnectionFailure:
            return False

    def close(self):
        """Close MongoDB connection"""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ==================== MODEL STORAGE (GridFS) ====================

    def save_model(self, model_bytes: bytes, model_type: str, metadata: Dict = None) -> str:
        """Save ML model to GridFS"""
        try:
            filename = f"{model_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pt"

            # Deactivate previous active model
            self.models.update_many(
                {"model_type": model_type, "active": True},
                {"$set": {"active": False}}
            )

            # Save new model
            file_id = self.fs.put(
                model_bytes,
                filename=filename,
                metadata={
                    "model_type": model_type,
                    "active": True,
                    "created_at": datetime.utcnow(),
                    **(metadata or {})
                }
            )

            # Store metadata
            self.models.update_one(
                {"model_type": model_type},
                {
                    "$set": {
                        "gridfs_id": file_id,
                        "filename": filename,
                        "active": True,
                        "updated_at": datetime.utcnow(),
                        **(metadata or {})
                    }
                },
                upsert=True
            )

            logger.info(f"Model saved: {model_type}, ID: {file_id}")
            return str(file_id)

        except Exception as e:
            logger.error(f"Failed to save model {model_type}: {e}")
            raise

    def load_model(self, model_type: str, farm_id: str = None, crop: str = None) -> Optional[bytes]:
        """Load ML model from GridFS"""
        try:
            query = {"model_type": model_type, "active": True}
            if farm_id:
                query["farm_id"] = farm_id
            if crop:
                query["crop"] = crop

            model_doc = self.models.find_one(query)
            if not model_doc:
                return None

            file_id = model_doc["gridfs_id"]
            grid_out = self.fs.get(ObjectId(file_id))
            return grid_out.read()

        except Exception as e:
            logger.error(f"Failed to load model {model_type}: {e}")
            return None

    def save_torch_model(self, model, model_type: str, farm_id: str = None, crop: str = None):
        """Save PyTorch model to GridFS"""
        import torch
        buffer = io.BytesIO()
        torch.save(model, buffer)
        buffer.seek(0)

        metadata = {"farm_id": farm_id, "crop": crop}
        return self.save_model(buffer.getvalue(), model_type, metadata)

    def load_torch_model(self, model_type: str, farm_id: str = None, crop: str = None):
        """Load PyTorch model from GridFS"""
        import torch
        model_bytes = self.load_model(model_type, farm_id, crop)
        if not model_bytes:
            return None

        buffer = io.BytesIO(model_bytes)
        return torch.load(buffer, map_location='cpu')

    def delete_model(self, model_type: str) -> bool:
        """Delete a model type"""
        try:
            model_doc = self.models.find_one({"model_type": model_type})
            if model_doc and model_doc.get("gridfs_id"):
                self.fs.delete(ObjectId(model_doc["gridfs_id"]))
            self.models.delete_many({"model_type": model_type})
            logger.info(f"Model deleted: {model_type}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete model {model_type}: {e}")
            return False

    # ==================== FILE UPLOADS ====================

    def save_upload(self, file_bytes: bytes, filename: str, farm_id: str,
                   content_type: str = "image/jpeg") -> Dict:
        """Save uploaded file - uses Cloudinary with GridFS fallback"""
        result = {"success": True, "farm_id": farm_id, "filename": filename}

        try:
            # Try Cloudinary first
            from cloudinary_upload import CloudinaryUploader
            uploader = CloudinaryUploader()
            cloud_result = uploader.upload_file(file_bytes, filename, f"farm_{farm_id}")

            if cloud_result["success"]:
                result["storage"] = "cloudinary"
                result["cloudinary_url"] = cloud_result["url"]
                result["cloudinary_public_id"] = cloud_result["public_id"]
            else:
                raise Exception("Cloudinary upload failed")

        except Exception as e:
            logger.warning(f"Cloudinary upload failed, using GridFS fallback: {e}")
            # Fallback to GridFS
            file_id = self.fs.put(
                file_bytes,
                filename=filename,
                metadata={
                    "farm_id": farm_id,
                    "content_type": content_type,
                    "uploaded_at": datetime.utcnow()
                }
            )
            result["storage"] = "gridfs"
            result["gridfs_id"] = str(file_id)

        # Store metadata
        upload_doc = {
            "filename": filename,
            "farm_id": farm_id,
            "uploaded_at": datetime.utcnow(),
            "size": len(file_bytes),
            "content_type": content_type,
            **result
        }
        self.uploads.insert_one(upload_doc)

        return upload_doc

    def get_upload_url(self, upload_doc: Dict) -> str:
        """Get URL for uploaded file"""
        if upload_doc.get("storage") == "cloudinary":
            return upload_doc["cloudinary_url"]
        else:
            return f"/api/uploads/{upload_doc.get('gridfs_id')}"

    def get_upload_by_id(self, file_id: str) -> Optional[bytes]:
        """Get file content by GridFS ID"""
        try:
            grid_out = self.fs.get(ObjectId(file_id))
            return grid_out.read()
        except Exception:
            return None

    # ==================== LOGGING ====================

    def log(self, level: str, message: str, source: str = None, metadata: Dict = None):
        """Store log in MongoDB"""
        log_doc = {
            "level": level,
            "message": message,
            "source": source,
            "timestamp": datetime.utcnow(),
            "metadata": metadata or {}
        }
        self.logs.insert_one(log_doc)

    def get_logs(self, level: str = None, limit: int = 100, skip: int = 0):
        """Retrieve logs"""
        query = {}
        if level:
            query["level"] = level

        cursor = self.logs.find(query).sort("timestamp", DESCENDING).skip(skip).limit(limit)
        return list(cursor)

    # ==================== TRAINING DATA ====================

    def save_training_data(self, data: Any, data_type: str, farm_id: str = None,
                          crop: str = None, metadata: Dict = None) -> str:
        """Save training data to MongoDB GridFS"""
        try:
            import pickle
            buffer = io.BytesIO()
            pickle.dump(data, buffer)
            buffer.seek(0)

            file_id = self.fs.put(
                buffer.getvalue(),
                filename=f"{data_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pkl",
                metadata={
                    "type": data_type,
                    "farm_id": farm_id,
                    "crop": crop,
                    "created_at": datetime.utcnow(),
                    **(metadata or {})
                }
            )

            self.training_data.update_one(
                {"type": data_type, "farm_id": farm_id, "crop": crop},
                {
                    "$set": {
                        "gridfs_id": file_id,
                        "updated_at": datetime.utcnow(),
                        "metadata": metadata
                    }
                },
                upsert=True
            )

            return str(file_id)

        except Exception as e:
            logger.error(f"Failed to save training data {data_type}: {e}")
            raise

    def load_training_data(self, data_type: str, farm_id: str = None, crop: str = None) -> Any:
        """Load training data from MongoDB GridFS"""
        try:
            import pickle
            doc = self.training_data.find_one({"type": data_type, "farm_id": farm_id, "crop": crop})
            if not doc:
                return None

            grid_out = self.fs.get(doc["gridfs_id"])
            return pickle.loads(grid_out.read())

        except Exception as e:
            logger.error(f"Failed to load training data {data_type}: {e}")
            return None

    # ==================== PREDICTIONS ====================

    def save_prediction(self, farm_id: str, prediction_type: str, result: Dict,
                       metadata: Dict = None) -> str:
        """Save prediction result"""
        pred_doc = {
            "farm_id": farm_id,
            "type": prediction_type,
            "result": result,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow()
        }
        return str(self.predictions.insert_one(pred_doc).inserted_id)

    def get_latest_prediction(self, farm_id: str, prediction_type: str) -> Optional[Dict]:
        """Get latest prediction for a farm"""
        doc = self.predictions.find_one(
            {"farm_id": farm_id, "type": prediction_type},
            sort=[("timestamp", DESCENDING)]
        )
        return doc

    def get_prediction_history(self, farm_id: str, prediction_type: str, limit: int = 10):
        """Get prediction history"""
        cursor = self.predictions.find(
            {"farm_id": farm_id, "type": prediction_type}
        ).sort("timestamp", DESCENDING).limit(limit)
        return list(cursor)

    # ==================== FARM MANAGEMENT ====================

    def generate_farm_id(self) -> str:
        """Generate unique farm ID"""
        import time
        import uuid
        timestamp = int(time.time())
        random_suffix = uuid.uuid4().hex[:6].upper()
        return f"FARM_{timestamp}_{random_suffix}"

    def create_farm(self, farm_data: Dict) -> str:
        """Create a new farm"""
        farm_id = self.generate_farm_id()
        farm_data["farm_id"] = farm_id
        farm_data["created_at"] = datetime.utcnow()
        farm_data["updated_at"] = datetime.utcnow()

        # Initialize analysis stats
        if "analysis" not in farm_data:
            farm_data["analysis"] = {
                "total_images_analyzed": 0,
                "diseased_images_found": 0,
                "last_updated": datetime.utcnow()
            }

        self.farms.insert_one(farm_data)
        logger.info(f"Farm created: {farm_id}")
        return farm_id

    def get_farm(self, farm_id: str) -> Optional[Dict]:
        """Get farm by ID"""
        return self.farms.find_one({"farm_id": farm_id})

    def get_farm_by_user(self, user_id: str) -> List[Dict]:
        """Get all farms for a user"""
        return list(self.farms.find({"user_id": user_id}))

    def get_all_farms(self) -> List[Dict]:
        """Get all farms"""
        return list(self.farms.find({}))

    def update_farm(self, farm_id: str, update_data: Dict) -> bool:
        """Update farm data"""
        update_data["updated_at"] = datetime.utcnow()
        result = self.farms.update_one(
            {"farm_id": farm_id},
            {"$set": update_data}
        )
        return result.modified_count > 0

    def update_farm_analysis_stats(self, farm_id: str, new_images: int,
                                   new_diseased: int, crop: str = None,
                                   disease: str = None, latitude: float = None,
                                   longitude: float = None):
        """Update farm analysis statistics with all related collections"""
        timestamp = datetime.utcnow()
        today_str = timestamp.strftime("%Y-%m-%d")

        # 1. Update farm document
        self.farms.update_one(
            {"farm_id": farm_id},
            {
                "$inc": {
                    "analysis.total_images_analyzed": new_images,
                    "analysis.diseased_images_found": new_diseased
                },
                "$set": {"analysis.last_updated": timestamp}
            }
        )

        # 2. Update daily stats
        self.farm_stats.update_one(
            {"farm_id": farm_id, "date": today_str},
            {
                "$inc": {
                    "total_images_analyzed": new_images,
                    "diseased_images_found": new_diseased,
                    f"crop_counts.{crop}": 1 if crop else 0,
                    f"disease_counts.{disease}": 1 if disease else 0
                },
                "$set": {"last_updated": timestamp},
                "$setOnInsert": {"created_at": timestamp}
            },
            upsert=True
        )

        # 3. Update geo stats
        if latitude is not None and longitude is not None:
            self.geo_stats.update_one(
                {"farm_id": farm_id, "date": today_str, "lat": latitude, "lon": longitude},
                {
                    "$inc": {
                        "total_images": new_images,
                        "diseased": new_diseased
                    },
                    "$set": {"last_updated": timestamp},
                    "$setOnInsert": {"created_at": timestamp}
                },
                upsert=True
            )

        # 4. Update user summary
        farm = self.get_farm(farm_id)
        user_id = farm.get("user_id") if farm else None

        if user_id:
            self.update_user_summary(user_id)

    def update_farm_nearby(self, farm_id: str, nearby_farms: List[Dict]):
        """Update nearby farms for a farm"""
        self.farms.update_one(
            {"farm_id": farm_id},
            {"$set": {"nearby_farms": nearby_farms}}
        )

    # ==================== DISEASE REPORTS ====================

    def save_disease_report(self, report_data: Dict) -> str:
        """Save a disease report"""
        report_data["timestamp"] = report_data.get("timestamp", datetime.utcnow())
        result = self.disease_reports.insert_one(report_data)
        return str(result.inserted_id)

    def get_disease_reports(self, farm_id: str = None, days: int = 30,
                           disease: str = None, limit: int = 100) -> List[Dict]:
        """Get disease reports with filters"""
        query = {}

        if farm_id:
            query["farm_id"] = farm_id

        if days:
            cutoff = datetime.utcnow() - timedelta(days=days)
            query["timestamp"] = {"$gte": cutoff}

        if disease:
            query["disease"] = disease

        cursor = self.disease_reports.find(query).sort("timestamp", DESCENDING).limit(limit)
        return list(cursor)

    def get_disease_reports_nearby(self, lat: float, lon: float, radius_km: float = 5,
                                   days: int = 10) -> List[Dict]:
        """Get disease reports within radius"""
        from geospatial.haversine import HaversineDistance

        haversine = HaversineDistance()
        cutoff = datetime.utcnow() - timedelta(days=days)

        # Get all reports in time range
        reports = list(self.disease_reports.find({
            "timestamp": {"$gte": cutoff},
            "latitude": {"$ne": None},
            "longitude": {"$ne": None},
            "disease": {"$ne": "healthy"}
        }))

        # Filter by distance
        nearby = []
        for report in reports:
            dist = haversine.haversine(
                lat, lon,
                float(report["latitude"]),
                float(report["longitude"])
            )
            if dist <= radius_km:
                report["distance_km"] = round(dist, 3)
                nearby.append(report)

        return nearby

    # ==================== STATISTICS & SUMMARY ====================

    def update_farm_stats(self, farm_id: str = None):
        """Update statistics for farm(s)"""
        if farm_id:
            farms = [self.get_farm(farm_id)]
        else:
            farms = self.get_all_farms()

        for farm in farms:
            farm_id = farm["farm_id"]
            today = datetime.utcnow().strftime("%Y-%m-%d")

            reports = list(self.disease_reports.find({"farm_id": farm_id}))
            if not reports:
                continue

            from collections import Counter

            diseases = [r["disease"] for r in reports if r.get("disease") and r["disease"] != "healthy"]
            crops = [r["crop"] for r in reports if r.get("crop")]

            total = len(reports)
            diseased = len(diseases)
            most_common_disease = Counter(diseases).most_common(1)
            most_common_crop = Counter(crops).most_common(1)

            # Get max risk from LSTM predictions
            max_risk = 0
            try:
                farm_preds = farm.get("lstm_prediction", [])
                if farm_preds:
                    max_risk = max(p.get("predicted_risk%", 0) for p in farm_preds)
            except:
                pass

            update_data = {
                "farm_id": farm_id,
                "date": today,
                "last_updated": datetime.utcnow(),
                "total_images_analyzed": total,
                "diseased_images_found": diseased,
                "crop_counts": dict(Counter(crops)),
                "disease_counts": dict(Counter(diseases)),
                "max_risk_percent": round(max_risk, 2),
                "most_common_disease": most_common_disease[0][0] if most_common_disease else None,
                "most_common_crop": most_common_crop[0][0] if most_common_crop else None,
            }

            self.farm_stats.update_one(
                {"farm_id": farm_id, "date": today},
                {"$set": update_data},
                upsert=True
            )

    def update_user_summary(self, user_id: str = None):
        """Update user summary statistics"""
        if user_id:
            farms = self.get_farm_by_user(user_id)
            if not farms:
                return
            users = [(user_id, farms)]
        else:
            # Group farms by user
            from collections import defaultdict
            user_farms = defaultdict(list)
            for farm in self.get_all_farms():
                user_farms[farm.get("user_id", "unknown")].append(farm)
            users = list(user_farms.items())

        for user_id, user_farms in users:
            total_images = 0
            diseased_images = 0
            all_diseases = []
            max_risk = 0

            for farm in user_farms:
                farm_id = farm["farm_id"]
                reports = list(self.disease_reports.find({"farm_id": farm_id}))
                total_images += len(reports)

                diseases_here = [r["disease"] for r in reports if r.get("disease") != "healthy"]
                diseased_images += len(diseases_here)
                all_diseases.extend(diseases_here)

                try:
                    if farm.get("lstm_prediction"):
                        farm_max = max(p.get("predicted_risk%", 0) for p in farm["lstm_prediction"])
                        max_risk = max(max_risk, farm_max)
                except:
                    pass

            from collections import Counter
            top_diseases = dict(Counter(all_diseases).most_common(5))

            update_data = {
                "user_id": user_id,
                "summary": {
                    "total_images": total_images,
                    "total_diseased": diseased_images,
                    "last_updated": datetime.utcnow(),
                    "max_risk_percent": round(max_risk, 2),
                    "top_diseases": top_diseases
                },
                "updated_at": datetime.utcnow()
            }

            self.user_summary.update_one(
                {"user_id": user_id},
                {"$set": update_data},
                upsert=True
            )

    def update_all_stats(self):
        """Update all statistics"""
        self.update_farm_stats()
        self.update_user_summary()
        logger.info("All statistics updated")

    # ==================== CROP TRAINING DATA ====================

    def save_crop_training_data(self, df, farm_id: str, crop: str):
        """Save crop training data to MongoDB"""
        import pandas as pd

        # Convert DataFrame to dict for storage
        records = df.to_dict(orient="records")

        # Delete old data for this farm/crop
        self.crop_training_data.delete_many({"farm_id": farm_id, "crop": crop})

        # Insert new data
        if records:
            for record in records:
                record["farm_id"] = farm_id
                record["crop"] = crop
                if "start_date" in record and record["start_date"]:
                    record["start_date"] = pd.to_datetime(record["start_date"]).isoformat()
                if "end_date" in record and record["end_date"]:
                    record["end_date"] = pd.to_datetime(record["end_date"]).isoformat()

            self.crop_training_data.insert_many(records)
            logger.info(f"Saved {len(records)} training records for {farm_id}/{crop}")

    def load_crop_training_data(self, farm_id: str, crop: str):
        """Load crop training data from MongoDB"""
        import pandas as pd

        records = list(self.crop_training_data.find(
            {"farm_id": farm_id, "crop": crop},
            {"_id": 0}
        ))

        if not records:
            return None

        df = pd.DataFrame(records)

        # Convert date columns
        if "start_date" in df.columns:
            df["start_date"] = pd.to_datetime(df["start_date"])
        if "end_date" in df.columns:
            df["end_date"] = pd.to_datetime(df["end_date"])

        return df

    # ==================== STATE YIELD MODELS ====================

    def save_state_yield_prediction(self, state: str, crop: str, season: str,
                                    version: str, data: Dict):
        """Save state yield prediction"""
        doc = {
            "state": state,
            "crop": crop,
            "season": season,
            "version": version,
            "data": data,
            "timestamp": datetime.utcnow()
        }
        self.state_yield_all.update_one(
            {"state": state, "crop": crop, "season": season, "version": version},
            {"$set": doc},
            upsert=True
        )

    def get_state_yield_prediction(self, state: str, crop: str, season: str) -> Optional[Dict]:
        """Get latest state yield prediction"""
        return self.state_yield_all.find_one(
            {"state": state, "crop": crop, "season": season},
            sort=[("timestamp", DESCENDING)]
        )

    # ==================== YIELD MODEL METADATA ====================

    def save_yield_model_metadata(self, metadata: Dict) -> bool:
        """Save yield model metadata"""
        try:
            metadata["saved_at"] = datetime.utcnow()
            self.yield_models.insert_one(metadata)
            return True
        except Exception as e:
            logger.error(f"Failed to save model metadata: {e}")
            return False

    def get_yield_model_metadata(self, farm_id: str, crop: str) -> Optional[Dict]:
        """Get yield model metadata"""
        return self.yield_models.find_one(
            {"farm_id": farm_id, "crop": crop},
            sort=[("saved_at", DESCENDING)]
        )

    def save_yield_prediction(self, prediction_doc: Dict) -> bool:
        """Save yield prediction"""
        try:
            if "timestamp" in prediction_doc and isinstance(prediction_doc["timestamp"], str):
                prediction_doc["timestamp"] = datetime.fromisoformat(prediction_doc["timestamp"])
            prediction_doc["created_at"] = datetime.utcnow()
            self.yield_predictions.insert_one(prediction_doc)
            return True
        except Exception as e:
            logger.error(f"Failed to save prediction: {e}")
            return False

    def get_latest_yield_prediction(self, farm_id: str, crop: str) -> Optional[Dict]:
        """Get latest yield prediction"""
        return self.yield_predictions.find_one(
            {"farm_id": farm_id, "crop": crop},
            sort=[("timestamp", DESCENDING)]
        )

    def save_simulation(self, simulation_doc: Dict) -> bool:
        """Save what-if simulation"""
        try:
            simulation_doc["created_at"] = datetime.utcnow()
            self.what_if_simulations.insert_one(simulation_doc)
            return True
        except Exception as e:
            logger.error(f"Failed to save simulation: {e}")
            return False

    # ==================== LLM & CHAT ====================

    def save_chat_message(self, session_id: str, query: str, response: str,
                          context_types: List[str], metadata: Dict = None):
        """Save chat message to history"""
        doc = {
            "session_id": session_id,
            "query": query,
            "response": response,
            "context_types": context_types,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow()
        }
        self.chat_history.insert_one(doc)

    def get_chat_history(self, session_id: str, limit: int = 15) -> List[Dict]:
        """Get chat history for session"""
        cursor = self.chat_history.find(
            {"session_id": session_id}
        ).sort("timestamp", DESCENDING).limit(limit)
        return list(cursor)

    def clear_chat_history(self, session_id: str) -> int:
        """Clear chat history for session"""
        result = self.chat_history.delete_many({"session_id": session_id})
        return result.deleted_count

    def save_news_item(self, news_item: Dict) -> str:
        """Save news item"""
        news_item["fetched_at"] = datetime.utcnow()
        result = self.stored_news.insert_one(news_item)
        return str(result.inserted_id)

    def get_news_items(self, limit: int = 50) -> List[Dict]:
        """Get recent news items"""
        cursor = self.stored_news.find().sort("fetched_at", DESCENDING).limit(limit)
        return list(cursor)

    def save_weather_data(self, weather_data: Dict, location: str) -> str:
        """Save weather data"""
        doc = {
            "location": location,
            "data": weather_data,
            "fetched_at": datetime.utcnow()
        }
        result = self.weather_data.insert_one(doc)
        return str(result.inserted_id)

    def get_weather_data(self, location: str, hours: int = 24) -> Optional[Dict]:
        """Get recent weather data"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return self.weather_data.find_one(
            {"location": location, "fetched_at": {"$gte": cutoff}},
            sort=[("fetched_at", DESCENDING)]
        )

    def save_bulletin(self, bulletin: Dict) -> str:
        """Save bulletin"""
        bulletin["fetched_at"] = datetime.utcnow()
        result = self.bulletins.insert_one(bulletin)
        return str(result.inserted_id)

    def get_bulletins(self, state: str = None, limit: int = 20) -> List[Dict]:
        """Get bulletins"""
        query = {}
        if state:
            query["state"] = state
        cursor = self.bulletins.find(query).sort("fetched_at", DESCENDING).limit(limit)
        return list(cursor)

    def create_training_job(self, job_data: Dict) -> str:
        """Create training job record"""
        job_data["created_at"] = datetime.utcnow()
        job_data["status"] = "pending"
        result = self.training_jobs.insert_one(job_data)
        return str(result.inserted_id)

    def update_training_job(self, job_id: str, status: str, result: Dict = None):
        """Update training job status"""
        update = {"status": status, "updated_at": datetime.utcnow()}
        if result:
            update["result"] = result
        self.training_jobs.update_one(
            {"_id": ObjectId(job_id)},
            {"$set": update}
        )

    def get_training_job(self, job_id: str) -> Optional[Dict]:
        """Get training job by ID"""
        return self.training_jobs.find_one({"_id": ObjectId(job_id)})

    def update_training_status(self, job_type: str, status: str, metadata: Dict = None):
        """Update training status"""
        self.training_status.update_one(
            {"job_type": job_type},
            {
                "$set": {
                    "status": status,
                    "updated_at": datetime.utcnow(),
                    "metadata": metadata or {}
                },
                "$setOnInsert": {"created_at": datetime.utcnow()}
            },
            upsert=True
        )

    def get_training_status(self, job_type: str) -> Optional[Dict]:
        """Get training status"""
        return self.training_status.find_one({"job_type": job_type})

    def save_uploaded_file_metadata(self, filename: str, index_type: str,
                                   file_size: int, file_path: str = None) -> str:
        """Save uploaded file metadata"""
        doc = {
            "filename": filename,
            "index_type": index_type,
            "size": file_size,
            "file_path": file_path,
            "uploaded_at": datetime.utcnow()
        }
        result = self.uploaded_files.insert_one(doc)
        return str(result.inserted_id)

    def get_uploaded_files(self, index_type: str = None) -> List[Dict]:
        """Get uploaded files list"""
        query = {}
        if index_type:
            query["index_type"] = index_type
        cursor = self.uploaded_files.find(query).sort("uploaded_at", DESCENDING)
        return list(cursor)

    def delete_uploaded_file_metadata(self, filename: str, index_type: str) -> int:
        """Delete uploaded file metadata"""
        result = self.uploaded_files.delete_many({
            "filename": filename,
            "index_type": index_type
        })
        return result.deleted_count


# Global instance
mongo_storage = MongoStorage()
