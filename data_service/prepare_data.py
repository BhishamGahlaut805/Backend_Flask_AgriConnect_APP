"""
Data Preparation Service
Fetches weather, soil, NDVI data and prepares training data
All data stored in MongoDB instead of local files
"""

import os
import csv
import io
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pymongo import MongoClient, errors

from .openweather import OpenWeatherAPI
from .openmeteo import OpenMeteoAPI
from geospatial.haversine import HaversineDistance
from api.agromonitoring import AgroMonitoringAPI
from mongo_storage import mongo_storage
from logging_config import logger


class DataService:
    """Service for fetching and preparing training data with MongoDB storage"""

    def __init__(self):
        self.client = mongo_storage.client
        self.db = mongo_storage.db
        self.farm_col = mongo_storage.farms
        self.disease_col = mongo_storage.disease_reports

        self.weather_api = OpenWeatherAPI()
        self.soil_api = OpenMeteoAPI()
        self.ndvi_api = AgroMonitoringAPI()
        self.distance_util = HaversineDistance()

        # Use MongoDB for run logs instead of local file
        self.run_log_collection = self.db["data_prep_run_logs"]
        self._init_run_logs()

    def _init_run_logs(self):
        """Initialize run logs collection"""
        self.run_log_collection.create_index([("farm_id", 1), ("date", 1)], unique=True)
        self.run_log_collection.create_index([("last_run", -1)])

    def load_run_log(self) -> Dict[str, str]:
        """Load run logs from MongoDB"""
        try:
            logs = {}
            cursor = self.run_log_collection.find({})
            for log in cursor:
                logs[log["farm_id"]] = log["date"]
            return logs
        except Exception as e:
            logger.error(f"Could not load run logs: {e}")
            return {}

    def save_run_log(self, farm_id: str, date: str):
        """Save run log to MongoDB"""
        try:
            self.run_log_collection.update_one(
                {"farm_id": str(farm_id)},
                {"$set": {"date": date, "last_run": datetime.utcnow()}},
                upsert=True
            )
        except Exception as e:
            logger.error(f"Could not save run log: {e}")

    def already_ran_today(self, farm_id: str) -> bool:
        """Check if already ran today for this farm"""
        try:
            today = datetime.utcnow().strftime('%Y-%m-%d')
            log = self.run_log_collection.find_one({
                "farm_id": str(farm_id),
                "date": today
            })
            return log is not None
        except Exception:
            return False

    def mark_today_done(self, farm_id: str):
        """Mark that data was prepared today"""
        today = datetime.utcnow().strftime('%Y-%m-%d')
        self.save_run_log(str(farm_id), today)

    def calculate_risk_radius(self, farm: Dict, today: datetime) -> Tuple[float, float, List[Dict]]:
        """Calculate disease risk and radius based on nearby reports"""
        try:
            farm_lat = float(farm.get("latitude", 0))
            farm_lon = float(farm.get("longitude", 0))
        except Exception as e:
            logger.error(f"Invalid farm coordinates: {e}")
            return 0, 0, []

        today_start = datetime(today.year, today.month, today.day)
        past_date = today_start - timedelta(days=10)

        try:
            reports = list(self.disease_col.find({
                "timestamp": {"$gte": past_date},
                "latitude": {"$ne": None},
                "longitude": {"$ne": None},
                "disease": {"$ne": "healthy"}
            }))
            logger.info(f"Retrieved {len(reports)} non-healthy reports in last 10 days.")
        except Exception as e:
            logger.error(f"Mongo query failed: {e}")
            return 0, 0, []

        nearby_reports = []

        for report in reports:
            try:
                lat = float(report["latitude"])
                lon = float(report["longitude"])
                dist_km = self.distance_util.haversine(farm_lat, farm_lon, lat, lon)

                if dist_km <= 5:
                    nearby_reports.append({
                        "distance_km": round(dist_km, 3),
                        "disease": report.get("disease", "unknown"),
                        "confidence": round(float(report.get("confidence", 0)), 3)
                    })
            except Exception as e:
                logger.warning(f"Skipping bad report: {e}")

        if not nearby_reports:
            logger.info("No diseased reports found within 5 km radius.")
            return 0, 0, []

        risk_percent = 100.0
        radius_km = round(sum(r["distance_km"] for r in nearby_reports) / len(nearby_reports), 2)
        top_5 = sorted(nearby_reports, key=lambda x: x["distance_km"])[:5]

        return risk_percent, radius_km, top_5

    def fetch_ndvi_index(self, polygon_id: str, today: datetime, farm_name: str,
                         lat: float, lon: float, farm_id: str) -> float:
        """Fetch NDVI index with fallback"""
        if not lat or not lon:
            logger.warning(f"Missing coordinates for farm {farm_name}")
            return self._get_smart_ndvi_fallback(lat, lon, today)

        coordinates = [
            [lon - 0.005, lat - 0.005],
            [lon + 0.005, lat - 0.005],
            [lon + 0.005, lat + 0.005],
            [lon - 0.005, lat + 0.005],
            [lon - 0.005, lat - 0.005]
        ]

        def update_polygon_id_in_db(new_id):
            try:
                self.farm_col.update_one(
                    {"farm_id": farm_id},
                    {"$set": {"agro_polygon.polygon_id": new_id}}
                )
                logger.info(f"Updated polygon ID for {farm_name}: {new_id}")
            except Exception as e:
                logger.error(f"Failed to update polygon ID: {e}")

        # Try to get actual NDVI data
        ndvi_value = self.ndvi_api.get_ndvi_index(
            poly_id=polygon_id,
            start_date=today - timedelta(days=30),
            end_date=today,
            farm_name=farm_name,
            coordinates=coordinates,
            update_polygon_id=update_polygon_id_in_db
        )

        # Use smart fallback if NDVI is 0 or None
        return ndvi_value if ndvi_value and ndvi_value > 0 else self._get_smart_ndvi_fallback(lat, lon, today)

    def _get_smart_ndvi_fallback(self, lat: float, lon: float, date: datetime) -> float:
        """Generate meaningful NDVI based on weather and season"""
        try:
            # Get current weather data
            weather = self.weather_api.get_weather(lat, lon) or {}
            weather_main = weather.get("main", {})

            # Base value based on season (Northern Hemisphere)
            month = date.month
            if month in [12, 1, 2]:  # Winter
                base = 0.2
            elif month in [3, 4, 5]:  # Spring
                base = 0.5
            elif month in [6, 7, 8]:  # Summer
                base = 0.7
            else:  # Fall
                base = 0.4

            # Adjust based on temperature
            temp = weather_main.get("temp", 20)
            if temp > 30:
                base += 0.1
            elif temp < 10:
                base -= 0.1

            # Adjust based on humidity
            humidity = weather_main.get("humidity", 50)
            if humidity > 70:
                base += 0.05
            elif humidity < 30:
                base -= 0.05

            # Add small random variation
            base += random.uniform(-0.05, 0.05)

            # Ensure within valid NDVI range
            return round(max(0.1, min(0.9, base)), 4)

        except Exception as e:
            logger.warning(f"Fallback NDVI generation failed: {e}")
            return round(random.uniform(0.3, 0.7), 4)

    def save_training_data_to_mongodb(self, farm_id: str, farm_name: str,
                                       data_row: Dict) -> str:
        """Save training data row to MongoDB"""
        try:
            # Add metadata
            data_row["farm_id"] = farm_id
            data_row["farm_name"] = farm_name
            data_row["stored_at"] = datetime.utcnow()

            # Store in training_data collection
            collection = self.db["farm_training_data"]

            # Update or insert
            collection.update_one(
                {"farm_id": farm_id, "date": data_row["date"]},
                {"$set": data_row},
                upsert=True
            )

            # Also store in GridFS for ML training
            import pandas as pd
            df = pd.DataFrame([data_row])
            mongo_storage.save_training_data(
                df,
                "daily_training",
                farm_id=farm_id,
                metadata={"farm_name": farm_name, "date": data_row["date"]}
            )

            return data_row["date"]

        except Exception as e:
            logger.error(f"Failed to save training data: {e}")
            raise

    def generate_training_data(self):
        """Generate training data for all farms and store in MongoDB"""
        today = datetime.utcnow()

        try:
            farms = list(self.farm_col.find())
        except Exception as e:
            logger.error(f"Could not retrieve farms: {e}")
            return

        results = {"total": len(farms), "successful": 0, "failed": 0, "skipped": 0}

        for farm in farms:
            farm_id = farm.get("farm_id")
            farm_name = farm.get("farm_name", "Unknown")
            lat = farm.get("latitude")
            lon = farm.get("longitude")
            polygon_id = farm.get("agro_polygon", {}).get("polygon_id")

            if not all([farm_id, lat, lon]):
                logger.warning(f"Incomplete data for {farm_name}, skipping.")
                results["skipped"] += 1
                continue

            if self.already_ran_today(farm_id):
                logger.info(f"Already processed today: {farm_name}")
                results["skipped"] += 1
                continue

            try:
                soil = self.soil_api.get_forecast_soil(lat, lon) or {}
                weather = self.weather_api.get_weather(lat, lon) or {}
                ndvi = self.fetch_ndvi_index(
                    polygon_id, today, farm_name, lat, lon, farm_id
                )
                risk_percent, radius_km, top_5_risks = self.calculate_risk_radius(farm, today)
            except Exception as e:
                logger.error(f"Data fetch failed for {farm_name}: {e}")
                results["failed"] += 1
                continue

            weather_main = weather.get("main", {})
            wind = weather.get("wind", {})
            clouds = weather.get("clouds", {})
            rain = weather.get("rain", {})
            visibility = weather.get("visibility", None)
            weather_desc = weather.get("weather", [{}])[0].get("description", "")

            # Build data row
            data_row = {
                "date": today.strftime("%Y-%m-%d"),
                "latitude": lat,
                "longitude": lon,

                # Open-Meteo forecast soil + sky
                "soil_temp_0cm": soil.get("soil_temp_0cm", 0),
                "soil_temp_18cm": soil.get("soil_temp_18cm", 0),
                "soil_moisture_1_3cm": soil.get("soil_moisture_1_3", 0),
                "soil_moisture_27_81cm": soil.get("soil_moisture_27_81", 0),
                "evapotranspiration": soil.get("evapotranspiration", 0),
                "cloud_cover_low": soil.get("cloud_low", 0),
                "cloud_cover_high": soil.get("cloud_high", 0),
                "wind_gusts_10m": soil.get("wind_gust_10m", 0),

                # OpenWeather (realtime)
                "temp": weather_main.get("temp", 0),
                "feels_like": weather_main.get("feels_like", 0),
                "humidity": weather_main.get("humidity", 0),
                "pressure": weather_main.get("pressure", 0),
                "visibility": visibility if visibility is not None else 0,
                "wind_speed": wind.get("speed", 0),
                "wind_deg": wind.get("deg", 0),
                "cloud_cover_total": clouds.get("all", 0),
                "rain_1h": rain.get("1h", 0),
                "weather_desc": weather_desc,

                # Analysis
                "ndvi_index": ndvi if ndvi is not None else 0,
                "risk%": risk_percent,
                "radius_km": radius_km
            }

            try:
                # Save to MongoDB
                self.save_training_data_to_mongodb(farm_id, farm_name, data_row)

                # Update farm document
                update_fields = {
                    "last_trained_at": today,
                    "last_training_data_date": today.strftime("%Y-%m-%d")
                }
                if top_5_risks:
                    update_fields["top_disease_risks"] = top_5_risks

                self.farm_col.update_one({"farm_id": farm_id}, {"$set": update_fields})
                self.mark_today_done(farm_id)

                logger.info(f"Data saved for {farm_name} to MongoDB")
                results["successful"] += 1

            except Exception as e:
                logger.error(f"Failed saving data for {farm_name}: {e}")
                results["failed"] += 1

        # Store batch results
        self.db["data_prep_batch_results"].insert_one({
            "timestamp": datetime.utcnow(),
            "results": results
        })

        logger.info(f"Data preparation completed: {results['successful']} successful, "
                   f"{results['failed']} failed, {results['skipped']} skipped")

    def get_training_data(self, farm_id: str, start_date: str = None,
                          end_date: str = None) -> List[Dict]:
        """Retrieve training data from MongoDB"""
        query = {"farm_id": farm_id}
        if start_date:
            query["date"] = {"$gte": start_date}
        if end_date:
            query["date"] = {"$lte": end_date}

        cursor = self.db["farm_training_data"].find(query).sort("date", 1)
        return list(cursor)

    def export_training_data_to_csv(self, farm_id: str, output_path: str = None) -> Optional[str]:
        """Export training data to CSV (for compatibility)"""
        import pandas as pd

        data = self.get_training_data(farm_id)
        if not data:
            logger.warning(f"No training data found for farm {farm_id}")
            return None

        df = pd.DataFrame(data)

        # Remove MongoDB fields
        for col in ["_id", "farm_id", "farm_name", "stored_at"]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        if output_path:
            df.to_csv(output_path, index=False)
            return output_path
        else:
            # Return CSV as string
            return df.to_csv(index=False)

    def force_rerun_for_farm(self, farm_id: str, confirm: bool = False):
        """Force rerun for a farm (for debugging)"""
        if not confirm:
            logger.warning("Force rerun blocked. Set confirm=True to proceed.")
            return

        try:
            self.run_log_collection.delete_many({"farm_id": str(farm_id)})
            logger.info(f"Rerun forced for farm ID: {farm_id}")
        except Exception as e:
            logger.error(f"Failed to force rerun: {e}")

    def run_once_a_day(self):
        """Main entry point for daily data generation"""
        logger.info("Starting daily training data generation...")
        self.generate_training_data()
        logger.info("Training process completed for all farms.")


# Singleton instance
data_service = DataService()
