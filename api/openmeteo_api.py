"""
OpenMeteo API Integration with MongoDB Storage
Fetches and processes weather data from NASA POWER and Open-Meteo APIs
Stores all data in MongoDB instead of local files
"""

import os
import requests
import pandas as pd
import io
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any
from mongo_storage import mongo_storage
from logging_config import logger


class WeatherDataProcessor:
    """Weather data processor with MongoDB storage backend"""

    def __init__(self, latitude: float, longitude: float, farm_id: str, farm_name: str):
        self.latitude = latitude
        self.longitude = longitude
        self.farm_id = farm_id
        self.farm_name = farm_name
        self.timezone = "auto"

        # MongoDB storage identifiers
        self.weather_collection = "weather_data"
        self.nasa_data_id = f"nasa_{farm_id}"
        self.final_weather_id = f"weather_yield_{farm_id}"
        self.log_id = f"weather_log_{farm_id}"

        # API URLs
        self.nasa_power_url = os.getenv("NASA_POWER_URL") or "https://power.larc.nasa.gov/api/temporal/daily/point"
        self.openmeteo_url = os.getenv("OPENMETEO_URL") or "https://api.open-meteo.com/v1/forecast"

        self.column_mappings = {
            "T2M": "temperature_2m_mean",
            "T2M_MAX": "temperature_2m_max",
            "T2M_MIN": "temperature_2m_min",
            "RH2M": "relative_humidity_2m_mean",
            "WS2M": "wind_speed_10m_max",
            "WD2M": "wind_direction_10m_dominant",
            "PRECTOTCORR": "precipitation_sum",
            "ALLSKY_SFC_SW_DWN": "shortwave_radiation_sum",
            "PS": "surface_pressure_mean",
            "CLOUD_AMT": "cloud_cover_mean",
        }

        self.common_columns = [
            'date', 'temperature_2m_mean', 'temperature_2m_max', 'temperature_2m_min',
            'relative_humidity_2m_mean', 'wind_speed_10m_max', 'wind_direction_10m_dominant',
            'precipitation_sum', 'shortwave_radiation_sum', 'surface_pressure_mean',
            'cloud_cover_mean', 'data_source'
        ]

        # Initialize storage
        self._init_storage()

    def _init_storage(self):
        """Initialize MongoDB storage for this farm"""
        # Ensure weather data collection exists with indexes
        if self.weather_collection not in mongo_storage.db.list_collection_names():
            mongo_storage.db.create_collection(self.weather_collection)

        # Create indexes for efficient queries
        mongo_storage.db[self.weather_collection].create_index([
            ("farm_id", 1),
            ("data_type", 1),
            ("date", 1)
        ], unique=True)

    def _save_dataframe_to_mongodb(self, df: pd.DataFrame, data_type: str,
                                   metadata: Dict = None) -> str:
        """Save DataFrame to MongoDB as records"""
        if df.empty:
            logger.warning(f"No data to save for {data_type}")
            return None

        # Convert DataFrame to records
        records = df.to_dict(orient="records")

        # Add metadata to each record
        for record in records:
            record["farm_id"] = self.farm_id
            record["farm_name"] = self.farm_name
            record["data_type"] = data_type
            record["latitude"] = self.latitude
            record["longitude"] = self.longitude
            record["stored_at"] = datetime.utcnow()

            # Convert date to string for MongoDB
            if "date" in record and isinstance(record["date"], pd.Timestamp):
                record["date"] = record["date"].isoformat()

        # Delete old data of same type (upsert with date-based overwrite)
        for record in records:
            mongo_storage.db[self.weather_collection].update_one(
                {
                    "farm_id": self.farm_id,
                    "data_type": data_type,
                    "date": record["date"]
                },
                {"$set": record},
                upsert=True
            )

        # Save raw data as training data for ML
        if data_type in ["nasa_power", "final_weather"]:
            mongo_storage.save_training_data(
                df,
                f"weather_{data_type}",
                farm_id=self.farm_id,
                crop=None,
                metadata={
                    "farm_name": self.farm_name,
                    "latitude": self.latitude,
                    "longitude": self.longitude,
                    **(metadata or {})
                }
            )

        logger.info(f"Saved {len(records)} records for {data_type} to MongoDB")
        return f"{self.farm_id}_{data_type}"

    def _load_dataframe_from_mongodb(self, data_type: str) -> Optional[pd.DataFrame]:
        """Load DataFrame from MongoDB"""
        cursor = mongo_storage.db[self.weather_collection].find({
            "farm_id": self.farm_id,
            "data_type": data_type
        })

        records = list(cursor)
        if not records:
            return None

        df = pd.DataFrame(records)

        # Convert date back to datetime
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        # Remove MongoDB-specific fields
        for col in ["_id", "farm_id", "farm_name", "data_type", "stored_at"]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        return df

    def process_nasa_power_data(self, raw_data: Dict) -> pd.DataFrame:
        """Process NASA POWER API response into DataFrame"""
        if not raw_data or "properties" not in raw_data:
            return pd.DataFrame()

        parameters = raw_data["properties"]["parameter"]
        dfs = []

        for param, values in parameters.items():
            if param in self.column_mappings:
                df = pd.DataFrame.from_dict(values, orient='index', columns=[param])
                dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        nasa_df = pd.concat(dfs, axis=1)
        nasa_df.index = pd.to_datetime(nasa_df.index)
        nasa_df.index.name = 'date'
        nasa_df.rename(columns=self.column_mappings, inplace=True)
        nasa_df['data_source'] = 'NASA_POWER'

        return nasa_df.reset_index()

    def process_openmeteo_data(self, raw_data: Dict, data_type: str) -> pd.DataFrame:
        """Process Open-Meteo API response into DataFrame"""
        if not raw_data or "daily" not in raw_data:
            return pd.DataFrame()

        daily_data = raw_data["daily"]

        om_df = pd.DataFrame({
            'date': pd.to_datetime(daily_data.get('time', [])),
            'temperature_2m_max': daily_data.get('temperature_2m_max', []),
            'temperature_2m_min': daily_data.get('temperature_2m_min', []),
            'temperature_2m_mean': daily_data.get('temperature_2m_mean', []),
            'relative_humidity_2m_mean': daily_data.get('relative_humidity_2m_mean', []),
            'wind_speed_10m_max': daily_data.get('wind_speed_10m_max', []),
            'wind_direction_10m_dominant': daily_data.get('wind_direction_10m_dominant', []),
            'precipitation_sum': daily_data.get('precipitation_sum', []),
            'shortwave_radiation_sum': daily_data.get('shortwave_radiation_sum', []),
            'surface_pressure_mean': daily_data.get('surface_pressure_mean', []),
            'cloud_cover_mean': daily_data.get('cloud_cover_mean', [])
        })

        om_df['data_source'] = f'OpenMeteo_{data_type}'
        return om_df

    def unify_dataframes(self, nasa_df: pd.DataFrame, om_hist_df: pd.DataFrame,
                        om_fcst_df: pd.DataFrame) -> pd.DataFrame:
        """Unify data from multiple sources"""
        combined_df = pd.concat([nasa_df, om_hist_df, om_fcst_df], ignore_index=True)

        for col in self.common_columns:
            if col not in combined_df.columns and col != 'date':
                combined_df[col] = None

        combined_df = combined_df[self.common_columns]
        combined_df.drop_duplicates(subset=['date'], keep='last', inplace=True)
        combined_df.sort_values('date', inplace=True)

        return combined_df

    def fetch_and_process_weather_data(self) -> pd.DataFrame:
        """
        Fetch weather data from NASA POWER and Open-Meteo APIs
        Stores data in MongoDB instead of local files
        """
        today = datetime.now().date()
        start_date = datetime(2023, 1, 1).date()

        logger.info(f"Fetching weather data for farm {self.farm_name} ({self.farm_id})")

        # Fetch NASA POWER data
        nasa_df = self._fetch_nasa_data(start_date, today)

        # Fetch Open-Meteo data
        om_hist_df = self._fetch_openmeteo_historical(start_date, today)
        om_fcst_df = self._fetch_openmeteo_forecast(today)

        # Combine all data
        final_df = self.unify_dataframes(nasa_df, om_hist_df, om_fcst_df)

        if not final_df.empty:
            final_df['date'] = pd.to_datetime(final_df['date'])
            final_df = final_df[(final_df['date'].dt.year >= 2023)]

            # Save to MongoDB
            self._save_dataframe_to_mongodb(
                final_df,
                "final_weather",
                metadata={
                    "start_date": start_date.isoformat(),
                    "end_date": today.isoformat(),
                    "has_nasa": not nasa_df.empty,
                    "has_openmeteo_hist": not om_hist_df.empty,
                    "has_openmeteo_forecast": not om_fcst_df.empty
                }
            )

            logger.info(f"Final weather data saved to MongoDB: {len(final_df)} records")
        else:
            logger.warning(f"No weather data fetched for farm {self.farm_name}")

        return final_df

    def _fetch_nasa_data(self, start_date, end_date) -> pd.DataFrame:
        """Fetch NASA POWER data"""
        logger.info("Fetching NASA POWER data...")

        try:
            nasa_params = {
                "start": start_date.strftime("%Y%m%d"),
                "end": (end_date - timedelta(days=1)).strftime("%Y%m%d"),
                "latitude": self.latitude,
                "longitude": self.longitude,
                "parameters": ",".join(self.column_mappings.keys()),
                "format": "JSON",
                "community": "AG"
            }

            response = requests.get(self.nasa_power_url, params=nasa_params, timeout=60)
            response.raise_for_status()

            nasa_df = self.process_nasa_power_data(response.json())

            if not nasa_df.empty:
                # Save to MongoDB
                self._save_dataframe_to_mongodb(
                    nasa_df,
                    "nasa_power",
                    metadata={"source": "NASA_POWER"}
                )
                logger.info(f"NASA data saved: {len(nasa_df)} records")
            else:
                logger.warning("NASA data is empty")

            return nasa_df

        except Exception as e:
            logger.error(f"Error fetching NASA data: {e}")
            return pd.DataFrame()

    def _fetch_openmeteo_historical(self, start_date, today) -> pd.DataFrame:
        """Fetch Open-Meteo historical data"""
        logger.info("Fetching Open-Meteo historical data...")

        try:
            om_hist_start = max(start_date, today - timedelta(days=60))
            om_hist_end = today - timedelta(days=1)

            if om_hist_start > om_hist_end:
                logger.info("No historical data needed")
                return pd.DataFrame()

            om_hist_params = {
                "latitude": self.latitude,
                "longitude": self.longitude,
                "start_date": om_hist_start.strftime("%Y-%m-%d"),
                "end_date": om_hist_end.strftime("%Y-%m-%d"),
                "daily": ",".join([
                    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
                    "relative_humidity_2m_mean", "wind_speed_10m_max",
                    "wind_direction_10m_dominant", "precipitation_sum",
                    "shortwave_radiation_sum", "surface_pressure_mean",
                    "cloud_cover_mean"
                ]),
                "timezone": self.timezone
            }

            response = requests.get(self.openmeteo_url, params=om_hist_params, timeout=60)
            response.raise_for_status()

            om_hist_df = self.process_openmeteo_data(response.json(), "Historical")

            if not om_hist_df.empty:
                self._save_dataframe_to_mongodb(
                    om_hist_df,
                    "openmeteo_historical",
                    metadata={"source": "OpenMeteo_Historical"}
                )
                logger.info(f"Open-Meteo historical data saved: {len(om_hist_df)} records")

            return om_hist_df

        except Exception as e:
            logger.error(f"Error fetching Open-Meteo historical data: {e}")
            return pd.DataFrame()

    def _fetch_openmeteo_forecast(self, today) -> pd.DataFrame:
        """Fetch Open-Meteo forecast data"""
        logger.info("Fetching Open-Meteo forecast data...")

        try:
            om_fcst_params = {
                "latitude": self.latitude,
                "longitude": self.longitude,
                "start_date": today.strftime("%Y-%m-%d"),
                "end_date": (today + timedelta(days=13)).strftime("%Y-%m-%d"),
                "daily": ",".join([
                    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
                    "relative_humidity_2m_mean", "wind_speed_10m_max",
                    "wind_direction_10m_dominant", "precipitation_sum",
                    "shortwave_radiation_sum", "surface_pressure_mean",
                    "cloud_cover_mean"
                ]),
                "timezone": self.timezone
            }

            response = requests.get(self.openmeteo_url, params=om_fcst_params, timeout=60)
            response.raise_for_status()

            om_fcst_df = self.process_openmeteo_data(response.json(), "Forecast")

            if not om_fcst_df.empty:
                self._save_dataframe_to_mongodb(
                    om_fcst_df,
                    "openmeteo_forecast",
                    metadata={"source": "OpenMeteo_Forecast"}
                )
                logger.info(f"Open-Meteo forecast data saved: {len(om_fcst_df)} records")

            return om_fcst_df

        except Exception as e:
            logger.error(f"Error fetching Open-Meteo forecast data: {e}")
            return pd.DataFrame()

    def get_saved_weather_data(self, data_type: str = "final_weather") -> Optional[pd.DataFrame]:
        """Retrieve saved weather data from MongoDB"""
        return self._load_dataframe_from_mongodb(data_type)

    def get_weather_for_date_range(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Get weather data for specific date range"""
        cursor = mongo_storage.db[self.weather_collection].find({
            "farm_id": self.farm_id,
            "data_type": "final_weather",
            "date": {"$gte": start_date, "$lte": end_date}
        }).sort("date", 1)

        records = list(cursor)
        if not records:
            return None

        df = pd.DataFrame(records)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        # Remove MongoDB fields
        for col in ["_id", "farm_id", "farm_name", "data_type", "stored_at"]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        return df

    def get_latest_weather(self) -> Optional[Dict]:
        """Get the most recent weather data"""
        cursor = mongo_storage.db[self.weather_collection].find({
            "farm_id": self.farm_id,
            "data_type": "final_weather"
        }).sort("date", -1).limit(1)

        records = list(cursor)
        if not records:
            return None

        record = records[0]
        # Remove MongoDB fields
        for col in ["_id", "farm_id", "farm_name", "data_type", "stored_at"]:
            if col in record:
                del record[col]

        return record

    def get_weather_summary(self) -> Dict:
        """Get summary statistics of weather data"""
        df = self.get_saved_weather_data()
        if df is None or df.empty:
            return {"error": "No weather data available"}

        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        summary = {}

        for col in numeric_cols:
            summary[col] = {
                "mean": float(df[col].mean()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "std": float(df[col].std())
            }

        summary["total_records"] = len(df)
        summary["date_range"] = {
            "start": df["date"].min().isoformat() if "date" in df.columns else None,
            "end": df["date"].max().isoformat() if "date" in df.columns else None
        }

        return summary

    def log(self, message: str):
        """Log message to MongoDB"""
        log_entry = {
            "farm_id": self.farm_id,
            "farm_name": self.farm_name,
            "message": message,
            "timestamp": datetime.utcnow(),
            "source": "WeatherDataProcessor"
        }
        mongo_storage.db["weather_logs"].insert_one(log_entry)
        logger.info(message)

    def delete_weather_data(self, data_type: str = None) -> int:
        """Delete weather data for this farm"""
        query = {"farm_id": self.farm_id}
        if data_type:
            query["data_type"] = data_type

        result = mongo_storage.db[self.weather_collection].delete_many(query)
        logger.info(f"Deleted {result.deleted_count} weather records for farm {self.farm_name}")
        return result.deleted_count


# Utility functions for batch processing

def batch_fetch_weather_for_all_farms() -> Dict:
    """
    Fetch weather data for all farms in the database
    Returns summary of results
    """
    from mongo_storage import mongo_storage

    farms = mongo_storage.get_all_farms()
    results = {
        "total_farms": len(farms),
        "successful": 0,
        "failed": 0,
        "details": []
    }

    for farm in farms:
        farm_id = farm.get("farm_id")
        farm_name = farm.get("farm_name")
        latitude = farm.get("latitude")
        longitude = farm.get("longitude")

        if not all([farm_id, latitude, longitude]):
            results["failed"] += 1
            results["details"].append({
                "farm_id": farm_id,
                "farm_name": farm_name,
                "status": "skipped",
                "reason": "Missing coordinates"
            })
            continue

        try:
            processor = WeatherDataProcessor(latitude, longitude, farm_id, farm_name)
            df = processor.fetch_and_process_weather_data()

            results["successful"] += 1
            results["details"].append({
                "farm_id": farm_id,
                "farm_name": farm_name,
                "status": "success",
                "records": len(df) if df is not None else 0
            })

        except Exception as e:
            results["failed"] += 1
            results["details"].append({
                "farm_id": farm_id,
                "farm_name": farm_name,
                "status": "failed",
                "error": str(e)
            })

    # Store batch results
    mongo_storage.db["weather_batch_logs"].insert_one({
        "timestamp": datetime.utcnow(),
        "results": results
    })

    return results

'''
# Example usage
if __name__ == "__main__":
    # Test the processor
    processor = WeatherDataProcessor(
        latitude=28.6139,
        longitude=77.2090,
        farm_id="TEST_FARM_001",
        farm_name="Test Farm Delhi"
    )

    # Fetch and store weather data
    df = processor.fetch_and_process_weather_data()
    print(f"Fetched {len(df)} weather records")

    # Get summary
    summary = processor.get_weather_summary()
    print(f"Weather Summary: {summary}")

    # Get latest weather
    latest = processor.get_latest_weather()
    print(f"Latest Weather: {latest}")

    '''
