"""
Agricultural Dataset Generator
Generates yield training data from weather and crop data
All data stored in MongoDB instead of local files
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from geopy.geocoders import Nominatim
from api.openmeteo_api import WeatherDataProcessor
from mongo_storage import mongo_storage
from logging_config import logger


class AgriDatasetGenerator:
    """Generates agricultural dataset for yield prediction"""

    SEASON_DATE_RANGES = {
        "rabi": ((11, 15), (4, 15)),
        "kharif": ((6, 15), (11, 15)),
        "summer": ((5, 15), (10, 15)),
    }
    WINDOW_SIZE_DAYS = 15

    # Default static features
    DEFAULT_STATIC_FEATURES = {
        'soil_type': 'loamy',
        'soil_pH': 6.5,
        'organic_matter_content': 2.0,
        'irrigation_type': 'rain-fed',
        'tillage_type': 'conventional',
        'sowing_method': 'drilling',
        'fertilizer_type_used': 'NPK blend',
        'seed_variety': 'high-yielding',
        'plant_population_density': 50000
    }

    # Allowed crops for training
    ALLOWED_CROPS = {
        'rice', 'cereals', 'total food grains', 'maize', 'nutri/coarse cereals',
        'tur', 'urad', 'moong', 'total pulses', 'jowar', 'bajra', 'ragi',
        'small millets', 'other pulses', 'gram', 'lentil', 'wheat', 'barley'
    }

    def __init__(self, farm_id: str, farm_name: str, latitude: float,
                 longitude: float, static_features: Dict = None):
        self.farm_id = farm_id.lower().replace(" ", "_")
        self.latitude = latitude
        self.longitude = longitude
        self.farm_name = farm_name.lower().replace(" ", "_")

        # Merge provided static features with defaults
        self.static_features = {**self.DEFAULT_STATIC_FEATURES, **(static_features or {})}

        self.geolocator = Nominatim(user_agent="agri-yield-gen")
        self.state = self.reverse_geocode_state(latitude, longitude).lower()
        logger.info(f"Detected state: {self.state}")

        # Load data from MongoDB
        self.df_all = self.load_yield_data_from_mongodb()
        self.npk_features = self.load_npk_data_from_mongodb()

        # Load area sown data from MongoDB
        self.area_sown_data = self.load_area_sown_data_from_mongodb()

        # Process weather data
        self.weather_processor = WeatherDataProcessor(latitude, longitude, farm_id, farm_name)
        self.weather_df = self.weather_processor.fetch_and_process_weather_data()
        if self.weather_df is not None and not self.weather_df.empty:
            self.weather_df['date'] = pd.to_datetime(self.weather_df['date'])
            self.weather_df.set_index("date", inplace=True)

    def reverse_geocode_state(self, lat: float, lon: float) -> str:
        """Reverse geocode to get state name"""
        try:
            location = self.geolocator.reverse((lat, lon), language="en", exactly_one=True)
            return location.raw.get("address", {}).get("state", "unknown").lower()
        except Exception as e:
            logger.error(f"Reverse geocoding failed: {e}")
            return "unknown"

    def load_yield_data_from_mongodb(self) -> pd.DataFrame:
        """Load yield data from MongoDB instead of CSV"""
        try:
            records = list(mongo_storage.db["yield_data"].find({
                "state": self.state,
                "year": 2023
            }))

            if not records:
                logger.warning(f"No yield data found for state {self.state}")
                return pd.DataFrame()

            df = pd.DataFrame(records)

            # Clean column names
            df.columns = [col.lower().replace(" ", "_") for col in df.columns]

            # Ensure required columns exist
            if 'yield' not in df.columns and 'yield_2023_24kg_hectare' in df.columns:
                df['yield'] = df['yield_2023_24kg_hectare']

            df["year"] = 2023

            return df

        except Exception as e:
            logger.error(f"Failed to load yield data: {e}")
            return pd.DataFrame()

    def load_npk_data_from_mongodb(self) -> pd.DataFrame:
        """Load NPK data from MongoDB"""
        try:
            records = list(mongo_storage.db["npk_data"].find({}))

            if records:
                df = pd.DataFrame(records)
                df["state"] = df["state"].str.strip().str.lower()

                # Standardize column names
                column_map = {
                    "n_kg_ha": "n",
                    "p2o5_kg_ha": "p",
                    "k2o_kg_ha": "k",
                    "total_kg_ha": "total_npk"
                }
                for old, new in column_map.items():
                    if old in df.columns and new not in df.columns:
                        df[new] = df[old]

                return df[["state", "n", "p", "k", "total_npk"]]

            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=["state", "n", "p", "k", "total_npk"])

        except Exception as e:
            logger.error(f"Failed to load NPK data: {e}")
            return pd.DataFrame(columns=["state", "n", "p", "k", "total_npk"])

    def load_area_sown_data_from_mongodb(self) -> Optional[Dict]:
        """Load area sown data from MongoDB"""
        try:
            records = list(mongo_storage.db["area_sown_data"].find({}))

            if not records:
                logger.warning("No area sown data found in MongoDB")
                return None

            df = pd.DataFrame(records)

            # Clean column names
            df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

            # Required columns
            required_cols = ["crop", "area_sown_2025_26", "area_sown_2024_25",
                           "difference_in_area_coverage_over_2024_25",
                           "percent_increase_decrease_over_2024_25"]

            # Calculate mean values for fallback
            mean_2025 = df["area_sown_2025_26"].mean() if "area_sown_2025_26" in df.columns else 0
            mean_2024 = df["area_sown_2024_25"].mean() if "area_sown_2024_25" in df.columns else 0
            mean_pct = df["percent_increase_decrease_over_2024_25"].mean() if "percent_increase_decrease_over_2024_25" in df.columns else 0

            # Crop mapping
            crop_mapping = {
                "rice": "rice",
                "total_pulses": "total_pulses",
                "tur": "tur",
                "urad": "urad",
                "moong": "moong",
                "other_pulses": "other_pulses",
                "jowar": "jowar",
                "bajra": "bajra",
                "ragi": "ragi",
                "other_small_millets": "small_millets",
                "total_coarse_cereals": "nutri/coarse_cereals"
            }

            area_data = {}
            for _, row in df.iterrows():
                crop = row["crop"].strip().lower()
                mapped_crop = crop_mapping.get(crop)

                if not mapped_crop:
                    continue

                area_data[mapped_crop] = {
                    "area_2025_26": float(row["area_sown_2025_26"]) if "area_sown_2025_26" in row else 0,
                    "area_2024_25": float(row["area_sown_2024_25"]) if "area_sown_2024_25" in row else 0,
                    "area_change_pct": float(row["percent_increase_decrease_over_2024_25"]) if "percent_increase_decrease_over_2024_25" in row else 0
                }

            # Add mean values for reference
            area_data["_mean_values"] = {
                "area_2025_26": mean_2025,
                "area_2024_25": mean_2024,
                "area_change_pct": mean_pct
            }

            return area_data

        except Exception as e:
            logger.error(f"Failed to load area sown data: {e}")
            return None

    def get_season_date_range(self, year: int, season: str) -> tuple:
        """Get start and end dates for a season"""
        season = season.lower()
        start_tuple, end_tuple = self.SEASON_DATE_RANGES.get(season, ((6, 15), (11, 15)))
        start_date = datetime(year, start_tuple[0], start_tuple[1])
        end_year = year + 1 if end_tuple[0] < start_tuple[0] else year
        end_date = datetime(end_year, end_tuple[0], end_tuple[1])
        return start_date, end_date

    def get_area_sown_values(self, crop: str) -> tuple:
        """Get area sown values for a crop"""
        if self.area_sown_data is None:
            return None, None, None

        crop_data = self.area_sown_data.get(crop.lower())
        if crop_data:
            return (
                crop_data["area_2025_26"],
                crop_data["area_2024_25"],
                crop_data["area_change_pct"]
            )
        else:
            mean_data = self.area_sown_data["_mean_values"]
            return (
                mean_data["area_2025_26"],
                mean_data["area_2024_25"],
                mean_data["area_change_pct"]
            )

    def generate_10_day_windows(self, season_df: pd.DataFrame, crop: str,
                                 season: str, year: int, district: str) -> List[Dict]:
        """Generate aggregated window data with all features"""
        windows = []
        num_windows = len(season_df) // self.WINDOW_SIZE_DAYS

        for window_num in range(num_windows):
            start_idx = window_num * self.WINDOW_SIZE_DAYS
            end_idx = start_idx + self.WINDOW_SIZE_DAYS

            # For last window, include remaining days
            if window_num == num_windows - 1:
                end_idx = len(season_df)

            window_df = season_df.iloc[start_idx:end_idx]

            # Create base row with all static features
            row = {
                "farm_id": self.farm_id,
                "state": self.state,
                "district": district,
                "latitude": self.latitude,
                "longitude": self.longitude,
                "crop": crop,
                "season": season,
                "year": year,
                "window_num": window_num + 1,
                "start_date": window_df.index[0].strftime("%Y-%m-%d"),
                "end_date": window_df.index[-1].strftime("%Y-%m-%d"),
                "is_season_end": (end_idx == len(season_df)),
                **self.static_features
            }

            # Add weather averages
            numeric_cols = window_df.select_dtypes(include='number').columns
            for col in numeric_cols:
                if col != "yield":
                    row[f"avg_{col}"] = round(window_df[col].mean(), 2)

            # Add yield only at season end
            row["yield"] = window_df["yield"].iloc[-1] if row["is_season_end"] else None

            # Add NPK values
            if not self.npk_features.empty:
                static_row = self.npk_features[self.npk_features["state"] == self.state]
                if not static_row.empty:
                    static_row = static_row.iloc[0]
                    row.update({
                        "n": static_row.get("n", 0),
                        "p": static_row.get("p", 0),
                        "k": static_row.get("k", 0),
                        "total_npk": static_row.get("total_npk", 0)
                    })

            # Add area sown data if available
            if self.area_sown_data is not None:
                area_2025, area_2024, area_pct = self.get_area_sown_values(crop)
                row.update({
                    "area_2025_26": area_2025,
                    "area_2024_25": area_2024,
                    "area_change_pct": area_pct
                })

            windows.append(row)

        return windows

    def save_to_mongodb(self, cropwise_data: Dict):
        """Save generated data to MongoDB"""
        for crop, records in cropwise_data.items():
            if not records:
                continue

            # Delete old data for this farm/crop
            mongo_storage.crop_training_data.delete_many({
                "farm_id": self.farm_id,
                "crop": crop
            })

            # Insert new records
            for record in records:
                record["created_at"] = datetime.utcnow()
                mongo_storage.crop_training_data.insert_one(record)

            # Also save as training data in GridFS
            df = pd.DataFrame(records)
            mongo_storage.save_training_data(
                df,
                f"yield_training_{crop}",
                farm_id=self.farm_id,
                crop=crop,
                metadata={"farm_name": self.farm_name, "state": self.state}
            )

            logger.info(f"Saved {len(records)} rows for crop: {crop}")

    def generate(self) -> Dict:
        """Generate complete dataset"""
        if self.weather_df is None or self.weather_df.empty:
            logger.error("No weather data available")
            return {}

        if self.df_all is None or self.df_all.empty:
            logger.error("No yield data available")
            return {}

        cropwise_data = {}
        weather_df = self.weather_df.copy()

        for _, row in self.df_all.iterrows():
            crop = row.get("crop_name", "").strip().lower()
            season = row.get("season", "").strip().lower()
            year = int(row.get("year", 2023))
            yield_val = row.get("yield")
            district = row.get("district_name", "unknown").strip().lower()

            if crop not in self.ALLOWED_CROPS:
                logger.debug(f"Crop '{crop}' not in allowed crop list.")
                continue

            if season not in self.SEASON_DATE_RANGES:
                logger.debug(f"Season '{season}' not supported.")
                continue

            start_date, end_date = self.get_season_date_range(year, season)
            season_weather = weather_df.loc[
                (weather_df.index >= start_date) & (weather_df.index <= end_date)
            ].copy()

            # Add yield only at season end
            season_weather["yield"] = None
            if not season_weather.empty and yield_val is not None:
                season_weather.iloc[-1, season_weather.columns.get_loc("yield")] = yield_val

            if season_weather.empty:
                logger.warning(f"No weather data for {crop} ({season}, {year})")
                continue

            # Generate 10-day windows
            windows = self.generate_10_day_windows(season_weather, crop, season, year, district)
            cropwise_data.setdefault(crop, []).extend(windows)

        # Save to MongoDB
        self.save_to_mongodb(cropwise_data)

        total_rows = sum(len(records) for records in cropwise_data.values())

        if total_rows == 0:
            logger.error("No data generated for any crop.")
        else:
            logger.info(f"Total records saved across crops: {total_rows}")

        return cropwise_data
    