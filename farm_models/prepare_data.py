"""
Weather Data Processor for Crop Recommendation
Fetches and processes weather data from APIs for user farms
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from mongo_storage import mongo_storage
from logging_config import logger


class WeatherDataProcessor:
    """Weather data processor with optional MongoDB caching"""

    def __init__(self, latitude: float, longitude: float, farm_id: str, farm_name: str):
        self.latitude = latitude
        self.longitude = longitude
        self.farm_id = farm_id
        self.farm_name = farm_name
        self.timezone = "auto"

        # API URLs
        self.nasa_power_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        self.openmeteo_url = "https://api.open-meteo.com/v1/forecast"

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

    def process_nasa_power_data(self, raw_data: Dict) -> pd.DataFrame:
        """Process NASA POWER API response"""
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
        """Process Open-Meteo API response"""
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

    def unify_dataframes(self, nasa_df, om_hist_df, om_fcst_df) -> pd.DataFrame:
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
        """Fetch weather data from APIs and return as DataFrame"""
        today = datetime.now().date()
        start_date = datetime(2023, 1, 1).date()

        logger.info(f"Fetching weather data for farm {self.farm_name}")

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
            logger.info(f"Processed {len(final_df)} weather records")

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

            return self.process_nasa_power_data(response.json())

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

            return self.process_openmeteo_data(response.json(), "Historical")

        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
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

            return self.process_openmeteo_data(response.json(), "Forecast")

        except Exception as e:
            logger.error(f"Error fetching forecast data: {e}")
            return pd.DataFrame()

    def get_weather_summary(self, df: pd.DataFrame) -> dict:
        """Get summary statistics from weather DataFrame"""
        if df.empty:
            return self._get_default_weather()

        # Get most recent record
        latest = df.iloc[-1]

        # Calculate averages for the last 30 days
        last_30_days = df.tail(30)

        return {
            "avg_temperature_2m_mean": float(latest.get("temperature_2m_mean", 0)),
            "avg_temperature_2m_max": float(latest.get("temperature_2m_max", 0)),
            "avg_temperature_2m_min": float(latest.get("temperature_2m_min", 0)),
            "avg_relative_humidity_2m_mean": float(latest.get("relative_humidity_2m_mean", 0)),
            "avg_wind_speed_10m_max": float(latest.get("wind_speed_10m_max", 0)),
            "avg_precipitation_sum": float(last_30_days["precipitation_sum"].mean() if not last_30_days.empty else 0),
            "avg_shortwave_radiation_sum": float(latest.get("shortwave_radiation_sum", 0)),
            "avg_surface_pressure_mean": float(latest.get("surface_pressure_mean", 0)),
            "avg_cloud_cover_mean": float(latest.get("cloud_cover_mean", 0)),
            "date": latest.get("date").isoformat() if hasattr(latest.get("date"), 'isoformat') else str(latest.get("date")),
            "records_available": len(df)
        }

    def _get_default_weather(self) -> dict:
        """Return default weather values"""
        return {
            "avg_temperature_2m_mean": 25.0,
            "avg_temperature_2m_max": 30.0,
            "avg_temperature_2m_min": 20.0,
            "avg_relative_humidity_2m_mean": 60.0,
            "avg_wind_speed_10m_max": 10.0,
            "avg_precipitation_sum": 50.0,
            "avg_shortwave_radiation_sum": 200.0,
            "avg_surface_pressure_mean": 1013.0,
            "avg_cloud_cover_mean": 40.0,
            "records_available": 0
        }
        