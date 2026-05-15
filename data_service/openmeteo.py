import requests
from datetime import datetime
import os

class OpenMeteoAPI:
    BASE_URL = os.getenv("OPEN_METEO_BASE_URL") or "https://api.open-meteo.com/v1/forecast"

    def get_forecast_soil(self, lat, lon):
        """
        Fetch forecasted soil temperature & moisture using Open-Meteo forecast API.
        """

        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join([
                "soil_temperature_0cm",
                "soil_temperature_18cm",
                "soil_moisture_1_to_3cm",
                "soil_moisture_27_to_81cm",
                "evapotranspiration",
                "wind_gusts_10m",
                "cloud_cover_low",
                "cloud_cover_high"
            ]),
            "timezone": "auto",
            "forecast_days": 1
        }

        try:
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            hourly_data = data.get("hourly", {})

            def latest(values):
                return round(values[-1], 3) if values else None

            return {
                "soil_temp_0cm": latest(hourly_data.get("soil_temperature_0cm")),
                "soil_temp_18cm": latest(hourly_data.get("soil_temperature_18cm")),
                "soil_moisture_1_3": latest(hourly_data.get("soil_moisture_1_to_3cm")),
                "soil_moisture_27_81": latest(hourly_data.get("soil_moisture_27_to_81cm")),
                "evapotranspiration": latest(hourly_data.get("evapotranspiration")),
                "wind_gust_10m": latest(hourly_data.get("wind_gusts_10m")),
                "cloud_low": latest(hourly_data.get("cloud_cover_low")),
                "cloud_high": latest(hourly_data.get("cloud_cover_high"))
            }

        except requests.exceptions.RequestException as e:
            print(f"[OpenMeteo ERROR] Forecast soil data fetch failed: {e}")
            return {}
        except Exception as e:
            print(f"[OpenMeteo ERROR] Unexpected error: {e}")
            return {}
