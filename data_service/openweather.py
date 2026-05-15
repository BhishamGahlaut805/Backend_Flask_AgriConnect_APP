import os
import requests

OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHER_API_KEY", "your_api_key_here")
class OpenWeatherAPI:
    def get_weather(self, lat, lon):
        url =os.getenv("OPENWEATHER_API_URL") or "http://api.openweathermap.org/data/2.5/weather"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": OPENWEATHERMAP_API_KEY,
            "units": "metric"
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
