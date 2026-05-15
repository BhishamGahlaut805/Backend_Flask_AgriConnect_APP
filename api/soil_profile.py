import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os

class SoilGridsFetcher:
    def __init__(self):
        self.api_url = os.getenv("SOIL_URL") or "https://rest.isric.org/soilgrids/v2.0/properties/query"
        self.depth = "0-5cm"

        self.groups = [
            ["phh2o", "soc", "bdod", "nitrogen"],  # Group 1
            ["sand", "silt", "clay", "cec"],       # Group 2
        ]

        # Retry mechanism
        retry_strategy = Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session = requests.Session()
        self.session.mount("https://", adapter)

    def fetch_soil_data(self, lat, lon):
        final_data = {}

        # Conversion factors
        factors = {
            "phh2o": 10, "soc": 10, "bdod": 100,
            "sand": 10, "silt": 10, "clay": 10,
            "cec": 10, "nitrogen": 100
        }

        for group in self.groups:
            try:
                params = {
                    "lon": lon,
                    "lat": lat,
                    "depths": self.depth,
                    "properties": ",".join(group),
                }
                response = self.session.get(self.api_url, params=params, timeout=25)
                response.raise_for_status()
                data = response.json()["properties"]

                for prop in group:
                    try:
                        mean_val = data[prop]["layers"][0]["values"]["mean"]
                        if mean_val is not None:
                            final_data[prop] = round(mean_val / factors[prop], 3)
                        else:
                            final_data[prop] = "Unavailable"
                    except Exception:
                        final_data[prop] = "Unavailable"

            except requests.exceptions.RequestException as e:
                print(f"[ERROR] Failed fetching group {group}: {e}")
                for prop in group:
                    final_data[prop] = "Unavailable"

        # Return in readable format
        return {
            "soil_ph": final_data["phh2o"],
            "soil_N_g_per_kg": final_data["nitrogen"],
            "organic_carbon_%": final_data["soc"],
            "bulk_density_g_cm3": final_data["bdod"],
            "sand_%": final_data["sand"],
            "silt_%": final_data["silt"],
            "clay_%": final_data["clay"],
            "cec_cmol_per_kg": final_data["cec"],
        }
'''
# === Example Usage ===
if __name__ == "__main__":
    fetcher = SoilGridsFetcher()
    lat, lon = 30.1666666666667, 76.4333333333333  # Your coordinates
    soil_data = fetcher.fetch_soil_data(lat, lon)
    print("Soil Profile (0 to 5 cm depth):")
    for k, v in soil_data.items():
        print(f"{k}: {v}")
'''
