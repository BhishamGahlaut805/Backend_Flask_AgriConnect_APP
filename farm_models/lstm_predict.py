'''
This file provides the code for generating the dataset for preparing the data which is required by LSTM model in time series from 2020 to 2025, it merges the weather data with the yield data and prepares the final dataset for training the LSTM model. The weather data is fetched from NASA POWER API and the yield data is fetched from the local CSV file. The final dataset is saved as a CSV file in the specified output directory.
'''

import os
import requests
import pandas as pd
from typing import Optional
from tqdm import tqdm   #type: ignore

class WeatherAveragesExporter:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.nasa_power_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        self.column_mappings = {
            "T2M": "avg_temperature_2m_mean",
            "T2M_MAX": "avg_temperature_2m_max",
            "T2M_MIN": "avg_temperature_2m_min",
            "RH2M": "avg_relative_humidity_2m_mean",
            "WS2M": "avg_wind_speed_10m_max",
            "WD2M": "avg_wind_direction_10m_dominant",
            "PRECTOTCORR": "avg_precipitation_sum",
            "ALLSKY_SFC_SW_DWN": "avg_shortwave_radiation_sum",
            "PS": "avg_surface_pressure_mean",
            "CLOUD_AMT": "avg_cloud_cover_mean",
        }
        self.state_coordinates = {
            "Andaman And Nicobar Islands": (11.7401, 92.6586),
            "Andhra Pradesh": (15.9129, 79.7400),
            "Arunachal Pradesh": (28.2180, 94.7278),
            "Assam": (26.2006, 92.9376),
            "Bihar": (25.0961, 85.3131),
            "Chandigarh": (30.7333, 76.7794),
            "Chhattisgarh": (21.2787, 81.8661),
            "Dadra And Nagar Haveli": (20.1809, 73.0169),
            "Daman And Diu": (20.4283, 72.8397),
            "Delhi": (28.7041, 77.1025),
            "Goa": (15.2993, 74.1240),
            "Gujarat": (22.2587, 71.1924),
            "Haryana": (29.0588, 76.0856),
            "Himachal Pradesh": (31.1048, 77.1734),
            "Jammu And Kashmir": (33.7782, 76.5762),
            "Jharkhand": (23.6102, 85.2799),
            "Karnataka": (15.3173, 75.7139),
            "Kerala": (10.8505, 76.2711),
            "Ladakh": (34.2268, 77.5619),
            "Lakshadweep": (10.5667, 72.6417),
            "Madhya Pradesh": (22.9734, 78.6569),
            "Maharashtra": (19.7515, 75.7139),
            "Manipur": (24.6637, 93.9063),
            "Meghalaya": (25.4670, 91.3662),
            "Mizoram": (23.1645, 92.9376),
            "Nagaland": (26.1584, 94.5624),
            "Odisha": (20.9517, 85.0985),
            "Puducherry": (11.9416, 79.8083),
            "Punjab": (31.1471, 75.3412),
            "Rajasthan": (27.0238, 74.2179),
            "Sikkim": (27.5330, 88.5122),
            "Tamil Nadu": (11.1271, 78.6569),
            "Telangana": (18.1124, 79.0193),
            "Tripura": (23.9408, 91.9882),
            "Uttar Pradesh": (26.8467, 80.9462),
            "Uttarakhand": (30.0668, 79.0193),
            "West Bengal": (22.9868, 87.8550),
            "Others": (22.9734, 78.6569),
            "All India": (22.9734, 78.6569),
        }

    def fetch_and_process_weather(self, state: str, year: int) -> Optional[pd.Series]:
        """
        Fetch daily weather data for a state and year, return yearly averages as Series.
        """
        lat, lon = self.state_coordinates[state]
        start_date = f"{year}0101"
        end_date = f"{year}1231"
        params = {
            "parameters": ",".join(self.column_mappings.keys()),
            "community": "AG",
            "longitude": lon,
            "latitude": lat,
            "start": start_date,
            "end": end_date,
            "format": "JSON",
        }
        try:
            response = requests.get(self.nasa_power_url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
            if "properties" not in data or "parameter" not in data["properties"]:
                return None
            parameters = data["properties"]["parameter"]
            dfs = []
            for param, values in parameters.items():
                if param in self.column_mappings:
                    df = pd.DataFrame.from_dict(values, orient='index', columns=[self.column_mappings[param]])
                    dfs.append(df)
            if not dfs:
                return None
            weather_df = pd.concat(dfs, axis=1)
            weather_df.index = pd.to_datetime(weather_df.index)
            avg = weather_df.mean()
            avg["state"] = state
            avg["year"] = year
            return avg
        except Exception as e:
            print(f"Failed to get weather data for {state} {year}: {str(e)}")
            return None

    def export_weather_averages(self, years=range(2020, 2026)):
        """
        Fetch weather data for all states and years, export to CSV.
        """
        records = []
        for state in tqdm(self.state_coordinates.keys(), desc="States"):
            for year in years:
                avg_series = self.fetch_and_process_weather(state, year)
                if avg_series is not None:
                    records.append(avg_series)
        if records:
            df = pd.DataFrame(records)
            cols = ["state", "year"] + [self.column_mappings[k] for k in self.column_mappings]
            df = df[cols]
            output_path = os.path.join(self.output_dir, "state_yearly_weather_averages_2020_2025.csv")
            df.to_csv(output_path, index=False)
            print(f"Saved weather averages to: {output_path}")
            return df
        else:
            print("No weather data was processed.")
            return pd.DataFrame()

'''
if __name__ == "__main__":
    exporter = WeatherAveragesExporter(
        output_dir=r""
    )
    exporter.export_weather_averages()
'''
