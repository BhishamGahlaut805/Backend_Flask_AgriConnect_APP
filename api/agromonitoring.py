import requests
import logging
from datetime import datetime
import os
import requests
import logging
import time
from datetime import timedelta

class AgroMonitoringAPI:
    BASE_URL = os.getenv("BASE_URL_AGROMONITORING", "https://api.agromonitoring.com/agro/1.0")

    def __init__(self):
        self.api_key = os.getenv("AGROMONITORING_API_KEY")
        self.logger = logging.getLogger(__name__)

    def create_polygon(self, name: str, coordinates: list) -> dict:
        """
        Creates a polygon. If name is duplicated, fetch and return the existing polygon info.
        If not found, retry with 'duplicated=true'.
        """
        payload = {
            "name": name,
            "geo_json": {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coordinates]
                }
            }
        }

        try:
            url = f"{self.BASE_URL}/polygons?appid={self.api_key}"
            response = requests.post(url, json=payload)

            if response.ok:
                self.logger.info(f"Polygon created successfully: {name}")
                return response.json()

            if response.status_code == 422:
                resp_text = response.text.lower()

                if "duplicated" in resp_text:
                    self.logger.warning(f"Polygon name conflict for '{name}'. Fetching existing polygon instead.")

                    # Try to fetch existing polygon
                    polygon_list_url = f"{self.BASE_URL}/polygons?appid={self.api_key}"
                    polygon_list_resp = requests.get(polygon_list_url)

                    if polygon_list_resp.ok:
                        polygons = polygon_list_resp.json()
                        for poly in polygons:
                            if poly.get("name") == name:
                                self.logger.info(f"Existing polygon found for '{name}'")
                                return poly

                        # Not found — retry with 'duplicated=true'
                        self.logger.warning(f"Polygon '{name}' not found in list. Retrying with duplicated=true flag...")
                        duplicate_payload = payload.copy()
                        retry_url = f"{self.BASE_URL}/polygons?appid={self.api_key}&duplicated=true"
                        retry_response = requests.post(retry_url, json=duplicate_payload)

                        if retry_response.ok:
                            self.logger.info(f"Polygon created successfully with duplicated=true: {name}")
                            return retry_response.json()
                        else:
                            self.logger.error(f"Failed to create polygon with duplicated=true: {retry_response.status_code} - {retry_response.text}")
                            retry_response.raise_for_status()
                    else:
                        self.logger.error(f"Failed to fetch polygon list: {polygon_list_resp.status_code}")
                        polygon_list_resp.raise_for_status()

                elif "area of the polygon" in resp_text:
                    self.logger.error("Polygon area validation failed.")
                    raise ValueError("Polygon area must be between 1 and 3000 hectares.")

            # Other error
            self.logger.error(f"Failed to create polygon: {response.status_code} - {response.text}")
            response.raise_for_status()

        except requests.exceptions.RequestException as req_err:
            self.logger.error(f"RequestException during polygon creation for '{name}': {str(req_err)}")
            raise

        except Exception as e:
            self.logger.exception(f"Unexpected error during polygon creation for '{name}': {str(e)}")
            raise


    def get_ndvi_index(self, poly_id, start_date=None, end_date=None,
                   farm_name=None, coordinates=None, update_polygon_id=None):
        """
        Simplified NDVI index fetcher that returns 0 on any failure
        Follows exact AgroMonitoring API pattern from the sample URL
        """
        try:
            # Validate inputs
            if not poly_id:
                self.logger.warning("No polygon ID provided")
                return 0

            # Set default date range (last 30 days)
            end_date = end_date or datetime.utcnow()
            start_date = start_date or (end_date - timedelta(days=30))

            # Convert to UNIX timestamps as shown in sample URL
            start_unix = int(start_date.timestamp())
            end_unix = int(end_date.timestamp())

            # Build API URL exactly as per documentation
            url = (
                f"{self.BASE_URL}/ndvi/history"
                f"?polyid={poly_id}&start={start_unix}&end={end_unix}&appid={self.api_key}"
            )

            # Make API request with timeout
            response = requests.get(url, timeout=10)

            # Return 0 immediately for any non-200 status
            if response.status_code != 200:
                self.logger.warning(f"API request failed with status {response.status_code}")
                return 0

            ndvi_data = response.json()

            # Return 0 if no data received
            if not ndvi_data or not isinstance(ndvi_data, list):
                self.logger.warning("No valid NDVI data received")
                return 0

            # Find most recent valid reading
            for record in sorted(ndvi_data, key=lambda x: x.get('dt', 0), reverse=True):
                try:
                    # Check data quality thresholds
                    if (record.get('cl', 100) <= 60 and  # Cloud cover <= 60%
                        record.get('dc', 0) >= 50 and     # Data coverage >= 50%
                        record.get('data', {}).get('mean') is not None):  # Has NDVI value

                        ndvi_value = float(record['data']['mean'])
                        # Ensure value is within valid NDVI range [-1, 1]
                        if -1 <= ndvi_value <= 1:
                            return round(ndvi_value, 4)
                except (TypeError, ValueError):
                    continue

            # If no valid records found
            self.logger.warning("No valid NDVI records meeting quality thresholds")
            return 0

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed: {str(e)}")
            return 0
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            return 0
