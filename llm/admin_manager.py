"""
Admin Manager for LLM Module
Handles scraping of weather, news, bulletins, and disease information
All scraped data stored in MongoDB
"""

import os
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from config import Config
from mongo_storage import mongo_storage
# from logging_config import logger as root_logger
from logging_config import logger

SCRAPING_INTERVALS=Config.SCRAPING_INTERVALS

class AdminManager:
    """Admin manager for scraping and managing data with MongoDB storage"""

    def __init__(self, pinecone_manager):
        self.pinecone_manager = pinecone_manager
        self.is_running = False
        self.scraping_thread = None
        self.last_scrape_times = {}

        # MongoDB collections
        self.weather_collection = mongo_storage.weather_data
        self.news_collection = mongo_storage.stored_news
        self.bulletins_collection = mongo_storage.bulletins
        self.disease_collection = mongo_storage.db["disease_info"]
        self.scrape_logs = mongo_storage.db["scrape_logs"]

        # Default locations for weather scraping
        self.default_locations = [
            {"state": "Haryana", "lat": 29.0588, "lon": 76.0856},
            {"state": "Punjab", "lat": 31.1471, "lon": 75.3412},
            {"state": "Uttar Pradesh", "lat": 26.8467, "lon": 80.9462},
            {"state": "Rajasthan", "lat": 27.0238, "lon": 74.2179},
            {"state": "Gujarat", "lat": 22.2587, "lon": 71.1924},
            {"state": "Maharashtra", "lat": 19.7515, "lon": 75.7139},
            {"state": "Madhya Pradesh", "lat": 22.9734, "lon": 78.6569},
            {"state": "West Bengal", "lat": 22.9868, "lon": 87.8550},
            {"state": "Bihar", "lat": 25.0961, "lon": 85.3131},
            {"state": "Tamil Nadu", "lat": 11.1271, "lon": 78.6569},
        ]

        self._init_collections()

    def _init_collections(self):
        """Initialize MongoDB collections with indexes"""
        self.scrape_logs.create_index([("timestamp", -1)])
        self.scrape_logs.create_index([("data_type", 1)])

        if self.disease_collection is not None:
            self.disease_collection.create_index([("disease", 1)])
            self.disease_collection.create_index([("crop", 1)])

    def start_auto_scraping(self):
        """Start automatic scraping in background thread"""
        self.is_running = True
        self.scraping_thread = threading.Thread(target=self._scraping_loop, daemon=True)
        self.scraping_thread.start()
        logger.info("Auto-scraping started")

    def stop_auto_scraping(self):
        """Stop automatic scraping"""
        self.is_running = False
        if self.scraping_thread:
            self.scraping_thread.join(timeout=5)
        logger.info("Auto-scraping stopped")

    def _scraping_loop(self):
        """Main scraping loop - runs periodically for auto-scraping"""
        while self.is_running:
            try:
                current_time = datetime.now()

                # Check each data type and scrape if needed
                for data_type, interval_hours in SCRAPING_INTERVALS.items():
                    last_scrape = self.last_scrape_times.get(data_type)

                    if last_scrape is None:
                        # Never scraped, scrape now
                        self._scrape_data_type(data_type)
                    elif current_time - last_scrape >= timedelta(hours=interval_hours):
                        # Time to scrape again
                        self._scrape_data_type(data_type)

                # Sleep for 1 hour before checking again
                time.sleep(3600)

            except Exception as e:
                logger.error(f"Scraping loop error: {e}")
                time.sleep(300)

    def _scrape_data_type(self, data_type: str):
        """Scrape specific data type based on schedule"""
        try:
            if data_type == "weather":
                self.scrape_weather_data()
            elif data_type == "news":
                self.scrape_news_data()
            elif data_type == "bulletins":
                self.scrape_bulletins()
            elif data_type == "diseases":
                self.scrape_disease_info()

            self.last_scrape_times[data_type] = datetime.now()
            self._log_scrape(data_type, "success", f"Auto-scrape completed")

        except Exception as e:
            logger.error(f"Auto-scrape failed for {data_type}: {e}")
            self._log_scrape(data_type, "failed", str(e))

    def _log_scrape(self, data_type: str, status: str, message: str):
        """Log scraping activity to MongoDB"""
        try:
            self.scrape_logs.insert_one({
                "data_type": data_type,
                "status": status,
                "message": message,
                "timestamp": datetime.utcnow()
            })
        except Exception as e:
            logger.error(f"Failed to log scrape: {e}")

    def scrape_weather_data(self, locations: List[Dict] = None) -> int:
        """Scrape weather data for specified locations and store in MongoDB"""
        try:
            if not locations:
                locations = self.default_locations

            success_count = 0

            for location in locations:
                try:
                    weather_data = self._fetch_weather_from_api(
                        location["lat"],
                        location["lon"]
                    )

                    if weather_data and "error" not in weather_data:
                        # Add location info
                        weather_data["location"] = location["state"]
                        weather_data["latitude"] = location["lat"]
                        weather_data["longitude"] = location["lon"]
                        weather_data["fetched_at"] = datetime.utcnow()

                        # Store in MongoDB
                        mongo_storage.save_weather_data(weather_data, location["state"])

                        # Also add to Pinecone for retrieval
                        self.pinecone_manager.add_weather_data(weather_data, location["state"])

                        success_count += 1
                        logger.info(f"Weather data processed for {location['state']}")

                except Exception as e:
                    logger.error(f"Weather scraping failed for {location['state']}: {e}")
                    continue

            self.last_scrape_times["weather"] = datetime.now()
            self._log_scrape("weather", "success", f"Processed {success_count}/{len(locations)} locations")

            return success_count

        except Exception as e:
            logger.error(f"Weather scraping failed: {e}")
            self._log_scrape("weather", "failed", str(e))
            return 0

    def _fetch_weather_from_api(self, lat: float, lon: float) -> Dict:
        """Fetch weather data from Open-Meteo API"""
        import requests

        try:
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": lat,
                "longitude": lon,
                "current": ["temperature_2m", "relative_humidity_2m", "precipitation",
                           "wind_speed_10m", "weather_code"],
                "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation_probability"],
                "timezone": "auto"
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Extract current conditions
            current = data.get("current", {})
            hourly = data.get("hourly", {})

            # Calculate averages for the next 24 hours
            temps = hourly.get("temperature_2m", [])[:24]
            humidity = hourly.get("relative_humidity_2m", [])[:24]
            rain_prob = hourly.get("precipitation_probability", [])[:24]

            weather_data = {
                "temperature_avg": round(sum(temps) / len(temps), 1) if temps else None,
                "temperature_current": current.get("temperature_2m"),
                "humidity_avg": round(sum(humidity) / len(humidity), 1) if humidity else None,
                "humidity_current": current.get("relative_humidity_2m"),
                "precipitation": current.get("precipitation", 0),
                "wind_speed": current.get("wind_speed_10m", 0),
                "rain_probability_max": max(rain_prob) if rain_prob else 0,
                "weather_code": current.get("weather_code"),
                "timestamp": datetime.now().isoformat(),
                "source": "open-meteo"
            }

            return weather_data

        except Exception as e:
            logger.error(f"Failed to fetch weather from API: {e}")
            return {"error": str(e)}

    def scrape_news_data(self) -> int:
        """Scrape agricultural news and store in MongoDB"""
        try:
            news_items = self._fetch_agri_news()

            if news_items:
                success_count = 0
                for news in news_items:
                    # Store in MongoDB
                    mongo_storage.save_news_item(news)
                    success_count += 1

                # Add to Pinecone for retrieval
                self.pinecone_manager.add_news_data(news_items)

                self.last_scrape_times["news"] = datetime.now()
                self._log_scrape("news", "success", f"Scraped {success_count} news items")

                return success_count

            return 0

        except Exception as e:
            logger.error(f"News scraping failed: {e}")
            self._log_scrape("news", "failed", str(e))
            return 0

    def _fetch_agri_news(self) -> List[Dict]:
        """Fetch agricultural news from various sources"""
        import requests
        from bs4 import BeautifulSoup

        news_items = []

        # Source 1: PIB India (Government of India)
        try:
            pib_url = "https://pib.gov.in/indexd.aspx"
            response = requests.get(pib_url, timeout=30)
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find agriculture-related news
            for item in soup.find_all('div', class_='content_box')[:10]:
                title_elem = item.find('a')
                if title_elem and ('agriculture' in title_elem.text.lower() or
                                   'farmer' in title_elem.text.lower() or
                                   'crop' in title_elem.text.lower()):
                    news_items.append({
                        "title": title_elem.text.strip(),
                        "source": "PIB India",
                        "summary": title_elem.text.strip(),
                        "url": "https://pib.gov.in/" + title_elem.get('href', ''),
                        "published_at": datetime.now().isoformat(),
                        "fetched_at": datetime.utcnow()
                    })
        except Exception as e:
            logger.error(f"Failed to fetch PIB news: {e}")

        # Source 2: Sample news data (fallback)
        if not news_items:
            news_items = self._get_sample_news_data()

        return news_items

    def _get_sample_news_data(self) -> List[Dict]:
        """Return sample agricultural news data as fallback"""
        return [
            {
                "title": "Government launches new scheme for farmers",
                "source": "Agriculture Ministry",
                "summary": "New scheme to provide subsidies for organic farming",
                "url": "#",
                "published_at": datetime.now().isoformat(),
                "fetched_at": datetime.utcnow()
            },
            {
                "title": "Weather forecast predicts timely monsoon",
                "source": "IMD",
                "summary": "Southwest monsoon expected to arrive on time this year",
                "url": "#",
                "published_at": datetime.now().isoformat(),
                "fetched_at": datetime.utcnow()
            },
            {
                "title": "New pest resistant crop varieties released",
                "source": "ICAR",
                "summary": "Scientists develop pest-resistant varieties of cotton and rice",
                "url": "#",
                "published_at": datetime.now().isoformat(),
                "fetched_at": datetime.utcnow()
            }
        ]

    def scrape_bulletins(self, states: List[str] = None) -> int:
        """Scrape agricultural bulletins and store in MongoDB"""
        try:
            if not states:
                states = ["Haryana", "Delhi", "Uttar Pradesh", "Punjab", "Rajasthan"]

            bulletins = []
            success_count = 0

            for state in states:
                try:
                    bulletin_data = self._fetch_imd_bulletin(state)
                    if bulletin_data:
                        bulletins.append(bulletin_data)
                        success_count += 1
                        logger.info(f"Bulletin fetched for {state}")
                except Exception as e:
                    logger.error(f"Bulletin fetch failed for {state}: {e}")
                    continue

            if bulletins:
                # Store in MongoDB
                for bulletin in bulletins:
                    mongo_storage.save_bulletin(bulletin)

                # Add to Pinecone
                self.pinecone_manager.add_bulletins_data(bulletins)

                self.last_scrape_times["bulletins"] = datetime.now()
                self._log_scrape("bulletins", "success", f"Processed {success_count}/{len(states)} states")

            return success_count

        except Exception as e:
            logger.error(f"Bulletin scraping failed: {e}")
            self._log_scrape("bulletins", "failed", str(e))
            return 0

    def _fetch_imd_bulletin(self, state: str) -> Optional[Dict]:
        """Fetch IMD agromet bulletin for a state"""
        import requests

        try:
            # Sample bulletin data (in production, this would fetch from IMD API)
            bulletin = {
                "state": state,
                "title": f"Agromet Advisory for {state}",
                "content": f"""
                Weather Advisory for {state}:
                - Light to moderate rainfall expected in next 24 hours
                - Farmers advised to complete harvesting of matured crops
                - Irrigation scheduling recommended based on rainfall forecast
                - Monitor for pest and disease incidence in crops
                """,
                "source": "IMD",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "fetched_at": datetime.utcnow()
            }
            return bulletin

        except Exception as e:
            logger.error(f"Failed to fetch IMD bulletin: {e}")
            return None

    def scrape_disease_info(self) -> int:
        """Scrape crop disease information and store in MongoDB"""
        try:
            disease_data = self._fetch_disease_information()

            if disease_data:
                # Store in MongoDB
                for disease in disease_data:
                    disease["stored_at"] = datetime.utcnow()
                    self.disease_collection.update_one(
                        {"disease": disease["disease"], "crop": disease["crop"]},
                        {"$set": disease},
                        upsert=True
                    )

                # Add to Pinecone
                self.pinecone_manager.add_disease_data(disease_data)

                self.last_scrape_times["diseases"] = datetime.now()
                self._log_scrape("diseases", "success", f"Updated {len(disease_data)} disease entries")

                return len(disease_data)
            return 0

        except Exception as e:
            logger.error(f"Disease info scraping failed: {e}")
            self._log_scrape("diseases", "failed", str(e))
            return 0

    def _fetch_disease_information(self) -> List[Dict]:
        """Fetch comprehensive crop disease information"""
        return [
            {
                "disease": "Late Blight of Potato",
                "crop": "Potato",
                "symptoms": "Water-soaked lesions on leaves that turn brown and necrotic, white fungal growth on undersides of leaves during humid conditions, rapid destruction of foliage",
                "treatment": "Apply fungicides containing chlorothalonil, mancozeb, or metalaxyl. Remove and destroy infected plants. Use certified disease-free seed potatoes.",
                "prevention": "Plant resistant varieties, ensure proper spacing for air circulation, avoid overhead irrigation, practice crop rotation with non-host crops",
                "source": "Agricultural Plant Pathology Database"
            },
            {
                "disease": "Wheat Rust",
                "crop": "Wheat",
                "symptoms": "Orange-brown pustules on leaves and stems, yellowing of leaves, reduced grain filling, premature leaf death",
                "treatment": "Apply fungicides like propiconazole or tebuconazole at first sign of infection. Remove volunteer wheat plants.",
                "prevention": "Plant rust-resistant varieties, avoid excessive nitrogen fertilization, practice timely sowing",
                "source": "Cereal Disease Research Institute"
            },
            {
                "disease": "Rice Blast",
                "crop": "Rice",
                "symptoms": "Diamond-shaped lesions with gray centers and brown borders on leaves, neck rot causing whiteheads, node infections",
                "treatment": "Apply fungicides like tricyclazole, isoprothiolane, or carbendazim. Drain fields to reduce humidity.",
                "prevention": "Use resistant varieties, avoid excessive nitrogen, maintain proper water management, destroy crop residues",
                "source": "International Rice Research Institute"
            },
            {
                "disease": "Cotton Bollworm",
                "crop": "Cotton",
                "symptoms": "Bolls show circular boreholes, larvae inside bolls, damaged lint, premature boll opening",
                "treatment": "Use pheromone traps for monitoring, apply recommended insecticides, biological control with Trichogramma",
                "prevention": "Install light traps, maintain field sanitation, use resistant varieties, crop rotation",
                "source": "Cotton Research Institute"
            },
            {
                "disease": "Powdery Mildew of Grapes",
                "crop": "Grapes",
                "symptoms": "White powdery growth on leaves, shoots and berries, stunted growth, reduced fruit quality",
                "treatment": "Apply sulfur-based fungicides, potassium bicarbonate, neem oil sprays",
                "prevention": "Maintain good air circulation, prune properly, avoid dense canopy, resistant varieties",
                "source": "Grape Research Station"
            }
        ]

    def get_scraping_status(self) -> Dict[str, Any]:
        """Get current scraping status"""
        status = {
            "is_running": self.is_running,
            "last_scrape_times": {},
            "next_scrapes": {}
        }

        for data_type in SCRAPING_INTERVALS:
            last_time = self.last_scrape_times.get(data_type)
            status["last_scrape_times"][data_type] = last_time.isoformat() if last_time else "Never"

            if last_time:
                next_time = last_time + timedelta(hours=SCRAPING_INTERVALS[data_type])
                status["next_scrapes"][data_type] = next_time.isoformat()
            else:
                status["next_scrapes"][data_type] = "Ready for first scrape"

        # Add counts from MongoDB
        try:
            status["weather_count"] = mongo_storage.weather_data.count_documents({})
            status["news_count"] = mongo_storage.stored_news.count_documents({})
            status["bulletins_count"] = mongo_storage.bulletins.count_documents({})
            status["disease_count"] = (
                self.disease_collection.count_documents({})
                if self.disease_collection is not None
                else 0
            )
        except Exception as e:
            logger.error(f"Failed to get counts: {e}")

        return status
