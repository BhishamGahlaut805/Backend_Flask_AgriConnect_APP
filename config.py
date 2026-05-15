import os
from datetime import timedelta
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Flask
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "default-secret-key")
    PORT = int(os.getenv("PORT", 5500))
    FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

    # MongoDB
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    MONGO_DB = "AgriSupportDB"

    # Cloudinary
    CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
    CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
    CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")

    # Model Collections in MongoDB
    MODEL_COLLECTIONS = {
        "crop_weed": "models_crop_weed",
        "potato_disease": "models_potato_disease",
        "cotton_disease": "models_cotton_disease",
        "multiple_disease": "models_multiple_disease",
        "lstm_yield": "models_lstm_yield",
        "crop_recommendation": "models_crop_recommendation",
        "lstm_outbreak": "models_lstm_outbreak"
    }

    # Upload Settings
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov', 'pdf'}

    # ML Settings
    IMAGE_SIZE = tuple(map(int, os.getenv("IMAGE_SIZE", "224,224").split(',')))
    HF_REPO_ID = os.getenv("HF_REPO_ID")

    # API Keys
    AGROMONITORING_API_KEY = os.getenv("AGROMONITORING_API_KEY")
    OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    API_KEY_GPT = os.getenv("API_KEY_GPT")  # Groq API key

    # Pinecone Indexes
    PINECONE_INDEXES = {
        "weather": "agribot-weather",
        "news": "agribot-news",
        "diseases": "agribot-diseases",
        "bulletins": "agribot-bulletins",
        "general": "agribot-general"
    }

    # Scraping Intervals
    SCRAPING_INTERVALS = {
        "weather": 6,
        "news": 24,
        "bulletins": 24,
        "diseases": 168
    }

    # Class Colors
    CLASS_COLORS = {
        "Soil": (0, 255, 0),
        "Weed": (0, 0, 255),
        "Cotton": (255, 165, 0)
    }

    # Training Configuration
    TRAINING_CONFIG = {
        'batch_size': 32,
        'epochs': 5,
        'lr': 0.001,
        'patience': 15,
        'min_delta': 0.0001,
        'aux_weight': 0.3,
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'bidirectional': True
    }

    # Feature Columns
    FEATURE_COLUMNS = [
        'avg_temperature_2m_mean',
        'avg_precipitation_sum',
        'soil_pH',
        'organic_matter_content',
        'plant_population_density'
    ]
    