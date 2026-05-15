"""
BackendFlask - Agricultural Support System
"""

__version__ = "1.0.0"
__author__ = "AgriSupport Team"

from mongo_storage import mongo_storage
from config import Config
from logging_config import logger

# Initialize logging
logger.info(f"AgriSupport Backend v{__version__} initialized")
logger.info(f"MongoDB connected: {mongo_storage.health_check()}")
