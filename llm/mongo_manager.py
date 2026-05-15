"""
MongoDBManager for LLM module - Backward compatibility wrapper
"""

from mongo_storage import mongo_storage
from datetime import datetime


class MongoDBManager:
    """Wrapper for LLM MongoDB operations"""

    def __init__(self, uri=None, db_name=None):
        self.mongo = mongo_storage
        self.db = self.mongo.db

        # Reference to existing collections
        self.uploaded_files = self.mongo.uploaded_files
        self.stored_news = self.mongo.stored_news
        self.weather_data = self.mongo.weather_data
        self.bulletins = self.mongo.bulletins
        self.chat_history = self.mongo.chat_history
        self.training_jobs = self.mongo.training_jobs
        self.training_status = self.mongo.training_status

    def health_check(self) -> bool:
        return self.mongo.health_check()

    def save_chat_message(self, session_id: str, query: str, response: str,
                         context_types: list, metadata: dict = None):
        self.mongo.save_chat_message(session_id, query, response, context_types, metadata)

    def get_chat_history(self, session_id: str, limit: int = 15) -> list:
        return self.mongo.get_chat_history(session_id, limit)

    def clear_chat_history(self, session_id: str) -> int:
        return self.mongo.clear_chat_history(session_id)

    def save_news_item(self, news_item: dict) -> str:
        return self.mongo.save_news_item(news_item)

    def get_news_items(self, limit: int = 50) -> list:
        return self.mongo.get_news_items(limit)

    def save_weather_data(self, weather_data: dict, location: str) -> str:
        return self.mongo.save_weather_data(weather_data, location)

    def get_weather_data(self, location: str, hours: int = 24) -> dict:
        return self.mongo.get_weather_data(location, hours)

    def save_bulletin(self, bulletin: dict) -> str:
        return self.mongo.save_bulletin(bulletin)

    def get_bulletins(self, state: str = None, limit: int = 20) -> list:
        return self.mongo.get_bulletins(state, limit)

    def create_training_job(self, job_data: dict) -> str:
        return self.mongo.create_training_job(job_data)

    def update_training_job(self, job_id: str, status: str, result: dict = None):
        self.mongo.update_training_job(job_id, status, result)

    def get_training_job(self, job_id: str) -> dict:
        return self.mongo.get_training_job(job_id)

    def update_training_status(self, job_type: str, status: str, metadata: dict = None):
        self.mongo.update_training_status(job_type, status, metadata)

    def get_training_status(self, job_type: str) -> dict:
        return self.mongo.get_training_status(job_type)

    def save_uploaded_file_metadata(self, filename: str, index_type: str,
                                   file_size: int, file_path: str = None) -> str:
        return self.mongo.save_uploaded_file_metadata(filename, index_type, file_size, file_path)

    def get_uploaded_files(self, index_type: str = None) -> list:
        return self.mongo.get_uploaded_files(index_type)

    def delete_uploaded_file_metadata(self, filename: str, index_type: str) -> int:
        return self.mongo.delete_uploaded_file_metadata(filename, index_type)
    