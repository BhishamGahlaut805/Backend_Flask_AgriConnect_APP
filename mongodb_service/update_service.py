"""
SummaryUpdateService - Backward compatibility wrapper
"""

from mongo_storage import mongo_storage


class SummaryUpdateService:
    """Wrapper for summary update service"""

    def update_farm_stats(self, farm_id: str = None):
        mongo_storage.update_farm_stats(farm_id)

    def update_user_summary(self, user_id: str = None):
        mongo_storage.update_user_summary(user_id)

    def run_all(self):
        mongo_storage.update_all_stats()


# For direct import compatibility
SummaryUpdateService = SummaryUpdateService()