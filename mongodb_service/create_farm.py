"""
CreateFarmService - Backward compatibility wrapper
"""

from mongo_storage import mongo_storage


class CreateFarmService:
    """Wrapper for farm creation service"""

    @staticmethod
    def generate_farm_id() -> str:
        return mongo_storage.generate_farm_id()

    @staticmethod
    def create_farm(farm_data: dict) -> str:
        return mongo_storage.create_farm(farm_data)

    @staticmethod
    def get_all_farms() -> list:
        return mongo_storage.get_all_farms()

    @staticmethod
    def update_farm_nearby(farm_id: str, updated_nearby: list):
        mongo_storage.update_farm_nearby(farm_id, updated_nearby)

    @staticmethod
    def update_farm_analysis_stats(farm_id: str, new_images_analyzed: int,
                                   new_diseased_images: int, crop: str = None,
                                   disease: str = None, latitude: float = None,
                                   longitude: float = None):
        mongo_storage.update_farm_analysis_stats(
            farm_id, new_images_analyzed, new_diseased_images,
            crop, disease, latitude, longitude
        )


# For direct import compatibility
CreateFarmService = CreateFarmService()