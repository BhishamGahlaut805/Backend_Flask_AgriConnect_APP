"""
DiseaseReportService - Backward compatibility wrapper
"""

from mongo_storage import mongo_storage


class DiseaseReportService:
    """Wrapper for disease report service"""

    @staticmethod
    def save_report(data: dict):
        return mongo_storage.save_disease_report(data)

    @staticmethod
    def get_reports(farm_id: str = None, days: int = 30,
                   disease: str = None, limit: int = 100) -> list:
        return mongo_storage.get_disease_reports(farm_id, days, disease, limit)

    @staticmethod
    def get_nearby_reports(lat: float, lon: float, radius_km: float = 5, days: int = 10) -> list:
        return mongo_storage.get_disease_reports_nearby(lat, lon, radius_km, days)


# For direct import compatibility
DiseaseReportService = DiseaseReportService()