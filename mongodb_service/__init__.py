"""
MongoDB Service Module
Provides backward compatibility for existing imports
"""

from mongo_storage import mongo_storage

# Export commonly used functions for easier access
get_farm = mongo_storage.get_farm
get_all_farms = mongo_storage.get_all_farms
create_farm = mongo_storage.create_farm
update_farm = mongo_storage.update_farm
save_disease_report = mongo_storage.save_disease_report
get_disease_reports = mongo_storage.get_disease_reports
update_all_stats = mongo_storage.update_all_stats

__all__ = [
    'mongo_storage',
    'get_farm',
    'get_all_farms',
    'create_farm',
    'update_farm',
    'save_disease_report',
    'get_disease_reports',
    'update_all_stats'
]