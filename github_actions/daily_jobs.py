#!/usr/bin/env python
"""
Daily jobs script for GitHub Actions
Runs data preparation, LSTM training, and summary updates
"""

import os
import sys
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment
from dotenv import load_dotenv
load_dotenv()

from mongo_storage import mongo_storage
from logging_config import logger
from data_service.prepare_data import DataService
from crop_disease_outbreak.lstm.lstm_outbreak import LSTMOutbreakPredictor
from mongodb_service.update_service import SummaryUpdateService

def run_daily_jobs():
    """Run all daily maintenance jobs"""
    logger.info("Starting daily jobs...")

    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "status": "running"
    }

    try:
        # 1. Data Preparation
        logger.info("Running DataService...")
        data_service = DataService()
        data_service.run_once_a_day()
        results["data_preparation"] = "completed"
        logger.info("DataService completed")
    except Exception as e:
        logger.error(f"DataService Error: {e}")
        results["data_preparation"] = f"error: {str(e)}"

    try:
        # 2. LSTM Training
        logger.info("Running LSTM Training...")
        predictor = LSTMOutbreakPredictor()
        predictor.run_for_all_farms()
        results["lstm_training"] = "completed"
        logger.info("LSTM training completed")
    except Exception as e:
        logger.error(f"LSTM Error: {e}")
        results["lstm_training"] = f"error: {str(e)}"

    try:
        # 3. Summary Update
        logger.info("Running Summary Update...")
        summary_service = SummaryUpdateService()
        summary_service.run_all()
        results["summary_update"] = "completed"
        logger.info("Summary update completed")
    except Exception as e:
        logger.error(f"Summary Error: {e}")
        results["summary_update"] = f"error: {str(e)}"

    results["status"] = "completed"

    # Store results in MongoDB
    mongo_storage.logs.insert_one({
        "type": "daily_job",
        "results": results,
        "timestamp": datetime.utcnow()
    })

    logger.info("All daily jobs completed")
    return results

if __name__ == "__main__":
    run_daily_jobs()
    