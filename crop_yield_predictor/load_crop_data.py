"""
Crop Data Loader - Loads crop data from MongoDB instead of local files
"""

import os
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from mongo_storage import mongo_storage
from config import Config
from datetime import datetime


class CropDataLoader:
    """Loads crop training data from MongoDB"""

    def __init__(self, farm_id: str, config_path: Optional[str] = None):
        self.farm_id = farm_id
        self.logger = logging.getLogger(__name__)

        # Use centralized config
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """Load configuration from centralized config"""
        default_config = {
            'data': {
                'required_columns': [
                    'farm_id', 'crop', 'season', 'year', 'window_num',
                    'start_date', 'end_date', 'is_season_end', 'yield'
                ],
                'weather_columns': [
                    'avg_temperature_2m_mean', 'avg_precipitation_sum',
                    'relative_humidity_2m_mean', 'shortwave_radiation_sum'
                ],
                'soil_columns': ['soil_pH', 'organic_matter_content'],
                'management_columns': ['plant_population_density', 'irrigation_type'],
                'categorical_columns': ['irrigation_type']
            }
        }
        return default_config

    def _validate_farm(self) -> bool:
        """Check if farm exists in MongoDB"""
        count = mongo_storage.crop_training_data.count_documents({
            "farm_id": self.farm_id
        })

        if count == 0:
            self.logger.warning(f"No crop training data found for farm_id={self.farm_id}")
            return False
        return True

    def load_crop_data(self, crop_name: str) -> Optional[pd.DataFrame]:
        """
        Load crop data from MongoDB
        """
        if not self._validate_farm():
            return None

        try:
            records = list(
                mongo_storage.crop_training_data.find(
                    {
                        "farm_id": self.farm_id,
                        "crop": {
                            "$regex": f"^{crop_name}$",
                            "$options": "i"
                        }
                    },
                    {"_id": 0}
                )
            )

            if not records:
                self.logger.warning(f"No crop data found for crop={crop_name}")
                return None

            df = pd.DataFrame(records)

            # Convert dates
            if 'start_date' in df.columns:
                df['start_date'] = pd.to_datetime(df['start_date'])

            if 'end_date' in df.columns:
                df['end_date'] = pd.to_datetime(df['end_date'])

            # Validate and standardize
            df = self._validate_data(df, crop_name)

            if df is None:
                return None

            # Add farm_id if missing
            if 'farm_id' not in df.columns:
                df['farm_id'] = self.farm_id

            # Sort data
            df = df.sort_values(['farm_id', 'crop', 'season', 'year', 'window_num'])

            # Calculate derived features
            df = self._calculate_features(df)

            self.logger.info(f"Loaded {len(df)} records for crop={crop_name}")
            return df

        except Exception as e:
            self.logger.error(f"Error loading crop data: {str(e)}", exc_info=True)
            return None

    def _validate_data(self, df: pd.DataFrame, crop_name: str) -> Optional[pd.DataFrame]:
        """Validate data structure and content"""
        required_cols = self.config['data']['required_columns']

        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            self.logger.error(f"Missing required columns: {missing_cols}")
            return None

        # Filter crop
        df = df[df['crop'].str.lower() == crop_name.lower()]

        if df.empty:
            self.logger.warning(f"No data found for crop: {crop_name}")
            return None

        # Validate seasonal sequence
        for (farm_id, crop, season, year), group in df.groupby(
            ['farm_id', 'crop', 'season', 'year']
        ):
            unique_windows = sorted(group['window_num'].unique())
            expected_windows = list(range(1, len(unique_windows) + 1))

            if unique_windows != expected_windows:
                self.logger.warning(
                    f"Window numbers not sequential for {farm_id}/{crop}/{season}/{year}. "
                    f"Found={unique_windows}, Expected={expected_windows}"
                )

        return df

    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived features"""
        df.columns = df.columns.str.strip()

        base_temp = 10

        # Growing Degree Days
        if 'avg_temperature_2m_mean' in df.columns:
            df['gdd'] = (
                df.groupby(['farm_id', 'crop', 'season', 'year'])['avg_temperature_2m_mean']
                .transform(lambda x: (x - base_temp).clip(lower=0).cumsum())
            )

        # Rolling weather features
        for window in [7, 14, 30]:
            if 'avg_temperature_2m_mean' in df.columns:
                df[f'temp_{window}d_avg'] = (
                    df.groupby(['farm_id', 'crop', 'season', 'year'])['avg_temperature_2m_mean']
                    .transform(lambda x: x.rolling(window, min_periods=1).mean())
                )

            if 'avg_precipitation_sum' in df.columns:
                df[f'precip_{window}d_sum'] = (
                    df.groupby(['farm_id', 'crop', 'season', 'year'])['avg_precipitation_sum']
                    .transform(lambda x: x.rolling(window, min_periods=1).sum())
                )

        # Growth stage
        if 'window_num' in df.columns:
            df['growth_stage'] = (
                df.groupby(['farm_id', 'crop', 'season', 'year'])['window_num']
                .transform(lambda x: x / x.max())
            )

        return df

    def get_seasonal_data(self, df: pd.DataFrame) -> Tuple[List[np.ndarray], List[float], List[Dict]]:
        """Prepare sequences for training"""
        sequences = []
        targets = []
        metadata = []

        # Convert categorical columns
        categorical_cols = self.config['data'].get('categorical_columns', [])

        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype('category').cat.codes

        # Group by season
        for (farm_id, crop, season, year), group in df.groupby(
            ['farm_id', 'crop', 'season', 'year']
        ):
            if not group['is_season_end'].any():
                continue

            final_yield = group.loc[group['is_season_end'], 'yield'].values[0]

            if pd.isna(final_yield):
                continue

            # Feature columns (exclude non-numeric and target)
            exclude_cols = ['farm_id', 'crop', 'season', 'year', 'window_num',
                           'start_date', 'end_date', 'is_season_end', 'yield']

            features = [col for col in df.columns if col not in exclude_cols and col in group.columns]

            # Numerical only
            group_features = group[features].select_dtypes(include=[np.number])

            # Build progressive sequences
            for i in range(1, len(group) + 1):
                current_seq = group_features.iloc[:i].values.astype(np.float32)
                sequences.append(current_seq)
                targets.append(float(final_yield))

                metadata.append({
                    'farm_id': farm_id,
                    'crop': crop,
                    'season': season,
                    'year': year,
                    'length': i,
                    'total_windows': len(group),
                    'start_date': group.iloc[0]['start_date'].isoformat() if 'start_date' in group.columns else None,
                    'end_date': group.iloc[i - 1]['end_date'].isoformat() if 'end_date' in group.columns else None
                })

        self.logger.info(f"Prepared {len(sequences)} sequences for training")
        return sequences, targets, metadata

    def save_crop_data(self, df: pd.DataFrame, crop_name: str):
        """Save crop data to MongoDB"""
        try:
            records = df.to_dict(orient="records")

            # Delete old data for this farm/crop
            mongo_storage.crop_training_data.delete_many({
                "farm_id": self.farm_id,
                "crop": crop_name
            })

            # Insert new data
            for record in records:
                record["farm_id"] = self.farm_id
                record["crop"] = crop_name
                record["created_at"] = datetime.utcnow()

                # Convert dates to string
                if "start_date" in record and record["start_date"]:
                    if hasattr(record["start_date"], 'isoformat'):
                        record["start_date"] = record["start_date"].isoformat()
                if "end_date" in record and record["end_date"]:
                    if hasattr(record["end_date"], 'isoformat'):
                        record["end_date"] = record["end_date"].isoformat()

            if records:
                mongo_storage.crop_training_data.insert_many(records)
                self.logger.info(f"Saved {len(records)} records for {self.farm_id}/{crop_name}")

        except Exception as e:
            self.logger.error(f"Failed to save crop data: {e}")
            raise
        