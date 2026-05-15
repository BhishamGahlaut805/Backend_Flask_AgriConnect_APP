import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import mutual_info_regression
from typing import Tuple, List, Dict
import warnings

class FeatureEngineer:
    def __init__(self, config: dict):
        self.config = config
        self.feature_columns = self._get_feature_columns()
        self.target_column = 'yield'
        self.static_features = ['soil_type', 'irrigation_type', 'tillage_type']

    def _get_feature_columns(self) -> List[str]:
        """Return all possible feature columns with categorization"""
        return {
            'soil': ['soil_pH', 'organic_matter_content'],
            'weather': [
                'avg_temperature_2m_mean', 'avg_precipitation_sum',
                'temp_7d_avg', 'temp_14d_avg', 'temp_30d_avg',
                'precip_7d_sum', 'precip_14d_sum', 'precip_30d_sum',
                'relative_humidity_2m_mean'
            ],
            'nutrients': ['n', 'p', 'k', 'total_npk'],
            'management': ['plant_population_density'],
            'derived': ['gdd', 'growth_stage']
        }

    def preprocess(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """Main preprocessing pipeline with feature importance"""
        df = self._clean_data(df)
        df = self._calculate_features(df)
        df, scalers = self._normalize_features(df)

        # Calculate feature importance
        importance = self._calculate_feature_importance(df)

        return df, {'scalers': scalers, 'importance': importance}

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values and data validation"""
        # Ensure required columns exist
        required_cols = ['farm_id', 'crop', 'season', 'start_date', 'window_num']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Forward fill static features within each season
        group_cols = ['farm_id', 'crop', 'season']
        for col in self.static_features:
            if col in df.columns:
                df[col] = df.groupby(group_cols)[col].ffill().bfill()

        # Fill numerical features with group means
        num_cols = [col for category in self.feature_columns.values()
                   for col in category if col in df.columns]
        df[num_cols] = df.groupby(group_cols)[num_cols].transform(
            lambda x: x.fillna(x.mean())
        )

        return df

    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived features and growth stages"""
        # Growing Degree Days (GDD)
        base_temp = self.config.get('gdd_base_temp', 10)
        df['gdd'] = (df['avg_temperature_2m_mean'] - base_temp).clip(lower=0)
        df['gdd'] = df.groupby(['farm_id', 'crop', 'season'])['gdd'].cumsum()

        # Rolling weather features
        for window in [7, 14, 30]:
            df[f'temp_{window}d_avg'] = df.groupby(['farm_id', 'crop', 'season'])[
                'avg_temperature_2m_mean'].rolling(window, min_periods=1).mean().reset_index(level=[0,1,2], drop=True)
            df[f'precip_{window}d_sum'] = df.groupby(['farm_id', 'crop', 'season'])[
                'avg_precipitation_sum'].rolling(window, min_periods=1).sum().reset_index(level=[0,1,2], drop=True)

        # Growth stage estimation (0-1 scale)
        df['growth_stage'] = df.groupby(['farm_id', 'crop', 'season'])['window_num'].transform(
            lambda x: x / x.max()
        )

        return df

    def _normalize_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """Normalize features with appropriate scalers"""
        scalers = {}

        # Use StandardScaler for weather features
        weather_cols = [col for col in self.feature_columns['weather'] if col in df.columns]
        weather_scaler = StandardScaler()
        df[weather_cols] = weather_scaler.fit_transform(df[weather_cols])
        scalers.update({col: weather_scaler for col in weather_cols})

        # Use MinMaxScaler for other numerical features
        other_cols = [
            col for category in ['soil', 'nutrients', 'management', 'derived']
            for col in self.feature_columns[category] if col in df.columns
        ]
        for col in other_cols:
            scaler = MinMaxScaler()
            df[col] = scaler.fit_transform(df[[col]])
            scalers[col] = scaler

        return df, scalers

    def _calculate_feature_importance(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate mutual information feature importance"""
        # Only calculate on complete cases
        temp_df = df[df['is_season_end']].copy()

        # Get all available features
        features = [col for category in self.feature_columns.values()
                   for col in category if col in temp_df.columns]

        if not features or len(temp_df) < 2:
            return {}

        # Calculate mutual information
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mi = mutual_info_regression(
                temp_df[features],
                temp_df[self.target_column],
                random_state=42
            )

        # Convert to percentages
        importance = {col: float(percent)
                     for col, percent in zip(features, mi / mi.sum() * 100)}

        return importance

    def prepare_sequences(self, df: pd.DataFrame) -> Tuple[list, list, list]:
        """
        Prepare sequences for training with variable lengths
        Returns:
            sequences: List of numpy arrays (variable length)
            targets: List of yield values
            sequence_info: List of dicts with metadata about each sequence
        """
        sequences = []
        targets = []
        sequence_info = []

        # Group by farm-crop-season combinations
        for (farm_id, crop, season), group in df.groupby(['farm_id', 'crop', 'season']):
            # Sort by window_num and get features
            group = group.sort_values('window_num')

            # Only use groups that have yield information
            if not group['is_season_end'].any():
                continue

            # Get the final yield value
            final_yield = group.loc[group['is_season_end'], self.target_column].values[0]
            if np.isnan(final_yield):
                continue

            # Get all available features
            features = [col for category in self.feature_columns.values()
                       for col in category if col in group.columns]

            # Create sequences of increasing length
            for i in range(1, len(group)+1):
                current_seq = group.iloc[:i][features].values
                sequences.append(current_seq)
                targets.append(final_yield)

                # Store sequence metadata
                sequence_info.append({
                    'farm_id': farm_id,
                    'crop': crop,
                    'season': season,
                    'length': i,
                    'total_windows': len(group),
                    'start_date': group.iloc[0]['start_date'],
                    'end_date': group.iloc[i-1]['end_date'],
                    'is_final': (i == len(group))
                })

        return sequences, targets, sequence_info
    