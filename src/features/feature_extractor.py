"""
Feature extraction module for transaction data.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from ..utils.config import config

logger = logging.getLogger(__name__)

class TransactionFeatureExtractor:
    """Extract structured features from parsed transaction data."""
    
    def __init__(self):
        """Initialize feature extractor with configuration."""
        self.temporal_window_hours = config.get('features.temporal_window_hours', 24)
        self.location_risk_threshold = config.get('features.location_risk_threshold', 0.1)
        self.amount_zscore_threshold = config.get('features.amount_zscore_threshold', 2.0)
        
        # Feature categories
        self.numerical_features = config.get('features.numerical_features', [])
        self.categorical_features = config.get('features.categorical_features', [])
        self.user_features = config.get('features.user_features', [])
        
        logger.info("Feature extractor initialized")
    
    def extract_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract basic features from parsed transaction data."""
        logger.info("Extracting basic features...")
        
        df_features = df.copy()
        
        # Temporal features
        if 'timestamp' in df.columns:
            df_features['hour'] = df['timestamp'].dt.hour
            df_features['day_of_week'] = df['timestamp'].dt.dayofweek
            df_features['is_weekend'] = df_features['day_of_week'].isin([5, 6])
            df_features['month'] = df['timestamp'].dt.month
            df_features['day_of_month'] = df['timestamp'].dt.day
            
            # Hour categories
            df_features['hour_category'] = pd.cut(
                df_features['hour'], 
                bins=[0, 6, 12, 18, 24], 
                labels=['night', 'morning', 'afternoon', 'evening'],
                include_lowest=True
            )
        
        # Amount features
        if 'amount' in df.columns:
            df_features['amount_log'] = np.log1p(df_features['amount'])
            df_features['amount_rounded'] = np.round(df_features['amount'], -1)  # Round to nearest 10
            
            # Amount categories
            amount_quantiles = df_features['amount'].quantile([0.25, 0.5, 0.75]).values
            df_features['amount_category'] = pd.cut(
                df_features['amount'],
                bins=[0] + list(amount_quantiles) + [float('inf')],
                labels=['low', 'medium', 'high', 'very_high'],
                include_lowest=True
            )
        
        # Currency features
        if 'currency' in df.columns:
            df_features['has_currency'] = df_features['currency'].notna()
            df_features['currency_filled'] = df_features['currency'].fillna('unknown')
        
        # Location features
        if 'location' in df.columns:
            df_features['has_location'] = df_features['location'].notna()
            df_features['location_filled'] = df_features['location'].fillna('unknown')
        
        # Device features
        if 'device' in df.columns:
            df_features['has_device'] = df_features['device'].notna()
            df_features['device_filled'] = df_features['device'].fillna('unknown')
            
            # Device type categories
            device_mapping = {
                'iPhone 13': 'iPhone',
                'Samsung Galaxy S10': 'Samsung',
                'Pixel 6': 'Pixel',
                'Nokia 3310': 'Nokia',
                'Xiaomi Mi 11': 'Xiaomi',
                'Huawei P30': 'Huawei',
                'None': 'unknown'
            }
            df_features['device_brand'] = df_features['device_filled'].map(device_mapping)
        
        # Transaction type features
        if 'transaction_type' in df.columns:
            df_features['transaction_type_filled'] = df_features['transaction_type'].fillna('unknown')
            
            # Group transaction types
            cash_transactions = ['withdrawal', 'cashout', 'debit']
            digital_transactions = ['transfer', 'top-up', 'purchase']
            account_transactions = ['deposit', 'refund']
            
            df_features['transaction_group'] = df_features['transaction_type_filled'].apply(
                lambda x: 'cash' if x in cash_transactions 
                         else 'digital' if x in digital_transactions
                         else 'account' if x in account_transactions
                         else 'other'
            )
        
        logger.info(f"Basic features extracted. Shape: {df_features.shape}")
        return df_features
    
    def extract_user_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract user behavioral features."""
        logger.info("Extracting user behavioral features...")
        
        if 'user_id' not in df.columns:
            logger.warning("No user_id column found. Skipping user behavioral features.")
            return df
        
        df_features = df.copy()
        
        # Calculate different statistics separately to avoid column conflicts
        
        # Amount statistics
        amount_stats = df.groupby('user_id')['amount'].agg([
            'count', 'mean', 'std', 'min', 'max', 'sum'
        ]).round(4)
        
        # Rename amount stat columns
        amount_stats.columns = [
            'user_tx_count', 'user_avg_amount', 'user_std_amount', 
            'user_min_amount', 'user_max_amount', 'user_total_amount'
        ]
        
        # Timestamp statistics
        timestamp_stats = df.groupby('user_id')['timestamp'].agg(['min', 'max'])
        timestamp_stats.columns = ['user_first_tx_time', 'user_last_tx_time']
        
        # Nunique statistics with proper names
        location_nunique = df.groupby('user_id')['location'].nunique().rename('user_unique_locations')
        device_nunique = df.groupby('user_id')['device'].nunique().rename('user_unique_devices')
        type_nunique = df.groupby('user_id')['transaction_type'].nunique().rename('user_unique_types')
        currency_nunique = df.groupby('user_id')['currency'].nunique().rename('user_unique_currencies')
        
        # Combine all statistics
        user_stats = amount_stats.join([
            timestamp_stats, 
            location_nunique, 
            device_nunique, 
            type_nunique, 
            currency_nunique
        ])
        
        # User activity timespan
        user_stats['user_activity_days'] = (
            user_stats['user_last_tx_time'] - user_stats['user_first_tx_time']
        ).dt.total_seconds() / (24 * 3600)
        
        # Average transactions per day
        user_stats['user_tx_per_day'] = (
            user_stats['user_tx_count'] / (user_stats['user_activity_days'] + 1)
        ).round(4)
        
        # Risk indicators
        user_stats['user_amount_cv'] = (
            user_stats['user_std_amount'] / user_stats['user_avg_amount']
        ).fillna(0).round(4)  # Coefficient of variation
        
        user_stats['user_location_diversity'] = (
            user_stats['user_unique_locations'] / user_stats['user_tx_count']
        ).round(4)
        
        user_stats['user_device_diversity'] = (
            user_stats['user_unique_devices'] / user_stats['user_tx_count']
        ).round(4)
        
        # Fill NaN values
        user_stats = user_stats.fillna(0)
        
        # Merge back to main dataframe
        df_features = df_features.merge(user_stats, left_on='user_id', right_index=True, how='left')
        
        logger.info(f"User behavioral features extracted. Added {len(user_stats.columns)} features.")
        return df_features
    
    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal and sequence-based features."""
        logger.info("Extracting temporal features...")
        
        if 'timestamp' not in df.columns or 'user_id' not in df.columns:
            logger.warning("Missing timestamp or user_id. Skipping temporal features.")
            return df
        
        df_features = df.copy().sort_values(['user_id', 'timestamp'])
        
        # Time since last transaction (per user)
        df_features['time_since_last_tx'] = (
            df_features.groupby('user_id')['timestamp']
            .diff()
            .dt.total_seconds() / 3600  # Convert to hours
        ).fillna(0)
        
        # Time until next transaction (per user)
        df_features['time_until_next_tx'] = (
            df_features.groupby('user_id')['timestamp']
            .shift(-1) - df_features['timestamp']
        ).dt.total_seconds().fillna(0) / 3600
        
        # Transaction velocity (transactions in last N hours)
        window_hours = self.temporal_window_hours
        
        def count_recent_transactions(group):
            timestamps = group['timestamp']
            counts = []
            for ts in timestamps:
                window_start = ts - timedelta(hours=window_hours)
                recent_count = ((timestamps >= window_start) & (timestamps < ts)).sum()
                counts.append(recent_count)
            return pd.Series(counts, index=group.index)
        
        df_features['tx_velocity_24h'] = (
            df_features.groupby('user_id')
            .apply(count_recent_transactions)
            .reset_index(level=0, drop=True)
        )
        
        # Location changes
        df_features['location_changed'] = (
            df_features.groupby('user_id')['location']
            .shift(1) != df_features['location']
        ).astype(int)
        
        # Device changes
        df_features['device_changed'] = (
            df_features.groupby('user_id')['device']
            .shift(1) != df_features['device']
        ).astype(int)
        
        # Amount deviation from user's pattern
        df_features['amount_zscore_user'] = (
            df_features.groupby('user_id')['amount']
            .transform(lambda x: (x - x.mean()) / np.maximum(x.std(), 1e-8))
        ).round(4)
        
        # Time patterns
        df_features['is_unusual_hour'] = (
            (df_features['hour'] < 6) | (df_features['hour'] > 22)
        ).astype(int)
        
        df_features['is_night_transaction'] = (
            (df_features['hour'] >= 0) & (df_features['hour'] < 6)
        ).astype(int)
        
        logger.info("Temporal features extracted.")
        return df_features
    
    def extract_contextual_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract contextual and interaction features."""
        logger.info("Extracting contextual features...")
        
        df_features = df.copy()
        
        # Location-based features
        if 'location' in df.columns:
            # Location frequency
            location_counts = df_features['location'].value_counts()
            df_features['location_frequency'] = df_features['location'].map(location_counts)
            df_features['location_rarity'] = 1 / (df_features['location_frequency'] + 1)
            
            # Location risk score (based on rarity)
            df_features['location_risk_score'] = (
                df_features['location_rarity'] > self.location_risk_threshold
            ).astype(int)
        
        # Device-based features
        if 'device' in df.columns:
            # Device frequency
            device_counts = df_features['device'].value_counts()
            df_features['device_frequency'] = df_features['device'].map(device_counts)
            df_features['device_rarity'] = 1 / (df_features['device_frequency'] + 1)
        
        # Transaction type patterns
        if 'transaction_type' in df.columns:
            type_counts = df_features['transaction_type'].value_counts()
            df_features['type_frequency'] = df_features['transaction_type'].map(type_counts)
            df_features['type_rarity'] = 1 / (df_features['type_frequency'] + 1)
        
        # Interaction features
        if all(col in df_features.columns for col in ['location', 'device']):
            # Location-Device combinations
            df_features['location_device'] = (
                df_features['location'].astype(str) + '_' + df_features['device'].astype(str)
            )
            loc_dev_counts = df_features['location_device'].value_counts()
            df_features['location_device_frequency'] = df_features['location_device'].map(loc_dev_counts)
            df_features['location_device_rarity'] = 1 / (df_features['location_device_frequency'] + 1)
        
        if all(col in df_features.columns for col in ['hour', 'transaction_type']):
            # Hour-Transaction type combinations
            df_features['hour_type'] = (
                df_features['hour'].astype(str) + '_' + df_features['transaction_type'].astype(str)
            )
            hour_type_counts = df_features['hour_type'].value_counts()
            df_features['hour_type_frequency'] = df_features['hour_type'].map(hour_type_counts)
            df_features['hour_type_rarity'] = 1 / (df_features['hour_type_frequency'] + 1)
        
        # Amount-based contextual features
        if 'amount' in df.columns:
            # Global amount percentiles
            df_features['amount_percentile'] = df_features['amount'].rank(pct=True)
            
            # Amount deviation from location average
            location_amount_mean = df_features.groupby('location')['amount'].transform('mean')
            df_features['amount_deviation_location'] = (
                df_features['amount'] - location_amount_mean
            ).abs()
            
            # Amount deviation from transaction type average
            type_amount_mean = df_features.groupby('transaction_type')['amount'].transform('mean')
            df_features['amount_deviation_type'] = (
                df_features['amount'] - type_amount_mean
            ).abs()
        
        logger.info("Contextual features extracted.")
        return df_features
    
    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract all features in the correct order."""
        logger.info("Starting comprehensive feature extraction...")
        
        # Validate input
        if df.empty:
            raise ValueError("Input DataFrame is empty")
        
        required_columns = ['timestamp', 'user_id', 'amount', 'transaction_type']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"Missing recommended columns: {missing_columns}")
        
        # Extract features in stages
        df_features = self.extract_basic_features(df)
        df_features = self.extract_user_behavioral_features(df_features)
        df_features = self.extract_temporal_features(df_features)
        df_features = self.extract_contextual_features(df_features)
        
        # Add feature metadata
        df_features['feature_extraction_timestamp'] = datetime.now()
        
        logger.info(f"Feature extraction complete. Final shape: {df_features.shape}")
        logger.info(f"Added {df_features.shape[1] - df.shape[1]} new features")
        
        return df_features
    
    def get_feature_names(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Get categorized feature names."""
        all_columns = set(df.columns)
        
        # Original columns (not features)
        original_columns = {
            'line_num', 'raw_log', 'timestamp', 'user_id', 'transaction_type',
            'amount', 'currency', 'location', 'device', 'is_parsed', 'parse_errors'
        }
        
        # Feature categories
        feature_categories = {
            'temporal': [col for col in all_columns if any(term in col.lower() 
                        for term in ['hour', 'day', 'weekend', 'month', 'time', 'night'])],
            'amount': [col for col in all_columns if 'amount' in col.lower()],
            'user': [col for col in all_columns if col.startswith('user_')],
            'location': [col for col in all_columns if 'location' in col.lower()],
            'device': [col for col in all_columns if 'device' in col.lower()],
            'transaction': [col for col in all_columns if any(term in col.lower() 
                           for term in ['transaction', 'type', 'tx'])],
            'contextual': [col for col in all_columns if any(term in col.lower() 
                          for term in ['frequency', 'rarity', 'risk', 'deviation', 'percentile'])],
            'interaction': [col for col in all_columns if '_' in col and 
                           any(term in col for term in ['location_device', 'hour_type'])],
            'derived': [col for col in all_columns if any(term in col.lower() 
                       for term in ['category', 'group', 'brand', 'filled', 'changed', 'zscore'])]
        }
        
        # Remove original columns from feature categories
        for category in feature_categories:
            feature_categories[category] = [
                col for col in feature_categories[category] 
                if col not in original_columns
            ]
        
        # Add uncategorized features
        all_features = set()
        for features in feature_categories.values():
            all_features.update(features)
        
        uncategorized = list(all_columns - original_columns - all_features)
        if uncategorized:
            feature_categories['other'] = uncategorized
        
        return feature_categories
