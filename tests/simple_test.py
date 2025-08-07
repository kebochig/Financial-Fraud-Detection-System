#!/usr/bin/env python3
"""
Simple test to verify the fix for the feature extractor.
"""

import pandas as pd
import numpy as np
from datetime import datetime

def test_timestamp_processing():
    """Test the timestamp processing logic that was causing the error."""
    print("🧪 Testing Timestamp Processing Fix")
    print("=" * 50)
    
    # Create sample data similar to what would cause the error
    sample_data = {
        'user_id': ['user1', 'user1', 'user2', 'user2', 'user3'],
        'timestamp': ['2024-01-01 10:00:00', '2024-01-01 15:00:00', 
                     '2024-01-01 12:00:00', '2024-01-02 09:00:00', '2024-01-01 14:00:00'],
        'amount': [100.0, 200.0, 150.0, 300.0, 75.0],
        'transaction_type': ['purchase', 'withdrawal', 'transfer', 'deposit', 'purchase'],
        'location': ['London', 'London', 'Glasgow', 'Birmingham', 'Cardiff'],
        'device': ['iPhone 13', 'iPhone 13', 'Samsung Galaxy S10', 'Pixel 6', 'Nokia 3310'],
        'currency': ['GBP', 'GBP', 'GBP', 'GBP', 'GBP']
    }
    
    df = pd.DataFrame(sample_data)
    print(f"Sample data shape: {df.shape}")
    print(f"Sample data:\n{df.head()}")
    
    try:
        # Test the problematic logic step by step
        
        # 1. Amount statistics
        print("\n📊 Testing Amount Statistics...")
        amount_stats = df.groupby('user_id')['amount'].agg([
            'count', 'mean', 'std', 'min', 'max', 'sum'
        ]).round(4)
        print(f"Amount stats shape: {amount_stats.shape}")
        print(f"Amount stats:\n{amount_stats}")
        
        # 2. Timestamp statistics (the problematic part)
        print("\n⏰ Testing Timestamp Statistics...")
        timestamp_stats = df.groupby('user_id')['timestamp'].agg(['min', 'max'])
        timestamp_stats.columns = ['user_first_tx_time', 'user_last_tx_time']
        print(f"Timestamp stats shape: {timestamp_stats.shape}")
        print(f"Timestamp stats before conversion:\n{timestamp_stats}")
        
        # 3. Convert timestamp columns to datetime (the fix)
        print("\n🔧 Testing Timestamp Conversion...")
        timestamp_stats['user_first_tx_time'] = pd.to_datetime(timestamp_stats['user_first_tx_time'], errors='coerce')
        timestamp_stats['user_last_tx_time'] = pd.to_datetime(timestamp_stats['user_last_tx_time'], errors='coerce')
        print(f"Timestamp stats after conversion:\n{timestamp_stats}")
        
        # 4. Nunique statistics
        print("\n🔢 Testing Nunique Statistics...")
        location_nunique = df.groupby('user_id')['location'].nunique().rename('user_unique_locations')
        device_nunique = df.groupby('user_id')['device'].nunique().rename('user_unique_devices')
        type_nunique = df.groupby('user_id')['transaction_type'].nunique().rename('user_unique_types')
        currency_nunique = df.groupby('user_id')['currency'].nunique().rename('user_unique_currencies')
        
        print(f"Location unique: {location_nunique}")
        print(f"Device unique: {device_nunique}")
        print(f"Type unique: {type_nunique}")
        print(f"Currency unique: {currency_nunique}")
        
        # 5. Combine all statistics
        print("\n🔗 Testing Statistics Combination...")
        user_stats = amount_stats.join([
            timestamp_stats, 
            location_nunique, 
            device_nunique, 
            type_nunique, 
            currency_nunique
        ])
        print(f"Combined stats shape: {user_stats.shape}")
        print(f"Combined stats:\n{user_stats}")
        
        # 6. Calculate activity days
        print("\n📅 Testing Activity Days Calculation...")
        user_stats['user_activity_days'] = (
            user_stats['user_last_tx_time'] - user_stats['user_first_tx_time']
        ).dt.total_seconds() / (24 * 3600)
        print(f"Activity days:\n{user_stats['user_activity_days']}")
        
        # 7. Calculate transactions per day
        print("\n📈 Testing Transactions Per Day...")
        user_stats['user_tx_per_day'] = (
            user_stats['user_tx_count'] / (user_stats['user_activity_days'] + 1)
        ).round(4)
        print(f"Transactions per day:\n{user_stats['user_tx_per_day']}")
        
        # 8. Calculate risk indicators
        print("\n⚠️ Testing Risk Indicators...")
        user_stats['user_amount_cv'] = (
            user_stats['user_std_amount'] / user_stats['user_avg_amount']
        ).fillna(0).round(4)
        
        user_stats['user_location_diversity'] = (
            user_stats['user_unique_locations'] / user_stats['user_tx_count']
        ).round(4)
        
        user_stats['user_device_diversity'] = (
            user_stats['user_unique_devices'] / user_stats['user_tx_count']
        ).round(4)
        
        print(f"Risk indicators:\n{user_stats[['user_amount_cv', 'user_location_diversity', 'user_device_diversity']]}")
        
        # 9. Final merge test
        print("\n🔀 Testing Final Merge...")
        df_features = df.copy()
        df_features = df_features.merge(user_stats, left_on='user_id', right_index=True, how='left')
        print(f"Final features shape: {df_features.shape}")
        print(f"New columns: {[col for col in df_features.columns if col not in df.columns]}")
        
        print("\n✅ All tests passed! The fix is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_timestamp_processing() 