#!/usr/bin/env python3
"""
Test script for the fixed feature extractor.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from features.feature_extractor import TransactionFeatureExtractor

def test_feature_extractor():
    """Test the feature extractor with sample data."""
    print("🧪 Testing Feature Extractor")
    print("=" * 50)
    
    # Create sample data
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
    print(f"Sample data columns: {df.columns.tolist()}")
    print(f"Sample data:\n{df.head()}")
    
    # Initialize feature extractor
    extractor = TransactionFeatureExtractor()
    
    try:
        # Test basic features
        print("\n📊 Testing Basic Features...")
        df_basic = extractor.extract_basic_features(df)
        print(f"Basic features shape: {df_basic.shape}")
        print(f"New basic features: {[col for col in df_basic.columns if col not in df.columns]}")
        
        # Test user behavioral features
        print("\n👤 Testing User Behavioral Features...")
        df_user_behavioral = extractor.extract_user_behavioral_features(df_basic)
        print(f"User behavioral features shape: {df_user_behavioral.shape}")
        print(f"New user behavioral features: {[col for col in df_user_behavioral.columns if col not in df_basic.columns]}")
        
        # Test temporal features
        print("\n⏰ Testing Temporal Features...")
        df_temporal = extractor.extract_temporal_features(df_user_behavioral)
        print(f"Temporal features shape: {df_temporal.shape}")
        print(f"New temporal features: {[col for col in df_temporal.columns if col not in df_user_behavioral.columns]}")
        
        # Test contextual features
        print("\n🌍 Testing Contextual Features...")
        df_contextual = extractor.extract_contextual_features(df_temporal)
        print(f"Contextual features shape: {df_contextual.shape}")
        print(f"New contextual features: {[col for col in df_contextual.columns if col not in df_temporal.columns]}")
        
        # Test all features
        print("\n🚀 Testing All Features...")
        df_all = extractor.extract_all_features(df)
        print(f"All features shape: {df_all.shape}")
        print(f"Total features added: {df_all.shape[1] - df.shape[1]}")
        
        # Show feature categories
        print("\n📋 Feature Categories:")
        feature_categories = extractor.get_feature_names(df_all)
        for category, features in feature_categories.items():
            if features:
                print(f"  {category}: {len(features)} features")
                print(f"    Examples: {features[:3]}")
        
        print("\n✅ All tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_feature_extractor() 