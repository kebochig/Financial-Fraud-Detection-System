#!/usr/bin/env python3
"""
Test script to demonstrate StatisticalAnomalyDetector methods and show top 5 users with their scores.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from features.feature_extractor import TransactionFeatureExtractor
from models.statistical import StatisticalAnomalyDetector

def test_statistical_methods():
    """Test all statistical methods and show top 5 users with their scores."""
    print("🧪 Testing Statistical Anomaly Detection Methods")
    print("=" * 60)
    
    # Create sample transaction data
    sample_data = {
        'user_id': [
            'user1', 'user1', 'user1', 'user1', 'user1',
            'user2', 'user2', 'user2', 'user2', 'user2',
            'user3', 'user3', 'user3', 'user3', 'user3',
            'user4', 'user4', 'user4', 'user4', 'user4',
            'user5', 'user5', 'user5', 'user5', 'user5',
            'user6', 'user6', 'user6', 'user6', 'user6',
            'user7', 'user7', 'user7', 'user7', 'user7',
            'user8', 'user8', 'user8', 'user8', 'user8',
            'user9', 'user9', 'user9', 'user9', 'user9',
            'user10', 'user10', 'user10', 'user10', 'user10'
        ],
        'timestamp': [
            '2024-01-01 10:00:00', '2024-01-01 15:00:00', '2024-01-02 09:00:00', '2024-01-02 14:00:00', '2024-01-03 11:00:00',
            '2024-01-01 12:00:00', '2024-01-01 18:00:00', '2024-01-02 10:00:00', '2024-01-02 16:00:00', '2024-01-03 13:00:00',
            '2024-01-01 14:00:00', '2024-01-01 20:00:00', '2024-01-02 12:00:00', '2024-01-02 18:00:00', '2024-01-03 15:00:00',
            '2024-01-01 16:00:00', '2024-01-01 22:00:00', '2024-01-02 14:00:00', '2024-01-02 20:00:00', '2024-01-03 17:00:00',
            '2024-01-01 18:00:00', '2024-01-02 00:00:00', '2024-01-02 16:00:00', '2024-01-02 22:00:00', '2024-01-03 19:00:00',
            '2024-01-01 20:00:00', '2024-01-02 02:00:00', '2024-01-02 18:00:00', '2024-01-03 00:00:00', '2024-01-03 21:00:00',
            '2024-01-01 22:00:00', '2024-01-02 04:00:00', '2024-01-02 20:00:00', '2024-01-03 02:00:00', '2024-01-03 23:00:00',
            '2024-01-02 00:00:00', '2024-01-02 06:00:00', '2024-01-02 22:00:00', '2024-01-03 04:00:00', '2024-01-04 01:00:00',
            '2024-01-02 02:00:00', '2024-01-02 08:00:00', '2024-01-03 00:00:00', '2024-01-03 06:00:00', '2024-01-04 03:00:00',
            '2024-01-02 04:00:00', '2024-01-02 10:00:00', '2024-01-03 02:00:00', '2024-01-03 08:00:00', '2024-01-04 05:00:00'
        ],
        'amount': [
            100.0, 200.0, 150.0, 300.0, 250.0,  # user1 - normal amounts
            50.0, 75.0, 60.0, 80.0, 90.0,       # user2 - low amounts
            500.0, 750.0, 600.0, 800.0, 900.0,   # user3 - high amounts (suspicious)
            25.0, 30.0, 35.0, 40.0, 45.0,       # user4 - very low amounts
            1000.0, 1500.0, 1200.0, 1800.0, 2000.0,  # user5 - very high amounts (very suspicious)
            120.0, 180.0, 140.0, 220.0, 160.0,   # user6 - normal amounts
            75.0, 85.0, 95.0, 105.0, 115.0,      # user7 - normal amounts
            300.0, 400.0, 350.0, 450.0, 500.0,   # user8 - medium-high amounts
            45.0, 55.0, 65.0, 75.0, 85.0,        # user9 - low-medium amounts
            2000.0, 2500.0, 3000.0, 3500.0, 4000.0  # user10 - extremely high amounts (most suspicious)
        ],
        'transaction_type': [
            'purchase', 'withdrawal', 'transfer', 'purchase', 'withdrawal',
            'purchase', 'purchase', 'purchase', 'purchase', 'purchase',
            'withdrawal', 'withdrawal', 'withdrawal', 'withdrawal', 'withdrawal',
            'purchase', 'purchase', 'purchase', 'purchase', 'purchase',
            'transfer', 'transfer', 'transfer', 'transfer', 'transfer',
            'purchase', 'withdrawal', 'transfer', 'purchase', 'withdrawal',
            'purchase', 'purchase', 'purchase', 'purchase', 'purchase',
            'withdrawal', 'withdrawal', 'withdrawal', 'withdrawal', 'withdrawal',
            'purchase', 'purchase', 'purchase', 'purchase', 'purchase',
            'transfer', 'transfer', 'transfer', 'transfer', 'transfer'
        ],
        'location': [
            'London', 'London', 'London', 'London', 'London',
            'Glasgow', 'Glasgow', 'Glasgow', 'Glasgow', 'Glasgow',
            'Birmingham', 'Birmingham', 'Birmingham', 'Birmingham', 'Birmingham',
            'Liverpool', 'Liverpool', 'Liverpool', 'Liverpool', 'Liverpool',
            'Cardiff', 'Cardiff', 'Cardiff', 'Cardiff', 'Cardiff',
            'Leeds', 'Leeds', 'Leeds', 'Leeds', 'Leeds',
            'Manchester', 'Manchester', 'Manchester', 'Manchester', 'Manchester',
            'London', 'Glasgow', 'Birmingham', 'Liverpool', 'Cardiff',  # user8 - multiple locations
            'Leeds', 'Leeds', 'Leeds', 'Leeds', 'Leeds',
            'London', 'London', 'London', 'London', 'London'  # user10 - high amounts in London
        ],
        'device': [
            'iPhone 13', 'iPhone 13', 'iPhone 13', 'iPhone 13', 'iPhone 13',
            'Samsung Galaxy S10', 'Samsung Galaxy S10', 'Samsung Galaxy S10', 'Samsung Galaxy S10', 'Samsung Galaxy S10',
            'Pixel 6', 'Pixel 6', 'Pixel 6', 'Pixel 6', 'Pixel 6',
            'Nokia 3310', 'Nokia 3310', 'Nokia 3310', 'Nokia 3310', 'Nokia 3310',
            'Xiaomi Mi 11', 'Xiaomi Mi 11', 'Xiaomi Mi 11', 'Xiaomi Mi 11', 'Xiaomi Mi 11',
            'Huawei P30', 'Huawei P30', 'Huawei P30', 'Huawei P30', 'Huawei P30',
            'iPhone 13', 'iPhone 13', 'iPhone 13', 'iPhone 13', 'iPhone 13',
            'iPhone 13', 'Samsung Galaxy S10', 'Pixel 6', 'Nokia 3310', 'Xiaomi Mi 11',  # user8 - multiple devices
            'Huawei P30', 'Huawei P30', 'Huawei P30', 'Huawei P30', 'Huawei P30',
            'iPhone 13', 'iPhone 13', 'iPhone 13', 'iPhone 13', 'iPhone 13'
        ],
        'currency': ['GBP'] * 50
    }
    
    df = pd.DataFrame(sample_data)
    print(f"Sample data shape: {df.shape}")
    print(f"Sample data:\n{df.head()}")
    
    try:
        # Extract features
        print("\n📊 Extracting Features...")
        feature_extractor = TransactionFeatureExtractor()
        df_features = feature_extractor.extract_all_features(df)
        print(f"Features extracted. Shape: {df_features.shape}")
        
        # Initialize and fit statistical detector
        print("\n🔧 Initializing Statistical Anomaly Detector...")
        statistical_detector = StatisticalAnomalyDetector()
        statistical_detector.fit(df_features)
        
        # Get predictions from all methods
        print("\n🔍 Getting Predictions from All Methods...")
        predictions = statistical_detector.predict_anomalies(df_features)
        
        # Create results DataFrame
        results_df = df_features[['user_id', 'amount', 'transaction_type', 'location']].copy()
        
        # Add scores from each method
        for method, scores in predictions.items():
            results_df[f'{method}_score'] = scores
        
        # Show top 5 users for each method
        print("\n" + "=" * 60)
        print("🏆 TOP 5 USERS BY ANOMALY SCORE FOR EACH METHOD")
        print("=" * 60)
        
        for method in predictions.keys():
            print(f"\n📈 {method.upper().replace('_', ' ')}")
            print("-" * 40)
            
            # Group by user and get mean score for each method
            user_scores = results_df.groupby('user_id')[f'{method}_score'].agg(['mean', 'count']).round(4)
            user_scores = user_scores.sort_values('mean', ascending=False)
            
            # Get top 5 users
            top_5_users = user_scores.head(5)
            
            print(f"{'User':<10} {'Mean Score':<12} {'Tx Count':<10} {'Risk Level':<12}")
            print("-" * 45)
            
            for user_id, row in top_5_users.iterrows():
                score = row['mean']
                tx_count = row['count']
                
                # Determine risk level
                if score > 0.8:
                    risk_level = "CRITICAL"
                elif score > 0.6:
                    risk_level = "HIGH"
                elif score > 0.4:
                    risk_level = "MEDIUM"
                elif score > 0.2:
                    risk_level = "LOW"
                else:
                    risk_level = "NORMAL"
                
                print(f"{user_id:<10} {score:<12.4f} {tx_count:<10} {risk_level:<12}")
        
        # Show ensemble scores
        print(f"\n🎯 ENSEMBLE SCORES (Weighted Average)")
        print("-" * 40)
        
        # Calculate ensemble scores
        ensemble_scores = statistical_detector.detect_anomalies_ensemble(df_features)
        results_df['ensemble_score'] = ensemble_scores
        
        # Group by user and get mean ensemble score
        user_ensemble_scores = results_df.groupby('user_id')['ensemble_score'].agg(['mean', 'count']).round(4)
        user_ensemble_scores = user_ensemble_scores.sort_values('mean', ascending=False)
        
        print(f"{'User':<10} {'Ensemble Score':<15} {'Tx Count':<10} {'Risk Level':<12}")
        print("-" * 50)
        
        for user_id, row in user_ensemble_scores.head(5).iterrows():
            score = row['mean']
            tx_count = row['count']
            
            # Determine risk level
            if score > 0.8:
                risk_level = "CRITICAL"
            elif score > 0.6:
                risk_level = "HIGH"
            elif score > 0.4:
                risk_level = "MEDIUM"
            elif score > 0.2:
                risk_level = "LOW"
            else:
                risk_level = "NORMAL"
            
            print(f"{user_id:<10} {score:<15.4f} {tx_count:<10} {risk_level:<12}")
        
        # Show method comparison
        print(f"\n📊 METHOD COMPARISON SUMMARY")
        print("-" * 40)
        
        method_stats = {}
        for method, scores in predictions.items():
            method_stats[method] = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'max_score': np.max(scores),
                'high_risk_count': np.sum(scores > 0.6)
            }
        
        print(f"{'Method':<20} {'Mean':<8} {'Std':<8} {'Max':<8} {'High Risk':<10}")
        print("-" * 55)
        
        for method, stats in method_stats.items():
            print(f"{method:<20} {stats['mean_score']:<8.4f} {stats['std_score']:<8.4f} "
                  f"{stats['max_score']:<8.4f} {stats['high_risk_count']:<10}")
        
        print("\n✅ Statistical methods analysis complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_statistical_methods() 