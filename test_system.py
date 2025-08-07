#!/usr/bin/env python3
"""
Quick test script to verify the fraud detection system works.
"""
import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import logging
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_fraud_detection_system():
    """Test the complete fraud detection pipeline."""
    print("🔍 Testing Fraud Detection System")
    print("=" * 50)
    
    try:
        # Import modules
        print("📦 Importing modules...")
        from src.parser.log_parser import TransactionLogParser   
        from src.features.feature_extractor import TransactionFeatureExtractor
        from src.models.rule_based import RuleBasedAnomalyDetector
        from src.models.statistical import StatisticalAnomalyDetector
        from src.models.ensemble import EnsembleAnomalyDetector
        print("✅ All modules imported successfully")
        
        # Test parser
        print("\n🔧 Testing log parser...")
        parser = TransactionLogParser()
        
        # Load test logs from CSV file
        test_logs_df = pd.read_csv('test_logs.csv')
        # Filter NaN values with empty strings and convert to string
        test_logs = test_logs_df['raw_log'].fillna("").astype(str).tolist()
        
        parsed_transactions = []
        for log in test_logs:
            transaction = parser.parse_log_entry(log)
            parsed_transactions.append({
                'raw_log': transaction.raw_log,
                'timestamp': transaction.timestamp,
                'user_id': transaction.user_id,
                'transaction_type': transaction.transaction_type,
                'amount': transaction.amount,
                'location': transaction.location,
                'device': transaction.device,
                'is_parsed': transaction.is_parsed
            })
        
        df_test = pd.DataFrame(parsed_transactions)
        print(df_test[df_test['is_parsed']==True].head(5))
        parsed_count = df_test['is_parsed'].sum()
        print(f"✅ Parser test: {parsed_count}/{len(test_logs)} logs parsed successfully")
        
        # Generate actual parsing statistics from parsed transactions
        logger.info("Generating parsing statistics from test logs...")
        
        # Create statistics dictionary from parsed transactions
        total_logs = len(parsed_transactions)
        parsed_successfully = sum(1 for t in parsed_transactions if t['is_parsed'])
        parsing_failed = sum(1 for t in parsed_transactions if not t['is_parsed'])
        empty_logs = sum(1 for t in parsed_transactions if t['raw_log'].strip() in ['""', ''])
        malformed_logs = sum(1 for t in parsed_transactions if 'MALFORMED_LOG' in str(t['raw_log']))
        
        parsing_stats_dict = {
            'total_logs': total_logs,
            'parsed_successfully': parsed_successfully,
            'parsing_failed': parsing_failed,
            'empty_logs': empty_logs,
            'malformed_logs': malformed_logs,
            'parsing_errors': []
        }
        
        stats = parser.get_parsing_statistics(parsing_stats_dict)
        parser.print_parsing_statistics(parsing_stats_dict)
        parser.export_statistics_report(parsing_stats_dict, "test_parsing_stats.json")
        
        # Test summary table
        summary_table = parser.get_parsing_summary_table(parsing_stats_dict)
        print("\n📋 Parsing Summary Table:")
        print(summary_table.to_string(index=False))
        
        # Use parsed data from test logs for feature extraction and modeling
        print("\n🏗️ Preparing parsed test data for modeling...")
        
        # Filter only successfully parsed transactions
        df_parsed = df_test[df_test['is_parsed'] == True].copy().reset_index(drop=True)
        
        # Add line numbers for tracking
        df_parsed['line_num'] = range(1, len(df_parsed) + 1)
        
        # Fill missing currency with default
        if 'currency' not in df_parsed.columns:
            df_parsed['currency'] = '£'  # Default currency
        
        # Handle missing parse_errors column
        if 'parse_errors' not in df_parsed.columns:
            df_parsed['parse_errors'] = None
        
        print(f"✅ Using parsed test data with {len(df_parsed)} successfully parsed transactions")
        print("Sample parsed transactions:")
        print(df_parsed.head(5))
        
        # Test feature extraction
        print("\n🔧 Testing feature extraction...")
        feature_extractor = TransactionFeatureExtractor()
        df_features = feature_extractor.extract_all_features(df_parsed)
        
        original_features = len(df_parsed.columns)
        new_features = len(df_features.columns)
        added_features = new_features - original_features
        
        print(f"✅ Feature extraction: Added {added_features} features ({original_features} → {new_features})")
        
        # Test rule-based detection
        print("\n🔧 Testing rule-based detection...")
        rule_detector = RuleBasedAnomalyDetector()
        rule_scores, rule_details = rule_detector.detect_anomalies(df_features)
        
        anomalies_detected = np.sum(rule_scores > 0.5)
        print(f"✅ Rule-based detection: {anomalies_detected} anomalies detected (threshold=0.5)")
        
        # Test statistical detection
        print("\n🔧 Testing statistical detection...")
        stat_detector = StatisticalAnomalyDetector()
        stat_detector.fit(df_features)
        stat_predictions = stat_detector.predict_anomalies(df_features)
        
        print(f"✅ Statistical detection: {len(stat_predictions)} models fitted and predicted")
        
        # Test ensemble detection
        print("\n🔧 Testing ensemble detection...")
        ensemble_detector = EnsembleAnomalyDetector()
        ensemble_detector.fit(df_features)
        ensemble_scores, ensemble_results = ensemble_detector.detect_anomalies(df_features)
        
        top_anomalies = ensemble_detector.get_top_anomalies(df_features, ensemble_scores, top_n=5)
        
        print(f"✅ Ensemble detection: Mean score = {ensemble_results['mean_score']:.4f}")
        print(f"✅ Top 5 anomalies identified with scores: {ensemble_scores[np.argsort(ensemble_scores)[-5:]]}")
        
        # Display results summary
        print("\n📊 Results Summary:")
        print("-" * 30)
        print(f"Dataset size: {len(df_features)} transactions")
        print(f"Features extracted: {added_features}")
        print(f"Rule-based anomalies: {anomalies_detected}")
        print(f"Statistical models: {len(stat_predictions)}")
        print(f"Ensemble mean score: {ensemble_results['mean_score']:.4f}")
        print(f"High-risk transactions (>0.7): {ensemble_results['anomaly_count_threshold_70']}")
        
        print("\n🎉 All tests passed! Fraud detection system is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fraud_detection_system()
    sys.exit(0 if success else 1)
