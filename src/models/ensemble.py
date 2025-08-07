"""
Ensemble anomaly detection combining multiple approaches.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from ..utils.config import config
from .rule_based import RuleBasedAnomalyDetector
from .statistical import StatisticalAnomalyDetector

logger = logging.getLogger(__name__)

class EnsembleAnomalyDetector:
    """Ensemble fraud detection system combining multiple approaches."""
    
    def __init__(self):
        """Initialize ensemble detector with configuration."""
        
        # Load ensemble configuration
        self.ensemble_config = config.get('models.ensemble', {
            'rule_weight': 0.3,
            'isolation_weight': 0.25,
            'autoencoder_weight': 0.25,
            'clustering_weight': 0.2
        })
        
        # Initialize component detectors
        self.rule_detector = RuleBasedAnomalyDetector()
        self.statistical_detector = StatisticalAnomalyDetector()
        
        # Ensemble state
        self.is_fitted = False
        self.component_scores = {}
        self.feature_importance = {}
        
        logger.info("Ensemble anomaly detector initialized")
    
    def fit(self, df: pd.DataFrame) -> 'EnsembleAnomalyDetector':
        """Fit all component detectors."""
        logger.info(f"Fitting ensemble detector on {len(df)} samples...")
        
        # Fit statistical models
        self.statistical_detector.fit(df)
        
        # Rule-based detector doesn't need fitting
        logger.info("✅ Rule-based detector ready")
        
        self.is_fitted = True
        logger.info("✅ Ensemble detector fitting complete")
        return self
    
    def detect_anomalies(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Detect anomalies using ensemble approach."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before detection")
        
        logger.info(f"Running ensemble anomaly detection on {len(df)} samples...")
        
        # Get predictions from all components
        component_results = {}
        
        # Rule-based detection
        logger.info("🔍 Running rule-based detection...")
        rule_scores, rule_details = self.rule_detector.detect_anomalies(df)
        component_results['rule_based'] = {
            'scores': rule_scores,
            'details': rule_details
        }
        
        # Statistical detection
        logger.info("📊 Running statistical detection...")
        statistical_predictions = self.statistical_detector.predict_anomalies(df)
        component_results['statistical'] = {
            'scores': statistical_predictions,
            'details': statistical_predictions
        }
        
        # Combine scores using weighted ensemble
        ensemble_scores = self._combine_scores(component_results)
        
        # Store component scores for analysis
        self.component_scores = component_results
        
        # Create ensemble results
        ensemble_results = {
            'component_scores': component_results,
            'ensemble_weights': self.ensemble_config,
            'mean_score': float(np.mean(ensemble_scores)),
            'std_score': float(np.std(ensemble_scores)),
            'anomaly_count_threshold_70': int(np.sum(ensemble_scores > 0.7)),
            'anomaly_count_threshold_80': int(np.sum(ensemble_scores > 0.8)),
            'anomaly_count_threshold_90': int(np.sum(ensemble_scores > 0.9))
        }
        
        # Generate comprehensive summary report
        self._print_ensemble_summary(df, ensemble_scores, ensemble_results)

        # Export report to file
        self._export_fraud_report(df, ensemble_scores, ensemble_results)
        
        logger.info(f"Ensemble detection complete. Mean score: {ensemble_results['mean_score']:.4f}")
        return ensemble_scores, ensemble_results
    
    def _combine_scores(self, component_results: Dict[str, Any]) -> np.ndarray:
        """Combine component scores using weighted ensemble."""
        logger.info("🔗 Combining component scores...")
        
        # Get rule-based scores
        rule_scores = component_results['rule_based']['scores']
        
        # Get statistical scores (use isolation forest as primary)
        statistical_scores = component_results['statistical']['scores']
        isolation_scores = statistical_scores.get('isolation_forest', np.zeros(len(rule_scores)))
        
        # Get clustering scores (use DBSCAN as primary)
        clustering_scores = statistical_scores.get('dbscan', np.zeros(len(rule_scores)))
        
        # For autoencoder, we'll use LOF as a proxy for now
        # In a full implementation, this would be a separate autoencoder model
        autoencoder_scores = statistical_scores.get('lof', np.zeros(len(rule_scores)))
        
        # Weighted combination
        ensemble_scores = (
            rule_scores * self.ensemble_config['rule_weight'] +
            isolation_scores * self.ensemble_config['isolation_weight'] +
            autoencoder_scores * self.ensemble_config['autoencoder_weight'] +
            clustering_scores * self.ensemble_config['clustering_weight']
        )
        
        # Normalize to [0, 1] range
        ensemble_scores = np.clip(ensemble_scores, 0, 1)
        
        logger.info(f"Ensemble combination complete. Score range: [{np.min(ensemble_scores):.3f}, {np.max(ensemble_scores):.3f}]")
        return ensemble_scores
    
    def _print_ensemble_summary(self, df: pd.DataFrame, ensemble_scores: np.ndarray, ensemble_results: Dict[str, Any]):
        """Print comprehensive ensemble summary report."""
        print("\n" + "="*100)
        print("🚨 ENSEMBLE FRAUD DETECTION SYSTEM - COMPREHENSIVE REPORT")
        print("="*100)
        
        # Overall Statistics
        print(f"\n📊 OVERALL STATISTICS")
        print("-" * 50)
        print(f"📈 Total Transactions Analyzed: {len(df):,}")
        print(f"🎯 Mean Ensemble Score: {ensemble_results['mean_score']:.4f}")
        print(f"📏 Score Standard Deviation: {ensemble_results['std_score']:.4f}")
        print(f"🔝 Highest Score: {np.max(ensemble_scores):.4f}")
        print(f"🔻 Lowest Score: {np.min(ensemble_scores):.4f}")
        
        # Risk Level Breakdown with User Details
        print(f"\n⚠️ RISK LEVEL BREAKDOWN")
        print("-" * 50)
        critical_count = np.sum(ensemble_scores > 0.8)
        high_count = np.sum((ensemble_scores > 0.7) & (ensemble_scores <= 0.8))
        medium_count = np.sum((ensemble_scores > 0.5) & (ensemble_scores <= 0.7))
        low_count = np.sum((ensemble_scores > 0.3) & (ensemble_scores <= 0.5))
        normal_count = np.sum(ensemble_scores <= 0.3)
        
        print(f"🔥 CRITICAL RISK (>0.8): {critical_count:,} transactions ({critical_count/len(df)*100:.1f}%)")
        print(f"⚠️ HIGH RISK (0.7-0.8): {high_count:,} transactions ({high_count/len(df)*100:.1f}%)")
        print(f"⚡ MEDIUM RISK (0.5-0.7): {medium_count:,} transactions ({medium_count/len(df)*100:.1f}%)")
        print(f"📊 LOW RISK (0.3-0.5): {low_count:,} transactions ({low_count/len(df)*100:.1f}%)")
        print(f"✅ NORMAL (<0.3): {normal_count:,} transactions ({normal_count/len(df)*100:.1f}%)")
        
        # User breakdown by risk category
        if 'user_id' in df.columns:
            print(f"\n👥 USERS BY RISK CATEGORY")
            print("-" * 50)
            
            # Create DataFrame for analysis
            risk_df = pd.DataFrame({
                'user_id': df['user_id'],
                'ensemble_score': ensemble_scores,
                'amount': df.get('amount', [0] * len(df)),
                'location': df.get('location', ['Unknown'] * len(df)),
                'transaction_type': df.get('transaction_type', ['Unknown'] * len(df))
            })
            
            # Critical Risk Users
            critical_users = risk_df[risk_df['ensemble_score'] > 0.8]
            if len(critical_users) > 0:
                print(f"\n🔥 CRITICAL RISK USERS ({len(critical_users['user_id'].unique())} unique users):")
                critical_user_summary = critical_users.groupby('user_id').agg({
                    'ensemble_score': ['mean', 'max', 'count'],
                    'amount': ['mean', 'sum'],
                    'location': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
                }).round(4)
                critical_user_summary.columns = ['mean_score', 'max_score', 'tx_count', 'avg_amount', 'total_amount', 'main_location']
                critical_user_summary = critical_user_summary.sort_values('mean_score', ascending=False)
                
                for user_id, row in critical_user_summary.head(10).iterrows():
                    print(f"   • {user_id}: Score {row['mean_score']:.4f} (Max: {row['max_score']:.4f}), "
                          f"{int(row['tx_count'])} tx, £{row['total_amount']:.0f} total, {row['main_location']}")
            
            # High Risk Users
            high_users = risk_df[(risk_df['ensemble_score'] > 0.7) & (risk_df['ensemble_score'] <= 0.8)]
            if len(high_users) > 0:
                print(f"\n⚠️ HIGH RISK USERS ({len(high_users['user_id'].unique())} unique users):")
                high_user_summary = high_users.groupby('user_id').agg({
                    'ensemble_score': ['mean', 'max', 'count'],
                    'amount': ['mean', 'sum'],
                    'location': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
                }).round(4)
                high_user_summary.columns = ['mean_score', 'max_score', 'tx_count', 'avg_amount', 'total_amount', 'main_location']
                high_user_summary = high_user_summary.sort_values('mean_score', ascending=False)
                
                for user_id, row in high_user_summary.head(10).iterrows():
                    print(f"   • {user_id}: Score {row['mean_score']:.4f} (Max: {row['max_score']:.4f}), "
                          f"{int(row['tx_count'])} tx, £{row['total_amount']:.0f} total, {row['main_location']}")
            
            # Medium Risk Users
            medium_users = risk_df[(risk_df['ensemble_score'] > 0.5) & (risk_df['ensemble_score'] <= 0.7)]
            if len(medium_users) > 0:
                print(f"\n⚡ MEDIUM RISK USERS ({len(medium_users['user_id'].unique())} unique users):")
                medium_user_summary = medium_users.groupby('user_id').agg({
                    'ensemble_score': ['mean', 'max', 'count'],
                    'amount': ['mean', 'sum'],
                    'location': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
                }).round(4)
                medium_user_summary.columns = ['mean_score', 'max_score', 'tx_count', 'avg_amount', 'total_amount', 'main_location']
                medium_user_summary = medium_user_summary.sort_values('mean_score', ascending=False)
                
                for user_id, row in medium_user_summary.head(5).iterrows():
                    print(f"   • {user_id}: Score {row['mean_score']:.4f} (Max: {row['max_score']:.4f}), "
                          f"{int(row['tx_count'])} tx, £{row['total_amount']:.0f} total, {row['main_location']}")
                if len(medium_user_summary) > 5:
                    print(f"   ... and {len(medium_user_summary) - 5} more users")
        
        # Component Analysis
        print(f"\n🔍 COMPONENT ANALYSIS")
        print("-" * 50)
        rule_scores = ensemble_results['component_scores']['rule_based']['scores']
        statistical_scores = ensemble_results['component_scores']['statistical']['scores']
        
        print(f"📋 Rule-Based Detection:")
        print(f"   • Mean Score: {np.mean(rule_scores):.4f}")
        print(f"   • Weight: {self.ensemble_config['rule_weight']:.2f}")
        print(f"   • High Risk (>0.7): {np.sum(rule_scores > 0.7):,} transactions")
        
        print(f"\n📊 Statistical Detection:")
        for method, scores in statistical_scores.items():
            if isinstance(scores, np.ndarray):
                print(f"   • {method.replace('_', ' ').title()}: {np.mean(scores):.4f} (High Risk: {np.sum(scores > 0.7):,})")
        
        # Top Anomalies by User
        if 'user_id' in df.columns:
            print(f"\n👤 TOP ANOMALOUS USERS")
            print("-" * 50)
            
            # Create user summary
            user_summary = pd.DataFrame({
                'user_id': df['user_id'],
                'ensemble_score': ensemble_scores,
                'rule_score': rule_scores,
                'amount': df.get('amount', [0] * len(df)),
                'location': df.get('location', ['Unknown'] * len(df)),
                'transaction_type': df.get('transaction_type', ['Unknown'] * len(df))
            })
            
            # Group by user
            user_analysis = user_summary.groupby('user_id').agg({
                'ensemble_score': ['mean', 'max', 'count'],
                'rule_score': 'mean',
                'amount': ['mean', 'sum'],
                'location': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown',
                'transaction_type': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
            }).round(4)
            
            # Flatten column names
            user_analysis.columns = ['mean_ensemble', 'max_ensemble', 'tx_count', 'mean_rule', 
                                   'avg_amount', 'total_amount', 'main_location', 'main_type']
            
            # Sort by maximum ensemble score
            top_users = user_analysis.sort_values('max_ensemble', ascending=False).head(10)
            
            print(f"{'Rank':<4} {'User':<12} {'Max Ensemble':<12} {'Rule':<8} {'Tx Count':<10} {'Avg Amount':<12} {'Total':<12} {'Location':<12} {'Type':<12}")
            print("-" * 100)
            
            for rank, (user_id, row) in enumerate(top_users.iterrows(), 1):
                max_ensemble = row['max_ensemble']
                mean_rule = row['mean_rule']
                tx_count = int(row['tx_count'])
                avg_amount = row['avg_amount']
                total_amount = row['total_amount']
                location = row['main_location']
                tx_type = row['main_type']
                
                # Risk level based on max score
                if max_ensemble > 0.8:
                    risk = "🔥 CRITICAL"
                elif max_ensemble > 0.7:
                    risk = "⚠️ HIGH"
                elif max_ensemble > 0.5:
                    risk = "⚡ MEDIUM"
                elif max_ensemble > 0.3:
                    risk = "📊 LOW"
                else:
                    risk = "✅ NORMAL"
                
                print(f"{rank:<4} {user_id:<12} {max_ensemble:<12.4f} {mean_rule:<8.4f} {tx_count:<10} "
                      f"{avg_amount:<12.1f} {total_amount:<12.1f} {location:<12} {tx_type:<12} {risk}")
        
        # Actionable Insights
        print(f"\n🎯 ACTIONABLE INSIGHTS & RECOMMENDATIONS")
        print("-" * 50)
        
        # Critical alerts
        if critical_count > 0:
            print(f"🚨 IMMEDIATE ACTION REQUIRED:")
            print(f"   • {critical_count} transactions flagged as CRITICAL risk")
            print(f"   • Recommend: BLOCK these transactions immediately")
            print(f"   • Review user accounts associated with critical scores")
        
        # High risk alerts
        if high_count > 0:
            print(f"\n⚠️ HIGH PRIORITY REVIEW:")
            print(f"   • {high_count} transactions flagged as HIGH risk")
            print(f"   • Recommend: MANUAL REVIEW required")
            print(f"   • Consider additional verification for these users")
        
        # Pattern analysis
        if 'user_id' in df.columns:
            # Find users with multiple high-risk transactions
            high_risk_users = user_summary[user_summary['ensemble_score'] > 0.7].groupby('user_id').size()
            if len(high_risk_users) > 0:
                print(f"\n🔍 SUSPICIOUS PATTERNS DETECTED:")
                print(f"   • {len(high_risk_users)} users have multiple high-risk transactions")
                print(f"   • Top suspicious user: {high_risk_users.index[0]} ({high_risk_users.iloc[0]} high-risk tx)")
        
        # System performance
        print(f"\n⚙️ SYSTEM PERFORMANCE:")
        print(f"   • Ensemble model successfully processed {len(df):,} transactions")
        print(f"   • Detection rate: {(critical_count + high_count)/len(df)*100:.1f}% of transactions flagged")
        print(f"   • Model confidence: {'HIGH' if ensemble_results['std_score'] > 0.1 else 'MEDIUM' if ensemble_results['std_score'] > 0.05 else 'LOW'}")
        
        # Next steps
        print(f"\n📋 RECOMMENDED NEXT STEPS:")
        if critical_count > 0:
            print(f"   1. 🔒 BLOCK all critical risk transactions immediately")
        if high_count > 0:
            print(f"   2. 👀 MANUAL REVIEW of high-risk transactions")
        if 'user_id' in df.columns and len(user_summary) > 0:
            print(f"   3. 👤 INVESTIGATE top 5 suspicious users")
        print(f"   4. 📊 MONITOR system performance and adjust thresholds if needed")
        print(f"   5. 🔄 RETRAIN models with new data if detection rate is too high/low")
        
        print("\n" + "="*100)
    
    def _export_fraud_report(self, df: pd.DataFrame, ensemble_scores: np.ndarray, ensemble_results: Dict[str, Any]):
        """Export comprehensive fraud report to file with timestamp."""
        import os
        from datetime import datetime
        
        # Create timestamp for filename
        timestamp = datetime.now().strftime("%d-%m-%y-%H:%M")
        filename = f"fraud_report_{timestamp}.txt"
        
        # Create reports directory if it doesn't exist
        reports_dir = "results"
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir)
        
        filepath = os.path.join(reports_dir, filename)
        
        # Capture all output in a string
        import io
        import sys
        
        # Redirect stdout to capture the report
        old_stdout = sys.stdout
        report_output = io.StringIO()
        sys.stdout = report_output
        
        # Re-run the summary without the export call to avoid recursion
        self._print_ensemble_summary_content(df, ensemble_scores, ensemble_results)
        
        # Get the captured output
        report_content = report_output.getvalue()
        sys.stdout = old_stdout
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report_content)
            f.write(f"\n📄 Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"📊 Total transactions processed: {len(df):,}\n")
            f.write(f"🎯 Ensemble model version: {self.__class__.__name__}\n")
        
        print(f"\n💾 FRAUD REPORT EXPORTED: {filepath}")
        print(f"📄 Report contains comprehensive analysis of {len(df):,} transactions")
        
        # Also export CSV with detailed user data if available
        if 'user_id' in df.columns:
            csv_filename = f"fraud_report_detailed_{timestamp}.csv"
            csv_filepath = os.path.join(reports_dir, csv_filename)
            
            # Create detailed CSV export
            detailed_df = pd.DataFrame({
                'user_id': df['user_id'],
                'ensemble_score': ensemble_scores,
                'rule_score': ensemble_results['component_scores']['rule_based']['scores'],
                'amount': df.get('amount', [0] * len(df)),
                'location': df.get('location', ['Unknown'] * len(df)),
                'transaction_type': df.get('transaction_type', ['Unknown'] * len(df)),
                'risk_level': pd.cut(ensemble_scores, 
                                   bins=[0, 0.3, 0.5, 0.7, 0.9, 1.0],
                                   labels=['NORMAL', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'])
            })
            
            detailed_df.to_csv(csv_filepath, index=False)
            print(f"📊 DETAILED CSV EXPORTED: {csv_filepath}")
    
    def _print_ensemble_summary_content(self, df: pd.DataFrame, ensemble_scores: np.ndarray, ensemble_results: Dict[str, Any]):
        """Print ensemble summary content without export functionality (to avoid recursion)."""
        print("\n" + "="*100)
        print("🚨 ENSEMBLE FRAUD DETECTION SYSTEM - COMPREHENSIVE REPORT")
        print("="*100)
        
        # Overall Statistics
        print(f"\n📊 OVERALL STATISTICS")
        print("-" * 50)
        print(f"📈 Total Transactions Analyzed: {len(df):,}")
        print(f"🎯 Mean Ensemble Score: {ensemble_results['mean_score']:.4f}")
        print(f"📏 Score Standard Deviation: {ensemble_results['std_score']:.4f}")
        print(f"🔝 Highest Score: {np.max(ensemble_scores):.4f}")
        print(f"🔻 Lowest Score: {np.min(ensemble_scores):.4f}")
        
        # Risk Level Breakdown with User Details
        print(f"\n⚠️ RISK LEVEL BREAKDOWN")
        print("-" * 50)
        critical_count = np.sum(ensemble_scores > 0.8)
        high_count = np.sum((ensemble_scores > 0.7) & (ensemble_scores <= 0.8))
        medium_count = np.sum((ensemble_scores > 0.5) & (ensemble_scores <= 0.7))
        low_count = np.sum((ensemble_scores > 0.3) & (ensemble_scores <= 0.5))
        normal_count = np.sum(ensemble_scores <= 0.3)
        
        print(f"🔥 CRITICAL RISK (>0.8): {critical_count:,} transactions ({critical_count/len(df)*100:.1f}%)")
        print(f"⚠️ HIGH RISK (0.7-0.8): {high_count:,} transactions ({high_count/len(df)*100:.1f}%)")
        print(f"⚡ MEDIUM RISK (0.5-0.7): {medium_count:,} transactions ({medium_count/len(df)*100:.1f}%)")
        print(f"📊 LOW RISK (0.3-0.5): {low_count:,} transactions ({low_count/len(df)*100:.1f}%)")
        print(f"✅ NORMAL (<0.3): {normal_count:,} transactions ({normal_count/len(df)*100:.1f}%)")
        
        # User breakdown by risk category
        if 'user_id' in df.columns:
            print(f"\n👥 USERS BY RISK CATEGORY")
            print("-" * 50)
            
            # Create DataFrame for analysis
            risk_df = pd.DataFrame({
                'user_id': df['user_id'],
                'ensemble_score': ensemble_scores,
                'amount': df.get('amount', [0] * len(df)),
                'location': df.get('location', ['Unknown'] * len(df)),
                'transaction_type': df.get('transaction_type', ['Unknown'] * len(df))
            })
                        
            # Critical Risk Users
            critical_users = risk_df[risk_df['ensemble_score'] > 0.8]
            if len(critical_users) > 0:
                print(f"\n🔥 CRITICAL RISK USERS ({len(critical_users['user_id'].unique())} unique users):")
                critical_users = risk_df[risk_df['user_id'].isin(critical_users['user_id'])]
                critical_user_summary = critical_users.groupby('user_id').agg({
                    'ensemble_score': ['mean', 'max', 'count'],
                    'amount': ['mean', 'sum'],
                    'location': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
                }).round(4)
                critical_user_summary.columns = ['mean_score', 'max_score', 'tx_count', 'avg_amount', 'total_amount', 'main_location']
                critical_user_summary = critical_user_summary.sort_values('mean_score', ascending=False)
                
                for user_id, row in critical_user_summary.head(10).iterrows():
                    print(f"   • {user_id}: Score {row['mean_score']:.4f} (Max: {row['max_score']:.4f}), "
                          f"{int(row['tx_count'])} tx, £{row['total_amount']:.0f} total, {row['main_location']}")

            # High Risk Users
            high_users = risk_df[(risk_df['ensemble_score'] > 0.7) & (risk_df['ensemble_score'] <= 0.8)]
            if len(high_users) > 0:
                print(f"\n⚠️ HIGH RISK USERS ({len(high_users['user_id'].unique())} unique users):")
                high_users = risk_df[risk_df['user_id'].isin(high_users['user_id'])]
                high_user_summary = high_users.groupby('user_id').agg({
                    'ensemble_score': ['mean', 'max', 'count'],
                    'amount': ['mean', 'sum'],
                    'location': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
                }).round(4)
                high_user_summary.columns = ['mean_score', 'max_score', 'tx_count', 'avg_amount', 'total_amount', 'main_location']
                high_user_summary = high_user_summary.sort_values('mean_score', ascending=False)
                
                for user_id, row in high_user_summary.head(10).iterrows():
                    print(f"   • {user_id}: Score {row['mean_score']:.4f} (Max: {row['max_score']:.4f}), "
                          f"{int(row['tx_count'])} tx, £{row['total_amount']:.0f} total, {row['main_location']}")
            
            # Medium Risk Users
            medium_users = risk_df[(risk_df['ensemble_score'] > 0.5) & (risk_df['ensemble_score'] <= 0.7)]
            if len(medium_users) > 0:
                print(f"\n⚡ MEDIUM RISK USERS ({len(medium_users['user_id'].unique())} unique users):")
                medium_user_summary = medium_users.groupby('user_id').agg({
                    'ensemble_score': ['mean', 'max', 'count'],
                    'amount': ['mean', 'sum'],
                    'location': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
                }).round(4)
                medium_user_summary.columns = ['mean_score', 'max_score', 'tx_count', 'avg_amount', 'total_amount', 'main_location']
                medium_user_summary = medium_user_summary.sort_values('mean_score', ascending=False)
                
                for user_id, row in medium_user_summary.head(5).iterrows():
                    print(f"   • {user_id}: Score {row['mean_score']:.4f} (Max: {row['max_score']:.4f}), "
                          f"{int(row['tx_count'])} tx, £{row['total_amount']:.0f} total, {row['main_location']}")
                if len(medium_user_summary) > 5:
                    print(f"   ... and {len(medium_user_summary) - 5} more users")
        
        # Component Analysis
        print(f"\n🔍 COMPONENT ANALYSIS")
        print("-" * 50)
        rule_scores = ensemble_results['component_scores']['rule_based']['scores']
        statistical_scores = ensemble_results['component_scores']['statistical']['scores']
        
        print(f"📋 Rule-Based Detection:")
        print(f"   • Mean Score: {np.mean(rule_scores):.4f}")
        print(f"   • Weight: {self.ensemble_config['rule_weight']:.2f}")
        print(f"   • High Risk (>0.7): {np.sum(rule_scores > 0.7):,} transactions")
        
        # print(f"\n📊 Statistical Detection:")
        # for method, scores in statistical_scores.items():
        #     if isinstance(scores, np.ndarray):
        #         print(f"   • {method.replace('_', ' ').title()}: {np.mean(scores):.4f} (High Risk: {np.sum(scores > 0.7):,})")
        
        # Top Anomalies by User
        if 'user_id' in df.columns:
            print(f"\n👤 TOP ANOMALOUS USERS")
            print("-" * 50)
            
            # Create user summary
            user_summary = pd.DataFrame({
                'user_id': df['user_id'],
                'ensemble_score': ensemble_scores,
                'rule_score': rule_scores,
                'amount': df.get('amount', [0] * len(df)),
                'location': df.get('location', ['Unknown'] * len(df)),
                'transaction_type': df.get('transaction_type', ['Unknown'] * len(df))
            })
            
            # Group by user
            user_analysis = user_summary.groupby('user_id').agg({
                'ensemble_score': ['mean', 'max', 'count'],
                'rule_score': 'mean',
                'amount': ['mean', 'sum'],
                'location': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown',
                'transaction_type': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
            }).round(4)
            
            # Flatten column names
            user_analysis.columns = ['mean_ensemble', 'max_ensemble', 'tx_count', 'mean_rule', 
                                   'avg_amount', 'total_amount', 'main_location', 'main_type']
            
            # Sort by mean ensemble score
            top_users = user_analysis.sort_values('mean_ensemble', ascending=False).head(10)
            
            print(f"{'Rank':<4} {'User':<12} {'Ensemble':<10} {'Rule':<8} {'Tx Count':<10} {'Avg Amount':<12} {'Total':<12} {'Location':<12} {'Type':<12}")
            print("-" * 100)
            
            for rank, (user_id, row) in enumerate(top_users.iterrows(), 1):
                mean_ensemble = row['mean_ensemble']
                mean_rule = row['mean_rule']
                tx_count = int(row['tx_count'])
                avg_amount = row['avg_amount']
                total_amount = row['total_amount']
                location = row['main_location']
                tx_type = row['main_type']
                
                # Risk level
                if mean_ensemble > 0.8:
                    risk = "🔥 CRITICAL"
                elif mean_ensemble > 0.7:
                    risk = "⚠️ HIGH"
                elif mean_ensemble > 0.5:
                    risk = "⚡ MEDIUM"
                elif mean_ensemble > 0.3:
                    risk = "📊 LOW"
                else:
                    risk = "✅ NORMAL"
                
                print(f"{rank:<4} {user_id:<12} {mean_ensemble:<10.4f} {mean_rule:<8.4f} {tx_count:<10} "
                      f"{avg_amount:<12.1f} {total_amount:<12.1f} {location:<12} {tx_type:<12} {risk}")
        
        # Actionable Insights
        print(f"\n🎯 ACTIONABLE INSIGHTS & RECOMMENDATIONS")
        print("-" * 50)
        
        # Critical alerts
        if critical_count > 0:
            print(f"🚨 IMMEDIATE ACTION REQUIRED:")
            print(f"   • {critical_count} transactions flagged as CRITICAL risk")
            print(f"   • Recommend: BLOCK these transactions immediately")
            print(f"   • Review user accounts associated with critical scores")
        
        # High risk alerts
        if high_count > 0:
            print(f"\n⚠️ HIGH PRIORITY REVIEW:")
            print(f"   • {high_count} transactions flagged as HIGH risk")
            print(f"   • Recommend: MANUAL REVIEW required")
            print(f"   • Consider additional verification for these users")
        
        # Pattern analysis
        if 'user_id' in df.columns:
            # Find users with multiple high-risk transactions
            high_risk_users = user_summary[user_summary['ensemble_score'] > 0.7].groupby('user_id').size()
            if len(high_risk_users) > 0:
                print(f"\n🔍 SUSPICIOUS PATTERNS DETECTED:")
                print(f"   • {len(high_risk_users)} users have multiple high-risk transactions")
                print(f"   • Top suspicious user: {high_risk_users.index[0]} ({high_risk_users.iloc[0]} high-risk tx)")
        
        # System performance
        print(f"\n⚙️ SYSTEM PERFORMANCE:")
        print(f"   • Ensemble model successfully processed {len(df):,} transactions")
        print(f"   • Detection rate: {(critical_count + high_count)/len(df)*100:.1f}% of transactions flagged")
        print(f"   • Model confidence: {'HIGH' if ensemble_results['std_score'] > 0.1 else 'MEDIUM' if ensemble_results['std_score'] > 0.05 else 'LOW'}")
        
        # Next steps
        print(f"\n📋 RECOMMENDED NEXT STEPS:")
        if critical_count > 0:
            print(f"   1. 🔒 BLOCK all critical risk transactions immediately")
        if high_count > 0:
            print(f"   2. 👀 MANUAL REVIEW of high-risk transactions")
        if 'user_id' in df.columns and len(user_summary) > 0:
            print(f"   3. 👤 INVESTIGATE top 5 suspicious users")
        print(f"   4. 📊 MONITOR system performance and adjust thresholds if needed")
        print(f"   5. 🔄 RETRAIN models with new data if detection rate is too high/low")
    
    def explain_anomalies(self, df: pd.DataFrame, ensemble_scores: np.ndarray, 
                         top_n: int = 10) -> List[Dict[str, Any]]:
        """Explain top anomalies with contributions from each component."""
        logger.info(f"Explaining top {top_n} anomalies...")
        
        # Get top anomaly indices
        top_indices = np.argsort(ensemble_scores)[-top_n:][::-1]
        
        explanations = []
        
        for idx in top_indices:
            explanation = {
                'index': int(idx),
                'ensemble_score': float(ensemble_scores[idx]),
                'transaction_details': df.iloc[idx].to_dict(),
                'component_contributions': {},
                'rule_explanations': None,
                'statistical_details': {}
            }
            
            # Get rule-based explanation
            if 'rule_based' in self.component_scores:
                rule_explanation = self.rule_detector.explain_anomaly(
                    df, idx, self.component_scores['rule_based']['details']
                )
                explanation['rule_explanations'] = rule_explanation
                explanation['component_contributions']['rule_based'] = {
                    'score': float(self.component_scores['rule_based']['scores'][idx]),
                    'weight': self.ensemble_config['rule_weight'],
                    'contribution': float(self.component_scores['rule_based']['scores'][idx] * 
                                        self.ensemble_config['rule_weight'])
                }
            
            # Get statistical contributions
            if 'statistical' in self.component_scores:
                statistical_scores = self.component_scores['statistical']['scores']
                
                for method, scores in statistical_scores.items():
                    method_weight = self._get_statistical_weight(method)
                    explanation['component_contributions'][f'statistical_{method}'] = {
                        'score': float(scores[idx]),
                        'weight': method_weight,
                        'contribution': float(scores[idx] * method_weight)
                    }
                    
                    explanation['statistical_details'][method] = float(scores[idx])
            
            explanations.append(explanation)
        
        logger.info(f"Generated explanations for {len(explanations)} anomalies")
        return explanations
    
    def _get_statistical_weight(self, method: str) -> float:
        """Get weight for statistical method based on ensemble configuration."""
        method_weights = {
            'isolation_forest': self.ensemble_config['isolation_weight'],
            'lof': self.ensemble_config['autoencoder_weight'],  # Using LOF as proxy
            'dbscan': self.ensemble_config['clustering_weight'],
            'hdbscan': self.ensemble_config['clustering_weight'] * 0.5,
            'one_class_svm': self.ensemble_config['isolation_weight'] * 0.5
        }
        return method_weights.get(method, 0.1)
    
    def get_feature_importance(self, df: pd.DataFrame) -> Dict[str, float]:
        """Get combined feature importance from all components."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before getting feature importance")
        
        logger.info("Calculating ensemble feature importance...")
        
        combined_importance = {}
        
        # Get statistical feature importance
        try:
            stat_importance = self.statistical_detector.get_feature_importance(df)
            
            # Weight by ensemble configuration
            for feature, importance in stat_importance.items():
                combined_importance[feature] = importance * self.ensemble_config['isolation_weight']
        
        except Exception as e:
            logger.warning(f"Could not get statistical feature importance: {str(e)}")
        
        # Rule-based feature importance is implicit in rule definitions
        # We can estimate it based on rule weights and feature usage
        rule_feature_importance = self._estimate_rule_feature_importance()
        
        # Combine with rule-based importance
        for feature, importance in rule_feature_importance.items():
            if feature in combined_importance:
                combined_importance[feature] += importance * self.ensemble_config['rule_weight']
            else:
                combined_importance[feature] = importance * self.ensemble_config['rule_weight']
        
        # Normalize importance scores
        total_importance = sum(combined_importance.values())
        if total_importance > 0:
            combined_importance = {k: v / total_importance for k, v in combined_importance.items()}
        
        # Sort by importance
        sorted_importance = dict(sorted(combined_importance.items(), 
                                      key=lambda x: x[1], reverse=True))
        
        logger.info(f"Calculated importance for {len(sorted_importance)} features")
        return sorted_importance
    
    def _estimate_rule_feature_importance(self) -> Dict[str, float]:
        """Estimate feature importance based on rule definitions."""
        # This maps rules to the features they primarily use
        rule_feature_mapping = {
            'large_amount': ['amount', 'amount_log', 'amount_category'],
            'unusual_hours': ['hour', 'is_unusual_hour', 'is_night_transaction'],
            'high_velocity': ['tx_velocity_24h', 'user_tx_count'],
            'unusual_location': ['location_rarity', 'location_risk_score'],
            'device_change': ['device_changed', 'user_device_diversity'],
            'amount_deviation': ['amount_zscore_user', 'user_amount_cv'],
            'weekend_activity': ['is_weekend', 'amount'],
            'rare_transaction_type': ['type_rarity', 'transaction_type'],
            'multiple_locations': ['user_unique_locations', 'location_diversity'],
            'round_amounts': ['amount', 'amount_rounded']
        }
        
        feature_importance = {}
        
        for rule_name, rule_config in self.rule_detector.rules.items():
            rule_weight = rule_config['weight']
            features = rule_feature_mapping.get(rule_name, [])
            
            # Distribute rule weight among its features
            feature_weight = rule_weight / len(features) if features else 0
            
            for feature in features:
                if feature in feature_importance:
                    feature_importance[feature] += feature_weight
                else:
                    feature_importance[feature] = feature_weight
        
        return feature_importance
    
    def get_ensemble_statistics(self, df: pd.DataFrame, ensemble_scores: np.ndarray) -> Dict[str, Any]:
        """Get comprehensive statistics for the ensemble."""
        statistics = {
            'ensemble_stats': {
                'mean_score': float(np.mean(ensemble_scores)),
                'std_score': float(np.std(ensemble_scores)),
                'min_score': float(np.min(ensemble_scores)),
                'max_score': float(np.max(ensemble_scores)),
                'percentiles': {
                    '50th': float(np.percentile(ensemble_scores, 50)),
                    '75th': float(np.percentile(ensemble_scores, 75)),
                    '90th': float(np.percentile(ensemble_scores, 90)),
                    '95th': float(np.percentile(ensemble_scores, 95)),
                    '99th': float(np.percentile(ensemble_scores, 99))
                }
            },
            'component_stats': {},
            'ensemble_config': self.ensemble_config
        }
        
        # Add component statistics
        if hasattr(self, 'component_scores'):
            # Rule-based statistics
            if 'rule_based' in self.component_scores:
                rule_scores = self.component_scores['rule_based']['scores']
                statistics['component_stats']['rule_based'] = {
                    'mean_score': float(np.mean(rule_scores)),
                    'triggered_transactions': int(np.sum(rule_scores > 0)),
                    'trigger_rate': float(np.sum(rule_scores > 0) / len(rule_scores))
                }
            
            # Statistical statistics
            if 'statistical' in self.component_scores:
                stat_stats = self.statistical_detector.get_model_statistics(df)
                statistics['component_stats']['statistical'] = stat_stats
        
        return statistics
    
    def update_ensemble_weights(self, new_weights: Dict[str, float]):
        """Update ensemble weights."""
        # Validate weights
        required_keys = ['rule_weight', 'isolation_weight', 'autoencoder_weight', 'clustering_weight']
        
        for key in required_keys:
            if key not in new_weights:
                raise ValueError(f"Missing weight: {key}")
        
        # Normalize weights to sum to 1
        total_weight = sum(new_weights.values())
        if total_weight <= 0:
            raise ValueError("Total weight must be positive")
        
        normalized_weights = {k: v / total_weight for k, v in new_weights.items()}
        
        # Update configuration
        self.ensemble_config.update(normalized_weights)
        
        logger.info(f"Updated ensemble weights: {self.ensemble_config}")
    
    def get_top_anomalies(self, df: pd.DataFrame, ensemble_scores: np.ndarray, 
                          top_n: int = 100) -> pd.DataFrame:
        """Get top N anomalies with detailed information."""
        # Get top anomaly indices
        top_indices = np.argsort(ensemble_scores)[-top_n:][::-1]
        
        # Create results dataframe
        top_anomalies = df.iloc[top_indices].copy()
        top_anomalies['ensemble_score'] = ensemble_scores[top_indices]
        
        # Add component scores
        if hasattr(self, 'component_scores'):
            if 'rule_based' in self.component_scores:
                top_anomalies['rule_score'] = self.component_scores['rule_based']['scores'][top_indices]
            
            if 'statistical' in self.component_scores:
                statistical_scores = self.component_scores['statistical']['scores']
                for method, scores in statistical_scores.items():
                    top_anomalies[f'{method}_score'] = scores[top_indices]
        
        # Add risk categorization
        top_anomalies['risk_level'] = pd.cut(
            top_anomalies['ensemble_score'],
            bins=[0, 0.3, 0.6, 0.8, 1.0],
            labels=['Low', 'Medium', 'High', 'Critical'],
            include_lowest=True
        )
        
        return top_anomalies.reset_index(drop=True)
    
    def save_model(self, filepath: str):
        """Save ensemble model state."""
        import pickle
        
        model_state = {
            'ensemble_config': self.ensemble_config,
            'is_fitted': self.is_fitted,
            'statistical_detector': self.statistical_detector,
            'rule_detector': self.rule_detector
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_state, f)
        
        logger.info(f"Ensemble model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load ensemble model state."""
        import pickle
        
        with open(filepath, 'rb') as f:
            model_state = pickle.load(f)
        
        self.ensemble_config = model_state['ensemble_config']
        self.is_fitted = model_state['is_fitted']
        self.statistical_detector = model_state['statistical_detector']
        self.rule_detector = model_state['rule_detector']
        
        logger.info(f"Ensemble model loaded from {filepath}")

