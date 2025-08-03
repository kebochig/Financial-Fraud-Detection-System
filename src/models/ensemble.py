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

