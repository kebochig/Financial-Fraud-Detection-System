"""
Rule-based anomaly detection for transaction fraud.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from ..utils.config import config

logger = logging.getLogger(__name__)

class RuleBasedAnomalyDetector:
    """Rule-based fraud detection system using business logic and heuristics."""
    
    def __init__(self):
        """Initialize rule-based detector with configuration."""
        self.rules = {}
        self.rule_weights = {}
        self.threshold = 0.5  # Default threshold for anomaly classification
        
        # Load configuration
        self.amount_zscore_threshold = config.get('features.amount_zscore_threshold', 2.0)
        self.location_risk_threshold = config.get('features.location_risk_threshold', 0.1)
        self.temporal_window_hours = config.get('features.temporal_window_hours', 24)
        
        # Initialize default rules
        self._initialize_default_rules()
        
        logger.info("Rule-based anomaly detector initialized")
    
    def _initialize_default_rules(self):
        """Initialize default fraud detection rules."""
        
        # Rule 1: Large amount transactions
        self.rules['large_amount'] = {
            'description': 'Transaction amount is unusually large',
            'function': self._rule_large_amount,
            'weight': 0.3,
            'parameters': {
                'amount_threshold_percentile': 0.95,
                'zscore_threshold': 3.0
            }
        }
        
        # Rule 2: Unusual hours
        self.rules['unusual_hours'] = {
            'description': 'Transaction at unusual hours (night/early morning)',
            'function': self._rule_unusual_hours,
            'weight': 0.15,
            'parameters': {
                'start_hour': 0,
                'end_hour': 6
            }
        }
        
        # Rule 3: High transaction velocity
        self.rules['high_velocity'] = {
            'description': 'High number of transactions in short time window',
            'function': self._rule_high_velocity,
            'weight': 0.25,
            'parameters': {
                'velocity_threshold': 5,
                'time_window_hours': 1
            }
        }
        
        # Rule 4: Unusual location
        self.rules['unusual_location'] = {
            'description': 'Transaction from unusual or rare location',
            'function': self._rule_unusual_location,
            'weight': 0.2,
            'parameters': {
                'rarity_threshold': 0.05
            }
        }
        
        # Rule 5: Device change
        self.rules['device_change'] = {
            'description': 'Transaction from different device than usual',
            'function': self._rule_device_change,
            'weight': 0.1,
            'parameters': {
                'consecutive_changes_threshold': 2
            }
        }
        
        # Rule 6: Amount deviation from user pattern
        self.rules['amount_deviation'] = {
            'description': 'Amount significantly deviates from user pattern',
            'function': self._rule_amount_deviation,
            'weight': 0.25,
            'parameters': {
                'zscore_threshold': 2.5
            }
        }
        
        # Rule 7: Weekend unusual activity
        self.rules['weekend_activity'] = {
            'description': 'High-value transactions during weekends',
            'function': self._rule_weekend_activity,
            'weight': 0.1,
            'parameters': {
                'amount_threshold_percentile': 0.8
            }
        }
        
        # Rule 8: Rare transaction type
        self.rules['rare_transaction_type'] = {
            'description': 'Using rare transaction type for user',
            'function': self._rule_rare_transaction_type,
            'weight': 0.15,
            'parameters': {
                'rarity_threshold': 0.1
            }
        }
        
        # Rule 9: Multiple locations same day
        self.rules['multiple_locations'] = {
            'description': 'Transactions from multiple locations same day',
            'function': self._rule_multiple_locations,
            'weight': 0.2,
            'parameters': {
                'location_threshold': 3,
                'time_window_hours': 24
            }
        }
        
        # Rule 10: Round amount pattern
        self.rules['round_amounts'] = {
            'description': 'Suspicious round amount patterns',
            'function': self._rule_round_amounts,
            'weight': 0.05,
            'parameters': {
                'round_threshold': 100
            }
        }
        
        logger.info(f"Initialized {len(self.rules)} fraud detection rules")
    
    def _rule_large_amount(self, df: pd.DataFrame, params: Dict) -> np.ndarray:
        """Rule: Detect unusually large transaction amounts."""
        if 'amount' not in df.columns:
            return np.zeros(len(df))
        
        # Global percentile threshold
        amount_threshold = df['amount'].quantile(params['amount_threshold_percentile'])
        
        # Z-score based detection
        amount_mean = df['amount'].mean()
        amount_std = df['amount'].std()
        zscore_threshold = params['zscore_threshold']
        
        large_amount_mask = (
            (df['amount'] > amount_threshold) | 
            (np.abs((df['amount'] - amount_mean) / amount_std) > zscore_threshold)
        )
        
        return large_amount_mask.astype(float)
    
    def _rule_unusual_hours(self, df: pd.DataFrame, params: Dict) -> np.ndarray:
        """Rule: Detect transactions at unusual hours."""
        if 'hour' not in df.columns:
            return np.zeros(len(df))
        
        start_hour = params['start_hour']
        end_hour = params['end_hour']
        
        unusual_hours_mask = (
            (df['hour'] >= start_hour) & (df['hour'] <= end_hour)
        )
        
        return unusual_hours_mask.astype(float)
    
    def _rule_high_velocity(self, df: pd.DataFrame, params: Dict) -> np.ndarray:
        """Rule: Detect high transaction velocity."""
        if 'tx_velocity_24h' not in df.columns:
            # Use user transaction count as proxy
            if 'user_tx_count' in df.columns:
                velocity_scores = (df['user_tx_count'] > params['velocity_threshold']).astype(float)
            else:
                velocity_scores = np.zeros(len(df))
        else:
            velocity_scores = (df['tx_velocity_24h'] > params['velocity_threshold']).astype(float)
        
        return velocity_scores
    
    def _rule_unusual_location(self, df: pd.DataFrame, params: Dict) -> np.ndarray:
        """Rule: Detect transactions from unusual locations."""
        if 'location_rarity' not in df.columns:
            return np.zeros(len(df))
        
        unusual_location_mask = df['location_rarity'] > params['rarity_threshold']
        return unusual_location_mask.astype(float)
    
    def _rule_device_change(self, df: pd.DataFrame, params: Dict) -> np.ndarray:
        """Rule: Detect frequent device changes."""
        if 'device_changed' not in df.columns:
            return np.zeros(len(df))
        
        # Look for patterns of frequent device changes
        device_change_scores = np.where(
            df['device_changed'] == 1,
            0.5,  # Single device change gets moderate score
            0.0
        )
        
        # Higher score for users with high device diversity
        if 'user_device_diversity' in df.columns:
            high_diversity_mask = df['user_device_diversity'] > 0.5
            device_change_scores = np.where(
                high_diversity_mask,
                device_change_scores + 0.3,
                device_change_scores
            )
        
        return np.clip(device_change_scores, 0, 1)
    
    def _rule_amount_deviation(self, df: pd.DataFrame, params: Dict) -> np.ndarray:
        """Rule: Detect amounts that deviate from user's typical pattern."""
        if 'amount_zscore_user' not in df.columns:
            return np.zeros(len(df))
        
        zscore_threshold = params['zscore_threshold']
        deviation_scores = (
            np.abs(df['amount_zscore_user']) > zscore_threshold
        ).astype(float)
        
        return deviation_scores
    
    def _rule_weekend_activity(self, df: pd.DataFrame, params: Dict) -> np.ndarray:
        """Rule: Detect high-value transactions during weekends."""
        if not all(col in df.columns for col in ['is_weekend', 'amount']):
            return np.zeros(len(df))
        
        amount_threshold = df['amount'].quantile(params['amount_threshold_percentile'])
        weekend_high_value_mask = (
            df['is_weekend'] & (df['amount'] > amount_threshold)
        )
        
        return weekend_high_value_mask.astype(float)
    
    def _rule_rare_transaction_type(self, df: pd.DataFrame, params: Dict) -> np.ndarray:
        """Rule: Detect rare transaction types."""
        if 'type_rarity' not in df.columns:
            return np.zeros(len(df))
        
        rare_type_mask = df['type_rarity'] > params['rarity_threshold']
        return rare_type_mask.astype(float)
    
    def _rule_multiple_locations(self, df: pd.DataFrame, params: Dict) -> np.ndarray:
        """Rule: Detect multiple locations in short time window."""
        if 'user_unique_locations' not in df.columns:
            return np.zeros(len(df))
        
        # Users with high location diversity get higher scores
        location_threshold = params['location_threshold']
        multiple_locations_mask = (
            df['user_unique_locations'] >= location_threshold
        )
        
        return multiple_locations_mask.astype(float)
    
    def _rule_round_amounts(self, df: pd.DataFrame, params: Dict) -> np.ndarray:
        """Rule: Detect suspicious round amount patterns."""
        if 'amount' not in df.columns:
            return np.zeros(len(df))
        
        round_threshold = params['round_threshold']
        
        # Check if amount is a round number (multiple of round_threshold)
        round_amounts_mask = (df['amount'] % round_threshold == 0) & (df['amount'] >= round_threshold)
        
        return round_amounts_mask.astype(float)
    
    def detect_anomalies(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Detect anomalies using all rules."""
        logger.info(f"Running rule-based anomaly detection on {len(df)} transactions...")
        
        rule_scores = {}
        weighted_scores = np.zeros(len(df))
        
        # Apply each rule
        for rule_name, rule_config in self.rules.items():
            try:
                # Apply rule function
                scores = rule_config['function'](df, rule_config['parameters'])
                rule_scores[rule_name] = scores
                
                # Add weighted score
                weighted_scores += scores * rule_config['weight']
                
                logger.debug(f"Rule '{rule_name}': {np.sum(scores > 0)} triggers")
                
            except Exception as e:
                logger.warning(f"Error applying rule '{rule_name}': {str(e)}")
                rule_scores[rule_name] = np.zeros(len(df))
        
        # Normalize scores to [0, 1] range
        max_possible_score = sum(rule['weight'] for rule in self.rules.values())
        normalized_scores = weighted_scores / max_possible_score
        
        logger.info(f"Rule-based detection complete. Mean anomaly score: {np.mean(normalized_scores):.4f}")
        
        return normalized_scores, rule_scores
    
    def explain_anomaly(self, df: pd.DataFrame, index: int, rule_scores: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Explain why a specific transaction was flagged as anomalous."""
        if index >= len(df):
            raise ValueError(f"Index {index} is out of range for DataFrame of length {len(df)}")
        
        transaction = df.iloc[index]
        explanation = {
            'transaction_index': index,
            'transaction_details': transaction.to_dict(),
            'triggered_rules': [],
            'rule_contributions': {},
            'total_score': 0.0
        }
        
        total_score = 0.0
        
        for rule_name, scores in rule_scores.items():
            rule_score = scores[index]
            rule_weight = self.rules[rule_name]['weight']
            weighted_contribution = rule_score * rule_weight
            
            if rule_score > 0:
                explanation['triggered_rules'].append({
                    'rule_name': rule_name,
                    'description': self.rules[rule_name]['description'],
                    'score': float(rule_score),
                    'weight': rule_weight,
                    'contribution': float(weighted_contribution)
                })
            
            explanation['rule_contributions'][rule_name] = {
                'score': float(rule_score),
                'contribution': float(weighted_contribution)
            }
            
            total_score += weighted_contribution
        
        # Normalize total score
        max_possible_score = sum(rule['weight'] for rule in self.rules.values())
        explanation['total_score'] = total_score / max_possible_score
        
        # Sort triggered rules by contribution
        explanation['triggered_rules'].sort(key=lambda x: x['contribution'], reverse=True)
        
        return explanation
    
    def get_rule_statistics(self, df: pd.DataFrame, rule_scores: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        """Get statistics for each rule's performance."""
        statistics = {}
        
        for rule_name, scores in rule_scores.items():
            triggered_count = np.sum(scores > 0)
            mean_score = np.mean(scores)
            max_score = np.max(scores)
            
            statistics[rule_name] = {
                'description': self.rules[rule_name]['description'],
                'weight': self.rules[rule_name]['weight'],
                'triggered_count': int(triggered_count),
                'trigger_rate': float(triggered_count / len(df)),
                'mean_score': float(mean_score),
                'max_score': float(max_score),
                'contribution_to_total': float(np.sum(scores * self.rules[rule_name]['weight']))
            }
        
        return statistics
    
    def set_rule_weight(self, rule_name: str, weight: float):
        """Update the weight of a specific rule."""
        if rule_name not in self.rules:
            raise ValueError(f"Rule '{rule_name}' not found")
        
        if not 0 <= weight <= 1:
            raise ValueError("Weight must be between 0 and 1")
        
        self.rules[rule_name]['weight'] = weight
        logger.info(f"Updated weight for rule '{rule_name}' to {weight}")
    
    def add_custom_rule(self, rule_name: str, rule_function, weight: float, 
                       description: str, parameters: Dict = None):
        """Add a custom rule to the detector."""
        if parameters is None:
            parameters = {}
        
        self.rules[rule_name] = {
            'description': description,
            'function': rule_function,
            'weight': weight,
            'parameters': parameters
        }
        
        logger.info(f"Added custom rule '{rule_name}' with weight {weight}")
    
    def disable_rule(self, rule_name: str):
        """Disable a rule by setting its weight to 0."""
        if rule_name not in self.rules:
            raise ValueError(f"Rule '{rule_name}' not found")
        
        self.rules[rule_name]['weight'] = 0.0
        logger.info(f"Disabled rule '{rule_name}'")
    
    def get_top_anomalies(self, df: pd.DataFrame, anomaly_scores: np.ndarray, 
                         rule_scores: Dict[str, np.ndarray], top_n: int = 10) -> pd.DataFrame:
        """Get top N anomalies with explanations."""
        # Get indices of top anomalies
        top_indices = np.argsort(anomaly_scores)[-top_n:][::-1]
        
        # Create results dataframe
        top_anomalies = df.iloc[top_indices].copy()
        top_anomalies['anomaly_score'] = anomaly_scores[top_indices]
        
        # Add rule trigger information
        for rule_name in self.rules.keys():
            top_anomalies[f'rule_{rule_name}'] = rule_scores[rule_name][top_indices]
        
        # Add explanation summary
        explanations = []
        for idx in top_indices:
            explanation = self.explain_anomaly(df, idx, rule_scores)
            triggered_rules = [rule['rule_name'] for rule in explanation['triggered_rules']]
            explanations.append('; '.join(triggered_rules[:3]))  # Top 3 rules
        
        top_anomalies['top_triggered_rules'] = explanations
        
        return top_anomalies.reset_index(drop=True)
