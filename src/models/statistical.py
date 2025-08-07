"""
Statistical and clustering-based anomaly detection models.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
import hdbscan
from typing import Dict, List, Tuple, Any, Optional
import logging
from ..utils.config import config

logger = logging.getLogger(__name__)

class StatisticalAnomalyDetector:
    """Statistical and clustering-based anomaly detection for fraud detection."""
    
    def __init__(self):
        """Initialize statistical detector with configuration."""
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.is_fitted = False
        
        # Load configuration
        self.config = {
            'isolation_forest': config.get('models.isolation_forest', {}),
            'dbscan': config.get('models.dbscan', {}),
            'random_seed': config.get('data.random_seed', 42)
        }
        
        # Initialize models
        self._initialize_models()
        
        logger.info("Statistical anomaly detector initialized")
    
    def _initialize_models(self):
        """Initialize statistical models with configuration."""
        
        # Isolation Forest
        self.models['isolation_forest'] = IsolationForest(
            contamination=self.config['isolation_forest'].get('contamination', 0.1),
            n_estimators=self.config['isolation_forest'].get('n_estimators', 100),
            random_state=self.config['random_seed'],
            n_jobs=-1
        )
        
        # DBSCAN
        self.models['dbscan'] = DBSCAN(
            eps=self.config['dbscan'].get('eps', 0.5),
            min_samples=self.config['dbscan'].get('min_samples', 5),
            n_jobs=-1
        )
        
        # HDBSCAN (more robust than DBSCAN)
        self.models['hdbscan'] = hdbscan.HDBSCAN(
            min_cluster_size=10,
            min_samples=5,
            cluster_selection_epsilon=0.5
        )
        
        # One-Class SVM
        self.models['one_class_svm'] = OneClassSVM(
            kernel='rbf',
            gamma='scale',
            nu=0.1  # Expected fraction of outliers
        )
        
        # Local Outlier Factor
        self.models['lof'] = LocalOutlierFactor(
            n_neighbors=20,
            contamination=0.1,
            novelty=True,
            n_jobs=-1
        )
        
        logger.info(f"Initialized {len(self.models)} statistical models")
    
    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Prepare features for statistical models."""
        logger.info("Preparing features for statistical models...")
        
        # Select numerical features
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove irrelevant columns
        exclude_cols = [
            'line_num', 'is_parsed', 'user_min', 'user_max', 
            'feature_extraction_timestamp'
        ]
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        # Handle categorical features
        categorical_cols = []
        for col in ['location_filled', 'device_filled', 'transaction_type_filled', 
                   'currency_filled', 'device_brand', 'transaction_group', 'hour_category']:
            if col in df.columns:
                categorical_cols.append(col)
        
        # Prepare numerical features
        X_numerical = df[numerical_cols].fillna(0).values
        
        # Prepare categorical features
        X_categorical = np.array([])
        if categorical_cols:
            X_categorical = np.column_stack([
                self._encode_categorical(df[col]) for col in categorical_cols
            ])
        
        # Combine features
        if X_categorical.size > 0:
            X = np.column_stack([X_numerical, X_categorical])
            feature_names = numerical_cols + [f"{col}_encoded" for col in categorical_cols]
        else:
            X = X_numerical
            feature_names = numerical_cols
        
        # Handle infinite values
        X = np.where(np.isinf(X), 0, X)
        
        logger.info(f"Prepared {X.shape[1]} features for {X.shape[0]} samples")
        return X, feature_names
    
    def _encode_categorical(self, series: pd.Series) -> np.ndarray:
        """Encode categorical series."""
        series_name = series.name
        
        # Convert to string first, then handle missing values
        series_str = series.astype(str)
        series_str = series_str.replace('nan', 'unknown')
        
        if series_name not in self.encoders:
            self.encoders[series_name] = LabelEncoder()
            # Fit and transform
            encoded = self.encoders[series_name].fit_transform(series_str)
        else:
            # Transform only
            try:
                encoded = self.encoders[series_name].transform(series_str)
            except ValueError:
                # Handle unseen categories
                known_categories = set(self.encoders[series_name].classes_)
                series_cleaned = series_str.apply(
                    lambda x: x if x in known_categories else 'unknown'
                )
                encoded = self.encoders[series_name].transform(series_cleaned)
        
        return encoded.astype(float)
    
    def fit(self, df: pd.DataFrame) -> 'StatisticalAnomalyDetector':
        """Fit statistical models on training data."""
        logger.info(f"Fitting statistical models on {len(df)} samples...")
        
        # Prepare features
        X, feature_names = self._prepare_features(df)
        
        # Scale features
        self.scalers['standard'] = StandardScaler()
        X_scaled = self.scalers['standard'].fit_transform(X)
        
        # Fit models
        fitted_models = []
        
        # Isolation Forest
        try:
            self.models['isolation_forest'].fit(X_scaled)
            fitted_models.append('isolation_forest')
            logger.info("✅ Isolation Forest fitted")
        except Exception as e:
            logger.error(f"❌ Error fitting Isolation Forest: {str(e)}")
        
        # One-Class SVM (can be slow on large datasets)
        if len(df) < 10000:  # Only fit if dataset is not too large
            try:
                self.models['one_class_svm'].fit(X_scaled)
                fitted_models.append('one_class_svm')
                logger.info("✅ One-Class SVM fitted")
            except Exception as e:
                logger.error(f"❌ Error fitting One-Class SVM: {str(e)}")
        else:
            logger.info("⚠️ Skipping One-Class SVM due to large dataset size")
        
        # Local Outlier Factor
        try:
            self.models['lof'].fit(X_scaled)
            fitted_models.append('lof')
            logger.info("✅ Local Outlier Factor fitted")
        except Exception as e:
            logger.error(f"❌ Error fitting Local Outlier Factor: {str(e)}")
        
        # Store feature names and fitted status
        self.feature_names = feature_names
        self.fitted_models = fitted_models
        self.is_fitted = True
        
        logger.info(f"Statistical models fitting complete. {len(fitted_models)} models fitted.")
        return self
    
    def predict_anomalies(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Predict anomalies using fitted statistical models."""
        if not self.is_fitted:
            raise ValueError("Models must be fitted before prediction")
        
        logger.info(f"Predicting anomalies for {len(df)} samples...")
        
        # Prepare features
        X, _ = self._prepare_features(df)
        X_scaled = self.scalers['standard'].transform(X)
        
        predictions = {}
        
        # Isolation Forest
        if 'isolation_forest' in self.fitted_models:
            try:
                # Get anomaly scores (lower values = more anomalous)
                scores = self.models['isolation_forest'].decision_function(X_scaled)
                # Convert to 0-1 range (higher values = more anomalous)
                predictions['isolation_forest'] = self._normalize_scores(-scores)
                logger.debug("✅ Isolation Forest predictions generated")
            except Exception as e:
                logger.error(f"❌ Error in Isolation Forest prediction: {str(e)}")
                predictions['isolation_forest'] = np.zeros(len(df))
        
        # One-Class SVM
        if 'one_class_svm' in self.fitted_models:
            try:
                scores = self.models['one_class_svm'].decision_function(X_scaled)
                predictions['one_class_svm'] = self._normalize_scores(-scores)
                logger.debug("✅ One-Class SVM predictions generated")
            except Exception as e:
                logger.error(f"❌ Error in One-Class SVM prediction: {str(e)}")
                predictions['one_class_svm'] = np.zeros(len(df))
        
        # Local Outlier Factor
        if 'lof' in self.fitted_models:
            try:
                scores = self.models['lof'].decision_function(X_scaled)
                predictions['lof'] = self._normalize_scores(-scores)
                logger.debug("✅ Local Outlier Factor predictions generated")
            except Exception as e:
                logger.error(f"❌ Error in Local Outlier Factor prediction: {str(e)}")
                predictions['lof'] = np.zeros(len(df))
        
        # Clustering-based detection
        predictions.update(self._cluster_based_detection(X_scaled))

        
        
        # Print top 5 scores and users for each method
        # self._print_top_scores(df, predictions)
        
        logger.info(f"Statistical anomaly detection complete. Generated {len(predictions)} score sets.")
        return predictions
    
    def _cluster_based_detection(self, X_scaled: np.ndarray) -> Dict[str, np.ndarray]:
        """Perform clustering-based anomaly detection."""
        cluster_predictions = {}
        
        # DBSCAN clustering
        try:
            dbscan_labels = self.models['dbscan'].fit_predict(X_scaled)
            # Points labeled as -1 are considered anomalies
            dbscan_scores = (dbscan_labels == -1).astype(float)
            
            # For non-anomalous points, calculate distance to cluster center
            if len(np.unique(dbscan_labels)) > 1:
                cluster_scores = np.zeros(len(X_scaled))
                for cluster_id in np.unique(dbscan_labels):
                    if cluster_id != -1:  # Skip noise points
                        cluster_mask = dbscan_labels == cluster_id
                        cluster_center = X_scaled[cluster_mask].mean(axis=0)
                        distances = np.linalg.norm(
                            X_scaled[cluster_mask] - cluster_center, axis=1
                        )
                        cluster_scores[cluster_mask] = distances
                
                # Normalize and combine with noise detection
                if cluster_scores.max() > 0:
                    cluster_scores = cluster_scores / cluster_scores.max()
                
                dbscan_scores = np.maximum(dbscan_scores, cluster_scores * 0.5)
            
            cluster_predictions['dbscan'] = dbscan_scores
            logger.debug(f"✅ DBSCAN: {np.sum(dbscan_labels == -1)} anomalies detected")
            
        except Exception as e:
            logger.error(f"❌ Error in DBSCAN clustering: {str(e)}")
            cluster_predictions['dbscan'] = np.zeros(len(X_scaled))
        
        # HDBSCAN clustering
        try:
            hdbscan_labels = self.models['hdbscan'].fit_predict(X_scaled)
            
            # Use outlier scores if available
            if hasattr(self.models['hdbscan'], 'outlier_scores_'):
                hdbscan_scores = self.models['hdbscan'].outlier_scores_
                hdbscan_scores = np.nan_to_num(hdbscan_scores, 0)
            else:
                # Fallback to binary classification
                hdbscan_scores = (hdbscan_labels == -1).astype(float)
            
            cluster_predictions['hdbscan'] = self._normalize_scores(hdbscan_scores)
            logger.debug(f"✅ HDBSCAN: {np.sum(hdbscan_labels == -1)} anomalies detected")
            
        except Exception as e:
            logger.error(f"❌ Error in HDBSCAN clustering: {str(e)}")
            cluster_predictions['hdbscan'] = np.zeros(len(X_scaled))
        
        return cluster_predictions
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range."""
        scores = np.array(scores)
        
        # Handle edge cases
        if len(scores) == 0:
            return scores
        
        if np.all(scores == scores[0]):  # All values are the same
            return np.zeros_like(scores)
        
        # Min-max normalization
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        if max_score == min_score:
            return np.zeros_like(scores)
        
        normalized = (scores - min_score) / (max_score - min_score)
        return np.clip(normalized, 0, 1)
    
    def _print_top_scores(self, df: pd.DataFrame, predictions: Dict[str, np.ndarray]):
        """Print top 5 scores and their users for each statistical method."""
        if 'user_id' not in df.columns:
            logger.warning("No user_id column found. Cannot print user-specific scores.")
            return
        
        print("\n" + "="*80)
        print("🏆 TOP 5 ANOMALY SCORES BY USER FOR EACH STATISTICAL METHOD")
        print("="*80)
        
        for method_name, scores in predictions.items():
            print(f"\n📊 {method_name.upper().replace('_', ' ')}")
            print("-" * 60)
            
            # Create DataFrame with user_id and scores
            score_df = pd.DataFrame({
                'user_id': df['user_id'],
                'score': scores,
                'amount': df.get('amount', [0] * len(df)),
                'location': df.get('location', ['Unknown'] * len(df)),
                'transaction_type': df.get('transaction_type', ['Unknown'] * len(df))
            })
            
            # Group by user and get mean score
            user_scores = score_df.groupby('user_id').agg({
                'score': ['mean', 'max', 'count'],
                'amount': 'mean',
                'location': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown',
                'transaction_type': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
            }).round(4)
            
            # Flatten column names
            user_scores.columns = ['mean_score', 'max_score', 'tx_count', 'avg_amount', 'main_location', 'main_type']
            
            # Sort by mean score (descending)
            user_scores = user_scores.sort_values('mean_score', ascending=False)
            
            # Get top 5 users
            top_5_users = user_scores.head(5)
            
            # Print header
            print(f"{'Rank':<4} {'User':<12} {'Mean Score':<12} {'Max Score':<12} {'Tx Count':<10} {'Avg Amount':<12} {'Location':<12} {'Type':<12}")
            print("-" * 100)
            
            for rank, (user_id, row) in enumerate(top_5_users.iterrows(), 1):
                mean_score = row['mean_score']
                max_score = row['max_score']
                tx_count = int(row['tx_count'])
                avg_amount = row['avg_amount']
                location = row['main_location']
                tx_type = row['main_type']
                
                # Determine risk level
                if mean_score > 0.8:
                    risk_level = "🔥 CRITICAL"
                elif mean_score > 0.6:
                    risk_level = "⚠️ HIGH"
                elif mean_score > 0.4:
                    risk_level = "⚡ MEDIUM"
                elif mean_score > 0.2:
                    risk_level = "📊 LOW"
                else:
                    risk_level = "✅ NORMAL"
                
                print(f"{rank:<4} {user_id:<12} {mean_score:<12.4f} {max_score:<12.4f} {tx_count:<10} "
                      f"{avg_amount:<12.1f} {location:<12} {tx_type:<12} {risk_level}")
            
            # Print summary statistics
            print(f"\n📈 {method_name} Summary:")
            print(f"   • Total users analyzed: {len(user_scores)}")
            print(f"   • Users with high risk (score > 0.6): {len(user_scores[user_scores['mean_score'] > 0.6])}")
            print(f"   • Users with critical risk (score > 0.8): {len(user_scores[user_scores['mean_score'] > 0.8])}")
            print(f"   • Average score across all users: {user_scores['mean_score'].mean():.4f}")
            print(f"   • Highest individual score: {user_scores['max_score'].max():.4f}")
        
        # Print ensemble comparison
        print(f"\n🎯 ENSEMBLE COMPARISON")
        print("-" * 60)
        
        # Calculate ensemble scores if multiple methods available
        if len(predictions) > 1:
            ensemble_scores = np.zeros(len(df))
            for scores in predictions.values():
                ensemble_scores += scores
            ensemble_scores /= len(predictions)
            
            # Create ensemble DataFrame
            ensemble_df = pd.DataFrame({
                'user_id': df['user_id'],
                'ensemble_score': ensemble_scores
            })
            
            # Group by user
            ensemble_user_scores = ensemble_df.groupby('user_id')['ensemble_score'].agg(['mean', 'max', 'count']).round(4)
            ensemble_user_scores = ensemble_user_scores.sort_values('mean', ascending=False)
            
            print(f"{'Rank':<4} {'User':<12} {'Ensemble Score':<15} {'Max Score':<12} {'Tx Count':<10}")
            print("-" * 65)
            
            for rank, (user_id, row) in enumerate(ensemble_user_scores.head(5).iterrows(), 1):
                mean_score = row['mean']
                max_score = row['max']
                tx_count = int(row['count'])
                
                if mean_score > 0.8:
                    risk_level = "🔥 CRITICAL"
                elif mean_score > 0.6:
                    risk_level = "⚠️ HIGH"
                elif mean_score > 0.4:
                    risk_level = "⚡ MEDIUM"
                elif mean_score > 0.2:
                    risk_level = "📊 LOW"
                else:
                    risk_level = "✅ NORMAL"
                
                print(f"{rank:<4} {user_id:<12} {mean_score:<15.4f} {max_score:<12.4f} {tx_count:<10} {risk_level}")
        
        print("\n" + "="*80)
    
    def get_feature_importance(self, df: pd.DataFrame, method: str = 'isolation_forest') -> Dict[str, float]:
        """Get feature importance for anomaly detection."""
        if not self.is_fitted:
            raise ValueError("Models must be fitted before getting feature importance")
        
        if method not in self.fitted_models:
            raise ValueError(f"Method '{method}' not available. Available: {self.fitted_models}")
        
        # For tree-based methods like Isolation Forest
        if method == 'isolation_forest' and hasattr(self.models[method], 'feature_importances_'):
            importances = self.models[method].feature_importances_
            return dict(zip(self.feature_names, importances))
        
        # For other methods, use permutation importance approximation
        logger.info(f"Calculating feature importance for {method}...")
        
        X, _ = self._prepare_features(df)
        X_scaled = self.scalers['standard'].transform(X)
        
        # Get baseline scores
        baseline_scores = self._get_model_scores(method, X_scaled)
        
        importances = {}
        for i, feature_name in enumerate(self.feature_names):
            # Permute feature
            X_permuted = X_scaled.copy()
            np.random.shuffle(X_permuted[:, i])
            
            # Get scores with permuted feature
            permuted_scores = self._get_model_scores(method, X_permuted)
            
            # Calculate importance as change in mean score
            importance = np.mean(np.abs(permuted_scores - baseline_scores))
            importances[feature_name] = importance
        
        # Normalize importances
        total_importance = sum(importances.values())
        if total_importance > 0:
            importances = {k: v / total_importance for k, v in importances.items()}
        
        return importances
    
    def _get_model_scores(self, method: str, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores from a specific model."""
        if method == 'isolation_forest':
            return -self.models[method].decision_function(X)
        elif method == 'one_class_svm':
            return -self.models[method].decision_function(X)
        elif method == 'lof':
            return -self.models[method].decision_function(X)
        else:
            # For clustering methods, return binary scores
            return np.zeros(len(X))
    
    def detect_anomalies_ensemble(self, df: pd.DataFrame, weights: Dict[str, float] = None) -> np.ndarray:
        """Detect anomalies using ensemble of statistical methods."""
        predictions = self.predict_anomalies(df)
        
        if weights is None:
            # Default equal weights
            weights = {method: 1.0 / len(predictions) for method in predictions.keys()}
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        # Compute weighted ensemble score
        ensemble_scores = np.zeros(len(df))
        for method, scores in predictions.items():
            method_weight = weights.get(method, 0)
            ensemble_scores += scores * method_weight
        
        logger.info(f"Ensemble anomaly detection complete. Mean score: {np.mean(ensemble_scores):.4f}")
        return ensemble_scores
    
    def get_model_statistics(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Get statistics for each statistical model."""
        if not self.is_fitted:
            raise ValueError("Models must be fitted before getting statistics")
        
        predictions = self.predict_anomalies(df)
        statistics = {}
        
        for method, scores in predictions.items():
            statistics[method] = {
                'mean_score': float(np.mean(scores)),
                'std_score': float(np.std(scores)),
                'min_score': float(np.min(scores)),
                'max_score': float(np.max(scores)),
                'anomaly_count_10pct': int(np.sum(scores > np.percentile(scores, 90))),
                'anomaly_count_5pct': int(np.sum(scores > np.percentile(scores, 95))),
                'anomaly_count_1pct': int(np.sum(scores > np.percentile(scores, 99)))
            }
        
        return statistics
    
    def reduce_dimensionality(self, df: pd.DataFrame, n_components: int = 2) -> Tuple[np.ndarray, PCA]:
        """Reduce dimensionality for visualization."""
        X, _ = self._prepare_features(df)
        X_scaled = self.scalers['standard'].transform(X) if self.is_fitted else StandardScaler().fit_transform(X)
        
        pca = PCA(n_components=n_components, random_state=self.config['random_seed'])
        X_reduced = pca.fit_transform(X_scaled)
        
        logger.info(f"Reduced dimensionality to {n_components} components. "
                   f"Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
        
        return X_reduced, pca

