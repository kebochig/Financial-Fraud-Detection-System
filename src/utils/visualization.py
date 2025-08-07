"""
Visualization utilities for fraud detection system.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from .config import config

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette(config.get('visualization.color_palette', 'viridis'))

class FraudVisualization:
    """Visualization utilities for fraud detection analysis."""
    
    def __init__(self):
        """Initialize visualization settings."""
        self.fig_size = config.get('visualization.figure_size', (12, 8))
        self.anomaly_color = config.get('visualization.anomaly_color', 'red')
        self.normal_color = config.get('visualization.normal_color', 'blue')
    
    def plot_data_quality_report(self, df: pd.DataFrame, parsed_stats: Dict) -> plt.Figure:
        """Create data quality visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Data Quality Report', fontsize=16, fontweight='bold')
        
        # Parsing success rate
        parsing_data = parsed_stats['parsing_success']
        axes[0, 0].pie([parsing_data['success'], parsing_data['failed']], 
                      labels=['Successful', 'Failed'], 
                      autopct='%1.1f%%',
                      colors=['green', 'red'])
        axes[0, 0].set_title('Parsing Success Rate')
        
        # Data completeness
        completeness = df.isnull().sum() / len(df) * 100
        axes[0, 1].barh(completeness.index, completeness.values)
        axes[0, 1].set_title('Data Completeness (%)')
        axes[0, 1].set_xlabel('Missing Data %')
        
        # Transaction types distribution
        if 'transaction_type' in df.columns:
            type_counts = df['transaction_type'].value_counts()
            axes[1, 0].bar(type_counts.index, type_counts.values)
            axes[1, 0].set_title('Transaction Types Distribution')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # # Amount distribution
        # if 'amount' in df.columns:
        #     axes[1, 1].hist(df['amount'].dropna(), bins=50, alpha=0.7)
        #     axes[1, 1].set_title('Amount Distribution')
        #     axes[1, 1].set_xlabel('Amount')
        #     axes[1, 1].set_ylabel('Frequency')

        # location distribution
        if 'location' in df.columns:
            type_counts = df['location'].value_counts()
            axes[1, 1].bar(type_counts.index, type_counts.values)
            axes[1, 1].set_title('location Distribution')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def plot_anomaly_scores_distribution(self, scores: np.ndarray, threshold: float = None) -> plt.Figure:
        """Plot distribution of anomaly scores."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram
        ax1.hist(scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        if threshold:
            ax1.axvline(threshold, color='red', linestyle='--', 
                       label=f'Threshold: {threshold:.3f}')
            ax1.legend()
        ax1.set_title('Anomaly Scores Distribution')
        ax1.set_xlabel('Anomaly Score')
        ax1.set_ylabel('Frequency')
        
        # Box plot
        ax2.boxplot(scores)
        if threshold:
            ax2.axhline(threshold, color='red', linestyle='--')
        ax2.set_title('Anomaly Scores Box Plot')
        ax2.set_ylabel('Anomaly Score')
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, importance_scores: Dict[str, float]) -> plt.Figure:
        """Plot feature importance scores."""
        features = list(importance_scores.keys())
        scores = list(importance_scores.values())
        
        fig, ax = plt.subplots(figsize=(10, 8))
        bars = ax.barh(features, scores)
        
        # Color bars based on importance
        colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(scores)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_title('Feature Importance for Anomaly Detection', fontsize=14, fontweight='bold')
        ax.set_xlabel('Importance Score')
        plt.tight_layout()
        return fig
    
    def plot_anomalies_scatter(self, df: pd.DataFrame, x_col: str, y_col: str, 
                              anomaly_col: str = 'is_anomaly') -> go.Figure:
        """Create interactive scatter plot of anomalies."""
        fig = px.scatter(df, x=x_col, y=y_col, 
                        color=anomaly_col,
                        color_discrete_map={True: self.anomaly_color, False: self.normal_color},
                        title=f'Anomaly Detection: {x_col} vs {y_col}',
                        hover_data=['user_id', 'transaction_type', 'amount'] if 'user_id' in df.columns else None)
        
        fig.update_layout(
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title(),
            legend_title='Transaction Type'
        )
        
        return fig
    
    def plot_temporal_patterns(self, df: pd.DataFrame) -> plt.Figure:
        """Plot temporal patterns in transactions."""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        if 'timestamp' in df.columns:
            # Transactions over time
            df_time = df.set_index('timestamp').resample('D').size()
            axes[0].plot(df_time.index, df_time.values)
            axes[0].set_title('Transactions Over Time')
            axes[0].set_ylabel('Transaction Count')
            
            # Hourly patterns
            if 'hour' in df.columns:
                hourly_counts = df.groupby('hour').size()
                axes[1].bar(hourly_counts.index, hourly_counts.values)
                axes[1].set_title('Hourly Transaction Patterns')
                axes[1].set_xlabel('Hour of Day')
                axes[1].set_ylabel('Transaction Count')
            
            # Day of week patterns
            if 'day_of_week' in df.columns:
                dow_counts = df.groupby('day_of_week').size()
                days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                axes[2].bar(range(len(dow_counts)), dow_counts.values)
                axes[2].set_xticks(range(len(days)))
                axes[2].set_xticklabels(days)
                axes[2].set_title('Day of Week Transaction Patterns')
                axes[2].set_xlabel('Day of Week')
                axes[2].set_ylabel('Transaction Count')
        
        plt.tight_layout()
        return fig
    
    def plot_user_behavior_analysis(self, df: pd.DataFrame) -> plt.Figure:
        """Analyze and plot user behavior patterns."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('User Behavior Analysis', fontsize=16, fontweight='bold')
        
        if 'user_id' in df.columns:
            # User transaction frequency
            user_freq = df.groupby('user_id').size().sort_values(ascending=False)
            axes[0, 0].hist(user_freq.values, bins=30, alpha=0.7)
            axes[0, 0].set_title('User Transaction Frequency Distribution')
            axes[0, 0].set_xlabel('Number of Transactions')
            axes[0, 0].set_ylabel('Number of Users')
            
            # User amount patterns
            if 'amount' in df.columns:
                user_amounts = df.groupby('user_id')['amount'].agg(['mean', 'std']).fillna(0)
                axes[0, 1].scatter(user_amounts['mean'], user_amounts['std'], alpha=0.6)
                axes[0, 1].set_title('User Amount Patterns')
                axes[0, 1].set_xlabel('Average Transaction Amount')
                axes[0, 1].set_ylabel('Amount Standard Deviation')
            
            # User location diversity
            if 'location' in df.columns:
                user_locations = df.groupby('user_id')['location'].nunique()
                axes[1, 0].hist(user_locations.values, bins=20, alpha=0.7)
                axes[1, 0].set_title('User Location Diversity')
                axes[1, 0].set_xlabel('Number of Unique Locations')
                axes[1, 0].set_ylabel('Number of Users')
            
            # User device diversity
            if 'device' in df.columns:
                user_devices = df.groupby('user_id')['device'].nunique()
                axes[1, 1].hist(user_devices.values, bins=10, alpha=0.7)
                axes[1, 1].set_title('User Device Diversity')
                axes[1, 1].set_xlabel('Number of Unique Devices')
                axes[1, 1].set_ylabel('Number of Users')
        
        plt.tight_layout()
        return fig
    
    def plot_user_behavior_analysis_advanced(self, df: pd.DataFrame) -> plt.Figure:
        """Advanced user behavior analysis with frequency, recency, monetary value, and other metrics."""
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Advanced User Behavior Analysis', fontsize=18, fontweight='bold')
        
        if 'user_id' not in df.columns:
            # If no user_id, create a simple message
            axes[1, 1].text(0.5, 0.5, 'No user_id column found\nin the dataset', 
                           ha='center', va='center', transform=axes[1, 1].transAxes,
                           fontsize=14, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
            plt.tight_layout()
            return fig
        
        # Convert timestamp to datetime if needed
        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 1. User Transaction Frequency Distribution (Top-left)
        user_freq = df.groupby('user_id').size().sort_values(ascending=False)
        axes[0, 0].hist(user_freq.values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('User Transaction Frequency Distribution', fontweight='bold')
        axes[0, 0].set_xlabel('Number of Transactions')
        axes[0, 0].set_ylabel('Number of Users')
        
        # Add statistics
        total_users = len(user_freq)
        avg_transactions = user_freq.mean()
        max_transactions = user_freq.max()
        stats_text = f'Total Users: {total_users}\nAvg: {avg_transactions:.1f}\nMax: {max_transactions}'
        axes[0, 0].text(0.7, 0.8, stats_text, transform=axes[0, 0].transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 2. Top Users by Transaction Count (Top-center)
        top_users = user_freq.head(15)
        bars = axes[0, 1].bar(range(len(top_users)), top_users.values, 
                              color=plt.cm.viridis(np.linspace(0, 1, len(top_users))))
        axes[0, 1].set_title('Top 15 Users by Transaction Count', fontweight='bold')
        axes[0, 1].set_xlabel('User Rank')
        axes[0, 1].set_ylabel('Transaction Count')
        axes[0, 1].set_xticks(range(len(top_users)))
        axes[0, 1].set_xticklabels([f'#{i+1}' for i in range(len(top_users))], rotation=45)
        
        # Add user IDs on bars
        for i, (user_id, count) in enumerate(top_users.items()):
            axes[0, 1].text(i, count + 0.1, str(user_id), ha='center', va='bottom', 
                           fontsize=7, rotation=45)
        
        # 3. User Recency Analysis (Top-right)
        if 'timestamp' in df.columns:
            # Calculate days since last transaction for each user
            latest_date = df['timestamp'].max()
            user_recency = df.groupby('user_id')['timestamp'].max().apply(
                lambda x: (latest_date - x).days)
            
            axes[0, 2].hist(user_recency.values, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
            axes[0, 2].set_title('User Recency Distribution', fontweight='bold')
            axes[0, 2].set_xlabel('Days Since Last Transaction')
            axes[0, 2].set_ylabel('Number of Users')
            
            # Add recency statistics
            avg_recency = user_recency.mean()
            recent_users = (user_recency <= 7).sum()
            recency_stats = f'Avg Days: {avg_recency:.1f}\nRecent (≤7 days): {recent_users}'
            axes[0, 2].text(0.7, 0.8, recency_stats, transform=axes[0, 2].transAxes,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 4. User Monetary Value Analysis (Middle-left)
        if 'amount' in df.columns:
            user_monetary = df.groupby('user_id')['amount'].agg(['sum', 'mean']).fillna(0)
            
            scatter = axes[1, 0].scatter(user_monetary['mean'], user_monetary['sum'], 
                                        alpha=0.6, c=user_monetary['sum'], cmap='viridis', s=50)
            axes[1, 0].set_title('User Monetary Value Analysis', fontweight='bold')
            axes[1, 0].set_xlabel('Average Transaction Amount')
            axes[1, 0].set_ylabel('Total Transaction Value')
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=axes[1, 0])
            cbar.set_label('Total Value')
            
            # Add statistics
            high_value_users = (user_monetary['sum'] > user_monetary['sum'].quantile(0.9)).sum()
            monetary_stats = f'High Value Users: {high_value_users}'
            axes[1, 0].text(0.7, 0.8, monetary_stats, transform=axes[1, 0].transAxes,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 5. User Location Diversity (Middle-center)
        if 'location' in df.columns:
            user_locations = df.groupby('user_id')['location'].nunique()
            
            axes[1, 1].hist(user_locations.values, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
            axes[1, 1].set_title('User Location Diversity', fontweight='bold')
            axes[1, 1].set_xlabel('Number of Unique Locations')
            axes[1, 1].set_ylabel('Number of Users')
            
            # Add location statistics
            avg_locations = user_locations.mean()
            max_locations = user_locations.max()
            mobile_users = (user_locations > 1).sum()
            loc_stats = f'Avg Locations: {avg_locations:.1f}\nMobile Users: {mobile_users}'
            axes[1, 1].text(0.7, 0.8, loc_stats, transform=axes[1, 1].transAxes,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 6. User Device Diversity (Middle-right)
        if 'device' in df.columns:
            user_devices = df.groupby('user_id')['device'].nunique()
            
            axes[1, 2].hist(user_devices.values, bins=10, alpha=0.7, color='gold', edgecolor='black')
            axes[1, 2].set_title('User Device Diversity', fontweight='bold')
            axes[1, 2].set_xlabel('Number of Unique Devices')
            axes[1, 2].set_ylabel('Number of Users')
            
            # Add device statistics
            avg_devices = user_devices.mean()
            multi_device_users = (user_devices > 1).sum()
            device_stats = f'Avg Devices: {avg_devices:.1f}\nMulti-Device: {multi_device_users}'
            axes[1, 2].text(0.7, 0.8, device_stats, transform=axes[1, 2].transAxes,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 7. User Transaction Type Diversity (Bottom-left)
        if 'transaction_type' in df.columns:
            user_types = df.groupby('user_id')['transaction_type'].nunique()
            
            axes[2, 0].hist(user_types.values, bins=10, alpha=0.7, color='lightblue', edgecolor='black')
            axes[2, 0].set_title('User Transaction Type Diversity', fontweight='bold')
            axes[2, 0].set_xlabel('Number of Transaction Types')
            axes[2, 0].set_ylabel('Number of Users')
            
            # Add type statistics
            avg_types = user_types.mean()
            diverse_users = (user_types > 1).sum()
            type_stats = f'Avg Types: {avg_types:.1f}\nDiverse Users: {diverse_users}'
            axes[2, 0].text(0.7, 0.8, type_stats, transform=axes[2, 0].transAxes,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 8. User Activity Timeline (Bottom-center)
        if 'timestamp' in df.columns:
            # Group by date and count transactions
            daily_activity = df.groupby(df['timestamp'].dt.date).size()
            
            axes[2, 1].plot(daily_activity.index, daily_activity.values, 
                           marker='o', linewidth=2, markersize=4)
            axes[2, 1].set_title('Daily Transaction Activity', fontweight='bold')
            axes[2, 1].set_xlabel('Date')
            axes[2, 1].set_ylabel('Transaction Count')
            axes[2, 1].tick_params(axis='x', rotation=45)
            
            # Add activity statistics
            peak_day = daily_activity.idxmax()
            peak_count = daily_activity.max()
            avg_daily = daily_activity.mean()
            timeline_stats = f'Peak Day: {peak_day}\nPeak Count: {peak_count}\nAvg Daily: {avg_daily:.1f}'
            axes[2, 1].text(0.7, 0.8, timeline_stats, transform=axes[2, 1].transAxes,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 9. User Behavior Clusters (Bottom-right)
        if 'amount' in df.columns and 'timestamp' in df.columns:
            # Create user behavior features
            user_features = df.groupby('user_id').agg({
                'amount': ['count', 'mean', 'std'],
                'timestamp': lambda x: (x.max() - x.min()).days
            }).fillna(0)
            
            user_features.columns = ['transaction_count', 'avg_amount', 'amount_std', 'activity_span']
            
            # Normalize features for clustering visualization
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(user_features)
            
            # Create scatter plot of user clusters
            scatter = axes[2, 2].scatter(features_scaled[:, 0], features_scaled[:, 1], 
                                        c=user_features['avg_amount'], cmap='plasma', alpha=0.6, s=30)
            axes[2, 2].set_title('User Behavior Clusters', fontweight='bold')
            axes[2, 2].set_xlabel('Transaction Count (normalized)')
            axes[2, 2].set_ylabel('Activity Span (normalized)')
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=axes[2, 2])
            cbar.set_label('Average Amount')
        
        plt.tight_layout()
        return fig
    
    def get_user_behavior_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive user behavior summary statistics."""
        if 'user_id' not in df.columns:
            return {}
        
        # Convert timestamp if needed
        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate user metrics
        user_metrics = df.groupby('user_id').agg({
            'amount': ['count', 'mean', 'std', 'min', 'max', 'sum'] if 'amount' in df.columns else [],
            'location': 'nunique' if 'location' in df.columns else [],
            'device': 'nunique' if 'device' in df.columns else [],
            'transaction_type': 'nunique' if 'transaction_type' in df.columns else []
        }).round(2)
        
        # Flatten column names
        user_metrics.columns = ['_'.join(col).strip() for col in user_metrics.columns]
        
        # Calculate recency if timestamp available
        recency_metrics = {}
        if 'timestamp' in df.columns:
            latest_date = df['timestamp'].max()
            user_recency = df.groupby('user_id')['timestamp'].max().apply(
                lambda x: (latest_date - x).days)
            recency_metrics = {
                'avg_days_since_last_transaction': user_recency.mean(),
                'recent_users_7_days': (user_recency <= 7).sum(),
                'recent_users_30_days': (user_recency <= 30).sum(),
                'inactive_users_90_days': (user_recency > 90).sum()
            }
        
        # Calculate activity span
        activity_span = {}
        if 'timestamp' in df.columns:
            user_activity_span = df.groupby('user_id')['timestamp'].agg(['min', 'max'])
            user_activity_span['span_days'] = (user_activity_span['max'] - user_activity_span['min']).dt.days
            activity_span = {
                'avg_activity_span_days': user_activity_span['span_days'].mean(),
                'max_activity_span_days': user_activity_span['span_days'].max(),
                'users_with_span_1_day': (user_activity_span['span_days'] == 0).sum()
            }
        
        # Calculate diversity metrics
        diversity_metrics = {}
        if 'location' in df.columns:
            user_locations = df.groupby('user_id')['location'].nunique()
            diversity_metrics['avg_unique_locations'] = user_locations.mean()
            diversity_metrics['mobile_users_multiple_locations'] = (user_locations > 1).sum()
        
        if 'device' in df.columns:
            user_devices = df.groupby('user_id')['device'].nunique()
            diversity_metrics['avg_unique_devices'] = user_devices.mean()
            diversity_metrics['multi_device_users'] = (user_devices > 1).sum()
        
        if 'transaction_type' in df.columns:
            user_types = df.groupby('user_id')['transaction_type'].nunique()
            diversity_metrics['avg_unique_transaction_types'] = user_types.mean()
            diversity_metrics['diverse_transaction_users'] = (user_types > 1).sum()
        
        # Calculate monetary metrics
        monetary_metrics = {}
        if 'amount' in df.columns:
            user_amounts = df.groupby('user_id')['amount'].agg(['sum', 'mean', 'std'])
            monetary_metrics['avg_total_value_per_user'] = user_amounts['sum'].mean()
            monetary_metrics['high_value_users_90th_percentile'] = (user_amounts['sum'] > user_amounts['sum'].quantile(0.9)).sum()
            monetary_metrics['avg_transaction_amount'] = user_amounts['mean'].mean()
        
        # Overall statistics
        overall_stats = {
            'total_users': len(user_metrics),
            'total_transactions': len(df),
            'avg_transactions_per_user': len(df) / len(user_metrics),
            'date_range': f"{df['timestamp'].min().date()} to {df['timestamp'].max().date()}" if 'timestamp' in df.columns else "N/A"
        }
        
        return {
            'overall_statistics': overall_stats,
            'recency_metrics': recency_metrics,
            'activity_span_metrics': activity_span,
            'diversity_metrics': diversity_metrics,
            'monetary_metrics': monetary_metrics,
            'user_metrics_dataframe': user_metrics
        }
    
    def create_anomaly_dashboard(self, df: pd.DataFrame, anomaly_scores: np.ndarray, 
                               top_anomalies: pd.DataFrame) -> go.Figure:
        """Create comprehensive anomaly detection dashboard."""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Anomaly Score Distribution', 'Top Anomalies by Amount',
                          'Anomalies by Location', 'Anomalies by Time'),
            specs=[[{"secondary_y": True}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Score distribution
        fig.add_trace(
            go.Histogram(x=anomaly_scores, nbinsx=50, name='Score Distribution'),
            row=1, col=1
        )
        
        # Top anomalies scatter
        if len(top_anomalies) > 0:
            fig.add_trace(
                go.Scatter(x=top_anomalies['amount'], y=top_anomalies['anomaly_score'],
                          mode='markers', name='Top Anomalies',
                          text=top_anomalies['user_id'],
                          hovertemplate='Amount: %{x}<br>Score: %{y}<br>User: %{text}'),
                row=1, col=2
            )
            
            # Anomalies by location
            if 'location' in top_anomalies.columns:
                location_counts = top_anomalies['location'].value_counts()
                fig.add_trace(
                    go.Bar(x=location_counts.index, y=location_counts.values,
                          name='Anomalies by Location'),
                    row=2, col=1
                )
            
            # Anomalies by time
            if 'hour' in top_anomalies.columns:
                fig.add_trace(
                    go.Scatter(x=top_anomalies['hour'], y=top_anomalies['anomaly_score'],
                              mode='markers', name='Anomalies by Hour'),
                    row=2, col=2
                )
        
        fig.update_layout(height=800, showlegend=True, 
                         title_text="Fraud Detection Dashboard")
        return fig
    
    def plot_model_comparison(self, model_scores: Dict[str, np.ndarray]) -> plt.Figure:
        """Compare different model performance."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Box plot comparison
        scores_data = []
        labels = []
        for model_name, scores in model_scores.items():
            scores_data.append(scores)
            labels.append(model_name)
        
        axes[0].boxplot(scores_data, labels=labels)
        axes[0].set_title('Model Score Comparison')
        axes[0].set_ylabel('Anomaly Score')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Correlation heatmap
        if len(model_scores) > 1:
            score_df = pd.DataFrame(model_scores)
            correlation = score_df.corr()
            sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, ax=axes[1])
            axes[1].set_title('Model Score Correlations')
        
        plt.tight_layout()
        return fig

# Global visualization instance
viz = FraudVisualization()
