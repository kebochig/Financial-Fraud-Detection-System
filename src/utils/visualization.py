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
        
        # Amount distribution
        if 'amount' in df.columns:
            axes[1, 1].hist(df['amount'].dropna(), bins=50, alpha=0.7)
            axes[1, 1].set_title('Amount Distribution')
            axes[1, 1].set_xlabel('Amount')
            axes[1, 1].set_ylabel('Frequency')
        
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
