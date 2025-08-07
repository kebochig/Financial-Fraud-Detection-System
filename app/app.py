import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add the project root to the path (parent directory)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from src.parser.log_parser import TransactionLogParser
from src.features.feature_extractor import TransactionFeatureExtractor
from src.models.rule_based import RuleBasedAnomalyDetector
from src.models.statistical import StatisticalAnomalyDetector
from src.models.ensemble import EnsembleAnomalyDetector
from src.utils.config import config
from src.utils.visualization import viz

# Set page config
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 2px solid #e9ecef;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        text-align: center;
    }
    .metric-card h3 {
        color: #495057;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card h2 {
        color: #212529;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .success-metric {
        border-left: 4px solid #28a745;
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    }
    .success-metric h2 {
        color: #155724;
    }
    .warning-metric {
        border-left: 4px solid #ffc107;
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
    }
    .warning-metric h2 {
        color: #856404;
    }
    .danger-metric {
        border-left: 4px solid #dc3545;
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
    }
    .danger-metric h2 {
        color: #721c24;
    }
    .info-metric {
        border-left: 4px solid #17a2b8;
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
    }
    .info-metric h2 {
        color: #0c5460;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'parsed_data' not in st.session_state:
    st.session_state.parsed_data = None
if 'parsing_stats' not in st.session_state:
    st.session_state.parsing_stats = None

def load_data():
    """Load and parse the transaction data"""
    if not st.session_state.data_loaded:
        with st.spinner("Loading and parsing transaction data..."):
            try:
                # Initialize parser
                parser = TransactionLogParser()
                
                # Parse the dataset
                data_file = '../synthetic_dirty_transaction_logs.csv'
                df_parsed, parsing_stats = parser.parse_dataset(data_file)
                
                # Filter valid transactions
                df_valid = df_parsed[df_parsed['is_parsed'] == True].copy()
                
                # Add temporal features
                df_valid['hour'] = df_valid['timestamp'].dt.hour
                df_valid['day_of_week'] = df_valid['timestamp'].dt.dayofweek
                df_valid['is_weekend'] = df_valid['day_of_week'].isin([5, 6])
                
                st.session_state.parsed_data = df_valid
                st.session_state.parsing_stats = parsing_stats
                st.session_state.data_loaded = True
                
                st.success("Data loaded successfully!")
                
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                return None, None
    
    return st.session_state.parsed_data, st.session_state.parsing_stats

def load_uploaded_data(uploaded_file):
    """Load and parse uploaded transaction data"""
    with st.spinner("Processing uploaded file..."):
        try:
            # Read the uploaded file
            if uploaded_file.name.endswith('.csv'):
                df_raw = pd.read_csv(uploaded_file)
            else:
                st.error("Please upload a CSV file")
                return None, None
            
            # Check if the file has the expected structure
            if 'raw_log' not in df_raw.columns:
                st.error("Uploaded file must contain a 'raw_log' column")
                return None, None
            
            # Initialize parser
            parser = TransactionLogParser()
            
            # Parse the uploaded data
            df_parsed, parsing_stats = parser.parse_dataset_from_dataframe(df_raw)
            
            # Filter valid transactions
            df_valid = df_parsed[df_parsed['is_parsed'] == True].copy()
            
            if len(df_valid) == 0:
                st.error("No valid transactions found in uploaded file")
                return None, None
            
            # Add temporal features
            df_valid['hour'] = df_valid['timestamp'].dt.hour
            df_valid['day_of_week'] = df_valid['timestamp'].dt.dayofweek
            df_valid['is_weekend'] = df_valid['day_of_week'].isin([5, 6])
            
            st.success(f"Successfully processed {len(df_valid)} transactions from uploaded file!")
            
            return df_valid, parsing_stats
            
        except Exception as e:
            st.error(f"Error processing uploaded file: {str(e)}")
            return None, None

def main():
    # Header
    st.markdown('<h1 class="main-header">🕵️ Fraud Detection System</h1>', unsafe_allow_html=True)
    
    # File upload section
    st.sidebar.title("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload transaction logs (CSV)",
        type=['csv'],
        help="Upload a CSV file with a 'raw_log' column containing transaction logs"
    )
    
    # Sample data download
    if st.sidebar.button("📥 Download Sample Data"):
        sample_data = pd.DataFrame({
            'raw_log': [
                "2025-07-05 19:18:10::user1069::withdrawal::2995.12::London::iPhone 13",
                "usr:user1076|cashout|€4821.85|Glasgow|2025-07-15 12:56:05|Pixel 6",
                "2025-07-20 05:38:14 >> [user1034] did top-up - amt=€2191.06 - None // dev:iPhone 13",
                "2025-06-23 14:45:58 - user=user1075 - action=debit $1215.74 - ATM: Leeds - device=Samsung Galaxy S10",
                "24/07/2025 22:47:06 ::: user1080 *** PURCHASE ::: amt:951.85$ @ Liverpool <Xiaomi Mi 11>"
            ]
        })
        
        csv = sample_data.to_csv(index=False)
        st.sidebar.download_button(
            label="Download sample.csv",
            data=csv,
            file_name="sample_transaction_logs.csv",
            mime="text/csv"
        )
    
    # Data loading logic
    df_valid = None
    parsing_stats = None
    
    if uploaded_file is not None:
        # Process uploaded file
        df_valid, parsing_stats = load_uploaded_data(uploaded_file)
        if df_valid is not None:
            st.session_state.parsed_data = df_valid
            st.session_state.parsing_stats = parsing_stats
            st.session_state.data_loaded = True
    # else:
    #     # Load default data
    #     df_valid, parsing_stats = load_data()
    
    if df_valid is None:
        st.error("No data available. Please upload a CSV file or ensure the default data file exists.")
        st.info("""
        **Expected CSV format:**
        - Must contain a 'raw_log' column
        - Each row should contain a transaction log entry
        - Example: `2025-07-05 19:18:10::user1069::withdrawal::2995.12::London::iPhone 13`
        """)
        return
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Dashboard", "📊 Data Exploration", "🔍 Log Parsing Analysis", 
         "📈 Data Quality Report", "👤 User Behavior Analysis", 
         "💰 Transaction Analysis", "🚨 Fraud Detection"] #"📋 Reports"
    )
    
    # Data source indicator
    if uploaded_file is not None:
        st.sidebar.success(f"📊 Using uploaded data: {uploaded_file.name}")
        st.sidebar.info(f"📈 {len(df_valid):,} transactions loaded")
    else:
        st.sidebar.info("📊 Using default dataset")
        st.sidebar.info(f"📈 {len(df_valid):,} transactions loaded")
    
    # Page routing
    if page == "🏠 Dashboard":
        show_dashboard(df_valid, parsing_stats)
    elif page == "📊 Data Exploration":
        show_data_exploration(df_valid)
    elif page == "🔍 Log Parsing Analysis":
        show_parsing_analysis(parsing_stats)
    elif page == "📈 Data Quality Report":
        show_data_quality(df_valid, parsing_stats)
    elif page == "👤 User Behavior Analysis":
        show_user_behavior(df_valid)
    elif page == "💰 Transaction Analysis":
        show_transaction_analysis(df_valid)
    elif page == "🚨 Fraud Detection":
        show_fraud_detection(df_valid)
    # elif page == "📋 Reports":
    #     show_reports(df_valid)

def show_dashboard(df_valid, parsing_stats):
    """Main dashboard with key metrics"""
    st.title("🏠 System Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card success-metric">
            <h3>Total Transactions</h3>
            <h2>{:,}</h2>
        </div>
        """.format(len(df_valid)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card info-metric">
            <h3>Unique Users</h3>
            <h2>{:,}</h2>
        </div>
        """.format(df_valid['user_id'].nunique()), unsafe_allow_html=True)
    
    with col3:
        success_rate = (parsing_stats['parsed_successfully'] / parsing_stats['total_logs']) * 100
        if success_rate >= 80:
            metric_class = "success-metric"
        elif success_rate >= 60:
            metric_class = "warning-metric"
        else:
            metric_class = "danger-metric"
        st.markdown("""
        <div class="metric-card {}">
            <h3>Parsing Success Rate</h3>
            <h2>{:.1f}%</h2>
        </div>
        """.format(metric_class, success_rate), unsafe_allow_html=True)
    
    with col4:
        avg_amount = df_valid['amount'].mean()
        st.markdown("""
        <div class="metric-card info-metric">
            <h3>Average Transaction</h3>
            <h2>£{:.0f}</h2>
        </div>
        """.format(avg_amount), unsafe_allow_html=True)
    
    # Date range
    st.subheader("📅 Data Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**Date Range:** {df_valid['timestamp'].min().strftime('%Y-%m-%d')} to {df_valid['timestamp'].max().strftime('%Y-%m-%d')}")
        st.info(f"**Transaction Types:** {df_valid['transaction_type'].nunique()}")
        st.info(f"**Locations:** {df_valid['location'].nunique()}")
    
    with col2:
        st.info(f"**Currencies:** {df_valid['currency'].nunique()}")
        st.info(f"**Devices:** {df_valid['device'].nunique()}")
        st.info(f"**Total Value:** £{df_valid['amount'].sum():,.0f}")
    
    # Quick charts
    st.subheader("📊 Quick Insights")
    col1, col2 = st.columns(2)
    
    with col1:
        # Transaction type distribution
        type_counts = df_valid['transaction_type'].value_counts()
        fig = px.pie(values=type_counts.values, names=type_counts.index, title="Transaction Types")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Amount distribution
        fig = px.histogram(df_valid, x='amount', nbins=50, title="Transaction Amount Distribution")
        st.plotly_chart(fig, use_container_width=True)

def show_data_exploration(df_valid):
    """Data exploration page"""
    st.title("📊 Data Exploration")
    
    # Basic statistics
    st.subheader("📈 Basic Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Amount Statistics:**")
        amount_stats = df_valid['amount'].describe()
        st.dataframe(amount_stats)
    
    with col2:
        st.write("**Transaction Count by Type:**")
        type_counts = df_valid['transaction_type'].value_counts()
        st.dataframe(type_counts)
    
    # Data completeness
    st.subheader("🔍 Data Completeness")
    
    completeness = {}
    for col in ['timestamp', 'user_id', 'transaction_type', 'amount', 'currency', 'location', 'device']:
        non_null = df_valid[col].notna().sum()
        total = len(df_valid)
        completeness[col] = (non_null / total) * 100
    
    completeness_df = pd.DataFrame(list(completeness.items()), columns=['Field', 'Completeness (%)'])
    fig = px.bar(completeness_df, x='Field', y='Completeness (%)', title="Data Completeness by Field")
    st.plotly_chart(fig, use_container_width=True)
    
    # Currency and location analysis
    st.subheader("💰 Currency & Location Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        currency_counts = df_valid['currency'].value_counts()
        fig = px.pie(values=currency_counts.values, names=currency_counts.index, title="Currency Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        location_counts = df_valid['location'].value_counts().head(10)
        fig = px.bar(x=location_counts.values, y=location_counts.index, orientation='h', title="Top 10 Locations")
        st.plotly_chart(fig, use_container_width=True)

def show_parsing_analysis(parsing_stats):
    """Log parsing analysis page"""
    st.title("🔍 Log Parsing Analysis")
    
    # Parsing statistics
    st.subheader("📊 Parsing Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Logs", f"{parsing_stats['total_logs']:,}")
    
    with col2:
        st.metric("Successfully Parsed", f"{parsing_stats['parsed_successfully']:,}")
    
    with col3:
        st.metric("Parsing Failed", f"{parsing_stats['parsing_failed']:,}")
    
    with col4:
        success_rate = (parsing_stats['parsed_successfully'] / parsing_stats['total_logs']) * 100
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    # Parsing breakdown
    st.subheader("📋 Parsing Breakdown")
    
    parsing_data = {
        'Category': ['Successfully Parsed', 'Parsing Failed', 'Empty Logs', 'Malformed Logs'],
        'Count': [
            parsing_stats['parsed_successfully'],
            parsing_stats['parsing_failed'],
            parsing_stats['empty_logs'],
            parsing_stats['malformed_logs']
        ]
    }
    
    parsing_df = pd.DataFrame(parsing_data)
    fig = px.pie(parsing_df, values='Count', names='Category', title="Log Parsing Results")
    st.plotly_chart(fig, use_container_width=True)
    
    # Quality metrics
    st.subheader("🎯 Quality Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if success_rate >= 80:
            st.success(f"Parsing Efficiency: High ({success_rate:.1f}%)")
        elif success_rate >= 60:
            st.warning(f"Parsing Efficiency: Medium ({success_rate:.1f}%)")
        else:
            st.error(f"Parsing Efficiency: Low ({success_rate:.1f}%)")
    
    with col2:
        usable_ratio = parsing_stats['parsed_successfully'] / parsing_stats['total_logs']
        if usable_ratio >= 0.8:
            st.success(f"Data Quality: Good ({usable_ratio:.1%})")
        elif usable_ratio >= 0.6:
            st.warning(f"Data Quality: Fair ({usable_ratio:.1%})")
        else:
            st.error(f"Data Quality: Poor ({usable_ratio:.1%})")
    
    with col3:
        error_rate = (parsing_stats['parsing_failed'] + parsing_stats['empty_logs'] + parsing_stats['malformed_logs']) / parsing_stats['total_logs']
        st.info(f"Error Rate: {error_rate:.1%}")

def show_data_quality(df_valid, parsing_stats):
    """Data quality report page"""
    st.title("📈 Data Quality Report")
    
    # Load saved visualization if available
    if os.path.exists('../results/data_quality_report.png'):
        st.image('../results/data_quality_report.png', caption="Data Quality Report", use_column_width=True)
    else:
        st.warning("Data quality report image not found. Generating live report...")
        
        # Create live data quality report
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Data Quality Report', fontsize=16, fontweight='bold')
        
        # Parsing success rate
        parsing_data = [parsing_stats['parsed_successfully'], 
                       parsing_stats['parsing_failed'] + parsing_stats['empty_logs'] + parsing_stats['malformed_logs']]
        axes[0, 0].pie(parsing_data, labels=['Success', 'Failed'], autopct='%1.1f%%')
        axes[0, 0].set_title('Parsing Success Rate')
        
        # Data completeness
        completeness = []
        labels = []
        for col in ['timestamp', 'user_id', 'transaction_type', 'amount', 'currency', 'location', 'device']:
            non_null = df_valid[col].notna().sum()
            total = len(df_valid)
            completeness.append((non_null / total) * 100)
            labels.append(col)
        
        axes[0, 1].bar(labels, completeness)
        axes[0, 1].set_title('Data Completeness by Field')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Transaction type distribution
        type_counts = df_valid['transaction_type'].value_counts()
        axes[1, 0].bar(range(len(type_counts)), type_counts.values)
        axes[1, 0].set_title('Transaction Type Distribution')
        axes[1, 0].set_xticks(range(len(type_counts)))
        axes[1, 0].set_xticklabels(type_counts.index, rotation=45)
        
        # Amount distribution
        axes[1, 1].hist(df_valid['amount'], bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Transaction Amount Distribution')
        axes[1, 1].set_xlabel('Amount')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Temporal patterns in data quality
    st.subheader("⏰ Temporal Patterns Analysis")
    try:
        fig_temporal = viz.plot_temporal_patterns(df_valid)
        st.pyplot(fig_temporal)
        st.success("Temporal patterns analysis completed!")
    except Exception as e:
        st.error(f"Error generating temporal patterns: {str(e)}")
        st.info("This visualization requires temporal data and may not be available for all datasets.")

def show_user_behavior(df_valid):
    """User behavior analysis page"""
    st.title("👤 User Behavior Analysis")
    
    # Load saved visualization if available
    if os.path.exists('../results/user_behavior_analysis.png'):
        st.image('../results/user_behavior_analysis.png', caption="User Behavior Analysis", use_column_width=True)
    
    # Advanced user behavior analysis
    st.subheader("📊 Advanced User Behavior Analysis")
    try:
        fig_advanced = viz.plot_user_behavior_analysis_advanced(df_valid)
        st.pyplot(fig_advanced)
        st.success("Advanced user behavior analysis generated successfully!")
    except Exception as e:
        st.error(f"Error generating advanced user behavior analysis: {str(e)}")
        st.info("This visualization requires additional data processing and may not be available for all datasets.")
    
    # User behavior summary
    st.subheader("📋 User Behavior Summary")
    try:
        if hasattr(viz, 'get_user_behavior_summary'):
            summary = viz.get_user_behavior_summary(df_valid)
            
            # Display summary statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Overall Statistics:**")
                overall_stats = summary['overall_statistics']
                st.write(f"- Total Users: {overall_stats['total_users']:,}")
                st.write(f"- Total Transactions: {overall_stats['total_transactions']:,}")
                st.write(f"- Avg Transactions per User: {overall_stats['avg_transactions_per_user']:.1f}")
                st.write(f"- Date Range: {overall_stats['date_range']}")
            
            with col2:
                st.write("**Activity Metrics:**")
                recency_metrics = summary['recency_metrics']
                st.write(f"- Recent Users (7 days): {recency_metrics['recent_users_7_days']:,}")
                st.write(f"- Recent Users (30 days): {recency_metrics['recent_users_30_days']:,}")
                st.write(f"- Inactive Users (90 days): {recency_metrics['inactive_users_90_days']:,}")
            
            # Display diversity metrics
            st.write("**Diversity Metrics:**")
            diversity_metrics = summary['diversity_metrics']
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"- Avg Unique Locations: {diversity_metrics['avg_unique_locations']:.1f}")
                st.write(f"- Multi-location Users: {diversity_metrics['mobile_users_multiple_locations']:,}")
            
            with col2:
                st.write(f"- Avg Unique Devices: {diversity_metrics['avg_unique_devices']:.1f}")
                st.write(f"- Multi-device Users: {diversity_metrics['multi_device_users']:,}")
            
            with col3:
                st.write(f"- Avg Transaction Types: {diversity_metrics['avg_unique_transaction_types']:.1f}")
                st.write(f"- Diverse Transaction Users: {diversity_metrics['diverse_transaction_users']:,}")
            
        else:
            st.info("User behavior summary not available in this version.")
    except Exception as e:
        st.error(f"Error generating user behavior summary: {str(e)}")
    
    # User statistics
    st.subheader("📊 User Statistics")
    
    user_stats = df_valid.groupby('user_id').agg({
        'amount': ['count', 'mean', 'std', 'min', 'max'],
        'location': 'nunique',
        'device': 'nunique',
        'transaction_type': 'nunique'
    }).round(2)
    
    user_stats.columns = ['tx_count', 'avg_amount', 'std_amount', 'min_amount', 'max_amount', 
                         'unique_locations', 'unique_devices', 'unique_types']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**User Statistics Summary:**")
        st.dataframe(user_stats.describe())
    
    with col2:
        st.write("**Top 10 Most Active Users:**")
        top_users = user_stats.sort_values('tx_count', ascending=False).head(10)
        st.dataframe(top_users)
    
    # User behavior visualizations
    st.subheader("📈 User Behavior Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Transaction count distribution
        fig = px.histogram(user_stats, x='tx_count', nbins=30, title="Transaction Count Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Average amount vs transaction count
        fig = px.scatter(user_stats, x='tx_count', y='avg_amount', title="Transaction Count vs Average Amount")
        st.plotly_chart(fig, use_container_width=True)
    
    # Location and device diversity
    st.subheader("🌍 Location & Device Diversity")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(user_stats, x='unique_locations', nbins=10, title="Location Diversity per User")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(user_stats, x='unique_devices', nbins=10, title="Device Diversity per User")
        st.plotly_chart(fig, use_container_width=True)

def show_transaction_analysis(df_valid):
    """Transaction analysis page"""
    st.title("💰 Transaction Analysis")
    
    # Load saved visualization if available
    if os.path.exists('../results/amount_analysis.png'):
        st.image('../results/amount_analysis.png', caption="Transaction Amount Analysis", use_column_width=True)
    
    if os.path.exists('../results/temporal_patterns.png'):
        st.image('../results/temporal_patterns.png', caption="Temporal Patterns", use_column_width=True)
    
    # Live temporal patterns analysis
    st.subheader("⏰ Live Temporal Patterns Analysis")
    try:
        fig_temporal = viz.plot_temporal_patterns(df_valid)
        st.pyplot(fig_temporal)
        st.success("Temporal patterns analysis generated successfully!")
    except Exception as e:
        st.error(f"Error generating temporal patterns: {str(e)}")
        st.info("This visualization requires temporal data and may not be available for all datasets.")
    
    # Amount analysis
    st.subheader("💵 Amount Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        amount_stats = df_valid['amount'].describe()
        st.write("**Amount Statistics:**")
        st.dataframe(amount_stats)
    
    with col2:
        # Amount by transaction type
        fig = px.box(df_valid, x='transaction_type', y='amount', title="Amount by Transaction Type")
        st.plotly_chart(fig, use_container_width=True)
    
    # Temporal analysis
    st.subheader("⏰ Temporal Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Hourly distribution
        hourly_counts = df_valid['hour'].value_counts().sort_index()
        fig = px.line(x=hourly_counts.index, y=hourly_counts.values, title="Transaction Volume by Hour")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Day of week distribution
        day_counts = df_valid['day_of_week'].value_counts().sort_index()
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        fig = px.bar(x=day_names, y=day_counts.values, title="Transaction Volume by Day of Week")
        st.plotly_chart(fig, use_container_width=True)
    
    # Currency analysis
    st.subheader("💱 Currency Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        currency_amounts = df_valid.groupby('currency')['amount'].agg(['mean', 'std', 'count'])
        st.write("**Amount Statistics by Currency:**")
        st.dataframe(currency_amounts)
    
    with col2:
        fig = px.box(df_valid, x='currency', y='amount', title="Amount Distribution by Currency")
        st.plotly_chart(fig, use_container_width=True)

def show_fraud_detection(df_valid):
    """Fraud detection page"""
    st.title("🚨 Fraud Detection")
    
    st.info("This section demonstrates the fraud detection capabilities of the system.")
    
    # Feature extraction
    st.subheader("🔧 Feature Extraction")
    
    if st.button("Extract Features"):
        with st.spinner("Extracting features..."):
            try:
                feature_extractor = TransactionFeatureExtractor()
                
                # Extract basic features
                df_basic = feature_extractor.extract_basic_features(df_valid)
                st.success(f"Basic features extracted: {len(df_basic.columns)} features")
                
                # Extract user behavioral features
                df_behavioral = feature_extractor.extract_user_behavioral_features(df_basic)
                st.success(f"Behavioral features extracted: {len(df_behavioral.columns)} features")
                
                # Extract temporal features
                df_temporal = feature_extractor.extract_temporal_features(df_behavioral)
                st.success(f"Temporal features extracted: {len(df_temporal.columns)} features")
                
                # Extract contextual features
                df_contextual = feature_extractor.extract_contextual_features(df_temporal)
                st.success(f"Contextual features extracted: {len(df_contextual.columns)} features")
                
                # Store in session state
                st.session_state.features = df_contextual
                st.success("All features extracted successfully!")
                
            except Exception as e:
                st.error(f"Error extracting features: {str(e)}")
    
    # Display engineered features
    if 'features' in st.session_state:
        st.subheader("📊 Engineered Features Overview")
        
        df_features = st.session_state.features
        
        # Feature statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Features", len(df_features.columns))
        
        with col2:
            st.metric("Total Records", len(df_features))
        
        with col3:
            numerical_features = df_features.select_dtypes(include=[np.number]).columns.tolist()
            st.metric("Numerical Features", len(numerical_features))
        
        with col4:
            categorical_features = df_features.select_dtypes(include=['object', 'category']).columns.tolist()
            st.metric("Categorical Features", len(categorical_features))
        
        # Feature categories
        st.subheader("🔍 Feature Categories")
        
        try:
            # Create a new feature extractor instance for categorization
            feature_extractor_for_cat = TransactionFeatureExtractor()
            feature_categories = feature_extractor_for_cat.get_feature_names(df_features)
            
            # Display each category
            for category, features in feature_categories.items():
                if features:  # Only show categories with features
                    with st.expander(f"📋 {category.title()} Features ({len(features)} features)"):
                        # Create a DataFrame for better display
                        feature_df = pd.DataFrame({
                            'Feature Name': features,
                            'Data Type': [str(df_features[col].dtype) for col in features],
                            'Non-Null Count': [df_features[col].count() for col in features],
                            'Null Count': [df_features[col].isnull().sum() for col in features],
                            'Null Percentage': [f"{(df_features[col].isnull().sum() / len(df_features) * 100):.1f}%" for col in features]
                        })
                        
                        # Add statistics for numerical features
                        if category in ['temporal', 'amount', 'user', 'contextual']:
                            min_values = []
                            max_values = []
                            mean_values = []
                            
                            for col in features:
                                if df_features[col].dtype in ['int64', 'float64']:
                                    min_values.append(df_features[col].min())
                                    max_values.append(df_features[col].max())
                                    mean_values.append(df_features[col].mean())
                                else:
                                    min_values.append(None)
                                    max_values.append(None)
                                    mean_values.append(None)
                            
                            feature_df['Min'] = min_values
                            feature_df['Max'] = max_values
                            feature_df['Mean'] = mean_values
                        
                        st.dataframe(feature_df, use_container_width=True)
                        
                        # Show sample values for first few features
                        if len(features) <= 5:
                            st.write("**Sample Values:**")
                            for feature in features[:3]:
                                if df_features[feature].dtype in ['int64', 'float64']:
                                    st.write(f"- {feature}: {df_features[feature].describe()}")
                                else:
                                    st.write(f"- {feature}: {df_features[feature].value_counts().head(3).to_dict()}")
        
        except Exception as e:
            st.error(f"Error categorizing features: {str(e)}")
            
            # Fallback: show all features
            st.write("**All Features:**")
            feature_info = pd.DataFrame({
                'Feature Name': df_features.columns,
                'Data Type': [str(df_features[col].dtype) for col in df_features.columns],
                'Non-Null Count': [df_features[col].count() for col in df_features.columns],
                'Null Count': [df_features[col].isnull().sum() for col in df_features.columns],
                'Null Percentage': [f"{(df_features[col].isnull().sum() / len(df_features) * 100):.1f}%" for col in df_features.columns]
            })
            st.dataframe(feature_info, use_container_width=True)
        
    
    # Fraud detection
    st.subheader("🚨 Anomaly Detection")
    
    if 'features' in st.session_state and st.button("Run Fraud Detection"):
        with st.spinner("Running fraud detection..."):
            try:
                # Prepare features for detection
                df_features = st.session_state.features.copy()
                
                # Select only numerical features for statistical models
                numerical_features = df_features.select_dtypes(include=[np.number]).columns.tolist()
                
                if len(numerical_features) < 5:
                    st.error("Not enough numerical features for fraud detection. Please ensure feature extraction is complete.")
                    return
                
                # Use only numerical features for statistical detection
                df_numerical = df_features[numerical_features]
                # Rule-based detection
                rule_detector = RuleBasedAnomalyDetector()
                rule_scores = rule_detector.detect_anomalies(st.session_state.features)
                
                # Statistical detection
                statistical_detector = StatisticalAnomalyDetector()
                # Fit the models first, then predict
                statistical_detector.fit(df_numerical)
                statistical_scores = statistical_detector.predict_anomalies(df_numerical)
                
                # Ensemble detection
                ensemble_detector = EnsembleAnomalyDetector()
                # Fit the ensemble detector first, then detect
                ensemble_detector.fit(st.session_state.features)
                
                # Capture the ensemble report output
                import io
                import sys
                
                # Redirect stdout to capture the report
                old_stdout = sys.stdout
                report_output = io.StringIO()
                sys.stdout = report_output
                
                try:
                    ensemble_results = ensemble_detector.detect_anomalies(st.session_state.features)
                    # Get the captured report
                    ensemble_report = report_output.getvalue()
                finally:
                    # Restore stdout
                    sys.stdout = old_stdout
                    report_output.close()
                
                st.success("Fraud detection completed!")
                
                # Display quick summary
                st.info(f"✅ Rule-based detection: {len(rule_scores)} scores generated")
                st.info(f"✅ Statistical detection: {len(statistical_scores)} methods completed")
                st.info(f"✅ Ensemble detection: Combined scores generated")
                
                # Display the comprehensive ensemble report
                st.subheader("📋 Comprehensive Fraud Detection Report")
                
                # Create expandable section for the full report
                with st.expander("📄 View Complete Ensemble Report", expanded=True):
                    # Format the report for better display
                    report_lines = ensemble_report.split('\n')
                    
                    # Display the report with proper formatting
                    for line in report_lines:
                        if line.strip():
                            if line.startswith('='):
                                st.markdown(f"**{line}**")
                            elif line.startswith('📊') or line.startswith('⚠️') or line.startswith('🔥') or line.startswith('⚡') or line.startswith('👥') or line.startswith('🔍') or line.startswith('👤') or line.startswith('🎯') or line.startswith('💾'):
                                st.markdown(f"**{line}**")
                            elif line.startswith('•'):
                                st.write(line)
                            elif line.startswith('Rank') or line.startswith('----'):
                                st.code(line)
                            elif line.startswith('1') or line.startswith('2') or line.startswith('3') or line.startswith('4') or line.startswith('5') or line.startswith('6') or line.startswith('7') or line.startswith('8') or line.startswith('9'):
                                st.code(line)
                            else:
                                st.write(line)
                
                # Show export status
                st.success("📄 Report has been automatically exported to the results directory")
                
                # Get the latest generated files
                reports_dir = "../results"
                if os.path.exists(reports_dir):
                    txt_files = [f for f in os.listdir(reports_dir) if f.startswith('fraud_report_') and f.endswith('.txt')]
                    csv_files = [f for f in os.listdir(reports_dir) if f.startswith('fraud_report_detailed_') and f.endswith('.csv')]
                    
                    if txt_files:
                        latest_txt = max(txt_files, key=lambda x: os.path.getmtime(os.path.join(reports_dir, x)))
                        latest_csv = max(csv_files, key=lambda x: os.path.getmtime(os.path.join(reports_dir, x))) if csv_files else None
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.info(f"📄 **Text Report**: `{latest_txt}`")
                            st.write(f"📁 Location: `results/{latest_txt}`")
                        
                        with col2:
                            if latest_csv:
                                st.info(f"📊 **CSV Report**: `{latest_csv}`")
                                st.write(f"📁 Location: `results/{latest_csv}`")
                
                st.info("💾 Check the 'Reports' page to download the generated files")
                
                # Display latest results summary
                st.subheader("📊 Latest Detection Summary")
                
                # Get ensemble results summary
                ensemble_summary = ensemble_results[1]  # Results dictionary
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Mean Score", f"{ensemble_summary.get('mean_score', 0):.4f}")
                
                with col2:
                    st.metric("Std Score", f"{ensemble_summary.get('std_score', 0):.4f}")
                
                with col3:
                    high_risk = ensemble_summary.get('anomaly_count_threshold_80', 0)
                    st.metric("High Risk (>0.8)", f"{high_risk:,}")
                
                with col4:
                    medium_risk = ensemble_summary.get('anomaly_count_threshold_70', 0) - high_risk
                    st.metric("Medium Risk (0.7-0.8)", f"{medium_risk:,}")
                
                # Show key insights
                st.write("**🔍 Key Insights:**")
                st.write(f"• **Total Transactions Analyzed**: {len(ensemble_results[0]):,}")
                st.write(f"• **Detection Rate**: {(high_risk + medium_risk) / len(ensemble_results[0]) * 100:.2f}%")
                st.write(f"• **Model Confidence**: {'HIGH' if ensemble_summary.get('mean_score', 0) > 0.5 else 'MEDIUM' if ensemble_summary.get('mean_score', 0) > 0.3 else 'LOW'}")
                
                # Show top anomalies if available
                if 'component_scores' in ensemble_summary:
                    st.write("**📈 Component Performance:**")
                    component_scores = ensemble_summary['component_scores']
                    
                    if 'rule_based' in component_scores:
                        rule_mean = np.mean(component_scores['rule_based']['scores'])
                        st.write(f"• **Rule-based**: Mean score {rule_mean:.4f}")
                    
                    if 'statistical' in component_scores:
                        stat_scores = component_scores['statistical']['scores']
                        if isinstance(stat_scores, dict):
                            for method, scores in stat_scores.items():
                                if isinstance(scores, np.ndarray):
                                    method_mean = np.mean(scores)
                                    st.write(f"• **{method.replace('_', ' ').title()}**: Mean score {method_mean:.4f}")
                
                # Store results in session state
                st.session_state.detection_results = {
                    'rule_scores': rule_scores,
                    'statistical_scores': statistical_scores,
                    'ensemble_scores': ensemble_results[0],  # First element is the scores array
                    'ensemble_results': ensemble_results[1]  # Second element is the results dict
                }
                
                # Display results
                st.subheader("📊 Detection Results")
                
                try:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if isinstance(rule_scores, np.ndarray):
                            rule_anomalies = np.sum(rule_scores > 0.5)
                        else:
                            rule_anomalies = 0
                        st.metric("Rule-based Anomalies", f"{rule_anomalies:,}")
                    
                    with col2:
                        # Count anomalies from statistical methods
                        total_statistical = 0
                        for method, scores in statistical_scores.items():
                            if isinstance(scores, np.ndarray):
                                total_statistical += np.sum(scores > 0.5)
                            else:
                                total_statistical += 0
                        st.metric("Statistical Anomalies", f"{total_statistical:,}")
                    
                    with col3:
                        ensemble_scores = ensemble_results[0]  # Get scores from tuple
                        if isinstance(ensemble_scores, np.ndarray):
                            ensemble_anomalies = np.sum(ensemble_scores > 0.5)
                        else:
                            ensemble_anomalies = 0
                        st.metric("Ensemble Anomalies", f"{ensemble_anomalies:,}")
                        
                except Exception as e:
                    st.error(f"Error displaying detection results: {str(e)}")
                    st.write("Debug info:")
                    st.write(f"Rule scores type: {type(rule_scores)}")
                    st.write(f"Statistical scores type: {type(statistical_scores)}")
                    st.write(f"Ensemble results type: {type(ensemble_results)}")
                    if isinstance(ensemble_results, tuple):
                        st.write(f"Ensemble results length: {len(ensemble_results)}")
                        st.write(f"Ensemble results[0] type: {type(ensemble_results[0])}")
                
            except Exception as e:
                st.error(f"Error in fraud detection: {str(e)}")
    
    # Display detailed results if available
    if 'detection_results' in st.session_state:
        st.subheader("📋 Detailed Results")
        
        ensemble_scores = st.session_state.detection_results['ensemble_scores']
        
        # Risk level breakdown
        risk_levels = {
            'CRITICAL': ensemble_scores > 0.8,
            'HIGH': (ensemble_scores > 0.7) & (ensemble_scores <= 0.8),
            'MEDIUM': (ensemble_scores > 0.5) & (ensemble_scores <= 0.7),
            'LOW': (ensemble_scores > 0.3) & (ensemble_scores <= 0.5),
            'NORMAL': ensemble_scores <= 0.3
        }
        
        risk_counts = {level: np.sum(mask) for level, mask in risk_levels.items()}
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(values=list(risk_counts.values()), names=list(risk_counts.keys()), 
                        title="Risk Level Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(x=ensemble_scores, nbins=50, title="Ensemble Score Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        # Display generated reports
        st.subheader("📄 Generated Reports")
        
        # Check for generated reports
        reports_dir = "../results"
        if os.path.exists(reports_dir):
            report_files = [f for f in os.listdir(reports_dir) if f.startswith('fraud_report_') and f.endswith('.txt')]
            
            if report_files:
                # Sort by modification time (newest first)
                report_files.sort(key=lambda x: os.path.getmtime(os.path.join(reports_dir, x)), reverse=True)
                
                st.write("**Latest Generated Reports:**")
                
                for i, report_file in enumerate(report_files[:3]):  # Show latest 3 reports
                    file_path = os.path.join(reports_dir, report_file)
                    file_size = os.path.getsize(file_path)
                    mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    
                    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                    
                    with col1:
                        st.write(f"**{report_file}**")
                    
                    with col2:
                        st.write(f"{file_size:,} bytes")
                    
                    with col3:
                        st.write(mod_time.strftime("%H:%M"))
                    
                    with col4:
                        if st.button(f"View {i+1}", key=f"view_{report_file}"):
                            with open(file_path, 'r') as f:
                                report_content = f.read()
                                st.text_area("Report Content", report_content, height=400)
                
                # Show CSV reports
                csv_files = [f for f in os.listdir(reports_dir) if f.startswith('fraud_report_detailed_') and f.endswith('.csv')]
                if csv_files:
                    csv_files.sort(key=lambda x: os.path.getmtime(os.path.join(reports_dir, x)), reverse=True)
                    
                    st.write("**Detailed CSV Reports:**")
                    for csv_file in csv_files[:2]:  # Show latest 2 CSV reports
                        file_path = os.path.join(reports_dir, csv_file)
                        file_size = os.path.getsize(file_path)
                        
                        col1, col2, col3 = st.columns([3, 1, 1])
                        
                        with col1:
                            st.write(f"**{csv_file}**")
                        
                        with col2:
                            st.write(f"{file_size:,} bytes")
                        
                        with col3:
                            if st.button(f"Download {csv_file}", key=f"download_{csv_file}"):
                                with open(file_path, 'r') as f:
                                    st.download_button(
                                        label="Download CSV",
                                        data=f.read(),
                                        file_name=csv_file,
                                        mime="text/csv"
                                    )
            else:
                st.info("No fraud reports generated yet. Run fraud detection to generate reports.")
        else:
            st.warning("Results directory not found.")

# def show_reports(df_valid):
#     """Reports page"""
#     st.title("📋 Reports")
    
#     st.info("This section provides access to generated reports and exports.")
    
#     # Check for existing reports
#     reports_dir = "../reports"
#     if os.path.exists(reports_dir):
#         report_files = [f for f in os.listdir(reports_dir) if f.endswith('.txt') or f.endswith('.csv')]
        
#         if report_files:
#             st.subheader("📄 Available Reports")
            
#             for report_file in sorted(report_files, reverse=True):
#                 file_path = os.path.join(reports_dir, report_file)
#                 file_size = os.path.getsize(file_path)
                
#                 col1, col2, col3 = st.columns([3, 1, 1])
                
#                 with col1:
#                     st.write(f"**{report_file}**")
                
#                 with col2:
#                     st.write(f"{file_size:,} bytes")
                
#                 with col3:
#                     if st.button(f"Download {report_file}", key=report_file):
#                         with open(file_path, 'r') as f:
#                             st.download_button(
#                                 label="Download",
#                                 data=f.read(),
#                                 file_name=report_file,
#                                 mime="text/plain"
#                             )
#         else:
#             st.warning("No reports found in the reports directory.")
#     else:
#         st.warning("Reports directory not found.")
    
#     # Generate new report
#     st.subheader("🔄 Generate New Report")
    
#     if st.button("Generate Comprehensive Report"):
#         with st.spinner("Generating report..."):
#             try:
#                 # This would integrate with the ensemble detector to generate a new report
#                 st.info("Report generation functionality would be integrated here.")
#                 st.success("Report generation completed!")
#             except Exception as e:
#                 st.error(f"Error generating report: {str(e)}")

if __name__ == "__main__":
    main() 