# Fraud Detection System - Streamlit App

A comprehensive web application for exploring and analyzing the fraud detection system capabilities.

## Features

### 🏠 Dashboard
- Key metrics overview
- Quick insights and visualizations
- System performance indicators

### 📊 Data Exploration
- Basic statistics and data overview
- Data completeness analysis
- Currency and location distribution

### 🔍 Log Parsing Analysis
- Parsing success rates and statistics
- Quality metrics and error analysis
- Parsing breakdown visualization

### 📈 Data Quality Report
- Comprehensive data quality assessment
- Field completeness analysis
- Data distribution visualizations

### 👤 User Behavior Analysis
- User statistics and patterns
- Transaction behavior analysis
- Location and device diversity

### 💰 Transaction Analysis
- Amount distribution analysis
- Temporal patterns
- Currency analysis

### 🚨 Fraud Detection
- Feature extraction capabilities
- Rule-based anomaly detection
- Statistical anomaly detection
- Ensemble detection results

### 📋 Reports
- Access to generated reports
- Download functionality
- Report generation capabilities

## Installation

1. **Install Streamlit dependencies:**
   ```bash
   pip install -r requirements_streamlit.txt
   ```

2. **Ensure the main fraud detection system is set up:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

3. **Navigate through the different pages** using the sidebar to explore various aspects of the fraud detection system

## Data Requirements

The app expects the following files to be present:
- `synthetic_dirty_transaction_logs.csv` - Raw transaction data
- `results/` directory with generated visualizations and reports

## Navigation

Use the sidebar to navigate between different sections:

- **Dashboard**: Overview of key metrics and quick insights
- **Data Exploration**: Detailed data analysis and statistics
- **Log Parsing Analysis**: Parsing performance and quality metrics
- **Data Quality Report**: Comprehensive data quality assessment
- **User Behavior Analysis**: User patterns and behavior insights
- **Transaction Analysis**: Transaction patterns and amount analysis
- **Fraud Detection**: Live fraud detection capabilities
- **Reports**: Access to generated reports and exports

## Features Highlight

### Interactive Visualizations
- Plotly charts for interactive exploration
- Matplotlib/Seaborn for static visualizations
- Real-time data analysis

### Comprehensive Analysis
- All analysis from the data exploration notebook
- Live feature extraction and fraud detection
- Detailed reporting capabilities

### User-Friendly Interface
- Clean, modern UI design
- Responsive layout
- Easy navigation

## Troubleshooting

1. **Import errors**: Ensure all dependencies are installed and the project structure is correct
2. **Data loading issues**: Check that the required data files exist in the correct locations
3. **Visualization errors**: Ensure matplotlib backend is compatible with Streamlit

## Contributing

To add new features or modify the app:
1. Update the main function to include new page routing
2. Add corresponding page functions
3. Update the sidebar navigation
4. Test thoroughly before deployment 