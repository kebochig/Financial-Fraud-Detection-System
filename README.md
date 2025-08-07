# 🔍 Fraud Detection System

A comprehensive, production-ready anomaly detection system for financial transaction fraud detection. This system processes unstructured transaction logs, extracts meaningful features, and uses multiple rule-based and machine learning approaches to identify fraudulent transactions.

## 🎯 Project Overview

This fraud detection system is designed to meet the requirements outlined in the test specification:

- **Data Understanding & Parsing**: Robust parsing of multiple log formats
- **Feature Engineering**: Creative and domain-relevant feature extraction
- **Anomaly Detection**: Multiple approaches including rule-based, statistical, and ensemble methods
- **Evaluation**: Comprehensive metrics and qualitative analysis
- **Interpretability**: Explainable AI with confidence scores.

## 🏗️ System Architecture

```
fraud_detection_system/
├── src/
│   ├── parser/              # Robust log parsing
│   ├── features/            # Feature extraction & engineering
│   ├── models/              # Anomaly detection models
│   ├── evaluation/          # Model evaluation metrics
│   └── utils/               # Configuration & visualization
├── notebooks/               # Analysis notebooks
├── app/                     # Web demo 
├── tests/                   # Unit tests
└── results/                 # Output files & visualizations
```

## 🚀 Key Features

### 📊 Robust Log Parsing
- **Multi-format support**: Handles 7+ different log formats automatically
- **Error handling**: Graceful handling of malformed and empty logs  
- **Edge case management**: Comprehensive validation and cleaning
- **High success rate**: 85%+ parsing success on dirty data

### 🔧 Advanced Feature Engineering
- **60+ features** extracted from raw transaction data
- **Temporal features**: Time-based patterns and sequences
- **Behavioral features**: User-specific patterns and deviations
- **Contextual features**: Location, device, and interaction patterns
- **Statistical features**: Z-scores, percentiles, and distributions

### 🤖 Multi-Model Approach
1. **Rule-Based Detection** (30% weight)
   - 10 business logic rules
   - Configurable weights and thresholds
   - Explainable decisions

2. **Statistical Methods** (45% weight)
   - Isolation Forest
   - One-Class SVM
   - Local Outlier Factor
   - DBSCAN/HDBSCAN clustering

3. **Ensemble Integration** (25% weight)
   - Weighted combination of all approaches
   - Adaptive thresholds
   - Confidence scoring

### 📈 Comprehensive Evaluation
- **Quantitative metrics**: Distribution analysis, silhouette scores
- **Qualitative analysis**: Business logic validation
- **Visualization**: Interactive dashboards and plots
- **Explainability**: Individual transaction explanations

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- 4GB+ RAM recommended
- Git

### Setup
```bash
# Clone repository
git clone <repository-url>
cd fraud_detection_system

# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install in development mode
pip install -e .
```

### Quick Start
```bash
python test_system.py
```

## 📖 Usage Guide

### 1. Data Exploration
```bash
jupyter notebook notebooks/1_data_exploration.ipynb
```
- Analyze raw transaction logs
- Test parser on different formats
- Generate data quality reports

### 2. Feature Engineering
```bash
jupyter notebook notebooks/2_feature_engineering.ipynb
```
- Extract comprehensive feature set
- Validate feature distributions
- Understand feature engineering reasoning

### 3. Model Training & Evaluation
```bash
jupyter notebook notebooks/3_anomaly_detection.ipynb
```
- Train multiple detection models
- Compare model performance
- Generate and export evaluation metrics

### 4. Web Demo
```bash
streamlit run app/streamlit_app.py
```
- Open Browser and load `http://localhost:8501`
- Interactive fraud detection interface
- Upload transaction logs
- Real-time anomaly detection
- Multiple Visualization dashboards (Side-bar)

#### Navigation

Use the sidebar to navigate between different sections:

- **Dashboard**: Overview of key metrics and quick insights
- **Data Exploration**: Detailed data analysis and statistics
- **Log Parsing Analysis**: Parsing performance and quality metrics
- **Data Quality Report**: Comprehensive data quality assessment
- **User Behavior Analysis**: User patterns and behavior insights
- **Transaction Analysis**: Transaction patterns and amount analysis
- **Fraud Detection**: Live fraud detection capabilities


## 🔍 Model Details

### Rule-Based Detection
- **Large Amount**: Unusually high transaction values
- **Unusual Hours**: Night/early morning transactions  
- **High Velocity**: Multiple transactions in short time
- **Location Anomalies**: Rare or unusual locations
- **Device Changes**: Frequent device switching
- **Amount Deviation**: Significant deviation from user patterns
- **Weekend Activity**: High-value weekend transactions
- **Rare Transaction Types**: Uncommon transaction categories
- **Multiple Locations**: Same-day multi-location activity
- **Round Amounts**: Suspicious round number patterns

### Statistical Methods
- **Isolation Forest**: Tree-based outlier detection
- **One-Class SVM**: Support vector-based novelty detection
- **Local Outlier Factor**: Density-based local outliers
- **DBSCAN**: Density-based clustering with noise detection
- **HDBSCAN**: Hierarchical density-based clustering

### Feature Categories
- **Temporal**: Hour, day, weekend patterns, velocity
- **Amount**: Values, percentiles, deviations, categories
- **User**: Transaction history, behavior patterns, diversity
- **Location**: Frequency, rarity, risk scores, changes
- **Device**: Types, changes, brand categories
- **Contextual**: Interactions, frequencies, global patterns

## 📊 Evaluation Results

### Parsing Performance
- **Success Rate**: 85.2% of logs parsed successfully
- **Format Coverage**: 7 different log formats supported
- **Error Handling**: Graceful degradation for malformed data

### Detection Performance
- **Feature Count**: 60+ engineered features
- **Model Coverage**: 5 different detection algorithms
- **Ensemble Accuracy**: Weighted combination approach
- **Explainability**: Rule-level and feature-level explanations


## 🎨 Visualizations

The system provides comprehensive visualizations:
- **Data Quality Reports**: Parsing success, completeness analysis
- **Feature Distributions**: Histograms, box plots, correlations
- **Anomaly Dashboards**: Interactive score distributions
- **Temporal Patterns**: Time-based transaction analysis
- **User Behavior**: Activity patterns and risk profiles


## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v
```

Test coverage includes:
- Parser robustness testing
- Feature extraction validation
- Model performance testing
- Integration testing

### Developer 
**Name:** Chigozilai Kejeh

**Email:** kebochig@gmail.com

**Profile:** [Link](https://chigozilai-portfolio.netlify.app/)
