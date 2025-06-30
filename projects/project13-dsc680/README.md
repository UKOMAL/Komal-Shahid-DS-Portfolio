# Advanced Fraud Detection System

## Project Overview
This project implements a comprehensive fraud detection system using machine learning techniques and advanced data visualizations. The system is designed to identify fraudulent transactions in financial data, with a focus on credit card fraud detection, enhanced with interactive visualizations and sophisticated analytics.

## Object-Oriented Design
The system follows object-oriented programming principles with the following key classes:

- **FraudDataHandler**: Manages data loading, exploratory analysis, and advanced visualizations
- **FraudDataPreprocessor**: Handles data preprocessing, resampling, and class imbalance
- **FraudModelTrainer**: Trains, evaluates, and optimizes machine learning models
- **FraudDetectionSystem**: Orchestrates the entire workflow and generates interactive dashboards

## Advanced Visualizations
The system includes sophisticated interactive visualizations inspired by the Python Graph Gallery:

### Data Exploration Visualizations
- **Feature Distribution Violin Matrix**: Visualizes the distribution of key features by class
- **Feature Correlation Network**: Interactive network graph showing relationships between features
- **Transaction Amount Ridgeline Plot**: Shows transaction amount distributions by hour
- **Anomaly Detection Visualization**: PCA projection with isolation forest anomaly scores
- **PCA Visualization with Density Contours**: 2D projection with density contours by class
- **Word Co-occurrence Matrix**: Analyzes patterns in transaction descriptions
- **Merchant Risk Analysis**: Bubble chart showing fraud risk by merchant
- **Transaction Type Analysis**: Radar chart comparing different transaction types
- **Geographic Fraud Distribution**: Choropleth map showing fraud rates by country

### Model Evaluation Visualizations
- **Interactive Confusion Matrices**: Detailed heatmaps with performance metrics
- **ROC Curve Comparison**: Interactive comparison of models with threshold markers
- **Precision-Recall Curve Comparison**: Detailed PR curves with AUC values
- **Feature Importance Comparison**: Horizontal bar chart comparing feature importance across models
- **Performance Radar Chart**: Radar visualization of multiple performance metrics
- **Learning Curves**: Visualizations to detect and quantify overfitting
- **Overfitting Comparison**: Bar chart comparing overfitting metrics across models
- **Fairness Metrics Visualization**: Bar charts showing model fairness across different demographic groups

## Features
- Data loading and exploration with advanced pattern detection
- Sophisticated preprocessing with standardization and outlier handling
- Advanced class imbalance handling using multiple techniques:
  - SMOTE (Synthetic Minority Over-sampling Technique)
  - ADASYN (Adaptive Synthetic Sampling)
  - Combined approach (undersampling + controlled SMOTE)
  - Ensemble-based balancing
- Multiple model training and comparison with overfitting prevention:
  - Logistic Regression with L2 regularization
  - Random Forest with depth and sample constraints
  - Gradient Boosting with controlled learning rate
  - XGBoost with comprehensive regularization
- Benford's Law analysis for fraud detection
- Advanced text analytics for transaction descriptions
- Interactive dashboard generation
- Comprehensive overfitting analysis and prevention
- **Autoencoder-based anomaly detection** for identifying unusual transactions
- **BERT-based linguistic analysis** for transaction descriptions
- **Fairness evaluation** across different transaction groups

## Datasets
The system is designed to work with:
1. Credit Card Fraud Detection dataset from Kaggle
2. AI-Powered Banking Fraud Detection Dataset (2025) from Kaggle

Both datasets contain transactions with features that help identify fraudulent activities.

## Requirements
See `requirements.txt` for the necessary dependencies. Key requirements include:
- Python 3.6+
- Plotly for interactive visualizations
- NetworkX for network visualizations
- Scikit-learn, XGBoost for machine learning
- Imbalanced-learn for handling class imbalance
- Pandas, NumPy for data manipulation
- Matplotlib, Seaborn for static visualizations
- NLTK, TextBlob for text analysis
- TensorFlow for autoencoder implementation
- Transformers for BERT-based text analysis
- AIF360 and Fairlearn for fairness evaluation
- Dash for interactive dashboards (optional)

## Usage
1. Download the datasets from Kaggle:
```
# The script will attempt to download the dataset automatically
# If it fails, download manually from Kaggle:
# https://www.kaggle.com/mlg-ulb/creditcardfraud
```

2. Run the fraud detection system:
```
python run_fraud_analysis.py
```

3. Run the Jupyter notebook:
```
python src/run_fraud_notebook.py
```

4. View the interactive dashboard:
```
# Open the generated HTML file in your browser:
projects/project13-dsc680/output/dashboard/index.html
```

## Output
The system generates:
- Interactive visualizations of data distributions and patterns
- Advanced model performance metrics and comparisons
- Interactive confusion matrices with detailed metrics
- ROC and Precision-Recall curves with threshold markers
- Feature importance visualizations and comparisons
- Overfitting analysis with learning curves
- Comprehensive interactive dashboard integrating all visualizations
- Fairness metrics and visualizations across different demographic groups

## Advanced Analytics Features
- **Anomaly Detection**: 
  - Isolation Forest for detecting outliers beyond simple classification
  - Autoencoder-based anomaly detection for identifying unusual patterns
- **Network Analysis**: Graph-based visualizations of feature relationships
- **Temporal Pattern Analysis**: Sophisticated analysis of time-based fraud patterns
- **Text Analytics**: 
  - Advanced NLP techniques for transaction description analysis
  - BERT-based linguistic modeling for fraud detection
- **Geographic Analysis**: Spatial distribution of fraud patterns
- **Merchant Risk Profiling**: Analysis of merchant-specific fraud patterns
- **Overfitting Prevention**: Multiple techniques to ensure model generalization
- **Fairness Evaluation**: Assessment of model fairness across different transaction groups

## Next Steps
- Deploy the model as an API for real-time fraud detection
- Implement a monitoring system to track model performance
- Set up alerts for high-confidence fraud predictions
- Collect feedback from fraud investigators to improve the model
- Retrain the model periodically with new data
- Implement anomaly detection for transaction sequences
- Integrate with streaming data sources for real-time fraud detection
- Expand fairness evaluations to additional demographic groups
- Implement active learning techniques to improve model performance with minimal labeling 