# Real-World Fraud Detection System

A comprehensive fraud detection system using authentic, real-world datasets with LightGBM optimization, Neural Networks, and SHAP analysis to identify fraudulent transactions in banking and credit card datasets.

## 🎯 **Ethical Data Science Approach**

This project uses **only real-world, authentic datasets** for fraud detection research:
- **ULB Credit Card Fraud Dataset** (Université Libre de Bruxelles)
- **IEEE-CIS Fraud Detection Dataset** (Real competition data)

**No synthetic or artificially generated data** - ensuring realistic, ethical fraud detection research.

## Architecture Diagram

```mermaid
graph TD
    subgraph "Data Sources"
        A1["Banking Fraud Dataset"] --> B
        A2["Credit Card Fraud Dataset"] --> B
    end
    
    B["Data Loading with DASK<br/>Distributed Computing"] --> C["Data Preprocessing"]
    C --> D["Feature Engineering"]
    
    subgraph "Model Training"
        D --> E1["SMOTE Class<br/>Balancing"]
        E1 --> F1["LightGBM with<br/>Optuna Optimization"]
        E1 --> F2["Neural Network<br/>with TensorFlow"]
    end
    
    subgraph "Evaluation & Analysis"
        F1 --> G1["Performance Metrics<br/>AUC, F1, Precision, Recall"]
        F2 --> G1
        F1 --> G2["SHAP Analysis<br/>Feature Importance"]
    end
    
    G1 --> H["Visualization<br/>& Reporting"]
    G2 --> H
```

## Key Features

- **Real-World Data** - Uses authentic fraud datasets (no synthetic data)
- **Memory Optimization** - Uses DASK for distributed computing
- **Advanced Modeling** - LightGBM with Optuna hyperparameter optimization
- **Class Balancing** - SMOTE implementation to handle extreme imbalance
- **Comprehensive Visualization** - Multiple visualization techniques for fraud patterns
- **Model Interpretability** - SHAP analysis for feature importance
- **Ethical Research** - Realistic fraud rates and patterns

## 📊 **Datasets**

### ULB Credit Card Fraud Dataset
- **Source:** Université Libre de Bruxelles
- **Size:** 284,807 transactions
- **Fraud Rate:** 0.17% (realistic)
- **Features:** 31 (including PCA-transformed V1-V28)

### IEEE-CIS Fraud Detection Dataset
- **Source:** IEEE-CIS Competition
- **Size:** Large-scale real competition data
- **Fraud Rate:** 1.50% (realistic)
- **Features:** 394 transaction + 41 identity features

## Usage

The main Jupyter notebook `notebooks/fraud_detection_final.ipynb` contains the complete analysis pipeline.

## 🏗️ **Project Structure**

```
project13-dsc680/
├── data/input/
│   ├── creditcard-fraud/          # ULB Credit Card Dataset
│   └── ieee-cis/                  # IEEE-CIS Competition Dataset
├── src/
│   ├── final_fraud_detection.py   # Main fraud detection system
│   ├── core/                      # Core components
│   └── utils/                     # Optimization utilities
├── notebooks/
│   ├── fraud_detection_final.ipynb # Complete analysis
│   └── enhanced_feature_engineering.py
└── docs/                          # Documentation
```
