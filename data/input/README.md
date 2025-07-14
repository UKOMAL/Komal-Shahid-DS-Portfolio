# Banking Fraud Datasets with Transaction Descriptions

This directory contains banking fraud datasets used for FinBERT analysis and multi-modal fraud detection.

## Datasets Overview

The project uses two banking fraud datasets with transaction descriptions:

### 1. Bank Transaction Fraud Detection Dataset
- **Source**: Kaggle Dataset by Maru Sagar
- **URL**: https://www.kaggle.com/datasets/marusagar/bank-transaction-fraud-detection
- **Size**: ~68MB (25.5MB compressed)
- **Features**: 
  - Customer information (ID, Name, Age, Gender, State, City)
  - Transaction details (Amount, Type, Date, Time, Location)
  - Merchant information (ID, Category)
  - Device and authentication details
  - **Transaction_Description**: Rich text descriptions of transactions
  - **Is_Fraud**: Target variable (1 for fraud, 0 for legitimate)

### 2. Banking Fraud Dataset (Enhanced)
- **Source**: Original banking dataset with created descriptions
- **Features**:
  - Transaction details (Amount, Type, Location, Time)
  - Customer information (ID, Account Age, Credit Score)
  - Device and location information
  - **Transaction_Description**: Created from transaction features
  - **Is_Fraud**: Target variable (1 for fraud, 0 for legitimate)

## Transaction Description Examples

### Dataset 1 (Real Descriptions):
- "Bitcoin transaction"
- "Grocery delivery"
- "Mutual fund investment"
- "Food delivery"

### Dataset 2 (Created Descriptions):
- "Cash deposit of $8527.58 at Hughesmouth, Mongolia via Mobile"
- "Online payment of $2202.49 for services at Patriciashire, Iceland via ATM"
- "Bank transfer of $9352.32 to account at Port Timothymouth, Palau via ATM"

## Usage for FinBERT Analysis

These datasets are specifically designed for:
1. **Text-based fraud detection** using FinBERT
2. **Multi-modal analysis** combining numerical and textual features
3. **Linguistic pattern analysis** in banking transactions
4. **Real-world banking fraud scenarios** (not e-commerce)

## Key Advantages

- **Domain-specific**: Banking transactions, not e-commerce
- **Rich descriptions**: Real transaction descriptions for text analysis
- **Balanced features**: Both numerical and textual data
- **Fraud labels**: Properly labeled for supervised learning
- **Realistic scenarios**: Based on actual banking fraud patterns

Note: These datasets are used for educational and research purposes only. 