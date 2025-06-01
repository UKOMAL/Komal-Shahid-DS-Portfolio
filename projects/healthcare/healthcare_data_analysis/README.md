# Healthcare Data Analysis Project

## Overview
This project analyzes two important healthcare datasets: the Pima Indians Diabetes Dataset and the Cleveland Heart Disease Dataset. Using advanced machine learning techniques, the project builds predictive models for disease classification and extracts valuable insights from the data.

## Project Structure
- `healthcare_dataset_analysis_report.md` - Comprehensive white paper detailing the analysis and findings
- `/code` - Python scripts for data analysis and modeling
- `/data` - Processed healthcare datasets
- `/visualizations` - Generated data visualizations and model performance metrics

## Main Files
- `code/analyze_healthcare_data.py` - Main analysis script with all data processing, visualization, and modeling code
- `healthcare_dataset_analysis_report.md` - Complete technical white paper with clinical context and findings

## Key Features
- Advanced ensemble modeling with Random Forest and Gradient Boosting
- Comprehensive data preprocessing and missing value handling
- Feature importance analysis with clinical interpretations
- ROC curves and performance metrics visualization
- Detailed correlation analysis

## Running the Analysis
To reproduce the analysis:

1. Ensure the required Python packages are installed:
```
pip install pandas numpy matplotlib seaborn scikit-learn
```

2. Run the analysis script:
```
python code/analyze_healthcare_data.py
```

This will process the datasets in `/data`, generate visualizations in `/visualizations`, and output model performance metrics.

## Results
The analysis achieved:
- 75.3% accuracy in diabetes prediction using ensemble methods
- 83.3% accuracy in heart disease prediction using optimized Random Forest
- Identification of key clinical predictors for both conditions
- Comprehensive visualization of disease patterns and risk factors

## Datasets
- **Diabetes Dataset**: 768 records of Pima Indian heritage women with diabetes diagnosis
- **Heart Disease Dataset**: 303 records from Cleveland Clinic with cardiac health metrics

## Author
Komal Shahid  
DSC680 - Applied Data Science  
Bellevue University 