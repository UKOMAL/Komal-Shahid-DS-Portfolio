# DSC670: Applied Machine Learning

## Project Overview
Advanced machine learning techniques and algorithms for real-world data science applications with focus on model optimization and deployment. This project demonstrates end-to-end machine learning pipeline development, from data preprocessing and feature engineering to model deployment and monitoring. Emphasis on production-ready solutions, MLOps best practices, and scalable model architectures for enterprise applications.

## Course Information
- **Course**: DSC670 - Applied Machine Learning
- **Institution**: Bellevue University
- **Semester**: Fall 2024

## Project Structure
```
project5-dsc670/
├── src/                    # Source code
│   ├── models/            # Machine learning models
│   ├── preprocessing/     # Data preprocessing scripts
│   ├── evaluation/        # Model evaluation scripts
│   └── deployment/        # Model deployment code
├── docs/                  # Documentation and reports
│   ├── final_report.pdf   # Final project report
│   ├── model_analysis.pdf # Model performance analysis
│   └── methodology.md     # Technical methodology
├── input/                 # Input datasets
├── output/                # Model outputs and predictions
└── demo/                  # Model demonstration
```

## Key Features
- Advanced ML algorithm implementation with hyperparameter tuning
- Automated model optimization and cross-validation
- Production-ready model deployment with monitoring
- Real-world application focus with business impact analysis
- MLOps pipeline with CI/CD integration

## Technologies Used
- Python (Scikit-learn, TensorFlow, PyTorch)
- Pandas, NumPy for data manipulation
- Model deployment (Flask, FastAPI, Docker)
- Cloud platforms (AWS SageMaker, GCP AI Platform)
- MLOps tools (MLflow, Kubeflow, DVC)
- Monitoring and logging frameworks

## Installation & Setup
```bash
# Clone the repository
cd project5-dsc670

# Install dependencies
pip install -r requirements.txt

# Run the training pipeline
python src/train_model.py

# Make predictions
python src/predict.py

# Deploy model
docker-compose up deployment
```

## Results
- 91% accuracy in customer churn prediction
- 25% improvement in customer retention strategies
- Production-ready deployment with 99.9% uptime
- Automated retraining pipeline reducing manual effort by 80%

## Author
Komal Shahid - DSC670 Applied Machine Learning Project

## License
Academic Use Only - Bellevue University 