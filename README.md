# Komal Shahid | AI & Machine Learning Engineer

<div align="center">

**M.S. Data Science | Bellevue University | GPA 4.0 | Class of 2025-2026**

[![Portfolio](https://img.shields.io/badge/🌐_Portfolio-Live_Demo-2E86AB?style=for-the-badge&logo=github-pages&logoColor=white)](https://ukomal.github.io/Komal-Shahid-DS-Portfolio/)
[![LinkedIn](https://img.shields.io/badge/💼_LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/komal-shahid-6b1704175)
[![Email](https://img.shields.io/badge/📧_Email-Contact-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:kshahid@my.bellevue.edu)

</div>

---

## About

I'm a data scientist obsessed with building intelligent systems that solve real human problems. Over the past two years at Bellevue University, I've moved systematically from foundational skills in Python and statistics through to production machine learning and ethical AI. My work focuses on three things: **impact** (projects that matter to people), **interpretability** (models you can understand and trust), and **rigor** (honest metrics, no inflated claims).

This portfolio showcases the breadth of the MSDS program—10 courses, each building toward a capstone—plus three major applied projects in fraud detection, mental health, and AI-powered tools.

---

## Featured Projects

### 1. Real-World Fraud Detection System
**DSC680 Capstone | Production ML Pipeline | $2.3M Annual Impact**

Engineered a sophisticated fraud detection system processing 800K+ financial transactions from real-world datasets (IEEE-CIS + Credit Card). The system combines **LightGBM gradient boosting** with a **TensorFlow autoencoder** for anomaly detection, achieving **88.6% AUC-ROC** in production-grade evaluation.

**Why this matters:** Fraud costs financial institutions billions annually. Most systems either reject too many legitimate transactions or miss sophisticated fraud. This project balances precision and recall using **business-cost aware thresholds** rather than arbitrary probability cutoffs.

**Technical highlights:**
- Advanced ensemble: LightGBM + Autoencoder + domain-engineered features
- Fairness auditing: SHAP values for explainability across demographic groups
- Production-ready: Docker containerization, MLflow tracking, model versioning
- Rigorous evaluation: 5-fold cross-validation on hold-out test sets, conservative metrics

**Key results:**
- 88.6% AUC-ROC on IEEE-CIS dataset (800K transactions)
- $2.3M estimated annual fraud prevention at 85% detection rate
- <100ms prediction latency for real-time scoring
- 60% reduction in manual review workload

**Tech Stack:** Python, LightGBM, TensorFlow, Pandas, scikit-learn, SHAP, MLflow, Docker

---

### 2. Mental Health Treatment-Seeking Prediction
**DSC680 Project 1 | Logistic Regression with Fairness Audit | 72.3% AUC**

Analyzed the Open Source Mental Illness (OSMI) 2016 survey (N=1,259 responses) to predict which individuals would seek mental health treatment. The challenge: understanding the human and organizational levers that drive treatment-seeking behavior in tech companies.

**Why this matters:** Mental health stigma and access barriers prevent people from getting help. HR teams want evidence-based guidance on what actually increases treatment rates. This project provides that evidence.

**Methodology:**
- 4-model comparison: Logistic Regression, Random Forest, XGBoost, Neural Network
- Winner: **Logistic Regression (AUC=0.723)** for interpretability over marginal performance gains
- SHAP analysis: Identified top drivers of treatment-seeking (interference with work, caring for family)
- Fairness audit: Group-level parity checks across age, gender, and company size
- SMOTE balancing: Addressed severe class imbalance in training data
- Bootstrapped confidence intervals: Quantified uncertainty in predictions

**Key insights:**
- Work interference is the strongest signal for treatment-seeking (drives behavior more than diagnosis alone)
- Younger workers and larger companies show different treatment-seeking patterns
- Model achieves reasonable AUC (0.723) with interpretable coefficients for HR communication

**Deliverables:**
- Jupyter notebook with full EDA and model pipeline
- SHAP explainability report for stakeholder communication
- Fairness audit documenting group-level performance
- Technical white paper with limitations and caveats

**Tech Stack:** Python, scikit-learn, XGBoost, SHAP, Pandas, Seaborn, Jupyter

---

### 3. Notely — AI-Powered Note-Taking Assistant
**DSC680 Creative Project | LLM Integration + Streamlit | Deployed App**

Built an intelligent note-taking system that combines **semantic search** with **AI summarization** and **smart templates**. Unlike linear note apps, Notely helps users capture, organize, and retrieve information using language rather than folders.

**Why this matters:** Everyone has note-taking tools. Few help you *find* information later. This project bridges that gap using modern NLP.

**Features:**
- Semantic search: Find notes by meaning, not just keywords
- AI summarization: Distill long notes into key takeaways
- Smart templates: Domain-specific templates for different note types
- Full-text search: Traditional search for fast recall
- Export to markdown: Keep your data portable

**Technical approach:**
- LLM API integration for summarization (OpenAI)
- Sentence embeddings for semantic search (Hugging Face)
- Streamlit UI for real-time interaction
- SQLite backend for lightweight persistence

**Live demo:** [notely.streamlit.app](https://notely.streamlit.app)

**Tech Stack:** Python, Streamlit, LLM APIs, Sentence Transformers, SQLite

---

## Complete MSDS Program Portfolio

Below is the full journey through 10 courses, each representing a distinct skill area in data science:

| # | Course | Project | Key Skills | Link |
|---|--------|---------|-----------|------|
| 1 | **DSC500** | Sales Analytics Dashboard | Excel, pivot tables, problem framing | [View →](projects/project12-dsc500/) |
| 2 | **DSC510** | Library Management System | Python scripting, functions, APIs | [View →](projects/project11-dsc510/) |
| 3 | **DSC520** | A/B Testing Framework | R, hypothesis testing, statistical inference | [View →](projects/project10-dsc520/) |
| 4 | **DSC530** | COVID-19 Exploratory Analysis | Python EDA, distributions, Jupyter | [View →](projects/project9-dsc530/) |
| 5 | **DSC540** | Healthcare Data Architecture | SQL, databases, ETL pipelines | [View →](projects/project8-dsc540/) |
| 6 | **DSC550** | E-commerce Association Rules | Data mining, clustering, pattern discovery | [View →](projects/project7-dsc550/) |
| 7 | **DSC630** | Financial Time Series Forecasting | ARIMA, SARIMA, exponential smoothing | [View →](projects/project6-dsc630/) |
| 8 | **DSC640** | Healthcare Dashboard | Tableau/Plotly, interactive visualization | [View →](projects/project4-dsc640/) |
| 9 | **DSC670** | Notely AI App | Deep learning, LLMs, Streamlit deployment | [View →](projects/project5-dsc670/) |
| 10 | **DSC680** | Capstone (Fraud + Mental Health) | Production ML, ethics, real-world impact | [View →](projects/project13-dsc680/) |

---

## Technical Skills

### Core Languages & Frameworks
**Python (Expert)** | **R (Proficient)** | **SQL (Advanced)** | **JavaScript (Intermediate)**

### Machine Learning & AI
- Supervised Learning: Logistic Regression, Decision Trees, Random Forest, XGBoost, LightGBM
- Unsupervised Learning: K-means, DBSCAN, hierarchical clustering, PCA
- Deep Learning: TensorFlow, Keras, autoencoders, neural networks
- NLP: BERT, DistilBERT, sentence embeddings, semantic search
- Explainability: SHAP, attention visualization, feature importance
- Ethical AI: fairness audits, bias detection, confidence calibration

### Data Engineering & Databases
- Pandas, NumPy for data manipulation
- SQL: complex queries, optimization, schema design
- ETL: data pipelines, feature engineering, data validation
- Databases: PostgreSQL, SQLite, NoSQL basics

### Visualization & Communication
- Tableau for interactive dashboards
- Plotly/Seaborn for exploratory visualization
- Jupyter notebooks for reproducible analysis
- Technical writing: white papers, README documentation

### MLOps & Deployment
- Docker containerization
- MLflow experiment tracking
- GitHub version control
- Streamlit for rapid prototyping
- AWS basics (EC2, S3)

---

## What Makes This Portfolio Different

### Human Impact First
Every project addresses a real problem: preventing fraud saves money and prevents crime, predicting mental health treatment-seeking helps HR teams support employees, and better note-taking saves knowledge workers time. These aren't toy datasets—they're authentic problems with measurable impact.

### Ethics-First ML
Machine learning systems have power, and power demands responsibility. I audit every model for fairness across demographic groups, quantify uncertainty with confidence intervals, and make models interpretable with SHAP. When in doubt, I choose simplicity and explainability over raw performance.

### Production-Grade Code
Academic projects often end at the notebook. Real ML work requires clean code, documentation, testing, versioning, and monitoring. My projects include Docker files, requirements.txt with pinned versions, code comments, and architectural diagrams. This is how code works in production.

### Honest Evaluation
The ML field has a problem: inflated metrics. A 99.9% accuracy sounds great until you realize it's a dataset with 99.5% negatives. I use **cross-validation**, report **conservative performance metrics**, compare against **baselines**, and document **limitations** openly. My fraud detection system achieved 88.6% AUC-ROC—excellent for production, but not magical.

---

## GPA & Recognition

- **M.S. Data Science, Bellevue University** | GPA: 4.0/4.0 | Expected graduation 2025-2026
- **Capstone Distinction**: Top-tier performance on applied projects with real-world datasets
- **Consistent Excellence**: 4.0 GPA maintained across 10 courses spanning foundational to advanced topics

---

## Let's Connect

I'm actively seeking roles as an **AI Engineer**, **ML Engineer**, or **Applied Research Scientist** where I can:
- Build production-ready AI systems that drive measurable business impact
- Lead ethical AI initiatives and champion responsible ML development
- Scale machine learning solutions to serve real users
- Contribute to cutting-edge research in applied machine learning

<div align="center">

**Ready to discuss how my production AI expertise can drive your next breakthrough?**

[![Schedule Call](https://img.shields.io/badge/📞_Schedule_Call-30min_Chat-28a745?style=for-the-badge)](mailto:kshahid@my.bellevue.edu?subject=AI%20Engineering%20Opportunity)
[![Download Resume](https://img.shields.io/badge/📄_Download-Resume_PDF-dc3545?style=for-the-badge)](https://ukomal.github.io/Komal-Shahid-DS-Portfolio/resume.pdf)

**Response time: Within 24 hours**

---

**Designed & built with care | Last updated April 2026**

</div>
