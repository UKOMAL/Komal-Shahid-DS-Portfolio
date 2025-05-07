# Privacy-Preserving Federated Learning for Healthcare

A comprehensive implementation of privacy-preserving federated learning designed specifically for modern healthcare applications. This project provides a robust framework for training machine learning models across distributed healthcare institutions without sharing sensitive patient data, addressing one of the most significant challenges in healthcare AI: accessing sufficient high-quality data while maintaining patient privacy.

![Federated Healthcare Overview](./docs/images/federated_healthcare_overview.png)

## 📋 Table of Contents
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Downloading Datasets](#downloading-datasets)
- [Usage Examples](#usage-examples)
- [Privacy Mechanisms](#privacy-mechanisms)
- [Visualization Capabilities](#visualization-capabilities)
- [Performance Metrics](#performance-metrics)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)
- [Relevant Research](#relevant-research)

## 🔑 Key Features

- **State-of-the-Art Privacy Protection**: Implementation of differential privacy, homomorphic encryption, secure multi-party computation, and advanced aggregation protocols
- **Multi-Modal Healthcare Models**: Specialized architectures for medical imaging, clinical tabular data, genetic sequences, and temporal biomedical signals
- **Personalization Capabilities**: Client-specific model adaptation while maintaining global knowledge sharing
- **Cross-Silo & Cross-Device Support**: Solutions for both institutional (hospital-to-hospital) and edge device (wearable/mobile) scenarios
- **Communication Efficiency**: Techniques to reduce bandwidth requirements through model compression and efficient updates
- **Regulatory Compliance**: HIPAA and GDPR-aligned data handling with comprehensive audit trails
- **Publication-Quality Analytics**: Advanced visualization techniques for result analysis and presentation

## 📂 Project Structure

```
project2-federated-healthcare-ai/
├── data/                  # Data handling and preprocessing
├── docs/                  # Documentation and project milestones
│   ├── images/            # Documentation images
│   └── white_paper.md     # Technical white paper
├── models/                # Model implementations for different modalities
│   └── federated/         # Federated learning model architectures
├── src/                   # Source code
│   ├── client/            # Federated learning client implementation
│   ├── data/              # Dataset loaders and preprocessing 
│   ├── models/            # Neural network architectures
│   ├── privacy/           # Privacy mechanisms
│   ├── server/            # Federated learning server
│   ├── utils/             # Utility functions and metrics
│   ├── visualization/     # Visualization tools
│   │   ├── visualizations.py          # Unified visualization module
│   │   └── run_visualizations.py      # Script to generate all visualizations  
│   ├── data_downloader.py # Dataset downloader script
│   └── federated_learning.py          # Main script
├── output/                # Results and model checkpoints
│   └── visualizations/    # Generated visualization images and HTML files
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
```

## 💻 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/username/project2-federated-healthcare-ai.git
   cd project2-federated-healthcare-ai
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended for larger models)

### Configuration

Create a configuration file for your federated learning experiment:

```bash
cp config/default_config.yaml config/my_experiment.yaml
```

Edit `my_experiment.yaml` to set your desired parameters such as the number of clients, privacy budget, and model architecture.

## 📊 Downloading Datasets

The project supports automatic downloading of several healthcare datasets. Use the data downloader script:

```bash
python src/data_downloader.py --dataset mimic --target-dir data/mimic
```

Available datasets:
- `mimic`: MIMIC-III clinical database 
- `isic`: ISIC 2019 skin lesion images
- `ecg`: PTB ECG diagnostic database
- `chexpert`: CheXpert chest X-ray dataset
- `physionet2019`: PhysioNet 2019 Challenge dataset

To list all available datasets:

```bash
python src/data_downloader.py --dataset list
```

To download all supported datasets (this may take time):

```bash
python src/data_downloader.py --dataset all --target-dir data/
```

## 🖥️ Usage Examples

### Running a Federated Learning Simulation

The simulation mode allows you to test the framework with virtual clients:

```bash
python src/federated_learning.py --config config/my_experiment.yaml --simulate --clients 5 --rounds 50 --privacy differential --epsilon 0.5 --modality image
```

### Starting a Server

To start a federated learning server that clients can connect to:

```bash
python -m src.server.server --port 8080 --max-clients 10 --config config/server_config.yaml
```

### Starting a Client

To start a client that connects to the server:

```bash
python -m src.client.client --server-address localhost:8080 --data-path /path/to/local/data --privacy-mechanism local_dp
```

### Generating Visualizations

To generate comprehensive visualizations from the project results:

```bash
python -m src.visualization.run_visualizations --results-dir output/results --output-dir output/visualizations
```

## 🔒 Privacy Mechanisms

The project implements several state-of-the-art privacy-preserving techniques:

1. **Differential Privacy**
   - Local DP: Adding calibrated noise at the client side
   - Central DP: Adding noise during aggregation on the server side
   - Moments Accountant: Advanced privacy budget tracking

2. **Cryptographic Methods**
   - Homomorphic Encryption: Enables computation on encrypted model updates
   - Secure Aggregation: Cryptographic protocol for secure model averaging
   - Zero-Knowledge Proofs: Verification without revealing information

3. **Model Protection**
   - Gradient Clipping: Limiting the influence of individual samples
   - Model Pruning: Reducing model complexity to minimize information leakage
   - Split Learning: Distributing model layers between clients and server

4. **Anonymization Techniques**
   - Client Pseudonymization: Protect client identities
   - Model Distillation: Transferring knowledge without sharing the original data
   - Synthetic Data Generation: Privacy-preserving synthetic healthcare data

## 📈 Visualization Capabilities

The project includes comprehensive visualization tools for analyzing federated learning experiments:

### Basic Visualizations
- **Model Convergence**: Track loss and accuracy across training rounds
- **Privacy Analysis**: Evaluate privacy-utility tradeoff at different privacy levels
- **Institutional Contribution**: Compare metrics across participating healthcare institutions
- **Performance Metrics**: Radar chart of multiple performance metrics

### Advanced Visualizations
- **Metric Correlation Heatmap**: Analyze relationships between different performance metrics
- **Parallel Coordinates**: Compare institutions across multiple dimensions simultaneously
- **3D Privacy-Performance Tradeoff**: Visualize the relationship between privacy, accuracy, and communication cost
- **Geographical Contribution**: Map institutional performance metrics to their geographic locations
- **Federated Performance Comparison**: Compare centralized, federated, and privacy-protected approaches

All visualizations are designed to publication quality standards and can be generated in both static (PNG) and interactive (HTML) formats.

## 📉 Performance Metrics

The framework includes specialized metrics for healthcare applications:

- **Standard metrics**: Accuracy, Precision, Recall, F1-score
- **Healthcare-specific**: Sensitivity, Specificity, AUC-ROC, Balanced Accuracy
- **Explainability metrics**: Feature importance, SHAP values
- **Fairness metrics**: Demographic parity, Equal opportunity
- **Federated performance**: Communication efficiency, Convergence rate, Client drift
- **Privacy evaluation**: Privacy budget consumption, Membership inference attack resistance, Model inversion attack resistance

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- The framework builds upon the [Flower](https://flower.dev/) federated learning library
- Privacy mechanisms are inspired by [OpenDP](https://opendp.org/) and [Opacus](https://opacus.ai/) libraries
- Medical model architectures adapted from established healthcare AI research
- Visualization techniques leverage Matplotlib, Seaborn, and Plotly libraries
- Special thanks to the healthcare partners who provided guidance on clinical requirements

## 📚 Relevant Research

### Recent Papers (2023-2025)

1. Li, T., et al. (2024). "FedBioMed: Enabling Federated Learning for Medical Imaging across International Institutions." *Nature Machine Intelligence*, 6(3), 245-258.

2. Smith, A., et al. (2024). "Client Heterogeneity in Federated Learning: A Comprehensive Analysis of Impact on Healthcare Applications." *IEEE Transactions on Medical Imaging*, 43(5), 1312-1327.

3. Johnson, K., & Chen, W. (2023). "Federated Multi-Modal Learning for Comprehensive Patient Profile Generation." *Proceedings of the Conference on Health, Inference, and Learning (CHIL 2023)*, 205-217.

4. Zhang, Y., et al. (2023). "FedDx: Privacy-Preserving Diagnostic Models with Federated Learning in International Healthcare Systems." *Journal of Biomedical Informatics*, 128, 104078.

5. Saha, P., Akbari, H., Safaei, S., & Subramanian, L. (2024). "Examining Modality Incongruity in Multimodal Federated Learning for Medical Vision and Language-based Disease Detection." *Proceedings of the Conference on Health, Inference, and Learning (CHIL 2024)*, 128-142. 