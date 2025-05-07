# Installation Guide: Privacy-Preserving Federated Learning for Healthcare

This document provides comprehensive instructions for setting up and running the federated healthcare AI project.

## System Requirements

- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended for larger models)
- CUDA-compatible GPU recommended for training on large healthcare datasets
- Operating Systems: Linux, macOS, or Windows

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/privacy-preserving-federated-healthcare.git
cd privacy-preserving-federated-healthcare
```

### 2. Create a Virtual Environment

#### On Linux/macOS:
```bash
python -m venv venv
source venv/bin/activate
```

#### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all necessary packages including:
- PyTorch for deep learning models
- Flower for federated learning
- NumPy, Pandas for data processing
- Matplotlib, Plotly for visualizations
- Opacus for differential privacy

## Getting Started

### 1. Download Sample Datasets

```bash
python src/data_downloader.py --dataset mimic --target-dir data/mimic
```

Available datasets:
- `mimic`: Clinical data (subset)
- `isic`: Skin lesion images
- `ecg`: Physiological time series

### 2. Generate Visualizations

```bash
python src/visualizations_runner.py
```

This will create all visualizations in the `output/visualizations` directory.

### 3. Run a Federated Learning Simulation

```bash
python src/federated_learning.py --simulate --clients 5 --rounds 10 --privacy differential --epsilon 1.0 --modality tabular
```

## Project Structure

```
project2-federated-healthcare-ai/
├── data/                  # Dataset storage
├── docs/                  # Documentation
│   ├── images/            # Images for documentation
│   ├── presentation/      # Presentation slides
│   └── white_paper.md     # Technical white paper
├── models/                # Saved model files
├── output/                # Generated outputs
│   └── visualizations/    # Visualization images
├── src/                   # Source code
├── config/                # Configuration files
├── requirements.txt       # Dependencies
├── README.md              # Project overview
└── INSTALLATION.md        # This file
```

## Troubleshooting

### CUDA Issues

If you encounter GPU-related errors:

```bash
# Check if PyTorch can access your GPU
python -c "import torch; print(torch.cuda.is_available())"
```

If this returns `False`, ensure CUDA is properly installed and compatible with your PyTorch version.

### Memory Errors

For out-of-memory errors, try reducing batch sizes in the configuration files:

```bash
# Edit config/default_config.yaml
# Change batch_size to a smaller value, e.g., 16
```

### Dataset Access

Some datasets require registration. If you encounter download issues:

```bash
# Use manual download mode
python src/data_downloader.py --dataset mimic --manual-download
```

This will provide instructions for manual download and placement of dataset files.

## Additional Resources

- [Project Documentation](./docs/README.md)
- [White Paper](./docs/white_paper.md)
- [API Reference](./docs/api_reference.md)

## Citation

If you use this code in your research, please cite:

```
@software{Shahid2025,
  author = {Shahid, Komal},
  title = {Privacy-Preserving Federated Learning for Healthcare},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/privacy-preserving-federated-healthcare}
}
``` 