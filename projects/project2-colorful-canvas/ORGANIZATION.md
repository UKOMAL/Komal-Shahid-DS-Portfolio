# Colorful Canvas Project Organization

## Project Structure

The project follows a well-organized structure:

```
/projects/project3/
├── run.py                      # Main entry point for the application
├── README.md                   # Project documentation
├── requirements.txt            # Package dependencies
├── setup.py                    # Package installation
├── src/                        # Source code
│   ├── data/                   # Data loading modules
│   │   ├── data_loader.py      # Main data loader
│   │   ├── get_data.py         # Data fetching utilities
│   │   └── github_data_loader.py # GitHub API data fetcher
│   ├── models/                 # Model definitions
│   │   └── train_models.py     # Model training code
│   ├── scripts/                # Utility scripts
│   │   ├── collect_and_train.sh # Data collection and training script
│   │   ├── validate_model.py   # Model validation
│   │   ├── download_hf_models.py # HuggingFace model downloader
│   │   └── download_models.py  # Model downloader
│   ├── utils/                  # Utility functions
│   ├── colorful_canvas_ai.py   # Core functionality
│   └── run_pipeline.py         # Pipeline for the entire process
├── models/                    # Trained model files
├── datasets/                  # Dataset storage
├── output/                    # Generated outputs
│   ├── analysis/              # Analysis results
│   └── anamorphic/            # Generated 3D illusions
├── examples/                  # Example images
├── docs/                      # Documentation
│   ├── milestone1/            # Project milestone 1
│   ├── milestone2/            # Project milestone 2
│   └── milestone3/            # Project milestone 3
```

## Key Components

### Main Entry Point

- `run.py`: The main entry point for the application. Provides a unified interface to access all functionality.

### Source Code

- `src/data/`: Data loading and management
  - `data_loader.py`: Core data loading functionality
  - `github_data_loader.py`: GitHub API integration for fetching datasets
  - `get_data.py`: Data collection utilities
  
- `src/models/`: Model definitions and training
  - `train_models.py`: Model training code for all model types
  
- `src/scripts/`: Utility scripts for various tasks
  - `collect_and_train.sh`: Shell script for data collection and model training
  - `validate_model.py`: Model validation script
  - `download_hf_models.py`: Script for downloading HuggingFace models
  - `download_models.py`: General model download script
  
- `src/colorful_canvas_ai.py`: Core functionality for creating various 3D visual effects
- `src/run_pipeline.py`: End-to-end pipeline for the entire process

### Data and Models

- `datasets/`: Storage for downloaded and processed datasets
- `models/`: Storage for trained models
- `output/`: Generated output files and visualizations
- `examples/`: Example images for demonstration

## Usage

1. Run the main entry point:
   ```
   python run.py --mode pipeline
   ```

2. Generate a specific effect:
   ```
   python run.py --mode shadow_box --input examples/landscape.jpg
   ```

3. Generate a 3D anamorphic illusion:
   ```
   python run.py --mode anamorphic --input examples/original.jpg
   ```

4. Train models with custom parameters:
   ```
   python run.py --mode pipeline --train --epochs 10 --max_samples 200
   ``` 