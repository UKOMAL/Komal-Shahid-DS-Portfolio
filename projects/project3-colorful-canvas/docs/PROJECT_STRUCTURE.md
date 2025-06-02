# Standard Project Structure

This document outlines the standardized project structure for all DSC680 projects. Following this structure ensures consistency across projects and makes it easier for others to understand and run your code.

## Directory Structure

```
project-name/
├── src/                       # Source code
│   ├── data/                  # Data loading and processing
│   ├── models/                # ML model definitions
│   ├── utils/                 # Utility functions
│   ├── main.py                # Main entry point
│   └── [project_name].py      # Core project implementation
│
├── models/                    # Trained model files
│
├── data/                      # Data directory
│   ├── raw/                   # Raw, immutable data
│   ├── processed/             # Processed data
│   ├── interim/               # Intermediate data
│   └── samples/               # Sample data for demo
│
├── output/                    # Output files
│
├── demo/                      # Demo applications
│   ├── web/                   # Web-based demo
│   │   ├── index.html         # Main HTML file
│   │   ├── styles.css         # CSS styles
│   │   └── script.js          # JavaScript code
│   └── demo.py                # Command-line demo
│
├── docs/                      # Documentation
│
├── tests/                     # Unit tests
│
├── README.md                  # Project README
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup script
└── run_demo.sh                # Demo launcher script
```

## Key Components

### 1. Source Code (`src/`)

- **Main module**: Core functionality in `[project_name].py`
- **Data processing**: Scripts for acquiring and processing data
- **Models**: Model architectures and training code
- **Utilities**: Helper functions and shared code

### 2. Demo Applications (`demo/`)

Every project should include at least two demo options:

- **Web Demo**: Interactive browser-based demo
- **Command Line Demo**: Text-based interactive demo

### 3. Documentation (`docs/`)

- Project structure
- API documentation
- Usage examples
- Technical background

### 4. Data Management

- Clear separation between raw, processed, and sample data
- Sample data available for demos

### 5. Project Execution

- Simple demo launcher script
- Clear README with installation and usage instructions

## Git Integration

When uploading to Git:
- Include sample data for demos
- Use `.gitignore` to exclude large datasets and model files
- Document how to acquire necessary data and models

## Demo Requirements

All demos should:
- Be self-contained and easy to run
- Include clear instructions
- Work with sample data without requiring additional downloads
- Showcase the core functionality of the project 