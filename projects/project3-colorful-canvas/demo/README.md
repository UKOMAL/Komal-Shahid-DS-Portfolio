# Colorful Canvas Demo

This directory contains demonstration applications for the Colorful Canvas AI Art Studio.

## Command Line Demo

The `demo.py` script provides an interactive command-line interface to showcase the capabilities of the Colorful Canvas AI.

### Usage

```bash
# Run the interactive demo
python demo.py --interactive

# Or simply
python demo.py
```

### Features

- Select from sample images in the `data/samples` directory
- Choose between different anamorphic illusion effects:
  - **Shadow Box Illusion**: Creates the illusion of objects floating in a glass display case
  - **Screen Pop Effect**: Makes objects appear to pop out of the screen surface
  - **Seoul Corner Projection**: Creates a special corner projection illusion
- Automatic depth map generation using AI
- Output images saved to the `output` directory

## Web Demo

The `web` directory contains an interactive web-based demo that can be run locally.

### Usage

```bash
# Start a simple web server
cd web
python -m http.server 8002

# Then open a browser and navigate to:
# http://localhost:8002
```

### Features

- User-friendly interface for uploading images and selecting effects
- Real-time preview of anamorphic illusions
- Viewing instructions for each effect
- Gallery of example results
- Educational content about how anamorphic illusions work

## Requirements

The demos require the main Colorful Canvas library to be installed and properly configured.
Make sure you've installed all dependencies from the main `requirements.txt` file:

```bash
pip install -r ../requirements.txt
``` 