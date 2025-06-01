#!/bin/bash

# Colorful Canvas: AI Art Studio - Data Collection and Training Script
# This script collects 3D anamorphic illusion data from web sources,
# incorporates existing data, and trains the generator model.

# Change to the project directory
cd "$(dirname "$0")"
echo "Working directory: $(pwd)"

# Find Python path
PYTHON_PATH=$(which python3)
if [ -z "$PYTHON_PATH" ]; then
    PYTHON_PATH=$(which python)
fi

if [ -z "$PYTHON_PATH" ]; then
    echo "Error: Python not found. Please make sure Python is installed."
    exit 1
fi

echo "Using Python at: $PYTHON_PATH"

# Create necessary directories
mkdir -p dataset/anamorphic/raw
mkdir -p dataset/anamorphic/processed
mkdir -p dataset/anamorphic/metadata
mkdir -p dataset/anamorphic/categories
mkdir -p models/anamorphic

# Install required dependencies if needed
if [ "$1" == "--install-deps" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Step 1: Collect data from web sources
echo -e "\n=== Step 1: Collecting data from web sources ==="
$PYTHON_PATH src/train_anamorphic.py --collect_data

# Step 2: Incorporate existing data (including wave videos and anamorphic outputs)
echo -e "\n=== Step 2: Incorporating existing data ==="
$PYTHON_PATH src/utils/incorporate_existing_data.py

# Step 3: Train the model
echo -e "\n=== Step 3: Training the model ==="
$PYTHON_PATH src/train_anamorphic.py --use_existing_data --epochs 200 --batch_size 8

echo -e "\n=== Process completed! ==="
echo "Check the models/anamorphic directory for the trained model." 