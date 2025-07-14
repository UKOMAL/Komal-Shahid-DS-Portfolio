#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Downloader for Federated Healthcare AI

This script provides functionality to download healthcare datasets programmatically.
It supports various publicly available healthcare datasets and handles authentication where required.

Usage:
    python data_downloader.py --dataset mimic --target-dir data/mimic
"""

import os
import argparse
import logging
from pathlib import Path
from typing import List, Optional

# Import the dataset download function from data_loader
from src.data.data_loader import download_dataset, prepare_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def list_available_datasets() -> List[str]:
    """Return a list of available datasets for download."""
    return [
        'mimic',      # Clinical data
        'isic',       # Skin lesion images
        'ecg',        # Electrocardiogram signals
        'chexpert',   # Chest X-rays
        'physionet2019'  # Vital signs time series
    ]

def download_multiple_datasets(
    datasets: List[str],
    target_dir: str,
    force_download: bool = False
) -> None:
    """
    Download multiple datasets.
    
    Args:
        datasets: List of dataset names to download
        target_dir: Directory to save the downloaded datasets
        force_download: Whether to download even if files already exist
    """
    for dataset_name in datasets:
        try:
            download_path = download_dataset(dataset_name, target_dir, force_download)
            logger.info(f"Successfully downloaded {dataset_name} to {download_path}")
            
            # Try to prepare the dataset to ensure it's usable
            try:
                prepare_dataset(dataset_name, download_path)
                logger.info(f"Successfully prepared {dataset_name} dataset")
            except (FileNotFoundError, ValueError) as e:
                logger.warning(f"Dataset preparation error: {str(e)}")
                
        except Exception as e:
            logger.error(f"Failed to download {dataset_name}: {str(e)}")

def main():
    """Main function for the data downloader."""
    parser = argparse.ArgumentParser(
        description="Download healthcare datasets for federated learning"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset to download (use 'all' for all available datasets, 'list' to see options)"
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default="./data",
        help="Directory to save the downloaded dataset"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download even if files already exist"
    )
    
    args = parser.parse_args()
    
    if args.dataset == "list":
        print("Available datasets:")
        for dataset in list_available_datasets():
            print(f" - {dataset}")
        return
    
    if args.dataset == "all":
        download_multiple_datasets(
            list_available_datasets(),
            args.target_dir,
            args.force
        )
    elif args.dataset:
        try:
            download_path = download_dataset(args.dataset, args.target_dir, args.force)
            logger.info(f"Successfully downloaded {args.dataset} to {download_path}")
        except ValueError as e:
            logger.error(str(e))
            print(f"Error: {str(e)}")
            print("Use --dataset list to see available options")
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 