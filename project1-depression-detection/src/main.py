#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Depression Detection System - Main Script
Version: 1.0.0 (Final Release)
Date: April 6, 2025
Authors: Komal Shahid
License: MIT

This script provides a comprehensive demo of the depression detection system,
showcasing its key functionalities including single text analysis, batch processing,
and visualization capabilities. It serves as both a demonstration and a starting
point for integrating the system into other applications.

Key components:
- Single text analysis demo showing prediction capabilities
- Batch analysis of multiple texts with result visualization
- Interactive mode for real-time testing
- System integration examples

IMPORTANT: This system is intended for research and screening purposes only,
not for clinical diagnosis. All predictions should be reviewed by qualified
mental health professionals.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.depression_detection import DepressionDetectionSystem

# Default paths for better organization
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Ensure output directory exists
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

def demo_single_text():
    """
    Demonstrate depression detection on a single text input
    
    This function shows how to use the system for analyzing a single piece of text,
    displaying the predicted depression severity and confidence scores.
    
    Returns:
        dict: The prediction result
    """
    print("\n=== Single Text Analysis Demo ===")
    
    # Initialize the depression detection system
    system = DepressionDetectionSystem(model_type="transformer")
    
    # Example text for analysis
    text = "I haven't been feeling like myself lately. It's hard to get out of bed in the morning."
    
    # Analyze the text
    print(f"Analyzing text: \"{text}\"")
    result = system.predict(text)
    
    # Display results
    print("\nResult:")
    print(f"Depression Severity: {result['depression_severity']}")
    print("\nConfidence Scores:")
    for label, score in result['confidence_scores'].items():
        print(f"  {label}: {score:.2f}")
    
    if 'guidance' in result:
        print(f"\nGuidance: {result['guidance']}")
    
    return result

def demo_batch_analysis():
    """
    Demonstrate depression detection on a batch of texts
    
    This function shows how to use the system for analyzing multiple texts
    from a CSV file, generating summary statistics, and creating visualizations.
    
    Returns:
        DataFrame: The batch analysis results
    """
    print("\n=== Batch Analysis Demo ===")
    
    # Initialize the depression detection system
    system = DepressionDetectionSystem(model_type="transformer")
    
    # Sample data path
    sample_path = os.path.join(DEFAULT_DATA_DIR, "sample", "sample_texts.csv")
    
    # Check if sample data exists, create it if not
    if not os.path.exists(sample_path):
        os.makedirs(os.path.dirname(sample_path), exist_ok=True)
        
        # Create sample data with varying depression indicators
        sample_data = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'text': [
                "I feel great today and am looking forward to the weekend!",
                "I've been feeling a bit down lately, but it's not too bad.",
                "Nothing brings me joy anymore. I can't remember when I last felt happy.",
                "I'm having trouble sleeping and can't concentrate on anything."
            ]
        })
        
        sample_data.to_csv(sample_path, index=False)
        print(f"Created sample data at {sample_path}")
    
    # Analyze the batch of texts
    print(f"Analyzing texts from: {sample_path}")
    output_path = os.path.join(DEFAULT_OUTPUT_DIR, "batch_results.csv")
    results = system.batch_analyze(
        sample_path,
        text_column="text",
        output_file=output_path
    )
    
    # Display summary statistics
    if results is not None:
        print("\nAnalysis Summary:")
        severity_counts = results['depression_severity'].value_counts()
        print(severity_counts)
        
        # Plot severity distribution
        plt.figure(figsize=(10, 6))
        sns.countplot(x='depression_severity', data=results, 
                      order=['minimum', 'mild', 'moderate', 'severe'],
                      palette='viridis')
        plt.title('Depression Severity Distribution')
        plt.xlabel('Severity')
        plt.ylabel('Count')
        plt.tight_layout()
        
        # Save the visualization
        plot_path = os.path.join(DEFAULT_OUTPUT_DIR, "severity_distribution.png")
        plt.savefig(plot_path)
        print(f"\nSeverity distribution plot saved to: {plot_path}")
        
        return results
    
    return None

def print_usage():
    """
    Print detailed usage information for the command-line interface
    
    This function displays the available options and example commands
    for using the depression detection system.
    """
    print("\nDepression Detection System - Usage")
    print("-----------------------------------")
    print("Usage: python main.py [options]")
    print("\nOptions:")
    print("  --mode MODE       Specify the mode: single, batch, or interactive")
    print("  --help            Show this help message and exit")
    print("\nExamples:")
    print("  python main.py --mode single       # Run the single text analysis demo")
    print("  python main.py --mode batch        # Run the batch analysis demo")
    print("  python main.py --mode interactive  # Start the interactive mode")
    print("\nFor more advanced options:")
    print("  Use the depression_detector.py script directly")

def main():
    """
    Main function that handles command-line arguments and runs the appropriate demo
    
    This function parses command-line arguments and runs the appropriate
    demonstration based on the specified mode.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Depression Detection System Demo")
    parser.add_argument("--mode", choices=["single", "batch", "interactive"], 
                        default="single", help="Demo mode")
    
    args = parser.parse_args()
    
    # Print banner with version and disclaimer
    print("\n" + "="*60)
    print("Depression Detection System Demo v1.0.0")
    print("="*60)
    print("DISCLAIMER: This system is for demonstration purposes only.")
    print("It is not a diagnostic tool and should not be used for clinical decisions.")
    print("="*60 + "\n")
    
    try:
        # Run the selected demo mode
        if args.mode == "single":
            demo_single_text()
        elif args.mode == "batch":
            demo_batch_analysis()
        elif args.mode == "interactive":
            system = DepressionDetectionSystem(model_type="transformer")
            system.interactive_mode()
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Display usage information
    print_usage()

if __name__ == "__main__":
    main() 