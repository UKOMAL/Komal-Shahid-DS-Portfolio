#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Depression Detection System - Main Script

This script provides a demo of the depression detection system,
showing how to use the different components together.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.depression_detection import DepressionDetectionSystem

def demo_single_text():
    """Demo the system on a single text input"""
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

def demo_batch_analysis():
    """Demo the system on a batch of texts from a sample file"""
    print("\n=== Batch Analysis Demo ===")
    
    # Initialize the depression detection system
    system = DepressionDetectionSystem(model_type="transformer")
    
    # Path to sample data
    sample_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              "src", "data", "sample", "sample_depression_data.csv")
    
    if not os.path.exists(sample_path):
        print(f"Sample data not found at {sample_path}")
        return
    
    # Analyze the batch of texts
    print(f"Analyzing texts from: {sample_path}")
    results = system.batch_analyze(
        sample_path,
        text_column="text",
        output_file=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                "output", "sample_results.csv")
    )
    
    # Display summary
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
        
        # Save the plot
        output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                  "output", "severity_distribution.png")
        plt.savefig(output_path)
        print(f"\nSeverity distribution plot saved to: {output_path}")

def print_usage():
    """Print usage information"""
    print("\nDepression Detection System - Usage")
    print("-----------------------------------")
    print("1. Single text analysis mode:")
    print("   python main.py --mode single")
    print("\n2. Batch analysis mode:")
    print("   python main.py --mode batch")
    print("\n3. Interactive mode:")
    print("   python main.py --mode interactive")
    print("\nFor more options:")
    print("   python main.py --help")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Depression Detection System Demo")
    parser.add_argument("--mode", choices=["single", "batch", "interactive"], 
                        default="single", help="Demo mode")
    
    args = parser.parse_args()
    
    # Print banner
    print("\n" + "="*60)
    print("Depression Detection System Demo")
    print("="*60)
    print("DISCLAIMER: This system is for demonstration purposes only.")
    print("It is not a diagnostic tool and should not be used for clinical decisions.")
    print("="*60 + "\n")
    
    if args.mode == "single":
        demo_single_text()
    elif args.mode == "batch":
        demo_batch_analysis()
    elif args.mode == "interactive":
        system = DepressionDetectionSystem(model_type="transformer")
        system.interactive_mode()
    
    print_usage()

if __name__ == "__main__":
    main() 