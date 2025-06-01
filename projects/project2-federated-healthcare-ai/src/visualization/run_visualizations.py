#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script runs the visualization pipeline for the federated healthcare AI project.
It generates both basic and advanced visualizations, along with explanations.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

# Import from local modules
from visualization.visualization_basic import (  # type: ignore
    plot_model_convergence,
    plot_privacy_analysis,
    plot_institutional_contribution,
    plot_performance_metrics
)
from visualization.visualization_advanced import (  # type: ignore
    plot_correlation_heatmap,
    plot_3d_privacy_performance,
    plot_geographical_contribution, 
    plot_parallel_coordinates,
    plot_federated_performance_comparison
)
from visualization.visualization_explanations import VisualizationExplainer  # type: ignore

# Define output directories
OUTPUT_DIR = Path("../../output")
BASIC_VIZ_DIR = OUTPUT_DIR / "visualizations" / "basic"
ADVANCED_VIZ_DIR = OUTPUT_DIR / "visualizations" / "advanced"
EXPLANATION_DIR = OUTPUT_DIR / "visualization_explanations"

def create_output_directories() -> None:
    """Create the output directories if they don't exist."""
    os.makedirs(BASIC_VIZ_DIR, exist_ok=True)
    os.makedirs(ADVANCED_VIZ_DIR, exist_ok=True)
    os.makedirs(EXPLANATION_DIR, exist_ok=True)
    
def load_data() -> Dict[str, Any]:
    """
    Load data for visualizations.
    
    Returns:
        Dict containing all necessary data for visualizations
    """
    # This is a placeholder - in a real implementation, this would load actual data
    print("Loading data for visualizations...")
    data = {
        "convergence_data": {
            "rounds": list(range(1, 51)),
            "loss": [1 - 0.02*i + 0.001*np.random.randn() for i in range(50)],
            "accuracy": [0.5 + 0.01*i + 0.001*np.random.randn() for i in range(50)]
        },
        "privacy_data": {
            "epsilon": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, np.inf],
            "accuracy": [0.65, 0.78, 0.83, 0.87, 0.91, 0.93, 0.95],
            "f1_score": [0.60, 0.74, 0.79, 0.84, 0.89, 0.92, 0.94]
        },
        "institution_data": {
            "names": ["Hospital A", "Hospital B", "Hospital C", "Clinic D", "Research E"],
            "samples": [1200, 950, 1500, 430, 780],
            "accuracy": [0.88, 0.91, 0.85, 0.82, 0.90],
            "precision": [0.85, 0.92, 0.83, 0.80, 0.89],
            "recall": [0.90, 0.88, 0.87, 0.85, 0.91],
            "f1_score": [0.87, 0.90, 0.85, 0.82, 0.90],
            "auc": [0.92, 0.94, 0.89, 0.87, 0.93]
        }
    }
    return data

def generate_basic_visualizations(data: Dict[str, Any]) -> List[str]:
    """
    Generate basic visualizations.
    
    Args:
        data: Dictionary containing the data for visualizations
        
    Returns:
        List of paths to the generated visualizations
    """
    print("Generating basic visualizations...")
    visualization_paths = []
    
    # Plot model convergence
    conv_path = BASIC_VIZ_DIR / "model_convergence.png"
    plot_model_convergence(
        data["convergence_data"]["rounds"],
        data["convergence_data"]["loss"],
        data["convergence_data"]["accuracy"],
        str(conv_path)
    )
    visualization_paths.append(str(conv_path))
    
    # Plot privacy analysis
    privacy_path = BASIC_VIZ_DIR / "privacy_analysis.png"
    plot_privacy_analysis(
        data["privacy_data"]["epsilon"],
        data["privacy_data"]["accuracy"],
        data["privacy_data"]["f1_score"],
        str(privacy_path)
    )
    visualization_paths.append(str(privacy_path))
    
    # Plot institutional contribution
    inst_path = BASIC_VIZ_DIR / "institutional_contribution.png"
    plot_institutional_contribution(
        data["institution_data"]["names"],
        data["institution_data"]["accuracy"],
        data["institution_data"]["precision"],
        data["institution_data"]["recall"],
        data["institution_data"]["f1_score"],
        str(inst_path)
    )
    visualization_paths.append(str(inst_path))
    
    # Plot performance metrics
    perf_path = BASIC_VIZ_DIR / "performance_metrics.png"
    plot_performance_metrics(
        data["institution_data"]["names"],
        data["institution_data"]["accuracy"],
        data["institution_data"]["precision"], 
        data["institution_data"]["recall"],
        data["institution_data"]["f1_score"],
        data["institution_data"]["auc"],
        str(perf_path)
    )
    visualization_paths.append(str(perf_path))
    
    return visualization_paths

def generate_advanced_visualizations(data: Dict[str, Any]) -> List[str]:
    """
    Generate advanced visualizations.
    
    Args:
        data: Dictionary containing the data for visualizations
        
    Returns:
        List of paths to the generated visualizations
    """
    print("Generating advanced visualizations...")
    visualization_paths = []
    
    # Correlation heatmap
    metrics = {
        "accuracy": data["institution_data"]["accuracy"],
        "precision": data["institution_data"]["precision"],
        "recall": data["institution_data"]["recall"],
        "f1_score": data["institution_data"]["f1_score"],
        "auc": data["institution_data"]["auc"]
    }
    corr_path = ADVANCED_VIZ_DIR / "correlation_heatmap.png"
    plot_correlation_heatmap(metrics, str(corr_path))
    visualization_paths.append(str(corr_path))
    
    # Generate other advanced visualizations
    # These are placeholders - in a real implementation, these would create actual visualizations
    
    # 3D privacy-performance visualization
    privacy_3d_path = ADVANCED_VIZ_DIR / "3d_privacy_performance.png"
    plot_3d_privacy_performance(data, str(privacy_3d_path))
    visualization_paths.append(str(privacy_3d_path))
    
    # Geographical contribution 
    geo_path = ADVANCED_VIZ_DIR / "geographical_contribution.png"
    plot_geographical_contribution(data, str(geo_path))
    visualization_paths.append(str(geo_path))
    
    # Parallel coordinates
    parallel_path = ADVANCED_VIZ_DIR / "parallel_coordinates.png"
    plot_parallel_coordinates(data, str(parallel_path))
    visualization_paths.append(str(parallel_path))
    
    # Federated performance comparison
    fed_comp_path = ADVANCED_VIZ_DIR / "federated_performance_comparison.png"
    plot_federated_performance_comparison(data, str(fed_comp_path))
    visualization_paths.append(str(fed_comp_path))
    
    return visualization_paths

def generate_visualization_explanations() -> None:
    """Generate explanations for all visualizations."""
    print("Generating visualization explanations...")
    explainer = VisualizationExplainer(str(EXPLANATION_DIR))
    explainer.generate_all_explanations()

def create_visualization_explanation_index(
    basic_viz_paths: List[str],
    advanced_viz_paths: List[str]
) -> None:
    """
    Create an HTML index file linking visualizations with their explanations.
    
    Args:
        basic_viz_paths: List of paths to basic visualizations
        advanced_viz_paths: List of paths to advanced visualizations
    """
    print("Creating visualization-explanation index...")
    
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Federated Healthcare AI Visualizations</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { display: flex; margin-bottom: 30px; border: 1px solid #ddd; border-radius: 5px; overflow: hidden; }
            .viz-container { flex: 1; padding: 15px; }
            .explanation-container { flex: 1; padding: 15px; background: #f9f9f9; max-height: 600px; overflow-y: auto; }
            h1, h2 { color: #2c3e50; }
            h3 { margin-top: 0; color: #3498db; }
            img { max-width: 100%; box-shadow: 0 0 5px rgba(0,0,0,0.2); }
            hr { margin: 30px 0; border: none; border-top: 1px solid #eee; }
        </style>
    </head>
    <body>
        <h1>Federated Healthcare AI Visualization Dashboard</h1>
        <p>This dashboard presents visualizations generated for the federated healthcare AI project alongside educational explanations.</p>
    """
    
    # Add basic visualizations section
    html_content += """
        <h2>Basic Visualizations</h2>
    """
    
    # Map visualization filenames to explanation filenames
    viz_to_explanation = {
        "model_convergence.png": "model_convergence_explanation.txt",
        "privacy_analysis.png": "privacy_analysis_explanation.txt",
        "institutional_contribution.png": "institutional_contribution_explanation.txt",
        "performance_metrics.png": "performance_metrics_explanation.txt",
        "correlation_heatmap.png": "correlation_heatmap_explanation.txt",
        "3d_privacy_performance.png": "3d_privacy_performance_explanation.txt",
        "geographical_contribution.png": "geographical_contribution_explanation.txt",
        "parallel_coordinates.png": "parallel_coordinates_explanation.txt",
        "federated_performance_comparison.png": "federated_performance_comparison_explanation.txt"
    }
    
    # Add basic visualizations with explanations
    for viz_path in basic_viz_paths:
        viz_filename = os.path.basename(viz_path)
        explanation_filename = viz_to_explanation.get(viz_filename)
        
        if explanation_filename:
            explanation_path = os.path.join(EXPLANATION_DIR, explanation_filename)
            
            # Read explanation content
            try:
                with open(explanation_path, 'r') as f:
                    explanation_content = f.read().replace('\n', '<br>')
            except FileNotFoundError:
                explanation_content = "Explanation file not found."
            
            # Create container for this visualization and its explanation
            html_content += f"""
            <div class="container">
                <div class="viz-container">
                    <h3>{viz_filename.replace('.png', '').replace('_', ' ').title()}</h3>
                    <img src="../{viz_path}" alt="{viz_filename.replace('.png', '').replace('_', ' ').title()}">
                </div>
                <div class="explanation-container">
                    <h3>Explanation</h3>
                    <div>{explanation_content}</div>
                </div>
            </div>
            """
    
    # Add advanced visualizations section
    html_content += """
        <hr>
        <h2>Advanced Visualizations</h2>
    """
    
    # Add advanced visualizations with explanations
    for viz_path in advanced_viz_paths:
        viz_filename = os.path.basename(viz_path)
        explanation_filename = viz_to_explanation.get(viz_filename)
        
        if explanation_filename:
            explanation_path = os.path.join(EXPLANATION_DIR, explanation_filename)
            
            # Read explanation content
            try:
                with open(explanation_path, 'r') as f:
                    explanation_content = f.read().replace('\n', '<br>')
            except FileNotFoundError:
                explanation_content = "Explanation file not found."
            
            # Create container for this visualization and its explanation
            html_content += f"""
            <div class="container">
                <div class="viz-container">
                    <h3>{viz_filename.replace('.png', '').replace('_', ' ').title()}</h3>
                    <img src="../{viz_path}" alt="{viz_filename.replace('.png', '').replace('_', ' ').title()}">
                </div>
                <div class="explanation-container">
                    <h3>Explanation</h3>
                    <div>{explanation_content}</div>
                </div>
            </div>
            """
    
    # Close HTML
    html_content += """
    </body>
    </html>
    """
    
    # Write HTML file
    index_path = OUTPUT_DIR / "visualization_index.html"
    with open(index_path, 'w') as f:
        f.write(html_content)
    
    print(f"Visualization index created at {index_path}")

def main():
    """Run the visualization pipeline."""
    # Create output directories
    create_output_directories()
    
    # Load data
    data = load_data()
    
    # Generate visualizations
    basic_viz_paths = generate_basic_visualizations(data)
    advanced_viz_paths = generate_advanced_visualizations(data)
    
    # Generate explanations
    generate_visualization_explanations()
    
    # Create index linking visualizations and explanations
    create_visualization_explanation_index(basic_viz_paths, advanced_viz_paths)
    
    print("Visualization pipeline completed successfully.")

if __name__ == "__main__":
    main() 