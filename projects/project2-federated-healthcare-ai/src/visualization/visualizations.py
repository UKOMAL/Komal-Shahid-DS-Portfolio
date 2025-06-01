#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualizations for Federated Healthcare AI

This module provides comprehensive visualization functions for analyzing
federated learning results in healthcare applications.
"""

import os
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from typing import Dict, List, Tuple, Union, Optional, Any
from pathlib import Path
import warnings

# Try to import optional dependencies for advanced visualizations
try:
    import plotly.express as px  # type: ignore
    import plotly.graph_objects as go  # type: ignore
    from plotly.subplots import make_subplots  # type: ignore
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Some advanced visualizations will be disabled.")

try:
    import folium  # type: ignore
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    warnings.warn("Folium not available. Geographical visualizations will be disabled.")

# Set default style
plt.style.use('seaborn-v0_8-whitegrid')


# ====================== Basic Visualizations ======================

def plot_model_convergence(
    rounds: List[int],
    loss: List[float],
    accuracy: List[float],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    dpi: int = 100
) -> plt.Figure:
    """
    Plot model convergence over federated learning rounds.
    
    Args:
        rounds: List of round numbers
        loss: List of loss values per round
        accuracy: List of accuracy values per round
        save_path: Path to save the figure
        figsize: Figure size
        dpi: Figure resolution
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    
    # Plot loss
    ax1.plot(rounds, loss, 'r-', linewidth=2, label='Training Loss')
    ax1.set_xlabel('Federated Learning Rounds')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Convergence')
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(rounds, accuracy, 'b-', linewidth=2, label='Accuracy')
    ax2.set_xlabel('Federated Learning Rounds')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig

def plot_privacy_analysis(
    epsilon: List[float],
    accuracy: List[float],
    f1_score: List[float],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 100
) -> plt.Figure:
    """
    Plot the privacy-utility tradeoff analysis.
    
    Args:
        epsilon: List of epsilon values (privacy parameter)
        accuracy: List of accuracy values corresponding to each epsilon
        f1_score: List of F1 scores corresponding to each epsilon
        save_path: Path to save the figure
        figsize: Figure size
        dpi: Figure resolution
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    ax.plot(epsilon, accuracy, 'b-o', linewidth=2, label='Accuracy')
    ax.plot(epsilon, f1_score, 'r-^', linewidth=2, label='F1 Score')
    
    # Format x-axis for epsilon values
    epsilon_labels = [str(e) if e != np.inf else '∞' for e in epsilon]
    ax.set_xticks(range(len(epsilon)))
    ax.set_xticklabels(epsilon_labels)
    
    ax.set_xlabel('Privacy Budget (ε)')
    ax.set_ylabel('Performance Metric')
    ax.set_title('Privacy-Utility Tradeoff')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Add annotations for important points
    for i, (eps, acc) in enumerate(zip(epsilon, accuracy)):
        if i == 0 or i == len(epsilon) - 1 or i == len(epsilon) // 2:
            eps_label = '∞' if eps == np.inf else str(eps)
            ax.annotate(f'ε={eps_label}, acc={acc:.3f}',
                       xy=(i, acc), xytext=(10, -15),
                       textcoords='offset points',
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig

def plot_institutional_contribution(
    institution_names: List[str],
    accuracy: List[float],
    precision: List[float],
    recall: List[float],
    f1_score: List[float],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 100
) -> plt.Figure:
    """
    Plot the contribution of each institution to the federated model.
    
    Args:
        institution_names: List of institution names
        accuracy: List of accuracy values per institution
        precision: List of precision values per institution
        recall: List of recall values per institution
        f1_score: List of F1 scores per institution
        save_path: Path to save the figure
        figsize: Figure size
        dpi: Figure resolution
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    x = np.arange(len(institution_names))
    width = 0.2
    
    ax.bar(x - 1.5*width, accuracy, width, label='Accuracy', color='#3498db')
    ax.bar(x - 0.5*width, precision, width, label='Precision', color='#2ecc71')
    ax.bar(x + 0.5*width, recall, width, label='Recall', color='#e74c3c')
    ax.bar(x + 1.5*width, f1_score, width, label='F1 Score', color='#f39c12')
    
    ax.set_ylabel('Performance Metric')
    ax.set_title('Institutional Contribution to Federated Model')
    ax.set_xticks(x)
    ax.set_xticklabels(institution_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig

def plot_performance_metrics(
    institution_names: List[str],
    accuracy: List[float],
    precision: List[float], 
    recall: List[float],
    f1_score: List[float],
    auc: List[float],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    dpi: int = 100
) -> plt.Figure:
    """
    Plot performance metrics as a radar chart for each institution.
    
    Args:
        institution_names: List of institution names
        accuracy: List of accuracy values
        precision: List of precision values
        recall: List of recall values
        f1_score: List of F1 scores
        auc: List of AUC values
        save_path: Path to save the figure
        figsize: Figure size
        dpi: Figure resolution
        
    Returns:
        Matplotlib figure
    """
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
    num_metrics = len(metrics)
    
    # Set up the figure
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    # Number of institutions to plot
    num_institutions = min(len(institution_names), 6)  # Limit to 6 for readability
    
    # Calculate grid dimensions
    if num_institutions <= 3:
        rows, cols = 1, num_institutions
    else:
        rows, cols = 2, min(3, (num_institutions + 1) // 2)
    
    # Create subplots
    for i in range(num_institutions):
        # Create polar axes
        ax = fig.add_subplot(rows, cols, i+1, projection='polar')
        
        # Get metrics for this institution
        values = [accuracy[i], precision[i], recall[i], f1_score[i], auc[i]]
        
        # Close the loop by repeating the first value
        values += [values[0]]
        metrics_for_plot = metrics + [metrics[0]]
        
        # Compute angle for each metric
        angles = np.linspace(0, 2*np.pi, len(metrics_for_plot), endpoint=True)
        
        # Plot metrics
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        
        # Set metric labels
        ax.set_thetagrids(angles[:-1] * 180/np.pi, metrics)
        
        # Set limits and title
        ax.set_ylim(0, 1)
        ax.set_title(institution_names[i])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


# ====================== Advanced Visualizations ======================

def plot_correlation_heatmap(
    metrics: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 100
) -> plt.Figure:
    """
    Plot a correlation heatmap of performance metrics across institutions.
    
    Args:
        metrics: Dictionary mapping metric names to lists of values
        save_path: Path to save the figure
        figsize: Figure size
        dpi: Figure resolution
        
    Returns:
        Matplotlib figure
    """
    # Create DataFrame from metrics
    df = pd.DataFrame(metrics)
    
    # Compute correlation matrix
    corr_matrix = df.corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Plot heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, annot=True, fmt='.2f', ax=ax)
    
    ax.set_title('Correlation Between Performance Metrics')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig

def plot_3d_privacy_performance(
    data: Dict[str, Any],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    dpi: int = 100
) -> Optional[plt.Figure]:
    """
    Plot 3D visualization of privacy-performance-cost tradeoff.
    
    Args:
        data: Dictionary of data for visualization
        save_path: Path to save the figure
        figsize: Figure size
        dpi: Figure resolution
        
    Returns:
        Matplotlib figure or None if Plotly is not available
    """
    if not PLOTLY_AVAILABLE:
        warnings.warn("Plotly is required for 3D visualizations.")
        return None
    
    # Extract data for the 3D plot
    # This is a placeholder; in a real scenario, you would extract actual data
    epsilons = data.get('privacy_data', {}).get('epsilon', [0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    accuracy = data.get('privacy_data', {}).get('accuracy', [0.65, 0.78, 0.83, 0.87, 0.91, 0.93])
    
    # Create a cost dimension (e.g., communication cost, computation time)
    # This is simulated data; real data would come from experiment measurements
    communication_cost = [10, 8, 7, 6, 5, 4]  # MB per round, decreasing with less privacy
    
    # Create a 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=epsilons,
        y=accuracy,
        z=communication_cost,
        mode='markers+lines',
        marker=dict(
            size=8,
            color=epsilons,
            colorscale='Viridis',
            opacity=0.8
        ),
        text=[f'ε={e}, Acc={a:.2f}, Cost={c}MB' for e, a, c in zip(epsilons, accuracy, communication_cost)],
        hoverinfo='text'
    )])
    
    # Update layout
    fig.update_layout(
        title='Privacy-Performance-Cost Tradeoff',
        scene=dict(
            xaxis_title='Privacy Budget (ε)',
            yaxis_title='Accuracy',
            zaxis_title='Communication Cost (MB)',
            aspectmode='cube'
        ),
        width=figsize[0]*100,
        height=figsize[1]*100
    )
    
    # Save the figure if path is provided
    if save_path:
        # For Plotly figures, we save as HTML for interactivity
        html_path = save_path.replace('.png', '.html')
        fig.write_html(html_path)
        
        # Also save as static image
        fig.write_image(save_path, scale=dpi/100)
    
    # For compatibility with Matplotlib interface, return a basic plot
    mpl_fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.scatter(epsilons, accuracy, c=communication_cost, cmap='viridis')
    ax.set_xlabel('Privacy Budget (ε)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Privacy-Performance Tradeoff (2D projection)')
    plt.colorbar(ax.collections[0], label='Communication Cost (MB)')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        mpl_save_path = save_path.replace('.png', '_mpl.png')
        plt.savefig(mpl_save_path, dpi=dpi, bbox_inches='tight')
    
    return mpl_fig

def plot_geographical_contribution(
    data: Dict[str, Any],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 100
) -> Optional[plt.Figure]:
    """
    Plot geographical visualization of institutional contributions.
    
    Args:
        data: Dictionary of data for visualization
        save_path: Path to save the figure
        figsize: Figure size
        dpi: Figure resolution
        
    Returns:
        Matplotlib figure or None if Folium is not available
    """
    if not FOLIUM_AVAILABLE:
        warnings.warn("Folium is required for geographical visualizations.")
        return None
    
    # Create a placeholder map
    m = folium.Map(location=[37.0902, -95.7129], zoom_start=4)
    
    # Placeholder hospital locations
    # In a real implementation, this would come from actual geographic data
    hospitals = [
        {"name": "Hospital A", "location": [40.7128, -74.0060], "accuracy": 0.88, "samples": 1200},
        {"name": "Hospital B", "location": [34.0522, -118.2437], "accuracy": 0.91, "samples": 950},
        {"name": "Hospital C", "location": [41.8781, -87.6298], "accuracy": 0.85, "samples": 1500},
        {"name": "Clinic D", "location": [29.7604, -95.3698], "accuracy": 0.82, "samples": 430},
        {"name": "Research E", "location": [47.6062, -122.3321], "accuracy": 0.90, "samples": 780}
    ]
    
    # Add markers for each hospital
    for hospital in hospitals:
        # Scale marker size based on number of samples
        radius = hospital["samples"] / 100
        
        # Color based on accuracy (red = low, green = high)
        color = f'#{int(255 * (1 - hospital["accuracy"])):02x}{int(255 * hospital["accuracy"]):02x}00'
        
        folium.CircleMarker(
            location=hospital["location"],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=f"{hospital['name']}<br>Accuracy: {hospital['accuracy']:.2f}<br>Samples: {hospital['samples']}"
        ).add_to(m)
    
    # Save the map if path is provided
    if save_path:
        html_path = save_path.replace('.png', '.html')
        m.save(html_path)
    
    # For compatibility with Matplotlib interface, create a simple visualization
    mpl_fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Simple map outline (placeholder)
    ax.text(0.5, 0.5, "Interactive map saved to HTML file", ha='center', va='center', fontsize=14)
    ax.text(0.5, 0.4, "Use HTML version for full interactivity", ha='center', va='center', fontsize=12)
    
    # Add a legend of institutions
    for i, hospital in enumerate(hospitals):
        y_pos = 0.8 - i * 0.1
        color = f'#{int(255 * (1 - hospital["accuracy"])):02x}{int(255 * hospital["accuracy"]):02x}00'
        ax.plot([0.2], [y_pos], 'o', color=color, markersize=hospital["samples"]/100)
        ax.text(0.25, y_pos, f"{hospital['name']} (Acc: {hospital['accuracy']:.2f}, Samples: {hospital['samples']})",
                va='center')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Geographical Contribution (Interactive version in HTML)')
    
    if save_path:
        mpl_save_path = save_path.replace('.png', '_mpl.png')
        plt.savefig(mpl_save_path, dpi=dpi, bbox_inches='tight')
    
    return mpl_fig

def plot_parallel_coordinates(
    data: Dict[str, Any],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8),
    dpi: int = 100
) -> Optional[plt.Figure]:
    """
    Create parallel coordinates plot for multi-dimensional analysis.
    
    Args:
        data: Dictionary of data for visualization
        save_path: Path to save the figure
        figsize: Figure size
        dpi: Figure resolution
        
    Returns:
        Matplotlib figure or None if Plotly is not available
    """
    if not PLOTLY_AVAILABLE:
        warnings.warn("Plotly is required for parallel coordinates visualizations.")
        return None
    
    # Extract institution data
    institution_data = data.get('institution_data', {})
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Institution': institution_data.get('names', []),
        'Samples': institution_data.get('samples', []),
        'Accuracy': institution_data.get('accuracy', []),
        'Precision': institution_data.get('precision', []),
        'Recall': institution_data.get('recall', []),
        'F1': institution_data.get('f1_score', []),
        'AUC': institution_data.get('auc', [])
    })
    
    # Create parallel coordinates plot
    fig = px.parallel_coordinates(
        df, 
        color="Accuracy",
        labels={"Institution": "Institution", 
                "Samples": "Number of Samples",
                "Accuracy": "Accuracy", 
                "Precision": "Precision",
                "Recall": "Recall", 
                "F1": "F1 Score",
                "AUC": "AUC"},
        color_continuous_scale=px.colors.diverging.Tealrose,
        title="Multi-dimensional Comparison of Institutions"
    )
    
    # Update layout
    fig.update_layout(
        font=dict(size=12),
        width=figsize[0]*100,
        height=figsize[1]*100
    )
    
    # Save the figure if path is provided
    if save_path:
        # For Plotly figures, we save as HTML for interactivity
        html_path = save_path.replace('.png', '.html')
        fig.write_html(html_path)
        
        # Also save as static image
        fig.write_image(save_path, scale=dpi/100)
    
    # For compatibility with Matplotlib interface, return a simple representation
    mpl_fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.text(0.5, 0.5, "Interactive parallel coordinates plot saved to HTML file", 
            ha='center', va='center', fontsize=14)
    ax.text(0.5, 0.4, "Use HTML version for full interactivity", 
            ha='center', va='center', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Multi-dimensional Comparison (Interactive version in HTML)')
    
    if save_path:
        mpl_save_path = save_path.replace('.png', '_mpl.png')
        plt.savefig(mpl_save_path, dpi=dpi, bbox_inches='tight')
    
    return mpl_fig

def plot_federated_performance_comparison(
    data: Dict[str, Any],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8),
    dpi: int = 100
) -> plt.Figure:
    """
    Compare performance of federated learning vs. centralized and local models.
    
    Args:
        data: Dictionary of data for visualization
        save_path: Path to save the figure
        figsize: Figure size
        dpi: Figure resolution
        
    Returns:
        Matplotlib figure
    """
    # Create simulated data for comparison
    # In a real scenario, this would come from actual experiment results
    
    # Extract institution data
    institution_data = data.get('institution_data', {})
    institutions = institution_data.get('names', ['Institution ' + str(i+1) for i in range(5)])
    local_accuracy = institution_data.get('accuracy', [0.82, 0.85, 0.79, 0.81, 0.83])
    
    # Federated model is usually better than average of local models
    # but might not be as good as a centralized model with all data
    federated_accuracy = [0.88] * len(institutions)
    
    # Centralized model (hypothetical, if all data were shared)
    centralized_accuracy = [0.93] * len(institutions)
    
    # Privacy-preserving federated learning (with differential privacy)
    private_federated_accuracy = [0.85] * len(institutions)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    x = np.arange(len(institutions))
    width = 0.2
    
    ax.bar(x - 1.5*width, local_accuracy, width, label='Local Models', color='#3498db')
    ax.bar(x - 0.5*width, federated_accuracy, width, label='Federated Learning', color='#2ecc71')
    ax.bar(x + 0.5*width, private_federated_accuracy, width, label='Private Federated Learning', color='#e74c3c')
    ax.bar(x + 1.5*width, centralized_accuracy, width, label='Centralized (Hypothetical)', color='#f39c12')
    
    # Add text annotations
    for i, acc in enumerate(federated_accuracy):
        ax.text(i - 0.5*width, acc + 0.01, f'{acc:.2f}', ha='center', va='bottom', fontsize=9)
    
    for i, acc in enumerate(private_federated_accuracy):
        ax.text(i + 0.5*width, acc + 0.01, f'{acc:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Accuracy')
    ax.set_title('Comparison of Model Performance Across Approaches')
    ax.set_xticks(x)
    ax.set_xticklabels(institutions, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add a text box explaining the comparison
    textbox = """
    Comparison:
    - Local: Models trained on institution's data only
    - Federated: Collaborative learning without sharing data
    - Private FL: Federated learning with differential privacy
    - Centralized: Hypothetical model if all data were shared
    """
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.05, textbox, transform=ax.transAxes, fontsize=10,
           verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig

# Simple test function to demonstrate the visualizations
def _test_visualizations():
    """Run test visualizations with sample data."""
    print("Testing basic visualizations...")
    
    # Sample data for convergence plot
    sample_rounds = [
        {'loss': 0.8, 'accuracy': 0.65},
        {'loss': 0.6, 'accuracy': 0.75},
        {'loss': 0.5, 'accuracy': 0.82},
        {'loss': 0.4, 'accuracy': 0.86},
        {'loss': 0.35, 'accuracy': 0.88},
        {'loss': 0.32, 'accuracy': 0.89},
        {'loss': 0.3, 'accuracy': 0.9},
        {'loss': 0.28, 'accuracy': 0.91},
        {'loss': 0.27, 'accuracy': 0.915},
        {'loss': 0.26, 'accuracy': 0.92}
    ]
    
    plot_model_convergence(sample_rounds)
    
    # Sample data for privacy analysis
    privacy_levels = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    utility_metrics = [0.82, 0.85, 0.88, 0.9, 0.91, 0.915]
    
    plot_privacy_analysis(privacy_levels, utility_metrics)
    
    # Sample metrics data
    sample_metrics_data = {
        'Hospital A': {'accuracy': 0.92, 'precision': 0.89, 'recall': 0.94, 'f1_score': 0.91, 'auc': 0.95},
        'Hospital B': {'accuracy': 0.88, 'precision': 0.86, 'recall': 0.89, 'f1_score': 0.87, 'auc': 0.90},
        'Hospital C': {'accuracy': 0.85, 'precision': 0.84, 'recall': 0.82, 'f1_score': 0.83, 'auc': 0.87},
        'Hospital D': {'accuracy': 0.90, 'precision': 0.87, 'recall': 0.92, 'f1_score': 0.89, 'auc': 0.92},
        'Hospital E': {'accuracy': 0.87, 'precision': 0.85, 'recall': 0.86, 'f1_score': 0.85, 'auc': 0.89}
    }
    
    # Test more visualizations
    plot_institutional_contribution(sample_metrics_data, metric_name='accuracy')
    plot_metric_correlation_heatmap(sample_metrics_data)
    
    # Sample data for advanced visualizations
    print("\nTesting advanced visualizations...")
    centralized_metrics = {'accuracy': 0.95, 'precision': 0.93, 'recall': 0.94, 'f1_score': 0.93, 'auc': 0.97}
    federated_metrics = {'accuracy': 0.92, 'precision': 0.90, 'recall': 0.92, 'f1_score': 0.91, 'auc': 0.94}
    privacy_metrics = {'accuracy': 0.89, 'precision': 0.87, 'recall': 0.88, 'f1_score': 0.87, 'auc': 0.91}
    
    plot_federated_performance_comparison(centralized_metrics, federated_metrics, privacy_metrics)
    
    if ADVANCED_MPLOT_AVAILABLE:
        print("\nTesting 3D visualization...")
        communication_costs = [120, 110, 100, 85, 75, 65]
        institution_names = ['Hospital A', 'Hospital B', 'Hospital C', 'Hospital D', 'Hospital E', 'Hospital F']
        plot_3d_privacy_performance(
            privacy_levels, 
            utility_metrics, 
            communication_costs,
            institution_names
        )
    
    if PLOTLY_AVAILABLE:
        print("\nTesting interactive visualizations...")
        institution_locations = {
            'Hospital A': (40.7128, -74.0060),  # New York
            'Hospital B': (34.0522, -118.2437), # Los Angeles
            'Hospital C': (41.8781, -87.6298),  # Chicago
            'Hospital D': (29.7604, -95.3698),  # Houston
            'Hospital E': (39.9526, -75.1652)   # Philadelphia
        }
        
        plot_parallel_coordinates(
            sample_metrics_data,
            highlight_institutions=['Hospital A', 'Hospital D']
        )
        
        plot_geographical_contribution(
            sample_metrics_data,
            institution_locations
        )


if __name__ == "__main__":
    _test_visualizations() 