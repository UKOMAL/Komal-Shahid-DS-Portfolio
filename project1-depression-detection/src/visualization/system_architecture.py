#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
System Architecture Diagram Generator

This script creates a visual representation of the depression detection system's
architecture, showing the flow of data through various components.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.path import Path

# Create output directory if it doesn't exist
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "output")
os.makedirs(output_dir, exist_ok=True)

def create_system_architecture_diagram():
    """Creates a system architecture diagram and saves it to the output directory."""
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define colors
    colors = {
        'input': '#e6f2ff',    # Light blue
        'process': '#e6ffe6',  # Light green
        'model': '#fff2e6',    # Light orange
        'output': '#f2e6ff',   # Light purple
        'arrow': '#404040'     # Dark gray
    }
    
    # Define components as rectangles (x, y, width, height)
    components = {
        'text_input': {'pos': (1, 7), 'size': (2, 1), 'label': 'Text Input', 'color': colors['input']},
        'preprocessing': {'pos': (4, 7), 'size': (2, 1), 'label': 'Preprocessing', 'color': colors['process']},
        'feature_extraction': {'pos': (7, 7), 'size': (2, 1), 'label': 'Feature Extraction', 'color': colors['process']},
        
        'transformer_model': {'pos': (5.5, 5), 'size': (3, 1.5), 'label': 'Transformer Model\n(RoBERTa)', 'color': colors['model']},
        'attention_layer': {'pos': (3, 5), 'size': (2, 0.8), 'label': 'Attention Layer', 'color': colors['model']},
        'feature_importance': {'pos': (9, 5), 'size': (2, 0.8), 'label': 'Feature Importance', 'color': colors['model']},
        
        'classification': {'pos': (5.5, 3), 'size': (3, 1), 'label': 'Classification', 'color': colors['process']},
        'confidence_scoring': {'pos': (9, 3), 'size': (2, 1), 'label': 'Confidence Scoring', 'color': colors['process']},
        
        'prediction_output': {'pos': (4, 1), 'size': (2, 1), 'label': 'Depression\nPrediction', 'color': colors['output']},
        'visualization': {'pos': (7, 1), 'size': (2, 1), 'label': 'Visualizations', 'color': colors['output']},
        'explanation': {'pos': (10, 1), 'size': (2, 1), 'label': 'Explanations', 'color': colors['output']},
    }
    
    # Draw components
    for name, comp in components.items():
        x, y = comp['pos']
        w, h = comp['size']
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='black',
                                 facecolor=comp['color'], alpha=0.8, zorder=1)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, comp['label'], ha='center', va='center', fontsize=10)
    
    # Define arrows for data flow
    arrows = [
        # Main pipeline flow
        ('text_input', 'preprocessing'),
        ('preprocessing', 'feature_extraction'),
        ('feature_extraction', 'transformer_model'),
        ('transformer_model', 'classification'),
        ('classification', 'prediction_output'),
        
        # Additional flows
        ('transformer_model', 'attention_layer'),
        ('transformer_model', 'feature_importance'),
        ('attention_layer', 'visualization'),
        ('feature_importance', 'explanation'),
        ('classification', 'confidence_scoring'),
        ('confidence_scoring', 'visualization'),
    ]
    
    # Draw arrows
    for start, end in arrows:
        start_comp = components[start]
        end_comp = components[end]
        
        # Calculate start and end points
        start_x = start_comp['pos'][0] + start_comp['size'][0]/2
        start_y = start_comp['pos'][1] + start_comp['size'][1]/2
        end_x = end_comp['pos'][0] + end_comp['size'][0]/2
        end_y = end_comp['pos'][1] + end_comp['size'][1]/2
        
        # Adjust points if components are side by side (horizontal arrow)
        if abs(start_y - end_y) < 0.5:
            if start_x < end_x:  # left to right
                start_x = start_comp['pos'][0] + start_comp['size'][0]
                end_x = end_comp['pos'][0]
            else:  # right to left
                start_x = start_comp['pos'][0]
                end_x = end_comp['pos'][0] + end_comp['size'][0]
        
        # Adjust points if components are one above the other (vertical arrow)
        elif abs(start_x - end_x) < 0.5:
            if start_y < end_y:  # bottom to top
                start_y = start_comp['pos'][1] + start_comp['size'][1]
                end_y = end_comp['pos'][1]
            else:  # top to bottom
                start_y = start_comp['pos'][1]
                end_y = end_comp['pos'][1] + end_comp['size'][1]
        
        # Draw arrow
        arrow = patches.FancyArrowPatch((start_x, start_y), (end_x, end_y),
                                        connectionstyle="arc3,rad=.1",
                                        arrowstyle="simple",
                                        color=colors['arrow'],
                                        linewidth=1.5,
                                        zorder=0)
        ax.add_patch(arrow)
    
    # Add legend for component types
    legend_elements = [
        patches.Patch(facecolor=colors['input'], edgecolor='black', label='Input'),
        patches.Patch(facecolor=colors['process'], edgecolor='black', label='Processing'),
        patches.Patch(facecolor=colors['model'], edgecolor='black', label='Model Components'),
        patches.Patch(facecolor=colors['output'], edgecolor='black', label='Output')
    ]
    ax.legend(handles=legend_elements, loc='upper center', 
              bbox_to_anchor=(0.5, 0.05), ncol=4)
    
    # Add title and labels
    plt.title('Depression Detection System Architecture', fontsize=14, fontweight='bold')
    
    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 9)
    
    # Add data flow direction label
    ax.text(1, 8.5, "Data Flow →", fontsize=10, fontweight='bold')
    
    # Save figure
    plt.tight_layout()
    output_path = os.path.join(output_dir, "system_architecture.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"System architecture diagram saved to {output_path}")
    return output_path

if __name__ == "__main__":
    create_system_architecture_diagram() 