#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization Explanations for Federated Healthcare AI

This module generates educational explanations for the visualizations 
used in the federated healthcare AI project, making them more 
interpretable for both technical and non-technical stakeholders.
"""

import os
from typing import Dict, List, Optional, Any
from pathlib import Path

class VisualizationExplainer:
    """Generates explanations for federated learning visualizations."""
    
    def __init__(self, output_dir: str):
        """
        Initialize the visualization explainer.
        
        Args:
            output_dir: Directory to save explanation files
        """
        self.output_dir = Path(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.explanations = self._initialize_explanations()
    
    def _initialize_explanations(self) -> Dict[str, str]:
        """
        Initialize explanation templates for each visualization type.
        
        Returns:
            Dictionary mapping visualization names to explanation texts
        """
        return {
            "model_convergence": self._get_model_convergence_explanation(),
            "privacy_analysis": self._get_privacy_analysis_explanation(),
            "institutional_contribution": self._get_institutional_contribution_explanation(),
            "performance_metrics": self._get_performance_metrics_explanation(),
            "correlation_heatmap": self._get_correlation_heatmap_explanation(),
            "3d_privacy_performance": self._get_3d_privacy_performance_explanation(),
            "geographical_contribution": self._get_geographical_contribution_explanation(),
            "parallel_coordinates": self._get_parallel_coordinates_explanation(),
            "federated_performance_comparison": self._get_federated_performance_comparison_explanation()
        }
    
    def generate_explanation(self, visualization_name: str, data: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate an explanation for a specific visualization.
        
        Args:
            visualization_name: Name of the visualization
            data: Data used to generate the visualization (for customizing explanation)
            
        Returns:
            Explanation text
        """
        base_explanation = self.explanations.get(
            visualization_name, 
            f"No explanation available for {visualization_name}."
        )
        
        # If we have data, we could customize the explanation further
        if data:
            # This would be expanded in a real implementation to generate
            # more insight-specific explanations based on the actual data
            pass
        
        return base_explanation
    
    def save_explanation(self, visualization_name: str, data: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate and save an explanation for a visualization.
        
        Args:
            visualization_name: Name of the visualization
            data: Data used to generate the visualization
            
        Returns:
            Path to the saved explanation file
        """
        explanation = self.generate_explanation(visualization_name, data)
        file_path = self.output_dir / f"{visualization_name}_explanation.txt"
        
        with open(file_path, 'w') as f:
            f.write(explanation)
        
        return str(file_path)
    
    def generate_all_explanations(self, data: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Generate and save explanations for all visualization types.
        
        Args:
            data: Data used to generate the visualizations
            
        Returns:
            List of paths to saved explanation files
        """
        saved_paths = []
        
        for viz_name in self.explanations:
            path = self.save_explanation(viz_name, data)
            saved_paths.append(path)
        
        return saved_paths
    
    def _get_model_convergence_explanation(self) -> str:
        """Explanation for model convergence visualization."""
        return "This visualization shows how the federated learning model converges over training rounds."
    
    def _get_privacy_analysis_explanation(self) -> str:
        """Explanation for privacy analysis visualization."""
        return "This visualization shows the tradeoff between privacy and model performance."
    
    def _get_institutional_contribution_explanation(self) -> str:
        """Explanation for institutional contribution visualization."""
        return "This visualization shows performance metrics across different healthcare institutions."
    
    def _get_performance_metrics_explanation(self) -> str:
        """Explanation for performance metrics visualization."""
        return "This visualization shows comprehensive performance metrics for each institution."
    
    def _get_correlation_heatmap_explanation(self) -> str:
        """Explanation for correlation heatmap visualization."""
        return "This heatmap shows correlations between different performance metrics."
    
    def _get_3d_privacy_performance_explanation(self) -> str:
        """Explanation for 3D privacy-performance visualization."""
        return "This 3D plot shows the relationship between privacy, performance, and communication cost."
    
    def _get_geographical_contribution_explanation(self) -> str:
        """Explanation for geographical contribution visualization."""
        return "This map shows the geographical distribution of healthcare institutions and their performance."
    
    def _get_parallel_coordinates_explanation(self) -> str:
        """Explanation for parallel coordinates visualization."""
        return "This parallel coordinates plot shows relationships between multiple metrics across institutions."
    
    def _get_federated_performance_comparison_explanation(self) -> str:
        """Explanation for federated performance comparison visualization."""
        return "This visualization compares federated, private federated, local, and centralized approaches." 