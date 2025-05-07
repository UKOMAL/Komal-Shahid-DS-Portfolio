#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Factory for Federated Healthcare AI

This module provides model architectures for different healthcare data modalities.
The factory pattern allows easy selection of appropriate models based on data type.
"""

from typing import Any, Callable, Dict, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

class TabularModel(nn.Module):
    """Neural network model for tabular clinical data."""
    
    def __init__(self, input_dim: int, hidden_dims: list = [256, 128, 64], 
                 output_dim: int = 1, dropout: float = 0.3):
        """
        Initialize TabularModel.
        
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
            output_dim: Dimension of output (1 for binary classification)
            dropout: Dropout probability
        """
        super(TabularModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Create layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
    def forward(self, x):
        """Forward pass."""
        x = self.layers(x)
        return self.output_layer(x)

class MedicalImageCNN(nn.Module):
    """Convolutional neural network for medical imaging."""
    
    def __init__(self, in_channels: int = 3, num_classes: int = 2):
        """
        Initialize MedicalImageCNN.
        
        Args:
            in_channels: Number of input channels (3 for RGB, 1 for grayscale)
            num_classes: Number of output classes
        """
        super(MedicalImageCNN, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # First convolution block
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second convolution block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third convolution block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth convolution block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        """Forward pass."""
        x = self.features(x)
        x = self.adaptive_pool(x)
        return self.classifier(x)

class TimeseriesLSTM(nn.Module):
    """LSTM model for physiological time series data."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 num_layers: int = 2, output_dim: int = 1, 
                 bidirectional: bool = True, dropout: float = 0.3):
        """
        Initialize TimeseriesLSTM.
        
        Args:
            input_dim: Number of input features per time step
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            output_dim: Dimension of output
            bidirectional: Whether to use bidirectional LSTM
            dropout: Dropout probability
        """
        super(TimeseriesLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * self.directions, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        """Forward pass."""
        # x shape: (batch_size, sequence_length, input_dim)
        lstm_out, _ = self.lstm(x)
        
        # Use the output from the last time step
        lstm_out = lstm_out[:, -1, :]
        
        # Pass through the classifier
        return self.classifier(lstm_out)

class MultimodalFusionModel(nn.Module):
    """Fusion model for multiple healthcare data modalities."""
    
    def __init__(self, tabular_dim: int, image_channels: int = 3, 
                 timeseries_dim: int = 10, output_dim: int = 1):
        """
        Initialize MultimodalFusionModel.
        
        Args:
            tabular_dim: Dimension of tabular features
            image_channels: Number of image channels
            timeseries_dim: Dimension of time series features
            output_dim: Dimension of output
        """
        super(MultimodalFusionModel, self).__init__()
        
        # Tabular pathway
        self.tabular_model = nn.Sequential(
            nn.Linear(tabular_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64)
        )
        
        # Image pathway (simplified CNN)
        self.image_model = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 64)
        )
        
        # Time series pathway
        self.timeseries_model = nn.LSTM(
            timeseries_dim,
            64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.timeseries_fc = nn.Linear(64 * 2, 64)
        
        # Fusion layer
        fusion_dim = 64 + 64 + 64  # Sum of all pathway output dimensions
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, tabular, image, timeseries):
        """
        Forward pass with multiple input modalities.
        
        Args:
            tabular: Tabular features tensor
            image: Image tensor
            timeseries: Time series tensor
            
        Returns:
            Output prediction
        """
        # Process each pathway
        tabular_features = self.tabular_model(tabular)
        image_features = self.image_model(image)
        
        # Process time series
        timeseries_out, _ = self.timeseries_model(timeseries)
        timeseries_out = timeseries_out[:, -1, :]  # Last time step
        timeseries_features = self.timeseries_fc(timeseries_out)
        
        # Concatenate features from all modalities
        combined_features = torch.cat(
            [tabular_features, image_features, timeseries_features], 
            dim=1
        )
        
        # Final prediction
        return self.fusion_layer(combined_features)

def get_model_for_modality(
    modality: str,
    **kwargs
) -> Optional[Callable[[], nn.Module]]:
    """
    Get appropriate model for a given data modality.
    
    Args:
        modality: Type of healthcare data ('tabular', 'image', 'timeseries', 'multimodal')
        **kwargs: Additional model parameters
        
    Returns:
        Function that creates and returns the appropriate model
    """
    if modality == 'tabular':
        input_dim = kwargs.get('input_dim', 128)
        hidden_dims = kwargs.get('hidden_dims', [256, 128, 64])
        output_dim = kwargs.get('output_dim', 1)
        
        return lambda: TabularModel(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim
        )
        
    elif modality == 'image':
        in_channels = kwargs.get('in_channels', 3)
        num_classes = kwargs.get('num_classes', 2)
        
        return lambda: MedicalImageCNN(
            in_channels=in_channels,
            num_classes=num_classes
        )
        
    elif modality == 'timeseries':
        input_dim = kwargs.get('input_dim', 10)
        hidden_dim = kwargs.get('hidden_dim', 128)
        output_dim = kwargs.get('output_dim', 1)
        
        return lambda: TimeseriesLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
        
    elif modality == 'multimodal':
        tabular_dim = kwargs.get('tabular_dim', 128)
        image_channels = kwargs.get('image_channels', 3)
        timeseries_dim = kwargs.get('timeseries_dim', 10)
        output_dim = kwargs.get('output_dim', 1)
        
        return lambda: MultimodalFusionModel(
            tabular_dim=tabular_dim,
            image_channels=image_channels,
            timeseries_dim=timeseries_dim,
            output_dim=output_dim
        )
        
    else:
        # Unknown modality
        return None 