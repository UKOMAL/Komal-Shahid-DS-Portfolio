#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Differential Privacy mechanisms for Federated Healthcare AI

This module implements various differential privacy techniques
to protect patient privacy in federated learning.
"""

import torch  # type: ignore
import numpy as np  # type: ignore
from typing import Dict, List, Tuple, Union, Any, Optional

class GaussianNoiseInjection:
    """
    Implements ε-differential privacy using Gaussian noise mechanism.
    """
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, sensitivity: float = 1.0):
        """
        Initialize the Gaussian noise mechanism.
        
        Args:
            epsilon: Privacy parameter (lower = more private)
            delta: Probability of privacy breach
            sensitivity: Maximum change one individual can have on the output
        """
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.noise_multiplier = self._calculate_noise_multiplier()
        
    def _calculate_noise_multiplier(self) -> float:
        """
        Calculate the noise multiplier based on epsilon and delta.
        
        Returns:
            Noise multiplier (sigma)
        """
        # Using the analytical Gaussian mechanism calibration
        # from Balle and Wang (ICML 2018)
        c = np.sqrt(2 * np.log(1.25 / self.delta))
        return c * self.sensitivity / self.epsilon
    
    def add_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Add calibrated Gaussian noise to a tensor.
        
        Args:
            tensor: Input tensor to be privatized
            
        Returns:
            Privatized tensor with noise added
        """
        noise = torch.normal(
            mean=0.0,
            std=self.noise_multiplier,
            size=tensor.shape,
            device=tensor.device if isinstance(tensor, torch.Tensor) else None
        )
        
        if isinstance(tensor, torch.Tensor):
            return tensor + noise
        else:
            # Handle numpy arrays or other types
            return tensor + noise.numpy()

def apply_differential_privacy(
    model_update: Union[Dict[str, torch.Tensor], List[torch.Tensor], torch.Tensor],
    epsilon: float = 1.0,
    delta: float = 1e-5,
    sensitivity: float = 1.0,
    clipping_norm: Optional[float] = 1.0,
    **kwargs
) -> Union[Dict[str, torch.Tensor], List[torch.Tensor], torch.Tensor]:
    """
    Apply differential privacy to model updates.
    
    Args:
        model_update: Model parameters or gradients to privatize
        epsilon: Privacy parameter (lower = more private)
        delta: Probability of privacy breach
        sensitivity: Sensitivity of the updates
        clipping_norm: L2 norm clipping threshold (None for no clipping)
        
    Returns:
        Privatized model update
    """
    # Create noise mechanism
    noise_mechanism = GaussianNoiseInjection(
        epsilon=epsilon,
        delta=delta,
        sensitivity=sensitivity
    )
    
    # Apply differential privacy based on the type of model_update
    if isinstance(model_update, dict):
        # Dictionary of parameters (common in PyTorch models)
        privatized_update: Dict[str, torch.Tensor] = {}
        for param_name, param_value in model_update.items():
            # Apply clipping if specified
            if clipping_norm is not None:
                param_value = clip_by_norm(param_value, clipping_norm)
            # Add noise
            privatized_update[param_name] = noise_mechanism.add_noise(param_value)
        return privatized_update
    
    elif isinstance(model_update, list):
        # List of parameters
        privatized_update: List[torch.Tensor] = []
        for param in model_update:
            # Apply clipping if specified
            if clipping_norm is not None:
                param = clip_by_norm(param, clipping_norm)
            # Add noise
            privatized_update.append(noise_mechanism.add_noise(param))
        return privatized_update
    
    else:
        # Single tensor
        # Apply clipping if specified
        if clipping_norm is not None:
            model_update = clip_by_norm(model_update, clipping_norm)
        # Add noise
        return noise_mechanism.add_noise(model_update)

def clip_by_norm(
    tensor: torch.Tensor, 
    clip_norm: float
) -> torch.Tensor:
    """
    Clip a tensor by its L2 norm.
    
    Args:
        tensor: Tensor to clip
        clip_norm: Maximum allowed L2 norm
        
    Returns:
        Clipped tensor
    """
    # Calculate the current L2 norm
    norm = torch.norm(tensor.float(), p=2)
    
    # Only clip if the norm exceeds the threshold
    if norm > clip_norm:
        # Scale the tensor to have norm equal to clip_norm
        scale = clip_norm / (norm + 1e-7)  # Avoid division by zero
        return tensor * scale
    
    return tensor

def calculate_privacy_spent(
    epsilon: float,
    delta: float,
    num_iterations: int,
    batch_size: int,
    dataset_size: int
) -> Dict[str, float]:
    """
    Calculate the privacy budget spent using moments accountant.
    
    Args:
        epsilon: Privacy parameter per iteration
        delta: Target delta
        num_iterations: Number of SGD iterations
        batch_size: Batch size
        dataset_size: Total number of samples
        
    Returns:
        Dictionary with privacy metrics
    """
    # Sampling rate (probability of including each example)
    q = batch_size / dataset_size
    
    # Using the simplified moments accountant formula (approximate)
    # This is a simplified approximation based on the paper
    # "Deep Learning with Differential Privacy" by Abadi et al.
    noise_multiplier = np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    
    # Privacy amplification by sampling and composition
    epsilon_total = np.sqrt(2 * np.log(1/delta)) * q * np.sqrt(num_iterations) / noise_multiplier
    
    return {
        "epsilon_per_iteration": epsilon,
        "epsilon_total": float(epsilon_total),
        "delta": delta,
        "noise_multiplier": float(noise_multiplier),
        "sampling_rate": float(q),
        "num_iterations": num_iterations
    }

class PrivacyEngine:
    """
    Privacy engine for tracking and enforcing privacy budget.
    """
    
    def __init__(self, 
                 target_epsilon: float = 1.0,
                 target_delta: float = 1e-5,
                 max_grad_norm: float = 1.0):
        """
        Initialize the privacy engine.
        
        Args:
            target_epsilon: Target privacy budget
            target_delta: Target delta (probability of privacy breach)
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.max_grad_norm = max_grad_norm
        
        # Privacy accounting
        self.steps = 0
        self.epsilon_spent = 0.0
        
    def step(self, grads: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Process gradients for one step of training.
        
        Args:
            grads: List of gradient tensors
            
        Returns:
            Privacy-preserving gradients
        """
        # Clip gradients
        clipped_grads = [clip_by_norm(g, self.max_grad_norm) for g in grads]
        
        # Dynamically adjust noise based on remaining privacy budget
        remaining_budget = max(0.0, self.target_epsilon - self.epsilon_spent)
        if remaining_budget <= 0:
            raise ValueError("Privacy budget exhausted. Cannot continue training.")
        
        # Determine noise multiplier for this step
        # (simplified; a real implementation would use advanced calibration)
        noise_multiplier = 1.0 / remaining_budget
        
        # Add noise to gradients
        private_grads = []
        for g in clipped_grads:
            noise = torch.normal(
                mean=0.0,
                std=noise_multiplier * self.max_grad_norm,
                size=g.shape,
                device=g.device
            )
            private_grads.append(g + noise)
        
        # Update privacy accounting
        self.steps += 1
        # Update epsilon spent (simplified; a real implementation would use 
        # moments accountant or advanced privacy accounting)
        self.epsilon_spent += (1.0 / noise_multiplier)
        
        return private_grads
    
    def get_privacy_spent(self) -> Dict[str, float]:
        """
        Get the current privacy budget spent.
        
        Returns:
            Dictionary with privacy metrics
        """
        return {
            "target_epsilon": self.target_epsilon,
            "epsilon_spent": self.epsilon_spent,
            "target_delta": self.target_delta,
            "remaining_budget": max(0.0, self.target_epsilon - self.epsilon_spent),
            "steps": self.steps
        }

class GaussianMechanism:
    """
    Implementation of the Gaussian Mechanism for differential privacy.
    This adds calibrated Gaussian noise to preserve ε-δ differential privacy.
    """
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, sensitivity: float = 1.0):
        """
        Initialize the Gaussian Mechanism.
        
        Args:
            epsilon: Privacy parameter ε (lower means more private)
            delta: Privacy parameter δ (probability of privacy breach)
            sensitivity: L2 sensitivity of the query function
        """
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        
        # Calculate sigma (noise scale) based on privacy parameters
        self.sigma = self._calculate_sigma()
        
    def _calculate_sigma(self) -> float:
        """Calculate the noise scale for the given privacy parameters."""
        # Formula for ε-δ differential privacy with Gaussian mechanism
        return self.sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
    
    def add_noise(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Add calibrated Gaussian noise to the data.
        
        Args:
            data: Input data (numpy array or PyTorch tensor)
            
        Returns:
            Data with added noise
        """
        if isinstance(data, torch.Tensor):
            noise = torch.normal(0, self.sigma, size=data.shape).to(data.device)
            return data + noise
        else:
            noise = np.random.normal(0, self.sigma, size=data.shape)
            return data + noise

class LaplaceMechanism:
    """
    Implementation of the Laplace Mechanism for differential privacy.
    This adds calibrated Laplace noise to preserve ε differential privacy.
    """
    
    def __init__(self, epsilon: float = 1.0, sensitivity: float = 1.0):
        """
        Initialize the Laplace Mechanism.
        
        Args:
            epsilon: Privacy parameter ε (lower means more private)
            sensitivity: L1 sensitivity of the query function
        """
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        
        # Calculate b (scale) parameter for Laplace distribution
        self.scale = self.sensitivity / self.epsilon
    
    def add_noise(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Add calibrated Laplace noise to the data.
        
        Args:
            data: Input data (numpy array or PyTorch tensor)
            
        Returns:
            Data with added noise
        """
        if isinstance(data, torch.Tensor):
            # Generate Laplace noise for PyTorch tensor
            uniform = torch.rand(data.shape, device=data.device) - 0.5
            noise = -self.scale * torch.sign(uniform) * torch.log(1 - 2 * torch.abs(uniform))
            return data + noise
        else:
            # Generate Laplace noise for numpy array
            noise = np.random.laplace(0, self.scale, size=data.shape)
            return data + noise

class PrivateAggregation:
    """
    Privacy-preserving aggregation methods for federated learning.
    These methods can be used to aggregate model updates while preserving privacy.
    """
    
    def __init__(self, 
                privacy_mechanism: str = 'gaussian',
                epsilon: float = 1.0, 
                delta: float = 1e-5,
                sensitivity: float = 1.0,
                clipping_norm: float = 1.0):
        """
        Initialize private aggregation.
        
        Args:
            privacy_mechanism: Type of privacy mechanism ('gaussian' or 'laplace')
            epsilon: Privacy parameter ε (lower means more private)
            delta: Privacy parameter δ (only used for Gaussian mechanism)
            sensitivity: Sensitivity of the query function
            clipping_norm: L2 norm for gradient clipping
        """
        self.privacy_mechanism = privacy_mechanism.lower()
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.clipping_norm = clipping_norm
        
        # Create the appropriate privacy mechanism
        if self.privacy_mechanism == 'gaussian':
            self.mechanism = GaussianMechanism(epsilon, delta, sensitivity)
        elif self.privacy_mechanism == 'laplace':
            self.mechanism = LaplaceMechanism(epsilon, sensitivity)
        else:
            raise ValueError(f"Unknown privacy mechanism: {privacy_mechanism}")
    
    def clip_gradients(self, gradients: torch.Tensor) -> torch.Tensor:
        """
        Clip gradients to limit sensitivity.
        
        Args:
            gradients: Model gradients or updates
            
        Returns:
            Clipped gradients
        """
        total_norm = torch.norm(gradients)
        clip_coef = self.clipping_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            return gradients * clip_coef
        return gradients
    
    def privatize_update(self, model_update: torch.Tensor) -> torch.Tensor:
        """
        Make a model update differentially private.
        
        Args:
            model_update: Model update (gradients or weights)
            
        Returns:
            Privatized model update
        """
        # Clip the update to bound sensitivity
        clipped_update = self.clip_gradients(model_update)
        
        # Add noise according to the chosen mechanism
        private_update = self.mechanism.add_noise(clipped_update)
        
        return private_update
    
    def aggregate_updates(self, updates: List[torch.Tensor]) -> torch.Tensor:
        """
        Privately aggregate multiple model updates.
        
        Args:
            updates: List of model updates from different clients
            
        Returns:
            Privately aggregated update
        """
        # Clip each update
        clipped_updates = [self.clip_gradients(update) for update in updates]
        
        # Average the updates
        avg_update = torch.mean(torch.stack(clipped_updates), dim=0)
        
        # Add noise to the average
        private_aggregate = self.mechanism.add_noise(avg_update)
        
        return private_aggregate

def compute_privacy_budget(
    sampling_rate: float,
    noise_multiplier: float,
    iterations: int,
    delta: float = 1e-5
) -> Tuple[float, float]:
    """
    Compute the privacy budget (ε, δ) based on Rényi Differential Privacy.
    This is a simplified implementation - for production, use a library like TensorFlow Privacy.
    
    Args:
        sampling_rate: Probability of sampling each client
        noise_multiplier: Noise scale relative to sensitivity
        iterations: Number of training iterations
        delta: Target δ for (ε, δ)-differential privacy
        
    Returns:
        Tuple of (epsilon, delta)
    """
    # This is a placeholder - in a real implementation, you would use
    # a proper accounting method like Rényi Differential Privacy or
    # moments accountant to compute the privacy budget.
    
    # Very rough approximation based on the analytical Gaussian mechanism
    epsilon = np.sqrt(2 * np.log(1.25 / delta)) / noise_multiplier
    
    # Scale by sampling rate and iterations
    epsilon = epsilon * np.sqrt(iterations * sampling_rate)
    
    return epsilon, delta

def secure_aggregation_protocol(client_updates: List[torch.Tensor], 
                              secure_weights: Optional[List[float]] = None) -> torch.Tensor:
    """
    Simulates a secure aggregation protocol for federated learning.
    In a real implementation, this would use cryptographic techniques to 
    ensure that individual updates cannot be inspected.
    
    Args:
        client_updates: List of model updates from different clients
        secure_weights: Optional weights for weighted aggregation
        
    Returns:
        Securely aggregated model update
    """
    # In a real implementation, this function would use secure multiparty computation
    # or homomorphic encryption to aggregate updates without revealing them.
    
    # For this simulation, we'll just do weighted averaging
    if secure_weights is None:
        secure_weights = [1.0 / len(client_updates)] * len(client_updates)
    
    # Simple weighted average
    weighted_updates = [w * update for w, update in zip(secure_weights, client_updates)]
    
    # Sum the weighted updates
    aggregate = sum(weighted_updates)
    
    return aggregate 