#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Secure Aggregation for Federated Healthcare AI

This module implements secure aggregation protocols to protect
privacy during model parameter aggregation in federated learning.
"""

import os
import numpy as np  # type: ignore
import torch  # type: ignore
from typing import Dict, List, Tuple, Union, Any, Optional
from collections import OrderedDict

def generate_random_seed() -> int:
    """Generate a cryptographically secure random seed."""
    return int.from_bytes(os.urandom(4), byteorder='big')

def generate_mask(
    shape: Union[Tuple[int, ...], List[int]],
    seed: Optional[int] = None
) -> Union[np.ndarray, torch.Tensor]:
    """
    Generate a random mask for secure aggregation.
    
    Args:
        shape: Shape of the mask to generate
        seed: Random seed for reproducibility
        
    Returns:
        Random mask of the specified shape
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Generate random mask
    # The mask should be large enough to hide the model parameters
    # but should cancel out when aggregated across all clients
    return torch.normal(0, 100.0, size=shape)

def apply_mask(
    model_parameters: Union[Dict[str, torch.Tensor], torch.Tensor],
    mask: Optional[Union[Dict[str, torch.Tensor], torch.Tensor]] = None,
    seed: Optional[int] = None
) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Apply a mask to model parameters for secure aggregation.
    
    Args:
        model_parameters: Model parameters to mask
        mask: Pre-generated mask to apply (if None, generate one using seed)
        seed: Random seed for mask generation
        
    Returns:
        Masked model parameters
    """
    if isinstance(model_parameters, dict):
        # Dictionary of parameters (common for PyTorch models)
        masked_params = OrderedDict()
        
        for name, param in model_parameters.items():
            # Generate or use provided mask
            if mask is None:
                param_mask = generate_mask(param.shape, seed)
            else:
                param_mask = mask[name] if isinstance(mask, dict) else mask
            
            # Apply mask
            masked_params[name] = param + param_mask
            
        return masked_params
    else:
        # Tensor parameters
        if mask is None:
            mask = generate_mask(model_parameters.shape, seed)
        
        # Apply mask
        return model_parameters + mask

def remove_mask(
    masked_parameters: Union[Dict[str, torch.Tensor], torch.Tensor],
    mask: Union[Dict[str, torch.Tensor], torch.Tensor]
) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Remove a mask from masked model parameters.
    
    Args:
        masked_parameters: Masked model parameters
        mask: Mask to remove
        
    Returns:
        Original model parameters
    """
    if isinstance(masked_parameters, dict):
        # Dictionary of parameters
        unmasked_params = OrderedDict()
        
        for name, param in masked_parameters.items():
            param_mask = mask[name] if isinstance(mask, dict) else mask
            unmasked_params[name] = param - param_mask
            
        return unmasked_params
    else:
        # Tensor parameters
        return masked_parameters - mask

def secure_aggregate(
    client_updates: List[Union[Dict[str, torch.Tensor], torch.Tensor]],
    masks: Optional[List[Union[Dict[str, torch.Tensor], torch.Tensor]]] = None,
    **kwargs
) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Securely aggregate model updates from multiple clients.
    
    Args:
        client_updates: List of model updates from clients
        masks: List of masks to remove (if applicable)
        
    Returns:
        Securely aggregated model update
    """
    if not client_updates:
        raise ValueError("No client updates provided for aggregation")
    
    # Handle dictionary-style updates (PyTorch models)
    if isinstance(client_updates[0], dict):
        # Initialize aggregated with zeros
        aggregated = OrderedDict()
        
        # Get parameter names from first client
        for name, param in client_updates[0].items():
            # Initialize with zeros of the same shape
            aggregated[name] = torch.zeros_like(param)
        
        # Sum up all client updates
        for i, update in enumerate(client_updates):
            for name, param in update.items():
                # Remove mask if provided
                if masks is not None and i < len(masks):
                    if isinstance(masks[i], dict):
                        param = param - masks[i][name]
                    else:
                        param = param - masks[i]
                
                aggregated[name] += param
        
        # Average the parameters
        num_clients = len(client_updates)
        for name in aggregated:
            aggregated[name] /= num_clients
            
        return aggregated
    
    # Handle tensor updates
    else:
        # Sum up all client updates
        aggregated = torch.zeros_like(client_updates[0])
        
        for i, update in enumerate(client_updates):
            # Remove mask if provided
            if masks is not None and i < len(masks):
                update = update - masks[i]
            
            aggregated += update
        
        # Average the parameters
        return aggregated / len(client_updates)

class SecureAggregationProtocol:
    """
    Implementation of a secure aggregation protocol for federated learning.
    
    This class implements a simplified version of the secure aggregation
    protocol described in the paper:
    "Practical Secure Aggregation for Privacy-Preserving Machine Learning"
    by Bonawitz et al.
    """
    
    def __init__(self, num_clients: int, threshold: Optional[int] = None):
        """
        Initialize the secure aggregation protocol.
        
        Args:
            num_clients: Number of participating clients
            threshold: Minimum number of clients required for aggregation
        """
        self.num_clients = num_clients
        self.threshold = threshold or max(2, num_clients // 2)
        self.client_seeds = {}
        self.client_masks = {}
        
    def setup(self) -> Dict[int, int]:
        """
        Set up the secure aggregation protocol.
        
        Returns:
            Dictionary mapping client IDs to random seeds
        """
        # Generate random seeds for each client
        client_seeds = {}
        
        for client_id in range(self.num_clients):
            seed = generate_random_seed()
            client_seeds[client_id] = seed
            self.client_seeds[client_id] = seed
        
        return client_seeds
    
    def generate_client_mask(
        self, 
        client_id: int, 
        model_shape: Union[Dict[str, Tuple[int, ...]], Tuple[int, ...]]
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Generate a mask for a specific client.
        
        Args:
            client_id: ID of the client
            model_shape: Shape of the model parameters
            
        Returns:
            Mask for the client
        """
        if client_id not in self.client_seeds:
            raise ValueError(f"Client {client_id} not initialized")
        
        seed = self.client_seeds[client_id]
        
        if isinstance(model_shape, dict):
            # Dictionary of parameter shapes
            mask = OrderedDict()
            
            for name, shape in model_shape.items():
                mask[name] = generate_mask(shape, seed)
            
            self.client_masks[client_id] = mask
            return mask
        else:
            # Single tensor shape
            mask = generate_mask(model_shape, seed)
            self.client_masks[client_id] = mask
            return mask
    
    def aggregate(
        self, 
        client_updates: Dict[int, Union[Dict[str, torch.Tensor], torch.Tensor]],
        dropped_clients: Optional[List[int]] = None
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Securely aggregate model updates from clients.
        
        Args:
            client_updates: Dictionary mapping client IDs to their model updates
            dropped_clients: List of client IDs that dropped out
            
        Returns:
            Securely aggregated model update
        """
        dropped_clients = dropped_clients or []
        
        # Check if we have enough clients
        participating_clients = [
            client_id for client_id in client_updates
            if client_id not in dropped_clients
        ]
        
        if len(participating_clients) < self.threshold:
            raise ValueError(
                f"Not enough clients for secure aggregation. "
                f"Need {self.threshold}, but only have {len(participating_clients)}"
            )
        
        # Convert to list for secure_aggregate function
        updates_list = [client_updates[client_id] for client_id in participating_clients]
        
        # Get masks for participating clients
        masks_list = [
            self.client_masks[client_id] for client_id in participating_clients
            if client_id in self.client_masks
        ]
        
        # Securely aggregate the updates
        return secure_aggregate(updates_list, masks_list)
    
    def reset(self):
        """Reset the protocol state."""
        self.client_seeds = {}
        self.client_masks = {} 