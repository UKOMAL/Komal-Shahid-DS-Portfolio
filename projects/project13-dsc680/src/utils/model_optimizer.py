"""
Model Optimizer Utility - Minimal Version
Provides function to fix feature mismatch issues
"""
import numpy as np
import gc
import tensorflow as tf
from tensorflow import keras
from typing import Tuple, Union, Optional

def fix_feature_mismatch(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fix feature mismatch issues by ensuring consistent features between train and test sets
    
    Args:
        X_train: Training features array
        X_test: Testing features array
        
    Returns:
        Tuple of (fixed X_train, fixed X_test)
    """
    print(f"Original shapes: X_train {X_train.shape}, X_test {X_test.shape}")
    
    # If there's a mismatch, ensure both have the same number of features
    min_features = min(X_train.shape[1], X_test.shape[1])
    X_train_fixed = X_train[:, :min_features]
    X_test_fixed = X_test[:, :min_features]
    
    print(f"Fixed shapes: X_train {X_train_fixed.shape}, X_test {X_test_fixed.shape}")
    return X_train_fixed, X_test_fixed

def optimize_lightgbm_params(n_workers: int = 4) -> dict:
    """
    Get optimized LightGBM parameters for parallel processing
    
    Args:
        n_workers: Number of parallel workers
        
    Returns:
        Dictionary of optimized parameters
    """
    return {
        'n_estimators': 100,
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'n_jobs': n_workers,
        'force_col_wise': True,  # Avoid overhead of auto-choosing
        'verbose': -1,           # Reduce verbosity
        'random_state': 42
    }

def optimize_neural_network(input_dim: int, hidden_layers: Optional[Tuple[int, ...]] = None) -> keras.Model:
    """
    Create an optimized neural network for fraud detection
    
    Args:
        input_dim: Number of input features
        hidden_layers: Optional tuple of hidden layer sizes (default: (64, 32))
        
    Returns:
        Compiled Keras model
    """
    if hidden_layers is None:
        hidden_layers = (64, 32)
    
    model = keras.Sequential()
    
    # Input layer with explicit shape
    model.add(keras.layers.Input(shape=(input_dim,)))
    
    # Hidden layers with BatchNormalization
    for units in hidden_layers:
        model.add(keras.layers.Dense(units, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.2))
    
    # Output layer
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    
    # Compile with AUC metric
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    return model

def get_memory_cleanup_callback(frequency: int = 5) -> keras.callbacks.Callback:
    """
    Create a Keras callback for memory cleanup
    
    Args:
        frequency: Cleanup frequency in epochs
        
    Returns:
        Keras callback for memory cleanup
    """
    class MemoryCleanupCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % frequency == 0:
                gc.collect()
                keras.backend.clear_session()
    
    return MemoryCleanupCallback()

def optimize_training_batch_size(n_samples: int) -> int:
    """
    Calculate optimal batch size based on dataset size
    
    Args:
        n_samples: Number of training samples
        
    Returns:
        Optimal batch size
    """
    # Heuristic for batch size: sqrt(n_samples) but capped
    batch_size = min(max(int(np.sqrt(n_samples)), 32), 256)
    
    # Round to nearest power of 2 for GPU optimization
    power = int(np.log2(batch_size))
    return 2 ** power
