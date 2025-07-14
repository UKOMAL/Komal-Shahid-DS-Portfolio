"""
Optimization Utilities - Static Class Implementation
Provides memory optimization and distributed computing utilities
"""
import os
import gc
import tensorflow as tf
import dask.dataframe as dd
import pandas as pd
import numpy as np
from dask.distributed import Client, LocalCluster
from pathlib import Path
from typing import Union, Callable, Optional

# Enable dask-expr if available
os.environ["DASK_EXPR"] = "1"

# Configure TensorFlow threading before importing
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "4"

class OptimUtils:
    """Static utility class for optimization functions"""
    
    # Global client and cluster
    client = None
    cluster = None
    
    @staticmethod
    def setup_tensorflow():
        """Configure TensorFlow for optimal memory usage"""
        # Reduce TF logging and set memory growth
        tf.get_logger().setLevel('ERROR')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        # Configure GPU memory growth if available
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    
    @staticmethod
    def setup_dask(n_workers=4, threads_per_worker=2, memory_limit='2GB'):
        """Set up DASK distributed computing cluster"""
        if OptimUtils.client:
            return OptimUtils.client
            
        OptimUtils.cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            memory_limit=memory_limit,
            dashboard_address=":8787"
        )
        
        OptimUtils.client = Client(OptimUtils.cluster)
        return OptimUtils.client
    
    @staticmethod
    def memory_cleanup_callback(frequency=5):
        """Create TensorFlow callback for memory cleanup"""
        class MemoryCleanupCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if (epoch + 1) % frequency == 0:
                    gc.collect()
                    tf.keras.backend.clear_session()
        return MemoryCleanupCallback()

    @staticmethod
    def process_dataframe(
        data: Union[str, pd.DataFrame, dd.DataFrame],
        func: Optional[Callable] = None,
        n_workers: int = 4,
        **kwargs
    ) -> dd.DataFrame:
        """Process DataFrame or load and process CSV data using DASK
        
        Args:
            data: Input data - can be:
                - Path to CSV file (str)
                - pandas DataFrame
                - dask DataFrame
            func: Optional processing function to apply
            n_workers: Number of workers for parallel processing
            **kwargs: Additional arguments passed to dd.read_csv if data is a path
            
        Returns:
            Processed Dask DataFrame
        """
        try:
            # Ensure client is set up
            if not OptimUtils.client:
                OptimUtils.setup_dask(n_workers=n_workers)

            # Handle input data type
            if isinstance(data, str):
                # Load CSV file with optimized settings
                kwargs.setdefault('assume_missing', True)  # Better type inference
                kwargs.setdefault('blocksize', '64MB')    # Optimal chunk size
                dask_df = dd.read_csv(data, **kwargs)
            
            elif isinstance(data, pd.DataFrame):
                # Convert pandas DataFrame to dask
                dask_df = dd.from_pandas(data, npartitions=n_workers)
            
            elif isinstance(data, dd.DataFrame):
                dask_df = data
            
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")

            # Apply processing function if provided
            if func:
                dask_df = dask_df.map_partitions(func)

            return dask_df

        except Exception as e:
            raise RuntimeError(f"Error processing data: {str(e)}") from e
    
    @staticmethod
    def cleanup():
        """Clean up all resources"""
        if OptimUtils.client:
            OptimUtils.client.close()
            OptimUtils.client = None
        if OptimUtils.cluster:
            OptimUtils.cluster.close()
            OptimUtils.cluster = None
        
        gc.collect()
        tf.keras.backend.clear_session()

# Initialize TensorFlow settings on import
OptimUtils.setup_tensorflow()