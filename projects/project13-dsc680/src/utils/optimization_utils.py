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
    def process_zip_data(
        zip_path: str,
        csv_filename: str = None,
        n_workers: int = 4,
        **kwargs
    ) -> dd.DataFrame:
        """Process CSV data directly from ZIP file without extraction
        
        Args:
            zip_path: Path to ZIP file
            csv_filename: Specific CSV file name in ZIP (optional)
            n_workers: Number of workers for parallel processing
            **kwargs: Additional arguments passed to dd.read_csv
            
        Returns:
            Dask DataFrame loaded from ZIP
        """
        import zipfile
        import io
        
        try:
            # Ensure client is set up
            if not OptimUtils.client:
                OptimUtils.setup_dask(n_workers=n_workers)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_file:
                # If no specific filename, get first CSV
                if csv_filename is None:
                    csv_files = [f for f in zip_file.namelist() if f.endswith('.csv')]
                    if not csv_files:
                        raise ValueError("No CSV files found in ZIP")
                    csv_filename = csv_files[0]
                
                # Read CSV directly from memory
                with zip_file.open(csv_filename) as csv_file:
                    # Load into memory-mapped file for Dask
                    csv_data = io.BytesIO(csv_file.read())
                    
                    # Use Dask to read the data
                    kwargs.setdefault('assume_missing', True)
                    kwargs.setdefault('blocksize', '64MB')
                    
                    return dd.read_csv(csv_data, **kwargs)
                    
        except Exception as e:
            raise RuntimeError(f"Error processing ZIP file {zip_path}: {str(e)}") from e

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
        
    @staticmethod
    def clear_memory():
        """
        Clear memory by forcing garbage collection and clearing TensorFlow session
        """
        print("Clearing memory...")
        # Clear Keras/TF session
        tf.keras.backend.clear_session()
        
        # Force garbage collection
        gc.collect()
        
        # Remove temporary files
        for f in Path('/tmp').glob('*dask-worker-space*'):
            try:
                if f.is_dir():
                    for sub_f in f.glob('*'):
                        try:
                            sub_f.unlink()
                        except:
                            pass
            except:
                pass

    @staticmethod
    def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage by converting to appropriate dtypes
        
        Args:
            df: Input DataFrame
            
        Returns:
            Optimized DataFrame
        """
        result = df.copy()
        
        # Process int and float columns
        int_cols = result.select_dtypes(include=['int']).columns
        for col in int_cols:
            result[col] = pd.to_numeric(result[col], downcast='integer')
            
        float_cols = result.select_dtypes(include=['float']).columns
        for col in float_cols:
            result[col] = pd.to_numeric(result[col], downcast='float')
        
        # Process object columns - convert to category if cardinality is low
        obj_cols = result.select_dtypes(include=['object']).columns
        for col in obj_cols:
            num_unique = result[col].nunique()
            if num_unique < len(result[col]) * 0.5:  # If less than 50% unique values
                result[col] = result[col].astype('category')
        
        # Report memory savings
        start_mem = df.memory_usage(deep=True).sum() / 1024**2
        end_mem = result.memory_usage(deep=True).sum() / 1024**2
        savings = 100 * (1 - end_mem/start_mem)
        
        print(f"Memory optimized: {start_mem:.2f} MB → {end_mem:.2f} MB ({savings:.1f}% reduction)")
        
        return result

# Initialize TensorFlow settings on import
OptimUtils.setup_tensorflow()

# Add standalone functions for direct import
def clear_memory():
    """Standalone function for memory cleanup"""
    OptimUtils.clear_memory()
    
def optimize_dtypes(df):
    """Standalone function for dtype optimization"""
    return OptimUtils.optimize_dtypes(df)