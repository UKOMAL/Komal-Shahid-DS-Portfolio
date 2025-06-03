#!/usr/bin/env python3
"""
Performance Optimization Module for Anamorphic Billboard Generator
Provides GPU acceleration, memory management, and rendering optimizations

Author: KShahid
Course: DSC680 - Applied Data Science
"""
import time
import psutil
import gc
from typing import Dict, List, Tuple, Optional
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    """
    Performance optimization utilities for billboard generation
    """
    
    def __init__(self):
        self.performance_metrics = {}
        self.memory_usage = []
        
    @contextmanager
    def performance_monitor(self, operation_name: str):
        """
        Context manager to monitor performance of operations
        
        Args:
            operation_name: Name of the operation being monitored
        """
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        logger.info(f"Starting operation: {operation_name}")
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            self.performance_metrics[operation_name] = {
                'duration': duration,
                'memory_start': start_memory,
                'memory_end': end_memory,
                'memory_delta': memory_delta
            }
            
            logger.info(f"Completed {operation_name}: {duration:.2f}s, Memory: {memory_delta:+.1f}MB")
    
    def optimize_blender_settings(self):
        """Optimize Blender settings for performance"""
        try:
            import bpy
            
            # Enable GPU rendering if available
            preferences = bpy.context.preferences
            cycles_preferences = preferences.addons['cycles'].preferences
            
            # Try to enable GPU compute
            cycles_preferences.refresh_devices()
            
            for device in cycles_preferences.devices:
                if device.type in ['CUDA', 'OPENCL', 'OPTIX']:
                    device.use = True
                    logger.info(f"Enabled GPU device: {device.name} ({device.type})")
            
            # Set compute device
            scene = bpy.context.scene
            if scene.cycles.device == 'CPU':
                available_devices = [d for d in cycles_preferences.devices if d.use]
                if available_devices:
                    scene.cycles.device = 'GPU'
                    logger.info("Switched to GPU rendering")
            
            # Optimize tile sizes for rendering
            if hasattr(scene.render, 'tile_x'):
                scene.render.tile_x = 256
                scene.render.tile_y = 256
                
        except ImportError:
            logger.warning("Blender not available for GPU optimization")
        except Exception as e:
            logger.error(f"GPU optimization failed: {e}")
    
    def optimize_memory_usage(self):
        """Optimize memory usage during rendering"""
        try:
            # Force garbage collection
            gc.collect()
            
            # Get current memory usage
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            self.memory_usage.append(current_memory)
            
            # Log memory usage if high
            if current_memory > 1000:  # 1GB
                logger.warning(f"High memory usage: {current_memory:.1f}MB")
                
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")
    
    def get_performance_report(self) -> Dict:
        """
        Generate a comprehensive performance report
        
        Returns:
            Dict: Performance metrics and recommendations
        """
        try:
            total_operations = len(self.performance_metrics)
            total_time = sum(metrics['duration'] for metrics in self.performance_metrics.values())
            max_memory = max(self.memory_usage) if self.memory_usage else 0
            
            # Find bottlenecks
            bottlenecks = []
            for operation, metrics in self.performance_metrics.items():
                if metrics['duration'] > 5.0:  # Operations taking more than 5 seconds
                    bottlenecks.append({
                        'operation': operation,
                        'duration': metrics['duration'],
                        'memory_delta': metrics['memory_delta']
                    })
            
            # Generate recommendations
            recommendations = []
            if max_memory > 2000:  # 2GB
                recommendations.append("Consider reducing render quality for better memory efficiency")
            if total_time > 60:  # 1 minute
                recommendations.append("Consider enabling GPU acceleration for faster rendering")
            if bottlenecks:
                recommendations.append(f"Optimize {len(bottlenecks)} slow operations")
            
            return {
                'total_operations': total_operations,
                'total_time': total_time,
                'max_memory_mb': max_memory,
                'bottlenecks': bottlenecks,
                'recommendations': recommendations,
                'detailed_metrics': self.performance_metrics
            }
            
        except Exception as e:
            logger.error(f"Performance report generation failed: {e}")
            return {'error': str(e)}

class BatchProcessor:
    """
    Batch processing utilities for multiple billboard generation
    """
    
    def __init__(self, generator, optimizer: PerformanceOptimizer):
        self.generator = generator
        self.optimizer = optimizer
        self.results = []
        
    def process_batch(self, image_paths: List[str], output_dir: str) -> List[Dict]:
        """
        Process multiple images in batch
        
        Args:
            image_paths: List of image file paths
            output_dir: Output directory for generated billboards
            
        Returns:
            List[Dict]: Results for each processed image
        """
        batch_results = []
        
        for i, image_path in enumerate(image_paths):
            try:
                with self.optimizer.performance_monitor(f"batch_item_{i}"):
                    output_path = f"{output_dir}/billboard_{i:03d}.png"
                    
                    # Generate billboard
                    self.generator.generate_billboard(image_path, output_path)
                    
                    batch_results.append({
                        'input': image_path,
                        'output': output_path,
                        'status': 'success',
                        'index': i
                    })
                    
                    # Optimize memory after each generation
                    self.optimizer.optimize_memory_usage()
                    
            except Exception as e:
                logger.error(f"Batch processing failed for {image_path}: {e}")
                batch_results.append({
                    'input': image_path,
                    'output': None,
                    'status': 'failed',
                    'error': str(e),
                    'index': i
                })
        
        return batch_results

def benchmark_system() -> Dict:
    """
    Benchmark the system for anamorphic billboard generation
    
    Returns:
        Dict: Benchmark results and system information
    """
    try:
        # System information
        cpu_count = psutil.cpu_count()
        memory_total = psutil.virtual_memory().total / 1024 / 1024 / 1024  # GB
        
        # CPU benchmark
        start_time = time.time()
        # Simple CPU intensive task
        result = sum(i * i for i in range(1000000))
        cpu_benchmark_time = time.time() - start_time
        
        # Memory benchmark
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        test_data = [i for i in range(1000000)]  # Create large list
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_benchmark = end_memory - start_memory
        del test_data  # Clean up
        
        return {
            'cpu_cores': cpu_count,
            'memory_total_gb': memory_total,
            'cpu_benchmark_time': cpu_benchmark_time,
            'memory_allocation_mb': memory_benchmark,
            'system_suitable': cpu_count >= 4 and memory_total >= 8,
            'recommendations': _get_system_recommendations(cpu_count, memory_total)
        }
        
    except Exception as e:
        logger.error(f"System benchmark failed: {e}")
        return {'error': str(e)}

def _get_system_recommendations(cpu_count: int, memory_gb: float) -> List[str]:
    """Generate system recommendations based on hardware"""
    recommendations = []
    
    if cpu_count < 4:
        recommendations.append("Consider upgrading to a CPU with more cores for better performance")
    if memory_gb < 8:
        recommendations.append("Consider upgrading RAM to at least 8GB for optimal performance")
    if cpu_count >= 8 and memory_gb >= 16:
        recommendations.append("System is well-suited for high-quality rendering")
        
    return recommendations 