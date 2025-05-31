"""
Image utility functions for the Colorful Canvas project.
"""
import os
import numpy as np
from PIL import Image
import cv2

def load_image(image_path):
    """
    Load an image from a file path.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        PIL.Image: Loaded image
    """
    try:
        image = Image.open(image_path)
        return image
    except Exception as e:
        print(f"Error loading image '{image_path}': {e}")
        return None

def save_image(image, output_path):
    """
    Save an image to a file.
    
    Args:
        image (PIL.Image or numpy.ndarray): Image to save
        output_path (str): Path to save the image to
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(
                np.clip(image, 0, 255).astype(np.uint8)
            )
        
        # Save the image
        image.save(output_path)
        print(f"Image saved to {output_path}")
    except Exception as e:
        print(f"Error saving image to '{output_path}': {e}")

def resize_image(image, max_size=1024):
    """
    Resize an image while maintaining aspect ratio.
    
    Args:
        image (PIL.Image): Image to resize
        max_size (int): Maximum size of the longest dimension
        
    Returns:
        PIL.Image: Resized image
    """
    # Get current size
    width, height = image.size
    
    # If image is already smaller than max_size, return as is
    if width <= max_size and height <= max_size:
        return image
    
    # Calculate new size
    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))
    
    # Resize image
    return image.resize((new_width, new_height), Image.LANCZOS)

def normalize_depth_map(depth_map):
    """
    Normalize a depth map to 0-1 range.
    
    Args:
        depth_map (numpy.ndarray): Depth map to normalize
        
    Returns:
        numpy.ndarray: Normalized depth map
    """
    depth_map = depth_map.astype(np.float32)
    min_val = depth_map.min()
    max_val = depth_map.max()
    
    # Avoid division by zero
    if max_val - min_val < 1e-8:
        return np.zeros_like(depth_map)
    
    # Normalize to 0-1 range
    normalized = (depth_map - min_val) / (max_val - min_val)
    
    return normalized 