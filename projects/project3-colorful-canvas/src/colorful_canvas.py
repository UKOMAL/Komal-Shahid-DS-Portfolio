#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Colorful Canvas Module - Core AI and Image Processing
Consolidates all core AI and image processing functionality for the Colorful Canvas project.

Author: Komal Shahid
Course: DSC680 - Bellevue University
Project: Colorful Canvas AI Art Studio
"""

# Core dependencies
import os
import sys
import random
import json
import time
import hashlib
import threading
import logging
import warnings
from typing import Union, Optional, Dict, List, Tuple, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Essential scientific computing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
from tqdm import tqdm

# Machine learning dependencies
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.feature_extraction import image as sk_image
import joblib

# Deep learning dependencies - No fallbacks
import torch
from transformers import pipeline
import requests

# Project-specific imports
from src.color_depth_midas import ColorDepthMiDaS

# Global constants
CACHE_DIR = Path("./data/cache")
MODEL_DIR = Path("./models")
OUTPUT_DIR = Path("./data/output")


class DependencyManager:
    """Manages dependencies and device selection for optimal performance"""
    
    def __init__(self) -> None:
        self.torch_available: bool = True  # Always true since we're not allowing fallbacks
        self.device: str = self._get_safe_device()
        self._print_status()
    
    def _get_safe_device(self) -> str:
        """Get the best available device with smart fallbacks"""
        if torch.cuda.is_available():
            print("🚀 CUDA GPU detected - using CUDA for maximum performance")
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("🍎 Apple Silicon MPS detected - using MPS for accelerated performance")
            return "mps"
        else:
            print("💻 Using CPU for PyTorch operations")
            return "cpu"
    
    def get_device(self) -> str:
        return self.device
    
    def _print_status(self) -> None:
        device_status = "CUDA GPU" if torch.cuda.is_available() else \
                      "Apple MPS" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else \
                      "CPU"
        print(f"🔧 System Configuration:")
        print(f"   PyTorch: ✅ (Device: {self.device})")
        print(f"   Acceleration: {device_status}")
        print(f"   Performance Mode: {'High' if self.device != 'cpu' else 'Standard'}")


# Initialize dependency manager
deps = DependencyManager()


class ColorfulCanvasAI:
    """
    Main AI class for creating 3D visual illusions and anamorphic effects
    Optimized for various hardware configurations with intelligent fallbacks
    """
    
    def __init__(self):
        """Initialize the AI system with optimal configuration"""
        self.device = deps.get_device()
        self.depth_estimator = None
        self.illusion_predictor = None
        self.performance_predictor = None
        
        # Initialize the ColorDepthMiDaS model
        self.color_depth_midas = ColorDepthMiDaS(device=self.device)
        
        # Create output directories
        for directory in [CACHE_DIR, MODEL_DIR, OUTPUT_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ ColorfulCanvasAI initialized with {self.device.upper()} acceleration")
    
    def load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """Load image from file path with comprehensive validation"""
        if isinstance(image_path, str):
            image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            image = Image.open(image_path).convert('RGB')
            print(f"✅ Image loaded: {image_path} ({image.size[0]}x{image.size[1]})")
            return image
        except Exception as e:
            raise ValueError(f"Failed to load image {image_path}: {e}")
    
    def save_image(self, image: Image.Image, output_path: Union[str, Path]) -> None:
        """Save image to file path with comprehensive validation"""
        if isinstance(output_path, str):
            output_path = Path(output_path)
        
        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            image.save(output_path)
            print(f"✅ Image saved: {output_path}")
        except Exception as e:
            raise ValueError(f"Failed to save image to {output_path}: {e}")
    
    def preprocess_image(self, image: Image.Image, target_size: Tuple[int, int] = (512, 512)) -> Image.Image:
        """Preprocess image for AI processing with quality preservation"""
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL Image, got {type(image)}")
        
        # Use LANCZOS for high-quality resizing
        return image.resize(target_size, Image.Resampling.LANCZOS)
    
    def generate_depth_map(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """
        Generate high-quality depth map using ColorDepthMiDaS
        Supports CUDA, MPS, and CPU execution
        """
        print("🏔️ Generating depth map...")
        print(f"   Using ColorDepthMiDaS on {self.device.upper()} (high quality)...")
        
        # Convert input to PIL Image if it's a numpy array
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Use our new ColorDepthMiDaS class to generate the depth map
        depth_map = self.color_depth_midas.generate_depth_map(image)
        
        print(f"✅ High-quality depth map generated using PyTorch {self.device.upper()}")
        return depth_map
    
    def generate_color_depth_map(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """
        Generate colorized depth map using ColorDepthMiDaS
        Produces a more visually appealing color representation of depth
        """
        print("🎨 Generating colorized depth map...")
        
        # Convert input to PIL Image if it's a numpy array
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Use our new ColorDepthMiDaS class to generate the colorized depth map
        color_depth = self.color_depth_midas.generate_colormap_depth(image)
        
        print(f"✅ Colorized depth map generated using PyTorch {self.device.upper()}")
        return color_depth
    
    def create_shadow_box_effect(self, image: Image.Image, depth_map: Image.Image, 
                                strength: float = 1.5, viewing_angle: float = 15) -> Image.Image:
        """
        Create true anamorphic shadow box effect with extreme distortion
        This should look completely warped from normal viewing angles
        """
        print("📦 Creating TRUE anamorphic shadow box illusion...")
        
        # Convert to numpy arrays
        img_array = np.array(image).astype(np.float32)
        depth_array = np.array(depth_map.convert('L')).astype(np.float32) / 255.0
        
        height, width = img_array.shape[:2]
        
        # Create massive canvas for extreme anamorphic stretching
        canvas_height = int(height * 12.0)  # Extreme vertical stretch
        canvas_width = int(width * 10.0)   # Extreme horizontal stretch
        result_img = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
        
        # Fill with dark surface color for better contrast
        result_img[:] = [25, 25, 35]
        
        # Real anamorphic mathematics based on viewing angle projection
        viewing_angle_rad = np.radians(viewing_angle)
        
        for i in range(height):
            for j in range(width):
                depth_factor = depth_array[i, j]
                
                if depth_factor < 0.05:  # Skip deep background
                    continue
                
                # Original position relative to center
                rel_x = (j - width/2) / width   # -0.5 to 0.5
                rel_y = (i - height/2) / height # -0.5 to 0.5
                
                # Extreme anamorphic projection calculation
                # Keystone projection for viewing angle
                keystone_factor = 1.0 / np.cos(viewing_angle_rad * (1 + depth_factor * 4))
                
                # Extreme perspective stretching
                perspective_stretch = 1.0 + (depth_factor * strength * 25.0)
                
                # Anamorphic Y-axis distortion
                anamorphic_y = rel_y * keystone_factor * perspective_stretch
                
                # X-axis perspective shift
                anamorphic_x = rel_x * (1.0 + depth_factor * strength * 8.0)
                
                # Apply viewing angle compensation
                angle_compensation = np.tan(viewing_angle_rad * depth_factor * 4.0)
                final_y = anamorphic_y + angle_compensation * strength * 10.0
                
                # Map to canvas coordinates
                canvas_x = int(canvas_width/2 + anamorphic_x * canvas_width * 0.9)
                canvas_y = int(canvas_height/2 + final_y * canvas_height * 0.95)
                
                # Bounds checking
                if 0 <= canvas_x < canvas_width and 0 <= canvas_y < canvas_height:
                    # Enhanced brightness for "floating" effect
                    brightness_factor = 1.0 + (depth_factor * 1.5)
                    enhanced_pixel = np.clip(img_array[i, j] * brightness_factor, 0, 255)
                    
                    # Fill larger area to prevent holes
                    fill_size = max(2, int(depth_factor * 15))
                    for dx in range(-fill_size, fill_size + 1):
                        for dy in range(-fill_size, fill_size + 1):
                            px, py = canvas_x + dx, canvas_y + dy
                            if 0 <= px < canvas_width and 0 <= py < canvas_height:
                                fade = 1.0 - (abs(dx) + abs(dy)) / (fill_size * 2 + 1)
                                result_img[py, px] = enhanced_pixel * fade
        
        # Convert back to PIL Image
        result_img = np.clip(result_img, 0, 255).astype(np.uint8)
        
        print("✅ Extreme anamorphic shadow box effect created")
        print(f"   Canvas size: {canvas_width}x{canvas_height} (extreme distortion)")
        
        return Image.fromarray(result_img)
    
    def create_screen_pop_effect(self, image: Image.Image, depth_map: Image.Image, 
                                strength: float = 1.5, chromatic: bool = True) -> Image.Image:
        """
        Create screen pop-out effect for digital displays
        Objects appear to pop out of the screen towards the viewer
        """
        print("🖥️ Creating screen pop-out effect...")
        
        img_array = np.array(image).astype(np.float32)
        depth_array = np.array(depth_map.convert('L')).astype(np.float32) / 255.0
        
        height, width = img_array.shape[:2]
        
        # Create larger canvas for pop-out effect
        canvas_height = int(height * 1.5)
        canvas_width = int(width * 1.5)
        result_img = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
        
        # Black background for screen effect
        result_img[:] = [0, 0, 0]
        
        for i in range(height):
            for j in range(width):
                depth_factor = depth_array[i, j]
                
                # Calculate pop-out displacement
                pop_out_x = int((depth_factor - 0.5) * strength * 50)
                pop_out_y = int((depth_factor - 0.5) * strength * 30)
                
                # Calculate screen position
                screen_x = int(canvas_width/2 - width/2 + j + pop_out_x)
                screen_y = int(canvas_height/2 - height/2 + i + pop_out_y)
                
                if 0 <= screen_x < canvas_width and 0 <= screen_y < canvas_height:
                    pixel = img_array[i, j]
                    
                    # Add chromatic aberration for depth
                    if chromatic and depth_factor > 0.6:
                        # Separate RGB channels slightly
                        result_img[screen_y, screen_x, 0] = pixel[0] * 1.1  # Red shift
                        result_img[screen_y, screen_x, 1] = pixel[1]        # Green normal
                        result_img[screen_y, screen_x, 2] = pixel[2] * 0.9  # Blue shift
                    else:
                        result_img[screen_y, screen_x] = pixel
        
        result_img = np.clip(result_img, 0, 255).astype(np.uint8)
        
        print("✅ Screen pop-out effect created")
        return Image.fromarray(result_img)
    
    def create_seoul_optimized_test_image(self, width: int = 800, height: int = 600) -> Image.Image:
        """
        Create optimized test image for Seoul-style LED display effects
        Designed specifically for curved LED screen anamorphic displays
        """
        print(f"🎨 Creating Seoul-optimized test image ({width}x{height})...")
        
        # Create base image
        img = Image.new('RGB', (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Seoul-inspired color palette
        seoul_colors = [
            (255, 0, 100),    # Hot pink
            (0, 255, 200),    # Cyan
            (255, 150, 0),    # Orange
            (100, 0, 255),    # Purple
            (255, 255, 0),    # Yellow
            (0, 255, 100),    # Green
        ]
        
        # Create geometric patterns optimal for LED displays
        center_x, center_y = width // 2, height // 2
        
        # Draw concentric circles
        for i in range(5):
            radius = 50 + i * 40
            color = seoul_colors[i % len(seoul_colors)]
            draw.ellipse([center_x - radius, center_y - radius, 
                         center_x + radius, center_y + radius], 
                        outline=color, width=8)
        
        # Add floating elements
        for i in range(8):
            angle = i * 45
            x = center_x + int(150 * np.cos(np.radians(angle)))
            y = center_y + int(150 * np.sin(np.radians(angle)))
            
            color = seoul_colors[i % len(seoul_colors)]
            draw.rectangle([x-20, y-20, x+20, y+20], fill=color)
        
        # Add central logo area
        draw.rectangle([center_x-50, center_y-30, center_x+50, center_y+30], 
                      fill=(255, 255, 255), outline=(200, 200, 200), width=3)
        
        print("✅ Seoul-optimized test image created")
        return img
    
    def analyze_for_seoul_effect(self, image: Image.Image) -> Dict[str, Any]:
        """
        Analyze image for optimal Seoul LED display parameters
        Returns recommended settings for LED display anamorphic effects
        """
        print("🔍 Analyzing image for Seoul LED display optimization...")
        
        img_array = np.array(image)
        
        # Analyze color distribution
        avg_brightness = np.mean(img_array)
        color_variance = np.var(img_array, axis=(0, 1))
        
        # Analyze contrast
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        contrast = gray.std()
        
        # Edge density for detail level
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Generate recommendations
        recommendations = {
            "brightness": {
                "value": avg_brightness,
                "recommendation": "increase" if avg_brightness < 100 else "optimal" if avg_brightness < 180 else "decrease"
            },
            "contrast": {
                "value": contrast,
                "recommendation": "increase" if contrast < 40 else "optimal" if contrast < 80 else "reduce"
            },
            "detail_level": {
                "value": edge_density,
                "recommendation": "high_detail" if edge_density > 0.1 else "medium_detail" if edge_density > 0.05 else "low_detail"
            },
            "suggested_led_intensity": min(1.0, avg_brightness / 255.0 + 0.3),
            "suggested_viewing_distance": "2-4 meters" if edge_density > 0.1 else "3-6 meters",
            "optimal_led_spacing": "fine" if edge_density > 0.1 else "standard"
        }
        
        print("✅ Seoul display analysis complete")
        return recommendations
    
    def create_seoul_corner_projection(self, image: Image.Image, depth_map: Optional[Image.Image] = None, 
                                     corner_position: str = 'left') -> Image.Image:
        """
        Create Seoul-style corner LED display projection
        Optimized for curved LED panels in corner installations
        """
        print(f"🏙️ Creating Seoul corner projection ({corner_position} corner)...")
        
        if depth_map is None:
            depth_map = self.generate_depth_map(image)
        
        img_array = np.array(image).astype(np.float32)
        depth_array = np.array(depth_map.convert('L')).astype(np.float32) / 255.0
        
        height, width = img_array.shape[:2]
        
        # Create dual-panel setup for corner display
        panel_width = width
        panel_height = height
        
        # Create left and right panels
        left_panel = np.zeros((panel_height, panel_width, 3), dtype=np.float32)
        right_panel = np.zeros((panel_height, panel_width, 3), dtype=np.float32)
        
        for i in range(height):
            for j in range(width):
                depth_factor = depth_array[i, j]
                pixel = img_array[i, j]
                
                # Calculate LED intensity based on depth
                led_intensity = self._calculate_led_intensity(depth_factor, i, 1.5)
                
                # Apply Seoul color grading
                enhanced_pixel = self._apply_seoul_color_grading(pixel, led_intensity)
                
                # Place pixel on appropriate panel(s)
                if corner_position == 'left':
                    # Left corner: more content on left panel
                    self._place_seoul_pixel_on_panel(left_panel, j, i, enhanced_pixel * 1.2, 
                                                   depth_factor, panel_width, panel_height)
                    if j > width * 0.6:  # Overlap area
                        self._place_seoul_pixel_on_panel(right_panel, j - int(width * 0.6), i, 
                                                       enhanced_pixel * 0.8, depth_factor, 
                                                       panel_width, panel_height)
                else:  # right corner
                    # Right corner: more content on right panel
                    self._place_seoul_pixel_on_panel(right_panel, j, i, enhanced_pixel * 1.2, 
                                                   depth_factor, panel_width, panel_height)
                    if j < width * 0.4:  # Overlap area
                        self._place_seoul_pixel_on_panel(left_panel, j + int(width * 0.6), i, 
                                                       enhanced_pixel * 0.8, depth_factor, 
                                                       panel_width, panel_height)
        
        # Apply Seoul post-processing to each panel
        left_panel = self._apply_seoul_post_processing(left_panel)
        right_panel = self._apply_seoul_post_processing(right_panel)
        
        # Combine panels for visualization
        combined = self._create_seoul_combined_view(left_panel, right_panel)
        
        result_img = np.clip(combined, 0, 255).astype(np.uint8)
        
        print("✅ Seoul corner projection created")
        return Image.fromarray(result_img)
    
    def _calculate_led_intensity(self, depth_value: float, world_z: float, strength: float) -> float:
        """Calculate LED intensity based on depth and position"""
        base_intensity = 0.3 + (depth_value * 0.7)
        distance_factor = 1.0 - (world_z / 1000.0)  # Assume max distance of 1000
        return min(1.0, base_intensity * distance_factor * strength)
    
    def _apply_seoul_color_grading(self, original_color: np.ndarray, led_intensity: float) -> np.ndarray:
        """Apply Seoul-style color grading for LED displays"""
        # Enhance saturation and brightness for LED effect
        enhanced = original_color * led_intensity
        
        # Add slight color temperature shift for LED look
        enhanced[0] = min(255, enhanced[0] * 1.1)  # Slight red boost
        enhanced[2] = min(255, enhanced[2] * 0.95)  # Slight blue reduction
        
        return enhanced
    
    def _place_seoul_pixel_on_panel(self, panel: np.ndarray, x: int, y: int, 
                                   color: np.ndarray, depth: float, 
                                   panel_width: int, panel_height: int) -> None:
        """Place pixel on Seoul LED panel with appropriate spreading"""
        if 0 <= x < panel_width and 0 <= y < panel_height:
            panel[y, x] = color
            
            # Add LED bloom effect for bright pixels
            if np.mean(color) > 150:
                bloom_size = int(2 + depth * 3)
                for dx in range(-bloom_size, bloom_size + 1):
                    for dy in range(-bloom_size, bloom_size + 1):
                        px, py = x + dx, y + dy
                        if 0 <= px < panel_width and 0 <= py < panel_height:
                            distance = np.sqrt(dx*dx + dy*dy)
                            if distance <= bloom_size:
                                fade = 1.0 - (distance / bloom_size)
                                panel[py, px] = np.maximum(panel[py, px], color * fade * 0.3)
    
    def _apply_seoul_post_processing(self, panel: np.ndarray) -> np.ndarray:
        """Apply Seoul-specific post-processing effects"""
        # Convert to uint8 for processing
        panel_uint8 = np.clip(panel, 0, 255).astype(np.uint8)
        
        # Slight Gaussian blur for LED smoothing
        blurred = cv2.GaussianBlur(panel_uint8, (3, 3), 0.5)
        
        # Enhance contrast slightly
        enhanced = cv2.convertScaleAbs(blurred, alpha=1.1, beta=5)
        
        return enhanced.astype(np.float32)
    
    def _create_seoul_combined_view(self, left_panel: np.ndarray, right_panel: np.ndarray) -> np.ndarray:
        """Create combined view of Seoul corner display panels"""
        panel_height, panel_width = left_panel.shape[:2]
        
        # Create side-by-side visualization
        combined_width = panel_width * 2 + 50  # Gap between panels
        combined = np.zeros((panel_height, combined_width, 3), dtype=np.float32)
        
        # Place left panel
        combined[:, :panel_width] = left_panel
        
        # Place right panel with gap
        combined[:, panel_width + 50:] = right_panel
        
        # Add separator line
        combined[:, panel_width + 20:panel_width + 30] = [100, 100, 100]
        
        return combined
    
    def generate_illusion(self, input_path: Union[str, Path], effect_type: str = "shadow_box", 
                         strength: float = 1.5, save_path: Optional[Union[str, Path]] = None) -> Optional[Image.Image]:
        """
        Generate anamorphic illusion with specified effect type
        Comprehensive method supporting multiple illusion types
        """
        print(f"🎭 Generating {effect_type} illusion (strength: {strength})...")
        
        try:
            # Load and preprocess image
            image = self.load_image(input_path)
            processed_image = self.preprocess_image(image)
            
            # Generate depth map
            depth_map = self.generate_depth_map(processed_image)
            
            # Apply specified effect
            if effect_type == "shadow_box":
                result = self.create_shadow_box_effect(processed_image, depth_map, strength)
            elif effect_type == "screen_pop":
                result = self.create_screen_pop_effect(processed_image, depth_map, strength)
            elif effect_type == "seoul_corner":
                result = self.create_seoul_corner_projection(processed_image, depth_map)
            else:
                print(f"❌ Unknown effect type: {effect_type}")
                return None
            
            # Save if path provided
            if save_path:
                self.save_image(result, save_path)
            
            print(f"✅ {effect_type} illusion generated successfully")
            return result
            
        except Exception as e:
            print(f"❌ Failed to generate {effect_type} illusion: {e}")
            return None


class DeviceManager:
    """Utility class for device management"""
    
    @staticmethod
    def get_device() -> str:
        """Get optimal device for processing"""
        return deps.get_device()


def main():
    """Main function for testing colorful_canvas module"""
    print("🎨 Colorful Canvas AI - Core Module")
    print("=" * 50)
    
    # Initialize AI system
    ai = ColorfulCanvasAI()
    
    # Create test image
    test_image = ai.create_seoul_optimized_test_image(400, 300)
    ai.save_image(test_image, "./data/output/test_image.png")
    
    # Generate depth map
    depth_map = ai.generate_depth_map(test_image)
    ai.save_image(depth_map, "./data/output/test_depth.png")
    
    # Create shadow box effect
    shadow_effect = ai.create_shadow_box_effect(test_image, depth_map, strength=1.5)
    ai.save_image(shadow_effect, "./data/output/test_shadow_box.png")
    
    # Analyze for Seoul effect
    analysis = ai.analyze_for_seoul_effect(test_image)
    print(f"📊 Seoul Analysis Results: {analysis}")
    
    print("✅ Core module testing complete!")


if __name__ == "__main__":
    main() 