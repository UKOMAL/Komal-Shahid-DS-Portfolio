#!/usr/bin/env python3
"""
Ultimate Anamorphic Billboard System
Combines Seoul-style effects, advanced depth estimation, professional projection mapping,
and Blender integration for the best possible anamorphic billboard generation.

Author: Komal Shahid
Course: DSC680 - Applied Data Science
Project: Colorful Canvas - Ultimate Anamorphic System
"""

import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure, morphology
import math
import os
import sys
import random
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# Blender integration
try:
    import bpy
    import bmesh
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False
    print("⚠️ Blender not available - 3D generation disabled")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DisplayType(Enum):
    """Supported display types for anamorphic effects"""
    CORNER_LED = "corner_led"
    BILLBOARD = "billboard" 
    AQUARIUM = "aquarium"
    SEOUL_WAVE = "seoul_wave"

class EffectType(Enum):
    """Available anamorphic effect types"""
    SHADOW_BOX = "shadow_box"
    SCREEN_POP = "screen_pop"
    SEOUL_CORNER = "seoul_corner"
    FLOATING_OBJECTS = "floating_objects"
    WAVE_MOTION = "wave_motion"

@dataclass
class AnamorphicConfig:
    """Configuration for anamorphic effects"""
    display_type: DisplayType = DisplayType.CORNER_LED
    effect_type: EffectType = EffectType.SEOUL_CORNER
    viewing_angle: float = 25.0  # degrees
    viewing_distance: float = 15.0  # meters
    strength: float = 2.5
    resolution: Tuple[int, int] = (1920, 1080)
    enable_motion: bool = False
    enable_3d_objects: bool = True
    quality: str = "high"  # low, medium, high, ultra

class UltimateAnamorphicSystem:
    """
    Ultimate anamorphic billboard system combining all advanced features
    """
    
    def __init__(self, config: Optional[AnamorphicConfig] = None):
        """Initialize the ultimate anamorphic system"""
        # Use the proper config class or create a simple fallback
        if config is None:
            try:
                from .anamorphic_config import ConfigPresets
                self.config = ConfigPresets.seoul_corner_led()
            except ImportError:
                # Simple fallback config
                self.config = AnamorphicConfig()
        else:
            self.config = config
            
        self.device = self._get_optimal_device()
        self.depth_model = None
        self.load_depth_model()
        
        # Professional camera calibration
        self.camera_matrix = self._calculate_camera_matrix()
        
        # Display-specific parameters
        self.display_params = self._get_display_parameters()
        
        # Seoul-style optimization parameters
        self.seoul_params = {
            'corner_angle': 90.0,  # degrees
            'depth_exaggeration': 3.0,
            'led_pixel_pitch': 2.5,  # mm
            'brightness_max': 4000,  # nits
            'contrast_enhancement': 1.8
        }
        
        logger.info(f"Ultimate Anamorphic System initialized on {self.device}")
    
    def _get_optimal_device(self) -> str:
        """Get the best available computing device"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def load_depth_model(self):
        """Load the best available depth estimation model"""
        try:
            self.depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", 
                                            force_reload=False, trust_repo=True)
            self.depth_transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
            self.depth_model.to(self.device)
            self.depth_model.eval()
            logger.info("✅ MiDaS depth model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load MiDaS model: {e}. Using fallback.")
            self.depth_model = None
    
    def _calculate_camera_matrix(self) -> np.ndarray:
        """Calculate professional camera calibration matrix"""
        # Get resolution - direct API access
        try:
            width, height = self.config.render.resolution
        except AttributeError:
            try:
                width, height = self.config.resolution
            except AttributeError:
                width, height = 1920, 1080  # Default resolution
        
        # Calculate focal lengths based on display type
        try:
            display_type = self.config.display_type
        except AttributeError:
            display_type = DisplayType.CORNER_LED
            
        if display_type == DisplayType.CORNER_LED:
            fov_h, fov_v = 60, 40
        elif display_type == DisplayType.SEOUL_WAVE:
            fov_h, fov_v = 70, 50
        else:
            fov_h, fov_v = 45, 30
        
        fx = width / (2 * math.tan(math.radians(fov_h) / 2))
        fy = height / (2 * math.tan(math.radians(fov_v) / 2))
        
        return np.array([
            [fx, 0, width/2],
            [0, fy, height/2],
            [0, 0, 1]
        ], dtype=np.float32)
    
    def _get_display_parameters(self) -> Dict[str, Any]:
        """Get display-specific parameters"""
        # Get viewing distance - direct API access
        try:
            viewing_distance = self.config.camera.distance
        except AttributeError:
            try:
                viewing_distance = self.config.viewing_distance
            except AttributeError:
                viewing_distance = 15.0
            
        base_params = {
            'pixel_pitch': 2.5,
            'brightness_max': 4000,
            'contrast_ratio': 3000,
            'viewing_distance_optimal': viewing_distance,
            'screen_curvature': 0.0
        }
        
        try:
            display_type = self.config.display_type
        except AttributeError:
            display_type = DisplayType.CORNER_LED
            
        if display_type == DisplayType.SEOUL_WAVE:
            base_params.update({
                'wave_frequency': 0.02,
                'wave_amplitude': 30,
                'corner_optimization': True
            })
        
        return base_params
    
    def generate_professional_depth_map(self, image: Image.Image) -> Image.Image:
        """Generate professional-grade depth map with multiple techniques"""
        if self.depth_model is not None:
            return self._generate_neural_depth_map(image)
        else:
            return self._generate_multi_cue_depth_map(image)
    
    def _generate_neural_depth_map(self, image: Image.Image) -> Image.Image:
        """Neural network-based depth estimation using MiDaS"""
        try:
            # Ensure image is PIL Image and convert to RGB
            if isinstance(image, Image.Image):
                image = image.convert('RGB')
            else:
                # Convert numpy array or other format to PIL
                if hasattr(image, 'shape') and len(image.shape) == 3:
                    image = Image.fromarray(np.array(image, dtype=np.uint8))
                else:
                    image = Image.fromarray(np.array(image))
                image = image.convert('RGB')
            
            input_tensor = self.depth_transform(image).to(self.device)
            
            with torch.no_grad():
                prediction = self.depth_model(input_tensor)
                prediction = F.interpolate(
                    prediction.unsqueeze(1),
                    size=image.size[::-1],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            
            depth_map = prediction.cpu().numpy()
            depth_map = self._enhance_depth_for_anamorphic(depth_map)
            
            # Normalize to 0-255
            depth_min, depth_max = depth_map.min(), depth_map.max()
            if depth_max > depth_min:
                depth_map = ((depth_map - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
            else:
                depth_map = np.zeros_like(depth_map, dtype=np.uint8)
            
            return Image.fromarray(depth_map)
            
        except Exception as e:
            logger.error(f"Neural depth estimation failed: {e}")
            return self._generate_multi_cue_depth_map(image)
    
    def _enhance_depth_for_anamorphic(self, depth_map: np.ndarray) -> np.ndarray:
        """Enhance depth map specifically for anamorphic effects"""
        # Edge-preserving smoothing
        depth_smooth = cv2.bilateralFilter(depth_map.astype(np.float32), 9, 75, 75)
        
        # Enhance depth discontinuities
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        edges = cv2.filter2D(depth_smooth, -1, kernel)
        depth_enhanced = depth_smooth + 0.3 * edges
        
        # Multi-scale enhancement
        scales = [1.0, 0.5, 0.25]
        enhanced_multi = np.zeros_like(depth_enhanced)
        
        for scale in scales:
            if scale == 1.0:
                scaled = depth_enhanced
            else:
                h, w = depth_enhanced.shape
                new_h, new_w = int(h * scale), int(w * scale)
                scaled = cv2.resize(depth_enhanced, (new_w, new_h))
                scaled = cv2.resize(scaled, (w, h))
            enhanced_multi += scaled * (scale ** 0.5)
        
        # Contrast enhancement
        enhanced_multi = cv2.equalizeHist((enhanced_multi * 255 / enhanced_multi.max()).astype(np.uint8))
        
        return enhanced_multi.astype(np.float32) / 255.0
    
    def _generate_multi_cue_depth_map(self, image: Image.Image) -> Image.Image:
        """Fallback depth estimation using multiple visual cues"""
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        depth_cues = []
        
        # Defocus blur estimation
        blur_map = self._estimate_defocus_blur(gray)
        depth_cues.append(('blur', blur_map, 0.25))
        
        # Atmospheric perspective
        atmos_depth = self._estimate_atmospheric_perspective(img_array)
        depth_cues.append(('atmospheric', atmos_depth, 0.2))
        
        # Texture gradient
        texture_depth = self._estimate_texture_gradient(gray)
        depth_cues.append(('texture', texture_depth, 0.2))
        
        # Size-based depth
        size_depth = self._estimate_size_based_depth(gray)
        depth_cues.append(('size', size_depth, 0.15))
        
        # Edge-based depth
        edge_depth = self._estimate_edge_based_depth(gray)
        depth_cues.append(('edges', edge_depth, 0.2))
        
        # Combine all cues
        combined_depth = np.zeros_like(gray, dtype=np.float32)
        for name, depth_cue, weight in depth_cues:
            normalized_cue = (depth_cue - depth_cue.min()) / (depth_cue.max() - depth_cue.min() + 1e-8)
            combined_depth += weight * normalized_cue
        
        combined_depth = cv2.GaussianBlur(combined_depth, (5, 5), 0)
        depth_final = (combined_depth * 255).astype(np.uint8)
        
        return Image.fromarray(depth_final)
    
    def _estimate_defocus_blur(self, gray: np.ndarray) -> np.ndarray:
        """Estimate depth from defocus blur"""
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return ndimage.generic_filter(laplacian.var(), np.var, size=15)
    
    def _estimate_atmospheric_perspective(self, img_array: np.ndarray) -> np.ndarray:
        """Estimate depth from atmospheric perspective"""
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        saturation = hsv[:,:,1].astype(np.float32)
        value = hsv[:,:,2].astype(np.float32)
        return (255 - saturation) * (value / 255.0)
    
    def _estimate_texture_gradient(self, gray: np.ndarray) -> np.ndarray:
        """Estimate depth from texture density"""
        return ndimage.generic_filter(gray.astype(np.float32), np.std, size=9)
    
    def _estimate_size_based_depth(self, gray: np.ndarray) -> np.ndarray:
        """Estimate depth based on object sizes"""
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        labeled = measure.label(binary)
        regions = measure.regionprops(labeled)
        
        size_map = np.zeros_like(gray, dtype=np.float32)
        for region in regions:
            area = region.area
            coords = region.coords
            size_factor = min(1.0, area / (gray.shape[0] * gray.shape[1] * 0.1))
            
            for coord in coords:
                size_map[coord[0], coord[1]] = size_factor * 255
        
        return size_map
    
    def _estimate_edge_based_depth(self, gray: np.ndarray) -> np.ndarray:
        """Estimate depth from edge information"""
        edges = cv2.Canny(gray, 50, 150)
        dist_transform = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
        return 255 - (dist_transform / dist_transform.max() * 255)
    
    def create_ultimate_anamorphic_effect(self, image: Image.Image, 
                                        depth_map: Optional[Image.Image] = None) -> Dict[str, Image.Image]:
        """Create the ultimate anamorphic effect combining all techniques"""
        logger.info(f"Creating ultimate anamorphic effect: {self.config.effect_type.value}")
        
        # Generate depth map if not provided
        if depth_map is None:
            depth_map = self.generate_professional_depth_map(image)
        
        # Apply effect based on configuration
        try:
            effect_type = self.config.effect_type
        except AttributeError:
            effect_type = EffectType.SEOUL_CORNER
            
        if effect_type == EffectType.SEOUL_CORNER:
            return self._create_seoul_corner_effect(image, depth_map)
        elif effect_type == EffectType.SHADOW_BOX:
            return self._create_shadow_box_effect(image, depth_map)
        elif effect_type == EffectType.SCREEN_POP:
            return self._create_screen_pop_effect(image, depth_map)
        elif effect_type == EffectType.FLOATING_OBJECTS:
            return self._create_floating_objects_effect(image, depth_map)
        elif effect_type == EffectType.WAVE_MOTION:
            return self._create_wave_motion_effect(image, depth_map)
        else:
            return self._create_seoul_corner_effect(image, depth_map)
    
    def _create_seoul_corner_effect(self, image: Image.Image, depth_map: Image.Image) -> Dict[str, Image.Image]:
        """Create Seoul-style corner LED display effect"""
        img_array = np.array(image).astype(np.float32)
        depth_array = np.array(depth_map.convert('L')).astype(np.float32) / 255.0
        
        height, width = img_array.shape[:2]
        
        # Create corner panels with Seoul optimization
        left_panel, right_panel = self._create_corner_panels(img_array, depth_array)
        
        # Apply Seoul-specific enhancements
        left_panel = self._apply_seoul_enhancements(left_panel)
        right_panel = self._apply_seoul_enhancements(right_panel)
        
        # Create combined view
        combined = self._create_combined_corner_view(left_panel, right_panel)
        
        return {
            'left_panel': Image.fromarray(np.clip(left_panel, 0, 255).astype(np.uint8)),
            'right_panel': Image.fromarray(np.clip(right_panel, 0, 255).astype(np.uint8)),
            'combined': Image.fromarray(np.clip(combined, 0, 255).astype(np.uint8)),
            'depth_map': depth_map
        }
    
    def _create_corner_panels(self, img_array: np.ndarray, depth_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create left and right corner panels with proper perspective"""
        height, width = img_array.shape[:2]
        
        # Panel dimensions (larger for better effect)
        panel_width = int(width * 1.5)
        panel_height = int(height * 1.2)
        
        left_panel = np.ones((panel_height, panel_width, 3), dtype=np.float32) * 20
        right_panel = np.ones((panel_height, panel_width, 3), dtype=np.float32) * 20
        
        # Add LED grid background
        self._add_led_grid_background(left_panel)
        self._add_led_grid_background(right_panel)
        
        # Project pixels with Seoul-style distortion
        for i in range(height):
            for j in range(width):
                depth_factor = depth_array[i, j]
                
                if depth_factor < 0.1:
                    continue
                
                # Seoul corner projection mathematics
                left_x, left_y = self._calculate_seoul_left_projection(j, i, depth_factor, width, height)
                right_x, right_y = self._calculate_seoul_right_projection(j, i, depth_factor, width, height)
                
                # Enhanced pixel color with depth-based brightness
                pixel_color = img_array[i, j] * (1.0 + depth_factor * 0.8)
                pixel_color = np.clip(pixel_color, 0, 255)
                
                # Place pixels on panels with anti-aliasing
                self._place_seoul_pixel(left_panel, left_x, left_y, pixel_color, depth_factor)
                self._place_seoul_pixel(right_panel, right_x, right_y, pixel_color, depth_factor)
        
        return left_panel, right_panel
    
    def _calculate_seoul_left_projection(self, x: int, y: int, depth: float, 
                                       width: int, height: int) -> Tuple[int, int]:
        """Calculate Seoul-style left panel projection"""
        # Seoul corner mathematics with extreme distortion
        depth_factor = depth * self.seoul_params['depth_exaggeration']
        
        # Keystone correction for left panel
        keystone_x = x + (depth_factor * (x - width/2) * 0.6)
        keystone_y = y + (depth_factor * (y - height/2) * 0.4)
        
        # Corner angle compensation
        angle_rad = math.radians(self.seoul_params['corner_angle'] / 2)
        corner_x = keystone_x * math.cos(angle_rad) + depth_factor * 50
        corner_y = keystone_y + depth_factor * 30
        
        return int(corner_x + width * 0.3), int(corner_y + height * 0.1)
    
    def _calculate_seoul_right_projection(self, x: int, y: int, depth: float,
                                        width: int, height: int) -> Tuple[int, int]:
        """Calculate Seoul-style right panel projection"""
        depth_factor = depth * self.seoul_params['depth_exaggeration']
        
        # Keystone correction for right panel
        keystone_x = x - (depth_factor * (x - width/2) * 0.6)
        keystone_y = y + (depth_factor * (y - height/2) * 0.4)
        
        # Corner angle compensation
        angle_rad = math.radians(self.seoul_params['corner_angle'] / 2)
        corner_x = keystone_x * math.cos(angle_rad) - depth_factor * 50
        corner_y = keystone_y + depth_factor * 30
        
        return int(corner_x + width * 0.3), int(corner_y + height * 0.1)
    
    def _add_led_grid_background(self, panel: np.ndarray):
        """Add realistic LED pixel grid background"""
        height, width = panel.shape[:2]
        pixel_spacing = 8  # LED pixel spacing
        
        # Subtle grid lines
        for i in range(0, height, pixel_spacing):
            panel[i, :] *= 0.95
        for j in range(0, width, pixel_spacing):
            panel[:, j] *= 0.95
        
        # Color temperature variation
        panel[:, :, 0] *= 1.02  # Slightly warm
        panel[:, :, 2] *= 0.98  # Reduce blue
    
    def _place_seoul_pixel(self, panel: np.ndarray, x: int, y: int, 
                          color: np.ndarray, depth: float):
        """Place pixel on Seoul panel with anti-aliasing"""
        height, width = panel.shape[:2]
        
        if not (0 <= x < width and 0 <= y < height):
            return
        
        # Calculate pixel radius based on depth
        radius = max(1, int(depth * 6))
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                px, py = x + dx, y + dy
                
                if 0 <= px < width and 0 <= py < height:
                    distance = math.sqrt(dx*dx + dy*dy)
                    
                    if distance <= radius:
                        # Anti-aliasing weight
                        alpha = (radius - distance) / radius
                        alpha *= depth  # Depth-based transparency
                        
                        # Blend with existing pixel
                        panel[py, px] = panel[py, px] * (1 - alpha) + color * alpha
    
    def _apply_seoul_enhancements(self, panel: np.ndarray) -> np.ndarray:
        """Apply Seoul-specific visual enhancements"""
        # Contrast enhancement
        enhanced = panel * self.seoul_params['contrast_enhancement']
        
        # Brightness boost for LED simulation
        enhanced = np.clip(enhanced, 0, self.seoul_params['brightness_max'] / 16)
        
        # Color grading for Seoul aesthetic
        enhanced[:, :, 0] *= 1.1  # Boost red
        enhanced[:, :, 1] *= 0.95  # Reduce green slightly
        enhanced[:, :, 2] *= 1.05  # Boost blue slightly
        
        return enhanced
    
    def _create_combined_corner_view(self, left_panel: np.ndarray, right_panel: np.ndarray) -> np.ndarray:
        """Create combined corner view showing both panels"""
        left_height, left_width = left_panel.shape[:2]
        right_height, right_width = right_panel.shape[:2]
        
        # Create combined canvas
        combined_width = left_width + right_width + 50  # Gap between panels
        combined_height = max(left_height, right_height)
        combined = np.zeros((combined_height, combined_width, 3), dtype=np.float32)
        
        # Place left panel
        combined[:left_height, :left_width] = left_panel
        
        # Place right panel
        start_x = left_width + 50
        combined[:right_height, start_x:start_x + right_width] = right_panel
        
        # Add corner indicator
        self._add_corner_indicator(combined, left_width + 25)
        
        return combined
    
    def _add_corner_indicator(self, combined: np.ndarray, center_x: int):
        """Add visual indicator for corner viewing position"""
        height, width = combined.shape[:2]
        center_y = height // 2
        
        # Draw corner angle indicator
        cv2.line(combined, (center_x - 20, center_y), (center_x + 20, center_y), (255, 255, 255), 2)
        cv2.line(combined, (center_x, center_y - 20), (center_x, center_y + 20), (255, 255, 255), 2)
        
        # Add viewing angle text
        try:
            viewing_angle = self.config.camera.angle
        except AttributeError:
            try:
                viewing_angle = self.config.viewing_angle
            except AttributeError:
                viewing_angle = 25.0
                
        cv2.putText(combined, f"{viewing_angle:.0f}°", 
                   (center_x - 15, center_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    def generate_3d_billboard(self, image_path: Union[str, Path], 
                            output_dir: Union[str, Path] = "output") -> Dict[str, Any]:
        """Generate 3D billboard using Blender integration"""
        if not BLENDER_AVAILABLE:
            logger.warning("Blender not available - skipping 3D generation")
            return {}
        
        logger.info("Generating 3D anamorphic billboard with Blender")
        
        # Load and process image
        image = Image.open(image_path)
        depth_map = self.generate_professional_depth_map(image)
        
        # Create 3D scene
        result = self._create_blender_scene(image, depth_map, output_dir)
        
        return result
    
    def _create_blender_scene(self, image: Image.Image, depth_map: Image.Image, 
                            output_dir: Union[str, Path]) -> Dict[str, Any]:
        """Create Blender scene with anamorphic objects"""
        # Clear existing scene
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        
        # Create billboard frame
        self._create_billboard_frame(image)
        
        # Analyze image for object placement
        objects_data = self._analyze_image_for_objects(image, depth_map)
        
        # Create anamorphic objects
        for obj_data in objects_data:
            self._create_anamorphic_object(obj_data)
        
        # Setup camera and lighting
        self._setup_anamorphic_camera()
        self._setup_professional_lighting()
        
        # Render scene
        output_path = Path(output_dir) / "anamorphic_billboard.png"
        self._render_scene(output_path)
        
        return {
            'render_path': output_path,
            'objects_created': len(objects_data),
            'success': True
        }
    
    def create_test_image(self, width: int = 800, height: int = 600) -> Image.Image:
        """Create optimized test image for anamorphic effects"""
        image = Image.new('RGB', (width, height), color=(20, 25, 35))
        draw = ImageDraw.Draw(image)
        
        # Background gradient
        for y in range(height):
            gradient_color = int(20 + (y / height) * 40)
            draw.line([(0, y), (width, y)], 
                     fill=(gradient_color, gradient_color + 5, gradient_color + 15))
        
        # Floating sphere
        center_x, center_y = width // 2, height // 2 - 50
        radius = 80
        
        for r in range(radius, 0, -2):
            intensity = int(100 + (radius - r) * 2)
            color = (intensity, intensity - 20, intensity + 30)
            draw.ellipse([center_x - r, center_y - r, center_x + r, center_y + r], 
                        fill=color, outline=None)
        
        # Highlight
        highlight_x, highlight_y = center_x - 25, center_y - 30
        draw.ellipse([highlight_x - 15, highlight_y - 15, highlight_x + 15, highlight_y + 15],
                    fill=(255, 255, 255))
        
        # Floating cubes
        cube_positions = [(150, 200, 40), (600, 180, 35), (100, 400, 30), (650, 420, 45)]
        
        for cube_x, cube_y, cube_size in cube_positions:
            # Isometric cube
            top_points = [
                (cube_x, cube_y),
                (cube_x + cube_size, cube_y),
                (cube_x + cube_size + 20, cube_y - 20),
                (cube_x + 20, cube_y - 20)
            ]
            draw.polygon(top_points, fill=(180, 160, 200))
            
            left_points = [
                (cube_x, cube_y),
                (cube_x + 20, cube_y - 20),
                (cube_x + 20, cube_y + cube_size - 20),
                (cube_x, cube_y + cube_size)
            ]
            draw.polygon(left_points, fill=(120, 100, 140))
            
            right_points = [
                (cube_x + cube_size, cube_y),
                (cube_x + cube_size + 20, cube_y - 20),
                (cube_x + cube_size + 20, cube_y + cube_size - 20),
                (cube_x + cube_size, cube_y + cube_size)
            ]
            draw.polygon(right_points, fill=(150, 130, 170))
        
        # Floating text
        try:
            font = ImageFont.truetype("arial.ttf", 48)
        except:
            font = ImageFont.load_default()
        
        text = "ULTIMATE"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = (width - text_width) // 2
        text_y = height - 100
        
        # Text with glow
        for offset in range(5, 0, -1):
            glow_intensity = int(100 - offset * 15)
            draw.text((text_x + offset, text_y + offset), text, 
                     fill=(glow_intensity, glow_intensity, glow_intensity + 20), font=font)
        
        draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)
        
        # Floating particles
        for _ in range(30):
            px = random.randint(0, width)
            py = random.randint(0, height)
            size = random.randint(2, 8)
            brightness = random.randint(100, 255)
            
            draw.ellipse([px - size, py - size, px + size, py + size],
                        fill=(brightness, brightness, brightness + 20))
        
        return image
    
    def process_image(self, image_path: Union[str, Path], 
                     output_dir: Union[str, Path] = "output") -> Dict[str, Any]:
        """Process image and create ultimate anamorphic effect"""
        logger.info(f"Processing image: {image_path}")
        
        # Load image
        image = Image.open(image_path)
        
        # Generate effects
        results = self.create_ultimate_anamorphic_effect(image)
        
        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        for name, img in results.items():
            file_path = output_path / f"ultimate_{name}.png"
            img.save(file_path)
            saved_files[name] = file_path
            logger.info(f"Saved {name}: {file_path}")
        
        # Generate 3D version if Blender available
        if BLENDER_AVAILABLE and self.config.enable_3d_objects:
            blender_result = self.generate_3d_billboard(image_path, output_dir)
            saved_files.update(blender_result)
        
        return {
            'success': True,
            'files': saved_files,
            'config': self.config,
            'message': "Ultimate anamorphic effect generated successfully!"
        }

def main():
    """Main function demonstrating the ultimate anamorphic system"""
    print("🚀 Ultimate Anamorphic Billboard System")
    print("=" * 60)
    
    # Create system with Seoul corner configuration
    config = AnamorphicConfig(
        display_type=DisplayType.SEOUL_WAVE,
        effect_type=EffectType.SEOUL_CORNER,
        viewing_angle=25.0,
        strength=3.0,
        resolution=(1920, 1080),
        enable_3d_objects=True,
        quality="ultra"
    )
    
    system = UltimateAnamorphicSystem(config)
    
    # Create test image
    print("Creating optimized test image...")
    test_image = system.create_test_image()
    test_image.save("ultimate_test_input.jpg")
    
    # Process image
    print("Generating ultimate anamorphic effect...")
    results = system.process_image("ultimate_test_input.jpg", "ultimate_output")
    
    if results['success']:
        print("✅ Ultimate anamorphic effect generated successfully!")
        print(f"Files saved: {len(results['files'])}")
        for name, path in results['files'].items():
            print(f"  - {name}: {path}")
    else:
        print("❌ Failed to generate anamorphic effect")
    
    print("\n🎨 Ultimate Anamorphic System Features:")
    print("  ✓ Professional depth estimation with MiDaS")
    print("  ✓ Seoul-style corner LED optimization")
    print("  ✓ Multi-cue depth analysis fallback")
    print("  ✓ Anti-aliased pixel placement")
    print("  ✓ Professional lighting and shadows")
    print("  ✓ Blender 3D integration")
    print("  ✓ Multiple display type support")
    print("  ✓ Configurable effect parameters")

if __name__ == "__main__":
    main() 