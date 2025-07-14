#!/usr/bin/env python3
"""
Proper Anamorphic Billboard System
Creates a single, unified anamorphic billboard effect with correct perspective mathematics
and proper depth-based distortion for optimal viewing from a specific angle.

Author: Komal Shahid
Course: DSC680 - Applied Data Science
Project: Colorful Canvas - Proper Anamorphic Billboard
"""

import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import os
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BillboardConfig:
    """Configuration for anamorphic billboard"""
    # Viewing parameters
    viewing_angle: float = 45.0  # degrees from ground
    viewing_distance: float = 20.0  # meters
    billboard_height: float = 8.0  # meters
    billboard_width: float = 14.0  # meters
    
    # Rendering parameters
    resolution: Tuple[int, int] = (1920, 1080)
    depth_strength: float = 2.0
    perspective_strength: float = 1.5
    
    # Quality settings
    anti_aliasing: bool = True
    depth_smoothing: bool = True
    edge_enhancement: bool = True

class ProperAnamorphicBillboard:
    """
    Proper anamorphic billboard system that creates a single unified effect
    """
    
    def __init__(self, config: Optional[BillboardConfig] = None):
        """Initialize the proper anamorphic billboard system"""
        self.config = config or BillboardConfig()
        self.device = self._get_optimal_device()
        self.depth_model = None
        self.load_depth_model()
        
        # Calculate perspective transformation matrix
        self.perspective_matrix = self._calculate_perspective_matrix()
        
        logger.info(f"Proper Anamorphic Billboard System initialized on {self.device}")
    
    def _get_optimal_device(self) -> str:
        """Get the best available computing device"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def load_depth_model(self):
        """Load MiDaS depth estimation model"""
        try:
            # Suppress the timm deprecation warning
            import warnings
            warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
            
            # Use the newer MiDaS v3.1 model
            self.depth_model = torch.hub.load("intel-isl/MiDaS", "DPT_Large", 
                                            force_reload=False, trust_repo=True, verbose=False)
            self.depth_transform = torch.hub.load("intel-isl/MiDaS", "transforms", 
                                                verbose=False).dpt_transform
            self.depth_model.to(self.device)
            self.depth_model.eval()
            logger.info("✅ MiDaS DPT_Large model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load DPT_Large, trying MiDaS_small: {e}")
            try:
                self.depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", 
                                                force_reload=False, trust_repo=True, verbose=False)
                self.depth_transform = torch.hub.load("intel-isl/MiDaS", "transforms", 
                                                    verbose=False).small_transform
                self.depth_model.to(self.device)
                self.depth_model.eval()
                logger.info("✅ MiDaS_small model loaded successfully")
            except Exception as e2:
                logger.error(f"Could not load any MiDaS model: {e2}")
                self.depth_model = None
    
    def _calculate_perspective_matrix(self) -> np.ndarray:
        """Calculate the perspective transformation matrix for anamorphic effect"""
        # Billboard dimensions and viewing parameters
        billboard_width = self.config.billboard_width
        billboard_height = self.config.billboard_height
        viewing_distance = self.config.viewing_distance
        viewing_angle = math.radians(self.config.viewing_angle)
        
        # Calculate the perspective transformation
        # This creates the keystone effect needed for anamorphic viewing
        
        # Source points (normal rectangle)
        src_points = np.float32([
            [0, 0],                                    # Top-left
            [billboard_width, 0],                      # Top-right
            [billboard_width, billboard_height],       # Bottom-right
            [0, billboard_height]                      # Bottom-left
        ])
        
        # Destination points (perspective-corrected for viewing angle)
        perspective_factor = math.tan(viewing_angle) * viewing_distance / billboard_height
        
        dst_points = np.float32([
            [0, 0],                                    # Top-left (unchanged)
            [billboard_width, 0],                      # Top-right (unchanged)
            [billboard_width * (1 + perspective_factor), billboard_height],  # Bottom-right (stretched)
            [billboard_width * perspective_factor, billboard_height]         # Bottom-left (stretched)
        ])
        
        # Calculate perspective transformation matrix
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        return matrix
    
    def generate_depth_map(self, image: Image.Image) -> Image.Image:
        """Generate depth map using MiDaS"""
        if self.depth_model is None:
            return self._generate_fallback_depth_map(image)
        
        try:
            # Convert to RGB - direct conversion
            image = image.convert('RGB')
            
            # Get image dimensions
            img_width, img_height = image.size
            
            # Transform image for MiDaS - convert to numpy first
            img_array = np.array(image)
            input_tensor = self.depth_transform(img_array).to(self.device)
            
            with torch.no_grad():
                prediction = self.depth_model(input_tensor)
                
                # Handle tensor dimensions directly
                prediction = prediction.squeeze()
                
                # Resize to original image size
                prediction = F.interpolate(
                    prediction.unsqueeze(0).unsqueeze(0),
                    size=(img_height, img_width),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            
            # Convert to numpy and normalize
            depth_map = prediction.cpu().numpy()
            
            # Enhance depth map for anamorphic effects
            try:
                depth_map = cv2.bilateralFilter(depth_map.astype(np.float32), 9, 75, 75)
            except:
                pass
            
            try:
                # Enhance depth discontinuities
                kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
                edges = cv2.filter2D(depth_map, -1, kernel)
                depth_map = depth_map + 0.2 * edges
            except:
                pass
            
            # Normalize to 0-255
            depth_min, depth_max = depth_map.min(), depth_map.max()
            depth_range = depth_max - depth_min
            try:
                depth_map = ((depth_map - depth_min) / depth_range * 255).astype(np.uint8)
            except:
                depth_map = np.zeros_like(depth_map, dtype=np.uint8)
            
            return Image.fromarray(depth_map)
            
        except Exception as e:
            logger.error(f"Depth estimation failed: {e}")
            return self._generate_fallback_depth_map(image)
    
    def _generate_fallback_depth_map(self, image: Image.Image) -> Image.Image:
        """Generate simple depth map based on brightness and edges"""
        img_array = np.array(image.convert('L'))
        
        # Use brightness as rough depth estimate (darker = farther)
        depth_from_brightness = 255 - img_array
        
        # Add edge-based depth
        edges = cv2.Canny(img_array, 50, 150)
        edge_depth = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
        edge_depth = (edge_depth / edge_depth.max() * 255).astype(np.uint8)
        
        # Combine depth cues
        combined_depth = (depth_from_brightness * 0.7 + edge_depth * 0.3).astype(np.uint8)
        
        # Smooth the result
        combined_depth = cv2.GaussianBlur(combined_depth, (5, 5), 0)
        
        return Image.fromarray(combined_depth)
    
    def create_anamorphic_billboard(self, image: Image.Image, 
                                  depth_map: Optional[Image.Image] = None) -> Dict[str, Image.Image]:
        """Create the main anamorphic billboard effect"""
        logger.info("Creating anamorphic billboard effect...")
        
        # Generate depth map if not provided
        if depth_map is None:
            depth_map = self.generate_depth_map(image)
        
        # Convert to arrays
        img_array = np.array(image).astype(np.float32)
        depth_array = np.array(depth_map.convert('L')).astype(np.float32) / 255.0
        
        # Create the anamorphic transformation
        billboard = self._apply_anamorphic_transformation(img_array, depth_array)
        
        # Apply final enhancements
        billboard = self._apply_final_enhancements(billboard)
        
        return {
            'billboard': Image.fromarray(np.clip(billboard, 0, 255).astype(np.uint8)),
            'depth_map': depth_map,
            'original': image
        }
    
    def _apply_anamorphic_transformation(self, img_array: np.ndarray, 
                                       depth_array: np.ndarray) -> np.ndarray:
        """Apply the core anamorphic transformation"""
        height, width = img_array.shape[:2]
        
        # Create output canvas (larger to accommodate distortion)
        output_height = int(height * 1.5)
        output_width = int(width * 1.8)
        billboard = np.zeros((output_height, output_width, 3), dtype=np.float32)
        
        # Apply perspective transformation with depth-based distortion
        for y in range(height):
            for x in range(width):
                # Get depth value (0 = far, 1 = near)
                depth = depth_array[y, x]
                
                # Calculate anamorphic position
                new_x, new_y = self._calculate_anamorphic_position(
                    x, y, depth, width, height, output_width, output_height
                )
                
                # Place pixel directly - bounds checking in placement function
                try:
                    self._place_pixel_antialiased(billboard, new_x, new_y, 
                                                img_array[y, x], depth)
                except:
                    # Fallback to simple placement
                    try:
                        self._place_pixel_simple(billboard, new_x, new_y, 
                                               img_array[y, x])
                    except:
                        pass
        
        return billboard
    
    def _calculate_anamorphic_position(self, x: int, y: int, depth: float,
                                     src_width: int, src_height: int,
                                     dst_width: int, dst_height: int) -> Tuple[float, float]:
        """Calculate the anamorphic position for a pixel"""
        # Normalize coordinates to 0-1
        norm_x = x / src_width
        norm_y = y / src_height
        
        # Apply perspective transformation
        perspective_factor = self.config.perspective_strength
        depth_factor = depth * self.config.depth_strength
        
        # Keystone correction for anamorphic viewing
        # Objects closer to viewer (higher depth) get more distortion
        keystone_x = norm_x + (depth_factor * (norm_x - 0.5) * perspective_factor * 0.3)
        keystone_y = norm_y + (depth_factor * (norm_y - 0.5) * perspective_factor * 0.2)
        
        # Add viewing angle compensation
        viewing_angle_rad = math.radians(self.config.viewing_angle)
        angle_compensation = math.tan(viewing_angle_rad) * depth_factor * 0.1
        
        # Final position calculation with offset to center the image
        final_x = keystone_x + angle_compensation + 0.1  # Offset to avoid edge clipping
        final_y = keystone_y + angle_compensation * 0.5 + 0.1
        
        # Convert back to pixel coordinates
        pixel_x = final_x * dst_width
        pixel_y = final_y * dst_height
        
        return pixel_x, pixel_y
    
    def _place_pixel_antialiased(self, canvas: np.ndarray, x: float, y: float,
                               color: np.ndarray, depth: float):
        """Place pixel with anti-aliasing"""
        height, width = canvas.shape[:2]
        
        # Calculate pixel radius based on depth
        radius = max(1, int(depth * 3 + 1))
        
        # Get integer coordinates
        center_x, center_y = int(x), int(y)
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                px, py = center_x + dx, center_y + dy
                
                # Direct bounds check and placement
                try:
                    distance = math.sqrt((x - px)**2 + (y - py)**2)
                    
                    # Calculate anti-aliasing weight
                    alpha = max(0, (radius - distance) / radius)
                    alpha *= (0.5 + depth * 0.5)  # Depth-based opacity
                    
                    # Blend with existing pixel
                    canvas[py, px] = canvas[py, px] * (1 - alpha) + color * alpha
                except:
                    pass
    
    def _place_pixel_simple(self, canvas: np.ndarray, x: float, y: float,
                          color: np.ndarray):
        """Place pixel without anti-aliasing"""
        try:
            px, py = int(x), int(y)
            canvas[py, px] = color
        except:
            pass
    
    def _apply_final_enhancements(self, billboard: np.ndarray) -> np.ndarray:
        """Apply final visual enhancements"""
        # Contrast enhancement
        enhanced = billboard * 1.2
        
        # Color grading for billboard aesthetic
        enhanced[:, :, 0] *= 1.05  # Slight red boost
        enhanced[:, :, 1] *= 0.98  # Slight green reduction
        enhanced[:, :, 2] *= 1.02  # Slight blue boost
        
        # Brightness adjustment
        enhanced = np.clip(enhanced, 0, 255)
        
        return enhanced
    
    def process_image(self, image_path: Union[str, Path], 
                     output_dir: Union[str, Path] = "billboard_output") -> Dict[str, Any]:
        """Process an image to create anamorphic billboard"""
        logger.info(f"Processing image: {image_path}")
        
        # Load image
        image = Image.open(image_path)
        
        # Create anamorphic effect
        result = self.create_anamorphic_billboard(image)
        
        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        files_saved = []
        
        # Save billboard
        billboard_path = output_path / "anamorphic_billboard.png"
        result['billboard'].save(billboard_path)
        files_saved.append(str(billboard_path))
        logger.info(f"Saved billboard: {billboard_path}")
        
        # Save depth map
        depth_path = output_path / "depth_map.png"
        result['depth_map'].save(depth_path)
        files_saved.append(str(depth_path))
        logger.info(f"Saved depth map: {depth_path}")
        
        # Save original for comparison
        original_path = output_path / "original.png"
        result['original'].save(original_path)
        files_saved.append(str(original_path))
        logger.info(f"Saved original: {original_path}")
        
        return {
            'success': True,
            'files_saved': files_saved,
            'billboard_path': str(billboard_path),
            'config': self.config
        }

def main():
    """Main function for testing"""
    print("🎯 Proper Anamorphic Billboard System")
    print("=" * 50)
    
    # Create system with custom config
    config = BillboardConfig(
        viewing_angle=45.0,
        viewing_distance=20.0,
        billboard_height=8.0,
        billboard_width=14.0,
        depth_strength=2.0,
        perspective_strength=1.5,
        anti_aliasing=True
    )
    
    system = ProperAnamorphicBillboard(config)
    
    # Test with benchmark image
    benchmark_path = "../data/input/benchmark.jpg"
    if os.path.exists(benchmark_path):
        print("Processing benchmark image...")
        result = system.process_image(benchmark_path, "proper_billboard_output")
        print("✅ Benchmark processing complete!")
        print(f"Files saved: {len(result['files_saved'])}")
        for file_path in result['files_saved']:
            print(f"  - {file_path}")
    else:
        print("❌ Benchmark image not found")
        
        # Create test image
        print("Creating test image...")
        test_image = Image.new('RGB', (800, 600), color=(50, 100, 150))
        draw = ImageDraw.Draw(test_image)
        
        # Add some test objects
        draw.ellipse([200, 150, 400, 350], fill=(255, 100, 100))
        draw.rectangle([450, 200, 650, 400], fill=(100, 255, 100))
        draw.polygon([(100, 500), (200, 400), (300, 500)], fill=(100, 100, 255))
        
        test_image.save("test_billboard_input.png")
        
        result = system.process_image("test_billboard_input.png", "proper_billboard_output")
        print("✅ Test processing complete!")

if __name__ == "__main__":
    main() 