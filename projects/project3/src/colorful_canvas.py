#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Colorful Canvas: AI Art Studio
A consolidated toolkit for creating various 3D visual illusions and effects

Requirements:
-------------
numpy==1.24.3
opencv-python==4.8.0.76
torch==2.0.1
transformers==4.30.2
pillow==10.0.0
matplotlib==3.7.2

Installation:
-------------
pip install numpy opencv-python torch transformers pillow matplotlib
"""
import os
import sys
import argparse
import numpy as np
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import pipeline
import warnings

# Suppress unnecessary warnings
warnings.filterwarnings("ignore")

# Add the project root to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Path to centralized models (if available)
CENTRAL_MODELS_PATH = "/Users/komalshahid/Desktop/Bellevue University/models"

class ColorfulCanvas:
    """Main class for generating various 3D visual illusions and effects."""
    
    def __init__(self):
        """Initialize the ColorfulCanvas generator."""
        self.depth_estimator = None
    
    def _load_depth_estimator(self):
        """Load the depth estimation model if not already loaded."""
        if self.depth_estimator is None:
            try:
                print("Loading depth estimation model...")
                self.depth_estimator = pipeline("depth-estimation", model="Intel/dpt-large")
                print("Depth estimation model loaded successfully")
            except Exception as e:
                print(f"Error loading depth estimation model: {e}")
                return False
        return True
    
    def generate_depth_map(self, image_path, output_path=None):
        """Generate a depth map from an image."""
        if not self._load_depth_estimator():
            return None
            
        try:
            # Load image
            image = Image.open(image_path)
            
            # Get depth map
            depth = self.depth_estimator(image)
            depth_map = depth["depth"]
            
            if output_path:
                depth_map.save(output_path)
                print(f"Depth map saved to {output_path}")
            
            return depth_map
        
        except Exception as e:
            print(f"Error generating depth map: {e}")
            return None
    
    def create_shadow_box_effect(self, input_image_path, output_dir="shadow_box_effects"):
        """Create a shadow box 3D illusion from an input image."""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get base filename
        base_name = os.path.splitext(os.path.basename(input_image_path))[0]
        
        # Load input image
        try:
            image = Image.open(input_image_path)
            print(f"Loaded image: {input_image_path}")
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
        
        # Generate depth map
        print("Generating depth map...")
        depth_map = self.generate_depth_map(input_image_path)
        if depth_map is None:
            print("Failed to generate depth map. Exiting.")
            return None
        
        # Save depth map for reference
        depth_map_path = os.path.join(output_dir, f"{base_name}_depth.png")
        depth_map.save(depth_map_path)
        
        # Create 3D object illusion
        print("Creating 3D object illusion...")
        enhanced_image = self._create_3d_object_illusion(image, depth_map, strength=0.7)
        enhanced_path = os.path.join(output_dir, f"{base_name}_enhanced.png")
        enhanced_image.save(enhanced_path)
        
        # Create shadow box frame
        print("Creating shadow box frame...")
        framed_image = self._create_shadow_box_frame(enhanced_image, 
                                              frame_width=0.15,
                                              shadow_size=25, 
                                              shadow_strength=0.6,
                                              glass_reflection=True)
        framed_path = os.path.join(output_dir, f"{base_name}_shadow_box.png")
        framed_image.save(framed_path)
        
        # Create visualization
        self._visualize_process(image, depth_map, enhanced_image, framed_image, 
                         output_dir, base_name)
        
        return framed_path
    
    def create_screen_pop_effect(self, input_image_path, output_dir="screen_pop_effects", depth_factor=2.0):
        """Create a screen pop effect that makes objects appear to come out of the screen."""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get base filename
        base_name = os.path.splitext(os.path.basename(input_image_path))[0]
        
        # Load input image
        try:
            image = Image.open(input_image_path)
            print(f"Loaded image: {input_image_path}")
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
        
        # Generate depth map
        print("Generating depth map...")
        depth_map = self.generate_depth_map(input_image_path)
        if depth_map is None:
            print("Failed to generate depth map. Exiting.")
            return None
        
        # Convert to numpy arrays
        img_array = np.array(image)
        depth_array = np.array(depth_map)
        
        # Normalize depth map
        depth_norm = depth_array.astype(np.float32)
        depth_norm = (depth_norm - depth_norm.min()) / (depth_norm.max() - depth_norm.min() + 1e-8)
        
        # Create screen pop effect
        print("Creating screen pop effect...")
        result = self._create_screen_pop_effect(img_array, depth_norm, depth_factor=depth_factor)
        
        # Save result
        result_path = os.path.join(output_dir, f"{base_name}_screen_pop.png")
        result_img = Image.fromarray(result)
        result_img.save(result_path)
        
        print(f"Screen pop effect saved to {result_path}")
        return result_path
    
    def _create_shadow_box_frame(self, image, frame_width=0.15, frame_color=(245, 245, 245), 
                               shadow_size=20, shadow_strength=0.7, glass_reflection=True):
        """Create a shadow box frame around the image with 3D effect."""
        if isinstance(image, Image.Image):
            # Convert PIL Image to numpy array
            image = np.array(image)
        
        h, w = image.shape[:2]
        
        # Calculate frame dimensions
        frame_h_size = int(h * frame_width)
        frame_w_size = int(w * frame_width)
        
        # Create larger canvas for the frame
        total_h = h + 2 * frame_h_size
        total_w = w + 2 * frame_w_size
        
        # Create white frame
        frame = np.ones((total_h, total_w, 3), dtype=np.uint8) * np.array(frame_color, dtype=np.uint8)
        
        # Add drop shadow
        shadow_mask = np.zeros((total_h, total_w), dtype=np.float32)
        
        # Inner box shadow edges
        inner_y1, inner_x1 = frame_h_size, frame_w_size
        inner_y2, inner_x2 = frame_h_size + h, frame_w_size + w
        
        # Create shadow gradient
        for i in range(shadow_size):
            # Shadow intensity decreases with distance
            intensity = shadow_strength * (1 - i/shadow_size)
            
            # Bottom shadow
            y_pos = inner_y2 + i
            if y_pos < total_h:
                shadow_mask[y_pos, inner_x1:inner_x2] = intensity
            
            # Right shadow
            x_pos = inner_x2 + i
            if x_pos < total_w:
                shadow_mask[inner_y1:inner_y2, x_pos] = intensity
        
        # Apply shadow
        for c in range(3):
            # Create a darkening effect for the shadow
            frame[:,:,c] = frame[:,:,c] * (1.0 - shadow_mask)
        
        # Place the image in the center
        frame[frame_h_size:frame_h_size+h, frame_w_size:frame_w_size+w] = image
        
        # Add subtle inner border to suggest depth
        inner_border_thickness = 2
        inner_border_color = (220, 220, 220)  # Slightly darker than frame
        
        # Top and left inner borders (lighter)
        frame[frame_h_size:frame_h_size+inner_border_thickness, frame_w_size:frame_w_size+w] = inner_border_color
        frame[frame_h_size:frame_h_size+h, frame_w_size:frame_w_size+inner_border_thickness] = inner_border_color
        
        # Bottom and right inner borders (darker)
        darker_border = (200, 200, 200)
        frame[frame_h_size+h-inner_border_thickness:frame_h_size+h, frame_w_size:frame_w_size+w] = darker_border
        frame[frame_h_size:frame_h_size+h, frame_w_size+w-inner_border_thickness:frame_w_size+w] = darker_border
        
        # Add glass reflection effect if enabled
        if glass_reflection:
            # Create a subtle specular highlight simulating glass
            reflection = np.zeros((total_h, total_w), dtype=np.float32)
            
            # Top-left to bottom-right gradient for reflection
            for i in range(total_h):
                for j in range(total_w):
                    # Normalize coordinates to [0,1]
                    y, x = i/total_h, j/total_w
                    
                    # Create a subtle diagonal highlight
                    highlight = np.exp(-50 * ((x-0.3)**2 + (y-0.2)**2))
                    reflection[i, j] = highlight * 0.15  # Control reflection intensity
            
            # Apply reflection
            for c in range(3):
                frame[:,:,c] = np.clip(frame[:,:,c] + reflection * 255, 0, 255).astype(np.uint8)
        
        return Image.fromarray(frame)
    
    def _create_3d_object_illusion(self, image, depth_map, strength=0.5):
        """Apply 3D transformations to make the object appear to pop out."""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if isinstance(depth_map, Image.Image):
            depth_map = np.array(depth_map)
            # Convert RGB depth map to grayscale if needed
            if len(depth_map.shape) == 3:
                depth_map = cv2.cvtColor(depth_map, cv2.COLOR_RGB2GRAY)
        
        # Normalize depth map to [0,1]
        depth_map = depth_map.astype(np.float32)
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        
        # Increase contrast in the depth map
        depth_map = np.power(depth_map, 0.7)  # Non-linear enhancement
        
        h, w = image.shape[:2]
        result = np.zeros_like(image, dtype=np.float32)
        
        # Apply displacement based on depth
        max_displacement = int(min(h, w) * 0.03 * strength)
        
        # Split into channels to apply different displacements for chromatic effect
        for c in range(3):
            # Different displacement for each channel creates subtle chromatic aberration
            channel_displacement = max_displacement * (1.0 - 0.2 * c)
            displaced = np.zeros((h, w), dtype=np.float32)
            
            for y in range(h):
                for x in range(w):
                    # Calculate displacement based on depth
                    disp = int(depth_map[y, x] * channel_displacement)
                    
                    # Displace toward the center for pop-out effect
                    src_y = y
                    src_x = x
                    
                    # Boundary check
                    if 0 <= src_y < h and 0 <= src_x < w:
                        displaced[y, x] = image[src_y, src_x, c]
            
            result[:, :, c] = displaced
        
        # Apply lighting enhancement based on depth
        for y in range(h):
            for x in range(w):
                # Higher parts appear brighter
                highlight = depth_map[y, x] * strength * 0.4
                result[y, x] = np.clip(result[y, x] * (1 + highlight), 0, 255)
        
        # Add subtle shadows
        shadow_strength = 0.3 * strength
        for y in range(1, h):
            for x in range(1, w):
                if depth_map[y, x] > depth_map[y-1, x] + 0.05:
                    # Apply shadow below raised areas
                    shadow_y = min(y + 3, h-1)
                    result[shadow_y, x] = result[shadow_y, x] * (1 - shadow_strength * (depth_map[y, x] - depth_map[y-1, x]))
        
        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
    
    def _create_screen_pop_effect(self, img, depth_map, depth_factor=2.0):
        """Create a screen pop effect with chromatic aberration and depth-based displacements."""
        h, w = img.shape[:2]
        result = np.zeros_like(img)
        
        # Calculate max displacement based on depth factor
        max_displacement = int(depth_factor * 10)
        
        # Apply different displacements for RGB channels for chromatic aberration effect
        shifts = [
            (max_displacement, max_displacement // 2),  # Red channel
            (0, 0),                                     # Green channel
            (-max_displacement, -max_displacement // 2)  # Blue channel
        ]
        
        # Apply shifts with depth-based scaling
        for c in range(3):
            dx, dy = shifts[c]
            for y in range(h):
                for x in range(w):
                    # Scale displacement based on depth
                    depth_scale = depth_map[y, x]
                    # Compute source coordinates with depth-scaled displacement
                    src_x = x - int(dx * depth_scale)
                    src_y = y - int(dy * depth_scale)
                    
                    # Ensure source coordinates are within bounds
                    if 0 <= src_x < w and 0 <= src_y < h:
                        result[y, x, c] = img[src_y, src_x, c]
                    else:
                        result[y, x, c] = 0
        
        # Enhance contrast based on depth
        for y in range(h):
            for x in range(w):
                highlight = depth_map[y, x] * 0.3
                result[y, x] = np.clip(result[y, x] * (1.0 + highlight), 0, 255)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _visualize_process(self, original, depth_map, enhanced, framed, output_dir, base_name):
        """Create a visualization of the process."""
        # Create figure
        plt.figure(figsize=(16, 12))
        
        # Original image
        plt.subplot(2, 2, 1)
        plt.imshow(original)
        plt.title("Original Image")
        plt.axis('off')
        
        # Depth map
        plt.subplot(2, 2, 2)
        plt.imshow(depth_map, cmap='viridis')
        plt.title("Depth Map")
        plt.axis('off')
        
        # Enhanced 3D image
        plt.subplot(2, 2, 3)
        plt.imshow(enhanced)
        plt.title("3D Enhanced")
        plt.axis('off')
        
        # Final effect
        plt.subplot(2, 2, 4)
        plt.imshow(framed)
        plt.title("Final Effect")
        plt.axis('off')
        
        # Save visualization
        viz_path = os.path.join(output_dir, f"{base_name}_process.png")
        plt.tight_layout()
        plt.savefig(viz_path)
        plt.close()
        print(f"Saved visualization to {viz_path}")

def main():
    """Main function to handle command-line arguments and execute effects."""
    parser = argparse.ArgumentParser(description='Colorful Canvas: AI Art Studio')
    parser.add_argument('--effect', type=str, required=True, 
                        choices=['shadow_box', 'screen_pop'],
                        help='Type of effect to generate')
    parser.add_argument('--input_image', type=str, required=True, 
                        help='Path to input image')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (defaults to effect-specific directory)')
    parser.add_argument('--depth_factor', type=float, default=2.0,
                        help='Depth factor for 3D effects (1.0-3.0)')
    
    args = parser.parse_args()
    
    print("=== Colorful Canvas: AI Art Studio ===")
    
    # Initialize the generator
    generator = ColorfulCanvas()
    
    # Determine output directory if not specified
    output_dir = args.output_dir if args.output_dir else f"{args.effect}_effects"
    
    # Generate the requested effect
    if args.effect == 'shadow_box':
        result_path = generator.create_shadow_box_effect(
            input_image_path=args.input_image,
            output_dir=output_dir
        )
    elif args.effect == 'screen_pop':
        result_path = generator.create_screen_pop_effect(
            input_image_path=args.input_image,
            output_dir=output_dir,
            depth_factor=args.depth_factor
        )
    
    if result_path:
        print(f"\n=== Effect Generation Complete ===")
        print(f"Check {result_path} for the final result")

if __name__ == "__main__":
    main() 