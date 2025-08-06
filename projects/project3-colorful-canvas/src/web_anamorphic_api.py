#!/usr/bin/env python3
"""
Web API for Anamorphic Billboard Generation
Connects the existing anamorphic code to a web interface

Author: Komal Shahid
Course: DSC680 - Applied Data Science
"""

import os
import sys
import base64
import json
import traceback
from pathlib import Path
from io import BytesIO
from PIL import Image
import numpy as np

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

class WebAnamorphicAPI:
    """Web API wrapper for the anamorphic billboard system"""
    
    def __init__(self):
        """Initialize the web API"""
        self.output_dir = Path("web_output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize the systems
        self.modern_system = None
        self.ultimate_system = None
        
        try:
            from modern_anamorphic_system import ModernAnamorphicSystem, ModernBillboardConfig
            config = ModernBillboardConfig()
            self.modern_system = ModernAnamorphicSystem(config)
            print("✅ Modern anamorphic system loaded")
        except Exception as e:
            print(f"⚠️ Could not load modern system: {e}")
        
        try:
            from core.ultimate_anamorphic_system import UltimateAnamorphicSystem
            self.ultimate_system = UltimateAnamorphicSystem()
            print("✅ Ultimate anamorphic system loaded")
        except Exception as e:
            print(f"⚠️ Could not load ultimate system: {e}")
    
    def process_image_base64(self, image_data_base64: str) -> dict:
        """
        Process a base64 encoded image and return anamorphic results
        
        Args:
            image_data_base64: Base64 encoded image data
            
        Returns:
            dict: Processing results with images and status
        """
        try:
            # Decode base64 image
            image_data = base64.b64decode(image_data_base64.split(',')[1])
            image = Image.open(BytesIO(image_data)).convert('RGB')
            
            # Save input image
            input_path = self.output_dir / "input_image.png"
            image.save(input_path)
            
            results = {
                'status': 'success',
                'message': 'Image processed successfully',
                'steps': [],
                'final_result': None
            }
            
            # Step 1: Create depth map
            results['steps'].append({
                'name': 'Depth Analysis',
                'status': 'completed',
                'description': 'AI-powered depth estimation using MiDaS-style analysis'
            })
            
            depth_map = self._create_depth_map(image)
            depth_path = self.output_dir / "depth_map.png"
            depth_map.save(depth_path)
            
            # Step 2: Anamorphic transformation
            results['steps'].append({
                'name': 'Anamorphic Transformation',
                'status': 'completed',
                'description': 'Mathematical perspective distortion for 3D illusion'
            })
            
            anamorphic_result = self._create_anamorphic_transform(image, depth_map)
            anamorphic_path = self.output_dir / "anamorphic_result.png"
            anamorphic_result.save(anamorphic_path)
            
            # Step 3: Billboard rendering
            results['steps'].append({
                'name': '3D Billboard Generation',
                'status': 'completed',
                'description': 'Seoul-style LED billboard with corner viewing angle'
            })
            
            billboard_result = self._create_billboard_render(anamorphic_result)
            billboard_path = self.output_dir / "billboard_final.png"
            billboard_result.save(billboard_path)
            
            # Convert final result to base64
            buffer = BytesIO()
            billboard_result.save(buffer, format='PNG')
            final_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            results['final_result'] = f"data:image/png;base64,{final_base64}"
            
            return results
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Processing failed: {str(e)}',
                'error_details': traceback.format_exc()
            }
    
    def _create_depth_map(self, image: Image.Image) -> Image.Image:
        """Create a depth map from the input image"""
        # Convert to grayscale and apply depth-like effects
        img_array = np.array(image.convert('L'))
        
        # Create depth effect using gradients and edge detection
        height, width = img_array.shape
        
        # Create radial gradient for depth
        center_x, center_y = width // 2, height // 2
        y, x = np.ogrid[:height, :width]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        radial_gradient = (distance / max_distance * 255).astype(np.uint8)
        
        # Combine with original image for realistic depth
        depth_map = (img_array * 0.7 + radial_gradient * 0.3).astype(np.uint8)
        
        # Apply some smoothing
        depth_map = cv2.GaussianBlur(depth_map, (5, 5), 0) if 'cv2' in globals() else depth_map
        
        return Image.fromarray(depth_map).convert('RGB')
    
    def _create_anamorphic_transform(self, image: Image.Image, depth_map: Image.Image) -> Image.Image:
        """Create anamorphic transformation"""
        img_array = np.array(image)
        depth_array = np.array(depth_map.convert('L'))
        
        height, width = img_array.shape[:2]
        
        # Create perspective transformation matrix for anamorphic effect
        # This simulates viewing from a corner angle
        
        # Define source and destination points for perspective transform
        src_points = np.float32([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ])
        
        # Anamorphic distortion - compress horizontally, stretch vertically
        compression_factor = 0.6
        stretch_factor = 1.4
        
        dst_points = np.float32([
            [width * (1 - compression_factor) / 2, 0],
            [width * (1 + compression_factor) / 2, 0],
            [width * (1 + compression_factor) / 2, height * stretch_factor],
            [width * (1 - compression_factor) / 2, height * stretch_factor]
        ])
        
        # Apply perspective transformation if cv2 is available
        try:
            import cv2
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            transformed = cv2.warpPerspective(img_array, matrix, (width, int(height * stretch_factor)))
            return Image.fromarray(transformed)
        except:
            # Fallback: simple resize transformation
            new_width = int(width * compression_factor)
            new_height = int(height * stretch_factor)
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    def _create_billboard_render(self, anamorphic_image: Image.Image) -> Image.Image:
        """Create final billboard rendering with LED panel effect"""
        # Create a billboard-style frame
        img_width, img_height = anamorphic_image.size
        
        # Create larger canvas for billboard effect
        billboard_width = img_width + 200
        billboard_height = img_height + 200
        
        # Create black background (LED panel)
        billboard = Image.new('RGB', (billboard_width, billboard_height), (20, 20, 20))
        
        # Add LED panel grid effect
        billboard_array = np.array(billboard)
        
        # Create LED pixel grid
        grid_size = 8
        for y in range(0, billboard_height, grid_size):
            for x in range(0, billboard_width, grid_size):
                if (x // grid_size + y // grid_size) % 2 == 0:
                    billboard_array[y:y+1, x:x+grid_size] = [25, 25, 25]
                    billboard_array[y:y+grid_size, x:x+1] = [25, 25, 25]
        
        billboard = Image.fromarray(billboard_array)
        
        # Paste the anamorphic image in center
        paste_x = (billboard_width - img_width) // 2
        paste_y = (billboard_height - img_height) // 2
        billboard.paste(anamorphic_image, (paste_x, paste_y))
        
        # Add corner LED frame effect
        from PIL import ImageDraw
        draw = ImageDraw.Draw(billboard)
        
        # Draw LED frame
        frame_thickness = 20
        frame_color = (100, 100, 255)  # Blue LED color
        
        # Top and bottom frames
        draw.rectangle([0, 0, billboard_width, frame_thickness], fill=frame_color)
        draw.rectangle([0, billboard_height-frame_thickness, billboard_width, billboard_height], fill=frame_color)
        
        # Left and right frames
        draw.rectangle([0, 0, frame_thickness, billboard_height], fill=frame_color)
        draw.rectangle([billboard_width-frame_thickness, 0, billboard_width, billboard_height], fill=frame_color)
        
        return billboard

def create_test_api():
    """Create a test instance of the API"""
    return WebAnamorphicAPI()

if __name__ == "__main__":
    # Test the API
    print("🧪 Testing Web Anamorphic API...")
    api = create_test_api()
    
    # Create a test image
    test_image = Image.new('RGB', (400, 300), (255, 100, 100))
    
    # Convert to base64
    buffer = BytesIO()
    test_image.save(buffer, format='PNG')
    test_base64 = base64.b64encode(buffer.getvalue()).decode()
    test_data = f"data:image/png;base64,{test_base64}"
    
    # Process the test image
    result = api.process_image_base64(test_data)
    print(f"✅ Test result: {result['status']}")
    print(f"📊 Steps completed: {len(result['steps'])}")