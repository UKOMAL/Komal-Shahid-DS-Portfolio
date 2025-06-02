"""
Color Enhancement Module - After Effects & Illustrator Equivalent
Provides professional color grading, vibrant effects, and vector graphics
for creating stunning anamorphic billboards like the Converse example
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import colorsys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage import color, exposure, filters
from typing import Tuple, List, Optional
import seaborn as sns

class ColorEnhancer:
    """After Effects equivalent for color grading and enhancement"""
    
    def __init__(self):
        self.vibrant_palettes = {
            'neon': ['#FF006E', '#00F5FF', '#FFFF00', '#FF4500', '#00FF7F'],
            'cyber': ['#FF0080', '#0080FF', '#80FF00', '#FF8000', '#8000FF'],
            'pop': ['#FF69B4', '#00CED1', '#FFD700', '#FF6347', '#98FB98'],
            'street': ['#DC143C', '#1E90FF', '#FFD700', '#32CD32', '#FF69B4'],
            'converse': ['#FF4500', '#1E90FF', '#FFD700', '#FF69B4', '#00FF7F']
        }
    
    def enhance_vibrancy(self, image: Image.Image, intensity: float = 1.5) -> Image.Image:
        """
        Boost color vibrancy like After Effects Color Vibrance effect
        """
        # Convert to HSV for saturation boost
        hsv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB_HSV)
        hsv = hsv.astype(np.float32)
        
        # Boost saturation with selective enhancement
        hsv[:, :, 1] = hsv[:, :, 1] * intensity
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        
        # Convert back
        rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV_RGB)
        return Image.fromarray(rgb)
    
    def color_grade_cinematic(self, image: Image.Image, style: str = 'vibrant') -> Image.Image:
        """
        Apply cinematic color grading like After Effects Lumetri Color
        """
        img_array = np.array(image).astype(np.float32) / 255.0
        
        if style == 'vibrant':
            # Boost midtones, enhance contrast
            img_array = exposure.adjust_gamma(img_array, gamma=0.8)
            img_array = exposure.adjust_sigmoid(img_array, cutoff=0.5, gain=10, inv=False)
        elif style == 'neon':
            # High contrast, saturated
            img_array = exposure.rescale_intensity(img_array, out_range=(0.1, 0.9))
            img_array = exposure.adjust_gamma(img_array, gamma=0.6)
        elif style == 'pop':
            # Bright, punchy colors
            img_array = exposure.adjust_gamma(img_array, gamma=0.7)
            img_array = exposure.rescale_intensity(img_array, out_range=(0.05, 0.95))
        
        return Image.fromarray((img_array * 255).astype(np.uint8))
    
    def apply_glow_effect(self, image: Image.Image, glow_radius: int = 20, glow_strength: float = 0.8) -> Image.Image:
        """
        Add glow effect like After Effects Glow filter
        """
        # Create glow layer
        glow = image.filter(ImageFilter.GaussianBlur(radius=glow_radius))
        
        # Blend modes
        result = Image.blend(image, glow, glow_strength)
        
        # Add outer glow
        outer_glow = image.filter(ImageFilter.GaussianBlur(radius=glow_radius * 2))
        result = Image.blend(result, outer_glow, glow_strength * 0.3)
        
        return result
    
    def create_color_palette_from_image(self, image: Image.Image, n_colors: int = 5) -> List[str]:
        """
        Extract dominant colors like Adobe Color wheel
        """
        # Resize for faster processing
        small_image = image.resize((150, 150))
        img_array = np.array(small_image)
        
        # Reshape and apply k-means clustering
        pixels = img_array.reshape(-1, 3)
        
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_colors, random_state=42)
        kmeans.fit(pixels)
        
        # Convert to hex colors
        colors = []
        for center in kmeans.cluster_centers_:
            hex_color = '#{:02x}{:02x}{:02x}'.format(int(center[0]), int(center[1]), int(center[2]))
            colors.append(hex_color)
        
        return colors
    
    def apply_vibrant_palette(self, image: Image.Image, palette_name: str = 'converse') -> Image.Image:
        """
        Remap image colors to vibrant palette like Illustrator color replacement
        """
        palette = self.vibrant_palettes.get(palette_name, self.vibrant_palettes['pop'])
        
        # Convert palette to RGB
        palette_rgb = []
        for hex_color in palette:
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))
            palette_rgb.append(rgb)
        
        img_array = np.array(image)
        result = np.zeros_like(img_array)
        
        # Simple color mapping based on brightness zones
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB_GRAY)
        
        for i, color in enumerate(palette_rgb):
            mask = (gray >= i * 255 // len(palette)) & (gray < (i + 1) * 255 // len(palette))
            result[mask] = color
        
        return Image.fromarray(result)

class VectorGraphics:
    """Illustrator equivalent for vector graphics and design elements"""
    
    def __init__(self):
        self.design_elements = []
    
    def create_geometric_overlay(self, width: int, height: int, style: str = 'modern') -> Image.Image:
        """
        Create geometric overlays like Illustrator shapes
        """
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.axis('off')
        
        if style == 'modern':
            # Add geometric shapes
            from matplotlib.patches import Circle, Rectangle, Polygon
            
            # Circles
            circle1 = Circle((width*0.2, height*0.7), width*0.1, color='#FF006E', alpha=0.7)
            circle2 = Circle((width*0.8, height*0.3), width*0.15, color='#00F5FF', alpha=0.6)
            ax.add_patch(circle1)
            ax.add_patch(circle2)
            
            # Rectangles
            rect = Rectangle((width*0.1, height*0.1), width*0.3, height*0.2, 
                           color='#FFFF00', alpha=0.5, angle=15)
            ax.add_patch(rect)
            
        elif style == 'street':
            # Street art style elements
            from matplotlib.patches import Wedge
            
            wedge = Wedge((width*0.5, height*0.5), width*0.2, 0, 120, 
                         color='#FF4500', alpha=0.8)
            ax.add_patch(wedge)
        
        plt.tight_layout()
        fig.canvas.draw()
        
        # Convert to PIL Image
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        return Image.fromarray(buf)
    
    def add_text_effects(self, image: Image.Image, text: str, position: Tuple[int, int], 
                        style: str = 'bold') -> Image.Image:
        """
        Add text with effects like Illustrator text styles
        """
        from PIL import ImageDraw, ImageFont
        
        draw = ImageDraw.Draw(image)
        
        try:
            # Try to use a bold font
            font_size = min(image.width, image.height) // 10
            font = ImageFont.truetype("/System/Library/Fonts/Arial Bold.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        if style == 'bold':
            # Add shadow effect
            shadow_offset = 3
            draw.text((position[0] + shadow_offset, position[1] + shadow_offset), 
                     text, fill='#000000', font=font)
            draw.text(position, text, fill='#FFFFFF', font=font)
        elif style == 'neon':
            # Neon glow effect
            for offset in range(5, 0, -1):
                alpha = 255 // offset
                color = f'#FF006E{alpha:02x}'
                draw.text((position[0] + offset, position[1] + offset), 
                         text, fill=color, font=font)
            draw.text(position, text, fill='#FFFFFF', font=font)
        
        return image

class AnamorphicColorProcessor:
    """
    Specialized color processing for anamorphic billboards
    Combines After Effects and Illustrator techniques
    """
    
    def __init__(self):
        self.enhancer = ColorEnhancer()
        self.vector = VectorGraphics()
    
    def create_converse_style_billboard(self, image: Image.Image, 
                                      add_effects: bool = True) -> Image.Image:
        """
        Create a Converse-style vibrant anamorphic billboard
        """
        # Step 1: Enhance base colors
        result = self.enhancer.enhance_vibrancy(image, intensity=1.8)
        
        # Step 2: Apply cinematic grading
        result = self.enhancer.color_grade_cinematic(result, style='vibrant')
        
        # Step 3: Add glow effects
        if add_effects:
            result = self.enhancer.apply_glow_effect(result, glow_radius=15, glow_strength=0.6)
        
        # Step 4: Create geometric overlay
        if add_effects:
            overlay = self.vector.create_geometric_overlay(
                result.width, result.height, style='modern'
            )
            # Blend overlay
            result = Image.blend(result, overlay, 0.3)
        
        return result
    
    def process_for_blender_material(self, image: Image.Image, 
                                   enhancement_level: str = 'high') -> Tuple[Image.Image, dict]:
        """
        Process image for optimal Blender material creation
        Returns enhanced image and material properties
        """
        if enhancement_level == 'high':
            # Maximum vibrancy for anamorphic effect
            processed = self.enhancer.enhance_vibrancy(image, intensity=2.0)
            processed = self.enhancer.color_grade_cinematic(processed, style='neon')
            
            material_props = {
                'emission_strength': 3.0,
                'roughness': 0.1,
                'metallic': 0.2,
                'specular': 0.8
            }
        elif enhancement_level == 'medium':
            processed = self.enhancer.enhance_vibrancy(image, intensity=1.5)
            processed = self.enhancer.color_grade_cinematic(processed, style='vibrant')
            
            material_props = {
                'emission_strength': 2.0,
                'roughness': 0.3,
                'metallic': 0.1,
                'specular': 0.5
            }
        else:  # low
            processed = self.enhancer.enhance_vibrancy(image, intensity=1.2)
            
            material_props = {
                'emission_strength': 1.5,
                'roughness': 0.5,
                'metallic': 0.0,
                'specular': 0.3
            }
        
        return processed, material_props
    
    def create_depth_aware_colors(self, image: Image.Image, depth_map: Image.Image) -> Image.Image:
        """
        Create color variations based on depth for better 3D effect
        """
        img_array = np.array(image).astype(np.float32)
        depth_array = np.array(depth_map.convert('L')).astype(np.float32) / 255.0
        
        # Enhance colors based on depth
        # Closer objects (higher depth values) get more vibrant colors
        for i in range(3):  # RGB channels
            img_array[:, :, i] = img_array[:, :, i] * (0.8 + 0.4 * depth_array)
        
        # Add color shift based on depth
        hsv = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB_HSV)
        hsv = hsv.astype(np.float32)
        
        # Shift hue slightly based on depth
        hsv[:, :, 0] = hsv[:, :, 0] + (depth_array * 10)  # Small hue shift
        hsv[:, :, 0] = np.mod(hsv[:, :, 0], 180)  # Keep in valid range
        
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV_RGB)
        return Image.fromarray(result)

# Example usage and testing
def test_color_enhancement():
    """Test the color enhancement capabilities"""
    processor = AnamorphicColorProcessor()
    
    # Create a test image
    test_img = Image.new('RGB', (800, 600), color=(100, 150, 200))
    
    # Apply Converse-style processing
    enhanced = processor.create_converse_style_billboard(test_img)
    
    # Process for Blender
    blender_img, props = processor.process_for_blender_material(test_img, 'high')
    
    print("✅ Color enhancement system ready!")
    print(f"Material properties: {props}")
    
    return enhanced, blender_img, props

if __name__ == "__main__":
    test_color_enhancement() 