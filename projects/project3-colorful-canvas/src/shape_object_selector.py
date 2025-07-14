#!/usr/bin/env python3
"""
Shape and Object Selector for Anamorphic Billboard System
Allows selection of different items and shapes as per project objectives
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
from typing import Dict, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass

class ShapeType(Enum):
    """Available shape types"""
    SPHERE = "sphere"
    CUBE = "cube"
    CYLINDER = "cylinder"
    PYRAMID = "pyramid"
    TORUS = "torus"
    CONE = "cone"
    PRISM = "prism"
    DODECAHEDRON = "dodecahedron"

class ObjectType(Enum):
    """Available object types"""
    CHARACTER = "character"
    BUILDING = "building"
    VEHICLE = "vehicle"
    FURNITURE = "furniture"
    NATURE = "nature"
    ABSTRACT = "abstract"
    GEOMETRIC = "geometric"
    ARCHITECTURAL = "architectural"

@dataclass
class ShapeConfig:
    """Configuration for a shape"""
    shape_type: ShapeType
    position: Tuple[int, int]  # x, y position
    size: int  # radius or side length
    color: Tuple[int, int, int]  # RGB color
    depth: float  # 0.0 (far) to 1.0 (near)
    rotation: float = 0.0  # rotation angle in degrees
    transparency: float = 1.0  # 0.0 (transparent) to 1.0 (opaque)

@dataclass
class ObjectConfig:
    """Configuration for an object"""
    object_type: ObjectType
    position: Tuple[int, int]
    size: Tuple[int, int]  # width, height
    color_scheme: str  # color scheme name
    depth: float
    details: Dict = None  # additional object-specific details

class ShapeObjectSelector:
    """
    System for selecting and creating various shapes and objects
    for anamorphic billboard generation
    """
    
    def __init__(self, canvas_size: Tuple[int, int] = (800, 600)):
        """Initialize the shape and object selector"""
        self.canvas_size = canvas_size
        self.shapes = []
        self.objects = []
        
        # Color schemes
        self.color_schemes = {
            'vibrant': [(255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100)],
            'pastel': [(255, 200, 200), (200, 255, 200), (200, 200, 255), (255, 255, 200)],
            'neon': [(255, 0, 255), (0, 255, 255), (255, 255, 0), (255, 128, 0)],
            'earth': [(139, 69, 19), (34, 139, 34), (70, 130, 180), (255, 215, 0)],
            'monochrome': [(255, 255, 255), (200, 200, 200), (150, 150, 150), (100, 100, 100)]
        }
    
    def add_shape(self, shape_config: ShapeConfig):
        """Add a shape to the scene"""
        self.shapes.append(shape_config)
    
    def add_object(self, object_config: ObjectConfig):
        """Add an object to the scene"""
        self.objects.append(object_config)
    
    def create_sphere(self, center: Tuple[int, int], radius: int, 
                     color: Tuple[int, int, int], depth: float) -> np.ndarray:
        """Create a 3D-looking sphere"""
        canvas = np.zeros((*self.canvas_size[::-1], 3), dtype=np.uint8)
        
        # Create gradient for 3D effect
        for y in range(max(0, center[1] - radius), min(self.canvas_size[1], center[1] + radius)):
            for x in range(max(0, center[0] - radius), min(self.canvas_size[0], center[0] + radius)):
                dx = x - center[0]
                dy = y - center[1]
                distance = np.sqrt(dx*dx + dy*dy)
                
                if distance <= radius:
                    # Calculate 3D lighting
                    normal_z = np.sqrt(max(0, radius*radius - dx*dx - dy*dy)) / radius
                    light_intensity = 0.3 + 0.7 * normal_z
                    
                    # Apply lighting to color
                    lit_color = tuple(int(c * light_intensity) for c in color)
                    canvas[y, x] = lit_color
        
        return canvas
    
    def create_cube(self, center: Tuple[int, int], size: int, 
                   color: Tuple[int, int, int], depth: float) -> np.ndarray:
        """Create a 3D-looking cube"""
        canvas = np.zeros((*self.canvas_size[::-1], 3), dtype=np.uint8)
        
        # Calculate cube vertices for isometric view
        half_size = size // 2
        
        # Front face
        front_color = color
        cv2.rectangle(canvas, 
                     (center[0] - half_size, center[1] - half_size),
                     (center[0] + half_size, center[1] + half_size),
                     front_color, -1)
        
        # Top face (lighter)
        top_color = tuple(min(255, int(c * 1.3)) for c in color)
        top_points = np.array([
            [center[0] - half_size, center[1] - half_size],
            [center[0] + half_size, center[1] - half_size],
            [center[0] + half_size + size//4, center[1] - half_size - size//4],
            [center[0] - half_size + size//4, center[1] - half_size - size//4]
        ], np.int32)
        cv2.fillPoly(canvas, [top_points], top_color)
        
        # Right face (darker)
        right_color = tuple(int(c * 0.7) for c in color)
        right_points = np.array([
            [center[0] + half_size, center[1] - half_size],
            [center[0] + half_size, center[1] + half_size],
            [center[0] + half_size + size//4, center[1] + half_size - size//4],
            [center[0] + half_size + size//4, center[1] - half_size - size//4]
        ], np.int32)
        cv2.fillPoly(canvas, [right_points], right_color)
        
        return canvas
    
    def create_pyramid(self, center: Tuple[int, int], size: int, 
                      color: Tuple[int, int, int], depth: float) -> np.ndarray:
        """Create a 3D-looking pyramid"""
        canvas = np.zeros((*self.canvas_size[::-1], 3), dtype=np.uint8)
        
        half_size = size // 2
        
        # Base
        base_color = tuple(int(c * 0.6) for c in color)
        base_points = np.array([
            [center[0] - half_size, center[1] + half_size],
            [center[0] + half_size, center[1] + half_size],
            [center[0] + half_size + size//4, center[1] + half_size - size//4],
            [center[0] - half_size + size//4, center[1] + half_size - size//4]
        ], np.int32)
        cv2.fillPoly(canvas, [base_points], base_color)
        
        # Left face
        left_color = tuple(int(c * 0.8) for c in color)
        left_points = np.array([
            [center[0] - half_size, center[1] + half_size],
            [center[0], center[1] - half_size],
            [center[0] + size//4, center[1] - half_size - size//4],
            [center[0] - half_size + size//4, center[1] + half_size - size//4]
        ], np.int32)
        cv2.fillPoly(canvas, [left_points], left_color)
        
        # Right face
        right_color = color
        right_points = np.array([
            [center[0] + half_size, center[1] + half_size],
            [center[0], center[1] - half_size],
            [center[0] + size//4, center[1] - half_size - size//4],
            [center[0] + half_size + size//4, center[1] + half_size - size//4]
        ], np.int32)
        cv2.fillPoly(canvas, [right_points], right_color)
        
        return canvas
    
    def create_cylinder(self, center: Tuple[int, int], radius: int, height: int,
                       color: Tuple[int, int, int], depth: float) -> np.ndarray:
        """Create a 3D-looking cylinder"""
        canvas = np.zeros((*self.canvas_size[::-1], 3), dtype=np.uint8)
        
        # Top ellipse (lighter)
        top_color = tuple(min(255, int(c * 1.2)) for c in color)
        cv2.ellipse(canvas, 
                   (center[0], center[1] - height//2),
                   (radius, radius//3),
                   0, 0, 360, top_color, -1)
        
        # Side rectangle
        side_color = color
        cv2.rectangle(canvas,
                     (center[0] - radius, center[1] - height//2),
                     (center[0] + radius, center[1] + height//2),
                     side_color, -1)
        
        # Bottom ellipse (darker)
        bottom_color = tuple(int(c * 0.8) for c in color)
        cv2.ellipse(canvas,
                   (center[0], center[1] + height//2),
                   (radius, radius//3),
                   0, 0, 360, bottom_color, -1)
        
        return canvas
    
    def create_scene_from_selection(self, shape_configs: List[ShapeConfig],
                                  object_configs: List[ObjectConfig] = None) -> Image.Image:
        """Create a scene from selected shapes and objects"""
        # Create canvas
        canvas = np.zeros((*self.canvas_size[::-1], 3), dtype=np.uint8)
        
        # Add background gradient
        for y in range(self.canvas_size[1]):
            gradient_value = int(20 + (y / self.canvas_size[1]) * 60)
            canvas[y, :] = [gradient_value, gradient_value + 10, gradient_value + 20]
        
        # Sort by depth (far to near)
        all_items = []
        
        # Add shapes
        for shape_config in shape_configs:
            all_items.append(('shape', shape_config))
        
        # Add objects if provided
        if object_configs:
            for object_config in object_configs:
                all_items.append(('object', object_config))
        
        # Sort by depth (render far objects first)
        all_items.sort(key=lambda item: item[1].depth)
        
        # Render each item
        for item_type, config in all_items:
            if item_type == 'shape':
                shape_canvas = self._render_shape(config)
                # Blend with main canvas
                mask = (shape_canvas.sum(axis=2) > 0).astype(np.float32)
                for c in range(3):
                    canvas[:, :, c] = canvas[:, :, c] * (1 - mask) + shape_canvas[:, :, c] * mask
            
            elif item_type == 'object':
                object_canvas = self._render_object(config)
                # Blend with main canvas
                mask = (object_canvas.sum(axis=2) > 0).astype(np.float32)
                for c in range(3):
                    canvas[:, :, c] = canvas[:, :, c] * (1 - mask) + object_canvas[:, :, c] * mask
        
        return Image.fromarray(canvas.astype(np.uint8))
    
    def _render_shape(self, config: ShapeConfig) -> np.ndarray:
        """Render a single shape"""
        if config.shape_type == ShapeType.SPHERE:
            return self.create_sphere(config.position, config.size, config.color, config.depth)
        elif config.shape_type == ShapeType.CUBE:
            return self.create_cube(config.position, config.size, config.color, config.depth)
        elif config.shape_type == ShapeType.PYRAMID:
            return self.create_pyramid(config.position, config.size, config.color, config.depth)
        elif config.shape_type == ShapeType.CYLINDER:
            return self.create_cylinder(config.position, config.size, config.size//2, config.color, config.depth)
        else:
            # Default to sphere for unsupported shapes
            return self.create_sphere(config.position, config.size, config.color, config.depth)
    
    def _render_object(self, config: ObjectConfig) -> np.ndarray:
        """Render a single object"""
        canvas = np.zeros((*self.canvas_size[::-1], 3), dtype=np.uint8)
        
        # Get colors from scheme
        colors = self.color_schemes.get(config.color_scheme, self.color_schemes['vibrant'])
        
        if config.object_type == ObjectType.CHARACTER:
            # Simple character representation
            color = colors[0]
            # Head
            cv2.circle(canvas, (config.position[0], config.position[1] - 20), 15, color, -1)
            # Body
            cv2.rectangle(canvas, 
                         (config.position[0] - 10, config.position[1] - 5),
                         (config.position[0] + 10, config.position[1] + 25),
                         color, -1)
        
        elif config.object_type == ObjectType.BUILDING:
            # Simple building representation
            color = colors[1]
            # Main structure
            cv2.rectangle(canvas,
                         (config.position[0] - config.size[0]//2, config.position[1] - config.size[1]),
                         (config.position[0] + config.size[0]//2, config.position[1]),
                         color, -1)
            # Roof
            roof_color = tuple(int(c * 0.8) for c in color)
            roof_points = np.array([
                [config.position[0] - config.size[0]//2 - 5, config.position[1] - config.size[1]],
                [config.position[0] + config.size[0]//2 + 5, config.position[1] - config.size[1]],
                [config.position[0], config.position[1] - config.size[1] - 20]
            ], np.int32)
            cv2.fillPoly(canvas, [roof_points], roof_color)
        
        elif config.object_type == ObjectType.VEHICLE:
            # Simple vehicle representation
            color = colors[2]
            # Body
            cv2.rectangle(canvas,
                         (config.position[0] - config.size[0]//2, config.position[1] - config.size[1]//2),
                         (config.position[0] + config.size[0]//2, config.position[1] + config.size[1]//2),
                         color, -1)
            # Wheels
            wheel_color = tuple(int(c * 0.5) for c in color)
            cv2.circle(canvas, (config.position[0] - config.size[0]//3, config.position[1] + config.size[1]//2), 8, wheel_color, -1)
            cv2.circle(canvas, (config.position[0] + config.size[0]//3, config.position[1] + config.size[1]//2), 8, wheel_color, -1)
        
        return canvas
    
    def create_preset_scene(self, preset_name: str) -> Image.Image:
        """Create a preset scene with predefined shapes and objects"""
        if preset_name == "geometric_showcase":
            shapes = [
                ShapeConfig(ShapeType.SPHERE, (200, 200), 60, (255, 100, 100), 0.8),
                ShapeConfig(ShapeType.CUBE, (400, 200), 80, (100, 255, 100), 0.6),
                ShapeConfig(ShapeType.PYRAMID, (600, 200), 70, (100, 100, 255), 0.4),
                ShapeConfig(ShapeType.CYLINDER, (300, 400), 50, (255, 255, 100), 0.7),
            ]
            return self.create_scene_from_selection(shapes)
        
        elif preset_name == "city_scene":
            shapes = [
                ShapeConfig(ShapeType.CUBE, (150, 400), 60, (120, 120, 120), 0.3),  # Building 1
                ShapeConfig(ShapeType.CUBE, (250, 350), 80, (140, 140, 140), 0.4),  # Building 2
                ShapeConfig(ShapeType.CUBE, (350, 380), 70, (100, 100, 100), 0.2),  # Building 3
            ]
            objects = [
                ObjectConfig(ObjectType.CHARACTER, (500, 450), (20, 40), "vibrant", 0.9),
                ObjectConfig(ObjectType.VEHICLE, (600, 430), (60, 30), "neon", 0.8),
            ]
            return self.create_scene_from_selection(shapes, objects)
        
        elif preset_name == "abstract_art":
            shapes = [
                ShapeConfig(ShapeType.SPHERE, (300, 200), 80, (255, 0, 255), 0.7),
                ShapeConfig(ShapeType.PYRAMID, (500, 300), 60, (0, 255, 255), 0.5),
                ShapeConfig(ShapeType.CYLINDER, (200, 400), 40, (255, 255, 0), 0.6),
                ShapeConfig(ShapeType.CUBE, (600, 150), 50, (255, 128, 0), 0.4),
            ]
            return self.create_scene_from_selection(shapes)
        
        else:
            # Default scene
            shapes = [
                ShapeConfig(ShapeType.SPHERE, (400, 300), 100, (255, 255, 255), 0.8),
            ]
            return self.create_scene_from_selection(shapes)

def main():
    """Demo of the shape and object selector"""
    print("🎨 Shape and Object Selector Demo")
    print("=" * 50)
    
    selector = ShapeObjectSelector((800, 600))
    
    # Create different preset scenes
    presets = ["geometric_showcase", "city_scene", "abstract_art"]
    
    for preset in presets:
        print(f"Creating {preset}...")
        scene = selector.create_preset_scene(preset)
        scene.save(f"scene_{preset}.png")
        print(f"✅ Saved scene_{preset}.png")
    
    print("\n🎯 Available Shapes:")
    for shape in ShapeType:
        print(f"  - {shape.value}")
    
    print("\n🏗️ Available Objects:")
    for obj in ObjectType:
        print(f"  - {obj.value}")
    
    print("\n🌈 Available Color Schemes:")
    for scheme in selector.color_schemes.keys():
        print(f"  - {scheme}")

if __name__ == "__main__":
    main() 