#!/usr/bin/env python3
"""
Integrated Anamorphic Billboard System
Combines shape/object selection with anamorphic billboard generation
"""

import numpy as np
from PIL import Image
import cv2
import os
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Import our modules
from shape_object_selector import ShapeObjectSelector, ShapeConfig, ObjectConfig, ShapeType, ObjectType
from core.proper_anamorphic_billboard import ProperAnamorphicBillboard, BillboardConfig

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

@dataclass
class IntegratedConfig:
    """Configuration for the integrated system"""
    # Shape/Object selection
    canvas_size: Tuple[int, int] = (1434, 755)  # Match benchmark size
    
    # Billboard configuration
    billboard_config: BillboardConfig = None
    
    # Output settings
    output_dir: str = "integrated_output"
    save_intermediate: bool = True

class IntegratedAnamorphicSystem:
    """
    Integrated system combining shape/object selection with anamorphic billboard generation
    """
    
    def __init__(self, config: IntegratedConfig = None):
        """Initialize the integrated system"""
        self.config = config or IntegratedConfig()
        
        # Initialize components
        self.shape_selector = ShapeObjectSelector(self.config.canvas_size)
        
        # Default billboard config if not provided
        if self.config.billboard_config is None:
            self.config.billboard_config = BillboardConfig()
        
        self.billboard_generator = ProperAnamorphicBillboard(self.config.billboard_config)
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
    
    def create_custom_scene(self, shapes: List[ShapeConfig], 
                          objects: List[ObjectConfig] = None) -> Image.Image:
        """Create a custom scene from user-selected shapes and objects"""
        return self.shape_selector.create_scene_from_selection(shapes, objects)
    
    def create_preset_scene(self, preset_name: str) -> Image.Image:
        """Create a preset scene"""
        return self.shape_selector.create_preset_scene(preset_name)
    
    def generate_anamorphic_from_scene(self, scene_image: Image.Image, 
                                     scene_name: str = "custom") -> Dict[str, Image.Image]:
        """Generate anamorphic billboard from a scene image"""
        print(f"🎬 Generating anamorphic billboard for {scene_name}...")
        
        # Save original scene if requested
        if self.config.save_intermediate:
            scene_path = os.path.join(self.config.output_dir, f"{scene_name}_original.png")
            scene_image.save(scene_path)
            print(f"💾 Saved original scene: {scene_path}")
        
        # Generate anamorphic billboard
        results = self.billboard_generator.create_anamorphic_billboard(scene_image)
        
        # Save results
        output_files = {}
        for key, image in results.items():
            filename = f"{scene_name}_{key}.png"
            filepath = os.path.join(self.config.output_dir, filename)
            image.save(filepath)
            output_files[key] = filepath
            print(f"✅ Saved {key}: {filepath}")
        
        return results
    
    def create_shape_showcase(self) -> Dict[str, Image.Image]:
        """Create a showcase of all available shapes with anamorphic effect"""
        print("🎯 Creating comprehensive shape showcase...")
        
        # Create a scene with all major shapes
        shapes = [
            # Front row (closer)
            ShapeConfig(ShapeType.SPHERE, (200, 500), 80, (255, 100, 100), 0.9),
            ShapeConfig(ShapeType.CUBE, (400, 500), 90, (100, 255, 100), 0.85),
            ShapeConfig(ShapeType.PYRAMID, (600, 500), 85, (100, 100, 255), 0.8),
            ShapeConfig(ShapeType.CYLINDER, (800, 500), 70, (255, 255, 100), 0.75),
            
            # Middle row
            ShapeConfig(ShapeType.SPHERE, (300, 350), 60, (255, 0, 255), 0.6),
            ShapeConfig(ShapeType.CUBE, (500, 350), 70, (0, 255, 255), 0.55),
            ShapeConfig(ShapeType.PYRAMID, (700, 350), 65, (255, 128, 0), 0.5),
            
            # Back row (farther)
            ShapeConfig(ShapeType.CYLINDER, (400, 200), 50, (128, 255, 128), 0.3),
            ShapeConfig(ShapeType.SPHERE, (600, 200), 45, (255, 128, 255), 0.25),
        ]
        
        scene = self.create_custom_scene(shapes)
        return self.generate_anamorphic_from_scene(scene, "shape_showcase")
    
    def create_object_showcase(self) -> Dict[str, Image.Image]:
        """Create a showcase of different object types with anamorphic effect"""
        print("🏗️ Creating object type showcase...")
        
        # Create mixed scene with shapes and objects
        shapes = [
            ShapeConfig(ShapeType.CUBE, (200, 400), 60, (120, 120, 120), 0.4),  # Building base
            ShapeConfig(ShapeType.CUBE, (300, 350), 80, (140, 140, 140), 0.45),  # Building base
            ShapeConfig(ShapeType.SPHERE, (600, 300), 70, (100, 150, 255), 0.7),  # Decorative sphere
        ]
        
        objects = [
            ObjectConfig(ObjectType.CHARACTER, (500, 450), (25, 50), "vibrant", 0.9),
            ObjectConfig(ObjectType.VEHICLE, (700, 430), (80, 40), "neon", 0.85),
            ObjectConfig(ObjectType.BUILDING, (150, 300), (60, 120), "earth", 0.3),
        ]
        
        scene = self.create_custom_scene(shapes, objects)
        return self.generate_anamorphic_from_scene(scene, "object_showcase")
    
    def create_color_scheme_showcase(self) -> Dict[str, Dict[str, Image.Image]]:
        """Create showcases for different color schemes"""
        print("🌈 Creating color scheme showcases...")
        
        results = {}
        color_schemes = ['vibrant', 'pastel', 'neon', 'earth', 'monochrome']
        
        for scheme in color_schemes:
            print(f"  Creating {scheme} scheme...")
            
            # Create scene with consistent shapes but different color scheme
            shapes = [
                ShapeConfig(ShapeType.SPHERE, (300, 400), 80, 
                          self.shape_selector.color_schemes[scheme][0], 0.8),
                ShapeConfig(ShapeType.CUBE, (500, 400), 90, 
                          self.shape_selector.color_schemes[scheme][1], 0.7),
                ShapeConfig(ShapeType.PYRAMID, (700, 400), 85, 
                          self.shape_selector.color_schemes[scheme][2], 0.6),
                ShapeConfig(ShapeType.CYLINDER, (400, 250), 60, 
                          self.shape_selector.color_schemes[scheme][3], 0.5),
            ]
            
            scene = self.create_custom_scene(shapes)
            results[scheme] = self.generate_anamorphic_from_scene(scene, f"color_{scheme}")
        
        return results
    
    def create_depth_showcase(self) -> Dict[str, Image.Image]:
        """Create a showcase demonstrating depth effects"""
        print("📏 Creating depth effect showcase...")
        
        # Create scene with objects at different depths
        shapes = [
            # Far background
            ShapeConfig(ShapeType.CUBE, (200, 200), 40, (100, 100, 150), 0.1),
            ShapeConfig(ShapeType.SPHERE, (600, 180), 35, (150, 100, 100), 0.15),
            
            # Middle ground
            ShapeConfig(ShapeType.PYRAMID, (350, 350), 60, (100, 150, 100), 0.4),
            ShapeConfig(ShapeType.CYLINDER, (550, 380), 50, (150, 150, 100), 0.45),
            
            # Foreground
            ShapeConfig(ShapeType.SPHERE, (300, 550), 90, (255, 100, 100), 0.8),
            ShapeConfig(ShapeType.CUBE, (500, 520), 100, (100, 255, 100), 0.85),
            
            # Very close
            ShapeConfig(ShapeType.PYRAMID, (400, 650), 70, (255, 255, 100), 0.95),
        ]
        
        scene = self.create_custom_scene(shapes)
        return self.generate_anamorphic_from_scene(scene, "depth_showcase")
    
    def run_comprehensive_demo(self):
        """Run a comprehensive demonstration of the integrated system"""
        print("🚀 Starting Comprehensive Anamorphic Billboard Demo")
        print("=" * 60)
        
        # 1. Shape showcase
        print("\n1️⃣ SHAPE SHOWCASE")
        print("-" * 30)
        self.create_shape_showcase()
        
        # 2. Object showcase
        print("\n2️⃣ OBJECT SHOWCASE")
        print("-" * 30)
        self.create_object_showcase()
        
        # 3. Color scheme showcases
        print("\n3️⃣ COLOR SCHEME SHOWCASES")
        print("-" * 30)
        self.create_color_scheme_showcase()
        
        # 4. Depth showcase
        print("\n4️⃣ DEPTH EFFECT SHOWCASE")
        print("-" * 30)
        self.create_depth_showcase()
        
        # 5. Process original benchmark for comparison
        print("\n5️⃣ ORIGINAL BENCHMARK PROCESSING")
        print("-" * 30)
        try:
            benchmark_path = "../data/input/benchmark.jpg"
            if os.path.exists(benchmark_path):
                benchmark_image = Image.open(benchmark_path)
                self.generate_anamorphic_from_scene(benchmark_image, "benchmark_comparison")
            else:
                print("⚠️ Benchmark image not found")
        except Exception as e:
            print(f"⚠️ Error processing benchmark: {e}")
        
        print(f"\n🎉 Demo complete! All outputs saved to: {self.config.output_dir}")
        print("\n📋 SUMMARY:")
        print("=" * 60)
        print("✅ Shape Showcase - All geometric shapes with 3D anamorphic effect")
        print("✅ Object Showcase - Characters, buildings, vehicles with anamorphic transformation")
        print("✅ Color Schemes - 5 different color palettes with anamorphic billboards")
        print("✅ Depth Effects - Multi-layer depth demonstration")
        print("✅ Benchmark Comparison - Original benchmark with anamorphic effect")
        
        # List available options
        print(f"\n🎯 Available Shapes ({len(ShapeType)}):")
        for shape in ShapeType:
            print(f"   • {shape.value}")
        
        print(f"\n🏗️ Available Objects ({len(ObjectType)}):")
        for obj in ObjectType:
            print(f"   • {obj.value}")
        
        print(f"\n🌈 Available Color Schemes ({len(self.shape_selector.color_schemes)}):")
        for scheme in self.shape_selector.color_schemes.keys():
            print(f"   • {scheme}")

def main():
    """Main function to run the integrated system"""
    # Create configuration
    config = IntegratedConfig(
        canvas_size=(1434, 755),  # Match benchmark dimensions
        output_dir="integrated_anamorphic_output",
        save_intermediate=True
    )
    
    # Initialize system
    system = IntegratedAnamorphicSystem(config)
    
    # Run comprehensive demo
    system.run_comprehensive_demo()

if __name__ == "__main__":
    main() 