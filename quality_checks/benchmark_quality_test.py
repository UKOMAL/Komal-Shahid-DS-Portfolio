#!/usr/bin/env python3
"""
Benchmark Quality Test - Match the stunning 3D pop-out billboard example
Uses professional 3D displacement, cinema lighting, and high-quality rendering
"""

import bpy
import sys
import os
from pathlib import Path

# Add src to path
src_path = str(Path(__file__).parent.parent / "src" / "blender")
if src_path not in sys.path:
    sys.path.append(src_path)

from anamorphic_billboard_consolidated import (
    clear_scene, 
    create_billboard_environment,
    create_billboard_geometry, 
    create_3d_popout_effect,
    setup_professional_lighting,
    setup_cinema_quality_render,
    render_from_camera
)

def create_benchmark_quality_billboard(image_path: str, output_path: str):
    """
    Create a billboard that matches the benchmark quality with 3D pop-out effects
    """
    print("🎯 CREATING BENCHMARK QUALITY BILLBOARD")
    print("="*60)
    
    # Clear scene
    clear_scene()
    
    # Step 1: Create professional environment
    print("🏗️ Step 1: Creating professional environment...")
    env_objects = create_billboard_environment(
        ground_size=40.0,
        building_height=12.0,
        building_width=8.0
    )
    
    # Step 2: Create high-detail billboard geometry
    print("📐 Step 2: Creating high-detail billboard geometry...")
    billboard_objects = create_billboard_geometry(
        width=10.0,
        height=6.0,
        location=(0, 5, 3),
        subdivision_levels=2,  # Start with base subdivision
        validate_geometry=True
    )
    
    # Step 3: Apply 3D POP-OUT EFFECT
    print("🎭 Step 3: Applying 3D POP-OUT EFFECT...")
    popout_material = create_3d_popout_effect(
        billboard_object="Billboard_Main",
        image_path=image_path,
        displacement_strength=1.5,  # Strong pop-out like benchmark
        subdivision_levels=4,       # High detail for smooth displacement
        use_depth_estimation=True
    )
    
    # Step 4: Setup professional cinema lighting
    print("💡 Step 4: Setting up professional cinema lighting...")
    lights = setup_professional_lighting(
        key_light_energy=1200.0,   # Bright key light
        fill_light_energy=400.0,   # Strong fill
        rim_light_energy=600.0,    # Dramatic rim
        environment_strength=0.2   # Subtle environment
    )
    
    # Step 5: Configure cinema-quality rendering
    print("🎬 Step 5: Configuring cinema-quality rendering...")
    render_settings = setup_cinema_quality_render(
        samples=256,               # High quality samples
        resolution_x=1920,
        resolution_y=1080,
        use_denoising=True,
        use_motion_blur=False
    )
    
    # Step 6: Position camera intelligently
    print("📷 Step 6: Positioning camera for optimal view...")
    bpy.ops.object.camera_add(location=(8, -8, 4))
    camera = bpy.context.active_object
    camera.name = "Benchmark_Camera"
    
    # Step 7: Render final benchmark-quality result
    print("🎯 Step 7: Rendering BENCHMARK QUALITY billboard...")
    render_from_camera(
        camera_name="Benchmark_Camera",
        output_path=output_path,
        obj_to_frame="Billboard_Main",
        margin=1.1,                # Tight framing
        resolution_x=1920,
        resolution_y=1080,
        resolution_percent=100,
        file_format='PNG'
    )
    
    print("🏆 BENCHMARK QUALITY BILLBOARD COMPLETE!")
    print(f"✅ Output: {output_path}")
    print(f"✅ Features: 3D Pop-out, Professional Lighting, Cinema Quality")
    
    return {
        'environment': env_objects,
        'billboard': billboard_objects, 
        'material': popout_material,
        'lights': lights,
        'render': render_settings,
        'output': output_path
    }

if __name__ == "__main__":
    if len(sys.argv) > 4:
        image_path = sys.argv[4]  # After --
        output_path = sys.argv[5]
        create_benchmark_quality_billboard(image_path, output_path)
    else:
        # Default test
        create_benchmark_quality_billboard(
            "data/input/Image.jpeg", 
            "output/BENCHMARK_QUALITY_bulgari.png"
        ) 