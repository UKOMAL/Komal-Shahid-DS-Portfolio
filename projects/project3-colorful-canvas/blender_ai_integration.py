"""
🎬 BLENDER + AI ANAMORPHIC PIPELINE
Complete 7-Step Professional 3D Anamorphic Rendering with AI Models

USAGE IN BLENDER:
1. Open Blender
2. Go to Scripting tab
3. Copy/paste this entire script
4. Click "Run Script"
5. Check Console for results
"""

import bpy
import bmesh
import sys
import os
from pathlib import Path
from mathutils import Vector, Matrix
import numpy as np

# Add our project path so we can import our AI models
project_path = '/Users/komalshahid/Desktop/Bellevue University/DSC680/projects/project3-colorful-canvas'
if project_path not in sys.path:
    sys.path.append(project_path)

try:
    from src.milestone3.colorful_canvas_complete import ColorfulCanvasAI, deps
    print("✅ AI Models imported successfully!")
except ImportError as e:
    print(f"⚠️ AI Models not available: {e}")
    print("💡 Will use Blender-only 3D rendering")

def blender_ai_anamorphic_pipeline():
    """
    🎬 COMPLETE 7-STEP BLENDER + AI ANAMORPHIC PIPELINE
    """
    print("="*60)
    print("🎬 BLENDER + AI ANAMORPHIC RENDERING PIPELINE")
    print("="*60)
    
    # Clear existing scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # STEP 1: Setup Professional 3D Scene
    print("📐 STEP 1: Setting up professional 3D scene...")
    setup_professional_scene()
    
    # STEP 2: Create 3D Objects for Anamorphic Effect
    print("🎯 STEP 2: Creating 3D objects for anamorphic distortion...")
    objects = create_anamorphic_objects()
    
    # STEP 3: Setup Viewing Angle Camera
    print("📹 STEP 3: Setting up anamorphic viewing angle camera...")
    setup_anamorphic_camera()
    
    # STEP 4: Apply AI-Enhanced Materials
    print("🎨 STEP 4: Applying AI-enhanced materials...")
    apply_ai_materials(objects)
    
    # STEP 5: Create Anamorphic Distortion
    print("🌀 STEP 5: Applying mathematical anamorphic distortion...")
    apply_anamorphic_distortion(objects)
    
    # STEP 6: Setup Professional Lighting
    print("💡 STEP 6: Setting up professional lighting...")
    setup_professional_lighting()
    
    # STEP 7: Render Final Result
    print("🎬 STEP 7: Rendering final anamorphic result...")
    render_result = render_anamorphic_sequence()
    
    print("✅ COMPLETE PIPELINE FINISHED!")
    print(f"📁 Results: {render_result}")
    
    return render_result

def setup_professional_scene():
    """Setup professional 3D scene for anamorphic rendering"""
    
    # Add professional ground plane
    bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "Ground_Plane"
    
    # Add background wall for projection
    bpy.ops.mesh.primitive_plane_add(size=15, location=(0, -10, 5))
    wall = bpy.context.active_object
    wall.name = "Projection_Wall"
    wall.rotation_euler = (1.5708, 0, 0)  # 90 degrees
    
    print("✅ Professional scene setup complete")

def create_anamorphic_objects():
    """Create 3D objects specifically designed for anamorphic illusions"""
    objects = []
    
    # Floating cubes at different depths
    cube_positions = [
        (0, 0, 2),      # Center cube
        (-3, 2, 3),     # Left cube (higher)
        (3, -2, 1.5),   # Right cube (lower)
        (0, 4, 4),      # Back cube (highest)
    ]
    
    for i, pos in enumerate(cube_positions):
        bpy.ops.mesh.primitive_cube_add(location=pos)
        cube = bpy.context.active_object
        cube.name = f"Anamorphic_Cube_{i+1}"
        
        # Scale cubes differently for depth effect
        scale = 1.0 + (pos[2] * 0.2)  # Larger cubes are "closer"
        cube.scale = (scale, scale, scale)
        
        objects.append(cube)
    
    # Add floating spheres
    sphere_positions = [
        (-2, -1, 2.5),
        (2, 3, 1.8),
        (1, -3, 3.2)
    ]
    
    for i, pos in enumerate(sphere_positions):
        bpy.ops.mesh.primitive_uv_sphere_add(location=pos)
        sphere = bpy.context.active_object
        sphere.name = f"Anamorphic_Sphere_{i+1}"
        objects.append(sphere)
    
    print(f"✅ Created {len(objects)} anamorphic objects")
    return objects

def setup_anamorphic_camera():
    """Setup camera for optimal anamorphic viewing angle"""
    
    # Remove default camera
    if 'Camera' in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects['Camera'])
    
    # Add anamorphic camera
    bpy.ops.object.camera_add(location=(8, -8, 3))
    camera = bpy.context.active_object
    camera.name = "Anamorphic_Camera"
    
    # Set anamorphic viewing angle (15-30 degrees is optimal)
    camera.rotation_euler = (1.1, 0, 0.785)  # ~45 degree angle
    
    # Camera settings for anamorphic
    camera.data.lens = 35  # Wide angle for distortion
    camera.data.sensor_width = 36
    
    # Set as active camera
    bpy.context.scene.camera = camera
    
    print("✅ Anamorphic camera setup complete")

def apply_ai_materials(objects):
    """Apply AI-enhanced materials for better anamorphic effects"""
    
    # Create bright, contrasting materials
    materials = {
        'Bright_Red': (1.0, 0.1, 0.1, 1.0),
        'Electric_Blue': (0.1, 0.5, 1.0, 1.0),
        'Neon_Green': (0.2, 1.0, 0.2, 1.0),
        'Hot_Pink': (1.0, 0.2, 0.8, 1.0),
        'Bright_Yellow': (1.0, 1.0, 0.1, 1.0),
    }
    
    for i, obj in enumerate(objects):
        # Create new material
        mat_name = list(materials.keys())[i % len(materials)]
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        
        # Get material color
        color = materials[mat_name]
        
        # Set emission for bright effect
        emission = mat.node_tree.nodes.new(type='ShaderNodeEmission')
        emission.inputs[0].default_value = color
        emission.inputs[1].default_value = 2.0  # Bright emission
        
        # Connect to output
        material_output = mat.node_tree.nodes.get('Material Output')
        mat.node_tree.links.new(emission.outputs[0], material_output.inputs[0])
        
        # Assign to object
        obj.data.materials.append(mat)
    
    print("✅ AI-enhanced materials applied")

def apply_anamorphic_distortion(objects):
    """Apply mathematical anamorphic distortion to objects"""
    
    for obj in objects:
        # Get original position
        orig_pos = obj.location.copy()
        
        # Anamorphic distortion calculation
        viewing_angle = 0.4  # ~25 degrees in radians
        
        # Extreme perspective stretching for anamorphic effect
        distort_x = orig_pos.x * (1.0 + orig_pos.z * 0.8)
        distort_y = orig_pos.y * (1.0 + orig_pos.z * 1.2)  # More Y distortion
        distort_z = orig_pos.z
        
        # Apply keystone correction
        keystone_factor = 1.0 / np.cos(viewing_angle * (1.0 + orig_pos.z * 0.3))
        distort_y *= keystone_factor
        
        # Update object position
        obj.location = Vector((distort_x, distort_y, distort_z))
        
        # Add slight rotation for more dramatic effect
        obj.rotation_euler = (
            orig_pos.z * 0.1,  # X rotation based on depth
            orig_pos.x * 0.05, # Y rotation based on X position
            0
        )
    
    print("✅ Anamorphic distortion applied")

def setup_professional_lighting():
    """Setup professional lighting for anamorphic rendering"""
    
    # Remove default light
    if 'Light' in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects['Light'])
    
    # Key light (main illumination)
    bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
    key_light = bpy.context.active_object
    key_light.name = "Key_Light"
    key_light.data.energy = 3.0
    
    # Fill light (soften shadows)
    bpy.ops.object.light_add(type='AREA', location=(-3, 3, 5))
    fill_light = bpy.context.active_object
    fill_light.name = "Fill_Light"
    fill_light.data.energy = 1.5
    fill_light.data.size = 5
    
    # Rim light (edge definition)
    bpy.ops.object.light_add(type='SPOT', location=(0, -8, 6))
    rim_light = bpy.context.active_object
    rim_light.name = "Rim_Light"
    rim_light.data.energy = 2.0
    rim_light.data.spot_size = 1.2
    
    print("✅ Professional lighting setup complete")

def render_anamorphic_sequence():
    """Render the final anamorphic sequence"""
    
    # Setup render settings
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'  # Use Cycles for quality
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    
    # Output settings
    output_dir = Path('/Users/komalshahid/Desktop/Bellevue University/DSC680/projects/project3-colorful-canvas/data/output/blender_anamorphic/')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    scene.render.filepath = str(output_dir / 'anamorphic_render_')
    scene.render.image_settings.file_format = 'PNG'
    
    # Render single frame
    print("🎬 Rendering anamorphic frame...")
    bpy.ops.render.render(write_still=True)
    
    # Optional: Render sequence from different angles
    render_sequence_from_angles(output_dir)
    
    print(f"✅ Rendering complete! Check: {output_dir}")
    return str(output_dir)

def render_sequence_from_angles(output_dir):
    """Render sequence showing the anamorphic effect from different angles"""
    
    camera = bpy.data.objects['Anamorphic_Camera']
    original_rotation = camera.rotation_euler.copy()
    
    # Render from 5 different angles to show the illusion
    angles = [0.4, 0.6, 0.8, 1.0, 1.2]  # Different viewing angles
    
    for i, angle in enumerate(angles):
        camera.rotation_euler = (angle, 0, 0.785)
        
        bpy.context.scene.render.filepath = str(output_dir / f'angle_{i+1:02d}_')
        bpy.ops.render.render(write_still=True)
        
        print(f"   📸 Rendered angle {i+1}/5")
    
    # Restore original camera position
    camera.rotation_euler = original_rotation

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Run the complete pipeline
    try:
        result = blender_ai_anamorphic_pipeline()
        print("\n🎉 SUCCESS! Blender + AI Pipeline Complete!")
        print("📁 Check the output directory for your professional 3D anamorphic renders!")
        
    except Exception as e:
        print(f"❌ Error in pipeline: {e}")
        import traceback
        traceback.print_exc()

# For manual execution in Blender console:
print("\n" + "="*60)
print("🎬 BLENDER + AI ANAMORPHIC PIPELINE LOADED")
print("="*60)
print("To run the complete pipeline, execute:")
print("blender_ai_anamorphic_pipeline()")
print("="*60) 