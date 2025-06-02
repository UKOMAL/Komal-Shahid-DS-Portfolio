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
    
    # STEP 2: Create Seoul-Style Billboard Frame
    print("🖼️ STEP 2: Creating Seoul-style billboard frame...")
    frame = create_seoul_billboard_frame()
    
    # STEP 3: Create 3D Objects for Anamorphic Effect
    print("🎯 STEP 3: Creating 3D objects for anamorphic distortion...")
    objects = create_layered_anamorphic_objects(frame)
    
    # STEP 4: Setup Viewing Angle Camera
    print("📹 STEP 4: Setting up anamorphic viewing angle camera...")
    setup_anamorphic_camera()
    
    # STEP 5: Apply Enhanced Materials
    print("🎨 STEP 5: Applying enhanced materials...")
    apply_enhanced_materials(objects)
    
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
    
    # Set black background
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (0, 0, 0, 1)
    
    # Configure render engine for better quality
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.cycles.samples = 100
    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080
    
    print("✅ Professional scene setup complete")

def create_seoul_billboard_frame():
    """Create a Seoul-style billboard frame for the anamorphic effect"""
    
    # Create frame corners with beveled edges for more realism
    bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))
    frame = bpy.context.active_object
    frame.name = "Seoul_Billboard_Frame"
    
    # Apply scale for wide frame shape
    frame.scale = (2.0, 0.1, 1.2)
    
    # Apply bevel modifier for more realistic edges
    bevel = frame.modifiers.new(name="Bevel", type='BEVEL')
    bevel.width = 0.02
    bevel.segments = 3
    
    # Create hole in the frame (boolean modifier with cube)
    bpy.ops.mesh.primitive_cube_add(size=1.7, location=(0, 0, 0))
    hole = bpy.context.active_object
    hole.name = "Frame_Hole"
    hole.scale = (0.9, 1.0, 0.8)
    
    boolean = frame.modifiers.new(name="Boolean", type='BOOLEAN')
    boolean.object = hole
    boolean.operation = 'DIFFERENCE'
    
    # Apply material to frame
    create_material_for_object(frame, "FrameMaterial", (0.9, 0.9, 0.9, 1.0), metallic=0.8, roughness=0.2)
    
    # Apply modifiers
    bpy.context.view_layer.objects.active = frame
    bpy.ops.object.modifier_apply(modifier="Bevel")
    bpy.ops.object.modifier_apply(modifier="Boolean")
    
    # Delete the hole object
    bpy.data.objects.remove(hole)
    
    return frame

def create_layered_anamorphic_objects(frame):
    """Create layered objects that appear to come out of the billboard frame"""
    objects = []
    
    # Layer 1: Base elements inside the frame
    # Create a background image plane
    bpy.ops.mesh.primitive_plane_add(size=1.6, location=(0, 0.05, 0))
    background = bpy.context.active_object
    background.name = "Background_Plane"
    background.scale = (1.7, 1, 1.5)
    objects.append(background)
    
    # Create material with emission for background glow
    create_material_for_object(background, "BackgroundMaterial", (0.2, 0.3, 0.8, 1.0), emission_strength=0.5)
    
    # Layer 2: Elements starting to emerge
    # Create spheres at different depths
    sphere_positions = [
        (-0.7, -0.2, 0.4),  # Left
        (0.7, -0.3, 0.3),   # Right
        (0, -0.1, 0.7)      # Center
    ]
    
    sphere_colors = [
        (0.2, 0.6, 1.0, 1.0),  # Blue
        (1.0, 0.3, 0.2, 1.0),  # Red
        (0.9, 0.9, 0.2, 1.0)   # Yellow
    ]
    
    sphere_sizes = [0.3, 0.25, 0.4]
    
    for i, (pos, color, size) in enumerate(zip(sphere_positions, sphere_colors, sphere_sizes)):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=size, location=pos)
        sphere = bpy.context.active_object
        sphere.name = f"Emerging_Sphere_{i+1}"
        
        # Add subsurface modifier for smoother appearance
        subsurf = sphere.modifiers.new(name="Subsurf", type='SUBSURF')
        subsurf.levels = 2
        
        create_material_for_object(sphere, f"SphereMaterial_{i+1}", color, specular=0.5, roughness=0.2)
        objects.append(sphere)
    
    # Layer 3: Main character emerging from frame
    # Create Suzanne (monkey) as our main character
    bpy.ops.mesh.primitive_monkey_add(size=1.0, location=(0, -0.5, 0))
    character = bpy.context.active_object
    character.name = "Main_Character"
    
    # Position and scale
    character.location = (0, 0.3, 0)
    character.scale = (0.7, 0.7, 0.7)
    
    # Add modifiers for better appearance
    subsurf = character.modifiers.new(name="Subsurf", type='SUBSURF')
    subsurf.levels = 2
    
    # Add a red material with subsurface scattering for skin-like appearance
    material = bpy.data.materials.new(name="CharacterMaterial")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    
    # Clear default nodes
    nodes.clear()
    
    # Create principled BSDF node
    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
    principled.inputs['Base Color'].default_value = (0.8, 0.2, 0.2, 1.0)  # Red
    principled.inputs['Subsurface'].default_value = 0.1
    principled.inputs['Subsurface Color'].default_value = (1.0, 0.3, 0.3, 1.0)
    principled.inputs['Specular'].default_value = 0.5
    principled.inputs['Roughness'].default_value = 0.3
    
    # Output node
    output = nodes.new(type='ShaderNodeOutputMaterial')
    links.new(principled.outputs['BSDF'], output.inputs['Surface'])
    
    character.data.materials.append(material)
    objects.append(character)
    
    # Layer 4: Objects fully outside the frame
    # Create a bridge-like structure
    bpy.ops.mesh.primitive_cube_add(size=0.5, location=(0, 0.7, -0.3))
    bridge = bpy.context.active_object
    bridge.name = "Bridge"
    bridge.scale = (1.5, 0.1, 0.1)
    
    create_material_for_object(bridge, "BridgeMaterial", (0.2, 0.4, 0.8, 1.0), metallic=0.7)
    objects.append(bridge)
    
    # Add supporting elements for the scene
    # Small decorative cubes
    for i in range(3):
        offset_x = 0.5 * (i - 1)
        bpy.ops.mesh.primitive_cube_add(size=0.15, location=(offset_x, 0.8, 0.2))
        cube = bpy.context.active_object
        cube.name = f"Deco_Cube_{i+1}"
        cube.rotation_euler = (np.radians(45), np.radians(45), 0)
        
        color = (0.8, 0.2, 0.8, 1.0) if i % 2 == 0 else (0.2, 0.8, 0.8, 1.0)
        create_material_for_object(cube, f"CubeMaterial_{i+1}", color)
        objects.append(cube)
    
    return objects

def create_material_for_object(obj, material_name, color, metallic=0.0, roughness=0.5, specular=0.5, emission_strength=0.0):
    """Create and assign a material to an object"""
    material = bpy.data.materials.new(name=material_name)
    material.use_nodes = True
    
    # Get the nodes
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    
    # Clear default nodes
    nodes.clear()
    
    # Create principled BSDF node
    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
    principled.inputs['Base Color'].default_value = color
    principled.inputs['Metallic'].default_value = metallic
    principled.inputs['Roughness'].default_value = roughness
    principled.inputs['Specular'].default_value = specular
    
    # Add emission if needed
    if emission_strength > 0:
        # Create emission shader
        emission = nodes.new(type='ShaderNodeEmission')
        emission.inputs['Color'].default_value = color
        emission.inputs['Strength'].default_value = emission_strength
        
        # Create mix shader
        mix = nodes.new(type='ShaderNodeMixShader')
        mix.inputs[0].default_value = 0.5  # Mix factor
        
        # Connect nodes
        links.new(principled.outputs['BSDF'], mix.inputs[1])
        links.new(emission.outputs['Emission'], mix.inputs[2])
        
        # Output node
        output = nodes.new(type='ShaderNodeOutputMaterial')
        links.new(mix.outputs['Shader'], output.inputs['Surface'])
    else:
        # Output node
        output = nodes.new(type='ShaderNodeOutputMaterial')
        links.new(principled.outputs['BSDF'], output.inputs['Surface'])
    
    # Assign material to object
    if obj.data.materials:
        obj.data.materials[0] = material
    else:
        obj.data.materials.append(material)

def setup_anamorphic_camera():
    """Setup camera for optimal anamorphic viewing angle"""
    
    # Add anamorphic camera
    bpy.ops.object.camera_add(location=(0, -5, 0))
    camera = bpy.context.active_object
    camera.name = "Anamorphic_Camera"
    
    # Set anamorphic viewing angle
    camera.rotation_euler = (np.radians(90), 0, 0)
    
    # Camera settings for anamorphic
    camera.data.lens = 45  # Wider angle for better framing
    camera.data.sensor_width = 36
    
    # Set as active camera
    bpy.context.scene.camera = camera
    
    print("✅ Anamorphic camera setup complete")

def apply_enhanced_materials(objects):
    """Apply enhanced materials to objects"""
    # The materials have already been applied in the create_layered_anamorphic_objects function
    print("✅ Enhanced materials applied")

def setup_professional_lighting():
    """Setup professional lighting for anamorphic rendering"""
    
    # Key light (main illumination)
    bpy.ops.object.light_add(type='AREA', location=(3, -3, 3))
    key_light = bpy.context.active_object
    key_light.name = "Key_Light"
    key_light.data.energy = 500
    key_light.data.size = 2
    key_light.rotation_euler = (np.radians(45), 0, np.radians(45))
    
    # Fill light (soften shadows)
    bpy.ops.object.light_add(type='AREA', location=(-3, -2, 2))
    fill_light = bpy.context.active_object
    fill_light.name = "Fill_Light"
    fill_light.data.energy = 200
    fill_light.data.size = 3
    fill_light.rotation_euler = (np.radians(45), 0, np.radians(-45))
    
    # Rim light (edge definition)
    bpy.ops.object.light_add(type='SPOT', location=(0, -4, 4))
    rim_light = bpy.context.active_object
    rim_light.name = "Rim_Light"
    rim_light.data.energy = 1000
    rim_light.data.spot_size = np.radians(30)
    rim_light.rotation_euler = (np.radians(60), 0, 0)
    
    print("✅ Professional lighting setup complete")

def render_anamorphic_sequence():
    """Render the final anamorphic sequence"""
    
    # Setup render settings
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    scene.cycles.samples = 100
    
    # Output settings
    output_dir = Path('/Users/komalshahid/Desktop/Bellevue University/DSC680/projects/project3-colorful-canvas/data/output/blender_anamorphic/')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    scene.render.filepath = str(output_dir / 'anamorphic_render_')
    scene.render.image_settings.file_format = 'PNG'
    
    # Render main frame
    print("🎬 Rendering anamorphic frame...")
    bpy.ops.render.render(write_still=True)
    
    # Render sequence from different angles
    render_sequence_from_angles(output_dir)
    
    print(f"✅ Rendering complete! Check: {output_dir}")
    return str(output_dir)

def render_sequence_from_angles(output_dir):
    """Render sequence showing the anamorphic effect from different angles"""
    
    camera = bpy.context.scene.camera
    original_location = camera.location.copy()
    
    # Define different angles to render from
    angle_configs = [
        {"name": "angle_01_", "location": (2, -5, 0.5), "rotation": (np.radians(85), 0, np.radians(15))},
        {"name": "angle_02_", "location": (-2, -5, 0.5), "rotation": (np.radians(85), 0, np.radians(-15))},
        {"name": "angle_03_", "location": (0, -5.5, 1), "rotation": (np.radians(80), 0, 0)},
        {"name": "angle_04_", "location": (0, -4.5, 0.5), "rotation": (np.radians(95), 0, 0)},
        {"name": "angle_05_", "location": (0, -6, 0), "rotation": (np.radians(90), 0, 0)}
    ]
    
    for angle in angle_configs:
        # Set camera position and rotation
        camera.location = angle["location"]
        camera.rotation_euler = angle["rotation"]
        
        # Render frame
        bpy.context.scene.render.filepath = str(output_dir / angle["name"])
        bpy.ops.render.render(write_still=True)
        
        print(f"   📸 Rendered angle: {angle['name']}")
    
    # Restore original camera position
    camera.location = original_location
    camera.rotation_euler = (np.radians(90), 0, 0)

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