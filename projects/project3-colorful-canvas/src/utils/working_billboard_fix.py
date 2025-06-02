#!/usr/bin/env python3
"""
Debug Billboard Texture - Simple Test
Find out why our billboard textures are not showing up
"""

import bpy
import os
import sys

def clear_scene():
    """Clear everything"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for mesh in list(bpy.data.meshes): 
        bpy.data.meshes.remove(mesh)
    for img in list(bpy.data.images):  
        bpy.data.images.remove(img)
    for mat in list(bpy.data.materials):
        bpy.data.materials.remove(mat)
    print("🧹 Scene cleared")

def create_simple_textured_plane(image_path, output_path):
    """Create the simplest possible textured plane"""
    print(f"🔧 Creating simple textured plane with {image_path}")
    
    # 1. Create a simple plane
    bpy.ops.mesh.primitive_plane_add(size=4, location=(0, 0, 0))
    plane = bpy.context.active_object
    plane.name = "Test_Plane"
    
    # 2. Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ ERROR: Image not found: {image_path}")
        return False
    
    # 3. Load the image
    try:
        image = bpy.data.images.load(image_path)
        print(f"✅ Image loaded: {image.name} ({image.size[0]}x{image.size[1]})")
    except Exception as e:
        print(f"❌ ERROR loading image: {e}")
        return False
    
    # 4. Create material
    material = bpy.data.materials.new(name="Test_Material")
    material.use_nodes = True
    plane.data.materials.append(material)
    
    # 5. Clear default nodes and add simple setup
    nodes = material.node_tree.nodes
    nodes.clear()
    
    # Create output node
    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (400, 0)
    
    # Create emission shader (bright and visible)
    emission = nodes.new(type='ShaderNodeEmission')
    emission.location = (200, 0)
    emission.inputs['Strength'].default_value = 3.0  # Bright
    
    # Create image texture node
    image_texture = nodes.new(type='ShaderNodeTexImage')
    image_texture.location = (0, 0)
    image_texture.image = image
    
    # Create texture coordinate node
    tex_coord = nodes.new(type='ShaderNodeTexCoord')
    tex_coord.location = (-200, 0)
    
    # Connect nodes
    links = material.node_tree.links
    links.new(tex_coord.outputs['UV'], image_texture.inputs['Vector'])
    links.new(image_texture.outputs['Color'], emission.inputs['Color'])
    links.new(emission.outputs['Emission'], output.inputs['Surface'])
    
    print("✅ Material and nodes created")
    
    # 6. Create camera
    bpy.ops.object.camera_add(location=(5, -5, 3))
    camera = bpy.context.active_object
    camera.rotation_euler = (1.1, 0, 0.785)  # Look at plane
    bpy.context.scene.camera = camera
    
    # 7. Add lighting
    bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
    light = bpy.context.active_object
    light.data.energy = 5
    
    # 8. Set render settings
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.render.resolution_x = 800
    scene.render.resolution_y = 600
    scene.cycles.samples = 64
    
    # 9. Render
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    
    print(f"✅ Simple test render saved: {output_path}")
    return True

def main():
    """Run the debug test"""
    if len(sys.argv) < 2:
        print("Usage: blender --background --python debug_billboard_texture.py -- <image_path> <output_path>")
        return
    
    # Get arguments after --
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    
    if len(args) < 2:
        print("Need image path and output path")
        return
    
    image_path = args[0]
    output_path = args[1]
    
    print("🚀 DEBUG: Testing Billboard Texture")
    print(f"📷 Input: {image_path}")
    print(f"💾 Output: {output_path}")
    
    # Clear and test
    clear_scene()
    success = create_simple_textured_plane(image_path, output_path)
    
    if success:
        print("✅ DEBUG TEST PASSED - Image should be visible")
    else:
        print("❌ DEBUG TEST FAILED - Problem with texture system")

if __name__ == "__main__":
    main() 