#!/usr/bin/env python3
"""
Test Intelligent Camera Positioning
"""

import bpy
import sys
import os
from pathlib import Path

# Add src to path
src_path = str(Path(__file__).parent.parent / "src" / "blender")
if src_path not in sys.path:
    sys.path.append(src_path)

from anamorphic_billboard_consolidated import render_from_camera, step_9_intelligent_camera_render

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

def test_intelligent_camera():
    """Test intelligent camera positioning"""
    print("🧪 Testing Intelligent Camera Positioning")
    
    # Clear scene
    clear_scene()
    
    # Create billboard
    bpy.ops.mesh.primitive_plane_add(size=8, location=(0, 5, 4))
    billboard = bpy.context.active_object
    billboard.name = "Test_Billboard"
    billboard.rotation_euler = (1.57, 0, 0)  # 90 degrees
    
    # Load Bulgari image
    image_path = "data/input/Image.jpeg"
    if os.path.exists(image_path):
        image = bpy.data.images.load(image_path)
        print(f"✅ Image loaded: {image.name} ({image.size[0]}x{image.size[1]})")
        
        # Apply material
        mat = bpy.data.materials.new("Test_Material")
        mat.use_nodes = True
        billboard.data.materials.append(mat)
        
        nodes = mat.node_tree.nodes
        nodes.clear()
        
        output = nodes.new('ShaderNodeOutputMaterial')
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')  
        img_node = nodes.new('ShaderNodeTexImage')
        img_node.image = image
        
        mat.node_tree.links.new(img_node.outputs['Color'], bsdf.inputs['Base Color'])
        mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
        
    # Create camera
    bpy.ops.object.camera_add(location=(10, -10, 5))
    camera = bpy.context.active_object
    camera.name = "Anamorphic_Camera"
    
    # Test intelligent positioning and rendering
    output_path = "output/INTELLIGENT_bulgari_test.png"
    step_9_intelligent_camera_render("Test_Billboard", output_path)
    
    print("✅ Intelligent camera test complete!")

if __name__ == "__main__":
    test_intelligent_camera() 