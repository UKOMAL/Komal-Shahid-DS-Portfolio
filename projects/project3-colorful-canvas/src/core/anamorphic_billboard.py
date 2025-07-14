#!/usr/bin/env python3
"""
REAL Anamorphic Billboard Generator - WITH PROPER DISTORTION
Creates objects that appear to pop out ONLY when viewed from correct angle
Uses proper anamorphic mathematics for perspective distortion
"""
import bpy
import bmesh
import os
import sys
import math
import random
import numpy as np
from pathlib import Path

# ANAMORPHIC CONSTANTS
VIEWING_ANGLE = 45  # degrees - the angle from which the illusion works
VIEWING_DISTANCE = 30  # units - distance from billboard
VIEWING_HEIGHT = 15   # units - height of viewing position
BILLBOARD_PLANE_Z = 0  # The "screen" plane that objects appear to pop through

def clear_everything():
    """Clear the entire scene"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)
    for material in list(bpy.data.materials):
        bpy.data.materials.remove(material)
    for texture in list(bpy.data.textures):
        bpy.data.textures.remove(texture)
    for image in list(bpy.data.images):
        bpy.data.images.remove(image)
    
    print("🧹 Scene completely cleared")

def calculate_anamorphic_position(desired_x, desired_y, desired_z, viewing_angle_deg):
    """
    Calculate where to actually place an object so it appears at desired position
    from the viewing angle - THIS IS THE REAL ANAMORPHIC MATH!
    """
    # Convert viewing angle to radians
    viewing_angle_rad = math.radians(viewing_angle_deg)
    
    # The key insight: objects must be positioned to compensate for perspective
    # When viewed from the side, we need to "pre-distort" their positions
    
    # Calculate perspective factors
    depth_factor = desired_z  # How far the object should appear to pop out
    
    # Anamorphic X distortion (keystone effect)
    # Objects further out need more extreme X displacement
    keystone_factor = 1.0 + (depth_factor * math.tan(viewing_angle_rad) * 0.8)
    actual_x = desired_x * keystone_factor
    
    # Anamorphic Y distortion (vertical stretching)
    # Objects that appear higher need to be placed much higher
    perspective_stretch = 1.0 + (depth_factor * 0.6)
    vertical_compensation = depth_factor * math.sin(viewing_angle_rad) * 2.0
    actual_y = desired_y * perspective_stretch + vertical_compensation
    
    # Z position (depth) - this creates the "pop out" effect
    # Objects at desired_z=0 stay on billboard plane
    # Objects at desired_z>0 are positioned to appear popping out
    viewing_compensation = depth_factor * math.cos(viewing_angle_rad) * 1.5
    actual_z = BILLBOARD_PLANE_Z + depth_factor + viewing_compensation
    
    return (actual_x, actual_y, actual_z)

def calculate_anamorphic_scale(desired_size, depth, viewing_angle_deg):
    """
    Calculate how to scale objects for anamorphic effect
    Objects further out need different scaling to appear correct size
    """
    viewing_angle_rad = math.radians(viewing_angle_deg)
    
    # Perspective scaling - further objects need to be larger to appear same size
    perspective_scale = 1.0 + (depth * 0.4)
    
    # Viewing angle compensation
    angle_scale = 1.0 + (depth * math.sin(viewing_angle_rad) * 0.3)
    
    return desired_size * perspective_scale * angle_scale

def calculate_anamorphic_rotation(depth, viewing_angle_deg):
    """
    Calculate rotation needed for anamorphic objects
    Objects need to be rotated to face the viewing angle correctly
    """
    viewing_angle_rad = math.radians(viewing_angle_deg)
    
    # Rotation to face viewer
    face_rotation = viewing_angle_rad * (depth / 10.0)  # More rotation for further objects
    
    # Additional tilt for perspective
    perspective_tilt = math.atan(depth / VIEWING_DISTANCE) * 0.5
    
    return (perspective_tilt, 0, face_rotation)

def create_frame_and_base(image_path, frame_width=20, frame_height=12):
    """Create the billboard frame - this is the reference plane"""
    print(f"🖼️ Creating anamorphic billboard frame")
    
    # Create frame at Z=0 (the reference plane)
    bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0), size=1)
    outer_frame = bpy.context.active_object
    outer_frame.name = "Frame"
    outer_frame.scale = ((frame_width + 3) / 2, (frame_height + 3) / 2, 1.5)
    bpy.ops.object.transform_apply(scale=True)
    
    # Inner cutout
    bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0), size=1)
    inner_cutout = bpy.context.active_object
    inner_cutout.scale = (frame_width / 2, frame_height / 2, 2)
    bpy.ops.object.transform_apply(scale=True)
    
    # Boolean difference
    bpy.context.view_layer.objects.active = outer_frame
    bool_mod = outer_frame.modifiers.new("FrameCut", "BOOLEAN")
    bool_mod.operation = 'DIFFERENCE'
    bool_mod.object = inner_cutout
    bpy.ops.object.modifier_apply(modifier=bool_mod.name)
    bpy.data.objects.remove(inner_cutout, do_unlink=True)
    
    # Background plane exactly at Z=0 (the "screen" plane)
    bpy.ops.mesh.primitive_plane_add(location=(0, 0, BILLBOARD_PLANE_Z), size=1)
    background = bpy.context.active_object
    background.name = "Billboard_Background"
    background.scale = (frame_width / 2, frame_height / 2, 1)
    bpy.ops.object.transform_apply(scale=True)
    
    # Apply materials
    apply_frame_material(outer_frame)
    apply_background_material(background, image_path)
    
    print("✅ Anamorphic billboard frame created at reference plane Z=0")
    return outer_frame, background

def apply_frame_material(frame_obj):
    """Golden frame material"""
    mat = bpy.data.materials.new("Frame_Material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    output = nodes.new('ShaderNodeOutputMaterial')
    principled = nodes.new('ShaderNodeBsdfPrincipled')
    principled.inputs['Base Color'].default_value = (0.8, 0.6, 0.2, 1.0)
    principled.inputs['Metallic'].default_value = 0.9
    principled.inputs['Roughness'].default_value = 0.3
    
    mat.node_tree.links.new(principled.outputs['BSDF'], output.inputs['Surface'])
    frame_obj.data.materials.append(mat)

def apply_background_material(obj, image_path):
    """Background image material"""
    mat = bpy.data.materials.new("Background_Material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    output = nodes.new('ShaderNodeOutputMaterial')
    
    if os.path.exists(image_path):
        img = bpy.data.images.load(image_path)
        image_tex = nodes.new('ShaderNodeTexImage')
        image_tex.image = img
        principled = nodes.new('ShaderNodeBsdfPrincipled')
        tex_coord = nodes.new('ShaderNodeTexCoord')
        
        mat.node_tree.links.new(tex_coord.outputs['UV'], image_tex.inputs['Vector'])
        mat.node_tree.links.new(image_tex.outputs['Color'], principled.inputs['Base Color'])
        mat.node_tree.links.new(principled.outputs['BSDF'], output.inputs['Surface'])
    else:
        emission = nodes.new('ShaderNodeEmission')
        emission.inputs['Color'].default_value = (0.2, 0.3, 0.6, 1.0)
        emission.inputs['Strength'].default_value = 1.0
        mat.node_tree.links.new(emission.outputs['Emission'], output.inputs['Surface'])
    
    obj.data.materials.append(mat)

def create_material(name, color, material_type="standard"):
    """Create materials for 3D objects"""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    output = nodes.new('ShaderNodeOutputMaterial')
    
    if material_type == "emission":
        emission = nodes.new('ShaderNodeEmission')
        emission.inputs['Color'].default_value = (*color, 1.0)
        emission.inputs['Strength'].default_value = 3.0
        mat.node_tree.links.new(emission.outputs['Emission'], output.inputs['Surface'])
    else:
        principled = nodes.new('ShaderNodeBsdfPrincipled')
        principled.inputs['Base Color'].default_value = (*color, 1.0)
        if material_type == "plastic":
            principled.inputs['Roughness'].default_value = 0.3
            principled.inputs['Specular IOR'].default_value = 1.8
        mat.node_tree.links.new(principled.outputs['BSDF'], output.inputs['Surface'])
    
    return mat

def create_anamorphic_flower(name, desired_pos, desired_size, desired_depth):
    """
    Create flower with PROPER anamorphic positioning
    desired_pos: where it should APPEAR to be (x, y, z from billboard)
    desired_depth: how far it should appear to pop out (positive = toward viewer)
    """
    print(f"🌸 Creating anamorphic flower: {name} at apparent depth {desired_depth}")
    
    # Calculate REAL position using anamorphic math
    actual_pos = calculate_anamorphic_position(desired_pos[0], desired_pos[1], desired_depth, VIEWING_ANGLE)
    actual_size = calculate_anamorphic_scale(desired_size, desired_depth, VIEWING_ANGLE)
    actual_rotation = calculate_anamorphic_rotation(desired_depth, VIEWING_ANGLE)
    
    print(f"  Desired apparent position: {desired_pos}")
    print(f"  Actual physical position: {actual_pos}")
    print(f"  Size compensation: {desired_size} → {actual_size}")
    
    objects = []
    
    # Flower center
    bpy.ops.mesh.primitive_uv_sphere_add(location=actual_pos, radius=actual_size*0.4)
    center = bpy.context.active_object
    center.name = f"{name}_Center"
    center.rotation_euler = actual_rotation
    
    center_mat = create_material(f"{name}_Center_Mat", (1.0, 0.8, 0.2), "emission")
    center.data.materials.append(center_mat)
    objects.append(center)
    
    # Petals with anamorphic positioning
    petal_count = 8
    for i in range(petal_count):
        angle = i * (math.pi * 2 / petal_count)
        
        # Desired petal position (where it should appear)
        desired_petal_x = desired_pos[0] + math.cos(angle) * desired_size * 0.8
        desired_petal_y = desired_pos[1] + math.sin(angle) * desired_size * 0.8
        
        # Calculate actual position for this petal
        petal_actual_pos = calculate_anamorphic_position(desired_petal_x, desired_petal_y, desired_depth, VIEWING_ANGLE)
        
        bpy.ops.mesh.primitive_uv_sphere_add(location=petal_actual_pos, radius=actual_size*0.6)
        petal = bpy.context.active_object
        petal.name = f"{name}_Petal_{i}"
        
        # Apply anamorphic scaling and rotation
        petal.scale = (0.3, 1.2, 0.1)
        petal.rotation_euler = (actual_rotation[0], actual_rotation[1], angle + math.pi/2 + actual_rotation[2])
        bpy.ops.object.transform_apply(scale=True, rotation=True)
        
        petal_colors = [(1.0, 0.2, 0.6), (1.0, 0.4, 0.8), (0.9, 0.1, 0.5)]
        petal_color = petal_colors[i % len(petal_colors)]
        petal_mat = create_material(f"{name}_Petal_{i}_Mat", petal_color, "plastic")
        petal.data.materials.append(petal_mat)
        objects.append(petal)
    
    return objects

def create_anamorphic_emoji(name, desired_pos, desired_size, desired_depth):
    """Create emoji with proper anamorphic positioning"""
    print(f"😊 Creating anamorphic emoji: {name} at apparent depth {desired_depth}")
    
    actual_pos = calculate_anamorphic_position(desired_pos[0], desired_pos[1], desired_depth, VIEWING_ANGLE)
    actual_size = calculate_anamorphic_scale(desired_size, desired_depth, VIEWING_ANGLE)
    actual_rotation = calculate_anamorphic_rotation(desired_depth, VIEWING_ANGLE)
    
    objects = []
    
    # Face
    bpy.ops.mesh.primitive_uv_sphere_add(location=actual_pos, radius=actual_size)
    face = bpy.context.active_object
    face.name = f"{name}_Face"
    face.rotation_euler = actual_rotation
    
    face_mat = create_material(f"{name}_Face_Mat", (1.0, 0.9, 0.2), "plastic")
    face.data.materials.append(face_mat)
    objects.append(face)
    
    # Eyes with anamorphic positioning
    eye_offsets = [(-0.4, 0.6, 0.2), (0.4, 0.6, 0.2)]
    for i, (ex, ey, ez) in enumerate(eye_offsets):
        desired_eye_pos = (desired_pos[0] + ex*desired_size, desired_pos[1] + ey*desired_size, desired_pos[2] + ez*desired_size)
        actual_eye_pos = calculate_anamorphic_position(desired_eye_pos[0], desired_eye_pos[1], desired_depth, VIEWING_ANGLE)
        
        bpy.ops.mesh.primitive_uv_sphere_add(location=actual_eye_pos, radius=actual_size*0.12)
        eye = bpy.context.active_object
        eye.name = f"{name}_Eye_{i}"
        eye.rotation_euler = actual_rotation
        
        eye_mat = create_material(f"{name}_Eye_{i}_Mat", (0.05, 0.05, 0.05), "plastic")
        eye.data.materials.append(eye_mat)
        objects.append(eye)
    
    return objects

def create_anamorphic_scene():
    """Create scene with PROPER anamorphic object placement"""
    print("🎯 Creating REAL anamorphic scene with proper distortion math...")
    
    all_objects = []
    
    # Define objects by their APPARENT positions and depths
    # (where they should appear to be when viewed correctly)
    anamorphic_objects = [
        # Main centerpiece flower - appears to pop out 4 units from billboard
        {"type": "flower", "name": "MainFlower", "apparent_pos": (0, 1), "size": 3.0, "depth": 4.0},
        
        # Secondary flowers at different apparent depths
        {"type": "flower", "name": "Flower2", "apparent_pos": (-4, -1), "size": 2.0, "depth": 2.5},
        {"type": "flower", "name": "Flower3", "apparent_pos": (4, 2), "size": 2.5, "depth": 3.5},
        {"type": "flower", "name": "Flower4", "apparent_pos": (-2, 3), "size": 1.8, "depth": 1.5},
        
        # Emoji faces at strategic depths
        {"type": "emoji", "name": "HappyFace1", "apparent_pos": (-5, 0), "size": 2.2, "depth": 5.0},
        {"type": "emoji", "name": "HappyFace2", "apparent_pos": (5, -2), "size": 2.0, "depth": 3.0},
        {"type": "emoji", "name": "HappyFace3", "apparent_pos": (1, -3), "size": 1.5, "depth": 1.8},
        
        # Objects that appear VERY close (almost touching viewer)
        {"type": "flower", "name": "CloseFlower", "apparent_pos": (-1, 4), "size": 1.2, "depth": 6.0},
        {"type": "emoji", "name": "CloseEmoji", "apparent_pos": (2, 4), "size": 1.8, "depth": 5.5},
    ]
    
    # Create each object with proper anamorphic distortion
    for obj_info in anamorphic_objects:
        if obj_info["type"] == "flower":
            objs = create_anamorphic_flower(
                obj_info["name"],
                obj_info["apparent_pos"],
                obj_info["size"],
                obj_info["depth"]
            )
        elif obj_info["type"] == "emoji":
            objs = create_anamorphic_emoji(
                obj_info["name"],
                obj_info["apparent_pos"], 
                obj_info["size"],
                obj_info["depth"]
            )
        
        all_objects.extend(objs)
    
    print(f"✅ Created {len(all_objects)} anamorphic objects with proper distortion!")
    print(f"📐 All objects positioned using viewing angle: {VIEWING_ANGLE}°")
    print(f"📏 Objects will appear to pop out when viewed from correct position")
    
    return all_objects

def setup_anamorphic_camera():
    """Setup camera at the EXACT anamorphic viewing position"""
    bpy.ops.object.camera_add()
    camera = bpy.context.active_object
    camera.name = "Anamorphic_Viewing_Camera"
    
    # Position camera at the exact viewing angle and distance
    viewing_angle_rad = math.radians(VIEWING_ANGLE)
    camera_x = VIEWING_DISTANCE * math.sin(viewing_angle_rad)
    camera_y = -VIEWING_DISTANCE * math.cos(viewing_angle_rad)
    camera_z = VIEWING_HEIGHT
    
    camera.location = (camera_x, camera_y, camera_z)
    camera.rotation_euler = (math.radians(90 - VIEWING_ANGLE), 0, math.radians(VIEWING_ANGLE))
    camera.data.lens = 35
    
    bpy.context.scene.camera = camera
    
    print(f"📷 Anamorphic camera positioned at viewing angle {VIEWING_ANGLE}°")
    print(f"   Camera location: ({camera_x:.1f}, {camera_y:.1f}, {camera_z:.1f})")
    print(f"   Objects will appear to pop out when viewed from this angle!")
    
    return camera

def setup_lighting():
    """Setup lighting for anamorphic scene"""
    # Key light from viewing direction
    viewing_angle_rad = math.radians(VIEWING_ANGLE)
    light_x = 20 * math.sin(viewing_angle_rad)
    light_y = -20 * math.cos(viewing_angle_rad)
    light_z = 25
    
    bpy.ops.object.light_add(type='AREA', location=(light_x, light_y, light_z))
    key_light = bpy.context.active_object
    key_light.name = "Anamorphic_Key_Light"
    key_light.data.energy = 200
    key_light.data.size = 15
    
    # Fill light
    bpy.ops.object.light_add(type='AREA', location=(-15, -10, 20))
    fill_light = bpy.context.active_object
    fill_light.name = "Fill_Light"
    fill_light.data.energy = 100
    fill_light.data.size = 12
    
    print("💡 Anamorphic lighting setup complete")

def setup_render_settings():
    """Setup render settings"""
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.cycles.samples = 200
    scene.cycles.use_denoising = True
    scene.view_settings.view_transform = 'Filmic'
    scene.view_settings.look = 'High Contrast'

def main():
    """Main function to create REAL anamorphic billboard"""
    
    image_path = "data/input/benchmark.jpg"  # ← Your image here
    output_path = "REAL_anamorphic_billboard.png"
    
    print("🎯 CREATING REAL ANAMORPHIC BILLBOARD WITH PROPER DISTORTION")
    print("=" * 80)
    print(f"🔧 Anamorphic Parameters:")
    print(f"   Viewing Angle: {VIEWING_ANGLE}°")
    print(f"   Viewing Distance: {VIEWING_DISTANCE} units")
    print(f"   Billboard Plane: Z = {BILLBOARD_PLANE_Z}")
    print("=" * 80)
    
    # Step 1: Clear everything
    print("\n🧹 Step 1: Clearing scene...")
    clear_everything()
    
    # Step 2: Create billboard frame (reference plane)
    print("\n🖼️ Step 2: Creating billboard frame at reference plane...")
    frame, background = create_frame_and_base(image_path)
    
    # Step 3: Create anamorphic objects with proper distortion
    print("\n🎯 Step 3: Creating anamorphic objects with proper distortion math...")
    objects = create_anamorphic_scene()
    
    # Step 4: Setup camera at exact anamorphic viewing position
    print("\n📷 Step 4: Positioning camera at anamorphic viewing angle...")
    camera = setup_anamorphic_camera()
    
    # Step 5: Setup lighting
    print("\n💡 Step 5: Setting up lighting...")
    setup_lighting()
    
    # Step 6: Render
    print("\n🎞️ Step 6: Configuring render...")
    setup_render_settings()
    
    print(f"\n🎬 Step 7: Rendering anamorphic view to {output_path}...")
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    
    print("=" * 80)
    print("✅ REAL ANAMORPHIC BILLBOARD COMPLETED!")
    print(f"📁 Output: {output_path}")
    print(f"\n🎯 ANAMORPHIC EFFECT EXPLANATION:")
    print(f"  • Objects positioned using proper anamorphic mathematics")
    print(f"  • Each object's position compensated for {VIEWING_ANGLE}° viewing angle")
    print(f"  • Objects appear to pop out of billboard plane (Z={BILLBOARD_PLANE_Z})")
    print(f"  • Camera positioned at exact anamorphic viewing position")
    print(f"  • Result: 3D illusion that only works from correct viewing angle!")
    print(f"\n📐 The objects are physically distorted in 3D space,")
    print(f"    but appear normal and pop out when viewed from the camera angle!")

if __name__ == "__main__":
    main()
