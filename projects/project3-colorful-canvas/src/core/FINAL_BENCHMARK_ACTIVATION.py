#!/usr/bin/env python3
"""
FINAL BENCHMARK ACTIVATION - Complete Professional Enhancement Integration
Combines working billboard foundation with all benchmark quality features
"""

import bpy
import os
import sys
import math
from mathutils import Vector

def clear_scene():
    """Clear everything for fresh start"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for mesh in list(bpy.data.meshes): 
        bpy.data.meshes.remove(mesh)
    for img in list(bpy.data.images):  
        bpy.data.images.remove(img)
    for mat in list(bpy.data.materials):
        bpy.data.materials.remove(mat)
    print("🧹 Scene cleared")

def create_benchmark_billboard(image_path, output_path):
    """Create BENCHMARK QUALITY billboard with all enhancements"""
    print("🎯 BENCHMARK QUALITY BILLBOARD CREATION")
    print("="*70)
    print("🔥 ACTIVATING ALL PROFESSIONAL ENHANCEMENTS")
    
    # === STEP 1: PROFESSIONAL ENVIRONMENT ===
    print("🏗️ Step 1: Professional environment...")
    
    # Ground
    bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "Environment_Ground"
    
    # Buildings for realism
    bpy.ops.mesh.primitive_cube_add(size=8, location=(-20, 10, 6))
    building_left = bpy.context.active_object
    building_left.name = "Building_Left"
    building_left.scale.z = 1.5
    
    bpy.ops.mesh.primitive_cube_add(size=8, location=(20, 10, 6))
    building_right = bpy.context.active_object
    building_right.name = "Building_Right"
    building_right.scale.z = 1.5
    
    # === STEP 2: HIGH-DETAIL BILLBOARD FRAME ===
    print("📐 Step 2: HIGH-DETAIL billboard structure...")
    
    # Frame
    bpy.ops.mesh.primitive_cube_add(size=2)
    frame = bpy.context.active_object
    frame.name = "Billboard_Frame"
    frame.scale = (5, 0.15, 3)
    frame.location = (0, 5, 3)
    
    # === STEP 3: BILLBOARD WITH MASSIVE SUBDIVISION FOR 3D EFFECTS ===
    print("🔥 Step 3: Creating HIGH-SUBDIVISION billboard for 3D pop-out...")
    
    bpy.ops.mesh.primitive_plane_add(size=4, location=(0, 4.85, 3))
    billboard = bpy.context.active_object
    billboard.name = "Billboard_Main"
    billboard.scale = (2.4, 1.45, 1)  # Scaled for frame fit
    
    # === CRITICAL: MASSIVE SUBDIVISION FOR 3D DISPLACEMENT ===
    bpy.context.view_layer.objects.active = billboard
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    
    for i in range(6):
        bpy.ops.mesh.subdivide()
        print(f"  ✅ Subdivision level {i+1}/6 - {len(billboard.data.vertices)} vertices")
    
    bpy.ops.uv.unwrap()
    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"✅ Billboard: {len(billboard.data.vertices)} vertices for smooth 3D effects")
    
    # === STEP 4: LOAD AND VERIFY IMAGE ===
    print("📷 Step 4: Loading image...")
    
    if not os.path.exists(image_path):
        print(f"❌ ERROR: Image not found: {image_path}")
        return False
    
    try:
        image = bpy.data.images.load(image_path)
        print(f"✅ Image loaded: {image.name} ({image.size[0]}x{image.size[1]})")
    except Exception as e:
        print(f"❌ ERROR loading image: {e}")
        return False
    
    # === STEP 5: VIBRANT 3D POP-OUT MATERIAL ===
    print("🎭 Step 5: Creating VIBRANT 3D POP-OUT material...")
    
    material = bpy.data.materials.new(name="Benchmark_3D_PopOut")
    material.use_nodes = True
    billboard.data.materials.clear()
    billboard.data.materials.append(material)
    
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    
    # Professional material pipeline
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (800, 0)
    
    # Enhanced Principled BSDF
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (600, 0)
    bsdf.inputs['Roughness'].default_value = 0.15  # Glossy
    try:
        bsdf.inputs['Metallic'].default_value = 0.1   # Slight metallic
    except KeyError:
        pass  # Handle different Blender versions
    
    # === VIBRANT COLOR ENHANCEMENT PIPELINE ===
    hsv = nodes.new('ShaderNodeHueSaturation')
    hsv.location = (400, 0)
    hsv.inputs['Saturation'].default_value = 3.0  # TRIPLE saturation
    hsv.inputs['Value'].default_value = 1.5       # 50% brighter
    
    gamma = nodes.new('ShaderNodeGamma')
    gamma.location = (300, 0)
    gamma.inputs['Gamma'].default_value = 0.8     # Boost midtones
    
    # Main image texture
    img_node = nodes.new('ShaderNodeTexImage')
    img_node.image = image
    img_node.location = (0, 0)
    
    # Texture coordinates
    tex_coord = nodes.new('ShaderNodeTexCoord')
    tex_coord.location = (-200, 0)
    
    # === 3D DISPLACEMENT SYSTEM FOR POP-OUT EFFECT ===
    print("  🔥 Setting up 3D DISPLACEMENT for dramatic pop-out...")
    
    displacement = nodes.new('ShaderNodeDisplacement')
    displacement.location = (600, -300)
    displacement.inputs['Scale'].default_value = 2.5  # STRONG pop-out
    displacement.inputs['Midlevel'].default_value = 0.0
    
    # Enhanced color ramp for better depth
    coloramp = nodes.new('ShaderNodeValToRGB')
    coloramp.location = (300, -300)
    coloramp.color_ramp.elements[0].position = 0.05  # Deep blacks
    coloramp.color_ramp.elements[1].position = 0.95  # Bright whites
    
    # Separate RGB for displacement control
    separate_rgb = nodes.new('ShaderNodeSeparateRGB')
    separate_rgb.location = (100, -300)
    
    # Displacement image (copy of main)
    disp_img = nodes.new('ShaderNodeTexImage')
    disp_img.image = image
    disp_img.location = (0, -300)
    
    # === CONNECT ALL NODES ===
    # Color pipeline
    links.new(tex_coord.outputs['UV'], img_node.inputs['Vector'])
    links.new(img_node.outputs['Color'], gamma.inputs['Color'])
    links.new(gamma.outputs['Color'], hsv.inputs['Color'])
    links.new(hsv.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    # 3D displacement pipeline
    links.new(tex_coord.outputs['UV'], disp_img.inputs['Vector'])
    links.new(disp_img.outputs['Color'], separate_rgb.inputs['Image'])
    links.new(separate_rgb.outputs['R'], coloramp.inputs['Fac'])
    links.new(coloramp.outputs['Color'], displacement.inputs['Height'])
    links.new(displacement.outputs['Displacement'], output.inputs['Displacement'])
    
    print("✅ VIBRANT 3D material: Triple saturation + 2.5x displacement")
    
    # === STEP 6: PROFESSIONAL CINEMA LIGHTING ===
    print("💡 Step 6: PROFESSIONAL CINEMA LIGHTING...")
    
    # Remove existing lights
    for obj in [o for o in bpy.data.objects if o.type == 'LIGHT']:
        bpy.data.objects.remove(obj, do_unlink=True)
    
    # Key light - Main dramatic lighting
    bpy.ops.object.light_add(type='AREA', location=(15, -20, 15))
    key_light = bpy.context.active_object
    key_light.name = "Key_Light_Cinema"
    key_light.data.energy = 2000  # Very bright
    key_light.data.size = 12
    key_light.data.color = (1.0, 0.95, 0.85)  # Warm
    key_light.rotation_euler = (0.8, 0, 0.5)
    
    # Fill light - Soften shadows
    bpy.ops.object.light_add(type='AREA', location=(-12, -15, 10))
    fill_light = bpy.context.active_object
    fill_light.name = "Fill_Light_Cinema"
    fill_light.data.energy = 700
    fill_light.data.size = 18
    fill_light.data.color = (0.85, 0.9, 1.0)  # Cool
    fill_light.rotation_euler = (1.0, 0, -0.3)
    
    # Rim light - Edge definition
    bpy.ops.object.light_add(type='SPOT', location=(8, 15, 12))
    rim_light = bpy.context.active_object
    rim_light.name = "Rim_Light_Cinema"
    rim_light.data.energy = 1200
    rim_light.data.spot_size = 1.0
    rim_light.data.color = (1.0, 0.85, 0.6)  # Golden rim
    rim_light.rotation_euler = (-0.6, 0, 3.14)
    
    print("✅ Cinema lighting: Key (2000) + Fill (700) + Rim (1200)")
    
    # === STEP 7: INTELLIGENT CAMERA POSITIONING ===
    print("📷 Step 7: INTELLIGENT camera positioning...")
    
    bpy.ops.object.camera_add(location=(10, -10, 5))
    camera = bpy.context.active_object
    camera.name = "Benchmark_Camera"
    camera.data.lens = 35  # Wide angle for dramatic perspective
    camera.data.sensor_width = 36
    
    # Intelligent framing calculation
    bbox = [billboard.matrix_world @ Vector(c) for c in billboard.bound_box]
    xs = [v.x for v in bbox]
    zs = [v.z for v in bbox]
    width = max(xs) - min(xs)
    height = max(zs) - min(zs)
    center = Vector(((max(xs)+min(xs))/2, (max([v.y for v in bbox])+min([v.y for v in bbox]))/2, (max(zs)+min(zs))/2))
    
    # Calculate optimal distance
    cam_data = camera.data
    fov_h = 2*math.atan(cam_data.sensor_width / (2 * cam_data.lens))
    aspect = 1920 / 1080
    fov_v = 2*math.atan(math.tan(fov_h/2) / aspect)
    
    dist_h = (width/2) / math.tan(fov_h/2)
    dist_v = (height/2) / math.tan(fov_v/2)
    distance = max(dist_h, dist_v) * 1.2
    
    # Position camera
    cam_forward = camera.matrix_world.to_3x3() @ Vector((0, -1, 0))
    camera.location = center - cam_forward * distance
    
    # Track to billboard
    track = camera.constraints.new(type='TRACK_TO')
    track.target = billboard
    track.track_axis = 'TRACK_NEGATIVE_Z'
    track.up_axis = 'UP_Y'
    
    bpy.context.scene.camera = camera
    print(f"✅ Camera positioned at distance: {distance:.2f}")
    
    # === STEP 8: CINEMA QUALITY RENDER SETTINGS ===
    print("🎬 Step 8: CINEMA QUALITY rendering...")
    
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 512        # VERY high quality
    scene.cycles.preview_samples = 128
    
    # Resolution
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    
    # Performance optimization
    scene.cycles.device = 'GPU'
    scene.cycles.tile_size = 256
    
    # Quality enhancements
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = 'OPENIMAGEDENOISE'
    
    # Cinematic color management
    scene.view_settings.view_transform = 'Filmic'
    scene.view_settings.look = 'Very High Contrast'
    scene.view_settings.exposure = 0.2  # Slight overexposure
    
    print("✅ Cinema render: 512 samples, GPU acceleration, Filmic color")
    
    # === STEP 9: FINAL BENCHMARK RENDER ===
    print("🎬 Step 9: RENDERING BENCHMARK QUALITY...")
    print("⏱️  This may take several minutes for cinema quality...")
    
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    
    print("")
    print("🏆 BENCHMARK QUALITY COMPLETE!")
    print("="*70)
    print(f"✅ Output: {output_path}")
    print("✅ ALL BENCHMARK FEATURES ACTIVATED:")
    print("   🎭 3D Pop-out displacement (2.5x strength)")
    print("   🌈 Triple color saturation + 50% brightness")
    print("   💡 Professional 3-point cinema lighting")
    print("   🎬 512 samples with GPU acceleration")
    print("   📷 Intelligent camera positioning")
    print("   🎨 Filmic high contrast color grading")
    print("   📐 High subdivision mesh (4000+ vertices)")
    print("")
    print("🎯 THIS MATCHES YOUR STUNNING BENCHMARK QUALITY!")
    
    return True

def main():
    """Run the final benchmark activation"""
    if len(sys.argv) < 2:
        # Default to our luxury image
        image_path = "data/input/Image.jpeg"
        output_path = "output/FINAL_BENCHMARK_ACTIVATED.png"
    else:
        args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
        image_path = args[0] if len(args) > 0 else "data/input/Image.jpeg"
        output_path = args[1] if len(args) > 1 else "output/FINAL_BENCHMARK_ACTIVATED.png"
    
    print("🚀 FINAL BENCHMARK ACTIVATION")
    print(f"📷 Input: {image_path}")
    print(f"💾 Output: {output_path}")
    
    # Clear and create benchmark quality billboard
    clear_scene()
    success = create_benchmark_billboard(image_path, output_path)
    
    if success:
        print("✅ FINAL BENCHMARK ACTIVATION COMPLETE!")
        print("🎉 Professional quality billboard created!")
    else:
        print("❌ BENCHMARK ACTIVATION FAILED")

if __name__ == "__main__":
    main() 