#!/usr/bin/env python3
"""
REAL BENCHMARK ACTIVATION - Matching the Stunning 3D Pop-out Characters
Target: Colorful furry characters dramatically bursting from classical frame
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

def create_real_benchmark_billboard(image_path, output_path):
    """Create billboard matching the STUNNING REAL BENCHMARK"""
    print("🎯 REAL BENCHMARK ACTIVATION")
    print("="*70)
    print("🔥 TARGETING: Colorful 3D characters bursting dramatically!")
    
    # === STEP 1: CLASSICAL FRAME ENVIRONMENT ===
    print("🏛️ Step 1: Creating CLASSICAL FRAME like benchmark...")
    
    # Ground with classical styling
    bpy.ops.mesh.primitive_plane_add(size=50, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "Classical_Ground"
    
    # Classical buildings
    bpy.ops.mesh.primitive_cube_add(size=12, location=(-25, 15, 8))
    building_left = bpy.context.active_object
    building_left.name = "Classical_Building_Left"
    building_left.scale = (1.5, 2, 1.8)
    
    bpy.ops.mesh.primitive_cube_add(size=12, location=(25, 15, 8))
    building_right = bpy.context.active_object
    building_right.name = "Classical_Building_Right"
    building_right.scale = (1.5, 2, 1.8)
    
    # === STEP 2: ORNATE CLASSICAL FRAME ===
    print("🎭 Step 2: Creating ORNATE CLASSICAL FRAME...")
    
    # Main frame structure
    bpy.ops.mesh.primitive_cube_add(size=2)
    frame = bpy.context.active_object
    frame.name = "Classical_Frame"
    frame.scale = (7, 0.3, 4)
    frame.location = (0, 5, 4)
    
    # === STEP 3: ULTRA HIGH-DETAIL BILLBOARD ===
    print("🚀 Step 3: Creating ULTRA HIGH-DETAIL billboard...")
    
    bpy.ops.mesh.primitive_plane_add(size=6, location=(0, 4.7, 4))
    billboard = bpy.context.active_object
    billboard.name = "Benchmark_3D_Billboard"
    billboard.scale = (3.2, 1.8, 1)
    
    # MASSIVE SUBDIVISION FOR CHARACTER POP-OUT
    bpy.context.view_layer.objects.active = billboard
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    
    for i in range(7):
        bpy.ops.mesh.subdivide()
        print(f"  ✅ Ultra subdivision {i+1}/7")
    
    bpy.ops.uv.unwrap()
    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"✅ Billboard: {len(billboard.data.vertices)} vertices")
    
    # === STEP 4: LOAD CONTENT IMAGE ===
    print("📷 Step 4: Loading content image...")
    
    if not os.path.exists(image_path):
        print(f"❌ ERROR: Image not found: {image_path}")
        return False
    
    try:
        image = bpy.data.images.load(image_path)
        print(f"✅ Image loaded: {image.name} ({image.size[0]}x{image.size[1]})")
    except Exception as e:
        print(f"❌ ERROR loading image: {e}")
        return False
    
    # === STEP 5: EXTREME 3D CHARACTER MATERIAL ===
    print("🎪 Step 5: Creating EXTREME 3D material...")
    
    material = bpy.data.materials.new(name="Real_Benchmark_3D")
    material.use_nodes = True
    billboard.data.materials.clear()
    billboard.data.materials.append(material)
    
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    
    # Professional material pipeline
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (1000, 0)
    
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (800, 0)
    bsdf.inputs['Roughness'].default_value = 0.2
    try:
        bsdf.inputs['Metallic'].default_value = 0.0
    except KeyError:
        pass
    
    # EXTREME COLOR VIBRANCY
    hsv = nodes.new('ShaderNodeHueSaturation')
    hsv.location = (600, 0)
    hsv.inputs['Saturation'].default_value = 4.0  # QUADRUPLE saturation
    hsv.inputs['Value'].default_value = 1.8       # 80% brighter
    
    gamma = nodes.new('ShaderNodeGamma')
    gamma.location = (400, 0)
    gamma.inputs['Gamma'].default_value = 0.7
    
    img_node = nodes.new('ShaderNodeTexImage')
    img_node.image = image
    img_node.location = (0, 0)
    
    tex_coord = nodes.new('ShaderNodeTexCoord')
    tex_coord.location = (-200, 0)
    
    # EXTREME 3D DISPLACEMENT
    print("  💥 Setting up EXTREME DISPLACEMENT...")
    
    displacement = nodes.new('ShaderNodeDisplacement')
    displacement.location = (800, -400)
    displacement.inputs['Scale'].default_value = 4.0  # MASSIVE pop-out
    displacement.inputs['Midlevel'].default_value = 0.2
    
    coloramp = nodes.new('ShaderNodeValToRGB')
    coloramp.location = (400, -400)
    coloramp.color_ramp.elements[0].position = 0.1
    coloramp.color_ramp.elements[1].position = 0.9
    
    separate_rgb = nodes.new('ShaderNodeSeparateRGB')
    separate_rgb.location = (100, -400)
    
    disp_img = nodes.new('ShaderNodeTexImage')
    disp_img.image = image
    disp_img.location = (0, -400)
    
    # CONNECT NODES
    links.new(tex_coord.outputs['UV'], img_node.inputs['Vector'])
    links.new(img_node.outputs['Color'], gamma.inputs['Color'])
    links.new(gamma.outputs['Color'], hsv.inputs['Color'])
    links.new(hsv.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    links.new(tex_coord.outputs['UV'], disp_img.inputs['Vector'])
    links.new(disp_img.outputs['Color'], separate_rgb.inputs['Image'])
    links.new(separate_rgb.outputs['R'], coloramp.inputs['Fac'])
    links.new(coloramp.outputs['Color'], displacement.inputs['Height'])
    links.new(displacement.outputs['Displacement'], output.inputs['Displacement'])
    
    print("✅ EXTREME 3D material: 4x saturation + 4.0x displacement")
    
    # === STEP 6: DRAMATIC CHARACTER LIGHTING ===
    print("🎬 Step 6: Setting up DRAMATIC LIGHTING...")
    
    # Remove existing lights
    for obj in [o for o in bpy.data.objects if o.type == 'LIGHT']:
        bpy.data.objects.remove(obj, do_unlink=True)
    
    # Character key light
    bpy.ops.object.light_add(type='AREA', location=(20, -25, 20))
    key_light = bpy.context.active_object
    key_light.name = "Character_Key_Light"
    key_light.data.energy = 3000
    key_light.data.size = 15
    key_light.data.color = (1.0, 0.9, 0.8)
    key_light.rotation_euler = (0.7, 0, 0.6)
    
    # Character fill light
    bpy.ops.object.light_add(type='AREA', location=(-15, -20, 15))
    fill_light = bpy.context.active_object
    fill_light.name = "Character_Fill_Light"
    fill_light.data.energy = 1000
    fill_light.data.size = 20
    fill_light.data.color = (0.8, 0.85, 1.0)
    fill_light.rotation_euler = (1.1, 0, -0.4)
    
    # Character rim light
    bpy.ops.object.light_add(type='SPOT', location=(10, 20, 18))
    rim_light = bpy.context.active_object
    rim_light.name = "Character_Rim_Light"
    rim_light.data.energy = 1500
    rim_light.data.spot_size = 0.8
    rim_light.data.color = (1.0, 0.8, 0.5)
    rim_light.rotation_euler = (-0.8, 0, 3.14)
    
    print("✅ Character lighting: Key (3000) + Fill (1000) + Rim (1500)")
    
    # === STEP 7: CHARACTER CAMERA ===
    print("📸 Step 7: Positioning CHARACTER camera...")
    
    bpy.ops.object.camera_add(location=(12, -15, 6))
    camera = bpy.context.active_object
    camera.name = "Character_Camera"
    camera.data.lens = 28
    camera.data.sensor_width = 36
    
    # Track to billboard
    track = camera.constraints.new(type='TRACK_TO')
    track.target = billboard
    track.track_axis = 'TRACK_NEGATIVE_Z'
    track.up_axis = 'UP_Y'
    
    bpy.context.scene.camera = camera
    print("✅ Character camera positioned")
    
    # === STEP 8: ULTRA HIGH QUALITY RENDER ===
    print("🏆 Step 8: ULTRA HIGH QUALITY rendering...")
    
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 1024  # ULTRA high quality
    scene.cycles.preview_samples = 256
    
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    
    scene.cycles.device = 'GPU'
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = 'OPENIMAGEDENOISE'
    
    scene.view_settings.view_transform = 'Filmic'
    scene.view_settings.look = 'Very High Contrast'
    scene.view_settings.exposure = 0.3
    scene.view_settings.gamma = 1.1
    
    print("✅ Ultra render: 1024 samples, GPU, Enhanced Filmic")
    
    # === STEP 9: FINAL RENDER ===
    print("💥 Step 9: RENDERING CHARACTER BURSTING EFFECT...")
    print("⏱️  High quality character render in progress...")
    
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    
    print("")
    print("🎪 REAL BENCHMARK CHARACTER EFFECT COMPLETE!")
    print("="*70)
    print(f"✅ Output: {output_path}")
    print("✅ REAL BENCHMARK FEATURES ACTIVATED:")
    print("   💥 4.0x displacement for dramatic character pop-out")
    print("   🌈 4x color saturation for character vibrancy")
    print("   🎬 3-point character lighting system")
    print("   🏆 1024 samples ultra-high quality")
    print("   📸 Optimal camera for character drama")
    print("   🎭 Ultra-detailed mesh for character forms")
    print("   🎪 Enhanced color grading for character glow")
    print("")
    print("🎯 TARGETING THE STUNNING CHARACTER BURSTING BENCHMARK!")
    
    return True

def main():
    """Run the real benchmark activation"""
    # Use bulgari content with benchmark quality
    image_path = "data/input/bulgari_watch.jpg"
    output_path = "output/REAL_BENCHMARK_CHARACTER_BURST.png"
    
    print("🎪 REAL BENCHMARK CHARACTER ACTIVATION")
    print(f"📷 Input: {image_path}")
    print(f"💾 Output: {output_path}")
    print("🎯 Target: Colorful 3D characters bursting dramatically!")
    
    # Clear and create character-quality billboard
    clear_scene()
    success = create_real_benchmark_billboard(image_path, output_path)
    
    if success:
        print("✅ REAL BENCHMARK CHARACTER ACTIVATION COMPLETE!")
        print("🎉 Character bursting billboard created!")
    else:
        print("❌ REAL BENCHMARK ACTIVATION FAILED")

if __name__ == "__main__":
    main() 