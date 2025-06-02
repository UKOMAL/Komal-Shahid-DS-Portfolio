"""
🎬 BLENDER + AI ANAMORPHIC BILLBOARD GENERATOR
Complete Professional 3D Anamorphic Rendering with AI Models Integration

Author: Komal Shahid
Course: DSC680 - Bellevue University
Date: June 1, 2025

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
import math
import numpy as np
from mathutils import Vector, Matrix

# Add our project path so we can import our AI models
project_path = '/Users/komalshahid/Desktop/Bellevue University/DSC680/projects/project3-colorful-canvas'
if project_path not in sys.path:
    sys.path.append(project_path)

try:
    from src.milestone3.colorful_canvas_complete import ColorfulCanvasAI, deps
    COLORFUL_CANVAS_AVAILABLE = True
    print("✅ ColorfulCanvasAI integration enabled")
except ImportError as e:
    COLORFUL_CANVAS_AVAILABLE = False
    print(f"⚠️ ColorfulCanvasAI not available: {e}")
    print("💡 Will use basic image processing")

class AnamorphicBillboardGenerator:
    """Main class that integrates ColorfulCanvasAI with Blender 3D generation"""
    
    def __init__(self):
        self.ai = None
        if COLORFUL_CANVAS_AVAILABLE:
            try:
                self.ai = ColorfulCanvasAI()
                print("🎨 ColorfulCanvasAI initialized for advanced processing")
            except Exception as e:
                print(f"⚠️ Could not initialize ColorfulCanvasAI: {e}")
                self.ai = None
    
    def process_image_with_ai(self, image_path, effect_type="shadow_box", strength=1.5):
        """Process image using ColorfulCanvasAI for advanced effects"""
        if not self.ai:
            print("Using basic image processing (ColorfulCanvasAI unavailable)")
            return image_path, None, None
        
        try:
            # Load and analyze image
            pil_image = self.ai.load_image(image_path)
            
            # Generate optimized depth map using AI
            depth_map = self.ai.generate_depth_map(pil_image)
            
            # Create anamorphic effect
            if effect_type == "seoul_corner":
                processed_image = self.ai.create_seoul_corner_projection(pil_image, depth_map)
            elif effect_type == "screen_pop":
                processed_image = self.ai.create_screen_pop_effect(pil_image, depth_map, strength)
            else:  # shadow_box
                processed_image = self.ai.create_shadow_box_effect(pil_image, depth_map, strength)
            
            # Analyze for Seoul display optimization
            analysis = self.ai.analyze_for_seoul_effect(pil_image)
            
            # Save processed images
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_dir = os.path.dirname(image_path) or "."
            
            processed_path = os.path.join(output_dir, f"{base_name}_anamorphic.png")
            depth_path = os.path.join(output_dir, f"{base_name}_depth.png")
            
            self.ai.save_image(processed_image, processed_path)
            self.ai.save_image(depth_map, depth_path)
            
            print(f"✅ AI processing complete: {effect_type} effect applied")
            return processed_path, depth_path, analysis
            
        except Exception as e:
            print(f"❌ AI processing failed: {e}")
            return image_path, None, None

def clear_scene():
    """Clear all mesh objects from the scene"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False, confirm=False)

def create_billboard_frame(width=16, height=9, depth=0.5, frame_thickness=0.8):
    """Create the main billboard frame with proper proportions"""
    
    # Create outer frame
    bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))
    outer_frame = bpy.context.active_object
    outer_frame.name = "Billboard_Frame_Outer"
    outer_frame.scale = (width/2 + frame_thickness, height/2 + frame_thickness, depth)
    
    # Create inner cutout
    bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))
    inner_cutout = bpy.context.active_object
    inner_cutout.name = "Billboard_Frame_Inner"
    inner_cutout.scale = (width/2, height/2, depth + 0.1)
    
    # Apply boolean modifier to create frame
    bool_mod = outer_frame.modifiers.new(name="Boolean", type='BOOLEAN')
    bool_mod.operation = 'DIFFERENCE'
    bool_mod.object = inner_cutout
    
    # Apply modifier
    bpy.context.view_layer.objects.active = outer_frame
    bpy.ops.object.modifier_apply(modifier="Boolean")
    
    # Delete inner cutout
    bpy.data.objects.remove(inner_cutout, do_unlink=True)
    
    # Create screen plane
    bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, -0.05))  # Slightly recessed for better depth effect
    screen = bpy.context.active_object
    screen.name = "Billboard_Screen"
    screen.scale = (width/2, height/2, 1)
    
    return outer_frame, screen

def create_material_with_image(name, image_path=None, emission_strength=2.0):
    """Create a material with image texture and emission"""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Clear default nodes
    nodes.clear()
    
    # Add nodes
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    emission_node = nodes.new(type='ShaderNodeEmission')
    
    if image_path and os.path.exists(image_path):
        # Add image texture node
        image_node = nodes.new(type='ShaderNodeTexImage')
        image_node.image = bpy.data.images.load(image_path)
        
        # Link nodes
        links.new(image_node.outputs['Color'], emission_node.inputs['Color'])
    
    emission_node.inputs['Strength'].default_value = emission_strength
    links.new(emission_node.outputs['Emission'], output_node.inputs['Surface'])
    
    # Position nodes
    output_node.location = (400, 0)
    emission_node.location = (200, 0)
    if image_path:
        image_node.location = (0, 0)
    
    return mat

def create_extruded_geometry_from_image(image_path, screen_obj, extrude_distance=3.0):
    """Create 3D extruded geometry based on image brightness"""
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None
    
    # Load image
    img = bpy.data.images.load(image_path)
    
    # Create displacement material
    mat = bpy.data.materials.new(name="Displacement_Material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    nodes.clear()
    
    # Add nodes for displacement
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    principled_node = nodes.new(type='ShaderNodeBsdfPrincipled')
    image_node = nodes.new(type='ShaderNodeTexImage')
    coloramp_node = nodes.new(type='ShaderNodeValToRGB')
    
    # Set image
    image_node.image = img
    
    # Configure color ramp for displacement
    coloramp_node.color_ramp.elements[0].position = 0.0
    coloramp_node.color_ramp.elements[1].position = 1.0
    
    # Link nodes
    links.new(image_node.outputs['Color'], coloramp_node.inputs['Fac'])
    links.new(image_node.outputs['Color'], principled_node.inputs['Base Color'])
    links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])
    links.new(coloramp_node.outputs['Color'], output_node.inputs['Displacement'])
    
    # Apply material to screen
    screen_obj.data.materials.append(mat)
    
    # Add subdivision surface modifier
    subdiv_mod = screen_obj.modifiers.new(name="Subdivision", type='SUBSURF')
    subdiv_mod.levels = 4
    
    # Add displacement modifier
    disp_mod = screen_obj.modifiers.new(name="Displacement", type='DISPLACE')
    disp_mod.strength = extrude_distance
    disp_mod.mid_level = 0.5
    
    return screen_obj

def create_extruded_geometry_from_ai_depth(image_path, depth_path, screen_obj, extrude_distance=3.0):
    """Create 3D extruded geometry using AI-generated depth map"""
    
    if not depth_path or not os.path.exists(depth_path):
        print("Using basic displacement (no AI depth map)")
        return create_extruded_geometry_from_image(image_path, screen_obj, extrude_distance)
    
    try:
        # Load AI-generated depth map
        depth_img = bpy.data.images.load(depth_path)
        
        # Create advanced displacement material
        mat = bpy.data.materials.new(name="AI_Displacement_Material")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        
        nodes.clear()
        
        # Enhanced node setup for AI depth
        output_node = nodes.new(type='ShaderNodeOutputMaterial')
        principled_node = nodes.new(type='ShaderNodeBsdfPrincipled')
        depth_node = nodes.new(type='ShaderNodeTexImage')
        color_node = nodes.new(type='ShaderNodeTexImage')
        coloramp_node = nodes.new(type='ShaderNodeValToRGB')
        multiply_node = nodes.new(type='ShaderNodeMath')
        
        # Set images
        depth_node.image = depth_img
        if os.path.exists(image_path):
            color_img = bpy.data.images.load(image_path)
            color_node.image = color_img
        
        # Configure nodes
        multiply_node.operation = 'MULTIPLY'
        multiply_node.inputs[1].default_value = extrude_distance
        
        # Enhanced color ramp for AI depth
        coloramp_node.color_ramp.elements[0].position = 0.1
        coloramp_node.color_ramp.elements[1].position = 0.9
        
        # Link nodes
        links.new(depth_node.outputs['Color'], coloramp_node.inputs['Fac'])
        links.new(coloramp_node.outputs['Color'], multiply_node.inputs[0])
        links.new(color_node.outputs['Color'], principled_node.inputs['Base Color'])
        links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])
        links.new(multiply_node.outputs['Value'], output_node.inputs['Displacement'])
        
        # Apply material
        screen_obj.data.materials.clear()
        screen_obj.data.materials.append(mat)
        
        # Enhanced modifiers for AI depth
        subdiv_mod = screen_obj.modifiers.new(name="AI_Subdivision", type='SUBSURF')
        subdiv_mod.levels = 5  # Higher subdivision for AI detail
        
        disp_mod = screen_obj.modifiers.new(name="AI_Displacement", type='DISPLACE')
        disp_mod.strength = extrude_distance * 1.5  # Enhanced for AI depth
        disp_mod.mid_level = 0.3
        
        print("✅ AI-enhanced 3D geometry created")
        return screen_obj
        
    except Exception as e:
        print(f"⚠️ AI depth processing failed: {e}, using basic method")
        return create_extruded_geometry_from_image(image_path, screen_obj, extrude_distance)

def create_floating_elements(base_location=(0, 0, 0), count=10):
    """Create floating 3D elements that appear to emerge from the screen"""
    floating_objects = []
    
    for i in range(count):
        # Create various primitive shapes
        shape_type = i % 4
        
        if shape_type == 0:
            bpy.ops.mesh.primitive_cube_add()
        elif shape_type == 1:
            bpy.ops.mesh.primitive_uv_sphere_add()
        elif shape_type == 2:
            bpy.ops.mesh.primitive_cylinder_add()
        else:
            bpy.ops.mesh.primitive_torus_add()
        
        obj = bpy.context.active_object
        obj.name = f"Floating_Element_{i}"
        
        # Position randomly around the billboard
        import random
        x_offset = random.uniform(-12, 12)
        y_offset = random.uniform(-2, 8)
        z_offset = random.uniform(2, 8)
        
        obj.location = (
            base_location[0] + x_offset,
            base_location[1] + y_offset,
            base_location[2] + z_offset
        )
        
        # Random scale
        scale_factor = random.uniform(0.5, 2.0)
        obj.scale = (scale_factor, scale_factor, scale_factor)
        
        # Random rotation
        obj.rotation_euler = (
            random.uniform(0, math.pi),
            random.uniform(0, math.pi),
            random.uniform(0, math.pi)
        )
        
        # Create colorful material
        mat = create_material_with_image(f"Floating_Material_{i}")
        mat.node_tree.nodes['Emission'].inputs['Color'].default_value = (
            random.uniform(0.3, 1.0),
            random.uniform(0.3, 1.0),
            random.uniform(0.3, 1.0),
            1.0
        )
        obj.data.materials.append(mat)
        
        floating_objects.append(obj)
    
    return floating_objects

def setup_camera_for_anamorphic_view(billboard_location=(0, 0, 0)):
    """Setup camera for proper anamorphic viewing angle"""
    
    # Create camera
    bpy.ops.object.camera_add()
    camera = bpy.context.active_object
    camera.name = "Anamorphic_Camera"
    
    # Position camera for optimal viewing angle (45-60 degrees from billboard)
    camera.location = (25, -15, 5)
    camera.rotation_euler = (math.radians(75), 0, math.radians(55))
    
    # Set camera properties
    camera.data.lens = 35  # Wide angle lens
    camera.data.clip_start = 0.1
    camera.data.clip_end = 1000
    
    # Set as active camera
    bpy.context.scene.camera = camera
    
    return camera

def setup_professional_lighting():
    """Setup professional lighting for the scene"""
    
    # Key light (main illumination)
    bpy.ops.object.light_add(type='AREA', location=(10, -10, 15))
    key_light = bpy.context.active_object
    key_light.name = "Key_Light"
    key_light.data.energy = 500
    key_light.data.size = 10
    key_light.data.color = (1.0, 0.95, 0.8)
    
    # Fill light (soften shadows)
    bpy.ops.object.light_add(type='AREA', location=(-8, -5, 8))
    fill_light = bpy.context.active_object
    fill_light.name = "Fill_Light"
    fill_light.data.energy = 200
    fill_light.data.size = 15
    fill_light.data.color = (0.8, 0.9, 1.0)
    
    # Rim light (edge highlighting)
    bpy.ops.object.light_add(type='SPOT', location=(0, 10, 12))
    rim_light = bpy.context.active_object
    rim_light.name = "Rim_Light"
    rim_light.data.energy = 300
    rim_light.data.spot_size = math.radians(45)
    rim_light.data.color = (0.9, 0.7, 1.0)
    
    # Ambient environment
    world = bpy.context.scene.world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.1  # Low ambient

def create_particle_effects(emitter_obj):
    """Add particle effects for extra visual appeal"""
    
    # Add particle system
    bpy.context.view_layer.objects.active = emitter_obj
    bpy.ops.object.particle_system_add()
    
    psys = emitter_obj.particle_systems[0]
    settings = psys.settings
    
    # Configure particle settings
    settings.count = 1000
    settings.lifetime = 120
    settings.emit_from = 'FACE'
    settings.physics_type = 'NEWTON'
    
    # Velocity settings
    settings.normal_factor = 2.0
    settings.factor_random = 0.5
    
    # Render settings
    settings.render_type = 'HALO'
    settings.material_slot = len(emitter_obj.data.materials)
    
    # Create particle material
    particle_mat = create_material_with_image("Particle_Material")
    particle_mat.node_tree.nodes['Emission'].inputs['Color'].default_value = (1, 0.8, 0.2, 1)
    emitter_obj.data.materials.append(particle_mat)

def setup_seoul_lighting():
    """Setup special lighting for Seoul LED billboard effect"""
    
    # Clear existing lights
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj, do_unlink=True)
    
    # Key blue light (LED glow)
    bpy.ops.object.light_add(type='AREA', location=(5, -8, 3))
    key_light = bpy.context.active_object
    key_light.name = "LED_Key_Light"
    key_light.data.energy = 700
    key_light.data.size = 15
    key_light.data.color = (0.7, 0.8, 1.0)  # Blue-ish LED color
    
    # Fill purple light (accent)
    bpy.ops.object.light_add(type='AREA', location=(-6, -3, 4))
    fill_light = bpy.context.active_object
    fill_light.name = "LED_Fill_Light"
    fill_light.data.energy = 300
    fill_light.data.size = 10
    fill_light.data.color = (0.8, 0.6, 1.0)  # Purple-ish accent
    
    # Rim red light (edge highlighting)
    bpy.ops.object.light_add(type='SPOT', location=(0, 8, 10))
    rim_light = bpy.context.active_object
    rim_light.name = "Neon_Rim_Light"
    rim_light.data.energy = 500
    rim_light.data.spot_size = math.radians(60)
    rim_light.data.color = (1.0, 0.5, 0.5)  # Red-ish neon glow
    
    # Low fill light (shadow detail)
    bpy.ops.object.light_add(type='POINT', location=(0, 0, -5))
    low_light = bpy.context.active_object
    low_light.name = "Shadow_Fill_Light"
    low_light.data.energy = 100
    low_light.data.color = (0.2, 0.3, 0.4)  # Cool shadow fill
    
    # Set dark environment for more contrast
    world = bpy.context.scene.world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.01, 0.01, 0.02, 1.0)  # Very dark blue
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.02  # Very low ambient

def setup_render_settings(effect_type="shadow_box"):
    """Configure render settings for high quality output"""
    
    scene = bpy.context.scene
    
    # Set render engine to Cycles for realistic lighting
    scene.render.engine = 'CYCLES'
    
    # Set resolution
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    
    # Cycles settings
    scene.cycles.samples = 128
    scene.cycles.use_denoising = True
    
    # Seoul-specific render settings
    if effect_type == "seoul_corner":
        # High contrast settings for LED effect
        scene.view_settings.view_transform = 'Filmic'
        scene.view_settings.look = 'Very High Contrast'
        
        # More samples for better light quality
        scene.cycles.samples = 160
        
        # Increase exposure for LED glow effect
        scene.view_settings.exposure = 0.5
    else:
        # Standard settings for other effects
        scene.view_settings.view_transform = 'Filmic'
        scene.view_settings.look = 'High Contrast'

def main(image_path=None, output_path=None, effect_type="shadow_box", ai_strength=1.5):
    """Main function to create the anamorphic billboard display with AI integration"""
    
    print("Creating AI-Enhanced Anamorphic 3D Billboard...")
    
    # Initialize AI processor
    generator = AnamorphicBillboardGenerator()
    
    # Clear existing scene
    clear_scene()
    
    # Process image with AI if available
    processed_image_path = image_path
    depth_map_path = None
    seoul_analysis = None
    
    if image_path and os.path.exists(image_path):
        processed_image_path, depth_map_path, seoul_analysis = generator.process_image_with_ai(
            image_path, effect_type, ai_strength
        )
        
        if seoul_analysis:
            print(f"📊 AI Analysis - Brightness: {seoul_analysis['brightness']['recommendation']}")
            print(f"📊 AI Analysis - Detail Level: {seoul_analysis['detail_level']['recommendation']}")
    
    # Create billboard frame and screen with more prominent frame
    frame, screen = create_billboard_frame(width=16, height=9, depth=1.0, frame_thickness=1.2)
    
    # Apply AI-processed image to screen
    if processed_image_path and os.path.exists(processed_image_path):
        # Create AI-enhanced material
        screen_material = None
        
        # Seoul-specific screen material with LED-like dots
        if effect_type == "seoul_corner":
            # Create more realistic LED screen material for Seoul corner
            screen_material = bpy.data.materials.new(name="Seoul_LED_Screen")
            screen_material.use_nodes = True
            nodes = screen_material.node_tree.nodes
            links = screen_material.node_tree.links
            
            # Clear default nodes
            nodes.clear()
            
            # Add nodes
            output_node = nodes.new(type='ShaderNodeOutputMaterial')
            emission_node = nodes.new(type='ShaderNodeEmission')
            image_node = nodes.new(type='ShaderNodeTexImage')
            mapping_node = nodes.new(type='ShaderNodeMapping')
            texcoord_node = nodes.new(type='ShaderNodeTexCoord')
            rgb_node = nodes.new(type='ShaderNodeRGB')
            mix_node = nodes.new(type='ShaderNodeMixRGB')
            
            # Load image
            if os.path.exists(processed_image_path):
                image_node.image = bpy.data.images.load(processed_image_path)
            
            # Configure nodes
            emission_node.inputs['Strength'].default_value = 4.0  # Higher emission for LED look
            rgb_node.outputs[0].default_value = (0.1, 0.1, 0.1, 1.0)  # Dark background
            mix_node.blend_type = 'ADD'
            mix_node.inputs[0].default_value = 0.8  # Mix factor
            
            # Link nodes
            links.new(texcoord_node.outputs['UV'], mapping_node.inputs['Vector'])
            links.new(mapping_node.outputs['Vector'], image_node.inputs['Vector'])
            links.new(image_node.outputs['Color'], mix_node.inputs[1])
            links.new(rgb_node.outputs[0], mix_node.inputs[2])
            links.new(mix_node.outputs[0], emission_node.inputs['Color'])
            links.new(emission_node.outputs['Emission'], output_node.inputs['Surface'])
            
            # Position nodes
            output_node.location = (600, 0)
            emission_node.location = (400, 0)
            mix_node.location = (200, 0)
            image_node.location = (0, 100)
            rgb_node.location = (0, -100)
            mapping_node.location = (-200, 100)
            texcoord_node.location = (-400, 100)
        else:
            # Standard material for other effects
            screen_material = create_material_with_image("AI_Screen_Material", processed_image_path, 3.0)
        
        screen.data.materials.append(screen_material)
        
        # Create AI-enhanced 3D displacement - increase extrusion for more obvious effect
        create_extruded_geometry_from_ai_depth(processed_image_path, depth_map_path, screen, 
                                             extrude_distance=3.0 * ai_strength)
        
        # Add particle effects based on AI analysis
        if seoul_analysis and seoul_analysis['detail_level']['value'] > 0.08:
            create_particle_effects(screen)
    else:
        # Fallback to default material
        default_mat = create_material_with_image("Default_Screen_Material")
        default_mat.node_tree.nodes['Emission'].inputs['Color'].default_value = (0.2, 0.6, 1.0, 1.0)
        screen.data.materials.append(default_mat)
    
    # Create more visible frame material with metallic finish
    frame_material = bpy.data.materials.new(name="Frame_Material")
    frame_material.use_nodes = True
    
    # More visible metallic frame for seoul_corner effect
    if effect_type == "seoul_corner":
        # Setup nodes for a more complex metallic frame
        nodes = frame_material.node_tree.nodes
        links = frame_material.node_tree.links
        
        # Clear default
        nodes.clear()
        
        # Add nodes
        output = nodes.new(type='ShaderNodeOutputMaterial')
        mix_shader = nodes.new(type='ShaderNodeMixShader')
        glossy = nodes.new(type='ShaderNodeBsdfGlossy')
        diffuse = nodes.new(type='ShaderNodeBsdfDiffuse')
        fresnel = nodes.new(type='ShaderNodeFresnel')
        
        # Set properties
        glossy.inputs['Color'].default_value = (0.9, 0.9, 0.95, 1.0)  # Silver blue color
        glossy.inputs['Roughness'].default_value = 0.05  # Very glossy
        diffuse.inputs['Color'].default_value = (0.2, 0.2, 0.25, 1.0)  # Darker base
        fresnel.inputs['IOR'].default_value = 3.0  # Higher for more metallic look
        
        # Link nodes
        links.new(fresnel.outputs[0], mix_shader.inputs[0])
        links.new(diffuse.outputs[0], mix_shader.inputs[1])
        links.new(glossy.outputs[0], mix_shader.inputs[2])
        links.new(mix_shader.outputs[0], output.inputs['Surface'])
        
        # Position nodes
        output.location = (300, 0)
        mix_shader.location = (100, 0)
        fresnel.location = (-100, 100)
        diffuse.location = (-100, 0)
        glossy.location = (-100, -100)
    else:
        frame_material.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.1, 0.1, 0.1, 1.0)
        frame_material.node_tree.nodes["Principled BSDF"].inputs[4].default_value = 0.9
        frame_material.node_tree.nodes["Principled BSDF"].inputs[7].default_value = 0.1
        
    frame.data.materials.append(frame_material)
    
    # Create floating elements with AI-informed parameters
    element_count = 15
    if seoul_analysis:
        # Adjust element count based on image complexity
        if seoul_analysis['detail_level']['value'] > 0.1:
            element_count = 20
        elif seoul_analysis['detail_level']['value'] < 0.05:
            element_count = 10
    
    floating_elements = create_floating_elements(count=element_count)
    
    # Setup camera with AI-optimized positioning
    camera = setup_camera_for_anamorphic_view()
    if effect_type == "seoul_corner":
        # Seoul effect needs a specific viewing angle
        camera.location = (16, -13, 5)
        camera.rotation_euler = (math.radians(75), 0, math.radians(30))
    elif seoul_analysis and seoul_analysis['suggested_viewing_distance'] == "2-4 meters":
        # Adjust camera for close viewing
        camera.location = (20, -12, 4)
    
    # Setup lighting optimized for AI-processed content
    if effect_type == "seoul_corner":
        # Special lighting for Seoul LED effect - neon style
        setup_seoul_lighting()
    else:
        setup_professional_lighting()
    
    # Configure render settings
    setup_render_settings(effect_type)
    
    print("AI-Enhanced Anamorphic 3D Billboard created successfully!")
    print("AI Integration Features:")
    print("- Advanced depth map generation using MiDaS")
    print("- Seoul-style LED display optimization")
    print("- Intelligent anamorphic distortion")
    print("- Automated scene parameter adjustment")
    
    if output_path:
        bpy.context.scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True)
        print(f"Render saved to: {output_path}")

# Enhanced usage examples
def create_seoul_style_billboard(image_path):
    """Create Seoul-style LED billboard with AI optimization"""
    return main(image_path, effect_type="seoul_corner", ai_strength=2.0)

def create_screen_popup_billboard(image_path):
    """Create screen pop-out effect billboard"""
    return main(image_path, effect_type="screen_pop", ai_strength=1.8)

def create_shadow_box_billboard(image_path):
    """Create shadow box anamorphic billboard"""
    return main(image_path, effect_type="shadow_box", ai_strength=1.5)

# Example usage with AI integration
if __name__ == "__main__":
    # Set paths for your specific setup
    output_dir = Path('/Users/komalshahid/Desktop/Bellevue University/DSC680/projects/project3-colorful-canvas/data/output/blender_anamorphic/')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    IMAGE_PATH = str(Path(project_path) / "data/sample_images/sample1.jpg")
    OUTPUT_PATH = str(output_dir / "anamorphic_render.png")
    
    print("\n" + "="*60)
    print("🎬 BLENDER + AI ANAMORPHIC BILLBOARD GENERATOR")
    print("="*60)
    print("Usage Options:")
    print("1. create_seoul_style_billboard(IMAGE_PATH)")
    print("2. create_screen_popup_billboard(IMAGE_PATH)")
    print("3. create_shadow_box_billboard(IMAGE_PATH)")
    print("4. main(IMAGE_PATH, OUTPUT_PATH, effect_type, ai_strength)")
    print("="*60)
    
    # Uncomment the effect you want to use:
    # create_seoul_style_billboard(IMAGE_PATH)
    # create_screen_popup_billboard(IMAGE_PATH)
    create_shadow_box_billboard(IMAGE_PATH) 