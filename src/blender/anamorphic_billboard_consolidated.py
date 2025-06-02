import bpy
import math
from mathutils import Vector
import os, sys, random
from pathlib import Path

def step_3_setup_anamorphic_camera():
    """Step 3: Setup camera with LESS EXTREME angle for better visibility"""
    print("📷 STEP 3: Setting Up ADJUSTED Angle Camera...")
    
    # Clear existing cameras
    for obj in [o for o in bpy.data.objects if o.type=='CAMERA']:
        bpy.data.objects.remove(obj, do_unlink=True)
    
    # ADJUSTED camera position - less extreme angle
    camera_location = (15, -15, 6)  # Reduced from (25, -25, 8)
    camera_rotation = (1.0, 0, 0.5)  # Reduced from (1.2217, 0, 0.6109)
    
    bpy.ops.object.camera_add(location=camera_location)
    camera = bpy.context.active_object
    camera.name = "Anamorphic_Camera"
    camera.rotation_euler = camera_rotation
    
    # Set as active camera
    bpy.context.scene.camera = camera
    
    # Camera settings for anamorphic effect
    camera.data.lens = 35  # Wider lens
    camera.data.sensor_width = 36
    
    print(f"✅ Camera positioned at {camera_location} with rotation {camera_rotation}")
    print(f"✅ ADJUSTED angle for better content visibility")
    
    return camera

def render_step_verification(step_name, step_number, output_base):
    """Render intermediate step for verification"""
    step_output = output_base.replace('.png', f'_step_{step_number}_{step_name}.png')
    
    print(f"📸 VERIFICATION RENDER - Step {step_number}: {step_name}")
    
    # Quick render settings for verification
    original_samples = bpy.context.scene.cycles.samples
    bpy.context.scene.cycles.samples = 16  # Fast preview
    
    # Render step
    bpy.context.scene.render.filepath = step_output
    bpy.ops.render.render(write_still=True)
    
    # Restore original samples
    bpy.context.scene.cycles.samples = original_samples
    
    print(f"✅ Step verification saved: {step_output}")
    return step_output

def render_from_camera(
    camera_name: str,
    output_path: str,
    obj_to_frame: str = None,
    margin: float = 1.05,
    resolution_x: int = None,
    resolution_y: int = None,
    resolution_percent: int = 100,
    file_format: str = 'PNG'
):
    """
    Positions the named camera, optionally fitting it to `obj_to_frame`,
    then renders and saves an image.

    Args:
      camera_name:      Name of a CAMERA object in bpy.data.objects
      output_path:      Absolute path where the file will be written
      obj_to_frame:     (Optional) Name of an object to exactly fill the view
      margin:           >1.0 to add padding (1.0 = exact)
      resolution_x:     Override scene.render.resolution_x
      resolution_y:     Override scene.render.resolution_y
      resolution_percent: Percentage scale (0–100)
      file_format:      'PNG', 'JPEG', etc.
    """
    scene = bpy.context.scene

    # 1) Validate camera
    cam_obj = bpy.data.objects.get(camera_name)
    if not cam_obj or cam_obj.type != 'CAMERA':
        raise ValueError(f"Camera '{camera_name}' not found or not a camera")
    cam_data = cam_obj.data

    # 2) Apply zero lens shifts/skew
    cam_data.shift_x = 0.0
    cam_data.shift_y = 0.0
    cam_data.sensor_fit = 'HORIZONTAL'

    # 3) Fit camera to object if requested
    if obj_to_frame:
        obj = bpy.data.objects.get(obj_to_frame)
        if not obj:
            raise ValueError(f"Object '{obj_to_frame}' not found")
        # compute world-space bbox corners
        bbox = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
        xs = [v.x for v in bbox]; zs = [v.z for v in bbox]
        width  = max(xs) - min(xs)
        height = max(zs) - min(zs)
        center = Vector((
            (max(xs)+min(xs))/2,
            (max([v.y for v in bbox])+min([v.y for v in bbox]))/2,
            (max(zs)+min(zs))/2
        ))

        # FOVs
        fov_h = 2*math.atan(cam_data.sensor_width / (2 * cam_data.lens))
        aspect = (
            (scene.render.resolution_x * scene.render.pixel_aspect_x) /
            (scene.render.resolution_y * scene.render.pixel_aspect_y)
        )
        fov_v = 2*math.atan(math.tan(fov_h/2) / aspect)

        # required distances
        dist_h = (width/2)  / math.tan(fov_h/2)
        dist_v = (height/2) / math.tan(fov_v/2)
        distance = max(dist_h, dist_v) * margin

        # position camera along its local -Y
        cam_forward = cam_obj.matrix_world.to_3x3() @ Vector((0, -1, 0))
        cam_obj.location = center - cam_forward * distance

        # aim camera
        track = cam_obj.constraints.new(type='TRACK_TO')
        track.target     = obj
        track.track_axis = 'TRACK_NEGATIVE_Z'
        track.up_axis    = 'UP_Y'

        # adjust clipping
        cam_data.clip_start = max(0.001, distance * 0.01)
        cam_data.clip_end   = distance * 10

        print(f"[Camera Fit] '{camera_name}' → distance={distance:.2f}, margin={margin}")
    else:
        print(f"[Camera] Using existing pose of '{camera_name}'")

    # 4) Set render settings
    if resolution_x: scene.render.resolution_x = resolution_x
    if resolution_y: scene.render.resolution_y = resolution_y
    scene.render.resolution_percentage = resolution_percent
    scene.render.image_settings.file_format = file_format

    # 5) Assign camera and render
    scene.camera = cam_obj
    scene.render.filepath = output_path
    print(f"Rendering → {output_path} @ {scene.render.resolution_x}×{scene.render.resolution_y} ({resolution_percent}%)")
    bpy.ops.render.render(write_still=True)
    print("Render complete.")

def step_9_intelligent_camera_render(billboard_object_name, output_path):
    """Step 9: Use INTELLIGENT camera positioning to properly frame billboard"""
    print("🎯 STEP 9: INTELLIGENT Camera Positioning and Render...")
    
    # Clear any existing camera constraints first
    camera = bpy.data.objects.get("Anamorphic_Camera")
    if camera:
        # Remove existing constraints
        for constraint in camera.constraints:
            camera.constraints.remove(constraint)
    
    # Use intelligent camera positioning to frame the billboard perfectly
    render_from_camera(
        camera_name="Anamorphic_Camera",
        output_path=output_path,
        obj_to_frame=billboard_object_name,  # Frame the billboard exactly
        margin=1.1,  # Small margin for better framing
        resolution_x=1920,
        resolution_y=1080,
        resolution_percent=100,
        file_format='PNG'
    )
    
    print(f"✅ INTELLIGENT rendering complete: {output_path}")
    return output_path

def create_billboard_environment(
    ground_size: float = 50.0,
    building_height: float = 15.0,
    building_width: float = 10.0,
    validate_scene: bool = True
) -> dict:
    """
    Create a realistic billboard environment with proper 3D textured framework.
    
    Args:
        ground_size: Size of the ground plane
        building_height: Height of background buildings  
        building_width: Width of background buildings
        validate_scene: Whether to validate the created objects
        
    Returns:
        dict: Created objects with their names and references
    """
    if ground_size <= 0:
        raise ValueError(f"Ground size must be positive, got {ground_size}")
    if building_height <= 0:
        raise ValueError(f"Building height must be positive, got {building_height}")
        
    created_objects = {}
    
    try:
        print("🏗️ Creating 3D TEXTURED BACKGROUND FRAMEWORK...")
        
        # Create textured ground with subdivision for 3D detail
        bpy.ops.mesh.primitive_plane_add(size=ground_size, location=(0, 0, 0))
        ground = bpy.context.active_object
        ground.name = "Environment_Ground"
        
        # Add subdivision for 3D ground texture
        bpy.context.view_layer.objects.active = ground
        bpy.ops.object.mode_set(mode='EDIT')
        for _ in range(3):  # 3 levels for ground detail
            bpy.ops.mesh.subdivide()
        bpy.ops.object.mode_set(mode='OBJECT')
        
        # Create ground material with texture
        ground_mat = bpy.data.materials.new("Ground_3D_Material")
        ground_mat.use_nodes = True
        ground.data.materials.append(ground_mat)
        
        # Ground material nodes
        nodes = ground_mat.node_tree.nodes
        nodes.clear()
        output = nodes.new('ShaderNodeOutputMaterial')
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        noise = nodes.new('ShaderNodeTexNoise')
        noise.inputs['Scale'].default_value = 5.0
        noise.inputs['Detail'].default_value = 15.0
        
        # Connect ground texture
        ground_mat.node_tree.links.new(noise.outputs['Color'], bsdf.inputs['Base Color'])
        ground_mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
        
        created_objects["ground"] = ground
        
        # Create LEFT building with 3D texture detail
        bpy.ops.mesh.primitive_cube_add(size=building_width, location=(-20, 10, building_height/2))
        building_left = bpy.context.active_object
        building_left.name = "Building_Left"
        building_left.scale.z = building_height / building_width
        
        # Add 3D detail to building
        bpy.context.view_layer.objects.active = building_left
        bpy.ops.object.mode_set(mode='EDIT')
        for _ in range(2):  # Add subdivision for 3D building detail
            bpy.ops.mesh.subdivide()
        bpy.ops.object.mode_set(mode='OBJECT')
        
        # Building material with brick texture
        building_mat = bpy.data.materials.new("Building_3D_Material")
        building_mat.use_nodes = True
        building_left.data.materials.append(building_mat)
        
        # Building texture nodes
        nodes = building_mat.node_tree.nodes
        nodes.clear()
        output = nodes.new('ShaderNodeOutputMaterial')
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        brick = nodes.new('ShaderNodeTexBrick')
        brick.inputs['Scale'].default_value = 8.0
        brick.inputs['Color1'].default_value = (0.8, 0.7, 0.6, 1.0)  # Light brick
        brick.inputs['Color2'].default_value = (0.6, 0.5, 0.4, 1.0)  # Dark brick
        
        building_mat.node_tree.links.new(brick.outputs['Color'], bsdf.inputs['Base Color'])
        building_mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
        
        created_objects["building_left"] = building_left
        
        # Create RIGHT building with different texture
        bpy.ops.mesh.primitive_cube_add(size=building_width, location=(20, 10, building_height/2))
        building_right = bpy.context.active_object
        building_right.name = "Building_Right"
        building_right.scale.z = building_height / building_width
        
        # Add 3D detail
        bpy.context.view_layer.objects.active = building_right
        bpy.ops.object.mode_set(mode='EDIT')
        for _ in range(2):
            bpy.ops.mesh.subdivide()
        bpy.ops.object.mode_set(mode='OBJECT')
        
        # Different building material
        building_mat2 = bpy.data.materials.new("Building_Glass_Material")
        building_mat2.use_nodes = True
        building_right.data.materials.append(building_mat2)
        
        # Glass building texture
        nodes = building_mat2.node_tree.nodes
        nodes.clear()
        output = nodes.new('ShaderNodeOutputMaterial')
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        bsdf.inputs['Base Color'].default_value = (0.7, 0.8, 0.9, 1.0)  # Blue glass
        bsdf.inputs['Metallic'].default_value = 0.1
        bsdf.inputs['Roughness'].default_value = 0.1
        bsdf.inputs['Transmission'].default_value = 0.8  # Glass effect
        
        building_mat2.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
        
        created_objects["building_right"] = building_right
        
        # Create 3D street pole with metallic texture
        bpy.ops.mesh.primitive_cylinder_add(radius=0.3, depth=8, location=(15, -5, 4))
        pole = bpy.context.active_object
        pole.name = "Street_Pole"
        
        # Pole material
        pole_mat = bpy.data.materials.new("Pole_Metal_Material")
        pole_mat.use_nodes = True
        pole.data.materials.append(pole_mat)
        
        nodes = pole_mat.node_tree.nodes
        nodes.clear()
        output = nodes.new('ShaderNodeOutputMaterial')
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        bsdf.inputs['Base Color'].default_value = (0.3, 0.3, 0.3, 1.0)  # Dark metal
        bsdf.inputs['Metallic'].default_value = 1.0
        bsdf.inputs['Roughness'].default_value = 0.2
        
        pole_mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
        
        created_objects["pole"] = pole
        
        # ADD CLASSICAL FRAME ELEMENTS for benchmark style
        bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 4, 8))
        frame_top = bpy.context.active_object
        frame_top.name = "Classical_Frame_Top"
        frame_top.scale = (8, 0.5, 0.8)
        
        # Classical frame material
        frame_mat = bpy.data.materials.new("Classical_Frame_Material")
        frame_mat.use_nodes = True
        frame_top.data.materials.append(frame_mat)
        
        nodes = frame_mat.node_tree.nodes
        nodes.clear()
        output = nodes.new('ShaderNodeOutputMaterial')
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        bsdf.inputs['Base Color'].default_value = (0.8, 0.7, 0.5, 1.0)  # Gold classical
        bsdf.inputs['Metallic'].default_value = 0.8
        bsdf.inputs['Roughness'].default_value = 0.3
        
        frame_mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
        
        created_objects["frame_top"] = frame_top
        
        if validate_scene:
            # Validate all objects were created
            for name, obj in created_objects.items():
                if not obj or obj.name not in bpy.data.objects:
                    raise RuntimeError(f"Failed to create {name} object")
                    
        print(f"✅ 3D TEXTURED ENVIRONMENT created with {len(created_objects)} objects")
        return created_objects
        
    except Exception as e:
        print(f"❌ Environment creation failed: {e}")
        # Cleanup on failure
        for obj in created_objects.values():
            if obj and obj.name in bpy.data.objects:
                bpy.data.objects.remove(obj, do_unlink=True)
        raise

def create_billboard_geometry(
    width: float = 12.0,
    height: float = 8.0,
    location: tuple = (0, 5, 4),
    subdivision_levels: int = 4,
    validate_geometry: bool = True
) -> dict:
    """
    Create billboard geometry with proper subdivision and validation.
    
    Args:
        width: Billboard width in Blender units
        height: Billboard height in Blender units  
        location: World position (x, y, z)
        subdivision_levels: Number of subdivision cuts for detail
        validate_geometry: Whether to validate the created geometry
        
    Returns:
        dict: Created billboard objects and their properties
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Dimensions must be positive: width={width}, height={height}")
    if subdivision_levels < 0:
        raise ValueError(f"Subdivision levels cannot be negative: {subdivision_levels}")
        
    created_objects = {}
    
    try:
        # Create frame
        bpy.ops.mesh.primitive_cube_add(size=2)
        frame = bpy.context.active_object
        frame.name = "Billboard_Frame"
        frame.scale = (width/2, 0.25, height/2)
        frame.location = location
        created_objects["frame"] = frame
        
        # Create main billboard surface
        bpy.ops.mesh.primitive_plane_add(size=2)
        billboard = bpy.context.active_object
        billboard.name = "Billboard_Main"
        billboard.scale = (width/2 * 0.95, height/2 * 0.95, 1)  # Slightly smaller than frame
        billboard.location = (location[0], location[1] - 0.1, location[2])
        created_objects["billboard"] = billboard
        
        # Add subdivision for detail
        if subdivision_levels > 0:
            bpy.context.view_layer.objects.active = billboard
            bpy.ops.object.mode_set(mode='EDIT')
            for _ in range(subdivision_levels):
                bpy.ops.mesh.subdivide()
            bpy.ops.object.mode_set(mode='OBJECT')
            
        # UV unwrap for texture mapping
        bpy.context.view_layer.objects.active = billboard
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.unwrap()
        bpy.ops.object.mode_set(mode='OBJECT')
        
        if validate_geometry:
            # Validate mesh integrity
            if len(billboard.data.vertices) < 4:
                raise RuntimeError("Billboard mesh has insufficient vertices")
            if len(billboard.data.polygons) < 1:
                raise RuntimeError("Billboard mesh has no faces")
                
        print(f"✅ Billboard geometry created: {width}x{height} with {subdivision_levels} subdivision levels")
        return created_objects
        
    except Exception as e:
        print(f"❌ Billboard geometry creation failed: {e}")
        # Cleanup on failure
        for obj in created_objects.values():
            if obj and obj.name in bpy.data.objects:
                bpy.data.objects.remove(obj, do_unlink=True)
        raise

def apply_billboard_material(
    billboard_object: str,
    image_path: str,
    color_boost: float = 1.8,  # Enhanced brightness 
    saturation_boost: float = 4.0,  # PROVEN: Quadruple saturation for vibrancy
    validate_material: bool = True
) -> str:
    """
    Apply material and texture to billboard with professional node setup.
    
    Args:
        billboard_object: Name of the billboard object
        image_path: Path to the image file
        color_boost: Multiplier for color intensity (>1.0 brightens)
        saturation_boost: Multiplier for color saturation (>1.0 more vibrant)
        validate_material: Whether to validate the material setup
        
    Returns:
        str: Name of the created material
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if color_boost <= 0 or saturation_boost <= 0:
        raise ValueError(f"Boost values must be positive: color={color_boost}, saturation={saturation_boost}")
        
    billboard = bpy.data.objects.get(billboard_object)
    if not billboard:
        raise ValueError(f"Billboard object '{billboard_object}' not found")
        
    try:
        # Load image
        image = bpy.data.images.load(image_path)
        print(f"✅ Image loaded: {image.name} ({image.size[0]}x{image.size[1]})")
        
        # Create material
        mat_name = f"{billboard_object}_Material"
        mat = bpy.data.materials.new(mat_name)
        mat.use_nodes = True
        billboard.data.materials.clear()
        billboard.data.materials.append(mat)
        
        # Setup professional node tree
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()
        
        # Output node
        output = nodes.new('ShaderNodeOutputMaterial')
        output.location = (600, 0)
        
        # Principled BSDF
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        bsdf.location = (400, 0)
        
        # Color enhancement nodes
        if color_boost != 1.0 or saturation_boost != 1.0:
            hsv = nodes.new('ShaderNodeHueSaturation')
            hsv.location = (200, 0)
            hsv.inputs['Saturation'].default_value = saturation_boost
            hsv.inputs['Value'].default_value = color_boost
            
            # Image texture
            img_node = nodes.new('ShaderNodeTexImage')
            img_node.image = image
            img_node.location = (0, 0)
            
            # Connect enhanced pipeline
            links.new(img_node.outputs['Color'], hsv.inputs['Color'])
            links.new(hsv.outputs['Color'], bsdf.inputs['Base Color'])
        else:
            # Direct connection
            img_node = nodes.new('ShaderNodeTexImage')
            img_node.image = image
            img_node.location = (0, 0)
            links.new(img_node.outputs['Color'], bsdf.inputs['Base Color'])
            
        # Final connection
        links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
        
        if validate_material:
            if not mat.node_tree.nodes:
                raise RuntimeError("Material node tree is empty")
            if not billboard.data.materials:
                raise RuntimeError("Material not applied to billboard")
                
        print(f"✅ Professional material applied with color boost: {color_boost}x, saturation: {saturation_boost}x")
        return mat_name
        
    except Exception as e:
        print(f"❌ Material application failed: {e}")
        raise

def create_3d_popout_effect(
    billboard_object: str,
    image_path: str,
    displacement_strength: float = 4.0,  # PROVEN: 4x displacement for dramatic pop-out
    subdivision_levels: int = 7,  # PROVEN: 7 levels for ultra-smooth displacement
    use_depth_estimation: bool = True
) -> str:
    """
    Create 3D pop-out effect from 2D image using displacement mapping.
    
    Args:
        billboard_object: Name of billboard to apply effect to
        image_path: Path to source image  
        displacement_strength: How far geometry pops out (0.0-5.0)
        subdivision_levels: Mesh detail level for displacement
        use_depth_estimation: Whether to estimate depth from image
        
    Returns:
        str: Name of created displacement material
    """
    billboard = bpy.data.objects.get(billboard_object)
    if not billboard:
        raise ValueError(f"Billboard '{billboard_object}' not found")
        
    print(f"🎭 Creating ENHANCED 3D POP-OUT effect: {displacement_strength}x strength with 4x color saturation")
    
    # Add high subdivision for smooth displacement
    bpy.context.view_layer.objects.active = billboard
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    
    # Add multiple subdivision levels for smooth 3D effect
    for i in range(subdivision_levels):
        bpy.ops.mesh.subdivide()
        print(f"  ✅ Subdivision level {i+1}/{subdivision_levels}")
    
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Create displacement material
    mat_name = f"{billboard_object}_3D_PopOut"
    mat = bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    billboard.data.materials.clear()
    billboard.data.materials.append(mat)
    
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    # Load main image
    image = bpy.data.images.load(image_path)
    
    # === ENHANCED 3D MATERIAL SETUP ===
    
    # Output
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (800, 0)
    
    # Principled BSDF with enhanced settings for 3D pop-out
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (600, 0)
    bsdf.inputs['Roughness'].default_value = 0.2  # More glossy for 3D effect
    bsdf.inputs['Specular'].default_value = 0.9   # Enhanced reflections for depth
    bsdf.inputs['Clearcoat'].default_value = 0.3  # Additional depth layer
    
    # ENHANCED COLOR PIPELINE - 4x saturation integration
    hsv = nodes.new('ShaderNodeHueSaturation') 
    hsv.location = (400, 0)
    hsv.inputs['Saturation'].default_value = 4.0  # ENHANCED: 4x saturation for dramatic colors
    hsv.inputs['Value'].default_value = 1.8       # ENHANCED: 80% brighter for pop-out visibility
    hsv.inputs['Hue'].default_value = 0.05        # Slight hue shift for vibrancy
    
    # Color boost for 3D depth perception
    gamma = nodes.new('ShaderNodeGamma')
    gamma.location = (350, 0)
    gamma.inputs['Gamma'].default_value = 1.1  # Enhance gamma for depth
    
    # Main image texture
    img_node = nodes.new('ShaderNodeTexImage')
    img_node.image = image
    img_node.location = (0, 0)
    
    # === ENHANCED 3D DISPLACEMENT SETUP ===
    
    # Displacement node with optimized settings
    displacement = nodes.new('ShaderNodeDisplacement')
    displacement.location = (600, -200)
    displacement.inputs['Scale'].default_value = displacement_strength
    displacement.inputs['Midlevel'].default_value = 0.2  # Better neutral point
    
    # Enhanced color ramp for better 3D depth mapping
    coloramp = nodes.new('ShaderNodeValToRGB')
    coloramp.location = (200, -200)
    # Optimize color ramp for dramatic 3D effect
    coloramp.color_ramp.elements[0].position = 0.15  # Better shadows
    coloramp.color_ramp.elements[1].position = 0.85  # Better highlights
    coloramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)  # Deep black
    coloramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)  # Pure white
    
    # RGB to BW conversion for enhanced displacement
    rgb_to_bw = nodes.new('ShaderNodeRGBToBW')
    rgb_to_bw.location = (100, -200)
    
    # Use same image for displacement (converted to BW)
    displacement_img = nodes.new('ShaderNodeTexImage') 
    displacement_img.image = image
    displacement_img.location = (0, -200)
    
    # === ENHANCED NODE CONNECTIONS ===
    
    # Enhanced color pipeline with multiple stages
    links.new(img_node.outputs['Color'], gamma.inputs['Color'])
    links.new(gamma.outputs['Color'], hsv.inputs['Color'])
    links.new(hsv.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    # Enhanced displacement pipeline with better depth mapping
    links.new(displacement_img.outputs['Color'], rgb_to_bw.inputs['Color'])
    links.new(rgb_to_bw.outputs['Val'], coloramp.inputs['Fac'])
    links.new(coloramp.outputs['Color'], displacement.inputs['Height'])
    links.new(displacement.outputs['Displacement'], output.inputs['Displacement'])
    
    # Enable displacement in material settings
    mat.cycles.displacement_method = 'BOTH'  # Use both bump and true displacement
    
    print(f"✅ ENHANCED 3D POP-OUT material created: {mat_name}")
    print(f"   🎨 4x color saturation + 1.8x brightness")
    print(f"   🎭 {displacement_strength}x displacement with {subdivision_levels} subdivision levels")
    print(f"   ✨ Enhanced depth perception with gamma correction")
    
    return mat_name

def setup_professional_lighting(
    key_light_energy: float = 3000.0,  # PROVEN: Enhanced dramatic lighting
    fill_light_energy: float = 1000.0,  # PROVEN: Stronger fill for character visibility
    rim_light_energy: float = 1500.0,  # PROVEN: Enhanced rim for character definition
    environment_strength: float = 0.3
) -> dict:
    """
    Setup professional 3-point lighting with shadows for cinema quality.
    
    Args:
        key_light_energy: Main light strength
        fill_light_energy: Fill light strength  
        rim_light_energy: Rim/back light strength
        environment_strength: HDRI environment strength
        
    Returns:
        dict: Created lighting objects
    """
    print("💡 Setting up PROFESSIONAL CINEMA LIGHTING...")
    
    # Remove existing lights
    for obj in [o for o in bpy.data.objects if o.type == 'LIGHT']:
        bpy.data.objects.remove(obj, do_unlink=True)
    
    lights = {}
    
    # === KEY LIGHT (Main dramatic light) ===
    bpy.ops.object.light_add(type='AREA', location=(15, -20, 15))
    key_light = bpy.context.active_object
    key_light.name = "Key_Light_Professional"
    key_light.data.energy = key_light_energy
    key_light.data.size = 8
    key_light.data.color = (1.0, 0.95, 0.85)  # Warm white
    key_light.rotation_euler = (0.8, 0, 0.5)
    lights["key"] = key_light
    
    # === FILL LIGHT (Soften shadows) ===
    bpy.ops.object.light_add(type='AREA', location=(-10, -15, 8))
    fill_light = bpy.context.active_object  
    fill_light.name = "Fill_Light_Professional"
    fill_light.data.energy = fill_light_energy
    fill_light.data.size = 12
    fill_light.data.color = (0.9, 0.95, 1.0)  # Cool fill
    fill_light.rotation_euler = (1.0, 0, -0.3)
    lights["fill"] = fill_light
    
    # === RIM LIGHT (Edge definition) ===
    bpy.ops.object.light_add(type='SPOT', location=(5, 15, 12))
    rim_light = bpy.context.active_object
    rim_light.name = "Rim_Light_Professional" 
    rim_light.data.energy = rim_light_energy
    rim_light.data.spot_size = 1.2
    rim_light.data.color = (1.0, 0.9, 0.7)  # Warm rim
    rim_light.rotation_euler = (-0.5, 0, 3.14)
    lights["rim"] = rim_light
    
    # === ENVIRONMENT LIGHTING ===
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("Professional_World")
        bpy.context.scene.world = world
        
    world.use_nodes = True
    world_nodes = world.node_tree.nodes
    world_nodes.clear()
    
    # Environment texture
    env_output = world_nodes.new('ShaderNodeOutputWorld')
    env_output.location = (400, 0)
    
    background = world_nodes.new('ShaderNodeBackground')
    background.location = (200, 0)
    background.inputs['Strength'].default_value = environment_strength
    background.inputs['Color'].default_value = (0.1, 0.15, 0.25, 1.0)  # Subtle blue
    
    world.node_tree.links.new(background.outputs['Background'], env_output.inputs['Surface'])
    
    print(f"✅ Professional lighting setup: {len(lights)} lights + environment")
    return lights

def setup_cinema_quality_render(
    samples: int = 1024,  # PROVEN: Ultra-high quality for character details
    resolution_x: int = 1920,
    resolution_y: int = 1080,
    use_denoising: bool = True,
    use_motion_blur: bool = False
) -> dict:
    """
    Setup cinema-quality render settings for professional output.
    
    Args:
        samples: Render samples (higher = better quality)
        resolution_x: Output width
        resolution_y: Output height  
        use_denoising: Enable AI denoising
        use_motion_blur: Enable motion blur
        
    Returns:
        dict: Applied render settings
    """
    print(f"🎬 Setting up CINEMA QUALITY rendering: {samples} samples")
    
    scene = bpy.context.scene
    
    # === CYCLES RENDER ENGINE ===
    scene.render.engine = 'CYCLES'
    
    # High quality sampling
    scene.cycles.samples = samples
    scene.cycles.preview_samples = samples // 4
    
    # === RESOLUTION & QUALITY ===
    scene.render.resolution_x = resolution_x
    scene.render.resolution_y = resolution_y
    scene.render.resolution_percentage = 100
    
    # === PERFORMANCE OPTIMIZATIONS ===
    scene.cycles.device = 'GPU'  # Use GPU if available
    scene.cycles.tile_size = 512  # Optimal for GPU
    
    # === POST-PROCESSING ===
    if use_denoising:
        scene.cycles.use_denoising = True
        scene.cycles.denoiser = 'OPENIMAGEDENOISE'
        
    # === COLOR MANAGEMENT ===
    scene.view_settings.view_transform = 'Filmic'
    scene.view_settings.look = 'High Contrast'
    scene.sequencer_colorspace_settings.name = 'sRGB'
    
    # === MOTION BLUR ===
    if use_motion_blur:
        scene.render.use_motion_blur = True
        scene.render.motion_blur_shutter = 0.5
    
    settings = {
        'samples': samples,
        'resolution': (resolution_x, resolution_y),
        'denoising': use_denoising,
        'color_management': 'Filmic High Contrast'
    }
    
    print(f"✅ Cinema quality render setup complete")
    return settings 