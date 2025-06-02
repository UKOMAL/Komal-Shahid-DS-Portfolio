import bpy
import os
import sys
import math
from mathutils import Vector

def test_intelligent_camera_positioning(image_path, output_path):
    """Test the new intelligent camera positioning"""
    print("🎯 Testing INTELLIGENT Camera Positioning")
    
    # Clear scene
    clear_scene()
    
    # Create billboard
    bpy.ops.mesh.primitive_plane_add(size=8, location=(0, 5, 4))
    billboard = bpy.context.active_object
    billboard.name = "Test_Billboard"
    billboard.rotation_euler = (1.57, 0, 0)  # 90 degrees
    
    # Load and apply image
    if os.path.exists(image_path):
        image = bpy.data.images.load(image_path)
        print(f"✅ Image loaded: {image.name} ({image.size[0]}x{image.size[1]})")
        
        # Create material
        mat = bpy.data.materials.new("Billboard_Material")
        mat.use_nodes = True
        billboard.data.materials.append(mat)
        
        # Setup nodes
        nodes = mat.node_tree.nodes
        nodes.clear()
        
        output_node = nodes.new('ShaderNodeOutputMaterial')
        output_node.location = (400, 0)
        
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        bsdf.location = (200, 0)
        
        img_node = nodes.new('ShaderNodeTexImage')
        img_node.image = image
        img_node.location = (0, 0)
        
        # Connect nodes
        links = mat.node_tree.links
        links.new(img_node.outputs['Color'], bsdf.inputs['Base Color'])
        links.new(bsdf.outputs['BSDF'], output_node.inputs['Surface'])
        
        print("✅ Material and texture applied")
    
    # Create camera
    bpy.ops.object.camera_add(location=(10, -10, 5))
    camera = bpy.context.active_object
    camera.name = "Anamorphic_Camera"
    
    # INTELLIGENT CAMERA POSITIONING
    scene = bpy.context.scene
    cam_data = camera.data
    
    # Apply zero lens shifts
    cam_data.shift_x = 0.0
    cam_data.shift_y = 0.0
    cam_data.sensor_fit = 'HORIZONTAL'
    
    # Fit camera to billboard
    obj = billboard
    bbox = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    xs = [v.x for v in bbox]; zs = [v.z for v in bbox]
    width  = max(xs) - min(xs)
    height = max(zs) - min(zs)
    center = Vector((
        (max(xs)+min(xs))/2,
        (max([v.y for v in bbox])+min([v.y for v in bbox]))/2,
        (max(zs)+min(zs))/2
    ))
    
    # Calculate FOVs and distance
    fov_h = 2*math.atan(cam_data.sensor_width / (2 * cam_data.lens))
    aspect = scene.render.resolution_x / scene.render.resolution_y
    fov_v = 2*math.atan(math.tan(fov_h/2) / aspect)
    
    dist_h = (width/2)  / math.tan(fov_h/2)
    dist_v = (height/2) / math.tan(fov_v/2)
    distance = max(dist_h, dist_v) * 1.1  # 10% margin
    
    # Position camera
    cam_forward = camera.matrix_world.to_3x3() @ Vector((0, -1, 0))
    camera.location = center - cam_forward * distance
    
    # Aim camera at billboard
    track = camera.constraints.new(type='TRACK_TO')
    track.target = billboard
    track.track_axis = 'TRACK_NEGATIVE_Z'
    track.up_axis = 'UP_Y'
    
    # Set clipping
    cam_data.clip_start = max(0.001, distance * 0.01)
    cam_data.clip_end = distance * 10
    
    print(f"🎯 INTELLIGENT camera positioned: distance={distance:.2f}")
    
    # Set as active camera and render
    scene.camera = camera
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 64
    
    # Render
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    
    print(f"✅ INTELLIGENT billboard rendered: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) > 4:
        image_path = sys.argv[4]  # After --
        output_path = sys.argv[5]
        
        if "INTELLIGENT" in output_path:
            test_intelligent_camera_positioning(image_path, output_path)
        else:
            create_simple_textured_plane(image_path, output_path)
    else:
        print("Usage: blender --background --python debug_billboard_texture.py -- <image_path> <output_path>") 