#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Builder Module - 3D Rendering and Blender Integration
Consolidates all Blender and 3D rendering functionality for the Colorful Canvas project.

Author: Komal Shahid
Course: DSC680 - Bellevue University
Project: Colorful Canvas AI Art Studio
"""

import os
import sys
from typing import Union, Optional, Dict, List, Tuple, Any
from pathlib import Path
import numpy as np
from PIL import Image
import subprocess
import json

# Blender Python for professional 3D anamorphic rendering
import bpy
import bmesh
from mathutils import Vector, Matrix
print("✅ Blender Python (bpy) available - Professional 3D anamorphic rendering enabled")


class BlenderAnamorphicRenderer:
    """
    Professional Blender-based anamorphic rendering system
    Creates complex 3D illusions and anamorphic effects using Blender's powerful rendering engine
    """
    
    def __init__(self):
        """Initialize Blender renderer with optimal settings"""
        self.blender_available = True
        self.setup_blender_scene()
        self.output_dir = Path("./data/output/blender_renders/")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def setup_blender_scene(self):
        """Configure Blender scene with optimal lighting and camera setup for anamorphic effects"""
        # Clear existing mesh objects
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False, confirm=False)
        
        # Set render engine to Cycles for photorealistic results
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.device = 'GPU'
        
        # Configure render settings for high quality
        bpy.context.scene.render.resolution_x = 1920
        bpy.context.scene.render.resolution_y = 1080
        bpy.context.scene.render.resolution_percentage = 100
        bpy.context.scene.cycles.samples = 128
        
        # Add camera with optimal positioning for anamorphic viewing
        bpy.ops.object.camera_add(location=(7, -7, 5))
        camera = bpy.context.object
        camera.rotation_euler = (1.1, 0, 0.785)
        bpy.context.scene.camera = camera
        
        # Add professional lighting setup
        self._setup_professional_lighting()
        
        print("✅ Blender scene configured for anamorphic rendering")
    
    def _setup_professional_lighting(self):
        """Setup professional 3-point lighting for optimal anamorphic effect visibility"""
        # Key light (main illumination)
        bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
        key_light = bpy.context.object
        key_light.data.energy = 5
        key_light.name = "KeyLight"
        
        # Fill light (soften shadows)
        bpy.ops.object.light_add(type='AREA', location=(-3, 3, 5))
        fill_light = bpy.context.object
        fill_light.data.energy = 2
        fill_light.data.size = 5
        fill_light.name = "FillLight"
        
        # Rim light (edge definition)
        bpy.ops.object.light_add(type='SPOT', location=(0, -8, 3))
        rim_light = bpy.context.object
        rim_light.data.energy = 3
        rim_light.data.spot_size = 1.5
        rim_light.name = "RimLight"
    
    def create_display_box(self):
        """Create a display box for corner anamorphic effects"""
        # Create the main display box structure
        bpy.ops.mesh.cube_add(size=2, location=(0, 0, 0))
        display_box = bpy.context.object
        display_box.name = "DisplayBox"
        
        # Scale to create box proportions
        display_box.scale = (2, 1.5, 0.8)
        
        # Add materials for realistic appearance
        material = bpy.data.materials.new(name="BoxMaterial")
        material.use_nodes = True
        
        # Configure material nodes for realistic box appearance
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        
        # Clear default nodes
        nodes.clear()
        
        # Add principled BSDF for realistic material
        principled = nodes.new(type='ShaderNodeBsdfPrincipled')
        principled.inputs['Base Color'].default_value = (0.8, 0.8, 0.8, 1.0)
        principled.inputs['Metallic'].default_value = 0.0
        principled.inputs['Roughness'].default_value = 0.3
        
        # Add output node
        output = nodes.new(type='ShaderNodeOutputMaterial')
        links.new(principled.outputs['BSDF'], output.inputs['Surface'])
        
        # Assign material to object
        display_box.data.materials.append(material)
        
        return display_box
    
    def create_floating_logo(self, logo_text="CG", position=(0, 0, 1)):
        """Create floating 3D logo for anamorphic display"""
        # Create text object
        bpy.ops.object.text_add(location=position)
        logo = bpy.context.object
        logo.data.body = logo_text
        logo.data.size = 1.5
        logo.data.extrude = 0.2  # Give text 3D depth
        
        # Add emission material for glowing effect
        material = bpy.data.materials.new(name="LogoMaterial")
        material.use_nodes = True
        
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        nodes.clear()
        
        # Create emission shader for glowing effect
        emission = nodes.new(type='ShaderNodeEmission')
        emission.inputs['Color'].default_value = (0.2, 0.8, 1.0, 1.0)  # Cyan glow
        emission.inputs['Strength'].default_value = 2.0
        
        output = nodes.new(type='ShaderNodeOutputMaterial')
        links.new(emission.outputs['Emission'], output.inputs['Surface'])
        
        logo.data.materials.append(material)
        
        return logo
    
    def create_floating_objects(self, frame_number=1):
        """Create animated floating objects for dynamic anamorphic scenes"""
        objects = []
        
        # Configuration for different floating objects
        object_configs = [
            {"type": "sphere", "location": (2, 1, 2), "scale": 0.5, "color": (1, 0.2, 0.2)},
            {"type": "cube", "location": (-1, 2, 1.5), "scale": 0.3, "color": (0.2, 1, 0.2)},
            {"type": "cylinder", "location": (1, -1, 2.5), "scale": 0.4, "color": (0.2, 0.2, 1)},
            {"type": "cone", "location": (-2, -1, 1.8), "scale": 0.6, "color": (1, 1, 0.2)}
        ]
        
        for i, config in enumerate(object_configs):
            obj = self.create_object_by_type(config)
            
            # Add animation for floating effect
            self._add_floating_animation(obj, frame_number, i)
            objects.append(obj)
        
        return objects
    
    def create_object_by_type(self, config):
        """Create different types of 3D objects based on configuration"""
        obj_type = config["type"]
        location = config["location"]
        scale = config["scale"]
        color = config["color"]
        
        # Create object based on type
        if obj_type == "sphere":
            bpy.ops.mesh.uv_sphere_add(location=location)
        elif obj_type == "cube":
            bpy.ops.mesh.cube_add(location=location)
        elif obj_type == "cylinder":
            bpy.ops.mesh.cylinder_add(location=location)
        elif obj_type == "cone":
            bpy.ops.mesh.cone_add(location=location)
        else:
            bpy.ops.mesh.cube_add(location=location)  # Default fallback
        
        obj = bpy.context.object
        obj.scale = (scale, scale, scale)
        
        # Add material with specified color
        material = bpy.data.materials.new(name=f"{obj_type}_material")
        material.use_nodes = True
        
        nodes = material.node_tree.nodes
        principled = nodes.get("Principled BSDF")
        if principled:
            principled.inputs['Base Color'].default_value = (*color, 1.0)
            principled.inputs['Metallic'].default_value = 0.1
            principled.inputs['Roughness'].default_value = 0.2
        
        obj.data.materials.append(material)
        
        return obj
    
    def _add_floating_animation(self, obj, frame_number, object_index):
        """Add floating animation to objects"""
        # Set keyframes for floating motion
        for frame in range(1, 250, 10):
            bpy.context.scene.frame_set(frame)
            
            # Calculate floating motion
            time_offset = object_index * 0.5  # Phase offset for each object
            height_offset = 0.5 * np.sin((frame + time_offset) * 0.1)
            rotation_offset = frame * 0.02 + object_index
            
            # Apply animation
            obj.location.z += height_offset
            obj.rotation_euler.z = rotation_offset
            
            # Insert keyframes
            obj.keyframe_insert(data_path="location", index=2)
            obj.keyframe_insert(data_path="rotation_euler", index=2)
    
    def render_anamorphic_sequence(self, num_frames=9, output_dir="./data/output/blender_sequence/"):
        """Render sequence of anamorphic frames from different viewing angles"""
        os.makedirs(output_dir, exist_ok=True)
        
        camera = bpy.context.scene.camera
        base_location = camera.location.copy()
        
        for frame in range(num_frames):
            # Calculate viewing angle variation
            angle_step = 360 / num_frames
            current_angle = frame * angle_step
            
            # Move camera in circular path
            radius = 8
            x = radius * np.cos(np.radians(current_angle))
            y = radius * np.sin(np.radians(current_angle))
            
            camera.location = (x, y, base_location.z)
            
            # Point camera at center
            direction = Vector((0, 0, 0)) - camera.location
            camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
            
            # Set frame and render
            bpy.context.scene.frame_set(frame + 1)
            
            # Configure output path
            output_path = os.path.join(output_dir, f"anamorphic_frame_{frame:03d}.png")
            bpy.context.scene.render.filepath = output_path
            
            # Render frame
            bpy.ops.render.render(write_still=True)
            
            print(f"✅ Rendered frame {frame + 1}/{num_frames}: {output_path}")
        
        print(f"🎬 Sequence complete: {num_frames} frames in {output_dir}")
        return output_dir
    
    def create_capital_group_sequence(self):
        """Create professional Capital Group branded anamorphic sequence"""
        # Setup professional scene
        self.setup_capital_group_scene()
        
        # Create logo and branding elements
        logo = self.create_floating_logo("CG", position=(0, 0, 2))
        
        # Add floating elements
        floating_objects = self.create_floating_objects()
        
        # Render sequence
        return self.render_anamorphic_sequence(
            num_frames=12,
            output_dir="./data/output/capital_group_sequence/"
        )
    
    def setup_capital_group_scene(self):
        """Setup scene with Capital Group branding and professional aesthetics"""
        # Enhanced lighting for professional look
        self._setup_professional_lighting()
        
        # Add professional backdrop
        bpy.ops.mesh.plane_add(size=20, location=(0, 0, -1))
        backdrop = bpy.context.object
        backdrop.name = "ProfessionalBackdrop"
        
        # Add professional material to backdrop
        material = bpy.data.materials.new(name="ProfessionalFloor")
        material.use_nodes = True
        
        nodes = material.node_tree.nodes
        nodes.clear()
        
        # Create sophisticated material
        principled = nodes.new(type='ShaderNodeBsdfPrincipled')
        principled.inputs['Base Color'].default_value = (0.05, 0.05, 0.1, 1.0)  # Dark professional
        principled.inputs['Metallic'].default_value = 0.8
        principled.inputs['Roughness'].default_value = 0.1
        
        output = nodes.new(type='ShaderNodeOutputMaterial')
        material.node_tree.links.new(principled.outputs['BSDF'], output.inputs['Surface'])
        
        backdrop.data.materials.append(material)

    def create_character_frame_projection(self, output_dir="./data/output/blender_anamorphic/"):
        """
        Create a character projecting from a frame effect similar to the reference image
        This creates a furry red character coming out of a frame with a bridge scene
        """
        print("🎭 Creating character frame projection effect...")
        
        # Setup scene
        self.setup_blender_scene()
        os.makedirs(output_dir, exist_ok=True)
        
        # Create frame structure
        bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))
        frame = bpy.context.object
        frame.name = "DisplayFrame"
        
        # Scale to create frame proportions
        frame.scale = (3, 0.1, 2)
        
        # Create frame material
        frame_material = bpy.data.materials.new(name="FrameMaterial")
        frame_material.use_nodes = True
        nodes = frame_material.node_tree.nodes
        links = frame_material.node_tree.links
        nodes.clear()
        
        # Add principled BSDF for marble-like frame
        principled = nodes.new(type='ShaderNodeBsdfPrincipled')
        principled.inputs['Base Color'].default_value = (0.9, 0.85, 0.75, 1.0)  # Ivory color
        principled.inputs['Metallic'].default_value = 0.1
        principled.inputs['Roughness'].default_value = 0.3
        
        # Add output node
        output = nodes.new(type='ShaderNodeOutputMaterial')
        links.new(principled.outputs['BSDF'], output.inputs['Surface'])
        
        # Assign material to frame
        frame.data.materials.append(frame_material)
        
        # Create character mesh
        bpy.ops.mesh.primitive_monkey_add(size=1.5, location=(0, 1, 0))
        character = bpy.context.object
        character.name = "FurryCharacter"
        
        # Position the character to appear coming out of the frame
        character.location = (0, 0.8, 0)
        character.rotation_euler = (0, 0, 0)
        character.scale = (1.2, 1.2, 1.2)
        
        # Subdivide for smoother geometry
        bpy.ops.object.select_all(action='DESELECT')
        character.select_set(True)
        bpy.context.view_layer.objects.active = character
        bpy.ops.object.modifier_add(type='SUBSURF')
        character.modifiers["Subdivision"].levels = 2
        
        # Add particle system for fur
        bpy.ops.object.particle_system_add()
        particle_system = character.particle_systems[0]
        particle_settings = particle_system.settings
        
        # Configure fur settings
        particle_settings.type = 'HAIR'
        particle_settings.count = 10000
        particle_settings.hair_length = 0.2
        particle_settings.child_type = 'INTERPOLATED'
        particle_settings.child_nbr = 5
        particle_settings.clump_factor = 0.8
        
        # Create red furry material
        fur_material = bpy.data.materials.new(name="RedFurMaterial")
        fur_material.use_nodes = True
        nodes = fur_material.node_tree.nodes
        links = fur_material.node_tree.links
        nodes.clear()
        
        # Create red fur material
        principled = nodes.new(type='ShaderNodeBsdfPrincipled')
        principled.inputs['Base Color'].default_value = (0.9, 0.1, 0.1, 1.0)  # Bright red
        principled.inputs['Subsurface'].default_value = 0.15
        principled.inputs['Subsurface Color'].default_value = (1.0, 0.2, 0.2, 1.0)
        principled.inputs['Roughness'].default_value = 0.9
        
        output = nodes.new(type='ShaderNodeOutputMaterial')
        links.new(principled.outputs['BSDF'], output.inputs['Surface'])
        
        # Assign material to character
        character.data.materials.append(fur_material)
        
        # Create background scene (inside the frame)
        # Bridge
        bpy.ops.mesh.primitive_cube_add(size=1, location=(0, -0.5, -0.5))
        bridge = bpy.context.object
        bridge.name = "Bridge"
        bridge.scale = (1.5, 0.1, 0.1)
        
        # Bridge material (blue)
        bridge_material = bpy.data.materials.new(name="BlueBridgeMaterial")
        bridge_material.use_nodes = True
        nodes = bridge_material.node_tree.nodes
        links = bridge_material.node_tree.links
        nodes.clear()
        
        principled = nodes.new(type='ShaderNodeBsdfPrincipled')
        principled.inputs['Base Color'].default_value = (0.1, 0.4, 0.8, 1.0)  # Blue
        principled.inputs['Metallic'].default_value = 0.7
        
        output = nodes.new(type='ShaderNodeOutputMaterial')
        links.new(principled.outputs['BSDF'], output.inputs['Surface'])
        
        bridge.data.materials.append(bridge_material)
        
        # Create balloon objects
        balloon_positions = [
            (1.5, -1, 0.5),   # Right
            (-1.5, -1, 0.5),  # Left
            (0, -1, 1.2)      # Top
        ]
        
        balloon_colors = [
            (0.1, 0.1, 0.9, 1.0),  # Blue
            (0.9, 0.1, 0.3, 1.0),  # Red
            (0.9, 0.7, 0.1, 1.0)   # Yellow
        ]
        
        for i, (pos, color) in enumerate(zip(balloon_positions, balloon_colors)):
            bpy.ops.mesh.uv_sphere_add(radius=0.3, location=pos)
            balloon = bpy.context.object
            balloon.name = f"Balloon_{i+1}"
            
            # Balloon material
            balloon_material = bpy.data.materials.new(name=f"BalloonMaterial_{i+1}")
            balloon_material.use_nodes = True
            nodes = balloon_material.node_tree.nodes
            links = balloon_material.node_tree.links
            nodes.clear()
            
            principled = nodes.new(type='ShaderNodeBsdfPrincipled')
            principled.inputs['Base Color'].default_value = color
            principled.inputs['Specular'].default_value = 0.5
            principled.inputs['Roughness'].default_value = 0.1
            
            output = nodes.new(type='ShaderNodeOutputMaterial')
            links.new(principled.outputs['BSDF'], output.inputs['Surface'])
            
            balloon.data.materials.append(balloon_material)
        
        # Create HDRI background
        world = bpy.context.scene.world
        world.use_nodes = True
        world_nodes = world.node_tree.nodes
        world_links = world.node_tree.links
        
        # Add environment texture for sky
        env_texture = world_nodes.new(type='ShaderNodeTexEnvironment')
        env_texture.image = bpy.data.images.new('SkyHDRI', 1024, 512)
        
        # Connect environment texture to background
        world_links.new(env_texture.outputs['Color'], world_nodes['Background'].inputs['Color'])
        
        # Create camera for better viewing angle
        bpy.ops.object.camera_add(location=(5, 3, 1))
        camera = bpy.context.object
        camera.rotation_euler = (1.57, 0, 2.3)
        bpy.context.scene.camera = camera
        
        # Render different angles
        self.render_character_angles(output_dir)
        
        print("✅ Character frame projection created successfully")
        return output_dir
    
    def render_character_angles(self, output_dir):
        """Render character from different angles"""
        camera = bpy.context.scene.camera
        original_location = camera.location.copy()
        original_rotation = camera.rotation_euler.copy()
        
        angles = [
            {"name": "anamorphic_render_", "loc": (5, 3, 1), "rot": (1.57, 0, 2.3)},
            {"name": "angle_01_", "loc": (5, 3, 2), "rot": (1.4, 0, 2.0)},
            {"name": "angle_02_", "loc": (6, 1, 2), "rot": (1.3, 0, 1.8)},
            {"name": "angle_03_", "loc": (5, 3, 1), "rot": (1.57, 0, 2.3)},
            {"name": "angle_04_", "loc": (5, 3, 1), "rot": (1.57, 0, 2.3)},
            {"name": "angle_05_", "loc": (5, 3, 1), "rot": (1.57, 0, 2.3)}
        ]
        
        for angle in angles:
            # Set camera position
            camera.location = angle["loc"]
            camera.rotation_euler = angle["rot"]
            
            # Render frame
            output_path = os.path.join(output_dir, f"{angle['name']}.png")
            bpy.context.scene.render.filepath = output_path
            bpy.ops.render.render(write_still=True)
            print(f"✅ Rendered angle: {angle['name']}")
        
        # Restore camera
        camera.location = original_location
        camera.rotation_euler = original_rotation


class BlenderIntegrationManager:
    """Manages integration between Python AI processing and Blender rendering"""
    
    def __init__(self):
        self.blender_available = True
        self.renderer = BlenderAnamorphicRenderer()
    
    def run_blender_pipeline(self, image_path: str, output_dir: str = "./data/output/blender/"):
        """Run complete Blender rendering pipeline for an input image"""
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Setup scene
            self.renderer.setup_blender_scene()
            
            # Create display elements
            display_box = self.renderer.create_display_box()
            logo = self.renderer.create_floating_logo()
            floating_objects = self.renderer.create_floating_objects()
            
            # Render sequence
            sequence_dir = self.renderer.render_anamorphic_sequence(
                num_frames=9,
                output_dir=output_dir
            )
            
            print(f"✅ Blender pipeline complete: {sequence_dir}")
            return sequence_dir
            
        except Exception as e:
            print(f"❌ Blender pipeline failed: {e}")
            return None
    
    def render_external_blender(self, python_script_path: str):
        """Run Blender externally with a Python script"""
        try:
            cmd = [
                "blender",
                "--background",
                "--python", python_script_path
            ]
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✅ External Blender render complete")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ External Blender render failed: {e}")
            return False
        except FileNotFoundError:
            print("❌ Blender not found in PATH")
            return False

    def run_blender_render(self) -> bool:
        """Test Blender rendering"""
        print("🎬 Running Blender Render...")
        
        try:
            # Check if Blender integration script exists
            blender_script = self.project_root / "blender_ai_integration.py"
            
            if blender_script.exists():
                # Execute Blender command
                result = subprocess.run([
                    "blender", "--background", "--python", str(blender_script)
                ], check=True, capture_output=True, text=True, cwd=self.project_root)
                
                print("✅ Blender render successful")
                return True
            else:
                # Create character frame projection directly
                renderer = BlenderAnamorphicRenderer()
                renderer.create_character_frame_projection()
                print("✅ Character frame projection rendered successfully")
                return True
                
        except Exception as e:
            print(f"❌ Blender render failed: {e}")
            return False


def demonstrate_blender_capital_group_pipeline():
    """
    Demonstrate complete Blender pipeline for Capital Group anamorphic display
    This creates professional 3D anamorphic content suitable for corner displays
    """
    print("🎬 Starting Blender Capital Group Pipeline...")
    
    # Initialize renderer
    renderer = BlenderAnamorphicRenderer()
    
    try:
        # Create professional Capital Group sequence
        output_dir = renderer.create_capital_group_sequence()
        
        print("✅ Capital Group pipeline complete!")
        print(f"📁 Output: {output_dir}")
        print("🎯 Anamorphic content ready for corner display deployment")
        
        return output_dir
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return None


def main():
    """Main function for testing builder module"""
    print("🔧 Builder Module - 3D Rendering and Blender Integration")
    print("=" * 60)
    
    # Create renderer
    renderer = BlenderAnamorphicRenderer()
    
    # Create character frame projection (similar to reference image)
    renderer.create_character_frame_projection()


if __name__ == "__main__":
    main() 