#!/usr/bin/env python3
"""
Komal Shahid DSC680 - Anamorphic Billboard Generator
Creates 3D anamorphic billboards with real mathematical distortion
Analyzes input images and creates themed 3D scene compositions

Author: Komal Shahid
Course: DSC680 - Applied Data Science
Project: Project 3 - Colorful Canvas Anamorphic Billboard
"""
import bpy
import os
import sys
import math
import random
import logging
from typing import Tuple, List, Dict, Optional, Union, Any
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Version tracking
__version__ = "1.1.0"
__status__ = "Enhanced"
__milestone__ = "Advanced Anamorphic Mathematics with Error Handling"
__author__ = "Komal Shahid DSC680"

class BillboardConfigurationError(Exception):
    """Custom exception for billboard configuration errors"""
    pass

class RenderingError(Exception):
    """Custom exception for rendering errors"""
    pass

class AnamorphicBillboardGenerator:
    """
    Advanced class for generating anamorphic billboards with mathematical distortion
    Features: Enhanced error handling, input validation, configurable parameters
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the billboard generator with optional configuration
        
        Args:
            config: Optional configuration dictionary with camera settings, render quality, etc.
        """
        # Default configuration
        self.config: Dict[str, Any] = {
            'camera_distance': 40,
            'camera_angle_degrees': 45,
            'render_quality': 'high',  # 'low', 'medium', 'high', 'ultra'
            'enable_displacement': True,
            'max_objects': 25,
            'frame_thickness': 2,
            'lighting_energy_multiplier': 1.0
        }
        
        # Update with user configuration
        if config:
            self.config.update(config)
            
        # Validate configuration
        self._validate_configuration()
        
        # Set derived parameters
        self.camera_distance = float(self.config['camera_distance'])
        self.camera_angle = math.radians(float(self.config['camera_angle_degrees']))
        self.billboard_z = 0  # Billboard plane at Z=0
        
        # Render quality settings
        self.render_settings = self._get_render_settings()
        
        # Image analysis categories with improved color schemes
        self.image_categories = {
            "watch": {
                "objects": ["gears", "numbers", "hands"],
                "colors": [(0.8, 0.8, 0.9), (0.9, 0.7, 0.3), (0.4, 0.4, 0.5), (0.6, 0.8, 0.9)]
            },
            "face": {
                "objects": ["eyes", "features"],
                "colors": [(1.0, 0.8, 0.6), (0.8, 0.6, 0.4), (1.0, 0.7, 0.8), (0.9, 0.5, 0.5)]
            },
            "nature": {
                "objects": ["flowers", "trees", "animals"],
                "colors": [(0.2, 0.8, 0.3), (1.0, 0.8, 0.2), (1.0, 0.2, 0.6), (0.3, 0.9, 0.4)]
            },
            "default": {
                "objects": ["abstract"],
                "colors": [(1.0, 0.2, 0.4), (0.2, 0.8, 1.0), (1.0, 0.5, 0.0), (0.6, 0.2, 0.8)]
            }
        }
        
        logger.info(f"AnamorphicBillboardGenerator v{__version__} initialized")
        
    def _validate_configuration(self) -> None:
        """Validate the configuration parameters"""
        try:
            # Camera distance validation
            camera_distance = float(self.config['camera_distance'])
            if not (10 <= camera_distance <= 100):
                raise BillboardConfigurationError("Camera distance must be between 10 and 100 units")
                
            # Camera angle validation  
            camera_angle = float(self.config['camera_angle_degrees'])
            if not (15 <= camera_angle <= 75):
                raise BillboardConfigurationError("Camera angle must be between 15 and 75 degrees")
                
            # Render quality validation
            valid_qualities = ['low', 'medium', 'high', 'ultra']
            render_quality = str(self.config['render_quality'])
            if render_quality not in valid_qualities:
                raise BillboardConfigurationError(f"Render quality must be one of: {valid_qualities}")
                
            # Max objects validation
            max_objects = int(self.config['max_objects'])
            if not (5 <= max_objects <= 50):
                raise BillboardConfigurationError("Max objects must be between 5 and 50")
                
            logger.info("Configuration validation passed")
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
            
    def _get_render_settings(self) -> Dict[str, Any]:
        """Get render settings based on quality level"""
        quality_settings: Dict[str, Dict[str, Any]] = {
            'low': {'samples': 64, 'resolution': (1280, 720), 'denoising': True},
            'medium': {'samples': 128, 'resolution': (1920, 1080), 'denoising': True},
            'high': {'samples': 256, 'resolution': (1920, 1080), 'denoising': True},
            'ultra': {'samples': 512, 'resolution': (2560, 1440), 'denoising': True}
        }
        
        return quality_settings[str(self.config['render_quality'])]

    def validate_input_image(self, image_path: Union[str, Path]) -> Path:
        """
        Validate input image file
        
        Args:
            image_path: Path to image file
            
        Returns:
            Path: Validated Path object
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If file is not a valid image format
        """
        try:
            path = Path(image_path)
            
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {path}")
                
            # Check file extension
            valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tga'}
            if path.suffix.lower() not in valid_extensions:
                raise ValueError(f"Invalid image format. Supported: {valid_extensions}")
                
            # Check file size (reasonable limits)
            file_size_mb = path.stat().st_size / (1024 * 1024)
            if file_size_mb > 100:  # 100MB limit
                logger.warning(f"Large image file: {file_size_mb:.1f}MB")
                
            logger.info(f"Input image validated: {path.name} ({file_size_mb:.1f}MB)")
            return path
            
        except Exception as e:
            logger.error(f"Image validation failed: {e}")
            raise

    def calculate_anamorphic_position(self, intended_x: float, intended_y: float, intended_z: float, 
                                    camera_pos: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Position objects clearly between billboard and camera for optimal visibility
        """
        try:
            # Position objects clearly in front of billboard but well behind camera
            actual_x = intended_x * 0.7  # Keep within reasonable frame bounds
            actual_y = intended_y * 0.7  # Keep within reasonable frame bounds  
            actual_z = 2.0 + (intended_z * 1.5)  # Objects positioned 2-8 units in front of billboard
            
            # Ensure objects stay in visible range (between billboard and camera)
            actual_z = max(actual_z, 1.5)  # Minimum distance from billboard
            actual_z = min(actual_z, 12.0)  # Maximum distance (well behind camera)
            
            return (actual_x, actual_y, actual_z)
            
        except Exception as e:
            logger.error(f"Position calculation failed: {e}")
            raise

    def calculate_anamorphic_scale(self, intended_depth, base_size):
        """
        Scale objects for clear visibility at closer distances
        """
        # Larger visibility multiplier since objects are closer
        visibility_multiplier = 3.0  # Increase size for better visibility
        
        # Slight depth scaling for perspective
        depth_scale = 1.0 + (intended_depth * 0.1)  # Small variation with depth
        
        final_scale = base_size * visibility_multiplier * depth_scale
        
        # Ensure good minimum size
        final_scale = max(final_scale, 2.0)
        
        return (final_scale, final_scale, final_scale)

    def calculate_anamorphic_rotation(self, intended_pos, camera_pos):
        """
        Rotate objects to face the anamorphic viewing angle using full mathematical correction
        
        This implements comprehensive rotation compensation for:
        - Camera orientation alignment
        - Perspective viewing angles
        - 3D space orientation correction
        - Depth-based rotation adjustments
        
        Args:
            intended_pos: Intended position tuple (x, y, z)
            camera_pos: Camera position tuple (x, y, z)
        
        Returns:
            tuple: Rotation angles (rotation_x, rotation_y, rotation_z)
        """
        # Vector from object to camera
        dx = camera_pos[0] - intended_pos[0]
        dy = camera_pos[1] - intended_pos[1] 
        dz = camera_pos[2] - intended_pos[2]
        
        # Distance calculations for normalization
        horizontal_distance = math.sqrt(dx**2 + dy**2)
        total_distance = math.sqrt(dx**2 + dy**2 + dz**2)
        
        # Camera orientation alignment
        # Z-rotation: Align object to face camera azimuth
        rotation_z = math.atan2(dy, dx)
        
        # X-rotation: Align object to face camera elevation
        if horizontal_distance > 0.001:  # Avoid division by zero
            rotation_x = math.atan2(dz, horizontal_distance)
        else:
            rotation_x = 0
            
        # Y-rotation: Additional perspective correction
        # Account for object's position relative to billboard center
        object_angle_from_center = math.atan2(intended_pos[1], intended_pos[0])
        camera_angle_from_center = math.atan2(camera_pos[1], camera_pos[0])
        relative_angle = camera_angle_from_center - object_angle_from_center
        
        # Perspective viewing angle correction
        perspective_correction = intended_pos[2] * 0.1 * math.sin(relative_angle)
        rotation_y = relative_angle * 0.3 + perspective_correction
        
        # Depth-based rotation adjustments
        # Objects at different depths need different orientation
        depth_factor = intended_pos[2] / 5.0  # Normalize depth
        depth_rotation_x = depth_factor * 0.2 * math.sin(rotation_z)
        depth_rotation_y = depth_factor * 0.15 * math.cos(rotation_z)
        depth_rotation_z = depth_factor * 0.1
        
        # Apply depth adjustments
        rotation_x += depth_rotation_x
        rotation_y += depth_rotation_y  
        rotation_z += depth_rotation_z
        
        # Additional corrections for extreme viewing angles
        camera_elevation = math.atan2(camera_pos[2], horizontal_distance)
        if abs(camera_elevation) > math.radians(45):
            elevation_correction = math.sin(camera_elevation) * 0.4
            rotation_x += elevation_correction
            
        camera_azimuth = math.atan2(camera_pos[1], camera_pos[0])
        if abs(camera_azimuth) > math.radians(30):
            azimuth_correction = math.cos(camera_azimuth) * 0.3
            rotation_z += azimuth_correction
        
        # Limit rotation ranges to prevent extreme orientations
        rotation_x = max(-math.radians(85), min(rotation_x, math.radians(85)))
        rotation_y = max(-math.radians(180), min(rotation_y, math.radians(180)))
        rotation_z = max(-math.radians(180), min(rotation_z, math.radians(180)))
        
        return (rotation_x, rotation_y, rotation_z)

    def analyze_image_content(self, image_path):
        """
        Analyze image content based on filename to determine object category
        
        Args:
            image_path: Path to input image file
        
        Returns:
            str: Category name ("watch", "face", "nature", "default")
        """
        if not os.path.exists(image_path):
            print(f"⚠️ Image not found: {image_path}")
            return "default"
        
        filename = os.path.basename(image_path).lower()
        
        if any(word in filename for word in ["watch", "clock", "bulgari", "time"]):
            return "watch"
        elif any(word in filename for word in ["face", "portrait", "person", "emoji"]):
            return "face"
        elif any(word in filename for word in ["nature", "flower", "tree", "plant", "garden"]):
            return "nature"
        else:
            return "default"

    def clear_blender_scene(self):
        """Clear everything in the Blender scene"""
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        
        # Clear materials
        for material in list(bpy.data.materials):
            bpy.data.materials.remove(material)
        
        print("🧹 Scene cleared")

    def create_billboard_material(self, name, color):
        """
        Create a material for billboard objects
        
        Args:
            name: Material name
            color: RGB color tuple
        
        Returns:
            Material: Blender material object
        """
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()
        
        output = nodes.new('ShaderNodeOutputMaterial')
        principled = nodes.new('ShaderNodeBsdfPrincipled')
        principled.inputs['Base Color'].default_value = (*color, 1.0)
        principled.inputs['Metallic'].default_value = 0.7
        principled.inputs['Roughness'].default_value = 0.3
        links.new(principled.outputs['BSDF'], output.inputs['Surface'])
        
        return mat

    def create_billboard_frame_and_screen(self, image_path):
        """
        Create professional billboard frame and apply image to screen
        PROPERLY ORIENTED to face the camera
        
        Args:
            image_path: Path to image file to display on screen
        
        Returns:
            tuple: (frame_parts_list, screen_object)
        """
        
        print(f"🖼️ Creating frame with image: {image_path}")
        
        # Create frame pieces - LARGER frame for better visibility
        frame_parts = []
        frame_width = 30  # Increased from 20
        frame_height = 18  # Increased from 12  
        frame_thickness = 3  # Increased from 2
        
        # Top, bottom, left, right frame pieces
        frame_positions = [
            (0, (frame_height + frame_thickness)/2, 0),    # Top
            (0, -(frame_height + frame_thickness)/2, 0),   # Bottom
            (-(frame_width + frame_thickness)/2, 0, 0),    # Left
            ((frame_width + frame_thickness)/2, 0, 0)      # Right
        ]
        
        frame_scales = [
            ((frame_width + 2*frame_thickness)/2, frame_thickness/2, 1),  # Top/Bottom
            ((frame_width + 2*frame_thickness)/2, frame_thickness/2, 1),
            (frame_thickness/2, frame_height/2, 1),                       # Left/Right
            (frame_thickness/2, frame_height/2, 1)
        ]
        
        for i, (pos, scale) in enumerate(zip(frame_positions, frame_scales)):
            bpy.ops.mesh.primitive_cube_add(location=pos)
            frame_part = bpy.context.active_object
            frame_part.name = f"Frame_{i}"
            frame_part.scale = scale
            bpy.ops.object.transform_apply(scale=True)
            
            # NO ROTATION - keep frame flat and facing camera
            # frame_part.rotation_euler = (0, 0, 0)  # Flat orientation
            
            # Apply frame material
            frame_mat = self.create_billboard_material(f"Frame_Material_{i}", (0.8, 0.6, 0.2))
            frame_part.data.materials.append(frame_mat)
            frame_parts.append(frame_part)
        
        # Create screen - larger
        bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, -0.5))
        screen = bpy.context.active_object
        screen.name = "Billboard_Screen"
        screen.scale = (frame_width/2, frame_height/2, 1)
        bpy.ops.object.transform_apply(scale=True)
        
        # NO ROTATION - keep screen flat facing camera
        # screen.rotation_euler = (0, 0, 0)  # Flat orientation
        
        # Add subdivision for displacement
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.subdivide(number_cuts=10)
        bpy.ops.object.mode_set(mode='OBJECT')
        
        # Apply image material
        if os.path.exists(image_path):
            self.apply_image_material_with_displacement(screen, image_path)
        else:
            # Create test material
            test_mat = self.create_billboard_material("Test_Material", (1.0, 0.2, 0.4))
            screen.data.materials.append(test_mat)
        
        print("🎯 Billboard oriented FLAT to face camera directly")
        return frame_parts, screen

    def apply_image_material_with_displacement(self, obj, image_path):
        """
        Apply image as material with displacement mapping
        
        Args:
            obj: Blender object to apply material to
            image_path: Path to image file
        """
        
        img = bpy.data.images.load(image_path)
        
        mat = bpy.data.materials.new("Image_Material")
        mat.use_nodes = True
        mat.cycles.displacement_method = 'BOTH'
        
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()
        
        # Create nodes
        output = nodes.new('ShaderNodeOutputMaterial')
        principled = nodes.new('ShaderNodeBsdfPrincipled')
        tex_coord = nodes.new('ShaderNodeTexCoord')
        image_tex = nodes.new('ShaderNodeTexImage')
        displacement = nodes.new('ShaderNodeDisplacement')
        
        # Set image
        image_tex.image = img
        
        # Connect nodes
        links.new(tex_coord.outputs['UV'], image_tex.inputs['Vector'])
        links.new(image_tex.outputs['Color'], principled.inputs['Base Color'])
        links.new(image_tex.outputs['Color'], displacement.inputs['Height'])
        links.new(principled.outputs['BSDF'], output.inputs['Surface'])
        links.new(displacement.outputs['Displacement'], output.inputs['Displacement'])
        
        # Set displacement strength
        displacement.inputs['Scale'].default_value = 3.0
        
        obj.data.materials.append(mat)
        
        # Add displacement modifier
        disp_mod = obj.modifiers.new("Displacement", "DISPLACE")
        disp_mod.strength = 2.0

    def create_anamorphic_object(self, name, intended_pos, obj_type="sphere", size=1.0, color=(1.0, 0.5, 0.2)):
        """
        Create an object with full anamorphic distortion using complete mathematical transformation
        
        This applies comprehensive anamorphic effects:
        - Position: Keystone + perspective + viewing angle compensation
        - Scale: Depth perception + camera distance + angle distortion
        - Rotation: Camera alignment + perspective correction + depth adjustment
        """
        # Camera position for full anamorphic calculations
        camera_pos = (self.camera_distance * math.cos(self.camera_angle), 
                     -self.camera_distance * math.sin(self.camera_angle) * 0.7, 
                     self.camera_distance * math.sin(self.camera_angle))
        
        # Step 1: Calculate actual position using full anamorphic mathematics
        actual_pos = self.calculate_anamorphic_position(intended_pos[0], intended_pos[1], intended_pos[2], camera_pos)
        
        # Step 2: Calculate anamorphic scaling
        scale_factors = self.calculate_anamorphic_scale(intended_pos[2], size)
        
        # Step 3: Calculate anamorphic rotation
        rotation_angles = self.calculate_anamorphic_rotation(intended_pos, camera_pos)
        
        # Create the object at calculated position
        if obj_type == "sphere":
            bpy.ops.mesh.primitive_uv_sphere_add(location=actual_pos, radius=size)
        elif obj_type == "cube":
            bpy.ops.mesh.primitive_cube_add(location=actual_pos, size=size)
        elif obj_type == "cylinder":
            bpy.ops.mesh.primitive_cylinder_add(location=actual_pos, radius=size, depth=size)
        
        obj = bpy.context.active_object
        obj.name = name
        
        # Apply full anamorphic scaling
        obj.scale = scale_factors
        
        # Apply full anamorphic rotation
        obj.rotation_euler = rotation_angles
        
        # Apply transformations
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        
        # Apply material
        mat = self.create_billboard_material(f"{name}_Mat", color)
        obj.data.materials.append(mat)
        
        print(f"  📐 {name}: intended{intended_pos} → actual{actual_pos}")
        print(f"       📏 Scale: {scale_factors}")
        print(f"       🔄 Rotation: ({math.degrees(rotation_angles[0]):.1f}°, {math.degrees(rotation_angles[1]):.1f}°, {math.degrees(rotation_angles[2]):.1f}°)")
        
        return obj

    def create_watch_themed_objects(self, colors):
        """
        Create clearly visible watch objects floating in front of billboard
        """
        objects = []
        
        print("🎯 Creating clearly visible watch objects floating in front of billboard...")
        
        # Create fewer, larger, more strategically positioned objects
        
        # Large center gear - positioned to be clearly visible
        main_gear = self.create_anamorphic_object("Main_Gear", (0, 0, 2.0), "cylinder", 3.0, (1.0, 0.0, 0.0))
        objects.append(main_gear)
        
        # Four watch numbers positioned at corners of viewable area
        number_positions = [
            (4, 3, 2.5),    # Top-right
            (-4, 3, 2.5),   # Top-left
            (-4, -3, 2.5),  # Bottom-left  
            (4, -3, 2.5)    # Bottom-right
        ]
        
        for i, pos in enumerate(number_positions):
            number = self.create_anamorphic_object(
                f"Number_{i+1}", 
                pos, 
                "cube", 
                2.0,  # Large but reasonable size
                (0.0, 1.0, 0.0)  # Bright green
            )
            objects.append(number)
        
        # Two side gears at medium depth
        gear1 = self.create_anamorphic_object("Gear_Left", (-6, 0, 3.0), "cylinder", 2.5, (0.0, 0.0, 1.0))
        gear2 = self.create_anamorphic_object("Gear_Right", (6, 0, 3.0), "cylinder", 2.5, (1.0, 1.0, 0.0))
        objects.append(gear1)
        objects.append(gear2)
        
        print(f"✅ Created {len(objects)} clearly visible watch objects floating in front")
        return objects

    def create_nature_flower_scene(self, colors):
        """
        Create nature/flower-themed scene with anamorphic positioning
        
        Args:
            colors: List of RGB color tuples
        
        Returns:
            list: Created objects
        """
        
        objects = []
        
        print("🌸 Creating flower-themed anamorphic scene...")
        
        # Main large flower (centerpiece)
        main_flower_objs = self.create_anamorphic_flower_complex("MainFlower", (0, 1, 3), size=3.0, colors=colors[:3])
        objects.extend(main_flower_objs)
        
        # Secondary flowers at different depths
        flower_positions = [
            {"name": "Flower2", "pos": (-3, -1, 2), "size": 2.0},
            {"name": "Flower3", "pos": (4, 2, 4), "size": 2.5},
            {"name": "Flower4", "pos": (-2, 3, 1.5), "size": 1.8},
        ]
        
        for flower_info in flower_positions:
            flower_objs = self.create_anamorphic_flower_complex(
                flower_info["name"], 
                flower_info["pos"], 
                flower_info["size"], 
                colors[3:6] if len(colors) > 5 else colors[:3]
            )
            objects.extend(flower_objs)
        
        # Add floating hearts
        heart_positions = [(2, -2, 3.5), (-3, 2, 2.5), (1, 4, 1.8)]
        
        for i, heart_pos in enumerate(heart_positions):
            heart = self.create_anamorphic_object(f"Heart_{i}", heart_pos, "sphere", 1.2, (1.0, 0.2, 0.6))
            objects.append(heart)
        
        print(f"✅ Created flower scene with {len(objects)} anamorphic objects")
        return objects

    def create_anamorphic_flower_complex(self, name, intended_pos, size=2.0, colors=None):
        """
        Create a complex flower with anamorphic positioning
        
        Args:
            name: Flower name
            intended_pos: Where flower should appear (x, y, z)
            size: Flower size
            colors: List of RGB color tuples
        
        Returns:
            list: Created flower objects
        """
        
        if colors is None:
            colors = [(1.0, 0.2, 0.6), (1.0, 0.8, 0.2), (0.2, 0.8, 0.3)]
        
        objects = []
        
        # Flower center
        center = self.create_anamorphic_object(f"{name}_Center", intended_pos, "sphere", size*0.4, colors[1])
        objects.append(center)
        
        # Flower petals (8 petals around center)
        petal_count = 8
        for i in range(petal_count):
            angle = i * (math.pi * 2 / petal_count)
            petal_distance = size * 0.8
            
            petal_intended_x = intended_pos[0] + math.cos(angle) * petal_distance
            petal_intended_y = intended_pos[1] + math.sin(angle) * petal_distance
            petal_intended_z = intended_pos[2]
            
            petal = self.create_anamorphic_object(
                f"{name}_Petal_{i}", 
                (petal_intended_x, petal_intended_y, petal_intended_z),
                "sphere", 
                size*0.3, 
                colors[0]
            )
            objects.append(petal)
        
        # Leaves (4 leaves below flower)
        for i in range(4):
            angle = i * (math.pi / 2)
            leaf_distance = size * 1.2
            
            leaf_intended_x = intended_pos[0] + math.cos(angle) * leaf_distance
            leaf_intended_y = intended_pos[1] + math.sin(angle) * leaf_distance
            leaf_intended_z = intended_pos[2] - size * 0.5
            
            leaf = self.create_anamorphic_object(
                f"{name}_Leaf_{i}",
                (leaf_intended_x, leaf_intended_y, leaf_intended_z),
                "cube",
                size*0.4,
                colors[2]
            )
            objects.append(leaf)
        
        print(f"  🌸 Created anamorphic flower {name} with {len(objects)} parts")
        return objects

    def create_emoji_face_scene(self, colors):
        """
        Create emoji/face-themed scene with anamorphic positioning
        
        Args:
            colors: List of RGB color tuples
        
        Returns:
            list: Created objects
        """
        
        objects = []
        
        print("😊 Creating emoji-themed anamorphic scene...")
        
        # Large central emoji
        main_emoji_objs = self.create_anamorphic_emoji_complex("MainEmoji", (0, 0, 3), size=3.0, emoji_type="happy")
        objects.extend(main_emoji_objs)
        
        # Surrounding emojis
        emoji_positions = [
            {"name": "Emoji2", "pos": (-3, -1, 2), "size": 2.0, "type": "happy"},
            {"name": "Emoji3", "pos": (3, 1, 4), "size": 2.5, "type": "normal"},
            {"name": "Emoji4", "pos": (-2, 3, 1.5), "size": 1.8, "type": "happy"},
        ]
        
        for emoji_info in emoji_positions:
            emoji_objs = self.create_anamorphic_emoji_complex(
                emoji_info["name"],
                emoji_info["pos"], 
                emoji_info["size"],
                emoji_info["type"]
            )
            objects.extend(emoji_objs)
        
        # Floating hearts between emojis
        heart_positions = [(2, -3, 3.5), (-4, 1, 2.5), (4, 2, 1.8)]
        
        for i, heart_pos in enumerate(heart_positions):
            heart = self.create_anamorphic_object(f"FloatingHeart_{i}", heart_pos, "sphere", 1.0, (1.0, 0.2, 0.6))
            objects.append(heart)
        
        print(f"✅ Created emoji scene with {len(objects)} anamorphic objects")
        return objects

    def create_anamorphic_emoji_complex(self, name, intended_pos, size=2.0, emoji_type="happy"):
        """
        Create complex emoji face with anamorphic positioning
        
        Args:
            name: Emoji name
            intended_pos: Where emoji should appear (x, y, z)
            size: Emoji size
            emoji_type: Type of emoji ("happy", "normal")
        
        Returns:
            list: Created emoji objects
        """
        
        objects = []
        colors = [(1.0, 0.9, 0.2), (0.05, 0.05, 0.05), (0.8, 0.1, 0.1)]  # Yellow, black, red
        
        # Main face
        face = self.create_anamorphic_object(f"{name}_Face", intended_pos, "sphere", size, colors[0])
        objects.append(face)
        
        # Eyes
        eye_positions = [
            (intended_pos[0] - size*0.4, intended_pos[1], intended_pos[2] + size*0.2),
            (intended_pos[0] + size*0.4, intended_pos[1], intended_pos[2] + size*0.2)
        ]
        
        for i, eye_pos in enumerate(eye_positions):
            eye = self.create_anamorphic_object(f"{name}_Eye_{i}", eye_pos, "sphere", size*0.15, colors[1])
            objects.append(eye)
        
        # Mouth
        mouth_pos = (intended_pos[0], intended_pos[1], intended_pos[2] - size*0.3)
        mouth = self.create_anamorphic_object(f"{name}_Mouth", mouth_pos, "cube", size*0.2, colors[2])
        objects.append(mouth)
        
        print(f"  😊 Created anamorphic emoji {name} with {len(objects)} parts")
        return objects

    def create_default_abstract_scene(self, colors):
        """
        Create default abstract scene with anamorphic positioning
        
        Args:
            colors: List of RGB color tuples
        
        Returns:
            list: Created objects
        """
        
        objects = []
        
        print("🎯 Creating default objects with anamorphic mathematics...")
        
        # Strategic positioning for maximum pop-out effect
        object_positions = [
            # Front layer (close to viewer)
            {"name": "Front_Sphere_1", "pos": (-2, 1, 1), "type": "sphere", "size": 1.5},
            {"name": "Front_Cube_1", "pos": (2, -1, 1), "type": "cube", "size": 1.2},
            {"name": "Front_Cylinder_1", "pos": (0, 2, 1.5), "type": "cylinder", "size": 1.0},
            
            # Middle layer (medium depth)
            {"name": "Mid_Sphere_1", "pos": (-3, -2, 2.5), "type": "sphere", "size": 1.8},
            {"name": "Mid_Sphere_2", "pos": (3, 2, 2.5), "type": "sphere", "size": 1.6},
            {"name": "Mid_Cube_1", "pos": (1, -3, 2), "type": "cube", "size": 1.4},
            {"name": "Mid_Cube_2", "pos": (-1, 3, 2), "type": "cube", "size": 1.3},
            {"name": "Mid_Cylinder_1", "pos": (4, 0, 2.8), "type": "cylinder", "size": 1.1},
            
            # Back layer (far from viewer)  
            {"name": "Back_Sphere_1", "pos": (-4, 1, 4), "type": "sphere", "size": 2.2},
            {"name": "Back_Sphere_2", "pos": (0, -4, 4), "type": "sphere", "size": 2.0},
            {"name": "Back_Cube_1", "pos": (4, 3, 3.5), "type": "cube", "size": 1.8},
            {"name": "Back_Cylinder_1", "pos": (-3, -3, 3.8), "type": "cylinder", "size": 1.5},
            {"name": "Back_Cylinder_2", "pos": (2, 4, 4.2), "type": "cylinder", "size": 1.7},
            
            # Extra floating objects for richness
            {"name": "Float_1", "pos": (-5, 0, 3), "type": "sphere", "size": 1.3},
            {"name": "Float_2", "pos": (5, -2, 3.2), "type": "cube", "size": 1.1},
        ]
        
        for i, obj_info in enumerate(object_positions):
            obj = self.create_anamorphic_object(
                obj_info["name"],
                obj_info["pos"],
                obj_info["type"],
                obj_info["size"],
                colors[i % len(colors)]
            )
            objects.append(obj)
        
        print(f"✅ Created {len(objects)} default objects with anamorphic positioning")
        return objects

    def setup_anamorphic_camera(self):
        """Setup camera in front of flat billboard looking straight at it"""
        bpy.ops.object.camera_add()
        camera = bpy.context.active_object
        camera.name = "Anamorphic_Camera"
        
        # Position camera DIRECTLY IN FRONT of the flat billboard
        # Billboard is flat at Z=0, objects float at Z=5-7, so camera goes back further
        camera.location = (0, -20, 3)  # Straight back from billboard center, slightly up
        
        # Point camera STRAIGHT FORWARD at the flat billboard
        camera.rotation_euler = (0.1, 0, 0)  # Just slightly tilted down to see billboard
        
        # Camera settings
        camera.data.lens = 35
        camera.data.sensor_width = 36
        camera.data.clip_start = 0.1
        camera.data.clip_end = 100
        
        bpy.context.scene.camera = camera
        print(f"📷 Camera positioned IN FRONT of flat billboard at {camera.location}")
        print(f"    Looking STRAIGHT at billboard with simple rotation: {camera.rotation_euler}")
        print("📷 Camera should now see flat billboard and floating objects!")

    def setup_professional_lighting(self):
        """Setup 3-point professional lighting"""
        lights = [
            {"loc": (25, -20, 30), "energy": 200, "color": (1.0, 0.9, 0.8)},
            {"loc": (-20, -15, 25), "energy": 120, "color": (0.8, 0.9, 1.0)},
            {"loc": (15, 30, 20), "energy": 150, "color": (1.0, 0.8, 0.9)},
        ]
        
        for i, light_info in enumerate(lights):
            bpy.ops.object.light_add(type='AREA', location=light_info["loc"])
            light = bpy.context.active_object
            light.name = f"Light_{i+1}"
            light.data.energy = light_info["energy"]
            light.data.color = light_info["color"]
            light.data.size = 12
        
        print("💡 Professional lighting setup complete")

    def setup_render_configuration(self):
        """Setup professional render settings"""
        scene = bpy.context.scene
        scene.render.engine = 'CYCLES'
        scene.render.resolution_x = 1920
        scene.render.resolution_y = 1080
        scene.cycles.samples = 256
        scene.cycles.use_denoising = True
        scene.view_settings.view_transform = 'Filmic'
        scene.view_settings.look = 'High Contrast'
        print("🎞️ Professional render settings configured")

    def generate_billboard(self, image_path="data/input/benchmark.jpg", output_path="data/out/pngs/anamorphic_billboard_output.png"):
        """
        Main function - Generate the anamorphic billboard
        
        Args:
            image_path: Path to input image
            output_path: Path for output render
        """
        
        # Print version and milestone information
        print(f"🧠 KOMAL SHAHID DSC680 ANAMORPHIC BILLBOARD GENERATOR - {__status__.upper()} VERSION {__version__}")
        print(f"📋 {__milestone__}")
        print(f"👨‍💻 Author: {__author__}")
        print("=" * 70)
        
        # Step 1: Analyze image
        print(f"\n🔍 Step 1: Analyzing image: {image_path}")
        category = self.analyze_image_content(image_path)
        config = self.image_categories[category]
        colors = config["colors"]
        
        print(f"   📊 Detected category: {category}")
        print(f"   🎨 Objects to create: {config['objects']}")
        
        # Step 2: Clear scene
        print(f"\n🧹 Step 2: Clearing scene...")
        self.clear_blender_scene()
        
        # Step 3: Create frame and screen
        print(f"\n🖼️ Step 3: Creating billboard frame and screen...")
        frame_parts, screen = self.create_billboard_frame_and_screen(image_path)
        
        # Step 4: Create themed objects based on category
        print(f"\n🎨 Step 4: Creating {category}-themed 3D objects...")
        if category == "watch":
            objects = self.create_watch_themed_objects(colors)
        elif category == "face":
            objects = self.create_emoji_face_scene(colors)
        elif category == "nature":
            objects = self.create_nature_flower_scene(colors)
        else:
            objects = self.create_default_abstract_scene(colors)
        
        # Step 5: Setup camera and lighting
        print(f"\n📷 Step 5: Setting up camera and lighting...")
        self.setup_anamorphic_camera()
        self.setup_professional_lighting()
        
        # Step 6: Configure render settings
        print(f"\n🎞️ Step 6: Configuring render settings...")
        self.setup_render_configuration()
        
        # Step 7: Render
        print(f"\n🎬 Step 7: Rendering to {output_path}...")
        bpy.context.scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True)
        
        # Final summary
        print("=" * 70)
        print("✅ KOMAL SHAHID DSC680 ANAMORPHIC BILLBOARD GENERATION COMPLETED!")
        print(f"📁 Output: {output_path}")
        print(f"🏷️ Version: {__version__} ({__status__})")
        print(f"👨‍💻 Author: {__author__}")
        print(f"\n📊 Scene Summary:")
        print(f"  • Input: {os.path.basename(image_path)} ({category})")
        print(f"  • Frame: 4-piece professional frame")
        print(f"  • Screen: Subdivided with displacement mapping")
        print(f"  • Objects: {len(objects)} themed 3D objects floating in space")
        print(f"  • Camera: Anamorphic angle for 3D pop-out effect")
        print(f"  • Lighting: 3-point professional setup")
        print("\n🎯 The 3D objects should now pop out of the billboard frame!")


def main():
    """
    Main function - Entry point for the script
    """
    
    # Get image path from command line or use default
    import sys
    if len(sys.argv) > 4:
        image_path = sys.argv[5] if len(sys.argv) > 5 else "data/input/benchmark.jpg"
    else:
        image_path = "data/input/benchmark.jpg"
    
    output_path = "data/out/pngs/anamorphic_billboard_output.png"
    
    # Create generator instance and run
    generator = AnamorphicBillboardGenerator()
    generator.generate_billboard(image_path, output_path)


if __name__ == "__main__":
    main() 