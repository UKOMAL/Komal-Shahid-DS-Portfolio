#!/usr/bin/env python3
"""
Anamorphic Configuration Module
Provides configuration presets and validation for the Ultimate Anamorphic System

Author: Komal Shahid
Course: DSC680 - Applied Data Science
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import json
from pathlib import Path

class DisplayType(Enum):
    """Supported display types"""
    CORNER_LED = "corner_led"
    BILLBOARD = "billboard"
    AQUARIUM = "aquarium"
    SEOUL_WAVE = "seoul_wave"
    CURVED_SCREEN = "curved_screen"

class EffectType(Enum):
    """Available effect types"""
    SHADOW_BOX = "shadow_box"
    SCREEN_POP = "screen_pop"
    SEOUL_CORNER = "seoul_corner"
    FLOATING_OBJECTS = "floating_objects"
    WAVE_MOTION = "wave_motion"
    HOLOGRAPHIC = "holographic"

class QualityLevel(Enum):
    """Quality levels for rendering"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"
    PROFESSIONAL = "professional"

@dataclass
class CameraConfig:
    """Camera configuration parameters"""
    distance: float = 15.0  # meters
    angle: float = 25.0  # degrees
    height: float = 1.8  # meters (eye level)
    fov_horizontal: float = 60.0  # degrees
    fov_vertical: float = 40.0  # degrees

@dataclass
class DisplayConfig:
    """Display-specific configuration"""
    pixel_pitch: float = 2.5  # mm per pixel
    brightness_max: int = 4000  # nits
    contrast_ratio: int = 3000
    screen_curvature: float = 0.0  # degrees
    refresh_rate: int = 60  # Hz
    color_gamut: str = "sRGB"  # sRGB, DCI-P3, Rec.2020

@dataclass
class EffectConfig:
    """Effect-specific parameters"""
    strength: float = 2.5
    depth_exaggeration: float = 3.0
    corner_angle: float = 90.0  # degrees
    wave_frequency: float = 0.02
    wave_amplitude: float = 30.0
    motion_speed: float = 1.0
    chromatic_aberration: bool = True
    anti_aliasing: bool = True

@dataclass
class RenderConfig:
    """Rendering configuration"""
    resolution: Tuple[int, int] = (1920, 1080)
    quality: QualityLevel = QualityLevel.HIGH
    samples: int = 256
    denoising: bool = True
    motion_blur: bool = False
    depth_of_field: bool = False

@dataclass
class AnamorphicConfig:
    """Complete anamorphic system configuration"""
    display_type: DisplayType = DisplayType.SEOUL_WAVE
    effect_type: EffectType = EffectType.SEOUL_CORNER
    camera: CameraConfig = field(default_factory=CameraConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    effect: EffectConfig = field(default_factory=EffectConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    enable_3d_objects: bool = True
    enable_motion: bool = False
    output_format: str = "PNG"
    save_intermediate: bool = True

class ConfigPresets:
    """Predefined configuration presets for different use cases"""
    
    @staticmethod
    def seoul_corner_led() -> AnamorphicConfig:
        """Seoul-style corner LED display preset"""
        config = AnamorphicConfig()
        config.display_type = DisplayType.SEOUL_WAVE
        config.effect_type = EffectType.SEOUL_CORNER
        config.camera.angle = 25.0
        config.camera.distance = 15.0
        config.effect.strength = 3.0
        config.effect.depth_exaggeration = 3.5
        config.effect.corner_angle = 90.0
        config.display.pixel_pitch = 2.5
        config.display.brightness_max = 4000
        config.render.quality = QualityLevel.HIGH
        return config
    
    @staticmethod
    def billboard_advertising() -> AnamorphicConfig:
        """Large billboard advertising preset"""
        config = AnamorphicConfig()
        config.display_type = DisplayType.BILLBOARD
        config.effect_type = EffectType.SCREEN_POP
        config.camera.angle = 15.0
        config.camera.distance = 25.0
        config.effect.strength = 2.0
        config.effect.depth_exaggeration = 2.5
        config.display.pixel_pitch = 5.0
        config.display.brightness_max = 6000
        config.render.resolution = (2560, 1440)
        config.render.quality = QualityLevel.ULTRA
        return config
    
    @staticmethod
    def aquarium_display() -> AnamorphicConfig:
        """Aquarium/wave tank display preset"""
        config = AnamorphicConfig()
        config.display_type = DisplayType.AQUARIUM
        config.effect_type = EffectType.WAVE_MOTION
        config.camera.angle = 35.0
        config.camera.distance = 8.0
        config.effect.strength = 4.0
        config.effect.wave_frequency = 0.03
        config.effect.wave_amplitude = 40.0
        config.enable_motion = True
        config.render.motion_blur = True
        return config
    
    @staticmethod
    def holographic_display() -> AnamorphicConfig:
        """Holographic effect preset"""
        config = AnamorphicConfig()
        config.display_type = DisplayType.CURVED_SCREEN
        config.effect_type = EffectType.HOLOGRAPHIC
        config.camera.angle = 45.0
        config.camera.distance = 12.0
        config.effect.strength = 5.0
        config.effect.chromatic_aberration = True
        config.display.screen_curvature = 15.0
        config.render.quality = QualityLevel.PROFESSIONAL
        config.render.samples = 512
        return config
    
    @staticmethod
    def floating_objects() -> AnamorphicConfig:
        """Floating objects effect preset"""
        config = AnamorphicConfig()
        config.display_type = DisplayType.CORNER_LED
        config.effect_type = EffectType.FLOATING_OBJECTS
        config.camera.angle = 30.0
        config.camera.distance = 10.0
        config.effect.strength = 3.5
        config.enable_3d_objects = True
        config.render.depth_of_field = True
        return config
    
    @staticmethod
    def shadow_box_illusion() -> AnamorphicConfig:
        """Shadow box illusion preset"""
        config = AnamorphicConfig()
        config.display_type = DisplayType.BILLBOARD
        config.effect_type = EffectType.SHADOW_BOX
        config.camera.angle = 20.0
        config.camera.distance = 20.0
        config.effect.strength = 2.8
        config.effect.depth_exaggeration = 4.0
        config.render.quality = QualityLevel.HIGH
        return config

class ConfigValidator:
    """Validates configuration parameters"""
    
    @staticmethod
    def validate_config(config: AnamorphicConfig) -> Tuple[bool, List[str]]:
        """Validate complete configuration"""
        errors = []
        
        # Camera validation
        camera_errors = ConfigValidator._validate_camera(config.camera)
        errors.extend(camera_errors)
        
        # Display validation
        display_errors = ConfigValidator._validate_display(config.display)
        errors.extend(display_errors)
        
        # Effect validation
        effect_errors = ConfigValidator._validate_effect(config.effect)
        errors.extend(effect_errors)
        
        # Render validation
        render_errors = ConfigValidator._validate_render(config.render)
        errors.extend(render_errors)
        
        return len(errors) == 0, errors
    
    @staticmethod
    def _validate_camera(camera: CameraConfig) -> List[str]:
        """Validate camera configuration"""
        errors = []
        
        if not (5.0 <= camera.distance <= 100.0):
            errors.append("Camera distance must be between 5 and 100 meters")
        
        if not (10.0 <= camera.angle <= 80.0):
            errors.append("Camera angle must be between 10 and 80 degrees")
        
        if not (0.5 <= camera.height <= 5.0):
            errors.append("Camera height must be between 0.5 and 5.0 meters")
        
        if not (30.0 <= camera.fov_horizontal <= 120.0):
            errors.append("Horizontal FOV must be between 30 and 120 degrees")
        
        if not (20.0 <= camera.fov_vertical <= 90.0):
            errors.append("Vertical FOV must be between 20 and 90 degrees")
        
        return errors
    
    @staticmethod
    def _validate_display(display: DisplayConfig) -> List[str]:
        """Validate display configuration"""
        errors = []
        
        if not (1.0 <= display.pixel_pitch <= 20.0):
            errors.append("Pixel pitch must be between 1.0 and 20.0 mm")
        
        if not (100 <= display.brightness_max <= 10000):
            errors.append("Max brightness must be between 100 and 10000 nits")
        
        if not (100 <= display.contrast_ratio <= 10000):
            errors.append("Contrast ratio must be between 100 and 10000")
        
        if not (-45.0 <= display.screen_curvature <= 45.0):
            errors.append("Screen curvature must be between -45 and 45 degrees")
        
        if not (24 <= display.refresh_rate <= 240):
            errors.append("Refresh rate must be between 24 and 240 Hz")
        
        return errors
    
    @staticmethod
    def _validate_effect(effect: EffectConfig) -> List[str]:
        """Validate effect configuration"""
        errors = []
        
        if not (0.5 <= effect.strength <= 10.0):
            errors.append("Effect strength must be between 0.5 and 10.0")
        
        if not (1.0 <= effect.depth_exaggeration <= 10.0):
            errors.append("Depth exaggeration must be between 1.0 and 10.0")
        
        if not (45.0 <= effect.corner_angle <= 180.0):
            errors.append("Corner angle must be between 45 and 180 degrees")
        
        if not (0.001 <= effect.wave_frequency <= 0.1):
            errors.append("Wave frequency must be between 0.001 and 0.1")
        
        if not (5.0 <= effect.wave_amplitude <= 100.0):
            errors.append("Wave amplitude must be between 5.0 and 100.0")
        
        if not (0.1 <= effect.motion_speed <= 5.0):
            errors.append("Motion speed must be between 0.1 and 5.0")
        
        return errors
    
    @staticmethod
    def _validate_render(render: RenderConfig) -> List[str]:
        """Validate render configuration"""
        errors = []
        
        width, height = render.resolution
        if not (640 <= width <= 7680 and 480 <= height <= 4320):
            errors.append("Resolution must be between 640x480 and 7680x4320")
        
        if render.quality == QualityLevel.LOW and render.samples > 64:
            errors.append("Low quality should use <= 64 samples")
        elif render.quality == QualityLevel.PROFESSIONAL and render.samples < 256:
            errors.append("Professional quality should use >= 256 samples")
        
        return errors

class ConfigManager:
    """Manages configuration loading, saving, and presets"""
    
    @staticmethod
    def save_config(config: AnamorphicConfig, filepath: Path) -> bool:
        """Save configuration to JSON file"""
        try:
            config_dict = ConfigManager._config_to_dict(config)
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
    
    @staticmethod
    def load_config(filepath: Path) -> Optional[AnamorphicConfig]:
        """Load configuration from JSON file"""
        try:
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
            return ConfigManager._dict_to_config(config_dict)
        except Exception as e:
            print(f"Error loading config: {e}")
            return None
    
    @staticmethod
    def _config_to_dict(config: AnamorphicConfig) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'display_type': config.display_type.value,
            'effect_type': config.effect_type.value,
            'camera': {
                'distance': config.camera.distance,
                'angle': config.camera.angle,
                'height': config.camera.height,
                'fov_horizontal': config.camera.fov_horizontal,
                'fov_vertical': config.camera.fov_vertical
            },
            'display': {
                'pixel_pitch': config.display.pixel_pitch,
                'brightness_max': config.display.brightness_max,
                'contrast_ratio': config.display.contrast_ratio,
                'screen_curvature': config.display.screen_curvature,
                'refresh_rate': config.display.refresh_rate,
                'color_gamut': config.display.color_gamut
            },
            'effect': {
                'strength': config.effect.strength,
                'depth_exaggeration': config.effect.depth_exaggeration,
                'corner_angle': config.effect.corner_angle,
                'wave_frequency': config.effect.wave_frequency,
                'wave_amplitude': config.effect.wave_amplitude,
                'motion_speed': config.effect.motion_speed,
                'chromatic_aberration': config.effect.chromatic_aberration,
                'anti_aliasing': config.effect.anti_aliasing
            },
            'render': {
                'resolution': config.render.resolution,
                'quality': config.render.quality.value,
                'samples': config.render.samples,
                'denoising': config.render.denoising,
                'motion_blur': config.render.motion_blur,
                'depth_of_field': config.render.depth_of_field
            },
            'enable_3d_objects': config.enable_3d_objects,
            'enable_motion': config.enable_motion,
            'output_format': config.output_format,
            'save_intermediate': config.save_intermediate
        }
    
    @staticmethod
    def _dict_to_config(config_dict: Dict[str, Any]) -> AnamorphicConfig:
        """Convert dictionary to configuration"""
        config = AnamorphicConfig()
        
        config.display_type = DisplayType(config_dict['display_type'])
        config.effect_type = EffectType(config_dict['effect_type'])
        
        # Camera config
        camera_dict = config_dict['camera']
        config.camera = CameraConfig(
            distance=camera_dict['distance'],
            angle=camera_dict['angle'],
            height=camera_dict['height'],
            fov_horizontal=camera_dict['fov_horizontal'],
            fov_vertical=camera_dict['fov_vertical']
        )
        
        # Display config
        display_dict = config_dict['display']
        config.display = DisplayConfig(
            pixel_pitch=display_dict['pixel_pitch'],
            brightness_max=display_dict['brightness_max'],
            contrast_ratio=display_dict['contrast_ratio'],
            screen_curvature=display_dict['screen_curvature'],
            refresh_rate=display_dict['refresh_rate'],
            color_gamut=display_dict['color_gamut']
        )
        
        # Effect config
        effect_dict = config_dict['effect']
        config.effect = EffectConfig(
            strength=effect_dict['strength'],
            depth_exaggeration=effect_dict['depth_exaggeration'],
            corner_angle=effect_dict['corner_angle'],
            wave_frequency=effect_dict['wave_frequency'],
            wave_amplitude=effect_dict['wave_amplitude'],
            motion_speed=effect_dict['motion_speed'],
            chromatic_aberration=effect_dict['chromatic_aberration'],
            anti_aliasing=effect_dict['anti_aliasing']
        )
        
        # Render config
        render_dict = config_dict['render']
        config.render = RenderConfig(
            resolution=tuple(render_dict['resolution']),
            quality=QualityLevel(render_dict['quality']),
            samples=render_dict['samples'],
            denoising=render_dict['denoising'],
            motion_blur=render_dict['motion_blur'],
            depth_of_field=render_dict['depth_of_field']
        )
        
        config.enable_3d_objects = config_dict['enable_3d_objects']
        config.enable_motion = config_dict['enable_motion']
        config.output_format = config_dict['output_format']
        config.save_intermediate = config_dict['save_intermediate']
        
        return config
    
    @staticmethod
    def get_preset_names() -> List[str]:
        """Get list of available preset names"""
        return [
            "seoul_corner_led",
            "billboard_advertising", 
            "aquarium_display",
            "holographic_display",
            "floating_objects",
            "shadow_box_illusion"
        ]
    
    @staticmethod
    def get_preset(name: str) -> Optional[AnamorphicConfig]:
        """Get configuration preset by name"""
        presets = {
            "seoul_corner_led": ConfigPresets.seoul_corner_led,
            "billboard_advertising": ConfigPresets.billboard_advertising,
            "aquarium_display": ConfigPresets.aquarium_display,
            "holographic_display": ConfigPresets.holographic_display,
            "floating_objects": ConfigPresets.floating_objects,
            "shadow_box_illusion": ConfigPresets.shadow_box_illusion
        }
        
        if name in presets:
            return presets[name]()
        return None

def main():
    """Demonstrate configuration system"""
    print("🔧 Anamorphic Configuration System")
    print("=" * 50)
    
    # Show available presets
    print("Available presets:")
    for preset_name in ConfigManager.get_preset_names():
        print(f"  - {preset_name}")
    
    # Load Seoul preset
    print("\nLoading Seoul Corner LED preset...")
    config = ConfigManager.get_preset("seoul_corner_led")
    
    # Validate configuration
    is_valid, errors = ConfigValidator.validate_config(config)
    print(f"Configuration valid: {is_valid}")
    if errors:
        print("Errors:")
        for error in errors:
            print(f"  - {error}")
    
    # Save configuration
    config_path = Path("seoul_config.json")
    if ConfigManager.save_config(config, config_path):
        print(f"Configuration saved to {config_path}")
    
    # Load configuration
    loaded_config = ConfigManager.load_config(config_path)
    if loaded_config:
        print("Configuration loaded successfully")
    
    print("\n✅ Configuration system demonstration complete")

if __name__ == "__main__":
    main() 