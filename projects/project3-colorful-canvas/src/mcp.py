#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MCP Module - Model Context Protocol and Command Line Interface
Consolidates all MCP wrapper and CLI functionality for the Colorful Canvas project.

Author: Komal Shahid
Course: DSC680 - Bellevue University
Project: Colorful Canvas AI Art Studio
"""

import sys
import subprocess
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union


class MCPWrapper:
    """
    Model Context Protocol wrapper for Project 3: Colorful Canvas
    Provides command-line interface for AI depth analysis, shadow effects, and Blender rendering
    """
    
    def __init__(self, config_path: str = "src/mcp_config.json"):
        """Initialize MCP wrapper with configuration"""
        self.config_path = config_path
        self.config = self._load_config()
        self.project_root = Path(__file__).parent.parent
        self.output_dir = self.project_root / "data" / "output"
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load MCP configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            print(f"✅ MCP configuration loaded: {self.config_path}")
            return config
        except FileNotFoundError:
            print(f"⚠️ Config file not found: {self.config_path}, using defaults")
            return self._get_default_config()
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in config file: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default MCP configuration"""
        return {
            "ai_module": "src.milestone3.colorful_canvas_complete",
            "ai_class": "ColorfulCanvasAI",
            "output_paths": {
                "depth": "./data/output/mcp_depth.png",
                "shadow": "./data/output/mcp_shadow.png",
                "blender": "./data/output/blender_renders/"
            },
            "test_image_params": {
                "width": 400,
                "height": 300
            },
            "effects": {
                "shadow_box_strength": 1.5,
                "depth_strength": 1.0,
                "chromatic_enabled": True
            }
        }
    
    def run_depth_analysis(self) -> bool:
        """Test AI depth map generation"""
        print("🧠 Running AI Depth Analysis...")
        
        try:
            # Get parameters from config
            ai_module = self.config["ai_module"]
            ai_class = self.config["ai_class"]
            output_path = self.config["output_paths"]["depth"]
            width = self.config["test_image_params"]["width"]
            height = self.config["test_image_params"]["height"]
            
            # Create Python command
            python_cmd = (
                f"from {ai_module} import {ai_class}; "
                f"ai = {ai_class}(); "
                f"img = ai.create_seoul_optimized_test_image({width}, {height}); "
                f"depth = ai.generate_depth_map(img); "
                f"ai.save_image(depth, '{output_path}'); "
                f"print('✅ Depth analysis complete: {output_path}')"
            )
            
            # Execute command
            result = subprocess.run([
                "python3", "-c", python_cmd
            ], check=True, capture_output=True, text=True, cwd=self.project_root)
            
            print(result.stdout)
            print(f"✅ Depth analysis successful: {output_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Depth analysis failed: {e}")
            if e.stderr:
                print(f"Error details: {e.stderr}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error in depth analysis: {e}")
            return False
    
    def run_shadow_effect(self) -> bool:
        """Test working shadow box anamorphic effect"""
        print("📦 Running Shadow Box Effect...")
        
        try:
            # Get parameters from config
            ai_module = self.config["ai_module"]
            ai_class = self.config["ai_class"]
            output_path = self.config["output_paths"]["shadow"]
            width = self.config["test_image_params"]["width"]
            height = self.config["test_image_params"]["height"]
            strength = self.config["effects"]["shadow_box_strength"]
            
            # Create Python command
            python_cmd = (
                f"from {ai_module} import {ai_class}; "
                f"ai = {ai_class}(); "
                f"img = ai.create_seoul_optimized_test_image({width}, {height}); "
                f"depth = ai.generate_depth_map(img); "
                f"result = ai.create_shadow_box_effect(img, depth, strength={strength}); "
                f"ai.save_image(result, '{output_path}'); "
                f"print('✅ Shadow box effect complete: {output_path}')"
            )
            
            # Execute command
            result = subprocess.run([
                "python3", "-c", python_cmd
            ], check=True, capture_output=True, text=True, cwd=self.project_root)
            
            print(result.stdout)
            print(f"✅ Shadow box effect successful: {output_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Shadow box failed: {e}")
            if e.stderr:
                print(f"Error details: {e.stderr}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error in shadow box: {e}")
            return False
    
    def run_blender_render(self) -> bool:
        """Test Blender rendering"""
        print("🎬 Running Blender Render...")
        
        try:
            # Check if Blender integration script exists
            blender_script = self.project_root / "blender_ai_integration.py"
            if not blender_script.exists():
                print(f"❌ Blender script not found: {blender_script}")
                return False
            
            # Execute Blender command
            result = subprocess.run([
                "blender", "--background", "--python", str(blender_script)
            ], check=True, capture_output=True, text=True, cwd=self.project_root)
            
            print("✅ Blender render successful")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Blender render failed: {e}")
            return False
        except FileNotFoundError:
            print("❌ Blender not found in PATH. Please install Blender to use this feature.")
            return False
        except Exception as e:
            print(f"❌ Unexpected error in Blender render: {e}")
            return False
    
    def run_complete_pipeline(self) -> bool:
        """Run complete AI + Blender pipeline"""
        print("🚀 Running Complete Pipeline...")
        
        success_count = 0
        total_steps = 3
        
        # Step 1: AI Depth Analysis
        print("\n--- Step 1: AI Depth Analysis ---")
        if self.run_depth_analysis():
            success_count += 1
        
        # Step 2: Shadow Box Effect
        print("\n--- Step 2: Shadow Box Effect ---")
        if self.run_shadow_effect():
            success_count += 1
        
        # Step 3: Blender Rendering
        print("\n--- Step 3: Blender Rendering ---")
        if self.run_blender_render():
            success_count += 1
        
        # Summary
        print(f"\n🎯 Pipeline Summary: {success_count}/{total_steps} steps successful")
        
        if success_count == total_steps:
            print("✅ Complete pipeline finished successfully")
            return True
        else:
            print("⚠️ Pipeline completed with some failures")
            return False
    
    def run_custom_effect(self, effect_type: str, **kwargs) -> bool:
        """Run custom anamorphic effect"""
        print(f"🎨 Running Custom Effect: {effect_type}")
        
        try:
            ai_module = self.config["ai_module"]
            ai_class = self.config["ai_class"]
            
            # Build custom command based on effect type
            if effect_type == "seoul_corner":
                python_cmd = self._build_seoul_corner_command(ai_module, ai_class, **kwargs)
            elif effect_type == "product_showcase":
                python_cmd = self._build_product_showcase_command(ai_module, ai_class, **kwargs)
            elif effect_type == "art_gallery":
                python_cmd = self._build_art_gallery_command(ai_module, ai_class, **kwargs)
            else:
                print(f"❌ Unknown effect type: {effect_type}")
                return False
            
            # Execute command
            result = subprocess.run([
                "python3", "-c", python_cmd
            ], check=True, capture_output=True, text=True, cwd=self.project_root)
            
            print(result.stdout)
            print(f"✅ Custom effect '{effect_type}' completed successfully")
            return True
            
        except Exception as e:
            print(f"❌ Custom effect '{effect_type}' failed: {e}")
            return False
    
    def _build_seoul_corner_command(self, ai_module: str, ai_class: str, **kwargs) -> str:
        """Build Seoul corner projection command"""
        corner_position = kwargs.get('corner_position', 'left')
        output_path = kwargs.get('output_path', './data/output/seoul_corner.png')
        
        return (
            f"from {ai_module} import {ai_class}; "
            f"ai = {ai_class}(); "
            f"img = ai.create_seoul_optimized_test_image(800, 600); "
            f"result = ai.create_seoul_corner_projection(img, corner_position='{corner_position}'); "
            f"ai.save_image(result, '{output_path}'); "
            f"print('✅ Seoul corner projection complete: {output_path}')"
        )
    
    def _build_product_showcase_command(self, ai_module: str, ai_class: str, **kwargs) -> str:
        """Build product showcase command"""
        showcase_type = kwargs.get('showcase_type', 'floating')
        strength = kwargs.get('strength', 2.0)
        output_path = kwargs.get('output_path', './data/output/product_showcase.png')
        
        return (
            f"from {ai_module} import {ai_class}; "
            f"ai = {ai_class}(); "
            f"img = ai.create_seoul_optimized_test_image(800, 600); "
            f"depth = ai.generate_depth_map(img); "
            f"result = ai.create_product_showcase_template(img, depth, strength={strength}, showcase_type='{showcase_type}'); "
            f"ai.save_image(result, '{output_path}'); "
            f"print('✅ Product showcase complete: {output_path}')"
        )
    
    def _build_art_gallery_command(self, ai_module: str, ai_class: str, **kwargs) -> str:
        """Build art gallery command"""
        frame_style = kwargs.get('frame_style', 'modern')
        strength = kwargs.get('strength', 2.0)
        output_path = kwargs.get('output_path', './data/output/art_gallery.png')
        
        return (
            f"from {ai_module} import {ai_class}; "
            f"ai = {ai_class}(); "
            f"img = ai.create_seoul_optimized_test_image(800, 600); "
            f"depth = ai.generate_depth_map(img); "
            f"result = ai.create_art_gallery_template(img, depth, strength={strength}, frame_style='{frame_style}'); "
            f"ai.save_image(result, '{output_path}'); "
            f"print('✅ Art gallery template complete: {output_path}')"
        )
    
    def show_help(self):
        """Show available commands and usage information"""
        print("🥷 MCP Wrapper for Project 3: Colorful Canvas")
        print("=" * 50)
        print("Available commands:")
        print("  depth      - Test AI depth map generation")
        print("  shadow     - Test shadow box anamorphic effect")
        print("  blender    - Test Blender rendering")
        print("  pipeline   - Run complete AI + Blender pipeline")
        print("  seoul      - Create Seoul corner projection")
        print("  showcase   - Create product showcase template")
        print("  gallery    - Create art gallery template")
        print("  character  - Create 3D character frame projection")
        print("  help       - Show this help")
        print("\nUsage: python3 src/mcp.py <command> [options]")
        print("Note: Run from project root directory")
        print("\nConfiguration file: src/mcp_config.json")
        print(f"Output directory: {self.output_dir}")
    
    def show_status(self):
        """Show current MCP status and configuration"""
        print("📊 MCP Status Report")
        print("=" * 30)
        print(f"Project Root: {self.project_root}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Config File: {self.config_path}")
        print(f"Config Loaded: {'✅' if self.config else '❌'}")
        
        # Check output directory
        if self.output_dir.exists():
            output_files = list(self.output_dir.glob("*"))
            print(f"Output Files: {len(output_files)} found")
        else:
            print("Output Files: Directory not found")
        
        # Check dependencies
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check status of required dependencies"""
        print("\n🔍 Dependency Check:")
        
        # Check Python modules
        modules_to_check = [
            "numpy", "PIL", "torch", "transformers", "cv2", "sklearn"
        ]
        
        for module in modules_to_check:
            try:
                __import__(module)
                print(f"  {module}: ✅")
            except ImportError:
                print(f"  {module}: ❌")
        
        # Check external tools
        external_tools = ["blender", "python3"]
        
        for tool in external_tools:
            try:
                result = subprocess.run([tool, "--version"], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    print(f"  {tool}: ✅")
                else:
                    print(f"  {tool}: ❌")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                print(f"  {tool}: ❌")


class MCPCommandLineInterface:
    """Command-line interface for MCP wrapper"""
    
    def __init__(self):
        self.wrapper = MCPWrapper()
    
    def parse_and_execute(self, args: List[str]) -> int:
        """Parse command line arguments and execute appropriate command"""
        if len(args) < 2:
            self.wrapper.show_help()
            return 1
        
        command = args[1].lower()
        
        # Route commands to appropriate methods
        if command in ["help", "-h", "--help"]:
            self.wrapper.show_help()
            return 0
        elif command == "status":
            self.wrapper.show_status()
            return 0
        elif command == "depth":
            success = self.wrapper.run_depth_analysis()
        elif command == "shadow":
            success = self.wrapper.run_shadow_effect()
        elif command == "blender":
            success = self.wrapper.run_blender_render()
        elif command == "pipeline":
            success = self.wrapper.run_complete_pipeline()
        elif command == "seoul":
            success = self._handle_seoul_command(args)
        elif command == "showcase":
            success = self._handle_showcase_command(args)
        elif command == "gallery":
            success = self._handle_gallery_command(args)
        elif command == "character":
            success = self._handle_character_command(args)
        else:
            print(f"❌ Unknown command: {command}")
            self.wrapper.show_help()
            return 1
        
        return 0 if success else 1
    
    def _handle_seoul_command(self, args: List[str]) -> bool:
        """Handle Seoul corner projection command"""
        kwargs = {}
        
        # Parse additional arguments
        for i, arg in enumerate(args[2:], 2):
            if arg.startswith("--corner="):
                kwargs['corner_position'] = arg.split("=")[1]
            elif arg.startswith("--output="):
                kwargs['output_path'] = arg.split("=")[1]
        
        return self.wrapper.run_custom_effect("seoul_corner", **kwargs)
    
    def _handle_showcase_command(self, args: List[str]) -> bool:
        """Handle product showcase command"""
        kwargs = {}
        
        # Parse additional arguments
        for i, arg in enumerate(args[2:], 2):
            if arg.startswith("--type="):
                kwargs['showcase_type'] = arg.split("=")[1]
            elif arg.startswith("--strength="):
                kwargs['strength'] = float(arg.split("=")[1])
            elif arg.startswith("--output="):
                kwargs['output_path'] = arg.split("=")[1]
        
        return self.wrapper.run_custom_effect("product_showcase", **kwargs)
    
    def _handle_gallery_command(self, args: List[str]) -> bool:
        """Handle art gallery command"""
        kwargs = {}
        
        # Parse additional arguments
        for i, arg in enumerate(args[2:], 2):
            if arg.startswith("--frame="):
                kwargs['frame_style'] = arg.split("=")[1]
            elif arg.startswith("--strength="):
                kwargs['strength'] = float(arg.split("=")[1])
            elif arg.startswith("--output="):
                kwargs['output_path'] = arg.split("=")[1]
        
        return self.wrapper.run_custom_effect("art_gallery", **kwargs)
    
    def _handle_character_command(self, args: List[str]) -> bool:
        """Handle 'character' command to create the character frame projection"""
        print("🐵 Creating 3D character frame projection with Blender...")
        
        try:
            # Use the existing blender_ai_integration.py script
            blender_script = self.wrapper.project_root / "blender_ai_integration.py"
            
            if not blender_script.exists():
                print(f"❌ Blender script not found: {blender_script}")
                return False
            
            # Execute Blender command
            result = subprocess.run([
                "blender", "--background", "--python", str(blender_script)
            ], check=True, capture_output=True, text=True, cwd=self.wrapper.project_root)
            
            print("✅ Character frame projection created successfully!")
            print("📁 Check the output directory: data/output/blender_anamorphic/")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Character frame projection failed: {e}")
            return False
        except FileNotFoundError:
            print("❌ Blender not found in PATH. Please ensure Blender is installed and available.")
            return False


def main():
    """Main entry point for MCP command-line interface"""
    # Ensure we're in the correct directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Create CLI and execute
    cli = MCPCommandLineInterface()
    return cli.parse_and_execute(sys.argv)


if __name__ == "__main__":
    sys.exit(main()) 