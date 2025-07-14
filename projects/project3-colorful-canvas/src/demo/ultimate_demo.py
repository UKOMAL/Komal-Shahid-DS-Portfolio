#!/usr/bin/env python3
"""
Ultimate Anamorphic System Demo
Comprehensive demonstration of all features and capabilities

Author: Komal Shahid
Course: DSC680 - Applied Data Science
"""

import sys
import os
from pathlib import Path
import time
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.ultimate_anamorphic_system import UltimateAnamorphicSystem
from core.anamorphic_config import (
    AnamorphicConfig, ConfigPresets, ConfigManager, ConfigValidator,
    DisplayType, EffectType, QualityLevel
)

class UltimateDemo:
    """Comprehensive demo of the Ultimate Anamorphic System"""
    
    def __init__(self):
        """Initialize the demo"""
        self.output_dir = Path("ultimate_demo_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("🚀 Ultimate Anamorphic System Demo")
        print("=" * 60)
        print("This demo showcases all features of the ultimate system:")
        print("  ✓ Multiple display types and effects")
        print("  ✓ Professional depth estimation")
        print("  ✓ Seoul-style corner optimization")
        print("  ✓ Configuration presets")
        print("  ✓ Quality comparisons")
        print("=" * 60)
    
    def run_complete_demo(self):
        """Run the complete demonstration"""
        try:
            # 1. Configuration Demo
            self.demo_configurations()
            
            # 2. Test Image Creation
            test_images = self.create_test_images()
            
            # 3. Effect Comparisons
            self.demo_effect_comparisons(test_images)
            
            # 4. Quality Comparisons
            self.demo_quality_levels(test_images[0])
            
            # 5. Display Type Comparisons
            self.demo_display_types(test_images[0])
            
            # 6. Performance Analysis
            self.demo_performance_analysis()
            
            # 7. Generate Final Report
            self.generate_demo_report()
            
            print("\n🎉 Ultimate Demo Complete!")
            print(f"All results saved to: {self.output_dir}")
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()
    
    def demo_configurations(self):
        """Demonstrate configuration system"""
        print("\n🔧 Configuration System Demo")
        print("-" * 40)
        
        # Show all presets
        preset_names = ConfigManager.get_preset_names()
        print(f"Available presets ({len(preset_names)}):")
        
        for i, preset_name in enumerate(preset_names, 1):
            config = ConfigManager.get_preset(preset_name)
            is_valid, errors = ConfigValidator.validate_config(config)
            status = "✅" if is_valid else "❌"
            print(f"  {i}. {preset_name} {status}")
            
            if errors:
                for error in errors[:2]:  # Show first 2 errors
                    print(f"     - {error}")
        
        # Save sample configurations
        config_dir = self.output_dir / "configurations"
        config_dir.mkdir(exist_ok=True)
        
        for preset_name in preset_names:
            config = ConfigManager.get_preset(preset_name)
            config_path = config_dir / f"{preset_name}.json"
            ConfigManager.save_config(config, config_path)
        
        print(f"✅ Configurations saved to {config_dir}")
    
    def create_test_images(self):
        """Create various test images for demonstration"""
        print("\n🎨 Creating Test Images")
        print("-" * 40)
        
        test_images = []
        
        # 1. Seoul-optimized test image
        print("Creating Seoul-optimized test image...")
        seoul_config = ConfigPresets.seoul_corner_led()
        seoul_system = UltimateAnamorphicSystem(seoul_config)
        seoul_image = seoul_system.create_test_image(800, 600)
        seoul_path = self.output_dir / "seoul_test_image.png"
        seoul_image.save(seoul_path)
        test_images.append(("Seoul Test", seoul_image, seoul_path))
        
        # 2. Billboard-optimized test image
        print("Creating billboard-optimized test image...")
        billboard_image = self.create_billboard_test_image()
        billboard_path = self.output_dir / "billboard_test_image.png"
        billboard_image.save(billboard_path)
        test_images.append(("Billboard Test", billboard_image, billboard_path))
        
        # 3. Aquarium-optimized test image
        print("Creating aquarium-optimized test image...")
        aquarium_image = self.create_aquarium_test_image()
        aquarium_path = self.output_dir / "aquarium_test_image.png"
        aquarium_image.save(aquarium_path)
        test_images.append(("Aquarium Test", aquarium_image, aquarium_path))
        
        print(f"✅ Created {len(test_images)} test images")
        return test_images
    
    def create_billboard_test_image(self):
        """Create billboard-optimized test image"""
        width, height = 1200, 800
        image = Image.new('RGB', (width, height), color=(40, 45, 55))
        
        # Add large text for billboard visibility
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(image)
        
        try:
            font_large = ImageFont.truetype("arial.ttf", 120)
            font_medium = ImageFont.truetype("arial.ttf", 60)
        except:
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
        
        # Main text
        text = "BILLBOARD"
        text_bbox = draw.textbbox((0, 0), text, font=font_large)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = (width - text_width) // 2
        text_y = height // 2 - 100
        
        # Text with outline
        for offset in [(2, 2), (-2, -2), (2, -2), (-2, 2)]:
            draw.text((text_x + offset[0], text_y + offset[1]), text, 
                     fill=(0, 0, 0), font=font_large)
        draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font_large)
        
        # Subtitle
        subtitle = "ANAMORPHIC EFFECT"
        sub_bbox = draw.textbbox((0, 0), subtitle, font=font_medium)
        sub_width = sub_bbox[2] - sub_bbox[0]
        sub_x = (width - sub_width) // 2
        sub_y = text_y + 150
        
        draw.text((sub_x, sub_y), subtitle, fill=(200, 200, 255), font=font_medium)
        
        # Add geometric shapes for depth
        shapes = [
            (200, 200, 100),  # x, y, size
            (900, 300, 80),
            (300, 500, 120),
            (800, 600, 90)
        ]
        
        for x, y, size in shapes:
            # Create 3D-looking rectangles
            draw.rectangle([x, y, x + size, y + size], fill=(100, 150, 200))
            draw.rectangle([x + 10, y - 10, x + size + 10, y + size - 10], 
                          fill=(150, 200, 255))
        
        return image
    
    def create_aquarium_test_image(self):
        """Create aquarium-optimized test image"""
        width, height = 800, 600
        image = Image.new('RGB', (width, height), color=(10, 30, 60))
        
        from PIL import ImageDraw
        draw = ImageDraw.Draw(image)
        
        # Water gradient background
        for y in range(height):
            blue_intensity = int(10 + (y / height) * 50)
            draw.line([(0, y), (width, y)], 
                     fill=(blue_intensity//3, blue_intensity//2, blue_intensity))
        
        # Floating fish-like shapes
        import random
        for _ in range(8):
            x = random.randint(50, width - 100)
            y = random.randint(50, height - 100)
            size = random.randint(30, 80)
            
            # Fish body (ellipse)
            draw.ellipse([x, y, x + size, y + size//2], 
                        fill=(100 + random.randint(0, 100), 
                              150 + random.randint(0, 50),
                              200 + random.randint(0, 55)))
            
            # Fish tail (triangle)
            tail_points = [
                (x, y + size//4),
                (x - size//3, y + size//6),
                (x - size//3, y + size//3)
            ]
            draw.polygon(tail_points, fill=(80, 120, 180))
        
        # Bubbles
        for _ in range(20):
            x = random.randint(0, width)
            y = random.randint(0, height)
            size = random.randint(5, 20)
            
            draw.ellipse([x, y, x + size, y + size], 
                        fill=(150, 200, 255, 100))
        
        # Water surface line
        surface_y = 50
        draw.line([(0, surface_y), (width, surface_y)], 
                 fill=(200, 230, 255), width=3)
        
        return image
    
    def demo_effect_comparisons(self, test_images):
        """Demonstrate different effect types"""
        print("\n🎭 Effect Type Comparisons")
        print("-" * 40)
        
        # Use the Seoul test image for comparisons
        test_name, test_image, test_path = test_images[0]
        
        effects = [
            (EffectType.SEOUL_CORNER, "Seoul Corner LED"),
            (EffectType.SHADOW_BOX, "Shadow Box Illusion"),
            (EffectType.SCREEN_POP, "Screen Pop Effect"),
            (EffectType.FLOATING_OBJECTS, "Floating Objects")
        ]
        
        effect_results = {}
        
        for effect_type, effect_name in effects:
            print(f"Generating {effect_name}...")
            
            # Create configuration for this effect
            config = AnamorphicConfig()
            config.effect_type = effect_type
            config.render.quality = QualityLevel.MEDIUM  # Faster for demo
            
            # Create system and generate effect
            system = UltimateAnamorphicSystem(config)
            start_time = time.time()
            
            try:
                results = system.create_ultimate_anamorphic_effect(test_image)
                processing_time = time.time() - start_time
                
                # Save results
                effect_dir = self.output_dir / "effects" / effect_type.value
                effect_dir.mkdir(parents=True, exist_ok=True)
                
                saved_files = {}
                for result_name, result_image in results.items():
                    file_path = effect_dir / f"{result_name}.png"
                    result_image.save(file_path)
                    saved_files[result_name] = file_path
                
                effect_results[effect_name] = {
                    'files': saved_files,
                    'processing_time': processing_time,
                    'success': True
                }
                
                print(f"  ✅ {effect_name} completed in {processing_time:.2f}s")
                
            except Exception as e:
                print(f"  ❌ {effect_name} failed: {e}")
                effect_results[effect_name] = {
                    'success': False,
                    'error': str(e)
                }
        
        # Create comparison grid
        self.create_effect_comparison_grid(effect_results, test_image)
        
        print(f"✅ Effect comparisons complete")
    
    def demo_quality_levels(self, test_image_data):
        """Demonstrate different quality levels"""
        print("\n🎯 Quality Level Comparisons")
        print("-" * 40)
        
        test_name, test_image, test_path = test_image_data
        
        quality_levels = [
            (QualityLevel.LOW, "Low Quality (Fast)"),
            (QualityLevel.MEDIUM, "Medium Quality"),
            (QualityLevel.HIGH, "High Quality"),
            (QualityLevel.ULTRA, "Ultra Quality")
        ]
        
        quality_results = {}
        
        for quality_level, quality_name in quality_levels:
            print(f"Processing {quality_name}...")
            
            # Create configuration
            config = ConfigPresets.seoul_corner_led()
            config.render.quality = quality_level
            
            # Adjust samples based on quality
            if quality_level == QualityLevel.LOW:
                config.render.samples = 64
            elif quality_level == QualityLevel.MEDIUM:
                config.render.samples = 128
            elif quality_level == QualityLevel.HIGH:
                config.render.samples = 256
            else:  # ULTRA
                config.render.samples = 512
            
            system = UltimateAnamorphicSystem(config)
            start_time = time.time()
            
            try:
                results = system.create_ultimate_anamorphic_effect(test_image)
                processing_time = time.time() - start_time
                
                # Save main result
                quality_dir = self.output_dir / "quality_levels"
                quality_dir.mkdir(exist_ok=True)
                
                main_result = results.get('combined', results.get('left_panel', list(results.values())[0]))
                file_path = quality_dir / f"{quality_level.value}_quality.png"
                main_result.save(file_path)
                
                quality_results[quality_name] = {
                    'file': file_path,
                    'processing_time': processing_time,
                    'samples': config.render.samples,
                    'success': True
                }
                
                print(f"  ✅ {quality_name}: {processing_time:.2f}s ({config.render.samples} samples)")
                
            except Exception as e:
                print(f"  ❌ {quality_name} failed: {e}")
                quality_results[quality_name] = {
                    'success': False,
                    'error': str(e)
                }
        
        # Create quality comparison chart
        self.create_quality_comparison_chart(quality_results)
        
        print("✅ Quality level comparisons complete")
    
    def demo_display_types(self, test_image_data):
        """Demonstrate different display types"""
        print("\n📺 Display Type Comparisons")
        print("-" * 40)
        
        test_name, test_image, test_path = test_image_data
        
        display_configs = [
            ("Seoul Wave", ConfigPresets.seoul_corner_led()),
            ("Billboard", ConfigPresets.billboard_advertising()),
            ("Aquarium", ConfigPresets.aquarium_display()),
            ("Holographic", ConfigPresets.holographic_display())
        ]
        
        display_results = {}
        
        for display_name, config in display_configs:
            print(f"Processing {display_name} display...")
            
            # Use medium quality for faster processing
            config.render.quality = QualityLevel.MEDIUM
            
            system = UltimateAnamorphicSystem(config)
            start_time = time.time()
            
            try:
                results = system.create_ultimate_anamorphic_effect(test_image)
                processing_time = time.time() - start_time
                
                # Save results
                display_dir = self.output_dir / "display_types" / display_name.lower().replace(" ", "_")
                display_dir.mkdir(parents=True, exist_ok=True)
                
                saved_files = {}
                for result_name, result_image in results.items():
                    file_path = display_dir / f"{result_name}.png"
                    result_image.save(file_path)
                    saved_files[result_name] = file_path
                
                display_results[display_name] = {
                    'files': saved_files,
                    'processing_time': processing_time,
                    'config': config,
                    'success': True
                }
                
                print(f"  ✅ {display_name}: {processing_time:.2f}s")
                
            except Exception as e:
                print(f"  ❌ {display_name} failed: {e}")
                display_results[display_name] = {
                    'success': False,
                    'error': str(e)
                }
        
        print("✅ Display type comparisons complete")
    
    def demo_performance_analysis(self):
        """Analyze system performance"""
        print("\n⚡ Performance Analysis")
        print("-" * 40)
        
        # Test different image sizes
        sizes = [(400, 300), (800, 600), (1200, 900), (1600, 1200)]
        performance_data = []
        
        config = ConfigPresets.seoul_corner_led()
        config.render.quality = QualityLevel.MEDIUM
        
        for width, height in sizes:
            print(f"Testing {width}x{height}...")
            
            system = UltimateAnamorphicSystem(config)
            test_image = system.create_test_image(width, height)
            
            start_time = time.time()
            try:
                results = system.create_ultimate_anamorphic_effect(test_image)
                processing_time = time.time() - start_time
                
                performance_data.append({
                    'resolution': f"{width}x{height}",
                    'pixels': width * height,
                    'processing_time': processing_time,
                    'pixels_per_second': (width * height) / processing_time,
                    'success': True
                })
                
                print(f"  ✅ {width}x{height}: {processing_time:.2f}s")
                
            except Exception as e:
                print(f"  ❌ {width}x{height} failed: {e}")
                performance_data.append({
                    'resolution': f"{width}x{height}",
                    'pixels': width * height,
                    'success': False,
                    'error': str(e)
                })
        
        # Create performance chart
        self.create_performance_chart(performance_data)
        
        print("✅ Performance analysis complete")
    
    def create_effect_comparison_grid(self, effect_results, original_image):
        """Create a comparison grid of different effects"""
        successful_effects = {name: data for name, data in effect_results.items() 
                            if data.get('success', False)}
        
        if not successful_effects:
            print("No successful effects to compare")
            return
        
        # Create comparison figure
        n_effects = len(successful_effects)
        fig, axes = plt.subplots(2, (n_effects + 1) // 2, figsize=(15, 10))
        fig.suptitle('Anamorphic Effect Comparisons', fontsize=16)
        
        if n_effects == 1:
            axes = [axes]
        elif len(axes.shape) == 1:
            axes = axes.reshape(1, -1)
        
        # Show original image first
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Show effect results
        for i, (effect_name, effect_data) in enumerate(successful_effects.items(), 1):
            row = i // 2
            col = i % 2
            
            if row < axes.shape[0] and col < axes.shape[1]:
                # Get the main result image
                files = effect_data['files']
                main_file = files.get('combined', files.get('left_panel', list(files.values())[0]))
                
                if main_file and main_file.exists():
                    result_image = Image.open(main_file)
                    axes[row, col].imshow(result_image)
                    axes[row, col].set_title(f'{effect_name}\n({effect_data["processing_time"]:.2f}s)')
                    axes[row, col].axis('off')
        
        # Hide unused subplots
        for i in range(len(successful_effects) + 1, axes.size):
            row = i // axes.shape[1]
            col = i % axes.shape[1]
            if row < axes.shape[0] and col < axes.shape[1]:
                axes[row, col].axis('off')
        
        plt.tight_layout()
        comparison_path = self.output_dir / "effect_comparison.png"
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Effect comparison saved to {comparison_path}")
    
    def create_quality_comparison_chart(self, quality_results):
        """Create quality vs performance comparison chart"""
        successful_results = {name: data for name, data in quality_results.items() 
                            if data.get('success', False)}
        
        if not successful_results:
            return
        
        # Extract data
        quality_names = list(successful_results.keys())
        processing_times = [data['processing_time'] for data in successful_results.values()]
        samples = [data['samples'] for data in successful_results.values()]
        
        # Create chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Processing time chart
        ax1.bar(quality_names, processing_times, color=['green', 'yellow', 'orange', 'red'])
        ax1.set_title('Processing Time by Quality Level')
        ax1.set_ylabel('Time (seconds)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Samples chart
        ax2.bar(quality_names, samples, color=['lightblue', 'blue', 'darkblue', 'navy'])
        ax2.set_title('Samples by Quality Level')
        ax2.set_ylabel('Number of Samples')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        chart_path = self.output_dir / "quality_comparison.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Quality comparison chart saved to {chart_path}")
    
    def create_performance_chart(self, performance_data):
        """Create performance analysis chart"""
        successful_data = [data for data in performance_data if data.get('success', False)]
        
        if not successful_data:
            return
        
        resolutions = [data['resolution'] for data in successful_data]
        times = [data['processing_time'] for data in successful_data]
        pixels = [data['pixels'] for data in successful_data]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Processing time vs resolution
        ax1.plot(resolutions, times, 'bo-', linewidth=2, markersize=8)
        ax1.set_title('Processing Time vs Resolution')
        ax1.set_ylabel('Time (seconds)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Pixels vs processing time (efficiency)
        ax2.scatter(pixels, times, c='red', s=100, alpha=0.7)
        ax2.set_title('Efficiency: Pixels vs Processing Time')
        ax2.set_xlabel('Total Pixels')
        ax2.set_ylabel('Time (seconds)')
        ax2.grid(True, alpha=0.3)
        
        # Add trend line
        if len(pixels) > 1:
            z = np.polyfit(pixels, times, 1)
            p = np.poly1d(z)
            ax2.plot(pixels, p(pixels), "r--", alpha=0.8)
        
        plt.tight_layout()
        perf_path = self.output_dir / "performance_analysis.png"
        plt.savefig(perf_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance analysis saved to {perf_path}")
    
    def generate_demo_report(self):
        """Generate comprehensive demo report"""
        print("\n📊 Generating Demo Report")
        print("-" * 40)
        
        report_path = self.output_dir / "demo_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Ultimate Anamorphic System Demo Report\n\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## System Overview\n")
            f.write("The Ultimate Anamorphic System combines:\n")
            f.write("- Professional depth estimation with MiDaS neural networks\n")
            f.write("- Seoul-style corner LED optimization\n")
            f.write("- Multiple display type support\n")
            f.write("- Configurable effect parameters\n")
            f.write("- Quality level optimization\n\n")
            
            f.write("## Demo Results\n\n")
            
            # List generated files
            f.write("### Generated Files\n")
            for file_path in self.output_dir.rglob("*.png"):
                rel_path = file_path.relative_to(self.output_dir)
                f.write(f"- {rel_path}\n")
            
            f.write("\n### Configuration Files\n")
            for file_path in self.output_dir.rglob("*.json"):
                rel_path = file_path.relative_to(self.output_dir)
                f.write(f"- {rel_path}\n")
            
            f.write("\n## Key Features Demonstrated\n")
            f.write("1. **Multiple Effect Types**: Seoul Corner, Shadow Box, Screen Pop, Floating Objects\n")
            f.write("2. **Quality Levels**: Low, Medium, High, Ultra with performance trade-offs\n")
            f.write("3. **Display Types**: Corner LED, Billboard, Aquarium, Holographic\n")
            f.write("4. **Performance Scaling**: Analysis across different image resolutions\n")
            f.write("5. **Configuration Management**: Presets, validation, and persistence\n\n")
            
            f.write("## Technical Achievements\n")
            f.write("- ✅ Professional-grade depth estimation\n")
            f.write("- ✅ Seoul-style mathematical projection\n")
            f.write("- ✅ Anti-aliased pixel placement\n")
            f.write("- ✅ Multiple depth cue fusion\n")
            f.write("- ✅ Display-specific optimization\n")
            f.write("- ✅ Comprehensive configuration system\n")
            f.write("- ✅ Performance optimization\n\n")
            
            f.write("## Usage Instructions\n")
            f.write("```python\n")
            f.write("from core.ultimate_anamorphic_system import UltimateAnamorphicSystem\n")
            f.write("from core.anamorphic_config import ConfigPresets\n\n")
            f.write("# Load preset configuration\n")
            f.write("config = ConfigPresets.seoul_corner_led()\n\n")
            f.write("# Create system\n")
            f.write("system = UltimateAnamorphicSystem(config)\n\n")
            f.write("# Process image\n")
            f.write("results = system.process_image('input.jpg', 'output')\n")
            f.write("```\n\n")
            
            f.write("## Conclusion\n")
            f.write("The Ultimate Anamorphic System successfully demonstrates professional-grade ")
            f.write("anamorphic billboard generation with multiple display types, effect styles, ")
            f.write("and quality levels. The system is optimized for both quality and performance, ")
            f.write("making it suitable for real-world applications.\n")
        
        print(f"✅ Demo report saved to {report_path}")

def main():
    """Run the ultimate demo"""
    demo = UltimateDemo()
    demo.run_complete_demo()

if __name__ == "__main__":
    main() 