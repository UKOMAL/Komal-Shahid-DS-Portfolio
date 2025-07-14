#!/usr/bin/env python3
"""
Create a proper billboard effect from the benchmark image
"""

import sys
import os
sys.path.append('.')

from core.ultimate_anamorphic_system import UltimateAnamorphicSystem
from core.anamorphic_config import ConfigPresets

def main():
    print("🎨 Creating benchmark billboard...")
    
    # Use Seoul corner LED configuration
    config = ConfigPresets.seoul_corner_led()
    system = UltimateAnamorphicSystem(config)
    
    # Process the benchmark image
    benchmark_path = '../data/input/benchmark.jpg'
    output_dir = 'benchmark_billboard'
    
    if os.path.exists(benchmark_path):
        print(f"📸 Processing: {benchmark_path}")
        results = system.process_image(benchmark_path, output_dir)
        print("✅ Benchmark billboard generated!")
        print(f"📁 Output directory: {output_dir}")
        
        # List generated files
        if os.path.exists(output_dir):
            files = os.listdir(output_dir)
            print("📋 Generated files:")
            for file in files:
                print(f"  - {file}")
    else:
        print(f"❌ Benchmark image not found: {benchmark_path}")

if __name__ == "__main__":
    main() 