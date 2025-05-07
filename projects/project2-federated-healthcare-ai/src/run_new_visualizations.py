#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run New Visualizations

This script runs the newly added advanced visualization scripts:
- model_complexity_3d.py
- convergence_analysis.py

This ensures they're properly generated even if they're not found in the
output visualizations directory by the main runner.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Main function to run the new visualization scripts."""
    print("Federated Healthcare AI - New Visualizations Generator")
    print("=" * 60)
    
    # Get absolute paths
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "output" / "visualizations"
    
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Project root: {project_root}")
    print(f"Visualizations will be saved to: {output_dir}")
    print("-" * 60)
    
    # Create the new advanced visualization scripts if they don't exist
    create_visualization_scripts(output_dir)
    
    # List of new visualization scripts
    new_scripts = [
        "model_complexity_3d.py",
        "convergence_analysis.py"
    ]
    
    # Run each visualization script
    success_count = 0
    for script in new_scripts:
        script_path = output_dir / script
        if script_path.exists():
            print(f"Running {script}...")
            try:
                result = subprocess.run([sys.executable, str(script_path)], 
                                        check=True, 
                                        capture_output=True,
                                        text=True)
                print(f"✅ Successfully generated visualization from {script}")
                print(f"Output: {result.stdout}")
                success_count += 1
            except subprocess.CalledProcessError as e:
                print(f"❌ Error running {script}: {e}")
                print(f"Error output: {e.stderr}")
        else:
            print(f"⚠️ Script {script} not found at {script_path}")
        
        print("-" * 60)
    
    # Copy visualization files to docs directory for the presentation
    docs_images_dir = project_root / "docs" / "images"
    os.makedirs(docs_images_dir, exist_ok=True)
    
    print(f"Copying visualization images to {docs_images_dir}")
    
    # Find all new image files in the visualizations directory
    copy_image_files = [
        "model_complexity_3d.png",
        "convergence_animation.gif",
        "convergence_final.png"
    ]
    
    # Copy the new files
    import shutil
    for img_name in copy_image_files:
        img_file = output_dir / img_name
        if img_file.exists():
            target_path = docs_images_dir / img_name
            # Use shutil.copy2 to preserve metadata
            shutil.copy2(img_file, target_path)
            print(f"✅ Copied {img_name} to {target_path}")
        else:
            print(f"⚠️ Image file {img_name} not found at {img_file}")
            
    print("\nSummary:")
    print(f"- Total new visualization scripts: {len(new_scripts)}")
    print(f"- Successfully generated: {success_count}")
    print(f"- Failed: {len(new_scripts) - success_count}")
    
    print("\nNew visualization process complete!")

def create_visualization_scripts(output_dir):
    """Create the visualization scripts if they don't exist."""
    # Check if model_complexity_3d.py exists
    model_complexity_path = output_dir / "model_complexity_3d.py"
    if not model_complexity_path.exists():
        print("Creating model_complexity_3d.py script...")
        # Get the script content from the source file in this directory
        script_content = """#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
3D Model Complexity Visualization

Generates a 3D scatter plot showing the relationship between model complexity,
privacy budget, and model performance in federated healthcare AI.
\"\"\"

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Create directory if it doesn't exist
os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)

# Set the style
plt.style.use('seaborn-v0_8-whitegrid')

# Create figure and 3D axis
fig = plt.figure(figsize=(14, 12))
ax = fig.add_subplot(111, projection='3d')

# Generate simulated data
np.random.seed(42)

# Define model complexity (number of parameters in millions)
model_complexity = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0])

# Define privacy budget (epsilon values, lower = more privacy)
privacy_budget = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0])

# Create a grid of values
X, Y = np.meshgrid(model_complexity, privacy_budget)

# Model performance (accuracy) - complex formula to simulate realistic relationship
Z = 0.75 + 0.15 * (1 - np.exp(-0.1 * X)) - 0.1 * np.exp(-Y/2)
# Add some noise
Z = Z + np.random.normal(0, 0.01, Z.shape)
# Ensure values are in reasonable range
Z = np.clip(Z, 0.5, 0.99)

# Convert to 1D arrays for scatter plot
x = X.flatten()
y = Y.flatten()
z = Z.flatten()

# Use different colors for different model complexities
norm = plt.Normalize(np.min(x), np.max(x))
colors = cm.viridis(norm(x))

# Create 3D scatter plot
scatter = ax.scatter(x, y, z, c=colors, s=70, alpha=0.8, edgecolors='black', linewidths=0.5)

# Add a color bar
cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.viridis), 
                    ax=ax, shrink=0.6, aspect=20, 
                    label='Model Complexity (millions of parameters)')

# Add surface fit to show the trend
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.3, linewidth=0, antialiased=True)

# Customize the plot
ax.set_title('Relationship Between Model Complexity, Privacy, and Performance', fontsize=16)
ax.set_xlabel('Model Complexity (millions of parameters)', fontsize=14)
ax.set_ylabel('Privacy Budget (ε)', fontsize=14)
ax.set_zlabel('Model Accuracy', fontsize=14)

# Use log scale for better visualization
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(min(model_complexity), max(model_complexity))
ax.set_ylim(min(privacy_budget), max(privacy_budget))
ax.set_zlim(0.5, 1.0)

# Add labels for notable points
idx_max_performance = np.argmax(z)
ax.text(x[idx_max_performance], y[idx_max_performance], z[idx_max_performance],
        f'Best Performance\\n({x[idx_max_performance]:.1f}M params, ε={y[idx_max_performance]:.1f})',
        color='red', fontsize=10, ha='center', va='bottom')

# Find optimal privacy-performance tradeoff (highest z for moderate privacy)
moderate_privacy = (y <= 2.0) & (z > 0.85)
if np.any(moderate_privacy):
    idx_best_tradeoff = np.where(moderate_privacy)[0][np.argmax(z[moderate_privacy])]
    ax.text(x[idx_best_tradeoff], y[idx_best_tradeoff], z[idx_best_tradeoff],
            f'Best Privacy-Utility\\nTradeoff',
            color='green', fontsize=10, ha='center', va='bottom')

# Add grid for better depth perception
ax.grid(True)

# Add horizontal plane at 90% accuracy
xx, yy = np.meshgrid(np.linspace(min(model_complexity), max(model_complexity), 2),
                    np.linspace(min(privacy_budget), max(privacy_budget), 2))
zz = np.ones(xx.shape) * 0.9
ax.plot_surface(xx, yy, zz, color='red', alpha=0.1)
ax.text(xx[0, 0], yy[0, 0], 0.9, '90% Accuracy Threshold', color='red', fontsize=10)

# Add viewpoint for optimal visibility
ax.view_init(elev=20, azim=-50)

# Add explanatory subtitle
plt.figtext(0.5, 0.01, 
           "This 3D visualization shows the relationship between model complexity (number of parameters),\\n"
           "privacy budget (epsilon parameter in differential privacy), and model accuracy.\\n"
           "Higher model complexity generally improves performance but with diminishing returns,\\n"
           "while stronger privacy guarantees (lower epsilon) typically reduce accuracy.",
           ha="center", fontsize=12, style='italic')

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)

# Save the figure
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_complexity_3d.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved visualization to {output_path}")

# Show the plot
plt.show()

if __name__ == "__main__":
    print("Generated 3D model complexity visualization.")
"""
        
        # Write the script to the output directory
        with open(model_complexity_path, 'w') as f:
            f.write(script_content)
        print(f"Created {model_complexity_path}")
    
    # Check if convergence_analysis.py exists
    convergence_path = output_dir / "convergence_analysis.py"
    if not convergence_path.exists():
        print("Creating convergence_analysis.py script...")
        # Get the script content from the source file in this directory
        script_content = """#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
Model Convergence Time-Series Visualization

Generates an animated visualization showing model convergence under different
federated learning strategies over training rounds.
\"\"\"

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import MaxNLocator

# Create directory if it doesn't exist
os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)

# Set the style
plt.style.use('seaborn-v0_8-whitegrid')

# Create figure and axes
fig, ax = plt.subplots(figsize=(14, 8))

# Define different federated learning strategies
strategies = [
    {'name': 'FedAvg (No Privacy)', 'color': '#3498db', 'linestyle': '-', 'final_acc': 0.95},
    {'name': 'FedAvg + DP (ε=10.0)', 'color': '#2ecc71', 'linestyle': '-', 'final_acc': 0.92},
    {'name': 'FedAvg + DP (ε=1.0)', 'color': '#e74c3c', 'linestyle': '-', 'final_acc': 0.88},
    {'name': 'FedProx (No Privacy)', 'color': '#9b59b6', 'linestyle': '--', 'final_acc': 0.94},
    {'name': 'FedProx + DP (ε=1.0)', 'color': '#f39c12', 'linestyle': '--', 'final_acc': 0.89},
    {'name': 'SCAFFOLD', 'color': '#1abc9c', 'linestyle': '-.', 'final_acc': 0.93}
]

# Define parameters
num_rounds = 100
animation_length = 20  # seconds
fps = 30
frames = fps * animation_length

# Generate simulated data
np.random.seed(42)
x = np.linspace(0, num_rounds, num_rounds)
convergence_curves = []

for strategy in strategies:
    # Generate convergence curve with logarithmic approach to final accuracy
    final_acc = strategy['final_acc']
    rate = np.random.uniform(0.1, 0.2)  # Different convergence rates
    curve = final_acc - (final_acc - 0.5) * np.exp(-rate * x)
    
    # Add some noise and realistic variations
    noise = np.random.normal(0, 0.01, num_rounds)
    # More noise in early rounds, less in later rounds
    noise = noise * np.exp(-0.03 * x)
    
    # Add occasional drops (simulating bad rounds)
    for i in range(3):
        drop_idx = np.random.randint(10, 80)
        drop_size = np.random.uniform(0.02, 0.06)
        curve[drop_idx:drop_idx+3] -= drop_size
    
    curve += noise
    # Ensure curve is monotonically increasing overall (with small dips)
    for i in range(1, num_rounds):
        if curve[i] < curve[i-1] - 0.03:
            curve[i] = curve[i-1] - np.random.uniform(0.005, 0.03)
    
    convergence_curves.append(curve)

# Function to update the animation frame
def update(frame):
    ax.clear()
    
    # Calculate the round to display based on frame number
    round_to_show = int(np.ceil((frame / frames) * num_rounds))
    
    # Plot each strategy up to the current round
    for i, strategy in enumerate(strategies):
        curve = convergence_curves[i][:round_to_show]
        rounds = x[:round_to_show]
        line, = ax.plot(rounds, curve, color=strategy['color'], 
                       linestyle=strategy['linestyle'], linewidth=3, 
                       label=f"{strategy['name']} ({curve[-1]:.3f})" if len(curve) > 0 else strategy['name'])
        
        # Add a marker at the latest point
        if len(curve) > 0:
            ax.plot(rounds[-1], curve[-1], 'o', color=strategy['color'], 
                   markersize=8, markeredgecolor='black', markeredgewidth=1.5)
    
    # Add annotations when certain rounds are reached
    if round_to_show == 15:
        ax.annotate("Early divergence for\\ndifferent strategies",
                   xy=(10, 0.65), xytext=(20, 0.55),
                   arrowprops=dict(arrowstyle='->'))
        
    if round_to_show == 50:
        # Find the strategy with the steepest recent improvement
        improvements = [convergence_curves[i][49] - convergence_curves[i][39] for i in range(len(strategies))]
        best_improvement_idx = np.argmax(improvements)
        strategy = strategies[best_improvement_idx]
        ax.annotate(f"{strategy['name']}\\nshowing fastest\\nimprovement",
                   xy=(45, convergence_curves[best_improvement_idx][49]), 
                   xytext=(55, convergence_curves[best_improvement_idx][49] - 0.1),
                   arrowprops=dict(arrowstyle='->'),
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    if round_to_show == num_rounds:
        # Final annotations showing the ranking
        sorted_indices = np.argsort([curve[-1] for curve in convergence_curves])[::-1]
        ax.annotate("Final Ranking:",
                   xy=(80, 0.55), xytext=(80, 0.55),
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black"))
        
        for i, idx in enumerate(sorted_indices):
            ax.annotate(f"{i+1}. {strategies[idx]['name']} ({convergence_curves[idx][-1]:.3f})",
                       xy=(80, 0.52 - i*0.03), xytext=(80, 0.52 - i*0.03))
    
    # Customize the plot
    ax.set_xlim(0, num_rounds)
    ax.set_ylim(0.45, 1.0)
    ax.set_xlabel('Training Rounds', fontsize=14)
    ax.set_ylabel('Model Accuracy', fontsize=14)
    ax.set_title(f'Federated Learning Convergence Analysis (Round {round_to_show}/{num_rounds})', fontsize=16)
    ax.legend(loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Add privacy-performance tradeoff annotation
    if round_to_show > 75:
        # Calculate the privacy-performance gap 
        no_privacy_idx = 0  # Index of FedAvg with no privacy
        dp_privacy_idx = 2  # Index of FedAvg+DP (ε=1.0) 
        
        privacy_gap = convergence_curves[no_privacy_idx][-1] - convergence_curves[dp_privacy_idx][-1]
        
        if round_to_show == num_rounds:
            highlight_x = [x[-1], x[-1]]
            highlight_y = [convergence_curves[dp_privacy_idx][-1], convergence_curves[no_privacy_idx][-1]]
            ax.plot(highlight_x, highlight_y, 'k--', alpha=0.5)
            ax.annotate(f"Privacy cost:\\n{privacy_gap:.3f} accuracy drop",
                       xy=(highlight_x[0], np.mean(highlight_y)),
                       xytext=(highlight_x[0]-30, np.mean(highlight_y)),
                       arrowprops=dict(arrowstyle='->'))
    
    return ax,

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=frames, blit=False, interval=1000/fps)

# Add explanatory subtitle
plt.figtext(0.5, 0.01,
           "This animation visualizes how different federated learning strategies converge over training rounds.\\n"
           "Notice how differential privacy (DP) strategies have lower final accuracy but offer privacy guarantees.\\n"
           "FedProx and SCAFFOLD methods help address client heterogeneity issues in federated learning.",
           ha="center", fontsize=12, style='italic')

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)

# Save as animated GIF
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'convergence_animation.gif')
# Reduce fps for file size considerations
ani.save(output_path, writer='pillow', fps=15, dpi=150)
print(f"Saved animated visualization to {output_path}")

# Save a static final frame as PNG for preview
output_static_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'convergence_final.png')
update(frames-1)  # Update to final frame
plt.savefig(output_static_path, dpi=300, bbox_inches='tight')
print(f"Saved static final frame to {output_static_path}")

# Show the plot
plt.show()

if __name__ == "__main__":
    print("Generated model convergence animation.")
"""
        
        # Write the script to the output directory
        with open(convergence_path, 'w') as f:
            f.write(script_content)
        print(f"Created {convergence_path}")

if __name__ == "__main__":
    main() 