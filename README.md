# 🧠 Machine Learning & AI Portfolio
**Komal Shahid** | DSC680 - Applied Data Science | Bellevue University

---

## 📋 **Portfolio Overview**

This repository showcases three comprehensive machine learning and AI projects demonstrating expertise in:
- **Deep Learning** & **Neural Networks**
- **Computer Vision** & **Image Processing**  
- **Natural Language Processing** & **Transformers**
- **Federated Learning** & **Privacy-Preserving AI**
- **MLOps** & **Production Deployment**

---

## 🚀 **Featured Projects**

### 1. 🎭 **Depression Detection System**
> *Advanced NLP & Computer Vision for Mental Health Assessment*

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Transformers](https://img.shields.io/badge/🤗_Transformers-FFD21E?style=flat&logoColor=black)

**📊 Key Features:**
- Multi-modal analysis combining text and image data
- BERT-based transformer models for sentiment analysis
- Computer vision for facial expression analysis
- Real-time prediction with 92% accuracy
- Production-ready REST API

**🔗 [View Project →](./project1-depression-detection/)**

---

### 2. 🏥 **Federated Healthcare AI**
> *Privacy-Preserving Machine Learning for Medical Diagnosis*

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Federated Learning](https://img.shields.io/badge/Federated_Learning-4285F4?style=flat&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)

**📊 Key Features:**
- Federated learning implementation with differential privacy
- Multi-hospital collaboration without data sharing
- Advanced CNN architectures for medical imaging
- HIPAA-compliant deployment pipeline
- Blockchain-based model verification

**🔗 [View Project →](./project2-federated-healthcare-ai/)**

---

### 3.  **Colorful Canvas: Anamorphic Illusion Studio**
> *3D Visual Illusion Generation Using Computer Vision*

![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=flat&logo=OpenCV&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![PIL](https://img.shields.io/badge/PIL-3776AB?style=flat&logoColor=white)

**📊 Key Features:**
- Advanced depth estimation using Intel's DPT model
- Three distinct 3D illusion effects (Shadow Box, Screen Pop, Anamorphic)
- Random Forest ensemble models for performance prediction
- Real-time image processing pipeline
- Commercial advertising performance optimization

**🔗 [View Project →](./project3-colorful-canvas/)**

---

## 🛠️ **Technical Stack**

### **Languages & Frameworks**
- **Python** (Primary) | **JavaScript** | **SQL**
- **TensorFlow** & **PyTorch** | **Scikit-learn** | **OpenCV**
- **Transformers** (Hugging Face) | **NLTK** | **SpaCy**

### **MLOps & Deployment** 
- **Docker** & **Kubernetes** | **AWS** & **Google Cloud**
- **MLflow** | **Git/GitHub** | **CI/CD Pipelines**

### **Specializations**
- **Computer Vision** | **Natural Language Processing**
- **Federated Learning** | **Privacy-Preserving AI**
- **Deep Learning** | **Ensemble Methods**

---

## 📈 **Project Impact & Results**

| Project | Accuracy/Performance | Key Innovation | Real-World Application |
|---------|---------------------|----------------|----------------------|
| Depression Detection | **92% Accuracy** | Multi-modal fusion | Mental health screening |
| Federated Healthcare | **89% F1-Score** | Privacy-preserving ML | Hospital collaboration |
| Colorful Canvas | **R² 0.85** | 3D illusion generation | Digital advertising |

---

## 🏆 **Key Achievements**

- ✅ **End-to-End ML Pipelines** with production deployment
- ✅ **Privacy-Preserving AI** implementation with federated learning
- ✅ **Multi-Modal AI Systems** combining text, image, and structured data
- ✅ **Real-Time Processing** with optimized performance
- ✅ **Industry-Ready Solutions** with comprehensive documentation

---

## 📞 **Contact & Collaboration**

**Komal Shahid**  
📧 **Email:** [kshahid@my.bellevue.edu](mailto:kshahid@my.bellevue.edu)  
🎓 **Program:** Master of Science in Data Science  
🏫 **Institution:** Bellevue University  
📅 **Graduation:** 2024

---

## 📄 **Repository Structure**

```
📁 DSC680-Portfolio/
├── 🎭 project1-depression-detection/     # Mental Health AI
├── 🏥 project2-federated-healthcare-ai/  # Privacy-Preserving ML  
├── 🎨 project3-colorful-canvas/          # Computer Vision Art
├── 📚 docs/                              # Documentation
└── 📋 README.md                          # This file
```

---

**⚡ Ready for production deployment | 🔒 Enterprise-grade security | 📊 Comprehensive documentation**

# Colorful Canvas: Anamorphic Illusion Studio

A comprehensive toolkit for creating stunning anamorphic illusions - 2D images with perspective distortion that appear 3D when viewed from specific angles.

## 🎯 What are Anamorphic Illusions?

**Anamorphic illusions** are specially distorted 2D images that create the illusion of 3D depth when viewed from the correct angle or perspective. These are **not actual 3D images**, but rather mathematically distorted 2D images that exploit human depth perception to create spectacular visual effects.

## 📁 Project Structure

### **Main Implementation**
- `src/colorful_canvas_main.py` - **MAIN FILE** (4,385 lines)
  - Complete AI implementation with all features
  - Seoul LED display optimization
  - Blender integration
  - Training and optimization functionality
  - All anamorphic effects and templates

### **Modular Interface (Optional)**
- `src/colorful_canvas.py` - Core AI & image processing interface
- `src/mcp.py` - Command line interface and MCP wrapper  
- `src/builder.py` - Blender integration interface

### **Documentation & Deliverables**
- `docs/milestone3/` - Milestone 3 documentation and deliverables
- `docs/milestone3/data/` - Project data files
- `docs/milestone3/models/` - Trained model files

### **Supporting Files**
- `blender_ai_integration.py` - Blender Python script
- `complete_anamorphic_system.py` - Alternative implementation
- `setup.py` - Package setup
- `requirements.txt` - Dependencies

## 🚀 Quick Start

### **Recommended: Use Main Implementation**
```python
# Import from main comprehensive file
from src.colorful_canvas_main import ColorfulCanvasAI

# Initialize AI system
ai = ColorfulCanvasAI()

# Create Seoul-optimized test image
test_image = ai.create_seoul_optimized_test_image(800, 600)

# Generate depth map with MPS/CUDA acceleration
depth_map = ai.generate_depth_map(test_image)

# Create anamorphic effects
shadow_effect = ai.create_shadow_box_effect(test_image, depth_map, strength=1.5)
seoul_corner = ai.create_seoul_corner_projection(test_image, depth_map)

# Save results
ai.save_image(shadow_effect, "./output/shadow_box.png")
ai.save_image(seoul_corner, "./output/seoul_corner.png")
```

### **Alternative: Use Modular Interface**
```python
# Use consolidated modules
from src.colorful_canvas import ColorfulCanvasAI
from src.mcp import MCPWrapper
from src.builder import BlenderAnamorphicRenderer

# Same functionality, modular interface
ai = ColorfulCanvasAI()
# ... rest is the same
```

### **Command Line Interface**
```bash
# Using MCP wrapper
python3 src/mcp.py help          # Show available commands
python3 src/mcp.py pipeline      # Run complete pipeline
python3 src/mcp.py seoul         # Create Seoul corner projection
python3 src/mcp.py status        # Check system status
```

## Features

- **Shadow Box Illusion**: Creates the illusion of objects floating in a glass display case when viewed from the front
- **Screen Pop Illusion**: Makes objects appear to pop out of the screen surface with perspective distortion
- **Seoul Corner Projection**: Optimized for curved LED corner displays
- **Anamorphic Billboard**: Creates dramatic perspective distortion perfect for large billboards and street art
- **AI Depth Estimation**: Uses state-of-the-art neural networks (MiDaS, PyTorch) with CUDA/MPS acceleration
- **Product Showcase Templates**: Professional product display effects
- **Art Gallery Templates**: Museum-quality frame effects
- **Professional Blender Integration**: 3D rendering and animation capabilities

## 🔧 Hardware Acceleration

The system automatically detects and uses the best available hardware:

- **NVIDIA CUDA**: Maximum performance on RTX/GTX GPUs
- **Apple Silicon MPS**: Optimized for M1/M2 Macs  
- **CPU Fallback**: Universal compatibility

## Available Illusion Types

- **shadow_box**: Creates illusion of objects floating in a display case
- **screen_pop**: Makes objects appear to emerge from the screen surface  
- **seoul_corner**: Corner LED display projection with dual panels
- **billboard**: Creates billboard-style perspective distortion for large displays
- **product_box**: Adaptive product packaging effects
- **corner_display**: Small/medium/large corner display boxes
- **art_gallery**: Modern/classic frame effects

## 👁️ How It Works

### 1. 🕸️ Depth Analysis
AI analyzes the input image using PyTorch MiDaS to create high-quality depth maps showing relative distances of objects.

### 2. 📐 Seoul Mathematical Transformation
Advanced mathematical transformation calculates pixel distortion based on:
- Object depth and intended viewing angle
- LED display curvature and pixel pitch
- Corner viewing geometry for dual-panel displays

### 3. 🎨 Distortion Application
The image is warped using perspective transformation:
- Closer objects are stretched with extreme distortion
- Farther objects are compressed
- LED intensity calculated for optimal display brightness

### 4. 👁️ Optical Illusion
When viewed from the correct angle (typically 15-45°), your brain interprets the distorted 2D image as having realistic 3D depth!

## Requirements

### Core Dependencies
- Python 3.8+
- NumPy, OpenCV, Pillow (PIL)
- Scikit-learn, Matplotlib, Pandas

### High-Performance Dependencies (Recommended)
- PyTorch 2.0+ (for MiDaS depth estimation)
- CUDA 11.8+ or Apple MPS (for GPU acceleration)
- Transformers library (for advanced AI models)

### Optional Professional Features
- Blender 3.0+ (for 3D rendering and animation)

See `requirements.txt` for complete dependency list.

## Real-World Applications

- **Seoul LED Displays**: Corner installations in shopping malls and public spaces
- **Street Art**: 3D chalk art and pavement illusions optimized for specific viewing angles
- **Digital Advertising**: Eye-catching billboard campaigns with extreme perspective effects
- **Museum Exhibits**: Interactive educational displays with product showcase templates
- **Social Media**: Viral content optimized for mobile viewing
- **Retail Displays**: Product packaging effects and corner display installations

## Scientific Principle

Anamorphic illusions exploit human visual perception through:
- **Perspective Mathematics**: Objects appear smaller when farther away
- **Seoul LED Optimization**: Dual-panel corner viewing with calculated pixel intensities
- **Depth Discontinuities**: Sharp transitions between depth layers enhance the illusion
- **Viewing Angle Compensation**: Precise mathematical compensation for non-perpendicular viewing
- **Brain Processing**: Visual cortex interprets calculated distortion cues as realistic 3D depth

## License

DSC680 - Bellevue University Course Project  
Author: Komal Shahid

## Architecture Notes

The main implementation (`colorful_canvas_main.py`) contains:
- `ColorfulCanvasAI` class (primary interface)
- `AdvancedDepthEstimator` (PyTorch MiDaS integration)
- `BlenderAnamorphicRenderer` (3D rendering capabilities)
- `IllusionPredictor` & `PerformancePredictor` (ML models)
- Seoul optimization and mathematical transformation functions
- Comprehensive training and testing utilities

The modular interfaces provide the same functionality with cleaner separation of concerns for specific use cases.
