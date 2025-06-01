# 🎭 Depression Detection System - Live Demo

## 🚀 **Quick Start**

### **Option 1: Web Interface (Recommended)**
```bash
cd project1-depression-detection/demo/web
python -m http.server 8000
# Open http://localhost:8000 in your browser
```

### **Option 2: Command Line Demo**
```bash
cd project1-depression-detection/demo
python demo.py
```

## 🎯 **Demo Features**

### **🌐 Web Interface** (`web/index.html`)
- **Text Analysis:** Real-time depression severity assessment
- **Confidence Scores:** Visual confidence indicators
- **Attention Visualization:** See which words the model focuses on
- **Sample Texts:** Pre-loaded examples for testing
- **System Architecture:** Detailed model explanation

### **💻 Command Line Demo** (`demo.py`)
- **Batch Analysis:** Process multiple texts at once
- **Visualization Generation:** Creates charts and graphs
- **Performance Metrics:** Precision, recall, F1-score
- **Feature Importance:** Shows most influential features

## 📊 **Sample Results**

**Input:** *"I feel like I'm stuck in a rut and can't get out."*
```
Depression Severity: moderate
Confidence Scores:
  minimum: 0.15
  mild: 0.25
  moderate: 0.45
  severe: 0.15
```

## 🎨 **Generated Visualizations**
- `attention_weights.png` - Word attention heatmap
- `severity_distribution.png` - Severity level distribution  
- `sentiment_distribution.png` - Sentiment vs severity analysis
- `feature_importance.png` - Most important text features

## 🔒 **Ethical Considerations**
- **Not for clinical diagnosis** - Educational/research purposes only
- **Privacy-preserving** - No data is stored or transmitted
- **Transparent AI** - Attention weights show model reasoning

---

**🎯 Ready for portfolio presentation and live demonstrations!** 