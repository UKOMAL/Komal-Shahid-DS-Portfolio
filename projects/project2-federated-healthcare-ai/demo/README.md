# 🏥 Federated Healthcare AI - Interactive Demo

## 🚀 **Quick Start**

### **Web-Based Simulation**
```bash
cd project2-federated-healthcare-ai/demo
python -m http.server 8001
# Open http://localhost:8001 in your browser
```

### **Command Line Demo**
```bash
cd project2-federated-healthcare-ai/src
python federated_demo.py --hospitals 4 --rounds 10
```

## 🎯 **Demo Features**

### **🌐 Interactive Simulation** (`index.html`)
- **Multi-Hospital Network:** Visualize 3-5 simulated hospitals
- **Real-Time Training:** Watch federated learning rounds
- **Privacy Metrics:** See differential privacy in action
- **Performance Tracking:** Model accuracy across institutions
- **Data Flow Visualization:** No raw data leaves hospitals

### **💻 Command Line Simulation**
- **Configurable Setup:** Choose number of hospitals and rounds
- **Performance Metrics:** Precision, recall, F1-score per hospital
- **Privacy Analysis:** Epsilon values and noise injection
- **Convergence Plots:** Model improvement over time

## 📊 **Sample Results**

**Federated Training Progress:**
```
Round 1: Global Accuracy = 0.72 ± 0.08
Round 5: Global Accuracy = 0.84 ± 0.05  
Round 10: Global Accuracy = 0.89 ± 0.03
```

**Privacy Guarantees:**
```
Differential Privacy: ε = 1.0
Noise Level: σ = 0.1
Patient Records Protected: 100%
```

## 🎨 **Generated Visualizations**
- `hospital_network.png` - Federated network topology
- `training_progress.png` - Accuracy convergence over rounds
- `privacy_metrics.png` - Privacy-utility tradeoff analysis
- `hospital_contributions.png` - Individual hospital performance

## 🔒 **Privacy Demonstrations**
- **No Data Sharing:** Only model updates are exchanged
- **Differential Privacy:** Added noise prevents individual identification
- **Secure Aggregation:** Encrypted gradient sharing
- **HIPAA Compliance:** Meets healthcare privacy standards

## 🏥 **Use Cases Demonstrated**
- **Medical Imaging:** Collaborative radiology AI training
- **EHR Analysis:** Cross-hospital patient outcome prediction
- **Drug Discovery:** Multi-pharmaceutical collaboration
- **Epidemic Modeling:** Public health surveillance

---

**🎯 Ready to demonstrate privacy-preserving AI for healthcare!** 