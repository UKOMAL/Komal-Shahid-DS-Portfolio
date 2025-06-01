---
layout: default
title: Interactive Demo - Federated Healthcare AI
description: "Try our federated learning simulation for healthcare institutions"
---

<style>
  .demo-container {
    background-color: #f8f9fa;
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 30px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
  }
  
  .control-panel {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 15px;
    margin-bottom: 20px;
  }
  
  .control-group {
    margin-bottom: 15px;
  }
  
  .control-group label {
    display: block;
    margin-bottom: 5px;
    font-weight: bold;
  }
  
  .button-row {
    display: flex;
    justify-content: space-between;
    margin-top: 20px;
  }
  
  button {
    background-color: #4CAF50;
    color: white;
    border: none;
    padding: 10px 15px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 16px;
  }
  
  button:hover {
    background-color: #45a049;
  }
  
  button:disabled {
    background-color: #cccccc;
    cursor: not-allowed;
  }
  
  button.secondary {
    background-color: #2196F3;
  }
  
  button.secondary:hover {
    background-color: #0b7dda;
  }
  
  .results-panel {
    border-top: 1px solid #ddd;
    padding-top: 20px;
    margin-top: 20px;
  }
  
  .progress-container {
    margin: 20px 0;
  }
  
  .progress-bar {
    width: 100%;
    background-color: #e0e0e0;
    border-radius: 4px;
    position: relative;
    height: 20px;
  }
  
  .progress-fill {
    height: 100%;
    border-radius: 4px;
    background-color: #4CAF50;
    transition: width 0.3s ease;
    width: 0%;
  }
  
  .metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 15px;
    margin: 20px 0;
  }
  
  .metric-box {
    background-color: white;
    border-radius: 4px;
    padding: 15px;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  }
  
  .metric-value {
    font-size: 24px;
    font-weight: bold;
    color: #2196F3;
  }
  
  .metric-label {
    font-size: 14px;
    color: #666;
  }
  
  .visualization-container {
    margin-top: 30px;
    text-align: center;
  }
  
  .tab-container {
    margin-top: 20px;
  }
  
  .tab-buttons {
    display: flex;
    border-bottom: 1px solid #ddd;
    margin-bottom: 15px;
  }
  
  .tab-button {
    padding: 10px 15px;
    background-color: transparent;
    border: none;
    border-bottom: 3px solid transparent;
    color: #333;
    cursor: pointer;
    margin-right: 5px;
  }
  
  .tab-button.active {
    border-bottom-color: #4CAF50;
    font-weight: bold;
  }
  
  .tab-content {
    display: none;
  }
  
  .tab-content.active {
    display: block;
  }
  
  .institution-list {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
    gap: 15px;
    margin-top: 20px;
  }
  
  .institution-card {
    background-color: white;
    border-radius: 8px;
    padding: 15px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  }
  
  .institution-status {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 12px;
    font-size: 12px;
    margin-top: 5px;
  }
  
  .status-active {
    background-color: #e8f5e9;
    color: #2e7d32;
  }
  
  .status-inactive {
    background-color: #ffebee;
    color: #c62828;
  }
</style>

<div class="demo-container">
  <h1>Interactive Federated Learning Demo</h1>
  <p>Experience how federated learning works by configuring various parameters and watching the system learn across simulated healthcare institutions without sharing patient data.</p>
  
  <div class="control-panel">
    <div class="control-group">
      <label for="num-institutions">Number of Institutions:</label>
      <select id="num-institutions" class="form-control">
        <option value="3">3 Institutions</option>
        <option value="5" selected>5 Institutions</option>
        <option value="10">10 Institutions</option>
        <option value="20">20 Institutions</option>
      </select>
    </div>
    
    <div class="control-group">
      <label for="data-modality">Healthcare Data Type:</label>
      <select id="data-modality" class="form-control">
        <option value="tabular" selected>Clinical Records (Tabular)</option>
        <option value="imaging">Medical Imaging</option>
        <option value="timeseries">Physiological Signals</option>
      </select>
    </div>
    
    <div class="control-group">
      <label for="privacy-level">Privacy Protection Level:</label>
      <select id="privacy-level" class="form-control">
        <option value="none">No Privacy (ε = ∞)</option>
        <option value="low">Low (ε = 10.0)</option>
        <option value="medium" selected>Medium (ε = 1.0)</option>
        <option value="high">High (ε = 0.1)</option>
      </select>
    </div>
    
    <div class="control-group">
      <label for="training-rounds">Training Rounds:</label>
      <select id="training-rounds" class="form-control">
        <option value="5">Quick (5 rounds)</option>
        <option value="10" selected>Standard (10 rounds)</option>
        <option value="20">Extended (20 rounds)</option>
      </select>
    </div>
  </div>
  
  <div class="button-row">
    <button id="start-button" onclick="startSimulation()">Start Simulation</button>
    <button id="reset-button" class="secondary" onclick="resetSimulation()" disabled>Reset</button>
  </div>
  
  <div id="simulation-container" style="display: none;">
    <div class="progress-container">
      <h3>Simulation Progress</h3>
      <div class="progress-bar">
        <div id="progress-fill" class="progress-fill"></div>
      </div>
      <p id="progress-text">Training round 0 of 10</p>
    </div>
    
    <div class="results-panel">
      <h3>Current Results</h3>
      
      <div class="metrics-grid">
        <div class="metric-box">
          <div id="metric-accuracy" class="metric-value">-</div>
          <div class="metric-label">Accuracy</div>
        </div>
        <div class="metric-box">
          <div id="metric-f1" class="metric-value">-</div>
          <div class="metric-label">F1 Score</div>
        </div>
        <div class="metric-box">
          <div id="metric-privacy" class="metric-value">-</div>
          <div class="metric-label">Privacy Budget</div>
        </div>
        <div class="metric-box">
          <div id="metric-comms" class="metric-value">-</div>
          <div class="metric-label">Communication (MB)</div>
        </div>
      </div>
      
      <div class="tab-container">
        <div class="tab-buttons">
          <button class="tab-button active" onclick="showTab('tab-visualization')">Visualizations</button>
          <button class="tab-button" onclick="showTab('tab-institutions')">Institutions</button>
          <button class="tab-button" onclick="showTab('tab-model')">Model Details</button>
        </div>
        
        <div id="tab-visualization" class="tab-content active">
          <div class="visualization-container">
            <img id="convergence-chart" src="../images/convergence_final.png" alt="Convergence Chart" style="max-width: 100%; border-radius: 5px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
            <p><em>Model convergence across training rounds</em></p>
          </div>
        </div>
        
        <div id="tab-institutions" class="tab-content">
          <h3>Participating Healthcare Institutions</h3>
          <div id="institution-container" class="institution-list">
            <!-- Will be populated by JavaScript -->
          </div>
        </div>
        
        <div id="tab-model" class="tab-content">
          <h3>Federated Model Architecture</h3>
          <pre id="model-architecture" style="background-color: #f5f5f5; padding: 15px; border-radius: 5px; overflow-x: auto;">
# Model will be displayed here during simulation
          </pre>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
  // Simulation variables
  let isRunning = false;
  let currentRound = 0;
  let totalRounds = 10;
  let simulationInterval;
  let institutions = [];
  
  // Models for different data modalities
  const modelArchitectures = {
    'tabular': `class TabularModel(nn.Module):
    def __init__(self, input_dim=42, hidden_dims=[256, 128, 64], output_dim=2):
        super(TabularModel, self).__init__()
        
        # Build layers dynamically based on hidden_dims
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.Dropout(0.3))
            prev_dim = dim
            
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)`,
        
    'imaging': `class MedicalImageCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(MedicalImageCNN, self).__init__()
        
        # Feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x`,
        
    'timeseries': `class TimeseriesLSTM(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=128, num_layers=2, 
                 bidirectional=True, num_classes=5):
        super(TimeseriesLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.3 if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Linear(
            hidden_dim * self.num_directions, 1
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim * self.num_directions, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch_size, seq_length, hidden_dim*num_directions)
        
        # Apply attention
        attention_weights = F.softmax(
            self.attention(lstm_out), dim=1
        )
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Final prediction
        output = self.fc(context_vector)
        return output`
  };
  
  // Privacy mechanisms based on privacy level
  const privacyMechanisms = {
    'none': 'No privacy protection applied. All gradients are shared as-is.',
    'low': 'Gradient clipping and minimal noise (ε=10.0) applied to updates.',
    'medium': 'Differential privacy (ε=1.0) with calibrated Gaussian noise.',
    'high': 'Strong differential privacy (ε=0.1) with secure aggregation.'
  };
  
  // Accuracy progression for different privacy levels
  const accuracyProgression = {
    'none': [0.50, 0.68, 0.78, 0.85, 0.88, 0.90, 0.92, 0.93, 0.94, 0.95],
    'low': [0.49, 0.65, 0.75, 0.82, 0.85, 0.88, 0.90, 0.91, 0.92, 0.93],
    'medium': [0.48, 0.62, 0.72, 0.78, 0.81, 0.83, 0.85, 0.86, 0.87, 0.88],
    'high': [0.45, 0.58, 0.65, 0.71, 0.74, 0.76, 0.78, 0.79, 0.80, 0.81]
  };
  
  // F1 score progression for different privacy levels
  const f1Progression = {
    'none': [0.48, 0.65, 0.76, 0.83, 0.86, 0.88, 0.90, 0.91, 0.92, 0.93],
    'low': [0.47, 0.63, 0.73, 0.80, 0.83, 0.86, 0.88, 0.89, 0.90, 0.91],
    'medium': [0.46, 0.60, 0.70, 0.76, 0.79, 0.81, 0.83, 0.84, 0.85, 0.86],
    'high': [0.43, 0.55, 0.63, 0.69, 0.72, 0.74, 0.76, 0.77, 0.78, 0.79]
  };
  
  // Create simulated institutions
  function createInstitutions(count) {
    const types = [
      'Academic Medical Center',
      'Community Hospital',
      'Rural Hospital',
      'Specialized Clinic',
      'Research Institute'
    ];
    
    const sizes = [
      'Large (10,000+ patients)',
      'Medium (5,000-10,000 patients)',
      'Small (1,000-5,000 patients)',
      'Very Small (<1,000 patients)'
    ];
    
    const locations = [
      'Northeast', 'Southeast', 'Midwest', 
      'Southwest', 'West Coast', 'Northwest'
    ];
    
    const institutions = [];
    
    for (let i = 0; i < count; i++) {
      const typeIndex = i % types.length;
      const sizeIndex = i % sizes.length;
      const locationIndex = i % locations.length;
      
      institutions.push({
        id: i + 1,
        name: `${types[typeIndex]} ${i + 1}`,
        type: types[typeIndex],
        size: sizes[sizeIndex],
        location: locations[locationIndex],
        dataSize: Math.floor(1000 + Math.random() * 9000),
        status: 'active'
      });
    }
    
    return institutions;
  }
  
  // Show tab content
  function showTab(tabId) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
      tab.classList.remove('active');
    });
    
    // Deactivate all buttons
    document.querySelectorAll('.tab-button').forEach(button => {
      button.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(tabId).classList.add('active');
    
    // Activate the clicked button
    event.currentTarget.classList.add('active');
  }
  
  // Update institution list
  function updateInstitutionList() {
    const container = document.getElementById('institution-container');
    container.innerHTML = '';
    
    institutions.forEach(inst => {
      const card = document.createElement('div');
      card.className = 'institution-card';
      
      const statusClass = inst.status === 'active' ? 'status-active' : 'status-inactive';
      const statusText = inst.status === 'active' ? 'Active' : 'Inactive';
      
      card.innerHTML = `
        <h4>${inst.name}</h4>
        <p><strong>Type:</strong> ${inst.type}</p>
        <p><strong>Size:</strong> ${inst.size}</p>
        <p><strong>Location:</strong> ${inst.location}</p>
        <p><strong>Data samples:</strong> ${inst.dataSize.toLocaleString()}</p>
        <span class="institution-status ${statusClass}">${statusText}</span>
      `;
      
      container.appendChild(card);
    });
  }
  
  // Start the simulation
  function startSimulation() {
    if (isRunning) return;
    
    // Get parameters
    const institutionCount = parseInt(document.getElementById('num-institutions').value);
    const dataModality = document.getElementById('data-modality').value;
    const privacyLevel = document.getElementById('privacy-level').value;
    totalRounds = parseInt(document.getElementById('training-rounds').value);
    
    // Reset simulation state
    currentRound = 0;
    isRunning = true;
    
    // Create institutions
    institutions = createInstitutions(institutionCount);
    updateInstitutionList();
    
    // Update UI
    document.getElementById('start-button').disabled = true;
    document.getElementById('reset-button').disabled = false;
    document.getElementById('simulation-container').style.display = 'block';
    document.getElementById('progress-text').textContent = `Training round 0 of ${totalRounds}`;
    document.getElementById('progress-fill').style.width = '0%';
    
    // Set model architecture
    document.getElementById('model-architecture').textContent = modelArchitectures[dataModality];
    
    // Update visualization based on privacy level
    if (privacyLevel === 'none') {
      document.getElementById('convergence-chart').src = '../images/convergence_final.png';
    } else {
      document.getElementById('convergence-chart').src = '../images/privacy_radar.png';
    }
    
    // Reset metrics
    document.getElementById('metric-accuracy').textContent = '-';
    document.getElementById('metric-f1').textContent = '-';
    document.getElementById('metric-privacy').textContent = '-';
    document.getElementById('metric-comms').textContent = '-';
    
    // Start simulation interval
    simulationInterval = setInterval(() => {
      currentRound++;
      
      // Update progress
      const progress = (currentRound / totalRounds) * 100;
      document.getElementById('progress-fill').style.width = `${progress}%`;
      document.getElementById('progress-text').textContent = `Training round ${currentRound} of ${totalRounds}`;
      
      // Update metrics
      if (currentRound <= 10) {
        const accuracyArray = accuracyProgression[privacyLevel] || accuracyProgression['medium'];
        const f1Array = f1Progression[privacyLevel] || f1Progression['medium'];
        
        const roundIndex = Math.min(currentRound - 1, accuracyArray.length - 1);
        const accuracy = accuracyArray[roundIndex];
        const f1 = f1Array[roundIndex];
        
        document.getElementById('metric-accuracy').textContent = accuracy.toFixed(2);
        document.getElementById('metric-f1').textContent = f1.toFixed(2);
        
        // Update privacy budget consumed
        let privacyBudget = '-';
        if (privacyLevel === 'low') {
          privacyBudget = ((currentRound / totalRounds) * 10).toFixed(1);
        } else if (privacyLevel === 'medium') {
          privacyBudget = ((currentRound / totalRounds) * 1).toFixed(2);
        } else if (privacyLevel === 'high') {
          privacyBudget = ((currentRound / totalRounds) * 0.1).toFixed(3);
        }
        document.getElementById('metric-privacy').textContent = privacyBudget;
        
        // Update communication cost
        const commsCost = (currentRound * 1.2 * institutionCount).toFixed(1);
        document.getElementById('metric-comms').textContent = commsCost;
      }
      
      // End simulation if complete
      if (currentRound >= totalRounds) {
        clearInterval(simulationInterval);
        isRunning = false;
        document.getElementById('start-button').disabled = true;
        document.getElementById('reset-button').disabled = false;
      }
    }, 1000);
  }
  
  // Reset the simulation
  function resetSimulation() {
    // Stop the interval
    if (simulationInterval) {
      clearInterval(simulationInterval);
    }
    
    // Reset state
    isRunning = false;
    currentRound = 0;
    
    // Reset UI
    document.getElementById('start-button').disabled = false;
    document.getElementById('reset-button').disabled = true;
    document.getElementById('simulation-container').style.display = 'none';
  }
  
  // Initialize the demo
  document.addEventListener('DOMContentLoaded', function() {
    // Initialize institutions 
    institutions = createInstitutions(5);
    updateInstitutionList();
  });
</script>

## How the Demo Works

This interactive demo simulates our federated learning framework for healthcare. It illustrates how multiple healthcare institutions can collaborate on AI model training without sharing patient data.

### What You're Seeing

1. **Multiple Institutions**: Each institution trains the model locally on their own data
2. **Privacy Protection**: Different levels of differential privacy can be applied
3. **Model Convergence**: Watch how the model improves over training rounds
4. **Performance Metrics**: Track accuracy and other key metrics

### Real System vs. Demo

While this demo simulates the process, our actual system includes:

- Real data preprocessing pipelines for healthcare data
- Secure aggregation protocols with cryptographic protection
- Multiple neural network architectures for different healthcare modalities
- Communication optimization for bandwidth-constrained environments
- Comprehensive privacy attack defenses

## Learn More

- [View the complete white paper](white_paper.md)
- [Explore the source code on GitHub](https://github.com/komal-shahid/federated-healthcare-ai)
- [Return to main page](index.md) 