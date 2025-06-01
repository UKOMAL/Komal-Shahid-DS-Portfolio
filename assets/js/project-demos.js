/**
 * Project Demos
 * JavaScript implementation of project demos for interactive portfolio
 */

// Depression Detection Demo
const depressionDetectionDemo = {
  // Sample model responses (simulating the Python model)
  modelResponses: {
    positive: {
      depression_severity: "minimum",
      confidence_scores: {
        "minimum": 0.92,
        "mild": 0.06,
        "moderate": 0.01,
        "severe": 0.01
      },
      key_features: ["positive sentiment", "future-oriented", "social connection"]
    },
    negative: {
      depression_severity: "moderate",
      confidence_scores: {
        "minimum": 0.05,
        "mild": 0.15,
        "moderate": 0.72,
        "severe": 0.08
      },
      key_features: ["negative sentiment", "hopelessness", "lack of energy"]
    },
    severe: {
      depression_severity: "severe",
      confidence_scores: {
        "minimum": 0.01,
        "mild": 0.03,
        "moderate": 0.18,
        "severe": 0.78
      },
      key_features: ["suicidal ideation", "extreme negativity", "isolation"]
    }
  },

  // Analyze text input and return results
  analyzeText: function(text) {
    // Simple classification based on text content
    // This is a simplified version of what the actual ML model would do
    const lowercaseText = text.toLowerCase();
    
    // Check for positive indicators
    if (lowercaseText.includes("happy") || 
        lowercaseText.includes("excited") || 
        lowercaseText.includes("looking forward") ||
        lowercaseText.includes("proud") ||
        lowercaseText.includes("accomplished")) {
      return this.modelResponses.positive;
    }
    
    // Check for severe indicators
    if (lowercaseText.includes("suicide") || 
        lowercaseText.includes("kill myself") || 
        lowercaseText.includes("end it all") ||
        lowercaseText.includes("no reason to live")) {
      return this.modelResponses.severe;
    }
    
    // Check for negative indicators
    if (lowercaseText.includes("sad") || 
        lowercaseText.includes("depressed") || 
        lowercaseText.includes("tired") ||
        lowercaseText.includes("don't care") ||
        lowercaseText.includes("hopeless") ||
        lowercaseText.includes("stuck") ||
        lowercaseText.includes("can't") ||
        lowercaseText.includes("never")) {
      return this.modelResponses.negative;
    }
    
    // Default to positive for neutral text
    return this.modelResponses.positive;
  },
  
  // Render results in the demo container
  renderResults: function(result, container) {
    const severityColorMap = {
      "minimum": "#32CD32", // Green
      "mild": "#FFA500",    // Orange
      "moderate": "#FF4500", // OrangeRed
      "severe": "#FF0000"    // Red
    };
    
    // Create results HTML
    let html = `
      <div class="demo-result-container">
        <h3>Analysis Results</h3>
        <div class="result-severity">
          <span>Depression Severity:</span>
          <span class="severity-label" style="color:${severityColorMap[result.depression_severity]}">
            ${result.depression_severity.toUpperCase()}
          </span>
        </div>
        
        <div class="confidence-scores">
          <h4>Confidence Scores</h4>
          <div class="score-bars">
    `;
    
    // Add bars for each confidence score
    for (const [label, score] of Object.entries(result.confidence_scores)) {
      html += `
        <div class="score-bar-container">
          <div class="score-label">${label}</div>
          <div class="score-bar-wrapper">
            <div class="score-bar" style="width:${score * 100}%; background-color:${severityColorMap[label] || '#ccc'}"></div>
          </div>
          <div class="score-value">${(score * 100).toFixed(0)}%</div>
        </div>
      `;
    }
    
    html += `
          </div>
        </div>
        
        <div class="key-features">
          <h4>Key Features Detected</h4>
          <ul>
    `;
    
    // Add key features
    for (const feature of result.key_features) {
      html += `<li>${feature}</li>`;
    }
    
    html += `
          </ul>
        </div>
        
        <div class="demo-disclaimer">
          <p><strong>Note:</strong> This is a simplified demonstration and should not be used for actual diagnosis. 
          The real system uses advanced NLP and machine learning techniques with much more comprehensive analysis.</p>
        </div>
      </div>
    `;
    
    // Set the HTML to the container
    container.innerHTML = html;
  }
};

// Federated Healthcare AI Demo 
const federatedHealthcareDemo = {
  // Simulate different healthcare institutions
  institutions: [
    { name: "Hospital A", patients: 2500, dataQuality: 0.89 },
    { name: "Hospital B", patients: 1800, dataQuality: 0.92 },
    { name: "Clinic C", patients: 950, dataQuality: 0.77 },
    { name: "Research Institute D", patients: 1200, dataQuality: 0.95 }
  ],
  
  // Simulated model performance metrics
  performanceMetrics: {
    accuracy: 0.87,
    precision: 0.84,
    recall: 0.83,
    f1Score: 0.835,
    auc: 0.91
  },
  
  // Simulated privacy budget tracking
  privacyBudget: {
    initial: 10.0,
    remaining: 8.2,
    used: 1.8,
    epsilon: 0.8,
    delta: 0.00001
  },
  
  // Simulate federated learning rounds
  simulateTraining: function(rounds = 10) {
    // Generate simulated training data (rounds, accuracy)
    const trainingData = [];
    let accuracy = 0.5; // Starting accuracy
    
    for (let i = 0; i < rounds; i++) {
      // Add some randomness to the learning progress
      const improvement = 0.04 * Math.random() + 0.02;
      accuracy = Math.min(accuracy + improvement, 0.95);
      trainingData.push({
        round: i + 1,
        accuracy: accuracy,
        privacyBudgetUsed: (i + 1) * 0.2
      });
    }
    
    return trainingData;
  },
  
  // Render federated learning simulation
  renderSimulation: function(container, rounds = 10) {
    const trainingData = this.simulateTraining(rounds);
    
    // Create chart container
    const chartContainer = document.createElement('div');
    chartContainer.className = 'federated-chart-container';
    container.appendChild(chartContainer);
    
    // Create HTML for the simulation dashboard
    let html = `
      <h3>Federated Learning Simulation</h3>
      
      <div class="simulation-metrics">
        <div class="metric-card">
          <div class="metric-value">${this.institutions.length}</div>
          <div class="metric-label">Participating Institutions</div>
        </div>
        <div class="metric-card">
          <div class="metric-value">${trainingData.length}</div>
          <div class="metric-label">Training Rounds</div>
        </div>
        <div class="metric-card">
          <div class="metric-value">${this.performanceMetrics.accuracy.toFixed(2)}</div>
          <div class="metric-label">Final Accuracy</div>
        </div>
        <div class="metric-card">
          <div class="metric-value">${this.privacyBudget.epsilon.toFixed(1)}</div>
          <div class="metric-label">Privacy Budget (ε)</div>
        </div>
      </div>
      
      <div class="chart-area">
        <canvas id="federated-chart" width="600" height="300"></canvas>
      </div>
      
      <div class="institution-participation">
        <h4>Participating Institutions</h4>
        <div class="institution-grid">
    `;
    
    // Add institution cards
    this.institutions.forEach(institution => {
      html += `
        <div class="institution-card">
          <div class="institution-name">${institution.name}</div>
          <div class="institution-patients">${institution.patients} patients</div>
          <div class="institution-quality">Data Quality: ${(institution.dataQuality * 100).toFixed(0)}%</div>
        </div>
      `;
    });
    
    html += `
        </div>
      </div>
      
      <div class="demo-disclaimer">
        <p><strong>Note:</strong> This is a simplified demonstration. The actual system implements 
        secure aggregation, differential privacy, and advanced federated optimization techniques.</p>
      </div>
    `;
    
    // Set the HTML to the container
    chartContainer.innerHTML = html;
    
    // Draw chart (if Chart.js is available)
    setTimeout(() => {
      if (window.Chart) {
        const ctx = document.getElementById('federated-chart').getContext('2d');
        new Chart(ctx, {
          type: 'line',
          data: {
            labels: trainingData.map(d => `Round ${d.round}`),
            datasets: [{
              label: 'Model Accuracy',
              data: trainingData.map(d => d.accuracy),
              borderColor: '#ff96c7',
              backgroundColor: 'rgba(255, 150, 199, 0.1)',
              tension: 0.3,
              yAxisID: 'y'
            }, {
              label: 'Privacy Budget Used',
              data: trainingData.map(d => d.privacyBudgetUsed),
              borderColor: '#a896ff',
              backgroundColor: 'rgba(168, 150, 255, 0.1)',
              tension: 0.3,
              yAxisID: 'y1'
            }]
          },
          options: {
            responsive: true,
            scales: {
              y: {
                beginAtZero: true,
                title: {
                  display: true,
                  text: 'Accuracy'
                },
                max: 1
              },
              y1: {
                beginAtZero: true,
                position: 'right',
                title: {
                  display: true,
                  text: 'Privacy Budget Used'
                },
                grid: {
                  drawOnChartArea: false
                }
              }
            }
          }
        });
      }
    }, 100);
  }
};

// Initialize demos when required
function initializeProjectDemos() {
  // Setup event handlers for demo forms
  const demoForm = document.getElementById('demo-form');
  
  if (demoForm) {
    demoForm.addEventListener('submit', function(e) {
      e.preventDefault();
      
      const demoInput = document.getElementById('demo-input');
      const demoResult = document.getElementById('demo-result');
      const projectId = this.getAttribute('data-project-id');
      
      if (projectId === 'depression-detection') {
        const result = depressionDetectionDemo.analyzeText(demoInput.value);
        depressionDetectionDemo.renderResults(result, demoResult);
      } else if (projectId === 'federated-healthcare-ai') {
        federatedHealthcareDemo.renderSimulation(demoResult);
      }
    });
  }
}

// Make functions available globally
window.projectDemos = {
  depression: depressionDetectionDemo,
  federated: federatedHealthcareDemo,
  initialize: initializeProjectDemos
}; 