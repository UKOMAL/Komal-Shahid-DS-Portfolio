// Sample ECG data (simulated)
const sampleData = {
    normal: [0.1, 0.2, 0.1, 0.1, 0.2, 0.5, 1.2, 0.9, 0.2, 0.1, 0.0, -0.1, -0.2, -0.1, 0.0, 0.1, 0.1, 0.1, 0.0, 0.1, 0.1, 0.1, 0.2, 0.5, 1.2, 0.9, 0.2, 0.1, 0.0, -0.1, -0.2, -0.1, 0.0, 0.1, 0.1, 0.1, 0.0, 0.1, 0.1, 0.1, 0.2, 0.5, 1.2, 0.9, 0.2, 0.1, 0.0, -0.1, -0.2, -0.1],
    afib: [0.1, 0.2, 0.1, 0.3, 0.7, 0.5, 0.2, 0.1, 0.0, 0.2, 0.4, 0.6, 0.2, 0.1, 0.0, -0.1, 0.2, 0.3, 0.1, 0.0, 0.2, 0.5, 0.3, 0.1, 0.1, 0.2, 0.1, 0.4, 0.3, 0.5, 0.6, 0.2, 0.1, 0.3, 0.1, 0.2, 0.1, 0.0, 0.4, 0.2, 0.3, 0.5, 0.2, 0.1, 0.0, 0.3, 0.2, 0.1, 0.3, 0.2],
    heartblock: [0.1, 0.2, 0.1, 0.1, 0.2, 0.5, 1.2, 0.9, 0.2, 0.1, 0.0, -0.1, -0.2, -0.1, 0.0, 0.1, 0.1, 0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.1, 0.1, 0.2, 0.5, 1.2, 0.9, 0.2, 0.1, 0.0, -0.1, -0.2, -0.1, 0.0, 0.1, 0.1, 0.1, 0.0, 0.1],
    arrhythmia: [0.1, 0.2, 0.3, 0.4, 0.7, 1.0, 0.7, 0.4, 0.2, 0.1, 0.0, -0.1, -0.2, -0.1, 0.0, 0.2, 0.1, 0.1, 0.3, 0.8, 1.2, 0.8, 0.4, 0.2, 0.1, 0.0, -0.1, -0.2, -0.1, 0.0, 0.1, 0.2, 0.1, 0.1, 0.2, 0.3, 0.4, 0.6, 1.4, 0.8, 0.5, 0.2, 0.1, 0.0, -0.1, -0.2, -0.1, 0.0, 0.1, 0.1]
};

// Predicted results for each condition
const predictionResults = {
    normal: {
        prediction: "Normal heart rhythm",
        confidence: 0.92,
        details: "Regular sinus rhythm detected. No abnormalities identified.",
        recommendations: "No action required. Continue regular monitoring."
    },
    afib: {
        prediction: "Atrial Fibrillation",
        confidence: 0.87,
        details: "Irregular rhythm detected with absence of consistent P waves.",
        recommendations: "Consult with a healthcare provider for further evaluation."
    },
    heartblock: {
        prediction: "Heart Block",
        confidence: 0.83,
        details: "Extended pause between atrial and ventricular activity detected.",
        recommendations: "Medical attention recommended for further assessment."
    },
    arrhythmia: {
        prediction: "Ventricular Arrhythmia",
        confidence: 0.89,
        details: "Irregular ventricular rhythm with wide QRS complexes.",
        recommendations: "Seek immediate medical attention."
    }
};

// DOM elements
const hospitalSelector = document.getElementById('hospital-selector');
const sampleButtons = document.querySelectorAll('.sample-btn');
const runInferenceBtn = document.getElementById('run-inference');
const ecgDisplay = document.getElementById('ecg-display');
const resultsDisplay = document.getElementById('results-display');
const animationStatus = document.getElementById('animation-status');
const nodes = document.querySelectorAll('.node');
const centralNode = document.getElementById('central');

// State
let selectedSample = null;
let selectedHospital = 'hospital1';
let canvas = null;
let ctx = null;

// Initialize the canvas
function initCanvas() {
    if (!canvas) {
        ecgDisplay.innerHTML = '';
        canvas = document.createElement('canvas');
        ecgDisplay.appendChild(canvas);
        ctx = canvas.getContext('2d');
        
        // Set canvas dimensions
        canvas.width = ecgDisplay.clientWidth;
        canvas.height = ecgDisplay.clientHeight;
    }
}

// Draw ECG on canvas
function drawECG(data) {
    initCanvas();
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Set line style
    ctx.strokeStyle = '#2e8b57';
    ctx.lineWidth = 2;
    
    // Calculate scaling factors
    const xStep = canvas.width / data.length;
    const yMiddle = canvas.height / 2;
    const yScale = canvas.height / 4;
    
    // Start drawing
    ctx.beginPath();
    ctx.moveTo(0, yMiddle - data[0] * yScale);
    
    // Draw lines for each data point
    for (let i = 1; i < data.length; i++) {
        ctx.lineTo(i * xStep, yMiddle - data[i] * yScale);
    }
    
    ctx.stroke();
    
    // Draw grid
    drawGrid();
}

// Draw ECG grid
function drawGrid() {
    ctx.strokeStyle = '#e0e0e0';
    ctx.lineWidth = 0.5;
    
    // Draw horizontal grid lines
    for (let y = 0; y <= canvas.height; y += 20) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
        ctx.stroke();
    }
    
    // Draw vertical grid lines
    for (let x = 0; x <= canvas.width; x += 20) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
        ctx.stroke();
    }
}

// Display results
function displayResults(condition) {
    const result = predictionResults[condition];
    
    resultsDisplay.innerHTML = `
        <div class="result-header">
            <h4>${result.prediction}</h4>
            <div class="confidence">Confidence: ${(result.confidence * 100).toFixed(1)}%</div>
        </div>
        <div class="result-details">
            <p><strong>Details:</strong> ${result.details}</p>
            <p><strong>Recommendations:</strong> ${result.recommendations}</p>
        </div>
    `;
}

// Simulate federated learning process
function simulateFederation() {
    // Reset all nodes
    nodes.forEach(node => {
        node.classList.remove('active', 'sending');
    });
    
    // Get selected hospital node
    const hospitalNode = document.getElementById(selectedHospital === 'hospital1' ? 'node1' : 
                                                 selectedHospital === 'hospital2' ? 'node2' : 'node3');
    
    // Animation sequence
    animationStatus.textContent = "Step 1: Local model training at " + hospitalNode.textContent;
    
    // Activate hospital node
    setTimeout(() => {
        hospitalNode.classList.add('active');
    }, 500);
    
    // Send updates to central server
    setTimeout(() => {
        animationStatus.textContent = "Step 2: Sending model updates (not patient data) to central server";
        hospitalNode.classList.add('sending');
    }, 2000);
    
    // Central aggregation
    setTimeout(() => {
        hospitalNode.classList.remove('sending');
        centralNode.classList.add('active');
        animationStatus.textContent = "Step 3: Central server aggregates model updates";
    }, 3500);
    
    // Global model update
    setTimeout(() => {
        animationStatus.textContent = "Step 4: Updated global model distributed to all institutions";
        centralNode.classList.add('sending');
    }, 5000);
    
    // Complete
    setTimeout(() => {
        nodes.forEach(node => node.classList.remove('active', 'sending'));
        animationStatus.textContent = "Federated learning process complete";
    }, 6500);
}

// Event listeners
sampleButtons.forEach(button => {
    button.addEventListener('click', () => {
        // Update active button
        sampleButtons.forEach(btn => btn.classList.remove('active'));
        button.classList.add('active');
        
        // Get selected sample
        selectedSample = button.dataset.sample;
        
        // Draw ECG
        drawECG(sampleData[selectedSample]);
        
        // Enable run button
        runInferenceBtn.disabled = false;
    });
});

hospitalSelector.addEventListener('change', (e) => {
    selectedHospital = e.target.value;
});

runInferenceBtn.addEventListener('click', () => {
    // Only run if sample is selected
    if (!selectedSample) return;
    
    // Display loading
    resultsDisplay.innerHTML = '<p class="loading">Processing data...</p>';
    
    // Simulate processing delay
    setTimeout(() => {
        // Display results
        displayResults(selectedSample);
        
        // Simulate federated learning
        simulateFederation();
    }, 1000);
});

// Handle window resize
window.addEventListener('resize', () => {
    if (selectedSample) {
        canvas = null; // Reset canvas to recreate with new dimensions
        drawECG(sampleData[selectedSample]);
    }
});

// Initialize with placeholder
document.addEventListener('DOMContentLoaded', () => {
    animationStatus.textContent = "Select a sample and run inference to see the federated learning process";
}); 