const express = require('express');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = 9000;

// Get the current directory path
const currentDir = process.cwd();

// Serve static files from the current directory
app.use(express.static(currentDir));

// Create the assets/files directory if it doesn't exist
const filesDir = path.join(currentDir, 'assets', 'files');
if (!fs.existsSync(filesDir)) {
  fs.mkdirSync(filesDir, { recursive: true });
}

// Create placeholder demo files for projects
const createPlaceholderFiles = () => {
  const placeholders = [
    {
      filename: 'federated_healthcare_whitepaper.pdf',
      content: '<h1>Federated Healthcare AI Research Paper</h1><p>This is a placeholder for the Federated Healthcare AI research paper.</p>'
    },
    {
      filename: 'network_viz_demo.html',
      content: '<h1>Network Visualization Demo</h1><p>This is a demo of the Network Visualization tool.</p>'
    },
    {
      filename: 'convergence_analysis_report.pdf',
      content: '<h1>Model Convergence Analysis Report</h1><p>This is a placeholder for the convergence analysis report.</p>'
    },
    {
      filename: 'performance_metrics.html',
      content: '<h1>Performance Metrics Dashboard</h1><p>This is a placeholder for the performance metrics dashboard.</p>'
    },
    {
      filename: 'communication_protocols.pdf',
      content: '<h1>Communication Protocols Optimization</h1><p>This is a placeholder for the communication protocols documentation.</p>'
    },
    {
      filename: 'privacy_budget_report.html',
      content: '<h1>Privacy Budget Analysis Report</h1><p>This is a placeholder for the privacy budget analysis report.</p>'
    }
  ];

  placeholders.forEach(file => {
    const filePath = path.join(filesDir, file.filename);
    // Only create if it doesn't exist
    if (!fs.existsSync(filePath)) {
      fs.writeFileSync(filePath, file.content);
      console.log(`Created placeholder file: ${file.filename}`);
    }
  });
};

// Create the assets/images directory if it doesn't exist
const imagesDir = path.join(currentDir, 'assets', 'images');
if (!fs.existsSync(imagesDir)) {
  fs.mkdirSync(imagesDir, { recursive: true });
}

// Root route
app.get('/', (req, res) => {
  res.sendFile(path.join(currentDir, 'index.html'));
});

// Start the server
app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}/`);
  console.log(`Current directory: ${currentDir}`);
  
  // Create placeholder files for demos
  createPlaceholderFiles();
}); 