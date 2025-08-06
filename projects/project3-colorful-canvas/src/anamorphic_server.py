#!/usr/bin/env python3
"""
Simple Flask Server for Anamorphic Billboard Generation
Provides web API endpoints for the ColorfulCanvas system

Author: Komal Shahid
Course: DSC680 - Applied Data Science
"""

import os
import sys
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import base64
import traceback

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import our API
from web_anamorphic_api import WebAnamorphicAPI

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the API
anamorphic_api = WebAnamorphicAPI()

# Simple HTML template for testing
TEST_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>ColorfulCanvas Anamorphic Generator</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #1a1a1a; color: white; }
        .container { max-width: 800px; margin: 0 auto; }
        .upload-area { 
            border: 3px dashed #667eea; 
            padding: 40px; 
            text-align: center; 
            margin: 20px 0;
            border-radius: 10px;
            background: #2a2a2a;
        }
        .upload-area:hover { background: #3a3a3a; }
        .result-area { margin: 20px 0; }
        .step { 
            background: #333; 
            padding: 15px; 
            margin: 10px 0; 
            border-radius: 5px;
            border-left: 4px solid #667eea;
        }
        .step.completed { border-left-color: #10b981; }
        .step.processing { border-left-color: #f59e0b; }
        .result-image { max-width: 100%; border-radius: 10px; margin: 20px 0; }
        .error { color: #ef4444; background: #2a1a1a; padding: 15px; border-radius: 5px; }
        .success { color: #10b981; }
        button { 
            background: #667eea; 
            color: white; 
            border: none; 
            padding: 12px 24px; 
            border-radius: 5px; 
            cursor: pointer;
            font-size: 16px;
        }
        button:hover { background: #5a67d8; }
        #loading { display: none; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎨 ColorfulCanvas Anamorphic Generator</h1>
        <p>Upload an image to create a stunning anamorphic 3D billboard effect!</p>
        
        <div class="upload-area" id="uploadArea">
            <p>📁 Click here or drag & drop an image</p>
            <input type="file" id="fileInput" accept="image/*" style="display: none;">
            <button onclick="document.getElementById('fileInput').click()">Choose Image</button>
        </div>
        
        <div id="loading">
            <h3>🚀 Processing your image...</h3>
            <div id="steps"></div>
        </div>
        
        <div id="results" class="result-area"></div>
    </div>

    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const loading = document.getElementById('loading');
        const results = document.getElementById('results');
        const steps = document.getElementById('steps');

        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.background = '#3a3a3a';
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.style.background = '#2a2a2a';
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.style.background = '#2a2a2a';
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                processImage(files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                processImage(e.target.files[0]);
            }
        });

        function processImage(file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                const imageData = e.target.result;
                
                // Show loading
                loading.style.display = 'block';
                results.innerHTML = '';
                steps.innerHTML = '<div class="step processing">Uploading image...</div>';
                
                // Send to server
                fetch('/process_image', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        image_data: imageData
                    })
                })
                .then(response => response.json())
                .then(data => {
                    loading.style.display = 'none';
                    
                    if (data.status === 'success') {
                        showResults(data);
                    } else {
                        showError(data.message);
                    }
                })
                .catch(error => {
                    loading.style.display = 'none';
                    showError('Network error: ' + error.message);
                });
            };
            reader.readAsDataURL(file);
        }

        function showResults(data) {
            let html = '<h3 class="success">✅ Anamorphic Billboard Generated!</h3>';
            
            // Show processing steps
            data.steps.forEach(step => {
                html += `<div class="step completed">
                    <strong>${step.name}:</strong> ${step.description}
                </div>`;
            });
            
            // Show final result
            if (data.final_result) {
                html += '<h4>🎭 Final Anamorphic Billboard:</h4>';
                html += `<img src="${data.final_result}" class="result-image" alt="Anamorphic Billboard">`;
                html += '<p><em>This billboard creates a 3D illusion when viewed from the correct angle!</em></p>';
            }
            
            results.innerHTML = html;
        }

        function showError(message) {
            results.innerHTML = `<div class="error">❌ Error: ${message}</div>`;
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Serve the test interface"""
    return render_template_string(TEST_TEMPLATE)

@app.route('/process_image', methods=['POST'])
def process_image():
    """Process an uploaded image and return anamorphic result"""
    try:
        data = request.get_json()
        image_data = data.get('image_data')
        
        if not image_data:
            return jsonify({
                'status': 'error',
                'message': 'No image data provided'
            }), 400
        
        # Process the image
        result = anamorphic_api.process_image_base64(image_data)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Server error: {str(e)}',
            'error_details': traceback.format_exc()
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'ColorfulCanvas Anamorphic API is running',
        'systems': {
            'modern_system': anamorphic_api.modern_system is not None,
            'ultimate_system': anamorphic_api.ultimate_system is not None
        }
    })

if __name__ == '__main__':
    print("🚀 Starting ColorfulCanvas Anamorphic Server...")
    print("📱 Open http://localhost:5000 to test the system")
    print("🔗 API endpoint: http://localhost:5000/process_image")
    print("❤️  Health check: http://localhost:5000/health")
    
    app.run(host='0.0.0.0', port=5000, debug=True)