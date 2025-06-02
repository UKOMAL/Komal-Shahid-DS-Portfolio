/**
 * Colorful Canvas AI Art Studio - Web Demo
 * Interactive frontend functionality
 */

// Constants and configuration
const API_ENDPOINT = '../api/process.php'; // Would connect to backend in production
const SAMPLE_IMAGES_PATH = '../samples/';
const EFFECTS = {
    'shadow_box': {
        name: 'Shadow Box Illusion',
        instructions: 'Look directly at the image from the front at eye level for the best effect. The illusion works best from a distance of 2-3 feet.',
    },
    'screen_pop': {
        name: 'Screen Pop Effect',
        instructions: 'View the image at approximately a 45-degree angle from the screen. You may need to adjust your viewing position slightly to see the optimal pop-out effect.',
    },
    'seoul_corner': {
        name: 'Seoul Corner Projection',
        instructions: 'For best results, print the image and place it in a corner where two walls meet. View from about 30 degrees to one side at a distance of 5-6 feet.',
    }
};

// DOM Elements
const uploadBox = document.getElementById('upload-box');
const imageUpload = document.getElementById('image-upload');
const effectsContainer = document.getElementById('effects-container');
const previewContainer = document.getElementById('preview-container');
const originalContainer = document.getElementById('original-container');
const resultContainer = document.getElementById('result-container');
const viewingInstructions = document.getElementById('viewing-instructions');
const downloadBtn = document.getElementById('download-btn');

// State variables
let currentImage = null;
let selectedEffect = null;
let resultImage = null;

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    setupUploadListeners();
    setupSampleImages();
    setupEffectSelection();
    setupDownloadButton();
    
    // For demo purposes, we'll use sample images
    loadSampleImages();
});

// Setup Functions
function setupUploadListeners() {
    uploadBox.addEventListener('click', () => {
        imageUpload.click();
    });
    
    imageUpload.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleImageUpload(e.target.files[0]);
        }
    });
    
    uploadBox.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadBox.style.borderColor = 'var(--secondary-color)';
    });
    
    uploadBox.addEventListener('dragleave', () => {
        uploadBox.style.borderColor = 'var(--primary-color)';
    });
    
    uploadBox.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadBox.style.borderColor = 'var(--primary-color)';
        
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleImageUpload(e.dataTransfer.files[0]);
        }
    });
}

function setupSampleImages() {
    const samples = document.querySelectorAll('.sample');
    
    samples.forEach(sample => {
        sample.addEventListener('click', () => {
            const imageName = sample.getAttribute('data-image');
            
            // In a real implementation, this would load from a server
            // For demo purposes, we'll simulate loading a sample image
            simulateLoadSampleImage(imageName);
        });
    });
}

function setupEffectSelection() {
    const effects = document.querySelectorAll('.effect');
    
    effects.forEach(effect => {
        effect.addEventListener('click', () => {
            // Remove selected class from all effects
            effects.forEach(e => e.classList.remove('selected'));
            
            // Add selected class to clicked effect
            effect.classList.add('selected');
            
            // Set the selected effect
            selectedEffect = effect.getAttribute('data-effect');
            
            // Process the image with the selected effect
            processImage();
        });
    });
}

function setupDownloadButton() {
    downloadBtn.addEventListener('click', () => {
        if (resultImage) {
            // In a real implementation, this would download the actual processed image
            // For demo purposes, we'll simulate a download
            simulateDownload();
        }
    });
}

// Image Handling Functions
function handleImageUpload(file) {
    const reader = new FileReader();
    
    reader.onload = (e) => {
        currentImage = e.target.result;
        
        // Display the uploaded image
        displayOriginalImage(currentImage);
        
        // Show the effects selection
        effectsContainer.classList.remove('hidden');
        
        // Reset the selected effect
        selectedEffect = null;
        document.querySelectorAll('.effect').forEach(e => e.classList.remove('selected'));
        
        // Hide the preview container
        previewContainer.classList.add('hidden');
    };
    
    reader.readAsDataURL(file);
}

function displayOriginalImage(imageData) {
    originalContainer.innerHTML = '';
    const img = document.createElement('img');
    img.src = imageData;
    originalContainer.appendChild(img);
}

function displayResultImage(imageData) {
    resultContainer.innerHTML = '';
    const img = document.createElement('img');
    img.src = imageData;
    resultContainer.appendChild(img);
    resultImage = imageData;
}

// Image Processing Functions
function processImage() {
    if (!currentImage || !selectedEffect) return;
    
    // Show the preview container
    previewContainer.classList.remove('hidden');
    
    // Update process status steps
    updateProcessStatus(1, 'completed');
    updateProcessStatus(2, 'active');
    
    // In a real implementation, this would call an API to process the image
    // For demo purposes, we'll simulate processing with a delay
    setTimeout(() => {
        updateProcessStatus(2, 'completed');
        updateProcessStatus(3, 'active');
        
        setTimeout(() => {
            updateProcessStatus(3, 'completed');
            updateProcessStatus(4, 'active');
            
            // Display the result image (using original for demo)
            simulateProcessedImage();
            
            // Show viewing instructions
            displayViewingInstructions();
            
            setTimeout(() => {
                updateProcessStatus(4, 'completed');
            }, 500);
        }, 1000);
    }, 1500);
}

function updateProcessStatus(step, status) {
    const stepElement = document.querySelector(`.process-status .step[data-step="${step}"]`);
    
    if (status === 'active') {
        stepElement.classList.add('active');
        stepElement.classList.remove('completed');
    } else if (status === 'completed') {
        stepElement.classList.remove('active');
        stepElement.classList.add('completed');
    } else {
        stepElement.classList.remove('active', 'completed');
    }
}

function displayViewingInstructions() {
    const instructionDetails = viewingInstructions.querySelector('.instruction-details');
    instructionDetails.textContent = EFFECTS[selectedEffect].instructions;
}

// Simulation Functions for Demo Purposes
function simulateLoadSampleImage(imageName) {
    // In a real implementation, this would load from a server
    // For demo, we'll use the currentImage as a placeholder
    currentImage = `sample_placeholder_${imageName}`;
    
    // Display a placeholder image
    displayOriginalImage('https://via.placeholder.com/400x300/3498db/ffffff?text=Sample+Image');
    
    // Show the effects selection
    effectsContainer.classList.remove('hidden');
    
    // Reset the selected effect
    selectedEffect = null;
    document.querySelectorAll('.effect').forEach(e => e.classList.remove('selected'));
    
    // Hide the preview container
    previewContainer.classList.add('hidden');
}

function simulateProcessedImage() {
    // Simulate different effects with placeholder images
    let effectColor;
    
    switch(selectedEffect) {
        case 'shadow_box':
            effectColor = '2ecc71';
            break;
        case 'screen_pop':
            effectColor = 'e74c3c';
            break;
        case 'seoul_corner':
            effectColor = '9b59b6';
            break;
        default:
            effectColor = '3498db';
    }
    
    // Display a placeholder processed image
    displayResultImage(`https://via.placeholder.com/400x300/${effectColor}/ffffff?text=${EFFECTS[selectedEffect].name}`);
}

function simulateDownload() {
    // Create a temporary link and click it
    const link = document.createElement('a');
    link.href = resultImage;
    link.download = `colorful_canvas_${selectedEffect}_demo.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// For demo, load sample images
function loadSampleImages() {
    // This would normally load real sample thumbnails from the server
    const samples = document.querySelectorAll('.sample');
    
    samples.forEach((sample, index) => {
        // Get the color for this sample
        const colors = ['3498db', 'e74c3c', '2ecc71'];
        const color = colors[index % colors.length];
        
        // Create a placeholder image
        sample.style.backgroundImage = `url(https://via.placeholder.com/150/${color}/ffffff?text=Sample+${index+1})`;
        sample.innerHTML = '';
    });
} 