import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """Basic CNN architecture for medical imaging classification tasks."""
    
    def __init__(self, in_channels=3, num_classes=2):
        """
        Initialize the CNN model.
        
        Args:
            in_channels: Number of input channels (1 for grayscale, 3 for RGB)
            num_classes: Number of output classes
        """
        super(SimpleCNN, self).__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 512)  # Assuming input size of 64x64
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Convolutional blocks with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the feature maps
        x = x.view(-1, 128 * 8 * 8)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class MedicalResNet(nn.Module):
    """Simplified ResNet-style architecture for medical imaging."""
    
    def __init__(self, in_channels=3, num_classes=2):
        super(MedicalResNet, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        
        # First residual block with potential downsampling
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        
        layers.append(ResidualBlock(in_channels, out_channels, stride, downsample))
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial processing
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Global average pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

class ResidualBlock(nn.Module):
    """Basic residual block for ResNet architecture."""
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        
        # First convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        # Forward pass through first convolution
        out = F.relu(self.bn1(self.conv1(x)))
        
        # Forward pass through second convolution
        out = self.bn2(self.conv2(out))
        
        # Apply shortcut connection
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Add residual connection and apply ReLU
        out += identity
        out = F.relu(out)
        
        return out

def get_model(model_name, in_channels=3, num_classes=2):
    """
    Factory function to get model by name.
    
    Args:
        model_name: Name of the model (simple_cnn or medical_resnet)
        in_channels: Number of input channels
        num_classes: Number of output classes
    
    Returns:
        PyTorch model instance
    """
    if model_name == 'simple_cnn':
        return SimpleCNN(in_channels=in_channels, num_classes=num_classes)
    elif model_name == 'medical_resnet':
        return MedicalResNet(in_channels=in_channels, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}") 