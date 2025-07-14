import os
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, NoReturn
import numpy as np
import torch
import flwr as fl
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FederatedClient(fl.client.NumPyClient):
    """
    Federated Learning Client for healthcare applications.
    Handles local training and communication with the federated server.
    """
    
    def __init__(self, 
                 model: torch.nn.Module,
                 trainset: torch.utils.data.Dataset,
                 testset: torch.utils.data.Dataset,
                 client_id: str,
                 dp_mechanism: Optional[str] = None,
                 dp_epsilon: float = 1.0,
                 dp_delta: float = 1e-5):
        """
        Initialize the Federated Client.
        
        Args:
            model: PyTorch model to train
            trainset: Training dataset
            testset: Testing dataset
            client_id: Unique identifier for this client
            dp_mechanism: Differential privacy mechanism (None, 'gaussian', 'laplace')
            dp_epsilon: Differential privacy epsilon parameter
            dp_delta: Differential privacy delta parameter
        """
        self.model = model
        self.trainset = trainset
        self.testset = testset
        self.client_id = client_id
        self.dp_mechanism = dp_mechanism
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta
        
        # Set device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Configure privacy mechanism if requested
        if dp_mechanism:
            from ..privacy.differential_privacy import GaussianMechanism, LaplaceMechanism
            
            if dp_mechanism.lower() == 'gaussian':
                self.privacy_mechanism = GaussianMechanism(
                    epsilon=dp_epsilon, delta=dp_delta, sensitivity=1.0
                )
            elif dp_mechanism.lower() == 'laplace':
                self.privacy_mechanism = LaplaceMechanism(
                    epsilon=dp_epsilon, sensitivity=1.0
                )
            else:
                logger.warning(f"Unknown privacy mechanism: {dp_mechanism}")
                self.privacy_mechanism = None
        else:
            self.privacy_mechanism = None
        
        logger.info(f"Initialized client {client_id} on {self.device}")
        if self.privacy_mechanism:
            logger.info(f"Privacy mechanism: {dp_mechanism}, epsilon: {dp_epsilon}")
    
    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """Return current model parameters as a list of NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from a list of NumPy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.Tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """
        Perform local training on the provided parameters.
        
        Args:
            parameters: Initial parameter values
            config: Training configuration
            
        Returns:
            Tuple containing updated parameters, number of training examples, and metrics
        """
        # Get training configuration
        epochs = config.get("epochs", 1)
        batch_size = config.get("batch_size", 32)
        learning_rate = config.get("learning_rate", 0.01)
        current_round = config.get("round", 0)
        
        # Set model parameters
        self.set_parameters(parameters)
        
        # Create data loader
        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if self.device.type == "cuda" else False
        )
        
        # Define loss function and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        
        # Train the model
        self.model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        train_examples = 0
        
        for _ in range(epochs):
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Zero the gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                
                # Apply gradient clipping to bound sensitivity
                if self.privacy_mechanism:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Add noise to model parameters if using differential privacy
                if self.privacy_mechanism:
                    with torch.no_grad():
                        for param in self.model.parameters():
                            noise = self.privacy_mechanism.add_noise(param.data)
                            if isinstance(noise, np.ndarray):
                                param.data = torch.tensor(noise, device=self.device)
                            else:
                                param.data = noise
                
                # Track metrics
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                train_accuracy += (predicted == targets).sum().item()
                train_examples += inputs.size(0)
        
        # Calculate average metrics
        train_loss /= train_examples
        train_accuracy /= train_examples
        
        # Log training results
        logger.info(f"Client {self.client_id} completed training for round {current_round}")
        logger.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        
        # Get updated parameters
        updated_parameters = self.get_parameters(config)
        
        metrics = {
            "train_loss": float(train_loss),
            "train_accuracy": float(train_accuracy),
        }
        
        return updated_parameters, train_examples, metrics
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[float, int, Dict[str, Any]]:
        """
        Evaluate the provided parameters.
        
        Args:
            parameters: Model parameters to evaluate
            config: Evaluation configuration
            
        Returns:
            Tuple containing loss, number of test examples, and metrics
        """
        # Get evaluation configuration
        batch_size = config.get("batch_size", 32)
        current_round = config.get("round", 0)
        
        # Set model parameters
        self.set_parameters(parameters)
        
        # Create data loader
        testloader = torch.utils.data.DataLoader(
            self.testset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.device.type == "cuda" else False
        )
        
        # Define loss function
        criterion = torch.nn.CrossEntropyLoss()
        
        # Evaluate the model
        self.model.eval()
        test_loss = 0.0
        test_accuracy = 0.0
        test_examples = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                # Track metrics
                test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                test_accuracy += (predicted == targets).sum().item()
                test_examples += inputs.size(0)
        
        # Calculate average metrics
        test_loss /= test_examples
        test_accuracy /= test_examples
        
        # Log evaluation results
        logger.info(f"Client {self.client_id} completed evaluation for round {current_round}")
        logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        
        metrics = {
            "test_loss": float(test_loss),
            "test_accuracy": float(test_accuracy),
        }
        
        return float(test_loss), test_examples, metrics

def load_model_and_data(model_name: str, 
                        data_dir: str, 
                        client_id: str):
    """
    Load the model and data for a client.
    
    Args:
        model_name: Name of the model to load
        data_dir: Directory containing datasets
        client_id: Client identifier
        
    Returns:
        Tuple containing model, training dataset, and test dataset
    """
    # Import the model
    from ..models.cnn_models import get_model
    
    # Load the model
    model = get_model(model_name)
    
    # For demonstration, we're using datasets that should be located in a specific directory
    # In a real implementation, you would load pre-partitioned data for each client
    from ..data.data_loader import TabularDataset, MedicalImageDataset, TimeseriesDataset
    
    # Get client-specific data directory
    client_data_dir = os.path.join(data_dir, f"client_{client_id}")
    
    # Load datasets
    # This is just an example - you would need to implement actual data loading logic
    # based on your specific dataset formats
    if not os.path.exists(client_data_dir):
        raise ValueError(f"Data directory for client {client_id} not found: {client_data_dir}")
        
    # For this example, we're assuming medical images
    train_dir = os.path.join(client_data_dir, "train")
    test_dir = os.path.join(client_data_dir, "test")
    train_labels = os.path.join(client_data_dir, "train_labels.csv")
    test_labels = os.path.join(client_data_dir, "test_labels.csv")
    
    trainset = MedicalImageDataset(train_dir, train_labels)
    testset = MedicalImageDataset(test_dir, test_labels)
    
    return model, trainset, testset

def start_client(client_config: Dict[str, Any]) -> NoReturn:
    """
    Start a federated learning client.
    
    Args:
        client_config: Client configuration
        
    Returns:
        This function does not return (it runs the client indefinitely)
    """
    # Extract configuration
    model_name = client_config.get("model", "simple_cnn")
    data_dir = client_config.get("data_dir", "./data")
    client_id = client_config.get("client_id", "1")
    server_address = client_config.get("server_address", "[::]:8080")
    dp_mechanism = client_config.get("dp_mechanism", None)
    dp_epsilon = client_config.get("dp_epsilon", 1.0)
    dp_delta = client_config.get("dp_delta", 1e-5)
    
    # Load model and data
    model, trainset, testset = load_model_and_data(model_name, data_dir, client_id)
    
    # Initialize the client
    client = FederatedClient(
        model=model,
        trainset=trainset,
        testset=testset,
        client_id=client_id,
        dp_mechanism=dp_mechanism,
        dp_epsilon=dp_epsilon,
        dp_delta=dp_delta
    )
    
    # Start the client
    logger.info(f"Starting client {client_id} connecting to server at {server_address}")
    fl.client.start_numpy_client(server_address, client=client)

if __name__ == "__main__":
    # Example usage
    client_config = {
        "model": "simple_cnn",
        "data_dir": "./data",
        "client_id": "1",
        "server_address": "[::]:8080",
        "dp_mechanism": "gaussian",
        "dp_epsilon": 0.5,
        "dp_delta": 1e-5
    }
    
    start_client(client_config) 