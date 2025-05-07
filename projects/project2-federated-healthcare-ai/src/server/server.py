import os
import logging
import numpy as np
import torch
import flwr as fl
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FederatedServer:
    """
    Federated Learning Server for healthcare applications.
    This server coordinates the federated learning process across multiple clients.
    """
    
    def __init__(self, 
                 model_fn,
                 num_rounds: int = 10,
                 min_clients: int = 2,
                 min_available_clients: int = 2,
                 eval_fn=None,
                 model_dir: str = "./models",
                 privacy_setting: Dict[str, Any] = None,
                 strategy: str = "fedavg"):
        """
        Initialize the Federated Server.
        
        Args:
            model_fn: Function that returns the initial global model
            num_rounds: Number of federated learning rounds
            min_clients: Minimum number of clients for training
            min_available_clients: Minimum number of available clients required
            eval_fn: Optional function for server-side evaluation
            model_dir: Directory to save model checkpoints
            privacy_setting: Privacy configuration
            strategy: Federated learning aggregation strategy
        """
        self.model_fn = model_fn
        self.num_rounds = num_rounds
        self.min_clients = min_clients
        self.min_available_clients = min_available_clients
        self.eval_fn = eval_fn
        self.model_dir = Path(model_dir)
        self.privacy_setting = privacy_setting or {}
        self.strategy_name = strategy.lower()
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize the global model
        self.global_model = self.model_fn()
        
        # Configure the federated learning strategy
        self.strategy = self._create_strategy()
        
        logger.info(f"Initialized federated server with {strategy} strategy")
        logger.info(f"Server will run for {num_rounds} rounds with minimum {min_clients} clients")
    
    def _create_strategy(self) -> fl.server.strategy.Strategy:
        """Create the appropriate federated learning strategy based on configuration."""
        
        # Define the model parameters function to get weights
        def get_model_parameters(model):
            return [val.cpu().numpy() for _, val in model.state_dict().items()]
        
        # Set initial parameters
        initial_parameters = fl.common.ndarrays_to_parameters(
            get_model_parameters(self.global_model)
        )
        
        # Define the evaluation function if provided
        def evaluate(server_round, parameters, config):
            if self.eval_fn is None:
                return None
            
            # Update the global model with the current parameters
            fl.common.parameters_to_ndarrays(parameters)
            # Perform evaluation
            metrics = self.eval_fn(self.global_model)
            
            return float(metrics["loss"]), metrics
        
        # Configure the strategy based on the chosen algorithm
        if self.strategy_name == "fedavg":
            return fl.server.strategy.FedAvg(
                fraction_fit=0.5,  # Sample 50% of available clients for training
                fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
                min_fit_clients=self.min_clients,
                min_evaluate_clients=self.min_clients,
                min_available_clients=self.min_available_clients,
                evaluate_fn=evaluate,
                initial_parameters=initial_parameters,
            )
        elif self.strategy_name == "fedprox":
            # FedProx adds a proximal term to client optimization
            return fl.server.strategy.FedProx(
                fraction_fit=0.5,
                fraction_evaluate=0.5,
                min_fit_clients=self.min_clients,
                min_evaluate_clients=self.min_clients,
                min_available_clients=self.min_available_clients,
                evaluate_fn=evaluate,
                initial_parameters=initial_parameters,
                mu=0.1,  # Proximal term weight
            )
        elif self.strategy_name == "fedopt":
            # FedOpt uses server optimizer (Adam)
            return fl.server.strategy.FedOpt(
                fraction_fit=0.5,
                fraction_evaluate=0.5,
                min_fit_clients=self.min_clients,
                min_evaluate_clients=self.min_clients,
                min_available_clients=self.min_available_clients,
                evaluate_fn=evaluate,
                initial_parameters=initial_parameters,
                server_optimizer_fn=lambda params: torch.optim.Adam(params, lr=0.001),
            )
        else:
            logger.warning(f"Unknown strategy {self.strategy_name}, defaulting to FedAvg")
            return fl.server.strategy.FedAvg(
                fraction_fit=0.5,
                fraction_evaluate=0.5,
                min_fit_clients=self.min_clients,
                min_evaluate_clients=self.min_clients,
                min_available_clients=self.min_available_clients,
                evaluate_fn=evaluate,
                initial_parameters=initial_parameters,
            )
    
    def save_model(self, round_num: int) -> None:
        """
        Save the global model after a federated round.
        
        Args:
            round_num: Current round number
        """
        model_path = self.model_dir / f"global_model_round_{round_num}.pt"
        torch.save(self.global_model.state_dict(), model_path)
        logger.info(f"Saved global model checkpoint to {model_path}")
    
    def start_server(self, server_address: str = "[::]:8080") -> None:
        """
        Start the federated learning server.
        
        Args:
            server_address: Address to bind the server to
        """
        logger.info(f"Starting federated server at {server_address}")
        
        # Define the fit configuration function
        def fit_config(server_round: int) -> Dict[str, Any]:
            """Return training configuration for clients."""
            config = {
                "round": server_round,
                "epochs": 1,  # Local epochs per round
                "batch_size": 32,
                "learning_rate": 0.01 * (0.99 ** server_round),  # Decay learning rate
            }
            
            # Add privacy configuration if provided
            if self.privacy_setting:
                config.update(self.privacy_setting)
                
            return config
        
        # Define the evaluate configuration function
        def evaluate_config(server_round: int) -> Dict[str, Any]:
            """Return evaluation configuration for clients."""
            return {
                "round": server_round,
                "batch_size": 64,
            }
        
        # Add configuration functions to the strategy
        self.strategy.on_fit_config_fn = fit_config
        self.strategy.on_evaluate_config_fn = evaluate_config
        
        # Start the server
        fl.server.start_server(
            server_address=server_address,
            config=fl.server.ServerConfig(num_rounds=self.num_rounds),
            strategy=self.strategy,
        )

def run_federated_server(config: Dict[str, Any]) -> None:
    """
    Start and run a federated learning server with the given configuration.
    
    Args:
        config: Server configuration
    """
    # Extract configuration
    model_name = config.get("model", "simple_cnn")
    num_rounds = config.get("num_rounds", 10)
    min_clients = config.get("min_clients", 2)
    strategy = config.get("strategy", "fedavg")
    model_dir = config.get("model_dir", "./models")
    privacy_setting = config.get("privacy", None)
    server_address = config.get("server_address", "[::]:8080")
    
    # Import the model function based on the model name
    if model_name == "simple_cnn":
        from ..models.cnn_models import get_model
        model_fn = lambda: get_model("simple_cnn")
    elif model_name == "medical_resnet":
        from ..models.cnn_models import get_model
        model_fn = lambda: get_model("medical_resnet")
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Initialize the server
    server = FederatedServer(
        model_fn=model_fn,
        num_rounds=num_rounds,
        min_clients=min_clients,
        min_available_clients=min_clients,
        model_dir=model_dir,
        privacy_setting=privacy_setting,
        strategy=strategy
    )
    
    # Start the server
    server.start_server(server_address=server_address)

if __name__ == "__main__":
    # Example usage
    config = {
        "model": "simple_cnn",
        "num_rounds": 5,
        "min_clients": 3,
        "strategy": "fedavg",
        "model_dir": "./models",
        "privacy": {
            "mechanism": "gaussian",
            "epsilon": 1.0,
            "delta": 1e-5
        },
        "server_address": "[::]:8080"
    }
    
    run_federated_server(config) 