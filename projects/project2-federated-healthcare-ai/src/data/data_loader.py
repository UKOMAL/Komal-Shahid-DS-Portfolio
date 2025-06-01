import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import requests
import zipfile
import io
import logging
import tqdm

def download_dataset(dataset_name: str, target_dir: str, force_download: bool = False) -> str:
    """
    Download healthcare datasets for federated learning experiments.
    
    Args:
        dataset_name: Name of the dataset to download ('mimic', 'isic', 'ecg', etc.)
        target_dir: Directory to save the downloaded dataset
        force_download: Whether to download even if files already exist
        
    Returns:
        Path to the downloaded dataset
    """
    os.makedirs(target_dir, exist_ok=True)
    target_path = Path(target_dir)
    
    # Define dataset sources
    dataset_urls = {
        'mimic': 'https://physionet.org/files/mimiciii-demo/1.4/mimic-iii-clinical-database-demo-1.4.zip',
        'isic': 'https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip',
        'ecg': 'https://physionet.org/files/ptbdb/1.0.0/RECORDS',
        'chexpert': 'https://stanfordmlgroup.github.io/competitions/chexpert/',
        'physionet2019': 'https://physionet.org/content/challenge-2019/1.0.0/'
    }
    
    # Check if dataset exists
    dataset_path = target_path / dataset_name
    if dataset_path.exists() and not force_download:
        logging.info(f"Dataset {dataset_name} already exists at {dataset_path}")
        return str(dataset_path)
    
    # Create dataset directory
    os.makedirs(dataset_path, exist_ok=True)
    
    if dataset_name not in dataset_urls:
        raise ValueError(f"Dataset {dataset_name} is not supported. Available datasets: {list(dataset_urls.keys())}")
    
    url = dataset_urls[dataset_name]
    
    # Handle direct download or instructions
    if url.startswith('http') and url.endswith(('.zip', '.gz', '.tar')):
        logging.info(f"Downloading {dataset_name} dataset from {url}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Get content length for progress bar if available
            total_size = int(response.headers.get('content-length', 0))
            
            # Setup progress bar
            progress_bar = tqdm.tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading {dataset_name}")
            
            # Save the file
            download_path = dataset_path / f"{dataset_name}.zip"
            with open(download_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    progress_bar.update(len(chunk))
            progress_bar.close()
            
            # Extract if it's a zip file
            if download_path.suffix == '.zip':
                logging.info(f"Extracting {download_path}...")
                with zipfile.ZipFile(download_path, 'r') as zip_ref:
                    zip_ref.extractall(dataset_path)
                
            logging.info(f"Dataset downloaded and extracted to {dataset_path}")
            return str(dataset_path)
            
        except Exception as e:
            logging.error(f"Error downloading dataset {dataset_name}: {str(e)}")
            raise
    else:
        # For datasets that require authentication or manual download
        logging.info(f"Please visit {url} to download the {dataset_name} dataset manually.")
        logging.info(f"After downloading, please place the files in: {dataset_path}")
        return str(dataset_path)

def prepare_dataset(dataset_name: str, data_dir: str, output_format: str = 'pytorch') -> Union[Dataset, pd.DataFrame]:
    """
    Prepare and preprocess dataset for federated learning.
    
    Args:
        dataset_name: Name of the dataset to prepare
        data_dir: Directory containing the dataset
        output_format: Format to return the dataset in ('pytorch', 'pandas', 'numpy')
        
    Returns:
        Prepared dataset in the requested format
    """
    data_path = Path(data_dir)
    
    if dataset_name == 'mimic':
        # Process MIMIC dataset
        patients_file = data_path / "PATIENTS.csv"
        admissions_file = data_path / "ADMISSIONS.csv"
        
        if not patients_file.exists() or not admissions_file.exists():
            raise FileNotFoundError(f"MIMIC files not found in {data_path}. Please download the dataset first.")
        
        # Simple preprocessing example (would be more complex in practice)
        patients = pd.read_csv(patients_file)
        admissions = pd.read_csv(admissions_file)
        
        # Join tables
        merged_data = pd.merge(patients, admissions, on='SUBJECT_ID')
        
        # Process for ML (simplified example)
        features = merged_data[['GENDER', 'ADMISSION_TYPE', 'INSURANCE']]
        features = pd.get_dummies(features)
        
        # Use mortality as target (simplified)
        labels = (merged_data['HOSPITAL_EXPIRE_FLAG'] == 1).astype(int)
        
        if output_format == 'pytorch':
            return TabularDataset(features, labels)
        else:
            # Return with labels
            result = features.copy()
            result['target'] = labels
            return result
            
    elif dataset_name == 'isic':
        # Process ISIC skin lesion images
        image_dir = data_path / "ISIC_2019_Training_Input"
        labels_file = data_path / "ISIC_2019_Training_GroundTruth.csv"
        
        if not image_dir.exists():
            raise FileNotFoundError(f"ISIC images not found in {data_path}. Please download the dataset first.")
        
        if output_format == 'pytorch':
            return MedicalImageDataset(str(image_dir), str(labels_file) if labels_file.exists() else None)
        else:
            raise ValueError("Non-PyTorch format not supported for image datasets")
    
    elif dataset_name == 'ecg':
        # Process ECG data
        data_file = data_path / "ptbdb_data.csv"
        
        if not data_file.exists():
            raise FileNotFoundError(f"ECG data not found in {data_path}. Please download the dataset first.")
        
        ecg_data = pd.read_csv(data_file)
        
        if output_format == 'pytorch':
            X = ecg_data.drop('target', axis=1) if 'target' in ecg_data.columns else ecg_data
            y = ecg_data['target'] if 'target' in ecg_data.columns else None
            return TimeseriesDataset(X, targets=y, window_size=250, stride=50)
        else:
            return ecg_data
    
    else:
        raise ValueError(f"Dataset {dataset_name} preparation is not implemented")

class TabularDataset(Dataset):
    """Dataset class for tabular healthcare data (e.g., MIMIC-III)."""
    
    def __init__(self, 
                 data: Union[pd.DataFrame, np.ndarray], 
                 targets: Optional[Union[pd.Series, np.ndarray]] = None,
                 transform=None):
        """
        Initialize TabularDataset.
        
        Args:
            data: Input features as DataFrame or numpy array
            targets: Target values if separate from data
            transform: Optional preprocessing transforms
        """
        if isinstance(data, pd.DataFrame):
            self.data = data.values.astype(np.float32)
        else:
            self.data = data.astype(np.float32)
            
        if targets is not None:
            if isinstance(targets, pd.Series):
                self.targets = targets.values
            else:
                self.targets = targets
        else:
            # Assume last column contains targets
            self.data, self.targets = self.data[:, :-1], self.data[:, -1]
            
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        
        if self.transform:
            x = self.transform(x)
            
        return x, y

class MedicalImageDataset(Dataset):
    """Dataset class for medical imaging data (e.g., ISIC, NIH, etc.)."""
    
    def __init__(self, 
                 image_dir: str,
                 labels_file: Optional[str] = None,
                 transform=None):
        """
        Initialize MedicalImageDataset.
        
        Args:
            image_dir: Directory containing image files
            labels_file: CSV file with image names and labels
            transform: Image transformations
        """
        self.image_dir = Path(image_dir)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Load labels if provided
        if labels_file:
            self.labels_df = pd.read_csv(labels_file)
            self.image_files = self.labels_df['image_name'].values
            self.labels = self.labels_df['label'].values
        else:
            # Scan for all image files
            self.image_files = [f.name for f in self.image_dir.glob('*.jpg') or 
                                self.image_dir.glob('*.png')]
            self.labels = None
            
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_dir / self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        if self.labels is not None:
            return image, self.labels[idx]
        else:
            return image, -1  # Return -1 when no label is available

class TimeseriesDataset(Dataset):
    """Dataset class for time series healthcare data (e.g., ECG, EEG)."""
    
    def __init__(self,
                 data: Union[pd.DataFrame, np.ndarray],
                 window_size: int = 100,
                 stride: int = 1,
                 targets: Optional[Union[pd.Series, np.ndarray]] = None,
                 transform=None):
        """
        Initialize TimeseriesDataset with sliding window approach.
        
        Args:
            data: Input time series data
            window_size: Size of each sliding window
            stride: Step size between consecutive windows
            targets: Target values for each window
            transform: Optional preprocessing transforms
        """
        if isinstance(data, pd.DataFrame):
            self.data = data.values.astype(np.float32)
        else:
            self.data = data.astype(np.float32)
            
        self.window_size = window_size
        self.stride = stride
        self.transform = transform
        
        # Create sliding windows
        num_windows = (len(self.data) - window_size) // stride + 1
        self.windows = []
        
        for i in range(num_windows):
            start_idx = i * stride
            end_idx = start_idx + window_size
            self.windows.append(self.data[start_idx:end_idx])
        
        # Handle targets
        if targets is not None:
            if isinstance(targets, pd.Series):
                targets = targets.values
                
            # Create target for each window (using the target from the last timestamp)
            self.targets = [targets[i * stride + window_size - 1] for i in range(num_windows)]
        else:
            self.targets = [-1] * num_windows
            
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        x = self.windows[idx]
        y = self.targets[idx]
        
        if self.transform:
            x = self.transform(x)
            
        return x, y

def create_dataloader(dataset: Dataset, 
                     batch_size: int = 32, 
                     shuffle: bool = True, 
                     num_workers: int = 4) -> DataLoader:
    """
    Create a PyTorch DataLoader from a Dataset.
    
    Args:
        dataset: PyTorch Dataset object
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle data
        num_workers: Number of subprocesses for data loading
        
    Returns:
        PyTorch DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

def load_federated_data(data_dir: str, 
                       modality: str = 'tabular',
                       num_clients: int = 5,
                       non_iid_factor: float = 0.5) -> List[Dataset]:
    """
    Load and partition data for federated learning simulation.
    
    Args:
        data_dir: Directory containing the healthcare datasets
        modality: Type of healthcare data (tabular, image, timeseries)
        num_clients: Number of federated clients to simulate
        non_iid_factor: Controls how non-IID the data partitioning is
        
    Returns:
        List of datasets, one for each client
    """
    data_path = Path(data_dir)
    client_datasets = []
    
    # Load appropriate dataset based on modality
    if modality == 'tabular':
        # For tabular data like MIMIC-III
        data_file = data_path / "mimic_processed.csv"
        if data_file.exists():
            df = pd.read_csv(data_file)
            # Split into features and targets
            X, y = df.iloc[:, :-1], df.iloc[:, -1]
            
            # Create non-IID partitions
            client_datasets = create_non_iid_partitions(
                TabularDataset(X, y),
                num_clients,
                non_iid_factor
            )
        else:
            raise FileNotFoundError(f"Tabular data file not found at {data_file}")
            
    elif modality == 'image':
        # For medical imaging datasets
        image_dir = data_path / "medical_images"
        labels_file = data_path / "image_labels.csv"
        
        if image_dir.exists() and labels_file.exists():
            dataset = MedicalImageDataset(image_dir, labels_file)
            client_datasets = create_non_iid_partitions(
                dataset,
                num_clients,
                non_iid_factor
            )
        else:
            raise FileNotFoundError(f"Image data not found at {image_dir}")
            
    elif modality == 'timeseries':
        # For timeseries data like ECG
        data_file = data_path / "ecg_data.csv"
        if data_file.exists():
            df = pd.read_csv(data_file)
            dataset = TimeseriesDataset(df, window_size=250, stride=50)
            client_datasets = create_non_iid_partitions(
                dataset,
                num_clients,
                non_iid_factor
            )
        else:
            raise FileNotFoundError(f"Timeseries data file not found at {data_file}")
    
    return client_datasets

def create_non_iid_partitions(
    dataset: Union[Dataset, pd.DataFrame], 
    num_clients: int, 
    non_iid_factor: float = 0.5
) -> List[Dataset]:
    """
    Create non-IID data partitions for federated learning clients.
    
    Args:
        dataset: Dataset to partition
        num_clients: Number of clients (partitions) to create
        non_iid_factor: Factor that controls how non-IID the partitions are (0.0 = IID, 1.0 = completely non-IID)
        
    Returns:
        List of datasets, one for each client
    """
    if isinstance(dataset, pd.DataFrame):
        return _create_non_iid_partitions_df(dataset, num_clients, non_iid_factor)
    else:
        return _create_non_iid_partitions_torch(dataset, num_clients, non_iid_factor)

def _create_non_iid_partitions_torch(
    dataset: Dataset, 
    num_clients: int, 
    non_iid_factor: float = 0.5
) -> List[Dataset]:
    """Create non-IID partitions for PyTorch datasets."""
    # Get targets from dataset
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
    elif hasattr(dataset, 'tensors') and len(dataset.tensors) > 1:
        # TensorDataset with target as second tensor
        targets = dataset.tensors[1]
    else:
        # If we can't easily get targets, fall back to random partitioning
        return _create_random_partitions_torch(dataset, num_clients)
    
    # Convert targets to numpy array
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()
    elif isinstance(targets, list):
        targets = np.array(targets)
    
    # Get unique classes
    unique_classes = np.unique(targets)
    num_classes = len(unique_classes)
    
    # Create class indices
    class_indices = {cls: np.where(targets == cls)[0] for cls in unique_classes}
    
    # Calculate samples per client
    total_samples = len(dataset)
    samples_per_client = total_samples // num_clients
    
    # Create partitions
    partitions = []
    
    for client_idx in range(num_clients):
        # Calculate number of samples per class for this client
        # Some classes will be over-represented based on non_iid_factor
        
        # Calculate base probability for each class
        base_probs = np.ones(num_classes) / num_classes
        
        # If non_iid_factor > 0, skew the distribution
        if non_iid_factor > 0:
            # Create a skewed distribution for this client
            # Focus on a subset of classes based on the client index
            preferred_classes = np.roll(np.arange(num_classes), client_idx)[:max(1, int(num_classes * (1 - non_iid_factor)))]
            
            # Create probabilities
            probs = np.ones(num_classes) * (1 - non_iid_factor) / num_classes
            extra_weight = non_iid_factor / len(preferred_classes)
            probs[preferred_classes] += extra_weight
        else:
            probs = base_probs
        
        # Sample class indices based on probabilities
        client_indices = []
        samples_to_take = min(samples_per_client, total_samples - len(client_indices))
        
        while len(client_indices) < samples_to_take:
            # Draw classes according to probability
            class_idx = np.random.choice(num_classes, p=probs)
            
            # Get available indices for this class
            available_indices = class_indices[unique_classes[class_idx]]
            
            if len(available_indices) > 0:
                # Take one sample
                sample_idx = np.random.choice(available_indices)
                client_indices.append(sample_idx)
                
                # Remove the index from available indices
                mask = available_indices != sample_idx
                class_indices[unique_classes[class_idx]] = available_indices[mask]
        
        # Create a Subset dataset for this client
        partition = torch.utils.data.Subset(dataset, client_indices)
        partitions.append(partition)
    
    return partitions

def _create_random_partitions_torch(
    dataset: Dataset, 
    num_clients: int
) -> List[Dataset]:
    """Create random partitions for PyTorch datasets."""
    # Calculate samples per client
    total_samples = len(dataset)
    samples_per_client = total_samples // num_clients
    
    # Create random permutation of indices
    indices = torch.randperm(total_samples).tolist()
    
    # Create partitions
    partitions = []
    
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = min((i + 1) * samples_per_client, total_samples)
        client_indices = indices[start_idx:end_idx]
        
        partition = torch.utils.data.Subset(dataset, client_indices)
        partitions.append(partition)
    
    return partitions

def _create_non_iid_partitions_df(
    df: pd.DataFrame, 
    num_clients: int, 
    non_iid_factor: float = 0.5
) -> List[pd.DataFrame]:
    """Create non-IID partitions for pandas DataFrames."""
    # Identify target column (assume it's the last column)
    if 'target' in df.columns:
        target_col = 'target'
    else:
        target_col = df.columns[-1]
    
    targets = df[target_col].values
    
    # Get unique classes
    unique_classes = np.unique(targets)
    num_classes = len(unique_classes)
    
    # Create class indices
    class_indices = {cls: np.where(targets == cls)[0] for cls in unique_classes}
    
    # Calculate samples per client
    total_samples = len(df)
    samples_per_client = total_samples // num_clients
    
    # Create partitions
    partitions = []
    
    for client_idx in range(num_clients):
        # Calculate base probability for each class
        base_probs = np.ones(num_classes) / num_classes
        
        # If non_iid_factor > 0, skew the distribution
        if non_iid_factor > 0:
            # Create a skewed distribution for this client
            preferred_classes = np.roll(np.arange(num_classes), client_idx)[:max(1, int(num_classes * (1 - non_iid_factor)))]
            
            # Create probabilities
            probs = np.ones(num_classes) * (1 - non_iid_factor) / num_classes
            extra_weight = non_iid_factor / len(preferred_classes)
            probs[preferred_classes] += extra_weight
        else:
            probs = base_probs
        
        # Sample class indices based on probabilities
        client_indices = []
        samples_to_take = min(samples_per_client, total_samples - len(client_indices))
        
        while len(client_indices) < samples_to_take:
            # Draw classes according to probability
            class_idx = np.random.choice(num_classes, p=probs)
            
            # Get available indices for this class
            available_indices = class_indices[unique_classes[class_idx]]
            
            if len(available_indices) > 0:
                # Take one sample
                sample_idx = np.random.choice(available_indices)
                client_indices.append(sample_idx)
                
                # Remove the index from available indices
                mask = available_indices != sample_idx
                class_indices[unique_classes[class_idx]] = available_indices[mask]
        
        # Create a DataFrame for this client
        partition = df.iloc[client_indices].copy()
        partitions.append(partition)
    
    return partitions 