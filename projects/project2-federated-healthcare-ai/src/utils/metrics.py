import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
import torch
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    confusion_matrix,
    precision_recall_curve,
    average_precision_score
)
import matplotlib.pyplot as plt
from pathlib import Path

def calculate_classification_metrics(
    y_true: Union[np.ndarray, List],
    y_pred: Union[np.ndarray, List],
    y_score: Optional[Union[np.ndarray, List]] = None,
    average: str = 'weighted'
) -> Dict[str, float]:
    """
    Calculate various classification metrics for healthcare tasks.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_score: Predicted probabilities or scores (for ROC AUC)
        average: Averaging strategy for multi-class metrics
        
    Returns:
        Dictionary containing calculated metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate sensitivity and specificity for binary classification
    if len(np.unique(y_true)) == 2:
        tn, fp, fn, tp = cm.ravel()
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # ROC AUC if scores are provided
        if y_score is not None:
            # Ensure y_score is for the positive class for binary classification
            if isinstance(y_score, list) or (isinstance(y_score, np.ndarray) and y_score.ndim > 1):
                # If we have probabilities for each class, select the positive class
                if isinstance(y_score, list):
                    y_score = np.array(y_score)
                y_score = y_score[:, 1]
                
            metrics['roc_auc'] = roc_auc_score(y_true, y_score)
            metrics['avg_precision'] = average_precision_score(y_true, y_score)
    
    return metrics

def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    multilabel: bool = False
) -> Dict[str, float]:
    """
    Evaluate a PyTorch model on healthcare data.
    
    Args:
        model: PyTorch model to evaluate
        dataloader: DataLoader containing evaluation data
        device: Device to run evaluation on
        threshold: Threshold for binary classification
        multilabel: Whether this is a multilabel classification task
        
    Returns:
        Dictionary containing calculated metrics
    """
    model.eval()
    y_true_list = []
    y_pred_list = []
    y_score_list = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # Handle different output formats
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Some models return (outputs, features)
            
            # Convert outputs to predictions based on task type
            if multilabel:
                # Multi-label classification
                y_pred = (torch.sigmoid(outputs) > threshold).float()
                y_score = torch.sigmoid(outputs)
            else:
                # Multi-class or binary classification
                if outputs.shape[1] == 1:  # Binary with single output
                    y_pred = (torch.sigmoid(outputs) > threshold).float()
                    y_score = torch.sigmoid(outputs)
                else:  # Multi-class
                    y_pred = torch.argmax(outputs, dim=1)
                    y_score = torch.softmax(outputs, dim=1)
            
            # Collect results
            y_true_list.append(targets.cpu().numpy())
            y_pred_list.append(y_pred.cpu().numpy())
            y_score_list.append(y_score.cpu().numpy())
    
    # Concatenate batches
    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    y_score = np.concatenate(y_score_list)
    
    # Calculate metrics
    metrics = calculate_classification_metrics(y_true, y_pred, y_score)
    
    return metrics

def plot_confusion_matrix(
    y_true: Union[np.ndarray, List],
    y_pred: Union[np.ndarray, List],
    class_names: List[str],
    output_dir: Optional[str] = None,
    normalize: bool = True,
    title: str = "Confusion Matrix"
) -> None:
    """
    Plot confusion matrix for model evaluation.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        output_dir: Directory to save the plot (if None, plot is displayed)
        normalize: Whether to normalize the confusion matrix
        title: Plot title
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    if output_dir:
        output_path = Path(output_dir) / f"{title.replace(' ', '_').lower()}.png"
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def plot_roc_curve(
    y_true: Union[np.ndarray, List],
    y_score: Union[np.ndarray, List],
    output_dir: Optional[str] = None,
    title: str = "ROC Curve"
) -> None:
    """
    Plot ROC curve for binary classification.
    
    Args:
        y_true: Ground truth labels
        y_score: Predicted probabilities for the positive class
        output_dir: Directory to save the plot (if None, plot is displayed)
        title: Plot title
    """
    from sklearn.metrics import roc_curve, auc
    
    # Ensure binary classification
    if len(np.unique(y_true)) != 2:
        raise ValueError("ROC curve plot is only applicable for binary classification")
    
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    
    if output_dir:
        output_path = Path(output_dir) / f"{title.replace(' ', '_').lower()}.png"
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def plot_precision_recall_curve(
    y_true: Union[np.ndarray, List],
    y_score: Union[np.ndarray, List],
    output_dir: Optional[str] = None,
    title: str = "Precision-Recall Curve"
) -> None:
    """
    Plot precision-recall curve for binary classification.
    
    Args:
        y_true: Ground truth labels
        y_score: Predicted probabilities for the positive class
        output_dir: Directory to save the plot (if None, plot is displayed)
        title: Plot title
    """
    # Compute precision-recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    avg_precision = average_precision_score(y_true, y_score)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(title)
    plt.legend(loc="lower left")
    
    if output_dir:
        output_path = Path(output_dir) / f"{title.replace(' ', '_').lower()}.png"
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def calculate_healthcare_metrics(
    y_true: Union[np.ndarray, List],
    y_pred: Union[np.ndarray, List],
    task_type: str = 'classification'
) -> Dict[str, float]:
    """
    Calculate healthcare-specific metrics based on task type.
    
    Args:
        y_true: Ground truth labels or values
        y_pred: Predicted labels or values
        task_type: Type of task ('classification', 'segmentation', 'regression')
        
    Returns:
        Dictionary of healthcare-specific metrics
    """
    metrics = {}
    
    if task_type == 'classification':
        # Standard classification metrics
        metrics = calculate_classification_metrics(y_true, y_pred)
        
    elif task_type == 'segmentation':
        # For medical image segmentation
        # Convert arrays if needed
        y_true_np = np.array(y_true) if isinstance(y_true, list) else y_true
        y_pred_np = np.array(y_pred) if isinstance(y_pred, list) else y_pred
        
        # Calculate Dice coefficient (F1 score for segmentation)
        intersection = np.sum(y_true_np * y_pred_np)
        union = np.sum(y_true_np) + np.sum(y_pred_np)
        
        metrics['dice'] = (2.0 * intersection) / (union + 1e-10)
        
        # Calculate IoU (Jaccard index)
        metrics['iou'] = intersection / (union - intersection + 1e-10)
        
        # Calculate sensitivity and specificity
        tp = np.sum(y_true_np * y_pred_np)
        fp = np.sum(y_pred_np) - tp
        fn = np.sum(y_true_np) - tp
        tn = np.prod(y_true_np.shape) - (tp + fp + fn)
        
        metrics['sensitivity'] = tp / (tp + fn + 1e-10)
        metrics['specificity'] = tn / (tn + fp + 1e-10)
        
    elif task_type == 'regression':
        # For regression tasks like predicting lab values
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        
    return metrics

def aggregate_client_metrics(client_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregate metrics from multiple federated clients.
    
    Args:
        client_metrics: List of metric dictionaries from clients
        
    Returns:
        Dictionary of aggregated metrics
    """
    if not client_metrics:
        return {}
    
    # Initialize with keys from the first client
    aggregated = {k: [] for k in client_metrics[0].keys()}
    
    # Collect metrics from all clients
    for metrics in client_metrics:
        for k, v in metrics.items():
            if k in aggregated:
                aggregated[k].append(v)
    
    # Calculate mean for each metric
    return {k: np.mean(v) for k, v in aggregated.items()} 