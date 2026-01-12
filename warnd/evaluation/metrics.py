"""
Evaluation metrics for depression prediction model.

Implements the competition metrics:
1. Primary: AUC-ROC
2. Tie-breaker 1: Log Loss
3. Tie-breaker 2: F1 Score
"""

import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import (
    roc_auc_score, 
    log_loss, 
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    roc_curve
)


# Constants for metric comparison
COMPARISON_TOLERANCE = 1e-6  # Tolerance for comparing floating point metrics


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray = None
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        y_pred_proba: Predicted probabilities (if available)
        
    Returns:
        Dictionary of metric name to value
    """
    metrics = {}
    
    # Basic classification metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_positives'] = tp
    metrics['true_negatives'] = tn
    metrics['false_positives'] = fp
    metrics['false_negatives'] = fn
    
    # Specificity
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Competition metrics (require probabilities)
    if y_pred_proba is not None:
        # AUC-ROC (primary metric)
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
        except ValueError:
            # Handle case where all predictions are the same class
            metrics['auc_roc'] = 0.5
        
        # Log Loss (first tie-breaker)
        try:
            metrics['log_loss'] = log_loss(y_true, y_pred_proba)
        except ValueError:
            metrics['log_loss'] = float('inf')
    
    return metrics


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray = None,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate model using competition metrics.
    
    Primary metric: AUC-ROC
    Tie-breaker 1: Log Loss
    Tie-breaker 2: F1 Score
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        y_pred_proba: Predicted probabilities (for positive class)
        verbose: Print results
        
    Returns:
        Dictionary of metrics
    """
    metrics = compute_metrics(y_true, y_pred, y_pred_proba)
    
    if verbose:
        print("=" * 60)
        print("MODEL EVALUATION RESULTS")
        print("=" * 60)
        
        # Competition metrics
        if y_pred_proba is not None:
            print("\nCompetition Metrics:")
            print(f"  Primary   - AUC-ROC:  {metrics['auc_roc']:.4f}")
            print(f"  Tie-break 1 - Log Loss: {metrics['log_loss']:.4f}")
        print(f"  Tie-break 2 - F1 Score: {metrics['f1_score']:.4f}")
        
        # Classification metrics
        print("\nClassification Metrics:")
        print(f"  Accuracy:   {metrics['accuracy']:.4f}")
        print(f"  Precision:  {metrics['precision']:.4f}")
        print(f"  Recall:     {metrics['recall']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        
        # Confusion matrix
        print("\nConfusion Matrix:")
        print(f"  True Positives:  {metrics['true_positives']}")
        print(f"  True Negatives:  {metrics['true_negatives']}")
        print(f"  False Positives: {metrics['false_positives']}")
        print(f"  False Negatives: {metrics['false_negatives']}")
        
        print("=" * 60)
    
    return metrics


def compare_models(
    metrics1: Dict[str, float],
    metrics2: Dict[str, float],
    model1_name: str = "Model 1",
    model2_name: str = "Model 2"
) -> str:
    """
    Compare two models using competition ranking criteria.
    
    Args:
        metrics1: Metrics for first model
        metrics2: Metrics for second model
        model1_name: Name of first model
        model2_name: Name of second model
        
    Returns:
        Name of better model
    """
    # Compare by AUC-ROC first
    if abs(metrics1['auc_roc'] - metrics2['auc_roc']) > COMPARISON_TOLERANCE:
        winner = model1_name if metrics1['auc_roc'] > metrics2['auc_roc'] else model2_name
        return f"{winner} wins (higher AUC-ROC)"
    
    # Tie on AUC-ROC, compare by Log Loss (lower is better)
    if abs(metrics1['log_loss'] - metrics2['log_loss']) > COMPARISON_TOLERANCE:
        winner = model1_name if metrics1['log_loss'] < metrics2['log_loss'] else model2_name
        return f"{winner} wins (lower Log Loss)"
    
    # Tie on Log Loss, compare by F1 Score
    if abs(metrics1['f1_score'] - metrics2['f1_score']) > COMPARISON_TOLERANCE:
        winner = model1_name if metrics1['f1_score'] > metrics2['f1_score'] else model2_name
        return f"{winner} wins (higher F1 Score)"
    
    return "Tie on all metrics"


def get_optimal_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    metric: str = "f1"
) -> Tuple[float, float]:
    """
    Find optimal classification threshold.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        metric: Metric to optimize ('f1', 'accuracy', 'youden')
        
    Returns:
        Tuple of (optimal_threshold, metric_value)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    
    if metric == "f1":
        # Find threshold that maximizes F1 score
        best_threshold = 0.5
        best_f1 = 0.0
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        return best_threshold, best_f1
    
    elif metric == "accuracy":
        # Find threshold that maximizes accuracy
        best_threshold = 0.5
        best_acc = 0.0
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            acc = accuracy_score(y_true, y_pred)
            if acc > best_acc:
                best_acc = acc
                best_threshold = threshold
        
        return best_threshold, best_acc
    
    elif metric == "youden":
        # Youden's J statistic = TPR - FPR
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        return thresholds[best_idx], j_scores[best_idx]
    
    else:
        raise ValueError(f"Unknown metric: {metric}")


def cross_validate_metrics(
    y_true_list: list,
    y_pred_list: list,
    y_pred_proba_list: list = None
) -> Dict[str, Tuple[float, float]]:
    """
    Compute metrics across multiple folds and return mean and std.
    
    Args:
        y_true_list: List of true labels for each fold
        y_pred_list: List of predictions for each fold
        y_pred_proba_list: List of probability predictions for each fold
        
    Returns:
        Dictionary mapping metric name to (mean, std) tuple
    """
    if y_pred_proba_list is None:
        y_pred_proba_list = [None] * len(y_true_list)
    
    all_metrics = []
    for y_true, y_pred, y_pred_proba in zip(y_true_list, y_pred_list, y_pred_proba_list):
        metrics = compute_metrics(y_true, y_pred, y_pred_proba)
        all_metrics.append(metrics)
    
    # Compute mean and std for each metric
    result = {}
    metric_names = all_metrics[0].keys()
    
    for metric_name in metric_names:
        values = [m[metric_name] for m in all_metrics]
        result[metric_name] = (np.mean(values), np.std(values))
    
    return result
