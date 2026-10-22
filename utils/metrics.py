import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def safe_metric(fn, *args, default: float = math.nan, **kwargs) -> float:
    """
    Safely executes a scikit-learn metric function. 
    Returns a default value (usually NaN) if an exception occurs 
    (e.g., calculating AUROC on a batch with only a single class).
    """
    try:
        return float(fn(*args, **kwargs))
    except Exception:
        return default


def evaluate_ranking_metrics(
    targets: np.ndarray, 
    probabilities: np.ndarray, 
    label_names: Sequence[str]
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Calculates threshold-independent metrics (AUROC, AUPRC).
    """
    rows = []
    aucs = []
    aps = []
    
    for index, label in enumerate(label_names):
        y_true = targets[:, index]
        y_score = probabilities[:, index]
        
        # AUROC requires at least one positive and one negative example
        auc = safe_metric(roc_auc_score, y_true, y_score) if np.unique(y_true).size > 1 else math.nan
        ap = safe_metric(average_precision_score, y_true, y_score)
        
        rows.append({
            "label": label,
            "positive_count": int(y_true.sum()),
            "prevalence": float(y_true.mean()),
            "auroc": auc,
            "auprc": ap,
        })
        aucs.append(auc)
        aps.append(ap)

    summary = {
        "macro_auroc": float(np.nanmean(aucs)),
        "macro_auprc": float(np.nanmean(aps)),
    }
    return pd.DataFrame(rows), summary


def find_optimal_thresholds(
    targets: np.ndarray, 
    probabilities: np.ndarray, 
    label_names: Sequence[str], 
    step: float = 0.001
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Sweeps through probability thresholds to find the threshold that maximizes F1 score per label.
    For single-label, this directly maximizes F1.
    For multilabel, independently maximizing F1 per label maximizes macro-F1.
    Uses candidate thresholds in exact steps of `step` (default: 0.001) plus distribution percentiles.
    """
    best_thresholds = []
    rows = []

    # Grid candidates in exact requested increments (e.g. 0.001 -> 999 thresholds)
    candidates = np.round(np.arange(step, 1.0, step, dtype=np.float32), decimals=4)

    for index, label in enumerate(label_names):
        y_true = targets[:, index]
        y_score = probabilities[:, index]
        
        # Vectorized confusion matrix calculation across all candidate thresholds
        preds = y_score[:, None] >= candidates[None, :]
        y_bool = (y_true > 0)[:, None]
        
        tp = np.logical_and(y_bool, preds).sum(axis=0)
        fp = np.logical_and(~y_bool, preds).sum(axis=0)
        fn = np.logical_and(y_bool, ~preds).sum(axis=0)
        tn = np.logical_and(~y_bool, ~preds).sum(axis=0)
        
        # F1 optimization (maximizes F1 for single-label, maximizes macro-F1 for multilabel)
        denom = 2 * tp + fp + fn
        f1_scores = np.where(denom > 0, (2.0 * tp) / denom, 0.0)
        
        best_f1_idx = int(np.argmax(f1_scores))
        best_f1_threshold = float(candidates[best_f1_idx])
        best_f1_val = float(f1_scores[best_f1_idx])
        
        # Optimal threshold selection strictly by F1
        best_primary_threshold = best_f1_threshold
                
        # Calculate full metrics using the optimal winning threshold
        primary_prediction = (y_score >= best_primary_threshold).astype(np.int8)
        has_both = np.unique(y_true).size > 1
        
        primary_acc = safe_metric(accuracy_score, y_true, primary_prediction, default=0.0)
        primary_bal_acc = safe_metric(balanced_accuracy_score, y_true, primary_prediction, default=0.0) if has_both else math.nan
        primary_f1_final = safe_metric(f1_score, y_true, primary_prediction, zero_division=0, default=0.0)
        primary_prec = safe_metric(precision_score, y_true, primary_prediction, zero_division=0, default=0.0)
        primary_rec = safe_metric(recall_score, y_true, primary_prediction, zero_division=0, default=0.0)
        primary_auroc = safe_metric(roc_auc_score, y_true, y_score, default=0.0) if has_both else math.nan
        primary_auprc = safe_metric(average_precision_score, y_true, y_score, default=0.0) if y_true.sum() > 0 else math.nan
        
        best_thresholds.append(best_primary_threshold)
        rows.append({
            "label": label,
            "optimal_threshold": best_primary_threshold,
            "threshold": best_primary_threshold,
            "optimal_f1_threshold": best_f1_threshold,
            "validation_f1": primary_f1_final,
            "f1": primary_f1_final,
            "validation_accuracy": primary_acc,
            "accuracy": primary_acc,
            "validation_balanced_accuracy": primary_bal_acc,
            "balanced_accuracy": primary_bal_acc,
            "precision": primary_prec,
            "recall": primary_rec,
            "auroc": primary_auroc,
            "auprc": primary_auprc,
        })

    return np.asarray(best_thresholds, dtype=np.float32), pd.DataFrame(rows)


def evaluate_multilabel_predictions(
    targets: np.ndarray,
    probabilities: np.ndarray,
    label_names: Sequence[str],
    thresholds: Union[float, np.ndarray] = 0.5
) -> Tuple[pd.DataFrame, Dict[str, float], np.ndarray]:
    """
    Unified evaluation pipeline. Calculates threshold-dependent metrics (Accuracy, F1, etc.)
    alongside ranking metrics (AUROC, AUPRC) for a comprehensive overview.
    """
    # Standardize thresholds to an array matching the number of labels
    if isinstance(thresholds, (float, int)):
        thresholds = np.full(len(label_names), float(thresholds), dtype=np.float32)
        
    predictions = (probabilities >= thresholds.reshape(1, -1)).astype(np.int8)
    rows = []

    for index, label in enumerate(label_names):
        y_true = targets[:, index].astype(np.int8)
        y_score = probabilities[:, index]
        y_pred = predictions[:, index]

        # Handle edge cases in confusion matrix safely
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        except Exception:
            tn, fp, fn, tp = 0, 0, 0, 0

        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else math.nan
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else math.nan
        has_both_classes = np.unique(y_true).size == 2

        rows.append({
            "label": label,
            "threshold": float(thresholds[index]),
            "accuracy": safe_metric(accuracy_score, y_true, y_pred),
            "balanced_accuracy": safe_metric(balanced_accuracy_score, y_true, y_pred) if has_both_classes else math.nan,
            "f1": safe_metric(f1_score, y_true, y_pred, zero_division=0),
            "precision": safe_metric(precision_score, y_true, y_pred, zero_division=0),
            "recall": safe_metric(recall_score, y_true, y_pred, zero_division=0),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "youden_j": (sensitivity + specificity - 1.0) if not math.isnan(sensitivity) and not math.isnan(specificity) else math.nan,
            "auroc": safe_metric(roc_auc_score, y_true, y_score) if has_both_classes else math.nan,
            "auprc": safe_metric(average_precision_score, y_true, y_score) if y_true.sum() > 0 else math.nan,
            "positive_count": int(y_true.sum()),
            "prevalence": float(y_true.mean()),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        })

    metrics_frame = pd.DataFrame(rows)
    
    # Generate Macro/Micro Summary
    summary = {
        "macro_accuracy": float(metrics_frame["accuracy"].mean()),
        "macro_balanced_accuracy": float(metrics_frame["balanced_accuracy"].mean(skipna=True)),
        "macro_youden_j": float(metrics_frame["youden_j"].mean(skipna=True)),
        "macro_f1": float(metrics_frame["f1"].mean(skipna=True)),
        "macro_precision": float(metrics_frame["precision"].mean(skipna=True)),
        "macro_recall": float(metrics_frame["recall"].mean(skipna=True)),
        "macro_auroc": float(metrics_frame["auroc"].mean(skipna=True)),
        "macro_auprc": float(metrics_frame["auprc"].mean(skipna=True)),
        "micro_f1": safe_metric(f1_score, targets, predictions, average="micro", zero_division=0),
        "samples_f1": safe_metric(f1_score, targets, predictions, average="samples", zero_division=0),
    }

    return metrics_frame, summary, predictions


def save_predictions(
    path: Path,
    entity_ids: Sequence[str],
    id_column_name: str,
    label_names: Sequence[str],
    targets: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    secondary_ids: Sequence[str] = None,
    secondary_id_name: str = None
) -> None:
    """
    Saves targets, probabilities, and binary predictions to a CSV file.
    Flexible enough to handle both patient-level (GNN) and image-level (ConvNeXt) datasets.
    """
    data = {id_column_name: entity_ids}
    
    if secondary_ids is not None and secondary_id_name is not None:
        data[secondary_id_name] = secondary_ids

    for index, label in enumerate(label_names):
        data[f"{label}_target"] = targets[:, index].astype(np.int8)
        data[f"{label}_probability"] = probabilities[:, index]
        data[f"{label}_prediction"] = predictions[:, index].astype(np.int8)
        
    pd.DataFrame(data).to_csv(path, index=False)