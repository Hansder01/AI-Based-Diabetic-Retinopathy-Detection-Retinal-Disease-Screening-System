"""
Evaluation Metrics Framework for Diabetic Retinopathy Ordinal Regression and Classification.

Implements:
1. Quadratic Weighted Kappa (QWK) - the gold standard benchmark for DR grading.
2. Mean Absolute Error (MAE) on ordinal stages.
3. Accuracy, Macro Precision, Recall, and Macro F1.
4. Confusion Matrix computation.
"""

from typing import Dict, Any, List, Union
import numpy as np
from sklearn.metrics import (
    cohen_kappa_score,
    mean_absolute_error,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

from src.data.dataset import ordinal_vector_to_grade


def compute_ordinal_metrics(
    y_true_grades: Union[List[int], np.ndarray],
    y_pred_probs: Union[List[List[float]], np.ndarray],
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Computes comprehensive clinical metrics comparing true integer grades
    against predicted ordinal probabilities.

    Args:
        y_true_grades: Array of ground-truth grades (0 to 4).
        y_pred_probs: Array of predicted ordinal probabilities of shape (N, 4).
        threshold: Decision threshold for decoding sigmoid probabilities.

    Returns:
        Dictionary of metrics including QWK, MAE, accuracy, macro F1, and confusion matrix.
    """
    y_true = np.asarray(y_true_grades, dtype=int)
    probs = np.asarray(y_pred_probs, dtype=np.float32)

    # Decode probabilities to discrete predictions
    y_pred = np.array([ordinal_vector_to_grade(p, threshold=threshold) for p in probs], dtype=int)

    # Quadratic Weighted Kappa (penalizes distance quadratically)
    try:
        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=[0, 1, 2, 3, 4])
    except Exception:
        qwk = 0.0

    # Mean Absolute Error
    mae = mean_absolute_error(y_true, y_pred)

    # Accuracy
    acc = accuracy_score(y_true, y_pred)

    # Precision, Recall, F1 (macro averaged across 5 classes)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3, 4],
        average="macro",
        zero_division=0
    )

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])

    return {
        "quadratic_weighted_kappa": round(float(qwk), 4),
        "mean_absolute_error": round(float(mae), 4),
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(precision), 4),
        "macro_recall": round(float(recall), 4),
        "macro_f1": round(float(f1), 4),
        "confusion_matrix": cm.tolist()
    }

