"""
This is a boilerplate pipeline 'log_metrics'
generated using Kedro 0.18.11
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np


def log_main_scores(
    y_true: pd.DataFrame | np.ndarray, y_pred: pd.DataFrame | np.ndarray
) -> Tuple[Dict[str, List[Dict[str, float]]], pd.DataFrame, str]:
    """returns all important metrics for logging

    Args:
        y_true (pd.DataFrame | np.ndarray): true labels
        y_pred (pd.DataFrame | np.ndarray): predicted labels
    Returns:
        All the metrics to log to mlflow
    """

    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)

    report: dict = classification_report(y_true, y_pred, output_dict=True)
    # Optionally, you can also log macro and weighted averages
    metrics = {}
    for average in ["macro avg", "weighted avg"]:
        for score in ["precision", "recall", "f1-score", "support"]:
            metrics[f"{average}_{score}"] = [
                {"value": report[average][score], "step": 0}
            ]
    return metrics, pd.DataFrame(report), np.array2string(cm)
