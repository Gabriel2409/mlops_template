"""
This is a boilerplate pipeline 'log_metrics'
generated using Kedro 0.18.11
"""
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def format_classification_report(cr: dict):
    """transforms the classification report into a dataframe"""
    return pd.DataFrame(cr).reset_index().rename(columns={"index": "score"})


def log_sklearn_scores(model, X_train, y_train, X_test, y_test):
    """returns all important metrics for logging"""

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    cm_train = confusion_matrix(y_true=y_train, y_pred=y_train_pred)

    report_train: dict = classification_report(
        y_true=y_train, y_pred=y_train_pred, output_dict=True
    )
    cm_test = confusion_matrix(y_true=y_test, y_pred=y_test_pred)

    report_test: dict = classification_report(
        y_true=y_test, y_pred=y_test_pred, output_dict=True
    )

    train_metrics = {}
    test_metrics = {}
    for average in ["macro avg", "weighted avg"]:
        for score in ["precision", "recall", "f1-score", "support"]:
            train_metrics[f"{average}_{score}"] = [
                {"value": report_train[average][score], "step": 0}
            ]
            test_metrics[f"{average}_{score}"] = [
                {"value": report_test[average][score], "step": 0}
            ]

    log = logging.getLogger(__name__)
    log.info(f'Test Macro F1 Score: {report_test["macro avg"]["f1-score"]}')
    log.info(f'Test Weighted F1 Score: {report_test["weighted avg"]["f1-score"]}')
    return (
        train_metrics,
        format_classification_report(report_train),
        np.array2string(cm_train),
        test_metrics,
        format_classification_report(report_test),
        np.array2string(cm_test),
    )
