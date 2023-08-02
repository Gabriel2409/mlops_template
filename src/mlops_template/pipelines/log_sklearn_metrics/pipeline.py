"""
This is a boilerplate pipeline 'log_metrics'
generated using Kedro 0.18.11
"""

from kedro.pipeline import Pipeline, node

from .nodes import log_sklearn_scores


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        nodes=[
            node(
                func=log_sklearn_scores,
                inputs=["model", "X_train", "y_train", "X_test", "y_test"],
                outputs=[
                    "train_mlflow_metrics",
                    "train_mlflow_classification_report",
                    "train_mlflow_confusion_matrix",
                    "test_mlflow_metrics",
                    "test_mlflow_classification_report",
                    "test_mlflow_confusion_matrix",
                ],
                name="log_sklearn_scores",
            )
        ]
    )
