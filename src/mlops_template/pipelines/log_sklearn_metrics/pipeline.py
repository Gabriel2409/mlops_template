"""
This is a boilerplate pipeline 'log_sklearn_metrics'
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
                    "sklearn_classifier_mlflow",
                    "train_metrics_mlflow",
                    "train_classification_report_mlflow",
                    "train_confusion_matrix_mlflow",
                    "test_metrics_mlflow",
                    "test_classification_report_mlflow",
                    "test_confusion_matrix_mlflow",
                ],
                name="log_sklearn_scores",
            )
        ]
    )
