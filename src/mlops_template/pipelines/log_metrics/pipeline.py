"""
This is a boilerplate pipeline 'log_metrics'
generated using Kedro 0.18.11
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import log_main_scores


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        nodes=[
            node(
                func=log_main_scores,
                inputs=["y_true", "y_pred"],
                outputs=[
                    "mlflow_metrics",
                    "mlflow_classification_report",
                    "mlflow_confusion_matrix",
                ],
                name="log_main_scores",
            )
        ]
    )
