"""
This is a boilerplate pipeline 'dummy_classifier'
generated using Kedro 0.18.11
"""

from kedro.pipeline import Pipeline, node
from .nodes import fit_dummy_classifier
from kedro.pipeline.modular_pipeline import pipeline
from mlops_template.pipelines import log_metrics


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        nodes=[
            node(
                func=fit_dummy_classifier,
                inputs=["projects_text", "params:strategy", "params:seed"],
                outputs=["dummy_classifier", "y_true", "y_pred"],
                name="dummy_classifier",
            )
        ]
    ) + pipeline(
        pipe=log_metrics.create_pipeline(),
        # inputs={"y_true": "y_true", "y_pred": "y_pred"},
        # outputs={"metrics": "dummy_metrics"},
    )
