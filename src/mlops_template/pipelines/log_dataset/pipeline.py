"""
This is a boilerplate pipeline 'log_dataset'
generated using Kedro 0.18.11
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import log_dataset


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        nodes=[
            node(
                func=log_dataset,
                inputs="dvc_file",
                outputs="dvc_dataset",
                name="log_dataset",
            )
        ]
    )
