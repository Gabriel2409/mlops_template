"""
This is a boilerplate pipeline 'log_dataset'
generated using Kedro 0.18.11
"""

from kedro.pipeline import Pipeline, node

from .nodes import log_dataset


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        nodes=[
            node(
                func=log_dataset,
                inputs=["train_dvc_file", "test_dvc_file"],
                outputs=["dvc_train_dataset_artifact", "dvc_test_dataset_artifact"],
                name="log_datasets",
            )
        ]
    )
