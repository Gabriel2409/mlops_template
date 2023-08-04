"""
This is a boilerplate pipeline 'log_dvc_ref_to_datasets'
generated using Kedro 0.18.11
"""

from kedro.pipeline import Pipeline, node

from .nodes import log_dvc


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        nodes=[
            node(
                func=log_dvc,
                inputs=["train_dvc_file", "test_dvc_file"],
                outputs=["dvc_train_dataset_artifact", "dvc_test_dataset_artifact"],
                name="log_dvc",
            )
        ]
    )
