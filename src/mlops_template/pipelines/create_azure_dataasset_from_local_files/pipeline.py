"""
This is a boilerplate pipeline 'create_azure_dataasset_from_local_files'
generated using Kedro 0.18.12
"""

from kedro.pipeline import Pipeline, node


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        nodes=[
            node(
                func=lambda x: x,
                inputs="projects_train_raw_local",
                outputs="projects_train_raw",
                name="create_train_dataasset",
            ),
            node(
                func=lambda x: x,
                inputs="projects_test_raw_local",
                outputs="projects_test_raw",
                name="create_test_dataasset",
            ),
        ]
    )
