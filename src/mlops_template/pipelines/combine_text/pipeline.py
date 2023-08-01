"""
This is a boilerplate pipeline 'combine_text'
generated using Kedro 0.18.11
"""

from kedro.pipeline import Pipeline, node

from .nodes import combine_title_and_desc


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        nodes=[
            node(
                func=combine_title_and_desc,
                inputs="projects_train_raw",
                outputs="projects_train_text",
                name="combine_title_and_desc_train",
            ),
            node(
                func=combine_title_and_desc,
                inputs="projects_test_raw",
                outputs="projects_test_text",
                name="combine_title_and_desc_test",
            ),
        ]
    )
