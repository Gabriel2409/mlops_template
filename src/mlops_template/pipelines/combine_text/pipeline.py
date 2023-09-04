"""
This is a boilerplate pipeline 'combine_text'
generated using Kedro 0.18.11
"""

from kedro.pipeline import Pipeline, node

from .nodes import combine_title_and_desc


def create_pipeline(**kwargs) -> Pipeline:
    """Kedro pipeline for combining 'title' and 'description' fields into 'text' field.
    for both the train and test set
    """
    return Pipeline(
        nodes=[
            node(
                func=combine_title_and_desc,
                inputs="projects_train",
                outputs="projects_train_combined_text",
                name="combine_title_and_desc_train",
            ),
            node(
                func=combine_title_and_desc,
                inputs="projects_test",
                outputs="projects_test_combined_text",
                name="combine_title_and_desc_test",
            ),
        ]
    )
