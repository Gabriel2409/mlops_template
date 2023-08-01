"""
This is a boilerplate pipeline 'encode_tag'
generated using Kedro 0.18.11
"""

from kedro.pipeline import Pipeline, node

from .nodes import encode_tag


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        nodes=[
            node(
                func=encode_tag,
                inputs=["train_df_to_encode", "test_df_to_encode"],
                outputs=[
                    "encoded_train_df",
                    "encoded_test_df",
                    "label_encoder_mapping",
                ],
                name="encode_tag",
            )
        ]
    )
