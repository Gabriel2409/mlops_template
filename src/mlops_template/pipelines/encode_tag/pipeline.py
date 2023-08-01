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
                inputs=["df_to_encode"],
                outputs=["encoded_df", "label_encoder_mapping"],
                name="encode_tag",
            )
        ]
    )
