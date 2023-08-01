"""
This is a boilerplate pipeline 'dummy_classifier'
generated using Kedro 0.18.11
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from mlops_template.pipelines import encode_tag, log_metrics, log_dataset

from .nodes import fit_dummy_classifier


def create_pipeline(**kwargs) -> Pipeline:
    return (
        pipeline(
            pipe=log_dataset.create_pipeline(), inputs={"dvc_file": "projects_text_dvc"}
        )
        + pipeline(
            pipe=encode_tag.create_pipeline(),
            inputs={"df_to_encode": "projects_text"},
            outputs={
                "encoded_df": "encoded_df",
                "label_encoder_mapping": "label_encoder_mapping",
            },
        )
        + Pipeline(
            nodes=[
                node(
                    func=fit_dummy_classifier,
                    inputs=["encoded_df", "params:strategy"],
                    outputs=["dummy_classifier", "y_true", "y_pred"],
                    name="dummy_classifier",
                )
            ]
        )
        + pipeline(
            pipe=log_metrics.create_pipeline(),
            # inputs={"y_true": "y_true", "y_pred": "y_pred"},
            # outputs={},
        )
    )
