"""
This is a boilerplate pipeline 'dummy_classifier'
generated using Kedro 0.18.11
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from mlops_template.pipelines import encode_tag, log_datasets, log_sklearn_metrics

from .nodes import fit_dummy_classifier


def create_pipeline(**kwargs) -> Pipeline:
    return (
        pipeline(
            pipe=log_datasets.create_pipeline(),
            inputs={
                "train_dvc_file": "projects_train_text_dvc",
                "test_dvc_file": "projects_test_text_dvc",
            },
        )
        + pipeline(
            pipe=encode_tag.create_pipeline(),
            inputs={
                "train_df_to_encode": "projects_train_text",
                "test_df_to_encode": "projects_test_text",
            },
            outputs={
                "encoded_train_df": "encoded_train_df",
                "encoded_test_df": "encoded_test_df",
                "label_encoder_mapping": "label_encoder_mapping",
            },
        )
        + Pipeline(
            nodes=[
                node(
                    func=fit_dummy_classifier,
                    inputs=["encoded_train_df", "params:strategy"],
                    outputs="dummy_classifier",
                    name="dummy_classifier",
                )
            ]
        )
        + pipeline(
            pipe=log_sklearn_metrics.create_pipeline(),
            inputs={
                "model": "dummy_classifier",
                "train_df": "encoded_train_df",
                "test_df": "encoded_test_df",
            }
            # inputs={"y_true": "y_true", "y_pred": "y_pred"},
        )
    )
