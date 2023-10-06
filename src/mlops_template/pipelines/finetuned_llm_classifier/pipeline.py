"""
This is a boilerplate pipeline 'finetuned_llm_classifier'
generated using Kedro 0.18.13
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from mlops_template.pipelines import encode_tag

from .nodes import train_llm_classifier


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        pipe=encode_tag.create_pipeline(),
        inputs={
            "train_df_to_encode": "projects_train_dataset#urifolder",
            "test_df_to_encode": "projects_test_dataset#urifolder",
        },
        outputs={
            "encoded_train_df": "encoded_train_df",
            "encoded_test_df": "encoded_test_df",
            "label_encoder_mapping": "label_encoder_mapping",
        },
    ) + Pipeline(
        nodes=[
            node(
                func=train_llm_classifier,
                inputs={
                    "train_val_df": "encoded_train_df",
                    "test_df": "encoded_test_df",
                    "config": "params:finetuned_llm_config",
                    "label_encoder_mapping": "label_encoder_mapping",
                },
                outputs="pytorch_classifier_mlflow",
                name="train_llm_classifier",
                tags=["gpu"],
            )
        ]
    )
