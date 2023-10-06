"""
This is a boilerplate pipeline 'finetuned_llm_classifier'
generated using Kedro 0.18.13
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from mlops_template.pipelines import combine_text, encode_tag


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        pipe=encode_tag.create_pipeline(),
        inputs={
            "train_df_to_encode": "projects_test_dataset#urifolder",
            "test_df_to_encode": "projects_test_dataset#urifolder",
        },
        outputs={
            "encoded_train_df": "encoded_train_df",
            "encoded_test_df": "encoded_test_df",
            "label_encoder_mapping": "label_encoder_mapping_mlflow",
        },
    ) + Pipeline(
        nodes=[
            node(
                func=train_model,
                inputs={"df": "encoded_train_df"},
                outputs="",
                name="train_model_aaaaaaa",
                tags=["gpu"],
            )
        ]
    )
