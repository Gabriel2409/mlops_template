"""
This is a boilerplate pipeline 'dummy_classifier'
generated using Kedro 0.18.11
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from mlops_template.pipelines import combine_text, encode_tag, log_sklearn_metrics

from .nodes import fit_dummy_classifier, get_features_and_target


def create_pipeline(**kwargs) -> Pipeline:
    return (
        pipeline(
            pipe=combine_text.create_pipeline(),
            inputs={
                "projects_train_raw": "projects_train_raw",
                "projects_test_raw": "projects_test_raw",
            },
            outputs={
                "projects_train_text": "projects_train_text",
                "projects_test_text": "projects_test_text",
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
                "label_encoder_mapping": "label_encoder_mapping_mlflow",
            },
        )
        + Pipeline(
            nodes=[
                node(
                    func=get_features_and_target,
                    inputs={"df": "encoded_train_df"},
                    outputs=["X_train", "y_train"],
                    name="get_features_and_target_train",
                ),
                node(
                    func=get_features_and_target,
                    inputs={"df": "encoded_test_df"},
                    outputs=["X_test", "y_test"],
                    name="get_features_and_target_test",
                ),
                node(
                    func=fit_dummy_classifier,
                    inputs=["X_train", "y_train", "params:strategy"],
                    outputs="mlflow_sklearn_classifier",
                    name="fit_dummy_classifier_on_train",
                ),
            ]
        )
        + pipeline(
            pipe=log_sklearn_metrics.create_pipeline(),
            inputs={
                "model": "mlflow_sklearn_classifier",
                "X_train": "X_train",
                "y_train": "y_train",
                "X_test": "X_test",
                "y_test": "y_test",
            },
            outputs={
                "sklearn_classifier_mlflow": "sklearn_classifier_mlflow",
                "train_metrics_mlflow": "train_metrics_mlflow",
                "train_classification_report_mlflow": "train_classification_report_mlflow",
                "train_confusion_matrix_mlflow": "train_confusion_matrix_mlflow",
                "test_metrics_mlflow": "test_metrics_mlflow",
                "test_classification_report_mlflow": "test_classification_report_mlflow",
                "test_confusion_matrix_mlflow": "test_confusion_matrix_mlflow",
            },
        )
    )
