"""
This is a boilerplate pipeline 'tfidf_vectorizer'
generated using Kedro 0.18.11
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from mlops_template.pipelines import combine_text, encode_tag, log_sklearn_metrics

from .nodes import get_features_and_target, optimize_tfidf_sgd


def create_pipeline(**kwargs) -> Pipeline:
    return (
        pipeline(
            pipe=combine_text.create_pipeline(),
            inputs={
                "projects_train_raw": "projects_train_dataset#urifolder",
                "projects_test_raw": "projects_test_dataset#urifolder",
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
                "label_encoder_mapping": "label_encoder_mapping",
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
                    # func=fit_tfidf_vectorizer_and_sgdclassifier,
                    func=optimize_tfidf_sgd,
                    inputs={
                        "X_train": "X_train",
                        "y_train": "y_train",
                        "n_trials": "params:tfidf_sgd_n_trials_optuna",
                    },
                    outputs=["mlflow_sklearn_classifier", "optuna_best_params"],
                    name="fit_tfidf_sgdclassifier_on_train",
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
