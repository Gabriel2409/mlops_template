"""
This is a boilerplate pipeline 'tfidf_vectorizer'
generated using Kedro 0.18.11
"""
from argparse import Namespace

import mlflow
import numpy as np
import optuna
import pandas as pd
from optuna.integration.mlflow import MLflowCallback
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline


def get_features_and_target(df: pd.DataFrame):
    X = df["text"]
    y = df["tag"]
    return X, y


def fit_tfidf_vectorizer_and_sgdclassifier(X_train: pd.Series, y_train: pd.Series):
    """fits a tfidf vectorizer to the training tet to predict the tag"""
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 7))
    sgdclassifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        max_iter=100,
        learning_rate="constant",
        eta0=1e-1,
        power_t=0.1,
        warm_start=True,
    )
    steps = (("tfidf", vectorizer), ("sgdclassifier", sgdclassifier))
    pipe = Pipeline(steps)
    pipe.fit(X_train, y_train)

    return pipe


def partial_fit_tfidf_vectorizer_and_sgdclassifier(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_val: np.ndarray,
    y_val: np.ndarray,
    args: Namespace,
    trial: optuna.trial.Trial,
):
    """fits a tfidf vectorizer to the training tet to predict the tag"""
    vectorizer = TfidfVectorizer(
        analyzer=args.analyzer, ngram_range=(2, args.ngram_max_range)
    )
    sgdclassifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        max_iter=100,
        learning_rate="constant",
        eta0=args.eta0,
        power_t=args.power_t,
        warm_start=True,
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    unique_classes = np.unique(y_train)
    for step in range(100):
        sgdclassifier.partial_fit(X_train_vec, y_train, unique_classes)
        intermediate_value = sgdclassifier.score(X_val_vec, y_val)
        trial.report(intermediate_value, step=step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return {
        "pipe": make_pipeline(vectorizer, sgdclassifier),
        "f1_macro": f1_score(y_val, sgdclassifier.predict(X_val_vec), average="macro"),
    }


def best_pipe_callback(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="best_pipe", value=trial.user_attrs["pipe"])


def optimize_tfidf_sgd(
    X_train: np.ndarray,
    y_train: pd.Series,
    n_trials: int,
):
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, train_size=0.7, stratify=y_train
    )

    args = Namespace(analyzer="char", ngram_max_range=10, eta0=1e-1, power_t=0.3)

    def objective(args, trial):
        args.analyzer = trial.suggest_categorical(
            "analyzer", ["word", "char", "char_wb"]
        )
        args.ngram_max_range = trial.suggest_int("ngram_max_range", 3, 10)
        args.eta0 = trial.suggest_float("eta0", 1e-2, 1e0, log=True)
        args.power_t = trial.suggest_float("power_t", 0.1, 0.5)
        artifacts = partial_fit_tfidf_vectorizer_and_sgdclassifier(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            args=args,
            trial=trial,
        )
        trial.set_user_attr("f1_macro", artifacts["f1_macro"])
        trial.set_user_attr("pipe", value=artifacts["pipe"])
        return artifacts["f1_macro"]

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    current_run_id = mlflow.active_run().info.run_id
    study = optuna.create_study(
        study_name=f"optuna_{current_run_id}",
        direction="maximize",
        pruner=pruner,
    )
    mlflow_callback = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        metric_name="f1_macro",
        create_experiment=False,
        mlflow_kwargs={
            "nested": True,
            "experiment_id": mlflow.active_run().info.experiment_id,
        },
    )
    study.optimize(
        lambda trial: objective(args, trial),
        n_trials=n_trials,
        callbacks=[best_pipe_callback, mlflow_callback],
    )
    return study.user_attrs["best_pipe"], study.best_params
