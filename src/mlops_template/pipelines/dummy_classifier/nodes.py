"""
This is a boilerplate pipeline 'dummy_classifier'
generated using Kedro 0.18.11
"""
from typing import Literal

import pandas as pd
from sklearn.dummy import DummyClassifier


def get_features_and_target(df: pd.DataFrame):
    X = df.drop(columns=["tag"])
    y = df["tag"]
    return X, y


def fit_dummy_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    strategy: Literal[
        "most_frequent", "prior", "stratified", "uniform", "constant"
    ] = "most_frequent",
):
    """Launces a dummy classifier to predict tag

    Args:
        df (pd.DataFrame): the dataframe
    """
    dummy = DummyClassifier(strategy=strategy)

    dummy.fit(X_train, y_train)

    return dummy
