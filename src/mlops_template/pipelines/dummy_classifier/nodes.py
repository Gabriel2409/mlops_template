"""
This is a boilerplate pipeline 'dummy_classifier'
generated using Kedro 0.18.11
"""
from typing import Literal

import pandas as pd
from sklearn.dummy import DummyClassifier


def fit_dummy_classifier(
    df: pd.DataFrame,
    strategy: Literal[
        "most_frequent", "prior", "stratified", "uniform", "constant"
    ] = "most_frequent",
):
    """Launces a dummy classifier to predict tag

    Args:
        df (pd.DataFrame): the dataframe
    """
    dummy = DummyClassifier(strategy=strategy)
    X = df.drop(columns=["tag"])
    y = df["tag"]
    dummy.fit(X, y)

    return dummy, y, dummy.predict(X)
