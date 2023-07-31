"""
This is a boilerplate pipeline 'dummy_classifier'
generated using Kedro 0.18.11
"""
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
from mlops_template.utils.seeds import set_seeds
from typing import Literal, Optional
import mlflow


def fit_dummy_classifier(
    df: pd.DataFrame,
    strategy: Literal[
        "most_frequent", "prior", "stratified", "uniform", "constant"
    ] = "most_frequent",
    seed: Optional[int] = None,
):
    """Launces a dummy classifier to predict tag

    Args:
        df (pd.DataFrame): the dataframe
    """
    if seed:
        set_seeds(seed)
    dummy = DummyClassifier(strategy=strategy)
    X = df.drop(columns=["tag"])
    y = df["tag"]
    dummy.fit(X, y)

    return dummy, y, dummy.predict(X)
