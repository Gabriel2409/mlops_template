"""
This is a boilerplate pipeline 'tfidf_vectorizer'
generated using Kedro 0.18.11
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline


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
