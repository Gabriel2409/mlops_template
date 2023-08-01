"""
This is a boilerplate pipeline 'encode_tag'
generated using Kedro 0.18.11
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def encode_tag(df: pd.DataFrame):
    """Encodes the tag column of the df and saves the label encoder.
    This pipeline should not be used by itself
    Args
        df (pd.DataFrame) the dataframe containing the tag column
    Returns
        pd.DataFrame : the df with the encoded tag
        dict: the dict representation of the label encoder
    """
    le = LabelEncoder()
    df["tag"] = le.fit_transform(df["tag"])
    mapping = dict(zip(le.classes_, [int(val) for val in le.transform(le.classes_)]))
    return df, mapping 
