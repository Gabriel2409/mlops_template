"""
This is a boilerplate pipeline 'combine_text'
generated using Kedro 0.18.11
"""

import pandas as pd


def combine_title_and_desc(df: pd.DataFrame) -> pd.DataFrame:
    """Combines title and desc fields into one text field

    Args:
        df (pd.DataFrame): the dataframe to modify
    Returns
        df(pd.DataFrame): the modified df
    """
    df.insert(2, "text", df["title"] + " " + df["description"])
    df = df.drop(columns=["title", "description"])
    return df
