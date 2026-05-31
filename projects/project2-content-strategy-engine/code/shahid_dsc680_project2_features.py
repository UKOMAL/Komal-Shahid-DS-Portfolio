"""
Feature engineering for DSC680 Project 2.

Pure functions: take a raw corpus DataFrame and return a model-ready matrix.
Kept separate from the analysis module so the same transform can be applied
during the LLM generation step.

Author: Komal Shahid (DSC680, May 2026)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORICAL_COLS = ["topic", "format", "hook", "day_of_week", "daypart", "subreddit"]
NUMERIC_COLS = ["title_words", "body_words", "subreddit_subs"]


def add_engineered_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns that the model can use directly."""
    df = df.copy()
    df["log_subs"] = np.log(df["subreddit_subs"].clip(lower=1))
    df["log_title_words"] = np.log1p(df["title_words"])
    df["log_body_words"] = np.log1p(df["body_words"])
    df["log_score"] = np.log1p(df["score"])
    df["log_comments"] = np.log1p(df["num_comments"])
    return df


def build_design_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build a one-hot design matrix for engagement regression.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with one-hot encoded categoricals and numeric columns.
    y : pd.Series
        ``log_score`` (log-transformed engagement) as the target.
    """
    df = add_engineered_columns(df)
    dummies = pd.get_dummies(df[CATEGORICAL_COLS], drop_first=True)
    numeric = df[["log_subs", "log_title_words", "log_body_words"]].copy()
    X = pd.concat([numeric, dummies], axis=1).astype(float)
    y = df["log_score"]
    return X, y
