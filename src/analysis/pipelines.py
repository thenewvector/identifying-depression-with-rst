# pipelines.py

from __future__ import annotations
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

__all__ = [
    "prep_train_test",
    "train_logreg",
    "evaluate_classifier",
]

def prep_train_test(
    Xy: pd.DataFrame,
    label_col: str = "label",
    test_size: float = 0.20,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, pd.DataFrame, pd.DataFrame]:
    """
    Split into train/test and scale features.
    Assumes label column is already 0/1.

    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train, X_test
    """
    # 1) Split features/labels
    y = Xy[label_col].astype(int).values
    X = Xy.drop(columns=[label_col]).copy()

    # 2) Coerce to numeric (just in case), replace inf with nan, then fill remaining nans with 0
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 3) Train/test split (stratify to preserve class balance)
    strat = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )

    # 4) Scale (fit on train, transform both)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_test_scaled  = scaler.transform(X_test.values)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train, X_test


def train_logreg(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    *,
    class_weight: str | dict = "balanced",
    random_state: int = 42,
    max_iter: int = 200,
    C: float = 1.0,
) -> LogisticRegression:
    """
    Train a Logistic Regression baseline.
    """
    clf = LogisticRegression(
        class_weight=class_weight,
        random_state=random_state,
        max_iter=max_iter,
        C=C,
    )
    clf.fit(X_train_scaled, y_train)
    return clf


def evaluate_classifier(
    clf: LogisticRegression,
    X_test_scaled: np.ndarray,
    y_test: np.ndarray,
    digits: int = 3,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Return (classification_report_df, confusion_matrix_array).
    """
    y_pred = clf.predict(X_test_scaled)
    report_dict = classification_report(y_test, y_pred, output_dict=True, digits=digits)
    report_df = pd.DataFrame(report_dict).transpose()
    cm = confusion_matrix(y_test, y_pred)
    return report_df, cm