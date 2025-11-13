# pipelines.py (additions)

from __future__ import annotations
from typing import Tuple, Callable, Dict, Any, List
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
)

from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = [
    "prep_train_test",
    "train_logreg",
    "train_logreg_l1",
    "evaluate_classifier",
    "cross_validate_logreg",
]

def prep_train_test(
    Xy: pd.DataFrame,
    label_col: str = "label",
    test_size: float = 0.20,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, pd.DataFrame, pd.DataFrame]:
    y = Xy[label_col].astype(int).values
    X = Xy.drop(columns=[label_col]).copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    strat = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )

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
    clf = LogisticRegression(
        class_weight=class_weight,
        random_state=random_state,
        max_iter=max_iter,
        C=C,
    )
    clf.fit(X_train_scaled, y_train)
    return clf


def train_logreg_l1(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    *,
    class_weight: str | dict = "balanced",
    random_state: int = 42,
    max_iter: int = 1000,
    C: float = 1.0,
) -> LogisticRegression:
    clf = LogisticRegression(
        penalty="l1",
        solver="liblinear",
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
    feature_names: List[str],
    digits: int = 3,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    y_pred = clf.predict(X_test_scaled)
    report_dict = classification_report(y_test, y_pred, output_dict=True, digits=digits)
    report_df = pd.DataFrame(report_dict).transpose()
    cm = confusion_matrix(y_test, y_pred)

    coefs = np.ravel(clf.coef_)
    coef_df = (
        pd.DataFrame({"feature": feature_names, "coef": coefs, "odds_ratio": np.exp(coefs)})
        .sort_values(by="coef", ascending=False, key=lambda s: s.abs())
        .reset_index(drop=True)
    )
    return report_df, cm, coef_df


# ==== Cross-validation =====

def _coerce_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

def _fold_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    # Positive class is 1
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    acc = (y_true == y_pred).mean()
    roc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan
    pr  = average_precision_score(y_true, y_prob)
    return {
        "accuracy": acc,
        "precision_pos": prec,
        "recall_pos": rec,
        "f1_pos": f1,
        "roc_auc": roc,
        "pr_auc": pr,
    }

def cross_validate_logreg(
    Xy: pd.DataFrame,
    *,
    label_col: str = "label",
    k: int = 5,
    random_state: int = 42,
    use_l1: bool = False,
    C: float = 1.0,
    class_weight: str | dict = "balanced",
    return_models: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[LogisticRegression] | None]:
    """
    Stratified K-fold CV with scaling in-fold.
    Returns:
      - folds_df: per-fold metrics
      - summary_df: mean ± std over folds
      - models (optional): list of fitted models (one per fold)
    """
    Xy = Xy.copy()
    y = Xy[label_col].astype(int).values
    X = _coerce_features(Xy.drop(columns=[label_col]))

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    rows, models = [], []

    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y[tr], y[te]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr.values)
        X_te_s = scaler.transform(X_te.values)

        if use_l1:
            clf = LogisticRegression(
                penalty="l1", solver="liblinear",
                class_weight=class_weight, C=C,
                random_state=random_state, max_iter=1000
            )
        else:
            clf = LogisticRegression(
                class_weight=class_weight, C=C,
                random_state=random_state, max_iter=200
            )
        clf.fit(X_tr_s, y_tr)

        y_prob = clf.predict_proba(X_te_s)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        m = _fold_metrics(y_te, y_prob, y_pred)
        m["fold"] = fold
        rows.append(m)
        if return_models:
            models.append(clf)

    folds_df = pd.DataFrame(rows).set_index("fold")
    summary = folds_df.agg(["mean", "std"]).T  # metrics x {mean,std}
    return folds_df, summary, (models if return_models else None)

# ------------------------------------------------------------
# Non-linear baseline: HistGradientBoosting + evaluation
# ------------------------------------------------------------
def prep_train_test_tabular(
    Xy: pd.DataFrame,
    label_col: str = "label",
    test_size: float = 0.20,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    
    y = Xy[label_col].astype(int).values
    X = Xy.drop(columns=[label_col]).apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    strat = y if stratify else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )
    return X_tr, X_te, y_tr, y_te

def train_hgb(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    max_depth: int | None = None,
    learning_rate: float = 0.06,
    max_iter: int = 400,
    l2_regularization: float = 0.0,
    random_state: int = 42
) -> HistGradientBoostingClassifier:
    """
    HistGradientBoostingClassifier with class balancing via sample weights.
    """
    clf = HistGradientBoostingClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        max_iter=max_iter,
        l2_regularization=l2_regularization,
        random_state=random_state
    )
    sw = compute_sample_weight(class_weight="balanced", y=y_train)
    clf.fit(X_train, y_train, sample_weight=sw)
    return clf

def evaluate_hgb(
    clf: HistGradientBoostingClassifier,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    *,
    feature_names: List[str] | None = None,
    n_perm_repeats: int = 30,
    random_state: int = 42
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, Dict[str, float]]:
    """
    Returns:
      - classification_report_df
      - confusion_matrix array
      - permutation importances (DataFrame)
      - dict of global metrics: roc_auc, pr_auc
    """
    # Probabilities for positive class
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # Reports
    report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True, digits=3)).T
    cm = confusion_matrix(y_test, y_pred)

    # Global ranking metrics
    metrics = {
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc":  average_precision_score(y_test, y_prob),
    }

    # Permutation importance (safer than relying on built-in importances)
    r = permutation_importance(
        clf, X_test, y_test, n_repeats=n_perm_repeats, random_state=random_state, scoring="roc_auc"
    )
    if feature_names is None:
        feature_names = list(X_test.columns)
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "pi_mean": r.importances_mean,
        "pi_std": r.importances_std
    }).sort_values("pi_mean", ascending=False)

    return report_df, cm, imp_df, metrics


# =====
# A helper to collapse 'rare' features based on X_train only
# =====

class CollapseRareRels(BaseEstimator, TransformerMixin):
    """
    Collapse rare *relation proportion* columns into a single 'other' column.

    Parameters
    ----------
    min_docs : int
        Keep a relation column if it is > 0 in at least `min_docs` training rows.
    other_col : str
        Name of the bucket column for collapsed relations.
    rel_cols : list[str] | None
        Explicit list of relation columns to operate on. If None, will infer as
        columns starting with 'rel_' (excluding `other_col`).
    """
    def __init__(self, min_docs: int = 3, other_col: str = "rel_OTHER", rel_cols=None):
        self.min_docs = min_docs
        self.other_col = other_col
        self.rel_cols = rel_cols

        # fitted attrs
        self.keep_cols_ = None
        self.all_rel_cols_ = None
        self.collapsed_cols_ = None
        self.last_dropped_cols_ = None

    def _infer_rel_cols(self, X: pd.DataFrame) -> list[str]:
        # Fallback inference if rel_cols not provided: anything that looks like a relation proportion
        return [c for c in X.columns if c.startswith("rel_") and c != self.other_col]

    def fit(self, X: pd.DataFrame, y=None):
        if self.rel_cols is None:
            rel_cols = self._infer_rel_cols(X)
        else:
            rel_cols = [c for c in self.rel_cols if c in X.columns and c != self.other_col]

        self.all_rel_cols_ = rel_cols

        if not rel_cols:
            self.keep_cols_ = []
            self.collapsed_cols_ = []
            return self

        # Count in how many docs a relation appears (>0)
        doc_counts = (X[rel_cols] > 0).sum(axis=0)

        self.keep_cols_ = [c for c in rel_cols if int(doc_counts.get(c, 0)) >= self.min_docs]
        self.collapsed_cols_ = [c for c in rel_cols if c not in self.keep_cols_]  # <-- store collapsed set
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Nothing to do if we had no rel cols at fit time
        if not self.all_rel_cols_:
            self.last_dropped_cols_ = []
            return X

        present_rel = [c for c in self.all_rel_cols_ if c in X.columns]
        keep = [c for c in (self.keep_cols_ or []) if c in present_rel]
        drop = [c for c in present_rel if c not in keep]
        self.last_dropped_cols_ = drop  # <-- optional: what we actually dropped on this X

        if drop:
            other = X[drop].sum(axis=1)
            if self.other_col in X.columns:
                X[self.other_col] = X[self.other_col].fillna(0.0) + other
            else:
                X[self.other_col] = other
            X = X.drop(columns=drop)

        return X