# statistics.py
from __future__ import annotations
from typing import Dict, List, Iterable, Optional
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, iqr

__all__ = ["test_features"]

# Optional: cliffs_delta dependency with a small fallback
try:
    from cliffs_delta import cliffs_delta  # returns (delta, magnitude_str)
except Exception:
    def cliffs_delta(x: List[float], y: List[float]):
        """
        Minimal fallback (O(n*m)). Returns (delta, magnitude_str='n/a').
        """
        x = list(x); y = list(y)
        gt = lt = 0
        for xi in x:
            for yi in y:
                if xi > yi: gt += 1
                elif xi < yi: lt += 1
        n = len(x) * len(y)
        delta = (gt - lt) / n if n else 0.0
        return float(delta), "n/a"


def _is_numeric_list(v) -> bool:
    if not isinstance(v, (list, tuple)):
        return False
    if not v:
        return True
    try:
        _ = [float(x) for x in v]
        return True
    except Exception:
        return False


def test_features(
    positive: Dict[str, List[float]],
    negative: Dict[str, List[float]],
    features: Optional[Iterable[str]] = None,   # if None: test all overlapping numeric keys
    exclude: Optional[Iterable[str]] = None,    # keys to skip
    correction: Optional[str] = "fdr_bh",       # "fdr_bh", "bonferroni", or None
    alpha: float = 0.05,
    decimals: int = 3,
    min_n: int = 2,                             # min per-group samples to run a test
) -> pd.DataFrame:
    """
    Compare distributions for each feature between two groups with Mann–Whitney U,
    add Cliff's delta, and apply optional multiple-comparison correction.

    Parameters
    ----------
    positive : dict[str, list[float]]
        Feature columns for the positive class.
    negative : dict[str, list[float]]
        Feature columns for the negative class.
    features : iterable[str] | None
        If provided, test only these keys. Otherwise test all overlapping numeric keys.
    exclude : iterable[str] | None
        Keys to skip (e.g., identifiers).
    correction : {"fdr_bh","bonferroni",None}
        Multiple-comparison correction method.
    alpha : float
        Significance threshold for 'reject'.
    decimals : int
        Round numeric outputs to this many decimals.
    min_n : int
        Minimum per-group sample size to run a test on a feature.

    Returns
    -------
    pandas.DataFrame
        One row per tested feature with medians, IQRs, U, raw/adjusted p, Cliff's delta, etc.
    """
    # choose candidate features
    k_pos = set(positive.keys())
    k_neg = set(negative.keys())

    if features is None:
        candidates = k_pos & k_neg
        candidates = {k for k in candidates if _is_numeric_list(positive[k]) and _is_numeric_list(negative[k])}
    else:
        candidates = set(features) & k_pos & k_neg

    if exclude:
        candidates -= set(exclude)

    rows = []
    for feat in sorted(candidates):
        pos_vals = np.asarray(positive.get(feat, []), dtype=float)
        neg_vals = np.asarray(negative.get(feat, []), dtype=float)

        pos = pos_vals[np.isfinite(pos_vals)]
        neg = neg_vals[np.isfinite(neg_vals)]

        if len(pos) < min_n or len(neg) < min_n:
            continue

        # Mann–Whitney U (two-sided)
        try:
            u_stat, p_raw = mannwhitneyu(pos, neg, alternative="two-sided")
        except ValueError:
            u_stat, p_raw = np.nan, 1.0

        # Cliff's delta
        try:
            delta, magnitude = cliffs_delta(pos.tolist(), neg.tolist())
        except Exception:
            delta, magnitude = np.nan, "n/a"

        rows.append({
            "Feature": feat,
            "Median_Positive": float(np.median(pos)),
            "Median_Negative": float(np.median(neg)),
            "IQR_Positive":   float(iqr(pos)) if len(pos) > 1 else 0.0,
            "IQR_Negative":   float(iqr(neg)) if len(neg) > 1 else 0.0,
            "Min_Positive":   float(np.min(pos)),
            "Min_Negative":   float(np.min(neg)),
            "Max_Positive":   float(np.max(pos)),
            "Max_Negative":   float(np.max(neg)),
            "MannWhitney_U":  float(u_stat),
            "p_raw":          float(p_raw),
            "Cliffs_Delta":   float(delta),
            "Effect_Size":    magnitude,
            "n_pos":          int(len(pos)),
            "n_neg":          int(len(neg)),
        })

    if not rows:
        return pd.DataFrame(columns=[
            "Feature","Median_Positive","Median_Negative","IQR_Positive","IQR_Negative",
            "Min_Positive","Min_Negative","Max_Positive","Max_Negative",
            "MannWhitney_U","p_raw","p_adj","reject","Cliffs_Delta","Effect_Size",
            "n_pos","n_neg"
        ])

    df = pd.DataFrame(rows)

    # multiple-comparison correction
    if correction is not None:
        try:
            from statsmodels.stats.multitest import multipletests
            reject, p_adj, _, _ = multipletests(df["p_raw"].values, method=correction)
        except Exception:
            if str(correction).lower() == "bonferroni":
                m = len(df)
                p_adj = np.minimum(df["p_raw"].values * m, 1.0)
                reject = p_adj < alpha
            else:
                p_adj = df["p_raw"].values
                reject = p_adj < alpha
        df["p_adj"] = p_adj
        df["reject"] = reject
    else:
        df["p_adj"] = df["p_raw"].values
        df["reject"] = df["p_adj"] < alpha

    # rounding
    num_cols = [
        "Median_Positive","Median_Negative","IQR_Positive","IQR_Negative",
        "Min_Positive","Min_Negative","Max_Positive","Max_Negative",
        "MannWhitney_U","p_raw","p_adj","Cliffs_Delta"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = df[c].astype(float).round(decimals)

    # sort by adjusted p then |delta|
    df["abs_delta"] = df["Cliffs_Delta"].abs()
    df = df.sort_values(["p_adj", "abs_delta"], ascending=[True, False]).drop(columns=["abs_delta"]).reset_index(drop=True)
    return df