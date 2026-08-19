# restructure.py
from __future__ import annotations
from typing import List, Dict, Any, Optional, Iterable
import pandas as pd
import numpy as np

from typing import List, Dict, Any


from typing import Any, Dict, List

# ===========
# Main function to extract data from dicts
# ===========

def get_data_vectors(
    relations_list: List[str],
    rst_data_subset: Dict[str, Dict[str, Any]],
    abs_nuclearity: bool = False,
    include_meta: bool = True,
    include_relation_counts: bool = False,
    engineered: bool = True,
) -> Dict[str, List[Any]]:
    """
    Build per-document feature columns from an RST feature subset.

    Parameters
    ----------
    relations_list : list[str]
        Relation labels to extract from each document.

        Relation proportions are stored under the original relation names,
        for example:
            - 'Solutionhood'
            - 'Cause-effect'

        If ``include_relation_counts=True``, raw counts are additionally
        stored with the suffix ``_count``, for example:
            - 'Solutionhood_count'
            - 'Cause-effect_count'

    rst_data_subset : dict[str, dict[str, Any]]
        A per-document subset of the RST database, for example:

        {
            "doc-1": {
                "rst_features": {...},
                "ds": "pos",
                "ds_num": 1,
            },
            ...
        }

    abs_nuclearity : bool, default=False
        If True, include absolute nuclearity counts:
            - 'nucl_NN'
            - 'nucl_NS'
            - 'nucl_SN'

    include_meta : bool, default=True
        If True, include:
            - 'doc_id'
            - 'ds'
            - 'ds_num'

    include_relation_counts : bool, default=False
        If True, include raw counts for every relation in
        ``relations_list`` using the ``_count`` suffix.

        Also includes:
            - 'total_relations'

        ``total_relations`` is calculated as the sum of all values in the
        document's ``relation_counts`` dictionary.

    engineered : bool, default=True
        If True, include engineered features returned by
        ``add_engineered_features``.

    Returns
    -------
    dict[str, list[Any]]
        A column-oriented dictionary aligned by document order.
    """

    doc_items = list(rst_data_subset.items())
    data: Dict[str, List[Any]] = {}

    def _rst_features(
        doc_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        return doc_data.get("rst_features", {}) or {}

    def _relation_proportions(
        doc_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        return (
            _rst_features(doc_data)
            .get("relation_proportions", {})
            or {}
        )

    def _relation_counts(
        doc_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        return (
            _rst_features(doc_data)
            .get("relation_counts", {})
            or {}
        )

    def _rst_float(
        doc_data: Dict[str, Any],
        key: str,
        default: float = 0.0,
    ) -> float:
        return float(
            _rst_features(doc_data).get(key, default)
        )

    def _nuclearity_counts(
        doc_data: Dict[str, Any],
    ) -> tuple[float, float, float]:
        npat = (
            _rst_features(doc_data)
            .get("nuclearity_patterns", {})
            or {}
        )

        return (
            float(npat.get("NN", 0.0)),
            float(npat.get("NS", 0.0)),
            float(npat.get("SN", 0.0)),
        )

    def _safe_div(
        nums: List[float],
        dens: List[float],
    ) -> List[float]:
        return [
            (num / den) if den else 0.0
            for num, den in zip(nums, dens)
        ]

    # Metadata
    if include_meta:
        data["doc_id"] = [
            doc_id
            for doc_id, _ in doc_items
        ]

        data["ds"] = [
            doc_data.get("ds", "")
            for _, doc_data in doc_items
        ]

        data["ds_num"] = [
            doc_data.get("ds_num", 0)
            for _, doc_data in doc_items
        ]

    # Relation proportions and optional raw counts
    for rel in relations_list:
        data[rel] = [
            float(
                _relation_proportions(doc_data)
                .get(rel, 0.0)
            )
            for _, doc_data in doc_items
        ]

        if include_relation_counts:
            data[f"{rel}_count"] = [
                int(
                    _relation_counts(doc_data)
                    .get(rel, 0)
                )
                for _, doc_data in doc_items
            ]

    # Total number of relations in each document
    if include_relation_counts:
        data["total_relations"] = [
            int(
                sum(
                    _relation_counts(doc_data).values()
                )
            )
            for _, doc_data in doc_items
        ]

    # General tree features
    data["tree_depth"] = [
        _rst_float(doc_data, "tree_depth")
        for _, doc_data in doc_items
    ]

    data["num_edus"] = [
        _rst_float(doc_data, "num_edus")
        for _, doc_data in doc_items
    ]

    # Nuclearity features
    nuclearity = [
        _nuclearity_counts(doc_data)
        for _, doc_data in doc_items
    ]

    nn_counts = [
        nn
        for nn, _, _ in nuclearity
    ]

    ns_counts = [
        ns
        for _, ns, _ in nuclearity
    ]

    sn_counts = [
        sn
        for _, _, sn in nuclearity
    ]

    if abs_nuclearity:
        data["nucl_NN"] = nn_counts
        data["nucl_NS"] = ns_counts
        data["nucl_SN"] = sn_counts

    nuc_totals = [
        nn + ns + sn
        for nn, ns, sn in nuclearity
    ]

    data["nucl_NN_relprop"] = _safe_div(
        nn_counts,
        nuc_totals,
    )

    data["nucl_NS_relprop"] = _safe_div(
        ns_counts,
        nuc_totals,
    )

    data["nucl_SN_relprop"] = _safe_div(
        sn_counts,
        nuc_totals,
    )

    # Engineered features
    if engineered:
        eng_features = add_engineered_features(
            rst_data_subset=rst_data_subset
        )
        data |= eng_features

    return data

def build_feature_matrix(
    pos_data: Dict[str, List[float]],
    neg_data: Dict[str, List[float]],
    features: Optional[Iterable[str]] = None,   # if None: use all overlapping keys
    label_col: str = "label",
    pos_label: int = 1,   # positive → 1
    neg_label: int = 0,   # negative → 0
) -> pd.DataFrame:
    """
    Create a single DataFrame of features with a numeric label column:
      1 for positive, 0 for negative.
    """
    # 1) decide feature set
    if features is None:
        feats = sorted(set(pos_data.keys()) & set(neg_data.keys()))
    else:
        feats = [f for f in features if f in pos_data and f in neg_data]
    if not feats:
        raise ValueError("No overlapping features to build the matrix from.")

    # 2) sanity: aligned lengths
    def _check_lengths(d: Dict[str, List[float]]) -> int:
        k0 = feats[0]; n = len(d[k0])
        for f in feats:
            if len(d[f]) != n:
                raise ValueError(f"Feature '{f}' len={len(d[f])} != '{k0}' len={n}.")
        return n
    _check_lengths(pos_data)
    _check_lengths(neg_data)

    # 3) build per-group frames
    df_pos = pd.DataFrame({f: pos_data[f] for f in feats})
    df_pos[label_col] = pos_label

    df_neg = pd.DataFrame({f: neg_data[f] for f in feats})
    df_neg[label_col] = neg_label

    # 4) concat
    df = pd.concat([df_pos, df_neg], ignore_index=True)

    # put 'label' label and 'meta' labels last (if they have been recieved from upstream)
    meta = ["doc_id", "ds", "ds_num"]
    safe_meta = [c for c in meta if c in df.columns]
    if safe_meta:
        all_meta = safe_meta + [label_col]
    else:
        all_meta = [label_col]
    cols = [c for c in df.columns if c not in all_meta] + all_meta
    
    return df[cols]

# ------------------------------------------------------------
# Extra RST-aware features computed from raw rst_data list
# ------------------------------------------------------------
def _entropy_from_props(p: Dict[str, float]) -> float:
    vals = np.array([v for v in p.values() if v > 0.0], dtype=float)
    if vals.size == 0:
        return 0.0
    return float(-(vals * np.log(vals)).sum())

def _top2_dom_from_props(p: Dict[str, float]) -> float:
    if not p:
        return 0.0
    arr = np.sort(np.array(list(p.values()), dtype=float))[::-1]
    if arr.size == 1:
        return float(arr[0])  # only one relation present
    return float(arr[0] - arr[1])

def add_engineered_features(rst_data_subset: Dict[str, Dict[str, Any]]) -> Dict[str, List[float]]:
    """
    Build engineered per-document feature columns from rst_data_subset.

    Returns a column-oriented dict aligned to rst_data_subset order:
      - depth_per_edu
      - rel_entropy
      - rel_top2_dom
      - edu_len_mean
      - edu_len_std
      - edu_len_p90
    """
    eng_features = {
        "depth_per_edu": [],
        "rel_entropy": [],
        "rel_top2_dom": [],
        "edu_len_mean": [],
        "edu_len_std": [],
        "edu_len_p90": [],
    }

    for per_doc_data in rst_data_subset.values():
        d = per_doc_data.get("rst_features", {}) or {}

        depth = float(d.get("tree_depth", 0.0))
        n_edus = max(1.0, float(d.get("num_edus", 1.0)))  # avoid /0
        props = d.get("relation_proportions", {}) or {}
        edus = d.get("edus", []) or []

        # depth per EDU
        depth_per_edu = depth / n_edus

        # relation entropy & dominance
        rel_entropy = _entropy_from_props(props)
        rel_top2_dom = _top2_dom_from_props(props)

        # EDU length stats (chars)
        lengths = np.array([len(e.strip()) for e in edus if isinstance(e, str)], dtype=float)
        if lengths.size == 0:
            edu_len_mean = edu_len_std = edu_len_p90 = 0.0
        else:
            edu_len_mean = float(lengths.mean())
            edu_len_std = float(lengths.std(ddof=0))
            edu_len_p90 = float(np.percentile(lengths, 90))

        eng_features["depth_per_edu"].append(depth_per_edu)
        eng_features["rel_entropy"].append(rel_entropy)
        eng_features["rel_top2_dom"].append(rel_top2_dom)
        eng_features["edu_len_mean"].append(edu_len_mean)
        eng_features["edu_len_std"].append(edu_len_std)
        eng_features["edu_len_p90"].append(edu_len_p90)

    return eng_features

# ------------------------------------------------------------
# Collapse rare relations in Xy feature matrix
# ------------------------------------------------------------
def collapse_rare_relations_df(
    Xy: pd.DataFrame,
    relation_cols: List[str],
    *,
    avg_prop_min: float = 0.01,   # drop relations with mean proportion < 1%
    other_col: str = "rel_OTHER"
) -> pd.DataFrame:
    """
    Takes your wide matrix (relations as proportion columns) and:
      - identifies rare relation columns (mean < threshold)
      - sums them into one 'other' column
      - drops the rare columns
    """
    Xy = Xy.copy()
    if not relation_cols:
        return Xy

    means = Xy[relation_cols].mean(axis=0)
    rare  = means.index[means < avg_prop_min].tolist()
    keep  = [c for c in relation_cols if c not in rare]

    if rare:
        Xy[other_col] = Xy.get(other_col, 0.0) + Xy[rare].sum(axis=1)
        Xy = Xy.drop(columns=rare)

    return Xy

# =============
# Helper to Analyze Distribution
# =============

def summarize_relation_distribution(
    df: pd.DataFrame,
    feature: str,
    group_col: str = "ds_num",
) -> pd.DataFrame:
    rows = []

    for group_value, group_df in df.groupby(group_col):
        values = pd.to_numeric(
            group_df[feature],
            errors="coerce",
        ).dropna()

        positive = values[values > 0]

        rows.append({
            "group": group_value,
            "n_documents": len(values),
            "n_present": len(positive),
            "present_pct": (values > 0).mean() * 100,
            "zero_pct": (values == 0).mean() * 100,
            "median_all": values.median(),
            "q25_all": values.quantile(0.25),
            "q75_all": values.quantile(0.75),
            "p90": values.quantile(0.90),
            "p95": values.quantile(0.95),
            "max": values.max(),
            "median_when_present": (
                positive.median() if not positive.empty else np.nan
            ),
            "q25_when_present": (
                positive.quantile(0.25)
                if not positive.empty else np.nan
            ),
            "q75_when_present": (
                positive.quantile(0.75)
                if not positive.empty else np.nan
            ),
        })

    return pd.DataFrame(rows)