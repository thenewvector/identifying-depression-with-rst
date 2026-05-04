# restructure.py
from __future__ import annotations
from typing import List, Dict, Any, Optional, Iterable
import pandas as pd
import numpy as np

from typing import List, Dict, Any


def get_data_vectors(
    relations_list: List[str],
    rst_data_subset: Dict[str, Dict[str, Any]],
    abs_nuclearity: bool = False,
    include_meta: bool = True,
) -> Dict[str, List[Any]]:
    """
    Build per-document feature columns from an RST feature subset.

    Parameters
    ----------
    relations_list : list[str]
        Relation labels to extract from each document's
        'relation_proportions' dict.

    rst_data_subset : dict[str, dict[str, Any]]
        A per-document subset of the RST database, e.g.:

        {
            "doc-1": {
                "rst_features": {...},
                "ds": "pos",
                "ds_num": 1
            },
            "doc-2": {
                "rst_features": {...},
                "ds": "neg",
                "ds_num": 0
            },
            ...
        }

    abs_nuclearity : bool, default=False
        If True, include absolute nuclearity counts.

    include_meta : bool, default=True
        If True, include:
            - 'doc_id'
            - 'ds'
            - 'ds_num'

    Returns
    -------
    dict[str, list[Any]]
        A column-oriented dictionary aligned by document order.
    """
    
    doc_items = list(rst_data_subset.items())
    data: Dict[str, List[Any]] = {}

    def _rst_features(doc_data: Dict[str, Any]) -> Dict[str, Any]:
        return doc_data.get("rst_features", {}) or {}

    def _rst_float(doc_data: Dict[str, Any], key: str, default: float = 0.0) -> float:
        return float(_rst_features(doc_data).get(key, default))

    def _nuclearity_counts(doc_data: Dict[str, Any]) -> tuple[float, float, float]:
        npat = _rst_features(doc_data).get("nuclearity_patterns", {}) or {}
        return (
            float(npat.get("NN", 0.0)),
            float(npat.get("NS", 0.0)),
            float(npat.get("SN", 0.0)),
        )

    def _safe_div(nums: List[float], dens: List[float]) -> List[float]:
        return [(a / b) if b else 0.0 for a, b in zip(nums, dens)]

    if include_meta:
        data["doc_id"] = [doc_id for doc_id, _ in doc_items]
        data["ds"] = [doc_data.get("ds", "") for _, doc_data in doc_items]
        data["ds_num"] = [doc_data.get("ds_num", 0) for _, doc_data in doc_items]

    for rel in relations_list:
        data[rel] = [
            float((_rst_features(doc_data).get("relation_proportions", {}) or {}).get(rel, 0.0))
            for _, doc_data in doc_items
        ]

    data["tree_depth"] = [_rst_float(doc_data, "tree_depth") for _, doc_data in doc_items]
    data["num_edus"] = [_rst_float(doc_data, "num_edus") for _, doc_data in doc_items]

    nuclearity = [_nuclearity_counts(doc_data) for _, doc_data in doc_items]
    nn_counts = [nn for nn, _, _ in nuclearity]
    ns_counts = [ns for _, ns, _ in nuclearity]
    sn_counts = [sn for _, _, sn in nuclearity]

    if abs_nuclearity:
        data["nucl_NN"] = nn_counts
        data["nucl_NS"] = ns_counts
        data["nucl_SN"] = sn_counts

    nuc_totals = [nn + ns + sn for nn, ns, sn in nuclearity]
    data["nucl_NN_relprop"] = _safe_div(nn_counts, nuc_totals)
    data["nucl_NS_relprop"] = _safe_div(ns_counts, nuc_totals)
    data["nucl_SN_relprop"] = _safe_div(sn_counts, nuc_totals)

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

def extra_rst_features_from_raw(rst_data: List[dict]) -> pd.DataFrame:
    """
    rst_data: list of per-document dicts you already produce:
      { 'tree_depth', 'num_edus', 'relation_proportions', 'edus', ... }

    Returns a DataFrame with:
      depth_per_edu, rel_entropy, rel_top2_dom,
      edu_len_mean, edu_len_std, edu_len_p90
    """
    rows = []
    for d in rst_data:
        depth = float(d.get("tree_depth", 0.0))
        n_edus = max(1.0, float(d.get("num_edus", 1.0)))  # avoid /0
        props = d.get("relation_proportions", {}) or {}
        edus  = d.get("edus", []) or []

        # depth per EDU
        depth_per_edu = depth / n_edus

        # relation entropy & dominance
        rel_entropy  = _entropy_from_props(props)
        rel_top2_dom = _top2_dom_from_props(props)

        # EDU length stats (chars)
        lengths = np.array([len(e.strip()) for e in edus if isinstance(e, str)], dtype=float)
        if lengths.size == 0:
            edu_len_mean = edu_len_std = edu_len_p90 = 0.0
        else:
            edu_len_mean = float(lengths.mean())
            edu_len_std  = float(lengths.std(ddof=0))
            edu_len_p90  = float(np.percentile(lengths, 90))

        rows.append({
            "depth_per_edu": depth_per_edu,
            "rel_entropy": rel_entropy,
            "rel_top2_dom": rel_top2_dom,
            "edu_len_mean": edu_len_mean,
            "edu_len_std": edu_len_std,
            "edu_len_p90": edu_len_p90,
        })
    return pd.DataFrame(rows)

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