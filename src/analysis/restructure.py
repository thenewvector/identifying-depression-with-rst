# restructuredata.py
from __future__ import annotations
from typing import List, Dict, Any

__all__ = ["get_data_vectors"]

def get_data_vectors(relations_list: List[str], rst_data_subset: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    """
    Build per-document feature columns from RST feature dicts.

    Inputs
    ------
    relations_list : canonical list of relation labels (strings) to extract proportions for
    rst_data_subset : list of per-doc dicts with keys like:
        - 'relation_proportions' : {rel: float}
        - 'tree_depth'           : int/float
        - 'num_edus'             : int/float
        - 'nuclearity_patterns'  : {'NN': int, 'NS': int, 'SN': int}

    Returns
    -------
    dict mapping column_name -> list aligned to rst_data_subset. Columns include:
        - one column per relation label in `relations_list` (proportions)
        - 'tree_depth', 'num_edus'
        - nuclearity counts:    'nucl_NN', 'nucl_NS', 'nucl_SN'
        - nuclearity proportions over relations (sum≈1 if any relations in doc):
                                 'nucl_NN_relprop', 'nucl_NS_relprop', 'nucl_SN_relprop'
    """
    # --- relation proportions (keep original keys = relation labels)
    data: Dict[str, List[float]] = {}
    for rel in relations_list:
        col: List[float] = []
        for feat in rst_data_subset:
            rp = feat.get("relation_proportions", {}) or {}
            col.append(float(rp.get(rel, 0.0)))
        data[rel] = col

    # --- scalars
    data["tree_depth"] = [float(feat.get("tree_depth", 0.0)) for feat in rst_data_subset]
    data["num_edus"]   = [float(feat.get("num_edus",   0.0)) for feat in rst_data_subset]

    # --- nuclearity counts
    nn_counts: List[float] = []
    ns_counts: List[float] = []
    sn_counts: List[float] = []
    for feat in rst_data_subset:
        npat = feat.get("nuclearity_patterns", {}) or {}
        nn_counts.append(float(npat.get("NN", 0.0)))
        ns_counts.append(float(npat.get("NS", 0.0)))
        sn_counts.append(float(npat.get("SN", 0.0)))

    data["nucl_NN"] = nn_counts
    data["nucl_NS"] = ns_counts
    data["nucl_SN"] = sn_counts

    # --- nuclearity proportions over relations (preferred normalisation)
    rel_totals = [nn + ns + sn for nn, ns, sn in zip(nn_counts, ns_counts, sn_counts)]
    def _safe_div(nums: List[float], dens: List[float]) -> List[float]:
        return [(a / b) if b else 0.0 for a, b in zip(nums, dens)]

    data["nucl_NN_relprop"] = _safe_div(nn_counts, rel_totals)
    data["nucl_NS_relprop"] = _safe_div(ns_counts, rel_totals)
    data["nucl_SN_relprop"] = _safe_div(sn_counts, rel_totals)

    return data