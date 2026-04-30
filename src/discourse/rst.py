# === Imports & Constants ===
from collections import Counter
from typing import List, Dict, Tuple, Set, Any
import re
import unicodedata

_PARSER = None


# === PARSING ===

def _require_parser() -> None:
    if _PARSER is None:
        raise RuntimeError("Parser not initialized")


def init_parser(model: str = "tchewik/isanlp_rst_v3",
                version: str = "rstreebank",
                cuda: int = -1) -> None:
    global _PARSER
    from isanlp_rst.parser import Parser
    _PARSER = Parser(hf_model_name=model, hf_model_version=version, cuda_device=cuda)


def _normalize_for_parser(s: str) -> str:
    s = s.replace("\u00A0", " ")
    s = s.replace("\u200b", "")
    s = s.replace("\ufeff", "")
    s = s.replace("…", "...")
    s = s.replace("—", " - ")
    s = s.replace("–", " - ")
    s = s.replace("−", " - ")
    s = re.sub(r"(?<=\w)-(?=\w)", " - ", s)
    s = re.sub(r"\r\n?", "\n", s)
    s = unicodedata.normalize("NFC", s)
    return s.strip()

def parse_document(document: str, normalize: bool = True) -> Dict:
    """
    Parses a SINGLE document text using the RST parser.

    Args:
        document: The raw text string to be parsed.
        normalize: If True, cleans the text before parsing. This standardizes 
            whitespace, normalizes dashes and ellipses, fixes line breaks, 
            and applies Unicode NFC normalization.

    Returns:
        A dictionary containing the parsed RST tree and related metadata.
    """

    _require_parser()

    if normalize:
        document = _normalize_for_parser(document)
    return _PARSER(document)

# === EXTRACTION ===

def _extract_edus(node):
    if getattr(node, "left", None) is None and getattr(node, "right", None) is None:
        txt = getattr(node, "text", None)
        return [txt.strip()] if isinstance(txt, str) and txt.strip() else []

    edus = []
    if getattr(node, "left", None) is not None:
        edus += _extract_edus(node.left)
    if getattr(node, "right", None) is not None:
        edus += _extract_edus(node.right)
    return edus


def _extract_rst_features(tree):
    relation_counter = Counter()
    nuclearity_counter = Counter()

    def walk(node):
        if getattr(node, "left", None) is None and getattr(node, "right", None) is None:
            return 1

        rel = getattr(node, "relation", None)
        nuc = getattr(node, "nuclearity", None)

        if rel:
            relation_counter[rel] += 1
        if nuc:
            nuclearity_counter[nuc] += 1

        ld = walk(node.left) if getattr(node, "left", None) is not None else 0
        rd = walk(node.right) if getattr(node, "right", None) is not None else 0
        return max(ld, rd) + 1

    tree_depth = walk(tree)
    edus = _extract_edus(tree)
    num_edus = len(edus)
    total_rel = sum(relation_counter.values())
    relation_proportions = {k: v / total_rel for k, v in relation_counter.items()} if total_rel else {}

    return {
        "tree_depth": tree_depth,
        "num_edus": num_edus,
        "relation_counts": dict(relation_counter),
        "relation_proportions": relation_proportions,
        "nuclearity_patterns": dict(nuclearity_counter),
        "edus": edus,
    }


def extract_all_rst_features(parsed_corpus: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Set[str]]:
    """
    Extracts RST features and aggregates all unique relations across a parsed corpus.

    Args:
        parsed_corpus: A nested dictionary containing parser outputs and diagnoses.

    Returns:
        A tuple of (all_features, all_relations).
        
        all_features structure:
        {
            "doc_id": {
                "rst_features": {
                    "tree_depth": int,
                    "num_edus": int,
                    "relation_counts": dict,
                    "relation_proportions": dict,
                    "nuclearity_patterns": dict,
                    "edus": list
                },
                "ds": str,       # Text diagnosis (e.g., 'депрессия')
                "ds_num": int    # Numeric diagnosis (e.g., 1)
            }
        }
        
        all_relations structure:
        A set of all unique RST relations found across the entire corpus.
    """
   
    all_features = {}

    for doc_id, doc_data in parsed_corpus.items():
        if not isinstance(doc_data["parser_output"], dict) or not doc_data["parser_output"].get("rst"):
            continue

        tree = doc_data["parser_output"]["rst"][0]
        all_features[doc_id] = {
            "rst_features": _extract_rst_features(tree),
            "ds": doc_data["ds"],
            "ds_num": doc_data["ds_num"]
                                }

    all_relations = set()
    for doc_id, doc_data in all_features.items():
        all_relations.update(doc_data["rst_features"]["relation_counts"].keys())

    return all_features, all_relations


def count_relations(sub_corpus_features: Dict[str, Dict[str, Any]]) -> Tuple[List[Tuple[str, int]], List[Tuple[str, float]]]:
    rel_counter = Counter()

    for doc_data in sub_corpus_features.values():
        doc_rst = doc_data.get("rst_features", {})
        doc_rels = doc_rst.get("relation_counts", {})
    
        rel_counter.update(doc_rels)

    abs_counts = rel_counter.most_common()
    total = sum(rel_counter.values())

    if total == 0:
        proportions = [(rel, 0.0) for rel, _ in abs_counts]
    else:
        proportions = [(rel, cnt / total) for rel, cnt in abs_counts]

    return abs_counts, proportions