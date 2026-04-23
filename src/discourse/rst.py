# === Imports & Constants ===
from collections import Counter
from typing import List, Dict, Tuple
import re
import unicodedata

_PARSER = None


# === PARSING ===

def _require_parser() -> None:
    if _PARSER is None:
        raise RuntimeError("Parser not initialized")


def init_parser(model: str = "tchewik/isanlp_rst_v3", version: str = "rstreebank") -> None:
    global _PARSER
    from isanlp_rst.parser import Parser
    _PARSER = Parser(hf_model_name=model, hf_model_version=version)


def _normalize_for_parser(s: str) -> str:
    s = s.replace("\u00A0", " ")        # NBSP -> space
    s = s.replace("\u200b", "")         # zero-width space
    s = s.replace("\ufeff", "")         # BOM
    s = re.sub(r"\r\n?", "\n", s)       # CRLF/CR -> LF
    s = unicodedata.normalize("NFC", s) # canonical compose
    return s.strip()


def parse_corpus(corpus: List[str]) -> Tuple[List[Dict | None], List[Dict]]:
    """
    Parse each document in the corpus once, without segmentation.
    Returns:
        parsed_corpus: list of parser outputs or None for failed docs
        errors: list of error dicts
    """
    _require_parser()
    parsed_corpus, errors = [], []

    for di, doc in enumerate(corpus):
        try:
            print(f"processing doc:{di}", flush=True)
            parsed_corpus.append(_PARSER(_normalize_for_parser(doc)))
        except Exception as e:
            errors.append({"doc_index": di, "error": str(e)})
            parsed_corpus.append(None)

    return parsed_corpus, errors


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


def extract_all_rst_features(parsed_corpus: List[Dict | None]):
    """
    Extract one feature dict per parsed document.
    """
    all_features = []

    for doc in parsed_corpus:
        if not isinstance(doc, dict) or not doc.get("rst"):
            continue

        tree = doc["rst"][0]
        all_features.append(_extract_rst_features(tree))

    all_relations = set(k for item in all_features for k in item["relation_counts"])
    return all_features, all_relations


def count_relations(sub_corpus_features: List[Dict]) -> Tuple[List[Tuple[str, int]], List[Tuple[str, float]]]:
    rel_counter = Counter()

    for item in sub_corpus_features:
        rel_counter.update(item.get("relation_counts", {}) or {})

    abs_counts = rel_counter.most_common()
    total = sum(rel_counter.values())

    if total == 0:
        proportions = [(rel, 0.0) for rel, _ in abs_counts]
    else:
        proportions = [(rel, cnt / total) for rel, cnt in abs_counts]

    return abs_counts, proportions