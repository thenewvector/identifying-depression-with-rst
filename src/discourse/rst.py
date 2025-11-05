# === Imports & Constants ===
from collections import Counter
import math

_PARSER = None

# === PARSING ===

# === Require pareser ===
def _require_parser() -> None:
    if _PARSER is None:
        raise RuntimeError("Parser not initialized")

# -----------------------
# Public: init the parser
# -----------------------
def init_parser(model: str='tchewik/isanlp_rst_v3', version: str='gumrrg') -> None:
    global _PARSER
    
    from isanlp_rst.parser import Parser
    _PARSER = Parser(hf_model_name=model, hf_model_version=version)


# === A helper to 'normalize' all documents to be list items (just in case)

# By default, the code expects the text either as one solid chunk or as two or more chunks
# in case the text has been segmented in the previous phase (segmentation)

def _as_segments(x):
    # normalize each item in corpus to a list[str] for cases where the structure of the corpus may be like
    # ["text1", ["seg1_of_text2", "seg2_of_text2"], "text3", ... ]
    # this is just an extra precaution
    
    if isinstance(x, str):
        return [x]
    return [s.strip() for s in x if isinstance(s, str) and s.strip()]

# === A helper to 'normalize' the texts just in case
import re, unicodedata

def _normalize_for_parser(s: str) -> str:
    s = s.replace("\u00A0", " ")        # NBSP -> space
    s = s.replace("\u200b", "")         # zero-width space
    s = s.replace("\ufeff", "")         # BOM
    s = re.sub(r"\r\n?", "\n", s)       # CRLF/CR -> LF
    s = unicodedata.normalize("NFC", s) # canonical compose
    return s.strip()

# ================================
# Public: run the whole corpus through the rst parser
# ================================

def parse_corpus(corpus: list) -> tuple[list[list], list[dict]]:
    _require_parser()
    parsed_corpus, errors = [], []
    for di, doc in enumerate(corpus):
        segments = _as_segments(doc)
        parsed_segments = []
        for si, seg in enumerate(segments):
            try:
                print(f"processing doc:{di}, segment:{si}", flush=True)
                parsed_segments.append(_PARSER(_normalize_for_parser(seg)))
            except Exception as e:
                errors.append({"doc_index": di, "seg_index": si, "error": str(e)})
                parsed_segments.append(None)
        parsed_corpus.append(parsed_segments)
    return parsed_corpus, errors

# === EXTRACTION ====

# === Basic helper to extract EDUs ===
def _extract_edus(node):
    # leaf?
    if getattr(node, 'left', None) is None and getattr(node, 'right', None) is None:
        txt = getattr(node, 'text', None)
        return [txt.strip()] if isinstance(txt, str) and txt.strip() else []
    edus = []
    if getattr(node, 'left', None) is not None:
        edus += _extract_edus(node.left)
    if getattr(node, 'right', None) is not None:
        edus += _extract_edus(node.right)
    return edus

# === Main helper function to extract RST features ===

def _extract_rst_features(tree):
    relation_counter = Counter()
    nuclearity_counter = Counter()

    def walk(node):
        # leaf
        if getattr(node, 'left', None) is None and getattr(node, 'right', None) is None:
            return 1
        rel = getattr(node, 'relation', None)
        nuc = getattr(node, 'nuclearity', None)
        if rel: relation_counter[rel] += 1
        if nuc: nuclearity_counter[nuc] += 1

        ld = walk(node.left)  if getattr(node, 'left',  None) is not None else 0
        rd = walk(node.right) if getattr(node, 'right', None) is not None else 0
        return max(ld, rd) + 1

    tree_depth = walk(tree)
    edus = _extract_edus(tree)
    num_edus = len(edus)
    total_rel = sum(relation_counter.values())
    relation_proportions = {k: v / total_rel for k, v in relation_counter.items()} if total_rel else {}

    return {
        'tree_depth': tree_depth,
        'num_edus': num_edus,
        'relation_counts': dict(relation_counter),
        'relation_proportions': relation_proportions,
        'nuclearity_patterns': dict(nuclearity_counter),
        'edus': edus,
    }

# === A helper function to merge all the features for the texts that had to be segmented
def _merge_rst_features(feature_list, *, 
                        merge_strategy: str = "balanced", 
                        link_label: str = "joint", 
                        link_nuclearity: str = "NN"):
    total_chunks = len(feature_list)
    merged_relation_counts = Counter()
    merged_nuclearity_patterns = Counter()
    max_depth = 0
    total_edus = 0
    all_edus = []

    for f in feature_list:
        merged_relation_counts += Counter(f['relation_counts'])
        merged_nuclearity_patterns += Counter(f['nuclearity_patterns'])
        max_depth = max(max_depth, f['tree_depth'])
        total_edus += f['num_edus']
        all_edus.extend(f['edus'])

    # how many binary merges to connect k roots?
    merges = max(total_chunks - 1, 0)
    if merge_strategy == "balanced":
        added_depth = int(math.ceil(math.log2(total_chunks))) if total_chunks > 1 else 0
    elif merge_strategy == "chain":
        added_depth = merges
    else:
        added_depth = 0  # "none"

    # add synthetic inter-segment relations (joint, NN) → k-1
    if merges > 0:
        merged_relation_counts[link_label] += merges
        merged_nuclearity_patterns[link_nuclearity] += merges

    total_rel = sum(merged_relation_counts.values())
    relation_proportions = {k: v / total_rel for k, v in merged_relation_counts.items()} if total_rel else {}

    return {
        'tree_depth': max_depth + added_depth,
        'num_edus': total_edus,
        'relation_counts': dict(merged_relation_counts),
        'relation_proportions': relation_proportions,
        'nuclearity_patterns': dict(merged_nuclearity_patterns),
        'edus': all_edus,
    }

# -----------------------
# Public: Main Pipline function
# to process the whole corpus and extract all the features
# -----------------------

def extract_all_rst_features(parsed_corpus: list[list]):
    
    all_features = []

    for doc in parsed_corpus:
        # doc is always a list of segment results (even if len==1)
        seg_results = [seg for seg in doc if isinstance(seg, dict) and seg.get('rst')]
        if not seg_results:
            continue  # or append a sentinel

        per_seg_feats = []
        for seg in seg_results:
            tree = seg['rst'][0]
            per_seg_feats.append(_extract_rst_features(tree))

        if len(per_seg_feats) == 1:
            all_features.append(per_seg_feats[0])
        else:
            all_features.append(_merge_rst_features(
                per_seg_feats,
                merge_strategy="balanced",   # or "chain" if you prefer worst-case
                link_label="joint",
                link_nuclearity="NN"
            ))
    all_relations = set(k for item in all_features for k in item["relation_counts"])
    return all_features, all_relations
        
# === FOLLOW-UP ===