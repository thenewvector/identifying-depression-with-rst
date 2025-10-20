# === Imports & Constants
from collections import Counter

_PARSER = None

# === Require pareser ===
def _require_parser() -> None:
    if _PARSER == None:
        raise RuntimeError("Parser hnot initialized")

# -----------------------
# Public: init the parser
# -----------------------
def init_parser(model: str='tchewik/isanlp_rst_v3', version: str='gumrrg') -> None:
    global _PARSER
    
    from isanlp_rst.parser import Parser
    _PARSER = Parser(hf_model_name=model, hf_model_version=version)

def _extract_edus(node):
    if node.left is None and node.right is None:
        return [node.text.strip()] if node.text else []
    
    edus = []
    
    if node.left is not None:
        edus += _extract_edus(node.left)
    if node.right is not None:
        edus += _extract_edus(node.right)
    return edus

# ===Main function to extract the features

def _extract_rst_features(tree):
    relation_counter = Counter()
    nuclearity_counter = Counter()

    def walk(node, depth=1):
        if node.left is None and node.right is None:
            return 1  # it's a leaf

        relation_counter[node.relation] += 1
        nuclearity_counter[node.nuclearity] += 1

        left_depth = walk(node.left, depth + 1) if node.left else 0
        right_depth = walk(node.right, depth + 1) if node.right else 0

        return max(left_depth, right_depth) + 1

    tree_depth = walk(tree)
    edus = _extract_edus(tree)
    num_edus = sum(1 for _ in edus)

    total_relations = sum(relation_counter.values())
    relation_proportions = {
        k: v / total_relations for k, v in relation_counter.items()
    } if total_relations > 0 else {}

    return {
        'tree_depth': tree_depth,
        'num_edus': num_edus,
        'relation_counts': dict(relation_counter),
        'relation_proportions': relation_proportions,
        'nuclearity_patterns': dict(nuclearity_counter),
        'edus': edus,
    }

# === A helper function to merge all the features for the texts that had to be segmented

def _merge_rst_features(feature_list):
    # Initialize merged structures
    total_chunks = len(feature_list)
    merged_relation_counts = Counter()
    merged_nuclearity_patterns = Counter()
    max_depth = 0
    total_edus = 0
    all_edus = []

    for features in feature_list:
        merged_relation_counts += Counter(features['relation_counts'])
        merged_nuclearity_patterns += Counter(features['nuclearity_patterns'])
        max_depth = max(max_depth, features['tree_depth'])
        total_edus += features['num_edus']
        all_edus.extend(features['edus'])

    total_relations = sum(merged_relation_counts.values())
    relation_proportions = {
        k: v / total_relations for k, v in merged_relation_counts.items()
    } if total_relations > 0 else {}

    return {
        'tree_depth': max_depth + (total_chunks - 1),  # simulate merges
        'num_edus': total_edus,
        'relation_counts': dict(merged_relation_counts),
        'relation_proportions': relation_proportions,
        'nuclearity_patterns': dict(merged_nuclearity_patterns),
        'edus': all_edus,
    }

def _count_relations(rst_features_segment):
    relation_counter = Counter()
    for item in rst_features_segment:
        relation_counter.update(item["relation_counts"])

    return relation_counter.most_common()

# -----------------------
# Public: Main Pipline function to process the whole corpus and extract the features
# -----------------------

def extract_all_rst_features(corpus: list[str]):
    all_features = []
    parsed_corpus = []

    for i in corpus:
        if len(i) > 1:
            interim = [_PARSER(f) for f in i]
            parsed_corpus.append(interim)
        else:
            interim = _PARSER(" ".join(i))
            parsed_corpus.append(interim)


    for item in parsed_corpus:
        if isinstance(item, dict):  # unsplit, single text
            tree = item['rst'][0]
            features = _extract_rst_features(tree)
            all_features.append(features)

        elif isinstance(item, list):  # split, multiple segments
            segment_features = []
            for segment in item:
                tree = segment['rst'][0]
                segment_features.append(_extract_rst_features(tree))
            combined = _merge_rst_features(segment_features)
            all_features.append(combined)

        else:
            raise ValueError("Unrecognized format in parsed_corpus")

    all_relations = set(k for item in all_features for k in item["relation_counts"])

    return all_features


# === Transform the diagnoses (now stored as real labels in the 'diagnoses' variable) into ML appropriate labels

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(diagnoses)

rst_features_dep = []
rst_features_ok = []

for i in range(len(rst_features)):
    if y_encoded[i] == 0:
        rst_features_dep.append(rst_features[i])
    else:
        rst_features_ok.append(rst_features[i])

relation_counts_dep = count_relations(rst_features_dep)
relation_counts_ok = count_relations(rst_features_ok)

def convert_abs_rel_counts(relation_counts):
    total = 0
    
    relative_rels = []
    for item in relation_counts:
        total += int(item[1])
    
    for item in relation_counts:
        interim_rels = (item[0], item[1]/total)
        relative_rels.append(interim_rels)
    return relative_rels

relative_relation_counts_dep = convert_abs_rel_counts(relation_counts_dep)
relative_relation_counts_ok = convert_abs_rel_counts(relation_counts_ok)

causal_rel_props_dep = [item['relation_proportions'].get("causal", 0.0) for item in rst_features_dep]
causal_rel_props_ok = [item['relation_proportions'].get("causal", 0.0) for item in rst_features_ok]