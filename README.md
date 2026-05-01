# Identifying Depression in Essays via RST Features

> **🚧 Project Status: Active Refactoring (May 2026)**
> This repository is currently undergoing a refactor to upgrade the pipeline. 
> 
> * **Current State:** The preprocessing (`01_preprocessing_documents.ipynb`) and parsing modules (`02_parsing_with_rst.ipynb`, `src/discourse/rst.py`) have been updated to the new standard. 
> * **Legacy State:** The statistical analysis (`03_`) and ML modeling (`04_`) notebooks still contain legacy code from the previous iteration (late 2025). The pipeline cannot currently be run end-to-end until these notebooks are updated.

## Project Overview

### Scope and Objectives
This project explores whether features derived from Rhetorical Structure Theory (RST) parsing can help distinguish essays (and, later, other discourse) written by individuals diagnosed with depression from those written by non-depressed controls.

There are two hypotheses:
1.  **H1 (primary):** RST-level features—e.g., the distribution of rhetorical relations, the directionality of nucleus–satellite links, and the density/number of elementary discourse units (EDUs)—may carry diagnostic signal beyond surface lexical features. This area appears somewhat under-explored.
2.  **H2 (secondary):** Hybrid modeling—combining RST-based discourse features with more conventional linguistic features (lexical, syntactic, semantic)—may improve classification and provide insight into how depressive language differs in discourse organization, not only in vocabulary or style. This has been attempted (e.g., [this study](https://ieeexplore.ieee.org/abstract/document/10600701)); however, semantic features seem to have contributed most of "the lift."

**Goal:** Evaluate whether discourse-level features from RST parsers are useful on their own and/or as augmentations to standard NLP pipelines for mental-health detection.

**Data:** The dataset used here is a closed, annotated corpus of Russian essays. Due to licensing and confidentiality, the raw data is not distributed with this repository.

### Quick Primer: What is RST?
RST models how parts of a text relate to one another.
* **EDUs:** Minimal spans (often clauses) identified by an RST parser.
* **Relations:** Links between EDUs/spans (e.g., Elaboration, Contrast, Cause).
* **Nucleus–Satellite:** The nucleus carries central meaning; the satellite supports/modifies it.
* **Tree structure:** Relations compose recursively into a discourse tree for the whole text.

RST has been used to study coherence, argumentation, and writing quality. Here we test whether structural cues also reveal patterns distinctive of depressive writing.

### The Parser Used
The project relies on [this RST parser](https://github.com/tchewik/isanlp_rst) (with `gumrrg` and `rstreebank` models).

## What’s Been Done
### H1: Using RST Features (Alone) to Predict Diagnosis
Preliminary results (achieved using the legacy version of the pipeline) indicate that on a limited corpus, RST features alone can carry a predictive signal:
* Statistical analysis shows significant differences in the proportions of certain relations and nuclearity patterns between the `1` (positive) and `0` (negative) groups, with moderate effect sizes. This is notably pronounced when using the `gumrrg` model—an observation that warrants further investigation.
* Baseline ML pipelines demonstrate predictive capability using only RST features, achieving moderate F1 scores.
* Adding engineered features like `depth_per_edu` and `rel_entropy` yields marginal improvements to model performance.

### Pipeline (four phases; each in its own notebook under `/notebooks/`)

1.  **Preprocessing (✅ Refactored)**
    * Loads raw CSVs and restructures them into a nested dictionary: `{"corpus_name": {"doc-1": {"text": …, "ds": …, "ds_num": 0/1}, ...}}`.
    * This structure supports processing two or more separate corpora simultaneously.
    * The `text` key contains the raw essay string.
    * The `ds` key contains the associated diagnosis string.
    * The `ds_num` key contains the numerical label: `1` for positive diagnoses (e.g., "высокая депрессивность" or "депрессия") and `0` for negative (e.g., "нет депрессивности" or "здоровые").
    * Saves the structured output to `/data/interim/preprocessed_corpora.json` for downstream tasks.
    * *(Note: For privacy reasons, actual data is excluded. Synthetic data samples may be added later for demonstration).*

2.  **RST Parsing (✅ Refactored)**
    * Runs each document through the parser to extract RST features via `src/discourse/rst.py`.
    * Generates two separate feature databases (one using the `gumrrg` model and one using `rstreebank`).
    * The output mimics the Phase 1 structure, appending a new feature dictionary to each document: `{"doc-id": {"rst_features": {"tree_depth": int, "num_edus": int, "relation_counts": dict, …}, "ds": str, "ds_num": int}}`.

3.  **Brief Statistical Analysis (🚧 Legacy Code)**
    * *Note: Currently being updated to handle the new Phase 2 parser output structure.*
    * Reshapes features into one vector per document (for stats/ML).
    * Converts nuclearity counts to proportions.
    * Runs Mann–Whitney U tests with Cliff’s delta to surface potentially meaningful group differences (positive vs. negative).

4.  **ML on RST Features (🚧 Legacy Code)**
    * *Note: Awaiting the completion of the Phase 3 refactor.*
    * Builds an Xy table (features + label).
    * Runs baselines using L2/L1 Logistic Regression and HistGradientBoostingClassifier.
    * Engineers additional features and re-evaluates:
        - `depth_per_edu` = tree_depth / num_edus
        - `rel_entropy` = -∑ p_i log p_i (relation distribution entropy)
        - `rel_top2_dom` = p_top1 − p_top2 (dominance gap)
        - `edu_len_mean`, `edu_len_std`, `edu_len_p90` (EDU length stats via simple tokenization)

## Next Steps

### Immediate
* Augment RST data with semantic and other linguistic features (e.g., lexical diversity) to test H2 on the current corpus.
* Conduct a qualitative analysis of the diverging rhetorical relations (across both the `rstreebank` and `gumrrg` models) between the positive and control groups. The goal is to determine if these structural differences map to established psychological markers of depression.

### Tentative
* Evaluate both hypotheses on a larger, web-collected corpus to test model generalization.