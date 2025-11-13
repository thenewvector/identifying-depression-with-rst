# Identifying Depression in Essays via RST Features

## Project Overview

### Scope and Objectives
This project explores whether features derived from Rhetorical Structure Theory (RST) parsing can help distinguish essays (and, later, other discourse) written by individuals diagnosed with depression from those written by non-depressed controls.

There are two hypotheses:
	1.	H1 (primary): RST-level features — e.g., the distribution of rhetorical relations, the directionality of nucleus–satellite links, and the density/number of elementary discourse units (EDUs) — may carry diagnostic signal beyond surface lexical features. This area appears somewhat under-explored.
	2.	H2 (secondary): Hybrid modeling — combining RST-based discourse features with more conventional linguistic features (lexical, syntactic, semantic) — may improve classification and provide insight into how depressive language differs in discourse organization, not only in vocabulary or style. This has been attempted (e.g., [this study](https://ieeexplore.ieee.org/abstract/document/10600701)); however, semantic features seem to contributed most of "the lift".

Goal: evaluate whether discourse-level features from RST parsers are useful on their own and/or as augmentations to standard NLP pipelines for mental-health detection.

Data. The dataset used here is a closed, annotated corpus of Russian essays. Due to licensing and confidentiality, the data is not distributed with this repository.

### Quick Primer: What is RST?
RST models how parts of a text relate to one another.
* EDUs: minimal spans (often clauses) identified by an RST parser.
* Relations: links between EDUs/spans (e.g., Elaboration, Contrast, Cause).
* Nucleus–Satellite: the nucleus carries central meaning; the satellite supports/modifies it.
* Tree structure: relations compose recursively into a discourse tree for the whole text.

RST has been used to study coherence, argumentation, and writing quality. Here we test whether structural cues also reveal patterns distinctive of depressive writing.

### The Parser Used
The project relies on [this RST parser](https://github.com/tchewik/isanlp_rst) (with `gumrrg` and `rstreebank` models).

## What’s Been Done
### H1: Using RST Features (Alone) to Predict Diagnosis
Preliminary result: on a limited corpus, RST features alone can to some extent predict the diagnosis associated with a text.

### Pipeline (four phases; each in its own notebook under /notebooks/)
1.	Preprocessing
    * Load raw CSVs and restructure into a dictionary: `{corpus_name: [doc1, doc2, …]}`.
	* Extract labels to a parallel dictionary and convert to 0/1 (`0` = negative, `1` = positive).
	* (Legacy/optional) Segmentation helpers in `src/discourse/segment.py`.
    * By default, do not segment for RST: the parser handles long texts via a sliding window and builds a single tree per document.
    * The transformers 512-token warning is expected and benign here.
2.	RST Parsing
    * Run each document through [the parser](https://github.com/tchewik/isanlp_rst) and extract features: relation counts/proportions, tree depth, number of EDUs, nuclearity patterns.
	* Notebook calls functions from `src/discourse/rst.py`.
	* Split corpora into positive/negative subsets for exploratory summaries (raw and relative relation counts).
    * Persist all RST outputs under a nested dictionary keyed by corpus name for downstream analysis.
3.	Brief Statistical Analysis
    * Reshape features into one vector per document (for stats/ML).
    * Convert nuclearity counts to proportions.
    * Run Mann–Whitney U tests with Cliff’s delta to surface potentially meaningful group differences (positive vs negative).
4.	ML on RST Features
    * Build an Xy table (features + label).
    * Baseline runs with L2 and L1 Logistic Regression and HistGradientBoostingClassifier.
    * Add a few engineered features, then re-run the same pipelines:
        - `depth_per_edu` = tree_depth / num_edus
        - `rel_entropy` = -∑ p_i log p_i (relation distribution entropy)
        - `rel_top2_dom` = p_top1 − p_top2 (dominance gap)
        - `edu_len_mean`, `edu_len_std`, `edu_len_p90` (EDU length stats via simple tokenization)

## Next (Tentatively)
    * Augment RST with semantic and other linguistic features (e.g., lexical diversity) to test H2 on the current corpus.
    * Test both hypotheses on a larger, web-collected corpus.