# Identifying Depression in Essays via RST Features

> **🚧 Project Status: Active Refactoring (June 2026)**
> This repository is currently undergoing a refactor to upgrade the pipeline. 
> 
> * **Current State:** The preprocessing (`01_preprocessing_documents.ipynb`), parsing modules (`02_parsing_with_rst.ipynb`, `src/discourse/rst.py`) and statistical analyses (`03_analyzing_stats.ipynb`, `src/analysis/statistics.py`) have been updated to the new standard.
> * **Legacy State:** The ML modeling (`04_`) notebooks still contain legacy code from the previous iteration (late 2025). The pipeline cannot currently be run end-to-end until these notebooks are updated.

## Project Overview

### Scope and Objectives

This project explores whether features derived from Rhetorical Structure Theory (RST) parsing can help distinguish essays (and, later, other discourse) written by individuals diagnosed with depression from those written by non-depressed controls.

There are two hypotheses:
1.  **H1 (primary):** RST-level features—e.g., the distribution of rhetorical relations, the directionality of nucleus–satellite links, and the density/number of elementary discourse units (EDUs)—may carry diagnostic signal beyond surface lexical features. This area appears somewhat under-explored.
2.  **H2 (secondary):** Hybrid modeling—combining RST-based discourse features with more conventional linguistic features (lexical, syntactic, semantic)—may improve classification and provide insight into how depressive language differs in discourse organization, not only in vocabulary or style. This has been attempted (e.g., [this study](https://ieeexplore.ieee.org/abstract/document/10600701)); however, semantic features seem to have contributed most of "the lift."

**Goal:** Evaluate whether discourse-level features from RST parsers are useful on their own and/or as augmentations to standard NLP pipelines for mental-health detection.

**Data:** The dataset used here is a closed, annotated corpus of Russian essays. Due to licensing and confidentiality, the raw data is not distributed with this repository.

### Quick Primer: What is RST?

RST (Rhetorical Structure Theory) models how parts of a text relate to one another.
* **EDUs:** Minimal spans (often clauses) [identified](https://github.com/thenewvector/field-notes/blob/main/03-concepts/rst/construing-edus-an-sfl-perspective-on-rst-parsing.md) by an RST parser.
* **Relations:** Links between EDUs/spans (e.g., Elaboration, Contrast, Cause).
* **Nucleus–Satellite:** The nucleus carries central meaning; the satellite supports/modifies it.
* **Tree structure:** Relations compose recursively into a discourse tree for the whole text.

See a more comprehensive summary of Rhetorical Structure Theory [here](https://github.com/thenewvector/field-notes/blob/main/03-concepts/rst/rhetorical-structure-theory.md) and my discussion of An SFL Perspective on RST Parsing [here](https://github.com/thenewvector/field-notes/blob/main/03-concepts/rst/construing-edus-an-sfl-perspective-on-rst-parsing.md).

RST has been used to study coherence, argumentation, and writing quality. Here we test whether structural cues also reveal patterns distinctive of depressive writing.

### The Parser Used

The project relies on [this RST parser](https://github.com/tchewik/isanlp_rst) (with `gumrrg` and `rstreebank` models).

## What’s Been Done (in the Previous Iteration)

### H1: Using RST Features (Alone) to Predict Diagnosis

Preliminary results (achieved using the legacy version of the pipeline) indicate that on a limited corpus, RST features alone can carry a predictive signal:
* Statistical analysis shows significant differences in the proportions of certain relations and nuclearity patterns between the `1` (positive) and `0` (negative) groups, with moderate effect sizes. This is notably pronounced when using the `gumrrg` model—an observation that warrants further investigation.
* Baseline ML pipelines demonstrate predictive capability using only RST features, achieving moderate F1 scores.
* Adding engineered features like `depth_per_edu` and `rel_entropy` yields marginal improvements to model performance.

## The Pipeline (four phases; each in its own notebook under `/notebooks/`)

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

3.  **Brief Statistical Analysis (✅ Refactored)**
    * Reshapes features into one vector per document (for stats/ML).
    * Converts nuclearity counts to proportions.
    * Runs Mann–Whitney U tests with Cliff’s delta to surface potentially meaningful group differences (positive vs. negative).
    * Adds the results of statistical analysis to [reports](https://github.com/thenewvector/identifying-depression-with-rst/tree/main/reports)

4.  **ML on RST Features (🚧 Legacy Code)**
    * *Note: Being refactored.*
    * Builds an Xy table (features + label).
    * Runs baselines using L2/L1 Logistic Regression and Histad GradientBoostingClassifier.
    * Engineers additional features and re-evaluates:
        - `depth_per_edu` = tree_depth / num_edus
        - `rel_entropy` = -∑ p_i log p_i (relation distribution entropy)
        - `rel_top2_dom` = p_top1 − p_top2 (dominance gap)
        - `edu_len_mean`, `edu_len_std`, `edu_len_p90` (EDU length stats via simple tokenization)

## The Research

### What's Been Done (Both Iterations)

* RST features have been extracted from the corpus using the `gumrrg` and `rstreebank` models
* Looking at the most "surface level" data for the `1` (positive) and `0` (negative) groups, it can be seen that the proportions of certain relations differ between the groups, with the difference being more pronounced when the `gumrrg` model is used. The exact figures can be found in the [reports directory](./reports).
* Looking at the "more refined" data of the statistical analysis, significant differences in the **proportions of certain relations**, **nuclearity patterns**, and **tree depth figures** can be seen between the `1` (positive) and `0` (negative) groups (with moderate effect sizes). This, as expected, is also more pronounced when using the `gumrrg` model. A preliminary interpretation of these differences is available [here](./reports/preliminary-report-cross-model-rst-depr-rst.md).
* Based on the preliminary interpretation, more refined variations of H1 and H2 can be put forth for further testing.
    * H1A: The tendencies revealed in the preliminary interpretation (more depth and more pragmatic/"argumentative" relations in group 0 as opposed to less depth and more semantic/sequential logic in group 1) are going to be reflected in the "engineered" features as well. This warrants adding these engineered features earlier in the pipeline, so they can be analyzed simultaneously with all the other features at the stage of statistical analysis.
    * H2A: The tendency towards a more rhetorically involved discourse with group zero may be maintained in the domain of other language features (e.g. lexical diversity).

### Next Steps

#### Immediate

* Add/move the incorporation of "engineered features" earlier in the pipeline, so they can be analyzed simultaneously with all the other features at the stage of statistical analysis (`03_analyzing_stats.ipynb`).
* Since differences are observed between the results of the two models (`gumrrg` and `rstreebank`), a brief theoretical overview of "what goes on under the hood" with these models (how EDUs are segmented, what relation inventories are used, what kind of corpora was used in the training of models) is called for and is going to be published [here](https://github.com/thenewvector/field-notes/blob/main/03-concepts/rst/) when it's ready.
* The same fact of there being such drastic differences between the results of the two models that have been used so far, necessitates that a third model (`unirst` in conjunction with the GUM inventory of relations) be used, to see if similar differences will be registered using this third model and whether our preliminary interpretation and H1A can be substantiated. 

#### Later

* Augment RST data with semantic and other linguistic features (e.g., lexical diversity) to test H2 (and H2A) on the current corpus.
* Conduct a qualitative analysis of the diverging rhetorical relations (across both the `rstreebank` and `gumrrg` models) between the positive and control groups. The goal is to determine if these structural differences map to established psychological markers of depression.

#### Tentative

* Evaluate both hypotheses on a larger, web-collected corpus to test model generalization.