# Identifying Depression in Essays via RST Features

## Project Overview
### Scope and Objectives
This project explores whether features derived from Rhetorical Structure Theory (RST) parsing can help distinguish essays written by individuals diagnosed with depression from those written by non-depressed controls.

The central hypothesis is twofold:
1.	RST-level features — such as the distribution of rhetorical relations, the directionality of nucleus–satellite links, and the density/number of elementary discourse units (EDUs) — may carry diagnostic signal beyond surface lexical features.
2.	Hybrid modeling — combining RST-based discourse features with more conventional linguistic features (lexical, syntactic, semantic) — may improve the reliability of classification and provide richer insight into how depressive language differs in its discourse organization, not only in its vocabulary or style.

Ultimately, the goal is to evaluate whether discourse-level features extracted from RST parsers can meaningfully be used on their own and/or to augment standard NLP pipelines for mental health detection.

The dataset used in this iteration of the project is a closed, annotated corpus of essays in Russian. Due to licensing and confidentiality constraints, the data is not distributed with this repository.

### Cliff Notes: What is RST?
Rhetorical Structure Theory (RST) is a framework from discourse linguistics that models how parts of a text relate to one another.

Key ideas:
* Elementary Discourse Units (EDUs): the minimal spans of text (often clauses) identified by an RST parser.
* Relations: links between EDUs (or groups of EDUs) that capture their rhetorical function — e.g., Elaboration, Contrast, Cause.
* Nucleus–Satellite distinction: in many relations, one span (the nucleus) carries the central meaning, while the satellite provides supporting or modifying information.
* Tree structure: The relations combine recursively to form a discourse tree, representing the global organization of the text.

RST has been used to study coherence, argumentation, and writing quality. Here, we investigate whether its structural cues might also reveal patterns distinctive of depressive writing.

## What's Been Done
### Segmentation Phase
A crucial preprocessing step is segmenting essays into chunks short enough to be processed by BERT-like RST parsers, which typically have a maximum input length of 512 tokens.

Our segmentation pipeline works as follows:
* Sentence splitting: Text is divided into sentences using either NLTK’s punkt tokenizer or a user-provided splitter.
* Token budget enforcement: Sentences (and -- potentially -- rare, very long words) are checked against the model’s tokenization budget (512* tokens including the special tokens like CLS and EOF). "Rogue cases" of single sentences that exceed this budget are split.

>\* Update 1: had to bump it down from 512 to a saf*er* 500, because the tokenizer of the RST parser apparently tokenizes differently and some resulting segments, though below the 512 budget with this tokenizer here (from the "sberbank-ai/sbert_large_nlu_ru" model), exceeded the 512 limit downstream when processed by the RST parser.
>\* Update 2: had to lower the token bugget a few more times to a safe 360, which results in segments that don't cause the parser downstream to panic (over spans exceeding the 512-token limit)

* Semantic splitting: Longer passages are recursively split at low-similarity valleys (based on sentence embeddings), so that resulting chunks are both under the token limit and semantically coherent.
* Output: A list of text segments, each safely processable by downstream RST parsers.

This segmentation module is deliberately designed to be robust — capable of handling pathological cases (e.g., URLs, excessively long run-on sentences) — while still aiming to preserve discourse integrity as much as possible.

The implementation is contained in /src/discourse/segment.py, which exposes two public functions:
* one for initializing the embedding model,
* another for running texts through the full segmentation pipeline.

Experimentation, configuration, and tweaking are coordinated from /notebooks/segment_documents.ipynb.

#### Future Plans for Segmentation Module
* Add the option to use the same tokenizer for segmentation that is going to be used for RST parsing downstream in the segmentation phase, so we don't run into the issue of chunks exceeding the capacity of the BERT models under the hood of the RST parser (see above).
* Semantic sentence splitting: Improve handling of very long sentences by incorporating semantic criteria, rather than the current mechanical approach of chopping at the 512-token boundary (or any other boundary, like the 'safe' 500-token threshold used now).
* Generality: Extend the module into a more universal semantic segmentation tool, applicable to a wider range of tasks beyond enforcing the a certain token limit.
* RST-driven segmentation (stretch goal): Explore integration with RST parsers to segment texts at the level of elementary discourse units (EDUs). This would allow more linguistically grounded splitting, where chunk boundaries are guided by rhetorical structure rather than token counts alone.

## Next Steps
### Working Right Now (in the dev branch)
- [x] Prep a very basic pipeline (in a notebook) to run RST parsing on (all) the segmented texts in the corpus/corpora.
- [ ] Complete the module extracting all the RST features to be used for analysis downstream (relations, directionality, EDU counts) and run the parser using the public functions of the module.
- [ ] Extract all the RST features from the essays in the corpus/corpora.

### On the Radar
* Evaluate predictive power of these features in identifying depressive essays.
* Combine with traditional linguistic features for joint modeling.