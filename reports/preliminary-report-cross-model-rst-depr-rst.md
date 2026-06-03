**Date:** 2026-06-03 | **Version:** 1.0

**Tags:** #rst #sfl #discourse #sem-segm #depr-rst

# Preliminary Report: Cross-Model RST Comparison for the Depr-RST Project

## On the Surface

As we can see, both from the very surface level comparison of proportions (see below) of certain relations between the two groups and the data resulting from the statistical analysis the two models are **not telling the same story at the level of raw relation names**.

### `rstreebank`: only 5 FDR-significant features

- `Solutionhood` higher in Group-0
- `Condition` higher in Group-0
- `Sequence` higher in Group-1
- `tree_depth` higher in Group-0
- `Evidence` higher in Group-0

### `gumrrg`: a much stronger signal

Significant in Group-0:

- `explanation`
- `topic`
- `contingency`
- `attribution`
- `elaboration`
- `same-unit`
- `tree_depth`
- `nucl_NS_relprop`
- `nucl_SN_relprop`
- `num_edus`

Significant in Group-1:

- `causal`
- `joint`
- `nucl_NN_relprop`
- `context`

So the first hard conclusion is: `gumrrg` is more contrastive on this dataset than `rstreebank`.

### What actually replicates cleanly, though?

`tree_depth` seems to be the cleanest replicated finding.

- `rstreebank`: Group-1 median `11`, Group-0 median `12`
- `gumrrg`: Group-1 median `12`, Group-0 median `13`

So in both models: Group-0 texts are more hierarchically deep. And that is your safest cross-model result.

## In-Depth Look and Interpretation

### What replicates at "the family level", not the exact label level for Group 0

A first feature of the negative group is that the discourse here seems to be slightly more prone to indicating **conditions and contingencies**, which is verified but these features:

- `Condition` in rstreebank
- `contingency` in gumrrg

Both in the RRT and the GUM inventories, these are similar (if not identical) relations. This is a relation whereby the satellite gives a hypothetical or future dependency under which whatever is stated in the nucleus happens. (The best examples of this relation is what goes under the label "Conditional Sentences" in EFL textbooks.) These relations are technically in the "semantic" category (or — in other words — the relations that more or less just register what's "out there" and are not about the author being more "argumentative"), but intuitively they still the less semantic than other relations in the category.

One piece of evidence that can be used to support the latter claim is that according to Martin and Rose (2007), the relation of **condition** is fundamentally tied to the modal meaning of **probability**. Because modality is an _interpersonal_ resource used to grade the space between "yes" and "no" to open up room for negotiation, setting up a condition inherently involves the author's subjective evaluation of an outcome's likelihood. Therefore, it functions less as an objective ("semantic") report of facts and more as an interactive, argumentative device. In other words, while parsers classify `Condition/contingency` as a "semantic" (or "Subject Matter") relation, its reliance on hypotheticality, probability, and modal assessment makes it a highly _interpersonal_ rhetorical device.

Another feature is that these essays lean more towards **explanatory and justificatory logic** (as in giving "facts" to substantiate their subjective-ish claims). This can be deduce by slightly higher medians for:

- `Evidence` in rstreebank
- `explanation` in gumrrg

Since `exlanation` in the GUM inventory is a "top-level" feature that encompasses explanation-evidence (which should be more or less the same as Evidence from the RRT inventory), explanation-justify, and explanation-motivation, we have no clear way of telling whether this group is mostly represented by the "evidence" or whether it's a mix of the three subtypes. We might want to look at the specific spans exemplifying this relation in the actual corpus to either refute or corroborate either of the hypotheses.

One thing may be assumed with some confidence, though, is that these relations belong to the "pragmatic" (Stede et. al, 2017) or "presentational" relations (GUM), and the essays in the group look like they lean more into the explanatory and justificatory logic (geared more towards affecting the recipient by putting "a spin" on the facts, rather then just registering what's "out there".)

Not identical labels, but clearly the same neighborhood.

Another tentative feature of the essays in the negative group is that they come off as **slightly more elaborated or structurally expanded**. This conclusion is based on the statistically significant prominence of such features as:
    
    - `tree_depth` in both
    - `elaboration` in gumrrg
    - `num_edus` in gumrrg
    - `same-unit` in gumrrg

This suggestion can not be deduced by just looking at a slightly higher tree depth numbers for the negative group alone, but at all the features listed above together. So, yes, tree depth, may indicate a slightly more elaborate logic of discourse. An so may elaboration, which in GUM, encompasses two subcategories elaboration-attribute (when S elaborates on a single participant/entity within N, like a defining or a non-defining relative clause would do) and elaboration-additional (S elaborates on N as a whole, which typically is what non-defining relative causes do).

If we can use the explanations for these relations from competing inventories, elaboration is adding specific details (in effect, moving down the cline of delicacy) to the state of affairs in the nucleus as a whole. This is often based on set:item, class:species, or general:example relationships. The same is true for the RST-DT inventory, which has a massive, highly delicate 8-relation Elaboration class (e.g., `part-whole`, `process-step`, `object-attribute`, `set-member`). The number of EDUs is arguably the least strong signal in the category and can only be seen as an indirect marker of more elaborate discourse if taken alongside the others in the category.

`same-unit` is not a rhetorical relation per se, but a mechanical device. It is used to stitch two halves of a single EDU back together when it has been interrupted by an embedded segment (like a parenthetical or a relative clause). As such, it *could* be interpreted as a feature of more elaborate discourse because the author more frequently chooses to break the even flow of the clause, as it were, to insert something into it.

So even though the labels don’t line up one-to-one, the **higher-level direction** does.

One feature to "keep and eye on" is `attribution` / `Attribution`, which is significant in the `gumrrg` data and "almost made it" (with the p-adjusted just a sliver above 0.05, at 0.053) in the `rstreebank`-sourced data. The figures for this relation, if taken into account, would also corroborate the general trend for the essays in Group 0 vs Group 1: Attribution, although a "Subject Matter" relation, can be still arguably interpreted as a feature of a more argumentative and less linear and sequential discourse.

### What Group-1 looks like across the two models

It looks like for Group 1 (Positive) we have fewer significant FDR survivors, than for Group 0, and they also seem to be completely different. 

According to the `rstreebank` Group-1 is higher on `Sequence` and according to `gumrrg` Group-1 is higher on:

- `joint`
- `causal`
- `context`
- `nucl_NN_relprop`

This is not the same label inventory, but the overall vibe is compatible: Group-1 looks somewhat more chain-like, additive, locally linked, or paratactic, rather than elaborated via more differentiated supporting structure. One model highlights sequential organization (`Sequence`), the other highlights joint/causal/contextual linking and more NN-heavy organization — together these suggest a relatively more additive or linearly progressive discourse mode in Group-1.

One notable common thread, so to say, is that all the relations identified as statistically significantly higher belong to the "Semantic" (Stede et al., 207) or "Subject Matter" relations — relations that are mostly about describing what's out there, rather than about being argumentative or persuasive. 


## The Biggest Discrepancy

`causal` vs `Cause-effect` / `explanation`

This is one of the places where the two models seem to be telling two completely different stories:

- In `rstreebank`, `Cause-effect` is not significant
- In `gumrrg`, `causal` is significantly higher in Group-1 while `explanation` is significantly higher in Group-0

That is **not** a clean contradiction. It probably means the models are slicing a similar semantic/rhetorical zone differently.

Very roughly:

- one scheme is separating **causal progression** and **explanatory justification** more strongly
- the other is bundling or redistributing them differently

So, we probably can't say that "one model says Group-1 is more causal, the other says Group-0 is more causal, so everything collapses" or "Group-1 is more causal and Group-2 is more explanative because `gumrrg` tells us just as much". (Having said that, this latter statement/interpretation will very much align with all the previous reasoning because `causal` is in the semantic group, while `explanation` is in the pragmatic). 

The most sensible conclusion for now would probably be to say that the cause/explanation/evidence family is scheme-sensitive; the direction of contrast depends on how the model segments and labels this area and the whole issue needs looking into.

## What the Relation-Share Tables Add

The share tables help because they show the **base profile** of each model, not just the significance tests.

### `rstreebank`

Top relations for both groups are:

- `Joint`
- `Elaboration`
- `Contrast`
- `Cause-effect`    
- `Interpretation-evaluation`

But the real group splits are:

- Group-1 more `Sequence`
- Group-0 more `Condition`, `Solutionhood`, `Evidence`, `Contrast`

### `gumrrg`

Top relations for both groups are:

- `joint`
- `elaboration`
- `adversative`
- `causal` / `context` / `attribution`

But the clearer splits are:

- Group-1 more `joint`, `causal`, `context`
- Group-0 more `elaboration`, `explanation`, `contingency`, `attribution`, `same-unit`, `topic`

That makes the group contrast look stronger in gumrrg and more diffuse in rstreebank. But the overall tendency described above (and summarized below) is still there.


## The Most Defensible Synthesis

### Stable cross-model result

**Group-0 texts show greater hierarchical discourse complexity**, as reflected most consistently in higher `tree_depth`.

### Group-0, broader tendency

Across the two models, Group-0 also tends to show more:

- conditional / contingent structure
- explanatory / evidential / justificatory structure
- elaborative / expanded structure

This is clearer in gumrrg, but rstreebank points in the same direction in a weaker way.

### Group-1, broader tendency

Group-1 tends to show more:

- sequential organization in rstreebank
- joint / causal / context-linked organization and stronger NN dominance in gumrrg

So Group-1 looks relatively more:

- additive
- chain-like
- linearly progressive
- less deeply elaborated

## The Takeaway (So Far, Very Tentative)

The most robust cross-model finding is greater discourse-tree depth in Group-0. At the level of individual relation labels, the two parser variants do not yield identical results, which is expected given their different segmentation principles and relation inventories. However, the differences are not random. The gumrrg-based analysis yields a stronger contrastive profile overall, with Group-0 showing higher proportions of explanation-, contingency-, elaboration-, and attribution-like structure, alongside higher EDU counts and lower NN dominance. The rstreebank-based analysis is more conservative, but points in a compatible direction via higher Condition, Evidence, and Solutionhood in Group-0 and higher Sequence in Group-1. Taken together, the results support **a cautious** higher-level contrast between relatively more hierarchically elaborated discourse in Group-0 and relatively more additive/sequential organization in Group-1.

## References

Martin, J. R., & Rose, D. (2007). _Working with discourse: Meaning beyond the clause_. Bloomsbury.

Stede, M., Taboada, M., & Das, D. (2017). Annotation guidelines for rhetorical structure. _Manuscript. University of Potsdam and Simon Fraser University_. [https://www.sfu.ca/~mtaboada/docs/research/RST_Annotation_Guidelines.pdf](https://www.sfu.ca/~mtaboada/docs/research/RST_Annotation_Guidelines.pdf)