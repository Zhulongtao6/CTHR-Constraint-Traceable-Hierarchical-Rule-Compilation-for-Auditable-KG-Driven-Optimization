# Strong Baseline Addendum: Lexical Retrieval + Constraint Solving

This addendum records a stronger non-CTHR baseline for Section 6.2. The purpose is to address the reviewer risk that a purely flat rule baseline is too weak and therefore exaggerates the advantage of CTHR. The added baseline uses a deterministic lexical retriever to select a task-specific rule subset, then compiles the selected rules directly into the existing CP-SAT and SCIP solver backends.

## Experiment Setting

The experiment reuses the existing Section 6.2 datasets, rule inventory, constraint templates, solver adapters, and evaluation metrics. It does not call a new LLM, does not download a new embedding model, and does not use deep semantic embeddings. The retrieval model is therefore named **Lexical Retrieval + CP-SAT/SCIP**, not semantic embedding retrieval.

For each task, the query text is built from method-visible fields:

- task identifier, task type, title-like description, design intent, engineering task text;
- scenario facts and guard-relevant fields;
- decision variables, objectives, and preference text where available.

For each candidate rule, the rule text is built from method-visible rule fields:

- rule id, rule name, rule type, domain;
- applicability guard fields;
- constraint templates and parameter descriptions;
- provenance/source fields.

The retriever ranks all candidate rules by a deterministic score:

```text
score(q, r) =
    IDFWeightedLexicalCoverage(q, r)
  + 2.0 * mapped_variable_count(q, r), capped at 4.0
  + 1.0 * unit_match(q, r)
  + 1.5 * matched_guard_field_count(q, r), capped at 3.0
  + guard_bonus(q, r)
  + 0.5 * provenance_domain_match(q, r)
```

where `guard_bonus` is `+3.0` for public scenario guards that evaluate true, `+0.5` for unknown guards, and `-4.0` for false guards. Rules with deterministically false public guards are pruned before top-k selection. The default configuration is `top_k = 16`, `threshold = 4.0`, and `min_k = 3`.

After retrieval, the selected rules are compiled into a single flat constraint set:

```text
F_lex(q) = X_q intersect intersection_{r in R_lex(q)} Gamma(r, q)
```

where `X_q` is the task variable space and `Gamma(r, q)` is the existing rule-to-constraint compilation used by the Section 6.2 solver backends. This baseline deliberately does not use CTHR's defeasible-rule recovery, dependency closure, override reasoning, precedence handling, mutual-exclusion decomposition, parameter propagation, or feasible-unit construction. In other words, it tests whether a reasonable rule selector plus a strong solver is already enough.

## Relation to CTHR Rules

The "rules" in this baseline are the same curated rule records and constraint templates available to Section 6.2. They are not newly generated rules. They are also not CTHR's final valid-rule structures. CTHR uses defeasible reasoning to recover which rules are active, overridden, dependent, or mutually exclusive, and then organizes them into feasible units. The retrieval baseline only selects records from the same candidate rule inventory using surface evidence, then treats the selected set as a flat conjunction.

This distinction is important for interpretation. High rule recall in the retrieval baseline means that the relevant records can often be found from task text and public scenario fields. It does not mean that the selected records are organized into a valid rule structure. The remaining gap to CTHR therefore measures the value of defeasible applicability recovery and feasible-unit modeling, not merely the value of access to more rules.

## Data Sources

The run uses the existing Section 6.2 assets:

- paper source: `neurosymbolic-research/cthr/paper_editing/cthr_paper.tex`;
- existing Section 6.2 evaluation pipeline: `neurosymbolic-research/cthr/submission_support/kbs_systematic_experiments_v1/scripts/run_section_6_2_table1_fullkg_pipeline.py`;
- solver adapters reused from: `neurosymbolic-research/cthr/submission_support/kbs_systematic_experiments_v1/scripts/run_section_6_2_table1_aviation_old_candidate_profile_all_methods.py`;
- existing submission-ready comparison tables: `neurosymbolic-research/cthr/submission_support/kbs_systematic_experiments_v1/results/submission_ready_20260528/table1`;
- new runner: `neurosymbolic-research/cthr/submission_support/kbs_systematic_experiments_v1/scripts/run_strong_lexical_retrieval_baseline.py`;
- new results: `neurosymbolic-research/cthr/submission_support/kbs_systematic_experiments_v1/results/strong_lexical_retrieval_baseline`.

## Main Result

The default lexical retrieval configuration is `top16_tau4`. CP-SAT and SCIP have identical aggregate values for this baseline.

| Dataset | Method | Rule Precision | Rule Recall | Formal CSR | Sem-CSR | Invalid cases |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Aviation | Flat baseline | 23.6% | 100.0% | 76.7% | 86.7% | 4/30 |
| Aviation | Native ASP + clingo | 52.2% | 84.7% | 96.7% | 80.0% | 6/30 |
| Aviation | Native CP-SAT + OR-Tools | 54.9% | 81.9% | 96.7% | 76.7% | 7/30 |
| Aviation | Native SCIP | 54.9% | 81.9% | 96.7% | 76.7% | 7/30 |
| Aviation | Lexical Retrieval + CP-SAT | 19.7% | 71.6% | 100.0% | 56.7% | 13/30 |
| Aviation | Lexical Retrieval + SCIP | 19.7% | 71.6% | 100.0% | 56.7% | 13/30 |
| Aviation | CTHR default | 93.8% | 100.0% | 100.0% | 100.0% | 0/30 |
| Aviation | CTHR-style CP-SAT + OR-Tools | 93.8% | 100.0% | 100.0% | 100.0% | 0/30 |
| Architecture | Flat baseline | 10.4% | 95.0% | 66.7% | 76.7% | 7/30 |
| Architecture | Native ASP + clingo | 80.9% | 95.0% | 96.7% | 93.3% | 2/30 |
| Architecture | Native CP-SAT + OR-Tools | 80.9% | 95.0% | 100.0% | 96.7% | 1/30 |
| Architecture | Native SCIP | 80.9% | 95.0% | 100.0% | 96.7% | 1/30 |
| Architecture | Lexical Retrieval + CP-SAT | 19.9% | 100.0% | 70.0% | 70.0% | 9/30 |
| Architecture | Lexical Retrieval + SCIP | 19.9% | 100.0% | 70.0% | 70.0% | 9/30 |
| Architecture | CTHR default | 96.7% | 95.0% | 96.7% | 93.3% | 2/30 |
| Architecture | CTHR-style CP-SAT + OR-Tools | 96.7% | 95.0% | 100.0% | 96.7% | 1/30 |

The lexical retriever is not a stronger replacement for CTHR. In Architecture, it reaches 100.0% rule recall, but its flat conjunction makes 9/30 cases unsupported or infeasible. This is the cleanest evidence that finding the relevant rules is insufficient: the method must also preserve rule structure, exceptions, and feasible branches. In Aviation, the lexical baseline has lower recall and much lower semantic validity, indicating that surface matching is also brittle when dependency and scenario-conditioned applicability are central.

## Hyperparameter Sensitivity

The sensitivity sweep changes `top_k` and the score threshold. The table reports CP-SAT because SCIP is identical at this aggregate level.

| Dataset | Config | top_k | threshold | Rule Precision | Rule Recall | Formal CSR | Sem-CSR | Invalid cases | Mean retrieved | Mean solver rules |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Aviation | top8_tau4 | 8 | 4.0 | 28.0% | 69.9% | 100.0% | 56.7% | 13/30 | 7.667 | 5.867 |
| Aviation | top12_tau4 | 12 | 4.0 | 21.0% | 71.6% | 100.0% | 56.7% | 13/30 | 10.333 | 8.333 |
| Aviation | top16_tau4 | 16 | 4.0 | 19.7% | 71.6% | 100.0% | 56.7% | 13/30 | 11.533 | 9.233 |
| Aviation | top24_tau4 | 24 | 4.0 | 19.2% | 71.6% | 100.0% | 56.7% | 13/30 | 12.033 | 9.667 |
| Aviation | top16_tau2 | 16 | 2.0 | 12.4% | 72.2% | 93.3% | 60.0% | 12/30 | 16.000 | 13.333 |
| Aviation | top16_tau6 | 16 | 6.0 | 40.9% | 64.2% | 100.0% | 53.3% | 14/30 | 4.033 | 3.100 |
| Architecture | top8_tau4 | 8 | 4.0 | 29.5% | 98.3% | 70.0% | 70.0% | 9/30 | 7.967 | 6.333 |
| Architecture | top12_tau4 | 12 | 4.0 | 23.2% | 100.0% | 70.0% | 70.0% | 9/30 | 11.367 | 8.867 |
| Architecture | top16_tau4 | 16 | 4.0 | 19.9% | 100.0% | 70.0% | 70.0% | 9/30 | 14.033 | 10.800 |
| Architecture | top24_tau4 | 24 | 4.0 | 17.8% | 100.0% | 70.0% | 70.0% | 9/30 | 17.967 | 13.300 |
| Architecture | top16_tau2 | 16 | 2.0 | 17.7% | 100.0% | 70.0% | 70.0% | 9/30 | 16.000 | 12.633 |
| Architecture | top16_tau6 | 16 | 6.0 | 60.3% | 98.3% | 76.7% | 76.7% | 7/30 | 3.700 | 3.333 |

The sweep shows the expected precision-recall tradeoff, but no setting closes the gap to CTHR. A stricter threshold improves Architecture precision and invalid cases, but still leaves 7/30 invalid cases. A looser threshold increases recall only marginally and often lowers precision or introduces unsupported conjunctions.

## Failure Cases

Default lexical retrieval failures concentrate in the same interaction types that motivate CTHR:

- Aviation has 13 invalid cases per solver: `AVI_OPT_01`, `AVI_OPT_02`, `AVI_OPT_08`, `AVI_OPT_09`, `AVI_OPT_10`, `AVI_OPT_16`, `AVI_OPT_17`, `AVI_OPT_18`, `AVI_OPT_22`, `AVI_OPT_24`, `AVI_OPT_27`, `AVI_OPT_28`, and `AVI_OPT_29`.
- Aviation failures are dominated by dependency or formula propagation, branch/exclusion handling, exception/override handling, scenario-conditioned applicability, and provenance-sensitive multi-rule conjunctions.
- Architecture has 9 invalid cases per solver: `ARCH_FKG_05`, `ARCH_FKG_07`, `ARCH_FKG_08`, `ARCH_FKG_14`, `ARCH_FKG_16`, `ARCH_FKG_21`, `ARCH_FKG_26`, `ARCH_FKG_28`, and `ARCH_FKG_29`.
- Architecture failures are mostly solver infeasibility from over-constraining the problem after flat conjunction. Several failures involve precedence, exception/override behavior, and multi-rule conjunction.

These failures indicate that a reasonable retriever can identify many relevant rule records, but cannot decide how incompatible branches, overrides, dependencies, and conditional subcases should be assembled.

## Paper-Ready Interpretation

The added baseline addresses a stronger alternative explanation than the original flat baseline: perhaps a simple retriever could pick a cleaner subset of rules and a mature solver could finish the task. The results do not support that explanation. In Aviation, lexical retrieval remains substantially below CTHR in both rule recovery and semantic constraint satisfaction. In Architecture, the retriever often recovers the reference rules, but flat compilation makes the solver infeasible on many tasks. Therefore, the empirical advantage of CTHR is not just caused by comparing against an overly broad flat rule set. The advantage comes from recovering valid defeasible rule structure and preserving feasible units before solver execution.

Recommended text for the paper:

```latex
To rule out the possibility that CTHR only benefits from comparing against an overly broad flat rule set, we add a deterministic lexical-retrieval baseline. The baseline constructs task and rule texts from the same public fields used in Section 6.2, scores candidate rules by IDF-weighted lexical overlap, variable overlap, unit compatibility, public guard matching, and provenance-domain matching, and then compiles the selected rules directly into CP-SAT or SCIP. This baseline does not use CTHR's defeasible recovery, override handling, dependency closure, or feasible-unit construction. On Architecture it achieves 100.0\% rule recall under the default setting, but still produces 9/30 invalid cases because the retrieved rules are compiled as one flat conjunction. On Aviation it reaches only 71.6\% rule recall and 56.7\% semantic CSR. These results show that task-specific rule retrieval alone is insufficient; CTHR's contribution is the recovery and organization of valid rule structure before optimization.
```

## LaTeX Table Draft

```latex
\begin{table*}[t]
\centering
\caption{Strong retrieval baseline against flat, native symbolic, and CTHR semantic modeling. Lexical retrieval uses IDF-weighted token overlap, variable/unit matching, public guard scoring, and provenance-domain matching, then compiles selected rules into a single flat constraint set.}
\label{tab:strong_retrieval_baseline}
\footnotesize
\begin{tabular*}{\textwidth}{@{}llrrrrr@{}}
\toprule
Dataset & Method & Rule P & Rule R & Formal CSR & Sem-CSR & Invalid \\
\midrule
Aviation & Flat baseline & 23.6 & 100.0 & 76.7 & 86.7 & 4/30 \\
Aviation & Native CP-SAT + OR-Tools & 54.9 & 81.9 & 96.7 & 76.7 & 7/30 \\
Aviation & Lexical retrieval + CP-SAT & 19.7 & 71.6 & 100.0 & 56.7 & 13/30 \\
Aviation & CTHR default & 93.8 & 100.0 & 100.0 & 100.0 & 0/30 \\
Aviation & CTHR-style CP-SAT + OR-Tools & 93.8 & 100.0 & 100.0 & 100.0 & 0/30 \\
\addlinespace
Architecture & Flat baseline & 10.4 & 95.0 & 66.7 & 76.7 & 7/30 \\
Architecture & Native CP-SAT + OR-Tools & 80.9 & 95.0 & 100.0 & 96.7 & 1/30 \\
Architecture & Lexical retrieval + CP-SAT & 19.9 & 100.0 & 70.0 & 70.0 & 9/30 \\
Architecture & CTHR default & 96.7 & 95.0 & 96.7 & 93.3 & 2/30 \\
Architecture & CTHR-style CP-SAT + OR-Tools & 96.7 & 95.0 & 100.0 & 96.7 & 1/30 \\
\bottomrule
\end{tabular*}
\end{table*}
```
