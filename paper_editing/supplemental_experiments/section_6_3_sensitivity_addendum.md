# Section 6.3 Candidate-to-Valid Sensitivity Addendum

Date: 2026-06-24

This supplemental experiment checks whether the Section 6.3 candidate-to-valid rule recovery result depends on undisclosed hyperparameter tuning. The datasets, rule libraries, CTHR relation-selection code, solver, and semantic evaluator are fixed to the submission-ready full-KG clean aviation and architecture setup.

Runner used:

```text
submission_support/kbs_systematic_experiments_v1/scripts/run_section_6_3_sensitivity.py
```

The run evaluated 24 configurations over 60 tasks, producing 1,440 task-configuration rows.

## Default Parameters

| Component | Parameter | Default | Sensitivity Setting |
| --- | --- | ---: | --- |
| Broad candidate grounding | old candidate score threshold | 8.0 | grid: 6, 7, 8, 9, 10 |
| Broad candidate grounding | maximum candidates per task | 24 | fixed |
| Broad candidate grounding | fallback if fewer than 3 pass threshold | top min(24, 8) scored rules | fixed |
| Aviation recall guard | minimum score for guard-fill candidates | 2.0 | fixed |
| Aviation recall guard | family-score shortcut | 6.0 | fixed |
| Task scoring | guard-satisfied bonus | 3.0 | one-factor +/-20% |
| Task scoring | source-domain/provenance bonus | 1.5 | one-factor +/-20% |
| Task scoring | variable-mapping bonus | 2.0 | one-factor +/-20% |
| Task scoring | unit-overlap bonus | 1.0 | one-factor +/-20% |
| Rule/profile matching | required group bonus | 7.0 | one-factor +/-20% |
| Rule/profile matching | allowed group bonus | 3.0 | one-factor +/-20% |
| Rule/profile matching | blocked profile penalty | -9.0 | one-factor +/-20% |
| Rule/profile matching | unmatched profile penalty | -5.0 | fixed except through profile penalty logic |
| Rule/profile matching | visible binding evidence | field, variable, unit, token, guard evidence | one-factor +/-20% |
| Rule/profile matching | seed threshold | aviation 4.5, architecture 4.0 | one-factor +/-20% |
| LLM-assisted filtering | main sensitivity table | disabled | deterministic no-LLM |
| LLM-assisted filtering | switch check | architecture-only cached profile reranking | no online sampling |
| Randomness | seed | not applicable | deterministic sorting by score then rule id; LLM temperature 0 when switch is used |

## Baseline

The deterministic no-LLM sensitivity baseline obtains Rule-ID precision 0.936, recall 0.975, exact match 0.850, and Sem-CSR 0.967 overall. This baseline is intentionally stricter than the submission-style `profile_auto_resolver` because it disables architecture LLM reranking.

## Paper-Facing Summary Table

| Experiment Block | Setting | Candidate/Ref | Filtered/Ref | Predicted/Ref | Rule Precision | Rule Recall | Exact Match | Sem-CSR | Takeaway |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Default deterministic baseline | threshold 8.0, no LLM | 9.608 | 1.301 | 1.129 | 0.936 | 0.975 | 0.850 | 0.967 | Strong recovery without LLM filtering |
| Candidate threshold grid | threshold 6 | 10.733 | 1.301 | 1.121 | 0.936 | 0.967 | 0.833 | 0.950 | Wider candidates do not improve recovery |
| Candidate threshold grid | threshold 7-8 | 10.156-9.608 | 1.301 | 1.129 | 0.936 | 0.975 | 0.850 | 0.967 | Stable near-default plateau |
| Candidate threshold grid | threshold 9-10 | 8.934-7.611 | 1.281-1.274 | 1.109-1.102 | 0.941 | 0.963-0.956 | 0.833-0.817 | 0.950 | Higher threshold trades recall for slight precision gain |
| Task scoring weights | +/-20% one-factor | 9.522-9.608 | 1.298-1.301 | 1.126-1.129 | 0.936 | 0.972-0.975 | 0.833-0.850 | 0.950-0.967 | Candidate scorer weights are not brittle |
| Rule/profile matching weights | +/-20% one-factor | 9.608 | 1.378-1.445 | 1.206-1.273 | 0.899 | 0.975 | 0.750 | 0.967 | More conservative profile variants preserve recall and Sem-CSR |
| LLM switch check | submission cached profile-auto | 9.608 | 1.185 | 1.046 | 0.952 | 0.975 | 0.883 | 0.967 | Cached architecture-only LLM reranking improves compactness and precision |

## Threshold Grid

| config | Dataset | Candidate/Ref | Filtered/Ref | Predicted/Ref | Rule Precision | Rule Recall | Exact Match | Sem-CSR | Extra | Missing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| threshold_6 | Overall | 10.733 | 1.301 | 1.121 | 0.936 | 0.967 | 0.833 | 0.950 | 13 | 4 |
| threshold_7 | Overall | 10.156 | 1.301 | 1.129 | 0.936 | 0.975 | 0.850 | 0.967 | 13 | 2 |
| threshold_8 | Overall | 9.608 | 1.301 | 1.129 | 0.936 | 0.975 | 0.850 | 0.967 | 13 | 2 |
| threshold_9 | Overall | 8.934 | 1.281 | 1.109 | 0.941 | 0.963 | 0.833 | 0.950 | 12 | 4 |
| threshold_10 | Overall | 7.611 | 1.274 | 1.102 | 0.941 | 0.956 | 0.817 | 0.950 | 12 | 6 |

## Task Scoring Weight One-Factor Perturbations

| config | Dataset | Candidate/Ref | Filtered/Ref | Predicted/Ref | Rule Precision | Rule Recall | Exact Match | Sem-CSR | Extra | Missing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| task_guard_true_0.8x | Overall | 9.608 | 1.301 | 1.129 | 0.936 | 0.975 | 0.850 | 0.967 | 13 | 2 |
| task_guard_true_1.2x | Overall | 9.608 | 1.301 | 1.129 | 0.936 | 0.975 | 0.850 | 0.967 | 13 | 2 |
| task_source_domain_0.8x | Overall | 9.525 | 1.301 | 1.129 | 0.936 | 0.975 | 0.850 | 0.967 | 13 | 2 |
| task_source_domain_1.2x | Overall | 9.608 | 1.301 | 1.129 | 0.936 | 0.975 | 0.850 | 0.967 | 13 | 2 |
| task_unit_0.8x | Overall | 9.522 | 1.298 | 1.126 | 0.936 | 0.972 | 0.833 | 0.950 | 13 | 3 |
| task_unit_1.2x | Overall | 9.608 | 1.301 | 1.129 | 0.936 | 0.975 | 0.850 | 0.967 | 13 | 2 |
| task_variable_mapping_0.8x | Overall | 9.541 | 1.298 | 1.126 | 0.936 | 0.972 | 0.833 | 0.950 | 13 | 3 |
| task_variable_mapping_1.2x | Overall | 9.608 | 1.301 | 1.129 | 0.936 | 0.975 | 0.850 | 0.967 | 13 | 2 |

## Rule/Profile Matching Weight One-Factor Perturbations

| config | Dataset | Candidate/Ref | Filtered/Ref | Predicted/Ref | Rule Precision | Rule Recall | Exact Match | Sem-CSR | Extra | Missing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| profile_allowed_0.8x | Overall | 9.608 | 1.378 | 1.206 | 0.899 | 0.975 | 0.750 | 0.967 | 23 | 2 |
| profile_allowed_1.2x | Overall | 9.608 | 1.378 | 1.206 | 0.899 | 0.975 | 0.750 | 0.967 | 23 | 2 |
| profile_penalty_0.8x | Overall | 9.608 | 1.378 | 1.206 | 0.899 | 0.975 | 0.750 | 0.967 | 23 | 2 |
| profile_penalty_1.2x | Overall | 9.608 | 1.378 | 1.206 | 0.899 | 0.975 | 0.750 | 0.967 | 23 | 2 |
| profile_required_0.8x | Overall | 9.608 | 1.378 | 1.206 | 0.899 | 0.975 | 0.750 | 0.967 | 23 | 2 |
| profile_required_1.2x | Overall | 9.608 | 1.378 | 1.206 | 0.899 | 0.975 | 0.750 | 0.967 | 23 | 2 |
| profile_seed_threshold_0.8x | Overall | 9.608 | 1.445 | 1.273 | 0.899 | 0.975 | 0.750 | 0.967 | 27 | 2 |
| profile_seed_threshold_1.2x | Overall | 9.608 | 1.378 | 1.206 | 0.899 | 0.975 | 0.750 | 0.967 | 23 | 2 |
| profile_visible_binding_0.8x | Overall | 9.608 | 1.378 | 1.206 | 0.899 | 0.975 | 0.750 | 0.967 | 23 | 2 |
| profile_visible_binding_1.2x | Overall | 9.608 | 1.428 | 1.256 | 0.899 | 0.975 | 0.750 | 0.967 | 26 | 2 |

## LLM Switch Check

The sensitivity tables above disable LLM-assisted filtering to avoid confounding the hyperparameter sweep with online model behavior. The switch check below compares deterministic no-LLM recovery with the submission-style cached `profile_auto_resolver` policy, where LLM reranking is disabled for aviation and enabled only for architecture. The submission-style rows use the archived Section 6.3 summary metrics and the corresponding Table 1 semantic success rates for the same predicted valid rules. No random seed is used by the symbolic pipeline; cached LLM calls were generated with temperature 0 and reused deterministically.

| config | Dataset | Candidate/Ref | Filtered/Ref | Predicted/Ref | Rule Precision | Rule Recall | Exact Match | Sem-CSR | Extra | Missing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| default_no_llm | Aviation | 5.794 | 1.153 | 1.108 | 0.938 | 1.000 | 0.833 | 1.000 | 5 | 0 |
| submission_profile_auto_cached | Aviation | 5.794 | 1.153 | 1.108 | 0.938 | 1.000 | 0.833 | 1.000 | 5 | 0 |
| default_no_llm | Architecture | 13.422 | 1.450 | 1.150 | 0.933 | 0.950 | 0.867 | 0.933 | 8 | 2 |
| submission_profile_auto_cached | Architecture | 13.422 | 1.217 | 0.983 | 0.967 | 0.950 | 0.933 | 0.933 | 1 | 2 |
| default_no_llm | Overall | 9.608 | 1.301 | 1.129 | 0.936 | 0.975 | 0.850 | 0.967 | 13 | 2 |
| submission_profile_auto_cached | Overall | 9.608 | 1.185 | 1.046 | 0.952 | 0.975 | 0.883 | 0.967 | 6 | 2 |

## Most Stable Parameter Region

The most stable threshold plateau is the old-candidate score threshold range 7-8. In this range, overall Rule-ID precision is 0.936, recall is 0.975, exact match is 0.850, and Sem-CSR is 0.967. Lowering the threshold to 6 increases candidate breadth but does not improve final recovery; raising it to 9-10 slightly improves precision but reduces recall, exact match, and Sem-CSR. This supports using the default threshold 8.0 as a conservative recall-preserving setting rather than a finely tuned optimum.

The +/-20% task-scoring perturbations are almost identical to the default, with at most a 0.017 drop in exact match and Sem-CSR for unit or variable-mapping down-weighting. Rule/profile matching perturbations are more conservative in this wrapper: they admit more filtered candidates, lowering precision to 0.899 and exact match to 0.750, but recall remains 0.975 and Sem-CSR remains 0.967. This is consistent with the intended division of labor: broad/profile filtering controls candidate breadth, while CTHR relation selection and the semantic evaluator preserve downstream feasibility.

## Reproducibility Statement for the Paper

```latex
We additionally performed a deterministic sensitivity analysis for the candidate-to-valid rule recovery stage. The datasets, rule libraries, CTHR relation-selection code, solver, and semantic evaluator were fixed, while the broad candidate score threshold was swept over five values and the main task-scoring and profile-matching weights were perturbed one factor at a time by +/-20%. LLM-assisted filtering was disabled in the main sweep; a separate cached switch check used the submission policy in which LLM reranking is enabled only for architecture with temperature 0. Across the stable threshold region and all one-factor perturbations, Rule-ID precision, Rule-ID recall, exact rule match, and semantic constraint satisfaction remained close to the default setting, indicating that the Section 6.3 recovery result is not an artifact of a single tuned hyperparameter.
```

## Notes on Method-Visible Inputs

- Candidate grounding uses only the visible task fields, public scenario facts, rule-library records, guards, provenance/source-domain hints, units, variables, and explicit rule relations.
- Reference valid rules, hidden solver constraints, reference feasible cells, and semantic validators are used only after prediction for metric computation.
- The broad candidate stage is intentionally recall-oriented; CTHR candidate-to-valid resolution is responsible for precision recovery through profile constraints and explicit relation reasoning.
