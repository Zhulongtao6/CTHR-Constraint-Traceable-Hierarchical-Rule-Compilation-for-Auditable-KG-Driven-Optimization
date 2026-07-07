# Section 6.2 Supplemental Baseline: Native ASP with Rule-Interaction Semantics

## Purpose

This supplemental baseline tests a stronger native answer-set programming baseline. It receives the same 60 tasks, broad candidate-rule sets, and visible rule-relation annotations used by the Section 6.2 CTHR experiments. The ASP program directly encodes rule applicability, dependencies, exclusions/conflicts, exception/override defeat, precedence defeat, and multi-rule conjunction.

The baseline is intentionally not allowed to use CTHR predicted valid-rule IDs, CTHR candidate-to-valid recovery, feasible-cell compilation, compiled constraint templates, or CTHR certificate-chain generation. After ASP selects rule IDs, numeric optimization uses only public scenario constraints plus native rule-library constraints for the selected rules.

## Script and Run Command

Script:

- `submission_support/kbs_systematic_experiments_v1/scripts/run_section_6_2_native_asp_semantic_baseline.py`

Run:

```powershell
python submission_support\kbs_systematic_experiments_v1\scripts\run_section_6_2_native_asp_semantic_baseline.py
```

Outputs:

- `submission_support/kbs_systematic_experiments_v1/results/section_6_2_native_asp_semantic_baseline/native_asp_semantic_baseline_overall.csv`
- `submission_support/kbs_systematic_experiments_v1/results/section_6_2_native_asp_semantic_baseline/native_asp_semantic_baseline_per_task.csv`
- `submission_support/kbs_systematic_experiments_v1/results/section_6_2_native_asp_semantic_baseline/native_asp_semantic_baseline_failure_cases.csv`
- `submission_support/kbs_systematic_experiments_v1/results/section_6_2_native_asp_semantic_baseline/native_asp_semantic_baseline_report.md`
- `submission_support/kbs_systematic_experiments_v1/results/section_6_2_native_asp_semantic_baseline/native_asp_semantic_baseline_summary.json`

## Result Table

| Dataset | Method | Rule Precision | Rule Recall | Exact Match | Formal CSR | Sem-CSR | Invalid cases | Avg runtime ms | Avg ASP selection ms | Source-preserving certificate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Aviation | Native ASP semantic baseline | 27.8% | 95.3% | 0.0% | 100.0% | 0.0% | 30/30 | 336.105 | 1.735 | No |
| Architecture | Native ASP semantic baseline | 12.4% | 80.0% | 0.0% | 100.0% | 0.0% | 30/30 | 374.952 | 3.228 | No |
| Overall | Native ASP semantic baseline | 20.1% | 87.6% | 0.0% | 100.0% | 0.0% | 60/60 | 355.529 | 2.482 | No |

## Failure Attribution

| Dataset | Failure attribution | Count |
| --- | --- | ---: |
| Aviation | rule_over_selection_without_cthr_resolution_pruning | 26 |
| Aviation | rule_selection_extra_and_missing | 4 |
| Architecture | rule_over_selection_without_cthr_resolution_pruning | 21 |
| Architecture | rule_selection_extra_and_missing | 9 |

The native ASP baseline is fast and returns formally feasible points with respect to its own selected rule-library constraints, but it over-selects from the broad candidate set. This yields high recall but very low precision, zero exact rule-set match, and zero Sem-CSR under the source-rule semantic evaluator.

## Interpretation

The result supports the paper's main distinction: native ASP can express hierarchical and exception-style symbolic rule relations, but this is not the same as CTHR's full compilation layer. Without CTHR's candidate-to-valid recovery and feasible-region compilation, ASP still lacks:

- reusable feasible cells over the continuous design variables;
- optimizer-ready compiled constraint templates;
- source-preserving certificates that remain valid across downstream backends;
- a disciplined mechanism for compressing broad candidate rules into the exact valid rule structure.

Thus, the failure is not that ASP cannot encode symbolic relations. The failure is that symbolic rule relation encoding alone does not produce the auditable continuous feasible region that CTHR contributes.

## Paper-Ready Text

```latex
We additionally compare against a native ASP baseline that receives the same tasks, broad candidate rules, and visible rule-relation annotations as \CTHR{}, including applicability, dependency, conflict/exclusion, exception override, precedence, and multi-rule conjunction. The ASP baseline directly encodes these relations as answer-set constraints, but it does not use \CTHR{}'s candidate-to-valid resolver, feasible-cell compiler, constraint-template compiler, or certificate generator.
```

```latex
This baseline is fast and achieves 100.0\% formal feasibility with respect to its own selected rule-library constraints, but it over-selects rules from the broad candidate set: overall rule precision is 20.1\%, exact rule-set match is 0.0\%, and Sem-CSR is 0.0\%. These results show that relation-aware ASP rule selection alone is insufficient for CTHR's claim. \CTHR{} contributes the additional compilation layer that recovers exact valid rule structures, converts them into reusable feasible cells over continuous design variables, and preserves source traceability across solver backends.
```

## Note for Paper-Writing Agent

Use this as a supplemental or appendix baseline, not as a replacement for the existing Table 1 rows. The safest wording is:

- ASP can encode the symbolic rule relations.
- ASP alone does not produce CTHR-style compiled feasible cells or backend-stable source certificates.
- The result strengthens the claim that CTHR is a semantic compilation framework, not merely a wrapper around a symbolic solver.
