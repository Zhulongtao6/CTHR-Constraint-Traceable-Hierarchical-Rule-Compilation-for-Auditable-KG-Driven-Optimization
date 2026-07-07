# Strong Lexical Retrieval Baseline on Architecture NCARB50 v2

This note records the strong-baseline rerun on the new 50-task architecture extension set:

`submission_support/kbs_systematic_experiments_v1/datasets/architecture_fullkg_ncarb50_v2`

The dataset manifest describes the split as an NCARB-practice-theme-derived semantic extension set, not as 50 fully independent real project cases. The retrieval method uses only method-visible inputs for rule selection: core algorithm inputs, core public scenario models, and the Qwen rule library. Evaluation references and compiled constraint templates are used only by the offline evaluator and solver adapters.

## Command

```powershell
python submission_support\kbs_systematic_experiments_v1\scripts\run_strong_lexical_retrieval_baseline.py --dataset ncarb50-v2
```

The runner was extended with `--dataset ncarb50-v2` and writes this split to a separate directory so that the original 60-task supplementary baseline is not overwritten.

## Output Files

- Per-task CSV: `submission_support/kbs_systematic_experiments_v1/results/strong_lexical_retrieval_baseline_ncarb50_v2/lexical_retrieval_per_task.csv`
- Sensitivity overall CSV: `submission_support/kbs_systematic_experiments_v1/results/strong_lexical_retrieval_baseline_ncarb50_v2/lexical_retrieval_sensitivity_overall.csv`
- Default overall CSV: `submission_support/kbs_systematic_experiments_v1/results/strong_lexical_retrieval_baseline_ncarb50_v2/lexical_retrieval_default_overall.csv`
- Default failures CSV: `submission_support/kbs_systematic_experiments_v1/results/strong_lexical_retrieval_baseline_ncarb50_v2/lexical_retrieval_default_failures.csv`
- Run summary: `submission_support/kbs_systematic_experiments_v1/results/strong_lexical_retrieval_baseline_ncarb50_v2/run_summary.json`

## Default Result

Default configuration: `top16_tau4`.

| Dataset | Method | Rule Precision | Rule Recall | Formal CSR | Sem-CSR | False accept | Invalid cases | Mean retrieved | Mean solver rules |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Architecture-NCARB50-v2 | Lexical Retrieval + CP-SAT | 21.2% | 91.0% | 70.0% | 46.0% | 24.0% | 27/50, 15 unsupported | 13.220 | 9.880 |
| Architecture-NCARB50-v2 | Lexical Retrieval + SCIP | 21.2% | 91.0% | 74.0% | 50.0% | 24.0% | 25/50, 13 unsupported | 13.220 | 9.880 |

The result is consistent with the original strong-baseline interpretation: a reasonable lexical retriever can recover many reference rules, but flat compilation remains weak. Here, rule recall is already 91.0%, yet Sem-CSR remains only 46.0% with CP-SAT and 50.0% with SCIP.

## Sensitivity

| Config | top_k | threshold | Solver | Rule Precision | Rule Recall | Formal CSR | Sem-CSR | Invalid cases |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| top8_tau4 | 8 | 4.0 | CP-SAT | 30.0% | 90.0% | 70.0% | 46.0% | 27/50 |
| top8_tau4 | 8 | 4.0 | SCIP | 30.0% | 90.0% | 74.0% | 50.0% | 25/50 |
| top12_tau4 | 12 | 4.0 | CP-SAT | 23.3% | 90.0% | 70.0% | 46.0% | 27/50 |
| top12_tau4 | 12 | 4.0 | SCIP | 23.3% | 90.0% | 74.0% | 50.0% | 25/50 |
| top16_tau4 | 16 | 4.0 | CP-SAT | 21.2% | 91.0% | 70.0% | 46.0% | 27/50 |
| top16_tau4 | 16 | 4.0 | SCIP | 21.2% | 91.0% | 74.0% | 50.0% | 25/50 |
| top24_tau4 | 24 | 4.0 | CP-SAT | 20.3% | 91.0% | 70.0% | 46.0% | 27/50 |
| top24_tau4 | 24 | 4.0 | SCIP | 20.3% | 91.0% | 74.0% | 50.0% | 25/50 |
| top16_tau2 | 16 | 2.0 | CP-SAT | 18.5% | 91.0% | 70.0% | 46.0% | 27/50 |
| top16_tau2 | 16 | 2.0 | SCIP | 18.5% | 91.0% | 74.0% | 50.0% | 25/50 |
| top16_tau6 | 16 | 6.0 | CP-SAT | 62.1% | 90.0% | 76.0% | 52.0% | 24/50 |
| top16_tau6 | 16 | 6.0 | SCIP | 62.1% | 90.0% | 80.0% | 56.0% | 22/50 |

The stricter threshold `top16_tau6` is the best lexical setting in this sweep. It raises rule precision to 62.1% and improves Sem-CSR to 52.0% for CP-SAT and 56.0% for SCIP, but it still leaves 24/50 and 22/50 invalid cases. Lower thresholds and larger `top_k` mainly add extra rules without improving semantic validity.

## Failure Pattern

Default `top16_tau4` failure summary:

| Solver | Invalid | Unsupported | False accept | Main unsupported reasons |
| --- | ---: | ---: | ---: | --- |
| CP-SAT | 27/50 | 15/50 | 12/50 | 13 infeasible, 2 unsupported nonlinear/division cases |
| SCIP | 25/50 | 13/50 | 12/50 | 13 infeasible |

Dominant invalid interaction labels are:

- `multi_rule_conjunction`: 16 CP-SAT failures and 14 SCIP failures.
- `scenario_conditioned_applicability`: 6 failures for both solvers.
- `life_safety_system_design`: 5 failures for both solvers.
- `override_resolution` and `precedence`: 3 failures each for both solvers.

Four default invalid tasks have zero rule recall under lexical retrieval: `ARCH_FKG_78`, `ARCH_FKG_79`, `ARCH_FKG_90`, and `ARCH_FKG_91`.

## Existing CTHR Context on the Same Split

The project already contains an all-methods NCARB50-v2 table:

`submission_support/kbs_systematic_experiments_v1/results/section_6_2_table1_architecture_ncarb50_v2_strict_profile_all_methods_overall.md`

Relevant rows:

| Method | Rule Precision | Rule Recall | Formal CSR | Sem-CSR | Invalid cases |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flat baseline | 77.1% | 99.0% | 90.0% | 76.0% | 12/50 |
| Native CP-SAT + OR-Tools | 82.3% | 87.0% | 90.0% | 64.0% | 18/50 |
| Native SCIP | 86.3% | 91.0% | 94.0% | 68.0% | 16/50 |
| CTHR default | 81.5% | 95.0% | 86.0% | 72.0% | 14/50 |
| CTHR-style SCIP | 81.5% | 95.0% | 88.0% | 68.0% | 16/50 |
| Lexical Retrieval + CP-SAT | 21.2% | 91.0% | 70.0% | 46.0% | 27/50 |
| Lexical Retrieval + SCIP | 21.2% | 91.0% | 74.0% | 50.0% | 25/50 |

The NCARB50-v2 comparison should be stated carefully. The new lexical retrieval baseline is clearly below the existing flat/native/CTHR rows, so it strengthens the point that retrieval plus flat solving is not enough. However, on this extension set CTHR default is not uniformly stronger than the flat baseline in Sem-CSR. This split is therefore better used as a robustness and stress-test appendix, not as a clean headline dominance table.

## Paper-Safe Statement

```latex
On the NCARB50-v2 architecture extension set, we reran the deterministic lexical-retrieval solver baseline. Under the default `top16_tau4` setting, the retriever attains 91.0\% rule recall, but the resulting flat CP-SAT/SCIP formulations reach only 46.0--50.0\% Sem-CSR and leave 25--27 of 50 cases invalid. A stricter threshold improves precision to 62.1\% and Sem-CSR to 52.0--56.0\%, but still leaves 22--24 invalid cases. Thus, even on the expanded architecture split, rule retrieval alone does not recover a valid rule structure or a reliable feasible region. We treat this result as evidence against a retrieval-only explanation rather than as a claim that this lexical baseline is a stronger optimizer.
```
