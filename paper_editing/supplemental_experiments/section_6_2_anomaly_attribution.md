# Section 6.2 Anomaly Attribution

Scope: this note only diagnoses task-level anomalies in Section 6.2. It does not add new experiments. The attribution below is based on the submission-ready per-task CSV files and report summaries archived on 2026-05-28.

## Evidence Files

- `submission_support/kbs_systematic_experiments_v1/results/submission_ready_20260528/table1/section_6_2_table1_architecture_old_candidate_profile_auto_resolver_all_methods_per_task.csv`
- `submission_support/kbs_systematic_experiments_v1/results/submission_ready_20260528/table1/section_6_2_table1_aviation_old_candidate_recall_guard_profile_auto_resolver_all_methods_per_task.csv`
- `submission_support/kbs_systematic_experiments_v1/results/submission_ready_20260528/table2/section_6_2_table2_cell_solver_per_task.csv`
- `submission_support/kbs_systematic_experiments_v1/results/submission_ready_20260528/section_6_3_architecture_old_candidate_recall_guard_profile_auto_resolver_candidate_to_valid_full.csv`
- `submission_support/kbs_systematic_experiments_v1/results/submission_ready_20260528/section_6_3_aviation_old_candidate_recall_guard_profile_auto_resolver_candidate_to_valid_full.csv`
- `submission_support/kbs_systematic_experiments_v1/results/submission_ready_20260528/table1/*_report.md`
- `submission_support/kbs_systematic_experiments_v1/results/submission_ready_20260528/table2/section_6_2_table2_cell_solver_report.md`

No standalone solver stdout or stderr logs are archived under `submission_ready_20260528`; the `report.md` files are therefore treated as the run-log summaries. Table 2 reports no unsupported or N/A cases.

## Summary

The main anomalies are concentrated in three tasks:

- Architecture CTHR default has two semantic-invalid tasks: `ARCH_FKG_05` and `ARCH_FKG_22`.
- `ARCH_FKG_05` is not recovered by CTHR-style CP-SAT or SCIP because the predicted valid rule is different from the reference valid rule. This is a grounding / valid-rule recovery anomaly.
- `ARCH_FKG_22` is recovered by CTHR-style CP-SAT and SCIP. The rule recovery is exact, so the anomaly is attributable to the default / ASP backend path rather than rule selection.
- Aviation default-solver objective gap is dominated by `AVI_OPT_20`. The task is rule-correct, semantically valid, and cell-valid, while ASP, CP-SAT, and SCIP find the near-best cell-valid objective. This supports a backend optimization / search-strength attribution, not a modeling-error attribution.

## Task-Level Attribution Table

| Task ID | Domain | Default semantic valid? | CTHR-style CP-SAT / SCIP semantic valid? | Rule recovery correct? | Constraint template / cell evidence | Objective gap evidence | Backend-caused? | Final attribution |
|---|---|---:|---:|---:|---|---|---:|---|
| `ARCH_FKG_05` | Architecture | No | No / No | No: precision `0.0`, recall `0.0`; predicted `ifc-2021-3204-1-high-piled-storage-commodity-class-designation`, reference `ifc-3203-9-2-highest-classification-rule` | Table 2 cells are internally cell-valid for the predicted rule, but the source-level rule is wrong; source semantics cannot be recovered by changing backend | Not an objective-gap anomaly; Table 2 default gap `0.0%` on the predicted cell | No | Candidate-to-valid recovery selected the wrong precedence rule. Because CP-SAT and SCIP consume the same wrong valid-rule set, they remain semantic-invalid. |
| `ARCH_FKG_22` | Architecture | No | Yes / Yes | Yes: precision `1.0`, recall `1.0`, exact match; predicted/reference rules are `ifc-907.5.2.2.1-paging-zones-capability` and `ifc-907.5.2.2.3-precedence-of-fire-alarm-use` | Same recovered rules; CP-SAT and SCIP are both formal-feasible and semantic-valid in Table 1. Table 2 also reports cell-valid solutions for all fixed-cell backends | Not an objective-gap anomaly; Table 2 default gap `0.0%` on fixed cells | Yes, for the Table 1 default/ASP backend path | Rule recovery and compiled-cell evidence are correct. The invalid default result is caused by backend-specific feasibility / solution-handling in the end-to-end Table 1 path. CP-SAT and SCIP recover the task. |
| `AVI_OPT_20` | Aviation | Yes | Yes / Yes | Yes: precision `1.0`, recall `1.0`, exact match; rule `sbas_fas_db_publication_requirements_5.8.7` | All fixed-cell backends are cell-valid on the same active cell `AVI_OPT_20_cthr_compiled_cell_1`; compiled cell count is `1` | Default task-level gap is `293.33%` (`2.933333321` in CSV): default objective `14.5`, best cell-valid objective `-7.500000045970003`. ASP and CP-SAT gaps are about `6.13e-7%`; SCIP gap is `0.0%` | Yes | The default solver finds a valid but poor point in the correct cell. Since the rule set, semantics, and cell validity are correct, the large gap is due to default backend optimization/search strength rather than a CTHR modeling error. |

## Architecture: Which Invalid Task Is Recovered?

CTHR default has two invalid architecture tasks:

| Task ID | Target interaction | CTHR default | CTHR-style ASP | CTHR-style CP-SAT | CTHR-style SCIP | Recovered by backend change? |
|---|---|---:|---:|---:|---:|---|
| `ARCH_FKG_05` | precedence; scenario_conditioned_applicability | invalid | invalid | invalid | invalid | No |
| `ARCH_FKG_22` | precedence; override_resolution; multi_rule_conjunction | invalid | invalid | valid | valid | Yes, recovered by CP-SAT and SCIP |

Interpretation:

- `ARCH_FKG_05` explains the irreducible 1/30 architecture invalid case that remains even after CP-SAT or SCIP. The error is upstream of the backend because the valid-rule resolver selected the wrong rule.
- `ARCH_FKG_22` explains the difference between CTHR default `93.3%` Sem-CSR and CTHR-style CP-SAT/SCIP `96.7%` Sem-CSR. This task is not a grounding failure: the predicted rules exactly match the reference rules. The recovery by CP-SAT and SCIP indicates a backend-specific issue in the default / ASP end-to-end path.

## Aviation: Why the Default Objective Gap Is Large

The average aviation objective gap for the default solver is `9.777903%`. This average is dominated by one task:

| Quantity | Value |
|---|---:|
| Largest-gap task | `AVI_OPT_20` |
| Default objective | `14.500000000000004` |
| Best cell-valid objective | `-7.500000045970003` |
| Default task-level gap | `293.33%` |
| Aviation default mean gap | `9.777903%` |
| Aviation default mean gap excluding `AVI_OPT_20` | `0.000129%` |

Backend comparison on `AVI_OPT_20`:

| Backend over CTHR cells | Cell valid? | Objective | Task-level objective gap |
|---|---:|---:|---:|
| CTHR default solver | Yes | `14.500000000000004` | `293.33%` |
| ASP/clingo over CTHR cells | Yes | `-7.499999999997431` | about `6.13e-7%` |
| CP-SAT + OR-Tools over CTHR cells | Yes | `-7.5` | about `6.13e-7%` |
| SCIP over CTHR cells | Yes | `-7.500000045970003` | `0.0%` |

Interpretation:

- `AVI_OPT_20` has exact rule recovery and is semantically valid under CTHR default and the CTHR-style backends.
- The default solver returns a cell-valid solution, so the anomaly is not a formal feasibility or compiled-cell validity failure.
- Since ASP, CP-SAT, and SCIP all find the near-best objective on the same fixed CTHR cell, the most supported attribution is default-backend optimization/search weakness on this task.
- The aviation average gap should therefore be described as a backend optimization artifact concentrated in `AVI_OPT_20`, not as evidence that CTHR compiled an incorrect feasible region.

## Recommended Wording for Paper or Rebuttal

```latex
The architecture gap between \CTHR{} default and the CP-SAT/SCIP adapters is caused by two task-level effects rather than a systematic loss of rule semantics. In `ARCH_FKG_05`, the valid-rule resolver selects a wrong precedence rule, so all backends using the same recovered rule set remain semantic-invalid. In `ARCH_FKG_22`, the recovered rules exactly match the reference rules, and CP-SAT/SCIP recover a semantic-valid decision, indicating a backend-specific feasibility/solution-handling issue in the default path.
```

```latex
The larger objective gap of the default aviation solver is dominated by a single task, `AVI_OPT_20`. The default solution is cell-valid and semantic-valid but has a much worse objective than the best cell-valid backend solution; excluding this task reduces the default aviation mean gap from 9.777903\% to 0.000129\%. Thus, the gap reflects backend search strength rather than an incorrect CTHR feasible-region compilation.
```

## Reviewer-Risk Note

The safest claim is not "all anomalies are solver bugs." The evidence supports a more precise split:

- `ARCH_FKG_05`: upstream valid-rule recovery error.
- `ARCH_FKG_22`: backend-specific feasibility / solution-handling issue, recovered by CP-SAT and SCIP.
- `AVI_OPT_20`: backend optimization/search-strength limitation, not a semantic or cell-validity failure.
