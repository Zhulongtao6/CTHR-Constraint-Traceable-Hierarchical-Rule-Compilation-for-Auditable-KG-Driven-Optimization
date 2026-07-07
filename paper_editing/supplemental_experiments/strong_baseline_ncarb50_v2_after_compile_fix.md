# Strong Lexical Retrieval Baseline on NCARB50 v2 After Compile Fix

This note records the rerun after fixing the post-retrieval rule-to-feasible-region adapter for the new 50-task architecture set.

## Fix Scope

The retrieval stage is unchanged. The fix only changes how retrieved rules are converted into executable constraints:

- Uses the latest compiled-template path from `run_section_6_2_table1_fullkg_pipeline.py`, including exact context matching followed by relaxed semantic context matching.
- Ignores over-specific NCARB scenario fields during relaxed template matching, such as `ncarb_scenario_theme`, `scenario_variant`, `facility_type`, `story_condition`, `locking_mechanism`, `review_context`, and `voice_alarm_context`.
- Disables generic numeric extraction from arbitrary rule text. For rules without audited templates, fallback compilation now accepts only explicit numeric values stored directly in structured rule constraints.
- Leaves rule retrieval unchanged and does not use reference valid rules as method input.

Output directory:

`submission_support/kbs_systematic_experiments_v1/results/strong_lexical_retrieval_baseline_ncarb50_v2_after_compile_fix`

Run timestamp: `2026-06-27 14:26:56`.

## Table-1-Style Main Metrics

Default configuration: `top16_tau4`.

| Method | Rule Precision | Rule Recall | Sem-CSR | Invalid cases |
| --- | ---: | ---: | ---: | ---: |
| Lexical Retrieval + CP-SAT, before fix | 21.2% | 91.0% | 46.0% | 27/50 |
| Lexical Retrieval + CP-SAT, after fix | 21.2% | 91.0% | 70.0% | 15/50 |
| Lexical Retrieval + SCIP, before fix | 21.2% | 91.0% | 50.0% | 25/50 |
| Lexical Retrieval + SCIP, after fix | 21.2% | 91.0% | 74.0% | 13/50 |

The fix does not change rule selection, so Rule Precision and Rule Recall remain the same. The improvement is entirely from the post-retrieval compilation/solver adapter.

## Sensitivity After Fix

| Config | Solver | Rule Precision | Rule Recall | Sem-CSR | Invalid cases |
| --- | --- | ---: | ---: | ---: | ---: |
| top8_tau4 | CP-SAT | 30.0% | 90.0% | 70.0% | 15/50 |
| top8_tau4 | SCIP | 30.0% | 90.0% | 74.0% | 13/50 |
| top12_tau4 | CP-SAT | 23.3% | 90.0% | 70.0% | 15/50 |
| top12_tau4 | SCIP | 23.3% | 90.0% | 74.0% | 13/50 |
| top16_tau4 | CP-SAT | 21.2% | 91.0% | 70.0% | 15/50 |
| top16_tau4 | SCIP | 21.2% | 91.0% | 74.0% | 13/50 |
| top24_tau4 | CP-SAT | 20.3% | 91.0% | 70.0% | 15/50 |
| top24_tau4 | SCIP | 20.3% | 91.0% | 74.0% | 13/50 |
| top16_tau2 | CP-SAT | 18.5% | 91.0% | 70.0% | 15/50 |
| top16_tau2 | SCIP | 18.5% | 91.0% | 74.0% | 13/50 |
| top16_tau6 | CP-SAT | 62.1% | 90.0% | 74.0% | 13/50 |
| top16_tau6 | SCIP | 62.1% | 90.0% | 78.0% | 11/50 |

The stricter `top16_tau6` setting is now the best retrieval configuration in this sweep: it trades a small recall decrease for much higher precision and reaches 74.0% Sem-CSR with CP-SAT and 78.0% with SCIP.

## Diagnostics

Formal CSR is kept as a diagnostic metric rather than a Table-1-style headline column:

| Config | Method | Formal CSR before fix | Formal CSR after fix |
| --- | --- | ---: | ---: |
| top16_tau4 | Lexical Retrieval + CP-SAT | 70.0% | 78.0% |
| top16_tau4 | Lexical Retrieval + SCIP | 74.0% | 82.0% |
| top16_tau6 | Lexical Retrieval + CP-SAT | 76.0% | 82.0% |
| top16_tau6 | Lexical Retrieval + SCIP | 80.0% | 86.0% |

Invalid-case details:

| Config | Method | Before fix invalid cases | After fix invalid cases |
| --- | --- | ---: | ---: |
| top16_tau4 | Lexical Retrieval + CP-SAT | 27/50, 15 unsupported | 15/50, 11 unsupported |
| top16_tau4 | Lexical Retrieval + SCIP | 25/50, 13 unsupported | 13/50, 9 unsupported |
| top16_tau6 | Lexical Retrieval + CP-SAT | 24/50, 12 unsupported | 13/50, 9 unsupported |
| top16_tau6 | Lexical Retrieval + SCIP | 22/50, 10 unsupported | 11/50, 7 unsupported |

Remaining default `top16_tau4` failures:

- CP-SAT: 15 invalid cases, including 11 unsupported cases. The unsupported cases are 9 infeasible flat conjunctions and 2 nonlinear/division cases.
- SCIP: 13 invalid cases, including 9 infeasible flat conjunctions.
- Four invalid tasks have zero rule recall under lexical retrieval: `ARCH_FKG_78`, `ARCH_FKG_79`, `ARCH_FKG_90`, and `ARCH_FKG_91`.
- The dominant remaining interaction types are `multi_rule_conjunction`, `scenario_conditioned_applicability`, `life_safety_system_design`, `multi_constraint_single_rule`, and `exception_or_override`.

## Interpretation

The before/after change confirms that a substantial part of the original weak NCARB50-v2 result came from stale post-retrieval compilation, especially unsafe fallback constraints and overly strict template context matching. After the fix, Sem-CSR improves by 22--24 percentage points across the default CP-SAT/SCIP rows.

The remaining gap is no longer mainly a template-context bug. It is mostly caused by:

- low rule precision under the default retriever, especially extra rules that over-constrain a flat feasible region;
- four zero-recall scenario-conditioned tasks;
- solver/backend limitations for nonlinear or division-style constraints in CP-SAT.

This supports the intended conclusion: retrieval alone can recover many relevant rules, but without CTHR-style valid-rule structure and feasible-unit organization, flat compilation remains fragile.
