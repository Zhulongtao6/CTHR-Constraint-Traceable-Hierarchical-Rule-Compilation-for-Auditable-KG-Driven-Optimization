# Section 6.6 Addendum: Component Ablation and Relation Stress Tests

Generated: 2026-06-24

## Scope

This addendum addresses the reviewer risk that the novelty may look limited if CTHR is read as only another optimizer wrapper. The results should be framed differently: CTHR contributes an auditable rule-semantic modeling layer before optimization. The solver consumes the feasible region after CTHR has recovered scenario-valid rules, resolved rule relations, and compiled rules into executable constraints.

Two result sets must be kept separate.

| Result set | Denominator | Role |
| --- | ---: | --- |
| Main benchmark | 60 tasks: 30 aviation and 30 architecture | Primary Section 6.6 component ablation result. |
| Supplemental relation stress set | 5 extra tasks | Diagnostic stress set for relation types that are sparse in one domain. It is not included in the main 60-task denominator. |

## Main 60-Task Component Ablation

The table below summarizes the existing Section 6.6 component ablation. All numbers are from `section_6_6_cthr_component_ablation_overall.csv`.

| Removed component | Aviation effect | Architecture effect | Interpretation |
| --- | --- | --- | --- |
| Candidate-to-valid profiling | Rule-ID precision drops from 93.8% to 29.3%; Sem-CSR drops from 100.0% to 93.3%. | Rule-ID precision drops from 96.7% to 13.6%; Sem-CSR drops from 93.3% to 80.0%. | This is the largest Rule-ID precision loss. The profile resolver contracts broad lexical candidates into a valid-rule set before optimization. |
| Scenario applicability | Rule-ID precision drops to 24.9%; Formal CSR drops to 86.7%. | Rule-ID precision drops to 9.9%; Formal CSR drops to 70.0%; Sem-CSR drops to 73.3%. | Treating all retrieved rules as applicable admits many scenario-invalid rules and contaminates the feasible region. |
| Positive closure | Rule-ID precision drops to 28.1%; recall drops to 98.6%; Sem-CSR drops to 93.3%. | Rule-ID precision drops to 13.5%; recall drops to 90.0%; Sem-CSR drops to 80.0%. | Dependency closure and formula or parameter propagation decide which supporting rules must enter the valid-rule set together. |
| Negative resolution | Rule-ID precision drops to 28.7%; recall drops to 98.6%; Formal CSR and Sem-CSR are 93.3%. | Rule-ID precision drops to 14.0%; Formal CSR drops to 70.0%; Sem-CSR drops to 80.0%. | Exclusion, override, and precedence relations remove conflicting branches, defeated base rules, and lower-priority rules. |
| Relation-aware recovery | Rule-ID precision drops mildly to 91.1%; Formal CSR and Sem-CSR remain 100.0%. | Rule-ID precision drops to 85.6%; Formal CSR and Sem-CSR remain 96.7% and 93.3%. | This ablation starts from already filtered candidates, so the smaller drop should not be read as evidence that relation modeling is unnecessary. Much of the contraction has already happened upstream. |
| Compiled rule-to-constraint templates | Rule P/R stays at 93.8%/100.0%, but Formal CSR drops to 63.3% and Sem-CSR drops to 13.3%. | Rule P/R stays at 96.7%/95.0%, but Formal CSR drops to 90.0% and Sem-CSR drops to 66.7%. | This is the clearest evidence that CTHR is not just rule retrieval. Even with correct rules, removing rule-to-constraint compilation prevents the optimizer output from satisfying source-rule semantics. |

## Paper-Ready Interpretation

The ablation results are layer-specific. Removing candidate-to-valid profiling or scenario applicability mainly reduces Rule-ID precision, because broad candidates contain many rules that are lexically related but invalid for the current scenario. Removing positive closure or negative resolution stresses the symbolic rule-recovery layer: dependency and parameter propagation prevent incomplete support chains, while exclusion, override, and precedence prevent mutually inconsistent valid-rule sets. The smaller change in the direct relation-aware recovery ablation is expected because it starts from already filtered candidates. In contrast, disabling compiled rule-to-constraint templates leaves Rule-ID precision and recall almost unchanged but sharply reduces semantic constraint satisfaction. Overall, the evidence supports CTHR as an auditable rule-semantic modeling layer before optimization, not as a solver-only improvement.

## Relation Coverage Gap in the Main Benchmark

The main 60-task benchmark covers all six rule-relation categories after combining both domains, but the coverage is uneven by domain.

| Domain | Scenario applicability | Branch/exclusion | Dependency/formula | Multi-rule conjunction | Exception/override | Precedence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Aviation, 30 tasks | 8 | 11 | 15 | 2 | 2 | 0 |
| Architecture, 30 tasks | 5 | 0 | 1 | 24 | 2 | 4 |
| Combined, 60 tasks | 13 | 11 | 16 | 26 | 4 | 4 |

The main coverage is therefore sufficient for a combined benchmark claim, but not for claiming that every relation type is equally represented in each domain. The supplemental stress set is designed for this gap: aviation precedence, architecture branch/exclusion, and architecture dependency/formula propagation.

## Supplemental Relation Stress Set

These five tasks are extra diagnostics. They use existing full-KG rules and should not be merged into the main benchmark denominator unless all main tables are rerun with a revised task set.

| Stress task | Domain | Relation type | Source rules | Expected valid rules | CTHR recovery | Flat or relation-removed behavior |
| --- | --- | --- | --- | --- | --- | --- |
| `SUP_AVI_PREC_01`: TAA buffer obstacle controls minimum altitude | Aviation | Precedence / calculation-basis priority | `ttaa_buffer_obstacle_height_adjustment`; `ttaa_buffer_zone_radius`; `ttaa_protected_area_boundary_radius`; `ttaa_obstacle_clearance_mountain_increase` | The buffer obstacle adjustment rule must be included when the buffer obstacle controls the altitude basis, together with the TAA buffer/radius rules. | Not fully correct. Default CTHR misses `ttaa_buffer_obstacle_height_adjustment`; recall is 0.75 and Sem-CSR is false. | Flat candidates and the scenario-applicability ablation keep the full reference set. This task exposes a current recovery/applicability boundary rather than a positive CTHR win. |
| `SUP_AVI_PROP_02`: turning missed-approach MOC formula propagation | Aviation | Dependency/formula propagation with a branch distractor | `moc_turn_area_formula`; `moc_turn_initiation_area_calculation`; `tnh_calculation_formula`; `tnh_obstacle_clearance_requirement`; `turn_init_area_der_extent`; distractor `turn_init_area_fato_extent` | The turn-area MOC and TNH support rules should survive; the FATO distractor should be removed for the fixed-wing DER scenario. | Partially correct. Default CTHR is semantically valid but retains the FATO distractor; precision is 0.833 and recall is 1.0. | Removing compiled templates makes the returned point fail Formal CSR and Sem-CSR. This supports the rule-to-constraint compilation claim more than the relation-recovery claim. |
| `SUP_ARCH_EXCL_01`: sprinkler exception to assisted-rescue stair width | Architecture | Branch/exclusion / exception override | `ifc-2021-1009.8-stair-clear-width-min`; `ifc-2021-1009.8-stair-clear-width-exception` | In a fully sprinklered assisted-rescue stair scenario, the exception should defeat the base 48-inch stair clear-width rule. | Not correct. Default CTHR selects the base rule rather than the exception; Rule P/R is 0/0. | The scenario-applicability ablation happens to match the exception. Because this pair has weak executable constraint support, the task should be reported as an audit case, not as a positive result. |
| `SUP_ARCH_EXCL_02`: I-2 existing-door exception | Architecture | Branch/exclusion / exception override | `ifc2021-moe-door-patient-bed-exception-i2-condition1`; `ifc2021-moe-door-patient-bed-min-width` | The existing I-2 door exception should survive and the defeated patient-bed base width rule should be removed. | Correct. Default CTHR exactly matches the expected surviving rule with Rule P/R 1.0/1.0. | Flat candidates and the no-negative-resolution variant keep both base and exception rules; precision drops to 0.5 and exact match becomes false. |
| `SUP_ARCH_PROP_01`: audible alarm SPL formula propagation | Architecture | Dependency/formula propagation | `ifc-2021-907-5-2-1-2-audible-alarm-spl-min`; `ifc-2021-907-5-2-1-2-audible-alarm-spl-max` | The minimum audible level formula and the 110 dBA upper bound must both survive. | Correct. Default CTHR exactly matches the two expected rules with Rule P/R 1.0/1.0. | This task verifies recovery of the formula pair. It does not isolate compiled-template benefit because the reused task already contains enough public/reference formula structure for the no-template variant to remain valid. |

## Supplemental Stress Results

| Variant | Component removed | Tasks | Solved | Formal CSR | Sem-CSR | Rule-ID Precision | Rule-ID Recall | Exact Rule Match |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| CTHR default | None | 5 | 5/5 (100.0%) | 100.0% | 80.0% | 76.7% | 75.0% | 2/5 (40.0%) |
| Flat candidates | Relation-aware recovery | 5 | 5/5 (100.0%) | 100.0% | 100.0% | 76.7% | 100.0% | 2/5 (40.0%) |
| w/o Scenario Applicability | Scenario applicability filtering | 5 | 5/5 (100.0%) | 100.0% | 100.0% | 96.7% | 100.0% | 4/5 (80.0%) |
| w/o Positive Closure | Dependency closure and parameter/formula propagation | 5 | 5/5 (100.0%) | 100.0% | 80.0% | 76.7% | 75.0% | 2/5 (40.0%) |
| w/o Negative Resolution | Exclusion, override, and precedence resolution | 5 | 5/5 (100.0%) | 100.0% | 80.0% | 66.7% | 75.0% | 1/5 (20.0%) |
| w/o Compiled Templates | Compiled rule-to-constraint templates | 5 | 5/5 (100.0%) | 80.0% | 60.0% | 76.7% | 75.0% | 2/5 (40.0%) |

## Recommended Claim Boundary

The supplemental stress set should be described as a diagnostic appendix, not as an additional main benchmark result. It shows that negative resolution and compiled rule-to-constraint templates affect relation-sensitive tasks, while also exposing two recovery boundary cases. The safest claim is: the main 60-task ablation demonstrates that CTHR's pre-solver rule-semantic layers matter for valid-rule precision and semantic constraint satisfaction; the supplemental stress set further identifies sparse relation categories and provides targeted regression tests for future improvements.

