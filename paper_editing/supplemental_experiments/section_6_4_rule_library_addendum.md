# Section 6.4 Addendum: Rule-Library Generator and Cross-Model Stability

Generated: 2026-06-24

## Scope

This addendum clarifies the multi-model rule-library experiment for Section 6.4. The experiment is diagnostic. It should not be presented as evidence that end-to-end rule extraction is fully solved. It tests whether CTHR remains usable when the fixed benchmark semantics are evaluated against different rule-library namespaces through strong semantic overlays.

The fixed architecture benchmark is not duplicated. The core tasks, scenario parameters, objectives, public scenario models, feasible regions, provenance, and source semantic references remain unchanged. Per-model overlays only project canonical source-rule references into each model's rule-ID namespace.

## Architecture Benchmark and Strict-Common Subset

The complete architecture benchmark contains 30 tasks. The strict-common subset contains 25 tasks. A task enters the strict-common subset only when every canonical expected surviving rule has exact or strong alignment in Qwen, DeepSeek, and Xiaomi MIMO overlays.

The strict-common subset is therefore an alignment-coverage subset, not a new benchmark and not a filtered performance claim. A strict-common task can still fail downstream if a model's candidate generator returns no candidate or returns the wrong candidate. This distinction is important for explaining why the 25-task subset still has candidate-zero and unsupported cases in the downstream table.

Strict-common task list:

`ARCH_FKG_01`, `ARCH_FKG_02`, `ARCH_FKG_03`, `ARCH_FKG_04`, `ARCH_FKG_06`, `ARCH_FKG_07`, `ARCH_FKG_08`, `ARCH_FKG_09`, `ARCH_FKG_10`, `ARCH_FKG_11`, `ARCH_FKG_12`, `ARCH_FKG_13`, `ARCH_FKG_15`, `ARCH_FKG_16`, `ARCH_FKG_17`, `ARCH_FKG_18`, `ARCH_FKG_19`, `ARCH_FKG_20`, `ARCH_FKG_22`, `ARCH_FKG_24`, `ARCH_FKG_26`, `ARCH_FKG_27`, `ARCH_FKG_28`, `ARCH_FKG_29`, `ARCH_FKG_30`.

## Excluded Architecture Tasks

The five excluded tasks are excluded only from the strict-common subset. They remain part of the full 30-task benchmark. The exclusion reason is unresolved or weak-only semantic alignment in at least one non-Qwen rule library.

| Task | Title | Canonical expected surviving rules | DeepSeek unresolved rules | Xiaomi MIMO unresolved rules | Exclusion reason |
| --- | --- | --- | --- | --- | --- |
| `ARCH_FKG_05` | High-piled commodity classification precedence | `ifc-3203-9-2-highest-classification-rule` | `ifc-3203-9-2-highest-classification-rule` | `ifc-3203-9-2-highest-classification-rule` | The highest-classification precedence rule has only weak candidates in both non-Qwen libraries. |
| `ARCH_FKG_14` | Rack storage flue-space and height envelope | `ifc2021-flue-space-transverse-min-3in`; `ifc2021-storage-height-ceiling-sprinkler-max-20ft` | `ifc2021-flue-space-transverse-min-3in` | `ifc2021-storage-height-ceiling-sprinkler-max-20ft` | Each non-Qwen library lacks a strong match for one of the two required storage rules. |
| `ARCH_FKG_21` | Existing ambulatory care most-restrictive upgrade | `ifc_k101_1_scope`; `ifc_k102_1_separation` | `ifc_k101_1_scope`; `ifc_k102_1_separation` | none | DeepSeek does not strongly align the two existing-building scope and separation rules. |
| `ARCH_FKG_23` | Construction-site firefighting access | `ifc-3311.1-emergency-contact-posting-alternative`; `ifc-3311.1-vehicle-access-distance` | none | `ifc-3311.1-vehicle-access-distance` | Xiaomi MIMO does not strongly align the vehicle-access distance rule. |
| `ARCH_FKG_25` | Roll-in shower control and distribution layout | `ada-2010-608-5-2-control-height`; `ada_shower_distribution_gender_separated_facilities` | none | `ada_shower_distribution_gender_separated_facilities` | Xiaomi MIMO does not strongly align the gender-separated roll-in shower distribution rule. |

## Multi-Model Rule-Library Quality

The following table combines rule-library quality diagnostics with downstream canonical-projected results on the full 30-task architecture benchmark.

| Model | Rules | Source validity | Constraint grounding | Relation grounding | Strong canonical coverage | Weak candidates filtered | Downstream Sem-CSR | Candidate zero | Unsupported tasks | Invalid cases |
| --- | ---: | --- | --- | --- | --- | ---: | --- | ---: | ---: | --- |
| Qwen | 1,585 | 100.0% | 100.0% | 100.0% | 51/51 | 0 | 93.3% | 0 | 0 | 2/30 (6.7%) |
| DeepSeek | 2,419 | 100.0% | 99.5% | 99.6% | 47/51 | 408 | 66.7% | 3 | 4 | 10/30 (33.3%), including 4 unsupported |
| Xiaomi MIMO | 2,157 | 99.9% | 99.6% | 99.8% | 47/51 | 408 | 40.0% | 7 | 10 | 18/30 (60.0%), including 10 unsupported |

## Section 6.4 Result Tables

Raw-ID results are namespace diagnostics only. DeepSeek and Xiaomi MIMO are expected to score poorly under raw Qwen IDs because their rule IDs live in different namespaces.

| Model | Raw-ID Rule Precision | Raw-ID Rule Recall | Raw-ID Sem-CSR | Unsupported tasks | Invalid cases |
| --- | --- | --- | --- | ---: | --- |
| Qwen | 91.9% | 96.7% | 93.3% | 0 | 2/30 (6.7%) |
| DeepSeek | 0.0% | 0.0% | 43.3% | 4 | 17/30 (56.7%), including 4 unsupported |
| Xiaomi MIMO | 2.1% | 5.0% | 30.0% | 10 | 21/30 (70.0%), including 10 unsupported |

The canonical-projected table is the main Section 6.4 result. Predicted model rule IDs are first projected back to canonical source-rule IDs through strong alignment, then compared against the fixed source-rule oracle.

| Model | Canonical Rule Precision | Canonical Rule Recall | Formal CSR | Sem-CSR | Candidate zero | Unsupported tasks | Invalid cases |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
| Qwen | 96.7% | 96.7% | 96.7% | 93.3% | 0 | 0 | 2/30 (6.7%) |
| DeepSeek | 76.7% | 66.7% | 83.3% | 66.7% | 3 | 4 | 10/30 (33.3%), including 4 unsupported |
| Xiaomi MIMO | 46.7% | 36.7% | 66.7% | 40.0% | 7 | 10 | 18/30 (60.0%), including 10 unsupported |

On the 25-task strict-common subset, all reference rules have strong alignment across Qwen, DeepSeek, and Xiaomi MIMO. This removes unresolved reference-rule coverage as a confound, but it does not remove downstream candidate-generation failures.

| Model | Strict-common Canonical Rule Precision | Strict-common Canonical Rule Recall | Formal CSR | Sem-CSR | Candidate zero | Unsupported tasks | Invalid cases |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
| Qwen | 100.0% | 100.0% | 96.0% | 96.0% | 0 | 0 | 1/25 (4.0%) |
| DeepSeek | 84.0% | 74.0% | 80.0% | 72.0% | 3 | 4 | 7/25 (28.0%), including 4 unsupported |
| Xiaomi MIMO | 44.0% | 36.0% | 60.0% | 40.0% | 7 | 10 | 15/25 (60.0%), including 10 unsupported |

## Paper-Ready Explanation

Section 6.4 is a diagnostic cross-model rule-library substitution experiment. The benchmark semantics are fixed by the Qwen-canonical source-rule oracle and are not regenerated by each model. For each non-Qwen rule library, we build an evaluation overlay that maps canonical source rules into the model's rule-ID namespace using strong semantic evidence from provenance, rule names, constraints, and relations. Weak chunk-only candidates are retained for audit but excluded from the main reference projection. The full architecture benchmark contains 30 tasks, while the 25-task strict-common subset contains only tasks whose expected surviving rules are strongly aligned in all three rule libraries. The five excluded tasks are not removed from the benchmark; they are excluded only from the strict-common diagnostic because at least one expected rule cannot be strongly aligned in DeepSeek or Xiaomi MIMO. These results should be interpreted conservatively: they do not show that end-to-end rule extraction is solved. They show that, when strong aligned rules are available, CTHR can be evaluated fairly across rule-library namespaces and retains partial robustness to rule libraries generated by different models.

## Claim Boundary

Safe claim: the overlay protocol prevents DeepSeek and Xiaomi MIMO from being scored as zero merely because their rule IDs differ from Qwen's namespace, and the strict-common subset isolates the effect of downstream reasoning when reference-rule alignment is available.

Unsafe claim: the experiment proves complete end-to-end rule extraction reliability. The unresolved rules, weak-only candidates, candidate-zero cases, and unsupported tasks show that rule-library generation remains a source of error.

