# Section 6.5 Certificate Robustness Addendum

This note supplements the Section 6.5 audit-traceability experiment. Its purpose
is to make the certificate evidence more concrete and to reduce the reviewer
risk that the certificate is interpreted as a mere pointer, a native solver
proof log, or a human audit study.

The analysis is based on the archived submission-ready results in
`submission_support/kbs_systematic_experiments_v1/results/submission_ready_20260528/section_6_5_audit_traceability/`
and the generator script
`submission_support/kbs_systematic_experiments_v1/scripts/run_section_6_5_audit_traceability.py`.
The certificate is interpreted only as a CTHR metadata object attached to the
selected compiled cell and selected rule chain. It is not a proof certificate
emitted by ASP/clingo, CP-SAT/OR-Tools, or SCIP, and the statistics below do not
measure human audit time or domain-expert usability.

## Main Finding

The Section 6.5 certificate is more than an isolated rule-id pointer. For every
tested backend, it records the selected CTHR rule chain for the active compiled
cell, and every rule identifier in that chain resolves to a provenance-bearing
rule-library record. Across the four backend rows, the same selected certificate
chain is retained for all 60 tasks.

## Certificate Coverage and Rule-Chain Size

| Method | Certificates | Certificate coverage | Rules in certificates | Rules per certificate | Rule-count range |
|---|---:|---:|---:|---:|---:|
| CTHR default | 60/60 | 100.0% | 129 | 2.15 | 1-6 |
| CTHR+ASP/clingo | 60/60 | 100.0% | 129 | 2.15 | 1-6 |
| CTHR+CP-SAT/OR-Tools | 60/60 | 100.0% | 129 | 2.15 | 1-6 |
| CTHR+SCIP | 60/60 | 100.0% | 129 | 2.15 | 1-6 |

The rule-chain size is not inflated by backend choice because all four rows use
the same CTHR compiled cells and selected valid-rule chains. For the CTHR
default row, the domain-level distribution is:

| Dataset | Certificates | Certificate rules | Mean rules/cert. | Median | Min | Max | Rule-count distribution |
|---|---:|---:|---:|---:|---:|---:|---|
| Aviation | 30 | 79 | 2.63 | 2 | 1 | 6 | 1:4, 2:14, 3:6, 4:2, 5:3, 6:1 |
| Architecture | 30 | 50 | 1.67 | 2 | 1 | 3 | 1:11, 2:18, 3:1 |

## Rule-Library Resolution and Provenance Fields

For each method, 129 certificate rule mentions are checked. All 129 resolve to
the current domain-specific rule library and all 129 have non-empty provenance
fields. Across the four methods, this corresponds to 516/516 rule mentions with
valid rule-library provenance.

Using one method row as the non-duplicated reference, the 129 certificate rule
mentions correspond to 113 unique domain-qualified rule records. Their
provenance coverage is:

| Provenance check | Mentions | Unique rule records |
|---|---:|---:|
| Rule record found in latest rule library | 129/129 | 113/113 |
| Non-empty provenance field | 129/129 | 113/113 |
| Source document present | 129/129 | 113/113 |
| Source section present and not `unknown` | 123/129 | 108/113 |
| Known PDF page present | 22/129 | 18/113 |
| KG/source chunk identifier present | 129/129 | 113/113 |
| Source node identifier present | 119/129 | 103/113 |
| Constraint or relation evidence field present | 129/129 | 113/113 |

This supports the narrower claim that the certificate provides a
machine-checkable route from optimized decision to selected CTHR rule chain, then
to rule-library records, and then to document/section/chunk-level provenance
metadata. It should not be phrased as guaranteed page-level PDF traceability,
because known PDF page fields are present only for a subset of rule mentions.

## Complete Valid-Chain Trace

| Method | Complete selected rule chains | Rate |
|---|---:|---:|
| CTHR default | 60/60 | 100.0% |
| CTHR+ASP/clingo | 60/60 | 100.0% |
| CTHR+CP-SAT/OR-Tools | 60/60 | 100.0% |
| CTHR+SCIP | 60/60 | 100.0% |

The completeness check verifies that the certificate rule list equals the
selected CTHR valid-rule chain used for the active compiled cell. It does not
claim that the downstream solver independently derived that rule chain.

## Backend Invariance of the Certificate Chain

For each of the 60 tasks, the certificate rule chain is identical across the
four backend rows. The active cell identifier is also identical across backends
for all 60 tasks.

| Invariance check | Count |
|---|---:|
| Same certificate rule chain across all four methods | 60/60 |
| Same active cell id across all four methods | 60/60 |
| Same CTHR compile source across all task-method rows | 240/240 |

The common compile source is
`["cthr_predicted_valid_rules_plus_compiled_templates"]`. This supports the
division of labor used in the paper: CTHR supplies the traceable compiled cell
and rule chain, while ASP/clingo, CP-SAT/OR-Tools, and SCIP are backend
optimizers over that CTHR-provided structure.

## Numerical Constraint Tolerance

Numerical constraint checking does require tolerance because several backends
return floating-point decisions or use continuous local refinement. The current
Table 2 backend script defines `FEAS_TOL = 1e-4` and applies it to decision
bounds, inequalities, strict inequalities, equalities, and disequalities during
final cell-validity checking.

Recommended reporting:

- Use `1e-4` as the default feasibility tolerance for the current benchmark.
- Apply the same tolerance after every backend, including ASP/clingo,
  CP-SAT/OR-Tools, and SCIP, so that backend comparisons are not affected by
  solver-specific numerical noise.
- For future larger-scale variables, prefer a mixed tolerance such as
  `abs(residual) <= 1e-4 + 1e-6 * scale`, where `scale` is the magnitude of the
  right-hand side or relevant variable range.
- State clearly that this tolerance is used for numeric cell-membership
  validation. It is separate from certificate traceability, which checks rule
  chain and provenance linkage rather than numerical proof logs.

## Multiple Feasible Cells

The current submission-ready Section 6.5 data do not exercise multi-cell
ambiguity. In the Table 2 per-task file, `compiled_cell_count` is 1 for all 240
task-method rows, so every tested decision is associated with a single compiled
cell.

Implementation note: the backend script contains a union-cell validity check
that scans compiled cells and returns the first cell that validates the returned
decision. Therefore, the current system records one active cell and one selected
certificate chain. If a future task produces overlapping feasible cells and the
same decision belongs to multiple cells, the current report would not list every
matching cell in the certificate. This should be stated as a limitation rather
than as an evaluated capability.

Suggested limitation wording:

> The present certificate records the active CTHR compiled cell selected by the
> runtime and the corresponding valid-rule chain. In the current 60-task
> submission-ready benchmark, every task has a single compiled cell, so
> multi-cell overlap is not exercised. For future tasks with overlapping valid
> feasible cells, the certificate should be extended to record all matching
> cells or an explicit deterministic tie-breaking policy.

## Suggested Paper Wording

The following sentence can be added to Section 6.5 or the discussion:

> In addition to coverage, the certificates contain an average of 2.15 rule
> identifiers per task-method row; for every backend, all 129 certificate rule
> mentions resolve to provenance-bearing rule-library records, and the selected
> rule chain is identical across all four backend variants for every task.

The following caution should also be included:

> These statistics evaluate machine-checkable linkage between returned
> decisions, selected CTHR rule chains, and source-linked rule-library records.
> They do not evaluate human audit time, domain-expert usability, or native
> solver proof generation.

