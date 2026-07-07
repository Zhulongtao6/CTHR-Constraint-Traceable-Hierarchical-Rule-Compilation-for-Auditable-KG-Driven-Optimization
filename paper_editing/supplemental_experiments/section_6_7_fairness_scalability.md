# Section 6.7 Fairness and Controlled Scalability Supplement

Date: 2026-06-24

## Scope

This supplement addresses two reviewer risks: whether membership/probe evaluation is biased toward a method, and whether the current submission lacks controlled scale evidence. It is deliberately framed as a controlled supplemental experiment, not as industrial-scale evidence.

The unified probe evaluation constructs one shared probe set per task and reuses it for every method. The probe set contains midpoint, variable-bound, near-boundary, random-box, and reference-feasible repaired-anchor points. The archived submission-ready artifacts do not store full decision vectors for each method: `returned_solution` is a Boolean field in the certificate table, not a vector. Therefore the requested union of returned method solutions cannot be reconstructed without re-running and extending the Table 1/2 runners to persist decision vectors. This limitation is reported explicitly rather than silently fabricating returned-solution probes.

For each probe, every method is evaluated using the public scenario constraints plus executable templates for the rule IDs selected by that method in Table 1. The source-semantic label is computed using evaluator-only source-semantic reference constraints. Thus all methods share the same probes and the same source oracle; only their own constructed feasible-region membership decision differs.

## Unified Probe Evaluation

| Dataset | Method | Method type | Probe evals | Source feasible | Method accepts | Semantic consistency | False accept | False reject | Accepted-point Sem-CSR |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Aviation | Flat baseline | flat | 1026 | 138/1026 (13.5%) | 126/1026 (12.3%) | 98.8% | 0.0% | 1.2% | 100.0% |
| Aviation | Native ASP + clingo | native symbolic | 1026 | 138/1026 (13.5%) | 174/1026 (17.0%) | 94.7% | 4.4% | 0.9% | 74.1% |
| Aviation | Native CP-SAT + OR-Tools | native symbolic | 1026 | 138/1026 (13.5%) | 165/1026 (16.1%) | 95.6% | 3.5% | 0.9% | 78.2% |
| Aviation | Native SCIP | native symbolic | 1026 | 138/1026 (13.5%) | 165/1026 (16.1%) | 95.6% | 3.5% | 0.9% | 78.2% |
| Aviation | CTHR default | CTHR semantic | 1026 | 138/1026 (13.5%) | 129/1026 (12.6%) | 99.1% | 0.0% | 0.9% | 100.0% |
| Aviation | CTHR-style ASP + clingo | CTHR semantic | 1026 | 138/1026 (13.5%) | 129/1026 (12.6%) | 99.1% | 0.0% | 0.9% | 100.0% |
| Aviation | CTHR-style CP-SAT + OR-Tools | CTHR semantic | 1026 | 138/1026 (13.5%) | 129/1026 (12.6%) | 99.1% | 0.0% | 0.9% | 100.0% |
| Aviation | CTHR-style SCIP | CTHR semantic | 1026 | 138/1026 (13.5%) | 129/1026 (12.6%) | 99.1% | 0.0% | 0.9% | 100.0% |
| Architecture | Flat baseline | flat | 1034 | 212/1034 (20.5%) | 226/1034 (21.9%) | 98.1% | 1.6% | 0.3% | 92.5% |
| Architecture | Native ASP + clingo | native symbolic | 1034 | 212/1034 (20.5%) | 226/1034 (21.9%) | 98.1% | 1.6% | 0.3% | 92.5% |
| Architecture | Native CP-SAT + OR-Tools | native symbolic | 1034 | 212/1034 (20.5%) | 226/1034 (21.9%) | 98.1% | 1.6% | 0.3% | 92.5% |
| Architecture | Native SCIP | native symbolic | 1034 | 212/1034 (20.5%) | 226/1034 (21.9%) | 98.1% | 1.6% | 0.3% | 92.5% |
| Architecture | CTHR default | CTHR semantic | 1034 | 212/1034 (20.5%) | 229/1034 (22.1%) | 98.4% | 1.6% | 0.0% | 92.6% |
| Architecture | CTHR-style ASP + clingo | CTHR semantic | 1034 | 212/1034 (20.5%) | 229/1034 (22.1%) | 98.4% | 1.6% | 0.0% | 92.6% |
| Architecture | CTHR-style CP-SAT + OR-Tools | CTHR semantic | 1034 | 212/1034 (20.5%) | 229/1034 (22.1%) | 98.4% | 1.6% | 0.0% | 92.6% |
| Architecture | CTHR-style SCIP | CTHR semantic | 1034 | 212/1034 (20.5%) | 229/1034 (22.1%) | 98.4% | 1.6% | 0.0% | 92.6% |

## Probe-Design Bias Check

Random-box probes are mostly source-infeasible: 4/360 (1.1%) in aviation and 21/360 (5.8%) in architecture are source-feasible. Interpreting random probes alone would therefore favor methods that reject large regions. To avoid this all-negative-label bias, the shared probe set also includes 120 reference-feasible repaired anchors in each domain. These anchors are shared by all methods and are used only for evaluation, not as method inputs.

No probe kind is method-specific. CTHR, flat, and native-symbolic methods are checked on the same task-level probe sets. The remaining differences therefore come from each method's selected rules and compiled membership region, not from different probe distributions.

Detailed probe-kind breakdowns are archived in:

- `submission_support/kbs_systematic_experiments_v1/results/section_6_7_fairness_scalability/section_6_7_fairness_scalability_summary.json`
- `submission_support/kbs_systematic_experiments_v1/results/section_6_7_fairness_scalability/section_6_7_fairness_probe_summary.csv`

## Runtime and Scale Statistics

| Dataset | Tasks | Rule-library rules | Mean candidate rules | Mean predicted valid rules | Template rules | Compiled templates | Mean feasible cells | Rule recovery time | Template assembly time (ms) | CTHR default runtime proxy (ms) | Certificate generation time | Mean certificate rules |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: |
| Aviation | 30 | 111 | 11.267 | 2.633 | 54 | 88 | 1.000 | not separately instrumented | 0.0078 | 528.548 | not separately instrumented | 2.633 |
| Architecture | 30 | 1585 | 20.733 | 1.667 | 50 | 59 | 1.000 | not separately instrumented | 0.0043 | 447.560 | not separately instrumented | 1.667 |

The current submission-ready artifacts do not separately instrument candidate-to-valid recovery time, full feasible-cell compilation time, or certificate generation time. The table therefore reports the available CTHR default end-to-end runtime proxy from Table 1, a lightweight template-assembly micro-timing for selected valid rules, solver times from Table 2, and certificate rule-chain size from Section 6.5. Future camera-ready instrumentation should add explicit timers around rule recovery, cell compilation, solver invocation, and certificate serialization.

### Solver-Backend Time

| Dataset | Solver | Mean solve time (ms) | Mean feasible cells | Cell CSR | Mean objective gap |
| --- | --- | ---: | ---: | ---: | ---: |
| Aviation | CTHR default solver | 514.533 | 1.000 | 100.0% | 0.098 |
| Aviation | ASP/clingo over CTHR cells | 311.351 | 1.000 | 100.0% | 0.000 |
| Aviation | CP-SAT + OR-Tools over CTHR cells | 1.734 | 1.000 | 100.0% | 0.000 |
| Aviation | SCIP over CTHR cells | 3.569 | 1.000 | 100.0% | 0.000 |
| Architecture | CTHR default solver | 416.322 | 1.000 | 100.0% | 0.000 |
| Architecture | ASP/clingo over CTHR cells | 265.989 | 1.000 | 100.0% | 0.000 |
| Architecture | CP-SAT + OR-Tools over CTHR cells | 2.197 | 1.000 | 100.0% | 0.000 |
| Architecture | SCIP over CTHR cells | 2.792 | 1.000 | 100.0% | 0.000 |

## Controlled Candidate-Noise Stress Test

This test expands the broad candidate set by 2x, 4x, and 8x with irrelevant distractor rules and evaluates the resulting noisy candidate feasible region on the same shared probes. It measures controlled candidate-noise sensitivity of template assembly and membership behavior. It is not a full industrial scalability benchmark and does not claim that candidate-to-valid recovery was re-trained or re-tuned under expanded candidate sets.

| Dataset | Candidate factor | Mean candidate rules | Mean compiled constraints | Template assembly time (ms) | Probe evals | Source feasible | Noisy-candidate accepts | Semantic consistency | False accept | False reject | Accepted-point Sem-CSR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Aviation | 1x | 11.267 | 4.967 | 0.0228 | 1026 | 135/1026 (13.2%) | 123/1026 (12.0%) | 98.8% | 0.0% | 1.2% | 100.0% |
| Aviation | 2x | 22.533 | 5.000 | 0.0409 | 1026 | 135/1026 (13.2%) | 123/1026 (12.0%) | 98.8% | 0.0% | 1.2% | 100.0% |
| Aviation | 4x | 44.067 | 5.067 | 0.0727 | 1026 | 135/1026 (13.2%) | 123/1026 (12.0%) | 98.8% | 0.0% | 1.2% | 100.0% |
| Aviation | 8x | 57.367 | 5.100 | 0.0845 | 1026 | 135/1026 (13.2%) | 119/1026 (11.6%) | 98.4% | 0.0% | 1.6% | 100.0% |
| Architecture | 1x | 20.733 | 4.033 | 0.0138 | 1034 | 212/1034 (20.5%) | 224/1034 (21.7%) | 98.3% | 1.5% | 0.3% | 93.3% |
| Architecture | 2x | 41.467 | 4.033 | 0.0348 | 1034 | 212/1034 (20.5%) | 224/1034 (21.7%) | 98.3% | 1.5% | 0.3% | 93.3% |
| Architecture | 4x | 64.267 | 4.033 | 0.0551 | 1034 | 212/1034 (20.5%) | 224/1034 (21.7%) | 98.3% | 1.5% | 0.3% | 93.3% |
| Architecture | 8x | 67.000 | 4.033 | 0.0584 | 1034 | 212/1034 (20.5%) | 224/1034 (21.7%) | 98.3% | 1.5% | 0.3% | 93.3% |

## Safe Paper Wording

```latex
We additionally evaluate feasible-region membership on a shared set of probe points for every task. The probe set is constructed once per task and reused by all methods; it contains midpoint, variable-bound, near-boundary, random, and source-feasible repaired-anchor points. A method's membership decision is compared with the evaluator-only source-semantic reference. This experiment is intended to check for probe-design bias in membership evaluation, not to replace final-solution Sem-CSR.

We also report controlled scale statistics and a candidate-noise stress test in which irrelevant distractor rules expand the broad candidate set by 2x, 4x, and 8x. These results should be interpreted as controlled scalability evidence for the benchmark setting, not as an industrial-scale deployment claim.
```

## Reproducibility

Runner:

`submission_support/kbs_systematic_experiments_v1/scripts/run_section_6_7_fairness_scalability.py`

Detailed generated artifacts:

- `submission_support/kbs_systematic_experiments_v1/results/section_6_7_fairness_scalability/section_6_7_fairness_scalability.md`
- `submission_support/kbs_systematic_experiments_v1/results/section_6_7_fairness_scalability/section_6_7_fairness_scalability_summary.json`
- `submission_support/kbs_systematic_experiments_v1/results/section_6_7_fairness_scalability/section_6_7_fairness_probe_summary.csv`
- `submission_support/kbs_systematic_experiments_v1/results/section_6_7_fairness_scalability/section_6_7_candidate_noise_stress.csv`
