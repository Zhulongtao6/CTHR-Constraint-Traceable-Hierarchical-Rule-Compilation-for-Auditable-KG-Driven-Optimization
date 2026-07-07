| Dataset | Method | Method type | Rule Precision | Rule Recall | Formal CSR | Sem-CSR | False accept | Invalid cases |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Architecture | Flat baseline | flat | 6.8% | 99.0% | 66.0% | 78.0% | 0.0% | 11/50 (22.0%) |
| Architecture | Native ASP + clingo | native_symbolic | 75.7% | 90.0% | 100.0% | 92.0% | 8.0% | 4/50 (8.0%) |
| Architecture | Native MILP + HiGHS | native_symbolic | 51.7% | 62.0% | 72.0% | 64.0% | 8.0% | 18/50 (36.0%) (14 unsupported) |
| Architecture | Native CP-SAT + OR-Tools | native_symbolic | 75.7% | 90.0% | 100.0% | 92.0% | 8.0% | 4/50 (8.0%) |
| Architecture | Native SCIP | native_symbolic | 75.7% | 90.0% | 100.0% | 92.0% | 8.0% | 4/50 (8.0%) |
| Architecture | CTHR default | cthr_semantic_modeling | 95.3% | 95.0% | 96.0% | 96.0% | 0.0% | 2/50 (4.0%) (2 unsupported) |
| Architecture | CTHR-style ASP + clingo | cthr_semantic_modeling | 95.3% | 95.0% | 96.0% | 96.0% | 0.0% | 2/50 (4.0%) (2 unsupported) |
| Architecture | CTHR-style HiGHS | cthr_semantic_modeling | 95.3% | 95.0% | 96.0% | 96.0% | 0.0% | 2/50 (4.0%) (2 unsupported) |
| Architecture | CTHR-style CP-SAT + OR-Tools | cthr_semantic_modeling | 95.3% | 95.0% | 96.0% | 96.0% | 0.0% | 2/50 (4.0%) (2 unsupported) |
| Architecture | CTHR-style SCIP | cthr_semantic_modeling | 95.3% | 95.0% | 96.0% | 96.0% | 0.0% | 2/50 (4.0%) (2 unsupported) |