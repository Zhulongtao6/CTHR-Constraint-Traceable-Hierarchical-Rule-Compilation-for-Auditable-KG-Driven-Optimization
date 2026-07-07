| Dataset | Method | Method type | Rule Precision | Rule Recall | Formal CSR | Sem-CSR | False accept | Invalid cases |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Aviation | Flat baseline | flat | 8.9% | 91.7% | 0.0% | 0.0% | 0.0% | 150/150 (100.0%) |
| Aviation | Native ASP + clingo | native_symbolic | 7.5% | 19.3% | 0.0% | 0.0% | 0.0% | 150/150 (100.0%) |
| Aviation | Native CP-SAT + OR-Tools | native_symbolic | 3.4% | 8.3% | 36.0% | 10.7% | 25.3% | 134/150 (89.3%) (96 unsupported) |
| Aviation | Native SCIP | native_symbolic | 3.4% | 8.3% | 36.0% | 8.0% | 28.0% | 138/150 (92.0%) (96 unsupported) |
| Aviation | CTHR default | cthr_semantic_modeling | 74.7% | 70.7% | 89.3% | 81.3% | 8.0% | 28/150 (18.7%) (16 unsupported) |
| Aviation | CTHR-style ASP + clingo | cthr_semantic_modeling | 74.7% | 70.7% | 0.0% | 0.0% | 0.0% | 150/150 (100.0%) |
| Aviation | CTHR-style CP-SAT + OR-Tools | cthr_semantic_modeling | 74.7% | 70.7% | 71.3% | 63.3% | 8.0% | 55/150 (36.7%) (31 unsupported) |
| Aviation | CTHR-style SCIP | cthr_semantic_modeling | 74.7% | 70.7% | 89.3% | 81.3% | 8.0% | 28/150 (18.7%) (16 unsupported) |
