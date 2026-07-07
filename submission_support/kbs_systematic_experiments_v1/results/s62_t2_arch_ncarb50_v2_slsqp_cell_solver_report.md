# Section 6.2 Table 2: Solver Backends over CTHR Compiled Cells

## Dataset

- Architecture-NCARB50-v2: 50 tasks.

## Solver Backends

- CTHR default solver: differential evolution followed by local SLSQP refinement over the CTHR cell union.
- ASP/clingo over CTHR cells: encodes CTHR cell selection in ASP and uses clingo to enumerate active-cell choices, then performs continuous refinement inside the selected CTHR cell. clingo itself is used for symbolic cell selection, not real-valued nonlinear optimization.
- CP-SAT + OR-Tools over CTHR cells: consumes the same CTHR compiled constraints with integer-scaled symbolic encoding.
- SCIP over CTHR cells: consumes the same CTHR compiled constraints with continuous nonlinear constraint support.

## Main Result

| Dataset | Solver over CTHR cells | Solve | Cell CSR | Objective gap |
| --- | --- | ---: | ---: | ---: |
| Architecture-NCARB50-v2 | CTHR default solver | 100.0 | 100.0 | 1.999532% |
| Architecture-NCARB50-v2 | ASP/clingo over CTHR cells | 100.0 | 100.0 | 2.003102% |
| Architecture-NCARB50-v2 | SLSQP over CTHR cells | 100.0 | 100.0 | 2.003342% |
| Architecture-NCARB50-v2 | CP-SAT + OR-Tools over CTHR cells | 100.0 | 100.0 | 2.004783% |
| Architecture-NCARB50-v2 | SCIP over CTHR cells | 100.0 | 100.0 | 0.003819% |

## Objective Gap Reference

For each task, the reference best objective is the best objective among cell-valid solutions returned by the evaluated CTHR-cell solver backends. If no backend returns a cell-valid solution, the task-level objective gap is N/A.

## Unsupported / N/A Reasons

- None.

## Conclusion

This experiment isolates solver consumption of CTHR compiled cells. A high Cell CSR means the backend can consume the exported CTHR cell geometry without redoing rule selection.
