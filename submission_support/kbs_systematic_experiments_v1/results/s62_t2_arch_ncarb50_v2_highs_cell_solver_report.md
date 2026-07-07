# Section 6.2 Table 2: Solver Backends over CTHR Compiled Cells

## Dataset

- Architecture: 50 tasks.

## Solver Backends

- CTHR default solver: differential evolution followed by local SLSQP refinement over the CTHR cell union.
- ASP/clingo over CTHR cells: encodes CTHR cell selection in ASP and uses clingo to enumerate active-cell choices, then performs continuous refinement inside the selected CTHR cell. clingo itself is used for symbolic cell selection, not real-valued nonlinear optimization.
- HiGHS over CTHR cells: consumes the same CTHR compiled constraints when they can be expressed as linear constraints.
- CP-SAT + OR-Tools over CTHR cells: consumes the same CTHR compiled constraints with integer-scaled symbolic encoding.
- SCIP over CTHR cells: consumes the same CTHR compiled constraints with continuous nonlinear constraint support.

## Main Result

| Dataset | Solver over CTHR cells | Solve | Cell CSR | Objective gap |
| --- | --- | ---: | ---: | ---: |
| Architecture | CTHR default solver | 100.0 | 100.0 | 1.999532% |
| Architecture | ASP/clingo over CTHR cells | 100.0 | 100.0 | 2.003102% |
| Architecture | SLSQP over CTHR cells | 100.0 | 100.0 | 2.003342% |
| Architecture | HiGHS over CTHR cells | 72.0 | 72.0 | 2.777088% |
| Architecture | CP-SAT + OR-Tools over CTHR cells | 100.0 | 100.0 | 2.004783% |
| Architecture | SCIP over CTHR cells | 100.0 | 100.0 | 0.003819% |

## Objective Gap Reference

For each task, the reference best objective is the best objective among cell-valid solutions returned by the evaluated CTHR-cell solver backends. If no backend returns a cell-valid solution, the task-level objective gap is N/A.

## Unsupported / N/A Reasons

- HiGHS over CTHR cells: 14 task-level unsupported or invalid records.

| Dataset | task_id | Solver | reason |
| --- | --- | --- | --- |
| Architecture | ARCH_FKG_51 | HiGHS over CTHR cells | ARCH_FKG_51_cthr_compiled_cell_1:division by variable or zero is nonlinear |
| Architecture | ARCH_FKG_52 | HiGHS over CTHR cells | ARCH_FKG_52_cthr_compiled_cell_1:variable multiplication is nonlinear |
| Architecture | ARCH_FKG_55 | HiGHS over CTHR cells | ARCH_FKG_55_cthr_compiled_cell_1:unsupported linear expression node Call |
| Architecture | ARCH_FKG_56 | HiGHS over CTHR cells | ARCH_FKG_56_cthr_compiled_cell_1:variable multiplication is nonlinear |
| Architecture | ARCH_FKG_58 | HiGHS over CTHR cells | ARCH_FKG_58_cthr_compiled_cell_1:variable multiplication is nonlinear |
| Architecture | ARCH_FKG_61 | HiGHS over CTHR cells | ARCH_FKG_61_cthr_compiled_cell_1:variable multiplication is nonlinear |
| Architecture | ARCH_FKG_63 | HiGHS over CTHR cells | ARCH_FKG_63_cthr_compiled_cell_1:division by variable or zero is nonlinear |
| Architecture | ARCH_FKG_64 | HiGHS over CTHR cells | ARCH_FKG_64_cthr_compiled_cell_1:unsupported linear expression node Call |
| Architecture | ARCH_FKG_72 | HiGHS over CTHR cells | ARCH_FKG_72_cthr_compiled_cell_1:unsupported linear expression node Call |
| Architecture | ARCH_FKG_73 | HiGHS over CTHR cells | ARCH_FKG_73_cthr_compiled_cell_1:division by variable or zero is nonlinear |
| Architecture | ARCH_FKG_95 | HiGHS over CTHR cells | ARCH_FKG_95_cthr_compiled_cell_1:unsupported linear expression node Call |
| Architecture | ARCH_FKG_96 | HiGHS over CTHR cells | ARCH_FKG_96_cthr_compiled_cell_1:variable multiplication is nonlinear |

## Conclusion

This experiment isolates solver consumption of CTHR compiled cells. A high Cell CSR means the backend can consume the exported CTHR cell geometry without redoing rule selection.
