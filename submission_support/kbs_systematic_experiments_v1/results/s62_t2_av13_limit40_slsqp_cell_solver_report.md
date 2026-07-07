# Section 6.2 Table 2: Solver Backends over CTHR Compiled Cells

## Dataset

- Aviation Rule Relation Balanced v13 150: 150 tasks.

## Solver Backends

- CTHR default solver: differential evolution followed by local SLSQP refinement over the CTHR cell union.
- ASP/clingo over CTHR cells: encodes CTHR cell selection in ASP and uses clingo to enumerate active-cell choices, then performs continuous refinement inside the selected CTHR cell. clingo itself is used for symbolic cell selection, not real-valued nonlinear optimization.
- CP-SAT + OR-Tools over CTHR cells: consumes the same CTHR compiled constraints with integer-scaled symbolic encoding.
- SCIP over CTHR cells: consumes the same CTHR compiled constraints with continuous nonlinear constraint support.

## Main Result

| Dataset | Solver over CTHR cells | Solve | Cell CSR | Objective gap |
| --- | --- | ---: | ---: | ---: |
| Aviation Rule Relation Balanced v13 150 | CTHR default solver | 96.667 | 96.667 | 0.0% |
| Aviation Rule Relation Balanced v13 150 | ASP/clingo over CTHR cells | 0.0 | 0.0 | N/A |
| Aviation Rule Relation Balanced v13 150 | SLSQP over CTHR cells | 6.667 | 6.667 | 14.925374% |
| Aviation Rule Relation Balanced v13 150 | CP-SAT + OR-Tools over CTHR cells | 96.667 | 95.333 | 0.717636% |
| Aviation Rule Relation Balanced v13 150 | SCIP over CTHR cells | 96.667 | 96.667 | 0.0% |

## Objective Gap Reference

For each task, the reference best objective is the best objective among cell-valid solutions returned by the evaluated CTHR-cell solver backends. If no backend returns a cell-valid solution, the task-level objective gap is N/A.

## Unsupported / N/A Reasons

- ASP/clingo over CTHR cells: 150 task-level unsupported or invalid records.
- CP-SAT + OR-Tools over CTHR cells: 7 task-level unsupported or invalid records.
- CTHR default solver: 5 task-level unsupported or invalid records.
- SCIP over CTHR cells: 5 task-level unsupported or invalid records.
- SLSQP over CTHR cells: 140 task-level unsupported or invalid records.

| Dataset | task_id | Solver | reason |
| --- | --- | --- | --- |
| Aviation Rule Relation Balanced v13 150 | AVI_SMIX_001 | ASP/clingo over CTHR cells | no_cell_valid_continuous_solution_after_clingo_selection |
| Aviation Rule Relation Balanced v13 150 | AVI_SMIX_001 | SLSQP over CTHR cells | slsqp_no_cell_valid_solution:1_failed_cells |
| Aviation Rule Relation Balanced v13 150 | AVI_SMIX_002 | CTHR default solver | AVI_SMIX_002_cthr_compiled_cell_1:scip_infeasible |
| Aviation Rule Relation Balanced v13 150 | AVI_SMIX_002 | ASP/clingo over CTHR cells | no_cell_valid_continuous_solution_after_clingo_selection |
| Aviation Rule Relation Balanced v13 150 | AVI_SMIX_002 | SLSQP over CTHR cells | slsqp_no_cell_valid_solution:1_failed_cells |
| Aviation Rule Relation Balanced v13 150 | AVI_SMIX_002 | CP-SAT + OR-Tools over CTHR cells | AVI_SMIX_002_cthr_compiled_cell_1:cp_sat_infeasible |
| Aviation Rule Relation Balanced v13 150 | AVI_SMIX_002 | SCIP over CTHR cells | AVI_SMIX_002_cthr_compiled_cell_1:scip_infeasible |
| Aviation Rule Relation Balanced v13 150 | AVI_SMIX_003 | ASP/clingo over CTHR cells | no_cell_valid_continuous_solution_after_clingo_selection |
| Aviation Rule Relation Balanced v13 150 | AVI_SMIX_003 | SLSQP over CTHR cells | slsqp_no_cell_valid_solution:1_failed_cells |
| Aviation Rule Relation Balanced v13 150 | AVI_SMIX_004 | ASP/clingo over CTHR cells | no_cell_valid_continuous_solution_after_clingo_selection |
| Aviation Rule Relation Balanced v13 150 | AVI_SMIX_005 | ASP/clingo over CTHR cells | no_cell_valid_continuous_solution_after_clingo_selection |
| Aviation Rule Relation Balanced v13 150 | AVI_SMIX_005 | SLSQP over CTHR cells | slsqp_no_cell_valid_solution:1_failed_cells |

## Conclusion

This experiment isolates solver consumption of CTHR compiled cells. A high Cell CSR means the backend can consume the exported CTHR cell geometry without redoing rule selection.
