# Section 6.2 Table 2: Solver Backends over CTHR Compiled Cells

## Dataset

- Aviation: 150 tasks.

## Solver Backends

- CTHR default solver: differential evolution followed by local SLSQP refinement over the CTHR cell union.
- ASP/clingo over CTHR cells: encodes CTHR cell selection in ASP and uses clingo to enumerate active-cell choices, then performs continuous refinement inside the selected CTHR cell. clingo itself is used for symbolic cell selection, not real-valued nonlinear optimization.
- HiGHS over CTHR cells: consumes the same CTHR compiled constraints when they can be expressed as linear constraints.
- CP-SAT + OR-Tools over CTHR cells: consumes the same CTHR compiled constraints with integer-scaled symbolic encoding.
- SCIP over CTHR cells: consumes the same CTHR compiled constraints with continuous nonlinear constraint support.

## Main Result

| Dataset | Solver over CTHR cells | Solve | Cell CSR | Objective gap |
| --- | --- | ---: | ---: | ---: |
| Aviation | CTHR default solver | 96.667 | 96.667 | 0.0% |
| Aviation | ASP/clingo over CTHR cells | 0.0 | 0.0 | N/A |
| Aviation | SLSQP over CTHR cells | 6.667 | 6.667 | 14.925374% |
| Aviation | HiGHS over CTHR cells | 84.667 | 84.667 | 1e-05% |
| Aviation | CP-SAT + OR-Tools over CTHR cells | 96.667 | 95.333 | 0.717636% |
| Aviation | SCIP over CTHR cells | 96.667 | 96.667 | 0.0% |

## Objective Gap Reference

For each task, the reference best objective is the best objective among cell-valid solutions returned by the evaluated CTHR-cell solver backends. If no backend returns a cell-valid solution, the task-level objective gap is N/A.

## Unsupported / N/A Reasons

- ASP/clingo over CTHR cells: 150 task-level unsupported or invalid records.
- CP-SAT + OR-Tools over CTHR cells: 7 task-level unsupported or invalid records.
- CTHR default solver: 5 task-level unsupported or invalid records.
- HiGHS over CTHR cells: 23 task-level unsupported or invalid records.
- SCIP over CTHR cells: 5 task-level unsupported or invalid records.
- SLSQP over CTHR cells: 140 task-level unsupported or invalid records.

| Dataset | task_id | Solver | reason |
| --- | --- | --- | --- |
| Aviation | AVI_SMIX_001 | ASP/clingo over CTHR cells | no_cell_valid_continuous_solution_after_clingo_selection |
| Aviation | AVI_SMIX_001 | SLSQP over CTHR cells | slsqp_no_cell_valid_solution:1_failed_cells |
| Aviation | AVI_SMIX_002 | CTHR default solver | AVI_SMIX_002_cthr_compiled_cell_1:scip_infeasible |
| Aviation | AVI_SMIX_002 | ASP/clingo over CTHR cells | no_cell_valid_continuous_solution_after_clingo_selection |
| Aviation | AVI_SMIX_002 | SLSQP over CTHR cells | slsqp_no_cell_valid_solution:1_failed_cells |
| Aviation | AVI_SMIX_002 | HiGHS over CTHR cells | AVI_SMIX_002_cthr_compiled_cell_1:linprog_The problem is infeasible. (HiGHS Status 8: model_status is Infeasible; primal_status is At lower/fixed bound) |
| Aviation | AVI_SMIX_002 | CP-SAT + OR-Tools over CTHR cells | AVI_SMIX_002_cthr_compiled_cell_1:cp_sat_infeasible |
| Aviation | AVI_SMIX_002 | SCIP over CTHR cells | AVI_SMIX_002_cthr_compiled_cell_1:scip_infeasible |
| Aviation | AVI_SMIX_003 | ASP/clingo over CTHR cells | no_cell_valid_continuous_solution_after_clingo_selection |
| Aviation | AVI_SMIX_003 | SLSQP over CTHR cells | slsqp_no_cell_valid_solution:1_failed_cells |
| Aviation | AVI_SMIX_004 | ASP/clingo over CTHR cells | no_cell_valid_continuous_solution_after_clingo_selection |
| Aviation | AVI_SMIX_004 | HiGHS over CTHR cells | AVI_SMIX_004_cthr_compiled_cell_1:variable multiplication is nonlinear |

## Conclusion

This experiment isolates solver consumption of CTHR compiled cells. A high Cell CSR means the backend can consume the exported CTHR cell geometry without redoing rule selection.
