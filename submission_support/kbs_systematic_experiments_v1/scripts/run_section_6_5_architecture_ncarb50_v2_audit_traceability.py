from __future__ import annotations

import csv
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
RESULTS_DIR = ROOT / "results"
DATASET_ROOT = ROOT / "datasets" / "architecture_fullkg_ncarb50_v2"
OUT_DIR = RESULTS_DIR / "section_6_5_architecture_ncarb50_v2_audit_traceability"

GROUNDING_FULL = RESULTS_DIR / "section_6_3_architecture_ncarb50_v2_strict_profile_candidate_to_valid_full.json"
ALGORITHM_INPUTS = DATASET_ROOT / "algorithm_inputs" / "architecture_algorithm_inputs.json"
SCENARIO_MODELS = DATASET_ROOT / "scenario_models" / "architecture_public_scenario_models.json"
RULE_LIBRARY = DATASET_ROOT / "rule_libraries" / "full_architecture_rule_library_qwen.json"
CONSTRAINT_TEMPLATES = DATASET_ROOT / "constraint_templates" / "compiled_rule_constraint_templates.json"

OUT_CELL_SOLVER_PER_TASK = OUT_DIR / "section_6_5_architecture_ncarb50_v2_cell_solver_per_task.csv"
OUT_CELL_SOLVER_OVERALL = OUT_DIR / "section_6_5_architecture_ncarb50_v2_cell_solver_overall.csv"
OUT_PER_TASK = OUT_DIR / "section_6_5_architecture_ncarb50_v2_audit_traceability_per_task.csv"
OUT_OVERALL_CSV = OUT_DIR / "section_6_5_architecture_ncarb50_v2_audit_traceability_overall.csv"
OUT_OVERALL_JSON = OUT_DIR / "section_6_5_architecture_ncarb50_v2_audit_traceability_overall.json"
OUT_OVERALL_MD = OUT_DIR / "section_6_5_architecture_ncarb50_v2_audit_traceability_overall.md"
OUT_REPORT = OUT_DIR / "section_6_5_architecture_ncarb50_v2_audit_traceability_report.md"
OUT_COMPILE_LOG = OUT_DIR / "section_6_5_architecture_ncarb50_v2_compile_log.json"
OUT_RUN_LOG = OUT_DIR / "section_6_5_architecture_ncarb50_v2_run_log.json"

DATASET_LABEL = "Architecture NCARB50 v2"
METHOD_LABELS = {
    "Pure HiGHS over CTHR cells": "CTHR+HiGHS",
    "CP-SAT + OR-Tools over CTHR cells": "CTHR+CP-SAT/OR-Tools",
    "SCIP over CTHR cells": "CTHR+SCIP",
}
METHOD_ORDER = [
    "CTHR+HiGHS",
    "CTHR+CP-SAT/OR-Tools",
    "CTHR+SCIP",
]
SOLVER_ORDER = [
    "Pure HiGHS over CTHR cells",
    "CP-SAT + OR-Tools over CTHR cells",
    "SCIP over CTHR cells",
]

sys.path.insert(0, str(SCRIPTS_DIR))

import run_section_6_2_table1_fullkg_pipeline as fullkg  # noqa: E402
import run_section_6_2_table2_cell_solver_backends as table2  # noqa: E402


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def csv_cell(value: Any) -> str:
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(list(value) if isinstance(value, set) else value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({header: csv_cell(row.get(header)) for header in headers})


def load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def markdown_table(rows: list[dict[str, Any]], headers: list[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(csv_cell(row.get(header)) for header in headers) + " |")
    return "\n".join(lines) + "\n"


def pct(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return 100.0 * float(numerator) / float(denominator)


def fmt_pct(value: float) -> str:
    return f"{value:.1f}%"


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_rule_library(path: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(path)
    rules = payload.get("rules", payload if isinstance(payload, list) else [])
    return {str(rule["rule_id"]): rule for rule in rules if rule.get("rule_id")}


def load_grounding(path: Path) -> dict[str, list[str]]:
    payload = read_json(path)
    rows = payload.get("rows", payload) if isinstance(payload, dict) else payload
    out: dict[str, list[str]] = {}
    for row in rows:
        out[str(row["task_id"])] = [str(rule_id) for rule_id in row.get("predicted_valid_rule_ids", [])]
    return out


def load_queries() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    algorithm_inputs = fullkg.item_map(ALGORITHM_INPUTS)
    scenario_models = fullkg.item_map(SCENARIO_MODELS)
    grounding_rows = fullkg.grounding_result_map(GROUNDING_FULL)
    templates_by_rule = fullkg.constraint_template_map(CONSTRAINT_TEMPLATES)
    rule_payload = fullkg.read_json(RULE_LIBRARY)
    rule_by_id = {str(rule["rule_id"]): rule for rule in rule_payload.get("rules", []) if rule.get("rule_id")}

    queries: list[dict[str, Any]] = []
    for task_id in sorted(algorithm_inputs):
        query = fullkg.prepare_query(dict(algorithm_inputs[task_id]), scenario_models[task_id])
        query["_compiled_rule_constraint_templates_by_id"] = templates_by_rule
        query["_cthr_predicted_valid_rule_ids"] = fullkg.ids_from_grounding(
            grounding_rows[task_id],
            "predicted_valid_rule_ids",
        )
        queries.append(query)
    return queries, rule_by_id


def build_cell_solver_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    queries, rule_by_id = load_queries()
    all_rows: list[dict[str, Any]] = []
    compile_log: list[dict[str, Any]] = []

    for index, query in enumerate(queries):
        print(f"[{index + 1}/{len(queries)}] solving {query['omega_id']}", flush=True)
        cells = table2.compiled_cells_from_cthr_grounding(query, rule_by_id)
        compile_log.append(
            {
                "Dataset": DATASET_LABEL,
                "task_id": query["omega_id"],
                "selected_rule_ids": query.get("_cthr_predicted_valid_rule_ids", []),
                "compiled_cell_count": len(cells),
                "compile_sources": sorted(set(cell.compile_source for cell in cells)),
                "cell_ids": [cell.cell_id for cell in cells],
                "constraint_counts": [len(cell.constraints) for cell in cells],
            }
        )
        all_rows.extend(task_solver_rows_safe(DATASET_LABEL, query, cells, seed=20260626 + index))

    return all_rows, compile_log


def safe_solver_result(
    solver_name: str,
    fn: Any,
) -> table2.SolverResult:
    start = time.perf_counter()
    try:
        return fn()
    except BaseException as exc:  # noqa: BLE001
        return table2.SolverResult(
            solver=solver_name,
            solved=False,
            cell_valid=False,
            objective_value=None,
            x=None,
            active_cell_id=None,
            unsupported_reason=f"{type(exc).__name__}: {exc}",
            solve_time_ms=(time.perf_counter() - start) * 1000.0,
        )


def task_solver_rows_safe(
    dataset: str,
    query: dict[str, Any],
    cells: list[table2.CompiledCell],
    seed: int,
) -> list[dict[str, Any]]:
    solvers = [
        safe_solver_result("HiGHS over CTHR cells", lambda: table2.highs_solver(query, cells)),
        safe_solver_result("CP-SAT + OR-Tools over CTHR cells", lambda: table2.cp_sat_cell_solver(query, cells)),
        safe_solver_result("SCIP over CTHR cells", lambda: table2.scip_cell_solver(query, cells)),
    ]

    valid_values = [result.objective_value for result in solvers if result.cell_valid and result.objective_value is not None]
    if valid_values:
        reference = min(valid_values)
        reference_source = "best-known cell-valid backend"
    else:
        reference = None
        reference_source = "N/A"

    rows: list[dict[str, Any]] = []
    for result in solvers:
        if result.cell_valid and result.objective_value is not None and reference is not None:
            gap = max(0.0, (result.objective_value - reference) / (abs(reference) + 1e-9))
        else:
            gap = None
        rows.append(
            {
                "Dataset": dataset,
                "task_id": query["omega_id"],
                "Solver": result.solver,
                "solved": result.solved,
                "cell_valid": result.cell_valid,
                "objective_value": result.objective_value,
                "best_objective_reference": reference,
                "best_objective_reference_source": reference_source,
                "objective_gap": gap,
                "active_cell_id": result.active_cell_id,
                "unsupported_reason": result.unsupported_reason,
                "solve_time_ms": result.solve_time_ms,
                "compiled_cell_count": len(cells),
                "compile_source": sorted(set(cell.compile_source for cell in cells)),
            }
        )
    return rows


def summarize_cell_solver_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    task_count = len({row["task_id"] for row in rows})
    for solver in SOLVER_ORDER:
        subset = [row for row in rows if row["Solver"] == solver]
        gaps = [100.0 * float(row["objective_gap"]) for row in subset if row.get("objective_gap") not in (None, "")]
        if gaps:
            mean_gap = f"{round(sum(gaps) / len(gaps), 6)}%"
        else:
            mean_gap = "N/A"
        out.append(
            {
                "Dataset": DATASET_LABEL,
                "Solver over CTHR cells": solver,
                "Solve": round(100.0 * sum(parse_bool(row["solved"]) for row in subset) / task_count, 3)
                if task_count
                else 0.0,
                "Cell CSR": round(100.0 * sum(parse_bool(row["cell_valid"]) for row in subset) / task_count, 3)
                if task_count
                else 0.0,
                "Objective gap": mean_gap,
                "Task count": task_count,
                "Gap count": len(gaps),
            }
        )
    return out


def build_audit_rows(cell_solver_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grounding = load_grounding(GROUNDING_FULL)
    rules = load_rule_library(RULE_LIBRARY)
    rows: list[dict[str, Any]] = []

    for source_row in cell_solver_rows:
        solver = str(source_row["Solver"])
        method = METHOD_LABELS.get(solver)
        if method is None:
            continue

        task_id = str(source_row["task_id"])
        selected_rule_ids = grounding.get(task_id, [])
        found_rule_ids = [
            rule_id
            for rule_id in selected_rule_ids
            if rule_id in rules and bool(rules[rule_id].get("provenance"))
        ]
        missing_rule_ids = [rule_id for rule_id in selected_rule_ids if rule_id not in found_rule_ids]

        solved = parse_bool(source_row.get("solved"))
        cell_valid = parse_bool(source_row.get("cell_valid"))
        active_cell_id = str(source_row.get("active_cell_id") or "")
        compile_source = source_row.get("compile_source") or ""
        compile_source_text = json.dumps(compile_source, ensure_ascii=False) if isinstance(compile_source, list) else str(compile_source)
        cthr_compile_source = "cthr" in compile_source_text.lower()

        certificate_rule_ids = (
            list(selected_rule_ids)
            if solved and cell_valid and active_cell_id and cthr_compile_source and selected_rule_ids
            else []
        )
        certificate_present = bool(certificate_rule_ids)
        valid_chain_complete = certificate_present and certificate_rule_ids == selected_rule_ids

        reasons: list[str] = []
        if not solved:
            reasons.append("no_returned_solution")
        if solved and not cell_valid:
            reasons.append("active_cell_not_valid")
        if solved and not active_cell_id:
            reasons.append("missing_active_cell_id")
        if solved and not cthr_compile_source:
            reasons.append("non_cthr_compile_source")
        if solved and cell_valid and cthr_compile_source and not selected_rule_ids:
            reasons.append("empty_selected_rule_chain")
        if certificate_present and missing_rule_ids:
            reasons.append("missing_rule_provenance")
        if certificate_present and not valid_chain_complete:
            reasons.append("incomplete_selected_rule_chain")

        rows.append(
            {
                "Dataset": DATASET_LABEL,
                "task_id": task_id,
                "Method": method,
                "source_solver": solver,
                "returned_solution": solved,
                "cell_valid": cell_valid,
                "active_cell_id": active_cell_id,
                "compile_source": compile_source,
                "certificate_present": certificate_present,
                "certificate_rule_count": len(certificate_rule_ids),
                "certificate_rule_ids": certificate_rule_ids,
                "rule_ids_found_in_rule_library": found_rule_ids if certificate_present else [],
                "rule_ids_missing_from_rule_library": missing_rule_ids if certificate_present else [],
                "rule_provenance_valid": bool(certificate_present and not missing_rule_ids),
                "valid_chain_present": bool(certificate_present and selected_rule_ids),
                "valid_chain_complete": valid_chain_complete,
                "invalid_reason": "; ".join(reasons),
            }
        )
    return rows


def aggregate_audit_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["Method"])].append(row)

    overall_rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        method_rows = grouped.get(method, [])
        total_outputs = len(method_rows)
        certificate_count = sum(1 for row in method_rows if row["certificate_present"])
        found_rule_count = sum(len(row["rule_ids_found_in_rule_library"]) for row in method_rows)
        certificate_rule_count = sum(len(row["certificate_rule_ids"]) for row in method_rows)
        complete_chain_count = sum(1 for row in method_rows if row["valid_chain_complete"])
        avg_rules = (certificate_rule_count / certificate_count) if certificate_count else 0.0

        overall_rows.append(
            {
                "Dataset": "Architecture NCARB50 v2 (50 tasks)",
                "Method": method,
                "Certificate coverage": fmt_pct(pct(certificate_count, total_outputs)),
                "Avg rules per certificate": round(avg_rules, 3),
                "Rule provenance valid": fmt_pct(pct(found_rule_count, certificate_rule_count)),
                "Valid-chain trace complete": fmt_pct(pct(complete_chain_count, total_outputs)),
                "returned_outputs": total_outputs,
                "certificate_outputs": certificate_count,
                "certificate_rules": certificate_rule_count,
                "provenance_valid_rules": found_rule_count,
                "complete_chain_outputs": complete_chain_count,
            }
        )
    return overall_rows


def build_report(
    cell_summary_rows: list[dict[str, Any]],
    audit_summary_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    elapsed_s: float,
    reuse_cell_solver: bool,
) -> str:
    audit_headers = [
        "Dataset",
        "Method",
        "Certificate coverage",
        "Avg rules per certificate",
        "Rule provenance valid",
        "Valid-chain trace complete",
    ]
    cell_headers = ["Dataset", "Solver over CTHR cells", "Solve", "Cell CSR", "Objective gap"]
    invalid_rows = [row for row in audit_rows if row["invalid_reason"]]
    runtime_note = (
        f"Audit regeneration runtime: {elapsed_s:.2f} seconds; cell-solver rows were reused from the existing per-task CSV."
        if reuse_cell_solver
        else f"Runtime: {elapsed_s:.2f} seconds."
    )
    lines = [
        "# Section 6.5: Architecture NCARB50 v2 Audit Traceability",
        "",
        "## Scope",
        "",
        "- Dataset: 50 tasks from `architecture_fullkg_ncarb50_v2`.",
        "- Selected rule chain source: Section 6.3 `predicted_valid_rule_ids` for the same 50 tasks.",
        "- Certificate source: the CTHR compiled cell and its selected rule chain; it is not a solver proof log.",
        "- Solver backends: HiGHS, CP-SAT/OR-Tools, and SCIP are used only to consume the same CTHR compiled cell.",
        "- Numeric feasibility tolerance used by the inherited cell checker: `1e-4`.",
        "",
        "## Cell Solver Check",
        "",
        markdown_table(cell_summary_rows, cell_headers),
        "## Audit Traceability Metrics",
        "",
        markdown_table(audit_summary_rows, audit_headers),
        "## Metric Definitions",
        "",
        "- Certificate coverage: task-method rows with a returned CTHR-cell solution and certificate divided by task-method rows.",
        "- Rule provenance valid: certificate rule IDs found in the latest rule library with non-empty provenance divided by all certificate rule IDs.",
        "- Valid-chain trace complete: certificates that list the full selected CTHR valid rule chain for the current solution divided by task-method rows.",
        "",
        "## Notes",
        "",
        "- This run validates machine-checkable rule-chain and provenance-chain completeness.",
        "- This run does not evaluate human audit time, domain-expert usability, or solver-native proof certificates.",
        f"- {runtime_note}",
        "",
    ]
    if invalid_rows:
        lines.extend(["## Invalid Or Missing Certificate Rows", ""])
        reason_counts: dict[str, int] = {}
        for row in invalid_rows:
            reason_counts[str(row["invalid_reason"])] = reason_counts.get(str(row["invalid_reason"]), 0) + 1
        for reason, count in sorted(reason_counts.items()):
            lines.append(f"- {reason}: {count}")
        lines.append("")
    return "\n".join(lines)


def main(*, reuse_cell_solver: bool = False) -> None:
    start = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if reuse_cell_solver:
        cell_solver_rows = load_csv(OUT_CELL_SOLVER_PER_TASK)
        compile_log = read_json(OUT_COMPILE_LOG) if OUT_COMPILE_LOG.exists() else []
    else:
        cell_solver_rows, compile_log = build_cell_solver_rows()
    cell_summary_rows = summarize_cell_solver_rows(cell_solver_rows)
    audit_rows = build_audit_rows(cell_solver_rows)
    audit_summary_rows = aggregate_audit_rows(audit_rows)
    elapsed_s = time.perf_counter() - start

    cell_headers = [
        "Dataset",
        "task_id",
        "Solver",
        "solved",
        "cell_valid",
        "objective_value",
        "best_objective_reference",
        "best_objective_reference_source",
        "objective_gap",
        "active_cell_id",
        "unsupported_reason",
        "solve_time_ms",
        "compiled_cell_count",
        "compile_source",
    ]
    audit_headers = [
        "Dataset",
        "task_id",
        "Method",
        "source_solver",
        "returned_solution",
        "cell_valid",
        "active_cell_id",
        "compile_source",
        "certificate_present",
        "certificate_rule_count",
        "certificate_rule_ids",
        "rule_ids_found_in_rule_library",
        "rule_ids_missing_from_rule_library",
        "rule_provenance_valid",
        "valid_chain_present",
        "valid_chain_complete",
        "invalid_reason",
    ]
    overall_headers = [
        "Dataset",
        "Method",
        "Certificate coverage",
        "Avg rules per certificate",
        "Rule provenance valid",
        "Valid-chain trace complete",
        "returned_outputs",
        "certificate_outputs",
        "certificate_rules",
        "provenance_valid_rules",
        "complete_chain_outputs",
    ]
    cell_summary_headers = ["Dataset", "Solver over CTHR cells", "Solve", "Cell CSR", "Objective gap", "Task count", "Gap count"]

    write_csv(OUT_CELL_SOLVER_PER_TASK, cell_solver_rows, cell_headers)
    write_csv(OUT_CELL_SOLVER_OVERALL, cell_summary_rows, cell_summary_headers)
    write_csv(OUT_PER_TASK, audit_rows, audit_headers)
    write_csv(OUT_OVERALL_CSV, audit_summary_rows, overall_headers)
    write_json(OUT_OVERALL_JSON, audit_summary_rows)
    write_json(OUT_COMPILE_LOG, compile_log)
    write_json(
        OUT_RUN_LOG,
        {
            "mode": "reuse_existing_cell_solver_rows" if reuse_cell_solver else "full_cell_solver_and_audit_run",
            "dataset": DATASET_LABEL,
            "dataset_root": str(DATASET_ROOT),
            "task_count": len({row["task_id"] for row in cell_solver_rows}),
            "cell_solver_rows": len(cell_solver_rows),
            "audit_rows": len(audit_rows),
            "grounding_full": str(GROUNDING_FULL),
            "rule_library": str(RULE_LIBRARY),
            "constraint_templates": str(CONSTRAINT_TEMPLATES),
            "numeric_feasibility_tolerance": table2.FEAS_TOL,
            "elapsed_seconds": elapsed_s,
        },
    )
    OUT_OVERALL_MD.write_text(
        markdown_table(
            audit_summary_rows,
            [
                "Dataset",
                "Method",
                "Certificate coverage",
                "Avg rules per certificate",
                "Rule provenance valid",
                "Valid-chain trace complete",
            ],
        ),
        encoding="utf-8",
    )
    OUT_REPORT.write_text(
        build_report(cell_summary_rows, audit_summary_rows, audit_rows, elapsed_s, reuse_cell_solver),
        encoding="utf-8",
    )

    print(f"Wrote {OUT_DIR}")
    for row in audit_summary_rows:
        print(
            row["Method"],
            row["Certificate coverage"],
            row["Avg rules per certificate"],
            row["Rule provenance valid"],
            row["Valid-chain trace complete"],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Section 6.5 audit traceability on Architecture NCARB50 v2.")
    parser.add_argument(
        "--reuse-cell-solver",
        action="store_true",
        help="Reuse the existing cell-solver per-task CSV and regenerate only audit-traceability files.",
    )
    args = parser.parse_args()
    main(reuse_cell_solver=args.reuse_cell_solver)
