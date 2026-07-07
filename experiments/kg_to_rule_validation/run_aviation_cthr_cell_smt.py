from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution

from baselines.cthr_cell_smt import (
    CELL_SELECTION_MODES,
    CthrCellSmtFormula,
    build_cthr_cell_smt_formula,
    check_membership,
    optimize_with_z3,
)
from run_aviation_dataset_experiments import (
    FEASIBLE_LABELS_PATH,
    OPT_QUERIES_PATH,
    PAPER_DIR,
    objective_value,
    read_json,
    source_semantics,
)
from run_aviation_smt_baseline import build_membership_probes, json_dump


DEFAULT_RESULTS_DIR = PAPER_DIR / "results"
OUT_MEMBERSHIP_CSV = "aviation_cthr_cell_smt_membership_results.csv"
OUT_MEMBERSHIP_JSON = "aviation_cthr_cell_smt_membership_summary.json"
OUT_OPT_CSV = "aviation_cthr_cell_smt_optimization_results.csv"
OUT_OPT_JSON = "aviation_cthr_cell_smt_optimization_summary.json"
OUT_REPORT = "aviation_cthr_cell_smt_report.md"


def pct(values: list[bool]) -> float:
    return 100.0 * sum(bool(v) for v in values) / len(values) if values else 0.0


def median(values: list[float]) -> float:
    return float(np.median(values)) if values else 0.0


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def formula_diagnostics(formula: CthrCellSmtFormula) -> dict[str, Any]:
    stats = formula.encoding_stats
    return {
        "cell_count": len(formula.cells),
        "cell_ids": [cell.cell_id for cell in formula.cells],
        "variable_names": list(formula.x.keys()),
        "total_cell_constraint_count": sum(len(cell.constraints) for cell in formula.cells),
        "le_constraint_count": stats.le_constraints,
        "ge_to_le_conversion_count": stats.ge_to_le_conversions,
        "eq_to_two_le_conversion_count": stats.eq_to_two_le_conversions,
        "strict_inequality_count": stats.strict_inequalities,
        "constant_constraint_count": stats.constant_constraints,
        "constraint_parse_failure_count": len(stats.parse_failures),
        "constraint_parse_failures": list(stats.parse_failures),
        "objective_mapping_failure": formula.objective_mapping_failure,
    }


def make_formulas(query: dict[str, Any]) -> dict[str, CthrCellSmtFormula]:
    return {
        mode: build_cthr_cell_smt_formula(query, cell_selection_mode=mode)
        for mode in sorted(CELL_SELECTION_MODES)
    }


def run_membership(
    queries: list[dict[str, Any]],
    feasible_items: list[dict[str, Any]],
    timeout_ms: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    query_by_id = {item["omega_id"]: item for item in queries}
    feasible_by_id = {item["omega_id"]: item for item in feasible_items}
    formulas: dict[tuple[str, str], CthrCellSmtFormula] = {}
    for query in queries:
        for mode, formula in make_formulas(query).items():
            formulas[(query["omega_id"], mode)] = formula

    rows: list[dict[str, Any]] = []
    for idx, probe in enumerate(build_membership_probes(feasible_items)):
        task_id = probe["omega_id"]
        query = query_by_id[task_id]
        feasible_item = feasible_by_id[task_id]
        reference_accept = source_semantics(feasible_item, np.array(probe["x"], dtype=float))
        for mode in sorted(CELL_SELECTION_MODES):
            formula = formulas[(task_id, mode)]
            result = check_membership(formula, query, probe["x"], timeout_ms=timeout_ms)
            rows.append(
                {
                    "task_id": task_id,
                    "title": query["title"],
                    "probe_id": f"{task_id}_{idx:03d}_{probe['probe_type']}",
                    "cell_selection_mode": mode,
                    "probe_type": probe["probe_type"],
                    "x": probe["x"],
                    "smt_status": result.status,
                    "smt_accept": result.accepted,
                    "reference_accept": reference_accept,
                    "formal_satisfied": result.accepted,
                    "semantic_valid": reference_accept,
                    "false_accept": result.accepted and not reference_accept,
                    "false_reject": (not result.accepted) and reference_accept,
                    "active_cell_ids": result.active_cell_ids,
                    "active_rule_ids": result.active_rule_ids,
                    "active_provenance": result.active_provenance,
                    "check_time_ms": result.check_time_ms,
                    "error": result.error,
                    **formula_diagnostics(formula),
                }
            )

    def summarize(subset: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "num_probes": len(subset),
            "formal_csr": pct([row["formal_satisfied"] for row in subset]),
            "semantic_csr": pct([row["reference_accept"] for row in subset]),
            "false_accept": pct([row["false_accept"] for row in subset]),
            "false_reject": pct([row["false_reject"] for row in subset]),
            "median_smt_check_time_ms": median([row["check_time_ms"] for row in subset]),
            "mean_cell_count": mean([row["cell_count"] for row in subset]),
            "mean_total_cell_constraint_count": mean([row["total_cell_constraint_count"] for row in subset]),
            "mean_constraint_parse_failure_count": mean([row["constraint_parse_failure_count"] for row in subset]),
        }

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_files": {
            "optimization_queries_with_compiled_cells": str(OPT_QUERIES_PATH),
            "feasible_region_labels_for_evaluation_only": str(FEASIBLE_LABELS_PATH),
        },
        "definition": "CTHR-cell-SMT uses CTHR compiled cells as solver input. SMT does not perform grounding, rule retrieval, valid-structure selection, or raw consequent mapping.",
        "modes": {
            mode: summarize([row for row in rows if row["cell_selection_mode"] == mode])
            for mode in sorted(CELL_SELECTION_MODES)
        },
    }
    return rows, summary


def external_optimizer_with_smt_oracle(
    formula: CthrCellSmtFormula,
    query: dict[str, Any],
    maxiter: int,
    seed: int,
    timeout_ms: int,
) -> dict[str, Any]:
    cache: dict[tuple[float, ...], Any] = {}
    bounds = [(float(spec["lower"]), float(spec["upper"])) for spec in query.get("decision_variables", {}).values()]

    def oracle(z: np.ndarray):
        key = tuple(round(float(v), 6) for v in z)
        if key not in cache:
            cache[key] = check_membership(formula, query, list(map(float, z)), timeout_ms=timeout_ms)
        return cache[key]

    def penalized(z: np.ndarray) -> float:
        result = oracle(z)
        penalty = 0.0 if result.accepted else 1e6
        try:
            scalar = objective_value(query, np.array(z, dtype=float))
        except Exception:
            scalar = 0.0
            penalty += 1e5
        return float(scalar + penalty)

    start = time.perf_counter()
    try:
        result = differential_evolution(
            penalized,
            bounds,
            seed=seed,
            maxiter=maxiter,
            popsize=4,
            polish=False,
            updating="immediate",
            workers=1,
            tol=1e-6,
        )
        x = [float(v) for v in result.x]
        check = oracle(np.array(x, dtype=float))
        return {
            "status": "sat" if check.accepted else "unknown",
            "optimized_x": x,
            "objective_value": float(objective_value(query, np.array(x, dtype=float))),
            "active_cell_ids": check.active_cell_ids,
            "active_rule_ids": check.active_rule_ids,
            "active_provenance": check.active_provenance,
            "solve_time_ms": (time.perf_counter() - start) * 1000.0,
            "formal_feasible": check.accepted,
            "oracle_calls": len(cache),
            "error": check.error,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "optimized_x": None,
            "objective_value": None,
            "active_cell_ids": [],
            "active_rule_ids": [],
            "active_provenance": [],
            "solve_time_ms": (time.perf_counter() - start) * 1000.0,
            "formal_feasible": False,
            "oracle_calls": len(cache),
            "error": str(exc),
        }


def run_optimization(
    queries: list[dict[str, Any]],
    feasible_items: list[dict[str, Any]],
    timeout_ms: int,
    external_maxiter: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    feasible_by_id = {item["omega_id"]: item for item in feasible_items}
    rows: list[dict[str, Any]] = []
    for idx, query in enumerate(queries):
        task_id = query["omega_id"]
        formulas = make_formulas(query)
        for mode in sorted(CELL_SELECTION_MODES):
            formula = formulas[mode]
            diag = formula_diagnostics(formula)

            z3_result = optimize_with_z3(formula, query, timeout_ms=timeout_ms)
            semantic = (
                source_semantics(feasible_by_id[task_id], np.array(z3_result.optimized_x, dtype=float))
                if z3_result.optimized_x is not None
                else False
            )
            rows.append(
                {
                    "task_id": task_id,
                    "title": query["title"],
                    "cell_selection_mode": mode,
                    "optimization_mode": "z3_optimize",
                    "status": z3_result.status,
                    "optimized_x": z3_result.optimized_x,
                    "objective_value": z3_result.objective_value,
                    "objective_value_if_semantic_valid": z3_result.objective_value if semantic else None,
                    "active_cell_ids": z3_result.active_cell_ids,
                    "active_rule_ids": z3_result.active_rule_ids,
                    "active_provenance": z3_result.active_provenance,
                    "formal_feasible": z3_result.status == "sat",
                    "semantic_valid": semantic,
                    "invalid_optimized_case": z3_result.status == "sat" and not semantic,
                    "solve_time_ms": z3_result.solve_time_ms,
                    "total_time_ms": z3_result.solve_time_ms,
                    "oracle_calls": "N/A",
                    "error": z3_result.error,
                    **diag,
                }
            )

            external = external_optimizer_with_smt_oracle(
                formula,
                query,
                maxiter=external_maxiter,
                seed=seed + idx,
                timeout_ms=timeout_ms,
            )
            semantic = (
                source_semantics(feasible_by_id[task_id], np.array(external["optimized_x"], dtype=float))
                if external["optimized_x"] is not None
                else False
            )
            rows.append(
                {
                    "task_id": task_id,
                    "title": query["title"],
                    "cell_selection_mode": mode,
                    "optimization_mode": "external_optimizer_smt_oracle",
                    "status": external["status"],
                    "optimized_x": external["optimized_x"],
                    "objective_value": external["objective_value"],
                    "objective_value_if_semantic_valid": external["objective_value"] if semantic else None,
                    "active_cell_ids": external["active_cell_ids"],
                    "active_rule_ids": external["active_rule_ids"],
                    "active_provenance": external["active_provenance"],
                    "formal_feasible": external["formal_feasible"],
                    "semantic_valid": semantic,
                    "invalid_optimized_case": external["formal_feasible"] and not semantic,
                    "solve_time_ms": external["solve_time_ms"],
                    "total_time_ms": external["solve_time_ms"],
                    "oracle_calls": external["oracle_calls"],
                    "error": external["error"],
                    **diag,
                }
            )

    def summarize(cell_selection_mode: str, optimization_mode: str) -> dict[str, Any]:
        subset = [
            row
            for row in rows
            if row["cell_selection_mode"] == cell_selection_mode
            and row["optimization_mode"] == optimization_mode
        ]
        valid_objectives = [
            row["objective_value_if_semantic_valid"]
            for row in subset
            if row["objective_value_if_semantic_valid"] is not None
        ]
        objective_failures = sum(
            1 for row in subset if row.get("error") and "objective_mapping_failure" in str(row.get("error"))
        )
        return {
            "num_cases": len(subset),
            "formal_feasible": pct([row["formal_feasible"] for row in subset]),
            "semantic_valid": pct([row["semantic_valid"] for row in subset]),
            "invalid_optimized_cases": sum(bool(row["invalid_optimized_case"]) for row in subset),
            "objective_mapping_failures": objective_failures,
            "median_solve_time_ms": median([row["solve_time_ms"] for row in subset]),
            "median_total_time_ms": median([row["total_time_ms"] for row in subset]),
            "mean_objective_value_semantic_valid_only": float(np.mean(valid_objectives)) if valid_objectives else None,
            "mean_cell_count": mean([row["cell_count"] for row in subset]),
            "mean_total_cell_constraint_count": mean([row["total_cell_constraint_count"] for row in subset]),
            "mean_constraint_parse_failure_count": mean([row["constraint_parse_failure_count"] for row in subset]),
        }

    modes = {
        f"{cell_mode}__{opt_mode}": summarize(cell_mode, opt_mode)
        for cell_mode in sorted(CELL_SELECTION_MODES)
        for opt_mode in ["z3_optimize", "external_optimizer_smt_oracle"]
    }
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_files": {
            "optimization_queries_with_compiled_cells": str(OPT_QUERIES_PATH),
            "feasible_region_labels_for_evaluation_only": str(FEASIBLE_LABELS_PATH),
        },
        "definition": "CTHR-cell-SMT uses CTHR compiled cells as solver input. SMT does not perform grounding, rule retrieval, valid-structure selection, or raw consequent mapping.",
        "external_optimizer_maxiter": external_maxiter,
        "modes": modes,
        "per_case": rows,
    }
    return rows, summary


def stringify_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for key in [
        "x",
        "optimized_x",
        "active_cell_ids",
        "active_rule_ids",
        "active_provenance",
        "cell_ids",
        "variable_names",
        "constraint_parse_failures",
    ]:
        if key in out:
            out[key] = json_dump(out[key])
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(stringify_row(row))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def report_table(summary: dict[str, Any]) -> list[str]:
    lines = [
        "| Mode | Items | Formal (%) | Semantic (%) | False accept / invalid | Objective map fail | Median ms | Cells | Constraints | Parse fail |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, item in summary["modes"].items():
        is_membership = "num_probes" in item
        lines.append(
            "| {mode} | {num} | {formal:.1f} | {semantic:.1f} | {bad} | {objfail} | {ms:.2f} | {cells:.1f} | {constraints:.1f} | {parse:.1f} |".format(
                mode=mode,
                num=item["num_probes"] if is_membership else item["num_cases"],
                formal=item["formal_csr"] if is_membership else item["formal_feasible"],
                semantic=item["semantic_csr"] if is_membership else item["semantic_valid"],
                bad=item["false_accept"] if is_membership else item["invalid_optimized_cases"],
                objfail="N/A" if is_membership else item["objective_mapping_failures"],
                ms=item["median_smt_check_time_ms"] if is_membership else item["median_total_time_ms"],
                cells=item["mean_cell_count"],
                constraints=item["mean_total_cell_constraint_count"],
                parse=item["mean_constraint_parse_failure_count"],
            )
        )
    return lines


def failure_source_diagnostics(
    membership_summary: dict[str, Any],
    optimization_summary: dict[str, Any],
) -> list[str]:
    at_z3 = optimization_summary["modes"].get("at_least_one__z3_optimize", {})
    ex_z3 = optimization_summary["modes"].get("exactly_one__z3_optimize", {})
    at_mem = membership_summary["modes"].get("at_least_one", {})
    ex_mem = membership_summary["modes"].get("exactly_one", {})
    return [
        "## Failure-Source Diagnostics",
        "",
        "- Cell export loss: membership false-accept is {:.1f}% and false-reject is {:.1f}% under at-least-one semantics. Near-zero values indicate that exported cells match the reference feasible-region behavior.".format(
            at_mem.get("false_accept", 0.0),
            at_mem.get("false_reject", 0.0),
        ),
        "- Variable mapping mismatch: the backend consumes executable compiled-cell expressions over task decision variables, so raw KG variable mapping is bypassed. Constraint parse failures average {:.1f} per case.".format(
            at_z3.get("mean_constraint_parse_failure_count", 0.0),
        ),
        "- Objective mapping mismatch: Z3 Optimize reports {} objective-mapping failures under at-least-one semantics.".format(
            at_z3.get("objective_mapping_failures", 0),
        ),
        "- At-least-one vs exactly-one: Z3 semantic validity is {:.1f}% vs {:.1f}%. A difference would indicate overlap-sensitive cell-selector semantics.".format(
            at_z3.get("semantic_valid", 0.0),
            ex_z3.get("semantic_valid", 0.0),
        ),
        "- External optimizer oracle integration: if Z3 Optimize is valid but external-oracle search is worse, the issue is search/oracle integration rather than the compiled-cell SMT encoding.",
        "- Membership sensitivity: at-least-one and exactly-one membership semantic rates are {:.1f}% and {:.1f}%.".format(
            at_mem.get("semantic_csr", 0.0),
            ex_mem.get("semantic_csr", 0.0),
        ),
    ]


def write_report(
    path: Path,
    membership_summary: dict[str, Any],
    optimization_summary: dict[str, Any],
) -> None:
    lines = [
        "# Aviation CTHR-Cell-SMT Backend",
        "",
        "This is a CTHR-guided SMT backend, not an SMT-only baseline.",
        "CTHR has already completed grounding, valid rule structure construction, and cell compilation. SMT/Z3 only encodes the exported compiled cells, checks membership, optimizes the objective, and returns active cells with source-linked rule/provenance records.",
        "",
        "The implementation uses the persisted compiled-cell interface in the aviation optimization-query artifact (`solver_constraints` and `solver_constraint_cells`). It does not use CTHR optimizer outputs, hidden reference valid cells, or the semantic validator as SMT input. The reference validator is used only for evaluation.",
        "",
        "## Membership / Feasible-Region Validation",
        "",
        *report_table(membership_summary),
        "",
        "## Optimization",
        "",
        *report_table(optimization_summary),
        "",
        *failure_source_diagnostics(membership_summary, optimization_summary),
        "",
        "## Interpretation Guide",
        "",
        "- If results match CTHR full, SMT can serve as a reliable backend for CTHR compiled cells.",
        "- If invalid cases remain, inspect cell export loss, objective encoding, selector semantics, and external optimizer oracle integration.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CTHR compiled-cell SMT backend on aviation benchmark.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument("--external-maxiter", type=int, default=80)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    queries = read_json(OPT_QUERIES_PATH)["items"]
    feasible_items = read_json(FEASIBLE_LABELS_PATH)["items"]

    membership_rows, membership_summary = run_membership(
        queries,
        feasible_items,
        timeout_ms=args.timeout_ms,
    )
    optimization_rows, optimization_summary = run_optimization(
        queries,
        feasible_items,
        timeout_ms=args.timeout_ms,
        external_maxiter=args.external_maxiter,
        seed=args.seed,
    )

    membership_fields = [
        "task_id",
        "title",
        "probe_id",
        "cell_selection_mode",
        "probe_type",
        "x",
        "smt_status",
        "smt_accept",
        "reference_accept",
        "formal_satisfied",
        "semantic_valid",
        "false_accept",
        "false_reject",
        "active_cell_ids",
        "active_rule_ids",
        "active_provenance",
        "check_time_ms",
        "cell_count",
        "cell_ids",
        "variable_names",
        "total_cell_constraint_count",
        "le_constraint_count",
        "ge_to_le_conversion_count",
        "eq_to_two_le_conversion_count",
        "strict_inequality_count",
        "constant_constraint_count",
        "constraint_parse_failure_count",
        "constraint_parse_failures",
        "objective_mapping_failure",
        "error",
    ]
    optimization_fields = [
        "task_id",
        "title",
        "cell_selection_mode",
        "optimization_mode",
        "status",
        "optimized_x",
        "objective_value",
        "objective_value_if_semantic_valid",
        "active_cell_ids",
        "active_rule_ids",
        "active_provenance",
        "formal_feasible",
        "semantic_valid",
        "invalid_optimized_case",
        "solve_time_ms",
        "total_time_ms",
        "oracle_calls",
        "cell_count",
        "cell_ids",
        "variable_names",
        "total_cell_constraint_count",
        "le_constraint_count",
        "ge_to_le_conversion_count",
        "eq_to_two_le_conversion_count",
        "strict_inequality_count",
        "constant_constraint_count",
        "constraint_parse_failure_count",
        "constraint_parse_failures",
        "objective_mapping_failure",
        "error",
    ]

    write_csv(args.results_dir / OUT_MEMBERSHIP_CSV, membership_rows, membership_fields)
    write_json(args.results_dir / OUT_MEMBERSHIP_JSON, membership_summary)
    write_csv(args.results_dir / OUT_OPT_CSV, optimization_rows, optimization_fields)
    write_json(args.results_dir / OUT_OPT_JSON, optimization_summary)
    write_report(args.results_dir / OUT_REPORT, membership_summary, optimization_summary)

    print(f"Wrote {args.results_dir / OUT_MEMBERSHIP_CSV}")
    print(f"Wrote {args.results_dir / OUT_MEMBERSHIP_JSON}")
    print(f"Wrote {args.results_dir / OUT_OPT_CSV}")
    print(f"Wrote {args.results_dir / OUT_OPT_JSON}")
    print(f"Wrote {args.results_dir / OUT_REPORT}")


if __name__ == "__main__":
    main()
