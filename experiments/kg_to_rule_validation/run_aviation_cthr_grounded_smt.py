from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution

from baselines.cthr_grounded_smt import (
    SELECTION_MODES,
    CthrGroundedSmtFormula,
    build_cthr_grounded_smt_formula,
    check_membership,
    cthr_safe_ground_candidate_rule_ids,
    optimize_with_z3,
)
from run_aviation_dataset_experiments import (
    FEASIBLE_LABELS_PATH,
    OPT_QUERIES_PATH,
    PAPER_DIR,
    RULE_LIBRARY_PATH,
    objective_value,
    read_json,
    source_semantics,
)
from run_aviation_smt_baseline import build_membership_probes, json_dump


DEFAULT_RESULTS_DIR = PAPER_DIR / "results"
OUT_MEMBERSHIP_CSV = "aviation_cthr_grounded_smt_membership_results.csv"
OUT_MEMBERSHIP_JSON = "aviation_cthr_grounded_smt_membership_summary.json"
OUT_OPT_CSV = "aviation_cthr_grounded_smt_optimization_results.csv"
OUT_OPT_JSON = "aviation_cthr_grounded_smt_optimization_summary.json"
OUT_REPORT = "aviation_cthr_grounded_smt_report.md"


def pct(values: list[bool]) -> float:
    return 100.0 * sum(bool(v) for v in values) / len(values) if values else 0.0


def median(values: list[float]) -> float:
    return float(np.median(values)) if values else 0.0


def mapping_failure_summary(formula: CthrGroundedSmtFormula) -> dict[str, Any]:
    by_reason: dict[str, int] = {}
    for failure in formula.mapping_failures:
        by_reason[failure.reason] = by_reason.get(failure.reason, 0) + 1
    return {
        "mapping_failure_count": len(formula.mapping_failures),
        "mapping_failure_reasons": by_reason,
        "mapping_failure_rules": sorted({failure.rule_id for failure in formula.mapping_failures}),
    }


def formula_diagnostics(formula: CthrGroundedSmtFormula) -> dict[str, Any]:
    failures = mapping_failure_summary(formula)
    return {
        "candidate_rule_ids": formula.candidate_rule_ids,
        "candidate_rule_count": len(formula.candidate_rule_ids),
        "applicable_rule_ids": formula.applicable_rule_ids,
        "applicable_rule_count": len(formula.applicable_rule_ids),
        "encoded_rule_library_constraint_count": formula.encoded_rule_library_constraint_count,
        "missing_required_numeric_mappings": failures["mapping_failure_count"],
        "mapping_failure_reasons": failures["mapping_failure_reasons"],
        "mapping_failure_rules": failures["mapping_failure_rules"],
        "dependency_count": len(formula.dependency_pairs),
        "exclusion_count": len(formula.exclusion_pairs),
        "override_count": len(formula.override_pairs),
        "precedence_count": len(formula.precedence_pairs),
        "conflict_class_count": len(formula.conflict_classes),
    }


def make_formulas(
    rule_library: dict[str, Any],
    query: dict[str, Any],
) -> dict[str, CthrGroundedSmtFormula]:
    candidate_ids = cthr_safe_ground_candidate_rule_ids(rule_library, query)
    return {
        mode: build_cthr_grounded_smt_formula(
            rule_library,
            query,
            selection_mode=mode,
            candidate_rule_ids=candidate_ids,
        )
        for mode in sorted(SELECTION_MODES)
    }


def run_membership(
    rule_library: dict[str, Any],
    queries: list[dict[str, Any]],
    feasible_items: list[dict[str, Any]],
    timeout_ms: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    query_by_id = {item["omega_id"]: item for item in queries}
    feasible_by_id = {item["omega_id"]: item for item in feasible_items}
    formulas: dict[tuple[str, str], CthrGroundedSmtFormula] = {}
    for query in queries:
        for mode, formula in make_formulas(rule_library, query).items():
            formulas[(query["omega_id"], mode)] = formula

    rows: list[dict[str, Any]] = []
    for probe in build_membership_probes(feasible_items):
        task_id = probe["omega_id"]
        query = query_by_id[task_id]
        feasible_item = feasible_by_id[task_id]
        semantic = source_semantics(feasible_item, np.array(probe["x"], dtype=float))
        for mode in sorted(SELECTION_MODES):
            formula = formulas[(task_id, mode)]
            result = check_membership(formula, query, probe["x"], timeout_ms=timeout_ms)
            diag = formula_diagnostics(formula)
            rows.append(
                {
                    "task_id": task_id,
                    "title": query["title"],
                    "selection_mode": mode,
                    "probe_type": probe["probe_type"],
                    "x": probe["x"],
                    "smt_status": result.status,
                    "smt_accepts": result.accepted,
                    "semantic_valid": semantic,
                    "false_accept": result.accepted and not semantic,
                    "false_reject": (not result.accepted) and semantic,
                    "selected_rule_ids": result.selected_rule_ids,
                    "defeated_rule_ids": result.defeated_rule_ids,
                    "check_time_ms": result.check_time_ms,
                    "error": result.error,
                    **diag,
                }
            )

    def summarize(subset: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "num_probes": len(subset),
            "formal_csr": pct([row["smt_accepts"] for row in subset]),
            "semantic_csr": pct([row["semantic_valid"] for row in subset]),
            "false_accept": pct([row["false_accept"] for row in subset]),
            "false_reject": pct([row["false_reject"] for row in subset]),
            "median_smt_check_time_ms": median([row["check_time_ms"] for row in subset]),
            "mean_candidate_rule_count": float(np.mean([row["candidate_rule_count"] for row in subset])) if subset else 0.0,
            "mean_applicable_rule_count": float(np.mean([row["applicable_rule_count"] for row in subset])) if subset else 0.0,
            "mean_encoded_rule_library_constraint_count": float(
                np.mean([row["encoded_rule_library_constraint_count"] for row in subset])
            )
            if subset
            else 0.0,
            "mean_missing_required_numeric_mappings": float(
                np.mean([row["missing_required_numeric_mappings"] for row in subset])
            )
            if subset
            else 0.0,
        }

    by_mode = {
        mode: summarize([row for row in rows if row["selection_mode"] == mode])
        for mode in sorted(SELECTION_MODES)
    }
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_files": {
            "rule_library": str(RULE_LIBRARY_PATH),
            "optimization_queries": str(OPT_QUERIES_PATH),
            "feasible_region_labels_for_evaluation_only": str(FEASIBLE_LABELS_PATH),
        },
        "candidate_grounding": "CTHR-safe visible grounding over task/query fields, rule metadata, guards, units, and relation closure",
        "modes": by_mode,
    }
    return rows, summary


def external_optimizer_with_smt_oracle(
    formula: CthrGroundedSmtFormula,
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
            "selected_rule_ids": check.selected_rule_ids,
            "defeated_rule_ids": check.defeated_rule_ids,
            "solve_time_ms": (time.perf_counter() - start) * 1000.0,
            "formal_feasible_smt": check.accepted,
            "oracle_calls": len(cache),
            "error": check.error,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "optimized_x": None,
            "objective_value": None,
            "selected_rule_ids": [],
            "defeated_rule_ids": [],
            "solve_time_ms": (time.perf_counter() - start) * 1000.0,
            "formal_feasible_smt": False,
            "oracle_calls": len(cache),
            "error": str(exc),
        }


def run_optimization(
    rule_library: dict[str, Any],
    queries: list[dict[str, Any]],
    feasible_items: list[dict[str, Any]],
    timeout_ms: int,
    external_maxiter: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    feasible_by_id = {item["omega_id"]: item for item in feasible_items}
    rows: list[dict[str, Any]] = []
    for idx, query in enumerate(queries):
        formulas = make_formulas(rule_library, query)
        for mode in sorted(SELECTION_MODES):
            formula = formulas[mode]
            diag = formula_diagnostics(formula)

            z3_result = optimize_with_z3(formula, query, timeout_ms=timeout_ms)
            semantic = (
                source_semantics(feasible_by_id[query["omega_id"]], np.array(z3_result.optimized_x, dtype=float))
                if z3_result.optimized_x is not None
                else False
            )
            rows.append(
                {
                    "task_id": query["omega_id"],
                    "title": query["title"],
                    "selection_mode": mode,
                    "optimization_mode": "z3_optimize",
                    "status": z3_result.status,
                    "optimized_x": z3_result.optimized_x,
                    "objective_value": z3_result.objective_value,
                    "objective_value_if_semantic_valid": z3_result.objective_value if semantic else None,
                    "selected_rule_ids": z3_result.selected_rule_ids,
                    "defeated_rule_ids": z3_result.defeated_rule_ids,
                    "formal_feasible_smt": z3_result.status == "sat",
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
                source_semantics(feasible_by_id[query["omega_id"]], np.array(external["optimized_x"], dtype=float))
                if external["optimized_x"] is not None
                else False
            )
            rows.append(
                {
                    "task_id": query["omega_id"],
                    "title": query["title"],
                    "selection_mode": mode,
                    "optimization_mode": "external_optimizer_smt_oracle",
                    "status": external["status"],
                    "optimized_x": external["optimized_x"],
                    "objective_value": external["objective_value"],
                    "objective_value_if_semantic_valid": external["objective_value"] if semantic else None,
                    "selected_rule_ids": external["selected_rule_ids"],
                    "defeated_rule_ids": external["defeated_rule_ids"],
                    "formal_feasible_smt": external["formal_feasible_smt"],
                    "semantic_valid": semantic,
                    "invalid_optimized_case": external["formal_feasible_smt"] and not semantic,
                    "solve_time_ms": external["solve_time_ms"],
                    "total_time_ms": external["solve_time_ms"],
                    "oracle_calls": external["oracle_calls"],
                    "error": external["error"],
                    **diag,
                }
            )

    def summarize(selection_mode: str, optimization_mode: str) -> dict[str, Any]:
        subset = [
            row
            for row in rows
            if row["selection_mode"] == selection_mode and row["optimization_mode"] == optimization_mode
        ]
        valid_objectives = [
            row["objective_value_if_semantic_valid"]
            for row in subset
            if row["objective_value_if_semantic_valid"] is not None
        ]
        return {
            "num_cases": len(subset),
            "formal_feasible_smt": pct([row["formal_feasible_smt"] for row in subset]),
            "semantic_valid": pct([row["semantic_valid"] for row in subset]),
            "invalid_optimized_cases": sum(bool(row["invalid_optimized_case"]) for row in subset),
            "median_solve_time_ms": median([row["solve_time_ms"] for row in subset]),
            "median_total_time_ms": median([row["total_time_ms"] for row in subset]),
            "mean_objective_value_semantic_valid_only": float(np.mean(valid_objectives)) if valid_objectives else None,
            "mean_candidate_rule_count": float(np.mean([row["candidate_rule_count"] for row in subset])) if subset else 0.0,
            "mean_encoded_rule_library_constraint_count": float(
                np.mean([row["encoded_rule_library_constraint_count"] for row in subset])
            )
            if subset
            else 0.0,
            "mean_missing_required_numeric_mappings": float(
                np.mean([row["missing_required_numeric_mappings"] for row in subset])
            )
            if subset
            else 0.0,
        }

    modes = {
        f"{selection_mode}__{optimization_mode}": summarize(selection_mode, optimization_mode)
        for selection_mode in sorted(SELECTION_MODES)
        for optimization_mode in ["z3_optimize", "external_optimizer_smt_oracle"]
    }
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_files": {
            "rule_library": str(RULE_LIBRARY_PATH),
            "optimization_queries": str(OPT_QUERIES_PATH),
            "feasible_region_labels_for_evaluation_only": str(FEASIBLE_LABELS_PATH),
        },
        "candidate_grounding": "CTHR-safe visible grounding over task/query fields, rule metadata, guards, units, and relation closure",
        "external_optimizer_maxiter": external_maxiter,
        "modes": modes,
        "per_case": rows,
    }
    return rows, summary


def stringify_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for key in [
        "candidate_rule_ids",
        "applicable_rule_ids",
        "x",
        "selected_rule_ids",
        "defeated_rule_ids",
        "optimized_x",
        "mapping_failure_reasons",
        "mapping_failure_rules",
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
        "| Mode | Items | Formal CSR / feasible (%) | Semantic CSR / valid (%) | False accept / invalid | Median ms | Missing mappings |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, item in summary["modes"].items():
        is_membership = "num_probes" in item
        lines.append(
            "| {mode} | {num} | {formal:.1f} | {semantic:.1f} | {bad} | {ms:.2f} | {maps:.1f} |".format(
                mode=mode,
                num=item["num_probes"] if is_membership else item["num_cases"],
                formal=item["formal_csr"] if is_membership else item["formal_feasible_smt"],
                semantic=item["semantic_csr"] if is_membership else item["semantic_valid"],
                bad=item["false_accept"] if is_membership else item["invalid_optimized_cases"],
                ms=item["median_smt_check_time_ms"] if is_membership else item["median_total_time_ms"],
                maps=item["mean_missing_required_numeric_mappings"],
            )
        )
    return lines


def failure_source_diagnostics(
    membership_summary: dict[str, Any],
    optimization_summary: dict[str, Any],
    feasible_items: list[dict[str, Any]],
) -> list[str]:
    max_mem = membership_summary["modes"].get("maximize_selected", {})
    req_mem = membership_summary["modes"].get("required_applicable", {})
    max_z3 = optimization_summary["modes"].get("maximize_selected__z3_optimize", {})
    max_ext = optimization_summary["modes"].get("maximize_selected__external_optimizer_smt_oracle", {})
    req_ext = optimization_summary["modes"].get("required_applicable__external_optimizer_smt_oracle", {})
    cell_tasks = [item["omega_id"] for item in feasible_items if item.get("valid_constraint_cells")]
    req_unsat = sum(
        1
        for row in optimization_summary.get("per_case", [])
        if row["selection_mode"] == "required_applicable"
        and row["optimization_mode"] == "z3_optimize"
        and row["status"] == "unsat"
    )
    return [
        "## Failure-Source Diagnostics",
        "",
        "- Candidate grounding: safe visible grounding produces a mean of {:.1f} candidates and {:.1f} guard-applicable rules per membership probe. This is intentionally pre-resolution and noisy; SMT must still select valid rules.".format(
            max_mem.get("mean_candidate_rule_count", 0.0),
            max_mem.get("mean_applicable_rule_count", 0.0),
        ),
        "- Rule activation completeness: required-applicable mode has {:.1f}% membership false rejects and {}/19 Z3 optimization cases become UNSAT, indicating that forcing all guard-applicable rules is too strict under noisy candidates and incomplete defeat/exclusion metadata.".format(
            req_mem.get("false_reject", 0.0),
            req_unsat,
        ),
        "- Numerical mapping: only {:.1f} rule-library constraints are encoded per optimization case on average, while {:.1f} rule consequents fail mapping. This is the dominant bottleneck for executable optimization.".format(
            max_z3.get("mean_encoded_rule_library_constraint_count", 0.0),
            max_z3.get("mean_missing_required_numeric_mappings", 0.0),
        ),
        "- Missing cell decomposition: {} aviation tasks contain reference valid-cell structure for piecewise alternatives ({}) used only for evaluation/diagnosis. The monolithic formula does not build these cells explicitly.".format(
            len(cell_tasks),
            ", ".join(cell_tasks) if cell_tasks else "none",
        ),
        "- SMT optimization mode: maximize-selected Z3 is formally feasible in {:.1f}% of cases but semantically valid in {:.1f}%; external optimizer plus SMT oracle reaches {:.1f}% / {:.1f}% semantic validity for maximize-selected / required-applicable.".format(
            max_z3.get("formal_feasible_smt", 0.0),
            max_z3.get("semantic_valid", 0.0),
            max_ext.get("semantic_valid", 0.0),
            req_ext.get("semantic_valid", 0.0),
        ),
    ]


def write_report(
    path: Path,
    membership_summary: dict[str, Any],
    optimization_summary: dict[str, Any],
    feasible_items: list[dict[str, Any]],
) -> None:
    lines = [
        "# Aviation CTHR-Grounded SMT Baseline",
        "",
        "This is CTHR-grounded SMT, not SMT-only and not CTHR full.",
        "CTHR provides only pre-resolution grounded candidate rules through a safe visible-grounding function. SMT is responsible for rule activation, dependency, exclusion, override, precedence, numerical consequent encoding, and solving.",
        "",
        "The grounding function intentionally avoids CTHR final valid structures, compiled cells, hidden expected labels, reference valid cells, semantic validator output, and previously diagnosed leakage-prone shared-grounding artifacts.",
        "",
        "## Membership / Feasible-Region Validation",
        "",
        *report_table(membership_summary),
        "",
        "## Optimization",
        "",
        *report_table(optimization_summary),
        "",
        *failure_source_diagnostics(membership_summary, optimization_summary, feasible_items),
        "",
        "## Interpretation Guide",
        "",
        "- If candidate counts are high and precision is low, the issue is candidate grounding noise rather than SMT expressiveness.",
        "- If required-applicable mode becomes UNSAT or rejects many valid probes, the issue is rule activation completeness or missing defeat/exclusion metadata.",
        "- If missing numeric mappings are high, the issue is conversion from KG rule consequents to executable task variables.",
        "- If cases with piecewise alternatives fail, the issue may be missing valid-cell decomposition rather than ordinary Boolean rule selection.",
        "- If Z3 Optimize is formally feasible but semantically invalid, the objective has reached a region accepted by the encoded formula but invalid under source-rule semantics.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CTHR-grounded SMT baseline on the aviation benchmark.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument("--external-maxiter", type=int, default=80)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    rule_library = read_json(RULE_LIBRARY_PATH)
    queries = read_json(OPT_QUERIES_PATH)["items"]
    feasible_items = read_json(FEASIBLE_LABELS_PATH)["items"]

    membership_rows, membership_summary = run_membership(
        rule_library,
        queries,
        feasible_items,
        timeout_ms=args.timeout_ms,
    )
    optimization_rows, optimization_summary = run_optimization(
        rule_library,
        queries,
        feasible_items,
        timeout_ms=args.timeout_ms,
        external_maxiter=args.external_maxiter,
        seed=args.seed,
    )

    membership_fields = [
        "task_id",
        "title",
        "selection_mode",
        "probe_type",
        "x",
        "smt_status",
        "smt_accepts",
        "semantic_valid",
        "false_accept",
        "false_reject",
        "selected_rule_ids",
        "defeated_rule_ids",
        "check_time_ms",
        "candidate_rule_ids",
        "candidate_rule_count",
        "applicable_rule_ids",
        "applicable_rule_count",
        "encoded_rule_library_constraint_count",
        "missing_required_numeric_mappings",
        "mapping_failure_reasons",
        "mapping_failure_rules",
        "dependency_count",
        "exclusion_count",
        "override_count",
        "precedence_count",
        "conflict_class_count",
        "error",
    ]
    optimization_fields = [
        "task_id",
        "title",
        "selection_mode",
        "optimization_mode",
        "status",
        "optimized_x",
        "objective_value",
        "objective_value_if_semantic_valid",
        "selected_rule_ids",
        "defeated_rule_ids",
        "formal_feasible_smt",
        "semantic_valid",
        "invalid_optimized_case",
        "solve_time_ms",
        "total_time_ms",
        "oracle_calls",
        "candidate_rule_ids",
        "candidate_rule_count",
        "applicable_rule_ids",
        "applicable_rule_count",
        "encoded_rule_library_constraint_count",
        "missing_required_numeric_mappings",
        "mapping_failure_reasons",
        "mapping_failure_rules",
        "dependency_count",
        "exclusion_count",
        "override_count",
        "precedence_count",
        "conflict_class_count",
        "error",
    ]

    write_csv(args.results_dir / OUT_MEMBERSHIP_CSV, membership_rows, membership_fields)
    write_json(args.results_dir / OUT_MEMBERSHIP_JSON, membership_summary)
    write_csv(args.results_dir / OUT_OPT_CSV, optimization_rows, optimization_fields)
    write_json(args.results_dir / OUT_OPT_JSON, optimization_summary)
    write_report(args.results_dir / OUT_REPORT, membership_summary, optimization_summary, feasible_items)

    print(f"Wrote {args.results_dir / OUT_MEMBERSHIP_CSV}")
    print(f"Wrote {args.results_dir / OUT_MEMBERSHIP_JSON}")
    print(f"Wrote {args.results_dir / OUT_OPT_CSV}")
    print(f"Wrote {args.results_dir / OUT_OPT_JSON}")
    print(f"Wrote {args.results_dir / OUT_REPORT}")


if __name__ == "__main__":
    main()
