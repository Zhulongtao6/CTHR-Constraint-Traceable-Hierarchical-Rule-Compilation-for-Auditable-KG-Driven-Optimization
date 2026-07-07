from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution

from baselines.asp_rule_structure import retrieve_candidate_rules
from baselines.smt_monolithic import (
    SmtFormula,
    build_smt_formula,
    check_membership,
    optimize_with_z3,
)
from run_aviation_dataset_experiments import (
    FEASIBLE_LABELS_PATH,
    OPT_QUERIES_PATH,
    OUT_JSON as AVIATION_FULL_JSON,
    PAPER_DIR,
    RULE_LIBRARY_PATH,
    flat_semantics,
    objective_value,
    read_json,
    source_semantics,
    targeted_flat_probes,
)


DEFAULT_RESULTS_DIR = PAPER_DIR / "results"
OUT_MEMBERSHIP_CSV = "aviation_smt_membership_results.csv"
OUT_MEMBERSHIP_JSON = "aviation_smt_membership_summary.json"
OUT_OPT_CSV = "aviation_smt_optimization_results.csv"
OUT_OPT_JSON = "aviation_smt_optimization_summary.json"
OUT_REPORT = "aviation_smt_baseline_report.md"


def pct(values: list[bool]) -> float:
    return 100.0 * sum(bool(v) for v in values) / len(values) if values else 0.0


def median(values: list[float]) -> float:
    return float(np.median(values)) if values else 0.0


def json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def load_aviation_full_results() -> dict[str, Any] | None:
    if AVIATION_FULL_JSON.exists():
        return read_json(AVIATION_FULL_JSON)
    return None


def build_membership_probes(feasible_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    probes: list[dict[str, Any]] = []
    full = load_aviation_full_results()
    if full:
        for row in full.get("section_6_4", {}).get("per_case", []):
            probes.append(
                {
                    "omega_id": row["omega_id"],
                    "probe_type": f"{row['method']}_optimized",
                    "x": row["solution"]["x"],
                }
            )
    for item in feasible_items:
        for idx, x in enumerate(targeted_flat_probes(item)):
            probes.append(
                {
                    "omega_id": item["omega_id"],
                    "probe_type": f"targeted_flat_probe_{idx}",
                    "x": x.tolist(),
                }
            )
    return probes


def row_selected_string(selected_rule_ids: list[str]) -> str:
    return json_dump(selected_rule_ids)


def run_membership(
    rule_library: dict[str, Any],
    queries: list[dict[str, Any]],
    feasible_items: list[dict[str, Any]],
    timeout_ms: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    query_by_id = {item["omega_id"]: item for item in queries}
    feasible_by_id = {item["omega_id"]: item for item in feasible_items}
    formulas: dict[str, SmtFormula] = {}
    for query in queries:
        candidate_ids = retrieve_candidate_rules(rule_library, query)
        formulas[query["omega_id"]] = build_smt_formula(rule_library, query, candidate_ids)

    rows: list[dict[str, Any]] = []
    probes = build_membership_probes(feasible_items)
    for probe in probes:
        task_id = probe["omega_id"]
        query = query_by_id[task_id]
        feasible_item = feasible_by_id[task_id]
        formula = formulas[task_id]
        result = check_membership(formula, query, probe["x"], timeout_ms=timeout_ms)
        semantic = source_semantics(feasible_item, np.array(probe["x"], dtype=float))
        rows.append(
            {
                "task_id": task_id,
                "title": query["title"],
                "probe_type": probe["probe_type"],
                "x": probe["x"],
                "smt_status": result.status,
                "smt_accepts": result.accepted,
                "semantic_valid": semantic,
                "false_accept": result.accepted and not semantic,
                "false_reject": (not result.accepted) and semantic,
                "selected_rule_ids": result.selected_rule_ids,
                "check_time_ms": result.check_time_ms,
                "candidate_rule_count": len(formula.candidate_rule_ids),
                "encoded_constraint_count": result.encoded_constraint_count,
                "skipped_constraint_count": result.skipped_constraint_count,
                "encoded_rule_library_constraint_count": formula.encoded_rule_library_constraint_count,
                "encoded_visible_constraint_count": formula.encoded_visible_constraint_count,
                "error": result.error,
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
            "mean_encoded_constraint_count": float(np.mean([row["encoded_constraint_count"] for row in subset])) if subset else 0.0,
            "mean_skipped_constraint_count": float(np.mean([row["skipped_constraint_count"] for row in subset])) if subset else 0.0,
        }

    per_task = {
        task_id: summarize([row for row in rows if row["task_id"] == task_id])
        for task_id in sorted({row["task_id"] for row in rows})
    }
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_files": {
            "rule_library": str(RULE_LIBRARY_PATH),
            "optimization_queries": str(OPT_QUERIES_PATH),
            "feasible_region_labels_for_evaluation_only": str(FEASIBLE_LABELS_PATH),
        },
        "global": summarize(rows),
        "per_task": per_task,
    }
    return rows, summary


def external_optimizer_with_smt_oracle(
    formula: SmtFormula,
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
        candidate_ids = retrieve_candidate_rules(rule_library, query)
        formula = build_smt_formula(rule_library, query, candidate_ids)

        z3_result = optimize_with_z3(formula, query, timeout_ms=timeout_ms)
        for result in [z3_result]:
            semantic = (
                source_semantics(feasible_by_id[query["omega_id"]], np.array(result.optimized_x, dtype=float))
                if result.optimized_x is not None
                else False
            )
            rows.append(
                {
                    "task_id": query["omega_id"],
                    "title": query["title"],
                    "mode": result.mode,
                    "status": result.status,
                    "optimized_x": result.optimized_x,
                    "objective_value": result.objective_value,
                    "selected_rule_ids": result.selected_rule_ids,
                    "formal_feasible_smt": result.status == "sat",
                    "semantic_valid": semantic,
                    "invalid_optimized_case": result.status == "sat" and not semantic,
                    "solve_time_ms": result.solve_time_ms,
                    "candidate_rule_count": len(formula.candidate_rule_ids),
                    "encoded_constraint_count": result.encoded_constraint_count,
                    "skipped_constraint_count": result.skipped_constraint_count,
                    "encoded_rule_library_constraint_count": formula.encoded_rule_library_constraint_count,
                    "encoded_visible_constraint_count": formula.encoded_visible_constraint_count,
                    "oracle_calls": "N/A",
                    "error": result.error,
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
                "mode": "external_optimizer_smt_oracle",
                "status": external["status"],
                "optimized_x": external["optimized_x"],
                "objective_value": external["objective_value"],
                "selected_rule_ids": external["selected_rule_ids"],
                "formal_feasible_smt": external["formal_feasible_smt"],
                "semantic_valid": semantic,
                "invalid_optimized_case": external["formal_feasible_smt"] and not semantic,
                "solve_time_ms": external["solve_time_ms"],
                "candidate_rule_count": len(formula.candidate_rule_ids),
                "encoded_constraint_count": formula.encoded_constraint_count,
                "skipped_constraint_count": formula.skipped_constraint_count,
                "encoded_rule_library_constraint_count": formula.encoded_rule_library_constraint_count,
                "encoded_visible_constraint_count": formula.encoded_visible_constraint_count,
                "oracle_calls": external["oracle_calls"],
                "error": external["error"],
            }
        )

    def summarize(mode: str) -> dict[str, Any]:
        subset = [row for row in rows if row["mode"] == mode]
        return {
            "num_cases": len(subset),
            "formal_feasible_smt": pct([row["formal_feasible_smt"] for row in subset]),
            "semantic_valid": pct([row["semantic_valid"] for row in subset]),
            "invalid_optimized_cases": sum(bool(row["invalid_optimized_case"]) for row in subset),
            "median_solve_time_ms": median([row["solve_time_ms"] for row in subset]),
            "mean_objective_value": float(np.mean([row["objective_value"] for row in subset if row["objective_value"] is not None]))
            if any(row["objective_value"] is not None for row in subset)
            else None,
            "mean_candidate_rule_count": float(np.mean([row["candidate_rule_count"] for row in subset])) if subset else 0.0,
            "mean_encoded_constraint_count": float(np.mean([row["encoded_constraint_count"] for row in subset])) if subset else 0.0,
            "mean_skipped_constraint_count": float(np.mean([row["skipped_constraint_count"] for row in subset])) if subset else 0.0,
        }

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_files": {
            "rule_library": str(RULE_LIBRARY_PATH),
            "optimization_queries": str(OPT_QUERIES_PATH),
            "feasible_region_labels_for_evaluation_only": str(FEASIBLE_LABELS_PATH),
        },
        "modes": {
            "z3_optimize": summarize("z3_optimize"),
            "external_optimizer_smt_oracle": summarize("external_optimizer_smt_oracle"),
        },
        "per_case": rows,
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            serializable = dict(row)
            for key, value in list(serializable.items()):
                if isinstance(value, (list, dict)):
                    serializable[key] = json_dump(value)
            writer.writerow(serializable)


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def build_report(membership_summary: dict[str, Any], optimization_summary: dict[str, Any]) -> str:
    full = load_aviation_full_results()
    asp = load_json_if_exists(PAPER_DIR / "results" / "aviation_asp_structure_summary.json")
    lines = [
        "# Aviation SMT-Only Monolithic Baseline",
        "",
        "This baseline encodes rule selection and numeric constraints as one Z3 formula. It does not explicitly construct CTHR valid rule structures or feasible cells.",
        "",
        "SMT can encode the rule semantics explicitly, but it does not natively provide KG grounding, reusable feasible-cell decomposition, or provenance-preserving certificates.",
        "",
        "The implementation uses ASP-v2 visible candidate retrieval, rule interaction metadata, rule-library numeric consequents when they can be mapped to task variables, and visible non-cell task constraints. Hidden labels and source-reference semantics are used only for evaluation.",
        "",
        "## Membership / Feasible-Region Validation",
        "",
        "| Method | Probes | Formal CSR (%) | Semantic CSR (%) | False accept (%) | False reject (%) | Median check ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    g = membership_summary["global"]
    lines.append(
        "| SMT-monolithic | {num} | {formal:.1f} | {sem:.1f} | {fa:.1f} | {fr:.1f} | {ms:.2f} |".format(
            num=g["num_probes"],
            formal=g["formal_csr"],
            sem=g["semantic_csr"],
            fa=g["false_accept"],
            fr=g["false_reject"],
            ms=g["median_smt_check_time_ms"],
        )
    )
    if full:
        s63 = full.get("section_6_3", {})
        for name, key in (("CTHR", "cthr"), ("Flat", "flat")):
            if key in s63:
                item = s63[key]
                lines.append(
                    "| {name} | {num} | {formal:.1f} | {sem:.1f} | {fa:.1f} | N/A | N/A |".format(
                        name=name,
                        num=item.get("num_probes", "N/A"),
                        formal=item.get("formal_csr", 0.0),
                        sem=item.get("sem_csr", 0.0),
                        fa=item.get("false_accept_rate", 0.0),
                    )
                )
    if asp:
        a = asp.get("after_candidate_scoped", {})
        lines.append(
            "| ASP-v2 structure-only | N/A | N/A | N/A | N/A | N/A | {ms:.2f} |".format(
                ms=a.get("mean_enumeration_time_ms", 0.0)
            )
        )

    lines.extend(
        [
            "",
            "## Optimization",
            "",
            "| Method | Cases | Formal feasible (%) | Semantic valid (%) | Invalid optimized cases | Median solve ms |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for mode, item in optimization_summary["modes"].items():
        lines.append(
            "| {mode} | {num} | {formal:.1f} | {sem:.1f} | {invalid} | {ms:.2f} |".format(
                mode=mode,
                num=item["num_cases"],
                formal=item["formal_feasible_smt"],
                sem=item["semantic_valid"],
                invalid=item["invalid_optimized_cases"],
                ms=item["median_solve_time_ms"],
            )
        )
    if full:
        s64 = full.get("section_6_4", {}).get("summary", {})
        for name, key in (("CTHR", "cthr"), ("Flat", "flat")):
            if key in s64:
                item = s64[key]
                lines.append(
                    "| {name} | {num} | {formal:.1f} | {sem:.1f} | N/A | {ms:.2f} |".format(
                        name=name,
                        num=item.get("num_cases", "N/A"),
                        formal=item.get("formal_csr", 0.0),
                        sem=item.get("sem_csr", 0.0),
                        ms=item.get("median_query_ms", 0.0),
                    )
                )
    lines.extend(
        [
            "",
            "## Encoding Diagnostics",
            "",
            "- Mean membership candidate rule count: {:.1f}".format(g["mean_candidate_rule_count"]),
            "- Mean encoded SMT constraints: {:.1f}".format(g["mean_encoded_constraint_count"]),
            "- Mean skipped numeric consequents/constraints: {:.1f}".format(g["mean_skipped_constraint_count"]),
            "- Skipped items are reported because some KG rule consequents are textual, formula-valued, or use variables that cannot be mapped unambiguously to the compact benchmark decision variables.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument("--external-maxiter", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    rule_library = read_json(RULE_LIBRARY_PATH)
    queries = read_json(OPT_QUERIES_PATH)["items"]
    feasible_items = read_json(FEASIBLE_LABELS_PATH)["items"]

    start = time.perf_counter()
    membership_rows, membership_summary = run_membership(rule_library, queries, feasible_items, args.timeout_ms)
    optimization_rows, optimization_summary = run_optimization(
        rule_library,
        queries,
        feasible_items,
        timeout_ms=args.timeout_ms,
        external_maxiter=args.external_maxiter,
        seed=args.seed,
    )
    runtime_s = time.perf_counter() - start
    membership_summary["runtime_s"] = runtime_s
    optimization_summary["runtime_s"] = runtime_s

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    membership_csv = out_dir / OUT_MEMBERSHIP_CSV
    membership_json = out_dir / OUT_MEMBERSHIP_JSON
    optimization_csv = out_dir / OUT_OPT_CSV
    optimization_json = out_dir / OUT_OPT_JSON
    report_path = out_dir / OUT_REPORT

    write_csv(
        membership_csv,
        membership_rows,
        [
            "task_id",
            "title",
            "probe_type",
            "x",
            "smt_status",
            "smt_accepts",
            "semantic_valid",
            "false_accept",
            "false_reject",
            "selected_rule_ids",
            "check_time_ms",
            "candidate_rule_count",
            "encoded_constraint_count",
            "skipped_constraint_count",
            "encoded_rule_library_constraint_count",
            "encoded_visible_constraint_count",
            "error",
        ],
    )
    write_csv(
        optimization_csv,
        optimization_rows,
        [
            "task_id",
            "title",
            "mode",
            "status",
            "optimized_x",
            "objective_value",
            "selected_rule_ids",
            "formal_feasible_smt",
            "semantic_valid",
            "invalid_optimized_case",
            "solve_time_ms",
            "candidate_rule_count",
            "encoded_constraint_count",
            "skipped_constraint_count",
            "encoded_rule_library_constraint_count",
            "encoded_visible_constraint_count",
            "oracle_calls",
            "error",
        ],
    )
    membership_json.write_text(json.dumps(membership_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    optimization_json.write_text(json.dumps(optimization_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(build_report(membership_summary, optimization_summary), encoding="utf-8")

    print(
        json.dumps(
            {
                "out_membership_csv": str(membership_csv),
                "out_membership_json": str(membership_json),
                "out_optimization_csv": str(optimization_csv),
                "out_optimization_json": str(optimization_json),
                "out_report": str(report_path),
                "membership": membership_summary["global"],
                "optimization": optimization_summary["modes"],
                "runtime_s": runtime_s,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
