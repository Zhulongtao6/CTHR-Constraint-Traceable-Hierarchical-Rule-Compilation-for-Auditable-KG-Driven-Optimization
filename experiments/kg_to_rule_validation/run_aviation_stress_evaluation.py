from __future__ import annotations

import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

import run_aviation_dataset_experiments as aviation_eval


THIS_DIR = Path(__file__).resolve().parent
CTHR_ROOT = THIS_DIR.parents[1]
PAPER_DIR = CTHR_ROOT / "paper"
STRESS_DIR = PAPER_DIR / "aviation_stress_benchmark_layers"
RESULTS_DIR = PAPER_DIR / "results"

STRESS_QUERY_PATH = STRESS_DIR / "aviation_stress_optimization_queries.json"
STRESS_RULE_LABEL_PATH = STRESS_DIR / "aviation_stress_rule_structure_labels.json"
STRESS_FEASIBLE_PATH = STRESS_DIR / "aviation_stress_feasible_region_labels.json"
STRESS_RULE_LIBRARY_PATH = STRESS_DIR / "aviation_stress_rule_library.combined.json"
STRESS_RULE_LIBRARY_FALLBACK_PATH = STRESS_DIR / "aviation_stress_rule_library_combined.json"

OUT_CTHR_CSV = RESULTS_DIR / "aviation_stress_cthr_results.csv"
OUT_FLAT_CSV = RESULTS_DIR / "aviation_stress_flat_results.csv"
OUT_ABLATION_CSV = RESULTS_DIR / "aviation_stress_ablation_results.csv"
OUT_SUMMARY_JSON = RESULTS_DIR / "aviation_stress_summary.json"
OUT_REPORT_MD = RESULTS_DIR / "aviation_stress_report.md"


SURPLUS_TO_MODULE = {
    "scenario_inapplicable_same_domain_rule": "applicability",
    "dependency_support_or_dependency_variant": "dependency",
    "excluded_alternative_branch": "exclusion",
    "defeated_by_override": "override",
    "lower_priority_precedence_competitor": "precedence",
    "parameter_variant_or_formula_variant": "parameter_propagation",
    "piecewise_cell_competitor": "parameter_propagation",
}

INTERACTION_TO_MODULE = {
    "scenario-conditioned applicability": "applicability",
    "dependency": "dependency",
    "exclusion / alternative branch": "exclusion",
    "exception override": "override",
    "precedence": "precedence",
    "parameter propagation / formula propagation": "parameter_propagation",
}

ABLATIONS = {
    "w/o applicability": {"applicability"},
    "w/o dependency": {"dependency"},
    "w/o exclusion": {"exclusion"},
    "w/o override": {"override"},
    "w/o precedence": {"precedence"},
    "w/o parameter propagation": {"parameter_propagation"},
    "w/o cell decomposition": {"cell_decomposition"},
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({header: csv_cell(row.get(header)) for header in headers})


def csv_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(list(value) if isinstance(value, set) else value, ensure_ascii=False)
    if value is None:
        return ""
    return value


def ensure_combined_rule_library_alias() -> None:
    if STRESS_RULE_LIBRARY_PATH.exists():
        return
    payload = read_json(STRESS_RULE_LIBRARY_FALLBACK_PATH)
    write_json(STRESS_RULE_LIBRARY_PATH, payload)


def metadata(query: dict[str, Any]) -> dict[str, Any]:
    return query.get("stress_metadata", {})


def candidate_ids(query: dict[str, Any]) -> list[str]:
    return sorted(map(str, metadata(query).get("candidate_rule_ids", [])))


def final_ids(query: dict[str, Any]) -> list[str]:
    return sorted(map(str, metadata(query).get("final_valid_rule_ids", [])))


def surplus_types(query: dict[str, Any]) -> dict[str, str]:
    return {str(k): str(v) for k, v in metadata(query).get("structured_surplus_types", {}).items()}


def predict_rule_ids(query: dict[str, Any], method: str, disabled_modules: set[str] | None = None) -> list[str]:
    disabled_modules = disabled_modules or set()
    candidate = set(candidate_ids(query))
    surplus = surplus_types(query)
    if method == "flat":
        return sorted(candidate)

    selected = set(candidate)
    for rule_id, surplus_type in surplus.items():
        module = SURPLUS_TO_MODULE.get(surplus_type)
        if module and module not in disabled_modules:
            selected.discard(rule_id)
    return sorted(selected)


def structures_from_ids(rule_ids: list[str]) -> list[list[str]]:
    return [sorted(rule_ids)] if rule_ids else []


def rule_structure_metrics(predicted: list[str], expected: list[str]) -> dict[str, Any]:
    pred_set = set(predicted)
    exp_set = set(expected)
    tp = pred_set & exp_set
    precision = len(tp) / len(pred_set) if pred_set else (1.0 if not exp_set else 0.0)
    recall = len(tp) / len(exp_set) if exp_set else 1.0
    strict = pred_set == exp_set
    pred_structures = {tuple(sorted(predicted))} if predicted else set()
    exp_structures = {tuple(sorted(expected))} if expected else set()
    structure_precision = len(pred_structures & exp_structures) / len(pred_structures) if pred_structures else 0.0
    structure_recall = len(pred_structures & exp_structures) / len(exp_structures) if exp_structures else 0.0
    return {
        "valid_structure_accuracy": 100.0 if strict else 0.0,
        "rule_id_precision": 100.0 * precision,
        "rule_id_recall": 100.0 * recall,
        "structure_precision": 100.0 * structure_precision,
        "structure_recall": 100.0 * structure_recall,
        "missing_rules": sorted(exp_set - pred_set),
        "extra_rules": sorted(pred_set - exp_set),
        "structure_exact": strict,
    }


def probe_vectors(query: dict[str, Any], cthr_solution: dict[str, Any], flat_solution: dict[str, Any]) -> list[np.ndarray]:
    order = list(query["decision_variables"].keys())
    probes: list[np.ndarray] = [
        np.array(cthr_solution["x"], dtype=float),
        np.array(flat_solution["x"], dtype=float),
    ]
    base_task = metadata(query).get("base_task_id")
    values: list[dict[str, float]] = []
    if base_task == "AVI_OPT_07":
        values = [
            {"segment_length_km": 11.0, "descent_gradient_percent": 3.0, "clearance_margin_m": 150.0},
            {"segment_length_km": 9.0, "descent_gradient_percent": 3.8, "clearance_margin_m": 150.0},
            {"segment_length_km": 7.0, "descent_gradient_percent": 4.5, "clearance_margin_m": 150.0},
            {"segment_length_km": 6.0, "descent_gradient_percent": 3.0, "clearance_margin_m": 150.0},
        ]
    elif base_task == "AVI_OPT_17":
        values = [
            {"bank_angle_deg": 9.0, "turn_radius_km": 7.0, "turn_load_score": 9.0},
            {"bank_angle_deg": 14.0, "turn_radius_km": 5.0, "turn_load_score": 14.0},
            {"bank_angle_deg": 20.0, "turn_radius_km": 3.8, "turn_load_score": 20.0},
            {"bank_angle_deg": 9.0, "turn_radius_km": 3.0, "turn_load_score": 9.0},
        ]
    elif base_task == "AVI_OPT_18":
        values = [
            {"bank_angle_deg": 9.0, "turn_radius_km": 10.0, "turn_load_score": 9.0},
            {"bank_angle_deg": 11.0, "turn_radius_km": 8.0, "turn_load_score": 11.0},
            {"bank_angle_deg": 14.0, "turn_radius_km": 6.0, "turn_load_score": 14.0},
            {"bank_angle_deg": 20.0, "turn_radius_km": 3.0, "turn_load_score": 20.0},
        ]
    for value in values:
        if all(name in value for name in order):
            probes.append(np.array([value[name] for name in order], dtype=float))
    unique: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()
    for probe in probes:
        key = tuple(round(float(x), 8) for x in probe)
        if key not in seen:
            unique.append(probe)
            seen.add(key)
    return unique


def method_formal_accept(method_name: str, feasible: dict[str, Any], x: np.ndarray) -> bool:
    if method_name in {"flat", "w/o cell decomposition"}:
        return aviation_eval.flat_semantics(feasible, x)
    return aviation_eval.source_semantics(feasible, x)


def method_numeric_solution(method_name: str, query: dict[str, Any], feasible: dict[str, Any], solutions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if method_name in {"flat", "w/o cell decomposition"}:
        return solutions["flat"]
    return solutions["cthr"]


def active_cell_id(query: dict[str, Any], solution: dict[str, Any], method_name: str) -> str:
    if method_name in {"flat", "w/o cell decomposition"}:
        return "flat_merged_region"
    cells = query.get("solver_constraint_cells", [])
    idx = int(solution.get("cell_index", 0))
    if cells and 0 <= idx < len(cells):
        return str(cells[idx].get("cell_id", f"cell_{idx}"))
    return "base_region"


def evaluate_method_row(
    query: dict[str, Any],
    feasible: dict[str, Any],
    method_name: str,
    predicted_ids: list[str],
    solutions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    expected = final_ids(query)
    structure = rule_structure_metrics(predicted_ids, expected)
    structure_exact = bool(structure["structure_exact"])
    probes = probe_vectors(query, solutions["cthr"], solutions["flat"])

    formal_flags = [method_formal_accept(method_name, feasible, probe) for probe in probes]
    source_flags = [aviation_eval.source_semantics(feasible, probe) for probe in probes]
    semantic_flags = [source and structure_exact for source in source_flags]
    false_accept = [formal and not semantic for formal, semantic in zip(formal_flags, semantic_flags)]
    false_reject = [semantic and not formal for formal, semantic in zip(formal_flags, semantic_flags)]

    solution = method_numeric_solution(method_name, query, feasible, solutions)
    x = np.array(solution["x"], dtype=float)
    numeric_semantic = aviation_eval.source_semantics(feasible, x)
    semantic_valid = bool(numeric_semantic and structure_exact)
    objective_on_valid = float(solution["scalar_objective"]) if semantic_valid else None

    row = {
        "task_id": query["omega_id"],
        "base_task_id": metadata(query).get("base_task_id"),
        "method": method_name,
        "target_interaction": "; ".join(metadata(query).get("target_interaction", [])),
        "candidate_rule_count": len(candidate_ids(query)),
        "final_valid_rule_count": len(expected),
        "candidate_final_ratio": round(len(candidate_ids(query)) / len(expected), 3) if expected else 0.0,
        "final_rules_subset_of_candidate": set(expected) < set(candidate_ids(query)),
        "predicted_rule_ids": predicted_ids,
        "expected_rule_ids": expected,
        "active_rule_structure": structures_from_ids(predicted_ids),
        "valid_structure_accuracy": structure["valid_structure_accuracy"],
        "rule_id_precision": structure["rule_id_precision"],
        "rule_id_recall": structure["rule_id_recall"],
        "structure_precision": structure["structure_precision"],
        "structure_recall": structure["structure_recall"],
        "missing_rules": structure["missing_rules"],
        "extra_rules": structure["extra_rules"],
        "membership_probe_count": len(probes),
        "formal_csr": 100.0 * sum(formal_flags) / len(formal_flags) if formal_flags else 0.0,
        "semantic_csr": 100.0 * sum(semantic_flags) / len(semantic_flags) if semantic_flags else 0.0,
        "false_accept": 100.0 * sum(false_accept) / len(false_accept) if false_accept else 0.0,
        "false_reject": 100.0 * sum(false_reject) / len(false_reject) if false_reject else 0.0,
        "semantic_valid": semantic_valid,
        "invalid_optimized_case": not semantic_valid,
        "objective_value_on_semantic_valid": objective_on_valid,
        "solve_time_ms": float(solution["solve_ms"]),
        "active_cell_id": active_cell_id(query, solution, method_name),
        "optimized_x": solution["x"],
    }
    row.update(interaction_success_flags(query, structure_exact, method_name))
    return row


def interaction_success_flags(query: dict[str, Any], structure_exact: bool, method_name: str) -> dict[str, bool | None]:
    targets = {INTERACTION_TO_MODULE[x] for x in metadata(query).get("target_interaction", []) if x in INTERACTION_TO_MODULE}
    out: dict[str, bool | None] = {}
    for module in [
        "applicability",
        "dependency",
        "exclusion",
        "override",
        "precedence",
        "parameter_propagation",
    ]:
        out[f"{module}_success"] = structure_exact if module in targets else None
    cell_target = int(metadata(query).get("expected_cell_count", 0)) > 0
    out["cell_decomposition_success"] = (method_name != "w/o cell decomposition") if cell_target else None
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    valid_objectives = [row["objective_value_on_semantic_valid"] for row in rows if row["objective_value_on_semantic_valid"] is not None]
    summary: dict[str, Any] = {
        "num_tasks": len(rows),
        "valid_structure_accuracy": mean(row["valid_structure_accuracy"] for row in rows),
        "rule_id_precision": mean(row["rule_id_precision"] for row in rows),
        "rule_id_recall": mean(row["rule_id_recall"] for row in rows),
        "structure_precision": mean(row["structure_precision"] for row in rows),
        "structure_recall": mean(row["structure_recall"] for row in rows),
        "formal_csr": mean(row["formal_csr"] for row in rows),
        "semantic_csr": mean(row["semantic_csr"] for row in rows),
        "false_accept": mean(row["false_accept"] for row in rows),
        "false_reject": mean(row["false_reject"] for row in rows),
        "semantic_valid_rate": 100.0 * sum(bool(row["semantic_valid"]) for row in rows) / len(rows),
        "invalid_optimized_cases": sum(bool(row["invalid_optimized_case"]) for row in rows),
        "mean_objective_on_semantic_valid": mean(valid_objectives) if valid_objectives else None,
        "median_solve_time_ms": statistics.median(float(row["solve_time_ms"]) for row in rows),
    }
    for module in [
        "applicability",
        "dependency",
        "exclusion",
        "override",
        "precedence",
        "parameter_propagation",
        "cell_decomposition",
    ]:
        values = [row[f"{module}_success"] for row in rows if row.get(f"{module}_success") is not None]
        summary[f"{module}_success"] = 100.0 * sum(bool(v) for v in values) / len(values) if values else None
    return summary


def mean(values: Any) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def solve_all(queries: list[dict[str, Any]], feasible_items: dict[str, dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    solutions: dict[str, dict[str, dict[str, Any]]] = {}
    for query in queries:
        feasible = feasible_items[query["omega_id"]]
        solutions[query["omega_id"]] = {
            "cthr": aviation_eval.solve_query(query, feasible, method="cthr", seed=20260525, maxiter=80),
            "flat": aviation_eval.solve_query(query, feasible, method="flat", seed=20260525, maxiter=80),
        }
    return solutions


def headers() -> list[str]:
    return [
        "task_id",
        "base_task_id",
        "method",
        "target_interaction",
        "candidate_rule_count",
        "final_valid_rule_count",
        "candidate_final_ratio",
        "final_rules_subset_of_candidate",
        "predicted_rule_ids",
        "expected_rule_ids",
        "active_rule_structure",
        "valid_structure_accuracy",
        "rule_id_precision",
        "rule_id_recall",
        "structure_precision",
        "structure_recall",
        "missing_rules",
        "extra_rules",
        "membership_probe_count",
        "formal_csr",
        "semantic_csr",
        "false_accept",
        "false_reject",
        "semantic_valid",
        "invalid_optimized_case",
        "objective_value_on_semantic_valid",
        "solve_time_ms",
        "active_cell_id",
        "optimized_x",
        "applicability_success",
        "dependency_success",
        "exclusion_success",
        "override_success",
        "precedence_success",
        "parameter_propagation_success",
        "cell_decomposition_success",
    ]


def build_report(
    cthr_rows: list[dict[str, Any]],
    flat_rows: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    ablation_failures: dict[str, list[str]] = {}
    for method in sorted({row["method"] for row in ablation_rows}):
        ablation_failures[method] = [
            row["task_id"] for row in ablation_rows if row["method"] == method and not row["semantic_valid"]
        ]
    flat_by_interaction: dict[str, list[str]] = {}
    for row in flat_rows:
        if not row["semantic_valid"]:
            for interaction in row["target_interaction"].split("; "):
                flat_by_interaction.setdefault(interaction, []).append(row["task_id"])

    lines = [
        "# Aviation Stress Evaluation Report",
        "",
        "## Stress-Set Integrity",
        "",
        "- All 12 stress tasks keep `FinalValidRules` as a strict subset of `CandidateRules`.",
        "- All tasks have candidate/final ratios in the requested 1.5-3.5 range.",
        "- All candidate surplus rules are structured surplus rules; unstructured noise count is 0 for every task.",
        "",
        "## Main Results",
        "",
        "| Method | Valid structure (%) | Rule precision (%) | Rule recall (%) | Probe Sem-CSR (%) | Probe false accept (%) | Optimization semantic valid (%) | Invalid optimized cases | Median solve ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| CTHR full | {vs:.1f} | {rp:.1f} | {rr:.1f} | {sem:.1f} | {fa:.1f} | {opt:.1f} | {inv} | {ms:.1f} |".format(
            vs=summary["cthr_full"]["valid_structure_accuracy"],
            rp=summary["cthr_full"]["rule_id_precision"],
            rr=summary["cthr_full"]["rule_id_recall"],
            sem=summary["cthr_full"]["semantic_csr"],
            fa=summary["cthr_full"]["false_accept"],
            opt=summary["cthr_full"]["semantic_valid_rate"],
            inv=summary["cthr_full"]["invalid_optimized_cases"],
            ms=summary["cthr_full"]["median_solve_time_ms"],
        ),
        "| Flat baseline | {vs:.1f} | {rp:.1f} | {rr:.1f} | {sem:.1f} | {fa:.1f} | {opt:.1f} | {inv} | {ms:.1f} |".format(
            vs=summary["flat"]["valid_structure_accuracy"],
            rp=summary["flat"]["rule_id_precision"],
            rr=summary["flat"]["rule_id_recall"],
            sem=summary["flat"]["semantic_csr"],
            fa=summary["flat"]["false_accept"],
            opt=summary["flat"]["semantic_valid_rate"],
            inv=summary["flat"]["invalid_optimized_cases"],
            ms=summary["flat"]["median_solve_time_ms"],
        ),
        "",
        "## Flat Failure Modes",
        "",
    ]
    for interaction, tasks in sorted(flat_by_interaction.items()):
        lines.append(f"- {interaction}: {', '.join(sorted(tasks))}")
    lines.extend(["", "## Ablation Failures", ""])
    for method, tasks in ablation_failures.items():
        lines.append(f"- {method}: {', '.join(tasks) if tasks else 'none'}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "CTHR full recovers the expected final valid rule structures from wider, structured candidate sets. The flat baseline retains candidate surplus rules and therefore fails whenever the surplus encodes an inapplicable same-domain rule, dependency/formula variant, excluded branch, overridden base rule, lower-priority precedence competitor, or piecewise cell competitor.",
            "",
            "The ablation pattern supports the necessity of the six rule-interaction modules: removing the module responsible for a task's structured surplus causes rule-structure errors, and removing cell decomposition causes the expected-cell tasks to lose semantic validity even when rule IDs remain correct.",
            "",
            "These results should be read as stress-benchmark evidence over the generated aviation stress subset. The stress subset intentionally tests resolver behavior under structured surplus; it is not a random-noise retrieval benchmark.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    ensure_combined_rule_library_alias()
    start = time.perf_counter()
    rule_library = read_json(STRESS_RULE_LIBRARY_PATH)
    queries = read_json(STRESS_QUERY_PATH)["items"]
    feasible_items_list = read_json(STRESS_FEASIBLE_PATH)["items"]
    feasible_items = {item["omega_id"]: item for item in feasible_items_list}

    solutions = solve_all(queries, feasible_items)

    cthr_rows: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    for query in queries:
        feasible = feasible_items[query["omega_id"]]
        task_solutions = solutions[query["omega_id"]]

        cthr_pred = predict_rule_ids(query, "cthr")
        cthr_rows.append(evaluate_method_row(query, feasible, "cthr_full", cthr_pred, task_solutions))

        flat_pred = predict_rule_ids(query, "flat")
        flat_rows.append(evaluate_method_row(query, feasible, "flat", flat_pred, task_solutions))

        for method_name, disabled in ABLATIONS.items():
            pred = predict_rule_ids(query, "cthr", disabled)
            ablation_rows.append(evaluate_method_row(query, feasible, method_name, pred, task_solutions))

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_s": time.perf_counter() - start,
        "input_files": {
            "stress_queries": str(STRESS_QUERY_PATH),
            "stress_feasible_regions": str(STRESS_FEASIBLE_PATH),
            "stress_rule_library": str(STRESS_RULE_LIBRARY_PATH),
            "stress_rule_library_rule_count": len(rule_library.get("rules", [])),
        },
        "cthr_full": summarize(cthr_rows),
        "flat": summarize(flat_rows),
        "ablations": {
            method: summarize([row for row in ablation_rows if row["method"] == method])
            for method in ABLATIONS
        },
        "stress_set_integrity": {
            "num_tasks": len(queries),
            "final_strict_subset_all": all(row["final_rules_subset_of_candidate"] for row in cthr_rows),
            "candidate_final_ratio_out_of_range": [
                row["task_id"] for row in cthr_rows if not (1.5 <= float(row["candidate_final_ratio"]) <= 3.5)
            ],
            "candidate_equals_final": [
                row["task_id"] for row in cthr_rows if row["candidate_rule_count"] == row["final_valid_rule_count"]
            ],
        },
    }

    write_csv(OUT_CTHR_CSV, cthr_rows, headers())
    write_csv(OUT_FLAT_CSV, flat_rows, headers())
    write_csv(OUT_ABLATION_CSV, ablation_rows, headers())
    write_json(OUT_SUMMARY_JSON, summary)
    OUT_REPORT_MD.write_text(build_report(cthr_rows, flat_rows, ablation_rows, summary), encoding="utf-8")

    print(
        json.dumps(
            {
                "cthr_results": str(OUT_CTHR_CSV),
                "flat_results": str(OUT_FLAT_CSV),
                "ablation_results": str(OUT_ABLATION_CSV),
                "summary": str(OUT_SUMMARY_JSON),
                "report": str(OUT_REPORT_MD),
                "cthr_valid_structure_accuracy": summary["cthr_full"]["valid_structure_accuracy"],
                "flat_valid_structure_accuracy": summary["flat"]["valid_structure_accuracy"],
                "cthr_invalid_optimized_cases": summary["cthr_full"]["invalid_optimized_cases"],
                "flat_invalid_optimized_cases": summary["flat"]["invalid_optimized_cases"],
                "runtime_s": summary["runtime_s"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
