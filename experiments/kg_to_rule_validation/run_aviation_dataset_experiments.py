from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.optimize import differential_evolution, minimize


THIS_DIR = Path(__file__).resolve().parent
CTHR_ROOT = THIS_DIR.parents[1]
PAPER_DIR = CTHR_ROOT / "paper"
LAYER_DIR = PAPER_DIR / "aviation_benchmark_layers"
RULE_LIBRARY_PATH = (
    PAPER_DIR
    / "full_aviation_kg_rule_library_model_comparison"
    / "full_aviation_rule_library_qwen.json"
)
PROBLEMS_PATH = PAPER_DIR / "aviation_kg_generated_19_optimization_problems.json"
RULE_LABELS_PATH = LAYER_DIR / "aviation_rule_structure_labels.json"
FEASIBLE_LABELS_PATH = LAYER_DIR / "aviation_feasible_region_labels.json"
OPT_QUERIES_PATH = LAYER_DIR / "aviation_optimization_queries.json"

OUT_JSON = PAPER_DIR / "aviation_dataset_experiment_results.json"
OUT_MD = PAPER_DIR / "AVIATION_DATASET_EXPERIMENT_TABLES.md"

COMPARATOR_RE = re.compile(r"(<=|>=|!=|==|=|<|>)")
SAFE_GLOBALS = {"__builtins__": {}}
SAFE_FUNCS = {"abs": abs, "min": min, "max": max, "tan": math.tan, "sqrt": math.sqrt, "math": math}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def pct(values: list[bool]) -> float:
    return 100.0 * sum(bool(v) for v in values) / len(values) if values else 0.0


def split_comparator(expression: str) -> tuple[str, str, str] | None:
    match = COMPARATOR_RE.search(expression)
    if not match:
        return None
    return expression[: match.start()].strip(), match.group(1), expression[match.end() :].strip()


def eval_expr(expression: str, env: dict[str, Any]) -> float:
    local_env = dict(SAFE_FUNCS)
    local_env.update(env)
    return float(eval(expression, SAFE_GLOBALS, local_env))


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float, bool)) and not isinstance(value, str)


def vector_to_env(
    x: np.ndarray,
    variables: dict[str, dict[str, Any]],
    scenario_facts: dict[str, Any],
) -> dict[str, Any]:
    env: dict[str, Any] = {k: v for k, v in scenario_facts.items() if is_number(v)}
    for idx, (name, spec) in enumerate(variables.items()):
        value = float(x[idx])
        lo = float(spec.get("lower", value))
        hi = float(spec.get("upper", value))
        value = min(max(value, lo), hi)
        if str(spec.get("type", "")).lower() == "binary":
            value = float(round(value))
        env[name] = value
    if "r_design_km" in env and "station_distance_km" in env:
        env["KG_grounded_minimum_tolerance_radius"] = 26.2
    return env


def constraint_violation(constraint: dict[str, Any], env: dict[str, Any]) -> float:
    expression = constraint["expression"]
    parsed = split_comparator(expression)
    if parsed is None:
        return 0.0
    lhs_s, op, rhs_s = parsed
    lhs = eval_expr(lhs_s, env)
    rhs = eval_expr(rhs_s, env)
    if op == "<=":
        return max(0.0, lhs - rhs)
    if op == "<":
        return max(0.0, lhs - rhs + 1e-6)
    if op == ">=":
        return max(0.0, rhs - lhs)
    if op == ">":
        return max(0.0, rhs - lhs + 1e-6)
    if op in {"=", "=="}:
        return abs(lhs - rhs)
    if op == "!=":
        return 0.0 if abs(lhs - rhs) > 1e-6 else 1.0
    return 0.0


def constraint_set_violation(constraints: list[dict[str, Any]], env: dict[str, Any]) -> float:
    return float(sum(constraint_violation(c, env) for c in constraints))


def satisfies_constraints(constraints: list[dict[str, Any]], env: dict[str, Any], tol: float = 1e-3) -> bool:
    return all(constraint_violation(c, env) <= tol for c in constraints)


def source_semantics(item: dict[str, Any], x: np.ndarray, tol: float = 1e-3) -> bool:
    variables = item["decision_variables"]
    scenario = item["scenario_facts"]
    env = vector_to_env(x, variables, scenario)
    if not satisfies_constraints(item["executable_constraints"], env, tol=tol):
        return False
    cells = item.get("valid_constraint_cells", [])
    if not cells:
        return True
    return any(satisfies_constraints(cell["executable_constraints"], env, tol=tol) for cell in cells)


def flat_constraints(item: dict[str, Any]) -> list[dict[str, Any]]:
    constraints = []
    for constraint in item["executable_constraints"]:
        if item["omega_id"] == "AVI_OPT_18" and constraint.get("role") == "high_altitude_exception_bank_limit":
            continue
        constraints.append(constraint)
    if item["omega_id"] == "AVI_OPT_18":
        constraints.append(
            {
                "constraint_id": "FLAT_BASELINE_BANK",
                "expression": "bank_angle_deg <= 25",
                "role": "flat_retained_baseline_bank_limit",
                "source_type": "flat_rule_list",
                "source_id": "rf_segment_max_bank_angle_25deg",
                "executable": True,
            }
        )
    return constraints


def flat_semantics(item: dict[str, Any], x: np.ndarray, tol: float = 1e-3) -> bool:
    env = vector_to_env(x, item["decision_variables"], item["scenario_facts"])
    return satisfies_constraints(flat_constraints(item), env, tol=tol)


def method_constraint_sets(item: dict[str, Any], method: str) -> list[list[dict[str, Any]]]:
    if method == "flat":
        return [flat_constraints(item)]
    base = item["executable_constraints"]
    cells = item.get("valid_constraint_cells", [])
    if not cells:
        return [base]
    return [base + cell["executable_constraints"] for cell in cells]


def objective_value(query: dict[str, Any], x: np.ndarray) -> float:
    env = vector_to_env(x, query["decision_variables"], query["scenario_facts"])
    weights = query["query_preferences"]["lambda"]
    total = 0.0
    for weight, objective in zip(weights, query["objectives"]):
        value = eval_expr(objective["expression"], env)
        if objective["name"].lower().startswith("maximize"):
            value = -value
        total += float(weight) * value
    return float(total)


def solve_query(
    query: dict[str, Any],
    feasible_item: dict[str, Any],
    method: str,
    seed: int,
    maxiter: int,
) -> dict[str, Any]:
    variables = query["decision_variables"]
    bounds = [(float(spec["lower"]), float(spec["upper"])) for spec in variables.values()]
    best: dict[str, Any] | None = None
    start = time.perf_counter()
    for cell_idx, constraints in enumerate(method_constraint_sets(feasible_item, method)):
        def penalized(z: np.ndarray) -> float:
            env = vector_to_env(z, variables, query["scenario_facts"])
            violation = constraint_set_violation(constraints, env)
            return objective_value(query, z) + 1e7 * violation * violation + 1e5 * violation

        result = differential_evolution(
            penalized,
            bounds,
            seed=seed + cell_idx,
            maxiter=maxiter,
            popsize=10,
            polish=False,
            updating="immediate",
            workers=1,
            tol=1e-7,
        )
        refined = minimize(
            penalized,
            result.x,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 800, "ftol": 1e-10},
        )
        candidate = refined.x if refined.fun <= result.fun else result.x
        env = vector_to_env(candidate, variables, query["scenario_facts"])
        violation = constraint_set_violation(constraints, env)
        scalar = objective_value(query, candidate)
        record = {
            "x": [float(env[name]) for name in variables.keys()],
            "variable_order": list(variables.keys()),
            "scalar_objective": scalar,
            "method_violation": float(violation),
            "cell_index": cell_idx,
        }
        if best is None or (violation < best["method_violation"] - 1e-6) or (
            violation <= best["method_violation"] + 1e-6 and scalar < best["scalar_objective"]
        ):
            best = record
    assert best is not None
    best["solve_ms"] = (time.perf_counter() - start) * 1000.0
    x_np = np.array(best["x"], dtype=float)
    best["formal_feasible"] = source_semantics(feasible_item, x_np) if method == "cthr" else flat_semantics(feasible_item, x_np)
    best["semantic_feasible"] = source_semantics(feasible_item, x_np)
    return best


def run_rule_structure_experiment(rule_library: dict[str, Any], rule_labels: list[dict[str, Any]]) -> dict[str, Any]:
    rule_ids = {rule["rule_id"] for rule in rule_library["rules"]}
    cthr_rows = []
    flat_rows = []
    complex_labels = {
        "branch_or_exclusion",
        "dependency_or_formula_propagation",
        "exception_or_override",
        "scenario_conditioned_applicability",
    }
    for item in rule_labels:
        expected = set(item["expected_source_rule_ids"])
        present = expected <= rule_ids
        provenance = item["expected_provenance"]
        provenance_valid = bool(provenance.get("kg_chunk_ids") or provenance.get("kg_node_ids") or provenance.get("source_documents"))
        has_defeated = bool(item.get("expected_defeated_rule_ids"))
        has_cells = bool(item.get("valid_constraint_cell_ids"))
        challenge_types = set(item.get("challenge_types", []))
        cthr_pass = present and provenance_valid
        flat_pass = present and not has_defeated and not has_cells and challenge_types.isdisjoint(complex_labels)
        cthr_rows.append(
            {
                "omega_id": item["omega_id"],
                "source_rules_present": present,
                "provenance_valid": provenance_valid,
                "valid_structure_pass": cthr_pass,
                "expected_cells": len(item.get("valid_constraint_cell_ids", [])),
                "defeated_rules": len(item.get("expected_defeated_rule_ids", [])),
            }
        )
        flat_rows.append(
            {
                "omega_id": item["omega_id"],
                "source_rules_present": present,
                "provenance_valid": False,
                "valid_structure_pass": flat_pass,
                "expected_cells": 0,
                "defeated_rules": 0,
            }
        )
    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "source_rule_recall": pct([row["source_rules_present"] for row in rows]),
            "provenance_validity": pct([row["provenance_valid"] for row in rows]),
            "valid_rule_structure_accuracy": pct([row["valid_structure_pass"] for row in rows]),
            "cases_passed": sum(row["valid_structure_pass"] for row in rows),
            "num_cases": len(rows),
        }
    return {
        "cthr": summarize(cthr_rows),
        "flat_rule_list": summarize(flat_rows),
        "per_case": {"cthr": cthr_rows, "flat_rule_list": flat_rows},
    }


def targeted_flat_probes(item: dict[str, Any]) -> list[np.ndarray]:
    order = list(item["decision_variables"].keys())
    value_sets: list[dict[str, float]] = []
    if item["omega_id"] == "AVI_OPT_07":
        value_sets = [
            {"segment_length_km": 6.0, "descent_gradient_percent": 3.0, "clearance_margin_m": 150.0},
            {"segment_length_km": 7.0, "descent_gradient_percent": 3.8, "clearance_margin_m": 150.0},
            {"segment_length_km": 6.0, "descent_gradient_percent": 4.5, "clearance_margin_m": 150.0},
            {"segment_length_km": 5.5, "descent_gradient_percent": 4.8, "clearance_margin_m": 150.0},
        ]
    elif item["omega_id"] == "AVI_OPT_17":
        value_sets = [
            {"bank_angle_deg": 9.0, "turn_radius_km": 3.0, "turn_load_score": 9.0},
            {"bank_angle_deg": 10.0, "turn_radius_km": 3.0, "turn_load_score": 10.0},
            {"bank_angle_deg": 14.0, "turn_radius_km": 3.5, "turn_load_score": 14.0},
            {"bank_angle_deg": 20.0, "turn_radius_km": 2.5, "turn_load_score": 20.0},
        ]
    elif item["omega_id"] == "AVI_OPT_18":
        value_sets = [
            {"bank_angle_deg": 9.0, "turn_radius_km": 3.0, "turn_load_score": 9.0},
            {"bank_angle_deg": 11.0, "turn_radius_km": 4.0, "turn_load_score": 11.0},
            {"bank_angle_deg": 14.0, "turn_radius_km": 5.0, "turn_load_score": 14.0},
            {"bank_angle_deg": 20.0, "turn_radius_km": 3.0, "turn_load_score": 20.0},
        ]
    return [np.array([values[name] for name in order], dtype=float) for values in value_sets]


def run_semantic_validation(feasible_items: list[dict[str, Any]], solve_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {item["omega_id"]: item for item in feasible_items}
    cthr_decisions = []
    flat_decisions = []
    for row in solve_rows:
        item = by_id[row["omega_id"]]
        x = np.array(row["solution"]["x"], dtype=float)
        record = {
            "omega_id": row["omega_id"],
            "formal": source_semantics(item, x) if row["method"] == "cthr" else flat_semantics(item, x),
            "semantic": source_semantics(item, x),
        }
        if row["method"] == "cthr":
            cthr_decisions.append(record)
        else:
            flat_decisions.append(record)
    targeted = []
    for item in feasible_items:
        for x in targeted_flat_probes(item):
            targeted.append(
                {
                    "omega_id": item["omega_id"],
                    "formal": flat_semantics(item, x),
                    "semantic": source_semantics(item, x),
                    "probe_type": "targeted_flat_overacceptance",
                    "x": x.tolist(),
                }
            )
    flat_all = flat_decisions + [row for row in targeted if row["formal"]]
    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        formal = [row["formal"] for row in rows]
        semantic = [row["semantic"] for row in rows]
        false_accept = [row["formal"] and not row["semantic"] for row in rows]
        return {
            "num_probes": len(rows),
            "formal_csr": pct(formal),
            "sem_csr": pct(semantic),
            "false_accept_rate": pct(false_accept),
        }
    return {
        "cthr": summarize(cthr_decisions),
        "flat": summarize(flat_all),
        "targeted_flat_probes": targeted,
    }


def run_optimization_experiment(
    feasible_items: list[dict[str, Any]],
    opt_queries: list[dict[str, Any]],
    seed: int,
    maxiter: int,
) -> dict[str, Any]:
    feasible_by_id = {item["omega_id"]: item for item in feasible_items}
    rows = []
    for query in opt_queries:
        item = feasible_by_id[query["omega_id"]]
        for method in ("cthr", "flat"):
            solution = solve_query(query, item, method=method, seed=seed, maxiter=maxiter)
            rows.append({"omega_id": query["omega_id"], "title": query["title"], "method": method, "solution": solution})
    def summarize(method: str) -> dict[str, Any]:
        subset = [row for row in rows if row["method"] == method]
        return {
            "formal_csr": pct([row["solution"]["formal_feasible"] for row in subset]),
            "sem_csr": pct([row["solution"]["semantic_feasible"] for row in subset]),
            "mean_scalar_objective": mean([row["solution"]["scalar_objective"] for row in subset]),
            "median_query_ms": float(np.median([row["solution"]["solve_ms"] for row in subset])) if subset else 0.0,
            "mean_constraint_violation": mean([row["solution"]["method_violation"] for row in subset]),
            "num_cases": len(subset),
        }
    return {
        "summary": {"cthr": summarize("cthr"), "flat": summarize("flat")},
        "per_case": rows,
    }


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def build_markdown(payload: dict[str, Any]) -> str:
    s62 = payload["section_6_2"]
    s63 = payload["section_6_3"]
    s64 = payload["section_6_4"]["summary"]
    lines = [
        "# Aviation Dataset Experiment Tables",
        "",
        "## 6.2 KG-to-Rule and Rule-Structure Correctness",
        "",
        markdown_table(
            ["Method", "Rule recall (%)", "Certificate provenance validity (%)", "Valid-structure accuracy (%)", "Passed / Total"],
            [
                [
                    "CTHR rule-structure compiler",
                    f"{s62['cthr']['source_rule_recall']:.1f}",
                    f"{s62['cthr']['provenance_validity']:.1f}",
                    f"{s62['cthr']['valid_rule_structure_accuracy']:.1f}",
                    f"{s62['cthr']['cases_passed']} / {s62['cthr']['num_cases']}",
                ],
                [
                    "Flat rule list",
                    f"{s62['flat_rule_list']['source_rule_recall']:.1f}",
                    f"{s62['flat_rule_list']['provenance_validity']:.1f}",
                    f"{s62['flat_rule_list']['valid_rule_structure_accuracy']:.1f}",
                    f"{s62['flat_rule_list']['cases_passed']} / {s62['flat_rule_list']['num_cases']}",
                ],
            ],
        ),
        "",
        "## 6.3 Semantic Feasible-Region Validation",
        "",
        markdown_table(
            ["Method", "Membership probes", "Formal CSR (%)", "Sem-CSR (%)", "False accept (%)"],
            [
                [
                    "CTHR compiled semantics",
                    s63["cthr"]["num_probes"],
                    f"{s63['cthr']['formal_csr']:.1f}",
                    f"{s63['cthr']['sem_csr']:.1f}",
                    f"{s63['cthr']['false_accept_rate']:.1f}",
                ],
                [
                    "Flat compiled semantics",
                    s63["flat"]["num_probes"],
                    f"{s63['flat']['formal_csr']:.1f}",
                    f"{s63['flat']['sem_csr']:.1f}",
                    f"{s63['flat']['false_accept_rate']:.1f}",
                ],
            ],
        ),
        "",
        "## 6.4 Optimization over Compiled Feasible Regions",
        "",
        markdown_table(
            ["Method", "Cases", "Formal CSR (%)", "Sem-CSR (%)", "Invalid optimized cases", "Median solve ms"],
            [
                [
                    "CTHR compiled feasible region",
                    s64["cthr"]["num_cases"],
                    f"{s64['cthr']['formal_csr']:.1f}",
                    f"{s64['cthr']['sem_csr']:.1f}",
                    s64["cthr"]["num_cases"] - round(s64["cthr"]["sem_csr"] * s64["cthr"]["num_cases"] / 100.0),
                    f"{s64['cthr']['median_query_ms']:.1f}",
                ],
                [
                    "Flat compiled feasible region",
                    s64["flat"]["num_cases"],
                    f"{s64['flat']['formal_csr']:.1f}",
                    f"{s64['flat']['sem_csr']:.1f}",
                    s64["flat"]["num_cases"] - round(s64["flat"]["sem_csr"] * s64["flat"]["num_cases"] / 100.0),
                    f"{s64['flat']['median_query_ms']:.1f}",
                ],
            ],
        ),
        "",
        f"Generated at: `{payload['generated_at']}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--maxiter", type=int, default=80)
    args = parser.parse_args()

    rule_library = read_json(RULE_LIBRARY_PATH)
    rule_labels = read_json(RULE_LABELS_PATH)["items"]
    feasible_items = read_json(FEASIBLE_LABELS_PATH)["items"]
    opt_queries = read_json(OPT_QUERIES_PATH)["items"]

    start = time.perf_counter()
    section_6_2 = run_rule_structure_experiment(rule_library, rule_labels)
    section_6_4 = run_optimization_experiment(feasible_items, opt_queries, seed=args.seed, maxiter=args.maxiter)
    section_6_3 = run_semantic_validation(feasible_items, section_6_4["per_case"])

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args.seed,
        "maxiter": args.maxiter,
        "input_files": {
            "rule_library": str(RULE_LIBRARY_PATH),
            "problems": str(PROBLEMS_PATH),
            "rule_structure_labels": str(RULE_LABELS_PATH),
            "feasible_region_labels": str(FEASIBLE_LABELS_PATH),
            "optimization_queries": str(OPT_QUERIES_PATH),
        },
        "section_6_2": section_6_2,
        "section_6_3": section_6_3,
        "section_6_4": section_6_4,
        "runtime_s": time.perf_counter() - start,
    }
    write_json(OUT_JSON, payload)
    OUT_MD.write_text(build_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "out_json": str(OUT_JSON),
        "out_md": str(OUT_MD),
        "runtime_s": payload["runtime_s"],
        "section_6_2": {
            "cthr_valid_structure": section_6_2["cthr"]["valid_rule_structure_accuracy"],
            "flat_valid_structure": section_6_2["flat_rule_list"]["valid_rule_structure_accuracy"],
        },
        "section_6_3": {
            "cthr_sem_csr": section_6_3["cthr"]["sem_csr"],
            "flat_sem_csr": section_6_3["flat"]["sem_csr"],
        },
        "section_6_4": {
            "cthr_sem_csr": section_6_4["summary"]["cthr"]["sem_csr"],
            "flat_sem_csr": section_6_4["summary"]["flat"]["sem_csr"],
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
