from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

import run_section_6_2_table1_fullkg_pipeline as fullkg
import run_section_6_4_architecture_ncarb50_rule_library_compare as backend_complete


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
DATASET_ROOT = ROOT / "datasets" / "aviation_strong_mixed_v11_150"
OVERLAYS = DATASET_ROOT / "evaluation_overlays"
RULE_LIBRARIES = DATASET_ROOT / "rule_libraries"
CANONICAL_EVALUATION_REFERENCES = (
    DATASET_ROOT / "evaluation_references" / "aviation_strong_mixed_evaluation_references.json"
)


MODEL_SPECS = [
    {
        "model": "Qwen-plus",
        "provider": "qwen",
        "overlay_key": "qwen",
        "rule_library": RULE_LIBRARIES / "qwen" / "full_aviation_rule_library_qwen.json",
    },
    {
        "model": "DeepSeek-Pro",
        "provider": "deepseek",
        "overlay_key": "deepseek",
        "rule_library": RULE_LIBRARIES / "deepseek" / "full_aviation_rule_library_deepseek_strict_repaired.json",
    },
    {
        "model": "Xiaomi MIMO",
        "provider": "xiaomi_mimo",
        "overlay_key": "xiaomi_mimo",
        "rule_library": RULE_LIBRARIES / "xiaomi_mimo" / "full_aviation_rule_library_mimo.json",
    },
]


MODE = {
    "mode": "backend_complete_oracle_valid_upper",
    "description": "Use strong-aligned valid rules and backend-complete compiled templates.",
    "pass_as_valid": True,
    "extend_relation_templates": True,
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(list(value) if isinstance(value, set) else value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({header: csv_cell(row.get(header)) for header in headers})


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def rule_precision(predicted: list[str], reference: list[str]) -> float:
    if not predicted:
        return 0.0
    return len(set(predicted) & set(reference)) / len(set(predicted))


def rule_recall(predicted: list[str], reference: list[str]) -> float:
    if not reference:
        return 1.0
    return len(set(predicted) & set(reference)) / len(set(reference))


def markdown_table(rows: list[dict[str, Any]], headers: list[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(csv_cell(row.get(header)) for header in headers) + " |")
    return "\n".join(lines)


def overlay_file(model_key: str, filename: str) -> Path:
    return OVERLAYS / model_key / filename


def aviation_spec(model_spec: dict[str, Any], overlay_key: str) -> fullkg.DatasetSpec:
    return fullkg.DatasetSpec(
        name="Aviation strong mixed v11 150",
        domain="aviation",
        root=DATASET_ROOT,
        algorithm_inputs=DATASET_ROOT / "algorithm_inputs" / "aviation_strong_mixed_algorithm_inputs.json",
        scenario_models=DATASET_ROOT / "scenario_models" / "aviation_strong_mixed_public_scenario_models.json",
        evaluation_references=overlay_file(overlay_key, "evaluation_references.json"),
        rule_library=Path(model_spec["rule_library"]),
        grounding_full=RESULTS_DIR / "unused_aviation_strong_mixed_v11_150_grounding.json",
        constraint_templates=overlay_file(overlay_key, "compiled_rule_constraint_templates.json"),
    )


def overlay_alignment_summary(model_key: str) -> dict[str, Any]:
    return dict(read_json(overlay_file(model_key, "rule_id_alignment.json")).get("summary", {}))


def overlay_model_to_canonical(model_key: str) -> dict[str, list[str]]:
    payload = read_json(overlay_file(model_key, "rule_id_alignment.json"))
    out: dict[str, list[str]] = {}
    for row in payload.get("model_to_canonical", []):
        model_rule_id = str(row.get("model_rule_id"))
        out[model_rule_id] = sorted(str(item) for item in row.get("canonical_rule_ids", []))
    if out:
        return out
    for row in payload.get("canonical_to_model", []):
        canonical_rule_id = str(row.get("canonical_rule_id"))
        for model_rule_id in row.get("aligned_model_rule_ids", []):
            if not model_rule_id:
                continue
            out.setdefault(str(model_rule_id), []).append(canonical_rule_id)
    out = {rule_id: sorted(set(canonical_ids)) for rule_id, canonical_ids in out.items()}
    return out


def overlay_canonical_to_model(model_key: str) -> dict[str, list[str]]:
    payload = read_json(overlay_file(model_key, "rule_id_alignment.json"))
    out: dict[str, list[str]] = {}
    for row in payload.get("canonical_to_model", []):
        canonical_rule_id = str(row.get("canonical_rule_id"))
        model_rule_ids = row.get("aligned_model_rule_ids", [])
        out[canonical_rule_id] = sorted(str(item) for item in model_rule_ids if item)
    return out


def executable_rule_constraints(reference: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    feasible = reference.get("feasible_region", {})
    out: list[tuple[str, dict[str, Any]]] = []
    for constraint in feasible.get("executable_constraints", []):
        if isinstance(constraint, dict):
            out.append(("feasible_region.executable_constraints", constraint))
    for cell in feasible.get("valid_constraint_cells", []):
        if not isinstance(cell, dict):
            continue
        cell_id = str(cell.get("cell_id", "unknown_cell"))
        for constraint in cell.get("constraints", []):
            if isinstance(constraint, dict):
                out.append((f"feasible_region.valid_constraint_cells.{cell_id}", constraint))
    return out


def build_backend_complete_templates(
    references: dict[str, dict[str, Any]],
    algorithm_inputs: dict[str, dict[str, Any]],
    base_templates_by_rule: dict[str, list[dict[str, Any]]],
    canonical_to_model: dict[str, list[str]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    out = {rule_id: [dict(template) for template in templates] for rule_id, templates in base_templates_by_rule.items()}
    seen = {
        (rule_id, str(template.get("template_id")))
        for rule_id, templates in out.items()
        for template in templates
    }
    added_by_rule: dict[str, int] = {}
    added_by_task: dict[str, int] = {}
    for task_id, reference in references.items():
        if task_id not in algorithm_inputs:
            continue
        context = backend_complete.scalar_scenario_context(algorithm_inputs[task_id])
        for source_container, constraint in executable_rule_constraints(reference):
            source_rule_id = str(constraint.get("source_id") or "")
            target_rule_ids = sorted({source_rule_id, *canonical_to_model.get(source_rule_id, [])})
            for target_rule_id in target_rule_ids:
                template = backend_complete.relation_extended_template(
                    task_id=task_id,
                    source_container=source_container,
                    constraint=constraint,
                    context=context,
                    target_rule_id=target_rule_id,
                )
                if template is None:
                    continue
                rule_id = str(template["source_rule_id"])
                key = (rule_id, str(template["template_id"]))
                if key in seen:
                    continue
                out.setdefault(rule_id, []).append(template)
                seen.add(key)
                added_by_rule[rule_id] = added_by_rule.get(rule_id, 0) + 1
                added_by_task[task_id] = added_by_task.get(task_id, 0) + 1
    stats = {
        "enabled": True,
        "added_template_count": sum(added_by_rule.values()),
        "added_rule_count": len(added_by_rule),
        "added_task_count": len(added_by_task),
        "added_by_rule": dict(sorted(added_by_rule.items())),
        "added_by_task": dict(sorted(added_by_task.items())),
    }
    return out, stats


def canonical_reference_by_task() -> dict[str, list[str]]:
    payload = read_json(CANONICAL_EVALUATION_REFERENCES)
    out: dict[str, list[str]] = {}
    for item in payload.get("items", []):
        task_id = str(item.get("omega_id"))
        out[task_id] = fullkg.reference_rule_ids(item)
    return out


def quality_row(model_spec: dict[str, Any]) -> dict[str, Any]:
    payload = read_json(Path(model_spec["rule_library"]))
    summary = payload.get("summary", {})
    return {
        "Domain": "Aviation strong mixed v11 150",
        "Generator": model_spec["model"],
        "Rules": len(payload.get("rules", [])),
        "Provenance valid": f"{float(summary.get('mean_provenance_validity_rate', 0.0)):.1f}%",
        "Constraint grounding": f"{float(summary.get('mean_constraint_grounding_rate', 0.0)):.1f}%",
        "Relation grounding": f"{float(summary.get('mean_relation_grounding_rate', 0.0)):.1f}%",
    }


def support_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "candidate_zero": sum(1 for row in rows if row.get("unsupported_reason") == "no_grounded_candidates"),
        "cthr_no_valid": sum(1 for row in rows if row.get("unsupported_reason") == "cthr_no_valid_rules"),
        "unsupported": sum(1 for row in rows if row.get("unsupported_reason")),
    }


def aggregate_rows(rows: list[dict[str, Any]], model_name: str) -> dict[str, Any]:
    total = len(rows)
    counts = support_counts(rows)
    invalid = sum(1 for row in rows if row.get("invalid_case"))
    return {
        "Domain": "Aviation strong mixed v11 150",
        "Generator": model_name,
        "Rule Precision": pct(sum(float(row["canonical_rule_precision"]) for row in rows) / max(1, total)),
        "Rule Recall": pct(sum(float(row["canonical_rule_recall"]) for row in rows) / max(1, total)),
        "Formal CSR": pct(sum(1 for row in rows if row.get("formal_feasible")) / max(1, total)),
        "Sem-CSR": pct(sum(1 for row in rows if row.get("semantic_valid")) / max(1, total)),
        "False accept": pct(sum(1 for row in rows if row.get("false_accept")) / max(1, total)),
        "Candidate zero": counts["candidate_zero"],
        "CTHR no valid": counts["cthr_no_valid"],
        "Invalid cases": f"{invalid}/{total}",
        "Unsupported tasks": counts["unsupported"],
        "Relation templates added": rows[0].get("relation_templates_added", 0) if rows else 0,
    }


def values_from_vector(query: dict[str, Any], variables: list[str], vector: np.ndarray) -> dict[str, float]:
    values = {name: float(value) for name, value in zip(variables, vector)}
    for key, value in query.get("scenario_facts", {}).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values[key] = float(value)
    fullkg.base.add_derived_aliases(values)
    return values


def simple_variable(expr: str, variables: set[str]) -> str | None:
    expr = expr.strip()
    if expr in variables:
        return expr
    return None


def clip_value(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), lower), upper)


def propagation_feasible_point(query: dict[str, Any], constraints: list[dict[str, Any]]) -> dict[str, float] | None:
    variables = list(query.get("decision_variables", {}))
    if not variables:
        return None
    variable_set = set(variables)
    bounds: dict[str, tuple[float, float]] = {}
    values: dict[str, float] = {}
    for name in variables:
        spec = query["decision_variables"][name]
        lower = float(spec.get("lower", 0.0))
        upper = float(spec.get("upper", 1.0))
        bounds[name] = (lower, upper)
        values[name] = (lower + upper) / 2.0
        if str(spec.get("type", "")).lower() == "binary":
            values[name] = 1.0 if upper >= 1.0 else upper

    def eval_side(expr: str) -> float | None:
        env = dict(values)
        for key, value in query.get("scenario_facts", {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                env[key] = float(value)
        fullkg.base.add_derived_aliases(env)
        try:
            out = fullkg.base.eval_arithmetic(expr, env)
        except Exception:
            return None
        return float(out) if np.isfinite(out) else None

    parsed: list[tuple[str, str, str]] = []
    for constraint in constraints:
        expression = constraint.get("checker_expression") or constraint.get("expression")
        if not expression:
            continue
        normalized = fullkg.base.normalize_expression(str(expression))
        match = fullkg.base.COMPARATOR_RE.search(normalized)
        if not match:
            continue
        parsed.append((normalized[: match.start()].strip(), match.group(1), normalized[match.end() :].strip()))

    for _round in range(40):
        changed = False
        for left, op, right in parsed:
            left_var = simple_variable(left, variable_set)
            right_var = simple_variable(right, variable_set)
            left_value = eval_side(left)
            right_value = eval_side(right)
            if op in {"=", "=="}:
                if left_var and right_value is not None:
                    lower, upper = bounds[left_var]
                    new_value = clip_value(right_value, lower, upper)
                    changed = changed or abs(values[left_var] - new_value) > 1e-9
                    values[left_var] = new_value
                elif right_var and left_value is not None:
                    lower, upper = bounds[right_var]
                    new_value = clip_value(left_value, lower, upper)
                    changed = changed or abs(values[right_var] - new_value) > 1e-9
                    values[right_var] = new_value
            elif op in {"<=", "<"}:
                if left_var and right_value is not None and left_value is not None and left_value > right_value:
                    lower, upper = bounds[left_var]
                    new_value = clip_value(right_value, lower, upper)
                    changed = changed or abs(values[left_var] - new_value) > 1e-9
                    values[left_var] = new_value
                elif right_var and left_value is not None and right_value is not None and left_value > right_value:
                    lower, upper = bounds[right_var]
                    new_value = clip_value(left_value, lower, upper)
                    changed = changed or abs(values[right_var] - new_value) > 1e-9
                    values[right_var] = new_value
            elif op in {">=", ">"}:
                if left_var and right_value is not None and left_value is not None and left_value < right_value:
                    lower, upper = bounds[left_var]
                    new_value = clip_value(right_value, lower, upper)
                    changed = changed or abs(values[left_var] - new_value) > 1e-9
                    values[left_var] = new_value
                elif right_var and left_value is not None and right_value is not None and left_value < right_value:
                    lower, upper = bounds[right_var]
                    new_value = clip_value(left_value, lower, upper)
                    changed = changed or abs(values[right_var] - new_value) > 1e-9
                    values[right_var] = new_value
        if not changed:
            break

    solution = {name: values[name] for name in variables}
    if fullkg.base.constraints_satisfied(constraints, fullkg.base.with_query_values(query, solution)):
        return solution
    return None


def fast_backend_optimize(query: dict[str, Any], constraints: list[dict[str, Any]]) -> dict[str, float] | None:
    propagated = propagation_feasible_point(query, constraints)
    if propagated is not None:
        return propagated
    if os.environ.get("CTHR_NUMERIC_FALLBACK", "0") != "1":
        return None

    variables = list(query.get("decision_variables", {}))
    if not variables:
        return None
    bounds: list[tuple[float, float]] = []
    for name in variables:
        spec = query["decision_variables"][name]
        lower = float(spec.get("lower", 0.0))
        upper = float(spec.get("upper", 1.0))
        bounds.append((lower, upper))
    midpoint = np.array([(lower + upper) / 2.0 for lower, upper in bounds], dtype=float)
    lower_point = np.array([lower for lower, _upper in bounds], dtype=float)
    upper_point = np.array([upper for _lower, upper in bounds], dtype=float)
    starts = [midpoint, lower_point, upper_point]

    def make_value(expr: str):
        def fun(vector: np.ndarray) -> float:
            try:
                return fullkg.base.eval_arithmetic(expr, values_from_vector(query, variables, np.asarray(vector, dtype=float)))
            except Exception:
                return float("nan")

        return fun

    def safe_diff(left_fun, right_fun, sign: float):
        def fun(vector: np.ndarray) -> float:
            left = left_fun(vector)
            right = right_fun(vector)
            if not np.isfinite(left) or not np.isfinite(right):
                return -1e6
            return sign * (left - right)

        return fun

    def safe_eq(left_fun, right_fun):
        def fun(vector: np.ndarray) -> float:
            left = left_fun(vector)
            right = right_fun(vector)
            if not np.isfinite(left) or not np.isfinite(right):
                return 1e6
            return left - right

        return fun

    scipy_constraints: list[dict[str, Any]] = []
    for constraint in constraints:
        expression = constraint.get("checker_expression") or constraint.get("expression")
        if not expression:
            continue
        normalized = fullkg.base.normalize_expression(str(expression))
        match = fullkg.base.COMPARATOR_RE.search(normalized)
        if not match:
            continue
        left = normalized[: match.start()].strip()
        op = match.group(1)
        right = normalized[match.end() :].strip()
        left_fun = make_value(left)
        right_fun = make_value(right)
        if op in {"<=", "<"}:
            scipy_constraints.append({"type": "ineq", "fun": safe_diff(right_fun, left_fun, 1.0)})
        elif op in {">=", ">"}:
            scipy_constraints.append({"type": "ineq", "fun": safe_diff(left_fun, right_fun, 1.0)})
        elif op in {"=", "=="}:
            scipy_constraints.append({"type": "eq", "fun": safe_eq(left_fun, right_fun)})

    def objective(vector: np.ndarray) -> float:
        try:
            return fullkg.base.objective_value(query, values_from_vector(query, variables, np.asarray(vector, dtype=float)))
        except Exception:
            return 0.0

    candidates: list[tuple[float, dict[str, float]]] = []
    for start in starts:
        if not scipy_constraints:
            continue
        try:
            result = minimize(
                objective,
                start,
                method="SLSQP",
                bounds=bounds,
                constraints=scipy_constraints,
                options={"maxiter": 250, "ftol": 1e-9, "disp": False},
            )
            points = [np.asarray(result.x, dtype=float)]
        except Exception:
            points = []
        for point in points:
            values = values_from_vector(query, variables, np.asarray(point, dtype=float))
            solution = {name: values[name] for name in variables}
            if fullkg.base.constraints_satisfied(constraints, fullkg.base.with_query_values(query, solution)):
                candidates.append((fullkg.base.objective_value(query, values), solution))
    if candidates:
        return min(candidates, key=lambda item: item[0])[1]

    def penalized(vector: np.ndarray) -> float:
        values = values_from_vector(query, variables, np.asarray(vector, dtype=float))
        penalty = 0.0
        for constraint in constraints:
            expression = constraint.get("checker_expression") or constraint.get("expression")
            if not expression:
                continue
            violation = fullkg.base.constraint_violation(str(expression), values)
            penalty += violation * violation
        try:
            objective = fullkg.base.objective_value(query, values)
        except Exception:
            objective = 0.0
        return objective + 1e8 * penalty

    for start in starts:
        try:
            result = minimize(
                penalized,
                start,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 220, "ftol": 1e-10, "maxls": 30},
            )
            points = [start, np.asarray(result.x, dtype=float)]
        except Exception:
            points = [start]
        for point in points:
            values = values_from_vector(query, variables, np.asarray(point, dtype=float))
            solution = {name: values[name] for name in variables}
            if fullkg.base.constraints_satisfied(constraints, fullkg.base.with_query_values(query, solution)):
                candidates.append((fullkg.base.objective_value(query, values), solution))
    if candidates:
        return min(candidates, key=lambda item: item[0])[1]

    try:
        result = minimize(
            penalized,
            midpoint,
            method="Powell",
            bounds=bounds,
            options={"maxiter": 350, "ftol": 1e-9, "xtol": 1e-9},
        )
        values = values_from_vector(query, variables, np.asarray(result.x, dtype=float))
        solution = {name: values[name] for name in variables}
        if fullkg.base.constraints_satisfied(constraints, fullkg.base.with_query_values(query, solution)):
            return solution
    except Exception:
        return None
    return None


def model_reference_ids_for(
    raw_reference_ids: list[str],
    rule_by_id: dict[str, dict[str, Any]],
    canonical_to_model: dict[str, list[str]],
) -> list[str]:
    out: list[str] = []
    for rule_id in raw_reference_ids:
        if rule_id in rule_by_id:
            out.append(rule_id)
            continue
        for model_rule_id in canonical_to_model.get(rule_id, []):
            if model_rule_id in rule_by_id:
                out.append(model_rule_id)
                break
    return sorted(set(out))


def evaluate_model(
    model_spec: dict[str, Any],
    canonical_references: dict[str, list[str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    overlay_key = str(model_spec["overlay_key"])
    model_name = str(model_spec["model"])
    spec = aviation_spec(model_spec, overlay_key)
    algorithm_inputs = fullkg.item_map(spec.algorithm_inputs)
    scenario_models = fullkg.item_map(spec.scenario_models)
    references = fullkg.item_map(spec.evaluation_references)
    canonical_to_model = overlay_canonical_to_model(overlay_key)
    _global_templates, extension_stats = build_backend_complete_templates(
        references,
        algorithm_inputs,
        {},
        canonical_to_model,
    )
    rule_library = read_json(spec.rule_library)
    rule_by_id = {str(rule["rule_id"]): rule for rule in rule_library.get("rules", []) if rule.get("rule_id")}
    model_to_canonical = overlay_model_to_canonical(overlay_key)

    rows: list[dict[str, Any]] = []
    task_ids = sorted(algorithm_inputs)
    task_limit = int(os.environ.get("CTHR_TASK_LIMIT", "0") or "0")
    if task_limit > 0:
        task_ids = task_ids[:task_limit]
    progress = os.environ.get("CTHR_PROGRESS", "0") == "1"
    for index, task_id in enumerate(task_ids, start=1):
        if progress:
            print(f"[{model_name}] {index}/{len(task_ids)} {task_id}", flush=True)
        grounding_task = dict(algorithm_inputs[task_id])
        query = fullkg.prepare_query(grounding_task, scenario_models[task_id])
        reference = references[task_id]
        task_templates_by_rule, _task_extension_stats = build_backend_complete_templates(
            {task_id: reference},
            {task_id: grounding_task},
            {},
            canonical_to_model,
        )
        query["_compiled_rule_constraint_templates_by_id"] = task_templates_by_rule
        feasible = fullkg.reference_feasible(reference, query)
        raw_reference_ids = fullkg.reference_rule_ids(reference)
        model_reference_ids = model_reference_ids_for(raw_reference_ids, rule_by_id, canonical_to_model)
        candidate_rules = [rule_by_id[rule_id] for rule_id in model_reference_ids if rule_id in rule_by_id]
        start = time.perf_counter()
        if candidate_rules:
            constraints = fullkg.method_constraints(query, model_reference_ids, rule_by_id)
            optimized_x = fast_backend_optimize(query, constraints)
            formal_feasible = (
                fullkg.base.constraints_satisfied(
                    constraints,
                    fullkg.base.with_query_values(query, optimized_x),
                )
                if optimized_x is not None
                else False
            )
            result = fullkg.MethodResult(
                True,
                sorted(model_reference_ids),
                optimized_x,
                formal_feasible,
                None,
            )
        else:
            result = fullkg.MethodResult(False, [], None, None, "no_grounded_candidates")
        elapsed = (time.perf_counter() - start) * 1000.0
        predicted_model = sorted(result.predicted_rule_ids) if result.supported else []
        predicted_canonical = backend_complete.project_rule_ids(predicted_model, model_to_canonical)
        reference_canonical = canonical_references.get(task_id, [])
        sem_ok = (
            fullkg.semantic_valid(feasible, result.optimized_x, predicted_model, model_reference_ids)
            if result.supported
            else False
        )
        formal_ok = bool(result.formal_feasible) if result.supported else False
        rows.append(
            {
                "Domain": "aviation_strong_mixed_v11_150",
                "Mode": MODE["mode"],
                "Generator": model_name,
                "Overlay": overlay_key,
                "task_id": task_id,
                "relation_templates_added": extension_stats.get("added_template_count", 0),
                "oracle_candidate_count": len(model_reference_ids),
                "candidate_rule_count_present": len(candidate_rules),
                "predicted_model_rule_ids": predicted_model,
                "model_reference_rule_ids": model_reference_ids,
                "projected_canonical_rule_ids": predicted_canonical,
                "canonical_reference_rule_ids": reference_canonical,
                "model_rule_precision": rule_precision(predicted_model, model_reference_ids),
                "model_rule_recall": rule_recall(predicted_model, model_reference_ids),
                "canonical_rule_precision": rule_precision(predicted_canonical, reference_canonical),
                "canonical_rule_recall": rule_recall(predicted_canonical, reference_canonical),
                "formal_feasible": formal_ok,
                "semantic_valid": sem_ok,
                "false_accept": bool(formal_ok and not sem_ok),
                "invalid_case": bool(not sem_ok),
                "unsupported_reason": "" if result.supported else result.unsupported_reason,
                "runtime_ms": round(elapsed, 3),
            }
        )
    return aggregate_rows(rows, model_name), rows


def main() -> None:
    quality_rows = [quality_row(model_spec) for model_spec in MODEL_SPECS]
    canonical_references = canonical_reference_by_task()
    downstream_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    alignment_summaries = {
        str(model_spec["overlay_key"]): overlay_alignment_summary(str(model_spec["overlay_key"]))
        for model_spec in MODEL_SPECS
    }
    for model_spec in MODEL_SPECS:
        aggregate, rows = evaluate_model(model_spec, canonical_references)
        downstream_rows.append(aggregate)
        task_rows.extend(rows)

    quality_headers = [
        "Domain",
        "Generator",
        "Rules",
        "Provenance valid",
        "Constraint grounding",
        "Relation grounding",
    ]
    downstream_headers = [
        "Domain",
        "Generator",
        "Rule Precision",
        "Rule Recall",
        "Formal CSR",
        "Sem-CSR",
        "False accept",
        "Candidate zero",
        "CTHR no valid",
        "Invalid cases",
        "Unsupported tasks",
        "Relation templates added",
    ]
    task_headers = [
        "Domain",
        "Mode",
        "Generator",
        "Overlay",
        "task_id",
        "relation_templates_added",
        "oracle_candidate_count",
        "candidate_rule_count_present",
        "predicted_model_rule_ids",
        "model_reference_rule_ids",
        "projected_canonical_rule_ids",
        "canonical_reference_rule_ids",
        "model_rule_precision",
        "model_rule_recall",
        "canonical_rule_precision",
        "canonical_rule_recall",
        "formal_feasible",
        "semantic_valid",
        "false_accept",
        "invalid_case",
        "unsupported_reason",
        "runtime_ms",
    ]

    quality_csv = RESULTS_DIR / "section_6_4_aviation_strong_mixed_v11_150_library_table.csv"
    downstream_csv = RESULTS_DIR / "section_6_4_aviation_strong_mixed_v11_150_downstream_table.csv"
    task_csv = RESULTS_DIR / "section_6_4_aviation_strong_mixed_v11_150_task_rows.csv"
    report_path = RESULTS_DIR / "section_6_4_aviation_strong_mixed_v11_150_report.md"
    summary_path = RESULTS_DIR / "section_6_4_aviation_strong_mixed_v11_150_summary.json"

    write_csv(quality_csv, quality_rows, quality_headers)
    write_csv(downstream_csv, downstream_rows, downstream_headers)
    write_csv(task_csv, task_rows, task_headers)
    write_json(
        summary_path,
        {
            "dataset": str(DATASET_ROOT),
            "mode": MODE,
            "alignment_summaries": alignment_summaries,
            "quality_rows": quality_rows,
            "downstream_rows": downstream_rows,
            "outputs": {
                "quality_table": str(quality_csv),
                "downstream_table": str(downstream_csv),
                "task_rows": str(task_csv),
                "report": str(report_path),
            },
        },
    )

    report = "\n\n".join(
        [
            "# Section 6.4 Aviation Strong Mixed v11 150 Rule-Library Comparison",
            "The first table reports raw LLM-generated rule-library quality. The second table evaluates downstream replacement under a controlled setting where strong-aligned valid rules are supplied and rule compilation is handled by the backend.",
            "## Table 1. Rule-library generation quality",
            markdown_table(quality_rows, quality_headers),
            "## Table 2. Downstream replacement performance",
            markdown_table(downstream_rows, downstream_headers),
        ]
    )
    report_path.write_text(report, encoding="utf-8")

    print(f"Wrote {quality_csv}")
    print(f"Wrote {downstream_csv}")
    print(f"Wrote {task_csv}")
    print(f"Wrote {report_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
