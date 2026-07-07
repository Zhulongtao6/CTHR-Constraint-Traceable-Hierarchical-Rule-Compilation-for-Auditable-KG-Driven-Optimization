from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from baselines.asp_rule_structure import retrieve_candidate_rules, relation_target, relation_type
from baselines.cthr_rule_resolver import (
    is_initially_applicable,
    relation_maps,
    remove_rules_with_missing_dependencies,
    dependency_closure,
    resolve_conflicts,
)


THIS_DIR = Path(__file__).resolve().parent
CTHR_ROOT = THIS_DIR.parents[1]
PAPER_DIR = CTHR_ROOT / "paper"
SUPPORT_DIR = CTHR_ROOT / "submission_support" / "kbs_systematic_experiments_v1"
RESULTS_DIR = PAPER_DIR / "results"

AVIATION_RULE_LIBRARY_PATH = (
    PAPER_DIR
    / "full_aviation_kg_rule_library_model_comparison"
    / "full_aviation_rule_library_qwen.json"
)
ARCHITECTURE_RULE_LIBRARY_PATH = (
    SUPPORT_DIR
    / "results"
    / "kg_to_rule_library"
    / "architecture"
    / "full_architecture_rule_library_qwen.json"
)

AVIATION_TASK_DIR = SUPPORT_DIR / "datasets" / "aviation_combined" / "tasks"
ARCHITECTURE_TASK_DIR = SUPPORT_DIR / "datasets" / "architecture" / "tasks"

OUT_OVERALL_CSV = RESULTS_DIR / "section_6_6_ablation_overall.csv"
OUT_OVERALL_MD = RESULTS_DIR / "section_6_6_ablation_overall.md"
OUT_OVERALL_JSON = RESULTS_DIR / "section_6_6_ablation_overall.json"
OUT_PER_TASK_CSV = RESULTS_DIR / "section_6_6_ablation_per_task.csv"
OUT_REPORT_MD = RESULTS_DIR / "section_6_6_ablation_report.md"

VARIANTS: dict[str, set[str]] = {
    "CTHR full": set(),
    "w/o applicability": {"applicability"},
    "w/o dependency": {"dependency"},
    "w/o exclusion": {"exclusion"},
    "w/o override": {"override"},
    "w/o precedence": {"precedence"},
    "w/o parameter propagation": {"parameter_propagation"},
    "w/o cell decomposition": {"cell_decomposition"},
}

COMPARATOR_RE = re.compile(r"(<=|>=|!=|==|=|<|>)")
SAFE_GLOBALS = {"__builtins__": {}}
SAFE_FUNCS = {"abs": abs, "min": min, "max": max, "tan": math.tan, "sqrt": math.sqrt, "math": math}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def csv_cell(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple, set)):
        return json.dumps(list(value) if isinstance(value, set) else value, ensure_ascii=False)
    if value is None:
        return ""
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_cell(row.get(field)) for field in fields})


def pct(numerator: float, denominator: float) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


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


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def vector_to_env(task: dict[str, Any], x: dict[str, float]) -> dict[str, Any]:
    env = {key: value for key, value in task.get("scenario_facts", {}).items() if is_number(value)}
    env.update(x)
    return env


def constraint_violation(constraint: dict[str, Any], env: dict[str, Any]) -> float:
    expression = constraint.get("checker_expression") or constraint.get("expression")
    if not expression:
        return 0.0
    parsed = split_comparator(str(expression))
    if parsed is None:
        return 0.0
    lhs_s, op, rhs_s = parsed
    try:
        lhs = eval_expr(lhs_s, env)
        rhs = eval_expr(rhs_s, env)
    except Exception:
        return 1.0
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


def satisfies_constraints(constraints: list[dict[str, Any]], task: dict[str, Any], x: dict[str, float], tol: float = 1e-3) -> bool:
    env = vector_to_env(task, x)
    return all(constraint_violation(constraint, env) <= tol for constraint in constraints)


def reference_constraints(wrapper: dict[str, Any], task: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hidden = wrapper.get("hidden_reference", {})
    base = hidden.get("solver_constraints")
    cells = hidden.get("solver_constraint_cells")
    if base is None:
        base = task.get("solver_constraints", [])
    if cells is None:
        cells = task.get("solver_constraint_cells", [])
    return list(base or []), list(cells or [])


def source_reference_accept(wrapper: dict[str, Any], task: dict[str, Any], x: dict[str, float]) -> bool:
    base, cells = reference_constraints(wrapper, task)
    if not satisfies_constraints(base, task, x):
        return False
    if not cells:
        return True
    for cell in cells:
        constraints = cell.get("constraints") or cell.get("executable_constraints") or []
        if satisfies_constraints(list(constraints), task, x):
            return True
    return False


def sanitize_task_for_method(task: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "omega_id",
        "title",
        "domain",
        "source_domain",
        "task_type",
        "engineering_task",
        "design_intent",
        "scenario_facts",
        "decision_variables",
        "objectives",
        "query_preferences",
        "preference_weights",
    }
    return {key: task.get(key) for key in allowed if key in task}


def scenario_for_runtime(task: dict[str, Any]) -> dict[str, Any]:
    scenario = dict(task.get("scenario_facts", {}))
    scenario.update(
        {
            "domain": task.get("domain"),
            "task_type": task.get("task_type"),
            "title": task.get("title"),
        }
    )
    scenario["decision_variable_names"] = list(task.get("decision_variables", {}).keys())
    return scenario


def reference_rule_ids(wrapper: dict[str, Any], task: dict[str, Any]) -> list[str]:
    metadata = task.get("stress_metadata") or {}
    ids = metadata.get("final_valid_rule_ids_expected_for_evaluation")
    if ids:
        return sorted(map(str, ids))
    hidden = wrapper.get("hidden_reference", {})
    label = hidden.get("rule_structure_label", {}) or wrapper.get("rule_structure_label", {})
    ids = label.get("expected_surviving_rule_ids") or label.get("expected_source_rule_ids") or []
    return sorted(map(str, ids))


def reference_structures(wrapper: dict[str, Any], task: dict[str, Any]) -> list[list[str]]:
    metadata = task.get("stress_metadata") or {}
    structures = metadata.get("valid_rule_structures_expected")
    if structures:
        return [sorted(map(str, item)) for item in structures]
    hidden = wrapper.get("hidden_reference", {})
    label = hidden.get("rule_structure_label", {}) or wrapper.get("rule_structure_label", {})
    structures = label.get("expected_valid_rule_structures")
    if structures:
        return [sorted(map(str, item)) for item in structures]
    ids = reference_rule_ids(wrapper, task)
    return [ids] if ids else []


def target_interaction(task: dict[str, Any]) -> str:
    metadata = task.get("stress_metadata") or {}
    value = metadata.get("target_interaction") or metadata.get("challenge_types") or task.get("task_type", "")
    if isinstance(value, list):
        return "; ".join(map(str, value))
    return str(value)


def load_tasks(task_dir: Path, pattern: str) -> list[dict[str, Any]]:
    wrappers: list[dict[str, Any]] = []
    for path in sorted(task_dir.glob(pattern)):
        payload = read_json(path)
        task = payload.get("task", payload)
        payload["_task_path"] = str(path)
        payload["_task_id"] = task.get("omega_id")
        wrappers.append(payload)
    return wrappers


def relation_maps_for_ablation(candidate_rules: list[dict[str, Any]], disabled: set[str]) -> dict[str, set[tuple[str, str]]]:
    maps = relation_maps(candidate_rules)
    if "dependency" in disabled:
        maps["depends"] = set()
    elif "parameter_propagation" in disabled:
        filtered: set[tuple[str, str]] = set()
        by_id = {str(rule.get("rule_id")): rule for rule in candidate_rules if rule.get("rule_id")}
        for source, target in maps["depends"]:
            source_rule = by_id.get(source, {})
            keep = True
            for relation in source_rule.get("relations", []):
                if relation_target(relation) == target and relation_type(relation) in {"uses_parameter"}:
                    keep = False
            if keep:
                filtered.add((source, target))
        maps["depends"] = filtered
    if "exclusion" in disabled:
        maps["excludes"] = set()
        maps["conflicts"] = set()
    if "override" in disabled:
        maps["overrides"] = set()
    if "precedence" in disabled:
        maps["precedes"] = set()
    return maps


def resolve_variant(candidate_rules: list[dict[str, Any]], scenario: dict[str, Any], disabled: set[str]) -> tuple[list[str], str]:
    by_id = {str(rule["rule_id"]): rule for rule in candidate_rules if rule.get("rule_id")}
    if not by_id:
        return [], "no_candidate_rules"
    maps = relation_maps_for_ablation(candidate_rules, disabled)
    if "applicability" in disabled:
        applicable = set(by_id)
    else:
        applicable = {rid for rid, rule in by_id.items() if is_initially_applicable(rule, scenario)}
    defeated: set[str] = set()
    for source, target in maps["overrides"]:
        if source in applicable:
            defeated.add(target)
    for source, target in maps["precedes"]:
        if source in applicable and target in applicable:
            defeated.add(target)
    available = applicable - defeated
    available = remove_rules_with_missing_dependencies(available, maps["depends"], set(by_id) - defeated)
    selected = dependency_closure(available, maps["depends"], set(by_id) - defeated)
    selected -= defeated
    if "exclusion" not in disabled:
        selected, _notes = resolve_conflicts(selected, maps, by_id, scenario)
    selected = dependency_closure(selected, maps["depends"], set(by_id) - defeated)
    selected -= defeated
    return sorted(selected), "success" if selected else "empty_valid_structure"


def map_rule_constraint_to_task(constraint: dict[str, Any], task: dict[str, Any]) -> dict[str, Any] | None:
    variable = str(constraint.get("variable", ""))
    op = str(constraint.get("op", "")).strip()
    value = constraint.get("value")
    if op not in {"<=", ">=", "<", ">", "=", "=="}:
        return None
    if not isinstance(value, (int, float)):
        return None
    variable_norm = normalize_name(variable)
    for task_var in task.get("decision_variables", {}):
        task_norm = normalize_name(task_var)
        if variable_norm == task_norm or variable_norm in task_norm or task_norm in variable_norm:
            return {
                "constraint_id": str(constraint.get("constraint_id") or constraint.get("source_quote") or variable),
                "expression": f"{task_var} {op} {float(value)}",
                "source_id": constraint.get("source_id"),
                "source_type": "real_rule_library",
            }
    return None


def formal_constraints_from_rules(predicted_rule_ids: list[str], by_id: dict[str, dict[str, Any]], task: dict[str, Any]) -> list[dict[str, Any]]:
    constraints: list[dict[str, Any]] = []
    for rule_id in predicted_rule_ids:
        rule = by_id.get(rule_id)
        if not rule:
            continue
        for constraint in rule.get("constraints", []):
            mapped = map_rule_constraint_to_task(constraint, task)
            if mapped is not None:
                constraints.append(mapped)
    return constraints


def construct_solution(task: dict[str, Any], constraints: list[dict[str, Any]]) -> tuple[bool, dict[str, float], str]:
    x: dict[str, float] = {}
    for name, spec in task.get("decision_variables", {}).items():
        lo = float(spec.get("lower", 0.0))
        hi = float(spec.get("upper", lo))
        x[name] = (lo + hi) / 2.0
    for _ in range(6):
        changed = False
        env = vector_to_env(task, x)
        for constraint in constraints:
            parsed = split_comparator(str(constraint.get("expression", "")))
            if not parsed:
                continue
            lhs_s, op, rhs_s = parsed
            lhs_var = lhs_s.strip()
            if lhs_var not in x:
                continue
            try:
                rhs = eval_expr(rhs_s, env)
            except Exception:
                continue
            spec = task["decision_variables"][lhs_var]
            lo = float(spec.get("lower", rhs))
            hi = float(spec.get("upper", rhs))
            old = x[lhs_var]
            if op in {">=", ">"}:
                x[lhs_var] = min(max(x[lhs_var], rhs), hi)
            elif op in {"<=", "<"}:
                x[lhs_var] = max(min(x[lhs_var], rhs), lo)
            elif op in {"=", "=="}:
                x[lhs_var] = min(max(rhs, lo), hi)
            changed = changed or abs(old - x[lhs_var]) > 1e-9
        if not changed:
            break
    formal_feasible = satisfies_constraints(constraints, task, x)
    if not formal_feasible:
        return False, x, "formal_constraints_infeasible_or_unmapped"
    return True, x, ""


def rule_metrics(predicted: list[str], reference: list[str]) -> tuple[float, float]:
    pred_set = set(predicted)
    ref_set = set(reference)
    inter = pred_set & ref_set
    precision = len(inter) / len(pred_set) if pred_set else (1.0 if not ref_set else 0.0)
    recall = len(inter) / len(ref_set) if ref_set else 1.0
    return precision, recall


def run_dataset(
    dataset_name: str,
    wrappers: list[dict[str, Any]],
    rule_library: dict[str, Any],
    min_score: float,
) -> list[dict[str, Any]]:
    by_id = {str(rule["rule_id"]): rule for rule in rule_library.get("rules", []) if rule.get("rule_id")}
    rows: list[dict[str, Any]] = []
    for wrapper in wrappers:
        raw_task = wrapper.get("task", wrapper)
        task = sanitize_task_for_method(raw_task)
        scenario = scenario_for_runtime(task)
        candidate_ids = retrieve_candidate_rules(rule_library, task, min_score=min_score)
        candidate_rules = [by_id[rule_id] for rule_id in candidate_ids if rule_id in by_id]
        reference_ids = reference_rule_ids(wrapper, raw_task)
        for variant, disabled in VARIANTS.items():
            predicted_ids, status = resolve_variant(candidate_rules, scenario, disabled)
            formal_constraints = formal_constraints_from_rules(predicted_ids, by_id, task)
            solved, solution, solve_reason = construct_solution(task, formal_constraints)
            formal_feasible = bool(solved and satisfies_constraints(formal_constraints, task, solution))
            semantic_valid = bool(solved and source_reference_accept(wrapper, raw_task, solution))
            precision, recall = rule_metrics(predicted_ids, reference_ids)
            reasons = []
            if status != "success":
                reasons.append(status)
            if not candidate_ids:
                reasons.append("grounding_returned_no_candidates")
            if not formal_constraints:
                reasons.append("no_numeric_constraints_mapped_from_real_rule_library")
            if solve_reason:
                reasons.append(solve_reason)
            ref_missing = [rule_id for rule_id in reference_ids if rule_id not in by_id]
            if ref_missing:
                reasons.append(f"reference_rule_ids_not_in_real_rule_library:{len(ref_missing)}")
            if variant == "w/o cell decomposition":
                reasons.append("cell_decomposition_is_evaluated_as_rule-selection-equivalent_without_hidden_cells")
            rows.append(
                {
                    "Dataset": dataset_name,
                    "task_id": raw_task.get("omega_id"),
                    "target_interaction": target_interaction(raw_task),
                    "Variant": variant,
                    "solved": solved,
                    "predicted_rule_ids": predicted_ids,
                    "reference_rule_ids": reference_ids,
                    "rule_precision": precision,
                    "rule_recall": recall,
                    "formal_feasible": formal_feasible,
                    "semantic_valid": semantic_valid,
                    "unsupported_reason": "; ".join(dict.fromkeys(reasons)),
                }
            )
    return rows


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    overall: list[dict[str, Any]] = []
    for dataset in ["Aviation", "Architecture"]:
        dataset_rows = [row for row in rows if row["Dataset"] == dataset]
        total = len({row["task_id"] for row in dataset_rows})
        for variant in VARIANTS:
            group = [row for row in dataset_rows if row["Variant"] == variant]
            solved = sum(bool(row["solved"]) for row in group)
            formal = sum(bool(row["formal_feasible"]) for row in group)
            sem = sum(bool(row["semantic_valid"]) for row in group)
            overall.append(
                {
                    "Dataset": dataset,
                    "Variant": variant,
                    "Solved": f"{solved}/{total}",
                    "Formal CSR": round(pct(formal, total), 1),
                    "Sem-CSR": round(pct(sem, total), 1),
                    "Rule Precision": round(100.0 * mean([float(row["rule_precision"]) for row in group]), 1),
                    "Rule Recall": round(100.0 * mean([float(row["rule_recall"]) for row in group]), 1),
                }
            )
    return overall


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines)


def write_report(overall: list[dict[str, Any]], per_task: list[dict[str, Any]]) -> None:
    def failing_tasks(dataset: str, variant: str) -> list[str]:
        return [
            str(row["task_id"])
            for row in per_task
            if row["Dataset"] == dataset and row["Variant"] == variant and float(row["rule_recall"]) < 1.0
        ]

    lines = [
        "# Section 6.6 Ablation Experiment",
        "",
        "This run compares CTHR full with six rule-interaction ablations on the Aviation and Architecture benchmarks.",
        "",
        "Input discipline:",
        "",
        "- Aviation uses the real full aviation rule library: `full_aviation_rule_library_qwen.json`.",
        "- The aviation stress combined rule library and stress-derived extension library are not used as method input.",
        "- Architecture uses `full_architecture_rule_library_qwen.json`.",
        "- Hidden reference labels, expected candidates, expected valid rules, solver constraints, solver cells, certificate targets, and semantic labels are used only for evaluation.",
        "",
        "Overall results:",
        "",
        markdown_table(overall, ["Dataset", "Variant", "Solved", "Formal CSR", "Sem-CSR", "Rule Precision", "Rule Recall"]),
        "",
        "Ablation failure notes:",
        "",
    ]
    for dataset in ["Aviation", "Architecture"]:
        lines.append(f"## {dataset}")
        for variant in VARIANTS:
            if variant == "CTHR full":
                continue
            tasks = failing_tasks(dataset, variant)
            sample = ", ".join(tasks[:10]) if tasks else "none"
            lines.append(f"- `{variant}` rule-recall failures: {len(tasks)} tasks; sample: {sample}.")
        if dataset == "Architecture":
            missing_refs = [
                row
                for row in per_task
                if row["Dataset"] == dataset
                and "reference_rule_ids_not_in_real_rule_library" in str(row.get("unsupported_reason", ""))
            ]
            lines.append(
                f"- Architecture benchmark reference IDs are in benchmark-template namespace for {len(missing_refs)} per-task rows; "
                "the method input remains the real Qwen architecture rule library as required."
            )
    lines.extend(
        [
            "",
            "Cell-decomposition note:",
            "",
            "`w/o cell decomposition` is included as a backend compilation ablation, not one of the six rule-interaction mechanisms. "
            "Because this run forbids hidden solver cells as method input and uses the real LLM rule libraries, it is evaluated as rule-selection-equivalent with an explicit per-task unsupported note where executable cells cannot be reconstructed from visible rule records.",
        ]
    )
    OUT_REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    aviation_library = read_json(AVIATION_RULE_LIBRARY_PATH)
    architecture_library = read_json(ARCHITECTURE_RULE_LIBRARY_PATH)
    aviation_tasks = load_tasks(AVIATION_TASK_DIR, "AVI_*.json")
    architecture_tasks = load_tasks(ARCHITECTURE_TASK_DIR, "ARCH_OPT_*.json")
    if len(aviation_tasks) != 31:
        raise RuntimeError(f"Expected 31 aviation tasks, found {len(aviation_tasks)}")
    if len(architecture_tasks) != 30:
        raise RuntimeError(f"Expected 30 architecture tasks, found {len(architecture_tasks)}")

    per_task = []
    per_task.extend(run_dataset("Aviation", aviation_tasks, aviation_library, min_score=2.0))
    per_task.extend(run_dataset("Architecture", architecture_tasks, architecture_library, min_score=2.0))
    overall = aggregate(per_task)

    overall_fields = ["Dataset", "Variant", "Solved", "Formal CSR", "Sem-CSR", "Rule Precision", "Rule Recall"]
    per_task_fields = [
        "Dataset",
        "task_id",
        "target_interaction",
        "Variant",
        "solved",
        "predicted_rule_ids",
        "reference_rule_ids",
        "rule_precision",
        "rule_recall",
        "formal_feasible",
        "semantic_valid",
        "unsupported_reason",
    ]
    write_csv(OUT_OVERALL_CSV, overall, overall_fields)
    OUT_OVERALL_MD.write_text(markdown_table(overall, overall_fields) + "\n", encoding="utf-8")
    write_json(OUT_OVERALL_JSON, overall)
    write_csv(OUT_PER_TASK_CSV, per_task, per_task_fields)
    write_report(overall, per_task)
    print(json.dumps({"overall": overall, "per_task_rows": len(per_task)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
