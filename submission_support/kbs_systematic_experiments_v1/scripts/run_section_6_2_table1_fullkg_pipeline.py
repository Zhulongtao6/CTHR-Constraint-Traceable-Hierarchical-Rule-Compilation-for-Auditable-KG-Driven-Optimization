from __future__ import annotations

import csv
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CTHR_ROOT = ROOT.parents[1]
RESULTS_DIR = ROOT / "results"
SCRIPTS_DIR = ROOT / "scripts"

sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(CTHR_ROOT))

import run_section_6_2_table1_pipeline as base  # noqa: E402
import run_section_6_3_aviation_candidate_to_valid as aviation_grounding  # noqa: E402
import run_section_6_3_architecture_candidate_to_valid as architecture_grounding  # noqa: E402
import run_section_6_3_candidate_to_valid as ctv  # noqa: E402
from experiments.kg_to_rule_validation.baselines.asp_rule_structure import (  # noqa: E402
    enumerate_rule_structures,
    eval_guard,
)
from experiments.kg_to_rule_validation.baselines.smt_monolithic import (  # noqa: E402
    build_smt_formula,
    map_rule_variable,
    optimize_with_z3,
)


METHOD_SPECS = [
    ("Flat baseline", "flat"),
    ("Native ASP + clingo", "native_symbolic"),
    ("Native SMT + Z3", "native_symbolic"),
    ("Native MILP + HiGHS", "native_symbolic"),
    ("CTHR default", "cthr_semantic_modeling"),
    ("CTHR-style ASP + clingo", "cthr_semantic_modeling"),
    ("CTHR-style SMT + Z3", "cthr_semantic_modeling"),
    ("CTHR-style MILP + HiGHS", "cthr_semantic_modeling"),
]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    domain: str
    root: Path
    algorithm_inputs: Path
    scenario_models: Path
    evaluation_references: Path
    rule_library: Path
    grounding_full: Path
    constraint_templates: Path


DATASETS = [
    DatasetSpec(
        name="Aviation",
        domain="aviation",
        root=ROOT / "datasets" / "aviation_fullkg_clean",
        algorithm_inputs=ROOT
        / "datasets"
        / "aviation_fullkg_clean"
        / "algorithm_inputs"
        / "aviation_algorithm_inputs.json",
        scenario_models=ROOT
        / "datasets"
        / "aviation_fullkg_clean"
        / "scenario_models"
        / "aviation_public_scenario_models.json",
        evaluation_references=ROOT
        / "datasets"
        / "aviation_fullkg_clean"
        / "evaluation_references"
        / "aviation_evaluation_references.json",
        rule_library=ROOT
        / "datasets"
        / "aviation_fullkg_clean"
        / "rule_libraries"
        / "full_aviation_rule_library_qwen.json",
        grounding_full=RESULTS_DIR / "section_6_3_aviation_latest_regen_20260527_candidate_to_valid_full.json",
        constraint_templates=RESULTS_DIR
        / "constraint_templates"
        / "aviation_fullkg_clean"
        / "compiled_rule_constraint_templates.json",
    ),
    DatasetSpec(
        name="Architecture",
        domain="architecture",
        root=ROOT / "datasets" / "architecture_fullkg_clean",
        algorithm_inputs=ROOT
        / "datasets"
        / "architecture_fullkg_clean"
        / "algorithm_inputs"
        / "architecture_algorithm_inputs.json",
        scenario_models=ROOT
        / "datasets"
        / "architecture_fullkg_clean"
        / "scenario_models"
        / "architecture_public_scenario_models.json",
        evaluation_references=ROOT
        / "datasets"
        / "architecture_fullkg_clean"
        / "evaluation_references"
        / "architecture_evaluation_references.json",
        rule_library=ROOT
        / "datasets"
        / "architecture_fullkg_clean"
        / "rule_libraries"
        / "full_architecture_rule_library_qwen.json",
        grounding_full=RESULTS_DIR
        / "section_6_3_architecture_latest_regen_20260527_candidate_to_valid_full.json",
        constraint_templates=RESULTS_DIR
        / "constraint_templates"
        / "architecture_fullkg_clean"
        / "compiled_rule_constraint_templates.json",
    ),
]


@dataclass
class MethodResult:
    supported: bool
    predicted_rule_ids: list[str]
    optimized_x: dict[str, float] | None
    formal_feasible: bool | None
    unsupported_reason: str = ""


def read_json(path: Path) -> dict[str, Any]:
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


def item_map(path: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(path)
    items = payload.get("items", [])
    return {str(item["omega_id"]): item for item in items}


def grounding_result_map(path: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(path)
    rows = payload.get("rows", payload) if isinstance(payload, dict) else payload
    return {str(row["task_id"]): row for row in rows}


def constraint_template_map(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        return {}
    payload = read_json(path)
    templates_by_rule = payload.get("templates_by_rule", {})
    return {
        str(rule_id): [dict(template) for template in templates]
        for rule_id, templates in templates_by_rule.items()
        if isinstance(templates, list)
    }


def public_constraints(model: dict[str, Any]) -> list[dict[str, Any]]:
    constraints = model.get("executable_constraints")
    if constraints is None:
        constraints = model.get("constraints", [])
    return [constraint for constraint in constraints if constraint.get("executable", True)]


def prepare_query(algorithm_input: dict[str, Any], scenario_model: dict[str, Any]) -> dict[str, Any]:
    query = dict(algorithm_input)
    query["solver_constraints"] = public_constraints(scenario_model)
    return query


def reference_rule_ids(reference: dict[str, Any]) -> list[str]:
    structure = reference.get("rule_structure", {})
    return sorted(str(rule_id) for rule_id in structure.get("expected_surviving_rule_ids", []))


def ids_from_grounding(row: dict[str, Any], key: str) -> list[str]:
    value = row.get(key, [])
    if isinstance(value, str):
        value = json.loads(value) if value.strip() else []
    return sorted(str(item) for item in value)


def reference_feasible(reference: dict[str, Any], query: dict[str, Any]) -> dict[str, Any]:
    feasible = dict(reference.get("feasible_region", {}))
    feasible["scenario_facts"] = dict(query.get("scenario_facts", {}))
    return feasible


def target_interaction(reference: dict[str, Any]) -> str:
    challenge = reference.get("rule_structure", {}).get("challenge_types", [])
    return "; ".join(str(item) for item in challenge)


def normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def variable_tokens(value: str) -> set[str]:
    tokens = {token for token in re.split(r"[^a-z0-9]+", str(value).lower()) if token}
    return tokens - {"maximum", "minimum", "max", "min", "required", "requirement", "design", "selected"}


def fallback_map_rule_variable(variable: str, query: dict[str, Any]) -> str | None:
    mapped = map_rule_variable(variable, query)
    if mapped:
        return mapped
    source = variable_tokens(variable)
    best_name = None
    best_score = 0.0
    for name in query.get("decision_variables", {}):
        target = variable_tokens(name)
        if not source or not target:
            continue
        overlap = len(source & target)
        containment = 1.0 if normalize_token(variable) in normalize_token(name) else 0.0
        score = overlap + containment
        if score > best_score:
            best_name = name
            best_score = score
    return best_name if best_score >= 2.0 else None


def numeric_constraint_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"[-+]?[0-9]+(?:\.[0-9]+)?", str(value))
    return float(match.group(0)) if match else None


def compatible_unit_value(value: float, source_unit: str, target_unit: str) -> float:
    source = source_unit.lower()
    target = target_unit.lower()
    if source in {"mm", "millimeter", "millimeters"} and target in {"inch", "in", "inches"}:
        return value / 25.4
    if source in {"inch", "in", "inches"} and target in {"mm", "millimeter", "millimeters"}:
        return value * 25.4
    if source in {"ft", "foot", "feet"} and target in {"inch", "in", "inches"}:
        return value * 12.0
    if source in {"inch", "in", "inches"} and target in {"ft", "foot", "feet"}:
        return value / 12.0
    if source in {"nm", "nmi", "nautical_mile", "nautical_miles", "nautical mile", "nautical miles"} and target in {
        "km",
        "kilometer",
        "kilometers",
    }:
        return value * 1.852
    if source in {"km", "kilometer", "kilometers"} and target in {
        "nm",
        "nmi",
        "nautical_mile",
        "nautical_miles",
        "nautical mile",
        "nautical miles",
    }:
        return value / 1.852
    if source in {"seconds", "second", "sec"} and target in {"s", "sec", "second", "seconds"}:
        return value
    return value


def rule_constraints(
    rule: dict[str, Any],
    query: dict[str, Any],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    equality_values_by_variable: dict[str, list[float]] = {}
    for idx, constraint in enumerate(rule.get("constraints", [])):
        op = str(constraint.get("op", "")).strip()
        if op not in {"<=", ">=", "<", ">", "=", "=="}:
            continue
        mapped = fallback_map_rule_variable(str(constraint.get("variable", "")), query)
        if not mapped:
            continue
        value = numeric_constraint_value(constraint.get("value"))
        if value is None:
            continue
        target_unit = str(query.get("decision_variables", {}).get(mapped, {}).get("unit", ""))
        source_unit = str(constraint.get("unit", ""))
        value = compatible_unit_value(value, source_unit, target_unit)
        if op in {"=", "=="}:
            previous_values = equality_values_by_variable.setdefault(mapped, [])
            if any(abs(value - previous) <= 0.01 for previous in previous_values):
                continue
            previous_values.append(value)
        out.append(
            {
                "constraint_id": f"rulelib_{rule['rule_id']}_{idx}",
                "expression": f"{mapped} {op} {value}",
                "checker_expression": f"{mapped} {op} {value}",
                "source_type": "rule_library",
                "source_id": str(rule["rule_id"]),
                "executable": True,
            }
        )
    return out


def available_numeric_symbols(query: dict[str, Any]) -> set[str]:
    symbols = set(query.get("decision_variables", {}))
    symbols.update(
        key
        for key, value in query.get("scenario_facts", {}).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )
    return symbols


def scenario_context(query: dict[str, Any]) -> dict[str, Any]:
    context: dict[str, Any] = {}
    for key, value in query.get("scenario_facts", {}).items():
        if isinstance(value, (str, bool, int, float)):
            context[key] = value
    return context


SOFT_CONTEXT_IGNORE_KEYS = {
    "ncarb_scenario_theme",
    "ncarb_adaptation_policy",
    "ncarb_source_summary",
    "scenario_variant",
    "scenario_variant_differentiator",
    "expansion_case_family",
    "facility_type",
    "locking_mechanism",
    "review_context",
    "story_condition",
    "voice_alarm_context",
}


def relaxed_context_value_matches(current: Any, expected: Any) -> bool:
    if current == expected:
        return True
    if isinstance(current, bool) or isinstance(expected, bool):
        return current == expected
    if isinstance(current, (int, float)) or isinstance(expected, (int, float)):
        return True
    current_norm = normalize_token(str(current))
    expected_norm = normalize_token(str(expected))
    return bool(current_norm and expected_norm and (current_norm in expected_norm or expected_norm in current_norm))


def relaxed_context_matches(current: dict[str, Any], context: dict[str, Any]) -> bool:
    checked = 0
    for key, expected in context.items():
        if key in SOFT_CONTEXT_IGNORE_KEYS:
            continue
        if key not in current:
            continue
        checked += 1
        if not relaxed_context_value_matches(current[key], expected):
            return False
    return checked > 0


def context_matches(query: dict[str, Any], template: dict[str, Any]) -> bool:
    contexts = template.get("applicability_contexts", [])
    if not contexts:
        return True
    current = scenario_context(query)
    for context in contexts:
        if not isinstance(context, dict):
            continue
        if all(current.get(key) == value for key, value in context.items()):
            return True
    for context in contexts:
        if isinstance(context, dict) and relaxed_context_matches(current, context):
            return True
    return False


def compiled_template_constraints(rule_id: str, query: dict[str, Any]) -> list[dict[str, Any]]:
    templates_by_rule = query.get("_compiled_rule_constraint_templates_by_id", {})
    templates = templates_by_rule.get(str(rule_id), [])
    if not templates:
        return []
    available = available_numeric_symbols(query)
    out: list[dict[str, Any]] = []
    for index, template in enumerate(templates):
        if not context_matches(query, template):
            continue
        required = set(str(symbol) for symbol in template.get("required_symbols", []))
        if required - available:
            continue
        expression = str(template.get("checker_expression") or template.get("expression") or "")
        if not expression:
            continue
        metadata = dict(template.get("metadata", {}))
        metadata["constraint_template_id"] = template.get("template_id")
        out.append(
            {
                "constraint_id": f"compiled_{rule_id}_{index}",
                "expression": expression,
                "checker_expression": expression,
                "source_type": "rule_library",
                "source_id": str(rule_id),
                "role": template.get("role", "compiled_rule_constraint"),
                "executable": True,
                "expression_language": template.get("expression_language", "python_safe_arithmetic_predicate"),
                "metadata": metadata,
            }
        )
    return out


def method_constraints(
    query: dict[str, Any],
    selected_rule_ids: list[str],
    rule_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    constraints = list(query.get("solver_constraints", []))
    seen = {str(constraint.get("constraint_id")) for constraint in constraints}
    for rule_id in selected_rule_ids:
        rule = rule_by_id.get(rule_id)
        if not rule:
            continue
        compiled = compiled_template_constraints(rule_id, query)
        templates_by_rule = query.get("_compiled_rule_constraint_templates_by_id", {})
        fallback = [] if str(rule_id) in templates_by_rule else rule_constraints(rule, query)
        for constraint in [*compiled, *fallback]:
            key = str(constraint["constraint_id"])
            if key not in seen:
                constraints.append(constraint)
                seen.add(key)
    return constraints


def strip_relations(rule: dict[str, Any]) -> dict[str, Any]:
    stripped = dict(rule)
    stripped["relations"] = []
    stripped.pop("conflict_class", None)
    stripped.pop("conflict_group", None)
    return stripped


def filtered_library(
    rule_library: dict[str, Any],
    candidate_ids: list[str],
    *,
    native: bool,
) -> dict[str, Any]:
    allowed = set(candidate_ids)
    rules = [rule for rule in rule_library.get("rules", []) if str(rule.get("rule_id")) in allowed]
    if native:
        rules = [strip_relations(rule) for rule in rules]
    out = dict(rule_library)
    out["rules"] = rules
    return out


def candidate_rules_for_domain(
    spec: DatasetSpec,
    rule_library: dict[str, Any],
    query: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    if spec.domain == "aviation":
        return aviation_grounding.generate_aviation_candidates(rule_library, query)
    if spec.domain == "architecture":
        return architecture_grounding.generate_architecture_candidates(rule_library, query)
    raise ValueError(spec.domain)


def scenario_for_domain(spec: DatasetSpec, query: dict[str, Any]) -> dict[str, Any]:
    if spec.domain == "aviation":
        return aviation_grounding.aviation_scenario(query)
    if spec.domain == "architecture":
        return architecture_grounding.architecture_scenario(query)
    raise ValueError(spec.domain)


def cthr_default_select(
    spec: DatasetSpec,
    candidate_rules: list[dict[str, Any]],
    grounding_task: dict[str, Any],
) -> list[str]:
    scenario = scenario_for_domain(spec, grounding_task)
    if spec.domain == "aviation":
        resolution_rules = aviation_grounding.aviation_resolution_candidate_rules(candidate_rules, grounding_task)
        result = ctv.cthr_recover_valid_rules(resolution_rules, scenario)
        return sorted(result.predicted_rule_ids)
    if spec.domain == "architecture":
        result = architecture_grounding.cthr_recover_architecture_valid_rules(candidate_rules, scenario)
        return sorted(result.predicted_rule_ids)
    raise ValueError(spec.domain)


def solve_with_default(
    method: str,
    query: dict[str, Any],
    selected: list[str],
    rule_by_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, float] | None, bool]:
    constraints = method_constraints(query, selected, rule_by_id)
    x = base.optimize_default(query, constraints, method, str(query["omega_id"]))
    formal = base.constraints_satisfied(constraints, base.with_query_values(query, x)) if x is not None else False
    return x, formal


def run_flat(
    query: dict[str, Any],
    candidate_ids: list[str],
    rule_by_id: dict[str, dict[str, Any]],
) -> MethodResult:
    if not candidate_ids:
        return MethodResult(False, [], None, None, "no_grounded_candidates")
    selected = sorted(candidate_ids)
    x, formal = solve_with_default("Flat baseline", query, selected, rule_by_id)
    return MethodResult(True, selected, x, formal)


def run_cthr_default(
    spec: DatasetSpec,
    query: dict[str, Any],
    grounding_task: dict[str, Any],
    candidate_rules: list[dict[str, Any]],
    rule_by_id: dict[str, dict[str, Any]],
    selected_valid_ids: list[str] | None = None,
) -> MethodResult:
    if not candidate_rules:
        return MethodResult(False, [], None, None, "no_grounded_candidates")
    selected = sorted(selected_valid_ids) if selected_valid_ids is not None else cthr_default_select(
        spec,
        candidate_rules,
        grounding_task,
    )
    if not selected:
        return MethodResult(False, [], None, None, "cthr_no_valid_rules")
    x, formal = solve_with_default("CTHR default", query, selected, rule_by_id)
    return MethodResult(True, selected, x, formal)


def run_asp(
    spec: DatasetSpec,
    query: dict[str, Any],
    grounding_task: dict[str, Any],
    candidate_ids: list[str],
    rule_library: dict[str, Any],
    rule_by_id: dict[str, dict[str, Any]],
    *,
    native: bool,
) -> MethodResult:
    if not candidate_ids:
        return MethodResult(False, [], None, None, "no_grounded_candidates")
    lib = filtered_library(rule_library, candidate_ids, native=native)
    result = enumerate_rule_structures(
        lib,
        scenario_for_domain(spec, grounding_task),
        str(query["omega_id"]),
        candidate_rule_ids=candidate_ids,
        applicable_rule_ids=None,
        max_answer_sets=100,
    )
    if result.status != "success":
        return MethodResult(False, [], None, None, f"asp_{result.status}:{result.error or ''}".strip(":"))
    selected = base.union_structures(result.asp_rule_structures)
    x, formal = solve_with_default(
        "Native ASP + clingo" if native else "CTHR-style ASP + clingo",
        query,
        selected,
        rule_by_id,
    )
    return MethodResult(True, selected, x, formal)


def run_smt(
    query: dict[str, Any],
    candidate_ids: list[str],
    rule_library: dict[str, Any],
    rule_by_id: dict[str, dict[str, Any]],
    *,
    native: bool,
) -> MethodResult:
    if not candidate_ids:
        return MethodResult(False, [], None, None, "no_grounded_candidates")
    lib = filtered_library(rule_library, candidate_ids, native=native)
    try:
        formula = build_smt_formula(
            lib,
            query,
            candidate_rule_ids=candidate_ids,
            include_visible_task_constraints=True,
        )
        result = optimize_with_z3(formula, query, timeout_ms=10000)
        if result.status != "sat" or result.optimized_x is None:
            return MethodResult(
                False,
                sorted(result.selected_rule_ids),
                None,
                None,
                f"smt_{result.status}:{result.error or ''}".strip(":"),
            )
        variables = list(query.get("decision_variables", {}))
        x = {name: float(value) for name, value in zip(variables, result.optimized_x)}
        selected = sorted(result.selected_rule_ids)
        constraints = method_constraints(query, selected, rule_by_id)
        formal = base.constraints_satisfied(constraints, base.with_query_values(query, x))
        return MethodResult(True, selected, x, formal)
    except Exception as exc:  # noqa: BLE001
        return MethodResult(False, [], None, None, f"smt_error:{exc}")


def select_native_applicable(
    spec: DatasetSpec,
    grounding_task: dict[str, Any],
    candidate_ids: list[str],
    rule_by_id: dict[str, dict[str, Any]],
) -> list[str]:
    scenario = scenario_for_domain(spec, grounding_task)
    selected: list[str] = []
    for rule_id in candidate_ids:
        rule = rule_by_id.get(rule_id, {})
        guard = rule.get("guard")
        if not guard:
            selected.append(rule_id)
            continue
        try:
            if eval_guard(guard, scenario):
                selected.append(rule_id)
        except Exception:
            continue
    return sorted(selected or candidate_ids[:1])


def run_milp(
    spec: DatasetSpec,
    query: dict[str, Any],
    grounding_task: dict[str, Any],
    candidate_ids: list[str],
    rule_library: dict[str, Any],
    rule_by_id: dict[str, dict[str, Any]],
    *,
    native: bool,
) -> MethodResult:
    if not candidate_ids:
        return MethodResult(False, [], None, None, "no_grounded_candidates")
    try:
        lib = filtered_library(rule_library, candidate_ids, native=native)
        lib_by_id = {str(rule["rule_id"]): rule for rule in lib.get("rules", []) if rule.get("rule_id")}
        if native:
            selected = select_native_applicable(spec, grounding_task, candidate_ids, lib_by_id)
        else:
            selected = base.milp_select_rules(candidate_ids, lib_by_id, scenario_for_domain(spec, grounding_task))
        if not selected:
            return MethodResult(False, [], None, None, "milp_rule_selection_infeasible")
        constraints = method_constraints(query, selected, rule_by_id)
        variables = list(query.get("decision_variables", {}))
        constants = {
            key: float(value)
            for key, value in query.get("scenario_facts", {}).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        rows, lb, ub = base.linear_constraint_rows(constraints, variables, constants)
        c = base.objective_linear(query, variables, constants)
        bounds = [
            (
                float(query["decision_variables"][name].get("lower", 0.0)),
                float(query["decision_variables"][name].get("upper", 1.0)),
            )
            for name in variables
        ]
        a_ub = [row for row, lo, hi in zip(rows, lb, ub) if math.isinf(lo) and not math.isinf(hi)]
        b_ub = [hi for lo, hi in zip(lb, ub) if math.isinf(lo) and not math.isinf(hi)]
        a_eq = [
            row
            for row, lo, hi in zip(rows, lb, ub)
            if not math.isinf(lo) and not math.isinf(hi) and abs(lo - hi) < 1e-9
        ]
        b_eq = [lo for lo, hi in zip(lb, ub) if not math.isinf(lo) and not math.isinf(hi) and abs(lo - hi) < 1e-9]
        lp_result = base.linprog(
            c=np.array(c),
            A_ub=np.array(a_ub) if a_ub else None,
            b_ub=np.array(b_ub) if b_ub else None,
            A_eq=np.array(a_eq) if a_eq else None,
            b_eq=np.array(b_eq) if b_eq else None,
            bounds=bounds,
            method="highs",
        )
        if not lp_result.success or lp_result.x is None:
            return MethodResult(False, selected, None, None, f"highs_{lp_result.message}")
        x = {name: float(value) for name, value in zip(variables, lp_result.x)}
        formal = base.constraints_satisfied(constraints, base.with_query_values(query, x))
        return MethodResult(True, selected, x, formal)
    except Exception as exc:  # noqa: BLE001
        return MethodResult(False, [], None, None, f"unsupported_nonlinear_or_mapping:{exc}")


def semantic_valid(
    feasible: dict[str, Any],
    x: dict[str, float] | None,
    predicted_rule_ids: list[str],
    reference_ids: list[str],
) -> bool:
    if x is None:
        return False
    _ = predicted_rule_ids, reference_ids
    return bool(base.constraints_satisfied(base.reference_constraints(feasible), base.with_scenario_values(feasible, x)))


def run_method(
    spec: DatasetSpec,
    method: str,
    query: dict[str, Any],
    grounding_task: dict[str, Any],
    candidate_rules: list[dict[str, Any]],
    rule_library: dict[str, Any],
    rule_by_id: dict[str, dict[str, Any]],
    cthr_valid_ids: list[str] | None = None,
) -> MethodResult:
    candidate_ids = sorted(str(rule["rule_id"]) for rule in candidate_rules)
    if method == "Flat baseline":
        return run_flat(query, candidate_ids, rule_by_id)
    if method == "CTHR default":
        return run_cthr_default(spec, query, grounding_task, candidate_rules, rule_by_id, cthr_valid_ids)
    if method == "Native ASP + clingo":
        return run_asp(spec, query, grounding_task, candidate_ids, rule_library, rule_by_id, native=True)
    if method == "CTHR-style ASP + clingo":
        return run_asp(spec, query, grounding_task, candidate_ids, rule_library, rule_by_id, native=False)
    if method == "Native SMT + Z3":
        return run_smt(query, candidate_ids, rule_library, rule_by_id, native=True)
    if method == "CTHR-style SMT + Z3":
        return run_smt(query, candidate_ids, rule_library, rule_by_id, native=False)
    if method == "Native MILP + HiGHS":
        return run_milp(spec, query, grounding_task, candidate_ids, rule_library, rule_by_id, native=True)
    if method == "CTHR-style MILP + HiGHS":
        return run_milp(spec, query, grounding_task, candidate_ids, rule_library, rule_by_id, native=False)
    raise ValueError(method)


def evaluate_dataset(spec: DatasetSpec) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    algorithm_inputs = item_map(spec.algorithm_inputs)
    scenario_models = item_map(spec.scenario_models)
    references = item_map(spec.evaluation_references)
    grounding_results = grounding_result_map(spec.grounding_full)
    templates_by_rule = constraint_template_map(spec.constraint_templates)
    rule_library = read_json(spec.rule_library)
    rule_by_id = {str(rule["rule_id"]): rule for rule in rule_library.get("rules", []) if rule.get("rule_id")}
    if set(algorithm_inputs) != set(scenario_models) or set(algorithm_inputs) != set(references):
        raise ValueError(f"{spec.name} layer IDs do not match")
    if set(algorithm_inputs) != set(grounding_results):
        raise ValueError(f"{spec.name} grounding result IDs do not match")

    rows: list[dict[str, Any]] = []
    grounding_audit: dict[str, Any] = {}
    for task_id in sorted(algorithm_inputs):
        grounding_task = dict(algorithm_inputs[task_id])
        query = prepare_query(grounding_task, scenario_models[task_id])
        query["_compiled_rule_constraint_templates_by_id"] = templates_by_rule
        reference = references[task_id]
        feasible = reference_feasible(reference, query)
        reference_ids = reference_rule_ids(reference)
        grounding_row = grounding_results[task_id]
        candidate_ids = ids_from_grounding(grounding_row, "candidate_rule_ids_generated")
        cthr_valid_ids = ids_from_grounding(grounding_row, "predicted_valid_rule_ids")
        candidate_rules = [rule_by_id[rule_id] for rule_id in candidate_ids if rule_id in rule_by_id]
        grounding_audit[task_id] = {
            "candidate_rule_ids": candidate_ids,
            "candidate_rule_count": len(candidate_ids),
            "cthr_predicted_valid_rule_ids": cthr_valid_ids,
            "grounding_result_exact_match": bool(grounding_row.get("Exact Match")),
        }
        for method, method_type in METHOD_SPECS:
            start = time.perf_counter()
            result = run_method(
                spec,
                method,
                query,
                grounding_task,
                candidate_rules,
                rule_library,
                rule_by_id,
                cthr_valid_ids,
            )
            elapsed = (time.perf_counter() - start) * 1000.0
            predicted = sorted(result.predicted_rule_ids) if result.supported else []
            precision = base.method_rule_precision(predicted, reference_ids)
            recall = base.method_rule_recall(predicted, reference_ids)
            if precision is None:
                precision = 0.0
            if recall is None:
                recall = 0.0
            sem_ok = semantic_valid(feasible, result.optimized_x, predicted, reference_ids) if result.supported else False
            formal_ok = bool(result.formal_feasible) if result.supported else False
            rows.append(
                {
                    "Dataset": spec.name,
                    "task_id": task_id,
                    "target_interaction": target_interaction(reference),
                    "Method": method,
                    "Method type": method_type,
                    "grounded_candidate_count": len(candidate_ids),
                    "predicted_rule_ids": predicted,
                    "reference_rule_ids": reference_ids,
                    "rule_precision": precision,
                    "rule_recall": recall,
                    "formal_feasible": formal_ok,
                    "semantic_valid": sem_ok,
                    "false_accept": bool(formal_ok and not sem_ok),
                    "invalid_case": bool(not sem_ok),
                    "unsupported_reason": "" if result.supported else result.unsupported_reason,
                    "runtime_ms": round(elapsed, 3),
                }
            )
    summary = {
        "dataset": spec.name,
        "domain": spec.domain,
        "root": str(spec.root),
        "tasks": len(algorithm_inputs),
        "rule_library": str(spec.rule_library),
        "grounding_result": str(spec.grounding_full),
        "constraint_templates": str(spec.constraint_templates),
        "rule_library_rules": len(rule_library.get("rules", [])),
        "constraint_template_rules": len(templates_by_rule),
        "constraint_template_count": sum(len(templates) for templates in templates_by_rule.values()),
        "grounding": {
            "source": "precomputed domain-specific Section 6.3 end-to-end grounding output",
            "mean_candidate_count": sum(item["candidate_rule_count"] for item in grounding_audit.values())
            / max(1, len(grounding_audit)),
            "cthr_exact_match_rate": sum(1 for item in grounding_audit.values() if item["grounding_result_exact_match"])
            / max(1, len(grounding_audit)),
        },
    }
    return rows, summary


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def aggregate(rows: list[dict[str, Any]], dataset: str, method: str, method_type: str) -> dict[str, Any]:
    subset = [row for row in rows if row["Dataset"] == dataset and row["Method"] == method]
    total = len(subset)
    if total == 0:
        return {
            "Dataset": dataset,
            "Method": method,
            "Method type": method_type,
            "Rule Precision": "N/A",
            "Rule Recall": "N/A",
            "Formal CSR": "N/A",
            "Sem-CSR": "N/A",
            "False accept": "N/A",
            "Invalid cases": "0/0 (N/A)",
        }
    unsupported = sum(1 for row in subset if row["unsupported_reason"])
    invalid = sum(1 for row in subset if row["invalid_case"])
    suffix = f" ({unsupported} unsupported)" if unsupported else ""
    return {
        "Dataset": dataset,
        "Method": method,
        "Method type": method_type,
        "Rule Precision": pct(sum(float(row["rule_precision"]) for row in subset) / total),
        "Rule Recall": pct(sum(float(row["rule_recall"]) for row in subset) / total),
        "Formal CSR": pct(sum(1 for row in subset if row["formal_feasible"]) / total),
        "Sem-CSR": pct(sum(1 for row in subset if row["semantic_valid"]) / total),
        "False accept": pct(sum(1 for row in subset if row["false_accept"]) / total),
        "Invalid cases": f"{invalid}/{total} ({100.0 * invalid / total:.1f}%){suffix}",
    }


def unsupported_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for row in rows:
        reason = str(row.get("unsupported_reason") or "")
        if not reason:
            continue
        key = f"{row['Dataset']}::{row['Method']}"
        out.setdefault(key, {})
        out[key][reason] = out[key].get(reason, 0) + 1
    return out


def markdown_table(rows: list[dict[str, Any]], headers: list[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(csv_cell(row.get(header)) for header in headers) + " |")
    return "\n".join(lines)


def build_report(aggregate_rows: list[dict[str, Any]], per_task_rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    headers = [
        "Dataset",
        "Method",
        "Method type",
        "Rule Precision",
        "Rule Recall",
        "Formal CSR",
        "Sem-CSR",
        "False accept",
        "Invalid cases",
    ]
    unsupported = unsupported_summary(per_task_rows)
    unsupported_lines = []
    for key, counts in sorted(unsupported.items()):
        detail = "; ".join(f"{reason}: {count}" for reason, count in sorted(counts.items()))
        unsupported_lines.append(f"- {key}: {detail}")
    if not unsupported_lines:
        unsupported_lines = ["- none"]

    return "\n".join(
        [
            "# Section 6.2 Table 1: Full-KG KG-to-Constraint Pipeline",
            "",
            "## Scope",
            "",
            "- Uses only the first table setup: rule library -> grounding -> method-native or CTHR-style symbolic modeling -> feasible-region construction -> optimization.",
            "- Does not run the compiled-cell backend reuse table.",
            "- Algorithms read rule libraries, algorithm inputs, and public scenario models. Evaluation references are used only for metrics.",
            "- Grounding uses the domain-specific Section 6.3 end-to-end grounding outputs for aviation and architecture.",
            "- Rule-to-constraint compilation uses the materialized compiled rule-constraint template layer, with raw rule-library constraints only as a fallback.",
            "",
            "## Datasets",
            "",
            f"- Aviation: {summary['datasets']['Aviation']['tasks']} tasks from `aviation_fullkg_clean`.",
            f"- Architecture: {summary['datasets']['Architecture']['tasks']} tasks from `architecture_fullkg_clean`.",
            "",
            "## Main Result",
            "",
            markdown_table(aggregate_rows, headers),
            "",
            "## Unsupported / N/A",
            "",
            *unsupported_lines,
            "",
            "## Metric Note",
            "",
            "- Percentages are computed over all tasks in each dataset, so unsupported task-method pairs count as non-solved/non-semantic-valid for CSR-style metrics.",
            "- Rule precision and recall are averaged per task; unsupported or empty-prediction cases contribute 0.",
            "- Sem-CSR, False accept, and Invalid cases test only the returned optimized solution against source-rule reference constraints; rule-set correctness is reported separately by Rule Precision/Recall.",
            "",
            "## Run Summary",
            "",
            "```json",
            json.dumps(summary, ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    per_task_rows: list[dict[str, Any]] = []
    dataset_summaries: dict[str, Any] = {}
    for spec in DATASETS:
        rows, dataset_summary = evaluate_dataset(spec)
        per_task_rows.extend(rows)
        dataset_summaries[spec.name] = dataset_summary

    aggregate_rows = []
    for spec in DATASETS:
        for method, method_type in METHOD_SPECS:
            aggregate_rows.append(aggregate(per_task_rows, spec.name, method, method_type))

    per_task_headers = [
        "Dataset",
        "task_id",
        "target_interaction",
        "Method",
        "Method type",
        "grounded_candidate_count",
        "predicted_rule_ids",
        "reference_rule_ids",
        "rule_precision",
        "rule_recall",
        "formal_feasible",
        "semantic_valid",
        "false_accept",
        "invalid_case",
        "unsupported_reason",
        "runtime_ms",
    ]
    aggregate_headers = [
        "Dataset",
        "Method",
        "Method type",
        "Rule Precision",
        "Rule Recall",
        "Formal CSR",
        "Sem-CSR",
        "False accept",
        "Invalid cases",
    ]
    outputs = {
        "per_task_csv": RESULTS_DIR / "section_6_2_table1_fullkg_pipeline_per_task.csv",
        "overall_csv": RESULTS_DIR / "section_6_2_table1_fullkg_pipeline_overall.csv",
        "overall_md": RESULTS_DIR / "section_6_2_table1_fullkg_pipeline_overall.md",
        "overall_json": RESULTS_DIR / "section_6_2_table1_fullkg_pipeline_overall.json",
        "report_md": RESULTS_DIR / "section_6_2_table1_fullkg_pipeline_report.md",
    }
    write_csv(outputs["per_task_csv"], per_task_rows, per_task_headers)
    write_csv(outputs["overall_csv"], aggregate_rows, aggregate_headers)
    outputs["overall_md"].write_text(markdown_table(aggregate_rows, aggregate_headers), encoding="utf-8")

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "datasets": dataset_summaries,
        "methods": [{"Method": method, "Method type": method_type} for method, method_type in METHOD_SPECS],
        "input_restrictions": {
            "rule_library_as_input": True,
            "algorithm_inputs_as_input": True,
            "public_scenario_models_as_input": True,
            "evaluation_references_as_method_input": False,
            "cthr_compiled_cells_as_method_input": False,
        },
        "metric_scope": "All percentages use total dataset task count as denominator. Unsupported task-method pairs count as non-success for CSR-style metrics.",
        "unsupported": unsupported_summary(per_task_rows),
        "outputs": {key: str(value) for key, value in outputs.items()},
        "aggregate_rows": aggregate_rows,
    }
    write_json(outputs["overall_json"], summary)
    outputs["report_md"].write_text(build_report(aggregate_rows, per_task_rows, summary), encoding="utf-8")
    print(json.dumps({"outputs": summary["outputs"], "aggregate_rows": aggregate_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
