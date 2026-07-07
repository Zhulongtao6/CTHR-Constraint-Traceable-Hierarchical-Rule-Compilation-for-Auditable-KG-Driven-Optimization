from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from aviation_fullkg_curated_extensions import build_curated_aviation_extensions


ROOT = Path(__file__).resolve().parents[1]
CTHR_ROOT = ROOT.parents[1]
PAPER_DIR = CTHR_ROOT / "paper"

SOURCE_LAYER_DIR = PAPER_DIR / "aviation_benchmark_layers"
SOURCE_QUERIES = SOURCE_LAYER_DIR / "aviation_optimization_queries.json"
SOURCE_RULE_LABELS = SOURCE_LAYER_DIR / "aviation_rule_structure_labels.json"
SOURCE_FEASIBLE = SOURCE_LAYER_DIR / "aviation_feasible_region_labels.json"
FULL_QWEN_RULE_LIBRARY = (
    PAPER_DIR
    / "full_aviation_kg_rule_library_model_comparison"
    / "full_aviation_rule_library_qwen.json"
)

OUT_DIR = ROOT / "datasets" / "aviation_fullkg_clean"
INPUT_DIR = OUT_DIR / "algorithm_inputs"
SCENARIO_MODEL_DIR = OUT_DIR / "scenario_models"
REFERENCE_DIR = OUT_DIR / "evaluation_references"
TASK_DIR = OUT_DIR / "tasks"
RULE_LIBRARY_DIR = OUT_DIR / "rule_libraries"


ALGORITHM_INPUT_KEYS = [
    "omega_id",
    "title",
    "domain",
    "task_type",
    "design_intent",
    "scenario_facts",
    "decision_variables",
    "objectives",
    "public_scenario_model",
    "query_preferences",
    "preference_weights",
    "visible_input_note",
]

FORBIDDEN_INPUT_KEYS = {
    "candidate_rule_ids",
    "candidate_rule_ids_expected_for_diagnostics",
    "certificate_targets",
    "executable_constraints",
    "expected_defeated_rule_ids",
    "expected_provenance",
    "expected_rule_behavior",
    "expected_source_rule_ids",
    "expected_surviving_rule_ids",
    "expected_valid_rule_structures",
    "feasible_region_label",
    "final_valid_rule_ids",
    "final_valid_rule_ids_expected_for_evaluation",
    "hidden_evaluation_reference",
    "pre_solver_structure_checks",
    "reference_semantics",
    "rule_structure_label",
    "solver_constraint_cells",
    "solver_constraints",
    "structure_only_constraints",
    "valid_constraint_cell_ids",
    "valid_constraint_cells",
    "valid_rule_structures_expected",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def items(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError(f"Expected an items list in {path}")
    return payload["items"]


def by_omega(records: list[dict[str, Any]], source_name: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for record in records:
        omega_id = str(record.get("omega_id", ""))
        if not omega_id:
            raise ValueError(f"Missing omega_id in {source_name}: {record}")
        if omega_id in out:
            raise ValueError(f"Duplicate omega_id {omega_id} in {source_name}")
        out[omega_id] = record
    return out


def rule_id_set(rule_library: dict[str, Any]) -> set[str]:
    return {
        str(rule["rule_id"])
        for rule in rule_library.get("rules", [])
        if isinstance(rule, dict) and rule.get("rule_id")
    }


def rule_lookup_by_id(rule_library: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for rule in rule_library.get("rules", []):
        if not isinstance(rule, dict) or not rule.get("rule_id"):
            continue
        rule_id = str(rule["rule_id"])
        if rule_id in out:
            raise ValueError(f"Duplicate rule_id in rule library: {rule_id}")
        out[rule_id] = rule
    return out


def collect_rule_library_source_ids(payload: Any) -> set[str]:
    refs: set[str] = set()
    if isinstance(payload, dict):
        if payload.get("source_type") == "rule_library" and payload.get("source_id"):
            refs.add(str(payload["source_id"]))
        for value in payload.values():
            refs.update(collect_rule_library_source_ids(value))
    elif isinstance(payload, list):
        for value in payload:
            refs.update(collect_rule_library_source_ids(value))
    return refs


def referenced_rule_ids(
    label: dict[str, Any],
    query: dict[str, Any],
    feasible: dict[str, Any] | None = None,
) -> set[str]:
    refs: set[str] = set()
    for field in (
        "expected_source_rule_ids",
        "expected_defeated_rule_ids",
        "expected_surviving_rule_ids",
    ):
        refs.update(str(rule_id) for rule_id in label.get(field, []) if rule_id)
    certificate_targets = query.get("certificate_targets", {})
    if isinstance(certificate_targets, dict):
        refs.update(str(rule_id) for rule_id in certificate_targets.get("source_rule_ids", []) if rule_id)
    if feasible is not None:
        refs.update(collect_rule_library_source_ids(feasible))
    return refs


def make_algorithm_input(query: dict[str, Any]) -> dict[str, Any]:
    algorithm_input = {key: query[key] for key in ALGORITHM_INPUT_KEYS if key in query}
    omega_id = str(query["omega_id"])
    algorithm_input["public_scenario_model"] = {
        "model_id": f"{omega_id}_scenario_model",
        "path": "scenario_models/aviation_public_scenario_models.json",
        "visibility": "public_algorithm_input",
        "purpose": "Non-normative task physics/objective-closure constraints visible to optimizers; contains no expected rule IDs or labels.",
    }
    algorithm_input["visible_input_note"] = (
        "Visible task input only. Algorithms may also read the public scenario model and rule library. "
        "Rule labels, rule-derived feasible-region answers, and rule-id bindings remain hidden evaluation references."
    )
    return algorithm_input


def rule_library_id_bindings(
    rule_lookup: dict[str, dict[str, Any]],
    rule_ids: set[str],
) -> list[dict[str, Any]]:
    bindings: list[dict[str, Any]] = []
    for rule_id in sorted(rule_ids):
        rule = rule_lookup[rule_id]
        bindings.append(
            {
                "rule_id": rule_id,
                "exists_in_rule_library": True,
                "rule_library_rule_id": rule_id,
                "rule_name": rule.get("name"),
                "rule_type": rule.get("rule_type"),
                "source_chunk_ids": rule.get("source_chunk_ids", []),
                "source_node_ids": rule.get("source_node_ids", []),
                "provenance": rule.get("provenance", []),
                "rule_library_constraints": [
                    {
                        "variable": item.get("variable"),
                        "op": item.get("op"),
                        "value": item.get("value"),
                        "unit": item.get("unit"),
                    }
                    for item in rule.get("constraints", [])
                    if isinstance(item, dict)
                ],
            }
        )
    return bindings


def reference_constraint_rule_bindings(
    feasible: dict[str, Any],
    rule_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    bindings: list[dict[str, Any]] = []

    def visit(payload: Any, container: str) -> None:
        if isinstance(payload, dict):
            if payload.get("source_type") == "rule_library" and payload.get("source_id"):
                rule_id = str(payload["source_id"])
                rule = rule_lookup[rule_id]
                bindings.append(
                    {
                        "reference_container": container,
                        "reference_constraint_id": payload.get("constraint_id"),
                        "reference_role": payload.get("role"),
                        "reference_expression": payload.get("expression"),
                        "source_rule_id": rule_id,
                        "rule_library_rule_id": rule_id,
                        "rule_library_rule_name": rule.get("name"),
                    }
                )
            for key, value in payload.items():
                visit(value, f"{container}.{key}")
        elif isinstance(payload, list):
            for index, value in enumerate(payload):
                visit(value, f"{container}[{index}]")

    visit(feasible.get("executable_constraints", []), "feasible_region.executable_constraints")
    visit(feasible.get("structure_only_constraints", []), "feasible_region.structure_only_constraints")
    visit(feasible.get("valid_constraint_cells", []), "feasible_region.valid_constraint_cells")
    return bindings


def make_reference(
    query: dict[str, Any],
    label: dict[str, Any],
    feasible: dict[str, Any],
    rule_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rule_ids = referenced_rule_ids(label, query, feasible)
    return {
        "omega_id": query["omega_id"],
        "title": query.get("title"),
        "rule_structure": {
            "expected_source_rule_ids": label.get("expected_source_rule_ids", []),
            "expected_defeated_rule_ids": label.get("expected_defeated_rule_ids", []),
            "expected_surviving_rule_ids": label.get("expected_surviving_rule_ids", []),
            "expected_rule_behavior": label.get("expected_rule_behavior", {}),
            "challenge_types": label.get("challenge_types", []),
            "valid_constraint_cell_ids": label.get("valid_constraint_cell_ids", []),
            "expected_valid_rule_structures": label.get("expected_valid_rule_structures", []),
            "expected_provenance": label.get("expected_provenance", {}),
        },
        "feasible_region": {
            "executable_constraints": feasible.get("executable_constraints", []),
            "structure_only_constraints": feasible.get("structure_only_constraints", []),
            "valid_constraint_cells": feasible.get("valid_constraint_cells", []),
            "reference_semantics": feasible.get("reference_semantics", {}),
        },
        "certificate_targets": query.get("certificate_targets", {}),
        "diagnostic_candidate_rule_ids_reference_only": label.get("expected_source_rule_ids", []),
        "rule_library_id_bindings": rule_library_id_bindings(rule_lookup, rule_ids),
        "reference_constraint_rule_bindings": reference_constraint_rule_bindings(feasible, rule_lookup),
    }


def make_public_scenario_model(
    query: dict[str, Any],
    feasible: dict[str, Any],
) -> dict[str, Any]:
    public_executable_constraints = [
        constraint
        for constraint in feasible.get("executable_constraints", [])
        if constraint.get("source_type") == "task_or_scenario_model"
    ]
    public_structure_constraints = [
        constraint
        for constraint in feasible.get("structure_only_constraints", [])
        if constraint.get("source_type") == "task_or_scenario_model"
    ]
    public_scenario_cells: list[dict[str, Any]] = []
    for cell in feasible.get("valid_constraint_cells", []):
        cell_executable_constraints = [
            constraint
            for key in ("constraints", "executable_constraints")
            for constraint in cell.get(key, [])
            if constraint.get("source_type") == "task_or_scenario_model"
        ]
        cell_structure_constraints = [
            constraint
            for constraint in cell.get("structure_only_constraints", [])
            if constraint.get("source_type") == "task_or_scenario_model"
        ]
        if cell_executable_constraints or cell_structure_constraints:
            public_scenario_cells.append(
                {
                    "cell_id": cell.get("cell_id"),
                    "description": cell.get("description"),
                    "executable_constraints": cell_executable_constraints,
                    "structure_only_constraints": cell_structure_constraints,
                }
            )
    return {
        "omega_id": query["omega_id"],
        "model_id": f"{query['omega_id']}_scenario_model",
        "title": query.get("title"),
        "visibility": "public_algorithm_input",
        "model_scope": "task_physics_and_objective_closure_only",
        "leakage_policy": "No expected rule IDs, defeated/surviving labels, provenance answers, certificate targets, or rule-library bindings are included.",
        "executable_constraints": public_executable_constraints,
        "structure_only_constraints": public_structure_constraints,
        "scenario_model_cells": public_scenario_cells,
    }


def count_public_scenario_model_constraints(public_scenario_models: list[dict[str, Any]]) -> int:
    count = 0
    for model in public_scenario_models:
        count += len(model.get("executable_constraints", []))
        count += len(model.get("structure_only_constraints", []))
        for cell in model.get("scenario_model_cells", []):
            count += len(cell.get("executable_constraints", []))
            count += len(cell.get("structure_only_constraints", []))
    return count


def normalize_executable_reference_expressions(payload: Any) -> None:
    if isinstance(payload, dict):
        checker_expression = payload.get("checker_expression")
        if payload.get("executable") is True and isinstance(checker_expression, str):
            payload["expression"] = checker_expression
        for value in payload.values():
            normalize_executable_reference_expressions(value)
    elif isinstance(payload, list):
        for value in payload:
            normalize_executable_reference_expressions(value)


def executable_constraint(
    constraint_id: str,
    expression: str,
    role: str,
    source_id: str,
    decision_variables: list[str],
    scenario_fields: list[str] | None = None,
    source_type: str = "task_or_scenario_model",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "constraint_id": constraint_id,
        "expression": expression,
        "role": role,
        "source_type": source_type,
        "source_id": source_id,
        "executable": True,
        "checker_expression": expression,
        "expression_language": "python_safe_arithmetic_predicate",
        "symbols": {
            "decision_variables": decision_variables,
            "scenario_fields": scenario_fields or [],
            "unresolved_symbols": [],
        },
    }
    if metadata:
        out["metadata"] = metadata
    return out


def guard_field_aliases(field: str) -> list[str]:
    aliases = [field]
    if "." in field:
        aliases.append(field.split(".")[-1])
    for prefix in ("scenario.", "aircraft.", "procedure.", "operation."):
        if field.startswith(prefix):
            aliases.append(field[len(prefix) :])
    out: list[str] = []
    for alias in aliases:
        if alias and alias not in out:
            out.append(alias)
    return out


def normalize_guard_value(value: Any) -> str:
    return str(value).strip().lower().replace("_", " ").replace("-", " ")


def guard_value_satisfies(actual: Any, op: str | None, expected: Any) -> bool:
    if isinstance(actual, list):
        if op in {"eq", "="}:
            return any(guard_value_satisfies(item, op, expected) for item in actual)
        if op == "in" and isinstance(expected, list):
            return any(guard_value_satisfies(item, "in", expected) for item in actual)
    if op in {"eq", "="}:
        return actual == expected
    if op == "neq":
        return actual != expected
    if op == "in" and isinstance(expected, list):
        if actual in expected:
            return True
        actual_norm = normalize_guard_value(actual)
        return any(actual_norm == normalize_guard_value(item) for item in expected)
    if op == "not_in" and isinstance(expected, list):
        return actual not in expected
    if op in {"gt", "gte", "lt", "lte"}:
        try:
            if op == "gt":
                return actual > expected
            if op == "gte":
                return actual >= expected
            if op == "lt":
                return actual < expected
            return actual <= expected
        except TypeError:
            return normalize_guard_value(actual) == normalize_guard_value(expected)
    return True


def existing_guard_value(field: str, scenario_facts: dict[str, Any]) -> tuple[str | None, Any]:
    for alias in guard_field_aliases(field):
        if alias in scenario_facts:
            return alias, scenario_facts[alias]
    plural = f"{field}s"
    if plural in scenario_facts:
        return plural, scenario_facts[plural]
    return None, None


def guard_condition_value(condition: dict[str, Any], scenario_facts: dict[str, Any]) -> Any:
    field = str(condition.get("field", ""))
    op = condition.get("op")
    expected = condition.get("value")
    _, actual = existing_guard_value(field, scenario_facts)
    if op == "in" and isinstance(expected, list) and expected:
        actual_norm = normalize_guard_value(actual)
        for item in expected:
            item_norm = normalize_guard_value(item)
            if actual_norm == item_norm or item_norm in actual_norm:
                return item
        return expected[0]
    if op == "gt" and type(expected) in {int, float}:
        return expected + 1
    if op == "lt" and type(expected) in {int, float}:
        return expected - 1
    if op == "gt" and isinstance(expected, str) and type(scenario_facts.get(expected)) in {int, float}:
        return scenario_facts[expected] + 1
    if op == "lt" and isinstance(expected, str) and type(scenario_facts.get(expected)) in {int, float}:
        return scenario_facts[expected] - 1
    if op in {"eq", "lte", "gte", "gt", "lt", "="}:
        return expected
    if op == "neq":
        return actual if actual not in {None, expected} else f"not_{expected}"
    return expected


def plural_guard_field(field: str) -> str:
    if field.endswith("y"):
        return f"{field[:-1]}ies"
    return f"{field}s"


def guard_aligned_scenario_facts(
    scenario_facts: dict[str, Any],
    surviving_rule_ids: list[str],
    rule_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    aligned = dict(scenario_facts)
    eq_values_by_field: dict[str, list[Any]] = {}
    for rule_id in surviving_rule_ids:
        guard = rule_lookup[rule_id].get("guard", {})
        for condition in guard.get("all", []) if isinstance(guard, dict) else []:
            if isinstance(condition, dict) and condition.get("op") in {"eq", "="} and condition.get("field"):
                field = str(condition["field"])
                value = condition.get("value")
                values = eq_values_by_field.setdefault(field, [])
                if value not in values:
                    values.append(value)

    multi_eq_fields = {field for field, values in eq_values_by_field.items() if len(values) > 1}
    for field in multi_eq_fields:
        aligned[field] = eq_values_by_field[field]
        aligned.setdefault(plural_guard_field(field), eq_values_by_field[field])

    for rule_id in surviving_rule_ids:
        guard = rule_lookup[rule_id].get("guard", {})
        for condition in guard.get("all", []) if isinstance(guard, dict) else []:
            if not isinstance(condition, dict) or not condition.get("field"):
                continue
            field = str(condition["field"])
            if field in multi_eq_fields:
                continue
            alias, actual = existing_guard_value(field, aligned)
            expected = condition.get("value")
            if isinstance(expected, str) and expected in aligned:
                expected = aligned[expected]
            if alias is not None and guard_value_satisfies(actual, condition.get("op"), expected):
                continue
            aligned[alias or field] = guard_condition_value(condition, aligned)
    return aligned


def numeric_like(value: Any) -> bool:
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        try:
            float(value)
            return True
        except ValueError:
            return False
    return False


def rule_is_textual_or_semantic(rule: dict[str, Any]) -> bool:
    constraints = rule.get("constraints", [])
    if not constraints:
        return True
    for item in constraints:
        if not isinstance(item, dict):
            continue
        if item.get("op") == "text":
            return True
        value = item.get("value")
        if isinstance(value, (bool, list, dict)):
            return True
        if isinstance(value, str) and not numeric_like(value):
            return True
    return False


def annotate_semantic_proxy_constraints(payload: Any, rule_lookup: dict[str, dict[str, Any]]) -> None:
    if isinstance(payload, dict):
        source_id = str(payload.get("source_id", ""))
        if (
            payload.get("executable") is True
            and payload.get("source_type") == "rule_library"
            and source_id in rule_lookup
            and rule_is_textual_or_semantic(rule_lookup[source_id])
        ):
            metadata = payload.setdefault("metadata", {})
            metadata.setdefault("semantic_proxy_encoding", True)
            metadata.setdefault("derived_from_text_rule", True)
            metadata.setdefault("derived_from_rule_id", source_id)
            metadata.setdefault(
                "derivation_note",
                "This executable predicate is a benchmark semantic proxy for a textual or categorical full-KG rule; it is not a raw numeric KG constraint.",
            )
        for value in payload.values():
            annotate_semantic_proxy_constraints(value, rule_lookup)
    elif isinstance(payload, list):
        for value in payload:
            annotate_semantic_proxy_constraints(value, rule_lookup)


def remove_constraints_by_source_id(payload: dict[str, Any], source_id: str) -> None:
    for key in ("solver_constraints", "executable_constraints"):
        if isinstance(payload.get(key), list):
            payload[key] = [
                item
                for item in payload[key]
                if not (isinstance(item, dict) and item.get("source_id") == source_id)
            ]


def set_branch_reference(
    query: dict[str, Any],
    label: dict[str, Any],
    surviving: list[str],
    defeated: list[str],
) -> None:
    label["expected_defeated_rule_ids"] = defeated
    label["expected_surviving_rule_ids"] = surviving
    label["expected_valid_rule_structures"] = [surviving]
    certificate_targets = query.get("certificate_targets", {})
    if isinstance(certificate_targets, dict):
        certificate_targets["source_rule_ids"] = surviving


def repair_known_source_task_issues(
    queries: dict[str, dict[str, Any]],
    labels: dict[str, dict[str, Any]],
    feasible: dict[str, dict[str, Any]],
) -> None:
    if "AVI_OPT_03" in feasible:
        c = executable_constraint(
            "C5",
            "extra_protection_buffer_km == r_design_km - 26.2",
            "extra_buffer_certificate",
            "scenario_tolerance_design_model",
            ["extra_protection_buffer_km", "r_design_km"],
        )
        for payload_key in ("solver_constraints",):
            queries["AVI_OPT_03"].setdefault(payload_key, [])
            if not any(item.get("constraint_id") == "C5" for item in queries["AVI_OPT_03"][payload_key]):
                queries["AVI_OPT_03"][payload_key].append(c)
        feasible["AVI_OPT_03"].setdefault("executable_constraints", [])
        if not any(item.get("constraint_id") == "C5" for item in feasible["AVI_OPT_03"]["executable_constraints"]):
            feasible["AVI_OPT_03"]["executable_constraints"].append(c)

    if "AVI_OPT_13" in labels:
        set_branch_reference(
            queries["AVI_OPT_13"],
            labels["AVI_OPT_13"],
            ["RA-6.4.1-turn-angle-threshold", "tnh_obstacle_clearance_requirement"],
            ["RA-6.3.4-align-final-approach-track"],
        )
        remove_constraints_by_source_id(queries["AVI_OPT_13"], "RA-6.3.4-align-final-approach-track")
        remove_constraints_by_source_id(feasible["AVI_OPT_13"], "RA-6.3.4-align-final-approach-track")

    if "AVI_OPT_14" in labels:
        set_branch_reference(
            queries["AVI_OPT_14"],
            labels["AVI_OPT_14"],
            ["turn_init_area_der_extent"],
            ["turn_init_area_fato_extent"],
        )

    if "AVI_OPT_15" in labels:
        set_branch_reference(
            queries["AVI_OPT_15"],
            labels["AVI_OPT_15"],
            ["turn_init_area_fato_extent"],
            ["turn_init_area_der_extent"],
        )


def prepare_aviation_task_records(
    queries: dict[str, dict[str, Any]],
    labels: dict[str, dict[str, Any]],
    feasible: dict[str, dict[str, Any]],
    rule_lookup: dict[str, dict[str, Any]],
) -> None:
    repair_known_source_task_issues(queries, labels, feasible)
    for omega_id, query in queries.items():
        label = labels[omega_id]
        feasible_label = feasible[omega_id]
        surviving = [str(rule_id) for rule_id in label.get("expected_surviving_rule_ids", [])]
        scenario_facts = guard_aligned_scenario_facts(query.get("scenario_facts", {}), surviving, rule_lookup)
        query["scenario_facts"] = scenario_facts
        label["scenario_facts"] = scenario_facts
        feasible_label["scenario_facts"] = scenario_facts
        annotate_semantic_proxy_constraints(query, rule_lookup)
        annotate_semantic_proxy_constraints(feasible_label, rule_lookup)


def find_forbidden_keys(payload: Any, path: str = "$") -> list[str]:
    hits: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            child_path = f"{path}.{key}"
            if key in FORBIDDEN_INPUT_KEYS:
                hits.append(child_path)
            hits.extend(find_forbidden_keys(value, child_path))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            hits.extend(find_forbidden_keys(value, f"{path}[{index}]"))
    return hits


def synthetic_rule_ids(rule_library: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for rule in rule_library.get("rules", []):
        if not isinstance(rule, dict):
            continue
        if rule.get("synthetic_stress_rule") or str(rule.get("rule_id", "")).startswith("stress_"):
            out.append(str(rule.get("rule_id")))
    return sorted(out)


def main() -> None:
    queries = by_omega(items(SOURCE_QUERIES), "source aviation queries")
    labels = by_omega(items(SOURCE_RULE_LABELS), "source aviation rule labels")
    feasible = by_omega(items(SOURCE_FEASIBLE), "source aviation feasible labels")
    if set(queries) != set(labels) or set(queries) != set(feasible):
        raise ValueError("Source aviation layer omega_id sets do not match")
    source_task_count = len(queries)

    rule_library = read_json(FULL_QWEN_RULE_LIBRARY)
    rule_lookup = rule_lookup_by_id(rule_library)
    full_rule_ids = set(rule_lookup)
    stress_rules = synthetic_rule_ids(rule_library)
    if stress_rules:
        raise ValueError(f"Full Qwen rule library unexpectedly contains stress rules: {stress_rules}")

    curated_extensions = build_curated_aviation_extensions(rule_lookup)
    for query, label, feasible_label in curated_extensions:
        omega_id = str(query["omega_id"])
        if omega_id in queries or omega_id in labels or omega_id in feasible:
            raise ValueError(f"Curated extension duplicates source omega_id: {omega_id}")
        queries[omega_id] = query
        labels[omega_id] = label
        feasible[omega_id] = feasible_label

    prepare_aviation_task_records(queries, labels, feasible, rule_lookup)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    SCENARIO_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    TASK_DIR.mkdir(parents=True, exist_ok=True)
    RULE_LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    for stale_task in TASK_DIR.glob("*.json"):
        stale_task.unlink()

    algorithm_inputs: list[dict[str, Any]] = []
    public_scenario_models: list[dict[str, Any]] = []
    evaluation_references: list[dict[str, Any]] = []
    leakage_hits: dict[str, list[str]] = {}
    missing_by_task: dict[str, list[str]] = {}

    for omega_id in sorted(queries):
        query = queries[omega_id]
        label = labels[omega_id]
        feasible_label = feasible[omega_id]
        normalize_executable_reference_expressions(feasible_label)
        algorithm_input = make_algorithm_input(query)
        missing = sorted(referenced_rule_ids(label, query, feasible_label) - full_rule_ids)
        if missing:
            missing_by_task[omega_id] = missing
            continue
        evaluation_reference = make_reference(query, label, feasible_label, rule_lookup)
        public_scenario_model = make_public_scenario_model(query, feasible_label)

        hits = find_forbidden_keys(algorithm_input)
        if hits:
            leakage_hits[omega_id] = hits

        task_payload = {
            "version": "aviation_fullkg_clean_task_v1",
            "algorithm_input": algorithm_input,
            "evaluation_reference": evaluation_reference,
            "metadata": {
                "split": query.get("_split", "aviation_fullkg_original19"),
                "rule_library": "rule_libraries/full_aviation_rule_library_qwen.json",
                "public_scenario_model": "scenario_models/aviation_public_scenario_models.json",
                "input_reference_policy": "algorithm_input plus public_scenario_models contain visible design-query and task-physics fields; evaluation_reference contains labels, rule-derived feasible-region answers, and rule-id bindings.",
            },
        }

        algorithm_inputs.append(algorithm_input)
        public_scenario_models.append(public_scenario_model)
        evaluation_references.append(evaluation_reference)
        write_json(TASK_DIR / f"{omega_id}.json", task_payload)

    shutil.copy2(FULL_QWEN_RULE_LIBRARY, RULE_LIBRARY_DIR / "full_aviation_rule_library_qwen.json")

    input_payload = {
        "version": "aviation_fullkg_clean_algorithm_inputs_v1",
        "items": algorithm_inputs,
    }
    reference_payload = {
        "version": "aviation_fullkg_clean_evaluation_references_v1",
        "items": evaluation_references,
    }
    write_json(INPUT_DIR / "aviation_algorithm_inputs.json", input_payload)
    write_json(
        SCENARIO_MODEL_DIR / "aviation_public_scenario_models.json",
        {
            "version": "aviation_fullkg_clean_public_scenario_models_v1",
            "visibility": "public_algorithm_input",
            "purpose": "Task physics and objective-closure constraints needed for fair optimization. These models do not contain expected rule IDs, defeated/surviving labels, provenance answers, certificate targets, or rule-library bindings.",
            "items": public_scenario_models,
        },
    )
    write_json(REFERENCE_DIR / "aviation_evaluation_references.json", reference_payload)

    all_referenced_rule_ids: set[str] = set()
    for tid in queries:
        all_referenced_rule_ids.update(referenced_rule_ids(labels[tid], queries[tid], feasible[tid]))

    audit = {
        "version": "aviation_fullkg_clean_leakage_audit_v1",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "task_count": len(algorithm_inputs),
        "source_task_count": source_task_count,
        "curated_extension_task_count": len(curated_extensions),
        "rule_library_rules": len(full_rule_ids),
        "synthetic_stress_rule_count": 0,
        "public_scenario_model_count": len(public_scenario_models),
        "public_scenario_model_constraint_count": count_public_scenario_model_constraints(public_scenario_models),
        "forbidden_input_keys": sorted(FORBIDDEN_INPUT_KEYS),
        "input_forbidden_key_hits": leakage_hits,
        "input_forbidden_key_hit_count": sum(len(value) for value in leakage_hits.values()),
        "referenced_rule_ids": len(all_referenced_rule_ids),
        "missing_reference_rule_ids_by_task": missing_by_task,
        "missing_reference_rule_id_count": sum(len(value) for value in missing_by_task.values()),
        "status": "pass" if not leakage_hits and not missing_by_task else "fail",
    }
    write_json(OUT_DIR / "LEAKAGE_AUDIT.json", audit)

    manifest = {
        "version": "aviation_fullkg_clean_manifest_v1",
        "generated_at": audit["generated_at"],
        "purpose": "Clean aviation benchmark using the Qwen full-KG rule library as the only source rule library and strict algorithm-input/reference separation.",
        "source_files": {
            "queries": str(SOURCE_QUERIES),
            "rule_structure_labels": str(SOURCE_RULE_LABELS),
            "feasible_region_labels": str(SOURCE_FEASIBLE),
            "full_qwen_rule_library": str(FULL_QWEN_RULE_LIBRARY),
        },
        "outputs": {
            "algorithm_inputs": str(INPUT_DIR / "aviation_algorithm_inputs.json"),
            "public_scenario_models": str(SCENARIO_MODEL_DIR / "aviation_public_scenario_models.json"),
            "evaluation_references": str(REFERENCE_DIR / "aviation_evaluation_references.json"),
            "task_files": str(TASK_DIR),
            "rule_library": str(RULE_LIBRARY_DIR / "full_aviation_rule_library_qwen.json"),
            "leakage_audit": str(OUT_DIR / "LEAKAGE_AUDIT.json"),
        },
        "counts": {
            "tasks": len(algorithm_inputs),
            "source_tasks": source_task_count,
            "curated_extension_tasks": len(curated_extensions),
            "rule_library_rules": len(full_rule_ids),
            "synthetic_stress_rules": 0,
            "public_scenario_model_constraints": audit["public_scenario_model_constraint_count"],
            "referenced_rule_ids": audit["referenced_rule_ids"],
            "missing_reference_rule_ids": audit["missing_reference_rule_id_count"],
            "forbidden_input_key_hits": audit["input_forbidden_key_hit_count"],
        },
        "excluded_from_main_dataset": {
            "aviation_stress_tasks": 12,
            "reason": "The stress tasks depend on synthetic stress-extension rules and are therefore diagnostic, not part of the full-KG-only main aviation benchmark.",
        },
    }
    write_json(OUT_DIR / "MANIFEST.json", manifest)

    readme = [
        "# Aviation Full-KG Clean Dataset",
        "",
        "This dataset treats `full_aviation_rule_library_qwen.json` as the full aviation rule library.",
        "It contains the 19 original aviation optimization tasks plus 11 curated full-KG extensions.",
        "It excludes the 12 aviation stress tasks because they depend on synthetic stress-extension rules.",
        "",
        "## Split Policy",
        "",
        "- `algorithm_inputs/aviation_algorithm_inputs.json`: fields visible to algorithms.",
        "- `scenario_models/aviation_public_scenario_models.json`: public task-physics and objective-closure constraints visible to algorithms.",
        "- `evaluation_references/aviation_evaluation_references.json`: reference answers used only by evaluators.",
        "- `tasks/*.json`: paired files with explicit `algorithm_input` and `evaluation_reference` sections.",
        "- `rule_libraries/full_aviation_rule_library_qwen.json`: the only source rule library for this dataset.",
        "- `evaluation_reference.rule_library_id_bindings`: explicit mapping from each correct/reference rule ID to the same `rule_id` in the rule library.",
        "- `evaluation_reference.reference_constraint_rule_bindings`: explicit mapping from each reference constraint to its source rule-library ID.",
        "",
        "Algorithm code should read only `algorithm_inputs/aviation_algorithm_inputs.json`, `scenario_models/aviation_public_scenario_models.json`, and the rule library.",
        "The paired `tasks/*.json` files are for auditing and evaluator-side debugging; do not pass full task files to algorithm code.",
        "",
        "## Counts",
        "",
        f"- Tasks: {len(algorithm_inputs)}",
        f"- Original source tasks: {source_task_count}",
        f"- Curated full-KG extension tasks: {len(curated_extensions)}",
        f"- Rule-library rules: {len(full_rule_ids)}",
        f"- Public task/scenario-model constraints: {audit['public_scenario_model_constraint_count']}",
        f"- Referenced source-rule IDs covered by the rule library: {audit['referenced_rule_ids'] - audit['missing_reference_rule_id_count']} / {audit['referenced_rule_ids']}",
        f"- Forbidden input-key hits: {audit['input_forbidden_key_hit_count']}",
        "",
        "## Diagnostic Stress Data",
        "",
        "The previous combined aviation dataset remains useful as a resolver stress test, but it should be described as diagnostic because it adds 14 synthetic stress rules.",
        "",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(readme), encoding="utf-8")

    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
