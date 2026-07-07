from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FULL_QWEN_RULE_LIBRARY = (
    ROOT
    / "results"
    / "kg_to_rule_library"
    / "architecture"
    / "full_architecture_rule_library_qwen.json"
)

OUT_DIR = ROOT / "datasets" / "architecture_fullkg_clean"
INPUT_DIR = OUT_DIR / "algorithm_inputs"
SCENARIO_MODEL_DIR = OUT_DIR / "scenario_models"
REFERENCE_DIR = OUT_DIR / "evaluation_references"
TASK_DIR = OUT_DIR / "tasks"
RULE_LIBRARY_DIR = OUT_DIR / "rule_libraries"


FORBIDDEN_INPUT_KEYS = {
    "candidate_rule_ids",
    "candidate_rule_ids_expected_for_diagnostics",
    "certificate_targets",
    "diagnostic_candidate_rule_ids_reference_only",
    "evaluation_reference",
    "executable_constraints",
    "expected_defeated_rule_ids",
    "expected_provenance",
    "expected_rule_behavior",
    "expected_source_rule_ids",
    "expected_surviving_rule_ids",
    "expected_valid_rule_structures",
    "feasible_region",
    "feasible_region_label",
    "final_valid_rule_ids",
    "final_valid_rule_ids_expected_for_evaluation",
    "hidden_evaluation_reference",
    "pre_solver_structure_checks",
    "reference_constraint_rule_bindings",
    "reference_semantics",
    "rule_library_id_bindings",
    "rule_structure",
    "rule_structure_label",
    "solver_constraint_cells",
    "solver_constraints",
    "stress_metadata",
    "structure_only_constraints",
    "structured_surplus_rule_ids",
    "valid_constraint_cell_ids",
    "valid_constraint_cells",
    "valid_rule_structures_expected",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def rule_lookup_by_id(rule_library: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for rule in rule_library.get("rules", []):
        if not isinstance(rule, dict) or not rule.get("rule_id"):
            continue
        # Keep the first instance when the full extraction has a duplicate ID.
        out.setdefault(str(rule["rule_id"]), rule)
    return out


def canonicalize_rule_library(rule_library: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seen: set[str] = set()
    canonical_rules: list[dict[str, Any]] = []
    duplicate_records: list[dict[str, Any]] = []
    first_index_by_id: dict[str, int] = {}

    for index, rule in enumerate(rule_library.get("rules", [])):
        if not isinstance(rule, dict) or not rule.get("rule_id"):
            canonical_rules.append(rule)
            continue
        rule_id = str(rule["rule_id"])
        if rule_id in seen:
            duplicate_records.append(
                {
                    "rule_id": rule_id,
                    "dropped_source_index": index,
                    "kept_source_index": first_index_by_id[rule_id],
                    "dropped_rule_name": rule.get("name"),
                    "policy": "first_rule_id_occurrence_wins",
                }
            )
            continue
        seen.add(rule_id)
        first_index_by_id[rule_id] = index
        canonical_rules.append(rule)

    canonical = dict(rule_library)
    canonical["rules"] = canonical_rules
    metadata = dict(canonical.get("submission_support_metadata", {}))
    metadata["duplicate_rule_id_resolution_policy"] = {
        "policy": "first_rule_id_occurrence_wins",
        "reason": "The source full-KG extraction contains duplicate rule_id values; the benchmark rule library must be deterministic for rule-id keyed retrieval.",
        "duplicate_rule_id_count_in_source": len(duplicate_records),
    }
    canonical["submission_support_metadata"] = metadata
    return canonical, duplicate_records


def constraint(
    constraint_id: str,
    expression: str,
    role: str,
    source_id: str,
    decision_variables: list[str] | None = None,
    scenario_fields: list[str] | None = None,
    source_type: str = "rule_library",
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
            "decision_variables": decision_variables or [],
            "scenario_fields": scenario_fields or [],
            "unresolved_symbols": [],
        },
    }
    if metadata:
        out["metadata"] = metadata
    return out


def objective_closure_metadata(note: str) -> dict[str, Any]:
    return {
        "objective_closure": True,
        "closure_source": "task_or_scenario_model",
        "closure_visibility": "public_algorithm_input",
        "closure_note": note,
    }


def semantic_indicator_metadata(rule_id: str, note: str) -> dict[str, Any]:
    return {
        "semantic_indicator_encoding": True,
        "derived_from_text_or_semantic_rule": True,
        "derived_from_rule_id": rule_id,
        "encoding_note": note,
    }


def exact_parameter_boundary_metadata(rule_id: str, note: str) -> dict[str, Any]:
    return {
        "normative_parameter_interpretation": "extracted_equality_used_as_design_boundary",
        "derived_from_rule_id": rule_id,
        "interpretation_note": note,
    }


def structure_constraint(
    constraint_id: str,
    expression: str,
    role: str,
    source_id: str,
    source_type: str = "rule_library",
) -> dict[str, Any]:
    return {
        "constraint_id": constraint_id,
        "expression": expression,
        "role": role,
        "source_type": source_type,
        "source_id": source_id,
        "executable": False,
        "reason_not_executable": "not represented as a direct numeric predicate",
    }


def var(
    var_type: str,
    unit: str,
    lower: float | int,
    upper: float | int,
) -> dict[str, Any]:
    return {"type": var_type, "unit": unit, "lower": lower, "upper": upper}


def rule_provenance(rule_lookup: dict[str, dict[str, Any]], rule_ids: list[str]) -> dict[str, Any]:
    chunk_ids: set[str] = set()
    node_ids: set[str] = set()
    edge_ids: set[str] = set()
    documents: list[dict[str, Any]] = []
    seen_documents: set[str] = set()

    for rule_id in rule_ids:
        rule = rule_lookup[rule_id]
        chunk_ids.update(str(value) for value in rule.get("source_chunk_ids", []) if value)
        node_ids.update(str(value) for value in rule.get("source_node_ids", []) if value)
        for item in rule.get("provenance", []):
            if isinstance(item, dict):
                key = json.dumps(item, sort_keys=True, ensure_ascii=False)
                if key not in seen_documents:
                    documents.append(item)
                    seen_documents.add(key)
        for item in rule.get("constraints", []):
            evidence = item.get("evidence", {}) if isinstance(item, dict) else {}
            edge_ids.update(str(value) for value in evidence.get("kg_edge_ids", []) if value)
        for item in rule.get("relations", []):
            evidence = item.get("evidence", {}) if isinstance(item, dict) else {}
            edge_ids.update(str(value) for value in evidence.get("kg_edge_ids", []) if value)

    return {
        "kg_chunk_ids": sorted(chunk_ids),
        "kg_node_ids": sorted(node_ids),
        "kg_edge_ids": sorted(edge_ids),
        "source_documents": documents,
    }


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


def guard_field_aliases(field: str) -> list[str]:
    aliases = [field]
    if "." in field:
        aliases.append(field.split(".")[-1])
    for prefix in ("scenario.", "building.", "door.", "compliance."):
        if field.startswith(prefix):
            aliases.append(field[len(prefix) :])
    out: list[str] = []
    for alias in aliases:
        if alias and alias not in out:
            out.append(alias)
    return out


def guard_condition_value(condition: dict[str, Any], scenario_facts: dict[str, Any]) -> Any:
    field = str(condition.get("field", ""))
    op = condition.get("op")
    value = condition.get("value")
    for alias in guard_field_aliases(field):
        if alias in scenario_facts:
            return scenario_facts[alias]
    if op == "in" and isinstance(value, list) and value:
        return value[0]
    if op == "gt" and type(value) in {int, float}:
        return value + 1
    if op == "lt" and type(value) in {int, float}:
        return value - 1
    if op in {"eq", "lte", "gte", "lt", "gt", "="}:
        return value
    if op == "neq":
        return f"not_{value}"
    return value


def guard_aligned_scenario_facts(
    scenario_facts: dict[str, Any],
    source_rule_ids: list[str],
    rule_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    aligned = dict(scenario_facts)
    for rule_id in source_rule_ids:
        guard = rule_lookup[rule_id].get("guard", {})
        for condition in guard.get("all", []) if isinstance(guard, dict) else []:
            if not isinstance(condition, dict) or not condition.get("field"):
                continue
            field = str(condition["field"])
            value = guard_condition_value(condition, aligned)
            for alias in guard_field_aliases(field):
                aligned.setdefault(alias, value)
    return aligned


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
        if isinstance(value, (str, list, dict, bool)):
            return True
    return False


def annotate_semantic_indicator_constraints(
    constraints: list[dict[str, Any]],
    rule_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for item in constraints:
        out = dict(item)
        expression = str(out.get("expression", ""))
        source_id = str(out.get("source_id", ""))
        if (
            out.get("source_type") == "rule_library"
            and source_id in rule_lookup
            and rule_is_textual_or_semantic(rule_lookup[source_id])
            and ("_indicator" in expression or "_count" in expression or "count" in expression)
        ):
            metadata = dict(out.get("metadata", {}))
            metadata.update(
                semantic_indicator_metadata(
                    source_id,
                    "This executable predicate is a task-level semantic encoding of a textual or categorical full-KG rule.",
                )
            )
            out["metadata"] = metadata
        annotated.append(out)
    return annotated


def build_task(
    *,
    omega_id: str,
    title: str,
    source_domain: str,
    task_type: str,
    design_intent: str,
    scenario_facts: dict[str, Any],
    decision_variables: dict[str, Any],
    objectives: list[dict[str, str]],
    query_preferences: dict[str, Any],
    source_rule_ids: list[str],
    executable_constraints: list[dict[str, Any]],
    challenge_types: list[str],
    structure_only_constraints: list[dict[str, Any]] | None = None,
    defeated_rule_ids: list[str] | None = None,
) -> dict[str, Any]:
    defeated = defeated_rule_ids or []
    surviving = sorted(rule_id for rule_id in source_rule_ids if rule_id not in set(defeated))
    cell = {
        "cell_id": f"{omega_id}_cell_1",
        "cell_type": "single_valid_fullkg_rule_structure",
        "rule_ids": surviving,
        "constraints": executable_constraints,
        "description": "Executable feasible cell induced by the expected full-KG rule structure for this grounded scenario.",
    }
    return {
        "omega_id": omega_id,
        "title": title,
        "source_domain": source_domain,
        "task_type": task_type,
        "design_intent": design_intent,
        "scenario_facts": scenario_facts,
        "decision_variables": decision_variables,
        "objectives": objectives,
        "query_preferences": query_preferences,
        "source_rule_ids": source_rule_ids,
        "defeated_rule_ids": defeated,
        "surviving_rule_ids": surviving,
        "executable_constraints": executable_constraints,
        "structure_only_constraints": structure_only_constraints or [],
        "valid_constraint_cells": [cell],
        "challenge_types": challenge_types,
    }


TASK_SPECS: list[dict[str, Any]] = [
    build_task(
        omega_id="ARCH_FKG_01",
        title="Pool barrier gate latch height",
        source_domain="IFC",
        task_type="egress_and_pool_barrier_design",
        design_intent="Set the latch release height for a pool barrier gate while balancing child-resistance margin against adult operability.",
        scenario_facts={"facility_type": "aquatic_center", "barrier_element": "pool_access_gate"},
        decision_variables={
            "latch_release_height_in": var("continuous", "inch", 36, 60),
            "child_resistance_margin_in": var("continuous", "inch", 0, 24),
            "operability_score": var("continuous", "score", 0, 10),
        },
        objectives=[
            {"name": "maximize_child_resistance_margin", "expression": "child_resistance_margin_in"},
            {"name": "maximize_adult_operability", "expression": "operability_score"},
        ],
        query_preferences={"lambda": [0.55, 0.45], "meaning": "prefer safety margin while preserving operability"},
        source_rule_ids=["ifc-2021-1010-2-4-pool-gate-latch-height"],
        executable_constraints=[
            constraint("C1", "latch_release_height_in <= 54", "pool_gate_latch_height_limit", "ifc-2021-1010-2-4-pool-gate-latch-height", ["latch_release_height_in"]),
            constraint("C2", "child_resistance_margin_in == latch_release_height_in - 36", "latch_height_child_resistance_proxy", "scenario_pool_gate_model", ["child_resistance_margin_in", "latch_release_height_in"], source_type="task_or_scenario_model"),
        ],
        challenge_types=["scenario_conditioned_applicability"],
    ),
    build_task(
        omega_id="ARCH_FKG_02",
        title="Address identification sign legibility",
        source_domain="IFC",
        task_type="fire_department_wayfinding_design",
        design_intent="Choose address character size and stroke width for a compact facade sign that remains legible for emergency response.",
        scenario_facts={"building_frontage": "street-facing", "fire_department_response_route": True},
        decision_variables={
            "character_height_in": var("continuous", "inch", 2, 12),
            "stroke_width_in": var("continuous", "inch", 0.1, 2),
            "sign_panel_area_sf": var("continuous", "sf", 0.5, 12),
            "legibility_margin_score": var("continuous", "score", 0, 10),
        },
        objectives=[
            {"name": "minimize_sign_panel_area", "expression": "sign_panel_area_sf"},
            {"name": "maximize_legibility_margin", "expression": "legibility_margin_score"},
        ],
        query_preferences={"lambda": [0.45, 0.55], "meaning": "slightly prioritize emergency legibility over compact sign area"},
        source_rule_ids=["ifc-2021-address-id-min-size"],
        executable_constraints=[
            constraint("C1", "character_height_in >= 4", "address_character_height", "ifc-2021-address-id-min-size", ["character_height_in"]),
            constraint("C2", "stroke_width_in >= 0.5", "address_stroke_width", "ifc-2021-address-id-min-size", ["stroke_width_in"]),
        ],
        challenge_types=["multi_constraint_single_rule"],
    ),
    build_task(
        omega_id="ARCH_FKG_03",
        title="Type IA fire-flow loop sizing",
        source_domain="IFC",
        task_type="fire_flow_supply_design",
        design_intent="Size the water-supply loop for a Type IA building while balancing available fire flow, duration reserve, and pipe-loop length.",
        scenario_facts={"construction_type": "Type IA", "fire_flow_calculation_area_sf": 21000},
        decision_variables={
            "available_fire_flow_gpm": var("continuous", "gpm", 1000, 2500),
            "flow_duration_hours": var("continuous", "hour", 1, 4),
            "site_loop_length_ft": var("continuous", "ft", 200, 2000),
            "fire_flow_margin_gpm": var("continuous", "gpm", 0, 1000),
        },
        objectives=[
            {"name": "minimize_site_loop_length", "expression": "site_loop_length_ft"},
            {"name": "maximize_fire_flow_margin", "expression": "fire_flow_margin_gpm"},
        ],
        query_preferences={"lambda": [0.5, 0.5], "meaning": "balance infrastructure compactness and fire-flow reserve"},
        source_rule_ids=["ifc2021_fire_flow_table_row_1"],
        executable_constraints=[
            constraint("C1", "available_fire_flow_gpm >= 1500", "required_fire_flow", "ifc2021_fire_flow_table_row_1", ["available_fire_flow_gpm"]),
            constraint("C2", "flow_duration_hours >= 2", "required_flow_duration", "ifc2021_fire_flow_table_row_1", ["flow_duration_hours"]),
            constraint("C3", "fire_flow_margin_gpm == available_fire_flow_gpm - 1500", "fire_flow_margin_certificate", "scenario_fire_flow_model", ["fire_flow_margin_gpm", "available_fire_flow_gpm"], source_type="task_or_scenario_model"),
        ],
        challenge_types=["scenario_conditioned_applicability", "multi_constraint_single_rule"],
    ),
    build_task(
        omega_id="ARCH_FKG_04",
        title="Type IV building fire-flow reserve",
        source_domain="IFC",
        task_type="fire_flow_supply_design",
        design_intent="Select fire-flow capacity for a Type IV building with a small calculation area, trading water reserve against service-connection size.",
        scenario_facts={"construction_type": "Type IV", "fire_flow_calculation_area_sf": 7600},
        decision_variables={
            "available_fire_flow_gpm": var("continuous", "gpm", 1000, 2400),
            "flow_duration_hours": var("continuous", "hour", 1, 4),
            "service_connection_size_score": var("continuous", "score", 0, 10),
            "duration_margin_hours": var("continuous", "hour", 0, 2),
        },
        objectives=[
            {"name": "minimize_service_connection_size", "expression": "service_connection_size_score"},
            {"name": "maximize_duration_margin", "expression": "duration_margin_hours"},
        ],
        query_preferences={"lambda": [0.55, 0.45], "meaning": "prefer compact service sizing while retaining duration margin"},
        source_rule_ids=["ifc2021_fire_flow_table_row_3"],
        executable_constraints=[
            constraint("C1", "available_fire_flow_gpm >= 1500", "required_fire_flow", "ifc2021_fire_flow_table_row_3", ["available_fire_flow_gpm"]),
            constraint("C2", "flow_duration_hours >= 2", "required_flow_duration", "ifc2021_fire_flow_table_row_3", ["flow_duration_hours"]),
            constraint("C3", "duration_margin_hours == flow_duration_hours - 2", "duration_margin_certificate", "scenario_fire_flow_model", ["duration_margin_hours", "flow_duration_hours"], source_type="task_or_scenario_model"),
        ],
        challenge_types=["scenario_conditioned_applicability"],
    ),
    build_task(
        omega_id="ARCH_FKG_05",
        title="High-piled commodity classification precedence",
        source_domain="IFC",
        task_type="commodity_classification_precedence_design",
        design_intent="Resolve a high-piled storage commodity classification when multiple candidate classifications are present, applying the highest-classification precedence rule while balancing storage density and documentation confidence.",
        scenario_facts={
            "storage_type": "high_piled_combustible_storage",
            "multiple_commodity_classifications_detected": True,
            "highest_detected_class_rank": 4,
            "lower_candidate_class_rank": 2,
        },
        decision_variables={
            "selected_commodity_class_rank": var("integer", "rank", 1, 4),
            "classification_documentation_level": var("continuous", "score", 0, 10),
            "storage_density_score": var("continuous", "score", 0, 10),
            "classification_confidence_score": var("continuous", "score", 0, 10),
        },
        objectives=[
            {"name": "maximize_storage_density", "expression": "storage_density_score"},
            {"name": "maximize_classification_confidence", "expression": "classification_confidence_score"},
        ],
        query_preferences={"lambda": [0.45, 0.55], "meaning": "slightly prioritize confidence in the governing commodity classification"},
        source_rule_ids=["ifc-3203-9-2-highest-classification-rule"],
        executable_constraints=[
            constraint(
                "C1",
                "selected_commodity_class_rank == highest_detected_class_rank",
                "highest_classification_governs",
                "ifc-3203-9-2-highest-classification-rule",
                ["selected_commodity_class_rank"],
                ["highest_detected_class_rank"],
                metadata=semantic_indicator_metadata(
                    "ifc-3203-9-2-highest-classification-rule",
                    "The textual precedence rule is encoded as the selected commodity class rank matching the highest detected candidate rank.",
                ),
            ),
        ],
        structure_only_constraints=[
            structure_constraint(
                "S1",
                "lower commodity classifications are not the winning classification when a higher applicable classification is present",
                "highest_classification_precedence_resolution",
                "ifc-3203-9-2-highest-classification-rule",
            )
        ],
        challenge_types=["precedence", "scenario_conditioned_applicability"],
    ),
    build_task(
        omega_id="ARCH_FKG_06",
        title="Metal hydride storage temperature envelope",
        source_domain="IFC",
        task_type="hazardous_material_storage_design",
        design_intent="Set a metal-hydride storage operating temperature while reducing cooling load and preserving temperature margin.",
        scenario_facts={"storage_system": "metal_hydride", "falling_object_exposure": "controlled_room"},
        decision_variables={
            "maximum_operating_temperature_f": var("continuous", "degF", 60, 140),
            "cooling_load_score": var("continuous", "score", 0, 10),
            "temperature_margin_f": var("continuous", "degF", 0, 65),
        },
        objectives=[
            {"name": "minimize_cooling_load", "expression": "cooling_load_score"},
            {"name": "maximize_temperature_margin", "expression": "temperature_margin_f"},
        ],
        query_preferences={"lambda": [0.45, 0.55], "meaning": "slightly prioritize temperature safety margin"},
        source_rule_ids=["ifc2021_metal_hydride_temp_limit", "ifc2021_metal_hydride_falling_objects_protection"],
        executable_constraints=[
            constraint("C1", "maximum_operating_temperature_f <= 125", "metal_hydride_temperature_limit", "ifc2021_metal_hydride_temp_limit", ["maximum_operating_temperature_f"]),
            constraint("C2", "temperature_margin_f == 125 - maximum_operating_temperature_f", "temperature_margin_certificate", "scenario_metal_hydride_model", ["temperature_margin_f", "maximum_operating_temperature_f"], source_type="task_or_scenario_model"),
        ],
        structure_only_constraints=[
            structure_constraint("S1", "storage placement avoids damage from falling objects", "falling_object_protection", "ifc2021_metal_hydride_falling_objects_protection")
        ],
        challenge_types=["multi_rule_conjunction"],
    ),
    build_task(
        omega_id="ARCH_FKG_07",
        title="Patient-bed egress door opening",
        source_domain="IFC",
        task_type="healthcare_egress_door_design",
        design_intent="Size a patient-bed movement door while balancing opening area against bed-transfer clearance.",
        scenario_facts={"occupancy_group": "I-2", "bed_movement_required": True, "door_status": "new"},
        decision_variables={
            "door_clear_width_in": var("continuous", "inch", 32, 60),
            "door_opening_height_in": var("continuous", "inch", 76, 96),
            "opening_area_score": var("continuous", "score", 0, 20),
            "bed_transfer_margin_in": var("continuous", "inch", 0, 20),
        },
        objectives=[
            {"name": "minimize_opening_area_score", "expression": "opening_area_score"},
            {"name": "maximize_bed_transfer_margin", "expression": "bed_transfer_margin_in"},
        ],
        query_preferences={"lambda": [0.45, 0.55], "meaning": "slightly prioritize bed-transfer clearance"},
        source_rule_ids=["ifc2021-moe-door-patient-bed-min-width", "ifc2021-moe-door-patient-bed-min-height"],
        executable_constraints=[
            constraint("C1", "door_clear_width_in >= 41.5", "patient_bed_door_width", "ifc2021-moe-door-patient-bed-min-width", ["door_clear_width_in"]),
            constraint("C2", "door_opening_height_in >= 80", "patient_bed_door_height", "ifc2021-moe-door-patient-bed-min-height", ["door_opening_height_in"]),
            constraint("C3", "bed_transfer_margin_in == door_clear_width_in - 41.5", "bed_transfer_margin_certificate", "scenario_healthcare_door_model", ["bed_transfer_margin_in", "door_clear_width_in"], source_type="task_or_scenario_model"),
        ],
        challenge_types=["multi_rule_conjunction"],
    ),
    build_task(
        omega_id="ARCH_FKG_08",
        title="Existing I-2 door width exception",
        source_domain="IFC",
        task_type="exception_resolution",
        design_intent="Retrofit an existing Group I-2 Condition 1 egress door, applying the existing-door exception while preserving usable transfer width.",
        scenario_facts={"occupancy_group": "I-2", "condition": "Condition 1", "door_status": "existing"},
        decision_variables={
            "door_clear_width_in": var("continuous", "inch", 28, 48),
            "retrofit_cost_score": var("continuous", "score", 0, 10),
            "usable_width_margin_in": var("continuous", "inch", 0, 16),
        },
        objectives=[
            {"name": "minimize_retrofit_cost", "expression": "retrofit_cost_score"},
            {"name": "maximize_usable_width_margin", "expression": "usable_width_margin_in"},
        ],
        query_preferences={"lambda": [0.55, 0.45], "meaning": "prefer lower retrofit cost while retaining width margin"},
        source_rule_ids=["ifc2021-moe-door-patient-bed-min-width", "ifc2021-moe-door-patient-bed-exception-i2-condition1"],
        defeated_rule_ids=["ifc2021-moe-door-patient-bed-min-width"],
        executable_constraints=[
            constraint("C1", "door_clear_width_in >= 32", "existing_i2_condition1_width_exception", "ifc2021-moe-door-patient-bed-exception-i2-condition1", ["door_clear_width_in"]),
            constraint("C2", "usable_width_margin_in == door_clear_width_in - 32", "exception_width_margin_certificate", "scenario_existing_door_model", ["usable_width_margin_in", "door_clear_width_in"], source_type="task_or_scenario_model"),
        ],
        structure_only_constraints=[
            structure_constraint("S1", "new patient-bed door width rule is defeated by the existing I-2 Condition 1 exception", "exception_defeats_baseline_width_rule", "ifc2021-moe-door-patient-bed-min-width")
        ],
        challenge_types=["exception_or_override"],
    ),
    build_task(
        omega_id="ARCH_FKG_09",
        title="Area-of-refuge wheelchair space layout",
        source_domain="IFC",
        task_type="accessible_egress_refuge_design",
        design_intent="Lay out an area of refuge, balancing wheelchair capacity against obstruction and smoke-separation requirements.",
        scenario_facts={"egress_component": "area_of_refuge", "adjacent_smoke_barrier": True},
        decision_variables={
            "adjoining_wheelchair_obstruction_count": var("integer", "count", 0, 4),
            "smoke_separation_indicator": var("binary", "indicator", 0, 1),
            "wheelchair_spaces_count": var("integer", "count", 1, 8),
            "refuge_area_sf": var("continuous", "sf", 30, 300),
        },
        objectives=[
            {"name": "maximize_wheelchair_spaces", "expression": "wheelchair_spaces_count"},
            {"name": "minimize_refuge_area", "expression": "refuge_area_sf"},
        ],
        query_preferences={"lambda": [0.55, 0.45], "meaning": "slightly prioritize refuge capacity"},
        source_rule_ids=["ifc-2021-1009-6-4-wheelchair-access", "ifc-2021-1009-6-4-separation"],
        executable_constraints=[
            constraint("C1", "adjoining_wheelchair_obstruction_count <= 1", "wheelchair_space_obstruction_limit", "ifc-2021-1009-6-4-wheelchair-access", ["adjoining_wheelchair_obstruction_count"]),
            constraint("C2", "smoke_separation_indicator == 1", "area_of_refuge_separation_provided", "ifc-2021-1009-6-4-separation", ["smoke_separation_indicator"]),
        ],
        challenge_types=["multi_rule_conjunction"],
    ),
    build_task(
        omega_id="ARCH_FKG_10",
        title="Fire alarm zone partitioning",
        source_domain="IFC",
        task_type="fire_alarm_zone_design",
        design_intent="Partition a fire alarm system into zones that keep each zone within area and linear-dimension limits while minimizing panel complexity.",
        scenario_facts={"building_use": "mixed_use", "alarm_zone_layout": "floor_by_floor"},
        decision_variables={
            "zone_area_sf": var("continuous", "sf", 5000, 30000),
            "zone_length_ft": var("continuous", "ft", 100, 400),
            "zone_count": var("integer", "count", 1, 12),
            "coverage_per_zone_score": var("continuous", "score", 0, 10),
        },
        objectives=[
            {"name": "minimize_zone_count", "expression": "zone_count"},
            {"name": "maximize_coverage_per_zone", "expression": "coverage_per_zone_score"},
        ],
        query_preferences={"lambda": [0.5, 0.5], "meaning": "balance panel simplicity and coverage efficiency"},
        source_rule_ids=["ifc_907_6_4_zone_area_max", "ifc_907_6_4_zone_dimension_max"],
        executable_constraints=[
            constraint("C1", "zone_area_sf <= 22500", "fire_alarm_zone_area_limit", "ifc_907_6_4_zone_area_max", ["zone_area_sf"]),
            constraint("C2", "zone_length_ft <= 300", "fire_alarm_zone_dimension_limit", "ifc_907_6_4_zone_dimension_max", ["zone_length_ft"]),
        ],
        challenge_types=["multi_rule_conjunction"],
    ),
    build_task(
        omega_id="ARCH_FKG_11",
        title="Audible alarm sound-pressure window",
        source_domain="IFC",
        task_type="fire_alarm_audibility_design",
        design_intent="Select alarm sound level in an occupiable space while balancing audibility reserve against occupant comfort.",
        scenario_facts={"ambient_avg_dba": 55, "ambient_max_60s_dba": 70, "space_type": "occupied_office"},
        decision_variables={
            "sound_pressure_level_dba": var("continuous", "dBA", 60, 115),
            "audibility_margin_dba": var("continuous", "dBA", 0, 40),
            "occupant_discomfort_score": var("continuous", "score", 0, 10),
        },
        objectives=[
            {"name": "maximize_audibility_margin", "expression": "audibility_margin_dba"},
            {"name": "minimize_occupant_discomfort", "expression": "occupant_discomfort_score"},
        ],
        query_preferences={"lambda": [0.55, 0.45], "meaning": "slightly prioritize audible warning margin"},
        source_rule_ids=["ifc-2021-907-5-2-1-2-audible-alarm-spl-min", "ifc-2021-907-5-2-1-2-audible-alarm-spl-max"],
        executable_constraints=[
            constraint("C1", "sound_pressure_level_dba >= max(ambient_avg_dba + 15, ambient_max_60s_dba + 5)", "audible_alarm_minimum_spl", "ifc-2021-907-5-2-1-2-audible-alarm-spl-min", ["sound_pressure_level_dba"], ["ambient_avg_dba", "ambient_max_60s_dba"]),
            constraint("C2", "sound_pressure_level_dba <= 110", "audible_alarm_maximum_spl", "ifc-2021-907-5-2-1-2-audible-alarm-spl-max", ["sound_pressure_level_dba"]),
            constraint("C3", "audibility_margin_dba == sound_pressure_level_dba - max(ambient_avg_dba + 15, ambient_max_60s_dba + 5)", "audibility_margin_certificate", "scenario_audibility_model", ["audibility_margin_dba", "sound_pressure_level_dba"], ["ambient_avg_dba", "ambient_max_60s_dba"], source_type="task_or_scenario_model"),
        ],
        challenge_types=["formula_propagation", "multi_rule_conjunction"],
    ),
    build_task(
        omega_id="ARCH_FKG_12",
        title="Exterior egress door landing and bottom clearance",
        source_domain="IFC",
        task_type="egress_door_threshold_design",
        design_intent="Detail an exterior egress door for an industrial tenant, balancing weather threshold performance with landing and bottom-clearance limits.",
        scenario_facts={"occupancy_group": "group f", "accessible_route": False, "exterior_door": True},
        decision_variables={
            "landing_height_differential_in": var("continuous", "inch", 0, 10),
            "door_bottom_clearance_in": var("continuous", "inch", 0, 2),
            "weather_resilience_score": var("continuous", "score", 0, 10),
        },
        objectives=[
            {"name": "maximize_weather_resilience", "expression": "weather_resilience_score"},
            {"name": "minimize_landing_height_differential", "expression": "landing_height_differential_in"},
        ],
        query_preferences={"lambda": [0.45, 0.55], "meaning": "slightly prioritize egress smoothness"},
        source_rule_ids=["ifc_exterior_door_landing_drop_f_h_r2_s", "ifc-2021-door-bottom-clearance-max"],
        executable_constraints=[
            constraint("C1", "landing_height_differential_in <= 7", "exterior_door_landing_drop_limit", "ifc_exterior_door_landing_drop_f_h_r2_s", ["landing_height_differential_in"]),
            constraint("C2", "door_bottom_clearance_in <= 1", "door_bottom_clearance_limit", "ifc-2021-door-bottom-clearance-max", ["door_bottom_clearance_in"]),
        ],
        challenge_types=["multi_rule_conjunction"],
    ),
    build_task(
        omega_id="ARCH_FKG_13",
        title="Temporary tent area and clearance planning",
        source_domain="IFC",
        task_type="temporary_structure_site_design",
        design_intent="Plan temporary tents for an outdoor event while trading aggregate covered area against clearance to structures and other tents.",
        scenario_facts={"temporary_structure": "multiple_tents", "event_type": "outdoor_assembly"},
        decision_variables={
            "aggregate_tent_area_sf": var("continuous", "sf", 100, 1000),
            "minimum_clearance_ft": var("continuous", "ft", 0, 30),
            "covered_area_score": var("continuous", "score", 0, 10),
        },
        objectives=[
            {"name": "maximize_covered_area", "expression": "covered_area_score"},
            {"name": "maximize_clearance_margin", "expression": "minimum_clearance_ft - 12"},
        ],
        query_preferences={"lambda": [0.5, 0.5], "meaning": "balance usable event area and fire-break clearance"},
        source_rule_ids=["ifc_tent_aggregate_area_limit", "ifc_tent_clearance_requirement"],
        executable_constraints=[
            constraint("C1", "aggregate_tent_area_sf <= 700", "tent_aggregate_area_limit", "ifc_tent_aggregate_area_limit", ["aggregate_tent_area_sf"]),
            constraint("C2", "minimum_clearance_ft >= 12", "tent_clearance_requirement", "ifc_tent_clearance_requirement", ["minimum_clearance_ft"]),
        ],
        challenge_types=["multi_rule_conjunction"],
    ),
    build_task(
        omega_id="ARCH_FKG_14",
        title="Rack storage flue-space and height envelope",
        source_domain="IFC",
        task_type="sprinklered_rack_storage_design",
        design_intent="Choose rack storage height and flue spacing under ceiling-sprinkler design constraints, balancing storage density against flue-space loss.",
        scenario_facts={"storage_type": "rack_storage", "automatic_sprinkler_protection": True},
        decision_variables={
            "transverse_flue_space_width_in": var("continuous", "inch", 0, 8),
            "maximum_storage_height_ft": var("continuous", "ft", 8, 30),
            "storage_density_score": var("continuous", "score", 0, 10),
            "flue_space_loss_score": var("continuous", "score", 0, 10),
        },
        objectives=[
            {"name": "maximize_storage_density", "expression": "storage_density_score"},
            {"name": "minimize_flue_space_loss", "expression": "flue_space_loss_score"},
        ],
        query_preferences={"lambda": [0.55, 0.45], "meaning": "slightly prioritize storage density while preserving sprinkler flue space"},
        source_rule_ids=["ifc2021-flue-space-transverse-min-3in", "ifc2021-storage-height-ceiling-sprinkler-max-20ft"],
        executable_constraints=[
            constraint("C1", "transverse_flue_space_width_in >= 3", "transverse_flue_space_minimum", "ifc2021-flue-space-transverse-min-3in", ["transverse_flue_space_width_in"]),
            constraint("C2", "maximum_storage_height_ft <= 20", "ceiling_sprinkler_storage_height_limit", "ifc2021-storage-height-ceiling-sprinkler-max-20ft", ["maximum_storage_height_ft"]),
        ],
        challenge_types=["multi_rule_conjunction"],
    ),
    build_task(
        omega_id="ARCH_FKG_15",
        title="Hazardous materials emergency alarm supervision",
        source_domain="IFC",
        task_type="hazardous_material_alarm_design",
        design_intent="Configure local alarm and supervision for a hazardous materials room while minimizing monitoring burden and maximizing occupant alert coverage.",
        scenario_facts={"hazardous_material_use": True, "emergency_alarm_context": "storage_room"},
        decision_variables={
            "local_alarm_provided_indicator": var("binary", "indicator", 0, 1),
            "supervision_monitoring_indicator": var("binary", "indicator", 0, 1),
            "monitoring_burden_score": var("continuous", "score", 0, 10),
            "occupant_alert_coverage_score": var("continuous", "score", 0, 10),
        },
        objectives=[
            {"name": "minimize_monitoring_burden", "expression": "monitoring_burden_score"},
            {"name": "maximize_occupant_alert_coverage", "expression": "occupant_alert_coverage_score"},
        ],
        query_preferences={"lambda": [0.45, 0.55], "meaning": "slightly prioritize occupant alert coverage"},
        source_rule_ids=["ifc-2021-5004-10-local-alarm", "ifc-2021-5004-10-supervision"],
        executable_constraints=[
            constraint("C1", "local_alarm_provided_indicator == 1", "local_alarm_activation", "ifc-2021-5004-10-local-alarm", ["local_alarm_provided_indicator"]),
            constraint("C2", "supervision_monitoring_indicator == 1", "alarm_supervision_monitoring", "ifc-2021-5004-10-supervision", ["supervision_monitoring_indicator"]),
        ],
        challenge_types=["multi_rule_conjunction"],
    ),
    build_task(
        omega_id="ARCH_FKG_16",
        title="Solvent distillation extinguisher placement",
        source_domain="IFC",
        task_type="hazardous_process_safety_design",
        design_intent="Plan portable extinguisher coverage for a solvent distillation area while keeping equipment count low and coverage robust.",
        scenario_facts={"process": "solvent_distillation", "hazardous_residue_present": True},
        decision_variables={
            "portable_fire_extinguisher_count": var("integer", "unit", 0, 6),
            "extinguisher_coverage_score": var("continuous", "score", 0, 10),
            "storage_compliance_indicator": var("binary", "indicator", 0, 1),
        },
        objectives=[
            {"name": "minimize_extinguisher_count", "expression": "portable_fire_extinguisher_count"},
            {"name": "maximize_extinguisher_coverage", "expression": "extinguisher_coverage_score"},
        ],
        query_preferences={"lambda": [0.45, 0.55], "meaning": "slightly prioritize coverage"},
        source_rule_ids=["ifc-2021-5705-4-9-portable-fire-extinguishers", "ifc-2021-5705-4-8-storage-hazardous-residue"],
        executable_constraints=[
            constraint("C1", "portable_fire_extinguisher_count >= 1", "portable_extinguisher_minimum", "ifc-2021-5705-4-9-portable-fire-extinguishers", ["portable_fire_extinguisher_count"]),
            constraint("C2", "storage_compliance_indicator == 1", "hazardous_residue_storage_compliance", "ifc-2021-5705-4-8-storage-hazardous-residue", ["storage_compliance_indicator"]),
        ],
        challenge_types=["multi_rule_conjunction"],
    ),
    build_task(
        omega_id="ARCH_FKG_17",
        title="Indoor cryogenic dispensing ventilation",
        source_domain="IFC",
        task_type="cryogenic_dispensing_room_design",
        design_intent="Design indoor cryogenic dispensing room ventilation and construction compliance while balancing exhaust capacity against compliance margin.",
        scenario_facts={"process": "indoor_cryogenic_fluid_dispensing", "leak_accumulation_risk": True},
        decision_variables={
            "mechanical_ventilation_indicator": var("binary", "indicator", 0, 1),
            "ibc_construction_compliance_indicator": var("binary", "indicator", 0, 1),
            "exhaust_capacity_score": var("continuous", "score", 0, 10),
            "energy_use_score": var("continuous", "score", 0, 10),
        },
        objectives=[
            {"name": "maximize_exhaust_capacity", "expression": "exhaust_capacity_score"},
            {"name": "minimize_energy_use", "expression": "energy_use_score"},
        ],
        query_preferences={"lambda": [0.55, 0.45], "meaning": "slightly prioritize leak-response ventilation capacity"},
        source_rule_ids=["ifc_5505_4_1_1_ventilation_requirement", "ifc_5505_4_1_building_code_compliance"],
        executable_constraints=[
            constraint("C1", "mechanical_ventilation_indicator == 1", "cryogenic_dispensing_ventilation_required", "ifc_5505_4_1_1_ventilation_requirement", ["mechanical_ventilation_indicator"]),
            constraint("C2", "ibc_construction_compliance_indicator == 1", "cryogenic_dispensing_ibc_construction_compliance", "ifc_5505_4_1_building_code_compliance", ["ibc_construction_compliance_indicator"]),
        ],
        challenge_types=["multi_rule_conjunction"],
    ),
    build_task(
        omega_id="ARCH_FKG_18",
        title="Class 3 oxidizer maintenance storage",
        source_domain="IFC",
        task_type="oxidizer_storage_design",
        design_intent="Choose maintenance oxidizer storage quantities while balancing operational inventory and cabinet burden.",
        scenario_facts={"hazardous_material": "Class 3 oxidizer", "use_case": "maintenance_operation"},
        decision_variables={
            "solid_oxidizer_storage_lb": var("continuous", "pound", 0, 300),
            "liquid_oxidizer_storage_gal": var("continuous", "gallon", 0, 40),
            "cabinet_count": var("integer", "count", 1, 10),
            "inventory_service_score": var("continuous", "score", 0, 10),
        },
        objectives=[
            {"name": "maximize_inventory_service", "expression": "inventory_service_score"},
            {"name": "minimize_cabinet_count", "expression": "cabinet_count"},
        ],
        query_preferences={"lambda": [0.5, 0.5], "meaning": "balance maintenance inventory and storage burden"},
        source_rule_ids=["ifc_6303_1_5_class3_oxidizer_storage_limit"],
        executable_constraints=[
            constraint("C1", "solid_oxidizer_storage_lb <= 220", "class3_oxidizer_solid_quantity_limit", "ifc_6303_1_5_class3_oxidizer_storage_limit", ["solid_oxidizer_storage_lb"]),
            constraint("C2", "liquid_oxidizer_storage_gal <= 22", "class3_oxidizer_liquid_quantity_limit", "ifc_6303_1_5_class3_oxidizer_storage_limit", ["liquid_oxidizer_storage_gal"]),
        ],
        challenge_types=["multi_constraint_single_rule"],
    ),
    build_task(
        omega_id="ARCH_FKG_19",
        title="Roof hatch guard extension",
        source_domain="IFC",
        task_type="roof_access_fall_protection_design",
        design_intent="Detail a roof hatch guard, balancing guard extension length against rooftop circulation clearance.",
        scenario_facts={
            "roof_hatch_near_edge": True,
            "height_differential_present": True,
            "roof_hatch_proximity_to_edge_ft": 10,
            "edge_height_above_floor_in": 42,
        },
        decision_variables={
            "guard_extension_beyond_hatch_end_in": var("continuous", "inch", 0, 60),
            "circulation_clearance_score": var("continuous", "score", 0, 10),
            "fall_protection_margin_in": var("continuous", "inch", 0, 30),
        },
        objectives=[
            {"name": "maximize_fall_protection_margin", "expression": "fall_protection_margin_in"},
            {"name": "maximize_rooftop_circulation_clearance", "expression": "circulation_clearance_score"},
        ],
        query_preferences={"lambda": [0.55, 0.45], "meaning": "slightly prioritize fall-protection margin"},
        source_rule_ids=["ifc_roof_hatch_guard_requirement"],
        executable_constraints=[
            constraint("C1", "guard_extension_beyond_hatch_end_in >= 30", "roof_hatch_guard_extension", "ifc_roof_hatch_guard_requirement", ["guard_extension_beyond_hatch_end_in"]),
            constraint("C2", "fall_protection_margin_in == guard_extension_beyond_hatch_end_in - 30", "guard_extension_margin_certificate", "scenario_roof_hatch_model", ["fall_protection_margin_in", "guard_extension_beyond_hatch_end_in"], source_type="task_or_scenario_model"),
        ],
        challenge_types=["parameter_limit"],
    ),
    build_task(
        omega_id="ARCH_FKG_20",
        title="Means-of-egress ramp slope and landing",
        source_domain="IFC",
        task_type="egress_ramp_design",
        design_intent="Choose ramp slope and landing dimensions for a means-of-egress ramp, balancing compact run length and usable landing space.",
        scenario_facts={"egress_component": "ramp", "change_in_direction": True},
        decision_variables={
            "running_slope": var("continuous", "ratio", 0.02, 0.12),
            "landing_minimum_dimension_in": var("continuous", "inch", 36, 84),
            "ramp_run_length_ft": var("continuous", "ft", 10, 120),
        },
        objectives=[
            {"name": "minimize_ramp_run_length", "expression": "ramp_run_length_ft"},
            {"name": "maximize_landing_dimension", "expression": "landing_minimum_dimension_in"},
        ],
        query_preferences={"lambda": [0.55, 0.45], "meaning": "prefer compact ramp length while keeping generous landings"},
        source_rule_ids=["ifc_ramp_slope_egress_1012_2", "ifc2021_ramp_landing_size_change_in_direction"],
        executable_constraints=[
            constraint("C1", "running_slope <= 0.08333333333333333", "egress_ramp_running_slope_limit", "ifc_ramp_slope_egress_1012_2", ["running_slope"]),
            constraint("C2", "landing_minimum_dimension_in >= 60", "ramp_landing_change_in_direction_dimension", "ifc2021_ramp_landing_size_change_in_direction", ["landing_minimum_dimension_in"]),
        ],
        challenge_types=["multi_rule_conjunction"],
    ),
    build_task(
        omega_id="ARCH_FKG_21",
        title="Existing ambulatory care most-restrictive upgrade",
        source_domain="IFC",
        task_type="existing_ambulatory_care_precedence_design",
        design_intent="Upgrade an existing ambulatory-care suite where a general Chapter 11 route allowance conflicts with the ambulatory-care provision, applying the most-restrictive-precedence rule while balancing retrofit burden and evacuation support.",
        scenario_facts={
            "building_type": "existing",
            "occupancy_subtype": "ambulatory_care",
            "non_self_preserving_recipients_count": 6,
            "chapter_11_candidate_clear_width_in": 36,
            "ambulatory_care_candidate_clear_width_in": 44,
            "most_restrictive_clear_width_in": 44,
        },
        decision_variables={
            "fire_partition_rating_hr": var("continuous", "hour", 0, 3),
            "selected_accessibility_clear_width_in": var("continuous", "inch", 32, 60),
            "retrofit_cost_score": var("continuous", "score", 0, 20),
            "evacuation_support_score": var("continuous", "score", 0, 20),
        },
        objectives=[
            {"name": "minimize_retrofit_cost", "expression": "retrofit_cost_score"},
            {"name": "maximize_evacuation_support", "expression": "evacuation_support_score"},
        ],
        query_preferences={"lambda": [0.5, 0.5], "meaning": "balance retrofit burden against evacuation support"},
        source_rule_ids=["ifc_k101_1_scope", "ifc_k102_1_separation"],
        executable_constraints=[
            constraint("C1", "fire_partition_rating_hr >= 1", "ambulatory_care_fire_partition_separation", "ifc_k102_1_separation", ["fire_partition_rating_hr"]),
            constraint(
                "C2",
                "selected_accessibility_clear_width_in >= most_restrictive_clear_width_in",
                "most_restrictive_requirement_survives",
                "ifc_k101_1_scope",
                ["selected_accessibility_clear_width_in"],
                ["most_restrictive_clear_width_in"],
                metadata=semantic_indicator_metadata(
                    "ifc_k101_1_scope",
                    "The textual precedence clause is encoded as the selected clear width meeting the most restrictive applicable candidate requirement.",
                ),
            ),
        ],
        structure_only_constraints=[
            structure_constraint(
                "S1",
                "the ambulatory-care clear-width candidate is the winning requirement when it is stricter than the general Chapter 11 candidate",
                "most_restrictive_precedence_resolution",
                "ifc_k101_1_scope",
            )
        ],
        challenge_types=["precedence", "multi_rule_conjunction", "scenario_conditioned_applicability"],
    ),
    build_task(
        omega_id="ARCH_FKG_22",
        title="Emergency voice alarm use precedence",
        source_domain="IFC",
        task_type="emergency_voice_alarm_precedence_design",
        design_intent="Configure an emergency voice/alarm communication system that supports selective and all-call paging while ensuring manual fire alarm use takes precedence over other announcement uses.",
        scenario_facts={
            "system_type": "emergency_voice_alarm_communication_system",
            "alternative_announcement_use_requested": True,
            "shared_voice_alarm_channel_capacity": 6,
        },
        decision_variables={
            "selective_paging_indicator": var("binary", "indicator", 0, 1),
            "all_call_paging_indicator": var("binary", "indicator", 0, 1),
            "manual_fire_alarm_priority_indicator": var("binary", "indicator", 0, 1),
            "reserved_alarm_priority_channels": var("integer", "count", 1, 6),
            "alternative_announcement_channel_count": var("integer", "count", 0, 6),
            "operator_flexibility_score": var("continuous", "score", 0, 12),
            "alarm_preemption_reliability_score": var("continuous", "score", 0, 16),
        },
        objectives=[
            {"name": "maximize_operator_flexibility", "expression": "operator_flexibility_score"},
            {"name": "maximize_alarm_preemption_reliability", "expression": "alarm_preemption_reliability_score"},
        ],
        query_preferences={"lambda": [0.45, 0.55], "meaning": "slightly prioritize alarm preemption reliability"},
        source_rule_ids=["ifc-907.5.2.2.1-paging-zones-capability", "ifc-907.5.2.2.3-precedence-of-fire-alarm-use"],
        executable_constraints=[
            constraint("C1", "selective_paging_indicator == 1", "selective_paging_capability", "ifc-907.5.2.2.1-paging-zones-capability", ["selective_paging_indicator"]),
            constraint("C2", "all_call_paging_indicator == 1", "all_call_paging_capability", "ifc-907.5.2.2.1-paging-zones-capability", ["all_call_paging_indicator"]),
            constraint("C3", "manual_fire_alarm_priority_indicator == 1", "manual_fire_alarm_use_precedence", "ifc-907.5.2.2.3-precedence-of-fire-alarm-use", ["manual_fire_alarm_priority_indicator"]),
            constraint(
                "C4",
                "reserved_alarm_priority_channels + alternative_announcement_channel_count <= shared_voice_alarm_channel_capacity",
                "shared_channel_capacity_model",
                "scenario_voice_alarm_model",
                ["reserved_alarm_priority_channels", "alternative_announcement_channel_count"],
                ["shared_voice_alarm_channel_capacity"],
                source_type="task_or_scenario_model",
            ),
        ],
        structure_only_constraints=[
            structure_constraint(
                "S1",
                "manual fire alarm use overrides alternative announcement use when both use the emergency voice alarm system",
                "manual_fire_alarm_precedence_resolution",
                "ifc-907.5.2.2.3-precedence-of-fire-alarm-use",
            )
        ],
        challenge_types=["precedence", "override_resolution", "multi_rule_conjunction"],
    ),
    build_task(
        omega_id="ARCH_FKG_23",
        title="Construction-site firefighting access",
        source_domain="IFC",
        task_type="construction_site_access_design",
        design_intent="Plan construction-site access so firefighting vehicles can reach the work area while reducing temporary roadwork.",
        scenario_facts={"site_phase": "construction", "active_building_area": "central_core"},
        decision_variables={
            "fire_fighting_vehicle_access_distance_ft": var("continuous", "ft", 20, 180),
            "temporary_road_length_ft": var("continuous", "ft", 100, 1200),
            "staging_area_score": var("continuous", "score", 0, 10),
            "emergency_contact_posting_indicator": var("binary", "indicator", 0, 1),
        },
        objectives=[
            {"name": "minimize_temporary_road_length", "expression": "temporary_road_length_ft"},
            {"name": "maximize_staging_area_score", "expression": "staging_area_score"},
        ],
        query_preferences={"lambda": [0.55, 0.45], "meaning": "prefer less temporary work while keeping fire access practical"},
        source_rule_ids=["ifc-3311.1-vehicle-access-distance", "ifc-3311.1-emergency-contact-posting-alternative"],
        executable_constraints=[
            constraint("C1", "fire_fighting_vehicle_access_distance_ft <= 100", "construction_fire_vehicle_access_distance", "ifc-3311.1-vehicle-access-distance", ["fire_fighting_vehicle_access_distance_ft"]),
            constraint("C2", "emergency_contact_posting_indicator == 1", "emergency_contact_posting_or_equivalent", "ifc-3311.1-emergency-contact-posting-alternative", ["emergency_contact_posting_indicator"]),
        ],
        challenge_types=["multi_rule_conjunction"],
    ),
    build_task(
        omega_id="ARCH_FKG_24",
        title="Accessible toilet compartment door operation",
        source_domain="ADA",
        task_type="accessible_toilet_compartment_design",
        design_intent="Lay out an accessible toilet compartment door while balancing partition compactness against latch-side clearance and independent operation.",
        scenario_facts={"facility_type": "public_restroom", "compartment_type": "accessible_toilet_compartment"},
        decision_variables={
            "latch_side_clearance_in": var("continuous", "inch", 24, 60),
            "self_closing_indicator": var("binary", "indicator", 0, 1),
            "dual_side_pull_indicator": var("binary", "indicator", 0, 1),
            "partition_footprint_score": var("continuous", "score", 0, 10),
            "accessible_operation_score": var("continuous", "score", 0, 10),
        },
        objectives=[
            {"name": "minimize_partition_footprint", "expression": "partition_footprint_score"},
            {"name": "maximize_accessible_operation", "expression": "accessible_operation_score"},
        ],
        query_preferences={"lambda": [0.45, 0.55], "meaning": "slightly prioritize accessible operation"},
        source_rule_ids=[
            "ada-toilet-compartment-door-latch-side-clearance",
            "ada-toilet-compartment-door-self-closing",
            "ada-toilet-compartment-door-pull-placement",
        ],
        executable_constraints=[
            constraint("C1", "latch_side_clearance_in >= 42", "toilet_compartment_latch_side_clearance", "ada-toilet-compartment-door-latch-side-clearance", ["latch_side_clearance_in"]),
            constraint("C2", "self_closing_indicator == 1", "toilet_compartment_self_closing_door", "ada-toilet-compartment-door-self-closing", ["self_closing_indicator"]),
            constraint("C3", "dual_side_pull_indicator == 1", "toilet_compartment_dual_side_door_pull", "ada-toilet-compartment-door-pull-placement", ["dual_side_pull_indicator"]),
        ],
        challenge_types=["multi_rule_conjunction", "accessibility_operability"],
    ),
    build_task(
        omega_id="ARCH_FKG_25",
        title="Roll-in shower control and distribution layout",
        source_domain="ADA",
        task_type="accessible_shower_room_design",
        design_intent="Place controls and allocate roll-in showers in gender-separated facilities while balancing plumbing compactness against accessible usability.",
        scenario_facts={
            "facility_type": "locker_room",
            "shower_facilities_separated_by_gender": True,
            "shower_type": "standard roll-in type shower compartment",
        },
        decision_variables={
            "control_height_in": var("continuous", "inch", 30, 60),
            "rollin_showers_per_gender_group": var("integer", "count", 0, 4),
            "plumbing_compactness_score": var("continuous", "score", 0, 10),
            "user_reachability_score": var("continuous", "score", 0, 10),
        },
        objectives=[
            {"name": "maximize_plumbing_compactness", "expression": "plumbing_compactness_score"},
            {"name": "maximize_user_reachability", "expression": "user_reachability_score"},
        ],
        query_preferences={"lambda": [0.45, 0.55], "meaning": "slightly prioritize user reachability"},
        source_rule_ids=["ada-2010-608-5-2-control-height", "ada_shower_distribution_gender_separated_facilities"],
        executable_constraints=[
            constraint("C1", "control_height_in <= 48", "roll_in_shower_control_height", "ada-2010-608-5-2-control-height", ["control_height_in"]),
            constraint("C2", "rollin_showers_per_gender_group >= 1", "roll_in_shower_distribution", "ada_shower_distribution_gender_separated_facilities", ["rollin_showers_per_gender_group"]),
        ],
        challenge_types=["multi_rule_conjunction", "accessibility_reach_range"],
    ),
    build_task(
        omega_id="ARCH_FKG_26",
        title="Accessible storage clear-floor space",
        source_domain="ADA",
        task_type="accessible_storage_layout_design",
        design_intent="Configure storage in an accessible room while balancing storage capacity against clear-floor maneuvering space.",
        scenario_facts={"room_type": "accessible_mobility_guest_room", "storage_elements_provided": True},
        decision_variables={
            "clear_floor_width_in": var("continuous", "inch", 24, 60),
            "clear_floor_depth_in": var("continuous", "inch", 36, 72),
            "accessible_storage_type_count": var("integer", "count", 0, 6),
            "storage_capacity_score": var("continuous", "score", 0, 10),
            "maneuvering_quality_score": var("continuous", "score", 0, 10),
        },
        objectives=[
            {"name": "maximize_storage_capacity", "expression": "storage_capacity_score"},
            {"name": "maximize_maneuvering_quality", "expression": "maneuvering_quality_score"},
        ],
        query_preferences={"lambda": [0.5, 0.5], "meaning": "balance storage capacity and accessible maneuvering space"},
        source_rule_ids=["ada-305-3-clear-floor-space-min-dimensions", "ada_225_2_storage_accessibility"],
        executable_constraints=[
            constraint("C1", "clear_floor_width_in >= 30", "clear_floor_space_width", "ada-305-3-clear-floor-space-min-dimensions", ["clear_floor_width_in"]),
            constraint("C2", "clear_floor_depth_in >= 48", "clear_floor_space_depth", "ada-305-3-clear-floor-space-min-dimensions", ["clear_floor_depth_in"]),
            constraint("C3", "accessible_storage_type_count >= 1", "accessible_storage_type_minimum", "ada_225_2_storage_accessibility", ["accessible_storage_type_count"]),
        ],
        challenge_types=["multi_rule_conjunction", "accessibility_space_planning"],
    ),
    build_task(
        omega_id="ARCH_FKG_27",
        title="Accessible visual sign character proportion",
        source_domain="ADA",
        task_type="accessible_wayfinding_sign_design",
        design_intent="Choose visual character proportions for accessible wayfinding signs while balancing panel compactness against readability.",
        scenario_facts={"sign_type": "visual_wayfinding", "viewer_group": "public_accessible_route"},
        decision_variables={
            "font_width_to_height_ratio": var("continuous", "dimensionless", 0.35, 1.35),
            "sign_panel_width_score": var("continuous", "score", 0, 10),
            "readability_score": var("continuous", "score", 0, 10),
        },
        objectives=[
            {"name": "minimize_sign_panel_width", "expression": "sign_panel_width_score"},
            {"name": "maximize_readability", "expression": "readability_score"},
        ],
        query_preferences={"lambda": [0.45, 0.55], "meaning": "slightly prioritize readability"},
        source_rule_ids=["ada-2010-703-5-5-font-proportion"],
        executable_constraints=[
            constraint("C1", "font_width_to_height_ratio >= 0.55", "visual_character_minimum_width_to_height_ratio", "ada-2010-703-5-5-font-proportion", ["font_width_to_height_ratio"]),
            constraint("C2", "font_width_to_height_ratio <= 1.1", "visual_character_maximum_width_to_height_ratio", "ada-2010-703-5-5-font-proportion", ["font_width_to_height_ratio"]),
        ],
        challenge_types=["multi_constraint_single_rule", "accessibility_wayfinding"],
    ),
    build_task(
        omega_id="ARCH_FKG_28",
        title="Accessible ramp landing footprint",
        source_domain="ADA",
        task_type="accessible_ramp_landing_design",
        design_intent="Size a constrained ramp landing while balancing footprint reduction against wheelchair maneuvering margin.",
        scenario_facts={"route_type": "accessible_ramp", "landing_condition": "permitted_alternative_size"},
        decision_variables={
            "landing_length_in": var("continuous", "inch", 36, 84),
            "landing_width_in": var("continuous", "inch", 48, 84),
            "landing_footprint_score": var("continuous", "score", 0, 10),
            "maneuvering_margin_score": var("continuous", "score", 0, 10),
        },
        objectives=[
            {"name": "minimize_landing_footprint", "expression": "landing_footprint_score"},
            {"name": "maximize_maneuvering_margin", "expression": "maneuvering_margin_score"},
        ],
        query_preferences={"lambda": [0.5, 0.5], "meaning": "balance compact footprint and maneuvering margin"},
        source_rule_ids=["ada_ramp_landing_min_size_permitted_alt"],
        executable_constraints=[
            constraint("C1", "landing_length_in >= 48", "ramp_landing_minimum_length", "ada_ramp_landing_min_size_permitted_alt", ["landing_length_in"]),
            constraint("C2", "landing_width_in >= 60", "ramp_landing_minimum_width", "ada_ramp_landing_min_size_permitted_alt", ["landing_width_in"]),
        ],
        challenge_types=["multi_constraint_single_rule", "accessibility_route_design"],
    ),
    build_task(
        omega_id="ARCH_FKG_29",
        title="Accessible egress elevator resilience",
        source_domain="IBC",
        task_type="accessible_means_of_egress_design",
        design_intent="Configure accessible egress support for a multistory building while balancing elevator-core impact against life-safety resilience.",
        scenario_facts={"occupied_floors_above_exit_discharge": 4, "accessible_means_of_egress_required": True},
        decision_variables={
            "accessible_egress_elevator_count": var("integer", "count", 0, 4),
            "emergency_power_system_indicator": var("binary", "indicator", 0, 1),
            "elevator_core_impact_score": var("continuous", "score", 0, 10),
            "life_safety_resilience_score": var("continuous", "score", 0, 10),
        },
        objectives=[
            {"name": "minimize_elevator_core_impact", "expression": "elevator_core_impact_score"},
            {"name": "maximize_life_safety_resilience", "expression": "life_safety_resilience_score"},
        ],
        query_preferences={"lambda": [0.45, 0.55], "meaning": "slightly prioritize life-safety resilience"},
        source_rule_ids=["ibc_1009_2_1_elevator_requirement", "ibc2021_emergency_power_requirement"],
        executable_constraints=[
            constraint("C1", "accessible_egress_elevator_count >= 1", "accessible_means_egress_elevator_minimum", "ibc_1009_2_1_elevator_requirement", ["accessible_egress_elevator_count"]),
            constraint("C2", "emergency_power_system_indicator == 1", "emergency_power_for_life_safety_loads", "ibc2021_emergency_power_requirement", ["emergency_power_system_indicator"]),
        ],
        challenge_types=["multi_rule_conjunction", "life_safety_system_design"],
    ),
    build_task(
        omega_id="ARCH_FKG_30",
        title="Single-exit tenant layout documentation",
        source_domain="IBC",
        task_type="egress_layout_and_occupant_load_design",
        design_intent="Evaluate a compact tenant layout using single-exit logic while balancing usable area against occupant-load substantiation quality.",
        scenario_facts={
            "tenant_area_type": "small residential suite",
            "occupancy_group": "group r-2 occupancy",
            "scenario.occupancy_group": "group r-2 occupancy",
            "story_condition": "first story",
            "exit_discharge_or_public_way_continuity_required": True,
            "single_exit_strategy_requested": True,
        },
        decision_variables={
            "exit_count": var("integer", "count", 1, 3),
            "proposed_occupant_load_factor": var("continuous", "occupant_per_sf", 0.1, 1.5),
            "approved_diagram_indicator": var("binary", "indicator", 0, 1),
            "usable_area_score": var("continuous", "score", 0, 10),
            "egress_verification_score": var("continuous", "score", 0, 10),
        },
        objectives=[
            {"name": "maximize_usable_area", "expression": "usable_area_score"},
            {"name": "maximize_egress_verification", "expression": "egress_verification_score"},
        ],
        query_preferences={"lambda": [0.5, 0.5], "meaning": "balance usable area and egress verification confidence"},
        source_rule_ids=["ibc_1006_3_4_single_exit_permitted", "ibc_1004_6_occupant_load_increase"],
        executable_constraints=[
            constraint("C1", "exit_count == 1", "single_exit_strategy", "ibc_1006_3_4_single_exit_permitted", ["exit_count"]),
            constraint("C2", "proposed_occupant_load_factor <= 1", "occupant_load_factor_limit", "ibc_1004_6_occupant_load_increase", ["proposed_occupant_load_factor"]),
            constraint("C3", "approved_diagram_indicator == 1", "occupant_load_diagram_substantiation", "ibc_1004_6_occupant_load_increase", ["approved_diagram_indicator"]),
        ],
        challenge_types=["applicability_resolution", "documentation_condition"],
    ),
]


def objective_constraint(
    constraint_id: str,
    expression: str,
    role: str,
    decision_variables: list[str],
    note: str,
    scenario_fields: list[str] | None = None,
) -> dict[str, Any]:
    return constraint(
        constraint_id,
        expression,
        role,
        "scenario_objective_model",
        decision_variables,
        scenario_fields or [],
        source_type="task_or_scenario_model",
        metadata=objective_closure_metadata(note),
    )


OBJECTIVE_CLOSURE_CONSTRAINTS: dict[str, list[dict[str, Any]]] = {
    "ARCH_FKG_01": [
        objective_constraint("C90", "operability_score <= 10 - 0.4 * (latch_release_height_in - 36)", "adult_operability_decreases_with_latch_height", ["operability_score", "latch_release_height_in"], "Higher latch placement improves child resistance but reduces adult operability."),
    ],
    "ARCH_FKG_02": [
        objective_constraint("C90", "sign_panel_area_sf >= character_height_in * stroke_width_in / 2", "sign_area_scales_with_letter_geometry", ["sign_panel_area_sf", "character_height_in", "stroke_width_in"], "Compact sign area is constrained by selected character height and stroke width."),
        objective_constraint("C91", "legibility_margin_score <= 2 * (character_height_in - 4) + 5 * (stroke_width_in - 0.5)", "legibility_margin_from_character_size", ["legibility_margin_score", "character_height_in", "stroke_width_in"], "Emergency legibility margin grows only when character height or stroke width exceeds the minimum."),
    ],
    "ARCH_FKG_03": [
        objective_constraint("C90", "site_loop_length_ft >= 0.25 * available_fire_flow_gpm + 100 * flow_duration_hours", "site_loop_length_scales_with_fire_flow", ["site_loop_length_ft", "available_fire_flow_gpm", "flow_duration_hours"], "Higher fire-flow and duration reserve require a larger site water loop proxy."),
    ],
    "ARCH_FKG_04": [
        objective_constraint("C90", "service_connection_size_score >= available_fire_flow_gpm / 300 + flow_duration_hours", "service_size_scales_with_fire_flow", ["service_connection_size_score", "available_fire_flow_gpm", "flow_duration_hours"], "Compact service sizing is coupled to selected fire-flow capacity and duration."),
    ],
    "ARCH_FKG_05": [
        objective_constraint("C90", "storage_density_score <= 12 - 2 * selected_commodity_class_rank - 0.3 * classification_documentation_level", "storage_density_trades_with_higher_class_and_documentation", ["storage_density_score", "selected_commodity_class_rank", "classification_documentation_level"], "Higher governing commodity classification and stronger classification documentation reduce the allowable storage density proxy."),
        objective_constraint("C91", "classification_confidence_score <= classification_documentation_level + selected_commodity_class_rank", "classification_confidence_from_documentation", ["classification_confidence_score", "classification_documentation_level", "selected_commodity_class_rank"], "Classification confidence grows with documentation effort and the selected governing class."),
    ],
    "ARCH_FKG_06": [
        objective_constraint("C90", "cooling_load_score >= 0.1 * (125 - maximum_operating_temperature_f)", "cooling_load_increases_with_temperature_margin", ["cooling_load_score", "maximum_operating_temperature_f"], "More conservative operating temperature increases cooling-load burden."),
    ],
    "ARCH_FKG_07": [
        objective_constraint("C90", "opening_area_score >= door_clear_width_in * door_opening_height_in / 200", "opening_area_scales_with_door_size", ["opening_area_score", "door_clear_width_in", "door_opening_height_in"], "Door opening area burden is tied to selected clear width and height."),
    ],
    "ARCH_FKG_08": [
        objective_constraint("C90", "retrofit_cost_score >= 0.5 * (door_clear_width_in - 32)", "retrofit_cost_scales_with_added_width", ["retrofit_cost_score", "door_clear_width_in"], "Wider existing-door retrofit increases cost proxy."),
    ],
    "ARCH_FKG_09": [
        objective_constraint("C90", "wheelchair_spaces_count >= 1", "minimum_refuge_wheelchair_space", ["wheelchair_spaces_count"], "The refuge layout must provide at least one wheelchair space when the refuge rule is activated."),
        objective_constraint("C91", "refuge_area_sf >= 30 * wheelchair_spaces_count", "refuge_area_scales_with_wheelchair_spaces", ["refuge_area_sf", "wheelchair_spaces_count"], "Refuge area is coupled to the number of wheelchair spaces."),
    ],
    "ARCH_FKG_10": [
        objective_constraint("C90", "zone_count >= zone_area_sf / 22500", "zone_count_scales_with_total_zone_area", ["zone_count", "zone_area_sf"], "The number of zones is coupled to the selected total protected area."),
        objective_constraint("C91", "coverage_per_zone_score <= zone_area_sf / (1000 * zone_count)", "coverage_per_zone_depends_on_zone_count", ["coverage_per_zone_score", "zone_area_sf", "zone_count"], "Coverage per zone decreases when the same area is split across more zones."),
    ],
    "ARCH_FKG_11": [
        objective_constraint("C90", "occupant_discomfort_score >= sound_pressure_level_dba - 75", "discomfort_scales_with_alarm_spl", ["occupant_discomfort_score", "sound_pressure_level_dba"], "Occupant discomfort is coupled to selected alarm sound pressure level."),
    ],
    "ARCH_FKG_12": [
        objective_constraint("C90", "weather_resilience_score <= landing_height_differential_in + (1 - door_bottom_clearance_in)", "weather_resilience_from_threshold_geometry", ["weather_resilience_score", "landing_height_differential_in", "door_bottom_clearance_in"], "Weather resilience is tied to threshold differential and door-bottom clearance choices."),
    ],
    "ARCH_FKG_13": [
        objective_constraint("C90", "covered_area_score <= aggregate_tent_area_sf / 70", "covered_area_score_scales_with_tent_area", ["covered_area_score", "aggregate_tent_area_sf"], "Covered-area utility is coupled to aggregate tent area."),
        objective_constraint("C91", "aggregate_tent_area_sf + 20 * minimum_clearance_ft <= 1000", "site_area_tradeoff_between_tent_area_and_clearance", ["aggregate_tent_area_sf", "minimum_clearance_ft"], "Finite event-site area creates a trade-off between tent area and clearance margin."),
    ],
    "ARCH_FKG_14": [
        objective_constraint("C90", "storage_density_score <= maximum_storage_height_ft / 2 - transverse_flue_space_width_in", "storage_density_from_height_and_flue_space", ["storage_density_score", "maximum_storage_height_ft", "transverse_flue_space_width_in"], "Storage density increases with height and decreases with wider flue spaces."),
        objective_constraint("C91", "flue_space_loss_score >= transverse_flue_space_width_in", "flue_space_loss_scales_with_width", ["flue_space_loss_score", "transverse_flue_space_width_in"], "Flue-space loss is tied to selected transverse flue width."),
    ],
    "ARCH_FKG_15": [
        objective_constraint("C90", "monitoring_burden_score >= local_alarm_provided_indicator + supervision_monitoring_indicator", "monitoring_burden_from_alarm_supervision", ["monitoring_burden_score", "local_alarm_provided_indicator", "supervision_monitoring_indicator"], "Alarm and supervision requirements increase monitoring burden."),
        objective_constraint("C91", "occupant_alert_coverage_score <= 5 * (local_alarm_provided_indicator + supervision_monitoring_indicator)", "alert_coverage_from_alarm_supervision", ["occupant_alert_coverage_score", "local_alarm_provided_indicator", "supervision_monitoring_indicator"], "Alert coverage depends on local alarm and supervision provision."),
    ],
    "ARCH_FKG_16": [
        objective_constraint("C90", "extinguisher_coverage_score <= 5 * portable_fire_extinguisher_count", "coverage_scales_with_extinguisher_count", ["extinguisher_coverage_score", "portable_fire_extinguisher_count"], "Extinguisher coverage is coupled to extinguisher count."),
    ],
    "ARCH_FKG_17": [
        objective_constraint("C90", "exhaust_capacity_score <= 10 * mechanical_ventilation_indicator", "exhaust_capacity_requires_mechanical_ventilation", ["exhaust_capacity_score", "mechanical_ventilation_indicator"], "Exhaust capacity is available only when mechanical ventilation is provided."),
        objective_constraint("C91", "energy_use_score >= 0.6 * exhaust_capacity_score", "energy_use_scales_with_exhaust_capacity", ["energy_use_score", "exhaust_capacity_score"], "Higher exhaust capacity carries an energy-use cost."),
    ],
    "ARCH_FKG_18": [
        objective_constraint("C90", "inventory_service_score <= solid_oxidizer_storage_lb / 22 + liquid_oxidizer_storage_gal / 2.2", "inventory_service_from_stored_quantity", ["inventory_service_score", "solid_oxidizer_storage_lb", "liquid_oxidizer_storage_gal"], "Inventory service is coupled to selected stored oxidizer quantities."),
        objective_constraint("C91", "cabinet_count >= solid_oxidizer_storage_lb / 110 + liquid_oxidizer_storage_gal / 11", "cabinet_count_scales_with_stored_quantity", ["cabinet_count", "solid_oxidizer_storage_lb", "liquid_oxidizer_storage_gal"], "Storage cabinet count grows with solid and liquid oxidizer quantity."),
    ],
    "ARCH_FKG_19": [
        objective_constraint("C90", "circulation_clearance_score <= 10 - 0.2 * guard_extension_beyond_hatch_end_in", "circulation_clearance_decreases_with_guard_extension", ["circulation_clearance_score", "guard_extension_beyond_hatch_end_in"], "Longer guard extension improves fall-protection margin but reduces rooftop circulation clearance."),
    ],
    "ARCH_FKG_20": [
        objective_constraint("C90", "ramp_run_length_ft >= 12 * landing_minimum_dimension_in * running_slope", "ramp_run_length_from_slope_and_landing", ["ramp_run_length_ft", "landing_minimum_dimension_in", "running_slope"], "Ramp run length is coupled to selected slope and landing dimension."),
    ],
    "ARCH_FKG_21": [
        objective_constraint("C90", "retrofit_cost_score >= 0.25 * selected_accessibility_clear_width_in + 2 * fire_partition_rating_hr", "retrofit_cost_from_width_and_partition_rating", ["retrofit_cost_score", "selected_accessibility_clear_width_in", "fire_partition_rating_hr"], "Wider clear routes and higher fire-partition ratings increase retrofit burden."),
        objective_constraint("C91", "evacuation_support_score <= 0.35 * (selected_accessibility_clear_width_in - 32) + 4 * fire_partition_rating_hr", "evacuation_support_from_width_and_partition_rating", ["evacuation_support_score", "selected_accessibility_clear_width_in", "fire_partition_rating_hr"], "Evacuation support is tied to surplus clear width and fire-partition rating."),
    ],
    "ARCH_FKG_22": [
        objective_constraint("C90", "operator_flexibility_score <= 2 * alternative_announcement_channel_count", "operator_flexibility_from_alternative_channels", ["operator_flexibility_score", "alternative_announcement_channel_count"], "Operator flexibility is tied to the number of non-alarm announcement channels left available."),
        objective_constraint("C91", "alarm_preemption_reliability_score <= 2 * reserved_alarm_priority_channels + 4 * manual_fire_alarm_priority_indicator", "preemption_reliability_from_reserved_priority_capacity", ["alarm_preemption_reliability_score", "reserved_alarm_priority_channels", "manual_fire_alarm_priority_indicator"], "Alarm preemption reliability grows with reserved priority capacity and the required manual alarm priority behavior."),
    ],
    "ARCH_FKG_23": [
        objective_constraint("C90", "temporary_road_length_ft >= fire_fighting_vehicle_access_distance_ft", "temporary_road_length_covers_fire_access", ["temporary_road_length_ft", "fire_fighting_vehicle_access_distance_ft"], "Temporary road length is coupled to required firefighting access distance."),
        objective_constraint("C91", "staging_area_score <= 10 - 0.02 * temporary_road_length_ft", "staging_area_decreases_with_temporary_road", ["staging_area_score", "temporary_road_length_ft"], "More temporary road consumes construction staging area."),
    ],
    "ARCH_FKG_24": [
        objective_constraint("C90", "partition_footprint_score >= 0.1 * latch_side_clearance_in + 2 * self_closing_indicator + 2 * dual_side_pull_indicator", "partition_footprint_from_accessible_door_features", ["partition_footprint_score", "latch_side_clearance_in", "self_closing_indicator", "dual_side_pull_indicator"], "Accessible door features and clearance increase partition footprint."),
        objective_constraint("C91", "accessible_operation_score <= latch_side_clearance_in / 6 + 2 * self_closing_indicator + 2 * dual_side_pull_indicator", "accessible_operation_from_clearance_and_door_features", ["accessible_operation_score", "latch_side_clearance_in", "self_closing_indicator", "dual_side_pull_indicator"], "Accessible operation score is tied to clearance and required door features."),
    ],
    "ARCH_FKG_25": [
        objective_constraint("C90", "plumbing_compactness_score <= 10 - rollin_showers_per_gender_group - 0.05 * (48 - control_height_in)", "plumbing_compactness_from_shower_count_and_control_height", ["plumbing_compactness_score", "rollin_showers_per_gender_group", "control_height_in"], "More accessible shower provision and lower controls reduce plumbing compactness."),
        objective_constraint("C91", "user_reachability_score <= (48 - control_height_in) / 2 + 2 * rollin_showers_per_gender_group", "reachability_from_control_height_and_shower_distribution", ["user_reachability_score", "control_height_in", "rollin_showers_per_gender_group"], "User reachability is tied to control height and roll-in shower availability."),
    ],
    "ARCH_FKG_26": [
        objective_constraint("C90", "storage_capacity_score <= 2 * accessible_storage_type_count - 0.02 * (clear_floor_width_in + clear_floor_depth_in)", "storage_capacity_trades_with_clear_floor_space", ["storage_capacity_score", "accessible_storage_type_count", "clear_floor_width_in", "clear_floor_depth_in"], "Accessible clear floor space consumes storage capacity."),
        objective_constraint("C91", "maneuvering_quality_score <= 0.1 * (clear_floor_width_in - 30) + 0.1 * (clear_floor_depth_in - 48)", "maneuvering_quality_from_clear_floor_space", ["maneuvering_quality_score", "clear_floor_width_in", "clear_floor_depth_in"], "Maneuvering quality is tied to clear floor width and depth beyond minimums."),
    ],
    "ARCH_FKG_27": [
        objective_constraint("C90", "sign_panel_width_score >= 5 * font_width_to_height_ratio", "sign_width_scales_with_font_proportion", ["sign_panel_width_score", "font_width_to_height_ratio"], "Wider font proportions increase sign-panel width."),
        objective_constraint("C91", "readability_score <= 10 - 10 * abs(font_width_to_height_ratio - 0.8)", "readability_from_font_proportion", ["readability_score", "font_width_to_height_ratio"], "Readability peaks near a balanced font proportion and drops near extremes."),
    ],
    "ARCH_FKG_28": [
        objective_constraint("C90", "landing_footprint_score >= landing_length_in * landing_width_in / 500", "landing_footprint_from_dimensions", ["landing_footprint_score", "landing_length_in", "landing_width_in"], "Landing footprint is tied to landing length and width."),
        objective_constraint("C91", "maneuvering_margin_score <= (landing_length_in - 48 + landing_width_in - 60) / 4", "maneuvering_margin_from_landing_dimensions", ["maneuvering_margin_score", "landing_length_in", "landing_width_in"], "Maneuvering margin is tied to landing dimension surplus."),
    ],
    "ARCH_FKG_29": [
        objective_constraint("C90", "elevator_core_impact_score >= 3 * accessible_egress_elevator_count + 2 * emergency_power_system_indicator", "core_impact_from_elevator_and_power", ["elevator_core_impact_score", "accessible_egress_elevator_count", "emergency_power_system_indicator"], "Elevator and emergency power provisions increase core impact."),
        objective_constraint("C91", "life_safety_resilience_score <= 5 * accessible_egress_elevator_count + 5 * emergency_power_system_indicator", "resilience_from_elevator_and_power", ["life_safety_resilience_score", "accessible_egress_elevator_count", "emergency_power_system_indicator"], "Life-safety resilience is tied to accessible egress elevators and emergency power."),
    ],
    "ARCH_FKG_30": [
        objective_constraint("C90", "usable_area_score <= 10 - 2 * exit_count - proposed_occupant_load_factor", "usable_area_from_exit_and_load_strategy", ["usable_area_score", "exit_count", "proposed_occupant_load_factor"], "Single-exit and occupant-load choices constrain usable-area score."),
        objective_constraint("C91", "egress_verification_score <= 4 * exit_count + 4 * approved_diagram_indicator + 2 * (1 - proposed_occupant_load_factor)", "egress_verification_from_exit_and_documentation", ["egress_verification_score", "exit_count", "approved_diagram_indicator", "proposed_occupant_load_factor"], "Egress verification is tied to exit count, approved documentation, and occupant-load factor."),
    ],
}


def make_algorithm_input(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "omega_id": spec["omega_id"],
        "title": spec["title"],
        "domain": "architecture_code_compliance",
        "source_domain": spec["source_domain"],
        "task_type": spec["task_type"],
        "design_intent": spec["design_intent"],
        "scenario_facts": spec["scenario_facts"],
        "decision_variables": spec["decision_variables"],
        "objectives": spec["objectives"],
        "query_preferences": spec["query_preferences"],
        "public_scenario_model": {
            "model_id": f"{spec['omega_id']}_scenario_model",
            "path": "scenario_models/architecture_public_scenario_models.json",
            "visibility": "public_algorithm_input",
            "purpose": "Non-normative task physics/objective-closure constraints visible to optimizers; contains no expected rule IDs or labels.",
        },
        "visible_input_note": "Visible task input only. Algorithms may also read the public scenario model and rule library. Rule labels, rule-derived feasible-region answers, and rule-id bindings remain hidden evaluation references.",
    }


def make_reference(spec: dict[str, Any], rule_lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    expected_ids = set(spec["source_rule_ids"])
    expected_ids.update(collect_rule_library_source_ids(spec["executable_constraints"]))
    expected_ids.update(collect_rule_library_source_ids(spec["structure_only_constraints"]))
    provenance = rule_provenance(rule_lookup, spec["source_rule_ids"])
    feasible = {
        "executable_constraints": spec["executable_constraints"],
        "structure_only_constraints": spec["structure_only_constraints"],
        "valid_constraint_cells": spec["valid_constraint_cells"],
        "reference_semantics": {
            "positive_membership_condition": "all executable_constraints evaluate true and at least one valid cell holds when cells are present",
            "structure_only_constraints_usage": "used to check rule resolution, provenance, exceptions, or semantic requirements before numeric membership checking",
        },
    }
    return {
        "omega_id": spec["omega_id"],
        "title": spec["title"],
        "rule_structure": {
            "expected_source_rule_ids": spec["source_rule_ids"],
            "expected_defeated_rule_ids": spec["defeated_rule_ids"],
            "expected_surviving_rule_ids": spec["surviving_rule_ids"],
            "expected_valid_rule_structures": [spec["surviving_rule_ids"]],
            "expected_rule_behavior": {
                "should_activate": [rule_lookup[rule_id].get("name", rule_id) for rule_id in spec["surviving_rule_ids"]],
                "should_exclude": [rule_lookup[rule_id].get("name", rule_id) for rule_id in spec["defeated_rule_ids"]],
                "should_resolve": spec["challenge_types"],
            },
            "challenge_types": spec["challenge_types"],
            "valid_constraint_cell_ids": [cell["cell_id"] for cell in spec["valid_constraint_cells"]],
            "expected_provenance": provenance,
        },
        "feasible_region": feasible,
        "certificate_targets": {
            "source_rule_ids": spec["source_rule_ids"],
            "provenance": provenance["source_documents"],
        },
        "diagnostic_candidate_rule_ids_reference_only": spec["source_rule_ids"],
        "rule_library_id_bindings": rule_library_id_bindings(rule_lookup, expected_ids),
        "reference_constraint_rule_bindings": reference_constraint_rule_bindings(feasible, rule_lookup),
    }


def make_public_scenario_model(spec: dict[str, Any]) -> dict[str, Any]:
    public_constraints = [
        constraint
        for constraint in spec["executable_constraints"]
        if constraint.get("source_type") == "task_or_scenario_model"
    ]
    return {
        "omega_id": spec["omega_id"],
        "model_id": f"{spec['omega_id']}_scenario_model",
        "title": spec["title"],
        "visibility": "public_algorithm_input",
        "model_scope": "task_physics_and_objective_closure_only",
        "leakage_policy": "No expected rule IDs, defeated/surviving labels, provenance answers, certificate targets, or rule-library bindings are included.",
        "constraints": public_constraints,
    }


def constraint_semantics_audit(evaluation_references: list[dict[str, Any]]) -> dict[str, Any]:
    unique_counts = {
        "rule_library_numeric_or_direct_executable_constraints": 0,
        "rule_library_semantic_indicator_encoding_constraints": 0,
        "public_task_or_scenario_model_constraints": 0,
        "other_executable_constraints": 0,
    }
    json_occurrence_counts = {
        "semantic_indicator_encoding_occurrences_including_valid_cell_mirrors": 0,
        "task_or_scenario_model_occurrences_including_valid_cell_mirrors": 0,
        "rule_library_executable_occurrences_including_valid_cell_mirrors": 0,
    }
    semantic_examples: list[dict[str, Any]] = []

    def visit(payload: Any) -> None:
        if isinstance(payload, dict):
            if payload.get("executable") is True and payload.get("source_type"):
                if payload.get("source_type") == "task_or_scenario_model":
                    json_occurrence_counts["task_or_scenario_model_occurrences_including_valid_cell_mirrors"] += 1
                if payload.get("source_type") == "rule_library":
                    json_occurrence_counts["rule_library_executable_occurrences_including_valid_cell_mirrors"] += 1
                if payload.get("metadata", {}).get("semantic_indicator_encoding"):
                    json_occurrence_counts["semantic_indicator_encoding_occurrences_including_valid_cell_mirrors"] += 1
            for value in payload.values():
                visit(value)
        elif isinstance(payload, list):
            for value in payload:
                visit(value)

    for reference in evaluation_references:
        feasible = reference.get("feasible_region", {})
        for item in feasible.get("executable_constraints", []):
            source_type = item.get("source_type")
            if source_type == "task_or_scenario_model":
                unique_counts["public_task_or_scenario_model_constraints"] += 1
            elif source_type == "rule_library" and item.get("metadata", {}).get("semantic_indicator_encoding"):
                unique_counts["rule_library_semantic_indicator_encoding_constraints"] += 1
                if len(semantic_examples) < 12:
                    semantic_examples.append(
                        {
                            "omega_id": reference["omega_id"],
                            "constraint_id": item.get("constraint_id"),
                            "expression": item.get("expression"),
                            "source_rule_id": item.get("source_id"),
                        }
                    )
            elif source_type == "rule_library":
                unique_counts["rule_library_numeric_or_direct_executable_constraints"] += 1
            else:
                unique_counts["other_executable_constraints"] += 1
        visit(feasible)

    return {
        "version": "architecture_fullkg_clean_constraint_semantics_audit_v1",
        "counting_policy": {
            "unique_constraint_counts": "Counts feasible_region.executable_constraints once per task.",
            "json_occurrence_counts": "Counts mirrored copies inside valid_constraint_cells as separate JSON occurrences.",
            "semantic_indicator_encoding": "A semantic/categorical full-KG rule encoded as an executable indicator/count predicate; report separately from direct numeric constraints.",
        },
        "unique_constraint_counts": unique_counts,
        "json_occurrence_counts": json_occurrence_counts,
        "semantic_indicator_examples": semantic_examples,
    }


def cthr_structure_challenge_audit(evaluation_references: list[dict[str, Any]]) -> dict[str, Any]:
    single_rule_tasks = 0
    tasks_with_defeated_rules = 0
    tasks_with_multiple_valid_cells = 0
    tasks_with_structure_only_constraints = 0
    referenced_rule_distribution: dict[int, int] = {}
    defeated_distribution: dict[int, int] = {}
    valid_cell_distribution: dict[int, int] = {}

    for reference in evaluation_references:
        rule_structure = reference.get("rule_structure", {})
        feasible = reference.get("feasible_region", {})
        referenced_count = len(set(rule_structure.get("expected_source_rule_ids", [])))
        defeated_count = len(rule_structure.get("expected_defeated_rule_ids", []))
        valid_cell_count = len(feasible.get("valid_constraint_cells", []))
        structure_only_count = len(feasible.get("structure_only_constraints", []))

        referenced_rule_distribution[referenced_count] = referenced_rule_distribution.get(referenced_count, 0) + 1
        defeated_distribution[defeated_count] = defeated_distribution.get(defeated_count, 0) + 1
        valid_cell_distribution[valid_cell_count] = valid_cell_distribution.get(valid_cell_count, 0) + 1
        single_rule_tasks += int(referenced_count == 1)
        tasks_with_defeated_rules += int(defeated_count > 0)
        tasks_with_multiple_valid_cells += int(valid_cell_count > 1)
        tasks_with_structure_only_constraints += int(structure_only_count > 0)

    return {
        "version": "architecture_fullkg_clean_cthr_structure_challenge_audit_v1",
        "scope_note": "This main split is a full-Qwen architecture KG optimization benchmark. It is not, by itself, a strong CTHR defeasible-chain stress split.",
        "counts": {
            "tasks": len(evaluation_references),
            "single_rule_tasks": single_rule_tasks,
            "tasks_with_defeated_rules": tasks_with_defeated_rules,
            "tasks_with_multiple_valid_cells": tasks_with_multiple_valid_cells,
            "tasks_with_structure_only_constraints": tasks_with_structure_only_constraints,
        },
        "distributions": {
            "referenced_rule_count_per_task": {str(key): value for key, value in sorted(referenced_rule_distribution.items())},
            "defeated_rule_count_per_task": {str(key): value for key, value in sorted(defeated_distribution.items())},
            "valid_cell_count_per_task": {str(key): value for key, value in sorted(valid_cell_distribution.items())},
        },
        "recommendation": "Use this split for full-KG grounded optimization and add a separate architecture_cthr_defeasible split for alternative compliance chains, exception override, stricter-rule survival, cross-source precedence, and multi-cell valid-chain selection.",
    }


def rule_relation_coverage_audit(
    evaluation_references: list[dict[str, Any]],
    rule_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    challenge_type_task_counts: dict[str, int] = {}
    referenced_rule_ids: set[str] = set()
    relation_type_occurrence_counts: dict[str, int] = {}
    relation_type_task_counts: dict[str, int] = {}
    rule_type_unique_counts: dict[str, int] = {}
    precedence_task_ids: list[str] = []

    for reference in evaluation_references:
        omega_id = str(reference.get("omega_id"))
        rule_structure = reference.get("rule_structure", {})
        challenge_types = list(rule_structure.get("challenge_types", []))
        source_rule_ids = list(rule_structure.get("expected_source_rule_ids", []))
        task_relation_types: set[str] = set()
        task_has_precedence = "precedence" in challenge_types

        for challenge_type in challenge_types:
            challenge_type_task_counts[challenge_type] = challenge_type_task_counts.get(challenge_type, 0) + 1

        for rule_id in source_rule_ids:
            referenced_rule_ids.add(rule_id)
            rule = rule_lookup.get(rule_id, {})
            if rule.get("rule_type") == "precedence":
                task_has_precedence = True
            for relation in rule.get("relations", []) if isinstance(rule.get("relations", []), list) else []:
                if not isinstance(relation, dict):
                    continue
                relation_type = str(relation.get("type", "unknown"))
                relation_type_occurrence_counts[relation_type] = relation_type_occurrence_counts.get(relation_type, 0) + 1
                task_relation_types.add(relation_type)
                if relation_type in {"precedes", "overrides"}:
                    task_has_precedence = True

        for relation_type in task_relation_types:
            relation_type_task_counts[relation_type] = relation_type_task_counts.get(relation_type, 0) + 1
        if task_has_precedence:
            precedence_task_ids.append(omega_id)

    for rule_id in sorted(referenced_rule_ids):
        rule_type = str(rule_lookup.get(rule_id, {}).get("rule_type", "missing"))
        rule_type_unique_counts[rule_type] = rule_type_unique_counts.get(rule_type, 0) + 1

    return {
        "version": "architecture_fullkg_clean_rule_relation_coverage_audit_v1",
        "counting_policy": {
            "challenge_type_task_counts": "A task contributes once to each declared challenge_type.",
            "referenced_rule_type_unique_counts": "Counts unique source rule IDs referenced by the 30 evaluation references.",
            "referenced_relation_type_occurrence_counts": "Counts relation objects attached to referenced full-KG rules.",
            "referenced_relation_type_task_counts": "Counts tasks with at least one referenced source rule carrying that relation type.",
            "precedence_coverage": "A task is counted when it declares precedence, references a rule_type=precedence rule, or references a rule relation of type precedes/overrides.",
        },
        "challenge_type_task_counts": {key: challenge_type_task_counts[key] for key in sorted(challenge_type_task_counts)},
        "referenced_rule_type_unique_counts": {key: rule_type_unique_counts[key] for key in sorted(rule_type_unique_counts)},
        "referenced_relation_type_occurrence_counts": {key: relation_type_occurrence_counts[key] for key in sorted(relation_type_occurrence_counts)},
        "referenced_relation_type_task_counts": {key: relation_type_task_counts[key] for key in sorted(relation_type_task_counts)},
        "precedence_coverage": {
            "task_count": len(precedence_task_ids),
            "task_ids": sorted(precedence_task_ids),
        },
    }


def main() -> None:
    source_rule_library = read_json(FULL_QWEN_RULE_LIBRARY)
    rule_library, duplicate_records = canonicalize_rule_library(source_rule_library)
    rule_lookup = rule_lookup_by_id(rule_library)
    unique_rule_ids = set(rule_lookup)
    source_duplicate_rule_count = len(source_rule_library.get("rules", [])) - len(
        {str(rule["rule_id"]) for rule in source_rule_library.get("rules", []) if isinstance(rule, dict) and rule.get("rule_id")}
    )
    output_duplicate_rule_count = len(rule_library.get("rules", [])) - len(unique_rule_ids)

    missing_by_task: dict[str, list[str]] = {}
    leakage_hits: dict[str, list[str]] = {}
    referenced_rule_ids: set[str] = set()

    for directory in (INPUT_DIR, SCENARIO_MODEL_DIR, REFERENCE_DIR, TASK_DIR, RULE_LIBRARY_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    for stale_task in TASK_DIR.glob("*.json"):
        stale_task.unlink()

    algorithm_inputs: list[dict[str, Any]] = []
    public_scenario_models: list[dict[str, Any]] = []
    evaluation_references: list[dict[str, Any]] = []

    seen_task_ids: set[str] = set()
    for spec in TASK_SPECS:
        omega_id = spec["omega_id"]
        if omega_id in seen_task_ids:
            raise ValueError(f"Duplicate task id: {omega_id}")
        seen_task_ids.add(omega_id)
        if len(spec["objectives"]) < 2:
            raise ValueError(f"{omega_id} has fewer than two objectives")

        prepared_constraints = annotate_semantic_indicator_constraints(
            spec["executable_constraints"] + OBJECTIVE_CLOSURE_CONSTRAINTS.get(omega_id, []),
            rule_lookup,
        )
        prepared_spec = dict(spec)
        prepared_spec["scenario_facts"] = guard_aligned_scenario_facts(
            spec["scenario_facts"],
            spec["source_rule_ids"],
            rule_lookup,
        )
        prepared_spec["executable_constraints"] = prepared_constraints
        prepared_spec["valid_constraint_cells"] = [
            {**cell, "constraints": prepared_constraints}
            for cell in spec["valid_constraint_cells"]
        ]

        refs = set(prepared_spec["source_rule_ids"])
        refs.update(collect_rule_library_source_ids(prepared_spec["executable_constraints"]))
        refs.update(collect_rule_library_source_ids(prepared_spec["structure_only_constraints"]))
        missing = sorted(refs - unique_rule_ids)
        if missing:
            missing_by_task[omega_id] = missing
            continue
        referenced_rule_ids.update(refs)

        algorithm_input = make_algorithm_input(prepared_spec)
        hits = find_forbidden_keys(algorithm_input)
        if hits:
            leakage_hits[omega_id] = hits
        evaluation_reference = make_reference(prepared_spec, rule_lookup)
        public_scenario_model = make_public_scenario_model(prepared_spec)

        task_payload = {
            "version": "architecture_fullkg_clean_task_v1",
            "algorithm_input": algorithm_input,
            "evaluation_reference": evaluation_reference,
            "metadata": {
                "split": "architecture_fullkg_curated30",
                "rule_library": "rule_libraries/full_architecture_rule_library_qwen.json",
                "public_scenario_model": "scenario_models/architecture_public_scenario_models.json",
                "input_reference_policy": "algorithm_input plus public_scenario_models contain visible design-query and task-physics fields; evaluation_reference contains labels, rule-derived feasible-region answers, and full-KG rule-id bindings.",
            },
        }
        algorithm_inputs.append(algorithm_input)
        public_scenario_models.append(public_scenario_model)
        evaluation_references.append(evaluation_reference)
        write_json(TASK_DIR / f"{omega_id}.json", task_payload)

    write_json(RULE_LIBRARY_DIR / "full_architecture_rule_library_qwen.json", rule_library)
    write_json(
        RULE_LIBRARY_DIR / "DUPLICATE_RULE_ID_POLICY.json",
        {
            "version": "architecture_fullkg_clean_duplicate_rule_id_policy_v1",
            "source_rule_library": str(FULL_QWEN_RULE_LIBRARY),
            "output_rule_library": str(RULE_LIBRARY_DIR / "full_architecture_rule_library_qwen.json"),
            "policy": "first_rule_id_occurrence_wins",
            "reason": "The raw full-Qwen architecture KG extraction contains duplicate rule_id values. The benchmark output rule library is canonicalized to one rule per rule_id so rule-id keyed retrieval is deterministic.",
            "duplicate_rule_id_count_in_source": source_duplicate_rule_count,
            "duplicate_rule_id_count_in_output": output_duplicate_rule_count,
            "duplicate_records": duplicate_records,
        },
    )

    write_json(
        INPUT_DIR / "architecture_algorithm_inputs.json",
        {
            "version": "architecture_fullkg_clean_algorithm_inputs_v1",
            "items": algorithm_inputs,
        },
    )
    write_json(
        SCENARIO_MODEL_DIR / "architecture_public_scenario_models.json",
        {
            "version": "architecture_fullkg_clean_public_scenario_models_v1",
            "visibility": "public_algorithm_input",
            "purpose": "Task physics and objective-closure constraints needed for fair optimization. These models do not contain expected rule IDs, defeated/surviving labels, provenance answers, certificate targets, or rule-library bindings.",
            "items": public_scenario_models,
        },
    )
    write_json(
        REFERENCE_DIR / "architecture_evaluation_references.json",
        {
            "version": "architecture_fullkg_clean_evaluation_references_v1",
            "items": evaluation_references,
        },
    )

    semantics_audit = constraint_semantics_audit(evaluation_references)
    structure_audit = cthr_structure_challenge_audit(evaluation_references)
    relation_coverage_audit = rule_relation_coverage_audit(evaluation_references, rule_lookup)
    write_json(OUT_DIR / "CONSTRAINT_SEMANTICS_AUDIT.json", semantics_audit)
    write_json(OUT_DIR / "CTHR_STRUCTURE_CHALLENGE_AUDIT.json", structure_audit)
    write_json(OUT_DIR / "RULE_RELATION_COVERAGE_AUDIT.json", relation_coverage_audit)

    audit = {
        "version": "architecture_fullkg_clean_leakage_audit_v1",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "task_count": len(algorithm_inputs),
        "target_task_count": len(TASK_SPECS),
        "source_rule_library_rules": len(source_rule_library.get("rules", [])),
        "source_rule_library_unique_rule_ids": len(unique_rule_ids),
        "source_duplicate_rule_id_count": source_duplicate_rule_count,
        "rule_library_rules": len(rule_library.get("rules", [])),
        "rule_library_unique_rule_ids": len(unique_rule_ids),
        "duplicate_rule_id_count_in_source_library": source_duplicate_rule_count,
        "duplicate_rule_id_count_in_output_rule_library": output_duplicate_rule_count,
        "referenced_rule_ids": len(referenced_rule_ids),
        "missing_reference_rule_ids_by_task": missing_by_task,
        "missing_reference_rule_id_count": sum(len(value) for value in missing_by_task.values()),
        "forbidden_input_keys": sorted(FORBIDDEN_INPUT_KEYS),
        "input_forbidden_key_hits": leakage_hits,
        "input_forbidden_key_hit_count": sum(len(value) for value in leakage_hits.values()),
        "status": "pass" if not leakage_hits and not missing_by_task and len(algorithm_inputs) == len(TASK_SPECS) else "fail",
    }
    write_json(OUT_DIR / "LEAKAGE_AUDIT.json", audit)

    manifest = {
        "version": "architecture_fullkg_clean_manifest_v1",
        "generated_at": audit["generated_at"],
        "purpose": "Clean architecture benchmark using the Qwen full architecture KG rule library as the only source rule library and strict algorithm-input/reference separation.",
        "source_files": {
            "full_qwen_rule_library": str(FULL_QWEN_RULE_LIBRARY),
        },
        "outputs": {
            "algorithm_inputs": str(INPUT_DIR / "architecture_algorithm_inputs.json"),
            "public_scenario_models": str(SCENARIO_MODEL_DIR / "architecture_public_scenario_models.json"),
            "evaluation_references": str(REFERENCE_DIR / "architecture_evaluation_references.json"),
            "task_files": str(TASK_DIR),
            "rule_library": str(RULE_LIBRARY_DIR / "full_architecture_rule_library_qwen.json"),
            "duplicate_rule_id_policy": str(RULE_LIBRARY_DIR / "DUPLICATE_RULE_ID_POLICY.json"),
            "leakage_audit": str(OUT_DIR / "LEAKAGE_AUDIT.json"),
            "constraint_semantics_audit": str(OUT_DIR / "CONSTRAINT_SEMANTICS_AUDIT.json"),
            "cthr_structure_challenge_audit": str(OUT_DIR / "CTHR_STRUCTURE_CHALLENGE_AUDIT.json"),
            "rule_relation_coverage_audit": str(OUT_DIR / "RULE_RELATION_COVERAGE_AUDIT.json"),
        },
        "counts": {
            "tasks": len(algorithm_inputs),
            "source_rule_library_rules": len(source_rule_library.get("rules", [])),
            "source_duplicate_rule_id_count": source_duplicate_rule_count,
            "rule_library_rules": audit["rule_library_rules"],
            "rule_library_unique_rule_ids": audit["rule_library_unique_rule_ids"],
            "duplicate_rule_id_count_in_source_library": source_duplicate_rule_count,
            "duplicate_rule_id_count_in_output_rule_library": output_duplicate_rule_count,
            "referenced_rule_ids": audit["referenced_rule_ids"],
            "missing_reference_rule_ids": audit["missing_reference_rule_id_count"],
            "forbidden_input_key_hits": audit["input_forbidden_key_hit_count"],
            "rule_library_numeric_or_direct_executable_constraints": semantics_audit["unique_constraint_counts"]["rule_library_numeric_or_direct_executable_constraints"],
            "rule_library_semantic_indicator_encoding_constraints": semantics_audit["unique_constraint_counts"]["rule_library_semantic_indicator_encoding_constraints"],
            "public_task_or_scenario_model_constraints": semantics_audit["unique_constraint_counts"]["public_task_or_scenario_model_constraints"],
            "precedence_coverage_tasks": relation_coverage_audit["precedence_coverage"]["task_count"],
        },
        "cthr_structure_scope_note": structure_audit["scope_note"],
        "excluded_from_main_dataset": {
            "legacy_architecture_dataset": "datasets/architecture",
            "reason": "The legacy benchmark references a 61-rule benchmark/stress library whose rule IDs do not overlap the Qwen full-KG rule IDs; it should be treated as diagnostic unless separately mapped.",
        },
    }
    write_json(OUT_DIR / "MANIFEST.json", manifest)

    readme = [
        "# Architecture Full-KG Clean Dataset",
        "",
        "This dataset treats `full_architecture_rule_library_qwen.json` as the only source rule library.",
        "It contains 30 curated architecture/code-compliance optimization tasks grounded directly to full-KG rule IDs.",
        "",
        "## Split Policy",
        "",
        "- `algorithm_inputs/architecture_algorithm_inputs.json`: fields visible to algorithms.",
        "- `scenario_models/architecture_public_scenario_models.json`: public task-physics and objective-closure constraints visible to algorithms.",
        "- `evaluation_references/architecture_evaluation_references.json`: reference answers used only by evaluators.",
        "- `tasks/*.json`: paired files with explicit `algorithm_input` and `evaluation_reference` sections.",
        "- `rule_libraries/full_architecture_rule_library_qwen.json`: canonical first-wins de-duplicated Qwen full-KG rule library for this dataset.",
        "- `rule_libraries/DUPLICATE_RULE_ID_POLICY.json`: duplicate `rule_id` resolution policy for the raw full-Qwen extraction.",
        "- `RULE_RELATION_COVERAGE_AUDIT.json`: coverage counts for task challenge types, referenced rule types, and referenced full-KG relation types.",
        "- `evaluation_reference.rule_library_id_bindings`: explicit mapping from each correct/reference rule ID to the same `rule_id` in the full-KG rule library.",
        "- `evaluation_reference.reference_constraint_rule_bindings`: explicit mapping from each reference constraint to its source full-KG rule ID.",
        "- Objective-score variables are closed by public `task_or_scenario_model` constraints, so objectives are tied to visible design variables instead of free upper/lower bounds.",
        "- Semantic indicator/count predicates include `semantic_indicator_encoding` metadata when they encode textual or categorical full-KG rules.",
        "",
        "Algorithm code should read only `algorithm_inputs/architecture_algorithm_inputs.json`, `scenario_models/architecture_public_scenario_models.json`, and the rule library.",
        "The paired `tasks/*.json` files are for auditing and evaluator-side debugging; do not pass full task files to algorithm code.",
        "",
        "## Counts",
        "",
        f"- Tasks: {len(algorithm_inputs)}",
        f"- Source rule-library rules before de-duplication: {audit['source_rule_library_rules']}",
        f"- Canonical output rule-library rules: {audit['rule_library_rules']}",
        f"- Unique rule IDs: {audit['rule_library_unique_rule_ids']}",
        f"- Duplicate rule IDs in canonical output rule library: {audit['duplicate_rule_id_count_in_output_rule_library']}",
        f"- Referenced full-KG rule IDs covered by the rule library: {audit['referenced_rule_ids'] - audit['missing_reference_rule_id_count']} / {audit['referenced_rule_ids']}",
        f"- Forbidden input-key hits: {audit['input_forbidden_key_hit_count']}",
        f"- Public task/scenario-model constraints: {semantics_audit['unique_constraint_counts']['public_task_or_scenario_model_constraints']}",
        f"- Direct numeric rule-library executable constraints: {semantics_audit['unique_constraint_counts']['rule_library_numeric_or_direct_executable_constraints']}",
        f"- Semantic/categorical rule-library executable encodings: {semantics_audit['unique_constraint_counts']['rule_library_semantic_indicator_encoding_constraints']}",
        f"- Tasks covering precedence/priority resolution: {relation_coverage_audit['precedence_coverage']['task_count']}",
        "",
        "## Constraint Semantics",
        "",
        "Numeric rule-library constraints and semantic/categorical executable encodings should be reported separately in experiments.",
        "The semantic audit counts unique top-level constraints separately from mirrored JSON occurrences inside valid constraint cells.",
        "",
        "## CTHR Challenge Scope",
        "",
        "This split is a full-KG grounded optimization benchmark, not a strong standalone defeasible-chain stress split.",
        "Use `CTHR_STRUCTURE_CHALLENGE_AUDIT.json` and `RULE_RELATION_COVERAGE_AUDIT.json` to report this limitation and add a separate CTHR-focused split for alternative compliance chains, exception override, stricter-rule survival, cross-source precedence, and multi-cell valid-chain selection.",
        "",
        "## Legacy Diagnostic Data",
        "",
        "The previous `datasets/architecture` benchmark remains useful as a diagnostic/stress set, but it is not a full-KG benchmark because its 61-rule library uses a different rule-ID namespace.",
        "",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(readme), encoding="utf-8")

    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
