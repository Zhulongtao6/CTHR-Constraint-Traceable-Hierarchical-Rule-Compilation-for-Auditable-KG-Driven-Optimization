from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "architecture_fullkg_clean"
TASK_IDS = [f"ARCH_FKG_{number:02d}" for number in range(86, 92)]
MATH_SYMBOLS = {"abs", "min", "max", "sqrt", "tan", "sin", "cos", "pi"}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def symbols(expression: str) -> list[str]:
    return sorted(set(re.findall(r"\b[A-Za-z_]\w*\b", expression)) - MATH_SYMBOLS)


def dv(unit: str, lower: float, upper: float, kind: str = "continuous") -> dict:
    return {"type": kind, "unit": unit, "lower": lower, "upper": upper}


def metadata_public(note: str) -> dict:
    return {
        "objective_closure": True,
        "closure_source": "task_or_scenario_model",
        "closure_visibility": "public_algorithm_input",
        "closure_note": note,
    }


def metadata_semantic(note: str) -> dict:
    return {
        "semantic_indicator_encoding": True,
        "derived_from_text_rule": True,
        "encoding_note": note,
    }


def constraint(
    constraint_id: str,
    expression: str,
    role: str,
    source_type: str,
    source_id: str,
    decision_variables: list[str],
    scenario_fields: list[str],
    metadata: dict | None = None,
) -> dict:
    expr_symbols = set(symbols(expression))
    dv_set = set(decision_variables)
    scenario_set = set(scenario_fields)
    out = {
        "constraint_id": constraint_id,
        "expression": expression,
        "role": role,
        "source_type": source_type,
        "source_id": source_id,
        "executable": True,
        "checker_expression": expression,
        "expression_language": "python_safe_arithmetic_predicate",
        "symbols": {
            "decision_variables": sorted(expr_symbols & dv_set),
            "scenario_fields": sorted(expr_symbols & scenario_set),
            "unresolved_symbols": sorted(expr_symbols - dv_set - scenario_set),
        },
    }
    if metadata:
        out["metadata"] = metadata
    return out


def rule_document(rule_id: str) -> str:
    lowered = rule_id.lower()
    if lowered.startswith("ada"):
        return "ADA 2010"
    if lowered.startswith("ibc"):
        return "IBC 2021"
    return "IFC 2021"


def provenance(rule_ids: list[str], rules_by_id: dict[str, dict], origin: dict) -> dict:
    chunks: list[str] = []
    nodes: list[str] = []
    edges: list[str] = []
    source_documents: list[dict] = []
    for rule_id in rule_ids:
        rule = rules_by_id[rule_id]
        for chunk_id in rule.get("source_chunk_ids", []) or []:
            if chunk_id not in chunks:
                chunks.append(chunk_id)
        for node_id in rule.get("source_node_ids", []) or []:
            if node_id not in nodes:
                nodes.append(node_id)
        for item in list(rule.get("constraints", []) or []) + list(rule.get("relations", []) or []):
            evidence = item.get("evidence", {}) if isinstance(item, dict) else {}
            for edge_id in evidence.get("kg_edge_ids", []) or []:
                if edge_id not in edges:
                    edges.append(edge_id)
        source_documents.append(
            {
                "document": rule_document(rule_id),
                "section": "source_chunk",
                "page": "unknown",
                "chunk_id": (rule.get("source_chunk_ids") or ["unknown"])[0],
                "rule_id": rule_id,
            }
        )
    source_documents.append(
        {
            "document": origin["document"],
            "section": origin["division"],
            "page": "practice-exam-question",
            "question_number": origin["question_number"],
            "derivation_policy": (
                "Paraphrased and parameterized from an NCARB practice-exam scenario; "
                "original question text is not copied."
            ),
        }
    )
    return {
        "kg_chunk_ids": chunks,
        "kg_node_ids": nodes,
        "kg_edge_ids": edges,
        "source_documents": source_documents,
    }


def make_origin(document: str, division: str, question: str, theme: str) -> dict:
    return {
        "source_family": "NCARB_ARE_practice_exam",
        "document": document,
        "division": division,
        "question_number": question,
        "theme": theme,
        "adaptation_policy": "paraphrased_parameterized_not_verbatim",
    }


def build_task(spec: dict, rules_by_id: dict[str, dict]) -> tuple[dict, dict, dict]:
    task_id = spec["omega_id"]
    decision_variables = spec["decision_variables"]
    scenario_facts = spec["scenario_facts"]
    dv_names = list(decision_variables)
    scenario_names = list(scenario_facts)
    rule_constraints = [
        constraint(
            f"C{index}",
            item["expression"],
            item["role"],
            "rule_library",
            item["source_id"],
            dv_names,
            scenario_names,
            item.get("metadata"),
        )
        for index, item in enumerate(spec["rule_constraints"], 1)
    ]
    public_constraints = [
        constraint(
            f"C{index}",
            item["expression"],
            item["role"],
            "task_or_scenario_model",
            item.get("source_id", "scenario_objective_model"),
            dv_names,
            scenario_names,
            item.get("metadata"),
        )
        for index, item in enumerate(spec["public_constraints"], 90)
    ]
    all_constraints = rule_constraints + public_constraints
    rule_ids = spec["rule_ids"]
    origin = spec["source_task_origin"]
    algorithm_input = {
        "omega_id": task_id,
        "title": spec["title"],
        "domain": "architecture_code_compliance",
        "source_domain": spec["source_domain"],
        "task_type": spec["task_type"],
        "design_intent": spec["design_intent"],
        "scenario_facts": scenario_facts,
        "source_task_origin": origin,
        "decision_variables": decision_variables,
        "objectives": spec["objectives"],
        "query_preferences": spec["query_preferences"],
        "public_scenario_model": {
            "model_id": f"{task_id}_scenario_model",
            "path": "scenario_models/architecture_public_scenario_models.json",
            "visibility": "public_algorithm_input",
            "purpose": (
                "Non-normative task physics/objective-closure constraints visible to optimizers; "
                "contains no expected rule IDs or labels."
            ),
        },
        "visible_input_note": (
            "Visible task input only. Algorithms may also read the public scenario model and rule library. "
            "Rule labels, rule-derived feasible-region answers, and rule-id bindings remain hidden evaluation references."
        ),
    }
    evaluation_reference = {
        "omega_id": task_id,
        "title": spec["title"],
        "rule_structure": {
            "expected_source_rule_ids": rule_ids,
            "expected_defeated_rule_ids": spec.get("defeated_rule_ids", []),
            "expected_surviving_rule_ids": rule_ids,
            "expected_valid_rule_structures": [rule_ids],
            "expected_rule_behavior": {
                "should_activate": [rules_by_id[rule_id].get("name", rule_id) for rule_id in rule_ids],
                "should_exclude": spec.get("should_exclude", []),
                "should_resolve": spec["challenge_types"],
            },
            "challenge_types": spec["challenge_types"],
            "valid_constraint_cell_ids": [f"{task_id}_cell_1"],
            "expected_provenance": provenance(rule_ids, rules_by_id, origin),
        },
        "task_derivation_source": origin,
        "benchmark_extension_metadata": {
            "extension_round": "architecture_ncarb_50_extension",
            "source_base_task_id": spec.get("source_base_task_id"),
            "ncarb_category": "site_and_zoning_like",
            "source_policy": "paraphrased_parameterized_not_verbatim",
        },
        "feasible_region": {
            "executable_constraints": all_constraints,
            "structure_only_constraints": [],
            "valid_constraint_cells": [
                {
                    "cell_id": f"{task_id}_cell_1",
                    "description": "Single source-grounded feasible cell after applying applicable rules and public scenario model constraints.",
                    "constraint_ids": [item["constraint_id"] for item in all_constraints],
                    "source_rule_ids": rule_ids,
                }
            ],
        },
        "evaluation_usage_note": (
            "Hidden reference only. Algorithms must not receive rule IDs, feasible-region answers, "
            "provenance targets, defeated/surviving labels, or valid-cell labels during prediction."
        ),
    }
    scenario_model = {
        "omega_id": task_id,
        "model_id": f"{task_id}_scenario_model",
        "title": spec["title"],
        "visibility": "public_algorithm_input",
        "model_scope": "task_physics_and_objective_closure_only",
        "leakage_policy": (
            "No expected rule IDs, defeated/surviving labels, provenance answers, certificate targets, "
            "or rule-library bindings are included."
        ),
        "constraints": deepcopy(public_constraints),
    }
    return algorithm_input, evaluation_reference, scenario_model


def main() -> None:
    algorithm_path = DATASET / "algorithm_inputs" / "architecture_algorithm_inputs.json"
    reference_path = DATASET / "evaluation_references" / "architecture_evaluation_references.json"
    scenario_path = DATASET / "scenario_models" / "architecture_public_scenario_models.json"
    rules = read_json(DATASET / "rule_libraries" / "full_architecture_rule_library_qwen.json")["rules"]
    rules_by_id = {rule["rule_id"]: rule for rule in rules if isinstance(rule, dict) and rule.get("rule_id")}

    programming = "ARE-Practice-Exam-Programming-and-Analysis.pdf"
    planning = "ARE-Practice-Exam-Project-Planning-and-Design.pdf"
    specs = [
        {
            "omega_id": "ARCH_FKG_86",
            "title": "NCARB-derived outdoor control area setback from buildable lot line",
            "source_domain": "IFC",
            "task_type": "site_hazard_control_area_layout",
            "design_intent": "Place an outdoor control area while balancing buildable site efficiency against required separation from public ways and buildable lot lines.",
            "scenario_facts": {
                "entity_type": "outdoor control area",
                "public_way_or_buildable_lot_line": True,
                "ncarb_scenario_theme": "utility easement and side-yard siting",
                "ncarb_practice_exam_division": "Programming and Analysis",
                "ncarb_practice_question_number": "Question 21",
                "ncarb_adaptation_policy": "paraphrased_parameterized_not_verbatim",
                "expansion_case_family": "site_and_zoning_like",
            },
            "source_task_origin": make_origin(programming, "Programming and Analysis", "Question 21", "utility easement and side-yard siting"),
            "decision_variables": {
                "minimum_setback_distance_ft": dv("ft", 0, 40),
                "site_efficiency_score": dv("score", 0, 15),
                "separation_safety_margin_score": dv("score", 0, 15),
            },
            "objectives": [
                {"name": "maximize_site_efficiency", "expression": "site_efficiency_score"},
                {"name": "maximize_separation_safety_margin", "expression": "separation_safety_margin_score"},
            ],
            "query_preferences": {"lambda": [0.45, 0.55], "meaning": "balance usable site efficiency and separation margin"},
            "rule_ids": ["ifc-outdoor-control-area-setback-20ft"],
            "rule_constraints": [
                {"expression": "minimum_setback_distance_ft >= 20", "role": "outdoor_control_area_minimum_setback", "source_id": "ifc-outdoor-control-area-setback-20ft"}
            ],
            "public_constraints": [
                {"expression": "site_efficiency_score <= 12 - 0.2 * minimum_setback_distance_ft", "role": "site_efficiency_from_setback", "metadata": metadata_public("Larger required setback reduces buildable site efficiency.")},
                {"expression": "separation_safety_margin_score <= minimum_setback_distance_ft - 20", "role": "separation_margin_from_setback", "metadata": metadata_public("Separation margin is computed above the 20 ft setback minimum.")},
            ],
            "challenge_types": ["scenario_conditioned_applicability", "parameter_limit"],
        },
        {
            "omega_id": "ARCH_FKG_87",
            "title": "NCARB-derived outdoor control area combustible clearance",
            "source_domain": "IFC",
            "task_type": "site_hazard_control_area_layout",
            "design_intent": "Set combustible-material clearance around an outdoor control area while balancing site compactness and hazard separation.",
            "scenario_facts": {
                "entity_type": "outdoor control area",
                "adjacent_combustible_materials": True,
                "ncarb_scenario_theme": "public park material selection and site infiltration",
                "ncarb_practice_exam_division": "Project Planning and Design",
                "ncarb_practice_question_number": "Question 31",
                "ncarb_adaptation_policy": "paraphrased_parameterized_not_verbatim",
                "expansion_case_family": "site_and_zoning_like",
            },
            "source_task_origin": make_origin(planning, "Project Planning and Design", "Question 31", "public park material selection and site infiltration"),
            "decision_variables": {
                "combustible_clearance_ft": dv("ft", 0, 30),
                "site_compactness_score": dv("score", 0, 15),
                "hazard_separation_margin_score": dv("score", 0, 15),
            },
            "objectives": [
                {"name": "maximize_site_compactness", "expression": "site_compactness_score"},
                {"name": "maximize_hazard_separation_margin", "expression": "hazard_separation_margin_score"},
            ],
            "query_preferences": {"lambda": [0.45, 0.55], "meaning": "balance compact site use and combustible clearance"},
            "rule_ids": ["ifc-outdoor-control-area-clearance-15ft"],
            "rule_constraints": [
                {"expression": "combustible_clearance_ft >= 15", "role": "outdoor_control_area_combustible_clearance", "source_id": "ifc-outdoor-control-area-clearance-15ft"}
            ],
            "public_constraints": [
                {"expression": "site_compactness_score <= 12 - 0.3 * combustible_clearance_ft", "role": "site_compactness_from_clearance", "metadata": metadata_public("Larger combustible clearance reduces compact site use.")},
                {"expression": "hazard_separation_margin_score <= combustible_clearance_ft - 15", "role": "hazard_margin_from_clearance", "metadata": metadata_public("Hazard separation margin is computed above the 15 ft minimum.")},
            ],
            "challenge_types": ["scenario_conditioned_applicability", "parameter_limit"],
        },
        {
            "omega_id": "ARCH_FKG_88",
            "title": "NCARB-derived fire-wall alternative to outdoor control area setback",
            "source_domain": "IFC",
            "task_type": "site_hazard_control_area_layout",
            "design_intent": "Evaluate a fire-wall alternative for an outdoor control area where site constraints make full separation costly.",
            "scenario_facts": {
                "entity_type": "outdoor control area",
                "hazardous_material_state": "solid_or_liquid",
                "public_way_or_buildable_lot_line": True,
                "alternative_fire_wall_considered": True,
                "ncarb_scenario_theme": "brownfield retail site planning",
                "ncarb_practice_exam_division": "Programming and Analysis",
                "ncarb_practice_question_number": "Question 20",
                "ncarb_adaptation_policy": "paraphrased_parameterized_not_verbatim",
                "expansion_case_family": "site_and_zoning_like",
            },
            "source_task_origin": make_origin(programming, "Programming and Analysis", "Question 20", "brownfield retail site planning"),
            "decision_variables": {
                "minimum_setback_distance_ft": dv("ft", 0, 40),
                "fire_wall_rating_hr": dv("hour", 0, 4),
                "site_efficiency_score": dv("score", 0, 15),
                "protection_redundancy_score": dv("score", 0, 15),
            },
            "objectives": [
                {"name": "maximize_site_efficiency", "expression": "site_efficiency_score"},
                {"name": "maximize_protection_redundancy", "expression": "protection_redundancy_score"},
            ],
            "query_preferences": {"lambda": [0.5, 0.5], "meaning": "balance site efficiency and fire-resistance protection"},
            "rule_ids": ["ifc-outdoor-control-area-setback-20ft", "ifc-outdoor-control-area-exception-fire-wall"],
            "rule_constraints": [
                {"expression": "minimum_setback_distance_ft >= 20", "role": "outdoor_control_area_base_setback", "source_id": "ifc-outdoor-control-area-setback-20ft"},
                {"expression": "fire_wall_rating_hr >= 2", "role": "two_hour_fire_wall_alternative", "source_id": "ifc-outdoor-control-area-exception-fire-wall"},
            ],
            "public_constraints": [
                {"expression": "site_efficiency_score <= 12 - 0.2 * minimum_setback_distance_ft - fire_wall_rating_hr", "role": "site_efficiency_from_setback_and_wall", "metadata": metadata_public("Setback and fire-wall construction both affect site efficiency.")},
                {"expression": "protection_redundancy_score <= (minimum_setback_distance_ft - 20) + 3 * fire_wall_rating_hr", "role": "protection_redundancy_from_setback_and_wall", "metadata": metadata_public("Protection redundancy combines setback margin and fire-wall rating.")},
            ],
            "challenge_types": ["exception_or_override", "multi_rule_conjunction"],
            "should_exclude": ["Do not treat site efficiency alone as satisfying the fire-wall alternative without applying the two-hour wall requirement."],
        },
        {
            "omega_id": "ARCH_FKG_89",
            "title": "NCARB-derived commercial parking garage sprinkler coverage",
            "source_domain": "IFC",
            "task_type": "parking_garage_fire_protection_design",
            "design_intent": "Select commercial parking garage sprinkler coverage while balancing system cost against fire-protection reliability.",
            "scenario_facts": {
                "occupancy_type": "commercial parking garage",
                "fire_area": 6200,
                "ncarb_scenario_theme": "parking capacity and building selection",
                "ncarb_practice_exam_division": "Programming and Analysis",
                "ncarb_practice_question_number": "Question 1",
                "ncarb_adaptation_policy": "paraphrased_parameterized_not_verbatim",
                "expansion_case_family": "site_and_zoning_like",
            },
            "source_task_origin": make_origin(programming, "Programming and Analysis", "Question 1", "parking capacity and building selection"),
            "decision_variables": {
                "sprinkler_system_indicator": dv("binary", 0, 1, "integer"),
                "sprinkler_zone_count": dv("count", 1, 8, "integer"),
                "fire_protection_cost_score": dv("score", 0, 20),
                "coverage_reliability_score": dv("score", 0, 20),
            },
            "objectives": [
                {"name": "minimize_fire_protection_cost", "expression": "fire_protection_cost_score"},
                {"name": "maximize_coverage_reliability", "expression": "coverage_reliability_score"},
            ],
            "query_preferences": {"lambda": [0.45, 0.55], "meaning": "balance sprinkler cost and fire-protection reliability"},
            "rule_ids": ["ifc_903_2_10_1_sprinklers_commercial_garage"],
            "rule_constraints": [
                {"expression": "sprinkler_system_indicator == 1", "role": "commercial_parking_garage_sprinkler_required", "source_id": "ifc_903_2_10_1_sprinklers_commercial_garage", "metadata": metadata_semantic("Binary variable encodes required automatic sprinkler provision for a commercial parking garage.")}
            ],
            "public_constraints": [
                {"expression": "fire_protection_cost_score >= 2 * sprinkler_zone_count + 4 * sprinkler_system_indicator", "role": "sprinkler_cost_from_zone_count", "metadata": metadata_public("Sprinkler cost increases with coverage zone count and required system provision.")},
                {"expression": "coverage_reliability_score <= 3 * sprinkler_zone_count + 5 * sprinkler_system_indicator", "role": "sprinkler_reliability_from_zone_count", "metadata": metadata_public("Coverage reliability grows with sprinkler zones and required system provision.")},
            ],
            "challenge_types": ["scenario_conditioned_applicability", "life_safety_system_design"],
        },
        {
            "omega_id": "ARCH_FKG_90",
            "title": "NCARB-derived mechanical-access parking sprinkler coverage",
            "source_domain": "IFC",
            "task_type": "parking_garage_fire_protection_design",
            "design_intent": "Plan sprinkler coverage for a mechanical-access enclosed parking garage while balancing cost and reliability.",
            "scenario_facts": {
                "occupancy_type": "mechanical-access enclosed parking garage",
                "fire_area": 5600,
                "ncarb_scenario_theme": "vehicle access and parking support planning",
                "ncarb_practice_exam_division": "Project Planning and Design",
                "ncarb_practice_question_number": "Question 26",
                "ncarb_adaptation_policy": "paraphrased_parameterized_not_verbatim",
                "expansion_case_family": "site_and_zoning_like",
            },
            "source_task_origin": make_origin(planning, "Project Planning and Design", "Question 26", "vehicle access and parking support planning"),
            "decision_variables": {
                "sprinkler_system_indicator": dv("binary", 0, 1, "integer"),
                "sprinkler_zone_count": dv("count", 1, 8, "integer"),
                "fire_protection_cost_score": dv("score", 0, 20),
                "coverage_reliability_score": dv("score", 0, 20),
            },
            "objectives": [
                {"name": "minimize_fire_protection_cost", "expression": "fire_protection_cost_score"},
                {"name": "maximize_coverage_reliability", "expression": "coverage_reliability_score"},
            ],
            "query_preferences": {"lambda": [0.45, 0.55], "meaning": "balance sprinkler cost and fire-protection reliability"},
            "rule_ids": ["ifc_903_2_10_2_sprinklers_mechanical_access_garage"],
            "rule_constraints": [
                {"expression": "sprinkler_system_indicator == 1", "role": "mechanical_access_parking_garage_sprinkler_required", "source_id": "ifc_903_2_10_2_sprinklers_mechanical_access_garage", "metadata": metadata_semantic("Binary variable encodes required automatic sprinkler provision for a mechanical-access enclosed parking garage.")}
            ],
            "public_constraints": [
                {"expression": "fire_protection_cost_score >= 2 * sprinkler_zone_count + 4 * sprinkler_system_indicator", "role": "sprinkler_cost_from_zone_count", "metadata": metadata_public("Sprinkler cost increases with coverage zone count and required system provision.")},
                {"expression": "coverage_reliability_score <= 3 * sprinkler_zone_count + 5 * sprinkler_system_indicator", "role": "sprinkler_reliability_from_zone_count", "metadata": metadata_public("Coverage reliability grows with sprinkler zones and required system provision.")},
            ],
            "challenge_types": ["scenario_conditioned_applicability", "life_safety_system_design"],
        },
        {
            "omega_id": "ARCH_FKG_91",
            "title": "NCARB-derived open parking garage sprinkler threshold",
            "source_domain": "IFC",
            "task_type": "parking_garage_fire_protection_design",
            "design_intent": "Evaluate sprinkler provision for an open parking garage exceeding the area threshold while balancing cost and reliability.",
            "scenario_facts": {
                "occupancy_type": "open parking garage",
                "fire_area": 52000,
                "ncarb_scenario_theme": "site selection with transit and parking",
                "ncarb_practice_exam_division": "Programming and Analysis",
                "ncarb_practice_question_number": "Question 23",
                "ncarb_adaptation_policy": "paraphrased_parameterized_not_verbatim",
                "expansion_case_family": "site_and_zoning_like",
            },
            "source_task_origin": make_origin(programming, "Programming and Analysis", "Question 23", "site selection with transit and parking"),
            "decision_variables": {
                "sprinkler_system_indicator": dv("binary", 0, 1, "integer"),
                "sprinkler_zone_count": dv("count", 1, 10, "integer"),
                "fire_protection_cost_score": dv("score", 0, 24),
                "coverage_reliability_score": dv("score", 0, 24),
            },
            "objectives": [
                {"name": "minimize_fire_protection_cost", "expression": "fire_protection_cost_score"},
                {"name": "maximize_coverage_reliability", "expression": "coverage_reliability_score"},
            ],
            "query_preferences": {"lambda": [0.45, 0.55], "meaning": "balance sprinkler cost and fire-protection reliability"},
            "rule_ids": ["ifc_open_parking_garage_sprinkler_threshold_ibc_406_5"],
            "rule_constraints": [
                {"expression": "sprinkler_system_indicator == 1", "role": "open_parking_garage_sprinkler_required", "source_id": "ifc_open_parking_garage_sprinkler_threshold_ibc_406_5", "metadata": metadata_semantic("Binary variable encodes required automatic sprinkler provision for an open parking garage above the threshold.")}
            ],
            "public_constraints": [
                {"expression": "fire_protection_cost_score >= 2 * sprinkler_zone_count + 4 * sprinkler_system_indicator", "role": "sprinkler_cost_from_zone_count", "metadata": metadata_public("Sprinkler cost increases with coverage zone count and required system provision.")},
                {"expression": "coverage_reliability_score <= 3 * sprinkler_zone_count + 5 * sprinkler_system_indicator", "role": "sprinkler_reliability_from_zone_count", "metadata": metadata_public("Coverage reliability grows with sprinkler zones and required system provision.")},
            ],
            "challenge_types": ["scenario_conditioned_applicability", "life_safety_system_design"],
        },
    ]

    algorithm_inputs = read_json(algorithm_path)
    evaluation_references = read_json(reference_path)
    scenario_models = read_json(scenario_path)
    replacements = {spec["omega_id"]: build_task(spec, rules_by_id) for spec in specs}

    def replace_items(items: list[dict], index: int) -> list[dict]:
        out = []
        for item in items:
            task_id = item.get("omega_id")
            out.append(replacements[task_id][index] if task_id in replacements else item)
        return sorted(out, key=lambda item: int(str(item["omega_id"]).rsplit("_", 1)[-1]))

    algorithm_inputs["items"] = replace_items(algorithm_inputs["items"], 0)
    evaluation_references["items"] = replace_items(evaluation_references["items"], 1)
    scenario_models["items"] = replace_items(scenario_models["items"], 2)
    write_json(algorithm_path, algorithm_inputs)
    write_json(reference_path, evaluation_references)
    write_json(scenario_path, scenario_models)
    write_json(DATASET / "core" / "algorithm_inputs" / "architecture_algorithm_inputs.json", algorithm_inputs)
    write_json(DATASET / "core" / "scenario_models" / "architecture_public_scenario_models.json", scenario_models)
    write_json(
        DATASET / "core" / "source_semantic_references" / "architecture_source_semantic_references.json",
        {
            "version": "architecture_fullkg_core_source_semantic_references_v1_ncarb_100_tasks",
            "canonical_rule_id_namespace": "qwen_canonical",
            "semantic_reference_policy": (
                "Fixed source-grounded task semantics, feasible regions, and provenance. This file is not "
                "model-specific and must not be used as a model-generated rule library."
            ),
            "items": evaluation_references["items"],
        },
    )
    references_by_id = {item["omega_id"]: item for item in evaluation_references["items"]}
    for algorithm_item in algorithm_inputs["items"]:
        task_id = algorithm_item["omega_id"]
        write_json(
            DATASET / "tasks" / f"{task_id}.json",
            {
                "version": "architecture_fullkg_clean_task_v1_ncarb_100_tasks",
                "algorithm_input": algorithm_item,
                "evaluation_reference": references_by_id[task_id],
            },
        )
    print(json.dumps({"replaced_site_parking_tasks": TASK_IDS}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
