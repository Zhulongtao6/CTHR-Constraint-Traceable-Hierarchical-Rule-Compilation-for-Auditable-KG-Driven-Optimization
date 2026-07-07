from __future__ import annotations

import csv
import json
import math
import re
import sys
import time
import argparse
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CTHR_ROOT = ROOT.parents[1]
RESULTS_DIR = ROOT / "results"
ARCHITECTURE_ROOT = ROOT / "datasets" / "architecture"
ARCHITECTURE_RULE_LIBRARY = ARCHITECTURE_ROOT / "architecture_stress_rule_library.combined.json"

SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import run_section_6_3_candidate_to_valid as ctv  # noqa: E402
import llm_grounding_reranker as llm_reranker  # noqa: E402


def default_architecture_rule_library(dataset_root: Path) -> Path:
    candidates = [
        dataset_root / "architecture_stress_rule_library.combined.json",
        dataset_root / "rule_libraries" / "full_architecture_rule_library_qwen.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


FORBIDDEN_INPUT_FIELDS = {
    "candidate_rule_ids_expected_for_diagnostics",
    "final_valid_rule_ids_expected_for_evaluation",
    "valid_rule_structures_expected",
    "solver_constraints",
    "solver_constraint_cells",
    "certificate_targets",
}

GENERIC_TOKENS = {
    "a",
    "ada",
    "accessibility",
    "accessible",
    "an",
    "and",
    "architecture",
    "are",
    "as",
    "at",
    "be",
    "building",
    "by",
    "choose",
    "clear",
    "code",
    "current",
    "design",
    "domain",
    "false",
    "for",
    "from",
    "general",
    "ibc",
    "ifc",
    "in",
    "into",
    "is",
    "max",
    "maximum",
    "min",
    "minimum",
    "new",
    "of",
    "on",
    "or",
    "out",
    "over",
    "plan",
    "required",
    "requirement",
    "requirements",
    "rule",
    "section",
    "select",
    "source",
    "standards",
    "standard",
    "template",
    "the",
    "to",
    "true",
    "type",
    "under",
    "unknown",
    "use",
    "user",
    "using",
    "while",
    "width",
    "length",
    "count",
    "ratio",
    "area",
    "amount",
    "number",
}

PARAMETER_VARIANT_TYPES = {
    "formula_variant_of",
    "parameter_variant_of",
    "piecewise_variant_of",
    "propagates_to",
}

DEPENDENCY_TYPES = {"depends_on", "requires", "uses_parameter"}
COMPETITION_TYPES = {"excludes", "mutually_exclusive", "conflicts_with", "conflict"}
OVERRIDE_TYPES = {"overrides", "can_override", "replaces", "defeats"}
PRECEDENCE_TYPES = {"precedes", "precedence", "higher_priority_than", "has_precedence_over"}
TYPED_SEED_MIN_SCORE = 4.0
TYPED_RELATION_MIN_SCORE = 2.5
TYPED_FAMILY_TOP_K = 4
GROUNDING_MODES = {"strict_profile", "relation_rich", "relation_stress"}
RELATION_RICH_SEED_MIN_SCORE = 2.2
RELATION_RICH_FAMILY_TOP_K = 10
RELATION_RICH_MAX_HOPS = 1
RELATION_RICH_RELATION_TYPES = (
    DEPENDENCY_TYPES | COMPETITION_TYPES | OVERRIDE_TYPES | PRECEDENCE_TYPES | PARAMETER_VARIANT_TYPES
)
RELATION_STRESS_SEED_MIN_SCORE = 2.0
RELATION_STRESS_FAMILY_TOP_K = 8
RELATION_STRESS_MAX_HOPS = 1
RELATION_STRESS_MIN_CANDIDATES = 8
RELATION_STRESS_MAX_CANDIDATES = 24


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def csv_cell(value: Any) -> str:
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(list(value) if isinstance(value, set) else value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({header: csv_cell(row.get(header)) for header in headers})


def markdown_table(rows: list[dict[str, Any]], headers: list[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(csv_cell(row.get(header)) for header in headers) + " |")
    return "\n".join(lines) + "\n"


def normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def token_set(value: Any) -> set[str]:
    if isinstance(value, dict):
        out: set[str] = set()
        for key, item in value.items():
            out |= token_set(key)
            out |= token_set(item)
        return out
    if isinstance(value, list):
        out: set[str] = set()
        for item in value:
            out |= token_set(item)
        return out
    return {
        token
        for token in normalize(value).split("_")
        if len(token) >= 2 and token not in GENERIC_TOKENS
    }


def architecture_scenario(task: dict[str, Any]) -> dict[str, Any]:
    scenario = dict(task.get("scenario_facts", {}))
    scenario["decision_variable_names"] = sorted(task.get("decision_variables", {}).keys())
    for key, value in list(scenario.items()):
        if isinstance(value, str):
            norm = normalize(value)
            if norm == "not_true":
                scenario[key] = False
            elif norm == "not_false":
                scenario[key] = True
    if normalize(scenario.get("locking_mechanism", "")).startswith("not_self_locking"):
        scenario["locking_mechanism"] = "standard latch mechanism"
    if normalize(scenario.get("door_type", "")).startswith("not_exterior_storm_or_screen"):
        scenario["door_type"] = "standard exterior egress door"
    approval = normalize(scenario.get("approval_status", ""))
    if approval.startswith("not_equivalent_communication_approved"):
        scenario["approval_status"] = "communication approval absent"
        scenario["scenario.approval_status"] = "communication approval absent"
    if "entity_type" in scenario:
        scenario.setdefault("scenario.entity_type", scenario["entity_type"])
    if "approval_status" in scenario:
        scenario.setdefault("scenario.approval_status", scenario["approval_status"])
    if "emergency_type" in scenario:
        scenario.setdefault("scenario.emergency_type", scenario["emergency_type"])
    if "location_type" in scenario:
        scenario.setdefault("scenario.location_type", scenario["location_type"])
    if "occupancy_group" in scenario:
        scenario.setdefault("scenario.occupancy_group", scenario["occupancy_group"])
    if "occupied_floor_stories_from_exit_discharge" in scenario:
        scenario.setdefault(
            "building.occupied_floor_stories_from_exit_discharge",
            scenario["occupied_floor_stories_from_exit_discharge"],
        )
    if "has_life_safety_risk" in scenario:
        scenario.setdefault("scenario.has_life_safety_risk", scenario["has_life_safety_risk"])
    return scenario


def architecture_rule_groups(rule: dict[str, Any]) -> set[str]:
    rid = normalize(rule.get("rule_id", ""))
    name = normalize(rule.get("name", ""))
    variables = "_".join(sorted(normalize(variable) for variable in rule_variables(rule)))
    text = f"{rid}_{name}_{variables}"
    groups: set[str] = set()
    if "pool" in text and ("latch" in text or "barrier" in text):
        groups.add("pool_barrier_latch")
    if "address_id" in text or "address_identification" in text:
        groups.add("address_identification")
        if "min_size" in text or "minimum_size" in text:
            groups.add("address_identification_size")
        if "additional_locations" in text:
            groups.add("address_identification_location_neighbor")
    if "fire_flow_table_row" in text:
        groups.add("fire_flow_table")
    if "fire_flow_calculation_area_definition" in text or "b104_1" in text:
        groups.add("fire_flow_definition")
    if "residential_fueling_appliance" in text or "rfa_capacity" in text:
        groups.add("residential_fueling_appliance")
    if "metal_hydride" in text:
        groups.add("metal_hydride")
        if "ownership_control" in text:
            groups.add("metal_hydride_neighbor")
    if "patient_bed" in text or ("moe_door" in text and "bed" in text):
        groups.add("patient_bed_door")
        if "exception" in text:
            groups.add("patient_bed_door_exception")
        else:
            if "min_width" in text or "minimum_clear_width" in text:
                groups.add("patient_bed_door_width")
            if "min_height" in text or "minimum_height" in text:
                groups.add("patient_bed_door_height")
    if "door_clear_width_min" in text or "door_clear_height_min" in text:
        groups.add("generic_door_clearance")
    if "area_of_refuge" in text or "wheelchair" in text and "refuge" in text:
        groups.add("area_refuge")
        if "1009_6_4" in text:
            groups.add("area_refuge_core")
        if "signage" in text or "exemption" in text or "1013_4" in text:
            groups.add("area_refuge_neighbor")
    if "907_6_4" in text or "fire_alarm_zone" in text or "zone_area" in text or "zone_dimension" in text:
        groups.add("fire_alarm_zone")
    if "audible_alarm" in text or "sound_pressure" in text:
        groups.add("fire_alarm_audibility")
    if "door_bottom_clearance" in text:
        groups.add("door_bottom_clearance")
    if "exterior_door_landing_drop" in text or "landing_height_differential" in text:
        groups.add("exterior_door_landing")
    if "tent" in text or "membrane" in text:
        groups.add("temporary_tent")
        if "aggregate_area" in text:
            groups.add("tent_area")
        if "clearance" in text:
            groups.add("tent_clearance")
    if "curtain" in text or "refueling_distance" in text or "flammable_liquid_fueled" in text:
        groups.add("temporary_tent_neighbor")
    if "crowd_manager" in text or "membrane_permit" in text or "open_sided" in text or "recreational_camping" in text:
        groups.add("temporary_tent_neighbor")
    if "stairway_ramp_signage" in text or "occupant_threshold" in text or "support_rope" in text:
        groups.add("temporary_tent_neighbor")
    if "flammable_liquid_storage_distance" in text:
        groups.add("temporary_tent_neighbor")
    if "flue_space" in text:
        groups.add("rack_flue_space")
        if "longitudinal" in text:
            groups.add("rack_flue_longitudinal")
        if "transverse" in text:
            groups.add("rack_flue_transverse")
    if "storage_height" in text or "sprinkler_max_20ft" in text:
        groups.add("rack_storage_height")
        if "ceiling_sprinkler_max_20ft" in text:
            groups.add("rack_storage_height_ceiling_sprinkler")
        if "palletized" in text or "shelving" in text or "hpcc" in text or "exception_5gal" in text:
            groups.add("rack_storage_neighbor")
    if "flammable_liquid_storage_height" in text or "combustible_liquid_storage" in text:
        groups.add("rack_storage_neighbor")
    if "5004_10" in text or "emergency_alarm" in text and "hazardous" in text:
        groups.add("hazmat_alarm")
    if "5705_4_8" in text or "5705_4_9" in text or "solvent_distillation" in text:
        groups.add("solvent_distillation")
        if "5705_4_8" in text or "5705_4_9" in text:
            groups.add("solvent_distillation_core")
        if "5705_4_6" in text or "solvent_recycling" in text:
            groups.add("solvent_distillation_neighbor")
    if "5505_4_1" in text or "cryogenic" in text:
        groups.add("cryogenic_dispensing")
        if "5505_4_1_1_ventilation" in text or "5505_4_1_building_code" in text:
            groups.add("cryogenic_dispensing_core")
        elif "5505_4_1_1_exception" in text:
            groups.add("cryogenic_dispensing_neighbor")
        else:
            groups.add("cryogenic_storage_neighbor")
    if "oxidizer" in text:
        groups.add("oxidizer_storage")
        if "6303_1_5_class3" in text:
            groups.add("oxidizer_class3_storage")
        else:
            groups.add("oxidizer_storage_neighbor")
    if "roof_hatch" in text:
        groups.add("roof_hatch_guard")
    if "ramp" in text and ("landing" in text or "slope" in text):
        groups.add("egress_ramp")
        if "ramp_slope_egress" in text or "ramp_landing_size_change" in text:
            groups.add("egress_ramp_core")
        if "accessible" in text or "405_7_4" in text or rid.startswith("ada_ramp_landing"):
            groups.add("accessible_ramp_landing")
        if "exception" in text:
            groups.add("ramp_landing_exception")
        if "signage" in text or "landing_extension" in text or "ramp_threshold" in text:
            groups.add("egress_ramp_neighbor")
        if "non_accessible_route" in text:
            groups.add("accessible_ramp_neighbor")
    if "5005_3_9" in text or "outdoor_hazardous" in text or "clearance_30ft" in text:
        groups.add("hazmat_outdoor_clearance")
    if "weather_protection_outdoor_use" in text:
        groups.add("hazmat_outdoor_neighbor")
    if "914_3" in text or "secondary_water_supply" in text:
        groups.add("secondary_water_supply")
        if "sec914_3_3_water_supply_duration" in text:
            groups.add("secondary_water_supply_duration")
        else:
            groups.add("secondary_water_supply_neighbor")
    if "3311_1" in text or "construction_site" in text:
        groups.add("construction_site_access")
        if "vehicle_access" in text:
            groups.add("construction_vehicle_access")
        if "emergency_contact" in text:
            groups.add("construction_emergency_contact")
    if "fire_apparatus_access" in text or "apparatus_access" in text:
        groups.add("fire_apparatus_access_neighbor")
    if "flammable_combustible_liquids_construction_site" in text:
        groups.add("construction_hazmat_neighbor")
    if "toilet_compartment" in text:
        groups.add("toilet_compartment")
    if "shower" in text:
        groups.add("accessible_shower")
    if "storage_accessibility" in text or "clear_floor_space" in text or "305_3" in text:
        groups.add("accessible_storage")
        if "811_2_clear_floor_space" in text:
            groups.add("accessible_storage_neighbor")
    if "703_5_5" in text or "font_proportion" in text or "visual_character" in text:
        groups.add("accessible_sign")
    if "emergency_power" in text:
        groups.add("emergency_power")
        if "ibc2021_emergency_power_requirement" in text:
            groups.add("ibc_emergency_power_requirement")
        else:
            groups.add("emergency_power_neighbor")
    if "ifc_k101" in text or "ifc_k102" in text or "ambulatory_care" in text:
        groups.add("existing_ambulatory_care")
        if "ifc_k101_1_scope" in text:
            groups.add("existing_ambulatory_scope")
        if "ifc_k102_1_separation" in text:
            groups.add("existing_ambulatory_separation")
        if "ifc_k101_2" in text or "ifc_k102_2" in text:
            groups.add("existing_ambulatory_neighbor")
    if "907_5_2_2" in text or "emergency_voice_alarm" in text or "voice_alarm_communication" in text:
        groups.add("emergency_voice_alarm")
        if "paging_zone" in text or "paging_zones" in text:
            groups.add("voice_alarm_paging")
        if "precedence" in text or "priority" in text or "manual_fire_alarm" in text:
            groups.add("voice_alarm_precedence")
    if "1009_2_1" in text or "elevator_requirement" in text:
        groups.add("accessible_egress_elevator")
    if "1004_6" in text or "occupant_load_increase" in text:
        groups.add("occupant_load")
    if "1006_3_4" in text or "single_exit" in text:
        groups.add("single_exit")
        if "table1006_3_4" in text or "smoke_detectors" in text:
            groups.add("single_exit_neighbor")
    if "two_exits" in text:
        groups.add("two_exit_neighbor")
    if "group_f_hazardous_materials" in text or "403_6_group_f" in text:
        groups.add("group_f_hazmat_neighbor")
    if "vertical_construction" in text or "type_iii_iv_v" in text:
        groups.add("construction_type_neighbor")
    return groups or {"generic"}


def architecture_task_profile(task: dict[str, Any]) -> dict[str, Any]:
    task_type = normalize(task.get("task_type", ""))
    title = normalize(task.get("title", ""))
    scenario = architecture_scenario(task)
    require: set[str] = set()
    allow: set[str] = set()
    block: set[str] = set()
    strict = True

    def add_required(*groups: str) -> None:
        require.update(groups)
        allow.update(groups)

    if task_type == "egress_and_pool_barrier_design":
        add_required("pool_barrier_latch")
        block.add("generic_door_clearance")
    elif task_type == "fire_department_wayfinding_design":
        add_required("address_identification_size")
        block.add("address_identification_location_neighbor")
    elif task_type == "fire_flow_supply_design":
        add_required("fire_flow_table")
        block.update({"fire_flow_definition", "construction_type_neighbor"})
    elif task_type == "fueling_appliance_capacity_design":
        add_required("residential_fueling_appliance")
    elif task_type == "hazardous_material_storage_design":
        add_required("metal_hydride")
        block.add("metal_hydride_neighbor")
    elif task_type == "healthcare_egress_door_design":
        add_required("patient_bed_door_width", "patient_bed_door_height")
        allow.add("patient_bed_door")
        block.update({"patient_bed_door_exception", "generic_door_clearance"})
    elif task_type == "exception_resolution":
        add_required("patient_bed_door_exception")
        block.update({"patient_bed_door_width", "patient_bed_door_height", "generic_door_clearance"})
    elif task_type == "accessible_egress_refuge_design":
        add_required("area_refuge_core")
        allow.add("area_refuge")
        block.add("area_refuge_neighbor")
    elif task_type == "fire_alarm_zone_design":
        add_required("fire_alarm_zone")
    elif task_type == "fire_alarm_audibility_design":
        add_required("fire_alarm_audibility")
    elif task_type == "egress_door_threshold_design":
        add_required("door_bottom_clearance", "exterior_door_landing")
        block.add("group_f_hazmat_neighbor")
    elif task_type == "temporary_structure_site_design":
        add_required("tent_area", "tent_clearance")
        allow.add("temporary_tent")
        block.add("temporary_tent_neighbor")
    elif task_type == "sprinklered_rack_storage_design":
        add_required("rack_flue_transverse", "rack_storage_height_ceiling_sprinkler")
        allow.add("rack_flue_space")
        block.update({"rack_flue_longitudinal", "rack_storage_neighbor"})
    elif task_type == "hazardous_material_alarm_design":
        add_required("hazmat_alarm")
    elif task_type == "hazardous_process_safety_design":
        add_required("solvent_distillation_core")
        allow.add("solvent_distillation")
        block.add("solvent_distillation_neighbor")
    elif task_type == "cryogenic_dispensing_room_design":
        add_required("cryogenic_dispensing_core")
        allow.add("cryogenic_dispensing")
        block.update({"cryogenic_storage_neighbor", "cryogenic_dispensing_neighbor"})
    elif task_type == "oxidizer_storage_design":
        add_required("oxidizer_class3_storage")
        block.add("oxidizer_storage_neighbor")
    elif task_type == "roof_access_fall_protection_design":
        add_required("roof_hatch_guard")
    elif task_type == "egress_ramp_design":
        add_required("egress_ramp_core")
        allow.add("egress_ramp")
        block.update({"accessible_ramp_landing", "ramp_landing_exception", "egress_ramp_neighbor"})
    elif task_type == "hazardous_material_site_planning":
        add_required("hazmat_outdoor_clearance")
        block.add("hazmat_outdoor_neighbor")
    elif task_type == "fire_protection_water_supply_design":
        add_required("secondary_water_supply_duration")
        block.add("secondary_water_supply_neighbor")
    elif task_type == "construction_site_access_design":
        add_required("construction_vehicle_access", "construction_emergency_contact")
        allow.add("construction_site_access")
        block.update({"fire_apparatus_access_neighbor", "construction_hazmat_neighbor"})
    elif task_type == "accessible_toilet_compartment_design":
        add_required("toilet_compartment")
    elif task_type == "accessible_shower_room_design":
        add_required("accessible_shower")
    elif task_type == "accessible_storage_layout_design":
        add_required("accessible_storage")
        block.add("accessible_storage_neighbor")
    elif task_type == "accessible_wayfinding_sign_design":
        add_required("accessible_sign")
    elif task_type == "accessible_ramp_landing_design":
        add_required("accessible_ramp_landing")
        block.add("accessible_ramp_neighbor")
    elif task_type == "accessible_means_of_egress_design":
        add_required("accessible_egress_elevator", "ibc_emergency_power_requirement")
        block.add("emergency_power_neighbor")
    elif task_type == "existing_ambulatory_care_precedence_design":
        add_required("existing_ambulatory_scope", "existing_ambulatory_separation")
        allow.add("existing_ambulatory_care")
        block.add("existing_ambulatory_neighbor")
    elif task_type == "emergency_voice_alarm_precedence_design":
        add_required("voice_alarm_paging", "voice_alarm_precedence")
        allow.add("emergency_voice_alarm")
    elif task_type == "egress_layout_and_occupant_load_design":
        add_required("occupant_load", "single_exit")
        block.update({"two_exit_neighbor", "single_exit_neighbor"})
    else:
        strict = False

    if "ibc" not in normalize(scenario.get("code_context", "")) and task_type != "egress_layout_and_occupant_load_design":
        block.add("occupant_load")
    if "accessible" not in task_type and "accessible" not in title:
        block.update({"toilet_compartment", "accessible_shower", "accessible_storage", "accessible_sign"})
    return {"require": require, "allow": allow, "block": block, "strict": strict}


def guard_fields(guard: Any) -> set[str]:
    if not guard:
        return set()
    if isinstance(guard, list):
        out: set[str] = set()
        for item in guard:
            out |= guard_fields(item)
        return out
    if not isinstance(guard, dict):
        return set()
    out: set[str] = set()
    for key in ("all", "any"):
        if key in guard:
            for item in guard.get(key, []):
                out |= guard_fields(item)
    if "not" in guard:
        out |= guard_fields(guard["not"])
    if "field" in guard:
        out.add(str(guard["field"]))
    return out


def rule_variables(rule: dict[str, Any]) -> set[str]:
    return {
        str(constraint.get("variable"))
        for constraint in rule.get("constraints", [])
        if constraint.get("variable")
    }


def visible_source_domains(task: dict[str, Any]) -> set[str]:
    domains = set(str(item) for item in task.get("source_domains", []) if item)
    stress_meta = task.get("stress_metadata") or {}
    domains.update(str(item) for item in stress_meta.get("source_domains", []) if item)
    source_domain = str(task.get("source_domain", "") or "")
    if source_domain and source_domain.lower() != "mixed":
        domains.add(source_domain)
    return domains


def source_domain_matches(rule: dict[str, Any], task: dict[str, Any]) -> bool:
    rule_source_domain = str(rule.get("source_domain") or "")
    if not rule_source_domain:
        return str(rule.get("domain", "")).lower() == "architecture"
    domains = visible_source_domains(task)
    if domains:
        return rule_source_domain in domains
    source_domain = str(task.get("source_domain", "") or "")
    return not source_domain or source_domain.lower() == "mixed" or rule_source_domain == source_domain


def visible_task_tokens(task: dict[str, Any]) -> set[str]:
    stress_meta = task.get("stress_metadata") or {}
    visible_metadata = {
        "source_domain": task.get("source_domain"),
        "source_domains": stress_meta.get("source_domains"),
        "task_type": task.get("task_type"),
        "engineering_task": task.get("engineering_task"),
        "design_intent": task.get("design_intent"),
        "scenario_facts": task.get("scenario_facts", {}),
        "decision_variables": list(task.get("decision_variables", {})),
    }
    return token_set(visible_metadata)


def rule_tokens(rule: dict[str, Any]) -> set[str]:
    visible_rule_metadata = {
        "rule_id": rule.get("rule_id"),
        "name": rule.get("name"),
        "source_domain": rule.get("source_domain"),
        "rule_type": rule.get("rule_type"),
        "guard": rule.get("guard"),
        "constraints": [
            {
                "variable": constraint.get("variable"),
                "unit": constraint.get("unit"),
            }
            for constraint in rule.get("constraints", [])
        ],
        "relations": rule.get("relations", []),
    }
    return token_set(visible_rule_metadata)


def variable_match_count(rule_vars: set[str], decision_vars: set[str]) -> int:
    count = 0
    for rule_var in rule_vars:
        rule_norm = normalize(rule_var)
        rule_tokens = token_set(rule_var)
        for decision_var in decision_vars:
            decision_norm = normalize(decision_var)
            decision_tokens = token_set(decision_var)
            if rule_norm == decision_norm or len(rule_tokens & decision_tokens) >= 2:
                count += 1
                break
    return count


def variable_matches_task(rule_vars: set[str], decision_vars: set[str]) -> bool:
    return variable_match_count(rule_vars, decision_vars) > 0


def is_empty_guard(rule: dict[str, Any]) -> bool:
    guard = rule.get("guard")
    return not guard or guard == {"all": []} or guard == {"any": []}


def is_formula_rule(rule: dict[str, Any]) -> bool:
    return str(rule.get("rule_id", "")).startswith("FORMULA_") or "formula" in str(rule.get("rule_type", "")).lower()


def is_precedence_rule(rule: dict[str, Any]) -> bool:
    rule_id = str(rule.get("rule_id", ""))
    if rule_id.startswith("XCODE_"):
        return True
    for relation in rule.get("relations", []):
        if str(relation.get("type", "")).lower() in {"precedes", "precedence", "higher_priority_than"}:
            return True
    return False


def rule_units(rule: dict[str, Any]) -> set[str]:
    return {
        normalize(constraint.get("unit"))
        for constraint in rule.get("constraints", [])
        if constraint.get("unit")
    }


def task_units(task: dict[str, Any]) -> set[str]:
    return {
        normalize(spec.get("unit"))
        for spec in task.get("decision_variables", {}).values()
        if spec.get("unit")
    }


def relation_type(relation: dict[str, Any]) -> str:
    return str(relation.get("type", "")).lower()


def relation_target(relation: dict[str, Any]) -> str:
    return str(relation.get("target", ""))


def rule_relation_types(rule: dict[str, Any]) -> set[str]:
    return {relation_type(relation) for relation in rule.get("relations", [])}


def relation_indexes(rules: list[dict[str, Any]]) -> tuple[dict[str, list[tuple[str, str]]], dict[str, list[tuple[str, str]]]]:
    ids = {str(rule.get("rule_id")) for rule in rules if rule.get("rule_id")}
    outgoing = {rule_id: [] for rule_id in ids}
    incoming = {rule_id: [] for rule_id in ids}
    for rule in rules:
        source = str(rule.get("rule_id"))
        for relation in rule.get("relations", []):
            target = relation_target(relation)
            if target not in ids:
                continue
            rtype = relation_type(relation)
            outgoing[source].append((rtype, target))
            incoming[target].append((rtype, source))
    return outgoing, incoming


def include_relation_neighbor(
    rtype: str,
    direction: str,
    task_type: str,
) -> bool:
    if rtype in DEPENDENCY_TYPES:
        return direction == "out" or "dependency" in task_type or "parameter" in task_type
    if rtype in COMPETITION_TYPES:
        return "exclusion" in task_type or "scenario" in task_type
    if rtype in OVERRIDE_TYPES:
        return True
    if rtype in PRECEDENCE_TYPES:
        return direction == "out" or "precedence" in task_type
    if rtype in PARAMETER_VARIANT_TYPES:
        return "parameter" in task_type or "dependency" in task_type or "scenario" in task_type
    return False


def relation_intent(task_type: str) -> set[str]:
    intent = set()
    if any(token in task_type for token in ("dependency", "parameter", "formula", "propagation")):
        intent.add("dependency")
    if any(token in task_type for token in ("exclusion", "alternative", "branch", "scenario", "applicability")):
        intent.add("competition")
    if any(token in task_type for token in ("override", "exception")):
        intent.add("override")
    if any(token in task_type for token in ("precedence", "priority")):
        intent.add("precedence")
    return intent


def rule_family(rule: dict[str, Any]) -> str:
    if rule.get("conflict_class") or rule.get("conflict_group"):
        return f"conflict:{rule.get('conflict_class') or rule.get('conflict_group')}"
    variables = sorted(normalize(variable) for variable in rule_variables(rule))
    if variables:
        return f"var:{variables[0]}"
    fields = sorted(normalize(field) for field in guard_fields(rule.get("guard")))
    if fields:
        return f"guard:{fields[0]}"
    rid_tokens = [token for token in normalize(rule.get("rule_id", "")).split("_") if token]
    return "id:" + "_".join(rid_tokens[:3])


def relation_compatible(
    source: dict[str, Any],
    neighbor: dict[str, Any],
    rtype: str,
    direction: str,
    task_type: str,
    task_tokens: set[str],
    decision_vars: set[str],
    scenario_fields: set[str],
) -> tuple[bool, str]:
    _relevant, overlap, variable_match = direct_task_relevance(neighbor, task_tokens, decision_vars)
    source_family = rule_family(source)
    neighbor_family = rule_family(neighbor)
    neighbor_fields = {normalize(field) for field in guard_fields(neighbor.get("guard"))}
    same_guard_field = bool(neighbor_fields & scenario_fields)
    intent = relation_intent(task_type)

    if rtype in DEPENDENCY_TYPES:
        if direction == "out":
            return True, f"typed_relation_out:{rtype}"
        if "dependency" in intent and (variable_match or overlap >= 2 or same_guard_field):
            return True, f"typed_relation_in:{rtype}"
        return False, "dependency_relation_not_task_visible"
    if rtype in PARAMETER_VARIANT_TYPES:
        if "dependency" in intent or variable_match or source_family == neighbor_family:
            return True, f"typed_relation_{direction}:{rtype}"
        return False, "parameter_variant_not_visible"
    if rtype in COMPETITION_TYPES:
        if (
            "competition" in intent
            and (source_family == neighbor_family or variable_match or same_guard_field or overlap >= 2)
        ):
            return True, f"typed_relation_{direction}:{rtype}"
        return False, "competition_relation_not_same_visible_family"
    if rtype in OVERRIDE_TYPES:
        if "override" in intent or same_guard_field or variable_match or overlap >= 2:
            return True, f"typed_relation_{direction}:{rtype}"
        return False, "override_relation_not_visible"
    if rtype in PRECEDENCE_TYPES:
        if "precedence" in intent or variable_match or same_guard_field or overlap >= 2:
            return True, f"typed_relation_{direction}:{rtype}"
        return False, "precedence_relation_not_visible"
    return False, "relation_type_not_grounded"


def direct_task_relevance(
    rule: dict[str, Any],
    task_tokens: set[str],
    decision_vars: set[str],
) -> tuple[bool, int, bool]:
    rule_var_set = rule_variables(rule)
    variable_match = variable_matches_task(rule_var_set, decision_vars)
    token_overlap = len(rule_tokens(rule) & task_tokens)
    strong_text_match = token_overlap >= 3
    return variable_match or strong_text_match, token_overlap, variable_match


def typed_grounding_score(
    rule: dict[str, Any],
    task: dict[str, Any],
    scenario: dict[str, Any],
    task_tokens: set[str],
    decision_vars: set[str],
    scenario_fields: set[str],
    units: set[str],
    grounding_mode: str = "strict_profile",
) -> tuple[float, list[str]]:
    guard = rule.get("guard")
    empty_guard = is_empty_guard(rule)
    status = "empty" if empty_guard else ctv.eval_guard(guard, scenario)
    fields = {normalize(field) for field in guard_fields(guard)}
    field_overlap = len(fields & scenario_fields)
    variable_matches = variable_match_count(rule_variables(rule), decision_vars)
    token_overlap = len(rule_tokens(rule) & task_tokens)
    unit_overlap = len(rule_units(rule) & units)
    rtypes = rule_relation_types(rule)
    rtype_bonus = 0.0
    if rtypes & (DEPENDENCY_TYPES | PARAMETER_VARIANT_TYPES):
        rtype_bonus += 0.6
    if rtypes & (COMPETITION_TYPES | OVERRIDE_TYPES | PRECEDENCE_TYPES):
        rtype_bonus += 0.6
    groups = architecture_rule_groups(rule)
    profile = architecture_task_profile(task)
    profile_required = bool(groups & profile["require"])
    profile_allowed = bool(groups & profile["allow"])
    profile_blocked = bool((groups & profile["block"]) - (groups & profile["require"]))
    profile_unmatched = bool(profile["strict"]) and not (profile_required or profile_allowed)

    score = 0.0
    reasons: list[str] = []
    if profile_required:
        score += 7.0
        reasons.append("domain_profile_required:" + ",".join(sorted(groups & profile["require"])))
    elif profile_allowed:
        score += 3.0
        reasons.append("domain_profile_allowed:" + ",".join(sorted(groups & profile["allow"])))
    if profile_blocked:
        if grounding_mode == "strict_profile":
            score -= 9.0
        elif grounding_mode == "relation_rich":
            score -= 2.0
        else:
            score -= 2.0
        reasons.append("domain_profile_blocked:" + ",".join(sorted(groups & profile["block"])))
    elif profile_unmatched:
        if grounding_mode == "strict_profile":
            score -= 5.0
        elif grounding_mode == "relation_rich":
            score -= 0.5
        else:
            score -= 1.0
        reasons.append("domain_profile_unmatched")
    if field_overlap:
        score += 2.5 * field_overlap
        reasons.append("guard_field_visible")
    if variable_matches:
        score += 3.0 + variable_matches
        reasons.append("decision_variable_bound")
    if unit_overlap:
        score += 0.8
        reasons.append("unit_match")
    if token_overlap:
        score += min(3.0, 0.55 * token_overlap)
        reasons.append(f"token_overlap:{token_overlap}")
    if not empty_guard and status == "true":
        score += 3.0
        reasons.append("guard_true")
    elif not empty_guard and status == "unknown" and (variable_matches or field_overlap):
        score += 0.8
        reasons.append("guard_unknown_but_bound")
    elif not empty_guard and status == "false":
        score -= 1.5
        reasons.append("guard_false")
    if is_formula_rule(rule):
        if variable_matches or "dependency" in relation_intent(normalize(task.get("task_type", ""))):
            score += 1.0
            reasons.append("formula_visible")
        else:
            score -= 2.0
            reasons.append("formula_unbound")
    if empty_guard and not variable_matches:
        score -= 1.2
        reasons.append("empty_guard_without_variable_binding")
    if str(task.get("source_domain", "")).lower() == "mixed" and is_precedence_rule(rule):
        score += 1.0
        reasons.append("mixed_domain_precedence")
    score += rtype_bonus

    if grounding_mode in {"relation_rich", "relation_stress"}:
        seed_min_score = (
            RELATION_STRESS_SEED_MIN_SCORE
            if grounding_mode == "relation_stress"
            else RELATION_RICH_SEED_MIN_SCORE
        )
        token_floor = 1 if grounding_mode == "relation_stress" else 2
        empty_guard_token_floor = 2 if grounding_mode == "relation_stress" else 3
        relation_signal = bool(rtypes & RELATION_RICH_RELATION_TYPES) and (
            profile_required or profile_allowed or variable_matches or field_overlap or unit_overlap or token_overlap >= 1
        )
        visible_signal = bool(
            profile_required
            or profile_allowed
            or variable_matches
            or field_overlap
            or unit_overlap
            or token_overlap >= token_floor
        )
        if score < seed_min_score and not relation_signal:
            return score, []
        if grounding_mode == "relation_rich" and profile_blocked and not (relation_signal or visible_signal):
            return score, []
        if (
            grounding_mode == "relation_rich"
            and not empty_guard
            and status == "false"
            and not (relation_signal or profile_required or field_overlap)
        ):
            return score, []
        if (
            empty_guard
            and not profile_required
            and not variable_matches
            and token_overlap < empty_guard_token_floor
            and not relation_signal
        ):
            return score, []
        if grounding_mode == "relation_stress" and not (visible_signal or relation_signal):
            return score, []
        reasons.append(f"grounding_mode:{grounding_mode}")
        return score, reasons

    if profile_blocked or profile_unmatched:
        return score, []
    if score < TYPED_SEED_MIN_SCORE:
        return score, []
    if not empty_guard and status == "false" and not (
        profile_required
        or ("competition" in relation_intent(normalize(task.get("task_type", ""))) and field_overlap)
    ):
        return score, []
    if empty_guard and not profile_required and not variable_matches and token_overlap < 4 and not is_precedence_rule(rule):
        return score, []
    return score, reasons


def select_top_rules_by_family(
    scored: dict[str, tuple[float, list[str]]],
    rule_by_id: dict[str, dict[str, Any]],
    grounding_mode: str = "strict_profile",
) -> set[str]:
    grouped: dict[str, list[tuple[float, str]]] = {}
    for rule_id, (score, _reasons) in scored.items():
        groups = sorted(architecture_rule_groups(rule_by_id[rule_id]) - {"generic"})
        family = f"architecture:{groups[0]}" if groups else rule_family(rule_by_id[rule_id])
        grouped.setdefault(family, []).append((score, rule_id))
    selected: set[str] = set()
    if grounding_mode == "relation_stress":
        top_k = RELATION_STRESS_FAMILY_TOP_K
    elif grounding_mode == "relation_rich":
        top_k = RELATION_RICH_FAMILY_TOP_K
    else:
        top_k = TYPED_FAMILY_TOP_K
    for items in grouped.values():
        for _score, rule_id in sorted(items, key=lambda item: (-item[0], item[1]))[:top_k]:
            selected.add(rule_id)
    return selected


def apply_architecture_profile_filter(
    selected: set[str],
    scored: dict[str, tuple[float, list[str]]],
    candidate_reasons: dict[str, list[str]],
    rule_by_id: dict[str, dict[str, Any]],
    task: dict[str, Any],
) -> set[str]:
    profile = architecture_task_profile(task)
    out = set(selected)
    for rule_id in list(out):
        groups = architecture_rule_groups(rule_by_id[rule_id])
        required = bool(groups & profile["require"])
        allowed = bool(groups & profile["allow"])
        blocked = bool((groups & profile["block"]) - (groups & profile["require"]))
        unmatched = bool(profile["strict"]) and not (required or allowed)
        if blocked or unmatched:
            out.remove(rule_id)
            candidate_reasons.setdefault(rule_id, []).append(
                "domain_profile_pruned:" + ("blocked" if blocked else "unmatched")
            )
    for rule_id, (_score, reasons) in scored.items():
        groups = architecture_rule_groups(rule_by_id[rule_id])
        if groups & profile["require"] and not ((groups & profile["block"]) - (groups & profile["require"])):
            out.add(rule_id)
            candidate_reasons.setdefault(rule_id, []).extend(
                ["domain_profile_required_reinjected", *reasons]
            )
    return out


def relation_rich_neighbor_compatible(
    source: dict[str, Any],
    neighbor: dict[str, Any],
    rtype: str,
    direction: str,
    task: dict[str, Any],
    task_tokens: set[str],
    decision_vars: set[str],
    scenario_fields: set[str],
    grounding_mode: str = "relation_rich",
) -> tuple[bool, str]:
    if rtype not in RELATION_RICH_RELATION_TYPES:
        return False, "relation_type_not_relation_rich"
    if not source_domain_matches(neighbor, task):
        return False, "relation_neighbor_wrong_domain"
    _relevant, overlap, variable_match = direct_task_relevance(neighbor, task_tokens, decision_vars)
    source_family = rule_family(source)
    neighbor_family = rule_family(neighbor)
    neighbor_fields = {normalize(field) for field in guard_fields(neighbor.get("guard"))}
    same_guard_field = bool(neighbor_fields & scenario_fields)
    groups = architecture_rule_groups(neighbor)
    profile = architecture_task_profile(task)
    profile_visible = bool(groups & (profile["require"] | profile["allow"]))
    same_family = source_family == neighbor_family
    task_type = normalize(task.get("task_type", ""))
    intent = relation_intent(task_type)
    visible = profile_visible or variable_match or same_guard_field or overlap >= 2 or same_family

    if grounding_mode == "relation_stress":
        weak_visible = profile_visible or variable_match or same_guard_field or overlap >= 1 or same_family
        if rtype in DEPENDENCY_TYPES and weak_visible:
            return True, f"relation_stress_{direction}:{rtype}"
        if rtype in (PARAMETER_VARIANT_TYPES | COMPETITION_TYPES | OVERRIDE_TYPES | PRECEDENCE_TYPES) and weak_visible:
            return True, f"relation_stress_{direction}:{rtype}"
        return False, "relation_stress_neighbor_not_visible"

    if rtype in DEPENDENCY_TYPES and direction == "out":
        return True, f"relation_rich_out:{rtype}"
    if rtype in DEPENDENCY_TYPES and (
        "dependency" in intent or profile_visible or (same_family and (variable_match or same_guard_field))
    ):
        return True, f"relation_rich_in:{rtype}"
    if rtype in (PARAMETER_VARIANT_TYPES | COMPETITION_TYPES | OVERRIDE_TYPES | PRECEDENCE_TYPES) and visible:
        return True, f"relation_rich_{direction}:{rtype}"
    return False, "relation_neighbor_not_visible"


def expand_relation_rich_candidates(
    selected: set[str],
    candidate_reasons: dict[str, list[str]],
    rule_by_id: dict[str, dict[str, Any]],
    outgoing: dict[str, list[tuple[str, str]]],
    incoming: dict[str, list[tuple[str, str]]],
    task: dict[str, Any],
    task_tokens: set[str],
    decision_vars: set[str],
    scenario_fields: set[str],
    grounding_mode: str = "relation_rich",
) -> set[str]:
    out = set(selected)
    frontier = set(selected)
    max_hops = RELATION_STRESS_MAX_HOPS if grounding_mode == "relation_stress" else RELATION_RICH_MAX_HOPS
    for hop in range(1, max_hops + 1):
        new_frontier: set[str] = set()
        for rule_id in sorted(frontier):
            source = rule_by_id[rule_id]
            for direction, edges in (("out", outgoing.get(rule_id, [])), ("in", incoming.get(rule_id, []))):
                for rtype, neighbor_id in edges:
                    if neighbor_id not in rule_by_id or neighbor_id in out:
                        continue
                    neighbor = rule_by_id[neighbor_id]
                    ok, reason = relation_rich_neighbor_compatible(
                        source,
                        neighbor,
                        rtype,
                        direction,
                        task,
                        task_tokens,
                        decision_vars,
                        scenario_fields,
                        grounding_mode=grounding_mode,
                    )
                    if ok:
                        out.add(neighbor_id)
                        new_frontier.add(neighbor_id)
                        candidate_reasons.setdefault(neighbor_id, []).append(f"{reason}:hop{hop}")
        if not new_frontier:
            break
        frontier = new_frontier
    return out


def relation_stress_candidate_rank(
    rule_id: str,
    scored: dict[str, tuple[float, list[str]]],
    candidate_reasons: dict[str, list[str]],
    rule_by_id: dict[str, dict[str, Any]],
    task: dict[str, Any],
    task_tokens: set[str],
    decision_vars: set[str],
    scenario_fields: set[str],
) -> tuple[float, float, str]:
    rule = rule_by_id[rule_id]
    score, scored_reasons = scored.get(rule_id, (0.0, []))
    reasons = candidate_reasons.get(rule_id, []) + scored_reasons
    groups = architecture_rule_groups(rule)
    profile = architecture_task_profile(task)
    rank = score
    if groups & profile["require"]:
        rank += 100.0
    if groups & profile["allow"]:
        rank += 25.0
    if any("typed_seed_score" in reason for reason in reasons):
        rank += 12.0
    if any("relation_stress" in reason or "relation_rich" in reason for reason in reasons):
        rank += 6.0
    if variable_match_count(rule_variables(rule), decision_vars):
        rank += 5.0
    rule_fields = {normalize(field) for field in guard_fields(rule.get("guard"))}
    if rule_fields & scenario_fields:
        rank += 4.0
    rank += min(4.0, 0.45 * len(rule_tokens(rule) & task_tokens))
    return rank, score, rule_id


def shape_relation_stress_candidates(
    selected: set[str],
    scored: dict[str, tuple[float, list[str]]],
    candidate_reasons: dict[str, list[str]],
    rule_by_id: dict[str, dict[str, Any]],
    task: dict[str, Any],
    task_tokens: set[str],
    decision_vars: set[str],
    scenario_fields: set[str],
) -> set[str]:
    out = set(selected)
    ranked_scored = sorted(
        scored,
        key=lambda rid: relation_stress_candidate_rank(
            rid, scored, candidate_reasons, rule_by_id, task, task_tokens, decision_vars, scenario_fields
        ),
        reverse=True,
    )
    for rule_id in ranked_scored:
        if len(out) >= RELATION_STRESS_MIN_CANDIDATES:
            break
        out.add(rule_id)
        candidate_reasons.setdefault(rule_id, []).append("relation_stress_rank_fill")
    if len(out) <= RELATION_STRESS_MAX_CANDIDATES:
        return out
    protected = {
        rid
        for rid in out
        if any("domain_profile_required" in reason for reason in candidate_reasons.get(rid, []))
    }
    ranked_out = sorted(
        out - protected,
        key=lambda rid: relation_stress_candidate_rank(
            rid, scored, candidate_reasons, rule_by_id, task, task_tokens, decision_vars, scenario_fields
        ),
        reverse=True,
    )
    trimmed = set(ranked_out[: max(0, RELATION_STRESS_MAX_CANDIDATES - len(protected))]) | protected
    for rule_id in sorted(out - trimmed):
        candidate_reasons.setdefault(rule_id, []).append("relation_stress_rank_pruned_for_llm_budget")
    return trimmed


def generate_architecture_candidates(
    rule_library: dict[str, Any],
    task: dict[str, Any],
    grounding_mode: str = "strict_profile",
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Generate typed symbolic candidates from visible task fields and rule metadata only."""
    if grounding_mode not in GROUNDING_MODES:
        raise ValueError(f"Unsupported grounding_mode: {grounding_mode}")
    rules = [rule for rule in rule_library.get("rules", []) if rule.get("rule_id")]
    rule_by_id = {str(rule["rule_id"]): rule for rule in rules}
    outgoing, incoming = relation_indexes(rules)

    scenario = task.get("scenario_facts", {})
    decision_vars = set(str(name) for name in task.get("decision_variables", {}))
    task_tokens = visible_task_tokens(task)
    scenario_fields = {normalize(field) for field in scenario}
    task_type = normalize(task.get("task_type", ""))
    units = task_units(task)

    scored: dict[str, tuple[float, list[str]]] = {}
    candidate_reasons: dict[str, list[str]] = {}
    if grounding_mode == "relation_rich":
        score_mode = "strict_profile"
    elif grounding_mode == "relation_stress":
        score_mode = "relation_rich"
    else:
        score_mode = grounding_mode

    for rule in rules:
        rule_id = str(rule["rule_id"])
        if not source_domain_matches(rule, task):
            continue
        score, reasons = typed_grounding_score(
            rule,
            task,
            scenario,
            task_tokens,
            decision_vars,
            scenario_fields,
            units,
            grounding_mode=score_mode,
        )
        if reasons:
            scored[rule_id] = (score, reasons)

    selected = select_top_rules_by_family(scored, rule_by_id, score_mode)
    for rule_id in selected:
        score, reasons = scored[rule_id]
        candidate_reasons.setdefault(rule_id, []).extend([f"typed_seed_score:{score:.2f}", *reasons])

    if grounding_mode in {"relation_rich", "relation_stress"}:
        selected = apply_architecture_profile_filter(selected, scored, candidate_reasons, rule_by_id, task)
        if grounding_mode == "relation_stress":
            profile = architecture_task_profile(task)
            for rule_id, (_score, reasons) in scored.items():
                groups = architecture_rule_groups(rule_by_id[rule_id])
                if groups & profile["require"]:
                    selected.add(rule_id)
                    candidate_reasons.setdefault(rule_id, []).extend(
                        ["domain_profile_required_reinjected", *reasons]
                    )
        selected = expand_relation_rich_candidates(
            selected,
            candidate_reasons,
            rule_by_id,
            outgoing,
            incoming,
            task,
            task_tokens,
            decision_vars,
            scenario_fields,
            grounding_mode=grounding_mode,
        )
        if grounding_mode == "relation_stress":
            selected = shape_relation_stress_candidates(
                selected,
                scored,
                candidate_reasons,
                rule_by_id,
                task,
                task_tokens,
                decision_vars,
                scenario_fields,
            )
        return [rule_by_id[rule_id] for rule_id in sorted(selected)], candidate_reasons

    for rule_id in list(selected):
        for rtype, target in outgoing.get(rule_id, []):
            if target not in rule_by_id or not include_relation_neighbor(rtype, "out", task_type):
                continue
            ok, reason = relation_compatible(
                rule_by_id[rule_id],
                rule_by_id[target],
                rtype,
                "out",
                task_type,
                task_tokens,
                decision_vars,
                scenario_fields,
            )
            neighbor_score = scored.get(target, (0.0, []))[0]
            if ok and (neighbor_score >= TYPED_RELATION_MIN_SCORE or rtype in DEPENDENCY_TYPES):
                selected.add(target)
                candidate_reasons.setdefault(target, []).append(reason)
        for rtype, source in incoming.get(rule_id, []):
            if source not in rule_by_id or not include_relation_neighbor(rtype, "in", task_type):
                continue
            ok, reason = relation_compatible(
                rule_by_id[rule_id],
                rule_by_id[source],
                rtype,
                "in",
                task_type,
                task_tokens,
                decision_vars,
                scenario_fields,
            )
            neighbor_score = scored.get(source, (0.0, []))[0]
            if ok and neighbor_score >= TYPED_RELATION_MIN_SCORE:
                selected.add(source)
                candidate_reasons.setdefault(source, []).append(reason)

    selected = apply_architecture_profile_filter(selected, scored, candidate_reasons, rule_by_id, task)
    return [rule_by_id[rule_id] for rule_id in sorted(selected)], candidate_reasons


def has_nonempty_guard(rule: dict[str, Any]) -> bool:
    return not is_empty_guard(rule)


def pair_protected_by_dependency(
    left: str,
    right: str,
    dependency_pairs: set[tuple[str, str]],
) -> bool:
    return (left, right) in dependency_pairs or (right, left) in dependency_pairs


def dependency_reaches(
    source: str,
    targets: set[str],
    depends: set[tuple[str, str]],
) -> bool:
    frontier = [source]
    seen: set[str] = set()
    while frontier:
        current = frontier.pop()
        if current in seen:
            continue
        seen.add(current)
        if current in targets:
            return True
        for left, right in depends:
            if left == current and right not in seen:
                frontier.append(right)
    return False


def resolve_architecture_conflict_classes(
    selected: set[str],
    candidate_rules: list[dict[str, Any]],
    scenario: dict[str, Any],
) -> tuple[set[str], list[str]]:
    by_id = {str(rule["rule_id"]): rule for rule in candidate_rules if rule.get("rule_id")}
    maps = ctv.relation_maps(candidate_rules)
    depends = set(maps["depends"])
    groups: dict[str, set[str]] = {}
    for rule in candidate_rules:
        conflict_class = rule.get("conflict_class")
        if conflict_class:
            groups.setdefault(str(conflict_class), set()).add(str(rule["rule_id"]))

    out = set(selected)
    notes: list[str] = []
    for conflict_class, members in groups.items():
        active_members = members & out
        if len(active_members) <= 1:
            continue

        true_anchors = {
            rid
            for rid in active_members
            if has_nonempty_guard(by_id[rid]) and ctv.guard_status(by_id[rid], scenario) == "true"
        }
        if true_anchors:
            keep = set(true_anchors)
            for rid in active_members - true_anchors:
                if dependency_reaches(rid, true_anchors, depends):
                    keep.add(rid)
            for rid in sorted(active_members - keep):
                out.remove(rid)
                notes.append(f"removed_inactive_conflict_branch:{conflict_class}:{rid}")
            continue

        ranked = sorted(
            active_members,
            key=lambda rid: (ctv.rule_specificity(by_id[rid], scenario), rid),
            reverse=True,
        )
        keep = {ranked[0]}
        for rid in sorted(active_members - keep):
            out.remove(rid)
            notes.append(f"removed_lower_specificity_conflict_branch:{conflict_class}:{rid}")
    return out, notes


def resolve_architecture_pairwise_conflicts(
    selected: set[str],
    candidate_rules: list[dict[str, Any]],
    scenario: dict[str, Any],
) -> tuple[set[str], list[str]]:
    by_id = {str(rule["rule_id"]): rule for rule in candidate_rules if rule.get("rule_id")}
    maps = ctv.relation_maps(candidate_rules)
    dependency_pairs = set(maps["depends"])
    conflict_pairs = set(maps["excludes"]) | set(maps["conflicts"])
    out = set(selected)
    notes: list[str] = []
    changed = True
    while changed:
        changed = False
        for left, right in sorted(conflict_pairs):
            if left not in out or right not in out:
                continue
            if pair_protected_by_dependency(left, right, dependency_pairs):
                continue
            left_class = by_id[left].get("conflict_class")
            right_class = by_id[right].get("conflict_class")
            if left_class and left_class == right_class:
                continue
            left_score = ctv.rule_specificity(by_id[left], scenario)
            right_score = ctv.rule_specificity(by_id[right], scenario)
            loser = right if left_score >= right_score else left
            out.remove(loser)
            notes.append(f"removed_pairwise_conflict:{loser}")
            changed = True
            break
    return out, notes


def cthr_recover_architecture_valid_rules(
    candidate_rules: list[dict[str, Any]],
    scenario: dict[str, Any],
) -> ctv.RecoveryResult:
    start = time.perf_counter()
    notes: list[str] = []
    try:
        by_id = {str(rule["rule_id"]): rule for rule in candidate_rules if rule.get("rule_id")}
        maps = ctv.relation_maps(candidate_rules)
        applicable = {rid for rid, rule in by_id.items() if ctv.initially_applicable(rule, scenario)}

        defeated: set[str] = set()
        for source, target in maps["overrides"]:
            if source in applicable:
                defeated.add(target)
                notes.append(f"defeated_by_override:{target}")
        for source, target in maps["precedes"]:
            if source in applicable:
                defeated.add(target)
                notes.append(f"defeated_by_precedence:{target}")

        selected = set(applicable - defeated)
        selected = ctv.dependency_closure(selected, maps["depends"], applicable - defeated)
        selected -= defeated

        selected, class_notes = resolve_architecture_conflict_classes(selected, candidate_rules, scenario)
        notes.extend(class_notes)
        selected, pair_notes = resolve_architecture_pairwise_conflicts(selected, candidate_rules, scenario)
        notes.extend(pair_notes)

        selected, variant_notes = ctv.prune_parameter_variants(selected, candidate_rules, scenario)
        notes.extend(variant_notes)
        selected, unit_notes = ctv.prune_visible_unit_mismatches(selected, candidate_rules, scenario)
        notes.extend(unit_notes)
        selected, quantity_notes = ctv.prune_unbound_visible_quantities(selected, candidate_rules, scenario)
        notes.extend(quantity_notes)

        selected = ctv.dependency_closure(selected, maps["depends"], applicable - defeated)
        selected -= defeated
        selected, class_notes = resolve_architecture_conflict_classes(selected, candidate_rules, scenario)
        notes.extend(class_notes)

        return ctv.RecoveryResult(
            predicted_rule_ids=sorted(selected),
            resolver_time_ms=(time.perf_counter() - start) * 1000.0,
            status="success",
            notes=notes,
        )
    except Exception as exc:  # noqa: BLE001 - per-task diagnostics are more useful here.
        return ctv.RecoveryResult(
            predicted_rule_ids=[],
            resolver_time_ms=(time.perf_counter() - start) * 1000.0,
            status="error",
            notes=[str(exc)],
        )


def visible_task(data: dict[str, Any]) -> dict[str, Any]:
    return data.get("task") or data.get("algorithm_input") or {}


def by_id(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item.get("omega_id")): item for item in items}


def rule_structure(data: dict[str, Any]) -> dict[str, Any]:
    return data.get("rule_structure") or data


def task_rule_structure(task_file: dict[str, Any]) -> dict[str, Any]:
    return (
        task_file.get("rule_structure_label")
        or task_file.get("hidden_reference", {}).get("rule_structure_label")
        or task_file.get("evaluation_reference", {}).get("rule_structure")
        or {}
    )


def load_architecture_tasks(dataset_root: Path = ARCHITECTURE_ROOT) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for path in sorted((dataset_root / "tasks").glob("*.json")):
        task = visible_task(read_json(path))
        if task.get("omega_id"):
            tasks.append(task)
    return tasks


def load_labels(dataset_root: Path = ARCHITECTURE_ROOT) -> dict[str, dict[str, Any]]:
    legacy_path = dataset_root / "architecture_rule_structure_labels.json"
    if legacy_path.exists():
        return by_id(read_json(legacy_path)["rule_structure_labels"])
    reference_path = dataset_root / "evaluation_references" / "architecture_evaluation_references.json"
    if reference_path.exists():
        return by_id(read_json(reference_path).get("items", []))
    return {}


def load_task_files_by_id(dataset_root: Path = ARCHITECTURE_ROOT) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in sorted((dataset_root / "tasks").glob("*.json")):
        data = read_json(path)
        task = visible_task(data)
        if task.get("omega_id"):
            out[str(task["omega_id"])] = data
    return out


def reference_valid_rule_ids(task_file: dict[str, Any], label: dict[str, Any]) -> list[str]:
    task_label = task_rule_structure(task_file)
    label_structure = rule_structure(label)
    refs = task_label.get("expected_surviving_rule_ids") or label_structure.get("expected_surviving_rule_ids") or []
    return sorted(str(rule_id) for rule_id in refs)


def expected_candidate_rule_ids(task_file: dict[str, Any], label: dict[str, Any]) -> list[str]:
    meta = task_file.get("stress_metadata") or visible_task(task_file).get("stress_metadata") or {}
    candidates = meta.get("candidate_rule_ids_expected_for_diagnostics")
    if not candidates:
        candidates = task_rule_structure(task_file).get("expected_source_rule_ids")
    if not candidates:
        candidates = rule_structure(label).get("expected_source_rule_ids", [])
    return sorted(str(rule_id) for rule_id in candidates)


def target_interaction(task_file: dict[str, Any], label: dict[str, Any]) -> str:
    meta = task_file.get("stress_metadata") or visible_task(task_file).get("stress_metadata") or {}
    target = meta.get("target_interaction")
    if target:
        return str(target)
    challenge_types = rule_structure(label).get("challenge_types") or task_rule_structure(task_file).get("challenge_types") or []
    if challenge_types:
        return "; ".join(str(item) for item in challenge_types)
    return str(visible_task(task_file).get("task_type", ""))


def safe_ratio(numer: int, denom: int) -> float:
    return float("nan") if denom == 0 else numer / denom


def rounded(value: float) -> float:
    return value if isinstance(value, float) and math.isnan(value) else round(float(value), 4)


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for key in (
        "Candidate / Reference Ratio",
        "Predicted / Reference Ratio",
        "Rule-ID Precision",
        "Rule-ID Recall",
    ):
        out[key] = rounded(out[key])
    return out


def run_experiment(
    dataset_root: Path = ARCHITECTURE_ROOT,
    rule_library_path: Path | None = None,
    grounding_mode: str = "strict_profile",
    llm_rerank: bool = False,
    llm_relation_filter: bool = False,
    llm_provider: str = "qwen",
    llm_model: str | None = None,
    llm_cache: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if grounding_mode not in GROUNDING_MODES:
        raise ValueError(f"Unsupported grounding_mode: {grounding_mode}")
    if grounding_mode == "relation_rich" and llm_rerank:
        raise ValueError("relation_rich keeps relation alternatives; disable minimal-set --llm-rerank for this mode.")
    if llm_rerank and llm_relation_filter:
        raise ValueError("--llm-rerank and --llm-relation-filter are mutually exclusive.")
    rule_library_path = rule_library_path or default_architecture_rule_library(dataset_root)
    rule_library = read_json(rule_library_path)
    labels = load_labels(dataset_root)
    task_files = load_task_files_by_id(dataset_root)
    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for task in load_architecture_tasks(dataset_root):
        task_id = str(task["omega_id"])
        task_file = task_files[task_id]
        label = labels.get(task_id, {})

        candidate_rules, candidate_reasons = generate_architecture_candidates(rule_library, task, grounding_mode)
        llm_diagnostics: dict[str, Any] | None = None
        if llm_rerank:
            candidate_rules, llm_diagnostics = llm_reranker.rerank_candidate_rules(
                domain="architecture",
                task=task,
                candidate_rules=candidate_rules,
                candidate_reasons=candidate_reasons,
                provider_name=llm_provider,
                model=llm_model,
                cache_path=llm_cache,
            )
        elif llm_relation_filter:
            candidate_rules, llm_diagnostics = llm_reranker.relation_filter_candidate_rules(
                domain="architecture",
                task=task,
                candidate_rules=candidate_rules,
                candidate_reasons=candidate_reasons,
                provider_name=llm_provider,
                model=llm_model,
                cache_path=llm_cache,
            )
        candidate_ids = sorted(str(rule["rule_id"]) for rule in candidate_rules)
        scenario = architecture_scenario(task)
        result = cthr_recover_architecture_valid_rules(candidate_rules, scenario)
        predicted_ids = sorted(result.predicted_rule_ids)
        reference_ids = reference_valid_rule_ids(task_file, label)
        expected_candidate_ids = expected_candidate_rule_ids(task_file, label)

        candidate_set = set(candidate_ids)
        predicted_set = set(predicted_ids)
        reference_set = set(reference_ids)
        expected_candidate_set = set(expected_candidate_ids)
        overlap = predicted_set & reference_set
        extra = sorted(predicted_set - reference_set)
        missing = sorted(reference_set - predicted_set)

        candidate_source = (
            "generated_by_architecture_grounding"
            if grounding_mode == "strict_profile"
            else f"generated_by_architecture_grounding_{grounding_mode}"
        )
        if llm_rerank:
            candidate_source = f"{candidate_source}_llm_reranked"
        if llm_relation_filter:
            candidate_source = f"{candidate_source}_llm_relation_filtered"

        row = {
            "Dataset": "Architecture",
            "task_id": task_id,
            "target_interaction": target_interaction(task_file, label),
            "candidate_source": candidate_source,
            "candidate_rule_count": len(candidate_ids),
            "reference_valid_rule_count": len(reference_ids),
            "predicted_valid_rule_count": len(predicted_ids),
            "Candidate / Reference Ratio": safe_ratio(len(candidate_ids), len(reference_ids)),
            "Predicted / Reference Ratio": safe_ratio(len(predicted_ids), len(reference_ids)),
            "Rule-ID Precision": safe_ratio(len(overlap), len(predicted_ids)),
            "Rule-ID Recall": safe_ratio(len(overlap), len(reference_ids)),
            "Exact Match": predicted_set == reference_set,
            "candidate_rule_ids_generated": candidate_ids,
            "reference_valid_rule_ids": reference_ids,
            "predicted_valid_rule_ids": predicted_ids,
            "extra_rule_ids": extra,
            "missing_rule_ids": missing,
            "_resolver_status": result.status,
            "_resolver_notes": result.notes,
            "_resolver_time_ms": result.resolver_time_ms,
            "_llm_rerank": llm_diagnostics,
        }
        rows.append(row)

        audit_rows.append(
            {
                "task_id": task_id,
                "candidate_source": candidate_source,
                "used_expected_candidate_field": False,
                "used_reference_valid_rules": False,
                "used_solver_constraints": False,
                "used_reference_cells": False,
                "used_semantic_validator": False,
                "generated_candidate_rule_ids": candidate_ids,
                "expected_candidate_rule_ids_for_comparison_only": expected_candidate_ids,
                "overlap_with_expected_candidate": sorted(candidate_set & expected_candidate_set),
                "missing_from_expected_candidate": sorted(expected_candidate_set - candidate_set),
                "extra_beyond_expected_candidate": sorted(candidate_set - expected_candidate_set),
            }
        )

    summary = {
        "task count": len(rows),
        "dataset_root": str(dataset_root),
        "mean Candidate / Reference Ratio": mean(row["Candidate / Reference Ratio"] for row in rows),
        "mean Predicted / Reference Ratio": mean(row["Predicted / Reference Ratio"] for row in rows),
        "mean Rule-ID Precision": mean(row["Rule-ID Precision"] for row in rows),
        "mean Rule-ID Recall": mean(row["Rule-ID Recall"] for row in rows),
        "exact match rate": mean(1.0 if row["Exact Match"] else 0.0 for row in rows),
        "total extra rules": sum(len(row["extra_rule_ids"]) for row in rows),
        "total missing rules": sum(len(row["missing_rule_ids"]) for row in rows),
        "candidate_source": (
            "generated_by_architecture_grounding"
            if grounding_mode == "strict_profile"
            else f"generated_by_architecture_grounding_{grounding_mode}"
        )
        + ("_llm_reranked" if llm_rerank else "")
        + ("_llm_relation_filtered" if llm_relation_filter else ""),
        "grounding_mode": grounding_mode,
        "rule_library": str(rule_library_path),
        "scope": "architecture candidate-to-valid rule recovery only; no optimizer, CSR, certificate, ASP, SMT, or MILP",
        "llm_rerank": {
            "enabled": llm_rerank,
            "provider": llm_provider if llm_rerank else None,
            "model": llm_model,
            "cache": str(llm_cache) if llm_cache else None,
        },
        "llm_relation_filter": {
            "enabled": llm_relation_filter,
            "provider": llm_provider if llm_relation_filter else None,
            "model": llm_model,
            "cache": str(llm_cache) if llm_cache else None,
        },
        "forbidden_input_fields": sorted(FORBIDDEN_INPUT_FIELDS),
        "audit": {
            "used_expected_candidate_field": False,
            "used_reference_valid_rules": False,
            "used_solver_constraints": False,
            "used_reference_cells": False,
            "used_semantic_validator": False,
        },
    }
    return rows, audit_rows, summary


def choose_main_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    categories = [
        "scenario-conditioned applicability",
        "dependency",
        "exclusion",
        "exception",
        "precedence",
        "parameter",
    ]
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def add_selected(row: dict[str, Any]) -> None:
        task_id = str(row["task_id"])
        if task_id not in selected_ids:
            selected.append(row)
            selected_ids.add(task_id)

    for category in categories:
        candidates = [
            row
            for row in rows
            if category in normalize(row["target_interaction"])
            and row["Candidate / Reference Ratio"] > 1.0
            and row["Exact Match"]
        ]
        if not candidates:
            candidates = [
                row
                for row in rows
                if category in normalize(row["target_interaction"])
                and row["Candidate / Reference Ratio"] > 1.0
            ]
        if candidates:
            add_selected(candidates[0])

    failure_rows = [row for row in rows if not row["Exact Match"] and str(row["task_id"]) not in selected_ids]
    if failure_rows:
        add_selected(failure_rows[0])

    for row in rows:
        if len(selected) >= 10:
            break
        if str(row["task_id"]) not in selected_ids and row["Candidate / Reference Ratio"] > 1.0:
            add_selected(row)
    return selected[:10]


def build_report(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    failures = [row for row in rows if row["extra_rule_ids"] or row["missing_rule_ids"]]
    wider_count = sum(1 for row in rows if row["Candidate / Reference Ratio"] > 1.0)
    lines = [
        "# Section 6.3 Architecture Candidate-to-Valid Rule Recovery",
        "",
        "## Scope",
        "",
        "This run evaluates only the architecture rule-recovery path: architecture rule library plus visible task grounding, generated candidate rules, CTHR valid-rule resolution, and comparison with reference valid rules. It does not run feasible-region validation, optimization, certificate generation, ASP, SMT, MILP, or solver backends.",
        "",
        "## Candidate Source",
        "",
        "- Candidate rules are generated by `generate_architecture_candidates(rule_library, task)`.",
        "- The grounding function uses visible task fields, rule guards, rule constraints, rule metadata, source-domain metadata, scenario facts, decision variables, visible engineering text, and rule relations.",
        "- The grounding function does not use expected candidate rules, reference valid rules, valid rule structures, solver constraints, reference cells, or semantic validator output.",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| task count | {summary['task count']} |",
        f"| mean Candidate / Reference Ratio | {summary['mean Candidate / Reference Ratio']:.4f} |",
        f"| mean Predicted / Reference Ratio | {summary['mean Predicted / Reference Ratio']:.4f} |",
        f"| mean Rule-ID Precision | {summary['mean Rule-ID Precision']:.4f} |",
        f"| mean Rule-ID Recall | {summary['mean Rule-ID Recall']:.4f} |",
        f"| exact match rate | {summary['exact match rate']:.4f} |",
        f"| total extra rules | {summary['total extra rules']} |",
        f"| total missing rules | {summary['total missing rules']} |",
        f"| tasks with Candidate / Reference Ratio > 1 | {wider_count}/{len(rows)} |",
        "",
        "## Interpretation",
        "",
        f"- Candidate/reference ratio is greater than 1 for {wider_count} of {len(rows)} tasks, so most candidate sets are wider than the final valid rule sets.",
        f"- The mean predicted/reference ratio is {summary['mean Predicted / Reference Ratio']:.4f}; values above 1 indicate that the resolver retained extra rules on some tasks.",
        f"- Rule-ID precision is {summary['mean Rule-ID Precision']:.4f} and recall is {summary['mean Rule-ID Recall']:.4f}.",
        f"- Exact match rate is {summary['exact match rate']:.4f}.",
        "",
        "## Missing Or Extra Rules",
        "",
    ]
    if not failures:
        lines.append("No missing or extra rules were observed.")
    else:
        lines.extend(
            [
                "| task_id | target_interaction | extra_rule_ids | missing_rule_ids | resolver_notes |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for row in failures:
            lines.append(
                f"| {row['task_id']} | {row['target_interaction']} | {csv_cell(row['extra_rule_ids'])} | {csv_cell(row['missing_rule_ids'])} | {csv_cell(row['_resolver_notes'])} |"
            )
    lines.extend(
        [
            "",
            "## Grounding Compared With Expected Candidate Labels",
            "",
            "The audit file reports overlap with expected candidate labels for comparison only. Differences indicate where visible grounding is broader or narrower than the curated diagnostic candidate set.",
            "",
        ]
    )
    return "\n".join(lines)


def build_audit_report(audit_rows: list[dict[str, Any]]) -> str:
    direct_read_failures = [
        row
        for row in audit_rows
        if row["used_expected_candidate_field"]
        or row["used_reference_valid_rules"]
        or row["used_solver_constraints"]
        or row["used_reference_cells"]
        or row["used_semantic_validator"]
        or row["candidate_source"] == "read_from_expected_candidate_field"
    ]
    headers = [
        "task_id",
        "candidate_source",
        "used_expected_candidate_field",
        "used_reference_valid_rules",
        "used_solver_constraints",
        "used_reference_cells",
        "used_semantic_validator",
        "overlap_with_expected_candidate",
        "missing_from_expected_candidate",
        "extra_beyond_expected_candidate",
    ]
    lines = [
        "# Candidate Source Audit",
        "",
        "This audit checks whether candidate rules were generated from visible task grounding or read from hidden expected fields.",
        "",
        f"Direct expected-field use detected: {'yes' if direct_read_failures else 'no'}",
        "",
        markdown_table(audit_rows, headers),
    ]
    return "\n".join(lines)


def output_prefix(tag: str = "") -> str:
    tag_part = f"_{tag}" if tag else ""
    return f"section_6_3_architecture{tag_part}_candidate_to_valid"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run architecture Section 6.3 candidate-to-valid recovery.")
    parser.add_argument("--dataset-root", type=Path, default=ARCHITECTURE_ROOT)
    parser.add_argument("--rule-library", type=Path, default=None)
    parser.add_argument("--grounding-mode", choices=sorted(GROUNDING_MODES), default="strict_profile")
    parser.add_argument("--tag", default="")
    parser.add_argument("--llm-rerank", action="store_true")
    parser.add_argument("--llm-relation-filter", action="store_true")
    parser.add_argument("--llm-provider", default="qwen")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument(
        "--llm-cache",
        type=Path,
        default=RESULTS_DIR / "llm_grounding_rerank_cache.json",
    )
    args = parser.parse_args()
    rows, audit_rows, summary = run_experiment(
        dataset_root=args.dataset_root,
        rule_library_path=args.rule_library,
        grounding_mode=args.grounding_mode,
        llm_rerank=args.llm_rerank,
        llm_relation_filter=args.llm_relation_filter,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_cache=args.llm_cache,
    )
    tag = args.tag or (args.grounding_mode if args.grounding_mode != "strict_profile" else "")
    prefix = output_prefix(tag)
    direct_read_detected = any(
        row["candidate_source"] == "read_from_expected_candidate_field"
        or row["used_expected_candidate_field"]
        or row["used_reference_valid_rules"]
        or row["used_solver_constraints"]
        or row["used_reference_cells"]
        or row["used_semantic_validator"]
        for row in audit_rows
    )
    if direct_read_detected:
        raise RuntimeError("Candidate source audit failed: a forbidden expected/reference field was used as input.")

    full_headers = [
        "Dataset",
        "task_id",
        "target_interaction",
        "candidate_source",
        "candidate_rule_count",
        "reference_valid_rule_count",
        "predicted_valid_rule_count",
        "Candidate / Reference Ratio",
        "Predicted / Reference Ratio",
        "Rule-ID Precision",
        "Rule-ID Recall",
        "Exact Match",
        "candidate_rule_ids_generated",
        "reference_valid_rule_ids",
        "predicted_valid_rule_ids",
        "extra_rule_ids",
        "missing_rule_ids",
    ]
    full_public = [public_row(row) for row in rows]
    write_csv(RESULTS_DIR / f"{prefix}_full.csv", full_public, full_headers)
    (RESULTS_DIR / f"{prefix}_full.md").write_text(
        markdown_table(full_public, full_headers),
        encoding="utf-8",
    )
    write_json(RESULTS_DIR / f"{prefix}_full.json", full_public)

    main_headers = [
        "Dataset",
        "task_id",
        "target_interaction",
        "Candidate / Reference Ratio",
        "Predicted / Reference Ratio",
        "Rule-ID Precision",
    ]
    main_rows = [public_row(row) for row in choose_main_rows(rows)]
    write_csv(RESULTS_DIR / f"{prefix}_main_table.csv", main_rows, main_headers)
    (RESULTS_DIR / f"{prefix}_main_table.md").write_text(
        markdown_table(main_rows, main_headers),
        encoding="utf-8",
    )

    write_json(RESULTS_DIR / f"{prefix}_summary.json", summary)
    (RESULTS_DIR / f"{prefix}_report.md").write_text(
        build_report(rows, summary),
        encoding="utf-8",
    )

    audit_headers = [
        "task_id",
        "candidate_source",
        "used_expected_candidate_field",
        "used_reference_valid_rules",
        "used_solver_constraints",
        "used_reference_cells",
        "used_semantic_validator",
        "generated_candidate_rule_ids",
        "expected_candidate_rule_ids_for_comparison_only",
        "overlap_with_expected_candidate",
        "missing_from_expected_candidate",
        "extra_beyond_expected_candidate",
    ]
    audit_prefix = prefix.replace("_candidate_to_valid", "_candidate_source_audit")
    write_csv(RESULTS_DIR / f"{audit_prefix}.csv", audit_rows, audit_headers)
    (RESULTS_DIR / f"{audit_prefix}.md").write_text(
        build_audit_report(audit_rows),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
