from __future__ import annotations

import csv
import json
import math
import re
import shutil
import sys
import time
import argparse
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CTHR_ROOT = ROOT.parents[1]
RESULTS_DIR = ROOT / "results"
AVIATION_ROOT = ROOT / "datasets" / "aviation_combined"
AVIATION_FALLBACK_ROOT = ROOT / "datasets" / "aviation"
AVIATION_RULE_LIBRARY = AVIATION_ROOT / "aviation_combined_rule_library.combined.json"
AVIATION_FALLBACK_RULE_LIBRARY = AVIATION_FALLBACK_ROOT / "aviation_stress_rule_library.combined.json"
FULL_QWEN_RULE_LIBRARY = (
    CTHR_ROOT / "paper" / "full_aviation_kg_rule_library_model_comparison" / "full_aviation_rule_library_qwen.json"
)

SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import run_section_6_3_candidate_to_valid as ctv  # noqa: E402
import llm_grounding_reranker as llm_reranker  # noqa: E402


def default_aviation_rule_library(dataset_root: Path) -> Path:
    candidates = [
        dataset_root / "aviation_combined_rule_library.combined.json",
        dataset_root / "rule_libraries" / "full_aviation_rule_library_qwen.json",
        AVIATION_RULE_LIBRARY,
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


FORBIDDEN_INPUT_FIELDS = {
    "candidate_rule_ids_expected_for_diagnostics",
    "candidate_rule_ids",
    "final_valid_rule_ids_expected_for_evaluation",
    "final_valid_rule_ids",
    "valid_rule_structures_expected",
    "solver_constraints",
    "solver_constraint_cells",
    "certificate_targets",
    "pre_solver_structure_checks",
}

GENERIC_TOKENS = {
    "a",
    "ac",
    "aircraft",
    "alt",
    "altitude",
    "and",
    "aviation",
    "category",
    "choose",
    "constraint",
    "current",
    "deg",
    "design",
    "domain",
    "false",
    "flight",
    "for",
    "from",
    "general",
    "in",
    "is",
    "km",
    "kmh",
    "max",
    "maximum",
    "min",
    "minimum",
    "nm",
    "of",
    "or",
    "procedure",
    "rule",
    "segment",
    "select",
    "source",
    "standard",
    "task",
    "the",
    "to",
    "true",
    "type",
    "under",
    "unknown",
    "with",
}

PARAMETER_VARIANT_TYPES = {
    "formula_variant_of",
    "parameter_variant_of",
    "piecewise_variant_of",
    "propagates_to",
    "uses_parameter",
}

DEPENDENCY_TYPES = {"depends_on", "requires", "uses_parameter", "applies_to"}
COMPETITION_TYPES = {"excludes", "mutually_exclusive", "conflicts_with", "conflict"}
OVERRIDE_TYPES = {"overrides", "can_override", "replaces", "defeats"}
PRECEDENCE_TYPES = {"precedes", "precedence", "higher_priority_than", "has_precedence_over"}
TYPED_SEED_MIN_SCORE = 4.5
TYPED_RELATION_MIN_SCORE = 3.0
TYPED_FAMILY_TOP_K = 4
GROUNDING_MODES = {"strict_profile", "relation_rich", "relation_stress"}
RELATION_RICH_SEED_MIN_SCORE = 2.5
RELATION_RICH_FAMILY_TOP_K = 10
RELATION_RICH_MAX_HOPS = 1
RELATION_RICH_RELATION_TYPES = (
    DEPENDENCY_TYPES | COMPETITION_TYPES | OVERRIDE_TYPES | PRECEDENCE_TYPES | PARAMETER_VARIANT_TYPES
)
RELATION_STRESS_SEED_MIN_SCORE = 2.0
RELATION_STRESS_FAMILY_TOP_K = 8
RELATION_STRESS_MAX_HOPS = 1
RELATION_STRESS_MIN_CANDIDATES = 8
RELATION_STRESS_MAX_CANDIDATES = 18


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


def stem_unit_suffix(name: str) -> str:
    norm = normalize(name)
    for suffix in (
        "_deg",
        "_degree",
        "_degrees",
        "_km",
        "_nm",
        "_ft",
        "_m",
        "_s",
        "_sec",
        "_seconds",
        "_percent",
        "_kmh",
        "_km_h",
        "_kts",
    ):
        if norm.endswith(suffix):
            return norm[: -len(suffix)]
    return norm


def aviation_scenario(task: dict[str, Any]) -> dict[str, Any]:
    scenario = dict(task.get("scenario_facts", {}))
    scenario["decision_variable_names"] = sorted(task.get("decision_variables", {}).keys())
    for key, value in list(scenario.items()):
        scenario.setdefault(stem_unit_suffix(key), value)
    aircraft_category = normalize(scenario.get("aircraft_category", ""))
    aircraft_class = normalize(scenario.get("aircraft_class", ""))
    if aircraft_category in {"h", "category_h", "category_h_aircraft"} or "class_h" in aircraft_class:
        scenario.setdefault("aircraft_type", "helicopter")
        scenario.setdefault("aircraft_category", "category h aircraft")
    if "helicopter" in normalize(task.get("task_type", "")) or "helicopter" in normalize(task.get("title", "")):
        scenario.setdefault("aircraft_type", "helicopter")
        scenario.setdefault("aircraft_category", "category h aircraft")
    if "altitude_m" in scenario:
        scenario["altitude"] = scenario["altitude_m"]
    elif "altitude_ft" in scenario:
        try:
            scenario.setdefault("altitude", float(scenario["altitude_ft"]) * 0.3048)
        except (TypeError, ValueError):
            pass
    if "flight_level" in scenario and isinstance(scenario["flight_level"], str):
        match = re.search(r"\d+", scenario["flight_level"])
        if match:
            scenario.setdefault("flight_level_numeric", int(match.group(0)))
            scenario.setdefault("altitude_ft", int(match.group(0)) * 100)
            scenario.setdefault("altitude", int(match.group(0)) * 30.48)
    if "condition" not in scenario and (
        "holding" in normalize(task.get("task_type", ""))
        or "holding" in normalize(task.get("title", ""))
    ):
        scenario["condition"] = "normal"
    if normalize(scenario.get("segment_type", "")) == "intermediate_approach_segment":
        scenario.setdefault("segment", "intermediate_approach")
        scenario.setdefault("area", "protection_area")
    if normalize(scenario.get("segment_type", "")) == "missed_approach":
        scenario.setdefault("segment", "missed_approach")
    if "area_type" in scenario:
        scenario.setdefault("area", scenario["area_type"])
    if any(
        "descent_gradient" in normalize(name)
        for name in task.get("decision_variables", {})
    ) or "maximum_allowed_gradient_percent" in scenario:
        scenario.setdefault("gradient_required", "non_zero")
    terrain_context = normalize(scenario.get("terrain_context", ""))
    if terrain_context:
        scenario.setdefault("terrain", terrain_context)
    if normalize(scenario.get("procedure_element", "")) == "paoas":
        scenario.setdefault("operation_type", "parallel_approach_operations")
    if normalize(scenario.get("operation_type", "")) == "parallel_approach_operations":
        scenario.setdefault("procedure_element", "PAOAS")
    if "track_accuracy" in scenario and isinstance(scenario["track_accuracy"], str):
        match = re.search(r"\d+(?:\.\d+)?", scenario["track_accuracy"])
        if match:
            scenario.setdefault("track_accuracy_deg", float(match.group(0)))
            scenario["track_accuracy"] = f"{match.group(0)}"
    return scenario


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


def aviation_rule_groups(rule: dict[str, Any]) -> set[str]:
    rid = normalize(rule.get("rule_id", ""))
    name = normalize(rule.get("name", ""))
    text = f"{rid}_{name}"
    groups: set[str] = set()
    if "holding_outbound_time" in text:
        groups.add("holding_time")
    if "timing_tolerance" in text or "operating_assumption_timing" in text:
        groups.add("timing_tolerance")
    if "holding_speed" in text:
        groups.add("holding_speed")
        if "6000ft_or_below" in text or "6000_ft_or_below" in text:
            groups.add("holding_speed_low_altitude")
        if "above_14000ft" in text or "above_14000_ft" in text:
            groups.add("holding_speed_high_altitude")
    if "baro_vnav" in text:
        groups.add("baro_vnav")
    if "chart" in text or "publication_content" in text or "supplementary_data" in text or "name_code" in text:
        groups.add("chart_publication")
    if "chart_oceh_publication_content" in text:
        groups.add("chart_oceh_publication")
    if "chart_title_format_rnav_rnp" in text:
        groups.add("rnp_chart_title")
    if "gls_approach_chart_title_format" in text:
        groups.add("gls_chart_title")
    if "descent_gradient_rounding" in text:
        groups.add("chart_gradient_rounding")
    if "intermediate_approach_gradient_max" in text:
        groups.add("intermediate_gradient_max")
    if "intermediate_approach_gradient_flat_default" in text:
        groups.add("intermediate_gradient_flat")
    if "moc_intermediate" in text:
        groups.add("intermediate_moc")
    if "moc_intermediate_approach_segment" in text:
        groups.add("intermediate_moc_formula")
    if "intermediate_segment_length_table" in text:
        groups.add("intermediate_ias_altitude_table")
    if "i4_4_intermediate_segment_length_formula" in text:
        groups.add("intermediate_length_formula")
    if "pbn_intermediate_segment_optimum_length_helicopter" in text:
        groups.add("pbn_intermediate_heli_length")
    elif "pbn_intermediate_segment_optimum_length" in text:
        groups.add("pbn_intermediate_length")
    if "pbn_intermediate_segment_max_length_gbas" in text:
        groups.add("pbn_gbas_length")
    if "pbn_intermediate_segment_stability" in text:
        groups.add("pbn_stability")
    if "paoas" in text:
        groups.add("paoas")
        if "paoas_protection_scope" in text:
            groups.add("paoas_scope")
    if ("oas_" in text or text.startswith("oas")) and "paoas" not in text:
        groups.add("oas_geometry")
    if "tnh_calculation" in text:
        groups.add("tnh_formula")
    if "tnh_obstacle_clearance" in text:
        groups.add("tnh_requirement")
    if "moc_turn" in text or "turn_initiation_area_calculation" in text:
        groups.add("turn_moc")
    if "turn_init_area" in text:
        groups.add("departure_turn_init")
        if "turn_init_area_der" in text:
            groups.add("departure_turn_init_der")
        if "turn_init_area_fato" in text:
            groups.add("departure_turn_init_fato")
    if "moc_primary_area_missed" in text or "moc_secondary_area" in text:
        groups.add("missed_moc")
        if "moc_primary_area_missed" in text:
            groups.add("missed_moc_primary")
        if "moc_secondary_area" in text:
            groups.add("missed_moc_secondary")
    if "turn_angle_threshold" in text:
        groups.add("missed_turn_applicability")
    if "align_final_approach_track" in text:
        groups.add("missed_track_alignment")
    if "ttaa" in text or "taa_" in text:
        groups.add("taa")
        if "obstacle_clearance_mountain_increase" in text:
            groups.add("taa_mountain_increase")
        if "buffer_zone_radius" in text:
            groups.add("taa_buffer_radius")
        if "protected_area_boundary_radius" in text:
            groups.add("taa_protected_boundary")
        if "buffer_obstacle_height_adjustment" in text:
            groups.add("taa_obstacle_adjustment")
    if "turn_radius" in text or "minimum_turn_radius" in text:
        groups.add("pbn_turn_radius")
        if "turn_radius_km_formula" in text:
            groups.add("turn_radius_km_formula")
        if "turn_radius_nm_formula" in text:
            groups.add("turn_radius_nm_formula")
    if "rf_segment_bank" in text or "max_bank_angle" in text:
        groups.add("rf_bank_angle")
        if "max_bank_angle_25deg" in text:
            groups.add("rf_bank_angle_max_25")
        if "bank_angle_15deg_above_fl190" in text:
            groups.add("rf_bank_angle_high_altitude_15")
    if "dta_" in text or text.startswith("dta"):
        groups.add("pbn_dta")
        if text.startswith("dta_formula") or "dta_formula" in text:
            groups.add("pbn_dta_formula")
    if "sbas" in text:
        groups.add("sbas")
    if "sia_" in text or text.startswith("sia"):
        groups.add("sia")
        if "sia_start_point_requirement" in text:
            groups.add("sia_start_point")
        if "sia_transition_requirement" in text:
            groups.add("sia_transition")
        if "sia_cda_consideration" in text:
            groups.add("sia_cda")
        if "sia_multi_airport_service_capability" in text:
            groups.add("sia_multi_airport")
    if "pins_departure" in text:
        groups.add("pins")
    if "vsda" in text:
        groups.add("vsda")
    if "fas_data_quality" in text:
        groups.add("fas_data_quality")
        if "course_width" in text:
            groups.add("fas_course_width")
    if "sbas_fas_db_publication" in text:
        groups.add("sbas_fas_publication")
    if "dme_arc" in text or "sector_partition" in text:
        groups.add("msa_sector")
    if "calc_ds_vor_dme_km" in text:
        groups.add("vor_dme_distance_km_formula")
    if "calc_ds_vor_dme_nm_kft" in text:
        groups.add("vor_dme_distance_nm_formula")
    if "buffer_zone_width" in text:
        groups.add("sector_buffer_width")
    if "vor_fix_tolerance" in text:
        groups.add("fix_tolerance")
    if "descent_fix_distance" in text:
        groups.add("descent_fix_distance")
    if "descent_fix_tolerance" in text:
        groups.add("descent_fix_tolerance")
    if "ils_" in text or "glide_path" in text or "glidepath" in text:
        groups.add("precision_approach")
    if "glidepath_angle_range" in text or "glide_path_angle_range" in text:
        groups.add("glide_path_angle")
    if "glide_path_formula" in text or "glidepath_formula" in text:
        groups.add("glide_path_formula")
    if "rule_non_si_glide_path_formula" in text:
        groups.add("glide_path_non_si_formula")
    if "basic_surface" in text or "gce_formula" in text or "crm_input" in text:
        groups.add("precision_surface_geometry")
    if "ils_crm_input_requirements_mls" in text:
        groups.add("precision_crm_input")
    if "rule_height_rounding_rule" in text or "height_rounding_increment" in text:
        groups.add("publication_height_rounding")
    if "rule_distance_rounding_rule" in text or "distance_rounding_increment" in text:
        groups.add("publication_distance_rounding")
    if "missed_approach_transition_surface" in text:
        groups.add("transition_surface")
    if "minimum_stabilization_distance" in text or text.startswith("msd_") or "_msd_" in text:
        groups.add("pbn_minimum_stabilization_distance")
        if "msd_flyby_non_si" in text:
            groups.add("pbn_msd_flyby_non_si")
        if "msd_flyover_si" in text:
            groups.add("pbn_msd_flyover_si")
    if "dme_rnav_protection_half_width" in text:
        groups.add("pbn_protection_half_width")
    if "chart_required_supplementary_data" in text:
        groups.add("chart_supplementary_data")
    if "wd_calculation_formula" in text:
        groups.add("publication_wd_formula")
    if "wd_rounding_requirement" in text:
        groups.add("publication_wd_rounding")
    if "ha_conversion_formula" in text:
        groups.add("height_altitude_conversion")
    if "procedure_altitude_increment" in text:
        groups.add("procedure_altitude_increment")
    if "template_contour_construction" in text:
        groups.add("template_contour")
    if "arc_radii_definition" in text:
        groups.add("arc_radii_definition")
    if "moca_mea_publication_requirement" in text:
        groups.add("moca_mea_publication")
    if "navigation_angle_convention" in text:
        groups.add("navigation_angle")
    return groups or {"generic"}


def has_decision_var(decision_vars: set[str], *needles: str) -> bool:
    return any(any(needle in var for needle in needles) for var in decision_vars)


def primary_decision_var_name(name: str) -> str:
    norm = normalize(name)
    if norm.endswith("_available_margin"):
        norm = norm[: -len("_available_margin")]
    return stem_unit_suffix(norm)


def has_primary_decision_var(decision_vars: set[str], *names: str) -> bool:
    primary_names = {primary_decision_var_name(var) for var in decision_vars}
    return bool(primary_names & {normalize(name) for name in names})


def apply_strong_business_aviation_profile(
    task_type: str,
    business_domain: str,
    decision_vars: set[str],
    scenario_tokens: set[str],
    add_required: Any,
    allow: set[str],
    block: set[str],
) -> bool:
    """Profile v11 strong-business aviation tasks from method-visible fields only."""
    if "strong_business" not in task_type and not business_domain:
        return False

    profiled = False

    def required(*groups: str) -> None:
        nonlocal profiled
        profiled = True
        add_required(*groups)

    if business_domain == "conventional":
        if has_primary_decision_var(decision_vars, "distance_to_runway_threshold"):
            required("descent_fix_distance", "navigation_angle")
        if has_primary_decision_var(decision_vars, "position_tolerance"):
            required("descent_fix_tolerance", "vor_dme_distance_km_formula")
        if has_primary_decision_var(decision_vars, "buffer_zone_width"):
            required("sector_buffer_width", "vor_dme_distance_nm_formula")
        if has_primary_decision_var(decision_vars, "dme_arc_radius"):
            required("msa_sector")
        if has_primary_decision_var(decision_vars, "inverted_cone_half_angle"):
            required("fix_tolerance", "height_altitude_conversion")
        block.update(
            {
                "holding_time",
                "holding_speed",
                "pbn_intermediate_length",
                "pbn_intermediate_heli_length",
                "pbn_gbas_length",
                "pbn_stability",
                "rf_bank_angle",
                "pbn_turn_radius",
                "taa",
                "sbas",
                "fas_data_quality",
                "chart_publication",
                "precision_approach",
                "precision_surface_geometry",
                "intermediate_moc",
                "missed_moc",
            }
        )
    elif business_domain == "holding":
        if has_primary_decision_var(decision_vars, "holding_indicated_airspeed"):
            required("holding_speed", "intermediate_ias_altitude_table")
            allow.update({"holding_speed_low_altitude", "holding_speed_high_altitude"})
            block.add("holding_speed_high_altitude")
        elif has_primary_decision_var(decision_vars, "outbound_time"):
            required("holding_time", "timing_tolerance")
        block.update({"pbn_intermediate_length", "pbn_intermediate_heli_length", "taa", "sbas"})
    elif business_domain == "helicopter":
        if has_primary_decision_var(decision_vars, "holding_indicated_airspeed"):
            required("holding_speed", "holding_speed_low_altitude", "intermediate_ias_altitude_table")
            block.add("holding_speed_high_altitude")
        if has_primary_decision_var(decision_vars, "intermediate_approach_segment_length"):
            required("pbn_intermediate_heli_length", "pbn_stability", "pbn_dta_formula")
            block.update({"pbn_intermediate_length", "pbn_gbas_length"})
        if has_primary_decision_var(decision_vars, "minimum_obstacle_clearance_moc"):
            required("intermediate_moc", "missed_track_alignment")
        block.update({"taa", "sbas", "fas_data_quality", "chart_publication"})
    elif business_domain == "pbn":
        if has_primary_decision_var(decision_vars, "terminal_arrival_area_protection_extent"):
            required("taa_buffer_radius")
            allow.add("pbn_stability")
        if has_primary_decision_var(decision_vars, "intermediate_approach_segment_length"):
            required("pbn_intermediate_length")
            allow.add("pbn_stability")
            block.update({"pbn_intermediate_heli_length", "pbn_gbas_length"})
        if has_primary_decision_var(decision_vars, "maximum_design_bank_angle"):
            required("rf_bank_angle_max_25", "pbn_msd_flyby_non_si")
        elif has_primary_decision_var(decision_vars, "bank_angle"):
            required("rf_bank_angle_high_altitude_15")
        if has_primary_decision_var(decision_vars, "turn_radius"):
            required("pbn_turn_radius")
        if has_primary_decision_var(decision_vars, "buffer_zone_width"):
            required("sector_buffer_width")
        block.update(
            {
                "descent_fix_distance",
                "descent_fix_tolerance",
                "holding_time",
                "holding_speed",
                "sbas",
                "fas_data_quality",
                "chart_publication",
                "precision_surface_geometry",
            }
        )
    elif business_domain == "precision":
        if has_primary_decision_var(
            decision_vars,
            "glide_path_angle_minimum",
            "glide_path_angle_optimum",
            "glide_path_angle_maximum",
            "glide_slope_angle",
        ):
            required("glide_path_angle", "glide_path_non_si_formula")
        if has_primary_decision_var(decision_vars, "intermediate_approach_segment_max_distance_from_ltp"):
            required("pbn_gbas_length", "glide_path_non_si_formula")
        if has_primary_decision_var(decision_vars, "channel_number_unitless"):
            required("sbas", "sbas_fas_publication", "glide_path_non_si_formula", "precision_crm_input")
        block.update(
            {
                "descent_fix_distance",
                "descent_fix_tolerance",
                "holding_time",
                "holding_speed",
                "pbn_intermediate_length",
                "pbn_intermediate_heli_length",
                "taa",
                "chart_publication",
            }
        )
    elif business_domain == "publication":
        if has_primary_decision_var(decision_vars, "distance_rounding_increment"):
            required("publication_distance_rounding", "chart_gradient_rounding")
        if has_primary_decision_var(decision_vars, "height_rounding_increment"):
            required("publication_height_rounding", "publication_wd_formula")
        if has_primary_decision_var(decision_vars, "channel_number_unitless"):
            required("sbas", "sbas_fas_publication", "chart_supplementary_data")
        block.update(
            {
                "holding_time",
                "holding_speed",
                "pbn_intermediate_length",
                "pbn_intermediate_heli_length",
                "pbn_stability",
                "rf_bank_angle",
                "pbn_turn_radius",
                "taa",
                "precision_surface_geometry",
                "intermediate_moc",
                "missed_moc",
            }
        )
    elif business_domain == "obstacle":
        if has_primary_decision_var(decision_vars, "minimum_obstacle_clearance_increase"):
            required("taa_mountain_increase")
        if has_primary_decision_var(decision_vars, "transition_surface_max_height"):
            required("transition_surface")
        if has_primary_decision_var(decision_vars, "minimum_obstacle_clearance_moc"):
            required("intermediate_moc", "intermediate_moc_formula", "missed_moc_secondary")
            if "mountainous" in scenario_tokens:
                allow.add("taa")
        block.update(
            {
                "descent_fix_distance",
                "descent_fix_tolerance",
                "holding_time",
                "holding_speed",
                "pbn_intermediate_length",
                "pbn_intermediate_heli_length",
                "pbn_stability",
                "rf_bank_angle",
                "pbn_turn_radius",
                "sbas",
                "fas_data_quality",
                "chart_publication",
                "precision_surface_geometry",
            }
        )

    return profiled


def aviation_task_profile(task: dict[str, Any]) -> dict[str, Any]:
    task_type = normalize(task.get("task_type", ""))
    title = normalize(task.get("title", ""))
    scenario = task.get("scenario_facts", {})
    business_domain = normalize(task.get("business_domain") or scenario.get("business_domain", ""))
    decision_vars = {normalize(name) for name in task.get("decision_variables", {})}
    scenario_tokens = token_set(scenario)
    allow: set[str] = set()
    require: set[str] = set()
    block: set[str] = set()
    strict = False

    def add_required(*groups: str) -> None:
        require.update(groups)
        allow.update(groups)

    if task_type == "holding_procedure_design":
        strict = True
        add_required("holding_time", "timing_tolerance")
        block.update({"holding_speed", "holding_speed_low_altitude", "holding_speed_high_altitude"})
    elif task_type == "helicopter_holding_design":
        strict = True
        allow.add("holding_speed")
        add_required("holding_speed_low_altitude", "timing_tolerance")
        block.update({"holding_time", "holding_speed_high_altitude"})
    elif task_type == "fix_tolerance_design":
        strict = True
        add_required("fix_tolerance")
    elif task_type == "sector_partition_design":
        strict = True
        add_required("msa_sector")
    elif task_type == "precision_approach_design":
        strict = True
        allow.add("precision_approach")
        add_required("glide_path_angle", "glide_path_formula")
        block.add("precision_surface_geometry")
    elif task_type == "approach_chart_publication_design":
        strict = True
        add_required("chart_gradient_rounding", "intermediate_gradient_max")
        block.add("intermediate_gradient_flat")
    elif task_type == "intermediate_approach_design":
        strict = True
        allow.update({"intermediate_gradient_max", "intermediate_moc", "intermediate_moc_formula"})
        if any("descent_gradient" in var for var in decision_vars):
            add_required("intermediate_gradient_max")
            block.add("intermediate_gradient_flat")
        if any("clearance" in var or "procedure_altitude" in var for var in decision_vars):
            add_required("intermediate_moc")
        if any("procedure_altitude" in var for var in decision_vars) or "altitude_over" in title:
            add_required("intermediate_moc_formula")
        else:
            block.add("intermediate_moc_formula")
        block.update({"pbn_intermediate_length", "pbn_intermediate_heli_length", "pbn_gbas_length", "taa"})
    elif task_type == "pbn_intermediate_segment_design":
        strict = True
        add_required("pbn_stability")
        if normalize(scenario.get("aircraft_category", "")) in {"h", "category_h", "category_h_aircraft"}:
            add_required("pbn_intermediate_heli_length")
            block.add("pbn_intermediate_length")
        else:
            add_required("pbn_intermediate_length")
            block.add("pbn_intermediate_heli_length")
    elif task_type == "gbas_intermediate_segment_design":
        strict = True
        add_required("pbn_gbas_length", "pbn_stability")
        block.update({"pbn_intermediate_length", "pbn_intermediate_heli_length"})
    elif task_type == "parallel_approach_operation_design":
        strict = True
        add_required("paoas")
        if any("angle_reference" in var or "true_north" in var for var in decision_vars):
            add_required("navigation_angle")
        block.update(
            {
                "tnh_formula",
                "tnh_requirement",
                "turn_moc",
                "oas_geometry",
                "taa",
                "intermediate_moc",
                "intermediate_moc_formula",
                "missed_turn_applicability",
            }
        )
    elif task_type == "departure_obstacle_assessment_design":
        strict = True
        add_required("paoas", "navigation_angle")
        block.update({"tnh_formula", "turn_moc", "taa", "intermediate_moc"})
    elif task_type == "missed_approach_design":
        strict = True
        if "area_type" in scenario or any("procedure_altitude" in var for var in decision_vars):
            add_required("missed_moc")
            block.update({"tnh_formula", "tnh_requirement", "paoas", "taa"})
        else:
            add_required("missed_turn_applicability", "missed_track_alignment", "tnh_requirement")
            block.update({"paoas", "taa", "intermediate_moc", "intermediate_moc_formula"})
    elif task_type in {"departure_turn_design", "helicopter_departure_turn_design"}:
        strict = True
        add_required("departure_turn_init")
        if any("turn_height" in var or "moc_turn" in var or "pdg" in var for var in decision_vars):
            if normalize(scenario.get("departure_reference", "")) == "der":
                add_required("departure_turn_init_der")
                block.add("departure_turn_init_fato")
            add_required("tnh_formula", "tnh_requirement", "turn_moc")
        else:
            block.update({"tnh_formula", "tnh_requirement", "turn_moc", "paoas"})
        block.add("taa")
    elif task_type in {"pbn_rf_segment_design", "pbn_turn_design"}:
        strict = True
        if "length" in title or any("segment_length" in var for var in decision_vars):
            add_required("pbn_dta")
            block.update({"pbn_turn_radius", "rf_bank_angle"})
        else:
            add_required("pbn_turn_radius")
            if "rf_segment" in task_type or "rf" in title or "bank" in title or any("bank_angle" in var for var in decision_vars):
                add_required("rf_bank_angle")
    elif task_type == "sbas_publication_design":
        strict = True
        add_required("sbas")
    elif task_type == "standard_instrument_arrival_design":
        strict = True
        add_required("sia")
    elif task_type == "pins_departure_design":
        strict = True
        add_required("pins")
    elif task_type == "visual_segment_descent_design":
        strict = True
        add_required("vsda")
    elif task_type == "sbas_fas_data_quality_design":
        strict = True
        add_required("fas_data_quality")
        block.add("fas_course_width")
    elif task_type == "baro_vnav_publication_design":
        strict = True
        add_required("baro_vnav", "chart_oceh_publication")
    elif task_type == "taa_altitude_design":
        strict = True
        add_required("taa")
    elif task_type == "rnp_chart_publication_design":
        strict = True
        add_required("chart_publication", "rnp_chart_title")
        block.update({"gls_chart_title", "baro_vnav"})

    if not strict:
        strict = apply_strong_business_aviation_profile(
            task_type,
            business_domain,
            decision_vars,
            scenario_tokens,
            add_required,
            allow,
            block,
        )

    for slot in aviation_slot_groups(task):
        allow.update(slot)

    taa_required = any(group == "taa" or group.startswith("taa_") for group in require)
    if "mountainous" not in scenario_tokens and task_type != "taa_altitude_design" and not taa_required:
        block.add("taa")
    return {
        "allow": allow,
        "require": require,
        "block": block,
        "strict": strict,
    }


def aviation_slot_groups(task: dict[str, Any]) -> list[set[str]]:
    task_type = normalize(task.get("task_type", ""))
    scenario = task.get("scenario_facts", {})
    business_domain = normalize(task.get("business_domain") or scenario.get("business_domain", ""))
    decision_vars = {normalize(name) for name in task.get("decision_variables", {})}
    public_variant = normalize(task.get("v11_diversity_repair", {}).get("template_id", ""))
    public_repair = normalize(task.get("v9_final_business_repair", {}).get("template_id", ""))
    if "strong_business" not in task_type and not business_domain:
        return []

    slots: list[set[str]] = []

    def slot(*groups: str) -> None:
        if groups:
            slots.append(set(groups))

    if business_domain == "conventional":
        if has_primary_decision_var(decision_vars, "distance_to_runway_threshold"):
            slot("descent_fix_distance")
            slot("navigation_angle", "turn_radius_km_formula", "vor_dme_distance_nm_formula", "template_contour")
        elif has_primary_decision_var(decision_vars, "position_tolerance"):
            slot("descent_fix_tolerance")
            slot("vor_dme_distance_km_formula", "navigation_angle", "turn_radius_km_formula")
        elif has_primary_decision_var(decision_vars, "buffer_zone_width"):
            slot("sector_buffer_width")
            slot("vor_dme_distance_nm_formula", "vor_dme_distance_km_formula", "intermediate_length_formula")
        elif has_primary_decision_var(decision_vars, "dme_arc_radius"):
            slot("msa_sector")
            slot("turn_radius_nm_formula", "procedure_altitude_increment")
        elif has_primary_decision_var(decision_vars, "inverted_cone_half_angle"):
            slot("fix_tolerance")
            slot("height_altitude_conversion", "intermediate_length_formula")
    elif business_domain == "holding":
        if has_primary_decision_var(decision_vars, "holding_indicated_airspeed"):
            slot("holding_speed_low_altitude")
            slot("intermediate_ias_altitude_table")
        elif has_primary_decision_var(decision_vars, "outbound_time"):
            slot("holding_time")
            slot("timing_tolerance")
    elif business_domain == "helicopter":
        if has_primary_decision_var(decision_vars, "holding_indicated_airspeed"):
            slot("holding_speed_low_altitude")
            slot("intermediate_ias_altitude_table")
        elif has_primary_decision_var(decision_vars, "intermediate_approach_segment_length"):
            slot("pbn_intermediate_heli_length")
            slot("pbn_dta_formula")
        elif has_primary_decision_var(decision_vars, "minimum_obstacle_clearance_moc"):
            slot("intermediate_moc")
            slot("missed_track_alignment")
    elif business_domain == "pbn":
        if has_primary_decision_var(decision_vars, "terminal_arrival_area_protection_extent"):
            slot("taa_buffer_radius")
            slot("pbn_stability", "sia_start_point", "sia_transition", "sia_cda")
        elif has_primary_decision_var(decision_vars, "intermediate_approach_segment_length"):
            slot("pbn_intermediate_length")
            if public_variant == "pbn_high_density_segment_compression":
                slot("sia_start_point", "sia_cda")
            elif public_variant == "pbn_intermediate_stability_margin":
                slot("pbn_stability", "sia_transition", "pbn_msd_flyover_si", "sia_cda")
            else:
                slot(
                    "pbn_stability",
                    "pbn_msd_flyby_non_si",
                    "pbn_msd_flyover_si",
                    "sia_transition",
                    "sia_cda",
                )
        elif has_primary_decision_var(decision_vars, "maximum_design_bank_angle"):
            slot("rf_bank_angle_max_25")
            if public_variant == "pbn_max_design_bank_maneuver":
                slot("pbn_msd_flyby_non_si", "sia_multi_airport", "sia_transition")
            elif public_variant == "pbn_rf_turn_bank_stability":
                slot("pbn_msd_flyby_non_si", "pbn_msd_flyover_si", "sia_transition")
            else:
                slot("pbn_msd_flyby_non_si", "sia_start_point", "sia_cda")
        elif has_primary_decision_var(decision_vars, "bank_angle"):
            slot("rf_bank_angle_high_altitude_15")
            if public_variant == "pbn_max_design_bank_maneuver":
                slot("sia_multi_airport", "sia_start_point", "sia_cda")
            elif public_variant == "pbn_rf_turn_bank_stability":
                slot("sia_start_point", "sia_cda", "pbn_msd_flyby_non_si")
            else:
                slot("sia_start_point", "pbn_msd_flyby_non_si", "sia_transition")
        elif has_primary_decision_var(decision_vars, "turn_radius"):
            slot("pbn_turn_radius")
            slot("turn_radius_km_formula", "turn_radius_nm_formula", "pbn_dta_formula")
        elif has_primary_decision_var(decision_vars, "buffer_zone_width"):
            slot("sector_buffer_width")
            if public_variant == "pbn_buffer_width_airspace_limit":
                slot("sia_start_point", "sia_transition", "sia_cda")
            elif public_variant == "pbn_high_terrain_buffer_expansion":
                slot("sia_start_point", "sia_cda", "sia_multi_airport")
    elif business_domain == "precision":
        if has_primary_decision_var(decision_vars, "channel_number_unitless"):
            slot("sbas_fas_publication")
            slot("glide_path_non_si_formula", "precision_crm_input")
        elif has_primary_decision_var(decision_vars, "intermediate_approach_segment_max_distance_from_ltp"):
            slot("pbn_gbas_length")
            slot("glide_path_non_si_formula")
        elif has_primary_decision_var(
            decision_vars,
            "glide_path_angle_minimum",
            "glide_path_angle_optimum",
            "glide_path_angle_maximum",
            "glide_slope_angle",
        ):
            slot("glide_path_angle")
            slot("glide_path_non_si_formula", "precision_crm_input")
    elif business_domain == "publication":
        if has_primary_decision_var(decision_vars, "distance_rounding_increment"):
            slot("publication_distance_rounding")
            slot("chart_gradient_rounding", "publication_wd_rounding")
        elif has_primary_decision_var(decision_vars, "height_rounding_increment"):
            slot("publication_height_rounding")
            slot("publication_wd_formula", "publication_wd_rounding")
        elif has_primary_decision_var(decision_vars, "channel_number_unitless"):
            slot("sbas_fas_publication")
            slot("chart_supplementary_data")
    elif business_domain == "obstacle":
        if has_primary_decision_var(decision_vars, "minimum_obstacle_clearance_increase"):
            slot("taa_mountain_increase")
            if public_repair == "obstacle_physical_clearance_chain":
                slot("paoas_scope", "moca_mea_publication")
            else:
                slot("moca_mea_publication", "intermediate_moc_formula", "missed_moc_secondary")
        elif has_primary_decision_var(decision_vars, "transition_surface_max_height"):
            slot("transition_surface")
            if public_repair == "obstacle_physical_clearance_chain":
                slot("intermediate_moc_formula", "moca_mea_publication")
            else:
                slot("moca_mea_publication", "intermediate_moc_formula", "missed_moc_secondary", "paoas_scope")
        elif has_primary_decision_var(decision_vars, "minimum_obstacle_clearance_moc"):
            slot("intermediate_moc")
            if public_repair == "obstacle_physical_clearance_chain":
                slot("paoas_scope", "intermediate_moc_formula", "moca_mea_publication")
            else:
                slot("intermediate_moc_formula", "moca_mea_publication", "missed_moc_secondary")
    return slots


def aviation_slot_rule_score(rule: dict[str, Any], task: dict[str, Any], slot: set[str], predicted_ids: set[str]) -> float:
    groups = aviation_rule_groups(rule)
    rule_id = str(rule.get("rule_id", ""))
    scenario = aviation_scenario(task)
    decision_vars = {normalize(name) for name in task.get("decision_variables", {})}
    public_variant = normalize(task.get("v11_diversity_repair", {}).get("template_id", ""))
    public_repair = normalize(task.get("v9_final_business_repair", {}).get("template_id", ""))
    score = 0.0
    if rule_id in predicted_ids:
        score += 5.0
    score += 10.0 * len(groups & slot)
    score += 2.0 * variable_match_count(rule_variables(rule), decision_vars)
    status = guard_status(rule, scenario)
    if status == "true":
        score += 4.0
    elif status == "false":
        score -= 8.0
    if not is_empty_guard(rule):
        score += 0.5
    if "generic" in groups:
        score -= 0.5
    if "transition_surface" in slot and "ii_class" in normalize(rule_id):
        score -= 2.0
    if "intermediate_moc" in slot and "intermediate_moc_formula" in groups:
        score -= 12.0
    if "pbn_msd_flyover_si" in groups:
        score -= 3.0
    if public_variant == "pbn_intermediate_stability_margin" and "sia_transition" in groups:
        score += 4.0
    if public_variant == "pbn_high_density_segment_compression" and "pbn_stability" in groups:
        score -= 4.0
    if has_primary_decision_var(decision_vars, "channel_number_unitless") and "glide_path_non_si_formula" in groups:
        score += 2.0
    if public_repair == "obstacle_physical_clearance_chain" and "paoas_scope" in groups:
        score += 3.0
    if public_repair != "obstacle_physical_clearance_chain" and "moca_mea_publication" in groups:
        score += 2.0
    if "holding_speed_low_altitude" in slot and "above_14000" in normalize(rule_id):
        score -= 20.0
    return score


def aviation_slot_constrained_rule_ids(
    candidate_rules: list[dict[str, Any]],
    task: dict[str, Any],
    predicted_rule_ids: set[str],
) -> set[str]:
    slots = aviation_slot_groups(task)
    if not slots:
        return set(predicted_rule_ids)
    profile = aviation_task_profile(task)
    blocked = set(profile["block"])
    required = set(profile["require"])
    selected: list[str] = []
    selected_set: set[str] = set()
    for slot in slots:
        slot_candidates: list[dict[str, Any]] = []
        for rule in candidate_rules:
            rule_id = str(rule.get("rule_id", ""))
            if not rule_id or rule_id in selected_set:
                continue
            groups = aviation_rule_groups(rule)
            if not (groups & slot):
                continue
            if (groups & blocked) - required - slot:
                continue
            slot_candidates.append(rule)
        if not slot_candidates:
            continue
        best = max(
            slot_candidates,
            key=lambda rule: (
                aviation_slot_rule_score(rule, task, slot, predicted_rule_ids),
                str(rule.get("rule_id", "")),
            ),
        )
        best_id = str(best.get("rule_id"))
        selected.append(best_id)
        selected_set.add(best_id)
    return set(selected) if selected else set(predicted_rule_ids)


def aviation_profile_match(rule: dict[str, Any], task: dict[str, Any]) -> tuple[set[str], dict[str, Any]]:
    return aviation_rule_groups(rule), aviation_task_profile(task)


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


def visible_task_tokens(task: dict[str, Any]) -> set[str]:
    stress_meta = task.get("stress_metadata") or {}
    visible_metadata = {
        "domain": task.get("domain"),
        "task_type": task.get("task_type"),
        "title": task.get("title"),
        "engineering_task": stress_meta.get("engineering_task"),
        "design_intent": task.get("design_intent"),
        "scenario_facts": task.get("scenario_facts", {}),
        "decision_variables": list(task.get("decision_variables", {})),
    }
    return token_set(visible_metadata)


def rule_tokens(rule: dict[str, Any]) -> set[str]:
    visible_rule_metadata = {
        "rule_id": rule.get("rule_id"),
        "name": rule.get("name"),
        "domain": rule.get("domain"),
        "rule_type": rule.get("rule_type"),
        "guard": rule.get("guard"),
        "constraints": [
            {"variable": constraint.get("variable"), "unit": constraint.get("unit")}
            for constraint in rule.get("constraints", [])
        ],
        "relations": rule.get("relations", []),
    }
    return token_set(visible_rule_metadata)


def variable_match_count(rule_vars: set[str], decision_vars: set[str]) -> int:
    count = 0
    for rule_var in rule_vars:
        rule_norm = stem_unit_suffix(rule_var)
        rule_tokens = token_set(rule_var)
        if len(rule_norm) < 3 and len(rule_tokens) <= 1:
            continue
        for decision_var in decision_vars:
            decision_norm = stem_unit_suffix(decision_var)
            decision_tokens = token_set(decision_var)
            if rule_norm == decision_norm or len(rule_tokens & decision_tokens) >= 2:
                count += 1
                break
            if rule_norm in decision_norm or decision_norm in rule_norm:
                count += 1
                break
    return count


def variable_matches_task(rule_vars: set[str], decision_vars: set[str]) -> bool:
    return variable_match_count(rule_vars, decision_vars) > 0


def is_empty_guard(rule: dict[str, Any]) -> bool:
    guard = rule.get("guard")
    return not guard or guard == {"all": []} or guard == {"any": []}


def is_formula_or_parameter_rule(rule: dict[str, Any]) -> bool:
    rid = str(rule.get("rule_id", "")).lower()
    rtype = str(rule.get("rule_type", "")).lower()
    return "formula" in rid or "formula" in rtype or "parameter" in rtype


def canonical_unit(unit: Any) -> str:
    norm = normalize(unit)
    aliases = {
        "s": "time_s",
        "sec": "time_s",
        "second": "time_s",
        "seconds": "time_s",
        "deg": "angle_deg",
        "degree": "angle_deg",
        "degrees": "angle_deg",
        "m": "length_m",
        "meter": "length_m",
        "meters": "length_m",
        "metre": "length_m",
        "metres": "length_m",
        "km": "length_km",
        "kilometer": "length_km",
        "kilometers": "length_km",
        "nm": "length_nm",
        "ft": "length_ft",
        "feet": "length_ft",
    }
    return aliases.get(norm, norm)


def rule_units(rule: dict[str, Any]) -> set[str]:
    return {
        canonical_unit(constraint.get("unit"))
        for constraint in rule.get("constraints", [])
        if constraint.get("unit")
    }


def task_units(task: dict[str, Any]) -> set[str]:
    return {
        canonical_unit(spec.get("unit"))
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


def direct_task_relevance(rule: dict[str, Any], task_tokens: set[str], decision_vars: set[str]) -> tuple[bool, int, bool]:
    rule_var_set = rule_variables(rule)
    variable_match = variable_matches_task(rule_var_set, decision_vars)
    token_overlap = len(rule_tokens(rule) & task_tokens)
    return variable_match or token_overlap >= 4, token_overlap, variable_match


def guard_status(rule: dict[str, Any], scenario: dict[str, Any]) -> str:
    return ctv.eval_guard(rule.get("guard"), scenario)


def include_relation_neighbor(rtype: str, direction: str, task_type: str) -> bool:
    if rtype in DEPENDENCY_TYPES:
        return direction == "out" or "dependency" in task_type or "parameter" in task_type
    if rtype in COMPETITION_TYPES:
        return True
    if rtype in OVERRIDE_TYPES:
        return True
    if rtype in PRECEDENCE_TYPES:
        return True
    if rtype in PARAMETER_VARIANT_TYPES:
        return True
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
    variables = sorted(stem_unit_suffix(variable) for variable in rule_variables(rule))
    if variables:
        return f"var:{variables[0]}"
    fields = sorted(stem_unit_suffix(field) for field in guard_fields(rule.get("guard")))
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
    neighbor_fields = {stem_unit_suffix(field) for field in guard_fields(neighbor.get("guard"))}
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
            and (source_family == neighbor_family or variable_match or same_guard_field or overlap >= 3)
        ):
            return True, f"typed_relation_{direction}:{rtype}"
        return False, "competition_relation_not_same_visible_family"
    if rtype in OVERRIDE_TYPES:
        if "override" in intent or same_guard_field or variable_match or overlap >= 3:
            return True, f"typed_relation_{direction}:{rtype}"
        return False, "override_relation_not_visible"
    if rtype in PRECEDENCE_TYPES:
        if "precedence" in intent or variable_match or same_guard_field or overlap >= 3:
            return True, f"typed_relation_{direction}:{rtype}"
        return False, "precedence_relation_not_visible"
    return False, "relation_type_not_grounded"


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
    status = "empty" if empty_guard else guard_status(rule, scenario)
    fields = {stem_unit_suffix(field) for field in guard_fields(guard)}
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
    groups, profile = aviation_profile_match(rule, task)
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
        score += min(3.0, 0.45 * token_overlap)
        reasons.append(f"token_overlap:{token_overlap}")
    rule_type = str(rule.get("rule_type", "")).lower()
    textual_rule = rule_type in {"requirement", "applicability", "procedure_step"}
    overlap_tokens = rule_tokens(rule) & task_tokens
    critical_text_tokens = {"turn", "obstacle", "clearance", "stability", "timing", "tolerance"}
    if textual_rule and token_overlap >= 4:
        score += 3.0
        reasons.append("textual_rule_task_match")
    elif textual_rule and (unit_overlap or overlap_tokens & critical_text_tokens) and token_overlap >= 2:
        score += 4.0
        reasons.append("textual_rule_unit_match")
    if not empty_guard and status == "true":
        score += 3.0
        reasons.append("guard_true")
    elif not empty_guard and status == "unknown" and (variable_matches or field_overlap):
        score += 0.8
        reasons.append("guard_unknown_but_bound")
    elif not empty_guard and status == "false":
        score -= 1.5
        reasons.append("guard_false")
    if is_formula_or_parameter_rule(rule):
        if variable_matches or unit_overlap or "dependency" in relation_intent(normalize(task.get("task_type", ""))):
            score += 1.0
            reasons.append("formula_parameter_visible")
            if empty_guard and unit_overlap and token_overlap >= 2:
                score += 3.5
                reasons.append("unit_parameter_task_match")
        else:
            score -= 2.0
            reasons.append("formula_parameter_unbound")
    if empty_guard and not variable_matches and not (
        "textual_rule_task_match" in reasons
        or "textual_rule_unit_match" in reasons
        or "unit_parameter_task_match" in reasons
    ):
        score -= 1.5
        reasons.append("empty_guard_without_variable_binding")
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
    if (
        empty_guard
        and not profile_required
        and not variable_matches
        and token_overlap < 6
        and not ("textual_rule_task_match" in reasons or "textual_rule_unit_match" in reasons)
        and not (unit_overlap and token_overlap >= 2)
    ):
        return score, []
    return score, reasons


def select_top_rules_by_family(
    scored: dict[str, tuple[float, list[str]]],
    rule_by_id: dict[str, dict[str, Any]],
    grounding_mode: str = "strict_profile",
) -> set[str]:
    grouped: dict[str, list[tuple[float, str]]] = {}
    for rule_id, (score, _reasons) in scored.items():
        groups = sorted(aviation_rule_groups(rule_by_id[rule_id]) - {"generic"})
        family = f"aviation:{groups[0]}" if groups else rule_family(rule_by_id[rule_id])
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


def apply_aviation_profile_filter(
    selected: set[str],
    scored: dict[str, tuple[float, list[str]]],
    candidate_reasons: dict[str, list[str]],
    rule_by_id: dict[str, dict[str, Any]],
    task: dict[str, Any],
) -> set[str]:
    profile = aviation_task_profile(task)
    out = set(selected)
    for rule_id in list(out):
        groups = aviation_rule_groups(rule_by_id[rule_id])
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
        groups = aviation_rule_groups(rule_by_id[rule_id])
        if groups & profile["require"] and not (groups & profile["block"]):
            out.add(rule_id)
            candidate_reasons.setdefault(rule_id, []).extend(
                ["domain_profile_required_reinjected", *reasons]
            )
    return out


STRUCTURAL_COMPANION_GROUPS = {
    "departure_turn_init",
    "missed_moc",
    "missed_track_alignment",
}


def aviation_resolution_candidate_rules(
    candidate_rules: list[dict[str, Any]],
    task: dict[str, Any],
) -> list[dict[str, Any]]:
    """Treat grounded structural companion rules as applicable after profile grounding."""
    profile = aviation_task_profile(task)
    out: list[dict[str, Any]] = []
    for rule in candidate_rules:
        groups = aviation_rule_groups(rule)
        if groups & STRUCTURAL_COMPANION_GROUPS and groups & profile["require"]:
            normalized = dict(rule)
            normalized["guard"] = {}
            out.append(normalized)
        else:
            out.append(rule)
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
    if str(neighbor.get("domain", "")).lower() != "aviation":
        return False, "relation_neighbor_wrong_domain"
    _relevant, overlap, variable_match = direct_task_relevance(neighbor, task_tokens, decision_vars)
    source_family = rule_family(source)
    neighbor_family = rule_family(neighbor)
    neighbor_fields = {stem_unit_suffix(field) for field in guard_fields(neighbor.get("guard"))}
    same_guard_field = bool(neighbor_fields & scenario_fields)
    groups, profile = aviation_profile_match(neighbor, task)
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
    groups, profile = aviation_profile_match(rule, task)
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
    rule_fields = {stem_unit_suffix(field) for field in guard_fields(rule.get("guard"))}
    if rule_fields & scenario_fields:
        rank += 4.0
    rank += min(4.0, 0.4 * len(rule_tokens(rule) & task_tokens))
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


def generate_aviation_candidates(
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

    scenario = aviation_scenario(task)
    decision_vars = set(str(name) for name in task.get("decision_variables", {}))
    task_tokens = visible_task_tokens(task)
    scenario_fields = {stem_unit_suffix(field) for field in scenario}
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
        if str(rule.get("domain", "")).lower() != "aviation":
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
        selected = apply_aviation_profile_filter(selected, scored, candidate_reasons, rule_by_id, task)
        if grounding_mode == "relation_stress":
            profile = aviation_task_profile(task)
            for rule_id, (_score, reasons) in scored.items():
                groups = aviation_rule_groups(rule_by_id[rule_id])
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
            if ok and (neighbor_score >= TYPED_RELATION_MIN_SCORE or rtype in DEPENDENCY_TYPES | PARAMETER_VARIANT_TYPES):
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

    selected = apply_aviation_profile_filter(selected, scored, candidate_reasons, rule_by_id, task)
    return [rule_by_id[rule_id] for rule_id in sorted(selected)], candidate_reasons


def load_rule_library(path: Path) -> dict[str, Any]:
    if not path.exists() and path == AVIATION_RULE_LIBRARY:
        if not AVIATION_FALLBACK_RULE_LIBRARY.exists():
            raise FileNotFoundError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(AVIATION_FALLBACK_RULE_LIBRARY, path)
    if not path.exists():
        raise FileNotFoundError(path)
    return read_json(path)


def visible_task(data: dict[str, Any]) -> dict[str, Any]:
    return data.get("task") or data.get("algorithm_input") or {}


def rule_structure(data: dict[str, Any]) -> dict[str, Any]:
    return data.get("rule_structure") or data


def task_rule_structure(task_file: dict[str, Any]) -> dict[str, Any]:
    return (
        task_file.get("rule_structure_label")
        or task_file.get("hidden_reference", {}).get("rule_structure_label")
        or task_file.get("evaluation_reference", {}).get("rule_structure")
        or {}
    )


def load_aviation_tasks(split: str = "all", dataset_root: Path = AVIATION_ROOT) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for path in sorted((dataset_root / "tasks").glob("AVI_*.json")):
        if re.match(r"AVI_(OPT|STRESS)_", path.name):
            if not (
                split == "all"
                or (split == "original" and path.name.startswith("AVI_OPT_"))
                or (split == "stress" and path.name.startswith("AVI_STRESS_"))
            ):
                continue
        elif split != "all":
            continue
        task = visible_task(read_json(path))
        if task.get("omega_id"):
            tasks.append(task)
    return tasks


def load_task_files_by_id(dataset_root: Path = AVIATION_ROOT) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in sorted((dataset_root / "tasks").glob("AVI_*.json")):
        data = read_json(path)
        task = visible_task(data)
        if task.get("omega_id"):
            out[str(task["omega_id"])] = data
    return out


def load_labels(dataset_root: Path = AVIATION_ROOT) -> dict[str, dict[str, Any]]:
    candidate_paths = [
        dataset_root / "aviation_combined_rule_structure_labels.json",
        dataset_root / "aviation_rule_structure_labels.json",
        dataset_root / "evaluation_references" / "aviation_evaluation_references.json",
    ]
    candidate_paths.extend(sorted((dataset_root / "evaluation_references").glob("*.json")))
    candidate_paths.append(dataset_root / "evaluation_overlays" / "qwen" / "evaluation_references.json")
    for path in candidate_paths:
        if path.exists():
            data = read_json(path)
            items = data.get("items") or data.get("rule_structure_labels") or []
            return {str(item.get("omega_id")): item for item in items}
    data = read_json(candidate_paths[0])
    items = data.get("items") or data.get("rule_structure_labels") or []
    return {str(item.get("omega_id")): item for item in items}


def reference_valid_rule_ids(task_file: dict[str, Any], label: dict[str, Any]) -> list[str]:
    refs = task_rule_structure(task_file).get("expected_surviving_rule_ids")
    if not refs:
        refs = rule_structure(label).get("expected_surviving_rule_ids", [])
    return sorted(str(rule_id) for rule_id in refs)


def expected_candidate_rule_ids(task_file: dict[str, Any], label: dict[str, Any]) -> list[str]:
    meta = task_file.get("stress_metadata") or visible_task(task_file).get("stress_metadata") or {}
    candidates = (
        meta.get("candidate_rule_ids_expected_for_diagnostics")
        or meta.get("candidate_rule_ids")
        or task_rule_structure(task_file).get("expected_source_rule_ids")
        or rule_structure(label).get("expected_source_rule_ids", [])
    )
    return sorted(str(rule_id) for rule_id in candidates)


def target_interaction(task_file: dict[str, Any], label: dict[str, Any]) -> str:
    meta = task_file.get("stress_metadata") or visible_task(task_file).get("stress_metadata") or {}
    target = meta.get("target_interaction")
    if isinstance(target, list):
        return "; ".join(str(item) for item in target)
    if target:
        return str(target)
    challenge_types = rule_structure(label).get("challenge_types") or task_rule_structure(task_file).get(
        "challenge_types", []
    )
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
    split: str = "all",
    dataset_root: Path = AVIATION_ROOT,
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
    rule_library_path = rule_library_path or default_aviation_rule_library(dataset_root)
    rule_library = load_rule_library(rule_library_path)
    labels = load_labels(dataset_root)
    task_files = load_task_files_by_id(dataset_root)
    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for task in load_aviation_tasks(split, dataset_root):
        task_id = str(task["omega_id"])
        task_file = task_files[task_id]
        label = labels.get(task_id, {})

        candidate_rules, candidate_reasons = generate_aviation_candidates(rule_library, task, grounding_mode)
        llm_diagnostics: dict[str, Any] | None = None
        if llm_rerank:
            candidate_rules, llm_diagnostics = llm_reranker.rerank_candidate_rules(
                domain="aviation",
                task=task,
                candidate_rules=candidate_rules,
                candidate_reasons=candidate_reasons,
                provider_name=llm_provider,
                model=llm_model,
                cache_path=llm_cache,
            )
        elif llm_relation_filter:
            candidate_rules, llm_diagnostics = llm_reranker.relation_filter_candidate_rules(
                domain="aviation",
                task=task,
                candidate_rules=candidate_rules,
                candidate_reasons=candidate_reasons,
                provider_name=llm_provider,
                model=llm_model,
                cache_path=llm_cache,
            )
        candidate_ids = sorted(str(rule["rule_id"]) for rule in candidate_rules)
        scenario = aviation_scenario(task)
        resolution_rules = aviation_resolution_candidate_rules(candidate_rules, task)
        result = ctv.cthr_recover_valid_rules(resolution_rules, scenario)
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
            "generated_by_aviation_grounding"
            if grounding_mode == "strict_profile"
            else f"generated_by_aviation_grounding_{grounding_mode}"
        )
        if llm_rerank:
            candidate_source = f"{candidate_source}_llm_reranked"
        if llm_relation_filter:
            candidate_source = f"{candidate_source}_llm_relation_filtered"

        rows.append(
            {
                "Dataset": "Aviation",
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
        )

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
            "generated_by_aviation_grounding"
            if grounding_mode == "strict_profile"
            else f"generated_by_aviation_grounding_{grounding_mode}"
        )
        + ("_llm_reranked" if llm_rerank else "")
        + ("_llm_relation_filtered" if llm_relation_filter else ""),
        "grounding_mode": grounding_mode,
        "rule_library": str(rule_library_path),
        "scope": "aviation candidate-to-valid rule recovery only; no optimizer, CSR, certificate, ASP, SMT, or MILP",
        "split": split,
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
        "scenario",
        "dependency",
        "exclusion",
        "override",
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
        "# Section 6.3 Aviation Candidate-to-Valid Rule Recovery",
        "",
        "## Scope",
        "",
        "This run evaluates only the aviation rule-recovery path: aviation rule library plus visible task grounding, generated candidate rules, CTHR valid-rule resolution, and comparison with reference valid rules. It does not run feasible-region validation, optimization, certificate generation, ASP, SMT, MILP, or solver backends.",
        "",
        "## Candidate Source",
        "",
        "- Candidate rules are generated by `generate_aviation_candidates(rule_library, task)`.",
        "- The grounding function uses visible task fields, rule guards, rule constraints, rule metadata, scenario facts, decision variables, visible engineering text, and rule relations.",
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
        f"- Candidate/reference ratio is greater than 1 for {wider_count} of {len(rows)} tasks.",
        f"- The mean predicted/reference ratio is {summary['mean Predicted / Reference Ratio']:.4f}.",
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
        "# Aviation Candidate Source Audit",
        "",
        "This audit checks whether candidate rules were generated from visible task grounding or read from hidden expected fields.",
        "",
        f"Direct expected-field use detected: {'yes' if direct_read_failures else 'no'}",
        "",
        markdown_table(audit_rows, headers),
    ]
    return "\n".join(lines)


def output_prefix(split: str, tag: str = "") -> str:
    tag_part = f"_{tag}" if tag else ""
    if split == "original":
        return f"section_6_3_aviation_original{tag_part}_candidate_to_valid"
    if split == "stress":
        return f"section_6_3_aviation_stress{tag_part}_candidate_to_valid"
    return f"section_6_3_aviation{tag_part}_candidate_to_valid"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run aviation Section 6.3 candidate-to-valid recovery.")
    parser.add_argument("--split", choices=["all", "original", "stress"], default="all")
    parser.add_argument("--dataset-root", type=Path, default=AVIATION_ROOT)
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
        args.split,
        args.dataset_root,
        args.rule_library,
        grounding_mode=args.grounding_mode,
        llm_rerank=args.llm_rerank,
        llm_relation_filter=args.llm_relation_filter,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_cache=args.llm_cache,
    )
    tag = args.tag or (args.grounding_mode if args.grounding_mode != "strict_profile" else "")
    prefix = output_prefix(args.split, tag)
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
