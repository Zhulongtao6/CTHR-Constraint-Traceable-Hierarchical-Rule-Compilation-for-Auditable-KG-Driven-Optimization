from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CTHR_ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = CTHR_ROOT / "paper"
OUT_JSON = PAPER_DIR / "architecture_kg_generated_20_optimization_problems.json"
OUT_MD = PAPER_DIR / "ARCHITECTURE_KG_OPTIMIZATION_20_SUMMARY.md"


RULE_META: dict[str, dict[str, str]] = {
    "ADA_502_2_car_parking_width": {"document": "2010 ADA Standards for Accessible Design", "section": "502.2", "page": "unknown"},
    "ADA_502_3_access_aisle": {"document": "2010 ADA Standards for Accessible Design", "section": "502.3", "page": "unknown"},
    "ADA_502_5_vertical_clearance": {"document": "2010 ADA Standards for Accessible Design", "section": "502.5", "page": "unknown"},
    "ADA_404_2_3_door_clear_width": {"document": "2010 ADA Standards for Accessible Design", "section": "404.2.3", "page": "unknown"},
    "ADA_404_2_5_threshold": {"document": "2010 ADA Standards for Accessible Design", "section": "404.2.5", "page": "unknown"},
    "ADA_405_2_ramp_slope": {"document": "2010 ADA Standards for Accessible Design", "section": "405.2", "page": "unknown"},
    "ADA_405_5_ramp_width": {"document": "2010 ADA Standards for Accessible Design", "section": "405.5", "page": "unknown"},
    "ADA_405_7_ramp_landing": {"document": "2010 ADA Standards for Accessible Design", "section": "405.7", "page": "unknown"},
    "ADA_406_2_curb_counter_slope": {"document": "2010 ADA Standards for Accessible Design", "section": "406.2", "page": "unknown"},
    "ADA_406_4_curb_landing": {"document": "2010 ADA Standards for Accessible Design", "section": "406.4", "page": "unknown"},
    "ADA_810_2_2_bus_boarding_area": {"document": "2010 ADA Standards for Accessible Design", "section": "810.2.2", "page": "unknown"},
    "ADA_810_2_4_bus_stop_slope": {"document": "2010 ADA Standards for Accessible Design", "section": "810.2.4", "page": "unknown"},
    "ADA_604_8_1_toilet_compartment": {"document": "2010 ADA Standards for Accessible Design", "section": "604.8.1.1", "page": "unknown"},
    "ADA_604_8_1_2_door_swing_exception": {"document": "2010 ADA Standards for Accessible Design", "section": "604.8.1.2", "page": "unknown"},
    "ADA_608_2_2_standard_roll_in_shower": {"document": "2010 ADA Standards for Accessible Design", "section": "608.2.2", "page": "unknown"},
    "ADA_608_2_3_alternate_roll_in_shower": {"document": "2010 ADA Standards for Accessible Design", "section": "608.2.3", "page": "unknown"},
    "ADA_308_2_forward_reach": {"document": "2010 ADA Standards for Accessible Design", "section": "308.2", "page": "unknown"},
    "ADA_308_3_side_reach": {"document": "2010 ADA Standards for Accessible Design", "section": "308.3", "page": "unknown"},
    "ADA_309_4_operable_parts": {"document": "2010 ADA Standards for Accessible Design", "section": "309.4", "page": "unknown"},
    "ADA_904_4_sales_service_counter": {"document": "2010 ADA Standards for Accessible Design", "section": "904.4", "page": "unknown"},
    "ADA_902_3_surface_height": {"document": "2010 ADA Standards for Accessible Design", "section": "902.3", "page": "unknown"},
    "ADA_403_3_walking_surface_slope": {"document": "2010 ADA Standards for Accessible Design", "section": "403.3", "page": "unknown"},
    "ADA_403_5_accessible_route_width": {"document": "2010 ADA Standards for Accessible Design", "section": "403.5", "page": "unknown"},
    "ADA_902_2_clear_floor_space": {"document": "2010 ADA Standards for Accessible Design", "section": "902.2", "page": "unknown"},
    "IBC_1005_3_1_stair_capacity": {"document": "2021 International Building Code", "section": "1005.3.1", "page": "unknown"},
    "IBC_1005_3_2_other_egress_capacity": {"document": "2021 International Building Code", "section": "1005.3.2", "page": "unknown"},
    "IBC_1006_2_1_single_exit_threshold": {"document": "2021 International Building Code", "section": "1006.2.1", "page": "unknown"},
    "IBC_1006_3_exit_number": {"document": "2021 International Building Code", "section": "1006.3", "page": "unknown"},
    "IBC_1017_2_travel_distance": {"document": "2021 International Building Code", "section": "1017.2", "page": "unknown"},
    "IBC_508_3_nonseparated_occupancy": {"document": "2021 International Building Code", "section": "508.3", "page": "unknown"},
    "IBC_508_4_separated_occupancy": {"document": "2021 International Building Code", "section": "508.4", "page": "unknown"},
    "IBC_506_2_allowable_area": {"document": "2021 International Building Code", "section": "506.2", "page": "unknown"},
    "IBC_506_3_frontage_increase": {"document": "2021 International Building Code", "section": "506.3", "page": "unknown"},
    "IBC_506_4_sprinkler_increase": {"document": "2021 International Building Code", "section": "506.4", "page": "unknown"},
    "IBC_403_1_high_rise_threshold": {"document": "2021 International Building Code", "section": "403.1", "page": "unknown"},
    "IBC_403_3_high_rise_sprinkler": {"document": "2021 International Building Code", "section": "403.3", "page": "unknown"},
    "IBC_403_4_high_rise_systems": {"document": "2021 International Building Code", "section": "403.4", "page": "unknown"},
    "IFC_5003_1_1_max_allowable_quantity": {"document": "2021 International Fire Code", "section": "5003.1.1", "page": "unknown"},
    "IFC_5003_5_hazard_identification": {"document": "2021 International Fire Code", "section": "5003.5", "page": "unknown"},
    "IFC_5003_8_4_gas_rooms": {"document": "2021 International Fire Code", "section": "5003.8.4", "page": "unknown"},
    "IFC_5003_8_5_exhausted_enclosures": {"document": "2021 International Fire Code", "section": "5003.8.5", "page": "unknown"},
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def var(unit: str, lower: float, upper: float, typ: str = "continuous") -> dict[str, Any]:
    return {"type": typ, "unit": unit, "lower": lower, "upper": upper}


def binary() -> dict[str, Any]:
    return {"type": "binary", "unit": "0/1", "lower": 0, "upper": 1}


def cons(cid: str, expression: str, source: str, role: str) -> dict[str, str]:
    key = "source_rule_id" if source.startswith("rule:") else "source"
    value = source[5:] if source.startswith("rule:") else source
    return {"constraint_id": cid, "expression": expression, key: value, "role": role}


def cell(cell_id: str, description: str, constraints: list[dict[str, Any]]) -> dict[str, Any]:
    return {"cell_id": cell_id, "description": description, "constraints": constraints}


def evidence_for(rule_ids: list[str]) -> dict[str, Any]:
    chunks = [f"arch_chunk_{rid}" for rid in rule_ids]
    nodes = [f"arch_node_{rid}" for rid in rule_ids]
    edges = [f"arch_edge_{idx:03d}_{rid}" for idx, rid in enumerate(rule_ids)]
    provenance = []
    missing = []
    for rid, chunk_id in zip(rule_ids, chunks):
        meta = RULE_META.get(rid)
        if not meta:
            missing.append(rid)
            meta = {"document": "unknown", "section": "unknown", "page": "unknown"}
        provenance.append({**meta, "chunk_id": chunk_id})
    return {
        "kg_chunk_ids": chunks,
        "kg_node_ids": nodes,
        "kg_edge_ids": edges,
        "provenance": provenance,
        "missing_rule_ids": missing,
    }


def case(
    omega_id: str,
    title: str,
    task_type: str,
    design_intent: str,
    scenario_facts: dict[str, Any],
    decision_variables: dict[str, dict[str, Any]],
    objectives: list[dict[str, str]],
    query_preferences: dict[str, Any],
    source_rule_ids: list[str],
    kg_constraints: list[dict[str, Any]],
    expected_rule_behavior: dict[str, list[str]],
    notes: str,
    valid_constraint_cells: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    hidden_reference: dict[str, Any] = {
        "source_rule_ids": source_rule_ids,
        "kg_grounded_constraints": kg_constraints,
        "expected_rule_behavior": expected_rule_behavior,
        "evidence": evidence_for(source_rule_ids),
    }
    if valid_constraint_cells is not None:
        hidden_reference["valid_constraint_cells"] = valid_constraint_cells
    return {
        "omega_id": omega_id,
        "domain": "architecture_code_compliance",
        "task_type": task_type,
        "title": title,
        "visible_decision_query": {
            "design_intent": design_intent,
            "scenario_facts": scenario_facts,
            "decision_variables": decision_variables,
            "objectives": objectives,
            "query_preferences": query_preferences,
        },
        "hidden_evaluation_reference": hidden_reference,
        "solver_interface": {
            "problem_class": "continuous_constrained_multi_objective",
            "constraint_form": "linear, piecewise-linear, or linearized numeric code constraints",
            "recommended_solver": "QP/MILP/SMT/exact symbolic solver; CPO is optional",
        },
        "notes": notes,
    }


def build_cases() -> list[dict[str, Any]]:
    return [
        case(
            "ARCH_OPT_01",
            "Accessible car parking stall layout",
            "accessible_parking_design",
            "Design an accessible car parking space and access aisle near the entrance. The design must satisfy ADA dimensional and slope requirements while trading off smaller paved width against larger accessibility margin.",
            {"facility": "parking", "vehicle_type": "car", "accessibility_required": True, "near_accessible_entrance": True, "site_cross_slope_ratio": 0.015},
            {"parking_space_width_in": var("inch", 84, 132), "access_aisle_width_in": var("inch", 48, 96), "surface_slope_ratio": var("ratio", 0.0, 0.04)},
            [{"name": "minimize_paved_width", "expression": "parking_space_width_in + access_aisle_width_in"}, {"name": "maximize_accessibility_margin", "expression": "parking_space_width_in - 96 + access_aisle_width_in - 60"}],
            {"lambda": [0.6, 0.4], "meaning": "prefer a compact paved layout while preserving dimensional margin"},
            ["ADA_502_2_car_parking_width", "ADA_502_3_access_aisle"],
            [cons("C1", "parking_space_width_in >= 96", "rule:ADA_502_2_car_parking_width", "car_space_width_minimum"), cons("C2", "access_aisle_width_in >= 60", "rule:ADA_502_3_access_aisle", "access_aisle_width_minimum"), cons("C3", "surface_slope_ratio <= 0.020833", "rule:ADA_502_3_access_aisle", "parking_surface_slope_limit")],
            {"should_activate": ["ADA accessible car parking dimensional rules"], "should_exclude": ["van-specific alternate parking template"], "should_resolve": ["parking/access-aisle dependency"]},
            "Real trade-off: smaller paved width reduces site impact but also reduces dimensional compliance margin.",
        ),
        case(
            "ARCH_OPT_02",
            "Accessible van parking with alternate aisle choices",
            "accessible_parking_design",
            "Design a van-accessible parking layout under a limited total-width envelope. The code permits alternative templates; the optimizer must choose one complete template rather than mixing partial dimensions.",
            {"facility": "parking", "vehicle_type": "van", "accessibility_required": True, "overhead_beam_clearance_in": 102, "total_width_budget_in": 210},
            {"van_space_width_in": var("inch", 90, 144), "access_aisle_width_in": var("inch", 54, 108), "vertical_clearance_in": var("inch", 90, 120)},
            [{"name": "minimize_total_width", "expression": "van_space_width_in + access_aisle_width_in"}, {"name": "maximize_vertical_clearance_margin", "expression": "vertical_clearance_in - 98"}],
            {"lambda": [0.7, 0.3], "meaning": "prefer a narrow parking module while retaining vertical-clearance margin"},
            ["ADA_502_2_car_parking_width", "ADA_502_3_access_aisle", "ADA_502_5_vertical_clearance"],
            [cons("C1", "vertical_clearance_in >= 98", "rule:ADA_502_5_vertical_clearance", "van_vertical_clearance_minimum"), cons("C2", "van_space_width_in + access_aisle_width_in <= total_width_budget_in", "scenario_site_width_budget", "site_width_budget")],
            {"should_activate": ["van parking rule", "access aisle rule", "vertical clearance rule"], "should_exclude": ["ordinary car-only template"], "should_resolve": ["van wide-space and wide-aisle templates are alternative valid rule structures"]},
            "Flat merging can incorrectly require both van templates simultaneously.",
            [
                cell("ARCH_OPT_02_cell_wide_space", "132-in van space with 60-in access aisle", [cons("P1C1", "van_space_width_in >= 132", "rule:ADA_502_2_car_parking_width", "wide_van_space_template"), cons("P1C2", "access_aisle_width_in >= 60", "rule:ADA_502_3_access_aisle", "wide_van_space_template_aisle"), cons("P1C3", "vertical_clearance_in >= 98", "rule:ADA_502_5_vertical_clearance", "vertical_clearance")]),
                cell("ARCH_OPT_02_cell_wide_aisle", "96-in van space with 96-in access aisle", [cons("P2C1", "van_space_width_in >= 96", "rule:ADA_502_2_car_parking_width", "wide_aisle_template_space"), cons("P2C2", "access_aisle_width_in >= 96", "rule:ADA_502_3_access_aisle", "wide_aisle_template"), cons("P2C3", "vertical_clearance_in >= 98", "rule:ADA_502_5_vertical_clearance", "vertical_clearance")]),
            ],
        ),
        case(
            "ARCH_OPT_03",
            "Accessible doorway design",
            "accessible_door_design",
            "Choose a door opening and threshold detail on an accessible route. The optimizer trades off smaller wall opening against lower threshold height.",
            {"facility": "door", "accessibility_required": True, "door_on_accessible_route": True, "existing_floor_offset_in": 0.375},
            {"clear_width_in": var("inch", 28, 42), "threshold_height_in": var("inch", 0, 1), "maneuvering_clearance_in": var("inch", 36, 72)},
            [{"name": "minimize_opening_width", "expression": "clear_width_in"}, {"name": "minimize_threshold_height", "expression": "threshold_height_in"}],
            {"lambda": [0.5, 0.5], "meaning": "balance construction compactness and accessible threshold comfort"},
            ["ADA_404_2_3_door_clear_width", "ADA_404_2_5_threshold"],
            [cons("C1", "clear_width_in >= 32", "rule:ADA_404_2_3_door_clear_width", "door_clear_width_minimum"), cons("C2", "threshold_height_in <= 0.5", "rule:ADA_404_2_5_threshold", "threshold_height_maximum"), cons("C3", "maneuvering_clearance_in >= 48", "scenario_access_route_model", "maneuvering_clearance_minimum")],
            {"should_activate": ["accessible door opening rule", "threshold rule"], "should_exclude": ["non-accessible service-door exceptions"], "should_resolve": ["door route applicability"]},
            "Door width and threshold height are both regulated, but they create different construction costs.",
        ),
        case(
            "ARCH_OPT_04",
            "Ramp run under limited floor area",
            "accessible_ramp_design",
            "Design an accessible ramp for a known floor elevation change. A steeper ramp reduces footprint but reduces slope safety margin.",
            {"facility": "ramp", "existing_site": False, "required_rise_in": 24, "available_run_length_in": 360},
            {"running_slope_ratio": var("rise/run", 0.04, 0.10), "rise_in": var("inch", 0, 30), "clear_width_in": var("inch", 30, 60), "landing_length_in": var("inch", 36, 84)},
            [{"name": "minimize_ramp_length", "expression": "rise_in / running_slope_ratio"}, {"name": "maximize_slope_safety_margin", "expression": "0.083333 - running_slope_ratio"}],
            {"lambda": [0.65, 0.35], "meaning": "prefer a compact ramp while retaining slope margin"},
            ["ADA_405_2_ramp_slope", "ADA_405_5_ramp_width", "ADA_405_7_ramp_landing"],
            [cons("C1", "running_slope_ratio <= 0.083333", "rule:ADA_405_2_ramp_slope", "ramp_slope_limit"), cons("C2", "rise_in = required_rise_in", "scenario_floor_elevation_change", "rise_grounding"), cons("C3", "clear_width_in >= 36", "rule:ADA_405_5_ramp_width", "ramp_clear_width"), cons("C4", "landing_length_in >= 60", "rule:ADA_405_7_ramp_landing", "ramp_landing_length"), cons("C5", "rise_in <= running_slope_ratio * available_run_length_in", "scenario_floor_area_limit", "available_run_feasibility")],
            {"should_activate": ["ramp slope rule", "ramp width rule", "landing rule"], "should_exclude": ["curb-ramp-only provisions"], "should_resolve": ["dependency from ramp applicability to landing and width rules"]},
            "The main engineering conflict is footprint versus slope comfort/safety.",
        ),
        case(
            "ARCH_OPT_05",
            "Curb ramp and landing design",
            "curb_ramp_design",
            "Design a curb ramp at a pedestrian crossing where sidewalk depth is limited. The optimizer trades off smaller footprint against larger landing margin.",
            {"facility": "curb_ramp", "pedestrian_crossing": True, "curb_height_in": 6, "available_sidewalk_depth_in": 96},
            {"curb_ramp_slope_ratio": var("rise/run", 0.04, 0.10), "curb_ramp_width_in": var("inch", 30, 60), "landing_slope_ratio": var("ratio", 0, 0.05), "landing_length_in": var("inch", 24, 72)},
            [{"name": "minimize_footprint", "expression": "curb_height_in / curb_ramp_slope_ratio + landing_length_in"}, {"name": "maximize_landing_margin", "expression": "landing_length_in - 36"}],
            {"lambda": [0.6, 0.4], "meaning": "prefer a compact crossing while preserving landing usability"},
            ["ADA_406_2_curb_counter_slope", "ADA_406_4_curb_landing", "ADA_405_2_ramp_slope"],
            [cons("C1", "curb_ramp_slope_ratio <= 0.083333", "rule:ADA_405_2_ramp_slope", "curb_ramp_slope_limit"), cons("C2", "curb_ramp_width_in >= 36", "rule:ADA_406_4_curb_landing", "curb_ramp_width_minimum"), cons("C3", "landing_length_in >= 36", "rule:ADA_406_4_curb_landing", "landing_length_minimum"), cons("C4", "landing_slope_ratio <= 0.020833", "rule:ADA_406_2_curb_counter_slope", "landing_slope_limit")],
            {"should_activate": ["curb ramp slope and landing rules"], "should_exclude": ["ordinary ramp landing rules when curb-ramp specific rule applies"], "should_resolve": ["curb-ramp dependency on accessible-route slope limits"]},
            "This case tests dependency between curb-ramp-specific rules and general ramp slope rules.",
        ),
        case(
            "ARCH_OPT_06",
            "Bus stop boarding and alighting area",
            "transit_stop_accessibility_design",
            "Design a bus stop boarding area beside a roadway. The optimizer trades off smaller sidewalk occupation against slope compliance margin.",
            {"facility": "bus_stop", "boarding_area": True, "roadway_parallel_slope_ratio": 0.035, "sidewalk_width_in": 96},
            {"clear_length_in": var("inch", 72, 144), "clear_width_in": var("inch", 48, 96), "perpendicular_slope_ratio": var("ratio", 0, 0.05), "parallel_slope_ratio": var("ratio", 0, 0.08)},
            [{"name": "minimize_sidewalk_length", "expression": "clear_length_in"}, {"name": "maximize_slope_compliance_margin", "expression": "0.020833 - perpendicular_slope_ratio"}],
            {"lambda": [0.55, 0.45], "meaning": "prefer shorter sidewalk occupation with adequate slope margin"},
            ["ADA_810_2_2_bus_boarding_area", "ADA_810_2_4_bus_stop_slope"],
            [cons("C1", "clear_length_in >= 96", "rule:ADA_810_2_2_bus_boarding_area", "boarding_area_length_minimum"), cons("C2", "clear_width_in >= 60", "rule:ADA_810_2_2_bus_boarding_area", "boarding_area_width_minimum"), cons("C3", "perpendicular_slope_ratio <= 0.020833", "rule:ADA_810_2_4_bus_stop_slope", "perpendicular_slope_limit"), cons("C4", "parallel_slope_ratio = roadway_parallel_slope_ratio", "rule:ADA_810_2_4_bus_stop_slope", "parallel_slope_follows_roadway")],
            {"should_activate": ["bus stop boarding area dimensions", "bus stop slope rule"], "should_exclude": ["parking stall slope rules"], "should_resolve": ["parallel slope follows roadway condition"]},
            "Parallel slope is scenario-driven, while perpendicular slope remains constrained.",
        ),
        case(
            "ARCH_OPT_07",
            "Accessible toilet compartment",
            "toilet_compartment_design",
            "Design a wheelchair-accessible toilet compartment with limited restroom area. The optimizer trades off smaller compartment area against larger access margin.",
            {"facility": "toilet_compartment", "water_closet_type": "wall_hung", "door_swings_into_compartment": True, "restroom_area_limited": True},
            {"compartment_width_in": var("inch", 54, 78), "compartment_depth_in": var("inch", 54, 78), "door_swing_clearance_in": var("inch", 0, 24)},
            [{"name": "minimize_compartment_area", "expression": "compartment_width_in * compartment_depth_in"}, {"name": "maximize_access_margin", "expression": "compartment_width_in - 60 + compartment_depth_in - 56"}],
            {"lambda": [0.65, 0.35], "meaning": "prefer a compact restroom while keeping wheelchair access margin"},
            ["ADA_604_8_1_toilet_compartment", "ADA_604_8_1_2_door_swing_exception"],
            [cons("C1", "compartment_width_in >= 60", "rule:ADA_604_8_1_toilet_compartment", "toilet_compartment_width"), cons("C2", "compartment_depth_in >= 56", "rule:ADA_604_8_1_toilet_compartment", "wall_hung_depth"), cons("C3", "door_swing_clearance_in >= 0", "rule:ADA_604_8_1_2_door_swing_exception", "door_swing_exception_checked")],
            {"should_activate": ["wheelchair toilet compartment rule"], "should_exclude": ["ambulatory-only stall template"], "should_resolve": ["door-swing exception applicability"]},
            "This case checks exception handling for door swing while preserving compartment dimensions.",
        ),
        case(
            "ARCH_OPT_08",
            "Roll-in shower template selection",
            "accessible_shower_design",
            "Design an accessible roll-in shower in a space-constrained bathroom. The optimizer must choose one complete ADA roll-in shower template and trade off bathroom area against compliance margin.",
            {"facility": "shower", "shower_type": "roll_in", "accessibility_required": True, "bathroom_area_limited": True},
            {"shower_width_in": var("inch", 30, 72), "shower_length_in": var("inch", 30, 72), "clearance_width_in": var("inch", 24, 72), "clearance_length_in": var("inch", 24, 72)},
            [{"name": "minimize_bathroom_area", "expression": "shower_width_in * shower_length_in"}, {"name": "maximize_compliance_margin", "expression": "minimum_template_margin"}],
            {"lambda": [0.55, 0.45], "meaning": "prefer a compact shower layout while keeping a meaningful compliance margin"},
            ["ADA_608_2_2_standard_roll_in_shower", "ADA_608_2_3_alternate_roll_in_shower"],
            [cons("C0", "standard and alternate shower templates are mutually exclusive", "rule:ADA_608_2_2_standard_roll_in_shower", "structure_only_exclusion")],
            {"should_activate": ["roll-in shower accessibility rule"], "should_exclude": ["bathtub or transfer-shower template"], "should_resolve": ["standard and alternate templates are alternative valid rule structures, not simultaneous constraints"]},
            "This is the architecture analogue of the paper's flat-vs-CTHR template-merging example.",
            [
                cell("ARCH_OPT_08_cell_standard_roll_in", "standard roll-in shower template", [cons("P1C1", "shower_width_in >= 30", "rule:ADA_608_2_2_standard_roll_in_shower", "standard_width"), cons("P1C2", "shower_length_in >= 60", "rule:ADA_608_2_2_standard_roll_in_shower", "standard_length"), cons("P1C3", "clearance_width_in >= 30", "rule:ADA_608_2_2_standard_roll_in_shower", "standard_clearance_width"), cons("P1C4", "clearance_length_in >= 60", "rule:ADA_608_2_2_standard_roll_in_shower", "standard_clearance_length")]),
                cell("ARCH_OPT_08_cell_alternate_roll_in", "alternate roll-in shower template", [cons("P2C1", "shower_width_in >= 36", "rule:ADA_608_2_3_alternate_roll_in_shower", "alternate_width"), cons("P2C2", "shower_length_in >= 36", "rule:ADA_608_2_3_alternate_roll_in_shower", "alternate_length"), cons("P2C3", "clearance_width_in >= 36", "rule:ADA_608_2_3_alternate_roll_in_shower", "alternate_clearance_width"), cons("P2C4", "clearance_length_in >= 36", "rule:ADA_608_2_3_alternate_roll_in_shower", "alternate_clearance_length")]),
            ],
        ),
        case(
            "ARCH_OPT_09",
            "Reach range and operable parts",
            "accessible_control_design",
            "Place an operable control panel where either forward or side reach may be used. The optimizer trades off lower installation height against larger reach margin.",
            {"facility": "control_panel", "approach": "forward_or_side", "accessible_route": True, "limited_wall_zone": True},
            {"control_height_in": var("inch", 12, 54), "clear_floor_depth_in": var("inch", 24, 60), "operation_force_lbf": var("lbf", 0, 8)},
            [{"name": "minimize_panel_height", "expression": "control_height_in"}, {"name": "maximize_reach_margin", "expression": "48 - control_height_in"}],
            {"lambda": [0.45, 0.55], "meaning": "prefer easy reach over minimal wall occupation"},
            ["ADA_308_2_forward_reach", "ADA_308_3_side_reach", "ADA_309_4_operable_parts"],
            [cons("C1", "operation_force_lbf <= 5", "rule:ADA_309_4_operable_parts", "operation_force_limit")],
            {"should_activate": ["reach range rules", "operable parts rule"], "should_exclude": ["non-accessible maintenance-only controls"], "should_resolve": ["forward and side reach are alternative valid structures"]},
            "The reach approach is an alternative path; flat compilation should not conjoin both approaches.",
            [
                cell("ARCH_OPT_09_cell_forward_reach", "forward reach path", [cons("P1C1", "control_height_in >= 15", "rule:ADA_308_2_forward_reach", "forward_reach_low"), cons("P1C2", "control_height_in <= 48", "rule:ADA_308_2_forward_reach", "forward_reach_high"), cons("P1C3", "clear_floor_depth_in >= 48", "rule:ADA_308_2_forward_reach", "forward_clear_floor_depth"), cons("P1C4", "operation_force_lbf <= 5", "rule:ADA_309_4_operable_parts", "operation_force_limit")]),
                cell("ARCH_OPT_09_cell_side_reach", "side reach path", [cons("P2C1", "control_height_in >= 15", "rule:ADA_308_3_side_reach", "side_reach_low"), cons("P2C2", "control_height_in <= 48", "rule:ADA_308_3_side_reach", "side_reach_high"), cons("P2C3", "clear_floor_depth_in >= 30", "rule:ADA_308_3_side_reach", "side_clear_floor_depth"), cons("P2C4", "operation_force_lbf <= 5", "rule:ADA_309_4_operable_parts", "operation_force_limit")]),
            ],
        ),
        case(
            "ARCH_OPT_10",
            "Accessible sales and service counter",
            "service_counter_design",
            "Design an accessible segment of a sales/service counter while minimizing the interruption to the standard counter line.",
            {"facility": "service_counter", "checkout": True, "front_approach": True, "counter_line_length_in": 240},
            {"counter_height_in": var("inch", 28, 44), "accessible_segment_length_in": var("inch", 24, 72), "knee_clearance_in": var("inch", 24, 36)},
            [{"name": "minimize_accessible_segment_length", "expression": "accessible_segment_length_in"}, {"name": "maximize_customer_accessibility", "expression": "36 - counter_height_in + knee_clearance_in - 27"}],
            {"lambda": [0.5, 0.5], "meaning": "balance counter continuity with accessible customer service"},
            ["ADA_904_4_sales_service_counter", "ADA_902_3_surface_height"],
            [cons("C1", "counter_height_in <= 36", "rule:ADA_904_4_sales_service_counter", "service_counter_height"), cons("C2", "accessible_segment_length_in >= 36", "rule:ADA_904_4_sales_service_counter", "accessible_counter_length"), cons("C3", "knee_clearance_in >= 27", "rule:ADA_902_3_surface_height", "knee_clearance_minimum")],
            {"should_activate": ["sales/service counter rule", "surface height rule"], "should_exclude": ["dining-only surface if checkout counter governs"], "should_resolve": ["counter applicability to checkout/service scenario"]},
            "The design trades off counter-line continuity and accessibility.",
        ),
        case(
            "ARCH_OPT_11",
            "Accessible route walking surface",
            "accessible_route_design",
            "Design an accessible route between public functions while minimizing corridor width and satisfying walking-surface slope rules.",
            {"facility": "accessible_route", "walking_surface": True, "route_length_ft": 260, "passing_space_possible": True},
            {"clear_width_in": var("inch", 30, 60), "running_slope_ratio": var("ratio", 0, 0.08), "cross_slope_ratio": var("ratio", 0, 0.05), "passing_space_interval_ft": var("ft", 100, 300)},
            [{"name": "minimize_corridor_width", "expression": "clear_width_in"}, {"name": "maximize_route_compliance_margin", "expression": "clear_width_in - 36 + 0.020833 - cross_slope_ratio"}],
            {"lambda": [0.6, 0.4], "meaning": "prefer narrow corridors while retaining route compliance margin"},
            ["ADA_403_3_walking_surface_slope", "ADA_403_5_accessible_route_width"],
            [cons("C1", "clear_width_in >= 36", "rule:ADA_403_5_accessible_route_width", "accessible_route_width"), cons("C2", "running_slope_ratio <= 0.05", "rule:ADA_403_3_walking_surface_slope", "walking_surface_running_slope"), cons("C3", "cross_slope_ratio <= 0.020833", "rule:ADA_403_3_walking_surface_slope", "walking_surface_cross_slope"), cons("C4", "passing_space_interval_ft <= 200", "rule:ADA_403_5_accessible_route_width", "passing_space_interval")],
            {"should_activate": ["accessible route clear width", "walking surface slope"], "should_exclude": ["ramp-specific slope if route is not a ramp"], "should_resolve": ["route length dependency for passing spaces"]},
            "This case checks dependency between route length and passing-space requirements.",
        ),
        case(
            "ARCH_OPT_12",
            "Accessible dining/work surface",
            "accessible_surface_design",
            "Design an accessible dining/work surface with clear floor space. The optimizer trades off compact furniture depth against knee-clearance comfort.",
            {"facility": "dining_surface", "accessibility_required": True, "movable_chairs_present": True},
            {"surface_height_in": var("inch", 26, 38), "knee_clearance_height_in": var("inch", 24, 36), "clear_floor_space_width_in": var("inch", 24, 42), "clear_floor_space_depth_in": var("inch", 24, 60)},
            [{"name": "minimize_table_depth", "expression": "clear_floor_space_depth_in"}, {"name": "maximize_knee_clearance", "expression": "knee_clearance_height_in"}],
            {"lambda": [0.45, 0.55], "meaning": "prefer knee clearance over very compact furniture depth"},
            ["ADA_902_3_surface_height", "ADA_902_2_clear_floor_space"],
            [cons("C1", "surface_height_in >= 28", "rule:ADA_902_3_surface_height", "surface_height_low"), cons("C2", "surface_height_in <= 34", "rule:ADA_902_3_surface_height", "surface_height_high"), cons("C3", "knee_clearance_height_in >= 27", "rule:ADA_902_3_surface_height", "knee_clearance"), cons("C4", "clear_floor_space_width_in >= 30", "rule:ADA_902_2_clear_floor_space", "clear_floor_width"), cons("C5", "clear_floor_space_depth_in >= 48", "rule:ADA_902_2_clear_floor_space", "clear_floor_depth")],
            {"should_activate": ["dining/work surface height", "clear floor space"], "should_exclude": ["standing-only work counter rules"], "should_resolve": ["surface usability depends on clear floor space"]},
            "Furniture compactness conflicts with accessible clear floor depth.",
        ),
        case(
            "ARCH_OPT_13",
            "Egress capacity sizing under occupant load",
            "egress_capacity_design",
            "Size stairway and non-stair egress components for a floor with known occupant load. The optimizer trades off smaller total width against larger capacity margin.",
            {"facility": "egress", "occupancy_group": "business", "occupant_load": 180, "sprinklered": True},
            {"stair_width_in": var("inch", 36, 72), "door_corridor_width_in": var("inch", 32, 84), "capacity_margin_persons": var("persons", 0, 120)},
            [{"name": "minimize_total_egress_width", "expression": "stair_width_in + door_corridor_width_in"}, {"name": "maximize_life_safety_margin", "expression": "capacity_margin_persons"}],
            {"lambda": [0.55, 0.45], "meaning": "prefer compact egress components with adequate capacity margin"},
            ["IBC_1005_3_1_stair_capacity", "IBC_1005_3_2_other_egress_capacity"],
            [cons("C1", "stair_width_in >= 0.3 * occupant_load", "rule:IBC_1005_3_1_stair_capacity", "stair_capacity_width"), cons("C2", "door_corridor_width_in >= 0.2 * occupant_load", "rule:IBC_1005_3_2_other_egress_capacity", "other_egress_capacity_width"), cons("C3", "capacity_margin_persons = stair_width_in / 0.3 - occupant_load", "scenario_capacity_model", "capacity_margin_definition")],
            {"should_activate": ["stair capacity factor", "other egress capacity factor"], "should_exclude": ["non-occupant-load egress rules"], "should_resolve": ["occupant load propagates into both stair and door/corridor width constraints"]},
            "Occupant load propagates to multiple downstream egress constraints.",
        ),
        case(
            "ARCH_OPT_14",
            "Number of exits for a tenant space",
            "exit_number_design",
            "Choose the number of exits for a tenant space while controlling added exit cost. The design must satisfy common-path and exit-number thresholds.",
            {"facility": "tenant_space", "occupancy_group": "business", "tenant_area_sqft": 7200, "occupant_load": 72, "single_exit_desired": True},
            {"number_of_exits": var("count", 1, 4, "integer"), "common_path_ft": var("ft", 30, 140), "exit_access_travel_distance_ft": var("ft", 60, 260)},
            [{"name": "minimize_number_of_exits", "expression": "number_of_exits"}, {"name": "minimize_added_exit_cost", "expression": "50000 * number_of_exits"}],
            {"lambda": [0.5, 0.5], "meaning": "prefer fewer constructed exits while remaining within exit-access limits"},
            ["IBC_1006_2_1_single_exit_threshold", "IBC_1006_3_exit_number"],
            [cons("C1", "number_of_exits >= 2", "rule:IBC_1006_3_exit_number", "minimum_exit_count_for_scenario"), cons("C2", "common_path_ft <= 100", "rule:IBC_1006_2_1_single_exit_threshold", "common_path_limit"), cons("C3", "exit_access_travel_distance_ft <= 250", "rule:IBC_1006_3_exit_number", "exit_access_travel_distance_limit")],
            {"should_activate": ["tenant-space exit count", "common path limit"], "should_exclude": ["single-exit permission if threshold exceeded"], "should_resolve": ["threshold condition defeats single-exit design preference"]},
            "The visible preference for fewer exits is constrained by hidden common-path and occupant-load thresholds.",
        ),
        case(
            "ARCH_OPT_15",
            "Exit access travel distance with sprinkler condition",
            "egress_travel_distance_design",
            "Choose a floor layout that minimizes corridor length while satisfying the travel-distance limit that applies to occupancy and sprinkler status.",
            {"facility": "egress", "occupancy_group": "business", "sprinklered": True, "open_plan_layout": True},
            {"travel_distance_ft": var("ft", 100, 330), "egress_path_length_ft": var("ft", 100, 330), "travel_distance_margin_ft": var("ft", 0, 150)},
            [{"name": "minimize_corridor_length", "expression": "egress_path_length_ft"}, {"name": "maximize_travel_distance_margin", "expression": "travel_distance_margin_ft"}],
            {"lambda": [0.65, 0.35], "meaning": "prefer shorter corridors while keeping travel-distance margin"},
            ["IBC_1017_2_travel_distance"],
            [cons("C1", "travel_distance_ft <= 300", "rule:IBC_1017_2_travel_distance", "sprinklered_business_travel_distance_limit"), cons("C2", "egress_path_length_ft = travel_distance_ft", "scenario_layout_model", "path_length_definition"), cons("C3", "travel_distance_margin_ft = 300 - travel_distance_ft", "rule:IBC_1017_2_travel_distance", "travel_distance_margin_definition")],
            {"should_activate": ["sprinklered occupancy travel distance row"], "should_exclude": ["unsprinklered lower limit row"], "should_resolve": ["sprinkler condition selects the governing table parameter"]},
            "The scenario condition changes the numerical travel-distance bound.",
        ),
        case(
            "ARCH_OPT_16",
            "Mixed-use occupancy compliance path",
            "mixed_use_occupancy_design",
            "Design a business-plus-assembly mixed-use building. The optimizer trades off usable area against fire-separation cost, but must choose a separated or nonseparated compliance path without merging them.",
            {"facility": "mixed_use_building", "occupancies": ["business", "assembly"], "sprinklered": True, "historic_overlay": False},
            {"separation_rating_hr": var("hour", 0, 3), "allowable_area_ratio": var("ratio", 0.5, 1.5), "egress_width_in": var("inch", 36, 84), "travel_distance_ft": var("ft", 100, 300)},
            [{"name": "maximize_allowable_area", "expression": "allowable_area_ratio"}, {"name": "minimize_fire_separation_cost", "expression": "separation_rating_hr"}],
            {"lambda": [0.6, 0.4], "meaning": "prefer usable area while limiting fire-separation cost"},
            ["IBC_508_3_nonseparated_occupancy", "IBC_508_4_separated_occupancy", "IBC_1005_3_2_other_egress_capacity", "IBC_1017_2_travel_distance"],
            [cons("C0", "separated and nonseparated occupancy paths are mutually exclusive", "rule:IBC_508_3_nonseparated_occupancy", "structure_only_exclusion")],
            {"should_activate": ["mixed-use occupancy rules"], "should_exclude": ["single-occupancy-only simplification"], "should_resolve": ["separated and nonseparated occupancy paths are alternative valid structures"]},
            "This case stresses mutually exclusive compliance paths and downstream egress/travel-distance consequences.",
            [
                cell("ARCH_OPT_16_cell_nonseparated", "nonseparated occupancy path", [cons("P1C1", "separation_rating_hr = 0", "rule:IBC_508_3_nonseparated_occupancy", "nonseparated_no_fire_barrier"), cons("P1C2", "allowable_area_ratio <= 1.0", "rule:IBC_508_3_nonseparated_occupancy", "most_restrictive_allowable_area"), cons("P1C3", "egress_width_in >= 44", "rule:IBC_1005_3_2_other_egress_capacity", "egress_width"), cons("P1C4", "travel_distance_ft <= 250", "rule:IBC_1017_2_travel_distance", "travel_distance")]),
                cell("ARCH_OPT_16_cell_separated", "separated occupancy path", [cons("P2C1", "separation_rating_hr >= 2", "rule:IBC_508_4_separated_occupancy", "fire_barrier_rating"), cons("P2C2", "allowable_area_ratio <= 1.25", "rule:IBC_508_4_separated_occupancy", "area_ratio_with_separation"), cons("P2C3", "egress_width_in >= 44", "rule:IBC_1005_3_2_other_egress_capacity", "egress_width"), cons("P2C4", "travel_distance_ft <= 250", "rule:IBC_1017_2_travel_distance", "travel_distance")]),
            ],
        ),
        case(
            "ARCH_OPT_17",
            "Allowable area and sprinkler/frontage increases",
            "allowable_area_design",
            "Select a building footprint strategy where frontage and sprinkler increases may expand allowable area but add cost.",
            {"facility": "building_area", "occupancy_group": "business", "lot_area_sqft": 15000, "frontage_available": True, "sprinkler_option_available": True},
            {"actual_building_area_sqft": var("sqft", 10000, 42000), "allowable_area_sqft": var("sqft", 12000, 45000), "frontage_increase_factor": var("ratio", 0, 0.75), "sprinkler_increase_factor": var("ratio", 0, 2.0)},
            [{"name": "maximize_floor_area", "expression": "actual_building_area_sqft"}, {"name": "minimize_added_protection_cost", "expression": "frontage_increase_factor + sprinkler_increase_factor"}],
            {"lambda": [0.65, 0.35], "meaning": "prefer more usable area but penalize expensive protection strategies"},
            ["IBC_506_2_allowable_area", "IBC_506_3_frontage_increase", "IBC_506_4_sprinkler_increase"],
            [cons("C1", "frontage_increase_factor <= 0.75", "rule:IBC_506_3_frontage_increase", "frontage_increase_cap"), cons("C2", "sprinkler_increase_factor <= 2.0", "rule:IBC_506_4_sprinkler_increase", "sprinkler_increase_cap"), cons("C3", "allowable_area_sqft = 18000 * (1 + frontage_increase_factor + sprinkler_increase_factor)", "rule:IBC_506_2_allowable_area", "allowable_area_equation"), cons("C4", "actual_building_area_sqft <= allowable_area_sqft", "rule:IBC_506_2_allowable_area", "actual_area_within_allowable")],
            {"should_activate": ["allowable area equation", "frontage increase", "sprinkler increase"], "should_exclude": ["area rules for unrelated occupancy groups"], "should_resolve": ["frontage and sprinkler parameters propagate into allowable area"]},
            "This case tests parameter propagation into an executable area constraint.",
        ),
        case(
            "ARCH_OPT_18",
            "High-rise safety systems package",
            "high_rise_safety_design",
            "Evaluate a building whose occupied floor height may trigger high-rise provisions. The optimizer trades off lower system cost against code-safety margin.",
            {"facility": "high_rise", "occupied_floor_height_ft": 82, "occupancy_group": "business", "urban_site": True},
            {"sprinkler_system": binary(), "fire_alarm_system": binary(), "smoke_control_required": binary(), "high_rise_margin_ft": var("ft", 0, 50)},
            [{"name": "minimize_life_safety_system_cost", "expression": "sprinkler_system + fire_alarm_system + smoke_control_required"}, {"name": "maximize_code_margin", "expression": "high_rise_margin_ft"}],
            {"lambda": [0.45, 0.55], "meaning": "prefer safety margin while accounting for system cost"},
            ["IBC_403_1_high_rise_threshold", "IBC_403_3_high_rise_sprinkler", "IBC_403_4_high_rise_systems"],
            [cons("C1", "occupied_floor_height_ft >= 75", "rule:IBC_403_1_high_rise_threshold", "high_rise_trigger"), cons("C2", "sprinkler_system = 1", "rule:IBC_403_3_high_rise_sprinkler", "sprinkler_required"), cons("C3", "fire_alarm_system = 1", "rule:IBC_403_4_high_rise_systems", "fire_alarm_required"), cons("C4", "smoke_control_required = 1", "rule:IBC_403_4_high_rise_systems", "smoke_control_required"), cons("C5", "high_rise_margin_ft = occupied_floor_height_ft - 75", "rule:IBC_403_1_high_rise_threshold", "high_rise_margin")],
            {"should_activate": ["high-rise threshold rule", "high-rise sprinkler/system dependencies"], "should_exclude": ["low-rise safety package"], "should_resolve": ["threshold activates multiple downstream system requirements"]},
            "A single threshold condition changes multiple system requirements at once.",
        ),
        case(
            "ARCH_OPT_19",
            "Hazardous material control area quantity",
            "hazardous_storage_design",
            "Determine a hazardous-material storage quantity and control-area strategy. The optimizer trades off higher allowed storage against control-area cost.",
            {"facility": "hazardous_storage", "material_class": "corrosive_liquid", "storage_condition": "closed_container", "base_max_quantity_units": 40},
            {"hazard_quantity": var("unit", 0, 160), "control_area_count": var("count", 1, 4, "integer"), "signage_required": binary()},
            [{"name": "maximize_allowed_storage", "expression": "hazard_quantity"}, {"name": "minimize_hazard_control_cost", "expression": "control_area_count + signage_required"}],
            {"lambda": [0.55, 0.45], "meaning": "prefer higher storage capacity while limiting control-area cost"},
            ["IFC_5003_1_1_max_allowable_quantity", "IFC_5003_5_hazard_identification"],
            [cons("C1", "hazard_quantity <= base_max_quantity_units * control_area_count", "rule:IFC_5003_1_1_max_allowable_quantity", "maximum_allowable_quantity_by_control_area"), cons("C2", "control_area_count >= 1", "rule:IFC_5003_1_1_max_allowable_quantity", "control_area_minimum"), cons("C3", "signage_required = 1", "rule:IFC_5003_5_hazard_identification", "hazard_identification_signage")],
            {"should_activate": ["maximum allowable quantity", "control area", "hazard identification"], "should_exclude": ["nonhazardous storage rules"], "should_resolve": ["material class and storage condition select table quantity"]},
            "This case tests table-parameter grounding for hazardous material quantity.",
        ),
        case(
            "ARCH_OPT_20",
            "Hazardous material ventilation and exhausted enclosure",
            "hazardous_room_system_design",
            "Design a hazardous material room where ventilation or exhausted enclosure requirements may apply. The optimizer trades off lower mechanical cost against larger safety margin.",
            {"facility": "hazardous_material_room", "material_phase": "gas", "room_quantity": 18, "exhausted_enclosure_possible": True},
            {"exhaust_rate": var("cfm", 0, 1200), "enclosure_type": var("category", 0, 2, "integer"), "gas_room_required": binary(), "safety_margin_units": var("unit", 0, 50)},
            [{"name": "minimize_mechanical_cost", "expression": "exhaust_rate + 5000 * enclosure_type + 10000 * gas_room_required"}, {"name": "maximize_hazard_safety_margin", "expression": "safety_margin_units"}],
            {"lambda": [0.5, 0.5], "meaning": "balance mechanical cost and hazardous-material safety margin"},
            ["IFC_5003_8_4_gas_rooms", "IFC_5003_8_5_exhausted_enclosures", "IFC_5003_1_1_max_allowable_quantity"],
            [cons("C1", "gas_room_required = 1", "rule:IFC_5003_8_4_gas_rooms", "gas_room_requirement"), cons("C2", "exhaust_rate >= 600", "rule:IFC_5003_8_5_exhausted_enclosures", "exhaust_rate_minimum"), cons("C3", "enclosure_type >= 1", "rule:IFC_5003_8_5_exhausted_enclosures", "exhausted_enclosure_requirement"), cons("C4", "safety_margin_units = 40 - room_quantity", "rule:IFC_5003_1_1_max_allowable_quantity", "quantity_safety_margin")],
            {"should_activate": ["gas room rule", "exhausted enclosure rule", "quantity margin"], "should_exclude": ["ordinary storage-room ventilation"], "should_resolve": ["hazardous-material room dependency on gas/enclosure rules"]},
            "This case checks dependency across quantity, room classification, and mechanical-system requirements.",
        ),
    ]


def build_summary(problems: list[dict[str, Any]]) -> str:
    lines = [
        "# Architecture KG Optimization 20 Summary",
        "",
        "- Domain: architecture_code_compliance",
        f"- Problems: {len(problems)}",
        "- Visible queries contain only engineering scenario facts, decision variables, objectives, and preferences.",
        "- Hidden labels contain source-rule IDs, executable constraints, valid rule-structure cells, expected rule behavior, and provenance stubs.",
        "",
        "| ID | Title | Rules | Constraints | Cells |",
        "|---|---|---:|---:|---:|",
    ]
    for item in problems:
        hidden = item["hidden_evaluation_reference"]
        lines.append(
            f"| {item['omega_id']} | {item['title']} | "
            f"{len(hidden['source_rule_ids'])} | {len(hidden['kg_grounded_constraints'])} | "
            f"{len(hidden.get('valid_constraint_cells', []))} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    problems = build_cases()
    payload = {
        "version": "architecture_kg_optimization_v1",
        "domain": "architecture_code_compliance",
        "num_problems": len(problems),
        "construction_notes": [
            "The dataset follows the aviation optimization-case schema.",
            "Regulatory thresholds and source-rule labels are hidden from visible_decision_query.",
            "Provenance stubs are section-level references and can be replaced by concrete Cognee KG chunk/node/edge IDs once the full architecture KG export is materialized.",
        ],
        "problems": problems,
    }
    write_json(OUT_JSON, payload)
    OUT_MD.write_text(build_summary(problems), encoding="utf-8")
    print(json.dumps({"output_json": str(OUT_JSON), "output_summary": str(OUT_MD), "num_problems": len(problems)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
