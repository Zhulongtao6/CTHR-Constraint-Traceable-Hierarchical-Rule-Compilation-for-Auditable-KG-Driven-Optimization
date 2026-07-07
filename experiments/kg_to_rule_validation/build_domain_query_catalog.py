from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "source_docs"
PAPER_DIR = ROOT / "paper"

DOCS = {
    "ADA2010": {
        "title": "2010 ADA Standards for Accessible Design",
        "pdf": r"D:\paper\LLMEhancebackgroud\建筑领域\规范\2010-design-standards.pdf",
        "text": SOURCE_DIR / "arch_ada_2010.txt",
    },
    "IBC2021": {
        "title": "2021 International Building Code",
        "pdf": r"D:\paper\LLMEhancebackgroud\建筑领域\规范\2021InternationalBuildingCode.pdf",
        "text": SOURCE_DIR / "arch_ibc_2021.txt",
    },
    "IFC2021": {
        "title": "2021 International Fire Code",
        "pdf": r"D:\paper\LLMEhancebackgroud\建筑领域\规范\IFC-2021.pdf",
        "text": SOURCE_DIR / "arch_ifc_2021.txt",
    },
    "CAAC": {
        "title": "航空器目视和仪表飞行程序设计规范",
        "pdf": r"D:\claudecodeproject\cognee-main\kg_research\aviation_research\knowdata\航空器目视和仪表飞行程序设计规范.pdf",
        "text": SOURCE_DIR / "caac_flight_procedure_design.txt",
    },
}


def norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_pages(doc_key: str) -> list[str]:
    text_path = Path(DOCS[doc_key]["text"])
    raw = text_path.read_text(encoding="utf-8", errors="ignore")
    return raw.split("\f")


PAGE_CACHE: dict[str, list[str]] = {}


def find_evidence(doc_key: str, section: str, keywords: list[str]) -> dict[str, Any]:
    if doc_key not in PAGE_CACHE:
        PAGE_CACHE[doc_key] = load_pages(doc_key)
    pages = PAGE_CACHE[doc_key]

    def score(page: str) -> tuple[int, int]:
        low = page.lower()
        hits = sum(1 for kw in keywords if kw.lower() in low)
        section_hit = 1 if section and section.lower() in low else 0
        return section_hit, hits

    best_idx = 0
    best_score = (-1, -1)
    for idx, page in enumerate(pages):
        sc = score(page)
        if sc > best_score:
            best_score = sc
            best_idx = idx

    page = pages[best_idx]
    page_low = page.lower()
    anchors = [section] + keywords
    pos = -1
    for anchor in anchors:
        if not anchor:
            continue
        pos = page_low.find(anchor.lower())
        if pos >= 0:
            break
    if pos < 0:
        pos = 0
    start = max(0, pos - 260)
    end = min(len(page), pos + 520)
    snippet = norm_text(page[start:end])
    return {
        "doc_key": doc_key,
        "document": DOCS[doc_key]["title"],
        "source_pdf": DOCS[doc_key]["pdf"],
        "section_or_clause": section,
        "pdf_text_page": best_idx + 1,
        "matched_keywords": keywords,
        "match_score": {"section_hit": best_score[0], "keyword_hits": best_score[1]},
        "snippet": snippet,
    }


def ev(doc_key: str, section: str, keywords: list[str]) -> dict[str, Any]:
    return {"doc_key": doc_key, "section": section, "keywords": keywords}


ARCHITECTURE_QUERIES: list[dict[str, Any]] = [
    {
        "query_id": "ARCH_Q01",
        "title": "Accessible car parking stall layout",
        "natural_language_query": "Design an accessible car parking space and access aisle while minimizing the total paved width.",
        "structured_scenario": {"domain": "architecture", "facility": "parking", "vehicle_type": "car", "accessibility_required": True},
        "decision_variables": ["parking_space_width_in", "access_aisle_width_in", "surface_slope_ratio"],
        "objectives": ["minimize_paved_width", "maximize_accessibility_margin"],
        "expected_rule_families": ["ADA vehicle parking space width", "ADA access aisle", "ADA surface slope"],
        "interaction_types": ["applicability", "dimensional_constraints"],
        "evidence_specs": [
            ev("ADA2010", "502.2", ["Car parking spaces", "96 inches", "van parking spaces"]),
            ev("ADA2010", "502.3", ["Access aisles", "marked", "vehicle spaces"]),
        ],
    },
    {
        "query_id": "ARCH_Q02",
        "title": "Accessible van parking with alternate aisle choices",
        "natural_language_query": "Design a van-accessible parking layout where the system may choose a wider van space or a wider access aisle, while preserving a valid ADA template.",
        "structured_scenario": {"domain": "architecture", "facility": "parking", "vehicle_type": "van", "accessibility_required": True},
        "decision_variables": ["van_space_width_in", "access_aisle_width_in", "vertical_clearance_in"],
        "objectives": ["minimize_total_width", "maximize_vertical_clearance_margin"],
        "expected_rule_families": ["ADA van parking space", "ADA access aisle alternatives", "ADA vertical clearance"],
        "interaction_types": ["alternative_valid_structures", "exclusion"],
        "evidence_specs": [
            ev("ADA2010", "502.2", ["van parking spaces", "132 inches", "96 inches"]),
            ev("ADA2010", "502.5", ["Vertical Clearance", "98 inches", "minimum"]),
        ],
    },
    {
        "query_id": "ARCH_Q03",
        "title": "Accessible doorway design",
        "natural_language_query": "Choose a door opening width and threshold detail for an accessible route while minimizing wall opening size.",
        "structured_scenario": {"domain": "architecture", "facility": "door", "accessibility_required": True},
        "decision_variables": ["clear_width_in", "threshold_height_in", "maneuvering_clearance_in"],
        "objectives": ["minimize_opening_width", "minimize_threshold_height"],
        "expected_rule_families": ["ADA clear door width", "ADA door thresholds", "ADA maneuvering clearance"],
        "interaction_types": ["dimensional_constraints", "applicability"],
        "evidence_specs": [
            ev("ADA2010", "404.2.3", ["Door openings", "clear width", "32 inches"]),
            ev("ADA2010", "404.2.5", ["Thresholds", "1/2 inch", "maximum"]),
        ],
    },
    {
        "query_id": "ARCH_Q04",
        "title": "Ramp run under limited floor area",
        "natural_language_query": "Design an accessible ramp under a space constraint while keeping the running slope compliant and the ramp width as small as allowed.",
        "structured_scenario": {"domain": "architecture", "facility": "ramp", "existing_site": False},
        "decision_variables": ["running_slope_ratio", "rise_in", "clear_width_in", "landing_length_in"],
        "objectives": ["minimize_ramp_length", "maximize_slope_safety_margin"],
        "expected_rule_families": ["ADA ramp slope", "ADA ramp rise", "ADA ramp clear width", "ADA ramp landings"],
        "interaction_types": ["dimensional_constraints", "scenario_condition"],
        "evidence_specs": [
            ev("ADA2010", "405.2", ["Ramp runs", "running slope", "1:12"]),
            ev("ADA2010", "405.5", ["clear width", "ramp run", "36 inch"]),
            ev("ADA2010", "405.7", ["Landings", "60 inches", "minimum"]),
        ],
    },
    {
        "query_id": "ARCH_Q05",
        "title": "Curb ramp and landing design",
        "natural_language_query": "Design a curb ramp at a pedestrian crossing while minimizing curb-ramp footprint and satisfying slope and landing requirements.",
        "structured_scenario": {"domain": "architecture", "facility": "curb_ramp", "pedestrian_crossing": True},
        "decision_variables": ["curb_ramp_slope_ratio", "curb_ramp_width_in", "landing_slope_ratio", "landing_length_in"],
        "objectives": ["minimize_footprint", "maximize_landing_margin"],
        "expected_rule_families": ["ADA curb ramp slope", "ADA curb ramp width", "ADA landing"],
        "interaction_types": ["dimensional_constraints", "dependency"],
        "evidence_specs": [
            ev("ADA2010", "406.1", ["Curb ramps", "accessible routes", "405.2"]),
            ev("ADA2010", "406.2", ["Counter Slope", "1:20", "adjoining gutters"]),
            ev("ADA2010", "406.4", ["Landings", "36 inches", "minimum"]),
        ],
    },
    {
        "query_id": "ARCH_Q06",
        "title": "Bus stop boarding and alighting area",
        "natural_language_query": "Design a bus stop boarding area beside a roadway while minimizing occupied sidewalk length.",
        "structured_scenario": {"domain": "architecture", "facility": "bus_stop", "boarding_area": True},
        "decision_variables": ["clear_length_in", "clear_width_in", "perpendicular_slope_ratio", "parallel_slope_matches_roadway"],
        "objectives": ["minimize_sidewalk_length", "maximize_slope_compliance_margin"],
        "expected_rule_families": ["ADA bus stop dimensions", "ADA bus stop slope"],
        "interaction_types": ["dimensional_constraints", "scenario_condition"],
        "evidence_specs": [
            ev("ADA2010", "810.2.2", ["Bus stop", "96 inches", "60 inches"]),
            ev("ADA2010", "810.2.4", ["Slope", "1:48", "parallel to the roadway"]),
        ],
    },
    {
        "query_id": "ARCH_Q07",
        "title": "Accessible toilet compartment",
        "natural_language_query": "Design a wheelchair-accessible toilet compartment while minimizing compartment area.",
        "structured_scenario": {"domain": "architecture", "facility": "toilet_compartment", "water_closet_type": "wall_hung"},
        "decision_variables": ["compartment_width_in", "compartment_depth_in", "door_swing_clearance_ok"],
        "objectives": ["minimize_compartment_area", "maximize_access_margin"],
        "expected_rule_families": ["ADA toilet compartment size", "ADA door swing exception"],
        "interaction_types": ["applicability", "exception"],
        "evidence_specs": [
            ev("ADA2010", "604.8.1.1", ["Wheelchair accessible compartments", "60 inches", "wall-hung"]),
            ev("ADA2010", "604.8.1.2", ["Doors", "shall not swing", "minimum required compartment area"]),
        ],
    },
    {
        "query_id": "ARCH_Q08",
        "title": "Roll-in shower template selection",
        "natural_language_query": "Design an accessible roll-in shower and choose either the standard or alternate template without mixing partial dimensions from both.",
        "structured_scenario": {"domain": "architecture", "facility": "shower", "shower_type": "roll_in"},
        "decision_variables": ["shower_width_in", "shower_length_in", "clearance_width_in", "clearance_length_in"],
        "objectives": ["minimize_bathroom_area", "maximize_compliance_margin"],
        "expected_rule_families": ["ADA standard roll-in shower", "ADA alternate roll-in shower"],
        "interaction_types": ["alternative_valid_structures", "exclusion"],
        "evidence_specs": [
            ev("ADA2010", "608.2.2", ["Standard Roll-In", "30 inch", "60 inch"]),
            ev("ADA2010", "608.2.3", ["Alternate Roll-In", "36 inch", "36 inch"]),
            ev("ADA2010", "608.2.2 or 608.2.3", ["shall comply with 608.2.2 or 608.2.3"]),
        ],
    },
    {
        "query_id": "ARCH_Q09",
        "title": "Reach range and operable parts",
        "natural_language_query": "Place an operable control panel on an accessible route while minimizing wall area and satisfying reach limits.",
        "structured_scenario": {"domain": "architecture", "facility": "control_panel", "approach": "forward_or_side"},
        "decision_variables": ["control_height_in", "clear_floor_depth_in", "operation_force_lbf"],
        "objectives": ["minimize_panel_height", "maximize_reach_margin"],
        "expected_rule_families": ["ADA reach ranges", "ADA operable parts"],
        "interaction_types": ["alternative_valid_structures", "applicability"],
        "evidence_specs": [
            ev("ADA2010", "308.2", ["Forward Reach", "15 inches", "48 inches"]),
            ev("ADA2010", "308.3", ["Side Reach", "15 inches", "48 inches"]),
            ev("ADA2010", "309.4", ["Operation", "5 pounds", "maximum"]),
        ],
    },
    {
        "query_id": "ARCH_Q10",
        "title": "Accessible sales and service counter",
        "natural_language_query": "Design a checkout/service counter segment while minimizing counter interruption length and satisfying accessibility requirements.",
        "structured_scenario": {"domain": "architecture", "facility": "service_counter", "checkout": True},
        "decision_variables": ["counter_height_in", "accessible_segment_length_in", "knee_clearance_in"],
        "objectives": ["minimize_accessible_segment_length", "maximize_customer_accessibility"],
        "expected_rule_families": ["ADA sales counters", "ADA checkout aisles", "ADA work surfaces"],
        "interaction_types": ["applicability", "dimensional_constraints"],
        "evidence_specs": [
            ev("ADA2010", "904.4", ["Sales and Service Counters", "36 inches", "maximum"]),
            ev("ADA2010", "902.3", ["Dining Surfaces", "28 inches", "34 inches"]),
        ],
    },
    {
        "query_id": "ARCH_Q11",
        "title": "Accessible route walking surface",
        "natural_language_query": "Design a walking route in a public facility while minimizing corridor width and keeping slope/cross-slope compliant.",
        "structured_scenario": {"domain": "architecture", "facility": "accessible_route", "walking_surface": True},
        "decision_variables": ["clear_width_in", "running_slope_ratio", "cross_slope_ratio", "passing_space_interval_ft"],
        "objectives": ["minimize_corridor_width", "maximize_route_compliance_margin"],
        "expected_rule_families": ["ADA walking surface slope", "ADA accessible route width", "ADA passing spaces"],
        "interaction_types": ["dimensional_constraints", "dependency"],
        "evidence_specs": [
            ev("ADA2010", "403.3", ["Slope", "1:20", "cross slope", "1:48"]),
            ev("ADA2010", "403.5", ["Clear Width", "36 inches", "minimum"]),
        ],
    },
    {
        "query_id": "ARCH_Q12",
        "title": "Accessible dining/work surface",
        "natural_language_query": "Design an accessible dining or work surface while minimizing table depth and satisfying knee/toe clearance.",
        "structured_scenario": {"domain": "architecture", "facility": "dining_surface", "accessibility_required": True},
        "decision_variables": ["surface_height_in", "knee_clearance_height_in", "clear_floor_space_width_in", "clear_floor_space_depth_in"],
        "objectives": ["minimize_table_depth", "maximize_knee_clearance"],
        "expected_rule_families": ["ADA dining/work surface height", "ADA clear floor space"],
        "interaction_types": ["dependency", "dimensional_constraints"],
        "evidence_specs": [
            ev("ADA2010", "902.3", ["Height", "28 inches", "34 inches"]),
            ev("ADA2010", "902.2", ["Clear Floor or Ground Space", "shall be positioned"]),
        ],
    },
    {
        "query_id": "ARCH_Q13",
        "title": "Egress capacity sizing under occupant load",
        "natural_language_query": "Size stairway and non-stair egress widths for a floor while minimizing egress width but satisfying occupant-load capacity.",
        "structured_scenario": {"domain": "architecture", "facility": "egress", "occupant_load_known": True},
        "decision_variables": ["occupant_load", "stair_width_in", "door_corridor_width_in", "sprinklered"],
        "objectives": ["minimize_total_egress_width", "maximize_life_safety_margin"],
        "expected_rule_families": ["IBC egress capacity factors", "IBC occupant-load-based capacity"],
        "interaction_types": ["scenario_condition", "dimensional_constraints"],
        "evidence_specs": [
            ev("IBC2021", "1005.3.1", ["Stairways", "capacity", "occupant load"]),
            ev("IBC2021", "1005.3.2", ["Other egress components", "capacity", "occupant load"]),
        ],
    },
    {
        "query_id": "ARCH_Q14",
        "title": "Number of exits for a tenant space",
        "natural_language_query": "Determine the minimum number of exits for a tenant space while minimizing constructed exit doors.",
        "structured_scenario": {"domain": "architecture", "facility": "tenant_space", "occupant_load_known": True},
        "decision_variables": ["occupant_load", "common_path_ft", "number_of_exits", "exit_access_travel_distance_ft"],
        "objectives": ["minimize_number_of_exits", "minimize_added_exit_cost"],
        "expected_rule_families": ["IBC exit number", "IBC common path", "IBC travel distance"],
        "interaction_types": ["threshold_condition", "scenario_condition"],
        "evidence_specs": [
            ev("IBC2021", "1006.2.1", ["Egress based on occupant load", "common path", "Table 1006.2.1"]),
            ev("IBC2021", "1006.3", ["Egress from stories", "number of exits", "access to exits"]),
        ],
    },
    {
        "query_id": "ARCH_Q15",
        "title": "Exit access travel distance with sprinkler condition",
        "natural_language_query": "Choose a floor layout that minimizes corridor length while keeping exit access travel distance within the limit that applies to the occupancy and sprinkler condition.",
        "structured_scenario": {"domain": "architecture", "facility": "egress", "sprinklered": "scenario_dependent", "occupancy_group": "variable"},
        "decision_variables": ["travel_distance_ft", "sprinklered", "occupancy_group", "egress_path_length_ft"],
        "objectives": ["minimize_corridor_length", "maximize_travel_distance_margin"],
        "expected_rule_families": ["IBC Table 1017.2", "sprinkler-dependent travel distance"],
        "interaction_types": ["scenario_condition", "parameterized_constraint"],
        "evidence_specs": [
            ev("IBC2021", "1017.2", ["Exit access travel distance", "Table 1017.2", "sprinkler"]),
        ],
    },
    {
        "query_id": "ARCH_Q16",
        "title": "Mixed-use occupancy compliance path",
        "natural_language_query": "Design a mixed-use building and choose a separated or nonseparated occupancy compliance path without merging incompatible assumptions.",
        "structured_scenario": {"domain": "architecture", "facility": "mixed_use_building", "occupancies": ["business", "assembly"]},
        "decision_variables": ["separation_rating_hr", "allowable_area_ratio", "egress_width_in", "travel_distance_ft"],
        "objectives": ["maximize_allowable_area", "minimize_fire_separation_cost"],
        "expected_rule_families": ["IBC nonseparated occupancies", "IBC separated occupancies", "IBC occupancy separation"],
        "interaction_types": ["alternative_valid_structures", "exclusion", "cross_rule_dependency"],
        "evidence_specs": [
            ev("IBC2021", "508.3", ["Nonseparated occupancies", "shall be individually classified"]),
            ev("IBC2021", "508.4", ["Separated occupancies", "fire barriers", "Table 508.4"]),
        ],
    },
    {
        "query_id": "ARCH_Q17",
        "title": "Allowable area and sprinkler/frontage increases",
        "natural_language_query": "Select a building footprint and allowable area strategy while balancing frontage and sprinkler increases.",
        "structured_scenario": {"domain": "architecture", "facility": "building_area", "frontage_available": True, "sprinklered": "scenario_dependent"},
        "decision_variables": ["actual_building_area_sqft", "allowable_area_sqft", "frontage_increase_factor", "sprinkler_increase_factor"],
        "objectives": ["maximize_floor_area", "minimize_added_protection_cost"],
        "expected_rule_families": ["IBC allowable area", "IBC frontage increase", "IBC sprinkler increase"],
        "interaction_types": ["parameterized_constraint", "scenario_condition"],
        "evidence_specs": [
            ev("IBC2021", "506.2", ["Allowable area", "Table 506.2", "Equation"]),
            ev("IBC2021", "506.3", ["Frontage increase", "public way", "open space"]),
        ],
    },
    {
        "query_id": "ARCH_Q18",
        "title": "High-rise safety systems package",
        "natural_language_query": "Evaluate whether a high-rise design triggers additional safety systems while minimizing added system cost.",
        "structured_scenario": {"domain": "architecture", "facility": "high_rise", "occupied_floor_height_ft": "scenario_dependent"},
        "decision_variables": ["occupied_floor_height_ft", "sprinkler_system", "fire_alarm_system", "smoke_control_required"],
        "objectives": ["minimize_life_safety_system_cost", "maximize_code_margin"],
        "expected_rule_families": ["IBC high-rise provisions", "sprinkler/alarm dependencies"],
        "interaction_types": ["threshold_condition", "dependency"],
        "evidence_specs": [
            ev("IBC2021", "403.1", ["High-rise buildings", "occupied floor", "75 feet"]),
            ev("IBC2021", "403.3", ["Automatic sprinkler system", "high-rise buildings"]),
        ],
    },
    {
        "query_id": "ARCH_Q19",
        "title": "Hazardous material control area quantity",
        "natural_language_query": "Determine whether a hazardous-material storage room stays below maximum allowable quantity while minimizing control-area upgrades.",
        "structured_scenario": {"domain": "architecture", "facility": "hazardous_storage", "control_area": True},
        "decision_variables": ["hazard_quantity", "material_class", "control_area_count", "storage_condition"],
        "objectives": ["maximize_allowed_storage", "minimize_hazard_control_cost"],
        "expected_rule_families": ["IFC maximum allowable quantities", "control areas", "hazard identification"],
        "interaction_types": ["table_lookup", "scenario_condition"],
        "evidence_specs": [
            ev("IFC2021", "5003.1.1", ["Maximum allowable quantity", "Table 5003.1.1", "control areas"]),
            ev("IFC2021", "5003.5", ["Hazard identification signs", "provided for quantities"]),
        ],
    },
    {
        "query_id": "ARCH_Q20",
        "title": "Hazardous material ventilation and exhausted enclosure",
        "natural_language_query": "Design a hazardous material room and choose ventilation/enclosure measures while minimizing mechanical system cost.",
        "structured_scenario": {"domain": "architecture", "facility": "hazardous_material_room", "exhausted_enclosure": "scenario_dependent"},
        "decision_variables": ["room_quantity", "exhaust_rate", "enclosure_type", "gas_room_required"],
        "objectives": ["minimize_mechanical_cost", "maximize_hazard_safety_margin"],
        "expected_rule_families": ["IFC gas rooms", "IFC exhausted enclosures", "IFC hazardous material systems"],
        "interaction_types": ["dependency", "scenario_condition"],
        "evidence_specs": [
            ev("IFC2021", "5003.8.4", ["Gas rooms", "shall comply", "5003.8.4"]),
            ev("IFC2021", "5003.8.5", ["Exhausted enclosures", "shall comply", "5003.8.5"]),
        ],
    },
]


AVIATION_QUERIES: list[dict[str, Any]] = [
    {
        "query_id": "AVI_Q01",
        "title": "Straight departure track adjustment",
        "natural_language_query": "Design a straight instrument departure with a small track adjustment while minimizing lateral protected-area expansion.",
        "structured_scenario": {"domain": "aviation", "procedure": "instrument_departure", "departure_type": "straight"},
        "decision_variables": ["track_adjustment_deg", "pdg_percent", "protected_area_half_width"],
        "objectives": ["minimize_lateral_protection_width", "maximize_obstacle_clearance_margin"],
        "expected_rule_families": ["straight departure definition", "track adjustment angle limit"],
        "interaction_types": ["threshold_condition", "dimensional_constraints"],
        "evidence_specs": [
            ev("CAAC", "3.2.1.1", ["起始离场航迹", "15°以内", "直线离场"]),
            ev("CAAC", "3.2.4.2.1", ["起始离场航迹调整", "不超过 15°"]),
        ],
    },
    {
        "query_id": "AVI_Q02",
        "title": "Turning departure MOC before turn",
        "natural_language_query": "Design a departure with a turn greater than 15 degrees and determine the earliest safe turn point.",
        "structured_scenario": {"domain": "aviation", "procedure": "instrument_departure", "turn_angle_gt_15": True},
        "decision_variables": ["turn_angle_deg", "turn_start_height_m", "moc_before_turn_m"],
        "objectives": ["minimize_turn_start_distance", "maximize_moc_margin"],
        "expected_rule_families": ["turn departure definition", "minimum obstacle clearance before turn"],
        "interaction_types": ["threshold_condition", "dependency"],
        "evidence_specs": [
            ev("CAAC", "3.3.1.1", ["大于 15°", "转弯离场"]),
            ev("CAAC", "2.2.9", ["大于 15°", "75 m", "超障余度"]),
        ],
    },
    {
        "query_id": "AVI_Q03",
        "title": "Departure PDG publication due to close-in obstacle",
        "natural_language_query": "Choose a departure gradient that clears a close-in obstacle and determine whether a higher PDG must be published.",
        "structured_scenario": {"domain": "aviation", "procedure": "instrument_departure", "close_in_obstacle": True},
        "decision_variables": ["obstacle_height_m", "pdg_percent", "publish_pdg"],
        "objectives": ["minimize_required_climb_gradient", "maximize_obstacle_margin"],
        "expected_rule_families": ["PDG publication", "close-in obstacle threshold"],
        "interaction_types": ["threshold_condition", "auditability"],
        "evidence_specs": [
            ev("CAAC", "2.4", ["近距障碍物", "较大梯度", "60 m"]),
            ev("CAAC", "PDG", ["PDG", "60 m", "公布"]),
        ],
    },
    {
        "query_id": "AVI_Q04",
        "title": "Parallel runway simultaneous departure divergence",
        "natural_language_query": "Design two simultaneous parallel runway departures and select initial tracks that diverge enough after takeoff.",
        "structured_scenario": {"domain": "aviation", "procedure": "parallel_runway_departures", "simultaneous": True},
        "decision_variables": ["departure_track_1_deg", "departure_track_2_deg", "divergence_deg"],
        "objectives": ["minimize_track_change", "maximize_parallel_operation_safety"],
        "expected_rule_families": ["parallel runway instrument departures", "minimum divergence"],
        "interaction_types": ["cross_procedure_dependency", "threshold_condition"],
        "evidence_specs": [
            ev("CAAC", "6.1", ["平行跑道", "仪表离场", "散开至少15°"]),
        ],
    },
    {
        "query_id": "AVI_Q05",
        "title": "Minimum sector altitude with controlling obstacle",
        "natural_language_query": "Compute the MSA around an aerodrome by selecting the controlling obstacle while minimizing published altitude.",
        "structured_scenario": {"domain": "aviation", "procedure": "minimum_sector_altitude", "radius_nm": 25},
        "decision_variables": ["obstacle_elevation_ft", "sector_radius_nm", "msa_ft", "clearance_ft"],
        "objectives": ["minimize_msa", "maximize_obstacle_clearance"],
        "expected_rule_families": ["MSA definition", "sector obstacle clearance"],
        "interaction_types": ["aggregation", "max_constraint"],
        "evidence_specs": [
            ev("CAAC", "第 8 章", ["最低扇区高度", "MSA"]),
            ev("CAAC", "1.1.77", ["最低扇区高度", "minimum sector altitude"]),
        ],
    },
    {
        "query_id": "AVI_Q06",
        "title": "Aircraft category from Vat",
        "natural_language_query": "Classify aircraft category from Vat and choose the applicable procedure speed envelope.",
        "structured_scenario": {"domain": "aviation", "procedure": "aircraft_category", "vat_known": True},
        "decision_variables": ["vat_kt", "aircraft_category", "max_speed_kt"],
        "objectives": ["minimize_category_conservatism", "maintain_category_validity"],
        "expected_rule_families": ["aircraft category speed table", "Vat definition"],
        "interaction_types": ["table_lookup", "threshold_condition"],
        "evidence_specs": [
            ev("CAAC", "Vat", ["航空器分类", "Vat", "Vso"]),
            ev("CAAC", "表", ["A", "B", "C", "D", "E", "Vat"]),
        ],
    },
    {
        "query_id": "AVI_Q07",
        "title": "Standard departure climb gradient",
        "natural_language_query": "Design a standard instrument departure and determine the minimum climb gradient needed from DER.",
        "structured_scenario": {"domain": "aviation", "procedure": "standard_departure", "obstacles_assessed": True},
        "decision_variables": ["pdg_percent", "der_height_m", "climb_gradient_margin"],
        "objectives": ["minimize_climb_gradient", "maximize_clearance_margin"],
        "expected_rule_families": ["standard PDG", "DER climb start"],
        "interaction_types": ["dimensional_constraints", "parameterized_constraint"],
        "evidence_specs": [
            ev("CAAC", "PDG", ["3.3%", "DER", "程序设计梯度"]),
            ev("CAAC", "DER", ["跑道起飞末端", "DER"]),
        ],
    },
    {
        "query_id": "AVI_Q08",
        "title": "Initial/intermediate approach protected-area width",
        "natural_language_query": "Design an initial approach segment and size the main/secondary protected areas while minimizing lateral width.",
        "structured_scenario": {"domain": "aviation", "procedure": "initial_approach", "segment": "initial_or_intermediate"},
        "decision_variables": ["main_area_half_width_km", "secondary_area_width_km", "segment_length_nm"],
        "objectives": ["minimize_protected_width", "maximize_containment_margin"],
        "expected_rule_families": ["primary area width", "secondary area width"],
        "interaction_types": ["dimensional_constraints", "dependency"],
        "evidence_specs": [
            ev("CAAC", "3.4", ["主区宽度", "4.6 km", "2.5 NM"]),
            ev("CAAC", "3.4", ["副区", "4.6 km", "2.5 NM"]),
        ],
    },
    {
        "query_id": "AVI_Q09",
        "title": "Procedure turn timing by aircraft category",
        "natural_language_query": "Design a procedure turn and choose outbound timing based on aircraft category while preserving protected airspace.",
        "structured_scenario": {"domain": "aviation", "procedure": "procedure_turn", "category_dependent": True},
        "decision_variables": ["aircraft_category", "outbound_time_sec", "turn_radius_nm", "protected_area_width_nm"],
        "objectives": ["minimize_procedure_time", "maximize_airspace_containment"],
        "expected_rule_families": ["procedure turn timing", "category-dependent speed/time"],
        "interaction_types": ["scenario_condition", "table_lookup"],
        "evidence_specs": [
            ev("CAAC", "3.4.6", ["出航航迹长度", "C、D", "1 min 15 s"]),
            ev("CAAC", "TAS", ["TAS", "170 kt", "坡度"]),
        ],
    },
    {
        "query_id": "AVI_Q10",
        "title": "DME arc protected area",
        "natural_language_query": "Design a DME arc approach segment and choose arc geometry that stays inside protected-area limits.",
        "structured_scenario": {"domain": "aviation", "procedure": "dme_arc_approach", "inbound_leg_length_lt_25nm": True},
        "decision_variables": ["arc_radius_nm", "outer_boundary_nm", "inner_boundary_nm", "iaf_position"],
        "objectives": ["minimize_arc_length", "maximize_protected_area_margin"],
        "expected_rule_families": ["DME arc geometry", "protected area width"],
        "interaction_types": ["geometric_constraints", "dimensional_constraints"],
        "evidence_specs": [
            ev("CAAC", "DME", ["DME", "弧", "保护区"]),
            ev("CAAC", "5.2 NM", ["5.2 NM", "8.0 NM", "3.5 NM"]),
        ],
    },
    {
        "query_id": "AVI_Q11",
        "title": "MAPt transition tolerance",
        "natural_language_query": "Compute MAPt transition tolerance when the missed approach point is defined by a fix or station.",
        "structured_scenario": {"domain": "aviation", "procedure": "missed_approach", "mapt_defined_by_fix": True},
        "decision_variables": ["mapt_position_nm", "transition_distance_x_nm", "longitudinal_tolerance_nm"],
        "objectives": ["minimize_tolerance_buffer", "maximize_missed_approach_safety"],
        "expected_rule_families": ["MAPt tolerance", "transition distance"],
        "interaction_types": ["parameterized_constraint", "dependency"],
        "evidence_specs": [
            ev("CAAC", "MAPt", ["MAPt", "过渡距离", "x"]),
            ev("CAAC", "6.1.6.2.1", ["MAPt容差", "过渡容差"]),
        ],
    },
    {
        "query_id": "AVI_Q12",
        "title": "Final approach obstacle clearance",
        "natural_language_query": "Set final approach minima so that obstacle clearance is maintained while minimizing MDA/OCA.",
        "structured_scenario": {"domain": "aviation", "procedure": "final_approach", "nonprecision": True},
        "decision_variables": ["obstacle_elevation_ft", "moc_ft", "mda_ft", "oca_ft"],
        "objectives": ["minimize_mda_or_oca", "maximize_obstacle_clearance"],
        "expected_rule_families": ["final approach MOC", "OCA/MDA computation"],
        "interaction_types": ["max_constraint", "dimensional_constraints"],
        "evidence_specs": [
            ev("CAAC", "MOC", ["最后进近", "MOC", "超障余度"]),
            ev("CAAC", "OCA", ["OCA", "障碍物", "超障"]),
        ],
    },
    {
        "query_id": "AVI_Q13",
        "title": "RNP APCH lateral accuracy and RF requirement",
        "natural_language_query": "Design an RNP APCH segment and determine whether the selected navigation accuracy and RF turn are valid.",
        "structured_scenario": {"domain": "aviation", "procedure": "rnp_apch", "rf_turn_candidate": True},
        "decision_variables": ["rnp_value_nm", "rf_required", "lateral_containment_nm", "turn_radius_nm"],
        "objectives": ["minimize_navigation_requirement", "maximize_path_containment"],
        "expected_rule_families": ["RNP APCH", "RNP 0.3", "RF requirement"],
        "interaction_types": ["scenario_condition", "dependency"],
        "evidence_specs": [
            ev("CAAC", "RNP APCH", ["RNP APCH", "RNP 0.3"]),
            ev("CAAC", "RF", ["RNP 0.3", "要求RF"]),
        ],
    },
    {
        "query_id": "AVI_Q14",
        "title": "RNAV/RNP navigation specification selection",
        "natural_language_query": "Select a navigation specification for a terminal route while minimizing equipment burden and satisfying accuracy needs.",
        "structured_scenario": {"domain": "aviation", "procedure": "pbn_route", "terminal_area": True},
        "decision_variables": ["nav_spec", "required_accuracy_nm", "equipment_capability", "alerting_required"],
        "objectives": ["minimize_equipment_burden", "maximize_navigation_integrity"],
        "expected_rule_families": ["RNAV 1", "RNP 1", "A-RNP", "RNP 0.3"],
        "interaction_types": ["alternative_valid_structures", "scenario_condition"],
        "evidence_specs": [
            ev("CAAC", "RNAV 1", ["RNAV 1", "RNP 1", "A RNP"]),
            ev("CAAC", "RNP 0.3", ["RNP 0.3", "导航精度"]),
        ],
    },
    {
        "query_id": "AVI_Q15",
        "title": "SBAS straight missed-approach protected width",
        "natural_language_query": "Size the SBAS straight missed-approach protected area while minimizing lateral protection width.",
        "structured_scenario": {"domain": "aviation", "procedure": "sbas_missed_approach", "straight": True},
        "decision_variables": ["main_half_width_km", "secondary_width_km", "total_width_km", "climb_gradient_percent"],
        "objectives": ["minimize_lateral_width", "maximize_sbas_clearance_margin"],
        "expected_rule_families": ["SBAS APV missed approach", "protected width"],
        "interaction_types": ["dimensional_constraints", "dependency"],
        "evidence_specs": [
            ev("CAAC", "SBAS APV", ["SBAS", "1.76 km", "0.95 NM"]),
            ev("CAAC", "SBAS OAS", ["爬升梯度", "2.5%", "20%扩张角"]),
        ],
    },
    {
        "query_id": "AVI_Q16",
        "title": "Vertical path angle and height-loss margin",
        "natural_language_query": "Evaluate whether a BARO-VNAV/RNP AR approach with a steep vertical path angle requires additional height-loss margin.",
        "structured_scenario": {"domain": "aviation", "procedure": "baro_vnav_or_rnp_ar", "vpa_gt_3_2": True},
        "decision_variables": ["vertical_path_angle_deg", "height_loss_margin_m", "oca_h_m"],
        "objectives": ["minimize_oca_h", "maximize_height_loss_margin"],
        "expected_rule_families": ["vertical path angle threshold", "height loss adjustment"],
        "interaction_types": ["threshold_condition", "parameter_propagation"],
        "evidence_specs": [
            ev("CAAC", "7.4.6.15", ["垂直航径角", "大于 3.2°", "高度损失"]),
            ev("CAAC", "7.4.6.16", ["大于 3.5°", "标称下降率"]),
        ],
    },
    {
        "query_id": "AVI_Q17",
        "title": "Helicopter PinS final segment obstacle clearance",
        "natural_language_query": "Design a helicopter PinS final segment from FAF to MAPt while minimizing OCA/H.",
        "structured_scenario": {"domain": "aviation", "procedure": "helicopter_pins_final", "rnp_0_3": True},
        "decision_variables": ["faf_position_nm", "mapt_position_nm", "moc_m", "oca_h_m"],
        "objectives": ["minimize_oca_h", "maximize_moc_margin"],
        "expected_rule_families": ["PinS final segment", "MOC 75 m", "FAF/MAPt protection"],
        "interaction_types": ["dimensional_constraints", "dependency"],
        "evidence_specs": [
            ev("CAAC", "2.7.4", ["主区的最低超障余度", "75 m", "246 ft"]),
            ev("CAAC", "2.7.3.4", ["FAF", "MAPt", "保护区"]),
        ],
    },
    {
        "query_id": "AVI_Q18",
        "title": "Helicopter visual segment OIS",
        "natural_language_query": "Design the visual segment from MAPt to HRP and check whether obstacles penetrate the visual OIS.",
        "structured_scenario": {"domain": "aviation", "procedure": "helicopter_pins_visual_segment", "day_or_night": "scenario_dependent"},
        "decision_variables": ["mapt_hrp_distance_m", "ois_buffer_m", "expansion_rate_percent", "obstacle_height_m"],
        "objectives": ["minimize_visual_segment_length", "maximize_ois_clearance"],
        "expected_rule_families": ["PinS visual segment", "OIS buffer", "day/night expansion"],
        "interaction_types": ["scenario_condition", "threshold_condition"],
        "evidence_specs": [
            ev("CAAC", "2.9.3.3.3.1", ["OIS", "MAPt", "741 m", "0.4 NM"]),
            ev("CAAC", "2.9.2.2.2.3", ["白天", "10%", "晚上", "15%"]),
        ],
    },
    {
        "query_id": "AVI_Q19",
        "title": "Near-parallel runway classification",
        "natural_language_query": "Determine whether two non-intersecting runways qualify as near-parallel for simultaneous instrument approach analysis.",
        "structured_scenario": {"domain": "aviation", "procedure": "near_parallel_runways", "runways_non_intersecting": True},
        "decision_variables": ["centerline_angle_deg", "near_parallel_flag", "simultaneous_operation_mode"],
        "objectives": ["maximize_runway_throughput", "maintain_separation_safety"],
        "expected_rule_families": ["near-parallel runway definition", "simultaneous approach applicability"],
        "interaction_types": ["threshold_condition", "classification"],
        "evidence_specs": [
            ev("CAAC", "1.1.90", ["近似平行的跑道", "等于或小于 15°", "非交叉跑道"]),
            ev("CAAC", "第 10 章", ["平行或近似平行仪表跑道", "同时进近"]),
        ],
    },
    {
        "query_id": "AVI_Q20",
        "title": "Procedure validation and source-data audit",
        "natural_language_query": "Check whether a newly designed instrument procedure has sufficient ground/flight validation evidence before publication.",
        "structured_scenario": {"domain": "aviation", "procedure": "procedure_validation", "publication_candidate": True},
        "decision_variables": ["ground_validation_done", "flight_validation_required", "source_data_verified", "publication_allowed"],
        "objectives": ["minimize_validation_cost", "maximize_publication_assurance"],
        "expected_rule_families": ["ground validation", "flight validation", "source data quality"],
        "interaction_types": ["dependency", "auditability"],
        "evidence_specs": [
            ev("CAAC", "4.6.1", ["验证", "飞行程序"]),
            ev("CAAC", "4.6.2", ["地面验证"]),
            ev("CAAC", "4.6.3", ["飞行验证"]),
            ev("CAAC", "4.6.4", ["数据来源于飞行程序设计"]),
        ],
    },
]


def materialize_query(query: dict[str, Any]) -> dict[str, Any]:
    evidence = [
        find_evidence(spec["doc_key"], spec["section"], spec["keywords"])
        for spec in query.pop("evidence_specs")
    ]
    out = dict(query)
    out["evidence"] = evidence
    out["evidence_found_rate"] = sum(
        1 for item in evidence if item["match_score"]["keyword_hits"] > 0
    ) / max(len(evidence), 1)
    return out


def coverage(queries: list[dict[str, Any]]) -> dict[str, Any]:
    docs: dict[str, int] = {}
    interactions: dict[str, int] = {}
    for q in queries:
        for e in q["evidence"]:
            docs[e["doc_key"]] = docs.get(e["doc_key"], 0) + 1
        for t in q["interaction_types"]:
            interactions[t] = interactions.get(t, 0) + 1
    return {"documents": docs, "interaction_types": interactions}


def to_markdown(queries: list[dict[str, Any]], title: str) -> str:
    lines = [f"# {title}", ""]
    lines.append("This catalog is built from original source PDFs, not from QA500 records.")
    lines.append("")
    lines.append("## Diversity Coverage")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(coverage(queries), ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    for q in queries:
        lines.append(f"## {q['query_id']}: {q['title']}")
        lines.append("")
        lines.append(f"- Natural-language query: {q['natural_language_query']}")
        lines.append(f"- Scenario omega: `{json.dumps(q['structured_scenario'], ensure_ascii=False)}`")
        lines.append(f"- Decision variables: {', '.join(q['decision_variables'])}")
        lines.append(f"- Objectives: {', '.join(q['objectives'])}")
        lines.append(f"- Expected rule families: {', '.join(q['expected_rule_families'])}")
        lines.append(f"- Interaction types: {', '.join(q['interaction_types'])}")
        lines.append("- Evidence:")
        for e in q["evidence"]:
            snippet = e["snippet"][:280].replace("|", "/")
            lines.append(
                f"  - {e['document']}, {e['section_or_clause']}, text-page {e['pdf_text_page']}: {snippet}"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    architecture = [materialize_query(q) for q in ARCHITECTURE_QUERIES]
    aviation = [materialize_query(q) for q in AVIATION_QUERIES]
    source_documents = {
        key: {
            "title": value["title"],
            "pdf": str(value["pdf"]),
            "text": str(value["text"]),
        }
        for key, value in DOCS.items()
    }
    payload = {
        "version": "domain-query-catalog-v1",
        "note": "Built from original source PDFs. QA500 files are intentionally not used.",
        "source_documents": source_documents,
        "architecture_queries": architecture,
        "aviation_queries": aviation,
        "coverage": {
            "architecture": coverage(architecture),
            "aviation": coverage(aviation),
        },
    }
    out_json = PAPER_DIR / "domain_query_catalog_40.json"
    out_md = PAPER_DIR / "DOMAIN_QUERY_CATALOG_40.md"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(
        to_markdown(architecture, "Architecture Domain Query Catalog")
        + "\n\n"
        + to_markdown(aviation, "Flight Procedure Design Query Catalog"),
        encoding="utf-8",
    )
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    print(json.dumps(payload["coverage"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
