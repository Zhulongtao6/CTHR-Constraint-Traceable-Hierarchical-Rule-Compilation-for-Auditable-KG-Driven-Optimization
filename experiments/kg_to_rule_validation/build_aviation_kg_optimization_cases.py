from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CTHR_ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = CTHR_ROOT / "paper"
RULE_LIBRARY = (
    PAPER_DIR
    / "full_aviation_kg_rule_library_model_comparison"
    / "full_aviation_rule_library_qwen.json"
)
OUT_JSON = PAPER_DIR / "aviation_kg_generated_19_optimization_problems.json"
OUT_MD = PAPER_DIR / "AVIATION_KG_OPTIMIZATION_19_SUMMARY.md"


def load_rule_index() -> dict[str, dict[str, Any]]:
    library = json.loads(RULE_LIBRARY.read_text(encoding="utf-8"))
    return {rule["rule_id"]: rule for rule in library["rules"]}


def evidence_for(rule_ids: list[str], rule_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    chunks: list[str] = []
    nodes: list[str] = []
    edges: list[str] = []
    provenance: list[dict[str, Any]] = []
    missing: list[str] = []
    for rid in rule_ids:
        rule = rule_index.get(rid)
        if not rule:
            missing.append(rid)
            continue
        chunks.extend(str(x) for x in rule.get("source_chunk_ids", []))
        nodes.extend(str(x) for x in rule.get("source_node_ids", []))
        provenance.extend(rule.get("provenance", []))
        for constraint in rule.get("constraints", []):
            ev = constraint.get("evidence", {})
            chunks.extend(str(x) for x in ev.get("chunk_ids", []))
            nodes.extend(str(x) for x in ev.get("kg_node_ids", []))
            edges.extend(str(x) for x in ev.get("kg_edge_ids", []))
        for relation in rule.get("relations", []):
            ev = relation.get("evidence", {})
            chunks.extend(str(x) for x in ev.get("chunk_ids", []))
            nodes.extend(str(x) for x in ev.get("kg_node_ids", []))
            edges.extend(str(x) for x in ev.get("kg_edge_ids", []))
    return {
        "kg_chunk_ids": sorted(set(chunks)),
        "kg_node_ids": sorted(set(nodes)),
        "kg_edge_ids": sorted(set(edges)),
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
    }
    if valid_constraint_cells is not None:
        hidden_reference["valid_constraint_cells"] = valid_constraint_cells
    return {
        "omega_id": omega_id,
        "domain": "aviation_procedure_design",
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
            "constraint_form": "linear, piecewise-linear, or linearized numeric regulatory constraints",
            "recommended_solver": "QP/MILP/SMT/exact symbolic solver; CPO is optional",
        },
        "notes": notes,
    }


def cons(cid: str, expression: str, source: str, role: str) -> dict[str, str]:
    key = "source_rule_id" if source.startswith("rule:") else "source"
    value = source[5:] if source.startswith("rule:") else source
    return {"constraint_id": cid, "expression": expression, key: value, "role": role}


def build_cases() -> list[dict[str, Any]]:
    return [
        case(
            "AVI_OPT_01",
            "Base-turn outbound timing with wind robustness",
            "holding_procedure_design",
            "Design the outbound timing for a 30 degree offset base-turn holding procedure under a headwind/tailwind uncertainty band. The timing must satisfy the KG-derived outbound-time limit and timing-tolerance rule, while the optimizer trades off a shorter holding segment against a larger timing buffer above the wind-corrected minimum.",
            {
                "maneuver_type": "base_turn",
                "offset_angle_deg": 30,
                "track_accuracy": "+/-5deg",
                "wind_uncertainty_s": 8,
                "wind_adjusted_reference_outbound_time_s": 68,
            },
            {
                "outbound_flight_time_s": {"type": "continuous", "unit": "s", "lower": 55, "upper": 100},
                "timing_buffer_s": {"type": "continuous", "unit": "s", "lower": 0, "upper": 32},
                "remaining_time_allowance_s": {"type": "continuous", "unit": "s", "lower": 0, "upper": 35},
            },
            [
                {"name": "minimize_holding_time", "expression": "outbound_flight_time_s"},
                {"name": "maximize_wind_robustness", "expression": "timing_buffer_s"},
            ],
            {"lambda": [0.55, 0.45], "meaning": "prefer a short hold, but keep enough timing buffer for wind uncertainty"},
            ["holding_outbound_time_limit_90s", "RA-3.6.3-operating-assumption-timing-tolerance"],
            [
                cons("C1", "outbound_flight_time_s <= 90", "rule:holding_outbound_time_limit_90s", "outbound_time_limit"),
                cons("C2", "outbound_flight_time_s >= wind_adjusted_reference_outbound_time_s", "scenario_wind_model", "wind_corrected_lower_bound"),
                cons("C3", "timing_buffer_s = outbound_flight_time_s - wind_adjusted_reference_outbound_time_s", "scenario_wind_model", "robustness_buffer_definition"),
                cons("C4", "remaining_time_allowance_s = 90 - outbound_flight_time_s", "rule:holding_outbound_time_limit_90s", "time_allowance_certificate"),
                cons("C5", "wind_uncertainty_s <= 10", "rule:RA-3.6.3-operating-assumption-timing-tolerance", "timing_tolerance_activation"),
            ],
            {
                "should_activate": ["base-turn outbound time limit", "timing tolerance rule"],
                "should_exclude": ["non-holding timing rules"],
                "should_resolve": ["dependency between holding design and operating-assumption timing tolerance"],
            },
            "This case has a real trade-off: reducing outbound time lowers delay, but also reduces wind-robustness buffer.",
        ),
        case(
            "AVI_OPT_02",
            "Base-turn scheduled timing under biased wind",
            "holding_procedure_design",
            "Choose a scheduled outbound time for a base-turn procedure when forecast wind introduces a biased timing target. The schedule must remain within the KG-derived timing tolerance, while the optimizer trades off a shorter published timing against lower expected timing error.",
            {
                "maneuver_type": "base_turn",
                "timing_reference": "outbound timing",
                "forecast_wind_bias_s": 4,
                "wind_corrected_target_time_s": 82,
            },
            {
                "scheduled_outbound_time_s": {"type": "continuous", "unit": "s", "lower": 65, "upper": 95},
                "absolute_timing_error_s": {"type": "continuous", "unit": "s", "lower": 0, "upper": 20},
                "tolerance_margin_s": {"type": "continuous", "unit": "s", "lower": 0, "upper": 20},
            },
            [
                {"name": "minimize_scheduled_time", "expression": "scheduled_outbound_time_s"},
                {"name": "minimize_expected_timing_error", "expression": "absolute_timing_error_s"},
            ],
            {"lambda": [0.45, 0.55], "meaning": "slightly prioritize timing accuracy over a shorter schedule"},
            ["RA-3.6.3-operating-assumption-timing-tolerance", "holding_outbound_time_limit_90s"],
            [
                cons("C1", "scheduled_outbound_time_s <= 90", "rule:holding_outbound_time_limit_90s", "outbound_time_limit"),
                cons("C2", "absolute_timing_error_s >= scheduled_outbound_time_s - wind_corrected_target_time_s", "scenario_wind_model", "absolute_error_positive_side"),
                cons("C3", "absolute_timing_error_s >= wind_corrected_target_time_s - scheduled_outbound_time_s", "scenario_wind_model", "absolute_error_negative_side"),
                cons("C4", "absolute_timing_error_s <= 10", "rule:RA-3.6.3-operating-assumption-timing-tolerance", "timing_tolerance_limit"),
                cons("C5", "tolerance_margin_s = 10 - absolute_timing_error_s", "rule:RA-3.6.3-operating-assumption-timing-tolerance", "certificate_margin"),
            ],
            {
                "should_activate": ["timing tolerance rule", "holding outbound-time limit"],
                "should_exclude": ["unrelated segment timing rules"],
                "should_resolve": ["scenario-dependent target time modifies the objective but not the source tolerance"],
            },
            "A very short schedule increases expected timing error; a schedule close to 82 s is robust but less efficient.",
        ),
        case(
            "AVI_OPT_03",
            "VOR fix tolerance radius with extra buffer",
            "fix_tolerance_design",
            "Design the tolerance radius for a VOR fix located 22 km from the station. The tolerance geometry must be grounded from the applicable VOR rule, while the optimizer trades off a smaller design radius against a larger extra protection buffer above the KG-derived minimum tolerance radius.",
            {
                "navigation_aid_type": "vor",
                "station_distance_km": 22,
                "design_task": "select VOR fix tolerance design radius",
            },
            {
                "r_design_km": {"type": "continuous", "unit": "km", "lower": 24, "upper": 36},
            },
            [
                {"name": "minimize_design_radius", "expression": "r_design_km"},
                {"name": "maximize_extra_protection_buffer", "expression": "r_design_km - KG_grounded_minimum_tolerance_radius"},
            ],
            {"lambda": [0.5, 0.5], "meaning": "balance compact VOR tolerance geometry and extra protection buffer"},
            ["vor_fix_tolerance_cone_half_angle"],
            [
                cons("C1", "VOR inverted-cone half-angle is grounded as 50 deg", "rule:vor_fix_tolerance_cone_half_angle", "kg_half_angle"),
                cons("C2", "KG-grounded minimum tolerance radius is computed from station distance and the grounded VOR half-angle", "rule:vor_fix_tolerance_cone_half_angle", "minimum_tolerance_radius_formula"),
                cons("C3", "KG-grounded minimum tolerance radius is 26.2 km for the 22 km station-distance scenario", "scenario_geometry_linearization", "grounded_minimum_tolerance_radius_for_22km_and_50deg"),
                cons("C4", "r_design_km >= 26.2", "scenario_tolerance_design_model", "design_radius_covers_minimum_tolerance_radius"),
            ],
            {
                "should_activate": ["VOR inverted-cone tolerance rule"],
                "should_exclude": ["DME-only tolerance rules"],
                "should_resolve": ["dependency from navigation-aid type to the VOR tolerance geometry", "rule-library half-angle is grounded before computing R_min"],
            },
            "The query supplies only the station distance and the design objective. CTHR supplies the VOR tolerance rule, grounds the half-angle, and derives the minimum tolerance-radius constraint.",
        ),
        case(
            "AVI_OPT_04",
            "VOR/DME MSA inner-sector arc-radius design",
            "sector_partition_design",
            "Design the DME arc boundary of an MSA inner sector in a VOR/DME procedure. A larger inner-sector radius allows the lower MSA sector to cover more tracks, but if the radius becomes too large it may include a higher obstacle and raise the published inner-sector MSA. The optimizer therefore trades off maximizing the inner-sector radius r against minimizing the inner-sector MSA.",
            {
                "procedure_type": "VOR/DME",
                "design_task": "select MSA inner-sector DME arc radius r",
                "sector_partition_goal": "MSA_inner_sector",
                "obstacles": [
                    {"id": "A", "bearing_deg": 45, "distance_from_vor_dme_km": 14, "elevation_m": 620},
                    {"id": "B", "bearing_deg": 60, "distance_from_vor_dme_km": 24, "elevation_m": 910},
                    {"id": "C", "bearing_deg": 90, "distance_from_vor_dme_km": 31, "elevation_m": 870},
                ],
            },
            {
                "r_km": {"type": "continuous", "unit": "km", "lower": 10, "upper": 35},
                "inner_sector_msa_m": {"type": "continuous", "unit": "m", "lower": 900, "upper": 1300},
                "obstacle_b_in_inner_sector": {"type": "binary", "unit": "indicator", "lower": 0, "upper": 1},
            },
            [
                {"name": "maximize_inner_sector_radius", "expression": "r_km"},
                {"name": "minimize_inner_sector_msa", "expression": "inner_sector_msa_m"},
            ],
            {"lambda": [0.5, 0.5], "meaning": "balance wider low-MSA inner-sector coverage against a lower published inner-sector MSA"},
            ["dme_arc_radius_range_sector_partition"],
            [
                cons("C0", "minimum obstacle clearance is grounded as 300 m", "scenario_msa_design_requirement", "hidden_msa_moc_parameter"),
                cons("C1", "r_km >= 19", "rule:dme_arc_radius_range_sector_partition", "minimum_dme_arc_radius"),
                cons("C2", "r_km <= 28", "rule:dme_arc_radius_range_sector_partition", "maximum_dme_arc_radius"),
                cons("C3", "obstacle_b_in_inner_sector in {0, 1}", "scenario_msa_sector_model", "binary_obstacle_b_membership"),
                cons("C4", "r_km <= 23.99 + 4.01 * obstacle_b_in_inner_sector", "scenario_msa_sector_model", "if_b_not_included_radius_stays_below_b_distance"),
                cons("C5", "r_km >= 19 + 5 * obstacle_b_in_inner_sector", "scenario_msa_sector_model", "if_b_included_radius_reaches_b_distance"),
                cons("C6", "inner_sector_msa_m >= 620 + 300 + 290 * obstacle_b_in_inner_sector", "scenario_msa_design_requirement", "msa_from_controlling_inner_sector_obstacle"),
                cons("C7", "31 > 28", "scenario_msa_sector_model", "obstacle_c_outside_maximum_inner_sector_radius"),
            ],
            {
                "should_activate": ["DME arc-radius range for sector partition", "MSA obstacle-clearance construction"],
                "should_exclude": ["non-DME sector partition rules"],
                "should_resolve": ["obstacle A is always inside the inner sector", "obstacle B enters only when radius reaches 24 km", "obstacle C remains outside because it is beyond the 28 km maximum radius"],
            },
            "Obstacle A is inside any valid 19--28 km inner-sector radius and gives MSA 920 m. Obstacle B enters once r reaches 24 km and raises the required MSA to 1210 m. Obstacle C is at 31 km and remains outside the valid inner-sector radius. This creates a real trade-off between larger low-MSA sector coverage and lower published MSA.",
        ),
        case(
            "AVI_OPT_05",
            "ILS glide path angle with controlling obstacle",
            "precision_approach_design",
            "Select an ILS glide path angle for an approach with a controlling obstacle 1.2 NM before the threshold. The angle must remain inside the KG-derived allowable ILS range and provide the required obstacle-clearance height, while the optimizer trades off closeness to the nominal 3.0 degree glide path against additional obstacle-clearance margin.",
            {
                "procedure_type": "instrument landing system (ils)",
                "operation_category": "I",
                "runway_code": 4,
                "threshold_crossing_height_ft": 50,
                "controlling_obstacle_distance_from_threshold_nm": 1.2,
                "controlling_obstacle_height_ft_above_threshold": 360,
            },
            {
                "glide_path_angle_deg": {"type": "continuous", "unit": "degree", "lower": 2.0, "upper": 4.0},
                "angle_deviation_from_optimum_deg": {"type": "continuous", "unit": "degree", "lower": 0, "upper": 1.0},
                "glide_path_height_at_obstacle_ft": {"type": "continuous", "unit": "ft", "lower": 300, "upper": 520},
                "obstacle_clearance_margin_ft": {"type": "continuous", "unit": "ft", "lower": 0, "upper": 80},
            },
            [
                {"name": "minimize_deviation_from_optimum", "expression": "angle_deviation_from_optimum_deg"},
                {"name": "maximize_obstacle_clearance_margin", "expression": "obstacle_clearance_margin_ft"},
            ],
            {"lambda": [0.6, 0.4], "meaning": "prefer a glide path close to 3.0 degrees while retaining obstacle-clearance margin"},
            ["ils_glidepath_angle_range", "rule_non_si_glide_path_formula"],
            [
                cons("C0", "required obstacle clearance is grounded as 95 ft", "scenario_obstacle_definition", "hidden_controlling_obstacle_clearance_requirement"),
                cons("C1", "glide_path_angle_deg >= 2.5", "rule:ils_glidepath_angle_range", "minimum_glide_angle"),
                cons("C2", "glide_path_angle_deg <= 3.5", "rule:ils_glidepath_angle_range", "maximum_glide_angle"),
                cons("C3", "angle_deviation_from_optimum_deg >= abs(glide_path_angle_deg - 3.0)", "rule:ils_glidepath_angle_range", "optimum_angle_auxiliary"),
                cons("C4", "glide_path_height_at_obstacle_ft = 50.7 + 127.6 * glide_path_angle_deg", "rule:rule_non_si_glide_path_formula", "linearized_height_at_obstacle_model"),
                cons("C5", "obstacle_clearance_margin_ft = glide_path_height_at_obstacle_ft - (360 + 95)", "scenario_obstacle_definition", "clearance_margin_definition"),
                cons("C6", "obstacle_clearance_margin_ft >= 0", "scenario_obstacle_definition", "controlling_obstacle_clearance_requirement"),
            ],
            {
                "should_activate": ["ILS glide-path range", "non-SI glide-path height formula"],
                "should_exclude": ["helicopter VSDA visual-segment rules"],
                "should_resolve": ["formula dependency from glide-path angle to obstacle-clearance height"],
            },
            "At 3.0 degrees, the linearized height at the obstacle is about 433.5 ft, below the required 455 ft height.",
        ),
        case(
            "AVI_OPT_07",
            "Intermediate approach descent gradient and segment length",
            "intermediate_approach_design",
            "Design an intermediate-approach vertical profile over a known obstacle environment. The descent profile must satisfy the KG-derived maximum gradient and MOC rules, while the optimizer trades off a shorter intermediate segment against a shallower descent gradient and larger clearance margin.",
            {
                "segment_type": "intermediate approach segment",
                "aircraft_category": "C",
                "max_obstacle_elevation_m": 640,
                "initial_altitude_m": 1040,
                "target_altitude_m": 820,
            },
            {
                "segment_length_km": {"type": "continuous", "unit": "km", "lower": 5, "upper": 14},
                "descent_gradient_percent": {"type": "continuous", "unit": "%", "lower": 0, "upper": 8},
                "clearance_margin_m": {"type": "continuous", "unit": "m", "lower": 0, "upper": 250},
            },
            [
                {"name": "minimize_segment_length", "expression": "segment_length_km"},
                {"name": "minimize_descent_gradient", "expression": "descent_gradient_percent"},
            ],
            {"lambda": [0.5, 0.5], "meaning": "balance shorter procedure footprint against a gentler descent"},
            ["intermediate_approach_gradient_max", "moc_intermediate_approach_min"],
            [
                cons("C1", "descent_gradient_percent <= 5.0", "rule:intermediate_approach_gradient_max", "maximum_descent_gradient"),
                cons("C2", "clearance_margin_m >= 150", "rule:moc_intermediate_approach_min", "minimum_obstacle_clearance"),
                cons("C3", "altitude-loss feasibility is represented by three piecewise-linear vertical-profile cells", "scenario_vertical_profile", "piecewise_linear_altitude_loss_feasibility"),
                cons("C4", "target_altitude_m >= max_obstacle_elevation_m + clearance_margin_m", "rule:moc_intermediate_approach_min", "obstacle_clearance_altitude"),
            ],
            {
                "should_activate": ["intermediate approach gradient limit", "intermediate MOC rule"],
                "should_exclude": ["missed-approach MOC rules"],
                "should_resolve": ["dependency between segment type and applicable MOC/gradient rules"],
            },
            "Shorter segments require steeper gradients; gentler gradients require more distance.",
            valid_constraint_cells=[
                {
                    "cell_id": "AVI_OPT_07_cell_low_gradient",
                    "description": "Low-gradient profile: gentler descent requires a longer intermediate segment.",
                    "constraints": [
                        cons("P1", "descent_gradient_percent >= 2.8", "scenario_vertical_profile", "cell_gradient_lower"),
                        cons("P2", "descent_gradient_percent <= 3.5", "scenario_vertical_profile", "cell_gradient_upper"),
                        cons("P3", "segment_length_km >= 10.0", "scenario_vertical_profile", "piecewise_length_lower_bound"),
                    ],
                },
                {
                    "cell_id": "AVI_OPT_07_cell_medium_gradient",
                    "description": "Medium-gradient profile.",
                    "constraints": [
                        cons("P1", "descent_gradient_percent >= 3.5", "scenario_vertical_profile", "cell_gradient_lower"),
                        cons("P2", "descent_gradient_percent <= 4.2", "scenario_vertical_profile", "cell_gradient_upper"),
                        cons("P3", "segment_length_km >= 8.0", "scenario_vertical_profile", "piecewise_length_lower_bound"),
                    ],
                },
                {
                    "cell_id": "AVI_OPT_07_cell_high_gradient",
                    "description": "High-gradient profile: shorter segment is possible but remains under the KG-derived gradient limit.",
                    "constraints": [
                        cons("P1", "descent_gradient_percent >= 4.2", "scenario_vertical_profile", "cell_gradient_lower"),
                        cons("P2", "descent_gradient_percent <= 5.0", "rule:intermediate_approach_gradient_max", "cell_gradient_upper"),
                        cons("P3", "segment_length_km >= 6.7", "scenario_vertical_profile", "piecewise_length_lower_bound"),
                    ],
                },
            ],
        ),
        case(
            "AVI_OPT_08",
            "Non-helicopter PBN intermediate segment stability",
            "pbn_intermediate_segment_design",
            "Choose the intermediate segment length for a non-helicopter PBN approach. The compiler must activate the non-helicopter PBN branch rather than the helicopter branch, while the optimizer trades off a shorter procedure footprint against stabilization margin before final approach.",
            {
                "procedure_type": "pbn",
                "aircraft_category": "C",
                "navigation_system": "rnav",
            },
            {
                "intermediate_segment_length_km": {"type": "continuous", "unit": "km", "lower": 4, "upper": 14},
                "length_deviation_from_recommendation_km": {"type": "continuous", "unit": "km", "lower": 0, "upper": 6},
                "stabilization_margin_km": {"type": "continuous", "unit": "km", "lower": 0, "upper": 7},
            },
            [
                {"name": "minimize_segment_length", "expression": "intermediate_segment_length_km"},
                {"name": "maximize_stabilization_margin", "expression": "stabilization_margin_km"},
            ],
            {"lambda": [0.45, 0.55], "meaning": "slightly prioritize final-approach stabilization over a shorter segment"},
            ["pbn_intermediate_segment_optimum_length", "pbn_intermediate_segment_stability_requirement"],
            [
                cons("C0", "minimum stabilization length is grounded as 7.0 km", "scenario_stability_model", "hidden_stability_parameter"),
                cons("C1", "intermediate_segment_length_km >= 7.0", "rule:pbn_intermediate_segment_stability_requirement", "stability_lower_bound"),
                cons("C2", "stabilization_margin_km = intermediate_segment_length_km - 7.0", "rule:pbn_intermediate_segment_stability_requirement", "stability_margin_definition"),
                cons("C3", "length_deviation_from_recommendation_km >= abs(intermediate_segment_length_km - 9.0)", "rule:pbn_intermediate_segment_optimum_length", "recommended_length_trace"),
                cons("C4", "aircraft_category != H", "scenario_branch_guard", "non_helicopter_branch_guard"),
            ],
            {
                "should_activate": ["non-helicopter PBN intermediate-segment rule"],
                "should_exclude": ["helicopter-specific intermediate-segment rule"],
                "should_resolve": ["mutually exclusive aircraft-category branches"],
            },
            "Unlike the previous version, the 9 km value is a traceable recommendation/target, not a hard equality that eliminates optimization.",
        ),
        case(
            "AVI_OPT_09",
            "Helicopter PBN intermediate segment with compact-route pressure",
            "pbn_intermediate_segment_design",
            "Choose the intermediate segment length for a helicopter PBN approach near constrained airspace. The compiler must activate the helicopter branch and exclude the non-helicopter branch, while the optimizer trades off compact route design against helicopter stabilization margin.",
            {
                "procedure_type": "pbn",
                "aircraft_category": "H",
                "navigation_system": "rnav",
                "nearby_restricted_airspace_distance_km": 5.0,
            },
            {
                "intermediate_segment_length_km": {"type": "continuous", "unit": "km", "lower": 1.0, "upper": 8.0},
                "restricted_airspace_margin_km": {"type": "continuous", "unit": "km", "lower": 0, "upper": 2.5},
                "stabilization_margin_km": {"type": "continuous", "unit": "km", "lower": 0, "upper": 3.2},
            },
            [
                {"name": "maximize_restricted_airspace_margin", "expression": "restricted_airspace_margin_km"},
                {"name": "maximize_stabilization_margin", "expression": "stabilization_margin_km"},
            ],
            {"lambda": [0.5, 0.5], "meaning": "balance avoiding restricted airspace against giving the helicopter time to stabilize"},
            ["pbn_intermediate_segment_optimum_length_helicopter", "pbn_intermediate_segment_stability_requirement"],
            [
                cons("C0", "minimum stabilization length is grounded as 2.8 km", "scenario_stability_model", "hidden_helicopter_stability_parameter"),
                cons("C1", "intermediate_segment_length_km >= 2.8", "rule:pbn_intermediate_segment_stability_requirement", "helicopter_stability_lower_bound"),
                cons("C2", "stabilization_margin_km = intermediate_segment_length_km - 2.8", "rule:pbn_intermediate_segment_stability_requirement", "stabilization_margin"),
                cons("C3", "restricted_airspace_margin_km = nearby_restricted_airspace_distance_km - intermediate_segment_length_km", "scenario_airspace_model", "airspace_margin"),
                cons("C4", "aircraft_category = H", "scenario_branch_guard", "helicopter_branch_guard"),
            ],
            {
                "should_activate": ["helicopter PBN intermediate-segment rule"],
                "should_exclude": ["non-helicopter PBN intermediate-segment rule"],
                "should_resolve": ["mutually exclusive aircraft-category branches"],
            },
            "Longer segments help stabilization but reduce margin to nearby restricted airspace.",
        ),
        case(
            "AVI_OPT_10",
            "GBAS intermediate segment distance with capture margin",
            "gbas_intermediate_segment_design",
            "Choose the GBAS intermediate-segment distance from LTP. The distance must respect the KG-derived maximum GBAS distance, while the optimizer trades off compact approach geometry against additional capture/stabilization distance before the final segment.",
            {
                "navigation_system": "gbas",
                "segment_type": "intermediate approach segment",
                "ltp_available": True,
            },
            {
                "distance_from_ltp_km": {"type": "continuous", "unit": "km", "lower": 10, "upper": 45},
                "capture_margin_km": {"type": "continuous", "unit": "km", "lower": 0, "upper": 19},
                "service_volume_margin_km": {"type": "continuous", "unit": "km", "lower": 0, "upper": 23},
            },
            [
                {"name": "minimize_distance_from_ltp", "expression": "distance_from_ltp_km"},
                {"name": "maximize_capture_margin", "expression": "capture_margin_km"},
            ],
            {"lambda": [0.5, 0.5], "meaning": "balance compact GBAS geometry with capture/stabilization margin"},
            ["pbn_intermediate_segment_max_length_gbas", "pbn_intermediate_segment_stability_requirement"],
            [
                cons("C0", "minimum capture distance is grounded as 18.0 km", "scenario_capture_model", "hidden_capture_distance_parameter"),
                cons("C1", "distance_from_ltp_km <= 37", "rule:pbn_intermediate_segment_max_length_gbas", "gbas_maximum_distance"),
                cons("C2", "distance_from_ltp_km >= 18.0", "rule:pbn_intermediate_segment_stability_requirement", "capture_distance_lower_bound"),
                cons("C3", "capture_margin_km = distance_from_ltp_km - 18.0", "rule:pbn_intermediate_segment_stability_requirement", "capture_margin"),
                cons("C4", "service_volume_margin_km = 37 - distance_from_ltp_km", "rule:pbn_intermediate_segment_max_length_gbas", "service_volume_margin_certificate"),
            ],
            {
                "should_activate": ["GBAS maximum-distance rule", "stability requirement"],
                "should_exclude": ["non-GBAS intermediate distance rules"],
                "should_resolve": ["dependency from navigation system to GBAS-specific distance bound"],
            },
            "A short distance is compact, but too short reduces capture/stabilization margin.",
        ),
        case(
            "AVI_OPT_11",
            "Intermediate approach altitude over controlling obstacle",
            "intermediate_approach_design",
            "Choose the minimum altitude for an intermediate approach segment over a known obstacle. The altitude must include the KG-derived MOC above the highest obstacle, while the optimizer trades off lower procedure altitude against additional obstacle-clearance margin.",
            {
                "segment_type": "intermediate approach segment",
                "max_obstacle_elevation_m": 620,
                "airport_elevation_m": 180,
                "preferred_low_altitude_m": 760,
            },
            {
                "minimum_clearance_altitude_m": {"type": "continuous", "unit": "m", "lower": 700, "upper": 900},
                "altitude_margin_m": {"type": "continuous", "unit": "m", "lower": 0, "upper": 280},
            },
            [
                {"name": "minimize_minimum_altitude", "expression": "minimum_clearance_altitude_m"},
                {"name": "maximize_altitude_margin", "expression": "altitude_margin_m"},
            ],
            {"lambda": [0.55, 0.45], "meaning": "prefer a low procedure altitude while retaining obstacle-clearance margin"},
            ["moc_intermediate_approach_segment_7.4.3.8", "moc_intermediate_approach_min"],
            [
                cons("C1", "minimum_clearance_altitude_m >= max_obstacle_elevation_m + 150", "rule:moc_intermediate_approach_segment_7.4.3.8", "intermediate_moca_formula"),
                cons("C2", "altitude_margin_m = minimum_clearance_altitude_m - max_obstacle_elevation_m", "rule:moc_intermediate_approach_min", "moc_margin_certificate"),
                cons("C3", "altitude_margin_m >= 150", "rule:moc_intermediate_approach_min", "minimum_moc"),
            ],
            {
                "should_activate": ["intermediate approach MOC formula", "minimum MOC rule"],
                "should_exclude": ["missed-approach primary-area MOC"],
                "should_resolve": ["dependency from segment type to the correct MOC family"],
            },
            "Lower altitude improves efficiency, but greater altitude improves clearance.",
        ),
        case(
            "AVI_OPT_12",
            "PAOAS climb-before-turn maneuver design",
            "parallel_approach_operation_design",
            "Design PAOAS protected maneuver parameters for a required heading change. The maneuver must satisfy the KG-derived obstacle-clearance start-height and turn-angle requirements, while the optimizer trades off lower climb-before-turn height against a smaller protected turn angle.",
            {
                "operation_type": "parallel_approach_operations",
                "oas_available": True,
                "planned_heading_change_deg": 32,
                "terrain_close_to_turn_area": True,
            },
            {
                "obstacle_clearance_start_height_m": {"type": "continuous", "unit": "m", "lower": 0, "upper": 250},
                "protected_turn_angle_deg": {"type": "continuous", "unit": "degree", "lower": 0, "upper": 60},
                "maneuver_complexity_score": {"type": "continuous", "unit": "score", "lower": 0, "upper": 30},
            },
            [
                {"name": "minimize_start_height", "expression": "obstacle_clearance_start_height_m"},
                {"name": "minimize_protected_turn_angle", "expression": "protected_turn_angle_deg"},
            ],
            {"lambda": [0.45, 0.55], "meaning": "slightly prefer simpler turning while avoiding excessive climb height"},
            ["paoas_protection_scope"],
            [
                cons("C1", "obstacle_clearance_start_height_m >= 120", "rule:paoas_protection_scope", "minimum_protected_maneuver_height"),
                cons("C2", "protected_turn_angle_deg <= 45", "rule:paoas_protection_scope", "maximum_protected_turn_angle"),
                cons("C3", "obstacle_clearance_start_height_m >= 120 + 2.0 * (planned_heading_change_deg - protected_turn_angle_deg)", "scenario_maneuver_model", "height_angle_tradeoff"),
                cons("C4", "maneuver_complexity_score = protected_turn_angle_deg + 0.1 * obstacle_clearance_start_height_m", "scenario_maneuver_model", "complexity_proxy"),
            ],
            {
                "should_activate": ["PAOAS protected maneuver rule"],
                "should_exclude": ["ordinary approach maneuver rules"],
                "should_resolve": ["dependency from PAOAS operation type to obstacle-clearance and turn-angle clauses"],
            },
            "Reducing the protected turn angle requires more climb-before-turn height in this scenario model.",
        ),
        case(
            "AVI_OPT_13",
            "Turning missed-approach obstacle avoidance",
            "missed_approach_design",
            "Design a turning missed-approach procedure around an obstacle sector. The compiler must activate the turning missed-approach branch and exclude the straight branch, while the optimizer trades off a smaller turn angle against larger lateral obstacle-avoidance margin.",
            {
                "procedure_type": "missed_approach",
                "missed_approach_type": "turning",
                "final_approach_track_deg": 85,
                "obstacle_sector_bearing_deg": 103,
            },
            {
                "turn_angle_deg": {"type": "continuous", "unit": "degree", "lower": 0, "upper": 45},
                "lateral_obstacle_margin_deg": {"type": "continuous", "unit": "degree", "lower": 0, "upper": 23},
                "track_deviation_deg": {"type": "continuous", "unit": "degree", "lower": 0, "upper": 45},
            },
            [
                {"name": "minimize_turn_angle", "expression": "turn_angle_deg"},
                {"name": "maximize_obstacle_avoidance_margin", "expression": "lateral_obstacle_margin_deg"},
            ],
            {"lambda": [0.5, 0.5], "meaning": "balance a simple missed-approach turn with obstacle-avoidance margin"},
            ["RA-6.4.1-turn-angle-threshold", "RA-6.3.4-align-final-approach-track", "tnh_obstacle_clearance_requirement"],
            [
                cons("C0", "minimum obstacle-avoidance turn is grounded as 22 deg", "scenario_obstacle_sector", "hidden_obstacle_sector_clearance_angle"),
                cons("C1", "turn_angle_deg > 15", "rule:RA-6.4.1-turn-angle-threshold", "turning_branch_activation"),
                cons("C2", "turn_angle_deg >= 22", "rule:tnh_obstacle_clearance_requirement", "obstacle_avoidance_lower_bound"),
                cons("C3", "lateral_obstacle_margin_deg = turn_angle_deg - 22", "scenario_obstacle_sector", "avoidance_margin"),
                cons("C4", "track_deviation_deg = turn_angle_deg", "rule:RA-6.3.4-align-final-approach-track", "track_continuation_penalty"),
            ],
            {
                "should_activate": ["turning missed-approach threshold", "obstacle-clearance turn-height requirement"],
                "should_exclude": ["straight missed-approach continuation as the sole valid branch"],
                "should_resolve": ["mutual exclusion between straight and turning missed-approach structures"],
            },
            "The case is both an optimization problem and a branch-selection test.",
        ),
        case(
            "AVI_OPT_14",
            "Fixed-wing departure turn-initiation protection",
            "departure_turn_design",
            "Size the turn-initiation area for a fixed-wing departure with an obstacle near the DER side. The compiler must activate the DER-based fixed-wing rule rather than the FATO helicopter rule, while the optimizer trades off a compact protected area against obstacle-buffer margin.",
            {
                "aircraft_class": "fixed-wing",
                "departure_reference": "DER",
                "nearest_obstacle_lateral_offset_m": 175,
            },
            {
                "lateral_extent_from_der_m": {"type": "continuous", "unit": "m", "lower": 100, "upper": 300},
                "obstacle_buffer_m": {"type": "continuous", "unit": "m", "lower": 0, "upper": 85},
                "protected_area_size_score": {"type": "continuous", "unit": "score", "lower": 0, "upper": 110},
            },
            [
                {"name": "minimize_protected_area_size", "expression": "protected_area_size_score"},
                {"name": "maximize_obstacle_buffer", "expression": "obstacle_buffer_m"},
            ],
            {"lambda": [0.55, 0.45], "meaning": "prefer a compact DER-side area while preserving obstacle buffer"},
            ["turn_init_area_der_extent", "turn_init_area_fato_extent"],
            [
                cons("C0", "minimum extra buffer is grounded as 10 m", "scenario_obstacle_geometry", "hidden_project_buffer_requirement"),
                cons("C1", "lateral_extent_from_der_m >= 150", "rule:turn_init_area_der_extent", "fixed_wing_der_extent"),
                cons("C2", "obstacle_buffer_m = lateral_extent_from_der_m - nearest_obstacle_lateral_offset_m", "scenario_obstacle_geometry", "obstacle_buffer"),
                cons("C3", "obstacle_buffer_m >= 10", "scenario_obstacle_geometry", "minimum_extra_buffer"),
                cons("C4", "protected_area_size_score = lateral_extent_from_der_m - 150", "rule:turn_init_area_der_extent", "area_size_proxy"),
            ],
            {
                "should_activate": ["fixed-wing DER-side turn-initiation extent"],
                "should_exclude": ["helicopter FATO-side extent"],
                "should_resolve": ["mutually exclusive aircraft-class branches"],
            },
            "A larger protected area improves obstacle buffer but consumes more airspace.",
        ),
        case(
            "AVI_OPT_15",
            "Helicopter departure turn-initiation protection",
            "helicopter_departure_turn_design",
            "Size the turn-initiation area for a helicopter departure with FATO-side clutter. The compiler must activate the helicopter/FATO rule and exclude the fixed-wing DER rule, while the optimizer trades off compact protected area against margin to nearby clutter.",
            {
                "aircraft_class": "class h aircraft",
                "departure_reference": "FATO",
                "nearest_clutter_lateral_offset_m": 58,
            },
            {
                "lateral_extent_from_fato_m": {"type": "continuous", "unit": "m", "lower": 20, "upper": 120},
                "clutter_buffer_m": {"type": "continuous", "unit": "m", "lower": 0, "upper": 42},
                "protected_area_size_score": {"type": "continuous", "unit": "score", "lower": 0, "upper": 55},
            },
            [
                {"name": "minimize_protected_area_size", "expression": "protected_area_size_score"},
                {"name": "maximize_clutter_buffer", "expression": "clutter_buffer_m"},
            ],
            {"lambda": [0.55, 0.45], "meaning": "prefer compact helicopter protection while retaining clutter margin"},
            ["turn_init_area_fato_extent", "turn_init_area_der_extent"],
            [
                cons("C0", "minimum extra buffer is grounded as 6 m", "scenario_clutter_geometry", "hidden_project_clutter_buffer_requirement"),
                cons("C1", "lateral_extent_from_fato_m >= 45", "rule:turn_init_area_fato_extent", "helicopter_fato_extent"),
                cons("C2", "clutter_buffer_m = lateral_extent_from_fato_m - nearest_clutter_lateral_offset_m", "scenario_clutter_geometry", "clutter_buffer"),
                cons("C3", "clutter_buffer_m >= 6", "scenario_clutter_geometry", "minimum_clutter_buffer"),
                cons("C4", "protected_area_size_score = lateral_extent_from_fato_m - 45", "rule:turn_init_area_fato_extent", "area_size_proxy"),
            ],
            {
                "should_activate": ["helicopter FATO-side turn-initiation extent"],
                "should_exclude": ["fixed-wing DER-side extent"],
                "should_resolve": ["mutually exclusive aircraft-class branches"],
            },
            "This is the helicopter counterpart of the fixed-wing case, with different valid rule structure and provenance.",
        ),
        case(
            "AVI_OPT_16",
            "Missed-approach primary-area altitude and MOC",
            "missed_approach_design",
            "Design the primary-area obstacle-clearance setting for a missed-approach segment. The procedure altitude must satisfy the KG-derived primary-area MOC, while the optimizer trades off lower procedure altitude against larger obstacle-clearance margin.",
            {
                "segment_type": "missed approach",
                "area_type": "primary",
                "highest_obstacle_elevation_m": 510,
                "preferred_low_procedure_altitude_m": 590,
            },
            {
                "minimum_obstacle_clearance_m": {"type": "continuous", "unit": "m", "lower": 0, "upper": 180},
                "procedure_altitude_m": {"type": "continuous", "unit": "m", "lower": 500, "upper": 760},
                "obstacle_clearance_margin_m": {"type": "continuous", "unit": "m", "lower": 0, "upper": 250},
            },
            [
                {"name": "minimize_procedure_altitude", "expression": "procedure_altitude_m"},
                {"name": "maximize_obstacle_clearance_margin", "expression": "obstacle_clearance_margin_m"},
            ],
            {"lambda": [0.5, 0.5], "meaning": "balance low missed-approach altitude against obstacle clearance"},
            ["moc_primary_area_missed_approach", "moc_secondary_area_linear_decay"],
            [
                cons("C1", "minimum_obstacle_clearance_m >= 75", "rule:moc_primary_area_missed_approach", "primary_area_moc"),
                cons("C2", "procedure_altitude_m >= highest_obstacle_elevation_m + minimum_obstacle_clearance_m", "rule:moc_primary_area_missed_approach", "procedure_altitude_lower_bound"),
                cons("C3", "obstacle_clearance_margin_m = procedure_altitude_m - highest_obstacle_elevation_m", "scenario_obstacle_geometry", "actual_clearance_margin"),
                cons("C4", "area_type = primary", "scenario_area_guard", "primary_area_branch_guard"),
            ],
            {
                "should_activate": ["primary-area missed-approach MOC"],
                "should_exclude": ["secondary-area linear-decay MOC"],
                "should_resolve": ["mutually exclusive primary/secondary area rule structures"],
            },
            "This case checks that primary-area MOC is not flattened with secondary-area decay rules.",
        ),
        case(
            "AVI_OPT_17",
            "RF segment bank angle and turn radius at FL180",
            "pbn_rf_segment_design",
            "Choose the bank angle and turn radius for a PBN RF segment at FL180. The bank angle must satisfy the KG-derived design limit, while the optimizer trades off a compact turn radius against lower bank angle and lower passenger/aircraft load.",
            {
                "procedure_type": "pbn",
                "segment_type": "rf_segment",
                "flight_level": "FL180",
                "true_airspeed_kmh": 330,
                "wind_speed_kmh": 30,
                "rnp_value_km": 1.0,
            },
            {
                "bank_angle_deg": {"type": "continuous", "unit": "degree", "lower": 5, "upper": 30},
                "turn_radius_km": {"type": "continuous", "unit": "km", "lower": 0, "upper": 12},
                "turn_load_score": {"type": "continuous", "unit": "score", "lower": 0, "upper": 30},
            },
            [
                {"name": "minimize_turn_radius", "expression": "turn_radius_km"},
                {"name": "minimize_turn_load", "expression": "turn_load_score"},
            ],
            {"lambda": [0.55, 0.45], "meaning": "prefer compact turns while avoiding excessive bank/load"},
            ["rf_segment_max_bank_angle_25deg", "turn_radius_km_formula", "minimum_turn_radius_rnp_constraint"],
            [
                cons("C1", "bank_angle_deg <= 25", "rule:rf_segment_max_bank_angle_25deg", "below_fl190_bank_limit"),
                cons("C2", "turn_radius_km >= 2 * rnp_value_km", "rule:minimum_turn_radius_rnp_constraint", "rnp_minimum_radius"),
                cons("C3", "bank-radius coupling is represented by three piecewise-linear turn-geometry cells", "rule:turn_radius_km_formula", "piecewise_linear_bank_radius_tradeoff"),
                cons("C4", "turn_load_score = bank_angle_deg", "scenario_turn_load_proxy", "bank_load_proxy"),
            ],
            {
                "should_activate": ["RF segment 25-degree bank limit", "turn-radius formula", "RNP radius constraint"],
                "should_exclude": ["above-FL190 15-degree exception"],
                "should_resolve": ["altitude-conditioned applicability below FL190"],
            },
            "A smaller radius requires a larger bank angle; the 25-degree rule caps the feasible compactness.",
            valid_constraint_cells=[
                {
                    "cell_id": "AVI_OPT_17_cell_low_bank",
                    "description": "Low-bank RF turn: lower load requires a larger turn radius.",
                    "constraints": [
                        cons("P1", "bank_angle_deg >= 8", "scenario_turn_geometry", "cell_bank_lower"),
                        cons("P2", "bank_angle_deg <= 12", "scenario_turn_geometry", "cell_bank_upper"),
                        cons("P3", "turn_radius_km >= 6.75", "rule:turn_radius_km_formula", "piecewise_radius_lower_bound"),
                    ],
                },
                {
                    "cell_id": "AVI_OPT_17_cell_medium_bank",
                    "description": "Medium-bank RF turn.",
                    "constraints": [
                        cons("P1", "bank_angle_deg >= 12", "scenario_turn_geometry", "cell_bank_lower"),
                        cons("P2", "bank_angle_deg <= 18", "scenario_turn_geometry", "cell_bank_upper"),
                        cons("P3", "turn_radius_km >= 4.5", "rule:turn_radius_km_formula", "piecewise_radius_lower_bound"),
                    ],
                },
                {
                    "cell_id": "AVI_OPT_17_cell_high_bank",
                    "description": "High-bank RF turn under the ordinary FL180 bank-angle rule.",
                    "constraints": [
                        cons("P1", "bank_angle_deg >= 18", "scenario_turn_geometry", "cell_bank_lower"),
                        cons("P2", "bank_angle_deg <= 25", "rule:rf_segment_max_bank_angle_25deg", "cell_bank_upper"),
                        cons("P3", "turn_radius_km >= 3.0", "rule:turn_radius_km_formula", "piecewise_radius_lower_bound"),
                    ],
                },
            ],
        ),
        case(
            "AVI_OPT_18",
            "High-altitude PBN turn under bank-angle exception",
            "pbn_turn_design",
            "Choose the bank angle and turn radius for a PBN turn at FL200. The compiler must apply the KG-derived high-altitude bank-angle exception instead of the ordinary RF rule, while the optimizer trades off compact turn geometry against lower bank/load.",
            {
                "procedure_type": "pbn",
                "segment_type": "turn",
                "flight_level": "FL200",
                "true_airspeed_kmh": 410,
                "wind_speed_kmh": 40,
                "rnp_value_km": 1.2,
            },
            {
                "bank_angle_deg": {"type": "continuous", "unit": "degree", "lower": 5, "upper": 30},
                "turn_radius_km": {"type": "continuous", "unit": "km", "lower": 0, "upper": 18},
                "turn_load_score": {"type": "continuous", "unit": "score", "lower": 0, "upper": 30},
            },
            [
                {"name": "minimize_turn_radius", "expression": "turn_radius_km"},
                {"name": "minimize_turn_load", "expression": "turn_load_score"},
            ],
            {"lambda": [0.5, 0.5], "meaning": "balance compact high-altitude turn geometry and lower turn load"},
            ["rf_segment_bank_angle_15deg_above_fl190", "rf_segment_max_bank_angle_25deg", "turn_radius_km_formula", "minimum_turn_radius_rnp_constraint"],
            [
                cons("C1", "bank_angle_deg <= 15", "rule:rf_segment_bank_angle_15deg_above_fl190", "high_altitude_exception_bank_limit"),
                cons("C2", "bank_angle_deg <= 25 is defeated by high-altitude exception", "rule:rf_segment_max_bank_angle_25deg", "defeated_baseline_rule"),
                cons("C3", "turn_radius_km >= 2 * rnp_value_km", "rule:minimum_turn_radius_rnp_constraint", "rnp_minimum_radius"),
                cons("C4", "bank-radius coupling is represented by three piecewise-linear high-altitude turn cells", "rule:turn_radius_km_formula", "piecewise_linear_high_altitude_bank_radius_tradeoff"),
                cons("C5", "turn_load_score = bank_angle_deg", "scenario_turn_load_proxy", "bank_load_proxy"),
            ],
            {
                "should_activate": ["above-FL190 15-degree bank exception"],
                "should_exclude": ["ordinary 25-degree RF bank limit as governing rule"],
                "should_resolve": ["exception override between ordinary RF rule and high-altitude turn rule"],
            },
            "This is an optimization case and an exception-override test: flat compilation may keep the wrong 25-degree feasible region.",
            valid_constraint_cells=[
                {
                    "cell_id": "AVI_OPT_18_cell_low_bank",
                    "description": "Low-bank high-altitude turn under the 15-degree exception.",
                    "constraints": [
                        cons("P1", "bank_angle_deg >= 8", "scenario_turn_geometry", "cell_bank_lower"),
                        cons("P2", "bank_angle_deg <= 10", "scenario_turn_geometry", "cell_bank_upper"),
                        cons("P3", "turn_radius_km >= 9.75", "rule:turn_radius_km_formula", "piecewise_radius_lower_bound"),
                    ],
                },
                {
                    "cell_id": "AVI_OPT_18_cell_medium_bank",
                    "description": "Medium-bank high-altitude turn under the 15-degree exception.",
                    "constraints": [
                        cons("P1", "bank_angle_deg >= 10", "scenario_turn_geometry", "cell_bank_lower"),
                        cons("P2", "bank_angle_deg <= 12.5", "scenario_turn_geometry", "cell_bank_upper"),
                        cons("P3", "turn_radius_km >= 7.8", "rule:turn_radius_km_formula", "piecewise_radius_lower_bound"),
                    ],
                },
                {
                    "cell_id": "AVI_OPT_18_cell_high_bank",
                    "description": "Highest admissible bank-angle cell after the 25-degree baseline is defeated.",
                    "constraints": [
                        cons("P1", "bank_angle_deg >= 12.5", "scenario_turn_geometry", "cell_bank_lower"),
                        cons("P2", "bank_angle_deg <= 15", "rule:rf_segment_bank_angle_15deg_above_fl190", "cell_bank_upper"),
                        cons("P3", "turn_radius_km >= 6.24", "rule:turn_radius_km_formula", "piecewise_radius_lower_bound"),
                    ],
                },
            ],
        ),
        case(
            "AVI_OPT_19",
            "RF segment length from turn anticipation",
            "pbn_rf_segment_design",
            "Choose the length of an RF segment constrained by turn anticipation. The segment must be no shorter than the KG-derived total DTA requirement, while the optimizer trades off a compact RF segment against additional length margin for execution robustness.",
            {
                "procedure_type": "pbn",
                "segment_type": "rf_segment",
                "turn_type": "fly-by_turn",
                "turn_angle_deg": 38,
                "turn_radius_km": 4.2,
            },
            {
                "segment_length_km": {"type": "continuous", "unit": "km", "lower": 2, "upper": 9},
                "length_margin_km": {"type": "continuous", "unit": "km", "lower": 0, "upper": 6.0},
            },
            [
                {"name": "minimize_segment_length", "expression": "segment_length_km"},
                {"name": "maximize_length_margin", "expression": "length_margin_km"},
            ],
            {"lambda": [0.55, 0.45], "meaning": "prefer a compact RF segment while keeping turn-anticipation margin"},
            ["dta_formula", "dta_minimum_segment_length_requirement"],
            [
                cons("C1", "segment_length_km >= 2.9", "rule:dta_minimum_segment_length_requirement", "minimum_segment_length_from_dta"),
                cons("C2", "total DTA is grounded as 2.9 km from 2 * turn_radius_km * tan(turn_angle_deg / 2)", "rule:dta_formula", "grounded_total_dta"),
                cons("C3", "length_margin_km = segment_length_km - 2.9", "rule:dta_minimum_segment_length_requirement", "dta_margin_certificate"),
            ],
            {
                "should_activate": ["DTA formula", "minimum segment length requirement"],
                "should_exclude": ["non-RF segment length rules"],
                "should_resolve": ["formula dependency from turn anticipation to segment-length lower bound"],
            },
            "Shorter segment length conflicts directly with larger DTA margin.",
        ),
        case(
            "AVI_OPT_20",
            "SBAS FAS channel assignment with interference separation",
            "sbas_publication_design",
            "Choose an SBAS FAS data-block channel for publication near an occupied-channel cluster. The channel must satisfy the KG-derived valid FAS channel range, while the optimizer trades off closeness to the preferred center channel against separation from nearby occupied channels.",
            {
                "navigation_system": "satellite-based augmentation system (sbas)",
                "publication_context": "FAS data block",
                "preferred_channel_center": 65000,
                "nearest_occupied_channel": 65020,
            },
            {
                "channel_number": {"type": "continuous", "unit": "unitless", "lower": 30000, "upper": 110000},
                "channel_deviation": {"type": "continuous", "unit": "channel", "lower": 0, "upper": 59999},
                "occupied_channel_separation": {"type": "continuous", "unit": "channel", "lower": 0, "upper": 59999},
            },
            [
                {"name": "minimize_preferred_channel_deviation", "expression": "channel_deviation"},
                {"name": "maximize_occupied_channel_separation", "expression": "occupied_channel_separation"},
            ],
            {"lambda": [0.55, 0.45], "meaning": "prefer the planned center channel while retaining separation from occupied channels"},
            ["sbas_fas_db_publication_requirements_5.8.7"],
            [
                cons("C0", "minimum channel separation is grounded as 35", "scenario_channel_planning", "hidden_interference_separation_requirement"),
                cons("C1", "channel_number >= 40000", "rule:sbas_fas_db_publication_requirements_5.8.7", "minimum_fas_channel"),
                cons("C2", "channel_number <= 99999", "rule:sbas_fas_db_publication_requirements_5.8.7", "maximum_fas_channel"),
                cons("C3", "channel_deviation >= abs(channel_number - preferred_channel_center)", "scenario_channel_planning", "preferred_center_deviation"),
                cons("C4", "occupied_channel_separation <= abs(channel_number - nearest_occupied_channel)", "scenario_channel_planning", "occupied_channel_separation_proxy"),
                cons("C5", "occupied_channel_separation >= 35", "scenario_channel_planning", "minimum_interference_separation"),
            ],
            {
                "should_activate": ["SBAS FAS data-block channel range"],
                "should_exclude": ["non-SBAS chart-title rules"],
                "should_resolve": ["document-level provenance for FAS channel bounds"],
            },
            "Auditability remains an output metric; it is no longer incorrectly used as an optimization objective.",
        ),
    ]


def main() -> None:
    rule_index = load_rule_index()
    problems = build_cases()
    for item in problems:
        source_rule_ids = item["hidden_evaluation_reference"]["source_rule_ids"]
        item["hidden_evaluation_reference"]["evidence"] = evidence_for(source_rule_ids, rule_index)
    missing = sorted(
        {
            rid
            for item in problems
            for rid in item["hidden_evaluation_reference"]["source_rule_ids"]
            if rid not in rule_index
        }
    )
    payload = {
        "version": "aviation_kg_optimization_problems_v2",
        "generated_from": str(RULE_LIBRARY),
        "purpose": (
            "Nineteen KG-grounded aviation decision optimization problems for testing CTHR symbolic modeling, "
            "valid rule structure construction, solver-facing constraints, and certificate provenance."
        ),
        "construction_policy": [
            "Each case is a concrete engineering optimization task, not a rule-activation prompt.",
            "Each case contains at least two engineering objectives whose preferred directions conflict under the stated constraints.",
            "Each case contains a rule-structure challenge: applicability, dependency, branch exclusion, exception override, formula propagation, or document provenance.",
            "Visible decision queries contain engineering scenario facts, design variables, objectives, and preference weights, but do not reveal the regulatory answer constraints.",
            "Regulatory constraints are stored only in hidden_evaluation_reference and must be recovered from the KG-derived rule library.",
            "Source-rule thresholds and KG-derived parameters are kept out of visible scenario_facts and are enforced only through hidden KG-grounded constraints.",
            "Objective functions and scenario constants are task inputs, not claimed as source regulations.",
            "Formula rules are instantiated as solver-facing constraints only after grounding the scenario.",
        ],
        "num_problems": len(problems),
        "missing_source_rule_ids": missing,
        "problems": problems,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Aviation KG-Generated Optimization Problems",
        "",
        f"- Source rule library: `{RULE_LIBRARY}`",
        f"- Number of problems: {len(problems)}",
        f"- Missing source rule ids: {missing or 'none'}",
        "",
        "| ID | Task | Source rules | Solver constraints | Evidence chunks |",
        "|---|---|---:|---:|---:|",
    ]
    for item in problems:
        lines.append(
            "| {id} | {title} | {rules} | {constraints} | {chunks} |".format(
                id=item["omega_id"],
                title=item["title"],
                rules=len(item["hidden_evaluation_reference"]["source_rule_ids"]),
                constraints=len(item["hidden_evaluation_reference"]["kg_grounded_constraints"]),
                chunks=len(item["hidden_evaluation_reference"]["evidence"]["kg_chunk_ids"]),
            )
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out_json": str(OUT_JSON), "out_md": str(OUT_MD), "missing": missing}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
