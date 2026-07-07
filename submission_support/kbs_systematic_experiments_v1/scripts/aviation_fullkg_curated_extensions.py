from __future__ import annotations

from typing import Any


REFERENCE_SEMANTICS = {
    "positive_membership_condition": (
        "all executable_constraints evaluate true and, when valid_constraint_cells are present, "
        "at least one cell's executable constraints also evaluate true"
    ),
    "structure_only_constraints_usage": (
        "used to check rule resolution, provenance, or specialized encoders before numeric membership checking"
    ),
}


def constraint(
    constraint_id: str,
    expression: str,
    role: str,
    source_id: str,
    decision_variables: list[str] | None = None,
    scenario_fields: list[str] | None = None,
    source_type: str = "rule_library",
    executable: bool = True,
    reason_not_executable: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "constraint_id": constraint_id,
        "expression": expression,
        "role": role,
        "source_type": source_type,
        "source_id": source_id,
        "executable": executable,
    }
    if executable:
        out.update(
            {
                "checker_expression": expression,
                "expression_language": "python_safe_arithmetic_predicate",
                "symbols": {
                    "decision_variables": decision_variables or [],
                    "scenario_fields": scenario_fields or [],
                    "unresolved_symbols": [],
                },
            }
        )
    else:
        out["reason_not_executable"] = reason_not_executable or "not represented as a direct numeric predicate"
    if metadata:
        out["metadata"] = metadata
    return out


def text_rule_proxy(rule_id: str, note: str) -> dict[str, Any]:
    return {
        "derived_from_text_rule": True,
        "derived_from_rule_id": rule_id,
        "derivation_note": note,
    }


def equality_parameter_boundary(rule_id: str, interpretation: str) -> dict[str, Any]:
    return {
        "normative_parameter_interpretation": "extracted_equality_used_as_design_boundary",
        "derived_from_rule_id": rule_id,
        "interpretation_note": interpretation,
    }


def structure_constraint(
    constraint_id: str,
    expression: str,
    role: str,
    source_id: str,
    source_type: str = "rule_library",
) -> dict[str, Any]:
    return constraint(
        constraint_id=constraint_id,
        expression=expression,
        role=role,
        source_id=source_id,
        source_type=source_type,
        executable=False,
    )


def rule_provenance(rule_lookup: dict[str, dict[str, Any]], rule_ids: list[str]) -> dict[str, Any]:
    chunk_ids: set[str] = set()
    node_ids: set[str] = set()
    edge_ids: set[str] = set()
    documents: list[dict[str, Any]] = []

    for rule_id in rule_ids:
        rule = rule_lookup[rule_id]
        chunk_ids.update(str(value) for value in rule.get("source_chunk_ids", []) if value)
        node_ids.update(str(value) for value in rule.get("source_node_ids", []) if value)
        for item in rule.get("provenance", []):
            if isinstance(item, dict):
                documents.append(item)
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


def make_label(
    rule_lookup: dict[str, dict[str, Any]],
    omega_id: str,
    title: str,
    scenario_facts: dict[str, Any],
    source_rule_ids: list[str],
    should_activate: list[str],
    should_exclude: list[str],
    should_resolve: list[str],
    challenge_types: list[str],
    defeated_rule_ids: list[str] | None = None,
) -> dict[str, Any]:
    defeated = defeated_rule_ids or []
    surviving = sorted(rule_id for rule_id in source_rule_ids if rule_id not in set(defeated))
    return {
        "omega_id": omega_id,
        "title": title,
        "scenario_facts": scenario_facts,
        "expected_source_rule_ids": source_rule_ids,
        "expected_defeated_rule_ids": defeated,
        "expected_surviving_rule_ids": surviving,
        "expected_rule_behavior": {
            "should_activate": should_activate,
            "should_exclude": should_exclude,
            "should_resolve": should_resolve,
        },
        "challenge_types": challenge_types,
        "valid_constraint_cell_ids": [],
        "expected_provenance": rule_provenance(rule_lookup, source_rule_ids),
    }


def make_record(
    rule_lookup: dict[str, dict[str, Any]],
    *,
    omega_id: str,
    title: str,
    task_type: str,
    design_intent: str,
    scenario_facts: dict[str, Any],
    decision_variables: dict[str, Any],
    objectives: list[dict[str, str]],
    query_preferences: dict[str, Any],
    source_rule_ids: list[str],
    executable_constraints: list[dict[str, Any]],
    structure_only_constraints: list[dict[str, Any]],
    should_activate: list[str],
    should_exclude: list[str],
    should_resolve: list[str],
    challenge_types: list[str],
    defeated_rule_ids: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    query = {
        "omega_id": omega_id,
        "title": title,
        "domain": "aviation_procedure_design",
        "task_type": task_type,
        "design_intent": design_intent,
        "scenario_facts": scenario_facts,
        "decision_variables": decision_variables,
        "objectives": objectives,
        "query_preferences": query_preferences,
        "solver_constraints": executable_constraints,
        "solver_constraint_cells": [],
        "pre_solver_structure_checks": structure_only_constraints,
        "certificate_targets": {
            "source_rule_ids": source_rule_ids,
            "provenance": rule_provenance(rule_lookup, source_rule_ids)["source_documents"],
        },
        "_split": "aviation_fullkg_curated_extensions11",
    }
    label = make_label(
        rule_lookup,
        omega_id,
        title,
        scenario_facts,
        source_rule_ids,
        should_activate,
        should_exclude,
        should_resolve,
        challenge_types,
        defeated_rule_ids,
    )
    feasible = {
        "omega_id": omega_id,
        "title": title,
        "scenario_facts": scenario_facts,
        "decision_variables": decision_variables,
        "executable_constraints": executable_constraints,
        "structure_only_constraints": structure_only_constraints,
        "valid_constraint_cells": [],
        "reference_semantics": REFERENCE_SEMANTICS,
    }
    return query, label, feasible


def build_curated_aviation_extensions(
    rule_lookup: dict[str, dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    records: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []

    records.append(
        make_record(
            rule_lookup,
            omega_id="AVI_OPT_06",
            title="Published descent gradient and angle rounding",
            task_type="approach_chart_publication_design",
            design_intent=(
                "Publish a charted descent gradient and descent angle for an intermediate approach segment. "
                "The design should preserve the computed vertical profile as closely as possible while keeping "
                "the charted gradient inside the KG-derived intermediate approach limit."
            ),
            scenario_facts={
                "publication_context": "charting",
                "segment_type": "intermediate approach segment",
                "computed_descent_gradient_percent": 4.37,
                "computed_descent_angle_deg": 2.51,
                "maximum_allowed_gradient_percent": 5.0,
            },
            decision_variables={
                "design_descent_gradient_percent": {
                    "type": "continuous",
                    "unit": "percent",
                    "lower": 3.8,
                    "upper": 5.0,
                },
                "design_descent_angle_deg": {
                    "type": "continuous",
                    "unit": "degree",
                    "lower": 2.0,
                    "upper": 3.0,
                },
                "published_descent_gradient_percent": {
                    "type": "continuous",
                    "unit": "percent",
                    "lower": 0.0,
                    "upper": 6.0,
                },
                "published_descent_angle_deg": {
                    "type": "continuous",
                    "unit": "degree",
                    "lower": 0.0,
                    "upper": 5.0,
                },
                "gradient_margin_percent": {
                    "type": "continuous",
                    "unit": "percent",
                    "lower": 0.0,
                    "upper": 5.0,
                },
                "rounding_error_score": {
                    "type": "continuous",
                    "unit": "score",
                    "lower": 0.0,
                    "upper": 1.0,
                },
            },
            objectives=[
                {"name": "minimize_chart_rounding_error", "expression": "rounding_error_score"},
                {"name": "maximize_gradient_limit_margin", "expression": "gradient_margin_percent"},
            ],
            query_preferences={
                "lambda": [0.55, 0.45],
                "meaning": "slightly prioritize publication fidelity while retaining descent-gradient margin",
            },
            source_rule_ids=["descent_gradient_rounding", "intermediate_approach_gradient_max"],
            executable_constraints=[
                constraint(
                    "C1",
                    "published_descent_gradient_percent >= design_descent_gradient_percent - 0.05",
                    "gradient_rounding_lower_envelope",
                    "descent_gradient_rounding",
                    ["published_descent_gradient_percent", "design_descent_gradient_percent"],
                ),
                constraint(
                    "C2",
                    "published_descent_gradient_percent <= design_descent_gradient_percent + 0.05",
                    "gradient_rounding_upper_envelope",
                    "descent_gradient_rounding",
                    ["published_descent_gradient_percent", "design_descent_gradient_percent"],
                ),
                constraint(
                    "C3",
                    "published_descent_angle_deg >= design_descent_angle_deg - 0.05",
                    "angle_rounding_lower_envelope",
                    "descent_gradient_rounding",
                    ["published_descent_angle_deg", "design_descent_angle_deg"],
                ),
                constraint(
                    "C4",
                    "published_descent_angle_deg <= design_descent_angle_deg + 0.05",
                    "angle_rounding_upper_envelope",
                    "descent_gradient_rounding",
                    ["published_descent_angle_deg", "design_descent_angle_deg"],
                ),
                constraint(
                    "C5",
                    "published_descent_gradient_percent <= 5.0",
                    "intermediate_gradient_limit",
                    "intermediate_approach_gradient_max",
                    ["published_descent_gradient_percent"],
                ),
                constraint(
                    "C6",
                    "abs(gradient_margin_percent - (5.0 - published_descent_gradient_percent)) <= 1e-6",
                    "gradient_margin_certificate",
                    "intermediate_approach_gradient_max",
                    ["gradient_margin_percent", "published_descent_gradient_percent"],
                ),
                constraint(
                    "C7",
                    "rounding_error_score >= abs(design_descent_gradient_percent - computed_descent_gradient_percent)",
                    "gradient_profile_deviation_proxy",
                    "scenario_charting_model",
                    ["rounding_error_score", "design_descent_gradient_percent"],
                    ["computed_descent_gradient_percent"],
                    source_type="task_or_scenario_model",
                ),
            ],
            structure_only_constraints=[],
            should_activate=["descent-gradient publication rounding", "intermediate-approach gradient maximum"],
            should_exclude=["holding timing rules", "PBN RF turn-radius rules"],
            should_resolve=["rounding is applied before checking the published gradient against the 5 percent limit"],
            challenge_types=["dependency_or_formula_propagation", "provenance_traceability"],
        )
    )

    records.append(
        make_record(
            rule_lookup,
            omega_id="AVI_OPT_21",
            title="RNAV SIA route with CDA and multi-airport service",
            task_type="standard_instrument_arrival_design",
            design_intent=(
                "Design a terminal RNAV arrival that connects an ATS route to the IAF, can serve two nearby "
                "airports, and supports a continuous-descent profile. The optimizer trades off compact route "
                "length, reduced radar-vector dependency, and more usable CDA/multi-airport service."
            ),
            scenario_facts={
                "procedure_type": "standard instrument arrival",
                "navigation_method": "rnav",
                "terminal_area_airports": 2,
                "ats_route_fix_available": True,
                "initial_approach_fix_available": True,
                "noise_sensitive_corridor": True,
            },
            decision_variables={
                "route_length_km": {"type": "continuous", "unit": "km", "lower": 25.0, "upper": 95.0},
                "cda_track_length_km": {"type": "continuous", "unit": "km", "lower": 0.0, "upper": 75.0},
                "radar_vector_dependency_score": {
                    "type": "continuous",
                    "unit": "score",
                    "lower": 0.0,
                    "upper": 1.0,
                },
                "served_airports_count": {"type": "integer", "unit": "count", "lower": 1, "upper": 2},
                "start_fix_defined_indicator": {"type": "binary", "unit": "indicator", "lower": 0, "upper": 1},
                "transition_to_iaf_indicator": {"type": "binary", "unit": "indicator", "lower": 0, "upper": 1},
            },
            objectives=[
                {"name": "minimize_route_length", "expression": "route_length_km"},
                {
                    "name": "maximize_cda_and_airport_service",
                    "expression": "cda_track_length_km + 8 * served_airports_count",
                },
                {"name": "minimize_radar_vector_dependency", "expression": "radar_vector_dependency_score"},
            ],
            query_preferences={
                "lambda": [0.35, 0.4, 0.25],
                "meaning": "prefer CDA and RNAV self-navigation, while avoiding an unnecessarily long arrival",
            },
            source_rule_ids=[
                "sia_start_point_requirement",
                "sia_transition_requirement",
                "sia_navigation_efficiency_requirement",
                "sia_multi_airport_service_capability",
                "sia_cda_consideration",
            ],
            executable_constraints=[
                constraint(
                    "C1",
                    "start_fix_defined_indicator == 1",
                    "arrival_starts_at_defined_fix",
                    "sia_start_point_requirement",
                    ["start_fix_defined_indicator"],
                ),
                constraint(
                    "C2",
                    "transition_to_iaf_indicator == 1",
                    "arrival_connects_ats_route_to_iaf",
                    "sia_transition_requirement",
                    ["transition_to_iaf_indicator"],
                ),
                constraint(
                    "C3",
                    "radar_vector_dependency_score >= (cda_track_length_km + 14 * served_airports_count - route_length_km) / 90",
                    "rnav_navigation_efficiency_proxy",
                    "scenario_rnav_efficiency_model",
                    [
                        "radar_vector_dependency_score",
                        "cda_track_length_km",
                        "served_airports_count",
                        "route_length_km",
                    ],
                    source_type="task_or_scenario_model",
                ),
                constraint(
                    "C4",
                    "radar_vector_dependency_score <= 0.25",
                    "rnav_navigation_efficiency_ceiling",
                    "sia_navigation_efficiency_requirement",
                    ["radar_vector_dependency_score"],
                ),
                constraint(
                    "C5",
                    "served_airports_count >= 1",
                    "terminal_area_service_lower_bound",
                    "sia_multi_airport_service_capability",
                    ["served_airports_count"],
                ),
                constraint(
                    "C6",
                    "served_airports_count <= terminal_area_airports",
                    "terminal_area_service_scope",
                    "sia_multi_airport_service_capability",
                    ["served_airports_count"],
                    ["terminal_area_airports"],
                ),
                constraint(
                    "C7",
                    "cda_track_length_km >= 35.0",
                    "continuous_descent_track_length_proxy",
                    "scenario_cda_profile_model",
                    ["cda_track_length_km"],
                    source_type="task_or_scenario_model",
                ),
                constraint(
                    "C8",
                    "route_length_km >= cda_track_length_km + 8 * served_airports_count",
                    "cda_is_subroute_of_arrival",
                    "scenario_arrival_geometry_model",
                    ["route_length_km", "cda_track_length_km", "served_airports_count"],
                    source_type="task_or_scenario_model",
                ),
            ],
            structure_only_constraints=[
                structure_constraint(
                    "S1",
                    "SIA route design supports RNAV navigation to reduce radar guidance",
                    "rnav_efficiency_rule_activation",
                    "sia_navigation_efficiency_requirement",
                ),
                structure_constraint(
                    "S2",
                    "SIA route design considers continuous descent approach benefits",
                    "cda_consideration_rule_activation",
                    "sia_cda_consideration",
                ),
            ],
            should_activate=["defined-fix SIA start", "ATS-to-IAF transition", "RNAV efficiency", "CDA consideration"],
            should_exclude=["missed-approach turn rules", "SBAS FAS data-block rules"],
            should_resolve=["multi-airport service is allowed but still constrained by terminal-area scenario facts"],
            challenge_types=["generic_rule_selection", "scenario_conditioned_applicability"],
        )
    )

    records.append(
        make_record(
            rule_lookup,
            omega_id="AVI_OPT_22",
            title="Departure turn TNH and MOC tradeoff near a DER obstacle",
            task_type="departure_turn_design",
            design_intent=(
                "Choose a departure climb gradient and turn initiation envelope for a fixed-wing SID with an "
                "obstacle close to the DER. The optimizer trades off a lower climb-gradient burden against "
                "larger turn-height and MOC margins."
            ),
            scenario_facts={
                "aircraft_class": "fixed-wing",
                "departure_reference": "DER",
                "dr_m": 1850,
                "do_m": 220,
                "obstacle_height_above_der_m": 56,
                "nominal_track_distance_from_der_m": 1850,
            },
            decision_variables={
                "pdg_percent": {"type": "continuous", "unit": "percent", "lower": 3.3, "upper": 7.0},
                "turn_height_m": {"type": "continuous", "unit": "m", "lower": 50.0, "upper": 180.0},
                "moc_turn_area_m": {"type": "continuous", "unit": "m", "lower": 0.0, "upper": 120.0},
                "turn_init_lateral_extent_m": {
                    "type": "continuous",
                    "unit": "m",
                    "lower": 100.0,
                    "upper": 260.0,
                },
                "route_extension_m": {"type": "continuous", "unit": "m", "lower": 0.0, "upper": 1200.0},
            },
            objectives=[
                {"name": "minimize_departure_climb_gradient", "expression": "pdg_percent"},
                {"name": "maximize_turn_height_margin", "expression": "turn_height_m - obstacle_height_above_der_m"},
                {"name": "minimize_route_extension", "expression": "route_extension_m"},
            ],
            query_preferences={
                "lambda": [0.4, 0.4, 0.2],
                "meaning": "balance aircraft performance burden, obstacle clearance, and route compactness",
            },
            source_rule_ids=[
                "turn_init_area_der_extent",
                "tnh_calculation_formula",
                "tnh_obstacle_clearance_requirement",
                "moc_turn_area_formula",
                "moc_turn_initiation_area_calculation",
            ],
            executable_constraints=[
                constraint(
                    "C1",
                    "turn_init_lateral_extent_m >= 150",
                    "fixed_wing_der_turn_initiation_extent",
                    "turn_init_area_der_extent",
                    ["turn_init_lateral_extent_m"],
                ),
                constraint(
                    "C2",
                    "turn_height_m == dr_m * pdg_percent / 100 + 5",
                    "tnh_from_pdg_and_distance",
                    "tnh_calculation_formula",
                    ["turn_height_m", "pdg_percent"],
                    ["dr_m"],
                ),
                constraint(
                    "C3",
                    "moc_turn_area_m >= 75",
                    "turn_area_minimum_moc_floor",
                    "moc_turn_area_formula",
                    ["moc_turn_area_m"],
                ),
                constraint(
                    "C4",
                    "turn_height_m >= obstacle_height_above_der_m + moc_turn_area_m",
                    "tnh_clears_obstacle_with_moc",
                    "tnh_obstacle_clearance_requirement",
                    ["turn_height_m", "moc_turn_area_m"],
                    ["obstacle_height_above_der_m"],
                    metadata=text_rule_proxy(
                        "tnh_obstacle_clearance_requirement",
                        "Textual obstacle-clearance semantics are instantiated as a numeric TNH >= obstacle height + MOC predicate for this scenario.",
                    ),
                ),
                constraint(
                    "C5",
                    "route_extension_m >= (7.0 - pdg_percent) * 300",
                    "lower_pdg_requires_route_extension_proxy",
                    "scenario_departure_geometry_model",
                    ["route_extension_m", "pdg_percent"],
                    source_type="task_or_scenario_model",
                ),
            ],
            structure_only_constraints=[
                structure_constraint(
                    "S1",
                    "MOC in the turn-initiation area is measured from DER along the nominal track",
                    "turn_initiation_moc_reference_line",
                    "moc_turn_initiation_area_calculation",
                )
            ],
            should_activate=["DER-side turn-initiation extent", "TNH formula", "turn-area MOC calculation"],
            should_exclude=["helicopter FATO-side turn-initiation extent"],
            should_resolve=["TNH and MOC must be jointly satisfied for the controlling DER obstacle"],
            challenge_types=["dependency_or_formula_propagation", "branch_or_exclusion"],
        )
    )

    records.append(
        make_record(
            rule_lookup,
            omega_id="AVI_OPT_23",
            title="PAOAS immediate climb-and-turn envelope",
            task_type="parallel_approach_operation_design",
            design_intent=(
                "Select a PAOAS breakout maneuver envelope for parallel approach operations. The design should "
                "track the planned escape heading, keep the protected altitude low, and preserve the KG-derived "
                "climb-start height and maximum turn-angle protection requirements."
            ),
            scenario_facts={
                "operation_type": "parallel_approach_operations",
                "procedure_element": "PAOAS",
                "planned_escape_heading_deg": 45,
                "assigned_altitude_floor_m": 760,
                "procedure_publication_context": "parallel approach obstacle assessment",
            },
            decision_variables={
                "obstacle_clearance_start_height_m": {
                    "type": "continuous",
                    "unit": "m",
                    "lower": 80.0,
                    "upper": 180.0,
                },
                "turn_angle_deg": {"type": "continuous", "unit": "degree", "lower": 0.0, "upper": 70.0},
                "heading_deviation_deg": {"type": "continuous", "unit": "degree", "lower": 0.0, "upper": 45.0},
                "protected_altitude_m": {"type": "continuous", "unit": "m", "lower": 600.0, "upper": 1100.0},
                "angle_reference_true_north_indicator": {
                    "type": "binary",
                    "unit": "indicator",
                    "lower": 0,
                    "upper": 1,
                },
            },
            objectives=[
                {"name": "minimize_turn_maneuver_load", "expression": "turn_angle_deg"},
                {"name": "minimize_escape_heading_deviation", "expression": "heading_deviation_deg"},
                {"name": "minimize_protected_altitude", "expression": "protected_altitude_m"},
                {"name": "maximize_start_height_margin", "expression": "obstacle_clearance_start_height_m - 120"},
            ],
            query_preferences={
                "lambda": [0.25, 0.3, 0.2, 0.25],
                "meaning": "prefer heading tracking and PAOAS clearance margin, while avoiding excessive turn load or protected altitude",
            },
            source_rule_ids=["paoas_protection_scope", "navigation_angle_convention"],
            executable_constraints=[
                constraint(
                    "C1",
                    "obstacle_clearance_start_height_m >= 120",
                    "paoas_clearance_start_height",
                    "paoas_protection_scope",
                    ["obstacle_clearance_start_height_m"],
                ),
                constraint(
                    "C2",
                    "turn_angle_deg <= 45",
                    "paoas_max_turn_angle",
                    "paoas_protection_scope",
                    ["turn_angle_deg"],
                ),
                constraint(
                    "C3",
                    "angle_reference_true_north_indicator == 1",
                    "planning_angle_reference_true_north",
                    "navigation_angle_convention",
                    ["angle_reference_true_north_indicator"],
                ),
                constraint(
                    "C4",
                    "heading_deviation_deg >= planned_escape_heading_deg - turn_angle_deg",
                    "planned_escape_heading_tracking",
                    "scenario_paoas_heading_model",
                    ["heading_deviation_deg", "turn_angle_deg"],
                    ["planned_escape_heading_deg"],
                    source_type="task_or_scenario_model",
                ),
                constraint(
                    "C5",
                    "turn_angle_deg <= planned_escape_heading_deg",
                    "no_overturn_beyond_escape_heading",
                    "scenario_paoas_heading_model",
                    ["turn_angle_deg"],
                    ["planned_escape_heading_deg"],
                    source_type="task_or_scenario_model",
                ),
                constraint(
                    "C6",
                    "protected_altitude_m >= assigned_altitude_floor_m + 0.8 * (obstacle_clearance_start_height_m - 120) + 2.0 * heading_deviation_deg",
                    "scenario_altitude_floor",
                    "scenario_paoas_altitude_model",
                    ["protected_altitude_m", "obstacle_clearance_start_height_m", "heading_deviation_deg"],
                    ["assigned_altitude_floor_m"],
                    source_type="task_or_scenario_model",
                ),
            ],
            structure_only_constraints=[
                structure_constraint(
                    "S1",
                    "PAOAS protects immediate climb and turn to the specified heading and altitude",
                    "protected_flight_maneuver_scope",
                    "paoas_protection_scope",
                )
            ],
            should_activate=["PAOAS climb-start height", "PAOAS maximum turn angle", "true-north planning convention"],
            should_exclude=["visual-segment-only PinS departure rules"],
            should_resolve=["planned escape heading is represented by a scenario heading-tracking model, while PAOAS supplies the 120 m and 45 degree protection bounds"],
            challenge_types=["dependency_or_formula_propagation", "provenance_traceability"],
        )
    )

    records.append(
        make_record(
            rule_lookup,
            omega_id="AVI_OPT_24",
            title="Low-altitude helicopter holding speed and timing buffer",
            task_type="helicopter_holding_design",
            design_intent=(
                "Choose a helicopter holding speed at or below 6000 ft while keeping enough timing buffer for "
                "outbound tracking. The optimizer trades off a smaller holding pattern against schedule robustness."
            ),
            scenario_facts={
                "aircraft_category": "H",
                "altitude_ft": 5800,
                "altitude_m": 1768,
                "holding_context": "terminal helicopter hold",
                "expected_wind_timing_error_s": 6,
            },
            decision_variables={
                "holding_indicated_airspeed_kmh": {
                    "type": "continuous",
                    "unit": "km/h",
                    "lower": 130.0,
                    "upper": 230.0,
                },
                "pattern_radius_proxy_km": {"type": "continuous", "unit": "km", "lower": 0.0, "upper": 10.0},
                "timing_error_s": {"type": "continuous", "unit": "s", "lower": 0.0, "upper": 20.0},
                "timing_buffer_s": {"type": "continuous", "unit": "s", "lower": 0.0, "upper": 10.0},
            },
            objectives=[
                {"name": "minimize_pattern_radius_proxy", "expression": "pattern_radius_proxy_km"},
                {"name": "maximize_timing_buffer", "expression": "timing_buffer_s"},
            ],
            query_preferences={
                "lambda": [0.5, 0.5],
                "meaning": "balance compact helicopter holding geometry and timing robustness",
            },
            source_rule_ids=[
                "holding_speed_helicopter_6000ft_or_below",
                "RA-3.6.3-operating-assumption-timing-tolerance",
            ],
            executable_constraints=[
                constraint(
                    "C1",
                    "holding_indicated_airspeed_kmh <= 185",
                    "helicopter_holding_speed_limit_below_6000ft",
                    "holding_speed_helicopter_6000ft_or_below",
                    ["holding_indicated_airspeed_kmh"],
                    metadata=equality_parameter_boundary(
                        "holding_speed_helicopter_6000ft_or_below",
                        "The extracted table value is treated as the maximum design IAS for the applicable altitude band.",
                    ),
                ),
                constraint(
                    "C2",
                    "timing_error_s <= 10",
                    "outbound_timing_tolerance",
                    "RA-3.6.3-operating-assumption-timing-tolerance",
                    ["timing_error_s"],
                ),
                constraint(
                    "C3",
                    "timing_buffer_s == 10 - timing_error_s",
                    "timing_buffer_certificate",
                    "RA-3.6.3-operating-assumption-timing-tolerance",
                    ["timing_buffer_s", "timing_error_s"],
                ),
                constraint(
                    "C4",
                    "pattern_radius_proxy_km >= 0.025 * holding_indicated_airspeed_kmh",
                    "speed_to_pattern_radius_proxy",
                    "scenario_holding_geometry_model",
                    ["pattern_radius_proxy_km", "holding_indicated_airspeed_kmh"],
                    source_type="task_or_scenario_model",
                ),
                constraint(
                    "C5",
                    "timing_error_s >= 0.04 * (185 - holding_indicated_airspeed_kmh)",
                    "low_speed_wind_timing_error_proxy",
                    "scenario_holding_timing_model",
                    ["timing_error_s", "holding_indicated_airspeed_kmh"],
                    source_type="task_or_scenario_model",
                ),
            ],
            structure_only_constraints=[],
            should_activate=["helicopter holding speed at or below 6000 ft", "outbound timing tolerance"],
            should_exclude=["Category H holding speed above 14000 ft"],
            should_resolve=["altitude-conditioned speed rule before optimizing holding radius"],
            challenge_types=["scenario_conditioned_applicability", "branch_or_exclusion"],
        )
    )

    records.append(
        make_record(
            rule_lookup,
            omega_id="AVI_OPT_25",
            title="PinS departure visual segment to IDF",
            task_type="pins_departure_design",
            design_intent=(
                "Design the visual segment from a heliport to the IDF for a PinS departure. The optimizer trades "
                "off short visual routing against VMC buffer and route deviation, while respecting that the visual "
                "segment has no obstacle protection and IMC entry before the IDF is not allowed."
            ),
            scenario_facts={
                "procedure_type": "pins departure",
                "operation": "helicopter",
                "heliport_to_idf_distance_km": 3.2,
                "nearby_vfr_corridor_width_km": 1.4,
                "minimum_vmc_buffer_min": 5,
            },
            decision_variables={
                "visual_segment_declared_indicator": {
                    "type": "binary",
                    "unit": "indicator",
                    "lower": 0,
                    "upper": 1,
                },
                "visual_segment_length_km": {"type": "continuous", "unit": "km", "lower": 1.0, "upper": 7.0},
                "vmc_buffer_min": {"type": "continuous", "unit": "min", "lower": 0.0, "upper": 20.0},
                "obstacle_protection_claim_indicator": {
                    "type": "binary",
                    "unit": "indicator",
                    "lower": 0,
                    "upper": 1,
                },
                "imc_before_idf_indicator": {"type": "binary", "unit": "indicator", "lower": 0, "upper": 1},
                "route_deviation_km": {"type": "continuous", "unit": "km", "lower": 0.0, "upper": 5.0},
            },
            objectives=[
                {"name": "minimize_visual_segment_length", "expression": "visual_segment_length_km"},
                {"name": "maximize_vmc_buffer", "expression": "vmc_buffer_min"},
                {"name": "minimize_route_deviation", "expression": "route_deviation_km"},
            ],
            query_preferences={
                "lambda": [0.4, 0.4, 0.2],
                "meaning": "prefer a short visual segment, but protect VMC margin and avoid lateral detours",
            },
            source_rule_ids=[
                "pins_departure_visual_segment_requirement",
                "pins_departure_visual_segment_obstacle_protection",
                "pins_departure_imc_entry_restriction",
            ],
            executable_constraints=[
                constraint(
                    "C1",
                    "visual_segment_declared_indicator == 1",
                    "pins_departure_visual_segment_required",
                    "pins_departure_visual_segment_requirement",
                    ["visual_segment_declared_indicator"],
                    metadata=text_rule_proxy(
                        "pins_departure_visual_segment_requirement",
                        "The textual requirement that the initial phase is a visual segment is encoded as a declared-segment indicator for this task.",
                    ),
                ),
                constraint(
                    "C2",
                    "obstacle_protection_claim_indicator == 0",
                    "no_obstacle_protection_in_visual_segment",
                    "pins_departure_visual_segment_obstacle_protection",
                    ["obstacle_protection_claim_indicator"],
                    metadata=text_rule_proxy(
                        "pins_departure_visual_segment_obstacle_protection",
                        "The textual 'no obstacle protection' statement is encoded as a prohibition on claiming protected obstacle clearance in the visual segment.",
                    ),
                ),
                constraint(
                    "C3",
                    "imc_before_idf_indicator == 0",
                    "imc_entry_forbidden_before_idf",
                    "pins_departure_imc_entry_restriction",
                    ["imc_before_idf_indicator"],
                ),
                constraint(
                    "C4",
                    "vmc_buffer_min >= minimum_vmc_buffer_min",
                    "scenario_vmc_buffer_requirement",
                    "scenario_visual_departure_model",
                    ["vmc_buffer_min"],
                    ["minimum_vmc_buffer_min"],
                    source_type="task_or_scenario_model",
                ),
                constraint(
                    "C5",
                    "visual_segment_length_km >= heliport_to_idf_distance_km + 0.15 * (vmc_buffer_min - minimum_vmc_buffer_min)",
                    "visual_route_covers_heliport_to_idf",
                    "scenario_visual_departure_model",
                    ["visual_segment_length_km", "vmc_buffer_min"],
                    ["heliport_to_idf_distance_km", "minimum_vmc_buffer_min"],
                    source_type="task_or_scenario_model",
                ),
                constraint(
                    "C6",
                    "route_deviation_km >= 0.2 * (vmc_buffer_min - minimum_vmc_buffer_min)",
                    "extra_vmc_buffer_route_deviation_proxy",
                    "scenario_visual_departure_model",
                    ["route_deviation_km", "vmc_buffer_min"],
                    ["minimum_vmc_buffer_min"],
                    source_type="task_or_scenario_model",
                ),
            ],
            structure_only_constraints=[],
            should_activate=["PinS visual segment to IDF", "no obstacle protection", "no IMC before IDF"],
            should_exclude=["standard fixed-wing SIA route rules"],
            should_resolve=["the visual segment is a required flight phase but is not an obstacle-protected segment"],
            challenge_types=["branch_or_exclusion", "scenario_conditioned_applicability"],
        )
    )

    records.append(
        make_record(
            rule_lookup,
            omega_id="AVI_OPT_26",
            title="Helicopter visual segment descent angle with OCS margin",
            task_type="visual_segment_descent_design",
            design_intent=(
                "Choose a visual-segment descent angle for a helicopter procedure. The design should stay close "
                "to the KG-derived nominal VSDA while preserving the required angular offset above the obstacle "
                "clearance surface and Annex 14 takeoff climb surface."
            ),
            scenario_facts={
                "procedure_type": "helicopter approach visual segment",
                "annex14_takeoff_climb_surface_angle_deg": 6.9,
                "candidate_ocs_angle_deg": 7.0,
                "nearby_urban_obstacle_environment": True,
            },
            decision_variables={
                "visual_segment_descent_angle_deg": {
                    "type": "continuous",
                    "unit": "degree",
                    "lower": 6.0,
                    "upper": 10.0,
                },
                "ocs_angle_deg": {"type": "continuous", "unit": "degree", "lower": 5.0, "upper": 9.0},
                "annex14_margin_deg": {"type": "continuous", "unit": "degree", "lower": 0.0, "upper": 3.0},
                "angle_deviation_deg": {"type": "continuous", "unit": "degree", "lower": 0.0, "upper": 3.0},
            },
            objectives=[
                {"name": "minimize_vsda_nominal_deviation", "expression": "angle_deviation_deg"},
                {"name": "maximize_obstacle_surface_margin", "expression": "annex14_margin_deg"},
            ],
            query_preferences={
                "lambda": [0.6, 0.4],
                "meaning": "prefer the nominal VSDA unless obstacle-surface margin requires extra conservatism",
            },
            source_rule_ids=[
                "vsda_nominal_value",
                "vsda_obstacle_clearance_surface_offset",
                "vsda_annex14_takeoff_climb_surface_clearance",
            ],
            executable_constraints=[
                constraint(
                    "C1",
                    "visual_segment_descent_angle_deg >= 8.3",
                    "nominal_vsda_design_floor",
                    "vsda_nominal_value",
                    ["visual_segment_descent_angle_deg"],
                    metadata=equality_parameter_boundary(
                        "vsda_nominal_value",
                        "The nominal VSDA parameter is used as the design target/floor; steeper angles are allowed only to buy obstacle-surface margin.",
                    ),
                ),
                constraint(
                    "C2",
                    "visual_segment_descent_angle_deg - ocs_angle_deg >= 1.12",
                    "vsda_above_obstacle_clearance_surface",
                    "vsda_obstacle_clearance_surface_offset",
                    ["visual_segment_descent_angle_deg", "ocs_angle_deg"],
                    metadata=equality_parameter_boundary(
                        "vsda_obstacle_clearance_surface_offset",
                        "The extracted angular offset is used as the minimum allowed VSDA-minus-OCS separation.",
                    ),
                ),
                constraint(
                    "C3",
                    "annex14_margin_deg >= 1.12",
                    "vsda_above_annex14_takeoff_climb_surface",
                    "vsda_annex14_takeoff_climb_surface_clearance",
                    ["annex14_margin_deg"],
                ),
                constraint(
                    "C4",
                    "angle_deviation_deg >= abs(visual_segment_descent_angle_deg - 8.3)",
                    "nominal_vsda_deviation_proxy",
                    "scenario_visual_segment_model",
                    ["angle_deviation_deg", "visual_segment_descent_angle_deg"],
                    source_type="task_or_scenario_model",
                ),
                constraint(
                    "C5",
                    "abs(annex14_margin_deg - (visual_segment_descent_angle_deg - annex14_takeoff_climb_surface_angle_deg)) <= 1e-6",
                    "annex14_margin_definition",
                    "scenario_visual_segment_model",
                    ["annex14_margin_deg", "visual_segment_descent_angle_deg"],
                    ["annex14_takeoff_climb_surface_angle_deg"],
                    source_type="task_or_scenario_model",
                ),
                constraint(
                    "C6",
                    "ocs_angle_deg == candidate_ocs_angle_deg",
                    "scenario_candidate_ocs_angle",
                    "scenario_visual_segment_model",
                    ["ocs_angle_deg"],
                    ["candidate_ocs_angle_deg"],
                    source_type="task_or_scenario_model",
                ),
            ],
            structure_only_constraints=[],
            should_activate=["nominal VSDA", "VSDA OCS angular offset", "Annex 14 clearance offset"],
            should_exclude=["Baro-VNAV DA/H publication rules"],
            should_resolve=["nominal angle and clearance offsets jointly define the feasible visual segment"],
            challenge_types=["dependency_or_formula_propagation"],
        )
    )

    records.append(
        make_record(
            rule_lookup,
            omega_id="AVI_OPT_27",
            title="SBAS FAS data quality allocation",
            task_type="sbas_fas_data_quality_design",
            design_intent=(
                "Allocate survey and validation effort for an SBAS FAS data block. The optimizer trades off "
                "survey cost against tighter data quality margins across FPAP, LTP/FTP, TCH, glide-slope angle, "
                "course width, and delta length offset parameters."
            ),
            scenario_facts={
                "navigation_system": "sbas",
                "publication_context": "FAS data block",
                "runway": "RWY 27",
                "survey_vendor_precision_class": "high_integrity",
                "fas_data_quality_items": [
                    {"data_element": "FPAP（纬度和经度）", "precision_variable": "fpap_latlon_precision_m"},
                    {"data_element": "LTP/FTP（纬度和经度）", "precision_variable": "ltp_ftp_latlon_precision_m"},
                    {"data_element": "LTP/FTP（椭球体高度）", "precision_variable": "ltp_ftp_height_precision_m"},
                    {"data_element": "进近 TCH", "precision_variable": "approach_tch_precision_m"},
                    {"data_element": "下滑道角度", "precision_variable": "glide_slope_angle_precision_deg"},
                    {"data_element": "Delta 长度偏移", "precision_variable": "delta_length_offset_resolution_m"},
                ],
            },
            decision_variables={
                "fpap_latlon_precision_m": {"type": "continuous", "unit": "m", "lower": 0.05, "upper": 1.0},
                "ltp_ftp_latlon_precision_m": {"type": "continuous", "unit": "m", "lower": 0.05, "upper": 1.0},
                "ltp_ftp_height_precision_m": {"type": "continuous", "unit": "m", "lower": 0.05, "upper": 1.0},
                "approach_tch_precision_m": {"type": "continuous", "unit": "m", "lower": 0.05, "upper": 1.0},
                "glide_slope_angle_precision_deg": {
                    "type": "continuous",
                    "unit": "degree",
                    "lower": 0.001,
                    "upper": 0.05,
                },
                "delta_length_offset_resolution_m": {
                    "type": "continuous",
                    "unit": "m",
                    "lower": 1.0,
                    "upper": 20.0,
                },
                "survey_cost_score": {"type": "continuous", "unit": "score", "lower": 0.0, "upper": 100.0},
            },
            objectives=[
                {"name": "minimize_survey_cost_score", "expression": "survey_cost_score"},
                {
                    "name": "maximize_fas_position_quality_margin",
                    "expression": "0.3 - max(fpap_latlon_precision_m, ltp_ftp_latlon_precision_m)",
                },
            ],
            query_preferences={
                "lambda": [0.45, 0.55],
                "meaning": "slightly prioritize data quality margin over survey cost",
            },
            source_rule_ids=[
                "fas_data_quality_fpap_latlon",
                "fas_data_quality_ltp_ftp_latlon",
                "fas_data_quality_ltp_ftp_ellipsoidal_height",
                "fas_data_quality_approach_tch",
                "fas_data_quality_glide_slope_angle",
                "fas_data_quality_delta_length_offset",
            ],
            executable_constraints=[
                constraint(
                    "C1",
                    "fpap_latlon_precision_m <= 0.3",
                    "fpap_latlon_precision",
                    "fas_data_quality_fpap_latlon",
                    ["fpap_latlon_precision_m"],
                    metadata=equality_parameter_boundary(
                        "fas_data_quality_fpap_latlon",
                        "The extracted precision value is interpreted as the maximum allowable horizontal position error.",
                    ),
                ),
                constraint(
                    "C2",
                    "ltp_ftp_latlon_precision_m <= 0.3",
                    "ltp_ftp_latlon_precision",
                    "fas_data_quality_ltp_ftp_latlon",
                    ["ltp_ftp_latlon_precision_m"],
                    metadata=equality_parameter_boundary(
                        "fas_data_quality_ltp_ftp_latlon",
                        "The extracted precision value is interpreted as the maximum allowable horizontal position error.",
                    ),
                ),
                constraint(
                    "C3",
                    "ltp_ftp_height_precision_m <= 0.25",
                    "ltp_ftp_ellipsoidal_height_precision",
                    "fas_data_quality_ltp_ftp_ellipsoidal_height",
                    ["ltp_ftp_height_precision_m"],
                    metadata=equality_parameter_boundary(
                        "fas_data_quality_ltp_ftp_ellipsoidal_height",
                        "The extracted precision value is interpreted as the maximum allowable ellipsoidal-height error.",
                    ),
                ),
                constraint(
                    "C4",
                    "approach_tch_precision_m <= 0.5",
                    "approach_tch_precision",
                    "fas_data_quality_approach_tch",
                    ["approach_tch_precision_m"],
                    metadata=equality_parameter_boundary(
                        "fas_data_quality_approach_tch",
                        "The extracted precision value is interpreted as the maximum allowable TCH error.",
                    ),
                ),
                constraint(
                    "C5",
                    "glide_slope_angle_precision_deg <= 0.01",
                    "glide_slope_angle_precision",
                    "fas_data_quality_glide_slope_angle",
                    ["glide_slope_angle_precision_deg"],
                    metadata=equality_parameter_boundary(
                        "fas_data_quality_glide_slope_angle",
                        "The extracted precision value is interpreted as the maximum allowable glide-slope-angle error.",
                    ),
                ),
                constraint(
                    "C6",
                    "delta_length_offset_resolution_m <= 8",
                    "delta_length_offset_resolution",
                    "fas_data_quality_delta_length_offset",
                    ["delta_length_offset_resolution_m"],
                    metadata=equality_parameter_boundary(
                        "fas_data_quality_delta_length_offset",
                        "The extracted resolution value is interpreted as the coarsest allowable encoded resolution.",
                    ),
                ),
                constraint(
                    "C7",
                    "survey_cost_score >= 160 * (0.3 - fpap_latlon_precision_m) + 160 * (0.3 - ltp_ftp_latlon_precision_m) + 80 * (0.25 - ltp_ftp_height_precision_m) + 20 * (0.5 - approach_tch_precision_m) + 1000 * (0.01 - glide_slope_angle_precision_deg) + (8 - delta_length_offset_resolution_m)",
                    "higher_precision_increases_survey_cost_proxy",
                    "scenario_fas_quality_cost_model",
                    [
                        "survey_cost_score",
                        "fpap_latlon_precision_m",
                        "ltp_ftp_latlon_precision_m",
                        "ltp_ftp_height_precision_m",
                        "approach_tch_precision_m",
                        "glide_slope_angle_precision_deg",
                        "delta_length_offset_resolution_m",
                    ],
                    source_type="task_or_scenario_model",
                ),
            ],
            structure_only_constraints=[
                structure_constraint(
                    "S1",
                    "FAS data quality constraints are checked per parameter rather than as a single aggregate score",
                    "fas_parameterwise_quality_semantics",
                    "scenario_fas_quality_model",
                    source_type="task_or_scenario_model",
                )
            ],
            should_activate=["FAS data quality rules for FPAP, LTP/FTP, TCH, glide-slope angle, and delta length"],
            should_exclude=["SBAS channel-number range as the primary publication decision"],
            should_resolve=["multiple FAS data-quality parameters must be satisfied simultaneously"],
            challenge_types=["multi_rule_conjunction", "provenance_traceability"],
        )
    )

    records.append(
        make_record(
            rule_lookup,
            omega_id="AVI_OPT_28",
            title="Baro-VNAV chart minima and waypoint-use decision",
            task_type="baro_vnav_publication_design",
            design_intent=(
                "Prepare the publication package for a Baro-VNAV APV procedure. The optimizer trades off a "
                "compact minima box against complete APV/Baro-VNAV semantics, using DA/H and avoiding FAF/MAPt "
                "identification where the KG rules require it."
            ),
            scenario_facts={
                "procedure_type": "baro-vnav",
                "approach_class": "APV",
                "runway": "RWY 18",
                "chart_contains_lnav_minima": True,
                "chart_contains_lnav_vnav_minima": True,
            },
            decision_variables={
                "apv_classification_indicator": {
                    "type": "binary",
                    "unit": "indicator",
                    "lower": 0,
                    "upper": 1,
                },
                "uses_da_h_indicator": {"type": "binary", "unit": "indicator", "lower": 0, "upper": 1},
                "faf_mapt_identification_count": {"type": "integer", "unit": "count", "lower": 0, "upper": 2},
                "minima_box_entries_count": {"type": "integer", "unit": "count", "lower": 0, "upper": 4},
                "publication_complexity_score": {"type": "continuous", "unit": "score", "lower": 0.0, "upper": 10.0},
            },
            objectives=[
                {"name": "minimize_publication_complexity", "expression": "publication_complexity_score"},
                {"name": "maximize_minima_completeness", "expression": "minima_box_entries_count"},
            ],
            query_preferences={
                "lambda": [0.45, 0.55],
                "meaning": "prefer complete minima semantics while keeping the chart concise",
            },
            source_rule_ids=[
                "baro_vnav_apv_classification",
                "baro_vnav_da_h_usage",
                "baro_vnav_faf_mapt_not_used",
                "chart_oceh_publication_content",
            ],
            executable_constraints=[
                constraint(
                    "C1",
                    "apv_classification_indicator == 1",
                    "baro_vnav_is_apv",
                    "baro_vnav_apv_classification",
                    ["apv_classification_indicator"],
                    metadata=text_rule_proxy(
                        "baro_vnav_apv_classification",
                        "The textual APV classification is encoded as a binary classification indicator for this publication task.",
                    ),
                ),
                constraint(
                    "C2",
                    "uses_da_h_indicator == 1",
                    "baro_vnav_uses_da_h",
                    "baro_vnav_da_h_usage",
                    ["uses_da_h_indicator"],
                    metadata=text_rule_proxy(
                        "baro_vnav_da_h_usage",
                        "The textual DA/H usage requirement is encoded as a binary minima-type indicator.",
                    ),
                ),
                constraint(
                    "C3",
                    "faf_mapt_identification_count == 0",
                    "baro_vnav_no_faf_mapt_identification",
                    "baro_vnav_faf_mapt_not_used",
                    ["faf_mapt_identification_count"],
                    metadata=text_rule_proxy(
                        "baro_vnav_faf_mapt_not_used",
                        "The textual 'FAF/MAPt not used' rule is encoded as zero required FAF/MAPt identifications.",
                    ),
                ),
                constraint(
                    "C4",
                    "minima_box_entries_count >= 2",
                    "minima_box_contains_lnav_and_lnav_vnav",
                    "chart_oceh_publication_content",
                    ["minima_box_entries_count"],
                    metadata=text_rule_proxy(
                        "chart_oceh_publication_content",
                        "The textual list of LNAV and LNAV/VNAV minima values is encoded as a minimum count of required minima-box entries.",
                    ),
                ),
                constraint(
                    "C5",
                    "publication_complexity_score >= 1.5 * minima_box_entries_count + apv_classification_indicator + uses_da_h_indicator",
                    "publication_complexity_increases_with_minima_completeness",
                    "scenario_baro_vnav_publication_model",
                    [
                        "publication_complexity_score",
                        "minima_box_entries_count",
                        "apv_classification_indicator",
                        "uses_da_h_indicator",
                    ],
                    source_type="task_or_scenario_model",
                ),
            ],
            structure_only_constraints=[],
            should_activate=["Baro-VNAV APV classification", "DA/H usage", "FAF/MAPt non-use", "OCA/H minima content"],
            should_exclude=["ILS OAS surface construction rules"],
            should_resolve=["Baro-VNAV publication uses APV/DA-H semantics rather than NPA MDA/H semantics"],
            challenge_types=["branch_or_exclusion", "generic_rule_selection"],
        )
    )

    records.append(
        make_record(
            rule_lookup,
            omega_id="AVI_OPT_29",
            title="Mountainous TAA sector altitude and buffer design",
            task_type="taa_altitude_design",
            design_intent=(
                "Select a mountainous TAA sector altitude and protected-area radii. The optimizer trades off "
                "lower published altitude against terrain-clearance margin and the airspace-coordination burden "
                "created by a larger protected-area boundary."
            ),
            scenario_facts={
                "procedure_type": "terminal arrival altitude",
                "terrain_context": "mountainous",
                "terrain": "mountainous",
                "taa_sector_highest_obstacle_m": 1420,
                "ttaa_highest_obstacle_elevation": 1420,
                "buffer_zone_highest_obstacle_m": 1485,
                "buffer_highest_obstacle_elevation": 1485,
                "nearby_airspace_constraint": "military training area east of sector",
                "nearby_airspace_constraint_radius_km": 50,
            },
            decision_variables={
                "minimum_flight_altitude_m": {
                    "type": "continuous",
                    "unit": "m",
                    "lower": 1500.0,
                    "upper": 2300.0,
                },
                "buffer_radius_km": {"type": "continuous", "unit": "km", "lower": 0.0, "upper": 20.0},
                "outer_boundary_radius_km": {"type": "continuous", "unit": "km", "lower": 30.0, "upper": 60.0},
                "selected_controlling_obstacle_elevation_m": {
                    "type": "continuous",
                    "unit": "m",
                    "lower": 1300.0,
                    "upper": 1700.0,
                },
                "altitude_margin_m": {"type": "continuous", "unit": "m", "lower": 0.0, "upper": 600.0},
                "airspace_coordination_score": {
                    "type": "continuous",
                    "unit": "score",
                    "lower": 0.0,
                    "upper": 15.0,
                },
            },
            objectives=[
                {"name": "minimize_minimum_flight_altitude", "expression": "minimum_flight_altitude_m"},
                {"name": "maximize_altitude_margin", "expression": "altitude_margin_m"},
                {"name": "minimize_airspace_coordination", "expression": "airspace_coordination_score"},
            ],
            query_preferences={
                "lambda": [0.45, 0.4, 0.15],
                "meaning": "prefer lower TAA altitude and compact airspace coordination, but not at the expense of terrain margin",
            },
            source_rule_ids=[
                "ttaa_obstacle_clearance_mountain_increase",
                "ttaa_buffer_zone_radius",
                "ttaa_protected_area_boundary_radius",
                "ttaa_buffer_obstacle_height_adjustment",
            ],
            executable_constraints=[
                constraint(
                    "C1",
                    "buffer_radius_km >= 9",
                    "taa_buffer_zone_radius",
                    "ttaa_buffer_zone_radius",
                    ["buffer_radius_km"],
                    metadata=equality_parameter_boundary(
                        "ttaa_buffer_zone_radius",
                        "The extracted TAA buffer radius is used as the minimum protected buffer radius in the design model.",
                    ),
                ),
                constraint(
                    "C2",
                    "outer_boundary_radius_km >= 46",
                    "taa_outer_boundary_radius",
                    "ttaa_protected_area_boundary_radius",
                    ["outer_boundary_radius_km"],
                    metadata=equality_parameter_boundary(
                        "ttaa_protected_area_boundary_radius",
                        "The extracted protected-area radius is used as the minimum outer-boundary radius in the design model.",
                    ),
                ),
                constraint(
                    "C3",
                    "selected_controlling_obstacle_elevation_m >= buffer_zone_highest_obstacle_m",
                    "buffer_obstacle_controls_when_higher",
                    "ttaa_buffer_obstacle_height_adjustment",
                    ["selected_controlling_obstacle_elevation_m"],
                    ["buffer_zone_highest_obstacle_m"],
                    metadata=text_rule_proxy(
                        "ttaa_buffer_obstacle_height_adjustment",
                        "The textual instruction to use the highest buffer-zone obstacle is encoded as a lower bound on the selected controlling obstacle elevation.",
                    ),
                ),
                constraint(
                    "C4",
                    "minimum_flight_altitude_m >= selected_controlling_obstacle_elevation_m + 300",
                    "mountainous_obstacle_clearance_increase",
                    "ttaa_obstacle_clearance_mountain_increase",
                    ["minimum_flight_altitude_m", "selected_controlling_obstacle_elevation_m"],
                ),
                constraint(
                    "C5",
                    "altitude_margin_m == minimum_flight_altitude_m - selected_controlling_obstacle_elevation_m - 300",
                    "terrain_clearance_margin_certificate",
                    "ttaa_obstacle_clearance_mountain_increase",
                    ["altitude_margin_m", "minimum_flight_altitude_m", "selected_controlling_obstacle_elevation_m"],
                ),
                constraint(
                    "C6",
                    "outer_boundary_radius_km >= buffer_radius_km + 37",
                    "outer_boundary_expands_with_buffer_zone",
                    "scenario_taa_geometry_model",
                    ["outer_boundary_radius_km", "buffer_radius_km"],
                    source_type="task_or_scenario_model",
                ),
                constraint(
                    "C7",
                    "altitude_margin_m <= 12 * (outer_boundary_radius_km - 46) + 40",
                    "larger_boundary_supports_higher_terrain_margin",
                    "scenario_taa_terrain_sampling_model",
                    ["altitude_margin_m", "outer_boundary_radius_km"],
                    source_type="task_or_scenario_model",
                ),
                constraint(
                    "C8",
                    "airspace_coordination_score >= outer_boundary_radius_km - nearby_airspace_constraint_radius_km",
                    "nearby_airspace_coordination_proxy",
                    "scenario_taa_airspace_model",
                    ["airspace_coordination_score", "outer_boundary_radius_km"],
                    ["nearby_airspace_constraint_radius_km"],
                    source_type="task_or_scenario_model",
                ),
            ],
            structure_only_constraints=[],
            should_activate=["mountainous MOC increase", "TAA buffer radius", "TAA outer boundary radius"],
            should_exclude=["non-mountainous TAA altitude assumptions"],
            should_resolve=["buffer-zone obstacle can become the controlling altitude basis"],
            challenge_types=["dependency_or_formula_propagation", "scenario_conditioned_applicability"],
        )
    )

    records.append(
        make_record(
            rule_lookup,
            omega_id="AVI_OPT_30",
            title="RNP approach chart title and supplementary data package",
            task_type="rnp_chart_publication_design",
            design_intent=(
                "Prepare an RNP approach chart package for a runway. The optimizer trades off concise chart "
                "payload against complete title, minima-box, supplementary-data, and waypoint-name-code requirements."
            ),
            scenario_facts={
                "procedure_type": "rnp",
                "runway_designator": "RWY 23",
                "terminal_area_name_code_pool": "local terminal area",
                "naming_convention": "five-letter-numeric name code",
                "requires_apv_baro_vnav_minima": True,
            },
            decision_variables={
                "chart_title_valid_indicator": {
                    "type": "binary",
                    "unit": "indicator",
                    "lower": 0,
                    "upper": 1,
                },
                "minima_box_entries_count": {"type": "integer", "unit": "count", "lower": 0, "upper": 4},
                "supplementary_data_items_count": {"type": "integer", "unit": "count", "lower": 0, "upper": 5},
                "name_code_unique_indicator": {"type": "binary", "unit": "indicator", "lower": 0, "upper": 1},
                "publication_payload_score": {"type": "continuous", "unit": "score", "lower": 0.0, "upper": 15.0},
            },
            objectives=[
                {"name": "minimize_publication_payload", "expression": "publication_payload_score"},
                {"name": "maximize_required_publication_completeness", "expression": "minima_box_entries_count + supplementary_data_items_count"},
            ],
            query_preferences={
                "lambda": [0.45, 0.55],
                "meaning": "prefer a complete RNP chart package with controlled payload and unambiguous name codes",
            },
            source_rule_ids=[
                "chart_title_format_rnav_rnp",
                "chart_oceh_publication_content",
                "chart_required_supplementary_data",
                "five_letter_numeric_name_code_uniqueness_terminal_area",
            ],
            executable_constraints=[
                constraint(
                    "C1",
                    "chart_title_valid_indicator == 1",
                    "rnav_rnp_chart_title_format",
                    "chart_title_format_rnav_rnp",
                    ["chart_title_valid_indicator"],
                    metadata=text_rule_proxy(
                        "chart_title_format_rnav_rnp",
                        "The textual RNAV/RNP chart-title format is encoded as a binary title-validity indicator.",
                    ),
                ),
                constraint(
                    "C2",
                    "minima_box_entries_count >= 2",
                    "oca_h_minima_box_contains_npa_and_apv",
                    "chart_oceh_publication_content",
                    ["minima_box_entries_count"],
                    metadata=text_rule_proxy(
                        "chart_oceh_publication_content",
                        "The textual list of required NPA and APV/Baro-VNAV OCA/H values is encoded as a minimum minima-box entry count.",
                    ),
                ),
                constraint(
                    "C3",
                    "supplementary_data_items_count >= 3",
                    "required_rdh_vpa_temperature_data",
                    "chart_required_supplementary_data",
                    ["supplementary_data_items_count"],
                    metadata=text_rule_proxy(
                        "chart_required_supplementary_data",
                        "The textual list of RDH, VPA, and temperature items is encoded as a minimum supplementary-data item count.",
                    ),
                ),
                constraint(
                    "C4",
                    "name_code_unique_indicator == 1",
                    "terminal_area_name_code_uniqueness",
                    "five_letter_numeric_name_code_uniqueness_terminal_area",
                    ["name_code_unique_indicator"],
                    metadata=text_rule_proxy(
                        "five_letter_numeric_name_code_uniqueness_terminal_area",
                        "The uniqueness relation is encoded as a binary terminal-area name-code uniqueness indicator.",
                    ),
                ),
                constraint(
                    "C5",
                    "publication_payload_score >= 1.5 * minima_box_entries_count + 1.2 * supplementary_data_items_count + chart_title_valid_indicator + 0.8 * name_code_unique_indicator",
                    "publication_payload_increases_with_required_content",
                    "scenario_rnp_publication_payload_model",
                    [
                        "publication_payload_score",
                        "minima_box_entries_count",
                        "supplementary_data_items_count",
                        "chart_title_valid_indicator",
                        "name_code_unique_indicator",
                    ],
                    source_type="task_or_scenario_model",
                ),
            ],
            structure_only_constraints=[
                structure_constraint(
                    "S1",
                    "Chart title text follows RNAV(GNSS) or RNP RWY XX semantics",
                    "chart_title_text_semantics",
                    "chart_title_format_rnav_rnp",
                )
            ],
            should_activate=["RNP chart title", "OCA/H minima box", "supplementary chart data", "name-code uniqueness"],
            should_exclude=["GLS chart title format"],
            should_resolve=["publication completeness is a conjunction of title, minima, supplementary data, and name-code checks"],
            challenge_types=["multi_rule_conjunction", "provenance_traceability"],
        )
    )

    return records
