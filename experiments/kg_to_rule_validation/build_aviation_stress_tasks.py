from __future__ import annotations

import copy
import csv
import json
import time
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
CTHR_ROOT = THIS_DIR.parents[1]
PAPER_DIR = CTHR_ROOT / "paper"
BASE_LAYER_DIR = PAPER_DIR / "aviation_benchmark_layers"
OUT_LAYER_DIR = PAPER_DIR / "aviation_stress_benchmark_layers"
TASK_DIR = OUT_LAYER_DIR / "tasks"
RESULTS_DIR = PAPER_DIR / "results"

BASE_QUERY_PATH = BASE_LAYER_DIR / "aviation_optimization_queries.json"
BASE_RULE_LABEL_PATH = BASE_LAYER_DIR / "aviation_rule_structure_labels.json"
BASE_FEASIBLE_PATH = BASE_LAYER_DIR / "aviation_feasible_region_labels.json"
BASE_RULE_LIBRARY_PATH = (
    PAPER_DIR
    / "full_aviation_kg_rule_library_model_comparison"
    / "full_aviation_rule_library_qwen.json"
)

OUT_QUERY_PATH = OUT_LAYER_DIR / "aviation_stress_optimization_queries.json"
OUT_RULE_LABEL_PATH = OUT_LAYER_DIR / "aviation_stress_rule_structure_labels.json"
OUT_FEASIBLE_PATH = OUT_LAYER_DIR / "aviation_stress_feasible_region_labels.json"
OUT_RULE_EXTENSION_PATH = OUT_LAYER_DIR / "aviation_stress_rule_library_extension.json"
OUT_COMBINED_RULE_LIBRARY_PATH = OUT_LAYER_DIR / "aviation_stress_rule_library_combined.json"
OUT_MANIFEST_PATH = OUT_LAYER_DIR / "aviation_stress_benchmark_layers_manifest.json"

OUT_TASK_MANIFEST_CSV = RESULTS_DIR / "aviation_stress_task_manifest.csv"
OUT_QUALITY_CSV = RESULTS_DIR / "aviation_stress_quality_validation.csv"
OUT_QUALITY_SUMMARY_JSON = RESULTS_DIR / "aviation_stress_quality_validation_summary.json"
OUT_QUALITY_REPORT_MD = RESULTS_DIR / "aviation_stress_quality_validation_report.md"
OUT_GENERATION_REPORT_MD = RESULTS_DIR / "aviation_stress_task_generation_report.md"


DEPENDENCY_TYPES = {"depends_on", "requires"}
EXCLUSION_TYPES = {"excludes", "mutually_exclusive", "conflicts_with", "conflict"}
OVERRIDE_TYPES = {"overrides", "can_override", "replaces", "defeats"}
PRECEDENCE_TYPES = {"precedes", "precedence", "higher_priority_than", "has_precedence_over"}
PARAMETER_TYPES = {"uses_parameter", "parameter_variant_of", "formula_variant_of", "propagates_to"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize_cell(row.get(key)) for key in headers})


def serialize_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(list(value) if isinstance(value, set) else value, ensure_ascii=False)
    return value


def first_provenance(rule_by_id: dict[str, dict[str, Any]], rule_id: str) -> list[dict[str, Any]]:
    rule = rule_by_id.get(rule_id, {})
    provenance = rule.get("provenance") or []
    return copy.deepcopy(provenance)


def source_chunks(rule_by_id: dict[str, dict[str, Any]], rule_id: str) -> list[str]:
    rule = rule_by_id.get(rule_id, {})
    return list(rule.get("source_chunk_ids") or [])


def source_nodes(rule_by_id: dict[str, dict[str, Any]], rule_id: str) -> list[str]:
    rule = rule_by_id.get(rule_id, {})
    return list(rule.get("source_node_ids") or [])


def synthetic_rule(
    rule_id: str,
    name: str,
    base_rule_id: str,
    rule_by_id: dict[str, dict[str, Any]],
    *,
    rule_type: str = "stress_requirement",
    guard: dict[str, Any] | None = None,
    constraints: list[dict[str, Any]] | None = None,
    relations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    base_rule = rule_by_id.get(base_rule_id, {})
    return {
        "rule_id": rule_id,
        "name": name,
        "domain": "aviation",
        "rule_type": rule_type,
        "source_chunk_ids": source_chunks(rule_by_id, base_rule_id),
        "source_node_ids": source_nodes(rule_by_id, base_rule_id),
        "guard": guard or copy.deepcopy(base_rule.get("guard", {})),
        "constraints": constraints if constraints is not None else copy.deepcopy(base_rule.get("constraints", [])),
        "relations": relations or [],
        "provenance": first_provenance(rule_by_id, base_rule_id),
        "synthetic_stress_rule": True,
        "derived_from_base_task": [],
        "derived_from_source_rule": base_rule_id,
        "extraction_notes": (
            "Synthetic stress rule derived from an existing source-grounded rule to activate "
            "a specific resolver interaction in the aviation stress benchmark."
        ),
        "confidence": 1.0,
    }


def make_synthetic_rules(rule_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        synthetic_rule(
            "stress_dta_legacy_nominal_formula",
            "Legacy DTA nominal formula variant for stress testing",
            "dta_formula",
            rule_by_id,
            rule_type="formula_variant",
            relations=[{"type": "formula_variant_of", "target": "dta_formula"}],
        ),
        synthetic_rule(
            "stress_turn_radius_legacy_low_speed_formula",
            "Legacy low-speed RF turn-radius formula variant",
            "turn_radius_km_formula",
            rule_by_id,
            rule_type="formula_variant",
            relations=[{"type": "formula_variant_of", "target": "turn_radius_km_formula"}],
        ),
        synthetic_rule(
            "stress_rf_high_altitude_bank_exception_15deg",
            "High-altitude RF bank-angle exception used as explicit override",
            "rf_segment_bank_angle_15deg_above_fl190",
            rule_by_id,
            rule_type="exception",
            relations=[
                {
                    "type": "overrides",
                    "target": "rf_segment_max_bank_angle_25deg",
                    "source_quote": "Derived stress encoding of the FL190 bank-angle exception.",
                }
            ],
        ),
        synthetic_rule(
            "stress_rf_boundary_bank_exception_15deg",
            "Boundary-case high-altitude RF bank-angle exception",
            "rf_segment_bank_angle_15deg_above_fl190",
            rule_by_id,
            rule_type="exception",
            relations=[
                {
                    "type": "overrides",
                    "target": "rf_segment_max_bank_angle_25deg",
                    "source_quote": "Derived stress encoding of the FL190 bank-angle exception.",
                }
            ],
        ),
        synthetic_rule(
            "stress_caac_gbas_length_precedence",
            "CAAC GBAS segment-length precedence marker",
            "pbn_intermediate_segment_max_length_gbas",
            rule_by_id,
            rule_type="precedence",
            relations=[
                {"type": "precedes", "target": "stress_icao_gbas_length_competitor"},
                {"type": "precedes", "target": "stress_legacy_gbas_length_competitor"},
            ],
        ),
        synthetic_rule(
            "stress_icao_gbas_length_competitor",
            "ICAO GBAS segment-length lower-priority competitor",
            "pbn_intermediate_segment_max_length_gbas",
            rule_by_id,
            rule_type="stress_requirement",
        ),
        synthetic_rule(
            "stress_legacy_gbas_length_competitor",
            "Legacy GBAS segment-length lower-priority competitor",
            "pbn_intermediate_segment_max_length_gbas",
            rule_by_id,
            rule_type="stress_requirement",
        ),
        synthetic_rule(
            "stress_caac_msa_arc_precedence",
            "CAAC MSA arc-radius precedence marker",
            "dme_arc_radius_range_sector_partition",
            rule_by_id,
            rule_type="precedence",
            relations=[
                {"type": "precedes", "target": "stress_icao_msa_arc_competitor"},
                {"type": "precedes", "target": "stress_environmental_overlay_arc_competitor"},
            ],
        ),
        synthetic_rule(
            "stress_icao_msa_arc_competitor",
            "ICAO MSA arc-radius lower-priority competitor",
            "dme_arc_radius_range_sector_partition",
            rule_by_id,
            rule_type="stress_requirement",
        ),
        synthetic_rule(
            "stress_environmental_overlay_arc_competitor",
            "Overlay MSA arc-radius competitor",
            "dme_arc_radius_range_sector_partition",
            rule_by_id,
            rule_type="stress_requirement",
        ),
        synthetic_rule(
            "stress_glidepath_rounded_publication_variant",
            "Rounded glide-path publication variant",
            "rule_non_si_glide_path_formula",
            rule_by_id,
            rule_type="formula_variant",
            relations=[
                {"type": "uses_parameter", "target": "ils_glidepath_angle_range"},
                {"type": "formula_variant_of", "target": "rule_non_si_glide_path_formula"},
            ],
        ),
        synthetic_rule(
            "stress_glidepath_unrounded_formula_variant",
            "Unrounded glide-path formula variant",
            "rule_non_si_glide_path_formula",
            rule_by_id,
            rule_type="formula_variant",
            relations=[
                {"type": "uses_parameter", "target": "ils_glidepath_angle_range"},
                {"type": "formula_variant_of", "target": "rule_non_si_glide_path_formula"},
            ],
        ),
        synthetic_rule(
            "stress_turn_radius_low_bank_cell_competitor",
            "Low-bank RF turn-radius cell competitor",
            "turn_radius_km_formula",
            rule_by_id,
            rule_type="piecewise_cell_variant",
            relations=[{"type": "formula_variant_of", "target": "turn_radius_km_formula"}],
        ),
        synthetic_rule(
            "stress_turn_radius_high_bank_cell_competitor",
            "High-bank RF turn-radius cell competitor",
            "turn_radius_km_formula",
            rule_by_id,
            rule_type="piecewise_cell_variant",
            relations=[{"type": "formula_variant_of", "target": "turn_radius_km_formula"}],
        ),
    ]


def stress_specs() -> list[dict[str, Any]]:
    return [
        {
            "task_id": "AVI_STRESS_01",
            "base_task_id": "AVI_OPT_08",
            "target_interaction": ["scenario-conditioned applicability"],
            "title": "Applicability stress: fixed-wing PBN intermediate segment",
            "engineering_task": "Select a fixed-wing PBN intermediate-segment length while rejecting the same-domain helicopter template.",
            "candidate": [
                "pbn_intermediate_segment_optimum_length",
                "pbn_intermediate_segment_optimum_length_helicopter",
                "pbn_intermediate_segment_stability_requirement",
            ],
            "final": [
                "pbn_intermediate_segment_optimum_length",
                "pbn_intermediate_segment_stability_requirement",
            ],
            "surplus_types": {
                "pbn_intermediate_segment_optimum_length_helicopter": "scenario_inapplicable_same_domain_rule"
            },
            "expected_cell_count": 0,
            "scenario_updates": {"aircraft_type": "fixed_wing", "segment_context": "pbn_intermediate"},
        },
        {
            "task_id": "AVI_STRESS_02",
            "base_task_id": "AVI_OPT_09",
            "target_interaction": ["scenario-conditioned applicability"],
            "title": "Applicability stress: helicopter PBN intermediate segment",
            "engineering_task": "Select a helicopter PBN intermediate-segment length while rejecting the fixed-wing template.",
            "candidate": [
                "pbn_intermediate_segment_optimum_length_helicopter",
                "pbn_intermediate_segment_optimum_length",
                "pbn_intermediate_segment_stability_requirement",
            ],
            "final": [
                "pbn_intermediate_segment_optimum_length_helicopter",
                "pbn_intermediate_segment_stability_requirement",
            ],
            "surplus_types": {"pbn_intermediate_segment_optimum_length": "scenario_inapplicable_same_domain_rule"},
            "expected_cell_count": 0,
            "scenario_updates": {"aircraft_type": "helicopter", "segment_context": "pbn_intermediate"},
        },
        {
            "task_id": "AVI_STRESS_03",
            "base_task_id": "AVI_OPT_19",
            "target_interaction": ["dependency", "parameter propagation / formula propagation"],
            "title": "Dependency stress: RF segment DTA formula closure",
            "engineering_task": "Choose RF segment length while preserving the dependency from minimum segment length to DTA computation.",
            "candidate": [
                "dta_minimum_segment_length_requirement",
                "dta_formula",
                "stress_dta_legacy_nominal_formula",
            ],
            "final": ["dta_minimum_segment_length_requirement", "dta_formula"],
            "surplus_types": {"stress_dta_legacy_nominal_formula": "dependency_support_or_dependency_variant"},
            "expected_cell_count": 0,
            "scenario_updates": {"formula_context": "dta", "rf_segment": True},
        },
        {
            "task_id": "AVI_STRESS_04",
            "base_task_id": "AVI_OPT_17",
            "target_interaction": ["dependency", "parameter propagation / formula propagation"],
            "title": "Dependency stress: RF turn-radius formula variants",
            "engineering_task": "Choose RF turn radius and bank angle while resolving formula dependencies and rejecting stale formula variants.",
            "candidate": [
                "minimum_turn_radius_rnp_constraint",
                "turn_radius_km_formula",
                "rf_segment_max_bank_angle_25deg",
                "turn_radius_nm_formula",
                "stress_turn_radius_legacy_low_speed_formula",
            ],
            "final": [
                "minimum_turn_radius_rnp_constraint",
                "turn_radius_km_formula",
                "rf_segment_max_bank_angle_25deg",
            ],
            "surplus_types": {
                "turn_radius_nm_formula": "dependency_support_or_dependency_variant",
                "stress_turn_radius_legacy_low_speed_formula": "parameter_variant_or_formula_variant",
            },
            "expected_cell_count": 3,
            "scenario_updates": {"procedure_type": "pbn", "segment_type": "rf_segment", "flight_level": "FL 180"},
        },
        {
            "task_id": "AVI_STRESS_05",
            "base_task_id": "AVI_OPT_13",
            "target_interaction": ["exclusion / alternative branch"],
            "title": "Exclusion stress: turning missed approach versus straight continuation",
            "engineering_task": "Design a turning missed-approach path while rejecting the mutually exclusive straight-continuation chain.",
            "candidate": [
                "RA-6.4.1-turn-angle-threshold",
                "RA-6.3.4-align-final-approach-track",
                "tnh_obstacle_clearance_requirement",
            ],
            "final": ["RA-6.4.1-turn-angle-threshold", "tnh_obstacle_clearance_requirement"],
            "surplus_types": {"RA-6.3.4-align-final-approach-track": "excluded_alternative_branch"},
            "expected_cell_count": 0,
            "scenario_updates": {"missed_approach_type": "turning", "requires_track_change": True},
        },
        {
            "task_id": "AVI_STRESS_06",
            "base_task_id": "AVI_OPT_07",
            "target_interaction": ["exclusion / alternative branch"],
            "title": "Exclusion stress: intermediate gradient maximum versus flat default",
            "engineering_task": "Select descent gradient and segment length under the active maximum-gradient rule while excluding the flat default.",
            "candidate": [
                "intermediate_approach_gradient_max",
                "intermediate_approach_gradient_flat_default",
                "moc_intermediate_approach_min",
            ],
            "final": ["intermediate_approach_gradient_max", "moc_intermediate_approach_min"],
            "surplus_types": {"intermediate_approach_gradient_flat_default": "excluded_alternative_branch"},
            "expected_cell_count": 3,
            "scenario_updates": {"approach_phase": "intermediate", "gradient_profile": "non_flat"},
        },
        {
            "task_id": "AVI_STRESS_07",
            "base_task_id": "AVI_OPT_18",
            "target_interaction": ["exception override", "parameter propagation / formula propagation"],
            "title": "Override stress: high-altitude RF bank-angle exception",
            "engineering_task": "Design a high-altitude RF turn where the 15-degree exception defeats the ordinary 25-degree bank rule.",
            "candidate": [
                "stress_rf_high_altitude_bank_exception_15deg",
                "rf_segment_max_bank_angle_25deg",
                "turn_radius_km_formula",
                "minimum_turn_radius_rnp_constraint",
                "turn_radius_nm_formula",
            ],
            "final": [
                "stress_rf_high_altitude_bank_exception_15deg",
                "turn_radius_km_formula",
                "minimum_turn_radius_rnp_constraint",
            ],
            "surplus_types": {
                "rf_segment_max_bank_angle_25deg": "defeated_by_override",
                "turn_radius_nm_formula": "parameter_variant_or_formula_variant",
            },
            "expected_cell_count": 3,
            "scenario_updates": {"procedure_type": "pbn", "segment_type": "turn", "flight_level": "FL 210"},
        },
        {
            "task_id": "AVI_STRESS_08",
            "base_task_id": "AVI_OPT_18",
            "target_interaction": ["exception override", "scenario-conditioned applicability"],
            "title": "Override stress: FL190 boundary bank-angle exception",
            "engineering_task": "Resolve a boundary high-altitude RF turn where the exception path must be selected over the baseline path.",
            "candidate": [
                "stress_rf_boundary_bank_exception_15deg",
                "rf_segment_max_bank_angle_25deg",
                "turn_radius_km_formula",
                "minimum_turn_radius_rnp_constraint",
                "stress_turn_radius_legacy_low_speed_formula",
            ],
            "final": [
                "stress_rf_boundary_bank_exception_15deg",
                "turn_radius_km_formula",
                "minimum_turn_radius_rnp_constraint",
            ],
            "surplus_types": {
                "rf_segment_max_bank_angle_25deg": "defeated_by_override",
                "stress_turn_radius_legacy_low_speed_formula": "parameter_variant_or_formula_variant",
            },
            "expected_cell_count": 3,
            "scenario_updates": {"procedure_type": "pbn", "segment_type": "turn", "flight_level": "FL 200"},
        },
        {
            "task_id": "AVI_STRESS_09",
            "base_task_id": "AVI_OPT_10",
            "target_interaction": ["precedence"],
            "title": "Precedence stress: CAAC GBAS segment-length source priority",
            "engineering_task": "Choose GBAS intermediate-segment distance when a CAAC source takes precedence over lower-priority competitors.",
            "candidate": [
                "stress_caac_gbas_length_precedence",
                "pbn_intermediate_segment_stability_requirement",
                "stress_icao_gbas_length_competitor",
                "stress_legacy_gbas_length_competitor",
            ],
            "final": ["stress_caac_gbas_length_precedence", "pbn_intermediate_segment_stability_requirement"],
            "surplus_types": {
                "stress_icao_gbas_length_competitor": "lower_priority_precedence_competitor",
                "stress_legacy_gbas_length_competitor": "lower_priority_precedence_competitor",
            },
            "expected_cell_count": 0,
            "scenario_updates": {"source_priority": "CAAC_over_ICAO", "navigation_source": "GBAS"},
        },
        {
            "task_id": "AVI_STRESS_10",
            "base_task_id": "AVI_OPT_04",
            "target_interaction": ["precedence"],
            "title": "Precedence stress: MSA DME arc overlay source priority",
            "engineering_task": "Select an MSA DME arc radius when a domestic overlay defeats lower-priority arc-radius interpretations.",
            "candidate": [
                "dme_arc_radius_range_sector_partition",
                "stress_caac_msa_arc_precedence",
                "stress_icao_msa_arc_competitor",
                "stress_environmental_overlay_arc_competitor",
            ],
            "final": ["dme_arc_radius_range_sector_partition", "stress_caac_msa_arc_precedence"],
            "surplus_types": {
                "stress_icao_msa_arc_competitor": "lower_priority_precedence_competitor",
                "stress_environmental_overlay_arc_competitor": "lower_priority_precedence_competitor",
            },
            "expected_cell_count": 0,
            "scenario_updates": {"source_priority": "CAAC_overlay", "procedure_type": "VOR/DME"},
        },
        {
            "task_id": "AVI_STRESS_11",
            "base_task_id": "AVI_OPT_05",
            "target_interaction": ["parameter propagation / formula propagation"],
            "title": "Parameter stress: glide-path angle formula propagation",
            "engineering_task": "Choose a glide-path angle while retaining the correct non-SI formula and rejecting publication-only formula variants.",
            "candidate": [
                "ils_glidepath_angle_range",
                "rule_non_si_glide_path_formula",
                "stress_glidepath_rounded_publication_variant",
                "stress_glidepath_unrounded_formula_variant",
            ],
            "final": ["ils_glidepath_angle_range", "rule_non_si_glide_path_formula"],
            "surplus_types": {
                "stress_glidepath_rounded_publication_variant": "parameter_variant_or_formula_variant",
                "stress_glidepath_unrounded_formula_variant": "parameter_variant_or_formula_variant",
            },
            "expected_cell_count": 0,
            "scenario_updates": {"formula_context": "non_si_glide_path", "publication_context": "design_computation"},
        },
        {
            "task_id": "AVI_STRESS_12",
            "base_task_id": "AVI_OPT_17",
            "target_interaction": ["parameter propagation / formula propagation", "dependency"],
            "title": "Formula stress: RF turn-radius piecewise cell competitors",
            "engineering_task": "Select RF turn radius and bank angle while avoiding partial mixing of piecewise turn-radius cells.",
            "candidate": [
                "minimum_turn_radius_rnp_constraint",
                "turn_radius_km_formula",
                "rf_segment_max_bank_angle_25deg",
                "stress_turn_radius_low_bank_cell_competitor",
                "stress_turn_radius_high_bank_cell_competitor",
            ],
            "final": [
                "minimum_turn_radius_rnp_constraint",
                "turn_radius_km_formula",
                "rf_segment_max_bank_angle_25deg",
            ],
            "surplus_types": {
                "stress_turn_radius_low_bank_cell_competitor": "piecewise_cell_competitor",
                "stress_turn_radius_high_bank_cell_competitor": "piecewise_cell_competitor",
            },
            "expected_cell_count": 3,
            "scenario_updates": {"procedure_type": "pbn", "segment_type": "rf_segment", "piecewise_radius_model": True},
        },
    ]


def update_query(base_query: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    query = copy.deepcopy(base_query)
    query["omega_id"] = spec["task_id"]
    query["title"] = spec["title"]
    query["design_intent"] = spec["engineering_task"]
    query["scenario_facts"].update(spec.get("scenario_updates", {}))
    query["stress_metadata"] = stress_metadata(spec)
    for idx, cell in enumerate(query.get("solver_constraint_cells", [])):
        cell["cell_id"] = f"{spec['task_id']}_cell_{idx + 1}"
    return query


def update_rule_label(base_label: dict[str, Any], spec: dict[str, Any], rule_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidate = sorted(spec["candidate"])
    final = sorted(spec["final"])
    surplus_types = spec["surplus_types"]
    label = copy.deepcopy(base_label)
    label["omega_id"] = spec["task_id"]
    label["title"] = spec["title"]
    label["scenario_facts"].update(spec.get("scenario_updates", {}))
    label["expected_source_rule_ids"] = candidate
    label["expected_defeated_rule_ids"] = sorted(
        rid
        for rid, typ in surplus_types.items()
        if typ in {"defeated_by_override", "lower_priority_precedence_competitor", "excluded_alternative_branch"}
    )
    label["expected_surviving_rule_ids"] = final
    label["expected_valid_rule_structures"] = [final]
    label["challenge_types"] = sorted(spec["target_interaction"])
    label["valid_constraint_cell_ids"] = [
        f"{spec['task_id']}_cell_{idx + 1}" for idx in range(int(spec.get("expected_cell_count", 0)))
    ]
    label["expected_rule_behavior"] = {
        "should_activate": final,
        "should_exclude": sorted(surplus_types),
        "should_resolve": sorted(spec["target_interaction"]),
    }
    label["stress_metadata"] = stress_metadata(spec)
    label["expected_provenance"] = provenance_for_rules(candidate, rule_by_id)
    return label


def update_feasible(base_feasible: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    feasible = copy.deepcopy(base_feasible)
    feasible["omega_id"] = spec["task_id"]
    feasible["title"] = spec["title"]
    feasible["scenario_facts"].update(spec.get("scenario_updates", {}))
    feasible["stress_metadata"] = stress_metadata(spec)
    for idx, cell in enumerate(feasible.get("valid_constraint_cells", [])):
        cell["cell_id"] = f"{spec['task_id']}_cell_{idx + 1}"
    return feasible


def provenance_for_rules(rule_ids: list[str], rule_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    chunk_ids: list[str] = []
    node_ids: list[str] = []
    edge_ids: list[str] = []
    documents: list[dict[str, Any]] = []
    for rule_id in rule_ids:
        rule = rule_by_id.get(rule_id, {})
        chunk_ids.extend(rule.get("source_chunk_ids") or [])
        node_ids.extend(rule.get("source_node_ids") or [])
        for constraint in rule.get("constraints", []):
            evidence = constraint.get("evidence") or {}
            edge_ids.extend(evidence.get("kg_edge_ids") or [])
        documents.extend(copy.deepcopy(rule.get("provenance") or []))
    return {
        "kg_chunk_ids": sorted(set(map(str, chunk_ids))),
        "kg_node_ids": sorted(set(map(str, node_ids))),
        "kg_edge_ids": sorted(set(map(str, edge_ids))),
        "source_documents": documents,
    }


def stress_metadata(spec: dict[str, Any]) -> dict[str, Any]:
    candidate = sorted(spec["candidate"])
    final = sorted(spec["final"])
    surplus = sorted(set(candidate) - set(final))
    return {
        "base_task_id": spec["base_task_id"],
        "target_interaction": list(spec["target_interaction"]),
        "engineering_task": spec["engineering_task"],
        "candidate_rule_ids": candidate,
        "final_valid_rule_ids": final,
        "valid_rule_structures_expected": [final],
        "structured_surplus_rule_ids": sorted(spec["surplus_types"]),
        "structured_surplus_types": dict(sorted(spec["surplus_types"].items())),
        "candidate_minus_final_rule_ids": surplus,
        "expected_cell_count": int(spec.get("expected_cell_count", 0)),
        "original_or_stress": "stress",
    }


def relation_counts(candidate_ids: list[str], rule_by_id: dict[str, dict[str, Any]]) -> dict[str, int]:
    candidate_set = set(candidate_ids)
    counts = {
        "dependency_edge_count": 0,
        "exclusion_edge_count": 0,
        "override_edge_count": 0,
        "precedence_edge_count": 0,
        "parameter_propagation_edge_count": 0,
    }
    for rule_id in candidate_ids:
        rule = rule_by_id.get(rule_id, {})
        for relation in rule.get("relations", []):
            target = relation.get("target") or relation.get("to") or relation.get("rule_id")
            if str(target) not in candidate_set:
                continue
            rel_type = str(relation.get("type", "")).lower().strip()
            if rel_type in DEPENDENCY_TYPES:
                counts["dependency_edge_count"] += 1
            if rel_type in EXCLUSION_TYPES:
                counts["exclusion_edge_count"] += 1
            if rel_type in OVERRIDE_TYPES:
                counts["override_edge_count"] += 1
            if rel_type in PRECEDENCE_TYPES:
                counts["precedence_edge_count"] += 1
            if rel_type in PARAMETER_TYPES:
                counts["parameter_propagation_edge_count"] += 1
    return counts


def quality_row(spec: dict[str, Any], rule_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidate = sorted(spec["candidate"])
    final = sorted(spec["final"])
    surplus = sorted(set(candidate) - set(final))
    structured = sorted(spec["surplus_types"])
    unstructured = sorted(set(surplus) - set(structured))
    ratio = len(candidate) / len(final) if final else 0.0
    row = {
        "task_id": spec["task_id"],
        "base_task_id": spec["base_task_id"],
        "target_interaction": "; ".join(spec["target_interaction"]),
        "candidate_rule_count": len(candidate),
        "final_valid_rule_count": len(final),
        "candidate_final_ratio": round(ratio, 3),
        "structured_surplus_count": len(structured),
        "structured_surplus_ratio": round(len(structured) / len(surplus), 3) if surplus else 0.0,
        "unstructured_noise_count": len(unstructured),
        "final_rules_subset_of_candidate": set(final) < set(candidate),
        "expected_cell_count": int(spec.get("expected_cell_count", 0)),
        "candidate_rule_ids": candidate,
        "final_valid_rule_ids": final,
        "structured_surplus_rule_ids": structured,
        "structured_surplus_types": spec["surplus_types"],
        "unstructured_noise_rule_ids": unstructured,
    }
    row.update(relation_counts(candidate, rule_by_id))
    return row


def manifest_row(spec: dict[str, Any], query: dict[str, Any], label: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": spec["task_id"],
        "base_task_id": spec["base_task_id"],
        "engineering_task": spec["engineering_task"],
        "target_interaction": "; ".join(spec["target_interaction"]),
        "candidate_rule_ids": sorted(spec["candidate"]),
        "final_valid_rule_ids": sorted(spec["final"]),
        "valid_rule_structures_expected": [sorted(spec["final"])],
        "structured_surplus_rule_ids": sorted(spec["surplus_types"]),
        "structured_surplus_types": spec["surplus_types"],
        "expected_cell_count": int(spec.get("expected_cell_count", 0)),
        "source_rules": label["expected_source_rule_ids"],
        "solver_constraints": [c.get("constraint_id") for c in query.get("solver_constraints", [])],
        "evidence_chunks": label["expected_provenance"].get("kg_chunk_ids", []),
        "task_file": str(TASK_DIR / f"{spec['task_id']}.json"),
    }


def build_reports(quality_rows: list[dict[str, Any]], manifest_rows: list[dict[str, Any]]) -> tuple[str, dict[str, Any], str]:
    by_interaction: dict[str, list[str]] = {}
    for row in manifest_rows:
        for interaction in str(row["target_interaction"]).split("; "):
            by_interaction.setdefault(interaction, []).append(row["task_id"])

    ratio_bad = [row["task_id"] for row in quality_rows if not (1.5 <= float(row["candidate_final_ratio"]) <= 3.5)]
    surplus_bad = [
        row["task_id"]
        for row in quality_rows
        if row["structured_surplus_count"] < 1 or float(row["structured_surplus_ratio"]) < 0.7
    ]
    candidate_equals_final = [
        row["task_id"] for row in quality_rows if row["candidate_rule_count"] == row["final_valid_rule_count"]
    ]
    subset_bad = [row["task_id"] for row in quality_rows if not row["final_rules_subset_of_candidate"]]
    interaction_bad = [name for name, tasks in by_interaction.items() if len(tasks) < 2]

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_stress_tasks": len(quality_rows),
        "interaction_coverage": by_interaction,
        "candidate_final_ratio_not_in_target": ratio_bad,
        "structured_surplus_insufficient": surplus_bad,
        "candidate_equals_final": candidate_equals_final,
        "final_subset_failures": subset_bad,
        "interactions_with_fewer_than_two_tasks": interaction_bad,
        "direct_pipeline_readiness": {
            "same_schema_as_existing_layers": True,
            "stress_optimization_queries": str(OUT_QUERY_PATH),
            "stress_rule_structure_labels": str(OUT_RULE_LABEL_PATH),
            "stress_feasible_region_labels": str(OUT_FEASIBLE_PATH),
            "combined_rule_library_with_synthetic_stress_rules": str(OUT_COMBINED_RULE_LIBRARY_PATH),
        },
    }

    lines = [
        "# Aviation Stress Task Quality Validation",
        "",
        "## Purpose",
        "",
        "The stress subset widens grounded candidate rule sets with structured surplus rules: competing, defeated, excluded, lower-priority, dependency-related, or formula-variant rules. The goal is to test resolver behavior rather than to add unrelated random rules.",
        "",
        "## Interaction Coverage",
        "",
    ]
    for interaction, tasks in sorted(by_interaction.items()):
        lines.append(f"- {interaction}: {', '.join(tasks)}")
    lines.extend(
        [
            "",
            "## Quality Flags",
            "",
            f"- Candidate/final ratio outside 1.5-3.5: {', '.join(ratio_bad) if ratio_bad else 'none'}",
            f"- Structured surplus insufficient: {', '.join(surplus_bad) if surplus_bad else 'none'}",
            f"- Candidate equals final: {', '.join(candidate_equals_final) if candidate_equals_final else 'none'}",
            f"- Final not a strict subset of candidate: {', '.join(subset_bad) if subset_bad else 'none'}",
            f"- Interactions covered by fewer than two tasks: {', '.join(interaction_bad) if interaction_bad else 'none'}",
            "",
            "## Per-Task Quality",
            "",
            "| task_id | target_interaction | candidate | final | ratio | structured surplus | unstructured noise | cells |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in quality_rows:
        lines.append(
            "| {task_id} | {target_interaction} | {candidate_rule_count} | {final_valid_rule_count} | {candidate_final_ratio} | {structured_surplus_count} | {unstructured_noise_count} | {expected_cell_count} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Pipeline Readiness",
            "",
            "The generated layer files keep the same `items` schema as the existing aviation benchmark. Tasks that reference synthetic stress rules should be evaluated with the combined stress rule library so those derived rule records are available to CTHR, ASP, and SMT baselines.",
            "",
        ]
    )

    generation_lines = [
        "# Aviation Stress Task Generation Report",
        "",
        f"Generated {len(manifest_rows)} interaction-rich aviation stress tasks.",
        "",
        "| task_id | base_task_id | target_interaction | structured surplus |",
        "| --- | --- | --- | --- |",
    ]
    for row in manifest_rows:
        generation_lines.append(
            "| {task_id} | {base_task_id} | {target_interaction} | {surplus} |".format(
                task_id=row["task_id"],
                base_task_id=row["base_task_id"],
                target_interaction=row["target_interaction"],
                surplus=", ".join(row["structured_surplus_rule_ids"]),
            )
        )
    generation_lines.extend(
        [
            "",
            "Synthetic stress rules are marked with `synthetic_stress_rule=true` in the stress rule-library extension and the combined stress rule library.",
            "",
        ]
    )
    return "\n".join(lines), summary, "\n".join(generation_lines)


def main() -> None:
    base_queries = {item["omega_id"]: item for item in read_json(BASE_QUERY_PATH)["items"]}
    base_labels = {item["omega_id"]: item for item in read_json(BASE_RULE_LABEL_PATH)["items"]}
    base_feasible = {item["omega_id"]: item for item in read_json(BASE_FEASIBLE_PATH)["items"]}
    base_library = read_json(BASE_RULE_LIBRARY_PATH)
    base_rules = copy.deepcopy(base_library.get("rules", []))
    rule_by_id = {str(rule["rule_id"]): rule for rule in base_rules if rule.get("rule_id")}
    synthetic_rules = make_synthetic_rules(rule_by_id)
    specs = stress_specs()
    synthetic_usage: dict[str, set[str]] = {
        str(rule["rule_id"]): set() for rule in synthetic_rules if rule.get("rule_id")
    }
    for spec in specs:
        used_rule_ids = set(spec["candidate"]) | set(spec["final"]) | set(spec["surplus_types"])
        for rule_id in used_rule_ids & set(synthetic_usage):
            synthetic_usage[rule_id].add(spec["base_task_id"])
    for rule in synthetic_rules:
        rule["derived_from_base_task"] = sorted(synthetic_usage.get(str(rule["rule_id"]), set()))
    for rule in synthetic_rules:
        rule_by_id[str(rule["rule_id"])] = rule

    queries: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    feasible_items: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []

    TASK_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for spec in specs:
        base_task_id = spec["base_task_id"]
        query = update_query(base_queries[base_task_id], spec)
        label = update_rule_label(base_labels[base_task_id], spec, rule_by_id)
        feasible = update_feasible(base_feasible[base_task_id], spec)
        queries.append(query)
        labels.append(label)
        feasible_items.append(feasible)
        manifest_rows.append(manifest_row(spec, query, label))
        quality_rows.append(quality_row(spec, rule_by_id))
        write_json(
            TASK_DIR / f"{spec['task_id']}.json",
            {
                "version": "aviation_stress_task_v1",
                "task": query,
                "rule_structure_label": label,
                "feasible_region_label": feasible,
                "stress_metadata": stress_metadata(spec),
            },
        )

    combined_library = copy.deepcopy(base_library)
    combined_library["rules"] = base_rules + synthetic_rules
    combined_library["stress_extension"] = {
        "synthetic_stress_rule_count": len(synthetic_rules),
        "synthetic_stress_rule_ids": [rule["rule_id"] for rule in synthetic_rules],
        "base_rule_library": str(BASE_RULE_LIBRARY_PATH),
    }

    write_json(OUT_QUERY_PATH, {"version": "aviation_stress_benchmark_layers_v1", "items": queries})
    write_json(OUT_RULE_LABEL_PATH, {"version": "aviation_stress_benchmark_layers_v1", "items": labels})
    write_json(OUT_FEASIBLE_PATH, {"version": "aviation_stress_benchmark_layers_v1", "items": feasible_items})
    write_json(
        OUT_RULE_EXTENSION_PATH,
        {
            "version": "aviation_stress_rule_library_extension_v1",
            "rules": synthetic_rules,
        },
    )
    write_json(OUT_COMBINED_RULE_LIBRARY_PATH, combined_library)
    write_json(
        OUT_MANIFEST_PATH,
        {
            "version": "aviation_stress_benchmark_layers_v1",
            "num_stress_tasks": len(queries),
            "files": {
                "optimization_queries": str(OUT_QUERY_PATH),
                "rule_structure_labels": str(OUT_RULE_LABEL_PATH),
                "feasible_region_labels": str(OUT_FEASIBLE_PATH),
                "rule_library_extension": str(OUT_RULE_EXTENSION_PATH),
                "combined_rule_library": str(OUT_COMBINED_RULE_LIBRARY_PATH),
                "task_directory": str(TASK_DIR),
            },
        },
    )

    manifest_headers = [
        "task_id",
        "base_task_id",
        "engineering_task",
        "target_interaction",
        "candidate_rule_ids",
        "final_valid_rule_ids",
        "valid_rule_structures_expected",
        "structured_surplus_rule_ids",
        "structured_surplus_types",
        "expected_cell_count",
        "source_rules",
        "solver_constraints",
        "evidence_chunks",
        "task_file",
    ]
    quality_headers = [
        "task_id",
        "base_task_id",
        "target_interaction",
        "candidate_rule_count",
        "final_valid_rule_count",
        "candidate_final_ratio",
        "structured_surplus_count",
        "structured_surplus_ratio",
        "unstructured_noise_count",
        "final_rules_subset_of_candidate",
        "dependency_edge_count",
        "exclusion_edge_count",
        "override_edge_count",
        "precedence_edge_count",
        "parameter_propagation_edge_count",
        "expected_cell_count",
        "candidate_rule_ids",
        "final_valid_rule_ids",
        "structured_surplus_rule_ids",
        "structured_surplus_types",
        "unstructured_noise_rule_ids",
    ]
    write_csv(OUT_TASK_MANIFEST_CSV, manifest_rows, manifest_headers)
    write_csv(OUT_QUALITY_CSV, quality_rows, quality_headers)

    report_md, summary_json, generation_report_md = build_reports(quality_rows, manifest_rows)
    write_json(OUT_QUALITY_SUMMARY_JSON, summary_json)
    OUT_QUALITY_REPORT_MD.write_text(report_md, encoding="utf-8")
    OUT_GENERATION_REPORT_MD.write_text(generation_report_md, encoding="utf-8")

    print(
        json.dumps(
            {
                "stress_tasks": len(queries),
                "task_dir": str(TASK_DIR),
                "manifest_csv": str(OUT_TASK_MANIFEST_CSV),
                "quality_csv": str(OUT_QUALITY_CSV),
                "quality_report": str(OUT_QUALITY_REPORT_MD),
                "ratio_flags": summary_json["candidate_final_ratio_not_in_target"],
                "surplus_flags": summary_json["structured_surplus_insufficient"],
                "candidate_equals_final": summary_json["candidate_equals_final"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
