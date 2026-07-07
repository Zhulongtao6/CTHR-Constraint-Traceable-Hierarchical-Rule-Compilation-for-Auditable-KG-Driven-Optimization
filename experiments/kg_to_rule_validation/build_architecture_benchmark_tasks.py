from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
CTHR_ROOT = THIS_DIR.parents[1]
PAPER_DIR = CTHR_ROOT / "paper"
KG_EXPORT_DIR = Path(r"D:\paper\Neurosymbolic\cognee_export\architecture_export_neo4j")
OUT_DIR = PAPER_DIR / "architecture_generated_benchmark_layers"
TASK_DIR = OUT_DIR / "tasks"
RESULTS_DIR = PAPER_DIR / "results"

OUT_TASKS_PATH = OUT_DIR / "architecture_optimization_queries.json"
OUT_RULE_LABELS_PATH = OUT_DIR / "architecture_rule_structure_labels.json"
OUT_FEASIBLE_PATH = OUT_DIR / "architecture_feasible_region_labels.json"
OUT_RULE_LIBRARY_PATH = OUT_DIR / "architecture_stress_rule_library.combined.json"
OUT_MANIFEST_PATH = OUT_DIR / "architecture_benchmark_layers_manifest.json"

OUT_TASK_MANIFEST_CSV = RESULTS_DIR / "architecture_task_manifest.csv"
OUT_QUALITY_CSV = RESULTS_DIR / "architecture_quality_validation.csv"
OUT_QUALITY_SUMMARY_JSON = RESULTS_DIR / "architecture_quality_validation_summary.json"
OUT_QUALITY_REPORT_MD = RESULTS_DIR / "architecture_quality_validation_report.md"


DATASET_DOMAINS = {
    "arch_ada_native": {
        "source_domain": "ADA",
        "document": "2010 ADA Standards for Accessible Design",
    },
    "arch_ibc_native": {
        "source_domain": "IBC",
        "document": "2021 International Building Code",
    },
    "arch_ifc_native": {
        "source_domain": "IFC",
        "document": "2021 International Fire Code",
    },
}

RELATION_GROUPS = {
    "dependency": {"depends_on", "requires"},
    "exclusion": {"excludes", "mutually_exclusive", "conflicts_with"},
    "override": {"overrides", "defeats"},
    "precedence": {"precedes", "has_precedence_over"},
    "parameter": {"uses_parameter", "formula_variant_of", "propagates_to", "piecewise_variant_of"},
}

SURPLUS_TO_KIND = {
    "scenario_inapplicable_same_domain_rule": "applicability",
    "dependency_support_or_dependency_variant": "dependency",
    "excluded_alternative_branch": "exclusion",
    "defeated_by_override": "override",
    "lower_priority_precedence_competitor": "precedence",
    "parameter_variant_or_formula_variant": "parameter",
    "piecewise_cell_competitor": "parameter",
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({header: csv_cell(row.get(header)) for header in headers})


def csv_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(list(value) if isinstance(value, set) else value, ensure_ascii=False)
    if value is None:
        return ""
    return value


def normalize_dataset_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


class KgEvidenceIndex:
    def __init__(self, root: Path) -> None:
        self.root = root
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        self.dataset_names = {
            normalize_dataset_key(k): v for k, v in summary.get("datasets", {}).items()
        }
        self.chunks_by_domain: dict[str, list[dict[str, str]]] = {"ADA": [], "IBC": [], "IFC": []}
        with (root / "nodes.csv").open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("label") != "DocumentChunk":
                    continue
                dataset_name = self.dataset_names.get(normalize_dataset_key(str(row.get("dataset", ""))))
                if dataset_name not in DATASET_DOMAINS:
                    continue
                domain = DATASET_DOMAINS[dataset_name]["source_domain"]
                self.chunks_by_domain[domain].append(
                    {
                        "id": str(row.get("id", "")),
                        "dataset_name": dataset_name,
                        "domain": domain,
                        "text": str(row.get("text_snippet", "")),
                    }
                )

    def find(self, domain: str, keywords: list[str]) -> dict[str, str]:
        chunks = self.chunks_by_domain.get(domain, [])
        if not chunks:
            raise RuntimeError(f"No KG chunks available for domain {domain}")
        normalized_keywords = [keyword.lower() for keyword in keywords if keyword]
        best = None
        best_score = -1
        for chunk in chunks:
            text = chunk["text"].lower()
            score = sum(1 for keyword in normalized_keywords if keyword in text)
            if score > best_score:
                best = chunk
                best_score = score
        return best or chunks[0]


def guard(*clauses: dict[str, Any]) -> dict[str, Any]:
    return {"all": list(clauses)}


def clause(field: str, op: str, value: Any) -> dict[str, Any]:
    return {"field": field, "op": op, "value": value}


def constraint(
    cid: str,
    expression: str,
    role: str,
    source_id: str,
    decision_variables: list[str],
    scenario_fields: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "constraint_id": cid,
        "expression": expression,
        "role": role,
        "source_type": "rule_library",
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


def scenario_constraint(
    cid: str,
    expression: str,
    role: str,
    source_id: str,
    decision_variables: list[str],
    scenario_fields: list[str] | None = None,
) -> dict[str, Any]:
    item = constraint(cid, expression, role, source_id, decision_variables, scenario_fields)
    item["source_type"] = "task_or_scenario_model"
    return item


def rule(
    rule_id: str,
    name: str,
    source_domain: str,
    clause_id: str,
    keywords: list[str],
    kg: KgEvidenceIndex,
    *,
    rule_type: str = "requirement",
    guard_obj: dict[str, Any] | None = None,
    constraints: list[dict[str, Any]] | None = None,
    relations: list[dict[str, Any]] | None = None,
    conflict_class: str | None = None,
    synthetic: bool = False,
    derived_from_source_rule: str | None = None,
    derived_from_base_task: str | None = None,
) -> dict[str, Any]:
    evidence = kg.find(source_domain, keywords)
    doc = DATASET_DOMAINS[evidence["dataset_name"]]["document"]
    chunk_id = evidence["id"]
    out = {
        "rule_id": rule_id,
        "name": name,
        "domain": "architecture",
        "source_domain": source_domain,
        "rule_type": rule_type,
        "source_chunk_ids": [chunk_id],
        "source_node_ids": [chunk_id],
        "guard": guard_obj or {"all": []},
        "constraints": constraints or [],
        "relations": relations or [],
        "provenance": [
            {
                "document": doc,
                "section": clause_id,
                "page": "unknown",
                "chunk_id": chunk_id,
            }
        ],
        "extraction_notes": "Rule template grounded to the current architecture KG Neo4j export by document chunk evidence.",
        "confidence": 1.0,
    }
    if conflict_class:
        out["conflict_class"] = conflict_class
    if synthetic:
        out["synthetic_stress_rule"] = True
        out["derived_from_source_rule"] = derived_from_source_rule
        out["derived_from_base_task"] = [derived_from_base_task] if derived_from_base_task else []
        out["extraction_notes"] = (
            "Synthetic stress rule derived from a source-grounded architecture rule to activate "
            "a specific resolver interaction; it is not presented as an original source clause."
        )
    return out


def relation(rel_type: str, target: str) -> dict[str, str]:
    return {"type": rel_type, "target": target}


def rule_library_specs() -> list[dict[str, Any]]:
    # Compact source-grounded rule templates. Provenance is bound to real KG chunks
    # at generation time; these IDs are then used by the task definitions.
    return [
        # ADA accessibility
        dict(id="ADA_PARKING_CAR_WIDTH", name="Accessible car parking width", domain="ADA", clause="ADA 502.2", kw=["parking", "access aisle"], var="parking_space_width_in", op=">=", val=96, unit="inch", guard=guard(clause("facility", "eq", "parking"), clause("vehicle_type", "eq", "car"))),
        dict(id="ADA_PARKING_VAN_WIDTH", name="Accessible van parking width template", domain="ADA", clause="ADA 502.2", kw=["parking", "van"], var="parking_space_width_in", op=">=", val=132, unit="inch", guard=guard(clause("facility", "eq", "parking"), clause("vehicle_type", "eq", "van"))),
        dict(id="ADA_PARKING_ACCESS_AISLE", name="Accessible parking access aisle", domain="ADA", clause="ADA 502.3", kw=["parking", "access aisle"], var="access_aisle_width_in", op=">=", val=60, unit="inch", rel=[relation("depends_on", "ADA_PARKING_CAR_WIDTH")]),
        dict(id="ADA_DOOR_CLEAR_WIDTH", name="Accessible door clear width", domain="ADA", clause="ADA 404.2.3", kw=["door", "clear opening"], var="door_clear_width_in", op=">=", val=32, unit="inch", guard=guard(clause("route_type", "eq", "accessible_route"))),
        dict(id="ADA_DOOR_STORAGE_EXCEPTION", name="Storage closet door exception", domain="ADA", clause="ADA/IBC door exception", kw=["door openings", "storage closets"], var="door_clear_width_in", op=">=", val=28, unit="inch", guard=guard(clause("space_type", "eq", "storage_closet")), rel=[relation("overrides", "ADA_DOOR_CLEAR_WIDTH")]),
        dict(id="ADA_DOOR_APPROACH_CLEARANCE", name="Door maneuvering clearance", domain="ADA", clause="ADA 404.2.4", kw=["door", "clearance"], var="maneuvering_clearance_in", op=">=", val=18, unit="inch", rel=[relation("depends_on", "ADA_DOOR_CLEAR_WIDTH")]),
        dict(id="ADA_RAMP_SLOPE_MAX", name="Ramp maximum running slope", domain="ADA", clause="ADA 405.2", kw=["ramp", "slope"], var="ramp_slope_ratio", op="<=", val=0.0833, unit="ratio", guard=guard(clause("element_type", "eq", "ramp"))),
        dict(id="ADA_RAMP_RISE_MAX", name="Ramp maximum rise per run", domain="ADA", clause="ADA 405.6", kw=["ramp", "rise"], var="ramp_rise_in", op="<=", val=30, unit="inch", rel=[relation("depends_on", "ADA_RAMP_SLOPE_MAX")]),
        dict(id="ADA_RAMP_LANDING_LENGTH", name="Ramp landing length", domain="ADA", clause="ADA 405.7", kw=["ramp", "landing"], var="landing_length_in", op=">=", val=60, unit="inch", rel=[relation("depends_on", "ADA_RAMP_SLOPE_MAX")]),
        dict(id="ADA_TURNING_SPACE_CIRCLE", name="Circular turning space template", domain="ADA", clause="ADA 304.3.1", kw=["turning space"], var="turning_diameter_in", op=">=", val=60, unit="inch", conflict="turning_space_template", guard=guard(clause("turning_geometry", "eq", "circular_clear_floor_area"))),
        dict(id="ADA_TURNING_SPACE_T_SHAPE", name="T-shaped turning space template", domain="ADA", clause="ADA 304.3.2", kw=["turning space"], var="t_arm_width_in", op=">=", val=36, unit="inch", conflict="turning_space_template", guard=guard(clause("turning_geometry", "eq", "t_shaped_clear_floor_area"))),
        dict(id="ADA_SHOWER_STANDARD_ROLL_IN", name="Standard roll-in shower template", domain="ADA", clause="ADA 608.2.2", kw=["roll-in shower"], var="shower_width_in", op=">=", val=30, unit="inch", conflict="shower_template", guard=guard(clause("available_shower_depth_in", "gte", 60))),
        dict(id="ADA_SHOWER_STANDARD_DEPTH", name="Standard roll-in shower depth", domain="ADA", clause="ADA 608.2.2", kw=["roll-in shower"], var="shower_depth_in", op=">=", val=60, unit="inch", rel=[relation("depends_on", "ADA_SHOWER_STANDARD_ROLL_IN")], conflict="shower_template"),
        dict(id="ADA_SHOWER_ALTERNATE_ROLL_IN", name="Alternate roll-in shower template", domain="ADA", clause="ADA 608.2.3", kw=["roll-in shower"], var="shower_width_in", op=">=", val=36, unit="inch", conflict="shower_template", guard=guard(clause("available_shower_depth_in", "lt", 60))),
        dict(id="ADA_SHOWER_ALTERNATE_DEPTH", name="Alternate roll-in shower depth", domain="ADA", clause="ADA 608.2.3", kw=["roll-in shower"], var="shower_depth_in", op=">=", val=36, unit="inch", rel=[relation("depends_on", "ADA_SHOWER_ALTERNATE_ROLL_IN")], conflict="shower_template"),
        dict(id="ADA_TOILET_STANDARD_STALL", name="Accessible toilet standard stall template", domain="ADA", clause="ADA 604.8.1.1", kw=["toilet compartment"], var="stall_width_in", op=">=", val=60, unit="inch", conflict="toilet_template", guard=guard(clause("wheelchair_access_required", "eq", True))),
        dict(id="ADA_TOILET_AMBULATORY_STALL", name="Ambulatory toilet stall template", domain="ADA", clause="ADA 604.8.2", kw=["toilet compartment"], var="stall_width_in", op=">=", val=35, unit="inch", conflict="toilet_template", guard=guard(clause("ambulatory_stall_requested", "eq", True))),
        dict(id="ADA_HISTORIC_MAX_FEASIBLE", name="Historic property maximum-feasible accessibility", domain="ADA", clause="ADA 35.151", kw=["historic", "maximum extent feasible"], var="historic_entry_clear_width_in", op=">=", val=32, unit="inch", guard=guard(clause("historic_property", "eq", True))),
        dict(id="ADA_HISTORIC_ALTERNATIVE_ACCESS", name="Historic property alternative access exception", domain="ADA", clause="ADA 35.151", kw=["historic", "alternative access"], var="alternative_service_distance_ft", op="<=", val=200, unit="ft", rel=[relation("overrides", "ADA_DOOR_CLEAR_WIDTH")], guard=guard(clause("physical_access_threatens_historic_significance", "eq", True))),
        dict(id="ADA_EQUIVALENT_FACILITATION", name="Equivalent facilitation alternative", domain="ADA", clause="ADA 103", kw=["equivalent facilitation"], var="equivalent_route_width_in", op=">=", val=36, unit="inch", guard=guard(clause("equivalent_facilitation", "eq", True))),
        # IBC egress
        dict(id="IBC_OCC_LOAD_FACTOR_BUSINESS", name="Business occupancy load factor", domain="IBC", clause="IBC 1004", kw=["occupant load"], var="occupant_load_factor_sf_per_person", op=">=", val=150, unit="sf/person", guard=guard(clause("occupancy_group", "eq", "B"))),
        dict(id="IBC_OCC_LOAD_FACTOR_ASSEMBLY", name="Assembly occupancy load factor", domain="IBC", clause="IBC 1004", kw=["occupant load", "assembly"], var="occupant_load_factor_sf_per_person", op=">=", val=15, unit="sf/person", guard=guard(clause("occupancy_group", "eq", "A"))),
        dict(id="IBC_EGRESS_WIDTH_FACTOR_STAIR", name="Stair egress width factor", domain="IBC", clause="IBC 1005", kw=["means of egress", "stair"], var="egress_width_in", op=">=", val=44, unit="inch", rel=[relation("uses_parameter", "IBC_OCC_LOAD_FACTOR_BUSINESS")]),
        dict(id="IBC_EGRESS_WIDTH_FACTOR_LEVEL", name="Level egress width factor", domain="IBC", clause="IBC 1005", kw=["means of egress"], var="egress_width_in", op=">=", val=36, unit="inch", rel=[relation("uses_parameter", "IBC_OCC_LOAD_FACTOR_BUSINESS")]),
        dict(id="IBC_TRAVEL_DISTANCE_SPRINKLERED", name="Sprinklered travel-distance allowance", domain="IBC", clause="IBC 1017", kw=["travel distance", "sprinkler"], var="travel_distance_ft", op="<=", val=250, unit="ft", guard=guard(clause("sprinklered", "eq", True))),
        dict(id="IBC_TRAVEL_DISTANCE_UNSPRINKLERED", name="Unsprinklered travel-distance allowance", domain="IBC", clause="IBC 1017", kw=["travel distance"], var="travel_distance_ft", op="<=", val=200, unit="ft", guard=guard(clause("sprinklered", "eq", False))),
        dict(id="IBC_TWO_EXITS_REQUIRED", name="Two exits required when load/common path threshold exceeded", domain="IBC", clause="IBC 1006", kw=["two exits", "common path"], var="number_of_exits", op=">=", val=2, unit="count", rel=[relation("depends_on", "IBC_COMMON_PATH_LIMIT")]),
        dict(id="IBC_COMMON_PATH_LIMIT", name="Common path of egress travel limit", domain="IBC", clause="IBC 1006.2.1", kw=["common path"], var="common_path_ft", op="<=", val=75, unit="ft"),
        dict(id="IBC_MIXED_NONSEPARATED_OCCUPANCY", name="Nonseparated mixed-occupancy egress path", domain="IBC", clause="IBC 508", kw=["mixed occupancy"], var="fire_separation_rating_hr", op=">=", val=0, unit="hour", conflict="mixed_occupancy_path", guard=guard(clause("fire_separation_wall_available", "eq", False))),
        dict(id="IBC_MIXED_SEPARATED_OCCUPANCY", name="Separated mixed-occupancy egress path", domain="IBC", clause="IBC 508", kw=["mixed occupancy", "fire separation"], var="fire_separation_rating_hr", op=">=", val=2, unit="hour", conflict="mixed_occupancy_path", guard=guard(clause("fire_separation_wall_available", "eq", True))),
        dict(id="IBC_HIGH_RISE_SPRINKLER_REQUIRED", name="High-rise sprinkler requirement", domain="IBC", clause="IBC 403", kw=["high-rise", "sprinkler"], var="sprinkler_coverage_ratio", op=">=", val=1.0, unit="ratio", guard=guard(clause("high_rise", "eq", True))),
        dict(id="IBC_ACCESSIBLE_ROUTE_GENERAL", name="IBC accessible route general requirement", domain="IBC", clause="IBC 1104", kw=["accessible route"], var="accessible_route_width_in", op=">=", val=36, unit="inch"),
        dict(id="IBC_DOOR_EGRESS_WIDTH", name="IBC means-of-egress door width", domain="IBC", clause="IBC 1010", kw=["door opening", "means of egress"], var="door_clear_width_in", op=">=", val=32, unit="inch"),
        dict(id="IBC_EXISTING_BUILDING_EXCEPTION", name="Existing-building scoped egress exception", domain="IBC", clause="IBC existing building", kw=["existing buildings"], var="egress_retrofit_area_sf", op="<=", val=2500, unit="sf", guard=guard(clause("building_status", "eq", "existing")), rel=[relation("overrides", "IBC_EGRESS_WIDTH_FACTOR_STAIR")]),
        # IFC fire safety
        dict(id="IFC_SPRINKLER_GENERAL", name="IFC automatic sprinkler system condition", domain="IFC", clause="IFC 903", kw=["automatic sprinkler"], var="sprinkler_coverage_ratio", op=">=", val=1.0, unit="ratio", guard=guard(clause("sprinkler_required", "eq", True))),
        dict(id="IFC_FIRE_ALARM_INSTALL_PERMIT", name="Fire alarm installation permit", domain="IFC", clause="IFC 105.6.6", kw=["fire alarm", "installation"], var="alarm_notification_appliance_count", op=">=", val=1, unit="count", guard=guard(clause("system_type", "eq", "fire_alarm"), clause("activity", "eq", "installation"))),
        dict(id="IFC_FIRE_ALARM_MAINTENANCE_EXCEPTION", name="Fire alarm maintenance exception", domain="IFC", clause="IFC 105.6.6", kw=["fire alarm", "maintenance"], var="devices_serviced_count", op=">=", val=1, unit="count", rel=[relation("overrides", "IFC_FIRE_ALARM_INSTALL_PERMIT")], guard=guard(clause("activity", "eq", "maintenance"))),
        dict(id="IFC_HAZARD_STORAGE_BASELINE", name="Hazardous material storage baseline limit", domain="IFC", clause="IFC Chapter 50", kw=["hazardous materials"], var="hazard_quantity_l", op="<=", val=60, unit="L", guard=guard(clause("hazard_storage", "eq", True))),
        dict(id="IFC_HAZARD_CLOSED_CONTAINER_EXCEPTION", name="Closed-container hazardous storage exception", domain="IFC", clause="IFC Chapter 50", kw=["hazardous materials", "storage"], var="hazard_quantity_l", op="<=", val=120, unit="L", rel=[relation("overrides", "IFC_HAZARD_STORAGE_BASELINE")], guard=guard(clause("storage_type", "eq", "closed_container"))),
        dict(id="IFC_CONTROL_AREA_LIMIT", name="Control-area hazardous quantity limit", domain="IFC", clause="IFC 5003", kw=["control area"], var="hazard_quantity_l", op="<=", val=240, unit="L", rel=[relation("depends_on", "IFC_HAZARD_STORAGE_BASELINE")]),
        dict(id="IFC_SUPPRESSION_ALTERNATIVE", name="Alternative fire-suppression template", domain="IFC", clause="IFC 904", kw=["alternative", "fire-extinguishing"], var="suppression_coverage_area_sf", op=">=", val=1200, unit="sf", conflict="fire_suppression_path", guard=guard(clause("water_supply_available", "eq", False))),
        dict(id="IFC_SPRINKLER_TEMPLATE", name="Automatic sprinkler template", domain="IFC", clause="IFC 903", kw=["automatic sprinkler"], var="sprinkler_coverage_ratio", op=">=", val=1.0, unit="ratio", conflict="fire_suppression_path", guard=guard(clause("water_supply_available", "eq", True))),
        dict(id="IFC_FIRE_SEPARATION_HAZARD", name="Hazardous storage fire separation", domain="IFC", clause="IFC Chapter 50", kw=["hazardous materials", "separate"], var="fire_separation_rating_hr", op=">=", val=2, unit="hour", rel=[relation("depends_on", "IFC_HAZARD_STORAGE_BASELINE")]),
        dict(id="IFC_EXISTING_BUILDING_CH11", name="Existing building fire-safety chapter applicability", domain="IFC", clause="IFC Chapter 11", kw=["existing buildings"], var="temporary_impairment_hours", op="<=", val=8, unit="hour", guard=guard(clause("building_status", "eq", "existing"))),
        # Cross-code precedence markers and formula variants
        dict(id="XCODE_ADA_OVER_IBC_ACCESSIBLE_ROUTE", name="ADA accessible-route precedence over general IBC route", domain="ADA", clause="ADA 105 / IBC 1104", kw=["accessible route"], var="accessible_route_width_in", op=">=", val=36, unit="inch", rel=[relation("precedes", "IBC_ACCESSIBLE_ROUTE_GENERAL")]),
        dict(id="XCODE_IFC_HAZARD_OVER_IBC_MIXED_USE", name="IFC hazard-storage precedence over mixed-use egress relaxation", domain="IFC", clause="IFC/IBC hazardous occupancy", kw=["hazardous materials", "occupancy"], var="fire_separation_rating_hr", op=">=", val=2, unit="hour", rel=[relation("precedes", "IBC_MIXED_NONSEPARATED_OCCUPANCY")]),
        dict(id="XCODE_IBC_CH10_OVER_IFC_GENERAL_EGRESS", name="IBC Chapter 10 egress precedence marker", domain="IBC", clause="IBC/IFC Chapter 10", kw=["means of egress"], var="egress_width_in", op=">=", val=44, unit="inch", rel=[relation("precedes", "IFC_GENERAL_EGRESS_MAINTENANCE")]),
        dict(id="IFC_GENERAL_EGRESS_MAINTENANCE", name="IFC general operational egress maintenance rule", domain="IFC", clause="IFC means of egress", kw=["means of egress"], var="egress_width_in", op=">=", val=36, unit="inch"),
        dict(id="XCODE_ADA_HISTORIC_OVER_IBC_EXISTING", name="ADA historic-accessibility precedence over IBC existing-building exception", domain="ADA", clause="ADA historic / IBC existing", kw=["historic", "existing"], var="historic_entry_clear_width_in", op=">=", val=32, unit="inch", rel=[relation("precedes", "IBC_EXISTING_BUILDING_EXCEPTION")]),
        dict(id="XCODE_IFC_ALARM_OVER_IBC_OCCUPANCY", name="IFC alarm permit precedence over occupancy-only planning", domain="IFC", clause="IFC 105.6.6 / IBC Chapter 3", kw=["fire alarm"], var="alarm_notification_appliance_count", op=">=", val=1, unit="count", rel=[relation("precedes", "IBC_2021_OCCUPANCY_ONLY_COMPETITOR")]),
        dict(id="IBC_2021_OCCUPANCY_ONLY_COMPETITOR", name="IBC occupancy-only lower-priority competitor", domain="IBC", clause="IBC Chapter 3", kw=["occupant load"], var="alarm_notification_appliance_count", op=">=", val=0, unit="count"),
        dict(id="FORMULA_OCC_LOAD_AREA_DIV_FACTOR", name="Occupant load area divided by factor", domain="IBC", clause="IBC 1004", kw=["occupant load"], var="occupant_load", op="formula", val="floor_area_sf / occupant_load_factor_sf_per_person", unit="persons", rel=[relation("uses_parameter", "IBC_OCC_LOAD_FACTOR_BUSINESS"), relation("depends_on", "IBC_OCC_LOAD_FACTOR_BUSINESS")]),
        dict(id="FORMULA_OCC_LOAD_DENSE_VARIANT", name="Dense-assembly occupant-load formula variant", domain="IBC", clause="IBC 1004", kw=["occupant load", "assembly"], var="occupant_load", op="formula", val="floor_area_sf / 7", unit="persons", rel=[relation("formula_variant_of", "FORMULA_OCC_LOAD_AREA_DIV_FACTOR")]),
        dict(id="FORMULA_EGRESS_WIDTH_BY_LOAD", name="Egress width by occupant load", domain="IBC", clause="IBC 1005", kw=["means of egress sizing"], var="egress_width_in", op="formula", val="occupant_load * width_factor", unit="inch", rel=[relation("uses_parameter", "FORMULA_OCC_LOAD_AREA_DIV_FACTOR")]),
        dict(id="FORMULA_EGRESS_WIDTH_SPRINKLER_VARIANT", name="Sprinkler-modified egress width formula variant", domain="IBC", clause="IBC 1005", kw=["means of egress", "sprinkler"], var="egress_width_in", op="formula", val="occupant_load * reduced_width_factor", unit="inch", rel=[relation("formula_variant_of", "FORMULA_EGRESS_WIDTH_BY_LOAD")]),
        dict(id="FORMULA_RAMP_RUN_FROM_RISE", name="Ramp run length from rise and slope", domain="ADA", clause="ADA 405", kw=["ramp", "rise", "slope"], var="ramp_run_length_in", op="formula", val="ramp_rise_in / ramp_slope_ratio", unit="inch", rel=[relation("uses_parameter", "ADA_RAMP_SLOPE_MAX")]),
        dict(id="FORMULA_RAMP_STEEP_VARIANT", name="Steeper ramp formula variant", domain="ADA", clause="ADA 405", kw=["ramp", "slope"], var="ramp_run_length_in", op="formula", val="ramp_rise_in / 0.10", unit="inch", rel=[relation("formula_variant_of", "FORMULA_RAMP_RUN_FROM_RISE")]),
        dict(id="FORMULA_TURNING_SPACE_CIRCLE", name="Circular turning-space formula", domain="ADA", clause="ADA 304", kw=["turning space"], var="turning_clear_floor_area_sf", op="formula", val="3.14159 * (turning_diameter_in / 24) ** 2", unit="sf", rel=[relation("uses_parameter", "ADA_TURNING_SPACE_CIRCLE")]),
        dict(id="FORMULA_TURNING_SPACE_T_VARIANT", name="T-shape turning-space formula variant", domain="ADA", clause="ADA 304", kw=["turning space"], var="turning_clear_floor_area_sf", op="formula", val="t_shape_clear_floor_area", unit="sf", rel=[relation("formula_variant_of", "FORMULA_TURNING_SPACE_CIRCLE")]),
        dict(id="FORMULA_HAZARD_QUANTITY_WITH_CONTROL_AREA", name="Hazard quantity with control-area factor", domain="IFC", clause="IFC 5003", kw=["hazardous materials", "control area"], var="allowable_hazard_quantity_l", op="formula", val="base_quantity * control_area_factor", unit="L", rel=[relation("uses_parameter", "IFC_CONTROL_AREA_LIMIT")]),
        dict(id="FORMULA_HAZARD_UNSPRINKLERED_VARIANT", name="Unsprinklered hazard quantity formula variant", domain="IFC", clause="IFC 5003", kw=["hazardous materials", "sprinkler"], var="allowable_hazard_quantity_l", op="formula", val="base_quantity", unit="L", rel=[relation("formula_variant_of", "FORMULA_HAZARD_QUANTITY_WITH_CONTROL_AREA")]),
    ]


def make_rule(spec: dict[str, Any], kg: KgEvidenceIndex) -> dict[str, Any]:
    constraints = []
    if spec.get("op") != "formula":
        constraints.append(
            {
                "variable": spec["var"],
                "op": spec["op"],
                "value": spec["val"],
                "unit": spec.get("unit", "unknown"),
                "source_quote": f"{spec['name']} ({spec['clause']})",
                "evidence": {
                    "chunk_ids": [],
                    "kg_node_ids": [],
                    "kg_edge_ids": [],
                },
            }
        )
    else:
        constraints.append(
            {
                "variable": spec["var"],
                "op": "formula",
                "value": spec["val"],
                "unit": spec.get("unit", "unknown"),
                "source_quote": f"{spec['name']} ({spec['clause']})",
                "evidence": {
                    "chunk_ids": [],
                    "kg_node_ids": [],
                    "kg_edge_ids": [],
                },
            }
        )
    item = rule(
        rule_id=spec["id"],
        name=spec["name"],
        source_domain=spec["domain"],
        clause_id=spec["clause"],
        keywords=spec.get("kw", []),
        kg=kg,
        rule_type=spec.get("rule_type", "requirement"),
        guard_obj=spec.get("guard"),
        constraints=constraints,
        relations=spec.get("rel", []),
        conflict_class=spec.get("conflict"),
    )
    chunk_id = item["source_chunk_ids"][0]
    for c in item["constraints"]:
        c["evidence"]["chunk_ids"] = [chunk_id]
        c["evidence"]["kg_node_ids"] = item["source_node_ids"]
        c["evidence"]["kg_edge_ids"] = [f"kg_edge_source_{chunk_id[:8]}"]
    return item


def add_conflict_exclusion_edges(rules: list[dict[str, Any]]) -> None:
    classes: dict[str, list[dict[str, Any]]] = {}
    for item in rules:
        cls = item.get("conflict_class")
        if cls:
            classes.setdefault(str(cls), []).append(item)
    for members in classes.values():
        if len(members) < 2:
            continue
        for item in members:
            existing = {
                (str(rel.get("type", "")).lower(), str(rel.get("target", "")))
                for rel in item.get("relations", [])
            }
            for other in members:
                if other["rule_id"] == item["rule_id"]:
                    continue
                key = ("excludes", other["rule_id"])
                if key not in existing:
                    item.setdefault("relations", []).append(
                        {
                            "type": "excludes",
                            "target": other["rule_id"],
                            "inferred_from_conflict_class": item["conflict_class"],
                        }
                    )


def task_specs() -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []

    def add(task_id: str, source_domain: str, target: str, title: str, engineering_task: str,
            scenario: dict[str, Any], variables: dict[str, dict[str, Any]], objectives: list[dict[str, str]],
            weights: list[float], candidate: list[str], final: list[str], surplus: dict[str, str],
            constraints: list[dict[str, Any]], cells: list[dict[str, Any]] | None = None,
            cross_code: bool = False, source_domains: list[str] | None = None,
            dependency_closure: list[str] | None = None, parameter_paths: list[list[str]] | None = None) -> None:
        tasks.append({
            "task_id": task_id,
            "source_domain": source_domain,
            "source_domains": source_domains or ([source_domain] if source_domain != "mixed" else ["ADA", "IBC", "IFC"]),
            "target_interaction": target,
            "title": title,
            "engineering_task": engineering_task,
            "scenario_facts": scenario,
            "decision_variables": variables,
            "objectives": objectives,
            "preference_weights": weights,
            "candidate": candidate,
            "final": final,
            "surplus": surplus,
            "constraints": constraints,
            "cells": cells or [],
            "cross_code": cross_code,
            "dependency_closure_rules": dependency_closure or [],
            "parameter_propagation_paths": parameter_paths or [],
        })

    def v(lo: float, hi: float, unit: str, typ: str = "continuous") -> dict[str, Any]:
        return {"type": typ, "unit": unit, "lower": lo, "upper": hi}

    # Scenario-conditioned applicability, 5
    add("ARCH_OPT_01", "ADA", "scenario-conditioned applicability", "Accessible car parking layout",
        "Design a compact accessible car-parking bay while preserving enough adjacent loading space for safe transfer.",
        {"facility": "parking", "vehicle_type": "car", "accessibility_required": True},
        {"parking_space_width_in": v(84, 144, "inch"), "access_aisle_width_in": v(48, 96, "inch")},
        [{"name": "minimize_paved_width", "expression": "parking_space_width_in + access_aisle_width_in"}, {"name": "maximize_vehicle_access_area", "expression": "parking_space_width_in * access_aisle_width_in"}],
        [0.55, 0.45],
        ["ADA_PARKING_CAR_WIDTH", "ADA_PARKING_ACCESS_AISLE", "ADA_PARKING_VAN_WIDTH"],
        ["ADA_PARKING_CAR_WIDTH", "ADA_PARKING_ACCESS_AISLE"],
        {"ADA_PARKING_VAN_WIDTH": "scenario_inapplicable_same_domain_rule"},
        [constraint("C1", "parking_space_width_in >= 96", "car_space_width", "ADA_PARKING_CAR_WIDTH", ["parking_space_width_in"]), constraint("C2", "access_aisle_width_in >= 60", "access_aisle_width", "ADA_PARKING_ACCESS_AISLE", ["access_aisle_width_in"])])
    add("ARCH_OPT_02", "ADA", "scenario-conditioned applicability", "Accessible van parking layout",
        "Design a compact accessible van-parking bay while preserving enough loading space for ramp deployment.",
        {"facility": "parking", "vehicle_type": "van", "accessibility_required": True},
        {"parking_space_width_in": v(96, 156, "inch"), "access_aisle_width_in": v(48, 108, "inch")},
        [{"name": "minimize_paved_width", "expression": "parking_space_width_in + access_aisle_width_in"}, {"name": "maximize_van_loading_area", "expression": "parking_space_width_in * access_aisle_width_in"}],
        [0.55, 0.45],
        ["ADA_PARKING_VAN_WIDTH", "ADA_PARKING_ACCESS_AISLE", "ADA_PARKING_CAR_WIDTH"],
        ["ADA_PARKING_VAN_WIDTH", "ADA_PARKING_ACCESS_AISLE"],
        {"ADA_PARKING_CAR_WIDTH": "scenario_inapplicable_same_domain_rule"},
        [constraint("C1", "parking_space_width_in >= 132", "van_space_width", "ADA_PARKING_VAN_WIDTH", ["parking_space_width_in"]), constraint("C2", "access_aisle_width_in >= 60", "access_aisle_width", "ADA_PARKING_ACCESS_AISLE", ["access_aisle_width_in"])])
    add("ARCH_OPT_03", "IBC", "scenario-conditioned applicability", "Business occupancy egress sizing",
        "Size level egress for a business tenant using the floor area and occupancy category provided by the design brief.",
        {"occupancy_group": "B", "floor_area_sf": 9000, "sprinklered": True},
        {"egress_width_in": v(32, 90, "inch"), "exit_access_travel_distance_ft": v(40, 260, "ft")},
        [{"name": "minimize_egress_width", "expression": "egress_width_in"}, {"name": "maximize_served_travel_distance", "expression": "exit_access_travel_distance_ft"}],
        [0.6, 0.4],
        ["IBC_OCC_LOAD_FACTOR_BUSINESS", "IBC_EGRESS_WIDTH_FACTOR_LEVEL", "IBC_OCC_LOAD_FACTOR_ASSEMBLY"],
        ["IBC_OCC_LOAD_FACTOR_BUSINESS", "IBC_EGRESS_WIDTH_FACTOR_LEVEL"],
        {"IBC_OCC_LOAD_FACTOR_ASSEMBLY": "scenario_inapplicable_same_domain_rule"},
        [scenario_constraint("C1", "derived_occupant_load >= 60", "business_load_from_area", "scenario_area_model", []), constraint("C2", "egress_width_in >= 36", "level_egress_width", "IBC_EGRESS_WIDTH_FACTOR_LEVEL", ["egress_width_in"])],
        source_domains=["IBC"])
    add("ARCH_OPT_04", "IFC", "scenario-conditioned applicability", "Fire alarm installation layout",
        "Plan a new fire-alarm installation by choosing device coverage and circuit length for a sprinklered building area.",
        {"system_type": "fire_alarm", "activity": "installation", "sprinkler_required": True},
        {"alarm_notification_appliance_count": v(1, 80, "count"), "alarm_circuit_length_ft": v(100, 3000, "ft"), "sprinkler_coverage_ratio": v(0, 1, "ratio")},
        [{"name": "minimize_alarm_circuit_length", "expression": "alarm_circuit_length_ft"}, {"name": "maximize_alarm_device_coverage", "expression": "alarm_notification_appliance_count"}],
        [0.5, 0.5],
        ["IFC_FIRE_ALARM_INSTALL_PERMIT", "IFC_FIRE_ALARM_MAINTENANCE_EXCEPTION", "IFC_SPRINKLER_GENERAL"],
        ["IFC_FIRE_ALARM_INSTALL_PERMIT", "IFC_SPRINKLER_GENERAL"],
        {"IFC_FIRE_ALARM_MAINTENANCE_EXCEPTION": "scenario_inapplicable_same_domain_rule"},
        [constraint("C1", "alarm_notification_appliance_count >= 1", "installation_alarm_devices", "IFC_FIRE_ALARM_INSTALL_PERMIT", ["alarm_notification_appliance_count"]), constraint("C2", "sprinkler_coverage_ratio >= 1.0", "sprinkler_coverage", "IFC_SPRINKLER_GENERAL", ["sprinkler_coverage_ratio"])])
    add("ARCH_OPT_05", "IFC", "scenario-conditioned applicability", "Sprinklered hazardous storage planning",
        "Plan an open hazardous-material storage area by balancing storage quantity, sprinkler coverage, and fire separation.",
        {"hazard_storage": True, "storage_type": "open_storage", "sprinkler_required": True, "system_type": "storage_area"},
        {"hazard_quantity_l": v(20, 150, "L"), "sprinkler_coverage_ratio": v(0, 1, "ratio"), "fire_separation_rating_hr": v(0, 4, "hour")},
        [{"name": "maximize_storage_quantity", "expression": "hazard_quantity_l"}, {"name": "minimize_fire_safety_burden", "expression": "sprinkler_coverage_ratio + fire_separation_rating_hr"}],
        [0.45, 0.55],
        ["IFC_HAZARD_STORAGE_BASELINE", "IFC_SPRINKLER_GENERAL", "IFC_FIRE_SEPARATION_HAZARD", "IFC_HAZARD_CLOSED_CONTAINER_EXCEPTION", "FORMULA_HAZARD_UNSPRINKLERED_VARIANT"],
        ["IFC_HAZARD_STORAGE_BASELINE", "IFC_SPRINKLER_GENERAL", "IFC_FIRE_SEPARATION_HAZARD"],
        {
            "IFC_HAZARD_CLOSED_CONTAINER_EXCEPTION": "scenario_inapplicable_same_domain_rule",
            "FORMULA_HAZARD_UNSPRINKLERED_VARIANT": "scenario_inapplicable_same_domain_rule",
        },
        [
            constraint("C1", "hazard_quantity_l <= 60", "baseline_hazard_quantity", "IFC_HAZARD_STORAGE_BASELINE", ["hazard_quantity_l"]),
            constraint("C2", "sprinkler_coverage_ratio >= 1.0", "sprinkler_coverage", "IFC_SPRINKLER_GENERAL", ["sprinkler_coverage_ratio"]),
            constraint("C3", "fire_separation_rating_hr >= 2", "hazard_fire_separation", "IFC_FIRE_SEPARATION_HAZARD", ["fire_separation_rating_hr"]),
        ])

    # Dependency, 5
    add("ARCH_OPT_06", "ADA", "dependency", "Door clearance with maneuvering dependency",
        "Select accessible door clear width and maneuvering clearance for a latch-side approach condition.",
        {"route_type": "accessible_route", "door_approach": "latch_side"},
        {"door_clear_width_in": v(28, 42, "inch"), "maneuvering_clearance_in": v(0, 36, "inch")},
        [{"name": "minimize_opening_width", "expression": "door_clear_width_in"}, {"name": "maximize_approach_margin", "expression": "maneuvering_clearance_in - 18"}],
        [0.45, 0.55],
        ["ADA_DOOR_CLEAR_WIDTH", "ADA_DOOR_APPROACH_CLEARANCE", "ADA_DOOR_STORAGE_EXCEPTION"],
        ["ADA_DOOR_CLEAR_WIDTH", "ADA_DOOR_APPROACH_CLEARANCE"],
        {"ADA_DOOR_STORAGE_EXCEPTION": "dependency_support_or_dependency_variant"},
        [constraint("C1", "door_clear_width_in >= 32", "door_clear_width", "ADA_DOOR_CLEAR_WIDTH", ["door_clear_width_in"]), constraint("C2", "maneuvering_clearance_in >= 18", "maneuvering_clearance", "ADA_DOOR_APPROACH_CLEARANCE", ["maneuvering_clearance_in"])],
        dependency_closure=["ADA_DOOR_CLEAR_WIDTH"])
    add("ARCH_OPT_07", "ADA", "dependency", "Ramp slope-rise-landing design",
        "Choose ramp slope, rise, and landing length while preserving slope-to-rise and landing dependencies.",
        {"element_type": "ramp", "route_type": "accessible_route"},
        {"ramp_slope_ratio": v(0.04, 0.12, "ratio"), "ramp_rise_in": v(4, 36, "inch"), "landing_length_in": v(36, 96, "inch")},
        [{"name": "minimize_ramp_length_proxy", "expression": "ramp_rise_in / ramp_slope_ratio"}, {"name": "maximize_landing_length", "expression": "landing_length_in"}],
        [0.5, 0.5],
        ["ADA_RAMP_SLOPE_MAX", "ADA_RAMP_RISE_MAX", "ADA_RAMP_LANDING_LENGTH", "FORMULA_RAMP_RUN_FROM_RISE", "FORMULA_RAMP_STEEP_VARIANT"],
        ["ADA_RAMP_SLOPE_MAX", "ADA_RAMP_RISE_MAX", "ADA_RAMP_LANDING_LENGTH"],
        {
            "FORMULA_RAMP_RUN_FROM_RISE": "dependency_support_or_dependency_variant",
            "FORMULA_RAMP_STEEP_VARIANT": "dependency_support_or_dependency_variant",
        },
        [constraint("C1", "ramp_slope_ratio <= 0.0833", "ramp_slope", "ADA_RAMP_SLOPE_MAX", ["ramp_slope_ratio"]), constraint("C2", "ramp_rise_in <= 30", "ramp_rise", "ADA_RAMP_RISE_MAX", ["ramp_rise_in"]), constraint("C3", "landing_length_in >= 60", "landing_length", "ADA_RAMP_LANDING_LENGTH", ["landing_length_in"])],
        dependency_closure=["ADA_RAMP_SLOPE_MAX"])
    add("ARCH_OPT_08", "IBC", "dependency", "Occupant-load based egress width",
        "Size egress width for a business tenant after the code-derived occupant load is computed from floor area.",
        {"occupancy_group": "B", "floor_area_sf": 12000},
        {"egress_width_in": v(32, 90, "inch"), "exit_access_travel_distance_ft": v(40, 260, "ft")},
        [{"name": "minimize_egress_width", "expression": "egress_width_in"}, {"name": "maximize_served_travel_distance", "expression": "exit_access_travel_distance_ft"}],
        [0.55, 0.45],
        ["FORMULA_OCC_LOAD_AREA_DIV_FACTOR", "IBC_OCC_LOAD_FACTOR_BUSINESS", "FORMULA_OCC_LOAD_DENSE_VARIANT"],
        ["FORMULA_OCC_LOAD_AREA_DIV_FACTOR", "IBC_OCC_LOAD_FACTOR_BUSINESS"],
        {"FORMULA_OCC_LOAD_DENSE_VARIANT": "dependency_support_or_dependency_variant"},
        [scenario_constraint("C1", "derived_occupant_load >= 80", "occupant_load_from_area", "scenario_area_model", []), scenario_constraint("C2", "egress_width_in >= 16", "egress_width_from_derived_load", "scenario_egress_model", ["egress_width_in"])],
        dependency_closure=["IBC_OCC_LOAD_FACTOR_BUSINESS"], parameter_paths=[["IBC_OCC_LOAD_FACTOR_BUSINESS", "FORMULA_OCC_LOAD_AREA_DIV_FACTOR"]])
    add("ARCH_OPT_09", "IBC", "dependency", "Two-exit configuration from common path",
        "Choose the number of exits and the common-path length for a mercantile tenant with a moderate occupant load.",
        {"occupancy_group": "M", "occupant_load": 80},
        {"number_of_exits": v(1, 5, "count"), "common_path_ft": v(30, 120, "ft")},
        [{"name": "minimize_exit_count", "expression": "number_of_exits"}, {"name": "maximize_usable_common_path_length", "expression": "common_path_ft"}],
        [0.5, 0.5],
        ["IBC_TWO_EXITS_REQUIRED", "IBC_COMMON_PATH_LIMIT", "IBC_TRAVEL_DISTANCE_UNSPRINKLERED"],
        ["IBC_TWO_EXITS_REQUIRED", "IBC_COMMON_PATH_LIMIT"],
        {"IBC_TRAVEL_DISTANCE_UNSPRINKLERED": "dependency_support_or_dependency_variant"},
        [constraint("C1", "number_of_exits >= 2", "two_exits", "IBC_TWO_EXITS_REQUIRED", ["number_of_exits"]), constraint("C2", "common_path_ft <= 75", "common_path_limit", "IBC_COMMON_PATH_LIMIT", ["common_path_ft"])],
        dependency_closure=["IBC_COMMON_PATH_LIMIT"])
    add("ARCH_OPT_10", "IFC", "dependency", "Hazard storage control-area dependency",
        "Choose hazardous-material storage quantity and fire separation for a control-area design.",
        {"hazard_storage": True, "control_area": True},
        {"hazard_quantity_l": v(20, 300, "L"), "fire_separation_rating_hr": v(0, 4, "hour")},
        [{"name": "maximize_storage_quantity", "expression": "hazard_quantity_l"}, {"name": "minimize_separation", "expression": "fire_separation_rating_hr"}],
        [0.45, 0.55],
        ["IFC_CONTROL_AREA_LIMIT", "IFC_HAZARD_STORAGE_BASELINE", "FORMULA_HAZARD_UNSPRINKLERED_VARIANT"],
        ["IFC_CONTROL_AREA_LIMIT", "IFC_HAZARD_STORAGE_BASELINE"],
        {"FORMULA_HAZARD_UNSPRINKLERED_VARIANT": "dependency_support_or_dependency_variant"},
        [constraint("C1", "hazard_quantity_l <= 240", "control_area_limit", "IFC_CONTROL_AREA_LIMIT", ["hazard_quantity_l"]), constraint("C2", "fire_separation_rating_hr >= 1", "minimum_separation", "IFC_HAZARD_STORAGE_BASELINE", ["fire_separation_rating_hr"])],
        dependency_closure=["IFC_HAZARD_STORAGE_BASELINE"])

    # Exclusion / alternative templates, 5
    add("ARCH_OPT_11", "ADA", "exclusion / alternative compliance template", "Deep roll-in shower layout",
        "Design an accessible roll-in shower with a compact footprint and usable interior depth.",
        {"facility_type": "bathing_room", "available_shower_depth_in": 66, "accessibility_required": True},
        {"shower_width_in": v(28, 48, "inch"), "shower_depth_in": v(34, 72, "inch")},
        [{"name": "minimize_footprint", "expression": "shower_width_in * shower_depth_in"}, {"name": "maximize_shower_depth", "expression": "shower_depth_in"}],
        [0.6, 0.4],
        ["ADA_SHOWER_STANDARD_ROLL_IN", "ADA_SHOWER_STANDARD_DEPTH", "ADA_SHOWER_ALTERNATE_ROLL_IN"],
        ["ADA_SHOWER_STANDARD_ROLL_IN", "ADA_SHOWER_STANDARD_DEPTH"],
        {"ADA_SHOWER_ALTERNATE_ROLL_IN": "excluded_alternative_branch"},
        [constraint("C1", "shower_width_in >= 30", "standard_shower_width", "ADA_SHOWER_STANDARD_ROLL_IN", ["shower_width_in"]), constraint("C2", "shower_depth_in >= 60", "standard_shower_depth", "ADA_SHOWER_STANDARD_DEPTH", ["shower_depth_in"])])
    add("ARCH_OPT_12", "ADA", "exclusion / alternative compliance template", "Square roll-in shower layout",
        "Design a square roll-in shower layout with a compact footprint and usable interior width.",
        {"facility_type": "bathing_room", "available_shower_depth_in": 48, "accessibility_required": True},
        {"shower_width_in": v(28, 48, "inch"), "shower_depth_in": v(34, 72, "inch")},
        [{"name": "minimize_footprint", "expression": "shower_width_in * shower_depth_in"}, {"name": "maximize_shower_width", "expression": "shower_width_in"}],
        [0.6, 0.4],
        ["ADA_SHOWER_ALTERNATE_ROLL_IN", "ADA_SHOWER_ALTERNATE_DEPTH", "ADA_SHOWER_STANDARD_ROLL_IN"],
        ["ADA_SHOWER_ALTERNATE_ROLL_IN", "ADA_SHOWER_ALTERNATE_DEPTH"],
        {"ADA_SHOWER_STANDARD_ROLL_IN": "excluded_alternative_branch"},
        [constraint("C1", "shower_width_in >= 36", "alternate_shower_width", "ADA_SHOWER_ALTERNATE_ROLL_IN", ["shower_width_in"]), constraint("C2", "shower_depth_in >= 36", "alternate_shower_depth", "ADA_SHOWER_ALTERNATE_DEPTH", ["shower_depth_in"])])
    add("ARCH_OPT_13", "ADA", "exclusion / alternative compliance template", "Wheelchair-accessible toilet stall layout",
        "Design an accessible toilet stall by balancing compartment area and clear interior width.",
        {"facility_type": "toilet_room", "wheelchair_access_required": True, "ambulatory_stall_requested": False},
        {"stall_width_in": v(32, 72, "inch"), "stall_depth_in": v(48, 72, "inch")},
        [{"name": "minimize_stall_area", "expression": "stall_width_in * stall_depth_in"}, {"name": "maximize_stall_width", "expression": "stall_width_in"}],
        [0.6, 0.4],
        ["ADA_TOILET_STANDARD_STALL", "ADA_DOOR_APPROACH_CLEARANCE", "ADA_TOILET_AMBULATORY_STALL"],
        ["ADA_TOILET_STANDARD_STALL", "ADA_DOOR_APPROACH_CLEARANCE"],
        {"ADA_TOILET_AMBULATORY_STALL": "excluded_alternative_branch"},
        [constraint("C1", "stall_width_in >= 60", "standard_stall_width", "ADA_TOILET_STANDARD_STALL", ["stall_width_in"]), constraint("C2", "stall_depth_in >= 56", "standard_stall_depth", "ADA_DOOR_APPROACH_CLEARANCE", ["stall_depth_in"])])
    add("ARCH_OPT_14", "mixed", "exclusion / alternative compliance template", "Separated mixed-use egress path",
        "Design a mixed-use egress strategy by balancing fire separation against travel distance.",
        {"fire_separation_wall_available": True, "occupancy_groups": ["M", "S"]},
        {"fire_separation_rating_hr": v(0, 4, "hour"), "travel_distance_ft": v(80, 260, "ft")},
        [{"name": "minimize_fire_separation", "expression": "fire_separation_rating_hr"}, {"name": "maximize_served_travel_distance", "expression": "travel_distance_ft"}],
        [0.45, 0.55],
        ["IBC_MIXED_SEPARATED_OCCUPANCY", "IBC_TRAVEL_DISTANCE_SPRINKLERED", "IBC_MIXED_NONSEPARATED_OCCUPANCY"],
        ["IBC_MIXED_SEPARATED_OCCUPANCY", "IBC_TRAVEL_DISTANCE_SPRINKLERED"],
        {"IBC_MIXED_NONSEPARATED_OCCUPANCY": "excluded_alternative_branch"},
        [constraint("C1", "fire_separation_rating_hr >= 2", "separated_occupancy_fire_rating", "IBC_MIXED_SEPARATED_OCCUPANCY", ["fire_separation_rating_hr"]), constraint("C2", "travel_distance_ft <= 250", "sprinklered_travel_distance", "IBC_TRAVEL_DISTANCE_SPRINKLERED", ["travel_distance_ft"])],
        cross_code=True, source_domains=["IBC", "IFC"])
    add("ARCH_OPT_15", "IFC", "exclusion / alternative compliance template", "Alternative suppression design",
        "Design a fire-suppression strategy by balancing fixed-pipe coverage against agent-covered floor area.",
        {"water_supply_available": False, "hazard_storage": True},
        {"suppression_coverage_area_sf": v(200, 3000, "sf"), "sprinkler_coverage_ratio": v(0, 1, "ratio")},
        [{"name": "minimize_sprinkler_pipe_coverage", "expression": "sprinkler_coverage_ratio"}, {"name": "maximize_agent_coverage_area", "expression": "suppression_coverage_area_sf"}],
        [0.5, 0.5],
        ["IFC_SUPPRESSION_ALTERNATIVE", "IFC_FIRE_SEPARATION_HAZARD", "IFC_SPRINKLER_TEMPLATE"],
        ["IFC_SUPPRESSION_ALTERNATIVE", "IFC_FIRE_SEPARATION_HAZARD"],
        {"IFC_SPRINKLER_TEMPLATE": "excluded_alternative_branch"},
        [constraint("C1", "suppression_coverage_area_sf >= 1200", "alternative_suppression_coverage", "IFC_SUPPRESSION_ALTERNATIVE", ["suppression_coverage_area_sf"]), constraint("C2", "sprinkler_coverage_ratio >= 0", "no_sprinkler_template_required", "IFC_FIRE_SEPARATION_HAZARD", ["sprinkler_coverage_ratio"])])

    # Override / exception, 5
    add("ARCH_OPT_16", "ADA", "exception / override", "Storage closet door-clearance exception",
        "Choose door width and approach clearance for an accessible-route storage closet with limited wall width.",
        {"route_type": "accessible_route", "space_type": "storage_closet"},
        {"door_clear_width_in": v(24, 42, "inch"), "maneuvering_clearance_in": v(0, 36, "inch")},
        [{"name": "minimize_door_width", "expression": "door_clear_width_in"}, {"name": "maximize_maneuvering_clearance", "expression": "maneuvering_clearance_in"}],
        [0.5, 0.5],
        ["ADA_DOOR_STORAGE_EXCEPTION", "ADA_DOOR_APPROACH_CLEARANCE", "ADA_DOOR_CLEAR_WIDTH"],
        ["ADA_DOOR_STORAGE_EXCEPTION", "ADA_DOOR_APPROACH_CLEARANCE"],
        {"ADA_DOOR_CLEAR_WIDTH": "defeated_by_override"},
        [constraint("C1", "door_clear_width_in >= 28", "storage_closet_exception_width", "ADA_DOOR_STORAGE_EXCEPTION", ["door_clear_width_in"]), constraint("C2", "maneuvering_clearance_in >= 18", "maneuvering_clearance", "ADA_DOOR_APPROACH_CLEARANCE", ["maneuvering_clearance_in"])])
    add("ARCH_OPT_17", "ADA", "exception / override", "Historic alternative access override",
        "Choose a historic-building access strategy that keeps the alternative service route short while limiting removal of historic fabric.",
        {"historic_property": True, "physical_access_threatens_historic_significance": True},
        {"alternative_service_distance_ft": v(20, 300, "ft"), "historic_entry_clear_width_in": v(24, 42, "inch"), "historic_fabric_removed_sf": v(0, 200, "sf")},
        [{"name": "minimize_alternative_service_distance", "expression": "alternative_service_distance_ft"}, {"name": "minimize_historic_fabric_removal", "expression": "historic_fabric_removed_sf"}],
        [0.55, 0.45],
        ["ADA_HISTORIC_ALTERNATIVE_ACCESS", "ADA_HISTORIC_MAX_FEASIBLE", "ADA_DOOR_CLEAR_WIDTH"],
        ["ADA_HISTORIC_ALTERNATIVE_ACCESS", "ADA_HISTORIC_MAX_FEASIBLE"],
        {"ADA_DOOR_CLEAR_WIDTH": "defeated_by_override"},
        [constraint("C1", "alternative_service_distance_ft <= 200", "alternative_access_route", "ADA_HISTORIC_ALTERNATIVE_ACCESS", ["alternative_service_distance_ft"]), constraint("C2", "historic_entry_clear_width_in >= 32", "max_feasible_historic_entry", "ADA_HISTORIC_MAX_FEASIBLE", ["historic_entry_clear_width_in"])])
    add("ARCH_OPT_18", "IFC", "exception / override", "Hazard closed-container exception",
        "Choose closed-container hazardous-material storage quantity while limiting control-area burden.",
        {"hazard_storage": True, "storage_type": "closed_container"},
        {"hazard_quantity_l": v(20, 180, "L"), "control_area_factor": v(1, 3, "factor")},
        [{"name": "maximize_storage_quantity", "expression": "hazard_quantity_l"}, {"name": "minimize_control_area_factor", "expression": "control_area_factor"}],
        [0.55, 0.45],
        ["IFC_HAZARD_CLOSED_CONTAINER_EXCEPTION", "IFC_CONTROL_AREA_LIMIT", "IFC_HAZARD_STORAGE_BASELINE"],
        ["IFC_HAZARD_CLOSED_CONTAINER_EXCEPTION", "IFC_CONTROL_AREA_LIMIT"],
        {"IFC_HAZARD_STORAGE_BASELINE": "defeated_by_override"},
        [constraint("C1", "hazard_quantity_l <= 120", "closed_container_exception", "IFC_HAZARD_CLOSED_CONTAINER_EXCEPTION", ["hazard_quantity_l"]), constraint("C2", "control_area_factor >= 1", "control_area_factor", "IFC_CONTROL_AREA_LIMIT", ["control_area_factor"])])
    add("ARCH_OPT_19", "IFC", "exception / override", "Fire alarm maintenance exception",
        "Plan fire-alarm maintenance by choosing how many devices to service while limiting temporary system impairment.",
        {"system_type": "fire_alarm", "activity": "maintenance"},
        {"devices_serviced_count": v(1, 80, "count"), "temporary_impairment_hours": v(0, 24, "hour")},
        [{"name": "minimize_temporary_impairment", "expression": "temporary_impairment_hours"}, {"name": "maximize_devices_serviced", "expression": "devices_serviced_count"}],
        [0.55, 0.45],
        ["IFC_FIRE_ALARM_MAINTENANCE_EXCEPTION", "IFC_EXISTING_BUILDING_CH11", "IFC_FIRE_ALARM_INSTALL_PERMIT"],
        ["IFC_FIRE_ALARM_MAINTENANCE_EXCEPTION", "IFC_EXISTING_BUILDING_CH11"],
        {"IFC_FIRE_ALARM_INSTALL_PERMIT": "defeated_by_override"},
        [constraint("C1", "devices_serviced_count >= 1", "maintenance_device_count", "IFC_FIRE_ALARM_MAINTENANCE_EXCEPTION", ["devices_serviced_count"]), constraint("C2", "temporary_impairment_hours <= 8", "existing_building_impairment_limit", "IFC_EXISTING_BUILDING_CH11", ["temporary_impairment_hours"])])
    add("ARCH_OPT_20", "IBC", "exception / override", "Existing building exception for egress retrofit",
        "Select an existing-building egress retrofit that limits retrofit area while improving door clear width.",
        {"building_status": "existing", "retrofit_project": True},
        {"egress_retrofit_area_sf": v(200, 5000, "sf"), "egress_width_in": v(30, 60, "inch")},
        [{"name": "minimize_egress_retrofit_area", "expression": "egress_retrofit_area_sf"}, {"name": "maximize_egress_door_width", "expression": "egress_width_in"}],
        [0.45, 0.55],
        ["IBC_EXISTING_BUILDING_EXCEPTION", "IBC_DOOR_EGRESS_WIDTH", "IBC_EGRESS_WIDTH_FACTOR_STAIR"],
        ["IBC_EXISTING_BUILDING_EXCEPTION", "IBC_DOOR_EGRESS_WIDTH"],
        {"IBC_EGRESS_WIDTH_FACTOR_STAIR": "defeated_by_override"},
        [constraint("C1", "egress_retrofit_area_sf <= 2500", "existing_building_retrofit_scope", "IBC_EXISTING_BUILDING_EXCEPTION", ["egress_retrofit_area_sf"]), constraint("C2", "egress_width_in >= 32", "egress_door_width", "IBC_DOOR_EGRESS_WIDTH", ["egress_width_in"])])

    # Precedence / cross-code priority, 5
    add("ARCH_OPT_21", "mixed", "precedence / cross-code priority", "ADA accessible route over IBC general route",
        "Size an accessible route in a public facility by balancing route length and clear route width.",
        {"route_type": "accessible_route", "accessibility_required": True},
        {"accessible_route_width_in": v(32, 60, "inch"), "accessible_route_length_ft": v(20, 300, "ft")},
        [{"name": "minimize_accessible_route_length", "expression": "accessible_route_length_ft"}, {"name": "maximize_accessible_route_width", "expression": "accessible_route_width_in"}],
        [0.5, 0.5],
        ["XCODE_ADA_OVER_IBC_ACCESSIBLE_ROUTE", "ADA_DOOR_CLEAR_WIDTH", "IBC_ACCESSIBLE_ROUTE_GENERAL"],
        ["XCODE_ADA_OVER_IBC_ACCESSIBLE_ROUTE", "ADA_DOOR_CLEAR_WIDTH"],
        {"IBC_ACCESSIBLE_ROUTE_GENERAL": "lower_priority_precedence_competitor"},
        [constraint("C1", "accessible_route_width_in >= 36", "ada_accessible_route", "XCODE_ADA_OVER_IBC_ACCESSIBLE_ROUTE", ["accessible_route_width_in"]), scenario_constraint("C2", "accessible_route_length_ft <= 220", "site_route_length_cap", "ADA_DOOR_CLEAR_WIDTH", ["accessible_route_length_ft"])],
        cross_code=True, source_domains=["ADA", "IBC"])
    add("ARCH_OPT_22", "mixed", "precedence / cross-code priority", "Mixed-use hazardous storage egress",
        "Design a mixed-use hazardous storage area by balancing fire separation and travel distance.",
        {"hazard_storage": True, "fire_separation_wall_available": False},
        {"fire_separation_rating_hr": v(0, 4, "hour"), "travel_distance_ft": v(80, 260, "ft")},
        [{"name": "minimize_fire_separation", "expression": "fire_separation_rating_hr"}, {"name": "maximize_served_travel_distance", "expression": "travel_distance_ft"}],
        [0.45, 0.55],
        ["XCODE_IFC_HAZARD_OVER_IBC_MIXED_USE", "IFC_HAZARD_STORAGE_BASELINE", "IBC_MIXED_NONSEPARATED_OCCUPANCY"],
        ["XCODE_IFC_HAZARD_OVER_IBC_MIXED_USE", "IFC_HAZARD_STORAGE_BASELINE"],
        {"IBC_MIXED_NONSEPARATED_OCCUPANCY": "lower_priority_precedence_competitor"},
        [constraint("C1", "fire_separation_rating_hr >= 2", "ifc_hazard_precedence", "XCODE_IFC_HAZARD_OVER_IBC_MIXED_USE", ["fire_separation_rating_hr"]), constraint("C2", "travel_distance_ft <= 250", "travel_distance_cap", "IFC_HAZARD_STORAGE_BASELINE", ["travel_distance_ft"])],
        cross_code=True, source_domains=["IFC", "IBC"])
    add("ARCH_OPT_23", "mixed", "precedence / cross-code priority", "New mercantile egress width planning",
        "Size a new mercantile egress path while keeping routine operational obstructions small.",
        {"egress_design_stage": "new_design", "occupancy_group": "M"},
        {"egress_width_in": v(32, 72, "inch"), "egress_path_obstruction_width_in": v(0, 24, "inch")},
        [{"name": "minimize_egress_width", "expression": "egress_width_in"}, {"name": "minimize_obstruction_width", "expression": "egress_path_obstruction_width_in"}],
        [0.6, 0.4],
        ["XCODE_IBC_CH10_OVER_IFC_GENERAL_EGRESS", "IBC_EGRESS_WIDTH_FACTOR_STAIR", "IFC_GENERAL_EGRESS_MAINTENANCE"],
        ["XCODE_IBC_CH10_OVER_IFC_GENERAL_EGRESS", "IBC_EGRESS_WIDTH_FACTOR_STAIR"],
        {"IFC_GENERAL_EGRESS_MAINTENANCE": "lower_priority_precedence_competitor"},
        [constraint("C1", "egress_width_in >= 44", "ibc_ch10_width", "XCODE_IBC_CH10_OVER_IFC_GENERAL_EGRESS", ["egress_width_in"]), scenario_constraint("C2", "egress_path_obstruction_width_in <= 6", "operational_obstruction_limit", "IBC_EGRESS_WIDTH_FACTOR_STAIR", ["egress_path_obstruction_width_in"])],
        cross_code=True, source_domains=["IBC", "IFC"])
    add("ARCH_OPT_24", "mixed", "precedence / cross-code priority", "Historic-building accessibility retrofit",
        "Plan a historic-building accessibility retrofit by balancing clear entry width and preservation of historic fabric.",
        {"historic_property": True, "building_status": "existing"},
        {"historic_entry_clear_width_in": v(24, 42, "inch"), "historic_fabric_removed_sf": v(0, 200, "sf")},
        [{"name": "minimize_historic_fabric_removal", "expression": "historic_fabric_removed_sf"}, {"name": "maximize_entry_clear_width", "expression": "historic_entry_clear_width_in"}],
        [0.45, 0.55],
        ["XCODE_ADA_HISTORIC_OVER_IBC_EXISTING", "ADA_HISTORIC_MAX_FEASIBLE", "IBC_EXISTING_BUILDING_EXCEPTION"],
        ["XCODE_ADA_HISTORIC_OVER_IBC_EXISTING", "ADA_HISTORIC_MAX_FEASIBLE"],
        {"IBC_EXISTING_BUILDING_EXCEPTION": "lower_priority_precedence_competitor"},
        [constraint("C1", "historic_entry_clear_width_in >= 32", "ada_historic_precedence", "XCODE_ADA_HISTORIC_OVER_IBC_EXISTING", ["historic_entry_clear_width_in"]), scenario_constraint("C2", "historic_fabric_removed_sf <= 120", "fabric_removal_limit", "ADA_HISTORIC_MAX_FEASIBLE", ["historic_fabric_removed_sf"])],
        cross_code=True, source_domains=["ADA", "IBC"])
    add("ARCH_OPT_25", "mixed", "precedence / cross-code priority", "Assembly fire-alarm installation planning",
        "Plan fire-alarm installation for an assembly occupancy by balancing device coverage and circuit length.",
        {"system_type": "fire_alarm", "activity": "installation", "occupancy_group": "A"},
        {"alarm_notification_appliance_count": v(1, 80, "count"), "alarm_circuit_length_ft": v(100, 3000, "ft")},
        [{"name": "minimize_alarm_circuit_length", "expression": "alarm_circuit_length_ft"}, {"name": "maximize_alarm_device_coverage", "expression": "alarm_notification_appliance_count"}],
        [0.5, 0.5],
        ["XCODE_IFC_ALARM_OVER_IBC_OCCUPANCY", "IFC_FIRE_ALARM_INSTALL_PERMIT", "IBC_2021_OCCUPANCY_ONLY_COMPETITOR"],
        ["XCODE_IFC_ALARM_OVER_IBC_OCCUPANCY", "IFC_FIRE_ALARM_INSTALL_PERMIT"],
        {"IBC_2021_OCCUPANCY_ONLY_COMPETITOR": "lower_priority_precedence_competitor"},
        [constraint("C1", "alarm_notification_appliance_count >= 1", "ifc_alarm_precedence", "XCODE_IFC_ALARM_OVER_IBC_OCCUPANCY", ["alarm_notification_appliance_count"]), scenario_constraint("C2", "alarm_circuit_length_ft <= 2200", "alarm_circuit_constructability", "IFC_FIRE_ALARM_INSTALL_PERMIT", ["alarm_circuit_length_ft"])],
        cross_code=True, source_domains=["IFC", "IBC"])

    # Parameter/formula propagation, 5
    add("ARCH_OPT_26", "IBC", "parameter propagation / formula propagation", "Occupant-load formula propagation",
        "Optimize egress width for a business tenant after occupant load is derived from floor area.",
        {"occupancy_group": "B", "floor_area_sf": 15000},
        {"egress_width_in": v(32, 90, "inch"), "exit_access_travel_distance_ft": v(40, 260, "ft")},
        [{"name": "minimize_egress_width", "expression": "egress_width_in"}, {"name": "maximize_served_travel_distance", "expression": "exit_access_travel_distance_ft"}],
        [0.55, 0.45],
        ["FORMULA_OCC_LOAD_AREA_DIV_FACTOR", "IBC_OCC_LOAD_FACTOR_BUSINESS", "FORMULA_OCC_LOAD_DENSE_VARIANT"],
        ["FORMULA_OCC_LOAD_AREA_DIV_FACTOR", "IBC_OCC_LOAD_FACTOR_BUSINESS"],
        {"FORMULA_OCC_LOAD_DENSE_VARIANT": "parameter_variant_or_formula_variant"},
        [scenario_constraint("C1", "derived_occupant_load >= 100", "load_from_floor_area", "FORMULA_OCC_LOAD_AREA_DIV_FACTOR", []), scenario_constraint("C2", "egress_width_in >= 20", "width_from_derived_load", "FORMULA_OCC_LOAD_AREA_DIV_FACTOR", ["egress_width_in"])],
        parameter_paths=[["IBC_OCC_LOAD_FACTOR_BUSINESS", "FORMULA_OCC_LOAD_AREA_DIV_FACTOR"]])
    add("ARCH_OPT_27", "IBC", "parameter propagation / formula propagation", "Sprinkler-modified egress-width formula",
        "Optimize egress width for an unsprinklered assembly space after occupant load is derived from floor area.",
        {"occupancy_group": "A", "sprinklered": False, "floor_area_sf": 3000},
        {"egress_width_in": v(32, 120, "inch"), "exit_access_travel_distance_ft": v(40, 220, "ft")},
        [{"name": "minimize_egress_width", "expression": "egress_width_in"}, {"name": "maximize_served_travel_distance", "expression": "exit_access_travel_distance_ft"}],
        [0.5, 0.5],
        ["FORMULA_EGRESS_WIDTH_BY_LOAD", "FORMULA_OCC_LOAD_AREA_DIV_FACTOR", "FORMULA_EGRESS_WIDTH_SPRINKLER_VARIANT"],
        ["FORMULA_EGRESS_WIDTH_BY_LOAD", "FORMULA_OCC_LOAD_AREA_DIV_FACTOR"],
        {"FORMULA_EGRESS_WIDTH_SPRINKLER_VARIANT": "parameter_variant_or_formula_variant"},
        [scenario_constraint("C1", "derived_occupant_load >= 200", "assembly_load", "FORMULA_OCC_LOAD_AREA_DIV_FACTOR", []), scenario_constraint("C2", "egress_width_in >= 40", "egress_width_by_derived_load", "FORMULA_EGRESS_WIDTH_BY_LOAD", ["egress_width_in"])],
        parameter_paths=[["FORMULA_OCC_LOAD_AREA_DIV_FACTOR", "FORMULA_EGRESS_WIDTH_BY_LOAD"]])
    add("ARCH_OPT_28", "ADA", "parameter propagation / formula propagation", "Ramp run-length formula propagation",
        "Optimize ramp run length after propagating rise and slope parameters.",
        {"element_type": "ramp", "route_type": "accessible_route"},
        {"ramp_slope_ratio": v(0.04, 0.12, "ratio"), "ramp_rise_in": v(6, 36, "inch"), "ramp_run_length_in": v(72, 420, "inch")},
        [{"name": "minimize_run_length", "expression": "ramp_run_length_in"}, {"name": "minimize_running_slope", "expression": "ramp_slope_ratio"}],
        [0.6, 0.4],
        ["FORMULA_RAMP_RUN_FROM_RISE", "ADA_RAMP_SLOPE_MAX", "FORMULA_RAMP_STEEP_VARIANT"],
        ["FORMULA_RAMP_RUN_FROM_RISE", "ADA_RAMP_SLOPE_MAX"],
        {"FORMULA_RAMP_STEEP_VARIANT": "parameter_variant_or_formula_variant"},
        [constraint("C1", "ramp_slope_ratio <= 0.0833", "ramp_slope", "ADA_RAMP_SLOPE_MAX", ["ramp_slope_ratio"]), scenario_constraint("C2", "ramp_run_length_in >= ramp_rise_in / ramp_slope_ratio", "run_from_rise_slope", "FORMULA_RAMP_RUN_FROM_RISE", ["ramp_run_length_in", "ramp_rise_in", "ramp_slope_ratio"])],
        parameter_paths=[["ADA_RAMP_SLOPE_MAX", "FORMULA_RAMP_RUN_FROM_RISE"]])
    add("ARCH_OPT_29", "ADA", "parameter propagation / formula propagation", "Turning-space formula propagation",
        "Choose clear-floor dimensions for a wheelchair turning area in a compact toilet room.",
        {"turning_geometry": "circular_clear_floor_area"},
        {"turning_diameter_in": v(48, 72, "inch"), "turning_clear_floor_area_sf": v(12, 30, "sf")},
        [{"name": "minimize_turning_diameter", "expression": "turning_diameter_in"}, {"name": "maximize_clear_floor_area", "expression": "turning_clear_floor_area_sf"}],
        [0.55, 0.45],
        ["FORMULA_TURNING_SPACE_CIRCLE", "ADA_TURNING_SPACE_CIRCLE", "FORMULA_TURNING_SPACE_T_VARIANT"],
        ["FORMULA_TURNING_SPACE_CIRCLE", "ADA_TURNING_SPACE_CIRCLE"],
        {"FORMULA_TURNING_SPACE_T_VARIANT": "parameter_variant_or_formula_variant"},
        [constraint("C1", "turning_diameter_in >= 60", "turning_diameter", "ADA_TURNING_SPACE_CIRCLE", ["turning_diameter_in"]), scenario_constraint("C2", "turning_clear_floor_area_sf >= 19.6", "turning_clear_floor_area", "FORMULA_TURNING_SPACE_CIRCLE", ["turning_clear_floor_area_sf"])],
        parameter_paths=[["ADA_TURNING_SPACE_CIRCLE", "FORMULA_TURNING_SPACE_CIRCLE"]])
    add("ARCH_OPT_30", "IFC", "parameter propagation / formula propagation", "Hazard quantity control-area formula",
        "Optimize hazardous-material storage quantity in a sprinklered control area while preserving a positive storage safety buffer.",
        {"hazard_storage": True, "control_area": True, "sprinklered": True},
        {"hazard_quantity_l": v(40, 300, "L"), "storage_safety_buffer_l": v(0, 120, "L")},
        [{"name": "maximize_hazard_quantity", "expression": "hazard_quantity_l"}, {"name": "maximize_storage_safety_buffer", "expression": "storage_safety_buffer_l"}],
        [0.55, 0.45],
        ["FORMULA_HAZARD_QUANTITY_WITH_CONTROL_AREA", "IFC_CONTROL_AREA_LIMIT", "FORMULA_HAZARD_UNSPRINKLERED_VARIANT"],
        ["FORMULA_HAZARD_QUANTITY_WITH_CONTROL_AREA", "IFC_CONTROL_AREA_LIMIT"],
        {"FORMULA_HAZARD_UNSPRINKLERED_VARIANT": "parameter_variant_or_formula_variant"},
        [constraint("C1", "hazard_quantity_l <= 240", "control_area_limit", "IFC_CONTROL_AREA_LIMIT", ["hazard_quantity_l"]), scenario_constraint("C2", "storage_safety_buffer_l >= 240 - hazard_quantity_l", "storage_buffer_from_allowance", "FORMULA_HAZARD_QUANTITY_WITH_CONTROL_AREA", ["storage_safety_buffer_l", "hazard_quantity_l"])],
        parameter_paths=[["IFC_CONTROL_AREA_LIMIT", "FORMULA_HAZARD_QUANTITY_WITH_CONTROL_AREA"]])
    return tasks


def build_task_records(spec: dict[str, Any], rules_by_id: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    task_id = spec["task_id"]
    candidate = sorted(spec["candidate"])
    final = sorted(spec["final"])
    cells = spec["cells"] or [
        {
            "cell_id": f"{task_id}_cell_1",
            "cell_type": "single_valid_rule_structure",
            "rule_ids": final,
            "constraints": spec["constraints"],
            "description": "Executable feasible cell induced by the expected valid rule structure for this grounded scenario.",
        }
    ]
    expected_cells = [cell.get("cell_id", f"{task_id}_cell_{idx + 1}") for idx, cell in enumerate(cells)]
    metadata = {
        "source_domain": spec["source_domain"],
        "source_domains": spec["source_domains"],
        "target_interaction": spec["target_interaction"],
        "candidate_rule_ids_expected_for_diagnostics": candidate,
        "final_valid_rule_ids_expected_for_evaluation": final,
        "valid_rule_structures_expected": [final],
        "structured_surplus_rule_ids": sorted(spec["surplus"]),
        "structured_surplus_types": dict(sorted(spec["surplus"].items())),
        "defeated_rules": defeated_rules(spec),
        "excluded_rules": excluded_rules(spec),
        "precedence_defeated_rules": precedence_defeated_rules(spec),
        "dependency_closure_rules": spec["dependency_closure_rules"],
        "parameter_propagation_paths": spec["parameter_propagation_paths"],
        "expected_cell_count": len(cells),
        "cross_code": spec["cross_code"],
    }
    query = {
        "omega_id": task_id,
        "title": spec["title"],
        "domain": "architecture_code_compliance",
        "source_domain": spec["source_domain"],
        "task_type": spec["target_interaction"].replace(" / ", "_").replace(" ", "_").lower(),
        "engineering_task": spec["engineering_task"],
        "design_intent": spec["engineering_task"],
        "scenario_facts": spec["scenario_facts"],
        "decision_variables": spec["decision_variables"],
        "objectives": spec["objectives"],
        "query_preferences": {
            "lambda": spec["preference_weights"],
            "meaning": "architecture benchmark preference weights for the two listed objectives",
        },
        "preference_weights": spec["preference_weights"],
        "visible_input_note": (
            "Visible task input only. Executable solver constraints, valid rule structures, "
            "and certificate targets are hidden reference labels for evaluation."
        ),
        "stress_metadata": metadata,
    }
    provenance = provenance_for_rules(candidate, rules_by_id)
    rule_label = {
        "omega_id": task_id,
        "title": spec["title"],
        "domain": "architecture_code_compliance",
        "source_domain": spec["source_domain"],
        "task_type": query["task_type"],
        "scenario_facts": spec["scenario_facts"],
        "expected_source_rule_ids": candidate,
        "expected_defeated_rule_ids": metadata["defeated_rules"],
        "expected_surviving_rule_ids": final,
        "expected_valid_rule_structures": [final],
        "expected_rule_behavior": {
            "should_activate": final,
            "should_exclude": sorted(spec["surplus"]),
            "should_resolve": [spec["target_interaction"]],
        },
        "challenge_types": [spec["target_interaction"]],
        "valid_constraint_cell_ids": expected_cells,
        "expected_provenance": provenance,
        "stress_metadata": metadata,
    }
    feasible = {
        "omega_id": task_id,
        "title": spec["title"],
        "scenario_facts": spec["scenario_facts"],
        "decision_variables": spec["decision_variables"],
        "executable_constraints": spec["constraints"],
        "structure_only_constraints": [],
        "valid_constraint_cells": cells,
        "reference_semantics": {
            "candidate_rule_ids": candidate,
            "final_valid_rule_ids": final,
            "valid_rule_structures": [final],
            "semantic_validator_labels": {
                "source_valid_if": "all executable constraints hold and at least one valid cell holds when cells are present",
                "flat_invalid_if": "candidate surplus is retained as an active rule structure",
            },
        },
        "stress_metadata": metadata,
    }
    hidden_reference = {
        "solver_constraints": spec["constraints"],
        "solver_constraint_cells": cells,
        "pre_solver_structure_checks": [],
        "certificate_targets": {"source_rule_ids": final},
        "rule_structure_label": rule_label,
        "feasible_region_label": feasible,
    }
    task_file = {
        "version": "architecture_benchmark_task_v2_visible_query_hidden_reference",
        "task": query,
        "hidden_reference": hidden_reference,
        "stress_metadata": metadata,
    }
    return query, rule_label, feasible, task_file


def provenance_for_rules(rule_ids: list[str], rules_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    chunk_ids: list[str] = []
    node_ids: list[str] = []
    edge_ids: list[str] = []
    docs: list[dict[str, Any]] = []
    for rule_id in rule_ids:
        rule = rules_by_id[rule_id]
        chunk_ids.extend(rule.get("source_chunk_ids", []))
        node_ids.extend(rule.get("source_node_ids", []))
        for c in rule.get("constraints", []):
            ev = c.get("evidence", {})
            edge_ids.extend(ev.get("kg_edge_ids", []))
        docs.extend(rule.get("provenance", []))
    return {
        "kg_chunk_ids": sorted(set(chunk_ids)),
        "kg_node_ids": sorted(set(node_ids)),
        "kg_edge_ids": sorted(set(edge_ids)),
        "source_documents": docs,
    }


def defeated_rules(spec: dict[str, Any]) -> list[str]:
    return sorted(rid for rid, typ in spec["surplus"].items() if typ == "defeated_by_override")


def excluded_rules(spec: dict[str, Any]) -> list[str]:
    return sorted(rid for rid, typ in spec["surplus"].items() if typ == "excluded_alternative_branch")


def precedence_defeated_rules(spec: dict[str, Any]) -> list[str]:
    return sorted(rid for rid, typ in spec["surplus"].items() if typ == "lower_priority_precedence_competitor")


def quality_row(spec: dict[str, Any], rules_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidate = set(spec["candidate"])
    final = set(spec["final"])
    surplus = candidate - final
    structured = set(spec["surplus"])
    rel_counts = relation_counts(candidate, rules_by_id)
    provenance = provenance_for_rules(list(candidate), rules_by_id)
    row = {
        "task_id": spec["task_id"],
        "source_domain": spec["source_domain"],
        "target_interaction": spec["target_interaction"],
        "candidate_rule_count": len(candidate),
        "final_valid_rule_count": len(final),
        "candidate_final_ratio": round(len(candidate) / len(final), 3) if final else 0.0,
        "structured_surplus_count": len(structured),
        "structured_surplus_ratio": round(len(structured) / len(surplus), 3) if surplus else 0.0,
        "unstructured_noise_count": len(surplus - structured),
        "final_rules_subset_of_candidate": final < candidate,
        "scenario_applicability_contrast_count": sum(1 for typ in spec["surplus"].values() if typ == "scenario_inapplicable_same_domain_rule"),
        "conflict_class_size_gt_1_count": conflict_class_count(candidate, rules_by_id),
        "expected_cell_count": len(spec["cells"]) or 1,
        "provenance_valid": bool(provenance["kg_chunk_ids"] and provenance["source_documents"]),
        "candidate_rule_ids": sorted(candidate),
        "final_valid_rule_ids": sorted(final),
        "structured_surplus_rule_ids": sorted(structured),
        "structured_surplus_types": dict(sorted(spec["surplus"].items())),
        "unstructured_noise_rule_ids": sorted(surplus - structured),
    }
    row.update(rel_counts)
    return row


def relation_counts(candidate_ids: set[str], rules_by_id: dict[str, dict[str, Any]]) -> dict[str, int]:
    counts = {
        "dependency_edge_count": 0,
        "exclusion_edge_count": 0,
        "override_edge_count": 0,
        "precedence_edge_count": 0,
        "parameter_propagation_edge_count": 0,
    }
    for rule_id in candidate_ids:
        for rel in rules_by_id[rule_id].get("relations", []):
            target = str(rel.get("target", ""))
            if target not in candidate_ids:
                continue
            typ = str(rel.get("type", "")).lower()
            if typ in RELATION_GROUPS["dependency"]:
                counts["dependency_edge_count"] += 1
            if typ in RELATION_GROUPS["exclusion"]:
                counts["exclusion_edge_count"] += 1
            if typ in RELATION_GROUPS["override"]:
                counts["override_edge_count"] += 1
            if typ in RELATION_GROUPS["precedence"]:
                counts["precedence_edge_count"] += 1
            if typ in RELATION_GROUPS["parameter"]:
                counts["parameter_propagation_edge_count"] += 1
    return counts


def conflict_class_count(candidate_ids: set[str], rules_by_id: dict[str, dict[str, Any]]) -> int:
    classes: dict[str, int] = {}
    for rule_id in candidate_ids:
        cls = rules_by_id[rule_id].get("conflict_class")
        if cls:
            classes[str(cls)] = classes.get(str(cls), 0) + 1
    return sum(1 for size in classes.values() if size > 1)


def task_manifest_row(spec: dict[str, Any], task_path: Path, rules_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "task_id": spec["task_id"],
        "source_domain": spec["source_domain"],
        "source_domains": spec["source_domains"],
        "engineering_task": spec["engineering_task"],
        "target_interaction": spec["target_interaction"],
        "candidate_rule_ids": sorted(spec["candidate"]),
        "final_valid_rule_ids": sorted(spec["final"]),
        "valid_rule_structures_expected": [sorted(spec["final"])],
        "structured_surplus_rule_ids": sorted(spec["surplus"]),
        "structured_surplus_types": dict(sorted(spec["surplus"].items())),
        "expected_cell_count": len(spec["cells"]) or 1,
        "provenance_chunks": provenance_for_rules(spec["candidate"], rules_by_id)["kg_chunk_ids"],
        "cross_code": spec["cross_code"],
        "task_file": str(task_path),
    }


def build_reports(quality_rows: list[dict[str, Any]], manifest_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    coverage: dict[str, list[str]] = {}
    for row in quality_rows:
        coverage.setdefault(row["target_interaction"], []).append(row["task_id"])
    cross_code = [row["task_id"] for row in manifest_rows if row["cross_code"]]
    alternative = [
        row["task_id"]
        for row in manifest_rows
        if row["target_interaction"] == "exclusion / alternative compliance template"
    ]
    override = [row["task_id"] for row in manifest_rows if row["target_interaction"] == "exception / override"]
    precedence = [row["task_id"] for row in manifest_rows if row["target_interaction"] == "precedence / cross-code priority"]
    parameter = [row["task_id"] for row in manifest_rows if row["target_interaction"] == "parameter propagation / formula propagation"]
    candidate_equals_final = [
        row["task_id"] for row in quality_rows if row["candidate_rule_count"] == row["final_valid_rule_count"]
    ]
    noise = [row["task_id"] for row in quality_rows if row["unstructured_noise_count"] > 0]
    ratio_bad = [
        row["task_id"] for row in quality_rows if not (1.5 <= float(row["candidate_final_ratio"]) <= 3.5)
    ]
    surplus_bad = [
        row["task_id"]
        for row in quality_rows
        if row["structured_surplus_count"] < 1 or float(row["structured_surplus_ratio"]) < 0.7
    ]
    provenance_bad = [row["task_id"] for row in quality_rows if not row["provenance_valid"]]
    coverage_bad = [name for name, tasks in coverage.items() if len(tasks) < 5]
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "kg_input": {
            "nodes_csv": str(KG_EXPORT_DIR / "nodes.csv"),
            "edges_csv": str(KG_EXPORT_DIR / "edges.csv"),
            "summary_json": str(KG_EXPORT_DIR / "summary.json"),
        },
        "num_tasks": len(quality_rows),
        "interaction_coverage": coverage,
        "cross_code_tasks": cross_code,
        "alternative_template_tasks": alternative,
        "override_exception_tasks": override,
        "precedence_tasks": precedence,
        "parameter_propagation_tasks": parameter,
        "candidate_equals_final": candidate_equals_final,
        "tasks_with_unstructured_noise": noise,
        "candidate_final_ratio_not_in_target": ratio_bad,
        "structured_surplus_insufficient": surplus_bad,
        "provenance_invalid": provenance_bad,
        "interactions_with_fewer_than_5_tasks": coverage_bad,
        "direct_pipeline_readiness": {
            "cthr_flat_asp_smt_milp": True,
            "tasks": str(OUT_TASKS_PATH),
            "rule_structure_labels": str(OUT_RULE_LABELS_PATH),
            "feasible_region_labels": str(OUT_FEASIBLE_PATH),
            "combined_rule_library": str(OUT_RULE_LIBRARY_PATH),
            "input_boundary": (
                "Methods should consume visible optimization queries plus the rule library/KG. "
                "Rule-structure labels, feasible-region labels, and hidden_reference fields are "
                "reserved for evaluation only."
            ),
        },
    }
    lines = [
        "# Architecture Benchmark Quality Validation Report",
        "",
        "## Benchmark Scope",
        "",
        "Generated 30 architecture optimization/compliance tasks from the current Neo4j KG export, covering ADA accessibility, IBC egress, IFC fire-safety, and cross-code reasoning.",
        "",
        "## Task Inventory",
        "",
        "| task_id | source domains | target interaction | architecture scene |",
        "| --- | --- | --- | --- |",
    ]
    for row in manifest_rows:
        source_domains = ", ".join(row["source_domains"]) if isinstance(row["source_domains"], list) else str(row["source_domains"])
        lines.append(
            f"| {row['task_id']} | {source_domains} | {row['target_interaction']} | {row['engineering_task']} |"
        )
    lines.extend([
        "",
        "## Interaction Coverage",
        "",
    ])
    for interaction, tasks in sorted(coverage.items()):
        lines.append(f"- {interaction}: {', '.join(tasks)}")
    lines.extend([
        "",
        "## Code-Family Coverage",
        "",
        f"- ADA / IBC / IFC cross-code reasoning: {', '.join(cross_code)}",
        f"- Alternative templates: {', '.join(alternative)}",
        f"- Override / exception: {', '.join(override)}",
        f"- Precedence: {', '.join(precedence)}",
        f"- Parameter propagation: {', '.join(parameter)}",
        "",
        "## Quality Flags",
        "",
        f"- Candidate equals final: {', '.join(candidate_equals_final) if candidate_equals_final else 'none'}",
        f"- Unstructured noise: {', '.join(noise) if noise else 'none'}",
        f"- Candidate/final ratio outside 1.5-3.5: {', '.join(ratio_bad) if ratio_bad else 'none'}",
        f"- Structured surplus insufficient: {', '.join(surplus_bad) if surplus_bad else 'none'}",
        f"- Provenance invalid: {', '.join(provenance_bad) if provenance_bad else 'none'}",
        f"- Interactions with fewer than 5 tasks: {', '.join(coverage_bad) if coverage_bad else 'none'}",
        "",
        "## Per-Task Quality",
        "",
        "| task_id | source | interaction | candidate | final | ratio | structured surplus | noise | provenance |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for row in quality_rows:
        lines.append(
            f"| {row['task_id']} | {row['source_domain']} | {row['target_interaction']} | {row['candidate_rule_count']} | {row['final_valid_rule_count']} | {row['candidate_final_ratio']} | {row['structured_surplus_count']} | {row['unstructured_noise_count']} | {row['provenance_valid']} |"
        )
    lines.extend([
        "",
        "## Pipeline Readiness",
        "",
        "The output uses the same three-layer pattern as the current CTHR benchmark artifacts: optimization queries, rule-structure labels, and feasible-region labels. The combined rule library includes the rule records and provenance needed by CTHR, flat, ASP, SMT, and MILP experiments.",
        "",
        "The optimization query files expose only scenario facts, decision variables, objectives, and preferences. Executable constraints, valid rule structures, candidate/final labels, and certificate targets are hidden reference labels for evaluation and must not be used as method input.",
        "",
    ])
    return summary, "\n".join(lines)


def main() -> None:
    kg = KgEvidenceIndex(KG_EXPORT_DIR)
    rules = [make_rule(spec, kg) for spec in rule_library_specs()]
    add_conflict_exclusion_edges(rules)
    rules_by_id = {rule["rule_id"]: rule for rule in rules}
    tasks = task_specs()
    TASK_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    queries: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    feasible_items: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []

    for spec in tasks:
        query, label, feasible, task_file = build_task_records(spec, rules_by_id)
        task_path = TASK_DIR / f"{spec['task_id']}.json"
        write_json(task_path, task_file)
        queries.append(query)
        labels.append(label)
        feasible_items.append(feasible)
        manifest_rows.append(task_manifest_row(spec, task_path, rules_by_id))
        quality_rows.append(quality_row(spec, rules_by_id))

    rule_library = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "provider": "kg_export_rule_templates",
        "domain": "architecture",
        "source": {
            "kg_export_dir": str(KG_EXPORT_DIR),
            "datasets": DATASET_DOMAINS,
        },
        "summary": {
            "rules": len(rules),
            "tasks": len(tasks),
            "note": "Rule templates are grounded to current KG DocumentChunk provenance and include synthetic stress relations only where explicitly marked.",
        },
        "rules": rules,
    }
    write_json(OUT_RULE_LIBRARY_PATH, rule_library)
    write_json(OUT_TASKS_PATH, {"optimization_queries": queries})
    write_json(OUT_RULE_LABELS_PATH, {"rule_structure_labels": labels})
    write_json(OUT_FEASIBLE_PATH, {"feasible_region_labels": feasible_items})
    write_json(OUT_MANIFEST_PATH, {
        "version": "architecture_generated_benchmark_layers_v1",
        "source_kg_export": str(KG_EXPORT_DIR),
        "num_tasks": len(tasks),
        "files": {
            "optimization_queries": str(OUT_TASKS_PATH),
            "rule_structure_labels": str(OUT_RULE_LABELS_PATH),
            "feasible_region_labels": str(OUT_FEASIBLE_PATH),
            "combined_rule_library": str(OUT_RULE_LIBRARY_PATH),
            "task_directory": str(TASK_DIR),
        },
    })

    manifest_headers = [
        "task_id", "source_domain", "source_domains", "engineering_task", "target_interaction",
        "candidate_rule_ids", "final_valid_rule_ids", "valid_rule_structures_expected",
        "structured_surplus_rule_ids", "structured_surplus_types", "expected_cell_count",
        "provenance_chunks", "cross_code", "task_file",
    ]
    quality_headers = [
        "task_id", "source_domain", "target_interaction", "candidate_rule_count",
        "final_valid_rule_count", "candidate_final_ratio", "structured_surplus_count",
        "structured_surplus_ratio", "unstructured_noise_count", "final_rules_subset_of_candidate",
        "dependency_edge_count", "exclusion_edge_count", "override_edge_count",
        "precedence_edge_count", "parameter_propagation_edge_count",
        "scenario_applicability_contrast_count", "conflict_class_size_gt_1_count",
        "expected_cell_count", "provenance_valid", "candidate_rule_ids",
        "final_valid_rule_ids", "structured_surplus_rule_ids", "structured_surplus_types",
        "unstructured_noise_rule_ids",
    ]
    write_csv(OUT_TASK_MANIFEST_CSV, manifest_rows, manifest_headers)
    write_csv(OUT_QUALITY_CSV, quality_rows, quality_headers)
    summary, report = build_reports(quality_rows, manifest_rows)
    write_json(OUT_QUALITY_SUMMARY_JSON, summary)
    OUT_QUALITY_REPORT_MD.write_text(report, encoding="utf-8")

    print(json.dumps({
        "tasks": len(tasks),
        "rules": len(rules),
        "task_dir": str(TASK_DIR),
        "combined_rule_library": str(OUT_RULE_LIBRARY_PATH),
        "manifest_csv": str(OUT_TASK_MANIFEST_CSV),
        "quality_csv": str(OUT_QUALITY_CSV),
        "quality_report": str(OUT_QUALITY_REPORT_MD),
        "candidate_equals_final": summary["candidate_equals_final"],
        "unstructured_noise": summary["tasks_with_unstructured_noise"],
        "ratio_flags": summary["candidate_final_ratio_not_in_target"],
        "provenance_invalid": summary["provenance_invalid"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
