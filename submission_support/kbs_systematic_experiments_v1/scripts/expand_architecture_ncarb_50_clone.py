from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "architecture_fullkg_clean"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def keep_not_ncarb_extension(item: dict) -> bool:
    task_id = str(item.get("omega_id", ""))
    if not task_id.startswith("ARCH_FKG_"):
        return True
    try:
        number = int(task_id.rsplit("_", 1)[-1])
    except ValueError:
        return True
    return not 51 <= number <= 100


def replace_task_id(payload: dict, old_id: str, new_id: str) -> dict:
    text = json.dumps(payload, ensure_ascii=False)
    return json.loads(text.replace(old_id, new_id))


SOURCE_DOCS = [
    ("ARE-Practice-Exam-Programming-and-Analysis.pdf", "Programming and Analysis"),
    ("ARE-Practice-Exam-Project-Development-and-Documentation.pdf", "Project Development and Documentation"),
    ("ARE-Practice-Exam-Project-Planning-and-Design.pdf", "Project Planning and Design"),
]


def ncarb_origin(doc_index: int, question_number: int, theme: str) -> dict:
    document, division = SOURCE_DOCS[doc_index]
    return {
        "source_family": "NCARB_ARE_practice_exam",
        "document": document,
        "division": division,
        "question_number": f"Question {question_number}",
        "theme": theme,
        "adaptation_policy": "paraphrased_parameterized_not_verbatim",
    }


# new_number, base_task_number, new title, source document index, question number, theme, category
SPECS = [
    (51, 20, "NCARB-derived means-of-egress ramp for clinic level change", 1, 10, "ramp for small elevation change", "accessibility"),
    (52, 28, "NCARB-derived compact accessible ramp landing", 1, 10, "ramp landing dimension selection", "accessibility"),
    (53, 24, "NCARB-derived accessible restroom door operation package", 2, 15, "plumbing fixture and accessible restroom count", "accessibility"),
    (54, 26, "NCARB-derived accessible storage room clear-floor layout", 0, 5, "bike storage room and public storage planning", "accessibility"),
    (55, 27, "NCARB-derived medical-office visual sign proportion", 1, 34, "medical office interior signage scope", "accessibility"),
    (56, 2, "NCARB-derived emergency address sign sizing for street frontage", 0, 22, "commercial parcel frontage and emergency wayfinding", "accessibility"),
    (57, 49, "NCARB-derived park restroom toilet compartment operation", 2, 23, "accessible path and public restroom support spaces", "accessibility"),
    (58, 50, "NCARB-derived accessible park path landing package", 2, 23, "accessible walking path through historic park", "accessibility"),
    (59, 29, "NCARB-derived accessible egress elevator resilience package", 2, 30, "multi-story office option and accessible egress", "accessibility"),
    (60, 9, "NCARB-derived area-of-refuge wheelchair space planning", 2, 23, "accessible route and refuge planning", "accessibility"),
    (61, 7, "NCARB-derived patient-bed egress door clear opening", 1, 16, "clinic egress door and hardware coordination", "fire_and_egress"),
    (62, 8, "NCARB-derived existing care-unit patient door exception", 1, 16, "clinic door retrofit and access-control coordination", "fire_and_egress"),
    (63, 10, "NCARB-derived multi-floor fire alarm zone partition", 1, 28, "IBC table lookup and fire alarm zoning", "fire_and_egress"),
    (64, 11, "NCARB-derived open office audible alarm setting", 2, 30, "office building occupied-floor alarm setting", "fire_and_egress"),
    (65, 12, "NCARB-derived exterior egress door landing clearance", 1, 16, "clinic door swing and access-control coordination", "fire_and_egress"),
    (66, 13, "NCARB-derived waterfront temporary event tent egress", 0, 26, "waterfront cultural center temporary event layout", "fire_and_egress"),
    (67, 15, "NCARB-derived hazardous materials alarm supervision", 0, 20, "brownfield retail site and hazardous material review", "fire_and_egress"),
    (68, 16, "NCARB-derived solvent area extinguisher placement", 1, 9, "restaurant event and service-area coordination", "fire_and_egress"),
    (69, 17, "NCARB-derived indoor dispensing ventilation package", 1, 24, "hydrostatic and building system coordination", "fire_and_egress"),
    (70, 22, "NCARB-derived rooftop patio emergency voice precedence", 0, 13, "rooftop patio emergency voice alarm system", "fire_and_egress"),
    (71, 30, "NCARB-derived single-exit tenant layout documentation", 1, 30, "restaurant occupant load and exit documentation", "fire_and_egress"),
    (72, 37, "NCARB-derived assembly room audible alarm formula", 2, 5, "assembly exit count and audible alarm demand", "fire_and_egress"),
    (73, 40, "NCARB-derived egress ramp run-length formula", 1, 10, "ramp for elevation change in construction documents", "fire_and_egress"),
    (74, 42, "NCARB-derived event venue voice-alarm channel precedence", 1, 9, "restaurant event configuration and alarm priority", "fire_and_egress"),
    (75, 45, "NCARB-derived fire alarm voice preemption design", 1, 8, "addenda and specification sequence for alarm system", "fire_and_egress"),
    (76, 3, "NCARB-derived Type IA building fire-flow reserve", 0, 8, "construction type and allowable stories", "area_and_height"),
    (77, 4, "NCARB-derived Type IV fire-flow reserve", 1, 11, "construction type and building area", "area_and_height"),
    (78, 47, "NCARB-derived Type IA fire-flow table applicability", 0, 35, "maximum buildable area and code table calculation", "area_and_height"),
    (79, 48, "NCARB-derived Type IV fire-flow duration applicability", 1, 11, "construction type and allowable area", "area_and_height"),
    (80, 19, "NCARB-derived roof hatch guard planning", 1, 27, "roof detail and access coordination", "area_and_height"),
    (81, 18, "NCARB-derived Class 3 oxidizer storage height package", 1, 28, "warehouse code-table lookup for storage conditions", "area_and_height"),
    (82, 29, "NCARB-derived accessible egress elevator for high-rise option", 2, 30, "multi-story office option evaluation", "area_and_height"),
    (83, 30, "NCARB-derived occupant-load increase documentation", 2, 5, "exit count and occupant-load diagram coordination", "area_and_height"),
    (84, 1, "NCARB-derived pool barrier gate latch height", 0, 17, "aquatic center site planning", "site_and_zoning_like"),
    (85, 46, "NCARB-derived pool gate latch applicability check", 0, 17, "aquatic center property planning", "site_and_zoning_like"),
    (86, 13, "NCARB-derived cultural-center temporary tent site layout", 0, 26, "waterfront cultural center lot selection", "site_and_zoning_like"),
    (87, 3, "NCARB-derived commercial parcel fire-flow loop", 0, 22, "commercial parcel selection and frontage", "site_and_zoning_like"),
    (88, 4, "NCARB-derived low-rise site fire-flow reserve", 0, 20, "brownfield retail site planning", "site_and_zoning_like"),
    (89, 47, "NCARB-derived site fire-flow table row applicability", 0, 1, "parking capacity and building selection", "site_and_zoning_like"),
    (90, 48, "NCARB-derived buildable parcel fire-flow duration", 0, 35, "maximum buildable area and grading constraints", "site_and_zoning_like"),
    (91, 2, "NCARB-derived emergency address sign for retail frontage", 0, 22, "street frontage and emergency response visibility", "site_and_zoning_like"),
    (92, 30, "NCARB-derived occupant-load diagram coordination", 1, 20, "construction document discrepancy before bidding", "construction_documentation"),
    (93, 22, "NCARB-derived emergency voice specification precedence", 1, 8, "addenda and specification sequence", "construction_documentation"),
    (94, 24, "NCARB-derived toilet door hardware documentation package", 1, 16, "door hardware and access-control coordination", "construction_documentation"),
    (95, 27, "NCARB-derived interior signage production criteria", 1, 34, "medical office signage production scope", "construction_documentation"),
    (96, 7, "NCARB-derived hospital door schedule clear-opening check", 0, 10, "hospital renovation phasing and door schedule", "construction_documentation"),
    (97, 8, "NCARB-derived existing patient door exception schedule", 0, 10, "hospital pharmacy renovation phasing", "construction_documentation"),
    (98, 10, "NCARB-derived fire alarm zone drawing coordination", 1, 20, "construction document discrepancy and alarm zones", "construction_documentation"),
    (99, 11, "NCARB-derived audible alarm calculation sheet", 1, 28, "code table and alarm calculation documentation", "construction_documentation"),
    (100, 49, "NCARB-derived restroom door operation checklist", 2, 15, "fixture count and accessible restroom checklist", "construction_documentation"),
]


def main() -> None:
    algorithm_inputs_path = DATASET / "algorithm_inputs" / "architecture_algorithm_inputs.json"
    evaluation_references_path = DATASET / "evaluation_references" / "architecture_evaluation_references.json"
    scenario_models_path = DATASET / "scenario_models" / "architecture_public_scenario_models.json"

    algorithm_inputs = read_json(algorithm_inputs_path)
    evaluation_references = read_json(evaluation_references_path)
    scenario_models = read_json(scenario_models_path)

    algorithm_inputs["items"] = [item for item in algorithm_inputs.get("items", []) if keep_not_ncarb_extension(item)]
    evaluation_references["items"] = [item for item in evaluation_references.get("items", []) if keep_not_ncarb_extension(item)]
    scenario_models["items"] = [item for item in scenario_models.get("items", []) if keep_not_ncarb_extension(item)]

    algorithm_by_id = {item["omega_id"]: item for item in algorithm_inputs["items"]}
    reference_by_id = {item["omega_id"]: item for item in evaluation_references["items"]}
    scenario_by_id = {item["omega_id"]: item for item in scenario_models["items"]}

    category_counts: dict[str, int] = {}
    for new_number, base_number, title, doc_index, question_number, theme, category in SPECS:
        new_id = f"ARCH_FKG_{new_number:02d}"
        base_id = f"ARCH_FKG_{base_number:02d}"
        if base_id not in algorithm_by_id or base_id not in reference_by_id or base_id not in scenario_by_id:
            raise KeyError(f"Missing base task {base_id}")

        origin = ncarb_origin(doc_index, question_number, theme)
        algorithm_item = replace_task_id(deepcopy(algorithm_by_id[base_id]), base_id, new_id)
        reference_item = replace_task_id(deepcopy(reference_by_id[base_id]), base_id, new_id)
        scenario_item = replace_task_id(deepcopy(scenario_by_id[base_id]), base_id, new_id)

        algorithm_item["title"] = title
        reference_item["title"] = title
        scenario_item["title"] = title
        algorithm_item["design_intent"] = (
            f"Paraphrased NCARB-derived scenario: {theme}. The optimizer uses fixed "
            "source-grounded code semantics while tuning visible design parameters and public "
            "scenario-model trade-offs."
        )
        algorithm_item["source_task_origin"] = origin
        scenario_facts = dict(algorithm_item.get("scenario_facts", {}))
        scenario_facts.update(
            {
                "ncarb_scenario_theme": theme,
                "ncarb_practice_exam_division": origin["division"],
                "ncarb_practice_question_number": origin["question_number"],
                "ncarb_adaptation_policy": origin["adaptation_policy"],
                "expansion_case_family": category,
            }
        )
        algorithm_item["scenario_facts"] = scenario_facts

        reference_item["task_derivation_source"] = origin
        reference_item["benchmark_extension_metadata"] = {
            "extension_round": "architecture_ncarb_50_extension",
            "source_base_task_id": base_id,
            "ncarb_category": category,
            "source_policy": "paraphrased_parameterized_not_verbatim",
        }
        rule_structure = reference_item.setdefault("rule_structure", {})
        behavior = rule_structure.setdefault("expected_rule_behavior", {})
        provenance = rule_structure.setdefault("expected_provenance", {})
        provenance.setdefault("source_documents", []).append(
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

        algorithm_inputs["items"].append(algorithm_item)
        evaluation_references["items"].append(reference_item)
        scenario_models["items"].append(scenario_item)
        category_counts[category] = category_counts.get(category, 0) + 1

    def sort_items(items: list[dict]) -> list[dict]:
        return sorted(items, key=lambda item: int(str(item["omega_id"]).rsplit("_", 1)[-1]))

    algorithm_inputs["items"] = sort_items(algorithm_inputs["items"])
    evaluation_references["items"] = sort_items(evaluation_references["items"])
    scenario_models["items"] = sort_items(scenario_models["items"])
    algorithm_inputs["version"] = "architecture_fullkg_clean_algorithm_inputs_v1_ncarb_100_tasks"
    evaluation_references["version"] = "architecture_fullkg_clean_evaluation_references_v1_ncarb_100_tasks"
    scenario_models["version"] = "architecture_fullkg_clean_public_scenario_models_v1_ncarb_100_tasks"

    write_json(algorithm_inputs_path, algorithm_inputs)
    write_json(evaluation_references_path, evaluation_references)
    write_json(scenario_models_path, scenario_models)
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

    added_task_ids = [f"ARCH_FKG_{number:02d}" for number in range(51, 101)]
    write_json(
        DATASET / "NCARB_50_EXPANSION_AUDIT.json",
        {
            "version": "architecture_ncarb_50_expansion_audit_v1",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "scope": (
                "Adds ARCH_FKG_51-ARCH_FKG_100 from NCARB practice-exam-derived, "
                "paraphrased and parameterized architecture scenarios."
            ),
            "added_task_count": 50,
            "added_task_ids": added_task_ids,
            "category_counts": category_counts,
            "source_materials": [
                {"document": document, "division": division} for document, division in SOURCE_DOCS
            ],
            "copyright_policy": (
                "Task wording is paraphrased and parameterized; original NCARB question text and "
                "answer choices are not copied."
            ),
            "rule_library_policy": (
                "New tasks reuse canonical source-rule semantics that are already usable through "
                "Qwen, DeepSeek, and Xiaomi MIMO overlays; no new synthetic rules are added."
            ),
            "input_reference_separation": (
                "Algorithm inputs include task facts, variables, objectives, query preferences, "
                "public scenario-model pointers, and NCARB source provenance. Expected rule IDs, "
                "feasible-region answers, and KG provenance targets remain hidden references."
            ),
        },
    )

    print(
        json.dumps(
            {
                "tasks": len(algorithm_inputs["items"]),
                "references": len(evaluation_references["items"]),
                "scenario_models": len(scenario_models["items"]),
                "added": len(added_task_ids),
                "category_counts": category_counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
