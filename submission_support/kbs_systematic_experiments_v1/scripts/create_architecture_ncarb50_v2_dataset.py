from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "datasets" / "architecture_fullkg_clean"
TARGET = ROOT / "datasets" / "architecture_fullkg_ncarb50_v2"
TASK_IDS = {f"ARCH_FKG_{number:02d}" for number in range(51, 101)}
OVERLAY_MODELS = ["qwen", "deepseek", "xiaomi_mimo"]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def filter_items(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    out["items"] = [item for item in payload.get("items", []) if item.get("omega_id") in TASK_IDS]
    return out


def task_number(task_id: str) -> int:
    return int(task_id.rsplit("_", 1)[-1])


def task_sort_key(item: dict[str, Any]) -> int:
    return task_number(str(item.get("omega_id", "ARCH_FKG_0")))


def referenced_rule_ids(reference_payload: dict[str, Any]) -> set[str]:
    rule_ids: set[str] = set()
    for item in reference_payload.get("items", []):
        structure = item.get("rule_structure", {})
        for rule_id in structure.get("expected_source_rule_ids", []):
            rule_ids.add(str(rule_id))
        for rule_id in structure.get("expected_surviving_rule_ids", []):
            rule_ids.add(str(rule_id))
        for constraint in item.get("feasible_region", {}).get("executable_constraints", []):
            if constraint.get("source_type") == "rule_library" and constraint.get("source_id"):
                rule_ids.add(str(constraint["source_id"]))
    return rule_ids


def compact_templates(payload: dict[str, Any], canonical_rule_ids: set[str], model_rule_ids: set[str] | None = None) -> dict[str, Any]:
    out = {key: value for key, value in payload.items() if key != "templates_by_rule"}
    by_rule = payload.get("templates_by_rule", {})
    compact: dict[str, list[dict[str, Any]]] = {}
    for rule_id, templates in by_rule.items():
        keep_rule = str(rule_id) in canonical_rule_ids
        if model_rule_ids is not None:
            keep_rule = keep_rule or str(rule_id) in model_rule_ids
        filtered_templates: list[dict[str, Any]] = []
        for template in templates:
            canonical_rule_id = str(template.get("metadata", {}).get("canonical_rule_id", template.get("source_rule_id", rule_id)))
            if canonical_rule_id not in canonical_rule_ids and not keep_rule:
                continue
            item = json.loads(json.dumps(template, ensure_ascii=False))
            if "observed_bindings" in item:
                item["observed_bindings"] = [
                    binding for binding in item["observed_bindings"] if binding.get("task_id") in TASK_IDS
                ]
            filtered_templates.append(item)
        if filtered_templates:
            compact[str(rule_id)] = filtered_templates
    out["templates_by_rule"] = {key: compact[key] for key in sorted(compact)}
    out["template_count"] = sum(len(items) for items in compact.values())
    out["rule_count"] = len(compact)
    out["source_constraint_occurrence_count"] = sum(
        len(template.get("observed_bindings", [None]))
        for templates in compact.values()
        for template in templates
    )
    return out


def compact_alignment_payload(payload: dict[str, Any], canonical_rule_ids: set[str]) -> dict[str, Any]:
    out = dict(payload)
    out["canonical_to_model"] = [
        row for row in payload.get("canonical_to_model", []) if str(row.get("canonical_rule_id")) in canonical_rule_ids
    ]
    model_ids = {
        str(model_rule_id)
        for row in out["canonical_to_model"]
        for model_rule_id in row.get("aligned_model_rule_ids", [])
    }
    out["model_to_canonical"] = [
        row
        for row in payload.get("model_to_canonical", [])
        if str(row.get("model_rule_id")) in model_ids
        or str(row.get("canonical_rule_id")) in canonical_rule_ids
    ]
    out["weak_model_to_canonical"] = [
        row
        for row in payload.get("weak_model_to_canonical", [])
        if str(row.get("canonical_rule_id")) in canonical_rule_ids
    ]
    out["filtered_weak_alignments"] = [
        row
        for row in payload.get("filtered_weak_alignments", [])
        if str(row.get("canonical_rule_id")) in canonical_rule_ids
    ]
    out["task_alignment_audit"] = [
        row for row in payload.get("task_alignment_audit", []) if row.get("task_id") in TASK_IDS
    ]
    out["summary"] = summarize_alignment(payload, out, canonical_rule_ids)
    return out


def compact_alignment_audit(payload: dict[str, Any], canonical_rule_ids: set[str]) -> dict[str, Any]:
    out = dict(payload)
    out["canonical_rule_alignment_audit"] = [
        row
        for row in payload.get("canonical_rule_alignment_audit", [])
        if str(row.get("canonical_rule_id")) in canonical_rule_ids
    ]
    out["filtered_weak_alignments"] = [
        row
        for row in payload.get("filtered_weak_alignments", [])
        if str(row.get("canonical_rule_id")) in canonical_rule_ids
    ]
    out["task_alignment_audit"] = [
        row for row in payload.get("task_alignment_audit", []) if row.get("task_id") in TASK_IDS
    ]
    out["summary"] = summarize_alignment(payload, out, canonical_rule_ids)
    return out


def summarize_alignment(source_payload: dict[str, Any], compact_payload: dict[str, Any], canonical_rule_ids: set[str]) -> dict[str, Any]:
    source_summary = source_payload.get("summary", {})
    rows = compact_payload.get("canonical_to_model", [])
    strong = [
        row
        for row in rows
        if row.get("status") == "exact_or_strong_alignment"
        and row.get("aligned_model_rule_ids")
    ]
    weak_only = [
        row
        for row in rows
        if not row.get("aligned_model_rule_ids")
        and row.get("weak_candidate_model_rule_ids")
    ]
    unresolved = [
        str(row.get("canonical_rule_id"))
        for row in rows
        if not row.get("aligned_model_rule_ids")
    ]
    tasks_with_unresolved = 0
    for task_row in compact_payload.get("task_alignment_audit", []):
        unresolved_rules = task_row.get("unresolved_canonical_rule_ids", [])
        if unresolved_rules:
            tasks_with_unresolved += 1
    return {
        "model": source_payload.get("model", source_summary.get("model")),
        "model_rule_count": source_summary.get("model_rule_count"),
        "canonical_rule_count": len(canonical_rule_ids),
        "aligned_canonical_rule_count": len(strong),
        "exact_or_strong_aligned_canonical_rule_count": len(strong),
        "weak_candidate_only_canonical_rule_count": len(weak_only),
        "filtered_weak_alignment_count": len(compact_payload.get("filtered_weak_alignments", [])),
        "unresolved_canonical_rule_count": len(unresolved),
        "tasks_with_unresolved_rules": tasks_with_unresolved,
        "unresolved_canonical_rule_ids": sorted(unresolved),
    }


def copy_rule_libraries() -> None:
    source_rule_dir = SOURCE / "rule_libraries"
    target_rule_dir = TARGET / "rule_libraries"
    target_rule_dir.mkdir(parents=True, exist_ok=True)
    for child in source_rule_dir.iterdir():
        destination = target_rule_dir / child.name
        if child.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(child, destination)
        elif child.is_file():
            shutil.copy2(child, destination)


def build_strict_common(filtered_references: dict[str, Any]) -> dict[str, Any]:
    source_strict = read_json(SOURCE / "STRICT_COMMON_TASKS.json")
    task_ids = sorted(
        [task_id for task_id in source_strict.get("task_ids", []) if task_id in TASK_IDS],
        key=task_number,
    )
    all_task_ids = {item["omega_id"] for item in filtered_references.get("items", [])}
    return {
        "version": "architecture_ncarb50_v2_strict_common_tasks_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "selection_policy": source_strict.get("selection_policy"),
        "task_count": len(task_ids),
        "task_ids": task_ids,
        "excluded_task_ids": sorted(all_task_ids - set(task_ids), key=task_number),
        "source_dataset": str(SOURCE),
    }


def write_readme() -> None:
    readme = """# Architecture NCARB50 v2 dataset

This dataset contains the 50 NCARB-practice-exam-derived architecture tasks from `ARCH_FKG_51` to `ARCH_FKG_100`.

The tasks are paraphrased and parameterized from NCARB ARE practice-exam scenario themes. Original question text and answer choices are not copied. The source-rule oracle remains the fixed canonical architecture source semantics, with per-model overlays for Qwen, DeepSeek, and Xiaomi MIMO.

Visible algorithm inputs are under `core/algorithm_inputs` and `core/scenario_models`. Hidden references are under `core/source_semantic_references` and `evaluation_overlays/*/evaluation_references.json`.
"""
    (TARGET / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)

    algorithm_inputs = filter_items(read_json(SOURCE / "algorithm_inputs" / "architecture_algorithm_inputs.json"))
    scenario_models = filter_items(read_json(SOURCE / "scenario_models" / "architecture_public_scenario_models.json"))
    evaluation_references = filter_items(read_json(SOURCE / "evaluation_references" / "architecture_evaluation_references.json"))
    for payload in [algorithm_inputs, scenario_models, evaluation_references]:
        payload["items"] = sorted(payload["items"], key=task_sort_key)

    canonical_rule_ids = referenced_rule_ids(evaluation_references)

    write_json(TARGET / "algorithm_inputs" / "architecture_algorithm_inputs.json", algorithm_inputs)
    write_json(TARGET / "scenario_models" / "architecture_public_scenario_models.json", scenario_models)
    write_json(TARGET / "evaluation_references" / "architecture_evaluation_references.json", evaluation_references)
    write_json(TARGET / "core" / "algorithm_inputs" / "architecture_algorithm_inputs.json", algorithm_inputs)
    write_json(TARGET / "core" / "scenario_models" / "architecture_public_scenario_models.json", scenario_models)
    write_json(
        TARGET / "core" / "source_semantic_references" / "architecture_source_semantic_references.json",
        {
            "version": "architecture_ncarb50_v2_core_source_semantic_references_v1",
            "canonical_rule_id_namespace": "qwen_canonical",
            "semantic_reference_policy": (
                "Fixed source-grounded task semantics, feasible regions, and provenance for the NCARB50 v2 subset."
            ),
            "items": evaluation_references["items"],
        },
    )

    references_by_id = {item["omega_id"]: item for item in evaluation_references["items"]}
    for algorithm_item in algorithm_inputs["items"]:
        task_id = algorithm_item["omega_id"]
        write_json(
            TARGET / "tasks" / f"{task_id}.json",
            {
                "version": "architecture_ncarb50_v2_task_v1",
                "algorithm_input": algorithm_item,
                "evaluation_reference": references_by_id[task_id],
            },
        )

    copy_rule_libraries()

    canonical_templates = compact_templates(
        read_json(SOURCE / "constraint_templates" / "compiled_rule_constraint_templates.json"),
        canonical_rule_ids,
    )
    write_json(TARGET / "constraint_templates" / "compiled_rule_constraint_templates.json", canonical_templates)

    overlay_manifest: dict[str, Any] = {
        "version": "architecture_ncarb50_v2_overlay_manifest_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "core": {
            "algorithm_inputs": str(TARGET / "core" / "algorithm_inputs" / "architecture_algorithm_inputs.json"),
            "scenario_models": str(TARGET / "core" / "scenario_models" / "architecture_public_scenario_models.json"),
            "source_semantic_references": str(
                TARGET / "core" / "source_semantic_references" / "architecture_source_semantic_references.json"
            ),
        },
        "rule_libraries": {
            "qwen": str(TARGET / "rule_libraries" / "qwen" / "full_architecture_rule_library_qwen.json"),
            "deepseek": str(TARGET / "rule_libraries" / "deepseek" / "full_architecture_rule_library_deepseek.json"),
            "xiaomi_mimo": str(TARGET / "rule_libraries" / "xiaomi_mimo" / "full_architecture_rule_library_mimo.json"),
        },
        "evaluation_overlays": {},
    }

    overlay_summaries: list[dict[str, Any]] = []
    for model in OVERLAY_MODELS:
        source_overlay = SOURCE / "evaluation_overlays" / model
        target_overlay = TARGET / "evaluation_overlays" / model
        overlay_refs = filter_items(read_json(source_overlay / "evaluation_references.json"))
        overlay_rule_ids = referenced_rule_ids(overlay_refs)
        alignment = compact_alignment_payload(read_json(source_overlay / "rule_id_alignment.json"), canonical_rule_ids)
        alignment_audit = compact_alignment_audit(read_json(source_overlay / "alignment_audit.json"), canonical_rule_ids)
        templates = compact_templates(
            read_json(source_overlay / "compiled_rule_constraint_templates.json"),
            canonical_rule_ids,
            overlay_rule_ids,
        )
        write_json(target_overlay / "evaluation_references.json", overlay_refs)
        write_json(target_overlay / "rule_id_alignment.json", alignment)
        write_json(target_overlay / "alignment_audit.json", alignment_audit)
        write_json(target_overlay / "compiled_rule_constraint_templates.json", templates)
        overlay_manifest["evaluation_overlays"][model] = {
            "rule_id_alignment": str(target_overlay / "rule_id_alignment.json"),
            "evaluation_references": str(target_overlay / "evaluation_references.json"),
            "compiled_rule_constraint_templates": str(target_overlay / "compiled_rule_constraint_templates.json"),
            "alignment_audit": str(target_overlay / "alignment_audit.json"),
        }
        overlay_summaries.append(alignment.get("summary", {}))

    strict_common = build_strict_common(evaluation_references)
    write_json(TARGET / "STRICT_COMMON_TASKS.json", strict_common)
    overlay_manifest["alignment_summaries"] = overlay_summaries
    overlay_manifest["strict_common_tasks"] = strict_common
    write_json(TARGET / "OVERLAY_MANIFEST.json", overlay_manifest)

    source_relation = read_json(SOURCE / "RELATION_COVERAGE_100_TASKS_AUDIT.json")
    relation_subset = {
        "version": "architecture_ncarb50_v2_relation_coverage_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "task_count": len(evaluation_references["items"]),
        "strict_common_task_count": strict_common["task_count"],
        "source_policy": source_relation.get("source_policy"),
        "model_overlay_policy": source_relation.get("model_overlay_policy"),
        "per_task_coverage": [
            row for row in source_relation.get("per_task_coverage", []) if row.get("task_id") in TASK_IDS
        ],
    }
    write_json(TARGET / "RELATION_COVERAGE_NCARB50_AUDIT.json", relation_subset)

    source_expansion = read_json(SOURCE / "NCARB_50_EXPANSION_AUDIT.json")
    write_json(TARGET / "NCARB_50_EXPANSION_AUDIT.json", source_expansion)

    leakage_audit = {
        "version": "architecture_ncarb50_v2_leakage_audit_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "task_count": len(algorithm_inputs["items"]),
        "forbidden_input_key_hit_count": 0,
        "forbidden_input_key_hits": [],
        "note": "The parent architecture_fullkg_clean leakage audit reported zero forbidden input key hits after NCARB50 construction.",
    }
    write_json(TARGET / "LEAKAGE_AUDIT.json", leakage_audit)

    manifest = {
        "version": "architecture_ncarb50_v2_manifest_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Standalone new-round architecture dataset containing only the 50 NCARB-derived tasks.",
        "source_dataset": str(SOURCE),
        "task_count": len(evaluation_references["items"]),
        "task_id_range": ["ARCH_FKG_51", "ARCH_FKG_100"],
        "canonical_referenced_rule_ids": len(canonical_rule_ids),
        "strict_common_tasks": strict_common["task_count"],
        "core": overlay_manifest["core"],
        "rule_libraries": overlay_manifest["rule_libraries"],
        "evaluation_overlays": overlay_manifest["evaluation_overlays"],
        "audits": {
            "ncarb_expansion": str(TARGET / "NCARB_50_EXPANSION_AUDIT.json"),
            "relation_coverage": str(TARGET / "RELATION_COVERAGE_NCARB50_AUDIT.json"),
            "leakage": str(TARGET / "LEAKAGE_AUDIT.json"),
            "strict_common_tasks": str(TARGET / "STRICT_COMMON_TASKS.json"),
        },
    }
    write_json(TARGET / "MANIFEST.json", manifest)
    write_readme()

    print(
        json.dumps(
            {
                "target": str(TARGET),
                "task_count": manifest["task_count"],
                "canonical_referenced_rule_ids": manifest["canonical_referenced_rule_ids"],
                "strict_common_tasks": manifest["strict_common_tasks"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
