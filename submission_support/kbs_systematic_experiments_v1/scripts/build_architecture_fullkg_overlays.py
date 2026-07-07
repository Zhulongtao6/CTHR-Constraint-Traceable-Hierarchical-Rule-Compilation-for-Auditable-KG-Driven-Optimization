from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from build_aviation_fullkg_overlays import (
    alignment_summary,
    build_alignment,
    canonical_reference_rule_ids,
    copy_json,
    item_list,
    model_ids_for,
    per_task_alignment_audit,
    read_json,
    weak_model_ids_for,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
DATASET_ROOT = ROOT / "datasets" / "architecture_fullkg_clean"
CORE_DIR = DATASET_ROOT / "core"
OVERLAY_DIR = DATASET_ROOT / "evaluation_overlays"
RULE_LIBRARY_DIR = DATASET_ROOT / "rule_libraries"

CANONICAL_EVAL_REFS = DATASET_ROOT / "evaluation_references" / "architecture_evaluation_references.json"
CANONICAL_ALGORITHM_INPUTS = DATASET_ROOT / "algorithm_inputs" / "architecture_algorithm_inputs.json"
CANONICAL_SCENARIO_MODELS = DATASET_ROOT / "scenario_models" / "architecture_public_scenario_models.json"
CANONICAL_RULE_LIBRARY = DATASET_ROOT / "rule_libraries" / "full_architecture_rule_library_qwen.json"
CANONICAL_TEMPLATES = DATASET_ROOT / "constraint_templates" / "compiled_rule_constraint_templates.json"
SECTION_6_4_CANONICAL_TEMPLATES = (
    RESULTS_DIR / "constraint_templates" / "architecture_fullkg_clean" / "compiled_rule_constraint_templates.json"
)

RULE_GEN_ROOT = RESULTS_DIR / "kg_to_rule_library"

LIBRARY_SPECS = {
    "qwen": {
        "display_name": "Qwen",
        "source": CANONICAL_RULE_LIBRARY,
        "output_name": "full_architecture_rule_library_qwen.json",
        "alignment_mode": "exact_id",
    },
    "deepseek": {
        "display_name": "DeepSeek",
        "source": RULE_GEN_ROOT / "architecture" / "full_architecture_rule_library_deepseek.json",
        "output_name": "full_architecture_rule_library_deepseek.json",
        "alignment_mode": "semantic_evidence",
    },
    "xiaomi_mimo": {
        "display_name": "Xiaomi MIMO",
        "source": RULE_GEN_ROOT / "architecture_mimo" / "full_architecture_rule_library_mimo.json",
        "output_name": "full_architecture_rule_library_mimo.json",
        "alignment_mode": "semantic_evidence",
    },
}


def remap_reference(reference: dict[str, Any], alignment_by_id: dict[str, dict[str, Any]], model_key: str) -> dict[str, Any]:
    out = json.loads(json.dumps(reference, ensure_ascii=False))
    structure = out.get("rule_structure", {})
    canonical_surviving = [str(rule_id) for rule_id in structure.get("expected_surviving_rule_ids", [])]
    aligned_surviving: list[str] = []
    unresolved: list[str] = []
    per_rule: list[dict[str, Any]] = []
    for canonical_id in canonical_surviving:
        mapped = model_ids_for(canonical_id, alignment_by_id)
        weak = weak_model_ids_for(canonical_id, alignment_by_id)
        if mapped:
            aligned_surviving.extend(mapped)
        else:
            unresolved.append(canonical_id)
        per_rule.append(
            {
                "canonical_rule_id": canonical_id,
                "model_rule_ids": mapped,
                "weak_candidate_model_rule_ids": weak,
                "status": "exact_or_strong_alignment" if mapped else "unresolved",
            }
        )

    aligned_unique = sorted(set(aligned_surviving))
    structure["expected_surviving_rule_ids"] = aligned_unique
    structure["expected_valid_rule_structures"] = [aligned_unique] if aligned_unique else []
    structure["overlay_rule_id_projection"] = {
        "model": model_key,
        "canonical_expected_surviving_rule_ids": canonical_surviving,
        "aligned_expected_surviving_rule_ids": aligned_unique,
        "unresolved_canonical_rule_ids": unresolved,
        "per_rule": per_rule,
        "feasible_region_policy": "feasible_region remains in canonical source-rule semantic space",
        "weak_candidate_policy": "weak_candidate_alignment entries are audit-only and are not placed in expected_surviving_rule_ids",
    }
    out["overlay_metadata"] = {
        "model": model_key,
        "rule_id_namespace": "model_overlay",
        "source_semantic_reference": "core/source_semantic_references/architecture_source_semantic_references.json",
        "only_rule_ids_projected": [
            "rule_structure.expected_surviving_rule_ids",
            "rule_structure.expected_valid_rule_structures",
        ],
        "projection_uses": "exact_or_strong_alignment only",
        "unresolved_canonical_rule_ids": unresolved,
    }
    return out


def build_overlay_references(
    canonical_reference_payload: dict[str, Any],
    alignment: dict[str, Any],
    model_key: str,
) -> dict[str, Any]:
    alignment_by_id = {entry["canonical_rule_id"]: entry for entry in alignment["canonical_to_model"]}
    return {
        "version": "architecture_fullkg_rule_id_overlay_evaluation_references_v1",
        "model": model_key,
        "semantic_reference_policy": (
            "Task semantics, feasible_region constraints, and provenance remain canonical/source-grounded; "
            "only expected surviving rule IDs with exact_or_strong_alignment are projected into the model rule-id "
            "namespace. Weak candidates are audit-only."
        ),
        "items": [remap_reference(item, alignment_by_id, model_key) for item in canonical_reference_payload.get("items", [])],
    }


def remap_templates(
    canonical_templates: dict[str, Any],
    alignment: dict[str, Any],
    model_key: str,
) -> dict[str, Any]:
    alignment_by_id = {entry["canonical_rule_id"]: entry for entry in alignment["canonical_to_model"]}
    templates_by_rule = canonical_templates.get("templates_by_rule", {})
    remapped: dict[str, list[dict[str, Any]]] = {}
    unresolved_template_rule_ids: list[str] = []
    for canonical_rule_id, templates in templates_by_rule.items():
        model_ids = model_ids_for(str(canonical_rule_id), alignment_by_id)
        if not model_ids:
            unresolved_template_rule_ids.append(str(canonical_rule_id))
            continue
        for model_rule_id in model_ids:
            bucket = remapped.setdefault(model_rule_id, [])
            for template in templates:
                item = json.loads(json.dumps(template, ensure_ascii=False))
                item["source_rule_id"] = model_rule_id
                metadata = dict(item.get("metadata", {}))
                metadata.update(
                    {
                        "canonical_rule_id": str(canonical_rule_id),
                        "model_rule_id": model_rule_id,
                        "rule_id_overlay_model": model_key,
                        "rule_id_projection": True,
                    }
                )
                item["metadata"] = metadata
                bucket.append(item)

    return {
        "schema_version": "cthr_rule_constraint_templates.v1",
        "dataset": canonical_templates.get("dataset", "Architecture"),
        "overlay_model": model_key,
        "semantic_template_policy": (
            "Template expressions are copied from canonical source semantics; templates_by_rule keys are projected "
            "into the model rule-id namespace using exact_or_strong_alignment only."
        ),
        "template_count": sum(len(items) for items in remapped.values()),
        "rule_count": len(remapped),
        "source_constraint_occurrence_count": canonical_templates.get("source_constraint_occurrence_count"),
        "unresolved_canonical_template_rule_ids": sorted(unresolved_template_rule_ids),
        "templates_by_rule": {rule_id: remapped[rule_id] for rule_id in sorted(remapped)},
    }


def write_core(canonical_reference_payload: dict[str, Any]) -> None:
    copy_json(CANONICAL_ALGORITHM_INPUTS, CORE_DIR / "algorithm_inputs" / "architecture_algorithm_inputs.json")
    copy_json(CANONICAL_SCENARIO_MODELS, CORE_DIR / "scenario_models" / "architecture_public_scenario_models.json")
    write_json(
        CORE_DIR / "source_semantic_references" / "architecture_source_semantic_references.json",
        {
            "version": "architecture_fullkg_core_source_semantic_references_v1",
            "canonical_rule_id_namespace": "qwen_canonical",
            "semantic_reference_policy": (
                "Fixed source-grounded task semantics, feasible regions, and provenance. This file is not "
                "model-specific and must not be used as a model-generated rule library."
            ),
            "items": canonical_reference_payload.get("items", []),
        },
    )


def sync_canonical_templates_if_needed(canonical_template_source: Path, canonical_templates: dict[str, Any]) -> bool:
    if canonical_template_source.resolve() == CANONICAL_TEMPLATES.resolve():
        return False
    if CANONICAL_TEMPLATES.exists():
        try:
            existing = read_json(CANONICAL_TEMPLATES)
        except json.JSONDecodeError:
            existing = None
        if existing == canonical_templates:
            return False
    try:
        write_json(CANONICAL_TEMPLATES, canonical_templates)
        return True
    except PermissionError:
        return False


def strict_common_task_ids(references: list[dict[str, Any]], summaries_by_model: dict[str, list[dict[str, Any]]]) -> list[str]:
    per_model_task_unresolved = {
        model: {
            str(row["task_id"]): set(str(rule_id) for rule_id in row.get("unresolved_canonical_rule_ids", []))
            for row in rows
        }
        for model, rows in summaries_by_model.items()
    }
    out: list[str] = []
    for reference in references:
        task_id = str(reference.get("omega_id"))
        if all(not per_model_task_unresolved[model].get(task_id, set()) for model in per_model_task_unresolved):
            out.append(task_id)
    return sorted(out)


def main() -> None:
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S")
    canonical_reference_payload = read_json(CANONICAL_EVAL_REFS)
    references = item_list(canonical_reference_payload)
    canonical_rule_library = read_json(CANONICAL_RULE_LIBRARY)
    canonical_template_source = SECTION_6_4_CANONICAL_TEMPLATES if SECTION_6_4_CANONICAL_TEMPLATES.exists() else CANONICAL_TEMPLATES
    canonical_templates = read_json(canonical_template_source)
    templates_synced = sync_canonical_templates_if_needed(canonical_template_source, canonical_templates)
    canonical_lookup = {
        str(rule["rule_id"]): rule
        for rule in canonical_rule_library.get("rules", [])
        if isinstance(rule, dict) and rule.get("rule_id")
    }
    canonical_ids = canonical_reference_rule_ids(references)
    canonical_rules = [canonical_lookup[rule_id] for rule_id in canonical_ids if rule_id in canonical_lookup]
    missing_canonical_rule_ids = sorted(set(canonical_ids) - set(canonical_lookup))

    write_core(canonical_reference_payload)
    all_summaries: list[dict[str, Any]] = []
    all_task_audits: dict[str, Any] = {}

    for model_key, spec in LIBRARY_SPECS.items():
        source = Path(spec["source"])
        if not source.exists():
            raise FileNotFoundError(source)
        model_library = read_json(source)
        model_rule_dir = RULE_LIBRARY_DIR / model_key
        overlay_dir = OVERLAY_DIR / model_key
        copy_json(source, model_rule_dir / str(spec["output_name"]))

        alignment = build_alignment(canonical_rules, model_library, str(spec["alignment_mode"]))
        task_audit = per_task_alignment_audit(references, alignment)
        summary = alignment_summary(model_key, model_library, alignment, task_audit)
        alignment_payload = {
            "version": "architecture_fullkg_rule_id_alignment_v1",
            "generated_at": generated_at,
            "model": model_key,
            "display_name": spec["display_name"],
            "alignment_policy": {
                "canonical_namespace": "qwen_canonical_source_semantics",
                "model_namespace": model_key,
                "many_to_many": True,
                "formal_alignment_classes": [
                    "exact_or_strong_alignment",
                    "weak_candidate_alignment",
                    "unresolved",
                ],
                "unresolved_policy": (
                    "Canonical rules without exact_or_strong_alignment are marked unresolved for main evaluation "
                    "and are not force-mapped."
                ),
                "weak_candidate_policy": (
                    "Chunk-only or confidence-insufficient candidates are retained for audit only and are filtered "
                    "from evaluation references and constraint templates."
                ),
                "signals": [
                    "source_chunk_ids",
                    "source_node_ids",
                    "constraint variable/op/value/unit/source_quote",
                    "rule name",
                    "extraction notes",
                    "relations",
                ],
                "semantic_alignment_threshold": 0.70 if spec["alignment_mode"] != "exact_id" else None,
                "chunk_only_policy": "source_chunk_ids overlap alone is never sufficient for exact_or_strong_alignment.",
            },
            "summary": summary,
            "canonical_to_model": alignment["canonical_to_model"],
            "model_to_canonical": alignment["model_to_canonical"],
            "weak_model_to_canonical": alignment["weak_model_to_canonical"],
            "filtered_weak_alignments": alignment["filtered_weak_alignments"],
            "task_alignment_audit": task_audit,
        }
        write_json(overlay_dir / "rule_id_alignment.json", alignment_payload)
        write_json(
            overlay_dir / "evaluation_references.json",
            build_overlay_references(canonical_reference_payload, alignment, model_key),
        )
        write_json(
            overlay_dir / "compiled_rule_constraint_templates.json",
            remap_templates(canonical_templates, alignment, model_key),
        )
        write_json(
            overlay_dir / "alignment_audit.json",
            {
                "version": "architecture_fullkg_overlay_alignment_audit_v1",
                "model": model_key,
                "summary": summary,
                "canonical_rule_alignment_audit": alignment["canonical_to_model"],
                "filtered_weak_alignments": alignment["filtered_weak_alignments"],
                "task_alignment_audit": task_audit,
            },
        )
        all_summaries.append(summary)
        all_task_audits[model_key] = task_audit

    strict_task_ids = strict_common_task_ids(references, all_task_audits)
    strict_common_payload = {
        "version": "architecture_fullkg_strict_common_tasks_v1",
        "generated_at": generated_at,
        "selection_policy": (
            "A task is included only when every canonical expected surviving rule has exact_or_strong_alignment in "
            "Qwen, DeepSeek, and Xiaomi MIMO overlays."
        ),
        "task_count": len(strict_task_ids),
        "task_ids": strict_task_ids,
        "excluded_task_ids": sorted({str(item.get("omega_id")) for item in references} - set(strict_task_ids)),
    }
    write_json(DATASET_ROOT / "STRICT_COMMON_TASKS.json", strict_common_payload)
    write_json(
        DATASET_ROOT / "ALIGNMENT_AUDIT.json",
        {
            "version": "architecture_fullkg_alignment_audit_v1",
            "generated_at": generated_at,
            "scope": (
                "Architecture full-KG clean benchmark overlays for Qwen, DeepSeek, and Xiaomi MIMO. "
                "GLM is excluded."
            ),
            "summaries": all_summaries,
            "missing_canonical_rule_ids": missing_canonical_rule_ids,
            "task_alignment_audit_by_model": all_task_audits,
            "strict_common_tasks": strict_common_payload,
        },
    )
    overlay_manifest = {
        "version": "architecture_fullkg_overlay_manifest_v1",
        "generated_at": generated_at,
        "core": {
            "algorithm_inputs": str(CORE_DIR / "algorithm_inputs" / "architecture_algorithm_inputs.json"),
            "scenario_models": str(CORE_DIR / "scenario_models" / "architecture_public_scenario_models.json"),
            "source_semantic_references": str(
                CORE_DIR / "source_semantic_references" / "architecture_source_semantic_references.json"
            ),
        },
        "rule_libraries": {
            model_key: str(RULE_LIBRARY_DIR / model_key / str(spec["output_name"]))
            for model_key, spec in LIBRARY_SPECS.items()
        },
        "evaluation_overlays": {
            model_key: {
                "rule_id_alignment": str(OVERLAY_DIR / model_key / "rule_id_alignment.json"),
                "evaluation_references": str(OVERLAY_DIR / model_key / "evaluation_references.json"),
                "compiled_rule_constraint_templates": str(OVERLAY_DIR / model_key / "compiled_rule_constraint_templates.json"),
                "alignment_audit": str(OVERLAY_DIR / model_key / "alignment_audit.json"),
            }
            for model_key in LIBRARY_SPECS
        },
        "alignment_summaries": all_summaries,
        "strict_common_tasks": strict_common_payload,
        "canonical_templates_synced": templates_synced,
    }
    write_json(DATASET_ROOT / "OVERLAY_MANIFEST.json", overlay_manifest)
    write_json(
        DATASET_ROOT / "MANIFEST.json",
        {
            "version": "architecture_fullkg_clean_manifest_v2",
            "generated_at": generated_at,
            "purpose": (
                "Clean architecture full-KG benchmark with fixed source-grounded core semantics and per-model "
                "evaluation overlays for Qwen, DeepSeek, and Xiaomi MIMO."
            ),
            "source_files": {
                "canonical_evaluation_references": str(CANONICAL_EVAL_REFS),
                "canonical_algorithm_inputs": str(CANONICAL_ALGORITHM_INPUTS),
                "canonical_scenario_models": str(CANONICAL_SCENARIO_MODELS),
                "canonical_constraint_templates_source": str(canonical_template_source),
                "qwen_rule_library_source": str(LIBRARY_SPECS["qwen"]["source"]),
                "deepseek_rule_library_source": str(LIBRARY_SPECS["deepseek"]["source"]),
                "xiaomi_mimo_rule_library_source": str(LIBRARY_SPECS["xiaomi_mimo"]["source"]),
            },
            "core": overlay_manifest["core"],
            "rule_libraries": overlay_manifest["rule_libraries"],
            "evaluation_overlays": overlay_manifest["evaluation_overlays"],
            "alignment_summaries": overlay_manifest["alignment_summaries"],
            "strict_common_tasks": overlay_manifest["strict_common_tasks"],
            "canonical_templates_synced": templates_synced,
            "counts": {
                "tasks": len(references),
                "canonical_referenced_rule_ids": len(canonical_ids),
                "missing_canonical_rule_ids": len(missing_canonical_rule_ids),
                "strict_common_tasks": len(strict_task_ids),
            },
        },
    )
    print(json.dumps({"alignment_summaries": all_summaries, "strict_common_tasks": strict_common_payload}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
