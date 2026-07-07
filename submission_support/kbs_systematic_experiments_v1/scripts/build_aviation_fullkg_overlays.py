from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CTHR_ROOT = ROOT.parents[1]
RESULTS_DIR = ROOT / "results"
DATASET_ROOT = ROOT / "datasets" / "aviation_fullkg_clean"
CORE_DIR = DATASET_ROOT / "core"
OVERLAY_DIR = DATASET_ROOT / "evaluation_overlays"
RULE_LIBRARY_DIR = DATASET_ROOT / "rule_libraries"

MODEL_COMPARE_ROOT = CTHR_ROOT / "paper" / "full_aviation_kg_rule_library_model_comparison"
MIMO_ROOT = CTHR_ROOT / "paper" / "mimo_aviation_kg_rule_library"

CANONICAL_EVAL_REFS = DATASET_ROOT / "evaluation_references" / "aviation_evaluation_references.json"
CANONICAL_ALGORITHM_INPUTS = DATASET_ROOT / "algorithm_inputs" / "aviation_algorithm_inputs.json"
CANONICAL_SCENARIO_MODELS = DATASET_ROOT / "scenario_models" / "aviation_public_scenario_models.json"
CANONICAL_TEMPLATES = DATASET_ROOT / "constraint_templates" / "compiled_rule_constraint_templates.json"
SECTION_6_4_CANONICAL_TEMPLATES = (
    RESULTS_DIR / "constraint_templates" / "aviation_fullkg_clean" / "compiled_rule_constraint_templates.json"
)


LIBRARY_SPECS = {
    "qwen": {
        "display_name": "Qwen",
        "source": MODEL_COMPARE_ROOT / "full_aviation_rule_library_qwen.json",
        "output_name": "full_aviation_rule_library_qwen.json",
        "alignment_mode": "exact_id",
    },
    "deepseek": {
        "display_name": "DeepSeek strict repaired",
        "source": MODEL_COMPARE_ROOT / "full_aviation_rule_library_deepseek_strict_repaired.json",
        "output_name": "full_aviation_rule_library_deepseek_strict_repaired.json",
        "alignment_mode": "semantic_evidence",
    },
    "xiaomi_mimo": {
        "display_name": "Xiaomi MIMO",
        "source": MIMO_ROOT / "full_aviation_rule_library_mimo.json",
        "output_name": "full_aviation_rule_library_mimo.json",
        "alignment_mode": "semantic_evidence",
    },
}


STOP_TOKENS = {
    "and",
    "for",
    "from",
    "maximum",
    "minimum",
    "min",
    "max",
    "procedure",
    "requirement",
    "required",
    "rule",
    "section",
    "shall",
    "the",
    "unknown",
    "with",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if path.exists():
        try:
            if path.read_text(encoding="utf-8") == text:
                return
        except OSError:
            pass
    try:
        if path.exists():
            with path.open("r+", encoding="utf-8") as handle:
                handle.seek(0)
                handle.write(text)
                handle.truncate()
        else:
            path.write_text(text, encoding="utf-8")
    except PermissionError:
        if path.exists() and path.read_text(encoding="utf-8") == text:
            return
        raise


def copy_json(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        try:
            if src.read_bytes() == dst.read_bytes():
                return
        except OSError:
            pass
    try:
        shutil.copy2(src, dst)
    except PermissionError:
        if dst.exists() and src.read_bytes() == dst.read_bytes():
            return
        raise


def item_list(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return list(payload.get("items", []))


def token_set(value: Any) -> set[str]:
    tokens = re.split(r"[^a-z0-9\u4e00-\u9fff]+", str(value or "").lower())
    return {token for token in tokens if len(token) > 1 and token not in STOP_TOKENS}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    denom = len(left | right)
    return len(left & right) / denom if denom else 0.0


def numeric_values(value: Any) -> set[float]:
    return {round(float(item), 6) for item in re.findall(r"[-+]?[0-9]+(?:\.[0-9]+)?", str(value or ""))}


def normalized_unit(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"s", "sec", "secs", "second", "seconds"}:
        return "second"
    if text in {"deg", "degree", "degrees", "°"}:
        return "degree"
    if text in {"m", "米", "meter", "meters", "metre", "metres"}:
        return "meter"
    if text in {"%", "percent", "percentage"}:
        return "percent"
    if text in {"ft", "feet", "foot"}:
        return "foot"
    if text in {"km", "kilometer", "kilometers", "kilometre", "kilometres"}:
        return "kilometer"
    if text in {"nm", "nautical mile", "nautical miles"}:
        return "nautical_mile"
    return text


def value_matches(left: Any, right: Any) -> bool:
    left_text = str(left or "").strip().lower()
    right_text = str(right or "").strip().lower()
    if left_text and left_text == right_text:
        return True
    left_numbers = numeric_values(left_text)
    right_numbers = numeric_values(right_text)
    if left_numbers and right_numbers and left_numbers & right_numbers:
        return True
    return bool(token_set(left_text) and jaccard(token_set(left_text), token_set(right_text)) >= 0.6)


def evidence_ids(rule: dict[str, Any], kind: str) -> set[str]:
    if kind == "chunk":
        direct_key = "source_chunk_ids"
        evidence_key = "chunk_ids"
    else:
        direct_key = "source_node_ids"
        evidence_key = "kg_node_ids"
    out = {str(item) for item in rule.get(direct_key, [])}
    for constraint in rule.get("constraints", []):
        if isinstance(constraint, dict):
            out.update(str(item) for item in constraint.get("evidence", {}).get(evidence_key, []))
    for relation in rule.get("relations", []):
        if isinstance(relation, dict):
            out.update(str(item) for item in relation.get("evidence", {}).get(evidence_key, []))
    if kind == "chunk":
        for provenance in rule.get("provenance", []):
            chunk_id = provenance.get("chunk_id") if isinstance(provenance, dict) else None
            if chunk_id:
                out.add(str(chunk_id))
    return out


def best_constraint_signals(canonical_rule: dict[str, Any], model_rule: dict[str, Any]) -> dict[str, Any]:
    best: dict[str, Any] = {
        "score": 0.0,
        "variable_similarity": 0.0,
        "source_quote_similarity": 0.0,
        "value_match": False,
        "op_match": False,
        "unit_match": False,
        "explicit_constraint_evidence": False,
    }
    for canonical_constraint in canonical_rule.get("constraints", []):
        for model_constraint in model_rule.get("constraints", []):
            variable_similarity = jaccard(
                token_set(canonical_constraint.get("variable")),
                token_set(model_constraint.get("variable")),
            )
            source_quote_similarity = jaccard(
                token_set(canonical_constraint.get("source_quote")),
                token_set(model_constraint.get("source_quote")),
            )
            value_match = value_matches(canonical_constraint.get("value"), model_constraint.get("value"))
            op_match = str(canonical_constraint.get("op")) == str(model_constraint.get("op"))
            left_unit = normalized_unit(canonical_constraint.get("unit"))
            right_unit = normalized_unit(model_constraint.get("unit"))
            unit_match = bool(left_unit and right_unit and left_unit == right_unit and left_unit != "unknown")
            score = (
                0.25 * variable_similarity
                + 0.25 * source_quote_similarity
                + (0.20 if value_match else 0.0)
                + (0.15 if op_match else 0.0)
                + (0.10 if unit_match else 0.0)
            )
            explicit = bool(
                (value_match and (variable_similarity >= 0.20 or source_quote_similarity >= 0.25))
                and (op_match or unit_match or source_quote_similarity >= 0.45)
            ) or bool(source_quote_similarity >= 0.65 and (variable_similarity >= 0.20 or value_match))
            if score > float(best["score"]):
                best = {
                    "score": round(score, 3),
                    "variable_similarity": round(variable_similarity, 3),
                    "source_quote_similarity": round(source_quote_similarity, 3),
                    "value_match": value_match,
                    "op_match": op_match,
                    "unit_match": unit_match,
                    "explicit_constraint_evidence": explicit,
                }
    return best


def alignment_score(canonical_rule: dict[str, Any], model_rule: dict[str, Any]) -> dict[str, Any]:
    canonical_chunks = evidence_ids(canonical_rule, "chunk")
    model_chunks = evidence_ids(model_rule, "chunk")
    canonical_nodes = evidence_ids(canonical_rule, "node")
    model_nodes = evidence_ids(model_rule, "node")
    chunk_overlap = bool(canonical_chunks & model_chunks)
    node_overlap = bool(canonical_nodes & model_nodes)
    canonical_relations = {str(item.get("type")) for item in canonical_rule.get("relations", []) if isinstance(item, dict)}
    model_relations = {str(item.get("type")) for item in model_rule.get("relations", []) if isinstance(item, dict)}
    constraint = best_constraint_signals(canonical_rule, model_rule)
    name_similarity = jaccard(token_set(canonical_rule.get("name")), token_set(model_rule.get("name")))
    notes_similarity = jaccard(token_set(canonical_rule.get("extraction_notes")), token_set(model_rule.get("extraction_notes")))
    relation_type_similarity = jaccard(canonical_relations, model_relations)

    score = 0.0
    if canonical_rule.get("rule_id") == model_rule.get("rule_id"):
        score += 100.0
    else:
        score += 8.0 * jaccard(canonical_chunks, model_chunks)
        score += 2.0 if chunk_overlap else 0.0
        score += 35.0 * jaccard(canonical_nodes, model_nodes)
        score += 10.0 if node_overlap else 0.0
        score += 35.0 * float(constraint["score"])
        score += 20.0 * name_similarity
        score += 8.0 * notes_similarity
        score += 5.0 * relation_type_similarity
        if canonical_rule.get("rule_type") == model_rule.get("rule_type"):
            score += 2.0

    only_chunk_evidence = bool(
        chunk_overlap
        and not node_overlap
        and not constraint["explicit_constraint_evidence"]
        and name_similarity < 0.20
        and notes_similarity < 0.20
    )
    confidence = 1.0 if canonical_rule.get("rule_id") == model_rule.get("rule_id") else round(min(1.0, score / 75.0), 3)
    strong_below_threshold = bool(
        confidence < 0.70
        and constraint["explicit_constraint_evidence"]
        and chunk_overlap
        and (
            float(constraint["score"]) >= 0.65
            or (
                float(constraint["source_quote_similarity"]) >= 0.65
                and (
                    constraint["value_match"]
                    or float(constraint["variable_similarity"]) >= 0.20
                    or constraint["op_match"]
                )
            )
        )
    )
    strong_alignment = bool(canonical_rule.get("rule_id") == model_rule.get("rule_id")) or bool(
        not only_chunk_evidence
        and (
            (
                confidence >= 0.70
                and (
                    (node_overlap and chunk_overlap)
                    or
                    (node_overlap and (constraint["explicit_constraint_evidence"] or name_similarity >= 0.20))
                    or (chunk_overlap and constraint["explicit_constraint_evidence"])
                    or (name_similarity >= 0.35 and (node_overlap or chunk_overlap))
                )
            )
            or strong_below_threshold
        )
    )

    return {
        "score": round(score, 3),
        "confidence": confidence,
        "chunk_overlap_count": len(canonical_chunks & model_chunks),
        "node_overlap_count": len(canonical_nodes & model_nodes),
        "constraint_score": constraint["score"],
        "constraint_signals": constraint,
        "name_similarity": round(name_similarity, 3),
        "notes_similarity": round(notes_similarity, 3),
        "relation_type_similarity": round(relation_type_similarity, 3),
        "only_chunk_evidence": only_chunk_evidence,
        "strong_below_threshold_due_to_explicit_constraint_evidence": strong_below_threshold,
        "strong_alignment": strong_alignment,
    }


def canonical_reference_rule_ids(references: list[dict[str, Any]]) -> list[str]:
    out: set[str] = set()
    for reference in references:
        structure = reference.get("rule_structure", {})
        out.update(str(rule_id) for rule_id in structure.get("expected_source_rule_ids", []))
        out.update(str(rule_id) for rule_id in structure.get("expected_surviving_rule_ids", []))
        out.update(str(rule_id) for rule_id in structure.get("expected_defeated_rule_ids", []))
    return sorted(out)


def choose_alignments(
    canonical_rule: dict[str, Any],
    model_rules: list[dict[str, Any]],
    mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    canonical_id = str(canonical_rule.get("rule_id"))
    if mode == "exact_id":
        exact = [rule for rule in model_rules if str(rule.get("rule_id")) == canonical_id]
        if not exact:
            return [], [], []
        return [
            {
                "model_rule_id": canonical_id,
                "alignment_type": "exact_or_strong_alignment",
                "confidence": 1.0,
                "score": 100.0,
                "signals": {"exact_id": True},
            }
        ], [], []

    scored: list[dict[str, Any]] = []
    for rule in model_rules:
        signals = alignment_score(canonical_rule, rule)
        scored.append(
            {
                "model_rule_id": str(rule.get("rule_id")),
                "alignment_type": (
                    "exact_or_strong_alignment" if signals.get("strong_alignment") else "weak_candidate_alignment"
                ),
                "confidence": signals["confidence"],
                "score": signals["score"],
                "signals": signals,
            }
        )
    scored.sort(key=lambda item: (-float(item["score"]), item["model_rule_id"]))
    strong = [item for item in scored if item["alignment_type"] == "exact_or_strong_alignment"]
    weak = [item for item in scored if item["alignment_type"] == "weak_candidate_alignment"]
    if strong:
        best = float(strong[0]["score"])
        strong = [
            item
            for item in strong
            if float(item["score"]) >= best - 8.0 or float(item["confidence"]) >= 0.90
        ][:4]
    return strong, weak[:8], scored[:8]


def build_alignment(
    canonical_rules: list[dict[str, Any]],
    model_library: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    model_rules = list(model_library.get("rules", []))
    entries: list[dict[str, Any]] = []
    model_to_canonical: dict[str, list[str]] = {}
    weak_model_to_canonical: dict[str, list[str]] = {}
    unresolved: list[str] = []
    filtered_weak_alignments: list[dict[str, Any]] = []
    for canonical_rule in canonical_rules:
        canonical_id = str(canonical_rule["rule_id"])
        selected, weak, candidates = choose_alignments(canonical_rule, model_rules, mode)
        status = "exact_or_strong_alignment" if selected else "unresolved"
        if not selected:
            unresolved.append(canonical_id)
        for item in selected:
            model_to_canonical.setdefault(item["model_rule_id"], []).append(canonical_id)
        for item in weak:
            weak_model_to_canonical.setdefault(item["model_rule_id"], []).append(canonical_id)
            filtered_weak_alignments.append(
                {
                    "canonical_rule_id": canonical_id,
                    "model_rule_id": item["model_rule_id"],
                    "score": item["score"],
                    "confidence": item["confidence"],
                    "signals": item["signals"],
                }
            )
        entries.append(
            {
                "canonical_rule_id": canonical_id,
                "canonical_rule_name": canonical_rule.get("name"),
                "status": status,
                "aligned_model_rule_ids": [item["model_rule_id"] for item in selected],
                "weak_candidate_model_rule_ids": [item["model_rule_id"] for item in weak],
                "exact_or_strong_alignment": selected,
                "weak_candidate_alignment": weak,
                "unresolved": not bool(selected),
                "top_candidates_for_audit": candidates,
            }
        )
    return {
        "canonical_to_model": entries,
        "model_to_canonical": [
            {"model_rule_id": model_id, "canonical_rule_ids": sorted(ids)}
            for model_id, ids in sorted(model_to_canonical.items())
        ],
        "weak_model_to_canonical": [
            {"model_rule_id": model_id, "canonical_rule_ids": sorted(ids)}
            for model_id, ids in sorted(weak_model_to_canonical.items())
        ],
        "unresolved_canonical_rule_ids": sorted(unresolved),
        "filtered_weak_alignments": filtered_weak_alignments,
    }


def model_ids_for(canonical_rule_id: str, alignment_by_id: dict[str, dict[str, Any]]) -> list[str]:
    return list(alignment_by_id.get(canonical_rule_id, {}).get("aligned_model_rule_ids", []))


def weak_model_ids_for(canonical_rule_id: str, alignment_by_id: dict[str, dict[str, Any]]) -> list[str]:
    return list(alignment_by_id.get(canonical_rule_id, {}).get("weak_candidate_model_rule_ids", []))


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

    structure["expected_surviving_rule_ids"] = sorted(set(aligned_surviving))
    structure["expected_valid_rule_structures"] = [sorted(set(aligned_surviving))]
    structure.setdefault("overlay_rule_id_projection", {})
    structure["overlay_rule_id_projection"] = {
        "model": model_key,
        "canonical_expected_surviving_rule_ids": canonical_surviving,
        "aligned_expected_surviving_rule_ids": sorted(set(aligned_surviving)),
        "unresolved_canonical_rule_ids": unresolved,
        "per_rule": per_rule,
        "feasible_region_policy": "feasible_region remains in canonical source-rule semantic space",
        "weak_candidate_policy": "weak_candidate_alignment entries are audit-only and are not placed in expected_surviving_rule_ids",
    }
    out["overlay_metadata"] = {
        "model": model_key,
        "rule_id_namespace": "model_overlay",
        "source_semantic_reference": "core/source_semantic_references/aviation_source_semantic_references.json",
        "only_rule_ids_projected": ["rule_structure.expected_surviving_rule_ids", "rule_structure.expected_valid_rule_structures"],
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
        "version": "aviation_fullkg_rule_id_overlay_evaluation_references_v1",
        "model": model_key,
        "semantic_reference_policy": "Task semantics, feasible_region constraints, and provenance remain canonical/source-grounded; only expected surviving rule IDs with exact_or_strong_alignment are projected into the model rule-id namespace. Weak candidates are audit-only.",
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
        "dataset": canonical_templates.get("dataset", "Aviation"),
        "overlay_model": model_key,
        "semantic_template_policy": "Template expressions are copied from canonical source semantics; templates_by_rule keys are projected into the model rule-id namespace using exact_or_strong_alignment only.",
        "template_count": sum(len(items) for items in remapped.values()),
        "rule_count": len(remapped),
        "source_constraint_occurrence_count": canonical_templates.get("source_constraint_occurrence_count"),
        "unresolved_canonical_template_rule_ids": sorted(unresolved_template_rule_ids),
        "templates_by_rule": {rule_id: remapped[rule_id] for rule_id in sorted(remapped)},
    }


def per_task_alignment_audit(
    references: list[dict[str, Any]],
    alignment: dict[str, Any],
) -> list[dict[str, Any]]:
    alignment_by_id = {entry["canonical_rule_id"]: entry for entry in alignment["canonical_to_model"]}
    rows = []
    for reference in references:
        structure = reference.get("rule_structure", {})
        canonical_ids = [str(rule_id) for rule_id in structure.get("expected_surviving_rule_ids", [])]
        rows.append(
            {
                "task_id": reference.get("omega_id"),
                "canonical_surviving_rule_ids": canonical_ids,
                "aligned_model_rule_ids": sorted(
                    {
                        model_id
                        for canonical_id in canonical_ids
                        for model_id in model_ids_for(canonical_id, alignment_by_id)
                    }
                ),
                "weak_candidate_model_rule_ids": sorted(
                    {
                        model_id
                        for canonical_id in canonical_ids
                        for model_id in weak_model_ids_for(canonical_id, alignment_by_id)
                    }
                ),
                "unresolved_canonical_rule_ids": [
                    canonical_id for canonical_id in canonical_ids if not model_ids_for(canonical_id, alignment_by_id)
                ],
                "per_rule": [
                    {
                        "canonical_rule_id": canonical_id,
                        "strong_model_rule_ids": model_ids_for(canonical_id, alignment_by_id),
                        "weak_candidate_model_rule_ids": weak_model_ids_for(canonical_id, alignment_by_id),
                        "status": "exact_or_strong_alignment"
                        if model_ids_for(canonical_id, alignment_by_id)
                        else "unresolved",
                    }
                    for canonical_id in canonical_ids
                ],
            }
        )
    return rows


def alignment_summary(
    model_key: str,
    model_library: dict[str, Any],
    alignment: dict[str, Any],
    task_audit: list[dict[str, Any]],
) -> dict[str, Any]:
    aligned = [entry for entry in alignment["canonical_to_model"] if entry["aligned_model_rule_ids"]]
    weak = [
        entry
        for entry in alignment["canonical_to_model"]
        if entry.get("weak_candidate_model_rule_ids") and not entry.get("aligned_model_rule_ids")
    ]
    unresolved = alignment["unresolved_canonical_rule_ids"]
    task_unresolved = [row for row in task_audit if row["unresolved_canonical_rule_ids"]]
    return {
        "model": model_key,
        "model_rule_count": len(model_library.get("rules", [])),
        "canonical_rule_count": len(alignment["canonical_to_model"]),
        "aligned_canonical_rule_count": len(aligned),
        "exact_or_strong_aligned_canonical_rule_count": len(aligned),
        "weak_candidate_only_canonical_rule_count": len(weak),
        "filtered_weak_alignment_count": len(alignment.get("filtered_weak_alignments", [])),
        "unresolved_canonical_rule_count": len(unresolved),
        "tasks_with_unresolved_rules": len(task_unresolved),
        "unresolved_canonical_rule_ids": unresolved,
    }


def write_core(canonical_reference_payload: dict[str, Any]) -> None:
    copy_json(CANONICAL_ALGORITHM_INPUTS, CORE_DIR / "algorithm_inputs" / "aviation_algorithm_inputs.json")
    copy_json(CANONICAL_SCENARIO_MODELS, CORE_DIR / "scenario_models" / "aviation_public_scenario_models.json")
    write_json(
        CORE_DIR / "source_semantic_references" / "aviation_source_semantic_references.json",
        {
            "version": "aviation_fullkg_core_source_semantic_references_v1",
            "canonical_rule_id_namespace": "qwen_canonical",
            "semantic_reference_policy": "Fixed source-grounded task semantics, feasible regions, and provenance. This file is not model-specific and must not be used as a model-generated rule library.",
            "items": canonical_reference_payload.get("items", []),
        },
    )


def strict_common_task_ids(references: list[dict[str, Any]], task_audits_by_model: dict[str, list[dict[str, Any]]]) -> list[str]:
    per_model_task_unresolved = {
        model: {
            str(row["task_id"]): set(str(rule_id) for rule_id in row.get("unresolved_canonical_rule_ids", []))
            for row in rows
        }
        for model, rows in task_audits_by_model.items()
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
    canonical_rule_library = read_json(DATASET_ROOT / "rule_libraries" / "full_aviation_rule_library_qwen.json")
    canonical_template_source = SECTION_6_4_CANONICAL_TEMPLATES if SECTION_6_4_CANONICAL_TEMPLATES.exists() else CANONICAL_TEMPLATES
    canonical_templates = read_json(canonical_template_source)
    write_json(CANONICAL_TEMPLATES, canonical_templates)
    canonical_lookup = {
        str(rule["rule_id"]): rule
        for rule in canonical_rule_library.get("rules", [])
        if isinstance(rule, dict) and rule.get("rule_id")
    }
    canonical_ids = canonical_reference_rule_ids(references)
    canonical_rules = [canonical_lookup[rule_id] for rule_id in canonical_ids if rule_id in canonical_lookup]

    write_core(canonical_reference_payload)
    all_summaries: list[dict[str, Any]] = []
    all_task_audits: dict[str, Any] = {}

    for model_key, spec in LIBRARY_SPECS.items():
        model_library = read_json(Path(spec["source"]))
        model_rule_dir = RULE_LIBRARY_DIR / model_key
        overlay_dir = OVERLAY_DIR / model_key
        copy_json(Path(spec["source"]), model_rule_dir / str(spec["output_name"]))

        alignment = build_alignment(canonical_rules, model_library, str(spec["alignment_mode"]))
        task_audit = per_task_alignment_audit(references, alignment)
        summary = alignment_summary(model_key, model_library, alignment, task_audit)
        alignment_payload = {
            "version": "aviation_fullkg_rule_id_alignment_v1",
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
                "unresolved_policy": "Canonical rules without exact_or_strong_alignment are marked unresolved for main evaluation and are not force-mapped.",
                "weak_candidate_policy": "Chunk-only or confidence-insufficient candidates are retained for audit only and are filtered from evaluation references and constraint templates.",
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
        write_json(overlay_dir / "evaluation_references.json", build_overlay_references(canonical_reference_payload, alignment, model_key))
        write_json(overlay_dir / "compiled_rule_constraint_templates.json", remap_templates(canonical_templates, alignment, model_key))
        write_json(
            overlay_dir / "alignment_audit.json",
            {
                "version": "aviation_fullkg_overlay_alignment_audit_v1",
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
        "version": "aviation_fullkg_strict_common_tasks_v1",
        "generated_at": generated_at,
        "selection_policy": (
            "A task is included only when every canonical expected surviving rule has exact_or_strong_alignment in "
            "Qwen, DeepSeek strict repaired, and Xiaomi MIMO overlays."
        ),
        "task_count": len(strict_task_ids),
        "task_ids": strict_task_ids,
        "excluded_task_ids": sorted({str(item.get("omega_id")) for item in references} - set(strict_task_ids)),
    }
    write_json(DATASET_ROOT / "STRICT_COMMON_TASKS.json", strict_common_payload)
    write_json(
        DATASET_ROOT / "ALIGNMENT_AUDIT.json",
        {
            "version": "aviation_fullkg_alignment_audit_v1",
            "generated_at": generated_at,
            "scope": "Aviation full-KG clean benchmark overlays for Qwen, DeepSeek strict repaired, and Xiaomi MIMO. GLM is excluded.",
            "summaries": all_summaries,
            "task_alignment_audit_by_model": all_task_audits,
            "strict_common_tasks": strict_common_payload,
        },
    )
    write_json(
        DATASET_ROOT / "OVERLAY_MANIFEST.json",
        {
            "version": "aviation_fullkg_overlay_manifest_v1",
            "generated_at": generated_at,
            "core": {
                "algorithm_inputs": str(CORE_DIR / "algorithm_inputs" / "aviation_algorithm_inputs.json"),
                "scenario_models": str(CORE_DIR / "scenario_models" / "aviation_public_scenario_models.json"),
                "source_semantic_references": str(
                    CORE_DIR / "source_semantic_references" / "aviation_source_semantic_references.json"
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
                    "compiled_rule_constraint_templates": str(
                        OVERLAY_DIR / model_key / "compiled_rule_constraint_templates.json"
                    ),
                    "alignment_audit": str(OVERLAY_DIR / model_key / "alignment_audit.json"),
                }
                for model_key in LIBRARY_SPECS
            },
            "alignment_summaries": all_summaries,
            "strict_common_tasks": strict_common_payload,
        },
    )
    write_json(
        DATASET_ROOT / "MANIFEST.json",
        {
            "version": "aviation_fullkg_clean_manifest_v2",
            "generated_at": generated_at,
            "purpose": (
                "Clean aviation full-KG benchmark with fixed source-grounded core semantics and "
                "per-model evaluation overlays for Qwen, DeepSeek strict repaired, and Xiaomi MIMO."
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
            "core": {
                "algorithm_inputs": str(CORE_DIR / "algorithm_inputs" / "aviation_algorithm_inputs.json"),
                "scenario_models": str(CORE_DIR / "scenario_models" / "aviation_public_scenario_models.json"),
                "source_semantic_references": str(
                    CORE_DIR / "source_semantic_references" / "aviation_source_semantic_references.json"
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
                    "compiled_rule_constraint_templates": str(
                        OVERLAY_DIR / model_key / "compiled_rule_constraint_templates.json"
                    ),
                    "alignment_audit": str(OVERLAY_DIR / model_key / "alignment_audit.json"),
                }
                for model_key in LIBRARY_SPECS
            },
            "counts": {
                "tasks": len(references),
                "source_tasks": 19,
                "curated_extension_tasks": 11,
                "canonical_referenced_rule_ids": len(canonical_ids),
                "public_scenario_model_constraints": 82,
                "synthetic_stress_rules": 0,
                "forbidden_input_key_hits": 0,
            },
            "alignment_summaries": all_summaries,
            "strict_common_tasks": strict_common_payload,
            "excluded_from_main_dataset": {
                "aviation_stress_tasks": 12,
                "reason": (
                    "The stress tasks depend on synthetic stress-extension rules and are diagnostic, "
                    "not part of the full-KG-only main aviation benchmark."
                ),
            },
        },
    )
    print(json.dumps({"alignment_summaries": all_summaries, "strict_common_tasks": strict_common_payload}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
