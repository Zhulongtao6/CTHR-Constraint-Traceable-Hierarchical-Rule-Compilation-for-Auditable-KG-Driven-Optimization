from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import run_section_6_2_table1_fullkg_pipeline as fullkg
import run_section_6_4_aviation_rule_library_compare as shared


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

ARCHITECTURE_ROOT = ROOT / "datasets" / "architecture_fullkg_clean"
ARCHITECTURE_CORE = ARCHITECTURE_ROOT / "core"
ARCHITECTURE_OVERLAYS = ARCHITECTURE_ROOT / "evaluation_overlays"
ARCHITECTURE_RULE_LIBRARIES = ARCHITECTURE_ROOT / "rule_libraries"
CANONICAL_EVALUATION_REFERENCES = (
    ARCHITECTURE_ROOT / "evaluation_references" / "architecture_evaluation_references.json"
)
CANONICAL_CONSTRAINT_TEMPLATES = (
    ARCHITECTURE_ROOT / "constraint_templates" / "compiled_rule_constraint_templates.json"
)
STRICT_COMMON_TASKS = ARCHITECTURE_ROOT / "STRICT_COMMON_TASKS.json"


MODEL_SPECS = [
    {
        "model": "Qwen",
        "provider": "qwen",
        "overlay_key": "qwen",
        "rule_library": ARCHITECTURE_RULE_LIBRARIES / "qwen" / "full_architecture_rule_library_qwen.json",
        "grounding_full": RESULTS_DIR / "section_6_3_architecture_fullkg_qwen_candidate_to_valid_full.json",
    },
    {
        "model": "DeepSeek",
        "provider": "deepseek",
        "overlay_key": "deepseek",
        "rule_library": ARCHITECTURE_RULE_LIBRARIES
        / "deepseek"
        / "full_architecture_rule_library_deepseek.json",
        "grounding_full": RESULTS_DIR / "section_6_3_architecture_fullkg_deepseek_candidate_to_valid_full.json",
    },
    {
        "model": "Xiaomi MIMO",
        "provider": "xiaomi_mimo",
        "overlay_key": "xiaomi_mimo",
        "rule_library": ARCHITECTURE_RULE_LIBRARIES / "xiaomi_mimo" / "full_architecture_rule_library_mimo.json",
        "grounding_full": RESULTS_DIR / "section_6_3_architecture_fullkg_mimo_candidate_to_valid_full.json",
    },
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def architecture_spec(
    rule_library: Path,
    grounding_full: Path,
    evaluation_references: Path,
    constraint_templates: Path,
) -> fullkg.DatasetSpec:
    return fullkg.DatasetSpec(
        name="Architecture",
        domain="architecture",
        root=ARCHITECTURE_ROOT,
        algorithm_inputs=ARCHITECTURE_CORE / "algorithm_inputs" / "architecture_algorithm_inputs.json",
        scenario_models=ARCHITECTURE_CORE / "scenario_models" / "architecture_public_scenario_models.json",
        evaluation_references=evaluation_references,
        rule_library=rule_library,
        grounding_full=grounding_full,
        constraint_templates=constraint_templates,
    )


def run_cthr_default_only(spec: fullkg.DatasetSpec) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    original_methods = list(fullkg.METHOD_SPECS)
    fullkg.METHOD_SPECS = [("CTHR default", "cthr_semantic_modeling")]
    try:
        rows, dataset_summary = fullkg.evaluate_dataset(spec)
    finally:
        fullkg.METHOD_SPECS = original_methods
    aggregate_row = fullkg.aggregate(rows, "Architecture", "CTHR default", "cthr_semantic_modeling")
    return aggregate_row, dataset_summary, rows


def overlay_file(model_key: str, filename: str) -> Path:
    return ARCHITECTURE_OVERLAYS / model_key / filename


def overlay_alignment_summary(model_key: str) -> dict[str, Any]:
    payload = read_json(overlay_file(model_key, "rule_id_alignment.json"))
    return dict(payload.get("summary", {}))


def overlay_model_to_canonical(model_key: str) -> dict[str, list[str]]:
    payload = read_json(overlay_file(model_key, "rule_id_alignment.json"))
    out: dict[str, list[str]] = {}
    for row in payload.get("model_to_canonical", []):
        model_rule_id = str(row.get("model_rule_id"))
        out[model_rule_id] = sorted(str(item) for item in row.get("canonical_rule_ids", []))
    return out


def canonical_reference_by_task() -> dict[str, list[str]]:
    payload = read_json(CANONICAL_EVALUATION_REFERENCES)
    out: dict[str, list[str]] = {}
    for item in payload.get("items", []):
        task_id = str(item.get("omega_id"))
        out[task_id] = fullkg.reference_rule_ids(item)
    return out


def unresolved_label(summary: dict[str, Any]) -> str:
    count = int(summary.get("unresolved_canonical_rule_count", 0))
    ids = summary.get("unresolved_canonical_rule_ids", [])
    if not ids:
        return "0"
    return f"{count}: " + ", ".join(str(item) for item in ids)


def library_quality_row(model_spec: dict[str, Any]) -> dict[str, Any]:
    payload = read_json(Path(model_spec["rule_library"]))
    summary = payload.get("summary", {})
    return {
        "Model": model_spec["model"],
        "Provider": model_spec["provider"],
        "Rules": len(payload.get("rules", [])),
        "Provenance valid": f"{float(summary.get('mean_provenance_validity_rate', 0.0)):.1f}%",
        "Constraint grounding": f"{float(summary.get('mean_constraint_grounding_rate', 0.0)):.1f}%",
        "Relation grounding": f"{float(summary.get('mean_relation_grounding_rate', 0.0)):.1f}%",
    }


def metric_row(model_name: str, aggregate_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "Model": model_name,
        "Rule Precision": aggregate_row["Rule Precision"],
        "Rule Recall": aggregate_row["Rule Recall"],
        "Formal CSR": aggregate_row["Formal CSR"],
        "Sem-CSR": aggregate_row["Sem-CSR"],
        "False accept": aggregate_row["False accept"],
        "Invalid cases": aggregate_row["Invalid cases"],
    }


def support_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    unsupported = [row for row in rows if row.get("unsupported_reason")]
    return {
        "candidate_zero": sum(1 for row in rows if row.get("unsupported_reason") == "no_grounded_candidates"),
        "unsupported": len(unsupported),
    }


def canonical_projected_task_rows(
    rows: list[dict[str, Any]],
    model_name: str,
    overlay_key: str,
    model_to_canonical: dict[str, list[str]],
    canonical_references: dict[str, list[str]],
) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        task_id = str(row["task_id"])
        predicted_model = [str(item) for item in row.get("predicted_rule_ids", [])]
        predicted_canonical = shared.project_rule_ids(predicted_model, model_to_canonical)
        reference_canonical = canonical_references.get(task_id, [])
        item = dict(row)
        item["Model"] = model_name
        item["Overlay"] = overlay_key
        item["evaluation_mode"] = "canonical_projected_semantic_main"
        item["predicted_model_rule_ids"] = predicted_model
        item["projected_canonical_rule_ids"] = predicted_canonical
        item["canonical_reference_rule_ids"] = reference_canonical
        item["canonical_rule_precision"] = shared.rule_precision(predicted_canonical, reference_canonical)
        item["canonical_rule_recall"] = shared.rule_recall(predicted_canonical, reference_canonical)
        out.append(item)
    return out


def aggregate_projected(rows: list[dict[str, Any]], model_name: str) -> dict[str, Any]:
    return shared.aggregate_projected(rows, model_name)


def annotate_task_rows(rows: list[dict[str, Any]], model_name: str, mode: str) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        item["Model"] = model_name
        item["evaluation_mode"] = mode
        out.append(item)
    return out


def filter_tasks(rows: list[dict[str, Any]], task_ids: list[str]) -> list[dict[str, Any]]:
    wanted = set(task_ids)
    return [row for row in rows if str(row.get("task_id")) in wanted]


def aggregate_model_id(rows: list[dict[str, Any]], model_name: str) -> dict[str, Any]:
    aggregate = fullkg.aggregate(rows, "Architecture", "CTHR default", "cthr_semantic_modeling")
    counts = support_counts(rows)
    return {
        **metric_row(model_name, aggregate),
        "Candidate zero": counts["candidate_zero"],
        "Unsupported tasks": counts["unsupported"],
    }


def strong_coverage_label(summary: dict[str, Any]) -> str:
    return (
        f"{summary.get('exact_or_strong_aligned_canonical_rule_count', summary.get('aligned_canonical_rule_count', 0))}/"
        f"{summary.get('canonical_rule_count', 0)}"
    )


def build_report(
    quality_rows: list[dict[str, Any]],
    raw_id_rows: list[dict[str, Any]],
    strict_model_id_rows: list[dict[str, Any]],
    canonical_projected_rows: list[dict[str, Any]],
    strict_common: dict[str, Any],
    strict_raw_rows: list[dict[str, Any]],
    strict_model_rows: list[dict[str, Any]],
    strict_projected_rows: list[dict[str, Any]],
    run_summary: dict[str, Any],
) -> str:
    quality_headers = ["Model", "Provider", "Rules", "Provenance valid", "Constraint grounding", "Relation grounding"]
    raw_headers = [
        "Model",
        "Reference namespace",
        "Rule Precision",
        "Rule Recall",
        "Formal CSR",
        "Sem-CSR",
        "False accept",
        "Candidate zero",
        "Unsupported tasks",
        "Invalid cases",
    ]
    strict_headers = [
        "Model",
        "Overlay",
        "Strong canonical coverage",
        "Weak candidate alignments",
        "Unresolved canonical rules",
        "Model-ID Rule Precision",
        "Model-ID Rule Recall",
        "Formal CSR",
        "Sem-CSR",
        "False accept",
        "Candidate zero",
        "Unsupported tasks",
        "Invalid cases",
    ]
    canonical_headers = [
        "Model",
        "Overlay",
        "Strong canonical coverage",
        "Weak candidate alignments",
        "Unresolved canonical rules",
        "Canonical Rule Precision",
        "Canonical Rule Recall",
        "Formal CSR",
        "Sem-CSR",
        "False accept",
        "Candidate zero",
        "Unsupported tasks",
        "Invalid cases",
    ]
    strict_subset_headers = [
        "Model",
        "Rule Precision",
        "Rule Recall",
        "Formal CSR",
        "Sem-CSR",
        "False accept",
        "Candidate zero",
        "Unsupported tasks",
        "Invalid cases",
    ]
    strict_projected_headers = [
        "Model",
        "Canonical Rule Precision",
        "Canonical Rule Recall",
        "Formal CSR",
        "Sem-CSR",
        "False accept",
        "Candidate zero",
        "Unsupported tasks",
        "Invalid cases",
    ]
    lines = [
        "# Section 6.4 Architecture Rule Library Comparison",
        "",
        "The benchmark uses one fixed architecture core and model-specific evaluation overlays. GLM is excluded.",
        "The raw-ID table diagnoses namespace mismatch. The strict model-ID table is a diagnostic at model rule granularity. The canonical-projected semantic table is the main result.",
        "",
        "## Rule Library Quality",
        "",
        shared.markdown_table(quality_rows, quality_headers),
        "",
        "## Raw-ID Diagnostic",
        "",
        shared.markdown_table(raw_id_rows, raw_headers),
        "",
        "## Strict Model-ID Aligned Diagnostic",
        "",
        shared.markdown_table(strict_model_id_rows, strict_headers),
        "",
        "## Canonical-Projected Semantic Main Result",
        "",
        shared.markdown_table(canonical_projected_rows, canonical_headers),
        "",
        "## Strict-Common Subset",
        "",
        f"Task count: {strict_common.get('task_count', 0)}",
        "",
        ", ".join(strict_common.get("task_ids", [])),
        "",
        "### Strict-Common Raw-ID Diagnostic",
        "",
        shared.markdown_table(strict_raw_rows, strict_subset_headers),
        "",
        "### Strict-Common Model-ID Diagnostic",
        "",
        shared.markdown_table(strict_model_rows, strict_subset_headers),
        "",
        "### Strict-Common Canonical-Projected Main Result",
        "",
        shared.markdown_table(strict_projected_rows, strict_projected_headers),
        "",
        "## Run Summary",
        "",
        "```json",
        json.dumps(run_summary, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    strict_common = read_json(STRICT_COMMON_TASKS)
    strict_task_ids = [str(item) for item in strict_common.get("task_ids", [])]
    canonical_references = canonical_reference_by_task()

    quality_rows: list[dict[str, Any]] = []
    raw_id_rows: list[dict[str, Any]] = []
    strict_model_id_rows: list[dict[str, Any]] = []
    canonical_projected_rows: list[dict[str, Any]] = []
    raw_task_rows: list[dict[str, Any]] = []
    strict_task_rows: list[dict[str, Any]] = []
    canonical_projected_task_rows_all: list[dict[str, Any]] = []
    strict_common_raw_rows: list[dict[str, Any]] = []
    strict_common_model_rows: list[dict[str, Any]] = []
    strict_common_projected_rows: list[dict[str, Any]] = []
    strict_common_raw_task_rows: list[dict[str, Any]] = []
    strict_common_model_task_rows: list[dict[str, Any]] = []
    strict_common_projected_task_rows: list[dict[str, Any]] = []
    per_model_summary: dict[str, Any] = {}

    for model_spec in MODEL_SPECS:
        model_name = model_spec["model"]
        provider = model_spec["provider"]
        overlay_key = model_spec["overlay_key"]
        rule_library = Path(model_spec["rule_library"])
        grounding_full = Path(model_spec["grounding_full"])
        quality_rows.append(library_quality_row(model_spec))

        raw_spec = architecture_spec(
            rule_library=rule_library,
            grounding_full=grounding_full,
            evaluation_references=CANONICAL_EVALUATION_REFERENCES,
            constraint_templates=CANONICAL_CONSTRAINT_TEMPLATES,
        )
        raw_aggregate, raw_dataset_summary, raw_per_task_rows = run_cthr_default_only(raw_spec)
        raw_row = metric_row(model_name, raw_aggregate)
        raw_row["Reference namespace"] = "canonical/qwen raw IDs"
        raw_counts = support_counts(raw_per_task_rows)
        raw_row["Candidate zero"] = raw_counts["candidate_zero"]
        raw_row["Unsupported tasks"] = raw_counts["unsupported"]
        raw_id_rows.append(raw_row)
        raw_task_rows.extend(annotate_task_rows(raw_per_task_rows, model_name, "raw_id_diagnostic"))

        strict_spec = architecture_spec(
            rule_library=rule_library,
            grounding_full=grounding_full,
            evaluation_references=overlay_file(overlay_key, "evaluation_references.json"),
            constraint_templates=overlay_file(overlay_key, "compiled_rule_constraint_templates.json"),
        )
        strict_aggregate, strict_dataset_summary, strict_per_task_rows = run_cthr_default_only(strict_spec)
        alignment_summary = overlay_alignment_summary(overlay_key)
        strict_counts = support_counts(strict_per_task_rows)
        strict_row = {
            **metric_row(model_name, strict_aggregate),
            "Overlay": overlay_key,
            "Strong canonical coverage": strong_coverage_label(alignment_summary),
            "Weak candidate alignments": alignment_summary.get("filtered_weak_alignment_count", 0),
            "Unresolved canonical rules": unresolved_label(alignment_summary),
            "Model-ID Rule Precision": strict_aggregate["Rule Precision"],
            "Model-ID Rule Recall": strict_aggregate["Rule Recall"],
            "Candidate zero": strict_counts["candidate_zero"],
            "Unsupported tasks": strict_counts["unsupported"],
        }
        strict_row.pop("Rule Precision", None)
        strict_row.pop("Rule Recall", None)
        strict_model_id_rows.append(strict_row)
        strict_task_rows.extend(annotate_task_rows(strict_per_task_rows, model_name, "strict_model_id_aligned_diagnostic"))

        projected_task_rows = canonical_projected_task_rows(
            strict_per_task_rows,
            model_name,
            overlay_key,
            overlay_model_to_canonical(overlay_key),
            canonical_references,
        )
        projected_aggregate = aggregate_projected(projected_task_rows, model_name)
        projected_counts = support_counts(projected_task_rows)
        projected_row = {
            **projected_aggregate,
            "Overlay": overlay_key,
            "Strong canonical coverage": strict_row["Strong canonical coverage"],
            "Weak candidate alignments": strict_row["Weak candidate alignments"],
            "Unresolved canonical rules": strict_row["Unresolved canonical rules"],
            "Candidate zero": projected_counts["candidate_zero"],
            "Unsupported tasks": projected_counts["unsupported"],
        }
        canonical_projected_rows.append(projected_row)
        canonical_projected_task_rows_all.extend(projected_task_rows)

        strict_raw_task_subset = filter_tasks(raw_per_task_rows, strict_task_ids)
        strict_model_task_subset = filter_tasks(strict_per_task_rows, strict_task_ids)
        strict_projected_task_subset = filter_tasks(projected_task_rows, strict_task_ids)
        strict_raw_row = aggregate_model_id(strict_raw_task_subset, model_name)
        strict_model_row = aggregate_model_id(strict_model_task_subset, model_name)
        strict_projected_row = aggregate_projected(strict_projected_task_subset, model_name)
        strict_projected_counts = support_counts(strict_projected_task_subset)
        strict_projected_row["Candidate zero"] = strict_projected_counts["candidate_zero"]
        strict_projected_row["Unsupported tasks"] = strict_projected_counts["unsupported"]
        strict_common_raw_rows.append(strict_raw_row)
        strict_common_model_rows.append(strict_model_row)
        strict_common_projected_rows.append(strict_projected_row)
        strict_common_raw_task_rows.extend(annotate_task_rows(strict_raw_task_subset, model_name, "strict_common_raw_id"))
        strict_common_model_task_rows.extend(annotate_task_rows(strict_model_task_subset, model_name, "strict_common_model_id"))
        strict_common_projected_task_rows.extend(strict_projected_task_subset)

        per_model_summary[model_name] = {
            "provider": provider,
            "overlay_key": overlay_key,
            "rule_library": str(rule_library),
            "grounding_full": str(grounding_full),
            "alignment_summary": alignment_summary,
            "raw_id_diagnostic": {
                "evaluation_references": str(CANONICAL_EVALUATION_REFERENCES),
                "constraint_templates": str(CANONICAL_CONSTRAINT_TEMPLATES),
                "dataset_summary": raw_dataset_summary,
                "cthr_default_aggregate": raw_aggregate,
                "candidate_zero": raw_counts["candidate_zero"],
                "unsupported_tasks": raw_counts["unsupported"],
            },
            "strict_model_id_aligned_diagnostic": {
                "evaluation_references": str(overlay_file(overlay_key, "evaluation_references.json")),
                "constraint_templates": str(overlay_file(overlay_key, "compiled_rule_constraint_templates.json")),
                "rule_id_alignment": str(overlay_file(overlay_key, "rule_id_alignment.json")),
                "alignment_audit": str(overlay_file(overlay_key, "alignment_audit.json")),
                "dataset_summary": strict_dataset_summary,
                "cthr_default_aggregate": strict_aggregate,
                "candidate_zero": strict_counts["candidate_zero"],
                "unsupported_tasks": strict_counts["unsupported"],
            },
            "canonical_projected_semantic_main": {
                "cthr_default_aggregate": projected_aggregate,
                "candidate_zero": projected_counts["candidate_zero"],
                "unsupported_tasks": projected_counts["unsupported"],
            },
        }

    outputs = {
        "rule_library_quality_csv": RESULTS_DIR / "section_6_4_architecture_rule_library_quality.csv",
        "raw_id_table_csv": RESULTS_DIR / "section_6_4_architecture_raw_id_table.csv",
        "strict_model_id_aligned_table_csv": RESULTS_DIR / "section_6_4_architecture_strict_model_id_aligned_table.csv",
        "canonical_projected_semantic_table_csv": RESULTS_DIR
        / "section_6_4_architecture_canonical_projected_semantic_table.csv",
        "raw_id_task_rows_csv": RESULTS_DIR / "section_6_4_architecture_raw_id_task_rows.csv",
        "strict_model_id_task_rows_csv": RESULTS_DIR / "section_6_4_architecture_strict_model_id_task_rows.csv",
        "canonical_projected_task_rows_csv": RESULTS_DIR / "section_6_4_architecture_canonical_projected_task_rows.csv",
        "strict_common_task_ids_json": RESULTS_DIR / "section_6_4_architecture_strict_common_task_ids.json",
        "strict_common_task_ids_csv": RESULTS_DIR / "section_6_4_architecture_strict_common_task_ids.csv",
        "strict_common_raw_id_table_csv": RESULTS_DIR / "section_6_4_architecture_strict_common_raw_id_table.csv",
        "strict_common_model_id_table_csv": RESULTS_DIR / "section_6_4_architecture_strict_common_model_id_table.csv",
        "strict_common_canonical_projected_table_csv": RESULTS_DIR
        / "section_6_4_architecture_strict_common_canonical_projected_table.csv",
        "strict_common_raw_task_rows_csv": RESULTS_DIR / "section_6_4_architecture_strict_common_raw_id_task_rows.csv",
        "strict_common_model_task_rows_csv": RESULTS_DIR / "section_6_4_architecture_strict_common_model_id_task_rows.csv",
        "strict_common_projected_task_rows_csv": RESULTS_DIR
        / "section_6_4_architecture_strict_common_canonical_projected_task_rows.csv",
        "report_md": RESULTS_DIR / "section_6_4_architecture_rule_library_comparison_report.md",
        "summary_json": RESULTS_DIR / "section_6_4_architecture_rule_library_comparison_summary.json",
    }

    quality_headers = ["Model", "Provider", "Rules", "Provenance valid", "Constraint grounding", "Relation grounding"]
    raw_headers = [
        "Model",
        "Reference namespace",
        "Rule Precision",
        "Rule Recall",
        "Formal CSR",
        "Sem-CSR",
        "False accept",
        "Candidate zero",
        "Unsupported tasks",
        "Invalid cases",
    ]
    strict_headers = [
        "Model",
        "Overlay",
        "Strong canonical coverage",
        "Weak candidate alignments",
        "Unresolved canonical rules",
        "Model-ID Rule Precision",
        "Model-ID Rule Recall",
        "Formal CSR",
        "Sem-CSR",
        "False accept",
        "Candidate zero",
        "Unsupported tasks",
        "Invalid cases",
    ]
    canonical_headers = [
        "Model",
        "Overlay",
        "Strong canonical coverage",
        "Weak candidate alignments",
        "Unresolved canonical rules",
        "Canonical Rule Precision",
        "Canonical Rule Recall",
        "Formal CSR",
        "Sem-CSR",
        "False accept",
        "Candidate zero",
        "Unsupported tasks",
        "Invalid cases",
    ]
    task_headers = [
        "Model",
        "evaluation_mode",
        "Dataset",
        "task_id",
        "target_interaction",
        "Method",
        "Method type",
        "grounded_candidate_count",
        "predicted_rule_ids",
        "reference_rule_ids",
        "rule_precision",
        "rule_recall",
        "formal_feasible",
        "semantic_valid",
        "false_accept",
        "invalid_case",
        "unsupported_reason",
        "runtime_ms",
    ]
    projected_task_headers = [
        "Model",
        "Overlay",
        "evaluation_mode",
        "Dataset",
        "task_id",
        "target_interaction",
        "Method",
        "Method type",
        "grounded_candidate_count",
        "predicted_model_rule_ids",
        "projected_canonical_rule_ids",
        "canonical_reference_rule_ids",
        "canonical_rule_precision",
        "canonical_rule_recall",
        "formal_feasible",
        "semantic_valid",
        "false_accept",
        "invalid_case",
        "unsupported_reason",
        "runtime_ms",
    ]
    subset_headers = [
        "Model",
        "Rule Precision",
        "Rule Recall",
        "Formal CSR",
        "Sem-CSR",
        "False accept",
        "Candidate zero",
        "Unsupported tasks",
        "Invalid cases",
    ]
    subset_projected_headers = [
        "Model",
        "Canonical Rule Precision",
        "Canonical Rule Recall",
        "Formal CSR",
        "Sem-CSR",
        "False accept",
        "Candidate zero",
        "Unsupported tasks",
        "Invalid cases",
    ]

    shared.write_csv(outputs["rule_library_quality_csv"], quality_rows, quality_headers)
    shared.write_csv(outputs["raw_id_table_csv"], raw_id_rows, raw_headers)
    shared.write_csv(outputs["strict_model_id_aligned_table_csv"], strict_model_id_rows, strict_headers)
    shared.write_csv(outputs["canonical_projected_semantic_table_csv"], canonical_projected_rows, canonical_headers)
    shared.write_csv(outputs["raw_id_task_rows_csv"], raw_task_rows, task_headers)
    shared.write_csv(outputs["strict_model_id_task_rows_csv"], strict_task_rows, task_headers)
    shared.write_csv(outputs["canonical_projected_task_rows_csv"], canonical_projected_task_rows_all, projected_task_headers)
    shared.write_csv(outputs["strict_common_raw_id_table_csv"], strict_common_raw_rows, subset_headers)
    shared.write_csv(outputs["strict_common_model_id_table_csv"], strict_common_model_rows, subset_headers)
    shared.write_csv(outputs["strict_common_canonical_projected_table_csv"], strict_common_projected_rows, subset_projected_headers)
    shared.write_csv(outputs["strict_common_raw_task_rows_csv"], strict_common_raw_task_rows, task_headers)
    shared.write_csv(outputs["strict_common_model_task_rows_csv"], strict_common_model_task_rows, task_headers)
    shared.write_csv(outputs["strict_common_projected_task_rows_csv"], strict_common_projected_task_rows, projected_task_headers)
    strict_task_rows_for_csv = [{"task_id": task_id} for task_id in strict_task_ids]
    shared.write_csv(outputs["strict_common_task_ids_csv"], strict_task_rows_for_csv, ["task_id"])
    write_json(outputs["strict_common_task_ids_json"], strict_common)

    run_summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "started_at": started_at,
        "dataset": str(ARCHITECTURE_ROOT),
        "core": {
            "algorithm_inputs": str(ARCHITECTURE_CORE / "algorithm_inputs" / "architecture_algorithm_inputs.json"),
            "scenario_models": str(ARCHITECTURE_CORE / "scenario_models" / "architecture_public_scenario_models.json"),
            "source_semantic_references": str(
                ARCHITECTURE_CORE / "source_semantic_references" / "architecture_source_semantic_references.json"
            ),
        },
        "raw_id_table_role": "diagnostic only: canonical/Qwen rule IDs are used without model namespace alignment",
        "strict_model_id_aligned_table_role": "diagnostic only: exact model rule IDs after strong-only overlay alignment",
        "canonical_projected_semantic_table_role": (
            "main result: predicted model rule IDs are projected back to canonical source-rule IDs before P/R"
        ),
        "strict_common_tasks": strict_common,
        "models": per_model_summary,
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    write_json(outputs["summary_json"], run_summary)
    outputs["report_md"].write_text(
        build_report(
            quality_rows,
            raw_id_rows,
            strict_model_id_rows,
            canonical_projected_rows,
            strict_common,
            strict_common_raw_rows,
            strict_common_model_rows,
            strict_common_projected_rows,
            run_summary,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "raw_id_table": raw_id_rows,
                "strict_model_id_aligned_table": strict_model_id_rows,
                "canonical_projected_semantic_table": canonical_projected_rows,
                "strict_common_task_count": len(strict_task_ids),
                "strict_common_tasks": strict_task_ids,
                "outputs": {name: str(path) for name, path in outputs.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
