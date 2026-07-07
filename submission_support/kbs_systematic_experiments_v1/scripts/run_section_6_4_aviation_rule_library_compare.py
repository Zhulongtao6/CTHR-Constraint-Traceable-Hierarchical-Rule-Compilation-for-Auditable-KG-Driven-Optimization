from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import run_section_6_2_table1_fullkg_pipeline as fullkg


ROOT = Path(__file__).resolve().parents[1]
CTHR_ROOT = ROOT.parents[1]
RESULTS_DIR = ROOT / "results"

AVIATION_ROOT = ROOT / "datasets" / "aviation_fullkg_clean"
AVIATION_CORE = AVIATION_ROOT / "core"
AVIATION_OVERLAYS = AVIATION_ROOT / "evaluation_overlays"
AVIATION_RULE_LIBRARIES = AVIATION_ROOT / "rule_libraries"
CANONICAL_EVALUATION_REFERENCES = (
    AVIATION_ROOT / "evaluation_references" / "aviation_evaluation_references.json"
)
CANONICAL_CONSTRAINT_TEMPLATES = (
    AVIATION_ROOT / "constraint_templates" / "compiled_rule_constraint_templates.json"
)
MODEL_COMPARE_ROOT = CTHR_ROOT / "paper" / "full_aviation_kg_rule_library_model_comparison"
MIMO_ROOT = CTHR_ROOT / "paper" / "mimo_aviation_kg_rule_library"


MODEL_SPECS = [
    {
        "model": "Qwen",
        "provider": "qwen",
        "overlay_key": "qwen",
        "rule_library": AVIATION_RULE_LIBRARIES / "qwen" / "full_aviation_rule_library_qwen.json",
        "library_summary": MODEL_COMPARE_ROOT / "summary_qwen.json",
        "grounding_full": RESULTS_DIR / "section_6_3_aviation_fullkg_qwen_20260527_candidate_to_valid_full.json",
    },
    {
        "model": "DeepSeek strict repaired",
        "provider": "deepseek",
        "overlay_key": "deepseek",
        "rule_library": AVIATION_RULE_LIBRARIES
        / "deepseek"
        / "full_aviation_rule_library_deepseek_strict_repaired.json",
        "library_summary": MODEL_COMPARE_ROOT / "summary_deepseek_strict_repaired.json",
        "grounding_full": RESULTS_DIR
        / "section_6_3_aviation_fullkg_deepseek_strict_20260527_candidate_to_valid_full.json",
    },
    {
        "model": "Xiaomi MIMO",
        "provider": "xiaomi_mimo",
        "overlay_key": "xiaomi_mimo",
        "rule_library": AVIATION_RULE_LIBRARIES / "xiaomi_mimo" / "full_aviation_rule_library_mimo.json",
        "library_summary": MIMO_ROOT / "summary_mimo.json",
        "grounding_full": RESULTS_DIR / "section_6_3_aviation_fullkg_mimo_20260528_candidate_to_valid_full.json",
    },
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(list(value) if isinstance(value, set) else value, ensure_ascii=False)
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({header: csv_cell(row.get(header)) for header in headers})


def markdown_table(rows: list[dict[str, Any]], headers: list[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(csv_cell(row.get(header)) for header in headers) + " |")
    return "\n".join(lines)


def aviation_spec(
    rule_library: Path,
    grounding_full: Path,
    evaluation_references: Path,
    constraint_templates: Path,
) -> fullkg.DatasetSpec:
    return fullkg.DatasetSpec(
        name="Aviation",
        domain="aviation",
        root=AVIATION_ROOT,
        algorithm_inputs=AVIATION_CORE / "algorithm_inputs" / "aviation_algorithm_inputs.json",
        scenario_models=AVIATION_CORE / "scenario_models" / "aviation_public_scenario_models.json",
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
    aggregate_row = fullkg.aggregate(rows, "Aviation", "CTHR default", "cthr_semantic_modeling")
    return aggregate_row, dataset_summary, rows


def overlay_file(model_key: str, filename: str) -> Path:
    return AVIATION_OVERLAYS / model_key / filename


def overlay_alignment_summary(model_key: str) -> dict[str, Any]:
    path = overlay_file(model_key, "rule_id_alignment.json")
    payload = read_json(path)
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


def support_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    unsupported = [row for row in rows if row.get("unsupported_reason")]
    return {
        "candidate_zero": sum(1 for row in rows if row.get("unsupported_reason") == "no_grounded_candidates"),
        "unsupported": len(unsupported),
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


def as_ratio(count: int, total: int) -> str:
    return f"{count}/{total}"


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def rule_precision(predicted: list[str], reference: list[str]) -> float:
    if not predicted:
        return 0.0
    return len(set(predicted) & set(reference)) / len(set(predicted))


def rule_recall(predicted: list[str], reference: list[str]) -> float:
    if not reference:
        return 1.0
    return len(set(predicted) & set(reference)) / len(set(reference))


def project_rule_ids(predicted: list[str], model_to_canonical: dict[str, list[str]]) -> list[str]:
    return sorted({canonical for rule_id in predicted for canonical in model_to_canonical.get(str(rule_id), [])})


def annotate_task_rows(rows: list[dict[str, Any]], model_name: str, mode: str) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        item["Model"] = model_name
        item["evaluation_mode"] = mode
        out.append(item)
    return out


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
        predicted_canonical = project_rule_ids(predicted_model, model_to_canonical)
        reference_canonical = canonical_references.get(task_id, [])
        item = dict(row)
        item["Model"] = model_name
        item["Overlay"] = overlay_key
        item["evaluation_mode"] = "canonical_projected_semantic_main"
        item["predicted_model_rule_ids"] = predicted_model
        item["projected_canonical_rule_ids"] = predicted_canonical
        item["canonical_reference_rule_ids"] = reference_canonical
        item["canonical_rule_precision"] = rule_precision(predicted_canonical, reference_canonical)
        item["canonical_rule_recall"] = rule_recall(predicted_canonical, reference_canonical)
        out.append(item)
    return out


def aggregate_projected(rows: list[dict[str, Any]], model_name: str) -> dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {
            "Model": model_name,
            "Canonical Rule Precision": "N/A",
            "Canonical Rule Recall": "N/A",
            "Formal CSR": "N/A",
            "Sem-CSR": "N/A",
            "False accept": "N/A",
            "Invalid cases": "0/0 (N/A)",
        }
    unsupported = sum(1 for row in rows if row.get("unsupported_reason"))
    invalid = sum(1 for row in rows if row.get("invalid_case"))
    suffix = f" ({unsupported} unsupported)" if unsupported else ""
    return {
        "Model": model_name,
        "Canonical Rule Precision": pct(sum(float(row["canonical_rule_precision"]) for row in rows) / total),
        "Canonical Rule Recall": pct(sum(float(row["canonical_rule_recall"]) for row in rows) / total),
        "Formal CSR": pct(sum(1 for row in rows if row.get("formal_feasible")) / total),
        "Sem-CSR": pct(sum(1 for row in rows if row.get("semantic_valid")) / total),
        "False accept": pct(sum(1 for row in rows if row.get("false_accept")) / total),
        "Invalid cases": f"{invalid}/{total} ({100.0 * invalid / total:.1f}%){suffix}",
    }


def build_report(
    quality_rows: list[dict[str, Any]],
    raw_id_rows: list[dict[str, Any]],
    strict_model_id_rows: list[dict[str, Any]],
    canonical_projected_rows: list[dict[str, Any]],
    run_summary: dict[str, Any],
) -> str:
    table_1_headers = ["Model", "Provider", "Rules", "Provenance valid", "Constraint grounding", "Relation grounding"]
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
    lines = [
        "# Section 6.4 Aviation Rule Library Comparison",
        "",
        "The benchmark uses one fixed aviation core and model-specific evaluation overlays. GLM is excluded.",
        "The raw-ID table diagnoses namespace mismatch. The strict model-ID table is a diagnostic at model rule granularity. The canonical-projected semantic table is the main result.",
        "",
        "## Table 1: Rule Library Quality",
        "",
        markdown_table(quality_rows, table_1_headers),
        "",
        "## Table 2: Raw-ID Diagnostic",
        "",
        markdown_table(raw_id_rows, raw_headers),
        "",
        "## Table 3: Strict Model-ID Aligned Diagnostic",
        "",
        markdown_table(strict_model_id_rows, strict_headers),
        "",
        "## Table 4: Canonical-Projected Semantic Main Result",
        "",
        markdown_table(canonical_projected_rows, canonical_headers),
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
    quality_rows: list[dict[str, Any]] = []
    raw_id_rows: list[dict[str, Any]] = []
    strict_model_id_rows: list[dict[str, Any]] = []
    canonical_projected_rows: list[dict[str, Any]] = []
    raw_task_rows: list[dict[str, Any]] = []
    strict_task_rows: list[dict[str, Any]] = []
    canonical_projected_task_rows_all: list[dict[str, Any]] = []
    per_model_summary: dict[str, Any] = {}
    canonical_references = canonical_reference_by_task()

    for model_spec in MODEL_SPECS:
        model_name = model_spec["model"]
        provider = model_spec["provider"]
        overlay_key = model_spec["overlay_key"]
        rule_library = Path(model_spec["rule_library"])
        library_summary_path = Path(model_spec["library_summary"])
        grounding_full = Path(model_spec["grounding_full"])

        library_summary = read_json(library_summary_path)
        quality_rows.append(
            {
                "Model": model_name,
                "Provider": provider,
                "Rules": int(library_summary.get("num_rules", 0)),
                "Provenance valid": f"{float(library_summary.get('mean_provenance_validity_rate', 0.0)):.1f}%",
                "Constraint grounding": f"{float(library_summary.get('mean_constraint_grounding_rate', 0.0)):.1f}%",
                "Relation grounding": f"{float(library_summary.get('mean_relation_grounding_rate', 0.0)):.1f}%",
            }
        )

        raw_spec = aviation_spec(
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

        strict_spec = aviation_spec(
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
            "Strong canonical coverage": (
                f"{alignment_summary.get('exact_or_strong_aligned_canonical_rule_count', alignment_summary.get('aligned_canonical_rule_count', 0))}/"
                f"{alignment_summary.get('canonical_rule_count', 0)}"
            ),
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

        per_model_summary[model_name] = {
            "provider": provider,
            "overlay_key": overlay_key,
            "rule_library": str(rule_library),
            "rule_library_summary": str(library_summary_path),
            "grounding_full": str(grounding_full),
            "alignment_summary": alignment_summary,
            "raw_id_diagnostic": {
                "evaluation_references": str(CANONICAL_EVALUATION_REFERENCES),
                "constraint_templates": str(CANONICAL_CONSTRAINT_TEMPLATES),
                "dataset_summary": raw_dataset_summary,
                "cthr_default_aggregate": raw_aggregate,
                "task_rows": len(raw_per_task_rows),
            },
            "strict_model_id_aligned_diagnostic": {
                "evaluation_references": str(overlay_file(overlay_key, "evaluation_references.json")),
                "constraint_templates": str(overlay_file(overlay_key, "compiled_rule_constraint_templates.json")),
                "rule_id_alignment": str(overlay_file(overlay_key, "rule_id_alignment.json")),
                "alignment_audit": str(overlay_file(overlay_key, "alignment_audit.json")),
                "dataset_summary": strict_dataset_summary,
                "cthr_default_aggregate": strict_aggregate,
                "task_rows": len(strict_per_task_rows),
                "candidate_zero": strict_counts["candidate_zero"],
                "unsupported_tasks": strict_counts["unsupported"],
            },
            "canonical_projected_semantic_main": {
                "cthr_default_aggregate": projected_aggregate,
                "task_rows": len(projected_task_rows),
                "candidate_zero": projected_counts["candidate_zero"],
                "unsupported_tasks": projected_counts["unsupported"],
            },
        }

    outputs = {
        "rule_library_quality_csv": RESULTS_DIR / "section_6_4_aviation_rule_library_quality.csv",
        "raw_id_table_csv": RESULTS_DIR / "section_6_4_aviation_raw_id_table.csv",
        "strict_model_id_aligned_table_csv": RESULTS_DIR / "section_6_4_aviation_strict_model_id_aligned_table.csv",
        "canonical_projected_semantic_table_csv": RESULTS_DIR
        / "section_6_4_aviation_canonical_projected_semantic_table.csv",
        "aligned_semantic_table_csv": RESULTS_DIR / "section_6_4_aviation_aligned_semantic_table.csv",
        "legacy_cthr_default_comparison_csv": RESULTS_DIR / "section_6_4_aviation_cthr_default_comparison.csv",
        "raw_id_task_rows_csv": RESULTS_DIR / "section_6_4_aviation_raw_id_task_rows.csv",
        "strict_model_id_task_rows_csv": RESULTS_DIR / "section_6_4_aviation_strict_model_id_task_rows.csv",
        "canonical_projected_task_rows_csv": RESULTS_DIR / "section_6_4_aviation_canonical_projected_task_rows.csv",
        "aligned_semantic_task_rows_csv": RESULTS_DIR / "section_6_4_aviation_aligned_semantic_task_rows.csv",
        "report_md": RESULTS_DIR / "section_6_4_aviation_rule_library_comparison_report.md",
        "summary_json": RESULTS_DIR / "section_6_4_aviation_rule_library_comparison_summary.json",
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
    write_csv(outputs["rule_library_quality_csv"], quality_rows, quality_headers)
    write_csv(outputs["raw_id_table_csv"], raw_id_rows, raw_headers)
    write_csv(outputs["strict_model_id_aligned_table_csv"], strict_model_id_rows, strict_headers)
    write_csv(outputs["canonical_projected_semantic_table_csv"], canonical_projected_rows, canonical_headers)
    write_csv(outputs["aligned_semantic_table_csv"], canonical_projected_rows, canonical_headers)
    write_csv(
        outputs["legacy_cthr_default_comparison_csv"],
        [
            {
                "Model": row["Model"],
                "Rule Precision": row["Canonical Rule Precision"],
                "Rule Recall": row["Canonical Rule Recall"],
                "Formal CSR": row["Formal CSR"],
                "Sem-CSR": row["Sem-CSR"],
                "False accept": row["False accept"],
                "Invalid cases": row["Invalid cases"],
            }
            for row in canonical_projected_rows
        ],
        ["Model", "Rule Precision", "Rule Recall", "Formal CSR", "Sem-CSR", "False accept", "Invalid cases"],
    )
    write_csv(outputs["raw_id_task_rows_csv"], raw_task_rows, task_headers)
    write_csv(outputs["strict_model_id_task_rows_csv"], strict_task_rows, task_headers)
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
    write_csv(outputs["canonical_projected_task_rows_csv"], canonical_projected_task_rows_all, projected_task_headers)
    write_csv(outputs["aligned_semantic_task_rows_csv"], canonical_projected_task_rows_all, projected_task_headers)

    run_summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "started_at": started_at,
        "dataset": str(AVIATION_ROOT),
        "core": {
            "algorithm_inputs": str(AVIATION_CORE / "algorithm_inputs" / "aviation_algorithm_inputs.json"),
            "scenario_models": str(AVIATION_CORE / "scenario_models" / "aviation_public_scenario_models.json"),
            "source_semantic_references": str(
                AVIATION_CORE / "source_semantic_references" / "aviation_source_semantic_references.json"
            ),
        },
        "raw_id_table_role": "diagnostic only: canonical/Qwen rule IDs are used without model namespace alignment",
        "strict_model_id_aligned_table_role": "diagnostic only: exact model rule IDs after strong-only overlay alignment",
        "canonical_projected_semantic_table_role": "main result: predicted model rule IDs are projected back to canonical source-rule IDs before P/R",
        "models": per_model_summary,
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    write_json(outputs["summary_json"], run_summary)
    outputs["report_md"].write_text(
        build_report(quality_rows, raw_id_rows, strict_model_id_rows, canonical_projected_rows, run_summary),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "rule_library_quality": quality_rows,
                "raw_id_table": raw_id_rows,
                "strict_model_id_aligned_table": strict_model_id_rows,
                "canonical_projected_semantic_table": canonical_projected_rows,
                "outputs": {name: str(path) for name, path in outputs.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
