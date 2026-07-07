from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import run_section_6_4_aviation_rule_library_compare as base


RESULTS_DIR = base.RESULTS_DIR

STRICT_COMMON_TASK_IDS = [
    "AVI_OPT_03",
    "AVI_OPT_04",
    "AVI_OPT_05",
    "AVI_OPT_08",
    "AVI_OPT_09",
    "AVI_OPT_10",
    "AVI_OPT_11",
    "AVI_OPT_12",
    "AVI_OPT_14",
    "AVI_OPT_16",
    "AVI_OPT_20",
    "AVI_OPT_23",
    "AVI_OPT_24",
]


def filter_tasks(rows: list[dict[str, Any]], task_ids: list[str]) -> list[dict[str, Any]]:
    wanted = set(task_ids)
    return [row for row in rows if str(row.get("task_id")) in wanted]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def aggregate_model_id(rows: list[dict[str, Any]], model_name: str) -> dict[str, Any]:
    aggregate = base.fullkg.aggregate(rows, "Aviation", "CTHR default", "cthr_semantic_modeling")
    counts = base.support_counts(rows)
    return {
        **base.metric_row(model_name, aggregate),
        "Candidate zero": counts["candidate_zero"],
        "Unsupported tasks": counts["unsupported"],
    }


def build_report(
    quality_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    strict_rows: list[dict[str, Any]],
    projected_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    quality_headers = ["Model", "Provider", "Rules", "Provenance valid", "Constraint grounding", "Relation grounding"]
    raw_headers = [
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
    strict_headers = [
        "Model",
        "Model-ID Rule Precision",
        "Model-ID Rule Recall",
        "Formal CSR",
        "Sem-CSR",
        "False accept",
        "Candidate zero",
        "Unsupported tasks",
        "Invalid cases",
    ]
    projected_headers = [
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
    return "\n".join(
        [
            "# Section 6.4 Aviation Strict-Common Subset",
            "",
            "This subset keeps only tasks whose reference surviving rules have strong alignment in Qwen, DeepSeek, and Xiaomi MIMO.",
            "",
            f"Task count: {len(STRICT_COMMON_TASK_IDS)}",
            "",
            "## Task IDs",
            "",
            ", ".join(STRICT_COMMON_TASK_IDS),
            "",
            "## Rule Library Quality",
            "",
            base.markdown_table(quality_rows, quality_headers),
            "",
            "## Raw-ID Diagnostic",
            "",
            base.markdown_table(raw_rows, raw_headers),
            "",
            "## Strict Model-ID Aligned Diagnostic",
            "",
            base.markdown_table(strict_rows, strict_headers),
            "",
            "## Canonical-Projected Semantic Main Result",
            "",
            base.markdown_table(projected_rows, projected_headers),
            "",
            "## Run Summary",
            "",
            "```json",
            json.dumps(summary, ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )


def main() -> None:
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    quality_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    strict_rows: list[dict[str, Any]] = []
    projected_rows: list[dict[str, Any]] = []
    raw_task_rows: list[dict[str, Any]] = []
    strict_task_rows: list[dict[str, Any]] = []
    projected_task_rows_all: list[dict[str, Any]] = []
    canonical_references = base.canonical_reference_by_task()
    per_model_summary: dict[str, Any] = {}

    for model_spec in base.MODEL_SPECS:
        model_name = model_spec["model"]
        provider = model_spec["provider"]
        overlay_key = model_spec["overlay_key"]
        rule_library = Path(model_spec["rule_library"])
        grounding_full = Path(model_spec["grounding_full"])
        library_summary_path = Path(model_spec["library_summary"])
        library_summary = base.read_json(library_summary_path)
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

        raw_spec = base.aviation_spec(
            rule_library=rule_library,
            grounding_full=grounding_full,
            evaluation_references=base.CANONICAL_EVALUATION_REFERENCES,
            constraint_templates=base.CANONICAL_CONSTRAINT_TEMPLATES,
        )
        raw_aggregate_all, _, raw_per_task_all = base.run_cthr_default_only(raw_spec)
        _ = raw_aggregate_all
        raw_per_task = filter_tasks(raw_per_task_all, STRICT_COMMON_TASK_IDS)
        raw_row = aggregate_model_id(raw_per_task, model_name)
        raw_rows.append(raw_row)
        raw_task_rows.extend(base.annotate_task_rows(raw_per_task, model_name, "strict_common_raw_id_diagnostic"))

        strict_spec = base.aviation_spec(
            rule_library=rule_library,
            grounding_full=grounding_full,
            evaluation_references=base.overlay_file(overlay_key, "evaluation_references.json"),
            constraint_templates=base.overlay_file(overlay_key, "compiled_rule_constraint_templates.json"),
        )
        _, _, strict_per_task_all = base.run_cthr_default_only(strict_spec)
        strict_per_task = filter_tasks(strict_per_task_all, STRICT_COMMON_TASK_IDS)
        strict_aggregate = base.fullkg.aggregate(strict_per_task, "Aviation", "CTHR default", "cthr_semantic_modeling")
        strict_counts = base.support_counts(strict_per_task)
        strict_row = {
            "Model": model_name,
            "Model-ID Rule Precision": strict_aggregate["Rule Precision"],
            "Model-ID Rule Recall": strict_aggregate["Rule Recall"],
            "Formal CSR": strict_aggregate["Formal CSR"],
            "Sem-CSR": strict_aggregate["Sem-CSR"],
            "False accept": strict_aggregate["False accept"],
            "Candidate zero": strict_counts["candidate_zero"],
            "Unsupported tasks": strict_counts["unsupported"],
            "Invalid cases": strict_aggregate["Invalid cases"],
        }
        strict_rows.append(strict_row)
        strict_task_rows.extend(
            base.annotate_task_rows(strict_per_task, model_name, "strict_common_model_id_aligned_diagnostic")
        )

        projected_task_rows = base.canonical_projected_task_rows(
            strict_per_task,
            model_name,
            overlay_key,
            base.overlay_model_to_canonical(overlay_key),
            canonical_references,
        )
        projected_aggregate = base.aggregate_projected(projected_task_rows, model_name)
        projected_counts = base.support_counts(projected_task_rows)
        projected_row = {
            **projected_aggregate,
            "Candidate zero": projected_counts["candidate_zero"],
            "Unsupported tasks": projected_counts["unsupported"],
        }
        projected_rows.append(projected_row)
        projected_task_rows_all.extend(projected_task_rows)

        missing_tasks = sorted(set(STRICT_COMMON_TASK_IDS) - {str(row.get("task_id")) for row in strict_per_task})
        per_model_summary[model_name] = {
            "provider": provider,
            "overlay_key": overlay_key,
            "task_rows": len(strict_per_task),
            "missing_requested_task_ids": missing_tasks,
            "raw_id": raw_row,
            "strict_model_id": strict_row,
            "canonical_projected": projected_row,
        }

    outputs = {
        "quality_csv": RESULTS_DIR / "section_6_4_aviation_strict_common_rule_library_quality.csv",
        "raw_id_csv": RESULTS_DIR / "section_6_4_aviation_strict_common_raw_id_table.csv",
        "strict_model_id_csv": RESULTS_DIR / "section_6_4_aviation_strict_common_model_id_table.csv",
        "canonical_projected_csv": RESULTS_DIR
        / "section_6_4_aviation_strict_common_canonical_projected_table.csv",
        "raw_task_rows_csv": RESULTS_DIR / "section_6_4_aviation_strict_common_raw_id_task_rows.csv",
        "strict_task_rows_csv": RESULTS_DIR / "section_6_4_aviation_strict_common_model_id_task_rows.csv",
        "projected_task_rows_csv": RESULTS_DIR
        / "section_6_4_aviation_strict_common_canonical_projected_task_rows.csv",
        "report_md": RESULTS_DIR / "section_6_4_aviation_strict_common_report.md",
        "summary_json": RESULTS_DIR / "section_6_4_aviation_strict_common_summary.json",
    }

    quality_headers = ["Model", "Provider", "Rules", "Provenance valid", "Constraint grounding", "Relation grounding"]
    raw_headers = [
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
    strict_headers = [
        "Model",
        "Model-ID Rule Precision",
        "Model-ID Rule Recall",
        "Formal CSR",
        "Sem-CSR",
        "False accept",
        "Candidate zero",
        "Unsupported tasks",
        "Invalid cases",
    ]
    projected_headers = [
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
    base.write_csv(outputs["quality_csv"], quality_rows, quality_headers)
    base.write_csv(outputs["raw_id_csv"], raw_rows, raw_headers)
    base.write_csv(outputs["strict_model_id_csv"], strict_rows, strict_headers)
    base.write_csv(outputs["canonical_projected_csv"], projected_rows, projected_headers)

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
    base.write_csv(outputs["raw_task_rows_csv"], raw_task_rows, task_headers)
    base.write_csv(outputs["strict_task_rows_csv"], strict_task_rows, task_headers)
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
    base.write_csv(outputs["projected_task_rows_csv"], projected_task_rows_all, projected_task_headers)

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "started_at": started_at,
        "scope": "aviation strict common subset",
        "task_ids": STRICT_COMMON_TASK_IDS,
        "task_count": len(STRICT_COMMON_TASK_IDS),
        "models": per_model_summary,
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    write_json(outputs["summary_json"], summary)
    outputs["report_md"].write_text(
        build_report(quality_rows, raw_rows, strict_rows, projected_rows, summary),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "raw_id_table": raw_rows,
                "strict_model_id_table": strict_rows,
                "canonical_projected_table": projected_rows,
                "outputs": {key: str(value) for key, value in outputs.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
