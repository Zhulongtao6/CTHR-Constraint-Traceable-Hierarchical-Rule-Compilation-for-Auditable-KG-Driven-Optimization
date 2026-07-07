from __future__ import annotations

import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CTHR_ROOT = ROOT.parents[1]
SCRIPTS_DIR = ROOT / "scripts"
RESULTS_DIR = ROOT / "results"

sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(CTHR_ROOT))

import run_section_6_2_table1_fullkg_pipeline as full  # noqa: E402
from run_section_6_2_table1_flat_old_candidate_mapping import (  # noqa: E402
    OLD_MAPPING_NAME,
    old_candidate_ids,
)


CURRENT_TABLE_PER_TASK = RESULTS_DIR / "section_6_2_table1_fullkg_pipeline_per_task.csv"


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def current_cthr_rows() -> dict[tuple[str, str], dict[str, Any]]:
    if not CURRENT_TABLE_PER_TASK.exists():
        return {}
    with CURRENT_TABLE_PER_TASK.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            (row["Dataset"], row["task_id"]): row
            for row in reader
            if row.get("Method") == "CTHR default"
        }


def evaluate_dataset(spec: full.DatasetSpec) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    algorithm_inputs = full.item_map(spec.algorithm_inputs)
    scenario_models = full.item_map(spec.scenario_models)
    references = full.item_map(spec.evaluation_references)
    grounding_results = full.grounding_result_map(spec.grounding_full)
    templates_by_rule = full.constraint_template_map(spec.constraint_templates)
    rule_library = full.read_json(spec.rule_library)
    rule_by_id = {str(rule["rule_id"]): rule for rule in rule_library.get("rules", []) if rule.get("rule_id")}
    current_rows = current_cthr_rows()

    if set(algorithm_inputs) != set(scenario_models) or set(algorithm_inputs) != set(references):
        raise ValueError(f"{spec.name} layer IDs do not match")
    if set(algorithm_inputs) != set(grounding_results):
        raise ValueError(f"{spec.name} grounding result IDs do not match")

    rows: list[dict[str, Any]] = []
    candidate_stats: list[dict[str, Any]] = []
    for task_id in sorted(algorithm_inputs):
        grounding_task = dict(algorithm_inputs[task_id])
        query = full.prepare_query(grounding_task, scenario_models[task_id])
        query["_compiled_rule_constraint_templates_by_id"] = templates_by_rule
        reference = references[task_id]
        reference_ids = full.reference_rule_ids(reference)
        feasible = full.reference_feasible(reference, query)
        grounding_row = grounding_results[task_id]
        current_candidate_ids = full.ids_from_grounding(grounding_row, "candidate_rule_ids_generated")
        current_valid_ids = full.ids_from_grounding(grounding_row, "predicted_valid_rule_ids")
        old_ids, scored = old_candidate_ids(spec, rule_library, query, grounding_task)
        old_rules = [rule_by_id[rule_id] for rule_id in old_ids if rule_id in rule_by_id]
        old_ids = sorted(str(rule["rule_id"]) for rule in old_rules)

        start = time.perf_counter()
        result = full.run_cthr_default(
            spec,
            query,
            grounding_task,
            old_rules,
            rule_by_id,
            selected_valid_ids=None,
        )
        runtime_ms = (time.perf_counter() - start) * 1000.0
        predicted = sorted(result.predicted_rule_ids) if result.supported else []
        precision = full.base.method_rule_precision(predicted, reference_ids)
        recall = full.base.method_rule_recall(predicted, reference_ids)
        precision = 0.0 if precision is None else precision
        recall = 0.0 if recall is None else recall
        sem_ok = full.semantic_valid(feasible, result.optimized_x, predicted, reference_ids) if result.supported else False
        formal_ok = bool(result.formal_feasible) if result.supported else False
        current = current_rows.get((spec.name, task_id), {})
        current_sem = parse_bool(current.get("semantic_valid")) if current else None
        current_precision = float(current["rule_precision"]) if current.get("rule_precision") else None
        current_recall = float(current["rule_recall"]) if current.get("rule_recall") else None

        reference_set = set(reference_ids)
        old_set = set(old_ids)
        predicted_set = set(predicted)
        current_candidate_set = set(current_candidate_ids)
        current_valid_set = set(current_valid_ids)
        candidate_stats.append(
            {
                "old_candidate_count": len(old_set),
                "current_candidate_count": len(current_candidate_set),
                "old_predicted_valid_count": len(predicted_set),
                "current_valid_count": len(current_valid_set),
                "old_candidate_extra_vs_ref": len(old_set - reference_set),
                "old_candidate_missing_ref": len(reference_set - old_set),
                "old_predicted_extra_vs_ref": len(predicted_set - reference_set),
                "old_predicted_missing_ref": len(reference_set - predicted_set),
                "resolver_removed_count": len(old_set - predicted_set),
            }
        )
        rows.append(
            {
                "Dataset": spec.name,
                "task_id": task_id,
                "target_interaction": full.target_interaction(reference),
                "Method": "CTHR default",
                "candidate_mapping": OLD_MAPPING_NAME,
                "old_candidate_count": len(old_set),
                "current_candidate_count": len(current_candidate_set),
                "old_predicted_valid_count": len(predicted_set),
                "current_valid_count": len(current_valid_set),
                "old_candidate_extra_vs_ref": len(old_set - reference_set),
                "old_candidate_missing_ref": len(reference_set - old_set),
                "old_predicted_extra_vs_ref": len(predicted_set - reference_set),
                "old_predicted_missing_ref": len(reference_set - predicted_set),
                "resolver_removed_count": len(old_set - predicted_set),
                "top_old_scores": [(round(score, 3), rule_id) for score, rule_id in scored[:5]],
                "predicted_rule_ids": predicted,
                "reference_rule_ids": reference_ids,
                "rule_precision": precision,
                "rule_recall": recall,
                "formal_feasible": formal_ok,
                "semantic_valid": sem_ok,
                "false_accept": bool(formal_ok and not sem_ok),
                "invalid_case": bool(not sem_ok),
                "current_cthr_rule_precision": current_precision,
                "current_cthr_rule_recall": current_recall,
                "current_cthr_semantic_valid": current_sem,
                "semantic_regressed_vs_current_cthr": bool(current_sem and not sem_ok) if current_sem is not None else "",
                "unsupported_reason": "" if result.supported else result.unsupported_reason,
                "runtime_ms": round(runtime_ms, 3),
            }
        )
    summary = {
        "dataset": spec.name,
        "tasks": len(rows),
        "rule_library": str(spec.rule_library),
        "grounding_result_compared_against": str(spec.grounding_full),
        "candidate_mapping": OLD_MAPPING_NAME,
        "old_mapping_limit": 24,
        "mean_old_candidate_count": statistics.mean(item["old_candidate_count"] for item in candidate_stats),
        "mean_current_candidate_count": statistics.mean(item["current_candidate_count"] for item in candidate_stats),
        "mean_old_predicted_valid_count": statistics.mean(item["old_predicted_valid_count"] for item in candidate_stats),
        "mean_current_valid_count": statistics.mean(item["current_valid_count"] for item in candidate_stats),
        "mean_old_candidate_extra_vs_ref": statistics.mean(item["old_candidate_extra_vs_ref"] for item in candidate_stats),
        "mean_old_candidate_missing_ref": statistics.mean(item["old_candidate_missing_ref"] for item in candidate_stats),
        "mean_old_predicted_extra_vs_ref": statistics.mean(item["old_predicted_extra_vs_ref"] for item in candidate_stats),
        "mean_old_predicted_missing_ref": statistics.mean(item["old_predicted_missing_ref"] for item in candidate_stats),
        "mean_resolver_removed_count": statistics.mean(item["resolver_removed_count"] for item in candidate_stats),
    }
    return rows, summary


def aggregate(rows: list[dict[str, Any]], dataset: str) -> dict[str, Any]:
    subset = [row for row in rows if row["Dataset"] == dataset]
    total = len(subset)
    invalid = sum(1 for row in subset if row["invalid_case"])
    unsupported = sum(1 for row in subset if row["unsupported_reason"])
    suffix = f" ({unsupported} unsupported)" if unsupported else ""
    return {
        "Dataset": dataset,
        "Method": "CTHR default",
        "candidate_mapping": OLD_MAPPING_NAME,
        "Rule Precision": full.pct(sum(float(row["rule_precision"]) for row in subset) / total),
        "Rule Recall": full.pct(sum(float(row["rule_recall"]) for row in subset) / total),
        "Formal CSR": full.pct(sum(1 for row in subset if row["formal_feasible"]) / total),
        "Sem-CSR": full.pct(sum(1 for row in subset if row["semantic_valid"]) / total),
        "False accept": full.pct(sum(1 for row in subset if row["false_accept"]) / total),
        "Invalid cases": f"{invalid}/{total} ({100.0 * invalid / total:.1f}%){suffix}",
        "Mean old candidates": f"{statistics.mean(int(row['old_candidate_count']) for row in subset):.2f}",
        "Mean old predicted valid": f"{statistics.mean(int(row['old_predicted_valid_count']) for row in subset):.2f}",
        "Mean current candidates": f"{statistics.mean(int(row['current_candidate_count']) for row in subset):.2f}",
        "Mean current valid": f"{statistics.mean(int(row['current_valid_count']) for row in subset):.2f}",
        "Mean resolver removed": f"{statistics.mean(int(row['resolver_removed_count']) for row in subset):.2f}",
        "Mean predicted extra rules": f"{statistics.mean(int(row['old_predicted_extra_vs_ref']) for row in subset):.2f}",
        "Mean predicted missing refs": f"{statistics.mean(int(row['old_predicted_missing_ref']) for row in subset):.2f}",
        "Semantic regressions vs current CTHR": f"{sum(1 for row in subset if row['semantic_regressed_vs_current_cthr'])}/{total}",
    }


def markdown_table(rows: list[dict[str, Any]], headers: list[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(full.csv_cell(row.get(header)) for header in headers) + " |")
    return "\n".join(lines)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    per_task_rows: list[dict[str, Any]] = []
    dataset_summaries: dict[str, Any] = {}
    for spec in full.DATASETS:
        rows, summary = evaluate_dataset(spec)
        per_task_rows.extend(rows)
        dataset_summaries[spec.name] = summary

    aggregate_rows = [aggregate(per_task_rows, spec.name) for spec in full.DATASETS]
    per_task_headers = [
        "Dataset",
        "task_id",
        "target_interaction",
        "Method",
        "candidate_mapping",
        "old_candidate_count",
        "current_candidate_count",
        "old_predicted_valid_count",
        "current_valid_count",
        "old_candidate_extra_vs_ref",
        "old_candidate_missing_ref",
        "old_predicted_extra_vs_ref",
        "old_predicted_missing_ref",
        "resolver_removed_count",
        "top_old_scores",
        "predicted_rule_ids",
        "reference_rule_ids",
        "rule_precision",
        "rule_recall",
        "formal_feasible",
        "semantic_valid",
        "false_accept",
        "invalid_case",
        "current_cthr_rule_precision",
        "current_cthr_rule_recall",
        "current_cthr_semantic_valid",
        "semantic_regressed_vs_current_cthr",
        "unsupported_reason",
        "runtime_ms",
    ]
    aggregate_headers = [
        "Dataset",
        "Method",
        "candidate_mapping",
        "Rule Precision",
        "Rule Recall",
        "Formal CSR",
        "Sem-CSR",
        "False accept",
        "Invalid cases",
        "Mean old candidates",
        "Mean old predicted valid",
        "Mean current candidates",
        "Mean current valid",
        "Mean resolver removed",
        "Mean predicted extra rules",
        "Mean predicted missing refs",
        "Semantic regressions vs current CTHR",
    ]
    outputs = {
        "per_task_csv": RESULTS_DIR / "section_6_2_table1_cthr_old_candidate_mapping_per_task.csv",
        "overall_csv": RESULTS_DIR / "section_6_2_table1_cthr_old_candidate_mapping_overall.csv",
        "overall_md": RESULTS_DIR / "section_6_2_table1_cthr_old_candidate_mapping_overall.md",
        "overall_json": RESULTS_DIR / "section_6_2_table1_cthr_old_candidate_mapping_overall.json",
        "report_md": RESULTS_DIR / "section_6_2_table1_cthr_old_candidate_mapping_report.md",
    }
    full.write_csv(outputs["per_task_csv"], per_task_rows, per_task_headers)
    full.write_csv(outputs["overall_csv"], aggregate_rows, aggregate_headers)
    outputs["overall_md"].write_text(markdown_table(aggregate_rows, aggregate_headers), encoding="utf-8")
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "Diagnostic run: replace CTHR default candidate generation with the older broad rule-library scorer, then rerun CTHR valid-rule recovery.",
        "candidate_mapping": OLD_MAPPING_NAME,
        "datasets": dataset_summaries,
        "outputs": {key: str(value) for key, value in outputs.items()},
        "aggregate_rows": aggregate_rows,
    }
    full.write_json(outputs["overall_json"], summary)
    outputs["report_md"].write_text(
        "\n".join(
            [
                "# Section 6.2 Table 1 Diagnostic: CTHR Default With Old Candidate Mapping",
                "",
                "## Scope",
                "",
                "- Only CTHR default is rerun.",
                "- Candidate rules are generated by the older broad scorer.",
                "- CTHR valid-rule recovery is rerun on those old candidate sets.",
                "- The solver/evaluator and compiled rule-constraint templates are unchanged from the latest fullkg Table 1 runner.",
                "",
                "## Main Result",
                "",
                markdown_table(aggregate_rows, aggregate_headers),
                "",
                "## Run Summary",
                "",
                "```json",
                json.dumps(summary, ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"outputs": summary["outputs"], "aggregate_rows": aggregate_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
