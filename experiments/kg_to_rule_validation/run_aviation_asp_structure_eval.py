from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from baselines.asp_rule_structure import (
    AspEnumerationResult,
    enumerate_rule_structures,
    evaluate_structures,
    reference_structures_from_label,
    retrieve_candidate_rules,
)


THIS_DIR = Path(__file__).resolve().parent
CTHR_ROOT = THIS_DIR.parents[1]
PAPER_DIR = CTHR_ROOT / "paper"
LAYER_DIR = PAPER_DIR / "aviation_benchmark_layers"

RULE_LIBRARY_PATH = (
    PAPER_DIR
    / "full_aviation_kg_rule_library_model_comparison"
    / "full_aviation_rule_library_qwen.json"
)
OPT_QUERIES_PATH = LAYER_DIR / "aviation_optimization_queries.json"
RULE_LABELS_PATH = LAYER_DIR / "aviation_rule_structure_labels.json"

DEFAULT_RESULTS_DIR = PAPER_DIR / "results"
OUT_CSV = "aviation_asp_structure_results.csv"
OUT_JSON = "aviation_asp_structure_summary.json"
OUT_MD = "aviation_asp_structure_report.md"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float) -> float:
    return 100.0 * value


def scenario_from_query(query: dict[str, Any]) -> dict[str, Any]:
    scenario = dict(query.get("scenario_facts", {}))
    scenario.update(
        {
            "domain": query.get("domain"),
            "task_type": query.get("task_type"),
            "title": query.get("title"),
        }
    )
    return scenario


def row_from_result(
    query: dict[str, Any],
    result: AspEnumerationResult,
    metrics: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "task_id": query["omega_id"],
        "title": query["title"],
        "status": result.status,
        "candidate_rule_count": result.candidate_rule_count,
        "selected_rule_count": result.selected_rule_count,
        "number_of_answer_sets": result.number_of_answer_sets,
        "enumeration_time_ms": result.enumeration_time_ms,
        "valid_structure_accuracy": metrics["valid_structure_accuracy"],
        "rule_structure_precision": metrics["rule_structure_precision"],
        "rule_structure_recall": metrics["rule_structure_recall"],
        "missing_rules": metrics["missing_rules"],
        "extra_rules": metrics["extra_rules"],
        "missing_structures": metrics["missing_structures"],
        "extra_structures": metrics["extra_structures"],
        "asp_rule_structures": result.asp_rule_structures,
        "candidate_rule_ids": result.candidate_rule_ids or [],
        "truncated": result.truncated,
        "error": result.error,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "mode",
        "task_id",
        "title",
        "status",
        "candidate_rule_count",
        "selected_rule_count",
        "number_of_answer_sets",
        "enumeration_time_ms",
        "valid_structure_accuracy",
        "rule_structure_precision",
        "rule_structure_recall",
        "missing_rules",
        "extra_rules",
        "missing_structures",
        "extra_structures",
        "asp_rule_structures",
        "candidate_rule_ids",
        "truncated",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            serializable = dict(row)
            for key in (
                "missing_rules",
                "extra_rules",
                "missing_structures",
                "extra_structures",
                "asp_rule_structures",
                "candidate_rule_ids",
            ):
                serializable[key] = json.dumps(serializable[key], ensure_ascii=False)
            writer.writerow(serializable)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    total = len(rows)
    return {
        "num_tasks": total,
        "status_counts": status_counts,
        "mean_candidate_rule_count": (
            sum(row["candidate_rule_count"] for row in rows) / total if total else 0.0
        ),
        "mean_selected_rule_count": (
            sum(row["selected_rule_count"] for row in rows) / total if total else 0.0
        ),
        "valid_structure_accuracy_percent": pct(
            sum(row["valid_structure_accuracy"] for row in rows) / total if total else 0.0
        ),
        "mean_rule_structure_precision_percent": pct(
            sum(row["rule_structure_precision"] for row in rows) / total if total else 0.0
        ),
        "mean_rule_structure_recall_percent": pct(
            sum(row["rule_structure_recall"] for row in rows) / total if total else 0.0
        ),
        "mean_enumeration_time_ms": (
            sum(row["enumeration_time_ms"] for row in rows) / total if total else 0.0
        ),
    }


def build_summary(rows: list[dict[str, Any]], runtime_s: float, args: argparse.Namespace) -> dict[str, Any]:
    before_rows = [row for row in rows if row["mode"] == "all_library_before"]
    after_rows = [row for row in rows if row["mode"] == "candidate_scoped_after"]
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_s": runtime_s,
        "input_files": {
            "rule_library": str(RULE_LIBRARY_PATH),
            "optimization_queries": str(OPT_QUERIES_PATH),
            "rule_structure_labels_for_evaluation_only": str(RULE_LABELS_PATH),
        },
        "max_answer_sets": args.max_answer_sets,
        "before_all_library": summarize_rows(before_rows),
        "after_candidate_scoped": summarize_rows(after_rows),
        "per_case": rows,
    }


def build_report(summary: dict[str, Any]) -> str:
    before = summary["before_all_library"]
    after = summary["after_candidate_scoped"]
    lines = [
        "# Aviation ASP Rule-Structure Baseline",
        "",
        "This is a structure-only baseline. ASP receives the typed aviation rule library and each visible grounded scenario. Hidden expected labels are used only after enumeration for evaluation.",
        "",
        "The corrected baseline first retrieves task-level candidate rules from visible task/scenario information and rule metadata, then runs ASP only over that candidate set. The all-library mode is reported only as the pre-fix comparison.",
        "",
        "## Summary",
        "",
        "`Valid-structure accuracy` is strict exact-set accuracy over complete rule structures. `Precision` and `recall` are diagnostic rule-ID overlap scores between ASP-selected rules and the reference surviving rules.",
        "",
        "| Mode | Status counts | Mean candidates | Mean selected | Accuracy (%) | Precision (%) | Recall (%) | Time ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
        "| all-library before | `{}` | {:.1f} | {:.1f} | {:.1f} | {:.1f} | {:.1f} | {:.2f} |".format(
            json.dumps(before["status_counts"], ensure_ascii=False),
            before["mean_candidate_rule_count"],
            before["mean_selected_rule_count"],
            before["valid_structure_accuracy_percent"],
            before["mean_rule_structure_precision_percent"],
            before["mean_rule_structure_recall_percent"],
            before["mean_enumeration_time_ms"],
        ),
        "| candidate-scoped after | `{}` | {:.1f} | {:.1f} | {:.1f} | {:.1f} | {:.1f} | {:.2f} |".format(
            json.dumps(after["status_counts"], ensure_ascii=False),
            after["mean_candidate_rule_count"],
            after["mean_selected_rule_count"],
            after["valid_structure_accuracy_percent"],
            after["mean_rule_structure_precision_percent"],
            after["mean_rule_structure_recall_percent"],
            after["mean_enumeration_time_ms"],
        ),
        "",
        "## Per-Case Candidate-Scoped Results",
        "",
        "| Task | Status | Candidates | Selected | Answer sets | Accuracy (%) | Precision (%) | Recall (%) | Time ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    after_rows = [row for row in summary["per_case"] if row["mode"] == "candidate_scoped_after"]
    for row in after_rows:
        lines.append(
            "| {task} | {status} | {cand} | {sel} | {answers} | {acc:.1f} | {prec:.1f} | {rec:.1f} | {time:.2f} |".format(
                task=row["task_id"],
                status=row["status"],
                cand=row["candidate_rule_count"],
                sel=row["selected_rule_count"],
                answers=row["number_of_answer_sets"],
                acc=pct(row["valid_structure_accuracy"]),
                prec=pct(row["rule_structure_precision"]),
                rec=pct(row["rule_structure_recall"]),
                time=row["enumeration_time_ms"],
            )
        )
    errors = [row for row in after_rows if row.get("error")]
    if errors:
        lines.extend(["", "## Errors", ""])
        for row in errors[:5]:
            lines.append(f"- `{row['task_id']}`: {row['error']}")
        if len(errors) > 5:
            lines.append(f"- ... {len(errors) - 5} more error rows omitted.")
    return "\n".join(lines) + "\n"


def run_mode(
    mode: str,
    rule_library: dict[str, Any],
    queries: list[dict[str, Any]],
    labels: dict[str, dict[str, Any]],
    max_answer_sets: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query in queries:
        task_id = query["omega_id"]
        scenario = scenario_from_query(query)
        if mode == "candidate_scoped_after":
            candidate_rule_ids = retrieve_candidate_rules(rule_library, query)
        elif mode == "all_library_before":
            candidate_rule_ids = None
        else:
            raise ValueError(f"Unknown mode: {mode}")
        result = enumerate_rule_structures(
            rule_library=rule_library,
            scenario=scenario,
            task_id=task_id,
            candidate_rule_ids=candidate_rule_ids,
            max_answer_sets=max_answer_sets,
        )
        reference = reference_structures_from_label(labels[task_id])
        metrics = evaluate_structures(result.asp_rule_structures, reference)
        rows.append(row_from_result(query, result, metrics, mode=mode))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-answer-sets", type=int, default=200)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    rule_library = read_json(RULE_LIBRARY_PATH)
    queries = read_json(OPT_QUERIES_PATH)["items"]
    labels = {item["omega_id"]: item for item in read_json(RULE_LABELS_PATH)["items"]}

    start = time.perf_counter()
    rows = []
    rows.extend(run_mode("all_library_before", rule_library, queries, labels, args.max_answer_sets))
    rows.extend(run_mode("candidate_scoped_after", rule_library, queries, labels, args.max_answer_sets))

    summary = build_summary(rows, runtime_s=time.perf_counter() - start, args=args)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / OUT_CSV
    json_path = out_dir / OUT_JSON
    md_path = out_dir / OUT_MD
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(build_report(summary), encoding="utf-8")

    print(
        json.dumps(
            {
                "out_csv": str(csv_path),
                "out_json": str(json_path),
                "out_md": str(md_path),
                "before_all_library": summary["before_all_library"],
                "after_candidate_scoped": summary["after_candidate_scoped"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
