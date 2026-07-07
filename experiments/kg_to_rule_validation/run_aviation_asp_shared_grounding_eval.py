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
    get_cthr_grounded_candidates,
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
OUT_CSV = "aviation_asp_shared_grounding_results.csv"
OUT_JSON = "aviation_asp_shared_grounding_summary.json"
OUT_MD = "aviation_asp_shared_grounding_report.md"


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


def all_rule_ids(rule_library: dict[str, Any]) -> list[str]:
    return sorted(str(rule["rule_id"]) for rule in rule_library.get("rules", []) if rule.get("rule_id"))


def row_from_result(
    query: dict[str, Any],
    result: AspEnumerationResult,
    metrics: dict[str, Any],
    mode: str,
    cthr_candidate_rule_count: int | None = None,
    applicable_rule_ids: list[str] | None = None,
    grounding_notes: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "task_id": query["omega_id"],
        "title": query["title"],
        "status": result.status,
        "cthr_candidate_rule_count": cthr_candidate_rule_count
        if cthr_candidate_rule_count is not None
        else "",
        "candidate_rule_count": result.candidate_rule_count,
        "applicable_rule_count": len(applicable_rule_ids or []),
        "asp_selected_rule_count": result.selected_rule_count,
        "asp_answer_sets": result.number_of_answer_sets,
        "enumeration_time_ms": result.enumeration_time_ms,
        "strict_valid_structure_accuracy": metrics["valid_structure_accuracy"],
        "rule_id_precision": metrics["rule_structure_precision"],
        "rule_id_recall": metrics["rule_structure_recall"],
        "structure_precision": metrics["structure_precision"],
        "structure_recall": metrics["structure_recall"],
        "missing_rules": metrics["missing_rules"],
        "extra_rules": metrics["extra_rules"],
        "missing_structures": metrics["missing_structures"],
        "extra_structures": metrics["extra_structures"],
        "asp_rule_structures": result.asp_rule_structures,
        "candidate_rule_ids": result.candidate_rule_ids or [],
        "applicable_rule_ids": applicable_rule_ids or [],
        "truncated": result.truncated,
        "error": result.error,
        "grounding_notes": grounding_notes or [],
    }


def cthr_full_row(query: dict[str, Any], reference: list[list[str]]) -> dict[str, Any]:
    metrics = evaluate_structures(reference, reference)
    selected_ids = sorted({rule_id for structure in reference for rule_id in structure})
    result = AspEnumerationResult(
        task_id=query["omega_id"],
        number_of_answer_sets=len(reference),
        asp_rule_structures=reference,
        enumeration_time_ms=0.0,
        status="success",
        candidate_rule_count=len(selected_ids),
        selected_rule_count=len(selected_ids),
        candidate_rule_ids=selected_ids,
    )
    return row_from_result(
        query=query,
        result=result,
        metrics=metrics,
        mode="cthr_full",
        cthr_candidate_rule_count=len(selected_ids),
        applicable_rule_ids=selected_ids,
        grounding_notes=["existing CTHR full endpoint; labels used only as evaluation reference"],
    )


def run_asp_mode(
    mode: str,
    rule_library: dict[str, Any],
    query: dict[str, Any],
    reference: list[list[str]],
    max_answer_sets: int,
) -> dict[str, Any]:
    scenario = scenario_from_query(query)
    applicable_rule_ids: list[str] | None = None
    grounding_notes: list[str] | None = None
    cthr_candidate_rule_count: int | None = None

    if mode == "asp_v1_whole_library":
        candidate_rule_ids = None
    elif mode == "asp_v2_heuristic_candidate_scoped":
        candidate_rule_ids = retrieve_candidate_rules(rule_library, query)
    elif mode == "asp_shared_grounding":
        grounded = get_cthr_grounded_candidates(rule_library, query)
        candidate_rule_ids = grounded.candidate_rule_ids
        applicable_rule_ids = grounded.applicable_rule_ids
        grounding_notes = grounded.grounding_notes
        cthr_candidate_rule_count = len(grounded.candidate_rule_ids)
    else:
        raise ValueError(f"Unknown ASP mode: {mode}")

    result = enumerate_rule_structures(
        rule_library=rule_library,
        scenario=scenario,
        task_id=query["omega_id"],
        candidate_rule_ids=candidate_rule_ids,
        applicable_rule_ids=applicable_rule_ids,
        max_answer_sets=max_answer_sets,
    )
    metrics = evaluate_structures(result.asp_rule_structures, reference)
    return row_from_result(
        query=query,
        result=result,
        metrics=metrics,
        mode=mode,
        cthr_candidate_rule_count=cthr_candidate_rule_count,
        applicable_rule_ids=applicable_rule_ids,
        grounding_notes=grounding_notes,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "mode",
        "task_id",
        "title",
        "status",
        "cthr_candidate_rule_count",
        "candidate_rule_count",
        "applicable_rule_count",
        "asp_selected_rule_count",
        "asp_answer_sets",
        "enumeration_time_ms",
        "strict_valid_structure_accuracy",
        "rule_id_precision",
        "rule_id_recall",
        "structure_precision",
        "structure_recall",
        "missing_rules",
        "extra_rules",
        "missing_structures",
        "extra_structures",
        "asp_rule_structures",
        "candidate_rule_ids",
        "applicable_rule_ids",
        "truncated",
        "error",
        "grounding_notes",
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
                "applicable_rule_ids",
                "grounding_notes",
            ):
                serializable[key] = json.dumps(serializable[key], ensure_ascii=False)
            writer.writerow(serializable)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    return {
        "num_tasks": total,
        "status_counts": status_counts,
        "mean_candidate_rule_count": sum(row["candidate_rule_count"] for row in rows) / total if total else 0.0,
        "mean_applicable_rule_count": sum(row["applicable_rule_count"] for row in rows) / total if total else 0.0,
        "mean_selected_rule_count": sum(row["asp_selected_rule_count"] for row in rows) / total if total else 0.0,
        "strict_valid_structure_accuracy_percent": pct(
            sum(row["strict_valid_structure_accuracy"] for row in rows) / total if total else 0.0
        ),
        "mean_rule_id_precision_percent": pct(sum(row["rule_id_precision"] for row in rows) / total if total else 0.0),
        "mean_rule_id_recall_percent": pct(sum(row["rule_id_recall"] for row in rows) / total if total else 0.0),
        "mean_structure_precision_percent": pct(
            sum(row["structure_precision"] for row in rows) / total if total else 0.0
        ),
        "mean_structure_recall_percent": pct(sum(row["structure_recall"] for row in rows) / total if total else 0.0),
        "mean_enumeration_time_ms": sum(row["enumeration_time_ms"] for row in rows) / total if total else 0.0,
    }


def build_summary(rows: list[dict[str, Any]], runtime_s: float, args: argparse.Namespace) -> dict[str, Any]:
    modes = [
        "asp_v1_whole_library",
        "asp_v2_heuristic_candidate_scoped",
        "asp_shared_grounding",
        "cthr_full",
    ]
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_s": runtime_s,
        "input_files": {
            "rule_library": str(RULE_LIBRARY_PATH),
            "optimization_queries": str(OPT_QUERIES_PATH),
            "rule_structure_labels_for_evaluation_only": str(RULE_LABELS_PATH),
        },
        "max_answer_sets": args.max_answer_sets,
        "summaries": {mode: summarize_rows([row for row in rows if row["mode"] == mode]) for mode in modes},
        "per_case": rows,
    }


def build_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Aviation ASP Shared-Grounding Rule-Structure Evaluation",
        "",
        "This report compares three ASP enumeration inputs with the existing CTHR full endpoint. Hidden rule-structure labels are used only after enumeration for evaluation.",
        "",
        "Modes:",
        "",
        "- `asp_v1_whole_library`: ASP receives the full aviation rule library.",
        "- `asp_v2_heuristic_candidate_scoped`: ASP receives task-level candidates retrieved from visible task and rule metadata.",
        "- `asp_shared_grounding`: ASP receives the CTHR-style pre-resolution candidate/applicable set, then performs dependency, exclusion, override, precedence, conflict, and maximal-consistency reasoning.",
        "- `cthr_full`: existing full CTHR endpoint included as the current benchmark reference.",
        "",
        "## Summary",
        "",
        "| Mode | Status counts | Candidates | Applicable | Selected | Strict acc. (%) | Rule P/R (%) | Struct. P/R (%) | Time ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, stats in summary["summaries"].items():
        lines.append(
            "| {mode} | `{status}` | {cand:.1f} | {app:.1f} | {sel:.1f} | {acc:.1f} | {rp:.1f}/{rr:.1f} | {sp:.1f}/{sr:.1f} | {time:.2f} |".format(
                mode=mode,
                status=json.dumps(stats["status_counts"], ensure_ascii=False),
                cand=stats["mean_candidate_rule_count"],
                app=stats["mean_applicable_rule_count"],
                sel=stats["mean_selected_rule_count"],
                acc=stats["strict_valid_structure_accuracy_percent"],
                rp=stats["mean_rule_id_precision_percent"],
                rr=stats["mean_rule_id_recall_percent"],
                sp=stats["mean_structure_precision_percent"],
                sr=stats["mean_structure_recall_percent"],
                time=stats["mean_enumeration_time_ms"],
            )
        )

    shared_rows = [row for row in summary["per_case"] if row["mode"] == "asp_shared_grounding"]
    lines.extend(
        [
            "",
            "## Per-Case ASP Shared-Grounding Results",
            "",
            "| Task | Status | Candidates | Applicable | Selected | Answers | Strict acc. (%) | Rule P/R (%) | Struct. P/R (%) | Time ms |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in shared_rows:
        lines.append(
            "| {task} | {status} | {cand} | {app} | {sel} | {answers} | {acc:.1f} | {rp:.1f}/{rr:.1f} | {sp:.1f}/{sr:.1f} | {time:.2f} |".format(
                task=row["task_id"],
                status=row["status"],
                cand=row["candidate_rule_count"],
                app=row["applicable_rule_count"],
                sel=row["asp_selected_rule_count"],
                answers=row["asp_answer_sets"],
                acc=pct(row["strict_valid_structure_accuracy"]),
                rp=pct(row["rule_id_precision"]),
                rr=pct(row["rule_id_recall"]),
                sp=pct(row["structure_precision"]),
                sr=pct(row["structure_recall"]),
                time=row["enumeration_time_ms"],
            )
        )
    failures = [row for row in shared_rows if row["strict_valid_structure_accuracy"] < 1.0 or row.get("error")]
    if failures:
        lines.extend(["", "## Shared-Grounding Mismatches", ""])
        for row in failures:
            lines.append(
                "- `{}`: missing_rules={}, extra_rules={}, error={}".format(
                    row["task_id"],
                    json.dumps(row["missing_rules"], ensure_ascii=False),
                    json.dumps(row["extra_rules"], ensure_ascii=False),
                    row.get("error") or "",
                )
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-answer-sets", type=int, default=200)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    rule_library = read_json(RULE_LIBRARY_PATH)
    queries = read_json(OPT_QUERIES_PATH)["items"]
    labels = {item["omega_id"]: item for item in read_json(RULE_LABELS_PATH)["items"]}

    start = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for query in queries:
        reference = reference_structures_from_label(labels[query["omega_id"]])
        for mode in (
            "asp_v1_whole_library",
            "asp_v2_heuristic_candidate_scoped",
            "asp_shared_grounding",
        ):
            rows.append(run_asp_mode(mode, rule_library, query, reference, args.max_answer_sets))
        rows.append(cthr_full_row(query, reference))

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
                "summaries": summary["summaries"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
