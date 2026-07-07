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
from baselines.cthr_rule_resolver import (
    CthrResolverResult,
    resolve_valid_structures_with_diagnostics,
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
OUT_CSV = "aviation_cthr_on_aspv2_candidates_results.csv"
OUT_JSON = "aviation_cthr_on_aspv2_candidates_summary.json"
OUT_MD = "aviation_cthr_on_aspv2_candidates_report.md"


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
            "design_intent": query.get("design_intent"),
            "decision_variable_names": list(query.get("decision_variables", {}).keys()),
            "objective_names": [item.get("name") for item in query.get("objectives", [])],
            "objective_expressions": [item.get("expression") for item in query.get("objectives", [])],
            "preference_meaning": query.get("query_preferences", {}).get("meaning"),
        }
    )
    return scenario


def flatten_structures(structures: list[list[str]]) -> list[str]:
    return sorted({rule_id for structure in structures for rule_id in structure})


def relation_target(relation: dict[str, Any]) -> str | None:
    target = relation.get("target") or relation.get("to") or relation.get("rule_id")
    return str(target) if target is not None else None


def relation_types_between(left: str, right: str, rules_by_id: dict[str, dict[str, Any]]) -> list[str]:
    rels: list[str] = []
    for source, target in ((left, right), (right, left)):
        for relation in rules_by_id.get(source, {}).get("relations", []):
            if relation_target(relation) == target:
                rels.append(f"{source}->{target}:{relation.get('type')}")
    return rels


def annotate_failure_causes(row: dict[str, Any], rules_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    reasons: list[str] = []
    if row["candidate_missing_reference_rules"]:
        reasons.append("candidate_set_missing_reference_rules")
    if row["extra_rules"]:
        reasons.append("resolver_did_not_remove_extra_rules")
    reference_ids = set(row["predicted_rule_ids"]) - set(row["extra_rules"])
    extras_without_metadata: list[str] = []
    for extra in row["extra_rules"]:
        related = []
        for ref in reference_ids:
            related.extend(relation_types_between(extra, ref, rules_by_id))
        if not related:
            extras_without_metadata.append(extra)
    if extras_without_metadata:
        reasons.append("extra_rules_lack_conflict_override_exclusion_metadata_to_reference")
    row["failure_reasons"] = reasons
    row["extra_rules_without_candidate_metadata"] = extras_without_metadata
    return row


def row_common(
    mode: str,
    query: dict[str, Any],
    candidate_rule_ids: list[str],
    predicted_structures: list[list[str]],
    reference: list[list[str]],
    resolver_time_ms: float,
    status: str,
    extra_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = evaluate_structures(predicted_structures, reference)
    predicted_ids = flatten_structures(predicted_structures)
    reference_ids = flatten_structures(reference)
    candidate_set = set(candidate_rule_ids)
    row = {
        "mode": mode,
        "task_id": query["omega_id"],
        "title": query["title"],
        "status": status,
        "candidate_rule_count": len(candidate_rule_ids),
        "predicted_rule_count": len(predicted_ids),
        "reference_rule_count": len(reference_ids),
        "resolver_time_ms": resolver_time_ms,
        "strict_valid_structure_accuracy": metrics["valid_structure_accuracy"],
        "rule_id_precision": metrics["rule_structure_precision"],
        "rule_id_recall": metrics["rule_structure_recall"],
        "structure_precision": metrics["structure_precision"],
        "structure_recall": metrics["structure_recall"],
        "missing_rules": metrics["missing_rules"],
        "extra_rules": metrics["extra_rules"],
        "missing_structures": metrics["missing_structures"],
        "extra_structures": metrics["extra_structures"],
        "candidate_missing_reference_rules": sorted(set(reference_ids) - candidate_set),
        "candidate_extra_reference_rules": sorted(candidate_set - set(reference_ids)),
        "candidate_rule_ids": candidate_rule_ids,
        "predicted_rule_ids": predicted_ids,
        "predicted_valid_structures": predicted_structures,
        "failure_reasons": [],
        "extra_rules_without_candidate_metadata": [],
    }
    if extra_payload:
        row.update(extra_payload)
    return row


def run_asp_candidate_mode(
    rule_library: dict[str, Any],
    query: dict[str, Any],
    candidate_rule_ids: list[str],
    reference: list[list[str]],
    max_answer_sets: int,
) -> tuple[dict[str, Any], AspEnumerationResult]:
    result = enumerate_rule_structures(
        rule_library=rule_library,
        scenario=scenario_from_query(query),
        task_id=query["omega_id"],
        candidate_rule_ids=candidate_rule_ids,
        max_answer_sets=max_answer_sets,
    )
    row = row_common(
        mode="asp_v2_candidates__asp_resolver",
        query=query,
        candidate_rule_ids=candidate_rule_ids,
        predicted_structures=result.asp_rule_structures,
        reference=reference,
        resolver_time_ms=result.enumeration_time_ms,
        status=result.status,
        extra_payload={
            "asp_answer_sets": result.number_of_answer_sets,
            "resolver_notes": [],
            "defeated_rule_ids": [],
            "removed_rule_ids": [],
        },
    )
    return row, result


def run_cthr_candidate_mode(
    mode: str,
    query: dict[str, Any],
    candidate_rules: list[dict[str, Any]],
    candidate_rule_ids: list[str],
    reference: list[list[str]],
) -> tuple[dict[str, Any], CthrResolverResult]:
    result = resolve_valid_structures_with_diagnostics(candidate_rules, scenario_from_query(query))
    row = row_common(
        mode=mode,
        query=query,
        candidate_rule_ids=candidate_rule_ids,
        predicted_structures=result.valid_rule_structures,
        reference=reference,
        resolver_time_ms=result.resolver_time_ms,
        status=result.status,
        extra_payload={
            "asp_answer_sets": "",
            "resolver_notes": result.notes,
            "defeated_rule_ids": result.defeated_rule_ids,
            "removed_rule_ids": result.removed_rule_ids,
            "applicable_rule_ids": result.applicable_rule_ids,
            "error": result.error,
        },
    )
    return row, result


def cthr_full_row(query: dict[str, Any], reference: list[list[str]]) -> dict[str, Any]:
    reference_ids = flatten_structures(reference)
    return row_common(
        mode="cthr_full_reference",
        query=query,
        candidate_rule_ids=reference_ids,
        predicted_structures=reference,
        reference=reference,
        resolver_time_ms=0.0,
        status="success",
        extra_payload={
            "asp_answer_sets": "",
            "resolver_notes": ["reference CTHR full endpoint, included for comparison"],
            "defeated_rule_ids": [],
            "removed_rule_ids": [],
            "applicable_rule_ids": reference_ids,
            "error": "",
        },
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "mode",
        "task_id",
        "title",
        "status",
        "candidate_rule_count",
        "predicted_rule_count",
        "reference_rule_count",
        "resolver_time_ms",
        "strict_valid_structure_accuracy",
        "rule_id_precision",
        "rule_id_recall",
        "structure_precision",
        "structure_recall",
        "missing_rules",
        "extra_rules",
        "missing_structures",
        "extra_structures",
        "candidate_missing_reference_rules",
        "candidate_extra_reference_rules",
        "candidate_rule_ids",
        "predicted_rule_ids",
        "predicted_valid_structures",
        "asp_answer_sets",
        "resolver_notes",
        "defeated_rule_ids",
        "removed_rule_ids",
        "applicable_rule_ids",
        "error",
        "failure_reasons",
        "extra_rules_without_candidate_metadata",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            serializable = dict(row)
            for key, value in list(serializable.items()):
                if isinstance(value, (list, dict)):
                    serializable[key] = json.dumps(value, ensure_ascii=False)
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
        "mean_predicted_rule_count": sum(row["predicted_rule_count"] for row in rows) / total if total else 0.0,
        "strict_valid_structure_accuracy_percent": pct(
            sum(row["strict_valid_structure_accuracy"] for row in rows) / total if total else 0.0
        ),
        "mean_rule_id_precision_percent": pct(sum(row["rule_id_precision"] for row in rows) / total if total else 0.0),
        "mean_rule_id_recall_percent": pct(sum(row["rule_id_recall"] for row in rows) / total if total else 0.0),
        "mean_structure_precision_percent": pct(
            sum(row["structure_precision"] for row in rows) / total if total else 0.0
        ),
        "mean_structure_recall_percent": pct(sum(row["structure_recall"] for row in rows) / total if total else 0.0),
        "mean_resolver_time_ms": sum(row["resolver_time_ms"] for row in rows) / total if total else 0.0,
        "tasks_with_candidate_missing_reference_rules": [
            row["task_id"] for row in rows if row["candidate_missing_reference_rules"]
        ],
        "tasks_with_extra_predicted_rules": [row["task_id"] for row in rows if row["extra_rules"]],
        "failure_reason_counts": {
            reason: sum(reason in row.get("failure_reasons", []) for row in rows)
            for reason in sorted({reason for row in rows for reason in row.get("failure_reasons", [])})
        },
    }


def build_summary(rows: list[dict[str, Any]], runtime_s: float, args: argparse.Namespace) -> dict[str, Any]:
    modes = [
        "asp_v2_candidates__asp_resolver",
        "asp_v2_candidates__cthr_resolver",
        "asp_v2_selected__cthr_resolver_diagnostic",
        "cthr_full_reference",
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
        "# CTHR-on-ASP-v2-Candidates Diagnostic",
        "",
        "This diagnostic uses the same ASP-v2 heuristic candidate set and compares two resolvers: ASP enumeration and the CTHR valid-rule resolver. Hidden labels are used only for evaluation.",
        "",
        "The `asp_v2_selected__cthr_resolver_diagnostic` mode is diagnostic only: it feeds ASP's selected rules into CTHR, so it is not the main comparison.",
        "",
        "## Summary",
        "",
        "| Mode | Status | Candidates | Predicted | Strict acc. (%) | Rule P/R (%) | Struct. P/R (%) | Time ms | Candidate-missing refs | Extra predicted | Failure reasons |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for mode, stats in summary["summaries"].items():
        lines.append(
            "| {mode} | `{status}` | {cand:.1f} | {pred:.1f} | {acc:.1f} | {rp:.1f}/{rr:.1f} | {sp:.1f}/{sr:.1f} | {time:.2f} | {cmiss} | {extra} | `{reasons}` |".format(
                mode=mode,
                status=json.dumps(stats["status_counts"], ensure_ascii=False),
                cand=stats["mean_candidate_rule_count"],
                pred=stats["mean_predicted_rule_count"],
                acc=stats["strict_valid_structure_accuracy_percent"],
                rp=stats["mean_rule_id_precision_percent"],
                rr=stats["mean_rule_id_recall_percent"],
                sp=stats["mean_structure_precision_percent"],
                sr=stats["mean_structure_recall_percent"],
                time=stats["mean_resolver_time_ms"],
                cmiss=", ".join(stats["tasks_with_candidate_missing_reference_rules"]) or "-",
                extra=", ".join(stats["tasks_with_extra_predicted_rules"]) or "-",
                reasons=json.dumps(stats["failure_reason_counts"], ensure_ascii=False),
            )
        )

    cthr_rows = [row for row in summary["per_case"] if row["mode"] == "asp_v2_candidates__cthr_resolver"]
    lines.extend(
        [
            "",
            "## Per-Case CTHR Resolver on ASP-v2 Candidates",
            "",
            "| Task | Candidates | Predicted | Strict acc. (%) | Rule P/R (%) | Missing rules | Extra rules | Failure reasons |",
            "|---|---:|---:|---:|---:|---|---|---|",
        ]
    )
    for row in cthr_rows:
        lines.append(
            "| {task} | {cand} | {pred} | {acc:.1f} | {rp:.1f}/{rr:.1f} | {missing} | {extra} | {reasons} |".format(
                task=row["task_id"],
                cand=row["candidate_rule_count"],
                pred=row["predicted_rule_count"],
                acc=pct(row["strict_valid_structure_accuracy"]),
                rp=pct(row["rule_id_precision"]),
                rr=pct(row["rule_id_recall"]),
                missing=", ".join(row["missing_rules"]) or "-",
                extra=", ".join(row["extra_rules"]) or "-",
                reasons=", ".join(row.get("failure_reasons", [])) or "-",
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-answer-sets", type=int, default=200)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    rule_library = read_json(RULE_LIBRARY_PATH)
    rules_by_id = {str(rule["rule_id"]): rule for rule in rule_library.get("rules", []) if rule.get("rule_id")}
    queries = read_json(OPT_QUERIES_PATH)["items"]
    labels = {item["omega_id"]: item for item in read_json(RULE_LABELS_PATH)["items"]}

    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    for query in queries:
        reference = reference_structures_from_label(labels[query["omega_id"]])
        candidate_rule_ids = retrieve_candidate_rules(rule_library, query)
        candidate_rules = [rules_by_id[rule_id] for rule_id in candidate_rule_ids if rule_id in rules_by_id]

        asp_row, asp_result = run_asp_candidate_mode(
            rule_library,
            query,
            candidate_rule_ids,
            reference,
            args.max_answer_sets,
        )
        rows.append(annotate_failure_causes(asp_row, rules_by_id))

        cthr_row, _cthr_result = run_cthr_candidate_mode(
            "asp_v2_candidates__cthr_resolver",
            query,
            candidate_rules,
            candidate_rule_ids,
            reference,
        )
        rows.append(annotate_failure_causes(cthr_row, rules_by_id))

        asp_selected_ids = flatten_structures(asp_result.asp_rule_structures)
        asp_selected_rules = [rules_by_id[rule_id] for rule_id in asp_selected_ids if rule_id in rules_by_id]
        diag_row, _diag_result = run_cthr_candidate_mode(
            "asp_v2_selected__cthr_resolver_diagnostic",
            query,
            asp_selected_rules,
            asp_selected_ids,
            reference,
        )
        rows.append(annotate_failure_causes(diag_row, rules_by_id))
        rows.append(annotate_failure_causes(cthr_full_row(query, reference), rules_by_id))

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
