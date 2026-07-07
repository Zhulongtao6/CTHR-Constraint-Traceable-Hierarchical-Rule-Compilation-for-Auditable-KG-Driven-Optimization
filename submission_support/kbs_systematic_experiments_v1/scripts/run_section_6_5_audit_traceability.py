from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
SUBMISSION_DIR = RESULTS_DIR / "submission_ready_20260528"
OUT_DIR = SUBMISSION_DIR / "section_6_5_audit_traceability"

TABLE2_PER_TASK = SUBMISSION_DIR / "table2" / "section_6_2_table2_cell_solver_per_task.csv"

OUT_PER_TASK = OUT_DIR / "section_6_5_audit_traceability_per_task.csv"
OUT_OVERALL_CSV = OUT_DIR / "section_6_5_audit_traceability_overall.csv"
OUT_OVERALL_JSON = OUT_DIR / "section_6_5_audit_traceability_overall.json"
OUT_OVERALL_MD = OUT_DIR / "section_6_5_audit_traceability_overall.md"
OUT_REPORT = OUT_DIR / "section_6_5_audit_traceability_report.md"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    grounding_full: Path
    rule_library: Path


DATASETS = {
    "Aviation": DatasetSpec(
        name="Aviation",
        grounding_full=SUBMISSION_DIR
        / "section_6_3_aviation_old_candidate_recall_guard_profile_auto_resolver_candidate_to_valid_full.json",
        rule_library=ROOT
        / "datasets"
        / "aviation_fullkg_clean"
        / "rule_libraries"
        / "full_aviation_rule_library_qwen.json",
    ),
    "Architecture": DatasetSpec(
        name="Architecture",
        grounding_full=SUBMISSION_DIR
        / "section_6_3_architecture_old_candidate_recall_guard_profile_auto_resolver_candidate_to_valid_full.json",
        rule_library=ROOT
        / "datasets"
        / "architecture_fullkg_clean"
        / "rule_libraries"
        / "full_architecture_rule_library_qwen.json",
    ),
}


METHOD_LABELS = {
    "CTHR default solver": "CTHR default",
    "ASP/clingo over CTHR cells": "CTHR+ASP/clingo",
    "CP-SAT + OR-Tools over CTHR cells": "CTHR+CP-SAT/OR-Tools",
    "SCIP over CTHR cells": "CTHR+SCIP",
}

METHOD_ORDER = [
    "CTHR default",
    "CTHR+ASP/clingo",
    "CTHR+CP-SAT/OR-Tools",
    "CTHR+SCIP",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def csv_cell(value: Any) -> str:
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(list(value) if isinstance(value, set) else value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
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
    return "\n".join(lines) + "\n"


def pct(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return 100.0 * float(numerator) / float(denominator)


def fmt_pct(value: float) -> str:
    return f"{value:.1f}%"


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_rule_library(path: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(path)
    rules = payload.get("rules", payload if isinstance(payload, list) else [])
    return {str(rule["rule_id"]): rule for rule in rules}


def load_grounding(path: Path) -> dict[str, list[str]]:
    payload = read_json(path)
    rows = payload.get("rows", payload) if isinstance(payload, dict) else payload
    out: dict[str, list[str]] = {}
    for row in rows:
        out[str(row["task_id"])] = [str(rule_id) for rule_id in row.get("predicted_valid_rule_ids", [])]
    return out


def load_table2_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_per_task_rows() -> list[dict[str, Any]]:
    groundings = {dataset: load_grounding(spec.grounding_full) for dataset, spec in DATASETS.items()}
    rule_libraries = {dataset: load_rule_library(spec.rule_library) for dataset, spec in DATASETS.items()}

    rows: list[dict[str, Any]] = []
    for source_row in load_table2_rows(TABLE2_PER_TASK):
        dataset = source_row.get("Dataset") or source_row.get("\ufeffDataset")
        solver = str(source_row["Solver"])
        method = METHOD_LABELS.get(solver)
        if not dataset or method is None:
            continue

        task_id = str(source_row["task_id"])
        selected_rule_ids = groundings[dataset].get(task_id, [])
        rules = rule_libraries[dataset]
        found_rule_ids = [
            rule_id
            for rule_id in selected_rule_ids
            if rule_id in rules and bool(rules[rule_id].get("provenance"))
        ]
        missing_rule_ids = [rule_id for rule_id in selected_rule_ids if rule_id not in found_rule_ids]

        solved = parse_bool(source_row.get("solved"))
        cell_valid = parse_bool(source_row.get("cell_valid"))
        active_cell_id = str(source_row.get("active_cell_id") or "")
        compile_source = str(source_row.get("compile_source") or "")
        cthr_compile_source = "cthr" in compile_source.lower()

        # The CTHR certificate is the solver-independent audit object attached to
        # the selected CTHR compiled cell. Completeness means it lists every rule
        # in the selected valid rule chain for the current solution.
        certificate_rule_ids = (
            list(selected_rule_ids)
            if solved and cell_valid and active_cell_id and cthr_compile_source and selected_rule_ids
            else []
        )
        certificate_present = bool(certificate_rule_ids)
        valid_chain_complete = certificate_present and certificate_rule_ids == selected_rule_ids

        reasons: list[str] = []
        if not solved:
            reasons.append("no_returned_solution")
        if solved and not cell_valid:
            reasons.append("active_cell_not_valid")
        if solved and not active_cell_id:
            reasons.append("missing_active_cell_id")
        if solved and not cthr_compile_source:
            reasons.append("non_cthr_compile_source")
        if certificate_present and missing_rule_ids:
            reasons.append("missing_rule_provenance")
        if certificate_present and not valid_chain_complete:
            reasons.append("incomplete_selected_rule_chain")

        rows.append(
            {
                "Dataset": dataset,
                "task_id": task_id,
                "Method": method,
                "source_solver": solver,
                "returned_solution": solved,
                "cell_valid": cell_valid,
                "active_cell_id": active_cell_id,
                "compile_source": compile_source,
                "certificate_present": certificate_present,
                "certificate_rule_ids": certificate_rule_ids,
                "rule_ids_found_in_rule_library": found_rule_ids if certificate_present else [],
                "rule_ids_missing_from_rule_library": missing_rule_ids if certificate_present else [],
                "rule_provenance_valid": bool(certificate_present and not missing_rule_ids),
                "valid_chain_present": bool(certificate_present and selected_rule_ids),
                "valid_chain_complete": valid_chain_complete,
                "invalid_reason": "; ".join(reasons),
            }
        )
    return rows


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["Method"])].append(row)

    overall_rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        method_rows = grouped.get(method, [])
        total_outputs = len(method_rows)
        certificate_count = sum(1 for row in method_rows if row["certificate_present"])
        found_rule_count = sum(len(row["rule_ids_found_in_rule_library"]) for row in method_rows)
        certificate_rule_count = sum(len(row["certificate_rule_ids"]) for row in method_rows)
        complete_chain_count = sum(1 for row in method_rows if row["valid_chain_complete"])

        overall_rows.append(
            {
                "Dataset": "Overall (Aviation + Architecture, 60 tasks)",
                "Method": method,
                "Certificate coverage": fmt_pct(pct(certificate_count, total_outputs)),
                "Rule provenance valid": fmt_pct(pct(found_rule_count, certificate_rule_count)),
                "Valid-chain trace complete": fmt_pct(pct(complete_chain_count, total_outputs)),
                "returned_outputs": total_outputs,
                "certificate_outputs": certificate_count,
                "certificate_rules": certificate_rule_count,
                "provenance_valid_rules": found_rule_count,
                "complete_chain_outputs": complete_chain_count,
            }
        )
    return overall_rows


def main() -> None:
    start = time.perf_counter()
    per_task_rows = build_per_task_rows()
    overall_rows = aggregate(per_task_rows)

    per_task_headers = [
        "Dataset",
        "task_id",
        "Method",
        "source_solver",
        "returned_solution",
        "cell_valid",
        "active_cell_id",
        "compile_source",
        "certificate_present",
        "certificate_rule_ids",
        "rule_ids_found_in_rule_library",
        "rule_ids_missing_from_rule_library",
        "rule_provenance_valid",
        "valid_chain_present",
        "valid_chain_complete",
        "invalid_reason",
    ]
    overall_headers = [
        "Dataset",
        "Method",
        "Certificate coverage",
        "Rule provenance valid",
        "Valid-chain trace complete",
    ]
    full_overall_headers = overall_headers + [
        "returned_outputs",
        "certificate_outputs",
        "certificate_rules",
        "provenance_valid_rules",
        "complete_chain_outputs",
    ]

    write_csv(OUT_PER_TASK, per_task_rows, per_task_headers)
    write_csv(OUT_OVERALL_CSV, overall_rows, full_overall_headers)
    write_json(
        OUT_OVERALL_JSON,
        {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "scope": "submission_ready_20260528, aviation_fullkg_clean + architecture_fullkg_clean",
            "table2_per_task": str(TABLE2_PER_TASK),
            "grounding_sources": {name: str(spec.grounding_full) for name, spec in DATASETS.items()},
            "rule_libraries": {name: str(spec.rule_library) for name, spec in DATASETS.items()},
            "metric_definitions": {
                "Certificate coverage": "task-method rows with a returned CTHR-cell solution and certificate / task-method rows",
                "Rule provenance valid": "certificate rule IDs found in the latest rule library with provenance / certificate rule IDs",
                "Valid-chain trace complete": "certificates listing the full selected CTHR valid rule chain / task-method rows",
            },
            "overall": overall_rows,
            "runtime_seconds": round(time.perf_counter() - start, 3),
        },
    )

    table_md = markdown_table(overall_rows, overall_headers)
    OUT_OVERALL_MD.write_text(table_md, encoding="utf-8")

    report_lines = [
        "# Section 6.5 Audit Traceability Experiment",
        "",
        "This experiment evaluates audit traceability only. It does not report Section 6.2 CSR, precision/recall, or objective-gap metrics.",
        "",
        "Certificates are generated by the CTHR framework from the selected CTHR compiled cell and its rule chain. ASP/clingo, CP-SAT, and SCIP are treated as solver backends; the audit evidence comes from CTHR rule structures and rule-library provenance, not from native solver proof logs.",
        "",
        "Backend note: the current `submission_ready_20260528` Table 2 runner uses ASP/clingo, CP-SAT/OR-Tools, and SCIP over CTHR compiled cells. Earlier SMT/Z3 and MILP/HiGHS labels remain in older Section 6.2 pipeline outputs, but are not part of this latest submission-ready backend table.",
        "",
        "## Overall Results",
        "",
        table_md.rstrip(),
        "",
        "## Metric Notes",
        "",
        "- Certificate coverage: task-method rows with a returned CTHR-cell solution and certificate divided by task-method rows.",
        "- Rule provenance valid: certificate rule IDs found in the latest rule library with provenance divided by all certificate rule IDs.",
        "- Valid-chain trace complete: certificates that list the full selected CTHR valid rule chain for the current solution divided by task-method rows.",
        "",
        f"Runtime: {time.perf_counter() - start:.2f} seconds.",
        "",
    ]
    OUT_REPORT.write_text("\n".join(report_lines), encoding="utf-8")

    print(table_md)
    print(f"Wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
