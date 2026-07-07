from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
LOGS_DIR = ROOT / "logs"
TABLE1_RUNNER = ROOT / "scripts" / "run_section_6_2_table1_aviation_old_candidate_profile_all_methods.py"
DATASET_ROOT = ROOT / "datasets" / "aviation_strong_mixed_v11_150"
GROUNDING_FULL = (
    RESULTS_DIR
    / "section_6_3_aviation_strong_mixed_v11_150_old_candidate_recall_guard_profile_auto_resolver_candidate_to_valid_full.json"
)

METHODS = [
    "Flat baseline",
    "Native ASP + clingo",
    "Native CP-SAT + OR-Tools",
    "Native SCIP",
    "CTHR default",
    "CTHR-style ASP + clingo",
    "CTHR-style CP-SAT + OR-Tools",
    "CTHR-style SCIP",
]


def csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(list(value) if isinstance(value, set) else value, ensure_ascii=False)
    return str(value)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
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


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def run_method(
    method: str,
    output_prefix: str,
    max_tasks: int | None,
    timeout: int,
    grounding_full: Path,
    grounding_source_label: str,
) -> dict[str, Any]:
    method_prefix = f"{output_prefix}_{slug(method)}"
    cmd = [
        sys.executable,
        str(TABLE1_RUNNER),
        "--domain",
        "aviation",
        "--dataset-root",
        str(DATASET_ROOT),
        "--algorithm-inputs",
        str(DATASET_ROOT / "algorithm_inputs" / "aviation_strong_mixed_algorithm_inputs.json"),
        "--scenario-models",
        str(DATASET_ROOT / "scenario_models" / "aviation_strong_mixed_public_scenario_models.json"),
        "--evaluation-references",
        str(DATASET_ROOT / "evaluation_overlays" / "qwen" / "evaluation_references.json"),
        "--rule-library",
        str(DATASET_ROOT / "rule_libraries" / "qwen" / "full_aviation_rule_library_qwen.json"),
        "--constraint-templates",
        str(DATASET_ROOT / "evaluation_overlays" / "qwen" / "compiled_rule_constraint_templates.json"),
        "--grounding-full",
        str(grounding_full),
        "--grounding-source-label",
        grounding_source_label,
        "--output-prefix",
        method_prefix,
        "--methods",
        method,
    ]
    if max_tasks is not None:
        cmd.extend(["--max-tasks", str(max_tasks)])

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{method_prefix}_subprocess.log"
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(ROOT.parents[1]),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        log_payload = {
            "method": method,
            "returncode": completed.returncode,
            "cmd": cmd,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except subprocess.TimeoutExpired as exc:
        log_payload = {
            "method": method,
            "returncode": "timeout",
            "cmd": cmd,
            "stdout": exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout,
            "stderr": exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr,
        }
    log_path.write_text(json.dumps(log_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return log_payload | {"output_prefix": method_prefix, "log_path": str(log_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run aviation strong mixed v11 150 Table 1 with method isolation.")
    parser.add_argument(
        "--output-prefix",
        default="section_6_2_table1_aviation_strong_mixed_v11_150_old_candidate_profile_auto_fast_isolated",
    )
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--methods", default=None)
    parser.add_argument("--grounding-full", type=Path, default=GROUNDING_FULL)
    parser.add_argument(
        "--grounding-source-label",
        default="old candidate mapping with aviation recall guard and profile auto valid-rule recovery",
    )
    args = parser.parse_args()

    selected_methods = METHODS
    if args.methods:
        requested = {item.strip() for item in args.methods.split(",") if item.strip()}
        selected_methods = [method for method in METHODS if method in requested]
        missing = requested - set(selected_methods)
        if missing:
            raise ValueError(f"Unknown method name(s): {sorted(missing)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    runs = [
        run_method(
            method,
            args.output_prefix,
            args.max_tasks,
            args.timeout,
            args.grounding_full,
            args.grounding_source_label,
        )
        for method in selected_methods
    ]

    all_overall: list[dict[str, Any]] = []
    all_per_task: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for run in runs:
        if run["returncode"] != 0:
            failures.append(run)
            continue
        prefix = run["output_prefix"]
        overall_path = RESULTS_DIR / f"{prefix}_overall.csv"
        per_task_path = RESULTS_DIR / f"{prefix}_per_task.csv"
        if overall_path.exists():
            all_overall.extend(read_csv(overall_path))
        if per_task_path.exists():
            all_per_task.extend(read_csv(per_task_path))

    overall_headers = [
        "Dataset",
        "Method",
        "Method type",
        "Rule Precision",
        "Rule Recall",
        "Formal CSR",
        "Sem-CSR",
        "False accept",
        "Invalid cases",
    ]
    per_task_headers = list(all_per_task[0].keys()) if all_per_task else []
    overall_csv = RESULTS_DIR / f"{args.output_prefix}_overall.csv"
    overall_md = RESULTS_DIR / f"{args.output_prefix}_overall.md"
    overall_json = RESULTS_DIR / f"{args.output_prefix}_overall.json"
    per_task_csv = RESULTS_DIR / f"{args.output_prefix}_per_task.csv"
    summary_json = RESULTS_DIR / f"{args.output_prefix}_summary.json"

    if all_overall:
        write_csv(overall_csv, all_overall, overall_headers)
        overall_md.write_text(markdown_table(all_overall, overall_headers), encoding="utf-8")
        overall_json.write_text(json.dumps(all_overall, ensure_ascii=False, indent=2), encoding="utf-8")
    if all_per_task and per_task_headers:
        write_csv(per_task_csv, all_per_task, per_task_headers)

    summary = {
        "dataset_root": str(DATASET_ROOT),
        "grounding_full": str(args.grounding_full),
        "grounding_source_label": args.grounding_source_label,
        "methods_requested": selected_methods,
        "methods_completed": [row.get("Method") for row in all_overall],
        "failures": failures,
        "outputs": {
            "overall_csv": str(overall_csv),
            "overall_md": str(overall_md),
            "overall_json": str(overall_json),
            "per_task_csv": str(per_task_csv),
            "summary_json": str(summary_json),
        },
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
