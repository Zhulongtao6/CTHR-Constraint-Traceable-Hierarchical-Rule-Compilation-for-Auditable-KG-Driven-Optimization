from __future__ import annotations

import json
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
CTHR_ROOT = THIS_DIR.parents[1]
PAPER_DIR = CTHR_ROOT / "paper"
INPUT_PATH = PAPER_DIR / "architecture_kg_generated_20_optimization_problems.json"
OUT_DIR = PAPER_DIR / "architecture_benchmark_layers"

RULE_LABELS_PATH = OUT_DIR / "architecture_rule_structure_labels.json"
FEASIBLE_LABELS_PATH = OUT_DIR / "architecture_feasible_region_labels.json"
OPT_QUERIES_PATH = OUT_DIR / "architecture_optimization_queries.json"
MANIFEST_PATH = OUT_DIR / "architecture_benchmark_layers_manifest.json"
SUMMARY_PATH = OUT_DIR / "ARCHITECTURE_BENCHMARK_LAYERS_SUMMARY.md"


import materialize_aviation_benchmark_layers as mat  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_summary(rule_labels: list[dict[str, Any]], feasible_labels: list[dict[str, Any]]) -> str:
    num_executable = sum(len(item["executable_constraints"]) for item in feasible_labels)
    num_structure_only = sum(len(item["structure_only_constraints"]) for item in feasible_labels)
    num_cells = sum(len(item.get("valid_constraint_cells", [])) for item in feasible_labels)
    challenge_counts: dict[str, int] = {}
    for item in rule_labels:
        for label in item["challenge_types"]:
            challenge_counts[label] = challenge_counts.get(label, 0) + 1

    lines = [
        "# Architecture Benchmark Layers",
        "",
        f"- Problems: {len(rule_labels)}",
        f"- Executable constraints: {num_executable}",
        f"- Structure-only constraints: {num_structure_only}",
        f"- Piecewise feasible cells: {num_cells}",
        "",
        "## Challenge Types",
        "",
        "| Challenge | Cases |",
        "|---|---:|",
    ]
    for label, count in sorted(challenge_counts.items()):
        lines.append(f"| {label} | {count} |")
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- `{RULE_LABELS_PATH}`",
            f"- `{FEASIBLE_LABELS_PATH}`",
            f"- `{OPT_QUERIES_PATH}`",
            f"- `{MANIFEST_PATH}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    payload = read_json(INPUT_PATH)
    problems = payload["problems"]
    rule_labels = [mat.build_rule_structure_label(problem) for problem in problems]
    feasible_labels = [mat.build_feasible_region_label(problem) for problem in problems]
    opt_queries = [mat.build_optimization_query(problem) for problem in problems]

    manifest = {
        "version": "architecture_benchmark_layers_v1",
        "source_problem_file": str(INPUT_PATH),
        "num_problems": len(problems),
        "files": {
            "rule_structure_labels": str(RULE_LABELS_PATH),
            "feasible_region_labels": str(FEASIBLE_LABELS_PATH),
            "optimization_queries": str(OPT_QUERIES_PATH),
        },
        "evaluation_layers": [
            {
                "name": "rule_structure_correctness",
                "file": str(RULE_LABELS_PATH),
                "purpose": "Check whether KG-to-rule compilation selects the expected source rules, defeated rules, valid structures, and provenance.",
            },
            {
                "name": "semantic_feasible_region",
                "file": str(FEASIBLE_LABELS_PATH),
                "purpose": "Check whether returned decisions satisfy the source-rule feasible region and valid rule-structure cells.",
            },
            {
                "name": "optimization_queries",
                "file": str(OPT_QUERIES_PATH),
                "purpose": "Run constrained multi-objective optimization over the compiled architecture feasible regions.",
            },
        ],
    }
    write_json(RULE_LABELS_PATH, {"rule_structure_labels": rule_labels})
    write_json(FEASIBLE_LABELS_PATH, {"feasible_region_labels": feasible_labels})
    write_json(OPT_QUERIES_PATH, {"optimization_queries": opt_queries})
    write_json(MANIFEST_PATH, manifest)
    SUMMARY_PATH.write_text(build_summary(rule_labels, feasible_labels), encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(MANIFEST_PATH),
                "num_problems": len(problems),
                "rule_labels": len(rule_labels),
                "feasible_labels": len(feasible_labels),
                "optimization_queries": len(opt_queries),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
