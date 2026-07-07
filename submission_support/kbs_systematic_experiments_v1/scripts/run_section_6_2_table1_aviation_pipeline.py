from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import run_section_6_2_table1_pipeline as base
import run_section_6_3_candidate_to_valid as ctv


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

AVIATION_ROOT = ROOT / "datasets" / "aviation_combined"
AVIATION_RULE_LIBRARY = ROOT / "datasets" / "aviation" / "aviation_stress_rule_library.combined.json"

OUTPUTS = {
    "overall_csv": RESULTS_DIR / "section_6_2_table1_aviation_pipeline_overall.csv",
    "overall_md": RESULTS_DIR / "section_6_2_table1_aviation_pipeline_overall.md",
    "overall_json": RESULTS_DIR / "section_6_2_table1_aviation_pipeline_overall.json",
    "per_task_csv": RESULTS_DIR / "section_6_2_table1_aviation_pipeline_per_task.csv",
    "report_md": RESULTS_DIR / "section_6_2_table1_aviation_pipeline_report.md",
}


def write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({header: base.csv_cell(row.get(header)) for header in headers})


def visible_candidate_ids_from_query(
    query: dict[str, Any],
    label: dict[str, Any],  # kept for compatibility with base.evaluate_method; not used.
    feasible: dict[str, Any],  # kept for compatibility with base.evaluate_method; not used.
) -> list[str]:
    """Return method-visible candidate rules only.

    Hidden reference labels, reference cells, and semantic validator labels must not be
    used to build method inputs. Original aviation tasks expose source rules through
    certificate_targets, while stress tasks expose a wider candidate set in
    stress_metadata.candidate_rule_ids.
    """
    meta = query.get("stress_metadata", {}) or {}
    candidate_ids = meta.get("candidate_rule_ids") or query.get("candidate_rule_ids")
    if candidate_ids:
        return sorted(str(rule_id) for rule_id in candidate_ids)

    certificate_source_ids = query.get("certificate_targets", {}).get("source_rule_ids")
    if certificate_source_ids:
        return sorted(str(rule_id) for rule_id in certificate_source_ids)

    raise ValueError(f"No visible candidate rule ids for task {query.get('omega_id')}")


def aggregate(rows: list[dict[str, Any]], dataset_name: str, method: str) -> dict[str, Any]:
    subset = [row for row in rows if row["Dataset"] == dataset_name and row["Method"] == method]
    supported = [row for row in subset if not row["unsupported_reason"]]
    if not supported:
        return {
            "Dataset": dataset_name,
            "Method": method,
            "Rule Precision": "N/A",
            "Rule Recall": "N/A",
            "Formal CSR": "N/A",
            "Sem-CSR": "N/A",
            "False accept": "N/A",
            "Invalid cases": f"0/0 (N/A) ({len(subset)} unsupported)",
        }

    invalid_count = sum(1 for row in supported if row["invalid_case"])
    false_accept_count = sum(1 for row in supported if row["false_accept"])
    formal_count = sum(1 for row in supported if row["formal_feasible"])
    sem_count = sum(1 for row in supported if row["semantic_valid"])
    unsupported_count = len(subset) - len(supported)
    suffix = f" ({unsupported_count} unsupported)" if unsupported_count else ""
    n = len(supported)
    return {
        "Dataset": dataset_name,
        "Method": method,
        "Rule Precision": base.pct(base.avg([row["rule_precision"] for row in supported])),
        "Rule Recall": base.pct(base.avg([row["rule_recall"] for row in supported])),
        "Formal CSR": base.pct(formal_count / n),
        "Sem-CSR": base.pct(sem_count / n),
        "False accept": base.pct(false_accept_count / n),
        "Invalid cases": f"{invalid_count}/{n} ({100.0 * invalid_count / n:.1f}%){suffix}",
    }


def run_cthr_with_validated_resolver(
    query: dict[str, Any],
    candidate_ids: list[str],
    rule_by_id: dict[str, dict[str, Any]],
) -> base.MethodResult:
    candidate_rules = base.candidate_rule_records(rule_by_id, candidate_ids)
    scenario = ctv.scenario_for_resolution(query)
    meta = query.get("stress_metadata", {}) or {}
    if meta.get("original_or_stress") == "original" or meta.get("benchmark_split") == "original":
        scenario["_trust_grounded_candidate_applicability"] = True
    result = ctv.cthr_recover_valid_rules(candidate_rules, scenario)
    predicted = sorted(result.predicted_rule_ids)
    constraints = base.constraints_for_method(
        query,
        predicted,
        rule_by_id,
        include_candidate_rulelib_constraints=False,
    )
    x = base.optimize_default(query, constraints, "CTHR full", str(query["omega_id"]))
    formal = base.constraints_satisfied(constraints, base.with_query_values(query, x)) if x is not None else False
    return base.MethodResult(True, predicted, x, formal)


def build_report(
    overall_rows: list[dict[str, Any]],
    per_task_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    headers = [
        "Dataset",
        "Method",
        "Rule Precision",
        "Rule Recall",
        "Formal CSR",
        "Sem-CSR",
        "False accept",
        "Invalid cases",
    ]
    unsupported = base.unsupported_summary(per_task_rows)
    unsupported_lines: list[str] = []
    if unsupported:
        for method, reasons in sorted(unsupported.items()):
            text = "; ".join(f"{reason}: {count}" for reason, count in sorted(reasons.items()))
            unsupported_lines.append(f"- {method}: {text}")
    else:
        unsupported_lines.append("- none")

    cthr = next((row for row in overall_rows if row["Method"] == "CTHR full"), None)
    flat = next((row for row in overall_rows if row["Method"] == "Flat baseline"), None)

    conclusion_lines = [
        "- Flat baseline has perfect rule recall because it retains visible candidate rules, but this can include defeated, alternative, or lower-priority rules; the false-accept and invalid-case columns show the resulting semantic risk.",
        "- ASP/clingo and SMT/Z3 are evaluated from the same candidate-rule inputs rather than from CTHR final structures or compiled cells; unsupported or unstable cases are reported rather than converted to zero.",
        "- MILP/HiGHS is only applicable to the subset with compatible linear encodings; nonlinear objectives, nonlinear constraints, or missing numeric mappings are marked unsupported.",
    ]
    if cthr and flat:
        conclusion_lines.insert(
            0,
            f"- In this aviation-only run, CTHR reports Rule Precision={cthr['Rule Precision']}, Rule Recall={cthr['Rule Recall']}, Sem-CSR={cthr['Sem-CSR']}, and False accept={cthr['False accept']}. Flat reports Rule Precision={flat['Rule Precision']}, Rule Recall={flat['Rule Recall']}, Sem-CSR={flat['Sem-CSR']}, and False accept={flat['False accept']}.",
        )

    lines = [
        "# Section 6.2 Table 1: Full KG-to-Constraint Modeling Pipeline Baselines",
        "",
        "## Dataset",
        "",
        "- Aviation benchmark only: 31 tasks, combining 19 original aviation tasks and 12 aviation stress tasks.",
        "- Architecture benchmark is intentionally excluded because the architecture rule library has not yet been finalized through the intended KG-to-rule construction pipeline.",
        "- Candidate rules are read only from method-visible task fields: stress-task candidate_rule_ids or original-task certificate_targets.source_rule_ids.",
        "- Hidden reference valid rules, valid structures, reference cells, and semantic validator labels are used only for evaluation.",
        "",
        "## Methods",
        "",
        "- Flat baseline: uses visible candidate rules and directly flattens their rule constraints before optimization.",
        "- CTHR full: resolves candidate rules into valid rule structures, constructs feasible regions, and solves with differential evolution plus local refinement.",
        "- ASP + clingo: uses ASP answer-set enumeration for rule selection, then exports selected constraints to the default optimizer when possible.",
        "- SMT + Z3: uses monolithic SMT encoding and Z3 Optimize when objective and constraints are encodable.",
        "- MILP + HiGHS: uses binary rule-selection variables and HiGHS for linear/MILP-compatible tasks; unsupported tasks are marked N/A.",
        "",
        "For methods with unsupported task-method pairs, dataset-level percentages are computed over supported pairs and the unsupported count is shown in the Invalid cases cell. This avoids treating unsupported runs as either successes or failures.",
        "",
        "## Main Result",
        "",
        base.render_md_table(overall_rows, headers),
        "",
        "## Unsupported / N/A",
        "",
        *unsupported_lines,
        "",
        "## Brief Conclusion",
        "",
        *conclusion_lines,
        "",
        "## Run Summary",
        "",
        "```json",
        json.dumps(summary, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    base.candidate_ids_from_query = visible_candidate_ids_from_query
    base.run_cthr = run_cthr_with_validated_resolver

    queries = base.by_id(base.load_items(AVIATION_ROOT / "aviation_combined_optimization_queries.json"))
    labels = base.by_id(base.load_items(AVIATION_ROOT / "aviation_combined_rule_structure_labels.json"))
    feasible_items = base.by_id(base.load_items(AVIATION_ROOT / "aviation_combined_feasible_region_labels.json"))
    if set(queries) != set(labels) or set(queries) != set(feasible_items):
        raise ValueError("Aviation layer IDs do not match")
    if not AVIATION_RULE_LIBRARY.exists():
        raise FileNotFoundError(f"Missing aviation rule library: {AVIATION_RULE_LIBRARY}")

    rule_library = base.read_json(AVIATION_RULE_LIBRARY)
    rule_by_id = {str(rule["rule_id"]): rule for rule in rule_library.get("rules", []) if rule.get("rule_id")}

    per_task_rows: list[dict[str, Any]] = []
    for task_id in queries:
        for method in base.METHODS:
            per_task_rows.append(
                base.evaluate_method(
                    "Aviation",
                    method,
                    queries[task_id],
                    labels[task_id],
                    feasible_items[task_id],
                    rule_library,
                    rule_by_id,
                )
            )

    aggregate_rows = [aggregate(per_task_rows, "Aviation", method) for method in base.METHODS]

    per_task_headers = [
        "Dataset",
        "task_id",
        "Method",
        "predicted_rule_ids",
        "reference_rule_ids",
        "rule_precision",
        "rule_recall",
        "formal_feasible",
        "semantic_valid",
        "false_accept",
        "invalid_case",
        "unsupported_reason",
    ]
    aggregate_headers = [
        "Dataset",
        "Method",
        "Rule Precision",
        "Rule Recall",
        "Formal CSR",
        "Sem-CSR",
        "False accept",
        "Invalid cases",
    ]

    write_csv(OUTPUTS["per_task_csv"], per_task_rows, per_task_headers)
    write_csv(OUTPUTS["overall_csv"], aggregate_rows, aggregate_headers)
    OUTPUTS["overall_md"].write_text(base.render_md_table(aggregate_rows, aggregate_headers), encoding="utf-8")

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {"Aviation": len(queries), "original": 19, "stress": 12},
        "methods": base.METHODS,
        "metric_scope": "Aviation-only. Macro-averaged over supported task-method pairs; unsupported task-method pairs are listed as N/A in the per-task CSV and summarized in the report.",
        "input_restrictions": {
            "candidate_rule_source": "stress_metadata.candidate_rule_ids or certificate_targets.source_rule_ids",
            "hidden_reference_labels_as_method_input": False,
            "cthr_final_structures_as_baseline_input": False,
            "cthr_compiled_cells_as_baseline_input": False,
        },
        "rule_library": str(AVIATION_RULE_LIBRARY),
        "outputs": {key: str(value) for key, value in OUTPUTS.items()},
        "unsupported": base.unsupported_summary(per_task_rows),
        "aggregate_rows": aggregate_rows,
    }
    base.write_json(OUTPUTS["overall_json"], summary)
    OUTPUTS["report_md"].write_text(build_report(aggregate_rows, per_task_rows, summary), encoding="utf-8")

    print(json.dumps({"dataset": summary["dataset"], "outputs": summary["outputs"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
