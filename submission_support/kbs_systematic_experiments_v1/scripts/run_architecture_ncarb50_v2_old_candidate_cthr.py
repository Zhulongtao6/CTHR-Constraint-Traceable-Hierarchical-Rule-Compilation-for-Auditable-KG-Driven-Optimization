from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CTHR_ROOT = ROOT.parents[1]
SCRIPTS_DIR = ROOT / "scripts"
DEFAULT_DATASET_ROOT = ROOT / "datasets" / "architecture_fullkg_ncarb50_v2"
DEFAULT_OUT_DIR = ROOT / "results" / "architecture_ncarb50_v2_20260626"

sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(CTHR_ROOT))

import run_old_candidate_llm_filter_grounding as oldrun  # noqa: E402
import run_section_6_2_table1_fullkg_pipeline as full  # noqa: E402


def dataset_spec(dataset_root: Path, out_dir: Path) -> full.DatasetSpec:
    return full.DatasetSpec(
        name="Architecture-NCARB50-v2",
        domain="architecture",
        root=dataset_root,
        algorithm_inputs=dataset_root / "algorithm_inputs" / "architecture_algorithm_inputs.json",
        scenario_models=dataset_root / "scenario_models" / "architecture_public_scenario_models.json",
        evaluation_references=dataset_root / "evaluation_references" / "architecture_evaluation_references.json",
        rule_library=dataset_root / "rule_libraries" / "full_architecture_rule_library_qwen.json",
        grounding_full=out_dir
        / "section_6_3_architecture_ncarb50_v2_old_candidate_profile_resolver_candidate_to_valid_full.json",
        constraint_templates=dataset_root / "constraint_templates" / "compiled_rule_constraint_templates.json",
    )


def ratio(numer: int, denom: int) -> float:
    return math.nan if denom == 0 else numer / denom


def rounded(value: float) -> float:
    return value if isinstance(value, float) and math.isnan(value) else round(float(value), 4)


def metric_value(value: Any) -> float:
    numeric = float(value)
    return 0.0 if math.isnan(numeric) else numeric


def task_semantic_status(
    spec: full.DatasetSpec,
    task_id: str,
    grounding_task: dict[str, Any],
    scenario_model: dict[str, Any],
    reference: dict[str, Any],
    predicted_ids: list[str],
    rule_by_id: dict[str, dict[str, Any]],
) -> tuple[bool, bool, str]:
    if not predicted_ids:
        return False, False, "no_predicted_rules"
    try:
        query = full.prepare_query(grounding_task, scenario_model)
        templates_by_rule = full.constraint_template_map(spec.constraint_templates)
        query["_compiled_rule_constraint_templates_by_id"] = templates_by_rule
        feasible = full.reference_feasible(reference, query)
        reference_ids = full.reference_rule_ids(reference)
        x, formal = full.solve_with_default("CTHR default", query, sorted(predicted_ids), rule_by_id)
        if x is None:
            return False, False, "no_solution"
        semantic = full.semantic_valid(feasible, x, predicted_ids, reference_ids)
        return bool(formal), bool(semantic), ""
    except Exception as exc:  # noqa: BLE001
        return False, False, f"{task_id}:solver_or_semantic_error:{type(exc).__name__}"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "task count": len(rows),
        "mean Candidate / Reference Ratio": statistics.mean(row["Candidate / Reference Ratio"] for row in rows),
        "mean Filtered / Reference Ratio": statistics.mean(row["Filtered / Reference Ratio"] for row in rows),
        "mean Predicted / Reference Ratio": statistics.mean(row["Predicted / Reference Ratio"] for row in rows),
        "mean Rule-ID Precision": statistics.mean(metric_value(row["Rule-ID Precision"]) for row in rows),
        "mean Rule-ID Recall": statistics.mean(metric_value(row["Rule-ID Recall"]) for row in rows),
        "exact match rate": statistics.mean(1.0 if row["Exact Match"] else 0.0 for row in rows),
        "semantic constraint satisfaction rate": statistics.mean(
            1.0 if row["Semantic Constraint Satisfied"] else 0.0 for row in rows
        ),
        "formal feasible rate": statistics.mean(1.0 if row["Formal Feasible"] else 0.0 for row in rows),
        "total extra rules": sum(len(row["extra_rule_ids"]) for row in rows),
        "total missing rules": sum(len(row["missing_rule_ids"]) for row in rows),
        "semantic unsupported/error count": sum(1 for row in rows if row["semantic_status_note"]),
    }


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for key in (
        "Candidate / Reference Ratio",
        "Filtered / Reference Ratio",
        "Predicted / Reference Ratio",
        "Rule-ID Precision",
        "Rule-ID Recall",
    ):
        out[key] = rounded(out[key])
    return out


def markdown_report(
    *,
    spec: full.DatasetSpec,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    valid_resolver: str,
    limit: int,
    out_paths: dict[str, str],
    elapsed_sec: float,
) -> str:
    failure_rows = [
        row
        for row in rows
        if row["extra_rule_ids"]
        or row["missing_rule_ids"]
        or not row["Semantic Constraint Satisfied"]
        or row["semantic_status_note"]
    ]
    headers = [
        "task_id",
        "Candidate / Reference Ratio",
        "Filtered / Reference Ratio",
        "Predicted / Reference Ratio",
        "Rule-ID Precision",
        "Rule-ID Recall",
        "Exact Match",
        "Semantic Constraint Satisfied",
        "extra_rule_ids",
        "missing_rule_ids",
        "semantic_status_note",
    ]
    main_rows = [public_row(row) for row in rows[:10]]
    llm_enabled = valid_resolver in {"profile_llm_resolver", "profile_auto_resolver"}
    lines = [
        "# Architecture NCARB50 v2 Old-Candidate CTHR Test",
        "",
        f"- Dataset root: `{spec.root}`",
        f"- Tasks: {summary['task count']}",
        f"- Candidate source: `{oldrun.oldmap.OLD_MAPPING_NAME}`",
        f"- Candidate limit: {limit}",
        f"- Candidate-to-valid resolver: `{valid_resolver}`",
        f"- LLM-assisted filtering: {'enabled' if llm_enabled else 'disabled'} in this run.",
        "- Method-visible inputs: algorithm inputs, public scenario models, and rule library.",
        "- Evaluation-only inputs: reference valid rules and semantic constraints.",
        f"- Elapsed seconds: {elapsed_sec:.3f}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Candidate / Reference Ratio | {summary['mean Candidate / Reference Ratio']:.4f} |",
        f"| Filtered / Reference Ratio | {summary['mean Filtered / Reference Ratio']:.4f} |",
        f"| Predicted / Reference Ratio | {summary['mean Predicted / Reference Ratio']:.4f} |",
        f"| Rule-ID Precision | {summary['mean Rule-ID Precision']:.4f} |",
        f"| Rule-ID Recall | {summary['mean Rule-ID Recall']:.4f} |",
        f"| Exact Match | {summary['exact match rate']:.4f} |",
        f"| Semantic Constraint Satisfaction | {summary['semantic constraint satisfaction rate']:.4f} |",
        f"| Formal Feasible | {summary['formal feasible rate']:.4f} |",
        f"| Total Extra Rules | {summary['total extra rules']} |",
        f"| Total Missing Rules | {summary['total missing rules']} |",
        f"| Semantic Unsupported/Error Count | {summary['semantic unsupported/error count']} |",
        "",
        "## First 10 Tasks",
        "",
        full.markdown_table(main_rows, headers),
        "",
        "## Non-Exact or Semantic-Invalid Tasks",
        "",
    ]
    if not failure_rows:
        lines.append("No rule mismatch or semantic-invalid task was observed.")
    else:
        lines.append(full.markdown_table([public_row(row) for row in failure_rows], headers))
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            *[f"- {name}: `{path}`" for name, path in out_paths.items()],
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = dataset_spec(args.dataset_root, out_dir)

    rows, summary = oldrun.run_dataset(
        spec,
        limit=args.limit,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_cache=args.llm_cache,
        valid_resolver=args.valid_resolver,
        aviation_recall_guard=False,
    )

    algorithm_inputs = full.item_map(spec.algorithm_inputs)
    scenario_models = full.item_map(spec.scenario_models)
    references = full.item_map(spec.evaluation_references)
    rule_library = full.read_json(spec.rule_library)
    rule_by_id = {str(rule["rule_id"]): rule for rule in rule_library.get("rules", []) if rule.get("rule_id")}

    enriched_rows: list[dict[str, Any]] = []
    for row in rows:
        task_id = row["task_id"]
        formal, semantic, note = task_semantic_status(
            spec,
            task_id,
            dict(algorithm_inputs[task_id]),
            scenario_models[task_id],
            references[task_id],
            row["predicted_valid_rule_ids"],
            rule_by_id,
        )
        enriched = dict(row)
        enriched["Rule-ID Precision"] = metric_value(enriched["Rule-ID Precision"])
        enriched["Rule-ID Recall"] = metric_value(enriched["Rule-ID Recall"])
        enriched["Formal Feasible"] = formal
        enriched["Semantic Constraint Satisfied"] = semantic
        enriched["semantic_status_note"] = note
        enriched.pop("_candidate_to_valid_resolver", None)
        enriched_rows.append(enriched)

    summary = summarize(enriched_rows) | {
        "dataset": spec.name,
        "dataset_root": str(spec.root),
        "candidate_source": oldrun.oldmap.OLD_MAPPING_NAME,
        "old_mapping_limit": args.limit,
        "candidate_to_valid": args.valid_resolver,
        "llm_assisted_filtering_enabled": args.valid_resolver in {"profile_llm_resolver", "profile_auto_resolver"},
        "llm_provider": args.llm_provider,
        "llm_model": args.llm_model,
        "rule_library": str(spec.rule_library),
        "constraint_templates": str(spec.constraint_templates),
        "audit": {
            "used_expected_candidate_field": False,
            "used_reference_valid_rules_as_method_input": False,
            "used_solver_constraints_as_method_input": False,
            "used_reference_cells_as_method_input": False,
            "used_semantic_validator_as_method_input": False,
        },
    }

    prefix = f"section_6_3_architecture_ncarb50_v2_old_candidate_{args.valid_resolver}_candidate_to_valid"
    full_headers = [
        "Dataset",
        "task_id",
        "target_interaction",
        "candidate_source",
        "candidate_rule_count",
        "filtered_candidate_rule_count",
        "reference_valid_rule_count",
        "predicted_valid_rule_count",
        "Candidate / Reference Ratio",
        "Filtered / Reference Ratio",
        "Predicted / Reference Ratio",
        "Rule-ID Precision",
        "Rule-ID Recall",
        "Exact Match",
        "Formal Feasible",
        "Semantic Constraint Satisfied",
        "candidate_rule_ids_generated",
        "candidate_rule_ids_after_llm_relation_filter",
        "reference_valid_rule_ids",
        "predicted_valid_rule_ids",
        "extra_rule_ids",
        "missing_rule_ids",
        "semantic_status_note",
    ]
    public_rows = [public_row(row) for row in enriched_rows]
    full_csv = out_dir / f"{prefix}_full.csv"
    full_md = out_dir / f"{prefix}_full.md"
    full_json = out_dir / f"{prefix}_full.json"
    summary_json = out_dir / f"{prefix}_summary.json"
    report_md = out_dir / f"{prefix}_report.md"

    full_csv.parent.mkdir(parents=True, exist_ok=True)
    full.write_csv(full_csv, public_rows, full_headers)
    full_md.write_text(full.markdown_table(public_rows, full_headers), encoding="utf-8")
    full.write_json(full_json, public_rows)
    full.write_json(summary_json, summary)

    out_paths = {
        "full_csv": str(full_csv),
        "full_md": str(full_md),
        "full_json": str(full_json),
        "summary_json": str(summary_json),
        "report_md": str(report_md),
    }
    elapsed_sec = time.perf_counter() - start
    report_md.write_text(
        markdown_report(
            spec=spec,
            rows=enriched_rows,
            summary=summary,
            valid_resolver=args.valid_resolver,
            limit=args.limit,
            out_paths=out_paths,
            elapsed_sec=elapsed_sec,
        ),
        encoding="utf-8",
    )

    return summary | {"outputs": out_paths, "elapsed_sec": round(elapsed_sec, 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run old-candidate CTHR recovery on architecture NCARB50 v2.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument(
        "--valid-resolver",
        choices=sorted(oldrun.VALID_RESOLVER_MODES),
        default="profile_resolver",
        help="Use profile_resolver for deterministic no-LLM testing; profile_auto_resolver may require uncached online LLM calls.",
    )
    parser.add_argument("--llm-provider", default="qwen")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument(
        "--llm-cache",
        type=Path,
        default=ROOT / "results" / "llm_grounding_relation_filter_cache.json",
    )
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
