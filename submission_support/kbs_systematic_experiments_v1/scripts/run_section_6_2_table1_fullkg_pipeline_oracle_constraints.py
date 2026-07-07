from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import run_section_6_2_table1_fullkg_pipeline as full


RESULTS_DIR = full.RESULTS_DIR


def oracle_rule_constraint_bank(feasible: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    bank: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for constraint in full.base.reference_constraints(feasible):
        if constraint.get("source_type") != "rule_library":
            continue
        source_id = str(constraint.get("source_id", ""))
        if source_id:
            bank[source_id].append(dict(constraint))
    return dict(bank)


def oracle_method_constraints(
    query: dict[str, Any],
    selected_rule_ids: list[str],
    rule_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    _ = rule_by_id
    constraints = list(query.get("solver_constraints", []))
    seen = {str(constraint.get("constraint_id")) for constraint in constraints}
    oracle_bank = query.get("_oracle_rule_constraints_by_id", {})
    for rule_id in selected_rule_ids:
        for constraint in oracle_bank.get(str(rule_id), []):
            key = str(constraint.get("constraint_id", f"oracle_{rule_id}_{len(seen)}"))
            if key in seen:
                continue
            constraints.append(dict(constraint))
            seen.add(key)
    return constraints


def oracle_run_smt(
    query: dict[str, Any],
    candidate_ids: list[str],
    rule_library: dict[str, Any],
    rule_by_id: dict[str, dict[str, Any]],
    *,
    native: bool,
) -> full.MethodResult:
    if not candidate_ids:
        return full.MethodResult(False, [], None, None, "no_grounded_candidates")
    lib = full.filtered_library(rule_library, candidate_ids, native=native)
    try:
        formula = full.build_smt_formula(
            lib,
            query,
            candidate_rule_ids=candidate_ids,
            include_visible_task_constraints=True,
        )
        result = full.optimize_with_z3(formula, query, timeout_ms=10000)
        if result.status != "sat":
            return full.MethodResult(
                False,
                sorted(result.selected_rule_ids),
                None,
                None,
                f"smt_{result.status}:{result.error or ''}".strip(":"),
            )
        selected = sorted(result.selected_rule_ids)
        method_name = "Native SMT + Z3" if native else "CTHR-style SMT + Z3"
        x, formal = full.solve_with_default(method_name, query, selected, rule_by_id)
        return full.MethodResult(True, selected, x, formal)
    except Exception as exc:  # noqa: BLE001
        return full.MethodResult(False, [], None, None, f"smt_error:{exc}")


def evaluate_dataset(spec: full.DatasetSpec) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    algorithm_inputs = full.item_map(spec.algorithm_inputs)
    scenario_models = full.item_map(spec.scenario_models)
    references = full.item_map(spec.evaluation_references)
    grounding_results = full.grounding_result_map(spec.grounding_full)
    rule_library = full.read_json(spec.rule_library)
    rule_by_id = {str(rule["rule_id"]): rule for rule in rule_library.get("rules", []) if rule.get("rule_id")}
    if set(algorithm_inputs) != set(scenario_models) or set(algorithm_inputs) != set(references):
        raise ValueError(f"{spec.name} layer IDs do not match")
    if set(algorithm_inputs) != set(grounding_results):
        raise ValueError(f"{spec.name} grounding result IDs do not match")

    rows: list[dict[str, Any]] = []
    grounding_audit: dict[str, Any] = {}
    oracle_constraint_count = 0
    for task_id in sorted(algorithm_inputs):
        grounding_task = dict(algorithm_inputs[task_id])
        query = full.prepare_query(grounding_task, scenario_models[task_id])
        reference = references[task_id]
        feasible = full.reference_feasible(reference, query)
        query["_oracle_rule_constraints_by_id"] = oracle_rule_constraint_bank(feasible)
        oracle_constraint_count += sum(len(items) for items in query["_oracle_rule_constraints_by_id"].values())

        reference_ids = full.reference_rule_ids(reference)
        grounding_row = grounding_results[task_id]
        candidate_ids = full.ids_from_grounding(grounding_row, "candidate_rule_ids_generated")
        cthr_valid_ids = full.ids_from_grounding(grounding_row, "predicted_valid_rule_ids")
        candidate_rules = [rule_by_id[rule_id] for rule_id in candidate_ids if rule_id in rule_by_id]
        grounding_audit[task_id] = {
            "candidate_rule_ids": candidate_ids,
            "candidate_rule_count": len(candidate_ids),
            "cthr_predicted_valid_rule_ids": cthr_valid_ids,
            "grounding_result_exact_match": bool(grounding_row.get("Exact Match")),
        }
        for method, method_type in full.METHOD_SPECS:
            start = time.perf_counter()
            result = full.run_method(
                spec,
                method,
                query,
                grounding_task,
                candidate_rules,
                rule_library,
                rule_by_id,
                cthr_valid_ids,
            )
            elapsed = (time.perf_counter() - start) * 1000.0
            predicted = sorted(result.predicted_rule_ids) if result.supported else []
            precision = full.base.method_rule_precision(predicted, reference_ids)
            recall = full.base.method_rule_recall(predicted, reference_ids)
            if precision is None:
                precision = 0.0
            if recall is None:
                recall = 0.0
            sem_ok = full.semantic_valid(feasible, result.optimized_x, predicted, reference_ids) if result.supported else False
            formal_ok = bool(result.formal_feasible) if result.supported else False
            rows.append(
                {
                    "Dataset": spec.name,
                    "task_id": task_id,
                    "target_interaction": full.target_interaction(reference),
                    "Method": method,
                    "Method type": method_type,
                    "grounded_candidate_count": len(candidate_ids),
                    "predicted_rule_ids": predicted,
                    "reference_rule_ids": reference_ids,
                    "rule_precision": precision,
                    "rule_recall": recall,
                    "formal_feasible": formal_ok,
                    "semantic_valid": sem_ok,
                    "false_accept": bool(formal_ok and not sem_ok),
                    "invalid_case": bool(not sem_ok),
                    "unsupported_reason": "" if result.supported else result.unsupported_reason,
                    "runtime_ms": round(elapsed, 3),
                }
            )
    summary = {
        "dataset": spec.name,
        "domain": spec.domain,
        "root": str(spec.root),
        "tasks": len(algorithm_inputs),
        "rule_library": str(spec.rule_library),
        "grounding_result": str(spec.grounding_full),
        "rule_library_rules": len(rule_library.get("rules", [])),
        "oracle_rule_constraints": oracle_constraint_count,
        "grounding": {
            "source": "precomputed domain-specific Section 6.3 end-to-end grounding output",
            "mean_candidate_count": sum(item["candidate_rule_count"] for item in grounding_audit.values())
            / max(1, len(grounding_audit)),
            "cthr_exact_match_rate": sum(1 for item in grounding_audit.values() if item["grounding_result_exact_match"])
            / max(1, len(grounding_audit)),
        },
    }
    return rows, summary


def report(aggregate_rows: list[dict[str, Any]], per_task_rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    headers = [
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
    unsupported = full.unsupported_summary(per_task_rows)
    unsupported_lines = []
    for key, counts in sorted(unsupported.items()):
        detail = "; ".join(f"{reason}: {count}" for reason, count in sorted(counts.items()))
        unsupported_lines.append(f"- {key}: {detail}")
    if not unsupported_lines:
        unsupported_lines = ["- none"]

    return "\n".join(
        [
            "# Section 6.2 Table 1 Oracle Constraint-Bank Sanity Check",
            "",
            "## Scope",
            "",
            "- Diagnostic upper bound only; not a leakage-safe paper result.",
            "- Rule selection uses the same latest Section 6.3 grounding outputs as the main full-KG table.",
            "- Feasible-region construction uses evaluation-reference executable rule constraints as an oracle constraint bank, filtered by each method's selected rule IDs.",
            "- Public scenario-model constraints are still read from the public scenario model layer.",
            "",
            "## Main Result",
            "",
            full.markdown_table(aggregate_rows, headers),
            "",
            "## Unsupported / N/A",
            "",
            *unsupported_lines,
            "",
            "## Summary",
            "",
            "```json",
            json.dumps(summary, ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )


def main() -> None:
    original_method_constraints = full.method_constraints
    original_run_smt = full.run_smt
    full.method_constraints = oracle_method_constraints
    full.run_smt = oracle_run_smt
    try:
        per_task_rows: list[dict[str, Any]] = []
        dataset_summaries: dict[str, Any] = {}
        for spec in full.DATASETS:
            rows, dataset_summary = evaluate_dataset(spec)
            per_task_rows.extend(rows)
            dataset_summaries[spec.name] = dataset_summary

        aggregate_rows = []
        for spec in full.DATASETS:
            for method, method_type in full.METHOD_SPECS:
                aggregate_rows.append(full.aggregate(per_task_rows, spec.name, method, method_type))

        per_task_headers = [
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
        aggregate_headers = [
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
        outputs = {
            "per_task_csv": RESULTS_DIR / "section_6_2_table1_fullkg_pipeline_oracle_constraints_per_task.csv",
            "overall_csv": RESULTS_DIR / "section_6_2_table1_fullkg_pipeline_oracle_constraints_overall.csv",
            "overall_md": RESULTS_DIR / "section_6_2_table1_fullkg_pipeline_oracle_constraints_overall.md",
            "overall_json": RESULTS_DIR / "section_6_2_table1_fullkg_pipeline_oracle_constraints_overall.json",
            "report_md": RESULTS_DIR / "section_6_2_table1_fullkg_pipeline_oracle_constraints_report.md",
        }
        full.write_csv(outputs["per_task_csv"], per_task_rows, per_task_headers)
        full.write_csv(outputs["overall_csv"], aggregate_rows, aggregate_headers)
        outputs["overall_md"].write_text(full.markdown_table(aggregate_rows, aggregate_headers), encoding="utf-8")

        summary = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "datasets": dataset_summaries,
            "methods": [{"Method": method, "Method type": method_type} for method, method_type in full.METHOD_SPECS],
            "input_restrictions": {
                "diagnostic_only": True,
                "evaluation_references_as_oracle_constraint_bank": True,
                "cthr_compiled_cells_as_method_input": False,
            },
            "metric_scope": "All percentages use total dataset task count as denominator. Unsupported task-method pairs count as non-success for CSR-style metrics.",
            "unsupported": full.unsupported_summary(per_task_rows),
            "outputs": {key: str(value) for key, value in outputs.items()},
            "aggregate_rows": aggregate_rows,
        }
        full.write_json(outputs["overall_json"], summary)
        outputs["report_md"].write_text(report(aggregate_rows, per_task_rows, summary), encoding="utf-8")
        print(json.dumps({"outputs": summary["outputs"], "aggregate_rows": aggregate_rows}, ensure_ascii=False, indent=2))
    finally:
        full.method_constraints = original_method_constraints
        full.run_smt = original_run_smt


if __name__ == "__main__":
    main()
