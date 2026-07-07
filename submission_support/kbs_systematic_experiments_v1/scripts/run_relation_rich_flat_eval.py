from __future__ import annotations

import csv
import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
SCRIPTS_DIR = ROOT / "scripts"

sys.path.insert(0, str(SCRIPTS_DIR))

import run_section_6_2_table1_fullkg_pipeline as full  # noqa: E402


METHODS = ("Flat baseline", "CTHR default")


GROUNDING_CONFIGS = {
    "relation_rich": {
        "label": "relation_rich candidates",
        "prefix": "relation_rich_method_eval",
        "files": {
            "Aviation": RESULTS_DIR / "section_6_3_aviation_relation_rich_candidate_to_valid_full.json",
            "Architecture": RESULTS_DIR / "section_6_3_architecture_relation_rich_candidate_to_valid_full.json",
        },
    },
    "relation_stress_llm_filter": {
        "label": "relation_stress + LLM relation-filter candidates",
        "prefix": "relation_stress_llm_filter_method_eval",
        "files": {
            "Aviation": RESULTS_DIR / "section_6_3_aviation_relation_stress_llm_filter_candidate_to_valid_full.json",
            "Architecture": RESULTS_DIR / "section_6_3_architecture_relation_stress_llm_filter_candidate_to_valid_full.json",
        },
    },
    "old_candidate_llm_filter": {
        "label": "old broad candidates",
        "prefix": "old_candidate_llm_filter_method_eval",
        "files": {
            "Aviation": RESULTS_DIR / "section_6_3_aviation_old_candidate_llm_filter_candidate_to_valid_full.json",
            "Architecture": RESULTS_DIR / "section_6_3_architecture_old_candidate_llm_filter_candidate_to_valid_full.json",
        },
    },
    "old_candidate_strict_relation_filter": {
        "label": "old broad candidates + strict six-relation filter",
        "prefix": "old_candidate_strict_relation_filter_method_eval",
        "files": {
            "Aviation": RESULTS_DIR
            / "section_6_3_aviation_old_candidate_strict_relation_filter_candidate_to_valid_full.json",
            "Architecture": RESULTS_DIR
            / "section_6_3_architecture_old_candidate_strict_relation_filter_candidate_to_valid_full.json",
        },
    },
    "old_candidate_profile_auto_resolver": {
        "label": "old broad candidates",
        "cthr_label": "old broad candidates + candidate-constrained profile resolver",
        "prefix": "old_candidate_profile_auto_resolver_method_eval",
        "files": {
            "Aviation": RESULTS_DIR
            / "section_6_3_aviation_old_candidate_profile_auto_resolver_candidate_to_valid_full.json",
            "Architecture": RESULTS_DIR
            / "section_6_3_architecture_old_candidate_profile_auto_resolver_candidate_to_valid_full.json",
        },
    },
    "old_candidate_recall_guard_profile_auto_resolver": {
        "label": "old broad candidates + aviation recall guard",
        "cthr_label": "old broad candidates + aviation recall guard + candidate-constrained profile resolver",
        "prefix": "old_candidate_recall_guard_profile_auto_resolver_method_eval",
        "files": {
            "Aviation": RESULTS_DIR
            / "section_6_3_aviation_old_candidate_recall_guard_profile_auto_resolver_candidate_to_valid_full.json",
            "Architecture": RESULTS_DIR
            / "section_6_3_architecture_old_candidate_profile_auto_resolver_candidate_to_valid_full.json",
        },
    },
    "old_candidate_profile_resolver": {
        "label": "old broad candidates",
        "cthr_label": "old broad candidates + candidate-constrained profile resolver",
        "prefix": "old_candidate_profile_resolver_method_eval",
        "files": {
            "Aviation": RESULTS_DIR / "section_6_3_aviation_old_candidate_profile_resolver_candidate_to_valid_full.json",
            "Architecture": RESULTS_DIR
            / "section_6_3_architecture_old_candidate_profile_resolver_candidate_to_valid_full.json",
        },
    },
    "old_candidate_profile_llm_resolver": {
        "label": "old broad candidates",
        "cthr_label": "old broad candidates + candidate-constrained profile+LLM resolver",
        "prefix": "old_candidate_profile_llm_resolver_method_eval",
        "files": {
            "Aviation": RESULTS_DIR
            / "section_6_3_aviation_old_candidate_profile_llm_resolver_candidate_to_valid_full.json",
            "Architecture": RESULTS_DIR
            / "section_6_3_architecture_old_candidate_profile_llm_resolver_candidate_to_valid_full.json",
        },
    },
}


def csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(list(value) if isinstance(value, set) else value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
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
    return "\n".join(lines)


def output_paths(mode: str) -> dict[str, Path]:
    prefix = str(GROUNDING_CONFIGS[mode]["prefix"])
    return {
        "per_task": RESULTS_DIR / f"{prefix}_per_task.csv",
        "overall_csv": RESULTS_DIR / f"{prefix}_overall.csv",
        "overall_json": RESULTS_DIR / f"{prefix}_overall.json",
        "overall_md": RESULTS_DIR / f"{prefix}_overall.md",
    }


def grounding_specs(mode: str) -> list[full.DatasetSpec]:
    specs: list[full.DatasetSpec] = []
    for spec in full.DATASETS:
        grounding = GROUNDING_CONFIGS[mode]["files"][spec.name]
        specs.append(replace(spec, grounding_full=grounding))
    return specs


def run_dataset(spec: full.DatasetSpec, mode: str) -> list[dict[str, Any]]:
    algorithm_inputs = full.item_map(spec.algorithm_inputs)
    scenario_models = full.item_map(spec.scenario_models)
    references = full.item_map(spec.evaluation_references)
    grounding_results = full.grounding_result_map(spec.grounding_full)
    templates_by_rule = full.constraint_template_map(spec.constraint_templates)
    rule_library = full.read_json(spec.rule_library)
    rule_by_id = {str(rule["rule_id"]): rule for rule in rule_library.get("rules", []) if rule.get("rule_id")}

    if set(algorithm_inputs) != set(scenario_models) or set(algorithm_inputs) != set(references):
        raise ValueError(f"{spec.name} layer IDs do not match")
    if set(algorithm_inputs) != set(grounding_results):
        raise ValueError(f"{spec.name} relation-rich grounding result IDs do not match")

    rows: list[dict[str, Any]] = []
    for task_id in sorted(algorithm_inputs):
        grounding_task = dict(algorithm_inputs[task_id])
        query = full.prepare_query(grounding_task, scenario_models[task_id])
        query["_compiled_rule_constraint_templates_by_id"] = templates_by_rule
        reference = references[task_id]
        feasible = full.reference_feasible(reference, query)
        reference_ids = full.reference_rule_ids(reference)
        grounding_row = grounding_results[task_id]
        candidate_ids = full.ids_from_grounding(grounding_row, "candidate_rule_ids_generated")

        candidate_rules = [rule_by_id[rule_id] for rule_id in candidate_ids if rule_id in rule_by_id]
        cthr_valid_ids = full.ids_from_grounding(grounding_row, "predicted_valid_rule_ids")

        for method in METHODS:
            start = time.perf_counter()
            if method == "Flat baseline":
                result = full.run_flat(query, candidate_ids, rule_by_id)
            elif method == "CTHR default":
                result = full.run_cthr_default(spec, query, grounding_task, candidate_rules, rule_by_id, cthr_valid_ids)
            else:
                raise ValueError(method)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            predicted = sorted(result.predicted_rule_ids) if result.supported else []
            precision = full.base.method_rule_precision(predicted, reference_ids) or 0.0
            recall = full.base.method_rule_recall(predicted, reference_ids) or 0.0
            formal_ok = bool(result.formal_feasible) if result.supported else False
            sem_ok = full.semantic_valid(feasible, result.optimized_x, predicted, reference_ids) if result.supported else False

            rows.append(
                {
                    "Dataset": spec.name,
                    "task_id": task_id,
                    "target_interaction": full.target_interaction(reference),
                    "Method": method,
                    "grounding_mode": mode,
                    "candidate_rule_count": len(candidate_ids),
                    "predicted_rule_count": len(predicted),
                    "reference_rule_count": len(reference_ids),
                    "predicted_rule_ids": predicted,
                    "reference_rule_ids": reference_ids,
                    "rule_precision": precision,
                    "rule_recall": recall,
                    "formal_feasible": formal_ok,
                    "semantic_valid": sem_ok,
                    "false_accept": bool(formal_ok and not sem_ok),
                    "invalid_case": bool(not sem_ok),
                    "unsupported_reason": "" if result.supported else result.unsupported_reason,
                    "runtime_ms": round(elapsed_ms, 3),
                }
            )
    return rows


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def aggregate(rows: list[dict[str, Any]], dataset: str, method: str, mode: str) -> dict[str, Any]:
    subset = [row for row in rows if row["Dataset"] == dataset and row["Method"] == method]
    total = len(subset)
    invalid = sum(1 for row in subset if row["invalid_case"])
    unsupported = sum(1 for row in subset if row["unsupported_reason"])
    invalid_text = f"{invalid}/{total} ({100.0 * invalid / total:.1f}%)"
    if unsupported:
        invalid_text += f" ({unsupported} unsupported)"
    return {
        "Dataset": dataset,
        "Method": f"{method} + {GROUNDING_CONFIGS[mode].get('cthr_label' if method == 'CTHR default' else 'label')}",
        "Rule Precision": pct(sum(float(row["rule_precision"]) for row in subset) / total),
        "Rule Recall": pct(sum(float(row["rule_recall"]) for row in subset) / total),
        "Sem-CSR": pct(sum(1 for row in subset if row["semantic_valid"]) / total),
        "Invalid": invalid_text,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Flat and CTHR methods with relation-aware grounding outputs.")
    parser.add_argument("--grounding-mode", choices=sorted(GROUNDING_CONFIGS), default="relation_rich")
    args = parser.parse_args()
    mode = args.grounding_mode
    paths = output_paths(mode)
    per_task_rows: list[dict[str, Any]] = []
    for spec in grounding_specs(mode):
        per_task_rows.extend(run_dataset(spec, mode))

    overall = [aggregate(per_task_rows, spec.name, method, mode) for spec in grounding_specs(mode) for method in METHODS]
    overall_headers = ["Dataset", "Method", "Rule Precision", "Rule Recall", "Sem-CSR", "Invalid"]
    per_task_headers = [
        "Dataset",
        "task_id",
        "target_interaction",
        "Method",
        "grounding_mode",
        "candidate_rule_count",
        "predicted_rule_count",
        "reference_rule_count",
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

    write_csv(paths["per_task"], per_task_rows, per_task_headers)
    write_csv(paths["overall_csv"], overall, overall_headers)
    paths["overall_md"].write_text(markdown_table(overall, overall_headers), encoding="utf-8")
    full.write_json(
        paths["overall_json"],
        {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "grounding_mode": mode,
            "methods": list(METHODS),
            "metric_note": (
                "Flat predicts all candidate_rule_ids from the selected grounding file as selected rules. "
                "CTHR default uses the relation-aware candidate-to-valid recovery output from the same grounding file. "
                "For old_candidate_llm_filter, Flat uses raw old candidates while CTHR uses "
                "LLM relation-filtered candidates followed by the strict CTHR resolver. "
                "Sem-CSR evaluates the optimized solution against source reference constraints."
            ),
            "grounding_files": {
                name: str(path) for name, path in GROUNDING_CONFIGS[mode]["files"].items()
            },
            "overall": overall,
            "outputs": {
                "per_task_csv": str(paths["per_task"]),
                "overall_csv": str(paths["overall_csv"]),
                "overall_md": str(paths["overall_md"]),
                "overall_json": str(paths["overall_json"]),
            },
        },
    )
    print(json.dumps({"overall": overall}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
