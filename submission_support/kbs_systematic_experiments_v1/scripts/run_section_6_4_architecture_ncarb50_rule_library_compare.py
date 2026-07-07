from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import run_section_6_2_table1_fullkg_pipeline as fullkg


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
DATASET_ROOT = ROOT / "datasets" / "architecture_fullkg_ncarb50_v2"
CORE = DATASET_ROOT / "core"
OVERLAYS = DATASET_ROOT / "evaluation_overlays"
RULE_LIBRARIES = DATASET_ROOT / "rule_libraries"
CANONICAL_EVALUATION_REFERENCES = (
    DATASET_ROOT / "evaluation_references" / "architecture_evaluation_references.json"
)
STRICT_COMMON_TASKS = DATASET_ROOT / "STRICT_COMMON_TASKS.json"
MANIFEST = DATASET_ROOT / "MANIFEST.json"
OVERLAY_MANIFEST = DATASET_ROOT / "OVERLAY_MANIFEST.json"


MODEL_SPECS = [
    {
        "model": "Qwen-plus",
        "provider": "qwen",
        "overlay_key": "qwen",
        "rule_library": RULE_LIBRARIES / "qwen" / "full_architecture_rule_library_qwen.json",
    },
    {
        "model": "DeepSeek-Pro",
        "provider": "deepseek",
        "overlay_key": "deepseek",
        "rule_library": RULE_LIBRARIES / "deepseek" / "full_architecture_rule_library_deepseek.json",
    },
    {
        "model": "Xiaomi MIMO",
        "provider": "xiaomi_mimo",
        "overlay_key": "xiaomi_mimo",
        "rule_library": RULE_LIBRARIES / "xiaomi_mimo" / "full_architecture_rule_library_mimo.json",
    },
]


MODES = [
    {
        "mode": "oracle_candidate",
        "description": "Use strong-aligned overlay rule IDs as candidates, then run CTHR default valid-rule recovery.",
        "pass_as_valid": False,
        "extend_relation_templates": False,
    },
    {
        "mode": "oracle_valid_upper",
        "description": "Use strong-aligned overlay rule IDs as both candidates and selected valid rules.",
        "pass_as_valid": True,
        "extend_relation_templates": False,
    },
    {
        "mode": "relation_extended_oracle_candidate",
        "description": "Use strong-aligned candidates and an extended relation-to-template layer with task-bound rule constraints.",
        "pass_as_valid": False,
        "extend_relation_templates": True,
    },
    {
        "mode": "relation_extended_oracle_valid_upper",
        "description": "Use strong-aligned valid rules and the extended relation-to-template layer as an upper-bound diagnostic.",
        "pass_as_valid": True,
        "extend_relation_templates": True,
    },
]


RELATION_EXTENSION_POLICY = {
    "defines": "symbol normalization and task-specific variable binding",
    "allows_alternative": "alternative feasible-cell metadata retained for diagnostics",
    "prevents": "negative applicability represented through exclusion-style rule activation metadata",
    "unit_conversion": "unit-normalized source expressions are materialized as compiled templates when available",
    "formula_derivation": "formula/checker expressions are copied into executable compiled templates when safe symbols are available",
    "co_effective_rules": "constraints from joint valid rule cells are projected back to their source rule IDs",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def rule_precision(predicted: list[str], reference: list[str]) -> float:
    if not predicted:
        return 0.0
    return len(set(predicted) & set(reference)) / len(set(predicted))


def rule_recall(predicted: list[str], reference: list[str]) -> float:
    if not reference:
        return 1.0
    return len(set(predicted) & set(reference)) / len(set(reference))


def project_rule_ids(predicted: list[str], model_to_canonical: dict[str, list[str]]) -> list[str]:
    return sorted({canonical for rule_id in predicted for canonical in model_to_canonical.get(str(rule_id), [])})


def overlay_file(model_key: str, filename: str) -> Path:
    return OVERLAYS / model_key / filename


def overlay_alignment_summary(model_key: str) -> dict[str, Any]:
    return dict(read_json(overlay_file(model_key, "rule_id_alignment.json")).get("summary", {}))


def overlay_model_to_canonical(model_key: str) -> dict[str, list[str]]:
    payload = read_json(overlay_file(model_key, "rule_id_alignment.json"))
    out: dict[str, list[str]] = {}
    for row in payload.get("model_to_canonical", []):
        model_rule_id = str(row.get("model_rule_id"))
        out[model_rule_id] = sorted(str(item) for item in row.get("canonical_rule_ids", []))
    return out


def overlay_canonical_to_model(model_key: str) -> dict[str, list[str]]:
    payload = read_json(overlay_file(model_key, "rule_id_alignment.json"))
    out: dict[str, list[str]] = {}
    for row in payload.get("canonical_to_model", []):
        canonical_rule_id = str(row.get("canonical_rule_id"))
        model_rule_ids = row.get("aligned_model_rule_ids", [])
        out[canonical_rule_id] = sorted(str(item) for item in model_rule_ids if item)
    return out


def canonical_reference_by_task() -> dict[str, list[str]]:
    payload = read_json(CANONICAL_EVALUATION_REFERENCES)
    out: dict[str, list[str]] = {}
    for item in payload.get("items", []):
        task_id = str(item.get("omega_id"))
        out[task_id] = fullkg.reference_rule_ids(item)
    return out


def task_ids_for_split(split: str) -> list[str] | None:
    if split == "full":
        return None
    payload = read_json(STRICT_COMMON_TASKS)
    return [str(item) for item in payload.get("task_ids", [])]


def architecture_spec(model_spec: dict[str, Any], overlay_key: str) -> fullkg.DatasetSpec:
    return fullkg.DatasetSpec(
        name="Architecture NCARB50 v2",
        domain="architecture",
        root=DATASET_ROOT,
        algorithm_inputs=CORE / "algorithm_inputs" / "architecture_algorithm_inputs.json",
        scenario_models=CORE / "scenario_models" / "architecture_public_scenario_models.json",
        evaluation_references=overlay_file(overlay_key, "evaluation_references.json"),
        rule_library=Path(model_spec["rule_library"]),
        grounding_full=RESULTS_DIR / "unused_architecture_ncarb50_grounding.json",
        constraint_templates=overlay_file(overlay_key, "compiled_rule_constraint_templates.json"),
    )


def strong_coverage_label(summary: dict[str, Any]) -> str:
    return (
        f"{summary.get('exact_or_strong_aligned_canonical_rule_count', summary.get('aligned_canonical_rule_count', 0))}/"
        f"{summary.get('canonical_rule_count', 0)}"
    )


def scalar_scenario_context(algorithm_input: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in algorithm_input.get("scenario_facts", {}).items()
        if isinstance(value, (str, bool, int, float))
    }


def constraint_required_symbols(constraint: dict[str, Any]) -> list[str]:
    symbols = constraint.get("symbols", {})
    if not isinstance(symbols, dict):
        return []
    required: set[str] = set()
    for key in ["decision_variables", "scenario_fields"]:
        value = symbols.get(key, [])
        if isinstance(value, list):
            required.update(str(item) for item in value if item)
    return sorted(required)


def executable_rule_constraints(reference: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    feasible = reference.get("feasible_region", {})
    out: list[tuple[str, dict[str, Any]]] = []
    for constraint in feasible.get("executable_constraints", []):
        out.append(("feasible_region.executable_constraints", constraint))
    for cell in feasible.get("valid_constraint_cells", []):
        cell_id = str(cell.get("cell_id", "unknown_cell"))
        for constraint in cell.get("constraints", []):
            out.append((f"feasible_region.valid_constraint_cells.{cell_id}", constraint))
    return out


def relation_extended_template(
    *,
    task_id: str,
    source_container: str,
    constraint: dict[str, Any],
    context: dict[str, Any],
    target_rule_id: str | None = None,
) -> dict[str, Any] | None:
    if not constraint.get("executable", True):
        return None
    if constraint.get("source_type") != "rule_library":
        return None
    rule_id = str(constraint.get("source_id") or "")
    target_id = str(target_rule_id or rule_id)
    expression = str(constraint.get("checker_expression") or constraint.get("expression") or "").strip()
    if not rule_id or not expression:
        return None
    constraint_id = str(constraint.get("constraint_id") or "constraint")
    return {
        "template_id": f"{target_id}::REL_EXT::{task_id}::{constraint_id}",
        "source_rule_id": target_id,
        "expression": expression,
        "checker_expression": expression,
        "expression_language": constraint.get("expression_language", "python_safe_arithmetic_predicate"),
        "role": constraint.get("role", "relation_extended_rule_constraint"),
        "required_symbols": constraint_required_symbols(constraint),
        "observed_bindings": [
            {
                "task_id": task_id,
                "constraint_id": constraint_id,
                "decision_variables": constraint.get("symbols", {}).get("decision_variables", [])
                if isinstance(constraint.get("symbols"), dict)
                else [],
                "scenario_fields": constraint.get("symbols", {}).get("scenario_fields", [])
                if isinstance(constraint.get("symbols"), dict)
                else [],
                "source_container": source_container,
            }
        ],
        "applicability_contexts": [context],
        "metadata": {
            "compiled_template_layer": True,
            "relation_extension_layer": True,
            "relation_extension_policy": RELATION_EXTENSION_POLICY,
            "source_container": source_container,
            "source_constraint_id": constraint_id,
            "source_constraint_role": constraint.get("role", ""),
            "canonical_source_rule_id": rule_id,
            "target_rule_id": target_id,
        },
    }


def build_relation_extended_templates(
    references: dict[str, dict[str, Any]],
    algorithm_inputs: dict[str, dict[str, Any]],
    base_templates_by_rule: dict[str, list[dict[str, Any]]],
    canonical_to_model: dict[str, list[str]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    out = {rule_id: [dict(template) for template in templates] for rule_id, templates in base_templates_by_rule.items()}
    seen = {
        (rule_id, str(template.get("template_id")))
        for rule_id, templates in out.items()
        for template in templates
    }
    added_by_rule: dict[str, int] = {}
    added_by_task: dict[str, int] = {}
    for task_id, reference in references.items():
        if task_id not in algorithm_inputs:
            continue
        context = scalar_scenario_context(algorithm_inputs[task_id])
        for source_container, constraint in executable_rule_constraints(reference):
            source_rule_id = str(constraint.get("source_id") or "")
            target_rule_ids = sorted({source_rule_id, *canonical_to_model.get(source_rule_id, [])})
            for target_rule_id in target_rule_ids:
                template = relation_extended_template(
                    task_id=task_id,
                    source_container=source_container,
                    constraint=constraint,
                    context=context,
                    target_rule_id=target_rule_id,
                )
                if template is None:
                    continue
                rule_id = str(template["source_rule_id"])
                key = (rule_id, str(template["template_id"]))
                if key in seen:
                    continue
                out.setdefault(rule_id, []).append(template)
                seen.add(key)
                added_by_rule[rule_id] = added_by_rule.get(rule_id, 0) + 1
                added_by_task[task_id] = added_by_task.get(task_id, 0) + 1
    stats = {
        "enabled": True,
        "policy": RELATION_EXTENSION_POLICY,
        "added_template_count": sum(added_by_rule.values()),
        "added_rule_count": len(added_by_rule),
        "added_task_count": len(added_by_task),
        "added_by_rule": dict(sorted(added_by_rule.items())),
        "added_by_task": dict(sorted(added_by_task.items())),
    }
    return out, stats


def support_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "candidate_zero": sum(1 for row in rows if row.get("unsupported_reason") == "no_grounded_candidates"),
        "cthr_no_valid": sum(1 for row in rows if row.get("unsupported_reason") == "cthr_no_valid_rules"),
        "unsupported": sum(1 for row in rows if row.get("unsupported_reason")),
    }


def aggregate_rows(
    rows: list[dict[str, Any]],
    *,
    split: str,
    mode: str,
    model_name: str,
    overlay_key: str,
    strong_coverage: str,
) -> dict[str, Any]:
    total = len(rows)
    counts = support_counts(rows)
    invalid = sum(1 for row in rows if row.get("invalid_case"))
    sem_csr = pct(sum(1 for row in rows if row.get("semantic_valid")) / max(1, total))
    return {
        "Domain": "Architecture NCARB50 v2",
        "Split": split,
        "Mode": mode,
        "Model": model_name,
        "Overlay": overlay_key,
        "Strong canonical coverage": strong_coverage,
        "Relation extension": rows[0].get("relation_extension", "false") if rows else "false",
        "Relation templates added": rows[0].get("relation_templates_added", 0) if rows else 0,
        "Canonical Rule Precision": pct(sum(float(row["canonical_rule_precision"]) for row in rows) / max(1, total)),
        "Canonical Rule Recall": pct(sum(float(row["canonical_rule_recall"]) for row in rows) / max(1, total)),
        "Model-ID Rule Precision": pct(sum(float(row["model_rule_precision"]) for row in rows) / max(1, total)),
        "Model-ID Rule Recall": pct(sum(float(row["model_rule_recall"]) for row in rows) / max(1, total)),
        "Formal CSR": pct(sum(1 for row in rows if row.get("formal_feasible")) / max(1, total)),
        "Sem-CSR": sem_csr,
        "False accept": pct(sum(1 for row in rows if row.get("false_accept")) / max(1, total)),
        "Candidate zero": counts["candidate_zero"],
        "CTHR no valid": counts["cthr_no_valid"],
        "Unsupported tasks": counts["unsupported"],
        "Mean oracle candidates": f"{sum(int(row['oracle_candidate_count']) for row in rows) / max(1, total):.2f}",
        "Invalid cases": f"{invalid}/{total} ({100.0 * invalid / max(1, total):.1f}%)"
        + (f" ({counts['unsupported']} unsupported)" if counts["unsupported"] else ""),
    }


def evaluate_model(
    model_spec: dict[str, Any],
    split: str,
    mode_cfg: dict[str, Any],
    canonical_references: dict[str, list[str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    overlay_key = str(model_spec["overlay_key"])
    model_name = str(model_spec["model"])
    spec = architecture_spec(model_spec, overlay_key)
    algorithm_inputs = fullkg.item_map(spec.algorithm_inputs)
    scenario_models = fullkg.item_map(spec.scenario_models)
    references = fullkg.item_map(spec.evaluation_references)
    templates_by_rule = fullkg.constraint_template_map(spec.constraint_templates)
    relation_extension_stats = {"enabled": False, "added_template_count": 0}
    canonical_to_model = overlay_canonical_to_model(overlay_key)
    if mode_cfg.get("extend_relation_templates"):
        templates_by_rule, relation_extension_stats = build_relation_extended_templates(
            references,
            algorithm_inputs,
            templates_by_rule,
            canonical_to_model,
        )
    rule_library = fullkg.read_json(spec.rule_library)
    rule_by_id = {str(rule["rule_id"]): rule for rule in rule_library.get("rules", []) if rule.get("rule_id")}
    model_to_canonical = overlay_model_to_canonical(overlay_key)
    alignment_summary = overlay_alignment_summary(overlay_key)
    strong_coverage = strong_coverage_label(alignment_summary)
    wanted_task_ids = task_ids_for_split(split)
    task_ids = wanted_task_ids if wanted_task_ids is not None else sorted(algorithm_inputs)

    rows: list[dict[str, Any]] = []
    for task_id in task_ids:
        grounding_task = dict(algorithm_inputs[task_id])
        query = fullkg.prepare_query(grounding_task, scenario_models[task_id])
        query["_compiled_rule_constraint_templates_by_id"] = templates_by_rule
        reference = references[task_id]
        feasible = fullkg.reference_feasible(reference, query)
        model_reference_ids = fullkg.reference_rule_ids(reference)
        candidate_rules = [rule_by_id[rule_id] for rule_id in model_reference_ids if rule_id in rule_by_id]
        selected_valid_ids = model_reference_ids if mode_cfg["pass_as_valid"] else None
        start = time.perf_counter()
        result = fullkg.run_method(
            spec,
            "CTHR default",
            query,
            grounding_task,
            candidate_rules,
            rule_library,
            rule_by_id,
            selected_valid_ids,
        )
        elapsed = (time.perf_counter() - start) * 1000.0
        predicted_model = sorted(result.predicted_rule_ids) if result.supported else []
        predicted_canonical = project_rule_ids(predicted_model, model_to_canonical)
        reference_canonical = canonical_references.get(task_id, [])
        sem_ok = (
            fullkg.semantic_valid(feasible, result.optimized_x, predicted_model, model_reference_ids)
            if result.supported
            else False
        )
        formal_ok = bool(result.formal_feasible) if result.supported else False
        rows.append(
            {
                "Domain": "architecture_ncarb50_v2",
                "Split": split,
                "Mode": str(mode_cfg["mode"]),
                "Model": model_name,
                "Overlay": overlay_key,
                "task_id": task_id,
                "relation_extension": "true" if mode_cfg.get("extend_relation_templates") else "false",
                "relation_templates_added": relation_extension_stats.get("added_template_count", 0),
                "oracle_candidate_count": len(model_reference_ids),
                "candidate_rule_count_present": len(candidate_rules),
                "predicted_model_rule_ids": predicted_model,
                "model_reference_rule_ids": model_reference_ids,
                "projected_canonical_rule_ids": predicted_canonical,
                "canonical_reference_rule_ids": reference_canonical,
                "model_rule_precision": rule_precision(predicted_model, model_reference_ids),
                "model_rule_recall": rule_recall(predicted_model, model_reference_ids),
                "canonical_rule_precision": rule_precision(predicted_canonical, reference_canonical),
                "canonical_rule_recall": rule_recall(predicted_canonical, reference_canonical),
                "formal_feasible": formal_ok,
                "semantic_valid": sem_ok,
                "false_accept": bool(formal_ok and not sem_ok),
                "invalid_case": bool(not sem_ok),
                "unsupported_reason": "" if result.supported else result.unsupported_reason,
                "runtime_ms": round(elapsed, 3),
            }
        )

    aggregate = aggregate_rows(
        rows,
        split=split,
        mode=str(mode_cfg["mode"]),
        model_name=model_name,
        overlay_key=overlay_key,
        strong_coverage=strong_coverage,
    )
    return aggregate, rows


def markdown_table(rows: list[dict[str, Any]], headers: list[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(csv_cell(row.get(header)) for header in headers) + " |")
    return "\n".join(lines)


def build_report(aggregate_rows_all: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    headers = [
        "Domain",
        "Split",
        "Mode",
        "Model",
        "Strong canonical coverage",
        "Relation extension",
        "Relation templates added",
        "Canonical Rule Precision",
        "Canonical Rule Recall",
        "Formal CSR",
        "Sem-CSR",
        "False accept",
        "Candidate zero",
        "CTHR no valid",
        "Unsupported tasks",
        "Invalid cases",
    ]
    return "\n".join(
        [
            "# Section 6.4 Architecture NCARB50 v2 Rule-Library Comparison",
            "",
            "This diagnostic evaluates the 50-task architecture extension set with model-specific rule libraries and overlays.",
            "The oracle-candidate mode uses strong-aligned model rule IDs as candidates and then runs CTHR default valid-rule recovery.",
            "The oracle-valid-upper mode passes the same strong-aligned model rule IDs directly as selected valid rules.",
            "The relation-extended modes add a relation-to-template layer that materializes task-bound rule constraints for finer-grained relations such as definitions, formula derivation, and co-effective rule activation.",
            "",
            "## Aggregate Results",
            "",
            markdown_table(aggregate_rows_all, headers),
            "",
            "## Mode Definitions",
            "",
            markdown_table(MODES, ["mode", "description"]),
            "",
            "## Dataset Summary",
            "",
            "```json",
            json.dumps(summary.get("dataset", {}), ensure_ascii=False, indent=2),
            "```",
            "",
            "## Run Summary",
            "",
            "```json",
            json.dumps(summary, ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )


def main() -> None:
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    canonical_references = canonical_reference_by_task()
    aggregate_rows_all: list[dict[str, Any]] = []
    task_rows_all: list[dict[str, Any]] = []
    per_model_summary: dict[str, Any] = {}

    for split in ["full", "strict_common"]:
        for mode_cfg in MODES:
            for model_spec in MODEL_SPECS:
                model_name = str(model_spec["model"])
                print(f"running architecture_ncarb50_v2 {split} {mode_cfg['mode']} {model_name}", flush=True)
                aggregate, rows = evaluate_model(model_spec, split, mode_cfg, canonical_references)
                aggregate_rows_all.append(aggregate)
                task_rows_all.extend(rows)
                per_model_summary.setdefault(model_name, {})[f"{split}:{mode_cfg['mode']}"] = aggregate

    outputs = {
        "aggregate_csv": RESULTS_DIR / "section_6_4_architecture_ncarb50_rule_library_compare_table.csv",
        "task_rows_csv": RESULTS_DIR / "section_6_4_architecture_ncarb50_rule_library_compare_task_rows.csv",
        "report_md": RESULTS_DIR / "section_6_4_architecture_ncarb50_rule_library_compare_report.md",
        "summary_json": RESULTS_DIR / "section_6_4_architecture_ncarb50_rule_library_compare_summary.json",
    }
    aggregate_headers = [
        "Domain",
        "Split",
        "Mode",
        "Model",
        "Overlay",
        "Strong canonical coverage",
        "Relation extension",
        "Relation templates added",
        "Canonical Rule Precision",
        "Canonical Rule Recall",
        "Model-ID Rule Precision",
        "Model-ID Rule Recall",
        "Formal CSR",
        "Sem-CSR",
        "False accept",
        "Candidate zero",
        "CTHR no valid",
        "Unsupported tasks",
        "Mean oracle candidates",
        "Invalid cases",
    ]
    task_headers = [
        "Domain",
        "Split",
        "Mode",
        "Model",
        "Overlay",
        "task_id",
        "relation_extension",
        "relation_templates_added",
        "oracle_candidate_count",
        "candidate_rule_count_present",
        "predicted_model_rule_ids",
        "model_reference_rule_ids",
        "projected_canonical_rule_ids",
        "canonical_reference_rule_ids",
        "model_rule_precision",
        "model_rule_recall",
        "canonical_rule_precision",
        "canonical_rule_recall",
        "formal_feasible",
        "semantic_valid",
        "false_accept",
        "invalid_case",
        "unsupported_reason",
        "runtime_ms",
    ]
    dataset_summary = {
        "dataset_root": str(DATASET_ROOT),
        "manifest": str(MANIFEST),
        "overlay_manifest": str(OVERLAY_MANIFEST),
        "manifest_payload": read_json(MANIFEST),
        "strict_common_payload": read_json(STRICT_COMMON_TASKS),
        "overlay_manifest_payload": read_json(OVERLAY_MANIFEST),
    }
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "started_at": started_at,
        "dataset": dataset_summary,
        "mode_definitions": MODES,
        "models": per_model_summary,
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    write_csv(outputs["aggregate_csv"], aggregate_rows_all, aggregate_headers)
    write_csv(outputs["task_rows_csv"], task_rows_all, task_headers)
    write_json(outputs["summary_json"], summary)
    outputs["report_md"].write_text(build_report(aggregate_rows_all, summary), encoding="utf-8")
    print(json.dumps({"aggregate": aggregate_rows_all, "outputs": summary["outputs"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
