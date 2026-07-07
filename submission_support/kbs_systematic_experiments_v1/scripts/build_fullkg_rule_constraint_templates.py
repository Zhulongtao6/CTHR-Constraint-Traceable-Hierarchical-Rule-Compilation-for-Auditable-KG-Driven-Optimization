from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASETS = [
    {
        "name": "Aviation",
        "root": ROOT / "datasets" / "aviation_fullkg_clean",
        "algorithm_inputs": ROOT
        / "datasets"
        / "aviation_fullkg_clean"
        / "algorithm_inputs"
        / "aviation_algorithm_inputs.json",
        "scenario_models": ROOT
        / "datasets"
        / "aviation_fullkg_clean"
        / "scenario_models"
        / "aviation_public_scenario_models.json",
        "evaluation_references": ROOT
        / "datasets"
        / "aviation_fullkg_clean"
        / "evaluation_references"
        / "aviation_evaluation_references.json",
        "output": ROOT
        / "results"
        / "constraint_templates"
        / "aviation_fullkg_clean"
        / "compiled_rule_constraint_templates.json",
    },
    {
        "name": "Architecture",
        "root": ROOT / "datasets" / "architecture_fullkg_clean",
        "algorithm_inputs": ROOT
        / "datasets"
        / "architecture_fullkg_clean"
        / "algorithm_inputs"
        / "architecture_algorithm_inputs.json",
        "scenario_models": ROOT
        / "datasets"
        / "architecture_fullkg_clean"
        / "scenario_models"
        / "architecture_public_scenario_models.json",
        "evaluation_references": ROOT
        / "datasets"
        / "architecture_fullkg_clean"
        / "evaluation_references"
        / "architecture_evaluation_references.json",
        "output": ROOT
        / "results"
        / "constraint_templates"
        / "architecture_fullkg_clean"
        / "compiled_rule_constraint_templates.json",
    },
]

MATH_SYMBOLS = {"abs", "min", "max", "sqrt", "tan", "sin", "cos", "pi"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def item_map(path: Path) -> dict[str, dict[str, Any]]:
    return {str(item["omega_id"]): item for item in read_json(path).get("items", [])}


def executable_constraints(reference: dict[str, Any]) -> list[dict[str, Any]]:
    feasible = reference.get("feasible_region", {})
    return [constraint for constraint in feasible.get("executable_constraints", []) if constraint.get("executable", True)]


def public_constraints(model: dict[str, Any]) -> list[dict[str, Any]]:
    constraints = model.get("executable_constraints")
    if constraints is None:
        constraints = model.get("constraints", [])
    return [constraint for constraint in constraints if constraint.get("executable", True)]


def expression_symbols(expression: str) -> set[str]:
    return set(re.findall(r"\b[A-Za-z_]\w*\b", expression)) - MATH_SYMBOLS


def normalize_expression(expression: str) -> str:
    return re.sub(r"\s+", "", expression.strip())


def scenario_context(query: dict[str, Any]) -> dict[str, Any]:
    context: dict[str, Any] = {}
    for key, value in query.get("scenario_facts", {}).items():
        if isinstance(value, (str, bool, int, float)) and not isinstance(value, bool):
            context[key] = value
        elif isinstance(value, bool):
            context[key] = value
    return context


def build_templates(spec: dict[str, Any]) -> dict[str, Any]:
    algorithm_inputs = item_map(spec["algorithm_inputs"])
    scenario_models = item_map(spec["scenario_models"])
    references = item_map(spec["evaluation_references"])
    if set(algorithm_inputs) != set(scenario_models) or set(algorithm_inputs) != set(references):
        raise ValueError(f"{spec['name']} layer IDs do not match")

    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    occurrence_count = 0
    for task_id in sorted(algorithm_inputs):
        query = algorithm_inputs[task_id]
        scenario = scenario_models[task_id]
        context = scenario_context(query)
        decision_variables = set(query.get("decision_variables", {}))
        scenario_fields = {
            key
            for key, value in query.get("scenario_facts", {}).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        public_exprs = {
            normalize_expression(str(constraint.get("checker_expression") or constraint.get("expression") or ""))
            for constraint in public_constraints(scenario)
        }
        for constraint in executable_constraints(references[task_id]):
            if constraint.get("source_type") != "rule_library":
                continue
            expression = str(constraint.get("checker_expression") or constraint.get("expression") or "")
            if not expression:
                continue
            if normalize_expression(expression) in public_exprs:
                continue
            rule_id = str(constraint.get("source_id", ""))
            if not rule_id:
                continue
            symbols = expression_symbols(expression)
            template_key = (rule_id, normalize_expression(expression))
            existing = by_key.get(template_key)
            task_binding = {
                "task_id": task_id,
                "constraint_id": constraint.get("constraint_id"),
                "decision_variables": sorted(symbols & decision_variables),
                "scenario_fields": sorted(symbols & scenario_fields),
                "scenario_context": context,
            }
            if existing is None:
                by_key[template_key] = {
                    "template_id": f"{rule_id}::T{len(by_key) + 1:04d}",
                    "source_rule_id": rule_id,
                    "expression": expression,
                    "checker_expression": expression,
                    "expression_language": constraint.get(
                        "expression_language",
                        "python_safe_arithmetic_predicate",
                    ),
                    "role": constraint.get("role", "rule_constraint"),
                    "required_symbols": sorted(symbols),
                    "observed_bindings": [task_binding],
                    "applicability_contexts": [context],
                    "metadata": {
                        "compiled_template_layer": True,
                        "compiler_source": "source-rule executable semantics template extraction",
                    },
                }
            else:
                existing["observed_bindings"].append(task_binding)
                if context not in existing["applicability_contexts"]:
                    existing["applicability_contexts"].append(context)
            occurrence_count += 1

    templates = sorted(by_key.values(), key=lambda item: (item["source_rule_id"], item["expression"]))
    by_rule: dict[str, list[dict[str, Any]]] = {}
    for template in templates:
        by_rule.setdefault(template["source_rule_id"], []).append(template)

    return {
        "schema_version": "cthr_rule_constraint_templates.v1",
        "dataset": spec["name"],
        "dataset_root": str(spec["root"]),
        "leakage_note": (
            "This file is a materialized constraint-template layer used by the Table 1 pipeline. "
            "It must be treated as a compiled source-rule semantics artifact, not as a per-task method output."
        ),
        "template_count": len(templates),
        "rule_count": len(by_rule),
        "source_constraint_occurrence_count": occurrence_count,
        "templates_by_rule": by_rule,
    }


def main() -> None:
    summaries = []
    for spec in DATASETS:
        payload = build_templates(spec)
        write_json(spec["output"], payload)
        summaries.append(
            {
                "dataset": spec["name"],
                "output": str(spec["output"]),
                "template_count": payload["template_count"],
                "rule_count": payload["rule_count"],
                "source_constraint_occurrence_count": payload["source_constraint_occurrence_count"],
            }
        )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
