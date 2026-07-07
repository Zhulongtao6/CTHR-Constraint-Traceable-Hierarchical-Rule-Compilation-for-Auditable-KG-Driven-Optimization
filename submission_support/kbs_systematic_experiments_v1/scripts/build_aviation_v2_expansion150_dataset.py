from __future__ import annotations

import json
import math
import re
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from build_aviation_fullkg_overlays import build_alignment, model_ids_for, weak_model_ids_for


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "datasets" / "aviation_fullkg_clean"
OUT_ROOT = ROOT / "datasets" / "aviation_fullkg_v2_expansion150"

QWEN_RULE_LIBRARY = SOURCE_ROOT / "rule_libraries" / "qwen" / "full_aviation_rule_library_qwen.json"
DEEPSEEK_RULE_LIBRARY = (
    SOURCE_ROOT
    / "rule_libraries"
    / "deepseek"
    / "full_aviation_rule_library_deepseek_strict_repaired.json"
)
MIMO_RULE_LIBRARY = SOURCE_ROOT / "rule_libraries" / "xiaomi_mimo" / "full_aviation_rule_library_mimo.json"

TASK_COUNT = 150


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def snake(text: str, fallback: str = "value") -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = fallback
    if text[0].isdigit():
        text = f"{fallback}_{text}"
    return text


def unit_suffix(unit: Any) -> str:
    mapping = {
        "feet": "ft",
        "foot": "ft",
        "ft": "ft",
        "meter": "m",
        "meters": "m",
        "m": "m",
        "kilometer": "km",
        "kilometers": "km",
        "km": "km",
        "nautical mile": "nm",
        "nautical miles": "nm",
        "nm": "nm",
        "degree": "deg",
        "degrees": "deg",
        "deg": "deg",
        "seconds": "s",
        "second": "s",
        "s": "s",
        "percent": "percent",
        "%": "percent",
    }
    u = str(unit or "").strip().lower()
    return mapping.get(u, snake(u, "unit") if u and u != "unknown" else "")


def numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return float(value)
        return None
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def compare_expr(var: str, op: str, value: Any) -> str:
    py_op = "==" if op in {"=", "eq"} else op
    if py_op == "≤":
        py_op = "<="
    if py_op == "≥":
        py_op = ">="
    if py_op == "lt":
        py_op = "<"
    if py_op == "le":
        py_op = "<="
    if py_op == "gt":
        py_op = ">"
    if py_op == "ge":
        py_op = ">="
    if py_op == "in":
        py_op = "=="
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return f"{var} {py_op} {json.dumps(value, ensure_ascii=False) if isinstance(value, str) else value}"


def range_for(op: str, value: float, unit: str) -> tuple[float, float]:
    magnitude = max(abs(value), 1.0)
    if op in {">=", ">", "ge", "gt"}:
        lower = max(0.0, value - 0.35 * magnitude)
        upper = value + 0.80 * magnitude + 5
    elif op in {"<=", "<", "le", "lt"}:
        lower = max(0.0, value - 0.80 * magnitude - 5)
        upper = value + 0.35 * magnitude + 5
    else:
        lower = max(0.0, value - 0.60 * magnitude - 2)
        upper = value + 0.60 * magnitude + 2
    if unit == "percent":
        lower = max(0.0, lower)
        upper = min(max(upper, value + 1), 100.0)
    if abs(upper - lower) < 1e-6:
        upper = lower + 1
    return round(lower, 3), round(upper, 3)


def extract_guard_facts(rule: dict[str, Any]) -> dict[str, Any]:
    guard = rule.get("guard") or {}
    facts: dict[str, Any] = {}
    for clause in guard.get("all", []) if isinstance(guard, dict) else []:
        field = str(clause.get("field", "")).replace("scenario.", "")
        if not field:
            continue
        op = str(clause.get("op", "eq"))
        value = clause.get("value")
        if op in {"eq", "="}:
            facts[field] = value
        elif op == "in" and isinstance(value, list) and value:
            facts[field] = value[0]
        elif op in {"lt", "<"}:
            facts[field] = float(value) - 1 if numeric(value) is not None else value
        elif op in {"le", "<="}:
            facts[field] = value
        elif op in {"gt", ">"}:
            facts[field] = float(value) + 1 if numeric(value) is not None else value
        elif op in {"ge", ">="}:
            facts[field] = value
    return facts


def classify_rule(rule: dict[str, Any]) -> tuple[str, str]:
    text = " ".join(
        str(rule.get(key, ""))
        for key in ("rule_id", "name", "extraction_notes")
    ).lower()
    if any(token in text for token in ["helicopter", "fato", "pins"]):
        return "helicopter_procedure", "直升机程序"
    if any(token in text for token in ["ils", "glide", "approach", "sbas", "fas", "taa", "ttaa", "missed", "intermediate", "descent", "pbn", "rnp", "rnav"]):
        return "approach_procedure", "进近程序"
    if any(token in text for token in ["holding", "vor", "dme", "departure", "der", "outbound", "turn initiation"]):
        return "conventional_procedure", "传统程序"
    if any(token in text for token in ["chart", "publication", "title", "supplementary"]):
        return "publication_procedure", "公布程序"
    return "fundamental_knowledge", "基础知识"


def relation_types(rule_ids: list[str], rule_lookup: dict[str, dict[str, Any]]) -> list[str]:
    labels = set()
    if len(rule_ids) > 1:
        labels.add("multi_rule_conjunction")
    for rid in rule_ids:
        rule = rule_lookup[rid]
        text = " ".join(
            [
                str(rule.get("rule_id", "")),
                str(rule.get("name", "")),
                str(rule.get("guard", "")),
                str(rule.get("relations", "")),
            ]
        ).lower()
        if rule.get("guard"):
            labels.add("scenario_conditioned_applicability")
        if any(token in text for token in ["formula", "calculation", "计算", "rounding", "取整"]):
            labels.add("dependency_or_formula_propagation")
        if any(token in text for token in ["exception", "permitted", "exclude", "straight", "turn initiation", "fato", "der"]):
            labels.add("branch_or_exclusion")
        if any(token in text for token in ["precedence", "priority"]):
            labels.add("precedence")
        if any(token in text for token in ["moc", "minimum", "maximum", "limit", "range", "radius", "height", "gradient", "angle"]):
            labels.add("parameter_limit")
        if any(token in text for token in ["chart", "publication", "title", "supplementary", "公布"]):
            labels.add("provenance_traceability")
    return sorted(labels or {"generic_rule_selection"})


def rule_constraints(rule: dict[str, Any]) -> list[dict[str, Any]]:
    constraints = rule.get("constraints")
    if isinstance(constraints, list) and constraints:
        return [c for c in constraints if isinstance(c, dict)]
    return [
        {
            "variable": f"{snake(rule.get('rule_id'))}_semantic_compliance",
            "op": "=",
            "value": "required",
            "unit": "unknown",
            "source_quote": rule.get("name", ""),
        }
    ]


def build_variables_and_constraints(
    task_id: str,
    selected_rules: list[dict[str, Any]],
    variant: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    decision_variables: dict[str, Any] = {}
    rule_constraints_out: list[dict[str, Any]] = []
    public_constraints: list[dict[str, Any]] = []
    numeric_vars: list[str] = []
    semantic_vars: list[str] = []
    constraint_index = 1

    for rule in selected_rules:
        rid = str(rule["rule_id"])
        seen_exprs: set[str] = set()
        for raw_constraint in rule_constraints(rule)[:3]:
            raw_var = str(raw_constraint.get("variable") or f"{rid}_parameter")
            unit = str(raw_constraint.get("unit") or "unknown")
            suffix = unit_suffix(unit)
            var_name = snake(raw_var, "rule_parameter")
            if suffix and not var_name.endswith(f"_{suffix}"):
                var_name = f"{var_name}_{suffix}"
            value = raw_constraint.get("value")
            op = str(raw_constraint.get("op") or "=")
            n_value = numeric(value)

            if n_value is not None:
                lower, upper = range_for(op, n_value, suffix)
                decision_variables.setdefault(
                    var_name,
                    {
                        "type": "continuous",
                        "unit": suffix or unit or "unitless",
                        "lower": lower,
                        "upper": upper,
                    },
                )
                expr = compare_expr(var_name, op, n_value)
                metadata = {
                    "canonical_rule_id": rid,
                    "source_quote": raw_constraint.get("source_quote", ""),
                    "kg_evidence": raw_constraint.get("evidence", {}),
                }
                numeric_vars.append(var_name)
            else:
                var_name = f"{snake(rid, 'rule')}_semantic_indicator"
                decision_variables.setdefault(
                    var_name,
                    {"type": "binary", "unit": "indicator", "lower": 0, "upper": 1},
                )
                expr = f"{var_name} == 1"
                metadata = {
                    "semantic_proxy_encoding": True,
                    "derived_from_text_rule": True,
                    "derived_from_rule_id": rid,
                    "source_quote": raw_constraint.get("source_quote", rule.get("name", "")),
                    "derivation_note": "Textual or categorical rule encoded as a benchmark semantic indicator.",
                }
                semantic_vars.append(var_name)

            if expr in seen_exprs:
                continue
            seen_exprs.add(expr)
            symbols = [var_name]
            rule_constraints_out.append(
                {
                    "constraint_id": f"C{constraint_index}",
                    "expression": expr,
                    "role": snake(raw_constraint.get("variable") or rule.get("name"), "rule_constraint"),
                    "source_type": "rule_library",
                    "source_id": rid,
                    "executable": True,
                    "checker_expression": expr,
                    "expression_language": "python_safe_arithmetic_predicate",
                    "symbols": {
                        "decision_variables": symbols,
                        "scenario_fields": [],
                        "unresolved_symbols": [],
                    },
                    "metadata": metadata,
                }
            )
            constraint_index += 1

    if not numeric_vars and semantic_vars:
        numeric_vars = semantic_vars[:]

    base_var = numeric_vars[0]
    decision_variables["procedure_complexity_score"] = {
        "type": "continuous",
        "unit": "score",
        "lower": 0,
        "upper": 20,
    }
    decision_variables["design_resilience_score"] = {
        "type": "continuous",
        "unit": "score",
        "lower": 0,
        "upper": 20,
    }
    if len(numeric_vars) > 1:
        secondary_var = numeric_vars[1]
    else:
        secondary_var = base_var

    c90 = {
        "constraint_id": "C90",
        "expression": f"procedure_complexity_score >= 0.1 * {base_var} + {1 + (variant % 4)}",
        "role": "procedure_complexity_from_design_parameter",
        "source_type": "task_or_scenario_model",
        "source_id": f"{task_id}_public_scenario_model",
        "executable": True,
        "checker_expression": f"procedure_complexity_score >= 0.1 * {base_var} + {1 + (variant % 4)}",
        "expression_language": "python_safe_arithmetic_predicate",
        "symbols": {
            "decision_variables": ["procedure_complexity_score", base_var],
            "scenario_fields": [],
            "unresolved_symbols": [],
        },
        "metadata": {
            "objective_closure": True,
            "closure_source": "task_or_scenario_model",
            "closure_visibility": "public_algorithm_input",
        },
    }
    c91 = {
        "constraint_id": "C91",
        "expression": f"design_resilience_score <= 20 - 0.05 * {secondary_var}",
        "role": "resilience_proxy_from_design_parameter",
        "source_type": "task_or_scenario_model",
        "source_id": f"{task_id}_public_scenario_model",
        "executable": True,
        "checker_expression": f"design_resilience_score <= 20 - 0.05 * {secondary_var}",
        "expression_language": "python_safe_arithmetic_predicate",
        "symbols": {
            "decision_variables": ["design_resilience_score", secondary_var],
            "scenario_fields": [],
            "unresolved_symbols": [],
        },
        "metadata": {
            "objective_closure": True,
            "closure_source": "task_or_scenario_model",
            "closure_visibility": "public_algorithm_input",
        },
    }
    public_constraints.extend([c90, c91])
    return decision_variables, rule_constraints_out, public_constraints, numeric_vars


def provenance_for_rules(selected_rules: list[dict[str, Any]]) -> dict[str, Any]:
    chunk_ids: list[str] = []
    node_ids: list[str] = []
    edge_ids: list[str] = []
    docs: list[dict[str, Any]] = []
    for rule in selected_rules:
        for chunk_id in rule.get("source_chunk_ids", []) or []:
            if chunk_id not in chunk_ids:
                chunk_ids.append(chunk_id)
        for node_id in rule.get("source_node_ids", []) or []:
            if node_id not in node_ids:
                node_ids.append(node_id)
        for edge_id in rule.get("source_edge_ids", []) or []:
            if edge_id not in edge_ids:
                edge_ids.append(edge_id)
        for chunk_id in (rule.get("source_chunk_ids", []) or [None])[:1]:
            docs.append(
                {
                    "document": "CAAC flight procedure design source KG",
                    "section": str(rule.get("rule_id")),
                    "page": "unknown",
                    "chunk_id": chunk_id,
                }
            )
    return {
        "kg_chunk_ids": chunk_ids,
        "kg_node_ids": node_ids,
        "kg_edge_ids": edge_ids,
        "source_documents": docs,
    }


def build_task(
    index: int,
    selected_ids: list[str],
    rule_lookup: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    task_id = f"AVI_V2_{index:03d}"
    selected_rules = [rule_lookup[rid] for rid in selected_ids]
    primary = selected_rules[0]
    task_type, section_cn = classify_rule(primary)
    title = f"{section_cn}强共同规则场景 {index:03d}"
    guard_facts: dict[str, Any] = {}
    for rule in selected_rules:
        guard_facts.update(extract_guard_facts(rule))
    scenario_facts = {
        "benchmark_round": "aviation_v2_expansion150",
        "document_section_family": section_cn,
        "task_family": task_type,
        "procedure_design_context": f"{section_cn}工程化规则应用",
        "scenario_variant_index": index,
        "traffic_density_level": ["low", "medium", "high"][index % 3],
        "terrain_context": ["plain", "hilly", "mountainous"][index % 3],
    }
    scenario_facts.update(guard_facts)

    decision_variables, rule_constraints_out, public_constraints, numeric_vars = build_variables_and_constraints(
        task_id, selected_rules, index
    )
    objectives = [
        {
            "name": "minimize_procedure_complexity",
            "expression": "procedure_complexity_score",
        },
        {
            "name": "maximize_design_resilience",
            "expression": "design_resilience_score",
        },
    ]
    if numeric_vars:
        objectives.append(
            {
                "name": "minimize_primary_design_parameter",
                "expression": numeric_vars[0],
            }
        )
        lambdas = [0.4, 0.35, 0.25]
    else:
        lambdas = [0.5, 0.5]

    algorithm_input = {
        "omega_id": task_id,
        "title": title,
        "domain": "aviation_procedure_design",
        "task_type": task_type,
        "design_intent": (
            f"Create a source-grounded aviation procedure design scenario in the {section_cn} section. "
            "The optimizer balances compact procedure geometry against design resilience while the hidden "
            "evaluation checks the corresponding common rules shared by Qwen, DeepSeek, and Xiaomi MIMO."
        ),
        "scenario_facts": scenario_facts,
        "decision_variables": decision_variables,
        "objectives": objectives,
        "query_preferences": {
            "lambda": lambdas,
            "meaning": "balance operational compactness, resilience margin, and the primary procedure parameter",
        },
        "public_scenario_model": {
            "model_id": f"{task_id}_scenario_model",
            "path": "scenario_models/aviation_public_scenario_models.json",
            "visibility": "public_algorithm_input",
            "purpose": "Non-normative task physics/objective-closure constraints visible to optimizers; contains no expected rule IDs or labels.",
        },
        "visible_input_note": (
            "Visible task input only. Algorithms may also read the public scenario model and rule library. "
            "Rule labels, rule-derived feasible-region answers, and rule-id bindings remain hidden evaluation references."
        ),
    }

    scenario_model = {
        "omega_id": task_id,
        "model_id": f"{task_id}_scenario_model",
        "title": title,
        "visibility": "public_algorithm_input",
        "model_scope": "task_physics_and_objective_closure_only",
        "leakage_policy": "No expected rule IDs, defeated/surviving labels, provenance answers, certificate targets, or rule-library bindings are included.",
        "constraints": public_constraints,
    }

    challenge_types = relation_types(selected_ids, rule_lookup)
    feasible_constraints = rule_constraints_out + public_constraints
    reference = {
        "omega_id": task_id,
        "title": title,
        "rule_structure": {
            "expected_source_rule_ids": selected_ids,
            "expected_defeated_rule_ids": [],
            "expected_surviving_rule_ids": selected_ids,
            "expected_valid_rule_structures": [selected_ids],
            "expected_rule_behavior": {
                "should_activate": [str(rule.get("name", rid)) for rid, rule in zip(selected_ids, selected_rules)],
                "should_exclude": [],
                "should_resolve": challenge_types,
            },
            "challenge_types": challenge_types,
            "valid_constraint_cell_ids": [f"{task_id}_cell_1"],
            "expected_provenance": provenance_for_rules(selected_rules),
        },
        "feasible_region": {
            "executable_constraints": feasible_constraints,
            "structure_only_constraints": [
                {
                    "constraint_id": "C0",
                    "expression": "scenario facts satisfy the selected source-rule guards",
                    "role": "scenario_applicability_guard",
                    "source_type": "task_or_scenario_model",
                    "source_id": f"{task_id}_scenario_guard",
                    "executable": False,
                    "reason_not_executable": "scenario guard used for applicability, not a numeric decision constraint",
                }
            ],
            "valid_constraint_cells": [
                {
                    "cell_id": f"{task_id}_cell_1",
                    "source_type": "combined_rule_cell",
                    "required_rule_ids": selected_ids,
                    "constraint_ids": [c["constraint_id"] for c in feasible_constraints],
                }
            ],
            "reference_semantics": {
                "positive_membership_condition": "all executable_constraints evaluate true and the valid source-rule cell is recovered",
                "structure_only_constraints_usage": "used to check rule resolution, provenance, or specialized encoders before numeric membership checking",
            },
        },
        "certificate_targets": {
            "source_rule_ids": selected_ids,
            "provenance": provenance_for_rules(selected_rules)["source_documents"],
        },
        "diagnostic_candidate_rule_ids_reference_only": selected_ids,
        "source_grounding": {
            "construction_policy": "generated from full-library Qwen/DeepSeek/Xiaomi MIMO exact-or-strong common rules",
            "common_rule_pool": "COMMON_STRONG_RULES.json",
            "benchmark_round": "aviation_v2_expansion150",
        },
    }

    task_file_payload = {
        "version": "aviation_v2_expansion150_task_v1",
        "algorithm_input": algorithm_input,
        "evaluation_reference": reference,
    }
    return algorithm_input, scenario_model, reference, task_file_payload


def build_templates(references: list[dict[str, Any]], algorithms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_rule: dict[str, list[dict[str, Any]]] = defaultdict(list)
    template_index = 1
    occurrence_count = 0
    seen: set[tuple[str, str]] = set()
    for reference in references:
        task_id = str(reference["omega_id"])
        scenario_context = {
            k: v
            for k, v in algorithms[task_id].get("scenario_facts", {}).items()
            if isinstance(v, (str, int, float, bool))
        }
        decision_vars = set(algorithms[task_id].get("decision_variables", {}))
        for constraint in reference.get("feasible_region", {}).get("executable_constraints", []):
            if constraint.get("source_type") != "rule_library":
                continue
            rule_id = str(constraint.get("source_id"))
            expr = str(constraint.get("checker_expression") or constraint.get("expression"))
            key = (rule_id, re.sub(r"\s+", "", expr))
            occurrence_count += 1
            if key in seen:
                continue
            seen.add(key)
            symbols = constraint.get("symbols", {}).get("decision_variables", [])
            by_rule[rule_id].append(
                {
                    "template_id": f"{rule_id}::V2T{template_index:04d}",
                    "source_rule_id": rule_id,
                    "expression": expr,
                    "checker_expression": expr,
                    "expression_language": constraint.get(
                        "expression_language", "python_safe_arithmetic_predicate"
                    ),
                    "role": constraint.get("role", "rule_constraint"),
                    "required_symbols": sorted(set(symbols)),
                    "observed_bindings": [
                        {
                            "task_id": task_id,
                            "constraint_id": constraint.get("constraint_id"),
                            "decision_variables": sorted(set(symbols) & decision_vars),
                            "scenario_fields": constraint.get("symbols", {}).get("scenario_fields", []),
                            "scenario_context": scenario_context,
                        }
                    ],
                    "applicability_contexts": [scenario_context],
                    "metadata": {
                        "compiled_template_layer": True,
                        "compiler_source": "aviation_v2_expansion150 source-rule executable semantics",
                    },
                }
            )
            template_index += 1
    return {
        "schema_version": "cthr_rule_constraint_templates.v1",
        "dataset": "Aviation V2 Expansion 150",
        "dataset_root": str(OUT_ROOT),
        "leakage_note": "Compiled source-rule semantics artifact; not visible as per-task answers.",
        "template_count": sum(len(items) for items in by_rule.values()),
        "rule_count": len(by_rule),
        "source_constraint_occurrence_count": occurrence_count,
        "templates_by_rule": {rid: by_rule[rid] for rid in sorted(by_rule)},
    }


def remap_reference(reference: dict[str, Any], alignment_by_id: dict[str, dict[str, Any]], model: str) -> dict[str, Any]:
    out = json.loads(json.dumps(reference, ensure_ascii=False))
    structure = out["rule_structure"]
    canonical_ids = [str(rid) for rid in structure.get("expected_surviving_rule_ids", [])]
    projected: list[str] = []
    per_rule: list[dict[str, Any]] = []
    unresolved: list[str] = []
    for rid in canonical_ids:
        model_ids = model_ids_for(rid, alignment_by_id)
        weak_ids = weak_model_ids_for(rid, alignment_by_id)
        if model_ids:
            projected.extend(model_ids)
        else:
            unresolved.append(rid)
        per_rule.append(
            {
                "canonical_rule_id": rid,
                "model_rule_ids": model_ids,
                "weak_candidate_model_rule_ids": weak_ids,
                "status": "exact_or_strong_alignment" if model_ids else "unresolved",
            }
        )
    projected = sorted(set(projected))
    structure["expected_surviving_rule_ids"] = projected
    structure["expected_valid_rule_structures"] = [projected] if projected else []
    structure["overlay_rule_id_projection"] = {
        "model": model,
        "canonical_expected_surviving_rule_ids": canonical_ids,
        "aligned_expected_surviving_rule_ids": projected,
        "unresolved_canonical_rule_ids": unresolved,
        "per_rule": per_rule,
        "feasible_region_policy": "feasible_region remains in canonical source-rule semantic space",
    }
    out["overlay_metadata"] = {
        "model": model,
        "rule_id_namespace": "model_overlay",
        "source_semantic_reference": "core/source_semantic_references/aviation_source_semantic_references.json",
        "projection_uses": "exact_or_strong_alignment only",
        "unresolved_canonical_rule_ids": unresolved,
    }
    return out


def remap_templates(templates: dict[str, Any], alignment_by_id: dict[str, dict[str, Any]], model: str) -> dict[str, Any]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unresolved: list[str] = []
    for canonical_id, items in templates.get("templates_by_rule", {}).items():
        model_ids = model_ids_for(str(canonical_id), alignment_by_id)
        if not model_ids:
            unresolved.append(str(canonical_id))
            continue
        for model_id in model_ids:
            for item in items:
                copied = json.loads(json.dumps(item, ensure_ascii=False))
                copied["source_rule_id"] = model_id
                metadata = dict(copied.get("metadata", {}))
                metadata.update(
                    {
                        "canonical_rule_id": str(canonical_id),
                        "model_rule_id": model_id,
                        "rule_id_overlay_model": model,
                        "rule_id_projection": True,
                    }
                )
                copied["metadata"] = metadata
                out[model_id].append(copied)
    return {
        "schema_version": "cthr_rule_constraint_templates.v1",
        "dataset": "Aviation V2 Expansion 150",
        "overlay_model": model,
        "semantic_template_policy": "Template expressions are copied from canonical source semantics; keys are projected into the model rule-id namespace using exact-or-strong alignment only.",
        "template_count": sum(len(v) for v in out.values()),
        "rule_count": len(out),
        "source_constraint_occurrence_count": templates.get("source_constraint_occurrence_count"),
        "unresolved_canonical_template_rule_ids": sorted(unresolved),
        "templates_by_rule": {rid: out[rid] for rid in sorted(out)},
    }


def leakage_audit(algorithm_inputs: list[dict[str, Any]], references: list[dict[str, Any]]) -> dict[str, Any]:
    forbidden = [
        "expected_",
        "certificate_targets",
        "solver_constraints",
        "feasible_region",
        "evaluation_reference",
        "source_rule_ids",
        "surviving_rule_ids",
        "defeated_rule_ids",
        "rule_id",
    ]
    hits: list[dict[str, Any]] = []
    for item in algorithm_inputs:
        text = json.dumps(item, ensure_ascii=False)
        for key in forbidden:
            if key in text:
                hits.append({"task_id": item["omega_id"], "forbidden_key": key})
    all_ids = {item["omega_id"] for item in algorithm_inputs}
    ref_ids = {item["omega_id"] for item in references}
    return {
        "version": "aviation_v2_expansion150_leakage_audit_v1",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task_count": len(algorithm_inputs),
        "forbidden_input_key_hits": hits,
        "forbidden_input_key_hit_count": len(hits),
        "algorithm_reference_id_match": all_ids == ref_ids,
    }


def main() -> None:
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S")
    qwen_payload = read_json(QWEN_RULE_LIBRARY)
    deepseek_payload = read_json(DEEPSEEK_RULE_LIBRARY)
    mimo_payload = read_json(MIMO_RULE_LIBRARY)
    qwen_rules = [r for r in qwen_payload.get("rules", []) if isinstance(r, dict) and r.get("rule_id")]
    rule_lookup = {str(r["rule_id"]): r for r in qwen_rules}

    alignments: dict[str, dict[str, Any]] = {
        "qwen": {
            "canonical_to_model": [
                {
                    "canonical_rule_id": str(rule["rule_id"]),
                    "canonical_rule_name": rule.get("name", ""),
                    "status": "exact_or_strong_alignment",
                    "aligned_model_rule_ids": [str(rule["rule_id"])],
                    "weak_candidate_model_rule_ids": [],
                    "exact_or_strong_alignment": [
                        {
                            "model_rule_id": str(rule["rule_id"]),
                            "alignment_type": "exact_or_strong_alignment",
                            "confidence": 1.0,
                            "signals": {"exact_id": True},
                        }
                    ],
                    "weak_candidate_alignment": [],
                }
                for rule in qwen_rules
            ]
        },
        "deepseek": build_alignment(qwen_rules, deepseek_payload, "semantic_evidence"),
        "xiaomi_mimo": build_alignment(qwen_rules, mimo_payload, "semantic_evidence"),
    }
    alignment_by_model = {
        model: {entry["canonical_rule_id"]: entry for entry in payload["canonical_to_model"]}
        for model, payload in alignments.items()
    }
    common_rule_ids = sorted(
        rid
        for rid in rule_lookup
        if model_ids_for(rid, alignment_by_model["deepseek"])
        and model_ids_for(rid, alignment_by_model["xiaomi_mimo"])
    )
    if len(common_rule_ids) < 50:
        raise RuntimeError(f"Common strong rule pool too small: {len(common_rule_ids)}")

    tasks_rule_ids: list[list[str]] = []
    for i in range(TASK_COUNT):
        primary = common_rule_ids[i % len(common_rule_ids)]
        if i < len(common_rule_ids):
            selected = [primary]
        else:
            secondary = common_rule_ids[(i * 7 + 13) % len(common_rule_ids)]
            selected = [primary] if secondary == primary else [primary, secondary]
        tasks_rule_ids.append(selected)

    algorithms: list[dict[str, Any]] = []
    scenario_models: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    task_payloads: list[dict[str, Any]] = []
    for index, selected_ids in enumerate(tasks_rule_ids, start=1):
        algorithm, scenario_model, reference, task_payload = build_task(index, selected_ids, rule_lookup)
        algorithms.append(algorithm)
        scenario_models.append(scenario_model)
        references.append(reference)
        task_payloads.append(task_payload)

    algorithms_by_id = {item["omega_id"]: item for item in algorithms}
    templates = build_templates(references, algorithms_by_id)

    for directory in [
        OUT_ROOT / "algorithm_inputs",
        OUT_ROOT / "scenario_models",
        OUT_ROOT / "evaluation_references",
        OUT_ROOT / "constraint_templates",
        OUT_ROOT / "tasks",
        OUT_ROOT / "core" / "algorithm_inputs",
        OUT_ROOT / "core" / "scenario_models",
        OUT_ROOT / "core" / "source_semantic_references",
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    write_json(
        OUT_ROOT / "algorithm_inputs" / "aviation_algorithm_inputs.json",
        {"version": "aviation_v2_expansion150_algorithm_inputs_v1", "items": algorithms},
    )
    write_json(
        OUT_ROOT / "scenario_models" / "aviation_public_scenario_models.json",
        {
            "version": "aviation_v2_expansion150_public_scenario_models_v1",
            "visibility": "public_algorithm_input",
            "purpose": "Task physics and objective-closure constraints needed for fair optimization. These models do not contain expected rule IDs, defeated/surviving labels, provenance answers, certificate targets, or rule-library bindings.",
            "items": scenario_models,
        },
    )
    write_json(
        OUT_ROOT / "evaluation_references" / "aviation_evaluation_references.json",
        {
            "version": "aviation_v2_expansion150_source_semantic_references_v1",
            "canonical_rule_id_namespace": "qwen_canonical",
            "semantic_reference_policy": "Fixed source-grounded task semantics, feasible regions, and provenance. This file is hidden from algorithms.",
            "items": references,
        },
    )
    write_json(OUT_ROOT / "constraint_templates" / "compiled_rule_constraint_templates.json", templates)
    shutil.copy2(
        OUT_ROOT / "algorithm_inputs" / "aviation_algorithm_inputs.json",
        OUT_ROOT / "core" / "algorithm_inputs" / "aviation_algorithm_inputs.json",
    )
    shutil.copy2(
        OUT_ROOT / "scenario_models" / "aviation_public_scenario_models.json",
        OUT_ROOT / "core" / "scenario_models" / "aviation_public_scenario_models.json",
    )
    write_json(
        OUT_ROOT / "core" / "source_semantic_references" / "aviation_source_semantic_references.json",
        {
            "version": "aviation_v2_expansion150_core_source_semantic_references_v1",
            "canonical_rule_id_namespace": "qwen_canonical",
            "semantic_reference_policy": "Fixed source-grounded task semantics, feasible regions, and provenance. This file is not model-specific and must not be used as a model-generated rule library.",
            "items": references,
        },
    )
    for task_payload in task_payloads:
        write_json(OUT_ROOT / "tasks" / f"{task_payload['algorithm_input']['omega_id']}.json", task_payload)

    rule_library_outputs = {
        "qwen": (QWEN_RULE_LIBRARY, "full_aviation_rule_library_qwen.json", qwen_payload),
        "deepseek": (
            DEEPSEEK_RULE_LIBRARY,
            "full_aviation_rule_library_deepseek_strict_repaired.json",
            deepseek_payload,
        ),
        "xiaomi_mimo": (MIMO_RULE_LIBRARY, "full_aviation_rule_library_mimo.json", mimo_payload),
    }

    overlay_manifest: dict[str, Any] = {
        "version": "aviation_v2_expansion150_overlay_manifest_v1",
        "generated_at": generated_at,
        "core": {
            "algorithm_inputs": str(OUT_ROOT / "core" / "algorithm_inputs" / "aviation_algorithm_inputs.json"),
            "scenario_models": str(OUT_ROOT / "core" / "scenario_models" / "aviation_public_scenario_models.json"),
            "source_semantic_references": str(
                OUT_ROOT / "core" / "source_semantic_references" / "aviation_source_semantic_references.json"
            ),
        },
        "rule_libraries": {},
        "evaluation_overlays": {},
        "alignment_summaries": [],
    }
    for model, (source, output_name, model_payload) in rule_library_outputs.items():
        library_dir = OUT_ROOT / "rule_libraries" / model
        overlay_dir = OUT_ROOT / "evaluation_overlays" / model
        library_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, library_dir / output_name)
        model_alignment = alignments[model]
        alignment_by_id = alignment_by_model[model]
        used_alignment_entries = [
            entry for entry in model_alignment["canonical_to_model"] if entry["canonical_rule_id"] in common_rule_ids
        ]
        unresolved = [
            rid for rid in common_rule_ids if not model_ids_for(rid, alignment_by_id)
        ]
        alignment_payload = {
            "version": "aviation_v2_expansion150_rule_id_alignment_v1",
            "generated_at": generated_at,
            "model": model,
            "alignment_policy": {
                "canonical_namespace": "qwen_canonical_source_semantics",
                "model_namespace": model,
                "many_to_many": True,
                "formal_alignment_classes": [
                    "exact_or_strong_alignment",
                    "weak_candidate_alignment",
                    "unresolved",
                ],
                "unresolved_policy": "Canonical rules without exact-or-strong alignment are excluded from the v2 expansion task pool.",
                "weak_candidate_policy": "Weak candidates are retained for audit only and are never used to build expected_surviving_rule_ids.",
            },
            "summary": {
                "model": model,
                "model_rule_count": len(model_payload.get("rules", [])),
                "canonical_rule_count": len(common_rule_ids),
                "exact_or_strong_aligned_canonical_rule_count": len(common_rule_ids) - len(unresolved),
                "unresolved_canonical_rule_count": len(unresolved),
                "unresolved_canonical_rule_ids": unresolved,
            },
            "canonical_to_model": used_alignment_entries,
        }
        overlay_refs = [remap_reference(reference, alignment_by_id, model) for reference in references]
        overlay_templates = remap_templates(templates, alignment_by_id, model)
        alignment_audit = {
            "version": "aviation_v2_expansion150_alignment_audit_v1",
            "generated_at": generated_at,
            "model": model,
            "task_count": len(references),
            "task_alignment": [
                {
                    "task_id": reference["omega_id"],
                    "canonical_rule_ids": reference["rule_structure"]["expected_surviving_rule_ids"],
                    "projected_model_rule_ids": remap_reference(reference, alignment_by_id, model)["rule_structure"][
                        "expected_surviving_rule_ids"
                    ],
                    "unresolved_canonical_rule_ids": remap_reference(reference, alignment_by_id, model)[
                        "overlay_metadata"
                    ]["unresolved_canonical_rule_ids"],
                }
                for reference in references
            ],
        }
        write_json(overlay_dir / "rule_id_alignment.json", alignment_payload)
        write_json(
            overlay_dir / "evaluation_references.json",
            {
                "version": "aviation_v2_expansion150_overlay_evaluation_references_v1",
                "model": model,
                "semantic_reference_policy": "Task semantics, feasible_region constraints, and provenance remain canonical/source-grounded; only expected surviving rule IDs are projected into the model namespace.",
                "items": overlay_refs,
            },
        )
        write_json(overlay_dir / "compiled_rule_constraint_templates.json", overlay_templates)
        write_json(overlay_dir / "alignment_audit.json", alignment_audit)
        overlay_manifest["rule_libraries"][model] = str(library_dir / output_name)
        overlay_manifest["evaluation_overlays"][model] = {
            "rule_id_alignment": str(overlay_dir / "rule_id_alignment.json"),
            "evaluation_references": str(overlay_dir / "evaluation_references.json"),
            "compiled_rule_constraint_templates": str(overlay_dir / "compiled_rule_constraint_templates.json"),
            "alignment_audit": str(overlay_dir / "alignment_audit.json"),
        }
        overlay_manifest["alignment_summaries"].append(alignment_payload["summary"])

    challenge_counter = Counter(
        challenge
        for reference in references
        for challenge in reference["rule_structure"].get("challenge_types", [])
    )
    section_counter = Counter(item["scenario_facts"]["document_section_family"] for item in algorithms)
    used_rule_counter = Counter(
        rid
        for reference in references
        for rid in reference["rule_structure"].get("expected_surviving_rule_ids", [])
    )
    common_rules_payload = {
        "version": "aviation_v2_expansion150_common_strong_rules_v1",
        "generated_at": generated_at,
        "policy": "Qwen canonical rules retained only when DeepSeek and Xiaomi MIMO have exact-or-strong semantic alignment.",
        "common_rule_count": len(common_rule_ids),
        "rules": [
            {
                "canonical_rule_id": rid,
                "name": rule_lookup[rid].get("name", ""),
                "task_family": classify_rule(rule_lookup[rid])[0],
                "section": classify_rule(rule_lookup[rid])[1],
                "deepseek_rule_ids": model_ids_for(rid, alignment_by_model["deepseek"]),
                "xiaomi_mimo_rule_ids": model_ids_for(rid, alignment_by_model["xiaomi_mimo"]),
                "used_task_count": used_rule_counter[rid],
            }
            for rid in common_rule_ids
        ],
    }
    write_json(OUT_ROOT / "COMMON_STRONG_RULES.json", common_rules_payload)
    write_json(
        OUT_ROOT / "STRICT_COMMON_TASKS.json",
        {
            "version": "aviation_v2_expansion150_strict_common_tasks_v1",
            "generated_at": generated_at,
            "selection_policy": "Every task is built only from canonical rules with exact-or-strong alignment in Qwen, DeepSeek, and Xiaomi MIMO.",
            "task_count": len(references),
            "task_ids": [reference["omega_id"] for reference in references],
            "excluded_task_ids": [],
        },
    )
    write_json(
        OUT_ROOT / "RELATION_COVERAGE_AUDIT.json",
        {
            "version": "aviation_v2_expansion150_relation_coverage_audit_v1",
            "generated_at": generated_at,
            "task_count": len(references),
            "common_rule_count": len(common_rule_ids),
            "document_section_counts": dict(sorted(section_counter.items())),
            "challenge_type_task_counts": dict(sorted(challenge_counter.items())),
            "rule_usage_counts": dict(sorted(used_rule_counter.items())),
        },
    )
    write_json(OUT_ROOT / "LEAKAGE_AUDIT.json", leakage_audit(algorithms, references))
    write_json(
        OUT_ROOT / "MANIFEST.json",
        {
            "version": "aviation_v2_expansion150_manifest_v1",
            "generated_at": generated_at,
            "purpose": "New paper-round aviation expansion dataset with 150 tasks built from Qwen/DeepSeek/Xiaomi MIMO common strong rules.",
            "dataset_root": str(OUT_ROOT),
            "task_count": len(references),
            "common_rule_count": len(common_rule_ids),
            "core": overlay_manifest["core"],
            "rule_libraries": overlay_manifest["rule_libraries"],
            "evaluation_overlays": overlay_manifest["evaluation_overlays"],
            "audits": {
                "common_strong_rules": str(OUT_ROOT / "COMMON_STRONG_RULES.json"),
                "strict_common_tasks": str(OUT_ROOT / "STRICT_COMMON_TASKS.json"),
                "relation_coverage": str(OUT_ROOT / "RELATION_COVERAGE_AUDIT.json"),
                "leakage": str(OUT_ROOT / "LEAKAGE_AUDIT.json"),
            },
        },
    )
    write_json(OUT_ROOT / "OVERLAY_MANIFEST.json", overlay_manifest)
    (OUT_ROOT / "README.md").write_text(
        "\n".join(
            [
                "# Aviation Full-KG V2 Expansion 150",
                "",
                "This folder contains 150 newly generated aviation procedure-design benchmark tasks for the next paper-round dataset.",
                "",
                "The task pool is built only from Qwen canonical aviation rules that have exact-or-strong semantic alignment in both DeepSeek and Xiaomi MIMO rule libraries.",
                "",
                "Algorithms should read only `core/algorithm_inputs`, `core/scenario_models`, and the selected model rule library. Evaluation references and overlays are hidden oracle artifacts.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "dataset_root": str(OUT_ROOT),
                "task_count": len(references),
                "common_rule_count": len(common_rule_ids),
                "document_section_counts": dict(sorted(section_counter.items())),
                "challenge_type_counts": dict(sorted(challenge_counter.items())),
                "leakage_hits": leakage_audit(algorithms, references)["forbidden_input_key_hit_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
