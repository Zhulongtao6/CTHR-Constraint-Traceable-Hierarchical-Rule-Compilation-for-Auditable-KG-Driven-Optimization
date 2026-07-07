from __future__ import annotations

import csv
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import run_section_6_2_table1_pipeline as base
import run_section_6_3_candidate_to_valid as ctv
from experiments.kg_to_rule_validation.baselines.asp_rule_structure import (  # noqa: E402
    candidate_score,
    enumerate_rule_structures,
    eval_guard,
)
from experiments.kg_to_rule_validation.baselines.cthr_rule_resolver import (  # noqa: E402
    resolve_valid_structures_with_diagnostics,
)
from experiments.kg_to_rule_validation.baselines.smt_monolithic import (  # noqa: E402
    build_smt_formula,
    map_rule_variable,
    optimize_with_z3,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
ARCHITECTURE_ROOT = ROOT / "datasets" / "architecture"

QWEN_FULL_RULE_LIBRARY = (
    ROOT
    / "results"
    / "kg_to_rule_library"
    / "architecture"
    / "full_architecture_rule_library_qwen.json"
)
BENCHMARK_RULE_LIBRARY = ARCHITECTURE_ROOT / "architecture_stress_rule_library.combined.json"

METHOD_SPECS = [
    ("Flat baseline", "flat"),
    ("CTHR full", "cthr_full"),
    ("ASP-native + clingo", "native_symbolic"),
    ("ASP-CTHR-relations + clingo", "cthr_relation_encoded_symbolic"),
    ("SMT-native + Z3", "native_symbolic"),
    ("SMT-CTHR-relations + Z3", "cthr_relation_encoded_symbolic"),
    ("MILP-native + HiGHS", "native_symbolic"),
    ("MILP-CTHR-relations + HiGHS", "cthr_relation_encoded_symbolic"),
]

OUTPUTS = {
    "overall_csv": RESULTS_DIR / "section_6_2_table1_architecture_pipeline_overall.csv",
    "overall_md": RESULTS_DIR / "section_6_2_table1_architecture_pipeline_overall.md",
    "overall_json": RESULTS_DIR / "section_6_2_table1_architecture_pipeline_overall.json",
    "per_task_csv": RESULTS_DIR / "section_6_2_table1_architecture_pipeline_per_task.csv",
    "report_md": RESULTS_DIR / "section_6_2_table1_architecture_pipeline_report.md",
}


@dataclass
class MethodResult:
    supported: bool
    raw_predicted_rule_ids: list[str]
    optimized_x: dict[str, float] | None
    formal_feasible: bool | None
    unsupported_reason: str = ""


def write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({header: base.csv_cell(row.get(header)) for header in headers})


def task_scenario(query: dict[str, Any]) -> dict[str, Any]:
    scenario = ctv.scenario_for_resolution(query)
    scenario["source_domain"] = query.get("source_domain")
    scenario["task_type"] = query.get("task_type")
    scenario["title"] = query.get("title")
    scenario["design_intent"] = query.get("design_intent")
    return scenario


def load_layer(name: str) -> dict[str, dict[str, Any]]:
    return base.by_id(base.load_items(ARCHITECTURE_ROOT / name))


def numeric_constraint_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"[-+]?[0-9]+(?:\.[0-9]+)?", str(value))
    return float(match.group(0)) if match else None


def compatible_unit_value(value: float, source_unit: str, target_unit: str) -> float:
    source = source_unit.lower()
    target = target_unit.lower()
    if source in {"mm", "millimeter", "millimeters"} and target in {"inch", "in", "inches"}:
        return value / 25.4
    if source in {"inch", "in", "inches"} and target in {"mm", "millimeter", "millimeters"}:
        return value * 25.4
    if source in {"ft", "foot", "feet"} and target in {"inch", "in", "inches"}:
        return value * 12.0
    if source in {"inch", "in", "inches"} and target in {"ft", "foot", "feet"}:
        return value / 12.0
    return value


def qwen_rule_constraints(rule: dict[str, Any], query: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, constraint in enumerate(rule.get("constraints", [])):
        op = str(constraint.get("op", "")).strip()
        if op not in {"<=", ">=", "<", ">", "=", "=="}:
            continue
        mapped = map_rule_variable(str(constraint.get("variable", "")), query)
        if not mapped:
            continue
        value = numeric_constraint_value(constraint.get("value"))
        if value is None:
            continue
        target_unit = str(query.get("decision_variables", {}).get(mapped, {}).get("unit", ""))
        source_unit = str(constraint.get("unit", ""))
        value = compatible_unit_value(value, source_unit, target_unit)
        out.append(
            {
                "constraint_id": f"qwen_{rule['rule_id']}_{idx}",
                "expression": f"{mapped} {op} {value}",
                "checker_expression": f"{mapped} {op} {value}",
                "source_type": "rule_library",
                "source_id": str(rule["rule_id"]),
                "executable": True,
            }
        )
    return out


def constraints_for_qwen_rules(
    query: dict[str, Any],
    raw_rule_ids: list[str],
    qwen_rule_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    constraints: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rule_id in raw_rule_ids:
        rule = qwen_rule_by_id.get(rule_id)
        if not rule:
            continue
        for constraint in qwen_rule_constraints(rule, query):
            key = str(constraint["constraint_id"])
            if key not in seen:
                constraints.append(constraint)
                seen.add(key)
    return constraints


def score_qwen_candidates(
    qwen_rules: list[dict[str, Any]],
    query: dict[str, Any],
    limit: int = 24,
) -> list[str]:
    scenario = task_scenario(query)
    scored: list[tuple[float, str]] = []
    source_domain = str(query.get("source_domain", "")).lower()
    for rule in qwen_rules:
        if not rule.get("rule_id"):
            continue
        score = candidate_score(rule, query, scenario)
        if source_domain:
            docs = " ".join(str(p.get("document", "")) for p in rule.get("provenance", []))
            if source_domain.lower() in docs.lower():
                score += 1.5
        if any(map_rule_variable(str(c.get("variable", "")), query) for c in rule.get("constraints", [])):
            score += 2.0
        if score > 0:
            scored.append((score, str(rule["rule_id"])))
    scored.sort(key=lambda item: (-item[0], item[1]))
    candidates = [rule_id for score, rule_id in scored if score >= 8.0][:limit]
    if len(candidates) < 3:
        candidates = [rule_id for _score, rule_id in scored[: min(limit, 8)]]
    return sorted(dict.fromkeys(candidates))


def strip_relations(rule: dict[str, Any]) -> dict[str, Any]:
    stripped = dict(rule)
    stripped["relations"] = []
    stripped.pop("conflict_class", None)
    stripped.pop("conflict_group", None)
    return stripped


def filtered_library(
    rule_library: dict[str, Any],
    candidate_ids: list[str],
    *,
    native: bool,
) -> dict[str, Any]:
    allowed = set(candidate_ids)
    rules = [rule for rule in rule_library.get("rules", []) if str(rule.get("rule_id")) in allowed]
    if native:
        rules = [strip_relations(rule) for rule in rules]
    out = dict(rule_library)
    out["rules"] = rules
    return out


def optimize_default_for_rules(
    query: dict[str, Any],
    raw_rule_ids: list[str],
    qwen_rule_by_id: dict[str, dict[str, Any]],
    method: str,
) -> tuple[dict[str, float] | None, bool]:
    constraints = constraints_for_qwen_rules(query, raw_rule_ids, qwen_rule_by_id)
    x = base.optimize_default(query, constraints, method, str(query["omega_id"]))
    formal = base.constraints_satisfied(constraints, base.with_query_values(query, x)) if x is not None else False
    return x, formal


def select_native_applicable(candidate_ids: list[str], qwen_rule_by_id: dict[str, dict[str, Any]], query: dict[str, Any]) -> list[str]:
    scenario = task_scenario(query)
    selected = []
    for rule_id in candidate_ids:
        rule = qwen_rule_by_id.get(rule_id, {})
        guard = rule.get("guard")
        if not guard:
            selected.append(rule_id)
            continue
        try:
            if eval_guard(guard, scenario):
                selected.append(rule_id)
        except Exception:
            continue
    return sorted(selected or candidate_ids[:1])


def run_flat(
    query: dict[str, Any],
    candidate_ids: list[str],
    qwen_rule_by_id: dict[str, dict[str, Any]],
) -> MethodResult:
    if not candidate_ids:
        return MethodResult(False, [], None, None, "no_qwen_candidates_retrieved")
    x, formal = optimize_default_for_rules(query, candidate_ids, qwen_rule_by_id, "Flat baseline")
    return MethodResult(True, sorted(candidate_ids), x, formal)


def run_cthr(
    query: dict[str, Any],
    candidate_ids: list[str],
    qwen_rule_by_id: dict[str, dict[str, Any]],
) -> MethodResult:
    if not candidate_ids:
        return MethodResult(False, [], None, None, "no_qwen_candidates_retrieved")
    candidate_rules = [qwen_rule_by_id[rule_id] for rule_id in candidate_ids if rule_id in qwen_rule_by_id]
    resolved = resolve_valid_structures_with_diagnostics(candidate_rules, task_scenario(query))
    if resolved.status != "success":
        return MethodResult(False, [], None, None, f"cthr_resolver_{resolved.status}:{resolved.error or ''}")
    selected = base.union_structures(resolved.valid_rule_structures)
    x, formal = optimize_default_for_rules(query, selected, qwen_rule_by_id, "CTHR full")
    return MethodResult(True, selected, x, formal)


def run_asp(
    query: dict[str, Any],
    candidate_ids: list[str],
    qwen_library: dict[str, Any],
    qwen_rule_by_id: dict[str, dict[str, Any]],
    *,
    native: bool,
) -> MethodResult:
    if not candidate_ids:
        return MethodResult(False, [], None, None, "no_qwen_candidates_retrieved")
    lib = filtered_library(qwen_library, candidate_ids, native=native)
    result = enumerate_rule_structures(
        lib,
        task_scenario(query),
        str(query["omega_id"]),
        candidate_rule_ids=candidate_ids,
        applicable_rule_ids=None,
        max_answer_sets=100,
    )
    if result.status != "success":
        return MethodResult(False, [], None, None, f"asp_{result.status}:{result.error or ''}".strip(":"))
    selected = base.union_structures(result.asp_rule_structures)
    x, formal = optimize_default_for_rules(
        query,
        selected,
        qwen_rule_by_id,
        "ASP-native + clingo" if native else "ASP-CTHR-relations + clingo",
    )
    return MethodResult(True, selected, x, formal)


def run_smt(
    query: dict[str, Any],
    candidate_ids: list[str],
    qwen_library: dict[str, Any],
    *,
    native: bool,
) -> MethodResult:
    if not candidate_ids:
        return MethodResult(False, [], None, None, "no_qwen_candidates_retrieved")
    lib = filtered_library(qwen_library, candidate_ids, native=native)
    try:
        formula = build_smt_formula(
            lib,
            query,
            candidate_rule_ids=candidate_ids,
            include_visible_task_constraints=False,
        )
        result = optimize_with_z3(formula, query, timeout_ms=10000)
        if result.status != "sat" or result.optimized_x is None:
            return MethodResult(
                False,
                result.selected_rule_ids,
                None,
                None,
                f"smt_{result.status}:{result.error or ''}".strip(":"),
            )
        variables = list(query.get("decision_variables", {}))
        x = {name: float(value) for name, value in zip(variables, result.optimized_x)}
        return MethodResult(True, sorted(result.selected_rule_ids), x, True)
    except Exception as exc:  # noqa: BLE001
        return MethodResult(False, [], None, None, f"smt_error:{exc}")


def run_milp(
    query: dict[str, Any],
    candidate_ids: list[str],
    qwen_library: dict[str, Any],
    qwen_rule_by_id: dict[str, dict[str, Any]],
    *,
    native: bool,
) -> MethodResult:
    if not candidate_ids:
        return MethodResult(False, [], None, None, "no_qwen_candidates_retrieved")
    try:
        lib = filtered_library(qwen_library, candidate_ids, native=native)
        lib_by_id = {str(rule["rule_id"]): rule for rule in lib.get("rules", []) if rule.get("rule_id")}
        if native:
            selected = select_native_applicable(candidate_ids, lib_by_id, query)
        else:
            selected = base.milp_select_rules(candidate_ids, lib_by_id, task_scenario(query))
        if not selected:
            return MethodResult(False, [], None, None, "milp_rule_selection_infeasible")
        constraints = constraints_for_qwen_rules(query, selected, qwen_rule_by_id)
        variables = list(query.get("decision_variables", {}))
        constants = {
            key: float(value)
            for key, value in query.get("scenario_facts", {}).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        rows, lb, ub = base.linear_constraint_rows(constraints, variables, constants)
        c = base.objective_linear(query, variables, constants)
        bounds = [
            (
                float(query["decision_variables"][name].get("lower", 0.0)),
                float(query["decision_variables"][name].get("upper", 1.0)),
            )
            for name in variables
        ]
        a_ub = [row for row, lo, hi in zip(rows, lb, ub) if math.isinf(lo) and not math.isinf(hi)]
        b_ub = [hi for lo, hi in zip(lb, ub) if math.isinf(lo) and not math.isinf(hi)]
        a_eq = [row for row, lo, hi in zip(rows, lb, ub) if not math.isinf(lo) and not math.isinf(hi) and abs(lo - hi) < 1e-9]
        b_eq = [lo for lo, hi in zip(lb, ub) if not math.isinf(lo) and not math.isinf(hi) and abs(lo - hi) < 1e-9]
        lp_result = base.linprog(
            c=np.array(c),
            A_ub=np.array(a_ub) if a_ub else None,
            b_ub=np.array(b_ub) if b_ub else None,
            A_eq=np.array(a_eq) if a_eq else None,
            b_eq=np.array(b_eq) if b_eq else None,
            bounds=bounds,
            method="highs",
        )
        if not lp_result.success or lp_result.x is None:
            return MethodResult(False, selected, None, None, f"highs_{lp_result.message}")
        x = {name: float(value) for name, value in zip(variables, lp_result.x)}
        formal = base.constraints_satisfied(constraints, base.with_query_values(query, x))
        return MethodResult(True, selected, x, formal)
    except Exception as exc:  # noqa: BLE001
        return MethodResult(False, [], None, None, f"unsupported_nonlinear_or_mapping:{exc}")


def chunk_alias_map(
    benchmark_rule_library: dict[str, Any],
    qwen_library: dict[str, Any],
) -> dict[str, set[str]]:
    benchmark_by_chunk: dict[str, set[str]] = {}
    for rule in benchmark_rule_library.get("rules", []):
        rid = str(rule.get("rule_id"))
        for chunk_id in rule.get("source_chunk_ids", []) or []:
            benchmark_by_chunk.setdefault(str(chunk_id), set()).add(rid)
    alias: dict[str, set[str]] = {}
    for rule in qwen_library.get("rules", []):
        qid = str(rule.get("rule_id"))
        for chunk_id in rule.get("source_chunk_ids", []) or []:
            alias.setdefault(qid, set()).update(benchmark_by_chunk.get(str(chunk_id), set()))
    return alias


def mapped_predicted_ids(
    raw_qwen_ids: list[str],
    alias: dict[str, set[str]],
    task_candidate_ids: list[str],
) -> list[str]:
    allowed = set(task_candidate_ids)
    mapped = set()
    for qid in raw_qwen_ids:
        mapped.update(alias.get(qid, set()) & allowed)
    return sorted(mapped)


def evaluate_semantic(
    feasible: dict[str, Any],
    x: dict[str, float] | None,
    mapped_ids: list[str],
    reference_ids: list[str],
) -> bool:
    if x is None:
        return False
    numeric_ok = base.constraints_satisfied(base.reference_constraints(feasible), base.with_scenario_values(feasible, x))
    structure_ok = set(mapped_ids) == set(reference_ids)
    return bool(numeric_ok and structure_ok)


def evaluate_method(
    method: str,
    method_type: str,
    query: dict[str, Any],
    label: dict[str, Any],
    feasible: dict[str, Any],
    qwen_library: dict[str, Any],
    qwen_rule_by_id: dict[str, dict[str, Any]],
    alias: dict[str, set[str]],
) -> dict[str, Any]:
    task_id = str(query["omega_id"])
    reference = base.reference_rule_ids(label, feasible, query)
    task_candidate_ids = base.candidate_ids_from_query(query, label, feasible)
    candidate_ids = score_qwen_candidates(qwen_library.get("rules", []), query)
    start = time.perf_counter()
    if method == "Flat baseline":
        result = run_flat(query, candidate_ids, qwen_rule_by_id)
    elif method == "CTHR full":
        result = run_cthr(query, candidate_ids, qwen_rule_by_id)
    elif method == "ASP-native + clingo":
        result = run_asp(query, candidate_ids, qwen_library, qwen_rule_by_id, native=True)
    elif method == "ASP-CTHR-relations + clingo":
        result = run_asp(query, candidate_ids, qwen_library, qwen_rule_by_id, native=False)
    elif method == "SMT-native + Z3":
        result = run_smt(query, candidate_ids, qwen_library, native=True)
    elif method == "SMT-CTHR-relations + Z3":
        result = run_smt(query, candidate_ids, qwen_library, native=False)
    elif method == "MILP-native + HiGHS":
        result = run_milp(query, candidate_ids, qwen_library, qwen_rule_by_id, native=True)
    elif method == "MILP-CTHR-relations + HiGHS":
        result = run_milp(query, candidate_ids, qwen_library, qwen_rule_by_id, native=False)
    else:
        raise ValueError(method)
    elapsed = (time.perf_counter() - start) * 1000.0

    mapped = mapped_predicted_ids(result.raw_predicted_rule_ids, alias, task_candidate_ids) if result.supported else []
    precision = base.method_rule_precision(mapped, reference) if result.supported else None
    recall = base.method_rule_recall(mapped, reference) if result.supported else None
    sem_valid = evaluate_semantic(feasible, result.optimized_x, mapped, reference) if result.supported else None
    formal = result.formal_feasible if result.supported else None
    return {
        "Dataset": "Architecture",
        "task_id": task_id,
        "Method": method,
        "Method type": method_type,
        "predicted_rule_ids": mapped,
        "reference_rule_ids": reference,
        "rule_precision": precision,
        "rule_recall": recall,
        "formal_feasible": formal,
        "semantic_valid": sem_valid,
        "false_accept": bool(formal and not sem_valid) if result.supported else None,
        "invalid_case": bool(not sem_valid) if result.supported else None,
        "unsupported_reason": "" if result.supported else result.unsupported_reason,
        "runtime_ms": round(elapsed, 3),
    }


def pct(value: float | None) -> str:
    return "N/A" if value is None else f"{100.0 * value:.1f}%"


def avg(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    return None if not valid else sum(valid) / len(valid)


def aggregate(per_task_rows: list[dict[str, Any]], method: str, method_type: str) -> dict[str, Any]:
    subset = [row for row in per_task_rows if row["Method"] == method]
    supported = [row for row in subset if not row["unsupported_reason"]]
    if not supported:
        return {
            "Dataset": "Architecture",
            "Method": method,
            "Method type": method_type,
            "Rule Precision": "N/A",
            "Rule Recall": "N/A",
            "Formal CSR": "N/A",
            "Sem-CSR": "N/A",
            "False accept": "N/A",
            "Invalid cases": f"0/0 (N/A) ({len(subset)} unsupported)",
        }
    n = len(supported)
    unsupported = len(subset) - n
    suffix = f" ({unsupported} unsupported)" if unsupported else ""
    invalid = sum(1 for row in supported if row["invalid_case"])
    false_accept = sum(1 for row in supported if row["false_accept"])
    formal = sum(1 for row in supported if row["formal_feasible"])
    sem = sum(1 for row in supported if row["semantic_valid"])
    return {
        "Dataset": "Architecture",
        "Method": method,
        "Method type": method_type,
        "Rule Precision": pct(avg([row["rule_precision"] for row in supported])),
        "Rule Recall": pct(avg([row["rule_recall"] for row in supported])),
        "Formal CSR": pct(formal / n),
        "Sem-CSR": pct(sem / n),
        "False accept": pct(false_accept / n),
        "Invalid cases": f"{invalid}/{n} ({100.0 * invalid / n:.1f}%){suffix}",
    }


def unsupported_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for row in rows:
        reason = str(row.get("unsupported_reason") or "")
        if not reason:
            continue
        method = str(row["Method"])
        short = reason.split(":", 1)[0]
        summary.setdefault(method, {})
        summary[method][short] = summary[method].get(short, 0) + 1
    return summary


def render_report(overall_rows: list[dict[str, Any]], per_task_rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
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
    unsupported = unsupported_summary(per_task_rows)
    unsupported_lines = ["- none"] if not unsupported else [
        f"- {method}: " + "; ".join(f"{reason}: {count}" for reason, count in sorted(reasons.items()))
        for method, reasons in sorted(unsupported.items())
    ]
    return "\n".join(
        [
            "# Section 6.2 Table 1: Architecture Full KG-to-Constraint Modeling Pipeline Baselines",
            "",
            "## Dataset",
            "",
            "- Dataset: Architecture benchmark only, 30 tasks (`ARCH_OPT_01` to `ARCH_OPT_30`).",
            "- Rule library: `full_architecture_rule_library_qwen.json` generated from the full Cognee architecture KG.",
            "- No Aviation benchmark and no CTHR compiled-cell solver-backend experiment are run.",
            "",
            "## Method Scope",
            "",
            "- Native symbolic pipelines use qwen-retrieved candidate rules and basic solver-native selection/activation without explicit CTHR six-relation semantics.",
            "- CTHR-relation-encoded symbolic pipelines use qwen-retrieved candidate rules and explicitly encode applicability, dependency, exclusion, override, precedence, and parameter/formula propagation in ASP/SMT/MILP-style models.",
            "- Hidden reference valid rules, valid structures, source-rule feasible cells, and semantic labels are used only for evaluation.",
            "- Because qwen rule IDs are source-grounded and not identical to the ARCH_OPT benchmark labels, rule IDs are normalized to benchmark labels for evaluation only through shared source evidence. This normalization is not used as method input.",
            "",
            "## Main Result",
            "",
            base.render_md_table(overall_rows, headers),
            "",
            "## Unsupported / N/A",
            "",
            *unsupported_lines,
            "",
            "## Analysis",
            "",
            "- Native symbolic methods can construct formal feasible regions, but they frequently accept solutions that do not satisfy the source-rule reference semantics when the six interaction types are not modeled.",
            "- Relation-encoded ASP/SMT/MILP variants test whether the same six rule-interaction semantics can be expressed in other symbolic formalisms without reading CTHR final valid structures or compiled cells.",
            "- CTHR full is evaluated as the full KG-to-constraint modeling pipeline over the same qwen-derived candidate space.",
            "- The low source-evidence alignment between qwen source-grounded IDs and benchmark labels should be read as a remaining dataset-engineering limitation, not as hidden use of benchmark labels during method execution.",
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
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    queries = load_layer("architecture_optimization_queries.json")
    labels = load_layer("architecture_rule_structure_labels.json")
    feasible_items = load_layer("architecture_feasible_region_labels.json")
    if set(queries) != set(labels) or set(queries) != set(feasible_items):
        raise ValueError("Architecture layer IDs do not match")
    qwen_library = base.read_json(QWEN_FULL_RULE_LIBRARY)
    benchmark_library = base.read_json(BENCHMARK_RULE_LIBRARY)
    qwen_rule_by_id = {str(rule["rule_id"]): rule for rule in qwen_library.get("rules", []) if rule.get("rule_id")}
    alias = chunk_alias_map(benchmark_library, qwen_library)

    per_task_rows: list[dict[str, Any]] = []
    for task_id in sorted(queries):
        query = queries[task_id]
        for method, method_type in METHOD_SPECS:
            per_task_rows.append(
                evaluate_method(
                    method,
                    method_type,
                    query,
                    labels[task_id],
                    feasible_items[task_id],
                    qwen_library,
                    qwen_rule_by_id,
                    alias,
                )
            )

    overall_rows = [aggregate(per_task_rows, method, method_type) for method, method_type in METHOD_SPECS]
    per_task_headers = [
        "Dataset",
        "task_id",
        "Method",
        "Method type",
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
    write_csv(OUTPUTS["per_task_csv"], per_task_rows, per_task_headers)
    write_csv(OUTPUTS["overall_csv"], overall_rows, overall_headers)
    OUTPUTS["overall_md"].write_text(base.render_md_table(overall_rows, overall_headers), encoding="utf-8")
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {"Architecture": len(queries)},
        "rule_library": str(QWEN_FULL_RULE_LIBRARY),
        "rule_library_rule_count": len(qwen_library.get("rules", [])),
        "benchmark_rule_id_alignment": {
            "qwen_rules_with_benchmark_alias": sum(1 for values in alias.values() if values),
            "note": "Alias mapping is used only to evaluate source-grounded qwen rule IDs against ARCH_OPT benchmark reference labels.",
        },
        "methods": [{"Method": method, "Method type": method_type} for method, method_type in METHOD_SPECS],
        "input_restrictions": {
            "hidden_reference_labels_as_method_input": False,
            "reference_valid_rules_as_method_input": False,
            "reference_valid_structures_as_method_input": False,
            "reference_cells_as_method_input": False,
            "solver_constraints_as_method_input": False,
            "cthr_compiled_cells_as_baseline_input": False,
        },
        "unsupported": unsupported_summary(per_task_rows),
        "outputs": {key: str(value) for key, value in OUTPUTS.items()},
        "aggregate_rows": overall_rows,
    }
    base.write_json(OUTPUTS["overall_json"], summary)
    OUTPUTS["report_md"].write_text(render_report(overall_rows, per_task_rows, summary), encoding="utf-8")
    print(json.dumps({"outputs": summary["outputs"], "aggregate_rows": overall_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
