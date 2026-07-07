from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


CTHR_ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = CTHR_ROOT / "paper"
INPUT_PATH = PAPER_DIR / "aviation_kg_generated_19_optimization_problems.json"
OUT_DIR = PAPER_DIR / "aviation_benchmark_layers"

RULE_LABELS_PATH = OUT_DIR / "aviation_rule_structure_labels.json"
FEASIBLE_LABELS_PATH = OUT_DIR / "aviation_feasible_region_labels.json"
OPT_QUERIES_PATH = OUT_DIR / "aviation_optimization_queries.json"
MANIFEST_PATH = OUT_DIR / "aviation_benchmark_layers_manifest.json"
SUMMARY_PATH = OUT_DIR / "AVIATION_BENCHMARK_LAYERS_SUMMARY.md"


NON_EXECUTABLE_MARKERS = (
    " is defeated ",
    "follows ",
    " grid",
    "branch_guard",
    "guard",
)

COMPARATOR_PATTERN = re.compile(r"(<=|>=|!=|==|=|<|>)")
IDENTIFIER_PATTERN = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
FUNCTION_NAMES = {"abs", "max", "min"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def source_ref(constraint: dict[str, Any]) -> dict[str, str]:
    if "source_rule_id" in constraint:
        return {"source_type": "rule_library", "source_id": str(constraint["source_rule_id"])}
    return {"source_type": "task_or_scenario_model", "source_id": str(constraint.get("source", "unknown"))}


def is_executable_expression(expression: str, role: str) -> bool:
    expr = f" {expression.strip().lower()} "
    role_l = role.lower()
    if not COMPARATOR_PATTERN.search(expression):
        return False
    if any(marker in expr for marker in NON_EXECUTABLE_MARKERS):
        return False
    if any(marker in role_l for marker in NON_EXECUTABLE_MARKERS):
        return False
    if "defeated" in expr or "defeated" in role_l:
        return False
    return True


def to_checker_expression(expression: str) -> str:
    """Convert paper-style equality into a Python-evaluable predicate string."""
    expression = expression.strip()
    expression = re.sub(r"(?<![<>=!])=(?!=)", "==", expression)
    return expression


def identifiers_in_expression(expression: str, variables: dict[str, Any], scenario_facts: dict[str, Any]) -> dict[str, list[str]]:
    names = set(IDENTIFIER_PATTERN.findall(expression))
    names -= FUNCTION_NAMES
    return {
        "decision_variables": sorted(name for name in names if name in variables),
        "scenario_fields": sorted(name for name in names if name in scenario_facts),
        "unresolved_symbols": sorted(name for name in names if name not in variables and name not in scenario_facts),
    }


def normalize_constraint(
    constraint: dict[str, Any],
    variables: dict[str, Any],
    scenario_facts: dict[str, Any],
) -> dict[str, Any]:
    expression = str(constraint.get("expression", "")).strip()
    role = str(constraint.get("role", ""))
    executable = is_executable_expression(expression, role)
    item: dict[str, Any] = {
        "constraint_id": constraint.get("constraint_id"),
        "expression": expression,
        "role": role,
        **source_ref(constraint),
        "executable": executable,
    }
    if executable:
        item.update(
            {
                "checker_expression": to_checker_expression(expression),
                "expression_language": "python_safe_arithmetic_predicate",
                "symbols": identifiers_in_expression(expression, variables, scenario_facts),
            }
        )
    else:
        item["reason_not_executable"] = infer_non_executable_reason(expression, role)
    return item


def infer_non_executable_reason(expression: str, role: str) -> str:
    text = f"{expression} {role}".lower()
    if "defeated" in text:
        return "rule-structure label: defeated baseline rule"
    if "follows" in text or "grid" in text:
        return "discrete grid constraint requiring a specialized integer/rounding encoder"
    if "guard" in text:
        return "scenario guard used for applicability, not a numeric decision constraint"
    return "not represented as a direct numeric predicate"


def classify_challenges(expected: dict[str, Any], constraints: list[dict[str, Any]]) -> list[str]:
    expected_values: list[str] = []
    for value in expected.values():
        if isinstance(value, list):
            expected_values.extend(str(item) for item in value)
        else:
            expected_values.append(str(value))
    blob = " ".join(expected_values).lower() + " " + json.dumps(constraints, ensure_ascii=False).lower()
    labels: set[str] = set()
    if any(word in blob for word in ("exception", "override", "defeated", "replaces")):
        labels.add("exception_or_override")
    if any(word in blob for word in ("branch", "mutual", "exclusion", "straight", "turning", "helicopter", "fixed-wing", "primary/secondary")):
        labels.add("branch_or_exclusion")
    if any(word in blob for word in ("depends", "dependency", "formula", "propagation")):
        labels.add("dependency_or_formula_propagation")
    if any(word in blob for word in ("provenance", "source-document", "document-level")):
        labels.add("provenance_traceability")
    if any(word in blob for word in ("condition", "applicability", "altitude-conditioned", "navigation system", "segment type")):
        labels.add("scenario_conditioned_applicability")
    return sorted(labels) or ["generic_rule_selection"]


def defeated_rule_ids(constraints: list[dict[str, Any]]) -> list[str]:
    ids = []
    for constraint in constraints:
        role = str(constraint.get("role", "")).lower()
        expr = str(constraint.get("expression", "")).lower()
        if "defeated" in role or "defeated" in expr:
            rid = constraint.get("source_rule_id")
            if rid:
                ids.append(str(rid))
    return sorted(set(ids))


def split_constraints(
    constraints: list[dict[str, Any]],
    variables: dict[str, Any],
    scenario_facts: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    executable = []
    structure_only = []
    for constraint in constraints:
        item = normalize_constraint(constraint, variables, scenario_facts)
        if item["executable"]:
            executable.append(item)
        else:
            structure_only.append(item)
    return executable, structure_only


def normalize_valid_constraint_cells(
    cells: list[dict[str, Any]],
    variables: dict[str, Any],
    scenario_facts: dict[str, Any],
) -> list[dict[str, Any]]:
    normalized = []
    for cell in cells:
        executable, structure_only = split_constraints(
            cell.get("constraints", []),
            variables,
            scenario_facts,
        )
        normalized.append(
            {
                "cell_id": cell.get("cell_id"),
                "description": cell.get("description", ""),
                "executable_constraints": executable,
                "structure_only_constraints": structure_only,
            }
        )
    return normalized


def build_rule_structure_label(problem: dict[str, Any]) -> dict[str, Any]:
    hidden = problem["hidden_evaluation_reference"]
    constraints = hidden.get("kg_grounded_constraints", [])
    evidence = hidden.get("evidence", {})
    defeated = defeated_rule_ids(constraints)
    source_rules = [str(x) for x in hidden.get("source_rule_ids", [])]
    return {
        "omega_id": problem["omega_id"],
        "title": problem["title"],
        "domain": problem["domain"],
        "task_type": problem["task_type"],
        "scenario_facts": problem["visible_decision_query"]["scenario_facts"],
        "expected_source_rule_ids": source_rules,
        "expected_defeated_rule_ids": defeated,
        "expected_surviving_rule_ids": sorted(rid for rid in source_rules if rid not in set(defeated)),
        "expected_rule_behavior": hidden.get("expected_rule_behavior", {}),
        "challenge_types": classify_challenges(hidden.get("expected_rule_behavior", {}), constraints),
        "valid_constraint_cell_ids": [cell.get("cell_id") for cell in hidden.get("valid_constraint_cells", [])],
        "expected_provenance": {
            "kg_chunk_ids": evidence.get("kg_chunk_ids", []),
            "kg_node_ids": evidence.get("kg_node_ids", []),
            "kg_edge_ids": evidence.get("kg_edge_ids", []),
            "source_documents": evidence.get("provenance", []),
        },
    }


def build_feasible_region_label(problem: dict[str, Any]) -> dict[str, Any]:
    visible = problem["visible_decision_query"]
    hidden = problem["hidden_evaluation_reference"]
    executable, structure_only = split_constraints(
        hidden.get("kg_grounded_constraints", []),
        visible["decision_variables"],
        visible["scenario_facts"],
    )
    valid_cells = normalize_valid_constraint_cells(
        hidden.get("valid_constraint_cells", []),
        visible["decision_variables"],
        visible["scenario_facts"],
    )
    return {
        "omega_id": problem["omega_id"],
        "title": problem["title"],
        "scenario_facts": visible["scenario_facts"],
        "decision_variables": visible["decision_variables"],
        "executable_constraints": executable,
        "structure_only_constraints": structure_only,
        "valid_constraint_cells": valid_cells,
        "reference_semantics": {
            "positive_membership_condition": (
                "all executable_constraints evaluate true and, when valid_constraint_cells are present, "
                "at least one cell's executable constraints also evaluate true"
            ),
            "structure_only_constraints_usage": "used to check rule resolution, provenance, or specialized encoders before numeric membership checking",
        },
    }


def build_optimization_query(problem: dict[str, Any]) -> dict[str, Any]:
    visible = problem["visible_decision_query"]
    hidden = problem["hidden_evaluation_reference"]
    executable, structure_only = split_constraints(
        hidden.get("kg_grounded_constraints", []),
        visible["decision_variables"],
        visible["scenario_facts"],
    )
    valid_cells = normalize_valid_constraint_cells(
        hidden.get("valid_constraint_cells", []),
        visible["decision_variables"],
        visible["scenario_facts"],
    )
    return {
        "omega_id": problem["omega_id"],
        "title": problem["title"],
        "domain": problem["domain"],
        "task_type": problem["task_type"],
        "design_intent": visible["design_intent"],
        "scenario_facts": visible["scenario_facts"],
        "decision_variables": visible["decision_variables"],
        "objectives": visible["objectives"],
        "query_preferences": visible["query_preferences"],
        "solver_constraints": executable,
        "solver_constraint_cells": valid_cells,
        "pre_solver_structure_checks": structure_only,
        "certificate_targets": {
            "source_rule_ids": hidden.get("source_rule_ids", []),
            "provenance": hidden.get("evidence", {}).get("provenance", []),
        },
    }


def build_summary(rule_labels: list[dict[str, Any]], feasible_labels: list[dict[str, Any]], opt_queries: list[dict[str, Any]]) -> str:
    num_executable = sum(len(item["executable_constraints"]) for item in feasible_labels)
    num_structure_only = sum(len(item["structure_only_constraints"]) for item in feasible_labels)
    num_cells = sum(len(item.get("valid_constraint_cells", [])) for item in feasible_labels)
    challenge_counts: dict[str, int] = {}
    for item in rule_labels:
        for label in item["challenge_types"]:
            challenge_counts[label] = challenge_counts.get(label, 0) + 1

    lines = [
        "# Aviation Benchmark Layers",
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
    rule_labels = [build_rule_structure_label(problem) for problem in problems]
    feasible_labels = [build_feasible_region_label(problem) for problem in problems]
    opt_queries = [build_optimization_query(problem) for problem in problems]

    manifest = {
        "version": "aviation_benchmark_layers_v1",
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
                "purpose": "Evaluate active/defeated/excluded rules, challenge type, and provenance validity.",
            },
            {
                "name": "feasible_region_semantics",
                "file": str(FEASIBLE_LABELS_PATH),
                "purpose": "Evaluate membership consistency against resolved executable constraints.",
            },
            {
                "name": "constrained_optimization",
                "file": str(OPT_QUERIES_PATH),
                "purpose": "Run solvers on task objectives under CTHR-compiled feasible regions.",
            },
        ],
    }

    write_json(RULE_LABELS_PATH, {"version": manifest["version"], "items": rule_labels})
    write_json(FEASIBLE_LABELS_PATH, {"version": manifest["version"], "items": feasible_labels})
    write_json(OPT_QUERIES_PATH, {"version": manifest["version"], "items": opt_queries})
    write_json(MANIFEST_PATH, manifest)
    SUMMARY_PATH.write_text(build_summary(rule_labels, feasible_labels, opt_queries), encoding="utf-8")

    print(json.dumps({
        "out_dir": str(OUT_DIR),
        "num_problems": len(problems),
        "rule_structure_labels": str(RULE_LABELS_PATH),
        "feasible_region_labels": str(FEASIBLE_LABELS_PATH),
        "optimization_queries": str(OPT_QUERIES_PATH),
        "manifest": str(MANIFEST_PATH),
        "summary": str(SUMMARY_PATH),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
