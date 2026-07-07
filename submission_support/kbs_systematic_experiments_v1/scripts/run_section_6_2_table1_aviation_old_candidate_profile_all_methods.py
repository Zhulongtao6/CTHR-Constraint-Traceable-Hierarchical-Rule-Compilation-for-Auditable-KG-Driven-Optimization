from __future__ import annotations

import argparse
import ast
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
SCRIPTS_DIR = ROOT / "scripts"

sys.path.insert(0, str(SCRIPTS_DIR))

import run_section_6_2_table1_fullkg_pipeline as full  # noqa: E402
import run_section_6_2_table2_cell_solver_backends as cell_backends  # noqa: E402
from ortools.sat.python import cp_model  # noqa: E402
from pyscipopt import Model, quicksum  # noqa: E402


DEFAULT_GROUNDING_BY_DOMAIN = {
    "aviation": RESULTS_DIR
    / "section_6_3_aviation_old_candidate_recall_guard_profile_auto_resolver_candidate_to_valid_full.json",
    "architecture": RESULTS_DIR
    / "section_6_3_architecture_old_candidate_profile_auto_resolver_candidate_to_valid_full.json",
}
OUTPUT_PREFIX_BY_DOMAIN = {
    "aviation": "section_6_2_table1_aviation_old_candidate_recall_guard_profile_auto_resolver_all_methods",
    "architecture": "section_6_2_table1_architecture_old_candidate_profile_auto_resolver_all_methods",
}
CTHR_GROUNDED_METHODS = {
    "CTHR default",
    "CTHR-style ASP + clingo",
    "CTHR-style SLSQP",
    "CTHR-style pure HiGHS",
    "CTHR-style HiGHS",
    "CTHR-style CP-SAT + OR-Tools",
    "CTHR-style SCIP",
}
METHOD_SPECS = [
    ("Flat baseline", "flat"),
    ("Native ASP + clingo", "native_symbolic"),
    ("Native SLSQP", "native_symbolic"),
    ("Native MILP + HiGHS", "native_symbolic"),
    ("Native CP-SAT + OR-Tools", "native_symbolic"),
    ("Native SCIP", "native_symbolic"),
    ("CTHR default", "cthr_semantic_modeling"),
    ("CTHR-style ASP + clingo", "cthr_semantic_modeling"),
    ("CTHR-style SLSQP", "cthr_semantic_modeling"),
    ("CTHR-style HiGHS", "cthr_semantic_modeling"),
    ("CTHR-style CP-SAT + OR-Tools", "cthr_semantic_modeling"),
    ("CTHR-style SCIP", "cthr_semantic_modeling"),
]
CP_VAR_SCALE = 10000
CP_COEFF_SCALE = 10000000
CP_OBJECTIVE_SCALE = 1000000
CP_ABS_TOL = 1e-5


def markdown_table(rows: list[dict[str, Any]], headers: list[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(full.csv_cell(row.get(header)) for header in headers) + " |")
    return "\n".join(lines)


def build_report(aggregate_rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
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
    dataset_name = str(summary.get("dataset", {}).get("dataset", "Dataset"))
    domain = str(summary.get("domain", dataset_name.lower()))
    dataset_root = Path(str(summary.get("dataset", {}).get("root", "")))
    dataset_root_name = dataset_root.name if str(dataset_root) else f"{domain}_fullkg_clean"
    recall_guard_text = " with aviation recall guard" if domain == "aviation" else ""
    return "\n".join(
        [
            f"# Section 6.2 Table 1: {dataset_name} All Methods With Old-Candidate Profile Grounding",
            "",
            "## Scope",
            "",
            f"- Dataset: `{dataset_root_name}` only.",
            f"- Candidate grounding: old broad rule-library scorer{recall_guard_text}.",
            "- CTHR default valid rules: candidate-constrained profile_auto_resolver output from the grounding file.",
            "- Flat and native symbolic baselines use the candidate_rule_ids_generated field as method-visible candidates.",
            "- CTHR-style ASP/CP-SAT/SCIP use exactly the same predicted_valid_rule_ids grounding as CTHR default.",
            "- Evaluation references are used only for metrics.",
            "",
            "## Result",
            "",
            markdown_table(aggregate_rows, headers),
            "",
            "## Run Summary",
            "",
            "```json",
            json.dumps(summary, ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )


def run_fixed_grounded_default(
    method: str,
    query: dict[str, Any],
    selected_valid_ids: list[str],
    rule_by_id: dict[str, dict[str, Any]],
) -> full.MethodResult:
    if not selected_valid_ids:
        return full.MethodResult(False, [], None, None, "cthr_no_valid_rules")
    x, formal = full.solve_with_default(method, query, sorted(selected_valid_ids), rule_by_id)
    return full.MethodResult(True, sorted(selected_valid_ids), x, formal)


def run_fixed_grounded_smt(
    query: dict[str, Any],
    selected_valid_ids: list[str],
    rule_library: dict[str, Any],
    rule_by_id: dict[str, dict[str, Any]],
) -> full.MethodResult:
    if not selected_valid_ids:
        return full.MethodResult(False, [], None, None, "cthr_no_valid_rules")
    selected = sorted(selected_valid_ids)
    try:
        lib = full.filtered_library(rule_library, selected, native=False)
        formula = full.build_smt_formula(
            lib,
            query,
            candidate_rule_ids=selected,
            include_visible_task_constraints=True,
        )
        for rule_id in selected:
            if rule_id in formula.y:
                formula.constraints.append(formula.y[rule_id])
        result = full.optimize_with_z3(formula, query, timeout_ms=10000)
        if result.status != "sat" or result.optimized_x is None:
            return full.MethodResult(
                False,
                selected,
                None,
                None,
                f"smt_{result.status}:{result.error or ''}".strip(":"),
            )
        variables = list(query.get("decision_variables", {}))
        x = {name: float(value) for name, value in zip(variables, result.optimized_x)}
        constraints = full.method_constraints(query, selected, rule_by_id)
        formal = full.base.constraints_satisfied(constraints, full.base.with_query_values(query, x))
        return full.MethodResult(True, selected, x, formal)
    except Exception as exc:  # noqa: BLE001
        return full.MethodResult(False, selected, None, None, f"smt_error:{exc}")


def run_fixed_grounded_milp(
    query: dict[str, Any],
    selected_valid_ids: list[str],
    rule_by_id: dict[str, dict[str, Any]],
) -> full.MethodResult:
    if not selected_valid_ids:
        return full.MethodResult(False, [], None, None, "cthr_no_valid_rules")
    selected = sorted(selected_valid_ids)
    try:
        constraints = full.method_constraints(query, selected, rule_by_id)
        variables = list(query.get("decision_variables", {}))
        constants = {
            key: float(value)
            for key, value in query.get("scenario_facts", {}).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        rows, lb, ub = full.base.linear_constraint_rows(constraints, variables, constants)
        c = full.base.objective_linear(query, variables, constants)
        bounds = [
            (
                float(query["decision_variables"][name].get("lower", 0.0)),
                float(query["decision_variables"][name].get("upper", 1.0)),
            )
            for name in variables
        ]
        a_ub = [row for row, lo, hi in zip(rows, lb, ub) if math.isinf(lo) and not math.isinf(hi)]
        b_ub = [hi for lo, hi in zip(lb, ub) if math.isinf(lo) and not math.isinf(hi)]
        a_eq = [
            row
            for row, lo, hi in zip(rows, lb, ub)
            if not math.isinf(lo) and not math.isinf(hi) and abs(lo - hi) < 1e-9
        ]
        b_eq = [lo for lo, hi in zip(lb, ub) if not math.isinf(lo) and not math.isinf(hi) and abs(lo - hi) < 1e-9]
        lp_result = full.base.linprog(
            c=full.np.array(c),
            A_ub=full.np.array(a_ub) if a_ub else None,
            b_ub=full.np.array(b_ub) if b_ub else None,
            A_eq=full.np.array(a_eq) if a_eq else None,
            b_eq=full.np.array(b_eq) if b_eq else None,
            bounds=bounds,
            method="highs",
        )
        if not lp_result.success or lp_result.x is None:
            return full.MethodResult(False, selected, None, None, f"highs_{lp_result.message}")
        x = {name: float(value) for name, value in zip(variables, lp_result.x)}
        formal = full.base.constraints_satisfied(constraints, full.base.with_query_values(query, x))
        return full.MethodResult(True, selected, x, formal)
    except Exception as exc:  # noqa: BLE001
        return full.MethodResult(False, selected, None, None, f"unsupported_nonlinear_or_mapping:{exc}")


def run_highs_compiled_backend(
    query: dict[str, Any],
    selected_valid_ids: list[str],
    rule_by_id: dict[str, dict[str, Any]],
) -> full.MethodResult:
    if not selected_valid_ids:
        return full.MethodResult(False, [], None, None, "cthr_no_valid_rules")
    selected = sorted(selected_valid_ids)
    query_for_solver = dict(query)
    query_for_solver["_cthr_predicted_valid_rule_ids"] = selected
    try:
        cells = cell_backends.compiled_cells_from_cthr_grounding(query_for_solver, rule_by_id)
        result = cell_backends.highs_solver(query_for_solver, cells)
        if not result.solved or result.x is None:
            return full.MethodResult(False, selected, None, None, result.unsupported_reason or "highs_not_solved")
        variables = list(query.get("decision_variables", {}))
        x = {name: float(value) for name, value in zip(variables, result.x)}
        return full.MethodResult(True, selected, x, bool(result.cell_valid), result.unsupported_reason)
    except Exception as exc:  # noqa: BLE001
        return full.MethodResult(False, selected, None, None, f"highs_compiled_error:{exc}")


def run_highs_with_scip_repair_backend(
    method: str,
    query: dict[str, Any],
    selected_valid_ids: list[str],
    rule_by_id: dict[str, dict[str, Any]],
) -> full.MethodResult:
    highs_result = run_highs_compiled_backend(query, selected_valid_ids, rule_by_id)
    if highs_result.supported and highs_result.formal_feasible:
        return highs_result
    scip_result = run_scip_backend(method, query, selected_valid_ids, rule_by_id)
    if scip_result.supported and scip_result.formal_feasible:
        return scip_result
    if highs_result.supported:
        return highs_result
    return scip_result


def slsqp_constraints_for_selected(
    query: dict[str, Any],
    constraints: list[dict[str, Any]],
    variables: list[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    def values_from_vector(z: np.ndarray) -> dict[str, float]:
        x = {name: float(value) for name, value in zip(variables, z)}
        return full.base.with_query_values(query, x)

    def make_value(expr: str):
        def fun(z: np.ndarray) -> float:
            try:
                return full.base.eval_arithmetic(expr, values_from_vector(np.asarray(z, dtype=float)))
            except Exception:
                return float("nan")

        return fun

    def safe_ineq(left_fun: Any, right_fun: Any, sign: float):
        def fun(z: np.ndarray) -> float:
            left = left_fun(z)
            right = right_fun(z)
            if not math.isfinite(left) or not math.isfinite(right):
                return -1e6
            return sign * (left - right)

        return fun

    def safe_eq(left_fun: Any, right_fun: Any):
        def fun(z: np.ndarray) -> float:
            left = left_fun(z)
            right = right_fun(z)
            if not math.isfinite(left) or not math.isfinite(right):
                return 1e6
            return left - right

        return fun

    for constraint in constraints:
        expression = constraint.get("checker_expression") or constraint.get("expression")
        if not expression:
            continue
        normalized = full.base.normalize_expression(str(expression))
        match = full.base.COMPARATOR_RE.search(normalized)
        if not match:
            continue
        left = normalized[: match.start()].strip()
        op = match.group(1)
        right = normalized[match.end() :].strip()
        left_fun = make_value(left)
        right_fun = make_value(right)
        if op in {"<=", "<"}:
            out.append({"type": "ineq", "fun": safe_ineq(right_fun, left_fun, 1.0)})
        elif op in {">=", ">"}:
            out.append({"type": "ineq", "fun": safe_ineq(left_fun, right_fun, 1.0)})
        elif op in {"=", "=="}:
            out.append({"type": "eq", "fun": safe_eq(left_fun, right_fun)})
    return out


def run_slsqp_backend(
    method: str,
    query: dict[str, Any],
    selected_rule_ids: list[str],
    rule_by_id: dict[str, dict[str, Any]],
) -> full.MethodResult:
    selected = sorted(selected_rule_ids)
    if not selected:
        return full.MethodResult(False, [], None, None, "no_selected_rules")
    variables = list(query.get("decision_variables", {}))
    if not variables:
        return full.MethodResult(False, selected, None, None, "no_decision_variables")
    bounds = [
        (
            float(query["decision_variables"][name].get("lower", 0.0)),
            float(query["decision_variables"][name].get("upper", 1.0)),
        )
        for name in variables
    ]
    constraints = constraints_for_selected(query, selected, rule_by_id)
    scipy_constraints = slsqp_constraints_for_selected(query, constraints, variables)
    seed = int(abs(hash((method, str(query.get("omega_id", ""))))) % (2**32))
    rng = np.random.default_rng(seed)
    starts = [
        np.array([(lo + hi) / 2.0 for lo, hi in bounds], dtype=float),
        np.array([rng.uniform(lo, hi) for lo, hi in bounds], dtype=float),
    ]

    def vector_to_x(z: np.ndarray) -> dict[str, float]:
        return {name: float(value) for name, value in zip(variables, z)}

    def objective(z: np.ndarray) -> float:
        values = full.base.with_query_values(query, vector_to_x(np.asarray(z, dtype=float)))
        try:
            return full.base.objective_value(query, values)
        except Exception:
            penalty = 0.0
            for constraint in constraints:
                expression = constraint.get("checker_expression") or constraint.get("expression")
                if expression:
                    penalty += full.base.constraint_violation(str(expression), values)
            return 1e6 + 1e6 * penalty

    best: tuple[float, dict[str, float]] | None = None
    last_status = "slsqp_no_feasible_candidate"
    for start in starts:
        trial_points = [start]
        try:
            local = minimize(
                objective,
                start,
                method="SLSQP",
                bounds=bounds,
                constraints=scipy_constraints,
                options={"maxiter": 120, "ftol": 1e-8, "disp": False},
            )
            trial_points.append(np.asarray(local.x, dtype=float))
            if not local.success:
                last_status = f"slsqp_{str(local.message).replace(' ', '_')[:80]}"
        except Exception as exc:  # noqa: BLE001
            last_status = f"slsqp_error:{exc}"
        for point in trial_points:
            x = vector_to_x(np.asarray(point, dtype=float))
            values = full.base.with_query_values(query, x)
            if full.base.constraints_satisfied(constraints, values):
                score = full.base.objective_value(query, values)
                if best is None or score < best[0]:
                    best = (score, x)
    if best is None:
        return full.MethodResult(False, selected, None, None, last_status)
    formal = full.base.constraints_satisfied(constraints, full.base.with_query_values(query, best[1]))
    return full.MethodResult(True, selected, best[1], formal)


def numeric_constants(query: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in query.get("scenario_facts", {}).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def abs_inner_expression(expr: str) -> str | None:
    try:
        node = ast.parse(str(expr), mode="eval").body
    except SyntaxError:
        return None
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "abs"
        and len(node.args) == 1
        and not node.keywords
    ):
        return ast.unparse(node.args[0])
    return None


def linear_diff(
    left: str,
    right: str,
    parser: full.base.LinearExpr,
) -> tuple[dict[str, float], float]:
    left_coeff, left_const = parser.parse(left)
    right_coeff, right_const = parser.parse(right)
    out = dict(left_coeff)
    for key, value in right_coeff.items():
        out[key] = out.get(key, 0.0) - value
    return {key: value for key, value in out.items() if abs(value) > 1e-12}, left_const - right_const


def negate_linear(coeff: dict[str, float], const: float) -> tuple[dict[str, float], float]:
    return {key: -value for key, value in coeff.items()}, -const


def combine_linear(
    first: tuple[dict[str, float], float],
    second: tuple[dict[str, float], float],
    *,
    second_sign: float = 1.0,
) -> tuple[dict[str, float], float]:
    coeff = dict(first[0])
    for key, value in second[0].items():
        coeff[key] = coeff.get(key, 0.0) + second_sign * value
    return {key: value for key, value in coeff.items() if abs(value) > 1e-12}, first[1] + second_sign * second[1]


def linear_constraint_variants(
    expression: str,
    variables: list[str],
    constants: dict[str, float],
) -> list[tuple[dict[str, float], float, str]]:
    """Return constraints of the form coeff*x + const sense 0.

    The CP-SAT/SCIP replacements intentionally support the linear fragment plus
    convex absolute-value epigraph constraints used in the benchmark closure
    cells, e.g. z >= abs(x - c) and abs(x - c) <= eps.
    """
    if not full.base.COMPARATOR_RE.search(str(expression)):
        return []
    parser = full.base.LinearExpr(set(variables), constants)
    left, op, right = full.base.split_constraint(str(expression))
    if op == "=":
        op = "=="
    if op == "!=":
        raise ValueError("non-equality comparator is unsupported")

    left_abs = abs_inner_expression(left)
    right_abs = abs_inner_expression(right)
    if left_abs and right_abs:
        raise ValueError("abs on both sides is unsupported")
    if left_abs:
        if op not in {"<=", "<"}:
            raise ValueError("nonconvex abs lower-bound is unsupported")
        inner = parser.parse(left_abs)
        bound = parser.parse(right)
        first = combine_linear(inner, bound, second_sign=-1.0)
        second = combine_linear(negate_linear(*inner), bound, second_sign=-1.0)
        return [(first[0], first[1], "<="), (second[0], second[1], "<=")]
    if right_abs:
        if op not in {">=", ">"}:
            raise ValueError("nonconvex abs upper-side constraint is unsupported")
        bound = parser.parse(left)
        inner = parser.parse(right_abs)
        first = combine_linear(inner, bound, second_sign=-1.0)
        second = combine_linear(negate_linear(*inner), bound, second_sign=-1.0)
        return [(first[0], first[1], "<="), (second[0], second[1], "<=")]

    coeff, const = linear_diff(left, right, parser)
    if op in {"<=", "<"}:
        return [(coeff, const, "<=")]
    if op in {">=", ">"}:
        return [(coeff, const, ">=")]
    if op == "==":
        return [(coeff, const, "==")]
    raise ValueError(f"unsupported comparator {op}")


def constraints_for_selected(
    query: dict[str, Any],
    selected_rule_ids: list[str],
    rule_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return full.method_constraints(query, sorted(selected_rule_ids), rule_by_id)


def merge_linear(
    left: tuple[dict[str, float], float, list[tuple[float, list[tuple[dict[str, float], float]]]]],
    right: tuple[dict[str, float], float, list[tuple[float, list[tuple[dict[str, float], float]]]]],
    *,
    right_sign: float = 1.0,
) -> tuple[dict[str, float], float, list[tuple[float, list[tuple[dict[str, float], float]]]]]:
    coeff = dict(left[0])
    for key, value in right[0].items():
        coeff[key] = coeff.get(key, 0.0) + right_sign * value
    max_terms = list(left[2])
    max_terms.extend((right_sign * coef, args) for coef, args in right[2])
    return {key: value for key, value in coeff.items() if abs(value) > 1e-12}, left[1] + right_sign * right[1], max_terms


def scale_objective_part(
    part: tuple[dict[str, float], float, list[tuple[float, list[tuple[dict[str, float], float]]]]],
    factor: float,
) -> tuple[dict[str, float], float, list[tuple[float, list[tuple[dict[str, float], float]]]]]:
    return (
        {key: factor * value for key, value in part[0].items()},
        factor * part[1],
        [(factor * coef, args) for coef, args in part[2]],
    )


class ObjectiveExpr(ast.NodeVisitor):
    def __init__(self, variables: list[str], constants: dict[str, float]):
        self.variables = set(variables)
        self.constants = constants
        self.linear_parser = full.base.LinearExpr(self.variables, constants)

    def parse(self, expr: str) -> tuple[dict[str, float], float, list[tuple[float, list[tuple[dict[str, float], float]]]]]:
        tree = ast.parse(expr, mode="eval")
        return self.visit(tree.body)

    def linear_arg(self, node: ast.AST) -> tuple[dict[str, float], float]:
        return self.linear_parser.parse(ast.unparse(node))

    def visit_Name(self, node: ast.Name) -> tuple[dict[str, float], float, list[tuple[float, list[tuple[dict[str, float], float]]]]]:
        coeff, const = self.linear_arg(node)
        return coeff, const, []

    def visit_Constant(
        self,
        node: ast.Constant,
    ) -> tuple[dict[str, float], float, list[tuple[float, list[tuple[dict[str, float], float]]]]]:
        coeff, const = self.linear_arg(node)
        return coeff, const, []

    def visit_UnaryOp(
        self,
        node: ast.UnaryOp,
    ) -> tuple[dict[str, float], float, list[tuple[float, list[tuple[dict[str, float], float]]]]]:
        part = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return scale_objective_part(part, -1.0)
        if isinstance(node.op, ast.UAdd):
            return part
        raise ValueError("unsupported objective unary op")

    def visit_BinOp(
        self,
        node: ast.BinOp,
    ) -> tuple[dict[str, float], float, list[tuple[float, list[tuple[dict[str, float], float]]]]]:
        if isinstance(node.op, ast.Add):
            return merge_linear(self.visit(node.left), self.visit(node.right))
        if isinstance(node.op, ast.Sub):
            return merge_linear(self.visit(node.left), self.visit(node.right), right_sign=-1.0)
        if isinstance(node.op, ast.Mult):
            try:
                left_coeff, left_const = self.linear_arg(node.left)
                right = self.visit(node.right)
                if left_coeff:
                    raise ValueError
                return scale_objective_part(right, left_const)
            except Exception:
                right_coeff, right_const = self.linear_arg(node.right)
                left = self.visit(node.left)
                if right_coeff:
                    raise ValueError("objective bilinear term")
                return scale_objective_part(left, right_const)
        if isinstance(node.op, ast.Div):
            right_coeff, right_const = self.linear_arg(node.right)
            if right_coeff or abs(right_const) < 1e-12:
                raise ValueError("objective nonconstant division")
            return scale_objective_part(self.visit(node.left), 1.0 / right_const)
        raise ValueError("unsupported objective binary op")

    def visit_Call(
        self,
        node: ast.Call,
    ) -> tuple[dict[str, float], float, list[tuple[float, list[tuple[dict[str, float], float]]]]]:
        if isinstance(node.func, ast.Name) and node.func.id == "max" and len(node.args) >= 2 and not node.keywords:
            return {}, 0.0, [(1.0, [self.linear_arg(arg) for arg in node.args])]
        raise ValueError("unsupported objective function call")


def objective_components(
    query: dict[str, Any],
    variables: list[str],
    constants: dict[str, float],
) -> tuple[dict[str, float], float, list[tuple[float, list[tuple[dict[str, float], float]]]]]:
    parser = ObjectiveExpr(variables, constants)
    weights = query.get("query_preferences", {}).get("lambda") or query.get("preference_weights") or []
    objectives = query.get("objectives", [])
    if not weights:
        weights = [1.0 / max(1, len(objectives))] * len(objectives)
    total: tuple[dict[str, float], float, list[tuple[float, list[tuple[dict[str, float], float]]]]] = ({}, 0.0, [])
    for weight, objective in zip(weights, objectives):
        sign = -1.0 if str(objective.get("name", "")).lower().startswith("maximize") else 1.0
        part = scale_objective_part(parser.parse(str(objective["expression"])), float(weight) * sign)
        total = merge_linear(total, part)
    return total


def cp_linear_expr(
    model_vars: dict[str, Any],
    coeff: dict[str, float],
    const: float,
) -> Any:
    terms = [int(round(value * CP_COEFF_SCALE)) * model_vars[name] for name, value in coeff.items()]
    offset = int(round(const * CP_VAR_SCALE * CP_COEFF_SCALE))
    return sum(terms, offset)


def cp_linear_bounds(
    query: dict[str, Any],
    coeff: dict[str, float],
    const: float,
) -> tuple[int, int]:
    lower = const
    upper = const
    for name, value in coeff.items():
        var_spec = query["decision_variables"][name]
        lb = float(var_spec.get("lower", 0.0))
        ub = float(var_spec.get("upper", 1.0))
        if value >= 0:
            lower += value * lb
            upper += value * ub
        else:
            lower += value * ub
            upper += value * lb
    return (
        math.floor(lower * CP_VAR_SCALE * CP_COEFF_SCALE) - 1,
        math.ceil(upper * CP_VAR_SCALE * CP_COEFF_SCALE) + 1,
    )


def cp_bound(value: float) -> int:
    return int(round(value * CP_VAR_SCALE * CP_COEFF_SCALE))


def add_cp_constraint(
    model: cp_model.CpModel,
    model_vars: dict[str, Any],
    coeff: dict[str, float],
    const: float,
    sense: str,
) -> None:
    if not coeff:
        ok = (
            (sense == "<=" and const <= CP_ABS_TOL)
            or (sense == ">=" and const >= -CP_ABS_TOL)
            or (sense == "==" and abs(const) <= CP_ABS_TOL)
        )
        if not ok:
            raise ValueError("constant constraint is infeasible")
        return
    expr = cp_linear_expr(model_vars, coeff, const)
    if sense == "<=":
        model.Add(expr <= 0)
    elif sense == ">=":
        model.Add(expr >= 0)
    elif sense == "==":
        tol = cp_bound(CP_ABS_TOL)
        model.Add(expr <= tol)
        model.Add(expr >= -tol)
    else:
        raise ValueError(f"unsupported CP sense {sense}")


def add_cp_expression_constraint(
    model: cp_model.CpModel,
    query: dict[str, Any],
    model_vars: dict[str, Any],
    expression: str,
    variables: list[str],
    constants: dict[str, float],
    aux_index: int,
) -> int:
    if not full.base.COMPARATOR_RE.search(str(expression)):
        return aux_index
    parser = full.base.LinearExpr(set(variables), constants)
    left, op, right = full.base.split_constraint(str(expression))
    if op == "=":
        op = "=="
    left_abs = abs_inner_expression(left)
    right_abs = abs_inner_expression(right)
    if not left_abs and not right_abs:
        for coeff, const, sense in linear_constraint_variants(str(expression), variables, constants):
            add_cp_constraint(model, model_vars, coeff, const, sense)
        return aux_index
    if left_abs and right_abs:
        raise ValueError("abs on both sides is unsupported")

    if left_abs:
        inner = parser.parse(left_abs)
        other = parser.parse(right)
        abs_side = "left"
    else:
        inner = parser.parse(right_abs or "")
        other = parser.parse(left)
        abs_side = "right"
    inner_expr = cp_linear_expr(model_vars, inner[0], inner[1])
    abs_lb, abs_ub = cp_linear_bounds(query, inner[0], inner[1])
    aux = model.NewIntVar(0, max(abs(abs_lb), abs(abs_ub), 1), f"abs_aux_{aux_index}")
    aux_index += 1
    model.AddAbsEquality(aux, inner_expr)
    other_expr = cp_linear_expr(model_vars, other[0], other[1])
    if abs_side == "left":
        lhs, rhs = aux, other_expr
    else:
        lhs, rhs = other_expr, aux
    if op in {"<=", "<"}:
        model.Add(lhs <= rhs)
    elif op in {">=", ">"}:
        model.Add(lhs >= rhs)
    elif op == "==":
        model.Add(lhs <= rhs + cp_bound(CP_ABS_TOL))
        model.Add(lhs >= rhs - cp_bound(CP_ABS_TOL))
    else:
        raise ValueError(f"unsupported abs comparator {op}")
    return aux_index


def merge_linear_abs(
    left: tuple[dict[str, float], float, list[tuple[float, dict[str, float], float]]],
    right: tuple[dict[str, float], float, list[tuple[float, dict[str, float], float]]],
    *,
    right_sign: float = 1.0,
) -> tuple[dict[str, float], float, list[tuple[float, dict[str, float], float]]]:
    coeff = dict(left[0])
    for key, value in right[0].items():
        coeff[key] = coeff.get(key, 0.0) + right_sign * value
    abs_terms = list(left[2])
    abs_terms.extend((right_sign * scale, inner_coeff, inner_const) for scale, inner_coeff, inner_const in right[2])
    return {key: value for key, value in coeff.items() if abs(value) > 1e-12}, left[1] + right_sign * right[1], abs_terms


def scale_linear_abs(
    expr: tuple[dict[str, float], float, list[tuple[float, dict[str, float], float]]],
    scale: float,
) -> tuple[dict[str, float], float, list[tuple[float, dict[str, float], float]]]:
    return (
        {key: scale * value for key, value in expr[0].items() if abs(scale * value) > 1e-12},
        scale * expr[1],
        [(scale * abs_scale, inner_coeff, inner_const) for abs_scale, inner_coeff, inner_const in expr[2]],
    )


class LinearAbsExpr(ast.NodeVisitor):
    """Linear expressions plus scalar multiples of abs(linear) and constant max()."""

    def __init__(self, variables: set[str], constants: dict[str, float]):
        self.variables = variables
        self.constants = constants
        self.linear_parser = full.base.LinearExpr(variables, constants)

    def parse(self, expr: str) -> tuple[dict[str, float], float, list[tuple[float, dict[str, float], float]]]:
        tree = ast.parse(expr, mode="eval")
        return self.visit(tree.body)

    @staticmethod
    def scalar_value(expr: tuple[dict[str, float], float, list[tuple[float, dict[str, float], float]]]) -> float:
        if expr[0] or expr[2]:
            raise ValueError("non-scalar term")
        return expr[1]

    def visit_Name(self, node: ast.Name) -> tuple[dict[str, float], float, list[tuple[float, dict[str, float], float]]]:
        if node.id in self.variables:
            return {node.id: 1.0}, 0.0, []
        if node.id in self.constants:
            return {}, float(self.constants[node.id]), []
        raise ValueError(f"unknown symbol {node.id}")

    def visit_Constant(self, node: ast.Constant) -> tuple[dict[str, float], float, list[tuple[float, dict[str, float], float]]]:
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return {}, float(node.value), []
        raise ValueError("non-numeric constant")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> tuple[dict[str, float], float, list[tuple[float, dict[str, float], float]]]:
        expr = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return scale_linear_abs(expr, -1.0)
        if isinstance(node.op, ast.UAdd):
            return expr
        raise ValueError("unsupported unary op")

    def visit_BinOp(self, node: ast.BinOp) -> tuple[dict[str, float], float, list[tuple[float, dict[str, float], float]]]:
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return merge_linear_abs(left, right)
        if isinstance(node.op, ast.Sub):
            return merge_linear_abs(left, right, right_sign=-1.0)
        if isinstance(node.op, ast.Mult):
            try:
                return scale_linear_abs(right, self.scalar_value(left))
            except ValueError:
                return scale_linear_abs(left, self.scalar_value(right))
        if isinstance(node.op, ast.Div):
            denom = self.scalar_value(right)
            if abs(denom) < 1e-12:
                raise ValueError("division by zero")
            return scale_linear_abs(left, 1.0 / denom)
        raise ValueError("unsupported binary op")

    def visit_Call(self, node: ast.Call) -> tuple[dict[str, float], float, list[tuple[float, dict[str, float], float]]]:
        if not isinstance(node.func, ast.Name) or node.keywords:
            raise ValueError("unsupported function call")
        if node.func.id == "abs" and len(node.args) == 1:
            inner_coeff, inner_const = self.linear_parser.parse(ast.unparse(node.args[0]))
            return {}, 0.0, [(1.0, inner_coeff, inner_const)]
        if node.func.id == "max" and len(node.args) >= 2:
            return {}, max(self.scalar_value(self.visit(arg)) for arg in node.args), []
        raise ValueError("unsupported function call")


def add_cp_linear_abs_constraint(
    model: cp_model.CpModel,
    query: dict[str, Any],
    model_vars: dict[str, Any],
    coeff: dict[str, float],
    const: float,
    abs_terms: list[tuple[float, dict[str, float], float]],
    sense: str,
    aux_index: int,
) -> int:
    expr = cp_linear_expr(model_vars, coeff, const)
    for scale, inner_coeff, inner_const in abs_terms:
        if abs(scale - round(scale)) > 1e-9:
            raise ValueError("fractional abs coefficient is unsupported")
        inner_expr = cp_linear_expr(model_vars, inner_coeff, inner_const)
        inner_lb, inner_ub = cp_linear_bounds(query, inner_coeff, inner_const)
        bound = max(abs(inner_lb), abs(inner_ub), 1)
        aux = model.NewIntVar(0, bound, f"abs_template_{aux_index}")
        aux_index += 1
        model.AddAbsEquality(aux, inner_expr)
        expr += int(round(scale)) * aux
    if sense == "<=":
        model.Add(expr <= 0)
    elif sense == ">=":
        model.Add(expr >= 0)
    elif sense == "==":
        tol = cp_bound(CP_ABS_TOL)
        model.Add(expr <= tol)
        model.Add(expr >= -tol)
    else:
        raise ValueError(f"unsupported CP sense {sense}")
    return aux_index


def add_cp_linear_abs_expression_constraint(
    model: cp_model.CpModel,
    query: dict[str, Any],
    model_vars: dict[str, Any],
    expression: str,
    variables: list[str],
    constants: dict[str, float],
    aux_index: int,
) -> int:
    left, op, right = full.base.split_constraint(str(expression))
    if op == "=":
        op = "=="
    parser = LinearAbsExpr(set(variables), constants)
    diff = merge_linear_abs(parser.parse(left), parser.parse(right), right_sign=-1.0)
    if op in {"<=", "<"}:
        return add_cp_linear_abs_constraint(model, query, model_vars, diff[0], diff[1], diff[2], "<=", aux_index)
    if op in {">=", ">"}:
        return add_cp_linear_abs_constraint(model, query, model_vars, diff[0], diff[1], diff[2], ">=", aux_index)
    if op == "==":
        return add_cp_linear_abs_constraint(model, query, model_vars, diff[0], diff[1], diff[2], "==", aux_index)
    raise ValueError(f"unsupported comparator {op}")


def flatten_scaled_product(
    node: ast.AST,
    variables: set[str],
    constants: dict[str, float],
) -> tuple[float, list[str]]:
    if isinstance(node, ast.Name):
        if node.id in variables:
            return 1.0, [node.id]
        if node.id in constants:
            return float(constants[node.id]), []
        raise ValueError(f"unknown symbol {node.id}")
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value), []
    if isinstance(node, ast.UnaryOp):
        scale, factors = flatten_scaled_product(node.operand, variables, constants)
        if isinstance(node.op, ast.USub):
            return -scale, factors
        if isinstance(node.op, ast.UAdd):
            return scale, factors
        raise ValueError("unsupported unary op")
    if isinstance(node, ast.BinOp):
        left_scale, left_factors = flatten_scaled_product(node.left, variables, constants)
        right_scale, right_factors = flatten_scaled_product(node.right, variables, constants)
        if isinstance(node.op, ast.Mult):
            return left_scale * right_scale, left_factors + right_factors
        if isinstance(node.op, ast.Div):
            if right_factors or abs(right_scale) < 1e-12:
                raise ValueError("nonconstant product division")
            return left_scale / right_scale, left_factors
    raise ValueError("not a scaled product")


def add_cp_bilinear_upper_template(
    model: cp_model.CpModel,
    query: dict[str, Any],
    model_vars: dict[str, Any],
    linear_expr: str,
    product_expr: str,
    variables: list[str],
    constants: dict[str, float],
) -> None:
    parser = full.base.LinearExpr(set(variables), constants)
    linear_coeff, linear_const = parser.parse(linear_expr)
    scale, factors = flatten_scaled_product(ast.parse(product_expr, mode="eval").body, set(variables), constants)
    if len(factors) != 2 or factors[0] == factors[1] or scale <= 0:
        raise ValueError("unsupported bilinear template")
    x_name, y_name = factors
    x_spec = query["decision_variables"][x_name]
    y_spec = query["decision_variables"][y_name]
    lx, ux = float(x_spec.get("lower", 0.0)), float(x_spec.get("upper", 1.0))
    ly, uy = float(y_spec.get("lower", 0.0)), float(y_spec.get("upper", 1.0))
    envelopes = [
        ({x_name: scale * ly, y_name: scale * ux}, -scale * ux * ly),
        ({x_name: scale * uy, y_name: scale * lx}, -scale * lx * uy),
    ]
    for env_coeff, env_const in envelopes:
        coeff = dict(linear_coeff)
        for key, value in env_coeff.items():
            coeff[key] = coeff.get(key, 0.0) - value
        add_cp_constraint(
            model,
            model_vars,
            {key: value for key, value in coeff.items() if abs(value) > 1e-12},
            linear_const - env_const,
            ">=",
        )


def extract_division_by_scaled_var(
    node: ast.AST,
    variables: set[str],
    constants: dict[str, float],
) -> tuple[str, float, str]:
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div) or not isinstance(node.left, ast.Name):
        raise ValueError("not a variable ratio")
    numerator = node.left.id
    if numerator not in variables:
        raise ValueError("ratio numerator is not a variable")
    scale, factors = flatten_scaled_product(node.right, variables, constants)
    if len(factors) != 1 or scale <= 0:
        raise ValueError("unsupported ratio denominator")
    return numerator, scale, factors[0]


def add_cp_ratio_upper_template(
    model: cp_model.CpModel,
    query: dict[str, Any],
    model_vars: dict[str, Any],
    linear_expr: str,
    ratio_expr: str,
    variables: list[str],
    constants: dict[str, float],
) -> None:
    parser = full.base.LinearExpr(set(variables), constants)
    linear_coeff, linear_const = parser.parse(linear_expr)
    numerator, denominator_scale, denominator = extract_division_by_scaled_var(
        ast.parse(ratio_expr, mode="eval").body,
        set(variables),
        constants,
    )
    num_lb = float(query["decision_variables"][numerator].get("lower", 0.0))
    den_ub = float(query["decision_variables"][denominator].get("upper", 1.0))
    if den_ub <= 0:
        raise ValueError("ratio denominator upper bound is nonpositive")
    safe_upper = num_lb / (denominator_scale * den_ub)
    add_cp_constraint(model, model_vars, linear_coeff, linear_const - safe_upper, "<=")


def add_cp_template_expression_constraint(
    model: cp_model.CpModel,
    query: dict[str, Any],
    model_vars: dict[str, Any],
    expression: str,
    variables: list[str],
    constants: dict[str, float],
    aux_index: int,
) -> int:
    left, op, right = full.base.split_constraint(str(expression))
    if op == "=":
        op = "=="
    template_errors: list[str] = []
    try:
        return add_cp_linear_abs_expression_constraint(
            model,
            query,
            model_vars,
            expression,
            variables,
            constants,
            aux_index,
        )
    except Exception as exc:  # noqa: BLE001
        template_errors.append(str(exc))
    try:
        if op in {">=", ">"}:
            add_cp_bilinear_upper_template(model, query, model_vars, left, right, variables, constants)
            return aux_index
        if op in {"<=", "<"}:
            add_cp_bilinear_upper_template(model, query, model_vars, right, left, variables, constants)
            return aux_index
    except Exception as exc:  # noqa: BLE001
        template_errors.append(str(exc))
    try:
        if op in {"<=", "<"}:
            add_cp_ratio_upper_template(model, query, model_vars, left, right, variables, constants)
            return aux_index
        if op in {">=", ">"}:
            add_cp_ratio_upper_template(model, query, model_vars, right, left, variables, constants)
            return aux_index
    except Exception as exc:  # noqa: BLE001
        template_errors.append(str(exc))
    raise ValueError("; ".join(error for error in template_errors if error) or "unsupported CP template")


def add_cp_expression_constraint_robust(
    model: cp_model.CpModel,
    query: dict[str, Any],
    model_vars: dict[str, Any],
    expression: str,
    variables: list[str],
    constants: dict[str, float],
    aux_index: int,
) -> int:
    try:
        return add_cp_expression_constraint(
            model,
            query,
            model_vars,
            expression,
            variables,
            constants,
            aux_index,
        )
    except Exception as primary_exc:
        try:
            return add_cp_template_expression_constraint(
                model,
                query,
                model_vars,
                expression,
                variables,
                constants,
                aux_index,
            )
        except Exception as template_exc:  # noqa: BLE001
            raise ValueError(f"{primary_exc}; template fallback failed: {template_exc}") from template_exc


def cp_int_bounds_from_float(lower: float, upper: float) -> tuple[int, int]:
    if lower > upper:
        lower, upper = upper, lower
    return math.floor(lower * CP_VAR_SCALE) - 2, math.ceil(upper * CP_VAR_SCALE) + 2


def cp_product_bounds(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
    values = [left[0] * right[0], left[0] * right[1], left[1] * right[0], left[1] * right[1]]
    return min(values), max(values)


def cp_materialize(
    model: cp_model.CpModel,
    expr: Any,
    bounds: tuple[int, int],
    name: str,
) -> Any:
    if isinstance(expr, (int, cp_model.IntVar)):
        return expr
    var = model.NewIntVar(int(bounds[0]), int(bounds[1]), name)
    model.Add(var == expr)
    return var


def cp_expr_general(
    model: cp_model.CpModel,
    query: dict[str, Any],
    model_vars: dict[str, Any],
    node: ast.AST,
    aux_state: dict[str, int],
) -> tuple[Any, tuple[int, int]]:
    if isinstance(node, ast.Name):
        if node.id in model_vars:
            spec = query["decision_variables"][node.id]
            return model_vars[node.id], cp_int_bounds_from_float(
                float(spec.get("lower", 0.0)),
                float(spec.get("upper", 1.0)),
            )
        if node.id in query.get("scenario_facts", {}) and isinstance(query["scenario_facts"][node.id], (int, float)):
            value = float(query["scenario_facts"][node.id])
            scaled = int(round(value * CP_VAR_SCALE))
            return scaled, (scaled, scaled)
        raise ValueError(f"unknown symbol {node.id}")
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        scaled = int(round(float(node.value) * CP_VAR_SCALE))
        return scaled, (scaled, scaled)
    if isinstance(node, ast.UnaryOp):
        expr, bounds = cp_expr_general(model, query, model_vars, node.operand, aux_state)
        if isinstance(node.op, ast.USub):
            return -expr, (-bounds[1], -bounds[0])
        if isinstance(node.op, ast.UAdd):
            return expr, bounds
        raise ValueError("unsupported unary op")
    if isinstance(node, ast.BinOp):
        left, left_bounds = cp_expr_general(model, query, model_vars, node.left, aux_state)
        right, right_bounds = cp_expr_general(model, query, model_vars, node.right, aux_state)
        if isinstance(node.op, ast.Add):
            return left + right, (left_bounds[0] + right_bounds[0], left_bounds[1] + right_bounds[1])
        if isinstance(node.op, ast.Sub):
            return left - right, (left_bounds[0] - right_bounds[1], left_bounds[1] - right_bounds[0])
        if isinstance(node.op, ast.Mult):
            left_var = cp_materialize(model, left, left_bounds, f"mul_left_{aux_state['i']}")
            right_var = cp_materialize(model, right, right_bounds, f"mul_right_{aux_state['i']}")
            product_bounds = cp_product_bounds(left_bounds, right_bounds)
            product = model.NewIntVar(product_bounds[0], product_bounds[1], f"mul_prod_{aux_state['i']}")
            out_bounds = (math.floor(product_bounds[0] / CP_VAR_SCALE) - 2, math.ceil(product_bounds[1] / CP_VAR_SCALE) + 2)
            out = model.NewIntVar(out_bounds[0], out_bounds[1], f"mul_out_{aux_state['i']}")
            aux_state["i"] += 1
            model.AddMultiplicationEquality(product, [left_var, right_var])
            model.AddDivisionEquality(out, product, CP_VAR_SCALE)
            return out, out_bounds
        if isinstance(node.op, ast.Div):
            numerator = cp_materialize(model, left, left_bounds, f"div_num_{aux_state['i']}")
            denominator = cp_materialize(model, right, right_bounds, f"div_den_{aux_state['i']}")
            if right_bounds[0] <= 0 <= right_bounds[1]:
                raise ValueError("division denominator may cross zero")
            numerator_scaled_bounds = (left_bounds[0] * CP_VAR_SCALE, left_bounds[1] * CP_VAR_SCALE)
            numerator_scaled = model.NewIntVar(
                min(numerator_scaled_bounds),
                max(numerator_scaled_bounds),
                f"div_num_scaled_{aux_state['i']}",
            )
            approx_candidates = [
                numerator_scaled_bounds[0] / right_bounds[0],
                numerator_scaled_bounds[0] / right_bounds[1],
                numerator_scaled_bounds[1] / right_bounds[0],
                numerator_scaled_bounds[1] / right_bounds[1],
            ]
            out_bounds = (math.floor(min(approx_candidates)) - 2, math.ceil(max(approx_candidates)) + 2)
            out = model.NewIntVar(out_bounds[0], out_bounds[1], f"div_out_{aux_state['i']}")
            aux_state["i"] += 1
            model.Add(numerator_scaled == numerator * CP_VAR_SCALE)
            model.AddDivisionEquality(out, numerator_scaled, denominator)
            return out, out_bounds
        raise ValueError("unsupported binary op")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "abs" and len(node.args) == 1 and not node.keywords:
            inner, inner_bounds = cp_expr_general(model, query, model_vars, node.args[0], aux_state)
            inner_var = cp_materialize(model, inner, inner_bounds, f"abs_inner_general_{aux_state['i']}")
            bound = max(abs(inner_bounds[0]), abs(inner_bounds[1]), 1)
            out = model.NewIntVar(0, bound, f"abs_general_{aux_state['i']}")
            aux_state["i"] += 1
            model.AddAbsEquality(out, inner_var)
            return out, (0, bound)
        if node.func.id == "max" and len(node.args) >= 2 and not node.keywords:
            arg_exprs = []
            arg_bounds = []
            for arg in node.args:
                expr, bounds = cp_expr_general(model, query, model_vars, arg, aux_state)
                arg_exprs.append(expr)
                arg_bounds.append(bounds)
            out_bounds = (min(lb for lb, _ in arg_bounds), max(ub for _, ub in arg_bounds))
            out = model.NewIntVar(out_bounds[0], out_bounds[1], f"max_general_{aux_state['i']}")
            aux_state["i"] += 1
            model.AddMaxEquality(out, arg_exprs)
            return out, out_bounds
    raise ValueError("unsupported expression node")


def add_cp_expression_constraint_general(
    model: cp_model.CpModel,
    query: dict[str, Any],
    model_vars: dict[str, Any],
    expression: str,
    aux_state: dict[str, int],
) -> None:
    if not full.base.COMPARATOR_RE.search(str(expression)):
        return
    left, op, right = full.base.split_constraint(str(expression))
    if op == "=":
        op = "=="
    left_expr, left_bounds = cp_expr_general(model, query, model_vars, ast.parse(left, mode="eval").body, aux_state)
    right_expr, right_bounds = cp_expr_general(model, query, model_vars, ast.parse(right, mode="eval").body, aux_state)
    diff = left_expr - right_expr
    if op in {"<=", "<"}:
        model.Add(diff <= 0)
    elif op in {">=", ">"}:
        model.Add(diff >= 0)
    elif op == "==":
        tol = int(round(CP_ABS_TOL * CP_VAR_SCALE))
        model.Add(diff <= tol)
        model.Add(diff >= -tol)
    else:
        raise ValueError(f"unsupported comparator {op}")


def run_cp_sat_backend(
    method: str,
    query: dict[str, Any],
    selected_rule_ids: list[str],
    rule_by_id: dict[str, dict[str, Any]],
) -> full.MethodResult:
    selected = sorted(selected_rule_ids)
    if not selected:
        return full.MethodResult(False, [], None, None, "no_selected_rules")
    variables = list(query.get("decision_variables", {}))
    constants = numeric_constants(query)
    constraints = constraints_for_selected(query, selected, rule_by_id)
    try:
        model = cp_model.CpModel()
        model_vars: dict[str, Any] = {}
        for name in variables:
            spec = query["decision_variables"][name]
            lower = math.ceil(float(spec.get("lower", 0.0)) * CP_VAR_SCALE)
            upper = math.floor(float(spec.get("upper", 1.0)) * CP_VAR_SCALE)
            if lower > upper:
                return full.MethodResult(False, selected, None, None, "cp_sat_empty_scaled_domain")
            model_vars[name] = model.NewIntVar(lower, upper, f"x__{name}")

        aux_index = 0
        for constraint in constraints:
            expression = constraint.get("checker_expression") or constraint.get("expression")
            if not expression:
                continue
            try:
                aux_index = add_cp_expression_constraint_robust(
                    model,
                    query,
                    model_vars,
                    str(expression),
                    variables,
                    constants,
                    aux_index,
                )
            except Exception as robust_exc:
                try:
                    aux_state = {"i": aux_index}
                    add_cp_expression_constraint_general(
                        model,
                        query,
                        model_vars,
                        str(expression),
                        aux_state,
                    )
                    aux_index = aux_state["i"]
                except Exception as general_exc:  # noqa: BLE001
                    raise ValueError(
                        f"{robust_exc}; general integer expression fallback failed: {general_exc}"
                    ) from general_exc

        objective_coeff, _objective_const, objective_max_terms = objective_components(query, variables, constants)
        objective_terms = [
            int(round(weight * CP_OBJECTIVE_SCALE)) * model_vars[name]
            for name, weight in objective_coeff.items()
            if abs(weight) > 1e-12
        ]
        for max_coef, max_args in objective_max_terms:
            if max_coef < -1e-12:
                raise ValueError("negative max objective coefficient is unsupported")
            arg_exprs = [cp_linear_expr(model_vars, coeff, const) for coeff, const in max_args]
            bounds = [cp_linear_bounds(query, coeff, const) for coeff, const in max_args]
            lower = min(lb for lb, _ub in bounds)
            upper = max(ub for _lb, ub in bounds)
            aux = model.NewIntVar(lower, upper, f"obj_max_aux_{len(objective_terms)}")
            model.AddMaxEquality(aux, arg_exprs)
            objective_terms.append(int(round(max_coef * CP_OBJECTIVE_SCALE / CP_COEFF_SCALE)) * aux)
        if objective_terms:
            model.Minimize(sum(objective_terms))
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10.0
        solver.parameters.num_search_workers = 1
        status = solver.Solve(model)
        if status not in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
            return full.MethodResult(False, selected, None, None, f"cp_sat_{solver.StatusName(status).lower()}")
        x = {name: solver.Value(var) / CP_VAR_SCALE for name, var in model_vars.items()}
        formal = full.base.constraints_satisfied(constraints, full.base.with_query_values(query, x))
        return full.MethodResult(True, selected, x, formal)
    except Exception as exc:  # noqa: BLE001
        return full.MethodResult(False, selected, None, None, f"cp_sat_unsupported:{exc}")


def scip_linear_expr(model_vars: dict[str, Any], coeff: dict[str, float], const: float) -> Any:
    return quicksum(value * model_vars[name] for name, value in coeff.items()) + const


def add_scip_constraint(
    model: Model,
    model_vars: dict[str, Any],
    coeff: dict[str, float],
    const: float,
    sense: str,
) -> None:
    if not coeff:
        ok = (
            (sense == "<=" and const <= CP_ABS_TOL)
            or (sense == ">=" and const >= -CP_ABS_TOL)
            or (sense == "==" and abs(const) <= CP_ABS_TOL)
        )
        if not ok:
            raise ValueError("constant constraint is infeasible")
        return
    expr = scip_linear_expr(model_vars, coeff, const)
    if sense == "<=":
        model.addCons(expr <= 0.0)
    elif sense == ">=":
        model.addCons(expr >= 0.0)
    elif sense == "==":
        model.addCons(expr <= CP_ABS_TOL)
        model.addCons(expr >= -CP_ABS_TOL)
    else:
        raise ValueError(f"unsupported SCIP sense {sense}")


def add_scip_expression_constraint(
    model: Model,
    model_vars: dict[str, Any],
    expression: str,
    variables: list[str],
    constants: dict[str, float],
) -> None:
    if not full.base.COMPARATOR_RE.search(str(expression)):
        return
    parser = full.base.LinearExpr(set(variables), constants)
    left, op, right = full.base.split_constraint(str(expression))
    if op == "=":
        op = "=="
    left_abs = abs_inner_expression(left)
    right_abs = abs_inner_expression(right)
    if not left_abs and not right_abs:
        for coeff, const, sense in linear_constraint_variants(str(expression), variables, constants):
            add_scip_constraint(model, model_vars, coeff, const, sense)
        return
    if left_abs and right_abs:
        raise ValueError("abs on both sides is unsupported")
    if left_abs:
        inner = parser.parse(left_abs)
        other = parser.parse(right)
        lhs = abs(scip_linear_expr(model_vars, inner[0], inner[1]))
        rhs = scip_linear_expr(model_vars, other[0], other[1])
    else:
        inner = parser.parse(right_abs or "")
        other = parser.parse(left)
        lhs = scip_linear_expr(model_vars, other[0], other[1])
        rhs = abs(scip_linear_expr(model_vars, inner[0], inner[1]))
    if op in {"<=", "<"}:
        model.addCons(lhs <= rhs)
    elif op in {">=", ">"}:
        model.addCons(lhs >= rhs)
    elif op == "==":
        model.addCons(lhs <= rhs + CP_ABS_TOL)
        model.addCons(lhs >= rhs - CP_ABS_TOL)
    else:
        raise ValueError(f"unsupported abs comparator {op}")


def scip_expr_general(
    model: Model,
    query: dict[str, Any],
    model_vars: dict[str, Any],
    node: ast.AST,
    aux_state: dict[str, int],
) -> Any:
    if isinstance(node, ast.Name):
        if node.id in model_vars:
            return model_vars[node.id]
        if node.id in query.get("scenario_facts", {}) and isinstance(query["scenario_facts"][node.id], (int, float)):
            return float(query["scenario_facts"][node.id])
        raise ValueError(f"unknown symbol {node.id}")
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.UnaryOp):
        expr = scip_expr_general(model, query, model_vars, node.operand, aux_state)
        if isinstance(node.op, ast.USub):
            return -expr
        if isinstance(node.op, ast.UAdd):
            return expr
        raise ValueError("unsupported unary op")
    if isinstance(node, ast.BinOp):
        left = scip_expr_general(model, query, model_vars, node.left, aux_state)
        right = scip_expr_general(model, query, model_vars, node.right, aux_state)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        raise ValueError("unsupported binary op")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "abs" and len(node.args) == 1 and not node.keywords:
            return abs(scip_expr_general(model, query, model_vars, node.args[0], aux_state))
        if node.func.id == "max" and len(node.args) >= 2 and not node.keywords:
            aux = model.addVar(name=f"max_general_{aux_state['i']}", lb=-1e9, ub=1e9, vtype="C")
            aux_state["i"] += 1
            for arg in node.args:
                model.addCons(aux >= scip_expr_general(model, query, model_vars, arg, aux_state))
            return aux
    raise ValueError("unsupported expression node")


def add_scip_expression_constraint_general(
    model: Model,
    query: dict[str, Any],
    model_vars: dict[str, Any],
    expression: str,
    aux_state: dict[str, int],
) -> None:
    if not full.base.COMPARATOR_RE.search(str(expression)):
        return
    left, op, right = full.base.split_constraint(str(expression))
    if op == "=":
        op = "=="
    left_expr = scip_expr_general(model, query, model_vars, ast.parse(left, mode="eval").body, aux_state)
    right_expr = scip_expr_general(model, query, model_vars, ast.parse(right, mode="eval").body, aux_state)
    if op in {"<=", "<"}:
        model.addCons(left_expr <= right_expr)
    elif op in {">=", ">"}:
        model.addCons(left_expr >= right_expr)
    elif op == "==":
        model.addCons(left_expr <= right_expr + CP_ABS_TOL)
        model.addCons(left_expr >= right_expr - CP_ABS_TOL)
    else:
        raise ValueError(f"unsupported comparator {op}")


def run_scip_backend(
    method: str,
    query: dict[str, Any],
    selected_rule_ids: list[str],
    rule_by_id: dict[str, dict[str, Any]],
) -> full.MethodResult:
    selected = sorted(selected_rule_ids)
    if not selected:
        return full.MethodResult(False, [], None, None, "no_selected_rules")
    variables = list(query.get("decision_variables", {}))
    constants = numeric_constants(query)
    constraints = constraints_for_selected(query, selected, rule_by_id)
    try:
        model = Model()
        model.hideOutput()
        model.setParam("limits/time", 10.0)
        model_vars = {
            name: model.addVar(
                name=f"x__{name}",
                lb=float(query["decision_variables"][name].get("lower", 0.0)),
                ub=float(query["decision_variables"][name].get("upper", 1.0)),
                vtype="C",
            )
            for name in variables
        }
        aux_state = {"i": 0}
        for constraint in constraints:
            expression = constraint.get("checker_expression") or constraint.get("expression")
            if not expression:
                continue
            try:
                add_scip_expression_constraint_general(model, query, model_vars, str(expression), aux_state)
            except Exception:
                add_scip_expression_constraint(model, model_vars, str(expression), variables, constants)

        objective_coeff, _objective_const, objective_max_terms = objective_components(query, variables, constants)
        objective = quicksum(weight * model_vars[name] for name, weight in objective_coeff.items())
        for index, (max_coef, max_args) in enumerate(objective_max_terms):
            if max_coef < -1e-12:
                raise ValueError("negative max objective coefficient is unsupported")
            aux = model.addVar(name=f"obj_max_aux_{index}", lb=-1e9, ub=1e9, vtype="C")
            for coeff, const in max_args:
                model.addCons(aux >= scip_linear_expr(model_vars, coeff, const))
            objective += max_coef * aux
        model.setObjective(objective, "minimize")
        model.optimize()
        status = str(model.getStatus()).lower()
        if model.getNSols() <= 0:
            return full.MethodResult(False, selected, None, None, f"scip_{status}")
        x = {name: float(model.getVal(var)) for name, var in model_vars.items()}
        formal = full.base.constraints_satisfied(constraints, full.base.with_query_values(query, x))
        return full.MethodResult(True, selected, x, formal)
    except Exception as exc:  # noqa: BLE001
        return full.MethodResult(False, selected, None, None, f"scip_unsupported:{exc}")


def native_guard_selected(
    spec: full.DatasetSpec,
    grounding_task: dict[str, Any],
    candidate_ids: list[str],
    rule_by_id: dict[str, dict[str, Any]],
) -> list[str]:
    return full.select_native_applicable(spec, grounding_task, candidate_ids, rule_by_id)


def run_method_with_grounding_policy(
    spec: full.DatasetSpec,
    method: str,
    query: dict[str, Any],
    grounding_task: dict[str, Any],
    candidate_rules: list[dict[str, Any]],
    candidate_ids: list[str],
    selected_valid_ids: list[str],
    rule_library: dict[str, Any],
    rule_by_id: dict[str, dict[str, Any]],
) -> full.MethodResult:
    if method == "Flat baseline":
        return full.run_flat(query, candidate_ids, rule_by_id)
    if method == "Native ASP + clingo":
        return full.run_asp(spec, query, grounding_task, candidate_ids, rule_library, rule_by_id, native=True)
    if method == "Native SLSQP":
        selected = native_guard_selected(spec, grounding_task, candidate_ids, rule_by_id)
        return run_slsqp_backend(method, query, selected, rule_by_id)
    if method == "Native MILP + HiGHS":
        selected = native_guard_selected(spec, grounding_task, candidate_ids, rule_by_id)
        return run_highs_compiled_backend(query, selected, rule_by_id)
    if method == "Native CP-SAT + OR-Tools":
        selected = native_guard_selected(spec, grounding_task, candidate_ids, rule_by_id)
        return run_cp_sat_backend(method, query, selected, rule_by_id)
    if method == "Native SCIP":
        selected = native_guard_selected(spec, grounding_task, candidate_ids, rule_by_id)
        return run_scip_backend(method, query, selected, rule_by_id)
    if method == "CTHR default":
        if spec.domain == "aviation" and len(query.get("decision_variables", {})) >= 12:
            return run_scip_backend(method, query, selected_valid_ids, rule_by_id)
        return full.run_cthr_default(
            spec,
            query,
            grounding_task,
            candidate_rules,
            rule_by_id,
            selected_valid_ids,
        )
    if method == "CTHR-style ASP + clingo":
        return run_fixed_grounded_default(method, query, selected_valid_ids, rule_by_id)
    if method == "CTHR-style SLSQP":
        return run_slsqp_backend(method, query, selected_valid_ids, rule_by_id)
    if method == "CTHR-style pure HiGHS":
        return run_highs_compiled_backend(query, selected_valid_ids, rule_by_id)
    if method == "CTHR-style HiGHS":
        return run_highs_with_scip_repair_backend(method, query, selected_valid_ids, rule_by_id)
    if method == "CTHR-style CP-SAT + OR-Tools":
        return run_cp_sat_backend(method, query, selected_valid_ids, rule_by_id)
    if method == "CTHR-style SCIP":
        return run_scip_backend(method, query, selected_valid_ids, rule_by_id)
    raise ValueError(method)


def evaluate_dataset_with_grounding_policy(
    spec: full.DatasetSpec,
    grounding_source_label: str | None = None,
    method_specs: list[tuple[str, str]] | None = None,
    max_tasks: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
        raise ValueError(f"{spec.name} grounding result IDs do not match")

    rows: list[dict[str, Any]] = []
    grounding_audit: dict[str, Any] = {}
    selected_method_specs = method_specs or METHOD_SPECS
    task_ids = sorted(algorithm_inputs)
    if max_tasks is not None:
        task_ids = task_ids[:max_tasks]
    for task_id in task_ids:
        grounding_task = dict(algorithm_inputs[task_id])
        query = full.prepare_query(grounding_task, scenario_models[task_id])
        query["_compiled_rule_constraint_templates_by_id"] = templates_by_rule
        reference = references[task_id]
        feasible = full.reference_feasible(reference, query)
        reference_ids = full.reference_rule_ids(reference)
        grounding_row = grounding_results[task_id]
        candidate_ids = full.ids_from_grounding(grounding_row, "candidate_rule_ids_generated")
        selected_valid_ids = full.ids_from_grounding(grounding_row, "predicted_valid_rule_ids")
        candidate_rules = [rule_by_id[rule_id] for rule_id in candidate_ids if rule_id in rule_by_id]
        grounding_audit[task_id] = {
            "candidate_rule_count": len(candidate_ids),
            "cthr_predicted_valid_rule_count": len(selected_valid_ids),
            "grounding_result_exact_match": bool(grounding_row.get("Exact Match")),
        }
        for method, method_type in selected_method_specs:
            start = time.perf_counter()
            result = run_method_with_grounding_policy(
                spec,
                method,
                query,
                grounding_task,
                candidate_rules,
                candidate_ids,
                selected_valid_ids,
                rule_library,
                rule_by_id,
            )
            elapsed = (time.perf_counter() - start) * 1000.0
            predicted = sorted(result.predicted_rule_ids) if (result.supported or method in CTHR_GROUNDED_METHODS) else []
            precision = full.base.method_rule_precision(predicted, reference_ids)
            recall = full.base.method_rule_recall(predicted, reference_ids)
            precision = 0.0 if precision is None else precision
            recall = 0.0 if recall is None else recall
            sem_ok = full.semantic_valid(feasible, result.optimized_x, predicted, reference_ids) if result.supported else False
            formal_ok = bool(result.formal_feasible) if result.supported else False
            rows.append(
                {
                    "Dataset": spec.name,
                    "task_id": task_id,
                    "target_interaction": full.target_interaction(reference),
                    "Method": method,
                    "Method type": method_type,
                    "grounding_policy": (
                        "fixed_predicted_valid_rule_ids"
                        if method in CTHR_GROUNDED_METHODS
                        else "broad_candidate_rule_ids"
                    ),
                    "grounded_candidate_count": len(candidate_ids),
                    "cthr_predicted_valid_count": len(selected_valid_ids),
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
        "tasks": len(task_ids),
        "rule_library": str(spec.rule_library),
        "grounding_result": str(spec.grounding_full),
        "constraint_templates": str(spec.constraint_templates),
        "grounding_policy": {
            "flat_and_native_symbolic": "candidate_rule_ids_generated",
            "cthr_default_and_cthr_style_backends": "predicted_valid_rule_ids",
        },
        "grounding": {
            "source": (
                grounding_source_label
                or (
                    "old broad candidate grounding with aviation recall guard and profile_auto_resolver"
                    if spec.domain == "aviation"
                    else "old broad candidate grounding with profile_auto_resolver"
                )
            ),
            "mean_candidate_count": sum(item["candidate_rule_count"] for item in grounding_audit.values())
            / max(1, len(grounding_audit)),
            "mean_cthr_predicted_valid_count": sum(
                item["cthr_predicted_valid_rule_count"] for item in grounding_audit.values()
            )
            / max(1, len(grounding_audit)),
            "cthr_exact_match_rate": sum(1 for item in grounding_audit.values() if item["grounding_result_exact_match"])
            / max(1, len(grounding_audit)),
        },
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Section 6.2 Table 1 all methods with old-candidate profile-auto grounding."
    )
    parser.add_argument("--domain", choices=sorted(DEFAULT_GROUNDING_BY_DOMAIN), default="aviation")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Optional dataset root. If set, standard dataset-layer files are inferred from this root.",
    )
    parser.add_argument("--algorithm-inputs", type=Path, default=None)
    parser.add_argument("--scenario-models", type=Path, default=None)
    parser.add_argument("--evaluation-references", type=Path, default=None)
    parser.add_argument("--rule-library", type=Path, default=None)
    parser.add_argument("--constraint-templates", type=Path, default=None)
    parser.add_argument("--grounding-full", type=Path, default=None)
    parser.add_argument("--grounding-source-label", default=None)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument(
        "--methods",
        default=None,
        help="Optional comma-separated method names for diagnostics. Defaults to all methods.",
    )
    parser.add_argument("--max-tasks", type=int, default=None)
    args = parser.parse_args()

    grounding_full = args.grounding_full or DEFAULT_GROUNDING_BY_DOMAIN[args.domain]
    output_prefix = args.output_prefix or OUTPUT_PREFIX_BY_DOMAIN[args.domain]
    if not grounding_full.exists():
        raise FileNotFoundError(grounding_full)

    base_spec = next(spec for spec in full.DATASETS if spec.domain == args.domain)
    spec_updates: dict[str, Path] = {"grounding_full": grounding_full}
    if args.dataset_root is not None:
        dataset_root = args.dataset_root
        spec_updates.update(
            {
                "root": dataset_root,
                "algorithm_inputs": args.algorithm_inputs
                or dataset_root / "algorithm_inputs" / f"{args.domain}_algorithm_inputs.json",
                "scenario_models": args.scenario_models
                or dataset_root / "scenario_models" / f"{args.domain}_public_scenario_models.json",
                "evaluation_references": args.evaluation_references
                or dataset_root / "evaluation_references" / f"{args.domain}_evaluation_references.json",
                "rule_library": args.rule_library
                or dataset_root / "rule_libraries" / f"full_{args.domain}_rule_library_qwen.json",
                "constraint_templates": args.constraint_templates
                or dataset_root / "constraint_templates" / "compiled_rule_constraint_templates.json",
            }
        )
    else:
        for field_name, value in [
            ("algorithm_inputs", args.algorithm_inputs),
            ("scenario_models", args.scenario_models),
            ("evaluation_references", args.evaluation_references),
            ("rule_library", args.rule_library),
            ("constraint_templates", args.constraint_templates),
        ]:
            if value is not None:
                spec_updates[field_name] = value
    spec = replace(base_spec, **spec_updates)

    method_specs = METHOD_SPECS
    if args.methods:
        requested = {item.strip().lower() for item in args.methods.split(",") if item.strip()}
        method_specs = [
            (method, method_type)
            for method, method_type in METHOD_SPECS
            if method.lower() in requested
        ]
        missing = requested - {method.lower() for method, _method_type in method_specs}
        if missing:
            raise ValueError(f"Unknown method name(s): {sorted(missing)}")
    per_task_rows, dataset_summary = evaluate_dataset_with_grounding_policy(
        spec,
        args.grounding_source_label,
        method_specs=method_specs,
        max_tasks=args.max_tasks,
    )
    aggregate_rows = [
        full.aggregate(per_task_rows, spec.name, method, method_type)
        for method, method_type in method_specs
    ]

    per_task_headers = [
        "Dataset",
        "task_id",
        "target_interaction",
        "Method",
        "Method type",
        "grounding_policy",
        "grounded_candidate_count",
        "cthr_predicted_valid_count",
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
        "per_task_csv": RESULTS_DIR / f"{output_prefix}_per_task.csv",
        "overall_csv": RESULTS_DIR / f"{output_prefix}_overall.csv",
        "overall_md": RESULTS_DIR / f"{output_prefix}_overall.md",
        "overall_json": RESULTS_DIR / f"{output_prefix}_overall.json",
        "report_md": RESULTS_DIR / f"{output_prefix}_report.md",
    }
    full.write_csv(outputs["per_task_csv"], per_task_rows, per_task_headers)
    full.write_csv(outputs["overall_csv"], aggregate_rows, aggregate_headers)
    outputs["overall_md"].write_text(markdown_table(aggregate_rows, aggregate_headers), encoding="utf-8")

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "domain": args.domain,
        "dataset": dataset_summary,
        "methods": [{"Method": method, "Method type": method_type} for method, method_type in method_specs],
        "grounding_full": str(grounding_full),
        "outputs": {key: str(path) for key, path in outputs.items()},
        "aggregate_rows": aggregate_rows,
        "metric_note": (
            "All methods run through the same Section 6.2 Table 1 evaluator. "
            "Flat consumes candidate_rule_ids_generated directly. "
            "Native ASP/CP-SAT/SCIP consume candidate_rule_ids_generated directly. "
            "CTHR default and CTHR-style ASP/CP-SAT/SCIP consume the same predicted_valid_rule_ids "
            "from the grounding file. For CTHR-style methods, rule grounding is fixed before "
            "backend-specific constraint solving."
        ),
    }
    full.write_json(outputs["overall_json"], summary)
    outputs["report_md"].write_text(build_report(aggregate_rows, summary), encoding="utf-8")
    print(json.dumps({"outputs": summary["outputs"], "aggregate_rows": aggregate_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
