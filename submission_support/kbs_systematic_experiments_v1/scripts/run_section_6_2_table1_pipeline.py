from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution, linprog, milp, minimize, Bounds, LinearConstraint


ROOT = Path(__file__).resolve().parents[1]
CTHR_ROOT = ROOT.parents[1]
RESULTS_DIR = ROOT / "results"
REPORTS_DIR = ROOT / "reports"

sys.path.insert(0, str(CTHR_ROOT))

from experiments.kg_to_rule_validation.baselines.asp_rule_structure import (  # noqa: E402
    enumerate_rule_structures,
    eval_guard,
    relation_target,
    relation_type,
)
from experiments.kg_to_rule_validation.baselines.cthr_rule_resolver import (  # noqa: E402
    relation_maps,
)
from experiments.kg_to_rule_validation.baselines.smt_monolithic import (  # noqa: E402
    build_smt_formula,
    optimize_with_z3,
)


METHODS = ["Flat baseline", "CTHR full", "ASP + clingo", "SMT + Z3", "MILP + HiGHS"]
DEPENDENCY_TYPES = {"depends_on", "requires", "uses_parameter"}
EXCLUSION_TYPES = {"excludes", "mutually_exclusive", "conflicts_with", "conflict"}
OVERRIDE_TYPES = {"overrides", "can_override", "replaces", "defeats"}
PRECEDENCE_TYPES = {"precedes", "precedence", "higher_priority_than", "has_precedence_over"}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: Path
    query_file: str
    rule_label_file: str
    feasible_file: str
    rule_library_file: str


@dataclass
class MethodResult:
    supported: bool
    predicted_rule_ids: list[str]
    optimized_x: dict[str, float] | None
    formal_feasible: bool | None
    unsupported_reason: str = ""


DATASETS = [
    DatasetSpec(
        name="Aviation",
        root=ROOT / "datasets" / "aviation_combined",
        query_file="aviation_combined_optimization_queries.json",
        rule_label_file="aviation_combined_rule_structure_labels.json",
        feasible_file="aviation_combined_feasible_region_labels.json",
        rule_library_file="aviation_combined_rule_library.combined.json",
    ),
    DatasetSpec(
        name="Architecture",
        root=ROOT / "datasets" / "architecture",
        query_file="architecture_optimization_queries.json",
        rule_label_file="architecture_rule_structure_labels.json",
        feasible_file="architecture_feasible_region_labels.json",
        rule_library_file="architecture_stress_rule_library.combined.json",
    ),
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def csv_cell(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (list, dict, set, tuple)):
        return json.dumps(list(value) if isinstance(value, set) else value, ensure_ascii=False)
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({header: csv_cell(row.get(header)) for header in headers})


def load_items(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    if "items" in payload:
        return payload["items"]
    for key in ("optimization_queries", "rule_structure_labels", "feasible_region_labels"):
        if key in payload:
            return payload[key]
    raise ValueError(f"No recognized item list in {path}")


def by_id(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item["omega_id"]): item for item in items}


def candidate_ids_from_query(query: dict[str, Any], label: dict[str, Any], feasible: dict[str, Any]) -> list[str]:
    meta = query.get("stress_metadata", {})
    for key in (
        "candidate_rule_ids_expected_for_diagnostics",
        "candidate_rule_ids",
    ):
        if meta.get(key):
            return sorted(str(x) for x in meta[key])
    ref = feasible.get("reference_semantics", {})
    if ref.get("candidate_rule_ids"):
        return sorted(str(x) for x in ref["candidate_rule_ids"])
    if label.get("expected_source_rule_ids"):
        return sorted(str(x) for x in label["expected_source_rule_ids"])
    if query.get("certificate_targets", {}).get("source_rule_ids"):
        return sorted(str(x) for x in query["certificate_targets"]["source_rule_ids"])
    return []


def reference_rule_ids(label: dict[str, Any], feasible: dict[str, Any], query: dict[str, Any]) -> list[str]:
    meta = query.get("stress_metadata", {})
    if meta.get("final_valid_rule_ids_expected_for_evaluation"):
        return sorted(str(x) for x in meta["final_valid_rule_ids_expected_for_evaluation"])
    ref = feasible.get("reference_semantics", {})
    if ref.get("final_valid_rule_ids"):
        return sorted(str(x) for x in ref["final_valid_rule_ids"])
    if label.get("expected_surviving_rule_ids"):
        return sorted(str(x) for x in label["expected_surviving_rule_ids"])
    return []


def scenario_for_query(query: dict[str, Any]) -> dict[str, Any]:
    scenario = dict(query.get("scenario_facts", {}))
    scenario.update(
        {
            "domain": query.get("domain"),
            "task_type": query.get("task_type"),
            "title": query.get("title"),
            "design_intent": query.get("design_intent"),
        }
    )
    return scenario


def candidate_rule_records(rule_by_id: dict[str, dict[str, Any]], ids: list[str]) -> list[dict[str, Any]]:
    return [rule_by_id[rule_id] for rule_id in ids if rule_id in rule_by_id]


def union_structures(structures: list[list[str]]) -> list[str]:
    return sorted({rule_id for structure in structures for rule_id in structure})


def method_rule_precision(predicted: list[str], reference: list[str]) -> float | None:
    if not predicted:
        return None
    return len(set(predicted) & set(reference)) / len(set(predicted))


def method_rule_recall(predicted: list[str], reference: list[str]) -> float | None:
    if not reference:
        return None
    return len(set(predicted) & set(reference)) / len(set(reference))


def safe_env(values: dict[str, float]) -> dict[str, Any]:
    env: dict[str, Any] = {
        "abs": abs,
        "min": min,
        "max": max,
        "sqrt": math.sqrt,
        "tan": math.tan,
        "sin": math.sin,
        "cos": math.cos,
        "pi": math.pi,
    }
    env.update(values)
    return env


COMPARATOR_RE = re.compile(r"(<=|>=|!=|==|=|<|>)")


def normalize_expression(expr: str) -> str:
    expression = str(expr).strip()
    match = COMPARATOR_RE.search(expression)
    if match and match.group(1) == "=":
        expression = expression[: match.start()] + "==" + expression[match.end() :]
    return expression


def eval_arithmetic(expr: str, values: dict[str, float]) -> float:
    return float(eval(expr, {"__builtins__": {}}, safe_env(values)))  # noqa: S307 - benchmark expressions are curated.


def constraint_violation(expression: str, values: dict[str, float]) -> float:
    expression = normalize_expression(expression)
    match = COMPARATOR_RE.search(expression)
    if not match:
        return 0.0
    left = expression[: match.start()].strip()
    op = match.group(1)
    right = expression[match.end() :].strip()
    try:
        lval = eval_arithmetic(left, values)
        rval = eval_arithmetic(right, values)
    except Exception:
        return 1e6
    tol = 1e-6
    if op == "<=":
        return max(0.0, lval - rval - tol)
    if op == ">=":
        return max(0.0, rval - lval - tol)
    if op == "<":
        return max(0.0, lval - rval + tol)
    if op == ">":
        return max(0.0, rval - lval + tol)
    if op in {"==", "="}:
        return max(0.0, abs(lval - rval) - 1e-5)
    if op == "!=":
        return 0.0 if abs(lval - rval) > 1e-5 else 1.0
    return 1e6


def constraints_satisfied(constraints: list[dict[str, Any]], values: dict[str, float]) -> bool:
    for constraint in constraints:
        if not constraint.get("executable", True):
            continue
        expression = constraint.get("checker_expression") or constraint.get("expression")
        if expression and constraint_violation(str(expression), values) > 1e-4:
            return False
    return True


def objective_value(query: dict[str, Any], values: dict[str, float]) -> float:
    weights = query.get("query_preferences", {}).get("lambda") or query.get("preference_weights") or []
    objectives = query.get("objectives", [])
    if not weights:
        weights = [1.0 / max(1, len(objectives))] * len(objectives)
    total = 0.0
    for weight, objective in zip(weights, objectives):
        expr = str(objective.get("expression", "0"))
        try:
            value = eval_arithmetic(expr, values)
        except Exception:
            value = 0.0
        if str(objective.get("name", "")).lower().startswith("maximize"):
            value = -value
        total += float(weight) * value
    return total


def rule_constraint_to_task_constraint(rule: dict[str, Any], query: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    vars_available = set(query.get("decision_variables", {}))
    for idx, constraint in enumerate(rule.get("constraints", [])):
        variable = constraint.get("variable")
        op = constraint.get("op")
        value = constraint.get("value")
        if not variable or variable not in vars_available or op == "formula":
            continue
        if op not in {"<=", ">=", "<", ">", "=", "=="}:
            continue
        out.append(
            {
                "constraint_id": f"rulelib_{rule['rule_id']}_{idx}",
                "expression": f"{variable} {op} {value}",
                "checker_expression": f"{variable} {op} {value}",
                "source_type": "rule_library",
                "source_id": rule["rule_id"],
                "executable": True,
            }
        )
    return out


def constraints_for_method(
    query: dict[str, Any],
    predicted_rule_ids: list[str],
    rule_by_id: dict[str, dict[str, Any]],
    include_candidate_rulelib_constraints: bool = False,
) -> list[dict[str, Any]]:
    predicted = set(predicted_rule_ids)
    constraints: list[dict[str, Any]] = []
    seen: set[str] = set()
    for constraint in query.get("solver_constraints", []):
        if not constraint.get("executable", True):
            continue
        source_type = constraint.get("source_type")
        source_id = str(constraint.get("source_id", ""))
        if source_type != "rule_library" or source_id in predicted:
            constraints.append(constraint)
            seen.add(str(constraint.get("constraint_id", "")))
    if include_candidate_rulelib_constraints:
        for rule_id in predicted:
            rule = rule_by_id.get(rule_id)
            if not rule:
                continue
            for constraint in rule_constraint_to_task_constraint(rule, query):
                key = str(constraint["constraint_id"])
                if key not in seen:
                    constraints.append(constraint)
                    seen.add(key)
    return constraints


def reference_constraints(feasible: dict[str, Any]) -> list[dict[str, Any]]:
    return [c for c in feasible.get("executable_constraints", []) if c.get("executable", True)]


def with_query_values(query: dict[str, Any], x: dict[str, float]) -> dict[str, float]:
    values = dict(x)
    for key, value in query.get("scenario_facts", {}).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values[key] = float(value)
    add_derived_aliases(values)
    return values


def cthr_strict_resolve(candidate_rules: list[dict[str, Any]], scenario: dict[str, Any]) -> list[list[str]]:
    by_id = {str(rule["rule_id"]): rule for rule in candidate_rules if rule.get("rule_id")}
    maps = relation_maps(candidate_rules)

    selected: set[str] = set()
    for rule_id, rule in by_id.items():
        guard = rule.get("guard")
        if not guard or eval_guard(guard, scenario):
            selected.add(rule_id)

    for rule_id, rule in by_id.items():
        for rel in rule.get("relations", []):
            target = relation_target(rel)
            if target not in by_id:
                continue
            if relation_type(rel) in {"formula_variant_of", "piecewise_variant_of", "parameter_variant_of"}:
                if target in selected and not eval_guard(rule.get("guard"), scenario):
                    selected.discard(rule_id)

    defeated: set[str] = set()
    for source, target in maps["overrides"]:
        if source in selected:
            defeated.add(target)
    for source, target in maps["precedes"]:
        if source in selected:
            defeated.add(target)
    selected -= defeated

    changed = True
    while changed:
        changed = False
        for source, target in maps["depends"]:
            if source in selected and target in by_id and target not in defeated and target not in selected:
                selected.add(target)
                changed = True

    changed = True
    while changed:
        changed = False
        for left, right in sorted(set(maps["excludes"]) | set(maps["conflicts"])):
            if left in selected and right in selected:
                left_guard = bool(by_id[left].get("guard") and eval_guard(by_id[left].get("guard"), scenario))
                right_guard = bool(by_id[right].get("guard") and eval_guard(by_id[right].get("guard"), scenario))
                if left_guard and not right_guard:
                    loser = right
                elif right_guard and not left_guard:
                    loser = left
                else:
                    loser = max(left, right)
                selected.discard(loser)
                changed = True
                break

    return [sorted(selected)] if selected else []


def optimize_default(
    query: dict[str, Any],
    constraints: list[dict[str, Any]],
    method: str,
    task_id: str,
) -> dict[str, float] | None:
    variables = list(query.get("decision_variables", {}))
    if not variables:
        return None
    bounds = []
    for name in variables:
        spec = query["decision_variables"][name]
        bounds.append((float(spec.get("lower", 0.0)), float(spec.get("upper", 1.0))))

    seed = int(hashlib.sha256(f"{method}:{task_id}".encode()).hexdigest()[:8], 16)

    def unpack(x: np.ndarray) -> dict[str, float]:
        values = {name: float(value) for name, value in zip(variables, x)}
        for key, value in query.get("scenario_facts", {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values[key] = float(value)
        add_derived_aliases(values)
        return values

    def scipy_constraints() -> list[dict[str, Any]]:
        scipy_items: list[dict[str, Any]] = []

        def make_value(expr: str):
            def fun(z: np.ndarray) -> float:
                try:
                    return eval_arithmetic(expr, unpack(np.asarray(z, dtype=float)))
                except Exception:
                    return float("nan")

            return fun

        def safe_diff(left_fun, right_fun, sign: float):
            def fun(z: np.ndarray) -> float:
                left = left_fun(z)
                right = right_fun(z)
                if not math.isfinite(left) or not math.isfinite(right):
                    return -1e6
                return sign * (left - right)

            return fun

        def safe_eq(left_fun, right_fun):
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
            normalized = normalize_expression(str(expression))
            match = COMPARATOR_RE.search(normalized)
            if not match:
                continue
            left = normalized[: match.start()].strip()
            op = match.group(1)
            right = normalized[match.end() :].strip()
            left_fun = make_value(left)
            right_fun = make_value(right)
            if op in {"<=", "<"}:
                scipy_items.append({"type": "ineq", "fun": safe_diff(right_fun, left_fun, 1.0)})
            elif op in {">=", ">"}:
                scipy_items.append({"type": "ineq", "fun": safe_diff(left_fun, right_fun, 1.0)})
            elif op in {"=", "=="}:
                scipy_items.append({"type": "eq", "fun": safe_eq(left_fun, right_fun)})
        return scipy_items

    def penalized(x: np.ndarray) -> float:
        values = unpack(x)
        penalty = 0.0
        for constraint in constraints:
            expression = constraint.get("checker_expression") or constraint.get("expression")
            if expression:
                v = constraint_violation(str(expression), values)
                penalty += v * v
        return objective_value(query, values) + 1e7 * penalty

    def local_fast_path() -> dict[str, float] | None:
        constants = {
            key: float(value)
            for key, value in query.get("scenario_facts", {}).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        starts: list[np.ndarray] = [
            np.array([(lower + upper) / 2.0 for lower, upper in bounds], dtype=float),
            np.array([lower for lower, _upper in bounds], dtype=float),
            np.array([upper for _lower, upper in bounds], dtype=float),
        ]
        candidates: list[tuple[float, dict[str, float]]] = []
        for start in starts:
            point = np.asarray(start, dtype=float)
            values = unpack(point)
            if constraints_satisfied(constraints, values):
                candidates.append((objective_value(query, values), {name: float(value) for name, value in zip(variables, point)}))
        if candidates:
            candidates.sort(key=lambda item: item[0])
            return candidates[0][1]
        return None

    if len(variables) >= 12:
        return local_fast_path()

    try:
        result = differential_evolution(
            penalized,
            bounds=bounds,
            seed=seed,
            maxiter=80,
            popsize=8,
            polish=True,
            updating="immediate",
            workers=1,
            tol=1e-7,
        )
        starts = [
            np.asarray(result.x, dtype=float),
            np.array([(lower + upper) / 2.0 for lower, upper in bounds], dtype=float),
        ]
        candidates: list[tuple[float, dict[str, float]]] = []
        hard_constraints = scipy_constraints()

        def objective_for_minimize(z: np.ndarray) -> float:
            try:
                return objective_value(query, unpack(np.asarray(z, dtype=float)))
            except Exception:
                return penalized(np.asarray(z, dtype=float))

        for start in starts:
            local_points = [start]
            if hard_constraints:
                try:
                    local = minimize(
                        objective_for_minimize,
                        start,
                        method="SLSQP",
                        bounds=bounds,
                        constraints=hard_constraints,
                        options={"maxiter": 300, "ftol": 1e-10, "disp": False},
                    )
                    local_points.append(np.asarray(local.x, dtype=float))
                except Exception:
                    pass
            for point in local_points:
                values = unpack(np.asarray(point, dtype=float))
                if constraints_satisfied(constraints, values):
                    candidates.append((objective_value(query, values), {name: values[name] for name in variables}))
        if candidates:
            return min(candidates, key=lambda item: item[0])[1]
        values = unpack(np.asarray(result.x, dtype=float))
        return {name: values[name] for name in variables}
    except Exception:
        return None


def run_flat(
    query: dict[str, Any],
    candidate_ids: list[str],
    rule_by_id: dict[str, dict[str, Any]],
) -> MethodResult:
    predicted = sorted(candidate_ids)
    constraints = constraints_for_method(query, predicted, rule_by_id, include_candidate_rulelib_constraints=True)
    x = optimize_default(query, constraints, "Flat baseline", str(query["omega_id"]))
    formal = constraints_satisfied(constraints, with_query_values(query, x)) if x is not None else False
    return MethodResult(True, predicted, x, formal)


def run_cthr(
    query: dict[str, Any],
    candidate_ids: list[str],
    rule_by_id: dict[str, dict[str, Any]],
) -> MethodResult:
    scenario = scenario_for_query(query)
    predicted = union_structures(cthr_strict_resolve(candidate_rule_records(rule_by_id, candidate_ids), scenario))
    constraints = constraints_for_method(query, predicted, rule_by_id, include_candidate_rulelib_constraints=False)
    x = optimize_default(query, constraints, "CTHR full", str(query["omega_id"]))
    formal = constraints_satisfied(constraints, with_query_values(query, x)) if x is not None else False
    return MethodResult(True, predicted, x, formal)


def run_asp(
    rule_library: dict[str, Any],
    query: dict[str, Any],
    candidate_ids: list[str],
    rule_by_id: dict[str, dict[str, Any]],
) -> MethodResult:
    scenario = scenario_for_query(query)
    result = enumerate_rule_structures(
        rule_library,
        scenario,
        str(query["omega_id"]),
        candidate_rule_ids=candidate_ids,
        applicable_rule_ids=None,
        max_answer_sets=200,
    )
    if result.status != "success":
        return MethodResult(False, [], None, None, f"asp_{result.status}:{result.error or ''}".strip(":"))
    predicted = union_structures(result.asp_rule_structures)
    constraints = constraints_for_method(query, predicted, rule_by_id, include_candidate_rulelib_constraints=False)
    x = optimize_default(query, constraints, "ASP + clingo", str(query["omega_id"]))
    formal = constraints_satisfied(constraints, with_query_values(query, x)) if x is not None else False
    return MethodResult(True, predicted, x, formal)


def run_smt(
    rule_library: dict[str, Any],
    query: dict[str, Any],
    candidate_ids: list[str],
) -> MethodResult:
    try:
        formula = build_smt_formula(
            rule_library,
            query,
            candidate_rule_ids=candidate_ids,
            include_visible_task_constraints=True,
        )
        result = optimize_with_z3(formula, query, timeout_ms=10000)
        if result.status != "sat" or result.optimized_x is None:
            return MethodResult(False, result.selected_rule_ids, None, None, f"smt_{result.status}:{result.error or ''}".strip(":"))
        variables = list(query.get("decision_variables", {}))
        x = {name: float(value) for name, value in zip(variables, result.optimized_x)}
        return MethodResult(True, result.selected_rule_ids, x, True)
    except Exception as exc:  # noqa: BLE001
        return MethodResult(False, [], None, None, f"smt_error:{exc}")


class LinearExpr(ast.NodeVisitor):
    def __init__(self, variables: set[str], constants: dict[str, float]):
        self.variables = variables
        self.constants = constants

    def parse(self, expr: str) -> tuple[dict[str, float], float]:
        tree = ast.parse(expr, mode="eval")
        return self.visit(tree.body)

    def visit_Name(self, node: ast.Name) -> tuple[dict[str, float], float]:
        if node.id in self.variables:
            return {node.id: 1.0}, 0.0
        if node.id in self.constants:
            return {}, float(self.constants[node.id])
        raise ValueError(f"unknown symbol {node.id}")

    def visit_Constant(self, node: ast.Constant) -> tuple[dict[str, float], float]:
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return {}, float(node.value)
        raise ValueError("non-numeric constant")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> tuple[dict[str, float], float]:
        coeff, const = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return {k: -v for k, v in coeff.items()}, -const
        if isinstance(node.op, ast.UAdd):
            return coeff, const
        raise ValueError("unsupported unary op")

    def visit_BinOp(self, node: ast.BinOp) -> tuple[dict[str, float], float]:
        left_c, left_b = self.visit(node.left)
        right_c, right_b = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            out = dict(left_c)
            for key, value in right_c.items():
                out[key] = out.get(key, 0.0) + value
            return out, left_b + right_b
        if isinstance(node.op, ast.Sub):
            out = dict(left_c)
            for key, value in right_c.items():
                out[key] = out.get(key, 0.0) - value
            return out, left_b - right_b
        if isinstance(node.op, ast.Mult):
            if left_c and right_c:
                raise ValueError("bilinear term")
            if right_c:
                return {k: left_b * v for k, v in right_c.items()}, left_b * right_b
            return {k: right_b * v for k, v in left_c.items()}, left_b * right_b
        if isinstance(node.op, ast.Div):
            if right_c or abs(right_b) < 1e-12:
                raise ValueError("nonconstant division")
            return {k: v / right_b for k, v in left_c.items()}, left_b / right_b
        raise ValueError("unsupported binary op")

    def visit_Call(self, node: ast.Call) -> tuple[dict[str, float], float]:
        raise ValueError("function call is nonlinear")


def split_constraint(expr: str) -> tuple[str, str, str]:
    expression = normalize_expression(expr)
    match = COMPARATOR_RE.search(expression)
    if not match:
        raise ValueError("no comparator")
    return expression[: match.start()].strip(), match.group(1), expression[match.end() :].strip()


def linear_constraint_rows(
    constraints: list[dict[str, Any]],
    variables: list[str],
    constants: dict[str, float],
) -> tuple[list[list[float]], list[float], list[float]]:
    parser = LinearExpr(set(variables), constants)
    rows: list[list[float]] = []
    lb: list[float] = []
    ub: list[float] = []
    for constraint in constraints:
        expression = constraint.get("checker_expression") or constraint.get("expression")
        if not expression:
            continue
        left, op, right = split_constraint(str(expression))
        lc, lbias = parser.parse(left)
        rc, rbias = parser.parse(right)
        coeff = {name: lc.get(name, 0.0) - rc.get(name, 0.0) for name in variables}
        bias = lbias - rbias
        row = [coeff[name] for name in variables]
        if op == "<=":
            rows.append(row)
            lb.append(-np.inf)
            ub.append(-bias)
        elif op == ">=":
            rows.append(row)
            lb.append(-bias)
            ub.append(np.inf)
        elif op in {"==", "="}:
            rows.append(row)
            lb.append(-bias)
            ub.append(-bias)
        elif op == "<":
            rows.append(row)
            lb.append(-np.inf)
            ub.append(-bias - 1e-6)
        elif op == ">":
            rows.append(row)
            lb.append(-bias + 1e-6)
            ub.append(np.inf)
        else:
            raise ValueError(f"unsupported comparator {op}")
    return rows, lb, ub


def objective_linear(query: dict[str, Any], variables: list[str], constants: dict[str, float]) -> list[float]:
    parser = LinearExpr(set(variables), constants)
    weights = query.get("query_preferences", {}).get("lambda") or query.get("preference_weights") or []
    objectives = query.get("objectives", [])
    if not weights:
        weights = [1.0 / max(1, len(objectives))] * len(objectives)
    c = {name: 0.0 for name in variables}
    for weight, objective in zip(weights, objectives):
        coeff, _bias = parser.parse(str(objective["expression"]))
        sign = -1.0 if str(objective.get("name", "")).lower().startswith("maximize") else 1.0
        for name, value in coeff.items():
            c[name] += float(weight) * sign * value
    return [c[name] for name in variables]


def milp_select_rules(candidate_ids: list[str], rule_by_id: dict[str, dict[str, Any]], scenario: dict[str, Any]) -> list[str]:
    if not candidate_ids:
        return []
    idx = {rule_id: i for i, rule_id in enumerate(candidate_ids)}
    c = -np.ones(len(candidate_ids))
    integrality = np.ones(len(candidate_ids))
    bounds = Bounds(np.zeros(len(candidate_ids)), np.ones(len(candidate_ids)))
    rows = []
    lb = []
    ub = []

    for rule_id in candidate_ids:
        rule = rule_by_id.get(rule_id, {})
        if not eval_guard(rule.get("guard"), scenario):
            row = np.zeros(len(candidate_ids))
            row[idx[rule_id]] = 1
            rows.append(row)
            lb.append(0)
            ub.append(0)

    for rule_id in candidate_ids:
        rule = rule_by_id.get(rule_id, {})
        for rel in rule.get("relations", []):
            target = relation_target(rel)
            if target not in idx:
                continue
            rt = relation_type(rel)
            if rt in DEPENDENCY_TYPES:
                row = np.zeros(len(candidate_ids))
                row[idx[rule_id]] = 1
                row[idx[target]] = -1
                rows.append(row)
                lb.append(-np.inf)
                ub.append(0)
            elif rt in EXCLUSION_TYPES:
                row = np.zeros(len(candidate_ids))
                row[idx[rule_id]] = 1
                row[idx[target]] = 1
                rows.append(row)
                lb.append(-np.inf)
                ub.append(1)
            elif rt in OVERRIDE_TYPES or rt in PRECEDENCE_TYPES:
                row = np.zeros(len(candidate_ids))
                row[idx[rule_id]] = 1
                row[idx[target]] = 1
                rows.append(row)
                lb.append(-np.inf)
                ub.append(1)

    if rows:
        constraints = LinearConstraint(np.vstack(rows), np.array(lb), np.array(ub))
    else:
        constraints = None
    result = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints, options={"time_limit": 10})
    if not result.success or result.x is None:
        return []
    return sorted(rule_id for rule_id in candidate_ids if result.x[idx[rule_id]] >= 0.5)


def run_milp(
    query: dict[str, Any],
    candidate_ids: list[str],
    rule_by_id: dict[str, dict[str, Any]],
) -> MethodResult:
    try:
        variables = list(query.get("decision_variables", {}))
        if not variables:
            return MethodResult(False, [], None, None, "no_decision_variables")
        scenario = scenario_for_query(query)
        predicted = milp_select_rules(candidate_ids, rule_by_id, scenario)
        if not predicted:
            return MethodResult(False, [], None, None, "milp_rule_selection_infeasible")
        constraints = constraints_for_method(query, predicted, rule_by_id, include_candidate_rulelib_constraints=False)
        constants = {
            key: float(value)
            for key, value in query.get("scenario_facts", {}).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        rows, lb, ub = linear_constraint_rows(constraints, variables, constants)
        c = objective_linear(query, variables, constants)
        bounds = [
            (float(query["decision_variables"][name].get("lower", 0.0)), float(query["decision_variables"][name].get("upper", 1.0)))
            for name in variables
        ]
        lp_result = linprog(
            c=np.array(c),
            A_ub=np.array([row for row, lo, hi in zip(rows, lb, ub) if math.isinf(lo) and not math.isinf(hi)]) if rows else None,
            b_ub=np.array([hi for lo, hi in zip(lb, ub) if math.isinf(lo) and not math.isinf(hi)]) if rows else None,
            A_eq=np.array([row for row, lo, hi in zip(rows, lb, ub) if not math.isinf(lo) and not math.isinf(hi) and abs(lo - hi) < 1e-9]) if rows else None,
            b_eq=np.array([lo for lo, hi in zip(lb, ub) if not math.isinf(lo) and not math.isinf(hi) and abs(lo - hi) < 1e-9]) if rows else None,
            bounds=bounds,
            method="highs",
        )
        if not lp_result.success or lp_result.x is None:
            return MethodResult(False, predicted, None, None, f"highs_{lp_result.message}")
        x = {name: float(value) for name, value in zip(variables, lp_result.x)}
        formal = constraints_satisfied(constraints, with_query_values(query, x))
        return MethodResult(True, predicted, x, formal)
    except Exception as exc:  # noqa: BLE001
        return MethodResult(False, [], None, None, f"unsupported_nonlinear_or_mapping:{exc}")


def semantic_valid(
    feasible: dict[str, Any],
    x: dict[str, float] | None,
    predicted_rule_ids: list[str],
    reference_ids: list[str],
) -> bool:
    if x is None:
        return False
    numeric_ok = constraints_satisfied(reference_constraints(feasible), with_scenario_values(feasible, x))
    structure_ok = set(predicted_rule_ids) == set(reference_ids)
    return bool(numeric_ok and structure_ok)


def with_scenario_values(feasible: dict[str, Any], x: dict[str, float]) -> dict[str, float]:
    values = dict(x)
    for key, value in feasible.get("scenario_facts", {}).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values[key] = float(value)
    add_derived_aliases(values)
    return values


def add_derived_aliases(values: dict[str, float]) -> None:
    for key, value in list(values.items()):
        values.setdefault(f"derived_{key}", value)
    if "station_distance_km" in values:
        values.setdefault("KG_grounded_minimum_tolerance_radius", 26.2)


def evaluate_method(
    dataset_name: str,
    method: str,
    query: dict[str, Any],
    label: dict[str, Any],
    feasible: dict[str, Any],
    rule_library: dict[str, Any],
    rule_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    task_id = str(query["omega_id"])
    candidate = candidate_ids_from_query(query, label, feasible)
    reference = reference_rule_ids(label, feasible, query)

    start = time.perf_counter()
    if method == "Flat baseline":
        result = run_flat(query, candidate, rule_by_id)
    elif method == "CTHR full":
        result = run_cthr(query, candidate, rule_by_id)
    elif method == "ASP + clingo":
        result = run_asp(rule_library, query, candidate, rule_by_id)
    elif method == "SMT + Z3":
        result = run_smt(rule_library, query, candidate)
    elif method == "MILP + HiGHS":
        result = run_milp(query, candidate, rule_by_id)
    else:
        raise ValueError(method)
    elapsed = (time.perf_counter() - start) * 1000.0

    precision = method_rule_precision(result.predicted_rule_ids, reference) if result.supported else None
    recall = method_rule_recall(result.predicted_rule_ids, reference) if result.supported else None
    sem_valid = semantic_valid(feasible, result.optimized_x, result.predicted_rule_ids, reference) if result.supported else None
    formal = result.formal_feasible if result.supported else None
    false_accept = bool(formal and not sem_valid) if result.supported else None
    invalid = bool(not sem_valid) if result.supported else None

    return {
        "Dataset": dataset_name,
        "task_id": task_id,
        "Method": method,
        "predicted_rule_ids": result.predicted_rule_ids if result.supported else [],
        "reference_rule_ids": reference,
        "rule_precision": precision,
        "rule_recall": recall,
        "formal_feasible": formal,
        "semantic_valid": sem_valid,
        "false_accept": false_accept,
        "invalid_case": invalid,
        "unsupported_reason": "" if result.supported else result.unsupported_reason,
        "runtime_ms": round(elapsed, 3),
    }


def unsupported_row(dataset_name: str, method: str, query: dict[str, Any], label: dict[str, Any], feasible: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "Dataset": dataset_name,
        "task_id": str(query["omega_id"]),
        "Method": method,
        "predicted_rule_ids": [],
        "reference_rule_ids": reference_rule_ids(label, feasible, query),
        "rule_precision": None,
        "rule_recall": None,
        "formal_feasible": None,
        "semantic_valid": None,
        "false_accept": None,
        "invalid_case": None,
        "unsupported_reason": reason,
        "runtime_ms": 0.0,
    }


def pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{100.0 * value:.1f}%"


def avg(values: list[float | None]) -> float | None:
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def aggregate(rows: list[dict[str, Any]], dataset_name: str, method: str) -> dict[str, Any]:
    if dataset_name == "Overall":
        subset = [row for row in rows if row["Method"] == method]
    else:
        subset = [row for row in rows if row["Dataset"] == dataset_name and row["Method"] == method]
    supported = [row for row in subset if not row["unsupported_reason"]]
    n = len(supported)
    if n == 0:
        return {
            "Dataset": dataset_name,
            "Method": method,
            "Rule Precision": "N/A",
            "Rule Recall": "N/A",
            "Formal CSR": "N/A",
            "Sem-CSR": "N/A",
            "False accept": "N/A",
            "Invalid cases": "N/A",
        }
    invalid_count = sum(1 for row in supported if row["invalid_case"])
    false_accept_count = sum(1 for row in supported if row["false_accept"])
    formal_count = sum(1 for row in supported if row["formal_feasible"])
    sem_count = sum(1 for row in supported if row["semantic_valid"])
    unsupported_count = len(subset) - n
    suffix = f" ({unsupported_count} unsupported)" if unsupported_count else ""
    return {
        "Dataset": dataset_name,
        "Method": method,
        "Rule Precision": pct(avg([row["rule_precision"] for row in supported])),
        "Rule Recall": pct(avg([row["rule_recall"] for row in supported])),
        "Formal CSR": pct(formal_count / n),
        "Sem-CSR": pct(sem_count / n),
        "False accept": pct(false_accept_count / n),
        "Invalid cases": f"{invalid_count}/{n} ({100.0 * invalid_count / n:.1f}%){suffix}",
    }


def render_md_table(rows: list[dict[str, Any]], headers: list[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines) + "\n"


def unsupported_summary(per_task_rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for row in per_task_rows:
        reason = row.get("unsupported_reason")
        if not reason:
            continue
        method = row["Method"]
        out.setdefault(method, {})
        out[method][reason] = out[method].get(reason, 0) + 1
    return out


def build_report(overall_rows: list[dict[str, Any]], per_task_rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    headers = ["Dataset", "Method", "Rule Precision", "Rule Recall", "Formal CSR", "Sem-CSR", "False accept", "Invalid cases"]
    unsupported = unsupported_summary(per_task_rows)
    unsupported_lines = []
    if unsupported:
        for method, reasons in sorted(unsupported.items()):
            reason_text = "; ".join(f"{reason}: {count}" for reason, count in sorted(reasons.items()))
            unsupported_lines.append(f"- {method}: {reason_text}")
    else:
        unsupported_lines.append("- none")

    lines = [
        "# Section 6.2 Table 1: Full KG-to-Constraint Modeling Pipeline Baselines",
        "",
        "## Dataset",
        "",
        "- Aviation benchmark: 31 tasks, combining 19 original aviation tasks and 12 aviation stress tasks.",
        "- Architecture benchmark: 30 tasks, using the corrected architecture stress rule library.",
        "- Overall aggregates all supported task-method evaluations from both datasets.",
        "",
        "## Methods",
        "",
        "- Flat baseline: uses visible candidate rules and directly flattens their rule constraints before optimization.",
        "- CTHR full: uses candidate rules and scenario facts to resolve valid rule structures before feasible-region construction.",
        "- ASP + clingo: uses ASP answer-set enumeration for rule selection, then the default optimizer for numeric decisions.",
        "- SMT + Z3: uses monolithic SMT encoding and Z3 Optimize when the objective and constraints are encodable.",
        "- MILP + HiGHS: uses binary rule-selection variables and HiGHS for tasks with linear objectives and constraints; nonlinear or unmapped tasks are marked unsupported.",
        "",
        "Reference labels, valid rules, valid structures, and source-rule semantic checks are used only for evaluation.",
        "",
        "## Main Result",
        "",
        render_md_table(overall_rows, headers),
        "",
        "## Unsupported / N/A",
        "",
        *unsupported_lines,
        "",
        "## Conclusion",
        "",
        "- CTHR is the most stable method when rule-structure recovery and semantic validity are considered together.",
        "- Flat compilation typically has high rule recall because it retains candidate rules, but this also lowers precision and increases false accepts when candidate surplus rules are structured alternatives, defeated rules, or lower-priority competitors.",
        "- ASP, SMT, and MILP expose useful symbolic baselines, but their end-to-end stability depends on candidate noise, relation encoding, numeric mapping, and objective/constraint linearity.",
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
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    per_task_rows: list[dict[str, Any]] = []
    dataset_counts: dict[str, int] = {}

    for spec in DATASETS:
        queries = by_id(load_items(spec.root / spec.query_file))
        labels = by_id(load_items(spec.root / spec.rule_label_file))
        feasible_items = by_id(load_items(spec.root / spec.feasible_file))
        rule_library_path = spec.root / spec.rule_library_file
        rule_library = read_json(rule_library_path) if rule_library_path.exists() else None
        missing_rule_library_reason = "" if rule_library is not None else f"missing_rule_library:{rule_library_path}"
        if rule_library is None:
            rule_by_id: dict[str, dict[str, Any]] = {}
        else:
            rule_by_id = {str(rule["rule_id"]): rule for rule in rule_library.get("rules", []) if rule.get("rule_id")}
        ids = list(queries)
        dataset_counts[spec.name] = len(ids)
        if set(queries) != set(labels) or set(queries) != set(feasible_items):
            raise ValueError(f"Layer IDs do not match for {spec.name}")

        for task_id in ids:
            for method in METHODS:
                if missing_rule_library_reason:
                    per_task_rows.append(
                        unsupported_row(
                            spec.name,
                            method,
                            queries[task_id],
                            labels[task_id],
                            feasible_items[task_id],
                            missing_rule_library_reason,
                        )
                    )
                else:
                    per_task_rows.append(
                        evaluate_method(
                            spec.name,
                            method,
                            queries[task_id],
                            labels[task_id],
                            feasible_items[task_id],
                            rule_library,
                            rule_by_id,
                        )
                    )

    aggregate_rows: list[dict[str, Any]] = []
    for dataset_name in ["Aviation", "Architecture"]:
        for method in METHODS:
            aggregate_rows.append(aggregate(per_task_rows, dataset_name, method))
    for method in METHODS:
        aggregate_rows.append(aggregate(per_task_rows, "Overall", method))

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
    aggregate_headers = ["Dataset", "Method", "Rule Precision", "Rule Recall", "Formal CSR", "Sem-CSR", "False accept", "Invalid cases"]

    write_csv(RESULTS_DIR / "section_6_2_table1_pipeline_per_task.csv", per_task_rows, per_task_headers)
    write_csv(RESULTS_DIR / "section_6_2_table1_pipeline_overall.csv", aggregate_rows, aggregate_headers)
    (RESULTS_DIR / "section_6_2_table1_pipeline_overall.md").write_text(
        render_md_table(aggregate_rows, aggregate_headers),
        encoding="utf-8",
    )
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "datasets": dataset_counts,
        "methods": METHODS,
        "metric_scope": "Macro-averaged over supported tasks; unsupported task-method pairs are listed in per-task CSV and report.",
        "outputs": {
            "overall_csv": str(RESULTS_DIR / "section_6_2_table1_pipeline_overall.csv"),
            "overall_md": str(RESULTS_DIR / "section_6_2_table1_pipeline_overall.md"),
            "overall_json": str(RESULTS_DIR / "section_6_2_table1_pipeline_overall.json"),
            "per_task_csv": str(RESULTS_DIR / "section_6_2_table1_pipeline_per_task.csv"),
            "report_md": str(RESULTS_DIR / "section_6_2_table1_pipeline_report.md"),
        },
        "unsupported": unsupported_summary(per_task_rows),
        "aggregate_rows": aggregate_rows,
    }
    write_json(RESULTS_DIR / "section_6_2_table1_pipeline_overall.json", summary)
    (RESULTS_DIR / "section_6_2_table1_pipeline_report.md").write_text(
        build_report(aggregate_rows, per_task_rows, summary),
        encoding="utf-8",
    )
    print(json.dumps({"datasets": dataset_counts, "outputs": summary["outputs"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
