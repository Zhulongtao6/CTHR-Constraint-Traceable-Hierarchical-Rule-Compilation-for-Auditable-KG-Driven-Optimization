from __future__ import annotations

import ast
import math
import re
import time
from dataclasses import dataclass
from typing import Any

try:
    import z3  # type: ignore
except ImportError as exc:  # pragma: no cover - exercised when z3 is absent.
    z3 = None
    Z3_IMPORT_ERROR = exc
else:
    Z3_IMPORT_ERROR = None

from .asp_rule_structure import (
    eval_guard,
    normalize_field_name,
    relation_target,
    relation_type,
    retrieve_candidate_rules,
)


DEPENDENCY_TYPES = {"depends_on", "requires", "uses_parameter"}
EXCLUSION_TYPES = {"excludes", "mutually_exclusive", "conflicts_with", "conflict"}
OVERRIDE_TYPES = {"overrides", "can_override", "replaces", "defeats"}
PRECEDENCE_TYPES = {"precedes", "precedence", "higher_priority_than", "has_precedence_over"}
COMPARATOR_RE = re.compile(r"(<=|>=|!=|==|=|<|>)")


@dataclass(frozen=True)
class SmtFormula:
    task_id: str
    candidate_rule_ids: list[str]
    y: dict[str, Any]
    x: dict[str, Any]
    constraints: list[Any]
    selectable_exprs: list[Any]
    objective_expr: Any | None
    encoded_constraint_count: int
    skipped_constraint_count: int
    encoded_rule_library_constraint_count: int
    encoded_visible_constraint_count: int
    notes: list[str]


@dataclass(frozen=True)
class SmtCheckResult:
    status: str
    accepted: bool
    check_time_ms: float
    selected_rule_ids: list[str]
    encoded_constraint_count: int
    skipped_constraint_count: int
    error: str | None = None


@dataclass(frozen=True)
class SmtOptimizeResult:
    status: str
    optimized_x: list[float] | None
    objective_value: float | None
    selected_rule_ids: list[str]
    solve_time_ms: float
    encoded_constraint_count: int
    skipped_constraint_count: int
    mode: str
    error: str | None = None


def ensure_z3_available() -> None:
    if z3 is None:
        raise RuntimeError(
            "The SMT baseline requires the z3-solver Python package. "
            "Install it with `pip install z3-solver` or install the project "
            "dependencies from `requirements.txt`."
        ) from Z3_IMPORT_ERROR


def task_scenario(query: dict[str, Any]) -> dict[str, Any]:
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


def numeric_scenario(query: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for key, value in query.get("scenario_facts", {}).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values[key] = float(value)
    if "station_distance_km" in values:
        values["KG_grounded_minimum_tolerance_radius"] = 26.2
    return values


def normalize_expr(expr: str) -> str:
    expr = str(expr).strip()
    match = COMPARATOR_RE.search(expr)
    if match and match.group(1) == "=":
        expr = expr[: match.start()] + "==" + expr[match.end() :]
    return expr


def split_comparator(expression: str) -> tuple[str, str, str] | None:
    match = COMPARATOR_RE.search(expression)
    if not match:
        return None
    return expression[: match.start()].strip(), match.group(1), expression[match.end() :].strip()


def safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    number = re.search(r"[-+]?[0-9]+(?:\.[0-9]+)?", text)
    if not number:
        return None
    return float(number.group(0))


def variable_aliases(name: str) -> set[str]:
    norm = normalize_field_name(name)
    aliases = {norm}
    for suffix in ("s", "sec", "second", "seconds", "m", "km", "nm", "ft", "deg", "degree", "percent", "pct"):
        if norm.endswith(suffix) and len(norm) > len(suffix) + 2:
            aliases.add(norm[: -len(suffix)])
    return aliases


def map_rule_variable(variable: str, query: dict[str, Any]) -> str | None:
    target_aliases = variable_aliases(variable)
    best_name = None
    best_score = 0
    for name in query.get("decision_variables", {}):
        aliases = variable_aliases(name)
        score = len(target_aliases & aliases)
        if score == 0:
            var_norm = normalize_field_name(variable)
            name_norm = normalize_field_name(name)
            if var_norm in name_norm or name_norm in var_norm:
                score = 1
        if score > best_score:
            best_name = name
            best_score = score
    return best_name


class Z3ExprBuilder(ast.NodeVisitor):
    def __init__(self, z3_vars: dict[str, Any], constants: dict[str, float]):
        self.z3_vars = z3_vars
        self.constants = constants

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id in self.z3_vars:
            return self.z3_vars[node.id]
        if node.id in self.constants:
            return z3.RealVal(str(self.constants[node.id]))
        raise ValueError(f"Unknown symbol: {node.id}")

    def visit_Constant(self, node: ast.Constant) -> Any:
        if isinstance(node.value, (int, float)):
            return z3.RealVal(str(float(node.value)))
        raise ValueError(f"Unsupported constant: {node.value!r}")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        value = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
        raise ValueError("Unsupported unary operator")

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        raise ValueError("Unsupported binary operator")

    def visit_Call(self, node: ast.Call) -> Any:
        if not isinstance(node.func, ast.Name):
            raise ValueError("Unsupported function call")
        args = [self.visit(arg) for arg in node.args]
        if node.func.id == "abs" and len(args) == 1:
            return z3.If(args[0] >= 0, args[0], -args[0])
        if node.func.id == "min" and len(args) == 2:
            return z3.If(args[0] <= args[1], args[0], args[1])
        if node.func.id == "max" and len(args) == 2:
            return z3.If(args[0] >= args[1], args[0], args[1])
        raise ValueError(f"Unsupported function: {node.func.id}")

    def visit_Compare(self, node: ast.Compare) -> Any:
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise ValueError("Chained comparisons are unsupported")
        left = self.visit(node.left)
        right = self.visit(node.comparators[0])
        op = node.ops[0]
        if isinstance(op, ast.LtE):
            return left <= right
        if isinstance(op, ast.GtE):
            return left >= right
        if isinstance(op, ast.Lt):
            return left < right
        if isinstance(op, ast.Gt):
            return left > right
        if isinstance(op, ast.Eq):
            return left == right
        if isinstance(op, ast.NotEq):
            return left != right
        raise ValueError("Unsupported comparison")

    def generic_visit(self, node: ast.AST) -> Any:
        raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def parse_z3_expr(expression: str, z3_vars: dict[str, Any], constants: dict[str, float]) -> Any:
    ensure_z3_available()
    tree = ast.parse(normalize_expr(expression), mode="eval")
    return Z3ExprBuilder(z3_vars, constants).visit(tree.body)


def encode_rule_library_constraint(
    constraint: dict[str, Any],
    query: dict[str, Any],
    z3_vars: dict[str, Any],
) -> Any | None:
    variable = constraint.get("variable")
    mapped = map_rule_variable(str(variable), query) if variable else None
    if not mapped or mapped not in z3_vars:
        return None
    value = safe_float(constraint.get("value"))
    if value is None:
        return None
    op = str(constraint.get("op", "")).strip()
    var = z3_vars[mapped]
    rhs = z3.RealVal(str(value))
    if op == "<=":
        return var <= rhs
    if op == "<":
        return var < rhs
    if op == ">=":
        return var >= rhs
    if op == ">":
        return var > rhs
    if op in {"=", "=="}:
        return var == rhs
    return None


def build_smt_formula(
    rule_library: dict[str, Any],
    query: dict[str, Any],
    candidate_rule_ids: list[str] | None = None,
    include_visible_task_constraints: bool = True,
) -> SmtFormula:
    ensure_z3_available()
    task_id = str(query["omega_id"])
    candidate_rule_ids = candidate_rule_ids or retrieve_candidate_rules(rule_library, query)
    candidates = set(candidate_rule_ids)
    by_id = {str(rule["rule_id"]): rule for rule in rule_library.get("rules", []) if rule.get("rule_id")}
    candidate_rules = [by_id[rule_id] for rule_id in candidate_rule_ids if rule_id in by_id]
    scenario = task_scenario(query)
    constants = numeric_scenario(query)

    x = {name: z3.Real(name) for name in query.get("decision_variables", {})}
    y = {rule_id: z3.Bool(f"y__{safe_z3_name(rule_id)}") for rule_id in candidate_rule_ids}
    constraints: list[Any] = []
    notes: list[str] = []
    encoded_visible = 0
    encoded_rule = 0
    skipped = 0

    for name, spec in query.get("decision_variables", {}).items():
        constraints.append(x[name] >= z3.RealVal(str(float(spec["lower"]))))
        constraints.append(x[name] <= z3.RealVal(str(float(spec["upper"]))))
        if str(spec.get("type", "")).lower() == "binary":
            constraints.append(z3.Or(x[name] == 0, x[name] == 1))

    applicable: dict[str, bool] = {}
    for rule in candidate_rules:
        rule_id = str(rule["rule_id"])
        app = eval_guard(rule.get("guard"), scenario)
        applicable[rule_id] = bool(app)
        constraints.append(z3.Implies(y[rule_id], z3.BoolVal(app)))

    dependency_pairs: set[tuple[str, str]] = set()
    for rule in candidate_rules:
        rid = str(rule["rule_id"])
        for relation in rule.get("relations", []):
            target = relation_target(relation)
            if target not in candidates:
                continue
            rt = relation_type(relation)
            if rt in DEPENDENCY_TYPES:
                dependency_pairs.add((rid, target))

    for left, right in dependency_pairs:
        constraints.append(z3.Implies(y[left], y[right]))

    for rule in candidate_rules:
        rid = str(rule["rule_id"])
        for relation in rule.get("relations", []):
            target = relation_target(relation)
            if target not in candidates:
                continue
            rt = relation_type(relation)
            if rt in EXCLUSION_TYPES:
                if str(rule.get("rule_type", "")).lower() == "exception":
                    constraints.append(z3.Implies(y[rid], z3.Not(y[target])))
                elif (rid, target) not in dependency_pairs and (target, rid) not in dependency_pairs:
                    constraints.append(z3.Not(z3.And(y[rid], y[target])))
            elif rt in OVERRIDE_TYPES:
                constraints.append(z3.Implies(y[rid], z3.Not(y[target])))
            elif rt in PRECEDENCE_TYPES:
                constraints.append(z3.Implies(y[rid], z3.Not(y[target])))

    conflict_classes: dict[str, list[str]] = {}
    for rule in candidate_rules:
        cls = rule.get("conflict_class") or rule.get("conflict_group")
        if cls:
            conflict_classes.setdefault(str(cls), []).append(str(rule["rule_id"]))
    for members in conflict_classes.values():
        for i, left in enumerate(members):
            for right in members[i + 1 :]:
                constraints.append(z3.Not(z3.And(y[left], y[right])))

    for rule in candidate_rules:
        rid = str(rule["rule_id"])
        for rule_constraint in rule.get("constraints", []):
            encoded = encode_rule_library_constraint(rule_constraint, query, x)
            if encoded is None:
                skipped += 1
                continue
            constraints.append(z3.Implies(y[rid], encoded))
            encoded_rule += 1

    if include_visible_task_constraints:
        for constraint in query.get("solver_constraints", []):
            if not constraint.get("executable", False):
                continue
            expression = constraint.get("checker_expression") or constraint.get("expression")
            if not expression:
                continue
            try:
                encoded = parse_z3_expr(str(expression), x, constants)
            except Exception as exc:  # noqa: BLE001
                skipped += 1
                notes.append(f"skip_visible_constraint:{constraint.get('constraint_id')}:{exc}")
                continue
            source_id = str(constraint.get("source_id", ""))
            if constraint.get("source_type") == "rule_library" and source_id in y:
                constraints.append(z3.Implies(y[source_id], encoded))
            else:
                constraints.append(encoded)
            encoded_visible += 1

    selectable_exprs = [z3.If(y[rule_id], z3.IntVal(1), z3.IntVal(0)) for rule_id in candidate_rule_ids]
    objective_expr = None
    try:
        weights = query.get("query_preferences", {}).get("lambda", [])
        terms = []
        for weight, objective in zip(weights, query.get("objectives", [])):
            expr = parse_z3_expr(str(objective["expression"]), x, constants)
            if str(objective.get("name", "")).lower().startswith("maximize"):
                expr = -expr
            terms.append(z3.RealVal(str(float(weight))) * expr)
        if terms:
            objective_expr = sum(terms[1:], terms[0])
    except Exception as exc:  # noqa: BLE001
        notes.append(f"objective_not_encoded:{exc}")

    return SmtFormula(
        task_id=task_id,
        candidate_rule_ids=candidate_rule_ids,
        y=y,
        x=x,
        constraints=constraints,
        selectable_exprs=selectable_exprs,
        objective_expr=objective_expr,
        encoded_constraint_count=len(constraints),
        skipped_constraint_count=skipped,
        encoded_rule_library_constraint_count=encoded_rule,
        encoded_visible_constraint_count=encoded_visible,
        notes=notes,
    )


def safe_z3_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value)


def model_selected_rules(model: Any, formula: SmtFormula) -> list[str]:
    selected = []
    for rule_id, var in formula.y.items():
        value = model.eval(var, model_completion=True)
        if z3.is_true(value):
            selected.append(rule_id)
    return sorted(selected)


def z3_value_to_float(value: Any) -> float:
    text = str(value)
    if text.endswith("?"):
        text = text[:-1]
    if "/" in text:
        num, den = text.split("/", 1)
        return float(num) / float(den)
    return float(text)


def model_x(model: Any, formula: SmtFormula, query: dict[str, Any]) -> list[float]:
    values = []
    for name in query.get("decision_variables", {}):
        values.append(z3_value_to_float(model.eval(formula.x[name], model_completion=True)))
    return values


def check_membership(formula: SmtFormula, query: dict[str, Any], x_values: list[float], timeout_ms: int = 5000) -> SmtCheckResult:
    ensure_z3_available()
    start = time.perf_counter()
    try:
        opt = z3.Optimize()
        opt.set(timeout=timeout_ms)
        for constraint in formula.constraints:
            opt.add(constraint)
        for name, value in zip(query.get("decision_variables", {}).keys(), x_values):
            opt.add(formula.x[name] == z3.RealVal(str(float(value))))
        if formula.selectable_exprs:
            opt.maximize(sum(formula.selectable_exprs[1:], formula.selectable_exprs[0]))
        status = opt.check()
        elapsed = (time.perf_counter() - start) * 1000.0
        if status == z3.sat:
            model = opt.model()
            return SmtCheckResult(
                status="sat",
                accepted=True,
                check_time_ms=elapsed,
                selected_rule_ids=model_selected_rules(model, formula),
                encoded_constraint_count=formula.encoded_constraint_count,
                skipped_constraint_count=formula.skipped_constraint_count,
            )
        return SmtCheckResult(
            status=str(status),
            accepted=False,
            check_time_ms=elapsed,
            selected_rule_ids=[],
            encoded_constraint_count=formula.encoded_constraint_count,
            skipped_constraint_count=formula.skipped_constraint_count,
        )
    except Exception as exc:  # noqa: BLE001
        return SmtCheckResult(
            status="error",
            accepted=False,
            check_time_ms=(time.perf_counter() - start) * 1000.0,
            selected_rule_ids=[],
            encoded_constraint_count=formula.encoded_constraint_count,
            skipped_constraint_count=formula.skipped_constraint_count,
            error=str(exc),
        )


def optimize_with_z3(formula: SmtFormula, query: dict[str, Any], timeout_ms: int = 10000) -> SmtOptimizeResult:
    ensure_z3_available()
    start = time.perf_counter()
    try:
        if formula.objective_expr is None:
            return SmtOptimizeResult(
                status="not_applicable",
                optimized_x=None,
                objective_value=None,
                selected_rule_ids=[],
                solve_time_ms=0.0,
                encoded_constraint_count=formula.encoded_constraint_count,
                skipped_constraint_count=formula.skipped_constraint_count,
                mode="z3_optimize",
                error="objective_not_encoded",
            )
        opt = z3.Optimize()
        opt.set(timeout=timeout_ms)
        for constraint in formula.constraints:
            opt.add(constraint)
        if formula.selectable_exprs:
            opt.maximize(sum(formula.selectable_exprs[1:], formula.selectable_exprs[0]))
        opt.minimize(formula.objective_expr)
        status = opt.check()
        elapsed = (time.perf_counter() - start) * 1000.0
        if status != z3.sat:
            return SmtOptimizeResult(
                status=str(status),
                optimized_x=None,
                objective_value=None,
                selected_rule_ids=[],
                solve_time_ms=elapsed,
                encoded_constraint_count=formula.encoded_constraint_count,
                skipped_constraint_count=formula.skipped_constraint_count,
                mode="z3_optimize",
            )
        model = opt.model()
        x_values = model_x(model, formula, query)
        objective_value = z3_value_to_float(model.eval(formula.objective_expr, model_completion=True))
        return SmtOptimizeResult(
            status="sat",
            optimized_x=x_values,
            objective_value=objective_value,
            selected_rule_ids=model_selected_rules(model, formula),
            solve_time_ms=elapsed,
            encoded_constraint_count=formula.encoded_constraint_count,
            skipped_constraint_count=formula.skipped_constraint_count,
            mode="z3_optimize",
        )
    except Exception as exc:  # noqa: BLE001
        return SmtOptimizeResult(
            status="error",
            optimized_x=None,
            objective_value=None,
            selected_rule_ids=[],
            solve_time_ms=(time.perf_counter() - start) * 1000.0,
            encoded_constraint_count=formula.encoded_constraint_count,
            skipped_constraint_count=formula.skipped_constraint_count,
            mode="z3_optimize",
            error=str(exc),
        )
