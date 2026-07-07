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
    relation_target,
    relation_type,
    retrieve_candidate_rules,
)
from .smt_monolithic import (
    Z3ExprBuilder,
    map_rule_variable,
    numeric_scenario,
    parse_z3_expr,
    safe_float,
    safe_z3_name,
    task_scenario,
    z3_value_to_float,
)


DEPENDENCY_TYPES = {"depends_on", "requires", "uses_parameter"}
EXCLUSION_TYPES = {"excludes", "mutually_exclusive", "conflicts_with", "conflict"}
OVERRIDE_TYPES = {"overrides", "can_override", "replaces", "defeats"}
PRECEDENCE_TYPES = {"precedes", "precedence", "higher_priority_than", "has_precedence_over"}
COMPARATOR_RE = re.compile(r"(<=|>=|!=|==|=|<|>)")
SELECTION_MODES = {"maximize_selected", "required_applicable"}


@dataclass(frozen=True)
class NumericMappingFailure:
    rule_id: str
    variable: str | None
    op: str | None
    value: Any
    reason: str


@dataclass(frozen=True)
class CthrGroundedSmtFormula:
    task_id: str
    selection_mode: str
    candidate_rule_ids: list[str]
    applicable_rule_ids: list[str]
    y: dict[str, Any]
    defeated: dict[str, Any]
    x: dict[str, Any]
    constraints: list[Any]
    selectable_exprs: list[Any]
    objective_expr: Any | None
    defeated_rule_ids_static: list[str]
    dependency_pairs: list[tuple[str, str]]
    exclusion_pairs: list[tuple[str, str]]
    override_pairs: list[tuple[str, str]]
    precedence_pairs: list[tuple[str, str]]
    conflict_classes: dict[str, list[str]]
    encoded_rule_library_constraint_count: int
    mapping_failures: list[NumericMappingFailure]
    notes: list[str]


@dataclass(frozen=True)
class GroundedSmtCheckResult:
    status: str
    accepted: bool
    check_time_ms: float
    selected_rule_ids: list[str]
    defeated_rule_ids: list[str]
    error: str | None = None


@dataclass(frozen=True)
class GroundedSmtOptimizeResult:
    status: str
    optimized_x: list[float] | None
    objective_value: float | None
    selected_rule_ids: list[str]
    defeated_rule_ids: list[str]
    solve_time_ms: float
    mode: str
    error: str | None = None


def ensure_z3_available() -> None:
    if z3 is None:
        raise RuntimeError(
            "The CTHR-grounded SMT baseline requires the z3-solver Python package. "
            "Install it with `pip install z3-solver` or install the project dependencies."
        ) from Z3_IMPORT_ERROR


def cthr_safe_ground_candidate_rule_ids(
    rule_library: dict[str, Any],
    task: dict[str, Any],
    min_score: float = 2.0,
    closure_rounds: int = 3,
) -> list[str]:
    """Return pre-resolution candidates from visible task/rule metadata only.

    This deliberately does not call the earlier shared-grounding diagnostic
    interface because that interface can read near-final task artifacts.
    """

    return retrieve_candidate_rules(
        rule_library,
        task,
        min_score=min_score,
        closure_rounds=closure_rounds,
    )


def normalize_expr(expr: str) -> str:
    expr = str(expr).strip()
    match = COMPARATOR_RE.search(expr)
    if match and match.group(1) == "=":
        expr = expr[: match.start()] + "==" + expr[match.end() :]
    return expr


def encode_rule_constraint_with_diagnostics(
    rule_id: str,
    rule_constraint: dict[str, Any],
    query: dict[str, Any],
    z3_vars: dict[str, Any],
) -> tuple[Any | None, NumericMappingFailure | None]:
    ensure_z3_available()
    variable = rule_constraint.get("variable")
    if not variable:
        return None, NumericMappingFailure(
            rule_id=rule_id,
            variable=None,
            op=rule_constraint.get("op"),
            value=rule_constraint.get("value"),
            reason="missing_variable",
        )
    mapped = map_rule_variable(str(variable), query)
    if not mapped or mapped not in z3_vars:
        return None, NumericMappingFailure(
            rule_id=rule_id,
            variable=str(variable),
            op=rule_constraint.get("op"),
            value=rule_constraint.get("value"),
            reason="variable_not_mapped_to_task_decision_variable",
        )
    value = safe_float(rule_constraint.get("value"))
    if value is None:
        return None, NumericMappingFailure(
            rule_id=rule_id,
            variable=str(variable),
            op=rule_constraint.get("op"),
            value=rule_constraint.get("value"),
            reason="rhs_value_not_numeric",
        )
    op = str(rule_constraint.get("op", "")).strip()
    var = z3_vars[mapped]
    rhs = z3.RealVal(str(value))
    if op == "<=":
        return var <= rhs, None
    if op == "<":
        return var < rhs, None
    if op == ">=":
        return var >= rhs, None
    if op == ">":
        return var > rhs, None
    if op in {"=", "=="}:
        return var == rhs, None
    return None, NumericMappingFailure(
        rule_id=rule_id,
        variable=str(variable),
        op=op,
        value=rule_constraint.get("value"),
        reason="unsupported_operator",
    )


def build_objective_expr(query: dict[str, Any], z3_vars: dict[str, Any]) -> tuple[Any | None, str | None]:
    try:
        weights = query.get("query_preferences", {}).get("lambda", [])
        constants = numeric_scenario(query)
        terms = []
        for weight, objective in zip(weights, query.get("objectives", [])):
            expr = parse_z3_expr(str(objective["expression"]), z3_vars, constants)
            if str(objective.get("name", "")).lower().startswith("maximize"):
                expr = -expr
            terms.append(z3.RealVal(str(float(weight))) * expr)
        if not terms:
            return None, "no_objective_terms"
        return sum(terms[1:], terms[0]), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def build_cthr_grounded_smt_formula(
    rule_library: dict[str, Any],
    query: dict[str, Any],
    selection_mode: str,
    candidate_rule_ids: list[str] | None = None,
) -> CthrGroundedSmtFormula:
    ensure_z3_available()
    if selection_mode not in SELECTION_MODES:
        raise ValueError(f"Unknown selection mode: {selection_mode}")

    task_id = str(query["omega_id"])
    candidate_rule_ids = candidate_rule_ids or cthr_safe_ground_candidate_rule_ids(rule_library, query)
    candidates = set(candidate_rule_ids)
    by_id = {str(rule["rule_id"]): rule for rule in rule_library.get("rules", []) if rule.get("rule_id")}
    candidate_rules = [by_id[rule_id] for rule_id in candidate_rule_ids if rule_id in by_id]
    scenario = task_scenario(query)

    x = {name: z3.Real(name) for name in query.get("decision_variables", {})}
    y = {rule_id: z3.Bool(f"y__{safe_z3_name(rule_id)}") for rule_id in candidate_rule_ids}
    defeated = {rule_id: z3.Bool(f"defeated__{safe_z3_name(rule_id)}") for rule_id in candidate_rule_ids}

    constraints: list[Any] = []
    notes: list[str] = []
    mapping_failures: list[NumericMappingFailure] = []
    encoded_rule_constraints = 0

    for name, spec in query.get("decision_variables", {}).items():
        constraints.append(x[name] >= z3.RealVal(str(float(spec["lower"]))))
        constraints.append(x[name] <= z3.RealVal(str(float(spec["upper"]))))
        if str(spec.get("type", "")).lower() == "binary":
            constraints.append(z3.Or(x[name] == 0, x[name] == 1))

    applicable: dict[str, bool] = {}
    for rule in candidate_rules:
        rule_id = str(rule["rule_id"])
        app = bool(eval_guard(rule.get("guard"), scenario))
        applicable[rule_id] = app
        constraints.append(z3.Implies(y[rule_id], z3.BoolVal(app)))
        if not app:
            constraints.append(z3.Not(y[rule_id]))

    dependency_pairs: set[tuple[str, str]] = set()
    exclusion_pairs: set[tuple[str, str]] = set()
    override_pairs: set[tuple[str, str]] = set()
    precedence_pairs: set[tuple[str, str]] = set()

    for rule in candidate_rules:
        rid = str(rule["rule_id"])
        for relation in rule.get("relations", []):
            target = relation_target(relation)
            if target not in candidates:
                continue
            rt = relation_type(relation)
            if rt in DEPENDENCY_TYPES:
                dependency_pairs.add((rid, target))
            elif rt in EXCLUSION_TYPES:
                exclusion_pairs.add((rid, target))
            elif rt in OVERRIDE_TYPES:
                override_pairs.add((rid, target))
            elif rt in PRECEDENCE_TYPES:
                precedence_pairs.add((rid, target))

    for left, right in sorted(dependency_pairs):
        constraints.append(z3.Implies(y[left], y[right]))

    for left, right in sorted(exclusion_pairs):
        constraints.append(z3.Not(z3.And(y[left], y[right])))

    defeat_causes: dict[str, list[Any]] = {rule_id: [] for rule_id in candidate_rule_ids}
    for overrider, base in sorted(override_pairs):
        if base in defeated:
            # Override defeat is triggered by the overriding rule being applicable.
            defeat_causes[base].append(z3.BoolVal(bool(applicable.get(overrider, False))))
            constraints.append(z3.Implies(y[overrider], defeated[base]))
    for high, low in sorted(precedence_pairs):
        if low in defeated:
            defeat_causes[low].append(y[high])
            constraints.append(z3.Implies(y[high], defeated[low]))

    for rule_id in candidate_rule_ids:
        causes = defeat_causes.get(rule_id, [])
        if causes:
            constraints.append(defeated[rule_id] == z3.Or(*causes))
        else:
            constraints.append(defeated[rule_id] == z3.BoolVal(False))
        constraints.append(z3.Implies(defeated[rule_id], z3.Not(y[rule_id])))

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
            encoded, failure = encode_rule_constraint_with_diagnostics(rid, rule_constraint, query, x)
            if failure is not None:
                mapping_failures.append(failure)
                continue
            constraints.append(z3.Implies(y[rid], encoded))
            encoded_rule_constraints += 1

    if y:
        constraints.append(z3.Or(*[y[rule_id] for rule_id in candidate_rule_ids]))

    if selection_mode == "required_applicable":
        for rule_id in candidate_rule_ids:
            if applicable.get(rule_id, False):
                constraints.append(z3.Implies(z3.Not(defeated[rule_id]), y[rule_id]))

    objective_expr, objective_error = build_objective_expr(query, x)
    if objective_error:
        notes.append(f"objective_not_encoded:{objective_error}")

    selectable_exprs = [z3.If(y[rule_id], z3.IntVal(1), z3.IntVal(0)) for rule_id in candidate_rule_ids]
    defeated_static = sorted(
        rule_id
        for rule_id, causes in defeat_causes.items()
        if any(isinstance(cause, z3.BoolRef) and z3.is_true(cause) for cause in causes)
    )

    return CthrGroundedSmtFormula(
        task_id=task_id,
        selection_mode=selection_mode,
        candidate_rule_ids=candidate_rule_ids,
        applicable_rule_ids=sorted(rule_id for rule_id, app in applicable.items() if app),
        y=y,
        defeated=defeated,
        x=x,
        constraints=constraints,
        selectable_exprs=selectable_exprs,
        objective_expr=objective_expr,
        defeated_rule_ids_static=defeated_static,
        dependency_pairs=sorted(dependency_pairs),
        exclusion_pairs=sorted(exclusion_pairs),
        override_pairs=sorted(override_pairs),
        precedence_pairs=sorted(precedence_pairs),
        conflict_classes=conflict_classes,
        encoded_rule_library_constraint_count=encoded_rule_constraints,
        mapping_failures=mapping_failures,
        notes=notes,
    )


def model_selected_rules(model: Any, formula: CthrGroundedSmtFormula) -> list[str]:
    selected = []
    for rule_id, var in formula.y.items():
        if z3.is_true(model.eval(var, model_completion=True)):
            selected.append(rule_id)
    return sorted(selected)


def model_defeated_rules(model: Any, formula: CthrGroundedSmtFormula) -> list[str]:
    defeated_rule_ids = []
    for rule_id, var in formula.defeated.items():
        if z3.is_true(model.eval(var, model_completion=True)):
            defeated_rule_ids.append(rule_id)
    return sorted(defeated_rule_ids)


def model_x(model: Any, formula: CthrGroundedSmtFormula, query: dict[str, Any]) -> list[float]:
    values = []
    for name in query.get("decision_variables", {}):
        values.append(z3_value_to_float(model.eval(formula.x[name], model_completion=True)))
    return values


def add_selection_objective(opt: Any, formula: CthrGroundedSmtFormula) -> None:
    if formula.selection_mode == "maximize_selected" and formula.selectable_exprs:
        opt.maximize(sum(formula.selectable_exprs[1:], formula.selectable_exprs[0]))


def check_membership(
    formula: CthrGroundedSmtFormula,
    query: dict[str, Any],
    x_values: list[float],
    timeout_ms: int = 5000,
) -> GroundedSmtCheckResult:
    ensure_z3_available()
    start = time.perf_counter()
    try:
        opt = z3.Optimize()
        opt.set(timeout=timeout_ms)
        for constraint in formula.constraints:
            opt.add(constraint)
        for name, value in zip(query.get("decision_variables", {}).keys(), x_values):
            opt.add(formula.x[name] == z3.RealVal(str(float(value))))
        add_selection_objective(opt, formula)
        status = opt.check()
        elapsed = (time.perf_counter() - start) * 1000.0
        if status == z3.sat:
            model = opt.model()
            return GroundedSmtCheckResult(
                status="sat",
                accepted=True,
                check_time_ms=elapsed,
                selected_rule_ids=model_selected_rules(model, formula),
                defeated_rule_ids=model_defeated_rules(model, formula),
            )
        return GroundedSmtCheckResult(
            status=str(status),
            accepted=False,
            check_time_ms=elapsed,
            selected_rule_ids=[],
            defeated_rule_ids=[],
        )
    except Exception as exc:  # noqa: BLE001
        return GroundedSmtCheckResult(
            status="error",
            accepted=False,
            check_time_ms=(time.perf_counter() - start) * 1000.0,
            selected_rule_ids=[],
            defeated_rule_ids=[],
            error=str(exc),
        )


def optimize_with_z3(
    formula: CthrGroundedSmtFormula,
    query: dict[str, Any],
    timeout_ms: int = 10000,
) -> GroundedSmtOptimizeResult:
    ensure_z3_available()
    start = time.perf_counter()
    try:
        if formula.objective_expr is None:
            return GroundedSmtOptimizeResult(
                status="not_applicable",
                optimized_x=None,
                objective_value=None,
                selected_rule_ids=[],
                defeated_rule_ids=[],
                solve_time_ms=0.0,
                mode="z3_optimize",
                error="objective_not_encoded",
            )
        opt = z3.Optimize()
        opt.set(timeout=timeout_ms)
        for constraint in formula.constraints:
            opt.add(constraint)
        add_selection_objective(opt, formula)
        opt.minimize(formula.objective_expr)
        status = opt.check()
        elapsed = (time.perf_counter() - start) * 1000.0
        if status != z3.sat:
            return GroundedSmtOptimizeResult(
                status=str(status),
                optimized_x=None,
                objective_value=None,
                selected_rule_ids=[],
                defeated_rule_ids=[],
                solve_time_ms=elapsed,
                mode="z3_optimize",
            )
        model = opt.model()
        return GroundedSmtOptimizeResult(
            status="sat",
            optimized_x=model_x(model, formula, query),
            objective_value=z3_value_to_float(model.eval(formula.objective_expr, model_completion=True)),
            selected_rule_ids=model_selected_rules(model, formula),
            defeated_rule_ids=model_defeated_rules(model, formula),
            solve_time_ms=elapsed,
            mode="z3_optimize",
        )
    except Exception as exc:  # noqa: BLE001
        return GroundedSmtOptimizeResult(
            status="error",
            optimized_x=None,
            objective_value=None,
            selected_rule_ids=[],
            defeated_rule_ids=[],
            solve_time_ms=(time.perf_counter() - start) * 1000.0,
            mode="z3_optimize",
            error=str(exc),
        )
