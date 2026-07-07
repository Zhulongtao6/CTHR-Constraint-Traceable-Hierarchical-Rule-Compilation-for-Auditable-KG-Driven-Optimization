from __future__ import annotations

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

from .asp_rule_structure import eval_guard
from .cthr_grounded_smt import NumericMappingFailure, encode_rule_constraint_with_diagnostics
from .smt_monolithic import (
    numeric_scenario,
    parse_z3_expr,
    safe_z3_name,
    task_scenario,
    z3_value_to_float,
)


CONSTRAINT_MODES = {"rule_library_only", "with_visible_task_constraints", "visible_task_constraints_only"}


@dataclass(frozen=True)
class CthrValidStructuresSmtFormula:
    task_id: str
    constraint_mode: str
    valid_rule_ids: list[str]
    y: dict[str, Any]
    x: dict[str, Any]
    constraints: list[Any]
    objective_expr: Any | None
    guard_mismatch_rule_ids: list[str]
    encoded_rule_library_constraint_count: int
    encoded_visible_constraint_count: int
    mapping_failures: list[NumericMappingFailure]
    visible_constraint_failures: list[str]
    notes: list[str]


@dataclass(frozen=True)
class ValidStructuresSmtCheckResult:
    status: str
    accepted: bool
    check_time_ms: float
    selected_rule_ids: list[str]
    error: str | None = None


@dataclass(frozen=True)
class ValidStructuresSmtOptimizeResult:
    status: str
    optimized_x: list[float] | None
    objective_value: float | None
    selected_rule_ids: list[str]
    solve_time_ms: float
    mode: str
    error: str | None = None


def ensure_z3_available() -> None:
    if z3 is None:
        raise RuntimeError(
            "The CTHR-valid-structures SMT baseline requires the z3-solver Python package. "
            "Install it with `pip install z3-solver` or install the project dependencies."
        ) from Z3_IMPORT_ERROR


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


def encode_visible_constraint(
    constraint: dict[str, Any],
    query: dict[str, Any],
    z3_vars: dict[str, Any],
) -> tuple[Any | None, str | None]:
    expression = constraint.get("checker_expression") or constraint.get("expression")
    if not expression:
        return None, "missing_expression"
    try:
        return parse_z3_expr(str(expression), z3_vars, numeric_scenario(query)), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def visible_constraint_allowed(constraint: dict[str, Any], valid_rule_ids: set[str]) -> bool:
    source_type = constraint.get("source_type")
    source_id = str(constraint.get("source_id", ""))
    if source_type == "rule_library":
        return source_id in valid_rule_ids
    return source_type in {"task_or_scenario_model", "scenario_model", "scenario_obstacle_definition"} or source_id.startswith(
        "scenario_"
    )


def build_cthr_valid_structures_smt_formula(
    rule_library: dict[str, Any],
    query: dict[str, Any],
    valid_rule_ids: list[str],
    constraint_mode: str,
) -> CthrValidStructuresSmtFormula:
    ensure_z3_available()
    if constraint_mode not in CONSTRAINT_MODES:
        raise ValueError(f"Unknown constraint mode: {constraint_mode}")

    task_id = str(query["omega_id"])
    valid_rule_ids = sorted(dict.fromkeys(map(str, valid_rule_ids)))
    valid_set = set(valid_rule_ids)
    by_id = {str(rule["rule_id"]): rule for rule in rule_library.get("rules", []) if rule.get("rule_id")}
    valid_rules = [by_id[rule_id] for rule_id in valid_rule_ids if rule_id in by_id]
    scenario = task_scenario(query)

    x = {name: z3.Real(name) for name in query.get("decision_variables", {})}
    y = {rule_id: z3.Bool(f"y__{safe_z3_name(rule_id)}") for rule_id in valid_rule_ids}
    constraints: list[Any] = []
    mapping_failures: list[NumericMappingFailure] = []
    visible_failures: list[str] = []
    notes: list[str] = []
    encoded_rule = 0
    encoded_visible = 0

    for name, spec in query.get("decision_variables", {}).items():
        constraints.append(x[name] >= z3.RealVal(str(float(spec["lower"]))))
        constraints.append(x[name] <= z3.RealVal(str(float(spec["upper"]))))
        if str(spec.get("type", "")).lower() == "binary":
            constraints.append(z3.Or(x[name] == 0, x[name] == 1))

    # CTHR has already resolved valid structures, so selected valid rules are
    # fixed true. Guard mismatches are recorded as diagnostics rather than used
    # to reject the structure, because the point of this baseline is to replace
    # the downstream solver, not re-run grounding.
    for rule_id in valid_rule_ids:
        constraints.append(y[rule_id] == z3.BoolVal(True))

    guard_mismatch = []
    for rule in valid_rules:
        rid = str(rule["rule_id"])
        if not eval_guard(rule.get("guard"), scenario):
            guard_mismatch.append(rid)
        if constraint_mode == "visible_task_constraints_only":
            continue
        for rule_constraint in rule.get("constraints", []):
            encoded, failure = encode_rule_constraint_with_diagnostics(rid, rule_constraint, query, x)
            if failure is not None:
                mapping_failures.append(failure)
                continue
            constraints.append(z3.Implies(y[rid], encoded))
            encoded_rule += 1

    if constraint_mode in {"with_visible_task_constraints", "visible_task_constraints_only"}:
        for constraint in query.get("solver_constraints", []):
            if not constraint.get("executable", False):
                continue
            if not visible_constraint_allowed(constraint, valid_set):
                continue
            encoded, failure = encode_visible_constraint(constraint, query, x)
            if failure is not None:
                visible_failures.append(f"{constraint.get('constraint_id')}:{failure}")
                continue
            source_id = str(constraint.get("source_id", ""))
            if constraint.get("source_type") == "rule_library" and source_id in y:
                constraints.append(z3.Implies(y[source_id], encoded))
            else:
                constraints.append(encoded)
            encoded_visible += 1

    objective_expr, objective_error = build_objective_expr(query, x)
    if objective_error:
        notes.append(f"objective_not_encoded:{objective_error}")

    return CthrValidStructuresSmtFormula(
        task_id=task_id,
        constraint_mode=constraint_mode,
        valid_rule_ids=valid_rule_ids,
        y=y,
        x=x,
        constraints=constraints,
        objective_expr=objective_expr,
        guard_mismatch_rule_ids=sorted(guard_mismatch),
        encoded_rule_library_constraint_count=encoded_rule,
        encoded_visible_constraint_count=encoded_visible,
        mapping_failures=mapping_failures,
        visible_constraint_failures=visible_failures,
        notes=notes,
    )


def model_x(model: Any, formula: CthrValidStructuresSmtFormula, query: dict[str, Any]) -> list[float]:
    values = []
    for name in query.get("decision_variables", {}):
        values.append(z3_value_to_float(model.eval(formula.x[name], model_completion=True)))
    return values


def check_membership(
    formula: CthrValidStructuresSmtFormula,
    query: dict[str, Any],
    x_values: list[float],
    timeout_ms: int = 5000,
) -> ValidStructuresSmtCheckResult:
    ensure_z3_available()
    start = time.perf_counter()
    try:
        solver = z3.Solver()
        solver.set(timeout=timeout_ms)
        for constraint in formula.constraints:
            solver.add(constraint)
        for name, value in zip(query.get("decision_variables", {}).keys(), x_values):
            solver.add(formula.x[name] == z3.RealVal(str(float(value))))
        status = solver.check()
        elapsed = (time.perf_counter() - start) * 1000.0
        return ValidStructuresSmtCheckResult(
            status=str(status),
            accepted=status == z3.sat,
            check_time_ms=elapsed,
            selected_rule_ids=formula.valid_rule_ids if status == z3.sat else [],
        )
    except Exception as exc:  # noqa: BLE001
        return ValidStructuresSmtCheckResult(
            status="error",
            accepted=False,
            check_time_ms=(time.perf_counter() - start) * 1000.0,
            selected_rule_ids=[],
            error=str(exc),
        )


def optimize_with_z3(
    formula: CthrValidStructuresSmtFormula,
    query: dict[str, Any],
    timeout_ms: int = 10000,
) -> ValidStructuresSmtOptimizeResult:
    ensure_z3_available()
    start = time.perf_counter()
    try:
        if formula.objective_expr is None:
            return ValidStructuresSmtOptimizeResult(
                status="not_applicable",
                optimized_x=None,
                objective_value=None,
                selected_rule_ids=[],
                solve_time_ms=0.0,
                mode="z3_optimize",
                error="objective_not_encoded",
            )
        opt = z3.Optimize()
        opt.set(timeout=timeout_ms)
        for constraint in formula.constraints:
            opt.add(constraint)
        opt.minimize(formula.objective_expr)
        status = opt.check()
        elapsed = (time.perf_counter() - start) * 1000.0
        if status != z3.sat:
            return ValidStructuresSmtOptimizeResult(
                status=str(status),
                optimized_x=None,
                objective_value=None,
                selected_rule_ids=[],
                solve_time_ms=elapsed,
                mode="z3_optimize",
            )
        model = opt.model()
        return ValidStructuresSmtOptimizeResult(
            status="sat",
            optimized_x=model_x(model, formula, query),
            objective_value=z3_value_to_float(model.eval(formula.objective_expr, model_completion=True)),
            selected_rule_ids=formula.valid_rule_ids,
            solve_time_ms=elapsed,
            mode="z3_optimize",
        )
    except Exception as exc:  # noqa: BLE001
        return ValidStructuresSmtOptimizeResult(
            status="error",
            optimized_x=None,
            objective_value=None,
            selected_rule_ids=[],
            solve_time_ms=(time.perf_counter() - start) * 1000.0,
            mode="z3_optimize",
            error=str(exc),
        )
