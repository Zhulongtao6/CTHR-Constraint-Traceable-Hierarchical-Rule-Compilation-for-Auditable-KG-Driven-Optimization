from __future__ import annotations

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

from .smt_monolithic import numeric_scenario, parse_z3_expr, z3_value_to_float


CELL_SELECTION_MODES = {"at_least_one", "exactly_one"}
COMPARATOR_RE = re.compile(r"(<=|>=|!=|==|=|<|>)")
DEFAULT_FEASIBILITY_TOL = 1e-6


@dataclass(frozen=True)
class CompiledCell:
    cell_id: str
    variable_names: list[str]
    constraints: list[dict[str, Any]]
    rule_ids: list[str]
    provenance: list[dict[str, Any]]
    guard_satisfied: bool = True
    description: str | None = None


@dataclass(frozen=True)
class CellEncodingStats:
    le_constraints: int = 0
    ge_to_le_conversions: int = 0
    eq_to_two_le_conversions: int = 0
    strict_inequalities: int = 0
    constant_constraints: int = 0
    parse_failures: tuple[str, ...] = ()


@dataclass(frozen=True)
class CthrCellSmtFormula:
    task_id: str
    cell_selection_mode: str
    cells: list[CompiledCell]
    x: dict[str, Any]
    z: dict[str, Any]
    constraints: list[Any]
    objective_expr: Any | None
    objective_mapping_failure: str | None
    encoding_stats: CellEncodingStats


@dataclass(frozen=True)
class CellSmtCheckResult:
    status: str
    accepted: bool
    active_cell_ids: list[str]
    active_rule_ids: list[str]
    active_provenance: list[dict[str, Any]]
    check_time_ms: float
    error: str | None = None


@dataclass(frozen=True)
class CellSmtOptimizeResult:
    status: str
    optimized_x: list[float] | None
    objective_value: float | None
    active_cell_ids: list[str]
    active_rule_ids: list[str]
    active_provenance: list[dict[str, Any]]
    solve_time_ms: float
    mode: str
    error: str | None = None


def ensure_z3_available() -> None:
    if z3 is None:
        raise RuntimeError(
            "The CTHR-cell-SMT backend requires the z3-solver Python package. "
            "Install it with `pip install z3-solver` or install the project dependencies."
        ) from Z3_IMPORT_ERROR


def split_comparator(expression: str) -> tuple[str, str, str] | None:
    match = COMPARATOR_RE.search(str(expression))
    if not match:
        return None
    return expression[: match.start()].strip(), match.group(1), expression[match.end() :].strip()


def normalize_constraint_expression(expression: str) -> str:
    parsed = split_comparator(expression)
    if parsed is None:
        return expression
    lhs, op, rhs = parsed
    if op == "=":
        op = "=="
    return f"{lhs} {op} {rhs}"


def source_rule_ids_from_constraints(constraints: list[dict[str, Any]]) -> list[str]:
    ids = []
    for constraint in constraints:
        if constraint.get("source_type") == "rule_library" and constraint.get("source_id"):
            ids.append(str(constraint["source_id"]))
    return sorted(dict.fromkeys(ids))


def query_provenance(query: dict[str, Any]) -> list[dict[str, Any]]:
    provenance = query.get("certificate_targets", {}).get("provenance", [])
    return provenance if isinstance(provenance, list) else []


def query_rule_ids(query: dict[str, Any], constraints: list[dict[str, Any]]) -> list[str]:
    rule_ids = list(map(str, query.get("certificate_targets", {}).get("source_rule_ids", [])))
    rule_ids.extend(source_rule_ids_from_constraints(constraints))
    return sorted(dict.fromkeys(rule_ids))


def export_compiled_cells_from_query(query: dict[str, Any]) -> list[CompiledCell]:
    """Export the persisted CTHR compiled-cell interface from the query artifact.

    This uses `solver_constraints` and `solver_constraint_cells`, not hidden
    feasible labels or optimizer results.
    """

    variable_names = list(query.get("decision_variables", {}).keys())
    base_constraints = [c for c in query.get("solver_constraints", []) if c.get("executable", False)]
    provenance = query_provenance(query)
    raw_cells = query.get("solver_constraint_cells", [])
    if not raw_cells:
        return [
            CompiledCell(
                cell_id=f"{query['omega_id']}_default_cell",
                variable_names=variable_names,
                constraints=base_constraints,
                rule_ids=query_rule_ids(query, base_constraints),
                provenance=provenance,
                description="Default single compiled cell",
            )
        ]
    cells = []
    for raw in raw_cells:
        cell_constraints = [c for c in raw.get("executable_constraints", []) if c.get("executable", False)]
        all_constraints = base_constraints + cell_constraints
        cells.append(
            CompiledCell(
                cell_id=str(raw.get("cell_id", f"{query['omega_id']}_cell_{len(cells)}")),
                variable_names=variable_names,
                constraints=all_constraints,
                rule_ids=query_rule_ids(query, all_constraints),
                provenance=provenance,
                description=raw.get("description"),
            )
        )
    return cells


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


def encode_constraint(
    constraint: dict[str, Any],
    query: dict[str, Any],
    z3_vars: dict[str, Any],
    feasibility_tol: float = DEFAULT_FEASIBILITY_TOL,
) -> tuple[Any | None, str | None, dict[str, int]]:
    expression = constraint.get("checker_expression") or constraint.get("expression")
    if not expression:
        return None, "missing_expression", {}
    parsed = split_comparator(str(expression))
    conversion = {"le": 0, "ge": 0, "eq": 0, "strict": 0, "constant": 0}
    if parsed:
        _lhs, op, _rhs = parsed
        if op in {"<=", "<"}:
            conversion["le"] += 1
        elif op in {">=", ">"}:
            conversion["ge"] += 1
        elif op in {"=", "=="}:
            conversion["eq"] += 1
        if op in {"<", ">"}:
            conversion["strict"] += 1
    try:
        constants = numeric_scenario(query)
        if parsed:
            lhs, op, rhs = parsed
            lhs_expr = parse_z3_expr(lhs, z3_vars, constants)
            rhs_expr = parse_z3_expr(rhs, z3_vars, constants)
            tol = z3.RealVal(str(float(feasibility_tol)))
            if op == "<=":
                expr = lhs_expr <= rhs_expr + tol
            elif op == "<":
                expr = lhs_expr < rhs_expr + tol
            elif op == ">=":
                expr = lhs_expr >= rhs_expr - tol
            elif op == ">":
                expr = lhs_expr > rhs_expr - tol
            elif op in {"=", "=="}:
                expr = z3.And(lhs_expr <= rhs_expr + tol, lhs_expr >= rhs_expr - tol)
            elif op == "!=":
                expr = z3.Or(lhs_expr < rhs_expr - tol, lhs_expr > rhs_expr + tol)
            else:
                expr = parse_z3_expr(normalize_constraint_expression(str(expression)), z3_vars, constants)
        else:
            expr = parse_z3_expr(normalize_constraint_expression(str(expression)), z3_vars, constants)
        if not any(name in str(expression) for name in query.get("decision_variables", {})):
            conversion["constant"] += 1
        return expr, None, conversion
    except Exception as exc:  # noqa: BLE001
        return None, str(exc), conversion


def build_cthr_cell_smt_formula(
    query: dict[str, Any],
    cell_selection_mode: str = "at_least_one",
) -> CthrCellSmtFormula:
    ensure_z3_available()
    if cell_selection_mode not in CELL_SELECTION_MODES:
        raise ValueError(f"Unknown cell selection mode: {cell_selection_mode}")

    cells = export_compiled_cells_from_query(query)
    x = {name: z3.Real(name) for name in query.get("decision_variables", {})}
    z = {cell.cell_id: z3.Bool(f"z__{re.sub(r'[^A-Za-z0-9_]+', '_', cell.cell_id)}") for cell in cells}
    constraints: list[Any] = []
    parse_failures: list[str] = []
    le_count = ge_count = eq_count = strict_count = const_count = 0

    for name, spec in query.get("decision_variables", {}).items():
        constraints.append(x[name] >= z3.RealVal(str(float(spec["lower"]))))
        constraints.append(x[name] <= z3.RealVal(str(float(spec["upper"]))))
        if str(spec.get("type", "")).lower() == "binary":
            constraints.append(z3.Or(x[name] == 0, x[name] == 1))

    for cell in cells:
        if not cell.guard_satisfied:
            constraints.append(z[cell.cell_id] == z3.BoolVal(False))
            continue
        cell_exprs = []
        for constraint in cell.constraints:
            expr, failure, conversion = encode_constraint(constraint, query, x)
            le_count += conversion.get("le", 0)
            ge_count += conversion.get("ge", 0)
            eq_count += conversion.get("eq", 0)
            strict_count += conversion.get("strict", 0)
            const_count += conversion.get("constant", 0)
            if failure is not None:
                parse_failures.append(f"{cell.cell_id}:{constraint.get('constraint_id')}:{failure}")
                continue
            cell_exprs.append(expr)
        constraints.append(z3.Implies(z[cell.cell_id], z3.And(*cell_exprs) if cell_exprs else z3.BoolVal(True)))

    selectors = [z[cell.cell_id] for cell in cells]
    if selectors:
        if cell_selection_mode == "at_least_one":
            constraints.append(z3.Or(*selectors))
        else:
            constraints.append(sum([z3.If(selector, z3.IntVal(1), z3.IntVal(0)) for selector in selectors]) == 1)
    else:
        constraints.append(z3.BoolVal(False))

    objective_expr, objective_error = build_objective_expr(query, x)
    return CthrCellSmtFormula(
        task_id=str(query["omega_id"]),
        cell_selection_mode=cell_selection_mode,
        cells=cells,
        x=x,
        z=z,
        constraints=constraints,
        objective_expr=objective_expr,
        objective_mapping_failure=objective_error,
        encoding_stats=CellEncodingStats(
            le_constraints=le_count,
            ge_to_le_conversions=ge_count,
            eq_to_two_le_conversions=eq_count,
            strict_inequalities=strict_count,
            constant_constraints=const_count,
            parse_failures=tuple(parse_failures),
        ),
    )


def active_cell_ids_from_model(model: Any, formula: CthrCellSmtFormula) -> list[str]:
    active = []
    for cell_id, var in formula.z.items():
        if z3.is_true(model.eval(var, model_completion=True)):
            active.append(cell_id)
    return sorted(active)


def active_cells(formula: CthrCellSmtFormula, active_cell_ids: list[str]) -> list[CompiledCell]:
    active = set(active_cell_ids)
    return [cell for cell in formula.cells if cell.cell_id in active]


def active_rule_ids(formula: CthrCellSmtFormula, active_cell_ids: list[str]) -> list[str]:
    ids = []
    for cell in active_cells(formula, active_cell_ids):
        ids.extend(cell.rule_ids)
    return sorted(dict.fromkeys(ids))


def active_provenance(formula: CthrCellSmtFormula, active_cell_ids: list[str]) -> list[dict[str, Any]]:
    seen = set()
    provenance = []
    for cell in active_cells(formula, active_cell_ids):
        for item in cell.provenance:
            key = tuple(sorted(item.items()))
            if key not in seen:
                seen.add(key)
                provenance.append(item)
    return provenance


def model_x(model: Any, formula: CthrCellSmtFormula, query: dict[str, Any]) -> list[float]:
    values = []
    for name in query.get("decision_variables", {}):
        values.append(z3_value_to_float(model.eval(formula.x[name], model_completion=True)))
    return values


def check_membership(
    formula: CthrCellSmtFormula,
    query: dict[str, Any],
    x_values: list[float],
    timeout_ms: int = 5000,
) -> CellSmtCheckResult:
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
        if status == z3.sat:
            model = solver.model()
            cell_ids = active_cell_ids_from_model(model, formula)
            return CellSmtCheckResult(
                status="sat",
                accepted=True,
                active_cell_ids=cell_ids,
                active_rule_ids=active_rule_ids(formula, cell_ids),
                active_provenance=active_provenance(formula, cell_ids),
                check_time_ms=elapsed,
            )
        return CellSmtCheckResult(
            status=str(status),
            accepted=False,
            active_cell_ids=[],
            active_rule_ids=[],
            active_provenance=[],
            check_time_ms=elapsed,
        )
    except Exception as exc:  # noqa: BLE001
        return CellSmtCheckResult(
            status="error",
            accepted=False,
            active_cell_ids=[],
            active_rule_ids=[],
            active_provenance=[],
            check_time_ms=(time.perf_counter() - start) * 1000.0,
            error=str(exc),
        )


def optimize_with_z3(
    formula: CthrCellSmtFormula,
    query: dict[str, Any],
    timeout_ms: int = 10000,
) -> CellSmtOptimizeResult:
    ensure_z3_available()
    start = time.perf_counter()
    try:
        if formula.objective_expr is None:
            return CellSmtOptimizeResult(
                status="not_applicable",
                optimized_x=None,
                objective_value=None,
                active_cell_ids=[],
                active_rule_ids=[],
                active_provenance=[],
                solve_time_ms=0.0,
                mode="z3_optimize",
                error=f"objective_mapping_failure:{formula.objective_mapping_failure}",
            )
        opt = z3.Optimize()
        opt.set(timeout=timeout_ms)
        for constraint in formula.constraints:
            opt.add(constraint)
        opt.minimize(formula.objective_expr)
        status = opt.check()
        elapsed = (time.perf_counter() - start) * 1000.0
        if status != z3.sat:
            return CellSmtOptimizeResult(
                status=str(status),
                optimized_x=None,
                objective_value=None,
                active_cell_ids=[],
                active_rule_ids=[],
                active_provenance=[],
                solve_time_ms=elapsed,
                mode="z3_optimize",
            )
        model = opt.model()
        cell_ids = active_cell_ids_from_model(model, formula)
        return CellSmtOptimizeResult(
            status="sat",
            optimized_x=model_x(model, formula, query),
            objective_value=z3_value_to_float(model.eval(formula.objective_expr, model_completion=True)),
            active_cell_ids=cell_ids,
            active_rule_ids=active_rule_ids(formula, cell_ids),
            active_provenance=active_provenance(formula, cell_ids),
            solve_time_ms=elapsed,
            mode="z3_optimize",
        )
    except Exception as exc:  # noqa: BLE001
        return CellSmtOptimizeResult(
            status="error",
            optimized_x=None,
            objective_value=None,
            active_cell_ids=[],
            active_rule_ids=[],
            active_provenance=[],
            solve_time_ms=(time.perf_counter() - start) * 1000.0,
            mode="z3_optimize",
            error=str(exc),
        )
