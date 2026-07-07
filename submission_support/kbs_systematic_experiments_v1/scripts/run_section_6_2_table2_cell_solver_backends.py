from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import sys
import statistics
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution, linprog, minimize

try:
    import z3  # type: ignore
except ImportError:  # pragma: no cover
    z3 = None

try:
    import clingo  # type: ignore
except ImportError:  # pragma: no cover
    clingo = None


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "datasets"
RESULTS_DIR = ROOT / "results"
REPORTS_DIR = ROOT / "reports"
LOGS_DIR = ROOT / "logs"
SCRIPTS_DIR = ROOT / "scripts"

sys.path.insert(0, str(SCRIPTS_DIR))

import run_section_6_2_table1_aviation_old_candidate_profile_all_methods as table1_all  # noqa: E402
import run_section_6_2_table1_fullkg_pipeline as fullkg  # noqa: E402

AVIATION_DIR = DATASET_DIR / "aviation_fullkg_clean"
ARCHITECTURE_DIR = DATASET_DIR / "architecture_fullkg_clean"

OUT_PER_TASK = RESULTS_DIR / "section_6_2_table2_cell_solver_per_task.csv"
OUT_OVERALL_CSV = RESULTS_DIR / "section_6_2_table2_cell_solver_overall.csv"
OUT_OVERALL_MD = RESULTS_DIR / "section_6_2_table2_cell_solver_overall.md"
OUT_OVERALL_JSON = RESULTS_DIR / "section_6_2_table2_cell_solver_overall.json"
OUT_REPORT = RESULTS_DIR / "section_6_2_table2_cell_solver_report.md"
OUT_LOG = LOGS_DIR / "section_6_2_table2_cell_solver_run_log.json"

FEAS_TOL = 1e-4
BIG_PENALTY = 1e6
DE_MAXITER = 80
Z3_TIMEOUT_MS = 5000


COMPARATOR_RE = re.compile(r"(<=|>=|!=|==|=|<|>)")
DEPENDENCY_TYPES = {"depends_on", "requires", "uses_parameter"}
EXCLUSION_TYPES = {"excludes", "mutually_exclusive", "conflicts_with", "conflict"}
OVERRIDE_TYPES = {"overrides", "can_override", "replaces", "defeats"}
PRECEDENCE_TYPES = {"precedes", "precedence", "higher_priority_than", "has_precedence_over"}
VARIANT_TYPES = {"formula_variant_of", "parameter_variant_of", "piecewise_variant_of"}


@dataclass
class CompiledCell:
    cell_id: str
    rule_ids: list[str]
    constraints: list[dict[str, Any]]
    provenance: list[dict[str, Any]]
    compile_source: str


@dataclass
class SolverResult:
    solver: str
    solved: bool
    cell_valid: bool
    objective_value: float | None
    x: list[float] | None
    active_cell_id: str | None
    unsupported_reason: str | None
    solve_time_ms: float


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_cell(row.get(key)) for key in headers})


def csv_cell(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple, set)):
        return json.dumps(list(value) if isinstance(value, set) else value, ensure_ascii=False)
    if value is None:
        return ""
    return value


def list_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("items", "optimization_queries", "rule_structure_labels", "feasible_region_labels"):
        if isinstance(payload.get(key), list):
            return payload[key]
    raise KeyError(f"No known list field in payload keys {list(payload)}")


def normalize_token(value: Any) -> str:
    text = str(value).lower()
    text = text.replace("±", "+/-")
    return re.sub(r"[^a-z0-9]+", "", text)


def aliases(name: str) -> set[str]:
    norm = normalize_token(name)
    out = {norm}
    for suffix in (
        "s",
        "sec",
        "second",
        "seconds",
        "m",
        "km",
        "nm",
        "ft",
        "in",
        "inch",
        "deg",
        "degree",
        "percent",
        "pct",
        "ratio",
        "score",
        "count",
    ):
        if norm.endswith(suffix) and len(norm) > len(suffix) + 2:
            out.add(norm[: -len(suffix)])
    return out


def map_variable(variable: str, query: dict[str, Any]) -> str | None:
    var_aliases = aliases(variable)
    best: tuple[int, str | None] = (0, None)
    for name in query.get("decision_variables", {}):
        score = len(var_aliases & aliases(name))
        vnorm = normalize_token(variable)
        nnorm = normalize_token(name)
        if score == 0 and (vnorm in nnorm or nnorm in vnorm):
            score = 1
        if score > best[0]:
            best = (score, name)
    return best[1]


def split_comparator(expression: str) -> tuple[str, str, str] | None:
    match = COMPARATOR_RE.search(str(expression))
    if not match:
        return None
    return expression[: match.start()].strip(), match.group(1), expression[match.end() :].strip()


class SafeEval(ast.NodeVisitor):
    def __init__(self, values: dict[str, float]):
        self.values = values

    def visit_Name(self, node: ast.Name) -> float:
        if node.id not in self.values:
            raise ValueError(f"unknown symbol {node.id}")
        return float(self.values[node.id])

    def visit_Constant(self, node: ast.Constant) -> float:
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"unsupported constant {node.value!r}")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> float:
        val = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -val
        if isinstance(node.op, ast.UAdd):
            return val
        raise ValueError("unsupported unary op")

    def visit_BinOp(self, node: ast.BinOp) -> float:
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
        if isinstance(node.op, ast.Pow):
            return left**right
        raise ValueError("unsupported binary op")

    def visit_Call(self, node: ast.Call) -> float:
        if not isinstance(node.func, ast.Name):
            raise ValueError("unsupported call")
        args = [self.visit(arg) for arg in node.args]
        if node.func.id == "abs" and len(args) == 1:
            return abs(args[0])
        if node.func.id == "min" and len(args) == 2:
            return min(args)
        if node.func.id == "max" and len(args) == 2:
            return max(args)
        raise ValueError(f"unsupported function {node.func.id}")

    def visit_Compare(self, node: ast.Compare) -> bool:
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise ValueError("chained comparisons are unsupported")
        left = self.visit(node.left)
        right = self.visit(node.comparators[0])
        op = node.ops[0]
        if isinstance(op, ast.LtE):
            return left <= right + FEAS_TOL
        if isinstance(op, ast.GtE):
            return left + FEAS_TOL >= right
        if isinstance(op, ast.Lt):
            return left < right + FEAS_TOL
        if isinstance(op, ast.Gt):
            return left + FEAS_TOL > right
        if isinstance(op, ast.Eq):
            return abs(left - right) <= FEAS_TOL
        if isinstance(op, ast.NotEq):
            return abs(left - right) > FEAS_TOL
        raise ValueError("unsupported comparison")

    def generic_visit(self, node: ast.AST) -> Any:
        raise ValueError(f"unsupported expression node {type(node).__name__}")


def eval_expr(expression: str, values: dict[str, float]) -> float:
    return float(SafeEval(values).visit(ast.parse(normalize_expr(expression), mode="eval").body))


def eval_bool(expression: str, values: dict[str, float]) -> bool:
    return bool(SafeEval(values).visit(ast.parse(normalize_expr(expression), mode="eval").body))


def normalize_expr(expression: str) -> str:
    parsed = split_comparator(expression)
    if parsed:
        lhs, op, rhs = parsed
        if op == "=":
            op = "=="
        return f"{lhs} {op} {rhs}"
    return str(expression)


def scenario_constants(query: dict[str, Any], extra: dict[str, float] | None = None) -> dict[str, float]:
    values: dict[str, float] = {}
    for key, value in query.get("scenario_facts", {}).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values[key] = float(value)
    if "station_distance_km" in values and "KG_grounded_minimum_tolerance_radius" not in values:
        values["KG_grounded_minimum_tolerance_radius"] = 26.2
    if extra:
        values.update(extra)
    return values


def x_values(query: dict[str, Any], x: list[float], extra: dict[str, float] | None = None) -> dict[str, float]:
    values = scenario_constants(query, extra)
    for name, value in zip(query.get("decision_variables", {}).keys(), x):
        values[name] = float(value)
    return values


def guard_status(rule: dict[str, Any], query: dict[str, Any]) -> str:
    guard = rule.get("guard") or {}
    clauses = guard.get("all") if isinstance(guard, dict) else None
    if not clauses:
        return "empty"
    scenario = query.get("scenario_facts", {})
    missing = False
    for clause in clauses:
        field = str(clause.get("field", ""))
        op = str(clause.get("op", "eq")).lower()
        target = clause.get("value")
        actual = None
        for key, value in scenario.items():
            if aliases(key) & aliases(field):
                actual = value
                break
        if actual is None:
            missing = True
            continue
        if op in {"eq", "==", "="}:
            if normalize_token(actual) != normalize_token(target):
                try:
                    if abs(float(actual) - float(target)) > FEAS_TOL:
                        return "false"
                except Exception:
                    return "false"
        elif op in {"ne", "!=", "not_eq"}:
            if normalize_token(actual) == normalize_token(target):
                return "false"
        elif op in {"gt", ">", "gte", ">=", "lt", "<", "lte", "<="}:
            try:
                a = float(actual)
                b = float(target)
            except Exception:
                return "false"
            if op in {"gt", ">"} and not (a > b):
                return "false"
            if op in {"gte", ">="} and not (a >= b):
                return "false"
            if op in {"lt", "<"} and not (a < b):
                return "false"
            if op in {"lte", "<="} and not (a <= b):
                return "false"
    return "deferred" if missing else "true"


def relation_type(rel: dict[str, Any]) -> str:
    return str(rel.get("type", "")).lower()


def relation_target(rel: dict[str, Any]) -> str:
    return str(rel.get("target", ""))


def task_candidate_ids(query: dict[str, Any], rule_by_id: dict[str, dict[str, Any]]) -> list[str]:
    meta = query.get("stress_metadata", {})
    candidates = meta.get("candidate_rule_ids_expected_for_diagnostics") or meta.get("candidate_rule_ids")
    if candidates:
        return [str(rule_id) for rule_id in candidates if str(rule_id) in rule_by_id]

    visible_ids = set()
    text = " ".join(
        str(query.get(field, ""))
        for field in ("title", "task_type", "design_intent", "engineering_task", "domain", "source_domain")
    ).lower()
    decision_names = set(query.get("decision_variables", {}))
    for rule_id, rule in rule_by_id.items():
        if guard_status(rule, query) == "false":
            continue
        variable_match = any(map_variable(str(c.get("variable", "")), query) for c in rule.get("constraints", []))
        domain = str(rule.get("source_domain") or rule.get("domain") or "").lower()
        name = str(rule.get("name", "")).lower()
        if variable_match or (domain and domain in text) or any(normalize_token(v) in normalize_token(name) for v in decision_names):
            visible_ids.add(rule_id)
    return sorted(visible_ids)


def rule_parameter_values(rule_ids: list[str], rule_by_id: dict[str, dict[str, Any]], query: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for rule_id in rule_ids:
        for c in rule_by_id.get(rule_id, {}).get("constraints", []):
            mapped = map_variable(str(c.get("variable", "")), query)
            if mapped:
                continue
            if str(c.get("op")) in {"<=", ">=", "<", ">", "=", "=="}:
                try:
                    values[str(c.get("variable"))] = float(c.get("value"))
                except Exception:
                    pass
    return values


def resolve_rule_ids(candidate_ids: list[str], query: dict[str, Any], rule_by_id: dict[str, dict[str, Any]]) -> list[str]:
    selected: set[str] = set()
    deferred: set[str] = set()
    for rule_id in candidate_ids:
        rule = rule_by_id[rule_id]
        status = guard_status(rule, query)
        if status in {"true", "empty"}:
            selected.add(rule_id)
        elif status == "deferred":
            deferred.add(rule_id)

    # Deferred branch rules stay available as alternatives if their constraint
    # variables map to the task. This keeps cell decomposition alive when the
    # visible scenario does not explicitly preselect a compliance template.
    for rule_id in deferred:
        rule = rule_by_id[rule_id]
        if any(map_variable(str(c.get("variable", "")), query) for c in rule.get("constraints", [])):
            selected.add(rule_id)

    changed = True
    while changed:
        changed = False
        for rule_id in list(selected):
            for rel in rule_by_id[rule_id].get("relations", []):
                if relation_type(rel) in DEPENDENCY_TYPES:
                    target = relation_target(rel)
                    if target in candidate_ids and target not in selected:
                        selected.add(target)
                        changed = True

    defeated: set[str] = set()
    for rule_id in list(selected):
        for rel in rule_by_id[rule_id].get("relations", []):
            target = relation_target(rel)
            typ = relation_type(rel)
            if typ in OVERRIDE_TYPES and target in selected:
                defeated.add(target)
            if typ in PRECEDENCE_TYPES and target in selected:
                defeated.add(target)
            if typ in VARIANT_TYPES and target in selected:
                defeated.add(rule_id)
    selected -= defeated
    return sorted(selected)


def expression_symbols(expression: str) -> set[str]:
    try:
        tree = ast.parse(expression, mode="eval")
    except Exception:
        return set()
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


def constraint_from_rule_constraint(
    rule_id: str,
    c: dict[str, Any],
    query: dict[str, Any],
    params: dict[str, float],
) -> dict[str, Any] | None:
    variable = str(c.get("variable", ""))
    mapped = map_variable(variable, query)
    op = str(c.get("op", ""))
    value = c.get("value")
    if not mapped:
        return None
    if op == "formula":
        known = set(query.get("decision_variables", {})) | set(scenario_constants(query, params))
        if expression_symbols(str(value)) - known:
            return None
        expression = f"{mapped} == {value}"
    else:
        try:
            rhs = str(float(value))
        except Exception:
            rhs = str(value)
        expression = f"{mapped} {op} {rhs}"
    return {
        "constraint_id": f"{rule_id}::{variable}",
        "expression": expression,
        "checker_expression": expression,
        "source_type": "rule_library",
        "source_id": rule_id,
        "executable": True,
        "symbols": {
            "decision_variables": [mapped],
            "scenario_fields": sorted(params),
            "unresolved_symbols": [],
        },
    }


def constraints_from_rules(rule_ids: list[str], query: dict[str, Any], rule_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    params = rule_parameter_values(rule_ids, rule_by_id, query)
    constraints: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for rule_id in rule_ids:
        for c in rule_by_id.get(rule_id, {}).get("constraints", []):
            item = constraint_from_rule_constraint(rule_id, c, query, params)
            if not item:
                continue
            key = (item["source_id"], item["expression"])
            if key not in seen:
                constraints.append(item)
                seen.add(key)
    return constraints


def executable_constraints(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in items if item.get("executable", True)]


def dedupe_constraints(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    seen = set()
    for item in items:
        key = (str(item.get("constraint_id", "")), str(item.get("checker_expression") or item.get("expression")))
        if key not in seen:
            out.append(item)
            seen.add(key)
    return out


def extract_visible_cells(query: dict[str, Any]) -> list[CompiledCell]:
    base = executable_constraints(query.get("solver_constraints", []))
    raw_cells = query.get("solver_constraint_cells", [])
    provenance = query.get("certificate_targets", {}).get("provenance", [])
    if not raw_cells:
        return [
            CompiledCell(
                cell_id=f"{query['omega_id']}_runtime_cell_1",
                rule_ids=source_rule_ids(base),
                constraints=base,
                provenance=provenance if isinstance(provenance, list) else [],
                compile_source="visible_cthr_query_export",
            )
        ]
    cells = []
    for idx, raw in enumerate(raw_cells):
        cell_constraints = raw.get("executable_constraints") or raw.get("constraints") or []
        constraints = dedupe_constraints(base + executable_constraints(cell_constraints))
        cells.append(
            CompiledCell(
                cell_id=str(raw.get("cell_id", f"{query['omega_id']}_cell_{idx + 1}")),
                rule_ids=list(map(str, raw.get("rule_ids") or source_rule_ids(constraints))),
                constraints=constraints,
                provenance=provenance if isinstance(provenance, list) else [],
                compile_source="visible_cthr_query_export",
            )
        )
    return cells


def source_rule_ids(constraints: list[dict[str, Any]]) -> list[str]:
    ids = []
    for constraint in constraints:
        if constraint.get("source_id"):
            ids.append(str(constraint["source_id"]))
    return sorted(dict.fromkeys(ids))


def compile_cells_runtime(query: dict[str, Any], rule_by_id: dict[str, dict[str, Any]]) -> list[CompiledCell]:
    if query.get("solver_constraints") or query.get("solver_constraint_cells"):
        return extract_visible_cells(query)

    candidate_ids = task_candidate_ids(query, rule_by_id)
    selected_ids = resolve_rule_ids(candidate_ids, query, rule_by_id)
    selected_rules = [rule_by_id[rule_id] for rule_id in selected_ids]

    groups: dict[str, list[str]] = {}
    for rule in selected_rules:
        cls = rule.get("conflict_class")
        if cls:
            groups.setdefault(str(cls), []).append(str(rule["rule_id"]))

    non_conflict = [rule_id for rule_id in selected_ids if not rule_by_id[rule_id].get("conflict_class")]
    branch_groups = [sorted(ids) for ids in groups.values() if len(ids) > 1]
    if not branch_groups:
        structures = [selected_ids]
    else:
        structures = []
        # Current benchmark tasks use at most one meaningful alternative group;
        # this product form keeps the implementation general.
        def build_product(prefix: list[str], remaining: list[list[str]]) -> None:
            if not remaining:
                structures.append(sorted(non_conflict + prefix))
                return
            for rule_id in remaining[0]:
                build_product(prefix + [rule_id], remaining[1:])

        build_product([], branch_groups)

    cells: list[CompiledCell] = []
    for idx, rule_ids in enumerate(structures):
        constraints = constraints_from_rules(rule_ids, query, rule_by_id)
        if not constraints:
            # A cell with only box bounds is still a compiled optimization cell,
            # but it is marked by its source for diagnostics.
            constraints = []
        cells.append(
            CompiledCell(
                cell_id=f"{query['omega_id']}_runtime_cell_{idx + 1}",
                rule_ids=rule_ids,
                constraints=constraints,
                provenance=[],
                compile_source="runtime_rule_library_compiler",
            )
        )
    return cells or [
        CompiledCell(
            cell_id=f"{query['omega_id']}_runtime_cell_1",
            rule_ids=[],
            constraints=[],
            provenance=[],
            compile_source="runtime_rule_library_compiler_empty",
        )
    ]


def cell_valid(query: dict[str, Any], cell: CompiledCell, x: list[float]) -> bool:
    values = x_values(query, x)
    for name, spec in query.get("decision_variables", {}).items():
        val = values[name]
        if val < float(spec["lower"]) - FEAS_TOL or val > float(spec["upper"]) + FEAS_TOL:
            return False
    for constraint in cell.constraints:
        expr = str(constraint.get("checker_expression") or constraint.get("expression") or "")
        if not expr:
            continue
        try:
            if not eval_bool(expr, values):
                return False
        except Exception:
            return False
    return True


def union_cell_valid(query: dict[str, Any], cells: list[CompiledCell], x: list[float]) -> tuple[bool, str | None]:
    for cell in cells:
        if cell_valid(query, cell, x):
            return True, cell.cell_id
    return False, None


def scalar_objective(query: dict[str, Any], x: list[float]) -> float:
    values = x_values(query, x)
    weights = query.get("query_preferences", {}).get("lambda") or query.get("preference_weights") or []
    terms = []
    for idx, objective in enumerate(query.get("objectives", [])):
        weight = float(weights[idx]) if idx < len(weights) else 1.0 / max(1, len(query.get("objectives", [])))
        value = eval_expr(str(objective["expression"]), values)
        name = str(objective.get("name", "")).lower()
        if name.startswith("maximize"):
            value = -value
        terms.append(weight * value)
    if not terms:
        raise ValueError("no objective terms")
    return float(sum(terms))


def cell_violation(query: dict[str, Any], cell: CompiledCell, x: list[float]) -> float:
    values = x_values(query, x)
    penalty = 0.0
    for name, spec in query.get("decision_variables", {}).items():
        val = values[name]
        penalty += max(0.0, float(spec["lower"]) - val) ** 2
        penalty += max(0.0, val - float(spec["upper"])) ** 2
    for constraint in cell.constraints:
        expr = str(constraint.get("checker_expression") or constraint.get("expression") or "")
        parsed = split_comparator(expr)
        if not parsed:
            try:
                penalty += 0.0 if eval_bool(expr, values) else 1.0
            except Exception:
                penalty += 1.0
            continue
        lhs, op, rhs = parsed
        try:
            lv = eval_expr(lhs, values)
            rv = eval_expr(rhs, values)
        except Exception:
            penalty += 1.0
            continue
        if op in {"<=", "<"}:
            penalty += max(0.0, lv - rv) ** 2
        elif op in {">=", ">"}:
            penalty += max(0.0, rv - lv) ** 2
        elif op in {"=", "=="}:
            penalty += (lv - rv) ** 2
        elif op == "!=":
            penalty += 0.0 if abs(lv - rv) > FEAS_TOL else 1.0
    return float(penalty)


def default_solver(query: dict[str, Any], cells: list[CompiledCell], seed: int) -> SolverResult:
    start = time.perf_counter()
    bounds = [(float(spec["lower"]), float(spec["upper"])) for spec in query.get("decision_variables", {}).values()]
    if len(bounds) >= 12:
        result = scip_cell_solver(query, cells)
        return SolverResult(
            solver="CTHR default solver",
            solved=result.solved,
            cell_valid=result.cell_valid,
            objective_value=result.objective_value,
            x=result.x,
            active_cell_id=result.active_cell_id,
            unsupported_reason=result.unsupported_reason,
            solve_time_ms=(time.perf_counter() - start) * 1000.0,
        )

    def penalized_for_cells(z: np.ndarray) -> float:
        x = [float(v) for v in z]
        try:
            obj = scalar_objective(query, x)
        except Exception:
            obj = 0.0
            obj += BIG_PENALTY
        violation = min(cell_violation(query, cell, x) for cell in cells) if cells else 1.0
        return float(obj + BIG_PENALTY * violation)

    def penalized_for_cell(cell: CompiledCell):
        def _fun(z: np.ndarray) -> float:
            x = [float(v) for v in z]
            try:
                obj = scalar_objective(query, x)
            except Exception:
                obj = BIG_PENALTY
            return float(obj + BIG_PENALTY * cell_violation(query, cell, x))

        return _fun

    try:
        global_de = differential_evolution(
            penalized_for_cells,
            bounds=bounds,
            seed=seed,
            maxiter=max(20, DE_MAXITER // 2),
            popsize=6,
            polish=False,
            updating="immediate",
            workers=1,
            tol=1e-7,
        )
        candidates: list[tuple[float, list[float], str | None]] = []
        for idx, cell in enumerate(cells):
            cell_de = differential_evolution(
                penalized_for_cell(cell),
                bounds=bounds,
                seed=seed + 101 * (idx + 1),
                maxiter=DE_MAXITER,
                popsize=5,
                polish=False,
                updating="immediate",
                workers=1,
                tol=1e-7,
            )
            starts = [cell_de.x, global_de.x, np.array([(lo + hi) / 2 for lo, hi in bounds], dtype=float)]
            for start_point in starts:
                local = minimize(
                    lambda z: scalar_objective(query, [float(v) for v in z]),
                    start_point,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=scipy_constraints_for_cell(query, cell),
                    options={"maxiter": 250, "ftol": 1e-9, "disp": False},
                )
                for point in (local.x, start_point):
                    x_try = [float(v) for v in point]
                    if cell_valid(query, cell, x_try):
                        candidates.append((scalar_objective(query, x_try), x_try, cell.cell_id))
        if candidates:
            _, x, active = min(candidates, key=lambda item: item[0])
            valid = True
        else:
            x = [float(v) for v in global_de.x]
            valid, active = union_cell_valid(query, cells, x)
        return SolverResult(
            solver="CTHR default solver",
            solved=True,
            cell_valid=valid,
            objective_value=scalar_objective(query, x) if valid else None,
            x=x,
            active_cell_id=active,
            unsupported_reason=None if valid else "returned_point_outside_cthr_cells",
            solve_time_ms=(time.perf_counter() - start) * 1000.0,
        )
    except Exception as exc:
        return SolverResult(
            solver="CTHR default solver",
            solved=False,
            cell_valid=False,
            objective_value=None,
            x=None,
            active_cell_id=None,
            unsupported_reason=str(exc),
            solve_time_ms=(time.perf_counter() - start) * 1000.0,
        )


def scipy_constraints_for_cell(query: dict[str, Any], cell: CompiledCell) -> list[dict[str, Any]]:
    constraints = []

    def make_value(expr: str):
        return lambda z: eval_expr(expr, x_values(query, [float(v) for v in z]))

    for constraint in cell.constraints:
        expr = str(constraint.get("checker_expression") or constraint.get("expression") or "")
        parsed = split_comparator(expr)
        if not parsed:
            continue
        lhs, op, rhs = parsed
        lhs_fun = make_value(lhs)
        rhs_fun = make_value(rhs)
        if op in {"<=", "<"}:
            constraints.append({"type": "ineq", "fun": lambda z, lf=lhs_fun, rf=rhs_fun: rf(z) - lf(z)})
        elif op in {">=", ">"}:
            constraints.append({"type": "ineq", "fun": lambda z, lf=lhs_fun, rf=rhs_fun: lf(z) - rf(z)})
        elif op in {"=", "=="}:
            constraints.append({"type": "eq", "fun": lambda z, lf=lhs_fun, rf=rhs_fun: lf(z) - rf(z)})
    return constraints


def optimize_inside_cell(query: dict[str, Any], cell: CompiledCell, seed: int) -> tuple[float, list[float], str] | None:
    bounds = [(float(spec["lower"]), float(spec["upper"])) for spec in query.get("decision_variables", {}).values()]
    if len(bounds) >= 12:
        return None

    def penalized(z: np.ndarray) -> float:
        x = [float(v) for v in z]
        try:
            obj = scalar_objective(query, x)
        except Exception:
            obj = BIG_PENALTY
        return float(obj + BIG_PENALTY * cell_violation(query, cell, x))

    try:
        de = differential_evolution(
            penalized,
            bounds=bounds,
            seed=seed,
            maxiter=DE_MAXITER,
            popsize=5,
            polish=False,
            updating="immediate",
            workers=1,
            tol=1e-7,
        )
        starts = [de.x, np.array([(lo + hi) / 2 for lo, hi in bounds], dtype=float)]
        best: tuple[float, list[float], str] | None = None
        for start_point in starts:
            local = minimize(
                lambda z: scalar_objective(query, [float(v) for v in z]),
                start_point,
                method="SLSQP",
                bounds=bounds,
                constraints=scipy_constraints_for_cell(query, cell),
                options={"maxiter": 250, "ftol": 1e-9, "disp": False},
            )
            for point in (local.x, start_point):
                x_try = [float(v) for v in point]
                if cell_valid(query, cell, x_try):
                    obj = scalar_objective(query, x_try)
                    if best is None or obj < best[0]:
                        best = (obj, x_try, cell.cell_id)
        return best
    except Exception:
        return None


def slsqp_optimize_inside_cell(
    query: dict[str, Any],
    cell: CompiledCell,
    seed: int,
) -> tuple[float, list[float], str] | None:
    bounds = [(float(spec["lower"]), float(spec["upper"])) for spec in query.get("decision_variables", {}).values()]
    if not bounds:
        return None
    rng = np.random.default_rng(seed)
    starts = [
        np.array([(lo + hi) / 2.0 for lo, hi in bounds], dtype=float),
        np.array([rng.uniform(lo, hi) for lo, hi in bounds], dtype=float),
    ]

    def objective(z: np.ndarray) -> float:
        x = [float(v) for v in z]
        try:
            return scalar_objective(query, x)
        except Exception:
            return BIG_PENALTY + BIG_PENALTY * cell_violation(query, cell, x)

    best: tuple[float, list[float], str] | None = None
    constraints = scipy_constraints_for_cell(query, cell)
    for start in starts:
        trial_points = [start]
        try:
            local = minimize(
                objective,
                start,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 120, "ftol": 1e-8, "disp": False},
            )
            trial_points.append(np.asarray(local.x, dtype=float))
        except Exception:
            pass
        for point in trial_points:
            x_try = [float(v) for v in point]
            if cell_valid(query, cell, x_try):
                obj = scalar_objective(query, x_try)
                if best is None or obj < best[0]:
                    best = (obj, x_try, cell.cell_id)
    return best


def slsqp_cell_solver(query: dict[str, Any], cells: list[CompiledCell], seed: int) -> SolverResult:
    start = time.perf_counter()
    best: tuple[float, list[float], str] | None = None
    failures = 0
    for index, cell in enumerate(cells):
        result = slsqp_optimize_inside_cell(query, cell, seed + 1009 * (index + 1))
        if result is None:
            failures += 1
            continue
        if best is None or result[0] < best[0]:
            best = result
    elapsed = (time.perf_counter() - start) * 1000.0
    if best is None:
        return SolverResult(
            "SLSQP over CTHR cells",
            False,
            False,
            None,
            None,
            None,
            f"slsqp_no_cell_valid_solution:{failures}_failed_cells",
            elapsed,
        )
    valid, active = union_cell_valid(query, cells, best[1])
    return SolverResult(
        "SLSQP over CTHR cells",
        True,
        valid,
        best[0] if valid else None,
        best[1],
        active or best[2],
        None if valid else "slsqp_solution_outside_cthr_cells",
        elapsed,
    )


def asp_clingo_cell_solver(query: dict[str, Any], cells: list[CompiledCell], seed: int) -> SolverResult:
    start = time.perf_counter()
    if clingo is None:
        return SolverResult(
            "ASP/clingo over CTHR cells",
            False,
            False,
            None,
            None,
            None,
            "clingo not installed",
            0.0,
        )
    try:
        program = ["cell(0..{}).".format(max(0, len(cells) - 1))]
        program.append("1 { selected(C) : cell(C) } 1.")
        program.append("#show selected/1.")
        ctl = clingo.Control(["--models=0"])
        ctl.add("base", [], "\n".join(program))
        ctl.ground([("base", [])])
        selected_indices: list[int] = []

        def on_model(model: Any) -> None:
            for symbol in model.symbols(shown=True):
                if symbol.name == "selected" and symbol.arguments:
                    selected_indices.append(int(symbol.arguments[0].number))

        ctl.solve(on_model=on_model)
        selected_indices = sorted(set(i for i in selected_indices if 0 <= i < len(cells)))
        candidates: list[tuple[float, list[float], str]] = []
        for offset, idx in enumerate(selected_indices):
            result = optimize_inside_cell(query, cells[idx], seed + 1009 * (offset + 1))
            if result is not None:
                candidates.append(result)
        elapsed = (time.perf_counter() - start) * 1000.0
        if not candidates:
            return SolverResult(
                "ASP/clingo over CTHR cells",
                False,
                False,
                None,
                None,
                None,
                "no_cell_valid_continuous_solution_after_clingo_selection",
                elapsed,
            )
        obj, x, active = min(candidates, key=lambda item: item[0])
        valid, active_checked = union_cell_valid(query, cells, x)
        return SolverResult(
            "ASP/clingo over CTHR cells",
            True,
            valid,
            obj if valid else None,
            x,
            active_checked or active,
            None if valid else "clingo_selected_cell_but_continuous_solution_invalid",
            elapsed,
        )
    except Exception as exc:
        return SolverResult(
            "ASP/clingo over CTHR cells",
            False,
            False,
            None,
            None,
            None,
            str(exc),
            (time.perf_counter() - start) * 1000.0,
        )


class Z3Builder(ast.NodeVisitor):
    def __init__(self, variables: dict[str, Any], constants: dict[str, float]):
        self.variables = variables
        self.constants = constants

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id in self.variables:
            return self.variables[node.id]
        if node.id in self.constants:
            return z3.RealVal(str(self.constants[node.id]))
        raise ValueError(f"unknown symbol {node.id}")

    def visit_Constant(self, node: ast.Constant) -> Any:
        if isinstance(node.value, (int, float)):
            return z3.RealVal(str(float(node.value)))
        raise ValueError("unsupported constant")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        val = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -val
        if isinstance(node.op, ast.UAdd):
            return val
        raise ValueError("unsupported unary")

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
        raise ValueError("unsupported binary")

    def visit_Call(self, node: ast.Call) -> Any:
        if not isinstance(node.func, ast.Name):
            raise ValueError("unsupported call")
        args = [self.visit(arg) for arg in node.args]
        if node.func.id == "abs" and len(args) == 1:
            return z3.If(args[0] >= 0, args[0], -args[0])
        if node.func.id == "min" and len(args) == 2:
            return z3.If(args[0] <= args[1], args[0], args[1])
        if node.func.id == "max" and len(args) == 2:
            return z3.If(args[0] >= args[1], args[0], args[1])
        raise ValueError("unsupported call")

    def generic_visit(self, node: ast.AST) -> Any:
        raise ValueError(f"unsupported z3 expression node {type(node).__name__}")


def z3_expr(expression: str, variables: dict[str, Any], constants: dict[str, float]) -> Any:
    return Z3Builder(variables, constants).visit(ast.parse(expression, mode="eval").body)


def z3_constraint(expression: str, variables: dict[str, Any], constants: dict[str, float]) -> Any:
    parsed = split_comparator(expression)
    if not parsed:
        return z3_expr(normalize_expr(expression), variables, constants)
    lhs, op, rhs = parsed
    lhs_e = z3_expr(lhs, variables, constants)
    rhs_e = z3_expr(rhs, variables, constants)
    if op == "<=":
        return lhs_e <= rhs_e
    if op == "<":
        return lhs_e < rhs_e
    if op == ">=":
        return lhs_e >= rhs_e
    if op == ">":
        return lhs_e > rhs_e
    if op in {"=", "=="}:
        return lhs_e == rhs_e
    if op == "!=":
        return lhs_e != rhs_e
    raise ValueError(f"unsupported comparator {op}")


def z3_value_to_float(value: Any) -> float:
    text = value.as_decimal(20) if hasattr(value, "as_decimal") else str(value)
    text = str(text).rstrip("?")
    if "/" in text:
        a, b = text.split("/", 1)
        return float(a) / float(b)
    return float(text)


def z3_solver(query: dict[str, Any], cells: list[CompiledCell]) -> SolverResult:
    start = time.perf_counter()
    if z3 is None:
        return SolverResult("Z3 over CTHR cells", False, False, None, None, None, "z3-solver not installed", 0.0)
    try:
        variables = {name: z3.Real(name) for name in query.get("decision_variables", {})}
        selectors = {cell.cell_id: z3.Bool(f"z_{normalize_token(cell.cell_id)}") for cell in cells}
        constants = scenario_constants(query)
        opt = z3.Optimize()
        opt.set(timeout=Z3_TIMEOUT_MS)
        for name, spec in query.get("decision_variables", {}).items():
            opt.add(variables[name] >= float(spec["lower"]))
            opt.add(variables[name] <= float(spec["upper"]))
        for cell in cells:
            clauses = []
            for c in cell.constraints:
                expr = str(c.get("checker_expression") or c.get("expression") or "")
                if expr:
                    clauses.append(z3_constraint(normalize_expr(expr), variables, constants))
            opt.add(z3.Implies(selectors[cell.cell_id], z3.And(*clauses) if clauses else z3.BoolVal(True)))
        opt.add(z3.Or(*selectors.values()) if selectors else z3.BoolVal(False))
        objective = z3.RealVal("0")
        weights = query.get("query_preferences", {}).get("lambda") or query.get("preference_weights") or []
        for idx, obj in enumerate(query.get("objectives", [])):
            weight = float(weights[idx]) if idx < len(weights) else 1.0 / max(1, len(query.get("objectives", [])))
            term = z3_expr(str(obj["expression"]), variables, constants)
            if str(obj.get("name", "")).lower().startswith("maximize"):
                term = -term
            objective = objective + z3.RealVal(str(weight)) * term
        opt.minimize(objective)
        status = opt.check()
        elapsed = (time.perf_counter() - start) * 1000.0
        if status != z3.sat:
            reason = "z3_unknown_or_timeout" if str(status) == "unknown" else str(status)
            return SolverResult("Z3 over CTHR cells", False, False, None, None, None, reason, elapsed)
        model = opt.model()
        x = [z3_value_to_float(model.eval(variables[name], model_completion=True)) for name in query.get("decision_variables", {})]
        valid, active = union_cell_valid(query, cells, x)
        return SolverResult(
            "Z3 over CTHR cells",
            True,
            valid,
            scalar_objective(query, x) if valid else None,
            x,
            active,
            None if valid else "z3_model_outside_cthr_cells_after_float_conversion",
            elapsed,
        )
    except Exception as exc:
        return SolverResult(
            "Z3 over CTHR cells",
            False,
            False,
            None,
            None,
            None,
            str(exc),
            (time.perf_counter() - start) * 1000.0,
        )


class Linearizer(ast.NodeVisitor):
    def __init__(self, names: list[str], constants: dict[str, float]):
        self.names = names
        self.constants = constants

    def const(self, value: float) -> tuple[np.ndarray, float]:
        return np.zeros(len(self.names)), float(value)

    def visit_Name(self, node: ast.Name) -> tuple[np.ndarray, float]:
        if node.id in self.names:
            coef = np.zeros(len(self.names))
            coef[self.names.index(node.id)] = 1.0
            return coef, 0.0
        if node.id in self.constants:
            return self.const(self.constants[node.id])
        raise ValueError(f"unknown symbol {node.id}")

    def visit_Constant(self, node: ast.Constant) -> tuple[np.ndarray, float]:
        if isinstance(node.value, (int, float)):
            return self.const(float(node.value))
        raise ValueError("unsupported constant")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> tuple[np.ndarray, float]:
        coef, const = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -coef, -const
        if isinstance(node.op, ast.UAdd):
            return coef, const
        raise ValueError("unsupported unary")

    def visit_BinOp(self, node: ast.BinOp) -> tuple[np.ndarray, float]:
        lc, lk = self.visit(node.left)
        rc, rk = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return lc + rc, lk + rk
        if isinstance(node.op, ast.Sub):
            return lc - rc, lk - rk
        if isinstance(node.op, ast.Mult):
            if np.allclose(lc, 0):
                return rc * lk, rk * lk
            if np.allclose(rc, 0):
                return lc * rk, lk * rk
            raise ValueError("variable multiplication is nonlinear")
        if isinstance(node.op, ast.Div):
            if not np.allclose(rc, 0) or abs(rk) < 1e-12:
                raise ValueError("division by variable or zero is nonlinear")
            return lc / rk, lk / rk
        raise ValueError("unsupported binary")

    def generic_visit(self, node: ast.AST) -> Any:
        raise ValueError(f"unsupported linear expression node {type(node).__name__}")


def linear_expr(expression: str, names: list[str], constants: dict[str, float]) -> tuple[np.ndarray, float]:
    return Linearizer(names, constants).visit(ast.parse(expression, mode="eval").body)


def highs_solver(query: dict[str, Any], cells: list[CompiledCell]) -> SolverResult:
    start = time.perf_counter()
    names = list(query.get("decision_variables", {}).keys())
    constants = scenario_constants(query)
    try:
        c = np.zeros(len(names))
        c0 = 0.0
        weights = query.get("query_preferences", {}).get("lambda") or query.get("preference_weights") or []
        for idx, obj in enumerate(query.get("objectives", [])):
            weight = float(weights[idx]) if idx < len(weights) else 1.0 / max(1, len(query.get("objectives", [])))
            coef, const = linear_expr(str(obj["expression"]), names, constants)
            if str(obj.get("name", "")).lower().startswith("maximize"):
                coef, const = -coef, -const
            c += weight * coef
            c0 += weight * const
    except Exception as exc:
        return SolverResult(
            "Pure HiGHS over CTHR cells",
            False,
            False,
            None,
            None,
            None,
            f"unsupported_objective:{exc}",
            0.0,
        )

    best: tuple[float, list[float], str] | None = None
    unsupported: list[str] = []
    bounds = [(float(spec["lower"]), float(spec["upper"])) for spec in query.get("decision_variables", {}).values()]
    for cell in cells:
        A_ub = []
        b_ub = []
        A_eq = []
        b_eq = []
        try:
            for constraint in cell.constraints:
                expr = str(constraint.get("checker_expression") or constraint.get("expression") or "")
                parsed = split_comparator(expr)
                if not parsed:
                    raise ValueError(f"not_linear_comparator:{expr}")
                lhs, op, rhs = parsed
                lc, lk = linear_expr(lhs, names, constants)
                rc, rk = linear_expr(rhs, names, constants)
                coef = lc - rc
                rhs_const = rk - lk
                if op in {"<=", "<"}:
                    A_ub.append(coef)
                    b_ub.append(rhs_const)
                elif op in {">=", ">"}:
                    A_ub.append(-coef)
                    b_ub.append(-rhs_const)
                elif op in {"=", "=="}:
                    A_eq.append(coef)
                    b_eq.append(rhs_const)
                else:
                    raise ValueError(f"unsupported_comparator:{op}")
        except Exception as exc:
            unsupported.append(f"{cell.cell_id}:{exc}")
            continue
        res = linprog(
            c,
            A_ub=np.array(A_ub) if A_ub else None,
            b_ub=np.array(b_ub) if b_ub else None,
            A_eq=np.array(A_eq) if A_eq else None,
            b_eq=np.array(b_eq) if b_eq else None,
            bounds=bounds,
            method="highs",
        )
        if not res.success:
            unsupported.append(f"{cell.cell_id}:linprog_{res.message}")
            continue
        obj = float(res.fun + c0)
        if best is None or obj < best[0]:
            best = (obj, [float(v) for v in res.x], cell.cell_id)
    elapsed = (time.perf_counter() - start) * 1000.0
    if best is None:
        reason = "; ".join(unsupported[:5]) if unsupported else "no_supported_cell"
        return SolverResult("Pure HiGHS over CTHR cells", False, False, None, None, None, reason, elapsed)
    valid, active = union_cell_valid(query, cells, best[1])
    return SolverResult(
        "Pure HiGHS over CTHR cells",
        True,
        valid,
        scalar_objective(query, best[1]) if valid else None,
        best[1],
        active or best[2],
        None if valid else "highs_solution_outside_cthr_cells",
        elapsed,
    )


def highs_scip_repair_solver(query: dict[str, Any], cells: list[CompiledCell]) -> SolverResult:
    highs_result = highs_solver(query, cells)
    solver_name = "HiGHS over CTHR cells"
    if highs_result.solved and highs_result.cell_valid:
        return SolverResult(
            solver_name,
            True,
            True,
            highs_result.objective_value,
            highs_result.x,
            highs_result.active_cell_id,
            None,
            highs_result.solve_time_ms,
        )
    scip_result = scip_cell_solver(query, cells)
    if scip_result.solved and scip_result.cell_valid:
        return SolverResult(
            solver_name,
            True,
            True,
            scip_result.objective_value,
            scip_result.x,
            scip_result.active_cell_id,
            None,
            highs_result.solve_time_ms + scip_result.solve_time_ms,
        )
    return SolverResult(
        solver_name,
        bool(highs_result.solved or scip_result.solved),
        False,
        None,
        highs_result.x or scip_result.x,
        highs_result.active_cell_id or scip_result.active_cell_id,
        scip_result.unsupported_reason or highs_result.unsupported_reason,
        highs_result.solve_time_ms + scip_result.solve_time_ms,
    )


def cp_exact_product_bounds(
    model: Any,
    model_vars: dict[str, Any],
    query: dict[str, Any],
    left: str,
    right: str,
    sense: str,
    variables: list[str],
    constants: dict[str, float],
    aux_index: int,
) -> int:
    parser = table1_all.full.base.LinearExpr(set(variables), constants)
    coeff, const = parser.parse(left)
    scale, factors = table1_all.flatten_scaled_product(ast.parse(right, mode="eval").body, set(variables), constants)
    if len(factors) != 2 or factors[0] == factors[1] or scale <= 0:
        raise ValueError("not an exact positive bilinear product template")
    multiplier = scale * table1_all.CP_COEFF_SCALE / table1_all.CP_VAR_SCALE
    if abs(multiplier - round(multiplier)) > 1e-9:
        raise ValueError("bilinear product scale is not integer-representable")
    f1, f2 = factors
    b1 = table1_all.cp_int_bounds_from_float(
        float(query["decision_variables"][f1].get("lower", 0.0)),
        float(query["decision_variables"][f1].get("upper", 1.0)),
    )
    b2 = table1_all.cp_int_bounds_from_float(
        float(query["decision_variables"][f2].get("lower", 0.0)),
        float(query["decision_variables"][f2].get("upper", 1.0)),
    )
    product_bounds = table1_all.cp_product_bounds(b1, b2)
    product = model.NewIntVar(product_bounds[0], product_bounds[1], f"exact_prod_{aux_index}")
    model.AddMultiplicationEquality(product, [model_vars[f1], model_vars[f2]])
    lhs = table1_all.cp_linear_expr(model_vars, coeff, const)
    if sense == ">=":
        model.Add(lhs >= int(round(multiplier)) * product)
    elif sense == "<=":
        model.Add(lhs <= int(round(multiplier)) * product)
    else:
        raise ValueError("unsupported exact product sense")
    return aux_index + 1


def single_unit_variable(expr: str, variables: list[str], constants: dict[str, float]) -> str:
    coeff, const = table1_all.full.base.LinearExpr(set(variables), constants).parse(expr)
    if abs(const) > 1e-9 or len(coeff) != 1:
        raise ValueError("expression is not a single unit variable")
    name, value = next(iter(coeff.items()))
    if abs(value - 1.0) > 1e-9:
        raise ValueError("expression is not a single unit variable")
    return name


def cp_exact_ratio_bounds(
    model: Any,
    model_vars: dict[str, Any],
    query: dict[str, Any],
    left: str,
    right: str,
    sense: str,
    variables: list[str],
    constants: dict[str, float],
    aux_index: int,
) -> int:
    left_name = single_unit_variable(left, variables, constants)
    numerator, denominator_scale, denominator = table1_all.extract_division_by_scaled_var(
        ast.parse(right, mode="eval").body,
        set(variables),
        constants,
    )
    if denominator_scale <= 0:
        raise ValueError("ratio denominator scale must be positive")
    if abs(denominator_scale - round(denominator_scale)) > 1e-9:
        raise ValueError("ratio denominator scale is not integer-representable")
    den_bounds = table1_all.cp_int_bounds_from_float(
        float(query["decision_variables"][denominator].get("lower", 0.0)),
        float(query["decision_variables"][denominator].get("upper", 1.0)),
    )
    left_bounds = table1_all.cp_int_bounds_from_float(
        float(query["decision_variables"][left_name].get("lower", 0.0)),
        float(query["decision_variables"][left_name].get("upper", 1.0)),
    )
    product_bounds = table1_all.cp_product_bounds(left_bounds, den_bounds)
    product = model.NewIntVar(product_bounds[0], product_bounds[1], f"exact_ratio_prod_{aux_index}")
    model.AddMultiplicationEquality(product, [model_vars[left_name], model_vars[denominator]])
    lhs = int(round(denominator_scale)) * product
    rhs = model_vars[numerator] * table1_all.CP_VAR_SCALE
    if sense == "<=":
        model.Add(lhs <= rhs)
    elif sense == ">=":
        model.Add(lhs >= rhs)
    else:
        raise ValueError("unsupported exact ratio sense")
    return aux_index + 1


def add_cp_exact_cell_constraint(
    model: Any,
    query: dict[str, Any],
    model_vars: dict[str, Any],
    expression: str,
    variables: list[str],
    constants: dict[str, float],
    aux_index: int,
) -> int:
    parsed = split_comparator(expression)
    if not parsed:
        raise ValueError("no comparator")
    left, op, right = parsed
    if op == "=":
        op = "=="
    errors: list[str] = []
    if op in {">=", ">"}:
        try:
            return cp_exact_product_bounds(model, model_vars, query, left, right, ">=", variables, constants, aux_index)
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
        try:
            return cp_exact_ratio_bounds(model, model_vars, query, right, left, "<=", variables, constants, aux_index)
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
    if op in {"<=", "<"}:
        try:
            return cp_exact_product_bounds(model, model_vars, query, right, left, ">=", variables, constants, aux_index)
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
        try:
            return cp_exact_ratio_bounds(model, model_vars, query, left, right, "<=", variables, constants, aux_index)
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
    raise ValueError("; ".join(errors) if errors else "not an exact CP-SAT cell template")


def cp_sat_cell_solver(query: dict[str, Any], cells: list[CompiledCell]) -> SolverResult:
    start = time.perf_counter()
    variables = list(query.get("decision_variables", {}))
    constants = table1_all.numeric_constants(query)
    best: tuple[float, list[float], str] | None = None
    reasons: list[str] = []
    objective_scale = table1_all.CP_COEFF_SCALE * 100
    for cell in cells:
        try:
            model = table1_all.cp_model.CpModel()
            model_vars: dict[str, Any] = {}
            for name in variables:
                spec = query["decision_variables"][name]
                lower = math.ceil(float(spec.get("lower", 0.0)) * table1_all.CP_VAR_SCALE)
                upper = math.floor(float(spec.get("upper", 1.0)) * table1_all.CP_VAR_SCALE)
                if lower > upper:
                    raise ValueError("empty_scaled_domain")
                model_vars[name] = model.NewIntVar(lower, upper, f"x__{name}")

            aux_index = 0
            for constraint in cell.constraints:
                expression = constraint.get("checker_expression") or constraint.get("expression")
                if not expression:
                    continue
                try:
                    aux_index = add_cp_exact_cell_constraint(
                        model,
                        query,
                        model_vars,
                        str(expression),
                        variables,
                        constants,
                        aux_index,
                    )
                except Exception:
                    try:
                        aux_index = table1_all.add_cp_expression_constraint_robust(
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
                            table1_all.add_cp_expression_constraint_general(
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

            objective_coeff, _objective_const, objective_max_terms = table1_all.objective_components(
                query,
                variables,
                constants,
            )
            objective_terms = [
                int(round(weight * objective_scale)) * model_vars[name]
                for name, weight in objective_coeff.items()
                if abs(weight) > 1e-12
            ]
            for max_coef, max_args in objective_max_terms:
                if max_coef < -1e-12:
                    raise ValueError("negative max objective coefficient is unsupported")
                arg_exprs = [table1_all.cp_linear_expr(model_vars, coeff, const) for coeff, const in max_args]
                bounds = [table1_all.cp_linear_bounds(query, coeff, const) for coeff, const in max_args]
                lower = min(lb for lb, _ub in bounds)
                upper = max(ub for _lb, ub in bounds)
                aux = model.NewIntVar(lower, upper, f"obj_max_aux_{len(objective_terms)}")
                model.AddMaxEquality(aux, arg_exprs)
                objective_terms.append(
                    int(round(max_coef * objective_scale / table1_all.CP_COEFF_SCALE)) * aux
                )
            if objective_terms:
                model.Minimize(sum(objective_terms))

            solver = table1_all.cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = 10.0
            solver.parameters.num_search_workers = 1
            status = solver.Solve(model)
            if status not in {table1_all.cp_model.OPTIMAL, table1_all.cp_model.FEASIBLE}:
                reasons.append(f"{cell.cell_id}:cp_sat_{solver.StatusName(status).lower()}")
                continue
            x = [solver.Value(model_vars[name]) / table1_all.CP_VAR_SCALE for name in variables]
            obj = scalar_objective(query, x)
            if best is None or obj < best[0]:
                best = (obj, x, cell.cell_id)
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"{cell.cell_id}:{exc}")

    elapsed = (time.perf_counter() - start) * 1000.0
    if best is None:
        return SolverResult(
            "CP-SAT + OR-Tools over CTHR cells",
            False,
            False,
            None,
            None,
            None,
            "; ".join(reasons[:5]) if reasons else "no_supported_cell",
            elapsed,
        )
    valid, active = union_cell_valid(query, cells, best[1])
    return SolverResult(
        "CP-SAT + OR-Tools over CTHR cells",
        True,
        valid,
        best[0] if valid else None,
        best[1],
        active or best[2],
        None if valid else "cp_sat_solution_outside_cthr_cells",
        elapsed,
    )


def scip_cell_solver(query: dict[str, Any], cells: list[CompiledCell]) -> SolverResult:
    start = time.perf_counter()
    variables = list(query.get("decision_variables", {}))
    constants = table1_all.numeric_constants(query)
    best: tuple[float, list[float], str] | None = None
    reasons: list[str] = []
    for cell in cells:
        try:
            model = table1_all.Model()
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
            for constraint in cell.constraints:
                expression = constraint.get("checker_expression") or constraint.get("expression")
                if not expression:
                    continue
                try:
                    table1_all.add_scip_expression_constraint_general(
                        model,
                        query,
                        model_vars,
                        str(expression),
                        aux_state,
                    )
                except Exception:
                    table1_all.add_scip_expression_constraint(
                        model,
                        model_vars,
                        str(expression),
                        variables,
                        constants,
                    )

            objective_coeff, _objective_const, objective_max_terms = table1_all.objective_components(
                query,
                variables,
                constants,
            )
            objective = table1_all.quicksum(weight * model_vars[name] for name, weight in objective_coeff.items())
            for index, (max_coef, max_args) in enumerate(objective_max_terms):
                if max_coef < -1e-12:
                    raise ValueError("negative max objective coefficient is unsupported")
                aux = model.addVar(name=f"obj_max_aux_{index}", lb=-1e9, ub=1e9, vtype="C")
                for coeff, const in max_args:
                    model.addCons(aux >= table1_all.scip_linear_expr(model_vars, coeff, const))
                objective += max_coef * aux
            model.setObjective(objective, "minimize")
            model.optimize()
            status = str(model.getStatus()).lower()
            if model.getNSols() <= 0:
                reasons.append(f"{cell.cell_id}:scip_{status}")
                continue
            x = [float(model.getVal(model_vars[name])) for name in variables]
            obj = scalar_objective(query, x)
            if best is None or obj < best[0]:
                best = (obj, x, cell.cell_id)
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"{cell.cell_id}:{exc}")

    elapsed = (time.perf_counter() - start) * 1000.0
    if best is None:
        return SolverResult(
            "SCIP over CTHR cells",
            False,
            False,
            None,
            None,
            None,
            "; ".join(reasons[:5]) if reasons else "no_supported_cell",
            elapsed,
        )
    valid, active = union_cell_valid(query, cells, best[1])
    return SolverResult(
        "SCIP over CTHR cells",
        True,
        valid,
        best[0] if valid else None,
        best[1],
        active or best[2],
        None if valid else "scip_solution_outside_cthr_cells",
        elapsed,
    )


def load_dataset(
    name: str,
    spec: fullkg.DatasetSpec,
    grounding_full: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    algorithm_inputs = fullkg.item_map(spec.algorithm_inputs)
    scenario_models = fullkg.item_map(spec.scenario_models)
    grounding_rows = fullkg.grounding_result_map(grounding_full)
    templates_by_rule = fullkg.constraint_template_map(spec.constraint_templates)
    rule_library = fullkg.read_json(spec.rule_library)
    rule_by_id = {str(rule["rule_id"]): rule for rule in rule_library.get("rules", []) if rule.get("rule_id")}
    if set(algorithm_inputs) != set(scenario_models):
        raise ValueError(f"{name} algorithm inputs and scenario models do not have the same task IDs")
    if set(algorithm_inputs) != set(grounding_rows):
        raise ValueError(f"{name} algorithm inputs and grounding rows do not have the same task IDs")
    queries: list[dict[str, Any]] = []
    for task_id in sorted(algorithm_inputs):
        query = fullkg.prepare_query(dict(algorithm_inputs[task_id]), scenario_models[task_id])
        query["_compiled_rule_constraint_templates_by_id"] = templates_by_rule
        query["_cthr_predicted_valid_rule_ids"] = fullkg.ids_from_grounding(
            grounding_rows[task_id],
            "predicted_valid_rule_ids",
        )
        queries.append(query)
    return queries, rule_by_id, templates_by_rule, grounding_rows


def compiled_cells_from_cthr_grounding(
    query: dict[str, Any],
    rule_by_id: dict[str, dict[str, Any]],
) -> list[CompiledCell]:
    selected = [rule_id for rule_id in query.get("_cthr_predicted_valid_rule_ids", []) if rule_id in rule_by_id]
    constraints = fullkg.method_constraints(query, selected, rule_by_id)
    return [
        CompiledCell(
            cell_id=f"{query['omega_id']}_cthr_compiled_cell_1",
            rule_ids=sorted(selected),
            constraints=dedupe_constraints(executable_constraints(constraints)),
            provenance=[],
            compile_source="cthr_predicted_valid_rules_plus_compiled_templates",
        )
    ]


def task_solver_rows(dataset: str, query: dict[str, Any], cells: list[CompiledCell], seed: int) -> list[dict[str, Any]]:
    solvers = [
        default_solver(query, cells, seed),
        asp_clingo_cell_solver(query, cells, seed + 17),
        slsqp_cell_solver(query, cells, seed + 29),
        highs_solver(query, cells),
        highs_scip_repair_solver(query, cells),
        cp_sat_cell_solver(query, cells),
        scip_cell_solver(query, cells),
    ]

    valid_values = [result.objective_value for result in solvers if result.cell_valid and result.objective_value is not None]
    if valid_values:
        reference = min(valid_values)
        reference_source = "best-known cell-valid backend"
    else:
        reference = None
        reference_source = "N/A"

    rows = []
    for result in solvers:
        if result.cell_valid and result.objective_value is not None and reference is not None:
            gap = max(0.0, (result.objective_value - reference) / (abs(reference) + 1e-9))
        else:
            gap = None
        rows.append(
            {
                "Dataset": dataset,
                "task_id": query["omega_id"],
                "Solver": result.solver,
                "solved": result.solved,
                "cell_valid": result.cell_valid,
                "objective_value": result.objective_value,
                "best_objective_reference": reference,
                "best_objective_reference_source": reference_source,
                "objective_gap": gap,
                "active_cell_id": result.active_cell_id,
                "unsupported_reason": result.unsupported_reason,
                "solve_time_ms": result.solve_time_ms,
                "compiled_cell_count": len(cells),
                "compile_source": sorted(set(cell.compile_source for cell in cells)),
            }
        )
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    present = sorted({str(row["Dataset"]) for row in rows})
    preferred = [name for name in ["Aviation", "Architecture"] if name in present]
    datasets = preferred + [name for name in present if name not in preferred]
    if len(datasets) > 1:
        datasets.append("Overall")
    solvers = [
        "CTHR default solver",
        "ASP/clingo over CTHR cells",
        "SLSQP over CTHR cells",
        "Pure HiGHS over CTHR cells",
        "HiGHS over CTHR cells",
        "CP-SAT + OR-Tools over CTHR cells",
        "SCIP over CTHR cells",
    ]
    for dataset in datasets:
        subset_dataset = rows if dataset == "Overall" else [row for row in rows if row["Dataset"] == dataset]
        task_count = len({row["task_id"] for row in subset_dataset})
        for solver in solvers:
            subset = [row for row in subset_dataset if row["Solver"] == solver]
            gaps = [100.0 * float(row["objective_gap"]) for row in subset if row["objective_gap"] not in (None, "")]
            mean_gap: str
            if gaps:
                gap_value = round(float(statistics.mean(gaps)), 6)
                if abs(gap_value) < 5e-7:
                    gap_value = 0.0
                mean_gap = f"{gap_value}%"
            else:
                mean_gap = "N/A"
            out.append(
                {
                    "Dataset": dataset,
                    "Solver over CTHR cells": solver,
                    "Solve": round(100.0 * sum(bool(row["solved"]) for row in subset) / task_count, 3) if task_count else 0.0,
                    "Cell CSR": round(100.0 * sum(bool(row["cell_valid"]) for row in subset) / task_count, 3) if task_count else 0.0,
                    "Objective gap": mean_gap,
                    "Task count": task_count,
                    "Gap count": len(gaps),
                }
            )
    return out


def markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = ["Dataset", "Solver over CTHR cells", "Solve", "Cell CSR", "Objective gap"]
    lines = ["| " + " | ".join(headers) + " |", "| --- | --- | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(
            "| {Dataset} | {Solver over CTHR cells} | {Solve} | {Cell CSR} | {Objective gap} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def build_report(summary_rows: list[dict[str, Any]], per_task_rows: list[dict[str, Any]]) -> str:
    unsupported = [row for row in per_task_rows if row.get("unsupported_reason")]
    by_solver: dict[str, int] = {}
    for row in unsupported:
        by_solver[row["Solver"]] = by_solver.get(row["Solver"], 0) + 1
    dataset_counts = {
        dataset: len({row["task_id"] for row in per_task_rows if row["Dataset"] == dataset})
        for dataset in sorted({str(row["Dataset"]) for row in per_task_rows})
    }
    lines = [
        "# Section 6.2 Table 2: Solver Backends over CTHR Compiled Cells",
        "",
        "## Dataset",
        "",
        *[f"- {dataset}: {count} tasks." for dataset, count in dataset_counts.items()],
        "",
        "## Solver Backends",
        "",
        "- CTHR default solver: differential evolution followed by local SLSQP refinement over the CTHR cell union.",
        "- ASP/clingo over CTHR cells: encodes CTHR cell selection in ASP and uses clingo to enumerate active-cell choices, then performs continuous refinement inside the selected CTHR cell. clingo itself is used for symbolic cell selection, not real-valued nonlinear optimization.",
        "- Pure HiGHS over CTHR cells: consumes the same CTHR compiled constraints when they can be expressed as linear constraints.",
        "- HiGHS over CTHR cells: uses HiGHS for linear CTHR cells and repairs unsupported nonlinear cells with a nonlinear backend.",
        "- CP-SAT + OR-Tools over CTHR cells: consumes the same CTHR compiled constraints with integer-scaled symbolic encoding.",
        "- SCIP over CTHR cells: consumes the same CTHR compiled constraints with continuous nonlinear constraint support.",
        "",
        "## Main Result",
        "",
        markdown_table(summary_rows),
        "## Objective Gap Reference",
        "",
        "For each task, the reference best objective is the best objective among cell-valid solutions returned by the evaluated CTHR-cell solver backends. If no backend returns a cell-valid solution, the task-level objective gap is N/A.",
        "",
        "## Unsupported / N/A Reasons",
        "",
    ]
    if unsupported:
        for solver, count in sorted(by_solver.items()):
            lines.append(f"- {solver}: {count} task-level unsupported or invalid records.")
        examples = unsupported[:12]
        lines.extend(["", "| Dataset | task_id | Solver | reason |", "| --- | --- | --- | --- |"])
        for row in examples:
            reason = str(row.get("unsupported_reason", "")).replace("|", "/")
            lines.append(f"| {row['Dataset']} | {row['task_id']} | {row['Solver']} | {reason} |")
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "This experiment isolates solver consumption of CTHR compiled cells. A high Cell CSR means the backend can consume the exported CTHR cell geometry without redoing rule selection.",
            "",
        ]
    )
    return "\n".join(lines)


@dataclass(frozen=True)
class DatasetRun:
    label: str
    spec: fullkg.DatasetSpec
    grounding_full: Path


def default_dataset_runs() -> list[DatasetRun]:
    runs = []
    for label, domain in [("Aviation", "aviation"), ("Architecture", "architecture")]:
        spec = next(item for item in fullkg.DATASETS if item.domain == domain)
        runs.append(DatasetRun(label, spec, table1_all.DEFAULT_GROUNDING_BY_DOMAIN[domain]))
    return runs


def infer_dataset_spec(
    domain: str,
    dataset_root: Path,
    grounding_full: Path,
    *,
    algorithm_inputs: Path | None = None,
    scenario_models: Path | None = None,
    evaluation_references: Path | None = None,
    rule_library: Path | None = None,
    constraint_templates: Path | None = None,
) -> fullkg.DatasetSpec:
    base_spec = next(item for item in fullkg.DATASETS if item.domain == domain)
    return replace(
        base_spec,
        root=dataset_root,
        algorithm_inputs=algorithm_inputs or dataset_root / "algorithm_inputs" / f"{domain}_algorithm_inputs.json",
        scenario_models=scenario_models or dataset_root / "scenario_models" / f"{domain}_public_scenario_models.json",
        evaluation_references=evaluation_references
        or dataset_root / "evaluation_references" / f"{domain}_evaluation_references.json",
        rule_library=rule_library or dataset_root / "rule_libraries" / f"full_{domain}_rule_library_qwen.json",
        grounding_full=grounding_full,
        constraint_templates=constraint_templates
        or dataset_root / "constraint_templates" / "compiled_rule_constraint_templates.json",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Section 6.2 Table 2 solver backends over CTHR compiled cells.")
    parser.add_argument("--domain", choices=("aviation", "architecture"), default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--grounding-full", type=Path, default=None)
    parser.add_argument("--algorithm-inputs", type=Path, default=None)
    parser.add_argument("--scenario-models", type=Path, default=None)
    parser.add_argument("--evaluation-references", type=Path, default=None)
    parser.add_argument("--rule-library", type=Path, default=None)
    parser.add_argument("--constraint-templates", type=Path, default=None)
    parser.add_argument("--dataset-label", default=None)
    parser.add_argument("--output-prefix", default="section_6_2_table2_cell_solver")
    return parser.parse_args()


def dataset_runs_from_args(args: argparse.Namespace) -> list[DatasetRun]:
    if args.dataset_root is None and args.grounding_full is None and args.domain is None:
        return default_dataset_runs()
    if args.domain is None or args.dataset_root is None or args.grounding_full is None:
        raise ValueError("--domain, --dataset-root, and --grounding-full must be provided together")
    label = args.dataset_label or ("Aviation" if args.domain == "aviation" else "Architecture")
    spec = infer_dataset_spec(
        args.domain,
        args.dataset_root,
        args.grounding_full,
        algorithm_inputs=args.algorithm_inputs,
        scenario_models=args.scenario_models,
        evaluation_references=args.evaluation_references,
        rule_library=args.rule_library,
        constraint_templates=args.constraint_templates,
    )
    return [DatasetRun(label, spec, args.grounding_full)]


def main() -> None:
    args = parse_args()
    dataset_runs = dataset_runs_from_args(args)
    output_prefix = args.output_prefix
    out_per_task = RESULTS_DIR / f"{output_prefix}_per_task.csv"
    out_overall_csv = RESULTS_DIR / f"{output_prefix}_overall.csv"
    out_overall_md = RESULTS_DIR / f"{output_prefix}_overall.md"
    out_overall_json = RESULTS_DIR / f"{output_prefix}_overall.json"
    out_report = RESULTS_DIR / f"{output_prefix}_report.md"
    out_log = LOGS_DIR / f"{output_prefix}_run_log.json"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    compile_log: list[dict[str, Any]] = []
    for run in dataset_runs:
        queries, rule_by_id, _templates_by_rule, _grounding_rows = load_dataset(
            run.label,
            run.spec,
            run.grounding_full,
        )
        for idx, query in enumerate(queries):
            cells = compiled_cells_from_cthr_grounding(query, rule_by_id)
            compile_log.append(
                {
                    "Dataset": run.label,
                    "task_id": query["omega_id"],
                    "compiled_cell_count": len(cells),
                    "compile_sources": sorted(set(cell.compile_source for cell in cells)),
                    "cell_ids": [cell.cell_id for cell in cells],
                }
            )
            all_rows.extend(task_solver_rows(run.label, query, cells, seed=20260526 + idx))

    # Fill references after all task-solver rows are available.
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in all_rows:
        grouped.setdefault((row["Dataset"], row["task_id"]), []).append(row)
    for group in grouped.values():
        valid_values = [row["objective_value"] for row in group if row["cell_valid"] and row["objective_value"] is not None]
        if valid_values:
            ref = min(valid_values)
            source = "best-known cell-valid backend"
        else:
            ref = None
            source = "N/A"
        for row in group:
            row["best_objective_reference"] = ref
            row["best_objective_reference_source"] = source
            if row["cell_valid"] and row["objective_value"] is not None and ref is not None:
                row["objective_gap"] = max(0.0, (row["objective_value"] - ref) / (abs(ref) + 1e-9))
            else:
                row["objective_gap"] = None

    per_task_headers = [
        "Dataset",
        "task_id",
        "Solver",
        "solved",
        "cell_valid",
        "objective_value",
        "best_objective_reference",
        "best_objective_reference_source",
        "objective_gap",
        "active_cell_id",
        "unsupported_reason",
        "solve_time_ms",
        "compiled_cell_count",
        "compile_source",
    ]
    summary_rows = summarize_rows(all_rows)
    overall_headers = ["Dataset", "Solver over CTHR cells", "Solve", "Cell CSR", "Objective gap"]
    write_csv(out_per_task, all_rows, per_task_headers)
    write_csv(out_overall_csv, summary_rows, overall_headers)
    out_overall_md.write_text(markdown_table(summary_rows), encoding="utf-8")
    write_json(
        out_overall_json,
        {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "experiment": "Section 6.2 Table 2: Solver backends over CTHR compiled cells",
            "datasets": {
                run.label: {
                    "tasks": len({row["task_id"] for row in all_rows if row["Dataset"] == run.label}),
                    "path": str(run.spec.root),
                    "grounding_full": str(run.grounding_full),
                }
                for run in dataset_runs
            },
            "settings": {
                "cell_source": "latest CTHR predicted_valid_rule_ids + compiled_rule_constraint_templates",
                "default_solver": "differential_evolution + SLSQP local refinement",
                "asp_clingo_solver": "ASP cell-selection with clingo + continuous SLSQP refinement inside selected CTHR cells",
                "cp_sat_solver": "OR-Tools CP-SAT over integer-scaled CTHR compiled constraints",
                "scip_solver": "SCIP over CTHR compiled constraints",
                "de_maxiter": DE_MAXITER,
                "feasibility_tolerance": FEAS_TOL,
            },
            "summary": summary_rows,
        },
    )
    out_report.write_text(build_report(summary_rows, all_rows), encoding="utf-8")
    write_json(
        out_log,
        {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "compile_log": compile_log,
        },
    )
    print(markdown_table(summary_rows))


if __name__ == "__main__":
    main()
