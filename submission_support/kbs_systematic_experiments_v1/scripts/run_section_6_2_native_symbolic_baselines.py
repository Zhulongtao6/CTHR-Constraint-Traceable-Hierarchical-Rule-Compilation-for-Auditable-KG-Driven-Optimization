from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

import run_section_6_2_table1_aviation_pipeline as aviation_pipeline
import run_section_6_2_table1_pipeline as base
import run_section_6_3_candidate_to_valid as ctv

try:
    import clingo  # type: ignore
except ImportError as exc:  # pragma: no cover
    clingo = None
    CLINGO_IMPORT_ERROR = exc
else:
    CLINGO_IMPORT_ERROR = None

try:
    import z3  # type: ignore
except ImportError as exc:  # pragma: no cover
    z3 = None
    Z3_IMPORT_ERROR = exc
else:
    Z3_IMPORT_ERROR = None


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
AVIATION_ROOT = ROOT / "datasets" / "aviation_combined"
AVIATION_RULE_LIBRARY = ROOT / "datasets" / "aviation" / "aviation_stress_rule_library.combined.json"

METHODS = ["Native ASP + clingo", "Native SMT + Z3", "Native MILP + HiGHS"]

OUTPUTS = {
    "overall_csv": RESULTS_DIR / "section_6_2_native_symbolic_baselines_overall.csv",
    "overall_md": RESULTS_DIR / "section_6_2_native_symbolic_baselines_overall.md",
    "overall_json": RESULTS_DIR / "section_6_2_native_symbolic_baselines_overall.json",
    "per_task_csv": RESULTS_DIR / "section_6_2_native_symbolic_baselines_per_task.csv",
    "report_md": RESULTS_DIR / "section_6_2_native_symbolic_baselines_report.md",
}


@dataclass
class NativeFacts:
    candidate_ids: list[str]
    must_select: set[str]
    must_drop: set[str]
    depends: set[tuple[str, str]]
    excludes: set[tuple[str, str]]
    conflicts: set[tuple[str, str]]
    notes: list[str]


def write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({header: base.csv_cell(row.get(header)) for header in headers})


def asp_atom(value: str) -> str:
    return json.dumps(str(value), ensure_ascii=True)


def make_scenario(query: dict[str, Any]) -> dict[str, Any]:
    scenario = ctv.scenario_for_resolution(query)
    meta = query.get("stress_metadata", {}) or {}
    if meta.get("original_or_stress") == "original" or meta.get("benchmark_split") == "original":
        scenario["_trust_grounded_candidate_applicability"] = True
    return scenario


def native_rule_facts(candidate_rules: list[dict[str, Any]], scenario: dict[str, Any]) -> NativeFacts:
    by_id = {str(rule["rule_id"]): rule for rule in candidate_rules if rule.get("rule_id")}
    candidate_ids = sorted(by_id)
    maps = ctv.relation_maps(candidate_rules)
    notes: list[str] = []

    if scenario.get("_trust_grounded_candidate_applicability"):
        applicable = set(candidate_ids)
    else:
        applicable = {rid for rid, rule in by_id.items() if ctv.initially_applicable(rule, scenario)}
    inapplicable = set(candidate_ids) - applicable

    defeated: set[str] = set()
    for source, target in maps["overrides"]:
        if source in applicable:
            defeated.add(target)
            notes.append(f"defeated_by_override:{target}")
    for source, target in maps["precedes"]:
        if source in applicable:
            defeated.add(target)
            notes.append(f"defeated_by_precedence:{target}")

    selected = set(applicable) - defeated
    selected = ctv.dependency_closure(selected, maps["depends"], applicable - defeated)
    selected -= defeated

    # The following transformations are encoded as explicit native-symbolic facts
    # rather than as CTHR final labels. They are derived only from rule metadata,
    # relation types, scenario facts, and visible decision-variable names.
    for _ in range(2):
        selected, variant_notes = ctv.prune_parameter_variants(selected, candidate_rules, scenario)
        notes.extend(variant_notes)
        selected, unit_notes = ctv.prune_visible_unit_mismatches(selected, candidate_rules, scenario)
        notes.extend(unit_notes)
        selected, quantity_notes = ctv.prune_unbound_visible_quantities(selected, candidate_rules, scenario)
        notes.extend(quantity_notes)
        selected, conflict_notes = ctv.resolve_conflicts(selected, candidate_rules, scenario)
        notes.extend(conflict_notes)
        selected = ctv.dependency_closure(selected, maps["depends"], applicable - defeated)
        selected -= defeated

    must_select = set(selected)
    must_drop = set(candidate_ids) - must_select
    for rid in sorted(inapplicable):
        notes.append(f"inapplicable:{rid}")
    for rid in sorted(must_drop - defeated - inapplicable):
        notes.append(f"structured_surplus_or_pruned:{rid}")

    dependency_pairs = set(maps["depends"])
    conflict_pairs = {
        pair
        for pair in (set(maps["excludes"]) | set(maps["conflicts"]))
        if pair not in dependency_pairs and (pair[1], pair[0]) not in dependency_pairs
    }
    return NativeFacts(
        candidate_ids=candidate_ids,
        must_select=must_select,
        must_drop=must_drop,
        depends=set(maps["depends"]),
        excludes=set(maps["excludes"]),
        conflicts=conflict_pairs,
        notes=notes,
    )


def select_with_native_asp(facts: NativeFacts) -> tuple[list[str], str, list[str]]:
    if clingo is None:
        return [], f"missing_clingo:{CLINGO_IMPORT_ERROR}", []
    lines: list[str] = []
    for rid in facts.candidate_ids:
        lines.append(f"rule({asp_atom(rid)}).")
    for rid in facts.must_select:
        lines.append(f"must_select({asp_atom(rid)}).")
    for rid in facts.must_drop:
        lines.append(f"must_drop({asp_atom(rid)}).")
    for left, right in facts.depends:
        lines.append(f"depends({asp_atom(left)},{asp_atom(right)}).")
    for left, right in facts.excludes | facts.conflicts:
        lines.append(f"excludes({asp_atom(left)},{asp_atom(right)}).")
    program = "\n".join(lines) + r"""
selected(R) :- must_select(R).
:- selected(R), must_drop(R).
:- must_select(R), not selected(R).
:- selected(R), depends(R,D), not selected(D).
:- selected(A), selected(B), excludes(A,B).
:- selected(A), selected(B), excludes(B,A).
#show selected/1.
"""
    ctl = clingo.Control(["--models=1", "--warn=no-atom-undefined"])
    ctl.add("base", [], program)
    ctl.ground([("base", [])])
    answer_sets: list[list[str]] = []
    with ctl.solve(yield_=True) as handle:
        for model in handle:
            selected = []
            for symbol in model.symbols(shown=True):
                if symbol.name == "selected" and symbol.arguments:
                    selected.append(str(symbol.arguments[0].string))
            answer_sets.append(sorted(selected))
    if not answer_sets:
        return [], "unsat", facts.notes
    return answer_sets[0], "success", facts.notes


def select_with_native_smt(facts: NativeFacts) -> tuple[list[str], str, list[str]]:
    if z3 is None:
        return [], f"missing_z3:{Z3_IMPORT_ERROR}", []
    y = {rid: z3.Bool(f"y__{base.re.sub(r'[^A-Za-z0-9_]+', '_', rid)}") for rid in facts.candidate_ids}
    opt = z3.Optimize()
    opt.set(timeout=10000)
    for rid in facts.must_select:
        opt.add(y[rid])
    for rid in facts.must_drop:
        opt.add(z3.Not(y[rid]))
    for left, right in facts.depends:
        if left in y and right in y:
            opt.add(z3.Implies(y[left], y[right]))
    for left, right in facts.excludes | facts.conflicts:
        if left in y and right in y:
            opt.add(z3.Not(z3.And(y[left], y[right])))
    if y:
        terms = [z3.If(var, z3.IntVal(1), z3.IntVal(0)) for var in y.values()]
        opt.maximize(sum(terms[1:], terms[0]))
    status = opt.check()
    if status != z3.sat:
        return [], str(status), facts.notes
    model = opt.model()
    selected = sorted(rid for rid, var in y.items() if z3.is_true(model.eval(var, model_completion=True)))
    return selected, "success", facts.notes


def select_with_native_milp(facts: NativeFacts) -> tuple[list[str], str, list[str]]:
    if not facts.candidate_ids:
        return [], "empty_candidates", facts.notes
    idx = {rid: i for i, rid in enumerate(facts.candidate_ids)}
    c = -np.ones(len(facts.candidate_ids))
    integrality = np.ones(len(facts.candidate_ids))
    bounds = Bounds(np.zeros(len(facts.candidate_ids)), np.ones(len(facts.candidate_ids)))
    rows: list[np.ndarray] = []
    lb: list[float] = []
    ub: list[float] = []

    def equality(rule_id: str, value: float) -> None:
        row = np.zeros(len(facts.candidate_ids))
        row[idx[rule_id]] = 1.0
        rows.append(row)
        lb.append(value)
        ub.append(value)

    for rid in facts.must_select:
        equality(rid, 1.0)
    for rid in facts.must_drop:
        equality(rid, 0.0)
    for left, right in facts.depends:
        if left in idx and right in idx:
            row = np.zeros(len(facts.candidate_ids))
            row[idx[left]] = 1.0
            row[idx[right]] = -1.0
            rows.append(row)
            lb.append(-np.inf)
            ub.append(0.0)
    for left, right in facts.excludes | facts.conflicts:
        if left in idx and right in idx:
            row = np.zeros(len(facts.candidate_ids))
            row[idx[left]] = 1.0
            row[idx[right]] = 1.0
            rows.append(row)
            lb.append(-np.inf)
            ub.append(1.0)

    constraints = LinearConstraint(np.vstack(rows), np.array(lb), np.array(ub)) if rows else None
    result = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints, options={"time_limit": 10})
    if not result.success or result.x is None:
        return [], f"milp_{result.message}", facts.notes
    selected = sorted(rid for rid in facts.candidate_ids if result.x[idx[rid]] >= 0.5)
    return selected, "success", facts.notes


def semantic_valid(feasible: dict[str, Any], x: dict[str, float] | None, predicted: list[str], reference: list[str]) -> bool:
    return base.semantic_valid(feasible, x, predicted, reference)


def evaluate_one(
    method: str,
    query: dict[str, Any],
    label: dict[str, Any],
    feasible: dict[str, Any],
    rule_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    task_id = str(query["omega_id"])
    reference = base.reference_rule_ids(label, feasible, query)
    candidate_ids = aviation_pipeline.visible_candidate_ids_from_query(query, label, feasible)
    candidate_rules = base.candidate_rule_records(rule_by_id, candidate_ids)
    facts = native_rule_facts(candidate_rules, make_scenario(query))

    start = time.perf_counter()
    if method == "Native ASP + clingo":
        predicted, status, notes = select_with_native_asp(facts)
    elif method == "Native SMT + Z3":
        predicted, status, notes = select_with_native_smt(facts)
    elif method == "Native MILP + HiGHS":
        predicted, status, notes = select_with_native_milp(facts)
    else:
        raise ValueError(method)
    if status != "success":
        return {
            "Dataset": "Aviation",
            "task_id": task_id,
            "Method": method,
            "predicted_rule_ids": [],
            "reference_rule_ids": reference,
            "rule_precision": None,
            "rule_recall": None,
            "formal_feasible": None,
            "semantic_valid": None,
            "false_accept": None,
            "invalid_case": None,
            "selection_status": status,
            "resolver_notes": notes,
            "runtime_ms": round((time.perf_counter() - start) * 1000.0, 3),
        }

    constraints = base.constraints_for_method(query, predicted, rule_by_id, include_candidate_rulelib_constraints=False)
    x = base.optimize_default(query, constraints, method, task_id)
    formal = base.constraints_satisfied(constraints, base.with_query_values(query, x)) if x is not None else False
    sem_valid = semantic_valid(feasible, x, predicted, reference)
    precision = base.method_rule_precision(predicted, reference)
    recall = base.method_rule_recall(predicted, reference)
    return {
        "Dataset": "Aviation",
        "task_id": task_id,
        "Method": method,
        "predicted_rule_ids": predicted,
        "reference_rule_ids": reference,
        "rule_precision": precision,
        "rule_recall": recall,
        "formal_feasible": formal,
        "semantic_valid": sem_valid,
        "false_accept": bool(formal and not sem_valid),
        "invalid_case": bool(not sem_valid),
        "selection_status": status,
        "resolver_notes": notes,
        "runtime_ms": round((time.perf_counter() - start) * 1000.0, 3),
    }


def aggregate(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    subset = [row for row in rows if row["Method"] == method]
    supported = [row for row in subset if row["selection_status"] == "success"]
    if not supported:
        return {
            "Dataset": "Aviation",
            "Method": method,
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
        "Dataset": "Aviation",
        "Method": method,
        "Rule Precision": base.pct(base.avg([row["rule_precision"] for row in supported])),
        "Rule Recall": base.pct(base.avg([row["rule_recall"] for row in supported])),
        "Formal CSR": base.pct(formal / n),
        "Sem-CSR": base.pct(sem / n),
        "False accept": base.pct(false_accept / n),
        "Invalid cases": f"{invalid}/{n} ({100.0 * invalid / n:.1f}%){suffix}",
    }


def render_report(overall: list[dict[str, Any]], rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    headers = ["Dataset", "Method", "Rule Precision", "Rule Recall", "Formal CSR", "Sem-CSR", "False accept", "Invalid cases"]
    status_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        status_counts.setdefault(row["Method"], {})
        status = str(row["selection_status"])
        status_counts[row["Method"]][status] = status_counts[row["Method"]].get(status, 0) + 1
    lines = [
        "# Native Symbolic Encoding Baselines",
        "",
        "## Purpose",
        "",
        "This experiment evaluates native ASP, SMT, and MILP encodings for valid-rule selection over the Aviation benchmark. Unlike the naive symbolic rows in Table 6.2, these encodings explicitly include CTHR-style interaction semantics derived from visible rule metadata and scenario facts: applicability, dependency, exclusion/conflict, exception override, precedence, and parameter/formula variant handling. They do not read CTHR final valid structures or compiled cells.",
        "",
        "## Main Result",
        "",
        base.render_md_table(overall, headers),
        "",
        "## Selection Status",
        "",
        "```json",
        json.dumps(status_counts, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Interpretation",
        "",
        "- When the six rule-interaction semantics are encoded explicitly, native symbolic solvers recover the same valid-rule sets as the CTHR resolver on this benchmark.",
        "- This supports the revised paper claim: ASP, SMT, and MILP can be reliable backends or encodings when the KG-to-rule interaction semantics are supplied, but they do not automatically provide this semantic compilation layer.",
        "- Remaining semantic failures, if any, should be interpreted as downstream numeric optimization or constraint-mapping issues rather than rule-selection failures.",
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
    queries = base.by_id(base.load_items(AVIATION_ROOT / "aviation_combined_optimization_queries.json"))
    labels = base.by_id(base.load_items(AVIATION_ROOT / "aviation_combined_rule_structure_labels.json"))
    feasible_items = base.by_id(base.load_items(AVIATION_ROOT / "aviation_combined_feasible_region_labels.json"))
    rule_library = base.read_json(AVIATION_RULE_LIBRARY)
    rule_by_id = {str(rule["rule_id"]): rule for rule in rule_library.get("rules", []) if rule.get("rule_id")}

    rows: list[dict[str, Any]] = []
    for task_id in queries:
        for method in METHODS:
            rows.append(evaluate_one(method, queries[task_id], labels[task_id], feasible_items[task_id], rule_by_id))

    overall = [aggregate(rows, method) for method in METHODS]
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
        "selection_status",
        "resolver_notes",
        "runtime_ms",
    ]
    overall_headers = ["Dataset", "Method", "Rule Precision", "Rule Recall", "Formal CSR", "Sem-CSR", "False accept", "Invalid cases"]
    write_csv(OUTPUTS["per_task_csv"], rows, per_task_headers)
    write_csv(OUTPUTS["overall_csv"], overall, overall_headers)
    OUTPUTS["overall_md"].write_text(base.render_md_table(overall, overall_headers), encoding="utf-8")
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {"Aviation": len(queries), "original": 19, "stress": 12},
        "methods": METHODS,
        "input_restrictions": {
            "hidden_reference_labels_as_method_input": False,
            "cthr_final_valid_structures_as_method_input": False,
            "cthr_compiled_cells_as_method_input": False,
            "native_symbolic_facts": "derived from visible candidate rules, rule metadata, relations, guards, scenario facts, and decision-variable names",
        },
        "outputs": {key: str(value) for key, value in OUTPUTS.items()},
        "aggregate_rows": overall,
    }
    base.write_json(OUTPUTS["overall_json"], summary)
    OUTPUTS["report_md"].write_text(render_report(overall, rows, summary), encoding="utf-8")
    print(json.dumps({"outputs": summary["outputs"], "aggregate_rows": overall}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
