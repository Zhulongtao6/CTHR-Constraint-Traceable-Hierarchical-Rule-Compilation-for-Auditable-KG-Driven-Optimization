from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

try:
    import clingo  # type: ignore
except ImportError as exc:  # pragma: no cover - exercised when clingo is absent.
    clingo = None
    CLINGO_IMPORT_ERROR = exc
else:
    CLINGO_IMPORT_ERROR = None


ASP_META_ENCODING = r"""
candidate(R) :- rule(R), applicable(R), not defeated(R).

defeated(Rb) :- overrides(Re,Rb), applicable(Re).
defeated(Rj) :- precedes(Ri,Rj), conflict(Ri,Rj), applicable(Ri), applicable(Rj).

{ selected(R) } :- candidate(R).

:- selected(R), not candidate(R).
:- selected(R), depends(R,D), not selected(D).
:- selected(R1), selected(R2), excludes(R1,R2).
:- selected(R1), selected(R2), excludes(R2,R1).
:- selected(R1), selected(R2), conflict(R1,R2), R1 != R2.

dep_closure(R,D) :- depends(R,D).
dep_closure(R,D2) :- depends(R,D), dep_closure(D,D2).

closure_member(R,R) :- candidate(R).
closure_member(R,D) :- candidate(R), dep_closure(R,D).

bad_closure(R) :- closure_member(R,M), not candidate(M).
bad_closure(R) :- closure_member(R,M), selected(S), excludes(M,S).
bad_closure(R) :- closure_member(R,M), selected(S), excludes(S,M).
bad_closure(R) :- closure_member(R,M), selected(S), conflict(M,S), M != S.
bad_closure(R) :- closure_member(R,A), closure_member(R,B), excludes(A,B), A != B.
bad_closure(R) :- closure_member(R,A), closure_member(R,B), conflict(A,B), A != B.

can_add(R) :- candidate(R), not selected(R), not bad_closure(R).
:- can_add(R).

#show selected/1.
"""


@dataclass(frozen=True)
class AspEnumerationResult:
    task_id: str
    number_of_answer_sets: int
    asp_rule_structures: list[list[str]]
    enumeration_time_ms: float
    status: str
    candidate_rule_count: int = 0
    selected_rule_count: int = 0
    candidate_rule_ids: list[str] | None = None
    error: str | None = None
    truncated: bool = False


@dataclass(frozen=True)
class GroundedCandidateSet:
    task_id: str
    candidate_rule_ids: list[str]
    applicable_rule_ids: list[str]
    grounded_rule_records: list[dict[str, Any]]
    relation_counts: dict[str, int]
    grounding_notes: list[str]


def ensure_clingo_available() -> None:
    if clingo is None:
        raise RuntimeError(
            "The ASP baseline requires the clingo Python package. "
            "Install it with `pip install clingo` or install the project "
            "dependencies from `requirements.txt`. No CTHR fallback is used."
        ) from CLINGO_IMPORT_ERROR


def normalize_token(value: Any) -> str:
    text = str(value).strip().lower()
    text = text.replace("±", "+/-")
    text = text.replace("°", "deg")
    text = text.replace("degree", "deg")
    return re.sub(r"[^a-z0-9.+/-]+", "", text)


def normalize_field_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "under",
    "with",
    "while",
    "must",
    "rule",
    "rules",
    "design",
    "procedure",
    "task",
    "select",
    "choose",
    "derived",
    "kg",
    "aviation",
    "continuous",
    "decision",
    "domain",
    "expression",
    "facts",
    "lambda",
    "metadata",
    "name",
    "objective",
    "objectives",
    "optimizer",
    "preference",
    "preferences",
    "query",
    "scenario",
    "source",
    "title",
    "type",
    "unit",
    "upper",
    "lower",
    "variable",
    "variables",
}


SYNONYMS = {
    "ils": {"instrument", "landing", "system"},
    "vor": {"vor"},
    "dme": {"dme"},
    "pbn": {"pbn", "rnav", "rnp"},
    "sbas": {"sbas", "fas"},
    "gbas": {"gbas"},
    "rf": {"rf", "radius", "fixed"},
    "moc": {"moc", "clearance", "obstacle"},
    "msa": {"msa", "sector", "arc"},
    "paoas": {"paoas", "oas"},
}


def tokens_from_value(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, dict):
        tokens: set[str] = set()
        for key, nested in value.items():
            tokens |= tokens_from_value(key)
            tokens |= tokens_from_value(nested)
        return tokens
    if isinstance(value, list):
        tokens: set[str] = set()
        for item in value:
            tokens |= tokens_from_value(item)
        return tokens
    text = str(value).lower().replace("±", " ")
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    parts = re.split(r"[^a-z0-9]+", text)
    tokens = {part for part in parts if len(part) >= 2 and part not in STOPWORDS}
    expanded = set(tokens)
    for token in tokens:
        expanded |= SYNONYMS.get(token, set())
    return expanded


def guard_terms(guard: Any) -> set[str]:
    if not guard:
        return set()
    if isinstance(guard, list):
        tokens: set[str] = set()
        for item in guard:
            tokens |= guard_terms(item)
        return tokens
    if not isinstance(guard, dict):
        return set()
    if "all" in guard:
        return guard_terms(guard.get("all", []))
    if "any" in guard:
        return guard_terms(guard.get("any", []))
    if "not" in guard:
        return guard_terms(guard.get("not"))
    return tokens_from_value(guard.get("field")) | tokens_from_value(guard.get("value"))


def guard_has_clauses(guard: Any) -> bool:
    if not guard:
        return False
    if isinstance(guard, list):
        return any(guard_has_clauses(item) for item in guard)
    if not isinstance(guard, dict):
        return False
    if "all" in guard:
        return any(guard_has_clauses(item) for item in guard.get("all", []))
    if "any" in guard:
        return any(guard_has_clauses(item) for item in guard.get("any", []))
    if "not" in guard:
        return guard_has_clauses(guard.get("not"))
    return "field" in guard


def scenario_lookup(scenario: dict[str, Any], field: str) -> Any:
    if field in scenario:
        return scenario[field]
    target = normalize_field_name(field)
    for key, value in scenario.items():
        key_norm = normalize_field_name(key)
        if key_norm == target or key_norm.startswith(target) or target.startswith(key_norm):
            return value
    raise KeyError(field)


def compare_values(actual: Any, op: str, expected: Any) -> bool:
    op = op.lower()
    if op in {"eq", "=", "=="}:
        return normalize_token(actual) == normalize_token(expected)
    if op in {"ne", "!=", "neq"}:
        return normalize_token(actual) != normalize_token(expected)
    if op in {"contains"}:
        return normalize_token(expected) in normalize_token(actual)
    if op in {"in"}:
        if isinstance(expected, list):
            return any(normalize_token(actual) == normalize_token(item) for item in expected)
        return normalize_token(actual) in normalize_token(expected)
    def numeric_value(value: Any) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip().lower()
        flight_level = re.search(r"\bfl\s*([0-9]+(?:\.[0-9]+)?)\b", text)
        if flight_level:
            return float(flight_level.group(1))
        number = re.search(r"[-+]?[0-9]+(?:\.[0-9]+)?", text)
        if number:
            return float(number.group(0))
        return float(value)

    try:
        actual_f = numeric_value(actual)
        expected_f = numeric_value(expected)
    except (TypeError, ValueError):
        return False
    if op in {"gt", ">"}:
        return actual_f > expected_f
    if op in {"ge", ">=", "gte"}:
        return actual_f >= expected_f
    if op in {"lt", "<"}:
        return actual_f < expected_f
    if op in {"le", "<=", "lte"}:
        return actual_f <= expected_f
    return False


def eval_guard(guard: Any, scenario: dict[str, Any]) -> bool:
    if not guard:
        return True
    if isinstance(guard, list):
        return all(eval_guard(item, scenario) for item in guard)
    if not isinstance(guard, dict):
        return False
    if "all" in guard:
        return all(eval_guard(item, scenario) for item in guard.get("all", []))
    if "any" in guard:
        return any(eval_guard(item, scenario) for item in guard.get("any", []))
    if "not" in guard:
        return not eval_guard(guard.get("not"), scenario)
    field = guard.get("field")
    op = guard.get("op", "eq")
    if field is None:
        return False
    try:
        actual = scenario_lookup(scenario, str(field))
    except KeyError:
        return False
    return compare_values(actual, str(op), guard.get("value"))


def asp_string(value: str) -> str:
    return json.dumps(str(value), ensure_ascii=True)


def fact(name: str, *args: str) -> str:
    return f"{name}({','.join(asp_string(arg) for arg in args)})."


def relation_target(relation: dict[str, Any]) -> str | None:
    target = relation.get("target") or relation.get("to") or relation.get("rule_id")
    return str(target) if target is not None else None


def relation_type(relation: dict[str, Any]) -> str:
    return str(relation.get("type", "")).lower().strip()


def task_tokens(task: dict[str, Any]) -> set[str]:
    tokens = set()
    tokens |= tokens_from_value(task.get("title"))
    tokens |= tokens_from_value(task.get("task_type"))
    tokens |= tokens_from_value(task.get("design_intent"))
    for key, value in task.get("scenario_facts", {}).items():
        tokens |= tokens_from_value(key)
        tokens |= tokens_from_value(value)
    for name, spec in task.get("decision_variables", {}).items():
        tokens |= tokens_from_value(name)
        tokens |= tokens_from_value(spec.get("unit"))
    for objective in task.get("objectives", []):
        tokens |= tokens_from_value(objective.get("name"))
        tokens |= tokens_from_value(objective.get("expression"))
    tokens |= tokens_from_value(task.get("query_preferences", {}).get("meaning"))
    return tokens


def task_units(task: dict[str, Any]) -> set[str]:
    units = set()
    for spec in task.get("decision_variables", {}).values():
        unit = spec.get("unit")
        if unit:
            units.add(normalize_token(unit))
    return units


def rule_tokens(rule: dict[str, Any]) -> set[str]:
    tokens = set()
    tokens |= tokens_from_value(rule.get("rule_id"))
    tokens |= tokens_from_value(rule.get("name"))
    tokens |= tokens_from_value(rule.get("rule_type"))
    tokens |= guard_terms(rule.get("guard"))
    for constraint in rule.get("constraints", []):
        tokens |= tokens_from_value(constraint.get("variable"))
        tokens |= tokens_from_value(constraint.get("unit"))
        tokens |= tokens_from_value(constraint.get("value"))
    for relation in rule.get("relations", []):
        tokens |= tokens_from_value(relation.get("target"))
    for prov in rule.get("provenance", []):
        tokens |= tokens_from_value(prov.get("section"))
    return tokens


def rule_units(rule: dict[str, Any]) -> set[str]:
    units = set()
    for constraint in rule.get("constraints", []):
        unit = constraint.get("unit")
        if unit:
            units.add(normalize_token(unit))
    return units


def candidate_score(rule: dict[str, Any], task: dict[str, Any], scenario: dict[str, Any]) -> float:
    task_tok = task_tokens(task)
    rule_tok = rule_tokens(rule)
    overlap = task_tok & rule_tok
    score = float(len(overlap))
    guard_tok = guard_terms(rule.get("guard"))
    score += 1.5 * len(task_tok & guard_tok)
    if guard_has_clauses(rule.get("guard")) and eval_guard(rule.get("guard"), scenario):
        score += 3.0
    units_overlap = task_units(task) & rule_units(rule)
    if units_overlap:
        score += 1.0
    return score


def retrieve_candidate_rules(
    rule_library: dict[str, Any],
    task: dict[str, Any],
    min_score: float = 2.0,
    closure_rounds: int = 3,
) -> list[str]:
    """Retrieve task-level candidate rules without using hidden labels."""
    rules = rule_library.get("rules", [])
    by_id = {str(rule["rule_id"]): rule for rule in rules if rule.get("rule_id")}
    scenario = task.get("scenario_facts", {})

    candidates = {
        rule_id
        for rule_id, rule in by_id.items()
        if candidate_score(rule, task, scenario) >= min_score
    }

    # Relation closure is part of grounding: if a retrieved rule refers to a
    # dependency, exclusion, override, or precedence target, include the target so
    # ASP can reason over the relation instead of silently dropping it.
    for _ in range(closure_rounds):
        before = set(candidates)
        for rule_id in list(before):
            rule = by_id.get(rule_id)
            if not rule:
                continue
            for relation in rule.get("relations", []):
                target = relation_target(relation)
                if target in by_id:
                    candidates.add(target)
        for rule_id, rule in by_id.items():
            if rule_id in candidates:
                continue
            for relation in rule.get("relations", []):
                target = relation_target(relation)
                rel_type = str(relation.get("type", "")).lower()
                if target in before and rel_type in {
                    "excludes",
                    "mutually_exclusive",
                    "conflicts_with",
                    "conflict",
                    "overrides",
                    "can_override",
                    "replaces",
                    "defeats",
                    "precedes",
                    "precedence",
                    "higher_priority_than",
                    "has_precedence_over",
                }:
                    candidates.add(rule_id)
        if candidates == before:
            break

    return sorted(candidates)


def _known_rule_ids(rule_library: dict[str, Any]) -> set[str]:
    return {str(rule["rule_id"]) for rule in rule_library.get("rules", []) if rule.get("rule_id")}


def _task_visible_rule_refs(task: dict[str, Any], known_rule_ids: set[str]) -> tuple[set[str], set[str], set[str]]:
    """Extract rule IDs from visible task artifacts without reading evaluation labels."""
    candidate_refs: set[str] = set()
    applicable_refs: set[str] = set()
    defeated_refs: set[str] = set()

    def maybe_add(value: Any, target: set[str]) -> None:
        if value is None:
            return
        text = str(value)
        if text in known_rule_ids:
            target.add(text)

    for constraint in task.get("solver_constraints", []):
        if constraint.get("source_type") == "rule_library":
            maybe_add(constraint.get("source_id"), candidate_refs)
            maybe_add(constraint.get("source_id"), applicable_refs)

    for cell in task.get("solver_constraint_cells", []):
        for constraint in cell.get("executable_constraints", []):
            if constraint.get("source_type") == "rule_library":
                maybe_add(constraint.get("source_id"), candidate_refs)
                maybe_add(constraint.get("source_id"), applicable_refs)

    for check in task.get("pre_solver_structure_checks", []):
        if check.get("source_type") == "rule_library":
            maybe_add(check.get("source_id"), candidate_refs)
            if "defeated" in str(check.get("role", "")).lower():
                maybe_add(check.get("source_id"), defeated_refs)

    for rule_id in task.get("certificate_targets", {}).get("source_rule_ids", []):
        maybe_add(rule_id, candidate_refs)
        if rule_id not in defeated_refs:
            maybe_add(rule_id, applicable_refs)

    return candidate_refs, applicable_refs, defeated_refs


def _relation_closure(rule_library: dict[str, Any], seed_ids: set[str], rounds: int = 3) -> set[str]:
    by_id = {str(rule["rule_id"]): rule for rule in rule_library.get("rules", []) if rule.get("rule_id")}
    candidates = set(seed_ids)
    relation_types = {
        "depends_on",
        "requires",
        "uses_parameter",
        "excludes",
        "mutually_exclusive",
        "conflicts_with",
        "conflict",
        "overrides",
        "can_override",
        "replaces",
        "defeats",
        "precedes",
        "precedence",
        "higher_priority_than",
        "has_precedence_over",
    }
    for _ in range(rounds):
        before = set(candidates)
        for rule_id in list(before):
            rule = by_id.get(rule_id)
            if not rule:
                continue
            for relation in rule.get("relations", []):
                target = relation_target(relation)
                if target in by_id:
                    candidates.add(target)
        for rule_id, rule in by_id.items():
            if rule_id in candidates:
                continue
            for relation in rule.get("relations", []):
                target = relation_target(relation)
                if target in before and relation_type(relation) in relation_types:
                    candidates.add(rule_id)
        if candidates == before:
            break
    return candidates


def get_cthr_grounded_candidates(
    rule_library: dict[str, Any],
    task: dict[str, Any],
    min_score: float = 2.0,
    include_heuristic_candidates: bool = False,
) -> GroundedCandidateSet:
    """Expose the CTHR-style grounding stage before defeasible resolution.

    This uses visible query/scenario artifacts and rule guards to form the
    candidate/applicable set. It does not read reference labels, valid chains,
    or hidden expected structures.
    """
    task_id = str(task.get("omega_id", "unknown_task"))
    scenario = dict(task.get("scenario_facts", {}))
    scenario.update(
        {
            "domain": task.get("domain"),
            "task_type": task.get("task_type"),
            "title": task.get("title"),
        }
    )
    known_rule_ids = _known_rule_ids(rule_library)
    visible_refs, visible_applicable, defeated_refs = _task_visible_rule_refs(task, known_rule_ids)
    heuristic_refs = set(retrieve_candidate_rules(rule_library, task, min_score=min_score))
    seed_refs = set(visible_refs)
    if include_heuristic_candidates or not seed_refs:
        seed_refs |= heuristic_refs
    candidate_ids = _relation_closure(rule_library, seed_refs)

    by_id = {str(rule["rule_id"]): rule for rule in rule_library.get("rules", []) if rule.get("rule_id")}
    guard_applicable = {
        rule_id
        for rule_id in candidate_ids
        if rule_id in by_id and eval_guard(by_id[rule_id].get("guard"), scenario)
    }
    applicable_ids = (visible_applicable | guard_applicable) & candidate_ids
    applicable_ids -= defeated_refs

    relation_counts: dict[str, int] = {}
    grounded_records = []
    for rule_id in sorted(candidate_ids):
        rule = by_id.get(rule_id, {})
        for relation in rule.get("relations", []):
            target = relation_target(relation)
            if target in candidate_ids:
                rel_type = relation_type(relation)
                relation_counts[rel_type] = relation_counts.get(rel_type, 0) + 1
        grounded_records.append(
            {
                "rule_id": rule_id,
                "applicable": rule_id in applicable_ids,
                "guard_applicable": rule_id in guard_applicable,
                "visible_task_reference": rule_id in visible_refs,
                "visible_applicable_reference": rule_id in visible_applicable,
                "visible_defeated_reference": rule_id in defeated_refs,
            }
        )

    notes = [
        f"visible_refs={len(visible_refs)}",
        f"heuristic_refs={len(heuristic_refs)}",
        f"heuristic_included={include_heuristic_candidates or not visible_refs}",
        f"guard_applicable={len(guard_applicable)}",
        f"visible_defeated_refs={len(defeated_refs)}",
    ]
    return GroundedCandidateSet(
        task_id=task_id,
        candidate_rule_ids=sorted(candidate_ids),
        applicable_rule_ids=sorted(applicable_ids),
        grounded_rule_records=grounded_records,
        relation_counts=relation_counts,
        grounding_notes=notes,
    )


def build_asp_facts(
    rule_library: dict[str, Any],
    scenario: dict[str, Any],
    candidate_rule_ids: list[str] | None = None,
    applicable_rule_ids: list[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    rules = rule_library.get("rules", [])
    allowed = set(candidate_rule_ids) if candidate_rule_ids is not None else None
    rules = [rule for rule in rules if rule.get("rule_id") and (allowed is None or str(rule["rule_id"]) in allowed)]
    rule_ids = {str(rule["rule_id"]) for rule in rules if rule.get("rule_id")}
    forced_applicable = set(applicable_rule_ids) if applicable_rule_ids is not None else None
    facts: list[str] = []
    conflict_classes: dict[str, list[str]] = {}
    provenance_map: dict[str, Any] = {}

    for rule in rules:
        rule_id = str(rule.get("rule_id", "")).strip()
        if not rule_id:
            continue
        facts.append(fact("rule", rule_id))
        provenance_map[rule_id] = {
            "source_chunk_ids": rule.get("source_chunk_ids", []),
            "source_node_ids": rule.get("source_node_ids", []),
            "provenance": rule.get("provenance", []),
        }
        if (forced_applicable is not None and rule_id in forced_applicable) or (
            forced_applicable is None and eval_guard(rule.get("guard"), scenario)
        ):
            facts.append(fact("applicable", rule_id))

        cls = rule.get("conflict_class") or rule.get("conflict_group")
        if cls:
            conflict_classes.setdefault(str(cls), []).append(rule_id)

    dependency_pairs: set[tuple[str, str]] = set()
    for rule in rules:
        rule_id = str(rule.get("rule_id", "")).strip()
        if not rule_id:
            continue
        for relation in rule.get("relations", []):
            target = relation_target(relation)
            if target in rule_ids and relation_type(relation) in {"depends_on", "requires", "uses_parameter"}:
                dependency_pairs.add((rule_id, target))

    for rule in rules:
        rule_id = str(rule.get("rule_id", "")).strip()
        if not rule_id:
            continue
        rule_type = str(rule.get("rule_type", "")).lower()
        for relation in rule.get("relations", []):
            target = relation_target(relation)
            if not target or target not in rule_ids:
                continue
            rel_type = str(relation.get("type", "")).lower()
            if rel_type in {"depends_on", "requires", "uses_parameter"}:
                facts.append(fact("depends", rule_id, target))
            elif rel_type in {"excludes", "mutually_exclusive", "conflicts_with", "conflict"}:
                if rule_type == "exception":
                    facts.append(fact("overrides", rule_id, target))
                elif (rule_id, target) in dependency_pairs or (target, rule_id) in dependency_pairs:
                    continue
                else:
                    facts.append(fact("excludes", rule_id, target))
                    facts.append(fact("excludes", target, rule_id))
            elif rel_type in {"overrides", "can_override", "replaces", "defeats"}:
                facts.append(fact("overrides", rule_id, target))
            elif rel_type in {"precedes", "precedence", "higher_priority_than", "has_precedence_over"}:
                facts.append(fact("precedes", rule_id, target))

    for members in conflict_classes.values():
        for left in members:
            for right in members:
                if left != right:
                    facts.append(fact("conflict", left, right))

    return "\n".join(sorted(set(facts))) + "\n", {"provenance_map": provenance_map}


def enumerate_rule_structures(
    rule_library: dict[str, Any],
    scenario: dict[str, Any],
    task_id: str,
    candidate_rule_ids: list[str] | None = None,
    applicable_rule_ids: list[str] | None = None,
    max_answer_sets: int = 200,
) -> AspEnumerationResult:
    start = time.perf_counter()
    candidate_ids = sorted(candidate_rule_ids) if candidate_rule_ids is not None else sorted(
        str(rule["rule_id"]) for rule in rule_library.get("rules", []) if rule.get("rule_id")
    )
    try:
        ensure_clingo_available()
        facts, _metadata = build_asp_facts(
            rule_library,
            scenario,
            candidate_rule_ids=candidate_ids,
            applicable_rule_ids=applicable_rule_ids,
        )
        control = clingo.Control(["--warn=none"])  # type: ignore[union-attr]
        control.configuration.solve.models = 0
        control.add("base", [], facts + "\n" + ASP_META_ENCODING)
        control.ground([("base", [])])

        structures: list[list[str]] = []
        truncated = False
        with control.solve(yield_=True) as handle:
            for model in handle:
                selected = sorted(
                    symbol.arguments[0].string
                    for symbol in model.symbols(shown=True)
                    if symbol.name == "selected" and symbol.arguments
                )
                structures.append(selected)
                if len(structures) >= max_answer_sets:
                    truncated = True
                    handle.cancel()
                    break
            solve_result = handle.get()

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if not solve_result.satisfiable:
            return AspEnumerationResult(
                task_id,
                0,
                [],
                elapsed_ms,
                "unsat",
                candidate_rule_count=len(candidate_ids),
                selected_rule_count=0,
                candidate_rule_ids=candidate_ids,
                truncated=truncated,
            )

        unique = sorted({tuple(structure) for structure in structures})
        selected_rule_ids = {rule_id for structure in unique for rule_id in structure}
        return AspEnumerationResult(
            task_id=task_id,
            number_of_answer_sets=len(unique),
            asp_rule_structures=[list(item) for item in unique],
            enumeration_time_ms=elapsed_ms,
            status="success",
            candidate_rule_count=len(candidate_ids),
            selected_rule_count=len(selected_rule_ids),
            candidate_rule_ids=candidate_ids,
            truncated=truncated,
        )
    except Exception as exc:  # noqa: BLE001 - errors are reported per task.
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return AspEnumerationResult(
            task_id=task_id,
            number_of_answer_sets=0,
            asp_rule_structures=[],
            enumeration_time_ms=elapsed_ms,
            status="error",
            candidate_rule_count=len(candidate_ids),
            selected_rule_count=0,
            candidate_rule_ids=candidate_ids,
            error=str(exc),
        )


def reference_structures_from_label(label: dict[str, Any]) -> list[list[str]]:
    if label.get("expected_valid_rule_structures"):
        return [sorted(map(str, item)) for item in label["expected_valid_rule_structures"]]
    return [sorted(map(str, label.get("expected_surviving_rule_ids", [])))]


def normalize_structures(structures: list[list[str]]) -> set[tuple[str, ...]]:
    return {tuple(sorted(map(str, structure))) for structure in structures}


def evaluate_structures(
    predicted_structures: list[list[str]],
    reference_structures: list[list[str]],
) -> dict[str, Any]:
    predicted = normalize_structures(predicted_structures)
    reference = normalize_structures(reference_structures)
    predicted_rules = {rule_id for structure in predicted for rule_id in structure}
    reference_rules = {rule_id for structure in reference for rule_id in structure}
    matched_rules = predicted_rules & reference_rules
    precision = len(matched_rules) / len(predicted_rules) if predicted_rules else 0.0
    recall = len(matched_rules) / len(reference_rules) if reference_rules else 0.0
    matched_structures = predicted & reference
    structure_precision = len(matched_structures) / len(predicted) if predicted else 0.0
    structure_recall = len(matched_structures) / len(reference) if reference else 0.0
    accuracy = 1.0 if predicted == reference else 0.0
    return {
        "valid_structure_accuracy": accuracy,
        "rule_structure_precision": precision,
        "rule_structure_recall": recall,
        "structure_precision": structure_precision,
        "structure_recall": structure_recall,
        "missing_rules": sorted(reference_rules - predicted_rules),
        "extra_rules": sorted(predicted_rules - reference_rules),
        "missing_structures": [list(item) for item in sorted(reference - predicted)],
        "extra_structures": [list(item) for item in sorted(predicted - reference)],
    }
