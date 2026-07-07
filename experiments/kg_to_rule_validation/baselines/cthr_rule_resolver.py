from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .asp_rule_structure import (
    compare_values,
    guard_has_clauses,
    normalize_field_name,
    normalize_token,
    relation_target,
    relation_type,
    rule_tokens,
    scenario_lookup,
    tokens_from_value,
)


DEPENDENCY_TYPES = {"depends_on", "requires", "uses_parameter"}
EXCLUSION_TYPES = {"excludes", "mutually_exclusive", "conflicts_with", "conflict"}
OVERRIDE_TYPES = {"overrides", "can_override", "replaces", "defeats"}
PRECEDENCE_TYPES = {"precedes", "precedence", "higher_priority_than", "has_precedence_over"}


@dataclass(frozen=True)
class CthrResolverResult:
    valid_rule_structures: list[list[str]]
    resolver_time_ms: float
    status: str
    candidate_rule_count: int
    predicted_rule_count: int
    applicable_rule_ids: list[str]
    defeated_rule_ids: list[str]
    removed_rule_ids: list[str]
    notes: list[str]
    error: str | None = None


def scenario_tokens(scenario: dict[str, Any]) -> set[str]:
    return tokens_from_value(scenario)


def rule_relevance(rule: dict[str, Any], scenario: dict[str, Any]) -> float:
    st = scenario_tokens(scenario)
    rt = rule_tokens(rule)
    score = float(len(st & rt))
    for constraint in rule.get("constraints", []):
        variable = constraint.get("variable")
        if variable and normalize_field_name(str(variable)) in {
            normalize_field_name(str(v)) for v in scenario.get("decision_variable_names", [])
        }:
            score += 2.0
    return score


def fuzzy_compare(actual: Any, op: str, expected: Any) -> bool:
    op = str(op).lower()
    if op in {"eq", "=", "=="}:
        actual_s = normalize_token(actual)
        expected_s = normalize_token(expected)
        if actual_s == expected_s:
            return True
        if len(actual_s) >= 4 and len(expected_s) >= 4:
            return actual_s in expected_s or expected_s in actual_s
        return False
    if op in {"in"} and isinstance(expected, list):
        return any(fuzzy_compare(actual, "eq", item) for item in expected)
    return compare_values(actual, op, expected)


def eval_guard_three_valued(guard: Any, scenario: dict[str, Any]) -> str:
    """Return true, false, or unknown for partially grounded rule guards."""
    if not guard:
        return "true"
    if isinstance(guard, list):
        values = [eval_guard_three_valued(item, scenario) for item in guard]
        if any(value == "false" for value in values):
            return "false"
        if any(value == "unknown" for value in values):
            return "unknown"
        return "true"
    if not isinstance(guard, dict):
        return "false"
    if "all" in guard:
        return eval_guard_three_valued(guard.get("all", []), scenario)
    if "any" in guard:
        values = [eval_guard_three_valued(item, scenario) for item in guard.get("any", [])]
        if any(value == "true" for value in values):
            return "true"
        if any(value == "unknown" for value in values):
            return "unknown"
        return "false"
    if "not" in guard:
        value = eval_guard_three_valued(guard.get("not"), scenario)
        if value == "true":
            return "false"
        if value == "false":
            return "true"
        return "unknown"
    field = guard.get("field")
    if field is None:
        return "unknown"
    try:
        actual = scenario_lookup(scenario, str(field))
    except KeyError:
        return "unknown"
    return "true" if fuzzy_compare(actual, guard.get("op", "eq"), guard.get("value")) else "false"


def rule_specificity(rule: dict[str, Any], scenario: dict[str, Any]) -> float:
    guard = rule.get("guard")
    guard_status = eval_guard_three_valued(guard, scenario)
    score = rule_relevance(rule, scenario)
    if guard_status == "true":
        score += 5.0
    elif guard_status == "unknown":
        score += 1.0
    if str(rule.get("rule_type", "")).lower() in {"exception", "precedence"}:
        score += 3.0
    if guard_has_clauses(guard):
        score += 0.5
    return score


def is_initially_applicable(rule: dict[str, Any], scenario: dict[str, Any]) -> bool:
    status = eval_guard_three_valued(rule.get("guard"), scenario)
    relevance = rule_relevance(rule, scenario)
    if status == "true":
        return True
    if status == "unknown" and relevance >= 2.0:
        return True
    if status == "false" and relevance >= 4.0:
        return True
    if not guard_has_clauses(rule.get("guard")) and relevance >= 1.0:
        return True
    return False


def relation_maps(candidate_rules: list[dict[str, Any]]) -> dict[str, set[tuple[str, str]]]:
    ids = {str(rule["rule_id"]) for rule in candidate_rules if rule.get("rule_id")}
    maps = {"depends": set(), "excludes": set(), "overrides": set(), "precedes": set(), "conflicts": set()}
    classes: dict[str, list[str]] = {}
    for rule in candidate_rules:
        rid = str(rule.get("rule_id"))
        cls = rule.get("conflict_class") or rule.get("conflict_group")
        if cls:
            classes.setdefault(str(cls), []).append(rid)
        for relation in rule.get("relations", []):
            target = relation_target(relation)
            if target not in ids:
                continue
            rt = relation_type(relation)
            pair = (rid, str(target))
            if rt in DEPENDENCY_TYPES:
                maps["depends"].add(pair)
            elif rt in EXCLUSION_TYPES:
                if str(rule.get("rule_type", "")).lower() == "exception":
                    maps["overrides"].add(pair)
                else:
                    maps["excludes"].add(pair)
                    maps["excludes"].add((str(target), rid))
            elif rt in OVERRIDE_TYPES:
                maps["overrides"].add(pair)
            elif rt in PRECEDENCE_TYPES:
                maps["precedes"].add(pair)
    for members in classes.values():
        for left in members:
            for right in members:
                if left != right:
                    maps["conflicts"].add((left, right))
    dependency_pairs = maps["depends"]
    maps["excludes"] = {
        pair
        for pair in maps["excludes"]
        if pair not in dependency_pairs and (pair[1], pair[0]) not in dependency_pairs
    }
    return maps


def dependency_closure(selected: set[str], depends: set[tuple[str, str]], available: set[str]) -> set[str]:
    closed = set(selected)
    changed = True
    while changed:
        changed = False
        for left, right in depends:
            if left in closed and right in available and right not in closed:
                closed.add(right)
                changed = True
    return closed


def remove_rules_with_missing_dependencies(
    selected: set[str],
    depends: set[tuple[str, str]],
    available: set[str],
) -> set[str]:
    stable = set(selected)
    changed = True
    while changed:
        changed = False
        for left, right in depends:
            if left in stable and right not in available:
                stable.remove(left)
                changed = True
    return stable


def resolve_conflicts(
    selected: set[str],
    maps: dict[str, set[tuple[str, str]]],
    by_id: dict[str, dict[str, Any]],
    scenario: dict[str, Any],
) -> tuple[set[str], list[str]]:
    notes: list[str] = []
    chosen = set(selected)
    conflict_pairs = set(maps["excludes"]) | set(maps["conflicts"])
    changed = True
    while changed:
        changed = False
        for left, right in sorted(conflict_pairs):
            if left not in chosen or right not in chosen:
                continue
            left_score = rule_specificity(by_id[left], scenario)
            right_score = rule_specificity(by_id[right], scenario)
            loser = right if left_score >= right_score else left
            chosen.remove(loser)
            notes.append(f"removed_conflict_loser:{loser}")
            changed = True
            break
    return chosen, notes


def resolve_valid_structures_with_diagnostics(
    candidate_rules: list[dict[str, Any]],
    scenario: dict[str, Any],
) -> CthrResolverResult:
    start = time.perf_counter()
    try:
        by_id = {str(rule["rule_id"]): rule for rule in candidate_rules if rule.get("rule_id")}
        maps = relation_maps(candidate_rules)

        applicable = {rid for rid, rule in by_id.items() if is_initially_applicable(rule, scenario)}
        defeated: set[str] = set()
        for source, target in maps["overrides"]:
            if source in applicable:
                defeated.add(target)
        for source, target in maps["precedes"]:
            if source in applicable and target in applicable:
                defeated.add(target)

        available = applicable - defeated
        available = remove_rules_with_missing_dependencies(available, maps["depends"], set(by_id) - defeated)
        selected = dependency_closure(available, maps["depends"], set(by_id) - defeated)
        selected -= defeated
        selected, conflict_notes = resolve_conflicts(selected, maps, by_id, scenario)
        selected = dependency_closure(selected, maps["depends"], set(by_id) - defeated)
        selected -= defeated

        structures = [sorted(selected)] if selected else []
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return CthrResolverResult(
            valid_rule_structures=structures,
            resolver_time_ms=elapsed_ms,
            status="success",
            candidate_rule_count=len(candidate_rules),
            predicted_rule_count=len({rid for structure in structures for rid in structure}),
            applicable_rule_ids=sorted(applicable),
            defeated_rule_ids=sorted(defeated),
            removed_rule_ids=sorted(set(by_id) - selected),
            notes=conflict_notes,
        )
    except Exception as exc:  # noqa: BLE001 - report per-task resolver errors.
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return CthrResolverResult(
            valid_rule_structures=[],
            resolver_time_ms=elapsed_ms,
            status="error",
            candidate_rule_count=len(candidate_rules),
            predicted_rule_count=0,
            applicable_rule_ids=[],
            defeated_rule_ids=[],
            removed_rule_ids=[],
            notes=[],
            error=str(exc),
        )


def resolve_valid_structures_from_candidates(
    candidate_rules: list[dict[str, Any]],
    scenario: dict[str, Any],
) -> list[list[str]]:
    return resolve_valid_structures_with_diagnostics(candidate_rules, scenario).valid_rule_structures
