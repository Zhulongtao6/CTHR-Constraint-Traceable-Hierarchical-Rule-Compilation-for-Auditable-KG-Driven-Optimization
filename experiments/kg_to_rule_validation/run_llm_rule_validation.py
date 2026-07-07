from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
CTHR_ROOT = THIS_DIR.parents[1]
PAPER_DIR = CTHR_ROOT / "paper"
LOG_DIR = PAPER_DIR / "experiment_logs"

PROMPT_PATH = THIS_DIR / "kg_to_rule_prompt.txt"
CASES_PATH = THIS_DIR / "validation_cases.json"
OUTPUT_PATH = PAPER_DIR / "kg_to_rule_llm_outputs.json"
RESULT_PATH = PAPER_DIR / "kg_to_rule_validation_results.json"
LOG_PATH = LOG_DIR / "kg_to_rule_validation.log"

ALLOWED_CONSTRAINT_OPS = {">=", "<=", "=", ">", "<"}
ALLOWED_GUARD_OPS = {"eq", "neq", "gt", "gte", "lt", "lte", "in"}
ALLOWED_RELATIONS = {"overrides", "precedes", "excludes", "depends_on"}


@dataclass(frozen=True)
class Provider:
    name: str
    env_key: str
    url: str
    model: str


PROVIDERS = {
    "qwen": Provider(
        name="qwen",
        env_key="QWEN_API_KEY",
        url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        model="qwen-plus",
    ),
    "deepseek": Provider(
        name="deepseek",
        env_key="DEEPSEEK_API_KEY",
        url="https://api.deepseek.com/chat/completions",
        model="deepseek-chat",
    ),
    "glm": Provider(
        name="glm",
        env_key="GLM_API_KEY",
        url="https://open.bigmodel.cn/api/paas/v4/chat/completions",
        model="glm-4-flash",
    ),
}


def log(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def compact_case_for_prompt(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "domain": case["domain"],
        "interaction_type": case["interaction_type"],
        "allowed_variables": case["allowed_variables"],
        "scenario_omega": case["scenario_omega"],
        "input_evidence": case["input_evidence"],
    }


def provider_order(name: str) -> list[Provider]:
    if name == "auto":
        return [PROVIDERS["qwen"], PROVIDERS["deepseek"], PROVIDERS["glm"]]
    return [PROVIDERS[name]]


def call_provider(
    provider: Provider,
    system_prompt: str,
    user_payload: dict[str, Any],
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> tuple[str, dict[str, Any]]:
    api_key = os.environ.get(provider.env_key)
    if not api_key:
        raise RuntimeError(f"Missing environment variable {provider.env_key}")

    body = {
        "model": provider.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Convert this KG evidence packet into the requested JSON rule base.\n\n"
                    + json.dumps(user_payload, ensure_ascii=False, indent=2)
                ),
            },
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        provider.url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{provider.name} HTTP {exc.code}: {err_body[:500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{provider.name} URL error: {exc}") from exc

    parsed = json.loads(raw)
    content = parsed["choices"][0]["message"]["content"]
    meta = {
        "provider": provider.name,
        "model": provider.model,
        "usage": parsed.get("usage", {}),
        "finish_reason": parsed["choices"][0].get("finish_reason"),
    }
    return content, meta


def extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    if start < 0:
        raise ValueError("No JSON object found in model output")
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(cleaned)):
        char = cleaned[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
        else:
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(cleaned[start : idx + 1])
    raise ValueError("Could not find a balanced JSON object")


def edge_map(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {edge["id"]: edge for edge in case["input_evidence"]["kg_edges"]}


def node_map(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {node["id"]: node for node in case["input_evidence"]["kg_nodes"]}


def chunk_map(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {chunk["chunk_id"]: chunk for chunk in case["input_evidence"]["source_chunks"]}


def normalize_relation(rel: str) -> str:
    rel = str(rel or "").strip()
    aliases = {
        "override": "overrides",
        "overridden_by": "overrides",
        "precedence": "precedes",
        "has_precedence_over": "precedes",
        "exclude": "excludes",
        "mutually_exclusive": "excludes",
        "dependency": "depends_on",
        "requires": "depends_on",
    }
    return aliases.get(rel, rel)


def normalize_op(op: str) -> str:
    op = str(op or "").strip()
    aliases = {
        "min": ">=",
        "minimum": ">=",
        "at_least": ">=",
        "max": "<=",
        "maximum": "<=",
        "at_most": "<=",
        "==": "=",
    }
    return aliases.get(op, op)


def evidence_ids(obj: dict[str, Any], key: str) -> list[str]:
    ev = obj.get("evidence", {})
    ids = ev.get(key, [])
    if isinstance(ids, str):
        return [ids]
    if isinstance(ids, list):
        return [str(item) for item in ids]
    return []


def approx_equal(a: Any, b: Any, tol: float = 1e-6) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return str(a).strip() == str(b).strip()


def relation_to_constraint_op(relation: str) -> str | None:
    if relation == "requires_minimum":
        return ">="
    if relation == "requires_maximum":
        return "<="
    return None


def variable_name_from_edge(edge: dict[str, Any], nodes: dict[str, dict[str, Any]]) -> str | None:
    target = edge.get("target")
    node = nodes.get(target)
    if node:
        return node.get("name")
    return None


def validate_schema(rule_base: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(rule_base, dict):
        return {"pass": False, "errors": ["rule_base is not an object"]}
    if rule_base.get("case_id") != case["case_id"]:
        errors.append("case_id mismatch")
    rules = rule_base.get("rules")
    if not isinstance(rules, list) or not rules:
        errors.append("rules must be a non-empty list")
    else:
        seen_ids = set()
        for idx, rule in enumerate(rules):
            prefix = f"rules[{idx}]"
            if not isinstance(rule, dict):
                errors.append(f"{prefix} is not an object")
                continue
            rule_id = rule.get("rule_id")
            if not rule_id or not isinstance(rule_id, str):
                errors.append(f"{prefix}.rule_id missing")
            elif rule_id in seen_ids:
                errors.append(f"{prefix}.rule_id duplicated")
            seen_ids.add(rule_id)
            if rule.get("domain") != case["domain"]:
                errors.append(f"{prefix}.domain mismatch")
            if not isinstance(rule.get("source_node_ids", []), list):
                errors.append(f"{prefix}.source_node_ids must be a list")
            if not isinstance(rule.get("constraints", []), list):
                errors.append(f"{prefix}.constraints must be a list")
            if not isinstance(rule.get("relations", []), list):
                errors.append(f"{prefix}.relations must be a list")
            if not isinstance(rule.get("provenance", []), list):
                errors.append(f"{prefix}.provenance must be a list")
    return {"pass": len(errors) == 0, "errors": errors}


def validate_guard_schema(rule: dict[str, Any]) -> bool:
    guard = rule.get("guard")
    if guard in (None, {}, []):
        return True
    clauses = guard.get("all", []) if isinstance(guard, dict) else []
    if not isinstance(clauses, list):
        return False
    for clause in clauses:
        if not isinstance(clause, dict):
            return False
        if clause.get("op") not in ALLOWED_GUARD_OPS:
            return False
        if "field" not in clause or "value" not in clause:
            return False
    return True


def validate_provenance(rule: dict[str, Any], chunks: dict[str, dict[str, Any]]) -> tuple[int, int]:
    valid = 0
    total = 0
    for prov in rule.get("provenance", []):
        if not isinstance(prov, dict):
            continue
        total += 1
        chunk = chunks.get(str(prov.get("chunk_id")))
        if not chunk:
            continue
        doc_ok = not prov.get("document") or prov.get("document") == chunk.get("document")
        section_ok = not prov.get("section") or prov.get("section") == chunk.get("section")
        page_ok = prov.get("page") in (None, "", chunk.get("page"))
        if doc_ok and section_ok and page_ok:
            valid += 1
    return valid, total


def validate_constraints(rule: dict[str, Any], case: dict[str, Any]) -> tuple[int, int, list[str]]:
    edges = edge_map(case)
    nodes = node_map(case)
    chunks = chunk_map(case)
    allowed_variables = set(case["allowed_variables"])
    valid = 0
    total = 0
    errors: list[str] = []
    for con in rule.get("constraints", []):
        if not isinstance(con, dict):
            continue
        total += 1
        variable = con.get("variable")
        op = normalize_op(con.get("op"))
        value = con.get("value")
        unit = con.get("unit")
        if variable not in allowed_variables:
            errors.append(f"constraint variable not allowed: {variable}")
            continue
        if op not in ALLOWED_CONSTRAINT_OPS:
            errors.append(f"constraint op not allowed: {op}")
            continue
        edge_ids = evidence_ids(con, "kg_edge_ids")
        chunk_ids = evidence_ids(con, "chunk_ids")
        edge_ok = False
        for edge_id in edge_ids:
            edge = edges.get(edge_id)
            if not edge:
                continue
            expected_op = relation_to_constraint_op(edge.get("relation"))
            edge_var = variable_name_from_edge(edge, nodes)
            if (
                expected_op == op
                and edge_var == variable
                and approx_equal(edge.get("value"), value)
                and (not unit or edge.get("unit") == unit)
            ):
                edge_ok = True
                break
        chunk_ok = all(chunk_id in chunks for chunk_id in chunk_ids) if chunk_ids else False
        if edge_ok and chunk_ok:
            valid += 1
        else:
            errors.append(f"constraint not grounded: {variable} {op} {value}")
    return valid, total, errors


def validate_relations(rule: dict[str, Any], case: dict[str, Any]) -> tuple[int, int, list[str]]:
    edges = edge_map(case)
    chunks = chunk_map(case)
    valid = 0
    total = 0
    errors: list[str] = []
    for rel in rule.get("relations", []):
        if not isinstance(rel, dict):
            continue
        total += 1
        rel_type = normalize_relation(rel.get("type"))
        if rel_type not in ALLOWED_RELATIONS:
            errors.append(f"relation type not allowed: {rel_type}")
            continue
        edge_ids = evidence_ids(rel, "kg_edge_ids")
        chunk_ids = evidence_ids(rel, "chunk_ids")
        edge_ok = any(edges.get(edge_id, {}).get("relation") == rel_type for edge_id in edge_ids)
        chunk_ok = all(chunk_id in chunks for chunk_id in chunk_ids) if chunk_ids else False
        if edge_ok and chunk_ok:
            valid += 1
        else:
            errors.append(f"relation not grounded: {rel_type}")
    return valid, total, errors


def grounding_metrics(rule_base: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    nodes = node_map(case)
    chunks = chunk_map(case)
    schema = validate_schema(rule_base, case)
    rules = rule_base.get("rules", []) if isinstance(rule_base, dict) else []

    grounded_rules = 0
    total_rules = 0
    valid_constraints = 0
    total_constraints = 0
    valid_relations = 0
    total_relations = 0
    valid_provenance = 0
    total_provenance = 0
    guard_valid = 0
    all_errors = list(schema["errors"])

    for rule in rules:
        if not isinstance(rule, dict):
            continue
        total_rules += 1
        source_ids = [str(node_id) for node_id in rule.get("source_node_ids", [])]
        source_ok = bool(source_ids) and all(node_id in nodes for node_id in source_ids)
        if validate_guard_schema(rule):
            guard_valid += 1
        else:
            all_errors.append(f"invalid guard schema: {rule.get('rule_id')}")

        vp, tp = validate_provenance(rule, chunks)
        valid_provenance += vp
        total_provenance += tp
        if source_ok and vp > 0:
            grounded_rules += 1

        vc, tc, con_errors = validate_constraints(rule, case)
        valid_constraints += vc
        total_constraints += tc
        all_errors.extend(con_errors)

        vr, tr, rel_errors = validate_relations(rule, case)
        valid_relations += vr
        total_relations += tr
        all_errors.extend(rel_errors)

    return {
        "schema_pass": schema["pass"],
        "rule_count": total_rules,
        "grounded_rule_rate": pct(grounded_rules, total_rules),
        "constraint_grounding_rate": pct(valid_constraints, total_constraints),
        "relation_grounding_rate": pct(valid_relations, total_relations),
        "provenance_validity_rate": pct(valid_provenance, total_provenance),
        "guard_schema_valid_rate": pct(guard_valid, total_rules),
        "constraint_count": total_constraints,
        "relation_count": total_relations,
        "provenance_count": total_provenance,
        "errors": all_errors,
    }


def pct(num: float, den: float) -> float:
    if den == 0:
        return 100.0
    return 100.0 * float(num) / float(den)


def candidate_ids(rule: dict[str, Any]) -> set[str]:
    ids = {str(rule.get("rule_id", "")), str(rule.get("name", "")).lower()}
    ids.update(str(node_id) for node_id in rule.get("source_node_ids", []))
    return {item for item in ids if item}


def guard_applies(rule: dict[str, Any], omega: dict[str, Any]) -> bool:
    guard = rule.get("guard")
    if guard in (None, {}, []):
        return True
    clauses = guard.get("all", []) if isinstance(guard, dict) else []
    for clause in clauses:
        field = clause.get("field")
        op = clause.get("op")
        expected = clause.get("value")
        actual = omega.get(field)
        if not compare_values(actual, op, expected):
            return False
    return True


def compare_values(actual: Any, op: str, expected: Any) -> bool:
    if op == "eq":
        return actual == expected or str(actual).lower() == str(expected).lower()
    if op == "neq":
        return not compare_values(actual, "eq", expected)
    if op == "in":
        if isinstance(expected, list):
            return actual in expected or str(actual) in [str(item) for item in expected]
        return False
    try:
        a = float(actual)
        b = float(expected)
    except (TypeError, ValueError):
        return False
    if op == "gt":
        return a > b
    if op == "gte":
        return a >= b
    if op == "lt":
        return a < b
    if op == "lte":
        return a <= b
    return False


def resolve_rule_target(target: Any, rules: list[dict[str, Any]]) -> str | None:
    if target is None:
        return None
    target_s = str(target)
    target_lower = target_s.lower()
    for rule in rules:
        if target_s in candidate_ids(rule) or target_lower in candidate_ids(rule):
            return str(rule.get("rule_id"))
    return None


def compile_rule_structures(rule_base: dict[str, Any], omega: dict[str, Any]) -> dict[str, Any]:
    raw_rules = rule_base.get("rules", []) if isinstance(rule_base, dict) else []
    rules = [rule for rule in raw_rules if isinstance(rule, dict) and rule.get("rule_id")]
    applicable = [rule for rule in rules if guard_applies(rule, omega)]
    applicable_ids = {str(rule["rule_id"]) for rule in applicable}

    defeated: set[str] = set()
    relation_events: list[dict[str, Any]] = []
    for rule in applicable:
        src_id = str(rule["rule_id"])
        for rel in rule.get("relations", []):
            rel_type = normalize_relation(rel.get("type"))
            if rel_type not in {"overrides", "precedes"}:
                continue
            target_id = resolve_rule_target(rel.get("target_rule"), applicable)
            if target_id and target_id in applicable_ids and src_id in applicable_ids:
                defeated.add(target_id)
                relation_events.append({"type": rel_type, "source": src_id, "target": target_id})

    survivors = [rule for rule in applicable if str(rule["rule_id"]) not in defeated]
    survivor_ids = {str(rule["rule_id"]) for rule in survivors}

    conflict_edges: list[tuple[str, str]] = []
    for rule in survivors:
        src_id = str(rule["rule_id"])
        for rel in rule.get("relations", []):
            rel_type = normalize_relation(rel.get("type"))
            if rel_type != "excludes":
                continue
            target_id = resolve_rule_target(rel.get("target_rule"), survivors)
            if target_id and target_id in survivor_ids and target_id != src_id:
                conflict_edges.append(tuple(sorted((src_id, target_id))))
                relation_events.append({"type": rel_type, "source": src_id, "target": target_id})
    conflict_edges = sorted(set(conflict_edges))

    components = connected_components(survivor_ids, conflict_edges)
    conflict_nodes = set().union(*components) if components else set()
    common = [rule for rule in survivors if str(rule["rule_id"]) not in conflict_nodes]
    rule_by_id = {str(rule["rule_id"]): rule for rule in survivors}

    if not components:
        structures = [survivors]
    else:
        choices = []
        for comp in components:
            choices.append([rule_by_id[rule_id] for rule_id in sorted(comp)])
        structures = []
        for combo in product(*choices):
            structures.append(common + list(combo))

    return {
        "applicable_rule_ids": sorted(applicable_ids),
        "defeated_rule_ids": sorted(defeated),
        "valid_structures": [
            [str(rule["rule_id"]) for rule in structure] for structure in structures
        ],
        "structure_rules": structures,
        "relation_events": relation_events,
    }


def connected_components(nodes: set[str], edges: list[tuple[str, str]]) -> list[set[str]]:
    adjacency: dict[str, set[str]] = {node: set() for node in nodes}
    for a, b in edges:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    visited: set[str] = set()
    comps: list[set[str]] = []
    for node in sorted(nodes):
        if node in visited or not adjacency.get(node):
            continue
        stack = [node]
        comp: set[str] = set()
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            comp.add(cur)
            stack.extend(sorted(adjacency.get(cur, set()) - visited))
        if len(comp) > 1:
            comps.append(comp)
    return comps


def constraint_holds(con: dict[str, Any], x: dict[str, Any]) -> bool:
    variable = con.get("variable")
    if variable not in x:
        return False
    try:
        lhs = float(x[variable])
        rhs = float(con.get("value"))
    except (TypeError, ValueError):
        return False
    op = normalize_op(con.get("op"))
    tol = 1e-8
    if op == ">=":
        return lhs + tol >= rhs
    if op == "<=":
        return lhs <= rhs + tol
    if op == "=":
        return math.isclose(lhs, rhs, rel_tol=1e-8, abs_tol=1e-8)
    if op == ">":
        return lhs > rhs + tol
    if op == "<":
        return lhs + tol < rhs
    return False


def decision_valid(structures: list[list[dict[str, Any]]], x: dict[str, Any]) -> bool:
    for structure in structures:
        ok = True
        for rule in structure:
            for con in rule.get("constraints", []):
                if not constraint_holds(con, x):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return True
    return False


def semantic_metrics(rule_base: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    compiled = compile_rule_structures(rule_base, case["scenario_omega"])
    expected = case["expected_behavior"]
    decisions = expected["candidate_decisions"]
    correct = 0
    valid_total = 0
    valid_kept = 0
    invalid_total = 0
    false_accept = 0
    false_reject = 0
    per_decision = []
    for decision in decisions:
        predicted = decision_valid(compiled["structure_rules"], decision["x"])
        expected_valid = bool(decision["expected_valid"])
        if predicted == expected_valid:
            correct += 1
        if expected_valid:
            valid_total += 1
            if predicted:
                valid_kept += 1
            else:
                false_reject += 1
        else:
            invalid_total += 1
            if predicted:
                false_accept += 1
        per_decision.append(
            {
                "id": decision["id"],
                "expected_valid": expected_valid,
                "predicted_valid": predicted,
                "pass": predicted == expected_valid,
            }
        )

    observed_rel_types = {event["type"] for event in compiled["relation_events"]}
    observed_rel_types.update(
        normalize_relation(rel.get("type"))
        for rule in rule_base.get("rules", [])
        for rel in rule.get("relations", [])
        if isinstance(rel, dict)
    )
    required_relations = set(expected.get("required_relation_types", []))
    relation_pass = required_relations.issubset(observed_rel_types)
    min_structures_pass = len(compiled["valid_structures"]) >= int(expected["min_valid_structures"])
    decision_pass = correct == len(decisions)
    case_pass = decision_pass and relation_pass and min_structures_pass

    return {
        "case_pass": case_pass,
        "decision_accuracy": pct(correct, len(decisions)),
        "sem_csr": pct(valid_kept, valid_total),
        "false_accept_rate": pct(false_accept, invalid_total),
        "false_reject_rate": pct(false_reject, valid_total),
        "relation_pass": relation_pass,
        "min_structures_pass": min_structures_pass,
        "valid_structure_count": len(compiled["valid_structures"]),
        "required_relation_types": sorted(required_relations),
        "observed_relation_types": sorted(observed_rel_types),
        "valid_structures": compiled["valid_structures"],
        "defeated_rule_ids": compiled["defeated_rule_ids"],
        "per_decision": per_decision,
    }


def aggregate_metrics(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(key_path: list[str]) -> float:
        vals = []
        for row in case_results:
            cur: Any = row
            for key in key_path:
                cur = cur[key]
            vals.append(float(cur))
        return float(sum(vals) / max(len(vals), 1))

    relation_success: dict[str, list[float]] = {rel: [] for rel in ALLOWED_RELATIONS}
    for row in case_results:
        req = row["semantic"]["required_relation_types"]
        obs = set(row["semantic"]["observed_relation_types"])
        for rel in req:
            relation_success.setdefault(rel, []).append(100.0 if rel in obs else 0.0)

    return {
        "num_cases": len(case_results),
        "schema_case_pass_rate": mean(["grounding", "schema_pass_numeric"]),
        "grounded_rule_rate": mean(["grounding", "grounded_rule_rate"]),
        "constraint_grounding_rate": mean(["grounding", "constraint_grounding_rate"]),
        "relation_grounding_rate": mean(["grounding", "relation_grounding_rate"]),
        "provenance_validity_rate": mean(["grounding", "provenance_validity_rate"]),
        "semantic_behavior_pass_rate": mean(["semantic", "case_pass_numeric"]),
        "decision_accuracy": mean(["semantic", "decision_accuracy"]),
        "sem_csr": mean(["semantic", "sem_csr"]),
        "false_accept_rate": mean(["semantic", "false_accept_rate"]),
        "false_reject_rate": mean(["semantic", "false_reject_rate"]),
        "relation_success_rate": {
            rel: (sum(vals) / len(vals) if vals else None)
            for rel, vals in sorted(relation_success.items())
        },
    }


def run_case(
    case: dict[str, Any],
    system_prompt: str,
    providers: list[Provider],
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    prompt_payload = compact_case_for_prompt(case)
    errors = []
    raw_text = ""
    meta: dict[str, Any] = {}
    for provider in providers:
        try:
            log(f"Calling {provider.name}/{provider.model} for {case['case_id']}")
            raw_text, meta = call_provider(
                provider=provider,
                system_prompt=system_prompt,
                user_payload=prompt_payload,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            rule_base = extract_json(raw_text)
            meta["used_provider"] = provider.name
            break
        except Exception as exc:  # noqa: BLE001 - preserve fallback information.
            errors.append({"provider": provider.name, "error": str(exc)})
            log(f"Provider {provider.name} failed for {case['case_id']}: {exc}")
    else:
        return {
            "case_id": case["case_id"],
            "domain": case["domain"],
            "status": "api_or_parse_failed",
            "provider_errors": errors,
            "raw_text": raw_text,
            "rule_base": None,
            "grounding": None,
            "semantic": None,
        }

    grounding = grounding_metrics(rule_base, case)
    semantic = semantic_metrics(rule_base, case)
    grounding["schema_pass_numeric"] = 100.0 if grounding["schema_pass"] else 0.0
    semantic["case_pass_numeric"] = 100.0 if semantic["case_pass"] else 0.0
    return {
        "case_id": case["case_id"],
        "domain": case["domain"],
        "interaction_type": case["interaction_type"],
        "status": "ok",
        "provider": meta,
        "provider_errors": errors,
        "raw_text": raw_text,
        "rule_base": rule_base,
        "grounding": grounding,
        "semantic": semantic,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["auto", *PROVIDERS.keys()], default="auto")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=2500)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cases_payload = read_json(CASES_PATH)
    cases = cases_payload["cases"][: args.limit]
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    providers = provider_order(args.provider)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log(
        "Starting LLM KG-to-rule validation: "
        f"cases={len(cases)}, provider={args.provider}, temperature={args.temperature}"
    )

    outputs: list[dict[str, Any]] = []
    for idx, case in enumerate(cases, start=1):
        result = run_case(
            case=case,
            system_prompt=system_prompt,
            providers=providers,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
        outputs.append(result)
        write_json(
            OUTPUT_PATH,
            {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "validation_set": cases_payload["version"],
                "results": outputs,
            },
        )
        log(f"Finished {idx}/{len(cases)}: {case['case_id']} status={result['status']}")

    ok_results = [row for row in outputs if row["status"] == "ok"]
    aggregate = aggregate_metrics(ok_results) if ok_results else {}
    failed = [row for row in outputs if row["status"] != "ok"]
    result_payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "validation_set": cases_payload["version"],
        "provider_mode": args.provider,
        "num_cases_requested": len(cases),
        "num_cases_ok": len(ok_results),
        "num_cases_failed": len(failed),
        "aggregate": aggregate,
        "case_results": [
            {
                "case_id": row["case_id"],
                "domain": row["domain"],
                "interaction_type": row.get("interaction_type"),
                "status": row["status"],
                "provider": row.get("provider"),
                "grounding": row.get("grounding"),
                "semantic": row.get("semantic"),
            }
            for row in outputs
        ],
        "failed_cases": failed,
    }
    write_json(RESULT_PATH, result_payload)
    log(f"Saved raw outputs to {OUTPUT_PATH}")
    log(f"Saved validation results to {RESULT_PATH}")
    print(json.dumps(result_payload["aggregate"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
