from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEPENDENCY_TYPES = {"depends_on", "requires", "uses_parameter", "applies_to"}
RELATION_FILTER_TYPES = {
    *DEPENDENCY_TYPES,
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
    "formula_variant_of",
    "parameter_variant_of",
    "piecewise_variant_of",
    "propagates_to",
}
RELATION_KEEP_CLASSES = {
    "required",
    "supporting_dependency",
    "dependency",
    "scenario_applicability",
    "scenario_conditioned_applicability",
    "guard_applicability",
    "conflict",
    "exception",
    "override",
    "alternative",
    "mutually_exclusive",
    "precedence",
    "refinement",
    "parameter_variant",
    "visible_guard_candidate",
    "relation_relevant",
    "uncertain",
}
STRICT_RELATION_KEEP_CLASSES = {
    "supporting_dependency",
    "dependency",
    "scenario_applicability",
    "scenario_conditioned_applicability",
    "guard_applicability",
    "conflict",
    "exclusion",
    "exception",
    "override",
    "alternative",
    "mutually_exclusive",
    "precedence",
    "priority",
    "refinement",
    "parameter_variant",
    "formula_variant",
    "parameter_propagation",
    "formula_propagation",
}
SCENARIO_APPLICABILITY_LABELS = {
    "scenario_applicability",
    "scenario_conditioned_applicability",
    "guard_applicability",
    "visible_guard_candidate",
}


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
}


def normalize_provider(provider: str, model: str | None = None) -> Provider:
    key = provider.strip().lower()
    if key not in PROVIDERS:
        raise ValueError(f"Unknown LLM rerank provider: {provider}")
    base = PROVIDERS[key]
    if key == "qwen" and not os.environ.get("QWEN_API_KEY") and os.environ.get("DASHSCOPE_API_KEY"):
        os.environ["QWEN_API_KEY"] = os.environ["DASHSCOPE_API_KEY"]
    return Provider(base.name, base.env_key, base.url, model or base.model)


def compact_task(task: dict[str, Any]) -> dict[str, Any]:
    stress_meta = task.get("stress_metadata") or {}
    return {
        "omega_id": task.get("omega_id"),
        "domain": task.get("domain"),
        "source_domain": task.get("source_domain"),
        "source_domains": stress_meta.get("source_domains"),
        "task_type": task.get("task_type"),
        "title": task.get("title"),
        "engineering_task": stress_meta.get("engineering_task") or task.get("engineering_task"),
        "design_intent": task.get("design_intent"),
        "scenario_facts": task.get("scenario_facts", {}),
        "decision_variables": task.get("decision_variables", {}),
        "objectives": task.get("objectives", []),
        "query_preferences": task.get("query_preferences", {}),
    }


def compact_rule(rule: dict[str, Any], reasons: list[str] | None = None) -> dict[str, Any]:
    constraints = []
    for constraint in rule.get("constraints", [])[:6]:
        constraints.append(
            {
                "variable": constraint.get("variable"),
                "op": constraint.get("op"),
                "value": constraint.get("value"),
                "unit": constraint.get("unit"),
            }
        )
    relations = []
    for relation in rule.get("relations", [])[:8]:
        relations.append({"type": relation.get("type"), "target": relation.get("target")})
    return {
        "rule_id": rule.get("rule_id"),
        "name": rule.get("name"),
        "domain": rule.get("domain"),
        "source_domain": rule.get("source_domain"),
        "rule_type": rule.get("rule_type"),
        "guard": rule.get("guard"),
        "constraints": constraints,
        "relations": relations,
        "conflict_class": rule.get("conflict_class") or rule.get("conflict_group"),
        "typed_grounding_reasons": reasons or [],
    }


def cache_key(
    domain: str,
    task: dict[str, Any],
    rules: list[dict[str, Any]],
    provider: Provider,
    purpose: str | None = None,
) -> str:
    payload = {
        "version": 4 if purpose else 3,
        "domain": domain,
        "task": compact_task(task),
        "rule_ids": sorted(str(rule.get("rule_id")) for rule in rules),
        "provider": provider.name,
        "model": provider.model,
    }
    if purpose:
        payload["purpose"] = purpose
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            selected_match = re.search(r'"selected_rule_ids"\s*:\s*\[(.*?)\]', stripped, flags=re.DOTALL)
            if not selected_match:
                raise
            return {
                "selected_rule_ids": re.findall(r'"([^"]+)"', selected_match.group(1)),
                "dropped_rule_ids": [],
                "rationales_by_rule_id": {},
                "_parse_recovery": "selected_rule_ids_regex",
            }
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            selected_match = re.search(r'"selected_rule_ids"\s*:\s*\[(.*?)\]', match.group(0), flags=re.DOTALL)
            if not selected_match:
                raise
            return {
                "selected_rule_ids": re.findall(r'"([^"]+)"', selected_match.group(1)),
                "dropped_rule_ids": [],
                "rationales_by_rule_id": {},
                "_parse_recovery": "selected_rule_ids_regex",
            }


def call_provider(
    provider: Provider,
    payload: dict[str, Any],
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    api_key = os.environ.get(provider.env_key)
    if not api_key:
        raise RuntimeError(f"Missing environment variable {provider.env_key}")
    system_prompt = (
        "You are a cautious rule-grounding reranker for a knowledge-based optimization system. "
        "Classify candidate rules by visible evidence and select a minimal sufficient rule set. "
        "A selected rule must be required for the task, a needed supporting dependency/classification rule, "
        "or a visible exception/precedence rule. Drop generic, sibling, mutually exclusive, or domain-family "
        "near-neighbor rules. Preserve rules marked by typed_grounding_reasons as domain_profile_required "
        "unless there is explicit contradictory visible evidence. Use only the provided JSON. Return strict JSON only."
    )
    body = {
        "model": provider.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Rerank this typed symbolic candidate set. For each rule, classify it as one of: "
                    "required, supporting_dependency, irrelevant, or mutually_exclusive. "
                    "Return JSON with keys selected_rule_ids, dropped_rule_ids, classifications_by_rule_id, "
                    "and rationales_by_rule_id. selected_rule_ids should be the minimal sufficient set, "
                    "not every semantically adjacent rule. All selected_rule_ids and dropped_rule_ids must be "
                    "from candidate_rules.\n\n"
                    + json.dumps(payload, ensure_ascii=False, indent=2)
                ),
            },
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        provider.url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{provider.name} HTTP {exc.code}: {detail[:500]}") from exc
    parsed = json.loads(raw)
    content = parsed["choices"][0]["message"]["content"]
    return extract_json_object(content), {
        "provider": provider.name,
        "model": provider.model,
        "usage": parsed.get("usage", {}),
        "finish_reason": parsed["choices"][0].get("finish_reason"),
    }


def call_relation_filter_provider(
    provider: Provider,
    payload: dict[str, Any],
    temperature: float,
    max_tokens: int,
    timeout: int,
    relation_only: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    api_key = os.environ.get(provider.env_key)
    if not api_key:
        raise RuntimeError(f"Missing environment variable {provider.env_key}")
    if relation_only:
        system_prompt = (
            "You are a strict six-relation rule-grounding filter for a CTHR system. "
            "Your job is NOT to choose final valid rules. Keep only rules that visibly participate in one of "
            "these six relation categories: scenario-conditioned applicability, dependency/supporting premise, "
            "exclusion or alternative branch, exception/override, precedence/priority, or parameter/formula propagation. "
            "Drop ordinary required rules, generic nearby rules, broad domain-family rules, and uncertain rules unless "
            "their role in one of the six categories is visible in the provided JSON. Use only the provided JSON. "
            "Return strict JSON only."
        )
    else:
        system_prompt = (
            "You are a conservative relation-preserving rule-grounding filter for a CTHR system. "
            "Your job is NOT to choose the final valid rules. Keep every rule that could participate in "
            "one of these rule relations: required rule, supporting dependency, exception/override, "
            "conflict or mutual exclusion, alternative branch, precedence/priority, refinement, or parameter/formula variant. "
            "Remove only rules that are clearly irrelevant to the task and to these relation neighborhoods. "
            "When uncertain, keep the rule. Use only the provided JSON. Return strict JSON only."
        )
    class_list = (
        "scenario_applicability, supporting_dependency, conflict, exception, alternative, "
        "mutually_exclusive, precedence, override, parameter_propagation, formula_propagation, "
        "refinement, or clearly_irrelevant"
        if relation_only
        else (
            "required, supporting_dependency, conflict, exception, alternative, mutually_exclusive, "
            "precedence, refinement, parameter_variant, visible_guard_candidate, uncertain, or clearly_irrelevant"
        )
    )
    selection_instruction = (
        "kept_rule_ids should include only six-relation-relevant rules; dropped_rule_ids should include "
        "ordinary required, generic, uncertain, and clearly irrelevant rules."
        if relation_only
        else (
            "kept_rule_ids should include all six-relation-relevant rules; dropped_rule_ids should include "
            "only clearly_irrelevant rules."
        )
    )
    body = {
        "model": provider.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Filter this broad candidate set before symbolic CTHR recovery. "
                    "Do not output a minimal valid set. For each rule, classify it as one of: "
                    f"{class_list}. "
                    "Return JSON with keys kept_rule_ids, dropped_rule_ids, classifications_by_rule_id, "
                    f"and rationales_by_rule_id. {selection_instruction}\n\n"
                    + json.dumps(payload, ensure_ascii=False, indent=2)
                ),
            },
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        provider.url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{provider.name} HTTP {exc.code}: {detail[:500]}") from exc
    parsed = json.loads(raw)
    content = parsed["choices"][0]["message"]["content"]
    return extract_json_object(content), {
        "provider": provider.name,
        "model": provider.model,
        "usage": parsed.get("usage", {}),
        "finish_reason": parsed["choices"][0].get("finish_reason"),
    }


def dependency_closure(selected: set[str], candidate_rules: list[dict[str, Any]]) -> set[str]:
    by_id = {str(rule.get("rule_id")): rule for rule in candidate_rules if rule.get("rule_id")}
    out = set(selected)
    changed = True
    while changed:
        changed = False
        for rule_id in list(out):
            rule = by_id.get(rule_id)
            if not rule:
                continue
            for relation in rule.get("relations", []):
                if str(relation.get("type", "")).lower() not in DEPENDENCY_TYPES:
                    continue
                target = str(relation.get("target", ""))
                if target in by_id and target not in out:
                    out.add(target)
                    changed = True
    return out


def relation_neighbor_closure(selected: set[str], candidate_rules: list[dict[str, Any]]) -> set[str]:
    by_id = {str(rule.get("rule_id")): rule for rule in candidate_rules if rule.get("rule_id")}
    out = set(selected)
    for rule in candidate_rules:
        rule_id = str(rule.get("rule_id", ""))
        if rule_id not in out:
            continue
        for relation in rule.get("relations", []):
            rtype = str(relation.get("type", "")).lower()
            target = str(relation.get("target", ""))
            if rtype in RELATION_FILTER_TYPES and target in by_id:
                out.add(target)
    for rule in candidate_rules:
        rule_id = str(rule.get("rule_id", ""))
        if rule_id in out:
            continue
        for relation in rule.get("relations", []):
            rtype = str(relation.get("type", "")).lower()
            target = str(relation.get("target", ""))
            if rtype in RELATION_FILTER_TYPES and target in out:
                out.add(rule_id)
                break
    return out


def has_relation_edge(rule: dict[str, Any], candidate_ids: set[str]) -> bool:
    for relation in rule.get("relations", []):
        rtype = str(relation.get("type", "")).lower()
        target = str(relation.get("target", ""))
        if rtype in RELATION_FILTER_TYPES and (not candidate_ids or target in candidate_ids):
            return True
    return False


def has_visible_guard(rule: dict[str, Any]) -> bool:
    guard = rule.get("guard")
    return bool(guard) and guard not in ({}, [], "true", True)


def strict_relation_only_selection(
    selected: set[str],
    classifications: dict[str, str],
    candidate_rules: list[dict[str, Any]],
) -> set[str]:
    candidate_ids = {str(rule.get("rule_id")) for rule in candidate_rules if rule.get("rule_id")}
    by_id = {str(rule.get("rule_id")): rule for rule in candidate_rules if rule.get("rule_id")}
    out: set[str] = set()
    for rule_id in selected:
        rule = by_id.get(rule_id)
        if not rule:
            continue
        label = classifications.get(rule_id, "")
        explicit_relation = has_relation_edge(rule, candidate_ids)
        scenario_labeled = label in SCENARIO_APPLICABILITY_LABELS and has_visible_guard(rule)
        if label in STRICT_RELATION_KEEP_CLASSES or explicit_relation or scenario_labeled:
            out.add(rule_id)
    return out


def protected_profile_rules(
    candidate_rules: list[dict[str, Any]],
    candidate_reasons: dict[str, list[str]],
) -> set[str]:
    protected: set[str] = set()
    for rule in candidate_rules:
        rule_id = str(rule.get("rule_id", ""))
        reasons = candidate_reasons.get(rule_id, [])
        if any("domain_profile_required" in str(reason) for reason in reasons):
            protected.add(rule_id)
    return protected


def rerank_candidate_rules(
    domain: str,
    task: dict[str, Any],
    candidate_rules: list[dict[str, Any]],
    candidate_reasons: dict[str, list[str]],
    provider_name: str = "qwen",
    model: str | None = None,
    cache_path: Path | None = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    timeout: int = 120,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    provider = normalize_provider(provider_name, model)
    candidate_ids = {str(rule.get("rule_id")) for rule in candidate_rules if rule.get("rule_id")}
    key = cache_key(domain, task, candidate_rules, provider)
    cache: dict[str, Any] = {}
    if cache_path and cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    if key in cache:
        response = cache[key]["response"]
        meta = cache[key].get("meta", {})
        cache_hit = True
    else:
        payload = {
            "domain": domain,
            "task": compact_task(task),
            "candidate_rules": [
                compact_rule(rule, candidate_reasons.get(str(rule.get("rule_id")), []))
                for rule in candidate_rules
            ],
        }
        response, meta = call_provider(provider, payload, temperature, max_tokens, timeout)
        cache_hit = False
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache[key] = {
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "response": response,
                "meta": meta,
            }
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

    raw_selected = response.get("selected_rule_ids", [])
    selected = {str(rule_id) for rule_id in raw_selected if str(rule_id) in candidate_ids}
    if not selected:
        selected = set(candidate_ids)
        status = "fallback_empty_llm_selection"
    else:
        status = "success"
    selected |= protected_profile_rules(candidate_rules, candidate_reasons)
    selected = dependency_closure(selected, candidate_rules)
    by_id = {str(rule.get("rule_id")): rule for rule in candidate_rules if rule.get("rule_id")}
    diagnostics = {
        "status": status,
        "cache_hit": cache_hit,
        "provider": provider.name,
        "model": provider.model,
        "input_candidate_count": len(candidate_rules),
        "selected_candidate_count": len(selected),
        "dropped_candidate_count": len(candidate_ids - selected),
        "llm_meta": meta,
    }
    return [by_id[rule_id] for rule_id in sorted(selected)], diagnostics


def relation_filter_candidate_rules(
    domain: str,
    task: dict[str, Any],
    candidate_rules: list[dict[str, Any]],
    candidate_reasons: dict[str, list[str]],
    provider_name: str = "qwen",
    model: str | None = None,
    cache_path: Path | None = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    timeout: int = 120,
    relation_only: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    provider = normalize_provider(provider_name, model)
    candidate_ids = {str(rule.get("rule_id")) for rule in candidate_rules if rule.get("rule_id")}
    purpose = "strict_six_relation_filter" if relation_only else "relation_preserving_filter"
    key = cache_key(domain, task, candidate_rules, provider, purpose=purpose)
    cache: dict[str, Any] = {}
    if cache_path and cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    if key in cache:
        response = cache[key]["response"]
        meta = cache[key].get("meta", {})
        cache_hit = True
    else:
        payload = {
            "domain": domain,
            "task": compact_task(task),
            "candidate_rules": [
                compact_rule(rule, candidate_reasons.get(str(rule.get("rule_id")), []))
                for rule in candidate_rules
            ],
        }
        response, meta = call_relation_filter_provider(
            provider,
            payload,
            temperature,
            max_tokens,
            timeout,
            relation_only=relation_only,
        )
        cache_hit = False
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache[key] = {
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "response": response,
                "meta": meta,
            }
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

    classifications = {
        str(rule_id): str(label).strip().lower()
        for rule_id, label in (response.get("classifications_by_rule_id") or {}).items()
    }
    raw_kept = response.get("kept_rule_ids", response.get("selected_rule_ids", []))
    raw_dropped = response.get("dropped_rule_ids", [])
    kept = {str(rule_id) for rule_id in raw_kept if str(rule_id) in candidate_ids}
    clearly_irrelevant = {
        str(rule_id)
        for rule_id in raw_dropped
        if str(rule_id) in candidate_ids
        and classifications.get(str(rule_id), "clearly_irrelevant") in {"clearly_irrelevant", "irrelevant", "unrelated"}
    }
    keep_by_class = {
        rule_id
        for rule_id, label in classifications.items()
        if rule_id in candidate_ids and label in RELATION_KEEP_CLASSES
    }
    selected = (candidate_ids - clearly_irrelevant) | kept | keep_by_class
    if not selected:
        selected = set(candidate_ids)
        status = "fallback_empty_relation_filter"
    else:
        status = "success"
    selected |= protected_profile_rules(candidate_rules, candidate_reasons)
    selected = dependency_closure(selected, candidate_rules)
    if relation_only:
        selected = strict_relation_only_selection(selected, classifications, candidate_rules)
        selected = dependency_closure(selected, candidate_rules)
    else:
        selected = relation_neighbor_closure(selected, candidate_rules)
    by_id = {str(rule.get("rule_id")): rule for rule in candidate_rules if rule.get("rule_id")}
    diagnostics = {
        "status": status,
        "cache_hit": cache_hit,
        "provider": provider.name,
        "model": provider.model,
        "input_candidate_count": len(candidate_rules),
        "selected_candidate_count": len(selected),
        "dropped_candidate_count": len(candidate_ids - selected),
        "purpose": purpose,
        "relation_only": relation_only,
        "llm_meta": meta,
    }
    return [by_id[rule_id] for rule_id in sorted(selected)], diagnostics
