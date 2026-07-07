from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
SCRIPTS_DIR = ROOT / "scripts"

sys.path.insert(0, str(SCRIPTS_DIR))

import llm_grounding_reranker as llm_reranker  # noqa: E402
import run_section_6_2_table1_fullkg_pipeline as full  # noqa: E402
import run_section_6_2_table1_flat_old_candidate_mapping as oldmap  # noqa: E402


VALID_RESOLVER_MODES = {
    "relation_filter",
    "strict_relation_filter",
    "profile_resolver",
    "profile_llm_resolver",
    "profile_auto_resolver",
}

OUT_PREFIX_BY_DOMAIN = {
    "relation_filter": {
        "aviation": "section_6_3_aviation_old_candidate_llm_filter_candidate_to_valid",
        "architecture": "section_6_3_architecture_old_candidate_llm_filter_candidate_to_valid",
    },
    "strict_relation_filter": {
        "aviation": "section_6_3_aviation_old_candidate_strict_relation_filter_candidate_to_valid",
        "architecture": "section_6_3_architecture_old_candidate_strict_relation_filter_candidate_to_valid",
    },
    "profile_resolver": {
        "aviation": "section_6_3_aviation_old_candidate_profile_resolver_candidate_to_valid",
        "architecture": "section_6_3_architecture_old_candidate_profile_resolver_candidate_to_valid",
    },
    "profile_llm_resolver": {
        "aviation": "section_6_3_aviation_old_candidate_profile_llm_resolver_candidate_to_valid",
        "architecture": "section_6_3_architecture_old_candidate_profile_llm_resolver_candidate_to_valid",
    },
    "profile_auto_resolver": {
        "aviation": "section_6_3_aviation_old_candidate_profile_auto_resolver_candidate_to_valid",
        "architecture": "section_6_3_architecture_old_candidate_profile_auto_resolver_candidate_to_valid",
    },
    "profile_auto_resolver_recall_guard": {
        "aviation": "section_6_3_aviation_old_candidate_recall_guard_profile_auto_resolver_candidate_to_valid",
        "architecture": "section_6_3_architecture_old_candidate_recall_guard_profile_auto_resolver_candidate_to_valid",
    },
}


def safe_ratio(numer: int, denom: int) -> float:
    return math.nan if denom == 0 else numer / denom


def rounded(value: float) -> float:
    return value if isinstance(value, float) and math.isnan(value) else round(float(value), 4)


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for key in (
        "Candidate / Reference Ratio",
        "Filtered / Reference Ratio",
        "Predicted / Reference Ratio",
        "Rule-ID Precision",
        "Rule-ID Recall",
    ):
        out[key] = rounded(out[key])
    return out


def candidate_reasons_from_scores(
    candidate_ids: list[str],
    scored: list[tuple[float, str]],
) -> dict[str, list[str]]:
    score_by_id = {rule_id: score for score, rule_id in scored}
    return {
        rule_id: [
            oldmap.OLD_MAPPING_NAME,
            f"old_candidate_mapping_score:{score_by_id.get(rule_id, 0.0):.3f}",
        ]
        for rule_id in candidate_ids
    }


AVIATION_RECALL_GUARD_TOKENS = {
    "assumption",
    "timing",
    "tolerance",
    "stability",
    "convention",
    "supplementary",
    "chart",
    "clearance",
    "obstacle",
    "navigation",
    "publication",
    "required",
    "requirement",
}


def aviation_profile_required_rule_ids(
    rule_library: dict[str, Any],
    grounding_task: dict[str, Any],
) -> list[str]:
    profile = full.aviation_grounding.aviation_task_profile(grounding_task)
    target_groups = set(profile.get("require", set()))
    if hasattr(full.aviation_grounding, "aviation_slot_groups"):
        for slot in full.aviation_grounding.aviation_slot_groups(grounding_task):
            target_groups.update(slot)
    if not target_groups:
        return []
    blocked_groups = set(profile.get("block", set()))
    out: list[str] = []
    for rule in rule_library.get("rules", []):
        rule_id = str(rule.get("rule_id", ""))
        if not rule_id or str(rule.get("domain", "")).lower() != "aviation":
            continue
        groups = full.aviation_grounding.aviation_rule_groups(rule)
        if groups & target_groups and not ((groups & blocked_groups) - target_groups):
            out.append(rule_id)
    return sorted(dict.fromkeys(out))


def old_candidate_ids_with_recall_guard(
    spec: full.DatasetSpec,
    rule_library: dict[str, Any],
    query: dict[str, Any],
    grounding_task: dict[str, Any],
    *,
    limit: int,
    recall_guard: bool,
) -> tuple[list[str], list[tuple[float, str]]]:
    candidate_ids, scored = oldmap.old_candidate_ids(
        spec,
        rule_library,
        query,
        grounding_task,
        limit=limit,
    )
    if not recall_guard or spec.domain != "aviation":
        return candidate_ids, scored
    task_text = " ".join(
        str(value)
        for value in (
            grounding_task.get("task_type"),
            grounding_task.get("title"),
            grounding_task.get("design_intent"),
            grounding_task.get("engineering_task"),
        )
        if value
    ).lower()
    selected = list(candidate_ids)
    selected_set = set(selected)
    for rule_id in aviation_profile_required_rule_ids(rule_library, grounding_task):
        if rule_id not in selected_set:
            selected.append(rule_id)
            selected_set.add(rule_id)
    for score, rule_id in scored:
        if len(selected) >= limit:
            break
        if rule_id in selected_set or score < 2.0:
            continue
        rule_text = str(rule_id).lower().replace("_", " ").replace("-", " ")
        shared_guard_token = any(token in task_text and token in rule_text for token in AVIATION_RECALL_GUARD_TOKENS)
        support_like = any(token in rule_text for token in AVIATION_RECALL_GUARD_TOKENS)
        same_family_hint = any(token in task_text and token in rule_text for token in ("holding", "intermediate", "chart", "paoas", "missed"))
        if support_like and (shared_guard_token or same_family_hint or score >= 6.0):
            selected.append(rule_id)
            selected_set.add(rule_id)
    return sorted(dict.fromkeys(selected)), scored


def candidate_constrained_profile_resolver(
    spec: full.DatasetSpec,
    raw_candidate_rules: list[dict[str, Any]],
    grounding_task: dict[str, Any],
    *,
    use_llm: bool,
    llm_provider: str,
    llm_model: str | None,
    llm_cache: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    restricted_library = {"rules": raw_candidate_rules}
    profile_rules, profile_reasons = full.candidate_rules_for_domain(
        spec,
        restricted_library,
        grounding_task,
    )
    raw_ids = {str(rule.get("rule_id")) for rule in raw_candidate_rules if rule.get("rule_id")}
    profile_ids = {str(rule.get("rule_id")) for rule in profile_rules if rule.get("rule_id")}
    diagnostics: dict[str, Any] = {
        "purpose": "candidate_constrained_profile_resolver",
        "input_candidate_count": len(raw_candidate_rules),
        "profile_candidate_count": len(profile_rules),
        "profile_selected_rule_ids": sorted(profile_ids),
        "dropped_by_profile_count": len(raw_ids - profile_ids),
        "llm_minimal_rerank": {"enabled": False},
    }
    if not profile_rules:
        diagnostics["status"] = "fallback_empty_profile_selection"
        return raw_candidate_rules, diagnostics
    diagnostics["status"] = "success"
    if use_llm:
        profile_rules, llm_diagnostics = llm_reranker.rerank_candidate_rules(
            domain=spec.domain,
            task=grounding_task,
            candidate_rules=profile_rules,
            candidate_reasons=profile_reasons,
            provider_name=llm_provider,
            model=llm_model,
            cache_path=llm_cache,
        )
        diagnostics["llm_minimal_rerank"] = llm_diagnostics | {"enabled": True}
    return profile_rules, diagnostics


def cthr_select_after_candidate_resolver(
    spec: full.DatasetSpec,
    filtered_rules: list[dict[str, Any]],
    grounding_task: dict[str, Any],
    *,
    trust_profile_applicability: bool,
) -> list[str]:
    if not trust_profile_applicability or spec.domain != "aviation":
        return full.cthr_default_select(spec, filtered_rules, grounding_task)
    scenario = full.scenario_for_domain(spec, grounding_task)
    scenario["_trust_grounded_candidate_applicability"] = True
    resolution_rules = full.aviation_grounding.aviation_resolution_candidate_rules(
        filtered_rules,
        grounding_task,
    )
    result = full.ctv.cthr_recover_valid_rules(resolution_rules, scenario)
    if hasattr(full.aviation_grounding, "aviation_slot_constrained_rule_ids"):
        return sorted(
            full.aviation_grounding.aviation_slot_constrained_rule_ids(
                resolution_rules,
                grounding_task,
                set(result.predicted_rule_ids),
            )
        )
    return sorted(result.predicted_rule_ids)


def run_dataset(
    spec: full.DatasetSpec,
    *,
    limit: int,
    llm_provider: str,
    llm_model: str | None,
    llm_cache: Path,
    valid_resolver: str,
    aviation_recall_guard: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    algorithm_inputs = full.item_map(spec.algorithm_inputs)
    scenario_models = full.item_map(spec.scenario_models)
    references = full.item_map(spec.evaluation_references)
    templates_by_rule = full.constraint_template_map(spec.constraint_templates)
    rule_library = full.read_json(spec.rule_library)
    rule_by_id = {str(rule["rule_id"]): rule for rule in rule_library.get("rules", []) if rule.get("rule_id")}

    if set(algorithm_inputs) != set(scenario_models) or set(algorithm_inputs) != set(references):
        raise ValueError(f"{spec.name} layer IDs do not match")

    rows: list[dict[str, Any]] = []
    for task_id in sorted(algorithm_inputs):
        grounding_task = dict(algorithm_inputs[task_id])
        query = full.prepare_query(grounding_task, scenario_models[task_id])
        query["_compiled_rule_constraint_templates_by_id"] = templates_by_rule
        reference = references[task_id]
        reference_ids = full.reference_rule_ids(reference)

        raw_candidate_ids, scored = old_candidate_ids_with_recall_guard(
            spec,
            rule_library,
            query,
            grounding_task,
            limit=limit,
            recall_guard=aviation_recall_guard,
        )
        raw_candidate_rules = [rule_by_id[rule_id] for rule_id in raw_candidate_ids if rule_id in rule_by_id]
        raw_candidate_ids = sorted(str(rule["rule_id"]) for rule in raw_candidate_rules)
        candidate_reasons = candidate_reasons_from_scores(raw_candidate_ids, scored)

        if valid_resolver in {"relation_filter", "strict_relation_filter"}:
            filtered_rules, resolver_diagnostics = llm_reranker.relation_filter_candidate_rules(
                domain=spec.domain,
                task=grounding_task,
                candidate_rules=raw_candidate_rules,
                candidate_reasons=candidate_reasons,
                provider_name=llm_provider,
                model=llm_model,
                cache_path=llm_cache,
                relation_only=valid_resolver == "strict_relation_filter",
            )
        elif valid_resolver in {"profile_resolver", "profile_llm_resolver", "profile_auto_resolver"}:
            use_llm = valid_resolver == "profile_llm_resolver" or (
                valid_resolver == "profile_auto_resolver" and spec.domain == "architecture"
            )
            filtered_rules, resolver_diagnostics = candidate_constrained_profile_resolver(
                spec,
                raw_candidate_rules,
                grounding_task,
                use_llm=use_llm,
                llm_provider=llm_provider,
                llm_model=llm_model,
                llm_cache=llm_cache,
            )
        else:
            raise ValueError(f"Unsupported valid_resolver: {valid_resolver}")
        filtered_ids = sorted(str(rule["rule_id"]) for rule in filtered_rules)
        predicted_ids = cthr_select_after_candidate_resolver(
            spec,
            filtered_rules,
            grounding_task,
            trust_profile_applicability=valid_resolver
            in {"profile_resolver", "profile_llm_resolver", "profile_auto_resolver"},
        )

        predicted_set = set(predicted_ids)
        reference_set = set(reference_ids)
        overlap = predicted_set & reference_set
        extra = sorted(predicted_set - reference_set)
        missing = sorted(reference_set - predicted_set)

        rows.append(
            {
                "Dataset": spec.name,
                "task_id": task_id,
                "target_interaction": full.target_interaction(reference),
                "candidate_source": oldmap.OLD_MAPPING_NAME,
                "candidate_rule_count": len(raw_candidate_ids),
                "filtered_candidate_rule_count": len(filtered_ids),
                "reference_valid_rule_count": len(reference_ids),
                "predicted_valid_rule_count": len(predicted_ids),
                "Candidate / Reference Ratio": safe_ratio(len(raw_candidate_ids), len(reference_ids)),
                "Filtered / Reference Ratio": safe_ratio(len(filtered_ids), len(reference_ids)),
                "Predicted / Reference Ratio": safe_ratio(len(predicted_ids), len(reference_ids)),
                "Rule-ID Precision": safe_ratio(len(overlap), len(predicted_ids)),
                "Rule-ID Recall": safe_ratio(len(overlap), len(reference_ids)),
                "Exact Match": predicted_set == reference_set,
                "candidate_rule_ids_generated": raw_candidate_ids,
                "candidate_rule_ids_after_llm_relation_filter": filtered_ids,
                "reference_valid_rule_ids": reference_ids,
                "predicted_valid_rule_ids": predicted_ids,
                "extra_rule_ids": extra,
                "missing_rule_ids": missing,
                "_candidate_to_valid_resolver": resolver_diagnostics,
            }
        )

    summary = {
        "task count": len(rows),
        "dataset_root": str(spec.root),
        "candidate_source": oldmap.OLD_MAPPING_NAME,
        "old_mapping_limit": limit,
        "aviation_recall_guard": aviation_recall_guard,
        "candidate_to_valid": valid_resolver,
        "mean Candidate / Reference Ratio": statistics.mean(row["Candidate / Reference Ratio"] for row in rows),
        "mean Filtered / Reference Ratio": statistics.mean(row["Filtered / Reference Ratio"] for row in rows),
        "mean Predicted / Reference Ratio": statistics.mean(row["Predicted / Reference Ratio"] for row in rows),
        "mean Rule-ID Precision": statistics.mean(row["Rule-ID Precision"] for row in rows),
        "mean Rule-ID Recall": statistics.mean(row["Rule-ID Recall"] for row in rows),
        "exact match rate": statistics.mean(1.0 if row["Exact Match"] else 0.0 for row in rows),
        "total extra rules": sum(len(row["extra_rule_ids"]) for row in rows),
        "total missing rules": sum(len(row["missing_rule_ids"]) for row in rows),
        "rule_library": str(spec.rule_library),
        "llm_relation_filter": {
            "enabled": valid_resolver in {"relation_filter", "strict_relation_filter"},
            "provider": llm_provider,
            "model": llm_model,
            "cache": str(llm_cache),
            "relation_only": valid_resolver == "strict_relation_filter",
        },
        "candidate_constrained_profile_resolver": {
            "enabled": valid_resolver in {"profile_resolver", "profile_llm_resolver", "profile_auto_resolver"},
            "llm_policy": (
                "architecture_only"
                if valid_resolver == "profile_auto_resolver"
                else ("always" if valid_resolver == "profile_llm_resolver" else "never")
            ),
            "provider": llm_provider
            if valid_resolver in {"profile_llm_resolver", "profile_auto_resolver"}
            else None,
            "model": llm_model,
            "cache": str(llm_cache),
        },
        "audit": {
            "used_expected_candidate_field": False,
            "used_reference_valid_rules": False,
            "used_solver_constraints": False,
            "used_reference_cells": False,
            "used_semantic_validator": False,
        },
    }
    return rows, summary


def write_dataset_outputs(
    spec: full.DatasetSpec,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    valid_resolver: str,
    aviation_recall_guard: bool,
    output_prefix: str | None = None,
) -> None:
    prefix_key = (
        "profile_auto_resolver_recall_guard"
        if valid_resolver == "profile_auto_resolver" and aviation_recall_guard
        else valid_resolver
    )
    prefix = output_prefix or OUT_PREFIX_BY_DOMAIN[prefix_key][spec.domain]
    full_headers = [
        "Dataset",
        "task_id",
        "target_interaction",
        "candidate_source",
        "candidate_rule_count",
        "filtered_candidate_rule_count",
        "reference_valid_rule_count",
        "predicted_valid_rule_count",
        "Candidate / Reference Ratio",
        "Filtered / Reference Ratio",
        "Predicted / Reference Ratio",
        "Rule-ID Precision",
        "Rule-ID Recall",
        "Exact Match",
        "candidate_rule_ids_generated",
        "candidate_rule_ids_after_llm_relation_filter",
        "reference_valid_rule_ids",
        "predicted_valid_rule_ids",
        "extra_rule_ids",
        "missing_rule_ids",
    ]
    public_rows = [public_row(row) for row in rows]
    full.write_csv(RESULTS_DIR / f"{prefix}_full.csv", public_rows, full_headers)
    (RESULTS_DIR / f"{prefix}_full.md").write_text(
        full.markdown_table(public_rows, full_headers),
        encoding="utf-8",
    )
    full.write_json(RESULTS_DIR / f"{prefix}_full.json", public_rows)
    full.write_json(RESULTS_DIR / f"{prefix}_summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run old broad candidate grounding plus LLM relation filter plus CTHR strict recovery."
    )
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--domain", choices=["all", "aviation", "architecture"], default="all")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--dataset-label", default=None)
    parser.add_argument("--algorithm-inputs", type=Path, default=None)
    parser.add_argument("--scenario-models", type=Path, default=None)
    parser.add_argument("--evaluation-references", type=Path, default=None)
    parser.add_argument("--rule-library", type=Path, default=None)
    parser.add_argument("--constraint-templates", type=Path, default=None)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--relation-only-filter", action="store_true")
    parser.add_argument("--aviation-recall-guard", action="store_true")
    parser.add_argument(
        "--valid-resolver",
        choices=sorted(VALID_RESOLVER_MODES),
        default="relation_filter",
    )
    parser.add_argument("--llm-provider", default="qwen")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument(
        "--llm-cache",
        type=Path,
        default=RESULTS_DIR / "llm_grounding_relation_filter_cache.json",
    )
    args = parser.parse_args()
    valid_resolver = "strict_relation_filter" if args.relation_only_filter else args.valid_resolver

    all_summaries: dict[str, Any] = {}
    specs = list(full.DATASETS)
    if args.dataset_root is not None:
        if args.domain == "all":
            raise ValueError("--dataset-root requires a concrete --domain")
        base_spec = next(spec for spec in full.DATASETS if spec.domain == args.domain)
        dataset_root = args.dataset_root
        spec_updates: dict[str, Any] = {
            "name": args.dataset_label or base_spec.name,
            "root": dataset_root,
            "algorithm_inputs": args.algorithm_inputs
            or dataset_root / "algorithm_inputs" / f"{args.domain}_algorithm_inputs.json",
            "scenario_models": args.scenario_models
            or dataset_root / "scenario_models" / f"{args.domain}_public_scenario_models.json",
            "evaluation_references": args.evaluation_references
            or dataset_root / "evaluation_references" / f"{args.domain}_evaluation_references.json",
            "rule_library": args.rule_library
            or dataset_root / "rule_libraries" / f"full_{args.domain}_rule_library_qwen.json",
            "constraint_templates": args.constraint_templates
            or dataset_root / "constraint_templates" / "compiled_rule_constraint_templates.json",
        }
        specs = [replace(base_spec, **spec_updates)]

    for spec in specs:
        if args.domain != "all" and spec.domain != args.domain:
            continue
        rows, summary = run_dataset(
            spec,
            limit=args.limit,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            llm_cache=args.llm_cache,
            valid_resolver=valid_resolver,
            aviation_recall_guard=args.aviation_recall_guard,
        )
        write_dataset_outputs(
            spec,
            rows,
            summary,
            valid_resolver=valid_resolver,
            aviation_recall_guard=args.aviation_recall_guard,
            output_prefix=args.output_prefix,
        )
        all_summaries[spec.name] = summary

    print(json.dumps(all_summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
