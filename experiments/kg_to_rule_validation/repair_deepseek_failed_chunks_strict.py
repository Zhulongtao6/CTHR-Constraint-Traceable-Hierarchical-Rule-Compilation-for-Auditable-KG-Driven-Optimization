from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import run_full_aviation_kg_rule_library as full  # noqa: E402
import run_llm_rule_validation as rv  # noqa: E402


OUT_ROOT = full.OUT_ROOT
OLD_REPAIR_DIR = OUT_ROOT / "deepseek_single_chunk_repair"
STRICT_REPAIR_DIR = OUT_ROOT / "deepseek_strict_chunk_retry"
STRICT_LIBRARY = OUT_ROOT / "full_aviation_rule_library_deepseek_strict_repaired.json"
STRICT_SUMMARY = OUT_ROOT / "summary_deepseek_strict_repaired.json"
STRICT_RUN_SUMMARY = OUT_ROOT / "deepseek_strict_chunk_retry_summary.json"


STRICT_SYSTEM_PROMPT = """
You convert exactly one aviation KG chunk into a CTHR rule-library JSON object.

Hard output rules:
- Return exactly one JSON object and nothing else.
- Do not use Markdown fences, explanations, comments, or trailing text.
- All strings must be valid JSON strings. Escape internal double quotes and line breaks.
- Use only IDs that appear in the input batch.
- If the chunk has no extractable normative rule, return rules=[] and put the chunk id in non_rule_chunk_ids.

Required JSON schema:
{
  "batch_id": "same batch_id as input",
  "rules": [
    {
      "rule_id": "stable_short_id",
      "name": "human-readable rule name",
      "domain": "aviation",
      "rule_type": "requirement|definition|calculation|parameter|exception|precedence|applicability|procedure_step",
      "source_chunk_ids": ["existing chunk id"],
      "source_node_ids": ["existing KG node id if relevant"],
      "guard": {"all": [{"field": "scenario/property name", "op": "eq|neq|gt|gte|lt|lte|in", "value": "condition value"}]},
      "constraints": [
        {
          "variable": "decision or regulatory quantity name",
          "op": ">=|<=|=|>|<|formula|text",
          "value": "number, expression, or text condition",
          "unit": "unit if stated, otherwise unknown",
          "source_quote": "short source phrase supporting the constraint",
          "evidence": {
            "chunk_ids": ["existing chunk id"],
            "kg_node_ids": ["existing KG node id if relevant"],
            "kg_edge_ids": ["existing KG edge id if relevant"]
          }
        }
      ],
      "relations": [
        {
          "type": "overrides|precedes|excludes|depends_on|applies_to|defines|uses_parameter|requires",
          "target": "target rule, entity, parameter, or condition",
          "source_quote": "short source phrase supporting the relation",
          "evidence": {
            "chunk_ids": ["existing chunk id"],
            "kg_node_ids": ["existing KG node id if relevant"],
            "kg_edge_ids": ["existing KG edge id if relevant"]
          }
        }
      ],
      "provenance": [
        {
          "document": "document/part name if known",
          "section": "section or page marker if known",
          "page": "page number if stated, otherwise unknown",
          "chunk_id": "existing chunk id"
        }
      ],
      "extraction_notes": "brief note explaining how the rule was grounded",
      "confidence": 0.0
    }
  ],
  "non_rule_chunk_ids": ["chunk ids with no extractable normative rule"]
}

Extraction rules:
- Do not invent numeric values, units, source chunks, sections, pages, node IDs, or edge IDs.
- Keep exception, precedence, dependency, and alternative-path relations explicit.
- If a formula is not reducible to a linear inequality, use op="formula".
- If the evidence is qualitative, use op="text".
- Preserve Chinese technical terms when they are the clearest variable names.
""".strip()


JSON_REPAIR_SYSTEM_PROMPT = """
You repair or regenerate one malformed CTHR rule-library JSON output.

Return exactly one valid JSON object and nothing else.
Use the source input batch as the ground truth.
If the malformed output cannot be repaired, regenerate the JSON from the source input batch.
If the source chunk has no extractable normative rule, return rules=[] and put the chunk id in non_rule_chunk_ids.
Do not invent IDs, values, units, source sections, or pages.
""".strip()


def safe_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def repair_checkpoint_path(batch_id: str) -> Path:
    return STRICT_REPAIR_DIR / f"{batch_id}.json"


def is_ok(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = load_json(path)
    except json.JSONDecodeError:
        return False
    return payload.get("status") == "ok"


def failed_single_chunk_ids(summary_path: Path) -> list[str]:
    summary = load_json(summary_path)
    failed = summary.get("summary", {}).get("single_chunk_repair_failed", [])
    return [row["batch_id"] for row in failed]


def split_failed_batches(input_index: dict[str, Any], failed_original_ids: set[str]) -> list[dict[str, Any]]:
    by_id = {batch["batch_id"]: batch for batch in input_index["batches"]}
    repair_batches: list[dict[str, Any]] = []
    for original_id in sorted(failed_original_ids):
        original = by_id[original_id]
        for idx, chunk in enumerate(original.get("chunks", [])):
            repair_batches.append(
                {
                    "batch_id": f"{original_id}_chunk_{idx:02d}",
                    "original_batch_id": original_id,
                    "domain": original.get("domain", "aviation"),
                    "source": original.get("source", "Cognee Neo4j CSV export"),
                    "chunks": [chunk],
                    "kg_nodes": original.get("kg_nodes", []),
                    "kg_edges": original.get("kg_edges", []),
                }
            )
    return repair_batches


def load_target_repair_batches(input_index_path: Path, previous_repair_summary_path: Path) -> list[dict[str, Any]]:
    failed_ids = failed_single_chunk_ids(previous_repair_summary_path)
    original_ids = {"_".join(batch_id.split("_")[:4]) for batch_id in failed_ids}
    input_index = load_json(input_index_path)
    all_single = split_failed_batches(input_index, original_ids)
    by_id = {batch["batch_id"]: batch for batch in all_single}
    return [by_id[batch_id] for batch_id in failed_ids]


def call_and_parse(
    provider: rv.Provider,
    system_prompt: str,
    user_payload: dict[str, Any],
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    raw, meta = rv.call_provider(
        provider=provider,
        system_prompt=system_prompt,
        user_payload=user_payload,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    parsed = rv.extract_json(raw)
    return parsed, raw, meta


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def first_text(*values: Any, default: str = "") -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return default


def canonical_evidence(rule: dict[str, Any], batch: dict[str, Any]) -> dict[str, Any]:
    input_chunk_ids = [chunk["chunk_id"] for chunk in batch["chunks"]]
    ev = rule.get("evidence") if isinstance(rule.get("evidence"), dict) else {}
    chunk_ids = as_list(ev.get("chunk_ids"))
    chunk_ids.extend(as_list(rule.get("source_chunk_ids")))
    chunk_ids.extend(as_list(rule.get("chunk_ids")))
    chunk_ids.extend(as_list(rule.get("chunk_id")))
    grounded_chunk_ids = [str(cid) for cid in chunk_ids if str(cid) in input_chunk_ids]
    if not grounded_chunk_ids and len(input_chunk_ids) == 1:
        grounded_chunk_ids = input_chunk_ids
    return {
        "chunk_ids": grounded_chunk_ids,
        "kg_node_ids": [str(x) for x in as_list(ev.get("kg_node_ids"))],
        "kg_edge_ids": [str(x) for x in as_list(ev.get("kg_edge_ids"))],
    }


def canonical_constraint(rule: dict[str, Any], batch: dict[str, Any]) -> dict[str, Any]:
    text = first_text(
        rule.get("rule"),
        rule.get("requirement"),
        rule.get("action"),
        rule.get("description"),
        rule.get("name"),
        default="qualitative regulatory requirement",
    )
    return {
        "variable": first_text(rule.get("variable"), rule.get("name"), default="regulatory_requirement"),
        "op": first_text(rule.get("op"), default="text"),
        "value": text,
        "unit": first_text(rule.get("unit"), default="unknown"),
        "source_quote": text[:220],
        "evidence": canonical_evidence(rule, batch),
    }


def canonical_provenance(rule: dict[str, Any], batch: dict[str, Any]) -> list[dict[str, Any]]:
    provenance = rule.get("provenance")
    if isinstance(provenance, list) and provenance:
        out = []
        for item in provenance:
            if not isinstance(item, dict):
                continue
            ev = canonical_evidence({**rule, "chunk_id": item.get("chunk_id")}, batch)
            chunk_id = ev["chunk_ids"][0] if ev["chunk_ids"] else ""
            out.append(
                {
                    "document": first_text(item.get("document"), rule.get("original_source"), default=batch.get("source", "unknown")),
                    "section": first_text(item.get("section"), rule.get("source_section"), default="unknown"),
                    "page": first_text(item.get("page"), rule.get("source_page"), rule.get("original_page"), rule.get("page"), default="unknown"),
                    "chunk_id": chunk_id,
                }
            )
        if out:
            return out
    ev = canonical_evidence(rule, batch)
    chunk_id = ev["chunk_ids"][0] if ev["chunk_ids"] else ""
    document = batch["chunks"][0].get("document", batch.get("source", "unknown")) if batch.get("chunks") else batch.get("source", "unknown")
    return [
        {
            "document": first_text(rule.get("document"), rule.get("original_source"), default=document),
            "section": first_text(rule.get("source_section"), default="unknown"),
            "page": first_text(rule.get("source_page"), rule.get("original_page"), rule.get("page"), default=batch["chunks"][0].get("page_hint", "unknown") if batch.get("chunks") else "unknown"),
            "chunk_id": chunk_id,
        }
    ]


def canonicalize_rule_batch(parsed: dict[str, Any], batch: dict[str, Any]) -> dict[str, Any]:
    rules = parsed.get("rules", []) if isinstance(parsed, dict) else []
    if not isinstance(rules, list):
        rules = []
    input_chunk_ids = [chunk["chunk_id"] for chunk in batch["chunks"]]
    canonical_rules: list[dict[str, Any]] = []
    for idx, rule in enumerate(rules, start=1):
        if not isinstance(rule, dict):
            continue
        evidence = canonical_evidence(rule, batch)
        source_chunk_ids = evidence["chunk_ids"] or input_chunk_ids[:1]
        constraints = rule.get("constraints")
        if isinstance(constraints, list) and constraints:
            canonical_constraints = []
            for con in constraints:
                if not isinstance(con, dict):
                    continue
                con_ev = con.get("evidence") if isinstance(con.get("evidence"), dict) else {}
                canonical_constraints.append(
                    {
                        "variable": first_text(con.get("variable"), default="regulatory_requirement"),
                        "op": first_text(con.get("op"), default="text"),
                        "value": con.get("value", ""),
                        "unit": first_text(con.get("unit"), default="unknown"),
                        "source_quote": first_text(con.get("source_quote"), con.get("value"), default=""),
                        "evidence": {
                            "chunk_ids": [str(x) for x in as_list(con_ev.get("chunk_ids")) if str(x) in input_chunk_ids] or source_chunk_ids,
                            "kg_node_ids": [str(x) for x in as_list(con_ev.get("kg_node_ids"))],
                            "kg_edge_ids": [str(x) for x in as_list(con_ev.get("kg_edge_ids"))],
                        },
                    }
                )
            if not canonical_constraints:
                canonical_constraints = [canonical_constraint(rule, batch)]
        else:
            canonical_constraints = [canonical_constraint(rule, batch)]

        relations = rule.get("relations") if isinstance(rule.get("relations"), list) else []
        canonical_relations = []
        for rel in relations:
            if not isinstance(rel, dict):
                continue
            rel_ev = rel.get("evidence") if isinstance(rel.get("evidence"), dict) else {}
            canonical_relations.append(
                {
                    "type": first_text(rel.get("type"), default="requires"),
                    "target": first_text(rel.get("target"), default="regulatory_requirement"),
                    "source_quote": first_text(rel.get("source_quote"), rel.get("target"), default=""),
                    "evidence": {
                        "chunk_ids": [str(x) for x in as_list(rel_ev.get("chunk_ids")) if str(x) in input_chunk_ids] or source_chunk_ids,
                        "kg_node_ids": [str(x) for x in as_list(rel_ev.get("kg_node_ids"))],
                        "kg_edge_ids": [str(x) for x in as_list(rel_ev.get("kg_edge_ids"))],
                    },
                }
            )

        condition = first_text(rule.get("condition"))
        guard = rule.get("guard") if isinstance(rule.get("guard"), dict) else {}
        if not guard and condition:
            guard = {"all": [{"field": "text_condition", "op": "eq", "value": condition}]}
        canonical_rules.append(
            {
                "rule_id": first_text(rule.get("rule_id"), rule.get("id"), default=f"{batch['batch_id']}_rule_{idx}"),
                "name": first_text(rule.get("name"), rule.get("rule"), rule.get("requirement"), rule.get("action"), default=f"Rule {idx}"),
                "domain": "aviation",
                "rule_type": first_text(rule.get("rule_type"), rule.get("rule_class"), default="requirement"),
                "source_chunk_ids": source_chunk_ids,
                "source_node_ids": [str(x) for x in as_list(rule.get("source_node_ids"))],
                "guard": guard,
                "constraints": canonical_constraints,
                "relations": canonical_relations,
                "provenance": canonical_provenance(rule, batch),
                "extraction_notes": first_text(rule.get("extraction_notes"), default="Canonicalized from DeepSeek strict-retry output."),
                "confidence": rule.get("confidence", 0.7),
            }
        )
    non_rule_ids = [str(x) for x in as_list(parsed.get("non_rule_chunk_ids") if isinstance(parsed, dict) else []) if str(x) in input_chunk_ids]
    if not canonical_rules and not non_rule_ids:
        non_rule_ids = input_chunk_ids
    return {
        "batch_id": batch["batch_id"],
        "rules": canonical_rules,
        "non_rule_chunk_ids": non_rule_ids,
    }


def strict_call_batch(
    batch: dict[str, Any],
    provider: rv.Provider,
    temperature: float,
    max_tokens: int,
    timeout: int,
    repair_max_tokens: int,
) -> dict[str, Any]:
    started_at = full.now()
    raw_attempts: list[dict[str, Any]] = []
    try:
        parsed, raw, meta = call_and_parse(
            provider=provider,
            system_prompt=STRICT_SYSTEM_PROMPT,
            user_payload=batch,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        raw_attempts.append({"stage": "strict_extract", "raw_text": raw, "parsed": True})
        canonical = canonicalize_rule_batch(parsed, batch)
        return {
            "status": "ok",
            "batch_id": batch["batch_id"],
            "original_batch_id": batch.get("original_batch_id"),
            "provider": meta,
            "model_family": "deepseek",
            "started_at": started_at,
            "finished_at": full.now(),
            "input_chunk_ids": [chunk["chunk_id"] for chunk in batch["chunks"]],
            "repair_stage": "strict_extract",
            "raw_attempts": raw_attempts,
            "raw_text": raw,
            "rule_batch_raw": parsed,
            "rule_batch": canonical,
            "schema_normalized": True,
            "validation": full.validate_batch(canonical, batch),
        }
    except Exception as exc:  # noqa: BLE001
        raw_attempts.append({"stage": "strict_extract", "parsed": False, "error": str(exc)})

    repair_payload = {
        "source_input_batch": batch,
        "malformed_output_or_error": raw_attempts[-1],
        "required_batch_id": batch["batch_id"],
        "required_chunk_ids": [chunk["chunk_id"] for chunk in batch["chunks"]],
    }
    try:
        parsed, raw, meta = call_and_parse(
            provider=provider,
            system_prompt=JSON_REPAIR_SYSTEM_PROMPT,
            user_payload=repair_payload,
            temperature=0.0,
            max_tokens=repair_max_tokens,
            timeout=timeout,
        )
        raw_attempts.append({"stage": "json_repair", "raw_text": raw, "parsed": True})
        canonical = canonicalize_rule_batch(parsed, batch)
        return {
            "status": "ok",
            "batch_id": batch["batch_id"],
            "original_batch_id": batch.get("original_batch_id"),
            "provider": meta,
            "model_family": "deepseek",
            "started_at": started_at,
            "finished_at": full.now(),
            "input_chunk_ids": [chunk["chunk_id"] for chunk in batch["chunks"]],
            "repair_stage": "json_repair",
            "raw_attempts": raw_attempts,
            "raw_text": raw,
            "rule_batch_raw": parsed,
            "rule_batch": canonical,
            "schema_normalized": True,
            "validation": full.validate_batch(canonical, batch),
        }
    except Exception as exc:  # noqa: BLE001
        raw_attempts.append({"stage": "json_repair", "parsed": False, "error": str(exc)})
        return {
            "status": "failed",
            "batch_id": batch["batch_id"],
            "original_batch_id": batch.get("original_batch_id"),
            "model_family": "deepseek",
            "started_at": started_at,
            "finished_at": full.now(),
            "input_chunk_ids": [chunk["chunk_id"] for chunk in batch["chunks"]],
            "error": str(exc),
            "raw_attempts": raw_attempts,
        }


def collect_original_ok_rows(input_index: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for batch in input_index["batches"]:
        path = full.checkpoint_path("deepseek", batch["batch_id"])
        if not path.exists():
            continue
        row = load_json(path)
        if row.get("status") == "ok":
            rows.append(row)
    return rows


def collect_old_repair_ok_rows() -> list[dict[str, Any]]:
    rows = []
    if not OLD_REPAIR_DIR.exists():
        return rows
    for path in sorted(OLD_REPAIR_DIR.glob("*.json")):
        if path.name == "repair_manifest.json":
            continue
        row = load_json(path)
        if row.get("status") == "ok":
            rows.append(row)
    return rows


def collect_strict_rows(target_batches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    batch_by_id = {batch["batch_id"]: batch for batch in target_batches}
    for batch in target_batches:
        path = repair_checkpoint_path(batch["batch_id"])
        if not path.exists():
            continue
        try:
            row = load_json(path)
            if row.get("status") == "ok":
                source_batch = batch_by_id[row["batch_id"]]
                raw_batch = row.get("rule_batch_raw") or row.get("rule_batch") or {}
                canonical = canonicalize_rule_batch(raw_batch, source_batch)
                row["rule_batch"] = canonical
                row["schema_normalized"] = True
                row["validation"] = full.validate_batch(canonical, source_batch)
            rows.append(row)
        except json.JSONDecodeError:
            rows.append({"batch_id": batch["batch_id"], "status": "corrupt_checkpoint"})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Strictly retry failed DeepSeek single-chunk aviation KG-to-rule batches.")
    parser.add_argument("--input-index", type=Path, default=OUT_ROOT / "input_batches_index.json")
    parser.add_argument("--previous-repair-summary", type=Path, default=OUT_ROOT / "deepseek_single_chunk_repair_summary.json")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=3500)
    parser.add_argument("--repair-max-tokens", type=int, default=3500)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--deepseek-model", default="deepseek-v4-pro")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("Missing environment variable DEEPSEEK_API_KEY")

    full.apply_model_overrides(qwen_model="qwen-plus", deepseek_model=args.deepseek_model, glm_model="GLM-4.7")
    provider = rv.PROVIDERS["deepseek"]
    target_batches = load_target_repair_batches(args.input_index, args.previous_repair_summary)
    input_index = load_json(args.input_index)

    STRICT_REPAIR_DIR.mkdir(parents=True, exist_ok=True)
    safe_write_json(
        STRICT_REPAIR_DIR / "strict_repair_manifest.json",
        {
            "generated_at": full.now(),
            "model": args.deepseek_model,
            "num_target_failed_single_chunks": len(target_batches),
            "target_batch_ids": [batch["batch_id"] for batch in target_batches],
            "source_input_index": str(args.input_index),
            "source_previous_repair_summary": str(args.previous_repair_summary),
        },
    )

    for idx, batch in enumerate(target_batches, start=1):
        path = repair_checkpoint_path(batch["batch_id"])
        if is_ok(path) and not args.force:
            print(f"[{full.now()}] strict {idx}/{len(target_batches)} {batch['batch_id']}: checkpoint ok, skip", flush=True)
            continue
        print(f"[{full.now()}] strict {idx}/{len(target_batches)} {batch['batch_id']}: calling DeepSeek", flush=True)
        row = strict_call_batch(
            batch=batch,
            provider=provider,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            repair_max_tokens=args.repair_max_tokens,
        )
        safe_write_json(path, row)
        print(
            f"[{full.now()}] strict {batch['batch_id']}: {row.get('status')} "
            f"stage={row.get('repair_stage')} rules={row.get('validation', {}).get('rule_count')}",
            flush=True,
        )

    original_ok_rows = collect_original_ok_rows(input_index)
    old_repair_ok_rows = collect_old_repair_ok_rows()
    strict_rows = collect_strict_rows(target_batches)
    strict_ok_rows = [row for row in strict_rows if row.get("status") == "ok"]
    merged_rows = original_ok_rows + old_repair_ok_rows + strict_ok_rows

    summary = full.summarize("deepseek", merged_rows)
    strict_failed_rows = [row for row in strict_rows if row.get("status") != "ok"]
    summary.update(
        {
            "model": args.deepseek_model,
            "repair_mode": "strict_single_chunk_retry_with_json_repair",
            "num_original_batches_total": len(input_index["batches"]),
            "num_original_batches_ok_reused": len(original_ok_rows),
            "num_previous_single_chunk_repairs_ok_reused": len(old_repair_ok_rows),
            "num_strict_retry_targets": len(target_batches),
            "num_strict_retry_ok": len(strict_ok_rows),
            "num_strict_retry_failed": len(strict_failed_rows),
            "strict_retry_failed": [
                {
                    "batch_id": row.get("batch_id"),
                    "original_batch_id": row.get("original_batch_id"),
                    "error": row.get("error"),
                }
                for row in strict_failed_rows
            ],
        }
    )
    library = full.build_library("deepseek", merged_rows, summary)
    library["model"] = args.deepseek_model
    library["source"] = "full Cognee aviation KG export with strict DeepSeek retry for failed single chunks"
    safe_write_json(STRICT_LIBRARY, library)
    safe_write_json(STRICT_SUMMARY, summary)
    safe_write_json(
        STRICT_RUN_SUMMARY,
        {
            "generated_at": full.now(),
            "strict_repair_dir": str(STRICT_REPAIR_DIR),
            "strict_library": str(STRICT_LIBRARY),
            "strict_summary": str(STRICT_SUMMARY),
            "summary": summary,
        },
    )
    print(json.dumps({"summary": summary, "library": str(STRICT_LIBRARY)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
