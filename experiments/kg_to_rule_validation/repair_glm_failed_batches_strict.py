from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import run_full_aviation_kg_rule_library as full  # noqa: E402
import run_llm_rule_validation as rv  # noqa: E402
import repair_deepseek_failed_chunks_strict as strict  # noqa: E402


OUT_ROOT = full.OUT_ROOT
STRICT_REPAIR_DIR = OUT_ROOT / "glm_strict_chunk_retry"
STRICT_LIBRARY = OUT_ROOT / "full_aviation_rule_library_glm_strict_repaired.json"
STRICT_SUMMARY = OUT_ROOT / "summary_glm_strict_repaired.json"
STRICT_RUN_SUMMARY = OUT_ROOT / "glm_strict_chunk_retry_summary.json"


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


def failed_batch_ids(summary_path: Path) -> list[str]:
    summary = load_json(summary_path)
    return [row["batch_id"] for row in summary.get("failed_batches", [])]


def split_failed_batches(input_index: dict[str, Any], failed_ids: set[str]) -> list[dict[str, Any]]:
    by_id = {batch["batch_id"]: batch for batch in input_index["batches"]}
    repair_batches: list[dict[str, Any]] = []
    for original_id in sorted(failed_ids):
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
            system_prompt=strict.STRICT_SYSTEM_PROMPT,
            user_payload=batch,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        raw_attempts.append({"stage": "strict_extract", "raw_text": raw, "parsed": True})
        canonical = strict.canonicalize_rule_batch(parsed, batch)
        return {
            "status": "ok",
            "batch_id": batch["batch_id"],
            "original_batch_id": batch.get("original_batch_id"),
            "provider": meta,
            "model_family": "glm",
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
            system_prompt=strict.JSON_REPAIR_SYSTEM_PROMPT,
            user_payload=repair_payload,
            temperature=0.0,
            max_tokens=repair_max_tokens,
            timeout=timeout,
        )
        raw_attempts.append({"stage": "json_repair", "raw_text": raw, "parsed": True})
        canonical = strict.canonicalize_rule_batch(parsed, batch)
        return {
            "status": "ok",
            "batch_id": batch["batch_id"],
            "original_batch_id": batch.get("original_batch_id"),
            "provider": meta,
            "model_family": "glm",
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
            "model_family": "glm",
            "started_at": started_at,
            "finished_at": full.now(),
            "input_chunk_ids": [chunk["chunk_id"] for chunk in batch["chunks"]],
            "error": str(exc),
            "raw_attempts": raw_attempts,
        }


def collect_original_ok_rows(input_index: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for batch in input_index["batches"]:
        path = full.checkpoint_path("glm", batch["batch_id"])
        if not path.exists():
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
                canonical = strict.canonicalize_rule_batch(raw_batch, source_batch)
                row["rule_batch"] = canonical
                row["schema_normalized"] = True
                row["validation"] = full.validate_batch(canonical, source_batch)
            rows.append(row)
        except json.JSONDecodeError:
            rows.append({"batch_id": batch["batch_id"], "status": "corrupt_checkpoint"})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Strictly retry failed GLM full-aviation KG-to-rule batches as single chunks.")
    parser.add_argument("--input-index", type=Path, default=OUT_ROOT / "input_batches_index.json")
    parser.add_argument("--summary-path", type=Path, default=OUT_ROOT / "summary_glm.json")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=3200)
    parser.add_argument("--repair-max-tokens", type=int, default=3200)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--glm-model", default="GLM-4.7")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--rate-limit-sleep-seconds", type=float, default=120.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("GLM_API_KEY"):
        raise RuntimeError("Missing environment variable GLM_API_KEY")

    full.apply_model_overrides(qwen_model="qwen-plus", deepseek_model="deepseek-v4-pro", glm_model=args.glm_model)
    provider = rv.PROVIDERS["glm"]
    input_index = load_json(args.input_index)
    failed_ids = failed_batch_ids(args.summary_path)
    target_batches = split_failed_batches(input_index, set(failed_ids))

    STRICT_REPAIR_DIR.mkdir(parents=True, exist_ok=True)
    safe_write_json(
        STRICT_REPAIR_DIR / "strict_repair_manifest.json",
        {
            "generated_at": full.now(),
            "model": args.glm_model,
            "original_failed_batches": failed_ids,
            "num_original_failed_batches": len(failed_ids),
            "num_single_chunk_targets": len(target_batches),
            "target_batch_ids": [batch["batch_id"] for batch in target_batches],
            "source_input_index": str(args.input_index),
            "source_summary": str(args.summary_path),
        },
    )

    for idx, batch in enumerate(target_batches, start=1):
        path = repair_checkpoint_path(batch["batch_id"])
        if is_ok(path) and not args.force:
            print(f"[{full.now()}] glm strict {idx}/{len(target_batches)} {batch['batch_id']}: checkpoint ok, skip", flush=True)
            continue
        print(f"[{full.now()}] glm strict {idx}/{len(target_batches)} {batch['batch_id']}: calling GLM", flush=True)
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
            f"[{full.now()}] glm strict {batch['batch_id']}: {row.get('status')} "
            f"stage={row.get('repair_stage')} rules={row.get('validation', {}).get('rule_count')}",
            flush=True,
        )
        err = str(row.get("error", ""))
        if "429" in err or "速率限制" in err:
            print(
                f"[{full.now()}] glm strict {batch['batch_id']}: rate limited; "
                f"sleeping {args.rate_limit_sleep_seconds:.1f}s before continuing",
                flush=True,
            )
            time.sleep(max(0.0, args.rate_limit_sleep_seconds))
        elif args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    original_ok_rows = collect_original_ok_rows(input_index)
    strict_rows = collect_strict_rows(target_batches)
    strict_ok_rows = [row for row in strict_rows if row.get("status") == "ok"]
    merged_rows = original_ok_rows + strict_ok_rows
    summary = full.summarize("glm", merged_rows)
    strict_failed_rows = [row for row in strict_rows if row.get("status") != "ok"]
    summary.update(
        {
            "model": args.glm_model,
            "repair_mode": "strict_single_chunk_retry_with_json_repair",
            "num_original_batches_total": len(input_index["batches"]),
            "num_original_batches_ok_reused": len(original_ok_rows),
            "num_original_failed_batches": len(failed_ids),
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
    library = full.build_library("glm", merged_rows, summary)
    library["model"] = args.glm_model
    library["source"] = "full Cognee aviation KG export with strict GLM retry for failed batches"
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
