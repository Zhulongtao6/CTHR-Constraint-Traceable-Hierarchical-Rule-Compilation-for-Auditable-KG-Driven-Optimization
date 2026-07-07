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


OUT_ROOT = full.OUT_ROOT
REPAIR_DIR = OUT_ROOT / "deepseek_single_chunk_repair"
REPAIRED_LIBRARY = OUT_ROOT / "full_aviation_rule_library_deepseek_repaired.json"
REPAIRED_SUMMARY = OUT_ROOT / "summary_deepseek_repaired.json"
REPAIR_RUN_SUMMARY = OUT_ROOT / "deepseek_single_chunk_repair_summary.json"


def safe_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def repair_checkpoint_path(batch_id: str) -> Path:
    return REPAIR_DIR / f"{batch_id}.json"


def is_ok(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return payload.get("status") == "ok"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
                    # Keep the original local KG context. Validation still requires
                    # extracted evidence to cite only the active single chunk.
                    "kg_nodes": original.get("kg_nodes", []),
                    "kg_edges": original.get("kg_edges", []),
                }
            )
    return repair_batches


def collect_original_ok_rows(original_batch_ids: list[str]) -> list[dict[str, Any]]:
    rows = []
    for batch_id in original_batch_ids:
        path = full.checkpoint_path("deepseek", batch_id)
        if not path.exists():
            continue
        row = load_json(path)
        if row.get("status") == "ok":
            rows.append(row)
    return rows


def collect_repair_rows(repair_batches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for batch in repair_batches:
        path = repair_checkpoint_path(batch["batch_id"])
        if not path.exists():
            continue
        try:
            rows.append(load_json(path))
        except json.JSONDecodeError:
            rows.append({"batch_id": batch["batch_id"], "status": "corrupt_checkpoint"})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair failed DeepSeek full-aviation batches by rerunning them as single-chunk batches.")
    parser.add_argument("--summary-path", type=Path, default=OUT_ROOT / "summary_deepseek.json")
    parser.add_argument("--input-index", type=Path, default=OUT_ROOT / "input_batches_index.json")
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--max-tokens", type=int, default=3500)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--deepseek-model", default="deepseek-v4-pro")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("Missing environment variable DEEPSEEK_API_KEY")

    full.apply_model_overrides(
        qwen_model="qwen-plus",
        deepseek_model=args.deepseek_model,
        glm_model="GLM-4.7",
    )
    system_prompt = full.PROMPT_PATH.read_text(encoding="utf-8")
    input_index = load_json(args.input_index)
    failed_ids = failed_batch_ids(args.summary_path)
    repair_batches = split_failed_batches(input_index, set(failed_ids))

    REPAIR_DIR.mkdir(parents=True, exist_ok=True)
    safe_write_json(
        REPAIR_DIR / "repair_manifest.json",
        {
            "generated_at": full.now(),
            "model": args.deepseek_model,
            "original_failed_batches": failed_ids,
            "num_original_failed_batches": len(failed_ids),
            "num_single_chunk_repair_batches": len(repair_batches),
            "source_summary": str(args.summary_path),
            "source_input_index": str(args.input_index),
        },
    )

    for idx, batch in enumerate(repair_batches, start=1):
        path = repair_checkpoint_path(batch["batch_id"])
        if is_ok(path) and not args.force:
            print(f"[{full.now()}] repair {idx}/{len(repair_batches)} {batch['batch_id']}: checkpoint ok, skip", flush=True)
            continue
        print(f"[{full.now()}] repair {idx}/{len(repair_batches)} {batch['batch_id']}: calling DeepSeek", flush=True)
        row = full.call_batch(
            provider_name="deepseek",
            batch=batch,
            system_prompt=system_prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
        row["original_batch_id"] = batch["original_batch_id"]
        safe_write_json(path, row)
        print(
            f"[{full.now()}] repair {batch['batch_id']}: {row.get('status')} "
            f"rules={row.get('validation', {}).get('rule_count')}",
            flush=True,
        )

    original_batch_ids = [batch["batch_id"] for batch in input_index["batches"]]
    original_ok_rows = collect_original_ok_rows(original_batch_ids)
    repair_rows = collect_repair_rows(repair_batches)
    merged_rows = original_ok_rows + [row for row in repair_rows if row.get("status") == "ok"]
    repaired_summary = full.summarize("deepseek", merged_rows)
    repair_failed_rows = [row for row in repair_rows if row.get("status") != "ok"]
    repaired_summary.update(
        {
            "model": args.deepseek_model,
            "repair_mode": "single_chunk_failed_batch_repair",
            "num_original_batches_total": len(original_batch_ids),
            "num_original_batches_ok_reused": len(original_ok_rows),
            "num_original_failed_batches": len(failed_ids),
            "num_single_chunk_repair_batches": len(repair_batches),
            "num_single_chunk_repair_batches_ok": len([row for row in repair_rows if row.get("status") == "ok"]),
            "num_single_chunk_repair_batches_failed": len(repair_failed_rows),
            "single_chunk_repair_failed": [
                {"batch_id": row.get("batch_id"), "original_batch_id": row.get("original_batch_id"), "error": row.get("error")}
                for row in repair_failed_rows
            ],
        }
    )
    library = full.build_library("deepseek", merged_rows, repaired_summary)
    library["model"] = args.deepseek_model
    library["source"] = "full Cognee aviation KG export with DeepSeek single-chunk repair for originally failed batches"
    safe_write_json(REPAIRED_LIBRARY, library)
    safe_write_json(REPAIRED_SUMMARY, repaired_summary)
    safe_write_json(
        REPAIR_RUN_SUMMARY,
        {
            "generated_at": full.now(),
            "repair_dir": str(REPAIR_DIR),
            "repaired_library": str(REPAIRED_LIBRARY),
            "repaired_summary": str(REPAIRED_SUMMARY),
            "summary": repaired_summary,
        },
    )
    print(json.dumps({"summary": repaired_summary, "library": str(REPAIRED_LIBRARY)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
