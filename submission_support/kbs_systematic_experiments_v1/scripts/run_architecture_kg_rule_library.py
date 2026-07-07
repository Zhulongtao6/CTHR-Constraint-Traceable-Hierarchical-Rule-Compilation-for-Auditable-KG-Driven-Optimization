from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CTHR_ROOT = ROOT.parents[1]
KG_VALIDATION_DIR = CTHR_ROOT / "experiments" / "kg_to_rule_validation"
DEFAULT_EXPORT_DIR = Path(r"D:\paper\Neurosymbolic\cognee_export\architecture_export_neo4j")
PROMPT_PATH = ROOT / "scripts" / "architecture_kg_to_rule_prompt.txt"
OUT_ROOT = ROOT / "results" / "kg_to_rule_library" / "architecture"

sys.path.insert(0, str(KG_VALIDATION_DIR))
import run_full_aviation_kg_rule_library as base  # noqa: E402


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def make_architecture_batches(
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    doc_by_dataset: dict[str, str],
    batch_size: int,
    chunk_char_limit: int,
    max_neighbors: int,
) -> list[dict[str, Any]]:
    batches = base.make_batches(
        nodes=nodes,
        edges=edges,
        doc_by_dataset=doc_by_dataset,
        batch_size=batch_size,
        chunk_char_limit=chunk_char_limit,
        max_neighbors=max_neighbors,
    )
    for idx, batch in enumerate(batches):
        batch["batch_id"] = f"arch_full_batch_{idx:04d}"
        batch["domain"] = "architecture"
        batch["source"] = "Cognee Neo4j CSV export for ADA, IBC, and IFC architecture KG"
    return batches


def checkpoint_path(provider_name: str, batch_id: str) -> Path:
    return OUT_ROOT / provider_name / f"{batch_id}.json"


def collect_rows(provider_name: str, batch_ids: list[str]) -> list[dict[str, Any]]:
    rows = []
    for batch_id in batch_ids:
        path = checkpoint_path(provider_name, batch_id)
        if not path.exists():
            continue
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            rows.append({"batch_id": batch_id, "status": "corrupt_checkpoint"})
    return rows


def is_completed(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return payload.get("status") == "ok"


def build_architecture_library(provider_name: str, rows: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    library = base.build_library(provider_name, rows, summary)
    library["source"] = "full Cognee architecture KG export: nodes.csv and edges.csv"
    library["domain"] = "architecture"
    library["submission_support_metadata"] = {
        "output_root": str(OUT_ROOT),
        "prompt": str(PROMPT_PATH),
        "note": "Generated from the architecture KG export for formal KBS support experiments.",
    }
    return library


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert the full Cognee architecture KG export into CTHR rule libraries.")
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--providers", default="qwen,deepseek")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--chunk-char-limit", type=int, default=1800)
    parser.add_argument("--max-neighbors", type=int, default=18)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--qwen-model", default="qwen-plus")
    parser.add_argument("--deepseek-model", default="deepseek-v4-pro")
    parser.add_argument("--glm-model", default="GLM-4.7")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit-batches", type=int, default=None)
    args = parser.parse_args()

    base.OUT_ROOT = OUT_ROOT
    base.apply_model_overrides(args.qwen_model, args.deepseek_model, args.glm_model)
    provider_names = base.normalize_provider_names(args.providers)
    nodes, edges, doc_by_dataset = base.load_kg(args.export_dir)
    batches = make_architecture_batches(
        nodes=nodes,
        edges=edges,
        doc_by_dataset=doc_by_dataset,
        batch_size=args.batch_size,
        chunk_char_limit=args.chunk_char_limit,
        max_neighbors=args.max_neighbors,
    )
    if args.limit_batches is not None:
        batches = batches[: args.limit_batches]
    batch_ids = [batch["batch_id"] for batch in batches]
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    base.safe_write_json(
        OUT_ROOT / "manifest.json",
        {
            "generated_at": now(),
            "export_dir": str(args.export_dir),
            "num_batches": len(batches),
            "batch_size": args.batch_size,
            "chunk_char_limit": args.chunk_char_limit,
            "max_neighbors": args.max_neighbors,
            "providers": provider_names,
            "models": {name: base.rv.PROVIDERS[name].model for name in provider_names},
            "prompt": str(PROMPT_PATH),
            "note": "API keys are read from environment variables and are not written to outputs.",
        },
    )
    base.safe_write_json(OUT_ROOT / "input_batches_index.json", {"batches": batches})

    for provider_name in provider_names:
        key_ok, key_name = base.ensure_provider_env(provider_name)
        if not key_ok:
            for batch in batches:
                path = checkpoint_path(provider_name, batch["batch_id"])
                if path.exists() and not args.force:
                    continue
                base.safe_write_json(
                    path,
                    {
                        "status": "skipped_missing_key",
                        "batch_id": batch["batch_id"],
                        "model_family": provider_name,
                        "required_env": key_name,
                        "input_chunk_ids": [chunk["chunk_id"] for chunk in batch["chunks"]],
                        "finished_at": now(),
                    },
                )
            print(f"[{now()}] {provider_name}: skipped because {key_name} is missing")
            continue
        for idx, batch in enumerate(batches, start=1):
            path = checkpoint_path(provider_name, batch["batch_id"])
            if is_completed(path) and not args.force:
                print(f"[{now()}] {provider_name} {idx}/{len(batches)} {batch['batch_id']}: checkpoint ok, skip")
                continue
            print(f"[{now()}] {provider_name} {idx}/{len(batches)} {batch['batch_id']}: calling API")
            row = base.call_batch(
                provider_name=provider_name,
                batch=batch,
                system_prompt=system_prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
            base.safe_write_json(path, row)
            print(
                f"[{now()}] {provider_name} {batch['batch_id']}: "
                f"{row.get('status')} rules={row.get('validation', {}).get('rule_count')}"
            )

    summaries = []
    for provider_name in provider_names:
        rows = collect_rows(provider_name, batch_ids)
        summary = base.summarize(provider_name, rows)
        summaries.append(summary)
        library = build_architecture_library(provider_name, rows, summary)
        base.safe_write_json(OUT_ROOT / f"full_architecture_rule_library_{provider_name}.json", library)
        base.safe_write_json(OUT_ROOT / f"summary_{provider_name}.json", summary)

    comparison = {
        "generated_at": now(),
        "export_dir": str(args.export_dir),
        "num_batches": len(batches),
        "num_chunks": sum(len(batch["chunks"]) for batch in batches),
        "summaries": summaries,
    }
    base.safe_write_json(OUT_ROOT / "architecture_model_comparison_summary.json", comparison)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
