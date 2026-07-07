from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote


THIS_DIR = Path(__file__).resolve().parent
CTHR_ROOT = THIS_DIR.parents[1]
PAPER_DIR = CTHR_ROOT / "paper"
DEFAULT_EXPORT_DIR = Path(r"D:\paper\Neurosymbolic\cognee_export\export_neo4j")
PROMPT_PATH = THIS_DIR / "full_aviation_kg_to_rule_prompt.txt"
OUT_ROOT = PAPER_DIR / "full_aviation_kg_rule_library_model_comparison"

sys.path.insert(0, str(THIS_DIR))
import run_llm_rule_validation as rv  # noqa: E402


PROVIDER_ALIASES = {
    "tongyi": "qwen",
    "qwen": "qwen",
    "deepseek": "deepseek",
    "glm": "glm",
}


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def safe_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def normalize_provider_names(raw: str) -> list[str]:
    names = []
    for item in raw.split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key not in PROVIDER_ALIASES:
            raise ValueError(f"Unknown provider: {item}")
        name = PROVIDER_ALIASES[key]
        if name not in names:
            names.append(name)
    return names


def apply_model_overrides(qwen_model: str, deepseek_model: str, glm_model: str) -> None:
    rv.PROVIDERS["qwen"] = rv.Provider(
        name=rv.PROVIDERS["qwen"].name,
        env_key=rv.PROVIDERS["qwen"].env_key,
        url=rv.PROVIDERS["qwen"].url,
        model=qwen_model,
    )
    rv.PROVIDERS["deepseek"] = rv.Provider(
        name=rv.PROVIDERS["deepseek"].name,
        env_key=rv.PROVIDERS["deepseek"].env_key,
        url=rv.PROVIDERS["deepseek"].url,
        model=deepseek_model,
    )
    rv.PROVIDERS["glm"] = rv.Provider(
        name=rv.PROVIDERS["glm"].name,
        env_key=rv.PROVIDERS["glm"].env_key,
        url=rv.PROVIDERS["glm"].url,
        model=glm_model,
    )


def ensure_provider_env(provider_name: str) -> tuple[bool, str]:
    if provider_name == "qwen":
        if os.environ.get("QWEN_API_KEY"):
            return True, "QWEN_API_KEY"
        if os.environ.get("DASHSCOPE_API_KEY"):
            os.environ["QWEN_API_KEY"] = os.environ["DASHSCOPE_API_KEY"]
            return True, "DASHSCOPE_API_KEY"
        return False, "QWEN_API_KEY or DASHSCOPE_API_KEY"
    provider = rv.PROVIDERS[provider_name]
    if os.environ.get(provider.env_key):
        return True, provider.env_key
    return False, provider.env_key


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def short_text(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def page_hint(text: str) -> str:
    match = re.search(r"Page\s+(\d+)", text or "", flags=re.IGNORECASE)
    return match.group(1) if match else "unknown"


def load_kg(export_dir: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    nodes_raw = read_csv(export_dir / "nodes.csv")
    edges_raw = read_csv(export_dir / "edges.csv")
    nodes = {row["id"]: row for row in nodes_raw}
    doc_by_dataset: dict[str, str] = {}
    for row in nodes_raw:
        if row.get("label") == "TextDocument":
            doc_by_dataset[row.get("dataset", "")] = unquote(row.get("name", "")).replace("_", " ")

    edges = []
    for idx, row in enumerate(edges_raw):
        edge = dict(row)
        edge["edge_id"] = f"kg_edge_{idx:05d}"
        edges.append(edge)
    return nodes, edges, doc_by_dataset


def chunk_neighbors(
    chunk_id: str,
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    max_neighbors: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kg_edges = []
    neighbor_ids = []
    for edge in edges:
        if edge.get("src_id") == chunk_id or edge.get("dst_id") == chunk_id:
            other_id = edge["dst_id"] if edge.get("src_id") == chunk_id else edge["src_id"]
            other = nodes.get(other_id, {})
            kg_edges.append(
                {
                    "id": edge["edge_id"],
                    "source": edge.get("src_id"),
                    "relation": edge.get("rel_type"),
                    "target": edge.get("dst_id"),
                    "neighbor_name": other.get("name", ""),
                    "neighbor_label": other.get("label", ""),
                }
            )
            neighbor_ids.append(other_id)
        if len(kg_edges) >= max_neighbors:
            break
    kg_nodes = []
    seen = set()
    for node_id in neighbor_ids:
        if node_id in seen:
            continue
        seen.add(node_id)
        node = nodes.get(node_id, {})
        kg_nodes.append(
            {
                "id": node_id,
                "label": node.get("label", ""),
                "name": node.get("name", ""),
                "description": short_text(node.get("description", ""), 300),
            }
        )
        if len(kg_nodes) >= max_neighbors:
            break
    return kg_nodes, kg_edges


def make_batches(
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    doc_by_dataset: dict[str, str],
    batch_size: int,
    chunk_char_limit: int,
    max_neighbors: int,
) -> list[dict[str, Any]]:
    chunks = [
        row
        for row in nodes.values()
        if row.get("label") == "DocumentChunk" and short_text(row.get("text_snippet", ""), 50)
    ]
    chunks.sort(key=lambda row: (row.get("dataset", ""), row.get("id", "")))
    batches = []
    for start in range(0, len(chunks), batch_size):
        selected = chunks[start : start + batch_size]
        batch_chunks = []
        all_nodes: dict[str, dict[str, Any]] = {}
        all_edges: dict[str, dict[str, Any]] = {}
        for chunk in selected:
            chunk_id = chunk["id"]
            kg_nodes, kg_edges = chunk_neighbors(chunk_id, nodes, edges, max_neighbors)
            for node in kg_nodes:
                all_nodes[node["id"]] = node
            for edge in kg_edges:
                all_edges[edge["id"]] = edge
            batch_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "dataset": chunk.get("dataset", ""),
                    "document": doc_by_dataset.get(chunk.get("dataset", ""), chunk.get("dataset", "")),
                    "page_hint": page_hint(chunk.get("text_snippet", "")),
                    "text": short_text(chunk.get("text_snippet", ""), chunk_char_limit),
                }
            )
        batch_id = f"avi_full_batch_{len(batches):04d}"
        batches.append(
            {
                "batch_id": batch_id,
                "domain": "aviation",
                "source": "Cognee Neo4j CSV export",
                "chunks": batch_chunks,
                "kg_nodes": list(all_nodes.values()),
                "kg_edges": list(all_edges.values()),
            }
        )
    return batches


def checkpoint_path(provider_name: str, batch_id: str) -> Path:
    return OUT_ROOT / provider_name / f"{batch_id}.json"


def is_completed(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return payload.get("status") == "ok"


def call_batch(
    provider_name: str,
    batch: dict[str, Any],
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    provider = rv.PROVIDERS[provider_name]
    started_at = now()
    try:
        raw, meta = rv.call_provider(
            provider=provider,
            system_prompt=system_prompt,
            user_payload=batch,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        parsed = rv.extract_json(raw)
        return {
            "status": "ok",
            "batch_id": batch["batch_id"],
            "provider": meta,
            "model_family": provider_name,
            "started_at": started_at,
            "finished_at": now(),
            "input_chunk_ids": [chunk["chunk_id"] for chunk in batch["chunks"]],
            "raw_text": raw,
            "rule_batch": parsed,
            "validation": validate_batch(parsed, batch),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "failed",
            "batch_id": batch["batch_id"],
            "model_family": provider_name,
            "started_at": started_at,
            "finished_at": now(),
            "input_chunk_ids": [chunk["chunk_id"] for chunk in batch["chunks"]],
            "error": str(exc),
        }


def validate_batch(rule_batch: dict[str, Any], input_batch: dict[str, Any]) -> dict[str, Any]:
    chunk_ids = {chunk["chunk_id"] for chunk in input_batch["chunks"]}
    node_ids = {node["id"] for node in input_batch.get("kg_nodes", [])}
    edge_ids = {edge["id"] for edge in input_batch.get("kg_edges", [])}
    rules = rule_batch.get("rules", []) if isinstance(rule_batch, dict) else []
    errors = []
    valid_rule_sources = 0
    valid_constraints = 0
    total_constraints = 0
    valid_relations = 0
    total_relations = 0
    valid_provenance = 0
    total_provenance = 0
    for idx, rule in enumerate(rules):
        if not isinstance(rule, dict):
            errors.append(f"rule[{idx}] is not object")
            continue
        src_chunks = set(str(x) for x in rule.get("source_chunk_ids", []))
        if src_chunks and src_chunks.issubset(chunk_ids):
            valid_rule_sources += 1
        else:
            errors.append(f"rule[{idx}] source_chunk_ids not grounded")
        for con in rule.get("constraints", []):
            total_constraints += 1
            ev = con.get("evidence", {}) if isinstance(con, dict) else {}
            cids = set(str(x) for x in ev.get("chunk_ids", []))
            nids = set(str(x) for x in ev.get("kg_node_ids", []))
            eids = set(str(x) for x in ev.get("kg_edge_ids", []))
            if cids and cids.issubset(chunk_ids) and nids.issubset(node_ids) and eids.issubset(edge_ids):
                valid_constraints += 1
            else:
                errors.append(f"constraint evidence not grounded in rule[{idx}]")
        for rel in rule.get("relations", []):
            total_relations += 1
            ev = rel.get("evidence", {}) if isinstance(rel, dict) else {}
            cids = set(str(x) for x in ev.get("chunk_ids", []))
            nids = set(str(x) for x in ev.get("kg_node_ids", []))
            eids = set(str(x) for x in ev.get("kg_edge_ids", []))
            if cids and cids.issubset(chunk_ids) and nids.issubset(node_ids) and eids.issubset(edge_ids):
                valid_relations += 1
            else:
                errors.append(f"relation evidence not grounded in rule[{idx}]")
        for prov in rule.get("provenance", []):
            total_provenance += 1
            if isinstance(prov, dict) and str(prov.get("chunk_id")) in chunk_ids:
                valid_provenance += 1
            else:
                errors.append(f"provenance not grounded in rule[{idx}]")
    non_rule_ids = set(str(x) for x in rule_batch.get("non_rule_chunk_ids", [])) if isinstance(rule_batch, dict) else set()
    non_rule_ok = non_rule_ids.issubset(chunk_ids)
    if not non_rule_ok:
        errors.append("non_rule_chunk_ids include ids outside batch")
    return {
        "rule_count": len(rules),
        "schema_has_rules_list": isinstance(rules, list),
        "source_grounding_rate": pct(valid_rule_sources, len(rules)),
        "constraint_grounding_rate": pct(valid_constraints, total_constraints),
        "relation_grounding_rate": pct(valid_relations, total_relations),
        "provenance_validity_rate": pct(valid_provenance, total_provenance),
        "non_rule_chunk_ids_valid": non_rule_ok,
        "constraint_count": total_constraints,
        "relation_count": total_relations,
        "provenance_count": total_provenance,
        "errors": errors[:50],
    }


def pct(num: float, den: float) -> float:
    if den == 0:
        return 100.0
    return 100.0 * float(num) / float(den)


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


def summarize(provider_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in rows if row.get("status") == "ok"]
    failed = [row for row in rows if row.get("status") not in {"ok", "skipped_missing_key"}]
    skipped = [row for row in rows if row.get("status") == "skipped_missing_key"]
    rules = []
    input_chunk_ids = set()
    non_rule_chunk_ids = set()
    vals = {
        "source_grounding_rate": [],
        "constraint_grounding_rate": [],
        "relation_grounding_rate": [],
        "provenance_validity_rate": [],
    }
    total_constraints = total_relations = 0
    for row in ok:
        input_chunk_ids.update(row.get("input_chunk_ids", []))
        batch = row.get("rule_batch") or {}
        rules.extend(batch.get("rules", []))
        non_rule_chunk_ids.update(batch.get("non_rule_chunk_ids", []))
        validation = row.get("validation") or {}
        for key in vals:
            vals[key].append(float(validation.get(key, 0.0)))
        total_constraints += int(validation.get("constraint_count", 0))
        total_relations += int(validation.get("relation_count", 0))
    covered_chunks = set()
    for rule in rules:
        covered_chunks.update(str(x) for x in rule.get("source_chunk_ids", []))
    covered_chunks.update(str(x) for x in non_rule_chunk_ids)
    return {
        "provider": provider_name,
        "num_batches_ok": len(ok),
        "num_batches_failed": len(failed),
        "num_batches_skipped_missing_key": len(skipped),
        "num_rules": len(rules),
        "num_input_chunks_seen": len(input_chunk_ids),
        "num_chunks_accounted_for": len(covered_chunks),
        "chunk_accounting_rate": pct(len(covered_chunks), len(input_chunk_ids)),
        "constraint_count": total_constraints,
        "relation_count": total_relations,
        "mean_source_grounding_rate": mean(vals["source_grounding_rate"]),
        "mean_constraint_grounding_rate": mean(vals["constraint_grounding_rate"]),
        "mean_relation_grounding_rate": mean(vals["relation_grounding_rate"]),
        "mean_provenance_validity_rate": mean(vals["provenance_validity_rate"]),
        "failed_batches": [
            {"batch_id": row.get("batch_id"), "status": row.get("status"), "error": row.get("error")}
            for row in failed
        ],
    }


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def build_library(provider_name: str, rows: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    rules = []
    batch_index = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        batch = row.get("rule_batch") or {}
        batch_rules = batch.get("rules", [])
        rules.extend(batch_rules)
        batch_index.append(
            {
                "batch_id": row.get("batch_id"),
                "input_chunk_ids": row.get("input_chunk_ids", []),
                "rule_count": len(batch_rules),
                "non_rule_chunk_ids": batch.get("non_rule_chunk_ids", []),
                "validation": row.get("validation", {}),
            }
        )
    return {
        "generated_at": now(),
        "provider": provider_name,
        "model": rv.PROVIDERS[provider_name].model,
        "source": "full Cognee aviation KG export: nodes.csv and edges.csv",
        "summary": summary,
        "rules": rules,
        "batch_index": batch_index,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert the full Cognee aviation KG export into CTHR rule libraries.")
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--providers", default="qwen,deepseek,glm")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--chunk-char-limit", type=int, default=1800)
    parser.add_argument("--max-neighbors", type=int, default=18)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--qwen-model", default="qwen-plus")
    parser.add_argument("--deepseek-model", default="deepseek-v4-flash")
    parser.add_argument("--glm-model", default="GLM-4.7")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit-batches", type=int, default=None)
    args = parser.parse_args()

    apply_model_overrides(args.qwen_model, args.deepseek_model, args.glm_model)
    provider_names = normalize_provider_names(args.providers)
    nodes, edges, doc_by_dataset = load_kg(args.export_dir)
    batches = make_batches(
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
    safe_write_json(
        OUT_ROOT / "manifest.json",
        {
            "generated_at": now(),
            "export_dir": str(args.export_dir),
            "num_batches": len(batches),
            "batch_size": args.batch_size,
            "chunk_char_limit": args.chunk_char_limit,
            "max_neighbors": args.max_neighbors,
            "providers": provider_names,
            "models": {name: rv.PROVIDERS[name].model for name in provider_names},
            "note": "API keys are read from environment variables and are not written to outputs.",
        },
    )
    safe_write_json(OUT_ROOT / "input_batches_index.json", {"batches": batches})

    for provider_name in provider_names:
        key_ok, key_name = ensure_provider_env(provider_name)
        if not key_ok:
            for batch in batches:
                path = checkpoint_path(provider_name, batch["batch_id"])
                if path.exists() and not args.force:
                    continue
                safe_write_json(
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
            row = call_batch(
                provider_name=provider_name,
                batch=batch,
                system_prompt=system_prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
            safe_write_json(path, row)
            print(
                f"[{now()}] {provider_name} {batch['batch_id']}: "
                f"{row.get('status')} rules={row.get('validation', {}).get('rule_count')}"
            )

    summaries = []
    for provider_name in provider_names:
        rows = collect_rows(provider_name, batch_ids)
        summary = summarize(provider_name, rows)
        summaries.append(summary)
        library = build_library(provider_name, rows, summary)
        safe_write_json(OUT_ROOT / f"full_aviation_rule_library_{provider_name}.json", library)
        safe_write_json(OUT_ROOT / f"summary_{provider_name}.json", summary)

    comparison = {
        "generated_at": now(),
        "export_dir": str(args.export_dir),
        "num_batches": len(batches),
        "num_chunks": sum(len(batch["chunks"]) for batch in batches),
        "summaries": summaries,
    }
    safe_write_json(OUT_ROOT / "full_model_comparison_summary.json", comparison)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
