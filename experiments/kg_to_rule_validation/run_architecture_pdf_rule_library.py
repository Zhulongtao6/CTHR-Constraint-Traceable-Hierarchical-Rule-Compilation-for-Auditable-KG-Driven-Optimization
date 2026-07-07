from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import fitz


THIS_DIR = Path(__file__).resolve().parent
CTHR_ROOT = THIS_DIR.parents[1]
PAPER_DIR = CTHR_ROOT / "paper"
DEFAULT_PDF_DIR = Path(r"D:\paper\LLMEhancebackgroud\建筑领域\规范")
OUT_ROOT = PAPER_DIR / "architecture_rule_library_model_comparison"

sys.path.insert(0, str(THIS_DIR))
import run_llm_rule_validation as rv  # noqa: E402
import run_full_aviation_kg_rule_library as full  # noqa: E402


PROVIDER_ALIASES = {
    "tongyi": "qwen",
    "qwen": "qwen",
    "deepseek": "deepseek",
}

DOC_TARGETS = {
    "2010-design-standards.pdf": [
        r"\b404(?:\.\d+)*\b",
        r"\b405(?:\.\d+)*\b",
        r"\b406(?:\.\d+)*\b",
        r"\b502(?:\.\d+)*\b",
        r"\b503(?:\.\d+)*\b",
        r"\b604(?:\.\d+)*\b",
        r"\b607(?:\.\d+)*\b",
        r"\b608(?:\.\d+)*\b",
        r"\b609(?:\.\d+)*\b",
        r"roll-in shower",
        r"toilet compartment",
        r"parking space",
        r"ramp",
        r"curb ramp",
    ],
    "2021InternationalBuildingCode.pdf": [
        r"\b414(?:\.\d+)*\b",
        r"\b506(?:\.\d+)*\b",
        r"\b508(?:\.\d+)*\b",
        r"\b903(?:\.\d+)*\b",
        r"\b1005(?:\.\d+)*\b",
        r"\b1006(?:\.\d+)*\b",
        r"\b1010(?:\.\d+)*\b",
        r"\b1017(?:\.\d+)*\b",
        r"\b1020(?:\.\d+)*\b",
        r"\b1106(?:\.\d+)*\b",
        r"allowable area",
        r"mixed occupancy",
        r"means of egress",
        r"travel distance",
        r"sprinkler",
    ],
    "IFC-2021.pdf": [
        r"\b5003(?:\.\d+)*\b",
        r"\b5004(?:\.\d+)*\b",
        r"\b5005(?:\.\d+)*\b",
        r"\b5006(?:\.\d+)*\b",
        r"\b5704(?:\.\d+)*\b",
        r"hazardous materials",
        r"maximum allowable quantity",
        r"closed container",
        r"control area",
        r"standby power",
    ],
}

SYSTEM_PROMPT = """
You convert source-linked architecture and building-code evidence into a CTHR
rule library. The input is a small batch of PDF source chunks from ADA, IBC, or
IFC building-code documents.

Return exactly one valid JSON object and nothing else. Do not use Markdown.

Required JSON schema:
{
  "batch_id": "...",
  "rules": [
    {
      "rule_id": "stable_short_id",
      "name": "human-readable rule name",
      "domain": "architecture",
      "rule_type": "requirement|definition|calculation|parameter|exception|precedence|applicability|procedure_step",
      "source_chunk_ids": ["existing chunk id"],
      "source_node_ids": ["existing source node id if relevant"],
      "guard": {
        "all": [
          {"field": "scenario/property name", "op": "eq|neq|gt|gte|lt|lte|in", "value": "condition value"}
        ]
      },
      "constraints": [
        {
          "variable": "decision or regulatory quantity name",
          "op": ">=|<=|=|>|<|formula|text",
          "value": "number, expression, or text condition",
          "unit": "unit if stated, otherwise unknown",
          "source_quote": "short source phrase supporting the constraint",
          "evidence": {
            "chunk_ids": ["existing chunk id"],
            "kg_node_ids": ["existing source node id if relevant"],
            "kg_edge_ids": ["existing source edge id if relevant"]
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
            "kg_node_ids": ["existing source node id if relevant"],
            "kg_edge_ids": ["existing source edge id if relevant"]
          }
        }
      ],
      "provenance": [
        {
          "document": "document name",
          "section": "section if known, otherwise unknown",
          "page": "PDF page number",
          "chunk_id": "existing chunk id"
        }
      ],
      "extraction_notes": "brief grounding note",
      "confidence": 0.0
    }
  ],
  "non_rule_chunk_ids": ["chunk ids with no extractable normative rule"]
}

Extraction rules:
- Extract only requirements, exceptions, precedence/applicability statements,
  definitions, formulae, dimensional limits, and parameter rules explicitly
  supported by the provided chunks.
- Do not invent numeric values, units, source chunks, sections, or pages.
- Keep alternative compliance paths as separate rules.
- Keep exception, precedence, and dependency relations explicit.
- If a requirement is qualitative, use op="text".
- If a formula is not a linear inequality, use op="formula".
- Source quotes must be short and copied from the provided evidence.
""".strip()

JSON_REPAIR_PROMPT = """
Repair or regenerate a malformed architecture CTHR rule-library JSON output.
Return exactly one valid JSON object and nothing else.
Use the source input batch as the ground truth.
If the chunk has no extractable normative rule, return rules=[] and put its
chunk id in non_rule_chunk_ids. Do not invent IDs, values, units, sections, or pages.
""".strip()


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def safe_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def short_text(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def normalize_provider_names(raw: str) -> list[str]:
    names: list[str] = []
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


def apply_model_overrides(qwen_model: str, deepseek_model: str) -> None:
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


def find_pdf_files(pdf_dir: Path) -> list[Path]:
    expected = set(DOC_TARGETS)
    pdfs = [p for p in pdf_dir.glob("*.pdf") if p.name in expected]
    if len(pdfs) == len(expected):
        return sorted(pdfs, key=lambda p: p.name)
    fallback_root = Path("D:/paper/LLMEhancebackgroud")
    found: dict[str, Path] = {}
    for path in fallback_root.rglob("*.pdf"):
        if path.name in expected and path.name not in found:
            found[path.name] = path
    return [found[name] for name in sorted(found)]


def select_pages_for_pdf(pdf_path: Path, max_pages_per_doc: int, pages_around: int) -> list[dict[str, Any]]:
    doc = fitz.open(str(pdf_path))
    patterns = [re.compile(pat, flags=re.IGNORECASE) for pat in DOC_TARGETS.get(pdf_path.name, [])]
    hits: set[int] = set()
    for page_idx in range(len(doc)):
        text = doc[page_idx].get_text("text")
        if any(pattern.search(text) for pattern in patterns):
            for offset in range(-pages_around, pages_around + 1):
                idx = page_idx + offset
                if 0 <= idx < len(doc):
                    hits.add(idx)
    selected = []
    for page_idx in sorted(hits)[:max_pages_per_doc]:
        text = doc[page_idx].get_text("text")
        if short_text(text, 80):
            selected.append(
                {
                    "pdf_path": str(pdf_path),
                    "document": pdf_path.name,
                    "page": page_idx + 1,
                    "text": text,
                }
            )
    return selected


def make_batches(
    pdf_dir: Path,
    batch_size: int,
    chunk_char_limit: int,
    max_pages_per_doc: int,
    pages_around: int,
) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    for pdf_path in find_pdf_files(pdf_dir):
        pages.extend(select_pages_for_pdf(pdf_path, max_pages_per_doc=max_pages_per_doc, pages_around=pages_around))
    chunks = []
    for idx, page in enumerate(pages):
        doc_slug = re.sub(r"[^A-Za-z0-9]+", "_", page["document"]).strip("_").lower()
        chunk_id = f"arch_{doc_slug}_p{page['page']:04d}"
        node_id = f"arch_node_{doc_slug}_p{page['page']:04d}"
        edge_id = f"arch_edge_{doc_slug}_p{page['page']:04d}_source"
        chunks.append(
            {
                "chunk_id": chunk_id,
                "document": page["document"],
                "page_hint": str(page["page"]),
                "text": short_text(page["text"], chunk_char_limit),
                "source_node_id": node_id,
                "source_edge_id": edge_id,
            }
        )
    batches = []
    for start in range(0, len(chunks), batch_size):
        selected = chunks[start : start + batch_size]
        kg_nodes = [
            {
                "id": chunk["source_node_id"],
                "label": "SourcePage",
                "name": f"{chunk['document']} page {chunk['page_hint']}",
                "description": short_text(chunk["text"], 300),
            }
            for chunk in selected
        ]
        kg_edges = [
            {
                "id": chunk["source_edge_id"],
                "source": chunk["source_node_id"],
                "relation": "source_page_for",
                "target": chunk["chunk_id"],
                "neighbor_name": chunk["document"],
                "neighbor_label": "DocumentChunk",
            }
            for chunk in selected
        ]
        batches.append(
            {
                "batch_id": f"arch_pdf_batch_{len(batches):04d}",
                "domain": "architecture",
                "source": "source-linked architecture PDF chunks",
                "chunks": selected,
                "kg_nodes": kg_nodes,
                "kg_edges": kg_edges,
            }
        )
    return batches


def checkpoint_path(provider_name: str, batch_id: str) -> Path:
    return OUT_ROOT / provider_name / f"{batch_id}.json"


def is_completed(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = load_json(path)
    except json.JSONDecodeError:
        return False
    return payload.get("status") == "ok"


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


def canonicalize_rule_batch(parsed: dict[str, Any], batch: dict[str, Any]) -> dict[str, Any]:
    rules = parsed.get("rules", []) if isinstance(parsed, dict) else []
    if not isinstance(rules, list):
        rules = []
    input_chunk_ids = [chunk["chunk_id"] for chunk in batch["chunks"]]
    chunk_by_id = {chunk["chunk_id"]: chunk for chunk in batch["chunks"]}
    node_ids = {node["id"] for node in batch.get("kg_nodes", [])}
    edge_ids = {edge["id"] for edge in batch.get("kg_edges", [])}
    canonical_rules = []
    for idx, rule in enumerate(rules, start=1):
        if not isinstance(rule, dict):
            continue
        raw_sources = as_list(rule.get("source_chunk_ids")) + as_list(rule.get("chunk_ids")) + as_list(rule.get("chunk_id"))
        source_chunk_ids = [str(cid) for cid in raw_sources if str(cid) in input_chunk_ids]
        if not source_chunk_ids and len(input_chunk_ids) == 1:
            source_chunk_ids = input_chunk_ids[:]
        if not source_chunk_ids:
            continue
        source_node_ids = [str(x) for x in as_list(rule.get("source_node_ids")) if str(x) in node_ids]
        if not source_node_ids:
            source_node_ids = [chunk_by_id[cid]["source_node_id"] for cid in source_chunk_ids if cid in chunk_by_id]

        constraints = rule.get("constraints") if isinstance(rule.get("constraints"), list) else []
        canonical_constraints = []
        for con in constraints:
            if not isinstance(con, dict):
                continue
            ev = con.get("evidence") if isinstance(con.get("evidence"), dict) else {}
            ev_chunks = [str(x) for x in as_list(ev.get("chunk_ids")) if str(x) in input_chunk_ids] or source_chunk_ids
            ev_nodes = [str(x) for x in as_list(ev.get("kg_node_ids")) if str(x) in node_ids]
            ev_edges = [str(x) for x in as_list(ev.get("kg_edge_ids")) if str(x) in edge_ids]
            if not ev_nodes:
                ev_nodes = [chunk_by_id[cid]["source_node_id"] for cid in ev_chunks if cid in chunk_by_id]
            if not ev_edges:
                ev_edges = [chunk_by_id[cid]["source_edge_id"] for cid in ev_chunks if cid in chunk_by_id]
            canonical_constraints.append(
                {
                    "variable": first_text(con.get("variable"), default="regulatory_requirement"),
                    "op": first_text(con.get("op"), default="text"),
                    "value": con.get("value", ""),
                    "unit": first_text(con.get("unit"), default="unknown"),
                    "source_quote": first_text(con.get("source_quote"), con.get("value"), default=""),
                    "evidence": {"chunk_ids": ev_chunks, "kg_node_ids": ev_nodes, "kg_edge_ids": ev_edges},
                }
            )
        if not canonical_constraints:
            quote = first_text(rule.get("rule"), rule.get("requirement"), rule.get("action"), rule.get("name"), default="qualitative regulatory requirement")
            canonical_constraints.append(
                {
                    "variable": first_text(rule.get("variable"), rule.get("name"), default="regulatory_requirement"),
                    "op": first_text(rule.get("op"), default="text"),
                    "value": quote,
                    "unit": first_text(rule.get("unit"), default="unknown"),
                    "source_quote": quote[:220],
                    "evidence": {
                        "chunk_ids": source_chunk_ids,
                        "kg_node_ids": source_node_ids,
                        "kg_edge_ids": [chunk_by_id[cid]["source_edge_id"] for cid in source_chunk_ids if cid in chunk_by_id],
                    },
                }
            )

        relations = rule.get("relations") if isinstance(rule.get("relations"), list) else []
        canonical_relations = []
        for rel in relations:
            if not isinstance(rel, dict):
                continue
            ev = rel.get("evidence") if isinstance(rel.get("evidence"), dict) else {}
            ev_chunks = [str(x) for x in as_list(ev.get("chunk_ids")) if str(x) in input_chunk_ids] or source_chunk_ids
            ev_nodes = [str(x) for x in as_list(ev.get("kg_node_ids")) if str(x) in node_ids] or [
                chunk_by_id[cid]["source_node_id"] for cid in ev_chunks if cid in chunk_by_id
            ]
            ev_edges = [str(x) for x in as_list(ev.get("kg_edge_ids")) if str(x) in edge_ids] or [
                chunk_by_id[cid]["source_edge_id"] for cid in ev_chunks if cid in chunk_by_id
            ]
            canonical_relations.append(
                {
                    "type": first_text(rel.get("type"), default="requires"),
                    "target": first_text(rel.get("target"), rel.get("target_rule"), default="regulatory_requirement"),
                    "source_quote": first_text(rel.get("source_quote"), rel.get("target"), rel.get("target_rule"), default=""),
                    "evidence": {"chunk_ids": ev_chunks, "kg_node_ids": ev_nodes, "kg_edge_ids": ev_edges},
                }
            )

        provenance = []
        raw_prov = rule.get("provenance") if isinstance(rule.get("provenance"), list) else []
        for cid in source_chunk_ids:
            chunk = chunk_by_id.get(cid, {})
            matching = [item for item in raw_prov if isinstance(item, dict) and str(item.get("chunk_id")) == cid]
            item = matching[0] if matching else {}
            provenance.append(
                {
                    "document": first_text(item.get("document"), default=chunk.get("document", "unknown")),
                    "section": first_text(item.get("section"), rule.get("source_section"), default="unknown"),
                    "page": first_text(item.get("page"), rule.get("source_page"), default=chunk.get("page_hint", "unknown")),
                    "chunk_id": cid,
                }
            )
        canonical_rules.append(
            {
                "rule_id": first_text(rule.get("rule_id"), rule.get("id"), default=f"{batch['batch_id']}_rule_{idx}"),
                "name": first_text(rule.get("name"), rule.get("rule"), rule.get("requirement"), rule.get("action"), default=f"Architecture rule {idx}"),
                "domain": "architecture",
                "rule_type": first_text(rule.get("rule_type"), rule.get("rule_class"), default="requirement"),
                "source_chunk_ids": source_chunk_ids,
                "source_node_ids": source_node_ids,
                "guard": rule.get("guard") if isinstance(rule.get("guard"), dict) else {},
                "constraints": canonical_constraints,
                "relations": canonical_relations,
                "provenance": provenance,
                "extraction_notes": first_text(rule.get("extraction_notes"), default="Canonicalized from architecture PDF LLM output."),
                "confidence": rule.get("confidence", 0.7),
            }
        )
    non_rule_ids = [str(x) for x in as_list(parsed.get("non_rule_chunk_ids") if isinstance(parsed, dict) else []) if str(x) in input_chunk_ids]
    if not canonical_rules and not non_rule_ids:
        non_rule_ids = input_chunk_ids
    return {"batch_id": batch["batch_id"], "rules": canonical_rules, "non_rule_chunk_ids": non_rule_ids}


def validate_batch(rule_batch: dict[str, Any], input_batch: dict[str, Any]) -> dict[str, Any]:
    chunk_ids = {chunk["chunk_id"] for chunk in input_batch["chunks"]}
    node_ids = {node["id"] for node in input_batch.get("kg_nodes", [])}
    edge_ids = {edge["id"] for edge in input_batch.get("kg_edges", [])}
    rules = rule_batch.get("rules", []) if isinstance(rule_batch, dict) else []
    errors: list[str] = []
    valid_rule_sources = valid_constraints = total_constraints = 0
    valid_relations = total_relations = 0
    valid_provenance = total_provenance = 0
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
        "source_grounding_rate": full.pct(valid_rule_sources, len(rules)),
        "constraint_grounding_rate": full.pct(valid_constraints, total_constraints),
        "relation_grounding_rate": full.pct(valid_relations, total_relations),
        "provenance_validity_rate": full.pct(valid_provenance, total_provenance),
        "non_rule_chunk_ids_valid": non_rule_ok,
        "constraint_count": total_constraints,
        "relation_count": total_relations,
        "provenance_count": total_provenance,
        "errors": errors[:50],
    }


def call_and_parse(provider: rv.Provider, system_prompt: str, payload: dict[str, Any], temperature: float, max_tokens: int, timeout: int) -> tuple[dict[str, Any], str, dict[str, Any]]:
    raw, meta = rv.call_provider(
        provider=provider,
        system_prompt=system_prompt,
        user_payload=payload,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    parsed = rv.extract_json(raw)
    return parsed, raw, meta


def call_batch(provider_name: str, batch: dict[str, Any], temperature: float, max_tokens: int, timeout: int) -> dict[str, Any]:
    provider = rv.PROVIDERS[provider_name]
    started_at = now()
    attempts: list[dict[str, Any]] = []
    try:
        parsed, raw, meta = call_and_parse(provider, SYSTEM_PROMPT, batch, temperature, max_tokens, timeout)
        attempts.append({"stage": "extract", "raw_text": raw, "parsed": True})
        canonical = canonicalize_rule_batch(parsed, batch)
        return {
            "status": "ok",
            "batch_id": batch["batch_id"],
            "provider": meta,
            "model_family": provider_name,
            "started_at": started_at,
            "finished_at": now(),
            "input_chunk_ids": [chunk["chunk_id"] for chunk in batch["chunks"]],
            "repair_stage": "extract",
            "raw_attempts": attempts,
            "raw_text": raw,
            "rule_batch_raw": parsed,
            "rule_batch": canonical,
            "schema_normalized": True,
            "validation": validate_batch(canonical, batch),
        }
    except Exception as exc:  # noqa: BLE001
        attempts.append({"stage": "extract", "parsed": False, "error": str(exc)})
    repair_payload = {
        "source_input_batch": batch,
        "malformed_output_or_error": attempts[-1],
        "required_batch_id": batch["batch_id"],
        "required_chunk_ids": [chunk["chunk_id"] for chunk in batch["chunks"]],
    }
    try:
        parsed, raw, meta = call_and_parse(provider, JSON_REPAIR_PROMPT, repair_payload, 0.0, max_tokens, timeout)
        attempts.append({"stage": "json_repair", "raw_text": raw, "parsed": True})
        canonical = canonicalize_rule_batch(parsed, batch)
        return {
            "status": "ok",
            "batch_id": batch["batch_id"],
            "provider": meta,
            "model_family": provider_name,
            "started_at": started_at,
            "finished_at": now(),
            "input_chunk_ids": [chunk["chunk_id"] for chunk in batch["chunks"]],
            "repair_stage": "json_repair",
            "raw_attempts": attempts,
            "raw_text": raw,
            "rule_batch_raw": parsed,
            "rule_batch": canonical,
            "schema_normalized": True,
            "validation": validate_batch(canonical, batch),
        }
    except Exception as exc:  # noqa: BLE001
        attempts.append({"stage": "json_repair", "parsed": False, "error": str(exc)})
        return {
            "status": "failed",
            "batch_id": batch["batch_id"],
            "model_family": provider_name,
            "started_at": started_at,
            "finished_at": now(),
            "input_chunk_ids": [chunk["chunk_id"] for chunk in batch["chunks"]],
            "error": str(exc),
            "raw_attempts": attempts,
        }


def collect_rows(provider_name: str, batch_ids: list[str]) -> list[dict[str, Any]]:
    rows = []
    for batch_id in batch_ids:
        path = checkpoint_path(provider_name, batch_id)
        if not path.exists():
            continue
        try:
            rows.append(load_json(path))
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
    vals = {key: [] for key in ["source_grounding_rate", "constraint_grounding_rate", "relation_grounding_rate", "provenance_validity_rate"]}
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
    covered_chunks = set(non_rule_chunk_ids)
    for rule in rules:
        covered_chunks.update(str(x) for x in rule.get("source_chunk_ids", []))
    return {
        "provider": provider_name,
        "num_batches_ok": len(ok),
        "num_batches_failed": len(failed),
        "num_batches_skipped_missing_key": len(skipped),
        "num_rules": len(rules),
        "num_input_chunks_seen": len(input_chunk_ids),
        "num_chunks_accounted_for": len(covered_chunks),
        "chunk_accounting_rate": full.pct(len(covered_chunks), len(input_chunk_ids)),
        "constraint_count": total_constraints,
        "relation_count": total_relations,
        "mean_source_grounding_rate": full.mean(vals["source_grounding_rate"]),
        "mean_constraint_grounding_rate": full.mean(vals["constraint_grounding_rate"]),
        "mean_relation_grounding_rate": full.mean(vals["relation_grounding_rate"]),
        "mean_provenance_validity_rate": full.mean(vals["provenance_validity_rate"]),
        "failed_batches": [
            {"batch_id": row.get("batch_id"), "status": row.get("status"), "error": row.get("error")}
            for row in failed
        ],
    }


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
        "domain": "architecture",
        "source": "source-linked PDF evidence chunks from ADA 2010, IBC 2021, and IFC 2021",
        "summary": summary,
        "rules": rules,
        "batch_index": batch_index,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate architecture rule libraries from source-linked PDF evidence with Qwen/DeepSeek.")
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--providers", default="qwen,deepseek")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--chunk-char-limit", type=int, default=2200)
    parser.add_argument("--max-pages-per-doc", type=int, default=18)
    parser.add_argument("--pages-around", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=3500)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--qwen-model", default="qwen-plus")
    parser.add_argument("--deepseek-model", default="deepseek-v4-pro")
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    apply_model_overrides(args.qwen_model, args.deepseek_model)
    provider_names = normalize_provider_names(args.providers)
    batches = make_batches(
        pdf_dir=args.pdf_dir,
        batch_size=args.batch_size,
        chunk_char_limit=args.chunk_char_limit,
        max_pages_per_doc=args.max_pages_per_doc,
        pages_around=args.pages_around,
    )
    if args.limit_batches is not None:
        batches = batches[: args.limit_batches]
    batch_ids = [batch["batch_id"] for batch in batches]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    safe_write_json(
        OUT_ROOT / "manifest.json",
        {
            "generated_at": now(),
            "pdf_dir": str(args.pdf_dir),
            "num_batches": len(batches),
            "num_chunks": sum(len(batch["chunks"]) for batch in batches),
            "batch_size": args.batch_size,
            "chunk_char_limit": args.chunk_char_limit,
            "max_pages_per_doc": args.max_pages_per_doc,
            "pages_around": args.pages_around,
            "providers": provider_names,
            "models": {name: rv.PROVIDERS[name].model for name in provider_names},
            "note": "This is source-linked PDF evidence, not a Cognee nodes/edges export.",
        },
    )
    safe_write_json(OUT_ROOT / "input_batches_index.json", {"batches": batches})
    if args.prepare_only:
        print(json.dumps({"out_root": str(OUT_ROOT), "num_batches": len(batches), "num_chunks": sum(len(batch["chunks"]) for batch in batches)}, ensure_ascii=False, indent=2))
        return

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
            print(f"[{now()}] {provider_name}: skipped because {key_name} is missing", flush=True)
            continue
        for idx, batch in enumerate(batches, start=1):
            path = checkpoint_path(provider_name, batch["batch_id"])
            if is_completed(path) and not args.force:
                print(f"[{now()}] {provider_name} {idx}/{len(batches)} {batch['batch_id']}: checkpoint ok, skip", flush=True)
                continue
            print(f"[{now()}] {provider_name} {idx}/{len(batches)} {batch['batch_id']}: calling API", flush=True)
            row = call_batch(provider_name, batch, args.temperature, args.max_tokens, args.timeout)
            safe_write_json(path, row)
            print(
                f"[{now()}] {provider_name} {batch['batch_id']}: {row.get('status')} "
                f"stage={row.get('repair_stage')} rules={row.get('validation', {}).get('rule_count')}",
                flush=True,
            )

    summaries = []
    for provider_name in provider_names:
        rows = collect_rows(provider_name, batch_ids)
        summary = summarize(provider_name, rows)
        summaries.append(summary)
        library = build_library(provider_name, rows, summary)
        safe_write_json(OUT_ROOT / f"full_architecture_rule_library_{provider_name}.json", library)
        safe_write_json(OUT_ROOT / f"summary_{provider_name}.json", summary)

    comparison = {
        "generated_at": now(),
        "num_batches": len(batches),
        "num_chunks": sum(len(batch["chunks"]) for batch in batches),
        "summaries": summaries,
    }
    safe_write_json(OUT_ROOT / "architecture_model_comparison_summary.json", comparison)
    print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
