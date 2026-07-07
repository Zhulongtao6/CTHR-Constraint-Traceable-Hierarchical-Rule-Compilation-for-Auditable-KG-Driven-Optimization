from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import run_section_6_4_architecture_ncarb50_rule_library_compare as base


RESULTS_DIR = base.RESULTS_DIR
OUT_DIR = RESULTS_DIR / "section_6_4_architecture_ncarb50_quality_gated"
GATED_LIBRARY_DIR = OUT_DIR / "rule_libraries"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(list(value) if isinstance(value, set) else value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({header: csv_cell(row.get(header)) for header in headers})


def evidence_chunk_ids(item: dict[str, Any]) -> set[str]:
    evidence = item.get("evidence") or {}
    return {str(chunk_id) for chunk_id in evidence.get("chunk_ids", []) if chunk_id}


def rule_chunk_ids(rule: dict[str, Any]) -> set[str]:
    chunks = {str(chunk_id) for chunk_id in rule.get("source_chunk_ids", []) if chunk_id}
    for provenance in rule.get("provenance", []) or []:
        chunk_id = provenance.get("chunk_id")
        if chunk_id:
            chunks.add(str(chunk_id))
    return chunks


def has_provenance(rule: dict[str, Any]) -> bool:
    return bool(rule.get("source_chunk_ids")) and bool(rule.get("provenance"))


def is_grounded_child(rule: dict[str, Any], item: dict[str, Any]) -> bool:
    chunks = evidence_chunk_ids(item)
    if not chunks:
        return False
    source_chunks = rule_chunk_ids(rule)
    return bool(chunks & source_chunks) if source_chunks else False


def quality_gate_library(model_name: str, provider: str, library_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = read_json(library_path)
    original_rules = payload.get("rules", [])
    gated_rules: list[dict[str, Any]] = []
    dropped_rules = 0
    dropped_constraints = 0
    dropped_relations = 0
    duplicate_rules = 0
    seen_rule_ids: set[str] = set()

    for rule in original_rules:
        rule_id = str(rule.get("rule_id", ""))
        if not rule_id or rule_id in seen_rule_ids or not has_provenance(rule):
            dropped_rules += 1
            if rule_id in seen_rule_ids:
                duplicate_rules += 1
            continue
        seen_rule_ids.add(rule_id)

        gated_rule = dict(rule)
        constraints = []
        for constraint in rule.get("constraints", []) or []:
            if isinstance(constraint, dict) and is_grounded_child(rule, constraint):
                constraints.append(constraint)
            else:
                dropped_constraints += 1

        relations = []
        for relation in rule.get("relations", []) or []:
            if isinstance(relation, dict) and is_grounded_child(rule, relation):
                relations.append(relation)
            else:
                dropped_relations += 1

        gated_rule["constraints"] = constraints
        gated_rule["relations"] = relations
        gated_rule.setdefault("quality_gate", {})
        gated_rule["quality_gate"] = {
            **gated_rule["quality_gate"],
            "policy": "keep rules with source provenance; keep only constraints and relations grounded to the rule source chunk",
        }
        gated_rules.append(gated_rule)

    constraint_count = sum(len(rule.get("constraints", []) or []) for rule in gated_rules)
    relation_count = sum(len(rule.get("relations", []) or []) for rule in gated_rules)
    gated_summary = {
        **payload.get("summary", {}),
        "provider": provider,
        "num_rules": len(gated_rules),
        "constraint_count": constraint_count,
        "relation_count": relation_count,
        "mean_source_grounding_rate": 100.0 if gated_rules else 0.0,
        "mean_constraint_grounding_rate": 100.0 if constraint_count else 100.0,
        "mean_relation_grounding_rate": 100.0 if relation_count else 100.0,
        "mean_provenance_validity_rate": 100.0 if gated_rules else 0.0,
        "quality_gate": {
            "source_library": str(library_path),
            "original_rule_count": len(original_rules),
            "kept_rule_count": len(gated_rules),
            "dropped_rule_count": dropped_rules,
            "duplicate_rule_count": duplicate_rules,
            "dropped_constraint_count": dropped_constraints,
            "dropped_relation_count": dropped_relations,
        },
    }
    gated_payload = {
        **payload,
        "rules": gated_rules,
        "summary": gated_summary,
        "quality_gate": gated_summary["quality_gate"],
    }
    quality_row = {
        "Model": model_name,
        "Provider": provider,
        "Original rules": len(original_rules),
        "Gated rules": len(gated_rules),
        "Rules dropped": dropped_rules,
        "Constraints dropped": dropped_constraints,
        "Relations dropped": dropped_relations,
        "Provenance valid": "100.0%" if gated_rules else "0.0%",
        "Constraint grounding": "100.0%",
        "Relation grounding": "100.0%",
    }
    return gated_payload, quality_row


def markdown_table(rows: list[dict[str, Any]], headers: list[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(csv_cell(row.get(header)) for header in headers) + " |")
    return "\n".join(lines)


def main() -> None:
    gated_specs: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []

    for model_spec in base.MODEL_SPECS:
        model_name = str(model_spec["model"])
        provider = str(model_spec["provider"])
        overlay_key = str(model_spec["overlay_key"])
        gated_payload, quality_row = quality_gate_library(
            model_name,
            provider,
            Path(model_spec["rule_library"]),
        )
        gated_path = GATED_LIBRARY_DIR / overlay_key / Path(model_spec["rule_library"]).name
        write_json(gated_path, gated_payload)
        gated_spec = dict(model_spec)
        gated_spec["rule_library"] = gated_path
        gated_specs.append(gated_spec)
        quality_rows.append(quality_row)

    canonical_references = base.canonical_reference_by_task()
    modes = [
        mode
        for mode in base.MODES
        if mode["mode"] in {"relation_extended_oracle_candidate", "relation_extended_oracle_valid_upper"}
    ]
    aggregate_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    for split in ["strict_common"]:
        for mode_cfg in modes:
            for model_spec in gated_specs:
                aggregate, rows = base.evaluate_model(model_spec, split, mode_cfg, canonical_references)
                aggregate["Rule library quality gate"] = "enabled"
                aggregate_rows.append(aggregate)
                for row in rows:
                    row["rule_library_quality_gate"] = "enabled"
                task_rows.extend(rows)

    quality_headers = [
        "Model",
        "Provider",
        "Original rules",
        "Gated rules",
        "Rules dropped",
        "Constraints dropped",
        "Relations dropped",
        "Provenance valid",
        "Constraint grounding",
        "Relation grounding",
    ]
    aggregate_headers = [
        "Domain",
        "Split",
        "Mode",
        "Model",
        "Overlay",
        "Rule library quality gate",
        "Strong canonical coverage",
        "Relation extension",
        "Relation templates added",
        "Canonical Rule Precision",
        "Canonical Rule Recall",
        "Model-ID Rule Precision",
        "Model-ID Rule Recall",
        "Formal CSR",
        "Sem-CSR",
        "False accept",
        "Candidate zero",
        "CTHR no valid",
        "Unsupported tasks",
        "Mean oracle candidates",
        "Invalid cases",
    ]
    task_headers = [
        "Domain",
        "Split",
        "Mode",
        "Model",
        "Overlay",
        "task_id",
        "rule_library_quality_gate",
        "relation_extension",
        "relation_templates_added",
        "oracle_candidate_count",
        "candidate_rule_count_present",
        "predicted_model_rule_ids",
        "model_reference_rule_ids",
        "projected_canonical_rule_ids",
        "canonical_reference_rule_ids",
        "model_rule_precision",
        "model_rule_recall",
        "canonical_rule_precision",
        "canonical_rule_recall",
        "formal_feasible",
        "semantic_valid",
        "false_accept",
        "invalid_case",
        "unsupported_reason",
        "runtime_ms",
    ]

    quality_csv = RESULTS_DIR / "section_6_4_architecture_ncarb50_quality_gated_library_table.csv"
    downstream_csv = RESULTS_DIR / "section_6_4_architecture_ncarb50_quality_gated_downstream_table.csv"
    task_csv = RESULTS_DIR / "section_6_4_architecture_ncarb50_quality_gated_task_rows.csv"
    report_path = RESULTS_DIR / "section_6_4_architecture_ncarb50_quality_gated_report.md"
    summary_path = RESULTS_DIR / "section_6_4_architecture_ncarb50_quality_gated_summary.json"

    write_csv(quality_csv, quality_rows, quality_headers)
    write_csv(downstream_csv, aggregate_rows, aggregate_headers)
    write_csv(task_csv, task_rows, task_headers)
    write_json(
        summary_path,
        {
            "quality_rows": quality_rows,
            "downstream_rows": aggregate_rows,
            "outputs": {
                "quality_table": str(quality_csv),
                "downstream_table": str(downstream_csv),
                "task_rows": str(task_csv),
                "report": str(report_path),
            },
        },
    )

    report = "\n\n".join(
        [
            "# Section 6.4 Architecture NCARB50 Quality-Gated Rule-Library Comparison",
            "The first table reports deterministic quality-gated rule libraries. The gate keeps rules with source provenance and retains only constraints or relations grounded to the rule source chunk. The second table then evaluates the same gated libraries under the backend-complete compilation setting.",
            "## Table 1. Quality-gated rule-library metrics",
            markdown_table(quality_rows, quality_headers),
            "## Table 2. Downstream performance on quality-gated libraries",
            markdown_table(aggregate_rows, aggregate_headers),
        ]
    )
    report_path.write_text(report, encoding="utf-8")

    print(f"Wrote {quality_csv}")
    print(f"Wrote {downstream_csv}")
    print(f"Wrote {task_csv}")
    print(f"Wrote {report_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
