from __future__ import annotations

import copy
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CTHR_ROOT = ROOT.parents[1]
PAPER_DIR = CTHR_ROOT / "paper"

OLD_DIR = PAPER_DIR / "aviation_benchmark_layers"
STRESS_DIR = PAPER_DIR / "aviation_stress_benchmark_layers"
OUT_DIR = ROOT / "datasets" / "aviation_combined"
TASK_DIR = OUT_DIR / "tasks"
REPORT_DIR = ROOT / "reports"
METADATA_DIR = ROOT / "metadata"

OLD_QUERIES = OLD_DIR / "aviation_optimization_queries.json"
OLD_RULE_LABELS = OLD_DIR / "aviation_rule_structure_labels.json"
OLD_FEASIBLE = OLD_DIR / "aviation_feasible_region_labels.json"

STRESS_QUERIES = STRESS_DIR / "aviation_stress_optimization_queries.json"
STRESS_RULE_LABELS = STRESS_DIR / "aviation_stress_rule_structure_labels.json"
STRESS_FEASIBLE = STRESS_DIR / "aviation_stress_feasible_region_labels.json"
STRESS_RULE_LIBRARY = STRESS_DIR / "aviation_stress_rule_library.combined.json"
STRESS_TASK_DIR = STRESS_DIR / "tasks"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def items(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    return copy.deepcopy(payload["items"])


def by_omega(records: list[dict[str, Any]], source_name: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for record in records:
        omega_id = str(record.get("omega_id", ""))
        if not omega_id:
            raise ValueError(f"Missing omega_id in {source_name}: {record}")
        if omega_id in out:
            raise ValueError(f"Duplicate omega_id {omega_id} in {source_name}")
        out[omega_id] = record
    return out


def add_origin(record: dict[str, Any], origin: str) -> dict[str, Any]:
    out = copy.deepcopy(record)
    meta = out.setdefault("stress_metadata", {})
    meta.setdefault("original_or_stress", origin)
    meta.setdefault("benchmark_split", origin)
    return out


def make_old_task_file(
    omega_id: str,
    query: dict[str, Any],
    label: dict[str, Any],
    feasible: dict[str, Any],
) -> dict[str, Any]:
    expected = label.get("expected_surviving_rule_ids", [])
    candidate = label.get("expected_source_rule_ids", [])
    metadata = {
        "original_or_stress": "original",
        "benchmark_split": "original",
        "candidate_rule_ids_expected_for_diagnostics": candidate,
        "final_valid_rule_ids_expected_for_evaluation": expected,
        "valid_rule_structures_expected": label.get("expected_valid_rule_structures") or ([expected] if expected else []),
        "expected_cell_count": len(feasible.get("valid_constraint_cells", [])),
        "challenge_types": label.get("challenge_types", []),
    }
    task = add_origin(query, "original")
    task.setdefault("stress_metadata", {}).update(metadata)
    label_out = add_origin(label, "original")
    label_out.setdefault("stress_metadata", {}).update(metadata)
    feasible_out = add_origin(feasible, "original")
    feasible_out.setdefault("stress_metadata", {}).update(metadata)
    return {
        "version": "aviation_combined_task_v1",
        "task": task,
        "rule_structure_label": label_out,
        "feasible_region_label": feasible_out,
        "stress_metadata": metadata,
    }


def collect_rule_ids(records: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for record in records:
        for field in (
            "expected_source_rule_ids",
            "expected_defeated_rule_ids",
            "expected_surviving_rule_ids",
        ):
            out.update(str(x) for x in record.get(field, []) if x)
        cert = record.get("certificate_targets", {})
        out.update(str(x) for x in cert.get("source_rule_ids", []) if x)
        ref = record.get("reference_semantics", {})
        out.update(str(x) for x in ref.get("candidate_rule_ids", []) if x)
        out.update(str(x) for x in ref.get("final_valid_rule_ids", []) if x)
    return out


def build_report(
    old_ids: list[str],
    stress_ids: list[str],
    missing_rule_ids: list[str],
) -> str:
    lines = [
        "# Combined Aviation Benchmark Dataset",
        "",
        "This dataset combines the original aviation optimization tasks and the interaction-rich aviation stress tasks for the formal KBS systematic experiments.",
        "",
        "## Counts",
        "",
        f"- Original aviation tasks: {len(old_ids)}",
        f"- Stress aviation tasks: {len(stress_ids)}",
        f"- Combined aviation tasks: {len(old_ids) + len(stress_ids)}",
        "",
        "## Original Tasks",
        "",
        ", ".join(old_ids),
        "",
        "## Stress Tasks",
        "",
        ", ".join(stress_ids),
        "",
        "## Rule Library Coverage",
        "",
        f"- Missing rule IDs in combined rule library: {', '.join(missing_rule_ids) if missing_rule_ids else 'none'}",
        "",
        "## Files",
        "",
        f"- Queries: `{OUT_DIR / 'aviation_combined_optimization_queries.json'}`",
        f"- Rule-structure labels: `{OUT_DIR / 'aviation_combined_rule_structure_labels.json'}`",
        f"- Feasible-region labels: `{OUT_DIR / 'aviation_combined_feasible_region_labels.json'}`",
        f"- Rule library: `{OUT_DIR / 'aviation_combined_rule_library.combined.json'}`",
        f"- Task files: `{TASK_DIR}`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TASK_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    old_queries = items(OLD_QUERIES)
    old_labels = items(OLD_RULE_LABELS)
    old_feasible = items(OLD_FEASIBLE)
    stress_queries = items(STRESS_QUERIES)
    stress_labels = items(STRESS_RULE_LABELS)
    stress_feasible = items(STRESS_FEASIBLE)

    old_q = by_omega(old_queries, "old queries")
    old_l = by_omega(old_labels, "old rule labels")
    old_f = by_omega(old_feasible, "old feasible labels")
    stress_q = by_omega(stress_queries, "stress queries")
    stress_l = by_omega(stress_labels, "stress rule labels")
    stress_f = by_omega(stress_feasible, "stress feasible labels")

    overlap = sorted(set(old_q) & set(stress_q))
    if overlap:
        raise ValueError(f"Old and stress task IDs overlap: {overlap}")
    if set(old_q) != set(old_l) or set(old_q) != set(old_f):
        raise ValueError("Old aviation layer omega_id sets do not match")
    if set(stress_q) != set(stress_l) or set(stress_q) != set(stress_f):
        raise ValueError("Stress aviation layer omega_id sets do not match")

    combined_queries = [add_origin(record, "original") for record in old_queries] + [
        add_origin(record, "stress") for record in stress_queries
    ]
    combined_labels = [add_origin(record, "original") for record in old_labels] + [
        add_origin(record, "stress") for record in stress_labels
    ]
    combined_feasible = [add_origin(record, "original") for record in old_feasible] + [
        add_origin(record, "stress") for record in stress_feasible
    ]

    for omega_id in sorted(old_q):
        write_json(
            TASK_DIR / f"{omega_id}.json",
            make_old_task_file(omega_id, old_q[omega_id], old_l[omega_id], old_f[omega_id]),
        )
    for source_path in sorted(STRESS_TASK_DIR.glob("AVI_STRESS_*.json")):
        shutil.copy2(source_path, TASK_DIR / source_path.name)

    rule_library = read_json(STRESS_RULE_LIBRARY)
    rule_ids = {str(rule.get("rule_id")) for rule in rule_library.get("rules", []) if rule.get("rule_id")}
    referenced_rule_ids = collect_rule_ids(combined_queries + combined_labels + combined_feasible)
    missing_rule_ids = sorted(rule_id for rule_id in referenced_rule_ids if rule_id not in rule_ids)

    write_json(
        OUT_DIR / "aviation_combined_optimization_queries.json",
        {
            "version": "aviation_combined_benchmark_layers_v1",
            "items": combined_queries,
        },
    )
    write_json(
        OUT_DIR / "aviation_combined_rule_structure_labels.json",
        {
            "version": "aviation_combined_benchmark_layers_v1",
            "items": combined_labels,
        },
    )
    write_json(
        OUT_DIR / "aviation_combined_feasible_region_labels.json",
        {
            "version": "aviation_combined_benchmark_layers_v1",
            "items": combined_feasible,
        },
    )
    rule_library["combined_dataset_metadata"] = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_rule_library": str(STRESS_RULE_LIBRARY),
        "note": "Rule library copied from aviation stress combined library; it contains the full aviation base rules plus stress extensions.",
    }
    write_json(OUT_DIR / "aviation_combined_rule_library.combined.json", rule_library)

    manifest = {
        "version": "aviation_combined_dataset_manifest_v1",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_dirs": {
            "original": str(OLD_DIR),
            "stress": str(STRESS_DIR),
        },
        "counts": {
            "original_tasks": len(old_queries),
            "stress_tasks": len(stress_queries),
            "combined_tasks": len(combined_queries),
            "combined_task_files": len(list(TASK_DIR.glob("AVI_*.json"))),
            "combined_rule_library_rules": len(rule_library.get("rules", [])),
            "missing_referenced_rule_ids": len(missing_rule_ids),
        },
        "task_ids": {
            "original": sorted(old_q),
            "stress": sorted(stress_q),
            "combined": [record["omega_id"] for record in combined_queries],
        },
        "files": {
            "queries": str(OUT_DIR / "aviation_combined_optimization_queries.json"),
            "rule_structure_labels": str(OUT_DIR / "aviation_combined_rule_structure_labels.json"),
            "feasible_region_labels": str(OUT_DIR / "aviation_combined_feasible_region_labels.json"),
            "rule_library": str(OUT_DIR / "aviation_combined_rule_library.combined.json"),
            "tasks": str(TASK_DIR),
        },
        "missing_referenced_rule_ids": missing_rule_ids,
    }
    write_json(OUT_DIR / "aviation_combined_benchmark_layers_manifest.json", manifest)
    write_json(METADATA_DIR / "aviation_combined_dataset_manifest.json", manifest)
    (REPORT_DIR / "aviation_combined_dataset_report.md").write_text(
        build_report(sorted(old_q), sorted(stress_q), missing_rule_ids),
        encoding="utf-8",
    )

    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
