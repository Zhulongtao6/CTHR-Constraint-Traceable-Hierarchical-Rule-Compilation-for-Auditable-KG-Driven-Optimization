from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

from baselines.asp_rule_structure import retrieve_candidate_rules
from baselines.cthr_rule_resolver import resolve_valid_structures_with_diagnostics


THIS_DIR = Path(__file__).resolve().parent
CTHR_ROOT = THIS_DIR.parents[1]
PAPER_DIR = CTHR_ROOT / "paper"
SUPPORT_DIR = CTHR_ROOT / "submission_support" / "kbs_systematic_experiments_v1"
RESULTS_DIR = PAPER_DIR / "results"

QWEN_RULE_LIBRARY_PATH = (
    PAPER_DIR
    / "full_aviation_kg_rule_library_model_comparison"
    / "full_aviation_rule_library_qwen.json"
)
STRESS_RULE_LIBRARY_PATH = (
    SUPPORT_DIR
    / "datasets"
    / "aviation"
    / "aviation_stress_rule_library.combined.json"
)
TASK_DIR = SUPPORT_DIR / "datasets" / "aviation_combined" / "tasks"

OUT_OVERALL_CSV = RESULTS_DIR / "section_6_5_aviation_auditability_overall.csv"
OUT_OVERALL_MD = RESULTS_DIR / "section_6_5_aviation_auditability_overall.md"
OUT_PER_TASK_CSV = RESULTS_DIR / "section_6_5_aviation_auditability_per_task.csv"
OUT_REPORT_MD = RESULTS_DIR / "section_6_5_aviation_auditability_report.md"

DATASET_NAME = "Aviation benchmark (19 original + 12 stress)"
METHODS = [
    "CTHR default",
    "CTHR+ASP/clingo",
    "CTHR+SMT/Z3",
    "CTHR+MILP/HiGHS",
]

FORBIDDEN_CERTIFICATE_FIELDS = {
    "final_valid_rule_ids_expected_for_evaluation",
    "valid_rule_structures_expected",
    "candidate_rule_ids_expected_for_diagnostics",
    "solver_constraints",
    "solver_constraint_cells",
    "certificate_targets",
    "semantic_validator_labels",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_cell(row.get(field)) for field in fields})


def csv_cell(value: Any) -> Any:
    if isinstance(value, (list, dict, set, tuple)):
        return json.dumps(list(value) if isinstance(value, set) else value, ensure_ascii=False)
    if value is None:
        return ""
    return value


def pct(numerator: float, denominator: float) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def load_effective_rule_library() -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Use Qwen as the source rule library and add stress-derived rules as runtime extensions."""

    qwen_library = read_json(QWEN_RULE_LIBRARY_PATH)
    stress_library = read_json(STRESS_RULE_LIBRARY_PATH)
    qwen_by_id = {str(rule["rule_id"]): rule for rule in qwen_library.get("rules", []) if rule.get("rule_id")}
    effective_by_id = dict(qwen_by_id)
    for rule in stress_library.get("rules", []):
        rule_id = str(rule.get("rule_id", ""))
        if rule_id and rule_id not in effective_by_id:
            effective_by_id[rule_id] = rule
    effective_library = {
        "rules": list(effective_by_id.values()),
        "source": "full_aviation_rule_library_qwen.json plus stress-derived extension records",
    }
    return effective_library, qwen_by_id, effective_by_id


def load_tasks() -> list[dict[str, Any]]:
    wrappers = []
    for path in sorted(TASK_DIR.glob("AVI_*.json")):
        payload = read_json(path)
        task = dict(payload["task"])
        task["_wrapper_path"] = str(path)
        task["_original_or_stress"] = (payload.get("stress_metadata") or {}).get("original_or_stress", "unknown")
        wrappers.append(task)
    return wrappers


def scenario_for_runtime(task: dict[str, Any]) -> dict[str, Any]:
    scenario = dict(task.get("scenario_facts", {}))
    scenario.update(
        {
            "domain": task.get("domain"),
            "task_type": task.get("task_type"),
            "title": task.get("title"),
        }
    )
    return scenario


def project_to_source_rule_id(rule_id: str, qwen_by_id: dict[str, Any], effective_by_id: dict[str, Any]) -> str:
    if rule_id in qwen_by_id:
        return rule_id
    rule = effective_by_id.get(rule_id, {})
    derived = rule.get("derived_from_source_rule")
    if isinstance(derived, str) and derived in qwen_by_id:
        return derived
    return rule_id


def rule_has_valid_provenance(rule_id: str, qwen_by_id: dict[str, Any]) -> bool:
    rule = qwen_by_id.get(rule_id)
    if not rule:
        return False
    provenance = rule.get("provenance")
    if not isinstance(provenance, list) or not provenance:
        return False
    return any(item.get("document") or item.get("section") or item.get("chunk_id") for item in provenance if isinstance(item, dict))


def build_cthr_certificate(
    task: dict[str, Any],
    effective_library: dict[str, Any],
    qwen_by_id: dict[str, Any],
    effective_by_id: dict[str, Any],
) -> dict[str, Any]:
    """Generate the audit certificate from visible scenario facts and rule-library records only."""

    candidate_ids = retrieve_candidate_rules(effective_library, task, min_score=2.0)
    candidate_rules = [effective_by_id[rule_id] for rule_id in candidate_ids if rule_id in effective_by_id]
    result = resolve_valid_structures_with_diagnostics(candidate_rules, scenario_for_runtime(task))
    active_structure = result.valid_rule_structures[0] if result.valid_rule_structures else []
    projected_rule_ids = sorted(
        dict.fromkeys(project_to_source_rule_id(rule_id, qwen_by_id, effective_by_id) for rule_id in active_structure)
    )
    found = [rule_id for rule_id in projected_rule_ids if rule_has_valid_provenance(rule_id, qwen_by_id)]
    missing = [rule_id for rule_id in projected_rule_ids if rule_id not in found]
    internal_rules_have_source = all(
        rule_has_valid_provenance(project_to_source_rule_id(rule_id, qwen_by_id, effective_by_id), qwen_by_id)
        for rule_id in active_structure
    )
    certificate_present = bool(active_structure and projected_rule_ids)
    return {
        "certificate_present": certificate_present,
        "candidate_rule_count": len(candidate_ids),
        "cthr_runtime_status": result.status,
        "cthr_runtime_error": result.error,
        "active_internal_rule_structure": active_structure,
        "certificate_rule_ids": projected_rule_ids,
        "rule_ids_found_in_rule_library": sorted(found),
        "rule_ids_missing_from_rule_library": sorted(missing),
        "rule_provenance_valid_rate": pct(len(found), len(projected_rule_ids)),
        "valid_chain_present": bool(active_structure),
        "valid_chain_complete": bool(certificate_present and internal_rules_have_source),
    }


def method_available(method: str) -> tuple[bool, str]:
    if method == "CTHR default":
        return True, "cthr_runtime"
    if method == "CTHR+ASP/clingo":
        try:
            import clingo  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            return False, f"clingo unavailable: {exc}"
        return True, "certificate inherited from CTHR; clingo is backend only"
    if method == "CTHR+SMT/Z3":
        try:
            import z3  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            return False, f"z3 unavailable: {exc}"
        return True, "certificate inherited from CTHR; Z3 is backend only"
    if method == "CTHR+MILP/HiGHS":
        try:
            import highspy  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            return False, f"HiGHS unavailable: {exc}"
        return True, "certificate inherited from CTHR; HiGHS is backend only"
    return False, "unknown method"


def invalid_reason(certificate: dict[str, Any], backend_ok: bool, backend_note: str) -> str:
    reasons = []
    if not backend_ok:
        reasons.append(backend_note)
    if not certificate["certificate_present"]:
        reasons.append("certificate_not_generated")
    if certificate["rule_ids_missing_from_rule_library"]:
        reasons.append("certificate_rule_id_missing_from_qwen_rule_library")
    if not certificate["valid_chain_complete"]:
        reasons.append("valid_chain_trace_incomplete")
    return "; ".join(reasons)


def build_markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def main() -> None:
    start = time.perf_counter()
    effective_library, qwen_by_id, effective_by_id = load_effective_rule_library()
    tasks = load_tasks()
    if len(tasks) != 31:
        raise RuntimeError(f"Expected 31 aviation tasks, found {len(tasks)} in {TASK_DIR}")

    certificates = {
        task["omega_id"]: build_cthr_certificate(task, effective_library, qwen_by_id, effective_by_id)
        for task in tasks
    }

    per_task_rows: list[dict[str, Any]] = []
    for task in tasks:
        certificate = certificates[task["omega_id"]]
        for method in METHODS:
            backend_ok, backend_note = method_available(method)
            certificate_present = bool(backend_ok and certificate["certificate_present"])
            valid_chain_present = bool(backend_ok and certificate["valid_chain_present"])
            valid_chain_complete = bool(backend_ok and certificate["valid_chain_complete"])
            per_task_rows.append(
                {
                    "Dataset": DATASET_NAME,
                    "task_id": task["omega_id"],
                    "Method": method,
                    "certificate_present": certificate_present,
                    "certificate_rule_ids": certificate["certificate_rule_ids"] if certificate_present else [],
                    "rule_ids_found_in_rule_library": certificate["rule_ids_found_in_rule_library"] if certificate_present else [],
                    "rule_ids_missing_from_rule_library": certificate["rule_ids_missing_from_rule_library"] if certificate_present else [],
                    "rule_provenance_valid_rate": certificate["rule_provenance_valid_rate"] if certificate_present else 0.0,
                    "valid_chain_present": valid_chain_present,
                    "valid_chain_complete": valid_chain_complete,
                    "invalid_reason": invalid_reason(certificate, backend_ok, backend_note),
                }
            )

    overall_rows: list[dict[str, Any]] = []
    for method in METHODS:
        rows = [row for row in per_task_rows if row["Method"] == method]
        total_certificate_rules = sum(len(row["certificate_rule_ids"]) for row in rows)
        valid_certificate_rules = sum(len(row["rule_ids_found_in_rule_library"]) for row in rows)
        overall_rows.append(
            {
                "Dataset": DATASET_NAME,
                "Method": method,
                "Certificate coverage": f"{pct(sum(bool(row['certificate_present']) for row in rows), len(rows)):.1f}%",
                "Rule provenance valid": f"{pct(valid_certificate_rules, total_certificate_rules):.1f}%",
                "Valid-chain trace complete": f"{pct(sum(bool(row['valid_chain_complete']) for row in rows), len(rows)):.1f}%",
            }
        )

    write_csv(
        OUT_PER_TASK_CSV,
        per_task_rows,
        [
            "Dataset",
            "task_id",
            "Method",
            "certificate_present",
            "certificate_rule_ids",
            "rule_ids_found_in_rule_library",
            "rule_ids_missing_from_rule_library",
            "rule_provenance_valid_rate",
            "valid_chain_present",
            "valid_chain_complete",
            "invalid_reason",
        ],
    )
    write_csv(
        OUT_OVERALL_CSV,
        overall_rows,
        ["Dataset", "Method", "Certificate coverage", "Rule provenance valid", "Valid-chain trace complete"],
    )

    OUT_OVERALL_MD.write_text(
        build_markdown_table(
            ["Dataset", "Method", "Certificate coverage", "Rule provenance valid", "Valid-chain trace complete"],
            [
                [
                    row["Dataset"],
                    row["Method"],
                    row["Certificate coverage"],
                    row["Rule provenance valid"],
                    row["Valid-chain trace complete"],
                ]
                for row in overall_rows
            ],
        )
        + "\n",
        encoding="utf-8",
    )

    backend_notes = {method: method_available(method)[1] for method in METHODS}
    report_lines = [
        "# Section 6.5 Aviation Auditability Experiment",
        "",
        "This experiment evaluates audit traceability only. It does not report Formal CSR, Sem-CSR, rule precision, rule recall, or objective-gap metrics.",
        "",
        "## Dataset",
        "",
        f"- Dataset: {DATASET_NAME}",
        f"- Task files: `{TASK_DIR}`",
        f"- Number of tasks: {len(tasks)}",
        "- Architecture benchmark is not included because the architecture baseline experiments are not complete yet.",
        "",
        "## Rule Library",
        "",
        f"- Primary source rule library: `{QWEN_RULE_LIBRARY_PATH}`",
        f"- Effective runtime extension for stress rules: `{STRESS_RULE_LIBRARY_PATH}`",
        "- The certificate projects stress-derived rule IDs back to their `derived_from_source_rule` when that source rule exists in the Qwen aviation rule library.",
        "",
        "## Certificate Source Control",
        "",
        "- Certificates are generated by the CTHR runtime from visible task/scenario fields, CTHR candidate retrieval, CTHR valid-structure resolution, and rule provenance in the aviation rule library.",
        "- ASP/clingo, SMT/Z3, and MILP/HiGHS are treated as CTHR-guided backends. Their certificates come from CTHR rule structures/provenance, not from native ASP/SMT/MILP solver output.",
        "- No expected/reference field is used to generate certificates.",
        "- Forbidden certificate-generation fields checked by design: `{}`.".format(
            ", ".join(sorted(FORBIDDEN_CERTIFICATE_FIELDS))
        ),
        "",
        "## Backend Availability",
        "",
    ]
    for method, note in backend_notes.items():
        report_lines.append(f"- {method}: {note}")
    report_lines.extend(
        [
            "",
            "## Overall Results",
            "",
            OUT_OVERALL_MD.read_text(encoding="utf-8").strip(),
            "",
            "## Notes",
            "",
            "- `Certificate coverage` is the fraction of returned method outputs that carry a CTHR certificate.",
            "- `Rule provenance valid` checks whether source-projected certificate rule IDs exist in `full_aviation_rule_library_qwen.json` and have provenance records.",
            "- `Valid-chain trace complete` checks whether every active internal CTHR runtime rule has a source-projected provenance entry in the certificate.",
            f"- Runtime: {time.perf_counter() - start:.2f} seconds.",
        ]
    )
    OUT_REPORT_MD.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "overall_csv": str(OUT_OVERALL_CSV),
                "overall_md": str(OUT_OVERALL_MD),
                "per_task_csv": str(OUT_PER_TASK_CSV),
                "report_md": str(OUT_REPORT_MD),
                "tasks": len(tasks),
                "methods": METHODS,
                "overall": overall_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
