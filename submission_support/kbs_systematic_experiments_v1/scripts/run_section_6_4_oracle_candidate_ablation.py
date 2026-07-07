from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import run_section_6_2_table1_fullkg_pipeline as fullkg
import run_section_6_4_architecture_rule_library_compare as architecture
import run_section_6_4_aviation_rule_library_compare as aviation


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


MODES = [
    {
        "mode": "oracle_candidate",
        "description": "Use strong-aligned overlay rule IDs as candidates, then run CTHR default valid-rule recovery.",
        "pass_as_valid": False,
    },
    {
        "mode": "oracle_valid_upper",
        "description": "Use strong-aligned overlay rule IDs as both candidates and selected valid rules.",
        "pass_as_valid": True,
    },
]

INCLUDE_PROVIDERS = {"qwen", "deepseek", "xiaomi_mimo"}


DOMAIN_CONFIGS = [
    {
        "domain": "aviation",
        "dataset": "Aviation",
        "module": aviation,
        "root": aviation.AVIATION_ROOT,
        "spec_builder": aviation.aviation_spec,
        "default_full_csv": RESULTS_DIR / "section_6_4_aviation_canonical_projected_semantic_table.csv",
        "default_strict_csv": RESULTS_DIR / "section_6_4_aviation_strict_common_canonical_projected_table.csv",
    },
    {
        "domain": "architecture",
        "dataset": "Architecture",
        "module": architecture,
        "root": architecture.ARCHITECTURE_ROOT,
        "spec_builder": architecture.architecture_spec,
        "default_full_csv": RESULTS_DIR / "section_6_4_architecture_canonical_projected_semantic_table.csv",
        "default_strict_csv": RESULTS_DIR / "section_6_4_architecture_strict_common_canonical_projected_table.csv",
    },
]


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
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({header: csv_cell(row.get(header)) for header in headers})


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def parse_pct(value: Any) -> float | None:
    text = str(value).strip()
    if not text or text == "N/A":
        return None
    if text.endswith("%"):
        text = text[:-1]
    try:
        return float(text)
    except ValueError:
        return None


def pct_delta(new_value: str, old_value: str | None) -> str:
    new_pct = parse_pct(new_value)
    old_pct = parse_pct(old_value)
    if new_pct is None or old_pct is None:
        return ""
    return f"{new_pct - old_pct:+.1f} pp"


def rule_precision(predicted: list[str], reference: list[str]) -> float:
    if not predicted:
        return 0.0
    return len(set(predicted) & set(reference)) / len(set(predicted))


def rule_recall(predicted: list[str], reference: list[str]) -> float:
    if not reference:
        return 1.0
    return len(set(predicted) & set(reference)) / len(set(reference))


def project_rule_ids(predicted: list[str], model_to_canonical: dict[str, list[str]]) -> list[str]:
    return sorted({canonical for rule_id in predicted for canonical in model_to_canonical.get(str(rule_id), [])})


def task_ids_for_split(domain_root: Path, split: str) -> list[str] | None:
    if split == "full":
        return None
    payload = read_json(domain_root / "STRICT_COMMON_TASKS.json")
    return [str(item) for item in payload.get("task_ids", [])]


def default_semcsr_by_model(path: Path) -> dict[str, str]:
    return {str(row["Model"]): str(row.get("Sem-CSR", "")) for row in read_csv_rows(path)}


def support_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "candidate_zero": sum(1 for row in rows if row.get("unsupported_reason") == "no_grounded_candidates"),
        "cthr_no_valid": sum(1 for row in rows if row.get("unsupported_reason") == "cthr_no_valid_rules"),
        "unsupported": sum(1 for row in rows if row.get("unsupported_reason")),
    }


def aggregate_rows(
    rows: list[dict[str, Any]],
    *,
    domain: str,
    split: str,
    mode: str,
    model_name: str,
    overlay_key: str,
    strong_coverage: str,
    default_semcsr: str | None,
) -> dict[str, Any]:
    total = len(rows)
    counts = support_counts(rows)
    invalid = sum(1 for row in rows if row.get("invalid_case"))
    sem_csr = pct(sum(1 for row in rows if row.get("semantic_valid")) / max(1, total))
    row = {
        "Domain": domain,
        "Split": split,
        "Mode": mode,
        "Model": model_name,
        "Overlay": overlay_key,
        "Strong canonical coverage": strong_coverage,
        "Canonical Rule Precision": pct(sum(float(row["canonical_rule_precision"]) for row in rows) / max(1, total)),
        "Canonical Rule Recall": pct(sum(float(row["canonical_rule_recall"]) for row in rows) / max(1, total)),
        "Model-ID Rule Precision": pct(sum(float(row["model_rule_precision"]) for row in rows) / max(1, total)),
        "Model-ID Rule Recall": pct(sum(float(row["model_rule_recall"]) for row in rows) / max(1, total)),
        "Formal CSR": pct(sum(1 for row in rows if row.get("formal_feasible")) / max(1, total)),
        "Sem-CSR": sem_csr,
        "Default Sem-CSR": default_semcsr or "",
        "Delta Sem-CSR": pct_delta(sem_csr, default_semcsr),
        "False accept": pct(sum(1 for row in rows if row.get("false_accept")) / max(1, total)),
        "Candidate zero": counts["candidate_zero"],
        "CTHR no valid": counts["cthr_no_valid"],
        "Unsupported tasks": counts["unsupported"],
        "Mean oracle candidates": f"{sum(int(row['oracle_candidate_count']) for row in rows) / max(1, total):.2f}",
        "Invalid cases": f"{invalid}/{total} ({100.0 * invalid / max(1, total):.1f}%)",
    }
    if counts["unsupported"]:
        row["Invalid cases"] += f" ({counts['unsupported']} unsupported)"
    return row


def evaluate_model(
    domain_cfg: dict[str, Any],
    model_spec: dict[str, Any],
    split: str,
    mode_cfg: dict[str, Any],
    default_semcsr: str | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    module = domain_cfg["module"]
    overlay_key = str(model_spec["overlay_key"])
    model_name = str(model_spec["model"])
    spec = domain_cfg["spec_builder"](
        rule_library=Path(model_spec["rule_library"]),
        grounding_full=Path(model_spec["grounding_full"]),
        evaluation_references=module.overlay_file(overlay_key, "evaluation_references.json"),
        constraint_templates=module.overlay_file(overlay_key, "compiled_rule_constraint_templates.json"),
    )
    algorithm_inputs = fullkg.item_map(spec.algorithm_inputs)
    scenario_models = fullkg.item_map(spec.scenario_models)
    references = fullkg.item_map(spec.evaluation_references)
    templates_by_rule = fullkg.constraint_template_map(spec.constraint_templates)
    rule_library = fullkg.read_json(spec.rule_library)
    rule_by_id = {str(rule["rule_id"]): rule for rule in rule_library.get("rules", []) if rule.get("rule_id")}
    canonical_references = module.canonical_reference_by_task()
    model_to_canonical = module.overlay_model_to_canonical(overlay_key)
    alignment_summary = module.overlay_alignment_summary(overlay_key)
    strong_coverage = (
        f"{alignment_summary.get('exact_or_strong_aligned_canonical_rule_count', alignment_summary.get('aligned_canonical_rule_count', 0))}/"
        f"{alignment_summary.get('canonical_rule_count', 0)}"
    )
    wanted_task_ids = task_ids_for_split(Path(domain_cfg["root"]), split)
    task_ids = wanted_task_ids if wanted_task_ids is not None else sorted(algorithm_inputs)

    rows: list[dict[str, Any]] = []
    for task_id in task_ids:
        grounding_task = dict(algorithm_inputs[task_id])
        query = fullkg.prepare_query(grounding_task, scenario_models[task_id])
        query["_compiled_rule_constraint_templates_by_id"] = templates_by_rule
        reference = references[task_id]
        feasible = fullkg.reference_feasible(reference, query)
        model_reference_ids = fullkg.reference_rule_ids(reference)
        candidate_rules = [rule_by_id[rule_id] for rule_id in model_reference_ids if rule_id in rule_by_id]
        selected_valid_ids = model_reference_ids if mode_cfg["pass_as_valid"] else None
        start = time.perf_counter()
        result = fullkg.run_method(
            spec,
            "CTHR default",
            query,
            grounding_task,
            candidate_rules,
            rule_library,
            rule_by_id,
            selected_valid_ids,
        )
        elapsed = (time.perf_counter() - start) * 1000.0
        predicted_model = sorted(result.predicted_rule_ids) if result.supported else []
        predicted_canonical = project_rule_ids(predicted_model, model_to_canonical)
        reference_canonical = canonical_references.get(task_id, [])
        sem_ok = (
            fullkg.semantic_valid(feasible, result.optimized_x, predicted_model, model_reference_ids)
            if result.supported
            else False
        )
        formal_ok = bool(result.formal_feasible) if result.supported else False
        rows.append(
            {
                "Domain": domain_cfg["domain"],
                "Split": split,
                "Mode": mode_cfg["mode"],
                "Model": model_name,
                "Overlay": overlay_key,
                "Dataset": domain_cfg["dataset"],
                "task_id": task_id,
                "oracle_candidate_count": len(model_reference_ids),
                "predicted_model_rule_ids": predicted_model,
                "model_reference_rule_ids": model_reference_ids,
                "projected_canonical_rule_ids": predicted_canonical,
                "canonical_reference_rule_ids": reference_canonical,
                "model_rule_precision": rule_precision(predicted_model, model_reference_ids),
                "model_rule_recall": rule_recall(predicted_model, model_reference_ids),
                "canonical_rule_precision": rule_precision(predicted_canonical, reference_canonical),
                "canonical_rule_recall": rule_recall(predicted_canonical, reference_canonical),
                "formal_feasible": formal_ok,
                "semantic_valid": sem_ok,
                "false_accept": bool(formal_ok and not sem_ok),
                "invalid_case": bool(not sem_ok),
                "unsupported_reason": "" if result.supported else result.unsupported_reason,
                "runtime_ms": round(elapsed, 3),
            }
        )

    aggregate = aggregate_rows(
        rows,
        domain=domain_cfg["domain"],
        split=split,
        mode=str(mode_cfg["mode"]),
        model_name=model_name,
        overlay_key=overlay_key,
        strong_coverage=strong_coverage,
        default_semcsr=default_semcsr,
    )
    return aggregate, rows


def markdown_table(rows: list[dict[str, Any]], headers: list[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(csv_cell(row.get(header)) for header in headers) + " |")
    return "\n".join(lines)


def build_report(aggregate_rows_all: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    headers = [
        "Domain",
        "Split",
        "Mode",
        "Model",
        "Canonical Rule Precision",
        "Canonical Rule Recall",
        "Formal CSR",
        "Sem-CSR",
        "Default Sem-CSR",
        "Delta Sem-CSR",
        "Candidate zero",
        "CTHR no valid",
        "Unsupported tasks",
    ]
    return "\n".join(
        [
            "# Section 6.4 Oracle Candidate Ablation",
            "",
            "This diagnostic keeps the core benchmark and source semantic oracle fixed. It replaces the candidate grounding input with the per-model overlay's strong-aligned rule IDs.",
            "",
            "## Aggregate Results",
            "",
            markdown_table(aggregate_rows_all, headers),
            "",
            "## Mode Definitions",
            "",
            markdown_table(MODES, ["mode", "description"]),
            "",
            "## Run Summary",
            "",
            "```json",
            json.dumps(summary, ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )


def main() -> None:
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    aggregate_rows_all: list[dict[str, Any]] = []
    task_rows_all: list[dict[str, Any]] = []
    per_domain_summary: dict[str, Any] = {}

    for domain_cfg in DOMAIN_CONFIGS:
        domain = str(domain_cfg["domain"])
        default_semcsr_maps = {
            "full": default_semcsr_by_model(Path(domain_cfg["default_full_csv"])),
            "strict_common": default_semcsr_by_model(Path(domain_cfg["default_strict_csv"])),
        }
        per_domain_summary[domain] = {"default_semcsr": default_semcsr_maps, "models": {}}
        for split in ["full", "strict_common"]:
            for mode_cfg in MODES:
                model_specs = [
                    model_spec
                    for model_spec in domain_cfg["module"].MODEL_SPECS
                    if str(model_spec.get("provider")) in INCLUDE_PROVIDERS
                ]
                for model_spec in model_specs:
                    model_name = str(model_spec["model"])
                    print(
                        f"running {domain} {split} {mode_cfg['mode']} {model_name}",
                        flush=True,
                    )
                    aggregate, rows = evaluate_model(
                        domain_cfg,
                        model_spec,
                        split,
                        mode_cfg,
                        default_semcsr_maps[split].get(model_name),
                    )
                    aggregate_rows_all.append(aggregate)
                    task_rows_all.extend(rows)
                    per_domain_summary[domain]["models"].setdefault(model_name, {})[
                        f"{split}:{mode_cfg['mode']}"
                    ] = aggregate

    outputs = {
        "aggregate_csv": RESULTS_DIR / "section_6_4_oracle_candidate_ablation_table.csv",
        "task_rows_csv": RESULTS_DIR / "section_6_4_oracle_candidate_ablation_task_rows.csv",
        "report_md": RESULTS_DIR / "section_6_4_oracle_candidate_ablation_report.md",
        "summary_json": RESULTS_DIR / "section_6_4_oracle_candidate_ablation_summary.json",
    }
    aggregate_headers = [
        "Domain",
        "Split",
        "Mode",
        "Model",
        "Overlay",
        "Strong canonical coverage",
        "Canonical Rule Precision",
        "Canonical Rule Recall",
        "Model-ID Rule Precision",
        "Model-ID Rule Recall",
        "Formal CSR",
        "Sem-CSR",
        "Default Sem-CSR",
        "Delta Sem-CSR",
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
        "Dataset",
        "task_id",
        "oracle_candidate_count",
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
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "started_at": started_at,
        "mode_definitions": MODES,
        "domains": per_domain_summary,
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    write_csv(outputs["aggregate_csv"], aggregate_rows_all, aggregate_headers)
    write_csv(outputs["task_rows_csv"], task_rows_all, task_headers)
    write_json(outputs["summary_json"], summary)
    outputs["report_md"].write_text(build_report(aggregate_rows_all, summary), encoding="utf-8")
    print(json.dumps({"aggregate": aggregate_rows_all, "outputs": summary["outputs"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
