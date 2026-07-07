from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CTHR_ROOT = ROOT.parents[1]
SCRIPTS_DIR = ROOT / "scripts"
DEFAULT_DATASET_ROOT = ROOT / "datasets" / "aviation_strong_mixed_v11_150"
DEFAULT_OUT_DIR = ROOT / "results" / "aviation_smix_v11_150_20260630"

sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(CTHR_ROOT))

import run_old_candidate_llm_filter_grounding as oldrun  # noqa: E402
import run_section_6_2_table1_fullkg_pipeline as full  # noqa: E402


def dataset_spec(dataset_root: Path, out_dir: Path) -> full.DatasetSpec:
    return full.DatasetSpec(
        name="Aviation-Strong-Mixed-v11-150",
        domain="aviation",
        root=dataset_root,
        algorithm_inputs=dataset_root / "algorithm_inputs" / "aviation_strong_mixed_algorithm_inputs.json",
        scenario_models=dataset_root / "scenario_models" / "aviation_strong_mixed_public_scenario_models.json",
        evaluation_references=dataset_root / "evaluation_references" / "aviation_strong_mixed_evaluation_references.json",
        rule_library=dataset_root / "rule_libraries" / "qwen" / "full_aviation_rule_library_qwen.json",
        grounding_full=out_dir
        / "section_6_3_aviation_smix_v11_150_old_candidate_recall_guard_profile_auto_resolver_candidate_to_valid_full.json",
        constraint_templates=dataset_root / "constraint_templates" / "compiled_rule_constraint_templates.json",
    )


def metric_value(value: Any) -> float:
    numeric = float(value)
    return 0.0 if math.isnan(numeric) else numeric


def rounded(value: float) -> float:
    return value if isinstance(value, float) and math.isnan(value) else round(float(value), 4)


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for key in (
        "Candidate / Reference Ratio",
        "Filtered / Reference Ratio",
        "Predicted / Reference Ratio",
        "Rule-ID Precision",
        "Rule-ID Recall",
    ):
        out[key] = rounded(metric_value(out[key]))
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "task count": len(rows),
        "mean Candidate / Reference Ratio": statistics.mean(
            metric_value(row["Candidate / Reference Ratio"]) for row in rows
        ),
        "mean Filtered / Reference Ratio": statistics.mean(
            metric_value(row["Filtered / Reference Ratio"]) for row in rows
        ),
        "mean Predicted / Reference Ratio": statistics.mean(
            metric_value(row["Predicted / Reference Ratio"]) for row in rows
        ),
        "mean Rule-ID Precision": statistics.mean(metric_value(row["Rule-ID Precision"]) for row in rows),
        "mean Rule-ID Recall": statistics.mean(metric_value(row["Rule-ID Recall"]) for row in rows),
        "exact match rate": statistics.mean(1.0 if row["Exact Match"] else 0.0 for row in rows),
        "total extra rules": sum(len(row["extra_rule_ids"]) for row in rows),
        "total missing rules": sum(len(row["missing_rule_ids"]) for row in rows),
        "zero prediction tasks": sum(1 for row in rows if int(row["predicted_valid_rule_count"]) == 0),
    }


def compact_failures(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures = [
        row
        for row in rows
        if row["extra_rule_ids"] or row["missing_rule_ids"] or not bool(row["Exact Match"])
    ]
    return failures


def markdown_report(
    *,
    spec: full.DatasetSpec,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    valid_resolver: str,
    limit: int,
    out_paths: dict[str, str],
    elapsed_sec: float,
) -> str:
    headers = [
        "task_id",
        "target_interaction",
        "candidate_rule_count",
        "filtered_candidate_rule_count",
        "reference_valid_rule_count",
        "predicted_valid_rule_count",
        "Candidate / Reference Ratio",
        "Filtered / Reference Ratio",
        "Predicted / Reference Ratio",
        "Rule-ID Precision",
        "Rule-ID Recall",
        "Exact Match",
        "extra_rule_ids",
        "missing_rule_ids",
    ]
    main_rows = [public_row(row) for row in rows[:10]]
    failure_rows = [public_row(row) for row in compact_failures(rows)]
    lines = [
        "# 航空强业务混合第十一版 150 题第 6.3 节实验",
        "",
        f"- 数据集路径：`{spec.root}`",
        f"- 题数：{summary['task count']}",
        f"- 候选生成：`{oldrun.oldmap.OLD_MAPPING_NAME}`，最多 {limit} 条，并启用航空召回保护（recall guard）",
        f"- 候选到有效规则恢复：`{valid_resolver}`",
        "- 大模型辅助过滤：未启用。航空领域的自动画像解析不会触发在线大模型重排。",
        "- 方法可见输入：算法输入、公开场景模型、规则库。",
        "- 仅评估使用：参考有效规则、隐藏语义参考、评估覆盖层。",
        f"- 运行耗时：{elapsed_sec:.3f} 秒",
        "",
        "## 汇总结果",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
        f"| 候选/参考（Candidate/Ref） | {summary['mean Candidate / Reference Ratio']:.4f} |",
        f"| 过滤/参考（Filtered/Ref） | {summary['mean Filtered / Reference Ratio']:.4f} |",
        f"| 预测/参考（Predicted/Ref） | {summary['mean Predicted / Reference Ratio']:.4f} |",
        f"| 规则精确率（Rule-ID Precision） | {summary['mean Rule-ID Precision']:.4f} |",
        f"| 规则召回率（Rule-ID Recall） | {summary['mean Rule-ID Recall']:.4f} |",
        f"| 精确匹配率（Exact Match） | {summary['exact match rate']:.4f} |",
        f"| 额外规则数（Extra） | {summary['total extra rules']} |",
        f"| 缺失规则数（Missing） | {summary['total missing rules']} |",
        f"| 零预测任务数 | {summary['zero prediction tasks']} |",
        "",
        "## 前 10 题结果",
        "",
        full.markdown_table(main_rows, headers),
        "",
        "## 非精确匹配任务",
        "",
    ]
    if failure_rows:
        lines.append(full.markdown_table(failure_rows, headers))
    else:
        lines.append("所有任务均达到规则集合精确匹配。")
    lines.extend(
        [
            "",
            "## 输出文件",
            "",
            *[f"- {name}: `{path}`" for name, path in out_paths.items()],
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = dataset_spec(args.dataset_root, out_dir)

    rows, _old_summary = oldrun.run_dataset(
        spec,
        limit=args.limit,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_cache=args.llm_cache,
        valid_resolver=args.valid_resolver,
        aviation_recall_guard=True,
    )
    public_rows: list[dict[str, Any]] = []
    for row in rows:
        clean = dict(row)
        clean["Rule-ID Precision"] = metric_value(clean["Rule-ID Precision"])
        clean["Rule-ID Recall"] = metric_value(clean["Rule-ID Recall"])
        clean.pop("_candidate_to_valid_resolver", None)
        public_rows.append(public_row(clean))

    summary = summarize(public_rows) | {
        "dataset": spec.name,
        "dataset_root": str(spec.root),
        "candidate_source": oldrun.oldmap.OLD_MAPPING_NAME,
        "old_mapping_limit": args.limit,
        "aviation_recall_guard": True,
        "candidate_to_valid": args.valid_resolver,
        "llm_assisted_filtering_enabled": False,
        "rule_library": str(spec.rule_library),
        "constraint_templates": str(spec.constraint_templates),
        "audit": {
            "used_expected_candidate_field": False,
            "used_reference_valid_rules_as_method_input": False,
            "used_solver_constraints_as_method_input": False,
            "used_reference_cells_as_method_input": False,
            "used_semantic_validator_as_method_input": False,
        },
    }

    prefix = f"section_6_3_aviation_smix_v11_150_old_candidate_recall_guard_{args.valid_resolver}_candidate_to_valid"
    headers = [
        "Dataset",
        "task_id",
        "target_interaction",
        "candidate_source",
        "candidate_rule_count",
        "filtered_candidate_rule_count",
        "reference_valid_rule_count",
        "predicted_valid_rule_count",
        "Candidate / Reference Ratio",
        "Filtered / Reference Ratio",
        "Predicted / Reference Ratio",
        "Rule-ID Precision",
        "Rule-ID Recall",
        "Exact Match",
        "candidate_rule_ids_generated",
        "candidate_rule_ids_after_llm_relation_filter",
        "reference_valid_rule_ids",
        "predicted_valid_rule_ids",
        "extra_rule_ids",
        "missing_rule_ids",
    ]
    full_csv = out_dir / f"{prefix}_full.csv"
    full_md = out_dir / f"{prefix}_full.md"
    full_json = out_dir / f"{prefix}_full.json"
    summary_json = out_dir / f"{prefix}_summary.json"
    report_md = out_dir / f"{prefix}_report.md"
    full.write_csv(full_csv, public_rows, headers)
    full_md.write_text(full.markdown_table(public_rows, headers), encoding="utf-8")
    full.write_json(full_json, public_rows)
    full.write_json(summary_json, summary)

    out_paths = {
        "full_csv": str(full_csv),
        "full_md": str(full_md),
        "full_json": str(full_json),
        "summary_json": str(summary_json),
        "report_md": str(report_md),
    }
    elapsed_sec = time.perf_counter() - start
    report_md.write_text(
        markdown_report(
            spec=spec,
            rows=public_rows,
            summary=summary,
            valid_resolver=args.valid_resolver,
            limit=args.limit,
            out_paths=out_paths,
            elapsed_sec=elapsed_sec,
        ),
        encoding="utf-8",
    )
    return summary | {"outputs": out_paths, "elapsed_sec": round(elapsed_sec, 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Section 6.3 recovery on aviation strong mixed v11 150.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument(
        "--valid-resolver",
        choices=sorted(oldrun.VALID_RESOLVER_MODES),
        default="profile_auto_resolver",
    )
    parser.add_argument("--llm-provider", default="qwen")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument(
        "--llm-cache",
        type=Path,
        default=ROOT / "results" / "llm_grounding_relation_filter_cache.json",
    )
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
