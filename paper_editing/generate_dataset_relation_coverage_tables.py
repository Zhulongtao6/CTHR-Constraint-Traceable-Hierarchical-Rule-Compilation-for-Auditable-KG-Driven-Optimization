import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(r"D:\paper\Neurosymbolic\neurosymbolic-research\cthr")
DATASETS = ROOT / "submission_support" / "kbs_systematic_experiments_v1" / "datasets"
OUT = ROOT / "paper_editing"

AVIATION = DATASETS / "aviation_strong_mixed_v11_150"
ARCHITECTURE = DATASETS / "architecture_fullkg_ncarb50_v2"

RELATION_COLUMNS = ["SCA", "BE", "DEP", "MRC", "EXC", "PRE"]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task", *RELATION_COLUMNS, "rules"])
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows):
    lines = [
        "| 任务 | SCA | BE | DEP | MRC | EXC | PRE | 规则数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {task} | {SCA} | {BE} | {DEP} | {MRC} | {EXC} | {PRE} | {rules} |".format(**row)
        )
    return "\n".join(lines)


def summarize(rows):
    summary = {col: sum(int(row[col]) for row in rows) for col in RELATION_COLUMNS}
    summary["tasks"] = len(rows)
    summary["rules_total"] = sum(int(row["rules"]) for row in rows)
    summary["rules_min"] = min(int(row["rules"]) for row in rows)
    summary["rules_max"] = max(int(row["rules"]) for row in rows)
    summary["rules_avg"] = summary["rules_total"] / len(rows)
    combo_counter = Counter(
        "+".join(col for col in RELATION_COLUMNS if int(row[col])) or "none"
        for row in rows
    )
    return summary, combo_counter


def aviation_rows():
    data = read_json(AVIATION / "evaluation_references" / "aviation_strong_mixed_evaluation_references.json")
    rows = []
    for item in data["items"]:
        task_id = item["omega_id"]
        structure = item.get("rule_structure", {})
        surviving = structure.get("expected_surviving_rule_ids", [])
        behavior = structure.get("expected_rule_behavior", {})
        metadata = item.get("benchmark_metadata", {})
        relation_text = json.dumps({
            "title": item.get("title", ""),
            "expected_source_rule_ids": structure.get("expected_source_rule_ids", []),
            "expected_activated_rule_ids": structure.get("expected_activated_rule_ids", []),
            "expected_defeated_rule_ids": structure.get("expected_defeated_rule_ids", []),
            "expected_surviving_rule_ids": surviving,
            "expected_rule_behavior": behavior,
            "challenge_types": structure.get("challenge_types", []),
        }, ensure_ascii=False).lower()

        rules = len(surviving)
        sca = int(metadata.get("semantic_rule_count", 0) > 0 or "semantic_rule_recovery" in structure.get("challenge_types", []))
        dep = int(metadata.get("numeric_rule_count", 0) > 0 or "numeric_constraint_compilation" in structure.get("challenge_types", []))
        mrc = int(rules >= 2)
        be = int(bool(structure.get("expected_defeated_rule_ids")) or bool(behavior.get("should_exclude")))
        exc = int("exception" in relation_text or "override" in relation_text or "except" in relation_text)
        pre = int("precedence" in relation_text or "priority" in relation_text or "优先级" in relation_text)

        rows.append({
            "task": task_id,
            "SCA": sca,
            "BE": be,
            "DEP": dep,
            "MRC": mrc,
            "EXC": exc,
            "PRE": pre,
            "rules": rules,
        })
    return rows


def architecture_rows():
    data = read_json(ARCHITECTURE / "RELATION_COVERAGE_NCARB50_AUDIT.json")
    sca_types = {"scenario_conditioned_applicability", "applicability_resolution", "documentation_condition"}
    dep_types = {"formula_propagation", "dependency_or_formula_propagation", "multi_constraint_single_rule"}
    mrc_types = {"multi_rule_conjunction"}
    be_types = {"branch_exclusion", "branch_or_exclusion", "branch", "exclusion"}
    exc_types = {"exception_or_override", "override_resolution"}
    pre_types = {"precedence"}

    rows = []
    for item in data["per_task_coverage"]:
        challenge_types = set(item.get("challenge_types", []))
        rows.append({
            "task": item["task_id"],
            "SCA": int(bool(challenge_types & sca_types)),
            "BE": int(bool(challenge_types & be_types)),
            "DEP": int(bool(challenge_types & dep_types)),
            "MRC": int(bool(challenge_types & mrc_types)),
            "EXC": int(bool(challenge_types & exc_types)),
            "PRE": int(bool(challenge_types & pre_types)),
            "rules": len(item.get("rule_ids", [])),
        })
    return rows


def write_summary(avi_rows, arch_rows):
    avi_summary, avi_combos = summarize(avi_rows)
    arch_summary, arch_combos = summarize(arch_rows)

    def summary_row(name, summary):
        return (
            f"| {name} | {summary['tasks']} | {summary['SCA']} | {summary['BE']} | "
            f"{summary['DEP']} | {summary['MRC']} | {summary['EXC']} | {summary['PRE']} | "
            f"{summary['rules_total']} | {summary['rules_avg']:.2f} |"
        )

    lines = [
        "# 新版数据集逐题规则关系覆盖统计",
        "",
        "## 六类关系缩写",
        "",
        "- SCA：场景条件适用性。",
        "- BE：分支或排斥处理。",
        "- DEP：依赖或公式传播。",
        "- MRC：多规则合取。",
        "- EXC：例外或覆盖解析。",
        "- PRE：优先级推理。",
        "",
        "表中每个关系列为逐题二值覆盖；1 表示该题覆盖对应关系，0 表示未覆盖。规则数为该题隐藏参考中的有效规则数量。",
        "",
        "## 总体分布",
        "",
        "| 数据集 | 任务数 | SCA | BE | DEP | MRC | EXC | PRE | 规则总数 | 平均规则数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        summary_row("航空强业务第三类 150 题", avi_summary),
        summary_row("建筑新增 50 题", arch_summary),
        "",
        "## 关系组合分布",
        "",
        "### 航空强业务第三类 150 题",
        "",
        "| 关系组合 | 题数 |",
        "|---|---:|",
    ]
    for combo, count in sorted(avi_combos.items()):
        lines.append(f"| {combo} | {count} |")
    lines += [
        "",
        "### 建筑新增 50 题",
        "",
        "| 关系组合 | 题数 |",
        "|---|---:|",
    ]
    for combo, count in sorted(arch_combos.items()):
        lines.append(f"| {combo} | {count} |")

    lines += [
        "",
        "## 航空逐题表",
        "",
        markdown_table(avi_rows),
        "",
        "## 建筑逐题表",
        "",
        markdown_table(arch_rows),
        "",
        "## 映射说明",
        "",
        "航空数据集没有单独的六关系覆盖审计文件，因此本表从隐藏评估参考推导：语义规则数量大于 0 记为 SCA，数值规则编译记为 DEP，有效规则数不少于 2 记为 MRC，存在被排除或失效规则记为 BE，显式异常或覆盖语义记为 EXC，显式优先级语义记为 PRE。",
        "",
        "建筑数据集使用 RELATION_COVERAGE_NCARB50_AUDIT.json 中的 challenge_types 归一化：scenario_conditioned_applicability、applicability_resolution、documentation_condition 记为 SCA；formula_propagation、dependency_or_formula_propagation、multi_constraint_single_rule 记为 DEP；multi_rule_conjunction 记为 MRC；exception_or_override、override_resolution 记为 EXC；precedence 记为 PRE。",
    ]
    (OUT / "dataset_relation_coverage_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    avi_rows = aviation_rows()
    arch_rows = architecture_rows()
    write_csv(OUT / "dataset_relation_coverage_aviation_v11.csv", avi_rows)
    write_csv(OUT / "dataset_relation_coverage_architecture_ncarb50.csv", arch_rows)
    write_summary(avi_rows, arch_rows)

    avi_summary, _ = summarize(avi_rows)
    arch_summary, _ = summarize(arch_rows)
    print(json.dumps({
        "aviation": avi_summary,
        "architecture": arch_summary,
        "outputs": [
            str(OUT / "dataset_relation_coverage_summary.md"),
            str(OUT / "dataset_relation_coverage_aviation_v11.csv"),
            str(OUT / "dataset_relation_coverage_architecture_ncarb50.csv"),
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
