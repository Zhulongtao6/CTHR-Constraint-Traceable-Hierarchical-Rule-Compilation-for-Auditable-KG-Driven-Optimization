# 建筑 50 题新版数据集测试结果

日期：2026-06-26

数据集：`datasets/architecture_fullkg_ncarb50_v2`

本次测试使用旧候选规则生成和 CTHR 候选到有效规则恢复流程。方法阶段只读取算法输入、公开场景模型和规则库；参考有效规则和语义约束只在评估阶段使用。

## 配置

| 配置 | 候选生成 | 候选到有效规则恢复 | 大模型辅助过滤 |
|---|---|---|---|
| 确定性画像版 | 旧候选打分，最多 24 条 | 候选约束画像解析（profile_resolver）+ CTHR 解析 | 禁用 |
| 自动画像版 | 旧候选打分，最多 24 条 | 候选约束画像自动解析（profile_auto_resolver）+ CTHR 解析 | 启用，使用 Qwen 缓存/补缓存 |

## 汇总结果

| 配置 | 题数 | 候选/参考 | 过滤/参考 | 预测/参考 | 规则精确率 | 规则召回率 | 精确匹配率 | 语义约束满足率 | 形式可行率 | 额外规则数 | 缺失规则数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 确定性画像版 | 50 | 15.3867 | 2.1900 | 1.7400 | 0.8432 | 0.9300 | 0.7800 | 0.7000 | 0.9200 | 44 | 4 |
| 自动画像版 | 50 | 15.3867 | 1.6700 | 1.3400 | 0.9333 | 0.9300 | 0.9000 | 0.7000 | 0.9200 | 21 | 4 |

## 主要观察

1. 自动画像版显著压缩了多余规则：过滤/参考比例从 2.1900 降到 1.6700，预测/参考比例从 1.7400 降到 1.3400，额外规则数从 44 降到 21。

2. 自动画像版提高了规则精确率和精确匹配率：规则精确率从 0.8432 提高到 0.9333，精确匹配率从 0.7800 提高到 0.9000；规则召回率保持 0.9300。

3. 语义约束满足率没有随规则精确率同步提升，两个配置均为 0.7000。这说明新版 50 题中存在一批“规则编号恢复正确，但默认求解或语义检查未通过”的任务，需要和规则恢复错误分开分析。

4. 自动画像版的规则不完全匹配任务为 5 个：`ARCH_FKG_56`、`ARCH_FKG_65`、`ARCH_FKG_78`、`ARCH_FKG_79`、`ARCH_FKG_88`。其中 `ARCH_FKG_65` 和 `ARCH_FKG_88` 语义检查通过，但规则编号不是精确匹配。

5. 自动画像版中有 12 个任务规则精确匹配但语义检查未通过，另有 3 个任务规则不匹配且语义检查未通过。这提示后续应优先区分规则恢复问题、默认求解问题和新版语义参考问题。

## 输出文件

确定性画像版：

- `section_6_3_architecture_ncarb50_v2_old_candidate_profile_resolver_candidate_to_valid_full.csv`
- `section_6_3_architecture_ncarb50_v2_old_candidate_profile_resolver_candidate_to_valid_full.md`
- `section_6_3_architecture_ncarb50_v2_old_candidate_profile_resolver_candidate_to_valid_full.json`
- `section_6_3_architecture_ncarb50_v2_old_candidate_profile_resolver_candidate_to_valid_summary.json`
- `section_6_3_architecture_ncarb50_v2_old_candidate_profile_resolver_candidate_to_valid_report.md`

自动画像版：

- `../ncarb50_auto_20260626/section_6_3_architecture_ncarb50_v2_old_candidate_profile_auto_resolver_candidate_to_valid_full.csv`
- `../ncarb50_auto_20260626/section_6_3_architecture_ncarb50_v2_old_candidate_profile_auto_resolver_candidate_to_valid_full.md`
- `../ncarb50_auto_20260626/section_6_3_architecture_ncarb50_v2_old_candidate_profile_auto_resolver_candidate_to_valid_full.json`
- `../ncarb50_auto_20260626/section_6_3_architecture_ncarb50_v2_old_candidate_profile_auto_resolver_candidate_to_valid_summary.json`
- `../ncarb50_auto_20260626/section_6_3_architecture_ncarb50_v2_old_candidate_profile_auto_resolver_candidate_to_valid_report.md`

运行脚本：

- `scripts/run_architecture_ncarb50_v2_old_candidate_cthr.py`
