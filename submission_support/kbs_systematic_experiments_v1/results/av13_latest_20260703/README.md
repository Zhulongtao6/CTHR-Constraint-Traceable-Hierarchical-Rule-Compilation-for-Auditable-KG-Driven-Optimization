# 第 6.3 节航空第十三版 150 题实验结果

## 数据与方法口径

- 数据集：`D:\paper\Neurosymbolic\neurosymbolic-research\cthr\submission_support\kbs_systematic_experiments_v1\datasets\aviation_rule_relation_balanced_v13_150`
- 题数：150
- 候选规则生成：旧版候选映射打分，每题最多 24 条，并启用航空召回保护。
- 候选到有效规则恢复：画像自动解析器，即当前第 6.3 节默认 CTHR 恢复方法。
- 大模型辅助过滤：未启用。
- 方法可见输入：算法输入、公开场景模型、规则库、约束模板。
- 仅用于评估：参考有效规则、隐藏语义参考、评估覆盖层、语义验证器。
- 生成脚本说明：本次复用第十一版航空脚本入口，因此输出文件名里仍含 `v11`；实际 `dataset_root` 已指向第十三版数据集。

## 汇总结果

| 指标 | 结果 |
| --- | ---: |
| 候选/参考比（Candidate/Reference） | 11.2367 |
| 过滤/参考比（Filtered/Reference） | 1.4133 |
| 预测/参考比（Predicted/Reference） | 0.9600 |
| 规则编号精确率（Rule-ID Precision） | 0.7467 |
| 规则编号召回率（Rule-ID Recall） | 0.7067 |
| 精确匹配率（Exact Match） | 0.4133 |
| 精确匹配题数 | 62 / 150 |
| 额外规则数 | 76 |
| 缺失规则数 | 88 |
| 零预测题数 | 0 |

## 逐题误差分解

| 类型 | 题数 |
| --- | ---: |
| 完全匹配 | 62 |
| 同时有额外规则和缺失规则 | 76 |
| 仅缺失规则 | 12 |
| 仅额外规则 | 0 |

| 计数项 | 数量 |
| --- | ---: |
| 候选规则总数 | 3371 |
| 过滤后候选规则总数 | 424 |
| 预测有效规则总数 | 288 |
| 参考有效规则总数 | 300 |

## 与上一轮第十三版结果对照

| 指标 | 上一轮第十三版 | 本轮第十三版 |
| --- | ---: | ---: |
| 候选/参考比（Candidate/Reference） | 11.2367 | 11.2367 |
| 过滤/参考比（Filtered/Reference） | 1.4133 | 1.4133 |
| 预测/参考比（Predicted/Reference） | 0.9600 | 0.9600 |
| 规则编号精确率（Rule-ID Precision） | 0.7467 | 0.7467 |
| 规则编号召回率（Rule-ID Recall） | 0.7067 | 0.7067 |
| 精确匹配率（Exact Match） | 0.4133 | 0.4133 |
| 额外规则数 | 76 | 76 |
| 缺失规则数 | 88 | 88 |

## 结果解释

本轮在最新第十三版航空 150 题上复现实验，结果与上一轮第十三版默认方法完全一致，说明当前第 6.3 节候选到有效规则恢复流程是确定且稳定的。

候选集合平均约为参考有效规则集合的 11.2367 倍，经过 CTHR 恢复后预测/参考比降到 0.9600，规则编号精确率为 0.7467，规则编号召回率为 0.7067。精确匹配率为 0.4133，主要因为每题参考规则均为 2 条，必须两条完全一致才计为完全匹配；未完全匹配的 88 题中没有零命中题。

## 输出文件

- 逐题表格：`section_6_3_aviation_smix_v11_150_old_candidate_recall_guard_profile_auto_resolver_candidate_to_valid_full.csv`
- 逐题明细：`section_6_3_aviation_smix_v11_150_old_candidate_recall_guard_profile_auto_resolver_candidate_to_valid_full.json`
- 汇总结果：`section_6_3_aviation_smix_v11_150_old_candidate_recall_guard_profile_auto_resolver_candidate_to_valid_summary.json`
- 自动报告：`section_6_3_aviation_smix_v11_150_old_candidate_recall_guard_profile_auto_resolver_candidate_to_valid_report.md`
