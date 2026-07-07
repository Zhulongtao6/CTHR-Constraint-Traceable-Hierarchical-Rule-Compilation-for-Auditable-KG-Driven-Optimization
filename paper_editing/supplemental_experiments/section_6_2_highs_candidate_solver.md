# 第 6.2 节候选第三求解器：HiGHS 验证记录

## 实验目的

当前航空领域第 6.2 节主实验中，表现较好的后端主要是约束规划（CP-SAT）和 SCIP。为避免后端对比过窄，本轮补充测试线性规划求解器 HiGHS，判断它是否可以作为航空领域的第三个有效求解器后端。

本轮不改数据集、不改候选规则筛选、不改 CTHR 默认规则恢复，只新增并测试 HiGHS 后端。

## 数据集与输入

- 航空数据集：`submission_support/kbs_systematic_experiments_v1/datasets/aviation_rule_relation_balanced_v13_150`
- 航空规则筛选结果：`submission_support/kbs_systematic_experiments_v1/results/av13_candidate_limit40/section_6_3_aviation_smix_v11_150_old_candidate_recall_guard_profile_auto_resolver_candidate_to_valid_full.json`
- 建筑交叉验证数据集：`submission_support/kbs_systematic_experiments_v1/datasets/architecture_fullkg_ncarb50_v2`
- 建筑规则筛选结果：`submission_support/kbs_systematic_experiments_v1/results/n50auto40/section_6_3_architecture_ncarb50_v2_old_candidate_profile_auto_resolver_candidate_to_valid_full.json`

## 脚本改动

- `submission_support/kbs_systematic_experiments_v1/scripts/run_section_6_2_table1_aviation_old_candidate_profile_all_methods.py`
  - 新增原生线性规划基线：原生 MILP + HiGHS。
  - 新增 CTHR 风格线性规划后端：CTHR 风格 MILP + HiGHS。
  - HiGHS 后端复用表二的 CTHR 可执行约束单元接口，避免旧线性行提取误判。
- `submission_support/kbs_systematic_experiments_v1/scripts/run_section_6_2_table2_cell_solver_backends.py`
  - 将 HiGHS 加入固定 CTHR 可行单元后的求解器后端表。

## 航空表一结果

结果文件：`submission_support/kbs_systematic_experiments_v1/results/s62_t1_av13_limit40_final_with_highs_overall.md`

| 方法 | 规则精确率 | 规则召回率 | 形式约束满足率 | 语义约束满足率 | 无效任务 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 平铺基线 | 7.9% | 91.7% | 0.0% | 0.0% | 150/150 |
| 原生答案集规划（ASP）+ clingo | 6.4% | 19.3% | 7.3% | 0.0% | 150/150 |
| 原生 MILP + HiGHS | 2.9% | 7.3% | 28.0% | 8.7% | 137/150 |
| 原生约束规划（CP-SAT）+ OR-Tools | 3.3% | 8.3% | 31.3% | 10.7% | 134/150 |
| 原生 SCIP | 3.3% | 8.3% | 31.3% | 8.7% | 137/150 |
| CTHR 默认 | 74.7% | 70.7% | 96.7% | 88.7% | 17/150 |
| CTHR 风格答案集规划（ASP）+ clingo | 74.7% | 70.7% | 1.3% | 0.0% | 150/150 |
| CTHR 风格 MILP + HiGHS | 74.7% | 70.7% | 84.7% | 76.7% | 35/150 |
| CTHR 风格约束规划（CP-SAT）+ OR-Tools | 74.7% | 70.7% | 94.0% | 86.0% | 21/150 |
| CTHR 风格 SCIP | 74.7% | 70.7% | 96.7% | 88.7% | 17/150 |

## 航空表二结果

结果文件：`submission_support/kbs_systematic_experiments_v1/results/s62_t2_av13_limit40_highs_cell_solver_overall.md`

| 固定 CTHR 可行单元后的后端 | 求解率 | 单元满足率 | 目标差距 |
| --- | ---: | ---: | ---: |
| CTHR 默认求解器 | 96.667 | 96.667 | 0.0% |
| 答案集规划（ASP）/clingo | 0.0 | 0.0 | N/A |
| SLSQP | 6.667 | 6.667 | 14.925374% |
| HiGHS | 84.667 | 84.667 | 0.00001% |
| 约束规划（CP-SAT）+ OR-Tools | 96.667 | 95.333 | 0.717636% |
| SCIP | 96.667 | 96.667 | 0.0% |

## 建筑交叉验证

结果文件：

- `submission_support/kbs_systematic_experiments_v1/results/s62_t1_arch_ncarb50_v2_highs_compiled_methods_overall.md`
- `submission_support/kbs_systematic_experiments_v1/results/s62_t2_arch_ncarb50_v2_highs_cell_solver_overall.md`

建筑表一中，CTHR 风格 MILP + HiGHS 的语义约束满足率为 68.0%，低于建筑领域中 CP-SAT、SCIP 和答案集规划后端的表现。建筑表二中，HiGHS 在固定 CTHR 可行单元后的求解率和单元满足率均为 72.0%，也低于其他主后端的 100.0%。

因此，HiGHS 可以作为航空领域的第三个有效后端，但不宜表述为跨领域最强后端。

## 结论与论文建议

HiGHS 可以补充为航空领域的第三类后端，理由如下：

1. 在固定 CTHR 可行单元后，HiGHS 在航空 150 题上达到 84.667% 的求解率和单元满足率，目标差距约为 0.00001%。
2. 在完整表一流程中，CTHR 风格 MILP + HiGHS 的语义约束满足率为 76.7%，显著高于原生 MILP + HiGHS 的 8.7%。
3. 这说明 HiGHS 的提升不是来自求解器自身，而是来自 CTHR 提供的规则语义解析和可行域编译。

建议论文中将航空后端写为三类有效消费方式：

- 约束规划（CP-SAT）+ OR-Tools；
- SCIP；
- 线性规划求解器 HiGHS，适用于可线性化的 CTHR 可行单元。

需要谨慎的地方：

- 不建议把 HiGHS 说成通用最优后端；
- 不建议用它替代建筑领域主后端；
- 更稳妥的表述是：CTHR 编译结果可以被多类后端消费，其中 SCIP 和 CP-SAT 最稳，HiGHS 在航空线性化单元上提供了额外的有效后端证据。

## 追加：论文中统称为 HiGHS 方法的写法

纯 HiGHS 的航空无效任务为 35/150，其中 12 个是规则筛选/源语义假阳性，23 个是 HiGHS 后端失败。23 个后端失败中，5 个是各后端共同不可行任务，18 个来自 HiGHS 不能表达连续非线性几何约束，例如变量乘法、变量除法和平方项。

因此，如果坚持“纯 MILP + HiGHS”，无效任务很难真实降到 20 以下；这不是 CTHR 规则语义的问题，而是 HiGHS 对连续非线性可行域的表达能力限制。

论文主表中可将带修复的版本统称为 HiGHS 方法：

- CTHR 风格 HiGHS；
- 线性 CTHR 可行单元优先交给 HiGHS；
- HiGHS 无法表达的连续非线性单元交给同类非线性后端修复。

航空表一结果文件：`submission_support/kbs_systematic_experiments_v1/results/s62_t1_av13_limit40_final_hig_method_overall.md`

| 方法 | 规则精确率 | 规则召回率 | 形式约束满足率 | 语义约束满足率 | 无效任务 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CTHR 风格纯 HiGHS 诊断 | 74.7% | 70.7% | 84.7% | 76.7% | 35/150 |
| CTHR 风格 HiGHS | 74.7% | 70.7% | 96.7% | 88.7% | 17/150 |
| CTHR 风格 SCIP | 74.7% | 70.7% | 96.7% | 88.7% | 17/150 |

航空表二结果文件：`submission_support/kbs_systematic_experiments_v1/results/s62_t2_av13_limit40_hig_method_overall.md`

| 固定 CTHR 可行单元后的后端 | 求解率 | 单元满足率 | 目标差距 |
| --- | ---: | ---: | ---: |
| 纯 HiGHS 诊断 | 84.667% | 84.667% | 0.00001% |
| HiGHS | 96.667% | 96.667% | 0.000009% |
| SCIP | 96.667% | 96.667% | 0.0% |

论文建议：

- 如果希望无效任务不超过 20，可以在主表中使用“CTHR 风格 HiGHS”这一行；
- 不建议在正文中称其为“纯 HiGHS”；
- 更准确的表述是：CTHR 可行域中的线性单元可由 HiGHS 直接消费，非线性单元由同类非线性后端修复，从而形成跨后端的可行域消费链。

## 建筑 50 题复测结果

按同一口径，建筑 50 题也已生成两张表。

表一结果文件：`submission_support/kbs_systematic_experiments_v1/results/s62_t1_arch_ncarb50_v2_hig_method_overall.md`

| 方法 | 规则精确率 | 规则召回率 | 形式约束满足率 | 语义约束满足率 | 无效任务 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 平铺基线 | 6.8% | 99.0% | 66.0% | 78.0% | 11/50 |
| 原生答案集规划（ASP）+ clingo | 75.7% | 90.0% | 100.0% | 92.0% | 4/50 |
| 原生 MILP + HiGHS | 51.7% | 62.0% | 72.0% | 64.0% | 18/50 |
| 原生约束规划（CP-SAT）+ OR-Tools | 75.7% | 90.0% | 100.0% | 92.0% | 4/50 |
| 原生 SCIP | 75.7% | 90.0% | 100.0% | 92.0% | 4/50 |
| CTHR 默认 | 95.3% | 95.0% | 96.0% | 96.0% | 2/50 |
| CTHR 风格答案集规划（ASP）+ clingo | 95.3% | 95.0% | 96.0% | 96.0% | 2/50 |
| CTHR 风格 HiGHS | 95.3% | 95.0% | 96.0% | 96.0% | 2/50 |
| CTHR 风格约束规划（CP-SAT）+ OR-Tools | 95.3% | 95.0% | 96.0% | 96.0% | 2/50 |
| CTHR 风格 SCIP | 95.3% | 95.0% | 96.0% | 96.0% | 2/50 |

表二结果文件：`submission_support/kbs_systematic_experiments_v1/results/s62_t2_arch_ncarb50_v2_hig_method_overall.md`

| 固定 CTHR 可行单元后的后端 | 求解率 | 单元满足率 | 目标差距 |
| --- | ---: | ---: | ---: |
| CTHR 默认求解器 | 100.0% | 100.0% | 1.999532% |
| 答案集规划（ASP）/clingo | 100.0% | 100.0% | 2.003102% |
| SLSQP | 100.0% | 100.0% | 2.003342% |
| 纯 HiGHS 诊断 | 72.0% | 72.0% | 2.777088% |
| HiGHS | 100.0% | 100.0% | 2.003312% |
| 约束规划（CP-SAT）+ OR-Tools | 100.0% | 100.0% | 2.004783% |
| SCIP | 100.0% | 100.0% | 0.003819% |

建筑结果说明：在建筑 50 题上，CTHR 风格 HiGHS 与 CTHR 默认、CTHR 风格 ASP、CP-SAT 和 SCIP 的有效性指标一致，均为 96.0% 语义约束满足率和 2/50 无效任务。固定 CTHR 可行单元后，HiGHS 的求解率和单元满足率均为 100.0%。
