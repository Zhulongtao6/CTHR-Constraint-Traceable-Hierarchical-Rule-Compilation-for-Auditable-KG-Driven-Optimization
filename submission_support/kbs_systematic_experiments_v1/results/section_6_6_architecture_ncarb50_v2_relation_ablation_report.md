# Section 6.6: Architecture NCARB50 v2 Relation Ablation

## Scope

- Dataset: `architecture_fullkg_ncarb50_v2`, 50 architecture tasks (`ARCH_FKG_51`--`ARCH_FKG_100`).
- Overlay/rule-id space: `qwen`.
- Grounding input: `D:\paper\Neurosymbolic\neurosymbolic-research\cthr\submission_support\kbs_systematic_experiments_v1\results\section_6_3_architecture_ncarb50_v2_strict_profile_candidate_to_valid_full.json`.
- This run uses the existing strict-profile candidate-to-valid grounding file; no new LLM calls are made.
- These results are separate from the main 60-task Section 6.6 benchmark denominator.
- Default row reads `predicted_valid_rule_ids` from the NCARB50 v2 strict-profile grounding output.
- Ablation rows rerun CTHR valid-rule recovery after disabling one relation family.
- The w/o Applicability row uses broad generated candidates because scenario applicability itself is ablated.

## Overall

| Dataset | Variant | Relation removed | Solved | Formal CSR | Sem-CSR | Rule-ID Precision | Rule-ID Recall | Exact Rule Match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Architecture NCARB50 v2 | CTHR default | None | 48/50 (96.0%) | 86.0% | 72.0% | 81.5% | 95.0% | 37/50 (74.0%) |
| Architecture NCARB50 v2 | w/o Applicability | scenario-conditioned applicability | 50/50 (100.0%) | 90.0% | 76.0% | 77.5% | 97.0% | 33/50 (66.0%) |
| Architecture NCARB50 v2 | w/o Dependency | dependency | 48/50 (96.0%) | 86.0% | 72.0% | 81.5% | 95.0% | 37/50 (74.0%) |
| Architecture NCARB50 v2 | w/o Exclusion | exclusion / alternative branch | 48/50 (96.0%) | 86.0% | 72.0% | 80.1% | 95.0% | 35/50 (70.0%) |
| Architecture NCARB50 v2 | w/o Override | exception override | 48/50 (96.0%) | 86.0% | 72.0% | 81.4% | 95.0% | 37/50 (74.0%) |
| Architecture NCARB50 v2 | w/o Precedence | precedence | 48/50 (96.0%) | 86.0% | 72.0% | 81.5% | 95.0% | 37/50 (74.0%) |
| Architecture NCARB50 v2 | w/o Parameter | parameter / formula propagation | 48/50 (96.0%) | 86.0% | 72.0% | 81.5% | 95.0% | 37/50 (74.0%) |


## Run Summary

```json
{
  "title": "Section 6.6: Architecture NCARB50 v2 Relation Ablation",
  "dataset": "architecture_fullkg_ncarb50_v2",
  "overlay": "qwen",
  "scope": "Section 6.6 ablation on the new 50-task architecture NCARB50 v2 dataset only; not part of the main 60-task benchmark denominator.",
  "dataset_summary": {
    "dataset": "Architecture NCARB50 v2",
    "domain": "architecture",
    "tasks": 50,
    "rule_library": "D:\\paper\\Neurosymbolic\\neurosymbolic-research\\cthr\\submission_support\\kbs_systematic_experiments_v1\\datasets\\architecture_fullkg_ncarb50_v2\\rule_libraries\\qwen\\full_architecture_rule_library_qwen.json",
    "grounding_full": "D:\\paper\\Neurosymbolic\\neurosymbolic-research\\cthr\\submission_support\\kbs_systematic_experiments_v1\\results\\section_6_3_architecture_ncarb50_v2_strict_profile_candidate_to_valid_full.json",
    "constraint_templates": "D:\\paper\\Neurosymbolic\\neurosymbolic-research\\cthr\\submission_support\\kbs_systematic_experiments_v1\\datasets\\architecture_fullkg_ncarb50_v2\\evaluation_overlays\\qwen\\compiled_rule_constraint_templates.json",
    "rule_library_rules": 1585,
    "constraint_template_rules": 46,
    "constraint_template_count": 56
  },
  "notes": [
    "Default row reads `predicted_valid_rule_ids` from the NCARB50 v2 strict-profile grounding output.",
    "Ablation rows rerun CTHR valid-rule recovery after disabling one relation family.",
    "The w/o Applicability row uses broad generated candidates because scenario applicability itself is ablated."
  ],
  "aggregate_rows": [
    {
      "Dataset": "Architecture NCARB50 v2",
      "Variant": "CTHR default",
      "Relation removed": "None",
      "Solved": "48/50 (96.0%)",
      "Formal CSR": "86.0%",
      "Sem-CSR": "72.0%",
      "Rule-ID Precision": "81.5%",
      "Rule-ID Recall": "95.0%",
      "Exact Rule Match": "37/50 (74.0%)"
    },
    {
      "Dataset": "Architecture NCARB50 v2",
      "Variant": "w/o Applicability",
      "Relation removed": "scenario-conditioned applicability",
      "Solved": "50/50 (100.0%)",
      "Formal CSR": "90.0%",
      "Sem-CSR": "76.0%",
      "Rule-ID Precision": "77.5%",
      "Rule-ID Recall": "97.0%",
      "Exact Rule Match": "33/50 (66.0%)"
    },
    {
      "Dataset": "Architecture NCARB50 v2",
      "Variant": "w/o Dependency",
      "Relation removed": "dependency",
      "Solved": "48/50 (96.0%)",
      "Formal CSR": "86.0%",
      "Sem-CSR": "72.0%",
      "Rule-ID Precision": "81.5%",
      "Rule-ID Recall": "95.0%",
      "Exact Rule Match": "37/50 (74.0%)"
    },
    {
      "Dataset": "Architecture NCARB50 v2",
      "Variant": "w/o Exclusion",
      "Relation removed": "exclusion / alternative branch",
      "Solved": "48/50 (96.0%)",
      "Formal CSR": "86.0%",
      "Sem-CSR": "72.0%",
      "Rule-ID Precision": "80.1%",
      "Rule-ID Recall": "95.0%",
      "Exact Rule Match": "35/50 (70.0%)"
    },
    {
      "Dataset": "Architecture NCARB50 v2",
      "Variant": "w/o Override",
      "Relation removed": "exception override",
      "Solved": "48/50 (96.0%)",
      "Formal CSR": "86.0%",
      "Sem-CSR": "72.0%",
      "Rule-ID Precision": "81.4%",
      "Rule-ID Recall": "95.0%",
      "Exact Rule Match": "37/50 (74.0%)"
    },
    {
      "Dataset": "Architecture NCARB50 v2",
      "Variant": "w/o Precedence",
      "Relation removed": "precedence",
      "Solved": "48/50 (96.0%)",
      "Formal CSR": "86.0%",
      "Sem-CSR": "72.0%",
      "Rule-ID Precision": "81.5%",
      "Rule-ID Recall": "95.0%",
      "Exact Rule Match": "37/50 (74.0%)"
    },
    {
      "Dataset": "Architecture NCARB50 v2",
      "Variant": "w/o Parameter",
      "Relation removed": "parameter / formula propagation",
      "Solved": "48/50 (96.0%)",
      "Formal CSR": "86.0%",
      "Sem-CSR": "72.0%",
      "Rule-ID Precision": "81.5%",
      "Rule-ID Recall": "95.0%",
      "Exact Rule Match": "37/50 (74.0%)"
    }
  ],
  "outputs": {
    "per_task_csv": "D:\\paper\\Neurosymbolic\\neurosymbolic-research\\cthr\\submission_support\\kbs_systematic_experiments_v1\\results\\section_6_6_architecture_ncarb50_v2_relation_ablation_per_task.csv",
    "per_task_json": "D:\\paper\\Neurosymbolic\\neurosymbolic-research\\cthr\\submission_support\\kbs_systematic_experiments_v1\\results\\section_6_6_architecture_ncarb50_v2_relation_ablation_per_task.json",
    "overall_csv": "D:\\paper\\Neurosymbolic\\neurosymbolic-research\\cthr\\submission_support\\kbs_systematic_experiments_v1\\results\\section_6_6_architecture_ncarb50_v2_relation_ablation_overall.csv",
    "overall_md": "D:\\paper\\Neurosymbolic\\neurosymbolic-research\\cthr\\submission_support\\kbs_systematic_experiments_v1\\results\\section_6_6_architecture_ncarb50_v2_relation_ablation_overall.md",
    "overall_json": "D:\\paper\\Neurosymbolic\\neurosymbolic-research\\cthr\\submission_support\\kbs_systematic_experiments_v1\\results\\section_6_6_architecture_ncarb50_v2_relation_ablation_overall.json",
    "report_md": "D:\\paper\\Neurosymbolic\\neurosymbolic-research\\cthr\\submission_support\\kbs_systematic_experiments_v1\\results\\section_6_6_architecture_ncarb50_v2_relation_ablation_report.md"
  }
}
```
