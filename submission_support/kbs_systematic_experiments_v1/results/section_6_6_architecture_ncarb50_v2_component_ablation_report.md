# Section 6.6: Architecture NCARB50 v2 Component Ablation

## Scope

- Dataset: `architecture_fullkg_ncarb50_v2`, 50 architecture tasks (`ARCH_FKG_51`--`ARCH_FKG_100`).
- Overlay/rule-id space: `qwen`.
- Grounding input: `D:\paper\Neurosymbolic\neurosymbolic-research\cthr\submission_support\kbs_systematic_experiments_v1\results\section_6_3_architecture_ncarb50_v2_strict_profile_candidate_to_valid_full.json`.
- This run uses the existing strict-profile candidate-to-valid grounding file; no new LLM calls are made.
- These results are separate from the main 60-task Section 6.6 benchmark denominator.
- Default row reads `predicted_valid_rule_ids` from the NCARB50 v2 strict-profile grounding output.
- Candidate-to-valid profiling ablation starts from broad generated candidates.
- Compiled-template ablation keeps valid-rule IDs fixed and disables the compiled rule-to-constraint template map.

## Overall

| Dataset | Variant | Component removed | Candidate source | Solved | Formal CSR | Sem-CSR | Rule-ID Precision | Rule-ID Recall | Exact Rule Match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Architecture NCARB50 v2 | CTHR default | None | predicted_valid_rule_ids | 48/50 (96.0%) | 86.0% | 72.0% | 81.5% | 95.0% | 37/50 (74.0%) |
| Architecture NCARB50 v2 | w/o Candidate-to-Valid Profiling | profile-based candidate-to-valid resolver | candidate_rule_ids_generated | 48/50 (96.0%) | 86.0% | 72.0% | 81.5% | 95.0% | 37/50 (74.0%) |
| Architecture NCARB50 v2 | w/o Scenario Applicability | scenario applicability filtering | candidate_rule_ids_generated | 50/50 (100.0%) | 90.0% | 76.0% | 77.5% | 97.0% | 33/50 (66.0%) |
| Architecture NCARB50 v2 | w/o Positive Closure | dependency closure + parameter/formula propagation | candidate_rule_ids_generated | 48/50 (96.0%) | 86.0% | 72.0% | 81.5% | 95.0% | 37/50 (74.0%) |
| Architecture NCARB50 v2 | w/o Negative Resolution | exclusion + override + precedence resolution | candidate_rule_ids_generated | 48/50 (96.0%) | 86.0% | 72.0% | 80.1% | 95.0% | 35/50 (70.0%) |
| Architecture NCARB50 v2 | w/o Relation-Aware Recovery | CTHR symbolic valid-rule recovery | candidate_rule_ids_after_llm_relation_filter | 50/50 (100.0%) | 90.0% | 76.0% | 77.1% | 99.0% | 31/50 (62.0%) |
| Architecture NCARB50 v2 | w/o Compiled Templates | compiled rule-to-constraint templates | predicted_valid_rule_ids | 48/50 (96.0%) | 80.0% | 66.0% | 81.5% | 95.0% | 37/50 (74.0%) |


## Run Summary

```json
{
  "title": "Section 6.6: Architecture NCARB50 v2 Component Ablation",
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
    "Candidate-to-valid profiling ablation starts from broad generated candidates.",
    "Compiled-template ablation keeps valid-rule IDs fixed and disables the compiled rule-to-constraint template map."
  ],
  "aggregate_rows": [
    {
      "Dataset": "Architecture NCARB50 v2",
      "Variant": "CTHR default",
      "Component removed": "None",
      "Candidate source": "predicted_valid_rule_ids",
      "Solved": "48/50 (96.0%)",
      "Formal CSR": "86.0%",
      "Sem-CSR": "72.0%",
      "Rule-ID Precision": "81.5%",
      "Rule-ID Recall": "95.0%",
      "Exact Rule Match": "37/50 (74.0%)"
    },
    {
      "Dataset": "Architecture NCARB50 v2",
      "Variant": "w/o Candidate-to-Valid Profiling",
      "Component removed": "profile-based candidate-to-valid resolver",
      "Candidate source": "candidate_rule_ids_generated",
      "Solved": "48/50 (96.0%)",
      "Formal CSR": "86.0%",
      "Sem-CSR": "72.0%",
      "Rule-ID Precision": "81.5%",
      "Rule-ID Recall": "95.0%",
      "Exact Rule Match": "37/50 (74.0%)"
    },
    {
      "Dataset": "Architecture NCARB50 v2",
      "Variant": "w/o Scenario Applicability",
      "Component removed": "scenario applicability filtering",
      "Candidate source": "candidate_rule_ids_generated",
      "Solved": "50/50 (100.0%)",
      "Formal CSR": "90.0%",
      "Sem-CSR": "76.0%",
      "Rule-ID Precision": "77.5%",
      "Rule-ID Recall": "97.0%",
      "Exact Rule Match": "33/50 (66.0%)"
    },
    {
      "Dataset": "Architecture NCARB50 v2",
      "Variant": "w/o Positive Closure",
      "Component removed": "dependency closure + parameter/formula propagation",
      "Candidate source": "candidate_rule_ids_generated",
      "Solved": "48/50 (96.0%)",
      "Formal CSR": "86.0%",
      "Sem-CSR": "72.0%",
      "Rule-ID Precision": "81.5%",
      "Rule-ID Recall": "95.0%",
      "Exact Rule Match": "37/50 (74.0%)"
    },
    {
      "Dataset": "Architecture NCARB50 v2",
      "Variant": "w/o Negative Resolution",
      "Component removed": "exclusion + override + precedence resolution",
      "Candidate source": "candidate_rule_ids_generated",
      "Solved": "48/50 (96.0%)",
      "Formal CSR": "86.0%",
      "Sem-CSR": "72.0%",
      "Rule-ID Precision": "80.1%",
      "Rule-ID Recall": "95.0%",
      "Exact Rule Match": "35/50 (70.0%)"
    },
    {
      "Dataset": "Architecture NCARB50 v2",
      "Variant": "w/o Relation-Aware Recovery",
      "Component removed": "CTHR symbolic valid-rule recovery",
      "Candidate source": "candidate_rule_ids_after_llm_relation_filter",
      "Solved": "50/50 (100.0%)",
      "Formal CSR": "90.0%",
      "Sem-CSR": "76.0%",
      "Rule-ID Precision": "77.1%",
      "Rule-ID Recall": "99.0%",
      "Exact Rule Match": "31/50 (62.0%)"
    },
    {
      "Dataset": "Architecture NCARB50 v2",
      "Variant": "w/o Compiled Templates",
      "Component removed": "compiled rule-to-constraint templates",
      "Candidate source": "predicted_valid_rule_ids",
      "Solved": "48/50 (96.0%)",
      "Formal CSR": "80.0%",
      "Sem-CSR": "66.0%",
      "Rule-ID Precision": "81.5%",
      "Rule-ID Recall": "95.0%",
      "Exact Rule Match": "37/50 (74.0%)"
    }
  ],
  "outputs": {
    "per_task_csv": "D:\\paper\\Neurosymbolic\\neurosymbolic-research\\cthr\\submission_support\\kbs_systematic_experiments_v1\\results\\section_6_6_architecture_ncarb50_v2_component_ablation_per_task.csv",
    "per_task_json": "D:\\paper\\Neurosymbolic\\neurosymbolic-research\\cthr\\submission_support\\kbs_systematic_experiments_v1\\results\\section_6_6_architecture_ncarb50_v2_component_ablation_per_task.json",
    "overall_csv": "D:\\paper\\Neurosymbolic\\neurosymbolic-research\\cthr\\submission_support\\kbs_systematic_experiments_v1\\results\\section_6_6_architecture_ncarb50_v2_component_ablation_overall.csv",
    "overall_md": "D:\\paper\\Neurosymbolic\\neurosymbolic-research\\cthr\\submission_support\\kbs_systematic_experiments_v1\\results\\section_6_6_architecture_ncarb50_v2_component_ablation_overall.md",
    "overall_json": "D:\\paper\\Neurosymbolic\\neurosymbolic-research\\cthr\\submission_support\\kbs_systematic_experiments_v1\\results\\section_6_6_architecture_ncarb50_v2_component_ablation_overall.json",
    "report_md": "D:\\paper\\Neurosymbolic\\neurosymbolic-research\\cthr\\submission_support\\kbs_systematic_experiments_v1\\results\\section_6_6_architecture_ncarb50_v2_component_ablation_report.md"
  }
}
```
