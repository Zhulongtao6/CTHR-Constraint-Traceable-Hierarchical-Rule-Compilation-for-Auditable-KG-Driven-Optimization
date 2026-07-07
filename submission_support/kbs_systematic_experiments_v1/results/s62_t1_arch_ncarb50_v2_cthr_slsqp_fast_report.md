# Section 6.2 Table 1: Architecture All Methods With Old-Candidate Profile Grounding

## Scope

- Dataset: `architecture_fullkg_ncarb50_v2` only.
- Candidate grounding: old broad rule-library scorer.
- CTHR default valid rules: candidate-constrained profile_auto_resolver output from the grounding file.
- Flat and native symbolic baselines use the candidate_rule_ids_generated field as method-visible candidates.
- CTHR-style ASP/CP-SAT/SCIP use exactly the same predicted_valid_rule_ids grounding as CTHR default.
- Evaluation references are used only for metrics.

## Result

| Dataset | Method | Method type | Rule Precision | Rule Recall | Formal CSR | Sem-CSR | False accept | Invalid cases |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Architecture | CTHR-style SLSQP | cthr_semantic_modeling | 95.3% | 95.0% | 96.0% | 96.0% | 0.0% | 2/50 (4.0%) (2 unsupported) |

## Run Summary

```json
{
  "generated_at": "2026-07-03 15:35:42",
  "domain": "architecture",
  "dataset": {
    "dataset": "Architecture",
    "domain": "architecture",
    "root": "submission_support\\kbs_systematic_experiments_v1\\datasets\\architecture_fullkg_ncarb50_v2",
    "tasks": 50,
    "rule_library": "submission_support\\kbs_systematic_experiments_v1\\datasets\\architecture_fullkg_ncarb50_v2\\rule_libraries\\full_architecture_rule_library_qwen.json",
    "grounding_result": "submission_support\\kbs_systematic_experiments_v1\\results\\n50auto40\\section_6_3_architecture_ncarb50_v2_old_candidate_profile_auto_resolver_candidate_to_valid_full.json",
    "constraint_templates": "submission_support\\kbs_systematic_experiments_v1\\datasets\\architecture_fullkg_ncarb50_v2\\constraint_templates\\compiled_rule_constraint_templates.json",
    "grounding_policy": {
      "flat_and_native_symbolic": "candidate_rule_ids_generated",
      "cthr_default_and_cthr_style_backends": "predicted_valid_rule_ids"
    },
    "grounding": {
      "source": "ncarb50 v2 auto40 profile auto grounding",
      "mean_candidate_count": 34.94,
      "mean_cthr_predicted_valid_count": 1.64,
      "cthr_exact_match_rate": 0.92
    }
  },
  "methods": [
    {
      "Method": "CTHR-style SLSQP",
      "Method type": "cthr_semantic_modeling"
    }
  ],
  "grounding_full": "submission_support\\kbs_systematic_experiments_v1\\results\\n50auto40\\section_6_3_architecture_ncarb50_v2_old_candidate_profile_auto_resolver_candidate_to_valid_full.json",
  "outputs": {
    "per_task_csv": "D:\\paper\\Neurosymbolic\\neurosymbolic-research\\cthr\\submission_support\\kbs_systematic_experiments_v1\\results\\s62_t1_arch_ncarb50_v2_cthr_slsqp_fast_per_task.csv",
    "overall_csv": "D:\\paper\\Neurosymbolic\\neurosymbolic-research\\cthr\\submission_support\\kbs_systematic_experiments_v1\\results\\s62_t1_arch_ncarb50_v2_cthr_slsqp_fast_overall.csv",
    "overall_md": "D:\\paper\\Neurosymbolic\\neurosymbolic-research\\cthr\\submission_support\\kbs_systematic_experiments_v1\\results\\s62_t1_arch_ncarb50_v2_cthr_slsqp_fast_overall.md",
    "overall_json": "D:\\paper\\Neurosymbolic\\neurosymbolic-research\\cthr\\submission_support\\kbs_systematic_experiments_v1\\results\\s62_t1_arch_ncarb50_v2_cthr_slsqp_fast_overall.json",
    "report_md": "D:\\paper\\Neurosymbolic\\neurosymbolic-research\\cthr\\submission_support\\kbs_systematic_experiments_v1\\results\\s62_t1_arch_ncarb50_v2_cthr_slsqp_fast_report.md"
  },
  "aggregate_rows": [
    {
      "Dataset": "Architecture",
      "Method": "CTHR-style SLSQP",
      "Method type": "cthr_semantic_modeling",
      "Rule Precision": "95.3%",
      "Rule Recall": "95.0%",
      "Formal CSR": "96.0%",
      "Sem-CSR": "96.0%",
      "False accept": "0.0%",
      "Invalid cases": "2/50 (4.0%) (2 unsupported)"
    }
  ],
  "metric_note": "All methods run through the same Section 6.2 Table 1 evaluator. Flat consumes candidate_rule_ids_generated directly. Native ASP/CP-SAT/SCIP consume candidate_rule_ids_generated directly. CTHR default and CTHR-style ASP/CP-SAT/SCIP consume the same predicted_valid_rule_ids from the grounding file. For CTHR-style methods, rule grounding is fixed before backend-specific constraint solving."
}
```
