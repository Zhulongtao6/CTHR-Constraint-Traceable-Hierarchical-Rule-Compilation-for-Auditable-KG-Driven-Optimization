# Section 6.2 Table 1: Aviation All Methods With Old-Candidate Profile Grounding

## Scope

- Dataset: `aviation_rule_relation_balanced_v13_150` only.
- Candidate grounding: old broad rule-library scorer with aviation recall guard.
- CTHR default valid rules: candidate-constrained profile_auto_resolver output from the grounding file.
- Flat and native symbolic baselines use the candidate_rule_ids_generated field as method-visible candidates.
- CTHR-style ASP/CP-SAT/SCIP use exactly the same predicted_valid_rule_ids grounding as CTHR default.
- Evaluation references are used only for metrics.

## Result

| Dataset | Method | Method type | Rule Precision | Rule Recall | Formal CSR | Sem-CSR | False accept | Invalid cases |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Aviation | Native MILP + HiGHS | native_symbolic | 3.0% | 7.7% | 4.7% | 2.0% | 2.7% | 147/150 (98.0%) (107 unsupported) |
| Aviation | CTHR-style MILP + HiGHS | cthr_semantic_modeling | 74.7% | 70.7% | 23.3% | 18.7% | 4.7% | 122/150 (81.3%) (20 unsupported) |

## Run Summary

```json
{
  "generated_at": "2026-07-03 15:50:03",
  "domain": "aviation",
  "dataset": {
    "dataset": "Aviation",
    "domain": "aviation",
    "root": "submission_support\\kbs_systematic_experiments_v1\\datasets\\aviation_rule_relation_balanced_v13_150",
    "tasks": 150,
    "rule_library": "submission_support\\kbs_systematic_experiments_v1\\datasets\\aviation_rule_relation_balanced_v13_150\\rule_libraries\\qwen\\full_aviation_rule_library_qwen.json",
    "grounding_result": "submission_support\\kbs_systematic_experiments_v1\\results\\av13_candidate_limit40\\section_6_3_aviation_smix_v11_150_old_candidate_recall_guard_profile_auto_resolver_candidate_to_valid_full.json",
    "constraint_templates": "submission_support\\kbs_systematic_experiments_v1\\datasets\\aviation_rule_relation_balanced_v13_150\\constraint_templates\\compiled_rule_constraint_templates.json",
    "grounding_policy": {
      "flat_and_native_symbolic": "candidate_rule_ids_generated",
      "cthr_default_and_cthr_style_backends": "predicted_valid_rule_ids"
    },
    "grounding": {
      "source": "av13_candidate_limit40",
      "mean_candidate_count": 26.893333333333334,
      "mean_cthr_predicted_valid_count": 1.92,
      "cthr_exact_match_rate": 0.41333333333333333
    }
  },
  "methods": [
    {
      "Method": "Native MILP + HiGHS",
      "Method type": "native_symbolic"
    },
    {
      "Method": "CTHR-style MILP + HiGHS",
      "Method type": "cthr_semantic_modeling"
    }
  ],
  "grounding_full": "submission_support\\kbs_systematic_experiments_v1\\results\\av13_candidate_limit40\\section_6_3_aviation_smix_v11_150_old_candidate_recall_guard_profile_auto_resolver_candidate_to_valid_full.json",
  "outputs": {
    "per_task_csv": "D:\\paper\\Neurosymbolic\\neurosymbolic-research\\cthr\\submission_support\\kbs_systematic_experiments_v1\\results\\s62_t1_av13_limit40_highs_methods_per_task.csv",
    "overall_csv": "D:\\paper\\Neurosymbolic\\neurosymbolic-research\\cthr\\submission_support\\kbs_systematic_experiments_v1\\results\\s62_t1_av13_limit40_highs_methods_overall.csv",
    "overall_md": "D:\\paper\\Neurosymbolic\\neurosymbolic-research\\cthr\\submission_support\\kbs_systematic_experiments_v1\\results\\s62_t1_av13_limit40_highs_methods_overall.md",
    "overall_json": "D:\\paper\\Neurosymbolic\\neurosymbolic-research\\cthr\\submission_support\\kbs_systematic_experiments_v1\\results\\s62_t1_av13_limit40_highs_methods_overall.json",
    "report_md": "D:\\paper\\Neurosymbolic\\neurosymbolic-research\\cthr\\submission_support\\kbs_systematic_experiments_v1\\results\\s62_t1_av13_limit40_highs_methods_report.md"
  },
  "aggregate_rows": [
    {
      "Dataset": "Aviation",
      "Method": "Native MILP + HiGHS",
      "Method type": "native_symbolic",
      "Rule Precision": "3.0%",
      "Rule Recall": "7.7%",
      "Formal CSR": "4.7%",
      "Sem-CSR": "2.0%",
      "False accept": "2.7%",
      "Invalid cases": "147/150 (98.0%) (107 unsupported)"
    },
    {
      "Dataset": "Aviation",
      "Method": "CTHR-style MILP + HiGHS",
      "Method type": "cthr_semantic_modeling",
      "Rule Precision": "74.7%",
      "Rule Recall": "70.7%",
      "Formal CSR": "23.3%",
      "Sem-CSR": "18.7%",
      "False accept": "4.7%",
      "Invalid cases": "122/150 (81.3%) (20 unsupported)"
    }
  ],
  "metric_note": "All methods run through the same Section 6.2 Table 1 evaluator. Flat consumes candidate_rule_ids_generated directly. Native ASP/CP-SAT/SCIP consume candidate_rule_ids_generated directly. CTHR default and CTHR-style ASP/CP-SAT/SCIP consume the same predicted_valid_rule_ids from the grounding file. For CTHR-style methods, rule grounding is fixed before backend-specific constraint solving."
}
```
