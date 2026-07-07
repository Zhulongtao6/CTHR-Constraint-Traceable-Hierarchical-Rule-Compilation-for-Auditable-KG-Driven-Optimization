# Section 6.2 Table 1: Aviation All Methods With Old-Candidate Profile Grounding

## Scope

- Dataset: `aviation_strong_mixed_v11_150` only.
- Candidate grounding: old broad rule-library scorer with aviation recall guard.
- CTHR default valid rules: candidate-constrained profile_auto_resolver output from the grounding file.
- Flat and native symbolic baselines use the candidate_rule_ids_generated field as method-visible candidates.
- CTHR-style ASP/CP-SAT/SCIP use exactly the same predicted_valid_rule_ids grounding as CTHR default.
- Evaluation references are used only for metrics.

## Result

| Dataset | Method | Method type | Rule Precision | Rule Recall | Formal CSR | Sem-CSR | False accept | Invalid cases |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Aviation | Native ASP + clingo | native_symbolic | 7.5% | 19.3% | 0.0% | 0.0% | 0.0% | 150/150 (100.0%) |

## Run Summary

```json
{
  "generated_at": "2026-07-01 21:00:37",
  "domain": "aviation",
  "dataset": {
    "dataset": "Aviation",
    "domain": "aviation",
    "root": "D:\\paper\\Neurosymbolic\\neurosymbolic-research\\cthr\\submission_support\\kbs_systematic_experiments_v1\\datasets\\aviation_strong_mixed_v11_150",
    "tasks": 150,
    "rule_library": "D:\\paper\\Neurosymbolic\\neurosymbolic-research\\cthr\\submission_support\\kbs_systematic_experiments_v1\\datasets\\aviation_strong_mixed_v11_150\\rule_libraries\\qwen\\full_aviation_rule_library_qwen.json",
    "grounding_result": "submission_support\\kbs_systematic_experiments_v1\\results\\section_6_3_aviation_strong_mixed_v11_150_latest_slot_profile_candidate_to_valid_full.json",
    "constraint_templates": "D:\\paper\\Neurosymbolic\\neurosymbolic-research\\cthr\\submission_support\\kbs_systematic_experiments_v1\\datasets\\aviation_strong_mixed_v11_150\\evaluation_overlays\\qwen\\compiled_rule_constraint_templates.json",
    "grounding_policy": {
      "flat_and_native_symbolic": "candidate_rule_ids_generated",
      "cthr_default_and_cthr_style_backends": "predicted_valid_rule_ids"
    },
    "grounding": {
      "source": "latest slot-constrained aviation profile grounding",
      "mean_candidate_count": 22.14,
      "mean_cthr_predicted_valid_count": 1.92,
      "cthr_exact_match_rate": 0.41333333333333333
    }
  },
  "methods": [
    {
      "Method": "Native ASP + clingo",
      "Method type": "native_symbolic"
    }
  ],
  "grounding_full": "submission_support\\kbs_systematic_experiments_v1\\results\\section_6_3_aviation_strong_mixed_v11_150_latest_slot_profile_candidate_to_valid_full.json",
  "outputs": {
    "per_task_csv": "D:\\paper\\Neurosymbolic\\neurosymbolic-research\\cthr\\submission_support\\kbs_systematic_experiments_v1\\results\\s62_t1_avi150_adaptive_all_native_asp_clingo_per_task.csv",
    "overall_csv": "D:\\paper\\Neurosymbolic\\neurosymbolic-research\\cthr\\submission_support\\kbs_systematic_experiments_v1\\results\\s62_t1_avi150_adaptive_all_native_asp_clingo_overall.csv",
    "overall_md": "D:\\paper\\Neurosymbolic\\neurosymbolic-research\\cthr\\submission_support\\kbs_systematic_experiments_v1\\results\\s62_t1_avi150_adaptive_all_native_asp_clingo_overall.md",
    "overall_json": "D:\\paper\\Neurosymbolic\\neurosymbolic-research\\cthr\\submission_support\\kbs_systematic_experiments_v1\\results\\s62_t1_avi150_adaptive_all_native_asp_clingo_overall.json",
    "report_md": "D:\\paper\\Neurosymbolic\\neurosymbolic-research\\cthr\\submission_support\\kbs_systematic_experiments_v1\\results\\s62_t1_avi150_adaptive_all_native_asp_clingo_report.md"
  },
  "aggregate_rows": [
    {
      "Dataset": "Aviation",
      "Method": "Native ASP + clingo",
      "Method type": "native_symbolic",
      "Rule Precision": "7.5%",
      "Rule Recall": "19.3%",
      "Formal CSR": "0.0%",
      "Sem-CSR": "0.0%",
      "False accept": "0.0%",
      "Invalid cases": "150/150 (100.0%)"
    }
  ],
  "metric_note": "All methods run through the same Section 6.2 Table 1 evaluator. Flat consumes candidate_rule_ids_generated directly. Native ASP/CP-SAT/SCIP consume candidate_rule_ids_generated directly. CTHR default and CTHR-style ASP/CP-SAT/SCIP consume the same predicted_valid_rule_ids from the grounding file. For CTHR-style methods, rule grounding is fixed before backend-specific constraint solving."
}
```
