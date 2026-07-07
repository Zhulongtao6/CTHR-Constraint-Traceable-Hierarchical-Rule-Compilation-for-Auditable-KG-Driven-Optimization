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
| Architecture | Flat baseline | flat | 6.8% | 99.0% | 66.0% | 78.0% | 0.0% | 11/50 (22.0%) |
| Architecture | Native ASP + clingo | native_symbolic | 75.7% | 90.0% | 100.0% | 92.0% | 8.0% | 4/50 (8.0%) |
| Architecture | Native MILP + HiGHS | native_symbolic | 51.7% | 62.0% | 72.0% | 64.0% | 8.0% | 18/50 (36.0%) (14 unsupported) |
| Architecture | Native CP-SAT + OR-Tools | native_symbolic | 75.7% | 90.0% | 100.0% | 92.0% | 8.0% | 4/50 (8.0%) |
| Architecture | Native SCIP | native_symbolic | 75.7% | 90.0% | 100.0% | 92.0% | 8.0% | 4/50 (8.0%) |
| Architecture | CTHR default | cthr_semantic_modeling | 95.3% | 95.0% | 96.0% | 96.0% | 0.0% | 2/50 (4.0%) (2 unsupported) |
| Architecture | CTHR-style ASP + clingo | cthr_semantic_modeling | 95.3% | 95.0% | 96.0% | 96.0% | 0.0% | 2/50 (4.0%) (2 unsupported) |
| Architecture | CTHR-style HiGHS | cthr_semantic_modeling | 95.3% | 95.0% | 96.0% | 96.0% | 0.0% | 2/50 (4.0%) (2 unsupported) |
| Architecture | CTHR-style CP-SAT + OR-Tools | cthr_semantic_modeling | 95.3% | 95.0% | 96.0% | 96.0% | 0.0% | 2/50 (4.0%) (2 unsupported) |
| Architecture | CTHR-style SCIP | cthr_semantic_modeling | 95.3% | 95.0% | 96.0% | 96.0% | 0.0% | 2/50 (4.0%) (2 unsupported) |

## Run Summary

```json
{
  "generated_at": "2026-07-03 16:57:09",
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
      "source": "n50auto40",
      "mean_candidate_count": 34.94,
      "mean_cthr_predicted_valid_count": 1.64,
      "cthr_exact_match_rate": 0.92
    }
  },
  "methods": [
    {
      "Method": "Flat baseline",
      "Method type": "flat"
    },
    {
      "Method": "Native ASP + clingo",
      "Method type": "native_symbolic"
    },
    {
      "Method": "Native MILP + HiGHS",
      "Method type": "native_symbolic"
    },
    {
      "Method": "Native CP-SAT + OR-Tools",
      "Method type": "native_symbolic"
    },
    {
      "Method": "Native SCIP",
      "Method type": "native_symbolic"
    },
    {
      "Method": "CTHR default",
      "Method type": "cthr_semantic_modeling"
    },
    {
      "Method": "CTHR-style ASP + clingo",
      "Method type": "cthr_semantic_modeling"
    },
    {
      "Method": "CTHR-style HiGHS",
      "Method type": "cthr_semantic_modeling"
    },
    {
      "Method": "CTHR-style CP-SAT + OR-Tools",
      "Method type": "cthr_semantic_modeling"
    },
    {
      "Method": "CTHR-style SCIP",
      "Method type": "cthr_semantic_modeling"
    }
  ],
  "grounding_full": "submission_support\\kbs_systematic_experiments_v1\\results\\n50auto40\\section_6_3_architecture_ncarb50_v2_old_candidate_profile_auto_resolver_candidate_to_valid_full.json",
  "outputs": {
    "per_task_csv": "D:\\paper\\Neurosymbolic\\neurosymbolic-research\\cthr\\submission_support\\kbs_systematic_experiments_v1\\results\\s62_t1_arch_ncarb50_v2_hig_method_per_task.csv",
    "overall_csv": "D:\\paper\\Neurosymbolic\\neurosymbolic-research\\cthr\\submission_support\\kbs_systematic_experiments_v1\\results\\s62_t1_arch_ncarb50_v2_hig_method_overall.csv",
    "overall_md": "D:\\paper\\Neurosymbolic\\neurosymbolic-research\\cthr\\submission_support\\kbs_systematic_experiments_v1\\results\\s62_t1_arch_ncarb50_v2_hig_method_overall.md",
    "overall_json": "D:\\paper\\Neurosymbolic\\neurosymbolic-research\\cthr\\submission_support\\kbs_systematic_experiments_v1\\results\\s62_t1_arch_ncarb50_v2_hig_method_overall.json",
    "report_md": "D:\\paper\\Neurosymbolic\\neurosymbolic-research\\cthr\\submission_support\\kbs_systematic_experiments_v1\\results\\s62_t1_arch_ncarb50_v2_hig_method_report.md"
  },
  "aggregate_rows": [
    {
      "Dataset": "Architecture",
      "Method": "Flat baseline",
      "Method type": "flat",
      "Rule Precision": "6.8%",
      "Rule Recall": "99.0%",
      "Formal CSR": "66.0%",
      "Sem-CSR": "78.0%",
      "False accept": "0.0%",
      "Invalid cases": "11/50 (22.0%)"
    },
    {
      "Dataset": "Architecture",
      "Method": "Native ASP + clingo",
      "Method type": "native_symbolic",
      "Rule Precision": "75.7%",
      "Rule Recall": "90.0%",
      "Formal CSR": "100.0%",
      "Sem-CSR": "92.0%",
      "False accept": "8.0%",
      "Invalid cases": "4/50 (8.0%)"
    },
    {
      "Dataset": "Architecture",
      "Method": "Native MILP + HiGHS",
      "Method type": "native_symbolic",
      "Rule Precision": "51.7%",
      "Rule Recall": "62.0%",
      "Formal CSR": "72.0%",
      "Sem-CSR": "64.0%",
      "False accept": "8.0%",
      "Invalid cases": "18/50 (36.0%) (14 unsupported)"
    },
    {
      "Dataset": "Architecture",
      "Method": "Native CP-SAT + OR-Tools",
      "Method type": "native_symbolic",
      "Rule Precision": "75.7%",
      "Rule Recall": "90.0%",
      "Formal CSR": "100.0%",
      "Sem-CSR": "92.0%",
      "False accept": "8.0%",
      "Invalid cases": "4/50 (8.0%)"
    },
    {
      "Dataset": "Architecture",
      "Method": "Native SCIP",
      "Method type": "native_symbolic",
      "Rule Precision": "75.7%",
      "Rule Recall": "90.0%",
      "Formal CSR": "100.0%",
      "Sem-CSR": "92.0%",
      "False accept": "8.0%",
      "Invalid cases": "4/50 (8.0%)"
    },
    {
      "Dataset": "Architecture",
      "Method": "CTHR default",
      "Method type": "cthr_semantic_modeling",
      "Rule Precision": "95.3%",
      "Rule Recall": "95.0%",
      "Formal CSR": "96.0%",
      "Sem-CSR": "96.0%",
      "False accept": "0.0%",
      "Invalid cases": "2/50 (4.0%) (2 unsupported)"
    },
    {
      "Dataset": "Architecture",
      "Method": "CTHR-style ASP + clingo",
      "Method type": "cthr_semantic_modeling",
      "Rule Precision": "95.3%",
      "Rule Recall": "95.0%",
      "Formal CSR": "96.0%",
      "Sem-CSR": "96.0%",
      "False accept": "0.0%",
      "Invalid cases": "2/50 (4.0%) (2 unsupported)"
    },
    {
      "Dataset": "Architecture",
      "Method": "CTHR-style HiGHS",
      "Method type": "cthr_semantic_modeling",
      "Rule Precision": "95.3%",
      "Rule Recall": "95.0%",
      "Formal CSR": "96.0%",
      "Sem-CSR": "96.0%",
      "False accept": "0.0%",
      "Invalid cases": "2/50 (4.0%) (2 unsupported)"
    },
    {
      "Dataset": "Architecture",
      "Method": "CTHR-style CP-SAT + OR-Tools",
      "Method type": "cthr_semantic_modeling",
      "Rule Precision": "95.3%",
      "Rule Recall": "95.0%",
      "Formal CSR": "96.0%",
      "Sem-CSR": "96.0%",
      "False accept": "0.0%",
      "Invalid cases": "2/50 (4.0%) (2 unsupported)"
    },
    {
      "Dataset": "Architecture",
      "Method": "CTHR-style SCIP",
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
