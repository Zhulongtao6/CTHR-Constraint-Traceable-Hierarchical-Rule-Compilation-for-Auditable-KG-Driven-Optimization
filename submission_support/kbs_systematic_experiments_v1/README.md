# KBS Systematic Experiment Support Package

This directory is the clean workspace for the systematic experiments intended to support the KBS submission.

Previous exploratory or validation runs remain under `paper/results` and the original benchmark-layer folders. New formal experiment scripts, outputs, logs, and reports should be placed here so the submission-support materials remain reproducible and easy to audit.

## Layout

- `datasets/aviation_combined`: formal aviation dataset combining the original `AVI_OPT_xx` tasks and the interaction-rich `AVI_STRESS_xx` tasks.
- `datasets/aviation_fullkg_clean`: source-grounded aviation full-KG benchmark with a fixed core and per-model rule-ID evaluation overlays.
- `datasets/aviation`: frozen stress-only aviation snapshot kept for traceability.
- `datasets/architecture`: frozen snapshot of the current architecture benchmark layers and task files.
- `scripts`: formal experiment scripts added for the systematic evaluation.
- `results`: machine-readable experiment outputs such as CSV and JSON summaries.
- `reports`: human-readable experiment reports and paper-ready tables.
- `logs`: command logs and runtime traces.
- `metadata`: dataset snapshot manifests, run manifests, and configuration records.

## Dataset Snapshots

The aviation snapshot comes from:

`D:\paper\Neurosymbolic\neurosymbolic-research\cthr\paper\aviation_stress_benchmark_layers`

The architecture snapshot comes from:

`D:\paper\Neurosymbolic\neurosymbolic-research\cthr\paper\architecture_generated_benchmark_layers`

Future formal aviation full-KG experiments should read from `datasets/aviation_fullkg_clean`. The older `datasets/aviation_combined` snapshot is retained as a diagnostic stress benchmark because it contains synthetic stress-extension rules. Future formal architecture experiments should read from `datasets/architecture`.

## Model-Comparison Scope

Formal aviation KG-to-rule model comparisons should use Qwen, DeepSeek strict repaired, and Xiaomi MIMO outputs. Section 6.4 uses one fixed aviation core plus per-model evaluation overlays so rule precision/recall are not dominated by rule-ID namespace mismatch. GLM retry runs are retained only as exploratory stability records and are excluded from the main formal comparison tables.
