# Architecture NCARB50 v2 dataset

This dataset contains 50 architecture-code benchmark tasks, numbered ARCH_FKG_51 through ARCH_FKG_100. The tasks are derived from NCARB ARE practice-exam scenario themes, then paraphrased, parameterized, and manually checked. They should be described as an NCARB-practice-theme-derived semantic extension set, not as 50 fully independent real project cases.

## Method-visible inputs

Methods may read only core/algorithm_inputs, core/scenario_models, rule_libraries, and public_tasks. The tasks directory is kept only as a compatibility public-task mirror and contains algorithm_input only. Original question text, answer choices, exact practice-question identifiers, source file names, and internal expanded_from_ARCH_FKG lineage are not included in algorithm-visible files.

## Hidden evaluator and audit artifacts

evaluation_references, core/source_semantic_references, evaluation_overlays/*/evaluation_references.json, paired_audit_tasks, and constraint_templates are hidden evaluator or offline-audit artifacts. They must not be passed to a method as input. paired_audit_tasks keeps algorithm_input paired with evaluation_reference only for manual audit and offline validation. constraint_templates is also hidden because it is an evaluator template artifact rather than a method-visible benchmark input.

## Scenario variants

Some tasks intentionally reuse related rule structures. They are retained as parameterized scenario variants with substantive differences in scenario facts, variable bounds, objective weights, applicability conditions, or document-review context. These groups are listed in MANIFEST.json and related dataset manifests, and should not be presented as fully independent engineering project cases.

## Semantic and nonlinear profiles

Semantic-indicator tasks are ARCH_FKG_67, ARCH_FKG_68, ARCH_FKG_69, ARCH_FKG_70, ARCH_FKG_74, ARCH_FKG_75, ARCH_FKG_89, ARCH_FKG_90, ARCH_FKG_91, and ARCH_FKG_93. They test rule applicability, precedence, categorical, or system-capability recovery, rather than pure continuous numeric optimization. ARCH_FKG_60 is a mixed numeric and semantic-indicator task.

Nonlinear or auxiliary-constraint tasks are ARCH_FKG_51, ARCH_FKG_52, ARCH_FKG_55, ARCH_FKG_56, ARCH_FKG_58, ARCH_FKG_61, ARCH_FKG_63, ARCH_FKG_64, ARCH_FKG_72, ARCH_FKG_73, ARCH_FKG_95, ARCH_FKG_96, ARCH_FKG_98, and ARCH_FKG_99. These tasks use max(), abs(), variable products, symbolic division, or division by a decision variable, and require a nonlinear/auxiliary-function checker or an explicit linearized surrogate before pure linear optimization.
