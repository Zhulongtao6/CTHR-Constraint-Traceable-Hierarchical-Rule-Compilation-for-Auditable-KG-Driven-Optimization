# CTHR

[Zhulongtao6/CTHR-Constraint-Traceable-Hierarchical-Rule-Compilation-for-Auditable-KG-Driven-Optimization](https://github.com/Zhulongtao6/CTHR-Constraint-Traceable-Hierarchical-Rule-Compilation-for-Auditable-KG-Driven-Optimization)

This repository contains the experiment code, benchmark data, prompts, result artifacts, and paper source for:

**CTHR: Constraint-Traceable Hierarchical Rule Compilation for Auditable KG-Driven Optimization**.

CTHR is a provenance-preserving KG-to-rule-to-feasible-cell modeling layer. It compiles source-linked rules into auditable feasible cells that can be consumed by downstream solver backends.

## Figures

![CTHR overview](paper_editing/figures/figs/fig.%201.%20overview.jpg)

![Flat modeling and CTHR valid-structure modeling example](paper_editing/figures/figs/fig.%202.%20an%20example.jpg)

![Example benchmark task inputs](paper_editing/figures/figs/fig.%203.%20an%20example%20of%20dataset%20.jpg)

## Repository Layout

- `paper_editing/`: LaTeX source, figures, references, compiled paper draft, and supplemental experiment notes.
- `prompts/`: KG-to-rule extraction prompts used to build rule libraries.
- `experiments/kg_to_rule_validation/`: earlier KG-to-rule validation utilities and symbolic baselines.
- `submission_support/kbs_systematic_experiments_v1/scripts/`: experiment runners for the KBS submission experiments.
- `submission_support/kbs_systematic_experiments_v1/datasets/aviation_rule_relation_balanced_v13_150/`: aviation benchmark tasks.
- `submission_support/kbs_systematic_experiments_v1/datasets/architecture_fullkg_ncarb50_v2/`: architecture benchmark tasks.
- `submission_support/kbs_systematic_experiments_v1/results/`: paper-facing result tables and reports for the main experiments.

## Data Note

The benchmark data included here are derived task and rule-library artifacts for reproducibility. The original source documents are not redistributed in this repository. The aviation corpus is based on the Civil Aviation Administration of China flight procedure design specification AC-97-FS-005R1. The architecture corpus is based on the 2010 ADA Standards for Accessible Design, the 2021 International Building Code, and the 2021 International Fire Code.

## Environment

Install the Python dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Some solver backends require optional system-level installations, such as SCIP, HiGHS, OR-Tools, or clingo. The experiment scripts are organized so that a backend can be replaced while keeping the CTHR feasible-cell inputs fixed.

## Reproduction

The main experiment scripts are under:

```text
submission_support/kbs_systematic_experiments_v1/scripts/
```

The corresponding result artifacts are under:

```text
submission_support/kbs_systematic_experiments_v1/results/
```

Run scripts from the repository root so that relative paths resolve consistently.

## Prompts

The KG-to-rule extraction prompts are provided under `prompts/`. They are included to make the rule-library construction process inspectable and reproducible.
